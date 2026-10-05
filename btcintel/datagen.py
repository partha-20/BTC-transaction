"""Synthetic Bitcoin P2P + blockchain metadata generator (offline, deterministic per seed).

Simulates a small Bitcoin economy with a real UTXO ledger, so amounts always balance:

  licit   : retail users (incl. privacy-minded CoinJoin users), merchants (payroll batches,
            consolidation), exchanges (batched withdrawals, deposit sweeps, promo airdrops)
  illicit : ransomware operators (victim payments -> consolidation -> peeling chain -> CoinJoin
            -> exchange cash-out), darknet markets (deposits -> sweeps -> vendor payout batches ->
            vendor peel/mix), scam "layering" rings (fan-out -> hop chains -> fan-in -> cash-out)
  mixer   : Wasabi/Whirlpool-style CoinJoin coordinator with fixed denominations

Every transaction is then observed by 3 passive network sensors (1-5 sightings, relay noise),
producing rows with: timestamp, src/dst ip+port, txid, input/output addresses+amounts, fee, script type.
The output is split over CSV / JSON / XML files (as the challenge requires) and a few % of rows
are deliberately corrupted so the ingestion/validation stage has real work to do.
Ground truth (ownership, roles, patterns, chains) is written separately and is *only* used
for training on the training dataset and for evaluation - never by the detection pipeline.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .common import DUST, SAT, script_of
from .geoip import COUNTRIES, TZ, SyntheticGeo

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BECH = "023456789acdefghjklmnpqrstuvwxyz"
SCRIPTS = ["p2pkh", "p2sh", "p2wpkh", "p2tr"]
CC_W = [.20, .08, .06, .12, .07, .06, .06, .05, .04, .03, .04, .05, .04, .04, .03, .03, .01, .01]
DENOMS = [0.01, 0.05, 0.1, 0.5]


@dataclass
class GenConfig:
    seed: int = 7
    days: int = 30
    n_users: int = 1000
    n_merchants: int = 25
    n_exchanges: int = 4
    n_ransom: int = 3
    n_markets: int = 2
    vendors_per_market: int = 8
    n_layer: int = 2
    start: str = "2026-08-01"
    dirty: float = 0.003


@dataclass
class Wallet:
    wid: str
    kind: str
    illicit: bool
    country: str
    ips: list                      # [(ip, weight)]
    tz: float
    script_p: tuple
    feerate: float
    fee_jit: float
    fee_follow: bool = True
    utxos: dict = field(default_factory=dict)   # (txid,vout) -> (addr, sats)
    addrs: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class Sim:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.py = random.Random(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.geo = SyntheticGeo(self.py)
        self.t0 = datetime.fromisoformat(cfg.start).replace(tzinfo=timezone.utc).timestamp()
        self.T = cfg.days * 86400.0
        self.wallets: dict[str, Wallet] = {}
        self.owner: dict[str, str] = {}
        self.txs: list[dict] = []
        self.heap: list = []
        self.seq = 0
        self.n = 0
        self.mixq = {d: [] for d in DENOMS}
        self.users, self.merchants, self.exchanges, self.mules = [], [], [], []
        self.chain_ctr = 0
        self.role_addrs: dict[str, set] = {}       # role -> addresses (for seed sampling)

    # ---------------------------------------------------------------- helpers
    def at(self, t, fn, *args):
        if t <= self.T:
            self.seq += 1
            heapq.heappush(self.heap, (t, self.seq, fn, args))

    def lognorm(self, median, sigma):
        return float(self.rng.lognormal(math.log(median), sigma))

    def fee_market(self, t):
        return 1.0 + 0.25 * math.sin(t / 43200.0 + 1.3)

    @staticmethod
    def diurnal(h):
        return 0.15 + 0.85 * max(0.0, math.sin(math.pi * (h - 7) / 14))

    def next_event(self, t, per_day, tz):
        while True:
            t += float(self.rng.exponential(86400.0 / per_day))
            h = ((self.t0 + t) / 3600.0 + tz) % 24
            if self.rng.random() < self.diurnal(h):
                return t

    def mkwallet(self, kind, country, illicit, ip_spec, script_p, feerate, fee_jit, fee_follow=True):
        wid = f"W{len(self.wallets):05d}"
        ips = []
        for cat, wgt, cc in ip_spec:
            ips.append((self.geo.pick_ip(cat, cc), wgt))
        w = Wallet(wid, kind, illicit, country, ips, TZ[country], script_p, feerate, fee_jit, fee_follow)
        self.wallets[wid] = w
        return w

    def new_addr(self, w, script=None, role=None):
        script = script or SCRIPTS[int(self.rng.choice(4, p=w.script_p))]
        while True:
            if script == "p2pkh":
                a = "1" + "".join(self.py.choices(B58, k=33))
            elif script == "p2sh":
                a = "3" + "".join(self.py.choices(B58, k=33))
            elif script == "p2wpkh":
                a = "bc1q" + "".join(self.py.choices(BECH, k=38))
            else:
                a = "bc1p" + "".join(self.py.choices(BECH, k=58))
            if a not in self.owner:
                break
        self.owner[a] = w.wid
        w.addrs.append(a)
        if role:
            self.role_addrs.setdefault(role, set()).add(a)
        return a

    def recv_addr(self, w, reuse=0.1):
        if w.addrs and self.py.random() < reuse:
            return self.py.choice(w.addrs)
        return self.new_addr(w)

    def balance(self, w):
        return sum(v[1] for v in w.utxos.values())

    def genesis(self, w, btc, parts=1, script=None):
        for i in range(parts):
            a = self.new_addr(w, script)
            w.utxos[(f"genesis-{w.wid}-{i}", 0)] = (a, int(btc * SAT / parts))

    # ------------------------------------------------------------ ledger core
    def _commit(self, t, ins, outs, fee, pattern, origin, chain=None):
        self.n += 1
        txid = hashlib.sha256(f"{self.cfg.seed}:{self.n}".encode()).hexdigest()
        rec_in = []
        for wid, uid in ins:
            rec_in.append(self.wallets[wid].utxos.pop(uid))
        for i, (addr, sats) in enumerate(outs):
            self.wallets[self.owner[addr]].utxos[(txid, i)] = (addr, sats)
        rec = dict(t=t, txid=txid, ins=rec_in, outs=list(outs), fee=int(fee), pattern=pattern,
                   origin=origin, chain=chain)
        self.txs.append(rec)
        return rec

    def _fr(self, t, w):
        fr = w.feerate * (1 + self.py.uniform(-w.fee_jit, w.fee_jit))
        return fr * (self.fee_market(t) if w.fee_follow else 1.0)

    def spend(self, t, w, dests, pattern, maxin=10, feerate=None, order="random", inputs=None,
              ufilter=None, reuse_change=0.0, shuffle=True, chain=None, change_script=None):
        need = sum(a for _, a in dests)
        fr = feerate if feerate is not None else self._fr(t, w)
        if inputs is None:
            items = [(u, v) for u, v in w.utxos.items() if ufilter is None or ufilter(u, v)]
            if order == "random":
                self.py.shuffle(items)
            elif order == "desc":
                items.sort(key=lambda kv: -kv[1][1])
            else:
                items.sort(key=lambda kv: kv[1][1])
            chosen, tot = [], 0
            for u, (_, s) in items:
                chosen.append(u)
                tot += s
                fee = int(fr * (10.5 + 68 * len(chosen) + 31 * (len(dests) + 1)))
                if tot >= need + fee or len(chosen) >= maxin:
                    break
        else:
            chosen = list(inputs)
            tot = sum(w.utxos[u][1] for u in chosen)
        if not chosen:
            return None
        fee = int(fr * (10.5 + 68 * len(chosen) + 31 * (len(dests) + 1)))
        if tot < need + fee:
            return None
        change = tot - need - fee
        outs = [(a, s, False) for a, s in dests]
        if change >= DUST:
            ca = self.recv_addr(w, reuse_change) if reuse_change else self.new_addr(w, change_script)
            outs.append((ca, change, True))
        else:
            fee += change
        if shuffle:
            self.py.shuffle(outs)
        cidx = next((i for i, o in enumerate(outs) if o[2]), None)
        rec = self._commit(t, [(w.wid, u) for u in chosen], [(a, s) for a, s, _ in outs], fee, pattern, w.wid, chain)
        rec["change_vout"] = cidx
        return rec

    def consolidate(self, t, w, uids, pattern, dest=None, feerate=None):
        fr = feerate if feerate is not None else self._fr(t, w)
        tot = sum(w.utxos[u][1] for u in uids)
        fee = int(fr * (10.5 + 68 * len(uids) + 31))
        if tot - fee < DUST:
            return None
        dest = dest or self.new_addr(w)
        return self._commit(t, [(w.wid, u) for u in uids], [(dest, tot - fee)], fee, pattern, w.wid)

    def send_all(self, t, w, uid, dest, pattern, feerate=None):
        return self.consolidate(t, w, [uid], pattern, dest, feerate)

    # ------------------------------------------------------------- exchanges
    def deposit_addr(self, ex, w):
        d = ex.meta["deposit"]
        if w.wid not in d:
            d[w.wid] = self.new_addr(ex, "p2sh", role="exchange_deposit")
            ex.meta["deposit_addrs"].add(d[w.wid])
        return d[w.wid]

    def exch_request(self, t, w, btc=None):
        ex = self.py.choice(self.exchanges)
        ex.meta["queue"].append((self.recv_addr(w, 0.05), int((btc or self.lognorm(0.15, 0.8)) * SAT)))

    def exchange_withdraw(self, t, ex):
        self.at(t + self.py.uniform(1200, 2400), self.exchange_withdraw, ex)
        q = ex.meta["queue"]
        hot = ex.meta["hot"]
        flt = lambda u, v: v[0] in hot
        if self.py.random() < 0.004:                                    # promo airdrop (hard negative)
            amt = int(self.py.choice([0.001, 0.002, 0.005]) * SAT)
            us = self.py.sample(self.users, 25)
            self.spend(t, ex, [(self.recv_addr(u, 0.0), amt) for u in us], "exchange_promo", maxin=2, order="desc", ufilter=flt)
            return
        if not q:
            return
        batch, ex.meta["queue"] = q[:35], q[35:]
        self.spend(t, ex, batch, "exchange_batch", maxin=3, order="desc", ufilter=flt)

    def exchange_sweep(self, t, ex):
        self.at(t + self.py.uniform(6, 10) * 3600, self.exchange_sweep, ex)
        dep = ex.meta["deposit_addrs"]
        uids = [u for u, v in ex.utxos.items() if v[0] in dep]
        for i in range(0, len(uids), 60):
            chunk = uids[i:i + 60]
            if len(chunk) >= 3:
                self.consolidate(t + i, ex, chunk, "exchange_sweep", dest=self.py.choice(ex.meta["hot"]))

    # ---------------------------------------------------------------- users
    def user_act(self, t, w):
        self.at(self.next_event(t, w.meta["rate"], w.tz), self.user_act, w)
        bal = self.balance(w)
        if bal < 0.003 * SAT:
            self.exch_request(t, w)
            return
        amt = int(min(self.lognorm(0.012, 1.1) * SAT, bal * 0.5))
        amt = max(amt, 20000)
        if self.py.random() < 0.15:                      # organic cash-in from an exchange
            self.exch_request(t, w, self.lognorm(0.05, 0.9))
            return
        r = self.py.random()
        if r < 0.55:
            dest = self.recv_addr(self.py.choice(self.users), 0.08)
        elif r < 0.72:
            dest = self.recv_addr(self.py.choice(self.merchants), 0.6)
        else:
            dest = self.deposit_addr(self.py.choice(self.exchanges), w)
        rec = self.spend(t, w, [(dest, amt)], "payment", maxin=8, reuse_change=0.05 if self.py.random() < 0.15 else 0.0)
        if rec is None:
            self.exch_request(t, w)

    def user_mix(self, t, w):
        self.at(self.next_event(t, 0.7, w.tz), self.user_mix, w)
        bal = self.balance(w) / SAT
        opts = [d for d in DENOMS if d * 1.5 < bal]
        if opts:
            self.mixq[self.py.choice(opts)].append(dict(wid=w.wid, cb=None))

    def merchant_daily(self, t, m):
        self.at(t + 86400 + self.py.uniform(-3600, 3600), self.merchant_daily, m)
        if len(m.utxos) >= 8:
            self.consolidate(t, m, list(m.utxos)[:50], "consolidation")
        if m.meta.get("payroll") and self.py.random() < 0.3 and self.balance(m) > 0.5 * SAT:
            k = self.py.randint(6, 12)
            amt = int(self.py.choice([0.01, 0.02, 0.03]) * SAT)
            us = self.py.sample(self.users, k)
            self.spend(t, m, [(self.recv_addr(u, 0.0), amt) for u in us], "payroll", maxin=3, order="desc")

    # ---------------------------------------------------------------- mixing
    def mix_round(self, t, D):
        self.at(t + self.py.uniform(2.5, 4.5) * 3600, self.mix_round, D)
        q, seen, parts = self.mixq[D], set(), []
        for req in q:
            if req["wid"] not in seen:
                seen.add(req["wid"])
                parts.append(req)
            if len(parts) >= 14:
                break
        if len(parts) < 4:
            return
        coord = self.coord
        dsat = int(D * SAT)
        fr = coord.feerate * self.fee_market(t)
        k = len(parts)
        share = int(fr * (10.5 + 68 * k + 31 * 2 * k)) // k
        ins, outs, meta = [], [], []
        for req in parts:
            w = self.wallets[req["wid"]]
            cands = sorted([(v[1], u) for u, v in w.utxos.items() if v[1] >= dsat + share + DUST])
            if not cands:
                continue
            sats, u = cands[0] if w.kind == "user" else cands[-1]
            ins.append((w.wid, u))
            maddr = self.new_addr(w, "p2wpkh")
            outs.append((maddr, dsat))
            meta.append((req, w, maddr))
            ch = sats - dsat - share
            if ch >= DUST:
                outs.append((self.new_addr(w), ch))
        for req in parts:                                   # consume the requests we looked at
            q.remove(req)
        if len(meta) < 4:
            return
        self.py.shuffle(outs)
        rec = self._commit(t, ins, outs, share * len(ins), "coinjoin", coord.wid)
        index = {a: i for i, (a, _) in enumerate(rec["outs"])}
        for req, w, maddr in meta:
            if req["cb"]:
                self.at(t + self.py.uniform(6, 60) * 3600, req["cb"], w, (rec["txid"], index[maddr]))

    # ---------------------------------------------------------- peeling chain
    def peel_step(self, t, w, uid, k, spec):
        if k >= spec["steps"] or uid not in w.utxos:
            if spec.get("on_end") and uid in w.utxos:
                spec["on_end"](t, w, uid)
            return
        bal = w.utxos[uid][1]
        peel = int(bal * self.py.uniform(*spec["frac"]))
        if self.py.random() < 0.3:
            peel = peel // 10000 * 10000
        peel = max(peel, 5000)
        if self.py.random() < spec["cash_p"]:
            dest = self.deposit_addr(self.py.choice(self.exchanges), w)
        else:
            mule = self.py.choice(spec["mules"])
            dest = self.new_addr(mule, role="mule")
        rec = self.spend(t, w, [(dest, peel)], "peel", inputs=[uid], feerate=spec["fr"] * self.py.uniform(0.98, 1.02),
                         chain=spec["id"], change_script=spec.get("script"))
        if rec is None or rec["change_vout"] is None:
            return
        gap = self.py.uniform(*spec["gap"]) * 3600
        self.at(t + gap, self.peel_step, w, (rec["txid"], rec["change_vout"]), k + 1, spec)

    def start_peel(self, t, w, uid, steps, frac, cash_p, gap, on_end=None):
        self.chain_ctr += 1
        mules = [self.new_mule() for _ in range(self.py.randint(2, 4))]
        spec = dict(steps=steps, frac=frac, cash_p=cash_p, gap=gap, fr=w.feerate, mules=mules, on_end=on_end,
                    id=f"C{self.chain_ctr:04d}", script=self.py.choice(["p2wpkh", "p2pkh"]))
        self.at(t, self.peel_step, w, uid, 0, spec)

    def new_mule(self):
        cc = self.py.choice(COUNTRIES)
        m = self.mkwallet("mule", cc, True, [("isp", .6, cc), ("vpn", .4, None)], (.3, .1, .5, .1), 12, .3)
        self.mules.append(m)
        return m

    def launder_mix(self, t, w, uid):
        bal = self.balance(w) / SAT
        D = 0.5 if bal > 5 else (0.1 if bal > 1 else 0.05)
        ex = self.py.choice(self.exchanges)

        def cashout(t2, w2, muid):
            if muid in w2.utxos:
                self.send_all(t2 + 60, w2, muid, self.deposit_addr(ex, w2), "cashout_deposit")

        for _ in range(self.py.randint(5, 10)):
            self.mixq[D].append(dict(wid=w.wid, cb=cashout))

    # ---------------------------------------------------------------- ransomware
    def setup_ransom(self, i):
        cc = self.py.choice(["RU", "UA", "CN", "BR"])
        op = self.mkwallet("ransom_op", cc, True, [("tor", .3, None), ("vpn", .3, None), ("hosting", .3, None), ("isp", .1, cc)],
                           (.7, 0, .3, 0), 9, .03, fee_follow=False)
        start = self.rng.uniform(0, self.T * 0.3)
        levels = self.py.sample([0.03, 0.05, 0.08, 0.15, 0.25], 2)
        for _ in range(self.py.randint(70, 110)):
            self.at(start + self.rng.uniform(0, 9 * 86400), self.ransom_pay, op, self.py.choice(levels))
        for d in range(2, 12, 2):
            self.at(start + d * 86400 + self.py.uniform(0, 8000), self.ransom_consolidate, op)
        self.at(start + 10.6 * 86400, self.ransom_launder, op)

    def ransom_pay(self, t, op, level):
        for _ in range(15):
            v = self.py.choice(self.users)
            amt = int(level * SAT * (1 + self.py.uniform(-.02, .02)))
            dest = self.new_addr(op, role="ransom_deposit")
            if self.spend(t, v, [(dest, amt)], "ransom_payment", maxin=6):
                return
            self.owner.pop(dest); op.addrs.remove(dest); self.role_addrs["ransom_deposit"].discard(dest)

    def ransom_consolidate(self, t, op):
        uids = list(op.utxos)
        for i in range(0, len(uids), 60):
            if len(uids[i:i + 60]) >= 5:
                self.consolidate(t + i * 30, op, uids[i:i + 60], "ransom_consolidation", feerate=op.feerate)

    def ransom_launder(self, t, op):
        uids = list(op.utxos)
        for i in range(0, len(uids), 60):
            if len(uids[i:i + 60]) >= 2:
                self.consolidate(t + i * 30, op, uids[i:i + 60], "ransom_consolidation", feerate=op.feerate)
        self.at(t + 3600 * 3, self._ransom_chain, op)

    def _ransom_chain(self, t, op):
        if not op.utxos:
            return
        uid = max(op.utxos, key=lambda u: op.utxos[u][1])
        self.start_peel(t, op, uid, self.py.randint(14, 32), (0.008, 0.05), 0.45, (0.3, 4.0), on_end=self.launder_mix)

    # ---------------------------------------------------------------- darknet
    def setup_market(self, i):
        m = self.mkwallet("darknet_market", "NL", True, [("tor", .5, None), ("hosting", .5, None)], (.2, 0, .7, .1), 15, .1)
        m.meta["dep"] = set()
        m.meta["hot"] = [self.new_addr(m, "p2wpkh", role="market_hot") for _ in range(3)]
        vendors = []
        for _ in range(self.cfg.vendors_per_market):
            cc = self.py.choice(COUNTRIES)
            vendors.append(self.mkwallet("vendor", cc, True, [("vpn", .4, None), ("tor", .3, None), ("isp", .3, cc)],
                                         (.3, .1, .5, .1), 6, .05, fee_follow=False))
        t = self.T * 0.03
        while t < self.T:
            t += float(self.rng.exponential(86400 / 8.0))
            self.at(t, self.market_buy, m)
        for d in range(1, self.cfg.days):
            self.at(d * 86400 + self.py.uniform(0, 4000), self.market_sweep, m)
        d = 3.0
        while d < self.cfg.days:
            self.at(d * 86400, self.market_payout, m, vendors)
            d += self.py.uniform(3, 4.5)

    def market_buy(self, t, m):
        for _ in range(10):
            b = self.py.choice(self.users)
            a = self.new_addr(m, role="market_deposit")
            if self.spend(t, b, [(a, int(self.lognorm(0.015, 0.8) * SAT) + 30000)], "darknet_deposit", maxin=6):
                m.meta["dep"].add(a)
                return
            self.owner.pop(a); m.addrs.remove(a); self.role_addrs["market_deposit"].discard(a)

    def market_sweep(self, t, m):
        uids = [u for u, v in m.utxos.items() if v[0] in m.meta["dep"]]
        for i in range(0, len(uids), 80):
            if len(uids[i:i + 80]) >= 4:
                self.consolidate(t + i * 20, m, uids[i:i + 80], "darknet_sweep", dest=self.py.choice(m.meta["hot"]))

    def market_payout(self, t, m, vendors):
        hot = set(m.meta["hot"])
        avail = sum(v[1] for v in m.utxos.values() if v[0] in hot)
        if avail < 0.05 * SAT:
            return
        vs = self.py.sample(vendors, self.py.randint(4, min(8, len(vendors))))
        shares = self.rng.dirichlet(np.ones(len(vs))) * avail * 0.6
        dests = [(self.recv_addr(v, 0.0), int(s)) for v, s in zip(vs, shares)]
        rec = self.spend(t, m, dests, "darknet_payout", maxin=4, order="desc", ufilter=lambda u, v: v[0] in hot)
        if rec:
            for v in vs:
                self.at(t + self.py.uniform(1, 20) * 3600, self.vendor_cashout, v)

    def vendor_cashout(self, t, v):
        if not v.utxos:
            return
        uid = max(v.utxos, key=lambda u: v.utxos[u][1])
        if self.py.random() < 0.7:
            self.start_peel(t, v, uid, self.py.randint(4, 10), (0.05, 0.25), 0.7, (3, 12))
        elif v.utxos[uid][1] > 0.06 * SAT:
            D = 0.05 if v.utxos[uid][1] < 0.6 * SAT else 0.1
            ex = self.py.choice(self.exchanges)
            cb = lambda t2, w2, muid: self.send_all(t2 + 60, w2, muid, self.deposit_addr(ex, w2), "cashout_deposit") if muid in w2.utxos else None
            for _ in range(self.py.randint(2, 4)):
                self.mixq[D].append(dict(wid=v.wid, cb=cb))

    # ---------------------------------------------------------------- layering ring
    def setup_layer(self, i):
        cc = self.py.choice(["NG", "VN", "ID", "RO", "TR"])
        w = self.mkwallet("layerer", cc, True, [("vpn", .5, None), ("hosting", .3, None), ("isp", .2, cc)],
                          (.1, .1, .8, 0), 11, .04, fee_follow=False)
        start = self.rng.uniform(0, self.T * 0.4)
        for _ in range(self.py.randint(20, 35)):
            self.at(start + self.rng.uniform(0, 8 * 86400), self.layer_victim, w)
        self.at(start + 9 * 86400, self.layer_run, w)

    def layer_victim(self, t, w):
        for _ in range(10):
            v = self.py.choice(self.users)
            a = self.new_addr(w, "p2wpkh", role="layer_deposit")
            if self.spend(t, v, [(a, int(self.lognorm(0.2, 0.9) * SAT))], "scam_payment", maxin=6):
                return
            self.owner.pop(a); w.addrs.remove(a); self.role_addrs["layer_deposit"].discard(a)

    def layer_run(self, t, w):
        uids = list(w.utxos)
        rec = None
        for i in range(0, len(uids), 60):
            rec = self.consolidate(t + i * 30, w, uids[i:i + 60], "layer_consolidation", feerate=w.feerate)
        self.at(t + 1800, self.layer_fanout, w)

    def layer_fanout(self, t, w):
        if not w.utxos:
            return
        uid = max(w.utxos, key=lambda u: w.utxos[u][1])
        bal = w.utxos[uid][1]
        k = self.py.randint(8, 16)
        each = int(bal * 0.98 / k)
        dests = [(self.new_addr(w, "p2wpkh"), int(each * self.py.uniform(0.996, 1.004))) for _ in range(k)]
        rec = self.spend(t, w, dests, "fanout", inputs=[uid], feerate=w.feerate)
        if not rec:
            return
        state = dict(pending=k)
        for i, (a, s) in enumerate(rec["outs"]):
            if self.owner[a] == w.wid and s >= each * 0.9:
                self.at(t + self.py.uniform(600, 5400), self.layer_hop, w, (rec["txid"], i), 2, state)

    def layer_hop(self, t, w, uid, hops, state):
        if uid not in w.utxos:
            return
        if hops == 0:
            state["pending"] -= 1
            if state["pending"] <= 0:
                self.at(t + 1200, self.layer_fanin, w)
            return
        rec = self.send_all(t, w, uid, self.new_addr(w, "p2wpkh"), "hop", feerate=w.feerate)
        if rec:
            self.at(t + self.py.uniform(600, 5400), self.layer_hop, w, (rec["txid"], 0), hops - 1, state)

    def layer_fanin(self, t, w):
        uids = list(w.utxos)[:60]
        rec = self.consolidate(t, w, uids, "fanin", feerate=w.feerate)
        if rec:
            ex = self.py.choice(self.exchanges)
            self.at(t + self.py.uniform(2, 6) * 3600, lambda t2, ww, u: self.send_all(t2, ww, u, self.deposit_addr(ex, ww), "cashout_deposit")
                    if u in ww.utxos else None, w, (rec["txid"], 0))

    # ---------------------------------------------------------------- setup + run
    def setup(self):
        g = self.geo
        # mixer coordinator (broadcasts every CoinJoin)
        self.coord = self.mkwallet("mixer_coord", "NL", False, [("hosting", 1, "NL")], (0, 0, 1, 0), 18, .05)
        # exchanges
        for i in range(self.cfg.n_exchanges):
            cc = self.py.choice(["NL", "SG", "US", "DE"])
            ex = self.mkwallet("exchange", cc, False, [("hosting", .7, cc), ("hosting", .3, cc)], (.05, .5, .45, 0), 22, .15)
            ex.meta.update(deposit={}, deposit_addrs=set(), queue=[])
            ex.meta["hot"] = [self.new_addr(ex, "p2sh", role="exchange_hot") for _ in range(4)]
            for a in ex.meta["hot"]:
                ex.utxos[(f"genesis-{ex.wid}-{a[:6]}", 0)] = (a, int(3000 * SAT))
            self.exchanges.append(ex)
            self.at(self.py.uniform(0, 1800), self.exchange_withdraw, ex)
            self.at(self.py.uniform(0, 8 * 3600), self.exchange_sweep, ex)
        # merchants
        for i in range(self.cfg.n_merchants):
            cc = self.py.choice(COUNTRIES)
            m = self.mkwallet("merchant", cc, False, [("hosting", .6, cc), ("isp", .4, cc)], (.2, .3, .5, 0), 14, .3)
            m.meta["payroll"] = i % 4 == 0
            self.genesis(m, 20)
            self.merchants.append(m)
            self.at(self.py.uniform(0, 86400), self.merchant_daily, m)
        # users
        by_cc = {}
        for i in range(self.cfg.n_users):
            cc = COUNTRIES[int(self.rng.choice(len(COUNTRIES), p=np.array(CC_W) / sum(CC_W)))]
            spec = [("isp", 1.0, cc)]
            if self.py.random() < 0.10 and by_cc.get(cc):
                spec = [(None, 1.0, None)]
            if self.py.random() < 0.15:
                spec.append(("vpn", .3, None))
            sp = self.py.choice([(.25, .15, .5, .1), (.05, .05, .8, .1), (.6, .2, .2, 0), (0, .1, .6, .3)])
            if spec[0][0] is None:
                w = self.mkwallet("user", cc, False, [("isp", 1.0, cc)] + spec[1:], sp, float(self.lognorm(14, .4)), .35)
                w.ips[0] = (self.py.choice(by_cc[cc]), 1.0)
            else:
                w = self.mkwallet("user", cc, False, spec, sp, float(self.lognorm(14, .4)), .35)
            by_cc.setdefault(cc, []).append(w.ips[0][0])
            w.meta["rate"] = float(min(4.0, self.lognorm(0.9, 0.7)))
            self.genesis(w, self.lognorm(0.4, 1.0), parts=self.py.randint(1, 4))
            self.users.append(w)
        for w in self.users:
            self.at(self.next_event(0, w.meta["rate"], w.tz), self.user_act, w)
            if self.py.random() < 0.09:
                self.at(self.py.uniform(0, 3 * 86400), self.user_mix, w)
        for D in DENOMS:
            self.at(self.py.uniform(0, 3 * 3600), self.mix_round, D)
        for i in range(self.cfg.n_ransom):
            self.setup_ransom(i)
        for i in range(self.cfg.n_markets):
            self.setup_market(i)
        for i in range(self.cfg.n_layer):
            self.setup_layer(i)

    def run(self):
        self.setup()
        while self.heap:
            t, _, fn, args = heapq.heappop(self.heap)
            if t > self.T:
                break
            fn(t, *args)
        self.txs.sort(key=lambda r: r["t"])
        return self

    # ---------------------------------------------------------------- observations
    def build_observations(self):
        rng, py = self.rng, self.py
        relay = [self.geo.pick_ip(py.choice(["isp", "isp", "hosting"])) for _ in range(1500)]
        sensors = ["198.51.100.10", "198.51.100.11", "198.51.100.12"]
        rows = []
        for tx in self.txs:
            w = self.wallets[tx["origin"]]
            ips, wts = zip(*w.ips)
            oip = ips[int(rng.choice(len(ips), p=np.array(wts) / sum(wts)))]
            n_obs = min(5, 1 + int(rng.poisson(1.6)))
            sc = script_of(tx["ins"][0][0])
            for j in range(n_obs):
                if j == 0 and rng.random() < 0.55:
                    src, dt = oip, abs(float(rng.normal(0.08, 0.03)))
                else:
                    src, dt = relay[int(rng.integers(len(relay)))], 0.05 + float(rng.exponential(0.4)) + 0.05 * j
                sp = 8333 if rng.random() < 0.7 else int(rng.integers(32768, 61000))
                dp = int(rng.integers(32768, 61000)) if sp == 8333 else 8333
                rows.append(dict(
                    timestamp=self.t0 + tx["t"] + dt, src_ip=src, dst_ip=sensors[int(rng.integers(3))],
                    src_port=sp, dst_port=dp, txid=tx["txid"],
                    input_addresses=[a for a, _ in tx["ins"]], output_addresses=[a for a, _ in tx["outs"]],
                    input_amounts=[s / SAT for _, s in tx["ins"]], output_amounts=[s / SAT for _, s in tx["outs"]],
                    fee=tx["fee"] / SAT, script_type=sc))
        rows.sort(key=lambda r: r["timestamp"])
        return rows


# ------------------------------------------------------------------ writers
def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _corrupt(rows, frac, rng):
    """Inject realistic dirt; returns rows and a manifest of what was injected."""
    kinds = ["no_txid", "bad_ip", "neg_amount", "len_mismatch", "dup", "bad_ts", "no_inputs"]
    man = {k: 0 for k in kinds}
    out = []
    for r in rows:
        out.append(r)
        if rng.random() < frac:
            k = kinds[int(rng.integers(len(kinds)))]
            man[k] += 1
            if k == "dup":
                out.append(dict(r))
                man[k] += 0
                continue
            bad = dict(r)
            out.pop()
            if k == "no_txid": bad["txid"] = ""
            elif k == "bad_ip": bad["src_ip"] = "999.12.1.7"
            elif k == "neg_amount": bad["output_amounts"] = [-abs(x) for x in bad["output_amounts"]]
            elif k == "len_mismatch": bad["output_amounts"] = bad["output_amounts"][:-1] or [0.0]
            elif k == "bad_ts": bad["timestamp"] = "not-a-date"
            elif k == "no_inputs": bad["input_addresses"], bad["input_amounts"] = [], []
            out.append(bad)
    return out, man


def _ts_out(v):
    return _iso(v) if isinstance(v, (int, float)) else v


def write_dataset(sim: Sim, out: Path, write_geo=True):
    out = Path(out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "geoip").mkdir(exist_ok=True)
    (out / "ground_truth").mkdir(exist_ok=True)
    rows = sim.build_observations()
    rows, man = _corrupt(rows, sim.cfg.dirty, sim.rng)
    n = len(rows)
    a, b = int(n * 0.60), int(n * 0.85)
    L = lambda xs: ";".join(f"{x:.8f}" if isinstance(x, float) else str(x) for x in xs)
    import csv
    with open(out / "raw" / "observations_part1.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid", "input_addresses",
                     "output_addresses", "input_amounts", "output_amounts", "fee", "script_type"])
        for r in rows[:a]:
            cw.writerow([_ts_out(r["timestamp"]), r["src_ip"], r["dst_ip"], r["src_port"], r["dst_port"], r["txid"],
                         ";".join(r["input_addresses"]), ";".join(r["output_addresses"]),
                         L(r["input_amounts"]), L(r["output_amounts"]), f"{r['fee']:.8f}", r["script_type"]])
    with open(out / "raw" / "observations_part2.jsonl", "w") as f:
        for r in rows[a:b]:
            f.write(json.dumps({**r, "timestamp": _ts_out(r["timestamp"])}) + "\n")
    with open(out / "raw" / "observations_part3.xml", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<observations>\n')
        for r in rows[b:]:
            f.write("<observation>")
            for k in ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid", "fee", "script_type"]:
                f.write(f"<{k}>{escape(str(_ts_out(r[k]) if k == 'timestamp' else r[k]))}</{k}>")
            for k in ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]:
                f.write(f"<{k}>" + "".join(f"<item>{escape(f'{x:.8f}' if isinstance(x, float) else str(x))}</item>" for x in r[k]) + f"</{k}>")
            f.write("</observation>\n")
        f.write("</observations>\n")
    if write_geo:
        sim.geo.write(out / "geoip")

    # ---- ground truth (evaluation / supervised training only) ----
    wal = {w.wid: dict(kind=w.kind, illicit=w.illicit, country=w.country) for w in sim.wallets.values()}
    seeds = []
    take = lambda role, frac, src: [(a, role, src) for a in sorted(sim.role_addrs.get(role, ())) if sim.rng.random() < frac]
    seeds += take("ransom_deposit", 0.22, "victim_report")
    seeds += take("market_deposit", 0.12, "market_takedown_report")
    seeds += take("layer_deposit", 0.25, "victim_report")
    seeds += take("mule", 0.10, "leo_list")
    with open(out / "raw" / "seeds.csv", "w") as f:
        f.write("address,label,source\n")
        for a, role, src in seeds:
            f.write(f"{a},{role},{src}\n")
    gt = dict(
        addr_owner=sim.owner, wallets=wal,
        tx_pattern={t["txid"]: t["pattern"] for t in sim.txs},
        tx_chain={t["txid"]: t["chain"] for t in sim.txs if t["chain"]},
        tx_origin={t["txid"]: t["origin"] for t in sim.txs},
        dirty_manifest=man)
    json.dump(gt, open(out / "ground_truth" / "ground_truth.json", "w"))
    summary = dict(n_tx=len(sim.txs), n_rows=n, n_wallets=len(wal), n_addresses=len(sim.owner), n_seeds=len(seeds),
                   patterns={p: sum(1 for t in sim.txs if t["pattern"] == p) for p in sorted({t["pattern"] for t in sim.txs})},
                   dirty=man, config=sim.cfg.__dict__)
    json.dump(summary, open(out / "manifest.json", "w"), indent=2)
    return summary


def generate(out_dir, **kw):
    cfg = GenConfig(**kw)
    sim = Sim(cfg).run()
    return write_dataset(sim, Path(out_dir))
