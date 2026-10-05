"""Laundering-chain candidates: peeling chains (1-in / 2-out) and hop chains (1-in / 1-out).

A chain step is a non-CoinJoin tx with exactly one input and <=2 outputs.  The "next" step is the tx
that spends the LARGER output (the presumed change).  Maximal paths of length >= 3 become candidates
and are scored by a supervised model (PeelModel) on the chain-level features below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import SAT, script_of
from .graphs import Ledger

CHAIN_FEATURES = ["length", "log_start_btc", "ret_mean", "ret_std", "ret_min", "peel_logstd", "gap_h_med", "gap_h_cv",
                  "gap_h_mean", "fee_cv", "fee_mean", "fresh_change", "script_share", "frac_two_out", "ip_top_share",
                  "anon_mean", "dest_hub_mean", "round_peel", "hour_entropy", "dest_distinct_frac", "night_share"]


def extract_chains(L: Ledger, coinjoin: np.ndarray, min_len: int = 3):
    tx = L.tx
    cand = (tx["n_in"].values == 1) & (tx["n_out"].values <= 2) & (~coinjoin)
    out_by_tid = L.txout.groupby("tid").indices
    sat, spent = L.txout["sat"].values, L.txout["spent_tid"].values
    nxt = {}
    for t in np.flatnonzero(cand):
        rows = out_by_tid[t]
        r = rows[np.argmax(sat[rows])]
        s = spent[r]
        if s >= 0 and cand[s] and s != t:
            nxt[t] = s
    has_pred = set(nxt.values())
    chains = []
    for t in np.flatnonzero(cand):
        if t in has_pred:
            continue
        path, cur = [t], t
        while cur in nxt:
            cur = nxt[cur]
            path.append(cur)
        if len(path) >= min_len:
            chains.append(path)
    return chains


def _entropy(h):
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(24))


def chain_features(L: Ledger, chains, cl_edges: pd.DataFrame, cl: np.ndarray) -> pd.DataFrame:
    tx, to, ti = L.tx, L.txout, L.txin
    fs, fr = tx["first_seen"].values, tx["fee_rate"].values
    insat = tx["in_sat"].values
    ip, anon = tx["first_src_ip"].values, tx["first_anon_score"].values
    out_by_tid = to.groupby("tid").indices
    sat, aid = to["sat"].values, to["aid"].values
    indeg = cl_edges.groupby("dst")["src"].nunique()
    # first appearance of each address
    first_tid = np.full(L.n_addr, 10 ** 9)
    np.minimum.at(first_tid, to["aid"].values, to["tid"].values)
    np.minimum.at(first_tid, ti["aid"].values, ti["tid"].values)
    scr = {}
    rows = []
    for k, path in enumerate(chains):
        ret, peel, fresh, two, dest, sh, rnd = [], [], [], [], [], [], []
        for t in path:
            r = out_by_tid[t]
            big = r[np.argmax(sat[r])]
            ret.append(sat[big] / max(insat[t], 1))
            two.append(len(r) == 2)
            fresh.append(first_tid[aid[big]] == t)
            a = aid[big]
            scr[a] = scr.get(a) or script_of(L.addrs[a])
            sh.append(scr[a])
            if len(r) == 2:
                small = r[np.argmin(sat[r])]
                peel.append(sat[small])
                rnd.append(sat[small] % 10000 == 0)
                dest.append(cl[aid[small]])
        ts = fs[path]
        gaps = np.diff(ts) / 3600.0
        feev = fr[path]
        ips = pd.Series(ip[path])
        hist = np.bincount(((ts // 3600) % 24).astype(int), minlength=24)
        hub = [np.log1p(indeg.get(c, 0)) for c in dest] if dest else [0.0]
        pl = np.log(np.maximum(peel, 1)) if peel else np.array([0.0])
        rows.append(dict(
            chain_id=k, length=len(path), log_start_btc=np.log10(max(insat[path[0]], 1) / SAT),
            ret_mean=np.mean(ret), ret_std=np.std(ret), ret_min=np.min(ret), peel_logstd=float(np.std(pl)),
            gap_h_med=float(np.median(gaps)), gap_h_cv=float(np.std(gaps) / max(np.mean(gaps), 1e-6)), gap_h_mean=float(np.mean(gaps)),
            fee_cv=float(np.std(feev) / max(np.mean(feev), 1e-6)), fee_mean=float(np.mean(feev)),
            fresh_change=float(np.mean(fresh)), script_share=float(pd.Series(sh).value_counts().iloc[0] / len(sh)),
            frac_two_out=float(np.mean(two)), ip_top_share=float(ips.value_counts().iloc[0] / len(ips)),
            anon_mean=float(np.mean(anon[path])), dest_hub_mean=float(np.mean(hub)),
            round_peel=float(np.mean(rnd)) if rnd else 0.0, hour_entropy=_entropy(hist),
            dest_distinct_frac=float(len(set(dest)) / max(len(dest), 1)),
            night_share=float(np.mean(((ts // 3600) % 24 < 6) | ((ts // 3600) % 24 >= 22)))))
    return pd.DataFrame(rows)
