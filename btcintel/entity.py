"""Entity clustering = CIOH  +  ML same-owner link model  (blockchain heuristics fused with IP-timing evidence).

Candidate pairs
  * change-candidates : for each 2-output tx, (sender cluster, output cluster)
  * ip-candidates     : clusters that repeatedly share the same first-seen relay IP
Pair features mix
  * blockchain  : output freshness, script-type match, round-amount tells, dwell time, ...
  * network     : IP concentration, count of shared origin IPs, "spender used the same IP" flag, anon class
  * graph       : spectral-embedding cosine, direct flow
  * behavioural : activity-hour histogram similarity, fee-rate habit similarity
A gradient-boosting model scores each pair; high-confidence links are merged (union-find, size-capped).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.stats import poisson

from .common import script_of
from .graphs import Ledger, cluster_edges, ip_evidence, merge_labels, spectral_embedding

LINK_FEATURES = ["is_change", "is_ip", "ip_shared_cnt", "ip_conc", "ip_shared_n", "ip_anon", "ip_log_n", "ip_ncl", "ip_share_max", "ip_sig", "ip_cnt_max", "loyal_min", "loyal_max", "emb_cos", "log_na", "log_nb",
                 "log_txa", "log_txb", "hour_sim", "fee_diff", "anon_diff", "flow_ntx",
                 "b_fresh", "script_match", "out_frac", "other_round", "b_round", "b_gt_other", "b_spent",
                 "b_dwell_h", "spend_ip_match", "n_in"]


def cluster_profiles(L: Ledger, cl: np.ndarray):
    """Per-cluster sender profile: tx count, address count, mean fee-rate, mean anon score, 24h histogram, IP counter."""
    tcl, pair, tot = ip_evidence(L, cl)
    tx = L.tx
    hr = ((tx["first_seen"].values // 3600) % 24).astype(int)
    n = cl.max() + 1
    ntx = np.bincount(tcl[tcl >= 0], minlength=n)
    naddr = np.bincount(cl, minlength=n)
    ok = tcl >= 0
    fee = np.bincount(tcl[ok], weights=np.log1p(tx["fee_rate"].values[ok]), minlength=n) / np.maximum(ntx, 1)
    anon = np.bincount(tcl[ok], weights=tx["first_anon_score"].values[ok], minlength=n) / np.maximum(ntx, 1)
    hist = np.zeros((n, 24))
    np.add.at(hist, (tcl[ok], hr[ok]), 1)
    hist /= np.maximum(hist.sum(1, keepdims=True), 1)
    ips: dict[int, dict] = defaultdict(dict)
    for ip, c, k in zip(pair["ip"].values, pair["cluster"].values, pair["n"].values):
        ips[c][ip] = k
    return dict(tcl=tcl, ntx=ntx, naddr=naddr, fee=fee, anon=anon, hist=hist, ips=ips, ip_tot=tot,
                ip_anon=tx.groupby("first_src_ip")["first_anon_score"].first())


def _key(a, b):
    return (a, b) if a < b else (b, a)


def build_pairs(L: Ledger, cl: np.ndarray, coinjoin: np.ndarray, emb: np.ndarray, edges: pd.DataFrame, min_hub=6):
    """Return DataFrame of candidate pairs with LINK_FEATURES (+ cl_a, cl_b)."""
    P = cluster_profiles(L, cl)
    tx, ti, to = L.tx, L.txin, L.txout
    fs, fip = tx["first_seen"].values, tx["first_src_ip"].values
    # first appearance of each address (tid)
    first_tid = np.full(L.n_addr, 10 ** 9)
    np.minimum.at(first_tid, ti["aid"].values, ti["tid"].values)
    np.minimum.at(first_tid, to["aid"].values, to["tid"].values)
    flow = {(_key(s, d)): k for s, d, k in zip(edges["src"].values, edges["dst"].values, edges["ntx"].values)}
    feats: dict[tuple, dict] = {}

    # ---- change candidates
    first_in = ti.groupby("tid")["aid"].first()
    in_cl = ti.assign(c=cl[ti["aid"].values]).groupby("tid")["c"].agg(set)
    two = to[to.groupby("tid")["tid"].transform("size") == 2]
    groups = two.groupby("tid").indices
    tid_arr, aid_arr, sat_arr = two["tid"].values, two["aid"].values, two["sat"].values
    sp_arr = two["spent_tid"].values
    for t, rows in groups.items():
        if coinjoin[t] or t not in first_in.index:
            continue
        a_aid = first_in.loc[t]
        A = cl[a_aid]
        sats = sat_arr[rows]
        for j, r in enumerate(rows):
            B = cl[aid_arr[r]]
            if B in in_cl[t]:
                continue
            other = sats[1 - j]
            sp = sp_arr[r]
            f = dict(is_change=1, b_fresh=float(first_tid[aid_arr[r]] == t),
                     script_match=float(script_of(L.addrs[aid_arr[r]]) == script_of(L.addrs[a_aid])),
                     out_frac=sats[j] / max(sats.sum(), 1), other_round=float(other % 10000 == 0),
                     b_round=float(sats[j] % 10000 == 0), b_gt_other=float(sats[j] > other), b_spent=float(sp >= 0),
                     b_dwell_h=(fs[sp] - fs[t]) / 3600 if sp >= 0 else np.nan,
                     spend_ip_match=float(fip[sp] == fip[t]) if sp >= 0 else np.nan, n_in=len(in_cl[t]))
            feats[_key(A, B)] = f

    # ---- ip candidates: "hub" IPs (far more first-relay sightings than relay-noise background)
    ipc = defaultdict(list)
    for c, d in P["ips"].items():
        for ip, k in d.items():
            ipc[ip].append((k, c))
    ip_ncl = {ip: len(v) for ip, v in ipc.items()}
    ipt0 = P["ip_tot"]
    for ip, lst in ipc.items():
        if ipt0[ip] < min_hub or len(lst) < 2:
            continue
        lst.sort(reverse=True)
        cs = [c for _, c in lst]
        for c in cs[1:]:
            feats.setdefault(_key(cs[0], c), {})["is_ip"] = 1
        for a, b in combinations(cs[:5], 2):
            feats.setdefault(_key(a, b), {})["is_ip"] = 1

    # ---- shared pair-level features
    rows = []
    ipt, ipa = P["ip_tot"], P["ip_anon"]
    lam = float(np.median(ipt.values))                      # relay-noise background rate (sightings per IP)
    ip_sig = pd.Series(-poisson.logsf(ipt.values - 1, lam) / np.log(10), index=ipt.index).clip(0, 50)
    for (a, b), f in feats.items():
        da, db = P["ips"].get(a, {}), P["ips"].get(b, {})
        shared = [ip for ip in (da if len(da) < len(db) else db) if ip in da and ip in db]
        cnt = sum(min(da[i], db[i]) for i in shared)
        conc, best, smax, bsig = 0.0, None, 0.0, -1.0
        for i in shared:
            cc = (da[i] + db[i]) / ipt[i]
            smax = max(smax, max(da[i], db[i]) / ipt[i])
            conc = max(conc, cc)
            if ip_sig[i] > bsig:
                bsig, best = ip_sig[i], i
        if best is not None:
            la, lb = da[best] / max(P["ntx"][a], 1), db[best] / max(P["ntx"][b], 1)
            cmax = max(da[best], db[best])
        else:
            la = lb = cmax = 0.0
        ha, hb = P["hist"][a], P["hist"][b]
        hs = float(ha @ hb / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-9))
        r = dict(cl_a=a, cl_b=b, is_change=f.get("is_change", 0), is_ip=f.get("is_ip", 0), ip_shared_cnt=cnt, ip_conc=conc,
                 ip_shared_n=len(shared), ip_anon=float(ipa.get(best, 0.0)) if best else 0.0,
                 ip_sig=float(bsig) if best else 0.0, ip_cnt_max=cmax, loyal_min=min(la, lb), loyal_max=max(la, lb),
                 ip_log_n=float(np.log(ipt[best])) if best else 0.0, ip_ncl=ip_ncl.get(best, 0) if best else 0, ip_share_max=smax,
                 emb_cos=float(emb[a] @ emb[b]), log_na=np.log1p(P["naddr"][a]), log_nb=np.log1p(P["naddr"][b]),
                 log_txa=np.log1p(P["ntx"][a]), log_txb=np.log1p(P["ntx"][b]), hour_sim=hs,
                 fee_diff=abs(P["fee"][a] - P["fee"][b]), anon_diff=abs(P["anon"][a] - P["anon"][b]),
                 flow_ntx=flow.get((a, b), 0))
        for k in ("b_fresh", "script_match", "out_frac", "other_round", "b_round", "b_gt_other", "b_spent", "b_dwell_h",
                  "spend_ip_match", "n_in"):
            r[k] = f.get(k, np.nan)
        rows.append(r)
    return pd.DataFrame(rows), P


def pair_labels(pairs: pd.DataFrame, cl: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Majority ground-truth owner per cluster -> same-owner label per pair (training/eval only)."""
    df = pd.DataFrame({"c": cl, "o": owner})
    maj = df.groupby("c")["o"].agg(lambda s: s.value_counts().index[0])
    oa, ob = maj.reindex(pairs["cl_a"]).values, maj.reindex(pairs["cl_b"]).values
    return ((oa == ob) & (oa != -1)).astype(int)


def resolve(L: Ledger, cl0: np.ndarray, pairs: pd.DataFrame, prob: np.ndarray, thr=0.9, max_size=60000):
    order = np.argsort(-prob)
    links = [(int(pairs["cl_a"].values[i]), int(pairs["cl_b"].values[i])) for i in order if prob[i] >= thr]
    return merge_labels(cl0, links, max_size=max_size), len(links)


def pairwise_prf(pred: np.ndarray, truth: np.ndarray):
    """Pairwise precision / recall / F1 of an address partition vs ground-truth owners (-1 = unknown, ignored)."""
    ok = truth != -1
    df = pd.DataFrame({"p": pred[ok], "t": truth[ok]})
    c2 = lambda x: (x * (x - 1) / 2).sum()
    tp = c2(df.groupby(["p", "t"]).size().values.astype(float))
    pp = c2(df.groupby("p").size().values.astype(float))
    tt = c2(df.groupby("t").size().values.astype(float))
    prec, rec = tp / max(pp, 1), tp / max(tt, 1)
    return dict(precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
                pred_clusters=int(df["p"].nunique()), true_entities=int(df["t"].nunique()))
