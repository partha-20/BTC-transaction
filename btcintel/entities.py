"""Entity-level feature table + risk propagation from seed (known-illicit) wallets.

Propagation family (all seeded from public/law-enforcement address lists):
  * haircut taint      - value-weighted taint flows UTXO-to-UTXO in time order (dilutes through CoinJoins/commingling)
  * hop distance       - unweighted shortest path to/from a seed entity in the entity flow graph (fwd / bwd)
  * personalised PageRank - fwd (where do seed funds tend to go) and bwd (who funds/feeds seed entities)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from .common import SAT
from .graphs import Ledger, cluster_edges

BEHAVIOUR_FEATURES = [
    "log_addr", "log_sent", "log_recv", "log_out_btc", "log_in_btc", "log_cp_out", "log_cp_in", "pass_ratio",
    "dwell_med_h", "frac_fast", "cj_frac", "cj_btc_frac", "peel_frac", "hop_frac", "fan_out_max", "fan_in_max",
    "equal_fanout_n", "fee_cv", "fee_mean", "anon_mean", "tor_vpn_share", "ip_top_share", "hour_entropy",
    "night_share", "addr_reuse", "p2pkh_share", "out_to_hub_frac", "log_avg_out_btc", "recv_burst"]
PROPAGATION_FEATURES = ["taint_in_frac", "taint_spent_frac", "log_taint_in_btc", "dist_fwd", "dist_bwd", "ppr_fwd", "ppr_bwd"]
ENTITY_FEATURES = BEHAVIOUR_FEATURES + PROPAGATION_FEATURES


# ------------------------------------------------------------------ propagation
def propagate_taint(L: Ledger, seed_addr: np.ndarray):
    """seed_addr: bool[n_addr].  Returns (taint_inherited, taint_final) per txout row and taint per txin row."""
    to, ti = L.txout, L.txin
    n_out = len(to)
    out_start = np.zeros(L.n_tx + 1, dtype=np.int64)
    np.cumsum(np.bincount(to["tid"].values, minlength=L.n_tx), out=out_start[1:])
    in_start = np.zeros(L.n_tx + 1, dtype=np.int64)
    np.cumsum(np.bincount(ti["tid"].values, minlength=L.n_tx), out=in_start[1:])
    inh, fin = np.zeros(n_out), np.zeros(n_out)
    in_taint = np.zeros(len(ti))
    isat, iaid, ptid, pidx = ti["sat"].values, ti["aid"].values, ti["prev_tid"].values, ti["prev_idx"].values
    osat, oaid = to["sat"].values, to["aid"].values
    for t in range(L.n_tx):
        a, b = in_start[t], in_start[t + 1]
        tv = np.empty(b - a)
        for k in range(a, b):
            if ptid[k] >= 0:
                tv[k - a] = fin[out_start[ptid[k]] + pidx[k]]
            else:                                   # funded before the window: tainted only if the address is a seed
                tv[k - a] = 1.0 if seed_addr[iaid[k]] else 0.0
        in_taint[a:b] = tv
        w = float((tv * isat[a:b]).sum() / max(isat[a:b].sum(), 1))
        c, d = out_start[t], out_start[t + 1]
        inh[c:d] = w
        fin[c:d] = np.where(seed_addr[oaid[c:d]], 1.0, w)
    return inh, fin, in_taint


def personalised_pagerank(edges: pd.DataFrame, n: int, seeds: np.ndarray, reverse=False, alpha=0.15, iters=60):
    """Power iteration on the value-weighted flow matrix; teleport = uniform over seed entities."""
    s, d = (edges["dst"].values, edges["src"].values) if reverse else (edges["src"].values, edges["dst"].values)
    w = np.log1p(edges["sat"].values / 1e5)
    M = sp.coo_matrix((w, (d, s)), shape=(n, n)).tocsr()      # column j -> distributes j's mass to rows i
    colsum = np.asarray(M.sum(0)).ravel()
    inv = np.where(colsum > 0, 1.0 / np.maximum(colsum, 1e-12), 0.0)
    M = M @ sp.diags(inv)
    e = np.zeros(n)
    e[seeds] = 1.0 / max(len(seeds), 1)
    r = e.copy()
    for _ in range(iters):
        dangling = r[colsum == 0].sum()
        r = alpha * e + (1 - alpha) * (M @ r + dangling * e)
    return r


def hop_distance(edges: pd.DataFrame, n: int, seeds: np.ndarray, reverse=False, cap=6):
    a, b = (edges["dst"].values, edges["src"].values) if reverse else (edges["src"].values, edges["dst"].values)
    G = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    if len(seeds) == 0:
        return np.full(n, float(cap))
    d = dijkstra(G, directed=True, indices=seeds, unweighted=True, min_only=True, limit=cap)
    d = np.where(np.isinf(d), cap, d)
    return d


# ------------------------------------------------------------------ entity feature table
def _entropy24(h):
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(24))


def entity_features(L: Ledger, ent: np.ndarray, F: pd.DataFrame, coinjoin: np.ndarray, peel_p: np.ndarray,
                    seed_addr: np.ndarray):
    """Return (feature DataFrame indexed by entity id, entity flow edges, meta dict)."""
    tx, ti, to = L.tx, L.txin, L.txout
    n = ent.max() + 1
    edges = cluster_edges(L, ent)
    seeds_e = np.unique(ent[seed_addr])
    inh, fin, in_taint = propagate_taint(L, seed_addr)

    # --- sender table: distinct (tid, entity)
    S = pd.DataFrame({"tid": ti["tid"].values, "e": ent[ti["aid"].values], "sat": ti["sat"].values, "taint": in_taint})
    Sg = S.groupby(["tid", "e"], as_index=False).agg(sat=("sat", "sum"), tv=("taint", lambda x: 0.0))
    Sg["tv"] = S.assign(tw=S["sat"] * S["taint"]).groupby(["tid", "e"])["tw"].sum().values
    t = Sg["tid"].values
    Sg["fs"] = tx["first_seen"].values[t]
    Sg["fee"] = tx["fee_rate"].values[t]
    Sg["anon"] = tx["first_anon_score"].values[t]
    Sg["ip"] = tx["first_src_ip"].values[t]
    Sg["cls"] = tx["first_net_class"].values[t]
    Sg["n_in"], Sg["n_out"] = tx["n_in"].values[t], tx["n_out"].values[t]
    Sg["cj"] = coinjoin[t]
    Sg["peel"] = peel_p[t] > 0.5
    Sg["cv"] = F["out_cv"].values[t]
    Sg["hop"] = (Sg["n_in"] == 1) & (Sg["n_out"] == 1)
    Sg["eqfan"] = (Sg["n_out"] >= 6) & (Sg["cv"] < 0.02) & (Sg["n_in"] <= 3) & ~Sg["cj"]
    Sg["hr"] = ((Sg["fs"] // 3600) % 24).astype(int)
    Sg["night"] = (Sg["hr"] < 6) | (Sg["hr"] >= 22)
    gs = Sg.groupby("e")
    X = pd.DataFrame(index=np.arange(n))
    X["n_sent"] = gs.size()
    X["sent_btc"] = gs["sat"].sum() / SAT
    X["cj_frac"] = gs["cj"].mean()
    X["cj_btc_frac"] = Sg.assign(v=Sg["sat"] * Sg["cj"]).groupby("e")["v"].sum() / gs["sat"].sum()
    X["peel_frac"], X["hop_frac"] = gs["peel"].mean(), gs["hop"].mean()
    X["fan_out_max"] = Sg[~Sg["cj"]].groupby("e")["n_out"].max()
    X["fan_in_max"] = gs["n_in"].max()
    X["equal_fanout_n"] = gs["eqfan"].sum()
    ok3 = gs.size() >= 3
    X["fee_cv"] = (gs["fee"].std(ddof=0) / gs["fee"].mean().clip(lower=1e-6))[ok3]
    X["fee_mean"] = gs["fee"].mean()[ok3]
    X["anon_mean"] = gs["anon"].mean()
    X["tor_vpn_share"] = Sg["cls"].isin(["tor", "vpn"]).groupby(Sg["e"]).mean()
    X["ip_top_share"] = gs["ip"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    H = np.zeros((n, 24))
    np.add.at(H, (Sg["e"].values, Sg["hr"].values), 1)
    X["hour_entropy"] = [_entropy24(h) if h.sum() >= 3 else np.nan for h in H]
    X["night_share"] = gs["night"].mean()
    X["taint_spent_frac"] = gs["tv"].sum() / gs["sat"].sum().clip(lower=1)
    X["first_active"], X["last_active"] = gs["fs"].min(), gs["fs"].max()

    # --- receiver table
    R = pd.DataFrame({"tid": to["tid"].values, "e": ent[to["aid"].values], "sat": to["sat"].values,
                      "inh": inh, "spent": to["spent_tid"].values})
    Rg = R.groupby(["tid", "e"], as_index=False).agg(sat=("sat", "sum"))
    X["n_recv"] = Rg.groupby("e").size()
    fsr = tx["first_seen"].values
    R["dwell"] = np.where(R["spent"] >= 0, (fsr[np.maximum(R["spent"].values, 0)] - fsr[R["tid"].values]) / 3600.0, np.nan)
    X["dwell_med_h"] = R.groupby("e")["dwell"].median()
    X["frac_fast"] = R.assign(f=(R["dwell"] < 6).astype(float)).groupby("e")["f"].mean()
    # external (cross-entity) flows
    out_b = edges.groupby("src")["sat"].sum() / SAT
    in_b = edges.groupby("dst")["sat"].sum() / SAT
    X["out_btc"], X["in_btc"] = out_b, in_b
    X["cp_out"] = edges.groupby("src")["dst"].nunique()
    X["cp_in"] = edges.groupby("dst")["src"].nunique()
    X = X.fillna({c: 0 for c in ["out_btc", "in_btc", "cp_out", "cp_in", "n_sent", "n_recv", "sent_btc", "equal_fanout_n",
                                 "fan_out_max", "fan_in_max", "cj_frac", "cj_btc_frac", "peel_frac", "hop_frac",
                                 "taint_spent_frac", "anon_mean", "tor_vpn_share", "ip_top_share", "night_share"]})
    X["pass_ratio"] = np.minimum(X["in_btc"], X["out_btc"]) / np.maximum(np.maximum(X["in_btc"], X["out_btc"]), 1e-9)
    # hub-ness of destinations: fraction of outgoing value going to high in-degree entities (exchange-like)
    hub = (X["cp_in"] >= 30).values
    e2 = edges.assign(h=hub[edges["dst"].values] * edges["sat"])
    X["out_to_hub_frac"] = (e2.groupby("src")["h"].sum() / edges.groupby("src")["sat"].sum()).reindex(X.index).fillna(0)
    X["log_addr"] = np.log1p(np.bincount(ent, minlength=n))
    X["log_sent"], X["log_recv"] = np.log1p(X["n_sent"]), np.log1p(X["n_recv"])
    X["log_out_btc"], X["log_in_btc"] = np.log1p(X["out_btc"]), np.log1p(X["in_btc"])
    X["log_cp_out"], X["log_cp_in"] = np.log1p(X["cp_out"]), np.log1p(X["cp_in"])
    X["log_avg_out_btc"] = np.log1p(X["out_btc"] / np.maximum(X["n_sent"], 1))
    X["recv_burst"] = np.log1p(X["in_btc"] / np.maximum(X["n_recv"], 1))
    # address level
    from .common import script_of
    scr = np.array([script_of(a) == "p2pkh" for a in L.addrs])
    X["p2pkh_share"] = pd.Series(scr).groupby(ent).mean().reindex(X.index).values
    uses = pd.Series(np.bincount(to["aid"].values, minlength=L.n_addr))
    X["addr_reuse"] = (uses >= 2).groupby(ent).mean().reindex(X.index).values
    # --- propagation
    ext_in = R.merge(Sg[["tid", "e"]].assign(is_self=1), on=["tid", "e"], how="left")
    ext = ext_in[ext_in["is_self"].isna()]
    tin = (ext["sat"] * ext["inh"]).groupby(ext["e"]).sum()
    X["taint_in_btc"] = (tin / SAT).reindex(X.index).fillna(0)
    X["taint_in_frac"] = (tin / ext.groupby("e")["sat"].sum().clip(lower=1)).reindex(X.index).fillna(0)
    X["log_taint_in_btc"] = np.log1p(X["taint_in_btc"])
    X["dist_fwd"] = hop_distance(edges, n, seeds_e, reverse=False)
    X["dist_bwd"] = hop_distance(edges, n, seeds_e, reverse=True)
    for name, rev in (("ppr_fwd", False), ("ppr_bwd", True)):
        r = personalised_pagerank(edges, n, seeds_e, reverse=rev)
        X[name] = np.log10(r + 1e-9)
    X["is_seed"] = np.isin(np.arange(n), seeds_e)
    X["active"] = (X["n_sent"] + X["n_recv"]) >= 3
    return X, edges, dict(taint_final=fin, taint_inherit=inh, in_taint=in_taint, seeds_e=seeds_e, Sg=Sg)
