"""Phase 3 - ledger indexing, entity/transaction graph, and network<->blockchain correlation.

Key ideas
---------
* Inputs in the challenge schema carry (address, amount) but no prev-outpoint, so spend edges are
  reconstructed by matching (address, amount) to the earliest unspent earlier output (FIFO).
* Common-Input-Ownership (CIOH) clustering via connected components - but CoinJoin-classified
  transactions are excluded (they merge unrelated owners) using the ML mixing detector.
* Cluster flow graph + spectral embeddings (truncated SVD of the normalised adjacency).
* IP correlation: for every cluster we count which *first-seen relay IPs* announce its transactions;
  IPs that are concentrated on one cluster expose the real origin node (the sensor sees the
  originating node first in a large share of cases).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds


@dataclass
class Ledger:
    tx: pd.DataFrame        # + tid
    txin: pd.DataFrame      # + tid, aid, prev_tid, prev_idx
    txout: pd.DataFrame     # + tid, aid, spent_tid, spent_idx
    addrs: np.ndarray       # aid -> address string
    n_addr: int

    @property
    def n_tx(self):
        return len(self.tx)


def build_ledger(tx: pd.DataFrame, txin: pd.DataFrame, txout: pd.DataFrame) -> Ledger:
    tx = tx.sort_values("first_seen").reset_index(drop=True).copy()
    tx["tid"] = np.arange(len(tx))
    tmap = pd.Series(tx["tid"].values, index=tx["txid"].values)
    txin, txout = txin.copy(), txout.copy()
    txin["tid"], txout["tid"] = txin["txid"].map(tmap).values, txout["txid"].map(tmap).values
    codes, uniq = pd.factorize(pd.concat([txin["addr"], txout["addr"]], ignore_index=True))
    txin["aid"], txout["aid"] = codes[:len(txin)], codes[len(txin):]
    txin = txin.sort_values(["tid", "idx"]).reset_index(drop=True)
    txout = txout.sort_values(["tid", "idx"]).reset_index(drop=True)

    # ---- reconstruct spends: FIFO match on (aid, sat) against outputs created earlier
    pending: dict[tuple, list] = {}
    prev_tid = np.full(len(txin), -1, dtype=np.int64)
    prev_idx = np.full(len(txin), -1, dtype=np.int64)
    spent_tid = np.full(len(txout), -1, dtype=np.int64)
    spent_idx = np.full(len(txout), -1, dtype=np.int64)
    out_pos = {(t, i): p for p, (t, i) in enumerate(zip(txout["tid"].values, txout["idx"].values))}
    in_by_tid = txin.groupby("tid").indices
    out_by_tid = txout.groupby("tid").indices
    ia, isat, oa, osat = txin["aid"].values, txin["sat"].values, txout["aid"].values, txout["sat"].values
    for t in range(len(tx)):
        for p in in_by_tid.get(t, ()):
            lst = pending.get((ia[p], isat[p]))
            if lst:
                pt, pi = lst.pop(0)
                prev_tid[p], prev_idx[p] = pt, pi
                q = out_pos[(pt, pi)]
                spent_tid[q], spent_idx[q] = t, txin["idx"].values[p]
        for p in out_by_tid.get(t, ()):
            pending.setdefault((oa[p], osat[p]), []).append((t, txout["idx"].values[p]))
    txin["prev_tid"], txin["prev_idx"] = prev_tid, prev_idx
    txout["spent_tid"], txout["spent_idx"] = spent_tid, spent_idx
    return Ledger(tx, txin, txout, np.asarray(uniq, dtype=object), len(uniq))


# -------------------------------------------------------------------- clustering
def cioh_clusters(L: Ledger, skip_tid: np.ndarray | None = None) -> np.ndarray:
    """Common-input-ownership clusters -> label per address id. `skip_tid` = bool mask over tid (CoinJoins)."""
    inn = L.txin
    if skip_tid is not None:
        inn = inn[~skip_tid[inn["tid"].values]]
    first = inn.groupby("tid")["aid"].transform("first").values
    m = inn["aid"].values != first
    r, c = first[m], inn["aid"].values[m]
    g = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(L.n_addr, L.n_addr))
    _, lab = connected_components(g, directed=False)
    return lab


def merge_labels(base: np.ndarray, links: list[tuple[int, int]], max_size: int = 60000) -> np.ndarray:
    """Union clusters (ids in `base` label space) with a size guard; returns new label per address."""
    n = base.max() + 1
    parent = np.arange(n)
    size = np.bincount(base, minlength=n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb and size[ra] + size[rb] <= max_size:
            parent[rb] = ra
            size[ra] += size[rb]
    roots = np.array([find(x) for x in range(n)])
    _, new = np.unique(roots, return_inverse=True)
    return new[base]


def cluster_edges(L: Ledger, cl: np.ndarray) -> pd.DataFrame:
    """Aggregated value flows between clusters (proportional attribution for multi-cluster inputs)."""
    inn = L.txin.assign(c=cl[L.txin["aid"].values])
    out = L.txout.assign(c=cl[L.txout["aid"].values])
    ig = inn.groupby(["tid", "c"], as_index=False)["sat"].sum().rename(columns={"c": "src", "sat": "in_sat"})
    ig["share"] = ig["in_sat"] / ig.groupby("tid")["in_sat"].transform("sum")
    og = out.groupby(["tid", "c"], as_index=False)["sat"].sum().rename(columns={"c": "dst", "sat": "out_sat"})
    m = ig.merge(og, on="tid")
    m["flow"] = m["out_sat"] * m["share"]
    m = m[m["src"] != m["dst"]]
    t = L.tx["first_seen"].values
    m["t"] = t[m["tid"].values]
    e = m.groupby(["src", "dst"]).agg(sat=("flow", "sum"), ntx=("tid", "nunique"), t_first=("t", "min"), t_last=("t", "max")).reset_index()
    return e


def spectral_embedding(edges: pd.DataFrame, n_nodes: int, k: int = 24, seed: int = 0) -> np.ndarray:
    """Graph embedding: top-k singular vectors of the symmetric-normalised adjacency (log-weighted)."""
    w = np.log1p(edges["ntx"].values.astype(float))
    A = sp.coo_matrix((w, (edges["src"].values, edges["dst"].values)), shape=(n_nodes, n_nodes)).tocsr()
    A = A + A.T
    d = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    An = sp.diags(dinv) @ A @ sp.diags(dinv)
    k = min(k, n_nodes - 2)
    rng = np.random.default_rng(seed)
    u, s, _ = svds(An.astype(float), k=k, v0=rng.standard_normal(n_nodes))
    emb = u * np.sqrt(np.abs(s))
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(nrm, 1e-9)


# -------------------------------------------------------------------- IP evidence
def ip_evidence(L: Ledger, cl: np.ndarray):
    """Return (cluster_of_tx, per-(ip,cluster) counts, per-ip totals).
    Uses the cluster owning the *first input* of each tx as the 'sender' cluster."""
    first_in = L.txin.groupby("tid")["aid"].first()
    tcl = pd.Series(-1, index=L.tx["tid"].values)
    tcl.loc[first_in.index] = cl[first_in.values]
    df = pd.DataFrame({"tid": L.tx["tid"].values, "cluster": tcl.values, "ip": L.tx["first_src_ip"].values})
    df = df[df["cluster"] >= 0]
    pair = df.groupby(["ip", "cluster"]).size().rename("n").reset_index()
    tot = df.groupby("ip").size().rename("n_ip")
    return tcl.values, pair.merge(tot, left_on="ip", right_index=True), tot
