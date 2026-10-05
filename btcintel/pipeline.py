"""Orchestration: raw data -> ledger -> models -> entities -> features.  Used by train / run / evaluate."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .chains import CHAIN_FEATURES, chain_features, extract_chains
from .entities import BEHAVIOUR_FEATURES, ENTITY_FEATURES, entity_features
from .entity import LINK_FEATURES, build_pairs, pair_labels, resolve
from .features import TX_FEATURES, tx_features
from .graphs import Ledger, cioh_clusters, cluster_edges, spectral_embedding
from .models import AnomalyModel, LinkModel, MixingModel, PeelModel, RiskModel
from .prep import prepare

LINK_THRESHOLDS = (0.90, 0.95)          # pass 1, pass 2


@dataclass
class State:
    L: Ledger
    F: pd.DataFrame
    coinjoin: np.ndarray
    coinjoin_p: np.ndarray
    cl0: np.ndarray
    ent: np.ndarray
    chains: list
    chain_X: pd.DataFrame
    chain_p: np.ndarray
    peel_p: np.ndarray
    X: pd.DataFrame
    edges: pd.DataFrame
    aux: dict
    seed_addr: np.ndarray
    seed_meta: pd.DataFrame
    link_pairs: pd.DataFrame = None


def load_seeds(data_dir, L: Ledger):
    p = Path(data_dir) / "raw" / "seeds.csv"
    mask = np.zeros(L.n_addr, dtype=bool)
    if not p.exists():
        return mask, pd.DataFrame(columns=["address", "label", "source"])
    s = pd.read_csv(p)
    idx = {a: i for i, a in enumerate(L.addrs)}
    hit = [idx[a] for a in s["address"] if a in idx]
    mask[hit] = True
    return mask, s


def resolve_entities(L, F, cj, cl0, link_model, train_labels=None):
    """Two-pass ML entity resolution. Returns (ent, first-pass pairs(with y if labels given))."""
    cl, first_pairs = cl0, None
    for k, thr in enumerate(LINK_THRESHOLDS):
        e = cluster_edges(L, cl)
        emb = spectral_embedding(e, cl.max() + 1)
        pairs, _ = build_pairs(L, cl, cj, emb, e)
        if k == 0:
            first_pairs = pairs
        if link_model is None or len(pairs) == 0:
            break
        prob = link_model.predict_proba(pairs)
        cl, _ = resolve(L, cl, pairs, prob, thr=thr)
    return cl, first_pairs


def analyse(L: Ledger, models: dict, seed_addr: np.ndarray, seed_meta=None) -> State:
    F = tx_features(L)
    cj_p = models["mixing"].predict_proba(F)
    cj = cj_p > 0.5
    cl0 = cioh_clusters(L, cj)
    ent, fp = resolve_entities(L, F, cj, cl0, models["link"])
    e_edges = cluster_edges(L, ent)
    chains = extract_chains(L, cj)
    cX = chain_features(L, chains, e_edges, ent)
    cp = models["peel"].predict_proba(cX) if len(cX) else np.zeros(0)
    peel_p = np.zeros(L.n_tx)
    for path, p in zip(chains, cp):
        peel_p[path] = np.maximum(peel_p[path], p)
    X, edges, aux = entity_features(L, ent, F, cj, peel_p, seed_addr)
    return State(L, F, cj, cj_p, cl0, ent, chains, cX, cp, peel_p, X, edges, aux, seed_addr,
                 seed_meta if seed_meta is not None else pd.DataFrame(), fp)


def load_truth(data_dir, L: Ledger):
    """Ground truth aligned to a ledger (training + evaluation only)."""
    gt = json.load(open(Path(data_dir) / "ground_truth" / "ground_truth.json"))
    codes = {w: i for i, w in enumerate(sorted(gt["wallets"]))}
    owner = np.array([codes.get(gt["addr_owner"].get(a), -1) for a in L.addrs])
    illicit_w = np.zeros(len(codes), bool)
    kind_w = np.empty(len(codes), dtype=object)
    for w, i in codes.items():
        illicit_w[i] = gt["wallets"][w]["illicit"]
        kind_w[i] = gt["wallets"][w]["kind"]
    pat = np.array([gt["tx_pattern"].get(t, "") for t in L.tx["txid"]])
    return dict(owner=owner, illicit_w=illicit_w, kind_w=kind_w, pattern=pat, gt=gt)


def entity_labels(ent: np.ndarray, T: dict) -> pd.DataFrame:
    """Majority-owner label per predicted entity: (illicit 0/1, kind, purity)."""
    df = pd.DataFrame({"e": ent, "o": T["owner"]})
    df = df[df["o"] >= 0]
    g = df.groupby("e")["o"]
    maj = g.agg(lambda s: s.value_counts().index[0])
    purity = g.agg(lambda s: s.value_counts().iloc[0] / len(s))
    out = pd.DataFrame({"owner": maj, "purity": purity})
    out["illicit"] = T["illicit_w"][out["owner"].values].astype(int)
    out["kind"] = T["kind_w"][out["owner"].values]
    return out


def train(train_dir, out_dir, verbose=True):
    """Fit every supervised model on the (separate) labelled training dataset + fit anomaly baseline stats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)
    L, *_ = prepare(train_dir)
    T = load_truth(train_dir, L)
    F = tx_features(L)
    models = {}
    log("[train] mixing detector")
    models["mixing"] = MixingModel().fit(F[TX_FEATURES], T["pattern"] == "coinjoin")
    cj = models["mixing"].predict_proba(F) > 0.5
    cl0 = cioh_clusters(L, cj)
    log("[train] entity link model")
    e0 = cluster_edges(L, cl0)
    pairs, _ = build_pairs(L, cl0, cj, spectral_embedding(e0, cl0.max() + 1), e0)
    y = pair_labels(pairs, cl0, T["owner"])
    models["link"] = LinkModel().fit(pairs[LINK_FEATURES], y)
    ent, _ = resolve_entities(L, F, cj, cl0, models["link"])
    log("[train] laundering-chain model")
    e_edges = cluster_edges(L, ent)
    chains = extract_chains(L, cj)
    cX = chain_features(L, chains, e_edges, ent)
    yc = np.array([np.isin(T["pattern"][p], ["peel", "hop"]).mean() >= 0.5 for p in chains])
    models["peel"] = PeelModel().fit(cX[CHAIN_FEATURES], yc)
    cp = models["peel"].predict_proba(cX)
    peel_p = np.zeros(L.n_tx)
    for p, s in zip(chains, cp):
        peel_p[p] = np.maximum(peel_p[p], s)
    seed_addr, _ = load_seeds(train_dir, L)
    X, _, _ = entity_features(L, ent, F, cj, peel_p, seed_addr)
    lab = entity_labels(ent, T)
    d = X.join(lab, how="inner")
    d = d[d["active"] & ~d["is_seed"] & (d["purity"] >= 0.8)]
    log(f"[train] risk model on {len(d)} entities ({int(d.illicit.sum())} illicit)")
    models["risk"] = RiskModel().fit(d[ENTITY_FEATURES], d["illicit"])
    models["meta"] = dict(
        link_features=LINK_FEATURES, chain_features=CHAIN_FEATURES, tx_features=TX_FEATURES,
        entity_features=ENTITY_FEATURES, n_train_entities=len(d), n_train_illicit=int(d.illicit.sum()),
        train_background=d[ENTITY_FEATURES].sample(min(300, len(d)), random_state=0).fillna(d[ENTITY_FEATURES].median()),
        chain_background=cX[CHAIN_FEATURES].sample(min(200, len(cX)), random_state=0))
    joblib.dump(models, out_dir / "models.joblib")
    log(f"[train] saved {out_dir/'models.joblib'}")
    return models
