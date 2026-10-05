"""Evaluation against synthetic ground truth (never used by detection).  Produces reports/metrics.json."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from .alerts import build_alerts
from .chains import CHAIN_FEATURES
from .entities import BEHAVIOUR_FEATURES, ENTITY_FEATURES, PROPAGATION_FEATURES
from .entity import pairwise_prf
from .graphs import cioh_clusters
from .models import PeelModel, RiskModel
from .pipeline import analyse, entity_labels, load_seeds, load_truth
from .prep import prepare

NETWORK_FEATURES = ["anon_mean", "tor_vpn_share", "ip_top_share"]
EVASION_SENSITIVE = ["fee_cv", "fee_mean", "hour_entropy", "night_share", "gap_h_med", "gap_h_cv", "gap_h_mean", "anon_mean", "ip_top_share"]


def _auc_ap(y, s):
    y = np.asarray(y).astype(int)
    return dict(auc=round(float(roc_auc_score(y, s)), 4), ap=round(float(average_precision_score(y, s)), 4), positives=int(y.sum()), n=int(len(y)))


def _chain_labels(chains, pattern):
    return np.array([np.isin(pattern[p], ["peel", "hop"]).mean() >= 0.5 for p in chains])


def _entity_table(st, T):
    d = st.X.join(entity_labels(st.ent, T), how="inner")
    return d[d["active"] & ~d["is_seed"] & (d["purity"] >= 0.8)]


def evaluate(train_dir, data_dir, models_path, out_path, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    models = joblib.load(models_path)
    L, obs, rej, stats = prepare(data_dir)
    T = load_truth(data_dir, L)
    sa, sm = load_seeds(data_dir, L)
    st = analyse(L, models, sa, sm)
    M, rows = {}, []

    # 1 ingestion
    man = T["gt"]["dirty_manifest"]; dup = man["dup"]; bad = sum(man.values()) - dup
    M["ingestion"] = dict(injected_bad_rows=bad, rejected=int(stats["rejected"]), injected_duplicates=dup, duplicates_dropped=int(stats["duplicates"]))
    rows.append(["Ingestion", "corrupted rows caught / injected", f"{stats['rejected']} / {bad}"])
    rows.append(["Ingestion", "duplicates dropped / injected", f"{stats['duplicates']} / {dup}"])

    # 2 mixing
    y = T["pattern"] == "coinjoin"
    pr, rc, f1, _ = precision_recall_fscore_support(y, st.coinjoin, average="binary")
    M["mixing"] = dict(**_auc_ap(y, st.coinjoin_p), precision=round(float(pr), 3), recall=round(float(rc), 3))
    rows.append(["CoinJoin detection", "AUC / precision / recall", f"{M['mixing']['auc']} / {pr:.3f} / {rc:.3f}"])

    # 3 entity clustering
    ill_addr = np.array([o >= 0 and T["illicit_w"][o] for o in T["owner"]])
    ent = {}
    for name, lab in (("cioh_ungated", cioh_clusters(L)), ("cioh_coinjoin_gated", st.cl0), ("cioh+ml_link+ip", st.ent)):
        ent[name] = dict(all=pairwise_prf(lab, T["owner"]), illicit_only=pairwise_prf(lab[ill_addr], T["owner"][ill_addr]))
    M["clustering"] = ent
    for n, v in ent.items():
        rows.append(["Entity clustering", f"{n}: pairwise P / R (illicit R)", f"{v['all']['precision']:.3f} / {v['all']['recall']:.3f} ({v['illicit_only']['recall']:.3f})"])

    # 4 chains
    yc = _chain_labels(st.chains, T["pattern"])
    M["chains"] = dict(model=_auc_ap(yc, st.chain_p), length_only_baseline=_auc_ap(yc, st.chain_X["length"].values))
    truth_tx = np.isin(T["pattern"], ["peel", "hop"])
    flagged = np.zeros(L.n_tx, bool)
    for p, s in zip(st.chains, st.chain_p):
        if s >= .5: flagged[p] = True
    M["chains"]["tx_recall"] = round(float((flagged & truth_tx).sum() / truth_tx.sum()), 3)
    M["chains"]["tx_false_flags"] = int((flagged & ~truth_tx).sum())
    # train-side state for ablations
    Ltr, *_ = prepare(train_dir); Ttr = load_truth(train_dir, Ltr); sat_, smt = load_seeds(train_dir, Ltr)
    tr = analyse(Ltr, models, sat_, smt)
    ytr = _chain_labels(tr.chains, Ttr["pattern"])
    keep = [f for f in CHAIN_FEATURES if f not in EVASION_SENSITIVE]
    m2 = PeelModel().fit(tr.chain_X[keep], ytr)
    M["chains"]["ablation_no_fee_timing_ip_features"] = _auc_ap(yc, m2.predict_proba(st.chain_X))
    rows.append(["Laundering chains", "AUC (model / length-only baseline)", f"{M['chains']['model']['auc']} / {M['chains']['length_only_baseline']['auc']}"])
    rows.append(["Laundering chains", "AUC without fee/timing/IP tells", str(M['chains']['ablation_no_fee_timing_ip_features']['auc'])])
    rows.append(["Laundering chains", "true laundering txs recovered / false-flag txs", f"{M['chains']['tx_recall']} / {M['chains']['tx_false_flags']}"])

    # 5 risk
    d = _entity_table(st, T); dtr = tr.X.join(entity_labels(tr.ent, Ttr), how="inner")
    dtr = dtr[dtr["active"] & ~dtr["is_seed"] & (dtr["purity"] >= .8)]
    med = d[ENTITY_FEATURES].median()
    p = models["risk"].predict_proba(d[ENTITY_FEATURES].fillna(med))
    risk = dict(fused=_auc_ap(d["illicit"], p))
    for nm, col, sign in (("taint_only", "taint_in_frac", 1), ("pagerank_fwd_only", "ppr_fwd", 1), ("hop_distance_only", "dist_fwd", -1)):
        risk[nm] = _auc_ap(d["illicit"], sign * d[col])
    for nm, feats in (("behaviour_only", BEHAVIOUR_FEATURES), ("propagation_only", PROPAGATION_FEATURES),
                      ("no_network_features", [f for f in ENTITY_FEATURES if f not in NETWORK_FEATURES])):
        m = RiskModel().fit(dtr[feats].fillna(dtr[feats].median()), dtr["illicit"])
        risk[nm] = _auc_ap(d["illicit"], m.predict_proba(d[feats].fillna(d[feats].median())))
    M["risk"] = risk
    for k, v in risk.items():
        rows.append(["Risk scoring", f"{k}: AUC / AP", f"{v['auc']} / {v['ap']}"])

    # 6 alerts
    alerts, aux = build_alerts(st, models)
    lab = entity_labels(st.ent, T)
    er = [a for a in alerts if a["type"] == "ENTITY_RISK"]
    ok = np.array([lab["illicit"].get(a["subject_id"], -1) == 1 for a in er]); known = np.array([a["subject_id"] in lab.index for a in er])
    sc = np.array([a["score"] for a in er])
    order = np.argsort(-sc)
    at = {}
    for k in (10, 25, 50):
        at[f"precision@{k}"] = round(float(ok[order][:k].mean()), 3)
    bands = {}
    for lo, hi, nm in ((.9, 1.01, "score>=0.9"), (.7, .9, "0.7-0.9"), (.5, .7, "0.5-0.7")):
        m = (sc >= lo) & (sc < hi)
        bands[nm] = dict(n=int(m.sum()), precision=round(float(ok[m].mean()), 3) if m.sum() else None)
    ill_entities = set(d.index[d["illicit"] == 1])
    rec = len(ill_entities & {a["subject_id"] for a in er}) / max(len(ill_entities), 1)
    by_kind = {}
    for kind, g in d.assign(flag=d.index.isin([a["subject_id"] for a in er]))[d["illicit"] == 1].groupby("kind"):
        by_kind[kind] = dict(entities=len(g), flagged=int(g["flag"].sum()))
    M["alerts"] = dict(entity_risk_alerts=len(er), illicit_entity_recall=round(float(rec), 3), precision_at=at, precision_by_score_band=bands,
                       illicit_entities_by_kind=by_kind, false_positive_kinds=dict(pd.Series([lab["kind"].get(a["subject_id"], "?") for a, o in zip(er, ok) if not o]).value_counts()))
    rows.append(["Alerts", "ENTITY_RISK leads / recall of illicit entities", f"{len(er)} / {rec:.3f}"])
    rows.append(["Alerts", "precision@10 / @25 / @50", f"{at['precision@10']} / {at['precision@25']} / {at['precision@50']}"])
    for nm, v in bands.items():
        rows.append(["Alerts", f"precision, {nm} (n={v['n']})", str(v["precision"])])
    M["summary_rows"] = rows
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(M, open(out_path, "w"), indent=1, default=str)
    if verbose:
        for r in rows: print(f"  {r[0]:20s} {r[1]:60s} {r[2]}")
    return M
