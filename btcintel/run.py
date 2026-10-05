"""`run` stage: raw files -> SQLite store -> analysis -> alerts -> report files + dashboard."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .alerts import build_alerts
from .dashboard import render
from .ingest import enrich_geo, ingest_paths
from .graphs import build_ledger
from .pipeline import analyse, load_seeds
from .prep import find_geo
from .store import Store


def run(data_dir, models_path, out_dir, metrics_path=None, verbose=True):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)
    log("[1/5] ingest + validate + geo-enrich")
    obs, tx, txin, txout, rej, stats = ingest_paths([data_dir / "raw"], verbose=verbose)
    obs, tx = enrich_geo(obs, tx, find_geo(data_dir))
    st_db = Store(out_dir / "btcintel.db")
    for name, df, idx in (("obs", obs, ["txid"]), ("tx", tx, ["txid"]), ("txin", txin, ["addr", "txid"]),
                          ("txout", txout, ["addr", "txid"]), ("rejects", rej, [])):
        st_db.write(name, df, idx)
    log("[2/5] ledger, spend graph, models, entity resolution")
    L = build_ledger(tx, txin, txout)
    models = joblib.load(models_path)
    seed_addr, seed_meta = load_seeds(data_dir, L)
    st = analyse(L, models, seed_addr, seed_meta)
    log(f"      {L.n_tx} tx, {L.n_addr} addresses -> {st.ent.max()+1} entities, "
        f"{int(st.coinjoin.sum())} CoinJoin txs, {int((st.chain_p>=0.5).sum())} laundering chains")
    log("[3/5] alerts + explanations")
    alerts, aux = build_alerts(st, models)
    log(f"      {len(alerts)} leads")
    log("[4/5] persist")
    st_db.write("entity_map", pd.DataFrame({"address": L.addrs, "entity": st.ent, "cioh_cluster": st.cl0}), ["address", "entity"])
    ent = st.X.copy(); ent["entity"] = ent.index; ent["risk"] = aux["risk"]; ent["anomaly"] = aux["anomaly"]
    st_db.write("entities", ent.reset_index(drop=True), ["entity"])
    flat = pd.DataFrame([{k: a[k] for k in ("id", "rank", "type", "subject", "score", "priority", "headline", "btc")} |
                         {"reasons": "; ".join(f"{r['label']}={r['shown']} ({r['contribution']:+.3f})" for r in a["reasons"])} for a in alerts])
    st_db.write("alerts", flat, ["id"])
    st_db.write("alert_json", pd.DataFrame({"id": [a["id"] for a in alerts], "json": [json.dumps(a, default=str) for a in alerts]}), ["id"])
    flat.to_csv(out_dir / "alerts.csv", index=False)
    json.dump(alerts, open(out_dir / "alerts.json", "w"), default=str)
    summary = dict(stats, entities=int(st.ent.max() + 1), seeds=int(seed_addr.sum()), coinjoin_txs=int(st.coinjoin.sum()))
    metrics = json.load(open(metrics_path)) if metrics_path and Path(metrics_path).exists() else None
    log("[5/5] dashboard")
    render(dict(summary=summary, alerts=alerts, metrics=metrics), out_dir / "dashboard.html")
    log(f"done -> {out_dir/'dashboard.html'}")
    return alerts, st, aux
