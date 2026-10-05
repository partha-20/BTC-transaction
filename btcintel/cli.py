"""Command line:  python -m btcintel {gen,train,run,evaluate,demo}"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="btcintel", description="Offline AI monitoring & analysis of Bitcoin transaction traffic")
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen", help="generate a synthetic dataset (+ ground truth, seeds, geo DB)")
    g.add_argument("--out", required=True); g.add_argument("--seed", type=int, default=7); g.add_argument("--days", type=int, default=30)
    g.add_argument("--users", type=int, default=1000); g.add_argument("--ransom", type=int, default=3)
    g.add_argument("--markets", type=int, default=2); g.add_argument("--layer", type=int, default=2)
    t = sp.add_parser("train", help="fit all ML models on a labelled training dataset")
    t.add_argument("--data", required=True); t.add_argument("--models", default="work/models")
    r = sp.add_parser("run", help="analyse a dataset: ingest -> detect -> alerts -> dashboard")
    r.add_argument("--data", required=True); r.add_argument("--models", default="work/models/models.joblib"); r.add_argument("--out", default="reports/run")
    r.add_argument("--metrics", default="reports/metrics.json")
    e = sp.add_parser("evaluate", help="score everything against ground truth (synthetic data only)")
    e.add_argument("--train", required=True); e.add_argument("--data", required=True); e.add_argument("--models", default="work/models/models.joblib")
    e.add_argument("--out", default="reports/metrics.json")
    d = sp.add_parser("demo", help="end-to-end: generate train+analysis data, train, run, evaluate, re-run with metrics")
    d.add_argument("--workdir", default="data")
    a = ap.parse_args(argv)

    if a.cmd == "gen":
        from .datagen import generate
        s = generate(a.out, seed=a.seed, days=a.days, n_users=a.users, n_ransom=a.ransom, n_markets=a.markets, n_layer=a.layer)
        print(json.dumps({k: s[k] for k in ("n_tx", "n_rows", "n_wallets", "n_addresses", "n_seeds")}))
    elif a.cmd == "train":
        from .pipeline import train
        train(a.data, a.models)
    elif a.cmd == "run":
        from .run import run
        run(a.data, a.models, a.out, a.metrics)
    elif a.cmd == "evaluate":
        from .evaluate import evaluate
        evaluate(a.train, a.data, a.models, a.out)
    elif a.cmd == "demo":
        from .datagen import generate
        from .pipeline import train
        from .run import run
        from .evaluate import evaluate
        w = Path(a.workdir)
        generate(w / "train", seed=11, n_ransom=9, n_markets=4, n_layer=6)
        generate(w / "analysis", seed=42)
        train(w / "train", "work/models")
        evaluate(w / "train", w / "analysis", "work/models/models.joblib", "reports/metrics.json")
        run(w / "analysis", "work/models/models.joblib", "reports/run", "reports/metrics.json")


if __name__ == "__main__":
    main()
