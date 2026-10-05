"""Glue: raw files -> validated, geo-enriched, indexed Ledger (with optional pickle cache for dev)."""
from __future__ import annotations
import pickle
from pathlib import Path
from .geoip import GeoIP
from .graphs import build_ledger
from .ingest import enrich_geo, ingest_paths


def find_geo(data_dir: Path):
    g = data_dir / "geoip"
    c = sorted(g.glob("*country*.csv*"))
    a = sorted(g.glob("*asn*.csv*"))
    t = g / "tor_exit_nodes.txt"
    return GeoIP(c[-1], a[-1] if a else None, t if t.exists() else None)


def prepare(data_dir, verbose=False, cache=False):
    data_dir = Path(data_dir)
    cp = data_dir / ".prep_cache.pkl"
    if cache and cp.exists():
        return pickle.load(open(cp, "rb"))
    obs, tx, txin, txout, rej, stats = ingest_paths([data_dir / "raw"], verbose=verbose)
    obs, tx = enrich_geo(obs, tx, find_geo(data_dir))
    L = build_ledger(tx, txin, txout)
    out = (L, obs, rej, stats)
    if cache:
        pickle.dump(out, open(cp, "wb"))
    return out
