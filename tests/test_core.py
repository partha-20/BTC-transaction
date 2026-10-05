"""Fast unit/integration tests (small synthetic dataset, no model training needed).  Run: python -m pytest -q"""
import json, tempfile
from pathlib import Path
import numpy as np

from btcintel.datagen import generate
from btcintel.explain import shapley
from btcintel.geoip import GeoIP, classify_org
from btcintel.graphs import build_ledger, cioh_clusters
from btcintel.ingest import ingest_paths, enrich_geo, parse_ts
from btcintel.prep import find_geo
from btcintel.entity import pairwise_prf


_CACHE = {}


def _ds():
    if "d" not in _CACHE:
        d = Path(tempfile.mkdtemp())
        _CACHE["d"] = (d, generate(d, seed=5, days=8, n_users=150, n_ransom=1, n_markets=1, n_layer=1))
    return _CACHE["d"]


def test_ingest_catches_every_injected_fault():
    d, man = _ds()
    obs, tx, txin, txout, rej, st = ingest_paths([d / "raw"], verbose=False)
    dirt = man["dirty"]
    assert st["rejected"] == sum(dirt.values()) - dirt["dup"]
    assert st["duplicates"] == dirt["dup"]
    assert set(st["files"]) >= {"observations_part1.csv", "observations_part2.jsonl", "observations_part3.xml"}


def test_amounts_balance_and_geo_enrichment():
    d, _ = _ds()
    obs, tx, txin, txout, rej, st = ingest_paths([d / "raw"], verbose=False)
    assert ((tx.in_sat - tx.out_sat) == tx.fee_sat).all()
    obs, tx = enrich_geo(obs, tx, find_geo(d))
    assert (tx.first_country != "ZZ").mean() > 0.99
    assert set(tx.first_net_class) <= {"residential", "hosting", "vpn", "tor", "unknown"}


def test_spend_reconstruction_and_cioh_purity():
    d, _ = _ds()
    obs, tx, txin, txout, rej, st = ingest_paths([d / "raw"], verbose=False)
    L = build_ledger(tx, txin, txout)
    assert (L.txin.prev_tid >= 0).mean() > 0.6   # rest = funds created before the window
    # every matched spend refers to an earlier transaction
    m = L.txin[L.txin.prev_tid >= 0]
    assert (m.prev_tid < m.tid).all()
    gt = json.load(open(d / "ground_truth" / "ground_truth.json"))
    codes = {w: i for i, w in enumerate(sorted(gt["wallets"]))}
    owner = np.array([codes.get(gt["addr_owner"].get(a), -1) for a in L.addrs])
    pat = np.array([gt["tx_pattern"].get(x, "") for x in L.tx.txid])
    r = pairwise_prf(cioh_clusters(L, pat == "coinjoin"), owner)
    assert r["precision"] > 0.999          # CIOH is exact when CoinJoins are excluded


def test_shapley_is_additive_for_linear_model():
    w = np.array([2.0, -1.0, 0.5])
    fn = lambda M: M @ w
    bg = np.random.default_rng(0).normal(size=(50, 3))
    x = np.array([1.0, 2.0, 3.0])
    phi, base, fx = shapley(fn, x, bg, n_perm=200)
    assert abs(base + phi.sum() - fx) < 1e-9
    assert np.allclose(phi, w * (x - bg.mean(0)), atol=0.35)


def test_geoip_org_classifier():
    assert classify_org("Relay Collective 3 (Tor exit)") == "tor"
    assert classify_org("Nordlys VPN Services") == "vpn"
    assert classify_org("Iron Cloud Ltd") == "hosting"
    assert classify_org("DE Broadband Telecom 1") == "residential"
    assert classify_org("Motor Vector Networks") == "residential"      # 'tor' inside a word must not match


def test_timestamp_parser():
    assert parse_ts("2026-08-01T00:00:01.500Z") == parse_ts("2026-08-01T00:00:01.5+00:00")
    assert parse_ts("garbage") is None


if __name__ == "__main__":          # runs without pytest:  python tests/test_core.py
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  ", f.__name__)
