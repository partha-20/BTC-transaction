"""Phase 2 - bulk ingestion of network+blockchain metadata (CSV / JSON / JSONL / XML, optionally .gz).

Streaming parsers -> strict per-row validation -> normalised tables:

    obs    one row per network sighting  (ts, src/dst ip:port, txid, + geo enrichment)
    tx     one row per unique transaction (first-seen time & IP, sizes, fee, fee-rate, ...)
    txin   (txid, idx, addr, sats)      txout  (txid, idx, addr, sats)
    rejects  every dropped row with the reason (nothing is silently discarded)
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import SAT, is_valid_ipv4, script_of

csv.field_size_limit(10 ** 8)
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
LIST_FIELDS = ("input_addresses", "output_addresses", "input_amounts", "output_amounts")
REQUIRED = ("timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid") + LIST_FIELDS


class NotObservationFile(Exception):
    pass


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, "r", encoding="utf-8", newline="")


def _kind(path: Path) -> str:
    n = path.name.lower().removesuffix(".gz")
    for ext, k in ((".csv", "csv"), (".tsv", "csv"), (".jsonl", "jsonl"), (".ndjson", "jsonl"), (".json", "json"), (".xml", "xml")):
        if n.endswith(ext):
            return k
    raise ValueError(f"unsupported file type: {path}")


def iter_records(path: Path):
    """Yield raw dict records from one file, streaming where the format allows."""
    k = _kind(path)
    with _open(path) as f:
        if k == "csv":
            rd = csv.DictReader(f, delimiter="\t" if ".tsv" in path.name.lower() else ",")
            if not rd.fieldnames or "txid" not in rd.fieldnames:
                raise NotObservationFile(path.name)
            yield from rd
        elif k == "jsonl":
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"_broken": line[:80]}
        elif k == "json":
            data = json.load(f)
            yield from (data["observations"] if isinstance(data, dict) else data)
    if k == "xml":
        with _open(path) as f:
            for _, el in ET.iterparse(f, events=("end",)):
                if el.tag == "observation":
                    rec = {}
                    for c in el:
                        rec[c.tag] = [i.text for i in c.findall("item")] if c.tag in LIST_FIELDS else (c.text or "")
                    yield rec
                    el.clear()


def parse_ts(v):
    """Fast timestamp parse -> epoch seconds (float) or None. Accepts ISO-8601 and epoch numbers."""
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    s = str(v).strip()
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() if not s.endswith(("Z", "z")) and "+" not in s[10:] \
            else datetime.fromisoformat(s.replace("z", "Z")).timestamp()
    except ValueError:
        pass
    try:
        x = float(s)
        return x if x > 0 else None
    except ValueError:
        pass
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return None if pd.isna(t) else t.timestamp()


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    s = str(v).strip()
    if s.startswith("["):
        return json.loads(s)
    return [x for x in s.split(";")] if s else []


def normalise(rec: dict):
    """Validate + convert one raw record. Returns (row_dict, None) or (None, reason)."""
    if "_broken" in rec:
        return None, "unparseable_line"
    for k in REQUIRED:
        if k not in rec:
            return None, f"missing_field:{k}"
    txid = str(rec["txid"]).strip().lower()
    if not _HEX64.match(txid):
        return None, "bad_txid"
    ts = parse_ts(rec["timestamp"])
    if ts is None:
        return None, "bad_timestamp"
    for k in ("src_ip", "dst_ip"):
        if not is_valid_ipv4(rec[k]):
            return None, f"bad_ip:{k}"
    try:
        sp, dp = int(rec["src_port"]), int(rec["dst_port"])
        if not (0 < sp < 65536 and 0 < dp < 65536):
            raise ValueError
    except (ValueError, TypeError):
        return None, "bad_port"
    try:
        ia, oa = _as_list(rec["input_addresses"]), _as_list(rec["output_addresses"])
        iv = [float(x) for x in _as_list(rec["input_amounts"])]
        ov = [float(x) for x in _as_list(rec["output_amounts"])]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None, "bad_list"
    if not ia or not oa:
        return None, "empty_inputs_or_outputs"
    if len(ia) != len(iv) or len(oa) != len(ov):
        return None, "length_mismatch"
    if any((not np.isfinite(x)) or x < 0 for x in iv + ov):
        return None, "bad_amount"
    isat = [int(round(x * SAT)) for x in iv]
    osat = [int(round(x * SAT)) for x in ov]
    try:
        fee = int(round(float(rec.get("fee", 0)) * SAT))
    except (ValueError, TypeError):
        return None, "bad_fee"
    if abs((sum(isat) - sum(osat)) - fee) > 2:
        return None, "fee_mismatch"
    return dict(ts=ts, src_ip=str(rec["src_ip"]), dst_ip=str(rec["dst_ip"]), src_port=sp, dst_port=dp,
                txid=txid, in_addr=ia, out_addr=oa, in_sat=isat, out_sat=osat, fee=fee,
                script_type=str(rec.get("script_type") or script_of(ia[0]))), None


def _peek(gen):
    yield from gen


def ingest_paths(paths, verbose=True):
    """Parse all files. Returns (obs, tx, txin, txout, rejects, stats) DataFrames/dict (pre-geo)."""
    files = []
    for p in map(Path, paths):
        files += sorted(x for x in p.rglob("*") if x.is_file() and x.name.lower().removesuffix(".gz").endswith(
            (".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml"))) if p.is_dir() else [p]
    obs_rows, rej, per_file = [], [], {}
    tx_first: dict[str, dict] = {}
    seen = set()
    n_in = n_ok = n_dup = 0
    for f in files:
        c = Counter()
        try:
            recs = list(_peek(iter_records(f)))
        except NotObservationFile:
            if verbose:
                print(f"  {f.name}: skipped (no 'txid' column - not an observation file)")
            continue
        for rec in recs:
            n_in += 1
            row, why = normalise(rec)
            if row is None:
                rej.append((f.name, why, str(rec.get("txid", ""))[:16]))
                c["rejected"] += 1
                continue
            key = (row["txid"], row["src_ip"], row["ts"])
            if key in seen:
                n_dup += 1
                c["duplicate"] += 1
                continue
            seen.add(key)
            n_ok += 1
            c["ok"] += 1
            obs_rows.append((row["ts"], row["src_ip"], row["dst_ip"], row["src_port"], row["dst_port"], row["txid"]))
            cur = tx_first.get(row["txid"])
            if cur is None or row["ts"] < cur["ts"]:
                nobs = cur["n_obs"] + 1 if cur else 1
                row["n_obs"] = nobs
                tx_first[row["txid"]] = row
            else:
                cur["n_obs"] += 1
        per_file[f.name] = dict(c)
        if verbose:
            print(f"  {f.name}: {dict(c)}")
    obs = pd.DataFrame(obs_rows, columns=["ts", "src_ip", "dst_ip", "src_port", "dst_port", "txid"])
    tx_rows, tin, tout = [], [], []
    for txid, r in tx_first.items():
        n_i, n_o = len(r["in_addr"]), len(r["out_addr"])
        vsize = 10.5 + 68 * n_i + 31 * n_o
        tx_rows.append((txid, r["ts"], r["src_ip"], r["src_port"], r["n_obs"], n_i, n_o, sum(r["in_sat"]), sum(r["out_sat"]),
                        r["fee"], r["fee"] / vsize, r["script_type"]))
        tin += [(txid, i, a, s) for i, (a, s) in enumerate(zip(r["in_addr"], r["in_sat"]))]
        tout += [(txid, i, a, s) for i, (a, s) in enumerate(zip(r["out_addr"], r["out_sat"]))]
    tx = pd.DataFrame(tx_rows, columns=["txid", "first_seen", "first_src_ip", "first_src_port", "n_obs", "n_in", "n_out",
                                        "in_sat", "out_sat", "fee_sat", "fee_rate", "script_type"]).sort_values("first_seen").reset_index(drop=True)
    txin = pd.DataFrame(tin, columns=["txid", "idx", "addr", "sat"])
    txout = pd.DataFrame(tout, columns=["txid", "idx", "addr", "sat"])
    rejects = pd.DataFrame(rej, columns=["file", "reason", "txid_prefix"])
    stats = dict(files=per_file, rows_in=n_in, rows_ok=n_ok, duplicates=n_dup, rejected=len(rej),
                 reject_reasons=dict(Counter(r[1] for r in rej)), unique_tx=len(tx), unique_addresses=int(pd.concat([txin.addr, txout.addr]).nunique()))
    return obs, tx, txin, txout, rejects, stats


def enrich_geo(obs: pd.DataFrame, tx: pd.DataFrame, geo) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = geo.lookup(obs["src_ip"])
    obs = pd.concat([obs, g.add_prefix("src_")], axis=1)
    gt = geo.lookup(tx["first_src_ip"])
    tx = pd.concat([tx, gt.add_prefix("first_")], axis=1)
    return obs, tx
