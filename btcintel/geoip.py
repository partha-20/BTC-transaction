"""Offline GeoIP / ASN enrichment.

Reads the open-source DB-IP "Lite" CSV files (CC-BY-4.0, IPv4 rows only):
    dbip-country-lite-YYYY-MM.csv[.gz]   ->  start_ip,end_ip,country
    dbip-asn-lite-YYYY-MM.csv[.gz]       ->  start_ip,end_ip,asn,organisation
(see scripts/fetch_geoip.sh).  The same format is used for the *synthetic* database
that ships with the demo dataset, so a real database is a drop-in replacement.

Because open ASN files carry only an organisation name, an "anonymity class"
(tor / vpn / hosting / residential) is inferred from the name and, optionally, from a
Tor exit-node list.  This class is a key network-layer feature for the ML models.
"""
from __future__ import annotations

import ipaddress
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ip_to_int

_RX_TOR = re.compile(r"\btor\b|exit relay", re.I)
_RX_VPN = re.compile(r"\bvpn\b|\bproxy\b|anonymi", re.I)
_RX_HOST = re.compile(r"hosting|cloud|data ?cent(er|re)|\bvps\b|\bservers?\b|colo", re.I)

ANON_SCORE = {"tor": 1.0, "vpn": 0.9, "hosting": 0.5, "residential": 0.0, "unknown": 0.2}


def classify_org(org: str) -> str:
    if not isinstance(org, str) or not org:
        return "unknown"
    if _RX_TOR.search(org):
        return "tor"
    if _RX_VPN.search(org):
        return "vpn"
    if _RX_HOST.search(org):
        return "hosting"
    return "residential"


def _read_ranges(path: Path, ncols: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False,
                     names=["start", "end", "a", "b"][:ncols], usecols=range(ncols))
    df = df[df["start"].str.contains(".", regex=False)]          # IPv4 only
    df["s"] = [ip_to_int(x) for x in df["start"]]
    df["e"] = [ip_to_int(x) for x in df["end"]]
    return df.sort_values("s").reset_index(drop=True)


class GeoIP:
    """Vectorised range lookup (numpy searchsorted) - handles millions of rows/second."""

    def __init__(self, country_csv: str | Path, asn_csv: str | Path | None = None,
                 tor_exit_list: str | Path | None = None):
        c = _read_ranges(Path(country_csv), 3)
        self._c_s, self._c_e, self._c_v = c["s"].to_numpy(), c["e"].to_numpy(), c["a"].to_numpy()
        self._a = None
        if asn_csv and Path(asn_csv).exists():
            a = _read_ranges(Path(asn_csv), 4)
            self._a = (a["s"].to_numpy(), a["e"].to_numpy(),
                       a["a"].to_numpy(), a["b"].to_numpy())
        self.tor_exits: set[str] = set()
        if tor_exit_list and Path(tor_exit_list).exists():
            self.tor_exits = {l.strip() for l in open(tor_exit_list) if l.strip() and not l.startswith("#")}

    @staticmethod
    def _find(starts, ends, values, ips):
        idx = np.searchsorted(starts, ips, side="right") - 1
        ok = (idx >= 0) & (ips <= ends[np.clip(idx, 0, len(ends) - 1)])
        return idx, ok

    def lookup(self, ips: pd.Series) -> pd.DataFrame:
        """Return DataFrame(index=ips.index): country, asn, asn_org, net_class, anon_score."""
        uniq = pd.Series(ips.dropna().unique())
        ints = np.array([ip_to_int(x) for x in uniq], dtype=np.int64)
        out = pd.DataFrame({"ip": uniq.values})
        idx, ok = self._find(self._c_s, self._c_e, self._c_v, ints)
        out["country"] = np.where(ok, self._c_v[np.clip(idx, 0, len(self._c_v) - 1)], "ZZ")
        if self._a is not None:
            s, e, asn, org = self._a
            idx, ok = self._find(s, e, asn, ints)
            j = np.clip(idx, 0, len(asn) - 1)
            out["asn"] = np.where(ok, asn[j], "0")
            out["asn_org"] = np.where(ok, org[j], "unknown")
        else:
            out["asn"], out["asn_org"] = "0", "unknown"
        out["net_class"] = out["asn_org"].map(classify_org)
        if self.tor_exits:
            out.loc[out["ip"].isin(self.tor_exits), "net_class"] = "tor"
        out["anon_score"] = out["net_class"].map(ANON_SCORE)
        m = out.set_index("ip")
        res = m.reindex(ips.values)
        res.index = ips.index
        return res[["country", "asn", "asn_org", "net_class", "anon_score"]]


# ------------------------------------------------------------------ synthetic DB
COUNTRIES = ["US", "DE", "GB", "IN", "BR", "RU", "CN", "NG", "FR", "NL", "UA", "ID", "VN", "TR", "CA", "JP", "SG", "RO"]
TZ = {"US": -6, "DE": 1, "GB": 0, "IN": 5.5, "BR": -3, "RU": 3, "CN": 8, "NG": 1, "FR": 1, "NL": 1,
      "UA": 2, "ID": 7, "VN": 7, "TR": 3, "CA": -5, "JP": 9, "SG": 8, "RO": 2}
_W1 = ["Blue", "Iron", "Nimbus", "Quartz", "Vertex", "Onyx", "Cobalt", "Argon", "Delta", "Helix", "Zenith", "Northwind"]
_W2 = ["Hosting", "Cloud", "Datacenter", "VPS Servers", "Colo"]


class SyntheticGeo:
    """Builds a self-consistent fake registry: private-use ASNs (64512+), /16 blocks."""

    def __init__(self, py_rng):
        self.py = py_rng
        self.rows: list[dict] = []
        asn, blk = 64512, 0

        def add(org, cc, cat):
            nonlocal asn, blk
            self.rows.append(dict(asn=asn, org=org, cc=cc, cat=cat,
                                  prefix=(20 + blk // 200, blk % 200)))
            asn += 1
            blk += 1

        for cc in COUNTRIES:
            for k in range(2):
                add(f"{cc} Broadband Telecom {k + 1}", cc, "isp")
        for i in range(14):
            add(f"{_W1[i % 12]} {_W2[i % 5]} Ltd", self.py.choice(["NL", "DE", "US", "SG", "RO", "GB"]), "hosting")
        for i in range(10):
            add(f"Relay Collective {i + 1} (Tor exit)", self.py.choice(["DE", "NL", "US", "FR", "RO"]), "tor")
        for i in range(8):
            add(f"{_W1[(i + 3) % 12]} VPN Services", self.py.choice(["NL", "SG", "US", "CA", "GB"]), "vpn")

    def pick_ip(self, cat: str, country: str | None = None) -> str:
        cand = [r for r in self.rows if r["cat"] == cat and (country is None or r["cc"] == country)]
        if not cand:
            cand = [r for r in self.rows if r["cat"] == cat]
        r = self.py.choice(cand)
        return f"{r['prefix'][0]}.{r['prefix'][1]}.{self.py.randint(0, 255)}.{self.py.randint(1, 254)}"

    def write(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "dbip-country-lite.csv", "w") as fc, open(out_dir / "dbip-asn-lite.csv", "w") as fa:
            for r in self.rows:
                a, b = r["prefix"]
                s, e = f"{a}.{b}.0.0", f"{a}.{b}.255.255"
                fc.write(f"{s},{e},{r['cc']}\n")
                fa.write(f"{s},{e},{r['asn']},{r['org']}\n")
