"""Shared constants and tiny helpers used by every phase."""
from __future__ import annotations
import ipaddress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAT = 100_000_000          # satoshi per BTC
DUST = 546                 # sat; outputs below this are not created by wallets

# keywords used to flag "anonymising / infrastructure" ASNs when the GeoIP database
# carries only an organisation name (true for DB-IP / GeoLite ASN files)
ANON_KEYWORDS = ("tor", "vpn", "proxy", "hosting", "cloud", "datacenter", "datacentre", "vps", "server")
TOR_VPN_KEYWORDS = ("tor", "vpn", "proxy")


def script_of(addr: str) -> str:
    """Infer the output script type from an address prefix (mainnet)."""
    if addr.startswith("bc1p"):
        return "p2tr"
    if addr.startswith("bc1"):
        return "p2wpkh"
    if addr.startswith("3"):
        return "p2sh"
    return "p2pkh"


def ip_to_int(ip: str) -> int:
    return int(ipaddress.IPv4Address(ip))


def is_valid_ipv4(ip) -> bool:
    try:
        ipaddress.IPv4Address(str(ip))
        return True
    except ValueError:
        return False
