#!/usr/bin/env bash
# Download the open-source DB-IP "Lite" country + ASN databases (CC-BY-4.0, https://db-ip.com) - run ONCE while online.
# The files are then used fully offline. Same CSV format as the synthetic DB shipped with the demo data.
# Usage: scripts/fetch_geoip.sh data/mydataset/geoip
set -euo pipefail
OUT="${1:?usage: fetch_geoip.sh <dataset>/geoip}"; mkdir -p "$OUT"; YM="$(date +%Y-%m)"
for kind in country asn; do
  curl -fSL "https://download.db-ip.com/free/dbip-${kind}-lite-${YM}.csv.gz" -o "$OUT/dbip-${kind}-lite-${YM}.csv.gz" \
    || echo "could not fetch ${kind} DB for ${YM}; try the previous month or download manually from https://db-ip.com/db/"
done
# optional: Tor exit list (one IP per line) improves the 'tor' class
curl -fsSL "https://check.torproject.org/torbulkexitlist" -o "$OUT/tor_exit_nodes.txt" || true
echo "GeoIP files in $OUT (attribution: IP Geolocation by DB-IP)"
