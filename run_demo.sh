#!/usr/bin/env bash
# One-command end-to-end demo (Linux, Python>=3.10, no network needed):
#   generate labelled training data + unlabelled analysis data -> train -> evaluate -> analyse -> dashboard
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt 2>/dev/null || true
PYTHONPATH=. python3 -m btcintel demo
echo; echo "Open reports/run/dashboard.html in a browser.  Metrics: reports/metrics.json"
