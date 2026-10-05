"""Render the self-contained (offline, no CDN) dashboard: data is embedded in one HTML file."""
from __future__ import annotations
import json
from pathlib import Path


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


def render(data: dict, out_path):
    tpl = (Path(__file__).parent / "dashboard_template.html").read_text()
    blob = json.dumps(data, default=_default).replace("</", "<\\/")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(tpl.replace("__DATA__", blob))
    return out_path
