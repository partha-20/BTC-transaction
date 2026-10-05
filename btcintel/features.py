"""Feature engineering: transaction-level (this file, part 1) and entity/chain-level (part 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import SAT, script_of
from .graphs import Ledger

TX_FEATURES = ["n_in", "n_out", "log_value", "fee_rate", "max_equal_out", "frac_equal_out", "equal_value_frac",
               "distinct_out_frac", "out_entropy", "in_out_ratio", "distinct_in_frac", "out_script_share",
               "modal_round", "out_cv", "noneq_minus_nin", "anon_score", "n_obs"]


def tx_features(L: Ledger) -> pd.DataFrame:
    """One row per tid; purely structural + first-seen network features (no labels, no clustering)."""
    tx, ti, to = L.tx, L.txin, L.txout
    f = pd.DataFrame({"tid": tx["tid"].values})
    f["n_in"], f["n_out"] = tx["n_in"].values, tx["n_out"].values
    f["log_value"] = np.log10(np.maximum(tx["out_sat"].values, 1) / SAT + 1e-9)
    f["fee_rate"] = tx["fee_rate"].values
    f["n_obs"] = tx["n_obs"].values
    f["anon_score"] = tx["first_anon_score"].values if "first_anon_score" in tx else 0.0

    vc = to.groupby(["tid", "sat"]).size().rename("k").reset_index()
    idx = vc.groupby("tid")["k"].idxmax()
    modal = vc.loc[idx].set_index("tid")
    f = f.merge(modal.rename(columns={"k": "max_equal_out", "sat": "modal_sat"}), left_on="tid", right_index=True, how="left")
    f["frac_equal_out"] = f["max_equal_out"] / f["n_out"]
    f["equal_value_frac"] = f["max_equal_out"] * f["modal_sat"] / np.maximum(tx["out_sat"].values, 1)
    f["modal_round"] = ((f["modal_sat"] % 10000) == 0).astype(float)
    f.loc[f["max_equal_out"] < 2, "modal_round"] = 0.0
    nd = to.groupby("tid")["sat"].nunique()
    f["distinct_out_frac"] = nd.reindex(f["tid"]).values / f["n_out"]
    f["in_out_ratio"] = f["n_in"] / f["n_out"]
    ndi = ti.groupby("tid")["aid"].nunique()
    f["distinct_in_frac"] = ndi.reindex(f["tid"]).values / f["n_in"]

    p = to["sat"] / to.groupby("tid")["sat"].transform("sum").clip(lower=1)
    h = -(p * np.log(p.clip(lower=1e-12))).groupby(to["tid"]).sum()
    f["out_entropy"] = (h.reindex(f["tid"]).values / np.log(np.maximum(f["n_out"], 2)))
    g = to.groupby("tid")["sat"]
    f["out_cv"] = (g.std(ddof=0) / g.mean().clip(lower=1)).reindex(f["tid"]).fillna(0).values
    sc = pd.Series([script_of(a) for a in L.addrs[to["aid"].values]], index=to.index)
    top = sc.groupby(to["tid"]).agg(lambda s: s.value_counts().iloc[0])
    f["out_script_share"] = top.reindex(f["tid"]).values / f["n_out"]
    f["noneq_minus_nin"] = (f["n_out"] - f["max_equal_out"]) - f["n_in"]
    return f.drop(columns=["modal_sat"]).fillna(0.0)
