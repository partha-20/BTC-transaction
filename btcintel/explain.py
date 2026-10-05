"""Model-agnostic local explanations: permutation-sampling Shapley values (no external deps).

phi_i = E over random feature orderings & background rows of  f(x with features up to i taken from x) - f(... before i).
Exact-in-expectation, additive:  f(x) = base + sum(phi)  (residual reported).  Works for any predict function
(gradient boosting risk/peel models and the Isolation-Forest anomaly score alike).
"""
from __future__ import annotations
import numpy as np

TEXT = {   # feature -> (label, formatter)
    "peel_frac": ("share of its outgoing transactions that sit in peeling/hop chains", lambda v: f"{v:.0%}"),
    "hop_frac": ("share of 1-in/1-out relay hops", lambda v: f"{v:.0%}"),
    "cj_frac": ("share of its spends made through CoinJoin rounds", lambda v: f"{v:.0%}"),
    "cj_btc_frac": ("share of sent value routed through CoinJoin", lambda v: f"{v:.0%}"),
    "taint_in_frac": ("share of received value tainted by known-illicit seed wallets", lambda v: f"{v:.1%}"),
    "taint_spent_frac": ("share of spent value that is seed-tainted", lambda v: f"{v:.1%}"),
    "log_taint_in_btc": ("tainted BTC received (log)", lambda v: f"{np.expm1(v):.2f} BTC"),
    "dist_fwd": ("hops downstream of a known-illicit wallet", lambda v: f"{v:.0f}"),
    "dist_bwd": ("hops upstream of a known-illicit wallet", lambda v: f"{v:.0f}"),
    "ppr_fwd": ("PageRank proximity to seeds (downstream)", lambda v: f"{v:.1f} log10"),
    "ppr_bwd": ("PageRank proximity to seeds (upstream)", lambda v: f"{v:.1f} log10"),
    "fee_cv": ("fee-rate variability (low = scripted fee habit)", lambda v: f"{v:.2f}"),
    "fee_mean": ("mean fee rate", lambda v: f"{v:.1f} (log1p sat/vB)"),
    "anon_mean": ("mean anonymity score of originating IPs (Tor/VPN/hosting)", lambda v: f"{v:.2f}"),
    "tor_vpn_share": ("share of txs first relayed from Tor/VPN IPs", lambda v: f"{v:.0%}"),
    "ip_top_share": ("concentration of txs on one originating IP", lambda v: f"{v:.0%}"),
    "hour_entropy": ("spread of activity over the 24h clock", lambda v: f"{v:.2f}"),
    "night_share": ("share of activity between 22:00-06:00 UTC", lambda v: f"{v:.0%}"),
    "pass_ratio": ("pass-through ratio (in ~ out)", lambda v: f"{v:.2f}"),
    "dwell_med_h": ("median hours funds rest before being spent", lambda v: f"{v:.1f} h"),
    "frac_fast": ("share of received outputs spent within 6 h", lambda v: f"{v:.0%}"),
    "fan_out_max": ("largest fan-out in a single tx", lambda v: f"{v:.0f} outputs"),
    "fan_in_max": ("largest consolidation in a single tx", lambda v: f"{v:.0f} inputs"),
    "equal_fanout_n": ("near-equal-value fan-out (layering) txs", lambda v: f"{v:.0f}"),
    "out_to_hub_frac": ("share of outgoing value sent to exchange-like hubs (cash-out)", lambda v: f"{v:.0%}"),
    "log_addr": ("addresses controlled (log)", lambda v: f"{np.expm1(v):.0f}"),
    "log_sent": ("outgoing transactions (log)", lambda v: f"{np.expm1(v):.0f}"),
    "log_recv": ("incoming transactions (log)", lambda v: f"{np.expm1(v):.0f}"),
    "log_out_btc": ("value sent (log)", lambda v: f"{np.expm1(v):.2f} BTC"),
    "log_in_btc": ("value received (log)", lambda v: f"{np.expm1(v):.2f} BTC"),
    "log_cp_out": ("distinct counterparties paid (log)", lambda v: f"{np.expm1(v):.0f}"),
    "log_cp_in": ("distinct counterparties paying it (log)", lambda v: f"{np.expm1(v):.0f}"),
    "addr_reuse": ("address reuse rate", lambda v: f"{v:.0%}"),
    "p2pkh_share": ("share of legacy P2PKH addresses", lambda v: f"{v:.0%}"),
    "log_avg_out_btc": ("average value per spend (log)", lambda v: f"{np.expm1(v):.2f} BTC"),
    "recv_burst": ("average value per receipt (log)", lambda v: f"{np.expm1(v):.2f} BTC"),
    # transaction features
    "n_in": ("number of inputs", lambda v: f"{v:.0f}"),
    "n_out": ("number of outputs", lambda v: f"{v:.0f}"),
    "log_value": ("transaction value (log10 BTC)", lambda v: f"{10**v:.3f} BTC"),
    "fee_rate": ("fee rate", lambda v: f"{v:.1f} sat/vB"),
    "fee_dev": ("fee rate vs. neighbouring transactions (log)", lambda v: f"{v:+.2f}"),
    "max_equal_out": ("outputs of identical value", lambda v: f"{v:.0f}"),
    "frac_equal_out": ("share of outputs with identical value", lambda v: f"{v:.0%}"),
    "equal_value_frac": ("share of value in identical outputs", lambda v: f"{v:.0%}"),
    "distinct_out_frac": ("distinct output values / outputs", lambda v: f"{v:.2f}"),
    "out_entropy": ("output-value entropy", lambda v: f"{v:.2f}"),
    "in_out_ratio": ("inputs per output", lambda v: f"{v:.2f}"),
    "distinct_in_frac": ("distinct input addresses / inputs", lambda v: f"{v:.2f}"),
    "out_script_share": ("share of outputs with the same script type", lambda v: f"{v:.0%}"),
    "modal_round": ("repeated output value is a round number", lambda v: "yes" if v > .5 else "no"),
    "out_cv": ("spread of output values (CV)", lambda v: f"{v:.2f}"),
    "noneq_minus_nin": ("non-identical outputs minus inputs", lambda v: f"{v:.0f}"),
    "anon_score": ("anonymity score of first relay IP", lambda v: f"{v:.2f}"),
    "n_obs": ("network sightings of the tx", lambda v: f"{v:.0f}"),
    "hour_sin": ("time of day (sin)", lambda v: f"{v:.2f}"),
    "hour_cos": ("time of day (cos)", lambda v: f"{v:.2f}"),
    # chain features
    "length": ("chain length (hops)", lambda v: f"{v:.0f}"),
    "ret_mean": ("mean share of value kept as change per hop", lambda v: f"{v:.0%}"),
    "ret_std": ("variation of the retained share", lambda v: f"{v:.3f}"),
    "gap_h_mean": ("mean hours between hops", lambda v: f"{v:.1f} h"),
    "gap_h_med": ("median hours between hops", lambda v: f"{v:.1f} h"),
    "gap_h_cv": ("timing irregularity between hops", lambda v: f"{v:.2f}"),
    "fresh_change": ("share of hops sending change to a brand-new address", lambda v: f"{v:.0%}"),
    "frac_two_out": ("share of hops that peel one payment off", lambda v: f"{v:.0%}"),
    "dest_hub_mean": ("hub-ness of peel destinations", lambda v: f"{v:.2f}"),
    "log_start_btc": ("starting value (log10 BTC)", lambda v: f"{v:.2f}"),
}


def shapley(fn, x: np.ndarray, background: np.ndarray, n_perm: int = 48, seed: int = 0):
    """Return (phi[d], base, fx).  fn maps (n,d) -> (n,) scores."""
    rng = np.random.default_rng(seed)
    d = len(x)
    bg = background[rng.integers(len(background), size=n_perm)]
    rows, orders = [], []
    for b in bg:
        cur, order = b.copy(), rng.permutation(d)
        rows.append(cur.copy())
        for j in order:
            cur[j] = x[j]
            rows.append(cur.copy())
        orders.append(order)
    pred = fn(np.array(rows)).reshape(n_perm, d + 1)
    phi = np.zeros(d)
    for k, order in enumerate(orders):
        np.add.at(phi, order, np.diff(pred[k]))
    phi /= n_perm
    return phi, float(pred[:, 0].mean()), float(pred[:, -1].mean())


def top_reasons(names, x, phi, pop: np.ndarray | None = None, k=5, positive_only=True):
    """Human-readable top contributions. pop = population matrix for percentile context."""
    idx = np.argsort(-phi if positive_only else -np.abs(phi))[:k]
    out = []
    for i in idx:
        if positive_only and phi[i] <= 0:
            continue
        lab, fmt = TEXT.get(names[i], (names[i], lambda v: f"{v:.3g}"))
        pct = float((pop[:, i] <= x[i]).mean() * 100) if pop is not None else None
        out.append(dict(feature=names[i], label=lab, value=float(x[i]), shown=fmt(x[i]), contribution=float(phi[i]),
                        percentile=None if pct is None else round(pct, 1)))
    return out
