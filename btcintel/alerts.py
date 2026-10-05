"""Phase 5 - ranked, explainable investigative leads.

Alert types
  ENTITY_RISK      entity classified illicit by the fused risk model (propagation + behaviour + network)
  LAUNDER_CHAIN    peeling / hop chain classified by the chain model
  ENTITY_ANOMALY   statistically unusual entity (Isolation Forest) not already covered by a risk alert
  TX_ANOMALY       statistically unusual single transaction
  SEED_EXPANSION   known-bad (seed) wallet whose ownership cluster grew via CoinJoin-safe CIOH + ML linking

score     = model output in [0,1] (ranking score, not a calibrated probability; see evaluation for precision by band)
priority  = score * (0.6 + 0.4 * impact),  impact = clip(log10(1 + BTC involved) / 2.5, 0, 1)
"""
from __future__ import annotations

from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .chains import CHAIN_FEATURES
from .common import SAT
from .entities import BEHAVIOUR_FEATURES, ENTITY_FEATURES
from .explain import shapley, top_reasons
from .features import TX_FEATURES
from .models import AnomalyModel
from .pipeline import State


def _iso(t):
    return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _impact(btc):
    return float(np.clip(np.log10(1 + btc) / 2.5, 0, 1))


def _priority(score, btc):
    return float(score * (0.6 + 0.4 * _impact(btc)))


def _headline(x: pd.Series, reasons):
    tags = []
    if x.get("peel_frac", 0) > 0.4: tags.append("peeling-chain laundering")
    if x.get("hop_frac", 0) > 0.3: tags.append("rapid relay hops")
    if x.get("equal_fanout_n", 0) > 0: tags.append("equal-value fan-out layering")
    if x.get("cj_btc_frac", 0) > 0.2: tags.append("CoinJoin mixing")
    if x.get("dist_fwd", 9) <= 2 and x.get("taint_in_frac", 0) > 0.02: tags.append("funds traced from known-illicit wallets")
    if x.get("tor_vpn_share", 0) > 0.4: tags.append("Tor/VPN-originated broadcasts")
    if x.get("out_to_hub_frac", 0) > 0.3: tags.append("cash-out to exchange-like hubs")
    if x.get("fan_in_max", 0) >= 20 and x.get("log_cp_in", 0) > 3: tags.append("collection of many small payments")
    if not tags and reasons:
        tags.append(reasons[0]["label"])
    return "; ".join(tags[:3]).capitalize()


def build_alerts(st: State, models: dict, max_entity=150, n_tx_anom=40, n_entity_anom=30, n_perm=40):
    L, X, tx = st.L, st.X, st.L.tx
    n = len(X)
    meta = models["meta"]
    risk_bg = meta["train_background"].values
    chain_bg = meta["chain_background"].values
    med = X[ENTITY_FEATURES].median()
    Xf = X[ENTITY_FEATURES].fillna(med)

    cand = X["active"] & ~X["is_seed"]
    risk = pd.Series(0.0, index=X.index)
    risk[cand] = models["risk"].predict_proba(Xf[cand])
    risk[X["is_seed"]] = 1.0

    # anomaly model (unsupervised, fitted on this dataset's active entities)
    Xa = X.loc[X["active"], BEHAVIOUR_FEATURES].fillna(X[BEHAVIOUR_FEATURES].median())
    am = AnomalyModel().fit(Xa)
    anom = pd.Series(0.0, index=X.index)
    anom[Xa.index] = am.score(Xa)

    # ---- shared lookups
    ip_info = tx.drop_duplicates("first_src_ip").set_index("first_src_ip")[["first_country", "first_asn_org", "first_net_class"]]
    G = nx.DiGraph()
    G.add_weighted_edges_from(zip(st.edges["src"], st.edges["dst"], st.edges["sat"]))
    seeds_e = st.aux["seeds_e"]
    Gf = G.copy(); Gr = G.reverse(copy=True)
    for s in seeds_e:
        Gf.add_edge(-1, int(s)); Gr.add_edge(-1, int(s))
    Sg = st.aux["Sg"]
    by_e = Sg.groupby("e")
    ent_addrs = pd.Series(np.arange(L.n_addr)).groupby(st.ent).apply(list)
    chain_of_tx = {}
    for k, path in enumerate(st.chains):
        if st.chain_p[k] >= 0.5:
            for t in path:
                chain_of_tx.setdefault(t, k)
    ent_chains = {}
    tid_first_in = L.txin.groupby("tid")["aid"].first()
    for k, path in enumerate(st.chains):
        if st.chain_p[k] >= 0.5:
            ent_chains.setdefault(int(st.ent[tid_first_in[path[0]]]), []).append(k)

    def seed_path(e):
        for G_, name, dist in ((Gf, "downstream", X.at[e, "dist_fwd"]), (Gr, "upstream", X.at[e, "dist_bwd"])):
            if dist <= 5 and e in G_:
                try:
                    p = nx.shortest_path(G_, -1, int(e))[1:]
                except nx.NetworkXNoPath:
                    continue
                hops = []
                for a, b in zip(p[:-1], p[1:]):
                    d = G[a][b] if name == "downstream" else G[b][a]
                    hops.append(dict(src=int(a if name == "downstream" else b), dst=int(b if name == "downstream" else a),
                                     btc=round(d["weight"] / SAT, 4)))
                return dict(direction=name, path=[int(v) for v in p], hops=hops)
        return None

    def ego(e, k=10):
        nodes = {int(e): dict(id=f"E{e}", kind="focus")}
        edges = []
        es = st.edges
        for role, df in (("out", es[es["src"] == e].nlargest(k, "sat")), ("in", es[es["dst"] == e].nlargest(k, "sat"))):
            for _, r in df.iterrows():
                o = int(r["dst"] if role == "out" else r["src"])
                nodes.setdefault(o, dict(id=f"E{o}", kind="other"))
                edges.append(dict(source=f"E{e}" if role == "out" else f"E{o}", target=f"E{o}" if role == "out" else f"E{e}",
                                  btc=round(r["sat"] / SAT, 4), ntx=int(r["ntx"])))
        out = []
        for o, nd in nodes.items():
            nd.update(risk=round(float(risk.get(o, 0)), 3), seed=bool(X.at[o, "is_seed"]), n_addr=int(np.expm1(X.at[o, "log_addr"]).round()),
                      hub=bool(X.at[o, "log_cp_in"] > np.log1p(30)))
            out.append(nd)
        if e in by_e.groups:
            for ip, cnt in by_e.get_group(e)["ip"].value_counts().head(2).items():
                inf = ip_info.loc[ip]
                out.append(dict(id=f"ip:{ip}", kind="ip", label=ip, country=inf["first_country"], net=inf["first_net_class"], risk=0, seed=False, n_addr=0))
                edges.append(dict(source=f"ip:{ip}", target=f"E{e}", btc=0, ntx=int(cnt), ip=True))
        return dict(nodes=out, links=edges)

    def entity_evidence(e):
        g = by_e.get_group(e) if e in by_e.groups else Sg.iloc[0:0]
        ips = []
        for ip, cnt in g["ip"].value_counts().head(3).items():
            inf = ip_info.loc[ip]
            ips.append(dict(ip=ip, txs=int(cnt), country=inf["first_country"], org=inf["first_asn_org"], net_class=inf["first_net_class"]))
        top = g.nlargest(5, "sat")
        ev = dict(addresses=len(ent_addrs[e]), sample_addresses=[L.addrs[a] for a in ent_addrs[e][:5]],
                  seed_addresses=int(st.seed_addr[ent_addrs[e]].sum()),
                  first_active=_iso(X.at[e, "first_active"]) if X.at[e, "n_sent"] else None,
                  last_active=_iso(X.at[e, "last_active"]) if X.at[e, "n_sent"] else None,
                  btc_in=round(float(X.at[e, "in_btc"]), 4), btc_out=round(float(X.at[e, "out_btc"]), 4),
                  sent_txs=int(X.at[e, "n_sent"]), recv_txs=int(X.at[e, "n_recv"]),
                  origin_ips=ips, key_txids=[tx.at[t, "txid"] for t in top["tid"]],
                  seed_path=seed_path(e), chains=[f"C{k:04d}" for k in ent_chains.get(e, [])[:5]])
        return ev

    alerts = []

    # ------------------------------------------------ ENTITY_RISK
    rk = risk[cand & (risk >= 0.5)].sort_values(ascending=False).head(max_entity)
    rfn = models["risk"].predict_fn()
    names = list(ENTITY_FEATURES)
    pop = Xf[cand].values
    for e, s in rk.items():
        x = Xf.loc[e].values.astype(float)
        phi, base, fx = shapley(rfn, x, risk_bg, n_perm=n_perm, seed=int(e))
        reasons = top_reasons(names, x, phi, pop, k=5)
        btc = max(X.at[e, "in_btc"], X.at[e, "out_btc"])
        alerts.append(dict(type="ENTITY_RISK", subject=f"E{e}", subject_id=int(e), score=float(s), priority=_priority(s, btc),
                           headline=_headline(X.loc[e], reasons), btc=round(float(btc), 3), reasons=reasons,
                           explain=dict(base=base, prediction=fx, residual=float(fx - base - phi.sum())),
                           evidence=entity_evidence(e), graph=ego(e), anomaly=round(float(anom[e]), 3)))

    # ------------------------------------------------ LAUNDER_CHAIN
    pfn = models["peel"].predict_fn()
    cn = list(CHAIN_FEATURES)
    cX = st.chain_X
    for k in np.flatnonzero(st.chain_p >= 0.5):
        path = st.chains[k]
        x = cX.iloc[k][cn].values.astype(float)
        phi, base, fx = shapley(pfn, x, chain_bg, n_perm=n_perm, seed=int(k))
        reasons = top_reasons(cn, x, phi, chain_bg, k=4)
        e0 = int(st.ent[tid_first_in[path[0]]])
        start_btc = float(tx.at[path[0], "in_sat"] / SAT)
        kind = "peeling chain" if cX.at[k, "frac_two_out"] > 0.5 else "relay-hop chain"
        alerts.append(dict(type="LAUNDER_CHAIN", subject=f"C{k:04d}", subject_id=int(k), score=float(st.chain_p[k]),
                           priority=_priority(st.chain_p[k], start_btc),
                           headline=f"{kind.capitalize()} of {len(path)} hops starting at entity E{e0}", btc=round(start_btc, 3),
                           reasons=reasons, explain=dict(base=base, prediction=fx, residual=float(fx - base - phi.sum())),
                           evidence=dict(entity=f"E{e0}", hops=len(path), start_btc=round(start_btc, 4),
                                         first_seen=_iso(tx.at[path[0], "first_seen"]), last_seen=_iso(tx.at[path[-1], "first_seen"]),
                                         txids=[tx.at[t, "txid"] for t in path[:40]],
                                         origin_ips=entity_evidence(e0)["origin_ips"]),
                           graph=ego(e0), anomaly=0.0))

    # ------------------------------------------------ ENTITY_ANOMALY (behavioural outliers not already flagged)
    an = anom[cand & (risk < 0.5) & (anom > 0.9)].sort_values(ascending=False).head(n_entity_anom)
    afn, bnames = am.predict_fn(), list(BEHAVIOUR_FEATURES)
    abg = Xa.sample(min(200, len(Xa)), random_state=0).values
    for e, s in an.items():
        x = Xa.loc[e].values.astype(float)
        phi, base, fx = shapley(afn, x, abg, n_perm=n_perm, seed=int(e))
        reasons = top_reasons(bnames, x, phi, Xa.values, k=4)
        btc = max(X.at[e, "in_btc"], X.at[e, "out_btc"])
        alerts.append(dict(type="ENTITY_ANOMALY", subject=f"E{e}", subject_id=int(e), score=float(s) * 0.8,
                           priority=_priority(float(s) * 0.8, btc) * 0.5,
                           headline="Behavioural outlier: " + (reasons[0]["label"] if reasons else "unusual profile")
                           + (" (exchange-like hub, likely benign)" if X.at[e, "log_cp_in"] > np.log1p(30) else ""),
                           btc=round(float(btc), 3), reasons=reasons, explain=dict(base=base, prediction=fx, residual=float(fx - base - phi.sum())),
                           evidence=entity_evidence(e), graph=ego(e), anomaly=round(float(s), 3)))

    # ------------------------------------------------ TX_ANOMALY
    F = st.F
    T = F[TX_FEATURES].copy()
    hour = ((tx["first_seen"].values // 3600) % 24)
    T["fee_dev"] = np.log1p(F["fee_rate"]) - pd.Series(np.log1p(F["fee_rate"].values)).rolling(101, center=True, min_periods=10).median().values
    T["hour_sin"], T["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    T = T.fillna(0.0)
    tam = AnomalyModel(n_estimators=200, contamination=0.01).fit(T)
    ts = tam.score(T)
    tn = list(T.columns)
    tbg = T.sample(200, random_state=0).values
    for t in np.argsort(-ts)[:n_tx_anom]:
        if st.coinjoin[t]:
            continue
        x = T.iloc[t].values.astype(float)
        phi, base, fx = shapley(tam.predict_fn(), x, tbg, n_perm=24, seed=int(t))
        reasons = top_reasons(tn, x, phi, T.values, k=3)
        btc = tx.at[t, "out_sat"] / SAT
        alerts.append(dict(type="TX_ANOMALY", subject=tx.at[t, "txid"][:16] + "…", subject_id=int(t), score=float(ts[t]) * 0.7,
                           priority=_priority(float(ts[t]) * 0.7, btc) * 0.5,
                           headline=f"Unusual transaction ({int(tx.at[t,'n_in'])} in / {int(tx.at[t,'n_out'])} out, {btc:.3f} BTC)",
                           btc=round(float(btc), 3), reasons=reasons, explain=dict(base=base, prediction=fx, residual=float(fx - base - phi.sum())),
                           evidence=dict(txid=tx.at[t, "txid"], first_seen=_iso(tx.at[t, "first_seen"]), first_relay_ip=tx.at[t, "first_src_ip"],
                                         country=tx.at[t, "first_country"], net_class=tx.at[t, "first_net_class"], fee_rate=round(float(tx.at[t, "fee_rate"]), 2)),
                           graph=None, anomaly=round(float(ts[t]), 3)))

    # ------------------------------------------------ SEED_EXPANSION
    for e in X.index[X["is_seed"]]:
        na, ns = len(ent_addrs[e]), int(st.seed_addr[ent_addrs[e]].sum())
        if na > ns:
            btc = max(X.at[e, "in_btc"], X.at[e, "out_btc"])
            alerts.append(dict(type="SEED_EXPANSION", subject=f"E{e}", subject_id=int(e), score=1.0, priority=_priority(1.0, btc) * 0.5,
                               headline=f"Known-bad wallet expanded: {ns} listed -> {na} attributed addresses", btc=round(float(btc), 3),
                               reasons=[], explain=None, evidence=entity_evidence(e), graph=ego(e), anomaly=round(float(anom[e]), 3)))

    alerts.sort(key=lambda a: -a["priority"])
    for i, a in enumerate(alerts, 1):
        a["rank"] = i
        a["id"] = f"A{i:04d}"
    return alerts, dict(risk=risk, anomaly=anom)
