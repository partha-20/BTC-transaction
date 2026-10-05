# Technical write-up: AI-powered monitoring & analysis of Bitcoin transaction traffic

## 1. Problem and approach
Bitcoin's pseudonymity lets illicit actors (ransomware, darknet markets, scam rings) layer and cash out funds. The system
ingests **network-layer** sightings (timestamp, src/dst IP:port) together with **blockchain-layer** data (TXID, input/output
addresses and amounts, fee, script type), correlates them, and outputs a **ranked, explainable list of investigative leads**.
Everything is offline (Linux, CPU, scikit-learn/networkx/scipy/SQLite); the only external artefact is an optional open GeoIP
database (DB-IP Lite, CC-BY) - a synthetic one in the same format ships with the demo.

Pipeline: `ingest -> ledger/spend graph -> [CoinJoin model] -> CIOH clusters -> [entity-link model] -> entity graph ->
[laundering-chain model] -> entity features + seed propagation -> [risk model] + [anomaly models] -> alerts + Shapley -> dashboard`.

## 2. Data
No real data is used. A simulator maintains a real UTXO ledger (amounts always balance) for retail users, merchants,
exchanges (batched withdrawals, deposit sweeps), 3-9 ransomware operators, darknet markets with vendors, layering rings
(fan-out -> hop chains -> fan-in) and a Whirlpool-style CoinJoin coordinator. Each tx is seen 1-5 times by 3 sensors; the
first sighting comes from the true origin node ~55% of the time, otherwise from a random relay - the central
network-layer signal. Output is split over CSV/JSONL/XML with ~0.3% injected corruption. **Training and analysis datasets are
generated from different seeds** (train has more illicit actors); ground truth is used only for training and evaluation.

## 3. Models (all AI/ML, none rule-only)
| Task | Model | Key features |
|---|---|---|
| CoinJoin/mixing detection | HistGradientBoosting (tx level) | equal-output share, in/out shape, output entropy, script uniformity. Gates CIOH so mixers don't merge unrelated owners (ungated CIOH creates impure clusters). |
| Entity clustering | CIOH + HistGB **same-owner link model** on candidate cluster pairs, 2 passes | change-output tells (freshness, script match, round amounts, dwell), **IP evidence** (Poisson-significance of a relay IP's sighting count, cluster loyalty, "spender used same IP"), spectral graph-embedding cosine, activity-hour histogram similarity, fee-habit similarity |
| Peeling / hop chains | HistGB on chains extracted from the reconstructed UTXO graph | retained-value ratio, hop-gap rhythm, fee discipline, fresh-change share, IP concentration, destination hub-ness |
| Risk scoring | seed propagation (haircut taint, hop distance fwd/bwd, personalised PageRank fwd/bwd) **fused** with behaviour + network features in HistGB | 36 entity features |
| Anomaly detection | Isolation Forest (unsupervised) on entity behaviour and on transactions | |

Spend edges are reconstructed by matching (address, amount) to the earliest unspent earlier output, because the challenge
schema carries no prev-outpoint.

## 4. Explainability
Local **permutation-sampling Shapley values** (own implementation, model-agnostic; additive by construction) are computed for every
alert on its own model (risk / chain / anomaly). Each alert lists the top positive factors with the value, its population
percentile and its contribution; plus concrete evidence: shortest **path to/from a known-bad seed**, originating IPs with
country/ASN/class, key TXIDs, involved chains and a link-analysis neighbourhood graph containing IP nodes.
`score` is a ranking score in [0,1], not a calibrated probability - precision by score band is reported instead.
`priority = score * (0.6 + 0.4 * impact)` with impact from BTC involved; anomaly leads are halved so they never outrank risk leads.

## 5. Results (held-out synthetic analysis set, 15.3k txs, 29.9k addresses, 1,129 true entities)
* Ingestion: 88/88 corrupted rows rejected, 13/13 duplicates dropped.
* CoinJoin detector: AUC 1.0, P = R = 1.0 (120 rounds); payroll/promo/exchange batches correctly rejected.
* Entity clustering (pairwise): gated CIOH P 1.00 / R 0.50; **+ML link & IP evidence P 0.958 / R 0.690 (F1 0.802)**.
  For illicit actors, which avoid multi-input spends, recall rises **0.062 -> 0.275 at precision 1.0**.
* Laundering chains: AUC 1.0 (length-only baseline 0.92, AP 0.33); still 0.9999 without fee/timing/IP features;
  87.6% of true laundering txs recovered, 0 false flags.
* Risk: fused AUC 0.9998 / AP 0.991 vs taint-only AP 0.62, PageRank-only 0.66, hop-distance-only 0.76; behaviour-only 0.950;
  propagation-only 0.864; without network features 0.978.
* Alerts: 37 ENTITY_RISK leads, recall of illicit entities 88.1% (darknet_market 6/8, layerer 4/4, ransom_op 5/5, vendor 22/25),
  precision 1.0 at @10/@25/@50 and in all score bands.

*Numbers vary by about +-0.02 between runs (unseeded set ordering in a few places); the figures above are from the shipped run.*

## 6. Limitations (please read)
* **The data are synthetic and the simulator's illicit actors are comparatively easy** (structured CoinJoins, stable fee
  habits, non-diurnal laundering timing). Near-perfect scores mean the pipeline works end-to-end and beat baselines, **not**
  that real-world accuracy would be similar. The chain ablation shows structure alone is enough here; real evasive actors
  (randomised fees, human-like timing, irregular peels, Taproot/PayJoin) need re-training on real labelled data.
* IP evidence is weaker than the blockchain heuristics: relay noise and NAT/shared IPs cap its precision, and addresses that only
  ever receive carry no IP evidence. It adds ~ +0.013 AP to risk scoring and a little to link AUC in this setup.
* Sensor coverage (55% origin visibility) is an assumption; Tor/VPN hides the true origin, which itself is a feature.
* Seeds are a hand-off from open-source intelligence in practice; here they are a synthetic sample of victim/LEA reports.
* Alerts are investigative **leads**, not attribution: exchange-like hubs can look anomalous and are labelled as such.
* Entity/ownership inference of real people has privacy and legal implications; use only with lawful authority.

## 7. Reproduce
`./run_demo.sh` (about 2 minutes on one CPU core) -> `reports/run/dashboard.html`, `reports/metrics.json`, `reports/run/alerts.csv`.
Real data: put CSV/JSON/XML in `<dir>/raw`, GeoIP CSVs in `<dir>/geoip` (`scripts/fetch_geoip.sh`), optional `raw/seeds.csv`
(`address,label,source`), then `python -m btcintel run --data <dir>`.
