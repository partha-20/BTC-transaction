# Roadmap - phase by phase

| # | Phase | What is built | Exit criterion (verified) |
|---|-------|---------------|---------------------------|
| 0 | Scaffold | repo, shared helpers, requirements (numpy/pandas/scikit-learn/networkx/scipy only) | runs offline on Linux |
| 1 | Data & GeoIP | UTXO-consistent Bitcoin economy simulator (users, exchanges, merchants, ransomware, darknet markets, layering rings, CoinJoin); 3 network sensors; CSV+JSONL+XML output with injected dirt; ground truth; seed list; DB-IP-format GeoIP/ASN DB | amounts balance; every tx observed; train (seed 11) and analysis (seed 42) sets are disjoint |
| 2 | Ingestion | streaming parsers, strict validation, dedup, reject log, GeoIP enrichment, SQLite store | 88/88 injected bad rows rejected, 13/13 duplicates dropped |
| 3 | Graph & correlation | (address,amount) spend reconstruction, CIOH clustering, cluster flow graph, spectral embeddings, IP-to-cluster evidence | CIOH precision 1.000 once CoinJoins are excluded |
| 4 | ML detectors | 4a CoinJoin/mixing GB model; 4b entity-link GB model (blockchain + IP + graph + behaviour); 4c laundering-chain GB model; 4d risk propagation (taint, hop distance, PPR) + fused risk GB model; Isolation-Forest anomaly models | held-out metrics in reports/metrics.json |
| 5 | Explainable alerts | permutation-sampling Shapley, evidence (seed path, origin IPs, chains, txids), ego graphs, priority ranking | Shapley additivity residual ~ 0 |
| 6 | Dashboard | single-file offline HTML: ranked queue, link-analysis canvas with IP nodes, factor bars, evidence, pipeline report | all leads render (Node DOM-stub test) |
| 7 | Evaluation & docs | metrics vs ground truth with baselines and ablations, unit tests, write-up | `python -m btcintel demo` reproduces everything |
