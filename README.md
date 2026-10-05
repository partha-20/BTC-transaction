# btcintel - offline AI monitoring & analysis of Bitcoin transaction traffic

Ingests bulk network+blockchain metadata (CSV / JSON(L) / XML), correlates IP/timing observations with wallets and TXIDs,
and produces **ranked, explainable investigative leads** with an offline link-analysis dashboard.

```bash
./run_demo.sh                     # generate data -> train -> evaluate -> analyse -> dashboard   (Linux, Python>=3.10)
xdg-open reports/run/dashboard.html
```
Individual steps: `python -m btcintel {gen,train,run,evaluate,demo} --help`. Tests: `PYTHONPATH=. python tests/test_core.py`.

| Path | Purpose |
|---|---|
| `btcintel/datagen.py`, `geoip.py` | synthetic economy + sensors + ground truth; DB-IP-format GeoIP/ASN loader (real DB is drop-in) |
| `btcintel/ingest.py`, `store.py`, `prep.py` | streaming multi-format parse, validation, SQLite store |
| `btcintel/graphs.py`, `entity.py`, `chains.py` | ledger/spend graph, CIOH, embeddings, IP correlation, entity resolution, chain extraction |
| `btcintel/features.py`, `entities.py`, `models.py` | features, seed propagation, ML models |
| `btcintel/explain.py`, `alerts.py` | Shapley explanations, evidence, ranked leads |
| `btcintel/dashboard*.py|html` | self-contained offline dashboard |
| `docs/ROADMAP.md`, `docs/TECHNICAL_WRITEUP.md` | phase plan; approach, models, explainability, results, limitations |

Data are synthetic; see the limitations section of the write-up before drawing conclusions about real-world accuracy.
