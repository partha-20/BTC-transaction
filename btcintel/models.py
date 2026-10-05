"""Phase 4 - ML models.  All are scikit-learn (CPU, offline) and serialisable with joblib.

  MixingModel      tx-level  gradient boosting: CoinJoin / mixer detection      (supervised)
  LinkModel        pair-level gradient boosting: same-owner probability          (supervised, entity resolution)
  PeelModel        chain-level gradient boosting: peeling-chain probability      (supervised)
  AnomalyModel     Isolation Forest on entity features                           (unsupervised)
  RiskModel        gradient boosting: illicit-entity probability from propagated risk + behaviour
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest


class _GB:
    features: list[str] = []

    def __init__(self, **kw):
        p = dict(max_iter=200, learning_rate=0.08, max_depth=5, l2_regularization=1.0, random_state=0,
                 class_weight="balanced")
        p.update(kw)
        self.clf = HistGradientBoostingClassifier(**p)

    def fit(self, X: pd.DataFrame, y):
        self.features = list(X.columns)
        self.clf.fit(X.values.astype(float), np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.features].values.astype(float))[:, 1]

    def predict_fn(self):
        f = self.features
        return lambda M: self.clf.predict_proba(M)[:, 1]


class MixingModel(_GB):
    pass


class LinkModel(_GB):
    pass


class PeelModel(_GB):
    pass


class RiskModel(_GB):
    def __init__(self, **kw):
        super().__init__(max_iter=250, learning_rate=0.06, max_depth=4, **kw)


class AnomalyModel:
    def __init__(self, n_estimators=300, contamination=0.02, seed=0):
        self.iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=seed)
        self.features: list[str] = []
        self._lo = self._hi = None

    def fit(self, X: pd.DataFrame):
        self.features = list(X.columns)
        self.iso.fit(X.values.astype(float))
        s = -self.iso.score_samples(X.values.astype(float))
        self._lo, self._hi = np.percentile(s, [50, 99.9])
        return self

    def raw(self, M: np.ndarray) -> np.ndarray:
        return -self.iso.score_samples(M)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """0..1 anomaly score (median->0, 99.9th percentile of training data->1)."""
        s = self.raw(X[self.features].values.astype(float))
        return np.clip((s - self._lo) / (self._hi - self._lo + 1e-9), 0, 1)

    def predict_fn(self):
        return lambda M: np.clip((self.raw(M) - self._lo) / (self._hi - self._lo + 1e-9), 0, 1)
