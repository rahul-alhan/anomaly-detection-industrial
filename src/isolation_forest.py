"""Isolation Forest wrapper with consistent score normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IFAnomalyModel:
    def __init__(self, contamination: float | str = "auto", random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols: list[str] | None = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "IFAnomalyModel":
        self.feature_cols = feature_cols
        X = self.scaler.fit_transform(df[feature_cols])
        self.model.fit(X)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.feature_cols, "Model not fit yet."
        X = self.scaler.transform(df[self.feature_cols])
        raw = -self.model.score_samples(X)             # higher = more anomalous
        # min-max normalize raw scores to [0, 1] using training-time-ish range
        lo, hi = float(np.percentile(raw, 5)), float(np.percentile(raw, 99))
        norm = np.clip((raw - lo) / max(1e-9, hi - lo), 0.0, 1.0)
        is_anom = self.model.predict(X) == -1
        return pd.DataFrame(
            {"if_score": norm, "if_is_anomaly": is_anom}, index=df.index
        )
