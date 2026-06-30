"""Multivariate KPI anomaly detection for PulseBoard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from config.settings import Settings, get_settings


DEFAULT_KPI_COLUMNS = [
    "dau",
    "new_signups",
    "activation_rate",
    "churn_rate",
    "arpu",
    "mrr",
    "pipeline_created",
    "pipeline_won",
    "trial_to_paid_rate",
    "net_revenue_retention",
    "revenue_at_risk",
    "feature_a_adoption",
    "feature_b_adoption",
    "nps",
    "retention_d30",
]


@dataclass
class AnomalyDetector:
    """Isolation Forest detector for multivariate product KPI streams."""

    settings: Settings | None = None
    feature_columns: list[str] | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        self.feature_columns = self.feature_columns or DEFAULT_KPI_COLUMNS
        self.pipeline = Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                (
                    "model",
                    IsolationForest(
                        n_estimators=250,
                        contamination=self.settings.anomaly_contamination,
                        random_state=self.settings.random_seed,
                    ),
                ),
            ]
        )

    def fit(self, metrics: pd.DataFrame) -> "AnomalyDetector":
        """Fit the Isolation Forest detector to daily KPI data."""

        self.pipeline.fit(metrics[self.feature_columns])
        return self

    def predict(self, metrics: pd.DataFrame, sensitivity: float | None = None) -> pd.DataFrame:
        """Return KPI data with anomaly scores, flags, and dominant changed metric."""

        if sensitivity is None:
            sensitivity = self.settings.anomaly_contamination
        sensitivity = float(np.clip(sensitivity, 0.005, 0.2))

        scored = metrics.copy()
        features = scored[self.feature_columns]
        score = self.pipeline.decision_function(features)
        raw_flag = self.pipeline.predict(features) == -1
        quantile_cutoff = np.quantile(score, sensitivity)
        scored["anomaly_score"] = score
        scored["is_anomaly"] = raw_flag | (score <= quantile_cutoff)
        scored["anomaly_severity"] = np.clip((quantile_cutoff - score) / (abs(quantile_cutoff) + 1e-6), 0, 4)
        scored["primary_anomaly_metric"] = self._dominant_metric(scored)
        return scored

    def fit_predict(self, metrics: pd.DataFrame, sensitivity: float | None = None) -> pd.DataFrame:
        """Fit the detector and return anomaly-scored metrics."""

        return self.fit(metrics).predict(metrics, sensitivity=sensitivity)

    def anomaly_records(self, scored_metrics: pd.DataFrame, limit: int = 20) -> list[dict[str, object]]:
        """Convert flagged anomalies into LLM-ready record dictionaries."""

        anomalies = scored_metrics.loc[scored_metrics["is_anomaly"]].copy()
        anomalies["_has_business_event"] = anomalies["injected_anomaly"].fillna("").astype(str).str.len() > 0
        anomalies = anomalies.sort_values(["_has_business_event", "anomaly_score"], ascending=[False, True]).head(limit)
        records: list[dict[str, object]] = []
        for _, row in anomalies.iterrows():
            metric = str(row["primary_anomaly_metric"])
            records.append(
                {
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "metric": metric,
                    "score": float(row["anomaly_score"]),
                    "severity": float(row["anomaly_severity"]),
                    "value": float(row[metric]) if metric in row else None,
                    "context": {
                        "dau": int(row["dau"]),
                        "mrr": round(float(row["mrr"]), 2),
                        "churn_rate": round(float(row["churn_rate"]), 4),
                        "trial_to_paid_rate": round(float(row.get("trial_to_paid_rate", 0.0)), 4),
                        "net_revenue_retention": round(float(row.get("net_revenue_retention", 1.0)), 4),
                        "pipeline_created": round(float(row.get("pipeline_created", 0.0)), 2),
                        "nps": round(float(row["nps"]), 1),
                        "injected_label": str(row.get("injected_anomaly", "")),
                    },
                }
            )
        return records

    def _dominant_metric(self, scored: pd.DataFrame) -> pd.Series:
        rolling = scored[self.feature_columns].rolling(14, min_periods=5).median().bfill()
        mad = (scored[self.feature_columns] - rolling).abs().rolling(30, min_periods=8).median().bfill()
        z_like = ((scored[self.feature_columns] - rolling).abs() / (mad + 1e-6)).replace([np.inf, -np.inf], 0)
        return z_like.idxmax(axis=1)
