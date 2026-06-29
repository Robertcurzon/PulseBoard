"""End-to-end ML pipeline for PulseBoard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import Settings, get_settings
from data.generators.synthetic_data import SyntheticDataset, generate_all
from ml.anomaly_detector import AnomalyDetector
from ml.explainer import ChurnExplainer, FeatureImportance
from ml.forecaster import ForecastResult, forecast_metrics


CHURN_FEATURE_COLUMNS = [
    "tenure_days",
    "sessions_30d",
    "feature_a_events",
    "feature_b_events",
    "support_tickets",
    "days_since_last_login",
    "seats",
    "plan_value",
    "nps_response",
    "onboarding_complete",
]


@dataclass
class PipelineResult:
    """Structured output from the PulseBoard ML pipeline."""

    datasets: SyntheticDataset
    scored_metrics: pd.DataFrame
    forecasts: dict[str, ForecastResult]
    churn_model: Pipeline
    churn_auc: float
    feature_importance: list[FeatureImportance]


def build_churn_model(settings: Settings | None = None) -> Pipeline:
    """Build the sklearn churn prediction pipeline."""

    settings = settings or get_settings()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                CHURN_FEATURE_COLUMNS,
            )
        ]
    )
    classifier = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=settings.random_seed)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_churn_model(churn_training: pd.DataFrame, settings: Settings | None = None) -> tuple[Pipeline, float, pd.DataFrame]:
    """Train and evaluate the churn model, returning model, ROC AUC, and holdout features."""

    settings = settings or get_settings()
    x = churn_training[CHURN_FEATURE_COLUMNS]
    y = churn_training["churned"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=settings.churn_test_size,
        random_state=settings.random_seed,
        stratify=y,
    )
    model = build_churn_model(settings)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    return model, auc, x_test


def weekly_kpi_summary(metrics: pd.DataFrame) -> dict[str, object]:
    """Build a compact weekly KPI summary for LLM insight generation."""

    recent = metrics.sort_values("date").tail(14).copy()
    current = recent.tail(7)
    previous = recent.head(7)

    def pct_delta(column: str) -> float:
        prev = float(previous[column].mean())
        curr = float(current[column].mean())
        return 0.0 if prev == 0 else (curr - prev) / prev

    return {
        "week_start": pd.Timestamp(current["date"].min()).strftime("%Y-%m-%d"),
        "week_end": pd.Timestamp(current["date"].max()).strftime("%Y-%m-%d"),
        "mrr": round(float(current["mrr"].mean()), 2),
        "mrr_wow": round(pct_delta("mrr"), 4),
        "dau": round(float(current["dau"].mean()), 1),
        "dau_wow": round(pct_delta("dau"), 4),
        "churn_rate": round(float(current["churn_rate"].mean()), 4),
        "churn_wow": round(pct_delta("churn_rate"), 4),
        "nps": round(float(current["nps"].mean()), 1),
        "nps_wow": round(pct_delta("nps"), 4),
        "feature_a_adoption": round(float(current["feature_a_adoption"].mean()), 4),
        "feature_b_adoption": round(float(current["feature_b_adoption"].mean()), 4),
        "retention_d30": round(float(current["retention_d30"].mean()), 4),
    }


def run_ml_pipeline(settings: Settings | None = None, sensitivity: float | None = None) -> PipelineResult:
    """Run synthetic data generation, anomaly detection, forecasting, churn modeling, and SHAP explanation."""

    settings = settings or get_settings()
    datasets = generate_all(settings)
    detector = AnomalyDetector(settings)
    scored_metrics = detector.fit_predict(datasets.daily_metrics, sensitivity=sensitivity)
    forecasts = forecast_metrics(datasets.daily_metrics, settings)
    churn_model, churn_auc, holdout = train_churn_model(datasets.churn_training, settings)
    feature_importance = ChurnExplainer(churn_model, CHURN_FEATURE_COLUMNS).top_features(holdout, limit=8)

    return PipelineResult(
        datasets=datasets,
        scored_metrics=scored_metrics,
        forecasts=forecasts,
        churn_model=churn_model,
        churn_auc=churn_auc,
        feature_importance=feature_importance,
    )
