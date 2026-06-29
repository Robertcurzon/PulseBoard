"""PulseBoard Streamlit dashboard entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from dashboard.components.anomaly_panel import render_anomaly_panel
from dashboard.components.cohort_heatmap import render_cohort_heatmap
from dashboard.components.insight_feed import render_insight_feed
from dashboard.components.kpi_cards import render_kpi_cards
from dashboard.components.trend_charts import render_trend_chart
from dashboard.layout import configure_page, render_header, sidebar_filters
from ml.anomaly_detector import AnomalyDetector
from ml.pipeline import run_ml_pipeline


@st.cache_data(show_spinner=False)
def cached_pipeline(sensitivity: float) -> dict[str, object]:
    """Run and cache the ML pipeline for Streamlit sessions."""

    settings = get_settings()
    result = run_ml_pipeline(settings, sensitivity=sensitivity)
    return {
        "daily_metrics": result.datasets.daily_metrics,
        "retention_cohorts": result.datasets.retention_cohorts,
        "scored_metrics": result.scored_metrics,
        "forecasts": result.forecasts,
        "feature_importance": result.feature_importance,
        "churn_auc": result.churn_auc,
    }


def filter_by_date(metrics: pd.DataFrame, date_range: tuple[object, object]) -> pd.DataFrame:
    """Filter metrics to the selected inclusive date range."""

    start, end = [pd.Timestamp(value) for value in date_range]
    return metrics.loc[(metrics["date"] >= start) & (metrics["date"] <= end)].copy()


def main() -> None:
    """Render the PulseBoard application."""

    configure_page()
    settings = get_settings()
    bootstrap = run_ml_pipeline(settings, sensitivity=settings.anomaly_contamination)
    date_range, selected_metric, sensitivity = sidebar_filters(bootstrap.datasets.daily_metrics)
    data = cached_pipeline(sensitivity)

    daily_metrics = filter_by_date(data["daily_metrics"], date_range)
    scored_metrics = filter_by_date(data["scored_metrics"], date_range)
    detector = AnomalyDetector(settings)
    detector.fit(data["daily_metrics"])

    render_header()
    render_kpi_cards(daily_metrics)

    left, right = st.columns([2, 1])
    with left:
        forecast_key = selected_metric if selected_metric in data["forecasts"] else "dau"
        render_trend_chart(daily_metrics, data["forecasts"][forecast_key], selected_metric)
    with right:
        st.markdown("#### Churn Drivers")
        st.metric("Holdout ROC AUC", f"{data['churn_auc']:.3f}")
        for item in data["feature_importance"]:
            st.progress(min(item.importance / 2.0, 1.0), text=f"{item.feature}: {item.direction}")

    st.divider()
    col_a, col_b = st.columns([1.15, 0.85])
    with col_a:
        render_anomaly_panel(scored_metrics, detector)
    with col_b:
        render_cohort_heatmap(data["retention_cohorts"])

    st.divider()
    st.markdown("### Executive Insight Feed")
    render_insight_feed(daily_metrics)


if __name__ == "__main__":
    main()
