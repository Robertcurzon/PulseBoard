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
from dashboard.components.operating_views import (
    render_acquisition_funnel,
    render_event_log,
    render_feature_adoption,
    render_filter_chips,
    render_pipeline_quality,
    render_segment_mix,
)
from dashboard.components.trend_charts import render_trend_chart
from dashboard.layout import configure_page, render_header, sidebar_filters
from data.generators.synthetic_data import aggregate_segment_metrics
from ml.anomaly_detector import AnomalyDetector
from ml.forecaster import ProphetForecaster
from ml.pipeline import run_ml_pipeline


@st.cache_data(show_spinner=False)
def cached_pipeline(sensitivity: float) -> dict[str, object]:
    """Run and cache the ML pipeline for Streamlit sessions."""

    settings = get_settings()
    result = run_ml_pipeline(settings, sensitivity=sensitivity)
    return {
        "daily_metrics": result.datasets.daily_metrics,
        "segment_metrics": result.datasets.segment_metrics,
        "retention_cohorts": result.datasets.retention_cohorts,
        "event_log": result.datasets.event_log,
        "scored_metrics": result.scored_metrics,
        "forecasts": result.forecasts,
        "feature_importance": result.feature_importance,
        "churn_auc": result.churn_auc,
    }


def filter_by_date(metrics: pd.DataFrame, date_range: tuple[object, object]) -> pd.DataFrame:
    """Filter metrics to the selected inclusive date range."""

    start, end = [pd.Timestamp(value) for value in date_range]
    return metrics.loc[(metrics["date"] >= start) & (metrics["date"] <= end)].copy()


def filter_segments(segment_metrics: pd.DataFrame, segments: list[str], regions: list[str], channels: list[str]) -> pd.DataFrame:
    """Filter segment-level metrics to selected demo slices."""

    all_segments = sorted(segment_metrics["segment"].unique())
    all_regions = sorted(segment_metrics["region"].unique())
    all_channels = sorted(segment_metrics["acquisition_channel"].unique())
    segments = segments or all_segments
    regions = regions or all_regions
    channels = channels or all_channels
    return segment_metrics.loc[
        segment_metrics["segment"].isin(segments)
        & segment_metrics["region"].isin(regions)
        & segment_metrics["acquisition_channel"].isin(channels)
    ].copy()


def main() -> None:
    """Render the PulseBoard application."""

    configure_page()
    settings = get_settings()
    bootstrap = cached_pipeline(settings.anomaly_contamination)
    date_range, selected_metric, sensitivity, selected_segments, selected_regions, selected_channels = sidebar_filters(
        bootstrap["daily_metrics"],
        bootstrap["segment_metrics"],
    )
    data = cached_pipeline(sensitivity)

    selected_segment_metrics_all_dates = filter_segments(
        data["segment_metrics"],
        selected_segments,
        selected_regions,
        selected_channels,
    )
    daily_all_dates = aggregate_segment_metrics(selected_segment_metrics_all_dates)
    daily_metrics = filter_by_date(daily_all_dates, date_range)
    selected_segment_metrics = filter_by_date(selected_segment_metrics_all_dates, date_range)
    events = filter_by_date(data["event_log"], date_range)
    detector = AnomalyDetector(settings)
    detector.fit(daily_all_dates)
    scored_metrics = detector.predict(daily_metrics, sensitivity=sensitivity)
    forecast_result = ProphetForecaster(settings).forecast(daily_all_dates, selected_metric)

    render_header()
    render_filter_chips(selected_segments, selected_regions, selected_channels)
    render_kpi_cards(daily_metrics)

    left, right = st.columns([2, 1])
    with left:
        render_trend_chart(daily_metrics, forecast_result, selected_metric, events=events)
    with right:
        st.markdown("#### Churn Drivers")
        st.metric("Holdout ROC AUC", f"{data['churn_auc']:.3f}")
        for item in data["feature_importance"]:
            st.progress(min(item.importance / 2.0, 1.0), text=f"{item.feature}: {item.direction}")

    st.divider()
    ops_left, ops_right = st.columns(2)
    with ops_left:
        render_segment_mix(selected_segment_metrics, metric="mrr")
    with ops_right:
        render_acquisition_funnel(selected_segment_metrics)

    ops_left, ops_right = st.columns(2)
    with ops_left:
        render_pipeline_quality(daily_metrics)
    with ops_right:
        render_feature_adoption(selected_segment_metrics)

    st.divider()
    col_a, col_b = st.columns([1.1, 0.9])
    with col_a:
        render_anomaly_panel(scored_metrics, detector)
    with col_b:
        render_event_log(events)
        render_cohort_heatmap(data["retention_cohorts"])

    st.divider()
    st.markdown("### Executive Insight Feed")
    render_insight_feed(daily_metrics)


if __name__ == "__main__":
    main()
