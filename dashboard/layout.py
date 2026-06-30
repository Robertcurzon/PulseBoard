"""Streamlit layout and theme helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st


METRIC_OPTIONS = {
    "DAU": "dau",
    "MRR": "mrr",
    "Churn Rate": "churn_rate",
    "NPS": "nps",
    "Trial-to-Paid": "trial_to_paid_rate",
    "Net Revenue Retention": "net_revenue_retention",
    "Feature A Adoption": "feature_a_adoption",
    "Feature B Adoption": "feature_b_adoption",
    "Pipeline Created": "pipeline_created",
}


def configure_page() -> None:
    """Configure Streamlit page metadata and CSS theme."""

    st.set_page_config(page_title="PulseBoard", page_icon="PB", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        :root {
            --pb-bg: #0b1020;
            --pb-panel: #121a2f;
            --pb-border: #26324f;
            --pb-text: #e6edf8;
            --pb-muted: #9aa9c0;
            --pb-green: #2fd17c;
            --pb-red: #ff6b6b;
            --pb-blue: #58a6ff;
            --pb-gold: #f2cc60;
        }
        .stApp { background: var(--pb-bg); color: var(--pb-text); }
        [data-testid="stSidebar"] { background: #0e1628; border-right: 1px solid var(--pb-border); }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: 0; }
        .pb-title { font-size: 2.1rem; font-weight: 760; margin-bottom: 0.15rem; }
        .pb-subtitle { color: var(--pb-muted); font-size: 1rem; margin-bottom: 1.25rem; }
        .pb-card {
            background: linear-gradient(180deg, #151e35 0%, #11192c 100%);
            border: 1px solid var(--pb-border);
            border-radius: 8px;
            padding: 1rem;
            min-height: 118px;
        }
        .pb-card-label { color: var(--pb-muted); font-size: 0.8rem; text-transform: uppercase; font-weight: 700; }
        .pb-card-value { color: var(--pb-text); font-size: 1.75rem; font-weight: 760; margin-top: 0.25rem; }
        .pb-card-delta-positive { color: var(--pb-green); font-size: 0.9rem; font-weight: 700; }
        .pb-card-delta-negative { color: var(--pb-red); font-size: 0.9rem; font-weight: 700; }
        .pb-chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .pb-chip {
            border: 1px solid var(--pb-border);
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            color: var(--pb-muted);
            background: #0f1729;
            font-size: 0.8rem;
            font-weight: 650;
        }
        .pb-panel {
            background: #11192c;
            border: 1px solid var(--pb-border);
            border-radius: 8px;
            padding: 0.85rem;
        }
        .pb-panel + .pb-panel { margin-top: 0.65rem; }
        .pb-feed {
            max-height: 430px;
            overflow-y: auto;
            padding-right: 0.35rem;
        }
        .pb-event {
            border-left: 3px solid var(--pb-gold);
            padding: 0.45rem 0 0.45rem 0.75rem;
            margin-bottom: 0.55rem;
            color: var(--pb-muted);
        }
        .pb-event strong { color: var(--pb-text); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the dashboard title area."""

    st.markdown('<div class="pb-title">PulseBoard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pb-subtitle">AI-powered product and business analytics intelligence for SaaS operating reviews.</div>',
        unsafe_allow_html=True,
    )


def sidebar_filters(metrics: pd.DataFrame, segment_metrics: pd.DataFrame) -> tuple[tuple[date, date], str, float, list[str], list[str], list[str]]:
    """Render sidebar controls and return selected filters."""

    min_date = pd.Timestamp(metrics["date"].min()).date()
    max_date = pd.Timestamp(metrics["date"].max()).date()
    st.sidebar.header("Controls")
    selected_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    metric_label = st.sidebar.selectbox("Trend metric", list(METRIC_OPTIONS.keys()), index=0)
    sensitivity = st.sidebar.slider("Anomaly sensitivity", min_value=0.01, max_value=0.12, value=0.035, step=0.005)
    st.sidebar.divider()
    st.sidebar.subheader("Demo Slices")
    segments = sorted(segment_metrics["segment"].unique())
    regions = sorted(segment_metrics["region"].unique())
    channels = sorted(segment_metrics["acquisition_channel"].unique())
    selected_segments = st.sidebar.multiselect("Customer segments", segments, default=segments)
    selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)
    selected_channels = st.sidebar.multiselect("Acquisition channels", channels, default=channels)
    if not isinstance(selected_range, tuple) or len(selected_range) != 2:
        selected_range = (min_date, max_date)
    return selected_range, METRIC_OPTIONS[metric_label], float(sensitivity), selected_segments, selected_regions, selected_channels
