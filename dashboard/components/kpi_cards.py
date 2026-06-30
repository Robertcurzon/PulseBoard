"""Top-line KPI card components."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _format_value(metric: str, value: float) -> str:
    if metric in {"mrr", "pipeline_created", "pipeline_won", "revenue_at_risk"}:
        return f"${value / 1_000_000:.2f}M"
    if metric in {"churn_rate", "net_revenue_retention", "trial_to_paid_rate"}:
        return f"{value:.2%}"
    if metric == "nps":
        return f"{value:.1f}"
    return f"{value:,.0f}"


def _wow_delta(metrics: pd.DataFrame, column: str) -> float:
    ordered = metrics.sort_values("date").tail(14)
    if len(ordered) < 8:
        return 0.0
    previous = ordered.head(7)[column].mean()
    current = ordered.tail(7)[column].mean()
    return 0.0 if previous == 0 else float((current - previous) / previous)


def render_kpi_cards(metrics: pd.DataFrame) -> None:
    """Render top-line metric cards with week-over-week deltas."""

    cards = [
        ("MRR", "mrr"),
        ("DAU", "dau"),
        ("Churn Rate", "churn_rate"),
        ("NPS", "nps"),
        ("NRR", "net_revenue_retention"),
        ("Pipeline", "pipeline_created"),
    ]
    cols = st.columns(6)
    recent = metrics.sort_values("date").tail(7)
    for col, (label, metric) in zip(cols, cards, strict=True):
        value = float(recent[metric].mean())
        delta = _wow_delta(metrics, metric)
        reverse = metric in {"churn_rate", "revenue_at_risk"}
        delta_class = "pb-card-delta-positive" if (delta >= 0 and not reverse) or (delta <= 0 and reverse) else "pb-card-delta-negative"
        direction = "+" if delta >= 0 else ""
        with col:
            st.markdown(
                f"""
                <div class="pb-card">
                    <div class="pb-card-label">{label}</div>
                    <div class="pb-card-value">{_format_value(metric, value)}</div>
                    <div class="{delta_class}">{direction}{delta:.1%} WoW</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
