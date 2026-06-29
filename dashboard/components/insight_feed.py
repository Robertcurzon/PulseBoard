"""LLM executive insight feed component."""

from __future__ import annotations

import asyncio

import pandas as pd
import streamlit as st

from llm.insight_generator import generate_executive_insight
from ml.pipeline import weekly_kpi_summary


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def weekly_summaries(metrics: pd.DataFrame, weeks: int = 5) -> list[dict[str, object]]:
    """Create recent weekly summaries for the insight feed."""

    ordered = metrics.sort_values("date").copy()
    summaries: list[dict[str, object]] = []
    for _, group in ordered.groupby(pd.Grouper(key="date", freq="W")):
        if len(group) >= 7:
            summaries.append(weekly_kpi_summary(ordered.loc[ordered["date"] <= group["date"].max()].tail(14)))
    return summaries[-weeks:][::-1]


def render_insight_feed(metrics: pd.DataFrame) -> None:
    """Render a scrollable feed of generated weekly executive summaries."""

    summaries = weekly_summaries(metrics)
    st.markdown('<div class="pb-feed">', unsafe_allow_html=True)
    for summary in summaries:
        insight = _run_async(generate_executive_insight(summary))
        st.markdown(
            f"""
            <div class="pb-panel">
                <strong>{summary['week_start']} to {summary['week_end']}</strong>
                <div>{insight}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
