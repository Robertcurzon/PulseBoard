"""Anomaly timeline and narrative components."""

from __future__ import annotations

import asyncio

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from llm.anomaly_narrator import narrate_anomaly
from ml.anomaly_detector import AnomalyDetector


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def render_anomaly_panel(scored_metrics: pd.DataFrame, detector: AnomalyDetector) -> None:
    """Render anomaly timeline and LLM narratives for flagged records."""

    anomalies = scored_metrics.loc[scored_metrics["is_anomaly"]].sort_values("date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scored_metrics["date"],
            y=scored_metrics["anomaly_score"],
            mode="lines",
            name="Anomaly score",
            line=dict(color="#9aa9c0", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomalies["date"],
            y=anomalies["anomaly_score"],
            mode="markers",
            name="Flagged",
            marker=dict(color="#ff6b6b", size=9, symbol="diamond"),
            text=anomalies["primary_anomaly_metric"],
        )
    )
    fig.update_layout(
        height=330,
        template="plotly_dark",
        paper_bgcolor="#11192c",
        plot_bgcolor="#11192c",
        margin=dict(l=20, r=20, t=40, b=20),
        title="Anomaly Timeline",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    records = detector.anomaly_records(scored_metrics, limit=5)
    st.markdown("#### Business Narratives")
    if not records:
        st.info("No material anomalies detected for the selected period.")
        return

    for record in records:
        narrative = _run_async(narrate_anomaly(record))
        st.markdown(
            f"""
            <div class="pb-panel">
                <strong>{record['date']} · {record['metric']}</strong><br/>
                <span style="color:#9aa9c0;">score {record['score']:.3f}</span>
                <p>{narrative}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
