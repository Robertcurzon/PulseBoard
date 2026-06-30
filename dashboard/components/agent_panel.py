"""Agentic analyst panel for PulseBoard."""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import streamlit as st

from llm.analyst_agent import answer_dashboard_question


SUGGESTED_QUESTIONS = [
    "What changed this week, and what should the business do next?",
    "Which segment is driving revenue risk?",
    "Are the anomalies explainable by product, GTM, or billing events?",
    "Where should the team focus to improve conversion?",
]


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def render_agent_panel(
    metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    anomalies: pd.DataFrame,
    filters: dict[str, Any],
) -> None:
    """Render an agentic analyst interface for the current dashboard slice."""

    st.markdown("### AI Analyst Agent")
    st.caption("The agent inspects KPI deltas, segment drivers, and anomaly context before answering.")
    selected_prompt = st.selectbox("Suggested analysis", SUGGESTED_QUESTIONS, index=0)
    question = st.text_area("Ask about this dashboard slice", value=selected_prompt, height=82)
    if st.button("Run agent analysis", type="primary"):
        with st.spinner("Agent is inspecting metrics, segment drivers, and anomalies..."):
            answer = _run_async(answer_dashboard_question(question, filters, metrics, segment_metrics, anomalies))
        st.markdown(answer)
