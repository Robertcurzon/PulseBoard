"""Retention cohort heatmap component."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_cohort_heatmap(cohorts: pd.DataFrame) -> None:
    """Render a monthly retention cohort heatmap."""

    pivot = cohorts.pivot(index="cohort_month", columns="age_month", values="retention_rate").sort_index()
    fig = px.imshow(
        pivot,
        color_continuous_scale=["#172033", "#245b8f", "#2fd17c"],
        aspect="auto",
        labels=dict(x="Months Since Signup", y="Signup Cohort", color="Retention"),
        zmin=0,
        zmax=1,
    )
    fig.update_layout(
        height=420,
        template="plotly_dark",
        paper_bgcolor="#11192c",
        plot_bgcolor="#11192c",
        margin=dict(l=20, r=20, t=45, b=20),
        title="Retention Cohorts",
    )
    fig.update_traces(hovertemplate="Cohort %{y}<br>Month %{x}<br>Retention %{z:.1%}<extra></extra>")
    st.plotly_chart(fig, width="stretch")
