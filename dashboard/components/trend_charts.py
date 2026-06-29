"""Plotly trend charts with forecast overlays."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml.forecaster import ForecastResult


def render_trend_chart(metrics: pd.DataFrame, forecast_result: ForecastResult, metric: str) -> None:
    """Render a KPI trend chart with forecast and confidence interval."""

    label = metric.replace("_", " ").upper()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics["date"],
            y=metrics[metric],
            mode="lines",
            name=f"Actual {label}",
            line=dict(color="#58a6ff", width=2.5),
        )
    )
    forecast = forecast_result.forecast
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(88, 166, 255, 0.18)",
            line=dict(width=0),
            name="80% interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["yhat"],
            mode="lines",
            name=f"Forecast ({forecast_result.engine})",
            line=dict(color="#f2cc60", width=2.5, dash="dash"),
        )
    )
    fig.update_layout(
        height=430,
        template="plotly_dark",
        paper_bgcolor="#11192c",
        plot_bgcolor="#11192c",
        margin=dict(l=20, r=20, t=45, b=20),
        title=f"{label} Trend With 30-Day Forecast",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
