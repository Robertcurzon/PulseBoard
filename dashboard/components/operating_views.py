"""Product and business operating views for the PulseBoard demo."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_filter_chips(segments: list[str], regions: list[str], channels: list[str]) -> None:
    """Render compact active-filter chips for demo context."""

    chips = [f"Segments: {', '.join(segments)}", f"Regions: {', '.join(regions)}", f"Channels: {', '.join(channels)}"]
    html = "".join(f'<span class="pb-chip">{chip}</span>' for chip in chips)
    st.markdown(f'<div class="pb-chip-row">{html}</div>', unsafe_allow_html=True)


def render_event_log(events: pd.DataFrame) -> None:
    """Render filtered business events as an operating timeline."""

    st.markdown("#### Operating Events")
    if events.empty:
        st.info("No annotated business events in the selected date range.")
        return
    for _, event in events.sort_values("date", ascending=False).iterrows():
        st.markdown(
            f"""
            <div class="pb-event">
                <strong>{pd.Timestamp(event['date']).strftime('%b %-d')} · {event['event']}</strong><br/>
                <span>{event['category']} · {event['description']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_segment_mix(segment_metrics: pd.DataFrame, metric: str = "mrr") -> None:
    """Render a stacked segment mix trend for the selected metric."""

    label = metric.replace("_", " ").title()
    trend = segment_metrics.groupby(["date", "segment"], as_index=False)[metric].sum()
    fig = px.area(
        trend,
        x="date",
        y=metric,
        color="segment",
        color_discrete_map={"Enterprise": "#2fd17c", "Mid-Market": "#58a6ff", "Startup": "#f2cc60"},
        labels={metric: label, "date": "Date", "segment": "Segment"},
    )
    fig.update_layout(
        height=330,
        template="plotly_dark",
        paper_bgcolor="#11192c",
        plot_bgcolor="#11192c",
        margin=dict(l=20, r=20, t=42, b=20),
        title=f"{label} Mix By Segment",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def render_acquisition_funnel(segment_metrics: pd.DataFrame) -> None:
    """Render acquisition funnel volume by channel."""

    funnel = (
        segment_metrics.groupby("acquisition_channel", as_index=False)[["new_signups", "activated_users", "paid_conversions"]]
        .sum()
        .melt(id_vars="acquisition_channel", var_name="stage", value_name="accounts")
    )
    stage_order = ["new_signups", "activated_users", "paid_conversions"]
    fig = px.bar(
        funnel,
        x="stage",
        y="accounts",
        color="acquisition_channel",
        barmode="group",
        category_orders={"stage": stage_order},
        color_discrete_sequence=["#58a6ff", "#f2cc60", "#2fd17c"],
        labels={"stage": "Funnel Stage", "accounts": "Accounts", "acquisition_channel": "Channel"},
    )
    fig.update_xaxes(ticktext=["Signups", "Activated", "Paid"], tickvals=stage_order)
    fig.update_layout(
        height=330,
        template="plotly_dark",
        paper_bgcolor="#11192c",
        plot_bgcolor="#11192c",
        margin=dict(l=20, r=20, t=42, b=20),
        title="Acquisition Funnel By Channel",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")


def render_pipeline_quality(metrics: pd.DataFrame) -> None:
    """Render pipeline created, won, and net revenue retention on one panel."""

    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics["date"], y=metrics["pipeline_created"], name="Pipeline Created", marker_color="#58a6ff"))
    fig.add_trace(go.Bar(x=metrics["date"], y=metrics["pipeline_won"], name="Pipeline Won", marker_color="#2fd17c"))
    fig.add_trace(
        go.Scatter(
            x=metrics["date"],
            y=metrics["net_revenue_retention"],
            name="NRR",
            yaxis="y2",
            line=dict(color="#f2cc60", width=2.2),
        )
    )
    fig.update_layout(
        height=340,
        template="plotly_dark",
        paper_bgcolor="#11192c",
        plot_bgcolor="#11192c",
        barmode="overlay",
        margin=dict(l=20, r=20, t=42, b=20),
        title="Pipeline Quality And NRR",
        yaxis=dict(title="Pipeline $"),
        yaxis2=dict(title="NRR", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def render_feature_adoption(segment_metrics: pd.DataFrame) -> None:
    """Render A/B and feature adoption curves by segment."""

    adoption = segment_metrics.groupby(["date", "segment"], as_index=False)[["feature_a_adoption", "feature_b_adoption"]].mean()
    fig = go.Figure()
    colors = {"Enterprise": "#2fd17c", "Mid-Market": "#58a6ff", "Startup": "#f2cc60"}
    for segment, group in adoption.groupby("segment"):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["feature_a_adoption"],
                mode="lines",
                name=f"{segment} Feature A",
                line=dict(color=colors.get(segment, "#9aa9c0"), width=2.1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["feature_b_adoption"],
                mode="lines",
                name=f"{segment} Feature B",
                line=dict(color=colors.get(segment, "#9aa9c0"), width=1.5, dash="dot"),
            )
        )
    fig.update_layout(
        height=340,
        template="plotly_dark",
        paper_bgcolor="#11192c",
        plot_bgcolor="#11192c",
        margin=dict(l=20, r=20, t=42, b=20),
        title="Feature Adoption: Control vs AI Copilot Motion",
        yaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
