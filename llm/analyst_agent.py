"""Agentic PulseBoard analyst that inspects dashboard state before answering."""

from __future__ import annotations

from typing import Any

import pandas as pd
from anthropic import AsyncAnthropic

from config.settings import Settings, get_settings
from llm.prompt_templates import render_analyst_agent_prompt


def build_agent_observations(
    metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    anomalies: pd.DataFrame,
    question: str,
) -> dict[str, Any]:
    """Run deterministic analysis tools over the current dashboard slice."""

    ordered = metrics.sort_values("date").copy()
    recent = ordered.tail(7)
    previous = ordered.tail(14).head(7)
    metric_columns = [
        "mrr",
        "dau",
        "churn_rate",
        "nps",
        "trial_to_paid_rate",
        "net_revenue_retention",
        "pipeline_created",
        "revenue_at_risk",
    ]
    deltas = {
        column: _pct_delta(previous[column].mean(), recent[column].mean())
        for column in metric_columns
        if column in ordered.columns and len(previous) > 0 and len(recent) > 0
    }
    current = {
        column: round(float(recent[column].mean()), 4)
        for column in metric_columns
        if column in ordered.columns and len(recent) > 0
    }
    segment_drivers = _segment_driver_table(segment_metrics)
    anomaly_summary = _anomaly_summary(anomalies)
    mentioned_metrics = [column for column in metric_columns if column.replace("_", " ") in question.lower() or column in question.lower()]

    return {
        "tools_run": [
            "weekly_kpi_delta_scan",
            "segment_driver_comparison",
            "anomaly_event_scan",
            "question_metric_focus",
        ],
        "current_week": current,
        "week_over_week_delta": deltas,
        "top_segment_drivers": segment_drivers,
        "anomaly_summary": anomaly_summary,
        "question_metric_focus": mentioned_metrics or ["mrr", "dau", "churn_rate"],
    }


async def answer_dashboard_question(
    question: str,
    filters: dict[str, Any],
    metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    anomalies: pd.DataFrame,
    settings: Settings | None = None,
) -> str:
    """Answer a dashboard question using tool observations and optional Claude synthesis."""

    settings = settings or get_settings()
    observations = build_agent_observations(metrics, segment_metrics, anomalies, question)
    if not settings.anthropic_api_key:
        return offline_agent_answer(question, filters, observations)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds)
    message = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=650,
        temperature=0.2,
        messages=[{"role": "user", "content": render_analyst_agent_prompt(question, filters, observations)}],
    )
    text_blocks = [block.text for block in message.content if getattr(block, "type", None) == "text"]
    return "\n".join(text_blocks).strip() or offline_agent_answer(question, filters, observations)


def offline_agent_answer(question: str, filters: dict[str, Any], observations: dict[str, Any]) -> str:
    """Deterministic agent answer used when Claude is not configured."""

    deltas = observations["week_over_week_delta"]
    current = observations["current_week"]
    top_driver = observations["top_segment_drivers"][0] if observations["top_segment_drivers"] else {}
    anomaly = observations["anomaly_summary"][0] if observations["anomaly_summary"] else None
    strongest_metric = max(deltas, key=lambda key: abs(deltas[key])) if deltas else "mrr"

    anomaly_line = (
        f"The most relevant flagged event is {anomaly['date']} / {anomaly['metric']} ({anomaly['event']})."
        if anomaly
        else "No material anomaly is dominating the selected slice."
    )
    driver_line = (
        f"{top_driver.get('segment', 'Selected segment')} contributes ${top_driver.get('mrr', 0):,.0f} MRR "
        f"with {top_driver.get('trial_to_paid_rate', 0):.1%} trial-to-paid."
        if top_driver
        else "Segment driver data is limited for this slice."
    )

    return "\n".join(
        [
            "#### Agent steps",
            "- Scanned current-week KPI levels and week-over-week movement.",
            "- Compared segment contribution, conversion, churn, and revenue at risk.",
            "- Checked flagged anomalies and annotated business events.",
            "",
            "#### Diagnosis",
            f"- The largest movement is `{strongest_metric}` at {deltas.get(strongest_metric, 0):+.1%} WoW; current MRR is ${current.get('mrr', 0):,.0f}.",
            f"- {driver_line}",
            f"- {anomaly_line}",
            "",
            "#### Recommended actions",
            "- Drill into the leading segment/channel pair before changing forecast assumptions.",
            "- Review anomaly dates against release, billing, campaign, and customer success timelines.",
            "- Use the uploaded-data validation notes to close missing KPI gaps before sharing externally.",
        ]
    )


def _pct_delta(previous: float, current: float) -> float:
    return 0.0 if previous == 0 else float((current - previous) / previous)


def _segment_driver_table(segment_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    if segment_metrics.empty:
        return []
    grouped = (
        segment_metrics.groupby("segment", as_index=False)
        .agg(
            mrr=("mrr", "sum"),
            dau=("dau", "sum"),
            trial_to_paid_rate=("trial_to_paid_rate", "mean"),
            churn_rate=("churn_rate", "mean"),
            revenue_at_risk=("revenue_at_risk", "mean"),
        )
        .sort_values("mrr", ascending=False)
        .head(5)
    )
    return grouped.round(4).to_dict(orient="records")


def _anomaly_summary(anomalies: pd.DataFrame) -> list[dict[str, Any]]:
    if anomalies.empty:
        return []
    rows = anomalies.loc[anomalies["is_anomaly"]].copy() if "is_anomaly" in anomalies.columns else anomalies.copy()
    if rows.empty:
        return []
    rows["_has_event"] = rows.get("injected_anomaly", "").fillna("").astype(str).str.len() > 0
    rows = rows.sort_values(["_has_event", "anomaly_score"], ascending=[False, True]).head(5)
    return [
        {
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "metric": str(row.get("primary_anomaly_metric", "kpi")),
            "score": round(float(row.get("anomaly_score", 0.0)), 4),
            "event": str(row.get("injected_anomaly", "")) or "unlabeled anomaly",
        }
        for _, row in rows.iterrows()
    ]
