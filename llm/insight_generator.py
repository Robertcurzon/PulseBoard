"""Claude-powered executive insight generation."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from config.settings import Settings, get_settings
from llm.prompt_templates import render_executive_insight_prompt


def offline_executive_insight(summary: dict[str, Any]) -> str:
    """Create deterministic executive insight bullets when Claude is unavailable."""

    mrr_wow = float(summary.get("mrr_wow", 0.0))
    dau_wow = float(summary.get("dau_wow", 0.0))
    churn_wow = float(summary.get("churn_wow", 0.0))
    nps_wow = float(summary.get("nps_wow", 0.0))

    return "\n".join(
        [
            f"- MRR averaged ${summary.get('mrr', 0):,.0f} this week ({mrr_wow:+.1%} WoW); prioritize expansion motions where adoption is strongest.",
            f"- DAU averaged {summary.get('dau', 0):,.0f} users ({dau_wow:+.1%} WoW); inspect activation and feature release cohorts for repeatable lift.",
            f"- Churn averaged {float(summary.get('churn_rate', 0)):.2%} ({churn_wow:+.1%} WoW) while NPS moved {nps_wow:+.1%}; route at-risk segments to customer success.",
        ]
    )


async def generate_executive_insight(summary: dict[str, Any], settings: Settings | None = None) -> str:
    """Generate a 3-bullet executive KPI insight using Claude, with offline fallback."""

    settings = settings or get_settings()
    if not settings.anthropic_api_key:
        return offline_executive_insight(summary)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds)
    message = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        messages=[{"role": "user", "content": render_executive_insight_prompt(summary)}],
    )
    text_blocks = [block.text for block in message.content if getattr(block, "type", None) == "text"]
    return "\n".join(text_blocks).strip() or offline_executive_insight(summary)
