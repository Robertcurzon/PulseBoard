"""Claude-powered anomaly narration."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from config.settings import Settings, get_settings
from llm.prompt_templates import render_anomaly_narrative_prompt


def offline_anomaly_narrative(anomaly: dict[str, Any]) -> str:
    """Create deterministic anomaly text when Claude is unavailable."""

    metric = anomaly.get("metric", "selected KPI")
    date = anomaly.get("date", "the selected date")
    severity = float(anomaly.get("severity", 0.0))
    context = anomaly.get("context", {})
    label = context.get("injected_label") if isinstance(context, dict) else ""
    cause = f" and aligns with the synthetic event '{label}'" if label else ""
    return (
        f"On {date}, {metric} moved outside its normal operating range with severity {severity:.2f}{cause}, "
        "which may distort weekly executive reads if left unsegmented. "
        "Review release, billing, acquisition, and support timelines for that window before changing forecasts or targets."
    )


async def narrate_anomaly(anomaly: dict[str, Any], settings: Settings | None = None) -> str:
    """Narrate a detected anomaly in two business-readable sentences."""

    settings = settings or get_settings()
    if not settings.anthropic_api_key:
        return offline_anomaly_narrative(anomaly)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds)
    message = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=180,
        temperature=0.2,
        messages=[{"role": "user", "content": render_anomaly_narrative_prompt(anomaly)}],
    )
    text_blocks = [block.text for block in message.content if getattr(block, "type", None) == "text"]
    return "\n".join(text_blocks).strip() or offline_anomaly_narrative(anomaly)
