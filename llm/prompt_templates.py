"""Versioned prompt templates for PulseBoard LLM calls."""

from __future__ import annotations

import json
from string import Template
from typing import Any


# v1.0: Executive weekly insight prompt optimized for concise board-facing summaries.
EXECUTIVE_INSIGHT_TEMPLATE = Template(
    """You are PulseBoard, an AI analytics chief of staff for a SaaS executive team.

Create exactly 3 markdown bullets. Each bullet must include:
- a clear business finding,
- one metric value or week-over-week change,
- a practical next action.

Avoid generic phrasing. Use plain executive language.

Weekly KPI summary JSON:
$summary_json
"""
)


# v1.0: Anomaly narrative prompt optimized for terse, plain-English incident context.
ANOMALY_NARRATIVE_TEMPLATE = Template(
    """You are PulseBoard, explaining an anomalous SaaS KPI movement to a business stakeholder.

Write exactly 2 sentences. Sentence 1 should explain what changed and why it matters.
Sentence 2 should name the most likely operational follow-up.

Anomaly JSON:
$anomaly_json
"""
)


def render_executive_insight_prompt(summary: dict[str, Any]) -> str:
    """Render the executive insight prompt from a weekly KPI summary."""

    return EXECUTIVE_INSIGHT_TEMPLATE.substitute(summary_json=json.dumps(summary, indent=2, sort_keys=True))


def render_anomaly_narrative_prompt(anomaly: dict[str, Any]) -> str:
    """Render the anomaly narrative prompt from an anomaly record."""

    return ANOMALY_NARRATIVE_TEMPLATE.substitute(anomaly_json=json.dumps(anomaly, indent=2, sort_keys=True))
