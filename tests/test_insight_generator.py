import pytest

from config.settings import Settings
from llm.insight_generator import generate_executive_insight, offline_executive_insight
from llm.prompt_templates import render_anomaly_narrative_prompt, render_executive_insight_prompt


def test_executive_prompt_includes_summary_values() -> None:
    summary = {"mrr": 1_200_000, "mrr_wow": 0.05, "dau": 20_100}
    prompt = render_executive_insight_prompt(summary)

    assert "exactly 3 markdown bullets" in prompt
    assert "1200000" in prompt
    assert "dau" in prompt


def test_anomaly_prompt_includes_record_values() -> None:
    anomaly = {"date": "2026-01-15", "metric": "dau", "score": -0.2}
    prompt = render_anomaly_narrative_prompt(anomaly)

    assert "exactly 2 sentences" in prompt
    assert "2026-01-15" in prompt
    assert "dau" in prompt


@pytest.mark.asyncio
async def test_generate_executive_insight_skips_api_without_key() -> None:
    settings = Settings(anthropic_api_key=None)
    summary = {"mrr": 1_200_000, "mrr_wow": 0.05, "dau": 20_100, "dau_wow": 0.02, "churn_rate": 0.04, "churn_wow": -0.01, "nps_wow": 0.03}
    text = await generate_executive_insight(summary, settings)

    assert text == offline_executive_insight(summary)
    assert text.count("- ") == 3
