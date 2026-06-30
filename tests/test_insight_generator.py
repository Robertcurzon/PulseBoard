import pytest

from config.settings import Settings
from llm.analyst_agent import answer_dashboard_question
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


@pytest.mark.asyncio
async def test_agent_answer_skips_api_without_key() -> None:
    import pandas as pd

    settings = Settings(anthropic_api_key=None)
    metrics = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=14),
            "mrr": [100_000 + i * 1000 for i in range(14)],
            "dau": [1000 + i * 10 for i in range(14)],
            "churn_rate": [0.03] * 14,
            "nps": [52] * 14,
            "trial_to_paid_rate": [0.22] * 14,
            "net_revenue_retention": [1.04] * 14,
            "pipeline_created": [40_000] * 14,
            "revenue_at_risk": [3000] * 14,
        }
    )
    segment_metrics = metrics.assign(segment="Enterprise")
    anomalies = metrics.assign(is_anomaly=False, anomaly_score=0.0, primary_anomaly_metric="mrr", injected_anomaly="")

    text = await answer_dashboard_question("What changed this week?", {}, metrics, segment_metrics, anomalies, settings)

    assert "Agent steps" in text
    assert "Recommended actions" in text
