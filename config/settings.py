"""Centralized runtime configuration for PulseBoard."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    llm_timeout_seconds: float = _float_env("PULSEBOARD_LLM_TIMEOUT_SECONDS", 20.0)
    llm_max_tokens: int = _int_env("PULSEBOARD_LLM_MAX_TOKENS", 450)
    llm_temperature: float = _float_env("PULSEBOARD_LLM_TEMPERATURE", 0.25)

    random_seed: int = _int_env("PULSEBOARD_RANDOM_SEED", 42)
    history_days: int = _int_env("PULSEBOARD_HISTORY_DAYS", 365)
    forecast_horizon_days: int = _int_env("PULSEBOARD_FORECAST_HORIZON_DAYS", 30)
    anomaly_contamination: float = _float_env("PULSEBOARD_ANOMALY_CONTAMINATION", 0.035)
    anomaly_score_threshold: float = _float_env("PULSEBOARD_ANOMALY_SCORE_THRESHOLD", -0.08)

    baseline_dau: int = _int_env("PULSEBOARD_BASELINE_DAU", 18_000)
    baseline_arpu: float = _float_env("PULSEBOARD_BASELINE_ARPU", 72.0)
    baseline_nps: float = _float_env("PULSEBOARD_BASELINE_NPS", 43.0)
    cohort_months: int = _int_env("PULSEBOARD_COHORT_MONTHS", 12)

    churn_test_size: float = _float_env("PULSEBOARD_CHURN_TEST_SIZE", 0.25)
    churn_positive_threshold: float = _float_env("PULSEBOARD_CHURN_POSITIVE_THRESHOLD", 0.5)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return memoized application settings."""

    return Settings()
