"""Synthetic SaaS product and business metrics for PulseBoard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for all synthetic data used by the dashboard."""

    daily_metrics: pd.DataFrame
    churn_training: pd.DataFrame
    retention_cohorts: pd.DataFrame


def generate_daily_metrics(settings: Settings | None = None) -> pd.DataFrame:
    """Generate 12 months of daily SaaS KPI metrics with injected anomalies."""

    settings = settings or get_settings()
    rng = np.random.default_rng(settings.random_seed)
    end_date = pd.Timestamp(date.today()).normalize()
    dates = pd.date_range(end=end_date, periods=settings.history_days, freq="D")
    t = np.arange(settings.history_days)

    annual_trend = 1.0 + 0.28 * (t / max(settings.history_days - 1, 1))
    weekly = 1.0 + 0.11 * np.sin(2 * np.pi * t / 7 - 0.8)
    monthly = 1.0 + 0.045 * np.sin(2 * np.pi * t / 30.5)
    launch_lift = 1.0 + 0.075 / (1 + np.exp(-(t - settings.history_days * 0.62) / 10))

    dau = settings.baseline_dau * annual_trend * weekly * monthly * launch_lift
    dau += rng.normal(0, settings.baseline_dau * 0.035, settings.history_days)
    dau = np.clip(dau, 2_500, None).round().astype(int)

    new_signups = (dau * rng.normal(0.052, 0.006, settings.history_days)).round().astype(int)
    activation_rate = np.clip(0.53 + 0.08 * (t / settings.history_days) + rng.normal(0, 0.018, settings.history_days), 0.35, 0.78)
    churn_rate = np.clip(0.043 - 0.008 * (t / settings.history_days) + rng.normal(0, 0.0045, settings.history_days), 0.012, 0.075)
    arpu = np.clip(settings.baseline_arpu * (1 + 0.09 * t / settings.history_days) + rng.normal(0, 1.8, settings.history_days), 38, None)

    mau = pd.Series(dau).rolling(30, min_periods=7).sum().mul(0.18).bfill().to_numpy()
    paid_accounts = mau * np.clip(0.115 + 0.02 * (t / settings.history_days), 0.1, 0.16)
    mrr = paid_accounts * arpu

    feature_a = np.clip(0.31 + 0.20 / (1 + np.exp(-(t - settings.history_days * 0.52) / 16)) + rng.normal(0, 0.018, settings.history_days), 0.12, 0.72)
    feature_b = np.clip(0.24 + 0.10 * np.sin(2 * np.pi * t / 91) + 0.12 * (t / settings.history_days) + rng.normal(0, 0.019, settings.history_days), 0.09, 0.62)
    ab_variant_b_share = np.where(t < settings.history_days * 0.48, 0.0, np.clip(0.5 + rng.normal(0, 0.018, settings.history_days), 0.42, 0.58))
    nps = np.clip(settings.baseline_nps + 9 * (t / settings.history_days) + 5.5 * feature_a - 80 * churn_rate + rng.normal(0, 2.8, settings.history_days), 5, 78)
    retention_d7 = np.clip(0.42 + 0.06 * activation_rate + 0.05 * feature_a - 0.9 * churn_rate + rng.normal(0, 0.012, settings.history_days), 0.23, 0.68)
    retention_d30 = np.clip(0.24 + 0.05 * activation_rate + 0.04 * feature_a - 0.75 * churn_rate + rng.normal(0, 0.01, settings.history_days), 0.1, 0.49)

    metric_modifiers = {
        int(settings.history_days * 0.23): {"dau": 0.82, "mrr": 0.96, "nps": -8, "label": "Onboarding incident"},
        int(settings.history_days * 0.58): {"dau": 1.19, "mrr": 1.12, "nps": 5, "label": "Feature launch spike"},
        int(settings.history_days * 0.81): {"dau": 0.88, "mrr": 0.91, "nps": -10, "label": "Billing outage"},
    }
    anomaly_label = np.array([""] * settings.history_days, dtype=object)
    for center, modifier in metric_modifiers.items():
        start = max(center - 2, 0)
        stop = min(center + 3, settings.history_days)
        dau[start:stop] = np.round(dau[start:stop] * modifier["dau"]).astype(int)
        mrr[start:stop] = mrr[start:stop] * modifier["mrr"]
        nps[start:stop] = np.clip(nps[start:stop] + modifier["nps"], 0, 100)
        churn_rate[start:stop] = np.clip(churn_rate[start:stop] * (1.28 if modifier["nps"] < 0 else 0.82), 0.01, 0.12)
        anomaly_label[start:stop] = modifier["label"]

    df = pd.DataFrame(
        {
            "date": dates,
            "dau": dau,
            "mau": mau.round().astype(int),
            "new_signups": new_signups,
            "activation_rate": activation_rate,
            "churn_rate": churn_rate,
            "arpu": arpu,
            "mrr": mrr,
            "feature_a_adoption": feature_a,
            "feature_b_adoption": feature_b,
            "ab_variant_b_share": ab_variant_b_share,
            "nps": nps,
            "retention_d7": retention_d7,
            "retention_d30": retention_d30,
            "injected_anomaly": anomaly_label,
        }
    )
    df["weekday"] = df["date"].dt.day_name()
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time
    return df


def generate_churn_training_data(settings: Settings | None = None, n_users: int = 8_000) -> pd.DataFrame:
    """Generate user-level training data for churn prediction."""

    settings = settings or get_settings()
    rng = np.random.default_rng(settings.random_seed + 7)

    tenure_days = rng.integers(7, 1_100, n_users)
    sessions_30d = rng.poisson(13, n_users) + rng.integers(0, 8, n_users)
    feature_a_events = rng.poisson(7, n_users)
    feature_b_events = rng.poisson(5, n_users)
    support_tickets = rng.poisson(0.55, n_users)
    days_since_last_login = np.clip(rng.gamma(shape=2.0, scale=4.5, size=n_users), 0, 60)
    seats = rng.choice([1, 2, 3, 5, 10, 25, 50], n_users, p=[0.32, 0.19, 0.15, 0.14, 0.11, 0.06, 0.03])
    plan_value = rng.choice([29, 79, 149, 399], n_users, p=[0.34, 0.38, 0.21, 0.07])
    nps_response = np.clip(rng.normal(47, 18, n_users), 0, 100)
    onboarding_complete = rng.binomial(1, np.clip(0.62 + sessions_30d / 120, 0.55, 0.92), n_users)

    linear_risk = (
        -2.3
        - 0.018 * sessions_30d
        - 0.035 * feature_a_events
        - 0.026 * feature_b_events
        - 0.002 * tenure_days
        - 0.011 * nps_response
        - 0.36 * onboarding_complete
        + 0.45 * support_tickets
        + 0.08 * days_since_last_login
        - 0.006 * seats
        - 0.0006 * plan_value
    )
    churn_probability = 1 / (1 + np.exp(-linear_risk))
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_days": tenure_days,
            "sessions_30d": sessions_30d,
            "feature_a_events": feature_a_events,
            "feature_b_events": feature_b_events,
            "support_tickets": support_tickets,
            "days_since_last_login": days_since_last_login,
            "seats": seats,
            "plan_value": plan_value,
            "nps_response": nps_response,
            "onboarding_complete": onboarding_complete,
            "churned": churned,
            "churn_probability": churn_probability,
        }
    )


def generate_retention_cohorts(settings: Settings | None = None) -> pd.DataFrame:
    """Generate monthly retention cohorts in long format."""

    settings = settings or get_settings()
    rng = np.random.default_rng(settings.random_seed + 21)
    cohort_starts = pd.date_range(end=pd.Timestamp(date.today()).normalize().replace(day=1), periods=settings.cohort_months, freq="MS")
    records: list[dict[str, object]] = []

    for cohort_index, cohort_month in enumerate(cohort_starts):
        base = 0.84 - 0.025 * cohort_index + rng.normal(0, 0.01)
        quality_lift = 0.018 * cohort_index
        for age_month in range(settings.cohort_months - cohort_index):
            decay = np.exp(-0.19 * age_month)
            retention = np.clip((base + quality_lift) * decay + 0.18 + rng.normal(0, 0.014), 0.08, 0.95)
            records.append(
                {
                    "cohort_month": cohort_month.strftime("%Y-%m"),
                    "age_month": age_month,
                    "retention_rate": retention,
                }
            )

    return pd.DataFrame(records)


def generate_all(settings: Settings | None = None) -> SyntheticDataset:
    """Generate all PulseBoard synthetic datasets."""

    settings = settings or get_settings()
    return SyntheticDataset(
        daily_metrics=generate_daily_metrics(settings),
        churn_training=generate_churn_training_data(settings),
        retention_cohorts=generate_retention_cohorts(settings),
    )
