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
    segment_metrics: pd.DataFrame
    churn_training: pd.DataFrame
    retention_cohorts: pd.DataFrame
    event_log: pd.DataFrame


SEGMENTS = {
    "Enterprise": {"dau": 0.42, "arpu": 2.65, "churn": 0.62, "nps": 7.0, "activation": 0.08},
    "Mid-Market": {"dau": 0.36, "arpu": 1.18, "churn": 0.92, "nps": 2.0, "activation": 0.03},
    "Startup": {"dau": 0.22, "arpu": 0.54, "churn": 1.34, "nps": -3.0, "activation": -0.02},
}

REGIONS = {
    "North America": {"dau": 0.54, "arpu": 1.18, "nps": 2.5},
    "EMEA": {"dau": 0.29, "arpu": 0.94, "nps": 0.0},
    "APAC": {"dau": 0.17, "arpu": 0.82, "nps": -1.5},
}

CHANNELS = {
    "Product-Led": {"signup": 0.44, "cac": 0.72, "conversion": 0.86},
    "Paid Search": {"signup": 0.31, "cac": 1.28, "conversion": 0.78},
    "Partner": {"signup": 0.25, "cac": 0.96, "conversion": 1.18},
}


def generate_event_log(settings: Settings | None = None) -> pd.DataFrame:
    """Return business events injected into the mock SaaS operating history."""

    settings = settings or get_settings()
    end_date = pd.Timestamp(date.today()).normalize()
    dates = pd.date_range(end=end_date, periods=settings.history_days, freq="D")
    events = [
        (0.18, "Pricing test", "Revenue", "Enterprise annual plan packaging lifted ARPU but softened signup conversion."),
        (0.34, "Search spend pullback", "Growth", "Paid search CAC spike forced a budget reallocation toward partner and product-led channels."),
        (0.51, "AI Copilot beta", "Product", "Feature A beta opened to 40% of accounts and increased activation for expansion-ready teams."),
        (0.63, "Partner launch", "GTM", "Partner channel campaign drove qualified pipeline and stronger trial-to-paid conversion."),
        (0.79, "Billing incident", "Reliability", "Invoice failures temporarily hit Enterprise MRR, churn risk, and NPS."),
        (0.91, "Win-back motion", "Customer Success", "Targeted save offers improved retention in Startup and Mid-Market cohorts."),
    ]
    return pd.DataFrame(
        {
            "date": [dates[min(int(settings.history_days * pct), settings.history_days - 1)] for pct, *_ in events],
            "event": [event for _, event, _, _ in events],
            "category": [category for _, _, category, _ in events],
            "description": [description for _, _, _, description in events],
        }
    )


def generate_segment_metrics(settings: Settings | None = None) -> pd.DataFrame:
    """Generate daily SaaS metrics by segment, region, and acquisition channel."""

    settings = settings or get_settings()
    rng = np.random.default_rng(settings.random_seed)
    end_date = pd.Timestamp(date.today()).normalize()
    dates = pd.date_range(end=end_date, periods=settings.history_days, freq="D")
    t = np.arange(settings.history_days)

    annual_trend = 1.0 + 0.28 * (t / max(settings.history_days - 1, 1))
    weekly = 1.0 + 0.11 * np.sin(2 * np.pi * t / 7 - 0.8)
    monthly = 1.0 + 0.045 * np.sin(2 * np.pi * t / 30.5)
    launch_lift = 1.0 + 0.075 / (1 + np.exp(-(t - settings.history_days * 0.62) / 10))

    base_dau = settings.baseline_dau * annual_trend * weekly * monthly * launch_lift
    base_activation = np.clip(0.53 + 0.08 * (t / settings.history_days), 0.35, 0.78)
    base_churn = np.clip(0.043 - 0.008 * (t / settings.history_days), 0.012, 0.075)
    base_arpu = np.clip(settings.baseline_arpu * (1 + 0.09 * t / settings.history_days), 38, None)
    base_feature_a = np.clip(0.31 + 0.20 / (1 + np.exp(-(t - settings.history_days * 0.52) / 16)), 0.12, 0.72)
    base_feature_b = np.clip(0.24 + 0.10 * np.sin(2 * np.pi * t / 91) + 0.12 * (t / settings.history_days), 0.09, 0.62)
    base_variant_b = np.where(t < settings.history_days * 0.48, 0.0, np.clip(0.5 + rng.normal(0, 0.018, settings.history_days), 0.42, 0.58))

    modifiers = {
        int(settings.history_days * 0.18): {"arpu": 1.05, "activation": 0.97, "label": "Pricing test", "segments": {"Enterprise"}},
        int(settings.history_days * 0.34): {"dau": 0.93, "signup": 0.78, "label": "Search spend pullback", "channels": {"Paid Search"}},
        int(settings.history_days * 0.51): {"dau": 1.08, "activation": 1.11, "feature_a": 1.18, "label": "AI Copilot beta"},
        int(settings.history_days * 0.63): {"signup": 1.25, "pipeline": 1.34, "conversion": 1.14, "label": "Partner launch", "channels": {"Partner"}},
        int(settings.history_days * 0.79): {"dau": 0.89, "mrr": 0.91, "nps": -12, "churn": 1.35, "label": "Billing incident", "segments": {"Enterprise", "Mid-Market"}},
        int(settings.history_days * 0.91): {"churn": 0.78, "nps": 5, "retention": 1.08, "label": "Win-back motion"},
    }

    records: list[dict[str, object]] = []
    for segment, segment_cfg in SEGMENTS.items():
        for region, region_cfg in REGIONS.items():
            for channel, channel_cfg in CHANNELS.items():
                grain_share = segment_cfg["dau"] * region_cfg["dau"] * channel_cfg["signup"]
                noise = rng.normal(1.0, 0.035, settings.history_days)
                dau = np.clip(base_dau * grain_share * noise, 35, None)
                signup_rate = rng.normal(0.052, 0.004, settings.history_days) * channel_cfg["conversion"]
                activation_rate = np.clip(base_activation + segment_cfg["activation"] + rng.normal(0, 0.015, settings.history_days), 0.28, 0.88)
                churn_rate = np.clip(base_churn * segment_cfg["churn"] + rng.normal(0, 0.0035, settings.history_days), 0.006, 0.11)
                arpu = np.clip(base_arpu * segment_cfg["arpu"] * region_cfg["arpu"] + rng.normal(0, 2.2, settings.history_days), 18, None)
                mau = pd.Series(dau).rolling(30, min_periods=7).sum().mul(0.18).bfill().to_numpy()
                paid_accounts = mau * np.clip(0.10 + 0.025 * (t / settings.history_days), 0.07, 0.19)
                mrr = paid_accounts * arpu
                feature_a = np.clip(base_feature_a + 0.05 * (segment == "Enterprise") + rng.normal(0, 0.018, settings.history_days), 0.08, 0.86)
                feature_b = np.clip(base_feature_b + 0.04 * (channel == "Product-Led") + rng.normal(0, 0.018, settings.history_days), 0.07, 0.76)
                nps = np.clip(settings.baseline_nps + 9 * (t / settings.history_days) + segment_cfg["nps"] + region_cfg["nps"] + 4.5 * feature_a - 76 * churn_rate + rng.normal(0, 2.7, settings.history_days), 0, 88)
                retention_d7 = np.clip(0.42 + 0.07 * activation_rate + 0.05 * feature_a - 0.85 * churn_rate + rng.normal(0, 0.012, settings.history_days), 0.2, 0.75)
                retention_d30 = np.clip(0.24 + 0.06 * activation_rate + 0.04 * feature_a - 0.72 * churn_rate + rng.normal(0, 0.01, settings.history_days), 0.08, 0.56)
                pipeline_created = mrr * rng.normal(0.31, 0.035, settings.history_days) * (1.0 + 0.18 * (segment == "Enterprise"))
                pipeline_won = pipeline_created * np.clip(0.25 + 0.16 * activation_rate + rng.normal(0, 0.015, settings.history_days), 0.12, 0.52)
                expansion_mrr = mrr * np.clip(0.014 + 0.018 * feature_a + rng.normal(0, 0.002, settings.history_days), 0.002, 0.055)
                contraction_mrr = mrr * np.clip(0.012 + 0.19 * churn_rate + rng.normal(0, 0.002, settings.history_days), 0.002, 0.06)
                cac = np.clip((75 + 18 * (segment == "Enterprise")) * channel_cfg["cac"] * rng.normal(1.0, 0.06, settings.history_days), 18, None)
                anomaly_label = np.array([""] * settings.history_days, dtype=object)

                for center, modifier in modifiers.items():
                    if segment not in modifier.get("segments", {segment}) or channel not in modifier.get("channels", {channel}):
                        continue
                    start = max(center - 3, 0)
                    stop = min(center + 4, settings.history_days)
                    dau[start:stop] *= modifier.get("dau", 1.0)
                    signup_rate[start:stop] *= modifier.get("signup", 1.0)
                    activation_rate[start:stop] = np.clip(activation_rate[start:stop] * modifier.get("activation", 1.0), 0.2, 0.9)
                    churn_rate[start:stop] = np.clip(churn_rate[start:stop] * modifier.get("churn", 1.0), 0.005, 0.16)
                    arpu[start:stop] *= modifier.get("arpu", 1.0)
                    mrr[start:stop] *= modifier.get("mrr", 1.0)
                    pipeline_created[start:stop] *= modifier.get("pipeline", 1.0)
                    pipeline_won[start:stop] *= modifier.get("conversion", 1.0)
                    feature_a[start:stop] = np.clip(feature_a[start:stop] * modifier.get("feature_a", 1.0), 0.05, 0.9)
                    retention_d30[start:stop] = np.clip(retention_d30[start:stop] * modifier.get("retention", 1.0), 0.08, 0.65)
                    nps[start:stop] = np.clip(nps[start:stop] + modifier.get("nps", 0), 0, 100)
                    anomaly_label[start:stop] = modifier["label"]

                new_signups = np.clip(dau * signup_rate, 1, None)
                activated_users = new_signups * activation_rate
                paid_conversions = activated_users * np.clip(0.18 + 0.11 * channel_cfg["conversion"] + 0.04 * (segment == "Enterprise"), 0.16, 0.47)
                churned_accounts = paid_accounts * churn_rate
                nrr = np.clip(1 + (expansion_mrr - contraction_mrr) / np.maximum(mrr, 1), 0.82, 1.18)

                for i, metric_date in enumerate(dates):
                    records.append(
                        {
                            "date": metric_date,
                            "segment": segment,
                            "region": region,
                            "acquisition_channel": channel,
                            "dau": int(round(dau[i])),
                            "mau": int(round(mau[i])),
                            "new_signups": int(round(new_signups[i])),
                            "activated_users": int(round(activated_users[i])),
                            "paid_conversions": int(round(paid_conversions[i])),
                            "paid_accounts": float(paid_accounts[i]),
                            "churned_accounts": float(churned_accounts[i]),
                            "activation_rate": float(activation_rate[i]),
                            "trial_to_paid_rate": float(paid_conversions[i] / max(new_signups[i], 1)),
                            "churn_rate": float(churn_rate[i]),
                            "arpu": float(arpu[i]),
                            "mrr": float(mrr[i]),
                            "expansion_mrr": float(expansion_mrr[i]),
                            "contraction_mrr": float(contraction_mrr[i]),
                            "pipeline_created": float(pipeline_created[i]),
                            "pipeline_won": float(pipeline_won[i]),
                            "cac": float(cac[i]),
                            "net_revenue_retention": float(nrr[i]),
                            "feature_a_adoption": float(feature_a[i]),
                            "feature_b_adoption": float(feature_b[i]),
                            "ab_variant_b_share": float(base_variant_b[i]),
                            "nps": float(nps[i]),
                            "retention_d7": float(retention_d7[i]),
                            "retention_d30": float(retention_d30[i]),
                            "revenue_at_risk": float(mrr[i] * churn_rate[i]),
                            "injected_anomaly": anomaly_label[i],
                        }
                    )

    df = pd.DataFrame(records)
    df["weekday"] = df["date"].dt.day_name()
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time
    return df


def aggregate_segment_metrics(segment_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment-level mock metrics into daily executive KPI streams."""

    grouped = segment_metrics.groupby("date", as_index=False)
    sums = grouped[
        [
            "dau",
            "mau",
            "new_signups",
            "activated_users",
            "paid_conversions",
            "paid_accounts",
            "churned_accounts",
            "mrr",
            "expansion_mrr",
            "contraction_mrr",
            "pipeline_created",
            "pipeline_won",
            "revenue_at_risk",
        ]
    ].sum()
    weighted_columns = ["feature_a_adoption", "feature_b_adoption", "ab_variant_b_share", "nps", "retention_d7", "retention_d30", "cac"]
    weighted = []
    for metric_date, group in segment_metrics.groupby("date"):
        dau_weight = np.maximum(group["dau"], 1)
        signup_weight = np.maximum(group["new_signups"], 1)
        weighted.append(
            {
                "date": metric_date,
                **{column: float(np.average(group[column], weights=dau_weight if column != "cac" else signup_weight)) for column in weighted_columns},
                "injected_anomaly": ", ".join(sorted({value for value in group["injected_anomaly"] if value})),
            }
        )
    df = sums.merge(pd.DataFrame(weighted), on="date", how="left")
    df["activation_rate"] = df["activated_users"] / df["new_signups"].clip(lower=1)
    df["trial_to_paid_rate"] = df["paid_conversions"] / df["new_signups"].clip(lower=1)
    df["churn_rate"] = df["churned_accounts"] / df["paid_accounts"].clip(lower=1)
    df["arpu"] = df["mrr"] / df["paid_accounts"].clip(lower=1)
    df["net_revenue_retention"] = 1 + (df["expansion_mrr"] - df["contraction_mrr"]) / df["mrr"].clip(lower=1)
    df["weekday"] = df["date"].dt.day_name()
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time
    return df.sort_values("date").reset_index(drop=True)


def generate_daily_metrics(settings: Settings | None = None) -> pd.DataFrame:
    """Generate aggregate daily SaaS KPI metrics with injected business events."""

    return aggregate_segment_metrics(generate_segment_metrics(settings))


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
    segment_metrics = generate_segment_metrics(settings)
    return SyntheticDataset(
        daily_metrics=aggregate_segment_metrics(segment_metrics),
        segment_metrics=segment_metrics,
        churn_training=generate_churn_training_data(settings),
        retention_cohorts=generate_retention_cohorts(settings),
        event_log=generate_event_log(settings),
    )
