"""CSV ingestion and normalization for user-supplied PulseBoard data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import IO

import numpy as np
import pandas as pd

from data.generators.synthetic_data import aggregate_segment_metrics


REQUIRED_COLUMNS = {"date"}
USEFUL_KPI_COLUMNS = {"dau", "mrr", "new_signups", "churn_rate", "nps"}
DIMENSION_DEFAULTS = {
    "segment": "Uploaded",
    "region": "All Regions",
    "acquisition_channel": "Uploaded",
}
NUMERIC_DEFAULTS = {
    "dau": 0.0,
    "mau": 0.0,
    "new_signups": 0.0,
    "activated_users": 0.0,
    "paid_conversions": 0.0,
    "paid_accounts": 0.0,
    "churned_accounts": 0.0,
    "activation_rate": 0.0,
    "trial_to_paid_rate": 0.0,
    "churn_rate": 0.0,
    "arpu": 0.0,
    "mrr": 0.0,
    "expansion_mrr": 0.0,
    "contraction_mrr": 0.0,
    "pipeline_created": 0.0,
    "pipeline_won": 0.0,
    "cac": 0.0,
    "net_revenue_retention": 1.0,
    "feature_a_adoption": 0.0,
    "feature_b_adoption": 0.0,
    "ab_variant_b_share": 0.0,
    "nps": 0.0,
    "retention_d7": 0.0,
    "retention_d30": 0.0,
    "revenue_at_risk": 0.0,
}


@dataclass(frozen=True)
class UploadedDataset:
    """Normalized data and validation notes from an uploaded CSV."""

    daily_metrics: pd.DataFrame
    segment_metrics: pd.DataFrame
    event_log: pd.DataFrame
    validation_messages: list[str]


def normalize_column_name(column: str) -> str:
    """Normalize external CSV headers to PulseBoard snake_case names."""

    aliases = {
        "day": "date",
        "period": "date",
        "customer_segment": "segment",
        "market_segment": "segment",
        "geo": "region",
        "country_region": "region",
        "channel": "acquisition_channel",
        "source": "acquisition_channel",
        "revenue": "mrr",
        "monthly_recurring_revenue": "mrr",
        "active_users": "dau",
        "daily_active_users": "dau",
        "signup": "new_signups",
        "signups": "new_signups",
        "trial_to_paid": "trial_to_paid_rate",
        "conversion_rate": "trial_to_paid_rate",
        "nrr": "net_revenue_retention",
        "feature_a": "feature_a_adoption",
        "feature_b": "feature_b_adoption",
    }
    normalized = column.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = "".join(char for char in normalized if char.isalnum() or char == "_")
    return aliases.get(normalized, normalized)


def load_uploaded_csv(file: IO[bytes] | str) -> UploadedDataset:
    """Read, validate, and normalize a PulseBoard-compatible CSV file."""

    raw = pd.read_csv(file)
    if raw.empty:
        raise ValueError("The uploaded CSV is empty.")

    df = raw.rename(columns={column: normalize_column_name(column) for column in raw.columns}).copy()
    missing_required = REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing_required))}.")

    present_kpis = USEFUL_KPI_COLUMNS & set(df.columns)
    if not present_kpis:
        raise ValueError("Include at least one useful KPI column such as dau, mrr, new_signups, churn_rate, or nps.")

    messages: list[str] = []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        messages.append(f"Dropped {invalid_dates} row(s) with invalid dates.")
    df = df.dropna(subset=["date"])
    if df.empty:
        raise ValueError("No valid dated rows remained after parsing the date column.")

    for column, default in DIMENSION_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
            messages.append(f"Column '{column}' was not provided; using '{default}'.")
        df[column] = df[column].fillna(default).astype(str)

    for column, default in NUMERIC_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)

    df = _derive_operating_metrics(df)
    df["injected_anomaly"] = df.get("event", "").fillna("").astype(str) if "event" in df.columns else ""
    df["weekday"] = df["date"].dt.day_name()
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time

    segment_metrics = (
        df.groupby(["date", "segment", "region", "acquisition_channel"], as_index=False)
        .agg(_aggregation_spec())
        .sort_values(["date", "segment", "region", "acquisition_channel"])
        .reset_index(drop=True)
    )
    segment_metrics = _derive_operating_metrics(segment_metrics)
    daily_metrics = aggregate_segment_metrics(segment_metrics)
    event_log = _extract_event_log(df)
    messages.extend(_quality_messages(daily_metrics, segment_metrics))

    return UploadedDataset(
        daily_metrics=daily_metrics,
        segment_metrics=segment_metrics,
        event_log=event_log,
        validation_messages=messages,
    )


def _derive_operating_metrics(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()
    if (derived["mau"] <= 0).all() and "dau" in derived:
        derived["mau"] = derived.groupby(["segment", "region", "acquisition_channel"], dropna=False)["dau"].transform(
            lambda series: series.rolling(30, min_periods=1).sum() * 0.18
        )
    if (derived["paid_accounts"] <= 0).all():
        derived["paid_accounts"] = np.maximum(derived["mau"] * 0.12, derived["paid_conversions"])
    if (derived["activated_users"] <= 0).all():
        derived["activated_users"] = derived["new_signups"] * derived["activation_rate"].replace(0, 0.58)
    if (derived["paid_conversions"] <= 0).all():
        derived["paid_conversions"] = derived["new_signups"] * derived["trial_to_paid_rate"].replace(0, 0.22)
    if (derived["churned_accounts"] <= 0).all():
        derived["churned_accounts"] = derived["paid_accounts"] * derived["churn_rate"]
    if (derived["arpu"] <= 0).all():
        derived["arpu"] = derived["mrr"] / derived["paid_accounts"].clip(lower=1)
    if (derived["pipeline_created"] <= 0).all():
        derived["pipeline_created"] = derived["mrr"] * 0.28
    if (derived["pipeline_won"] <= 0).all():
        derived["pipeline_won"] = derived["pipeline_created"] * 0.31
    if (derived["expansion_mrr"] <= 0).all():
        derived["expansion_mrr"] = derived["mrr"] * 0.018
    if (derived["contraction_mrr"] <= 0).all():
        derived["contraction_mrr"] = derived["mrr"] * np.maximum(derived["churn_rate"], 0.012)
    derived["activation_rate"] = np.where(
        derived["activation_rate"] > 0,
        derived["activation_rate"],
        derived["activated_users"] / derived["new_signups"].clip(lower=1),
    )
    derived["trial_to_paid_rate"] = np.where(
        derived["trial_to_paid_rate"] > 0,
        derived["trial_to_paid_rate"],
        derived["paid_conversions"] / derived["new_signups"].clip(lower=1),
    )
    derived["churn_rate"] = np.where(
        derived["churn_rate"] > 0,
        derived["churn_rate"],
        derived["churned_accounts"] / derived["paid_accounts"].clip(lower=1),
    )
    derived["net_revenue_retention"] = np.where(
        derived["net_revenue_retention"] != 1.0,
        derived["net_revenue_retention"],
        1 + (derived["expansion_mrr"] - derived["contraction_mrr"]) / derived["mrr"].clip(lower=1),
    )
    derived["revenue_at_risk"] = np.where(
        derived["revenue_at_risk"] > 0,
        derived["revenue_at_risk"],
        derived["mrr"] * derived["churn_rate"],
    )
    return derived


def _aggregation_spec() -> dict[str, str]:
    sum_columns = {
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
    }
    mean_columns = set(NUMERIC_DEFAULTS) - sum_columns
    spec = {column: "sum" for column in sum_columns}
    spec.update({column: "mean" for column in mean_columns})
    spec["injected_anomaly"] = lambda values: ", ".join(sorted({str(value) for value in values if str(value)}))
    return spec


def _extract_event_log(df: pd.DataFrame) -> pd.DataFrame:
    if "event" not in df.columns:
        return pd.DataFrame(columns=["date", "event", "category", "description"])
    event_rows = df.loc[df["event"].fillna("").astype(str).str.len() > 0].copy()
    if event_rows.empty:
        return pd.DataFrame(columns=["date", "event", "category", "description"])
    event_rows["category"] = event_rows.get("event_category", "Uploaded Event")
    event_rows["description"] = event_rows.get("event_description", event_rows["event"])
    return event_rows[["date", "event", "category", "description"]].drop_duplicates().sort_values("date")


def _quality_messages(daily_metrics: pd.DataFrame, segment_metrics: pd.DataFrame) -> list[str]:
    messages = [
        f"Loaded {len(daily_metrics):,} daily period(s) across {segment_metrics['segment'].nunique()} segment(s), "
        f"{segment_metrics['region'].nunique()} region(s), and {segment_metrics['acquisition_channel'].nunique()} channel(s)."
    ]
    missing_mrr_share = float((daily_metrics["mrr"] <= 0).mean())
    if missing_mrr_share > 0:
        messages.append(f"{missing_mrr_share:.0%} of daily rows have zero MRR; revenue panels may be less informative.")
    if len(daily_metrics) < 45:
        messages.append("Forecasting works best with at least 45 daily rows; upload more history for steadier intervals.")
    return messages
