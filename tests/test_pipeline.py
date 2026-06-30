from config.settings import Settings
from data.generators.synthetic_data import aggregate_segment_metrics, generate_all
from ml.pipeline import run_ml_pipeline, weekly_kpi_summary


def test_demo_dataset_contains_showcase_slices() -> None:
    settings = Settings(history_days=90, random_seed=9)
    datasets = generate_all(settings)

    assert {"Enterprise", "Mid-Market", "Startup"}.issubset(set(datasets.segment_metrics["segment"]))
    assert {"North America", "EMEA", "APAC"}.issubset(set(datasets.segment_metrics["region"]))
    assert {"Product-Led", "Paid Search", "Partner"}.issubset(set(datasets.segment_metrics["acquisition_channel"]))
    assert len(datasets.event_log) >= 5
    assert {"pipeline_created", "net_revenue_retention", "trial_to_paid_rate"}.issubset(datasets.daily_metrics.columns)


def test_segment_aggregation_preserves_daily_metric_shape() -> None:
    settings = Settings(history_days=60, random_seed=10)
    datasets = generate_all(settings)
    subset = datasets.segment_metrics.loc[datasets.segment_metrics["segment"].isin(["Enterprise", "Mid-Market"])]
    aggregated = aggregate_segment_metrics(subset)

    assert len(aggregated) == 60
    assert aggregated["mrr"].gt(0).all()
    assert aggregated["trial_to_paid_rate"].between(0, 1).all()
    assert aggregated["net_revenue_retention"].between(0.8, 1.2).all()


def test_run_ml_pipeline_returns_core_outputs() -> None:
    settings = Settings(history_days=180, forecast_horizon_days=14, random_seed=11, anomaly_contamination=0.05)
    result = run_ml_pipeline(settings)

    assert len(result.datasets.daily_metrics) == 180
    assert len(result.datasets.segment_metrics) > len(result.datasets.daily_metrics)
    assert result.scored_metrics["is_anomaly"].sum() > 0
    assert set(result.forecasts) == {"dau", "mrr"}
    assert len(result.forecasts["dau"].forecast) == 14
    assert result.churn_auc > 0.7
    assert len(result.feature_importance) > 0


def test_weekly_kpi_summary_has_required_fields() -> None:
    settings = Settings(history_days=90, random_seed=15)
    result = run_ml_pipeline(settings)
    summary = weekly_kpi_summary(result.datasets.daily_metrics)

    assert summary["mrr"] > 0
    assert summary["dau"] > 0
    assert "feature_a_adoption" in summary
    assert "week_start" in summary
