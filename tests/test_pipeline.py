from config.settings import Settings
from ml.pipeline import run_ml_pipeline, weekly_kpi_summary


def test_run_ml_pipeline_returns_core_outputs() -> None:
    settings = Settings(history_days=180, forecast_horizon_days=14, random_seed=11, anomaly_contamination=0.05)
    result = run_ml_pipeline(settings)

    assert len(result.datasets.daily_metrics) == 180
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
