from config.settings import Settings
from data.generators.synthetic_data import generate_daily_metrics
from ml.anomaly_detector import AnomalyDetector


def test_anomaly_detector_flags_records() -> None:
    settings = Settings(history_days=365, anomaly_contamination=0.04, random_seed=42)
    metrics = generate_daily_metrics(settings)
    scored = AnomalyDetector(settings).fit_predict(metrics)

    assert "anomaly_score" in scored.columns
    assert "is_anomaly" in scored.columns
    assert scored["is_anomaly"].sum() > 0
    assert scored.loc[scored["is_anomaly"], "primary_anomaly_metric"].notna().all()


def test_anomaly_records_are_llm_ready() -> None:
    settings = Settings(history_days=365, anomaly_contamination=0.04, random_seed=42)
    metrics = generate_daily_metrics(settings)
    detector = AnomalyDetector(settings)
    scored = detector.fit_predict(metrics)
    records = detector.anomaly_records(scored, limit=3)

    assert len(records) > 0
    assert {"date", "metric", "score", "severity", "context"}.issubset(records[0])
