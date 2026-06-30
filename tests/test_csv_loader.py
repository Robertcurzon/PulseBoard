from io import BytesIO

from data.ingestion.csv_loader import load_uploaded_csv


def test_load_uploaded_csv_with_minimal_kpis() -> None:
    csv_bytes = b"""date,segment,region,channel,dau,mrr,churn_rate,nps
2026-01-01,Enterprise,North America,Partner,4200,710000,0.018,58
2026-01-02,Enterprise,North America,Partner,4350,719500,0.017,59
2026-01-03,Mid-Market,EMEA,Product-Led,3100,285000,0.031,48
"""
    uploaded = load_uploaded_csv(BytesIO(csv_bytes))

    assert len(uploaded.daily_metrics) == 3
    assert {"segment", "region", "acquisition_channel"}.issubset(uploaded.segment_metrics.columns)
    assert uploaded.daily_metrics["mrr"].gt(0).all()
    assert uploaded.validation_messages


def test_load_uploaded_csv_rejects_missing_date() -> None:
    csv_bytes = b"""segment,dau,mrr
Enterprise,4200,710000
"""
    try:
        load_uploaded_csv(BytesIO(csv_bytes))
    except ValueError as exc:
        assert "Missing required column" in str(exc)
    else:
        raise AssertionError("Expected missing-date CSV to fail validation.")
