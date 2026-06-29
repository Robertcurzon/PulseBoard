"""CLI entry point for running the PulseBoard ML pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from llm.anomaly_narrator import narrate_anomaly
from llm.insight_generator import generate_executive_insight
from ml.anomaly_detector import AnomalyDetector
from ml.pipeline import run_ml_pipeline, weekly_kpi_summary


async def main() -> None:
    """Run the full PulseBoard pipeline and print key outputs."""

    settings = get_settings()
    result = run_ml_pipeline(settings)
    detector = AnomalyDetector(settings).fit(result.datasets.daily_metrics)
    records = detector.anomaly_records(result.scored_metrics, limit=3)
    insight = await generate_executive_insight(weekly_kpi_summary(result.datasets.daily_metrics), settings)

    print("PulseBoard pipeline complete")
    print(f"Rows: {len(result.datasets.daily_metrics):,}")
    print(f"Churn ROC AUC: {result.churn_auc:.3f}")
    print(f"DAU forecast engine: {result.forecasts['dau'].engine}")
    print(f"MRR forecast engine: {result.forecasts['mrr'].engine}")
    print("\nTop churn drivers:")
    for item in result.feature_importance[:5]:
        print(f"- {item.feature}: {item.importance:.3f} ({item.direction})")
    print("\nExecutive insight:")
    print(insight)
    print("\nAnomaly narratives:")
    for record in records:
        print(f"- {await narrate_anomaly(record, settings)}")


if __name__ == "__main__":
    asyncio.run(main())
