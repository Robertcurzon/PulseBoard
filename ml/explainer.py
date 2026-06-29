"""SHAP feature explanations for churn prediction."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

cache_root = Path(tempfile.gettempdir()) / "pulseboard-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import shap
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class FeatureImportance:
    """A single model feature attribution."""

    feature: str
    importance: float
    direction: str


class ChurnExplainer:
    """Compute SHAP-based global feature importance for a fitted churn model."""

    def __init__(self, model: Pipeline, feature_columns: list[str]) -> None:
        self.model = model
        self.feature_columns = feature_columns

    def top_features(self, samples: pd.DataFrame, limit: int = 8) -> list[FeatureImportance]:
        """Return top churn drivers by mean absolute SHAP value."""

        x = samples[self.feature_columns].copy()
        preprocessor = self.model.named_steps["preprocessor"]
        classifier = self.model.named_steps["classifier"]
        transformed = preprocessor.transform(x)
        feature_names = list(preprocessor.get_feature_names_out())

        background = transformed[: min(100, transformed.shape[0])]
        explainer = shap.LinearExplainer(classifier, background)
        shap_values = explainer.shap_values(transformed)
        values = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(values).mean(axis=0)
        coefficients = classifier.coef_[0]
        order = np.argsort(mean_abs)[::-1][:limit]

        return [
            FeatureImportance(
                feature=feature_names[i].replace("num__", "").replace("cat__", ""),
                importance=float(mean_abs[i]),
                direction="increases churn risk" if coefficients[i] > 0 else "reduces churn risk",
            )
            for i in order
        ]
