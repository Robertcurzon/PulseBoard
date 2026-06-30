"""Time-series forecasting utilities for PulseBoard."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config.settings import Settings, get_settings


@dataclass
class ForecastResult:
    """Forecast output in a chart-ready format."""

    metric: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    engine: str


class ProphetForecaster:
    """Prophet-first forecaster with a statsmodels fallback for lightweight installs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def forecast(self, metrics: pd.DataFrame, metric: str, horizon_days: int | None = None) -> ForecastResult:
        """Forecast a KPI for the configured horizon."""

        horizon_days = horizon_days or self.settings.forecast_horizon_days
        history = metrics[["date", metric]].rename(columns={"date": "ds", metric: "y"}).copy()
        history["ds"] = pd.to_datetime(history["ds"])
        history = history.sort_values("ds")
        if len(history) < 14 or history["y"].nunique() < 2:
            return self._forecast_with_naive(history, metric, horizon_days)

        try:
            return self._forecast_with_prophet(history, metric, horizon_days)
        except Exception:
            return self._forecast_with_statsmodels(history, metric, horizon_days)

    def _forecast_with_prophet(self, history: pd.DataFrame, metric: str, horizon_days: int) -> ForecastResult:
        cache_root = Path(tempfile.gettempdir()) / "pulseboard-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

        from prophet import Prophet

        model = Prophet(
            interval_width=0.8,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.08,
        )
        model.fit(history)
        future = model.make_future_dataframe(periods=horizon_days, freq="D", include_history=False)
        raw = model.predict(future)
        forecast = raw[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
        forecast["metric"] = metric
        return ForecastResult(
            metric=metric,
            history=history.rename(columns={"ds": "date", "y": "actual"}),
            forecast=forecast,
            engine="prophet",
        )

    def _forecast_with_statsmodels(self, history: pd.DataFrame, metric: str, horizon_days: int) -> ForecastResult:
        try:
            series = history.set_index("ds")["y"].asfreq("D").interpolate()
            model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated")
            fitted = model.fit(optimized=True)
            predicted = fitted.forecast(horizon_days)
            resid_std = float(np.std(fitted.resid.dropna())) or float(series.std() * 0.05)
        except Exception:
            return self._forecast_with_naive(history, metric, horizon_days)
        forecast = pd.DataFrame(
            {
                "date": predicted.index,
                "yhat": predicted.values,
                "yhat_lower": predicted.values - 1.28 * resid_std,
                "yhat_upper": predicted.values + 1.28 * resid_std,
                "metric": metric,
            }
        )
        return ForecastResult(
            metric=metric,
            history=history.rename(columns={"ds": "date", "y": "actual"}),
            forecast=forecast,
            engine="statsmodels",
        )

    def _forecast_with_naive(self, history: pd.DataFrame, metric: str, horizon_days: int) -> ForecastResult:
        """Return a robust baseline forecast for short or low-variance uploads."""

        ordered = history.sort_values("ds")
        last_date = pd.Timestamp(ordered["ds"].max())
        recent = ordered["y"].tail(min(7, len(ordered)))
        level = float(recent.mean()) if len(recent) else 0.0
        resid_std = float(recent.std()) if len(recent) > 1 else abs(level) * 0.05
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        forecast = pd.DataFrame(
            {
                "date": future_dates,
                "yhat": level,
                "yhat_lower": level - 1.28 * resid_std,
                "yhat_upper": level + 1.28 * resid_std,
                "metric": metric,
            }
        )
        return ForecastResult(
            metric=metric,
            history=ordered.rename(columns={"ds": "date", "y": "actual"}),
            forecast=forecast,
            engine="naive-baseline",
        )


def forecast_metrics(metrics: pd.DataFrame, settings: Settings | None = None) -> dict[str, ForecastResult]:
    """Forecast DAU and MRR for dashboard rendering."""

    forecaster = ProphetForecaster(settings)
    return {
        "dau": forecaster.forecast(metrics, "dau"),
        "mrr": forecaster.forecast(metrics, "mrr"),
    }
