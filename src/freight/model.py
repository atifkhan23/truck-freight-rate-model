"""Two-stage freight rate model.

log(rate) = load_effect(load features)  +  day_level(date)

* load_effect: LightGBM on lane geometry, distance, equipment, weight and quote.
* day_level:   a small linear model on the daily market index, a time trend and a
               quarter-end ramp. It is fitted on the median daily residual of the
               load model, so it captures the market level shared by all loads
               on a given day.

Why two stages: validation (Nov-Dec 2025) lies after all training data
(Jan-Oct 2025). Tree models cannot extrapolate in time and, when given the
market index directly, they over-fit its Jan-Aug relationship to rates (it
broke in Sep-Oct). A linear daily level extrapolates smoothly and is robust.
The two stages are fitted alternately for a few rounds (backfitting).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import build_features

TREND_ORIGIN = pd.Timestamp("2025-01-01")
QUARTER_END_MONTHS = (3, 6, 9, 12)
LGBM_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
DAY_TERMS = ["intercept", "market_index", "months_elapsed", "quarter_end", "month_progress", "quarter_end_x_progress"]


def day_design(dates: pd.Series | pd.DatetimeIndex, market_index_by_day: pd.Series) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    mi = market_index_by_day.reindex(dates).to_numpy()
    if np.isnan(mi).any():
        missing = sorted({str(d.date()) for d, m in zip(dates, mi) if np.isnan(m)})[:5]
        raise ValueError(f"No daily market index for dates such as {missing}")
    quarter_end = np.isin(dates.month, QUARTER_END_MONTHS).astype(float)
    progress = ((dates.day - 1) / (dates.days_in_month - 1)).to_numpy()
    return pd.DataFrame(
        {
            "intercept": 1.0,
            "market_index": mi,
            "months_elapsed": ((dates - TREND_ORIGIN).days / 30.4).to_numpy(),
            "quarter_end": quarter_end,
            "month_progress": progress,
            "quarter_end_x_progress": quarter_end * progress,
        },
        index=dates,
    )[DAY_TERMS]


@dataclass
class TwoStageRateModel:
    market_index_by_day: pd.Series
    rounds: int = 3
    params: dict = field(default_factory=lambda: dict(LGBM_PARAMS))
    load_model: lgb.LGBMRegressor | None = None
    day_coef: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, target_log_rate: np.ndarray) -> "TwoStageRateModel":
        X = build_features(frame)
        day_X = day_design(frame["date"], self.market_index_by_day).to_numpy()
        offset = np.zeros(len(frame))
        for _ in range(self.rounds):
            self.load_model = lgb.LGBMRegressor(**self.params).fit(
                X, target_log_rate - offset, categorical_feature=["equipment"]
            )
            residual = pd.Series(target_log_rate - self.load_model.predict(X), index=frame["date"].to_numpy())
            daily = residual.groupby(level=0).median()
            design = day_design(daily.index, self.market_index_by_day).to_numpy()
            self.day_coef, *_ = np.linalg.lstsq(design, daily.to_numpy(), rcond=None)
            offset = day_X @ self.day_coef
        return self

    def day_level(self, dates) -> np.ndarray:
        return day_design(dates, self.market_index_by_day).to_numpy() @ self.day_coef

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        if self.load_model is None:
            raise RuntimeError("Model is not fitted")
        return self.load_model.predict(build_features(frame)) + self.day_level(frame["date"])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(frame))

    def day_coefficients(self) -> dict:
        return dict(zip(DAY_TERMS, np.round(self.day_coef, 5)))


def fit_robust(frame: pd.DataFrame, market_index_by_day: pd.Series, max_abs_log_residual: float = 0.4):
    """Fit, drop training rows the model cannot explain at all (|log residual| > 0.4,
    i.e. off by more than ~50%), then refit. Catches label outliers that sit inside
    the rate-per-mile bounds. Returns (model, number of rows dropped)."""
    y = np.log(frame["posted_rate"].to_numpy())
    first = TwoStageRateModel(market_index_by_day).fit(frame, y)
    keep = np.abs(y - first.predict_log(frame)) <= max_abs_log_residual
    final = TwoStageRateModel(market_index_by_day).fit(frame.loc[keep], y[keep])
    return final, int((~keep).sum())
