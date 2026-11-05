"""Loading and cleaning.

Data-quality issues found during exploration (see report):

* Negative weights (~0.6% of rows): same rate behaviour as the positive value,
  so they are sign errors -> use the absolute value.
* Missing weights (~0.6%): left as NaN; LightGBM handles missing values natively.
* Missing market_index (~0.8%): irrelevant, because the model uses the *daily*
  market index (mean over all loads that day), not the per-load value.
* Label outliers in posted_rate (~0.3%): rates around 1/6x or 2.5-6x the normal
  rate for the lane. They cannot be predicted from the inputs, so they are
  removed from training only (never from the data we predict).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Rate-per-mile bounds for obvious label errors. Normal loads sit between
# ~1.6 and ~3.2 $/mile (1st-99th percentile).
MIN_RATE_PER_MILE = 1.0
MAX_RATE_PER_MILE = 6.0


def load_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / name)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def fix_weight(weight: pd.Series) -> pd.Series:
    """Negative weights are sign errors; zero is treated as missing."""
    fixed = weight.abs()
    return fixed.where(fixed > 0)


@dataclass
class CleaningReport:
    rows_in: int
    negative_weight: int
    missing_weight: int
    missing_market_index: int
    rate_per_mile_outliers: int
    residual_outliers: int = 0

    @property
    def rows_used(self) -> int:
        return self.rows_in - self.rate_per_mile_outliers - self.residual_outliers

    def as_dict(self) -> dict:
        return {**self.__dict__, "rows_used": self.rows_used}


def clean_training(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove obviously wrong labels and fix weights. Returns cleaned copy + report."""
    rate_per_mile = frame["posted_rate"] / frame["distance"]
    bad_label = (rate_per_mile < MIN_RATE_PER_MILE) | (rate_per_mile > MAX_RATE_PER_MILE)
    report = CleaningReport(
        rows_in=len(frame),
        negative_weight=int((frame["weight"] < 0).sum()),
        missing_weight=int(frame["weight"].isna().sum()),
        missing_market_index=int(frame["market_index"].isna().sum()),
        rate_per_mile_outliers=int(bad_label.sum()),
    )
    cleaned = frame.loc[~bad_label].copy()
    cleaned["weight"] = fix_weight(cleaned["weight"])
    return cleaned, report


def prepare_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Same weight fix for rows we predict (validation / December)."""
    prepared = frame.copy()
    prepared["weight"] = fix_weight(prepared["weight"])
    return prepared


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per calendar day across all loads.

    The per-load value is the daily national index plus small noise (within-day
    std ~0.025 vs ~0.17 overall), so the daily mean is the cleaner signal. It is
    available for every day from Jan to Dec because validation.csv contains it.
    """
    stacked = pd.concat([f[["date", "market_index"]] for f in frames])
    return stacked.groupby("date")["market_index"].mean().sort_index()


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Lookup table city -> (lat, lon), used to fill the December chart inputs."""
    parts = []
    for f in frames:
        for side in ("pickup", "delivery"):
            parts.append(
                f[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
            )
    table = pd.concat(parts).drop_duplicates("city").set_index("city")
    return table


def rate_per_mile(frame: pd.DataFrame) -> pd.Series:
    return frame["posted_rate"] / frame["distance"]


def log_rate(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame["posted_rate"].to_numpy())
