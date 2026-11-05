"""Load-level features for the gradient-boosting stage.

Cities are represented by their coordinates instead of IDs: validation.csv has
8 cities that never appear in training, and coordinates let the model place
them correctly (simulated unseen-city MAPE 2.15% vs 1.82% for seen cities).
Date features are deliberately NOT used here; the date effect is handled by the
linear daily-level stage, which can extrapolate to Nov-Dec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
FEATURES = [
    "distance",
    "great_circle_miles",
    "detour_ratio",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "equipment",
    "weight",
    "quote_signal",
]


def great_circle_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(h))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    unknown = set(frame["equipment"].dropna()) - set(EQUIPMENT_CODES)
    if unknown:
        raise ValueError(f"Unknown equipment types: {sorted(unknown)}")
    gc = great_circle_miles(
        frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"]
    )
    out = pd.DataFrame(index=frame.index)
    out["distance"] = frame["distance"]
    out["great_circle_miles"] = gc
    out["detour_ratio"] = frame["distance"] / np.maximum(gc, 1.0)
    for col in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        out[col] = frame[col]
    out["equipment"] = frame["equipment"].map(EQUIPMENT_CODES)
    out["weight"] = frame["weight"]
    out["quote_signal"] = frame["quote_signal"]
    return out[FEATURES]
