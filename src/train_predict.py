"""Train on all labelled data (Jan-Oct 2025) and write both prediction files.

Usage:  python -m src.train_predict

Outputs
  validation_predictions.csv        load_id,predicted_rate for the 12,000 loads
  data/december_chart_inputs.csv    the same file with predicted_rate filled in
  reports/*.png, reports/model_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freight.data import (  # noqa: E402
    DATA_DIR,
    city_coordinates,
    clean_training,
    daily_market_index,
    load_csv,
    prepare_inputs,
    rate_per_mile,
)
from freight.model import fit_robust  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DECEMBER_FILE = DATA_DIR / "december_chart_inputs.csv"
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def december_frame(december: pd.DataFrame, coords: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """The December file only has city names, distance, equipment, weight and date.
    Fill what the model needs: coordinates from the city table, and quote_signal
    with the median for that equipment in Nov-Dec (constant across the 31 days,
    so it sets the level but not the shape of the curve)."""
    frame = december.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for side in ("pickup", "delivery"):
        missing = set(frame[side]) - set(coords.index)
        if missing:
            raise ValueError(f"No coordinates for {sorted(missing)}")
        frame[f"{side}_lat"] = frame[side].map(coords["lat"])
        frame[f"{side}_lon"] = frame[side].map(coords["lon"])
    quote = validation.groupby("equipment")["quote_signal"].median()
    frame["quote_signal"] = frame["equipment"].map(quote)
    return frame


def main() -> None:
    raw = load_csv("train_test.csv")
    validation = prepare_inputs(load_csv("validation.csv"))
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    december_raw = pd.read_csv(DECEMBER_FILE)

    train, report = clean_training(raw)
    mi_by_day = daily_market_index(raw, validation)
    model, dropped = fit_robust(train, mi_by_day)
    report.residual_outliers = dropped
    print("cleaning:", report.as_dict())
    print("daily level coefficients:", model.day_coefficients())

    # Validation predictions, in the template's order
    preds = pd.Series(model.predict(validation), index=validation["load_id"])
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(preds).round(2)
    if out["predicted_rate"].isna().any() or (out["predicted_rate"] <= 0).any():
        raise SystemExit("Some validation loads have no valid prediction")
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # December chart: fill predicted_rate, keep the original seven columns
    dec = december_frame(december_raw, city_coordinates(raw, validation), validation)
    filled = december_raw.copy()
    filled["predicted_rate"] = np.round(model.predict(dec), 2)
    filled[DECEMBER_COLUMNS].to_csv(DECEMBER_FILE, index=False)

    REPORTS.mkdir(exist_ok=True)
    summary = {
        "cleaning": report.as_dict(),
        "day_level_coefficients": model.day_coefficients(),
        "validation_prediction_stats": out["predicted_rate"].describe().round(2).to_dict(),
        "december_range_$": [float(filled.predicted_rate.min()), float(filled.predicted_rate.max())],
    }
    (REPORTS / "model_summary.json").write_text(json.dumps(summary, indent=2))
    save_figures(raw, train, model, mi_by_day)
    print(json.dumps(summary, indent=2))


def save_figures(raw, train, model, mi_by_day) -> None:
    # 1. Market level over time: daily median $/mile vs the fitted daily level
    rpm = rate_per_mile(train).groupby(train["date"]).median()
    days = pd.date_range("2025-01-01", "2025-12-31")
    level = pd.Series(model.day_level(days), index=days)
    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=160)
    ax1.plot(rpm.index, rpm.values, color="#9DB8BD", lw=1, label="Daily median $/mile (training data)")
    ax1.set_ylabel("$/mile")
    ax2 = ax1.twinx()
    ax2.plot(level.index, np.exp(level.values), color="#064A56", lw=2, label="Fitted market level (model)")
    ax2.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color="#F2B705", alpha=0.12, label="Forecast period")
    ax2.set_ylabel("Relative rate level")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper left", fontsize=8, frameon=False)
    ax1.spines[["top"]].set_visible(False)
    ax1.set_title("Market level: training data and the model's daily level (extrapolated to Nov-Dec)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "market_level.png")
    plt.close(fig)

    # 2. Data-quality: rate per mile distribution showing label outliers
    all_rpm = rate_per_mile(raw)
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
    ax.hist(np.log10(all_rpm), bins=150, color="#064A56")
    ax.set_yscale("log")
    ax.axvline(np.log10(1.0), color="#C0392B", ls="--", lw=1)
    ax.axvline(np.log10(6.0), color="#C0392B", ls="--", lw=1)
    ticks = [0.3, 0.5, 1, 2, 3, 6, 10, 15]
    ax.set_xticks(np.log10(ticks), [str(t) for t in ticks])
    ax.set_xlabel("posted_rate / distance ($/mile, log scale); red lines = cleaning bounds")
    ax.set_ylabel("loads (log scale)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "rate_per_mile_outliers.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
