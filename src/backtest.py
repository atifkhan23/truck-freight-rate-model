"""Time-based validation of the approach.

The real task is a forecast: train on Jan-Oct 2025, predict Nov-Dec 2025. A random
split would leak the future into training and overstate accuracy, so every
evaluation here trains on earlier months and tests on the *next two* months.

Usage:  python -m src.backtest        (writes reports/metrics.json and figures)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freight.data import clean_training, daily_market_index, load_csv, rate_per_mile  # noqa: E402
from freight.features import build_features  # noqa: E402
from freight.model import LGBM_PARAMS, fit_robust  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FOLDS = [5, 6, 7, 8, 9]  # train on months < k, test on months k and k+1


def metrics(pred: np.ndarray, actual: pd.Series) -> dict:
    err = pred - actual.to_numpy()
    return {
        "MAPE_%": round(float(np.mean(np.abs(err) / actual) * 100), 3),
        "MAE_$": round(float(np.mean(np.abs(err))), 1),
        "RMSE_$": round(float(np.sqrt(np.mean(err ** 2))), 1),
    }


def baseline_equipment_rpm(train, test):
    rpm = rate_per_mile(train).groupby(train["equipment"]).median()
    return test["distance"] * test["equipment"].map(rpm)


def baseline_lane_rpm(train, test):
    lane_rpm = rate_per_mile(train).groupby([train["pickup"], train["delivery"], train["equipment"]]).median()
    equip_rpm = rate_per_mile(train).groupby(train["equipment"]).median()
    key = pd.MultiIndex.from_frame(test[["pickup", "delivery", "equipment"]])
    rpm = pd.Series(lane_rpm.reindex(key).to_numpy(), index=test.index)
    return test["distance"] * rpm.fillna(test["equipment"].map(equip_rpm))


def single_gbm(train, test, with_market_index: bool):
    Xa, Xb = build_features(train), build_features(test)
    if with_market_index:
        Xa["market_index"], Xb["market_index"] = train["market_index"], test["market_index"]
    model = lgb.LGBMRegressor(**LGBM_PARAMS).fit(Xa, np.log(train["posted_rate"]))
    return np.exp(model.predict(Xb))


def main() -> None:
    raw = load_csv("train_test.csv")
    validation = load_csv("validation.csv")
    data, report = clean_training(raw)
    mi_by_day = daily_market_index(raw, validation)
    data["month"] = data["date"].dt.month

    methods = {
        "Baseline: equipment $/mile x distance": baseline_equipment_rpm,
        "Baseline: lane median $/mile x distance": baseline_lane_rpm,
        "LightGBM, no date signal": lambda a, b: single_gbm(a, b, False),
        "LightGBM + raw market_index": lambda a, b: single_gbm(a, b, True),
        "Two-stage (final)": lambda a, b: fit_robust(a, mi_by_day)[0].predict(b),
    }
    rows = []
    for k in FOLDS:
        train, test = data[data.month < k], data[data.month.isin([k, k + 1])]
        for name, fn in methods.items():
            rows.append({"fold": f"train Jan-M{k - 1}, test M{k}-M{k + 1}", "method": name, **metrics(fn(train, test), test["posted_rate"])})
        print(f"fold {k} done", flush=True)
    table = pd.DataFrame(rows)
    summary = table.groupby("method", sort=False)[["MAPE_%", "MAE_$", "RMSE_$"]].mean().round(2)
    main_fold = table[table.fold.str.contains("M9-M10")].set_index("method")[["MAPE_%", "MAE_$", "RMSE_$"]]

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(
        json.dumps(
            {
                "cleaning": report.as_dict(),
                "rolling_backtest_mean": summary.to_dict(orient="index"),
                "holdout_train_jan_aug_test_sep_oct": main_fold.to_dict(orient="index"),
                "per_fold": rows,
            },
            indent=2,
        )
    )
    print(summary.to_string())
    print(main_fold.to_string())

    # Figure: backtest MAPE per method
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    s = summary["MAPE_%"].iloc[::-1]
    ax.barh(s.index, s.values, color=["#064A56" if "final" in i else "#9DB8BD" for i in s.index])
    for y, v in enumerate(s.values):
        ax.text(v + 0.1, y, f"{v:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("Mean MAPE over 5 rolling two-months-ahead folds (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "backtest_mape.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
