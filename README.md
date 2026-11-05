# Freight Rate Prediction

Predicts the `posted_rate` of truck loads for the Spotter Machine Learning Engineer assessment.

The labelled data covers **Jan–Oct 2025** and the 12,000 loads to predict are from **Nov–Dec 2025**, so this is a forecasting problem. All validation is time-based: the model is always trained on earlier months and tested on the following two months.

**Result:** a two-stage model (LightGBM for the load + a linear daily market level) with **2.23% MAPE** in a rolling two-months-ahead backtest, against 2.91% for a plain LightGBM and 5.00% for a lane-average baseline.

- Report (approach, data split, findings, December chart): [`reports/Freight_Rate_Model_Report.docx`](reports/Freight_Rate_Model_Report.docx)
- Final predictions: [`validation_predictions.csv`](validation_predictions.csv)
- December chart: [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png)
- Loom walkthrough: _add link_

## Run it

Python 3.10+.

```bash
python -m pip install -r requirements.txt

# 1. Train on all labelled data and write both prediction files (about 6 minutes)
python -m src.train_predict

# 2. Validate the files with Spotter's scorer and draw the December chart
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# 3. Optional: reproduce the time-based backtest (about 25 minutes)
python -m src.backtest
```

`src.train_predict` writes `validation_predictions.csv`, fills `predicted_rate` in `data/december_chart_inputs.csv`, and saves figures and `reports/model_summary.json`. `src.backtest` writes `reports/metrics.json` and `reports/backtest_mape.png`.

## Approach

**Data-quality fixes** (`src/freight/data.py`)

| Issue | Handling |
|---|---|
| Label outliers: rate per mile < $1 or > $6 (about 1/6× or 2.5–6× normal) | Removed from training only |
| Labels the first model misses by more than ~50% | Removed from training only, then refit |
| Negative weights (sign errors) | Absolute value |
| Missing weight | Left missing (LightGBM handles it) |
| Missing `market_index` | Not needed: the daily mean is used |
| 8 cities in validation that never appear in training | Cities described by coordinates, not IDs |

**Model** (`src/freight/model.py`)

```
log(rate) = LightGBM(distance, great-circle miles, detour ratio, coordinates, equipment, weight, quote_signal)
          + linear daily level(daily market index, months elapsed, quarter-end month, month progress)
```

- Trees cannot extrapolate in time, and when given `market_index` directly they memorise its Jan–Aug relationship with price, which broke in Sep–Oct. A small linear daily level extrapolates smoothly into Nov–Dec.
- The daily level also captures a **quarter-end effect**: rates rise through March, June and September and drop at the start of the next month. December is a quarter-end month.
- The two stages are fitted alternately for three rounds (backfitting).

**December chart:** the December file has only city names, distance, equipment, weight and date. Coordinates come from the city table, the daily market index from the December loads in `validation.csv`, and `quote_signal` is the Nov–Dec Dry Van median (constant, so it sets the level but not the shape).

## Backtest (time-based)

| Method | Rolling MAPE | Rolling MAE | Sep–Oct MAPE | Sep–Oct MAE |
|---|---|---|---|---|
| Baseline: equipment $/mile x distance | 8.44% | $176 | 8.20% | $184 |
| Baseline: lane median $/mile x distance | 5.00% | $123 | 3.89% | $98 |
| LightGBM, no date signal | 2.91% | $75 | 1.77% | $49 |
| LightGBM + raw market_index | 2.98% | $76 | 3.56% | $90 |
| **Two-stage (final)** | 2.23% | $58 | 1.53% | $42 |

Rolling = mean over 5 folds (train on months before k, test on months k and k+1, k = May…Sep). Sep–Oct = train Jan–Aug, test Sep–Oct, the closest copy of the real task. Errors are on test rows inside the rate-per-mile bounds.

## Project structure

```
data/                      assessment data (december_chart_inputs.csv is filled by the pipeline)
docs/                      assessment instructions
src/freight/data.py        loading, cleaning, daily market index, city coordinates
src/freight/features.py    load-level features
src/freight/model.py       two-stage model
src/train_predict.py       final training and prediction files
src/backtest.py            rolling time-based validation
reports/                   report, metrics and figures
score.py                   Spotter's scorer (unchanged)
validation_predictions.csv final predictions (load_id,predicted_rate)
```
