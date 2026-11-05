// Builds reports/Freight_Rate_Model_Report.docx from metrics.json, model_summary.json and the figures.
// Usage: node reports/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, LevelFormat, BorderStyle,
} = require("docx");

const R = __dirname;
const ROOT = path.join(R, "..");
const metrics = JSON.parse(fs.readFileSync(path.join(R, "metrics.json")));
const summary = JSON.parse(fs.readFileSync(path.join(R, "model_summary.json")));
const c = summary.cleaning;
const coef = summary.day_level_coefficients;

const FONT = "Arial";
const INK = "1F2D33";
const ACCENT = "064A56";
const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, ...opts, children: [].concat(text).map(t => typeof t === "string" ? new TextRun({ text: t, font: FONT, size: 21, color: INK }) : t) });
const b = (text) => new TextRun({ text, bold: true, font: FONT, size: 21, color: INK });
const t = (text) => new TextRun({ text, font: FONT, size: 21, color: INK });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT })] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT })] });
const bullet = (runs) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [].concat(runs).map(r => typeof r === "string" ? t(r) : r) });
const caption = (text) => new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text, italics: true, font: FONT, size: 18, color: "55656B" })] });

function image(file, widthPx, heightPx) {
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, keepLines: true, spacing: { before: 80, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: widthPx, height: heightPx } })] });
}
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}
function fitted(file, width) {
  const { w, h } = pngSize(file);
  return image(file, width, Math.round(width * h / w));
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "C9D6D9" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, x) => a + x, 0);
  const cell = (text, i, header, bold) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: header ? { fill: "E6EEF0", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT, children: [new TextRun({ text: String(text), font: FONT, size: 19, bold: header || bold, color: INK })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((x, i) => cell(x, i, true)) }),
      ...rows.map(r => new TableRow({ children: r.cells.map((x, i) => cell(x, i, false, r.bold)) }))],
  });
}

const mean = metrics.rolling_backtest_mean;
const hold = metrics.holdout_train_jan_aug_test_sep_oct;
const methods = Object.keys(mean);
const finalKey = methods.find(m => m.includes("final"));
const fmt = (x, d = 2) => Number(x).toFixed(d);
const pct = (x) => (x * 100).toFixed(1) + "%";
const backtestRows = methods.map(m => ({ bold: m === finalKey, cells: [m, fmt(mean[m]["MAPE_%"]) + "%", "$" + fmt(mean[m]["MAE_$"], 0), fmt(hold[m]["MAPE_%"]) + "%", "$" + fmt(hold[m]["MAE_$"], 0)] }));
const plain = mean["LightGBM, no date signal"]["MAPE_%"];
const withMi = mean["LightGBM + raw market_index"]["MAPE_%"];
const final = mean[finalKey]["MAPE_%"];
const lane = mean["Baseline: lane median $/mile x distance"]["MAPE_%"];

const sep = hold;  // Jan to Aug training, Sep to Oct test
const sepFinal = sep[finalKey]["MAPE_%"];
const sepPlain = sep["LightGBM, no date signal"]["MAPE_%"];
const sepMi = sep["LightGBM + raw market_index"]["MAPE_%"];

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun({ text: "Freight Rate Prediction", font: FONT })] }),
  p([t("Technical report for the Spotter Machine Learning Engineer assessment")], { spacing: { after: 40 } }),
  p([t("Atif Ullah Khan")], { spacing: { after: 280 } }),

  h1("Executive summary"),
  p([t("The goal is to predict the posted rate of a truck load from its lane, distance, equipment type, weight and date. The labelled data covers January to October 2025, while the 12,000 loads to be predicted are all from November and December 2025. This makes the task a forecasting problem rather than a standard regression problem, and that observation shaped every decision in this report.")]),
  p([t("The final model combines two parts. A gradient boosting model (LightGBM) learns how the characteristics of a load affect its price, and a small linear model learns the overall market level for each day. Separating the two allows the model to carry the market level forward into November and December, which a tree model cannot do on its own.")]),
  p([t("Evaluated on future months only, the model achieves a mean absolute percentage error (MAPE) of "), b(fmt(final) + "%"), t(", compared with " + fmt(plain) + "% for a standard LightGBM model and " + fmt(lane) + "% for a lane average baseline. On the most recent period available, September and October, the error falls to "), b(fmt(sepFinal) + "%"), t(".")]),

  h1("1. Problem and data"),
  p([t("The training file contains 48,000 loads. Each load records the pickup and delivery city with their coordinates, the trip distance, the equipment type (Dry Van, Reefer or Flatbed), the weight, the date, a market index, a quote signal and the target, the posted rate in dollars. The validation file contains the same inputs, without the rate, for 12,000 loads. A third file lists a single fixed load, Lexington to Fort Wayne, on every day of December, and is used to show how the predictions change over time.")]),
  p([t("Prediction quality is reported mainly as MAPE, the average error expressed as a percentage of the true rate. It suits freight pricing because rates range from about $200 for short trips to over $15,000 for long ones, and a $50 error means something very different at each end of that range. Mean absolute error (MAE) in dollars is reported alongside it.")]),

  h1("2. Key findings from the data"),
  h2("2.1 Distance and equipment drive the price"),
  p([t("The typical rate is about $2.15 per mile. Short trips cost more per mile, around $2.90 for loads under 150 miles, because fixed costs such as loading time are spread over fewer miles. Reefer loads cost about 13% more per mile than Dry Van, and Flatbed about 9% more. Heavier loads are slightly more expensive.")]),
  h2("2.2 The market index is a daily national signal"),
  p([t("The market index barely varies between loads on the same day (standard deviation 0.025, against 0.17 across the year), so it behaves like a single national index with a little noise added. The model therefore uses the average index for each day rather than the value attached to each load. This also solves the 374 missing values, and it means the index is known for every December date, because the validation file contains it.")]),
  p([t("From January to August the index follows the general level of rates closely. In September and October that relationship weakens: the index is low, but rates stay at average levels. As section 4 shows, this has a large effect on how the index should be used.")]),
  h2("2.3 Rates rise towards the end of each quarter"),
  p([t("After accounting for the market index and a slow upward trend over the year, a clear pattern remains. Rates climb through March, June and September and fall back at the start of the following month. This is consistent with the capacity pressure carriers often see at quarter end. December is also a quarter end month, so the pattern matters for the forecast.")]),
  h2("2.4 Some validation cities never appear in training"),
  p([t("Eight cities in the validation file, including Chicago, Charlotte and San Diego, do not appear in the training data. If cities were treated as labels, the model would know nothing about them. Instead, each city is described by its latitude and longitude, so a new city is placed near the cities it resembles geographically. In a simulation that hid eight training cities, the error on loads involving those cities was 2.15%, close to the 1.82% for familiar cities.")]),
  h2("2.5 The quote signal adds information"),
  p([t("The quote signal has an unusual U shaped relationship with rate per mile, and its overall correlation with price is weak. Removing it nevertheless made predictions worse, so it was kept and the model was left to learn its shape.")]),

  h1("3. Data quality issues"),
  p("The table below lists each issue found and how it was handled. Rows were only ever removed from the training data. Every load in the validation and December files receives a prediction."),
  table(["Issue", "Rows", "Handling"], [
    { cells: ["Rates far outside the normal range: below $1 or above $6 per mile, roughly one sixth of or two and a half to six times the usual rate for the lane", String(c.rate_per_mile_outliers), "Removed from training"] },
    { cells: ["Further rates that the first model fit missed by more than about 50%", String(c.residual_outliers), "Removed from training"] },
    { cells: ["Negative weights, which behave exactly like the positive value", String(c.negative_weight), "Treated as sign errors"] },
    { cells: ["Missing weights", String(c.missing_weight), "Left missing for the model"] },
    { cells: ["Missing market index values", String(c.missing_market_index), "Replaced by the daily average"] },
    { cells: ["Weights recorded at exactly 47,500 lb, an apparent cap", "1,204", "Kept, as they behave normally"] },
    { cells: ["Validation cities absent from training", "8 cities", "Described by coordinates"] },
  ], [5200, 1100, 3060]),
  caption(`Table 1. Data quality issues. After cleaning, ${c.rows_used.toLocaleString()} of the ${c.rows_in.toLocaleString()} training loads were used.`),
  fitted(path.join(R, "rate_per_mile_outliers.png"), 500),
  caption("Figure 1. Rate per mile across all training loads, on a log scale. The main body sits between about $1.60 and $3.20 per mile. The two small clusters outside the red lines are recording errors."),
  p("Checks on the short trips confirmed that the upper bound does not remove genuine loads: even the most expensive one percent of trips under 150 miles stays below $3.50 per mile."),

  h1("4. Validation strategy"),
  h2("4.1 Why a random split would mislead"),
  p("A common approach is to shuffle the data and hold out a random 20% for testing. Here that would place loads from September in the training set and loads from March in the test set, so the model would effectively see the future market. The resulting error would look better than anything achievable on the real task, where every validation load is later than every training load."),
  h2("4.2 Rolling time based backtest"),
  p("Instead, the model was evaluated with a rolling origin backtest. For each month from May to September, it was trained on all earlier months and tested on that month and the next one. Each test therefore asks the same question as the real task: how well can we price loads up to two months after the last data we have? The final fold, which trains on January to August and tests on September and October, is the closest copy of the real situation."),
  table(["Method", "Rolling MAPE", "Rolling MAE", "Sep and Oct MAPE", "Sep and Oct MAE"], backtestRows, [4160, 1300, 1300, 1300, 1300]),
  caption("Table 2. Average over five rolling folds, and the single fold trained on January to August and tested on September and October. Errors are measured on test loads within the normal rate range."),
  fitted(path.join(R, "backtest_mape.png"), 500),
  caption("Figure 2. Average error across the five folds for each method."),
  h2("4.3 What the comparison shows"),
  p([t("Simple baselines that multiply distance by a typical rate per mile reach 5% to 8% error, which shows how much of the price depends on more than distance alone. A standard LightGBM model brings this down to " + fmt(plain) + "%. Giving that same model the market index directly does not help on average (" + fmt(withMi) + "%), and on the most recent fold it doubles the error, from " + fmt(sepPlain) + "% to " + fmt(sepMi) + "%. The trees learn the strong January to August link between index and price, and that link no longer held in September and October.")]),
  p([t("The two part model uses the same index in a constrained, linear way alongside a trend and the quarter end pattern. It is the most accurate method on every fold, with an average error of " + fmt(final) + "%.")]),

  h1("5. Final model"),
  p("The model predicts the logarithm of the rate, so errors are treated in proportion to the size of the rate. The logarithm is split into two additive parts."),
  p([b("Load effect. "), t("A LightGBM model uses the distance, the straight line distance between the two cities, the ratio between the two (how indirect the route is), the pickup and delivery coordinates, the equipment type, the weight and the quote signal. It captures everything about the price that does not depend on the date.")]),
  p([b("Daily market level. "), t("A linear model describes the level shared by all loads on a given day. Its inputs are the daily market index, the number of months since January, a flag for quarter end months, how far through the month the day falls, and the combination of the last two. Because it is linear, it extends naturally into dates beyond the training period.")]),
  p([t("The two parts are fitted in turn for three rounds, a technique known as backfitting. LightGBM is fitted on the rates after removing the current daily level, the daily level is then refitted on what LightGBM leaves unexplained, and the cycle repeats. After a first complete fit, the 75 training loads the model misses by more than about 50% are set aside and the model is fitted again. The final model is trained on all cleaned data from January to October.")]),
  p([t(`The fitted daily level implies an upward trend of about ${pct(coef.months_elapsed)} per month and a rise of about ${pct(coef.quarter_end_x_progress)} from the first to the last day of a quarter end month.`)]),
  fitted(path.join(R, "market_level.png"), 540),
  caption("Figure 3. Daily median rate per mile in the training data (light line) and the market level estimated by the model (dark line), including its extension into November and December (shaded)."),

  h1("6. December prediction chart"),
  p("The December file describes one load, Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb, repeated for every day of the month. It contains only the city names, distance, equipment, weight and date, so the remaining inputs were filled as follows. The coordinates come from the city table built from the other files. The daily market index comes from the December loads in the validation file. The quote signal is set to the November and December median for Dry Van loads. Because it is the same on every day, it affects the level of the curve but not its shape."),
  fitted(path.join(ROOT, "scorer_results", "candidate_december.png"), 560),
  caption("Figure 4. December 2025 predicted rate for the fixed load, produced by the provided score.py."),
  p([t(`The predictions range from $${fmt(summary["december_range_$"][0], 0)} to $${fmt(summary["december_range_$"][1], 0)}, or about $2.24 to $2.34 per mile. This agrees with the real Dry Van loads on this lane in September and October, which were priced between $778 and $865. The curve rises gently through the month because of the quarter end pattern, and it moves in a weekly cycle that peaks on Thursdays. That cycle comes from the daily market index and matches the weekday pattern in the training data, where mid week loads are slightly more expensive than weekend ones.`)]),

  h1("7. Limitations and next steps"),
  bullet("The training data contains no December or holiday periods, so effects specific to Christmas week cannot be learned. The December forecast relies on the market index and the quarter end pattern seen in earlier months."),
  bullet("The trend is estimated from ten months of data. A longer history would make it more reliable when extended into the future."),
  bullet("About one percent of recorded rates are clearly erroneous. If similar errors exist in the validation answers, they will raise the measured error for any model, especially on squared error metrics such as RMSE."),
  bullet("Natural next steps are lane level price history features, prediction intervals so that each quote carries a measure of uncertainty, and monitoring the error by lane once the November and December rates are known."),

  h1("Appendix: reproducing the results"),
  p("All results in this report can be reproduced from the repository. Install the dependencies with pip install -r requirements.txt, then run python -m src.train_predict to train the model and write both prediction files, and python -m src.backtest to repeat the evaluation in section 4. The provided score.py validates both files and produces the December chart."),
];

const doc = new Document({
  creator: "Atif Ullah Khan",
  title: "Freight Rate Prediction",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 44, bold: true, color: ACCENT }, paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 28, bold: true, color: ACCENT }, paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: INK }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } }, children }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(R, "Freight_Rate_Model_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
