# Sales Forecast

Forecasts monthly sales per company with Holt-Winters exponential smoothing
(additive trend + 12-month seasonality). Comes as a Streamlit web app and a CLI.

## Setup

Requires Python 3.12+.

```bash
pip install -r requirements.txt
```

## Web app

```bash
streamlit run web.py
```

- One tab per company. Every CSV in `datasets/` loads automatically, and more can be uploaded in the page.
- Slider sets forecast horizon (1–36 months).
- Each tab shows backtest MAPE, forecast total, peak month, the chart, a monthly table and a CSV download.
- Files without usable columns are skipped and listed at the top. Exact duplicate files are shown once.

## CLI

```bash
python3 app.py        # 12-month forecast for datasets/company1_data.csv
python3 app.py 24     # custom horizon
```

Prints backtest MAPE and the forecast, and writes `forecast.csv` and `forecast.png`.

## Data format

One CSV per company, named after the company (`acme.csv` becomes tab "acme"). Needs:

| Column | Accepted names |
|---|---|
| Order date | `ORDERDATE` or `Date` |
| Sale amount | `SALES` or `Revenue` |

Other columns are ignored. Rows are summed per month. At least 2 years of monthly data is needed for the seasonal model.

## Datasets

- `company1_data.csv`: Kaggle "Sample Sales Data" (classic car models), Jan 2003 – May 2005.
- `company2_data.csv`: **synthetic** data for a fictional outdoor-gear retailer, Jan 2019 – Dec 2024. Generated, not real sales.

## Model check

The last 5 months are held out, the model is fit on the rest, and MAPE is reported on the held-out months. Lower is better.
