import io
from pathlib import Path

import pandas as pd
import streamlit as st

from app import DATE_COLS, HOLDOUT, SALES_COLS, plot, run

st.set_page_config(page_title="Sales forecast", layout="wide")
st.title("Sales forecast")

uploads = st.file_uploader(
    "Add company datasets (one CSV per company, needs ORDERDATE/Date and "
    "SALES/Revenue columns)",
    type="csv",
    accept_multiple_files=True,
)
horizon = st.slider("Months to forecast", 1, 36, 12)

# ponytail: company name = file name; add a name field if filenames are messy
paths = sorted(Path("datasets").glob("*.csv"), key=lambda p: p.stem)
files = [(p.stem, p.read_bytes()) for p in paths]
files += [(Path(f.name).stem, f.getvalue()) for f in uploads]
datasets, seen, skipped = {}, set(), []
for company, data in files:
    if data in seen:  # skip byte-identical copies
        continue
    seen.add(data)
    try:
        cols = pd.read_csv(io.BytesIO(data), encoding="latin-1", nrows=0).columns
    except Exception:  # empty or unparseable upload
        cols = []
    if any(c in cols for c in DATE_COLS) and any(c in cols for c in SALES_COLS):
        datasets[company] = data
    else:
        skipped.append(company)
if skipped:
    st.caption(f"Skipped (no date + sales columns to forecast): {', '.join(skipped)}")
if not datasets:
    st.info("No datasets yet. Upload a CSV or put one in datasets/.")
    st.stop()


@st.cache_data
def forecast_for(data, horizon):
    return run(horizon, io.BytesIO(data))


for tab, (company, data) in zip(st.tabs(list(datasets)), datasets.items()):
    with tab:
        try:
            monthly, forecast, mape = forecast_for(data, horizon)
        except Exception as e:
            st.error(
                f"Can't forecast {company}: {e}. Also needs at least "
                "2 years of monthly data."
            )
            continue

        a, b, c = st.columns(3)
        a.metric(f"Backtest MAPE (last {HOLDOUT} mo)", f"{mape:.1f}%")
        b.metric("Forecast total", f"${forecast.sum():,.0f}")
        c.metric(
            "Peak month",
            f"{forecast.idxmax():%b %Y}",
            f"${forecast.max():,.0f}",
            delta_color="off",
        )

        buf = io.BytesIO()
        plot(monthly, forecast, buf)
        st.image(buf)

        table = forecast.round(2).rename("PREDICTED_SALES").rename_axis("MONTH").to_frame()
        table.index = table.index.strftime("%Y-%m")
        st.dataframe(table, width="stretch")
        st.download_button(
            "Download CSV",
            table.to_csv(),
            f"{company}_forecast.csv",
            "text/csv",
            key=f"dl-{company}",
        )
