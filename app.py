import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HOLDOUT = 5


DEFAULT_DATA = "datasets/company1_data.csv"
DATE_COLS = ("ORDERDATE", "Date")
SALES_COLS = ("SALES", "Revenue")


def load_monthly(path=DEFAULT_DATA):
    df = pd.read_csv(path, encoding="latin-1")
    date = next((c for c in DATE_COLS if c in df), None)
    sales = next((c for c in SALES_COLS if c in df), None)
    if not (date and sales):
        raise ValueError(
            f"needs a date column ({' or '.join(DATE_COLS)}) "
            f"and a sales column ({' or '.join(SALES_COLS)})"
        )
    df[date] = pd.to_datetime(df[date])
    return df.resample("MS", on=date)[sales].sum()


def fit(series):
    return ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit()


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ACTUAL, FORECAST = "#2a78d6", "#eb6834"


def plot(monthly, forecast, path="forecast.png"):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 2])
    money = FuncFormatter(
        lambda v, _: (
            f"${v / 1e6:.1f}M" if v >= 1e6 else "$0" if v == 0 else f"${v / 1e3:.0f}K"
        )
    )

    joined = pd.concat([monthly[-1:], forecast])
    top.plot(monthly.index, monthly, color=ACTUAL, lw=2, label="Actual")
    top.plot(joined.index, joined, color=FORECAST, lw=2, ls="--", label="Forecast")
    top.axvspan(forecast.index[0], forecast.index[-1], color=GRID, alpha=0.4, lw=0)
    for s, c in ((monthly, ACTUAL), (forecast, FORECAST)):
        peak = s.idxmax()
        top.plot(peak, s[peak], "o", ms=8, color=c, mec=SURFACE, mew=2)
        top.annotate(
            f"{peak:%b %Y}  {money(s[peak])}",
            (peak, s[peak]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )
    top.set_title(
        "Monthly sales: actual and forecast", loc="left", color=INK, fontsize=13
    )
    top.legend(frameon=False, loc="upper left", labelcolor=INK)

    bottom.bar(
        forecast.index,
        forecast,
        width=24,
        color=FORECAST,
        edgecolor=SURFACE,
        linewidth=2,
    )
    bottom.set_xticks(forecast.index, [f"{d:%b\n%Y}" for d in forecast.index])
    bottom.set_title("Forecast by month", loc="left", color=INK, fontsize=13)

    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.yaxis.set_major_formatter(money)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run(horizon, path=DEFAULT_DATA):
    monthly = load_monthly(path)

    train, test = monthly[:-HOLDOUT], monthly[-HOLDOUT:]
    pred = fit(train).forecast(HOLDOUT)
    mape = ((pred - test).abs() / test).mean() * 100

    forecast = fit(monthly).forecast(horizon).clip(lower=0)
    assert len(forecast) == horizon and (forecast >= 0).all()
    return monthly, forecast, mape


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    monthly, forecast, mape = run(horizon)
    print(f"Backtest on last {HOLDOUT} months: MAPE = {mape:.1f}%\n")

    print("Forecast (monthly sales):")
    print(forecast.round(2).rename_axis("MONTH").to_string())
    forecast.rename("PREDICTED_SALES").to_csv("forecast.csv", index_label="MONTH")
    plot(monthly, forecast)
    print("\nSaved forecast.csv, forecast.png")


if __name__ == "__main__":
    main()
