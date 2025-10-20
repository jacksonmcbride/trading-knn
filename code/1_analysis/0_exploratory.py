from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import numpy as np

def count_assets(parquet_path: Path) -> int:
    df = pd.read_parquet(parquet_path)
    return int(df.shape[1])

def count_missing_on_last_date(parquet_path: Path) -> int:
    df = pd.read_parquet(parquet_path)
    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    last_date = df.index.max()
    last_row = df.loc[last_date]
    return int(last_row.isna().sum())

def plot_histogram(
    values,
    out_path: Path,
    bins: int = 200,
    is_date: bool = False,
    xlabel: str = "",
    ylabel: str = "Count",
    title: str = "",
    use_robust_xlim: bool = False,
    annotate: bool = False,
    value_units: str = "",
    annotate_n: bool = False,
    stat: str = "mean",
    fixed_range: tuple | None = None,
    label_side: str = "left",
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black")

    # Axis range: prefer fixed_range if provided, else optional robust bounds
    hist_range = fixed_range
    if hist_range is None and use_robust_xlim and len(values) > 0:
        s = pd.Series(values).dropna()
        if not s.empty:
            lo, hi = s.quantile([0.01, 0.99]).values
            if lo == hi:
                lo, hi = s.min(), s.max()
            if lo != hi:
                pad = 0.05 * (hi - lo)
                hist_range = (lo - pad, hi + pad)

    plt.hist(values, bins=bins, range=hist_range, color="#0b1f3b", edgecolor="black", linewidth=0.4)
    # Compute statistic strictly on the data that appears in the histogram
    s_all = pd.Series(values).dropna()
    if hist_range is not None:
        lo, hi = hist_range
        s_stat = s_all[(s_all >= lo) & (s_all <= hi)]
    else:
        s_stat = s_all

    if stat == "median":
        stat_val = s_stat.median() if not s_stat.empty else float('nan')
        stat_name = "Median"
    else:
        stat_val = s_stat.mean() if not s_stat.empty else float('nan')
        stat_name = "Average"

    if pd.notna(stat_val):
        plt.axvline(stat_val, color="black", linestyle="--", linewidth=1)
    if annotate and pd.notna(stat_val):
        ymin, ymax = ax.get_ylim()
        label = f"{stat_name}: {stat_val:.2f}{value_units}"
        if is_date:
            try:
                label = f"{stat_name}: {mdates.num2date(stat_val).date().isoformat()}"
            except Exception:
                pass
        # Determine text x-position and alignment
        x_left, x_right = ax.get_xlim()
        dx = 0.02 * (x_right - x_left)
        if label_side.lower() == "right":
            x_text = min(stat_val + dx, x_right)
            ha = "left"
        else:
            x_text = stat_val
            ha = "right"
        ax.text(
            x_text,
            ymax * 0.95,
            label,
            ha=ha,
            va="top",
            color="black",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.3),
        )

    if annotate_n:
        # Count the number of points actually contributing to the histogram range
        if hist_range is not None:
            lo, hi = hist_range
            n = int((pd.Series(values).dropna().between(lo, hi)).sum())
        else:
            n = int(pd.Series(values).dropna().shape[0])
        ax.text(
            0.01,
            0.98,
            f"n = {n}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="black",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.3),
        )

    if hist_range is not None:
        plt.xlim(*hist_range)

    if is_date:
        ax.xaxis_date()
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
        plt.gcf().autofmt_xdate()

    if title:
        plt.title(title, color="black")
    plt.xlabel(xlabel, color="black")
    plt.ylabel(ylabel, color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    returns_dir = root / "data" / "processed" / "returns"
    plots_dir = root / "results" / "exploratory" / "plots"
    raw_dir = root / "data" / "raw" / "yahoo"

    etf_path = returns_dir / "etf_returns.parquet"
    stock_path = returns_dir / "stock_returns.parquet"

    etf_count = count_assets(etf_path)
    stock_count = count_assets(stock_path)

    print(f"ETF assets: {etf_count}")
    print(f"Stock assets: {stock_count}")

    etf_missing_last = count_missing_on_last_date(etf_path)
    stock_missing_last = count_missing_on_last_date(stock_path)

    print(f"ETF missing on most recent date: {etf_missing_last}")
    print(f"Stock missing on most recent date: {stock_missing_last}")

    # Load parquets to determine which tickers have data on the most recent date (exclude delisted)
    etf_df = pd.read_parquet(etf_path)
    stock_df = pd.read_parquet(stock_path)
    etf_last = etf_df.index.max()
    stock_last = stock_df.index.max()
    valid_etf = set(etf_df.columns[etf_df.loc[etf_last].notna()])
    valid_stock = set(stock_df.columns[stock_df.loc[stock_last].notna()])

    # Compute average annual return (first->last CAGR) and start dates from raw CSVs, only for non-delisted tickers
    def returns_and_starts(asset_type: str, allowed: set[str]):
        folder = raw_dir / asset_type
        rets = []
        starts = []
        details = []
        files = [f for f in folder.glob("*.csv") if f.stem.upper() in allowed]
        for file in tqdm(files, desc=asset_type.upper(), ncols=80):
            try:
                with open(file, "r", newline="") as f:
                    rdr = csv.DictReader(f)
                    first = None
                    last = None
                    row_count = 0
                    for row in rdr:
                        price_raw = row.get("adj_close")
                        date_raw = row.get("date")
                        if price_raw is None or date_raw is None:
                            continue
                        try:
                            price = float(price_raw)
                        except Exception:
                            continue
                        if price <= 0:
                            continue
                        row_count += 1
                        if first is None:
                            first = (date_raw, price)
                        last = (date_raw, price)
                if first and last:
                    first_dt = pd.to_datetime(first[0])
                    last_dt = pd.to_datetime(last[0])
                    years = (last_dt - first_dt).days / 365.25
                    # Require at least ~1 year of data to compute a stable annualized return
                    if years >= 1.0 and first[1] > 0:
                        ann_ret_pct = ((last[1] / first[1]) ** (1.0 / years) - 1.0) * 100.0
                        rets.append(ann_ret_pct)
                        starts.append(first_dt)
                        details.append({
                            "ticker": file.stem.upper(),
                            "rows": row_count,
                            "first_date": first_dt,
                            "last_date": last_dt,
                            "years": years,
                            "first_price": first[1],
                            "last_price": last[1],
                            "ann_ret_pct": ann_ret_pct,
                        })
            except Exception:
                continue
        return rets, starts, details

    etf_avg_ret_pct, etf_start_dates, etf_details = returns_and_starts("etf", valid_etf)
    stock_avg_ret_pct, stock_start_dates, stock_details = returns_and_starts("stock", valid_stock)

    etf_start_nums = mdates.date2num(pd.to_datetime(etf_start_dates).to_pydatetime()) if len(etf_start_dates) else []
    stock_start_nums = mdates.date2num(pd.to_datetime(stock_start_dates).to_pydatetime()) if len(stock_start_dates) else []

    # Helper to remove outliers by percentile trimming
    def trim_percentile(values, lo_p=1.0, hi_p=99.0):
        s = pd.Series(values, dtype="float64").dropna()
        if s.empty:
            return []
        lo, hi = s.quantile([lo_p / 100.0, hi_p / 100.0]).values
        return s[(s >= lo) & (s <= hi)].values

    # Trim extreme annual returns for stability in plots/stats
    etf_ann_trim = trim_percentile(etf_avg_ret_pct, 1.0, 99.0)
    stock_ann_trim = trim_percentile(stock_avg_ret_pct, 1.0, 99.0)

    # Diagnostics: investigate potentially incorrect negative averages for stocks
    try:
        if stock_details:
            sd = pd.DataFrame(stock_details)
            sd_sorted = sd.sort_values("ann_ret_pct")
            worst = sd_sorted.head(10)
            # Compute last recorded daily return and NaN counts from parquet for these tickers
            print("\n=== Diagnostics: Worst Performing Stocks (by annualized return) ===")
            for _, row in worst.iterrows():
                tkr = row["ticker"]
                last_ret = None
                n_nans = None
                if tkr in stock_df.columns:
                    try:
                        last_ret = float(stock_df.loc[stock_last, tkr])
                    except Exception:
                        last_ret = None
                    try:
                        n_nans = int(stock_df[tkr].isna().sum())
                    except Exception:
                        n_nans = None
                print(
                    f"{tkr}: rows={int(row['rows'])}, first={row['first_date'].date()} @ {row['first_price']:.4f}, "
                    f"last={row['last_date'].date()} @ {row['last_price']:.4f}, years={row['years']:.2f}, "
                    f"ann_ret={row['ann_ret_pct']:.2f}%, last_daily_ret={'' if last_ret is None else f'{last_ret*100:.2f}%'}"
                    f"{'' if n_nans is None else f', n_nan={n_nans}'}"
                )
            # Aggregate checks
            sd_neg = (sd["ann_ret_pct"] < 0).sum()
            print(f"Total stocks with negative annualized return: {sd_neg} of {sd.shape[0]}")
    except Exception as e:
        print(f"Diagnostics failed: {e}")

    # Save minimal histograms with average line/label
    plot_histogram(
        etf_ann_trim,
        plots_dir / "etf_avg_returns_hist.png",
        bins=240,
        is_date=False,
        xlabel="Average Annual Return (%)",
        title="ETFs: Average Annual Returns (%)",
        use_robust_xlim=False,
        annotate=True,
        value_units="%",
        annotate_n=True,
        stat="mean",
        fixed_range=(-100, 200),
    )
    plot_histogram(
        stock_ann_trim,
        plots_dir / "stock_avg_returns_hist.png",
        bins=240,
        is_date=False,
        xlabel="Average Annual Return (%)",
        title="Stocks: Average Annual Returns (%)",
        use_robust_xlim=False,
        annotate=True,
        value_units="%",
        annotate_n=True,
        stat="mean",
        fixed_range=(-100, 200),
    )

    # Daily average returns from parquets (non-delisted only), scaled to percent
    etf_daily_avg_pct = etf_df[sorted(valid_etf)].mean(skipna=True).dropna().values * 100.0 if valid_etf else []
    stock_daily_avg_pct = stock_df[sorted(valid_stock)].mean(skipna=True).dropna().values * 100.0 if valid_stock else []

    plot_histogram(
        etf_daily_avg_pct,
        plots_dir / "etf_avg_daily_returns_hist.png",
        bins=240,
        is_date=False,
        xlabel="Average Daily Return (%)",
        title="ETFs: Average Daily Returns (%)",
        use_robust_xlim=True,
        annotate=True,
        value_units="%",
        annotate_n=True,
        stat="mean",
    )
    plot_histogram(
        stock_daily_avg_pct,
        plots_dir / "stock_avg_daily_returns_hist.png",
        bins=240,
        is_date=False,
        xlabel="Average Daily Return (%)",
        title="Stocks: Average Daily Returns (%)",
        use_robust_xlim=True,
        annotate=True,
        value_units="%",
        annotate_n=True,
        stat="mean",
    )

    # Number of observations (days) per ticker from parquets (non-delisted only)
    etf_days = etf_df[sorted(valid_etf)].count().astype(float).values if valid_etf else []
    stock_days = stock_df[sorted(valid_stock)].count().astype(float).values if valid_stock else []

    plot_histogram(
        etf_days,
        plots_dir / "etf_days_of_data_hist.png",
        bins=240,
        is_date=False,
        xlabel="Days of Data",
        title="ETFs: Days of Data per Ticker",
        use_robust_xlim=True,
        annotate=True,
        annotate_n=True,
        stat="mean",
        label_side="right",
    )
    plot_histogram(
        stock_days,
        plots_dir / "stock_days_of_data_hist.png",
        bins=240,
        is_date=False,
        xlabel="Days of Data",
        title="Stocks: Days of Data per Ticker",
        use_robust_xlim=True,
        annotate=True,
        annotate_n=True,
        stat="mean",
    )

    # ---------------------
    # Terminal summary stats
    # ---------------------
    def summarize(arr):
        s = pd.Series(arr, dtype="float64").dropna()
        if s.empty:
            return None
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        return {
            "n": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.shape[0] > 1 else float("nan"),
            "min": float(s.min()),
            "p5": float(q.loc[0.05]),
            "median": float(q.loc[0.5]),
            "p95": float(q.loc[0.95]),
            "max": float(s.max()),
        }

    def print_block(title, stats, units=""):
        if not stats:
            print(f"\n{title}: no data")
            return
        u = units
        print(f"\n{title}")
        print(f"  n: {stats['n']}")
        print(f"  mean: {stats['mean']:.2f}{u}  median: {stats['median']:.2f}{u}  std: {stats['std']:.2f}{u}")
        print(f"  min: {stats['min']:.2f}{u}  p5: {stats['p5']:.2f}{u}  p95: {stats['p95']:.2f}{u}  max: {stats['max']:.2f}{u}")

    def summarize_dates(dates):
        s = pd.to_datetime(pd.Series(dates)).dropna()
        if s.empty:
            return None
        q = s.quantile([0.05, 0.5, 0.95])
        return {
            "n": int(s.shape[0]),
            "min": s.min(),
            "p5": q.loc[0.05],
            "median": q.loc[0.5],
            "p95": q.loc[0.95],
            "max": s.max(),
        }

    def print_date_block(title, stats):
        if not stats:
            print(f"\n{title}: no data")
            return
        fmt = lambda x: pd.to_datetime(x).date().isoformat()
        print(f"\n{title}")
        print(f"  n: {stats['n']}")
        print(
            "  min: {min}  p5: {p5}  median: {median}  p95: {p95}  max: {max}".format(
                min=fmt(stats["min"]),
                p5=fmt(stats["p5"]),
                median=fmt(stats["median"]),
                p95=fmt(stats["p95"]),
                max=fmt(stats["max"]),
            )
        )

    # Annual returns (CAGR, %) — non-delisted
    print("\n=== Summary Statistics ===")
    print(f"Included ETFs (non-delisted): {len(valid_etf)} of {etf_count}")
    print(f"Included Stocks (non-delisted): {len(valid_stock)} of {stock_count}")

    print_block("ETFs: Average Annual Returns (%) [trim 1-99%]", summarize(etf_ann_trim), units="%")
    print_block("Stocks: Average Annual Returns (%) [trim 1-99%]", summarize(stock_ann_trim), units="%")

    # Average daily returns from parquet (%) — non-delisted
    print_block("ETFs: Average Daily Returns (%)", summarize(etf_daily_avg_pct), units="%")
    print_block("Stocks: Average Daily Returns (%)", summarize(stock_daily_avg_pct), units="%")

    # Daily return volatility per ticker (std of daily returns, %) — non-delisted
    etf_daily_std_pct = etf_df[sorted(valid_etf)].std(skipna=True).dropna().values * 100.0 if valid_etf else []
    stock_daily_std_pct = stock_df[sorted(valid_stock)].std(skipna=True).dropna().values * 100.0 if valid_stock else []
    print_block("ETFs: Daily Return Volatility (Std, %)", summarize(etf_daily_std_pct), units="%")
    print_block("Stocks: Daily Return Volatility (Std, %)", summarize(stock_daily_std_pct), units="%")

    # Days of data per ticker
    print_block("ETFs: Days of Data per Ticker", summarize(etf_days), units="")
    print_block("Stocks: Days of Data per Ticker", summarize(stock_days), units="")

    # Start and end dates per ticker (from parquet, non-delisted)
    def get_starts_ends(df, cols):
        starts = []
        ends = []
        for col in sorted(cols):
            s = df[col]
            starts.append(s.first_valid_index())
            ends.append(s.last_valid_index())
        return starts, ends

    etf_starts_pq, etf_ends_pq = get_starts_ends(etf_df, valid_etf)
    stock_starts_pq, stock_ends_pq = get_starts_ends(stock_df, valid_stock)

    print_date_block("ETFs: Start Dates (per ticker)", summarize_dates(etf_starts_pq))
    print_date_block("ETFs: End Dates (per ticker)", summarize_dates(etf_ends_pq))
    print_date_block("Stocks: Start Dates (per ticker)", summarize_dates(stock_starts_pq))
    print_date_block("Stocks: End Dates (per ticker)", summarize_dates(stock_ends_pq))

    # ---------------------
    # Save summary stats to CSVs (for paper)
    # ---------------------
    out_stats_dir = root / "results" / "explanatory"
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    # Numeric summaries
    numeric_rows = []
    def add_numeric(name, asset, stats, units=""):
        if not stats:
            return
        row = {"metric": name, "asset_class": asset, "units": units}
        row.update(stats)
        numeric_rows.append(row)

    add_numeric("Average Annual Return (trim 1-99%)", "ETF", summarize(etf_ann_trim), "%")
    add_numeric("Average Annual Return (trim 1-99%)", "Stock", summarize(stock_ann_trim), "%")
    add_numeric("Average Daily Return", "ETF", summarize(etf_daily_avg_pct), "%")
    add_numeric("Average Daily Return", "Stock", summarize(stock_daily_avg_pct), "%")
    add_numeric("Daily Return Volatility (Std)", "ETF", summarize(etf_daily_std_pct), "%")
    add_numeric("Daily Return Volatility (Std)", "Stock", summarize(stock_daily_std_pct), "%")
    add_numeric("Days of Data per Ticker", "ETF", summarize(etf_days), "days")
    add_numeric("Days of Data per Ticker", "Stock", summarize(stock_days), "days")

    if numeric_rows:
        pd.DataFrame(numeric_rows)[
            [
                "metric","asset_class","units","n","mean","median","std","min","p5","p95","max",
            ]
        ].to_csv(out_stats_dir / "summary_numeric.csv", index=False)

    # Date summaries
    date_rows = []
    def add_dates(name, asset, stats):
        if not stats:
            return
        fmt = lambda x: pd.to_datetime(x).date().isoformat()
        date_rows.append({
            "metric": name,
            "asset_class": asset,
            "n": stats["n"],
            "min": fmt(stats["min"]),
            "p5": fmt(stats["p5"]),
            "median": fmt(stats["median"]),
            "p95": fmt(stats["p95"]),
            "max": fmt(stats["max"]),
        })

    add_dates("Start Date (per ticker)", "ETF", summarize_dates(etf_starts_pq))
    add_dates("End Date (per ticker)", "ETF", summarize_dates(etf_ends_pq))
    add_dates("Start Date (per ticker)", "Stock", summarize_dates(stock_starts_pq))
    add_dates("End Date (per ticker)", "Stock", summarize_dates(stock_ends_pq))

    if date_rows:
        pd.DataFrame(date_rows)[["metric","asset_class","n","min","p5","median","p95","max"]].to_csv(
            out_stats_dir / "summary_dates.csv", index=False
        )
    plot_histogram(
        etf_start_nums,
        plots_dir / "etf_start_dates_hist.png",
        bins=200,
        is_date=True,
        xlabel="Start Date",
        title="ETFs: Start Dates",
        annotate=True,
        annotate_n=True,
        stat="median",
    )
    plot_histogram(
        stock_start_nums,
        plots_dir / "stock_start_dates_hist.png",
        bins=200,
        is_date=True,
        xlabel="Start Date",
        title="Stocks: Start Dates",
        annotate=True,
        annotate_n=True,
        stat="median",
    )


if __name__ == "__main__":
    main()
