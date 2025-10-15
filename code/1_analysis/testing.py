from pathlib import Path
import argparse
import pandas as pd


def first_data_date(df: pd.DataFrame):
    mask = df.notna().any(axis=1)
    if mask.any():
        return pd.to_datetime(df.index[mask][0]).date()
    return None


def first_start_date(df: pd.DataFrame, k: int, floor: int):
    counts = df.notna().sum(axis=1)
    threshold = max(10 * k, floor)
    meets = counts >= threshold
    if meets.any():
        idx = meets[meets].index[0]
        return pd.to_datetime(idx).date(), int(counts.loc[idx]), int(threshold)
    return None, None, int(threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick checks on earliest data dates and start date thresholds.")
    parser.add_argument("--k", type=int, default=30, help="K for KNN; start when N >= max(10*k, floor)")
    parser.add_argument("--stock-floor", type=int, default=200, help="Minimum stock universe size")
    parser.add_argument("--etf-floor", type=int, default=100, help="Minimum ETF universe size")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    returns_dir = root / "data" / "processed" / "returns"
    etf_path = returns_dir / "etf_returns.parquet"
    stock_path = returns_dir / "stock_returns.parquet"

    etf_df = pd.read_parquet(etf_path)
    stock_df = pd.read_parquet(stock_path)

    etf_first = first_data_date(etf_df)
    stock_first = first_data_date(stock_df)
    print(f"ETF first data date:   {etf_first}")
    print(f"Stock first data date: {stock_first}")

    # Threshold-based start dates
    etf_start, etf_N, etf_thresh = first_start_date(etf_df, args.k, args.etf_floor)
    stock_start, stock_N, stock_thresh = first_start_date(stock_df, args.k, args.stock_floor)

    print(f"\nParams: k={args.k}, ETF floor={args.etf_floor}, Stock floor={args.stock_floor}")
    print(f"ETF start (N>={etf_thresh}):   {etf_start}   N={etf_N}")
    print(f"Stock start (N>={stock_thresh}): {stock_start}   N={stock_N}")

    # If you require both universes to satisfy their thresholds, take the later of the two
    both_start = max(d for d in [etf_start, stock_start] if d is not None) if (etf_start or stock_start) else None
    print(f"\nRecommended overall start (both satisfy thresholds): {both_start}")


if __name__ == "__main__":
    main()
