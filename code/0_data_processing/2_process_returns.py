from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR.parent / "data" / "raw" / "yahoo"
OUT_DIR = BASE_DIR.parent / "data" / "processed" / "returns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_returns_matrix(asset_type):
    folder = RAW_DIR / asset_type
    returns = {}
    filtered_out = 0
    files = list(folder.glob("*.csv"))
    print(f"Searching in {folder}, found {len(files)} files")
    for file in tqdm(list(folder.glob("*.csv")), desc=f"{asset_type.upper()}", ncols=80):
        ticker = file.stem.upper()
        try:
            # Read required columns; include volume for stocks if available
            if asset_type == "stock":
                try:
                    df = pd.read_csv(file, usecols=["date", "adj_close", "volume"])
                except Exception:
                    df = pd.read_csv(file, usecols=["date", "adj_close"])  # fallback
            else:
                df = pd.read_csv(file, usecols=["date", "adj_close"])
            df = df.dropna()
            df = df[df["adj_close"] > 0]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").drop_duplicates("date", keep="last")
            # Optional filtering for problematic stocks (apply only to stocks)
            if asset_type == "stock" and not df.empty:
                win = df.tail(252)  # roughly 1 trading year
                if not win.empty:
                    last_price = float(win["adj_close"].iloc[-1])
                    pct_below_2 = float((win["adj_close"] < 2).mean())
                    days = int(win.shape[0])
                    # Average daily dollar volume if volume exists
                    avg_dollar_vol = None
                    if "volume" in win.columns:
                        try:
                            vol = pd.to_numeric(win["volume"], errors="coerce")
                            avg_dollar_vol = float((win["adj_close"] * vol).dropna().mean()) if vol.notna().any() else None
                        except Exception:
                            avg_dollar_vol = None
                    # Rules: require sufficient history, price floor, penny prevalence, liquidity if available
                    if (
                        days < 252
                        or last_price < 5.0
                        or pct_below_2 > 0.25
                        or (avg_dollar_vol is not None and avg_dollar_vol < 1_000_000)
                    ):
                        filtered_out += 1
                        continue
            ret = np.log(df["adj_close"]).diff().astype(np.float32)
            returns[ticker] = pd.Series(ret.values, index=df["date"].values)
        except Exception:
            continue
    returns_df = pd.DataFrame(returns).sort_index().astype(np.float32)
    returns_df.index.name = "date"
    if asset_type == "stock":
        print(f"Filtered out {filtered_out} stock tickers by price/liquidity rules")
    return returns_df

def main():
    for asset_type in ["stock", "etf"]:
        mat = build_returns_matrix(asset_type)
        out_path = OUT_DIR / f"{asset_type}_returns.parquet"
        mat.to_parquet(out_path, compression="zstd")
        print(f"Saved {asset_type} returns matrix: {mat.shape} -> {out_path}")

if __name__ == "__main__":
    main()
