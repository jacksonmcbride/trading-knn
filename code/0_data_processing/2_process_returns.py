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
    files = list(folder.glob("*.csv"))
    print(f"Searching in {folder}, found {len(files)} files")
    for file in tqdm(list(folder.glob("*.csv")), desc=f"{asset_type.upper()}", ncols=80):
        ticker = file.stem.upper()
        try:
            df = pd.read_csv(file, usecols=["date", "adj_close"])
            df = df.dropna()
            df = df[df["adj_close"] > 0]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").drop_duplicates("date", keep="last")
            ret = np.log(df["adj_close"]).diff().astype(np.float32)
            returns[ticker] = pd.Series(ret.values, index=df["date"].values)
        except Exception:
            continue
    returns_df = pd.DataFrame(returns).sort_index().astype(np.float32)
    returns_df.index.name = "date"
    return returns_df

def main():
    for asset_type in ["stock", "etf"]:
        mat = build_returns_matrix(asset_type)
        out_path = OUT_DIR / f"{asset_type}_returns.parquet"
        mat.to_parquet(out_path, compression="zstd")
        print(f"Saved {asset_type} returns matrix: {mat.shape} -> {out_path}")

if __name__ == "__main__":
    main()