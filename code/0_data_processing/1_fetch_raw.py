# Imports
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import contextlib
import os

# Config and directories
BASE_DIR = Path(__file__).resolve().parent.parent
UNIVERSE = BASE_DIR / "configs" / "universe.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "yahoo"

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install yfinance: pip install yfinance")


def load_universe(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit("configs/universe.csv not found")
    df = pd.read_csv(path)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    return (
        df[col].astype(str).str.strip().str.upper().dropna().drop_duplicates().tolist()
    )


def fetch_raw(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    df = pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
            for kw in (
                {"period": "max", "interval": "1d", "auto_adjust": False, "actions": True},
                {"start": "1950-01-01", "interval": "1d", "auto_adjust": False, "actions": True},
            ):
                try:
                    df = tk.history(**kw)
                except Exception:
                    # Suppress all errors for fetching history
                    continue
                if not df.empty:
                    break
    if df.empty:
        return df
    idx = pd.to_datetime(df.index).strftime("%Y-%m-%d")

    def pick(name, alt=None):
        if name in df:  # prefer exact
            return df[name]
        if alt and alt in df:
            return df[alt]
        return pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(
        {
            "date": idx,
            "open": pick("Open"),
            "high": pick("High"),
            "low": pick("Low"),
            "close": pick("Close"),
            "adj_close": pick("Adj Close", alt="AdjClose"),
            "volume": pick("Volume"),
            "dividends": pick("Dividends"),
            "splits": pick("Stock Splits", alt="StockSplits"),
        }
    )
    return out.drop_duplicates(subset=["date"]).sort_values("date")


def classify_kind(ticker: str) -> str:
    try:
        tk = yf.Ticker(ticker)
        try:
            info = tk.get_info()
        except Exception:
            info = getattr(tk, "info", {}) or {}
        qt = str(info.get("quoteType") or info.get("quote_type") or "").upper()
        if qt == "ETF":
            return "etf"
        name = (str(info.get("shortName") or info.get("longName") or "")).upper()
        if "ETF" in name:
            return "etf"
    except Exception:
        pass
    return "stock"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_universe(UNIVERSE)
    tickers = [t for t in tickers if not t.startswith("$")]  # skip special symbols
    existing = {p.stem.upper() for p in RAW_DIR.rglob("*.csv")}
    missing = [t for t in tickers if t.upper() not in existing]
    print(f"missing: {len(missing)}")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):
            return x

    failed = []
    no_data = []
    saved = 0
    pbar = tqdm(missing, ncols=80, unit="ticker")
    for t in pbar:
        pbar.set_description(f"Ticker: {t}")
        try:
            df = fetch_raw(t)
        except Exception as e:
            failed.append((t, str(e)))
            continue
        if df.empty:
            no_data.append(t)
            continue
        kind = classify_kind(t)
        base = RAW_DIR / kind
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{t}.csv"
        df.to_csv(path, index=False)
        saved += 1
    print(f"\nSummary: saved {saved} tickers, {len(no_data)} had no data, {len(failed)} failed.")
    if failed:
        print("Failed tickers:")
        for t, err in failed:
            print(f"  {t}: {err}")


if __name__ == "__main__":
    main()
