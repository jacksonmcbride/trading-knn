from pathlib import Path
import pandas as pd

NASDAQ_URL = "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
CONFIGS = BASE_DIR / "configs"
UNIVERSE = CONFIGS / "universe.csv"


def load_nasdaq() -> pd.DataFrame:
    df = pd.read_csv(NASDAQ_URL, sep="|", dtype=str)
    df = df[df["Symbol"].notna() & (df["Test Issue"] == "N")]
    df = df[df["Symbol"] != "File Creation Time"]
    df = df.rename(columns={"Symbol": "symbol"})
    df["exchange"] = "NASDAQ"
    return df[["symbol", "exchange"]]


def load_other() -> pd.DataFrame:
    df = pd.read_csv(OTHER_URL, sep="|", dtype=str)
    df = df[df["ACT Symbol"].notna() & (df["Test Issue"] == "N")]
    df = df.rename(columns={"ACT Symbol": "symbol", "Exchange": "ex"})
    exmap = {"N": "NYSE", "A": "NYSEMKT", "P": "NYSEARCA"}
    df["exchange"] = df["ex"].map(exmap).fillna(df["ex"]).astype(str)
    return df[["symbol", "exchange"]]


def to_yahoo(sym: str) -> str:
    return sym.strip().upper().replace(".", "-")


def main():
    TEMP_DIR.mkdir(exist_ok=True)
    CONFIGS.mkdir(exist_ok=True)

    nas = load_nasdaq()
    oth = load_other()

    nas.to_csv(TEMP_DIR / "nasdaqlisted.csv", index=False)
    oth.to_csv(TEMP_DIR / "otherlisted.csv", index=False)

    all_syms = pd.concat([nas, oth], ignore_index=True).drop_duplicates()
    all_syms["ticker"] = all_syms["symbol"].map(to_yahoo)
    out = all_syms[["ticker", "exchange"]].dropna().drop_duplicates().sort_values(["exchange", "ticker"])  # minimal
    out.to_csv(UNIVERSE, index=False)
    print(f"wrote {UNIVERSE} with {len(out)} rows")


if __name__ == "__main__":
    main()

