from pathlib import Path
import argparse
import re
import pandas as pd


def find_default_year(results_dir: Path) -> int | None:
    if not results_dir.exists():
        return None
    years = []
    for p in results_dir.glob("*.csv"):
        m = re.match(r"(\d{4})\.csv$", p.name)
        if m:
            years.append(int(m.group(1)))
    return max(years) if years else None


def load_year(results_dir: Path, year: int) -> pd.DataFrame:
    path = results_dir / f"{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path)
    # Ensure expected columns exist
    expected = {"date", "target", "half_life", "k", "window", "n_obs", "he", "te_ann"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    # Coerce numeric
    for c in ["half_life", "k", "window", "n_obs", "he", "te_ann"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    out = (
        df.groupby(by, dropna=False)
        .agg(
            rows=("he", "size"),
            he_mean=("he", "mean"),
            te_ann_mean=("te_ann", "mean"),
            n_obs_mean=("n_obs", "mean"),
        )
        .reset_index()
        .sort_values(by)
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Summarize hedge results for a single year.")
    parser.add_argument("--year", type=int, default=None, help="Year to summarize (default: latest available)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results" / "hedge_results"

    year = args.year or find_default_year(results_dir)
    if year is None:
        print(f"No results found in {results_dir}")
        return

    df = load_year(results_dir, year)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 120)
    pd.set_option("display.precision", 4)

    print(f"Year: {year}  |  Rows: {len(df)}  |  Unique dates: {df['date'].nunique()}")

    print("\nAverages By k:")
    print(summarize(df, "k").to_string(index=False))

    print("\nAverages By Half-Life:")
    print(summarize(df, "half_life").to_string(index=False))

    print("\nAverages By Lookforward (window):")
    print(summarize(df, "window").to_string(index=False))

    # Combined: by half_life, k, and window, sorted best (highest he_mean) to worst
    combo = (
        df.groupby(["half_life", "k", "window"], dropna=False)
          .agg(rows=("he", "size"), he_mean=("he", "mean"), te_ann_mean=("te_ann", "mean"), n_obs_mean=("n_obs", "mean"))
          .reset_index()
          .sort_values(["he_mean", "half_life", "k", "window"], ascending=[False, True, True, True])
    )
    print("\nAverages By Half-Life, k, Window (best → worst by he_mean):")
    print(combo.to_string(index=False))


if __name__ == "__main__":
    main()
