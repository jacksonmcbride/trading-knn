from pathlib import Path
import json
import re
import time
import pandas as pd
from tqdm import tqdm


def load_candidates_long(asset_class: str = "stock") -> pd.DataFrame:
    root = Path(__file__).resolve().parents[2]
    cand_dir = root / "results" / "candidates" / asset_class
    rows = []
    files = sorted(cand_dir.glob("param=H*.json"))
    t_global_start = time.time()
    # Progress over files
    for path in tqdm(files, desc=f"files[{asset_class}]", ncols=80, unit="file"):
        t0 = time.time()
        m = re.search(r"param=H(\d+)\.json$", path.name)
        if not m:
            continue
        hl = int(m.group(1))
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        n_dates = len(data)
        # Inner progress over dates
        for date_end, targets in tqdm(
            data.items(), desc=path.name, ncols=80, unit="date", leave=False, total=n_dates
        ):
            for target, neighbors in targets.items():
                for rank, item in enumerate(neighbors, start=1):
                    try:
                        nbr, dist = item
                    except Exception:
                        # In case neighbor entries are dicts or malformed
                        if isinstance(item, dict) and "0" in item:
                            nbr, dist = item.get("0"), item.get("1")
                        else:
                            continue
                    rows.append(
                        {
                            "date_end": pd.to_datetime(date_end).date(),
                            "half_life": hl,
                            "target": str(target),
                            "neighbor": str(nbr),
                            "rank": int(rank),
                            "distance": float(dist),
                        }
                    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["date_end", "half_life", "target", "rank"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    elapsed_total = time.time() - t_global_start
    # Save the candidates to Parquet for efficient downstream validation
    try:
        out_parquet = cand_dir / "candidates_long.parquet"
        # Ensure date_end is a pandas datetime (better parquet typing)
        if not df.empty:
            df["date_end"] = pd.to_datetime(df["date_end"])
            df.to_parquet(out_parquet, compression="zstd")
            print(f"Saved candidates: rows={len(df)} -> {out_parquet}")
    except Exception:
        pass
    print(f"Built DataFrame shape={df.shape}; elapsed={elapsed_total:.1f}s")
    return df


# Build the long candidates DataFrame and show quick views
df_long = load_candidates_long("stock")
print("Candidates (long) head:")
print(df_long.head(10))

if not df_long.empty:
    # Show first row per date (top-ranked neighbor) across dates
    per_date_head = (
        df_long.sort_values(["date_end", "rank"]).groupby("date_end", as_index=False).head(1)
    )
    print("\nPer-date first entry (top-ranked) head:")
    print(per_date_head.head(20))
