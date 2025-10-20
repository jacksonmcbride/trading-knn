from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[2]
results_dir = root / "results" / "hedge_eval"
output_dir = root / "results" / "hedge_results"
paths = sorted(results_dir.glob("*.csv"))

cols = [
    "date","target","half_life","k","window",
    "n_obs","he","te_ann","mdd_reduct","var95_reduct",
    "alpha_ann","hit_rate","cum_return_diff",
]

frames = []
for p in paths:
    if not p.is_file():
        continue
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if "target" not in df.columns:
        continue
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    try:
        df["year"] = int(p.stem)
    except Exception:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    frames.append(df[[*cols, "year"]])

if not frames:
    raise SystemExit("No valid hedge result CSVs found.")

df = pd.concat(frames, ignore_index=True)
for c in ["year","half_life","k","window","n_obs","he","te_ann","mdd_reduct","var95_reduct","alpha_ann","hit_rate","cum_return_diff"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["year","half_life","k","window"])  # ensure keys present

g = df.groupby(["year","half_life","k","window"], dropna=False)
rows = g.size().rename("rows")
n_dates = g["date"].nunique().rename("n_dates")
n_tickers = g["target"].nunique().rename("n_tickers")
n_obs_mean = g["n_obs"].mean().rename("n_obs_mean")

he_mean = g["he"].mean().rename("he_mean")
he_sd = g["he"].std(ddof=1).rename("he_sd")
he_se = (he_sd / np.sqrt(rows.replace(0, np.nan))).rename("he_se")
he_ci_low = (he_mean - 1.96 * he_se).rename("he_ci95_low")
he_ci_high = (he_mean + 1.96 * he_se).rename("he_ci95_high")
he_median = g["he"].median().rename("he_median")
he_pos_share = g.apply(lambda x: float(np.mean(pd.to_numeric(x["he"], errors="coerce") > 0))).rename("he_pos_share")

te_ann_mean = g["te_ann"].mean().rename("te_ann_mean")
mdd_reduct_mean = g["mdd_reduct"].mean().rename("mdd_reduct_mean")
var95_reduct_mean = g["var95_reduct"].mean().rename("var95_reduct_mean")
alpha_ann_mean = g["alpha_ann"].mean().rename("alpha_ann_mean")
hit_rate_mean = g["hit_rate"].mean().rename("hit_rate_mean")
crd_mean = g["cum_return_diff"].mean().rename("cum_return_diff_mean")

out = (
    pd.concat([
        rows, n_dates, n_tickers, n_obs_mean,
        he_mean, he_sd, he_se, he_ci_low, he_ci_high, he_median, he_pos_share,
        te_ann_mean, mdd_reduct_mean, var95_reduct_mean, alpha_ann_mean, hit_rate_mean, crd_mean,
    ], axis=1)
    .reset_index()
    .sort_values(["year","half_life","k","window"]) 
)

out_path = output_dir / "year_hlk_summary.csv"
out.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

# Print top/bottom 5 rows by he_mean
valid = out.dropna(subset=["he_mean"])
if not valid.empty:
    cols_print = ["year","half_life","k","window","rows","he_mean"]
    top5 = valid.sort_values("he_mean", ascending=False)[cols_print].head(5)
    bot5 = valid.sort_values("he_mean", ascending=True)[cols_print].head(5)
    print("Top 5 (by he_mean):")
    print(top5.to_string(index=False))
    print("\nBottom 5 (by he_mean):")
    print(bot5.to_string(index=False))
