from pathlib import Path
import pandas as pd
import numpy as np
import math

root = Path(__file__).resolve().parents[2]
results_dir = root / "results" / "hedge_eval"
output_dir = root / "results" / "hedge_results"
paths = sorted(results_dir.glob("*.csv"))
cols = [
    "date","target","half_life","k","window",
    "n_obs","he","te_ann","mdd_reduct","var95_reduct",
    "alpha_ann","hit_rate","cum_return_diff"
]
frames = []
for p in paths:
    if not p.is_file():
        continue
    try:
        f = pd.read_csv(p)
    except Exception:
        continue
    if "target" not in f.columns:
        continue
    for c in cols:
        if c not in f.columns:
            f[c] = np.nan
    # add year from filename if possible
    try:
        f["year"] = int(p.stem)
    except Exception:
        pass
    frames.append(f[[*cols, "year"]] if "year" in f.columns else f[cols])
if not frames:
    raise SystemExit("No valid hedge result CSVs with required columns were found.")
df = pd.concat(frames, ignore_index=True)

# Coerce numerics
for c in ["half_life","k","window","n_obs","he","te_ann","mdd_reduct","var95_reduct","alpha_ann","hit_rate","cum_return_diff","year"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Coverage per ticker
base = (
    df.groupby("target", dropna=False)
      .agg(count=("he","size"), n_dates=("date","nunique"))
      .reset_index()
)

WINDOWS = [5, 21, 126, 252]
def per_window(w):
    g = df[df["window"]==w].copy()
    if g.empty:
        return pd.DataFrame({"target": []})
    g["he_pos"] = (g["he"]>0).astype(float)
    agg = (
        g.groupby("target", dropna=False)
         .agg(
             **{f"he_mean_{w}": ("he","mean")},
             **{f"he_median_{w}": ("he","median")},
             **{f"he_pos_share_{w}": ("he_pos","mean")},
             **{f"te_ann_mean_{w}": ("te_ann","mean")},
             **{f"mdd_reduct_mean_{w}": ("mdd_reduct","mean")},
             **{f"var95_reduct_mean_{w}": ("var95_reduct","mean")},
             **{f"alpha_ann_mean_{w}": ("alpha_ann","mean")},
             **{f"hit_rate_mean_{w}": ("hit_rate","mean")},
         )
         .reset_index()
    )
    return agg

out = base.copy()
for w in WINDOWS:
    out = out.merge(per_window(w), on="target", how="left")

# Overall metrics (across all windows)
df["he_pos"] = (df["he"]>0).astype(float)
overall = (
    df.groupby("target", dropna=False)
      .agg(he_mean_overall=("he","mean"), he_pos_share_overall=("he_pos","mean"), te_ann_mean_overall=("te_ann","mean"))
      .reset_index()
)
out = out.merge(overall, on="target", how="left")

# Attach metadata (Name, MarketCap, Sector, Industry) from data/processed/stock_metadata/nyse.csv and data/processed/stock_metadata/nasdaq.csv
data_dir = root / "data" / "processed" / "stock_metadata"
meta_frames = []
for fname in ("nyse.csv", "nasdaq.csv"):
    p = data_dir / fname
    if p.exists():
        try:
            m = pd.read_csv(p)
            # Determine symbol column
            sym_col = "Symbol" if "Symbol" in m.columns else ("Ticker" if "Ticker" in m.columns else ("target" if "target" in m.columns else None))
            if sym_col is None:
                continue
            cols_keep = [sym_col] + [c for c in ["Name", "MarketCap", "Sector", "industry"] if c in m.columns]
            m = m[cols_keep].copy()
            m.rename(columns={sym_col: "target"}, inplace=True)
            m = m.drop_duplicates(subset=["target"], keep="first")
            meta_frames.append(m)
        except Exception:
            pass
meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates(subset=["target"], keep="first") if meta_frames else pd.DataFrame(columns=["target"]) 
out = out.merge(meta, on="target", how="left")

# Reorder columns to place metadata at the beginning
meta_order = [c for c in ["Name", "MarketCap", "Sector", "industry"] if c in out.columns]
front = ["target"] + meta_order
rest = [c for c in out.columns if c not in front]
out = out[front + rest]

# Save and print top/bottom by overall hedge effectiveness
out_path = output_dir / "ticker_summary.csv"
out.sort_values(["target"]).to_csv(out_path, index=False)

best = out.sort_values("he_mean_overall", ascending=False).head(10)[["target","count","he_mean_overall","he_pos_share_overall","te_ann_mean_overall"]]
worst = out.sort_values("he_mean_overall", ascending=True).head(10)[["target","count","he_mean_overall","he_pos_share_overall","te_ann_mean_overall"]]

print("Top tickers (by he_mean_overall):")
print(best.to_string(index=False))
print("\nBottom tickers (by he_mean_overall):")
print(worst.to_string(index=False))
print(f"\nSaved: {out_path}")

# -----------------------------
# Per-run results by Industry and Sector (multi-metric)
# -----------------------------
try:
    meta_subset_cols = ["target"] + [c for c in ["Sector", "industry"] if c in meta.columns]
    if len(meta_subset_cols) > 1:
        df_runs = df.merge(meta[meta_subset_cols], on="target", how="left")

        metrics = ["he", "te_ann", "mdd_reduct", "var95_reduct", "alpha_ann"]

        def summarize_group(g: pd.DataFrame) -> pd.Series:
            outd = {"rows": int(g.shape[0])}
            for m in metrics:
                s = pd.to_numeric(g[m], errors="coerce").dropna()
                n = int(s.shape[0])
                mean = float(s.mean()) if n > 0 else float("nan")
                sd = float(s.std(ddof=1)) if n > 1 else float("nan")
                if n > 1 and pd.notna(sd) and sd > 0:
                    se = sd / np.sqrt(n)
                    t = mean / se
                    z = abs(float(t))
                    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
                else:
                    p = float("nan")
                outd[f"{m}_mean"] = mean
                outd[f"{m}_sd"] = sd
                outd[f"{m}_p"] = float(p)
                outd[f"{m}_n"] = n
            return pd.Series(outd)

        # Industry-level (include both Sector and industry columns)
        if set(["Sector", "industry"]).issubset(df_runs.columns):
            ind = (
                df_runs.groupby(["Sector", "industry"], dropna=False)
                       .apply(summarize_group)
                       .reset_index()
            )
            out_industry = output_dir / "industry_metrics.csv"
            ind.to_csv(out_industry, index=False)
            print(f"Saved: {out_industry}")
            # Print top/bottom 3 industries by hedge effectiveness
            if "he_mean" in ind.columns:
                valid = ind.dropna(subset=["he_mean"]).copy()
                if not valid.empty:
                    cols_print = ["Sector", "industry", "he_mean", "he_sd", "he_p", "he_n", "rows"]
                    top3 = valid.sort_values("he_mean", ascending=False)[cols_print].head(3)
                    bot3 = valid.sort_values("he_mean", ascending=True)[cols_print].head(3)
                    print("Top 3 industries (by he_mean):")
                    print(top3.to_string(index=False))
                    print("\nBottom 3 industries (by he_mean):")
                    print(bot3.to_string(index=False))

        # Sector-level
        if "Sector" in df_runs.columns:
            sec = (
                df_runs.groupby(["Sector"], dropna=False)
                       .apply(summarize_group)
                       .reset_index()
            )
            out_sector = output_dir / "sector_metrics.csv"
            sec.to_csv(out_sector, index=False)
            print(f"Saved: {out_sector}")
            # Print top/bottom 3 sectors by hedge effectiveness
            if "he_mean" in sec.columns:
                valid = sec.dropna(subset=["he_mean"]).copy()
                if not valid.empty:
                    cols_print = ["Sector", "he_mean", "he_sd", "he_p", "he_n", "rows"]
                    top3 = valid.sort_values("he_mean", ascending=False)[cols_print].head(3)
                    bot3 = valid.sort_values("he_mean", ascending=True)[cols_print].head(3)
                    print("Top 3 sectors (by he_mean):")
                    print(top3.to_string(index=False))
                    print("\nBottom 3 sectors (by he_mean):")
                    print(bot3.to_string(index=False))
except Exception as e:
    print(f"Industry/Sector metrics skipped: {e}")
