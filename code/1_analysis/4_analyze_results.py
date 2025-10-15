from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[2]
path = root / "results" / "hedge_eval" / "stock_metrics.parquet"

try:
    df = pd.read_parquet(path)
except FileNotFoundError:
    print(f"Not found: {path}")
    raise SystemExit(1)
except Exception as e:
    print(f"Error reading {path}: {e}")
    raise SystemExit(1)

# Identify available improvement metrics (some may not be present yet)
improve_cols = [c for c in ["he", "te_ratio", "mdd_reduct", "var95_reduct", "alpha_ann"] if c in df.columns]
context_cols = [c for c in ["te_ann", "mdd", "var_95"] if c in df.columns]

if not improve_cols and not context_cols:
    print("No expected metrics found in file.")
    raise SystemExit(1)

def summarize_group(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = []
    for c in cols:
        s = g[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        n = int(s.shape[0])
        if n == 0:
            out.append({"metric": c, "n": 0, "mean": np.nan, "std": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
            continue
        m = float(s.mean())
        sd = float(s.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 0 else np.nan
        ci = 1.96 * se if np.isfinite(se) else np.nan
        out.append({"metric": c, "n": n, "mean": m, "std": sd, "ci_lo": m - ci if np.isfinite(ci) else np.nan, "ci_hi": m + ci if np.isfinite(ci) else np.nan})
    return pd.DataFrame(out)

grp_keys = ["half_life", "k", "horizon"]
have_keys = [k for k in grp_keys if k in df.columns]
if len(have_keys) < 3:
    print("Missing grouping keys; require half_life, k, horizon.")
    raise SystemExit(1)

rows = []
for keys, g in df.groupby(have_keys):
    if not isinstance(keys, tuple):
        keys = (keys,)
    key_map = dict(zip(have_keys, keys))
    if improve_cols:
        summ = summarize_group(g, improve_cols)
        for _, r in summ.iterrows():
            rows.append({**key_map, **r.to_dict()})
    if context_cols:
        summ_ctx = summarize_group(g, context_cols)
        for _, r in summ_ctx.iterrows():
            rows.append({**key_map, **r.to_dict()})

summary = pd.DataFrame(rows).sort_values(["horizon", "metric", "half_life", "k"]).reset_index(drop=True)

print("Summary (mean, sd, 95% CI) by half_life, k, horizon for each metric:")
print(summary.to_string(index=False))
