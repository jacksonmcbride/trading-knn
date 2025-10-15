from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Minimal, single-file evaluator:
# - Iterates dates in candidates parquet
# - Loads returns only for needed tickers
# - Computes hedge metrics per (date_end, target, half_life, k, horizon)

root = Path(__file__).resolve().parents[2]
cand_path = root / "results" / "candidates" / "stock" / "candidates_long.parquet"
returns_path = root / "data" / "processed" / "returns" / "stock_returns.parquet"
out_path = root / "results" / "hedge_eval" / "stock_metrics.parquet"

K_LIST = [1, 3, 5, 10, 30]
HORIZONS = {
    "1w": pd.Timedelta(days=7),
    "1m": pd.Timedelta(days=30),
    "3m": pd.Timedelta(days=90),
    "6m": pd.Timedelta(days=180),
    "1y": pd.Timedelta(days=365),
}

EPS = 1e-6


def _max_drawdown(cum: np.ndarray) -> float:
    if cum.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd)) if dd.size else float("nan")


# Load candidates (long)
dfc = pd.read_parquet(cand_path)
if not pd.api.types.is_datetime64_any_dtype(dfc["date_end"]):
    dfc["date_end"] = pd.to_datetime(dfc["date_end"])  # ensure timestamp

# Process in date batches (and half_life) to minimize IO
group_cols = ["date_end", "half_life"]
dfc = dfc[["date_end", "half_life", "target", "neighbor", "rank", "distance"]].sort_values(group_cols + ["target", "rank"])  # minimal cols

rows = []
BATCH_ROWS = 100000
writer = None

total_groups = dfc[group_cols].drop_duplicates().shape[0]
for key, g in tqdm(dfc.groupby(group_cols), total=total_groups, desc="dates", ncols=80):
    # Robustly unpack group key
    if isinstance(key, tuple):
        date_end, half_life = key
    else:
        date_end = key
        half_life = int(g["half_life"].iloc[0])
    date_end = pd.Timestamp(date_end)
    # Tickerset for this date
    targets = g["target"].unique().tolist()
    neighbors = g["neighbor"].unique().tolist()
    cols = sorted(set(targets) | set(neighbors))
    if not cols:
        continue

    # Load only needed columns; restrict to (date_end, date_end + 1y]
    try:
        ret = pd.read_parquet(returns_path, columns=cols)
    except Exception:
        # Fallback: read full and then prune if columns filter unsupported
        ret = pd.read_parquet(returns_path)[cols]
    ret.index = pd.to_datetime(ret.index)
    end_1y = date_end + HORIZONS["1y"]
    ret = ret[(ret.index > date_end) & (ret.index <= end_1y)]
    if ret.empty:
        continue

    # Per target computations
    for t in targets:
        if t not in ret.columns:
            continue
        r_tgt_full = ret[t]
        sub = g[g["target"] == t].sort_values("rank").head(max(K_LIST))
        nbrs = [x for x in sub["neighbor"].tolist() if x in ret.columns]
        if len(nbrs) == 0:
            continue
        dists = sub.set_index("neighbor").loc[nbrs, "distance"].astype(float).to_numpy(dtype=np.float64)
        R = ret[nbrs]  # T x M

        # Base weights from distances (L2-style)
        w_base = 1.0 / (np.maximum(dists, EPS) + EPS)

        for h_name, delta in HORIZONS.items():
            rwin = R[(R.index > date_end) & (R.index <= date_end + delta)]
            tgt = r_tgt_full[(r_tgt_full.index > date_end) & (r_tgt_full.index <= date_end + delta)]
            if rwin.empty or tgt.empty:
                continue

            # Align index first to avoid repeated alignment cost per k
            idx = tgt.index.intersection(rwin.index)
            if idx.empty:
                continue
            tgt_i = tgt.loc[idx]
            R_i = rwin.loc[idx]
            # Convert to numpy once per horizon for speed
            tgt_arr = tgt_i.to_numpy(dtype=np.float32)
            R_arr = R_i.to_numpy(dtype=np.float32)
            neigh_ok = ~np.isnan(R_arr)
            tgt_ok = ~np.isnan(tgt_arr)

            for k in K_LIST:
                m = min(k, R_i.shape[1])
                if m <= 0:
                    continue
                # Slice arrays instead of DataFrame ops
                X = R_arr[:, :m]
                wk = w_base[:m]
                denom = float(np.sum(wk))
                if denom <= 0:
                    continue
                # Valid rows: target and all k neighbors must be non-NaN
                mask = tgt_ok & np.all(neigh_ok[:, :m], axis=1)
                if not mask.any():
                    continue
                rt = tgt_arr[mask]
                X = X[mask]
                wnorm = (wk / denom).astype(np.float32)
                port = X @ wnorm  # T
                diff = rt - port
                n_obs = int(diff.shape[0])
                if n_obs < 5:
                    continue
                var_t = float(np.var(rt))
                var_d = float(np.var(diff))
                he = float(1.0 - (var_d / var_t)) if var_t > 0 else float("nan")
                te_ann = float(np.std(diff) * np.sqrt(252.0))
                mdd = _max_drawdown(np.cumsum(diff))
                p5 = float(np.percentile(diff, 5))
                var95 = float(max(0.0, -p5))
                # Additional effectiveness metrics vs target
                eps_denom = 1e-12
                std_t = float(np.std(rt))
                te_ratio = float(np.std(diff) / (std_t + eps_denom)) if std_t > 0 else float("nan")
                mdd_t = _max_drawdown(np.cumsum(rt))
                mdd_reduct = float("nan") if not np.isfinite(mdd_t) or mdd_t <= 0 else float(1.0 - (mdd / (mdd_t + eps_denom)))
                mdd_abs_reduct = float("nan") if not np.isfinite(mdd_t) else float(mdd_t - mdd)
                p5_t = float(np.percentile(rt, 5))
                var95_t = float(max(0.0, -p5_t))
                var95_reduct = float("nan") if var95_t <= 0 else float(1.0 - (var95 / (var95_t + eps_denom)))
                var95_abs_reduct = float("nan") if not np.isfinite(var95_t) else float(var95_t - var95)
                alpha_ann = float(252.0 * np.mean(diff))
                rows.append(
                    {
                        "date_end": date_end,
                        "target": t,
                        "half_life": int(half_life),
                        "k": int(k),
                        "horizon": h_name,
                        "he": he,
                        "te_ann": te_ann,
                        "mdd": mdd,
                        "te_ratio": te_ratio,
                        "mdd_reduct": mdd_reduct,
                        "mdd_abs_reduct": mdd_abs_reduct,
                        "var_95": var95,
                        "var95_reduct": var95_reduct,
                        "var95_abs_reduct": var95_abs_reduct,
                        "alpha_ann": alpha_ann,
                        "n_obs": n_obs,
                    }
                )
                # Flush batch to Parquet to cap memory
                if len(rows) >= BATCH_ROWS:
                    batch_df = pd.DataFrame(rows)
                    batch_df["date_end"] = pd.to_datetime(batch_df["date_end"])  # ensure dtype
                    table = pa.Table.from_pandas(batch_df, preserve_index=False)
                    if writer is None:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                    writer.write_table(table)
                    rows.clear()
    # Free memory early
    del ret

# Final flush
if rows:
    batch_df = pd.DataFrame(rows)
    batch_df["date_end"] = pd.to_datetime(batch_df["date_end"])  # ensure dtype
    table = pa.Table.from_pandas(batch_df, preserve_index=False)
    if writer is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()

if writer is None:
    # No data; write an empty file with schema
    empty = pd.DataFrame(
        columns=[
            "date_end", "target", "half_life", "k", "horizon",
            "he", "te_ann", "te_ratio", "mdd", "mdd_reduct", "mdd_abs_reduct",
            "var_95", "var95_reduct", "var95_abs_reduct", "alpha_ann", "n_obs"
        ]
    )
    table = pa.Table.from_pandas(empty, preserve_index=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
else:
    writer.close()
