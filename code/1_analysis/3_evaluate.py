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

# Process in date batches to minimize IO (handle half_life inside date loop)
group_cols = ["date_end"]
dfc = dfc[["date_end", "half_life", "target", "neighbor", "rank", "distance"]].sort_values(["date_end", "half_life", "target", "rank"])  # minimal cols

rows = []
BATCH_ROWS = 100000
writer = None

total_groups = dfc[group_cols].drop_duplicates().shape[0]
print(f"Total date groups to process: {total_groups}")
for key, g_date in tqdm(dfc.groupby(group_cols), total=total_groups, desc="dates", ncols=80):
    # Unpack key
    date_end = pd.Timestamp(key if not isinstance(key, tuple) else key[0])
    # Tickerset for this date across all half_life groups
    targets = g_date["target"].unique().tolist()
    neighbors = g_date["neighbor"].unique().tolist()
    cols = sorted(set(targets) | set(neighbors))
    if not cols:
        continue

    # Compute global pre-window to cover all half_life groups at this date (super-set, exact slices applied later)
    # Approx pre length per half-life: ~6*half_life trading days ≈ 6*H * 7/5 calendar days; min 30 days
    if not g_date["half_life"].empty:
        pre_days_list = [int(max(30, round(6 * float(h) * 7.0 / 5.0))) for h in g_date["half_life"].unique()]
        pre_cal_days = max(pre_days_list) if pre_days_list else 30
    else:
        pre_cal_days = 30
    pre_start_global = date_end - pd.Timedelta(days=pre_cal_days)

    # Single read covering both pre-window and post 1y window; then slice
    try:
        ret_all = pd.read_parquet(returns_path, columns=cols)
    except Exception:
        ret_all = pd.read_parquet(returns_path)[cols]
    ret_all.index = pd.to_datetime(ret_all.index)
    end_1y = date_end + HORIZONS["1y"]
    ret_all = ret_all[(ret_all.index > pre_start_global) & (ret_all.index <= end_1y)]
    if ret_all.empty:
        continue
    # Post and pre slices
    ret = ret_all[(ret_all.index > date_end) & (ret_all.index <= end_1y)]
    ret_pre_all = ret_all[(ret_all.index > pre_start_global) & (ret_all.index <= date_end)]
    if ret.empty:
        continue

    # Iterate per half-life within this date to reuse same returns
    for half_life, g in g_date.groupby("half_life"):
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

        # Compute hedge ratios b_hat per k using pre-window data (no lookahead)
        b_hat = {}
        if t in ret_pre_all.columns and not ret_pre_all.empty:
            # Slice the global pre window (already loaded) — exact same dates per half-life as before when constrained by masks
            r_tgt_pre = ret_pre_all[t]
            R_pre = ret_pre_all[nbrs] if all(n in ret_pre_all.columns for n in nbrs) else ret_pre_all[[c for c in nbrs if c in ret_pre_all.columns]]
            if not R_pre.empty and t in ret_pre_all.columns:
                idx_pre = r_tgt_pre.index.intersection(R_pre.index)
                # Constrain to this half-life's pre-window length to preserve logic
                pre_days_hl = int(max(30, round(6 * float(half_life) * 7.0 / 5.0)))
                win_start = date_end - pd.Timedelta(days=pre_days_hl)
                if len(idx_pre) > 0:
                    idx_pre = idx_pre[idx_pre > win_start]
                if not idx_pre.empty:
                    rt_pre_arr = r_tgt_pre.loc[idx_pre].to_numpy(dtype=np.float32)
                    Rp_pre_arr = R_pre.loc[idx_pre].to_numpy(dtype=np.float32)
                    neigh_ok_pre = ~np.isnan(Rp_pre_arr)
                    tgt_ok_pre = ~np.isnan(rt_pre_arr)
                    for k in K_LIST:
                        m = min(k, Rp_pre_arr.shape[1])
                        if m <= 0:
                            continue
                        Xp = Rp_pre_arr[:, :m]
                        wk = w_base[:m]
                        denom_w = float(np.sum(wk))
                        if denom_w <= 0:
                            continue
                        mask_pre = tgt_ok_pre & np.all(neigh_ok_pre[:, :m], axis=1)
                        if not np.any(mask_pre):
                            continue
                        rt_p = rt_pre_arr[mask_pre]
                        Xp_v = Xp[mask_pre]
                        wnorm = (wk / denom_w).astype(np.float32)
                        port_p = Xp_v @ wnorm
                        den = float(np.dot(port_p, port_p))
                        if den > 0:
                            num = float(np.dot(port_p, rt_p))
                            b_hat[m] = num / den
        # default b=1 for any k lacking a pre-window estimate

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
                # Scale hedge by b_hat estimated on pre-window (fallback to 1.0)
                bh = b_hat.get(m, 1.0)
                diff = rt - (bh * port)
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
