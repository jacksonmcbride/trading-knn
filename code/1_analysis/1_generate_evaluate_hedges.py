import numpy as np
import pandas as pd
from tqdm import tqdm

EPS = 1e-12
# Limit neighbor search to top-N candidate tickers by data availability in lookback
CANDIDATE_CAP = 1000


def _max_drawdown(cum: np.ndarray) -> float:
    if cum.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd)) if dd.size else float("nan")


def _metrics_from_arrays(t: np.ndarray, p: np.ndarray) -> dict:
    diff = t - p
    n = int(diff.size)
    if n == 0:
        return {"n_obs": 0}

    std_t = float(np.std(t))
    var_t = float(std_t * std_t)
    std_d = float(np.std(diff))
    var_d = float(std_d * std_d)
    he = float(1.0 - (var_d / var_t)) if var_t > 0 else float("nan")
    var_reduct_abs = float(var_t - var_d) if np.isfinite(var_t) and np.isfinite(var_d) else float("nan")

    te_ann = float(std_d * np.sqrt(252.0))
    te_ratio = float(std_d / (float(std_t) + EPS)) if std_t > 0 else float("nan")

    mdd_diff = _max_drawdown(np.cumsum(diff))
    mdd_t = _max_drawdown(np.cumsum(t))
    mdd_reduct_pct = float("nan") if not np.isfinite(mdd_t) or mdd_t <= 0 else float(1.0 - (mdd_diff / (mdd_t + EPS)))
    mdd_abs_reduct = float("nan") if not np.isfinite(mdd_t) else float(mdd_t - mdd_diff)

    p5_d = float(np.percentile(diff, 5))
    var95 = float(max(0.0, -p5_d))
    p5_t = float(np.percentile(t, 5))
    var95_t = float(max(0.0, -p5_t))
    var95_reduct = float("nan") if var95_t <= 0 else float(1.0 - (var95 / (var95_t + EPS)))
    var95_abs_reduct = float("nan") if not np.isfinite(var95_t) else float(var95_t - var95)

    alpha_ann = float(252.0 * np.mean(diff))
    hit_rate = float(np.mean(diff > 0.0))

    cum_t = float(np.prod(1.0 + t) - 1.0)
    cum_p = float(np.prod(1.0 + p) - 1.0)
    cum_diff = float(cum_t - cum_p)

    return {
        "n_obs": n,
        "he": he,
        "var_reduct_abs": var_reduct_abs,
        "te_ann": te_ann,
        "te_ratio": te_ratio,
        "mdd": mdd_diff,
        "mdd_reduct": mdd_reduct_pct,
        "mdd_abs_reduct": mdd_abs_reduct,
        "var_95": var95,
        "var95_reduct": var95_reduct,
        "var95_abs_reduct": var95_abs_reduct,
        "alpha_ann": alpha_ann,
        "hit_rate": hit_rate,
        "cum_return_target": cum_t,
        "cum_return_portfolio": cum_p,
        "cum_return_diff": cum_diff,
    }


def evaluate_pair(target: pd.Series, portfolio: pd.Series) -> dict:
    """
    Compute concise hedge metrics from two return series.
    Both inputs can be misaligned and contain missing values.
    Metrics use overlapping, non-missing rows only.
    """
    # Align and mask
    idx = target.index.intersection(portfolio.index)
    if len(idx) == 0:
        return {"n_obs": 0}
    t = pd.to_numeric(target.loc[idx], errors="coerce").to_numpy(dtype=np.float64)
    p = pd.to_numeric(portfolio.loc[idx], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(p)
    if not np.any(mask):
        return {"n_obs": 0}
    t = t[mask]
    p = p[mask]
    return _metrics_from_arrays(t, p)


def evaluate_horizons(target: pd.Series, portfolio: pd.Series, lengths: list[int], show_progress: bool = False) -> list[dict]:
    """Evaluate multiple lookforward lengths using nested subsetting (longest → shortest).
    lengths are counts of observations (e.g., 252, 90, 30).
    If show_progress, displays a tqdm bar over horizons.
    """
    # Align once
    idx = target.index.intersection(portfolio.index)
    if len(idx) == 0:
        return []
    t = pd.to_numeric(target.loc[idx], errors="coerce").to_numpy(dtype=np.float64)
    p = pd.to_numeric(portfolio.loc[idx], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(p)
    t = t[mask]
    p = p[mask]

    res = []
    Ls = [int(x) for x in lengths if int(x) > 0]
    Ls = sorted({x for x in Ls if x <= t.size}, reverse=True)
    iterator = tqdm(Ls, desc="horizons", ncols=80, leave=False) if show_progress else Ls
    for L in iterator:
        m = _metrics_from_arrays(t[-L:], p[-L:])
        m["window_len"] = int(L)
        res.append(m)
    return res


def evaluate_over_k(
    target: pd.Series,
    portfolios: dict,
    lengths: list[int],
    ks: list[int],
    show_progress_k: bool = False,
    show_progress_horizons: bool = False,
    source: str | None = None,
) -> list[dict]:
    """Evaluate a dict of portfolios (one per k) across horizons.
    portfolios keys are assumed as 'p0','p1',... matching ks order.
    If show_progress_k, displays a tqdm bar over ks.
    """
    # Order portfolios by numeric suffix to align with ks
    items = sorted(portfolios.items(), key=lambda kv: int(str(kv[0])[1:]) if str(kv[0]).startswith('p') else 0)
    rows = []
    iterator = tqdm(range(len(items)), desc=(f"{source} k" if source else "k"), ncols=80) if show_progress_k else range(len(items))
    for i in iterator:
        label, p = items[i]
        k = ks[i] if i < len(ks) else None
        res = evaluate_horizons(target, p, lengths, show_progress=show_progress_horizons)
        for m in res:
            row = {"k": k, "window": m["window_len"], "n_obs": m["n_obs"], "he": m["he"], "te_ann": m["te_ann"]}
            if source is not None:
                row["source"] = source
            rows.append(row)
    return rows


def build_series(df: pd.DataFrame, target: str, weights) -> tuple[pd.Series, dict]:
    """Return target series and dict of portfolio series for one or many weight dicts."""
    # Normalize to list of dicts
    if isinstance(weights, dict):
        weights_list = [weights]
    else:
        weights_list = list(weights)
    # Collect required columns
    needed = {target}
    for w in weights_list:
        needed.update(w.keys())
    sub = df[list(needed)].copy()
    t = pd.to_numeric(sub[target], errors="coerce")
    portfolios = {}
    for i, w in enumerate(weights_list):
        if w:
            ws = pd.Series(w, dtype="float64")
            p = sub[ws.index].mul(ws.values, axis=1).sum(axis=1)
        else:
            p = pd.Series(index=sub.index, dtype="float64")
        portfolios[f"p{i}"] = pd.to_numeric(p, errors="coerce")
    return t, portfolios


def subset_returns(
    stock_df: pd.DataFrame,
    etf_df: pd.DataFrame | None,
    start_date,
    target: str,
    neighbors: list,
    end_date=None,
    neighbors_source: str = "stock",
) -> pd.DataFrame:
    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date) if end_date is not None else sd + pd.Timedelta(days=365)
    stock_df = stock_df.copy()
    stock_df.index = pd.to_datetime(stock_df.index)
    if etf_df is not None:
        etf_df = etf_df.copy()
        etf_df.index = pd.to_datetime(etf_df.index)
    win = (slice(None),)
    tgt = stock_df.loc[(stock_df.index > sd) & (stock_df.index <= ed), [target]] if target in stock_df.columns else pd.DataFrame(index=stock_df.index, columns=[target])
    src = stock_df if neighbors_source == "stock" else etf_df
    neigh = [n for n in neighbors if src is not None and n in src.columns and n != target]
    nbr = src.loc[(src.index > sd) & (src.index <= ed), neigh] if src is not None and len(neigh) else pd.DataFrame(index=tgt.index)
    return tgt.join(nbr, how="inner")


def make_weights(distances: dict, ks: list[int], eps: float = 1e-12) -> dict:
    """For each k, pick k smallest distances and assign inverse-distance weights that sum to 1."""
    items = [(t, float(d)) for t, d in distances.items() if np.isfinite(d)]
    items.sort(key=lambda x: x[1])
    out = {}
    for k in sorted({int(x) for x in ks if int(x) > 0}):
        top = items[:k]
        if not top:
            out[k] = {}
            continue
        w = np.array([1.0 / (d + eps) for _, d in top], dtype=float)
        s = float(w.sum())
        if s <= 0:
            out[k] = {t: 0.0 for t, _ in top}
        else:
            w /= s
            out[k] = {t: float(w[i]) for i, (t, _) in enumerate(top)}
    return out


def find_neighbors(weighted_df: pd.DataFrame, target: str, k: int = 60) -> dict:
    """Return dict of {ticker: distance} using correlation distance over overlapping rows.
    Uses distance = 1 - |corr(target, asset)| (smaller is stronger relationship).
    """
    if target not in weighted_df.columns:
        return {}
    t = pd.to_numeric(weighted_df[target], errors="coerce").to_numpy(dtype=float)
    other_cols = [c for c in weighted_df.columns if c != target]
    if not other_cols:
        return {}
    X = pd.DataFrame({c: pd.to_numeric(weighted_df[c], errors="coerce") for c in other_cols}).to_numpy(dtype=np.float32)
    tm = np.isfinite(t)[:, None]
    Xm = np.isfinite(X)
    M = tm & Xm
    # Weighted by overlap only: compute corr per column w.r.t target
    prod = t[:, None] * X
    prod[~M] = 0.0
    num = prod.sum(axis=0)
    t2 = (t * t)[:, None]
    x2 = (X * X)
    t2[~tm] = 0.0
    x2[~Xm] = 0.0
    den_t = (t2 * M).sum(axis=0)
    den_x = (x2 * M).sum(axis=0)
    den = np.sqrt(den_t * den_x)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = num / den
    corr = np.clip(corr, -1.0, 1.0)
    counts = M.sum(axis=0)
    min_overlap = 10
    corr[(den <= 0) | (counts < min_overlap)] = np.nan
    dist = 1.0 - np.abs(corr)
    dist[~np.isfinite(dist)] = np.inf
    order = np.argsort(dist)
    top = [(other_cols[i], float(dist[i])) for i in order[:min(k, len(other_cols))] if np.isfinite(dist[i])]
    return {c: d for c, d in top}

def neighbors_by_halflife(df: pd.DataFrame, target: str, halflifes: list[int], k: int = 60) -> dict:
    """For each half-life, compute weighted lookback (6×HL) and return top-k neighbor distances.
    Assumes df is indexed by time in ascending order; uses the last rows as lookback.
    """
    out = {}
    for hl in sorted({int(h) for h in halflifes if int(h) > 0}):
        L = int(6 * hl)
        look = df.tail(L)
        if look.empty or target not in look.columns:
            out[hl] = {}
            continue
        lam = np.log(2.0) / float(hl)
        n = look.shape[0]
        w = pd.Series(np.exp(-lam * np.arange(n - 1, -1, -1, dtype=float)), index=look.index, dtype="float64")
        # Weighted z-score per column within lookback (vectorized)
        X = look.to_numpy(dtype=float)
        Fin = np.isfinite(X)
        wv = w.to_numpy(dtype=float)[:, None]
        sw = (wv * Fin).sum(axis=0)
        X_f = np.where(Fin, X, 0.0)
        mu = (wv * X_f).sum(axis=0) / np.where(sw > 0, sw, np.nan)
        diff = np.where(Fin, X - mu, 0.0)
        var = (wv * diff * diff).sum(axis=0) / np.where(sw > 0, sw, np.nan)
        sd = np.sqrt(var)
        sd_pos = np.isfinite(sd) & (sd > 0)
        sd_row = sd[None, :]
        mu_row = mu[None, :]
        Z = np.full_like(X, np.nan, dtype=float)
        # Compute only for columns with valid sd to avoid invalid divisions
        valid_idx = np.flatnonzero(sd_pos)
        if valid_idx.size > 0:
            Z_valid = (X[:, valid_idx] - mu_row[:, valid_idx]) / sd_row[:, valid_idx]
            Fin_valid = Fin[:, valid_idx]
            Z_slice = Z[:, valid_idx]
            Z_slice[Fin_valid] = Z_valid[Fin_valid]
            Z[:, valid_idx] = Z_slice
        # For columns with invalid sd but finite data, set z to 0
        invalid_idx = np.flatnonzero(~sd_pos)
        if invalid_idx.size > 0:
            Fin_invalid = Fin[:, invalid_idx]
            Z_slice = Z[:, invalid_idx]
            Z_slice[Fin_invalid] = 0.0
            Z[:, invalid_idx] = Z_slice
        z_df = pd.DataFrame(Z, index=look.index, columns=look.columns)
        out[hl] = find_neighbors(z_df, target, k=k)
    return out

if __name__ == "__main__":
    # Iterate years from 1980: weekly dates, random target per week; save per-year CSVs
    import pandas as pd
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    s_path = root / "data" / "processed" / "returns" / "stock_returns.parquet"
    out_dir = root / "results" / "hedge_eval"
    try:
        rng = np.random.default_rng(7)
        s_df = pd.read_parquet(s_path)
        s_df.index = pd.to_datetime(s_df.index)
        first_year = max(1989, int(s_df.index.min().year))
        last_year = int(s_df.index.max().year)
        years = list(range(first_year, last_year + 1))
        halflifes = [5, 30, 90, 180, 365, 720]
        ks = [1, 3, 5, 10, 30, 60]
        horizons = [5, 21, 126, 252]
        tickers = list(s_df.columns)
        out_dir.mkdir(parents=True, exist_ok=True)
        with tqdm(total=len(years), desc="years", ncols=80) as ty:
            for year in years:
                rows = []
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="7D")
                with tqdm(total=len(dates), desc=f"dates {year}", leave=False, ncols=80) as td:
                    for d in dates:
                        # Pick a target with sufficient history for the shortest half-life
                        L_short = int(6 * min(halflifes))
                        lookback_df = s_df[s_df.index <= d]
                        short_win = lookback_df.tail(L_short)
                        if short_win.empty:
                            print(f"{d.date()}: no lookback window for shortest half-life; skip")
                            td.update(1)
                            continue
                        counts = short_win.notna().sum(axis=0)
                        elig = counts[counts >= L_short].index.tolist()
                        if not elig:
                            print(f"{d.date()}: 0 eligible tickers for shortest half-life; skip")
                            td.update(1)
                            continue
                        target = elig[int(rng.integers(0, len(elig)))]
                        print(f"{d.date()}: eligible={len(elig)} target={target}")
                        for hl in halflifes:
                            dist_dict = neighbors_by_halflife(lookback_df, target, [hl], k=60).get(hl, {})
                            if not dist_dict:
                                continue
                            neigh = list(dist_dict.keys())
                            # Use inverse-distance weights per k (minimal, fast)
                            w_by_k = make_weights(dist_dict, ks)
                            fwd_df = subset_returns(s_df, None, d, target, neigh, end_date=d + pd.Timedelta(days=365), neighbors_source="stock")
                            t_series, port_dict = build_series(fwd_df, target, [w_by_k.get(k, {}) for k in ks])
                            for i, (label, p) in enumerate(port_dict.items()):
                                k_val = ks[i]
                                neigh_list = list(w_by_k.get(k_val, {}).keys())
                                # Evaluate forward on regression-weighted portfolio
                                for m in evaluate_horizons(t_series, p, horizons):
                                    row = {
                                        "date": d.date(),
                                        "target": target,
                                        "half_life": hl,
                                        "k": k_val,
                                        "window": m.get("window_len"),
                                        "n_obs": m.get("n_obs"),
                                        "he": m.get("he"),
                                        "te_ann": m.get("te_ann"),
                                        "neighbors": ",".join(neigh_list),
                                    }
                                    # Include additional metrics if present
                                    extra_keys = [
                                        "te_ratio",
                                        "var_reduct_abs",
                                        "mdd",
                                        "mdd_reduct",
                                        "mdd_abs_reduct",
                                        "var_95",
                                        "var95_reduct",
                                        "var95_abs_reduct",
                                        "alpha_ann",
                                        "hit_rate",
                                        "cum_return_target",
                                        "cum_return_portfolio",
                                        "cum_return_diff",
                                    ]
                                    for kx in extra_keys:
                                        if kx in m:
                                            row[kx] = m[kx]
                                    rows.append(row)
                        td.update(1)
                out = pd.DataFrame(rows).sort_values(["date", "half_life", "k", "window"]).reset_index(drop=True)
                out_path = out_dir / f"{year}.csv"
                out.to_csv(out_path, index=False)
                ty.update(1)
    except Exception as e:
        print(f"Yearly run skipped: {e}")
