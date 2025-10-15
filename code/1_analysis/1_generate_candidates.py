from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# Start dates are fixed to ensure a sufficiently large search universe
# for KNN candidate generation. We require the number of available
# assets N to be comfortably larger than k (rule of thumb: N >= 10*k)
# and also above a minimum floor. With k=30 and floors of 200 (stocks)
# and 100 (ETFs), the earliest dates meeting N >= 300 were:
# - Stocks: 1980-03-18 (N=423)
# - ETFs:   2006-11-20 (N=300)
# Hard-coding these dates prevents starting in periods with too few
# neighbors, which would make large-k searches unstable/uninformative.

stock_start_date = date(1980, 3, 18)
etf_start_date = date(2006, 11, 20)


def _exp_weights(T: int, half_life: int) -> np.ndarray:
    lam = np.log(2.0) / max(1, half_life)
    ages = np.arange(T - 1, -1, -1, dtype=np.float64)
    w = np.exp(-lam * ages)
    return w / w.max()


def _weighted_standardize(X: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # X: T x N (no NaNs), W: T
    w = W[:, None]
    denom = np.sum(w, axis=0)
    mu = (w * X).sum(axis=0) / denom
    C = X - mu
    var = (w * (C * C)).sum(axis=0) / denom
    inv_std = np.where(var > 0, 1.0 / np.sqrt(var), 0.0)
    Z = C * inv_std
    B = (W[:, None] ** 0.5) * Z
    norms = np.sqrt((B * B).sum(axis=0))
    return B, norms


class Window:
    """Minimal sliding window over returns with exponential weights.

    Defined by asset_class, start_date, and length. Columns with any NaNs
    inside the window are dropped. Methods: increment, generate_targets,
    generate_candidates.
    """

    def __init__(self, asset_class: str, start_date: date, half_life: int = 15, length: int | None = None, standardize: bool = False):
        self.asset_class = asset_class
        self.start_date = pd.Timestamp(start_date)
        self.half_life = int(half_life)
        # Window length is a function of half-life (6 × H). "length" is ignored for simplicity/back-compat.
        self.length = int(6 * max(1, self.half_life))
        self.standardize = bool(standardize)

        root = Path(__file__).resolve().parents[2]
        path = root / "data" / "processed" / "returns" / (
            "stock_returns.parquet" if asset_class == "stock" else "etf_returns.parquet"
        )
        # load returns once
        self.df = pd.read_parquet(path)

        # Locate start position; window is [start_pos : start_pos+length)
        match = np.where(self.df.index == self.start_date)[0]
        if match.size == 0:
            raise ValueError(f"Start date {self.start_date.date()} not found in index for {asset_class}")
        self.start_pos = int(match[0])
        if self.start_pos + self.length > len(self.df.index):
            raise ValueError("Window exceeds available data; shorten length")

        self.w = _exp_weights(self.length, self.half_life)
        self.rng = np.random.default_rng(42)

        self._build_window()

    @property
    def end_date(self) -> pd.Timestamp:
        return self.df.index[self.start_pos + self.length - 1]

    def _build_window(self) -> None:
        i0 = self.start_pos
        i1 = self.start_pos + self.length
        win = self.df.iloc[i0:i1]
        # Keep only columns with no NaNs in window
        win = win.dropna(axis=1, how="any")
        self.win = win
        self.cols = win.columns.to_numpy()
        # Precompute weighted matrix for fast KNN
        X = win.to_numpy(dtype=np.float64)
        if X.shape[0] != self.length:
            # Pad top if needed (rare if start_date chosen correctly)
            pad = self.length - X.shape[0]
            if pad > 0:
                X = np.vstack([np.zeros((pad, X.shape[1])), X])
        if self.standardize:
            self.B, self.norms = _weighted_standardize(X, self.w)
            self._metric = "corr"
        else:
            # Raw, weighted returns (no centering or scaling)
            B = (self.w[:, None] ** 0.5) * X
            self.B = B
            # Store sum of squares for Euclidean distance
            self.norms = np.sum(B * B, axis=0)
            self._metric = "l2"
        # window slice prepared

    def increment(self, step_size: int = 7) -> bool:
        new_start = self.start_pos + int(step_size)
        if new_start + self.length > len(self.df.index):
            return False
        self.start_pos = new_start
        self._build_window()
        return True

    def generate_targets(self, n: int = 3) -> list[str]:
        if len(self.cols) == 0:
            return []
        pick = min(n, len(self.cols))
        idx = self.rng.choice(len(self.cols), size=pick, replace=False)
        return [str(self.cols[i]) for i in idx]

    def generate_candidates(self, target: str, k: int = 30):
        if target not in self.win.columns:
            return []
        ti = int(np.where(self.cols == target)[0][0])
        v = self.B[:, ti]
        if self._metric == "corr":
            vnorm = self.norms[ti]
            if not np.isfinite(vnorm) or vnorm == 0:
                return []
            cov = self.B.T @ v
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = cov / (self.norms * vnorm)
            dist = 1.0 - corr
        else:
            # Weighted Euclidean distance on raw returns
            dot = self.B.T @ v
            vss = self.norms[ti]
            dist2 = self.norms + vss - 2.0 * dot
            dist2[ti] = np.inf
            dist2 = np.where(dist2 < 0, 0, dist2)
            dist = np.sqrt(dist2)
        dist[ti] = np.inf
        k_eff = min(int(k), len(dist) - 1)
        nn = np.argpartition(dist, k_eff)[:k_eff]
        nn = nn[np.argsort(dist[nn])]
        out = [(str(self.cols[j]), float(dist[j])) for j in nn if np.isfinite(dist[j])]
        return out


# Removed verbose demo prints to keep console output minimal


def run_single_iteration(asset_class: str, half_life: int, step_size: int = 7, n_targets: int = 10, k: int = 30) -> None:
    """Iterate all end-dates and save one JSON per parameterization under results/candidates/.

    Output: results/candidates/<asset_class>/param=H{half_life}.json
    Maps { date: { target: [[neighbor, distance], ...] } }.
    """
    start_fix = stock_start_date if asset_class == "stock" else etf_start_date
    win = Window(asset_class, start_date=start_fix, half_life=half_life)
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "results" / "candidates" / asset_class
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"param=H{half_life}.json"

    L = len(win.df.index)
    max_start = L - win.length
    total = 1 if max_start < win.start_pos else ((max_start - win.start_pos) // max(1, step_size) + 1)

    all_out = {}
    with tqdm(total=total, desc=f"iterate[{asset_class}]", ncols=80) as pbar:
        for _ in range(total):
            date_key = str(win.end_date.date())
            targets = win.generate_targets(n=min(n_targets, len(win.cols)))
            record = {t: win.generate_candidates(t, k=k) for t in targets}
            all_out[date_key] = record
            pbar.update(1)
            if not win.increment(step_size=step_size):
                break
    # write one big JSON per parameterization
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_out, f)

def run_stock_halflife_sweep():
    for H in [5, 10, 15, 30, 60, 90, 120, 240, 360]:
        run_single_iteration("stock", half_life=H, step_size=7, n_targets=10, k=30)

# Run the sweep
run_stock_halflife_sweep()
