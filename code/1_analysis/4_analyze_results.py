from pathlib import Path
import pandas as pd
import math

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

root = Path(__file__).resolve().parents[2]
in_path = root / "results" / "hedge_eval" / "stock_metrics.parquet"
counts_path = root / "results" / "hedge_eval" / "counts.csv"
ticker_counts_path = root / "results" / "hedge_eval" / "ticker_counts.csv"
obs_stats_path = root / "results" / "hedge_eval" / "n_obs_by_group.csv"
plot_path = root / "results" / "hedge_eval" / "ticker_counts.png"

try:
    df = pd.read_parquet(in_path)
except FileNotFoundError:
    print(f"Not found: {in_path}")
    raise SystemExit(1)
except Exception as e:
    print(f"Error reading {in_path}: {e}")
    raise SystemExit(1)

# Aggregate counts by half_life, k, and lookforward horizon
counts = (
    df.groupby(["half_life", "k", "horizon"], dropna=False)
      .size()
      .reset_index(name="count")
      .sort_values(["half_life", "k", "horizon"])
)

# Per-ticker counts (by target)
ticker_counts = (
    df.groupby(["target"], dropna=False)
      .size()
      .reset_index(name="count")
      .sort_values(["count", "target"], ascending=[True, True])
)

# Number of observations within each group (check for consistency)
obs_stats = (
    df.groupby(["half_life", "k", "horizon"], dropna=False)
      .agg(
          num_rows=("n_obs", "size"),
          min_n_obs=("n_obs", "min"),
          max_n_obs=("n_obs", "max"),
          mean_n_obs=("n_obs", "mean"),
          median_n_obs=("n_obs", "median"),
          nunique_n_obs=("n_obs", "nunique"),
      )
      .reset_index()
      .sort_values(["half_life", "k", "horizon"])
)

# Write outputs
counts.to_csv(counts_path, index=False)
ticker_counts.to_csv(ticker_counts_path, index=False)
obs_stats.to_csv(obs_stats_path, index=False)

print(f"Wrote counts: {counts_path}")
print(f"Wrote per-ticker counts: {ticker_counts_path}")
print(f"Wrote n_obs stats: {obs_stats_path}")

# Print simple n_obs consistency check summary
same_obs_groups = int((obs_stats["nunique_n_obs"] == 1).sum())
total_groups = int(obs_stats.shape[0])
print(f"Groups with constant n_obs across rows: {same_obs_groups} / {total_groups}")

# Plot distribution of ticker counts, ordered least to most
if plt is not None:
    try:
        out_dir = plot_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # Large figure to accommodate many tickers; rotate labels if needed
        fig_w = max(12, min(48, len(ticker_counts) * 0.05))
        fig_h = 6
        plt.figure(figsize=(fig_w, fig_h))
        plt.bar(range(len(ticker_counts)), ticker_counts["count"].to_numpy())
        plt.title("Distribution of counts per ticker (target)")
        plt.xlabel("Tickers (sorted ascending)")
        plt.ylabel("Count")
        # Optionally add sparse x tick labels for readability
        if len(ticker_counts) <= 50:
            plt.xticks(range(len(ticker_counts)), ticker_counts["target"].tolist(), rotation=90, fontsize=8)
        else:
            step = max(1, len(ticker_counts) // 50)
            idxs = list(range(0, len(ticker_counts), step))
            labels = ticker_counts["target"].iloc[idxs].tolist()
            plt.xticks(idxs, labels, rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved ticker count distribution plot: {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")
else:
    print("matplotlib not available; skipped plot generation.")
