from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import numpy as np

# Dark blue
COLOR = "#0b1f3b"

root = Path(__file__).resolve().parents[2]
results_dir = root / "results" / "hedge_results"
plots_dir = results_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

summary_path = results_dir / "ticker_summary.csv"
if not summary_path.exists():
    print(f"No summary CSV found: {summary_path}")
else:
    df = pd.read_csv(summary_path)
    if "target" not in df.columns:
        print(f"Missing 'target' in {summary_path}")
    else:
        if "count" in df.columns:
            counts = (
                df[["target", "count"]]
                .dropna()
                .groupby("target", as_index=True)["count"].sum()
                .sort_values()
            )
        else:
            counts = df["target"].value_counts().sort_values()
        # Minimal bar plot: one bar per ticker (count)
        # Trim extreme outliers (top 1%) to improve readability
        counts_plot = counts.copy()
        try:
            q_hi = counts_plot.quantile(0.99)
            counts_plot = counts_plot[counts_plot <= q_hi]
            if counts_plot.empty:
                counts_plot = counts
        except Exception:
            counts_plot = counts

        out_path = plots_dir / "ticker_counts_bar.png"
        plt.figure(figsize=(18, 10))
        ax = counts_plot.plot.bar(color=COLOR, width=1.0)
        ax.margins(x=0)
        # Remove individual ticker labels, add axis labels
        ax.set_xticks([])
        plt.xlabel("Tickers")
        plt.ylabel("Count")
        # Legend: Top 10 tickers with counts (upper-left, left-aligned)
        top10 = counts_plot.sort_values(ascending=False).head(10)
        labels = [f"{t}: {int(c)}" for t, c in top10.items()]
        handles = [Line2D([0], [0], color="none") for _ in labels]
        plt.legend(
            handles,
            labels,
            loc="upper left",
            frameon=True,
            fontsize=9,
            title="Top 10 Selected Tickers",
            handlelength=0,
            handletextpad=0,
        )
        plt.title("Counts per Ticker")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor="white")
        plt.close()
        print(f"Saved: {out_path}")

        # -----------------------------
        # Boxplots: hedge effectiveness
        # -----------------------------
        def boxplot_by_category(
            df_cat: pd.DataFrame,
            cat_col: str,
            val_col: str,
            title: str,
            out_name: str,
            top_n: int | None = None,
            width_scale: float = 1.0,
        ):
            d = df_cat[[cat_col, val_col]].dropna()
            if d.empty:
                print(f"No data for {cat_col} boxplot")
                return
            if top_n is not None:
                keep = (
                    d[cat_col]
                    .value_counts()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index
                )
                d = d[d[cat_col].isin(keep)]
            if d.empty:
                print(f"No data after filtering for {cat_col} boxplot")
                return
            med = d.groupby(cat_col)[val_col].median().sort_values(ascending=False)
            cats = med.index.tolist()
            data = [d[d[cat_col] == c][val_col].values for c in cats]
            base_w = min(30, max(8, 0.6 * len(cats) + 2))
            w = max(6, base_w * float(width_scale))
            plt.figure(figsize=(w, 5))
            plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
            plt.boxplot(
                data,
                labels=cats,
                showfliers=False,
                medianprops={"color": COLOR, "linewidth": 2},
            )
            plt.ylabel("Mean Hedge Effectiveness")
            plt.title(title)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            outp = plots_dir / out_name
            plt.savefig(outp, dpi=150)
            plt.close()
            print(f"Saved: {outp}")

        he_col = "he_mean_overall" if "he_mean_overall" in df.columns else (
            "he_mean_21" if "he_mean_21" in df.columns else None
        )
        if he_col:
            # Sector boxplot
            if "Sector" in df.columns:
                boxplot_by_category(
                    df, "Sector", he_col,
                    title="Hedge Effectiveness by Sector",
                    out_name="he_by_sector_box.png",
                )
            # Industry boxplot (top 20 most represented industries)
            if "industry" in df.columns:
                boxplot_by_category(
                    df, "industry", he_col,
                    title="Hedge Effectiveness by Industry (Top 20)",
                    out_name="he_by_industry_box.png",
                    top_n=20,
                    width_scale=0.5,
                )

            # CSV creation moved to 2_analyze_stocks.py (per-run statistics)

# ---------------------------------
# Time plots by k and by half-life
# ---------------------------------
summary_hlk = results_dir / "year_hlk_summary.csv"
if summary_hlk.exists():
    yh = pd.read_csv(summary_hlk)
    for c in ["year", "half_life", "k", "window", "he_mean"]:
        if c in yh.columns:
            yh[c] = pd.to_numeric(yh[c], errors="coerce")
    yh = yh.dropna(subset=["year", "window", "he_mean"]).copy()
    if not yh.empty:
        windows = sorted(yh["window"].dropna().unique().tolist())

        def plot_by_key(key_col: str, title_prefix: str, out_name: str):
            if key_col not in yh.columns:
                return
            keys = sorted([x for x in yh[key_col].dropna().unique().tolist()])
            if not keys:
                return
            n = len(windows)
            ncols = 2 if n > 1 else 1
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
            # Neutral blue palette
            try:
                import numpy as _np
                cols_arr = plt.cm.Blues(_np.linspace(0.35, 0.85, len(keys)))
                color_map = {v: (COLOR if i == 0 else cols_arr[i]) for i, v in enumerate(keys)}
            except Exception:
                color_map = {v: COLOR for v in keys}
            for i, w in enumerate(windows):
                r, c = divmod(i, ncols)
                ax = axes[r][c]
                sub = yh[yh["window"] == w]
                yrs = sorted(sub["year"].dropna().unique().tolist())
                for v in keys:
                    s = sub[sub[key_col] == v].sort_values("year")
                    if s.empty:
                        continue
                    ax.plot(s["year"], s["he_mean"], label=str(v), color=color_map[v], linewidth=1.5)
                ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
                # Symmetric log scale to compress spikes while keeping negatives
                try:
                    ax.set_yscale("symlog", linthresh=0.01)
                except Exception:
                    pass
                ax.set_title(f"window = {int(w)}")
                if r == nrows - 1:
                    ax.set_xlabel("Year")
                if c == 0:
                    ax.set_ylabel("Hedge Effectiveness (mean)")
                ax.grid(alpha=0.15)
            # Hide any unused axes
            for j in range(len(windows), nrows * ncols):
                r, c = divmod(j, ncols)
                fig.delaxes(axes[r][c])
            # Single shared legend
            handles = [Line2D([0], [0], color=color_map[v], lw=2) for v in keys]
            labels = [f"{key_col}={v}" for v in keys]
            fig.suptitle(f"{title_prefix} by Year", y=0.99)
            # Place legend below title, avoid overlap
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(len(keys), 6),
                frameon=False,
                bbox_to_anchor=(0.5, 0.955),
            )
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            outp = plots_dir / out_name
            fig.savefig(outp, dpi=150)
            plt.close(fig)
            print(f"Saved: {outp}")

        # Plots
        plot_by_key("k", "Hedge Effectiveness (lines: k)", "he_over_time_by_k.png")
        plot_by_key("half_life", "Hedge Effectiveness (lines: half-life)", "he_over_time_by_halflife.png")
        # Table: k vs half-life (mean HE across years and windows)
        try:
            pivot = (
                yh.dropna(subset=["k", "half_life", "he_mean"])
                  .groupby(["k", "half_life"], as_index=True)["he_mean"].mean()
                  .unstack("half_life")
                  .sort_index()
                  .sort_index(axis=1)
            )
            out_table = results_dir / "he_k_by_halflife.csv"
            pivot.to_csv(out_table, float_format="%.6f")
            print(f"Saved: {out_table}")
            # P-values for k x half-life using across-row dispersion of yearly/window means
            def _p_from_series(s: pd.Series) -> float:
                x = pd.to_numeric(s, errors="coerce").dropna()
                n = int(x.shape[0])
                if n < 2:
                    return float("nan")
                m = float(x.mean())
                sd = float(x.std(ddof=1))
                if not (pd.notna(sd) and sd > 0):
                    return float("nan")
                se = sd / np.sqrt(n)
                t = m / se
                z = abs(float(t))
                return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))
            pvt_p = (
                yh.dropna(subset=["k", "half_life", "he_mean"]) 
                  .groupby(["k", "half_life"], as_index=True)["he_mean"]
                  .apply(_p_from_series)
                  .unstack("half_life")
                  .sort_index().sort_index(axis=1)
            )
            out_table_p = results_dir / "he_k_by_halflife_p.csv"
            pvt_p.to_csv(out_table_p, float_format="%.6g")
            print(f"Saved: {out_table_p}")
        except Exception as e:
            print(f"k x half-life table skipped: {e}")

        # Table: k vs lookforward window
        try:
            pivot_kw = (
                yh.dropna(subset=["k", "window", "he_mean"])
                  .groupby(["k", "window"], as_index=True)["he_mean"].mean()
                  .unstack("window")
                  .sort_index()
                  .sort_index(axis=1)
            )
            out_table_kw = results_dir / "he_k_by_window.csv"
            pivot_kw.to_csv(out_table_kw, float_format="%.6f")
            print(f"Saved: {out_table_kw}")
            # p-values
            pvt_p = (
                yh.dropna(subset=["k", "window", "he_mean"]) 
                  .groupby(["k", "window"], as_index=True)["he_mean"]
                  .apply(_p_from_series)
                  .unstack("window")
                  .sort_index().sort_index(axis=1)
            )
            out_table_kw_p = results_dir / "he_k_by_window_p.csv"
            pvt_p.to_csv(out_table_kw_p, float_format="%.6g")
            print(f"Saved: {out_table_kw_p}")
        except Exception as e:
            print(f"k x window table skipped: {e}")

        # Table: half-life vs lookforward window
        try:
            pivot_hw = (
                yh.dropna(subset=["half_life", "window", "he_mean"]) 
                  .groupby(["half_life", "window"], as_index=True)["he_mean"].mean()
                  .unstack("window")
                  .sort_index()
                  .sort_index(axis=1)
            )
            out_table_hw = results_dir / "he_halflife_by_window.csv"
            pivot_hw.to_csv(out_table_hw, float_format="%.6f")
            print(f"Saved: {out_table_hw}")
            # p-values
            pvt_p = (
                yh.dropna(subset=["half_life", "window", "he_mean"]) 
                  .groupby(["half_life", "window"], as_index=True)["he_mean"]
                  .apply(_p_from_series)
                  .unstack("window")
                  .sort_index().sort_index(axis=1)
            )
            out_table_hw_p = results_dir / "he_halflife_by_window_p.csv"
            pvt_p.to_csv(out_table_hw_p, float_format="%.6g")
            print(f"Saved: {out_table_hw_p}")
        except Exception as e:
            print(f"half-life x window table skipped: {e}")

        # Additional metrics: produce same tables for other evaluation metrics
        metrics_map = {
            "te_ann_mean": "te",
            "mdd_reduct_mean": "mdd_reduct",
            "var95_reduct_mean": "var95_reduct",
            "alpha_ann_mean": "alpha_ann",
            "hit_rate_mean": "hit_rate",
        }
        for col, prefix in metrics_map.items():
            if col not in yh.columns:
                continue
            try:
                pv = (
                    yh.dropna(subset=["k", "half_life", col])
                      .groupby(["k", "half_life"], as_index=True)[col].mean()
                      .unstack("half_life").sort_index().sort_index(axis=1)
                )
                pth = results_dir / f"{prefix}_k_by_halflife.csv"
                pv.to_csv(pth, float_format="%.6f")
                print(f"Saved: {pth}")
            except Exception as e:
                print(f"{prefix} k x half-life table skipped: {e}")
            # p-values for k x half-life
            try:
                pvt_p = (
                    yh.dropna(subset=["k", "half_life", col])
                      .groupby(["k", "half_life"], as_index=True)[col]
                      .apply(_p_from_series)
                      .unstack("half_life").sort_index().sort_index(axis=1)
                )
                pth = results_dir / f"{prefix}_k_by_halflife_p.csv"
                pvt_p.to_csv(pth, float_format="%.6g")
                print(f"Saved: {pth}")
            except Exception as e:
                print(f"{prefix} k x half-life p table skipped: {e}")
            try:
                pv = (
                    yh.dropna(subset=["k", "window", col])
                      .groupby(["k", "window"], as_index=True)[col].mean()
                      .unstack("window").sort_index().sort_index(axis=1)
                )
                pth = results_dir / f"{prefix}_k_by_window.csv"
                pv.to_csv(pth, float_format="%.6f")
                print(f"Saved: {pth}")
            except Exception as e:
                print(f"{prefix} k x window table skipped: {e}")
            # p-values for k x window
            try:
                pvt_p = (
                    yh.dropna(subset=["k", "window", col])
                      .groupby(["k", "window"], as_index=True)[col]
                      .apply(_p_from_series)
                      .unstack("window").sort_index().sort_index(axis=1)
                )
                pth = results_dir / f"{prefix}_k_by_window_p.csv"
                pvt_p.to_csv(pth, float_format="%.6g")
                print(f"Saved: {pth}")
            except Exception as e:
                print(f"{prefix} k x window p table skipped: {e}")
            try:
                pv = (
                    yh.dropna(subset=["half_life", "window", col])
                      .groupby(["half_life", "window"], as_index=True)[col].mean()
                      .unstack("window").sort_index().sort_index(axis=1)
                )
                pth = results_dir / f"{prefix}_halflife_by_window.csv"
                pv.to_csv(pth, float_format="%.6f")
                print(f"Saved: {pth}")
            except Exception as e:
                print(f"{prefix} half-life x window table skipped: {e}")
            # p-values for half-life x window
            try:
                pvt_p = (
                    yh.dropna(subset=["half_life", "window", col])
                      .groupby(["half_life", "window"], as_index=True)[col]
                      .apply(_p_from_series)
                      .unstack("window").sort_index().sort_index(axis=1)
                )
                pth = results_dir / f"{prefix}_halflife_by_window_p.csv"
                pvt_p.to_csv(pth, float_format="%.6g")
                print(f"Saved: {pth}")
            except Exception as e:
                print(f"{prefix} half-life x window p table skipped: {e}")
else:
    print(f"No yearly summary found: {summary_hlk}")
