# Trading- Project

## Overview
End‑to‑end pipeline to analyze whether KNN can generate effective hedges from historical return series. The workflow builds a universe, fetches data, computes returns, selects neighbors via correlation distance, and evaluates hedges out‑of‑sample across years, lookforward windows, ks, and half‑lives. Summary tables and plots are produced for use in a paper.

Key choices (current defaults)
- Neighbor selection: correlation distance `1 − |corr|` on half‑life‑weighted, backward lookback windows.
- Portfolio weights: inverse‑distance (sum to 1, closer neighbors get higher weight).

## Layout
```
code/
  0_data_processing/
    0_make_universe.py      # Build configs/universe.csv
    1_fetch_raw.py          # Download Yahoo price data to data/raw/yahoo
    2_process_returns.py    # Build data/processed/returns/{stock,etf}_returns.parquet
  1_analysis/
    0_exploratory.py        # Data coverage + summary plots/CSVs
    1_generate_evaluate_hedges.py  # Yearly OOS evaluation (writes results/hedge_eval/*.csv)
    2_analyze_stocks.py     # Aggregates to results/hedge_results/*.csv (ticker + industry/sector metrics)
    3_analyze_hedges.py     # Year/HL/k/window summary (writes year_hlk_summary.csv)
    4_results.py            # Plots and comparison tables (reads hedge_results/*)

configs/
  universe.csv              # Ticker universe

data/
  raw/yahoo/{stock,etf}/    # Fetched daily data (CSV)
  processed/returns/        # Returns matrices (parquet)

results/
  hedge_eval/*.csv          # Per‑run evaluation rows (by year)
  hedge_results/
    ticker_summary.csv      # Per‑ticker aggregates
    industry_metrics.csv    # Per‑industry (and sector) per‑run stats
    sector_metrics.csv      # Per‑sector per‑run stats
    year_hlk_summary.csv    # Year × half_life × k × window summary
    plots/*                 # Figures
```

## Quickstart
1) Environment
```
conda env create -f environment.yml
conda activate trading-env
```
2) Build inputs
```
python code/0_data_processing/0_make_universe.py
python code/0_data_processing/1_fetch_raw.py
python code/0_data_processing/2_process_returns.py
```
3) Evaluate hedges and summarize
```
python code/1_analysis/1_generate_evaluate_hedges.py
python code/1_analysis/2_analyze_stocks.py
python code/1_analysis/3_analyze_hedges.py
python code/1_analysis/4_results.py
```

## Outputs (high level)
- Plots (results/hedge_results/plots):
  - Ticker counts bar with readable legend
  - Boxplots of hedge effectiveness by Sector and top‑20 Industries (median line in dark blue, zero reference)
  - Time‑series HE by year: subplots per window, lines by k and by half‑life (symlog y‑scale)
- Tables (results/hedge_results):
  - k × half‑life, k × window, and half‑life × window comparison tables for HE and other metrics (TE, MDD/VaR reductions, alpha, hit rate) with companion p‑value tables
  - industry_metrics.csv, sector_metrics.csv: per‑run mean, sd, p‑value, and counts per group

## Notes
- Neighboring now uses correlation distance; weights are inverse‑distance for speed and stability.
- Scripts are defensive to missing data and align on overlapping dates.

## License
MIT

## Author
Jackson McBride
