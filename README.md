# Trading- Project

## Overview
This project provides a robust pipeline for analyzing financial return series and performing K-Nearest Neighbors (KNN) analysis on stocks and ETFs. It is designed for scalable, reproducible research in quantitative finance, with efficient data processing and clear modular structure.

## Features
- **Universe Creation:** Automatically builds a universe of tickers from NASDAQ and NYSE listings.
- **Data Fetching:** Downloads historical price data for all tickers using Yahoo Finance, with error handling and progress bars.
- **Return Processing:** Efficiently computes log returns for all assets, storing results in compressed columnar formats for fast access.
- **KNN Analysis:** Ready-to-use framework for finding nearest neighbors in return space, supporting random sampling and moving windows.
- **Modular Design:** All data processing scripts are organized in `code/data_processing` for clarity and maintainability.

## Directory Structure
```
code/
  data_processing/
    0_make_universe.py   # Build ticker universe
    1_fetch_raw.py       # Download price data
    2_process_returns.py # Build returns matrix
  knn.py                 # KNN analysis on returns
configs/
  universe.csv           # List of tickers

# Data folders (ignored by git)
data/
  raw/                   # Raw price data
  processed/returns/     # Returns matrices
```

## Quickstart
1. **Set up environment:**
   ```bash
   conda env create -f environment.yml
   conda activate trading-env
   ```
2. **Build universe:**
   ```bash
   python code/data_processing/0_make_universe.py
   ```
3. **Fetch price data:**
   ```bash
   python code/data_processing/1_fetch_raw.py
   ```
4. **Process returns:**
   ```bash
   python code/data_processing/2_process_returns.py
   ```
5. **Run KNN analysis:**
   ```bash
   python code/knn.py
   ```

## Best Practices
- All large data files are excluded from git via `.gitignore`.
- Scripts use memory-efficient data types and robust error handling.
- Modular code makes it easy to extend or adapt for new research questions.

## License
MIT

## Author
Jackson McBride
