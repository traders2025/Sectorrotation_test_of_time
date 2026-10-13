# Data used in "Do Sector Rotation Rules Survive a Century?"

All data used in the paper are in this folder (and in `../etf_era/data`). Nothing is downloaded at run time.

| Folder / file | Content | Source | Used for |
|---|---|---|---|
| `french/*.zip` (12 files) | Original Kenneth French data-library files, unchanged: 10, 12 and 49 industry portfolios (monthly and daily, value- and equal-weighted), Fama–French 3 factors, 5 factors (2x3), momentum factor (monthly and daily). Files created from the 202608 CRSP database. | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html, downloaded 9 Oct 2026 with `../get_french_data.py` | Century analysis 1928–2025; risk-free rate and market; factor regressions; head-to-head and out-of-sample (FF industries) |
| `french_clean/*.csv` (37 files) | The same files split into their separate tables (Date column; returns in percent, as published; -99.99/-999 as empty). Convenience copies, regenerable with `get_french_data.py`; the two 49-industry daily tables are omitted for size (the original is in `french/`). | derived from `french/` | Inspection only (the code reads the zips) |
| `etf_published_repo/*_weekly_return_detailed.csv` (10 files) | Daily prices of XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY and SPY, 22 Dec 1998 – 31 Dec 2024 (Yahoo Finance adjusted closes), exactly as published with the Journal of Investing study. | https://github.com/traders2025/SPY_ETFs_Rotation_Trading_Strategy | Provenance of `../etf_era/data/daily` |
| `../etf_era/data/daily/*.csv` (10 files) | Date, Close, Adj Close, Volume extracted from `etf_published_repo/`. | derived | ETF era 2000–2024 (Section 4) and head-to-head (Section 6) |
| `../etf_era/data/tb3ms.csv` | Monthly 3-month Treasury bill rate (TB3MS, percent), Jan 1998 – Dec 2025. | FRED, https://fred.stlouisfed.org/series/TB3MS | Risk-free rate for the ETF-era results |
| `etf_daily/*.csv` (10 files) | Daily Close, Adj Close, Volume for the same nine ETFs and SPY, 22 Dec 1998 – 9 Oct 2026. Over 2000–2024 their daily returns match `etf_published_repo` (correlation >= 0.9998, same trading days). | Yahoo Finance via `yfinance`, downloaded 10 Oct 2026 with `../get_sector_etf_data.py` | Out-of-sample period 2 Jan 2025 – 31 Aug 2026 (Section 6.1) |

Yahoo Finance data are subject to Yahoo's terms of use; check before posting the ETF files publicly.
