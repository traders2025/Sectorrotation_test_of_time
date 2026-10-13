# A Century of Winners, Median, and Losers — industry rotation 1928–2025 (code)

All data are included: see `data/README.md`. `etf_era/` holds the ETF-era (2000–2024) engine, data and scripts (`engine_sec.py`, `analysis.py`, `offsets.py`).
Main universe: 10 Fama–French industries, value-weighted daily returns; Winners = top 3, Median = ranks 4–7,
Losers = bottom 3, Ex-Winners = ranks 4–10, EW = all 10. Benchmark: CRSP value-weighted market (Mkt-RF + RF).

Windows (calendar months ending at the rebalancing date): Annual 1/3/6/12; Semiannual 1/3/6; Quarterly 1/3;
Monthly 1; Weekly 1 week; plus longer windows as an extension.

```
pip install -r requirements.txt
python run_all.py
```
| Script | Purpose |
|---|---|
| `engine_ind.py` | data loading (parses the French zips), Universe class, backtests |
| `analysis_ind.py` | main results, chunked stationary bootstrap, Romano–Wolf, reality check, pooled and sub-period tests, random portfolios, lag, leave-one-out, alternative universes (9, 12, 49, EW-weighted), event time, December analysis, FF3+MOM / FF5+MOM regressions |
| `offsets_ind.py` | calendar robustness (12 annual, 6 semiannual, 3 quarterly calendars) |
| `bridge.py` | head-to-head 2000–2024: SPDR ETFs vs FF9 and FF10 industries (uses `etf_era/`) |
| `oos.py` | after the published sample (2 Jan 2025 – 31 Aug 2026): same rules for SPDR ETFs (`data/etf_daily`, Yahoo Finance, downloaded 10 Oct 2026 with `get_sector_etf_data.py`) and FF9/FF10 |
| `gen_tables_ind.py`, `figs_ind.py` | LaTeX tables and figures for `../paper` (the ETF-era table reads `etf_era/results`) |

Annualization uses the sample's average trading days per year (262, Saturday trading before 1952). Seeds fixed.
