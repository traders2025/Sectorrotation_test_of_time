# -*- coding: utf-8 -*-
"""
Download daily prices for the nine Select Sector SPDR ETFs and SPY from Yahoo Finance.

Run in Spyder (F5) or from a terminal:  python get_sector_etf_data.py

What it does
  1. Downloads daily Close, Adj Close and Volume for XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY and SPY
     from 22 December 1998 to today.
  2. Saves one CSV per ticker (columns: Date, Close, Adj Close, Volume) in <OUTPUT_DIR>/daily/
  3. Writes <OUTPUT_DIR>/sector_etf_data_for_upload.zip, ready to upload.

Requirements: Python 3 with pandas and yfinance.  If yfinance is missing, run in the Spyder console:
    pip install yfinance
(or:  conda install -c conda-forge yfinance)
"""
import os
import sys
import time
import zipfile

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is not installed. In the Spyder console run:  pip install yfinance   then run this script again.")

# ----------------------------------------------------------------------------- settings
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "sector_etf_data")   # change if you like
START = "1998-12-22"
END = None                       # None = up to the latest available day
TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SPY"]


def tidy(df, ticker):
    """Return a DataFrame with columns Date, Close, Adj Close, Volume from a yfinance download."""
    if isinstance(df.columns, pd.MultiIndex):           # newer yfinance versions return (field, ticker) columns
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).strip())
    if "Adj Close" not in df.columns:
        raise ValueError("no 'Adj Close' column (download with auto_adjust=False)")
    out = df[["Close", "Adj Close", "Volume"]].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "Date"
    out = out.dropna(subset=["Adj Close"])
    return out.reset_index()


def download(ticker, tries=3):
    for k in range(tries):
        try:
            df = yf.download(ticker, start=START, end=END, auto_adjust=False, actions=False,
                             progress=False, threads=False)
            if df is not None and len(df) > 1000:
                return tidy(df, ticker)
            print(f"    {ticker}: only {0 if df is None else len(df)} rows, retrying")
        except Exception as e:
            print(f"    {ticker}: {e}")
        time.sleep(3 * (k + 1))
    return None


def main():
    daily = os.path.join(OUTPUT_DIR, "daily")
    os.makedirs(daily, exist_ok=True)
    print(f"Saving to {OUTPUT_DIR}  (yfinance {yf.__version__})")
    saved, failed = [], []
    for t in TICKERS:
        d = download(t)
        if d is None:
            print(f"- {t:4s} FAILED")
            failed.append(t)
            continue
        path = os.path.join(daily, f"{t}.csv")
        d.to_csv(path, index=False, date_format="%Y-%m-%d")
        saved.append(path)
        gaps = d["Date"].diff().dt.days.max()
        print(f"- {t:4s} {len(d):5d} rows  {d['Date'].iloc[0]:%Y-%m-%d} .. {d['Date'].iloc[-1]:%Y-%m-%d}"
              f"  (largest gap {gaps:.0f} days)")
        time.sleep(1)
    bundle = os.path.join(OUTPUT_DIR, "sector_etf_data_for_upload.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in saved:
            z.write(p, arcname=os.path.basename(p))
    print(f"\nDownloaded {len(saved)} of {len(TICKERS)} tickers.")
    print(f"Upload this:  {bundle}")
    if failed:
        print("Failed: " + ", ".join(failed) + "  - run the script again (Yahoo sometimes rate-limits).")


if __name__ == "__main__":
    main()
