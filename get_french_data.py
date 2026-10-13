# -*- coding: utf-8 -*-
"""
Download Kenneth French data-library files for the industry-rotation study.

Run in Spyder (F5) or from a terminal:  python get_french_data.py

What it does
  1. Downloads the industry portfolios (10, 12, 49 industries; monthly and daily),
     the Fama-French 3 and 5 factors, and the momentum factor (monthly and daily).
  2. Saves the original zip files in  <OUTPUT_DIR>/raw/
  3. Extracts each CSV and splits it into its separate tables (for example
     "Average Value Weighted Returns -- Monthly"), saved as clean CSV files in
     <OUTPUT_DIR>/clean/  with a Date column and returns in PERCENT, exactly as published.
     Missing values (-99.99 / -999) become empty cells.
  4. Writes  <OUTPUT_DIR>/french_data_for_upload.zip  containing the raw zips,
     ready to upload.

Requirements: Python 3 with pandas and requests (both included in Anaconda).
"""
import io
import os
import re
import sys
import time
import zipfile

import pandas as pd
import requests

# ----------------------------------------------------------------------------- settings
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "french_data")   # change if you like

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
LIBRARY_PAGE = BASE + "data_library.html"
FTP = BASE + "ftp/"

# Files to fetch: (short name, list of candidate file names on the server).
# Several spellings are listed because the site is not consistent about "daily"/"Daily".
FILES = [
    ("ind10_monthly", ["10_Industry_Portfolios_CSV.zip"]),
    ("ind10_daily",   ["10_Industry_Portfolios_daily_CSV.zip", "10_Industry_Portfolios_Daily_CSV.zip"]),
    ("ind12_monthly", ["12_Industry_Portfolios_CSV.zip"]),
    ("ind12_daily",   ["12_Industry_Portfolios_daily_CSV.zip", "12_Industry_Portfolios_Daily_CSV.zip"]),
    ("ind49_monthly", ["49_Industry_Portfolios_CSV.zip"]),
    ("ind49_daily",   ["49_Industry_Portfolios_daily_CSV.zip", "49_Industry_Portfolios_Daily_CSV.zip"]),
    ("ff3_monthly",   ["F-F_Research_Data_Factors_CSV.zip"]),
    ("ff3_daily",     ["F-F_Research_Data_Factors_daily_CSV.zip", "F-F_Research_Data_Factors_Daily_CSV.zip"]),
    ("ff5_monthly",   ["F-F_Research_Data_5_Factors_2x3_CSV.zip"]),
    ("ff5_daily",     ["F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Research_Data_5_Factors_2x3_Daily_CSV.zip"]),
    ("mom_monthly",   ["F-F_Momentum_Factor_CSV.zip"]),
    ("mom_daily",     ["F-F_Momentum_Factor_daily_CSV.zip", "F-F_Momentum_Factor_Daily_CSV.zip"]),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (research download script; pandas/requests)"}


# ----------------------------------------------------------------------------- download
def library_links():
    """Return the set of *_CSV.zip file names linked from the data-library page (if reachable)."""
    try:
        r = requests.get(LIBRARY_PAGE, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return {os.path.basename(h) for h in re.findall(r'href="([^"]+_CSV\.zip)"', r.text, flags=re.I)}
    except Exception as e:
        print(f"  (could not read the library page: {e}; trying the standard file names)")
        return set()


def download(candidates, links, raw_dir):
    """Try each candidate name (preferring names actually linked on the page). Return local path or None."""
    linked = {n.lower(): n for n in links}
    ordered = [linked[c.lower()] for c in candidates if c.lower() in linked] + candidates
    seen = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        url = FTP + name
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200 and r.content[:2] == b"PK":      # a real zip file
                path = os.path.join(raw_dir, name)
                with open(path, "wb") as f:
                    f.write(r.content)
                return path
        except Exception as e:
            print(f"    {name}: {e}")
        time.sleep(0.5)
    return None


# ----------------------------------------------------------------------------- parsing
def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()[:60]


def _to_date(s):
    s = str(s).strip()
    if len(s) == 8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if len(s) == 6:
        return pd.to_datetime(s, format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)
    if len(s) == 4:
        return pd.to_datetime(s, format="%Y", errors="coerce") + pd.offsets.YearEnd(0)
    return pd.NaT


def parse_french_csv(text):
    """Split a Ken French CSV into its tables. Returns a list of (title, DataFrame)."""
    lines = text.replace("\r", "").split("\n")
    tables, title, header, rows = [], "", None, []

    def flush():
        nonlocal header, rows
        if header is not None and rows:
            df = pd.DataFrame(rows, columns=["Date"] + header[: len(rows[0]) - 1])
            df["Date"] = df["Date"].map(_to_date)
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce").replace([-99.99, -999.0, -999], pd.NA)
            tables.append((title or "data", df.dropna(subset=["Date"])))
        header, rows = None, []

    for raw in lines:
        line = raw.strip()
        cells = [c.strip() for c in raw.split(",")]
        if line == "":
            flush()
            continue
        if raw.lstrip().startswith(",") or (cells[0] == "" and len(cells) > 1):   # header row
            flush()
            header = [c for c in cells[1:] if c != ""] or cells[1:]
            continue
        if re.fullmatch(r"\d{4,8}", cells[0]) and header is not None:          # data row
            rows.append(cells[: len(header) + 1])
            continue
        # anything else is a title / description line
        flush()
        if len(line) < 120 and "missing" not in line.lower() and "copyright" not in line.lower():
            title = line.strip(" :")
        elif not title:
            title = "main"

    flush()
    return tables


def clean(zip_path, short, clean_dir):
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue
            text = z.read(member).decode("latin-1")
            for i, (title, df) in enumerate(parse_french_csv(text)):
                fn = f"{short}__{i:02d}_{_slug(title)}.csv"
                df.to_csv(os.path.join(clean_dir, fn), index=False)
                out.append((fn, len(df), df["Date"].min(), df["Date"].max()))
    return out


# ----------------------------------------------------------------------------- main
def main():
    raw_dir = os.path.join(OUTPUT_DIR, "raw")
    clean_dir = os.path.join(OUTPUT_DIR, "clean")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    print(f"Saving to {OUTPUT_DIR}")

    links = library_links()
    got, failed = [], []
    for short, candidates in FILES:
        print(f"- {short:14s} ", end="", flush=True)
        path = download(candidates, links, raw_dir)
        if path is None:
            print("FAILED")
            failed.append((short, candidates[0]))
            continue
        print(os.path.basename(path))
        got.append(path)
        for fn, n, d0, d1 in clean(path, short, clean_dir):
            print(f"      {fn:70s} {n:6d} rows  {d0:%Y-%m-%d} .. {d1:%Y-%m-%d}")

    bundle = os.path.join(OUTPUT_DIR, "french_data_for_upload.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in got:
            z.write(p, arcname=os.path.basename(p))
    print(f"\nDownloaded {len(got)} of {len(FILES)} files.")
    print(f"Clean tables:  {clean_dir}")
    print(f"Upload this:   {bundle}  ({os.path.getsize(bundle) / 1e6:.1f} MB)")
    if failed:
        print("\nCould not download (get these by hand from the library page, CSV links):")
        for short, name in failed:
            print(f"   {short}: {name}")
        print(f"   {LIBRARY_PAGE}")


if __name__ == "__main__":
    main()
