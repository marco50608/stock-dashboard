"""
Bootstrap / refresh BOTH CBOE wide-format caches in one pass.

Usage:
    cd %USERPROFILE%\\Desktop\\stock
    .venv\\Scripts\\python scripts\\bootstrap_cboe.py

What it does:
    1. Read cache/cboe_ratios_wide.csv + cache/cboe_volume_oi_wide.csv,
       pick the earlier "last date" as the resume point
    2. For each missing trading day, hit CBOE daily JSON endpoint
    3. Save raw JSON to cache/cboe_raw/<date>_daily_options.json (gitignored)
    4. Extract P/C ratios + per-product volume/OI in one parse
    5. Append both into their respective wide CSVs

This is just _cboe_daily_cache() with a much bigger fetch budget. From scratch:
~1700 trading days × ~0.4 sec = ~10-15 min. Subsequent runs: a handful of days.
"""
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data import (
    _fetch_cboe_daily,
    _stats_long_to_wide,
    CBOE_RATIOS_WIDE_CSV,
    CBOE_VOL_OI_WIDE_CSV,
    CBOE_START_DATE,
)


def main():
    # 1. Load existing wide caches, pick the earlier resume point
    if CBOE_RATIOS_WIDE_CSV.exists():
        ratios_wide = pd.read_csv(CBOE_RATIOS_WIDE_CSV)
        ratios_wide["date"] = pd.to_datetime(ratios_wide["date"])
    else:
        ratios_wide = pd.DataFrame()

    if CBOE_VOL_OI_WIDE_CSV.exists():
        voloi_wide = pd.read_csv(CBOE_VOL_OI_WIDE_CSV)
        voloi_wide["date"] = pd.to_datetime(voloi_wide["date"])
    else:
        voloi_wide = pd.DataFrame()

    last_dates = []
    if not ratios_wide.empty: last_dates.append(ratios_wide["date"].max().date())
    if not voloi_wide.empty:  last_dates.append(voloi_wide["date"].max().date())

    if last_dates:
        start = min(last_dates) + dt.timedelta(days=1)
        print(f"Resuming from {start}  (ratios last: {ratios_wide['date'].max().date() if not ratios_wide.empty else 'empty'}, "
              f"vol/OI last: {voloi_wide['date'].max().date() if not voloi_wide.empty else 'empty'})")
    else:
        start = dt.date.fromisoformat(CBOE_START_DATE)
        print(f"No cache -> starting fresh from {start}")

    end = dt.date.today()
    trading_days = sum(1 for d in pd.date_range(start, end) if d.weekday() < 5)
    print(f"Will fetch up to {trading_days} trading days through {end}")
    if trading_days == 0:
        print("Nothing to do."); return

    # 2. Fetch loop
    new_ratio_rows = []
    new_voloi_rows = []
    success = miss = 0
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            r_rows, v_rows = _fetch_cboe_daily(cur.isoformat())
            if r_rows is None and v_rows is None:
                miss += 1
            else:
                success += 1
                if r_rows: new_ratio_rows.extend(r_rows)
                if v_rows: new_voloi_rows.extend(v_rows)
            total = success + miss
            if total % 50 == 0:
                print(f"  Progress: {total} days tried, {success} got data, {miss} missing")
        cur += dt.timedelta(days=1)
        time.sleep(0.02)

    # 3a. Merge new ratios into wide
    if new_ratio_rows:
        dn = pd.DataFrame(new_ratio_rows)
        dn["date"] = pd.to_datetime(dn["date"])
        dn["value"] = pd.to_numeric(dn["value"], errors="coerce")
        dn = dn.dropna(subset=["ratio_column"])
        if not dn.empty:
            pivoted = (
                dn.drop_duplicates(["date","ratio_column"], keep="last")
                .pivot(index="date", columns="ratio_column", values="value")
                .reset_index()
            )
            if not ratios_wide.empty:
                ratios_wide = pd.concat([ratios_wide, pivoted], ignore_index=True)
                ratios_wide = (
                    ratios_wide.drop_duplicates(subset=["date"], keep="last")
                    .sort_values("date").reset_index(drop=True)
                )
            else:
                ratios_wide = pivoted.sort_values("date").reset_index(drop=True)
            ratios_wide.to_csv(CBOE_RATIOS_WIDE_CSV, index=False)
            print(f"\n[OK] Ratios: {len(ratios_wide)} total rows "
                  f"({ratios_wide['date'].min().date()} -> {ratios_wide['date'].max().date()})")

    # 3b. Merge new volume/OI into wide
    if new_voloi_rows:
        ds = pd.DataFrame(new_voloi_rows)
        new_wide = _stats_long_to_wide(ds)
        if not new_wide.empty:
            if not voloi_wide.empty:
                voloi_wide = pd.concat([voloi_wide, new_wide], ignore_index=True)
                voloi_wide = (
                    voloi_wide.drop_duplicates(subset=["date","product"], keep="last")
                    .sort_values(["date","product"]).reset_index(drop=True)
                )
            else:
                voloi_wide = new_wide.sort_values(["date","product"]).reset_index(drop=True)
            voloi_wide.to_csv(CBOE_VOL_OI_WIDE_CSV, index=False)
            print(f"[OK] Vol/OI: {len(voloi_wide)} total rows "
                  f"({voloi_wide['date'].min().date()} -> {voloi_wide['date'].max().date()})")

    if not new_ratio_rows and not new_voloi_rows:
        print(f"\n[!] No new data. {success} success / {miss} miss out of {success+miss} tries.")


if __name__ == "__main__":
    main()
