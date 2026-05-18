"""
Data fetchers.

Each function is independent, cached aggressively, and returns either a
pandas object or raises a clearly-typed error the page can catch. No silent
failures — if a source is down or unavailable, the page should know.

All fetchers use Streamlit's @st.cache_data with a 1-day TTL (configurable
via the `ttl` argument to streamlit's cache).
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timedelta, date as _date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _config(key: str, default: str | None = None) -> str | None:
    """Get a config value from os.environ (local .env) or Streamlit secrets (cloud).
    Order: os.environ → st.secrets → default."""
    val = os.getenv(key)
    if val:
        return val
    try:
        v = st.secrets.get(key)
        if v:
            return str(v)
    except Exception:
        pass
    return default


class DataUnavailable(Exception):
    """Raised when a data source can't be fetched. Pages should catch and show a clear message."""


# ---------------------------------------------------------------------------
# Price data via yfinance
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_price(symbol: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV for a single ticker. Returns a DataFrame indexed by date."""
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            raise DataUnavailable(f"yfinance returned no data for {symbol}")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except DataUnavailable:
        raise
    except Exception as e:
        raise DataUnavailable(f"yfinance error for {symbol}: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def get_close(symbols: list[str], period: str = "10y") -> pd.DataFrame:
    """Close-only DataFrame for many tickers (wide format)."""
    out = {}
    for s in symbols:
        try:
            df = get_price(s, period=period)
            out[s] = df["Close"]
        except DataUnavailable:
            continue
    if not out:
        raise DataUnavailable(f"None of {symbols} returned price data")
    return pd.concat(out, axis=1).dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def get_vix_term_structure(period: str = "5y") -> pd.DataFrame:
    """VIX term structure proxy: spot + 1m / 3m / 6m."""
    symbols = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M"]
    return get_close(symbols, period=period)


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def _fred_key() -> str | None:
    return _config("FRED_API_KEY")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_fred_series(series_id: str, start: str = "2000-01-01") -> pd.Series:
    """Fetch a FRED series. Requires FRED_API_KEY in env or .env."""
    key = _fred_key()
    if not key:
        raise DataUnavailable(
            "FRED_API_KEY not set. Get a free key at "
            "https://fredaccount.stlouisfed.org/apikeys and add it to .env"
        )
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            raise DataUnavailable(f"FRED returned no observations for {series_id}")
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].dropna()
        s.name = series_id
        return s
    except DataUnavailable:
        raise
    except Exception as e:
        raise DataUnavailable(f"FRED fetch failed for {series_id}: {e}")


def get_hy_spread(start: str = "2000-01-01") -> pd.Series:
    """ICE BofA US High Yield Index Option-Adjusted Spread."""
    return get_fred_series("BAMLH0A0HYM2", start=start)


# ---------------------------------------------------------------------------
# CBOE Put/Call Ratio
# ---------------------------------------------------------------------------

def _append_pcr_snapshot():
    """Compute today's QQQ + SPY put/call ratio from yfinance option chains
    and append to cache/options_pcr_history.csv. Idempotent per day."""
    snap_path = CACHE_DIR / "options_pcr_history.csv"
    today = datetime.now().date()
    try:
        existing = pd.read_csv(snap_path) if snap_path.exists() else pd.DataFrame()
        if not existing.empty and "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
            if today in set(existing["date"]):
                return  # already have today
    except Exception:  # noqa: BLE001
        existing = pd.DataFrame()

    snapshots = {}
    for sym in ("QQQ", "SPY"):
        try:
            tk = yf.Ticker(sym)
            exps = tk.options[:3] if tk.options else []
            put_v = call_v = 0
            for exp in exps:
                ch = tk.option_chain(exp)
                put_v += float(ch.puts["volume"].fillna(0).sum())
                call_v += float(ch.calls["volume"].fillna(0).sum())
            if call_v > 0:
                snapshots[f"{sym.lower()}_pcr"] = put_v / call_v
        except Exception:  # noqa: BLE001
            continue
    if snapshots:
        row = {"date": today, **snapshots}
        new = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        new.to_csv(snap_path, index=False)


# ---------------------------------------------------------------------------
# CBOE daily Market Statistics — P/C ratios + per-product Volume/OI.
# Storage layout (in cache/):
#   cboe_raw/<YYYY-MM-DD>_daily_options.json  — per-day raw (gitignored)
#   cboe_ratios_wide.csv                      — wide P/C ratios (committed, canonical)
#   cboe_volume_oi_wide.csv                   — wide per-product volume + OI (committed)
#   cboe_ratios_long.csv                      — long P/C ratios (gitignored, redundant)
#   cboe_volume_oi_long.csv                   — long volume/OI (gitignored, redundant)
#   cboe_put_call_ratios.csv                  — original 3-col long (gitignored, redundant)
#
# JSON structure of a CBOE daily file (observed):
#   {
#     "ratios": [{"name": "TOTAL PUT/CALL RATIO", "value": "0.93"}, ...],
#     "SUM OF ALL PRODUCTS": [{"name": "VOLUME", "call": ..., "put": ..., "total": ...},
#                             {"name": "OPEN INTEREST", ...}],
#     "EQUITY OPTIONS": [...],
#     "EXCHANGE TRADED PRODUCTS": [...],
#     "INDEX OPTIONS": [...],
#     "CBOE VOLATILITY INDEX (VIX)": [...],
#     "SPX + SPXW": [...], "OEX": [...], "MRUT": [...], "MXEA": [...], ...
#   }
# ---------------------------------------------------------------------------

import re as _re

CBOE_RAW_DIR = CACHE_DIR / "cboe_raw"
CBOE_RATIOS_WIDE_CSV = CACHE_DIR / "cboe_ratios_wide.csv"
CBOE_RATIOS_LONG_CSV = CACHE_DIR / "cboe_ratios_long.csv"
CBOE_VOL_OI_WIDE_CSV = CACHE_DIR / "cboe_volume_oi_wide.csv"
CBOE_VOL_OI_LONG_CSV = CACHE_DIR / "cboe_volume_oi_long.csv"
# Legacy filename kept for back-compat with the original bootstrap script
CBOE_LONG_CSV = CACHE_DIR / "cboe_put_call_ratios.csv"
CBOE_START_DATE = "2019-10-07"  # first day with data on the CDN endpoint

# Friendly aliases used elsewhere in the app
CBOE_RATIO_FRIENDLY = {
    "total_pc":    "total",
    "index_pc":    "index",
    "equity_pc":   "equity",
    "etp_pc":      "etp",
    "spx_spxw_pc": "spx_spxw",
    "vix_pc":      "vix",
}

# Explicit ratio_name -> column code map (matches the user's extended extractor)
_RATIO_NAME_MAP = {
    "TOTAL PUT/CALL RATIO": "total_pc",
    "INDEX PUT/CALL RATIO": "index_pc",
    "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "etp_pc",
    "EQUITY PUT/CALL RATIO": "equity_pc",
    "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO": "vix_pc",
    "SPX + SPXW PUT/CALL RATIO": "spx_spxw_pc",
    "OEX PUT/CALL RATIO": "oex_pc",
    "MRUT PUT/CALL RATIO": "mrut_pc",
    "MXEA PUT/CALL RATIO": "mxea_pc",
    "MXEF PUT/CALL RATIO": "mxef_pc",
    "MXACW PUT/CALL RATIO": "mxacw_pc",
    "MXWLD PUT/CALL RATIO": "mxwld_pc",
    "MXUSA PUT/CALL RATIO": "mxusa_pc",
    "CBTX PUT/CALL RATIO": "cbtx_pc",
    "MBTX PUT/CALL RATIO": "mbtx_pc",
    "SPEQX PUT/CALL RATIO": "speqx_pc",
    "SPEQW PUT/CALL RATIO": "speqw_pc",
    "MGTN PUT/CALL RATIO": "mgtn_pc",
    "MGTNW PUT/CALL RATIO": "mgtnw_pc",
}


def _ratio_name_to_code(name: str) -> str | None:
    """Map ratio_name to short column code. Mirrors user's clean_ratio_column()."""
    if not name:
        return None
    if name in _RATIO_NAME_MAP:
        return _RATIO_NAME_MAP[name]
    col = str(name).lower().replace(" put/call ratio", "")
    col = _re.sub(r"[^a-z0-9]+", "_", col).strip("_")
    return f"{col}_pc" if col else None


def _extract_cboe_full(data: dict, date_str: str, source_file: str = "") -> tuple[list, list]:
    """Extract P/C ratios + per-product volume/OI from a CBOE daily JSON.

    Returns (ratio_rows, vol_oi_rows), both long-format lists of dicts:
      ratio_rows:  date, ratio_name, ratio_column, value, source_file
      vol_oi_rows: date, product, metric, call, put, total, source_file
    """
    ratio_rows = []
    for item in (data.get("ratios") or []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        ratio_rows.append({
            "date": date_str,
            "ratio_name": name,
            "ratio_column": _ratio_name_to_code(name),
            "value": item.get("value"),
            "source_file": source_file,
        })

    vol_oi_rows = []
    for product, items in data.items():
        if product == "ratios" or not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            metric = item.get("name")
            if metric not in ("VOLUME", "OPEN INTEREST"):
                continue
            vol_oi_rows.append({
                "date": date_str,
                "product": product,
                "metric": metric,
                "call": item.get("call"),
                "put": item.get("put"),
                "total": item.get("total"),
                "source_file": source_file,
            })
    return ratio_rows, vol_oi_rows


def _fetch_cboe_daily(date_str: str, save_raw: bool = True) -> tuple[list, list] | tuple[None, None]:
    """Fetch one trading day of CBOE daily Market Statistics JSON.

    Returns (ratio_rows, vol_oi_rows). Either may be empty list if section
    is missing; returns (None, None) on HTTP/parse failure.
    Reuses cached raw JSON in cache/cboe_raw/ when present (zero network).
    """
    CBOE_RAW_DIR.mkdir(exist_ok=True)
    raw_path = CBOE_RAW_DIR / f"{date_str}_daily_options.json"

    if raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None, None
    else:
        url = f"https://cdn.cboe.com/data/us/options/market_statistics/daily/{date_str}_daily_options"
        try:
            r = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json,text/plain,*/*",
                    "Referer": "https://www.cboe.com/markets/us/options/market-statistics/daily/",
                },
                timeout=30,
            )
            if r.status_code != 200:
                return None, None
            data = r.json()
        except Exception:
            return None, None
        if save_raw:
            try:
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    return _extract_cboe_full(data, date_str, source_file=raw_path.name)


def _wide_ratios_to_friendly(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Pick canonical P/C columns out of cboe_ratios_wide.csv with friendly names.

    Output: indexed by date, columns total/index/equity/etp/spx_spxw/vix.
    """
    if df_wide is None or df_wide.empty:
        return pd.DataFrame()
    df = df_wide.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    for src, alias in CBOE_RATIO_FRIENDLY.items():
        if src in df.columns:
            out[alias] = pd.to_numeric(df[src], errors="coerce")
    return out.dropna(how="all")


def _stats_long_to_wide(stats_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format volume/OI rows to wide layout matching user's extractor.

    Output columns: date, product, open_interest_call, volume_call,
                    open_interest_put, volume_put, open_interest_total, volume_total
    """
    if stats_long is None or stats_long.empty:
        return pd.DataFrame()
    s = stats_long.copy()
    s["date"] = pd.to_datetime(s["date"])
    for c in ("call", "put", "total"):
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    pivoted = (
        s.drop_duplicates(["date", "product", "metric"], keep="last")
        .pivot_table(
            index=["date", "product"],
            columns="metric",
            values=["call", "put", "total"],
            aggfunc="first",
        )
    )
    # Flatten MultiIndex columns: ("call","VOLUME") -> "volume_call"
    pivoted.columns = [
        f"{metric.lower().replace(' ', '_')}_{side}"
        for side, metric in pivoted.columns
    ]
    return pivoted.reset_index()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _cboe_daily_cache(max_fetch_per_call: int = 90) -> pd.DataFrame:
    """Incrementally update both CBOE wide caches; return friendly P/C ratios DF.

    Reads cache/cboe_ratios_wide.csv to find the last cached date, then for
    each missing trading day fetches the JSON and updates BOTH wide CSVs
    (ratios + volume/OI). Returns the friendly-aliased ratios DataFrame.
    """
    # Step 1: load existing ratios wide
    if CBOE_RATIOS_WIDE_CSV.exists():
        try:
            ratios_wide = pd.read_csv(CBOE_RATIOS_WIDE_CSV)
            if "date" in ratios_wide.columns:
                ratios_wide["date"] = pd.to_datetime(ratios_wide["date"])
        except Exception:
            ratios_wide = pd.DataFrame()
    else:
        ratios_wide = pd.DataFrame()

    # Step 2: load existing volume/OI wide
    if CBOE_VOL_OI_WIDE_CSV.exists():
        try:
            voloi_wide = pd.read_csv(CBOE_VOL_OI_WIDE_CSV)
            if "date" in voloi_wide.columns:
                voloi_wide["date"] = pd.to_datetime(voloi_wide["date"])
        except Exception:
            voloi_wide = pd.DataFrame()
    else:
        voloi_wide = pd.DataFrame()

    # Step 3: resume from earlier of the two (so a stale vol_oi file catches up too)
    candidates = []
    if not ratios_wide.empty and "date" in ratios_wide.columns:
        try: candidates.append(ratios_wide["date"].max().date())
        except Exception: pass
    if not voloi_wide.empty and "date" in voloi_wide.columns:
        try: candidates.append(voloi_wide["date"].max().date())
        except Exception: pass
    start = (min(candidates) + timedelta(days=1)) if candidates else _date.fromisoformat(CBOE_START_DATE)
    end = _date.today()

    # Step 4: fetch missing days, accumulate long-format rows
    new_ratio_rows = []
    new_voloi_rows = []
    fetched = 0
    cur = start
    while cur <= end and fetched < max_fetch_per_call:
        if cur.weekday() < 5:
            r_rows, v_rows = _fetch_cboe_daily(cur.isoformat())
            if r_rows:
                new_ratio_rows.extend(r_rows)
            if v_rows:
                new_voloi_rows.extend(v_rows)
            fetched += 1
            time.sleep(0.15)
        cur += timedelta(days=1)

    # Step 5a: fold new ratios into wide, persist
    if new_ratio_rows:
        dn = pd.DataFrame(new_ratio_rows)
        dn["date"] = pd.to_datetime(dn["date"])
        dn["value"] = pd.to_numeric(dn["value"], errors="coerce")
        dn = dn.dropna(subset=["ratio_column"])
        if not dn.empty:
            pivoted = (
                dn.drop_duplicates(["date", "ratio_column"], keep="last")
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
            try:
                ratios_wide.to_csv(CBOE_RATIOS_WIDE_CSV, index=False)
            except Exception:
                pass

    # Step 5b: fold new vol/OI into wide, persist
    if new_voloi_rows:
        ds = pd.DataFrame(new_voloi_rows)
        new_wide = _stats_long_to_wide(ds)
        if not new_wide.empty:
            if not voloi_wide.empty:
                voloi_wide = pd.concat([voloi_wide, new_wide], ignore_index=True)
                voloi_wide = (
                    voloi_wide.drop_duplicates(subset=["date", "product"], keep="last")
                    .sort_values(["date", "product"]).reset_index(drop=True)
                )
            else:
                voloi_wide = new_wide.sort_values(["date", "product"]).reset_index(drop=True)
            try:
                voloi_wide.to_csv(CBOE_VOL_OI_WIDE_CSV, index=False)
            except Exception:
                pass

    return _wide_ratios_to_friendly(ratios_wide)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_cboe_volume_oi() -> pd.DataFrame:
    """Per-product daily options volume + open interest from CBOE.

    Triggers the unified cache updater (so vol/OI catches up alongside ratios),
    then returns the wide-format DataFrame:
      columns: date, product, open_interest_call, volume_call,
               open_interest_put, volume_put, open_interest_total, volume_total
    Common products: 'CBOE VOLATILITY INDEX (VIX)', 'EQUITY OPTIONS',
                     'EXCHANGE TRADED PRODUCTS', 'INDEX OPTIONS',
                     'SPX + SPXW', 'SUM OF ALL PRODUCTS', ...
    """
    # Update path: forces _cboe_daily_cache to run, which updates both files
    try:
        _cboe_daily_cache()
    except Exception:
        pass
    if not CBOE_VOL_OI_WIDE_CSV.exists():
        raise DataUnavailable(
            "cache/cboe_volume_oi_wide.csv not found. Run "
            "notebooks/bootstrap_cboe.py to populate it."
        )
    try:
        df = pd.read_csv(CBOE_VOL_OI_WIDE_CSV)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["date", "product"]).reset_index(drop=True)
    except Exception as e:
        raise DataUnavailable(f"Failed to read cboe_volume_oi_wide.csv: {e}") from e



@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_put_call_ratio() -> pd.DataFrame:
    """CBOE Put/Call Ratio.

    Strategy (in order):
      0. **cache/cboe_pcr.csv** (manual download — most reliable, weekly update)
      1. Nasdaq.com public API for .CPC / .CPCE / .CPCI
      2. Stooq historical CSV
      3. CBOE JSON API
      4. yfinance tickers
      5. yfinance option chain → auto-accumulating snapshot
    """
    # 0a. Load manual archive CSV if provided (e.g. pre-2019 historical)
    archive_df = pd.DataFrame()
    local = CACHE_DIR / "cboe_pcr.csv"
    if local.exists():
        try:
            df = pd.read_csv(local)
            date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            num = df.select_dtypes(include="number")
            if not num.empty:
                archive_df = num[~num.index.duplicated(keep="last")]
        except Exception:
            pass

    # 0b. Auto-accumulate CBOE daily JSON (2019+, incrementally fetched)
    try:
        daily_df = _cboe_daily_cache()
    except Exception:
        daily_df = pd.DataFrame()

    # Merge: archive first, then daily takes precedence on overlap
    if not archive_df.empty or not daily_df.empty:
        combined = pd.concat([archive_df, daily_df])
        if not combined.empty:
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            return combined

    # 0. Nasdaq.com public API — historical data for .CPC / .CPCE / .CPCI.
    nasdaq_out = {}
    for label, sym in [("total", "CPC"), ("equity", "CPCE"), ("index", "CPCI")]:
        try:
            url = (
                f"https://api.nasdaq.com/api/quote/.{sym}/historical"
                f"?assetclass=index&fromdate=2018-01-01&limit=5000"
            )
            r = requests.get(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                    "Origin": "https://www.nasdaq.com",
                    "Referer": "https://www.nasdaq.com/",
                },
                timeout=20,
            )
            r.raise_for_status()
            payload = r.json().get("data", {})
            rows = payload.get("tradesTable", {}).get("rows", [])
            if not rows:
                continue
            df = pd.DataFrame(rows)
            # close field may be "close" or under another key; usually "close" is "$x.xx" or "x.xx"
            close_col = next((c for c in df.columns if "close" in c.lower()), None)
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            if close_col and date_col:
                s = pd.Series(
                    pd.to_numeric(
                        df[close_col].astype(str).str.replace(r"[\$,]", "", regex=True),
                        errors="coerce",
                    ).values,
                    index=pd.to_datetime(df[date_col], errors="coerce"),
                ).dropna().sort_index()
                if not s.empty:
                    nasdaq_out[label] = s
        except Exception:  # noqa: BLE001
            continue
    if nasdaq_out:
        return pd.concat(nasdaq_out, axis=1).dropna(how="all")

    # 1. Stooq — try both raw and URL-encoded symbol forms
    stooq_out = {}
    for label, sym in [("total", "cpc"), ("equity", "cpce"), ("index", "cpci")]:
        for prefix in ("%5E", "^"):  # URL-encoded caret + raw
            try:
                url = f"https://stooq.com/q/d/l/?s={prefix}{sym}&i=d"
                r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
                r.raise_for_status()
                text = r.text.strip()
                if not text or text.lower().startswith("no data") or "<html" in text.lower():
                    continue
                df = pd.read_csv(io.StringIO(text))
                if "Date" in df.columns and "Close" in df.columns:
                    s = pd.Series(
                        pd.to_numeric(df["Close"], errors="coerce").values,
                        index=pd.to_datetime(df["Date"]),
                    ).dropna()
                    if not s.empty:
                        stooq_out[label] = s
                        break
            except Exception:  # noqa: BLE001
                continue
    if stooq_out:
        return pd.concat(stooq_out, axis=1).dropna(how="all")

    # 1. CBOE JSON endpoints — try a few known patterns
    cboe_candidates = [
        ("total",  "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_CPC.json"),
        ("equity", "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_CPCE.json"),
        ("index",  "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_CPCI.json"),
    ]
    cboe_out = {}
    for label, url in cboe_candidates:
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            r.raise_for_status()
            data = r.json()
            # CBOE format: nested under 'data' with 'close' or 'price' fields
            payload = data.get("data", data)
            # Try common shapes
            series_data = None
            if isinstance(payload, dict):
                # shape: {"close": [...], "date": [...]}
                if "close" in payload and "date" in payload:
                    series_data = pd.Series(payload["close"], index=pd.to_datetime(payload["date"]))
                # shape: {"prices": [{"date":..., "close":...}, ...]}
                elif "prices" in payload:
                    df = pd.DataFrame(payload["prices"])
                    if "date" in df.columns and "close" in df.columns:
                        series_data = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
            elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
                df = pd.DataFrame(payload)
                date_col = next((c for c in df.columns if "date" in c.lower()), None)
                val_col = next((c for c in df.columns if c.lower() in ("close", "value", "price")), None)
                if date_col and val_col:
                    series_data = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col]))
            if series_data is not None and not series_data.empty:
                cboe_out[label] = pd.to_numeric(series_data, errors="coerce").dropna()
        except Exception:  # noqa: BLE001
            continue
    if cboe_out:
        return pd.concat(cboe_out, axis=1).dropna(how="all")

    # 2. yfinance backup
    yf_out = {}
    for label, tk in [("total", "^CPC"), ("equity", "^CPCE"), ("index", "^CPCI")]:
        try:
            df = get_price(tk, period="10y")
            yf_out[label] = df["Close"]
        except DataUnavailable:
            continue
    if yf_out:
        return pd.concat(yf_out, axis=1).dropna(how="all")

    # 3. yfinance option chain snapshot — append today's QQQ/SPY P/C to local CSV.
    #    This auto-bootstraps a history: run the app daily and it accumulates.
    try:
        _append_pcr_snapshot()
    except Exception:  # noqa: BLE001
        pass
    snap_path = CACHE_DIR / "options_pcr_history.csv"
    if snap_path.exists():
        df = pd.read_csv(snap_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            num = df.select_dtypes(include="number")
            if not num.empty:
                return num

    # 4. User-provided CBOE historical CSV
    local = CACHE_DIR / "cboe_pcr.csv"
    if local.exists():
        df = pd.read_csv(local)
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        return df.select_dtypes(include="number")

    raise DataUnavailable(
        "Put/Call ratio unavailable. Stooq/CBOE/Yahoo all failed, and no "
        f"local cache exists. Download CSV from cboe.com historical data and "
        f"save to {local}, OR run the app daily — it will auto-snapshot from "
        f"QQQ/SPY option chains into {snap_path}."
    )


# ---------------------------------------------------------------------------
# CFTC Commitment of Traders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_cot_financial() -> pd.DataFrame:
    """CFTC Traders in Financial Futures — weekly.

    Public ZIP file at CFTC. We download and filter to relevant markets.
    Returns a long-format DataFrame.
    """
    # CFTC publishes annual TXT files; we pull a single 'historical' archive
    # The 'FinFutWk.txt' is the most-recent week. For history, FinFutWk_YYYY.zip.
    url = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df
    except Exception as e:  # noqa: BLE001
        raise DataUnavailable(f"CFTC fetch failed: {e}")


# ---------------------------------------------------------------------------
# FINRA Margin Debt
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_margin_debt() -> pd.DataFrame:
    """FINRA monthly Margin Statistics.

    Strategy:
      1. FINRA Data API (if FINRA_CLIENT_ID/SECRET set in .env).
         API doesn't actually have this dataset; we still try but fall through silently.
      2. Direct XLSX URL (2021-03/margin-statistics.xlsx) — FINRA's stable file path.
      3. Scrape landing pages for any xlsx link.
      4. Local fallback at cache/finra_margin.xlsx.
    """
    # 1. Official API — don't let API failure block the scraping path
    try:
        api_df = get_margin_debt_via_api()
        if api_df is not None and not api_df.empty:
            return api_df
    except DataUnavailable:
        pass  # API doesn't have this dataset; fall through to scraping

    # Build a fetcher that can handle Cloudflare bot protection.
    # cloudscraper is optional — fall back to plain requests if not installed.
    try:
        import cloudscraper
        fetcher = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows"})
    except ImportError:
        fetcher = requests.Session()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics",
    }
    urls_to_try: list[str] = [
        # FINRA keeps the master file at this fixed path; they overwrite it monthly.
        # The "2021-03" in the path is just the directory, NOT the data month.
        "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx",
    ]

    # Scrape both possible landing pages
    landing_pages = [
        "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics",
        "https://www.finra.org/finra-data/browse-catalog/margin-statistics",
    ]
    import re
    for landing in landing_pages:
        try:
            r = fetcher.get(landing, headers=headers, timeout=20)
            r.raise_for_status()
            found = re.findall(r'href=["\']([^"\']+\.xlsx?)["\']', r.text, flags=re.I)
            for u in found:
                full = u if u.startswith("http") else "https://www.finra.org" + u
                if full not in urls_to_try:
                    urls_to_try.insert(0, full)  # prepend — scraped links are more likely current
        except Exception:  # noqa: BLE001
            continue

    # Static backups
    urls_to_try.append("https://www.finra.org/sites/default/files/margin-statistics.xlsx")

    attempts = []  # record all attempts for clearer error message
    for url in urls_to_try:
        try:
            r = fetcher.get(url, headers=headers, timeout=30)
            attempts.append(f"  {url} -> HTTP {r.status_code}")
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content))
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next((c for c in df.columns if "year" in c.lower() or "month" in c.lower() or "date" in c.lower()), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            return df
        except Exception as e:  # noqa: BLE001
            attempts.append(f"  {url} -> {type(e).__name__}: {str(e)[:80]}")
            continue

    local = CACHE_DIR / "finra_margin.xlsx"
    if local.exists():
        df = pd.read_excel(local)
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next((c for c in df.columns if "year" in c.lower() or "month" in c.lower() or "date" in c.lower()), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    raise DataUnavailable(
        "FINRA 自動下載失敗（Cloudflare 擋住）。\n\n試過的 URL：\n" +
        "\n".join(attempts) +
        f"\n\n請用瀏覽器到 https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics "
        f"下載 XLSX 存到 {local}（每月一次）。"
    )


# ---------------------------------------------------------------------------
# NAAIM Exposure Index
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_naaim() -> pd.DataFrame:
    """NAAIM Exposure Index — weekly active manager exposure.

    Strategy:
      1. Scrape https://www.naaim.org/programs/naaim-exposure-index/ for the
         current XLSX link.
      2. Try year-month patterns for the upload directory.
      3. Local fallback at cache/naaim.xlsx.
    """
    headers = {"User-Agent": USER_AGENT}
    urls_to_try: list[str] = []

    # 1. Scrape the page
    try:
        r = requests.get(
            "https://www.naaim.org/programs/naaim-exposure-index/",
            headers=headers, timeout=20,
        )
        r.raise_for_status()
        import re
        found = re.findall(r'href=["\']([^"\']+\.xlsx?)["\']', r.text, flags=re.I)
        for u in found:
            if "naaim" in u.lower() or "exposure" in u.lower():
                urls_to_try.append(u if u.startswith("http") else "https://www.naaim.org" + u)
    except Exception:  # noqa: BLE001
        pass

    # 2. Year-month directory guesses for many filename variants
    today = datetime.now()
    filename_variants = [
        "NAAIM-Exposure-Index-Data.xlsx",
        "NAAIM-Exposure-Index.xlsx",
        "NAAIM-Exposure.xlsx",
        "naaim-exposure-index-data.xlsx",
        "naaim_exposure_index_data.xlsx",
    ]
    for back in range(0, 36):
        y = today.year
        m = today.month - back
        while m <= 0:
            m += 12
            y -= 1
        for fname in filename_variants:
            urls_to_try.append(
                f"https://www.naaim.org/wp-content/uploads/{y}/{m:02d}/{fname}"
            )

    # 3. Legacy URLs
    urls_to_try.extend([
        "https://www.naaim.org/wp-content/uploads/2014/04/NAAIM-Exposure-Index-Data.xlsx",
        "https://www.naaim.org/wp-content/uploads/2013/05/NAAIM-Exposure-Index-Data.xlsx",
    ])

    last_err = None
    for url in urls_to_try:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content))
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next((c for c in df.columns if "date" in c.lower() or "week" in c.lower()), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    # Local fallback
    local = CACHE_DIR / "naaim.xlsx"
    if local.exists():
        df = pd.read_excel(local)
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c.lower() or "week" in c.lower()), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    raise DataUnavailable(
        f"NAAIM fetch failed (last error: {last_err}). "
        f"Download from https://www.naaim.org/programs/naaim-exposure-index/ "
        f"and save to {local}"
    )


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AAII Sentiment - manual CSV fallback
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_aaii_sentiment() -> pd.DataFrame:
    """AAII Sentiment Survey weekly readings.

    AAII requires login for their data. We try a known public URL; if it
    fails, look for cache/aaii_sentiment.{csv,xlsx}.
    """
    candidates = [
        "https://www.aaii.com/files/surveys/sentiment.xls",
    ]
    for url in candidates:
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content), skiprows=3)
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next((c for c in df.columns if "date" in c.lower() or "week" in c.lower()), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            return df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        except Exception:
            continue

    # Local fallbacks — accept .csv, .xls, .xlsx with various filenames
    candidates_local = [
        CACHE_DIR / "aaii_sentiment.csv",
        CACHE_DIR / "aaii_sentiment.xls",
        CACHE_DIR / "aaii_sentiment.xlsx",
        CACHE_DIR / "sentiment.xls",
        CACHE_DIR / "sentiment.xlsx",
    ]
    for p in candidates_local:
        if not p.exists():
            continue
        try:
            if p.suffix == ".csv":
                df = pd.read_csv(p)
            else:
                # AAII's sentiment.xls has a "SENTIMENT" sheet with skiprows=3.
                # Try that first; fall back to default sheet.
                try:
                    df = pd.read_excel(p, sheet_name="SENTIMENT", skiprows=3)
                except Exception:
                    df = pd.read_excel(p)
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next(
                (c for c in df.columns if "date" in c.lower() or "week" in c.lower()),
                df.columns[0],
            )
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            # Detect decimal vs percentage. AAII publishes as 0..1; we want 0..100
            for col in ("Bullish", "Bearish", "Neutral", "Spread"):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if s.abs().max() < 2:  # values like 0.38 — decimal form
                        df[col] = s * 100
                    else:
                        df[col] = s
            return df
        except Exception:  # noqa: BLE001
            continue

    raise DataUnavailable(
        "AAII sentiment unavailable. AAII requires login. "
        "Download from https://www.aaii.com/sentimentsurvey/sent_results and "
        f"save to {CACHE_DIR / 'aaii_sentiment.xls'}"
    )


# ---------------------------------------------------------------------------
# Breadth (legacy, kept for back-compat)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_breadth_index() -> pd.DataFrame:
    """Legacy: Yahoo S&P500 breadth (^MMTH, ^MMFI). Often empty now."""
    tickers = {"spx_above_200dma": "^MMTH", "spx_above_50dma": "^MMFI"}
    out = {}
    for label, t in tickers.items():
        try:
            df = get_price(t, period="10y")
            out[label] = df["Close"]
        except DataUnavailable:
            continue
    if out:
        return pd.concat(out, axis=1)
    raise DataUnavailable("Yahoo breadth tickers (^MMTH, ^MMFI) returned no data")


# ---------------------------------------------------------------------------
# Mag 7 helpers
# ---------------------------------------------------------------------------

MAG7 = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_mag7_marketcaps() -> dict:
    """Current market caps of the Mag 7 (approx, via yfinance fast_info)."""
    caps = {}
    for s in MAG7:
        try:
            info = yf.Ticker(s).fast_info
            mc = getattr(info, "market_cap", None)
            if mc is None:
                try:
                    mc = info["market_cap"]
                except Exception:
                    mc = None
            caps[s] = float(mc) if mc is not None else float("nan")
        except Exception:
            caps[s] = float("nan")
    return caps


# ---------------------------------------------------------------------------
# NDX constituents + computed breadth (from slickcharts)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_ndx_constituents() -> pd.DataFrame:
    """Scrape Nasdaq-100 constituents and weights from slickcharts."""
    url = "https://www.slickcharts.com/nasdaq100"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        if not tables:
            raise DataUnavailable("slickcharts returned no tables")
        df = tables[0].copy()
        df.columns = [str(c).strip() for c in df.columns]
        sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
        wt_col = next((c for c in df.columns if "weight" in c.lower() or c.endswith("%")), None)
        if sym_col is None:
            raise DataUnavailable(f"slickcharts: symbol column not found; cols={list(df.columns)}")
        if wt_col is not None:
            df["weight_pct"] = pd.to_numeric(
                df[wt_col].astype(str).str.replace("%", "", regex=False).str.strip(),
                errors="coerce",
            )
        df["Symbol"] = df[sym_col].astype(str).str.strip()
        keep = ["Symbol"] + (["weight_pct"] if "weight_pct" in df.columns else [])
        comp_col = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None)
        if comp_col:
            df["Company"] = df[comp_col]
            keep.append("Company")
        out = df[keep].dropna(subset=["Symbol"]).reset_index(drop=True)
        if "weight_pct" in out.columns:
            out = out.sort_values("weight_pct", ascending=False).reset_index(drop=True)
        return out
    except DataUnavailable:
        raise
    except Exception as e:
        raise DataUnavailable(f"slickcharts NDX scrape failed: {e}")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def compute_ndx_breadth(period: str = "3y", limit: int = 100) -> pd.DataFrame:
    """Compute breadth metrics from NDX constituents."""
    try:
        const = get_ndx_constituents()
    except DataUnavailable as e:
        raise DataUnavailable(f"need constituents: {e}")
    symbols = const["Symbol"].dropna().astype(str).tolist()[:limit]
    try:
        data = yf.download(
            tickers=symbols, period=period, interval="1d",
            auto_adjust=False, progress=False, threads=True, group_by="ticker",
        )
    except Exception as e:
        raise DataUnavailable(f"yfinance batch download failed: {e}")
    closes = {}
    if isinstance(data.columns, pd.MultiIndex):
        for s in symbols:
            if (s, "Close") in data.columns:
                closes[s] = data[(s, "Close")]
    else:
        closes[symbols[0]] = data.get("Close")
    px = pd.DataFrame(closes).dropna(how="all")
    if px.empty:
        raise DataUnavailable("no close data for any NDX constituent")
    above50 = (px > px.rolling(50).mean()).sum(axis=1) / px.notna().sum(axis=1) * 100
    above200 = (px > px.rolling(200).mean()).sum(axis=1) / px.notna().sum(axis=1) * 100
    ret = px.pct_change()
    adv = (ret > 0).sum(axis=1)
    dec = (ret < 0).sum(axis=1)
    high52 = (px >= px.rolling(252).max()).sum(axis=1)
    low52 = (px <= px.rolling(252).min()).sum(axis=1)
    out = pd.DataFrame({
        "pct_above_50dma": above50,
        "pct_above_200dma": above200,
        "advances": adv,
        "declines": dec,
        "new_highs_52w": high52,
        "new_lows_52w": low52,
    })
    if hasattr(out.index, "tz") and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.to_datetime(out.index)
    return out.dropna(how="all")


# ---------------------------------------------------------------------------
# Valuations (trailing PE / forward PE)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_stock_valuations(symbols: tuple) -> pd.DataFrame:
    """Trailing PE, forward PE, name and market cap for each ticker."""
    rows = []
    for s in symbols:
        d = {"symbol": s, "name": s, "trailing_pe": None,
             "forward_pe": None, "market_cap": None}
        try:
            info = yf.Ticker(s).info or {}
            d["name"] = info.get("shortName") or info.get("longName") or s
            d["trailing_pe"] = info.get("trailingPE")
            d["forward_pe"] = info.get("forwardPE")
            d["market_cap"] = info.get("marketCap")
        except Exception:
            pass
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FINRA Data API (OAuth2)
# ---------------------------------------------------------------------------

def _finra_oauth_token():
    """Exchange FINRA client_id/secret for an access token.
    Returns (token, diagnostic_message)."""
    cid = _config("FINRA_CLIENT_ID")
    secret = _config("FINRA_CLIENT_SECRET")
    if not cid or not secret:
        return None, "未設定 FINRA_CLIENT_ID / FINRA_CLIENT_SECRET"
    if cid.strip() != cid or secret.strip() != secret:
        return None, "credentials 前後有空白"
    url = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
    try:
        r = requests.post(url, auth=(cid.strip(), secret.strip()),
                          data={"grant_type": "client_credentials"},
                          timeout=15, headers={"Accept": "application/json"})
        if r.status_code == 401:
            return None, f"OAuth 401: {r.text[:200]}"
        if r.status_code >= 400:
            return None, f"OAuth HTTP {r.status_code}: {r.text[:200]}"
        tok = r.json().get("access_token")
        if not tok:
            return None, f"no access_token in response: {r.text[:200]}"
        return tok, "OK"
    except Exception as e:
        return None, f"OAuth exception: {e}"


def get_margin_debt_via_api():
    """Try FINRA Data API. Returns None if no creds; raises DataUnavailable if API path fails."""
    cid = _config("FINRA_CLIENT_ID")
    if not cid:
        return None
    token, diag = _finra_oauth_token()
    if not token:
        raise DataUnavailable(f"FINRA OAuth: {diag}")
    # FINRA Data Platform does not actually expose Margin Statistics.
    # Try a few endpoints just in case the catalog has changed; otherwise raise honestly.
    endpoints = [
        ("otcMarket", "MarginStatistics"),
        ("regulatoryAffairs", "MarginStatistics"),
        ("equityMarketTransparency", "MarginStatistics"),
    ]
    last_err = ""
    for group, name in endpoints:
        url = f"https://api.finra.org/data/group/{group}/name/{name}"
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {token}",
                                            "Accept": "application/json"},
                             params={"limit": 100}, timeout=20)
            if r.status_code == 200:
                batch = r.json()
                if isinstance(batch, list) and batch:
                    df = pd.DataFrame(batch)
                    date_col = next((c for c in df.columns
                                     if any(k in c.lower() for k in ("date", "period", "month", "year"))),
                                    None)
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
                    return df
            last_err = f"HTTP {r.status_code} at /{group}/{name}"
        except Exception as e:
            last_err = f"exception at /{group}/{name}: {e}"
            continue
    raise DataUnavailable(
        "FINRA Data Platform API 不包含 Margin Statistics 資料集。"
        "該資料只透過 FINRA 網站每月 XLSX 公佈。"
        "OAuth 認證雖然成功，但這個資料集本來就拉不到。"
        f"請用瀏覽器到 https://www.finra.org/finra-data/browse-catalog/margin-statistics "
        f"下載最新 XLSX 存到 {CACHE_DIR / 'finra_margin.xlsx'}"
    )


# ---------------------------------------------------------------------------
# SEC EDGAR - 13F filings
# ---------------------------------------------------------------------------

TRACKED_FUNDS = {
    "Berkshire Hathaway": "0001067983",
    "Pershing Square": "0001336528",
    "Tiger Global": "0001167483",
    "Renaissance Tech": "0001037389",
    "Bridgewater": "0001350694",
    "Coatue": "0001135730",
    "Third Point": "0001040273",
    "Soros Fund": "0001029160",
    "Appaloosa": "0001656456",
    "Greenlight Capital": "0001079114",
}


def _sec_headers():
    return {"User-Agent": _config("SEC_USER_AGENT", "Personal Research personal@example.com")}


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_latest_13f(cik: str) -> dict:
    """Find the latest 13F-HR filing metadata for a CIK."""
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        r = requests.get(url, headers=_sec_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise DataUnavailable(f"SEC EDGAR submissions fetch failed: {e}")
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accs = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    periods = filings.get("reportDate", [])
    for i, f in enumerate(forms):
        if f.upper() in ("13F-HR", "13F-HR/A"):
            return {
                "accession": accs[i].replace("-", ""),
                "accession_dashed": accs[i],
                "filing_date": dates[i],
                "period": periods[i] if i < len(periods) else "",
                "fund_name": data.get("name", ""),
            }
    raise DataUnavailable(f"No 13F-HR filing found for CIK {cik}")


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def parse_13f_holdings(cik: str, accession_clean: str) -> pd.DataFrame:
    """Parse a 13F-HR info table XML and return aggregated holdings."""
    cik_int = int(cik)
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/"
    idx_url = base + "index.json"
    try:
        r = requests.get(idx_url, headers=_sec_headers(), timeout=20)
        r.raise_for_status()
        idx = r.json()
    except Exception as e:
        raise DataUnavailable(f"SEC EDGAR index fetch failed: {e}")

    info_xml_name = None
    xml_candidates = []
    for item in idx.get("directory", {}).get("item", []):
        name = item.get("name", "")
        nl = name.lower()
        if not nl.endswith(".xml"):
            continue
        if "primary_doc" in nl or "primarydoc" in nl:
            continue
        xml_candidates.append(name)
        if "infotable" in nl or "informationtable" in nl or "form13f" in nl:
            info_xml_name = name
            break
    if info_xml_name is None and xml_candidates:
        info_xml_name = xml_candidates[0]
    if info_xml_name is None:
        names = [it.get("name", "") for it in idx.get("directory", {}).get("item", [])]
        raise DataUnavailable(f"13F info table XML not found. Files: {names}")

    xml_url = base + info_xml_name
    try:
        r = requests.get(xml_url, headers=_sec_headers(), timeout=30)
        r.raise_for_status()
        text = r.text
    except Exception as e:
        raise DataUnavailable(f"13F XML fetch failed: {e}")

    rows = []
    try:
        from lxml import etree
        root = etree.fromstring(text.encode("utf-8"))
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]
        for it in root.iter("infoTable"):
            d = {child.tag: (child.text or "").strip() for child in it}
            shrs_node = next((c for c in it if c.tag == "shrsOrPrnAmt"), None)
            if shrs_node is not None:
                for sub in shrs_node:
                    d[sub.tag] = (sub.text or "").strip()
            rows.append(d)
    except Exception as e:
        raise DataUnavailable(f"13F XML parse failed: {e}")

    if not rows:
        raise DataUnavailable("13F XML contained no holdings")

    df = pd.DataFrame(rows)
    if "value" in df.columns:
        df["value_usd"] = pd.to_numeric(df["value"], errors="coerce") * 1000
    if "sshPrnamt" in df.columns:
        df["shares"] = pd.to_numeric(df["sshPrnamt"], errors="coerce")
    keep = [c for c in ["nameOfIssuer", "cusip", "value_usd", "shares"] if c in df.columns]
    df = df[keep]
    if "value_usd" in df.columns:
        df = df.dropna(subset=["value_usd"])
    if "nameOfIssuer" in df.columns:
        agg_spec = {"value_usd": "sum"}
        if "shares" in df.columns:
            agg_spec["shares"] = "sum"
        if "cusip" in df.columns:
            agg_spec["cusip"] = "first"
        agg = df.groupby("nameOfIssuer", as_index=False).agg(agg_spec)
        return agg.sort_values("value_usd", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 10-year Treasury yield + CNN Fear & Greed Index
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_us10y(period: str = "5y") -> pd.Series:
    """US 10-year Treasury yield. Primary: yfinance ^TNX (divided by 10 to get %).
    Fallback: FRED DGS10."""
    try:
        df = get_price("^TNX", period=period)
        s = df["Close"] / 10  # ^TNX is reported as basis points
        s.name = "us10y"
        return s
    except DataUnavailable:
        pass
    try:
        return get_fred_series("DGS10", start="2018-01-01")
    except DataUnavailable as e:
        raise DataUnavailable(f"10y yield unavailable: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def get_cnn_fear_greed() -> dict:
    """CNN Business Fear & Greed Index + 7 sub-indicators.

    Returns dict with:
      - score: current 0-100 value
      - rating: textual rating
      - timestamp: ISO datetime
      - historical: list of {x: timestamp_ms, y: score, rating}
      - indicators: dict of 7 sub-indicators, each with same structure
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.cnn.com",
        "Referer": "https://www.cnn.com/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        raise DataUnavailable(f"CNN F&G fetch failed: {e}")

    fg = d.get("fear_and_greed", {})
    out = {
        "score": fg.get("score"),
        "rating": fg.get("rating"),
        "timestamp": fg.get("timestamp"),
        "previous_close": fg.get("previous_close"),
        "previous_1_week": fg.get("previous_1_week"),
        "previous_1_month": fg.get("previous_1_month"),
        "previous_1_year": fg.get("previous_1_year"),
        "historical": d.get("fear_and_greed_historical", {}).get("data", []),
    }
    # 7 sub-indicators
    sub_keys = [
        ("market_momentum_sp500", "市場動能 (S&P500 vs 125日均線)"),
        ("stock_price_strength", "股價強弱 (52週新高 vs 新低)"),
        ("stock_price_breadth", "股價廣度 (McClellan 量能)"),
        ("put_call_options", "Put/Call 比率 (5日)"),
        ("market_volatility_vix", "市場波動 (VIX)"),
        ("safe_haven_demand", "避險需求 (股 vs 債 20日報酬)"),
        ("junk_bond_demand", "垃圾債需求 (利差)"),
    ]
    indicators = {}
    for key, label in sub_keys:
        sub = d.get(key, {})
        indicators[key] = {
            "label": label,
            "score": sub.get("score"),
            "rating": sub.get("rating"),
            "data": sub.get("data", []),
        }
    out["indicators"] = indicators
    return out


@st.cache_data(ttl=300, show_spinner=False)
def get_intraday(symbol: str, interval: str = "5m") -> pd.DataFrame:
    """Today's intraday OHLCV. Defaults to 5-minute bars."""
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="1d", interval=interval, auto_adjust=False)
        if df is None or df.empty:
            # Market may be closed; try 5d for last session
            df = t.history(period="5d", interval=interval, auto_adjust=False)
            if df is not None and not df.empty:
                last_day = df.index.date.max()
                df = df[df.index.date == last_day]
        if df is None or df.empty:
            raise DataUnavailable(f"no intraday data for {symbol}")
        df.index = pd.to_datetime(df.index)
        return df
    except DataUnavailable:
        raise
    except Exception as e:
        raise DataUnavailable(f"intraday fetch failed for {symbol}: {e}")


def find_support_levels(price_series: pd.Series, n: int = 2, lookback: int = 90) -> list:
    """Find the N most significant support levels below current price.

    Uses local minima within a sliding window, falls back to N-day rolling lows.
    Returns list of (date, level) tuples sorted by level descending (closest support first).
    """
    s = price_series.dropna().tail(lookback)
    if s.empty:
        return []
    current = float(s.iloc[-1])

    # 1. Detect local minima with progressively narrower windows
    candidates = []
    vals = s.values
    for window in (5, 3, 2):  # try increasingly narrow windows for more candidates
        for i in range(window, len(vals) - window):
            if vals[i] == min(vals[i-window : i+window+1]) and vals[i] < current * 0.995:
                candidates.append((s.index[i], float(vals[i])))
        if len(candidates) >= n * 3:  # enough candidates from this window
            break

    # 2. Fallback: rolling lows of various lookback windows
    for w in (20, 40, lookback):
        sub = s.tail(w)
        if not sub.empty:
            idx_min = sub.idxmin()
            lvl = float(sub.loc[idx_min])
            if lvl < current * 0.995:
                candidates.append((idx_min, lvl))

    # Sort by level descending (closest below current first)
    candidates.sort(key=lambda x: -x[1])

    # Deduplicate near-identical levels (within 1% of current price)
    deduped = []
    for dt, lvl in candidates:
        too_close = any(abs(lvl - d[1]) / current < 0.01 for d in deduped)
        if not too_close:
            deduped.append((dt, lvl))
        if len(deduped) >= n:
            break

    return deduped
