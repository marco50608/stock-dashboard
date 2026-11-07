"""下載研究用的原始資料到 research/raw/（已被 .gitignore 排除，不會進 repo）。

用法（在 stock 專案根目錄）：
    PYTHONUTF8=1 .venv/Scripts/python research/fetch_data.py            # 全部
    PYTHONUTF8=1 .venv/Scripts/python research/fetch_data.py prices fred  # 只抓某幾類

所有來源都是免費公開資料。FRED 金鑰從 .env 的 FRED_API_KEY 讀取，絕不印出。
每個檔案都保留「觀測日」；「實際拿得到的日期」由 lib/data.py 的 availability_rules() 統一處理。
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _get(url: str, tries: int = 3, **kw) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=kw.pop("headers", UA), timeout=kw.pop("timeout", 60), **kw)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} -> {last}")


# ---------------------------------------------------------------------------
# 1. 價格（yfinance）。QQQ/SPY 用還原股息的收盤價 = 總報酬
# ---------------------------------------------------------------------------
YF_TICKERS = [
    "QQQ", "SPY", "^NDX", "^GSPC", "^IXIC", "^RUT", "^SOX", "^DJT",
    "^VIX", "^VIX3M", "^VIX9D", "^VVIX", "^SKEW", "^MOVE", "^IRX", "^TNX",
    "IWM", "RSP", "QQEW", "SMH", "SOXX",
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "HYG", "JNK", "LQD", "TLT", "IEF", "SHY", "GLD", "SLV", "UUP",
    "HG=F", "GC=F", "CL=F", "DX-Y.NYB", "JPY=X", "AUDJPY=X",
    "EEM", "EFA", "SPHB", "SPLV", "XHB", "XRT", "KRE", "IGV", "ARKK",
    "^VXN", "^PUT",   # 賣 put 補充研究（run_csp.py）：NDX 隱含波動、CBOE 官方 S&P 500 賣 put 指數
]


def fetch_prices() -> None:
    import yfinance as yf
    out = {}
    for t in YF_TICKERS:
        try:
            df = yf.download(t, period="max", auto_adjust=True, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            out[t] = df
            _log(f"  {t}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as e:  # noqa: BLE001
            _log(f"  {t}: FAILED {e}")
        time.sleep(0.5)
    for t, df in out.items():
        df.to_csv(RAW / "prices" / f"{t.replace('^', 'IDX_').replace('=', '_')}.csv")


# ---------------------------------------------------------------------------
# 2. CBOE 指數歷史（VIX 家族、SKEW）與 Put/Call 舊檔（2019-10 以前）
# ---------------------------------------------------------------------------
CBOE_INDEX = ["VIX", "VIX3M", "VIX9D", "VIX6M", "VVIX", "SKEW", "VXN", "COR3M", "COR1M", "VXD", "RVX"]
CBOE_PC_OLD = {
    # 這些 CSV 是 CBOE 過去提供的歷史檔，到 2019-10 停止更新；之後接 cache/ 的逐日資料
    "totalpc": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
    "equitypc": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
    "indexpc": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
    "etppc": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/etppc.csv",
    "vixpc": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/vixpc.csv",
    "spxpc": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/spxpc.csv",
    "totalpc_archive": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpcarchive.csv",
    "equitypc_archive": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypcarchive.csv",
    "indexpc_archive": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv",
}


def fetch_cboe() -> None:
    d = RAW / "cboe"
    d.mkdir(exist_ok=True)
    for name in CBOE_INDEX:
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
        try:
            r = _get(url)
            (d / f"{name}_History.csv").write_bytes(r.content)
            _log(f"  {name}: {len(r.content)} bytes")
        except Exception as e:  # noqa: BLE001
            _log(f"  {name}: FAILED {e}")
    for name, url in CBOE_PC_OLD.items():
        try:
            r = _get(url)
            (d / f"{name}.csv").write_bytes(r.content)
            _log(f"  {name}: {len(r.content)} bytes")
        except Exception as e:  # noqa: BLE001
            _log(f"  {name}: FAILED {e}")


def fetch_repo_cache() -> None:
    """GitHub 上的 cache 每週自動更新，比本機 cache 新。只下載到 raw/，不覆蓋使用者的 cache/。"""
    d = RAW / "repo_cache"
    d.mkdir(exist_ok=True)
    base = "https://raw.githubusercontent.com/marco50608/stock-dashboard/main/cache/"
    for f in ["cboe_ratios_wide.csv", "cboe_volume_oi_wide.csv", "naaim.xlsx",
              "aaii_sentiment.xls", "aaii_sentiment.xlsx", "finra_margin.xlsx"]:
        try:
            r = _get(base + f)
            (d / f).write_bytes(r.content)
            _log(f"  {f}: {len(r.content)} bytes")
        except Exception as e:  # noqa: BLE001
            _log(f"  {f}: FAILED {e}")


# ---------------------------------------------------------------------------
# 3. FRED / ALFRED（金鑰從 .env 讀，不印出）
# ---------------------------------------------------------------------------
FRED_SERIES = [
    # 信用
    "BAA10Y", "AAA10Y", "BAA", "AAA", "BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A0HYM2EY",
    # 利率／曲線
    "DGS10", "DGS2", "DGS3MO", "DGS1", "DTB3", "T10Y2Y", "T10Y3M", "DFII10", "T5YIE", "DFF",
    # 金融狀況、流動性
    "NFCI", "ANFCI", "STLFSI4", "WALCL", "WTREGEN", "RRPONTSYD", "WRESBAL", "M2SL",
    # 其他市場
    "VIXCLS", "DCOILWTICO", "DTWEXBGS", "DEXJPUS", "GOLDAMGBD228NLBM",
    # 景氣（只當標籤，絕不當訊號：NBER 衰退日期是事後公布的）
    "USREC", "USRECD",
    # 初領失業金目前版本（只用在 2009 年以前沒有 ALFRED vintage 的部分）
    "ICSA", "UMCSENT",
    # 第三輪（2026-10-25，見 PREREG_batch3_2026-10-25.md）：銀行放款標準、政策不確定性
    "DRTSCILM", "USEPUINDXD",
]
# 會被修正的總經資料：用 ALFRED 取「第一次公布的數字」和公布日
ALFRED_FIRST_RELEASE = ["ICSA", "CCSA", "PAYEMS", "UNRATE", "CPIAUCSL", "INDPRO", "RSAFS",
                        "PERMIT", "HOUST", "UMCSENT", "NFCI", "DRTSCILM"]


def _fred_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT.parent / ".env")
    except Exception:  # noqa: BLE001
        pass
    k = os.environ.get("FRED_API_KEY", "").strip()
    if not k:
        raise RuntimeError("找不到 FRED_API_KEY（.env）")
    return k


def fetch_fred(only: list[str] | None = None) -> None:
    key = _fred_key()
    d = RAW / "fred"
    d.mkdir(exist_ok=True)
    url = "https://api.stlouisfed.org/fred/series/observations"
    for sid in only or FRED_SERIES:
        try:
            r = requests.get(url, params={"series_id": sid, "api_key": key, "file_type": "json",
                                          "observation_start": "1900-01-01"}, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            obs = r.json().get("observations", [])
            s = pd.Series({o["date"]: o["value"] for o in obs})
            s = pd.to_numeric(s.replace(".", None), errors="coerce").dropna()
            s.index = pd.to_datetime(s.index)
            s.rename(sid).to_csv(d / f"{sid}.csv", index_label="date")
            _log(f"  {sid}: {len(s)} obs {s.index[0].date()} -> {s.index[-1].date()}")
        except Exception as e:  # noqa: BLE001
            _log(f"  {sid}: FAILED {str(e)[:80]}")
        time.sleep(0.3)


def fetch_alfred(only: list[str] | None = None) -> None:
    """每個觀測值第一次公布時的數字與公布日（output_type=4 = initial release only）。"""
    key = _fred_key()
    d = RAW / "alfred"
    d.mkdir(exist_ok=True)
    url = "https://api.stlouisfed.org/fred/series/observations"
    for sid in only or ALFRED_FIRST_RELEASE:
        try:
            r = requests.get(url, params={"series_id": sid, "api_key": key, "file_type": "json",
                                          "output_type": 4, "realtime_start": "1776-07-04",
                                          "realtime_end": "9999-12-31",
                                          "observation_start": "1960-01-01"}, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} {r.text[:120]}")
            obs = r.json().get("observations", [])
            df = pd.DataFrame(obs)
            df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
            df = df.rename(columns={"date": "obs_date", "realtime_start": "release_date"})
            df[["obs_date", "release_date", "value"]].to_csv(d / f"{sid}.csv", index=False)
            _log(f"  {sid}: {len(df)} first releases; earliest vintage {df['release_date'].min()}")
        except Exception as e:  # noqa: BLE001
            _log(f"  {sid}: FAILED {str(e)[:120]}")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# 4. CFTC COT（期貨部位）。TFF 2006+；Legacy（含 S&P 期貨）1986+
# ---------------------------------------------------------------------------
def fetch_cot() -> None:
    d = RAW / "cot"
    d.mkdir(exist_ok=True)
    jobs = [("fin_fut_txt_2006_2016.zip", "https://www.cftc.gov/files/dea/history/fin_fut_txt_2006_2016.zip"),
            ("deacot1986_2016.zip", "https://www.cftc.gov/files/dea/history/deacot1986_2016.zip")]
    for y in range(2017, 2027):
        jobs.append((f"fut_fin_txt_{y}.zip", f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip"))
        jobs.append((f"deacot{y}.zip", f"https://www.cftc.gov/files/dea/history/deacot{y}.zip"))
    for name, url in jobs:
        p = d / name
        if p.exists() and p.stat().st_size > 1000 and "2026" not in name:
            continue
        try:
            r = _get(url, timeout=120)
            p.write_bytes(r.content)
            _log(f"  {name}: {len(r.content)//1024} KB")
        except Exception as e:  # noqa: BLE001
            _log(f"  {name}: FAILED {e}")


# ---------------------------------------------------------------------------
# 5. Ken French 資料庫（產業投組，用來做「沒有倖存者偏差」的廣度）
# ---------------------------------------------------------------------------
FRENCH = {
    "F-F_Research_Data_Factors_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "49_Industry_Portfolios_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip",
    "12_Industry_Portfolios_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip",
}


def fetch_french() -> None:
    d = RAW / "french"
    d.mkdir(exist_ok=True)
    for name, url in FRENCH.items():
        try:
            r = _get(url, timeout=120)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for n in z.namelist():
                (d / n).write_bytes(z.read(n))
            _log(f"  {name}: ok ({', '.join(z.namelist())})")
        except Exception as e:  # noqa: BLE001
            _log(f"  {name}: FAILED {e}")


# ---------------------------------------------------------------------------
# 6. SEC 內部人交易（Form 3/4/5 結構化資料，2006Q1 起，每季一個 zip 約 15 MB）
#    只留下需要的欄位存成 parquet，zip 不保留。用 FILING_DATE 當「拿得到的日期」。
# ---------------------------------------------------------------------------
def _sec_ua() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT.parent / ".env")
    except Exception:  # noqa: BLE001
        pass
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC 要求可識別的 User-Agent：請在 .env 設定 SEC_USER_AGENT")
    return {"User-Agent": ua}


def fetch_insider() -> None:
    d = RAW / "insider"
    d.mkdir(exist_ok=True)
    headers = _sec_ua()
    base = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/"
    today = pd.Timestamp.today()
    quarters = [f"{y}q{q}" for y in range(2006, today.year + 1) for q in range(1, 5)]
    for qn in quarters:
        out = d / f"{qn}.parquet"
        if out.exists():
            continue
        try:
            r = requests.get(base + f"{qn}_form345.zip", headers=headers, timeout=180)
            if r.status_code != 200:
                _log(f"  {qn}: HTTP {r.status_code}（可能尚未發布）")
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            names = {n.upper().split("/")[-1]: n for n in z.namelist()}

            def tsv(n, cols):
                return pd.read_csv(z.open(names[n]), sep="\t", usecols=lambda c: c in cols,
                                   dtype=str, quoting=3, on_bad_lines="skip")
            sub = tsv("SUBMISSION.TSV", {"ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT",
                                         "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"})
            tr = tsv("NONDERIV_TRANS.TSV", {"ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                                            "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                            "TRANS_ACQUIRED_DISP_CD", "EQUITY_SWAP_INVOLVED"})
            own = tsv("REPORTINGOWNER.TSV", {"ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP", "RPTOWNERCIK"})
            own = own.groupby("ACCESSION_NUMBER").agg(
                RPTOWNER_RELATIONSHIP=("RPTOWNER_RELATIONSHIP", lambda x: "|".join(sorted(set(map(str, x))))),
                N_OWNERS=("RPTOWNERCIK", "nunique")).reset_index()
            tr = tr[tr["TRANS_CODE"].isin(["P", "S"])]
            m = tr.merge(sub, on="ACCESSION_NUMBER", how="left").merge(own, on="ACCESSION_NUMBER", how="left")
            m.to_parquet(out, index=False)
            _log(f"  {qn}: {len(m)} open-market P/S rows")
        except Exception as e:  # noqa: BLE001
            _log(f"  {qn}: FAILED {type(e).__name__}: {str(e)[:100]}")
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# 7. FOMC 例行會議日期（聲明發布日 = 會議最後一天）。不含臨時會議與電話會議
# ---------------------------------------------------------------------------
def fetch_fomc() -> None:
    import re
    names = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
    months = {m: i for i, m in enumerate(names, 1)}
    months.update({m[:3]: i for i, m in enumerate(names, 1)})
    months["Sept"] = 9
    rows = []
    # 例：「June 13-14 Meeting - 2017」「Jan/Feb 31-1 Meeting - 2017」「April 30-May 1 Meeting - 2002」
    pat_hist = re.compile(r"<h5[^>]*>\s*([A-Za-z/]+)\s+(\d+)(?:-(?:([A-Za-z]+)\s+)?(\d+))?\s+Meeting\s*-\s*(\d{4})\s*</h5>")
    for y in range(1994, 2021):
        html = _get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm").text
        for m1, d1, m2, d2, yy in pat_hist.findall(html):
            mon_end = m2 or m1.split("/")[-1]
            dd = int(d2) if d2 else int(d1)
            rows.append((f"{int(yy):04d}-{months[mon_end]:02d}-{dd:02d}", "historical"))
        time.sleep(0.3)
    html = _get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm").text
    for block in re.split(r'<div class="panel panel-default">', html)[1:]:
        ym = re.search(r"(\d{4}) FOMC Meetings", block)
        if not ym:
            continue
        y = int(ym.group(1))
        for mon, dates in re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<',
                                     block, flags=re.S):
            if "unscheduled" in dates.lower() or "notation" in dates.lower():
                continue
            mon_last = mon.split("/")[-1].strip()
            d_last = re.findall(r"\d+", dates)
            if mon_last in months and d_last:
                rows.append((f"{y:04d}-{months[mon_last]:02d}-{int(d_last[-1]):02d}", "calendar"))
            else:
                _log(f"  無法解析：{y} {mon} {dates}")
    df = pd.DataFrame(rows, columns=["date", "source"]).drop_duplicates("date").sort_values("date")
    d = pd.to_datetime(df["date"])
    df = df[~(d.shift(-1) - d <= pd.Timedelta(days=3)).values]   # 2003-09-15/16 這種連續兩天只留後一天
    df.to_csv(RAW / "fomc_dates.csv", index=False)
    _log(f"  FOMC: {len(df)} 例行會議 {df['date'].min()} -> {df['date'].max()}")


# ---------------------------------------------------------------------------
# 第三輪：Gilchrist-Zakrajšek 超額債券溢酬（Fed 官網，月資料，每次更新會重新估計整段歷史）
# ---------------------------------------------------------------------------
EBP_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"


def fetch_ebp() -> None:
    d = RAW / "fed"
    d.mkdir(exist_ok=True)
    r = requests.get(EBP_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"date", "ebp"} <= set(df.columns):
        raise RuntimeError(f"EBP 檔案格式變了：{list(df.columns)}")
    df.to_csv(d / "ebp.csv", index=False)
    _log(f"  EBP: {len(df)} months {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")


SECTIONS = {"fomc": fetch_fomc, "prices": fetch_prices, "cboe": fetch_cboe, "repo": fetch_repo_cache, "fred": fetch_fred,
            "alfred": fetch_alfred, "cot": fetch_cot, "french": fetch_french, "insider": fetch_insider, "ebp": fetch_ebp}

if __name__ == "__main__":
    # 用法：fetch_data.py [段落 ...]；fred、alfred 可以只抓幾個序列，例如 fred:DRTSCILM,USEPUINDXD
    (RAW / "prices").mkdir(exist_ok=True)
    which = sys.argv[1:] or list(SECTIONS)
    for w in which:
        name, _, ids = w.partition(":")
        _log(f"== {w}")
        if ids:
            SECTIONS[name](ids.split(","))
        else:
            SECTIONS[name]()
