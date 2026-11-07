"""研究用資料載入：每個序列都換成「在交易日 D 收盤後，當時實際拿得到的值」。

核心規則（所有訊號共用，見 availability_rules()）：
- 每筆資料有「觀測日」和「公開日」。公開日之前，回測完全看不到它。
- 訊號在 D 收盤後計算，交易在 D+1 收盤執行（lib/backtest.py 的 EXEC_LAG=2：
  部位在 D+2 的報酬才生效）。這比實際（台灣早上看訊號、當晚美股盤中下單）更保守。
- 會被修正的總經數字（初領失業金、非農）用 ALFRED 的「第一次公布值」。

只用指數／ETF 層級的資料，不用「現在的成分股」回推歷史，避免倖存者偏差。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "raw"
STOCK_ROOT = Path(__file__).resolve().parent.parent.parent

NDX_DIV_YIELD_PRE_QQQ = 0.007  # QQQ 上市前用 ^NDX 價格指數 + 每年約 0.7% 股息近似總報酬
WEEKLY_STALE = 15    # 週資料超過 15 個交易日沒有新值就當作缺值（不無限往後補）
MONTHLY_STALE = 50   # 月資料


def availability_rules() -> pd.DataFrame:
    rows = [
        ("價格、VIX 家族、CBOE Put/Call", "收盤當天", "D 收盤即公開（CBOE 統計當晚發布）"),
        ("FRED 日資料（殖利率、BAA 利差）", "觀測日 + 1 個交易日", "H.15 隔日才公布"),
        ("AAII 散戶調查", "報告日（週四）", "週四早上公布；檔案日期就是公布日"),
        ("NAAIM 曝險指數", "調查日（週三）+ 1 個交易日", "週三收件、週四公布"),
        ("CFTC COT", "部位日（週二）+ 6 天（下週一）", "正常週五 15:30 ET 公布；多留一天。政府關門期間另行延後"),
        ("FINRA 融資餘額", "該月最後一天 + 1 個月（下月月底）", "通常下月第三週公布；取月底更保守"),
        ("初領失業金", "ALFRED 第一次公布日", "2009 年以前沒有 vintage：週六 + 5 天，值用目前版本（小幅修正偏差）"),
        ("非農、失業率、CPI", "ALFRED 第一次公布日", "第一次公布值"),
        ("SEC Form 4 內部人", "申報日 + 1 個交易日", "申報晚上 10 點前都可能進來"),
        ("Fed 資產負債表 / TGA", "週三觀測 + 1 個交易日", "週四下午公布"),
        ("Ken French 產業報酬", "收盤當天", "只用當天報酬，相當於即時可算的產業指數"),
    ]
    return pd.DataFrame(rows, columns=["資料", "公開日", "說明"])


# ---------------------------------------------------------------------------
# 交易日曆與對齊
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _raw_price(tk: str) -> pd.DataFrame:
    f = RAW / "prices" / f"{tk.replace('^', 'IDX_').replace('=', '_')}.csv"
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


@lru_cache(maxsize=None)
def calendar() -> pd.DatetimeIndex:
    """美股交易日：^GSPC 的交易日，加上 Ken French 資料裡有、但 Yahoo 沒有的日子。
    （1952 年以前週六也開盤，Yahoo 的 ^GSPC 沒有這些日子；少了會漏掉約 1,085 天的報酬。）"""
    g = _raw_price("^GSPC").index
    ff = french_factors().index
    return g.union(ff[ff < g[-1]]).sort_values()


def avail_to_daily(values: pd.Series, avail: pd.Series | pd.DatetimeIndex, cal: pd.DatetimeIndex | None = None,
                   ffill_limit: int | None = None) -> pd.Series:
    """把 (公開日 → 值) 轉成交易日序列：D 的值 = D 當天（含）以前最後一筆已公開的值。"""
    cal = calendar() if cal is None else cal
    v = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(pd.to_datetime(np.asarray(avail))))
    v = v[~v.index.isna()].sort_index()
    v = v.groupby(level=0).last()
    # 公開日落在非交易日時，對到下一個交易日
    pos = cal.searchsorted(v.index, side="left")
    ok = pos < len(cal)
    v = pd.Series(v.values[ok], index=cal[pos[ok]]).groupby(level=0).last()
    return v.reindex(cal).ffill(limit=ffill_limit)


def shift_bdays(dates, n: int, cal: pd.DatetimeIndex | None = None) -> pd.DatetimeIndex:
    """日期往後 n 個交易日（日期本身不是交易日時，先對到下一個交易日）。"""
    cal = calendar() if cal is None else cal
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    pos = cal.searchsorted(d, side="left") + n
    # 超過交易日曆最後一天的，回傳 NaT（還拿不到），不要擠到最後一天（那樣最後一天會用到之後才公布的資料）
    inside = pos < len(cal)
    return cal[np.clip(pos, 0, len(cal) - 1)].where(inside)


# ---------------------------------------------------------------------------
# 報酬序列
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def rf_daily() -> pd.Series:
    """現金的日報酬：3 個月國庫券（DTB3，年化 %）/ 252，用前一天的值（當天就知道）。"""
    s = pd.read_csv(RAW / "fred" / "DTB3.csv", index_col=0, parse_dates=True).iloc[:, 0]
    s = (s.reindex(calendar()).ffill() / 100.0 / 252.0).shift(1)
    # 1954 年以前沒有 DTB3：用 Ken French 的一個月國庫券日報酬
    ff = french_factors()["RF"].reindex(calendar()) / 100.0
    return s.fillna(ff).fillna(0.0)


@lru_cache(maxsize=None)
def total_return(name: str) -> pd.Series:
    """日總報酬。
    NDX: 1985-10 起 ^NDX 價格 + 0.7%/年股息，1999-03-10 起接 QQQ（還原股息）
    SPX: 1993-01-29 以前用 Ken French 美股市值加權總報酬（CRSP 全市場，接近 S&P 500），之後接 SPY（還原股息）
    其他：yfinance 的還原股息收盤價。
    """
    cal = calendar()
    if name == "NDX":
        ndx = _raw_price("^NDX")["Close"].pct_change() + NDX_DIV_YIELD_PRE_QQQ / 252
        qqq = _raw_price("QQQ")["Close"].pct_change()
        r = ndx.copy()
        r.loc[qqq.index[1]:] = np.nan
        r = r.combine_first(qqq.loc[qqq.index[1]:])
        return r.reindex(cal).dropna()
    if name == "SPX":
        ff = french_factors()
        mkt = (ff["Mkt-RF"] + ff["RF"]) / 100.0
        spy = _raw_price("SPY")["Close"].pct_change()
        r = mkt.loc[: spy.index[0]]
        r = pd.concat([r, spy.loc[spy.index[1]:]])
        r = r[~r.index.duplicated(keep="last")]
        return r.reindex(cal).dropna()
    if name == "QQQ":
        return _raw_price("QQQ")["Close"].pct_change().dropna()
    if name == "SPY":
        return _raw_price("SPY")["Close"].pct_change().dropna()
    return _raw_price(name)["Close"].pct_change().dropna()


def price_index(name: str) -> pd.Series:
    """由總報酬累乘出的指數（起點 = 1）。"""
    r = total_return(name)
    return (1 + r).cumprod()


def close(tk: str) -> pd.Series:
    return _raw_price(tk)["Close"].reindex(calendar())


# ---------------------------------------------------------------------------
# Ken French
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def french_factors() -> pd.DataFrame:
    f = RAW / "french" / "F-F_Research_Data_Factors_daily.csv"
    lines = f.read_text(encoding="latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    rows = []
    for l in lines[start + 1:]:
        p = l.split(",")
        if len(p) < 5 or not p[0].strip().isdigit():
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    df.index = pd.to_datetime(df.pop("date").str.strip(), format="%Y%m%d")
    return df.astype(float)


@lru_cache(maxsize=None)
def french_industries(n: int = 49, weight: str = "value") -> pd.DataFrame:
    """產業日報酬（小數）。-99.99 = 缺值。"""
    f = RAW / "french" / f"{n}_Industry_Portfolios_Daily.csv"
    lines = f.read_text(encoding="latin-1").splitlines()
    tag = "Average Value Weighted Returns -- Daily" if weight == "value" else "Average Equal Weighted Returns -- Daily"
    i0 = next(i for i, l in enumerate(lines) if tag in l)
    header = lines[i0 + 1].split(",")
    rows = []
    for l in lines[i0 + 2:]:
        p = l.split(",")
        if len(p) < 2 or not p[0].strip().isdigit():
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=["date"] + [h.strip() for h in header[1:]])
    df.index = pd.to_datetime(df.pop("date").str.strip(), format="%Y%m%d")
    df = df.astype(float).replace([-99.99, -999.0], np.nan) / 100.0
    return df


# ---------------------------------------------------------------------------
# CBOE 指數（VIX 家族、SKEW、隱含相關）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def cboe_index(name: str) -> pd.Series:
    """CBOE 官方歷史檔；VIX3M 的官方檔 2009-09 才開始，更早的部分用 yfinance 的 ^VIX3M 補。"""
    s = _cboe_index_file(name)
    if name in ("VIX3M", "VIX9D", "VVIX"):
        try:
            y = _raw_price("^" + name)["Close"].reindex(calendar())
            s = s.combine_first(y.loc[: s.first_valid_index()])
        except FileNotFoundError:
            pass
    return s


@lru_cache(maxsize=None)
def _cboe_index_file(name: str) -> pd.Series:
    df = pd.read_csv(RAW / "cboe" / f"{name}_History.csv")
    df.columns = [c.strip().upper() for c in df.columns]
    col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="mixed")).dropna()
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(calendar()).ffill(limit=3)


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def fred_raw(sid: str) -> pd.Series:
    s = pd.read_csv(RAW / "fred" / f"{sid}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return s.dropna().sort_index()


def fred_daily_avail(sid: str, lag_bdays: int = 1) -> pd.Series:
    """FRED 日資料：觀測日 + lag 個交易日才公開。"""
    s = fred_raw(sid)
    return avail_to_daily(s.values, shift_bdays(s.index, lag_bdays))


def fred_weekly_avail(sid: str, lag_days: int) -> pd.Series:
    s = fred_raw(sid)
    return avail_to_daily(s.values, s.index + pd.Timedelta(days=lag_days), ffill_limit=WEEKLY_STALE)


@lru_cache(maxsize=None)
def alfred_first(sid: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / "alfred" / f"{sid}.csv", parse_dates=["obs_date", "release_date"])
    return df.dropna(subset=["value"]).sort_values("obs_date")


def claims_avail() -> pd.Series:
    """初領失業金（週）。2009-06 起用第一次公布值；之前用目前版本 + 週六後 5 天公布。"""
    first = alfred_first("ICSA")
    cur = fred_raw("ICSA") if (RAW / "fred" / "ICSA.csv").exists() else None
    parts_v, parts_a = [first["value"].values], [first["release_date"].values]
    if cur is not None:
        early = cur[cur.index < first["obs_date"].min()]
        parts_v.append(early.values)
        parts_a.append((early.index + pd.Timedelta(days=5)).values)
    v = np.concatenate(parts_v[::-1])
    a = np.concatenate(parts_a[::-1])
    return avail_to_daily(v, a, ffill_limit=WEEKLY_STALE)


# ---------------------------------------------------------------------------
# 第三輪（PREREG_batch3_2026-10-25.md）：SLOOS、EBP、EPU
# ---------------------------------------------------------------------------
SLOOS_LAG_DAYS = 50   # 沒有 ALFRED 公布日的早期 SLOOS：觀測日 + 50 天（ALFRED 實際 29～46 天）
EBP_LAG_MONTHS = 2    # EBP：觀測月第一天 + 2 個月（實際約 1 個多月）
QUARTERLY_STALE = 100  # 季資料最多沿用 100 個交易日


def sloos_releases() -> pd.DataFrame:
    """SLOOS（DRTSCILM，銀行收緊工商放款標準的淨比例，季）每一次公布：value、prev（前一次公布值）、avail（公布日）。
    2010-08 起用 ALFRED 第一次公布的數字和公布日；更早的用目前版本、公布日 = 觀測日 + 50 天。"""
    first = alfred_first("DRTSCILM")
    cur = fred_raw("DRTSCILM")
    early = cur[cur.index < first["obs_date"].min()]
    df = pd.DataFrame({
        "obs": list(early.index) + list(first["obs_date"]),
        "value": list(early.values) + list(first["value"].values),
        "avail": list(early.index + pd.Timedelta(days=SLOOS_LAG_DAYS)) + list(first["release_date"]),
    }).sort_values("obs").reset_index(drop=True)
    df["prev"] = df["value"].shift(1)
    return df


def sloos_avail() -> pd.DataFrame:
    """每天「已經公布的最新一次」SLOOS 和前一次公布值。"""
    r = sloos_releases()
    return pd.DataFrame({c: avail_to_daily(r[c].values, r["avail"].values, ffill_limit=QUARTERLY_STALE)
                         for c in ("value", "prev")})


def ebp_monthly() -> pd.DataFrame:
    """Gilchrist-Zakrajšek 超額債券溢酬（月）。每次更新會重新估計整段歷史，沒有歷史版本（見事先登記的限制）。"""
    df = pd.read_csv(RAW / "fed" / "ebp.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["chg3"] = df["ebp"] - df["ebp"].shift(3)
    df["max6"] = df["ebp"].rolling(6, min_periods=6).max()
    df["avail"] = df["date"] + pd.DateOffset(months=EBP_LAG_MONTHS)
    return df


def ebp_avail(extra_months: int = 0) -> pd.DataFrame:
    """每天已經公布的最新 EBP、3 個月變化、6 個月最高點。extra_months：敏感度檢查用（多延後幾個月）。"""
    m = ebp_monthly()
    a = (m["avail"] + pd.DateOffset(months=extra_months)).values
    return pd.DataFrame({c: avail_to_daily(m[c].values, a, ffill_limit=MONTHLY_STALE) for c in ("ebp", "chg3", "max6")})


def epu30_avail(extra_bdays: int = 0) -> pd.Series:
    """EPU 每日指數（USEPUINDXD，含週末）的 30 個日曆天平均，觀測日 + 1 個交易日才算拿得到。
    extra_bdays：敏感度檢查用（多延後幾個交易日）。"""
    s = fred_raw("USEPUINDXD")
    m = s.rolling("30D", min_periods=20).mean().dropna()
    return avail_to_daily(m.values, shift_bdays(m.index, 1 + extra_bdays), ffill_limit=WEEKLY_STALE)


# ---------------------------------------------------------------------------
# 情緒與籌碼（AAII、NAAIM、FINRA、CBOE Put/Call、COT）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def aaii_raw() -> pd.DataFrame:
    a = pd.read_excel(RAW / "repo_cache" / "aaii_sentiment.xls", header=None).iloc[5:, :4]
    a.columns = ["date", "bull", "neutral", "bear"]
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a = a.dropna(subset=["date"]).set_index("date").apply(pd.to_numeric, errors="coerce").dropna()
    return a[~a.index.duplicated(keep="last")].sort_index()


def aaii_avail() -> pd.DataFrame:
    a = aaii_raw()
    out = pd.DataFrame({c: avail_to_daily(a[c].values, a.index, ffill_limit=WEEKLY_STALE) for c in a.columns})
    out["spread"] = out["bull"] - out["bear"]
    return out


@lru_cache(maxsize=None)
def naaim_raw() -> pd.Series:
    n = pd.read_excel(RAW / "repo_cache" / "naaim.xlsx", header=None).iloc[1:, :2]
    n.columns = ["date", "mean"]
    n["date"] = pd.to_datetime(n["date"], errors="coerce")
    n = n.dropna().set_index("date")["mean"].astype(float)
    return n[~n.index.duplicated(keep="last")].sort_index()


def naaim_avail() -> pd.Series:
    n = naaim_raw()
    return avail_to_daily(n.values, shift_bdays(n.index, 1), ffill_limit=WEEKLY_STALE)


@lru_cache(maxsize=None)
def finra_margin_raw() -> pd.DataFrame:
    f = pd.read_excel(RAW / "repo_cache" / "finra_margin.xlsx", header=None).iloc[1:, :4]
    f.columns = ["month", "debit", "free_cash", "free_margin"]
    f["month"] = pd.PeriodIndex(f["month"].astype(str).str[:7], freq="M")
    f = f.set_index("month").apply(pd.to_numeric, errors="coerce")
    return f.sort_index()


def finra_margin_avail() -> pd.DataFrame:
    f = finra_margin_raw()
    avail = (f.index + 1).to_timestamp(how="end").normalize()  # 下個月最後一天
    return pd.DataFrame({c: avail_to_daily(f[c].values, avail, ffill_limit=MONTHLY_STALE) for c in f.columns})


def _read_cboe_pc(path: Path) -> pd.Series:
    lines = path.read_text(encoding="latin-1").splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.upper().startswith(("DATE", "TRADE_DATE")))
    rows = [l.split(",") for l in lines[i0 + 1:] if l.strip()]
    d = pd.to_datetime([r[0].strip() for r in rows], format="mixed", errors="coerce")
    c = pd.to_numeric([r[1].strip() for r in rows], errors="coerce")
    p = pd.to_numeric([r[2].strip() for r in rows], errors="coerce")
    df = pd.DataFrame({"call": c, "put": p}, index=d).dropna()
    return df[~df.index.duplicated(keep="last")].sort_index()


@lru_cache(maxsize=None)
def cboe_putcall(kind: str = "equity") -> pd.Series:
    """CBOE Put/Call（以成交量重算）：2003-10 舊檔 → 2006-11 舊檔 → 2019-10 起 repo 的逐日資料。
    kind: equity / index / total。2012-06 起 equity 不含 ETP（水準有斷層，所以訊號一律用滾動百分位）。"""
    d = RAW / "cboe"
    parts = []
    if (d / f"{kind}pc_archive.csv").exists():
        parts.append(_read_cboe_pc(d / f"{kind}pc_archive.csv"))
    parts.append(_read_cboe_pc(d / f"{kind}pc.csv"))
    vo = pd.read_csv(RAW / "repo_cache" / "cboe_volume_oi_wide.csv", parse_dates=["date"])
    prod = {"equity": "EQUITY OPTIONS", "index": "INDEX OPTIONS", "total": "SUM OF ALL PRODUCTS"}[kind]
    v = vo[vo["product"] == prod].set_index("date")[["volume_call", "volume_put"]]
    v.columns = ["call", "put"]
    parts.append(v)
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["call"] > 0) & (df["put"] > 0)]
    pc = (df["put"] / df["call"]).rename(f"{kind}_pc")
    return pc.reindex(calendar())


# ---------------------------------------------------------------------------
# CFTC COT
# ---------------------------------------------------------------------------
# 政府關門時 COT 延後公布。保守做法：這些區間內的報告，公開日一律延到補發完成日之後
COT_SHUTDOWN_DELAYS = [
    # (部位日起, 部位日迄, 改成的公開日)。來源：CFTC 新聞稿 6745-13、2019-01 補發時程、2025-11/12 補發時程
    ("2013-10-01", "2013-10-29", "2013-11-08"),   # 2013 關門：10/25 起補發，11/8 恢復正常
    ("2018-12-24", "2019-02-26", "2019-03-08"),   # 2018-19 關門：2/1 起補發，3/8 恢復正常
    ("2025-09-30", "2025-12-23", "2025-12-30"),   # 2025 關門：11/19 起補發，12/29 補完
]


def _cot_avail_dates(report_dates: pd.Series) -> pd.DatetimeIndex:
    a = pd.DatetimeIndex(report_dates) + pd.Timedelta(days=6)  # 週二 → 下週一
    a = pd.Series(a, index=pd.DatetimeIndex(report_dates))
    for s, e, rel in COT_SHUTDOWN_DELAYS:
        m = (a.index >= s) & (a.index <= e)
        a[m] = a[m].clip(lower=pd.Timestamp(rel))
    return pd.DatetimeIndex(a.values)


@lru_cache(maxsize=None)
def cot_legacy_raw() -> pd.DataFrame:
    import zipfile
    frames = []
    for p in sorted((RAW / "cot").glob("deacot*.zip")):
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            df = pd.read_csv(z.open(n), low_memory=False)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["As of Date in Form YYYY-MM-DD"], format="mixed").dt.normalize()
    return df


@lru_cache(maxsize=None)
def cot_tff_raw() -> pd.DataFrame:
    import zipfile
    frames = []
    for p in sorted((RAW / "cot").glob("f*_fin_txt_*.zip")) + sorted((RAW / "cot").glob("fin_fut_txt_*.zip")):
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            frames.append(pd.read_csv(z.open(n), low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"], format="mixed").dt.normalize()
    return df.drop_duplicates(subset=["date", "CFTC_Contract_Market_Code"])


def cot_series(df: pd.DataFrame, codes: list[str], cols: list[str]) -> pd.DataFrame:
    """把幾個合約代碼（例如 E-mini 與大型合約）的欄位加總成一個週序列（以部位日為索引）。"""
    code_col = "CFTC Contract Market Code" if "CFTC Contract Market Code" in df.columns else "CFTC_Contract_Market_Code"
    sub = df[df[code_col].astype(str).str.strip().isin(codes)]
    g = sub.groupby("date")[cols].sum(min_count=1)
    return g.sort_index()


def cot_to_daily(weekly: pd.Series) -> pd.Series:
    return avail_to_daily(weekly.values, _cot_avail_dates(weekly.index), ffill_limit=WEEKLY_STALE + 30)
