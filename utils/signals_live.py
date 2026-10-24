"""
進出場訊號（依 research/REPORT.md 的結論，2026-10）。

研究結論（106 個變體、嚴格驗證）：
  - 沒有任何訊號通過驗證。最一致的是「全面恐慌」：VIX 收盤首次 ≥ 40 之後 3 個月通常比平常好，
    但只是觀察名單（多重檢定後不顯著）。
  - 最確定的是：每月存的錢不要留著等訊號，直接照計畫買。
  - 沒有出場訊號通過驗證。

所以這個模組只做「提醒和紀錄」，不是交易訊號：
  - capitulation_state()：正常 / 留意（VIX ≥ 30）/ 全面恐慌（今天 VIX 收盤首次站上 40）/
    加碼期間（觸發後 126 個交易日內）/ 資料過期 / 無法判斷
    加碼期間照研究的規則（research/run_robustness2.py 的 overlay）：VIX 收盤「剛站上」40 那天觸發，
    維持 126 個交易日；期間內再觸發就從新的觸發日重新起算（不是再買一次）。
  - 確認指標：9 檔產業 SPDR 同時創一年新低的檔數、CBOE 隱含相關性 COR3M、QQQ 距高點跌幅
  - 10 個月均線狀態：只當「有代價的保險」參考

資料：只用美股交易日、已經收盤的日線（依紐約時間判斷；盤中那根 K 線、CBOE 檔裡休市日的夜盤報價都丟掉）。
VIX 以 CBOE 官方檔為主，Yahoo ^VIX 有比較新的收盤時接在後面（CBOE 收盤後要好幾個小時才更新）。
其他來源只用到 VIX 的資料日為止。資料日落後「最近一個已收盤的美股交易日」超過 MAX_LAG_SESSIONS 個交易日，
就標成過期。

上半部是純函式（不需要網路、不 import streamlit，可以單元測試）；
gather_live() 才會抓資料（CBOE 官方 CSV 和 Yahoo ^VIX；ETF 和標的用 yfinance）。
"""
from __future__ import annotations

import io
import json
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EVIDENCE_PATH = Path(__file__).resolve().parent / "signal_evidence.json"

VIX_WATCH = 30.0           # 留意
VIX_PANIC = 40.0           # 全面恐慌（研究的主要觀察訊號）
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SECTOR_CONFIRM = 4         # ≥ 4 檔同時創一年新低 = 賣壓已全面化（只當確認）
COR3M_CONFIRM_PCT = 95.0   # COR3M 在一年 95 百分位以上（只當確認）
COOLDOWN = 21              # 兩次觸發至少相隔 21 個交易日才算新的一次（事件研究、歷史清單用）
HOLD_DAYS = 126            # 加碼期間：觸發日起 126 個交易日（約 6 個月），研究 overlay 的設定
MAX_LAG_SESSIONS = 2       # 資料日落後最近一個已收盤交易日超過 2 個交易日 = 過期
RECENT_END = 21            # 加碼期間結束後 21 個交易日內，提醒把股票比例調回平常

NY = ZoneInfo("America/New_York")
CLOSE_CUTOFF = time(16, 30)   # 美股 16:00、VIX 16:15 收盤；紐約時間 16:30 以後才把當天的日線當成已收盤

STATE_NORMAL, STATE_WATCH, STATE_PANIC = "正常", "留意", "全面恐慌"
STATE_HOLD = "加碼期間"
STATE_STALE = "資料過期"
STATE_UNKNOWN = "無法判斷"
ACTIVE_STATES = (STATE_PANIC, STATE_HOLD)        # 這兩種狀態下，股票目標比例 = 平常 + 加碼
UNKNOWN_STATES = (STATE_STALE, STATE_UNKNOWN)    # 這兩種狀態下不做任何判斷

SOURCE_KEYS = ("vix", "cor3m", "sectors", "px")
SOURCE_NAMES = {"vix": "VIX", "cor3m": "COR3M", "sectors": "產業 ETF", "px": None}   # px 用標的代號

MANUAL_RULE = (f"請自己查 VIX 收盤：最近 {HOLD_DAYS} 個交易日內如果有「收盤首次站上 40」（≥ 40 且前一天 < 40），"
               "就是加碼期間，照你的規則加到目標比例；沒有就不動。")

VIX_CALM = 20.0           # 第 1 級和第 2 級的分界（只是描述用的門檻，研究沒有拿它當訊號）
# 目前等級（第 8 頁最上方）。門檻要和 research/export_evidence.py 的 vix_levels() 一致。
LEVELS = {
    1: {"name": "平靜", "short": "平靜", "rule": "VIX < 20",
        "action": "新錢照你原本的比例投入；現金桶不動"},
    2: {"name": "正常波動", "short": "正常波動", "rule": "20 ≤ VIX < 30",
        "action": "新錢照你原本的比例投入；現金桶不動"},
    3: {"name": "緊張", "short": "緊張", "rule": "VIX ≥ 30（不在加碼期間）",
        "action": "新錢照你原本的比例投入；現金桶先不動（還不是加碼條件）"},
    4: {"name": "全面恐慌／加碼期間", "short": "加碼期間", "rule": "VIX 收盤首次站上 40 後 126 個交易日內",
        "action": "新錢照常投入；現金桶可以依你的規則加碼到目標比例"},
}

# 第 8 頁「你的現金桶加碼規則」的參數：預設值與合法範圍（讀檔時超出範圍就截到範圍內）
DEFAULT_ENTRY_PARAMS = {"stock_value": 400_000, "cash": 600_000, "base_equity_pct": 40, "tilt_pp": 10,
                        "cash_bucket": 200_000, "hold_months": 6, "holdings_date": None}
_MONEY_MAX = 10 ** 13     # 金額上限（避免壞掉的參數檔把超大數字塞進輸入框）
ENTRY_PARAM_LIMITS = {"stock_value": (0, _MONEY_MAX), "cash": (0, _MONEY_MAX), "base_equity_pct": (0, 100),
                      "tilt_pp": (0, 50), "cash_bucket": (0, _MONEY_MAX), "hold_months": (1, 36)}


def _now() -> datetime:
    """現在時間（紐約）。測試時可以換掉。"""
    return datetime.now(NY)


# ---------------------------------------------------------------------------
# 美股交易日
# ---------------------------------------------------------------------------
# 1985 年以後的臨時休市（颶風、911、國喪日）
SPECIAL_CLOSURES = frozenset(date.fromisoformat(d) for d in (
    "1985-09-27", "1994-04-27", "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14", "2004-06-11",
    "2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"))


def _observed(d: date) -> date:
    """固定日期的假日遇到週末：週六 → 前一個週五、週日 → 下一個週一。"""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """某月第 n 個星期幾（n = -1 是最後一個）。weekday：星期一 = 0。"""
    if n > 0:
        d = date(year, month, 1)
        return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


@lru_cache(maxsize=None)
def nyse_holidays(year: int) -> frozenset:
    """紐約證交所的固定休市日（臨時休市另外列在 SPECIAL_CLOSURES）。"""
    from dateutil.easter import easter
    h = {
        _nth_weekday(year, 2, 0, 3),            # 總統日
        easter(year) - timedelta(days=2),       # 耶穌受難日
        _nth_weekday(year, 5, 0, -1),           # 陣亡將士紀念日
        _observed(date(year, 7, 4)),            # 獨立紀念日
        _nth_weekday(year, 9, 0, 1),            # 勞動節
        _nth_weekday(year, 11, 3, 4),           # 感恩節
        _observed(date(year, 12, 25)),          # 聖誕節
    }
    if year >= 1998:
        h.add(_nth_weekday(year, 1, 0, 3))      # 馬丁路德金恩紀念日（1998 年起休市）
    if date(year, 1, 1).weekday() != 5:         # 元旦遇到週六時，前一個週五照常開市
        h.add(_observed(date(year, 1, 1)))
    if year >= 2022:
        h.add(_observed(date(year, 6, 19)))     # 六月節
    return frozenset(h)


def is_session(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year) and d not in SPECIAL_CLOSURES


def last_closed_session(now: datetime | None = None) -> date:
    """最近一個已經收盤的美股交易日（紐約時間 16:30 以後，當天才算收盤）。"""
    now = (now or _now()).astimezone(NY)
    d = now.date()
    if is_session(d) and now.time() >= CLOSE_CUTOFF:
        return d
    d -= timedelta(days=1)
    while not is_session(d):
        d -= timedelta(days=1)
    return d


def sessions_after(d: date, until: date) -> int:
    """d 之後到 until（含）有幾個美股交易日；d ≥ until 時是 0。"""
    n, x = 0, d + timedelta(days=1)
    while x <= until:
        n += is_session(x)
        x += timedelta(days=1)
    return n


def add_sessions(d: date, n: int) -> date:
    """d 之後第 n 個美股交易日（只用來估計日期）。"""
    while n > 0:
        d += timedelta(days=1)
        n -= is_session(d)
    return d


def closed_only(obj, last_session: date):
    """只留 last_session（含）以前、美股有開盤的日線：盤中還沒收完的那根、時區造成的「未來」日期、
    休市日的報價（CBOE 檔裡有休市日的夜盤數字）都丟掉。"""
    if obj is None:
        return None
    o = obj.copy()
    ix = pd.DatetimeIndex(pd.to_datetime(o.index))
    if ix.tz is not None:
        ix = ix.tz_convert(NY).tz_localize(None)
    o.index = ix.normalize()
    o = o[o.index <= pd.Timestamp(last_session)]
    o = o[~o.index.duplicated(keep="last")].sort_index()
    o = o[np.array([is_session(t.date()) for t in o.index], dtype=bool)]
    o = o.dropna(how="all") if isinstance(o, pd.DataFrame) else o.dropna()
    return o if len(o) else None


def align_live(raw: dict, now: datetime | None = None, symbol: str = "QQQ") -> tuple[dict, dict]:
    """把各來源對齊到同一個資料日。

    1. 每個來源只留已收盤、美股有開盤的日線（closed_only）。
    2. VIX：CBOE 為主；Yahoo ^VIX（raw["vix_yahoo"]）有比較新的收盤時接在後面；CBOE 沒資料時整段用 Yahoo。
    3. 資料日 = VIX 的最後一天；其他來源只用到這一天為止（不會出現「VIX 是週五、ETF 是週一盤中」的混用）。
       VIX 過期時不拿它當基準（不把新的資料往回截），改用沒過期的來源裡最晚的一天。
    4. 每個來源「自己」的最後一天落後最近一個已收盤的美股交易日超過 MAX_LAG_SESSIONS 個交易日 → 過期。
    回傳 (對齊後的資料, freshness)；freshness["sources"][key] = {label, date, lag, stale, missing}。
    """
    expected = last_closed_session(now)
    clean = {k: closed_only(raw.get(k), expected) for k in SOURCE_KEYS}
    notes: list[str] = []
    y = closed_only(raw.get("vix_yahoo"), expected)
    via = "CBOE" if clean["vix"] is not None else None
    if clean["vix"] is None and y is not None:
        clean["vix"], via = y, "Yahoo"
        if raw.get("vix") is not None:
            notes.append("CBOE 的 VIX 檔沒有可用的已收盤資料，改用 Yahoo ^VIX")
    elif clean["vix"] is not None and y is not None and y.index[-1] > clean["vix"].index[-1]:
        newer = y[y.index > clean["vix"].index[-1]]
        clean["vix"], via = pd.concat([clean["vix"], newer]), "CBOE+Yahoo"
        notes.append(f"VIX 最近 {len(newer)} 天（{newer.index[0].date()} 起）用 Yahoo ^VIX 的收盤；CBOE 官方檔還沒更新")
    lag = {k: None if v is None else sessions_after(v.index[-1].date(), expected) for k, v in clean.items()}
    stale = {k: lag[k] is not None and lag[k] > MAX_LAG_SESSIONS for k in SOURCE_KEYS}
    if clean["vix"] is not None and not stale["vix"]:
        asof = clean["vix"].index[-1].date()
    else:
        ends = [v.index[-1].date() for k, v in clean.items() if v is not None and not stale[k]] or \
               [v.index[-1].date() for v in clean.values() if v is not None]
        asof = max(ends) if ends else None
    aligned, sources = {}, {}
    for k in SOURCE_KEYS:
        v = clean[k]
        if v is not None and asof is not None:
            v = v[v.index <= pd.Timestamp(asof)]
            v = v if len(v) else None
        aligned[k] = v
        label = SOURCE_NAMES.get(k) or symbol
        if v is None:
            sources[k] = {"label": label, "date": None, "lag": None, "stale": False, "missing": True}
        else:
            sources[k] = {"label": label, "date": v.index[-1].date().isoformat(), "lag": lag[k],
                          "stale": stale[k], "missing": False}
    sources["vix"]["via"] = via
    return aligned, {"expected": expected.isoformat(), "asof": asof.isoformat() if asof else None,
                     "max_lag": MAX_LAG_SESSIONS, "sources": sources, "notes": notes}


# ---------------------------------------------------------------------------
# 純函式
# ---------------------------------------------------------------------------
def rising_edges(cond: pd.Series, cooldown: int = COOLDOWN) -> pd.DatetimeIndex:
    """條件「剛成立」的日子；同一段持續成立只算一次，且兩次之間至少隔 cooldown 個交易日。"""
    c = pd.Series(cond).fillna(False).astype(bool)
    edge = c & ~c.shift(1, fill_value=False)
    starts, last = [], -10 ** 9
    for i in np.flatnonzero(edge.values):
        if i - last >= cooldown:
            starts.append(i)
            last = i
    return c.index[starts]


def panic_window(vix: pd.Series, hold: int = HOLD_DAYS) -> dict:
    """研究的加碼期間（research/run_robustness2.py：edge(VIX ≥ 40).rolling(126).max()）。

    VIX 收盤「剛站上」40（前一天 < 40）那天觸發；觸發日起 hold 個交易日（含觸發日）都在加碼期間；
    期間內再觸發，就從新的觸發日重新起算。since 是這一段連續加碼期間最早的觸發日。
    不在期間內時，ended / ended_ago 是上一段加碼期間的最後一天、距今幾個交易日。
    vix 要先只留美股交易日（closed_only），研究也是把 VIX 對到交易日曆之後才算。
    """
    s = pd.Series(vix).dropna()
    if s.empty:
        return {"active": False}
    b = s >= VIX_PANIC
    e = np.flatnonzero((b & ~b.shift(1, fill_value=False)).values)
    j = len(s) - 1
    if not len(e):
        return {"active": False}
    if j - e[-1] > hold - 1:
        end = int(e[-1] + hold - 1)
        return {"active": False, "ended": s.index[end].date().isoformat(), "ended_ago": int(j - end)}
    start = e[-1]
    for i in e[::-1][1:]:
        if start - i > hold - 1:
            break
        start = i
    k = int(j - e[-1])
    since, trigger = s.index[start].date().isoformat(), s.index[e[-1]].date().isoformat()
    return {"active": True, "trigger": trigger, "since": since, "day": k + 1, "remaining": hold - 1 - k,
            "today": bool(k == 0), "retrigger_today": bool(k == 0 and since != trigger),
            "end_est": add_sessions(s.index[j].date(), hold - 1 - k).isoformat()}


def vix_status(vix: pd.Series) -> dict:
    s = pd.Series(vix).dropna()
    if s.empty:
        return {"value": None, "date": None, "level": None, "window": {"active": False}}
    v = float(s.iloc[-1])
    level = STATE_PANIC if v >= VIX_PANIC else STATE_WATCH if v >= VIX_WATCH else STATE_NORMAL
    edges = rising_edges(s >= VIX_PANIC)
    last = edges[-1] if len(edges) else None
    return {
        "value": v, "date": s.index[-1].date().isoformat(), "level": level,
        "to_watch": max(VIX_WATCH - v, 0.0), "to_panic": max(VIX_PANIC - v, 0.0),
        "max_1y": float(s.iloc[-252:].max()),
        "last_panic": last.date().isoformat() if last is not None else None,
        "days_since_panic": int((s.index[-1] - last).days) if last is not None else None,
        "window": panic_window(s),
    }


def _fill_gaps(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """中間偶爾缺幾天的收盤價用前一天補（最多 limit 天）；最後面缺的不補，算缺資料。"""
    return df.ffill(limit=limit).where(df.bfill().notna())


def sectors_at_low(closes: pd.DataFrame, window: int = 252) -> pd.Series:
    """每天有幾檔產業 ETF 收在「過去 window 個交易日（含今天）」的最低點。"""
    df = _fill_gaps(pd.DataFrame(closes).dropna(how="all"))
    lo = df.rolling(window, min_periods=window).min()
    at_low = (df <= lo) & lo.notna()
    return at_low.sum(axis=1).where(lo.notna().any(axis=1))


def sector_status(closes: pd.DataFrame | None) -> dict:
    """只看最後一天：分母是最後一天有收盤價、而且有滿一年資料的 ETF；缺的列在 missing。"""
    if closes is None or len(closes) == 0:
        return {"count": None, "of": 0, "missing": list(SECTOR_ETFS), "confirm": False}
    df = _fill_gaps(pd.DataFrame(closes).dropna(how="all"))
    lo = df.rolling(252, min_periods=252).min()
    last, lo_last = df.iloc[-1], lo.iloc[-1]
    valid = [c for c in df.columns if pd.notna(last[c]) and pd.notna(lo_last[c])]
    missing = [t for t in SECTOR_ETFS if t not in valid]
    if not valid:
        return {"count": None, "of": 0, "missing": missing, "confirm": False}
    names = [c for c in valid if last[c] <= lo_last[c]]
    hist = sectors_at_low(df).dropna()
    edges = rising_edges(hist >= SECTOR_CONFIRM)
    return {"count": len(names), "of": len(valid), "missing": missing, "confirm": len(names) >= SECTOR_CONFIRM,
            "date": df.index[-1].date().isoformat(),
            "max_60d": int(max(hist.iloc[-60:].max(), len(names))) if len(hist) else len(names),
            "names": names, "last_confirm": edges[-1].date().isoformat() if len(edges) else None}


def cor3m_status(cor: pd.Series | None, window: int = 252) -> dict:
    s = pd.Series(cor).dropna() if cor is not None else pd.Series(dtype=float)
    if len(s) < 60:
        return {"value": None, "pct": None, "confirm": False}
    w = s.iloc[-window:]
    pct = float((w <= w.iloc[-1]).mean() * 100)
    return {"value": float(s.iloc[-1]), "pct": pct, "confirm": pct >= COR3M_CONFIRM_PCT,
            "date": s.index[-1].date().isoformat()}


def drawdown_status(px: pd.Series | None) -> dict:
    s = pd.Series(px).dropna() if px is not None else pd.Series(dtype=float)
    if s.empty:
        return {"from_high": None}
    hi = s.cummax()
    return {"from_high": float(s.iloc[-1] / hi.iloc[-1] - 1), "high_date": s.idxmax().date().isoformat(),
            "date": s.index[-1].date().isoformat()}


def ten_month_trend(px: pd.Series | None) -> dict:
    """10 個月均線（每月底檢查一次）。研究結論：不是 alpha，是「有代價的保險」。
    最後一根 K 線是當月最後一個交易日時，當月就算收完（月底當天晚上就能看到這個月的結果）。"""
    s = pd.Series(px).dropna() if px is not None else pd.Series(dtype=float)
    if len(s) < 260:
        return {"above": None}
    m = s.groupby(s.index.to_period("M")).last()
    last = s.index[-1].date()
    complete = m if add_sessions(last, 1).month != last.month else m.iloc[:-1]
    if len(complete) < 10:
        return {"above": None}
    sma = complete.rolling(10).mean()
    last_close, last_sma = float(complete.iloc[-1]), float(sma.iloc[-1])
    return {"month": str(complete.index[-1]), "month_close": last_close, "sma10": last_sma,
            "above": last_close > last_sma, "gap": last_close / last_sma - 1,
            "current_vs_sma": float(s.iloc[-1]) / last_sma - 1}


def capitulation_state(vix: pd.Series, sector_closes: pd.DataFrame | None = None,
                       cor3m: pd.Series | None = None, px: pd.Series | None = None,
                       freshness: dict | None = None) -> dict:
    """主要依據只有 VIX；其他指標只是「賣壓是否全面化」的確認，不單獨觸發。

    freshness（align_live 的結果）裡標成過期的來源：確認指標過期 → 不算確認；
    VIX 過期 → 「資料過期」，但如果過期前的資料已經證明現在一定還在加碼期間（剩下的天數 ≥ 落後的天數；
    之後再觸發只會延長期間），就照樣顯示「加碼期間」並標出 VIX 過期。
    """
    vs = vix_status(vix)
    ss = sector_status(sector_closes)
    cs = cor3m_status(cor3m)
    ds = drawdown_status(px)
    fr = freshness or {}
    src = fr.get("sources", {})
    for key, st_ in (("sectors", ss), ("cor3m", cs), ("px", ds)):
        if src.get(key, {}).get("stale"):
            st_["stale"] = True
            if "confirm" in st_:
                st_["confirm"] = False
    if fr.get("expected") and vs.get("last_panic"):
        vs["days_since_panic"] = (date.fromisoformat(fr["expected"]) - date.fromisoformat(vs["last_panic"])).days
    w = vs["window"]
    vsrc = src.get("vix", {})
    vix_stale = bool(vsrc.get("stale"))
    if vs["value"] is None:
        state = STATE_UNKNOWN
    elif vix_stale:
        lag = vsrc.get("lag") or 0
        if w["active"] and w["remaining"] >= lag:
            state = STATE_HOLD
            w["stale_lag"], w["remaining_min"] = lag, w["remaining"] - lag
        else:
            state = STATE_STALE
    elif w["active"]:
        state = STATE_PANIC if w["today"] and not w["retrigger_today"] else STATE_HOLD
    else:
        state = STATE_WATCH if vs["value"] >= VIX_WATCH else STATE_NORMAL
    confirms = [name for name, ok in (("產業 ETF 同時新低", ss["confirm"]), ("隱含相關性高檔", cs["confirm"])) if ok]
    return {"state": state, "vix": vs, "vix_stale": vix_stale, "sectors": ss, "cor3m": cs, "drawdown": ds,
            "confirmations": confirms, "trend10m": ten_month_trend(px)}


def overweight_plan(state: str | None, stock_value: float, cash: float, base_equity_pct: float, tilt_pp: float,
                    cash_bucket: float, hold_months: int) -> dict:
    """依「目前的」股票市值和現金，算加碼期間要買多少、買完股票比例實際是多少。只做算術，不做判斷。

    目標比例 = 平常比例 + 加碼百分點（研究：平常 50% → 加碼期間 70%）。恐慌時股價已經跌了，
    股票比例通常已經低於平常，所以要買的金額 = 補回平常比例 + 加碼，再受現金桶（和手上現金）限制。
    cash_bucket 是「現金桶還剩多少」：同一段加碼期間買過之後要調低，才不會加超過自己訂的上限。
    """
    s, c = max(float(stock_value), 0.0), max(float(cash), 0.0)
    total = s + c
    target = min(max(base_equity_pct, 0) + max(tilt_pp, 0), 100)
    current = s / total * 100 if total > 0 else None
    want = max(target / 100 * total - s, 0.0)
    to_base = max(min(base_equity_pct, 100) / 100 * total - s, 0.0)
    cap = min(max(float(cash_bucket), 0.0), c)
    amount = min(want, cap)
    after = (s + amount) / total * 100 if total > 0 else None
    active = state in ACTIVE_STATES
    return {"active": active, "known": state is not None and state not in UNKNOWN_STATES,
            "amount": amount if active else 0.0, "amount_if_triggered": amount, "want": want,
            "to_base": min(to_base, amount), "limited_by_cash": after is not None and target - after >= 0.05,
            "at_target": current is not None and current >= target - 0.05,
            "current_pct": current, "target_pct": target, "after_pct": after, "total": total,
            "hold_months": hold_months}


def clean_entry_params(raw) -> dict:
    """讀進來的參數（檔案或 session）整理成合法的值：型別不對就用預設，超出範圍就截到範圍內。
    舊版參數檔只有 total_assets：照平常比例拆成股票和現金，日期留空（頁面會提醒更新成今天的市值）。"""
    out = dict(DEFAULT_ENTRY_PARAMS)
    if not isinstance(raw, dict):
        return out
    raw = dict(raw)
    if "stock_value" not in raw and "total_assets" in raw:
        try:
            tot = max(float(raw["total_assets"]), 0.0)
            pct = min(max(float(raw.get("base_equity_pct", out["base_equity_pct"])), 0.0), 100.0)
            raw["stock_value"], raw["cash"] = tot * pct / 100, tot * (1 - pct / 100)
        except (TypeError, ValueError, OverflowError):
            pass
    for k, (lo, hi) in ENTRY_PARAM_LIMITS.items():
        try:
            v = int(float(raw[k]))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        out[k] = min(max(v, lo), hi)
    d = raw.get("holdings_date")
    try:
        d = date.fromisoformat(d) if isinstance(d, str) else None
        out["holdings_date"] = d.isoformat() if d is not None and d <= date.today() else None
    except ValueError:
        out["holdings_date"] = None
    return out


def load_evidence(path: Path | str = EVIDENCE_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def _vix_txt(v: float) -> str:
    """VIX 數字：接近 40 時多顯示一位，避免 39.97 顯示成 40.0。"""
    return f"{v:.2f}" if VIX_PANIC - 0.1 < v < VIX_PANIC else f"{v:.1f}"


def source_dates_text(res: dict) -> str:
    """「VIX 2026-10-20、COR3M 2026-10-20、產業 ETF 2026-10-20、QQQ 2026-10-20」，過期或抓不到的標出來。"""
    fr = res.get("freshness") or {}
    parts = []
    for k in SOURCE_KEYS:
        s = (fr.get("sources") or {}).get(k)
        if not s:
            continue
        if s["missing"]:
            parts.append(f"{s['label']} 抓不到")
            continue
        via = {"Yahoo": "，Yahoo", "CBOE+Yahoo": "，最新幾天來自 Yahoo"}.get(s.get("via"), "")
        parts.append(f"{s['label']} {s['date']}" + (f"（過期，落後 {s['lag']} 個交易日）" if s["stale"]
                                                    else f"（{via[1:]}）" if via else ""))
    return "、".join(parts)


def _lag_note(res: dict) -> str:
    """VIX 落後最近一個已收盤交易日 1～2 天（還不算過期）時，燈號的說明要講清楚：今天的收盤還沒進來。"""
    fr = res.get("freshness") or {}
    s = (fr.get("sources") or {}).get("vix") or {}
    if s.get("lag") and not s.get("stale"):
        return f"（VIX 還沒有 {fr['expected']} 的收盤，這是 {s['date']} 的判斷；晚一點再看一次）"
    return ""


def state_headline(res: dict) -> str:
    """燈號下面那一句說明（頁面和簡報共用）。"""
    vs, state = res["vix"], res["state"]
    w = vs.get("window") or {}
    if state == STATE_UNKNOWN:
        return f"抓不到 VIX，無法判斷。{MANUAL_RULE}"
    if state == STATE_STALE:
        s = ((res.get("freshness") or {}).get("sources") or {}).get("vix", {})
        txt = (f"VIX 資料停在 {s.get('date')}，最近一個已收盤的美股交易日是 {res['freshness']['expected']}"
               f"（落後 {s.get('lag')} 個交易日），無法判斷。")
        if w.get("active"):
            txt += f"資料停住時還在加碼期間（約到 {w['end_est']}），現在可能已經結束。"
        return txt + MANUAL_RULE
    if state == STATE_PANIC:
        return (f"VIX {_vix_txt(vs['value'])}：今天收盤首次站上 {VIX_PANIC:.0f}，加碼期間開始。"
                f"依你的規則從現金桶加碼，之後維持 {HOLD_DAYS} 個交易日（約到 {w['end_est']}），不要因為反彈就賣。")
    if state == STATE_HOLD:
        keep = "已經加碼就維持，不要再加、也不要因為反彈賣；還沒加碼就照規則補上。"
        if w.get("stale_lag"):
            return (f"VIX 資料停在 {vs['date']}（落後 {w['stale_lag']} 個交易日），但那時加碼期間還剩 {w['remaining']} 個"
                    f"交易日，所以現在一定還在加碼期間（觸發日 {w['trigger']}，至少還剩 {w['remaining_min']} 個交易日，"
                    f"至少到約 {w['end_est']}）。" + keep)
        if w.get("retrigger_today"):
            return (f"今天 VIX 收盤再次站上 {VIX_PANIC:.0f}（{_vix_txt(vs['value'])}）：加碼期間從今天重新起算 "
                    f"{HOLD_DAYS} 個交易日（約到 {w['end_est']}），這段期間從 {w['since']} 開始。"
                    "這不是再買一次：" + keep)
        again = f"（這段期間從 {w['since']} 開始，之後又再觸發）" if w["since"] != w["trigger"] else ""
        return (f"加碼期間中：觸發日 {w['trigger']}{again}，還剩 {w['remaining']} 個交易日（約到 {w['end_est']}）。"
                f"目前 VIX {_vix_txt(vs['value'])}。" + keep
                + f"期間內 VIX 再站上 {VIX_PANIC:.0f} 只會重新起算期間，不是再買一次。")
    v = vs["value"]
    if v >= VIX_PANIC:
        txt = (f"VIX {_vix_txt(v)}：在 {VIX_PANIC:.0f} 以上，但不是新的觸發"
               + (f"（這段的加碼期間已在 {w['ended']} 結束，照手冊把股票比例調回平常的比例）" if w.get("ended") else "")
               + f"。現金桶不動；VIX 先回到 {VIX_PANIC:.0f} 以下、再收盤站上 {VIX_PANIC:.0f} 才是新的觸發。")
    else:
        gap = vs["to_panic"]
        txt = (f"VIX {_vix_txt(v)}，距「全面恐慌」{VIX_PANIC:.0f} 還差 {gap:.2f} 點" if gap < 0.1 else
               f"VIX {_vix_txt(v)}，距「全面恐慌」{VIX_PANIC:.0f} 還差 {gap:.1f} 點")
        txt += "（已達「留意」：先不動）" if state == STATE_WATCH else ""
        if w.get("ended") and w.get("ended_ago", 10 ** 6) <= RECENT_END:
            txt += f"。加碼期間已在 {w['ended']} 結束：照手冊把股票比例調回平常的比例"
    return txt + _lag_note(res)


def market_level(res: dict) -> dict:
    """依目前的資料分級（1～4 級，數字越大，歷史上之後 3 個月的表現越好）。資料有問題時是 0 級（無法分級）。

    第 4 級 = 全面恐慌當天或加碼期間；第 3 級 = VIX ≥ 30 但不在加碼期間（包括期間結束後 VIX 仍 ≥ 40）；
    第 2 級 = 20 ≤ VIX < 30；第 1 級 = VIX < 20。只看 VIX，因為研究裡其他指標都沒有一致的證據。
    """
    state, v = res["state"], res["vix"].get("value")
    if state in UNKNOWN_STATES or v is None:
        return {"level": 0, "name": "無法分級", "rule": "VIX 抓不到或過期", "action": "先自己確認 VIX"}
    k = 4 if state in ACTIVE_STATES else 3 if v >= VIX_WATCH else 2 if v >= VIX_CALM else 1
    return {"level": k, **LEVELS[k]}


def level_history(evidence: dict, level: int) -> dict | None:
    """證據檔裡這個等級的歷史統計（之後 3 個月 NDX 的表現）；沒有就回傳 None。"""
    for row in ((evidence or {}).get("vix_levels") or {}).get("levels", []):
        if row.get("level") == level:
            return row
    return None


def plain_summary(res: dict) -> dict:
    """第 8 頁最上方的白話結論：目前是第幾級、照「你手冊寫的規則」該做什麼、歷史上這個等級之後的表現。
    只是分級和檢查你自己訂的規則，不判斷你該不該買、買多少。回傳 {"level", "title", "body", "history"}。"""
    state, vs = res["state"], res["vix"]
    w = vs.get("window") or {}
    v = vs.get("value")
    lv = market_level(res)
    no_sell = "不因為指標或反彈賣出"
    if lv["level"] == 0:
        title = "無法分級：資料有問題"
        body = "請先看最上方的紅色提示，或自己查 VIX 收盤。新錢照你原本的計畫；現金桶要等確認 VIX 之後再決定。"
    else:
        title = f"第 {lv['level']} 級（共 4 級）：{lv['name']}"
        if state == STATE_PANIC:
            lead = f"今天 VIX 收盤首次站上 {VIX_PANIC:.0f}，達到你手冊寫的加碼條件。"
            rule = f"可以從現金桶加碼到目標比例（金額看下方計算器），之後至少持有、{no_sell}；新錢照常投入。"
        elif state == STATE_HOLD:
            left = (f"至少還剩 {w['remaining_min']} 個交易日" if w.get("stale_lag")
                    else f"還剩 {w['remaining']} 個交易日")
            lead = (("今天 VIX 再次站上 40，加碼期間重新起算，這不是再買一次。" if w.get("retrigger_today") else "")
                    + f"加碼期間中（觸發日 {w['trigger']}，{left}）。")
            rule = f"已經加碼就維持、不要再加；還沒加碼可以照規則補上（金額看下方計算器）；新錢照常投入；{no_sell}。"
        elif v >= VIX_PANIC:
            lead = (f"VIX {_vix_txt(v)} 在 40 以上，但上一段加碼期間已經結束，不是新的加碼條件"
                    f"（要先回到 {VIX_PANIC:.0f} 以下、再收盤站上 {VIX_PANIC:.0f}）。")
            rule = f"新錢照你原本的比例投入；現金桶不動；{no_sell}。"
        else:
            lead = f"VIX {_vix_txt(v)}（{lv['rule']}）。"
            rule = f"{lv['action']}；{no_sell}。"
        body = lead + "照你手冊寫的規則：" + rule
    if state not in ACTIVE_STATES and w.get("ended") and w.get("ended_ago", 10 ** 6) <= RECENT_END:
        body += f"另外，加碼期間已在 {w['ended']} 結束：手冊的做法是把股票比例調回平常的比例。"
    hist = level_history(res.get("evidence"), lv["level"]) if lv["level"] else None
    allh = ((res.get("evidence") or {}).get("vix_levels") or {}).get("all")
    history = None
    if hist and allh:
        history = (f"歷史上（1990 起）第 {lv['level']} 級之後 3 個月，NDX 平均 {_pct(hist['mean_63'])}、"
                   f"{_pct(hist['hit_63'], 0, False)} 的時候是正報酬、最差的 5% 是 {_pct(hist['p05_63'], 0)}；"
                   f"所有日子平均 {_pct(allh['mean_63'])}。")
    return {"level": lv["level"], "title": title, "body": body + _lag_note(res), "history": history}


def exit_summary(res: dict) -> dict:
    """第 8 頁「出場參考」：研究裡 40 個出場規則沒有一個通過驗證，所以這裡不給出場訊號，
    只整理三件事：照你手冊的規則該不該調回平常比例、10 個月均線（有代價的保險）在哪裡、不要用指標賣出。
    標題的優先順序：資料有問題 > 加碼期間剛結束 > 加碼期間中 > 10 個月均線跌破 > 沒事。回傳 {"title", "items", "footer"}。"""
    vs, tr = res["vix"], res.get("trend10m") or {}
    w = vs.get("window") or {}
    ev = res.get("evidence") or {}
    t10, er = ev.get("trend_10m_ndx") or {}, ev.get("exit_rules") or {}
    symbol = res.get("symbol", "QQQ")
    items = []
    recently_ended = bool(w.get("ended")) and w.get("ended_ago", 10 ** 6) <= RECENT_END
    if res["state"] in UNKNOWN_STATES:
        title = "資料有問題，先不做任何調整"
        items.append("加碼期間：VIX 抓不到或過期，無法確認，先看最上方的提示。")
    elif w.get("active"):
        title = "加碼期間中：先不調回"
        items.append(f"加碼期間約到 {w['end_est']} 結束（期間內 VIX 再站上 40 會延後）；結束後照手冊把股票比例調回平常。"
                     "期間內不要因為反彈賣出。")
    elif recently_ended:
        title = "加碼期間已結束：照手冊調回平常比例"
        items.append(f"加碼期間在 {w['ended']} 結束。這是你手冊裡唯一照規則的減碼：把股票比例調回平常的比例，"
                     "不是在預測行情。")
    else:
        title = "沒有要照規則減碼的部位"
        items.append("目前不在加碼期間，也沒有剛結束的加碼期間，所以沒有要調回的部位。")
    if tr.get("above") is not None:
        if tr["above"] is False and title == "沒有要照規則減碼的部位":
            title = "10 個月均線跌破：只有你選它當保險才出場"
        cost = ""
        if t10:
            cost = (f"歷史上（NDX {str(t10.get('start', ''))[:4]} 起）用它出場年化 {_pct(t10.get('cagr'), 1, False)}，"
                    f"一直持有 {_pct(t10.get('bh_cagr'), 1, False)}；最大回撤 {_pct(t10.get('maxdd'), 0)}，"
                    f"一直持有 {_pct(t10.get('bh_maxdd'), 0)}。")
        items.append(f"10 個月均線（{tr['month']} 月底）：{symbol} {'在均線之上' if tr['above'] else '跌破均線'}"
                     f"（{_pct(tr['gap'])}）。這是有代價的保險，不是訊號：{cost}"
                     "只有你事先決定要用它，才在月底收盤跌破時出場、站回時買回。")
    if er:
        vh = er.get("vix_high_riskoff") or {}
        items.append(f"研究測了 {er.get('n_tested')} 個出場規則，通過 {er.get('n_passed')} 個。不要因為情緒、籌碼、總經指標賣出"
                     + (f"；「VIX ≥ 30 就出場」每年比固定比例少 {_pct(abs(vh['alpha']), 1, False)}，是最差的之一。"
                        if vh.get("alpha") is not None else "。"))
    return {"title": title, "items": items, "footer": "這是研究結果和你自己規則的整理，不是賣出建議。"}


def format_brief(res: dict, symbol: str = "QQQ") -> str:
    """純文字簡報（給排程、通知或複製）。警告和資料過期放在最上面。"""
    vs, ss, cs, ds, tr = res["vix"], res["sectors"], res["cor3m"], res["drawdown"], res["trend10m"]
    ev = res.get("evidence") or {}
    v40 = ev.get("vix40") or {}
    state = res["state"]
    w = vs.get("window") or {}
    lines = [f"📊 {symbol} 進出場訊號簡報 — {res.get('as_of', date.today().isoformat())}"
             f"（資料日 {res.get('data_date') or vs.get('date') or '—'}）"]
    if state in UNKNOWN_STATES:
        lines.append(f"⚠️ {state}：{state_headline(res)}")
    for x in res.get("warnings", []):
        lines.append(f"⚠️ {x}")
    src = source_dates_text(res)
    if src:
        lines.append(f"各來源資料日：{src}")
    for x in res.get("notes", []):
        lines.append(f"註：{x}")
    lines.append("")
    lv = market_level(res)
    lv_txt = "無法分級" if lv["level"] == 0 else f"第 {lv['level']} 級（共 4 級）：{lv['name']}"
    lines.append(f"目前等級：{lv_txt}")
    if state not in UNKNOWN_STATES:
        lines.append(f"狀態：{state}（{state_headline(res)}）")
    sec = ("—" if ss.get("count") is None else f"{ss['count']}/{ss.get('of')} 檔"
           + (f"（缺 {'、'.join(ss['missing'])}）" if ss.get("missing") else ""))
    conf = [f"產業 ETF 一年新低 {sec}",
            f"COR3M 一年百分位 {cs['pct']:.0f}%" if cs.get("pct") is not None else "COR3M —",
            f"{symbol} 距高點 {_pct(ds.get('from_high'))}"]
    lines.append("確認指標：" + "；".join(conf)
                 + ("（賣壓已全面化）" if res.get("confirmations") else ""))
    lines.append("")
    lines.append("▶ 新存的錢：照計畫直接買（研究：留錢等訊號，1990–2026 一律輸給直接買）")
    if state in UNKNOWN_STATES:
        lines.append(f"▶ 現金桶：{state}。{MANUAL_RULE}")
    elif state == STATE_PANIC:
        lines.append("▶ 現金桶：全面恐慌 —— 依你事先寫好的比例從現金桶加碼；加碼後不要因為反彈就賣")
    elif state == STATE_HOLD:
        again = "今天再次站上 40，期間重新起算，不是再買一次；" if w.get("retrigger_today") else ""
        left = (f"至少還剩 {w['remaining_min']} 個交易日，VIX 資料過期" if w.get("stale_lag")
                else f"還剩 {w['remaining']} 個交易日")
        lines.append(f"▶ 現金桶：加碼期間中（觸發日 {w['trigger']}，{left}）—— "
                     f"{again}已加碼就維持，不要再加也不要賣；還沒加碼就照規則補上")
    elif state == STATE_WATCH:
        lines.append(f"▶ 現金桶：留意（VIX ≥ {VIX_WATCH:.0f}）—— 先不動，VIX 收盤首次站上 {VIX_PANIC:.0f} 才是加碼條件"
                     + _lag_note(res))
    else:
        lines.append("▶ 現金桶：不動" + _lag_note(res))
    if state not in ACTIVE_STATES and w.get("ended") and w.get("ended_ago", 10 ** 6) <= RECENT_END:
        lines.append(f"▶ 加碼期間已在 {w['ended']} 結束：照手冊把股票比例調回平常的比例")
    if tr.get("above") is not None:
        lines.append(f"▶ 出場：沒有通過驗證的出場訊號，不因情緒指標賣出。"
                     f"10 個月均線（{tr['month']} 月底）：{'在均線之上' if tr['above'] else '跌破均線'}"
                     f"（{_pct(tr['gap'])}，只是保險參考）")
    else:
        lines.append("▶ 出場：沒有通過驗證的出場訊號，不因情緒指標賣出。")
    if vs.get("last_panic"):
        lines.append(f"上次全面恐慌：{vs['last_panic']}（{vs['days_since_panic']} 天前）")
    if v40:
        q = v40.get("q")
        lines.append(f"歷史：VIX 首次 ≥ 40 共 {int(v40.get('n_events') or 0)} 次，之後 3 個月 NDX 平均 "
                     f"{_pct(v40.get('mean_63'))}（平常 {_pct(v40.get('base_63'))}）；"
                     f"多重檢定後不顯著（q = {q if q is None else f'{q:.2f}'}），屬於觀察名單")
    lines.append("")
    lines.append("（研究細節見 research/REPORT.md。非投資建議，實際買賣由你判斷。）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 抓資料（網路）
# ---------------------------------------------------------------------------
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def fetch_cboe_index(name: str, timeout: int = 20) -> pd.Series:
    """CBOE 官方指數歷史（VIX、COR3M…），回傳收盤價序列。回傳空檔或格式不對時丟例外（讓呼叫端改用備援）。"""
    import requests
    r = requests.get(CBOE_URL.format(name=name), headers=_UA, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().upper() for c in df.columns]
    col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="mixed")).dropna()
    s = s[s > 0]
    if s.empty:
        raise ValueError(f"CBOE {name} 回傳空的資料")
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_yahoo_close(tickers: list[str], period: str) -> pd.DataFrame:
    """yfinance 日線收盤（不經過 streamlit 快取）。整個沒有資料時丟例外；個別代號沒資料時那一欄是空的。"""
    import yfinance as yf
    df = yf.download(tickers, period=period, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance 沒有回傳 {'、'.join(tickers)}")
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.reindex(columns=tickers).dropna(how="all")
    if close.empty:
        raise RuntimeError(f"yfinance 沒有 {'、'.join(tickers)} 的收盤價")
    return close


def fetch_yahoo_vix(period: str = "10y") -> pd.Series:
    """Yahoo ^VIX 收盤，和 CBOE 一樣丟掉 ≤ 0 的數字；沒有可用的數字時丟例外。"""
    s = pd.to_numeric(fetch_yahoo_close(["^VIX"], period)["^VIX"], errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        raise ValueError("Yahoo ^VIX 沒有可用的收盤價")
    return s


def gather_live(symbol: str = "QQQ") -> tuple[dict, list[str]]:
    """抓即時資料（原始序列，還沒對齊）。任何一個來源失敗都只記警告，不讓整頁壞掉。
    VIX 同時抓 CBOE 官方檔和 Yahoo ^VIX：CBOE 為主，Yahoo 補最新幾天或在 CBOE 失敗時頂替（align_live 處理）。"""
    warnings: list[str] = []
    out: dict = {"vix": None, "vix_yahoo": None, "cor3m": None, "sectors": None, "px": None}
    cboe_err = yahoo_err = None
    try:
        out["vix"] = fetch_cboe_index("VIX")
    except Exception as e:  # noqa: BLE001
        cboe_err = e
    try:
        out["vix_yahoo"] = fetch_yahoo_vix("10y")
    except Exception as e:  # noqa: BLE001
        yahoo_err = e
    if out["vix"] is None:
        if out["vix_yahoo"] is not None:
            warnings.append(f"CBOE 的 VIX 抓不到（{type(cboe_err).__name__}），改用 Yahoo ^VIX（只用已收盤的日線）")
        else:
            warnings.append(f"VIX 抓不到（CBOE：{type(cboe_err).__name__}；Yahoo：{type(yahoo_err).__name__}），"
                            "無法判斷燈號")
    try:
        out["cor3m"] = fetch_cboe_index("COR3M")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"COR3M 抓不到（{type(e).__name__}），略過這個確認指標")
    try:
        out["sectors"] = fetch_yahoo_close(SECTOR_ETFS, "5y")
        missing = [t for t in SECTOR_ETFS if out["sectors"][t].dropna().empty]
        if missing:
            warnings.append(f"產業 ETF 缺少：{'、'.join(missing)}")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"產業 ETF 抓不到（{type(e).__name__}）")
    try:
        out["px"] = fetch_yahoo_close([symbol], "max")[symbol]
    except Exception as e:  # noqa: BLE001
        warnings.append(f"{symbol} 價格抓不到（{type(e).__name__}）")
    return out, warnings


def compute_live_signal(symbol: str = "QQQ", now: datetime | None = None) -> dict:
    now = now or _now()
    raw, warnings = gather_live(symbol)
    fetch_problems = bool(warnings)
    live, fresh = align_live(raw, now, symbol)
    vix = live["vix"] if live["vix"] is not None else pd.Series(dtype=float)
    res = capitulation_state(vix, live["sectors"], live["cor3m"], live["px"], freshness=fresh)
    src = fresh["sources"]
    stale = [s for s in src.values() if s["stale"]]
    missing = [s for s in src.values() if s["missing"]]
    if stale:
        warnings.insert(0, "資料過期：" + "、".join(f"{s['label']} 停在 {s['date']}（落後 {s['lag']} 個交易日）"
                                                  for s in stale)
                        + f"；最近一個已收盤的美股交易日是 {fresh['expected']}")
    vlag = src["vix"].get("lag") or 0
    if 0 < vlag and not src["vix"]["stale"]:
        warnings.append(f"VIX 還沒更新到 {fresh['expected']} 的收盤（目前到 {src['vix']['date']}）；晚一點再重新抓取")
    res["as_of"] = date.today().isoformat()
    res["data_date"] = fresh["asof"]
    res["freshness"] = fresh
    res["notes"] = fresh["notes"]
    res["symbol"] = symbol
    res["warnings"] = warnings
    # 有任何來源失敗、過期、缺檔，或 VIX 還沒更新到最近的收盤 → 頁面只把結果留 5 分鐘，不快取 1 小時
    res["incomplete"] = bool(stale or missing or fetch_problems or res["sectors"].get("missing") or vlag > 0)
    res["evidence"] = load_evidence()
    if not res["evidence"]:
        warnings.append("找不到 utils/signal_evidence.json（研究證據檔）；請執行 research/export_evidence.py")
    res["series"] = live          # 給頁面畫圖用；寫 JSON 前要拿掉
    return res
