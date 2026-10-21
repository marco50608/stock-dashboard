"""
進出場訊號（依 research/REPORT.md 的結論，2026-10）。

研究結論（106 個變體、嚴格驗證）：
  - 沒有任何訊號通過驗證。最一致的是「全面恐慌」：VIX 收盤 ≥ 40 之後 3 個月通常比平常好，
    但只是觀察名單（多重檢定後不顯著）。
  - 最確定的是：每月存的錢不要留著等訊號，直接照計畫買。
  - 沒有出場訊號通過驗證。

所以這個模組只做「提醒和紀錄」，不是交易訊號：
  - capitulation_state()：正常 / 留意（VIX ≥ 30）/ 全面恐慌（VIX ≥ 40）
  - 確認指標：9 檔產業 SPDR 同時創一年新低的檔數、CBOE 隱含相關性 COR3M、QQQ 距高點跌幅
  - 10 個月均線狀態：只當「有代價的保險」參考

上半部是純函式（不需要網路、不 import streamlit，可以單元測試）；
gather_live() 才會抓資料（CBOE 官方 CSV，失敗時改用 yfinance）。
"""
from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

EVIDENCE_PATH = Path(__file__).resolve().parent / "signal_evidence.json"

VIX_WATCH = 30.0           # 留意
VIX_PANIC = 40.0           # 全面恐慌（研究的主要觀察訊號）
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SECTOR_CONFIRM = 4         # ≥ 4 檔同時創一年新低 = 賣壓已全面化（只當確認）
COR3M_CONFIRM_PCT = 95.0   # COR3M 在一年 95 百分位以上（只當確認）
COOLDOWN = 21              # 兩次觸發至少相隔 21 個交易日才算新的一次

STATE_NORMAL, STATE_WATCH, STATE_PANIC = "正常", "留意", "全面恐慌"


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


def vix_status(vix: pd.Series) -> dict:
    s = pd.Series(vix).dropna()
    if s.empty:
        return {"value": None, "date": None, "level": None}
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
    }


def sectors_at_low(closes: pd.DataFrame, window: int = 252) -> pd.Series:
    """每天有幾檔產業 ETF 收在「過去 window 個交易日（含今天）」的最低點。"""
    df = pd.DataFrame(closes).dropna(how="all")
    lo = df.rolling(window, min_periods=window).min()
    at_low = (df <= lo) & lo.notna()
    return at_low.sum(axis=1).where(lo.notna().any(axis=1))


def sector_status(closes: pd.DataFrame | None) -> dict:
    if closes is None or len(closes) == 0:
        return {"count": None, "of": len(SECTOR_ETFS), "confirm": False}
    n = sectors_at_low(closes).dropna()
    if n.empty:
        return {"count": None, "of": closes.shape[1], "confirm": False}
    cnt = int(n.iloc[-1])
    edges = rising_edges(n >= SECTOR_CONFIRM)
    return {"count": cnt, "of": int(closes.shape[1]), "confirm": cnt >= SECTOR_CONFIRM,
            "date": n.index[-1].date().isoformat(), "max_60d": int(n.iloc[-60:].max()),
            "names": [c for c in closes.columns
                      if closes[c].dropna().size >= 252 and closes[c].iloc[-1] <= closes[c].iloc[-252:].min()],
            "last_confirm": edges[-1].date().isoformat() if len(edges) else None}


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
    """10 個月均線（每月底檢查一次）。研究結論：不是 alpha，是「有代價的保險」。"""
    s = pd.Series(px).dropna() if px is not None else pd.Series(dtype=float)
    if len(s) < 260:
        return {"above": None}
    m = s.groupby(s.index.to_period("M")).last()
    complete = m.iloc[:-1] if s.index[-1].to_period("M") == m.index[-1] else m   # 只用已收完的月份
    if len(complete) < 10:
        return {"above": None}
    sma = complete.rolling(10).mean()
    last_close, last_sma = float(complete.iloc[-1]), float(sma.iloc[-1])
    return {"month": str(complete.index[-1]), "month_close": last_close, "sma10": last_sma,
            "above": last_close > last_sma, "gap": last_close / last_sma - 1,
            "current_vs_sma": float(s.iloc[-1]) / last_sma - 1}


def capitulation_state(vix: pd.Series, sector_closes: pd.DataFrame | None = None,
                       cor3m: pd.Series | None = None, px: pd.Series | None = None) -> dict:
    """主要依據只有 VIX；其他指標只是「賣壓是否全面化」的確認，不單獨觸發。"""
    vs = vix_status(vix)
    ss = sector_status(sector_closes)
    cs = cor3m_status(cor3m)
    ds = drawdown_status(px)
    confirms = [name for name, ok in (("產業 ETF 同時新低", ss["confirm"]), ("隱含相關性高檔", cs["confirm"])) if ok]
    return {"state": vs["level"], "vix": vs, "sectors": ss, "cor3m": cs, "drawdown": ds,
            "confirmations": confirms, "trend10m": ten_month_trend(px)}


def overweight_plan(state: str | None, total_assets: float, base_equity_pct: float, tilt_pp: float,
                    cash_bucket: float, hold_months: int) -> dict:
    """全面恐慌時，從現金桶加碼多少。只做算術，不做判斷。"""
    want = max(total_assets, 0.0) * max(tilt_pp, 0.0) / 100.0
    amount = min(want, max(cash_bucket, 0.0))
    active = state == STATE_PANIC
    return {"active": active, "amount": amount if active else 0.0, "amount_if_triggered": amount,
            "limited_by_cash": want > cash_bucket, "target_equity_pct": base_equity_pct + tilt_pp,
            "hold_months": hold_months}


def load_evidence(path: Path | str = EVIDENCE_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def format_brief(res: dict, symbol: str = "QQQ") -> str:
    """純文字簡報（給排程、通知或複製）。"""
    vs, ss, cs, ds, tr = res["vix"], res["sectors"], res["cor3m"], res["drawdown"], res["trend10m"]
    ev = res.get("evidence") or {}
    v40 = ev.get("vix40") or {}
    lines = [f"📊 {symbol} 進出場訊號簡報 — {res.get('as_of', date.today().isoformat())}"
             f"（資料日 {vs.get('date') or '—'}）"]
    if vs.get("value") is None:
        lines.append("狀態：無法取得 VIX，請稍後再試")
    else:
        lines.append(f"狀態：{res['state']}（VIX {vs['value']:.1f}"
                     + (f"，距「全面恐慌」{VIX_PANIC:.0f} 還差 {vs['to_panic']:.1f} 點）" if vs["to_panic"] > 0 else "）"))
    conf = [f"產業 ETF 一年新低 {ss['count'] if ss.get('count') is not None else '—'}/{ss.get('of', 9)} 檔",
            f"COR3M 一年百分位 {cs['pct']:.0f}%" if cs.get("pct") is not None else "COR3M —",
            f"{symbol} 距高點 {_pct(ds.get('from_high'))}"]
    lines.append("確認指標：" + "；".join(conf)
                 + ("（賣壓已全面化）" if res.get("confirmations") else ""))
    lines.append("")
    lines.append("▶ 新存的錢：照計畫直接買（研究：留錢等訊號，1990–2026 一律輸給直接買）")
    if res["state"] == STATE_PANIC:
        lines.append("▶ 現金桶：全面恐慌 —— 可依你事先寫好的比例從現金桶加碼；加碼後不要因為反彈就賣")
    elif res["state"] == STATE_WATCH:
        lines.append(f"▶ 現金桶：留意（VIX ≥ {VIX_WATCH:.0f}）—— 先不動，VIX ≥ {VIX_PANIC:.0f} 才是加碼條件")
    else:
        lines.append("▶ 現金桶：不動")
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
    for w in res.get("warnings", []):
        lines.append(f"⚠️ {w}")
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
    """CBOE 官方指數歷史（VIX、COR3M…），回傳收盤價序列。"""
    import requests
    r = requests.get(CBOE_URL.format(name=name), headers=_UA, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().upper() for c in df.columns]
    col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="mixed")).dropna()
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def gather_live(symbol: str = "QQQ") -> tuple[dict, list[str]]:
    """抓即時資料。任何一個來源失敗都只記警告，不讓整頁壞掉。"""
    warnings: list[str] = []
    out: dict = {"vix": None, "cor3m": None, "sectors": None, "px": None}
    try:
        out["vix"] = fetch_cboe_index("VIX")
    except Exception as e:  # noqa: BLE001
        try:
            from utils import data as D
            out["vix"] = D.get_price("^VIX", period="5y")["Close"]
            warnings.append(f"CBOE VIX 抓不到，改用 Yahoo（{type(e).__name__}）")
        except Exception as e2:  # noqa: BLE001
            warnings.append(f"VIX 抓不到：{e2}")
    try:
        out["cor3m"] = fetch_cboe_index("COR3M")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"COR3M 抓不到（{type(e).__name__}），略過這個確認指標")
    try:
        from utils import data as D
        out["sectors"] = D.get_close(SECTOR_ETFS, period="5y")
        missing = [t for t in SECTOR_ETFS if t not in out["sectors"].columns]
        if missing:
            warnings.append(f"產業 ETF 缺少：{'、'.join(missing)}")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"產業 ETF 抓不到：{e}")
    try:
        from utils import data as D
        out["px"] = D.get_price(symbol, period="max")["Close"]
    except Exception as e:  # noqa: BLE001
        warnings.append(f"{symbol} 價格抓不到：{e}")
    return out, warnings


def compute_live_signal(symbol: str = "QQQ") -> dict:
    live, warnings = gather_live(symbol)
    vix = live["vix"] if live["vix"] is not None else pd.Series(dtype=float)
    res = capitulation_state(vix, live["sectors"], live["cor3m"], live["px"])
    res["as_of"] = date.today().isoformat()
    res["symbol"] = symbol
    res["warnings"] = warnings
    res["evidence"] = load_evidence()
    if not res["evidence"]:
        warnings.append("找不到 utils/signal_evidence.json（研究證據檔）；請執行 research/export_evidence.py")
    res["series"] = live          # 給頁面畫圖用；寫 JSON 前要拿掉
    return res
