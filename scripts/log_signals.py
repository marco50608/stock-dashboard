"""
每週訊號紀錄：把「全面恐慌」觀察燈的狀態寫進 cache/signal_log.csv。

目的：research/REPORT.md 的觀察名單（VIX ≥ 40 等）沒有通過多重檢定，要升級成「通過」只能靠
上線之後才累積的真正樣本外資料。這個腳本由 GitHub Actions 每週更新資料時順便執行
（.github/workflows/update-data.yml 的「Log weekly signal state」步驟），也可以在本機手動跑。

不依賴 streamlit（GitHub Actions 那一步只多裝 yfinance）：
  - VIX、COR3M：CBOE 官方 CSV（utils.signals_live.fetch_cboe_index）
  - 9 檔產業 ETF、標的價格：yfinance
同一個資料日只保留一列（重跑會覆蓋，不會重複）。

Usage:  python scripts/log_signals.py [SYMBOL]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import signals_live as L  # noqa: E402  （這個模組在 import 時不會載入 streamlit）

LOG_PATH = ROOT / "cache" / "signal_log.csv"
COLUMNS = ["data_date", "logged_at_utc", "symbol", "state", "vix", "vix_to_panic", "sectors_at_low",
           "sectors_of", "cor3m", "cor3m_pct", "confirmations", "close", "from_high", "trend10m_above",
           "last_panic", "evidence_generated", "warnings"]


def _yf_close(tickers: list[str], period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, period=period, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance 沒有回傳 {tickers}")
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all")


def build_row(symbol: str = "QQQ") -> dict:
    warnings: list[str] = []
    vix = L.fetch_cboe_index("VIX")                     # VIX 抓不到就整個失敗（不寫半套紀錄）
    try:
        cor = L.fetch_cboe_index("COR3M")
    except Exception as e:  # noqa: BLE001
        cor = None
        warnings.append(f"COR3M: {type(e).__name__}")
    try:
        secs = _yf_close(L.SECTOR_ETFS, "5y")
    except Exception as e:  # noqa: BLE001
        secs = None
        warnings.append(f"sectors: {type(e).__name__}")
    try:
        px = _yf_close([symbol], "max")[symbol]
    except Exception as e:  # noqa: BLE001
        px = None
        warnings.append(f"{symbol}: {type(e).__name__}")

    res = L.capitulation_state(vix, secs, cor, px)
    vs, ss, cs, ds, tr = res["vix"], res["sectors"], res["cor3m"], res["drawdown"], res["trend10m"]
    ev = L.load_evidence()
    return {
        "data_date": vs["date"],
        "logged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": symbol,
        "state": res["state"],
        "vix": round(vs["value"], 2),
        "vix_to_panic": round(vs["to_panic"], 2),
        "sectors_at_low": ss.get("count"),
        "sectors_of": ss.get("of"),
        "cor3m": None if cs.get("value") is None else round(cs["value"], 2),
        "cor3m_pct": None if cs.get("pct") is None else round(cs["pct"], 1),
        "confirmations": "、".join(res["confirmations"]),
        "close": None if px is None else round(float(px.dropna().iloc[-1]), 2),
        "from_high": None if ds.get("from_high") is None else round(ds["from_high"], 4),
        "trend10m_above": tr.get("above"),
        "last_panic": vs.get("last_panic"),
        "evidence_generated": ev.get("generated"),
        "warnings": "; ".join(warnings),
    }


def append_row(row: dict, path: Path | None = None) -> pd.DataFrame:
    """寫入（同一個 data_date + symbol 只留最新一列），回傳整份紀錄。"""
    path = Path(path) if path is not None else LOG_PATH
    new = pd.DataFrame([row], columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        df = pd.concat([old, new.astype(str)], ignore_index=True)
    else:
        df = new.astype(str)
    df = df.replace({"None": "", "nan": ""})
    df = df.drop_duplicates(subset=["data_date", "symbol"], keep="last").sort_values(["data_date", "symbol"])
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def main(symbol: str = "QQQ") -> dict:
    row = build_row(symbol)
    df = append_row(row, LOG_PATH)
    print(f"[log_signals] {row['data_date']} {symbol}：{row['state']}（VIX {row['vix']}，"
          f"產業 ETF 新低 {row['sectors_at_low']}/{row['sectors_of']}，COR3M 百分位 {row['cor3m_pct']}）"
          f"→ {LOG_PATH}（共 {len(df)} 列）")
    if row["warnings"]:
        print(f"[log_signals] 警告：{row['warnings']}")
    return row


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "QQQ"
    try:
        main(sym)
    except Exception as e:  # noqa: BLE001
        print(f"[log_signals] FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
