"""
每週訊號紀錄：把「全面恐慌」觀察燈的狀態寫進 cache/signal_log.csv。

目的：research/REPORT.md 的觀察名單（VIX ≥ 40 等）沒有通過多重檢定，要升級成「通過」只能靠
上線之後才累積的真正樣本外資料。這個腳本由 GitHub Actions 每週更新資料時順便執行
（.github/workflows/update-data.yml 的「Log weekly signal state」步驟），也可以在本機手動跑。

不依賴 streamlit（GitHub Actions 那一步只多裝 yfinance）。資料和第 8 頁共用 utils.signals_live：
  - VIX、COR3M：CBOE 官方 CSV（VIX 失敗或回傳空檔時改用 Yahoo ^VIX）
  - 9 檔產業 ETF、標的價格：yfinance
  - 只用已收盤的日線，各來源對齊到 VIX 的資料日；過期的來源寫在 stale_sources
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
COLUMNS = ["data_date", "logged_at_utc", "symbol", "state", "vix", "vix_to_panic", "vix_via", "window_trigger",
           "window_since", "window_remaining", "sectors_at_low", "sectors_of", "cor3m", "cor3m_pct",
           "confirmations", "close", "from_high", "trend10m_above", "last_panic", "expected_session",
           "stale_sources", "evidence_generated", "warnings"]


def build_row(symbol: str = "QQQ") -> dict:
    """和第 8 頁、每日簡報同一套流程（utils.signals_live.compute_live_signal）：只用已收盤的日線、
    各來源對齊到 VIX 的資料日、過期的來源會標出來。VIX 完全抓不到時整個失敗（不寫半套紀錄）。"""
    res = L.compute_live_signal(symbol)
    vs, ss, cs, ds, tr = res["vix"], res["sectors"], res["cor3m"], res["drawdown"], res["trend10m"]
    if vs["value"] is None:
        raise RuntimeError("；".join(res["warnings"]) or "VIX 抓不到")
    fr = res["freshness"]
    if fr["sources"]["vix"]["stale"]:
        # 過期的 VIX 會用舊的資料日寫一列「資料過期」，蓋掉那天原本準時記下的紀錄，所以不寫
        raise RuntimeError(f"VIX 資料過期（停在 {fr['sources']['vix']['date']}，最近一個已收盤交易日 "
                           f"{fr['expected']}），這次不寫紀錄")
    w = vs.get("window") or {}
    px = res["series"].get("px")
    return {
        "data_date": res["data_date"],
        "logged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": symbol,
        "state": res["state"],
        "vix": round(vs["value"], 2),
        "vix_to_panic": round(vs["to_panic"], 2),
        "vix_via": fr["sources"]["vix"].get("via"),
        "window_trigger": w.get("trigger"),
        "window_since": w.get("since"),
        "window_remaining": w.get("remaining"),
        "sectors_at_low": ss.get("count"),
        "sectors_of": ss.get("of"),
        "cor3m": None if cs.get("value") is None else round(cs["value"], 2),
        "cor3m_pct": None if cs.get("pct") is None else round(cs["pct"], 1),
        "confirmations": "、".join(res["confirmations"]),
        "close": None if px is None else round(float(px.dropna().iloc[-1]), 2),
        "from_high": None if ds.get("from_high") is None else round(ds["from_high"], 4),
        "trend10m_above": tr.get("above"),
        "last_panic": vs.get("last_panic"),
        "expected_session": fr["expected"],
        "stale_sources": "、".join(f"{x['label']} {x['date']}" for x in fr["sources"].values() if x["stale"]),
        "evidence_generated": (res.get("evidence") or {}).get("generated"),
        "warnings": "; ".join(res["warnings"]),
    }


def append_row(row: dict, path: Path | None = None) -> pd.DataFrame:
    """寫入（同一個 data_date + symbol 只留最新一列），回傳整份紀錄。
    例外：那天已經有一列沒有過期來源的紀錄，新的這列卻有過期來源時，保留原本那列。"""
    path = Path(path) if path is not None else LOG_PATH
    new = pd.DataFrame([row], columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        same = old[(old["data_date"] == str(row["data_date"])) & (old["symbol"] == str(row["symbol"]))]
        prev_stale = same["stale_sources"].fillna("").iloc[-1] if len(same) and "stale_sources" in same else None
        if row.get("stale_sources") and prev_stale == "":
            return old
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
