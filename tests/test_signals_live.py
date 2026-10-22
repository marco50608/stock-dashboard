"""utils/signals_live.py 的單元測試（不需要網路，也不需要 pytest）：
    PYTHONUTF8=1 .venv/Scripts/python tests/test_signals_live.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import signals_live as L  # noqa: E402

FAILS: list[str] = []
idx = pd.bdate_range("2020-01-01", periods=600)


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def test_rising_edges():
    c = pd.Series(False, index=idx)
    c.iloc[[10, 11, 12, 20, 40, 41]] = True
    e = L.rising_edges(c, cooldown=21)
    check("連續成立只算一次、21 天內不重複", list(e) == [idx[10], idx[40]])


def test_vix_levels():
    v = pd.Series(18.0, index=idx)
    check("VIX 18 → 正常", L.vix_status(v)["level"] == L.STATE_NORMAL)
    v.iloc[-1] = 31
    s = L.vix_status(v)
    check("VIX 31 → 留意，距 40 還差 9", s["level"] == L.STATE_WATCH and abs(s["to_panic"] - 9) < 1e-9)
    v.iloc[-1] = 40
    check("VIX 剛好 40 → 全面恐慌", L.vix_status(v)["level"] == L.STATE_PANIC)
    v2 = pd.Series(18.0, index=idx)
    v2.iloc[100:105] = 45
    s2 = L.vix_status(v2)
    check("記得上次全面恐慌的日期", s2["last_panic"] == idx[100].date().isoformat() and s2["level"] == L.STATE_NORMAL)
    check("空序列不會壞", L.vix_status(pd.Series(dtype=float))["level"] is None)


def test_sectors():
    rng = np.random.default_rng(0)
    up = pd.DataFrame(np.cumprod(1 + rng.normal(0.001, 0.01, (600, 9)), axis=0), index=idx, columns=L.SECTOR_ETFS)
    check("一路上漲時沒有 ETF 在一年新低", L.sector_status(up)["count"] == 0)
    crash = up.copy()
    crash.iloc[-1, :5] = crash.iloc[:, :5].min().values * 0.9
    s = L.sector_status(crash)
    check("5 檔創新低 → 計數 5、達到確認門檻", s["count"] == 5 and s["confirm"] and len(s["names"]) == 5)
    check("資料不足一年時回傳 None", L.sector_status(up.iloc[:100])["count"] is None)


def test_cor3m_and_drawdown():
    c = pd.Series(np.linspace(10, 30, 300), index=idx[:300])
    s = L.cor3m_status(c)
    check("COR3M 創一年新高 → 百分位 100、確認", s["pct"] == 100 and s["confirm"])
    px = pd.Series([100, 120, 90, 108], index=idx[:4], dtype=float)
    check("距高點跌幅 = 108/120 − 1", abs(L.drawdown_status(px)["from_high"] - (108 / 120 - 1)) < 1e-12)


def test_ten_month():
    days = pd.bdate_range("2023-01-02", "2025-06-10")
    up = pd.Series(np.linspace(100, 200, len(days)), index=days)
    t = L.ten_month_trend(up)
    check("一路上漲 → 在 10 個月均線之上，用已收完的月份（2025-05）",
          t["above"] is True and t["month"] == "2025-05")
    down = pd.Series(np.linspace(200, 100, len(days)), index=days)
    check("一路下跌 → 跌破均線", L.ten_month_trend(down)["above"] is False)


def test_state_and_plan():
    v = pd.Series(18.0, index=idx)
    v.iloc[-1] = 42
    res = L.capitulation_state(v)
    check("只有 VIX 也能判斷狀態（今天剛站上 40 → 全面恐慌）", res["state"] == L.STATE_PANIC and res["confirmations"] == [])
    p = L.overweight_plan(L.STATE_PANIC, stock_value=400_000, cash=600_000, base_equity_pct=40, tilt_pp=10,
                          cash_bucket=60_000, hold_months=6)
    check("加碼金額受現金桶限制（想加 10 萬、只有 6 萬）", p["amount"] == 60_000 and p["limited_by_cash"])
    p2 = L.overweight_plan(L.STATE_NORMAL, 400_000, 600_000, 40, 10, 500_000, 6)
    check("不是加碼期間時建議金額為 0", p2["amount"] == 0 and p2["amount_if_triggered"] == 100_000)


def test_brief_and_evidence():
    v = pd.Series(18.0, index=idx)
    res = L.capitulation_state(v)
    res["evidence"] = L.load_evidence()
    txt = L.format_brief(res)
    check("簡報包含新錢直接買、出場說明", "新存的錢：照計畫直接買" in txt and "出場" in txt)
    with tempfile.TemporaryDirectory() as d:
        check("證據檔不存在時回傳空 dict", L.load_evidence(Path(d) / "nope.json") == {})
    ev = L.load_evidence()
    check("證據檔可讀且結論是 0 個通過", ev.get("n_passed") == 0 and len(ev.get("vix40", {}).get("events", [])) > 0)
    json.dumps({k: v for k, v in res.items() if k != "series"}, ensure_ascii=False, default=str)
    check("結果可以存成 JSON", True)


if __name__ == "__main__":
    for f in [test_rising_edges, test_vix_levels, test_sectors, test_cor3m_and_drawdown, test_ten_month,
              test_state_and_plan, test_brief_and_evidence]:
        print(f.__name__)
        f()
    print("\n全部通過" if not FAILS else f"\n失敗：{FAILS}")
    sys.exit(1 if FAILS else 0)
