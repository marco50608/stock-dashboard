"""lib/options.py 的單元測試（不需要網路、不需要 pytest）：
    PYTHONUTF8=1 .venv/Scripts/python research/tests/test_options.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import options as O  # noqa: E402

FAILS = []


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def test_bs():
    p = float(O.bs_put(100, 100, 1, 0.05, 0, 0.2))
    check(f"BS put 教科書值 5.5735（得到 {p:.4f}）", abs(p - 5.5735) < 1e-3)
    c = float(O.bs_call(100, 95, 0.5, 0.03, 0.01, 0.3))
    pp = float(O.bs_put(100, 95, 0.5, 0.03, 0.01, 0.3))
    parity = c - pp - (100 * np.exp(-0.01 * 0.5) - 95 * np.exp(-0.03 * 0.5))
    check("put-call parity 成立", abs(parity) < 1e-9)


def _flat(idx, v):
    return pd.Series(v, index=idx, dtype=float)


def test_accumulate():
    idx = pd.bdate_range("2010-01-01", periods=600)
    up = pd.Series(np.linspace(100, 160, len(idx)), index=idx)
    tr = up.pct_change().fillna(0)
    steep = pd.Series(np.linspace(100, 400, len(idx)), index=idx)      # 每年約 +78%，遠大於權利金
    res = O.accumulate_csp(steep, steep.pct_change().fillna(0), _flat(idx, 0.2), _flat(idx, 0.0), haircut=0.0)
    check("急漲時 put 從不被指派，等於一直抱現金（只有權利金）→ 輸給直接買",
          res["assigned"] == 0 and res["ratio"] < 1)
    res0 = O.accumulate_csp(up, tr, _flat(idx, 1e-6), _flat(idx, 0.0), haircut=0.0, max_wait=1)
    check("波動 ≈ 0、等 1 個月就買：和直接買差不多（只差每月一個月的延遲）", 0.9 < res0["ratio"] <= 1.0)
    down = pd.Series(np.linspace(160, 100, len(idx)), index=idx)
    trd = down.pct_change().fillna(0)
    rd = O.accumulate_csp(down, trd, _flat(idx, 0.2), _flat(idx, 0.0), haircut=0.0)
    check("一路下跌時每個月都被指派，而且因為收了權利金，比直接買好", rd["assign_rate"] > 0.9 and rd["ratio"] > 1)


def test_putwrite_index():
    idx = pd.bdate_range("2010-01-01", periods=300)
    flat = _flat(idx, 100.0)
    nav = O.putwrite_index(flat, _flat(idx, 0.2), _flat(idx, 0.0), haircut=0.0)["nav"]
    check("價格不動時賣 ATM put 每個月都賺到權利金（淨值上升）", nav.iloc[-1] > nav.iloc[0] * 1.05)


if __name__ == "__main__":
    for f in [test_bs, test_accumulate, test_putwrite_index]:
        print(f.__name__)
        f()
    print("\n全部通過" if not FAILS else f"\n失敗：{FAILS}")
    sys.exit(1 if FAILS else 0)
