"""第 8 頁「進出場訊號」的離線渲染測試（不需要網路）：把資料換成假的，分別跑正常與全面恐慌兩種狀態。
    PYTHONUTF8=1 .venv/Scripts/python tests/test_page8_render.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest  # noqa: E402

from utils import signals_live as L  # noqa: E402

FAILS: list[str] = []


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def fake_signal(last_vix: float):
    idx = pd.bdate_range("2021-01-04", periods=1400)
    rng = np.random.default_rng(1)
    vix = pd.Series(18 + rng.normal(0, 2, len(idx)), index=idx).clip(lower=10)
    vix.iloc[-1] = last_vix
    secs = pd.DataFrame(np.cumprod(1 + rng.normal(0.0003, 0.01, (len(idx), 9)), axis=0), index=idx,
                        columns=L.SECTOR_ETFS)
    px = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.012, len(idx))) * 300, index=idx)
    cor = pd.Series(20 + rng.normal(0, 3, len(idx)), index=idx)

    def _fn(symbol="QQQ"):
        res = L.capitulation_state(vix, secs, cor, px)
        res.update({"as_of": "2026-10-19", "symbol": symbol, "warnings": [], "evidence": L.load_evidence(),
                    "series": {"vix": vix, "cor3m": cor, "sectors": secs, "px": px}})
        return res
    return _fn


def run(last_vix: float, page: Path | None = None):
    import streamlit as st
    st.cache_data.clear()            # 頁面用 st.cache_data 快取結果；每次換假資料前要清掉
    L.compute_live_signal = fake_signal(last_vix)
    at = AppTest.from_file(str(page or ROOT / "pages" / "8_進出場訊號.py"), default_timeout=60)
    at.run()
    return at


def run_online(last_vix: float):
    """模擬線上部署：把頁面複製到沒有 .env 的暫存資料夾執行（不能存檔）。"""
    import os
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    old = os.environ.pop("STOCK_LOCAL_SAVE", None)
    try:
        (tmp / "pages").mkdir()
        page = tmp / "pages" / "8.py"
        shutil.copy(ROOT / "pages" / "8_進出場訊號.py", page)
        return run(last_vix, page), tmp
    finally:
        if old is not None:
            os.environ["STOCK_LOCAL_SAVE"] = old


if __name__ == "__main__":
    at = run(18.0)
    check("正常狀態：沒有例外", len(at.exception) == 0)
    md = " ".join(m.value for m in at.markdown)
    check("正常狀態：顯示「正常」與新錢直接買", "⚪ 正常" in md and "照計畫直接買" in md)
    check("正常狀態：現金桶不動", any("現金桶不動" in i.value for i in at.info))
    local_save = (ROOT / ".env").exists()
    check("本機：有 .env 時才有儲存按鈕", any(b.label == "儲存我的參數" for b in at.button) == local_save)

    at, tmp = run_online(18.0)
    check("線上版：沒有例外", len(at.exception) == 0)
    check("線上版：沒有儲存按鈕", not any(b.label == "儲存我的參數" for b in at.button))
    check("線上版：提醒不要輸入真實金額", any("不要在線上版輸入真實金額" in c.value for c in at.caption))
    check("線上版：沒有寫出參數檔", not (tmp / "cache").exists())
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    at = run(45.0)
    check("全面恐慌：沒有例外", len(at.exception) == 0)
    md = " ".join(m.value for m in at.markdown)
    check("全面恐慌：燈號顯示全面恐慌", "🟢 全面恐慌" in md)
    check("全面恐慌：算出加碼金額", any("從現金桶買進" in s.value for s in at.success))
    print("\n全部通過" if not FAILS else f"\n失敗：{FAILS}")
    sys.exit(1 if FAILS else 0)
