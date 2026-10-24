"""第 8 頁「進出場訊號」的離線渲染測試（不需要網路）：把抓資料換成合成資料、把時間固定，
跑正常、全面恐慌、加碼期間、抓不到 VIX、資料過期、產業 ETF 缺值、計算器、參數檔超出範圍、失敗不快取。
    PYTHONUTF8=1 .venv/Scripts/python tests/test_page8_render.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from streamlit.testing.v1 import AppTest  # noqa: E402

from test_live_fixes import ny, raw_data  # noqa: E402
from utils import signals_live as L  # noqa: E402

FAILS: list[str] = []
PAGE = ROOT / "pages" / "8_進出場訊號.py"
NOW = ny(2026, 10, 20, 18, 0)           # 週二收盤後；最近一個已收盤交易日 = 2026-10-20
CALLS = {"gather": 0}


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def install(raw: dict, warnings: list[str] | None = None, now=NOW):
    def fake_gather(symbol="QQQ"):
        CALLS["gather"] += 1
        return raw, list(warnings or [])
    L.gather_live = fake_gather
    L._now = lambda: now


def run(page: Path = PAGE, clear: bool = True) -> AppTest:
    import streamlit as st
    if clear:
        st.cache_data.clear()        # 頁面用 st.cache_data 快取完整的結果；每個情境前清掉
    at = AppTest.from_file(str(page), default_timeout=60)
    at.run()
    return at


def tmp_page(local: bool, params=None) -> Path:
    """把頁面複製到暫存資料夾：local=True 時放一個空的 .env（可以存檔），params 寫成參數檔。"""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "pages").mkdir()
    shutil.copy(PAGE, tmp / "pages" / "8.py")
    if local:
        (tmp / ".env").write_text("", encoding="utf-8")
    if params is not None:
        (tmp / "cache").mkdir()
        (tmp / "cache" / "entry_params.json").write_text(json.dumps(params), encoding="utf-8")
    return tmp / "pages" / "8.py"


def card(at) -> str:
    return " ".join(m.value for m in at.markdown if "恐慌觀察燈" in m.value)


def level_card(at) -> str:
    return " ".join(m.value for m in at.markdown if "目前等級（依今天的資料）" in m.value)


def exit_card(at) -> str:
    return " ".join(m.value for m in at.markdown if "出場參考（研究裡沒有任何出場訊號通過驗證）" in m.value)


def order(at) -> list[tuple[str, str]]:
    """頁面元素的先後順序（類型、文字）。有些輸入元件讀 value 會丟例外，當成空字串。"""
    def _val(e):
        try:
            return str(getattr(e, "value", ""))
        except Exception:  # noqa: BLE001
            return ""
    return [(type(e).__name__, _val(e)) for e in at.main]


def first(at, pred) -> int:
    return next((i for i, x in enumerate(order(at)) if pred(x)), 10 ** 6)


def panic_raw(days_after: int, vix_after: float = 35.0) -> dict:
    """最後 days_after+1 天：第一天 VIX 45（首次站上 40），之後是 vix_after。"""
    return raw_data("2026-10-20", vix_tail=[45.0] + [vix_after] * days_after)


def set_inputs(at, values: dict):
    ni = {n.label: n for n in at.number_input}
    for label, v in values.items():
        ni[label].set_value(v)
    at.run()


if __name__ == "__main__":
    os.environ.pop("STOCK_LOCAL_SAVE", None)

    print("正常")
    install(raw_data("2026-10-20"))
    at = run()
    check("沒有例外", len(at.exception) == 0)
    md = " ".join(m.value for m in at.markdown)
    raw0 = raw_data("2026-10-20")
    exp = 1 if raw0["vix"].iloc[-1] < 20 else 2
    check("顯示「正常」，最上方是目前等級和新錢照原本比例投入",
          "⚪ 正常" in card(at) and f"第 {exp} 級（共 4 級）" in level_card(at)
          and "新錢照你原本的比例投入" in level_card(at))
    check("最上方說明：分成 4 個等級、要不要買由你決定、不是投資建議",
          any("分成 4 個等級" in i.value and "最後由你決定" in i.value and "不是投資建議" in i.value for i in at.info))
    check("等級卡寫出這個等級歷史上之後 3 個月的表現", f"歷史上（1990 起）第 {exp} 級之後 3 個月" in level_card(at))
    check("右上有出場參考卡：寫明 40 個出場規則通過 0 個、不是賣出建議",
          "研究測了 40 個出場規則，通過 0 個" in exit_card(at) and "不是賣出建議" in exit_card(at)
          and "10 個月均線" in exit_card(at))
    check("恐慌觀察燈的細節移到兩張卡下面", "⚪ 正常" in card(at)
          and first(at, lambda x: "出場參考" in x[1]) < first(at, lambda x: "恐慌觀察燈" in x[1]))
    tbl = [d for d in at.dataframe if "等級" in d.value.columns]
    check("有四個等級的表，標出現在是哪一級", bool(tbl) and len(tbl[0].value) == 5
          and tbl[0].value.loc[tbl[0].value[""] == "▶ 現在", "等級"].tolist() == [f"{exp} {L.LEVELS[exp]['name']}"])
    check("現金桶不動", any("目前不是加碼期間：現金桶不動" in i.value for i in at.info))
    check("各來源資料日直接顯示在頁面上（不是只在滑鼠提示）",
          any("VIX 2026-10-20" in c.value and "QQQ 2026-10-20" in c.value for c in at.caption))
    check("資料完整時沒有「重新抓取」按鈕", not any(b.label == "🔄 重新抓取資料" for b in at.button))
    local_save = (ROOT / ".env").exists()
    check("本機：有 .env 時才有儲存按鈕", any(b.label == "儲存我的參數" for b in at.button) == local_save)

    print("線上版（沒有 .env）")
    page = tmp_page(local=False)
    at = run(page)
    check("沒有例外", len(at.exception) == 0)
    check("沒有儲存按鈕", not any(b.label == "儲存我的參數" for b in at.button))
    check("提醒不要輸入真實金額", any("不要在線上版輸入真實金額" in c.value for c in at.caption))
    check("沒有寫出參數檔", not (page.parent.parent / "cache").exists())
    shutil.rmtree(page.parent.parent, ignore_errors=True)

    print("Bug 3：全面恐慌當天與加碼期間")
    install(panic_raw(0))
    at = run(tmp_page(local=False))
    check("觸發當天：燈號全面恐慌", "🟢 全面恐慌" in card(at) and "今天收盤首次站上 40" in card(at))
    check("觸發當天：第 4 級，寫出達到加碼條件", "第 4 級（共 4 級）" in level_card(at) and "達到你手冊寫的加碼條件" in level_card(at))
    check("觸發當天：出場參考寫「加碼期間中：先不調回」", "加碼期間中：先不調回" in exit_card(at))
    check("觸發當天：算出加碼金額", any("從現金桶買進" in s.value for s in at.success))
    install(panic_raw(3, vix_after=35.0))
    at = run(tmp_page(local=False))
    check("觸發後第 4 天、VIX 回到 35：還是加碼期間（不是「留意：先不動」）",
          "🟢 加碼期間" in card(at) and "觸發日" in card(at) and "還剩 122 個交易日" in card(at))
    check("加碼期間的說明：期間內再站上 40 不是再買一次", "不是再買一次" in card(at))
    check("加碼期間的計算器仍然是加碼（不是「現金桶不動」）",
          any(s.value.startswith("加碼期間") for s in at.success) and not any("現金桶不動" in i.value for i in at.info))
    check("加碼期間 → 第 4 級", "第 4 級（共 4 級）" in level_card(at))

    print("Bug 4：計算器依目前市值算實際比例")
    install(panic_raw(0))
    at = run(tmp_page(local=False))
    set_inputs(at, {"目前股票市值": 280_000, "目前可投資現金（含現金桶）": 600_000, "平常的股票比例（%）": 40,
                    "加碼期間多加（百分點）": 10, "現金桶還剩多少": 200_000})
    msg = " ".join(s.value for s in at.success)
    check("股價跌了 30%：買 160,000，比例從 31.8% 變成 50.0%",
          "160,000" in msg and "從目前 31.8% 變成約 50.0%" in msg and "72,000" in msg)
    set_inputs(at, {"現金桶還剩多少": 100_000})
    msg = " ".join(s.value for s in at.success)
    check("現金桶只有 10 萬：寫出實際只到 43.2%，並說明被卡住", "變成約 43.2%" in msg and "卡住" in msg)
    check("輸入今天的市值後，不再提醒市值是舊的", not any("⚠️ 股票市值" in c.value for c in at.caption))

    print("Bug 1：抓不到 VIX")
    raw = raw_data("2026-10-20")
    install(dict(raw, vix=None), warnings=["VIX 抓不到（CBOE：ValueError；Yahoo：RuntimeError），無法判斷燈號"])
    at = run(tmp_page(local=False))
    check("沒有例外", len(at.exception) == 0)
    check("燈號：無法判斷", "⚠️ 無法判斷" in card(at))
    check("最上方紅色提示：請自己查 VIX 收盤（首次站上 40、126 個交易日的加碼期間）",
          any("請自己查 VIX 收盤" in e.value and "收盤首次站上 40" in e.value and "126 個交易日" in e.value
              for e in at.error))
    check("計算器沒有寫「目前不是加碼期間：現金桶不動」", not any("現金桶不動" in i.value for i in at.info))
    check("抓不到 VIX → 無法分級", "無法分級" in level_card(at))
    i_err = first(at, lambda x: x[0] == "Error")
    i_warn = first(at, lambda x: x[0] == "Warning" and "VIX 抓不到" in x[1])
    i_card = first(at, lambda x: "恐慌觀察燈" in x[1])
    check("警告在燈號卡之前（頁面最上方）", i_err < i_card and i_warn < i_card)
    check("資料不完整時有「重新抓取」按鈕", any(b.label == "🔄 重新抓取資料" for b in at.button))

    print("Bug 2：資料過期")
    install(raw_data("2026-09-25"))
    at = run(tmp_page(local=False))
    check("燈號：資料過期（不是正常）", "⚠️ 資料過期" in card(at) and "正常" not in card(at))
    check("最上方：資料過期，列出每個來源的資料日",
          any(w.value.startswith("資料過期") and "VIX 停在 2026-09-25" in w.value and "QQQ 停在 2026-09-25" in w.value
              for w in at.warning))
    check("過期的警告在燈號卡之前",
          first(at, lambda x: x[0] == "Warning" and x[1].startswith("資料過期")) < first(at, lambda x: "恐慌觀察燈" in x[1]))
    check("計算器寫無法判斷", any("無法判斷要不要加碼" in w.value for w in at.warning))

    print("Bug 5：週一盤中的 K 線不混進來")
    raw = raw_data("2026-10-19", vix_tail=[20, 21, 45])
    install(raw, now=ny(2026, 10, 19, 11, 0))
    at = run(tmp_page(local=False))
    check("盤中 VIX 45 不亮全面恐慌；資料日是上週五", "⚪ 正常" in card(at)
          and any("資料日 2026-10-16" in c.value for c in at.caption))

    print("小問題：產業 ETF 缺值")
    raw = raw_data("2026-10-20")
    raw["sectors"] = raw["sectors"].copy()
    raw["sectors"].iloc[-1, :5] = np.nan
    install(raw)
    at = run(tmp_page(local=False))
    m2 = [m for m in at.metric if m.label == "產業 ETF 在一年新低"][0]
    check("分母是實際有資料的 4 檔", m2.value.endswith("/ 4 檔"))
    check("頁面說明缺哪幾檔", any("缺 XLB、XLE、XLF、XLI、XLK" in c.value for c in at.caption))

    print("小問題：參數檔超出範圍")
    install(raw_data("2026-10-20"))
    page = tmp_page(local=True, params={"base_equity_pct": 120, "hold_months": 99, "stock_value": -5,
                                        "tilt_pp": "abc"})
    at = run(page)
    check("參數檔有超出範圍的值 → 沒有例外", len(at.exception) == 0)
    ni = {n.label: n.value for n in at.number_input}
    check("超出範圍的值被截到範圍內", ni["平常的股票比例（%）"] == 100 and ni["加碼後至少持有（月）"] == 36
          and ni["目前股票市值"] == 0 and ni["加碼期間多加（百分點）"] == L.DEFAULT_ENTRY_PARAMS["tilt_pp"])
    page.parent.parent.joinpath("cache", "entry_params.json").write_text("[1, 2", encoding="utf-8")
    at = run(page)
    check("參數檔壞掉（不是合法 JSON）→ 用預設值、沒有例外", len(at.exception) == 0)
    shutil.rmtree(page.parent.parent, ignore_errors=True)

    print("小問題：抓資料失敗的結果不快取 1 小時")
    good = raw_data("2026-10-20")
    install(dict(good, vix=None), warnings=["VIX 抓不到"])
    page = tmp_page(local=False)
    at1 = run(page)
    check("先失敗：無法判斷", "無法判斷" in card(at1))
    install(good)
    n0 = CALLS["gather"]
    at2 = run(page, clear=False)
    check("另一個分頁再開：重新抓資料，拿到正常結果（失敗沒有被快取）",
          "⚪ 正常" in card(at2) and CALLS["gather"] == n0 + 1)
    install(dict(good, vix=None))
    n1 = CALLS["gather"]
    at3 = run(page, clear=False)
    check("完整的結果有快取：再開一次不重抓", "⚪ 正常" in card(at3) and CALLS["gather"] == n1)
    install(good)
    [b for b in at1.button if b.label == "🔄 重新抓取資料"][0].click()
    at1.run()
    check("原本失敗的分頁按「重新抓取資料」→ 拿到正常結果", "⚪ 正常" in card(at1))
    shutil.rmtree(page.parent.parent, ignore_errors=True)

    print("第二輪：輸入框連續修改不會被吃掉")
    install(panic_raw(0))
    page = tmp_page(local=False)
    at = run(page)
    seen = []
    for v in (300_000, 280_000, 260_000, 240_000):
        set_inputs(at, {"目前股票市值": v})
        seen.append({n.label: n.value for n in at.number_input}["目前股票市值"])
    check("連續改 4 次股票市值，每次都生效", seen == [300_000, 280_000, 260_000, 240_000])
    set_inputs(at, {"目前可投資現金（含現金桶）": 600_000, "現金桶還剩多少": 150_000})
    set_inputs(at, {"現金桶還剩多少": 100_000})
    check("連續改兩次現金桶，第二次也生效", {n.label: n.value for n in at.number_input}["現金桶還剩多少"] == 100_000)

    print("第二輪：現金桶是 0")
    set_inputs(at, {"目前股票市值": 280_000, "現金桶還剩多少": 0})
    check("現金桶是 0 → 不說「已經達到目標」，說現金桶用完",
          not any("已經達到目標" in x.value for x in at.success)
          and any("現金桶已經用完" in w.value for w in at.warning))
    set_inputs(at, {"目前股票市值": 700_000, "目前可投資現金（含現金桶）": 300_000, "現金桶還剩多少": 100_000})
    check("已經在目標以上 → 不用再買", any("已經達到目標" in x.value for x in at.success))
    shutil.rmtree(page.parent.parent, ignore_errors=True)

    print("第二輪：儲存時不幫沒更新過的市值蓋上今天的日期")
    install(raw_data("2026-10-20"))
    page = tmp_page(local=True)
    at = run(page)
    set_inputs(at, {"平常的股票比例（%）": 50})
    [b for b in at.button if b.label == "儲存我的參數"][0].click()
    at.run()
    saved = json.loads((page.parent.parent / "cache" / "entry_params.json").read_text(encoding="utf-8"))
    check("只改比例就儲存 → 市值日期仍是空的（下次還會提醒）", saved["holdings_date"] is None
          and saved["base_equity_pct"] == 50)
    [b for b in at.button if b.label == "✔ 這是今天的數字"][0].click()
    at.run()
    [b for b in at.button if b.label == "儲存我的參數"][0].click()
    at.run()
    saved = json.loads((page.parent.parent / "cache" / "entry_params.json").read_text(encoding="utf-8"))
    from datetime import date as _date
    check("按「這是今天的數字」再儲存 → 日期是今天", saved["holdings_date"] == _date.today().isoformat())
    shutil.rmtree(page.parent.parent, ignore_errors=True)

    print("第二輪：過了收盤時間就重新計算（快取的鍵包含交易日）")
    install(raw_data("2026-10-20"), now=ny(2026, 10, 21, 16, 29))
    page = tmp_page(local=False)
    at = run(page)
    check("16:29 → 資料到 10/20，完整", "⚪ 正常" in card(at) and not at.warning)
    install(raw_data("2026-10-20"), now=ny(2026, 10, 21, 17, 0))
    at2 = run(page, clear=False)
    check("17:00 → 不沿用快取，提醒 VIX 還沒更新到 10/21",
          any("VIX 還沒更新到 2026-10-21" in w.value for w in at2.warning))
    shutil.rmtree(page.parent.parent, ignore_errors=True)

    print("第二輪：VIX 過期但加碼期間一定還在")
    raw = raw_data("2026-10-20")
    vv = raw["vix"].loc[:"2026-10-14"].copy()
    vv.iloc[-4] = 30.0
    vv.iloc[-3:] = 45.0
    raw["vix"] = vv
    install(raw)
    at = run(tmp_page(local=False))
    m1 = [m for m in at.metric if m.label == "VIX 收盤"][0]
    check("燈號：加碼期間，並說明 VIX 過期但一定還在期間內",
          "🟢 加碼期間" in card(at) and "一定還在加碼期間" in card(at))
    check("VIX 數字標出（過期）", m1.value.endswith("（過期）"))
    check("最上方仍有資料過期的警告", any(w.value.startswith("資料過期") for w in at.warning))

    print("第二輪：加碼期間剛結束")
    raw = raw_data("2026-10-20")
    vv = raw["vix"].copy()
    sess = [t for t in vv.index if L.is_session(t.date())]
    vv.loc[sess[-131]] = 30.0
    vv.loc[sess[-130]] = 45.0                 # 期間在 4 個交易日前結束
    raw["vix"] = vv
    install(raw)
    at = run(tmp_page(local=False))
    set_inputs(at, {"目前股票市值": 700_000, "目前可投資現金（含現金桶）": 300_000, "平常的股票比例（%）": 50,
                    "加碼期間多加（百分點）": 20})
    check("提醒把股票比例調回平常的 50%", any("調回平常的 50%" in i.value for i in at.info))

    print("\n全部通過" if not FAILS else f"\n失敗：{FAILS}")
    sys.exit(1 if FAILS else 0)
