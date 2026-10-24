"""第 8 頁和每日簡報 5 個 bug（加 3 個小問題）的回歸測試（不需要網路，也不需要 pytest）：
    PYTHONUTF8=1 .venv/Scripts/python tests/test_live_fixes.py

1. 抓不到 VIX：CBOE 回空檔要改用 Yahoo；完全沒有 VIX 時寫「無法判斷」，警告放最上面
2. 資料過期：資料日落後最近一個已收盤交易日太多時標成過期，燈號不顯示「正常」
3. 燈號照研究規則：首次站上 40 觸發，維持 126 個交易日，期間內再觸發重新起算
4. 加碼計算器：依目前股票市值和現金算實際比例
5. 只用已收盤的日線，各來源對齊到同一個資料日
小問題：產業 ETF 缺值時的分母、參數檔超出範圍
（「抓資料失敗的結果被快取 1 小時」要跑頁面才測得到，在 tests/test_page8_render.py）
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import signals_live as L  # noqa: E402

FAILS: list[str] = []
NY = L.NY


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def ny(*args) -> datetime:
    return datetime(*args, tzinfo=NY)


def raw_data(end: str, n: int = 1400, vix_tail: list[float] | None = None, seed: int = 1) -> dict:
    """合成的原始資料（交易日到 end 為止）。vix_tail：最後幾天的 VIX。"""
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    vix = pd.Series(18 + rng.normal(0, 1.5, n), index=idx).clip(12, 28)
    if vix_tail:
        vix.iloc[-len(vix_tail):] = vix_tail
    secs = pd.DataFrame(np.cumprod(1 + rng.normal(0.0004, 0.01, (n, 9)), axis=0) * 50, index=idx,
                        columns=L.SECTOR_ETFS)
    px = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.012, n)) * 300, index=idx)
    cor = pd.Series(20 + rng.normal(0, 3, n), index=idx)
    return {"vix": vix, "cor3m": cor, "sectors": secs, "px": px}


def run_pipeline(raw: dict, now: datetime, warnings=None) -> dict:
    """和 compute_live_signal 一樣的流程，但不連網（直接給原始資料）。"""
    orig_gather, orig_now = L.gather_live, L._now
    L.gather_live = lambda symbol="QQQ": (raw, list(warnings or []))
    L._now = lambda: now
    try:
        return L.compute_live_signal("QQQ")
    finally:
        L.gather_live, L._now = orig_gather, orig_now


# ---------------------------------------------------------------------------
# 美股交易日
# ---------------------------------------------------------------------------
def test_calendar():
    check("週二 18:00（紐約）→ 當天已收盤", L.last_closed_session(ny(2026, 10, 20, 18, 0)) == date(2026, 10, 20))
    check("週二 14:00（盤中）→ 前一個交易日", L.last_closed_session(ny(2026, 10, 20, 14, 0)) == date(2026, 10, 19))
    check("週一 10:30（盤中）→ 上週五", L.last_closed_session(ny(2026, 10, 19, 10, 30)) == date(2026, 10, 16))
    check("週日 → 週五", L.last_closed_session(ny(2026, 10, 18, 12, 0)) == date(2026, 10, 16))
    check("感恩節晚上 → 前一天", L.last_closed_session(ny(2026, 11, 26, 20, 0)) == date(2026, 11, 25))
    check("耶穌受難日 2026-04-03 休市", not L.is_session(date(2026, 4, 3)))
    check("六月節遇週六 → 2027-06-18 週五休市", not L.is_session(date(2027, 6, 18)))
    check("元旦遇週六 → 2021-12-31 照常開市", L.is_session(date(2021, 12, 31)))
    check("台灣時間早上 = 紐約前一天晚上（時區換算）",
          L.last_closed_session(datetime(2026, 10, 21, 8, 0, tzinfo=L.ZoneInfo("Asia/Taipei"))) == date(2026, 10, 20))
    check("週五到下週二之間有 2 個交易日", L.sessions_after(date(2026, 10, 16), date(2026, 10, 20)) == 2)


# ---------------------------------------------------------------------------
# Bug 1：抓不到 VIX
# ---------------------------------------------------------------------------
def test_bug1_missing_vix():
    import requests

    class Resp:
        text = "DATE,OPEN,HIGH,LOW,CLOSE\n"

        def raise_for_status(self):
            pass

    orig_get = requests.get
    requests.get = lambda *a, **k: Resp()
    try:
        try:
            L.fetch_cboe_index("VIX")
            raised = False
        except ValueError:
            raised = True
    finally:
        requests.get = orig_get
    check("CBOE 回 HTTP 200 但只有表頭 → 丟例外（才會改用備援）", raised)

    raw = raw_data("2026-10-20")
    yahoo_vix = raw["vix"].rename("^VIX")
    orig = (L.fetch_cboe_index, L.fetch_yahoo_close)

    def cboe_empty(name, timeout=20):
        raise ValueError(f"CBOE {name} 回傳空的資料")

    def yahoo_ok(tickers, period):
        if tickers == ["^VIX"]:
            return yahoo_vix.to_frame("^VIX")
        if tickers == L.SECTOR_ETFS:
            return raw["sectors"]
        return raw["px"].to_frame(tickers[0])

    def yahoo_fail(tickers, period):
        if tickers == ["^VIX"]:
            raise RuntimeError("yfinance 沒有回傳 ^VIX")
        return yahoo_ok(tickers, period)

    try:
        L.fetch_cboe_index, L.fetch_yahoo_close = cboe_empty, yahoo_ok
        out, warns = L.gather_live("QQQ")
        aligned, fr = L.align_live(out, ny(2026, 10, 20, 18, 0))
        check("CBOE 空檔 → 改用 Yahoo 的 VIX，並留下警告",
              out["vix"] is None and aligned["vix"] is not None and fr["sources"]["vix"]["via"] == "Yahoo"
              and any("改用 Yahoo" in w for w in warns))
        L.fetch_yahoo_close = yahoo_fail
        out, warns = L.gather_live("QQQ")
        check("CBOE 和 Yahoo 都失敗 → VIX 是 None，警告寫無法判斷",
              out["vix"] is None and out["vix_yahoo"] is None and any("無法判斷" in w for w in warns))

        def yahoo_zero(tickers, period):
            if tickers == ["^VIX"]:
                v = yahoo_vix.copy()
                v.iloc[-1] = 0.0
                return v.to_frame("^VIX")
            return yahoo_ok(tickers, period)
        L.fetch_yahoo_close = yahoo_zero
        out, warns = L.gather_live("QQQ")
        check("Yahoo ^VIX 最後一天是 0 → 丟掉（和 CBOE 一樣），用前一天", out["vix_yahoo"].iloc[-1] > 0
              and out["vix_yahoo"].index[-1] == yahoo_vix.index[-2])
    finally:
        L.fetch_cboe_index, L.fetch_yahoo_close = orig

    raw_none = dict(raw, vix=None)
    res = run_pipeline(raw_none, ny(2026, 10, 20, 18, 0), warnings=["VIX 抓不到（CBOE：ValueError；Yahoo：RuntimeError），無法判斷燈號"])
    check("沒有 VIX → 狀態「無法判斷」", res["state"] == L.STATE_UNKNOWN)
    plan = L.overweight_plan(res["state"], 400_000, 600_000, 40, 10, 200_000, 6)
    check("沒有 VIX → 計算器不判斷（不是「現金桶不動」）", not plan["known"] and not plan["active"])
    brief = L.format_brief(res).splitlines()
    txt = "\n".join(brief)
    check("簡報寫「無法判斷，請自己查 VIX 收盤」，並說明首次站上 40 和 126 個交易日的加碼期間",
          "無法判斷" in txt and "請自己查 VIX 收盤" in txt and "收盤首次站上 40" in txt and "126 個交易日" in txt)
    check("簡報沒有「現金桶：不動」", "現金桶：不動" not in txt)
    first_warn = next(i for i, x in enumerate(brief) if x.startswith("⚠️"))
    new_money = next(i for i, x in enumerate(brief) if "新存的錢" in x)
    check("簡報的警告在最上面（在建議之前）", first_warn < new_money and first_warn <= 2)
    check("沒有 VIX 的結果標成不完整（頁面不會快取 1 小時）", res["incomplete"])


# ---------------------------------------------------------------------------
# Bug 2：資料過期
# ---------------------------------------------------------------------------
def test_bug2_stale():
    res = run_pipeline(raw_data("2026-09-25"), ny(2026, 10, 20, 18, 0))
    fr = res["freshness"]
    check("資料停在 9/25、今天 10/20 → 每個來源都標過期",
          all(s["stale"] for s in fr["sources"].values()) and fr["sources"]["vix"]["lag"] == 17)
    check("VIX 過期 → 燈號是「資料過期」，不是「正常」", res["state"] == L.STATE_STALE)
    check("過期的結果標成不完整", res["incomplete"])
    check("警告第一條就是資料過期，並列出資料日",
          res["warnings"] and res["warnings"][0].startswith("資料過期") and "2026-09-25" in res["warnings"][0])
    txt = L.format_brief(res)
    check("簡報列出各來源資料日", "各來源資料日：VIX 2026-09-25（過期" in txt)
    check("簡報不顯示「狀態：正常」", "狀態：正常" not in txt and "資料過期" in txt)

    raw = raw_data("2026-10-20")
    raw["sectors"] = raw["sectors"].loc[:"2026-10-02"]
    res = run_pipeline(raw, ny(2026, 10, 20, 18, 0))
    check("只有產業 ETF 停在 10/2 → 燈號照 VIX，但產業 ETF 標過期、不算確認",
          res["state"] == L.STATE_NORMAL and res["sectors"].get("stale") and not res["sectors"]["confirm"])
    check("警告列出過期的產業 ETF", any("產業 ETF 停在 2026-10-02" in w for w in res["warnings"]))

    res = run_pipeline(raw_data("2026-10-16"), ny(2026, 10, 20, 9, 0))
    check("週二早上、CBOE 還停在上週五（落後 1 個交易日）→ 不算過期",
          res["state"] == L.STATE_NORMAL and not res["freshness"]["sources"]["vix"]["stale"])
    res = run_pipeline(raw_data("2026-10-15"), ny(2026, 10, 20, 18, 0))
    check("落後 3 個交易日 → 過期", res["state"] == L.STATE_STALE)


# ---------------------------------------------------------------------------
# Bug 3：研究的加碼期間
# ---------------------------------------------------------------------------
def test_bug3_window():
    idx = pd.bdate_range("2020-01-01", periods=800)
    v = pd.Series(18.0, index=idx)
    t0 = 500
    v.iloc[t0:t0 + 2] = [45, 42]        # 首次站上 40、隔天還在 40 以上
    v.iloc[t0 + 2] = 35                 # 回到 40 以下
    v.iloc[t0 + 3] = 41                 # 期間內再觸發

    def state_at(j):
        return L.capitulation_state(v.iloc[:j + 1])

    r = state_at(t0)
    check("觸發當天 → 全面恐慌", r["state"] == L.STATE_PANIC and r["vix"]["window"]["remaining"] == 125)
    r = state_at(t0 + 1)
    check("隔天（VIX 42）→ 加碼期間，不是再一次全面恐慌", r["state"] == L.STATE_HOLD)
    r = state_at(t0 + 2)
    check("VIX 回到 35 → 還在加碼期間（不是「留意：先不動」）", r["state"] == L.STATE_HOLD)
    r = state_at(t0 + 3)
    w = r["vix"]["window"]
    check("期間內再觸發 → 從新的觸發日重新起算 126 天，並記得這段期間的開始",
          w["trigger"] == idx[t0 + 3].date().isoformat() and w["since"] == idx[t0].date().isoformat()
          and w["remaining"] == 125)
    check("期間內再觸發那天 → 還是「加碼期間」，不是再一次「全面恐慌：買」",
          r["state"] == L.STATE_HOLD and "重新起算" in L.state_headline(r) and "不是再買一次" in L.state_headline(r))
    check("簡報在再觸發那天也說不是再買一次", "期間重新起算，不是再買一次" in L.format_brief(dict(r, evidence={})))
    r = state_at(t0 + 3 + 125)
    check("新觸發日起第 126 個交易日 → 最後一天，還在期間內",
          r["state"] == L.STATE_HOLD and r["vix"]["window"]["remaining"] == 0)
    r = state_at(t0 + 3 + 126)
    check("第 127 個交易日 → 期間結束，回到正常", r["state"] == L.STATE_NORMAL)
    hold = L.capitulation_state(v.iloc[:t0 + 2])
    txt = L.format_brief(dict(hold, evidence={}))
    check("簡報寫出觸發日和還剩幾個交易日", f"觸發日 {idx[t0].date()}" in txt and "還剩 124 個交易日" in txt)

    v2 = pd.Series(18.0, index=idx)
    v2.iloc[300:300 + 150] = 45          # 連續 150 天都在 40 以上（期間 126 天後結束）
    check("VIX 一直在 40 以上但期間已經過了 → 不再是加碼期間（留意）",
          L.capitulation_state(v2.iloc[:300 + 130])["state"] == L.STATE_WATCH)

    # 和研究完全一致：research/run_robustness2.py 的 overlay 是 edge(VIX ≥ 40).rolling(126).max()
    path = ROOT / "research" / "raw" / "cboe" / "VIX_History.csv"
    gspc = ROOT / "research" / "raw" / "prices" / "IDX_GSPC.csv"
    if not (path.exists() and gspc.exists()):
        print("  SKIP 沒有 research/raw（要先跑 research/fetch_data.py）：略過和研究的逐日比對")
        return
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    vix = pd.Series(pd.to_numeric(df["CLOSE"], errors="coerce").values,
                    index=pd.to_datetime(df["DATE"], format="%m/%d/%Y")).dropna()
    vix = vix[vix > 0].sort_index()
    # 研究的做法（research/lib/data.py _cboe_index_file）：對到 ^GSPC 的交易日曆、最多往前補 3 天
    cal = pd.to_datetime(pd.read_csv(gspc, index_col=0).index).normalize()
    cal = cal[(cal >= vix.index[0]) & (cal <= vix.index[-1])].unique()
    rv = vix.reindex(cal).ffill(limit=3)
    b = rv >= 40
    edge = b & ~b.shift(1, fill_value=False)
    mask = edge.astype(float).rolling(L.HOLD_DAYS, min_periods=1).max() > 0
    live = L.closed_only(vix, vix.index[-1].date())          # 即時版：只留美股交易日
    days = set(range(0, len(cal), 25))
    for i in np.flatnonzero(edge.values):
        days.update(range(max(i - 3, 0), min(i + L.HOLD_DAYS + 5, len(cal))))
    bad = [cal[j].date() for j in sorted(days)
           if L.panic_window(live.loc[:cal[j]])["active"] != bool(mask.iloc[j])]
    check(f"和研究 overlay 的加碼期間逐日一致（研究的交易日曆，{len(days)} 天，1990 起）{bad[:3] if bad else ''}",
          not bad)
    check("加碼期間約占 16.9% 的交易日（研究 time_tilted = 0.1685）", abs(mask.mean() - 0.1685) < 0.002)


# ---------------------------------------------------------------------------
# Bug 4：加碼計算器的實際比例
# ---------------------------------------------------------------------------
def test_bug4_calculator():
    # 平常 40%：100 萬裡 40 萬股票。恐慌時股票跌 30% → 28 萬股票 + 60 萬現金
    p = L.overweight_plan(L.STATE_PANIC, 280_000, 600_000, 40, 10, 200_000, 6)
    check("目前比例 = 28 / 88 = 31.8%", abs(p["current_pct"] - 280 / 880 * 100) < 1e-9)
    check("買到目標 50%：88 萬 × 50% − 28 萬 = 16 萬", abs(p["amount"] - 160_000) < 1e-6)
    check("買完的實際比例 = 50.0%", abs(p["after_pct"] - 50.0) < 1e-9 and not p["limited_by_cash"])
    check("其中 7.2 萬是補回平常 40%", abs(p["to_base"] - (0.4 * 880_000 - 280_000)) < 1e-6)
    p = L.overweight_plan(L.STATE_PANIC, 280_000, 600_000, 40, 10, 100_000, 6)
    check("現金桶只有 10 萬 → 被卡住，實際只到 43.2%",
          p["limited_by_cash"] and abs(p["after_pct"] - 380 / 880 * 100) < 1e-9)
    p = L.overweight_plan(L.STATE_HOLD, 600_000, 400_000, 40, 10, 200_000, 6)
    check("已經在目標以上（60%）→ 不用買", p["amount"] == 0 and p["active"])
    p = L.overweight_plan(L.STATE_PANIC, 280_000, 50_000, 90, 10, 200_000, 6)
    check("目標 100%、現金桶比手上現金多 → 最多買光手上的現金（比例 100%）",
          p["amount"] == 50_000 and abs(p["after_pct"] - 100) < 1e-9)
    p = L.overweight_plan(L.STATE_PANIC, 280_000, 600_000, 40, 10, 159_999, 6)
    check("只差 1 元到目標 → 不說「被卡住」（顯示 50.0%）", not p["limited_by_cash"])
    p = L.overweight_plan(L.STATE_PANIC, 280_000, 600_000, 40, 10, 0, 6)
    check("現金桶是 0 → 買 0，但不是「已經達到目標」", p["amount"] == 0 and not p["at_target"] and p["limited_by_cash"])
    p = L.overweight_plan(L.STATE_PANIC, 0, 0, 40, 10, 200_000, 6)
    check("市值和現金都是 0 → 不會除以 0", p["current_pct"] is None and p["amount"] == 0)


# ---------------------------------------------------------------------------
# Bug 5：只用已收盤的日線、對齊同一個資料日
# ---------------------------------------------------------------------------
def test_bug5_closed_bars():
    raw = raw_data("2026-10-19")               # Yahoo 有週一（10/19）的盤中 K 線
    raw["vix"] = raw["vix"].loc[:"2026-10-16"]  # CBOE 的 VIX 只到上週五
    res = run_pipeline(raw, ny(2026, 10, 19, 10, 30))
    s = res["series"]
    check("週一盤中：ETF 和 QQQ 的週一 K 線被丟掉，全部對齊到上週五",
          res["data_date"] == "2026-10-16" and s["sectors"].index[-1] == pd.Timestamp("2026-10-16")
          and s["px"].index[-1] == pd.Timestamp("2026-10-16"))

    raw = raw_data("2026-10-19", vix_tail=[20, 21, 45])   # Yahoo 備援的 ^VIX 盤中衝到 45
    res = run_pipeline(raw, ny(2026, 10, 19, 11, 0))
    check("盤中 VIX 衝上 45 → 不亮全面恐慌（盤中 K 線不算）",
          res["state"] == L.STATE_NORMAL and res["vix"]["value"] == 21)
    res = run_pipeline(raw, ny(2026, 10, 19, 17, 0))
    check("同一根 K 線收盤後（17:00）→ 全面恐慌", res["state"] == L.STATE_PANIC)

    raw = raw_data("2026-10-19")
    raw["vix"] = raw["vix"].loc[:"2026-10-16"]
    res = run_pipeline(raw, ny(2026, 10, 19, 20, 0))
    check("收盤後 CBOE 還沒更新：其他來源只用到 VIX 的資料日（不混用週一）",
          res["data_date"] == "2026-10-16" and res["series"]["px"].index[-1] == pd.Timestamp("2026-10-16")
          and not res["freshness"]["sources"]["vix"]["stale"])

    idx = pd.date_range("2026-10-13", periods=5, freq="B", tz="America/New_York")
    s = L.closed_only(pd.Series([1.0, 2, 3, 4, 5], index=idx), date(2026, 10, 16))
    check("有時區的日線（紐約）也能正確丟掉還沒收盤的那天", list(s.index.date) == [date(2026, 10, d) for d in (13, 14, 15, 16)])
    utc = pd.DatetimeIndex(["2026-10-16 20:00", "2026-10-19 14:30"], tz="UTC")
    s = L.closed_only(pd.Series([1.0, 2.0], index=utc), date(2026, 10, 16))
    check("UTC 時間戳記先換成紐約日期再判斷", len(s) == 1 and s.index[0] == pd.Timestamp("2026-10-16"))


# ---------------------------------------------------------------------------
# 小問題
# ---------------------------------------------------------------------------
def test_minor_sectors_and_params():
    raw = raw_data("2026-10-20")
    secs = raw["sectors"].copy()
    secs.iloc[-1, :5] = np.nan
    ss = L.sector_status(secs)
    check("最後一天 5 檔缺值 → 分母是 4 檔，缺的列出來",
          ss["of"] == 4 and ss["missing"] == L.SECTOR_ETFS[:5] and ss["count"] <= 4)
    res = L.capitulation_state(raw["vix"], secs)
    check("簡報顯示「x/4 檔（缺 …）」", "/4 檔（缺 XLB、XLE、XLF、XLI、XLK）" in L.format_brief(dict(res, evidence={})))

    p = L.clean_entry_params({"base_equity_pct": 120, "tilt_pp": -5, "cash": "abc", "hold_months": 0,
                              "stock_value": 1e30, "cash_bucket": float("nan"), "holdings_date": "not a date"})
    check("超出範圍的值截到範圍內",
          p["base_equity_pct"] == 100 and p["tilt_pp"] == 0 and p["hold_months"] == 1
          and p["stock_value"] == L._MONEY_MAX)
    check("型別不對的值用預設", p["cash"] == L.DEFAULT_ENTRY_PARAMS["cash"]
          and p["cash_bucket"] == L.DEFAULT_ENTRY_PARAMS["cash_bucket"] and p["holdings_date"] is None)
    check("不是 dict（例如 list）→ 全部用預設", L.clean_entry_params([1, 2]) == L.DEFAULT_ENTRY_PARAMS)
    old = L.clean_entry_params({"total_assets": 1_000_000, "base_equity_pct": 40, "tilt_pp": 10})
    check("舊版參數檔（只有總資產）→ 依平常比例拆成股票 40 萬、現金 60 萬，日期留空",
          old["stock_value"] == 400_000 and old["cash"] == 600_000 and old["holdings_date"] is None)


# ---------------------------------------------------------------------------
# 第二輪（獨立驗證找到的問題）
# ---------------------------------------------------------------------------
def test_round2():
    # 休市日：CBOE 檔裡有休市日的夜盤報價，不能拿來觸發
    check("MLK 紀念日 1998 年以前照常開市", L.is_session(date(1990, 1, 15)) and not L.is_session(date(2026, 1, 19)))
    check("臨時休市（2025-01-09 國喪日）不是交易日", not L.is_session(date(2025, 1, 9)))
    raw = raw_data("2026-09-09")
    v = raw["vix"].copy()
    v.loc[pd.Timestamp("2026-09-04")] = 39.0
    v.loc[pd.Timestamp("2026-09-07")] = 41.0          # 勞動節（休市）的夜盤報價
    v.loc[pd.Timestamp("2026-09-08")] = 39.0
    v.loc[pd.Timestamp("2026-09-09")] = 37.0
    raw["vix"] = v.sort_index()
    res = run_pipeline(raw, ny(2026, 9, 9, 18, 0))
    check("休市日的 VIX 41 不會打開加碼期間", res["state"] == L.STATE_WATCH and not res["vix"]["window"]["active"])

    # CBOE 收盤後要幾個小時才更新：Yahoo 已經有今天 ≥ 40 的收盤時要用上
    raw = raw_data("2026-10-20", vix_tail=[30.0, 41.0])
    yv = raw["vix"].copy()
    raw["vix"] = raw["vix"].loc[:"2026-10-19"]
    raw["vix_yahoo"] = yv
    res = run_pipeline(raw, ny(2026, 10, 20, 18, 0))
    check("CBOE 還停在昨天、Yahoo 有今天 41 → 全面恐慌，資料日是今天",
          res["state"] == L.STATE_PANIC and res["data_date"] == "2026-10-20"
          and res["freshness"]["sources"]["vix"]["via"] == "CBOE+Yahoo" and bool(res["notes"]))
    raw = raw_data("2026-10-20")
    yv = raw["vix"].copy()
    raw["vix"] = raw["vix"].loc[:"2026-10-13"]
    raw["vix_yahoo"] = yv
    res = run_pipeline(raw, ny(2026, 10, 20, 18, 0))
    check("CBOE 過期但 Yahoo 是新的 → 用 Yahoo 補上，不算過期", res["state"] == L.STATE_NORMAL
          and not res["freshness"]["sources"]["vix"]["stale"])
    raw = raw_data("2026-10-20")
    raw["vix"] = raw["vix"].loc[:"2026-10-19"]
    res = run_pipeline(raw, ny(2026, 10, 20, 18, 0))
    check("VIX 落後 1 天（兩邊都還沒更新）→ 警告、結果不完整（頁面只留 5 分鐘）",
          any("VIX 還沒更新到 2026-10-20" in w for w in res["warnings"]) and res["incomplete"])
    check("VIX 落後 1 天 → 燈號說明和簡報都講清楚今天的收盤還沒進來",
          "還沒有 2026-10-20 的收盤" in L.state_headline(res) and "還沒有 2026-10-20 的收盤" in L.format_brief(res))

    # VIX 過期但加碼期間一定還沒結束
    raw = raw_data("2026-10-20")
    vv = raw["vix"].loc[:"2026-10-14"].copy()           # 停在 10/14，落後 4 個交易日
    vv.iloc[-4] = 30.0
    vv.iloc[-3:] = 45.0                                   # 10/12 首次站上 40
    raw["vix"] = vv
    res = run_pipeline(raw, ny(2026, 10, 20, 18, 0))
    check("VIX 過期但加碼期間剩下的天數 > 落後的天數 → 還是「加碼期間」，並標出 VIX 過期",
          res["state"] == L.STATE_HOLD and res["vix_stale"] and "一定還在加碼期間" in L.state_headline(res))
    w = res["vix"]["window"]
    check("VIX 過期時，簡報寫「至少還剩」（扣掉落後的天數）",
          f"至少還剩 {w['remaining'] - w['stale_lag']} 個交易日" in L.format_brief(res))
    check("VIX 過期時，其他新的來源不會被說成過期", not res["freshness"]["sources"]["px"]["stale"]
          and res["freshness"]["sources"]["px"]["date"] == "2026-10-20")
    raw = raw_data("2026-10-20")
    vv = raw["vix"].loc[:"2026-10-14"].copy()
    sess = [t for t in vv.index if L.is_session(t.date())]   # 合成資料的工作日裡有休市日，要用真正的交易日數
    vv.loc[sess[-127]] = 30.0
    vv.loc[sess[-126]] = 45.0                             # 觸發日在 125 個交易日前，期間只剩 10/14 當天
    raw["vix"] = vv
    res = run_pipeline(raw, ny(2026, 10, 20, 18, 0))
    check("VIX 過期而且加碼期間可能已經結束 → 資料過期，說明可能已結束",
          res["state"] == L.STATE_STALE and "可能已經結束" in L.state_headline(res))

    # 期間結束、VIX 還在 40 以上、接近 40
    idx = pd.bdate_range(end="2026-10-14", periods=400)
    v2 = pd.Series(18.0, index=idx)
    v2.iloc[-140:] = 45.0
    r = L.capitulation_state(v2)
    h = L.state_headline(r)
    check("VIX 一直在 40 以上但期間已結束 → 不說「還差 0.0 點」，說不是新的觸發",
          "還差 0.0" not in h and "不是新的觸發" in h and "。。" not in h)
    v3 = pd.Series(18.0, index=idx)
    v3.iloc[-1] = 39.97
    check("VIX 39.97 不會顯示成 40.0", "39.97" in L.state_headline(L.capitulation_state(v3)))
    v4 = pd.Series(18.0, index=idx)
    v4.iloc[-130] = 45.0                                  # 期間在 4 個交易日前結束
    r = L.capitulation_state(v4)
    check("加碼期間剛結束 → 提醒把股票比例調回平常", "調回平常" in L.state_headline(r)
          and "調回平常" in L.format_brief(dict(r, evidence={})))

    # 10 個月均線：月底最後一個交易日收盤後就算這個月
    days = pd.bdate_range("2024-01-02", "2026-10-30")
    px = pd.Series(np.linspace(100, 200, len(days)), index=days)
    check("資料到 2026-10-30（10 月最後一個交易日）→ 用 10 月", L.ten_month_trend(px)["month"] == "2026-10")
    check("資料到 2026-10-29 → 10 月還沒收完，用 9 月", L.ten_month_trend(px.loc[:"2026-10-29"])["month"] == "2026-09")

    # 產業 ETF：中間缺一天不影響分母；最後一天的計數和分母一致
    raw = raw_data("2026-10-20")
    secs = raw["sectors"].copy()
    secs.iloc[-200, 0] = np.nan
    check("一年內中間缺一天的 ETF 照樣算進分母", L.sector_status(secs)["of"] == 9)
    secs = raw["sectors"].copy()
    secs.iloc[-1, 1:] = np.nan
    ss = L.sector_status(secs)
    check("最後一天只有 1 檔有資料 → 計數不超過分母", ss["of"] == 1 and ss["count"] <= 1)
    raw = raw_data("2026-10-20")
    raw["sectors"] = raw["sectors"].copy()
    raw["sectors"]["XLB"] = np.nan
    res = run_pipeline(raw, ny(2026, 10, 20, 18, 0), warnings=["產業 ETF 缺少：XLB"])
    check("產業 ETF 有缺 → 結果不完整（不快取 1 小時）", res["incomplete"])

    future = L.clean_entry_params({"holdings_date": "2099-01-01"})
    check("參數檔的日期在未來 → 當成沒填", future["holdings_date"] is None)


# ---------------------------------------------------------------------------
# 目前等級（第 8 頁最上方）
# ---------------------------------------------------------------------------
def test_levels():
    idx = pd.bdate_range(end="2026-10-20", periods=400)

    def lvl(v):
        return L.market_level(L.capitulation_state(v))["level"]

    v = pd.Series(18.0, index=idx)
    check("VIX 18 → 第 1 級", lvl(v) == 1)
    v.iloc[-1] = 25
    check("VIX 25 → 第 2 級", lvl(v) == 2)
    v.iloc[-1] = 31
    check("VIX 31（不在加碼期間）→ 第 3 級", lvl(v) == 3)
    v.iloc[-1] = 45
    check("今天首次站上 40 → 第 4 級", lvl(v) == 4)
    v2 = pd.Series(18.0, index=idx)
    v2.iloc[-5] = 45
    check("加碼期間中（VIX 已回到 18）→ 第 4 級", lvl(v2) == 4)
    v3 = pd.Series(18.0, index=idx)
    v3.iloc[-140:] = 45
    check("VIX 一直在 40 以上但加碼期間已結束 → 第 3 級", lvl(v3) == 3)
    check("抓不到 VIX → 0 級（無法分級）",
          L.market_level(L.capitulation_state(pd.Series(dtype=float)))["level"] == 0)
    res = run_pipeline(raw_data("2026-09-25"), ny(2026, 10, 20, 18, 0))
    check("VIX 過期 → 0 級（無法分級）", L.market_level(res)["level"] == 0)

    ev = L.load_evidence()
    lv = (ev.get("vix_levels") or {}).get("levels", [])
    check("證據檔有 4 個等級的歷史統計，占比加起來是 100%",
          [x["level"] for x in lv] == [1, 2, 3, 4] and abs(sum(x["share"] for x in lv) - 1) < 0.002)
    check("等級越高，歷史上之後 3 個月的平均越好（第 1、2 級接近，第 4 級最好）",
          lv[3]["mean_63"] > lv[2]["mean_63"] > max(lv[0]["mean_63"], lv[1]["mean_63"]))
    r = L.capitulation_state(v2)
    r["evidence"] = ev
    ps = L.plain_summary(r)
    check("白話結論：第 4 級、寫出歷史表現", ps["level"] == 4 and ps["title"].startswith("第 4 級")
          and "歷史上（1990 起）第 4 級之後 3 個月" in (ps["history"] or ""))
    check("簡報第一段寫出目前等級", "目前等級：第 4 級（共 4 級）：全面恐慌／加碼期間" in L.format_brief(r))
    r0 = L.capitulation_state(pd.Series(dtype=float))
    check("無法分級時白話結論不寫歷史表現", L.plain_summary(r0)["history"] is None)


# ---------------------------------------------------------------------------
# 出場參考（第 8 頁右上）
# ---------------------------------------------------------------------------
def test_exit_summary():
    ev = L.load_evidence()
    er = ev.get("exit_rules") or {}
    check("證據檔有出場規則的總結：測了 40 個、通過 0 個", er.get("n_tested") == 40 and er.get("n_passed") == 0
          and (er.get("vix_high_riskoff") or {}).get("alpha", 0) < 0)
    days = pd.bdate_range("2023-01-02", "2026-10-20")
    up = pd.Series(np.linspace(100, 200, len(days)), index=days)
    down = pd.Series(np.linspace(200, 100, len(days)), index=days)
    calm = pd.Series(18.0, index=days)

    def ex(vix, px):
        r = L.capitulation_state(vix, px=px)
        r["evidence"] = ev
        return L.exit_summary(r)

    e = ex(calm, up)
    txt = " ".join(e["items"])
    check("平常、在 10 個月均線之上 → 沒有要照規則減碼的部位", e["title"] == "沒有要照規則減碼的部位")
    check("出場參考寫出均線位置、它的歷史代價、40 個規則通過 0 個、VIX ≥ 30 就出場最差",
          "在均線之上" in txt and "一直持有" in txt and "研究測了 40 個出場規則，通過 0 個" in txt
          and "「VIX ≥ 30 就出場」每年比固定比例少 4.8%" in txt)
    check("跌破 10 個月均線 → 標題說只有選它當保險才出場",
          ex(calm, down)["title"] == "10 個月均線跌破：只有你選它當保險才出場")
    hold = calm.copy()
    hold.iloc[-5] = 45.0
    e = ex(hold, down)
    check("加碼期間中（即使均線跌破）→ 先不調回", e["title"] == "加碼期間中：先不調回" and "約到" in e["items"][0])
    ended = calm.copy()
    ended.iloc[-130] = 45.0
    check("加碼期間剛結束 → 照手冊調回平常比例", ex(ended, up)["title"] == "加碼期間已結束：照手冊調回平常比例")
    r = L.capitulation_state(pd.Series(dtype=float), px=up)
    r["evidence"] = ev
    check("抓不到 VIX → 資料有問題，先不做任何調整", L.exit_summary(r)["title"] == "資料有問題，先不做任何調整")
    r = L.capitulation_state(calm)
    check("沒有證據檔、沒有價格也不會壞", L.exit_summary(r)["title"] == "沒有要照規則減碼的部位")


# ---------------------------------------------------------------------------
# 每週訊號紀錄（scripts/log_signals.py）走同一套流程
# ---------------------------------------------------------------------------
def test_log_signals():
    import tempfile
    sys.path.insert(0, str(ROOT / "scripts"))
    import log_signals as G
    orig_gather, orig_now = L.gather_live, L._now
    try:
        raw = raw_data("2026-10-19", vix_tail=[20, 45])
        raw["vix"] = raw["vix"].loc[:"2026-10-16"]
        L.gather_live = lambda symbol="QQQ": (raw, [])
        L._now = lambda: ny(2026, 10, 19, 20, 0)
        row = G.build_row("QQQ")
        check("紀錄的資料日對齊到 VIX（上週五），收盤價也是上週五的", row["data_date"] == "2026-10-16"
              and row["close"] == round(float(raw["px"].loc["2026-10-16"]), 2))
        check("紀錄的欄位都在 COLUMNS 裡", set(row) == set(G.COLUMNS))
        with tempfile.TemporaryDirectory() as d:
            df = G.append_row(row, Path(d) / "log.csv")
            check("寫入暫存檔（不碰 cache/signal_log.csv）", len(df) == 1 and (Path(d) / "log.csv").exists())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.csv"
            G.append_row(row, path)
            stale_row = dict(row, state=L.STATE_STALE, stale_sources="VIX 2026-10-16")
            df = G.append_row(stale_row, path)
            check("同一個資料日：過期的紀錄不會蓋掉原本準時的紀錄", len(df) == 1 and df["state"].iloc[0] == row["state"])
        L._now = lambda: ny(2026, 10, 23, 18, 0)        # 一週後 VIX 還停在 10/16
        try:
            G.build_row("QQQ")
            raised = False
        except RuntimeError:
            raised = True
        check("VIX 過期 → 不寫紀錄", raised)
        L._now = lambda: ny(2026, 10, 19, 20, 0)
        L.gather_live = lambda symbol="QQQ": (dict(raw, vix=None), ["VIX 抓不到"])
        try:
            G.build_row("QQQ")
            raised = False
        except RuntimeError:
            raised = True
        check("VIX 抓不到 → 整個失敗，不寫半套紀錄", raised)
    finally:
        L.gather_live, L._now = orig_gather, orig_now


if __name__ == "__main__":
    for f in [test_calendar, test_bug1_missing_vix, test_bug2_stale, test_bug3_window, test_bug4_calculator,
              test_bug5_closed_bars, test_minor_sectors_and_params, test_round2, test_levels, test_exit_summary,
              test_log_signals]:
        print(f.__name__)
        f()
    print("\n全部通過" if not FAILS else f"\n失敗：{FAILS}")
    sys.exit(1 if FAILS else 0)
