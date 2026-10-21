"""cash-secured put（賣 put 等著被指派）對 QQQ 累積者的回測。

    PYTHONUTF8=1 .venv/Scripts/python research/run_csp.py

步驟（設定在跑之前寫死）：
  1. 校準：用同一套方法（Black-Scholes + VIX，每月第一個交易日賣 ATM put）重建 S&P 500 賣 put 指數，
     和 CBOE 官方 PUT 指數比較。方法夠準，才拿來估 NDX（VXN）。
  2. 當成一檔基金：NDX 賣 put 指數（ATM、5% 價外） vs 直接持有 NDX。
  3. 累積者的做法：每月存 1 元，直接買 vs 新錢拿去賣 put 等被指派（可設「最多等 3 個月」）。
  4. 恐慌時（VIX ≥ 40）：1 元現金直接買 vs 賣 put，比較半年後的價值。
基準設定：成交只拿到理論權利金的 90%（haircut 10%），價外偏斜 0.4（5% 價外 +2 個波動點），NDX 股息率 0.7%。
敏感度：haircut 5% / 20%，偏斜 0 / 0.8。
輸出 results/csp_*.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib import backtest as BT  # noqa: E402
from lib import data as D  # noqa: E402
from lib import options as O  # noqa: E402
from lib.signals import edge  # noqa: E402

R = ROOT / "results"
BASE = {"haircut": 0.10, "skew_slope": 0.4}
WINDOWS = {"全期": (None, None), "2001–2012": (None, "2012-12-31"), "2013–2026": ("2013-01-01", None)}


def rf_annual() -> pd.Series:
    s = D.fred_raw("DTB3").reindex(D.calendar()).ffill() / 100
    return s.shift(1)                     # 前一天就知道的利率


def main():
    rf = rf_annual()

    # ---------------- 1. 校準（S&P 500 vs CBOE PUT） ----------------
    spx = D.close("^GSPC").dropna()
    vix = D.cboe_index("VIX") / 100
    put = D._raw_price("^PUT")["Close"].reindex(D.calendar()).dropna()
    rows = []
    for hc in (0.0, 0.05, 0.10):
        nav = O.putwrite_index(spx, vix, rf, otm=0.0, skew_slope=0.0, haircut=hc, q=0.018,
                               start=put.index[0])["nav"]
        common = nav.index.intersection(put.index)
        a = nav.reindex(common)
        b = put.reindex(common)
        ma = a.resample("ME").last().pct_change().dropna()
        mb = b.resample("ME").last().pct_change().dropna()
        st_a, st_b = O.ann_stats(a), O.ann_stats(b)
        rows.append({"haircut": hc, "model_CAGR": st_a["CAGR"], "PUT_CAGR": st_b["CAGR"],
                     "model_vol": st_a["vol"], "PUT_vol": st_b["vol"], "model_MaxDD": st_a["MaxDD"],
                     "PUT_MaxDD": st_b["MaxDD"], "monthly_corr": float(ma.corr(mb)),
                     "start": st_a["start"], "end": st_a["end"]})
    # 反推：讓模型年化報酬對上 CBOE PUT 指數的成本比例（只校準「定價偏高」這一個參數）
    put_cagr = O.ann_stats(put)["CAGR"]
    best_hc, best_gap = None, 9e9
    for hc in np.arange(0.0, 0.41, 0.01):
        nav = O.putwrite_index(spx, vix, rf, otm=0.0, skew_slope=0.0, haircut=float(hc), q=0.018,
                               start=put.index[0])["nav"]
        nav = nav.reindex(nav.index.intersection(put.index))
        gap = abs(O.ann_stats(nav)["CAGR"] - put_cagr)
        if gap < best_gap:
            best_hc, best_gap = float(round(hc, 2)), gap
    cal = pd.DataFrame(rows)
    cal["calibrated_haircut"] = best_hc
    cal.to_csv(R / "csp_calibration.csv", index=False, encoding="utf-8-sig")
    print("== 校準：模型 vs CBOE PUT 指數")
    print(cal.round(4).to_string(index=False))
    print(f"讓模型對上 PUT 指數的成本比例：{best_hc:.2f}（之後的 NDX 結果同時報 0.10 與這個「校準後」設定）")
    HCS = [BASE["haircut"], best_hc]

    # ---------------- NDX 輸入 ----------------
    ndx = D.close("^NDX").dropna()
    vxn = D._raw_price("^VXN")["Close"].reindex(D.calendar()) / 100
    tr = D.total_return("NDX")
    start = vxn.dropna().index[0]

    # ---------------- 2. 當成一檔基金 ----------------
    rows = []
    bh_all = (1 + tr.loc[start:]).cumprod()
    for wname, (a, b) in WINDOWS.items():
        for otm in (0.0, 0.05):
            for hc in sorted({0.05, 0.10, 0.20, best_hc}):
                nav = O.putwrite_index(ndx, vxn, rf, otm=otm, skew_slope=BASE["skew_slope"], haircut=hc, q=0.007,
                                       start=a or start, end=b)["nav"]
                st = O.ann_stats(nav)
                bh = O.ann_stats(bh_all.loc[nav.index[0]:nav.index[-1]])
                rows.append({"window": wname, "otm": otm, "haircut": hc, "CAGR": st["CAGR"], "vol": st["vol"],
                             "MaxDD": st["MaxDD"], "bh_CAGR": bh["CAGR"], "bh_vol": bh["vol"], "bh_MaxDD": bh["MaxDD"],
                             "start": st["start"], "end": st["end"]})
    fund = pd.DataFrame(rows)
    fund.to_csv(R / "csp_fund.csv", index=False, encoding="utf-8-sig")
    print("\n== NDX 賣 put 指數 vs 持有 NDX")
    print(fund.round(4).to_string(index=False))

    # ---------------- 3. 累積者：每月存 1 元 ----------------
    rows = []
    variants = [(otm, mw) for otm in (0.0, 0.05) for mw in (None, 3)]
    for wname, (a, b) in WINDOWS.items():
        for otm, mw in variants:
            for hc in HCS:
                res = O.accumulate_csp(ndx, tr, vxn, rf, otm=otm, max_wait=mw, q=0.007, start=a or start, end=b,
                                       haircut=hc, skew_slope=BASE["skew_slope"])
                rows.append({"window": wname, "otm": otm, "max_wait": mw or "不限", "haircut": hc,
                             "skew_slope": BASE["skew_slope"], **res})
    # 敏感度（全期）
    for otm, mw in variants:
        for hc, sk in ((0.05, 0.4), (0.20, 0.4), (0.10, 0.0), (0.10, 0.8)):
            res = O.accumulate_csp(ndx, tr, vxn, rf, otm=otm, max_wait=mw, q=0.007, start=start,
                                   haircut=hc, skew_slope=sk)
            rows.append({"window": "全期（敏感度）", "otm": otm, "max_wait": mw or "不限", "haircut": hc,
                         "skew_slope": sk, **res})
    # 滾動 10 年（每年 1 月開始）
    for y in range(start.year + 1, 2017):
        for otm, mw in variants:
            for hc in HCS:
                res = O.accumulate_csp(ndx, tr, vxn, rf, otm=otm, max_wait=mw, q=0.007, start=f"{y}-01-01",
                                       end=f"{y + 9}-12-31", haircut=hc, skew_slope=BASE["skew_slope"])
                rows.append({"window": f"10 年：{y}–{y + 9}", "otm": otm, "max_wait": mw or "不限", "haircut": hc,
                             "skew_slope": BASE["skew_slope"], **res})
    acc = pd.DataFrame(rows)
    acc.to_csv(R / "csp_accumulate.csv", index=False, encoding="utf-8-sig")
    print("\n== 累積者：賣 put 等指派 vs 直接買（期末財富比值）")
    main_rows = acc[~acc["window"].str.startswith("10 年")]
    print(main_rows[["window", "otm", "max_wait", "haircut", "skew_slope", "ratio", "assign_rate", "avg_cash_share"]]
          .round(4).to_string(index=False))
    roll = acc[acc["window"].str.startswith("10 年")]
    summ = roll.groupby(["haircut", "otm", "max_wait"])["ratio"].agg(
        windows="count", beat_share=lambda x: float((x > 1).mean()), median="median", worst="min", best="max")
    print("\n滾動 10 年窗：")
    print(summ.round(4).to_string())

    # ---------------- 4. 恐慌時（VIX ≥ 40） ----------------
    v = D.cboe_index("VIX")
    trig = BT.episode_starts(edge(v >= 40, v).dropna().astype(bool), 21)
    trig = trig[trig >= start]
    rows = []
    for otm in (0.0, 0.05):
        for hc in HCS:
            p = O.panic_csp_vs_buy(ndx, tr, vxn, rf, trig, horizon=126, otm=otm, q=0.007, haircut=hc,
                                   skew_slope=BASE["skew_slope"])
            p["otm"] = otm
            p["haircut"] = hc
            rows.append(p)
    pan = pd.concat(rows)
    pan["csp_minus_buy"] = pan["csp"] - pan["buy"]
    pan.to_csv(R / "csp_panic.csv", index=False, encoding="utf-8-sig")
    print("\n== 恐慌時（VIX 首次 ≥ 40）1 元現金：半年後價值")
    print(pan[pan["haircut"] == BASE["haircut"]].round(4).to_string(index=False))
    print(pan.groupby(["haircut", "otm"])[["buy", "csp", "csp_minus_buy"]].agg(["mean", "median"]).round(4).to_string())
    print(pan.groupby(["haircut", "otm"])["csp_minus_buy"].apply(
        lambda x: f"賣 put 比較好的次數 {(x > 0).sum()}/{len(x)}").to_string())


if __name__ == "__main__":
    main()
