"""通過（和差一點通過）的進場訊號的穩健性檢查。檢查項目在看結果之前寫好：

A. 沒用過的年代：S&P 500 1928–1989（VIX 還不存在，用 21 日已實現波動率代替）；
   產業廣度用 1926–1959（主研究從 1960 開始）
B. 參數鄰近值：好的訊號應該是「一片高原」，不是「一根針」
C. 多晚一天進場（D+2 收盤）
D. 可即時計算的版本：產業廣度改用 9 檔產業 SPDR ETF（1999 起）
E. 其他市場：美國 VIX 飆高後，已開發（EFA）與新興市場（EEM）
輸出 results/robustness.csv
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
from lib.inputs import Bundle  # noqa: E402
from lib.signals import drawdown, edge, frac_above_sma, rpct  # noqa: E402

H = (21, 63, 126, 252)
rows = []


def es(label, check, trig, px, start=None, end=None, entry_lag=1, cooldown=21):
    r = BT.event_study(trig, px, H, cooldown, start, end, 2000, seed=len(rows), entry_lag=entry_lag)
    row = {"check": check, "label": label, "n_events": r.get("n_events", 0),
           "start": start, "end": end}
    for h in H:
        if h in r:
            row[f"x{h}"] = r[h]["excess"]
            row[f"p{h}"] = r[h]["p"]
            row[f"hit{h}"] = r[h]["hit"]
            row[f"hitbase{h}"] = r[h]["hit_base"]
            row[f"worst{h}"] = r[h]["worst"]
    rows.append(row)
    print(f"  [{check}] {label}: n={row['n_events']} x63={row.get('x63', np.nan) * 100:+.1f}% p63={row.get('p63', np.nan):.3f}",
          flush=True)


def main():
    B = Bundle()
    ndx, spx = B.px("NDX"), B.px("SPX")
    vix = B.cboe("VIX")

    # ---------------- A. 沒用過的年代 ----------------
    r_spx = B.tr("SPX")
    rv = r_spx.rolling(21, min_periods=21).std() * np.sqrt(252) * 100
    for th in (30, 40):
        trig = edge(rv >= th, rv)
        es(f"SPX 已實現波動 ≥ {th}%（1928–1989）", "A_untouched", trig, spx, "1928-01-01", "1989-12-31")
        es(f"SPX 已實現波動 ≥ {th}%（1990–2026，對照）", "A_untouched", trig, spx, "1990-01-01", None)
    # 用 1990–2012 對照 VIX≥40 和 RV 的對應關係（只用樣本內）
    v40 = edge(vix >= 40, vix)
    ev = BT.episode_starts(v40.dropna().astype(bool), 21)
    ev = ev[ev <= "2012-12-31"]
    print("  VIX≥40 觸發時的已實現波動（1990–2012）中位數：", round(float(rv.reindex(ev).median()), 1))
    for lvl in (0.10, 0.20):
        trig = edge(drawdown(spx) <= -lvl, spx)
        es(f"SPX 距高點跌 {int(lvl * 100)}%（1928–1989）", "A_untouched", trig, spx, "1928-01-01", "1989-12-31")
    # 產業廣度 1926–1959
    ind = D.french_industries(49, "value")
    lev = (1 + ind.fillna(0)).cumprod().where(ind.notna())
    f50 = frac_above_sma(lev, 50)
    for low, high in ((0.1, 0.5), (0.1, 0.7), (0.2, 0.5), (0.2, 0.7)):
        washed = (f50 <= low).rolling(15, min_periods=1).max().astype(bool)
        trig = edge(washed & (f50 >= high), f50)
        es(f"產業廣度回升 low={low} high={high}（1926–1959）", "A_untouched", trig, spx, "1926-09-01", "1959-12-31")

    # ---------------- B. 參數鄰近值 ----------------
    for th in (30, 35, 37.5, 40, 42.5, 45, 50):
        es(f"NDX VIX ≥ {th}", "B_neighbors", edge(vix >= th, vix), ndx)
        es(f"SPX VIX ≥ {th}", "B_neighbors", edge(vix >= th, vix), spx)
    for lvl in (0.08, 0.10, 0.12):
        es(f"NDX 距歷史高點跌 {int(lvl * 100)}%", "B_neighbors", edge(drawdown(ndx) <= -lvl, ndx), ndx)
        dd52 = ndx / ndx.rolling(252, min_periods=60).max() - 1
        es(f"NDX 距 52 週高點跌 {int(lvl * 100)}%（儀表板的算法）", "B_neighbors", edge(dd52 <= -lvl, dd52), ndx)
    f = frac_above_sma(B.industries(), 50)
    for low in (0.15, 0.20, 0.25):
        for high in (0.45, 0.5, 0.55, 0.6, 0.7):
            washed = (f <= low).rolling(15, min_periods=1).max().astype(bool)
            trig = edge(washed & (f >= high), f)
            es(f"NDX 產業廣度 low={low} high={high}", "B_neighbors", trig, ndx)
            es(f"SPX 產業廣度 low={low} high={high}", "B_neighbors", trig, spx, "1960-01-01")

    # ---------------- C. 晚一天進場 ----------------
    es("NDX VIX ≥ 40（D+2 收盤進場）", "C_delay", edge(vix >= 40, vix), ndx, entry_lag=2)
    es("NDX 距高點跌 10%（D+2）", "C_delay", edge(drawdown(ndx) <= -0.10, ndx), ndx, entry_lag=2)
    washed = (f <= 0.2).rolling(15, min_periods=1).max().astype(bool)
    es("NDX 產業廣度 0.2/0.5（D+2）", "C_delay", edge(washed & (f >= 0.5), f), ndx, entry_lag=2)

    # ---------------- D. 可即時計算：9 檔產業 SPDR ----------------
    secs = pd.DataFrame({t: B.close(t) for t in ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]})
    fs = frac_above_sma(secs, 50)
    for low, high in ((0.2, 0.5), (0.2, 0.7), (0.12, 0.55), (0.25, 0.75)):
        washed = (fs <= low).rolling(15, min_periods=1).max().astype(bool)
        trig = edge(washed & (fs >= high), fs)
        es(f"NDX 9 檔產業 ETF 廣度 low={low} high={high}", "D_live_proxy", trig, ndx, "1999-10-01")
        es(f"SPX 9 檔產業 ETF 廣度 low={low} high={high}", "D_live_proxy", trig, spx, "1999-10-01")
    # 同期間 French 49 產業版本對照
    washed = (f <= 0.2).rolling(15, min_periods=1).max().astype(bool)
    es("NDX 49 產業廣度 0.2/0.5（1999 起，對照）", "D_live_proxy", edge(washed & (f >= 0.5), f), ndx, "1999-10-01")
    both = pd.concat([f, fs], axis=1).dropna()
    print("  49 產業 vs 9 檔 ETF 的 50 日線廣度相關係數：", round(float(both.corr().iloc[0, 1]), 2))

    # ---------------- E. 其他市場 ----------------
    for tk in ("EFA", "EEM", "IWM"):
        p = (1 + B.close(tk).pct_change()).cumprod().dropna()
        es(f"{tk}：VIX ≥ 40", "E_other_mkts", edge(vix >= 40, vix).reindex(p.index), p)
        es(f"{tk}：VIX ≥ 30", "E_other_mkts", edge(vix >= 30, vix).reindex(p.index), p)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "robustness.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
