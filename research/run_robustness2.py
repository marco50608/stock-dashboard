"""第二輪通過者（產業同時創新低）的穩健性檢查，以及「固定比例 + 恐慌時暫時加碼」的用法評估。

穩健性（和 run_robustness.py 同一套，事先寫好）：
  A. 1926–1959（沒用過的年代）  B. 門檻鄰近值  C. 晚一天進場  D. 9 檔產業 ETF 的即時版本
加碼評估（overlay）：平常持有 50% 股票 / 50% 現金；訊號觸發後把股票拉到 70%，維持 N 天再回到 50%。
  和「同樣平均曝險的固定比例」比較（timing alpha），含 5 bps 成本。
輸出 results/robustness2.csv、results/overlay.csv
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
from lib.signals import edge  # noqa: E402

H = (21, 63, 126, 252)
rows = []


def es(label, check, trig, px, start=None, end=None, entry_lag=1):
    r = BT.event_study(trig, px, H, 21, start, end, 2000, seed=len(rows), entry_lag=entry_lag)
    row = {"check": check, "label": label, "n_events": r.get("n_events", 0), "start": start, "end": end}
    for h in H:
        if h in r:
            row[f"x{h}"] = r[h]["excess"]
            row[f"p{h}"] = r[h]["p"]
            row[f"hit{h}"] = r[h]["hit"]
            row[f"hitbase{h}"] = r[h]["hit_base"]
    rows.append(row)
    print(f"  [{check}] {label}: n={row['n_events']} x63={row.get('x63', np.nan) * 100:+.1f}% "
          f"p63={row.get('p63', np.nan):.3f} x126={row.get('x126', np.nan) * 100:+.1f}%", flush=True)


def frac_at_low(lev: pd.DataFrame, n: int = 252) -> pd.Series:
    lo = lev.rolling(n, min_periods=n).min()
    valid = lo.notna() & lev.notna()
    return ((lev <= lo) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)


def main():
    B = Bundle()
    ndx, spx = B.px("NDX"), B.px("SPX")
    ind = D.french_industries(49, "value")
    lev_all = (1 + ind.fillna(0)).cumprod().where(ind.notna())
    f_all = frac_at_low(lev_all)
    f = frac_at_low(B.industries())

    # A. 沒用過的年代
    for th in (0.3, 0.5):
        es(f"產業新低 ≥ {th}（1927–1959）", "A_untouched", edge(f_all >= th, f_all), spx, "1927-07-01", "1959-12-31")
    # B. 鄰近值
    for th in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        es(f"NDX 產業新低 ≥ {th}", "B_neighbors", edge(f >= th, f), ndx)
        es(f"SPX 產業新低 ≥ {th}（1960 起）", "B_neighbors", edge(f >= th, f), spx, "1960-01-01")
    # C. 晚一天
    es("NDX 產業新低 ≥ 0.3（D+2）", "C_delay", edge(f >= 0.3, f), ndx, entry_lag=2)
    # D. 即時版本：9 檔產業 SPDR 同時創一年新低的比例
    secs = pd.DataFrame({t: B.close(t) for t in ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]})
    fs = frac_at_low(secs)
    for th in (0.22, 0.33, 0.44):     # 2、3、4 檔
        es(f"NDX 9 檔 ETF 同時新低 ≥ {round(th * 9)} 檔", "D_live_proxy", edge(fs >= th, fs), ndx, "1999-12-01")
        es(f"SPX 9 檔 ETF 同時新低 ≥ {round(th * 9)} 檔", "D_live_proxy", edge(fs >= th, fs), spx, "1999-12-01")
    es("NDX 49 產業新低 ≥ 0.3（1999-12 起，對照）", "D_live_proxy", edge(f >= 0.3, f), ndx, "1999-12-01")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "robustness2.csv", index=False, encoding="utf-8-sig")

    # 觸發日清單
    ev = BT.episode_starts(edge(f >= 0.3, f).dropna().astype(bool), 21)
    f63 = BT.forward_returns(ndx, 63)
    f126 = BT.forward_returns(ndx, 126)
    print(pd.DataFrame({"NDX_63d": f63.reindex(ev) * 100, "NDX_126d": f126.reindex(ev) * 100}).round(1).to_string())

    # -------- 加碼用法 --------
    rf = B.rf()
    vix = B.cboe("VIX")
    sigs = {
        "VIX ≥ 40": edge(vix >= 40, vix),
        "VIX ≥ 35": edge(vix >= 35, vix),
        "產業新低 ≥ 30%": edge(f >= 0.3, f),
        "距高點跌 10%": edge(ndx / ndx.cummax() - 1 <= -0.10, ndx),
        "恐慌綜合分數 ≥ 75（現行）": None,
    }
    from lib.signals import composite_fear_score
    cf = composite_fear_score(B, "NDX")
    sigs["恐慌綜合分數 ≥ 75（現行）"] = edge(cf >= 75, cf)
    out = []
    for name, trig in sigs.items():
        for hold in (63, 126):
            for asset in ("NDX", "SPX"):
                ret = B.tr(asset)
                t = trig.reindex(ret.index)
                active = t.fillna(0).rolling(hold, min_periods=1).max()
                pos = (0.5 + 0.2 * active).where(t.notna())
                for wname, (a, b) in {"full": (None, None), "IS": (None, "2012-12-31"), "OOS": ("2013-01-01", None)}.items():
                    r = BT.run_timing(ret, pos, rf, 5.0, start=a, end=b)
                    s = r["strat"]
                    base = BT.run_timing(ret, pd.Series(0.5, index=pos.index).where(pos.notna()), rf, 5.0, start=a, end=b)["strat"]
                    out.append({"signal": name, "hold": hold, "asset": asset, "window": wname, "start": s["start"],
                                "time_tilted": float((r["df"]["pos"] > 0.55).mean()),
                                "alpha_ann": r["alpha_ann"], "alpha_t": r["alpha_t"],
                                "CAGR": s["CAGR"], "CAGR_static50": base["CAGR"], "MaxDD": s["MaxDD"],
                                "MaxDD_static50": base["MaxDD"], "Sharpe": s["Sharpe"], "Sharpe_static50": base["Sharpe"]})
            print(f"  overlay {name} hold={hold}", flush=True)
    od = pd.DataFrame(out)
    od.to_csv(ROOT / "results" / "overlay.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    x = od[od["window"] == "full"].copy()
    for c in ("time_tilted", "alpha_ann", "CAGR", "CAGR_static50", "MaxDD", "MaxDD_static50"):
        x[c] = (x[c] * 100).round(2)
    print(x[["signal", "hold", "asset", "start", "time_tilted", "alpha_ann", "alpha_t", "CAGR", "CAGR_static50",
             "MaxDD", "MaxDD_static50", "Sharpe", "Sharpe_static50"]].round(2).to_string(index=False))
    y = od[od["window"] != "full"][["signal", "hold", "asset", "window", "alpha_ann", "alpha_t"]].copy()
    y["alpha_ann"] = (y["alpha_ann"] * 100).round(2)
    print(y.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
