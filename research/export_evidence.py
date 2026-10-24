"""把研究結論匯出成儀表板用的證據檔：utils/signal_evidence.json

    PYTHONUTF8=1 .venv/Scripts/python research/export_evidence.py

儀表板（pages/8_進出場訊號.py、scripts/daily_brief.py）只讀這個 JSON，不需要 research/ 的原始資料。
研究重跑後再執行一次，證據就會更新。
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib import backtest as BT  # noqa: E402
from lib.inputs import Bundle  # noqa: E402
from lib.signals import edge  # noqa: E402

R = ROOT / "results"
OUT = ROOT.parent / "utils" / "signal_evidence.json"


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def event_rows(trig: pd.Series, B: Bundle, start=None) -> list[dict]:
    ndx, spx = B.px("NDX"), B.px("SPX")
    ev = BT.episode_starts(trig.dropna().astype(bool), 21)
    if start is not None:
        ev = ev[ev >= pd.Timestamp(start)]
    f = {h: BT.forward_returns(ndx, h) for h in (21, 63, 126)}
    s63 = BT.forward_returns(spx, 63)
    m63 = BT.forward_min(ndx, 63)
    rows = []
    for d in ev:
        rows.append({"date": d.date().isoformat(),
                     "ndx_21": r(f[21].get(d)), "ndx_63": r(f[63].get(d)), "ndx_126": r(f[126].get(d)),
                     "spx_63": r(s63.get(d)), "ndx_min_63": r(m63.get(d))})
    return rows


def entry_summary(e: pd.DataFrame, v: pd.DataFrame, variant: str) -> dict:
    g = e[e.variant == variant].set_index(["asset", "window"])
    vv = v.set_index("variant").loc[variant]

    def G(a, w, c):
        try:
            return r(g.loc[(a, w), c])
        except KeyError:
            return None
    return {"variant": variant, "n_events": G("NDX", "full", "h63_n"), "n_clusters": G("NDX", "full", "h63_n_clusters"),
            "mean_63": G("NDX", "full", "h63_mean"), "base_63": G("NDX", "full", "h63_base"),
            "excess_21": G("NDX", "full", "h21_excess"), "excess_63": G("NDX", "full", "h63_excess"),
            "excess_126": G("NDX", "full", "h126_excess"),
            "hit_63": G("NDX", "full", "h63_hit"), "hit_base_63": G("NDX", "full", "h63_hit_base"),
            "p": G("NDX", "full", "h63_p"), "q": r(vv["q"]),
            "excess_is": G("NDX", "IS", "h63_excess"), "excess_oos": G("NDX", "OOS", "h63_excess"),
            "spx_excess_63": G("SPX", "full", "h63_excess"), "spx_p": G("SPX", "full", "h63_p"),
            "worst_63": G("NDX", "full", "h63_worst")}


def vix_levels(B: Bundle) -> dict:
    """第 8 頁「目前等級」的歷史對照：每天依 VIX 分成 4 級，之後 3、6 個月 NDX 的表現（隔天收盤買進）。

    第 4 級 = 加碼期間（VIX 收盤首次站上 40 後 126 個交易日，和 overlay 研究相同）；
    第 3 級 = VIX ≥ 30 但不在加碼期間；第 2 級 = 20 ≤ VIX < 30；第 1 級 = VIX < 20。
    這是描述性統計：每天的觀察值彼此重疊，第 4 級只來自約 9 次危機，沒有經過多重檢定。
    分級的門檻要和 utils/signals_live.py 的 market_level() 一致。
    """
    vix = B.cboe("VIX")
    ndx = B.px("NDX")
    win = edge(vix >= 40, vix).fillna(0).rolling(126, min_periods=1).max() > 0
    lvl = pd.Series(np.select([win, vix >= 30, vix >= 20], [4, 3, 2], default=1), index=vix.index).where(vix.notna())
    f63 = BT.forward_returns(ndx, 63).reindex(vix.index)
    f126 = BT.forward_returns(ndx, 126).reindex(vix.index)
    df = pd.DataFrame({"lvl": lvl, "f63": f63, "f126": f126}).dropna(subset=["lvl", "f63"])

    def stats(d: pd.DataFrame) -> dict:
        return {"days": int(len(d)), "share": r(len(d) / len(df)), "mean_63": r(d.f63.mean()),
                "median_63": r(d.f63.median()), "hit_63": r((d.f63 > 0).mean()), "p05_63": r(d.f63.quantile(0.05)),
                "worst_63": r(d.f63.min()), "mean_126": r(d.f126.mean()),
                "mean_63_pre2013": r(d[d.index < "2013-01-01"].f63.mean()),
                "mean_63_post2013": r(d[d.index >= "2013-01-01"].f63.mean())}

    levels = []
    for k in (1, 2, 3, 4):
        m = (lvl == k).reindex(df.index).fillna(False)
        episodes = int((m & ~m.shift(1, fill_value=False)).sum())
        levels.append({"level": k, "episodes": episodes, **stats(df[m])})
    return {"start": df.index[0].date().isoformat(), "end": df.index[-1].date().isoformat(), "asset": "NDX",
            "levels": levels, "all": stats(df)}


def exit_rules(v: pd.DataFrame) -> dict:
    """40 個出場／擇時規則的總結：通過幾個、看起來最好的、和「VIX ≥ 30 就出場」。alpha = 每年比固定比例多（少）多少。"""
    tm = v[v.kind == "timing"].sort_values("alpha_full", ascending=False)

    def row(x):
        return {"variant": x.variant, "title": x.title, "alpha": r(x.alpha_full), "t": r(x.alpha_t, 2),
                "alpha_oos": r(x.alpha_oos)}
    vh = tm[tm.variant == "vix_high_riskoff"]
    return {"n_tested": int(len(tm)), "n_passed": int((tm.verdict == "通過").sum()),
            "best": row(tm.iloc[0]), "worst": row(tm.iloc[-1]),
            "vix_high_riskoff": row(vh.iloc[0]) if len(vh) else None}


def main():
    B = Bundle()
    e = pd.read_csv(R / "entry_results.csv")
    v = pd.read_csv(R / "verdicts.csv")
    t = pd.read_csv(R / "timing_results.csv")
    res = pd.read_csv(R / "reserve_results_selected.csv")
    ov = pd.read_csv(R / "overlay.csv")

    vix = B.cboe("VIX")
    secs = pd.DataFrame({tk: B.close(tk) for tk in ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]})
    lo = secs.rolling(252, min_periods=252).min()
    n_low = ((secs <= lo) & lo.notna()).sum(axis=1).where(lo.notna().all(axis=1))

    t10 = t[(t.variant == "trend_10m") & (t.asset == "NDX")].set_index("window")
    out = {
        "generated": date.today().isoformat(),
        "data_end": B.px("NDX").index[-1].date().isoformat(),
        "n_variants_tested": int(len(v)),
        "n_entry_variants": int((v.kind == "entry").sum()),
        "n_timing_variants": int((v.kind == "timing").sum()),
        "n_passed": int((v.verdict == "通過").sum()),
        "best_entry_q": r(v[v.kind == "entry"]["q"].min(), 3),
        "vix40": {**entry_summary(e, v, "vix_level[rule=vix>=40]"),
                  "events": event_rows(edge(vix >= 40, vix), B)},
        "vix30": entry_summary(e, v, "vix_level[rule=vix>=30]"),
        "cor3m95": entry_summary(e, v, "cor_high[pct=95]"),
        "sector_lows4": {"events": event_rows(edge(n_low >= 4, n_low), B, start="1999-12-01")},
        "vix_levels": vix_levels(B),
        "reserve_ndx": [
            {"variant": row.variant, "ratio_minus_1": r(row.ratio - 1), "random_median_minus_1": r(row.null_median - 1),
             "avg_wait_days": r(row.avg_wait_days, 0)}
            for row in res[(res.asset == "NDX") & (res.config == 0) & (res.window == "full")].itertuples()],
        "overlay_vix40_ndx": {k: r(ov[(ov.signal == "VIX ≥ 40") & (ov.hold == 126) & (ov.asset == "NDX") &
                                     (ov.window == "full")][k].iloc[0]) for k in ("alpha_ann", "alpha_t", "time_tilted")},
        "trend_10m_ndx": {"start": t10.loc["full", "s_start"], "cagr": r(t10.loc["full", "s_CAGR"]),
                          "bh_cagr": r(t10.loc["full", "bh_CAGR"]), "matched_cagr": r(t10.loc["full", "m_CAGR"]),
                          "maxdd": r(t10.loc["full", "s_MaxDD"]), "bh_maxdd": r(t10.loc["full", "bh_MaxDD"]),
                          "matched_maxdd": r(t10.loc["full", "m_MaxDD"]), "oos_alpha": r(t10.loc["OOS", "alpha_ann"])},
        # 出場／擇時規則的總結（和同樣平均持股的固定比例相比，每年多或少多少）
        "exit_rules": exit_rules(v),
        # 已驗證沒有用（或證據太弱）的指標：一句話原因。數字取自 results/verdicts.csv
        "rejected": [],
    }
    picks = [
        ("equity_pc_high[ma=10,pct=90]", "股票 Put/Call 偏高", "最常見的反向指標；之後 3 個月和平常差不多或更差"),
        ("aaii_low[rule=spread_pct5]", "AAII 散戶極度悲觀", "方向是正的，但不顯著，S&P 500 上更弱"),
        ("naaim_low[rule=<=30]", "NAAIM 經理人曝險 ≤ 30", "幾乎沒有效果"),
        ("vix_term_inv[th=1.0]", "VIX 期限結構倒掛", "倒掛當下買，之後和平常一樣；倒掛時出場反而更差"),
        ("composite_fear[th=75]", "儀表板舊的恐慌綜合分數 ≥ 75", "1986–2026 沒有優勢，只有 2013 年後是正的"),
        ("cot_extreme[rule=ES_lev_pct5]", "COT 槓桿基金極空", "不顯著，walk-forward 不成立"),
        ("insider_buy[pct=95]", "SEC 內部人買進比例高檔", "2006–2026 大盤層級不顯著"),
        ("fed_put", "Fed put（FOMC 前大跌）", "反向：之後 3 個月比平常差"),
        ("dd_tranche[level=0.1]", "距高點跌 10%（舊手冊的分批）", "只對 NDX 有效，換到 S&P 500 任何年代都沒效果"),
        ("breadth_thrust[low=0.2,high=0.5]", "產業廣度急速回升", "1926–1959 反向，即時版本弱"),
    ]
    ve = v.set_index("variant")
    f = e[(e.asset == "NDX") & (e.window == "full")].set_index("variant")
    for var, name, why in picks:
        out["rejected"].append({"variant": var, "name": name, "why": why,
                                "excess_63": r(f.loc[var, "h63_excess"]), "q": r(ve.loc[var, "q"], 2)})
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"寫入 {OUT}（VIX≥40 事件 {len(out['vix40']['events'])} 筆、產業 ETF 新低事件 {len(out['sector_lows4']['events'])} 筆）")


if __name__ == "__main__":
    main()
