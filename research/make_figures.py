"""報告用的圖（PNG）：research/results/figs/
    PYTHONUTF8=1 .venv/Scripts/python research/make_figures.py
需要先跑 run_study.py、run_reserve.py（選定變體）、run_robustness*.py。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib import backtest as BT  # noqa: E402
from lib import data as D  # noqa: E402
from lib.inputs import Bundle  # noqa: E402
from lib.signals import REGISTRY, drawdown, edge  # noqa: E402

R = ROOT / "results"
FIG = R / "figs"
FIG.mkdir(exist_ok=True)

# 參考色票（dataviz 的預設，淺色模式）
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
POS, NEG = "#2a78d6", "#e34948"
GRAYBAR = "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "Segoe UI", "Segoe UI Symbol"], "font.size": 10, "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", color=INK, fontsize=12, fontweight="bold", pad=28 if sub else 10)
    if sub:
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                    color=INK2, fontsize=9, va="bottom")


def fig_overview():
    """每個訊號家族：所有參數變體觸發後 3 個月比平常多（少）賺多少（NDX，全期）。"""
    e = pd.read_csv(R / "entry_results.csv")
    v = pd.read_csv(R / "verdicts.csv")
    f = e[(e.asset == "NDX") & (e.window == "full")]
    titles = {s.family: s.title for s in REGISTRY}
    g = f.groupby("family")["h63_excess"].agg(["min", "max", "median"]).sort_values("median")
    # 觀察名單：NDX 全期未校正 p < 0.05，且樣本外與 S&P 500 方向一致（都還沒通過多重檢定）
    w = v[(v.kind == "entry")].merge(f[["variant", "h63_p"]], on="variant")
    passed = set(w[(w["h63_p"] < 0.05) & (w["excess_oos"] > 0) & (w["excess_spx"] > 0)]["family"])
    fig, ax = plt.subplots(figsize=(9, 9.5))
    y = np.arange(len(g))
    for i, (fam, row) in enumerate(g.iterrows()):
        c = S1 if fam in passed else GRAYBAR
        ax.plot([row["min"] * 100, row["max"] * 100], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter([row["median"] * 100], [i], color=c, s=40, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.axvline(0, color=BASE, lw=1)
    lab = [titles.get(fam, fam) for fam in g.index]
    lab = [l if len(l) <= 26 else l[:25] + "…" for l in lab]
    ax.set_yticks(y, lab, fontsize=8.5)
    ax.set_xlabel("觸發後 3 個月報酬 − 平常任一天買進的 3 個月報酬（百分點）")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], color=S1, s=40, label="觀察名單（單獨看顯著、方向一致，但沒通過多重檢定）")
    ax.scatter([], [], color=GRAYBAR, s=40, label="其他")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    title(ax, "進場訊號：觸發後 3 個月的超額報酬（NDX，1985–2026）",
          "線 = 同一個訊號不同參數的範圍；點 = 中位數。66 個變體沒有一個通過多重檢定（q ≤ 0.10）")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_entry_overview.png", dpi=150)
    plt.close(fig)


def era_table() -> pd.DataFrame:
    B = Bundle()
    spx = B.px("SPX")
    r = B.tr("SPX")
    rv = r.rolling(21, min_periods=21).std() * np.sqrt(252) * 100
    ind = D.french_industries(49, "value")
    lev = (1 + ind.fillna(0)).cumprod().where(ind.notna())
    lo = lev.rolling(252, min_periods=252).min()
    valid = lo.notna() & lev.notna()
    fl = ((lev <= lo) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    sigs = {"已實現波動 ≥ 30%\n（VIX 的長歷史替代）": edge(rv >= 30, rv),
            "已實現波動 ≥ 40%": edge(rv >= 40, rv),
            "49 產業中 ≥ 30%\n同時創一年新低": edge(fl >= 0.3, fl),
            "S&P 500 距高點跌 10%": edge(drawdown(spx) <= -0.10, spx)}
    eras = [("1928–1945", "1928-01-01", "1945-12-31"), ("1946–1989", "1946-01-01", "1989-12-31"),
            ("1990–2026", "1990-01-01", None)]
    rows = []
    for name, t in sigs.items():
        for en, a, b in eras:
            es = BT.event_study(t, spx, (63,), 21, a, b, 1000, seed=1)
            rows.append({"signal": name, "era": en, "n": es.get("n_events", 0),
                         "x63": es.get(63, {}).get("excess", np.nan), "p": es.get(63, {}).get("p", np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(R / "eras.csv", index=False, encoding="utf-8-sig")
    return df


def fig_eras():
    df = era_table()
    sigs = list(dict.fromkeys(df["signal"]))
    eras = list(dict.fromkeys(df["era"]))
    cols = [S1, S2, S3]
    fig, ax = plt.subplots(figsize=(9, 6.0))
    h = 0.24
    for j, era in enumerate(eras):
        for i, s in enumerate(sigs):
            row = df[(df.signal == s) & (df.era == era)].iloc[0]
            yy = i + (j - 1) * (h + 0.03)
            val = row["x63"] * 100
            shown = max(val, -7.6)
            ax.barh(yy, shown, height=h, color=cols[j], label=era if i == 0 else None)
            txt = f"{val:+.1f}（{int(row['n'])} 次）" + ("，超出圖框" if val < -7.6 else "") +                 ("，次數太少" if row["n"] < 5 else "")
            if shown < 0:
                ax.text(0.25, yy, txt, va="center", ha="left", fontsize=8, color=INK2)
            else:
                ax.text(shown + 0.25, yy, txt, va="center", ha="left", fontsize=8, color=INK2)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(range(len(sigs)), sigs)
    ax.invert_yaxis()
    ax.set_xlim(-8, 11)
    ax.set_xlabel("觸發後 3 個月超額報酬（百分點，S&P 500）")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9, title="年代", title_fontsize=9)
    title(ax, "同一個規則放到不同年代（S&P 500）",
          "波動率恐慌各年代都是正的、但都不顯著；產業同時創新低在大蕭條期間反向；跌 10% 各年代都接近 0")
    fig.tight_layout()
    fig.subplots_adjust(left=0.24)
    fig.savefig(FIG / "fig2_eras.png", dpi=150)
    plt.close(fig)


def fig_decay():
    t = pd.read_csv(R / "timing_results.csv")
    items = [("fomc_even_weeks", "NDX", "FOMC 週期（偶數週持有）\n論文 2014"),
             ("pre_fomc", "NDX", "FOMC 聲明前漂移\n論文 2011"),
             ("turn_of_month", "SPX", "月底月初效應\n論文 1987"),
             ("halloween", "NDX", "萬聖節效應（Sell in May）\n論文 2002"),
             ("trend_10m", "SPX", "10 個月均線擇時\n論文 2007"),
             ("abs_momentum", "SPX", "絕對動能\n論文 2014")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (v, a, lab) in enumerate(items):
        pre = t[(t.variant == v) & (t.asset == a) & (t.window == "pre_pub")]["alpha_ann"].iloc[0] * 100
        post = t[(t.variant == v) & (t.asset == a) & (t.window == "post_pub")]["alpha_ann"].iloc[0] * 100
        ax.plot([pre, post], [i, i], color=BASE, lw=2, zorder=1)
        ax.scatter([pre], [i], color=S1, s=70, zorder=3, edgecolor=SURF, linewidth=2, label="發表前" if i == 0 else None)
        ax.scatter([post], [i], color=S2, s=70, zorder=3, edgecolor=SURF, linewidth=2, label="發表後" if i == 0 else None)
        ax.text(pre, i - 0.32, f"{pre:+.1f}", ha="center", fontsize=8, color=INK2)
        ax.text(post, i - 0.32, f"{post:+.1f}", ha="center", fontsize=8, color=INK2)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(range(len(items)), [x[2] for x in items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("每年擇時 alpha（百分點；相對同樣平均持股比例的固定配置，扣成本）")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    title(ax, "已發表的擇時規則，發表後幾乎都失效", "同一條規則在論文發表年之前與之後的表現")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_publication_decay.png", dpi=150)
    plt.close(fig)


def fig_reserve():
    d = pd.read_csv(R / "reserve_results_selected.csv")
    d = d[(d.asset == "NDX") & (d.config == 0) & (d.window == "full")].copy()
    names = {"vix_level[rule=vix>=40]": "VIX ≥ 40", "vix_level[rule=vix>=30]": "VIX ≥ 30",
             "dd_tranche[level=0.1]": "距高點跌 10%", "dd_tranche[level=0.2]": "距高點跌 20%",
             "breadth_thrust[low=0.2,high=0.5]": "產業廣度回升", "cor_high[pct=95]": "隱含相關性高檔",
             "composite_fear[th=75]": "現行恐慌綜合分數 ≥ 75", "aaii_low[rule=spread_pct5]": "AAII 極度悲觀"}
    d["name"] = d["variant"].map(names)
    d = d.sort_values("ratio")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(d))
    ax.barh(y, (d["ratio"] - 1) * 100, height=0.55, color=S1, label="等訊號再買")
    ax.scatter((d["null_median"] - 1) * 100, y, color=INK, s=28, zorder=3, marker="|", linewidths=2,
               label="隨機時間點部署（中位數）")
    for yy, val in zip(y, (d["ratio"] - 1) * 100):
        ax.text(0.12, yy, f"{val:.1f}%", va="center", ha="left", fontsize=8.5, color=INK)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(y, d["name"])
    ax.set_xlim(-6.5, 1.0)
    ax.set_xlabel("期末財富 相對「每月直接全部買」（%；NDX，1990–2026）")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    title(ax, "每月存的錢留一半等訊號：全部都輸給直接買",
          "50% 馬上買、50% 放國庫券等訊號，最多等 12 個月；0 = 和直接買一樣")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_reserve.png", dpi=150)
    plt.close(fig)


def fig_vix40():
    B = Bundle()
    ndx = B.px("NDX")
    vix = B.cboe("VIX")
    ev = BT.episode_starts(edge(vix >= 40, vix).dropna().astype(bool), 21)
    f63 = BT.forward_returns(ndx, 63)
    base = f63.loc["1990":].mean()
    vals = f63.reindex(ev) * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.6, color=[POS if v >= 0 else NEG for v in vals])
    ax.axhline(base * 100, color=INK2, lw=1.2, ls="--")
    ax.set_xlim(-0.6, len(vals) + 2.2)
    ax.text(len(vals) - 0.3, base * 100, f"平常任一天買進\n平均 {base * 100:+.1f}%", ha="left", va="center",
            fontsize=8.5, color=INK2)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(x, [d.strftime("%Y-%m") for d in ev], rotation=60, fontsize=8)
    ax.set_ylabel("NDX 之後 3 個月報酬（%）")
    ax.grid(axis="x", visible=False)
    title(ax, "VIX 收盤首次站上 40 之後的 3 個月（NDX，1990–2026）",
          f"{len(vals)} 次觸發，分屬 11 群（相隔 3 個月以上算不同群）；隔天收盤進場")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_vix40_events.png", dpi=150)
    plt.close(fig)


def fig_csp():
    """賣 put 研究（run_csp.py）：NDX 賣 put 指數 vs 持有 NDX，以及恐慌時賣 put vs 直接買。"""
    if not (R / "csp_panic.csv").exists():
        return
    from lib import options as O
    cal = pd.read_csv(R / "csp_calibration.csv")
    hc = float(cal["calibrated_haircut"].iloc[0])
    rf = (D.fred_raw("DTB3").reindex(D.calendar()).ffill() / 100).shift(1)
    ndx = D.close("^NDX").dropna()
    vxn = D._raw_price("^VXN")["Close"].reindex(D.calendar()) / 100
    tr = D.total_return("NDX")
    start = vxn.dropna().index[0]
    navs = {"持有 NDX": (1 + tr.loc[start:]).cumprod(),
            "每月賣 ATM put": O.putwrite_index(ndx, vxn, rf, otm=0.0, skew_slope=0.4, haircut=hc, q=0.007,
                                            start=start)["nav"],
            "每月賣 5% 價外 put": O.putwrite_index(ndx, vxn, rf, otm=0.05, skew_slope=0.4, haircut=hc, q=0.007,
                                               start=start)["nav"]}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for (name, s), col, nudge in zip(navs.items(), (S1, S2, S3), (1.0, 1.08, 0.88)):
        s = s / s.iloc[0]
        ax.plot(s.index, s.values, color=col, lw=1.6, label=name)
        ax.text(s.index[-1], s.iloc[-1] * nudge, f" {s.iloc[-1]:.1f} 倍", va="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylabel("淨值（起點 = 1，對數刻度）")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xlim(right=ax.get_xlim()[1] + 900)
    title(ax, "當成一檔基金：賣 put 降低回撤，但報酬也比較低（NDX，2001–2026）",
          f"權利金用 VXN 套 Black-Scholes，扣 {hc:.0%}（這個比例讓同樣方法對上 CBOE 官方 PUT 指數）")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_csp_fund.png", dpi=150)
    plt.close(fig)

    p = pd.read_csv(R / "csp_panic.csv")
    p = p[(p["haircut"] == hc) & (p["otm"] == 0.0)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(p))
    vals = (p["csp_minus_buy"] * 100).values
    ax.bar(x, vals, width=0.6, color=[POS if v >= 0 else NEG for v in vals])
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(x, [d[:7] for d in p["date"]], rotation=60, fontsize=8)
    ax.set_ylabel("賣 put − 直接買（半年後，每 1 元）%")
    ax.grid(axis="x", visible=False)
    title(ax, "VIX 首次 ≥ 40 時：賣 ATM put 平均比直接買差",
          f"藍 = 賣 put 比較好（{int((vals > 0).sum())}/{len(vals)} 次）；紅 = 直接買比較好。V 型反彈時賣 put 會錯過大部分漲幅")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_csp_panic.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_overview()
    fig_eras()
    fig_decay()
    fig_reserve()
    fig_vix40()
    fig_csp()
    print("ok:", sorted(p.name for p in FIG.glob("*.png")))
