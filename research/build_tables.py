"""把 verdicts / entry / timing 結果整理成報告用的 Markdown 表格 → results/tables.md"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib.signals import REGISTRY  # noqa: E402

R = ROOT / "results"


def pct(x, d=1):
    return "—" if pd.isna(x) else f"{x * 100:+.{d}f}%"


def main():
    v = pd.read_csv(R / "verdicts.csv")
    e = pd.read_csv(R / "entry_results.csv")
    t = pd.read_csv(R / "timing_results.csv")
    spec = {s.family: s for s in REGISTRY}
    out = []

    # ---------------- 進場 ----------------
    after = {"vix_level": "觀察名單（最強）：1990 年前用已實現波動代替，各年代方向一致為正但不顯著",
             "cor_high": "觀察名單：2006 年才有資料，無法做長歷史測試",
             "industry_lows": "觀察名單（弱）：1928–1945 反向；只算每次危機第一次觸發時不顯著",
             "dd_tranche": "觀察名單（弱）：只對 NDX 有效，S&P 500 長歷史無效",
             "breadth_thrust": "觀察名單（弱）：1926–1959 反向、即時版本弱"}
    f = e[(e.asset == "NDX") & (e.window == "full")].set_index("variant")
    out.append("| 訊號 | 類別 | 冷門度 | 變體數 | 代表變體（p 最小）3 個月超額（NDX 全期） | 未校正 p | q 值 | 樣本外 2013+ | S&P 500 | 判定 | 備註／未通過原因 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    ve = v[v.kind == "entry"].copy()
    ve["p"] = ve["variant"].map(f["h63_p"])
    rows = []
    for fam, g in ve.groupby("family", sort=False):
        best = g.sort_values("p").iloc[0]
        verdicts = g["verdict"].value_counts()
        if "通過" in verdicts:
            vd = f"{verdicts['通過']}/{len(g)} 通過"
        elif verdicts.get("證據不足", 0) == len(g):
            vd = "證據不足"
        else:
            vd = "淘汰"
        reason = best["reason"] if isinstance(best["reason"], str) and best["reason"] else "—"
        note = after.get(fam)
        rows.append((best["p"], f"| {spec[fam].title} | {spec[fam].category} | {spec[fam].niche} | {len(g)} | "
                     f"{pct(best['excess_full'])}（{int(best['n_full']) if pd.notna(best['n_full']) else 0} 次） | "
                     f"{best['p']:.3f} | {best['q']:.2f} | {pct(best['excess_oos'])} | {pct(best['excess_spx'])} | {vd} | "
                     f"{note + '；' if note else ''}{reason} |"))
    for _, line in sorted(rows, key=lambda x: x[0] if pd.notna(x[0]) else 9):
        out.append(line)

    # ---------------- 擇時 ----------------
    out.append("\n\n")
    out.append("| 規則 | 類別 | 擇時 alpha（年） | t 值 | DSR | 樣本外 alpha | 年化報酬 / 買進持有 | 最大回撤 / 買進持有 | 平均持股 | 判定 | 主要原因 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    vt = v[v.kind == "timing"]
    tf = t[(t.asset == "NDX") & (t.window == "full")].set_index("variant")
    rows = []
    for _, r in vt.iterrows():
        x = tf.loc[r["variant"]]
        rows.append((r["alpha_full"], f"| {spec[r['family']].title}{'' if r['variant'] == r['family'] else '（' + r['variant'].split('[')[1].rstrip(']') + '）'} | "
                     f"{spec[r['family']].category} | {pct(r['alpha_full'])} | {r['alpha_t']:.1f} | {r['DSR']:.2f} | "
                     f"{pct(r['alpha_oos'])} | {pct(x['s_CAGR'])} / {pct(x['bh_CAGR'])} | {pct(x['s_MaxDD'], 0)} / {pct(x['bh_MaxDD'], 0)} | "
                     f"{x['s_exposure'] * 100:.0f}% | {r['verdict']} | {r['reason'] if isinstance(r['reason'], str) else '—'} |"))
    for _, line in sorted(rows, key=lambda x: -x[0]):
        out.append(line)
    (R / "tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
