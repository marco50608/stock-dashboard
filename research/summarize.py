"""把 results/ 裡的結果印成精簡表格（看結果用）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)


def pct(x):
    return f"{x * 100:+.1f}%" if pd.notna(x) else "—"


def main(section="all"):
    v = pd.read_csv(R / "verdicts.csv")
    if section in ("all", "verdicts"):
        print(v["verdict"].value_counts().to_string())
        e = v[v["kind"] == "entry"].copy()
        for c in ("excess_full", "excess_oos", "excess_spx", "wf_excess"):
            e[c] = e[c].map(pct)
        print(e[["variant", "verdict", "n_full", "excess_full", "q", "n_oos", "excess_oos", "excess_spx", "wf_excess",
                 "reason"]].to_string(index=False))
        t = v[v["kind"] == "timing"].copy()
        for c in ("alpha_full", "alpha_oos", "alpha_spx", "alpha_20bps", "wf_alpha"):
            t[c] = t[c].map(pct)
        print(t[["variant", "verdict", "alpha_full", "alpha_t", "DSR", "alpha_oos", "alpha_spx", "alpha_20bps",
                 "wf_alpha", "reason"]].to_string(index=False))
    if section in ("all", "mt"):
        mt = json.loads((R / "multiple_testing.json").read_text(encoding="utf-8"))
        for k in ("timing_RC_NDX", "timing_RC_SPX"):
            print(k, {kk: vv for kk, vv in mt[k].items() if kk != "top_t"})
            print("  top t:", {kk: round(vv, 2) for kk, vv in list(mt[k]["top_t"].items())[:8]})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
