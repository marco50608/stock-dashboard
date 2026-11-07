"""第三輪追加測試的敏感度檢查（事先登記，不算進判定）：EPU 多延後 21 個交易日、EBP 多延後 1 個月，看結論會不會變。

    PYTHONUTF8=1 .venv/Scripts/python research/run_batch3_sensitivity.py

原因：EPU 的新聞樣本會修正、EBP 每次更新都重新估計整段歷史，兩者都沒有歷史版本，回測有一點前視。
輸出 results/batch3_sensitivity.csv：每個受影響的變體，原本的延遲 vs 多延後之後的結果。
"""
from __future__ import annotations

import sys
import warnings
import zlib
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_study as RS  # noqa: E402
from lib import backtest as BT  # noqa: E402
from lib.inputs import Bundle  # noqa: E402
from lib.signals import all_variants  # noqa: E402

AFFECTED = {"epu_spike", "ebp_peak", "ebp_rise", "tightening_2of3"}


class LaggedBundle(Bundle):
    """EPU 多延後 21 個交易日、EBP 多延後 1 個月。"""

    def epu30(self, extra_bdays=0):
        return super().epu30(extra_bdays + 21)

    def ebp(self, extra_months=0):
        return super().ebp(extra_months + 1)


def main():
    base, lag = Bundle(), LaggedBundle()
    rf = base.rf()
    rows = []
    for kind in ("entry", "timing"):
        for spec, params, name in all_variants(kind):
            if spec.family not in AFFECTED:
                continue
            for label, B in (("原本", base), ("多延後", lag)):
                for asset in RS.ASSETS:
                    sig = spec.func(B, asset, **params)
                    if kind == "entry":
                        es = BT.event_study(sig, B.px(asset), RS.HORIZONS, RS.COOLDOWN, None, None, RS.N_NULL,
                                            seed=zlib.crc32(f"{name}|{asset}|full".encode()))
                        h = es.get(RS.PRIMARY_H, {})
                        rows.append({"variant": name, "kind": kind, "asset": asset, "lag": label,
                                     "n": h.get("n"), "excess_63": h.get("excess"), "p": h.get("p")})
                    else:
                        r = BT.run_timing(B.tr(asset), sig, rf, RS.COST_BPS)
                        rows.append({"variant": name, "kind": kind, "asset": asset, "lag": label,
                                     "alpha_ann": r.get("alpha_ann"), "alpha_t": r.get("alpha_t"),
                                     "exposure": float(r["df"]["pos"].mean()) if not r["df"].empty else None})
            print(f"  {name}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "batch3_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
