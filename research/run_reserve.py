"""累積者的實際問題：每月存的錢，一半直接買、一半先放現金等訊號再買，會不會比全部直接買好？

    PYTHONUTF8=1 .venv/Scripts/python research/run_reserve.py [變體名稱 ...]

不給名稱時，跑所有「進場」變體（每個約 30 秒）。
規則（reserve_sim）：每月第一個交易日存 1 元；A = 全部馬上買；B = 50% 馬上買、50% 進預備金（賺國庫券利率），
訊號觸發 → 隔天收盤把預備金全部買進；放滿 12 個月還沒等到訊號就直接買。
p 值：把觸發日整串隨機平移 300 次（保留次數和群聚），看隨機時機有多常比實際好。
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
from lib.inputs import Bundle  # noqa: E402
from lib.signals import all_variants  # noqa: E402

CONFIGS = [
    {"reserve_frac": 0.5, "max_wait_months": 12, "tranche_frac": 1.0},
    {"reserve_frac": 0.5, "max_wait_months": 24, "tranche_frac": 0.5},
]


def main(names):
    B = Bundle()
    rf = B.rf()
    rows = []
    series = pd.read_pickle(ROOT / "results" / "entry_series.pkl") if (ROOT / "results" / "entry_series.pkl").exists() else {}
    for spec, params, name in all_variants("entry"):
        if names and name not in names:
            continue
        for asset in ("NDX", "SPX"):
            trig = series.get((name, asset))
            if trig is None:
                trig = spec.func(B, asset, **params)
            valid = trig.dropna()
            if valid.empty:
                continue
            start = max(valid.index[0], pd.Timestamp("1990-01-01"))
            end = valid.index[-1]
            ret = B.tr(asset)
            for ci, cfg in enumerate(CONFIGS):
                for wname, (a, b) in {"full": (start, end), "OOS": (max(start, pd.Timestamp("2013-01-01")), end)}.items():
                    r = BT.reserve_sim_test(ret, rf, trig.fillna(0).astype(bool), n_null=300, seed=ci,
                                            start=a, end=b, **cfg)
                    rows.append({"variant": name, "asset": asset, "config": ci, "window": wname,
                                 "start": a.date(), "end": b.date(), **cfg,
                                 "ratio": r["ratio"], "null_median": r["null_median"], "p": r["p"],
                                 "avg_wait_days": r["avg_wait_days"],
                                 "share_by_signal": r["share_deployed_by_signal"]})
            print(f"  {name} {asset} ratio={rows[-4]['ratio']:.4f} p={rows[-4]['p']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / ("reserve_results.csv" if not names else "reserve_results_selected.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
