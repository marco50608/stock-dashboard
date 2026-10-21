"""每個登記的訊號都做：能不能算、涵蓋期間、截斷測試（有沒有偷看未來）。
    PYTHONUTF8=1 .venv/Scripts/python research/tests/run_signal_checks.py
截斷測試：用截至 X 的資料重算一次，X 以前的訊號必須和用完整資料算的一模一樣。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.inputs import Bundle  # noqa: E402
from lib.signals import all_variants  # noqa: E402

CUTOFFS = ["2012-06-29", "2020-03-02"]


def same(a: pd.Series, b: pd.Series) -> tuple[bool, str]:
    a = a.astype(float)
    b = b.astype(float)
    idx = b.index
    a = a.reindex(idx)
    both_nan = a.isna() & b.isna()
    diff = ~both_nan & ~(np.isclose(a.fillna(-9e9), b.fillna(-9e9), atol=1e-9))
    if diff.any():
        d = diff[diff].index[0]
        return False, f"first diff {d.date()} full={a.loc[d]} cut={b.loc[d]}"
    return True, ""


def main():
    full = Bundle()
    cuts = {c: Bundle(cutoff=c) for c in CUTOFFS}
    rows = []
    bad = []
    for spec, params, name in all_variants():
        t0 = time.time()
        try:
            s = spec.func(full, "NDX", **params)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            bad.append(name)
            continue
        s = s.dropna()
        info = {"name": name, "kind": spec.kind, "start": s.index[0].date() if len(s) else None,
                "end": s.index[-1].date() if len(s) else None}
        if spec.kind == "entry":
            info["n_true"] = int(s.astype(bool).sum())
        else:
            info["exposure"] = round(float(s.mean()), 3)
        ok_all = True
        for c, Bc in cuts.items():
            try:
                sc = spec.func(Bc, "NDX", **params)
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"cut error {e}"
            else:
                ref = spec.func(full, "NDX", **params)
                ref = ref.loc[: pd.Timestamp(c)]
                sc = sc.loc[: pd.Timestamp(c)]
                if len(sc.dropna()) == 0 and len(ref.dropna()) == 0:
                    ok, msg = True, ""
                else:
                    ok, msg = same(ref, sc)
            if not ok:
                ok_all = False
                print(f"LOOKAHEAD? {name} @ {c}: {msg}")
        info["trunc_ok"] = ok_all
        info["secs"] = round(time.time() - t0, 1)
        rows.append(info)
        if not ok_all:
            bad.append(name)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    out = Path(__file__).resolve().parent.parent / "results" / "signal_checks.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n全部通過" if not bad else f"\n有問題：{bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
