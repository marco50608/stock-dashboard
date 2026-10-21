"""回測引擎的單元測試（不需要 pytest）：
    PYTHONUTF8=1 .venv/Scripts/python research/tests/run_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import backtest as BT  # noqa: E402
from lib import stats as S  # noqa: E402

idx = pd.bdate_range("2000-01-03", periods=3000)
rf0 = pd.Series(0.0, index=idx)
FAILS = []


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def test_exec_lag():
    ret = pd.Series(0.0, index=idx)
    ret.iloc[100] = 0.10
    sig = pd.Series(0.0, index=idx)
    sig.iloc[98] = 1.0                   # D=98 收盤後決定 → 99 收盤買 → 吃到 100 的報酬
    r = BT.run_timing(ret, sig, rf0, cost_bps=0)
    check("訊號在 D，吃到 D+2 的報酬", abs(r["df"]["strat"].loc[idx[100]] - 0.10) < 1e-12)
    sig2 = pd.Series(0.0, index=idx)
    sig2.iloc[99] = 1.0                  # 太晚：99 收盤後才知道，100 收盤才買
    r2 = BT.run_timing(ret, sig2, rf0, cost_bps=0)
    check("訊號在 D+1 吃不到 D+2 的報酬", abs(r2["df"]["strat"].loc[idx[100]]) < 1e-12)


def test_costs():
    ret = pd.Series(0.0, index=idx)
    sig = pd.Series(0.0, index=idx)
    sig.iloc[500:] = 1.0
    r = BT.run_timing(ret, sig, rf0, cost_bps=5)
    check("一次進場只收一次 5 bps", abs(r["df"]["strat"].sum() + 5e-4) < 1e-12)


def test_forward_returns():
    p = pd.Series(np.arange(1, 3001, dtype=float), index=idx)
    fr = BT.forward_returns(p, 10)
    # D=0：D+1 收盤（2）買，D+11（12）賣
    check("未來報酬從 D+1 收盤起算", abs(fr.iloc[0] - (12 / 2 - 1)) < 1e-12)
    mn = BT.forward_min(p, 10)
    check("持有期最低點（單調上升時 > 0）", abs(mn.iloc[0] - (3 / 2 - 1)) < 1e-12)


def test_null_calibration():
    rng = np.random.default_rng(1)
    ps = []
    for k in range(60):
        r = pd.Series(rng.normal(0.0004, 0.012, len(idx)), index=idx)
        p = (1 + r).cumprod()
        trig = pd.Series(rng.random(len(idx)) < 0.01, index=idx)
        es = BT.event_study(trig, p, horizons=(21,), cooldown=21, n_null=300, seed=k)
        if 21 in es:
            ps.append((es[21]["p"], es[21]["p_shift"]))
    ps = np.array(ps)
    for j, nm in ((0, "危機群 t 檢定"), (1, "循環平移")):
        x = ps[:, j][~np.isnan(ps[:, j])]
        check(f"隨機訊號 p 值大致均勻（{nm}：平均 {x.mean():.2f}，<0.05 的比例 {np.mean(x < 0.05):.2f}）",
              0.35 < x.mean() < 0.65 and np.mean(x < 0.05) < 0.15)


def test_null_heteroskedastic():
    """觸發剛好落在高波動期、但真實超額報酬是 0：循環平移會過度拒絕，危機群 t 檢定不會。"""
    rng = np.random.default_rng(5)
    rej_shift, rej_cl, n = 0, 0, 0
    for k in range(40):
        vol = np.where((np.arange(len(idx)) // 120) % 8 == 0, 0.035, 0.009)
        r = pd.Series(rng.normal(0.0004, 1, len(idx)) * vol, index=idx)
        p = (1 + r).cumprod()
        trig = pd.Series(((np.arange(len(idx)) % 960) == 5).astype(bool), index=idx)
        es = BT.event_study(trig, p, horizons=(63,), cooldown=21, n_null=300, seed=k)
        if 63 in es and not np.isnan(es[63]["p"]):
            n += 1
            rej_shift += es[63]["p_shift"] < 0.05
            rej_cl += es[63]["p"] < 0.05
    check(f"高波動期觸發：危機群 t 檢定的誤判率 {rej_cl / n:.2f}（循環平移 {rej_shift / n:.2f}）",
          rej_cl / n <= 0.15)


def test_reserve_sim():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0004, 0.012, len(idx)), index=idx)
    trig = pd.Series(False, index=idx)
    a = BT.reserve_sim(r, rf0, trig, reserve_frac=0.0)
    check("沒有預備金時兩種做法一樣", abs(a["ratio"] - 1) < 1e-12)
    b = BT.reserve_sim(r, rf0, pd.Series(True, index=idx), reserve_frac=0.5, cost_bps=0)
    check("每天都觸發 ≈ 隔天就買（差距很小）", abs(b["ratio"] - 1) < 0.01)


def test_rc_spa():
    rng = np.random.default_rng(3)
    T = 2000
    null = pd.DataFrame(rng.normal(0, 0.01, (T, 20)))
    r0 = S.reality_check(null, B=300, seed=1)
    good = null.copy()
    good[0] += 0.002
    r1 = S.reality_check(good, B=300, seed=1)
    check(f"RC/SPA：全部沒效時 p 不小（RC {r0['p_RC']:.2f} / SPA {r0['p_SPA']:.2f}）", r0["p_SPA"] > 0.05)
    check(f"RC/SPA：有一個真的有效時 p 很小（RC {r1['p_RC']:.3f} / SPA {r1['p_SPA']:.3f}）", r1["p_SPA"] < 0.05)


def test_dsr():
    rng = np.random.default_rng(4)
    trials = [S.sharpe_per_period(rng.normal(0, 0.01, 2500)) for _ in range(100)]
    best = rng.normal(0, 0.01, 2500)
    d = S.deflated_sharpe(best, np.array(trials))
    check(f"DSR：雜訊策略不會被判有效（DSR {d['DSR']:.2f}）", d["DSR"] < 0.95)


def test_bh():
    p = pd.Series([0.001, 0.01, 0.02, 0.5, 0.9])
    r = S.benjamini_hochberg(p, 0.05)
    check("BH：前三個被接受", r["reject"].sum() == 3)


if __name__ == "__main__":
    for f in [test_exec_lag, test_costs, test_forward_returns, test_null_calibration, test_null_heteroskedastic,
              test_reserve_sim,
              test_rc_spa, test_dsr, test_bh]:
        print(f.__name__)
        f()
    print("\n全部通過" if not FAILS else f"\n失敗：{FAILS}")
    sys.exit(1 if FAILS else 0)
