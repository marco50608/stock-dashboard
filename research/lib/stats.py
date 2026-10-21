"""多重檢定與穩健統計。

- deflated_sharpe：Bailey & López de Prado (2014)。考慮「試了 N 個策略、挑最好的」造成的高估。
- reality_check / spa：White (2000) Reality Check 與 Hansen (2005) SPA，用 stationary bootstrap
  （Politis & Romano 1994）保留時間序列的自我相關。虛無假設：所有試過的策略都沒有比基準好。
- benjamini_hochberg：控制錯誤發現率（FDR）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

EULER = 0.5772156649


def stationary_bootstrap_idx(T: int, avg_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / avg_block
    starts = rng.random(T) < p
    starts[0] = True
    rnd = rng.integers(T, size=T)
    ar = np.arange(T)
    last = np.maximum.accumulate(np.where(starts, ar, 0))   # 每個 t 所在區塊的起點
    return (rnd[last] + (ar - last)) % T


def sharpe_per_period(x: np.ndarray) -> float:
    s = np.std(x, ddof=1)
    return float(np.mean(x) / s) if s > 0 else 0.0


def expected_max_sr(sr_var: float, n_trials: int) -> float:
    """N 個「真實 SR = 0」的策略中，最好的那個 SR 的期望值（每期單位）。"""
    if n_trials <= 1:
        return 0.0
    z1 = st.norm.ppf(1 - 1.0 / n_trials)
    z2 = st.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_var) * ((1 - EULER) * z1 + EULER * z2))


def probabilistic_sharpe(x: np.ndarray, sr_benchmark: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    sr = sharpe_per_period(x)
    g3 = st.skew(x)
    g4 = st.kurtosis(x, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom))


def deflated_sharpe(x: np.ndarray, trial_srs: np.ndarray) -> dict:
    """x：被選中的策略的每期（超額）報酬；trial_srs：所有試過的策略的每期 SR。"""
    trial_srs = np.asarray(trial_srs, dtype=float)
    trial_srs = trial_srs[~np.isnan(trial_srs)]
    n = len(trial_srs)
    sr0 = expected_max_sr(np.var(trial_srs, ddof=1) if n > 1 else 0.0, n)
    return {"DSR": probabilistic_sharpe(x, sr0), "SR0_ann": sr0 * np.sqrt(252),
            "SR_ann": sharpe_per_period(np.asarray(x)[~np.isnan(x)]) * np.sqrt(252), "n_trials": n}


def reality_check(d: pd.DataFrame, B: int = 1000, avg_block: float = 20.0, seed: int = 0) -> dict:
    """d：T × K，每欄是「策略 − 基準」的每期差異（正 = 策略比較好）。
    回傳 White RC p 值與 Hansen SPA（consistent）p 值，以及每個策略的 t 值。"""
    d = d.dropna(how="any")
    X = d.values
    T, K = X.shape
    rng = np.random.default_rng(seed)
    mean = X.mean(axis=0)
    # 每個策略的長期標準差（用 bootstrap 估）
    boot_means = np.empty((B, K))
    for b in range(B):
        idx = stationary_bootstrap_idx(T, avg_block, rng)
        boot_means[b] = X[idx].mean(axis=0)
    omega = boot_means.std(axis=0, ddof=1)
    omega[omega == 0] = np.nan
    # White RC：V = max_k sqrt(T) mean_k
    V = np.sqrt(T) * np.nanmax(mean)
    V_star = np.sqrt(T) * np.nanmax(boot_means - mean, axis=1)
    p_rc = float((1 + (V_star >= V).sum()) / (B + 1))
    # Hansen SPA（studentized；明顯很差的策略保留它的負平均，其餘當作剛好在虛無邊界 0）
    t_obs = mean / omega
    T_spa = max(np.nanmax(t_obs), 0.0)
    thresh = -np.sqrt(2 * np.log(np.log(T))) * omega
    mu_c = np.where(mean <= thresh, mean, 0.0)
    Z = (boot_means - mean + mu_c) / omega
    T_star = np.maximum(np.nanmax(Z, axis=1), 0.0)
    p_spa = float((1 + (T_star >= T_spa).sum()) / (B + 1))
    return {"p_RC": p_rc, "p_SPA": p_spa, "T": T, "K": K,
            "t_by_strategy": pd.Series(t_obs, index=d.columns).sort_values(ascending=False)}


def benjamini_hochberg(p: pd.Series, q: float = 0.10) -> pd.DataFrame:
    p = pd.Series(p).dropna().sort_values()
    m = len(p)
    rank = np.arange(1, m + 1)
    adj = np.minimum.accumulate((p.values * m / rank)[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    return pd.DataFrame({"p": p.values, "q_value": adj, "reject": adj <= q}, index=p.index)


def block_bootstrap_ci(x: pd.Series, func, B: int = 1000, avg_block: float = 20.0, alpha: float = 0.10,
                       seed: int = 0) -> tuple[float, float]:
    x = pd.Series(x).dropna().values
    rng = np.random.default_rng(seed)
    vals = [func(x[stationary_bootstrap_idx(len(x), avg_block, rng)]) for _ in range(B)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
