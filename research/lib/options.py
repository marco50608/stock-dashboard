"""賣 put（cash-secured put）的定價與模擬。

沒有免費的 QQQ／NDX 選擇權歷史報價，所以：
  - 權利金：Black-Scholes，波動率用 VXN（NDX 30 天隱含波動；S&P 500 用 VIX），
    價外的 put 再加偏斜：iv = 隱含波動 + skew_slope × 價外幅度（例如 0.4 × 5% = +2 個波動點）。
  - 成交成本：賣出時只拿到理論權利金的 (1 − haircut)，代表買賣價差與滑價（保守設定，另做敏感度）。
  - 每月第一個交易日收盤賣出約 1 個月的 put，下個月第一個交易日收盤結算，再滾下一筆。
  - 用 CBOE 官方的 S&P 500 賣 put 指數（PUT）驗證這套方法（tests 與 run_csp.py）。

時間規則和其他研究一樣：用的都是當天收盤就知道的資料（價格、VIX/VXN 收盤、前一天的國庫券利率）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def bs_put(S, K, T, r, q, sigma):
    """歐式 put 價格（Black-Scholes-Merton，連續股息率 q）。T 以年為單位。"""
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_call(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def roll_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每個月第一個交易日。"""
    m = index.to_period("M")
    first = np.r_[True, m[1:] != m[:-1]]
    return index[first]


def putwrite_index(price: pd.Series, iv: pd.Series, rf_annual: pd.Series, *, otm: float = 0.0,
                   skew_slope: float = 0.4, haircut: float = 0.10, q: float = 0.0,
                   start=None, end=None) -> pd.DataFrame:
    """完全以現金擔保的賣 put 指數（像 CBOE PUT）：每月用全部淨值當擔保，賣出 淨值/履約價 口 put。
    price：標的價格（不含股息）；iv：隱含波動（小數）；rf_annual：國庫券年利率（小數，已是當時可得）。
    回傳每日淨值與部位資訊。"""
    df = pd.DataFrame({"S": price, "iv": iv, "r": rf_annual}).dropna()
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    rolls = set(roll_dates(df.index))
    idx = df.index
    nav = np.empty(len(df))
    cash = 1.0
    n = 0.0
    K = np.nan
    exp_i = None
    prev_r = df["r"].iloc[0]
    roll_list = [i for i, d in enumerate(idx) if d in rolls]
    next_roll = {roll_list[j]: roll_list[j + 1] for j in range(len(roll_list) - 1)}
    for i, d in enumerate(idx):
        S, sig, r = df["S"].iat[i], df["iv"].iat[i], df["r"].iat[i]
        if i > 0:
            cash *= 1 + prev_r / 252
        prev_r = r
        if exp_i is not None and i == exp_i:              # 到期結算
            cash -= n * max(K - S, 0.0)
            n, exp_i = 0.0, None
        if i in next_roll and exp_i is None:               # 賣下一個月的 put
            K = S * (1 - otm)
            T = (next_roll[i] - i) / 252
            n = cash / K
            prem = float(bs_put(S, K, T, r, q, sig + skew_slope * otm)) * (1 - haircut)
            cash += n * prem
            exp_i = next_roll[i]
        if exp_i is not None:
            T_left = (exp_i - i) / 252
            liab = n * float(bs_put(S, K, T_left, r, q, sig + skew_slope * otm))
        else:
            liab = 0.0
        nav[i] = cash - liab
    return pd.DataFrame({"nav": nav}, index=idx)


def accumulate_csp(price: pd.Series, tr: pd.Series, iv: pd.Series, rf_annual: pd.Series, *,
                   otm: float = 0.0, skew_slope: float = 0.4, haircut: float = 0.10, q: float = 0.0,
                   max_wait: int | None = None, start=None, end=None) -> dict:
    """累積者的做法：每月第一個交易日存 1 元。
    A：直接買（每月 1 元買進標的，總報酬 tr）。
    B：新錢放進擔保現金，用全部擔保現金賣 1 個月 put；被指派就照履約價買進（以當時市價計入持股），
       沒被指派就繼續滾；max_wait 個月一直沒被指派，就把擔保現金全部用市價買進。
    回傳兩者期末財富、被指派次數、平均現金比例。"""
    df = pd.DataFrame({"S": price, "tr": tr, "iv": iv, "r": rf_annual}).dropna()
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    idx = df.index
    roll_list = [i for i, d in enumerate(idx) if d in set(roll_dates(idx))]
    next_roll = {roll_list[j]: roll_list[j + 1] for j in range(len(roll_list) - 1)}
    rollset = set(roll_list)
    A = 0.0
    E, C, n, K, exp_i = 0.0, 0.0, 0.0, np.nan, None
    months_unassigned = 0
    assigned = 0
    rolls_done = 0
    cash_share = []
    contributed = 0.0
    prev_r = df["r"].iloc[0]
    for i in range(len(idx)):
        S, sig, r = df["S"].iat[i], df["iv"].iat[i], df["r"].iat[i]
        if i > 0:
            g = 1 + df["tr"].iat[i]
            A *= g
            E *= g
            C *= 1 + prev_r / 252
        prev_r = r
        if exp_i is not None and i == exp_i:
            if S < K:                                       # 被指派：照 K 買，持股以市價 S 計
                E += n * S
                C -= n * K
                assigned += 1
                months_unassigned = 0
            else:
                months_unassigned += 1
            n, exp_i = 0.0, None
        if i in rollset:
            A += 1.0
            C += 1.0
            contributed += 1.0
            if max_wait is not None and months_unassigned >= max_wait and C > 0:
                E += C
                C = 0.0
                months_unassigned = 0
            if i in next_roll and C > 1e-12:
                K = S * (1 - otm)
                T = (next_roll[i] - i) / 252
                n = C / K
                C += n * float(bs_put(S, K, T, r, q, sig + skew_slope * otm)) * (1 - haircut)
                exp_i = next_roll[i]
                rolls_done += 1
            tot = E + C
            cash_share.append(C / tot if tot > 0 else np.nan)
    # 期末：未到期的 put 以理論價結算（買回）
    if exp_i is not None:
        S, sig, r = df["S"].iat[-1], df["iv"].iat[-1], df["r"].iat[-1]
        C -= n * float(bs_put(S, K, (exp_i - (len(idx) - 1)) / 252, r, q, sig + skew_slope * otm))
    B = E + C
    return {"A_buy": A, "B_csp": B, "ratio": B / A if A else np.nan, "contributed": contributed,
            "assigned": assigned, "rolls": rolls_done, "assign_rate": assigned / max(rolls_done, 1),
            "avg_cash_share": float(np.nanmean(cash_share)) if cash_share else np.nan,
            "start": idx[0].date().isoformat(), "end": idx[-1].date().isoformat()}


def panic_csp_vs_buy(price: pd.Series, tr: pd.Series, iv: pd.Series, rf_annual: pd.Series,
                     trigger_days: pd.DatetimeIndex, *, horizon: int = 126, otm: float = 0.0,
                     skew_slope: float = 0.4, haircut: float = 0.10, q: float = 0.0) -> pd.DataFrame:
    """恐慌時（例如 VIX ≥ 40）用 1 元現金：直接買（D+1 收盤） vs 賣 1 個月 put（D+1 收盤賣，每 21 個交易日滾一次，
    被指派就持有到期末），比較 horizon 天後的價值。"""
    df = pd.DataFrame({"S": price, "tr": tr, "iv": iv, "r": rf_annual}).dropna()
    idx = df.index
    out = []
    for d in trigger_days:
        k0 = idx.searchsorted(d) + 1                        # D+1 收盤
        k_end = k0 + horizon
        if k_end >= len(idx):
            continue
        buy = float((1 + df["tr"].iloc[k0 + 1:k_end + 1]).prod())
        C, E = 1.0, 0.0
        i = k0
        assigned_at = None
        while i < k_end and C > 1e-12:
            j = min(i + 21, k_end)
            S, sig, r = df["S"].iat[i], df["iv"].iat[i], df["r"].iat[i]
            K = S * (1 - otm)
            n = C / K
            C += n * float(bs_put(S, K, (j - i) / 252, r, q, sig + skew_slope * otm)) * (1 - haircut)
            C *= float(np.prod(1 + df["r"].iloc[i:j].values / 252))
            Sj = df["S"].iat[j]
            if j == k_end and Sj >= K:
                pass
            elif Sj < K:
                E += n * Sj
                C -= n * K
                assigned_at = idx[j]
                E *= float((1 + df["tr"].iloc[j + 1:k_end + 1]).prod())
                C *= float(np.prod(1 + df["r"].iloc[j:k_end].values / 252))   # 剩下的權利金現金繼續生息
                break
            i = j
        out.append({"date": d.date().isoformat(), "buy": buy, "csp": E + C,
                    "assigned": assigned_at.date().isoformat() if assigned_at is not None else None,
                    "iv_at_sale": float(df["iv"].iat[k0])})
    return pd.DataFrame(out)


def ann_stats(nav: pd.Series) -> dict:
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    dd = (nav / nav.cummax() - 1).min()
    return {"CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1, "vol": r.std() * math.sqrt(252),
            "MaxDD": float(dd), "start": nav.index[0].date().isoformat(), "end": nav.index[-1].date().isoformat()}
