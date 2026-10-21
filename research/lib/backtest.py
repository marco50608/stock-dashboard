"""回測引擎：擇時（進出場）回測、進場事件研究、現金分批部署模擬。

時間規則（全部函式共用）：
    訊號值在 D 收盤後才算得出來 → D+1 收盤成交 → 部位從 D+2 的日報酬開始生效。
    所以部位 = 訊號.shift(EXEC_LAG)，EXEC_LAG = 2。
    事件研究的進場價 = D+1 收盤價（同一個規則）。
成本：每次調整部位收 |Δ部位| × cost_bps（預設 5 bps = 手續費 + 價差 + 滑價；另做 10、20 bps 敏感度）。
現金：沒有投入的部分賺 3 個月國庫券利率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

EXEC_LAG = 2
DEFAULT_COST_BPS = 5.0
ANN = 252


# ---------------------------------------------------------------------------
# 績效指標
# ---------------------------------------------------------------------------
def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min())


def perf_stats(r: pd.Series, rf: pd.Series, pos: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    rf = rf.reindex(r.index).fillna(0)
    ex = r - rf
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    ppy = len(r) / years          # 每年交易日數（1952 年以前含週六，所以不是固定 252）
    growth = float((1 + r).prod())
    out = {
        "start": r.index[0].date().isoformat(), "end": r.index[-1].date().isoformat(), "years": round(years, 1),
        "CAGR": growth ** (1 / years) - 1,
        "vol": float(r.std() * np.sqrt(ppy)),
        "Sharpe": float(ex.mean() / ex.std() * np.sqrt(ppy)) if ex.std() > 0 else np.nan,
        "Sortino": float(ex.mean() / ex[ex < 0].std() * np.sqrt(ppy)) if (ex < 0).sum() > 5 else np.nan,
        "MaxDD": max_drawdown(r),
    }
    out["Calmar"] = out["CAGR"] / abs(out["MaxDD"]) if out["MaxDD"] < 0 else np.nan
    if pos is not None:
        p = pos.reindex(r.index)
        out["exposure"] = float(p.mean())
        out["time_in_mkt"] = float((p > 0.5).mean())
        dp = p.diff().abs().fillna(0)
        out["trades_per_yr"] = float((dp > 1e-9).sum() / years)
        out["turnover_per_yr"] = float(dp.sum() / years)
    return out


# ---------------------------------------------------------------------------
# 擇時回測
# ---------------------------------------------------------------------------
def run_timing(ret: pd.Series, signal_pos: pd.Series, rf: pd.Series, cost_bps: float = DEFAULT_COST_BPS,
               lag: int = EXEC_LAG, start=None, end=None) -> dict:
    """signal_pos：D 收盤後決定的目標部位（0~1）。回傳策略日報酬、實際部位、基準。"""
    sp = signal_pos.reindex(ret.index)
    pos = sp.shift(lag)
    df = pd.DataFrame({"ret": ret, "pos": pos, "rf": rf.reindex(ret.index).fillna(0)})
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    df = df.dropna(subset=["ret", "pos"])
    if df.empty:
        return {"df": df}
    cost = df["pos"].diff().abs().fillna(df["pos"].abs()) * cost_bps / 1e4
    df["strat"] = df["pos"] * df["ret"] + (1 - df["pos"]) * df["rf"] - cost
    beta = float(df["pos"].mean())
    df["matched"] = beta * df["ret"] + (1 - beta) * df["rf"]      # 同樣平均曝險的固定比例
    df["alpha"] = df["strat"] - df["matched"]                      # 擇時本身的貢獻
    return {"df": df, "beta": beta,
            "strat": perf_stats(df["strat"], df["rf"], df["pos"]),
            "bh": perf_stats(df["ret"], df["rf"]),
            "matched": perf_stats(df["matched"], df["rf"]),
            "alpha_ann": float(df["alpha"].mean() * len(df) / max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)),
            "alpha_t": _nw_t(df["alpha"])}


def _nw_t(x: pd.Series, lags: int | None = None) -> float:
    """Newey-West t 值（處理自我相關）。"""
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < 30:
        return np.nan
    lags = lags if lags is not None else int(4 * (n / 100) ** (2 / 9))
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else np.nan


# ---------------------------------------------------------------------------
# 進場事件研究
# ---------------------------------------------------------------------------
def episode_starts(trigger: pd.Series, cooldown: int = 21) -> pd.DatetimeIndex:
    """觸發日：訊號成立，且距離上一次觸發至少 cooldown 個交易日。"""
    t = trigger.fillna(False).astype(bool)
    idx = np.flatnonzero(t.values)
    starts, last = [], -10 ** 9
    for i in idx:
        if i - last >= cooldown:
            starts.append(i)
            last = i
    return t.index[starts]


def forward_returns(tr_index: pd.Series, h: int, entry_lag: int = 1) -> pd.Series:
    """D 的值 = 從 D+entry_lag 收盤買進、持有 h 個交易日的總報酬。"""
    p = tr_index
    return p.shift(-(entry_lag + h)) / p.shift(-entry_lag) - 1


def forward_min(tr_index: pd.Series, h: int, entry_lag: int = 1) -> pd.Series:
    """D 的值 = 進場後 h 天內最低點相對進場價（< 0 代表之後還有更低的價格）。"""
    p = tr_index
    entry = p.shift(-entry_lag)
    fut_min = p[::-1].rolling(h, min_periods=1).min()[::-1].shift(-(entry_lag + 1))
    return fut_min / entry - 1


def event_study(trigger: pd.Series, tr_index: pd.Series, horizons=(21, 63, 126), cooldown: int = 21,
                start=None, end=None, n_null: int = 2000, seed: int = 0, entry_lag: int = 1) -> dict:
    """觸發後的未來報酬 vs 同期間所有日子的平均（無條件）。
    p 值用「循環平移」：把整串觸發日一起平移隨機天數（保留觸發的群聚性），看平均未來報酬多常比實際好。"""
    rng = np.random.default_rng(seed)
    trig = trigger.reindex(tr_index.index)
    if start is not None:
        trig = trig.loc[pd.Timestamp(start):]
    if end is not None:
        trig = trig.loc[: pd.Timestamp(end)]
    trig = trig[trig.notna()]
    out = {"n_events": 0}
    if trig.empty:
        return out
    ev = episode_starts(trig, cooldown)
    out["n_events"] = len(ev)
    out["events"] = ev
    for h in horizons:
        fr = forward_returns(tr_index, h, entry_lag).reindex(trig.index).dropna()
        if len(fr) < 2 * h or len(ev) == 0:
            continue
        e = fr.reindex(ev).dropna()
        if len(e) == 0:
            continue
        base = float(fr.mean())
        obs = float(e.mean())
        # 循環平移的虛無分佈
        T = len(fr)
        pos = fr.index.get_indexer(e.index)
        vals = fr.values
        null = np.empty(n_null)
        for b in range(n_null):
            k = rng.integers(h, max(T - h, h + 1))
            null[b] = vals[(pos + k) % T].mean()
        p_shift = float((1 + (null >= obs).sum()) / (n_null + 1))
        # 主要的 p 值：平均超額報酬（每次觸發權重相同）配「危機群穩健標準誤」的單尾 t 檢定。
        # 彼此相隔不到 h 個交易日的觸發算同一群（未來報酬期間重疊、又常落在同一段高波動期）。
        # 循環平移會把事件搬到平靜期，低估恐慌期報酬的波動、讓 p 值偏小（2026-10-19 稽核發現），
        # 所以只留作參考（p_shift）。
        order = np.argsort(pos)
        ps = pos[order]
        cl = np.r_[0, np.cumsum(np.diff(ps) > h)]
        x = e.values[order] - base
        u = x - x.mean()
        G = int(cl.max()) + 1
        n_ev = len(x)
        if G >= 3:
            Sc = pd.Series(u).groupby(cl).sum().values
            var = (G / (G - 1)) * float((Sc ** 2).sum()) / n_ev ** 2
            tstat = x.mean() / np.sqrt(var) if var > 0 else np.nan
            p_cl = float(1 - st.t.cdf(tstat, G - 1)) if var > 0 else float("nan")
        else:
            p_cl = float("nan")
        k_cl = G
        mn = forward_min(tr_index, h, entry_lag).reindex(e.index)
        mn_all = forward_min(tr_index, h, entry_lag).reindex(fr.index)
        out[h] = {"n": len(e), "mean": obs, "base": base, "excess": obs - base,
                  "median": float(e.median()), "hit": float((e > 0).mean()), "hit_base": float((fr > 0).mean()),
                  "worst": float(e.min()), "p": p_cl, "p_shift": p_shift, "n_clusters": k_cl,
                  "fwd_min_mean": float(mn.mean()), "fwd_min_base": float(mn_all.mean())}
    return out


# ---------------------------------------------------------------------------
# 現金分批部署模擬（累積者的實際問題：每月存的錢，要馬上買還是等訊號？）
# ---------------------------------------------------------------------------
def reserve_sim(ret: pd.Series, rf: pd.Series, trigger: pd.Series, reserve_frac: float = 0.5,
                max_wait_months: int = 12, tranche_frac: float = 1.0, cost_bps: float = DEFAULT_COST_BPS,
                start=None, end=None) -> dict:
    """每個月第一個交易日存入 1 元。
    A（立刻買）：全部馬上買。
    B（等訊號）：(1-reserve_frac) 馬上買；reserve_frac 放進預備金賺國庫券利率，
       訊號觸發（D 收盤後）→ D+1 收盤把預備金的 tranche_frac 買進；
       放超過 max_wait_months 的錢，不管訊號直接買（避免永遠抱現金）。
    回傳兩者期末財富、比值、預備金平均等待天數。"""
    df = pd.DataFrame({"ret": ret, "rf": rf.reindex(ret.index).fillna(0),
                       "trig": trigger.reindex(ret.index).fillna(False).astype(bool)})
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    df = df.dropna(subset=["ret"])
    c = cost_bps / 1e4
    month = df.index.to_period("M")
    first = np.r_[True, month[1:] != month[:-1]]
    trig_exec = df["trig"].shift(1, fill_value=False).values   # D 觸發 → D+1 收盤成交
    A_sh = 0.0
    B_sh = 0.0
    lots: list[list[float]] = []      # 預備金每一筆 [金額, 存入的月序號]
    m_idx = -1
    waits, wait_amt = 0.0, 0.0
    deployed_by_signal = 0.0
    p = 1.0
    for i, (r, f, tx) in enumerate(zip(df["ret"].values, df["rf"].values, trig_exec)):
        p *= 1 + r
        for lot in lots:
            lot[0] *= 1 + f
        if first[i]:
            m_idx += 1
            A_sh += (1 - c) / p
            B_sh += (1 - reserve_frac) * (1 - c) / p
            if reserve_frac > 0:
                lots.append([reserve_frac, m_idx, i])
        buy = 0.0
        keep = []
        for lot in lots:
            if m_idx - lot[1] >= max_wait_months:
                buy += lot[0]
                waits += (i - lot[2]) * lot[0]
                wait_amt += lot[0]
            else:
                keep.append(lot)
        lots = keep
        if tx and lots:
            for lot in lots:
                amt = lot[0] * tranche_frac
                buy += amt
                deployed_by_signal += amt
                waits += (i - lot[2]) * amt
                wait_amt += amt
                lot[0] -= amt
            lots = [l for l in lots if l[0] > 1e-12]
        if buy > 0:
            B_sh += buy * (1 - c) / p
    cash_left = sum(l[0] for l in lots)
    A = A_sh * p
    B = B_sh * p + cash_left
    return {"A_immediate": A, "B_signal": B, "ratio": B / A if A else np.nan, "months": m_idx + 1,
            "avg_wait_days": waits / wait_amt if wait_amt else np.nan,
            "share_deployed_by_signal": deployed_by_signal / max(wait_amt, 1e-12)}


def reserve_sim_test(ret: pd.Series, rf: pd.Series, trigger: pd.Series, n_null: int = 300, seed: int = 0,
                     **kw) -> dict:
    """reserve_sim 加上虛無分佈：把觸發日整串循環平移（保留觸發次數與群聚），
    看「隨機時間點部署」得到的比值有多常比實際好。p 小 = 訊號的時機真的有價值，不只是「晚點買」的效果。"""
    rng = np.random.default_rng(seed)
    base = reserve_sim(ret, rf, trigger, **kw)
    t = trigger.reindex(ret.index)
    lo, hi = kw.get("start"), kw.get("end")
    if lo is not None:
        t = t.loc[pd.Timestamp(lo):]
    if hi is not None:
        t = t.loc[: pd.Timestamp(hi)]
    vals = t.fillna(False).astype(bool).values
    T = len(vals)
    null = []
    for _ in range(n_null):
        k = rng.integers(21, max(T - 21, 22))
        sh = pd.Series(np.roll(vals, k), index=t.index)
        null.append(reserve_sim(ret, rf, sh, **kw)["ratio"])
    null = np.array(null)
    base["null_median"] = float(np.median(null))
    base["p"] = float((1 + (null >= base["ratio"]).sum()) / (n_null + 1))
    return base
