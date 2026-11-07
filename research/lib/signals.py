"""候選訊號清單（事先登記）。

這份清單和每個訊號的參數組合，是在看任何回測結果「之前」寫下的，之後不再增刪參數。
每一個變體都算進多重檢定的試驗次數。

兩種訊號：
- entry（進場事件）：回傳 bool，True 的那天 = 這天收盤後訊號成立（D），D+1 收盤買進。
- timing（擇時部位）：回傳 0~1 的目標部位（D 收盤後決定），由 backtest.run_timing 套 EXEC_LAG。

每個函式的參數 B 是 lib.inputs.Bundle，asset 是要交易的標的（"NDX" 或 "SPX"）。
所有計算只能往回看（rolling / expanding / shift 正數）；tests/run_signal_checks.py 會做截斷測試。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from . import data as D


# ---------------------------------------------------------------------------
# 小工具（全部只往回看）
# ---------------------------------------------------------------------------
def rpct(s: pd.Series, w: int, minp: int | None = None) -> pd.Series:
    """滾動百分位（0~100）：今天的值在過去 w 天（含今天）裡排第幾。"""
    return s.rolling(w, min_periods=minp or w // 2).rank(pct=True) * 100


def epct(s: pd.Series, minp: int = 252) -> pd.Series:
    """擴張視窗百分位：今天的值在「到今天為止全部歷史」裡排第幾。"""
    return s.expanding(min_periods=minp).rank(pct=True) * 100


def edge(b: pd.Series, *bases: pd.Series) -> pd.Series:
    """條件「剛成立」的那天 = 1。bases：底層資料，任何一個還沒有資料的日子回傳 NaN
    （資料開始前的年份不算進比較基準）。"""
    v = b.fillna(False).astype(bool)
    out = (v & ~v.shift(1, fill_value=False)).astype(float)
    if bases:
        valid = pd.concat([x.reindex(out.index).notna() for x in bases], axis=1).all(axis=1)
        out = out.where(valid)
    return out


def off_for(trigger: pd.Series, n: int) -> pd.Series:
    """觸發後 n 個交易日部位為 0，其餘 1。"""
    t = trigger.fillna(False).astype(float)
    return 1.0 - t.rolling(n, min_periods=1).max()


def hysteresis(off_cond: pd.Series, on_cond: pd.Series) -> pd.Series:
    """off_cond 成立就出場，直到 on_cond 成立才回來。回傳 0/1。"""
    off = off_cond.fillna(False).values
    on = on_cond.fillna(False).values
    out = np.ones(len(off))
    state = 1.0
    for i in range(len(off)):
        if state == 1.0 and off[i]:
            state = 0.0
        elif state == 0.0 and on[i]:
            state = 1.0
        out[i] = state
    return pd.Series(out, index=off_cond.index)


def drawdown(px: pd.Series) -> pd.Series:
    return px / px.cummax() - 1


def dd_252(px: pd.Series) -> pd.Series:
    return px / px.rolling(252, min_periods=60).max() - 1


def rsi(px: pd.Series, n: int) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def frac_above_sma(levels: pd.DataFrame, n: int) -> pd.Series:
    sma = levels.rolling(n, min_periods=n).mean()
    valid = sma.notna() & levels.notna()
    return ((levels > sma) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)


def calendar_signal(on_return_day: pd.Series) -> pd.Series:
    """把「想在哪些日子持有（吃那天的報酬）」轉成訊號（提前 EXEC_LAG 天決定）。
    只用交易所行事曆和事先公布的日期（FOMC、BLS 發布行程），不用未來的市場資料。"""
    from .backtest import EXEC_LAG
    return on_return_day.astype(float).shift(-EXEC_LAG)


def month_end_flags(cal: pd.DatetimeIndex) -> pd.Series:
    m = cal.to_period("M")
    return pd.Series(np.r_[m[1:] != m[:-1], True], index=cal)


def bond_tr_from_yield() -> pd.Series:
    """10 年期公債總報酬近似（日）：殖利率收益 − 存續期間 × 殖利率變化。只用來算月內股債相對表現。
    用 ^TNX（CBOE 10 年殖利率指數，收盤當天就有），不用 FRED 的 DGS10（隔天才公布）。"""
    y = D.close("^TNX").ffill() / 100
    return (y.shift(1) / 252 - 8.0 * y.diff()).fillna(0)


# ---------------------------------------------------------------------------
# 登記表
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    family: str
    kind: str                 # entry / timing
    func: callable
    grid: dict
    title: str                # 中文名稱
    category: str             # 類別
    ref: str = ""             # 出處
    pub_year: int | None = None
    niche: str = "常見"        # 常見 / 半冷門 / 冷門
    note: str = ""
    batch: int = 1            # 1 = 第一輪；2 = 看過第一輪結果後追加（一樣事先固定參數、一樣算進多重檢定）
    variants: list = field(default_factory=list)

    def __post_init__(self):
        keys = list(self.grid)
        self.variants = [dict(zip(keys, v)) for v in product(*[self.grid[k] for k in keys])] if keys else [{}]


REGISTRY: list[Spec] = []


def register(family, kind, grid, title, category, ref="", pub_year=None, niche="常見", note="", batch=1):
    def deco(f):
        REGISTRY.append(Spec(family, kind, f, grid, title, category, ref, pub_year, niche, note, batch))
        return f
    return deco


def variant_name(spec: Spec, params: dict) -> str:
    if not params:
        return spec.family
    return spec.family + "[" + ",".join(f"{k}={v}" for k, v in params.items()) + "]"


# ===========================================================================
# 進場事件（entry）
# ===========================================================================
@register("dd_tranche", "entry", {"level": [0.10, 0.15, 0.20, 0.25, 0.30]},
          "距歷史高點跌幅達 X%（手冊的分批規則）", "價格", ref="使用者的進場手冊", niche="常見")
def dd_tranche(B, asset, level):
    px = B.px(asset)
    return edge(drawdown(px) <= -level, px)


@register("vix_level", "entry", {"rule": ["vix>=30", "vix>=40", "pct252>=95"]},
          "VIX 飆高", "波動率", ref="實務常見", niche="常見")
def vix_level(B, asset, rule):
    v = B.cboe("VIX")
    if rule == "vix>=30":
        return edge(v >= 30, v)
    if rule == "vix>=40":
        return edge(v >= 40, v)
    p = rpct(v, 252)
    return edge(p >= 95, p)


@register("vix_term_inv", "entry", {"th": [1.0, 1.1]},
          "VIX 期限結構倒掛（VIX / VIX3M ≥ th）", "波動率", ref="實務常見（儀表板已有）", niche="半冷門")
def vix_term_inv(B, asset, th):
    r = (B.cboe("VIX") / B.cboe("VIX3M")).dropna()
    return edge(r >= th, r)


@register("vix_term_normalize", "entry", {"th": [1.0, 0.95]},
          "VIX 期限結構倒掛結束（恐慌退潮）", "波動率", ref="實務者（倒掛結束才進場）", niche="半冷門")
def vix_term_normalize(B, asset, th):
    r = (B.cboe("VIX") / B.cboe("VIX3M")).dropna()
    inv_recent = (r.shift(1) > 1.0).rolling(10, min_periods=10).sum() >= 3
    return ((r < th) & (r.shift(1) >= th) & inv_recent).astype(float).where(r.shift(10).notna())


@register("vix_retreat", "entry", {"drop": [0.15, 0.25]},
          "VIX 從高點回落 X%（高點須在一年 90 百分位以上）", "波動率", ref="實務者", niche="半冷門")
def vix_retreat(B, asset, drop):
    v = B.cboe("VIX")
    mx = v.rolling(10, min_periods=10).max()
    hi = rpct(v, 252).rolling(10, min_periods=10).max() >= 90
    return edge(hi & (v <= (1 - drop) * mx), rpct(v, 252))


@register("equity_pc_high", "entry", {"ma": [5, 10, 21], "pct": [90, 95]},
          "股票 Put/Call 均值在一年高檔（散戶避險／恐慌）", "期權", ref="McMillan 等實務者；Pan & Poteshman 2006（個股層級）",
          niche="常見")
def equity_pc_high(B, asset, ma, pct):
    pc = B.putcall("equity").rolling(ma, min_periods=ma).mean()
    p = rpct(pc, 252)
    return edge(p >= pct, p)


@register("index_pc_high", "entry", {},
          "指數 Put/Call 10 日均在一年 95 百分位以上", "期權", ref="實務者", niche="半冷門")
def index_pc_high(B, asset):
    pc = B.putcall("index").rolling(10, min_periods=10).mean()
    p = rpct(pc, 252)
    return edge(p >= 95, p)


@register("aaii_low", "entry", {"rule": ["spread_pct5", "spread_pct10", "bears>=50"]},
          "AAII 散戶極度悲觀", "情緒", ref="Fisher & Statman 2000", pub_year=2000, niche="常見")
def aaii_low(B, asset, rule):
    a = B.aaii()
    if rule == "bears>=50":
        return edge(a["bear"] >= 0.50, a["bear"])
    p = epct(a["spread"].dropna(), 520)
    return edge(p <= (5 if rule == "spread_pct5" else 10), p)


@register("naaim_low", "entry", {"rule": ["<=30", "pct10"]},
          "NAAIM 經理人曝險極低", "情緒", ref="實務者", niche="半冷門")
def naaim_low(B, asset, rule):
    n = B.naaim()
    if rule == "<=30":
        return edge(n <= 30, n)
    p = epct(n, 260)
    return edge(p <= 10, p)


@register("credit_ease", "entry", {"bp": [0.10, 0.20]},
          "信用利差在高檔開始收斂（BAA−10Y 從 21 日高點回落）", "信用", ref="實務者：信用領先股市", niche="半冷門")
def credit_ease(B, asset, bp):
    s = B.fred("BAA10Y")
    hi = (rpct(s, 504) >= 90).rolling(21, min_periods=1).max().astype(bool)
    mx = s.rolling(21, min_periods=5).max()
    return edge(hi & (s <= mx - bp), rpct(s, 504))


@register("credit_spike", "entry", {},
          "信用利差 21 日急升（兩年 95 百分位）", "信用", ref="實務者", niche="半冷門")
def credit_spike(B, asset):
    s = B.fred("BAA10Y")
    p = rpct(s.diff(21), 504)
    return edge(p >= 95, p)


@register("breadth_thrust", "entry", {"low": [0.10, 0.20], "high": [0.50, 0.70]},
          "產業廣度 washout 後急速回升（49 產業站上 50 日線比例）", "廣度",
          ref="Zweig 1986 的概念，改用 Ken French 49 產業（沒有倖存者偏差）", pub_year=1986, niche="冷門")
def breadth_thrust(B, asset, low, high):
    f = frac_above_sma(B.industries(), 50)
    washed = (f <= low).rolling(15, min_periods=1).max().astype(bool)
    return edge(washed & (f >= high), f)


@register("zweig_industries", "entry", {},
          "Zweig 廣度衝刺（用 49 產業上漲比例的 10 日 EMA：<0.40 → >0.615）", "廣度",
          ref="Zweig 1986", pub_year=1986, niche="半冷門")
def zweig_industries(B, asset):
    r = B.industries().pct_change()
    adv = (r > 0).sum(axis=1) / r.notna().sum(axis=1).replace(0, np.nan)
    e = adv.ewm(span=10, adjust=False).mean()
    return edge((e > 0.615) & (e.rolling(10, min_periods=10).min() < 0.40), adv)


@register("rsi2", "entry", {"th": [5, 10], "trend": [False, True]},
          "RSI(2) 超賣（可加 200 日線上的條件）", "價格", ref="Connors & Alvarez 2008", pub_year=2008, niche="常見")
def rsi2(B, asset, th, trend):
    px = B.px(asset)
    c = rsi(px, 2) <= th
    if trend:
        c &= px > px.rolling(200, min_periods=200).mean()
    return edge(c, px)


@register("down_streak", "entry", {},
          "連跌 5 天以上", "價格", ref="實務者", niche="常見")
def down_streak(B, asset):
    dn = (B.px(asset).diff() < 0).astype(int)
    run = dn.groupby((dn == 0).cumsum()).cumsum()
    return (run == 5).astype(float).where(B.px(asset).notna())


@register("insider_buy", "entry", {"pct": [90, 95]},
          "全市場內部人買進家數比例在兩年高檔（SEC Form 4）", "內部人",
          ref="Seyhun 1988；Lakonishok & Lee 2001（加總內部人買賣預測大盤）", pub_year=2001, niche="冷門")
def insider_buy(B, asset, pct):
    g = B.insider()
    buy = g["firms_P"].rolling(21, min_periods=21).sum()
    sell = g["firms_S"].rolling(21, min_periods=21).sum()
    ratio = buy / (buy + sell)
    p = rpct(ratio, 504)
    return edge(p >= pct, p)


@register("cot_extreme", "entry",
          {"rule": ["ES_lev_pct5", "ES_lev_pct10", "NQ_lev_pct10", "ES_am_pct10", "ES_noncomm_pct10",
                    "VIX_lev_pct90"]},
          "COT 期貨部位極端（槓桿基金極空、資產管理人極低、VIX 期貨投機客偏多）", "籌碼",
          ref="實務者（Briese 等）", niche="冷門")
def cot_extreme(B, asset, rule):
    W = 756
    if rule.startswith("ES_lev"):
        p = rpct(B.cot_tff("13874A", "Lev_Money"), W)
        return edge(p <= (5 if rule.endswith("5") else 10), p)
    if rule == "NQ_lev_pct10":
        p = rpct(B.cot_tff("209742", "Lev_Money"), W)
        return edge(p <= 10, p)
    if rule == "ES_am_pct10":
        p = rpct(B.cot_tff("13874A", "Asset_Mgr"), W)
        return edge(p <= 10, p)
    if rule == "ES_noncomm_pct10":
        p = rpct(B.cot_legacy("13874A", "Noncommercial"), W)
        return edge(p <= 10, p)
    if rule == "VIX_lev_pct90":
        p = rpct(B.cot_tff("1170E1", "Lev_Money"), W)
        return edge(p >= 90, p)
    raise ValueError(rule)


@register("vrp_high", "entry", {},
          "波動率風險溢酬（VIX² − 實現變異）在一年 90 百分位以上", "波動率",
          ref="Bollerslev, Tauchen & Zhou 2009", pub_year=2009, niche="半冷門")
def vrp_high(B, asset):
    v = B.cboe("VIX")
    rv = B.tr("SPX").rolling(21, min_periods=21).std() * np.sqrt(252) * 100
    vrp = (v ** 2 - rv ** 2).dropna()
    p = rpct(vrp, 252)
    return edge(p >= 90, p)


@register("cor_high", "entry", {"pct": [90, 95]},
          "CBOE 隱含相關性（COR3M）在一年高檔", "波動率", ref="Driessen, Maenhout & Vilkov 2009 的概念", pub_year=2009,
          niche="冷門")
def cor_high(B, asset, pct):
    p = rpct(B.cboe("COR3M"), 252)
    return edge(p >= pct, p)


@register("vvix_high", "entry", {},
          "VVIX 在一年 95 百分位以上", "波動率", ref="實務者", niche="半冷門")
def vvix_high(B, asset):
    p = rpct(B.cboe("VVIX"), 252)
    return edge(p >= 95, p)


@register("claims_peak", "entry", {},
          "初領失業金升高後開始回落", "總經", ref="實務者", niche="半冷門")
def claims_peak(B, asset):
    c = B.claims().rolling(20, min_periods=15).mean()   # 4 週均（日序列上 20 天 ≈ 4 週）
    lo = c.rolling(252, min_periods=200).min()
    stressed = (c >= 1.25 * lo).rolling(63, min_periods=1).max().astype(bool)
    rolling_over = c <= 0.9 * c.rolling(65, min_periods=20).max()
    return edge(stressed & rolling_over, lo)


@register("margin_delever", "entry", {"th": [-0.10, -0.20]},
          "融資餘額年增率轉負（去槓桿）", "籌碼", ref="實務者", niche="半冷門")
def margin_delever(B, asset, th):
    m = B.margin()
    yoy = m / m.shift(252) - 1
    return edge(yoy <= th, yoy)


@register("flight_reversal", "entry", {},
          "避險潮極端（公債 20 日報酬減股市 20 日報酬，兩年 95 百分位）", "跨資產", ref="實務者", niche="半冷門")
def flight_reversal(B, asset):
    tlt = B.close("TLT").pct_change(20)
    spy = B.close("SPY").pct_change(20)
    p = rpct((tlt - spy).dropna(), 504)
    return edge(p >= 95, p)


@register("carry_unwind", "entry", {},
          "套息交易平倉（AUD/JPY 10 日跌幅兩年 3 百分位以下）", "跨資產", ref="實務者（2024-08 日圓平倉）", niche="冷門")
def carry_unwind(B, asset):
    p = rpct(B.close("AUDJPY=X").pct_change(10).dropna(), 504)
    return edge(p <= 3, p)


@register("fed_put", "entry", {},
          "Fed put：FOMC 前的會議間隔期間大跌 5% 以上，在會議前一天收盤買", "事件",
          ref="Cieslak & Vissing-Jorgensen 2021", pub_year=2021, niche="冷門")
def fed_put(B, asset):
    px = B.px(asset)
    cal = D.calendar()
    fomc = B.fomc()
    out = pd.Series(np.nan, index=px.index)
    out.loc[out.index >= fomc[0]] = 0.0
    for i in range(1, len(fomc)):
        prev, nxt = fomc[i - 1], fomc[i]
        k = cal.searchsorted(nxt)
        if k - 2 < 0 or k >= len(cal):
            continue
        sig_day = cal[k - 2]           # D = 會議前兩天收盤後決定 → 會議前一天收盤買
        if sig_day not in px.index or prev not in px.index:
            continue
        if px.loc[sig_day] / px.loc[prev] - 1 <= -0.05:
            out.loc[sig_day] = 1.0
    return out


@register("composite_fear", "entry", {"th": [70, 75, 80]},
          "儀表板現有的恐慌綜合分數（用長歷史資料重建）", "綜合", ref="stock 專案 utils/decision.py", niche="—")
def composite_fear(B, asset, th):
    f = composite_fear_score(B, asset)
    return edge(f >= th, f)


def composite_fear_score(B, asset) -> pd.Series:
    """照 utils/research.py 的 fear_score_panel：每個指標的擴張百分位（≥252 天）加權平均。
    差異：HY 利差改用 BAA−10Y（FRED 的 ICE HY 只剩 3 年資料）；VIX9D/VIX3M 改用 VIX/VIX3M（VIX9D 2011 才有）；
    廣度改用 49 產業站上 200 日線比例；CNN 恐懼貪婪沒有歷史資料，不納入。"""
    px = B.px(asset)
    comps = {
        "drawdown": (dd_252(px), "low", 1.5),
        "vix": (B.cboe("VIX"), "high", 1.0),
        "credit": (B.fred("BAA10Y"), "high", 1.0),
        "put_call": (B.putcall("equity").rolling(5, min_periods=5).mean(), "high", 1.0),
        "aaii": (B.aaii()["spread"], "low", 1.0),
        "naaim": (B.naaim(), "low", 1.0),
        "vix_term": ((B.cboe("VIX") / B.cboe("VIX3M")), "high", 1.0),
        "breadth": (frac_above_sma(B.industries(), 200), "low", 1.0),
    }
    idx = px.index
    num = pd.Series(0.0, index=idx)
    den = pd.Series(0.0, index=idx)
    for k, (s, kind, w) in comps.items():
        s = s.reindex(idx)
        p = s.dropna().expanding(min_periods=252).rank(pct=True).reindex(idx) * 100
        fp = p if kind == "high" else 100 - p
        m = fp.notna()
        num += fp.fillna(0) * w
        den += m.astype(float) * w
    return (num / den).where(den > 0)


# ===========================================================================
# 擇時部位（timing）：1 = 持有，0 = 現金
# ===========================================================================
@register("trend_sma", "timing", {"n": [100, 150, 200]},
          "價格在 N 日均線之上才持有", "趨勢", ref="Siegel 1994；Faber 2007", pub_year=2007, niche="常見")
def trend_sma(B, asset, n):
    px = B.px(asset)
    return (px > px.rolling(n, min_periods=n).mean()).astype(float).where(px.rolling(n).count() >= n)


@register("trend_10m", "timing", {},
          "月底價格在 10 個月均線之上才持有（每月檢查一次）", "趨勢", ref="Faber 2007", pub_year=2007, niche="常見")
def trend_10m(B, asset):
    px = B.px(asset)
    me = month_end_flags(D.calendar()).reindex(px.index).fillna(False)
    m = px[me]
    sig = (m > m.rolling(10, min_periods=10).mean()).astype(float).where(m.rolling(10).count() >= 10)
    return sig.reindex(px.index).ffill()


@register("trend_band", "timing", {},
          "200 日均線 ±2% 緩衝（跌破 98% 出場，站上 102% 回來）", "趨勢", ref="實務者", niche="常見")
def trend_band(B, asset):
    px = B.px(asset)
    sma = px.rolling(200, min_periods=200).mean()
    return hysteresis(px < 0.98 * sma, px > 1.02 * sma).where(sma.notna())


@register("vol_target", "timing", {"tv": [0.15, 0.20]},
          "波動率目標：部位 = 目標波動 / 近期波動（上限 100%）", "波動率",
          ref="Moreira & Muir 2017；Harvey et al. 2018", pub_year=2017, niche="半冷門")
def vol_target(B, asset, tv):
    r = B.tr(asset)
    vol = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean() * 252)
    w = (tv / vol).clip(upper=1.0)
    out = np.empty(len(w))
    cur = np.nan
    for i, x in enumerate(w.values):          # 變化超過 10 個百分點才調整，減少交易
        if np.isnan(cur) or abs(x - cur) > 0.10:
            cur = x
        out[i] = cur
    return pd.Series(out, index=w.index).where(r.expanding().count() >= 60)


@register("vix_term_riskoff", "timing", {},
          "VIX 期限結構倒掛時出場（VIX > VIX3M）", "波動率", ref="實務者", niche="半冷門")
def vix_term_riskoff(B, asset):
    r = (B.cboe("VIX") / B.cboe("VIX3M")).dropna()
    return (r <= 1.0).astype(float)


@register("vix_high_riskoff", "timing", {},
          "VIX ≥ 30 出場，< 25 回來", "波動率", ref="實務者", niche="常見")
def vix_high_riskoff(B, asset):
    v = B.cboe("VIX")
    return hysteresis(v >= 30, v < 25)


@register("credit_widen", "timing", {"th": [0.3, 0.5]},
          "信用利差 63 日擴大超過 th 個百分點時出場", "信用", ref="實務者", niche="半冷門")
def credit_widen(B, asset, th):
    s = B.fred("BAA10Y")
    return (s.diff(63) < th).astype(float).where(s.diff(63).notna())


@register("claims_recession", "timing", {},
          "初領失業金 4 週均比一年低點高 20% 時出場，回到 10% 以內回來", "總經", ref="實務者（類 Sahm）", niche="半冷門")
def claims_recession(B, asset):
    c = B.claims().rolling(20, min_periods=15).mean()
    lo = c.rolling(252, min_periods=200).min()
    return hysteresis(c >= 1.2 * lo, c <= 1.1 * lo).where(lo.notna())


@register("sahm_rt", "timing", {},
          "即時 Sahm 規則（失業率首次公布值）觸發時出場", "總經", ref="Sahm 2019", pub_year=2019, niche="半冷門")
def sahm_rt(B, asset):
    u = B.unrate_first()
    u3 = u.rolling(63, min_periods=40).mean()
    gap = u3 - u3.rolling(252, min_periods=200).min()
    return hysteresis(gap >= 0.5, gap < 0.3).where(gap.notna())


@register("margin_froth", "timing", {"rule": ["yoy>=40%", "turn_neg_after_20%"]},
          "融資餘額過熱出場（年增 ≥ 40%，或從 > 20% 轉負）", "籌碼", ref="實務者", niche="常見")
def margin_froth(B, asset, rule):
    m = B.margin()
    yoy = m / m.shift(252) - 1
    if rule == "yoy>=40%":
        return (yoy < 0.40).astype(float).where(yoy.notna())
    was_hot = (yoy > 0.20).rolling(252, min_periods=1).max().astype(bool)
    return hysteresis(was_hot & (yoy < 0), yoy > 0).where(yoy.notna())


@register("curve_uninvert", "timing", {},
          "殖利率曲線（10Y−3M）倒掛結束後出場一年", "總經", ref="Estrella & Mishkin 1998", pub_year=1998, niche="常見",
          note="事件極少（約 5 次），統計上無法驗證")
def curve_uninvert(B, asset):
    s = B.fred("T10Y3M")
    inv = (s < 0).rolling(63, min_periods=63).sum() >= 40
    trig = (s > 0) & inv.shift(1, fill_value=False)
    return off_for(edge(trig), 252)


def _fomc_day_index(B) -> pd.Series:
    """每個交易日距離「最近一次 FOMC 聲明日」幾個交易日（聲明日 = 0）。"""
    cal = D.calendar()
    f = B.fomc()
    pos = cal.searchsorted(f)
    pos = pos[pos < len(cal)]
    marker = pd.Series(np.nan, index=cal)
    marker.iloc[pos] = np.arange(len(pos))
    last = marker.ffill()
    ar = pd.Series(np.arange(len(cal)), index=cal, dtype=float)
    last_pos = pd.Series(np.where(marker.notna(), ar, np.nan), index=cal).ffill()
    since = ar - last_pos
    # 聲明日前一天算第 0 週的開始（-1）
    nxt_pos = pd.Series(np.where(marker.notna(), ar, np.nan), index=cal).bfill()
    since = since.where(~(nxt_pos - ar == 1), -1)
    return since.where(last.notna())


@register("fomc_even_weeks", "timing", {},
          "FOMC 週期：只在第 0、2、4、6 週持有", "事件",
          ref="Cieslak, Morse & Vissing-Jorgensen 2019（SSRN 2014）", pub_year=2014, niche="冷門")
def fomc_even_weeks(B, asset):
    t = _fomc_day_index(B)
    wk = np.floor((t + 1) / 5)
    on = ((wk % 2 == 0) & (wk <= 6)).astype(float).where(t.notna())
    return calendar_signal(on).reindex(B.px(asset).index)


@register("pre_fomc", "timing", {},
          "只在 FOMC 聲明日持有（前一天收盤 → 當天收盤）", "事件",
          ref="Lucca & Moench 2015（2011 工作論文）", pub_year=2011, niche="半冷門")
def pre_fomc(B, asset):
    t = _fomc_day_index(B)
    return calendar_signal((t == 0).astype(float).where(t.notna())).reindex(B.px(asset).index)


@register("turn_of_month", "timing", {},
          "只在月底最後一天到下月第 3 個交易日持有", "季節性", ref="Ariel 1987；Lakonishok & Smidt 1988",
          pub_year=1987, niche="常見")
def turn_of_month(B, asset):
    cal = D.calendar()
    m = cal.to_period("M")
    s = pd.Series(m, index=cal)
    day_of_month = s.groupby(s).cumcount() + 1
    last = month_end_flags(cal)
    on = last | (day_of_month <= 3)
    return calendar_signal(on).reindex(B.px(asset).index)


@register("halloween", "timing", {},
          "萬聖節效應：11 月到 4 月持有，5 到 10 月現金", "季節性", ref="Bouman & Jacobsen 2002", pub_year=2002,
          niche="常見")
def halloween(B, asset):
    cal = D.calendar()
    on = pd.Series(~cal.month.isin([5, 6, 7, 8, 9, 10]), index=cal)
    return calendar_signal(on).reindex(B.px(asset).index)


@register("opex_week_after", "timing", {},
          "每月選擇權到期週的下一週空手", "季節性", ref="Stivers & Sun 2013", pub_year=2013, niche="冷門")
def opex_week_after(B, asset):
    cal = D.calendar()
    # 到期日如果是假日，實際是前一個交易日（第三個週五在行事曆上不存在時，用那週最後一個交易日）
    opex = set()
    for p in pd.period_range(cal[0].to_period("M"), cal[-1].to_period("M"), freq="M"):
        first = p.to_timestamp()
        fri = pd.date_range(first, first + pd.offsets.MonthEnd(0), freq="W-FRI")
        if len(fri) < 3:
            continue
        d = fri[2]
        k = cal.searchsorted(d, side="right") - 1
        if k >= 0:
            opex.add(cal[k])
    flag = pd.Series(cal.isin(list(opex)), index=cal)
    # 到期日之後的 5 個交易日不持有
    off = flag.shift(1, fill_value=False).rolling(5, min_periods=1).max().astype(bool)
    return calendar_signal(~off).reindex(B.px(asset).index)


@register("pension_rebal", "timing", {"th": [0.03, 0.05]},
          "月底退休基金再平衡：當月股票贏債券超過 th，最後 3 個交易日空手", "資金流",
          ref="Harvey, Mazzoleni & Melone 2025", pub_year=2025, niche="冷門")
def pension_rebal(B, asset, th):
    cal = D.calendar()
    r = B.tr(asset).reindex(cal)
    b = bond_tr_from_yield().reindex(cal).where(D.close("^TNX").reindex(cal).ffill().notna())
    m = pd.Series(cal.to_period("M"), index=cal)
    eq_mtd = (1 + r.fillna(0)).groupby(m).cumprod() - 1
    bd_mtd = (1 + b.fillna(0)).groupby(m).cumprod() - 1
    rel = eq_mtd - bd_mtd
    k_from_end = m.groupby(m).cumcount(ascending=False)          # 0 = 月底最後一天
    # 在倒數第 5 天收盤（k=4）判斷；成立 → 訊號在 k=4,3,2 為 0 → 部位（晚 2 天生效）在 k=2,1,0 為 0
    decide = rel.where(k_from_end == 4).groupby(m).transform("max")
    sig = pd.Series(1.0, index=cal)
    sig[(k_from_end.isin([4, 3, 2])) & (decide >= th)] = 0.0
    sig[b.isna()] = np.nan                     # 沒有公債殖利率（1962 年以前）就不判斷
    return sig.reindex(B.px(asset).index).where(r.reindex(B.px(asset).index).notna())


@register("announce_days", "timing", {},
          "只在 FOMC、非農、CPI 公布日持有", "事件", ref="Savor & Wilson 2013", pub_year=2013, niche="冷門")
def announce_days(B, asset):
    cal = D.calendar()
    days = set(B.fomc())
    for sid in ("PAYEMS", "CPIAUCSL"):
        days |= set(pd.to_datetime(D.alfred_first(sid)["release_date"]))
    pos = cal.searchsorted(pd.DatetimeIndex(sorted(days)))
    pos = pos[pos < len(cal)]
    on = pd.Series(0.0, index=cal)
    on.iloc[pos] = 1.0
    on[on.index < B.fomc()[0]] = np.nan      # FOMC 日期 1994 年才開始有
    return calendar_signal(on).reindex(B.px(asset).index)


@register("near_52w_high", "timing", {},
          "離 52 週高點 10% 以內才持有", "價格", ref="Li & Yu 2012（大盤接近 52 週高點）", pub_year=2012, niche="冷門")
def near_52w_high(B, asset):
    return (dd_252(B.px(asset)) >= -0.10).astype(float).where(B.px(asset).expanding().count() >= 60)


@register("insider_exit", "timing", {},
          "內部人買進比例兩年 5 百分位以下（大量賣出）→ 出場 63 天", "內部人", ref="Seyhun 1988", pub_year=1988,
          niche="冷門")
def insider_exit(B, asset):
    g = B.insider()
    buy = g["firms_P"].rolling(21, min_periods=21).sum()
    sell = g["firms_S"].rolling(21, min_periods=21).sum()
    p = rpct(buy / (buy + sell), 504)
    return off_for(edge(p <= 5), 63).where(p.notna())


@register("cot_froth_exit", "timing", {},
          "資產管理人 S&P 期貨淨多單三年 95 百分位以上 → 出場 63 天", "籌碼", ref="實務者", niche="冷門")
def cot_froth_exit(B, asset):
    p = rpct(B.cot_tff("13874A", "Asset_Mgr"), 756)
    return off_for(edge(p >= 95), 63).where(p.notna())


@register("aaii_euphoria_exit", "timing", {},
          "AAII 多空差歷史 95 百分位以上 → 出場 63 天", "情緒", ref="Fisher & Statman 2000", pub_year=2000, niche="常見")
def aaii_euphoria_exit(B, asset):
    p = epct(B.aaii()["spread"].dropna(), 520)
    return off_for(edge(p >= 95), 63).where(p.notna())


@register("naaim_euphoria_exit", "timing", {},
          "NAAIM 歷史 95 百分位以上 → 出場 63 天", "情緒", ref="實務者", niche="半冷門")
def naaim_euphoria_exit(B, asset):
    p = epct(B.naaim(), 260)
    return off_for(edge(p >= 95), 63).where(p.notna())


@register("skew_exit", "timing", {},
          "CBOE SKEW 一年 95 百分位以上 → 出場 21 天", "期權", ref="實務者（「黑天鵝指數」）", niche="常見")
def skew_exit(B, asset):
    p = rpct(B.cboe("SKEW"), 252)
    return off_for(edge(p >= 95), 21).where(p.notna())


@register("breadth_divergence_exit", "timing", {},
          "指數離高點 3% 內、但 49 產業只有一半以下站上 200 日線 → 出場 63 天", "廣度", ref="實務者（廣度背離）",
          niche="半冷門")
def breadth_divergence_exit(B, asset):
    f = frac_above_sma(B.industries(), 200).reindex(B.px(asset).index)
    near = dd_252(B.px(asset)) >= -0.03
    return off_for(edge(near & (f <= 0.5)), 63).where(f.notna())


@register("abs_momentum", "timing", {},
          "絕對動能：過去 12 個月報酬 > 國庫券才持有（每月檢查）", "趨勢", ref="Antonacci 2014", pub_year=2014,
          niche="常見")
def abs_momentum(B, asset):
    px = B.px(asset)
    cash = (1 + B.rf()).cumprod().reindex(px.index)
    me = month_end_flags(D.calendar()).reindex(px.index).fillna(False)
    ex = (px / px.shift(252)) - (cash / cash.shift(252))
    sig = (ex[me] > 0).astype(float).where(ex[me].notna())
    return sig.reindex(px.index).ffill()


@register("liquidity_exit", "timing", {},
          "Fed 淨流動性（資產負債表 − TGA − 逆回購）13 週減少時出場", "總經", ref="實務者（2022 年流行）", niche="半冷門")
def liquidity_exit(B, asset):
    w = B.fred("WALCL", 1)
    tga = B.fred("WTREGEN", 1)
    rrp = B.fred("RRPONTSYD", 0) * 1000
    net = (w - tga.reindex(w.index).ffill() - rrp.reindex(w.index).ffill().fillna(0))
    ch = net - net.shift(63)
    return (ch >= 0).astype(float).where(ch.notna())


@register("copper_gold_exit", "timing", {},
          "銅金比跌破 200 日均線時出場", "跨資產", ref="實務者", niche="半冷門")
def copper_gold_exit(B, asset):
    r = (B.close("HG=F") / B.close("GC=F")).dropna()
    sma = r.rolling(200, min_periods=200).mean()
    return (r >= sma).astype(float).where(sma.notna())


@register("xly_xlp_exit", "timing", {},
          "非必需消費 / 必需消費（XLY/XLP）跌破 200 日均線時出場", "跨資產", ref="實務者", niche="半冷門")
def xly_xlp_exit(B, asset):
    r = (B.close("XLY") / B.close("XLP")).dropna()
    sma = r.rolling(200, min_periods=200).mean()
    return (r >= sma).astype(float).where(sma.notna())


@register("semis_exit", "timing", {},
          "半導體相對 NDX（SOX/NDX）跌破 100 日均線時出場", "跨資產", ref="實務者（半導體領先）", niche="半冷門")
def semis_exit(B, asset):
    r = (B.close("^SOX") / B.close("^NDX")).dropna()
    sma = r.rolling(100, min_periods=100).mean()
    return (r >= sma).astype(float).where(sma.notna())


@register("composite_froth_exit", "timing", {},
          "儀表板「過熱」規則：恐慌分數 ≤ 20 且高於 200 日線 10% 以上時出場", "綜合", ref="stock 專案 utils/decision.py",
          niche="—")
def composite_froth_exit(B, asset):
    f = composite_fear_score(B, asset)
    px = B.px(asset)
    ext = px / px.rolling(200, min_periods=200).mean() - 1
    return (~((f <= 20) & (ext > 0.10))).astype(float).where(f.notna() & ext.notna())


@register("dd_stop", "timing", {},
          "停損：離 52 週高點跌 15% 出場，站回 50 日線再回來", "價格", ref="Kaminski & Lo 2014", pub_year=2014,
          niche="常見")
def dd_stop(B, asset):
    px = B.px(asset)
    sma50 = px.rolling(50, min_periods=50).mean()
    return hysteresis(dd_252(px) <= -0.15, px > sma50).where(sma50.notna())


# ===========================================================================
# 第二輪（2026-10-19 看過第一輪結果後追加；參數在跑之前固定）
# ===========================================================================
@register("umich_low", "entry", {"pct": [5, 10]},
          "密西根消費者信心在歷史低檔（首次公布值）", "情緒", ref="實務者（消費者信心低點常接近股市底部）",
          niche="半冷門", batch=2)
def umich_low(B, asset, pct):
    first = D.alfred_first("UMCSENT").set_index("obs_date")
    # ALFRED 的首次公布值 1998-08 才開始；更早的月份用目前版本（密西根指數幾乎不修正），公布日 = 當月月底
    cur = D.fred_raw("UMCSENT")
    early = cur[cur.index < first.index.min()]
    a = pd.concat([pd.DataFrame({"value": early.values, "release_date": early.index + pd.offsets.MonthEnd(0)},
                                index=early.index), first[["value", "release_date"]]]).sort_index()
    p = a["value"].expanding(min_periods=120).rank(pct=True) * 100   # 在月資料上算百分位
    rel = a["release_date"]
    daily = D.avail_to_daily(p.values, rel.values, ffill_limit=D.MONTHLY_STALE)
    daily = daily.loc[: B.cutoff] if B.cutoff is not None else daily
    return edge(daily <= pct, daily)


@register("capitulation_volume", "entry", {},
          "恐慌爆量：SPY 成交量 ≥ 50 日均量 2.5 倍且當天跌 2% 以上", "價格", ref="實務者（Lowry 90% 下跌日的簡化版）",
          niche="半冷門", batch=2)
def capitulation_volume(B, asset):
    v = B.volume("SPY")
    r = B.close("SPY").pct_change()
    avg = v.rolling(50, min_periods=50).mean().shift(1)
    return edge((v >= 2.5 * avg) & (r <= -0.02), avg)


@register("industry_lows", "entry", {"th": [0.3, 0.5]},
          "49 產業中創一年新低的比例 ≥ th（全面性賣壓）", "廣度", ref="實務者（新低家數）", niche="冷門", batch=2)
def industry_lows(B, asset, th):
    lev = B.industries()
    lo = lev.rolling(252, min_periods=252).min()
    valid = lo.notna() & lev.notna()
    frac = ((lev <= lo) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    return edge(frac >= th, frac)


@register("dollar_spike", "entry", {},
          "美元指數 21 日急升（兩年 97 百分位）", "跨資產", ref="實務者（美元急升 = 全球去風險）", niche="半冷門", batch=2)
def dollar_spike(B, asset):
    p = rpct(B.close("DX-Y.NYB").pct_change(21).dropna(), 504)
    return edge(p >= 97, p)


@register("pc_complacency_exit", "timing", {},
          "股票 Put/Call 10 日均在一年 5 百分位以下（過度樂觀）→ 出場 21 天", "期權", ref="McMillan 等實務者",
          niche="常見", batch=2)
def pc_complacency_exit(B, asset):
    p = rpct(B.putcall("equity").rolling(10, min_periods=10).mean(), 252)
    return off_for(edge(p <= 5), 21).where(p.notna())


@register("vix_complacency_exit", "timing", {},
          "VIX 在一年 5 百分位以下（過度平靜）→ 出場 21 天", "波動率", ref="實務者", niche="常見", batch=2)
def vix_complacency_exit(B, asset):
    p = rpct(B.cboe("VIX"), 252)
    return off_for(edge(p <= 5), 21).where(p.notna())


def all_variants(kind: str | None = None):
    for spec in REGISTRY:
        if kind and spec.kind != kind:
            continue
        for params in spec.variants:
            yield spec, params, variant_name(spec, params)


# ===========================================================================
# 第三輪（2026-10-25，使用者要求；規則見 research/PREREG_batch3_2026-10-25.md，跑之前就固定）
# 「貨幣信用緊縮」：SLOOS、相對短期利率、EBP；「政策不確定性」：EPU。全部算進多重檢定。
# ===========================================================================
def _sloos_tight(B) -> pd.Series:
    """最新公布的 SLOOS ≥ 20，而且 ≥ 前一次公布值。"""
    s = B.sloos()
    return ((s["value"] >= 20) & (s["value"] >= s["prev"])).where(s["value"].notna() & s["prev"].notna())


def _tbill_rel(B) -> pd.Series:
    """3 個月國庫券利率 − 過去 252 個交易日的平均（百分點）。"""
    t = B.fred("DTB3", 1)
    return t - t.rolling(252, min_periods=200).mean()


def _ebp_rise(B) -> pd.Series:
    e = B.ebp()
    return (e["chg3"] >= 0.30).where(e["chg3"].notna())


@register("epu_spike", "entry", {"pct": [90, 95]},
          "政策不確定性（EPU）30 日平均剛站上 10 年 X 百分位", "總經",
          ref="Brogaard & Detzel 2015（Management Science）", pub_year=2015, niche="半冷門", batch=3)
def epu_spike(B, asset, pct):
    e = B.epu30()
    p = rpct(e, 2520, 1260)
    return edge(p >= pct, p)


@register("ebp_peak", "entry", {},
          "超額債券溢酬（EBP）≥ 1.0 且比 6 個月高點低 0.25 以上（信用壓力見頂回落）", "信用",
          ref="Gilchrist & Zakrajšek 2012（AER）", pub_year=2012, niche="冷門", batch=3)
def ebp_peak(B, asset):
    e = B.ebp()
    cond = (e["ebp"] >= 1.0) & (e["ebp"] <= e["max6"] - 0.25)
    return edge(cond.where(e["max6"].notna()), e["max6"])


@register("sloos_tight", "timing", {},
          "銀行收緊放款標準（SLOOS ≥ 20 且上升）時出場", "信用",
          ref="Chava, Gallmeyer & Park 2015（JME）", pub_year=2015, niche="冷門", batch=3)
def sloos_tight(B, asset):
    c = _sloos_tight(B)
    return (1.0 - c.astype(float)).where(c.notna())


@register("tbill_shock", "timing", {"th": [0.5, 1.0]},
          "3 個月國庫券比一年平均高 th 個百分點以上時出場（相對短期利率）", "總經",
          ref="Ang & Bekaert 2007（RFS）", pub_year=2007, niche="半冷門", batch=3)
def tbill_shock(B, asset, th):
    r = _tbill_rel(B)
    return (r < th).astype(float).where(r.notna())


@register("ebp_rise", "timing", {},
          "超額債券溢酬（EBP）3 個月上升 ≥ 0.30 個百分點時出場", "信用",
          ref="Gilchrist & Zakrajšek 2012（AER）", pub_year=2012, niche="冷門", batch=3)
def ebp_rise(B, asset):
    c = _ebp_rise(B)
    return (1.0 - c.astype(float)).where(c.notna())


@register("tightening_2of3", "timing", {},
          "貨幣信用緊縮三個條件（SLOOS、短期利率 1.0、EBP）至少兩個成立時出場", "總經",
          ref="組合（事先登記）", niche="冷門", batch=3)
def tightening_2of3(B, asset):
    a = _sloos_tight(B)
    r = _tbill_rel(B)
    b = (r >= 1.0).where(r.notna())
    c = _ebp_rise(B)
    df = pd.concat([a, b, c], axis=1)
    n = df.fillna(False).astype(float).sum(axis=1)
    return (n < 2).astype(float).where(df.notna().all(axis=1))

