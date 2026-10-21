"""把原始資料整理成訊號要用的「當時拿得到」的日序列（Bundle）。

Bundle(cutoff=X) 會把所有輸入截斷在 X：用來做「截斷測試」——
同一個訊號用完整資料和截斷資料算，X 以前的值必須一模一樣，否則就是用到了未來資料。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data as D


@lru_cache(maxsize=None)
def insider_daily() -> pd.DataFrame:
    """全市場內部人公開市場買賣（Form 4、董事或高階主管、股價 ≥ 2 美元）。
    以「申報日 + 1 個交易日」為公開日，每天統計有人買進／賣出的公司家數。"""
    frames = []
    for p in sorted((D.RAW / "insider").glob("*.parquet")):
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["DOCUMENT_TYPE"].astype(str).str.strip() == "4"]
    rel = df["RPTOWNER_RELATIONSHIP"].fillna("")
    df = df[rel.str.contains("Director|Officer")]
    df["price"] = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    df["shares"] = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    df = df[(df["price"] >= 2) & (df["shares"] > 0)]
    df["filed"] = pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["filed"])
    df["value"] = df["price"] * df["shares"]
    df["avail"] = D.shift_bdays(df["filed"].values, 1)
    g = df.groupby(["avail", "TRANS_CODE"]).agg(firms=("ISSUERCIK", "nunique"), value=("value", "sum")).unstack(fill_value=0)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    cal = D.calendar()
    g = g.reindex(cal[(cal >= g.index.min()) & (cal <= g.index.max())]).fillna(0.0)
    return g


class Bundle:
    def __init__(self, cutoff=None):
        self.cutoff = pd.Timestamp(cutoff) if cutoff is not None else None
        self._c = {}

    def _cut(self, s):
        if self.cutoff is None:
            return s
        return s.loc[: self.cutoff]

    def _memo(self, key, fn):
        if key not in self._c:
            self._c[key] = self._cut(fn())
        return self._c[key]

    # 價格
    def tr(self, asset="NDX"):
        return self._memo(("tr", asset), lambda: D.total_return(asset))

    def px(self, asset="NDX"):
        return self._memo(("px", asset), lambda: (1 + D.total_return(asset)).cumprod())

    def close(self, tk):
        return self._memo(("close", tk), lambda: D.close(tk).dropna())

    def volume(self, tk):
        return self._memo(("vol", tk), lambda: D._raw_price(tk)["Volume"].reindex(D.calendar()).dropna())

    def rf(self):
        return self._memo("rf", D.rf_daily)

    # 波動率／期權
    def cboe(self, name):
        return self._memo(("cboe", name), lambda: D.cboe_index(name).dropna())

    def putcall(self, kind="equity"):
        return self._memo(("pc", kind), lambda: D.cboe_putcall(kind).dropna())

    # 調查與籌碼
    def aaii(self):
        return self._memo("aaii", lambda: D.aaii_avail().dropna(how="all"))

    def naaim(self):
        return self._memo("naaim", lambda: D.naaim_avail().dropna())

    def margin(self):
        return self._memo("margin", lambda: D.finra_margin_avail()["debit"].dropna())

    def claims(self):
        return self._memo("claims", lambda: D.claims_avail().dropna())

    def fred(self, sid, lag=1):
        return self._memo(("fred", sid, lag), lambda: D.fred_daily_avail(sid, lag).dropna())

    def fred_weekly(self, sid, lag_days):
        return self._memo(("fredw", sid, lag_days), lambda: D.fred_weekly_avail(sid, lag_days).dropna())

    def unrate_first(self):
        def f():
            a = D.alfred_first("UNRATE")
            return D.avail_to_daily(a["value"].values, a["release_date"].values, ffill_limit=D.MONTHLY_STALE).dropna()
        return self._memo("unrate", f)

    def insider(self):
        return self._memo("insider", insider_daily)

    def cot_tff(self, code, trader):
        """TFF：某類交易人淨部位 / 未平倉量（%），以公開日為索引。"""
        def f():
            raw = D.cot_tff_raw()
            cols = [f"{trader}_Positions_Long_All", f"{trader}_Positions_Short_All", "Open_Interest_All"]
            w = D.cot_series(raw, [code], cols)
            net = (w[cols[0]] - w[cols[1]]) / w[cols[2]] * 100
            return D.cot_to_daily(net).dropna()
        return self._memo(("tff", code, trader), f)

    def cot_legacy(self, code, trader):
        def f():
            raw = D.cot_legacy_raw()
            cols = [f"{trader} Positions-Long (All)", f"{trader} Positions-Short (All)", "Open Interest (All)"]
            w = D.cot_series(raw, [code], cols)
            net = (w[cols[0]] - w[cols[1]]) / w[cols[2]] * 100
            return D.cot_to_daily(net).dropna()
        return self._memo(("leg", code, trader), f)

    # 產業（Ken French 49 產業，市值加權，日報酬 → 指數）
    def industries(self):
        def f():
            r = D.french_industries(49, "value")
            r = r.loc["1960":]
            return (1 + r.fillna(0)).cumprod().where(r.notna())
        return self._memo("ind", f)

    def fomc(self):
        d = pd.read_csv(D.RAW / "fomc_dates.csv", parse_dates=["date"])["date"]
        d = pd.DatetimeIndex(d)
        if self.cutoff is not None:
            # 例行會議日期事先公布（隔年行程前一年就公告），所以未來的會議日期本來就知道；
            # 只截掉超過 cutoff 一年以上的部分，避免任何誤用
            d = d[d <= self.cutoff + pd.Timedelta(days=366)]
        return d
