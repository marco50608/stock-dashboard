"""
QQQ 專屬指標。

    - NDX 權重排行（精確，來自 slickcharts）
    - 七巨頭集中度（精確）
    - 七巨頭相對 QQQ 表現
    - 半導體領導力（SMH / QQQ）
    - NDX 廣度代理（QQEW / QQQ）
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import (
    DataUnavailable,
    MAG7,
    get_close,
    get_ndx_constituents,
    get_price,
)
from utils.plots import apply_hover_style, _clip

st.set_page_config(page_title="QQQ 專屬", page_icon="🎯", layout="wide")

st.title("🎯 QQQ 專屬指標")
st.caption(
    "QQQ 高度集中於超大型科技股。權重資料來自 slickcharts.com 的 NDX-100 表格。"
)


# ---------------------------------------------------------------------------
# 1. NDX top weights + Mag 7 concentration (real numbers)
# ---------------------------------------------------------------------------
st.subheader("1. NDX-100 權重排行 & 七巨頭集中度")
try:
    const = get_ndx_constituents()
    if "weight_pct" not in const.columns:
        st.warning("slickcharts 沒給權重欄位，僅顯示成分股清單")
        st.dataframe(const.head(25), hide_index=True, use_container_width=True)
    else:
        const_sorted = const.sort_values("weight_pct", ascending=False).reset_index(drop=True)

        # Mag 7 concentration
        mag7_share = const_sorted[const_sorted["Symbol"].isin(MAG7)]["weight_pct"].sum()
        top10_share = const_sorted.head(10)["weight_pct"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("七巨頭合計權重", f"{mag7_share:.2f}%")
        c2.metric("前 10 大合計", f"{top10_share:.2f}%")
        c3.metric("成分股數量", f"{len(const_sorted)}")

        st.markdown("**前 25 大成分股（依權重由高到低）**")
        show = const_sorted.head(25)[[c for c in ["Symbol", "Company", "weight_pct"] if c in const_sorted.columns]].copy()
        show["weight_pct"] = show["weight_pct"].map(lambda v: f"{v:.2f}%")
        show.columns = ["代號"] + (["名稱"] if "Company" in const_sorted.columns else []) + ["權重"]
        st.dataframe(show, hide_index=True, use_container_width=True)

        # Bar chart of Mag 7 weights (sorted desc)
        mag7_df = const_sorted[const_sorted["Symbol"].isin(MAG7)].sort_values("weight_pct", ascending=False)
        if not mag7_df.empty:
            fig = go.Figure(go.Bar(
                x=mag7_df["Symbol"],
                y=mag7_df["weight_pct"],
                text=[f"{v:.2f}%" for v in mag7_df["weight_pct"]],
                textposition="outside",
                marker_color="#3498db",
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            ))
            fig.update_layout(height=320, yaxis_title="權重 %", title="七巨頭在 NDX 的權重")
            apply_hover_style(fig)
            st.plotly_chart(fig, use_container_width=True)
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 2. Mag 7 relative performance vs QQQ
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. 七巨頭相對 QQQ 表現 — 誰在領漲、誰在落後")
try:
    px_full = get_close(MAG7 + ["QQQ"], period="2y").dropna()
    px = _clip(px_full)
    if px.empty:
        px = px_full.tail(252)
    norm = px.div(px.iloc[0]).mul(100)
    fig = go.Figure()
    for col in norm.columns:
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], mode="lines", name=str(col),
            line=dict(width=2.5 if col == "QQQ" else 1.2),
            hovertemplate="%{y:.1f}<extra>" + str(col) + "</extra>",
        ))
    fig.update_layout(height=400, legend=dict(orientation="h"), yaxis_title="標準化（起點=100）")
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    rel_30 = (px_full.pct_change(30).iloc[-1] - px_full["QQQ"].pct_change(30).iloc[-1]) * 100
    rel_30 = rel_30.drop("QQQ").sort_values(ascending=False)
    st.markdown("**過去 30 日相對 QQQ 的超額報酬**")
    fig_rel = go.Figure(go.Bar(
        x=rel_30.index.astype(str).tolist(),
        y=rel_30.values,
        marker_color=["#27ae60" if v >= 0 else "#c0392b" for v in rel_30.values],
        text=[f"{v:+.1f}%" for v in rel_30.values],
        textposition="outside",
        hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
    ))
    fig_rel.update_layout(height=300, yaxis_title="%")
    apply_hover_style(fig_rel)
    st.plotly_chart(fig_rel, use_container_width=True)
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 3. Top 10 NDX leaders — P/E and Forward P/E
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. NDX 前 10 領漲股 — P/E 與 Forward P/E")
st.caption(
    "從 NDX-100 成分股中，挑過去 30 日漲幅最高的 10 檔，列出 "
    "Trailing P/E（過去 12 個月）與 Forward P/E（未來 12 個月預估）。"
    "Forward P/E 低於 Trailing 通常代表市場預期獲利會成長；高於 Trailing 則相反。"
)
try:
    from utils.data import get_ndx_constituents, get_stock_valuations
    const = get_ndx_constituents()
    cand = const["Symbol"].dropna().astype(str).tolist()[:60]  # top 60 by weight as pool
    with st.spinner("抓取 NDX 成分股價格 / 估值..."):
        px2 = get_close(cand, period="6mo").dropna(how="all")
    if px2.empty:
        st.warning("無法取得成分股價格")
    else:
        ret_30 = px2.pct_change(30).iloc[-1].dropna().sort_values(ascending=False)
        top10 = ret_30.head(10).index.tolist()
        val = get_stock_valuations(tuple(top10))
        val["30d_return"] = [ret_30[s] * 100 for s in val["symbol"]]
        # Pretty
        show = val[["symbol", "name", "30d_return", "trailing_pe", "forward_pe", "market_cap"]].copy()
        show["30d_return"] = show["30d_return"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        show["trailing_pe"] = show["trailing_pe"].map(lambda v: f"{v:.1f}" if pd.notna(v) and v else "—")
        show["forward_pe"] = show["forward_pe"].map(lambda v: f"{v:.1f}" if pd.notna(v) and v else "—")
        show["market_cap"] = show["market_cap"].map(
            lambda v: f"${v/1e9:.0f}B" if pd.notna(v) and v else "—"
        )
        show.columns = ["代號", "名稱", "30日漲幅", "Trailing P/E", "Forward P/E", "市值"]
        st.dataframe(show, hide_index=True, use_container_width=True)

        # Spread between trailing and forward — a simple "expected growth" view
        spread_rows = []
        for _, r in val.iterrows():
            tp, fp = r.get("trailing_pe"), r.get("forward_pe")
            if tp and fp and not (pd.isna(tp) or pd.isna(fp)):
                spread_rows.append({"symbol": r["symbol"], "spread": float(tp) - float(fp)})
        if spread_rows:
            sdf = pd.DataFrame(spread_rows).sort_values("spread", ascending=False)
            fig_pe = go.Figure(go.Bar(
                x=sdf["symbol"], y=sdf["spread"],
                marker_color=["#27ae60" if v >= 0 else "#c0392b" for v in sdf["spread"]],
                text=[f"{v:+.1f}" for v in sdf["spread"]],
                textposition="outside",
                hovertemplate="%{x}: %{y:+.2f}<extra></extra>",
            ))
            fig_pe.update_layout(
                height=280,
                yaxis_title="Trailing P/E − Forward P/E",
                title="正值＝預期獲利成長；負值＝預期獲利衰退",
            )
            apply_hover_style(fig_pe)
            st.plotly_chart(fig_pe, use_container_width=True)
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 4. SMH/QQQ
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("4. 半導體領導力（SMH / QQQ 比值）")
st.caption("半導體歷史上是科技週期的領頭羊。SMH/QQQ 翻轉常是科技領導變化的早期訊號。")
try:
    px = get_close(["SMH", "QQQ"], period="5y").dropna()
    ratio = px["SMH"] / px["QQQ"]
    ratio_disp = _clip(ratio)
    ma50_full = ratio.rolling(50).mean()
    ma50_disp = _clip(ma50_full)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio_disp.index, y=ratio_disp.values, mode="lines", name="SMH/QQQ",
                             hovertemplate="%{y:.4f}<extra>SMH/QQQ</extra>"))
    fig.add_trace(go.Scatter(x=ma50_disp.index, y=ma50_disp.values, mode="lines", name="50 日均線",
                             line=dict(dash="dot"),
                             hovertemplate="%{y:.4f}<extra>50d MA</extra>"))
    fig.update_layout(height=350)
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)
    last = ratio.iloc[-1]
    ma50_last = ma50_full.iloc[-1]
    st.metric("SMH/QQQ 相對 50 日均線", f"{(last/ma50_last-1)*100:+.2f}%",
              "之上" if last > ma50_last else "之下")
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 5. NDX breadth proxy — QQEW/QQQ
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("5. NDX 廣度代理 — QQEW（等權重）/ QQQ（市值權重）")
st.caption(
    "QQEW 是 QQQ 的等權重版本。QQEW/QQQ 走跌時，市值權重版本被少數最大的股票"
    "拉著走——領導非常窄。"
)
try:
    px = get_close(["QQEW", "QQQ"], period="3y").dropna()
    ratio = px["QQEW"] / px["QQQ"]
    norm = ratio / ratio.iloc[0]
    norm_disp = _clip(norm)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm_disp.index, y=norm_disp.values, mode="lines",
                             name="QQEW/QQQ（標準化）",
                             hovertemplate="%{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(dash="dot", color="gray"))
    fig.update_layout(height=350)
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)
    cur = norm.iloc[-1]
    if cur < 0.95:
        st.warning(f"QQEW/QQQ = {cur:.3f}——顯著低於基準，領導集中於少數股。")
    elif cur > 1.05:
        st.info(f"QQEW/QQQ = {cur:.3f}——高於基準，廣度健康。")
    else:
        st.write(f"QQEW/QQQ = {cur:.3f}——靠近基準，廣度中性。")
except DataUnavailable as e:
    st.warning(str(e))
