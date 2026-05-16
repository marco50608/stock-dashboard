"""
Technical / Breadth — 價格行為與市場內部結構（基於 NDX-100 成分股）。

包含（與 MVP 重疊者標註）：
    - 價格 + 均線（也在總覽）
    - NDX 成分股在 50/200 日均線之上的比例
    - NDX 漲跌家數累計（A/D Line）
    - McClellan 震盪指標（NDX 版）
    - NDX 新高 vs 新低家數
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import (
    DataUnavailable,
    compute_ndx_breadth,
    get_price,
)
from utils.plots import apply_hover_style, price_with_ma, _clip

st.set_page_config(page_title="技術／廣度", page_icon="📈", layout="wide")

symbol = st.session_state.get("symbol", "QQQ")
st.title(f"📈 技術／廣度 — {symbol}")
st.caption("廣度資料用 NDX-100 成分股自行計算，比 Yahoo 的舊指數 ticker 可靠。")


# ---------------------------------------------------------------------------
# 1. Price + MAs
# ---------------------------------------------------------------------------
st.subheader("1. 價格與 20／50／200 日均線")
try:
    df = get_price(symbol, period="5y")
    fig = price_with_ma(df["Close"], mas=(20, 50, 200))
    st.plotly_chart(fig, use_container_width=True)
    last = df["Close"].iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("相對 20 日均線", f"{(last/df['Close'].rolling(20).mean().iloc[-1]-1)*100:+.2f}%")
    c2.metric("相對 50 日均線", f"{(last/df['Close'].rolling(50).mean().iloc[-1]-1)*100:+.2f}%")
    c3.metric("相對 200 日均線", f"{(last/df['Close'].rolling(200).mean().iloc[-1]-1)*100:+.2f}%")
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# Compute NDX breadth once (cached) and reuse
# ---------------------------------------------------------------------------
with st.spinner("計算 NDX 成分股廣度（首次需 10–30 秒，之後一天快取）..."):
    try:
        breadth = compute_ndx_breadth(period="3y")
    except DataUnavailable as e:
        breadth = None
        st.error(f"NDX 廣度計算失敗：{e}")


# ---------------------------------------------------------------------------
# 2. % above 50 / 200 dma
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. NDX 成分股在 50／200 日均線之上的比例")
st.caption(
    "市場廣度——健康的多頭走勢是廣的。"
    "若越來越少股票撐住指數，代表行情變窄、警訊浮現。"
)
if breadth is not None:
    fig = go.Figure()
    disp = _clip(breadth[["pct_above_50dma", "pct_above_200dma"]].dropna(how="all"))
    fig.add_trace(go.Scatter(x=disp.index, y=disp["pct_above_50dma"],
                             mode="lines", name="% 在 50 日均線之上",
                             line=dict(color="#3498db")))
    fig.add_trace(go.Scatter(x=disp.index, y=disp["pct_above_200dma"],
                             mode="lines", name="% 在 200 日均線之上",
                             line=dict(color="#c0392b")))
    fig.add_hline(y=50, line=dict(dash="dot", color="gray"))
    fig.update_layout(height=350, yaxis_title="%")
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)
    cur50 = breadth["pct_above_50dma"].dropna().iloc[-1]
    cur200 = breadth["pct_above_200dma"].dropna().iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("目前 % 在 50 日均線之上", f"{cur50:.1f}%")
    c2.metric("目前 % 在 200 日均線之上", f"{cur200:.1f}%")


# ---------------------------------------------------------------------------
# 3. A/D Line (NDX-based)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. NDX 漲跌家數累計（A/D Line）")
st.caption("每日上漲家數減下跌家數的累計值。用來確認或背離指數走勢。")
if breadth is not None:
    ad_cum = (breadth["advances"] - breadth["declines"]).cumsum()
    disp = _clip(ad_cum.dropna())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=disp.index, y=disp.values, mode="lines",
                             name="A/D 累計", line=dict(color="#16a085"),
                             hovertemplate="%{y:.0f}<extra>A/D</extra>"))
    fig.update_layout(height=300)
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 4. McClellan Oscillator (NDX-based)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("4. McClellan 震盪指標（19/39 日 EMA of A−D，NDX 版）")
st.caption(
    "上漲減下跌家數的 19 日 EMA 與 39 日 EMA 之差。"
    "極端負值代表內部超賣；極端正值代表內部超買。"
)
if breadth is not None:
    ad = (breadth["advances"] - breadth["declines"]).dropna()
    ema19 = ad.ewm(span=19, adjust=False).mean()
    ema39 = ad.ewm(span=39, adjust=False).mean()
    mcc = ema19 - ema39
    disp = _clip(mcc)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=disp.index, y=disp.values, mode="lines",
                             line=dict(color="#8e44ad"),
                             hovertemplate="%{y:+.1f}<extra>McClellan</extra>"))
    # Adaptive thresholds: use 5th / 95th percentile of full history
    upper = mcc.quantile(0.95)
    lower = mcc.quantile(0.05)
    fig.add_hline(y=upper, line=dict(dash="dash", color="red"))
    fig.add_hline(y=lower, line=dict(dash="dash", color="green"))
    fig.add_hline(y=0, line=dict(dash="dot", color="gray"))
    fig.update_layout(height=300)
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("當前 McClellan", f"{mcc.iloc[-1]:+.1f}",
              help=f"歷史 5%/95% 分位門檻：{lower:+.1f} / {upper:+.1f}")


# ---------------------------------------------------------------------------
# 5. New Highs vs New Lows (NDX-based)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("5. NDX 52 週新高 vs 新低")
st.caption(
    "當指數走高但新高家數越來越少、新低家數擴張時，內部動能正在減弱。"
    "新低持續擴張是真正的警訊。"
)
if breadth is not None:
    nh = breadth["new_highs_52w"]
    nl = breadth["new_lows_52w"]
    net = (nh - nl).dropna()
    disp = _clip(net)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=disp.index, y=disp.values,
                          marker_color=["#27ae60" if v >= 0 else "#c0392b" for v in disp.values],
                          name="新高−新低",
                          hovertemplate="%{y:+.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(dash="dot", color="gray"))
    fig.update_layout(height=300, yaxis_title="家數")
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)
    cnh = nh.dropna().iloc[-1]
    cnl = nl.dropna().iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("最新新高家數", int(cnh))
    c2.metric("最新新低家數", int(cnl))
    c3.metric("淨值", f"{int(cnh - cnl):+d}")
