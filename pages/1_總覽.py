"""
Overview — 5 headline indicators + intraday + 10y yield.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import (
    DataUnavailable,
    find_support_levels,
    get_aaii_sentiment,
    get_hy_spread,
    get_intraday,
    get_margin_debt,
    get_price,
    get_put_call_ratio,
    get_us10y,
)
from utils.plots import apply_hover_style, indicator_card, price_with_ma


st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

symbol = st.session_state.get("symbol", "QQQ")
st.title(f"📊 總覽 — {symbol}")
st.caption("五個頭條指標 + 當日即時走勢。各分類頁有更深入的內容。")


def render_card(col, title, fig_meta, fmt="{:.2f}"):
    if fig_meta is None:
        col.warning(f"{title}：資料無法取得")
        return
    fig, meta = fig_meta
    if meta["value"] is None:
        col.warning(f"{title}：無資料")
        return
    col.markdown(f"**{title}**")
    col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    pct = meta["pct"]
    col.markdown(
        f"**{fmt.format(meta['value'])}{meta.get('unit','')}** "
        f"&nbsp;·&nbsp; <span style='color:{meta['color']}'>"
        f"{meta['label']}</span> "
        f"&nbsp;·&nbsp; 歷史第 {pct:.0f} 百分位",
        unsafe_allow_html=True,
    )
    if meta.get("description"):
        col.caption(meta["description"])


# ---------------------------------------------------------------------------
# 0. Intraday — today's live chart with support levels
# ---------------------------------------------------------------------------
st.subheader(f"0. {symbol} 當日即時走勢 + 支撐位")
with st.expander("📖 詳細說明", expanded=False):
    st.markdown("""
**這是什麼？**
- **上方**：今天的 5 分鐘 K 線圖（盤前/盤後通常沒資料）
- **下方藍/紅虛線**：基於過去 60 個交易日找出的兩個支撐位（local minima），
  代表近期跌到那邊就有買盤的價位
- **數字**：S1 = 最近的支撐、S2 = 第二近的支撐

**怎麼用？**
- 價格逼近 S1 時觀察是否止跌（買盤介入）
- 跌破 S1 → 看 S2 是否守住，跌破 S2 = 趨勢可能轉變
- 支撐位**不是預測**，是「歷史上這個價位有過買盤」的標記。需配合成交量、其他指標判斷。

**快取**：即時 5 分鐘 bar 每 5 分鐘刷新。網頁不會自動 reload，需要手動重新整理或按右上 Reload。
""")
try:
    # Daily for support detection
    daily = get_price(symbol, period="6mo")
    supports = find_support_levels(daily["Close"], n=2, lookback=60)

    # Intraday
    intra = get_intraday(symbol, interval="5m")
    if not intra.empty:
        fig_intra = go.Figure(go.Candlestick(
            x=intra.index,
            open=intra["Open"], high=intra["High"],
            low=intra["Low"], close=intra["Close"],
            name=symbol,
        ))
        cur_price = float(intra["Close"].iloc[-1])
        # Constrain Y axis to intraday range with small padding (don't let
        # support hlines stretch the axis and flatten the candlesticks)
        intra_low = float(intra["Low"].min())
        intra_high = float(intra["High"].max())
        padding = max((intra_high - intra_low) * 0.05, 0.5)
        y_min = intra_low - padding
        y_max = intra_high + padding
        # Overlay supports only if they're within / near the visible range
        for i, (dt, lvl) in enumerate(supports):
            if y_min - padding * 2 <= lvl <= y_max + padding * 2:
                color = "#3498db" if i == 0 else "#c0392b"
                fig_intra.add_hline(
                    y=lvl, line=dict(dash="dash", color=color, width=1.5),
                    annotation_text=f"S{i+1} ${lvl:.2f}",
                    annotation_position="right",
                    annotation_font_color=color,
                )
        fig_intra.update_layout(
            height=420,
            xaxis_rangeslider_visible=False,
            yaxis_title="USD",
            yaxis=dict(range=[y_min, y_max], fixedrange=False),
            title=f"當前 ${cur_price:.2f} · 最新 bar: {intra.index[-1].strftime('%H:%M')}",
        )
        apply_hover_style(fig_intra)
        st.plotly_chart(fig_intra, use_container_width=True)

        # Numeric support display — always show slots for S1 and S2
        c0, c1, c2 = st.columns(3)
        c0.metric("當前價", f"${cur_price:.2f}")
        slots = [c1, c2]
        for i in range(2):
            if i < len(supports):
                dt, lvl = supports[i]
                distance_pct = (lvl - cur_price) / cur_price * 100
                slots[i].metric(
                    f"S{i+1}（{dt.strftime('%m-%d')}）",
                    f"${lvl:.2f}",
                    f"{distance_pct:+.2f}%",
                )
            else:
                slots[i].metric(f"S{i+1}", "—", "近期無更深支撐")
    else:
        st.info("沒有當日即時資料（可能盤未開或假日）")
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 1. Price + MAs
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("1. 價格與趨勢（日線）")
with st.expander("📖 詳細說明", expanded=False):
    st.markdown("""
**這是什麼？** 過去 5 年的日 K 收盤價 + 50 日均線 + 200 日均線。
- **黑線**：每日收盤
- **橘線**：50 日均線（短期趨勢）
- **紅線**：200 日均線（長期趨勢）

**怎麼用？**
- 價格 > 50dma > 200dma 是典型的多頭排列
- 50dma 上穿 200dma = 黃金交叉，下穿 = 死亡交叉
- 價格距 200dma 超過 +15% 通常代表過熱
- 跌破 200dma 是長期趨勢轉變的常見訊號

**注意**：均線是滯後指標——告訴你「已發生的趨勢」，不是「即將發生的方向」。
""")
try:
    df = get_price(symbol, period="5y")
    fig = price_with_ma(df["Close"], mas=(50, 200))
    st.plotly_chart(fig, use_container_width=True)
    last_close = df["Close"].iloc[-1]
    ma200 = df["Close"].rolling(200).mean().iloc[-1]
    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("最新收盤", f"${last_close:.2f}")
    c2.metric("相對 50 日均線", f"{(last_close/ma50-1)*100:+.2f}%", "之上" if last_close > ma50 else "之下")
    c3.metric("相對 200 日均線", f"{(last_close/ma200-1)*100:+.2f}%", "之上" if last_close > ma200 else "之下")
except DataUnavailable as e:
    st.error(str(e))


# ---------------------------------------------------------------------------
# 2. US 10-year yield
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. 美國 10 年期公債殖利率")
with st.expander("📖 詳細說明", expanded=False):
    st.markdown("""
**這是什麼？** 美國 10 年期公債殖利率（10Y Treasury Yield），是全球風險資產定價的基準利率。

**怎麼用？**
- **殖利率上升 → 股市（尤其成長股、科技股）通常承壓**：折現率變高，未來現金流現值降低
- **殖利率下降 → 成長股受惠**：折現率變低，估值擴張
- QQQ 對 10Y 敏感度比 SPY 高（因為科技股久期長）

**歷史參考點**：
- < 2%：很低（2009–2021 多數時候）
- 2–4%：歷史正常區間
- > 4.5%：相對高，市場開始緊張
- > 5%：歷史高點（最近一次是 2023/10）

**注意**：殖利率突然飆升（短時間內 +50bp）比絕對水準更重要。
""")
try:
    y10 = get_us10y(period="10y")  # need 10y to cover 2018 Q4 reference
    fig, meta = indicator_card(
        y10, unit="%", invert=False,
        description="高殖利率 = 折現率高 = 對成長股不利。",
    )
    st.plotly_chart(fig, use_container_width=True)
    if meta["value"] is not None:
        st.write(f"當前：**{meta['value']:.2f}%** · {meta['label']} · 第 {meta['pct']:.0f} 百分位")

    st.info("💡 想看 10Y 殖利率在 2018Q4 / 2022 / 2024 三次熊市的走勢對比？側欄打開「📚 歷史熊市對照」頁")
except DataUnavailable as e:
    st.warning(f"10年期殖利率無法取得：{e}")


# ---------------------------------------------------------------------------
# 3. Sentiment & positioning cards
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. 情緒與籌碼卡片")
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

try:
    pc = get_put_call_ratio()
    col = next((c for c in pc.columns if "total" in str(c).lower()), pc.columns[0])
    series = pc[col].astype(float).rolling(5).mean().dropna()
    render_card(
        c1,
        "Put/Call 比率（5 日平均）· CBOE",
        indicator_card(
            series,
            invert=False,
            description="買 Put 量 ÷ 買 Call 量。高 = 恐慌；低 = 貪婪。",
        ),
    )
except DataUnavailable as e:
    c1.warning(f"Put/Call 無法取得：{e}")

try:
    aaii = get_aaii_sentiment()
    bull_col = next((c for c in aaii.columns if "bull" in str(c).lower()), None)
    bear_col = next((c for c in aaii.columns if "bear" in str(c).lower()), None)
    if bull_col and bear_col:
        spread = (aaii[bull_col].astype(float) - aaii[bear_col].astype(float)).dropna()
        render_card(
            c2,
            "AAII 散戶情緒（多空差）",
            indicator_card(
                spread,
                invert=True,
                description="散戶看多比例減看空比例。+30 以上 = 過度樂觀；-20 以下 = 過度悲觀。",
            ),
        )
    else:
        c2.warning("AAII 欄位無法辨識，請檢查 cache 內檔案")
except DataUnavailable as e:
    c2.warning(f"AAII 無法取得：{e}")

try:
    md = get_margin_debt()
    debt_col = None
    for c in md.columns:
        lc = str(c).lower()
        if ("margin" in lc and "debit" in lc) or "debit balance" in lc:
            debt_col = c
            break
    if debt_col is None:
        nums = md.select_dtypes(include="number")
        debt_col = nums.columns[0] if not nums.empty else None
    if debt_col is not None:
        s = md[debt_col].astype(float).dropna()
        yoy = s.pct_change(12) * 100
        render_card(
            c3,
            "FINRA 融資餘額 — 年增率",
            indicator_card(
                yoy, unit="%", invert=True,
                description="散戶借錢買股的年增率。+50% 以上的飆升歷史上常出現在大頂之前。",
            ),
        )
    else:
        c3.warning("FINRA 融資餘額：找不到數值欄位")
except DataUnavailable as e:
    c3.warning(f"融資餘額無法取得：{e}")

try:
    hy = get_hy_spread()
    render_card(
        c4,
        "高收益債利差（FRED · BAMLH0A0HYM2）",
        indicator_card(
            hy, unit=" bp", invert=False,
            description="高 = 信用市場壓力（常與股市底部相伴）。低 = 風險偏好高。",
        ),
        fmt="{:.0f}",
    )
except DataUnavailable as e:
    c4.warning(f"HY 利差無法取得：{e}")

st.markdown("---")
st.caption(
    "提醒：這些是統計上的脈絡指標，不是預測。"
    "多個指標同時極端時，歷史機率分佈會偏移——這是有用的資訊，但不是訊號燈。"
)
