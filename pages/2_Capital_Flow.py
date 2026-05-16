"""
Capital Flow (籌碼面) — 誰在實際買進。
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import (
    DataUnavailable,
    TRACKED_FUNDS,
    get_cot_financial,
    get_latest_13f,
    get_margin_debt,
    get_naaim,
    get_price,
    parse_13f_holdings,
)
from utils.plots import apply_hover_style, indicator_card, _clip

st.set_page_config(page_title="籌碼面", page_icon="💰", layout="wide")

st.title("💰 籌碼面（資金流向）")
st.caption("誰在拿錢進場、誰在加槓桿、誰在大買大賣。")


# ---------------------------------------------------------------------------
# 1. Margin Debt
# ---------------------------------------------------------------------------
st.subheader("1. FINRA 融資餘額 — 散戶槓桿")
st.caption(
    "融資餘額會隨通膨自然成長，所以重點看年增率（YoY）。"
    "+50% 以上的爆衝在歷史上多次出現在大頂之前（2000、2007、2021）。"
    "資料優先用 FINRA Data API（需 .env 設 FINRA_CLIENT_ID/SECRET）；"
    "未設定時退為手動下載到 cache/finra_margin.xlsx。"
)
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
        st.caption(f"📅 最新資料：{s.index.max().strftime('%Y-%m')}　·　共 {len(s)} 個月")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**融資餘額（名目值）**")
            s_disp = _clip(s)
            fig = go.Figure(go.Scatter(
                x=s_disp.index, y=s_disp.values, mode="lines",
                line=dict(color="#3498db"),
            ))
            fig.update_layout(height=280)
            apply_hover_style(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**年增率 %**")
            yoy = s.pct_change(12) * 100
            yoy_disp = _clip(yoy)
            fig = go.Figure(go.Scatter(
                x=yoy_disp.index, y=yoy_disp.values, mode="lines",
                line=dict(color="#e67e22"),
            ))
            fig.add_hline(y=0, line=dict(dash="dot", color="gray"))
            fig.add_hline(y=50, line=dict(dash="dash", color="red"))
            fig.update_layout(height=280)
            apply_hover_style(fig)
            st.plotly_chart(fig, use_container_width=True)
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 2. CFTC COT positioning
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. CFTC 期貨持倉報告（COT）")
st.caption(
    "每週公布期貨的多空部位，分成 dealer、asset manager、leveraged funds、other reportables。"
    "小型投機者極端做多通常是反指標。"
)
try:
    cot = get_cot_financial()
    market_col = next((c for c in cot.columns if "market" in c.lower() and "name" in c.lower()), None)
    if market_col is None:
        market_col = next((c for c in cot.columns if "name" in c.lower() or "contract" in c.lower()), cot.columns[0])
    options = sorted({str(m) for m in cot[market_col].dropna().unique() if any(k in str(m).upper() for k in ["NASDAQ", "S&P", "MICRO", "MINI"])})
    if not options:
        options = sorted({str(m) for m in cot[market_col].dropna().unique()})[:30]
    pick = st.selectbox("選擇市場", options)
    row = cot[cot[market_col] == pick]
    if not row.empty:
        st.dataframe(row.T, use_container_width=True)
    st.caption("注意：這只是最近一週的快照。完整歷史請從 CFTC 下載年度 XLS。")
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 3. ETF flows proxy
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. ETF 資金流量代理（價 × 量）")
st.caption(
    "真實的 ETF 創建／贖回單位資料在 ICI／etf.com（付費）。這裡用簡單代理："
    "20 日累計成交金額。極端高點常標記行情轉折。"
)
sym = st.session_state.get("symbol", "QQQ")
try:
    df = get_price(sym, period="5y")
    dollar_vol = (df["Close"] * df["Volume"]).rolling(20).sum()
    dv_disp = _clip(dollar_vol.dropna())
    fig = go.Figure(go.Scatter(
        x=dv_disp.index, y=dv_disp.values, mode="lines",
        line=dict(color="#16a085"),
    ))
    fig.update_layout(height=300, yaxis_title=f"{sym} 20 日累計成交金額")
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 4. Volume anomaly — 替代暗池
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("4. 異常成交量偵測（替代暗池指標）")
st.caption(
    "暗池資料因 FINRA 反爬太嚴難自動化，改用「異常成交量」當代理。"
    "當日成交量超過 20 日均量 +2σ 標準差時，往往代表機構大單進出。"
    "綠色 = 上漲且爆量（買盤吸納），紅色 = 下跌且爆量（賣壓宣洩）。"
)
try:
    df = get_price(sym, period="2y")
    vol = df["Volume"]
    ret = df["Close"].pct_change()
    mu = vol.rolling(20).mean()
    sigma = vol.rolling(20).std()
    z = (vol - mu) / sigma
    anomalies = df.loc[z >= 2].copy()
    anomalies["z"] = z[z >= 2]
    anomalies["ret_pct"] = ret.loc[anomalies.index] * 100
    anomalies = anomalies.tail(30).sort_index()  # last 30 unusual days

    if not anomalies.empty:
        # Scatter: x=date, y=volume, color by return sign
        fig = go.Figure()
        # Background: 20d volume line
        vol_disp = _clip(vol)
        mu_disp = _clip(mu)
        fig.add_trace(go.Scatter(
            x=vol_disp.index, y=vol_disp.values, mode="lines",
            name="日成交量",
            line=dict(color="rgba(128,128,128,0.5)", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=mu_disp.index, y=mu_disp.values, mode="lines",
            name="20 日均量",
            line=dict(color="#3498db", dash="dash", width=1.5),
        ))
        # Anomaly markers
        clip_an = anomalies.loc[anomalies.index >= pd.Timestamp("2025-01-01")]
        fig.add_trace(go.Scatter(
            x=clip_an.index, y=clip_an["Volume"],
            mode="markers", name="異常成交日",
            marker=dict(
                size=[max(8, min(20, abs(z)*3)) for z in clip_an["z"]],
                color=["#27ae60" if r >= 0 else "#c0392b" for r in clip_an["ret_pct"]],
                line=dict(color="#000", width=0.5),
            ),
        ))
        fig.update_layout(height=380, yaxis_title=f"{sym} 成交量")
        apply_hover_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**最近 10 次異常成交日**")
        show = clip_an.tail(10).copy()
        show_df = pd.DataFrame({
            "日期": show.index.strftime("%Y-%m-%d"),
            "收盤": [f"${v:.2f}" for v in show["Close"]],
            "成交量": [f"{int(v):,}" for v in show["Volume"]],
            "Z 值": [f"+{v:.1f}σ" for v in show["z"]],
            "當日漲跌": [f"{v:+.2f}%" for v in show["ret_pct"]],
        })
        st.dataframe(show_df, hide_index=True, use_container_width=True)
    else:
        st.info("近期沒有 ±2σ 的異常成交日。")
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 5. NAAIM
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("5. NAAIM 經理人曝險指數")
st.caption(
    "美國主動投資經理人協會的每週調查。0（完全做空）到 200（兩倍槓桿做多）。"
    "超過 90 通常代表經理人擁擠地站在多方。"
)
try:
    n = get_naaim()
    expo_col = None
    for c in n.columns:
        if "expos" in str(c).lower() or "naaim number" in str(c).lower():
            expo_col = c
            break
    if expo_col is None:
        nums = n.select_dtypes(include="number")
        expo_col = nums.columns[0] if not nums.empty else None
    if expo_col is not None:
        s = n[expo_col].astype(float).dropna()
        fig, meta = indicator_card(
            s, invert=True,
            description="高 = 經理人擁擠做多。會均值回歸。",
        )
        st.plotly_chart(fig, use_container_width=True)
        if meta["value"] is None:
            st.warning("NAAIM 沒有可用資料")
        else:
            st.write(f"當前：**{meta['value']:.1f}** · {meta['label']} · 第 {meta['pct']:.0f} 百分位")
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 6. 13F via SEC EDGAR
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("6. 機構 13F 季報持股（SEC EDGAR）")
st.caption(
    "從 SEC EDGAR 公開 API 抓取追蹤基金的最新 13F-HR 季報持股。"
    "注意：13F 有 45 天申報延遲，是脈絡指標，不是時點工具。"
)

fund_pick = st.selectbox("選擇基金", list(TRACKED_FUNDS.keys()))
cik = TRACKED_FUNDS[fund_pick]
try:
    with st.spinner(f"從 SEC EDGAR 載入 {fund_pick} 最新 13F..."):
        meta = get_latest_13f(cik)
        holdings = parse_13f_holdings(cik, meta["accession"])
    c1, c2, c3 = st.columns(3)
    c1.metric("基金", meta.get("fund_name", fund_pick))
    c2.metric("申報日期", meta.get("filing_date", "—"))
    c3.metric("持股期末日", meta.get("period", "—"))

    total_value = holdings["value_usd"].sum() if "value_usd" in holdings.columns else 0
    st.markdown(f"**前 20 大持股**　·　組合總市值：${total_value/1e9:.1f}B　·　持股數量：{len(holdings)}")

    top20 = holdings.head(20).copy()
    if "value_usd" in top20.columns:
        top20["weight"] = top20["value_usd"] / total_value * 100
        show = pd.DataFrame({
            "#": range(1, len(top20)+1),
            "證券": top20["nameOfIssuer"],
            "市值": [f"${v/1e6:.0f}M" for v in top20["value_usd"]],
            "權重": [f"{v:.2f}%" for v in top20["weight"]],
        })
        if "shares" in top20.columns:
            show["股數"] = [f"{int(v):,}" if pd.notna(v) else "—" for v in top20["shares"]]
        st.dataframe(show, hide_index=True, use_container_width=True)

        # Concentration bar chart
        fig = go.Figure(go.Bar(
            x=top20["nameOfIssuer"].str[:20],
            y=top20["weight"],
            marker_color="#3498db",
            text=[f"{v:.1f}%" for v in top20["weight"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=380, yaxis_title="權重 %",
            title=f"{fund_pick} 前 20 大持股權重",
        )
        apply_hover_style(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(top20, hide_index=True, use_container_width=True)

    st.caption(
        f"📄 [在 SEC 看完整 filing](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR)"
    )
except DataUnavailable as e:
    st.warning(str(e))
