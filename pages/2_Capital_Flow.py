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
    get_cboe_volume_oi,
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
        r = row.iloc[0]
        # Helper: pick the first column matching any of several keywords (case-insensitive)
        def pick_col(*keywords):
            for c in cot.columns:
                cl = str(c).lower()
                if all(k.lower() in cl for k in keywords):
                    return c
            return None

        def fmt(v):
            try: return f"{int(float(v)):,}"
            except Exception: return str(v)

        # Build readable positioning table
        groups = [
            ("總未平倉量", [
                ("Open Interest", pick_col("open_interest", "all"))]),
            ("Dealer / Intermediary（造市商，避險為主）", [
                ("多單", pick_col("dealer", "long")),
                ("空單", pick_col("dealer", "short")),
                ("Spread", pick_col("dealer", "spread"))]),
            ("Asset Manager / Institutional（資產管理機構）", [
                ("多單", pick_col("asset", "long")),
                ("空單", pick_col("asset", "short")),
                ("Spread", pick_col("asset", "spread"))]),
            ("Leveraged Funds（避險基金 / CTA — 通常較積極）", [
                ("多單", pick_col("lev", "money", "long")),
                ("空單", pick_col("lev", "money", "short")),
                ("Spread", pick_col("lev", "money", "spread"))]),
            ("Other Reportables（其他大型可申報部位）", [
                ("多單", pick_col("other", "rept", "long")),
                ("空單", pick_col("other", "rept", "short")),
                ("Spread", pick_col("other", "rept", "spread"))]),
            ("Non-Reportable（小型投機者 — 反指標）", [
                ("多單", pick_col("nonrept", "long")),
                ("空單", pick_col("nonrept", "short"))]),
        ]

        for group_name, rows in groups:
            st.markdown(f"**{group_name}**")
            data = []
            for label, col in rows:
                if col is not None and col in r.index:
                    val = r[col]
                    data.append({"項目": label, "口數": fmt(val)})
            if data:
                # Also compute net (long - short) for each group when applicable
                longs = next((d for d in data if d["項目"] == "多單"), None)
                shorts = next((d for d in data if d["項目"] == "空單"), None)
                if longs and shorts:
                    try:
                        net = int(float(str(longs["口數"]).replace(",", ""))) - int(float(str(shorts["口數"]).replace(",", "")))
                        data.append({"項目": "**淨多單**", "口數": f"**{net:+,}**"})
                    except Exception:
                        pass
                st.table(pd.DataFrame(data).set_index("項目"))

        # Show filing/report date if available
        date_col = next((c for c in cot.columns if "report" in c.lower() and "date" in c.lower()), None)
        if date_col and date_col in r.index:
            st.caption(f"📅 報告日期：{r[date_col]}　·　資料來源：CFTC Traders in Financial Futures (週公布)")
        st.caption("**Leveraged Funds 淨部位**通常是最敏感的——他們轉向時市場常跟著動。"
                   "**Non-Reportable 淨部位**極端時往往是反指標。")
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


# ---------------------------------------------------------------------------
# 7. CBOE 選擇權市場活動 — Volume / Open Interest
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("7. CBOE 選擇權市場活動（成交量 + 未平倉）")
with st.expander("📖 詳細說明", expanded=False):
    st.markdown("""
**這是什麼？** 從 CBOE 官方每日 JSON 抓的全市場選擇權成交量（VOLUME）與未平倉量（OPEN INTEREST），
按產品（EQUITY OPTIONS / INDEX OPTIONS / VIX / SPX+SPXW / ETP / ...）拆分。

**三張子圖在看什麼？**

1. **市場熱度（總成交量 + 20 日均線）**：
   日成交量遠超過 20DMA = 投機/恐慌活動；遠低於 20DMA = 市場觀望。
   重要的是**相對水準**，不是絕對數值。

2. **散戶 vs 機構偏好（Equity / (Equity + Index) 成交量比，5 日均線）**：
   單一股票（EQUITY）選擇權主要是散戶；指數（INDEX, SPX, RUT）選擇權主要是機構。
   - 比值高（> 0.65）：散戶活躍 → 個股投機行情、可能接近頂
   - 比值低（< 0.5）：機構主導、避險為主 → 通常在恐慌或盤整期

3. **VIX 避險堆積（VIX OI 總量 + 60 日均線）**：
   未平倉量代表「現有的避險倉位」。OI 攀升 = 越來越多人建立 VIX 多頭部位。
   持續高 OI 通常代表市場避險需求積壓——可能是擔憂事件、也可能是技術性避險。

**注意**：成交量只反映「有人買賣」，**不告訴你方向**。Put/Call 比率（已在情緒面頁）才反映看漲/看跌。
這裡的指標主要看**活動規模**和**參與者結構**。
""")

try:
    voloi = get_cboe_volume_oi()

    # --- 7a. Total options volume + 20d MA ---
    total = voloi[voloi["product"] == "SUM OF ALL PRODUCTS"].set_index("date").sort_index()
    if not total.empty:
        vol = total["volume_total"].astype(float).dropna()
        ma20 = vol.rolling(20).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vol.index, y=vol / 1e6, name="日成交量",
            marker_color="#95a5a6", opacity=0.6,
            hovertemplate="%{x|%Y-%m-%d}<br>成交量: %{y:.1f}M 口<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=ma20.index, y=ma20 / 1e6, name="20 日均線",
            line=dict(color="#e74c3c", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>20DMA: %{y:.1f}M<extra></extra>",
        ))
        cur_val = float(vol.iloc[-1])
        ma_val = float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else None
        rel_label = ""
        if ma_val:
            ratio = cur_val / ma_val
            rel_label = f"（當前 {ratio:.2f}× 20DMA）"
        fig.update_layout(
            title=f"7a. 全市場選擇權日成交量 + 20DMA  {rel_label}",
            yaxis_title="百萬口",
            height=380,
            hovermode="x unified",
        )
        apply_hover_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("最新成交量", f"{cur_val/1e6:.1f}M 口")
        if ma_val:
            c2.metric("20 日均線", f"{ma_val/1e6:.1f}M 口")
            c3.metric("相對 20DMA", f"{(cur_val/ma_val - 1)*100:+.1f}%")

    # --- 7b. Equity vs Index volume ratio ---
    eq = voloi[voloi["product"] == "EQUITY OPTIONS"].set_index("date").sort_index()["volume_total"]
    idx = voloi[voloi["product"] == "INDEX OPTIONS"].set_index("date").sort_index()["volume_total"]
    if not eq.empty and not idx.empty:
        df_r = pd.concat({"equity": eq, "index": idx}, axis=1).dropna()
        df_r["ratio"] = df_r["equity"] / (df_r["equity"] + df_r["index"])
        ratio_ma = df_r["ratio"].rolling(5).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_r.index, y=df_r["ratio"], name="日比值",
            line=dict(color="#bdc3c7", width=1),
            opacity=0.5,
            hovertemplate="%{x|%Y-%m-%d}<br>當日: %{y:.3f}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=ratio_ma.index, y=ratio_ma, name="5 日均線",
            line=dict(color="#3498db", width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>5DMA: %{y:.3f}<extra></extra>",
        ))
        # Reference lines
        fig2.add_hline(y=0.65, line=dict(dash="dash", color="#e74c3c", width=1),
                       annotation_text="散戶活躍 (>0.65)", annotation_position="right")
        fig2.add_hline(y=0.50, line=dict(dash="dash", color="#95a5a6", width=1),
                       annotation_text="平衡 (0.5)", annotation_position="right")
        cur_ratio = float(ratio_ma.dropna().iloc[-1]) if ratio_ma.notna().any() else None
        fig2.update_layout(
            title=f"7b. 散戶偏好 = Equity / (Equity + Index) 成交量比"
                  + (f"   當前 5DMA: {cur_ratio:.3f}" if cur_ratio else ""),
            yaxis_title="比值",
            yaxis=dict(range=[0.3, 0.85]),
            height=380,
            hovermode="x unified",
        )
        apply_hover_style(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        if cur_ratio is not None:
            if cur_ratio > 0.65:
                interp = "🟠 散戶活躍區 — 個股投機行情，歷史上常接近階段頂"
            elif cur_ratio < 0.50:
                interp = "🔵 機構主導 — 通常是恐慌或盤整期，避險為主"
            else:
                interp = "⚪ 平衡區間"
            st.caption(interp)

    # --- 7c. VIX OI + 60d MA ---
    vix_oi = voloi[voloi["product"] == "CBOE VOLATILITY INDEX (VIX)"].set_index("date").sort_index()
    if not vix_oi.empty:
        oi = vix_oi["open_interest_total"].astype(float).dropna()
        oi_ma = oi.rolling(60).mean()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=oi.index, y=oi / 1e6, name="VIX OI 總量",
            line=dict(color="#9b59b6", width=1.5),
            fill="tozeroy", fillcolor="rgba(155,89,182,0.15)",
            hovertemplate="%{x|%Y-%m-%d}<br>OI: %{y:.1f}M 口<extra></extra>",
        ))
        fig3.add_trace(go.Scatter(
            x=oi_ma.index, y=oi_ma / 1e6, name="60 日均線",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>60DMA: %{y:.1f}M<extra></extra>",
        ))
        cur_oi = float(oi.iloc[-1])
        cur_ma = float(oi_ma.iloc[-1]) if pd.notna(oi_ma.iloc[-1]) else None
        rel_oi = f"（當前 {(cur_oi/cur_ma - 1)*100:+.1f}% vs 60DMA）" if cur_ma else ""
        fig3.update_layout(
            title=f"7c. VIX 未平倉量總量 + 60DMA  {rel_oi}",
            yaxis_title="百萬口",
            height=380,
            hovermode="x unified",
        )
        apply_hover_style(fig3)
        st.plotly_chart(fig3, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("VIX OI 當前", f"{cur_oi/1e6:.1f}M 口")
        if cur_ma:
            c2.metric("相對 60DMA", f"{(cur_oi/cur_ma - 1)*100:+.1f}%")

        # Percentile vs own history
        pct = (oi.rank(pct=True).iloc[-1]) * 100
        if pct >= 80:
            badge = "🔴 歷史高位區 — 避險倉位堆積"
        elif pct <= 20:
            badge = "🟢 歷史低位區 — 市場放鬆警戒"
        else:
            badge = "⚪ 歷史正常區間"
        st.caption(f"VIX OI 歷史百分位：第 {pct:.0f} 百分位 · {badge}")

except DataUnavailable as e:
    st.warning(f"CBOE 成交量/未平倉資料無法取得：{e}")
