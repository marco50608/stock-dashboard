"""
Sentiment (情緒面) — 投資人感受。
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import (
    DataUnavailable,
    get_aaii_sentiment,
    get_cnn_fear_greed,
    get_hy_spread,
    get_put_call_ratio,
    get_vix_term_structure,
)
from utils.plots import apply_hover_style, indicator_card, _clip

st.set_page_config(page_title="情緒面", page_icon="😊", layout="wide")

st.title("😊 情緒面（市場情緒）")
st.caption("投資人現在感覺如何——極端讀值適合做為反向指標。")


# ---------------------------------------------------------------------------
# 1. CNN Fear & Greed Index + 7 sub-indicators
# ---------------------------------------------------------------------------
st.subheader("1. CNN Fear & Greed Index（市場恐懼貪婪指標）")
with st.expander("📖 詳細說明（點開）", expanded=False):
    st.markdown("""
**這是什麼？** CNN Business 從 1995 年開始發布的綜合情緒指標，**0 = 極度恐懼、100 = 極度貪婪**。
由下面 7 個子指標等權平均而成：

1. **市場動能** — S&P 500 vs 125 日均線。指數遠高於 MA = 動能強 = 貪婪。
2. **股價強弱** — 紐約交易所創 52 週新高股票數 vs 新低數。新高多 = 貪婪。
3. **股價廣度** — McClellan 量能指標，看上漲股票成交量 vs 下跌股票。
4. **Put/Call 比率** — 過去 5 日選擇權 Put/Call 比。低 = 大家買 Call = 貪婪。
5. **市場波動** — VIX 與其 50 日均線比較。VIX 低 = 平靜 = 貪婪。
6. **避險需求** — 股票 20 日報酬 vs 公債 20 日報酬。股票領先 = 貪婪。
7. **垃圾債需求** — 高收益債與投資等級債利差。利差窄 = 風險胃納高 = 貪婪。

**怎麼用？** 極端值是反向指標：
- **< 25 (極度恐懼)** 歷史上常是逢低承接的好時機
- **> 75 (極度貪婪)** 歷史上常先於回檔出現

但**單看不準**——要跟其他指標、價格趨勢一起看。
""")

try:
    fg = get_cnn_fear_greed()
    score = fg.get("score")
    rating = fg.get("rating", "")
    if score is not None:
        # Big gauge for current
        color = ("#c0392b" if score < 25 else
                "#e67e22" if score < 45 else
                "#f1c40f" if score < 55 else
                "#27ae60" if score < 75 else
                "#16a085")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"當前：{rating}", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 25], "color": "rgba(192,57,43,0.2)"},
                    {"range": [25, 45], "color": "rgba(230,126,34,0.2)"},
                    {"range": [45, 55], "color": "rgba(241,196,15,0.2)"},
                    {"range": [55, 75], "color": "rgba(39,174,96,0.2)"},
                    {"range": [75, 100], "color": "rgba(22,160,133,0.2)"},
                ],
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        def _read(v):
            """previous_close etc. can be a float OR a {'score': x, 'rating': y} dict."""
            if isinstance(v, dict):
                return v.get("score"), v.get("rating", "")
            if isinstance(v, (int, float)):
                return float(v), ""
            return None, ""

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("當前", f"{score:.0f}", rating)
        pc_v, pc_r = _read(fg.get("previous_close"))
        if pc_v is not None:
            c2.metric("前一交易日", f"{pc_v:.0f}", pc_r)
        wk_v, _ = _read(fg.get("previous_1_week"))
        if wk_v is not None:
            c3.metric("一週前", f"{wk_v:.0f}")
        mo_v, _ = _read(fg.get("previous_1_month"))
        if mo_v is not None:
            c4.metric("一個月前", f"{mo_v:.0f}")

    # Historical chart
    hist = fg.get("historical", [])
    if hist:
        hist_df = pd.DataFrame(hist)
        hist_df["date"] = pd.to_datetime(hist_df["x"], unit="ms")
        hist_df = hist_df.set_index("date")
        st.markdown("**過去歷史走勢**")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df["y"], mode="lines",
            line=dict(color="#3498db", width=2),
            hovertemplate="%{y:.0f}<extra></extra>",
        ))
        for thr, color, txt in [(75, "#16a085", "極度貪婪"), (55, "#27ae60", "貪婪"),
                                  (45, "#f1c40f", "中性"), (25, "#e67e22", "恐懼")]:
            fig_hist.add_hline(y=thr, line=dict(dash="dot", color=color, width=1))
        fig_hist.update_layout(height=300, yaxis_title="F&G Score", yaxis_range=[0, 100])
        apply_hover_style(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    # 7 sub-indicators
    st.markdown("**7 個子指標當前讀值**")
    indicators = fg.get("indicators", {})
    cols = st.columns(4)
    items = list(indicators.items())
    for i, (key, sub) in enumerate(items):
        col = cols[i % 4]
        s_score = sub.get("score")
        s_rating = sub.get("rating", "")
        if s_score is not None:
            col.metric(sub.get("label", key), f"{s_score:.0f}", s_rating)
        else:
            col.metric(sub.get("label", key), "—")
except DataUnavailable as e:
    st.warning(f"CNN F&G 無法取得：{e}")


# ---------------------------------------------------------------------------
# 2. Put/Call ratio
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. CBOE Put/Call 比率")
with st.expander("📖 詳細說明", expanded=False):
    st.markdown("""
**這是什麼？** Chicago Board Options Exchange（CBOE）每日公布的選擇權交易量比率：
**Put 成交量 ÷ Call 成交量**。

**怎麼用？** 一般當作**反向指標**：
- **比率高（>1.0）** = 大家瘋狂買 Put 避險 = 恐慌 → 常出現在底部附近
- **比率低（<0.6）** = 大家瘋狂買 Call 看多 = 貪婪 → 常出現在頂部附近

**注意**：個別的單日讀值雜訊大，所以這裡用 5 日平均平滑。極端讀值的歷史百分位
比單看數值更有意義（這也是為什麼下方有「第 X 百分位」標示）。

**資料來源**：優先 Nasdaq.com 官方 API，失敗時退到 Stooq、CBOE、Yahoo，最後用 yfinance
選擇權鏈自累積。
""")

try:
    pc = get_put_call_ratio()
    cols = list(pc.columns)
    pick = st.selectbox("選擇序列", cols, index=0)
    raw = pc[pick].astype(float).dropna()
    n_days = len(raw)
    if n_days >= 30:
        s = raw.rolling(5).mean().dropna()
        smooth_label = "5 日均值"
    elif n_days >= 5:
        s = raw.rolling(min(3, n_days)).mean().dropna()
        smooth_label = "3 日均值"
    else:
        s = raw
        smooth_label = "原始值"
    fig, meta = indicator_card(s, invert=False, description="高 = 恐慌；低 = 貪婪。")
    st.plotly_chart(fig, use_container_width=True)
    if meta["value"] is None:
        st.warning(f"序列「{pick}」沒有可用資料")
    else:
        st.write(
            f"當前 {smooth_label}：**{meta['value']:.2f}** · {meta['label']} · "
            f"第 {meta['pct']:.0f} 百分位"
        )
        if n_days < 30:
            st.caption(
                f"📈 累積中：目前 {n_days} 個資料點。"
                f"由於外部 Put/Call 來源都失敗，這個指標是用 yfinance 選擇權鏈自動計算的，"
                f"每天開一次 app 就會多累積一筆，30 天後百分位才會有意義。"
            )
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 3. VIX term structure
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. VIX 期限結構")
with st.expander("📖 詳細說明", expanded=False):
    st.markdown("""
**這是什麼？** 不同到期日的 VIX（隱含波動率）對比：
- **VIX9D**：未來 9 天 S&P 500 隱含波動率
- **VIX**：未來 30 天（最常見的「恐慌指數」）
- **VIX3M**：未來 3 個月
- **VIX6M**：未來 6 個月

**正價差（contango）**：短天期 < 長天期 → 市場平靜，這是常態。
**逆價差（backwardation）**：短天期 > 長天期 → 市場有壓力，這是異常狀態。

**怎麼用？**
- `VIX9D > VIX3M` 是逆價差訊號，歷史上多次出現在恐慌賣壓的尾端，常是進場時機
- VIX 突然飆高（單日 +30%）也常先於回檔結束
- 但「VIX 很低 = 即將崩盤」的說法不準——低 VIX 可以維持很久

**注意**：VIX 是用 S&P 500 選擇權算的，不是 QQQ。NDX 對應的是 VXN，但流動性比 VIX 差很多。
""")
try:
    vix = get_vix_term_structure(period="3y")
    vix_disp = _clip(vix)
    fig = go.Figure()
    palette = {"^VIX9D": "#c0392b", "^VIX": "#e67e22", "^VIX3M": "#3498db", "^VIX6M": "#16a085"}
    for col in vix_disp.columns:
        fig.add_trace(go.Scatter(x=vix_disp.index, y=vix_disp[col],
                                 name=str(col), mode="lines",
                                 line=dict(color=palette.get(col, "#888"), width=1.5)))
    fig.update_layout(height=400, legend=dict(orientation="h"), yaxis_title="VIX")
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    if "^VIX9D" in vix.columns and "^VIX3M" in vix.columns:
        spread = (vix["^VIX9D"] - vix["^VIX3M"]).dropna()
        spread_disp = _clip(spread)
        st.markdown("**VIX9D − VIX3M**（正值 = 逆價差 = 壓力）")
        fig2 = go.Figure(go.Scatter(x=spread_disp.index, y=spread_disp.values, mode="lines",
                                    line=dict(color="#8e44ad")))
        fig2.add_hline(y=0, line=dict(dash="dot", color="gray"))
        fig2.update_layout(height=260)
        apply_hover_style(fig2)
        st.plotly_chart(fig2, use_container_width=True)
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 4. AAII Sentiment
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("4. AAII 散戶情緒調查")
with st.expander("📖 詳細說明", expanded=False):
    st.markdown("""
**這是什麼？** 美國散戶投資人協會（American Association of Individual Investors）
每週對會員的調查，問「未來 6 個月你看多、看空、還是中性」？得到三個百分比。

**多空差（Bull − Bear spread）** 是最常用的單一數字。

**怎麼用？** 經典的**反向指標**：
- **多空差 > +30** = 散戶過度樂觀 → 歷史上頂部附近常見
- **多空差 < −20** = 散戶過度悲觀 → 歷史上底部附近常見

**例子**：2022 年 9 月多空差 −43（極度悲觀），之後 6 個月 S&P 500 漲了 16%；
2018 年 1 月多空差 +37（極度樂觀），之後一個月跌 10%。

**注意**：
- 每週才公布一次（通常週四），所以時效性沒那麼好
- 樣本是 AAII 會員，主要是長期投資者，跟全市場散戶不完全一樣
- 單看不準，要跟其他指標配合
""")
try:
    aaii = get_aaii_sentiment()
    bull = next((c for c in aaii.columns if "bull" in str(c).lower()), None)
    bear = next((c for c in aaii.columns if "bear" in str(c).lower()), None)
    neut = next((c for c in aaii.columns if "neut" in str(c).lower()), None)
    if bull and bear:
        df = aaii[[c for c in [bull, bear, neut] if c]].astype(float).dropna()
        df_disp = _clip(df)
        palette = {str(bull): "#27ae60", str(bear): "#c0392b", str(neut): "#7f8c8d"}
        fig = go.Figure()
        for col in df_disp.columns:
            fig.add_trace(go.Scatter(x=df_disp.index, y=df_disp[col],
                                     name=str(col), mode="lines",
                                     line=dict(color=palette.get(str(col), "#888"))))
        fig.update_layout(height=350, legend=dict(orientation="h"), yaxis_title="%")
        apply_hover_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        spread = df[bull] - df[bear]
        st.markdown("**多空差（Bull − Bear）**")
        fig2, meta = indicator_card(
            spread, invert=True,
            description="極正 = 散戶過度樂觀（常見頂部）。極負 = 散戶過度悲觀（常見底部）。",
        )
        st.plotly_chart(fig2, use_container_width=True)
        if meta["value"] is None:
            st.warning("多空差沒有可用資料")
        else:
            st.write(f"當前多空差：**{meta['value']:+.1f}** · {meta['label']} · 第 {meta['pct']:.0f} 百分位")
    else:
        st.warning("無法辨識 Bull/Bear 欄位")
except DataUnavailable as e:
    st.warning(str(e))


# ---------------------------------------------------------------------------
# 5. HY Spread
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("5. 高收益債利差 — 信用市場情緒")
with st.expander("📖 詳細說明", expanded=False):
    st.markdown("""
**這是什麼？** ICE BofA US High Yield Index Option-Adjusted Spread（HY OAS）——
**垃圾債** vs **同期 US 公債**的殖利率差。單位是基點（basis points，1bp = 0.01%）。

**怎麼用？**
- **利差擴大（飆升）** = 投資人擔心違約風險 = 信用市場壓力，幾乎總是與股市壓力同時或先行發生
- **利差緊縮（很低）** = 風險胃納高，投資人不在意違約風險（風險偏好階段）

**歷史參考點**：
- 正常時期：300–500 bp
- 牛市熱絡：< 300 bp（2007、2021）
- 壓力：> 700 bp（2008/3、2020/3）
- 危機：> 1500 bp（2008 雷曼後）

**為什麼這個指標重要？**
信用市場通常比股市先反應壓力。當 HY 利差開始擴大但 S&P 500 還在創新高，是早期警訊。
反過來說，HY 利差開始收斂時，股市底部也往往近了。

**資料來源**：FRED `BAMLH0A0HYM2`，免費 API 但需要 key（已設定）。
""")
try:
    hy = get_hy_spread()
    fig, meta = indicator_card(
        hy, unit=" bp", invert=False,
        description="利差高 = 壓力；利差低 = 風險偏好高。",
    )
    st.plotly_chart(fig, use_container_width=True)
    if meta["value"] is None:
        st.warning("HY 利差沒有可用資料")
    else:
        st.write(f"當前：**{meta['value']:.0f} bp** · {meta['label']} · 第 {meta['pct']:.0f} 百分位")
except DataUnavailable as e:
    st.warning(str(e))
