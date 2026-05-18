"""
歷史熊市對照 — 把每個關鍵指標拿來跟過去三次熊市並列比較。

四張小圖 / 指標：
  - 2018 Q4 (Powell pivot)
  - 2022 升息熊市
  - 2024 AI 修正
  - 現在（近 6 個月）

X 軸是「自該期間第一天的交易日數」，方便比形狀。
"""
import pandas as pd
import streamlit as st

from utils.data import (
    DataUnavailable,
    get_aaii_sentiment,
    get_hy_spread,
    get_naaim,
    get_price,
    get_put_call_ratio,
    get_us10y,
    get_vix_term_structure,
)
from utils.plots import compare_historical_bears

st.set_page_config(page_title="歷史熊市對照", page_icon="📚", layout="wide")

symbol = st.session_state.get("symbol", "QQQ")
st.title("📚 歷史熊市對照")
st.caption("把當前指標走勢跟三次歷史熊市並列比較，看現在最像哪一段。")

with st.expander("📖 設計說明（點開）", expanded=False):
    st.markdown("""
**為什麼這頁有用？** 純看單一指標的當前讀值很難判斷「這算正常、警示、還是已經到底了」。
把它跟歷史熊市對比就能形成直覺：

| 期間 | 背景 |
|---|---|
| **2018 Q4 (Powell pivot)** | Powell 12 月還在升息，市場崩跌 20%。12/24 觸底，1 月 pivot 後快速反彈 |
| **2022 升息熊市** | 聯準會狂升息，QQQ 全年跌 33%，殖利率從 1.5% → 4.3% |
| **2024 AI 修正** | 7-8 月日圓 carry trade unwind + Yen 升息恐慌，QQQ 短暫跌 13% 後 V 型反彈 |
| **現在（近 6 個月）** | 你現在所在的位置 |

**怎麼用？**
- 比形狀，不比絕對值（每張小圖 Y 軸獨立縮放）
- 注意「現在」這張的走勢跟哪一段最像
- 多個指標都指向同一段歷史 → 機率上現在的情境近似那一段

**注意**：歷史不會完全重演。對照只是參考，不是預測。
""")


def section(title: str, intro: str, series, value_label: str = "value"):
    """Render one indicator's 4-panel comparison."""
    st.markdown(f"### {title}")
    st.caption(intro)
    if series is None or (hasattr(series, "empty") and series.empty):
        st.warning("無資料")
        return
    fig = compare_historical_bears(series, title="", value_label=value_label)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


# Load all data sources
with st.spinner("載入 10 年歷史資料中..."):
    qqq = vix = us10y = hy = pc = aaii = naaim = None
    errs = []
    try:
        qqq = get_price(symbol, period="10y")["Close"]
    except DataUnavailable as e:
        errs.append(f"QQQ: {e}")
    try:
        vix_df = get_vix_term_structure(period="10y")
        vix = vix_df.get("^VIX")
    except DataUnavailable as e:
        errs.append(f"VIX: {e}")
    try:
        us10y = get_us10y(period="10y")
    except DataUnavailable as e:
        errs.append(f"10Y: {e}")
    try:
        hy = get_hy_spread()
    except DataUnavailable as e:
        errs.append(f"HY: {e}")
    try:
        pc_df = get_put_call_ratio()
        pc = pc_df.iloc[:, 0].astype(float).dropna().rolling(5).mean()
    except DataUnavailable as e:
        errs.append(f"Put/Call: {e}")
    try:
        aaii_df = get_aaii_sentiment()
        bull = next((c for c in aaii_df.columns if "bull" in str(c).lower()), None)
        bear = next((c for c in aaii_df.columns if "bear" in str(c).lower()), None)
        if bull and bear:
            aaii = (aaii_df[bull].astype(float) - aaii_df[bear].astype(float)).dropna()
    except DataUnavailable as e:
        errs.append(f"AAII: {e}")
    try:
        naaim_df = get_naaim()
        expo = next((c for c in naaim_df.columns if "expos" in str(c).lower() or "number" in str(c).lower()), None)
        if expo is None:
            nums = naaim_df.select_dtypes(include="number")
            if not nums.empty: expo = nums.columns[0]
        if expo: naaim = naaim_df[expo].astype(float).dropna()
    except DataUnavailable as e:
        errs.append(f"NAAIM: {e}")

if errs:
    st.warning("部分資料源無法載入，會略過該指標：\n\n" + "\n".join(f"- {e}" for e in errs))


# ----- Sections -----

section(
    "1. QQQ 價格走勢",
    "看每個熊市期間 QQQ 從起點如何發展。形狀像「先跌後反彈」、「持續下行」、還是「V 型」？",
    qqq,
    "QQQ",
)

section(
    "2. 美國 10 年期公債殖利率",
    "升息熊市時殖利率會飆，AI 修正時殖利率反而鬆動。看現在更像哪一種利率環境。",
    us10y,
    "10Y %",
)

section(
    "3. VIX 恐慌指數",
    "VIX 在熊市初期通常飆升、在恐慌底部達到峰值。對比形狀可看出當前是「悶燒」還是「恐慌爆發」。",
    vix,
    "VIX",
)

section(
    "4. 高收益債利差 (HY OAS)",
    "信用市場壓力。利差擴張時通常與股市底部同時發生。看現在的利差走勢是否在累積壓力。",
    hy,
    "HY bp",
)

section(
    "5. AAII 散戶情緒（多空差）",
    "散戶看多 − 看空 %。負值極端時（散戶投降）常是底部訊號。看現在散戶情緒在哪個階段。",
    aaii,
    "Bull−Bear %",
)

section(
    "6. NAAIM 經理人曝險",
    "主動經理人 0–200 曝險。熊市初期經理人會持續減倉到極低（< 30），底部前後才回升。",
    naaim,
    "NAAIM",
)

section(
    "7. CBOE Put/Call 比率（5 日均值）",
    "高 = 散戶大量買 Put 避險 = 恐慌。常在底部附近出現極端高讀值。",
    pc,
    "P/C",
)

st.markdown("---")
st.caption(
    "💡 **解讀提示**：如果**多個指標**都顯示「現在的走勢」與某一段歷史熊市相似，"
    "代表當前情境的歷史 analogue 比較明確；如果各指標各自像不同段歷史，那就是 mixed signal，"
    "更需要謹慎判斷。**沒有任何 pattern 是必然會重演的**。"
)
