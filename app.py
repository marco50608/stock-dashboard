"""
Stock Dashboard — entry point.

執行方式：
    streamlit run app.py
"""
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "symbol" not in st.session_state:
    st.session_state.symbol = "QQQ"

st.title("📊 美股儀表板")
st.caption(f"最後重新整理：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

c1, c2 = st.columns([1, 5])
with c1:
    symbol = st.text_input(
        "標的代號",
        value=st.session_state.symbol,
        help="預設 QQQ。多數指標可換成其他標的；少數（七巨頭集中度、NDX 廣度）為 QQQ 專屬。",
    ).strip().upper()
    if symbol and symbol != st.session_state.symbol:
        st.session_state.symbol = symbol
        st.rerun()
with c2:
    st.info(
        "從左側側欄切換頁面。**總覽**頁有 5 個頭條指標；"
        "各分類頁深入單一面向。"
    )

st.markdown("---")

st.markdown(
    """
### 怎麼看這個儀表板

每個指標都會顯示三件事：

1. **當前數值** — 現在的位置
2. **歷史百分位** — 這個數值在歷史分佈中的位置
3. **解讀脈絡** — 極端讀值在歷史上的意義

目的是 **機率框架**，不是預測。當多個指標同時處於極端時，歷史上的機率分佈會偏移；
這對於部位調整是有用的資訊，不是紅綠燈。

### 頁面

| 頁面 | 內容 |
|---|---|
| 📊 總覽 | 5 個頭條指標：價格+均線、Put/Call、AAII、融資餘額、HY 利差 |
| 💰 籌碼面 | 誰在實際買 — COT、ETF 量、融資、暗池、NAAIM |
| 😊 情緒面 | 大家感覺如何 — Put/Call、VIX 期限結構、AAII、II、HY 利差 |
| 📈 技術／廣度 | 內部結構與價格行為 — 均線、%above200dma、A/D、McClellan |
| 🎯 QQQ 專屬 | 七巨頭集中度、半導體領導力、NDX 廣度 |

"""
)
