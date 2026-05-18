"""
Stock Dashboard - entry point.

Run with:
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
    st.info("從左側側欄切換頁面。**總覽**頁有 5 個頭條指標；各分類頁深入單一面向。")

st.markdown("---")

st.markdown("""
### 怎麼看這個儀表板

每個指標都會顯示三件事：

1. **當前數值** — 現在的位置
2. **歷史百分位** — 這個數值在歷史分佈中的位置
3. **解讀脈絡** — 極端讀值在歷史上的意義

目的是 **機率框架**，不是預測。多個指標同時極端時，歷史機率分佈會偏移；
這對部位調整是有用的資訊，不是紅綠燈。

### 頁面

| 頁面 | 內容 |
|---|---|
| 📊 總覽 | 5 個頭條指標 + 當日即時走勢 + 支撐位 |
| 💰 籌碼面 | COT、ETF 量、融資、異常成交量、NAAIM、13F |
| 😊 情緒面 | CNN F&G、Put/Call、VIX 期限、AAII、HY 利差 |
| 📈 技術／廣度 | 均線、%above200dma、A/D、McClellan、新高新低 |
| 🎯 QQQ 專屬 | NDX 權重、七巨頭、領漲股 PE、SMH/QQQ、QQEW/QQQ |
| 📚 歷史熊市對照 | 把當前指標對比 2018Q4 / 2022 / 2024 三次熊市 |

---

### 📅 資料更新提醒

| 資料源 | 更新頻率 | 怎麼做 |
|---|---|---|
| **AAII 散戶情緒** | 每週四晚 / 週日 | 手動下載 .xls 覆蓋 cache/，git push |
| **FINRA 融資餘額** | 每月中 | 手動下載 .xlsx 覆蓋 cache/，git push |
| **CBOE Put/Call 比率** | 自動逐日累積 | 第一次跑 `notebooks/bootstrap_cboe.py` 補齊 2019/10 起的歷史；GitHub Actions 每週自動補新日子；應用開頁也會補（每次最多 90 天） |
| **NAAIM 經理人曝險** | 每週一 | 自動（GitHub Actions） |
| **其他所有資料** | 即時 / 每日 | 自動 |

**下載連結**：
- AAII: https://www.aaii.com/sentimentsurvey/sent_results
- FINRA: https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics
- CBOE Put/Call & 成交量/未平倉：
  - **自動（GitHub Actions 每週）**：抓 CBOE 每日 JSON，一次解析出 P/C 比率 + 各產品 volume/OI，分別累積到：
    - `cache/cboe_ratios_wide.csv` — `date + equity_pc, etp_pc, index_pc, spx_spxw_pc, total_pc, vix_pc, ...`
    - `cache/cboe_volume_oi_wide.csv` — 每天每產品的 `open_interest_call/put/total`、`volume_call/put/total`
  - **首次 bootstrap（本機，~10–15 分鐘）**：
    ```
    .venv\Scripts\python notebooks/bootstrap_cboe.py
    ```
    從 2019-10-07 起補齊兩個檔案。Raw JSON 存 `cache/cboe_raw/`（gitignored）。
  - **應用內 fallback**：每次開頁 `_cboe_daily_cache()` 也會自動補新天數（每次最多 90 天），萬一 GHA 失敗也能補上。
  - **2019/10 之前歷史（選做）**：去 https://www.cboe.com/us/options/market_statistics/historical_data/ 下載 archive，存成 `cache/cboe_pcr.csv`（欄位至少要有 `Date` 和 `Close`），會跟自動累積的資料合併

**手動更新指令**（在 stock 資料夾跑）：

```
git add cache/aaii_sentiment.xls cache/finra_margin.xlsx cache/cboe_ratios_wide.csv cache/cboe_volume_oi_wide.csv
git commit -m "Update market data"
git push
```

push 後 1-2 分鐘 Streamlit Cloud 自動重新部署。

### 提醒

這是個人自用工具，**不是投資建議**。沒有指標能預測未來；
這些工具能做到的，是告訴你目前狀況是普通還是極端——之後怎麼決定，是你的事。
""")
