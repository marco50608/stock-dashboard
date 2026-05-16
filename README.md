# Stock Dashboard

A personal dashboard for tracking US equity market positioning, sentiment, and breadth — focused on QQQ but generalizable to other tickers.

## What's in here

- **Overview** — 5 headline indicators for a daily glance
- **Capital Flow (籌碼面)** — Who's buying: CFTC COT, ETF flows, margin debt, dark pool, NAAIM
- **Sentiment (情緒面)** — Put/Call, VIX term structure, AAII, II survey, HY spread
- **Technical / Breadth** — Price + moving averages, % above 200dma, advance/decline, McClellan, new highs/lows
- **QQQ Specific** — Mag 7 concentration, semis (SMH) leadership, NDX breadth

Each indicator shows its current value plus historical percentile, so you can see whether you're in a typical or extreme regime.

## First-time setup

```bash
# 1. cd into this folder
cd "%USERPROFILE%\Desktop\stock"

# 2. (Optional but recommended) create a virtualenv
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get a free FRED API key
#    Register at: https://fredaccount.stlouisfed.org/apikeys
#    Then copy .env.example to .env and paste your key in

# 5. Run
streamlit run app.py
```

Streamlit will open the dashboard in your browser at http://localhost:8501.

## Data sources

| Source | What it provides | Key needed |
|---|---|---|
| Yahoo Finance (yfinance) | Prices, VIX, ETFs, individual stocks | No |
| FRED | HY spread, macro series | Yes (free) |
| CFTC | COT futures positioning | No |
| FINRA | Margin debt, dark pool ATS | No (scraped) |
| AAII | Retail sentiment survey | Manual CSV (login wall) |
| NAAIM | Active manager exposure | No |
| CBOE | Put/Call ratio | No (scraped via Yahoo `^CPC`) |

When a source fails or requires manual setup, the page shows a clear message — no silent failures.

## Notes

This is built for personal use. It is **not** investment advice. Indicators show probability and historical context, not predictions. Use it to recognize when you're at sentiment / positioning extremes, not to time the next tick.
