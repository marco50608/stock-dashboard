"""
進出場訊號 — 依 research/REPORT.md（2026-10）的研究結論。

結論很簡單，所以這頁也很簡單：
  1. 新存的錢：照計畫直接買（留錢等訊號，歷史上一律輸給直接買）
  2. 恐慌觀察燈：VIX ≥ 40 才是「全面恐慌」，可以從你事先決定的現金桶加碼（觀察名單，不是驗證過的訊號）
  3. 出場：沒有任何出場訊號通過驗證
邏輯在 utils/signals_live.py；歷史證據在 utils/signal_evidence.json（由 research/export_evidence.py 產生）。
"""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import signals_live as L
from utils.plots import apply_hover_style

st.set_page_config(page_title="進出場訊號", page_icon="🎯", layout="wide")

symbol = st.session_state.get("symbol", "QQQ")
st.title(f"🎯 進出場訊號 — {symbol}")
st.caption("依 research/REPORT.md：106 個訊號變體嚴格驗證後，沒有一個通過。"
           "這頁只做「提醒和紀錄」，不是交易訊號，也不是投資建議。")

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "cache" / "entry_params.json"
# 只在自己的電腦上存檔：本機有 .env（雲端部署沒有），或明確設定 STOCK_LOCAL_SAVE=1。
# 部署到 Streamlit Cloud 時檔案是所有訪客共用的，絕不能把個人金額寫進去。
LOCAL_SAVE = (ROOT / ".env").exists() or os.environ.get("STOCK_LOCAL_SAVE") == "1"
DEFAULT_PARAMS = {"total_assets": 1_000_000, "base_equity_pct": 40, "tilt_pp": 10, "cash_bucket": 200_000,
                  "hold_months": 6}


@st.cache_data(ttl=3600, show_spinner=False)
def _signal(sym: str) -> dict:
    return L.compute_live_signal(sym)


def _pct(x, d=1, sign=True):
    return L._pct(x, d, sign)


with st.spinner("抓取 VIX、產業 ETF、COR3M…"):
    try:
        res = _signal(symbol)
    except Exception as e:  # noqa: BLE001
        st.error(f"計算失敗：{e}")
        st.stop()

ev = res.get("evidence") or {}
vs, ss, cs, ds, tr = res["vix"], res["sectors"], res["cor3m"], res["drawdown"], res["trend10m"]
state = res["state"]
v40 = ev.get("vix40") or {}

# ---------------------------------------------------------------------------
# 上方兩張卡
# ---------------------------------------------------------------------------
STYLE = {L.STATE_PANIC: ("#1E7A46", "🟢"), L.STATE_WATCH: ("#B9770E", "🟡"), L.STATE_NORMAL: ("#34495E", "⚪")}
box, emoji = STYLE.get(state, ("#7f8c8d", "⚪"))

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "<div style='background:#1F4E79;color:#fff;padding:14px 18px;border-radius:10px;min-height:150px'>"
        "<span style='font-size:13px;opacity:.85'>新存的錢</span><br>"
        "<span style='font-size:22px;font-weight:700'>照計畫直接買</span><br>"
        "<span style='font-size:14px'>不要留一部分等恐慌。研究裡每一個訊號（包括 VIX ≥ 40），"
        "「一半先放現金等訊號」的期末財富都比直接買少。</span></div>",
        unsafe_allow_html=True)
with c2:
    if vs.get("value") is None:
        detail = "無法取得 VIX"
    elif state == L.STATE_PANIC:
        detail = f"VIX {vs['value']:.1f}（≥ {L.VIX_PANIC:.0f}）：可依下方你寫好的規則，從現金桶加碼"
    else:
        detail = (f"VIX {vs['value']:.1f}，距「全面恐慌」{L.VIX_PANIC:.0f} 還差 {vs['to_panic']:.1f} 點"
                  + ("（已達「留意」）" if state == L.STATE_WATCH else ""))
    conf = "；賣壓已全面化：" + "、".join(res["confirmations"]) if res.get("confirmations") else ""
    st.markdown(
        f"<div style='background:{box};color:#fff;padding:14px 18px;border-radius:10px;min-height:150px'>"
        f"<span style='font-size:13px;opacity:.85'>恐慌觀察燈（觀察名單，未通過多重檢定）</span><br>"
        f"<span style='font-size:22px;font-weight:700'>{emoji} {state or '—'}</span><br>"
        f"<span style='font-size:14px'>{detail}{conf}</span></div>",
        unsafe_allow_html=True)

st.write("")
m1, m2, m3, m4 = st.columns(4)
m1.metric("VIX 收盤", "—" if vs.get("value") is None else f"{vs['value']:.1f}",
          help=f"資料日 {vs.get('date')}。≥ {L.VIX_WATCH:.0f} 留意、≥ {L.VIX_PANIC:.0f} 全面恐慌。")
m2.metric("產業 ETF 在一年新低", "—" if ss.get("count") is None else f"{ss['count']} / {ss.get('of', 9)} 檔",
          help=f"9 檔產業 SPDR；≥ {L.SECTOR_CONFIRM} 檔 = 賣壓已全面化（只當確認，不單獨觸發）。")
m3.metric("COR3M 一年百分位", "—" if cs.get("pct") is None else f"{cs['pct']:.0f}%",
          help="CBOE 3 個月隱含相關性。≥ 95 百分位 = 大家一起賣（只當確認）。")
m4.metric(f"{symbol} 距高點", _pct(ds.get("from_high")), help="只當脈絡，不是訊號。")

if vs.get("last_panic"):
    st.caption(f"上次 VIX 首次站上 {L.VIX_PANIC:.0f}：{vs['last_panic']}（{vs['days_since_panic']} 天前）。"
               f"過去一年 VIX 最高 {vs['max_1y']:.1f}。")

# VIX 圖
vix = res.get("series", {}).get("vix")
if vix is not None and len(vix):
    v = vix.loc[vix.index >= vix.index[-1] - pd.DateOffset(years=5)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v.index, y=v.values, name="VIX", line=dict(color="#2a78d6", width=1.6),
                             hovertemplate="%{y:.1f}"))
    for lvl, lab, col in ((L.VIX_WATCH, "留意 30", "#B9770E"), (L.VIX_PANIC, "全面恐慌 40", "#1E7A46")):
        fig.add_hline(y=lvl, line=dict(color=col, dash="dash", width=1), annotation_text=lab,
                      annotation_position="top left")
    edges = L.rising_edges(v >= L.VIX_PANIC)
    if len(edges):
        fig.add_trace(go.Scatter(x=edges, y=v.reindex(edges).values, mode="markers", name="首次 ≥ 40",
                                 marker=dict(size=9, color="#1E7A46", line=dict(color="white", width=1.5)),
                                 hovertemplate="%{x|%Y-%m-%d} 首次 ≥ 40"))
    fig.update_layout(title="VIX（近 5 年）", height=300, showlegend=False)
    apply_hover_style(fig)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 你的加碼規則
# ---------------------------------------------------------------------------
st.markdown("### 你的現金桶加碼規則")
if LOCAL_SAVE:
    st.caption("這些數字由你決定；頁面只負責在「全面恐慌」時提醒並算出金額。按「儲存」會存在這台電腦的 "
               "cache/entry_params.json（不會進 git）。")
else:
    st.caption("這些數字由你決定；頁面只負責在「全面恐慌」時提醒並算出金額。"
               "線上版不會保存，關掉頁面就清空——請不要在線上版輸入真實金額。")


def _load_params() -> dict:
    if not LOCAL_SAVE:
        return dict(DEFAULT_PARAMS)
    try:
        return {**DEFAULT_PARAMS, **json.loads(PARAMS_PATH.read_text(encoding="utf-8"))}
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_PARAMS)


if "entry_params" not in st.session_state:
    st.session_state.entry_params = _load_params()
p = st.session_state.entry_params
a1, a2, a3, a4, a5 = st.columns(5)
p["total_assets"] = a1.number_input("可投資資產總額", min_value=0, value=int(p["total_assets"]), step=10_000)
p["base_equity_pct"] = a2.number_input("平常的股票比例（%）", min_value=0, max_value=100,
                                       value=int(p["base_equity_pct"]), step=5)
p["tilt_pp"] = a3.number_input("全面恐慌時加碼（百分點）", min_value=0, max_value=50, value=int(p["tilt_pp"]), step=5)
p["cash_bucket"] = a4.number_input("可以動用的現金桶", min_value=0, value=int(p["cash_bucket"]), step=10_000,
                                   help="只放你真的願意在大跌時投入的錢；緊急預備金不算。")
p["hold_months"] = a5.number_input("加碼後至少持有（月）", min_value=1, max_value=36, value=int(p["hold_months"]))
if LOCAL_SAVE and st.button("儲存我的參數"):
    try:
        PARAMS_PATH.parent.mkdir(exist_ok=True)
        PARAMS_PATH.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
        st.success("已儲存")
    except Exception as e:  # noqa: BLE001
        st.error(f"儲存失敗：{e}")

plan = L.overweight_plan(state, p["total_assets"], p["base_equity_pct"], p["tilt_pp"], p["cash_bucket"],
                         p["hold_months"])
if plan["active"]:
    st.success(f"全面恐慌：依你的規則，從現金桶買進 **{plan['amount']:,.0f}**"
               f"（股票比例拉到約 {plan['target_equity_pct']}%），之後至少持有 {plan['hold_months']} 個月，"
               f"不要因為反彈就賣。" + ("（金額已被現金桶上限限制）" if plan["limited_by_cash"] else ""))
else:
    st.info(f"目前不是全面恐慌：現金桶不動。觸發時依你的規則會買進約 **{plan['amount_if_triggered']:,.0f}**"
            f"（股票比例拉到約 {plan['target_equity_pct']}%）。"
            + ("（金額已被現金桶上限限制）" if plan["limited_by_cash"] else ""))
st.caption("研究裡的版本是「平常 50%、觸發後加 20 個百分點、6 個月後調回」：每年約多 0.5～0.9 個百分點，"
           "但拿掉 2009、2020 兩年後就不顯著。把它當成「恐慌時至少不賣、有閒錢加一點」的行為規則，不要期待多賺很多。")

# ---------------------------------------------------------------------------
# 歷史證據
# ---------------------------------------------------------------------------
st.markdown("### 歷史證據")
if not ev:
    st.warning("找不到研究證據檔 utils/signal_evidence.json。請執行：`python research/export_evidence.py`")
else:
    st.caption(f"研究資料到 {ev.get('data_end')}；測了 {ev.get('n_variants_tested')} 個變體，"
               f"通過 {ev.get('n_passed')} 個（進場訊號最好的 q 值 {ev.get('best_entry_q')}，門檻 0.10）。")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("VIX 首次 ≥ 40 次數", f"{int(v40.get('n_events') or 0)} 次",
              help=f"分屬 {int(v40.get('n_clusters') or 0)} 群（相隔 3 個月以上算不同群），大致是 9 次不同的危機。")
    e2.metric("之後 3 個月 NDX 平均", _pct(v40.get("mean_63")), f"平常 {_pct(v40.get('base_63'))}",
              delta_color="off")
    e3.metric("3 個月後是正報酬", _pct(v40.get("hit_63"), 0, False), f"平常 {_pct(v40.get('hit_base_63'), 0, False)}",
              delta_color="off")
    e4.metric("多重檢定 q 值", f"{v40.get('q'):.2f}" if v40.get("q") is not None else "—",
              help="≤ 0.10 才算通過。單獨看 p = %.3f。" % (v40.get("p") or float("nan")))

    rows = v40.get("events", [])
    if rows:
        df = pd.DataFrame(rows)
        tbl = pd.DataFrame({
            "觸發日": df["date"],
            "NDX 1 個月": df["ndx_21"].map(_pct),
            "NDX 3 個月": df["ndx_63"].map(_pct),
            "NDX 6 個月": df["ndx_126"].map(_pct),
            "S&P 500 3 個月": df["spx_63"].map(_pct),
            "3 個月內最深": df["ndx_min_63"].map(_pct),
        })
        st.markdown("**VIX 首次 ≥ 40 之後（隔天收盤買進）**")
        st.dataframe(tbl, width="stretch", hide_index=True)
    with st.expander("確認指標：9 檔產業 ETF 中 ≥ 4 檔同時創一年新低（1999 起）", expanded=False):
        srows = (ev.get("sector_lows4") or {}).get("events", [])
        if srows:
            sd = pd.DataFrame(srows)
            st.dataframe(pd.DataFrame({"觸發日": sd["date"], "NDX 3 個月": sd["ndx_63"].map(_pct),
                                       "NDX 6 個月": sd["ndx_126"].map(_pct), "S&P 500 3 個月": sd["spx_63"].map(_pct)}),
                         width="stretch", hide_index=True)
        st.caption("只算每次危機第一次觸發時效果就不顯著，1928–1945 甚至反向，所以只當確認，不單獨觸發。")
    with st.expander("為什麼不留錢等訊號（現金部署模擬）", expanded=False):
        rr = ev.get("reserve_ndx", [])
        if rr:
            names = {"vix_level[rule=vix>=40]": "VIX ≥ 40", "vix_level[rule=vix>=30]": "VIX ≥ 30",
                     "dd_tranche[level=0.1]": "距高點跌 10%", "dd_tranche[level=0.2]": "距高點跌 20%",
                     "breadth_thrust[low=0.2,high=0.5]": "產業廣度回升", "cor_high[pct=95]": "隱含相關性高檔",
                     "composite_fear[th=75]": "舊的恐慌綜合分數 ≥ 75", "aaii_low[rule=spread_pct5]": "AAII 極度悲觀"}
            rd = pd.DataFrame(rr)
            st.dataframe(pd.DataFrame({"等的訊號": rd["variant"].map(lambda x: names.get(x, x)),
                                       "期末財富 vs 直接買": rd["ratio_minus_1"].map(_pct),
                                       "隨機時間點（中位數）": rd["random_median_minus_1"].map(_pct),
                                       "平均等待天數": rd["avg_wait_days"]}),
                         width="stretch", hide_index=True)
        st.caption("每月存 1 元：一半馬上買、一半放國庫券等訊號（最多等 12 個月）。NDX，1990–2026。")

    st.markdown("### 已驗證沒有用（或證據太弱）的指標")
    st.caption("這些指標可以留在其他頁描述市場，但不要拿來決定買賣。")
    rej = ev.get("rejected", [])
    if rej:
        st.dataframe(pd.DataFrame({"指標": [r["name"] for r in rej],
                                   "觸發後 3 個月超額（NDX）": [_pct(r["excess_63"]) for r in rej],
                                   "q 值": [r["q"] for r in rej],
                                   "原因": [r["why"] for r in rej]}),
                     width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 出場
# ---------------------------------------------------------------------------
st.markdown("### 出場")
st.markdown("**沒有任何出場訊號通過驗證。** 不要因為情緒、籌碼、總經指標賣出；"
            "研究裡「VIX 飆高就出場」是最差的規則之一，等於賣在之後報酬最好的時候。")
t10 = ev.get("trend_10m_ndx") or {}
with st.expander("如果你擔心回撤：10 個月均線「保險」（有代價，不是 alpha）", expanded=False):
    if tr.get("above") is not None:
        st.markdown(f"{symbol} {tr['month']} 月底收盤"
                    f"**{'在 10 個月均線之上' if tr['above'] else '跌破 10 個月均線'}**"
                    f"（{_pct(tr['gap'])}）；目前價格相對均線 {_pct(tr['current_vs_sma'])}。")
    if t10:
        st.dataframe(pd.DataFrame({
            "做法": ["10 個月均線（跌破出場）", "一直持有", "同樣平均持股的固定比例"],
            "年化報酬": [_pct(t10.get("cagr")), _pct(t10.get("bh_cagr")), _pct(t10.get("matched_cagr"))],
            "最大回撤": [_pct(t10.get("maxdd"), 0), _pct(t10.get("bh_maxdd"), 0), _pct(t10.get("matched_maxdd"), 0)],
        }), width="stretch", hide_index=True)
        oos = t10.get("oos_alpha") or 0
        st.caption(f"NDX {t10.get('start', '')[:4]}–2026，含成本。2013 年後它每年比固定比例"
                   f"{'少' if oos < 0 else '多'} {_pct(abs(oos), 1, False)}。"
                   f"比較簡單的替代方案：事先把股票比例定在睡得著的位置。")

for w in res.get("warnings", []):
    st.warning(w)

with st.expander("📋 純文字簡報（可複製）", expanded=False):
    st.code(L.format_brief(res, symbol), language="text")

st.markdown("---")
st.caption(f"計算日 {res.get('as_of')}。研究細節見 research/REPORT.md。"
           "這是歷史統計與提醒，不是投資建議；實際買賣由你判斷。")
