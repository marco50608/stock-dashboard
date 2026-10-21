"""主研究：所有登記的訊號 × {NDX, SPX} × {全期, 樣本內, 樣本外, 發表前, 發表後}，加多重檢定與 walk-forward。

    PYTHONUTF8=1 .venv/Scripts/python research/run_study.py

輸出（research/results/）：
    entry_results.csv     進場事件研究（每個變體 × 標的 × 期間）
    timing_results.csv    擇時回測
    walkforward_entry.csv、walkforward_timing.csv
                          家族內逐年重新挑參數的樣本外結果（進場、擇時各一份）
    multiple_testing.json BH-FDR、Reality Check / SPA、Deflated Sharpe
    verdicts.csv          依事先寫好的標準判定：通過 / 證據不足 / 淘汰
"""
from __future__ import annotations

import json
import zlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib import backtest as BT  # noqa: E402
from lib import stats as S  # noqa: E402
from lib.inputs import Bundle  # noqa: E402
from lib.signals import REGISTRY, all_variants  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 事先寫好的設定與判定標準（看結果之前就固定）
# ---------------------------------------------------------------------------
ASSETS = ["NDX", "SPX"]
IS_END = "2012-12-31"
OOS_START = "2013-01-01"
HORIZONS = (21, 63, 126)
PRIMARY_H = 63
COOLDOWN = 21
N_NULL = 2000
COST_BPS = 5.0
COST_SENS = (10.0, 20.0)
FDR_Q = 0.10
MIN_EVENTS_TOTAL = 10       # 全期觸發次數少於這個 → 「證據不足」
MIN_EVENTS_OOS = 5
TIMING_T_MIN = 2.0          # 擇時 alpha 的 Newey-West t 值門檻
DSR_MIN = 0.95


def windows(spec):
    w = {"full": (None, None), "IS": (None, IS_END), "OOS": (OOS_START, None)}
    if spec.pub_year:
        w["pre_pub"] = (None, f"{spec.pub_year - 1}-12-31")
        w["post_pub"] = (f"{spec.pub_year + 1}-01-01", None)
    return w


def run_entry(B, px_cache):
    rows, series = [], {}
    for spec, params, name in all_variants("entry"):
        for asset in ASSETS:
            t0 = time.time()
            trig = spec.func(B, asset, **params)
            series[(name, asset)] = trig
            px = px_cache[asset]
            for wname, (a, b) in windows(spec).items():
                es = BT.event_study(trig, px, HORIZONS, COOLDOWN, a, b, N_NULL, seed=zlib.crc32(f"{name}|{asset}|{wname}".encode()))
                row = {"variant": name, "family": spec.family, "asset": asset, "window": wname,
                       "n_events": es.get("n_events", 0)}
                for h in HORIZONS:
                    if h in es:
                        for k in ("n", "mean", "base", "excess", "median", "hit", "hit_base", "worst", "p",
                                  "p_shift", "n_clusters", "fwd_min_mean", "fwd_min_base"):
                            row[f"h{h}_{k}"] = es[h][k]
                if "events" in es and wname == "full":
                    row["first_event"] = es["events"][0].date().isoformat() if len(es["events"]) else None
                rows.append(row)
            print(f"  entry {name} {asset} {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows), series


def run_timing_all(B, ret_cache, rf):
    rows, alpha, strat = [], {}, {}
    for spec, params, name in all_variants("timing"):
        for asset in ASSETS:
            pos = spec.func(B, asset, **params)
            ret = ret_cache[asset]
            for wname, (a, b) in windows(spec).items():
                r = BT.run_timing(ret, pos, rf, COST_BPS, start=a, end=b)
                if r["df"].empty or not r.get("strat"):
                    continue
                row = {"variant": name, "family": spec.family, "asset": asset, "window": wname,
                       "alpha_ann": r["alpha_ann"], "alpha_t": r["alpha_t"], "beta": r["beta"]}
                for k, v in r["strat"].items():
                    row[f"s_{k}"] = v
                for k in ("CAGR", "vol", "Sharpe", "MaxDD", "Calmar"):
                    row[f"bh_{k}"] = r["bh"].get(k)
                    row[f"m_{k}"] = r["matched"].get(k)
                if wname == "full":
                    alpha[(name, asset)] = r["df"]["alpha"]
                    strat[(name, asset)] = r["df"]["strat"]
                    for c in COST_SENS:
                        rc = BT.run_timing(ret, pos, rf, c)
                        row[f"alpha_ann_{int(c)}bps"] = rc["alpha_ann"]
                        row[f"CAGR_{int(c)}bps"] = rc["strat"]["CAGR"]
                rows.append(row)
            print(f"  timing {name} {asset}", flush=True)
    return pd.DataFrame(rows), alpha, strat


# ---------------------------------------------------------------------------
# Walk-forward：家族內每年用「到去年為止」的結果挑參數，套用在今年
# ---------------------------------------------------------------------------
def walkforward_entry(series, px_cache, first_year=2000, min_ev=5):
    rows = []
    fams = {}
    for spec, params, name in all_variants("entry"):
        fams.setdefault(spec.family, []).append(name)
    for fam, names in fams.items():
        for asset in ASSETS:
            px = px_cache[asset]
            fr = BT.forward_returns(px, PRIMARY_H)
            base = fr
            ev = {n: BT.episode_starts(series[(n, asset)].dropna().astype(bool), COOLDOWN) for n in names}
            last_year = px.index[-1].year
            oos_ex = []
            chosen_log = []
            for Y in range(first_year, last_year + 1):
                y0 = pd.Timestamp(f"{Y}-01-01")
                cutoff_idx = px.index.searchsorted(y0) - PRIMARY_H - 2   # 未來報酬在 Y 年以前已經實現的事件
                if cutoff_idx <= 0:
                    continue
                cut = px.index[cutoff_idx]
                best, best_score = None, -np.inf
                for n in names:
                    e = ev[n][ev[n] <= cut]
                    if len(e) < min_ev:
                        continue
                    valid = series[(n, asset)].dropna().index
                    valid = valid[valid <= cut]
                    if len(valid) == 0:
                        continue
                    sc = fr.reindex(e).mean() - fr.reindex(valid).mean()
                    if sc > best_score:
                        best, best_score = n, sc
                if best is None:
                    continue
                if len(names) == 1:
                    best = names[0]
                e_y = ev[best][(ev[best] >= y0) & (ev[best] < pd.Timestamp(f"{Y + 1}-01-01"))]
                valid_y = series[(best, asset)].dropna().index
                valid_y = valid_y[(valid_y >= y0) & (valid_y < pd.Timestamp(f"{Y + 1}-01-01"))]
                b_y = fr.reindex(valid_y).dropna()
                for d in e_y:
                    v = fr.get(d, np.nan)
                    if pd.notna(v) and len(b_y):
                        oos_ex.append((d, best, v, b_y.mean()))
                chosen_log.append(f"{Y}:{best}")
            if oos_ex:
                df = pd.DataFrame(oos_ex, columns=["date", "variant", "fwd", "base"])
                rows.append({"family": fam, "asset": asset, "n_oos_events": len(df),
                             "oos_mean": df["fwd"].mean(), "oos_base": df["base"].mean(),
                             "oos_excess": (df["fwd"] - df["base"]).mean(),
                             "oos_hit": (df["fwd"] > 0).mean(),
                             "chosen": ";".join(chosen_log[-6:])})
            else:
                rows.append({"family": fam, "asset": asset, "n_oos_events": 0})
    return pd.DataFrame(rows)


def walkforward_timing(B, ret_cache, rf, first_year=2000):
    rows = []
    fams = {}
    for spec, params, name in all_variants("timing"):
        fams.setdefault(spec.family, []).append((spec, params, name))
    for fam, items in fams.items():
        for asset in ASSETS:
            ret = ret_cache[asset]
            res = {n: BT.run_timing(ret, sp.func(B, asset, **pa), rf, COST_BPS)["df"] for sp, pa, n in items}
            pieces = []
            for Y in range(first_year, ret.index[-1].year + 1):
                y0 = pd.Timestamp(f"{Y}-01-01")
                best, best_sc = None, -np.inf
                for n, df in res.items():
                    h_ = df.loc[: y0 - pd.Timedelta(days=1)]
                    if len(h_) < 252 * 5:
                        continue
                    b_ = h_["pos"].mean()          # 只用過去的平均部位當比較基準（不偷看未來）
                    hist = h_["strat"] - (b_ * h_["ret"] + (1 - b_) * h_["rf"])
                    sc = hist.mean() / hist.std() if hist.std() > 0 else -np.inf
                    if sc > best_sc:
                        best, best_sc = n, sc
                if best is None:
                    continue
                pieces.append(res[best].loc[f"{Y}-01-01":f"{Y}-12-31", ["strat", "ret", "rf", "pos", "alpha"]])
            if not pieces:
                continue
            wf = pd.concat(pieces)
            bw = wf["pos"].mean()
            wf["alpha"] = wf["strat"] - (bw * wf["ret"] + (1 - bw) * wf["rf"])
            st = BT.perf_stats(wf["strat"], wf["rf"], wf["pos"])
            bh = BT.perf_stats(wf["ret"], wf["rf"])
            rows.append({"family": fam, "asset": asset, "start": st.get("start"), "alpha_ann": wf["alpha"].mean() * 252,
                         "alpha_t": BT._nw_t(wf["alpha"]), "CAGR": st.get("CAGR"), "MaxDD": st.get("MaxDD"),
                         "Sharpe": st.get("Sharpe"), "exposure": st.get("exposure"),
                         "bh_CAGR": bh.get("CAGR"), "bh_MaxDD": bh.get("MaxDD"), "bh_Sharpe": bh.get("Sharpe")})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    B = Bundle()
    rf = B.rf()
    px_cache = {a: B.px(a) for a in ASSETS}
    ret_cache = {a: B.tr(a) for a in ASSETS}

    print("== 進場事件研究", flush=True)
    entry, series = run_entry(B, px_cache)
    entry.to_csv(OUT / "entry_results.csv", index=False, encoding="utf-8-sig")
    pd.to_pickle(series, OUT / "entry_series.pkl")

    print("== 擇時回測", flush=True)
    timing, alpha, strat = run_timing_all(B, ret_cache, rf)
    timing.to_csv(OUT / "timing_results.csv", index=False, encoding="utf-8-sig")
    pd.to_pickle({"alpha": alpha, "strat": strat}, OUT / "timing_series.pkl")

    print("== 多重檢定", flush=True)
    mt = {}
    for asset in ASSETS:
        f = entry[(entry["asset"] == asset) & (entry["window"] == "full")].set_index("variant")
        p = f[f"h{PRIMARY_H}_p"].dropna()
        bh = S.benjamini_hochberg(p, FDR_Q)
        mt[f"entry_BH_{asset}"] = bh.reset_index().rename(columns={"index": "variant"}).to_dict(orient="records")
        # 擇時：Reality Check / SPA（沒有資料的期間 alpha 當 0 = 等於基準）
        A = pd.DataFrame({n: s for (n, a), s in alpha.items() if a == asset}).fillna(0.0)
        rc = S.reality_check(A, B=1000, avg_block=20, seed=7)
        srs = np.array([S.sharpe_per_period(A[c].values[A[c].values != 0]) for c in A.columns])
        dsr = {c: S.deflated_sharpe(alpha[(c, asset)].dropna().values, srs) for c in A.columns}
        mt[f"timing_RC_{asset}"] = {"p_RC": rc["p_RC"], "p_SPA": rc["p_SPA"], "K": rc["K"], "T": rc["T"],
                                    "top_t": rc["t_by_strategy"].head(10).to_dict()}
        mt[f"timing_DSR_{asset}"] = dsr
    (OUT / "multiple_testing.json").write_text(json.dumps(mt, indent=1, default=float, ensure_ascii=False),
                                              encoding="utf-8")

    print("== Walk-forward", flush=True)
    wfe = walkforward_entry(series, px_cache)
    wft = walkforward_timing(B, ret_cache, rf)
    wfe.to_csv(OUT / "walkforward_entry.csv", index=False, encoding="utf-8-sig")
    wft.to_csv(OUT / "walkforward_timing.csv", index=False, encoding="utf-8-sig")

    print("== 判定", flush=True)
    verdicts = judge(entry, timing, mt, wfe, wft)
    verdicts.to_csv(OUT / "verdicts.csv", index=False, encoding="utf-8-sig")
    print(verdicts[["variant", "kind", "verdict", "reason"]].to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def judge(entry, timing, mt, wfe, wft):
    """依事先寫好的標準判定每個變體。"""
    out = []
    H = PRIMARY_H
    bh = {r["variant"]: r for r in mt["entry_BH_NDX"]}
    for spec, params, name in all_variants("entry"):
        g = entry[entry["variant"] == name].set_index(["asset", "window"])

        def get(asset, win, col):
            try:
                return g.loc[(asset, win), col]
            except KeyError:
                return np.nan
        n_full = get("NDX", "full", f"h{H}_n")
        ex_full = get("NDX", "full", f"h{H}_excess")
        q = bh.get(name, {}).get("q_value", np.nan)
        ex_oos = get("NDX", "OOS", f"h{H}_excess")
        n_oos = get("NDX", "OOS", f"h{H}_n")
        ex_spx = get("SPX", "full", f"h{H}_excess")
        wf = wfe[(wfe["family"] == spec.family) & (wfe["asset"] == "NDX")]
        wf_ex = wf["oos_excess"].iloc[0] if len(wf) and "oos_excess" in wf else np.nan
        reasons = []
        if not (n_full >= MIN_EVENTS_TOTAL):
            verdict = "證據不足"
            reasons.append(f"全期只有 {0 if pd.isna(n_full) else int(n_full)} 次觸發")
        else:
            ok = True
            if not (ex_full > 0):
                ok = False
                reasons.append("全期觸發後 3 個月報酬沒有比平常好")
            elif not (q <= FDR_Q):
                ok = False
                reasons.append(f"多重檢定後不顯著（q={q:.2f}）")
            if not (ex_oos > 0 and n_oos >= MIN_EVENTS_OOS):
                ok = False
                reasons.append("樣本外（2013 起）沒有延續" if n_oos >= MIN_EVENTS_OOS else "樣本外觸發太少")
            if not (ex_spx > 0):
                ok = False
                reasons.append("換成 S&P 500 不成立")
            if not (wf_ex > 0):
                ok = False
                reasons.append("walk-forward 挑參數後樣本外不成立")
            verdict = "通過" if ok else "淘汰"
        out.append({"variant": name, "family": spec.family, "kind": "entry", "title": spec.title,
                    "category": spec.category, "niche": spec.niche, "verdict": verdict, "reason": "；".join(reasons),
                    "n_full": n_full, "excess_full": ex_full, "q": q, "excess_oos": ex_oos, "n_oos": n_oos,
                    "excess_spx": ex_spx, "wf_excess": wf_ex})
    dsr = mt["timing_DSR_NDX"]
    for spec, params, name in all_variants("timing"):
        g = timing[timing["variant"] == name].set_index(["asset", "window"])

        def get(asset, win, col):
            try:
                return g.loc[(asset, win), col]
            except KeyError:
                return np.nan
        t_full = get("NDX", "full", "alpha_t")
        a_full = get("NDX", "full", "alpha_ann")
        a_oos = get("NDX", "OOS", "alpha_ann")
        a_spx = get("SPX", "full", "alpha_ann")
        a_20 = get("NDX", "full", "alpha_ann_20bps")
        d = dsr.get(name, {}).get("DSR", np.nan)
        wf = wft[(wft["family"] == spec.family) & (wft["asset"] == "NDX")]
        wf_a = wf["alpha_ann"].iloc[0] if len(wf) else np.nan
        reasons = []
        ok = True
        if not (a_full > 0):
            ok = False
            reasons.append("擇時 alpha ≤ 0（不如同曝險的固定比例）")
        elif not (t_full >= TIMING_T_MIN):
            ok = False
            reasons.append(f"alpha 不顯著（t={t_full:.1f}）")
        if not (d >= DSR_MIN):
            ok = False
            reasons.append(f"Deflated Sharpe {d:.2f} < {DSR_MIN}")
        if not (a_oos > 0):
            ok = False
            reasons.append("樣本外 alpha ≤ 0")
        if not (a_spx > 0):
            ok = False
            reasons.append("S&P 500 不成立")
        if not (a_20 > 0):
            ok = False
            reasons.append("成本 20 bps 時 alpha ≤ 0")
        if not (wf_a > 0):
            ok = False
            reasons.append("walk-forward 不成立")
        verdict = "通過" if ok else "淘汰"
        if spec.note and "無法驗證" in spec.note:
            verdict = "證據不足" if not ok else verdict
            reasons.append(spec.note)
        out.append({"variant": name, "family": spec.family, "kind": "timing", "title": spec.title,
                    "category": spec.category, "niche": spec.niche, "verdict": verdict, "reason": "；".join(reasons),
                    "alpha_full": a_full, "alpha_t": t_full, "DSR": d, "alpha_oos": a_oos, "alpha_spx": a_spx,
                    "alpha_20bps": a_20, "wf_alpha": wf_a})
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
