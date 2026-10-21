"""一鍵重跑整個研究（建議每年一次），最後更新儀表板用的證據檔 utils/signal_evidence.json。

    PYTHONUTF8=1 .venv/Scripts/python research/run_all.py                # 全部（約 40–60 分鐘，大多花在下載與現金部署模擬）
    PYTHONUTF8=1 .venv/Scripts/python research/run_all.py --skip-fetch   # 不重新下載資料
    PYTHONUTF8=1 .venv/Scripts/python research/run_all.py --skip-reserve # 跳過最慢的現金部署模擬

任何一步失敗就停下來，不會用一半的結果覆蓋證據檔。
跑完請看 research/results/verdicts.csv 和 utils/signal_evidence.json 的 best_entry_q：
觀察名單的訊號要 q ≤ 0.10 才算通過；如果結論變了，REPORT.md 也要跟著更新。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESERVE_VARIANTS = ["vix_level[rule=vix>=40]", "vix_level[rule=vix>=30]", "dd_tranche[level=0.1]",
                    "dd_tranche[level=0.2]", "breadth_thrust[low=0.2,high=0.5]", "cor_high[pct=95]",
                    "composite_fear[th=75]", "aaii_low[rule=spread_pct5]"]


def main(argv: list[str]) -> int:
    steps = []
    if "--skip-fetch" not in argv:
        steps.append(("下載資料", ["fetch_data.py"]))
    steps += [
        ("回測引擎測試", ["tests/run_tests.py"]),
        ("訊號截斷（前視）測試", ["tests/run_signal_checks.py"]),
        ("主研究", ["run_study.py"]),
        ("穩健性檢查", ["run_robustness.py"]),
        ("第二輪穩健性與加碼用法", ["run_robustness2.py"]),
    ]
    if "--skip-reserve" not in argv:
        steps.append(("現金部署模擬", ["run_reserve.py", *RESERVE_VARIANTS]))
    steps += [
        ("選擇權定價測試", ["tests/test_options.py"]),
        ("賣 put 補充研究", ["run_csp.py"]),
        ("圖表", ["make_figures.py"]),
        ("報告表格", ["build_tables.py"]),
        ("匯出儀表板證據檔", ["export_evidence.py"]),
    ]
    t_all = time.time()
    for name, args in steps:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / args[0]), *args[1:]], cwd=ROOT.parent)
        if r.returncode != 0:
            print(f"\n失敗：{name}（結束碼 {r.returncode}）。後面的步驟沒有執行，證據檔沒有更新。")
            return r.returncode
        print(f"--- {name} 完成（{(time.time() - t0) / 60:.1f} 分鐘）", flush=True)
    print(f"\n全部完成（{(time.time() - t_all) / 60:.1f} 分鐘）。請檢查 results/verdicts.csv 與 REPORT.md 是否需要更新。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
