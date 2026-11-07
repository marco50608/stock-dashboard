# research/：進出場訊號研究

- **報告**：[REPORT.md](REPORT.md)（結論、通過與淘汰的訊號、模組設計提案、重現步驟）
- **原始資料**：`raw/`（不進 repo，`fetch_data.py` 會重新下載）
- **結果**：`results/`（CSV、圖）

除了專案的 `requirements.txt`，研究另外需要 matplotlib（`pip install matplotlib`）。

一鍵重跑（建議每年一次；依序跑完下載、測試、主研究、穩健性、現金部署模擬、賣 put、圖表、表格，最後更新儀表板用的 `utils/signal_evidence.json`）：

```bash
PYTHONUTF8=1 .venv/Scripts/python research/run_all.py
```

逐步重跑（在 stock 專案根目錄）：

```bash
PYTHONUTF8=1 .venv/Scripts/python research/fetch_data.py
PYTHONUTF8=1 .venv/Scripts/python research/tests/run_tests.py
PYTHONUTF8=1 .venv/Scripts/python research/tests/run_signal_checks.py
PYTHONUTF8=1 .venv/Scripts/python research/run_study.py
```

完整步驟見 REPORT.md 第 8 節。2026-10-25 的第三輪追加測試（SLOOS、短期利率、EBP、EPU）事先登記在 [PREREG_batch3_2026-10-25.md](PREREG_batch3_2026-10-25.md)，結果在 REPORT.md 第 9 節；敏感度檢查是 `run_batch3_sensitivity.py`。新增訊號請加在 `lib/signals.py`（用 `@register`，參數在跑之前寫死），它會自動算進多重檢定。
