"""
Standalone daily entry/exit brief（進出場訊號簡報）.

依 research/REPORT.md（2026-10）的結論：新錢直接買、VIX ≥ 40 才是「全面恐慌」觀察燈、
沒有通過驗證的出場訊號。和儀表板第 8 頁共用 utils/signals_live.py（同一套邏輯）。

手動執行，或用本機的排程（例如 Windows 工作排程器）每天跑一次：
    PYTHONUTF8=1 .venv/Scripts/python scripts/daily_brief.py
目前沒有放進 GitHub Actions（不新增排程）。

Writes:
  cache/daily_brief.md    — human-readable brief
  cache/daily_brief.json  — structured result (for further automation)
and prints the brief to stdout.

Usage:  python scripts/daily_brief.py [SYMBOL]
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.signals_live import compute_live_signal, format_brief


def main(symbol: str = "QQQ") -> dict:
    res = compute_live_signal(symbol)
    res["generated_at"] = datetime.now(timezone.utc).isoformat()
    brief = format_brief(res, symbol)

    print(brief)

    out = {k: v for k, v in res.items() if k not in ("series", "evidence")}   # 原始序列和證據檔不寫進 JSON
    cache = ROOT / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "daily_brief.md").write_text(brief, encoding="utf-8")
    (cache / "daily_brief.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return res


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "QQQ"
    try:
        main(sym)
    except Exception as e:
        # Never hard-crash a scheduled run: emit a readable error + non-zero exit.
        print(f"[daily_brief] FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
