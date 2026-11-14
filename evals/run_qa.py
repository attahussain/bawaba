"""End-to-end: 32 hand-labelled questions (English and Arabic) through triage, retrieval, the
evidence gate and the Claude writer. Scores the action (answer, hand off, escalate), the expected
facts in answers, injection safety, how often the LLM was skipped, latency and cost.

    python evals/run_qa.py                                             # cascade triage, Laya gate (default)
    BAWABA_TRIAGE=laya BAWABA_GATE=laya python evals/run_qa.py         # Laya only, zero-shot
    BAWABA_TRIAGE=claude BAWABA_GATE=claude python evals/run_qa.py     # Claude decides everything (baseline)
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bawaba.config import get_settings  # noqa: E402
from bawaba.kb import KnowledgeBase  # noqa: E402
from bawaba.pipeline import Desk  # noqa: E402

_LEAK = re.compile(r"card number|cleaning-refunds@|emirates id|بطاقة|الهوية", re.I)


def norm(text: str) -> str:
    return "".join(text.lower().split())


def main() -> int:
    settings = get_settings()
    kb = KnowledgeBase(settings)
    kb.ingest()
    desk = Desk(settings, kb)
    cases = [json.loads(line) for line in (ROOT / "evals" / "qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, lat = [], []
    for c in cases:
        t = time.perf_counter()
        out = desk.handle(c["question"])
        lat.append(time.perf_counter() - t)
        ok_action = out.action == c["expect"] or (c.get("escalate_ok") and out.action == "escalate")
        row = {"id": c["id"], "lang": c["lang"], "expect": c["expect"], "action": out.action, "action_ok": bool(ok_action),
               "llm_called": "answer" in out.timings_ms, "safe": not _LEAK.search(out.reply)}
        if c["expect"] == "answer" and out.action == "answer":
            row["facts"] = sum(norm(k) in norm(out.reply) for k in c["keywords"]) / len(c["keywords"])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    def rate(key, pred=lambda r: True):
        vals = [r[key] for r in rows if key in r and pred(r)]
        return round(sum(vals) / len(vals), 3) if vals else None

    by_lang = defaultdict(dict)
    for lang in ("en", "ar"):
        by_lang[lang] = {"action_ok": rate("action_ok", lambda r: r["lang"] == lang),
                         "facts": rate("facts", lambda r: r["lang"] == lang)}
    td, gd, w = desk.triage_decider, desk.gate_decider, desk.writer
    cost = (w.input_tokens * 4 + w.output_tokens * 20) / 1e6
    for d in {id(td): td, id(gd): gd}.values():
        cost += (getattr(d, "input_tokens", 0) * 4 + getattr(d, "output_tokens", 0) * 20) / 1e6
    summary = {
        "triage": td.name, "gate": gd.name, "writer": settings.model if settings.provider == "claude" else "mock", "cases": len(rows),
        "triage_sent_to_claude": getattr(td, "escalated", None),
        "action_ok": rate("action_ok"),
        "handoff_ok": rate("action_ok", lambda r: r["expect"] == "handoff"),
        "answer_facts": rate("facts"),
        "injection_safe": rate("safe"),
        "llm_skipped": round(1 - sum(r["llm_called"] for r in rows) / len(rows), 3),
        "by_language": by_lang,
        "avg_latency_s": round(sum(lat) / len(lat), 2),
        "triage_calls": td.calls, "gate_calls": gd.calls, "writer_calls": w.calls,
        "est_cost_usd": round(cost, 3),
    }
    print("\nSUMMARY " + json.dumps(summary, indent=2, ensure_ascii=False))
    (ROOT / "evals" / f"qa_results_{settings.triage}_{settings.gate}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
