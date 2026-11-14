"""Cascade: Laya (fitted heads) decides when it is confident; Claude decides the rest.

Laya and Claude each answer every test message once; each threshold is then scored from those
cached answers (a message goes to Claude when Laya's lower confidence of intent and urgency is
below the threshold). Run evals/triage_eval.py first: it saves the fitted heads.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bawaba.deciders import ClaudeDecider  # noqa: E402
from bawaba.triage import QUESTIONS  # noqa: E402
from evals.triage_eval import HEADS, split  # noqa: E402

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]  # 0.0 = Laya only, 1.01 = Claude only


def main() -> int:
    import laya

    rows = [json.loads(line) for line in (ROOT / "data" / "tickets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _, test = split(rows)
    reader = laya.Agent(vegaml=True)
    reader.load_calibration(str(HEADS))
    laya_out, laya_ms = [], []
    for r in test:
        t = time.perf_counter()
        a = reader.predict(r["text"], QUESTIONS)["answers"]
        laya_ms.append((time.perf_counter() - t) * 1000)
        conf = min(max(a[q]["probabilities"].values()) for q in ("intent", "urgency"))
        laya_out.append({"intent": a["intent"]["choice"], "urgency": a["urgency"]["choice"], "conf": conf})

    judge = ClaudeDecider()
    claude_out: dict[int, dict] = {}
    for i, r in enumerate(test):  # every row: the Claude-only point of the sweep needs them all
        a = judge.predict(r["text"], QUESTIONS)
        claude_out[i] = {"intent": a["intent"]["choice"], "urgency": a["urgency"]["choice"]}
    claude_ms = judge.seconds * 1000 / max(judge.calls, 1)
    cost_per_call = (judge.input_tokens * 4 + judge.output_tokens * 20) / 1e6 / max(judge.calls, 1)

    results = []
    for th in THRESHOLDS:
        acc = {"intent": 0, "urgency": 0}
        tp = fn = 0
        per_lang = {"en": [0, 0], "ar": [0, 0]}  # [urgency correct, n]
        sent = 0
        for i, r in enumerate(test):
            use_claude = laya_out[i]["conf"] < th
            sent += use_claude
            p = claude_out[i] if use_claude else laya_out[i]
            acc["intent"] += p["intent"] == r["intent"]
            ok = p["urgency"] == r["urgency"]
            acc["urgency"] += ok
            per_lang[r["lang"]][0] += ok
            per_lang[r["lang"]][1] += 1
            if r["urgency"] == "emergency":
                tp += p["urgency"] == "emergency"
                fn += p["urgency"] != "emergency"
        n = len(test)
        share = sent / n
        res = {
            "threshold": th,
            "to_claude": round(share, 3),
            "intent_acc": round(acc["intent"] / n, 3),
            "urgency_acc": round(acc["urgency"] / n, 3),
            "urgency_acc_en": round(per_lang["en"][0] / per_lang["en"][1], 3),
            "urgency_acc_ar": round(per_lang["ar"][0] / per_lang["ar"][1], 3),
            "emergency_recall": round(tp / (tp + fn), 3),
            "avg_ms": round(sum(laya_ms) / n + share * claude_ms, 1),
            "cost_per_1000_usd": round(share * cost_per_call * 1000, 2),
        }
        results.append(res)
        print(json.dumps(res))
    (ROOT / "evals" / "cascade_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
