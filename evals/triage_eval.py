"""Triage accuracy by language: Laya zero-shot, Laya reader zero-shot, Laya reader with fitted heads,
and Claude as the baseline. The 240 synthetic messages are split 50/50 per (language, intent):
fitted heads only ever see the fit half, and every configuration is scored on the same test half.

    python evals/triage_eval.py                 # Laya configurations (local, free)
    python evals/triage_eval.py --claude        # also the Claude baseline (API cost, ~USD 0.5)
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bawaba.deciders import ClaudeDecider  # noqa: E402
from bawaba.triage import QUESTIONS  # noqa: E402

HEADS = ROOT / "data" / "heads"


def split(rows: list[dict], seed: int = 7) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lang"], r["intent"])].append(r)
    fit, test = [], []
    rng = random.Random(seed)
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        fit += g[: len(g) // 2]
        test += g[len(g) // 2:]
    return fit, test


def score(name: str, predict, test: list[dict]) -> dict:
    per = defaultdict(lambda: {"n": 0, "intent": 0, "urgency": 0, "emerg_tp": 0, "emerg_fn": 0, "emerg_fp": 0, "review": 0})
    secs = []
    for r in test:
        t = time.perf_counter()
        a = predict(r["text"])
        secs.append(time.perf_counter() - t)
        for key in (r["lang"], "all"):
            s = per[key]
            s["n"] += 1
            s["intent"] += a["intent"]["choice"] == r["intent"]
            s["urgency"] += a["urgency"]["choice"] == r["urgency"]
            gold_e, pred_e = r["urgency"] == "emergency", a["urgency"]["choice"] == "emergency"
            s["emerg_tp"] += gold_e and pred_e
            s["emerg_fn"] += gold_e and not pred_e
            s["emerg_fp"] += pred_e and not gold_e
            s["review"] += any(x.get("route") == "review" for x in a.values())
    out = {"config": name, "median_ms": round(sorted(secs)[len(secs) // 2] * 1000, 1)}
    for key, s in per.items():
        tp, fn, fp = s["emerg_tp"], s["emerg_fn"], s["emerg_fp"]
        out[key] = {
            "intent_acc": round(s["intent"] / s["n"], 3),
            "urgency_acc": round(s["urgency"] / s["n"], 3),
            "emergency_recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "emergency_precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "sent_to_review": round(s["review"] / s["n"], 3),
        }
    print(json.dumps(out, ensure_ascii=False))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claude", action="store_true", help="also score the Claude baseline")
    parser.add_argument("--skip-laya", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data" / "tickets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    fit, test = split(rows)
    print(f"fit {len(fit)} rows, test {len(test)} rows")
    results = []

    if not args.skip_laya:
        import laya

        router = laya.Router()
        results.append(score("laya Router, zero-shot", lambda m: router.predict(m, QUESTIONS)["answers"], test))
        reader = laya.Agent(vegaml=True)
        results.append(score("laya reader (vega), zero-shot", lambda m: reader.predict(m, QUESTIONS)["answers"], test))
        t = time.perf_counter()
        report = reader.fit([{"state": r["text"], "gold": {"intent": r["intent"], "urgency": r["urgency"]}} for r in fit],
                            questions=QUESTIONS)
        fit_s = time.perf_counter() - t
        print("fit report:", json.dumps({k: report.get(k) for k in ("n_fit", "n_calibrate", "heads", "kept", "calibration")},
                                        default=str, ensure_ascii=False)[:1500])
        print(f"fit + calibrate took {fit_s:.1f} s")
        HEADS.mkdir(parents=True, exist_ok=True)
        reader.save_calibration(str(HEADS))
        results.append(score("laya reader (vega), fitted heads", lambda m: reader.predict(m, QUESTIONS)["answers"], test))

    if args.claude:
        judge = ClaudeDecider()
        res = score(f"Claude judge ({judge._model}, effort=low)", lambda m: judge.predict(m, QUESTIONS), test)
        res["cost_usd"] = round((judge.input_tokens * 4 + judge.output_tokens * 20) / 1e6, 3)
        results.append(res)

    out = ROOT / "evals" / ("triage_results_claude.json" if args.skip_laya else "triage_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
