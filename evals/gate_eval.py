"""Evidence gate: does each retrieved passage contain the answer? Laya vs Claude, by language.

Pairs come from evals/qa.jsonl: every question is searched and each of its top passages is
labelled positive when it comes from the expected document and contains an expected fact.
Handoff questions contribute only negatives. Questions are split in half (per language); gate
heads are fitted on the first half's pairs and every configuration is scored on the second half.

    python evals/gate_eval.py            # Laya configurations (local)
    python evals/gate_eval.py --claude   # also Claude as the judge (API cost)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bawaba.config import Settings  # noqa: E402
from bawaba.deciders import ClaudeDecider  # noqa: E402
from bawaba.gate import EVIDENCE  # noqa: E402
from bawaba.kb import KnowledgeBase  # noqa: E402

ANSWERS_Q = {"answers": EVIDENCE["answers"]}
GATE_HEADS = ROOT / "data" / "heads_gate"
TOP = 6


def norm(text: str) -> str:
    return "".join(text.lower().split())


def build_pairs() -> tuple[list[dict], list[dict]]:
    settings = Settings(embedder="fastembed", db_dir=ROOT / "data" / "lancedb_eval", top_k=TOP)
    kb = KnowledgeBase(settings)
    kb.ingest()
    cases = [json.loads(line) for line in (ROOT / "evals" / "qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_lang = defaultdict(list)
    for c in cases:
        by_lang[c["lang"]].append(c)
    fit_ids = {c["id"] for group in by_lang.values() for c in group[::2]}  # every other question, per language
    fit, test = [], []
    for c in cases:
        for p in kb.search(c["question"], k=TOP):
            positive = (c["expect"] == "answer" and p["doc_id"] == c.get("doc")
                        and any(norm(k) in norm(p["text"]) for k in c.get("keywords", [])))
            pair = {"qid": c["id"], "lang": c["lang"], "question": c["question"], "passage": p["text"],
                    "chunk_id": p["chunk_id"], "label": positive}
            (fit if c["id"] in fit_ids else test).append(pair)
    return fit, test


def score(name: str, prob, test: list[dict], threshold: float = 0.5) -> dict:
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    secs = []
    for pair in test:
        t = time.perf_counter()
        keep = prob(pair) >= threshold
        secs.append(time.perf_counter() - t)
        for key in (pair["lang"], "all"):
            s = stats[key]
            s["tp" if keep and pair["label"] else "fp" if keep else "fn" if pair["label"] else "tn"] += 1
    out = {"config": name, "pairs": len(test), "median_ms": round(sorted(secs)[len(secs) // 2] * 1000, 1)}
    for key, s in stats.items():
        tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
        out[key] = {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                    "accuracy": round((tp + tn) / (tp + fp + fn + tn), 3), "positives": tp + fn}
    print(json.dumps(out, ensure_ascii=False))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claude", action="store_true")
    parser.add_argument("--skip-laya", action="store_true")
    args = parser.parse_args()
    fit, test = build_pairs()
    print(f"fit {len(fit)} pairs ({sum(p['label'] for p in fit)} positive), test {len(test)} pairs ({sum(p['label'] for p in test)} positive)")
    state = lambda p: {"question": p["question"], "passage": p["passage"]}  # noqa: E731
    results = []
    if not args.skip_laya:
        import laya

        router = laya.Router()
        results.append(score("laya Router, zero-shot", lambda p: router.predict(state(p), ANSWERS_Q)["answers"]["answers"]["noul"], test))
        reader = laya.Agent(vegaml=True)
        results.append(score("laya reader (vega), zero-shot", lambda p: reader.predict(state(p), ANSWERS_Q)["answers"]["answers"]["noul"], test))
        report = reader.fit([{"state": state(p), "gold": {"answers": bool(p["label"])}} for p in fit], questions=ANSWERS_Q)
        print("fit report:", json.dumps(report.get("heads"), default=str)[:600])
        reader.save_calibration(str(GATE_HEADS))  # separate from the triage heads the pipeline loads
        results.append(score("laya reader (vega), fitted gate head", lambda p: reader.predict(state(p), ANSWERS_Q)["answers"]["answers"]["noul"], test))
    if args.claude:
        judge = ClaudeDecider()
        res = score(f"Claude judge ({judge._model}, effort=low)", lambda p: judge.predict(state(p), ANSWERS_Q)["answers"]["noul"], test)
        res["cost_usd"] = round((judge.input_tokens * 4 + judge.output_tokens * 20) / 1e6, 3)
        results.append(res)
    name = "gate_results_claude.json" if args.skip_laya else "gate_results.json"
    (ROOT / "evals" / name).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
