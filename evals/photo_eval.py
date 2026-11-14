"""Photo triage on 15 public-domain photos (data/photos, labels by hand): Laya's reader vs Claude,
and the cascade between them. Each engine sees each photo once; cascade thresholds are scored
from those cached answers.

    python evals/photo_eval.py          # needs ANTHROPIC_API_KEY for the Claude column (~USD 0.2)
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bawaba.deciders import ClaudeDecider, LayaDecider  # noqa: E402
from bawaba.photo import PHOTO_QUESTIONS  # noqa: E402

PHOTOS = ROOT / "data" / "photos"


def main() -> int:
    rows = [json.loads(line) for line in (PHOTOS / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    engines = {"laya": LayaDecider("vega"), "claude": ClaudeDecider()}
    preds = {name: [] for name in engines}
    for name, engine in engines.items():
        engine.predict_image(PHOTOS / rows[0]["file"], PHOTO_QUESTIONS)  # warm-up, not timed below
        for r in rows:
            t = time.perf_counter()
            a = engine.predict_image(PHOTOS / r["file"], PHOTO_QUESTIONS)
            ms = (time.perf_counter() - t) * 1000
            conf = min((max(x["probabilities"].values()) for x in a.values() if x.get("probabilities")), default=1.0)
            preds[name].append({"urgency": a["urgency"]["choice"], "issue": a["issue"]["choice"], "conf": conf, "ms": ms})

    def score(label, picks):
        n = len(rows)
        emerg = [i for i, r in enumerate(rows) if r["urgency"] == "emergency"]
        flagged = [i for i, p in enumerate(picks) if p["urgency"] == "emergency"]
        res = {
            "config": label,
            "urgency_acc": round(sum(p["urgency"] == r["urgency"] for p, r in zip(picks, rows)) / n, 3),
            "issue_acc": round(sum(p["issue"] == r["issue"] for p, r in zip(picks, rows)) / n, 3),
            "emergency_recall": round(sum(i in flagged for i in emerg) / len(emerg), 3),
            "false_emergencies": sum(rows[i]["urgency"] != "emergency" for i in flagged),
            "median_ms": round(sorted(p["ms"] for p in picks)[n // 2]),
        }
        print(json.dumps(res))
        return res

    results = [score("Laya reader, zero-shot", preds["laya"]), score("Claude", preds["claude"])]
    for th in (0.5, 0.6):
        picks = [c if lay["conf"] < th else lay for lay, c in zip(preds["laya"], preds["claude"])]
        res = score(f"cascade below {th}", picks)
        res["sent_to_claude"] = sum(lay["conf"] < th for lay in preds["laya"])
        results.append(res)
    claude = engines["claude"]
    results[1]["cost_usd_incl_warmup"] = round((claude.input_tokens * 4 + claude.output_tokens * 20) / 1e6, 3)
    detail = [{"file": r["file"], "gold": r["urgency"], "laya": lay["urgency"], "laya_conf": round(lay["conf"], 2),
               "claude": c["urgency"]} for r, lay, c in zip(rows, preds["laya"], preds["claude"])]
    (ROOT / "evals" / "photo_results.json").write_text(json.dumps({"results": results, "per_photo": detail}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
