"""Decision engines with one interface: predict(state, questions) -> {question_id: answer}.

Questions use Laya's typed format: `choice` (pick one label from `criteria`) and `noul`
(probability that a statement is true). Answers keep Laya's shape: `choice` + `probabilities`,
or `noul`; Laya's vega reader also adds `route` ("auto" or "review") and `abstain`.
"""

import re
import time
from typing import Protocol

from pydantic import BaseModel


class Decider(Protocol):
    name: str
    calls: int
    seconds: float

    def predict(self, state: str | dict, questions: dict) -> dict: ...


class LayaDecider:
    """Laya, local. "router" picks the English or multilingual encoder per input; "vega" uses the
    0.8B reader, which adds conformal prediction sets, abstain and a review route, and can carry
    heads fitted on our own labels."""

    def __init__(self, mode: str = "router", heads_dir: str = ""):
        import laya

        self.name = f"laya-{mode}" + ("-fitted" if heads_dir else "")
        self._model = laya.Router() if mode == "router" else laya.Agent(vegaml=True)
        if heads_dir:
            self._model.load_calibration(heads_dir)
        self.calls, self.seconds = 0, 0.0

    def predict(self, state, questions):
        t = time.perf_counter()
        out = self._model.predict(state, questions)
        self.calls += 1
        self.seconds += time.perf_counter() - t
        return out["answers"]


class _Judgement(BaseModel):
    question_id: str
    choice: str | None = None
    probability: float | None = None  # for yes/no statements: probability the statement is true


class _Judgements(BaseModel):
    answers: list[_Judgement]


class ClaudeDecider:
    """The same typed questions answered by Claude: the baseline Laya is compared against."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low"):
        import anthropic

        self.name = f"claude-judge ({model}, effort={effort})"
        self._client, self._model, self._effort = anthropic.Anthropic(), model, effort
        self.calls, self.seconds = 0, 0.0
        self.input_tokens = self.output_tokens = 0

    def predict(self, state, questions):
        spec = []
        for qid, q in questions.items():
            if q["type"] == "choice":
                spec.append(f"- {qid} (choice): {q['instructions']} Options: {q['criteria']}")
            else:
                spec.append(f"- {qid} (yes/no, give probability 0-1 that it is true): {q['instructions']}")
        prompt = f"<state>\n{state}\n</state>\n\nAnswer each question about the state:\n" + "\n".join(spec)
        t = time.perf_counter()
        resp = self._client.beta.messages.parse(
            model=self._model, max_tokens=2000, messages=[{"role": "user", "content": prompt}],
            output_format=_Judgements, output_config={"effort": self._effort},
        )
        self.calls += 1
        self.seconds += time.perf_counter() - t
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        got = {a.question_id: a for a in resp.parsed_output.answers}
        answers = {}
        for qid, q in questions.items():
            a = got.get(qid)
            if q["type"] == "choice":
                answers[qid] = {"choice": a.choice if a else None, "probabilities": {}}
            else:
                answers[qid] = {"noul": float(a.probability) if a and a.probability is not None else 0.0}
        return answers


_INJECTION = re.compile(r"ignore (all |any )?(previous|prior|above) instructions|disregard .* instructions", re.I)
_WORD = re.compile(r"\w{3,}")
_EMERGENCY = re.compile(r"emergency|urgent|fire|smoke|sparks|flood|gas smell|طوارئ|عاجل|حريق|دخان", re.I)


class KeywordDecider:
    """Deterministic stand-in for tests and CI: word overlap and patterns, no model download."""

    name = "keyword"

    def __init__(self):
        self.calls, self.seconds = 0, 0.0

    def predict(self, state, questions):
        self.calls += 1
        text = state if isinstance(state, str) else " ".join(str(v) for v in state.values())
        question = state.get("question", "") if isinstance(state, dict) else ""
        passage = state.get("passage", text) if isinstance(state, dict) else text
        overlap = len(set(_WORD.findall(question.lower())) & set(_WORD.findall(passage.lower())))
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice" and "urgent" in q["instructions"]:
                answers[qid] = {"choice": "emergency" if _EMERGENCY.search(text) else "routine", "probabilities": {}}
            elif q["type"] == "choice":
                scores = {opt: len(set(_WORD.findall(desc.lower())) & set(_WORD.findall(text.lower())))
                          for opt, desc in q["criteria"].items()}
                best = max(scores, key=scores.get)
                answers[qid] = {"choice": best if scores[best] else list(q["criteria"])[-1], "probabilities": {}}
            elif "instructions addressed" in q["instructions"]:
                answers[qid] = {"noul": 0.95 if _INJECTION.search(passage) else 0.02}
            else:
                answers[qid] = {"noul": min(1.0, overlap / 3)}
        return answers


class CascadeDecider:
    """Ask the fast engine first; when its least confident choice is below the threshold, ask the
    fallback instead. Only `choice` answers carry probabilities, so yes/no questions never escalate."""

    def __init__(self, primary: Decider, fallback: Decider, threshold: float):
        self.primary, self.fallback, self.threshold = primary, fallback, threshold
        self.name = f"cascade({primary.name} -> {fallback.name} below {threshold})"
        self.escalated = 0

    @property
    def calls(self) -> int:
        return self.primary.calls + self.fallback.calls

    @property
    def seconds(self) -> float:
        return self.primary.seconds + self.fallback.seconds

    @property
    def input_tokens(self) -> int:
        return getattr(self.fallback, "input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return getattr(self.fallback, "output_tokens", 0)

    def predict(self, state, questions):
        answers = self.primary.predict(state, questions)
        confidences = [max(a["probabilities"].values()) for a in answers.values() if a.get("probabilities")]
        if confidences and min(confidences) < self.threshold:
            self.escalated += 1
            return self.fallback.predict(state, questions)
        return answers


def make_decider(kind: str, settings) -> Decider:
    if kind == "laya":
        return LayaDecider("router")
    if kind == "laya-fitted":
        return LayaDecider("vega", settings.heads_dir)
    if kind == "claude":
        return ClaudeDecider(settings.model, settings.effort)
    if kind == "cascade":
        return CascadeDecider(LayaDecider("vega", settings.heads_dir), ClaudeDecider(settings.model, settings.effort),
                              settings.cascade_threshold)
    if kind == "keyword":
        return KeywordDecider()
    raise ValueError(f"Unknown decision engine: {kind}")
