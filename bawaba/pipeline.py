"""Message in, decision out: answer it, hand it to a person, or escalate it as an emergency."""

import time
from dataclasses import asdict, dataclass, field

from bawaba.answer import make_writer
from bawaba.deciders import make_decider
from bawaba.gate import gate
from bawaba.kb import KnowledgeBase, language
from bawaba.triage import triage

HANDOFF = {
    "en": "Thanks for your message. I could not find this in our policies, so I have passed it to the service team; they will reply shortly.",
    "ar": "شكراً لرسالتك. لم أجد هذه المعلومة في سياساتنا، لذلك حوّلتها إلى فريق الخدمة وسيتواصلون معك قريباً.",
}
ESCALATE = {
    "en": "This sounds urgent. I have raised an emergency ticket and alerted the 24-hour team. If anyone is in danger, call 999.",
    "ar": "يبدو أن الأمر عاجل. فتحتُ بلاغ طوارئ وأبلغت فريق الصيانة على مدار الساعة. إذا كان أي شخص في خطر، اتصل على 999.",
}


@dataclass
class Outcome:
    action: str  # "answer" | "handoff" | "escalate"
    reply: str
    lang: str
    intent: str
    urgency: str
    citations: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Desk:
    def __init__(self, settings, kb: KnowledgeBase | None = None, triage_decider=None, gate_decider=None, writer=None):
        self.settings = settings
        self.kb = kb or KnowledgeBase(settings)
        self.triage_decider = triage_decider or make_decider(settings.triage, settings)
        self.gate_decider = gate_decider or make_decider(settings.gate, settings)
        self.writer = writer or make_writer(settings)

    def handle(self, message: str) -> Outcome:
        lang, ms = language(message), {}
        t = time.perf_counter()
        tri = triage(self.triage_decider, message)
        ms["triage"] = round((time.perf_counter() - t) * 1000, 1)
        base = {"lang": lang, "intent": tri.intent, "urgency": tri.urgency}
        if tri.urgency == "emergency":
            return Outcome("escalate", ESCALATE[lang], timings_ms=ms, **base)

        t = time.perf_counter()
        passages = self.kb.search(message)
        ms["retrieve"] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        g = gate(self.gate_decider, message, passages, self.settings)
        ms["gate"] = round((time.perf_counter() - t) * 1000, 1)
        trace = {"kept": [p["chunk_id"] for p in g.kept], "stripped": g.stripped,
                 "dropped": [{"chunk_id": d["chunk_id"], "reason": d["reason"]} for d in g.dropped]}
        if not g.kept:  # no evidence passed the gate: the LLM is never called
            return Outcome("handoff", HANDOFF[lang], timings_ms=ms, **base, **trace)

        t = time.perf_counter()
        ans = self.writer.write(message, g.kept)
        ms["answer"] = round((time.perf_counter() - t) * 1000, 1)
        if not ans.answerable:
            return Outcome("handoff", HANDOFF[lang], timings_ms=ms, **base, **trace)
        return Outcome("answer", ans.answer, citations=ans.citations, timings_ms=ms, **base, **trace)
