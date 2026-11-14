"""Offline tests: keyword decision engine, mock writer, hashing embedder. No model download, no API key."""

import pytest

from bawaba.config import ROOT, Settings
from bawaba.deciders import CascadeDecider, KeywordDecider
from bawaba.gate import gate
from bawaba.kb import KnowledgeBase, chunk, language
from bawaba.pipeline import Desk
from evals.triage_eval import split


@pytest.fixture()
def settings(tmp_path):
    return Settings(provider="mock", triage="keyword", gate="keyword", embedder="hash",
                    kb_dir=ROOT / "data" / "okf", db_dir=tmp_path / "db")


@pytest.fixture()
def desk(settings):
    kb = KnowledgeBase(settings)
    kb.ingest()
    return Desk(settings, kb)


def test_language_detection_and_bilingual_chunks():
    assert language("How do I get my deposit back?") == "en"
    assert language("كيف أسترد مبلغ التأمين؟") == "ar"
    rows = chunk("deposits", (ROOT / "data" / "okf" / "policies" / "deposits.md").read_text(encoding="utf-8").split("---", 2)[2])
    assert {r["lang"] for r in rows} == {"en", "ar"}
    assert all(r["chunk_id"].startswith("deposits#") for r in rows)


def test_answer_path_cites_the_policy(desk):
    out = desk.handle("How long until my security deposit is refunded after move-out?")
    assert out.action == "answer" and out.citations and out.citations[0].startswith("deposits#")
    assert "30 days" in out.reply


def test_emergencies_are_escalated_before_any_retrieval(desk):
    out = desk.handle("Water is leaking onto the sockets, this is an emergency!")
    assert out.action == "escalate" and "retrieve" not in out.timings_ms
    assert desk.handle("يوجد حريق ودخان في الممر، طوارئ").reply.startswith("يبدو")  # replies in Arabic


def test_no_evidence_means_handoff_without_calling_the_llm(desk):
    before = desk.writer.calls
    out = desk.handle("Can I keep a cat?")
    assert out.action == "handoff" and "answer" not in out.timings_ms
    assert desk.writer.calls == before


def test_injected_sentence_is_stripped_and_facts_kept(desk):
    out = desk.handle("When is the corridor cleaning schedule?")
    assert "contractor-note#0" in out.stripped
    assert out.action == "answer" and "Tuesday" in out.reply
    assert "card" not in out.reply.lower() and "@" not in out.reply


def test_gate_drops_irrelevant_passages(settings):
    passages = [{"chunk_id": "x#0", "text": "Swimming pool hours are 6 am to 10 pm."}]
    result = gate(KeywordDecider(), "What is the fee for a bounced cheque?", passages, settings)
    assert not result.kept and result.dropped[0]["reason"] == "not relevant"


class _Fixed:
    def __init__(self, name, answers):
        self.name, self.answers, self.calls, self.seconds = name, answers, 0, 0.0

    def predict(self, state, questions):
        self.calls += 1
        return self.answers


def test_cascade_falls_back_only_when_unsure():
    sure = {"intent": {"choice": "payments", "probabilities": {"payments": 0.9, "other": 0.1}}}
    unsure = {"intent": {"choice": "payments", "probabilities": {"payments": 0.4, "other": 0.35, "move": 0.25}}}
    fallback = _Fixed("claude", {"intent": {"choice": "move", "probabilities": {}}})
    assert CascadeDecider(_Fixed("laya", sure), fallback, 0.5).predict("m", {})["intent"]["choice"] == "payments"
    cascade = CascadeDecider(_Fixed("laya", unsure), fallback, 0.5)
    assert cascade.predict("m", {})["intent"]["choice"] == "move" and cascade.escalated == 1


def test_triage_split_is_stratified_and_disjoint():
    rows = [{"text": f"{lang}{i}{intent}", "lang": lang, "intent": intent}
            for lang in ("en", "ar") for intent in ("a", "b") for i in range(6)]
    fit, test = split(rows)
    assert len(fit) == len(test) == 12
    assert not {r["text"] for r in fit} & {r["text"] for r in test}
    assert {(r["lang"], r["intent"]) for r in fit} == {(r["lang"], r["intent"]) for r in test}


def test_photo_escalates_even_when_the_message_is_calm(desk):
    out = desk.handle("Please have a look when you can", photo=str(ROOT / "data" / "photos" / "ceiling-collapse-1.jpg"))
    assert out.action == "escalate" and out.photo == {"urgency": "emergency", "issue": "structural"}


def test_harmless_photo_does_not_raise_urgency(desk):
    out = desk.handle("How long until my security deposit is refunded after move-out?",
                      photo=str(ROOT / "data" / "photos" / "living-room-1.jpg"))
    assert out.action == "answer" and out.photo["urgency"] == "routine"


def test_deprecated_policy_is_never_indexed_and_drafts_are_labelled(desk):
    texts = [h["text"] for h in desk.kb.search("bounced cheque fee", k=10)]
    assert not any("1,000" in t or "1000" in t for t in texts)  # the 2025 fee is deprecated
    bbq = [h for h in desk.kb.search("balcony barbecue grill", k=10) if h["doc_id"] == "balcony-bbq"]
    assert bbq and bbq[0]["status"] == "draft" and "unverified" in bbq[0]["provenance"]


def test_policies_carry_trust_from_the_bundle():
    from bawaba.okf import load_bundle

    concepts = {c.concept_id: c for c in load_bundle(ROOT / "data" / "okf")}
    assert concepts["policies/rent"].trust == "human-reviewed"
    assert concepts["external/contractor-note"].trust == "unverified"
    assert concepts["policies/rent-rev0"].status == "deprecated"


def test_provenance_stays_out_of_the_text_the_gate_judges(desk):
    note = next(h for h in desk.kb.search("corridor cleaning schedule", k=10) if h["doc_id"] == "contractor-note")
    assert "unverified" in note["provenance"] and "unverified" not in note["text"]
