"""Triage: intent and urgency of a tenant message, decided before any LLM call."""

from dataclasses import dataclass

INTENTS = {
    "maintenance": "repairs: AC, leaks, plumbing, electrical, appliances, pests",
    "payments": "rent, cheques, direct debit, fees, receipts",
    "move": "moving in or out, renewal, early termination, move-out inspection, deposit refund after leaving",
    "complaint": "noise, neighbours, security staff, cleanliness, parking disputes",
    "other": "amenities, visitor parking, general questions, anything else",
}
URGENCY = {
    "routine": "can wait for normal working hours",
    "soon": "should be handled within a day or two",
    "emergency": "safety risk, active leak, no water, power or AC in summer, locked out at night",
}
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What does the tenant need help with?", "criteria": INTENTS},
    "urgency": {"type": "choice", "instructions": "How urgent is the tenant's request?", "criteria": URGENCY},
}


@dataclass
class Triage:
    intent: str
    urgency: str
    needs_review: bool  # the decision engine itself asked for a person to check (Laya vega route)
    confidence: dict


def triage(decider, message: str) -> Triage:
    answers = decider.predict(message, QUESTIONS)
    intent, urgency = answers["intent"], answers["urgency"]
    conf = {qid: max(a.get("probabilities", {}).values(), default=None) for qid, a in answers.items()}
    review = any(a.get("route") == "review" for a in answers.values())
    return Triage(intent["choice"], urgency["choice"], review, conf)
