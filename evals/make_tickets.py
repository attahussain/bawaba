"""Generate the synthetic, labelled tenant-message set used to fit and evaluate triage.

Claude writes 24 messages per (intent, language) with an even urgency split. The output is
committed (data/tickets.jsonl) so results are reproducible without rerunning this script.
"""

import json
import os
from pathlib import Path

import anthropic
from pydantic import BaseModel

INTENTS = {
    "maintenance": "repairs: AC, leaks, plumbing, electrical, appliances, pests",
    "payments": "rent, cheques, direct debit, fees, deposits being paid, receipts",
    "move": "moving in or out, renewal, early termination, move-out inspection, deposit refund after leaving",
    "complaint": "noise, neighbours, security staff, cleanliness, parking disputes",
    "other": "amenities, visitor parking, general questions, anything else",
}
URGENCY = ["routine", "soon", "emergency"]


class Message(BaseModel):
    text: str
    urgency: str  # routine | soon | emergency


class Batch(BaseModel):
    messages: list[Message]


def main() -> None:
    client = anthropic.Anthropic()
    out = Path(__file__).resolve().parent.parent / "data" / "tickets.jsonl"
    rows = []
    for intent, scope in INTENTS.items():
        for lang in ("en", "ar"):
            language = ("English, as written by UAE residents of many nationalities" if lang == "en"
                        else "Arabic: half Modern Standard Arabic, half Gulf dialect, as typed on a phone")
            prompt = (
                f"Write 24 distinct messages that tenants of a residential tower in Dubai send to the building "
                f"management service desk. Topic: {intent} ({scope}). Language: {language}. "
                f"Exactly 8 must be urgency 'routine', 8 'soon' and 8 'emergency' (emergency = safety risk, "
                f"no water/power/AC in summer, active leak, locked out at night, or similar; for topics that rarely "
                f"have emergencies, use the most urgent realistic cases). Vary length (5 to 60 words), tone and "
                f"detail; include a few typos. No real company names, no real phone numbers or personal data."
            )
            resp = client.beta.messages.parse(
                model=os.getenv("BAWABA_MODEL", "claude-opus-5-5"), max_tokens=8000,
                messages=[{"role": "user", "content": prompt}], output_format=Batch,
            )
            for m in resp.parsed_output.messages:
                if m.urgency in URGENCY:
                    rows.append({"text": m.text.strip(), "lang": lang, "intent": intent, "urgency": m.urgency})
            print(intent, lang, len(resp.parsed_output.messages))
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print("wrote", len(rows), "rows to", out)


if __name__ == "__main__":
    main()
