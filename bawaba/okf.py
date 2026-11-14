"""Policies as an Open Knowledge Format (OKF v0.2) bundle.

Each concept is a markdown file with YAML frontmatter (spec: github.com/GoogleCloudPlatform/open-knowledge-format).
The fields used here: `type`, `title`, `status` (draft | stable | deprecated; absent means stable)
and `verified`, from which the trust tier is derived (spec §5.3). Deprecated concepts stay in the
bundle for history but are never indexed, so a superseded rule cannot be quoted.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

RESERVED = {"index.md", "log.md"}


@dataclass
class Concept:
    concept_id: str  # bundle path without ".md"
    title: str
    type: str
    status: str
    trust: str  # unverified | machine-confirmed | human-reviewed
    verified: str  # "human:property-manager on 2026-10-01", or ""
    body: str


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    head, _, body = text[3:].partition("\n---")
    meta = yaml.safe_load(head) or {}
    return (meta if isinstance(meta, dict) else {}), body.lstrip("-").lstrip("\n")


def trust_tier(verified) -> str:
    events = verified if isinstance(verified, list) else [verified] if verified else []
    actors = [str(e.get("by", "")) for e in events if isinstance(e, dict)]
    if not actors:
        return "unverified"
    return "human-reviewed" if any(a.startswith("human:") for a in actors) else "machine-confirmed"


def load_bundle(root: Path) -> list[Concept]:
    concepts = []
    for path in sorted(root.rglob("*.md")):
        if path.name in RESERVED:
            continue
        meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        if not meta.get("type"):
            continue  # not a conformant concept
        verified = meta.get("verified")
        events = verified if isinstance(verified, list) else [verified] if verified else []
        checks = ", ".join(f"{e.get('by')} on {str(e.get('at', ''))[:10]}" for e in events if isinstance(e, dict))
        concepts.append(Concept(
            concept_id=path.relative_to(root).with_suffix("").as_posix(),
            title=str(meta.get("title") or path.stem),
            type=str(meta["type"]),
            status=str(meta.get("status") or "stable"),
            trust=trust_tier(verified),
            verified=checks,
            body=body.strip(),
        ))
    return concepts


def provenance(c: Concept) -> str:
    line = f"[{c.title}: {c.status}, {c.trust}"
    return line + (f", verified by {c.verified}]" if c.verified else "]")
