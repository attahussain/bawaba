"""Runtime settings, read from environment variables so nothing secret lives in code."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Answer writer: "claude" (Anthropic API) or "mock" (extractive, offline).
    provider: str = field(default_factory=lambda: _env("BAWABA_PROVIDER", "claude"))
    model: str = field(default_factory=lambda: _env("BAWABA_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("BAWABA_EFFORT", "low"))
    # Triage engine: "cascade" (Laya reader with fitted heads; Claude when Laya is unsure), "laya",
    # "laya-fitted", "claude" or "keyword" (tests). The evidence gate: "laya", "claude" or "keyword".
    triage: str = field(default_factory=lambda: _env("BAWABA_TRIAGE", "cascade"))
    gate: str = field(default_factory=lambda: _env("BAWABA_GATE", "laya"))
    # Below this Laya confidence (lower of intent and urgency), the cascade asks Claude instead.
    cascade_threshold: float = field(default_factory=lambda: float(_env("BAWABA_CASCADE_THRESHOLD", "0.5")))
    # Heads fitted by evals/triage_eval.py (laya-fitted and cascade).
    heads_dir: str = field(default_factory=lambda: _env("BAWABA_HEADS", str(ROOT / "data" / "heads")))
    # "fastembed" (multilingual MiniLM, local) or "hash" (tiny offline embedder for tests).
    embedder: str = field(default_factory=lambda: _env("BAWABA_EMBEDDER", "fastembed"))
    kb_dir: Path = field(default_factory=lambda: Path(_env("BAWABA_KB", str(ROOT / "data" / "okf"))))
    db_dir: Path = field(default_factory=lambda: Path(_env("BAWABA_DB", str(ROOT / "data" / "lancedb"))))
    top_k: int = field(default_factory=lambda: int(_env("BAWABA_TOP_K", "4")))
    # Evidence gate thresholds on Laya's noul probabilities.
    min_relevant: float = field(default_factory=lambda: float(_env("BAWABA_MIN_RELEVANT", "0.5")))
    min_answers: float = field(default_factory=lambda: float(_env("BAWABA_MIN_ANSWERS", "0.5")))
    max_injection: float = field(default_factory=lambda: float(_env("BAWABA_MAX_INJECTION", "0.7")))


def get_settings() -> Settings:
    return Settings()
