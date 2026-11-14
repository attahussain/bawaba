"""Bilingual knowledge base: section chunks, multilingual embeddings, BM25 + vector search on LanceDB."""

import hashlib
import math
import re
from pathlib import Path

import lancedb
from lancedb.index import FTS

from bawaba.config import Settings

TABLE = "kb"
_ARABIC = re.compile(r"[؀-ۿ]")


def language(text: str) -> str:
    letters = re.findall(r"[A-Za-z؀-ۿ]", text)
    if not letters:
        return "en"
    return "ar" if sum(bool(_ARABIC.match(c)) for c in letters) / len(letters) > 0.3 else "en"


def chunk(doc_id: str, text: str) -> list[dict]:
    """One chunk per `##` section, prefixed with the document title so each chunk stands alone."""
    lines = text.splitlines()
    title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), doc_id)
    sections: list[tuple[str, list[str]]] = []
    for ln in lines:
        if ln.startswith("## "):
            sections.append((ln[3:].strip(), []))
        elif sections and not ln.startswith("# "):
            sections[-1][1].append(ln)
    rows = []
    for i, (heading, body) in enumerate(sections):
        content = "\n".join(body).strip()
        if content:
            rows.append({"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "title": f"{title} - {heading}",
                         "text": f"{heading}\n{content}", "lang": language(content)})
    return rows


class Embedder:
    def __init__(self, kind: str):
        self._model = None
        if kind == "fastembed":
            from fastembed import TextEmbedding

            self._model = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return [v.tolist() for v in self._model.embed(texts)]
        return [_hash_embed(t) for t in texts]


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"\w+", text.lower()):
        vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedder)
        settings.db_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(settings.db_dir))

    def ingest(self, kb_dir: Path | None = None) -> int:
        rows = []
        for path in sorted((kb_dir or self.settings.kb_dir).glob("*.md")):
            rows.extend(chunk(path.stem, path.read_text(encoding="utf-8")))
        for row, vec in zip(rows, self.embedder.embed([r["text"] for r in rows])):
            row["vector"] = vec
        table = self.db.create_table(TABLE, data=rows, mode="overwrite")
        table.create_index("text", config=FTS(), replace=True)
        return len(rows)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """BM25 and vector rankings merged by reciprocal rank fusion."""
        k = k or self.settings.top_k
        table = self.db.open_table(TABLE)
        pool = max(k * 3, 10)
        vec = table.search(self.embedder.embed([query])[0]).metric("cosine").limit(pool).to_list()
        cleaned = re.sub(r"[^\w\s-]", " ", query)
        try:
            kw = table.search(cleaned, query_type="fts").limit(pool).to_list() if cleaned.strip() else []
        except Exception:  # the full-text parser rejects some inputs; vector search still answers
            kw = []
        fused: dict[str, dict] = {}
        for ranking in (vec, kw):
            for rank, hit in enumerate(ranking, start=1):
                entry = fused.setdefault(hit["chunk_id"], {key: hit[key] for key in ("chunk_id", "doc_id", "title", "text", "lang")} | {"score": 0.0})
                entry["score"] += 1.0 / (60 + rank)
        return sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:k]
