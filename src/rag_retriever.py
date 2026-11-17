"""Lightweight in-memory TF-IDF + Cosine Similarity RAG retriever (0 external DB, Vercel ready)."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"

_STOPWORDS = {
    "a",
    "o",
    "e",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "para",
    "por",
    "com",
    "sem",
    "um",
    "uma",
    "que",
    "se",
    "ou",
    "ao",
    "os",
    "as",
    "the",
    "and",
    "or",
    "in",
    "on",
    "of",
    "to",
    "for",
    "with",
    "is",
    "are",
}


class KnowledgeDoc(BaseModel):
    """Single competitive programming knowledge base document."""

    id: str
    source: str
    title: str
    category: str
    tags: list[str] = Field(default_factory=list)
    expected_complexity: str = ""
    n_limits: str = ""
    technical_summary: str
    common_mistakes: str
    url: str


class RAGReference(BaseModel):
    """Retrieved reference returned to the monitor and LLM."""

    id: str
    source: str
    title: str
    category: str
    technical_summary: str
    common_mistakes: str
    url: str
    score: float = 0.0


def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Tokenize Portuguese/English prose and C++/Python code tokens."""
    norm = _normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9_]+|<=|>=|<<|>>", norm)
    return [tok for tok in raw_tokens if tok not in _STOPWORDS and len(tok) > 1]


class RAGRetriever:
    """In-memory TF-IDF + cosine similarity retriever across the 5 canonical sources."""

    def __init__(self, kb_dir: Path = DEFAULT_KB_DIR) -> None:
        self.kb_dir = kb_dir
        self.docs: list[KnowledgeDoc] = self._load_docs()
        self._doc_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._doc_vectors: list[dict[str, float]] = []
        self._build_index()

    def _load_docs(self) -> list[KnowledgeDoc]:
        docs: list[KnowledgeDoc] = []
        if not self.kb_dir.exists():
            return docs
        for json_file in sorted(self.kb_dir.glob("*.json")):
            raw_list = json.loads(json_file.read_text(encoding="utf-8"))
            for item in raw_list:
                docs.append(KnowledgeDoc.model_validate(item))
        return docs

    def _build_index(self) -> None:
        n_docs = len(self.docs)
        if n_docs == 0:
            return

        df: Counter[str] = Counter()
        for doc in self.docs:
            # Weight title and tags higher by repeating them in the indexed text
            searchable = (
                f"{doc.title} {doc.title} {' '.join(doc.tags)} {' '.join(doc.tags)} "
                f"{doc.category} {doc.technical_summary} {doc.common_mistakes} {doc.n_limits}"
            )
            tokens = tokenize(searchable)
            self._doc_tokens.append(tokens)
            for unique_tok in set(tokens):
                df[unique_tok] += 1

        self._idf = {
            tok: math.log((n_docs + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()
        }

        for tokens in self._doc_tokens:
            vec = self._tfidf_vector(tokens)
            self._doc_vectors.append(vec)

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        vec = {
            tok: (count / total) * self._idf.get(tok, 1.0) for tok, count in tf.items()
        }
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    @staticmethod
    def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        return sum(val * v2.get(tok, 0.0) for tok, val in v1.items())

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        source_filter: str | None = None,
        tag_filter: str | None = None,
    ) -> list[RAGReference]:
        """Return top_k most relevant knowledge documents with cosine similarity scores."""
        if not self.docs:
            return []

        q_vec = self._tfidf_vector(tokenize(query))
        scored: list[tuple[float, KnowledgeDoc]] = []

        src_norm = _normalize_text(source_filter) if source_filter else None
        tag_norm = _normalize_text(tag_filter) if tag_filter else None

        for doc, doc_vec in zip(self.docs, self._doc_vectors, strict=True):
            if src_norm and src_norm not in _normalize_text(doc.source):
                continue
            if tag_norm and not any(tag_norm in _normalize_text(t) for t in doc.tags):
                continue
            score = self._cosine(q_vec, doc_vec)
            scored.append((score, doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[RAGReference] = []
        for score, doc in scored[:top_k]:
            results.append(
                RAGReference(
                    id=doc.id,
                    source=doc.source,
                    title=doc.title,
                    category=doc.category,
                    technical_summary=doc.technical_summary,
                    common_mistakes=doc.common_mistakes,
                    url=doc.url,
                    score=round(score, 4),
                )
            )
        return results
