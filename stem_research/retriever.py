"""A small deterministic local evidence retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from stem_research.schemas import EvidenceItem, PaperSection


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}


class EvidenceRetrieverTool:
    """Retrieve likely evidence chunks using deterministic keyword overlap."""

    def retrieve(
        self,
        question: str,
        sections: list[dict] | list[PaperSection],
        top_k: int = 5,
    ) -> list[EvidenceItem]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = _content_tokens(question)
        chunks = _chunks_from_sections(sections)
        scored: list[EvidenceItem] = []
        for index, chunk in enumerate(chunks):
            text_tokens = _content_tokens(chunk.text)
            score = _score(query_tokens, text_tokens)
            if score <= 0:
                continue
            scored.append(
                EvidenceItem(
                    section_name=chunk.section_name,
                    text=chunk.text,
                    score=round(score, 4),
                    evidence_id=f"{chunk.section_name}:{index}",
                )
            )
        scored.sort(key=lambda item: (-item.score, item.section_name.lower(), item.text[:80]))
        return scored[:top_k]


def _chunks_from_sections(sections: list[dict] | list[PaperSection]) -> list[PaperSection]:
    chunks: list[PaperSection] = []
    for raw_section in sections:
        if isinstance(raw_section, PaperSection):
            section_name = raw_section.section_name
            text = raw_section.text
        else:
            section_name = str(raw_section.get("section_name") or raw_section.get("name") or "Unknown")
            text = str(raw_section.get("text") or "")
        for paragraph in _paragraph_chunks(text):
            chunks.append(PaperSection(section_name=section_name, text=paragraph))
    return chunks


def _paragraph_chunks(text: str, max_words: int = 120) -> Iterable[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= max_words:
            yield paragraph
            continue
        for start in range(0, len(words), max_words):
            yield " ".join(words[start : start + max_words])


def _content_tokens(text: str) -> Counter[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9-]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]
    return Counter(tokens)


def _score(query_tokens: Counter[str], text_tokens: Counter[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = set(query_tokens) & set(text_tokens)
    if not overlap:
        return 0.0
    weighted_overlap = sum(query_tokens[token] * math.log1p(text_tokens[token]) for token in overlap)
    coverage = len(overlap) / len(query_tokens)
    density = len(overlap) / math.sqrt(max(1, sum(text_tokens.values())))
    return weighted_overlap + coverage + density


def normalized_tokens(text: str) -> set[str]:
    return set(_content_tokens(text))
