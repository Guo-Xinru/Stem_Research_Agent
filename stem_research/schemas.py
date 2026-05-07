"""Typed data structures for the minimal StemResearch loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ResearchMode = Literal["baseline", "specialized"]
GoldFactLabel = Literal["addressed", "partially_addressed", "not_addressed"]


@dataclass(frozen=True)
class Question:
    id: str
    question: str


@dataclass(frozen=True)
class ResearchProtocol:
    search_strategy: list[str]
    source_selection_criteria: list[str]
    answer_structure: list[str]
    verification_rules: list[str]
    citation_requirements: list[str]
    stopping_criteria: list[str]
    failure_modes_to_avoid: list[str]


@dataclass(frozen=True)
class ResearchOutput:
    question_id: str
    mode: ResearchMode
    question: str
    answer: str
    major_claims: list[str]
    citations: list[str]
    sources_used: list[str]
    uncertainty_notes: list[str]


@dataclass(frozen=True)
class GoldFactEvaluation:
    fact_id: str
    fact: str
    label: GoldFactLabel
    matched_keywords: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    question_id: str
    mode: ResearchMode
    evaluation_mode: Literal["heuristic", "llm"]
    gold_fact_evaluations: list[GoldFactEvaluation]
    gold_fact_recall: float
    unsupported_claim_count: int
    citation_support_notes: list[str]
    source_quality_notes: list[str]
    brief_critique: str


@dataclass(frozen=True)
class ExperimentResult:
    metadata: dict
    generated_protocol: ResearchProtocol
    protocol_provenance: dict
    per_question: list[dict]
    summary_metrics: dict
