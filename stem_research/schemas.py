"""Typed data structures for the minimal StemResearch loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ResearchMode = Literal[
    "baseline_no_tool",
    "baseline_with_tool",
    "specialized_with_protocol_and_tool",
]
RunMode = Literal["offline", "llm"]


@dataclass(frozen=True)
class PaperSection:
    section_name: str
    text: str


@dataclass(frozen=True)
class PaperContext:
    paper_title: str
    abstract: str
    sections: list[PaperSection]


@dataclass(frozen=True)
class EvidenceItem:
    section_name: str
    text: str
    score: float = 0.0
    evidence_id: str | None = None


@dataclass(frozen=True)
class QasperExample:
    id: str
    domain: str
    question: str
    context: PaperContext
    reference_answer: str | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    answer_type: str = "abstractive"

    def without_references(self) -> "QasperExample":
        """Return the example as it is allowed to be seen by the researcher."""
        return QasperExample(
            id=self.id,
            domain=self.domain,
            question=self.question,
            context=self.context,
            reference_answer=None,
            evidence=[],
            answer_type=self.answer_type,
        )


@dataclass(frozen=True)
class ResearchProtocol:
    task_class: str
    observed_question_types: list[str]
    evidence_strategy: list[str]
    answer_rules: list[str]
    tool_policy: dict[str, Any]
    verification_rules: list[str]
    failure_modes: list[str]


@dataclass(frozen=True)
class ResearchOutput:
    id: str
    mode: ResearchMode
    question: str
    answer: str
    selected_evidence: list[EvidenceItem]
    used_protocol: bool


@dataclass(frozen=True)
class ProtocolAdherence:
    used_required_tool: bool
    selected_evidence_present: bool
    answer_grounded_in_evidence: bool
    avoided_known_failure_modes: bool
    score: float


@dataclass(frozen=True)
class EvaluationResult:
    id: str
    mode: ResearchMode
    answer_token_f1: float
    evidence_recall: float
    evidence_precision: float
    unsupported_claim_count: int
    protocol_adherence: ProtocolAdherence | None
    answer_length_words: int


@dataclass(frozen=True)
class AggregateMetrics:
    mode: ResearchMode
    num_examples: int
    avg_answer_token_f1: float
    avg_evidence_recall: float
    avg_evidence_precision: float
    avg_unsupported_claim_count: float
    avg_protocol_adherence: float | None
    avg_answer_length_words: float
