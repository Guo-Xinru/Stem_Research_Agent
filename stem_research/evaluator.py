"""Concrete deterministic metrics for QASPER-mini outputs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict
from statistics import mean
from typing import Any

from stem_research.retriever import normalized_tokens
from stem_research.schemas import (
    AggregateMetrics,
    EvaluationResult,
    EvidenceItem,
    ProtocolAdherence,
    QasperExample,
    ResearchOutput,
)


class Evaluator:
    """Scores outputs against hidden reference answers and gold evidence."""

    def evaluate_qasper(self, example: QasperExample, output: ResearchOutput) -> EvaluationResult:
        reference = example.reference_answer or ""
        answer_f1 = answer_token_f1(output.answer, reference)
        recall, precision = evidence_scores(output.selected_evidence, example.evidence)
        unsupported = unsupported_claim_count(output.answer, output.selected_evidence)
        adherence = (
            protocol_adherence(output)
            if output.mode == "specialized_with_protocol_and_tool"
            else None
        )
        return EvaluationResult(
            id=example.id,
            mode=output.mode,
            answer_token_f1=answer_f1,
            evidence_recall=recall,
            evidence_precision=precision,
            unsupported_claim_count=unsupported,
            protocol_adherence=adherence,
            answer_length_words=len(_tokens(output.answer)),
        )

    def evaluate(self, example: QasperExample, output: ResearchOutput) -> EvaluationResult:
        return self.evaluate_qasper(example, output)


def answer_token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def evidence_scores(selected: list[EvidenceItem], gold: list[EvidenceItem]) -> tuple[float, float]:
    if not gold:
        return (1.0, 1.0 if not selected else 0.0)
    matched_gold = [item for item in gold if any(_evidence_match(item, candidate) for candidate in selected)]
    recall = len(matched_gold) / len(gold)
    if not selected:
        precision = 0.0
    else:
        matched_selected = [
            item for item in selected if any(_evidence_match(gold_item, item) for gold_item in gold)
        ]
        precision = len(matched_selected) / len(selected)
    return round(recall, 4), round(precision, 4)


def unsupported_claim_count(answer: str, selected_evidence: list[EvidenceItem]) -> int:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", answer) if item.strip()]
    if not sentences:
        return 0
    evidence_text = " ".join(item.text for item in selected_evidence)
    evidence_tokens = normalized_tokens(evidence_text)
    unsupported = 0
    for sentence in sentences:
        sentence_tokens = normalized_tokens(sentence)
        if not sentence_tokens:
            continue
        if not evidence_tokens:
            unsupported += 1
            continue
        overlap = len(sentence_tokens & evidence_tokens) / len(sentence_tokens)
        if overlap < 0.3:
            unsupported += 1
    return unsupported


def protocol_adherence(output: ResearchOutput) -> ProtocolAdherence:
    used_required_tool = output.used_protocol and bool(output.selected_evidence)
    selected_evidence_present = bool(output.selected_evidence)
    answer_grounded = unsupported_claim_count(output.answer, output.selected_evidence) == 0
    failure_words = ("general ml knowledge", "not in the paper", "i assume")
    avoided_failures = not any(word in output.answer.lower() for word in failure_words)
    checks = [used_required_tool, selected_evidence_present, answer_grounded, avoided_failures]
    return ProtocolAdherence(
        used_required_tool=used_required_tool,
        selected_evidence_present=selected_evidence_present,
        answer_grounded_in_evidence=answer_grounded,
        avoided_known_failure_modes=avoided_failures,
        score=round(sum(1 for item in checks if item) / len(checks), 4),
    )


def aggregate_results(results: list[EvaluationResult]) -> list[AggregateMetrics]:
    aggregates: list[AggregateMetrics] = []
    by_mode: dict[str, list[EvaluationResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)
    for mode, items in sorted(by_mode.items()):
        adherence_scores = [
            item.protocol_adherence.score
            for item in items
            if item.protocol_adherence is not None
        ]
        aggregates.append(
            AggregateMetrics(
                mode=mode,  # type: ignore[arg-type]
                num_examples=len(items),
                avg_answer_token_f1=round(mean(item.answer_token_f1 for item in items), 4),
                avg_evidence_recall=round(mean(item.evidence_recall for item in items), 4),
                avg_evidence_precision=round(mean(item.evidence_precision for item in items), 4),
                avg_unsupported_claim_count=round(mean(item.unsupported_claim_count for item in items), 4),
                avg_protocol_adherence=(
                    round(mean(adherence_scores), 4) if adherence_scores else None
                ),
                avg_answer_length_words=round(mean(item.answer_length_words for item in items), 4),
            )
        )
    return aggregates


def evaluation_to_record(result: EvaluationResult) -> dict[str, Any]:
    return asdict(result)


def _evidence_match(gold: EvidenceItem, selected: EvidenceItem) -> bool:
    gold_text = _normalize_text(gold.text)
    selected_text = _normalize_text(selected.text)
    if not gold_text or not selected_text:
        return False
    if gold_text in selected_text or selected_text in gold_text:
        return True
    gold_tokens = normalized_tokens(gold_text)
    selected_tokens = normalized_tokens(selected_text)
    if not gold_tokens or not selected_tokens:
        return False
    return len(gold_tokens & selected_tokens) / len(gold_tokens) >= 0.5


def _normalize_text(text: str) -> str:
    return " ".join(_tokens(text))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
