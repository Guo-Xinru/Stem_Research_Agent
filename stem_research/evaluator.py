"""Evaluator module with deterministic gold fact coverage checks."""

from __future__ import annotations

import re
from stem_research.schemas import EvaluationResult, GoldFactEvaluation, ResearchOutput


LABEL_SCORES = {
    "addressed": 1.0,
    "partially_addressed": 0.5,
    "not_addressed": 0.0,
}


class Evaluator:
    """Scores outputs against manually curated gold facts using heuristics."""

    def evaluate(
        self,
        question: dict[str, str],
        research_output: ResearchOutput,
        gold_facts: dict,
    ) -> EvaluationResult:
        facts = gold_facts.get("facts", [])
        if not facts:
            raise ValueError(f"No gold facts found for question {question['id']}")

        fact_evaluations = [
            _evaluate_fact(research_output.answer, fact)
            for fact in facts
        ]
        recall = sum(LABEL_SCORES[item.label] for item in fact_evaluations) / len(fact_evaluations)
        unsupported_claim_count = _count_unsupported_claims(research_output)
        citation_notes = _citation_notes(research_output, unsupported_claim_count)
        source_notes = _source_quality_notes(research_output)

        return EvaluationResult(
            question_id=question["id"],
            mode=research_output.mode,
            gold_fact_evaluations=fact_evaluations,
            gold_fact_recall=round(recall, 3),
            unsupported_claim_count=unsupported_claim_count,
            citation_support_notes=citation_notes,
            source_quality_notes=source_notes,
            brief_critique=_brief_critique(research_output.mode, recall, unsupported_claim_count),
        )


def _evaluate_fact(answer: str, fact: dict) -> GoldFactEvaluation:
    keywords = fact.get("keywords") or _keywords_from_text(fact["fact"])
    answer_text = answer.lower()
    matched = [keyword for keyword in keywords if keyword.lower() in answer_text]

    if len(matched) == len(keywords):
        label = "addressed"
    elif matched:
        label = "partially_addressed"
    else:
        label = "not_addressed"

    return GoldFactEvaluation(
        fact_id=fact["id"],
        fact=fact["fact"],
        label=label,
        matched_keywords=matched,
        notes=f"Matched {len(matched)} of {len(keywords)} keywords.",
    )


def _keywords_from_text(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text.lower())
    return sorted(set(words))[:4]


def _count_unsupported_claims(research_output: ResearchOutput) -> int:
    supported_claim_numbers = set()
    for citation in research_output.citations:
        match = re.search(r"claim_(\d+)\s*->", citation)
        if match:
            supported_claim_numbers.add(int(match.group(1)))
    return sum(
        1
        for index, _claim in enumerate(research_output.major_claims, start=1)
        if index not in supported_claim_numbers
    )


def _citation_notes(research_output: ResearchOutput, unsupported_claim_count: int) -> list[str]:
    if not research_output.citations:
        return ["No citation references were provided."]
    notes = [f"{len(research_output.citations)} claim-level citation references provided."]
    if unsupported_claim_count:
        notes.append(f"{unsupported_claim_count} major claim(s) lack claim-level citation references.")
    else:
        notes.append("All major claims include claim-level citation references.")
    return notes


def _source_quality_notes(research_output: ResearchOutput) -> list[str]:
    if all(source.startswith("fixture:") for source in research_output.sources_used):
        return ["Only fixture/mock sources were used; this is acceptable for the smoke test."]
    return ["Non-fixture sources detected; verify source quality manually."]


def _brief_critique(mode: str, recall: float, unsupported_claim_count: int) -> str:
    if recall >= 0.75 and unsupported_claim_count == 0:
        return f"{mode} output covers most gold facts and cites each major claim in this fixture evaluation."
    if recall >= 0.5:
        return f"{mode} output covers some gold facts but still needs stronger support and coverage."
    return f"{mode} output misses many gold facts in this deterministic check."
