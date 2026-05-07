"""Evaluator module with deterministic gold fact coverage checks."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from stem_research.llm import request_strict_json
from stem_research.schemas import EvaluationResult, GoldFactEvaluation, ResearchOutput


LABEL_SCORES = {
    "addressed": 1.0,
    "partially_addressed": 0.5,
    "not_addressed": 0.0,
}

VALID_EVAL_MODES = ("heuristic", "llm")
VALID_LABELS = set(LABEL_SCORES)


class Evaluator:
    """Scores outputs against manually curated gold facts."""

    def __init__(self, eval_mode: str = "heuristic") -> None:
        if eval_mode not in VALID_EVAL_MODES:
            raise ValueError("eval_mode must be either 'heuristic' or 'llm'")
        self.eval_mode = eval_mode

    def evaluate(
        self,
        question: dict[str, str],
        research_output: ResearchOutput,
        gold_facts: dict,
        rubric: dict | None = None,
        source_snippets: list[dict] | None = None,
    ) -> EvaluationResult:
        facts = gold_facts.get("facts", [])
        if not facts:
            raise ValueError(f"No gold facts found for question {question['id']}")

        if self.eval_mode == "llm":
            if rubric is None:
                raise ValueError("rubric is required for llm evaluation mode")
            return _evaluate_with_llm(
                question=question,
                research_output=research_output,
                gold_facts=gold_facts,
                rubric=rubric,
                source_snippets=source_snippets or [],
            )

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
            evaluation_mode="heuristic",
            gold_fact_evaluations=fact_evaluations,
            gold_fact_recall=round(recall, 3),
            unsupported_claim_count=unsupported_claim_count,
            citation_support_notes=citation_notes,
            source_quality_notes=source_notes,
            brief_critique=_brief_critique(research_output.mode, recall, unsupported_claim_count),
        )


def _evaluate_with_llm(
    *,
    question: dict[str, str],
    research_output: ResearchOutput,
    gold_facts: dict,
    rubric: dict,
    source_snippets: list[dict],
) -> EvaluationResult:
    response, _metadata = request_strict_json(
        system_prompt=_llm_evaluator_system_prompt(),
        user_prompt=_llm_evaluator_user_prompt(
            question=question,
            research_output=research_output,
            gold_facts=gold_facts,
            rubric=rubric,
            source_snippets=source_snippets,
        ),
        validate=lambda data: validate_llm_evaluation_response(data, gold_facts),
    )
    fact_by_id = {fact["id"]: fact for fact in gold_facts["facts"]}
    fact_evaluations = [
        GoldFactEvaluation(
            fact_id=item["fact_id"],
            fact=fact_by_id[item["fact_id"]]["fact"],
            label=item["label"],
            matched_keywords=[],
            notes=(
                f"Rationale: {item['rationale']} "
                f"Evidence from answer: {item['evidence_from_answer']}"
            ),
        )
        for item in response["gold_fact_evaluations"]
    ]
    recall = _gold_fact_recall(fact_evaluations)
    return EvaluationResult(
        question_id=question["id"],
        mode=research_output.mode,
        evaluation_mode="llm",
        gold_fact_evaluations=fact_evaluations,
        gold_fact_recall=round(recall, 3),
        unsupported_claim_count=int(response["unsupported_claim_count"]),
        citation_support_notes=_as_notes_list(response["citation_support_notes"]),
        source_quality_notes=_as_notes_list(response["source_quality_notes"]),
        brief_critique=str(response["brief_critique"]).strip(),
    )


def validate_llm_evaluation_response(data: Any, gold_facts: dict) -> dict:
    """Validate strict LLM evaluator JSON before converting to EvaluationResult."""
    if not isinstance(data, dict):
        raise ValueError("LLM evaluator output must be a JSON object")

    required_fields = {
        "gold_fact_evaluations",
        "unsupported_claim_count",
        "citation_support_notes",
        "source_quality_notes",
        "brief_critique",
    }
    missing = sorted(required_fields - set(data))
    if missing:
        raise ValueError(f"LLM evaluator output missing required fields: {', '.join(missing)}")

    evaluations = data["gold_fact_evaluations"]
    if not isinstance(evaluations, list):
        raise ValueError("gold_fact_evaluations must be a list")

    expected_ids = [fact["id"] for fact in gold_facts.get("facts", [])]
    expected_id_set = set(expected_ids)
    returned_ids: list[str] = []

    for item in evaluations:
        if not isinstance(item, dict):
            raise ValueError("Each gold_fact_evaluation must be an object")
        for field in ("fact_id", "label", "rationale", "evidence_from_answer"):
            if field not in item:
                raise ValueError(f"gold_fact_evaluation missing required field: {field}")
        fact_id = item["fact_id"]
        label = item["label"]
        if fact_id not in expected_id_set:
            raise ValueError(f"LLM evaluator returned unknown fact id: {fact_id}")
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid LLM evaluator label for {fact_id}: {label}")
        returned_ids.append(fact_id)

    duplicate_ids = sorted({fact_id for fact_id in returned_ids if returned_ids.count(fact_id) > 1})
    if duplicate_ids:
        raise ValueError(f"LLM evaluator returned duplicate fact ids: {', '.join(duplicate_ids)}")

    missing_ids = sorted(expected_id_set - set(returned_ids))
    if missing_ids:
        raise ValueError(f"LLM evaluator missing fact ids: {', '.join(missing_ids)}")

    if len(returned_ids) != len(expected_ids):
        raise ValueError("LLM evaluator must return exactly one evaluation per gold fact")
    if not isinstance(data["unsupported_claim_count"], int) or data["unsupported_claim_count"] < 0:
        raise ValueError("unsupported_claim_count must be a non-negative integer")
    for field in ("citation_support_notes", "source_quality_notes", "brief_critique"):
        if not str(data[field]).strip():
            raise ValueError(f"{field} must be non-empty")

    return data


def _llm_evaluator_system_prompt() -> str:
    return (
        "You are an evaluation judge for StemResearch. Return only JSON. "
        "Classify semantic coverage of each gold fact in the answer as exactly "
        "addressed, partially_addressed, or not_addressed. Do not assign holistic "
        "1-10 scores. Do not improve, rewrite, or repair the answer. Do not revise "
        "any protocol. Source snippets may inform citation/source notes, but do not "
        "mark a gold fact as addressed merely because it appears in a source snippet."
    )


def _llm_evaluator_user_prompt(
    *,
    question: dict[str, str],
    research_output: ResearchOutput,
    gold_facts: dict,
    rubric: dict,
    source_snippets: list[dict],
) -> str:
    payload = {
        "question_id": question["id"],
        "question_text": question["question"],
        "research_output": asdict(research_output),
        "gold_facts": gold_facts.get("facts", []),
        "rubric": rubric,
        "source_snippets_used": source_snippets,
        "required_output_shape": {
            "gold_fact_evaluations": [
                {
                    "fact_id": "gold fact id",
                    "label": "addressed | partially_addressed | not_addressed",
                    "rationale": "brief reason for the label",
                    "evidence_from_answer": "short answer text or empty string",
                }
            ],
            "unsupported_claim_count": 0,
            "citation_support_notes": "brief citation support notes",
            "source_quality_notes": "brief source quality notes",
            "brief_critique": "brief critique without holistic scores",
        },
    }
    return (
        "Evaluate the answer against the gold facts. Judge gold fact coverage only "
        "from the research_output.answer and major_claims, not from source snippets. "
        "Return strict JSON matching required_output_shape.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
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


def _gold_fact_recall(fact_evaluations: list[GoldFactEvaluation]) -> float:
    return sum(LABEL_SCORES[item.label] for item in fact_evaluations) / len(fact_evaluations)


def _as_notes_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


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
    if all(_is_fixture_source_id(source) for source in research_output.sources_used):
        return ["Only fixture/mock source ids were used; this is acceptable for the smoke test."]
    return ["Non-fixture sources detected; verify source quality manually."]


def _is_fixture_source_id(source_id: str) -> bool:
    return source_id.startswith("fixture:") or source_id.startswith("src_")


def _brief_critique(mode: str, recall: float, unsupported_claim_count: int) -> str:
    if recall >= 0.75 and unsupported_claim_count == 0:
        return f"{mode} output covers most gold facts and cites each major claim in this fixture evaluation."
    if recall >= 0.5:
        return f"{mode} output covers some gold facts but still needs stronger support and coverage."
    return f"{mode} output misses many gold facts in this deterministic check."
