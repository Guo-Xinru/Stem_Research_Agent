"""Run the deterministic StemResearch smoke experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Sequence

from stem_research.evaluator import Evaluator
from stem_research.io_utils import load_json, timestamped_run_path, write_json
from stem_research.researcher import SpecializedResearcher
from stem_research.schemas import ExperimentResult
from stem_research.stem import Stem


TASK_CLASS_DESCRIPTION = (
    "AI engineering research questions about LLM agents, coding agents, "
    "tool use, evaluation, context management, and autonomy."
)


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    questions = load_json(data_dir / "questions.json")
    solved_examples = load_json(data_dir / "solved_examples.json")
    gold_facts = load_json(data_dir / "gold_facts.json")
    rubric = load_json(data_dir / "rubric.json")
    source_snippets = load_json(data_dir / "source_snippets.json")

    selected_questions = _select_questions(questions, args.limit)
    gold_by_question = _gold_by_question_id(gold_facts)

    # generate the stem protocol
    protocol = Stem().generate_protocol(
        task_class_description=TASK_CLASS_DESCRIPTION,
        solved_examples=solved_examples,
        rubric=rubric,
    )

    # create researcher and evaluator instances
    researcher = SpecializedResearcher()
    evaluator = Evaluator()

    per_question = []
    for question in selected_questions:
        question_id = question["id"]
        gold = gold_by_question[question_id]
        selected_snippets = _select_source_snippets(source_snippets, question_id)
        # baseline and specialized answers
        baseline_output = researcher.answer(
            question,
            mode="baseline",
            source_snippets=selected_snippets,
        )
        specialized_output = researcher.answer(
            question,
            mode="specialized",
            source_snippets=selected_snippets,
            protocol=protocol,
        )
        # evaluations against gold facts
        baseline_evaluation = evaluator.evaluate(question, baseline_output, gold)
        specialized_evaluation = evaluator.evaluate(question, specialized_output, gold)

        per_question.append(
            {
                "question": question,
                "gold_review_status": gold.get("review_status", "unknown"),
                "source_snippets_used": selected_snippets,
                "baseline_output": asdict(baseline_output),
                "specialized_output": asdict(specialized_output),
                "baseline_evaluation": asdict(baseline_evaluation),
                "specialized_evaluation": asdict(specialized_evaluation),
            }
        )

    result = ExperimentResult(
        metadata={
            "experiment": "deterministic_smoke_test",
            "limit": len(selected_questions),
            "uses_live_openai_api": False,
            "uses_live_web_search": False,
        },
        generated_protocol=protocol,
        per_question=per_question,
        summary_metrics=_summary_metrics(per_question),
    )

    output_path = timestamped_run_path(output_dir=args.output_dir)
    write_json(output_path, result)
    print(output_path)
    return output_path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the StemResearch smoke experiment.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of questions to run.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Directory for JSON results.")
    return parser.parse_args(argv)


def _select_questions(questions: list[dict], limit: int | None) -> list[dict]:
    if not questions:
        raise ValueError("questions.json must contain at least one question")
    if limit is None:
        return questions
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    return questions[:limit]


def _gold_by_question_id(gold_facts: list[dict]) -> dict[str, dict]:
    indexed = {item["question_id"]: item for item in gold_facts}
    if len(indexed) != len(gold_facts):
        raise ValueError("gold_facts.json contains duplicate question_id values")
    return indexed


def _select_source_snippets(source_snippets: list[dict], question_id: str) -> list[dict]:
    selected = [
        snippet
        for snippet in source_snippets
        if question_id in snippet.get("related_question_ids", [])
    ]
    if not selected:
        raise ValueError(f"No source snippets found for question id: {question_id}")
    return selected


def _summary_metrics(per_question: list[dict]) -> dict:
    baseline_recalls = [
        item["baseline_evaluation"]["gold_fact_recall"]
        for item in per_question
    ]
    specialized_recalls = [
        item["specialized_evaluation"]["gold_fact_recall"]
        for item in per_question
    ]
    baseline_unsupported = [
        item["baseline_evaluation"]["unsupported_claim_count"]
        for item in per_question
    ]
    specialized_unsupported = [
        item["specialized_evaluation"]["unsupported_claim_count"]
        for item in per_question
    ]
    return {
        "average_gold_fact_recall_baseline": round(mean(baseline_recalls), 3),
        "average_gold_fact_recall_specialized": round(mean(specialized_recalls), 3),
        "total_unsupported_claims_baseline": sum(baseline_unsupported),
        "total_unsupported_claims_specialized": sum(specialized_unsupported),
        "comparison_note": "Fixture-only deterministic smoke result; not a final experiment finding.",
    }


if __name__ == "__main__":
    main()
