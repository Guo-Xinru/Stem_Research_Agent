"""Command line interface for the minimal StemResearch experiments."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from stem_research.data import load_qasper_jsonl, require_qasper_files
from stem_research.evaluator import Evaluator, aggregate_results
from stem_research.io_utils import ensure_dir, load_jsonl, write_json, write_jsonl
from stem_research.researcher import SpecializedResearcher
from stem_research.schemas import ResearchMode
from stem_research.stem import Stem


QASPER_TASK_CLASS = (
    "Scientific-paper question answering over a single paper context with "
    "reference answers and evidence annotations available only for solved "
    "examples and evaluator scoring."
)
QASPER_MODES: tuple[ResearchMode, ...] = (
    "baseline_no_tool",
    "baseline_with_tool",
    "specialized_with_protocol_and_tool",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-qasper-mini":
        return _prepare_qasper_mini(args)
    if args.command == "run-qasper":
        return _run_qasper(args)
    if args.command == "evaluate":
        return _evaluate_predictions(args)
    if args.command == "run-ai-demo":
        return _run_ai_demo(args)
    parser.print_help()
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StemResearch CLI")
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-qasper-mini")
    prepare.add_argument("--output-dir", type=Path, default=Path("data/qasper_mini"))
    prepare.add_argument("--train-size", type=int, default=30)
    prepare.add_argument("--eval-size", type=int, default=50)
    prepare.add_argument("--seed", type=int, default=13)

    run = subparsers.add_parser("run-qasper")
    run.add_argument("--data", type=Path, default=Path("data/qasper_mini"))
    run.add_argument("--run-mode", choices=("offline", "llm"), default="offline")
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--top-k", type=int, default=5)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, default=Path("outputs/qasper_metrics.json"))

    demo = subparsers.add_parser("run-ai-demo")
    demo.add_argument("--run-mode", choices=("offline", "llm"), default="offline")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def _run_qasper(args: argparse.Namespace) -> int:
    try:
        train_path, eval_path = require_qasper_files(args.data)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    train_examples = load_qasper_jsonl(train_path)
    eval_examples = load_qasper_jsonl(eval_path)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        eval_examples = eval_examples[: args.limit]
    if not train_examples:
        raise ValueError("QASPER-mini train.jsonl must contain at least one solved example")
    if not eval_examples:
        raise ValueError("QASPER-mini eval.jsonl must contain at least one eval example")

    ensure_dir(args.output_dir)
    stem = Stem()
    protocol = stem.generate_protocol(
        task_class_description=QASPER_TASK_CLASS,
        solved_examples=train_examples,
        run_mode=args.run_mode,
    )
    protocol_path = args.output_dir / "qasper_protocol.json"
    write_json(protocol_path, asdict(protocol))

    researcher = SpecializedResearcher(run_mode=args.run_mode)
    evaluator = Evaluator()
    predictions = []
    evaluations = []
    for example in eval_examples:
        safe_example = example.without_references()
        for mode in QASPER_MODES:
            output = researcher.answer_qasper(
                safe_example,
                mode=mode,
                protocol=protocol if mode == "specialized_with_protocol_and_tool" else None,
                top_k=args.top_k,
            )
            predictions.append(asdict(output))
            evaluations.append(evaluator.evaluate_qasper(example, output))

    predictions_path = args.output_dir / "qasper_predictions.jsonl"
    metrics_path = args.output_dir / "qasper_metrics.json"
    write_jsonl(predictions_path, predictions)
    write_json(
        metrics_path,
        {
            "metadata": {
                "experiment": "qasper_mini",
                "run_mode": args.run_mode,
                "num_train_examples": len(train_examples),
                "num_eval_examples": len(eval_examples),
                "modes": list(QASPER_MODES),
                "protocol_path": str(protocol_path),
            },
            "per_example": [asdict(item) for item in evaluations],
            "aggregate_by_mode": [asdict(item) for item in aggregate_results(evaluations)],
        },
    )
    print(f"Wrote {protocol_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")
    return 0


def _evaluate_predictions(args: argparse.Namespace) -> int:
    eval_examples = {example.id: example for example in load_qasper_jsonl(args.data)}
    predictions = load_jsonl(args.predictions)
    evaluator = Evaluator()
    evaluations = []
    for prediction in predictions:
        example_id = str(prediction.get("id") or "")
        if example_id not in eval_examples:
            raise ValueError(f"Prediction id not found in eval data: {example_id}")
        output = _prediction_to_output(prediction)
        evaluations.append(evaluator.evaluate_qasper(eval_examples[example_id], output))
    write_json(
        args.output,
        {
            "per_example": [asdict(item) for item in evaluations],
            "aggregate_by_mode": [asdict(item) for item in aggregate_results(evaluations)],
        },
    )
    print(f"Wrote {args.output}")
    return 0


def _prediction_to_output(prediction: dict) -> object:
    from stem_research.schemas import EvidenceItem, ResearchOutput

    return ResearchOutput(
        id=str(prediction["id"]),
        mode=prediction["mode"],
        question=str(prediction["question"]),
        answer=str(prediction["answer"]),
        selected_evidence=[
            EvidenceItem(
                section_name=str(item.get("section_name") or ""),
                text=str(item.get("text") or ""),
                score=float(item.get("score", 0.0) or 0.0),
                evidence_id=item.get("evidence_id"),
            )
            for item in prediction.get("selected_evidence", [])
        ],
        used_protocol=bool(prediction.get("used_protocol")),
    )


def _prepare_qasper_mini(args: argparse.Namespace) -> int:
    from stem_research.prepare_qasper_mini import prepare_qasper_mini

    prepare_qasper_mini(
        output_dir=args.output_dir,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    return 0


def _run_ai_demo(args: argparse.Namespace) -> int:
    data_dir = Path("data/ai_engineering")
    train_path = data_dir / "solved_examples.jsonl"
    eval_path = data_dir / "eval.jsonl"
    if not train_path.exists() or not eval_path.exists():
        print(
            "AI-engineering demo JSONL files are missing under data/ai_engineering.",
            file=sys.stderr,
        )
        return 2
    ensure_dir(args.output_dir)
    train_examples = load_qasper_jsonl(train_path)
    eval_examples = load_qasper_jsonl(eval_path)
    protocol = Stem().generate_protocol(
        task_class_description="AI engineering research demo questions over local fixture notes.",
        solved_examples=train_examples,
        run_mode=args.run_mode,
    )
    researcher = SpecializedResearcher(run_mode=args.run_mode)
    evaluator = Evaluator()
    predictions = []
    evaluations = []
    for example in eval_examples:
        safe_example = example.without_references()
        output = researcher.answer_qasper(
            safe_example,
            mode="specialized_with_protocol_and_tool",
            protocol=protocol,
        )
        predictions.append(asdict(output))
        evaluations.append(evaluator.evaluate_qasper(example, output))
    predictions_path = args.output_dir / "ai_engineering_demo_predictions.jsonl"
    metrics_path = args.output_dir / "ai_engineering_demo_metrics.json"
    write_jsonl(predictions_path, predictions)
    write_json(
        metrics_path,
        {
            "metadata": {
                "experiment": "ai_engineering_demo",
                "run_mode": args.run_mode,
                "note": "The AI-engineering mini-set is retained as a domain demo; QASPER-mini is the main experiment.",
            },
            "per_example": [asdict(item) for item in evaluations],
            "aggregate_by_mode": [asdict(item) for item in aggregate_results(evaluations)],
        },
    )
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
