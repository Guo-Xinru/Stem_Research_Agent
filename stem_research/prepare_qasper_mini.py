"""Prepare a deterministic local QASPER-mini subset from HuggingFace datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from stem_research.io_utils import write_jsonl


def prepare_qasper_mini(
    *,
    output_dir: Path,
    train_size: int = 30,
    eval_size: int = 50,
    seed: int = 13,
) -> tuple[Path, Path]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required to prepare QASPER-mini. "
            "Install it with `uv add datasets` or provide local JSONL files at "
            "data/qasper_mini/train.jsonl and data/qasper_mini/eval.jsonl."
        ) from exc

    dataset = load_dataset("allenai/qasper")
    train_records = _collect_records(dataset["train"], split_name="train")
    validation_split = dataset["validation"] if "validation" in dataset else dataset["train"]
    eval_records = _collect_records(validation_split, split_name="validation")

    rng = random.Random(seed)
    rng.shuffle(train_records)
    rng.shuffle(eval_records)

    if len(train_records) < train_size:
        raise SystemExit(f"Only found {len(train_records)} usable train examples; need {train_size}.")
    if len(eval_records) < eval_size:
        raise SystemExit(f"Only found {len(eval_records)} usable eval examples; need {eval_size}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(train_path, train_records[:train_size])
    write_jsonl(eval_path, eval_records[:eval_size])
    print(f"Wrote {train_path}")
    print(f"Wrote {eval_path}")
    return train_path, eval_path


def _collect_records(split: Any, *, split_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for paper_index, paper in enumerate(split):
        context = _paper_context(paper)
        if not context["sections"]:
            continue
        qas = paper.get("qas") or paper.get("questions") or []
        for qa_index, qa in enumerate(qas):
            record = _record_from_qa(
                qa,
                context=context,
                fallback_id=f"qasper_{split_name}_{paper_index:05d}_{qa_index:03d}",
            )
            if record is not None:
                records.append(record)
    return records


def _paper_context(paper: dict[str, Any]) -> dict[str, Any]:
    sections: list[dict[str, str]] = []
    full_text = paper.get("full_text") or {}
    raw_sections = full_text.get("section_name") or full_text.get("sections") or []
    raw_paragraphs = full_text.get("paragraphs") or []
    if raw_paragraphs and raw_sections and len(raw_sections) == len(raw_paragraphs):
        for section_name, paragraphs in zip(raw_sections, raw_paragraphs):
            text = _join_paragraphs(paragraphs)
            if _usable_section(text):
                sections.append({"section_name": str(section_name), "text": text})
    elif isinstance(raw_sections, list):
        for section in raw_sections:
            if isinstance(section, dict):
                text = _join_paragraphs(section.get("paragraphs") or section.get("text") or "")
                if _usable_section(text):
                    sections.append(
                        {
                            "section_name": str(section.get("section_name") or section.get("name") or "Unknown"),
                            "text": text,
                        }
                    )
    return {
        "paper_title": str(paper.get("title") or paper.get("paper_title") or ""),
        "abstract": _join_paragraphs(paper.get("abstract") or ""),
        "sections": sections,
    }


def _record_from_qa(
    qa: dict[str, Any],
    *,
    context: dict[str, Any],
    fallback_id: str,
) -> dict[str, Any] | None:
    question = str(qa.get("question") or "").strip()
    if not question:
        return None
    answers = qa.get("answers") or []
    if not answers:
        return None
    answer = _first_usable_answer(answers)
    if answer is None:
        return None
    evidence_texts = [_clean_text(text) for text in answer.get("evidence", []) if _clean_text(text)]
    if not evidence_texts:
        return None
    if any(_looks_table_or_figure(text) for text in evidence_texts):
        return None
    return {
        "id": str(qa.get("question_id") or fallback_id),
        "domain": "scientific_paper_qa",
        "question": question,
        "context": context,
        "reference_answer": answer["reference_answer"],
        "evidence": [
            {
                "section_name": _section_for_evidence(text, context["sections"]),
                "text": text,
            }
            for text in evidence_texts
        ],
        "answer_type": answer["answer_type"],
    }


def _first_usable_answer(answers: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in answers:
        answer = item.get("answer") if isinstance(item.get("answer"), dict) else item
        if answer.get("unanswerable"):
            continue
        free_form = _clean_text(answer.get("free_form_answer") or "")
        extractive = [_clean_text(text) for text in answer.get("extractive_spans", []) if _clean_text(text)]
        yes_no = answer.get("yes_no")
        if free_form:
            reference = free_form
            answer_type = "abstractive"
        elif extractive:
            reference = " ".join(extractive)
            answer_type = "extractive"
        elif yes_no is not None:
            reference = "yes" if yes_no else "no"
            answer_type = "boolean"
        else:
            continue
        evidence = answer.get("evidence") or []
        if reference and evidence:
            return {"reference_answer": reference, "evidence": evidence, "answer_type": answer_type}
    return None


def _section_for_evidence(evidence_text: str, sections: list[dict[str, str]]) -> str:
    normalized = _clean_text(evidence_text).lower()
    for section in sections:
        if normalized and normalized in section["text"].lower():
            return section["section_name"]
    return "Unknown"


def _join_paragraphs(value: Any) -> str:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, list):
                parts.append(_join_paragraphs(item))
            else:
                parts.append(str(item))
        return _clean_text(" ".join(parts))
    return _clean_text(str(value or ""))


def _clean_text(text: Any) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def _usable_section(text: str) -> bool:
    return len(text.split()) >= 20 and not _looks_table_or_figure(text)


def _looks_table_or_figure(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith(("table ", "figure ", "fig. ")) or lowered.count("|") >= 3
