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
    debug_schema: bool = False,
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
    if debug_schema:
        _print_schema_debug(dataset["train"][0])

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
    skipped_malformed = 0
    skipped_unusable = 0
    for paper_index, paper in enumerate(split):
        if not isinstance(paper, dict):
            skipped_malformed += 1
            continue
        context = _paper_context(paper)
        if not context["sections"]:
            skipped_unusable += 1
            continue
        qas = paper.get("qas") or paper.get("questions") or []
        for qa_index, qa in enumerate(_iter_qas(qas)):
            record = _record_from_qa(
                qa,
                paper=paper,
                context=context,
                fallback_id=f"qasper_{split_name}_{paper_index:05d}_{qa_index:03d}",
            )
            if record is not None:
                records.append(record)
            else:
                skipped_unusable += 1
        if not qas:
            skipped_malformed += 1
    print(
        f"QASPER {split_name}: collected={len(records)} "
        f"skipped_malformed={skipped_malformed} skipped_unusable={skipped_unusable}"
    )
    return records


def _paper_context(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_title": str(paper.get("title") or paper.get("paper_title") or ""),
        "abstract": _join_paragraphs(paper.get("abstract") or ""),
        "sections": _extract_sections(paper),
    }


def _extract_sections(paper: dict[str, Any]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    full_text = paper.get("full_text") or {}
    if not isinstance(full_text, dict):
        return sections
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
    return sections


def _record_from_qa(
    qa: dict[str, Any],
    *,
    paper: dict[str, Any],
    context: dict[str, Any],
    fallback_id: str,
) -> dict[str, Any] | None:
    if not isinstance(qa, dict):
        return None
    question = _extract_question(qa)
    if not question:
        return None
    answers = _extract_answers(qa)
    if not answers:
        return None
    answer = _first_usable_answer(answers)
    if answer is None:
        return None
    evidence_texts = _extract_evidence(answer, paper)
    if not evidence_texts:
        return None
    if any(_looks_table_or_figure(text) for text in evidence_texts):
        return None
    return {
        "id": _extract_question_id(qa, fallback_id),
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


def _iter_qas(qas: Any) -> list[dict[str, Any]]:
    """Normalize QASPER QA containers into per-question dictionaries."""
    if isinstance(qas, list):
        return [item for item in qas if isinstance(item, dict)]
    if not isinstance(qas, dict):
        return []

    list_fields = {key: value for key, value in qas.items() if isinstance(value, list)}
    if not list_fields:
        return [qas]

    length = max((len(value) for value in list_fields.values()), default=0)
    normalized: list[dict[str, Any]] = []
    for index in range(length):
        item: dict[str, Any] = {}
        for key, value in qas.items():
            if isinstance(value, list):
                item[key] = value[index] if index < len(value) else None
            else:
                item[key] = value
        normalized.append(item)
    return normalized


def _extract_question(qa: dict[str, Any]) -> str:
    return _clean_text(qa.get("question") or qa.get("query") or "")


def _extract_question_id(qa: dict[str, Any], fallback_id: str) -> str:
    return _clean_text(qa.get("question_id") or qa.get("id") or fallback_id)


def _extract_answers(qa: dict[str, Any]) -> list[dict[str, Any]]:
    raw_answers = qa.get("answers")
    if raw_answers is None:
        raw_answers = qa.get("answer")

    if isinstance(raw_answers, list):
        answers: list[dict[str, Any]] = []
        for item in raw_answers:
            if isinstance(item, dict) and isinstance(item.get("answer"), list):
                answers.extend(answer for answer in item["answer"] if isinstance(answer, dict))
            elif isinstance(item, dict):
                answers.append(item)
        return answers

    if isinstance(raw_answers, dict):
        nested_answers = raw_answers.get("answer")
        if isinstance(nested_answers, list):
            return [item for item in nested_answers if isinstance(item, dict)]
        if isinstance(nested_answers, dict):
            return [nested_answers]
        return [raw_answers]

    return []


def _extract_evidence(answer: dict[str, Any], paper: dict[str, Any]) -> list[str]:
    del paper
    evidence = answer.get("evidence") or answer.get("highlighted_evidence") or []
    if isinstance(evidence, str):
        evidence_items = [evidence]
    elif isinstance(evidence, list):
        evidence_items = evidence
    else:
        evidence_items = []
    return [_clean_text(text) for text in evidence_items if _clean_text(text)]


def _first_usable_answer(answers: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in answers:
        if not isinstance(item, dict):
            continue
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
        evidence = answer.get("evidence") or answer.get("highlighted_evidence") or []
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


def _print_schema_debug(example: dict[str, Any]) -> None:
    print("QASPER schema debug:")
    print(f"top_level_keys={list(example.keys())}")
    for key in ("id", "title", "abstract", "full_text", "qas", "questions", "answers"):
        if key in example:
            value = example[key]
            print(f"{key}: type={type(value).__name__}")
    qas = example.get("qas") or example.get("questions")
    print(f"qa_field_type={type(qas).__name__}")
    if isinstance(qas, dict):
        print(f"qa_field_keys={list(qas.keys())}")
        compact = {
            key: _debug_preview(value[0] if isinstance(value, list) and value else value)
            for key, value in qas.items()
            if key in {"question", "question_id", "answers", "answer"}
        }
        print(f"qa_sample={compact}")
    elif isinstance(qas, list) and qas:
        print(f"qa_sample={_debug_preview(qas[0])}")


def _debug_preview(value: Any, max_chars: int = 500) -> str:
    text = repr(value)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text
