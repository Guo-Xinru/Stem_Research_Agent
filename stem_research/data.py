"""Load and normalize small JSONL datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from stem_research.io_utils import load_jsonl
from stem_research.schemas import EvidenceItem, PaperContext, PaperSection, QasperExample


def load_qasper_jsonl(path: str | Path) -> list[QasperExample]:
    return [normalize_qasper_record(record, source=Path(path)) for record in load_jsonl(path)]


def normalize_qasper_record(record: dict[str, Any], *, source: Path | None = None) -> QasperExample:
    """Normalize the repo JSONL shape and a few older fixture-style variants."""
    source_label = str(source) if source else "record"
    try:
        context = record["context"]
        raw_sections = context.get("sections", [])
        sections = [
            PaperSection(
                section_name=str(section.get("section_name") or section.get("name") or "Unknown"),
                text=str(section.get("text") or "").strip(),
            )
            for section in raw_sections
            if str(section.get("text") or "").strip()
        ]
        paper_context = PaperContext(
            paper_title=str(context.get("paper_title") or context.get("title") or "").strip(),
            abstract=str(context.get("abstract") or "").strip(),
            sections=sections,
        )
    except KeyError as exc:
        raise ValueError(f"{source_label} is missing QASPER context: {exc}") from exc

    evidence = [
        EvidenceItem(
            section_name=str(item.get("section_name") or "Unknown"),
            text=str(item.get("text") or "").strip(),
            score=float(item.get("score", 0.0) or 0.0),
            evidence_id=item.get("evidence_id"),
        )
        for item in record.get("evidence", [])
        if str(item.get("text") or "").strip()
    ]

    example_id = str(record.get("id") or record.get("question_id") or "").strip()
    question = str(record.get("question") or "").strip()
    if not example_id or not question:
        raise ValueError(f"{source_label} record requires non-empty id and question")

    return QasperExample(
        id=example_id,
        domain=str(record.get("domain") or "scientific_paper_qa"),
        question=question,
        context=paper_context,
        reference_answer=(
            str(record["reference_answer"]).strip()
            if record.get("reference_answer") is not None
            else None
        ),
        evidence=evidence,
        answer_type=str(record.get("answer_type") or "abstractive"),
    )


def qasper_to_record(example: QasperExample) -> dict[str, Any]:
    return asdict(example)


def require_qasper_files(data_dir: str | Path) -> tuple[Path, Path]:
    data_path = Path(data_dir)
    train_path = data_path / "train.jsonl"
    eval_path = data_path / "eval.jsonl"
    missing = [path for path in (train_path, eval_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "QASPER-mini local JSONL files are missing: "
            f"{missing_text}. Run `uv run python scripts/prepare_qasper_mini.py` "
            "or `uv run python -m stemresearch.cli prepare-qasper-mini`."
        )
    return train_path, eval_path
