"""SpecializedResearcher with three QASPER-mini comparison modes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from stem_research.llm import request_strict_json
from stem_research.retriever import EvidenceRetrieverTool, normalized_tokens
from stem_research.schemas import EvidenceItem, QasperExample, ResearchMode, ResearchOutput, ResearchProtocol, RunMode


VALID_MODES: tuple[ResearchMode, ...] = (
    "baseline_no_tool",
    "baseline_with_tool",
    "specialized_with_protocol_and_tool",
)


class SpecializedResearcher:
    """Answers paper QA questions in baseline and protocol-specialized modes."""

    def __init__(
        self,
        run_mode: RunMode | str = "offline",
        retriever: EvidenceRetrieverTool | None = None,
        researcher_mode: str | None = None,
    ) -> None:
        if researcher_mode is not None:
            run_mode = "llm" if researcher_mode == "live" else "offline"
        if run_mode not in ("offline", "llm"):
            raise ValueError("run_mode must be either 'offline' or 'llm'")
        self.run_mode = run_mode
        self.retriever = retriever or EvidenceRetrieverTool()
        self.last_metadata: dict[str, Any] = {
            "generated_by": run_mode,
            "model": None,
            "api_error": None,
        }

    def answer_qasper(
        self,
        example: QasperExample,
        mode: ResearchMode,
        protocol: ResearchProtocol | None = None,
        top_k: int = 5,
    ) -> ResearchOutput:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of: {', '.join(VALID_MODES)}")
        if mode == "specialized_with_protocol_and_tool" and protocol is None:
            raise ValueError("specialized_with_protocol_and_tool requires a protocol")

        safe_example = example.without_references()
        selected_evidence: list[EvidenceItem] = []
        if mode in ("baseline_with_tool", "specialized_with_protocol_and_tool"):
            selected_evidence = self.retriever.retrieve(
                safe_example.question,
                safe_example.context.sections,
                top_k=top_k,
            )

        if self.run_mode == "llm":
            output, metadata = request_strict_json(
                system_prompt=_system_prompt(mode),
                user_prompt=_user_prompt(
                    example=safe_example,
                    mode=mode,
                    selected_evidence=selected_evidence,
                    protocol=protocol,
                ),
                validate=lambda data: validate_research_output(
                    data,
                    example=safe_example,
                    mode=mode,
                    selected_evidence=selected_evidence,
                    used_protocol=mode == "specialized_with_protocol_and_tool",
                ),
                validation_label="researcher output",
            )
            self.last_metadata = {"generated_by": "openai", "model": metadata["model"], "api_error": None}
            return output

        answer = _offline_answer(
            example=safe_example,
            mode=mode,
            selected_evidence=selected_evidence,
            protocol=protocol,
        )
        return ResearchOutput(
            id=safe_example.id,
            mode=mode,
            question=safe_example.question,
            answer=answer,
            selected_evidence=selected_evidence,
            used_protocol=mode == "specialized_with_protocol_and_tool",
        )


def validate_research_output(
    data: Any,
    *,
    example: QasperExample,
    mode: ResearchMode,
    selected_evidence: list[EvidenceItem],
    used_protocol: bool,
) -> ResearchOutput:
    if not isinstance(data, dict):
        raise ValueError("researcher output must be a JSON object")
    answer = str(data.get("answer") or "").strip()
    if not answer:
        raise ValueError("researcher output requires a non-empty answer")
    raw_evidence = data.get("selected_evidence", None)
    if raw_evidence is None:
        evidence = selected_evidence
    else:
        if not isinstance(raw_evidence, list):
            raise ValueError("selected_evidence must be a list")
        evidence = [
            EvidenceItem(
                section_name=str(item.get("section_name") or "").strip(),
                text=str(item.get("text") or "").strip(),
                score=float(item.get("score", 0.0) or 0.0),
                evidence_id=item.get("evidence_id"),
            )
            for item in raw_evidence
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
    if mode == "baseline_no_tool" and evidence:
        raise ValueError("baseline_no_tool must not return selected evidence")
    if mode != "baseline_no_tool" and not evidence:
        raise ValueError(f"{mode} must return selected evidence")
    return ResearchOutput(
        id=example.id,
        mode=mode,
        question=example.question,
        answer=answer,
        selected_evidence=evidence,
        used_protocol=used_protocol,
    )


def _offline_answer(
    *,
    example: QasperExample,
    mode: ResearchMode,
    selected_evidence: list[EvidenceItem],
    protocol: ResearchProtocol | None,
) -> str:
    if mode == "baseline_no_tool":
        source_text = _best_context_text(example)
        return _first_sentences(source_text, max_sentences=2) or "The paper context does not provide enough information to answer deterministically."

    if not selected_evidence:
        return "The retrieved paper evidence is insufficient to answer the question."

    evidence_text = " ".join(item.text for item in selected_evidence[:2])
    answer = _first_sentences(evidence_text, max_sentences=2)
    if mode == "baseline_with_tool":
        return answer or "The retrieved evidence does not clearly answer the question."

    verification_note = ""
    if protocol and _weakly_grounded(answer, selected_evidence):
        verification_note = " The evidence is weak, so this answer should be treated as uncertain."
    return (answer or "The selected evidence is insufficient for a confident answer.") + verification_note


def _best_context_text(example: QasperExample) -> str:
    question_tokens = normalized_tokens(example.question)
    candidates = [example.context.abstract, *[section.text for section in example.context.sections]]
    best = ""
    best_score = -1
    for candidate in candidates:
        score = len(question_tokens & normalized_tokens(candidate))
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _first_sentences(text: str, max_sentences: int) -> str:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item.strip()]
    return " ".join(sentences[:max_sentences]).strip()


def _weakly_grounded(answer: str, evidence: list[EvidenceItem]) -> bool:
    evidence_tokens = normalized_tokens(" ".join(item.text for item in evidence))
    answer_tokens = normalized_tokens(answer)
    if not answer_tokens:
        return True
    return len(answer_tokens & evidence_tokens) / len(answer_tokens) < 0.35


def _system_prompt(mode: ResearchMode) -> str:
    if mode == "specialized_with_protocol_and_tool":
        return (
            "You are the protocol-specialized researcher in StemResearch. "
            "Answer only from the provided paper evidence and protocol. Return JSON only."
        )
    return (
        "You are a generic paper QA researcher in StemResearch. "
        "Answer only from the provided paper context and evidence. Return JSON only."
    )


def _user_prompt(
    *,
    example: QasperExample,
    mode: ResearchMode,
    selected_evidence: list[EvidenceItem],
    protocol: ResearchProtocol | None,
) -> str:
    payload: dict[str, Any] = {
        "id": example.id,
        "mode": mode,
        "question": example.question,
        "paper_context": asdict(example.context),
        "selected_evidence": [asdict(item) for item in selected_evidence],
        "required_output_shape": {
            "answer": "concise answer string",
            "selected_evidence": [
                {"section_name": "section", "text": "evidence text", "score": 0.0}
            ],
        },
    }
    if mode == "baseline_no_tool":
        payload["instructions"] = [
            "Do not use or invent selected evidence.",
            "Use a generic concise paper QA style.",
        ]
    elif mode == "baseline_with_tool":
        payload["instructions"] = [
            "Use the selected evidence, but do not use a Stem protocol.",
            "If evidence is insufficient, say so.",
        ]
    else:
        payload["protocol"] = asdict(protocol) if protocol else None
        payload["instructions"] = [
            "Follow the protocol answer rules, tool policy, verification rules, and failure modes.",
            "If evidence is insufficient, say so instead of using general ML knowledge.",
        ]
    return (
        "Answer the paper question. Do not use reference answers or gold evidence. "
        "Return strict JSON only.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )
