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
BASELINE_TOOL_TOP_K = 3
GENERIC_SNIPPET_TOKENS = {
    "approach",
    "contribution",
    "experiment",
    "experiments",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "propose",
    "proposed",
    "result",
    "results",
    "show",
    "shows",
    "study",
    "task",
    "tasks",
    "use",
    "used",
    "using",
}


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
        if mode == "baseline_with_tool":
            selected_evidence = self.retriever.retrieve(
                safe_example.question,
                safe_example.context.sections,
                top_k=BASELINE_TOOL_TOP_K,
            )
        elif mode == "specialized_with_protocol_and_tool":
            evidence_selection = protocol.evidence_selection if protocol else {}
            raw_top_k = _int_policy_value(evidence_selection, "top_k_raw", default=max(8, top_k))
            candidate_evidence = self.retriever.retrieve(
                safe_example.question,
                safe_example.context.sections,
                top_k=raw_top_k,
            )
            selected_evidence = _select_evidence_with_protocol(
                safe_example.question,
                candidate_evidence,
                protocol,
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


def _select_evidence_with_protocol(
    question: str,
    candidate_evidence: list[EvidenceItem],
    protocol: ResearchProtocol | None,
) -> list[EvidenceItem]:
    """Apply deterministic protocol-guided evidence selection."""
    if not candidate_evidence:
        return []
    evidence_selection = protocol.evidence_selection if protocol else {}
    top_k_final = _int_policy_value(evidence_selection, "top_k_final", default=3)
    min_overlap = _int_policy_value(evidence_selection, "min_question_token_overlap", default=2)
    prefer_sections = [
        str(item).strip().lower()
        for item in evidence_selection.get("prefer_sections", [])
        if str(item).strip()
    ]
    discard_generic = bool(evidence_selection.get("discard_generic_snippets", False))
    question_tokens = normalized_tokens(question)

    scored: list[tuple[float, int, int, EvidenceItem]] = []
    for index, item in enumerate(candidate_evidence):
        evidence_tokens = normalized_tokens(item.text)
        overlap = len(question_tokens & evidence_tokens)
        section_label = f"{item.section_name} {item.evidence_id or ''}".lower()
        section_bonus = 1.5 if any(section in section_label for section in prefer_sections) else 0.0
        generic_penalty = 2.0 if discard_generic and _is_generic_snippet(evidence_tokens) else 0.0
        score = (overlap * 2.0) + section_bonus + (item.score * 0.1) - generic_penalty
        scored.append((score, overlap, index, item))

    filtered = [item for item in scored if item[1] >= min_overlap]
    if not filtered:
        filtered = scored
    filtered.sort(key=lambda item: (-item[0], item[2]))
    return [item[3] for item in filtered[:top_k_final]]


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
        return _insufficient_evidence_answer(protocol)

    evidence_text = " ".join(item.text for item in selected_evidence[:2])
    answer = _first_sentences(evidence_text, max_sentences=2)
    if mode == "baseline_with_tool":
        return answer or "The retrieved evidence does not clearly answer the question."

    answer_policy = protocol.answer_policy if protocol else {}
    if (
        protocol
        and answer_policy.get("require_evidence_grounding", True)
        and _weakly_grounded(answer, selected_evidence)
    ):
        if answer_policy.get("avoid_unverifiable_claims", True):
            answer = _insufficient_evidence_answer(protocol)
        else:
            answer += " The evidence is weak, so this answer should be treated as uncertain."
    answer = answer or _insufficient_evidence_answer(protocol)
    return _apply_answer_policy(answer, protocol)


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


def _is_generic_snippet(tokens: set[str]) -> bool:
    if not tokens:
        return True
    specific_tokens = tokens - GENERIC_SNIPPET_TOKENS
    return len(specific_tokens) <= 2


def _int_policy_value(policy: dict[str, Any], key: str, *, default: int) -> int:
    try:
        value = int(policy.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _insufficient_evidence_answer(protocol: ResearchProtocol | None) -> str:
    answer_policy = protocol.answer_policy if protocol else {}
    if answer_policy.get("allow_insufficient_evidence_answer", True):
        return "The selected paper evidence is insufficient for a confident answer."
    return "The retrieved evidence does not clearly answer the question."


def _apply_answer_policy(answer: str, protocol: ResearchProtocol | None) -> str:
    if not protocol:
        return answer
    max_words = _int_policy_value(protocol.answer_policy, "max_words", default=80)
    words = answer.split()
    if len(words) <= max_words:
        return answer
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


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
