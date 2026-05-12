"""SpecializedResearcher module with fixture and optional live OpenAI behavior."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from stem_research.llm import request_strict_json
from stem_research.schemas import ResearchMode, ResearchOutput, ResearchProtocol


FIXTURE_SOURCES = {
    "fixture:agent_failure_notes": "Notes on coding-agent failures: long tasks drift, hidden state accumulates, tests are skipped, and tool errors compound.",
    "fixture:context_management_notes": "Notes on context management: agents need summarization, retrieval discipline, explicit state tracking, and guardrails against stale context.",
    "fixture:tool_use_notes": "Notes on tool use: autonomous agents fail when tool schemas are brittle, errors are not checked, or execution feedback is ignored.",
}

VALID_RESEARCHER_MODES = ("fixture", "live")
RESEARCHER_OUTPUT_FIELDS = {
    "answer",
    "major_claims",
    "citations",
    "sources_used",
    "uncertainty_notes",
}
EXTERNAL_REFERENCE_PATTERN = re.compile(
    r"https?://|www\.|doi:|arxiv:",
    re.IGNORECASE,
)
CITATION_PATTERN = re.compile(r"^claim_(\d+)\s*->\s*(\S+)$")


class SpecializedResearcher:
    """Answers in baseline or specialized mode without live search."""

    def __init__(self, researcher_mode: str = "fixture") -> None:
        if researcher_mode not in VALID_RESEARCHER_MODES:
            raise ValueError("researcher_mode must be either 'fixture' or 'live'")
        self.researcher_mode = researcher_mode
        self.last_metadata: dict[str, Any] = {
            "generated_by": "fixture",
            "model": None,
            "api_error": None,
        }

    def answer(
        self,
        question: dict[str, str],
        mode: ResearchMode,
        source_snippets: list[dict] | None = None,
        protocol: ResearchProtocol | None = None,
    ) -> ResearchOutput:
        if mode not in ("baseline", "specialized"):
            raise ValueError("mode must be either 'baseline' or 'specialized'")
        if mode == "specialized" and protocol is None:
            raise ValueError("specialized mode requires a protocol")

        question_id = question["id"]
        question_text = question["question"]
        source_ids = _source_ids(source_snippets, question_text)

        if self.researcher_mode == "live":
            try:
                output, metadata = request_strict_json(
                    system_prompt=_researcher_system_prompt(mode),
                    user_prompt=_researcher_user_prompt(
                        question=question,
                        mode=mode,
                        source_snippets=source_snippets or [],
                        source_ids=source_ids,
                        protocol=protocol,
                    ),
                    validate=lambda data: validate_research_output(
                        data,
                        question_id=question_id,
                        mode=mode,
                        question=question_text,
                        allowed_source_ids=source_ids,
                    ),
                    validation_label="researcher output",
                )
            except Exception as exc:
                self.last_metadata = {
                    "generated_by": "openai",
                    "model": self.last_metadata.get("model"),
                    "api_error": f"{exc.__class__.__name__}: {exc}",
                }
                raise
            self.last_metadata = {
                "generated_by": "openai",
                "model": metadata["model"],
                "api_error": None,
            }
            return output

        if mode == "baseline":
            return _baseline_answer(question_id, question_text, source_ids)
        return _specialized_answer(question_id, question_text, source_ids, protocol)


def _select_fixture_source(question: str) -> str:
    normalized = question.lower()
    if "context" in normalized:
        return "fixture:context_management_notes"
    if "tool" in normalized:
        return "fixture:tool_use_notes"
    return "fixture:agent_failure_notes"


def _source_ids(source_snippets: list[dict] | None, question: str) -> list[str]:
    if source_snippets:
        return [snippet["id"] for snippet in source_snippets]
    return [_select_fixture_source(question)]


def _baseline_answer(question_id: str, question: str, source_ids: list[str]) -> ResearchOutput:
    claims = [
        "Long-horizon agents often degrade when intermediate assumptions are not checked.",
        "A useful baseline mitigation is to keep outputs structured and run tests frequently.",
    ]
    answer = (
        f"Baseline answer for {question_id}: {question} The main issue is reliability over a long task. "
        "Agents can drift from the original goal, miss failed tool calls, and produce unsupported claims. "
        "A practical response is to keep steps short, inspect intermediate outputs, and run tests."
    )
    return ResearchOutput(
        question_id=question_id,
        mode="baseline",
        question=question,
        answer=answer,
        major_claims=claims,
        citations=[f"claim_1 -> {source_ids[0]}"],
        sources_used=source_ids,
        uncertainty_notes=[
            "This is fixture-backed placeholder behavior, not live research.",
            "The answer is intentionally generic for baseline comparison.",
        ],
    )


def _specialized_answer(
    question_id: str,
    question: str,
    source_ids: list[str],
    protocol: ResearchProtocol | None,
) -> ResearchOutput:
    if protocol is None:
        raise ValueError("protocol is required")

    claims = [
        "The failure mode should be separated into planning drift, context loss, and tool-feedback errors.",
        "Reliable answers need explicit claim checks against cited or fixture sources.",
        "Stopping criteria reduce overextension when only weak evidence is available.",
    ]
    answer = (
        f"Specialized answer for {question_id}: {question} Short answer: the risk is not just that an "
        "agent makes one mistake, but that planning drift, stale context, brittle tool use, and missed "
        "verification compound over many steps. Following the generated protocol, the answer should name "
        "the failure category, cite fixture evidence, check each major claim, and mark uncertainty where "
        "the fixture does not support a stronger conclusion. Practical mitigations include explicit state "
        "tracking, frequent test or tool-result checks, conservative stopping criteria, and citations for "
        "major factual claims."
    )
    return ResearchOutput(
        question_id=question_id,
        mode="specialized",
        question=question,
        answer=answer,
        major_claims=claims,
        citations=[
            f"claim_1 -> {source_ids[0]}",
            f"claim_2 -> {source_ids[min(1, len(source_ids) - 1)]}",
            f"claim_3 -> {source_ids[min(2, len(source_ids) - 1)]}",
        ],
        sources_used=source_ids,
        uncertainty_notes=[
            "This uses a deterministic protocol and fixture source, not live retrieval.",
            "Claims should be treated as starter hypotheses until manually reviewed.",
        ],
    )


def validate_research_output(
    data: Any,
    *,
    question_id: str,
    mode: ResearchMode,
    question: str,
    allowed_source_ids: list[str],
) -> ResearchOutput:
    """Validate live researcher JSON and convert it to ResearchOutput."""
    if not isinstance(data, dict):
        raise ValueError("researcher output must be a JSON object")
    missing = sorted(RESEARCHER_OUTPUT_FIELDS - set(data))
    if missing:
        raise ValueError(f"researcher output missing required fields: {', '.join(missing)}")

    answer = _required_string(data["answer"], "answer")
    major_claims = _required_string_list(data["major_claims"], "major_claims")
    citations = _string_list(data["citations"], "citations")
    sources_used = _string_list(data["sources_used"], "sources_used")
    uncertainty_notes = _string_list(data["uncertainty_notes"], "uncertainty_notes")

    allowed = set(allowed_source_ids)
    unknown_sources = sorted(set(sources_used) - allowed)
    if unknown_sources:
        raise ValueError(f"sources_used contains unknown source ids: {', '.join(unknown_sources)}")

    for citation in citations:
        match = CITATION_PATTERN.match(citation)
        if not match:
            raise ValueError(f"citation must use 'claim_N -> source_id' format: {citation}")
        claim_number = int(match.group(1))
        source_id = match.group(2)
        if claim_number < 1 or claim_number > len(major_claims):
            raise ValueError(f"citation references missing major claim: {citation}")
        if source_id not in allowed:
            raise ValueError(f"citation references unknown source id: {source_id}")

    _reject_external_references(
        [answer, *major_claims, *citations, *sources_used, *uncertainty_notes]
    )

    return ResearchOutput(
        question_id=question_id,
        mode=mode,
        question=question,
        answer=answer,
        major_claims=major_claims,
        citations=citations,
        sources_used=sources_used,
        uncertainty_notes=uncertainty_notes,
    )


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_string_list(value: Any, field: str) -> list[str]:
    items = _string_list(value, field)
    if not items:
        raise ValueError(f"{field} must be a non-empty list of strings")
    return items


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must contain only strings")
    return [item.strip() for item in value if item.strip()]


def _reject_external_references(values: list[str]) -> None:
    for value in values:
        if EXTERNAL_REFERENCE_PATTERN.search(value):
            raise ValueError("researcher output must not include URLs or external citations")


def _researcher_system_prompt(mode: ResearchMode) -> str:
    if mode == "baseline":
        return (
            "You are the generic baseline researcher in a minimal StemResearch experiment. "
            "Answer only from the provided fixture source snippets. Return only JSON."
        )
    return (
        "You are the Stem-specialized researcher in a minimal StemResearch experiment. "
        "Use the provided ResearchProtocol and fixture source snippets. Return only JSON."
    )


def _researcher_user_prompt(
    *,
    question: dict[str, str],
    mode: ResearchMode,
    source_snippets: list[dict],
    source_ids: list[str],
    protocol: ResearchProtocol | None,
) -> str:
    payload: dict[str, Any] = {
        "question": question,
        "source_snippets": source_snippets,
        "allowed_source_ids": source_ids,
        "required_output_shape": {
            "answer": "non-empty answer string",
            "major_claims": ["non-empty claim string"],
            "citations": ["claim_1 -> source_id"],
            "sources_used": ["source_id"],
            "uncertainty_notes": ["uncertainty or evidence limitation string"],
        },
    }
    if mode == "baseline":
        payload["instructions"] = [
            "Use a generic research style.",
            "Answer the question from the provided source snippets only.",
            "Tie major claims to fixture source IDs using claim-level citations.",
            "Do not use external sources, live web search, URLs, paper titles, or source IDs not listed in allowed_source_ids.",
        ]
    else:
        payload["research_protocol"] = asdict(protocol) if protocol else None
        payload["instructions"] = [
            "Use the generated ResearchProtocol to structure the answer.",
            "Answer the question from the provided source snippets only.",
            "Tie major claims to fixture source IDs using claim-level citations.",
            "Do not use external sources, live web search, URLs, paper titles, or source IDs not listed in allowed_source_ids.",
        ]

    return (
        "Produce a researcher answer for this experiment mode. "
        "Return strict JSON matching required_output_shape. "
        "Citation strings must be exactly in the form 'claim_N -> source_id'. "
        "Use only source IDs from allowed_source_ids. "
        "Only cite claim_N when major_claims has an item at position N; "
        "for example, claim_3 is valid only if there are at least three major_claims.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )
