"""SpecializedResearcher module with deterministic fixture behavior."""

from __future__ import annotations

from stem_research.schemas import ResearchMode, ResearchOutput, ResearchProtocol


FIXTURE_SOURCES = {
    "fixture:agent_failure_notes": "Notes on coding-agent failures: long tasks drift, hidden state accumulates, tests are skipped, and tool errors compound.",
    "fixture:context_management_notes": "Notes on context management: agents need summarization, retrieval discipline, explicit state tracking, and guardrails against stale context.",
    "fixture:tool_use_notes": "Notes on tool use: autonomous agents fail when tool schemas are brittle, errors are not checked, or execution feedback is ignored.",
}


class SpecializedResearcher:
    """Answers in baseline or specialized mode without live search."""

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
