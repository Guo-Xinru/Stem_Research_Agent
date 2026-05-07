"""Stem module: generate an explicit research protocol."""

from __future__ import annotations

import json
from typing import Any

from stem_research.llm import DEFAULT_MODEL, request_strict_json
from stem_research.schemas import ResearchProtocol


REQUIRED_PROTOCOL_FIELDS = [
    "search_strategy",
    "source_selection_criteria",
    "answer_structure",
    "verification_rules",
    "citation_requirements",
    "stopping_criteria",
    "failure_modes_to_avoid",
]


class Stem:
    """Creates a fixture or live OpenAI-generated protocol for the research task."""

    def __init__(self) -> None:
        self.last_protocol_metadata: dict[str, Any] = {}

    def generate_protocol(
        self,
        task_class_description: str,
        solved_examples: list[dict[str, Any]],
        rubric: dict[str, Any],
        protocol_mode: str = "fixture",
    ) -> ResearchProtocol:
        if not task_class_description:
            raise ValueError("task_class_description is required")
        if not solved_examples:
            raise ValueError("At least one solved example is required")
        if not rubric:
            raise ValueError("rubric is required")
        if protocol_mode not in ("fixture", "live"):
            raise ValueError("protocol_mode must be either 'fixture' or 'live'")

        if protocol_mode == "live":
            protocol, metadata = request_strict_json(
                system_prompt=_protocol_system_prompt(),
                user_prompt=_protocol_user_prompt(
                    task_class_description=task_class_description,
                    solved_examples=solved_examples,
                    rubric=rubric,
                ),
                validate=validate_protocol,
            )
            self.last_protocol_metadata = {
                "generated_by": "openai",
                "model": metadata["model"],
            }
            return protocol

        self.last_protocol_metadata = {
            "generated_by": "fixture",
            "model": None,
            "default_live_model": DEFAULT_MODEL,
        }
        return validate_protocol(
            {
                "search_strategy": [
                    "Start from the exact research question and identify the agent failure mode being asked about.",
                    "Prefer evidence about AI engineering systems, coding agents, context windows, and tool-use reliability.",
                    "Separate observed failures from proposed mitigations.",
                ],
                "source_selection_criteria": [
                    "Prefer primary papers, benchmark reports, technical postmortems, and reproducible evaluations.",
                    "Use fixture sources only in the current smoke-test skeleton.",
                    "Treat blog posts and product claims as weak evidence unless corroborated.",
                ],
                "answer_structure": [
                    "State the short answer first.",
                    "List major claims as inspectable bullets.",
                    "Tie each claim to a source reference or mark it as uncertain.",
                    "End with limitations and practical implications.",
                ],
                "verification_rules": [
                    "Check every major claim against at least one cited or fixture source.",
                    "Distinguish context limits, planning limits, and tool-use failures.",
                    "Avoid claiming causal certainty from anecdotal observations.",
                ],
                "citation_requirements": [
                    "Do not invent URLs, paper titles, or citations.",
                    "Use clearly labeled fixture source identifiers until live retrieval exists.",
                    "Cite only sources that the answer actually relies on.",
                ],
                "stopping_criteria": [
                    "Stop after the answer covers the requested failure mode, likely causes, and mitigations.",
                    "Stop if only weak fixture evidence is available and mark uncertainty explicitly.",
                ],
                "failure_modes_to_avoid": [
                    "Overstating agent autonomy.",
                    "Collapsing distinct failure categories into one generic reliability problem.",
                    "Using evaluator feedback to revise the protocol in the main experiment.",
                    "Presenting fixture content as live web research.",
                ],
            }
        )


def validate_protocol(protocol: Any) -> ResearchProtocol:
    """Validate required protocol fields and return a typed ResearchProtocol."""
    if isinstance(protocol, ResearchProtocol):
        protocol_data = {field: getattr(protocol, field) for field in REQUIRED_PROTOCOL_FIELDS}
    elif isinstance(protocol, dict):
        protocol_data = protocol
    else:
        raise ValueError("protocol must be a dict or ResearchProtocol")

    missing = [field for field in REQUIRED_PROTOCOL_FIELDS if field not in protocol_data]
    if missing:
        raise ValueError(f"protocol missing required fields: {', '.join(missing)}")

    cleaned: dict[str, list[str]] = {}
    for field in REQUIRED_PROTOCOL_FIELDS:
        value = protocol_data[field]
        if not isinstance(value, list):
            raise ValueError(f"protocol field must be a non-empty list: {field}")
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            raise ValueError(f"protocol field must be non-empty: {field}")
        cleaned[field] = items

    return ResearchProtocol(**cleaned)


def _protocol_system_prompt() -> str:
    return (
        "You generate explicit, inspectable research protocols for a minimal "
        "StemResearch experiment. Return only JSON. Do not include markdown."
    )


def _protocol_user_prompt(
    *,
    task_class_description: str,
    solved_examples: list[dict[str, Any]],
    rubric: dict[str, Any],
) -> str:
    schema = {field: ["non-empty string rule"] for field in REQUIRED_PROTOCOL_FIELDS}
    return (
        "Generate a research protocol for the task class below.\n\n"
        f"Task class description:\n{task_class_description}\n\n"
        f"Solved examples JSON:\n{json.dumps(solved_examples, indent=2, sort_keys=True)}\n\n"
        f"Rubric JSON:\n{json.dumps(rubric, indent=2, sort_keys=True)}\n\n"
        "Return strict JSON matching this shape exactly:\n"
        f"{json.dumps(schema, indent=2, sort_keys=True)}\n\n"
        "Rules:\n"
        "- Each required field must be a non-empty list of concrete protocol rules.\n"
        "- Do not add extra top-level commentary.\n"
        "- Do not invent experiment results, sources, citations, or web search output.\n"
        "- Preserve the baseline-vs-specialized comparison and no protocol self-revision.\n"
    )
