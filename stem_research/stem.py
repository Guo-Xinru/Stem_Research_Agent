"""Stem module: generate an explicit scientific-paper QA protocol."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from stem_research.llm import DEFAULT_OPENAI_MODEL, LLMConfigurationError, request_strict_json
from stem_research.schemas import QasperExample, ResearchProtocol, RunMode


REQUIRED_PROTOCOL_FIELDS = [
    "task_class",
    "observed_question_types",
    "evidence_strategy",
    "answer_rules",
    "tool_policy",
    "verification_rules",
    "failure_modes",
]


class Stem:
    """Creates an inspectable protocol from solved examples."""

    def __init__(self) -> None:
        self.last_protocol_metadata: dict[str, Any] = {}

    def generate_protocol(
        self,
        task_class_description: str,
        solved_examples: list[QasperExample] | list[dict[str, Any]],
        rubric: dict[str, Any] | None = None,
        run_mode: RunMode | str = "offline",
        known_failure_modes: list[str] | None = None,
        protocol_mode: str | None = None,
    ) -> ResearchProtocol:
        """Generate a protocol.

        ``protocol_mode`` is accepted as a backward-compatible alias for the
        old fixture/live terminology.
        """
        if protocol_mode is not None:
            run_mode = "llm" if protocol_mode == "live" else "offline"
        if run_mode not in ("offline", "llm"):
            raise ValueError("run_mode must be either 'offline' or 'llm'")
        if not task_class_description:
            raise ValueError("task_class_description is required")
        if not solved_examples:
            raise ValueError("At least one solved example is required")

        examples = [_example_to_dict(example) for example in solved_examples]
        if run_mode == "llm":
            try:
                protocol, metadata = request_strict_json(
                    system_prompt=_protocol_system_prompt(),
                    user_prompt=_protocol_user_prompt(
                        task_class_description=task_class_description,
                        solved_examples=examples,
                        known_failure_modes=known_failure_modes,
                    ),
                    validate=validate_protocol,
                    validation_label="protocol",
                )
                self.last_protocol_metadata = {
                    "generated_by": "openai",
                    "model": metadata["model"],
                    "validated": True,
                    "fallback_used": False,
                    "api_error": None,
                }
                return protocol
            except LLMConfigurationError:
                raise
            except Exception as exc:
                protocol = _offline_protocol(examples, known_failure_modes=known_failure_modes)
                self.last_protocol_metadata = {
                    "generated_by": "offline_fallback",
                    "model": DEFAULT_OPENAI_MODEL,
                    "validated": True,
                    "fallback_used": True,
                    "api_error": f"{exc.__class__.__name__}: {exc}",
                }
                return protocol

        protocol = _offline_protocol(examples, known_failure_modes=known_failure_modes)
        self.last_protocol_metadata = {
            "generated_by": "offline",
            "model": None,
            "default_llm_model": DEFAULT_OPENAI_MODEL,
            "validated": True,
            "fallback_used": False,
            "api_error": None,
        }
        return protocol


def validate_protocol(protocol: Any) -> ResearchProtocol:
    if isinstance(protocol, ResearchProtocol):
        protocol_data = asdict(protocol)
    elif isinstance(protocol, dict):
        protocol_data = protocol
    else:
        raise ValueError("protocol must be a dict or ResearchProtocol")

    missing = [field for field in REQUIRED_PROTOCOL_FIELDS if field not in protocol_data]
    if missing:
        raise ValueError(f"protocol missing required fields: {', '.join(missing)}")

    cleaned: dict[str, Any] = {}
    for field in REQUIRED_PROTOCOL_FIELDS:
        value = protocol_data[field]
        if field == "task_class":
            task_class = str(value).strip()
            if not task_class:
                raise ValueError("protocol task_class must be non-empty")
            cleaned[field] = task_class
            continue
        if field == "tool_policy":
            cleaned[field] = _validate_tool_policy(value)
            continue
        if not isinstance(value, list):
            raise ValueError(f"protocol field must be a non-empty list: {field}")
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            raise ValueError(f"protocol field must be non-empty: {field}")
        cleaned[field] = items

    if str(cleaned["task_class"]) != "scientific_paper_qa":
        cleaned["task_class"] = "scientific_paper_qa"
    return ResearchProtocol(**cleaned)


def _validate_tool_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("protocol tool_policy must be an object")
    required = {"required_tool", "when_to_use", "how_to_use"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"protocol tool_policy missing required fields: {', '.join(missing)}")
    how_to_use = value["how_to_use"]
    if not isinstance(how_to_use, list) or not [item for item in how_to_use if str(item).strip()]:
        raise ValueError("protocol tool_policy.how_to_use must be a non-empty list")
    return {
        "required_tool": str(value["required_tool"]).strip() or "EvidenceRetrieverTool",
        "when_to_use": str(value["when_to_use"]).strip(),
        "how_to_use": [str(item).strip() for item in how_to_use if str(item).strip()],
    }


def _offline_protocol(
    solved_examples: list[dict[str, Any]],
    *,
    known_failure_modes: list[str] | None,
) -> ResearchProtocol:
    observed_question_types = _infer_question_types(solved_examples)
    failure_modes = known_failure_modes or [
        "hallucinating details not in the paper",
        "answering from general ML knowledge instead of paper evidence",
        "citing irrelevant evidence",
        "ignoring unanswerable or weakly supported cases",
    ]
    evidence_patterns = _evidence_patterns(solved_examples)
    return validate_protocol(
        {
            "task_class": "scientific_paper_qa",
            "observed_question_types": observed_question_types,
            "evidence_strategy": [
                "Read the question for the requested paper-specific target before selecting evidence.",
                "Use EvidenceRetrieverTool to retrieve candidate passages from the paper sections.",
                *evidence_patterns,
                "Prefer passages that directly mention methods, datasets, metrics, results, limitations, or comparisons named in the question.",
            ],
            "answer_rules": [
                "Answer only from paper context and selected evidence, not general ML knowledge.",
                "Keep the answer concise and include uncertainty when evidence is weak or absent.",
                "Use wording that stays close to the paper evidence for factual claims.",
            ],
            "tool_policy": {
                "required_tool": "EvidenceRetrieverTool",
                "when_to_use": "before answering paper-grounded questions",
                "how_to_use": [
                    "retrieve candidate evidence from paper sections",
                    "answer only from retrieved or provided evidence",
                    "cite or reference evidence chunks used",
                ],
            },
            "verification_rules": [
                "Check that each factual sentence has lexical overlap with selected evidence.",
                "Do not include paper details that are absent from retrieved evidence.",
                "If retrieved evidence does not answer the question, say that the paper evidence is insufficient.",
            ],
            "failure_modes": failure_modes,
        }
    )


def _infer_question_types(examples: list[dict[str, Any]]) -> list[str]:
    type_keywords = {
        "method": ("method", "approach", "model", "algorithm", "propose"),
        "dataset": ("dataset", "data", "corpus", "benchmark"),
        "metric": ("metric", "measure", "score", "evaluation"),
        "result": ("result", "perform", "improve", "accuracy", "f1"),
        "limitation": ("limitation", "fail", "error", "future work"),
        "comparison": ("compare", "baseline", "outperform", "versus"),
        "motivation": ("why", "motivation", "problem", "goal"),
    }
    observed: list[str] = []
    question_blob = " ".join(str(example.get("question", "")).lower() for example in examples)
    for question_type, keywords in type_keywords.items():
        if any(keyword in question_blob for keyword in keywords):
            observed.append(question_type)
    return observed or ["method", "result", "comparison"]


def _evidence_patterns(examples: list[dict[str, Any]]) -> list[str]:
    sections: list[str] = []
    for example in examples:
        for evidence in example.get("evidence", []) or []:
            section_name = str(evidence.get("section_name") or "").strip()
            if section_name and section_name not in sections:
                sections.append(section_name)
    if not sections:
        return ["Inspect abstract, introduction, method, experiments, and conclusion passages when available."]
    joined = ", ".join(sections[:6])
    return [f"Solved examples frequently used evidence from these sections: {joined}."]


def _example_to_dict(example: QasperExample | dict[str, Any]) -> dict[str, Any]:
    if isinstance(example, QasperExample):
        return asdict(example)
    return example


def _protocol_system_prompt() -> str:
    return (
        "You are the Stem component in StemResearch. Generate a task-specific "
        "protocol for scientific-paper question answering. Return only valid JSON. "
        "Do not mention evaluator scores, protocol revision, web search, or external sources."
    )


def _protocol_user_prompt(
    *,
    task_class_description: str,
    solved_examples: list[dict[str, Any]],
    known_failure_modes: list[str] | None = None,
) -> str:
    schema = {
        "task_class": "scientific_paper_qa",
        "observed_question_types": ["method"],
        "evidence_strategy": ["concrete evidence selection rule"],
        "answer_rules": ["concrete answer rule"],
        "tool_policy": {
            "required_tool": "EvidenceRetrieverTool",
            "when_to_use": "before answering paper-grounded questions",
            "how_to_use": ["retrieve candidate evidence from paper sections"],
        },
        "verification_rules": ["concrete verification rule"],
        "failure_modes": ["concrete failure mode to avoid"],
    }
    return (
        "Generate a protocol for this task class.\n\n"
        f"Task class description:\n{task_class_description}\n\n"
        f"Solved examples JSON:\n{json.dumps(solved_examples, indent=2, sort_keys=True)}\n\n"
        f"Known failure modes JSON:\n{json.dumps(known_failure_modes or [], indent=2)}\n\n"
        "Return strict JSON matching this shape:\n"
        f"{json.dumps(schema, indent=2, sort_keys=True)}\n\n"
        "Rules: use solved examples only as examples; do not use eval reference answers; "
        "require EvidenceRetrieverTool; tell the researcher to answer from paper evidence only."
    )
