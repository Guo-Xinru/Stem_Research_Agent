"""Stem module: generate an explicit research protocol."""

from __future__ import annotations

from typing import Any

from stem_research.schemas import ResearchProtocol


class Stem:
    """Creates a deterministic placeholder protocol for the research task."""

    def generate_protocol(
        self,
        task_class_description: str,
        solved_examples: list[dict[str, Any]],
        rubric: dict[str, Any],
    ) -> ResearchProtocol:
        if not task_class_description:
            raise ValueError("task_class_description is required")
        if not solved_examples:
            raise ValueError("At least one solved example is required")
        if not rubric:
            raise ValueError("rubric is required")

        # Future OpenAI API call location:
        # send task_class_description, solved_examples, and rubric to a model,
        # then validate the returned JSON against ResearchProtocol.
        return ResearchProtocol(
            search_strategy=[
                "Start from the exact research question and identify the agent failure mode being asked about.",
                "Prefer evidence about AI engineering systems, coding agents, context windows, and tool-use reliability.",
                "Separate observed failures from proposed mitigations.",
            ],
            source_selection_criteria=[
                "Prefer primary papers, benchmark reports, technical postmortems, and reproducible evaluations.",
                "Use fixture sources only in the current smoke-test skeleton.",
                "Treat blog posts and product claims as weak evidence unless corroborated.",
            ],
            answer_structure=[
                "State the short answer first.",
                "List major claims as inspectable bullets.",
                "Tie each claim to a source reference or mark it as uncertain.",
                "End with limitations and practical implications.",
            ],
            verification_rules=[
                "Check every major claim against at least one cited or fixture source.",
                "Distinguish context limits, planning limits, and tool-use failures.",
                "Avoid claiming causal certainty from anecdotal observations.",
            ],
            citation_requirements=[
                "Do not invent URLs, paper titles, or citations.",
                "Use clearly labeled fixture source identifiers until live retrieval exists.",
                "Cite only sources that the answer actually relies on.",
            ],
            stopping_criteria=[
                "Stop after the answer covers the requested failure mode, likely causes, and mitigations.",
                "Stop if only weak fixture evidence is available and mark uncertainty explicitly.",
            ],
            failure_modes_to_avoid=[
                "Overstating agent autonomy.",
                "Collapsing distinct failure categories into one generic reliability problem.",
                "Using evaluator feedback to revise the protocol in the main experiment.",
                "Presenting fixture content as live web research.",
            ],
        )
