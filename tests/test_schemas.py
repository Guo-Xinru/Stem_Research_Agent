from dataclasses import asdict

from stem_research.schemas import ResearchProtocol, ResearchOutput


def test_research_output_serializes_to_dict() -> None:
    output = ResearchOutput(
        question_id="q1",
        mode="baseline",
        question="What fails?",
        answer="A short answer.",
        major_claims=["Claim one."],
        citations=["claim_1 -> fixture:agent_failure_notes"],
        sources_used=["fixture:agent_failure_notes"],
        uncertainty_notes=["Fixture behavior."],
    )

    serialized = asdict(output)

    assert serialized["question_id"] == "q1"
    assert serialized["mode"] == "baseline"
    assert serialized["sources_used"] == ["fixture:agent_failure_notes"]


def test_protocol_contains_required_sections() -> None:
    protocol = ResearchProtocol(
        search_strategy=["search"],
        source_selection_criteria=["sources"],
        answer_structure=["structure"],
        verification_rules=["verify"],
        citation_requirements=["cite"],
        stopping_criteria=["stop"],
        failure_modes_to_avoid=["avoid"],
    )

    serialized = asdict(protocol)

    assert set(serialized) == {
        "search_strategy",
        "source_selection_criteria",
        "answer_structure",
        "verification_rules",
        "citation_requirements",
        "stopping_criteria",
        "failure_modes_to_avoid",
    }
