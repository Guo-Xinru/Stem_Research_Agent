from dataclasses import asdict

from stem_research.schemas import EvidenceItem, PaperContext, PaperSection, QasperExample, ResearchOutput


def test_qasper_example_hides_references_for_researcher() -> None:
    example = QasperExample(
        id="q1",
        domain="scientific_paper_qa",
        question="What did the paper evaluate?",
        context=PaperContext(
            paper_title="Paper",
            abstract="Abstract.",
            sections=[PaperSection(section_name="Experiments", text="The paper evaluates F1.")],
        ),
        reference_answer="It evaluated F1.",
        evidence=[EvidenceItem(section_name="Experiments", text="The paper evaluates F1.")],
    )

    safe = example.without_references()

    assert safe.reference_answer is None
    assert safe.evidence == []


def test_research_output_serializes_requested_shape() -> None:
    output = ResearchOutput(
        id="q1",
        mode="baseline_with_tool",
        question="Question?",
        answer="Answer.",
        selected_evidence=[EvidenceItem(section_name="Intro", text="Answer evidence.", score=1.0)],
        used_protocol=False,
    )

    serialized = asdict(output)

    assert serialized["id"] == "q1"
    assert serialized["mode"] == "baseline_with_tool"
    assert serialized["selected_evidence"][0]["section_name"] == "Intro"
