from stem_research.researcher import SpecializedResearcher
from stem_research.schemas import EvidenceItem, PaperContext, PaperSection, QasperExample, ResearchProtocol


def test_baseline_no_tool_does_not_use_selected_evidence() -> None:
    output = SpecializedResearcher().answer_qasper(_example(), mode="baseline_no_tool")

    assert output.selected_evidence == []
    assert output.used_protocol is False


def test_baseline_with_tool_uses_evidence_but_no_protocol() -> None:
    output = SpecializedResearcher().answer_qasper(_example(), mode="baseline_with_tool")

    assert output.selected_evidence
    assert output.used_protocol is False


def test_specialized_uses_evidence_and_protocol() -> None:
    output = SpecializedResearcher().answer_qasper(
        _example(),
        mode="specialized_with_protocol_and_tool",
        protocol=_protocol(),
    )

    assert output.selected_evidence
    assert output.used_protocol is True


def _example() -> QasperExample:
    return QasperExample(
        id="q1",
        domain="scientific_paper_qa",
        question="What method improves retrieval?",
        context=PaperContext(
            paper_title="Paper",
            abstract="The paper studies retrieval.",
            sections=[
                PaperSection(
                    section_name="Method",
                    text="The proposed method improves retrieval by reranking evidence passages.",
                )
            ],
        ),
        reference_answer="The method reranks evidence passages.",
        evidence=[
            EvidenceItem(
                section_name="Method",
                text="The proposed method improves retrieval by reranking evidence passages.",
            )
        ],
    )


def _protocol() -> ResearchProtocol:
    return ResearchProtocol(
        task_class="scientific_paper_qa",
        observed_question_types=["method"],
        evidence_strategy=["Use retrieved evidence."],
        answer_rules=["Answer only from evidence."],
        tool_policy={
            "required_tool": "EvidenceRetrieverTool",
            "when_to_use": "before answering",
            "how_to_use": ["retrieve evidence"],
        },
        verification_rules=["Check grounding."],
        failure_modes=["hallucinating details not in the paper"],
    )
