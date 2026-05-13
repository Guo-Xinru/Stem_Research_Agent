import pytest

from stem_research.schemas import EvidenceItem, PaperContext, PaperSection, QasperExample, ResearchProtocol
from stem_research.stem import Stem, validate_protocol


VALID_PROTOCOL = {
    "task_class": "scientific_paper_qa",
    "observed_question_types": ["method"],
    "evidence_strategy": ["retrieve evidence"],
    "answer_rules": ["answer from evidence"],
    "tool_policy": {
        "required_tool": "EvidenceRetrieverTool",
        "when_to_use": "before answering",
        "how_to_use": ["retrieve chunks"],
    },
    "verification_rules": ["check overlap"],
    "failure_modes": ["hallucinating details not in the paper"],
}


def test_validate_protocol_accepts_required_fields() -> None:
    protocol = validate_protocol(VALID_PROTOCOL)

    assert isinstance(protocol, ResearchProtocol)
    assert protocol.tool_policy["required_tool"] == "EvidenceRetrieverTool"
    assert protocol.evidence_selection["top_k_raw"] == 8
    assert protocol.answer_policy["max_words"] == 80


def test_validate_protocol_rejects_missing_field() -> None:
    invalid = dict(VALID_PROTOCOL)
    invalid.pop("tool_policy")

    with pytest.raises(ValueError, match="missing required fields"):
        validate_protocol(invalid)


def test_stem_generates_scientific_paper_protocol_offline() -> None:
    protocol = Stem().generate_protocol(
        task_class_description="Scientific-paper QA.",
        solved_examples=[_example()],
        run_mode="offline",
    )

    assert protocol.task_class == "scientific_paper_qa"
    assert protocol.tool_policy["required_tool"] == "EvidenceRetrieverTool"
    assert protocol.observed_question_types
    assert protocol.evidence_selection["top_k_final"] == 3
    assert protocol.answer_policy["require_evidence_grounding"] is True
    assert "answering from general ML knowledge instead of paper evidence" in protocol.failure_modes


def test_llm_mode_without_openai_api_key_fails_clearly(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        Stem().generate_protocol(
            task_class_description="Scientific-paper QA.",
            solved_examples=[_example()],
            run_mode="llm",
        )

    captured = capsys.readouterr()
    assert "OpenAI diagnostic" in captured.err


def _example() -> QasperExample:
    return QasperExample(
        id="q1",
        domain="scientific_paper_qa",
        question="What method does the paper propose?",
        context=PaperContext(
            paper_title="Paper",
            abstract="The paper proposes a retrieval method.",
            sections=[PaperSection(section_name="Method", text="The method retrieves evidence.")],
        ),
        reference_answer="It proposes a retrieval method.",
        evidence=[EvidenceItem(section_name="Method", text="The method retrieves evidence.")],
    )
