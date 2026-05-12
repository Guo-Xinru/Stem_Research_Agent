import pytest

from stem_research.researcher import SpecializedResearcher, validate_research_output
from stem_research.schemas import ResearchOutput, ResearchProtocol


VALID_LIVE_OUTPUT = {
    "answer": "Agents struggle because dependent steps can compound mistakes.",
    "major_claims": ["Errors can compound across dependent steps."],
    "citations": ["claim_1 -> src_q1_error_compounding"],
    "sources_used": ["src_q1_error_compounding"],
    "uncertainty_notes": ["Fixture snippets are starter evidence."],
}


def test_validate_research_output_accepts_fixture_source_ids() -> None:
    output = validate_research_output(
        VALID_LIVE_OUTPUT,
        question_id="q1",
        mode="baseline",
        question="Why do coding agents struggle?",
        allowed_source_ids=["src_q1_error_compounding"],
    )

    assert isinstance(output, ResearchOutput)
    assert output.citations == ["claim_1 -> src_q1_error_compounding"]


def test_validate_research_output_rejects_unknown_citation_source() -> None:
    invalid = dict(VALID_LIVE_OUTPUT)
    invalid["citations"] = ["claim_1 -> https://example.com/source"]

    with pytest.raises(ValueError, match="unknown source id|URLs"):
        validate_research_output(
            invalid,
            question_id="q1",
            mode="baseline",
            question="Why do coding agents struggle?",
            allowed_source_ids=["src_q1_error_compounding"],
        )


def test_live_researcher_uses_openai_json_wrapper(monkeypatch) -> None:
    captured = {}

    def fake_request_strict_json(*, system_prompt, user_prompt, validate, validation_label):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["validation_label"] = validation_label
        return validate(VALID_LIVE_OUTPUT), {"model": "test-model"}

    monkeypatch.setattr("stem_research.researcher.request_strict_json", fake_request_strict_json)

    researcher = SpecializedResearcher(researcher_mode="live")
    output = researcher.answer(
        {"id": "q1", "question": "Why do coding agents struggle?"},
        mode="baseline",
        source_snippets=[
            {
                "id": "src_q1_error_compounding",
                "title": "Error compounding",
                "text": "Dependent engineering steps can compound early mistakes.",
            }
        ],
    )

    assert output.mode == "baseline"
    assert researcher.last_metadata["generated_by"] == "openai"
    assert researcher.last_metadata["model"] == "test-model"
    assert researcher.last_metadata["api_error"] is None
    assert captured["validation_label"] == "researcher output"
    assert "src_q1_error_compounding" in captured["user_prompt"]


def test_specialized_live_researcher_requires_protocol() -> None:
    researcher = SpecializedResearcher(researcher_mode="live")

    with pytest.raises(ValueError, match="specialized mode requires a protocol"):
        researcher.answer(
            {"id": "q1", "question": "Why do coding agents struggle?"},
            mode="specialized",
            source_snippets=[],
        )


def test_specialized_live_researcher_prompt_includes_protocol(monkeypatch) -> None:
    captured = {}

    def fake_request_strict_json(*, system_prompt, user_prompt, validate, validation_label):
        captured["user_prompt"] = user_prompt
        return validate(VALID_LIVE_OUTPUT), {"model": "test-model"}

    monkeypatch.setattr("stem_research.researcher.request_strict_json", fake_request_strict_json)

    researcher = SpecializedResearcher(researcher_mode="live")
    researcher.answer(
        {"id": "q1", "question": "Why do coding agents struggle?"},
        mode="specialized",
        source_snippets=[
            {
                "id": "src_q1_error_compounding",
                "title": "Error compounding",
                "text": "Dependent engineering steps can compound early mistakes.",
            }
        ],
        protocol=ResearchProtocol(
            search_strategy=["Inspect provided snippets."],
            source_selection_criteria=["Use fixture snippets only."],
            answer_structure=["Answer then claims."],
            verification_rules=["Cite each claim."],
            citation_requirements=["Use claim_N -> source_id."],
            stopping_criteria=["Stop after source-backed answer."],
            failure_modes_to_avoid=["Do not invent sources."],
        ),
    )

    assert "research_protocol" in captured["user_prompt"]
    assert "Cite each claim." in captured["user_prompt"]
