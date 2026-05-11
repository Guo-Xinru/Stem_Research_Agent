import pytest

from stem_research.schemas import ResearchProtocol
from stem_research.stem import Stem, validate_protocol


VALID_PROTOCOL = {
    "search_strategy": ["search"],
    "source_selection_criteria": ["sources"],
    "answer_structure": ["structure"],
    "verification_rules": ["verify"],
    "citation_requirements": ["cite"],
    "stopping_criteria": ["stop"],
    "failure_modes_to_avoid": ["avoid"],
}


def test_validate_protocol_accepts_required_fields() -> None:
    protocol = validate_protocol(VALID_PROTOCOL)

    assert isinstance(protocol, ResearchProtocol)
    assert protocol.search_strategy == ["search"]


def test_validate_protocol_rejects_missing_field() -> None:
    invalid = dict(VALID_PROTOCOL)
    invalid.pop("search_strategy")

    with pytest.raises(ValueError, match="missing required fields"):
        validate_protocol(invalid)


def test_validate_protocol_rejects_empty_field() -> None:
    invalid = dict(VALID_PROTOCOL)
    invalid["search_strategy"] = []

    with pytest.raises(ValueError, match="non-empty"):
        validate_protocol(invalid)


def test_validate_protocol_ignores_extra_fields() -> None:
    protocol_data = dict(VALID_PROTOCOL)
    protocol_data["unexpected_model_field"] = ["ignore me"]

    protocol = validate_protocol(protocol_data)

    assert isinstance(protocol, ResearchProtocol)
    assert not hasattr(protocol, "unexpected_model_field")


def test_fixture_mode_works_without_openai_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    protocol = Stem().generate_protocol(
        task_class_description="AI engineering research questions.",
        solved_examples=[{"id": "example", "outline": ["Check claims."]}],
        rubric={"scoring_rules": {"gold_fact_recall": "explicit"}},
        protocol_mode="fixture",
    )

    assert isinstance(protocol, ResearchProtocol)


def test_live_mode_without_openai_api_key_fails_clearly(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        Stem().generate_protocol(
            task_class_description="AI engineering research questions.",
            solved_examples=[{"id": "example", "outline": ["Check claims."]}],
            rubric={"scoring_rules": {"gold_fact_recall": "explicit"}},
            protocol_mode="live",
        )

    captured = capsys.readouterr()
    assert "OpenAI diagnostic" in captured.err
    assert "OPENAI_API_KEY detected=no" in captured.err
    assert "generated_by" not in captured.err
