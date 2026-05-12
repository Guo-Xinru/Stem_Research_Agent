import pytest

from stem_research.schemas import ResearchProtocol
from stem_research.stem import Stem, _protocol_system_prompt, _protocol_user_prompt, validate_protocol


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


def test_stem_debug_prompt_writes_local_prompt_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STEM_DEBUG_PROMPT", "1")

    def fake_request_strict_json(*, system_prompt, user_prompt, validate, validation_label):
        return validate(VALID_PROTOCOL), {"model": "test-model"}

    monkeypatch.setattr("stem_research.stem.request_strict_json", fake_request_strict_json)

    Stem().generate_protocol(
        task_class_description="AI engineering research questions.",
        solved_examples=[{"id": "example", "outline": ["Check claims."]}],
        rubric={"scoring_rules": {"gold_fact_recall": "explicit"}},
        protocol_mode="live",
    )

    debug_files = list((tmp_path / "runs").glob("stem_debug_prompt_*.json"))
    assert len(debug_files) == 1
    debug_text = debug_files[0].read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in debug_text
    assert "AI engineering research questions." in debug_text


def test_live_protocol_prompt_focuses_on_research_answers() -> None:
    system_prompt = _protocol_system_prompt()
    user_prompt = _protocol_user_prompt(
        task_class_description="AI engineering research questions.",
        solved_examples=[{"id": "example", "outline": ["Check claims."]}],
        rubric={"scoring_rules": {"gold_fact_recall": "explicit"}},
    )

    assert "provided source snippets" in system_prompt
    assert "not writing a protocol for comparing baseline and specialized agents" in system_prompt.lower()
    assert "Preserve the baseline-vs-specialized comparison" not in user_prompt
    assert "The protocol is for answering AI engineering research questions" in user_prompt
    assert "Do not write about baseline agents" in user_prompt
    assert "Do not copy evaluator, gold fact, recall, score, or critique terminology" in user_prompt
