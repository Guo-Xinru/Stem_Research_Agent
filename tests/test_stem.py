import pytest

from stem_research.schemas import ResearchProtocol
from stem_research.stem import validate_protocol


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
