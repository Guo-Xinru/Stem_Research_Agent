import pytest

from stem_research.evaluator import Evaluator, validate_llm_evaluation_response
from stem_research.schemas import ResearchOutput


def test_evaluator_computes_weighted_gold_fact_recall() -> None:
    question = {"id": "q1", "question": "Why do agents fail?"}
    output = ResearchOutput(
        question_id="q1",
        mode="specialized",
        question=question["question"],
        answer="Planning drift appears over many steps. Verification is mentioned.",
        major_claims=["Planning drift matters.", "Verification helps."],
        citations=["claim_1 -> fixture:agent_failure_notes"],
        sources_used=["fixture:agent_failure_notes"],
        uncertainty_notes=["Fixture behavior."],
    )
    gold_facts = {
        "question_id": "q1",
        "facts": [
            {
                "id": "f1",
                "fact": "Planning drift appears over many steps.",
                "keywords": ["planning drift", "steps"],
            },
            {
                "id": "f2",
                "fact": "Verification should include tests.",
                "keywords": ["verification", "tests"],
            },
            {
                "id": "f3",
                "fact": "Tool feedback can compound.",
                "keywords": ["tool", "feedback"],
            },
        ],
    }

    result = Evaluator().evaluate(question, output, gold_facts)

    assert result.evaluation_mode == "heuristic"
    assert result.gold_fact_recall == 0.5
    assert [item.label for item in result.gold_fact_evaluations] == [
        "addressed",
        "partially_addressed",
        "not_addressed",
    ]
    assert result.unsupported_claim_count == 1


def test_llm_evaluator_response_validation_accepts_complete_output() -> None:
    validated = validate_llm_evaluation_response(_valid_llm_response(), _gold_facts())

    assert validated["gold_fact_evaluations"][0]["label"] == "addressed"


def test_llm_evaluator_response_validation_rejects_invalid_label() -> None:
    response = _valid_llm_response()
    response["gold_fact_evaluations"][0]["label"] = "mostly_addressed"

    with pytest.raises(ValueError, match="Invalid LLM evaluator label"):
        validate_llm_evaluation_response(response, _gold_facts())


def test_llm_evaluator_response_validation_rejects_missing_fact_id() -> None:
    response = _valid_llm_response()
    response["gold_fact_evaluations"] = response["gold_fact_evaluations"][:1]

    with pytest.raises(ValueError, match="missing fact ids"):
        validate_llm_evaluation_response(response, _gold_facts())


def test_llm_evaluator_response_validation_rejects_duplicate_fact_id() -> None:
    response = _valid_llm_response()
    response["gold_fact_evaluations"][1]["fact_id"] = "f1"

    with pytest.raises(ValueError, match="duplicate fact ids"):
        validate_llm_evaluation_response(response, _gold_facts())


def _gold_facts() -> dict:
    return {
        "question_id": "q1",
        "facts": [
            {"id": "f1", "fact": "Fact one.", "keywords": ["one"]},
            {"id": "f2", "fact": "Fact two.", "keywords": ["two"]},
        ],
    }


def _valid_llm_response() -> dict:
    return {
        "gold_fact_evaluations": [
            {
                "fact_id": "f1",
                "label": "addressed",
                "rationale": "The answer states fact one.",
                "evidence_from_answer": "fact one",
            },
            {
                "fact_id": "f2",
                "label": "not_addressed",
                "rationale": "The answer does not state fact two.",
                "evidence_from_answer": "",
            },
        ],
        "unsupported_claim_count": 0,
        "citation_support_notes": "Claims are cited.",
        "source_quality_notes": "Fixture sources only.",
        "brief_critique": "Semantic coverage is mixed.",
    }
