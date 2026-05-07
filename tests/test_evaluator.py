from stem_research.evaluator import Evaluator
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

    assert result.gold_fact_recall == 0.5
    assert [item.label for item in result.gold_fact_evaluations] == [
        "addressed",
        "partially_addressed",
        "not_addressed",
    ]
    assert result.unsupported_claim_count == 1
