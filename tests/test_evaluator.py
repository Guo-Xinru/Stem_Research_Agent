from stem_research.evaluator import Evaluator, answer_token_f1, evidence_scores, unsupported_claim_count
from stem_research.schemas import EvidenceItem, PaperContext, PaperSection, QasperExample, ResearchOutput


def test_answer_token_f1_exact_partial_and_empty() -> None:
    assert answer_token_f1("A retrieval method.", "A retrieval method.") == 1.0
    assert 0 < answer_token_f1("retrieval method", "retrieval reranking method") < 1
    assert answer_token_f1("", "reference") == 0.0
    assert answer_token_f1("", "") == 1.0


def test_evidence_recall_and_precision_simple_examples() -> None:
    gold = [EvidenceItem(section_name="Method", text="The method reranks evidence passages.")]
    selected = [EvidenceItem(section_name="Method", text="The method reranks evidence passages well.")]

    recall, precision = evidence_scores(selected, gold)

    assert recall == 1.0
    assert precision == 1.0


def test_unsupported_claim_count_handles_empty_answer() -> None:
    assert unsupported_claim_count("", []) == 0


def test_evaluator_scores_qasper_output() -> None:
    example = _example()
    output = ResearchOutput(
        id="q1",
        mode="specialized_with_protocol_and_tool",
        question=example.question,
        answer="The method reranks evidence passages.",
        selected_evidence=example.evidence,
        used_protocol=True,
    )

    result = Evaluator().evaluate_qasper(example, output)

    assert result.answer_token_f1 == 1.0
    assert result.evidence_recall == 1.0
    assert result.evidence_precision == 1.0
    assert result.protocol_adherence is not None


def _example() -> QasperExample:
    return QasperExample(
        id="q1",
        domain="scientific_paper_qa",
        question="What method is used?",
        context=PaperContext(
            paper_title="Paper",
            abstract="Abstract.",
            sections=[PaperSection(section_name="Method", text="The method reranks evidence passages.")],
        ),
        reference_answer="The method reranks evidence passages.",
        evidence=[EvidenceItem(section_name="Method", text="The method reranks evidence passages.")],
    )
