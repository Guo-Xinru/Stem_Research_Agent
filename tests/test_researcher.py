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


def test_specialized_uses_protocol_guided_evidence_selection() -> None:
    retriever = _FakeRetriever(_candidate_evidence())
    protocol = _protocol(top_k_raw=4, top_k_final=2, min_overlap=1)

    output = SpecializedResearcher(retriever=retriever).answer_qasper(
        _example_with_many_sections(),
        mode="specialized_with_protocol_and_tool",
        protocol=protocol,
    )

    assert retriever.last_top_k == 4
    assert len(output.selected_evidence) == 2
    assert output.selected_evidence[0].evidence_id == "method"
    assert all(item.evidence_id != "generic" for item in output.selected_evidence)


def test_baseline_with_tool_does_not_use_protocol_filtering() -> None:
    retriever = _FakeRetriever(_candidate_evidence())
    protocol = _protocol(top_k_raw=4, top_k_final=1, min_overlap=3)

    output = SpecializedResearcher(retriever=retriever).answer_qasper(
        _example_with_many_sections(),
        mode="baseline_with_tool",
        protocol=protocol,
    )

    assert retriever.last_top_k == 3
    assert [item.evidence_id for item in output.selected_evidence] == ["generic", "results", "appendix"]
    assert output.used_protocol is False


def test_protocol_top_k_final_controls_specialized_evidence_count() -> None:
    retriever = _FakeRetriever(_candidate_evidence())
    protocol = _protocol(top_k_raw=4, top_k_final=1, min_overlap=1)

    output = SpecializedResearcher(retriever=retriever).answer_qasper(
        _example_with_many_sections(),
        mode="specialized_with_protocol_and_tool",
        protocol=protocol,
    )

    assert len(output.selected_evidence) == 1


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


def _example_with_many_sections() -> QasperExample:
    return QasperExample(
        id="q_many",
        domain="scientific_paper_qa",
        question="What method improves retrieval with evidence passages?",
        context=PaperContext(
            paper_title="Paper",
            abstract="The paper studies retrieval.",
            sections=[
                PaperSection(section_name="Background", text="This paper studies models."),
                PaperSection(
                    section_name="Results",
                    text="The method improves retrieval by reranking evidence passages.",
                ),
                PaperSection(
                    section_name="Appendix",
                    text="Retrieval improves when evidence passages are reranked by the method.",
                ),
                PaperSection(section_name="Method", text="The method improves retrieval."),
            ],
        ),
    )


def _protocol(top_k_raw: int = 8, top_k_final: int = 3, min_overlap: int = 2) -> ResearchProtocol:
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
        evidence_selection={
            "top_k_raw": top_k_raw,
            "top_k_final": top_k_final,
            "min_question_token_overlap": min_overlap,
            "prefer_sections": ["method"],
            "discard_generic_snippets": True,
        },
        answer_policy={
            "max_words": 80,
            "require_evidence_grounding": True,
            "avoid_unverifiable_claims": True,
            "allow_insufficient_evidence_answer": True,
        },
    )


def _candidate_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            section_name="Background",
            text="This paper studies models.",
            score=99.0,
            evidence_id="generic",
        ),
        EvidenceItem(
            section_name="Results",
            text="The method improves retrieval by reranking evidence passages.",
            score=4.0,
            evidence_id="results",
        ),
        EvidenceItem(
            section_name="Appendix",
            text="Retrieval improves when evidence passages are reranked by the method.",
            score=3.0,
            evidence_id="appendix",
        ),
        EvidenceItem(
            section_name="Method",
            text="The method improves retrieval with evidence passages.",
            score=1.0,
            evidence_id="method",
        ),
    ]


class _FakeRetriever:
    def __init__(self, evidence: list[EvidenceItem]) -> None:
        self.evidence = evidence
        self.last_top_k: int | None = None

    def retrieve(self, question: str, sections: list[PaperSection], top_k: int = 5) -> list[EvidenceItem]:
        self.last_top_k = top_k
        return self.evidence[:top_k]
