from stem_research.retriever import EvidenceRetrieverTool


def test_evidence_retriever_returns_deterministic_top_k() -> None:
    sections = [
        {"section_name": "Intro", "text": "This paper motivates the task."},
        {"section_name": "Experiments", "text": "The retrieval method improves F1 on the dataset."},
        {"section_name": "Limitations", "text": "The method fails on long documents."},
    ]

    first = EvidenceRetrieverTool().retrieve("What retrieval method improves F1?", sections, top_k=2)
    second = EvidenceRetrieverTool().retrieve("What retrieval method improves F1?", sections, top_k=2)

    assert first == second
    assert len(first) == 2
    assert first[0].section_name == "Experiments"
