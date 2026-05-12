from stem_research.data import load_qasper_jsonl
from stem_research.io_utils import write_jsonl


def test_load_qasper_jsonl_normalizes_records(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "q1",
                "domain": "scientific_paper_qa",
                "question": "Question?",
                "context": {
                    "paper_title": "Paper",
                    "abstract": "Abstract.",
                    "sections": [{"section_name": "Intro", "text": "Useful evidence text."}],
                },
                "reference_answer": "Answer.",
                "evidence": [{"section_name": "Intro", "text": "Useful evidence text."}],
                "answer_type": "abstractive",
            }
        ],
    )

    examples = load_qasper_jsonl(path)

    assert examples[0].id == "q1"
    assert examples[0].context.sections[0].section_name == "Intro"
    assert examples[0].evidence[0].text == "Useful evidence text."
