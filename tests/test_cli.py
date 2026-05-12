from stem_research.cli import main
from stem_research.io_utils import load_json, write_jsonl


def test_cli_run_qasper_offline_smoke(tmp_path) -> None:
    data_dir = tmp_path / "qasper_mini"
    output_dir = tmp_path / "outputs"
    records = [_record("q1"), _record("q2")]
    write_jsonl(data_dir / "train.jsonl", records[:1])
    write_jsonl(data_dir / "eval.jsonl", records)

    exit_code = main(
        [
            "run-qasper",
            "--data",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--run-mode",
            "offline",
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    metrics = load_json(output_dir / "qasper_metrics.json")
    assert metrics["metadata"]["experiment"] == "qasper_mini"
    assert metrics["metadata"]["num_eval_examples"] == 1
    assert len(metrics["aggregate_by_mode"]) == 3


def test_cli_run_qasper_missing_data_prints_clear_message(tmp_path, capsys) -> None:
    exit_code = main(["run-qasper", "--data", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "prepare_qasper_mini.py" in captured.err


def _record(record_id: str) -> dict:
    return {
        "id": record_id,
        "domain": "scientific_paper_qa",
        "question": "What method improves retrieval?",
        "context": {
            "paper_title": "Paper",
            "abstract": "The paper studies retrieval.",
            "sections": [
                {
                    "section_name": "Method",
                    "text": "The proposed method improves retrieval by reranking evidence passages.",
                }
            ],
        },
        "reference_answer": "The method reranks evidence passages.",
        "evidence": [
            {
                "section_name": "Method",
                "text": "The proposed method improves retrieval by reranking evidence passages.",
            }
        ],
        "answer_type": "abstractive",
    }
