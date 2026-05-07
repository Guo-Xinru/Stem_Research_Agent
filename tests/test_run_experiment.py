from experiments.run_experiment import main
from stem_research.io_utils import load_json


def test_run_experiment_smoke(tmp_path) -> None:
    output_path = main(["--limit", "1", "--output-dir", str(tmp_path)])

    result = load_json(output_path)

    assert output_path.exists()
    assert result["metadata"]["uses_live_openai_api"] is False
    assert len(result["per_question"]) == 1
    assert "generated_protocol" in result
    assert "summary_metrics" in result
