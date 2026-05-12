from experiments.run_experiment import main
from stem_research.io_utils import load_json


def test_run_experiment_smoke(tmp_path) -> None:
    output_path = main(["--limit", "1", "--output-dir", str(tmp_path)])

    result = load_json(output_path)

    assert output_path.exists()
    assert result["metadata"]["uses_live_openai_api"] is False
    assert result["metadata"]["researcher_mode"] == "fixture"
    assert result["metadata"]["eval_mode"] == "heuristic"
    assert len(result["per_question"]) == 1
    assert "generated_protocol" in result
    assert result["protocol_provenance"]["generated_by"] == "fixture"
    assert result["protocol_provenance"]["validated"] is True
    assert "validation_error" in result["protocol_provenance"]
    assert result["protocol_provenance"]["validation_error"] is None
    assert "api_error" in result["protocol_provenance"]
    assert result["protocol_provenance"]["api_error"] is None
    assert "summary_metrics" in result
    assert result["researcher_provenance"]["researcher_mode"] == "fixture"
    assert result["researcher_provenance"]["researcher_generated_by"] == "fixture"
    assert result["researcher_provenance"]["baseline_live_used"] is False
    assert result["researcher_provenance"]["specialized_live_used"] is False
    assert result["researcher_provenance"]["researcher_api_error"] is None
    question_result = result["per_question"][0]
    snippet_ids = [snippet["id"] for snippet in question_result["source_snippets_used"]]
    assert snippet_ids
    assert question_result["baseline_output"]["sources_used"] == snippet_ids
    assert question_result["specialized_output"]["sources_used"] == snippet_ids
    assert all(
        citation.split("->", maxsplit=1)[1].strip() in snippet_ids
        for citation in question_result["specialized_output"]["citations"]
    )
