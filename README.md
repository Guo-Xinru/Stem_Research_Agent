# StemResearch

StemResearch is a minimal experiment in explicit protocol-based specialization. It is not a general agent framework, web-search system, or multi-agent app.

The main experiment asks whether a Stem-generated protocol, learned from solved scientific-paper QA examples, improves behavior compared with generic baselines.

## Architecture

The code keeps three conceptual modules:

- `Stem`: observes solved examples and generates an explicit JSON research protocol.
- `SpecializedResearcher`: answers the same eval questions in three comparison modes.
- `Evaluator`: reports concrete, non-holistic metrics.

Small support files handle JSONL loading, a deterministic local evidence retriever, OpenAI-compatible calls, and CLI plumbing.

## Main Experiment: QASPER-Mini

QASPER is used because it provides an external scientific-paper QA task distribution with reference answers and evidence annotations. The local QASPER-mini subset is intended to contain:

- 30 train solved examples for Stem protocol generation
- 50 eval questions for the baseline vs specialized comparison

The researcher never sees eval `reference_answer` or gold `evidence`. The evaluator uses them only during scoring.

Expected local files:

```text
data/qasper_mini/train.jsonl
data/qasper_mini/eval.jsonl
```

## Prepare Data

The main CLI does not require HuggingFace or network access if the JSONL files already exist. To generate them from HuggingFace QASPER:

```bash
uv run python scripts/prepare_qasper_mini.py
```

Equivalent CLI command:

```bash
uv run python -m stemresearch.cli prepare-qasper-mini
```

If `datasets` is not installed, the script fails with a clear message. The runtime experiment itself does not depend on `datasets`.

## Run Offline

Offline mode is deterministic and is the default recommended smoke path:

```bash
uv run python -m stemresearch.cli run-qasper --data data/qasper_mini --run-mode offline
```

Outputs:

```text
outputs/qasper_protocol.json
outputs/qasper_predictions.jsonl
outputs/qasper_metrics.json
```

If local QASPER-mini files are missing, the command tells you to run the preparation script.

## Run OpenAI-Compatible LLM Mode

Only OpenAI-compatible API mode is supported.

Environment variables:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=...
OPENAI_BASE_URL=...
```

`OPENAI_MODEL` defaults to the model in `stem_research/llm.py`. `OPENAI_BASE_URL` is optional.

Run:

```bash
uv run python -m stemresearch.cli run-qasper --data data/qasper_mini --run-mode llm
```

If LLM mode is requested without `OPENAI_API_KEY`, it fails clearly.

## Compared Modes

- `baseline_no_tool`: generic paper QA using question plus paper context, without the retriever.
- `baseline_with_tool`: generic paper QA using the deterministic local `EvidenceRetrieverTool`.
- `specialized_with_protocol_and_tool`: uses the same retriever plus the Stem-generated protocol.

The two tool modes get the same paper context and retriever. Their main difference is protocol access.

## Metrics

The evaluator reports per-example and aggregate metrics by mode:

- `answer_token_f1`
- `evidence_recall`
- `evidence_precision`
- `unsupported_claim_count`
- `protocol_adherence`
- `answer_length_words`

`protocol_adherence` is only meaningful for `specialized_with_protocol_and_tool`; the other modes report it as not applicable.

## AI-Engineering Demo

The AI-engineering mini-set is retained as a small domain demo aligned with the JetBrains task:

```bash
uv run python -m stemresearch.cli run-ai-demo --run-mode offline
```

It is not the main statistical experiment.

## Evaluate Existing Predictions

```bash
uv run python -m stemresearch.cli evaluate \
  --predictions outputs/qasper_predictions.jsonl \
  --data data/qasper_mini/eval.jsonl
```

## Tests

```bash
uv run pytest
```

## Limitations

- QASPER is single-paper QA, not multi-source research synthesis.
- Evidence matching is heuristic.
- Offline mode is deterministic and not a real LLM agent.
- No evaluator-driven protocol revision is performed.
- No web search, vector database, persistent memory, LangChain, LangGraph, CrewAI, AutoGen, MCP, browser automation, or multi-agent orchestration is used.
