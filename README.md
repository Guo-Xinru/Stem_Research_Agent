# StemResearch

The full write-up is in [`report.md`](report.md).

StemResearch is a minimal prototype for protocol-level specialization in research agents.
A `Stem` generates an explicit `ResearchProtocol` from solved examples, and the system compares generic vs protocol-guided researchers on QASPER-mini scientific QA.

## Core Idea

```text
solved examples -> Stem-generated protocol -> baseline vs specialized researcher -> evaluation
```

The experiment compares three modes:

| mode                                 | tool use | protocol use | implementation logic                                                                                                                         |
| ------------------------------------ | -------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline_no_tool`                   | no       | no           | Answers directly from the question/context without evidence retrieval.                                                                       |
| `baseline_with_tool`                 | yes      | no           | Uses `EvidenceRetrieverTool` to retrieve raw top-3 evidence snippets, then answers from them.                                                |
| `specialized_with_protocol_and_tool` | yes      | yes          | Uses the same retriever, retrieves more raw candidates, then applies the Stem-generated protocol to rerank/filter evidence before answering. |

In short, baseline_no_tool tests the no-retrieval baseline, baseline_with_tool tests whether retrieval helps, and specialized_with_protocol_and_tool tests whether the generated protocol adds behavior beyond using the same retrieval tool.

The project focuses on a small, inspectable experiment rather than a fully autonomous agent. No LangChain, vector database, web search, multi-agent orchestration, or evaluator-driven revision.

## Components

- `Stem`: generates `ResearchProtocol`
- `SpecializedResearcher`: answers in baseline or protocol-guided mode
- `Evaluator`: scores answers against hidden references and evidence
- `EvidenceRetrieverTool`: simple deterministic local retriever

## Setup

```bash
uv sync
```

## Run Tests

```bash
uv run pytest
```

Expected:

```text
25 passed
```

## Run Main Experiment

```bash
uv run python -m stemresearch.cli run-qasper \
  --data data/qasper_mini \
  --run-mode offline \
  --output-dir outputs/qasper_protocol_selection_full
```

## Results

Latest QASPER-mini offline run:

| mode                               | answer_f1 | evidence_recall | evidence_precision | unsupported_claims |
| ---------------------------------- | --------: | --------------: | -----------------: | -----------------: |
| baseline_no_tool                   |    0.0595 |          0.0000 |             0.0000 |             2.0000 |
| baseline_with_tool                 |    0.0817 |          0.3212 |             0.1767 |             0.0400 |
| specialized_with_protocol_and_tool |    0.0777 |          0.2962 |             0.1900 |             0.0400 |

Interpretation: retrieval gives the largest gain. The protocol-guided mode changes behavior and improves evidence precision, but lowers recall and answer F1 in this run.

## Useful Commands

Run a small smoke test:

```bash
uv run python -m stemresearch.cli run-qasper \
  --data data/qasper_mini \
  --run-mode offline \
  --limit 3 \
  --output-dir outputs/smoke
```

Regenerate QASPER-mini data:

```bash
uv run python -m stemresearch.cli prepare-qasper-mini
```

Run older AI-engineering demo:

```bash
uv run python -m stemresearch.cli run-ai-demo
```
