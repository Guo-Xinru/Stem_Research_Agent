# StemResearch

StemResearch is an evaluation-driven LLM research agent.
It starts from a generic agent and adapts to AI engineering research questions through an explicit generated protocol.

## Architecture

The conceptual architecture has three modules:

- `Stem`: generates one research protocol from the task class, examples, and rubric. The default is deterministic fixture mode; optional live mode uses OpenAI for protocol generation.
- `SpecializedResearcher`: answers each question in `baseline` or `specialized` mode using fixture behavior.
- `Evaluator`: checks gold fact coverage and citation support notes using deterministic heuristic behavior in the documented experiment.

Small support files handle schemas, prompts, JSON I/O, and the CLI experiment runner.

## Setup

```bash
uv sync
```

The default smoke test does not require OpenAI API access.

## Run Smoke Test

```bash
python -m experiments.run_experiment --limit 1
```

The command writes a timestamped JSON file under `runs/` and prints the result path.

To generate the protocol with OpenAI instead of the fixture protocol:

```bash
python -m experiments.run_experiment --limit 3 --protocol-mode live
```

Live mode requires `OPENAI_API_KEY`. If `OPENAI_MODEL` is absent, the code uses the conservative fallback defined in `stem_research/llm.py`.

Configure live protocol generation in `.env`:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

`OPENAI_MODEL` is optional and overrides the conservative default model defined in `stem_research/llm.py`.

In v0.3, live mode changes only how the Stem protocol is generated. Answer generation still uses the same fixture snippets, and the documented evaluator remains heuristic/offline.

## Run Tests

```bash
pytest
```

## Data

Starter fixtures live in `data/`:

- `questions.json`
- `solved_examples.json`
- `gold_facts.json`
- `source_snippets.json`
- `rubric.json`

`source_snippets.json` contains curated fixture snippets. The experiment selects snippets whose `related_question_ids` include the current question id, then passes the same snippets to both baseline and specialized researcher modes.

Gold facts are marked `needs_manual_review` and should not be treated as final benchmark data.

<!-- ## Current Limitations

- No live web search.
- OpenAI API calls are only used for `--protocol-mode live` in the documented v0.3 flow.
- LLM evaluator support is future work for the main experiment and is not part of v0.3.
- No protocol self-revision.
- Fixture sources are clearly labeled and are not real citations.

## Next Steps

- Manually review and improve gold facts.
- Add real retrieval behind a small plain-Python interface.
- Extend researcher or evaluator live behavior while preserving deterministic tests.
- Expand evaluation with inspectable citation checks. -->
