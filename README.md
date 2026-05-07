# StemResearch

StemResearch is a minimal JetBrains AI Engineering Intern test-task prototype. It compares a generic baseline researcher with a stem-specialized researcher that uses an explicit, inspectable research protocol.

## Architecture

The conceptual architecture has three modules:

- `Stem`: generates one deterministic research protocol from the task class, examples, and rubric.
- `SpecializedResearcher`: answers each question in `baseline` or `specialized` mode using fixture behavior.
- `Evaluator`: checks deterministic gold fact coverage and citation support notes.

Small support files handle schemas, prompts, JSON I/O, and the CLI experiment runner.

## Setup

```bash
uv sync
```

The current smoke test does not require OpenAI API access.

## Run Smoke Test

```bash
python -m experiments.run_experiment --limit 1
```

The command writes a timestamped JSON file under `runs/` and prints the result path.

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

## Current Limitations

- No live web search.
- No OpenAI API calls.
- No LLM evaluator.
- No protocol self-revision.
- Fixture sources are clearly labeled and are not real citations.

## Next Steps

- Manually review and improve gold facts.
- Add real retrieval behind a small plain-Python interface.
- Add optional OpenAI calls while preserving deterministic tests.
- Expand evaluation with inspectable citation checks.
