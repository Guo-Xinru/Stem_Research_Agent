# AGENTS.md

## Project Purpose

This repository contains a minimal `StemResearch` agent for a JetBrains AI Engineering Intern test task.

The goal is to demonstrate a small, runnable stem-agent loop:

1. A generic stem agent observes a task class and a few solved examples.
2. It generates an explicit research protocol.
3. A specialized research agent uses that protocol to answer questions.
4. The result is evaluated against a generic baseline.

The chosen domain is AI engineering research questions, especially questions about LLM agents, coding agents, tool use, evaluation, context management, and autonomy.

Prioritize:

- Clear control flow
- Reproducible command-line experiments
- Measurable before/after comparison
- Simple interfaces
- Honest failure analysis

Do not optimize for framework sophistication. Optimize for a system that is easy to run, inspect, evaluate, and explain.

---

## Core Experimental Claim

The project compares two systems:

### 1. Generic Baseline Researcher

Answers AI engineering research questions with a generic research prompt.

### 2. Stem-Specialized Researcher

First uses a stem agent to generate a domain-specific research protocol, then answers the same questions using that protocol.

The experiment should measure whether the stem-generated protocol improves research quality over the generic baseline.

Primary metrics:

- Gold fact recall
- Citation support
- Unsupported claim count
- Source quality notes
- Optional: cost, number of searches, number of sources used

The before/after comparison is a core deliverable. Do not remove it.

---

## Architecture

Keep the project architecture to three core conceptual modules:

1. `Stem`
2. `SpecializedResearcher`
3. `Evaluator`

Small supporting utility files are allowed, but do not turn them into additional agent layers.

---

## 1. Stem

The `Stem` component generates a research protocol for the task class.

Inputs may include:

- Task class description
- Solved examples
- Scoring rubric
- Known failure modes

`solved_examples.json` should be small: 3–5 examples are enough. Quality matters more than quantity. The stem agent does not need many examples to generate a useful protocol.

The stem output must be explicit and inspectable, preferably JSON or a typed dataclass.

The generated protocol may include:

- Search strategy
- Source selection criteria
- Answer structure
- Claim verification rules
- Citation requirements
- Stopping criteria
- Known failure modes to avoid

The stem component is the main mechanism that makes this a stem-agent project rather than a generic research agent.

Do not implement evaluator-driven protocol revision in the main experiment. The protocol should not be updated using feedback from the evaluation set. Treat protocol revision as future work, not part of the minimal implementation.

---

## 2. SpecializedResearcher

The `SpecializedResearcher` component answers questions.

It must support two modes:

### Baseline Mode

Answer with a generic research prompt.

### Specialized Mode

Answer using the stem-generated research protocol.

The interface should stay as identical as possible between the two modes so that evaluation is fair.

The researcher should produce structured output containing at least:

- Question
- Answer
- Major claims
- Citations or source references
- Sources used
- Notes about uncertainty or weak evidence

Do not fake sources, citations, URLs, paper titles, or retrieved content.

---

## 3. Evaluator

The `Evaluator` scores baseline and specialized outputs against explicit criteria.

It should support evaluation against a small manually curated gold set.

Expected inputs:

- Question
- Model answer
- Gold facts
- Citations or source references
- Optional evaluator rubric

Expected outputs:

- Gold fact recall
- Citation support score or citation support notes
- Unsupported claim count
- Source quality notes
- Brief critique

Evaluation can be partly LLM-assisted, but the scoring criteria must be explicit and inspectable.

The LLM evaluator's role is to check whether each gold fact is addressed in the answer, not to generate a vague holistic quality score. Holistic scores are hard to reproduce and hard to explain.

Preferred gold fact labels:

- `addressed`
- `partially_addressed`
- `not_addressed`

Gold fact recall should be computed from these explicit labels.

---

## Allowed Supporting Code

The three modules above are the core architecture. Small supporting files are allowed when they keep the system clearer, for example:

- `llm.py` for OpenAI API calls
- `search.py` for search or retrieval wrappers
- `schemas.py` for dataclasses or typed dictionaries
- `prompts.py` for prompt templates
- `io_utils.py` for JSON read/write helpers

Do not turn supporting utilities into additional agent layers.

---

## Non-Goals

Do not build:

- A universal agent
- A multi-agent system
- A long-term autonomous background agent
- Persistent memory
- Dynamic tool learning
- A vector database RAG system
- A complex planner, router, or supervisor architecture
- A general web automation framework
- An evaluator-driven protocol revision loop in the main experiment

Do not use:

- LangChain
- LangGraph
- CrewAI
- AutoGen
- Vector databases
- Hidden orchestration frameworks
- Complex plugin systems

If an external API is needed, wrap it in a small plain-Python function with an interface that is easy to mock.

---

## Implementation Style

- Use plain Python.
- Prefer the standard library unless a dependency clearly improves reproducibility or clarity.
- Keep modules small and readable.
- Make data flow explicit through function arguments and return values.
- Prefer typed dataclasses, `TypedDict`, or simple dictionaries for structured results.
- Keep prompts, scoring rubrics, task examples, and experiment settings visible in source-controlled files.
- Avoid clever abstractions that obscure the research loop.
- Prefer deterministic experiment structure even if LLM outputs are nondeterministic.
- Fail loudly on missing API keys, malformed JSON, missing citations, or missing gold facts.

---

## Data Files

Use small, reviewable data files.

Recommended structure:

```text
data/
  questions.json
  solved_examples.json
  gold_facts.json
  rubric.json
````

`questions.json` should contain the evaluation questions.

`gold_facts.json` should contain manually curated expected facts for each question.

`solved_examples.json` should contain 3–5 examples of high-quality AI engineering research answers or answer outlines.

Do not auto-generate final gold facts without human review.

---

## CLI Experiments

Experiments should be reproducible from the command line.

Preferred smoke-test command:

```bash
python -m experiments.run_experiment --limit 3
```

Preferred full comparison command:

```bash
python -m experiments.run_experiment --limit 10
```

Optional command for reusing a previously generated protocol:

```bash
python -m experiments.run_experiment --protocol runs/protocol.json --limit 10
```

The main experiment runner may generate the protocol internally before running baseline and specialized comparisons. A separate `Stem` CLI is not required.

Experiment outputs should be written to `runs/` or `experiments/results/`.

Each output file should include enough metadata to understand what was run:

* Timestamp
* Model name
* Mode: baseline or specialized
* Question ID
* Generated protocol, if applicable
* Answer
* Claims
* Citations or sources
* Evaluation scores
* Errors or warnings

---

## Evaluation Requirements

The evaluator must compare baseline and specialized outputs on the same question set.

Do not change the question set between baseline and specialized runs.

Do not let the specialized agent see gold facts.

Do not let the evaluator influence the answer before scoring, unless running a clearly labeled additional revision experiment outside the main comparison.

The minimum acceptable experiment should produce a table or JSON summary with:

* Average gold fact recall for baseline
* Average gold fact recall for specialized
* Average citation support for baseline
* Average citation support for specialized
* Unsupported claim count for baseline
* Unsupported claim count for specialized
* Short notes on where the specialized protocol helped or failed

---

## Research and Citation Rules

For research outputs:

* Prefer primary or high-quality sources where possible.
* Distinguish empirical evidence from product claims, blog opinions, and speculation.
* Cite or reference sources for major factual claims.
* Mark uncertainty explicitly.
* Do not invent citations.
* Do not cite a source unless the answer actually relies on it.
* Do not use weak sources to support strong claims without caveats.

If real web search is unavailable, use clearly labeled fixtures or mock documents.

Never present fixture content as live web results.

---

## Testing

Prioritize focused tests for:

* Stem protocol generation output shape
* Baseline and specialized researcher output shape
* Evaluator scoring behavior
* CLI argument parsing
* JSON read/write behavior
* Handling of missing API keys or malformed model outputs

Use fixtures or mocked API responses for reproducible tests.

Network-dependent tests should be opt-in and clearly marked.

---

## Repository Hygiene

* Keep generated outputs, caches, and local secrets out of git.
* Do not commit API keys or local environment files.
* Document required environment variables in `.env.example` or README.
* Keep README instructions aligned with the actual CLI.
* Commit small, reviewable changes.
* Create a git checkpoint before large refactors.

Recommended `.gitignore` entries:

```text
.env
runs/
experiments/results/
__pycache__/
.pytest_cache/
*.log
```

---

## Write-Up Alignment

The code should support the final write-up.

The write-up needs to explain:

* Why this domain was chosen
* How the stem agent generates a research protocol
* What changed between baseline and specialized researcher
* How before/after was measured
* What improved
* What failed
* What was surprising
* What would be done with more time

Do not overstate autonomy. This is a minimal stem-agent prototype, not a complete autonomous researcher.

---

## Agent Instructions

When modifying this project:

1. Preserve the three-module conceptual architecture: `Stem`, `SpecializedResearcher`, and `Evaluator`.
2. Keep the stem mechanism centered on generating an explicit research protocol.
3. Preserve the baseline vs specialized comparison.
4. Preserve measurable evaluation against gold facts.
5. Do not implement evaluator-driven protocol revision in the main experiment.
6. Prefer direct, plain-Python implementation over adding dependencies.
7. Keep experiments runnable from the CLI.
8. Add or update tests when behavior changes.
9. Avoid framework creep.
10. Do not invent sources, citations, experiment results, or evaluation scores.
11. If a feature seems to require a framework, first try a small local abstraction.

