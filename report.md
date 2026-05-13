# StemResearch: Protocol-Level Specialization for Research Agents

## 1. Problem Interpretation and Design Goal

I interpret a “stem agent” as an agent that starts with generic behavior and specializes for a task class by producing an explicit task protocol. In this project, specialization does not mean self-modifying code, dynamic tool acquisition, or a fully autonomous long-running system. It means generating an inspectable `ResearchProtocol` that changes how a researcher selects evidence and answers questions.

The project originally focused on AI engineering research questions. During implementation, I moved the main experiment to QASPER-mini, a paper-grounded QA setting, because it gives a more reproducible way to measure evidence use with hidden reference answers and evidence annotations. The AI-engineering version remains as a domain demo, while QASPER-mini is used for the controlled experiment.

The design goal is a minimal, runnable loop:

```text
solved examples + rubric -> Stem-generated protocol -> baseline vs specialized researcher -> evaluator comparison
```

The main question is not whether the agent becomes fully autonomous, but whether a generated protocol can produce measurable behavioral differences compared with a generic tool-using baseline.

## 2. Approach

### 2.1 Protocol-Level Specialization

The core method is protocol-level specialization. `Stem` reads solved examples and generates a structured `ResearchProtocol`. The protocol includes both descriptive research rules and executable fields:

- evidence selection policy
- answer policy
- source-grounding requirements
- failure modes to avoid
- stopping and verification rules

The most important change in the current experiment is that the protocol is not only shown to the researcher as text. It directly controls evidence selection in the specialized mode.

The three compared modes are:

1. `baseline_no_tool`: answers without retrieval.
2. `baseline_with_tool`: uses the local evidence retriever and answers from raw top-k evidence.
3. `specialized_with_protocol_and_tool`: uses the same retriever, but applies protocol-guided evidence filtering before answering.

This keeps the tool constant while testing whether the protocol adds behavior beyond retrieval.

### 2.2 Architecture

The project has three components.

`Stem` generates the protocol from solved examples. In offline mode, it produces a deterministic protocol so the system can run without live API access.

`SpecializedResearcher` answers questions. In baseline-with-tool mode, it retrieves the raw top 3 evidence snippets. In specialized mode, it retrieves a larger candidate set, then uses the protocol to rerank and filter evidence before answering.

`Evaluator` scores outputs against hidden reference answers and evidence. The researcher never sees gold answers or gold evidence.

The implementation intentionally avoids LangChain, vector databases, multi-agent orchestration, persistent memory, web search, and evaluator-driven protocol revision. The point is to keep the experiment inspectable and reproducible.

### 2.3 Evidence Retrieval and Protocol Execution

The retriever is a simple deterministic keyword/overlap retriever, not a vector database. This is deliberate: the goal is not to maximize retrieval quality, but to make the experiment easy to inspect.

The baseline-with-tool mode uses raw retriever output directly.

The specialized mode uses protocol fields such as:

```text
top_k_raw = 8
top_k_final = 3
min_question_token_overlap = 2
preferred sections = abstract, methods, experiments, results, conclusion
discard_generic_snippets = true
```

It retrieves more candidate evidence, applies deterministic reranking/filtering, and then answers from the selected evidence. This gives the protocol a real behavioral role.

## 3. Experiment

### 3.1 Setup

The main experiment uses QASPER-mini:

- 30 training / solved examples
- 50 evaluation questions
- offline deterministic run mode
- local evidence retriever
- hidden reference answers and evidence

The command was:

```bash
uv run python -m stemresearch.cli run-qasper \
  --data data/qasper_mini \
  --run-mode offline \
  --output-dir outputs/qasper_protocol_selection_full
```

The evaluator reports:

- `answer_token_f1`
- `evidence_recall`
- `evidence_precision`
- `unsupported_claim_count`
- `protocol_adherence`
- `answer_length_words`

The metrics are heuristic and deterministic. They are useful for comparison, but they do not fully capture semantic equivalence.

### 3.2 Results

| mode                               | answer_token_f1 | evidence_recall | evidence_precision | unsupported_claim_count | protocol_adherence | answer_length_words |
| ---------------------------------- | --------------: | --------------: | -----------------: | ----------------------: | -----------------: | ------------------: |
| baseline_no_tool                   |          0.0595 |          0.0000 |             0.0000 |                  2.0000 |                n/a |               44.86 |
| baseline_with_tool                 |          0.0817 |          0.3212 |             0.1767 |                  0.0400 |                n/a |               41.20 |
| specialized_with_protocol_and_tool |          0.0777 |          0.2962 |             0.1900 |                  0.0400 |             0.9700 |               39.38 |

The largest improvement comes from adding the retrieval tool. Compared with `baseline_no_tool`, `baseline_with_tool` improves answer F1, evidence recall, evidence precision, and unsupported-claim count.

The specialized mode now behaves differently from the tool baseline. It improves evidence precision from `0.1767` to `0.1900`, but evidence recall drops from `0.3212` to `0.2962`, and answer F1 drops slightly from `0.0817` to `0.0777`.

This means the protocol acted as a conservative evidence filter. It made evidence selection more selective, but did not improve overall answer quality in this run.

## 4. Interpretation

The experiment supports three conclusions.

First, retrieval is essential. Without tools, the researcher has no evidence recall and produces more unsupported claims.

Second, protocol specialization is now real and measurable. The specialized mode no longer behaves identically to the tool baseline; it uses protocol fields to change evidence selection and answer length.

Third, specialization is not automatically beneficial. The protocol improved evidence precision but reduced recall and slightly lowered lexical answer F1. This is a useful negative result: an inspectable protocol can change behavior, but better specialization requires better protocol design, retrieval quality, and evaluation.

## 5. What Worked and What Failed

What worked:

- The full offline pipeline is runnable.
- The protocol is explicit and inspectable.
- The specialized researcher uses the protocol in executable evidence selection.
- The evaluator can detect differences between no-tool, tool, and protocol-guided modes.
- The system remains small and reproducible.

What failed or remains weak:

- The specialized mode did not improve answer F1.
- The retriever has low evidence precision overall.
- The evaluator is heuristic and may miss semantic equivalence.
- QASPER-mini is only a small controlled benchmark.
- There is no web search, vector retrieval, semantic judge, or automatic protocol revision.

These limitations are intentional to some extent. The project prioritizes a minimal measurable prototype over a complex agent framework.

## 6. Future Work

The next step would be to improve evidence retrieval while keeping the same comparison structure. Better section-aware retrieval or a simple BM25-style scorer could reduce noise without adding a vector database.

I would also add a semantic evaluator as a secondary metric, with human spot checks, because token F1 is too strict for scientific QA. Protocol revision could be tested later, but only as a separate experiment so that evaluator feedback does not leak into the main comparison.

Finally, I would run multiple generated protocols and compare their behavior, because this experiment shows that the content of the protocol matters as much as the existence of a protocol.

## 7. Conclusion

StemResearch demonstrates a minimal, inspectable version of agent specialization. A generic researcher can be specialized through a generated protocol that changes evidence-selection behavior.

The result is mixed, not a broad success claim. The protocol-guided researcher improved evidence precision but reduced evidence recall and slightly lowered answer F1. The main value of the prototype is that it makes specialization explicit, executable, and measurable, while also showing that protocol-level specialization is not automatically better than a strong tool-using baseline.
