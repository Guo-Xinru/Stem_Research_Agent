# StemResearch: Protocol-Level Specialization for AI Engineering Research Agents

## 1. Problem Interpretation and Design Goal

I interpret a "stem agent" as an agent that starts with a generic capability and then specializes for a task class by learning an explicit task protocol. In this project, specialization does not mean unrestricted self-modifying code, dynamic tool acquisition, or a long-running autonomous system. It means generating an inspectable `ResearchProtocol` that changes how a researcher answers a class of questions.

I chose AI engineering research questions as the task class because they expose common failure modes in agent systems: long-horizon coding reliability, context management, tool-use brittleness, evaluation, and autonomy limits. These questions are narrow enough for a small prototype, but still require source grounding, mechanism extraction, citation discipline, and explicit uncertainty.

The design goal is a minimal, runnable loop:

task class + solved examples + rubric -> Stem-generated protocol -> baseline vs specialized researcher -> evaluator comparison

The experiment asks whether giving the researcher a generated protocol improves answer quality compared with a generic baseline prompt.

The repository therefore prioritizes runnable code, visible intermediate artifacts, and measurable before/after comparison over complex autonomy.

## 2. Approach

### 2.1 Protocol-Level Specialization

The Stem receives a task class description, a small set of solved examples, and a scoring rubric. From these inputs it generates a structured `ResearchProtocol` with seven fields:

- `search_strategy`
- `source_selection_criteria`
- `answer_structure`
- `verification_rules`
- `citation_requirements`
- `stopping_criteria`
- `failure_modes_to_avoid`

The specialized researcher then receives the same question and source snippets as the baseline researcher, plus this generated protocol. The comparison isolates the effect of protocol guidance as much as this small prototype allows.

### 2.2 Architecture

The project has three conceptual components.

`Stem` generates the protocol. In fixture mode it returns a deterministic protocol. In live mode it calls OpenAI to generate the protocol, validates the result, and records provenance such as model, validation status, and API error status.

`SpecializedResearcher` answers questions in two modes. Baseline mode uses generic research instructions. Specialized mode receives the generated protocol in addition to the same question and fixture source snippets. Gold facts are hidden from both modes.

In live researcher mode, both baseline and specialized researchers use the same model, same question, same source snippets, and same output schema; the intended difference is the presence or absence of the generated protocol.

`Evaluator` scores answers against manually curated gold facts. It reports gold-fact recall, citation support notes, unsupported claim counts, and source quality notes. It does not use holistic 1-10 scoring.

The implementation intentionally avoids LangChain, vector databases, multi-agent orchestration, persistent memory, web search, and evaluator-driven protocol revision. The point is to keep the loop inspectable rather than framework-heavy.

### 2.3 How the Stem Infers the Task Approach

The Stem infers the task approach from the task class description, solved examples, and rubric. The interface also supports optional known failure modes, although the current live run did not pass additional failure-mode inputs. The solved examples show repeated answer patterns, such as checking evidence, preserving exact constraints, marking uncertainty, and grounding claims in source IDs. The rubric provides task-level quality criteria, while gold facts and evaluator outputs remain hidden from Stem.

The Stem does not learn by seeing evaluation scores. It converts visible task-level inputs into a reusable protocol for future questions. In v0.4.1, the live generated protocol focused on reading all provided snippets, extracting mechanisms, grounding claims in source IDs, synthesizing across snippets, avoiding unsupported generalizations, and stopping only after major claims have support.

### 2.4 How the Agent Decides What to Become

This prototype does not let the agent choose arbitrary architectures, tools, or skills. It specializes within a constrained design space: it becomes a protocol-guided AI engineering research agent.

The "skills" it obtains are represented as protocol rules rather than new code. Examples include source grounding, claim-level citation discipline, mechanism extraction, cross-source synthesis, unsupported-claim checks, and stopping rules. This is deliberately modest, but it makes the specialization easy to inspect and evaluate.

### 2.5 How It Rebuilds Without Breaking

The agent does not rewrite executable code. It "rebuilds" itself by replacing an inspectable protocol object. This keeps the system safer and easier to debug: fixture mode remains the default offline path, live mode is optional, and the run output records protocol provenance.

The project also records whether validation passed and whether API or validation errors occurred. After an earlier live protocol generated the wrong kind of protocol, I added a safe debug hook to save the exact Stem prompt locally when explicitly enabled. That made it possible to diagnose prompt contamination without exposing secrets.

### 2.6 When It Stops Evolving and Starts Executing

The first stopping gate is structural validation. `validate_protocol` requires all seven protocol fields, each as a non-empty list of strings. If live generation fails, the code fails loudly rather than silently falling back to fixture mode.

This was not enough by itself. An earlier protocol was structurally valid but semantically wrong: it described how to compare baseline and specialized agents rather than how to answer research questions. In v0.4.1, the prompt was fixed so the protocol is explicitly for answering AI engineering research questions using provided source snippets. The current implementation still relies on prompt constraints and inspection for semantic task alignment; automatic semantic protocol validation is future work.

Once the protocol is validated, it is frozen for the run. The evaluator does not revise it, and gold-fact scores are not fed back into Stem in the main experiment.

## 3. Experiments

### 3.1 Setup

The default smoke test remains offline, but the reported v0.4.1 comparison uses live protocol and live researcher modes. The live experiment used three starter questions:

- `q1_long_horizon_coding_agents`
- `q2_context_management`
- `q3_tool_use_brittleness`

Both baseline and specialized modes received the same curated fixture snippets for each question. Gold facts were hidden from Stem and both researchers. Live mode used OpenAI for protocol generation and researcher answer generation. The evaluator remained offline and heuristic.

The command was:

```text
.\.venv\Scripts\python.exe -m experiments.run_experiment --limit 3 --protocol-mode live --researcher-mode live
```

The cleaned evidence files are under `results/`, especially `v0.4.1_live_protocol_summary.md` and `v0.4.1_comparison_3q_summary.md`.

### 3.2 Metrics

The main metric is `gold_fact_recall`, computed from explicit gold-fact labels. The run also reports `citation_support_notes` and `unsupported_claim_count`. There is no vague holistic score.

The current evaluator is exact/keyword-based, so it can undercount semantic paraphrases. This is useful for deterministic smoke testing, but it is not a final human-quality evaluation.

### 3.3 Results

| question_id                   | baseline_gold_fact_recall | specialized_gold_fact_recall | delta_gold_fact_recall | baseline_unsupported_claim_count | specialized_unsupported_claim_count |
| ----------------------------- | ------------------------: | ---------------------------: | ---------------------: | -------------------------------: | ----------------------------------: |
| q1_long_horizon_coding_agents |                     0.167 |                        0.167 |                    0.0 |                                0 |                                   0 |
| q2_context_management         |                     0.083 |                        0.167 |                  0.084 |                                0 |                                   0 |
| q3_tool_use_brittleness       |                     0.167 |                          0.0 |                 -0.167 |                                0 |                                   0 |

Average baseline gold-fact recall: `0.139`

Average specialized gold-fact recall: `0.111`

Average delta: `-0.028`

Unsupported claim totals:

- Baseline: `0`
- Specialized: `0`

This does not show an average improvement from specialization on the 3-question starter set. The specialized researcher improved on the context-management question, tied on the long-horizon coding-agent question, and underperformed on the tool-use brittleness question under the heuristic evaluator.

### 3.4 Qualitative Interpretation

The end-to-end live pipeline works: OpenAI generates the Stem protocol, OpenAI generates baseline and specialized answers, and the offline evaluator scores both modes on the same questions. The v0.4.1 protocol is now research-answering focused after removing prompt contamination. I treat this as a pipeline success, but not as evidence that specialization improved answer quality.

The mixed result is still important. It suggests protocol-level specialization is feasible and inspectable, but not automatically beneficial. Likely reasons include the strength of the generic baseline, the very small three-question set, the exact keyword evaluator, and the fact that a protocol may improve discipline and citation structure more than keyword recall. It also shows that protocol-answer alignment remains hard: a good-looking protocol must still shape the actual answer in ways the evaluator can measure.

## 4. What Surprised Me

The hardest part was not API integration. The harder problem was keeping task levels separated. The Stem should learn how to answer AI engineering research questions, not how to write up the outer experiment comparing baseline and specialized agents.

One earlier live protocol passed schema validation but was semantically wrong. It generated an experiment-comparison protocol because the live Stem prompt accidentally contained baseline-vs-specialized framing. This made it clear that schema validation alone is not enough; generated protocols also need task-alignment checks.

I was also surprised by how difficult fair measurement is. The specialized protocol can make the answer more disciplined, but that does not guarantee higher keyword recall on a tiny fixture set.

## 5. What Failed or Remained Weak

The specialized researcher did not improve average recall in v0.4.1. The evaluation set has only three starter questions, and the heuristic evaluator can miss paraphrases or semantically correct coverage that does not match expected keywords. The gold set is small and should be treated as a starter evaluation set.

The source snippets are curated fixtures, not live research. There is no web search, no LLM evaluator, no automatic protocol revision, and no statistical generalization. Protocol validation is mostly structural, although the prompt debug hook and manual inspection helped catch the earlier semantic alignment problem.

## 6. Future Work

The next step is to expand the gold set to 6-10 manually reviewed questions. I would also add an optional LLM evaluator for semantic gold-fact coverage, with human spot checks rather than trusting it blindly.

Other useful extensions would be controlled retrieval or web search behind a small plain-Python interface, comparison across multiple generated protocols, semantic protocol validation, and protocol revision as a clearly separate experiment. I would keep revision outside the main comparison so the specialized agent does not train on evaluator feedback from the evaluation set.

## 7. Conclusion

StemResearch demonstrates a minimal, inspectable version of agent specialization. A generic researcher can be specialized through an explicit generated protocol rather than through hidden orchestration or self-modifying code.

The current result is mixed, not a success claim. The specialized researcher did not improve average heuristic recall on the 3-question starter set. The main value of the prototype is the measurable, debuggable pipeline and the finding that protocol specialization requires careful task alignment, not just a valid JSON schema.
