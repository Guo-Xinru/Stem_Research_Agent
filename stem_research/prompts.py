"""Visible placeholder prompt templates.

These templates document the intended future LLM calls. The current skeleton does
not call an LLM; all behavior is deterministic for smoke tests.
"""

STEM_PROTOCOL_PROMPT = """
You are the Stem component in StemResearch.

Your task is to generate a task-specific research protocol for answering AI engineering research questions.

The protocol will be used by a researcher to answer questions about LLM agents, coding agents, tool use, context management, evaluation, autonomy, and limitations of autonomous AI software engineers.

Important:
You are NOT designing the StemResearch experiment.
You are NOT writing a protocol for comparing baseline and specialized agents.
You are NOT writing instructions about run artifacts, experiment logs, traces, benchmark records, or measurement plans.
You are NOT revising your own protocol during execution.

Instead, generate a protocol that tells a researcher how to answer AI engineering research questions using the provided source snippets.

The protocol should teach the researcher to:
- identify distinct mechanisms, causes, limitations, and failure modes;
- use all relevant provided source snippets before forming the final answer;
- ground every major claim in the provided source IDs;
- separate evidence from speculation;
- synthesize across multiple snippets instead of summarizing each snippet independently;
- include concrete technical mechanisms, not only generic statements;
- avoid unsupported claims and invented citations;
- stop when all relevant snippets have been inspected and all major claims are cited.

Use the task class description, solved examples, rubric, and known failure modes only to infer how this type of research question should be answered.

Do not use gold facts.
Do not mention gold facts.
Do not assume access to evaluator outputs, scores, or critiques.

Return only valid JSON with exactly these seven fields:
- search_strategy
- source_selection_criteria
- answer_structure
- verification_rules
- citation_requirements
- stopping_criteria
- failure_modes_to_avoid

Each field must be a non-empty list of concrete strings.

The generated protocol should be useful for answering questions such as:
- Why do autonomous coding agents struggle with long-horizon software engineering tasks?
- Why is context management difficult for LLM agents?
- What makes tool use brittle in LLM agents?

A good protocol should guide the researcher to produce answers organized around source-grounded mechanisms, evidence, citations, and cross-source synthesis.
"""

BASELINE_RESEARCH_PROMPT = """
Answer the AI engineering research question using a concise generic research
style. Mark uncertainty and avoid unsupported claims.
"""

SPECIALIZED_RESEARCH_PROMPT = """
Answer the AI engineering research question using the generated StemResearch
protocol. Follow the protocol's structure and citation requirements.
"""
