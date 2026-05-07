"""Visible placeholder prompt templates.

These templates document the intended future LLM calls. The current skeleton does
not call an LLM; all behavior is deterministic for smoke tests.
"""

STEM_PROTOCOL_PROMPT = """
Given a STEM research task class, solved examples, and an explicit rubric,
produce an inspectable research protocol with search, source, answer,
verification, citation, stopping, and failure-mode rules.
"""

BASELINE_RESEARCH_PROMPT = """
Answer the AI engineering research question using a concise generic research
style. Mark uncertainty and avoid unsupported claims.
"""

SPECIALIZED_RESEARCH_PROMPT = """
Answer the AI engineering research question using the generated StemResearch
protocol. Follow the protocol's structure and citation requirements.
"""
