"""Visible prompt notes for the optional OpenAI-compatible paths."""

STEM_PROTOCOL_PROMPT = """
Generate an explicit scientific-paper QA protocol from solved QASPER-style
examples. The protocol should require local evidence retrieval, paper-grounded
answers, concise uncertainty handling, and verification against selected
evidence. Do not use evaluator feedback or reference answers from eval data.
"""

BASELINE_RESEARCH_PROMPT = """
Answer the paper question in a generic concise style using only the paper
context and, when provided, selected evidence.
"""

SPECIALIZED_RESEARCH_PROMPT = """
Answer the paper question using the Stem-generated protocol and selected
evidence. Do not add details that are absent from the paper evidence.
"""
