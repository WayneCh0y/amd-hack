"""Per-category system prompts.

Each prompt is deliberately terse: it steers the model to the output shape the
LLM-judge expects while discouraging filler, restated questions and unrequested
chain-of-thought, all of which cost completion tokens. The user's original prompt
is always sent verbatim as the user message, so any explicit format or length
constraint it contains still applies.

Terse is not the same as underspecified, and two of these prompts were losing
answers on the practice set:

  * **Factual** said "in as few words as the question needs", and a multi-part
    question ("What is the capital of Australia, *and what body of water is it
    near?*") came back answering only the first half. Brevity now has to yield to
    completeness — the judge grades against the question's full intent.
  * **Sentiment** offered only Positive / Negative / Neutral, so a genuinely mixed
    review ("battery is great, but the screen scratches") was forced into a wrong
    bucket. The guide defines this category as labelling sentiment *and justifying
    the classification*, so the justification is required, not optional.
"""

from __future__ import annotations

from .categories import Category

_COMMON = "Answer in English."

SYSTEM_PROMPTS: dict[Category, str] = {
    Category.FACTUAL: (
        "You are a precise assistant. Answer correctly and completely: if the "
        "question has several parts, answer every part. Be brief — no preamble, "
        f"no restating the question — but never at the cost of a missing part. {_COMMON}"
    ),
    Category.MATH: (
        "You are a careful mathematician. Work through the problem step by step, "
        "using only the quantities the problem actually states — do not invent "
        "intermediate values. Re-check each arithmetic step, then end with a "
        f"final line 'Answer: <result>'. Keep the working concise. {_COMMON}"
    ),
    Category.SENTIMENT: (
        "You classify sentiment. Reply with exactly one label — Positive, "
        "Negative, Neutral, or Mixed — then one short sentence justifying it. Use "
        "Mixed when the text clearly praises some aspects and criticises others. "
        f"Nothing else. {_COMMON}"
    ),
    Category.SUMMARIZATION: (
        "You are a summarizer. Produce only the summary — no preamble, no "
        "'Here is', no commentary. Obey exactly any length or format constraint "
        f"stated in the request. Preserve the key facts. {_COMMON}"
    ),
    Category.NER: (
        "Extract named entities as 'TYPE: entity', one per line "
        f"(PERSON, ORG, LOCATION, DATE, or similar). Only entities in the text. {_COMMON}"
    ),
    Category.CODE_DEBUG: (
        "Identify the bug in one line, then give the corrected code in a single "
        "self-contained code block that runs as-is. Preserve the original "
        f"function name and signature. {_COMMON}"
    ),
    Category.CODE_GEN: (
        "You are an expert programmer. Return only the requested code, in a single "
        "self-contained code block that runs as-is, matching the specified "
        "signature and language. Handle the edge cases the request names. No "
        f"explanation unless explicitly asked. {_COMMON}"
    ),
    Category.LOGIC: (
        "You solve logic puzzles. Reason step by step, checking your conclusion "
        "against every stated constraint, then end with a final line "
        f"'Answer: <conclusion>'. Be concise. {_COMMON}"
    ),
}

_DEFAULT_SYSTEM = (
    f"You are a helpful, precise assistant. Answer directly and concisely. {_COMMON}"
)


def system_prompt_for(category: Category) -> str:
    return SYSTEM_PROMPTS.get(category, _DEFAULT_SYSTEM)
