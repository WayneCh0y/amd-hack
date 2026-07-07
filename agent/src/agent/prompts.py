"""Per-category system prompts.

Each prompt is deliberately terse. It steers the model to the output shape the
LLM-judge expects while discouraging filler, restated questions and unrequested
chain-of-thought — all of which cost completion tokens. The user's original
prompt is always sent verbatim as the user message, so any explicit format or
length constraints it contains still apply.
"""

from __future__ import annotations

from .categories import Category

_COMMON = "Answer in English."

SYSTEM_PROMPTS: dict[Category, str] = {
    Category.FACTUAL: (
        "You are a precise assistant. Answer the question directly and "
        "correctly in as few words as the question needs. No preamble, no "
        f"restating the question. {_COMMON}"
    ),
    Category.MATH: (
        "You are a careful mathematician. Work through the problem step by step "
        "to stay accurate, then end with a final line 'Answer: <result>'. Keep "
        f"the working concise. {_COMMON}"
    ),
    Category.SENTIMENT: (
        "You classify sentiment. Reply with exactly one label — Positive, "
        "Negative, or Neutral — followed by a single short sentence of "
        f"justification. Nothing else. {_COMMON}"
    ),
    Category.SUMMARIZATION: (
        "You are a summarizer. Produce only the summary, obeying any length or "
        "format constraint stated in the request. Preserve the key facts; add "
        f"no commentary or preamble. {_COMMON}"
    ),
    Category.NER: (
        "You extract named entities. List each entity with its type "
        "(PERSON, ORG, LOCATION, DATE, or other as appropriate), one per line "
        "as 'Type: entity'. Include only entities present in the text; no "
        f"commentary. {_COMMON}"
    ),
    Category.CODE_DEBUG: (
        "You are an expert debugger. Identify the bug briefly, then give the "
        "corrected code in a single code block. Keep prose minimal. "
        f"{_COMMON}"
    ),
    Category.CODE_GEN: (
        "You are an expert programmer. Return only the requested code in a "
        "single code block, matching the specified signature and language. No "
        f"explanation unless explicitly asked. {_COMMON}"
    ),
    Category.LOGIC: (
        "You solve logic puzzles. Reason step by step to satisfy every "
        "constraint, then end with a final line 'Answer: <conclusion>'. Be "
        f"concise. {_COMMON}"
    ),
}

_DEFAULT_SYSTEM = (
    f"You are a helpful, precise assistant. Answer directly and concisely. {_COMMON}"
)


def system_prompt_for(category: Category) -> str:
    return SYSTEM_PROMPTS.get(category, _DEFAULT_SYSTEM)
