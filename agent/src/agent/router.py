"""Lightweight, zero-token task classifier.

Rather than spending a model call (and tokens) to decide what kind of task a
prompt is, we classify with ordered heuristics over keywords and simple
patterns. Getting the category slightly wrong is cheap — every category still
produces a sensible general answer — so we optimise for the common, clearly
signalled cases and fall back to factual Q&A otherwise.

Order matters: earlier checks win. Strong, explicit signals (code fences,
"summarise", "sentiment") are tested before fuzzier ones (math, logic).
"""

from __future__ import annotations

import re

from .categories import Category

# --- signal helpers ---------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```|~~~")
_CODE_HINT_RE = re.compile(
    r"\b(def |class |import |func\b|public\s+|private\s+|console\.log|"
    r"printf|System\.out|#include|return\b|=>|\bvar\b|\blet\b|\bconst\b)"
)

_DEBUG_RE = re.compile(
    r"\b(bug|debug|fix|fixes|fixed|error|errors|wrong|incorrect|broken|"
    r"doesn'?t work|not working|fails?|failing|corrected?|what'?s wrong|"
    r"why (does|is|isn'?t))\b"
)
_CODEGEN_RE = re.compile(
    r"\b(write|implement|create|generate|complete|define|build)\b"
    r"[^.?!]*\b(function|method|program|script|class|code|snippet|api|"
    r"algorithm|regex|query|sql)\b"
)

_SUMMARY_RE = re.compile(
    r"\b(summari[sz]e|summari[sz]ation|summary|tl;?dr|condense|"
    r"in (one|a single|two|three) sentences?|in \d+ words?|"
    r"key points|main idea)\b"
)
_SENTIMENT_RE = re.compile(
    r"\b(sentiment|positive or negative|negative or positive|"
    r"positive,? negative,? or neutral|emotional tone|"
    r"is this (review|text|tweet|comment) (positive|negative))\b"
)
_NER_RE = re.compile(
    r"\b(named entit(y|ies)|\bner\b|extract .*entit|identify .*entit|"
    r"extract (the )?(names|people|organi[sz]ations|locations|dates)|"
    r"person,? org|people,? organi[sz]ations)\b"
)

_MATH_KEYWORD_RE = re.compile(
    r"\b(calculate|compute|how much|how many|what is the (sum|product|"
    r"average|mean|total|value)|percent|percentage|\bsolve\b|equation|"
    r"probability|derivative|integral|factorial|square root|divisible)\b"
)
_MATH_EXPR_RE = re.compile(r"\d\s*[-+*/^×÷%]\s*\d|\d+\s*%|\$\d")

_LOGIC_RE = re.compile(
    r"\b(puzzle|riddle|deduce|deduction|logically|if and only if|"
    r"seating|arrange|ordering|rank(ing)? them|who (is|sits|owns|likes)|"
    r"each (of|person)|exactly (one|two|three)|no two|"
    r"knights? and knaves|true or false statement)\b"
)


def _has_code(text: str) -> bool:
    return bool(_CODE_FENCE_RE.search(text) or _CODE_HINT_RE.search(text))


def classify(prompt: str) -> Category:
    """Return the most likely :class:`Category` for ``prompt``."""
    text = prompt.lower()

    # 1) Code tasks — a fence or code-like syntax is a strong signal.
    if _has_code(text) or _CODEGEN_RE.search(text):
        if _DEBUG_RE.search(text):
            return Category.CODE_DEBUG
        if _CODEGEN_RE.search(text):
            return Category.CODE_GEN
        # Code present but no explicit generate/fix intent -> assume debugging.
        return Category.CODE_DEBUG

    # 2) Explicit single-purpose NL tasks.
    if _SUMMARY_RE.search(text):
        return Category.SUMMARIZATION
    if _SENTIMENT_RE.search(text):
        return Category.SENTIMENT
    if _NER_RE.search(text):
        return Category.NER

    # 3) Math: keyword or a bare arithmetic expression.
    if _MATH_KEYWORD_RE.search(text) or _MATH_EXPR_RE.search(text):
        return Category.MATH

    # 4) Constraint / deductive puzzles.
    if _LOGIC_RE.search(text):
        return Category.LOGIC

    # 5) Default: factual / general knowledge Q&A.
    return Category.FACTUAL
