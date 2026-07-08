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

# Word-problem shapes with no arithmetic symbol and no math keyword: age
# puzzles, "in N years", "combined age", "twice as old". These currently fall
# through to FACTUAL and get the small tier, which is likely to miss.
_MATH_WORDPROBLEM_RE = re.compile(
    r"\b(twice as old|combined ages?|in \d+ years?|"
    r"(their|his|her) (age|combined))\b"
)

# Words that carry a numeric value in plainly-worded prompts (e.g. "Tom is
# twice as old as Jerry. In five years, their combined age will be forty").
_SPELLED_NUMBER_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|twice|thrice|half|double|triple|"
    r"quarter)\b"
)
_DIGIT_RUN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
# "How old / how many / how much / what is / what will / find / determine …?"
# — questions that, combined with 2+ numeric tokens, are almost always math.
_QUANTITY_QUESTION_RE = re.compile(
    r"\b(how (old|many|much|long|far|fast|tall|heavy|old is)|"
    r"what (is|are|was|were|will) (the )?(final|total|result|answer|"
    r"value|price|cost|amount|number|sum|difference|product)|"
    r"\bfind\b|\bdetermine\b)"
)

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

    # 3) Math: keyword, a bare arithmetic expression, or a recognisable
    #    word-problem shape (age puzzles, "in N years, combined age...").
    if (
        _MATH_KEYWORD_RE.search(text)
        or _MATH_EXPR_RE.search(text)
        or _MATH_WORDPROBLEM_RE.search(text)
    ):
        return Category.MATH

    # 4) Constraint / deductive puzzles.
    if _LOGIC_RE.search(text):
        return Category.LOGIC

    # 5) Plainly-worded math fallback: 2+ numeric tokens (digits or spelled)
    #    AND a quantity-style question. Catches word problems that have no
    #    math keyword and no arithmetic symbol.
    numeric_hits = (
        len(_DIGIT_RUN_RE.findall(text))
        + len(_SPELLED_NUMBER_RE.findall(text))
    )
    if numeric_hits >= 2 and _QUANTITY_QUESTION_RE.search(text):
        return Category.MATH

    # 6) Default: factual / general knowledge Q&A.
    return Category.FACTUAL
