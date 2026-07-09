"""Decide whether a locally-produced answer is trustworthy enough to keep.

The local model answers first at zero Fireworks-token cost; a verifier then
decides keep-vs-escalate. Crucially, the evaluation prompts are **unseen**, so we
have NO ground truth to check correctness against. These checks are therefore
deliberately cheap and *conservative*: they catch generation **failures** —
empty output, refusals, a math answer with no number, a "code" answer with no
code, an unlabeled sentiment reply — but they cannot catch an answer that is
well-formed yet wrong.

The asymmetry drives the policy: escalating a good answer wastes tokens for no
accuracy gain, while keeping an empty/broken one fails the accuracy gate. So we
**reject only on clear failure and otherwise keep the local answer**. For the
categories with no cheap structural signal (factual / summarization / logic),
"non-empty and not a refusal" is the best check available, so those default to
keep.
"""

from __future__ import annotations

import ast
import re

from .categories import Category

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
_SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")
# Heds/refusals that signal the model punted — worth a Fireworks retry.
_REFUSAL_RE = re.compile(
    r"\b(i (cannot|can't|am unable|am not able)\b|as an ai\b|"
    r"i'?m sorry,? but\b|i (do|don'?t) (not )?have (enough|access)\b|"
    r"i'?m not sure\b)",
    re.IGNORECASE,
)
# "def", "=>", ";", etc. — cues that a no-fence answer is probably source code.
_CODE_HINT_RE = re.compile(r"\b(def|class|import|return|lambda)\b|=>|;|\{")


def is_trustworthy(category: Category, prompt: str, answer: str) -> bool:
    """Return True to keep the local answer, False to escalate to Fireworks."""
    text = (answer or "").strip()
    if not text:
        return False
    if _REFUSAL_RE.search(text):
        return False

    if category is Category.MATH:
        return _has_number(text)
    if category in (Category.CODE_GEN, Category.CODE_DEBUG):
        return _has_code(text)
    if category is Category.SENTIMENT:
        return _has_sentiment_label(text)
    if category is Category.NER:
        return _has_entities(text)

    # factual / summarization / logic: no cheap structural check exists.
    # Non-empty and non-refusal is the best available signal — keep it.
    return True


def _has_number(text: str) -> bool:
    return bool(_NUM_RE.search(text))


def _has_sentiment_label(text: str) -> bool:
    low = text.lower()
    return any(label in low for label in _SENTIMENT_LABELS)


def _has_entities(text: str) -> bool:
    # Our NER prompt asks for "TYPE: entity" lines; a colon is the strongest
    # signal. Fall back to any capitalized (multi-word) entity-looking token.
    if ":" in text:
        return True
    return bool(re.search(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", text))


def _has_code(text: str) -> bool:
    # A non-empty fenced code block means the model followed the "single code
    # block" instruction — accept it (works for any language on unseen prompts).
    block = _CODE_BLOCK_RE.search(text)
    if block and block.group(1).strip():
        return True
    # No fenced block: accept only if the whole answer parses as Python (the
    # common case) — otherwise it's likely prose, not code, so escalate.
    if _CODE_HINT_RE.search(text):
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False
    return False
