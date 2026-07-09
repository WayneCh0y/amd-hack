"""Tests for the keep-local-vs-escalate verifiers.

The contract is conservative: reject only clear generation failures (empty,
refusal, math-without-a-number, code-without-code, unlabeled sentiment) and keep
everything else, since we have no ground truth for unseen prompts.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agent.categories import Category  # noqa: E402
from agent.verifiers import is_trustworthy  # noqa: E402


@pytest.mark.parametrize("category", list(Category))
def test_empty_answer_always_escalates(category):
    assert is_trustworthy(category, "prompt", "") is False
    assert is_trustworthy(category, "prompt", "   \n ") is False


@pytest.mark.parametrize("category", list(Category))
def test_refusal_escalates(category):
    assert is_trustworthy(category, "p", "I'm sorry, but I cannot help with that.") is False
    assert is_trustworthy(category, "p", "As an AI, I don't have access to that.") is False


def test_math_needs_a_number():
    assert is_trustworthy(Category.MATH, "2+2?", "Answer: 4") is True
    assert is_trustworthy(Category.MATH, "2+2?", "It is four.") is False


def test_sentiment_needs_a_label():
    assert is_trustworthy(Category.SENTIMENT, "p", "Positive — the review praises it.") is True
    assert is_trustworthy(Category.SENTIMENT, "p", "The reviewer seems happy overall.") is False


def test_ner_accepts_typed_lines():
    good = "PERSON: Tim Cook\nORG: Apple\nLOCATION: California"
    assert is_trustworthy(Category.NER, "p", good) is True


def test_code_accepts_fenced_block_any_language():
    js = "```javascript\nconst f = () => 1;\n```"
    assert is_trustworthy(Category.CODE_GEN, "p", js) is True


def test_code_accepts_bare_python_that_parses():
    py = "def add(a, b):\n    return a + b"
    assert is_trustworthy(Category.CODE_GEN, "p", py) is True


def test_code_escalates_on_prose_only():
    assert is_trustworthy(Category.CODE_GEN, "p", "You should iterate over the list.") is False


def test_code_escalates_on_broken_python_without_fence():
    broken = "def add(a, b) return a + b"  # missing colon, no fence
    assert is_trustworthy(Category.CODE_DEBUG, "p", broken) is False


def test_open_categories_keep_any_nonempty_nonrefusal():
    for cat in (Category.FACTUAL, Category.SUMMARIZATION, Category.LOGIC):
        assert is_trustworthy(cat, "p", "Canberra is the capital of Australia.") is True
