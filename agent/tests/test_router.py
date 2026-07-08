"""Unit tests for the heuristic task router and model selector."""

from __future__ import annotations

import pathlib
import sys

import pytest

# Make ``agent`` importable without installing the package.
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agent.categories import Category  # noqa: E402
from agent.model_selector import ModelSelector  # noqa: E402
from agent.router import classify  # noqa: E402


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("What is Newton's First Law of Motion?", Category.FACTUAL),
        ("Explain how a transformer neural network works.", Category.FACTUAL),
        ("A shirt costs $40 and is discounted by 25%. What is the final price?", Category.MATH),
        ("Calculate the average of 12, 18 and 30.", Category.MATH),
        ("What is 15 * 7 + 3?", Category.MATH),
        # Plainly-worded word problems: no math keyword, no arithmetic symbol.
        (
            "Tom is twice as old as Jerry. In five years, their combined age "
            "will be forty. How old is Jerry now?",
            Category.MATH,
        ),
        (
            "Alice is 3 times as old as Bob. In 4 years, she will be twice "
            "his age. How old is Bob now?",
            Category.MATH,
        ),
        # Negative: a plain factual "how old is X" with no numbers must stay
        # FACTUAL — the fallback needs 2+ numeric tokens to fire.
        ("How old is the Eiffel Tower?", Category.FACTUAL),
        ("Classify the sentiment of this review: the food was amazing.", Category.SENTIMENT),
        ("Is this tweet positive or negative? 'I love the new update!'", Category.SENTIMENT),
        ("Summarise the following article in one sentence: ...", Category.SUMMARIZATION),
        ("Give me a TL;DR of this paragraph.", Category.SUMMARIZATION),
        ("Extract the named entities from: Barack Obama visited Berlin.", Category.NER),
        ("Identify all the people, organizations and locations in the text.", Category.NER),
        (
            "This function has a bug, please fix it:\n```python\ndef f(): return x\n```",
            Category.CODE_DEBUG,
        ),
        (
            "Write a Python function is_even(n) that returns True for even numbers.",
            Category.CODE_GEN,
        ),
        (
            "Three friends each own a different pet. Ben owns the fish. Who owns the cat?",
            Category.LOGIC,
        ),
    ],
)
def test_classify(prompt, expected):
    assert classify(prompt) == expected


def test_selector_small_and_large():
    allowed = [
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "accounts/fireworks/models/qwen3-235b-a22b",
    ]
    sel = ModelSelector(allowed)
    assert sel.small() == "accounts/fireworks/models/llama-v3p1-8b-instruct"
    assert sel.large() == "accounts/fireworks/models/qwen3-235b-a22b"


def test_selector_single_model():
    sel = ModelSelector(["accounts/fireworks/models/only-model-13b"])
    assert sel.small() == sel.large() == "accounts/fireworks/models/only-model-13b"


def test_selector_moe_size():
    allowed = [
        "accounts/fireworks/models/mistral-7b-instruct",
        "accounts/fireworks/models/mixtral-8x7b-instruct",
    ]
    sel = ModelSelector(allowed)
    # 8x7b (~56B) should outrank a plain 7b for the large tier.
    assert sel.large() == "accounts/fireworks/models/mixtral-8x7b-instruct"
    assert sel.small() == "accounts/fireworks/models/mistral-7b-instruct"


def test_selector_prefers_non_reasoning_for_small():
    allowed = [
        "accounts/fireworks/models/deepseek-r1-7b",
        "accounts/fireworks/models/qwen2p5-7b-instruct",
    ]
    sel = ModelSelector(allowed)
    # Same size; the non-reasoning model is cheaper, so pick it for small.
    assert sel.small() == "accounts/fireworks/models/qwen2p5-7b-instruct"
