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


# --- prose must not be mistaken for code -------------------------------------
#
# `code_debug` / `code_gen` are in categories.LOCAL_OK, so a misroute here is
# expensive twice over: the task goes to the bundled local model, burns 30-45s of
# the 150s local budget, fails the execute-the-code verifier (prose is not code),
# and then escalates to Fireworks still wearing the debugger prompt. None of the
# published sample tasks trip this, which is exactly why it needs a test — the
# evaluation prompts are unseen variants.


@pytest.mark.parametrize(
    "prompt,expected",
    [
        (
            "Summarize the following passage in two sentences:\n\n"
            "'Many firms now expect a return to the office. Public health rules "
            "and private sector norms let each employer set its own policy, and "
            "a new class of hybrid roles has emerged as a constant compromise.'",
            Category.SUMMARIZATION,
        ),
        (
            "Classify the sentiment of this review: 'I let it charge overnight "
            "and the return process was painless, though the class of materials "
            "feels cheap.'",
            Category.SENTIMENT,
        ),
        (
            "What is the difference between a public company and a private company?",
            Category.FACTUAL,
        ),
        (
            "Extract all named entities from the following text and label each as "
            "PERSON, ORGANIZATION, LOCATION, or DATE:\n\n'Public Health England "
            "let its contract with Acme Var Ltd return to open tender in Leeds.'",
            Category.NER,
        ),
    ],
    ids=["summary", "sentiment", "factual", "ner"],
)
def test_prose_is_not_mistaken_for_code(prompt, expected):
    assert classify(prompt) == expected


def test_real_code_is_still_detected_without_a_fence():
    prompt = (
        "The snippet below is broken, please fix it:\n\n"
        "def total(items):\n"
        "    total = 0\n"
        "    for i in items:\n"
        "        total += i\n"
        "    return total_\n"
    )
    assert classify(prompt) == Category.CODE_DEBUG


def test_codegen_outranks_ner_when_asking_for_a_function():
    prompt = "Write a Python function that extracts all named entities from a string."
    assert classify(prompt) == Category.CODE_GEN


# --- the published Track 1 sample tasks ---------------------------------------
# The shapes the judge actually grades. A routing regression here is a scoring
# regression, so they are pinned.


@pytest.mark.parametrize(
    "prompt,expected",
    [
        (
            "Name the three primary colors in the RGB color model and briefly "
            "explain why displays use RGB instead of RYB.",
            Category.FACTUAL,
        ),
        (
            "What is the difference between machine learning and deep learning? "
            "Briefly explain how each works.",
            Category.FACTUAL,
        ),
        (
            "Explain the difference between RAM and ROM in a computer. What is "
            "each type used for?",
            Category.FACTUAL,
        ),
        (
            "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In "
            "Q2 it restocks 800 units. In Q3 it sells 640 units. How many units "
            "remain at the end of Q3?",
            Category.MATH,
        ),
        (
            "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is "
            "needed for 30 cookies? If sugar costs $2.40 per cup, what is the "
            "total cost of sugar for 30 cookies?",
            Category.MATH,
        ),
        (
            "Classify the sentiment of this customer review as Positive, Negative, "
            "or Neutral and give a one-sentence reason: 'The product arrived two "
            "days late and the packaging was damaged, but the item worked "
            "perfectly and customer support resolved my complaint within an hour.'",
            Category.SENTIMENT,
        ),
        (
            "Classify the sentiment of this tweet as Positive, Negative, or "
            "Neutral and give a one-sentence reason: 'Just got my order. Box was "
            "dented and the manual was missing, but honestly the device itself is "
            "flawless and set up in under 5 minutes.'",
            Category.SENTIMENT,
        ),
        (
            "Summarize the following passage in exactly two sentences:\n\n"
            "'Machine learning is increasingly deployed in healthcare for "
            "diagnosis, treatment planning, and patient monitoring. However, "
            "concerns remain about model interpretability, data privacy, and "
            "liability when errors occur.'",
            Category.SUMMARIZATION,
        ),
        (
            "Summarize the following passage in exactly three bullet points, each "
            "no longer than 15 words:\n\n'Remote work has transformed how "
            "companies operate globally. Employees gain flexibility and reduced "
            "commute times. However, challenges persist around collaboration and "
            "company culture.'",
            Category.SUMMARIZATION,
        ),
        (
            "Extract all named entities from the following text and label each as "
            "PERSON, ORGANIZATION, LOCATION, or DATE:\n\n'On March 15 2023, Sundar "
            "Pichai announced that Google would open a new AI research lab in "
            "Zurich, partnering with ETH Zurich to focus on large language model "
            "safety.'",
            Category.NER,
        ),
    ],
    ids="T01 T01b T01c T02 T02b T03 T03b T04 T04b T05".split(),
)
def test_classify_published_sample_tasks(prompt, expected):
    assert classify(prompt) == expected
