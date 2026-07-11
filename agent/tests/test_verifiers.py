"""Tests for the keep-local-vs-escalate verifiers.

The contract changed after the agent failed the 80% accuracy gate. It used to be
"reject clear generation failures, keep everything else", which meant shape checks
waved through fluent-but-wrong answers. It is now: **only the categories we can
actually check are answerable locally** (code — by executing it; summarization —
against the constraint the prompt states). Every other category escalates
unconditionally, so a wrong local answer can never reach the judge.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agent.categories import Category  # noqa: E402
from agent.verifiers import is_trustworthy  # noqa: E402

# The categories the local model is allowed to answer at all.
_LOCAL = (Category.CODE_GEN, Category.CODE_DEBUG, Category.SUMMARIZATION)
_NOT_LOCAL = (
    Category.FACTUAL,
    Category.MATH,
    Category.SENTIMENT,
    Category.NER,
    Category.LOGIC,
)


@pytest.mark.parametrize("category", list(Category))
def test_empty_answer_always_escalates(category):
    assert is_trustworthy(category, "prompt", "") is False
    assert is_trustworthy(category, "prompt", "   \n ") is False


@pytest.mark.parametrize("category", list(Category))
def test_refusal_escalates(category):
    assert is_trustworthy(category, "p", "I'm sorry, but I cannot help with that.") is False
    assert is_trustworthy(category, "p", "As an AI, I don't have access to that.") is False


@pytest.mark.parametrize("category", _NOT_LOCAL)
def test_unverifiable_categories_never_trusted(category):
    """The heart of the accuracy fix.

    Each of these answers is well-formed and would have passed the old shape
    checks — and each is wrong. `108` is the exact wrong answer the local model
    gave on the guide's practice math task (the answer is 144). With no ground
    truth there is no way to tell, so these categories are never answered locally.
    """
    plausible_but_unverifiable = {
        Category.FACTUAL: "Canberra; it is near the Australian Alps.",
        Category.MATH: "Answer: 108",
        Category.SENTIMENT: "Negative",
        Category.NER: "PERSON: Maria Sanchez",
        Category.LOGIC: "Answer: Sam owns the cat.",
    }
    assert is_trustworthy(category, "p", plausible_but_unverifiable[category]) is False


# -- code: verified by execution --------------------------------------------


def test_code_accepts_python_that_parses_and_runs():
    py = "```python\ndef add(a, b):\n    return a + b\n```"
    assert is_trustworthy(Category.CODE_GEN, "p", py) is True


def test_code_accepts_bare_python_that_runs():
    py = "def add(a, b):\n    return a + b"
    assert is_trustworthy(Category.CODE_GEN, "p", py) is True


def test_code_escalates_on_syntax_error():
    broken = "```python\ndef add(a, b) return a + b\n```"
    assert is_trustworthy(Category.CODE_DEBUG, "p", broken) is False


def test_code_escalates_when_it_throws_on_import():
    """Parses fine, blows up on execution — exactly what `ast.parse` alone misses."""
    throws = "```python\nimport definitely_not_a_real_module\n\ndef f():\n    return 1\n```"
    assert is_trustworthy(Category.CODE_GEN, "p", throws) is False


def test_code_escalates_on_runaway_loop():
    spins = "```python\ndef f():\n    return 1\n\nwhile True:\n    pass\n```"
    assert is_trustworthy(Category.CODE_GEN, "p", spins) is False


def test_code_escalates_on_prose_only():
    assert is_trustworthy(Category.CODE_GEN, "p", "You should iterate over the list.") is False


def test_code_escalates_on_code_free_block():
    assert is_trustworthy(Category.CODE_GEN, "p", "```python\n\n```") is False


def test_code_accepts_fenced_block_in_another_language():
    # No interpreter for it in the image, so a non-empty block is all we can check.
    js = "```javascript\nconst f = () => 1;\n```"
    assert is_trustworthy(Category.CODE_GEN, "p", js) is True


# -- summarization: verified against the stated constraint -------------------

_PASSAGE = (
    "Summarize the following in exactly one sentence: The Amazon rainforest spans "
    "nine countries and produces roughly twenty percent of the world's oxygen. "
    "Deforestation driven by cattle ranching has removed nearly a fifth of its "
    "original area since 1970, and scientists warn that continued clearing could "
    "push the ecosystem past a tipping point into savannah."
)


def test_summary_accepts_one_sentence_when_one_is_asked_for():
    good = (
        "The Amazon rainforest, which supplies about 20% of global oxygen, has lost "
        "nearly a fifth of its area to deforestation since 1970 and may tip into savannah."
    )
    assert is_trustworthy(Category.SUMMARIZATION, _PASSAGE, good) is True


def test_summary_escalates_when_it_blows_the_sentence_limit():
    too_many = (
        "The Amazon spans nine countries. It produces a fifth of the world's oxygen. "
        "Deforestation has cleared much of it. Scientists are worried."
    )
    assert is_trustworthy(Category.SUMMARIZATION, _PASSAGE, too_many) is False


def test_summary_escalates_on_preamble():
    preamble = "Here is a one-sentence summary: the Amazon is shrinking fast."
    assert is_trustworthy(Category.SUMMARIZATION, _PASSAGE, preamble) is False


def test_summary_escalates_when_it_does_not_compress():
    # Echoing the passage back is not a summary.
    echoed = _PASSAGE.split(":", 1)[1].strip()
    assert is_trustworthy(Category.SUMMARIZATION, _PASSAGE, echoed) is False


def test_summary_respects_a_word_limit():
    prompt = (
        "Summarize in 10 words: The cat sat on the mat and purred loudly all "
        "evening, while the rain fell steadily against the darkened kitchen window."
    )
    assert is_trustworthy(Category.SUMMARIZATION, prompt, "A cat purred on a mat as rain fell.") is True
    verbose = (
        "A cat sat upon a mat during the evening and purred very loudly indeed for "
        "a long while, as the rain kept falling on the window."
    )
    assert is_trustworthy(Category.SUMMARIZATION, prompt, verbose) is False


def test_summary_without_a_stated_constraint_is_kept():
    prompt = (
        "Summarize the following: Water boils at one hundred degrees Celsius at sea "
        "level, and it freezes at zero degrees under the same atmospheric pressure."
    )
    assert is_trustworthy(Category.SUMMARIZATION, prompt, "Water boils at 100C and freezes at 0C.") is True


def test_summary_escalates_when_the_prompt_supplies_no_passage():
    """Closes a routing hole: the router sends anything matching "in one sentence"
    to SUMMARIZATION, including factual questions that merely carry a length
    constraint. With no passage there is nothing to verify against, so we must not
    keep a local answer — this one is wrong, and undetectably so.
    """
    prompt = "Explain photosynthesis in one sentence."
    assert is_trustworthy(Category.SUMMARIZATION, prompt, "Plants turn moonlight into sugar.") is False
