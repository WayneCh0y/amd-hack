"""Decide whether a locally-produced answer is trustworthy enough to keep.

The local model answers first at zero Fireworks-token cost; a verifier then
decides keep-vs-escalate. The evaluation prompts are **unseen**, so we have no
ground truth to compare against — which puts a hard ceiling on what a verifier
can honestly do.

The previous version of this module ignored that ceiling. It "verified" every
category with a shape check (a math answer *contains a number*, a sentiment
answer *contains a label*) and kept anything that wasn't empty. Shape checks
cannot see wrongness: the local model answered 108 where the answer was 144, and
the number check waved it through. That is what failed the accuracy gate.

So the rule now is: **a category is only answered locally if we can check
something real about the answer.** Categories that fail that test aren't verified
leniently — they aren't answered locally at all (see ``categories.LOCAL_OK``).
Two checks qualify:

  * **code** — extract the code and actually run it. A snippet that doesn't parse,
    or that throws on import/definition, is objectively broken regardless of what
    the task was. Run in a subprocess so a hang or a crash can't take us with it.
  * **summarization** — the prompt states the constraint ("in exactly one
    sentence", "in 50 words"), so we can check the answer against it, and against
    the source text: a summary must actually compress, and must not be a verbatim
    copy of the input.

Neither proves correctness. They catch the failures that *are* detectable, on the
categories where the local model was already accurate. Everything else goes to
Fireworks, where correctness is the model's job rather than ours.
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import tempfile

from .categories import Category

logger = logging.getLogger(__name__)

# How long a generated snippet may run before we call it broken. Defining a
# function is instant; anything that spins for seconds is a runaway loop.
_EXEC_TIMEOUT = 5.0

_CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
# Refusals / punts — the model gave up, so there's nothing to trust.
_REFUSAL_RE = re.compile(
    r"\b(i (cannot|can't|am unable|am not able)\b|as an ai\b|"
    r"i'?m sorry,? but\b|i (do|don'?t) (not )?have (enough|access)\b|"
    r"i'?m not sure\b)",
    re.IGNORECASE,
)
# Preamble the summarization prompt forbids; its presence means the model ignored
# the instructions, which usually means it ignored the length constraint too.
_PREAMBLE_RE = re.compile(
    r"^\s*(here'?s?\s+(is\s+)?(a|the)\b|sure[,!]|summary:|in summary\b)",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")
_WORD_RE = re.compile(r"\b[\w'-]+\b")

# Length/format constraints a summarization prompt can state, mapped to the
# sentence or word count they demand.
_NUMBER_WORDS = {
    "one": 1, "a single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
}
_SENTENCE_CONSTRAINT_RE = re.compile(
    r"\b(?:in|to|within|using)?\s*(?:exactly\s+|at most\s+|no more than\s+)?"
    r"(\d+|one|a single|two|three|four|five)\s+sentences?\b",
    re.IGNORECASE,
)
_WORD_CONSTRAINT_RE = re.compile(
    r"\b(?:in|to|within|under|using)?\s*(?:exactly\s+|at most\s+|no more than\s+|fewer than\s+)?"
    r"(\d+)\s+words?\b",
    re.IGNORECASE,
)


def is_trustworthy(category: Category, prompt: str, answer: str) -> bool:
    """Return True to keep the local answer, False to escalate to Fireworks."""
    text = (answer or "").strip()
    if not text:
        return False
    if _REFUSAL_RE.search(text):
        return False

    if category in (Category.CODE_GEN, Category.CODE_DEBUG):
        return _code_runs(prompt, text)
    if category is Category.SUMMARIZATION:
        return _summary_respects_constraints(prompt, text)

    # No sound local check exists for this category, so it should never have been
    # answered locally (categories.LOCAL_OK gates that). Escalating is the safe
    # answer if one slips through: a wasted call costs tokens, a wrong answer
    # costs the gate.
    return False


# -- code --------------------------------------------------------------------


def _extract_code(text: str) -> tuple[str, str]:
    """Return (language, source) from the answer's first non-empty code block.

    Falls back to treating the whole answer as code when the model skipped the
    fence but clearly emitted source (the prompt asks for a fence, so this is the
    uncommon path).
    """
    for language, body in _CODE_BLOCK_RE.findall(text):
        if body.strip():
            return language.lower(), body
    if re.search(r"^\s*(def|class|import|from)\s", text, re.MULTILINE):
        return "python", text
    return "", ""


def _code_runs(prompt: str, answer: str) -> bool:
    """True if the answer contains code that parses and executes cleanly.

    For non-Python we can only check that a non-empty code block exists — we have
    no interpreter for it in the image, and the eval set is Python-centric, so
    this is a rare path.
    """
    language, source = _extract_code(answer)
    if not source.strip():
        return False

    non_python = language and language not in ("python", "py", "python3")
    if non_python:
        return True

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    # Prose that happens to trip the fallback regex can still parse (a bare name
    # is a valid expression). Require an actual definition or import.
    if not any(
        isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
        )
        for node in tree.body
    ):
        return False

    return _executes_cleanly(source)


def _executes_cleanly(source: str) -> bool:
    """Run ``source`` in a subprocess; True if it exits 0 within the timeout.

    A subprocess (rather than ``exec``) so that a runaway loop, a crash, or a
    ``sys.exit`` in generated code cannot take the agent down with it. This
    catches import errors, NameErrors at module level, and non-termination —
    objective breakage — but says nothing about whether the logic is right.
    """
    with tempfile.TemporaryDirectory() as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-c", source],
                cwd=workdir,
                capture_output=True,
                timeout=_EXEC_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.info("Local code answer did not terminate in %.0fs", _EXEC_TIMEOUT)
            return False
        except OSError as exc:
            # Can't run the check — don't claim the code is bad on our account.
            logger.warning("Could not execute local code answer: %s; escalating", exc)
            return False

    if completed.returncode != 0:
        logger.info(
            "Local code answer failed to execute: %s",
            completed.stderr.decode("utf-8", "replace").strip()[-200:],
        )
        return False
    return True


# -- summarization -----------------------------------------------------------


def _summary_respects_constraints(prompt: str, answer: str) -> bool:
    """True if the summary obeys the prompt's stated length/format constraint.

    Summarization is the one non-code category with a checkable contract: the
    prompt supplies a passage and says what shape the output must take, so a
    violation is objective. We check that the answer compresses the passage, isn't
    a verbatim copy of it, and honours the stated limit.

    **A passage is mandatory.** The router sends anything matching "in one
    sentence" / "in 50 words" here, which also catches questions that merely carry
    a length constraint ("Explain photosynthesis in one sentence") — those are
    factual questions in disguise, with no source to check the answer against. If
    we can't find a passage, we can't verify anything, so we escalate.
    """
    if _PREAMBLE_RE.search(answer):
        return False

    source = _source_text(prompt)
    if not source:
        return False

    source_words = len(_WORD_RE.findall(source))
    answer_words = len(_WORD_RE.findall(answer))
    # No compression, or the model just echoed the passage back at us.
    if answer_words >= source_words:
        return False
    if answer.strip().lower() in source.strip().lower():
        return False

    # The constraint is stated in the instruction, not in the passage — otherwise
    # a passage that happens to mention "two sentences" would read as a constraint.
    instruction = prompt[: prompt.find(source)]

    sentence_limit = _stated_sentence_limit(instruction)
    if sentence_limit is not None and _count_sentences(answer) > sentence_limit:
        return False

    word_limit = _stated_word_limit(instruction)
    if word_limit is not None:
        # 20% slack: the judge grades intent, not an exact word count, and a rigid
        # check would escalate good summaries for no accuracy gain.
        if len(_WORD_RE.findall(answer)) > word_limit * 1.2:
            return False

    return True


# Below this, the trailing text is an instruction fragment, not a passage worth
# summarising — so there is nothing to verify a summary against.
_MIN_SOURCE_WORDS = 15


def _source_text(prompt: str) -> str:
    """The passage being summarised, or "" if the prompt doesn't supply one.

    Two shapes cover the real prompts: "Summarise the following ...: <passage>"
    and an instruction followed by a blank line and the passage. Anything shorter
    than ``_MIN_SOURCE_WORDS`` is the tail of an instruction, not a passage.
    """
    candidates: list[str] = []

    _, sep, after_colon = prompt.partition(":")
    if sep:
        candidates.append(after_colon.strip())

    blocks = [block.strip() for block in prompt.split("\n\n") if block.strip()]
    if len(blocks) > 1:
        candidates.append(blocks[-1])

    for candidate in candidates:
        if len(_WORD_RE.findall(candidate)) >= _MIN_SOURCE_WORDS:
            return candidate
    return ""


def _stated_sentence_limit(instruction: str) -> int | None:
    match = _SENTENCE_CONSTRAINT_RE.search(instruction)
    if not match:
        return None
    raw = match.group(1).lower()
    if raw.isdigit():
        return int(raw)
    return _NUMBER_WORDS.get(raw)


def _stated_word_limit(instruction: str) -> int | None:
    match = _WORD_CONSTRAINT_RE.search(instruction)
    return int(match.group(1)) if match else None


def _count_sentences(text: str) -> int:
    return len([part for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()])
