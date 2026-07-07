"""Deterministic answer checkers for the benchmark.

The real competition uses an LLM judge on unseen prompts; we can't reproduce
that offline. But for *comparing models* a fixed labelled set with programmatic
checks is exactly right — it tells us, consistently, which model gets which
category right and how many tokens it spends doing so.

Each benchmark task carries a ``check`` dict whose ``type`` selects a checker:

  keywords  – answer must contain all `all` substrings and, for each group in
              `any`, at least one member (all case-insensitive).
  numeric   – some number in the answer must equal `value` within `tol`.
  label     – the first sentiment label found must be in `expected`.
  entities  – every string in `expected` must appear (case-insensitive).
  code      – extract the code block, exec it, then run `tests` (asserts);
              passes iff nothing raises (with a wall-clock timeout).
"""

from __future__ import annotations

import re
import signal
import threading
from contextlib import contextmanager

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
_SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")


def verify(check: dict, answer: str) -> bool:
    """Return True if ``answer`` satisfies ``check``. Never raises."""
    try:
        kind = check["type"]
        if kind == "keywords":
            return _check_keywords(check, answer)
        if kind == "numeric":
            return _check_numeric(check, answer)
        if kind == "label":
            return _check_label(check, answer)
        if kind == "entities":
            return _check_entities(check, answer)
        if kind == "code":
            return _check_code(check, answer)
        raise ValueError(f"unknown check type {kind!r}")
    except Exception:  # noqa: BLE001 - a broken check counts as a failure, not a crash
        return False


def _check_keywords(check: dict, answer: str) -> bool:
    low = answer.lower()
    for word in check.get("all", []):
        if word.lower() not in low:
            return False
    for group in check.get("any", []):
        if not any(w.lower() in low for w in group):
            return False
    return True


def _numbers(text: str) -> list[float]:
    out = []
    for raw in _NUM_RE.findall(text):
        raw = raw.rstrip(".").replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def _check_numeric(check: dict, answer: str) -> bool:
    target = float(check["value"])
    tol = float(check.get("tol", 0.01))
    return any(abs(n - target) <= tol for n in _numbers(answer))


def _check_label(check: dict, answer: str) -> bool:
    expected = check["expected"]
    if isinstance(expected, str):
        expected = [expected]
    expected = {e.lower() for e in expected}
    low = answer.lower()
    hits = [(low.find(lab), lab) for lab in _SENTIMENT_LABELS if lab in low]
    if not hits:
        return False
    first = min(hits)[1]  # earliest-appearing label
    return first in expected


def _check_entities(check: dict, answer: str) -> bool:
    low = answer.lower()
    return all(e.lower() in low for e in check["expected"])


def _extract_code(answer: str) -> str:
    match = _CODE_BLOCK_RE.search(answer)
    return match.group(1) if match else answer


@contextmanager
def _timeout(seconds: int):
    """Wall-clock guard so a bad generated snippet can't hang the benchmark.

    ``signal.alarm`` only works on the main thread; the benchmark evaluates in a
    thread pool, so outside the main thread we run without the guard (our test
    snippets are tiny and safe).
    """
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError("code execution timed out")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _check_code(check: dict, answer: str) -> bool:
    code = _extract_code(answer)
    namespace: dict = {}
    with _timeout(int(check.get("timeout", 4))):
        exec(compile(code, "<candidate>", "exec"), namespace)  # noqa: S102 - trusted local tests
        exec(compile(check["tests"], "<tests>", "exec"), namespace)  # noqa: S102
    return True
