"""Unit tests for the pipeline orchestration.

We drive the pipeline with a fake ``FireworksClient`` so no network is needed.
The scripted responses let us exercise the cross-tier retry, empty-answer
fallback, deadline guard, and same-model de-duplication branches without
depending on any real model behaviour.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time
from collections import defaultdict

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agent.config import Config  # noqa: E402
from agent.local_model import LocalUsage  # noqa: E402
from agent.model_selector import ModelSelector  # noqa: E402
from agent.pipeline import Pipeline, Task, normalize_tasks  # noqa: E402


SMALL_MODEL = "accounts/fireworks/models/tinyfake-8b-instruct"
LARGE_MODEL = "accounts/fireworks/models/tinyfake-70b-instruct"


class FakeClient:
    """Stub with the same ``complete`` signature Pipeline uses.

    Configure per-model response queues; each ``complete`` call pops the next
    item. A queued ``Exception`` is raised (simulating an API failure); anything
    else is returned as the assistant text. Every call is recorded for asserts.
    """

    def __init__(self, responses: dict[str, list] | None = None):
        self._responses: dict[str, list] = defaultdict(list)
        for model, queue in (responses or {}).items():
            self._responses[model] = list(queue)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def complete(self, *, model: str, system: str, user: str,
                 max_tokens: int, temperature: float = 0.0) -> str:
        with self._lock:
            self.calls.append({"model": model, "user": user})
            queue = self._responses.get(model, [])
            if not queue:
                # No scripted answer for this model → default to empty (mimics
                # a model that returned no content).
                return ""
            item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLocal:
    """Stub local model with the same ``complete_with_usage`` kwargs Pipeline uses.

    Returns a scripted answer (or raises a queued Exception). Records calls so
    tests can assert the local path was exercised, and how it was bounded.
    """

    def __init__(self, answer, finish_reason: str = "stop", sleep: float = 0.0):
        self._answer = answer
        self._finish_reason = finish_reason
        self._sleep = sleep
        self.calls: list[str] = []
        self.max_tokens: list[int] = []
        self.timeouts: list[float | None] = []

    def complete_with_usage(self, *, system: str, user: str, max_tokens: int,
                            temperature: float = 0.0,
                            timeout: float | None = None) -> tuple[str, LocalUsage]:
        self.calls.append(user)
        self.max_tokens.append(max_tokens)
        self.timeouts.append(timeout)
        if self._sleep:
            time.sleep(self._sleep)
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer, LocalUsage(finish_reason=self._finish_reason)


def _config(**overrides) -> Config:
    """Minimal Config for tests; overrides win over these defaults."""
    kwargs = dict(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        allowed_models=[SMALL_MODEL, LARGE_MODEL],
        max_concurrency=2,
        time_budget=60,
        max_retries=0,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def _pipeline(client: FakeClient, local=None, **config_overrides) -> Pipeline:
    config = _config(**config_overrides)
    selector = ModelSelector(config.allowed_models)
    return Pipeline(config, client, selector, local=local)


def test_primary_tier_fails_alt_succeeds():
    # FACTUAL routes to Tier.SMALL as primary; when the small model raises, the
    # alt tier (large) should be tried and its answer returned.
    client = FakeClient({
        SMALL_MODEL: [RuntimeError("boom")],
        LARGE_MODEL: ["42"],
    })
    pipeline = _pipeline(client)

    results = pipeline.run([Task(task_id="t1", prompt="What is 6 times 7?")])

    assert results == [{"task_id": "t1", "answer": "42"}]
    called_models = [c["model"] for c in client.calls]
    assert called_models == [SMALL_MODEL, LARGE_MODEL]


def test_primary_returns_empty_alt_tried():
    # Empty text from the primary is treated as no-answer → try alt tier.
    client = FakeClient({
        SMALL_MODEL: [""],
        LARGE_MODEL: ["Paris"],
    })
    pipeline = _pipeline(client)

    results = pipeline.run([Task(task_id="cap", prompt="Capital of France?")])

    assert results == [{"task_id": "cap", "answer": "Paris"}]
    assert [c["model"] for c in client.calls] == [SMALL_MODEL, LARGE_MODEL]


def test_same_model_both_tiers_no_duplicate_call():
    # With a single allowed model, small() == large(); a failure on the primary
    # must not trigger a wasted second call to the same model.
    only = "accounts/fireworks/models/solo-13b-instruct"
    client = FakeClient({only: [RuntimeError("boom")]})
    pipeline = _pipeline(client, allowed_models=[only])

    results = pipeline.run([Task(task_id="q", prompt="What is 2 plus 2?")])

    assert results == [{"task_id": "q", "answer": ""}]
    # Exactly one call — no cross-tier retry against the same model.
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == only


def test_deadline_exhausted_returns_fallback_no_calls():
    # time_budget=0 → deadline is already reached before the first task; every
    # result should be the fallback and the client must never be called.
    # (from_env clamps time_budget to >=30, but the dataclass ctor doesn't —
    # tests bypass env parsing.)
    client = FakeClient({SMALL_MODEL: ["should not be used"]})
    pipeline = _pipeline(client, time_budget=0)

    tasks = [
        Task(task_id="t1", prompt="Anything?"),
        Task(task_id="t2", prompt="Also anything?"),
    ]
    results = pipeline.run(tasks)

    assert results == [
        {"task_id": "t1", "answer": ""},
        {"task_id": "t2", "answer": ""},
    ]
    assert client.calls == []


def test_empty_prompt_returns_fallback_no_call():
    client = FakeClient({SMALL_MODEL: ["nope"], LARGE_MODEL: ["nope"]})
    pipeline = _pipeline(client)

    results = pipeline.run([Task(task_id="blank", prompt="   ")])

    assert results == [{"task_id": "blank", "answer": ""}]
    assert client.calls == []


def test_all_tiers_fail_returns_fallback():
    client = FakeClient({
        SMALL_MODEL: [RuntimeError("a")],
        LARGE_MODEL: [RuntimeError("b")],
    })
    pipeline = _pipeline(client)

    # FACTUAL primary → small first, then large as retry.
    results = pipeline.run([Task(task_id="doomed", prompt="Capital of Spain?")])

    assert results == [{"task_id": "doomed", "answer": ""}]
    assert [c["model"] for c in client.calls] == [SMALL_MODEL, LARGE_MODEL]


def test_results_preserve_input_order_with_concurrency():
    # ThreadPoolExecutor may finish tasks out of order; results must still be
    # emitted in input order.
    client = FakeClient({
        SMALL_MODEL: ["A1", "A2", "A3"],
        LARGE_MODEL: ["A1L", "A2L", "A3L"],
    })
    pipeline = _pipeline(client, max_concurrency=3)

    tasks = [
        Task(task_id="t1", prompt="Capital of France?"),
        Task(task_id="t2", prompt="Capital of Germany?"),
        Task(task_id="t3", prompt="Capital of Italy?"),
    ]
    results = pipeline.run(tasks)

    assert [r["task_id"] for r in results] == ["t1", "t2", "t3"]


def test_trusted_local_answer_skips_fireworks():
    # A well-formed local answer that passes verification is used as-is; the
    # Fireworks client must never be called (zero tokens).
    client = FakeClient({SMALL_MODEL: ["should not be used"]})
    local = FakeLocal("Answer: 42")
    pipeline = _pipeline(client, local=local)

    results = pipeline.run([Task(task_id="m", prompt="What is 6 times 7?")])

    assert results == [{"task_id": "m", "answer": "Answer: 42"}]
    assert local.calls == ["What is 6 times 7?"]
    assert client.calls == []  # no escalation


def test_untrustworthy_local_answer_escalates():
    # A math answer with no number fails verification → escalate to Fireworks.
    # "Calculate ..." routes to MATH, whose verifier requires a number.
    client = FakeClient({SMALL_MODEL: ["36"], LARGE_MODEL: ["36"]})
    local = FakeLocal("I think it is quite large.")
    pipeline = _pipeline(client, local=local)

    results = pipeline.run([Task(task_id="m", prompt="Calculate 15% of 240.")])

    assert results == [{"task_id": "m", "answer": "36"}]
    assert len(local.calls) == 1
    assert len(client.calls) >= 1  # escalated


def test_local_exception_escalates():
    client = FakeClient({SMALL_MODEL: ["fallback answer"]})
    local = FakeLocal(RuntimeError("model blew up"))
    pipeline = _pipeline(client, local=local)

    results = pipeline.run([Task(task_id="q", prompt="Capital of France?")])

    assert results == [{"task_id": "q", "answer": "fallback answer"}]


@pytest.mark.parametrize("reason", ["length", "timeout"])
def test_truncated_local_answer_escalates(reason):
    # A truncated answer is a fragment: this one would sail through the MATH
    # verifier (it contains a number) but the derivation was cut off mid-way, so
    # trusting it would silently ship a wrong answer. Truncation must escalate
    # regardless of how well-formed the fragment looks.
    client = FakeClient({SMALL_MODEL: ["36"], LARGE_MODEL: ["36"]})
    local = FakeLocal("First, 10% of 240 is 24, so 5% is 12", finish_reason=reason)
    pipeline = _pipeline(client, local=local)

    results = pipeline.run([Task(task_id="m", prompt="Calculate 15% of 240.")])

    assert results == [{"task_id": "m", "answer": "36"}]
    assert len(client.calls) >= 1  # escalated despite passing _has_number


def test_local_calls_are_bounded_by_config():
    # The per-category cap is sized for Fireworks (1024 for math); locally that
    # is minutes of CPU decoding, so it must be clamped to local_max_tokens and
    # carry a wall-clock timeout.
    client = FakeClient({LARGE_MODEL: ["42"]})
    local = FakeLocal("Answer: 42")
    pipeline = _pipeline(client, local=local, local_max_tokens=64, local_task_timeout=9)

    pipeline.run([Task(task_id="m", prompt="Calculate 15% of 240.")])

    assert local.max_tokens == [64]
    assert local.timeouts == [9.0]


def test_local_budget_exhaustion_escalates_the_rest():
    # The local phase is serialized, so its cost is the SUM over tasks. Once the
    # budget is spent the remaining tasks must go straight to Fireworks rather
    # than run the container past its hard limit.
    client = FakeClient({SMALL_MODEL: ["from fireworks"] * 5})
    local = FakeLocal("Paris", sleep=0.05)
    pipeline = _pipeline(client, local=local, local_budget=1, local_task_timeout=1)

    tasks = [Task(task_id=f"t{i}", prompt=f"Capital of country {i}?") for i in range(5)]
    # Budget is 1s and the phase stops with <=1s left, so no local call fits.
    results = pipeline.run(tasks)

    assert [r["task_id"] for r in results] == [f"t{i}" for i in range(5)]
    assert local.calls == []
    assert len(client.calls) == 5


def test_local_phase_runs_cheapest_tasks_first():
    # Ordering by token cap fits the most tasks into a fixed budget, which is
    # what maximises the Fireworks tokens saved.
    client = FakeClient({})
    local = FakeLocal("positive")
    pipeline = _pipeline(client, local=local)

    tasks = [
        Task(task_id="math", prompt="Calculate 15% of 240."),          # cap 1024
        Task(task_id="sent", prompt="What is the sentiment of: great!"),  # cap 256
    ]
    pipeline.run(tasks)

    # Sentiment (cheapest) is attempted before math, despite input order.
    assert local.calls[0].startswith("What is the sentiment")
    assert len(client.calls) >= 1


@pytest.mark.parametrize("raw,expected_id,expected_prompt", [
    ({"task_id": "a", "prompt": "hello"}, "a", "hello"),
    ({"prompt": "no id"}, "task_0", "no id"),
    ({"task_id": "b"}, "b", ""),
    ({"task_id": 123, "prompt": None}, "123", ""),
    ("not a dict", "task_0", ""),
])
def test_normalize_tasks_tolerates_bad_input(raw, expected_id, expected_prompt):
    tasks = normalize_tasks([raw])
    assert len(tasks) == 1
    assert tasks[0].task_id == expected_id
    assert tasks[0].prompt == expected_prompt


def test_normalize_tasks_stable_indexes_across_mixed_input():
    tasks = normalize_tasks([
        {"task_id": "keep-me", "prompt": "one"},
        "garbage",
        {"prompt": "no id here"},
    ])
    assert [t.task_id for t in tasks] == ["keep-me", "task_1", "task_2"]
    assert [t.prompt for t in tasks] == ["one", "", "no id here"]
