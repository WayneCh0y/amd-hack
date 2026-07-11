"""Orchestration: turn a list of tasks into a list of answers.

Each task is classified (no tokens) into a category, which fixes its policy:
model tier, output-token cap, temperature. Work then runs in two phases, because
the two backends have opposite performance shapes:

  * **Local phase — sequential, hard-bounded.** llama.cpp shares one context, so
    local generations are serialized no matter how many threads call in: the cost
    of the phase is the SUM over tasks, not the max. On the 2-vCPU grading box a
    single verbose answer takes minutes, so the phase runs under a wall-clock
    budget with a per-task timeout, cheapest tasks first. Whatever the budget
    doesn't reach just falls through to Fireworks — escalating is always safe.
  * **Fireworks phase — concurrent.** These calls are IO-bound and genuinely
    parallel, so everything left over is fanned out across a thread pool.

Mixing the two in one pool (the earlier design) was the worst of both: threads
queued on the local lock, which serialized the API calls too.

Robustness guarantees:
  * every input ``task_id`` gets exactly one result entry, in input order;
  * a failed primary call retries once on the other model tier before giving up;
  * once the wall-clock budget is exhausted, remaining tasks return a safe
    fallback instead of risking the container being killed mid-write.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .categories import Category, CategoryPolicy, Tier, policy_for
from .config import Config
from .fireworks_client import FireworksClient
from .local_model import LocalModel
from .model_selector import ModelSelector
from .prompts import system_prompt_for
from .router import classify
from .verifiers import is_trustworthy

logger = logging.getLogger(__name__)

# Answer returned when every attempt fails or the time budget is gone. Keeps
# results.json valid and complete; an empty string is a safe, neutral value.
_FALLBACK_ANSWER = ""

# Assumed cost of a local task before we've measured one on this machine. Roughly
# the cheapest task observed on a 2-vCPU box, which is dominated by the fixed
# prompt prefill rather than by how long the answer is.
_LOCAL_TASK_FLOOR = 15.0


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str


@dataclass(frozen=True)
class _Plan:
    """A task plus everything classification decided about it. Built once so the
    local and Fireworks phases don't each redo the routing."""

    index: int
    task: Task
    category: Category
    policy: CategoryPolicy
    system: str


def normalize_tasks(raw_tasks: list[dict]) -> list[Task]:
    """Coerce parsed JSON into ``Task`` objects, tolerating missing fields."""
    tasks: list[Task] = []
    for i, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            tasks.append(Task(task_id=f"task_{i}", prompt=""))
            continue
        task_id = item.get("task_id")
        task_id = str(task_id) if task_id is not None else f"task_{i}"
        prompt = item.get("prompt") or ""
        tasks.append(Task(task_id=task_id, prompt=str(prompt)))
    return tasks


class Pipeline:
    def __init__(
        self,
        config: Config,
        client: FireworksClient,
        selector: ModelSelector,
        local: LocalModel | None = None,
        started_at: float | None = None,
    ):
        self._config = config
        self._client = client
        self._selector = selector
        # When present, tasks are answered locally first (zero Fireworks tokens)
        # and only escalated to the API when the local answer fails verification.
        self._local = local
        # Process start, so ``time_budget`` covers everything the container is
        # charged for — including the model load that happens before run(). Left
        # to run() when absent, which is what unit tests want.
        self._started_at = started_at

    def _model_for(self, tier: Tier) -> str:
        return self._selector.small() if tier is Tier.SMALL else self._selector.large()

    def _try_local(self, plan: _Plan, timeout: float) -> str | None:
        """Return a trusted local answer, or None to signal escalation."""
        task = plan.task
        # Cap output tokens well below the Fireworks policy: on CPU those caps
        # translate to minutes of decoding, not seconds.
        max_tokens = min(plan.policy.max_tokens, self._config.local_max_tokens)
        try:
            answer, usage = self._local.complete_with_usage(
                system=plan.system,
                user=task.prompt,
                max_tokens=max_tokens,
                temperature=plan.policy.temperature,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - any local failure escalates
            logger.warning("Local model failed on task %s: %s; escalating", task.task_id, exc)
            return None

        # A truncated answer is a fragment, not an answer — and a fragment can
        # still pass the structural verifiers (a cut-off math derivation contains
        # a number). Escalate on truncation before trusting the content at all.
        if usage.truncated:
            logger.info(
                "Local answer for task %s (%s) hit the %s cap; escalating",
                task.task_id,
                plan.category.value,
                usage.finish_reason,
            )
            return None

        if answer and is_trustworthy(plan.category, task.prompt, answer):
            return answer
        logger.info(
            "Local answer for task %s (%s) failed verification; escalating",
            task.task_id,
            plan.category.value,
        )
        return None

    def _run_local_phase(self, plans: list[_Plan], deadline: float) -> dict[int, str]:
        """Answer as many tasks as the local budget allows; return {index: answer}.

        Cheapest-first, by token cap: most of a task's local cost is a fixed
        prompt prefill, but the decode tail scales with the cap, so the cheap
        categories fit the most tasks — and therefore save the most Fireworks
        tokens — into a fixed budget. Tasks the budget doesn't reach are simply
        absent from the result and get escalated by the caller.

        A task is only started if the budget can plausibly *finish* it, estimated
        from what tasks have actually cost so far. Abandoning a generation
        part-way is pure waste: the prefill is already paid and yields nothing.
        """
        budget_end = min(deadline, time.monotonic() + self._config.local_budget)
        answers: dict[int, str] = {}
        durations: list[float] = []

        for plan in sorted(plans, key=lambda p: (p.policy.max_tokens, p.index)):
            remaining = budget_end - time.monotonic()
            # Until we've measured this box, assume a task costs the floor we saw
            # on 2 vCPUs; after that, trust the running mean.
            expected = sum(durations) / len(durations) if durations else _LOCAL_TASK_FLOOR
            if remaining < expected:
                logger.info(
                    "Local budget spent after %d/%d task(s) (%.0fs left, ~%.0fs needed); "
                    "rest go to Fireworks",
                    len(durations),
                    len(plans),
                    remaining,
                    expected,
                )
                break

            started = time.monotonic()
            # Never let one task overrun what's left of the phase.
            timeout = min(float(self._config.local_task_timeout), remaining)
            answer = self._try_local(plan, timeout)
            durations.append(time.monotonic() - started)
            if answer is not None:
                answers[plan.index] = answer

        logger.info(
            "Local phase: %d/%d task(s) answered locally in %.0fs",
            len(answers),
            len(plans),
            sum(durations),
        )
        return answers

    def _answer_via_fireworks(self, task: Task, system: str, policy) -> str:
        primary_tier = policy.tier
        alt_tier = Tier.LARGE if primary_tier is Tier.SMALL else Tier.SMALL

        for tier in (primary_tier, alt_tier):
            model = self._model_for(tier)
            try:
                answer = self._client.complete(
                    model=model,
                    system=system,
                    user=task.prompt,
                    max_tokens=policy.max_tokens,
                    temperature=policy.temperature,
                )
                if answer:
                    return answer
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Task %s failed on %s model %s: %s",
                    task.task_id,
                    tier.value,
                    model,
                    exc,
                )
            # If the two tiers resolve to the same model, don't retry it.
            if self._model_for(alt_tier) == self._model_for(primary_tier):
                break

        logger.error("Task %s produced no answer; using fallback", task.task_id)
        return _FALLBACK_ANSWER

    def _answer_via_fireworks_guarded(self, plan: _Plan, deadline: float) -> str:
        if time.monotonic() >= deadline:
            logger.warning("Time budget exhausted; skipping task %s", plan.task.task_id)
            return _FALLBACK_ANSWER
        return self._answer_via_fireworks(plan.task, plan.system, plan.policy)

    def run(self, tasks: list[Task]) -> list[dict]:
        """Answer every task and return result dicts in input order."""
        started = self._started_at if self._started_at is not None else time.monotonic()
        deadline = started + self._config.time_budget

        answers: dict[int, str] = {}
        plans: list[_Plan] = []
        for index, task in enumerate(tasks):
            if not task.prompt.strip():
                answers[index] = _FALLBACK_ANSWER
                continue
            category = classify(task.prompt)
            plans.append(
                _Plan(
                    index=index,
                    task=task,
                    category=category,
                    policy=policy_for(category),
                    system=system_prompt_for(category),
                )
            )

        # Phase 1: local-first, at zero Fireworks cost. Serialized and hard-bounded.
        if self._local is not None and plans:
            answers.update(self._run_local_phase(plans, deadline))

        # Phase 2: everything the local phase didn't answer. IO-bound, so fan out.
        pending = [p for p in plans if p.index not in answers]
        if pending:
            logger.info("Escalating %d task(s) to Fireworks", len(pending))
            workers = min(self._config.max_concurrency, len(pending))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for plan, answer in zip(
                    pending,
                    pool.map(lambda p: self._answer_via_fireworks_guarded(p, deadline), pending),
                ):
                    answers[plan.index] = answer

        return [
            {"task_id": task.task_id, "answer": answers.get(i, _FALLBACK_ANSWER)}
            for i, task in enumerate(tasks)
        ]
