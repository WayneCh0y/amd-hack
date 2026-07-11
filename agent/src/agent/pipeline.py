"""Orchestration: turn a list of tasks into a list of answers.

Each task is classified (no tokens) into a category, which fixes its policy:
model tier, output-token cap, temperature, and — the decision that matters most —
whether the bundled local model is allowed to answer it at all.

**Only categories with a real verifier are answered locally.** The local model is
accurate on code and summarization and can be checked there (code is executed;
a summary is checked against the length constraint the prompt states). On
factual / math / sentiment / NER / logic it produces fluent, well-formed, wrong
answers that no ground-truth-free verifier can catch, so those go straight to
Fireworks. Trying every category locally and "verifying" it with a shape check is
what failed the 80% accuracy gate; see ``categories.LOCAL_OK``.

Work runs in two overlapping phases, because the backends have opposite shapes:

  * **Fireworks — concurrent, IO-bound.** Tasks the local model may not answer are
    dispatched to a thread pool immediately, and run *while* the local phase works.
  * **Local — sequential, CPU-bound, hard-bounded.** llama.cpp shares one context,
    so local generations serialize: the phase costs the SUM over tasks, not the
    max. On the 2-vCPU grading box one answer takes ~45 s, so the phase runs under
    a wall-clock budget, cheapest tasks first. Whatever the budget doesn't reach
    falls through to Fireworks — escalating is always safe.

Overlapping them keeps the local phase off the critical path: llama.cpp releases
the GIL while decoding, so the API calls genuinely progress alongside it.

Robustness guarantees:
  * every input ``task_id`` gets exactly one result entry, in input order;
  * a failed primary call retries once on the other model tier;
  * **no task ever ships an empty answer while we hold any text for it.** An empty
    answer is graded wrong with certainty; an unverified local draft is merely
    likely wrong. If Fireworks fails, we fall back to the draft, and as a last
    resort generate one locally.
  * **every wait is bounded by the run deadline** — not just checked against it
    before starting. Both backends take the deadline and enforce it *inside* their
    own loops (the Fireworks retry ladder; the queue for the single llama.cpp
    context). Checking the clock and then entering an unbounded wait is what ran
    the container past its 10-minute limit and scored a ``TIMEOUT``.
  * answers are published to ``results_for`` as they land, so the watchdog in
    ``main`` can write a complete, schema-valid file at any instant.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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

# Answer of last resort: only used when we hold no text at all for a task.
_FALLBACK_ANSWER = ""

# Assumed cost of a local task before we've measured one on this machine.
# Measured on 2 vCPU / 4 GB (the grading box's shape): ~45 s per task, dominated
# by prompt prefill rather than by answer length. The old value of 15 s was
# optimistic enough that the phase kept starting tasks it could not finish.
_LOCAL_TASK_FLOOR = 45.0


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
        # When present, the local-eligible categories are answered here first
        # (zero Fireworks tokens) and escalated only when verification fails.
        self._local = local
        # Process start, so ``time_budget`` covers everything the container is
        # charged for — including the model load that happens before run(). Left
        # to run() when absent, which is what unit tests want.
        self._started_at = started_at
        # Answers published as soon as each one lands, plus whatever the local
        # model produced (trusted or not). Both are read by the watchdog thread
        # while run() is still working, so both are behind a lock.
        self._answers: dict[int, str] = {}
        self._drafts: dict[int, str] = {}
        self._results_lock = threading.Lock()

    def attach_local(self, local: LocalModel | None) -> None:
        """Supply the local model after construction.

        ``main`` builds the pipeline *before* loading the ~1.9 GB GGUF so that the
        watchdog is already armed while the load runs — a load that stalls would
        otherwise be a hang with no results file to show for it.
        """
        self._local = local

    def _model_for(self, tier: Tier) -> str:
        return self._selector.small() if tier is Tier.SMALL else self._selector.large()

    # -- partial results (read by the watchdog, mid-run) ----------------------

    def _publish(self, index: int, answer: str) -> None:
        with self._results_lock:
            self._answers[index] = answer

    def _record_draft(self, index: int, draft: str) -> None:
        with self._results_lock:
            self._drafts[index] = draft

    def results_for(self, tasks: list[Task]) -> list[dict]:
        """Result entries for every task, from whatever has been answered so far.

        Complete by construction — a task with no answer yet gets its local draft,
        or the empty fallback — so this is always a schema-valid results.json. Safe
        to call from another thread at any point in the run, which is the whole
        point: it is what the watchdog writes when the clock runs out.
        """
        with self._results_lock:
            answers = dict(self._answers)
            drafts = dict(self._drafts)
        return [
            {
                "task_id": task.task_id,
                "answer": answers.get(i) or drafts.get(i) or _FALLBACK_ANSWER,
            }
            for i, task in enumerate(tasks)
        ]

    # -- local phase ---------------------------------------------------------

    def _try_local(self, plan: _Plan, timeout: float, deadline: float) -> str | None:
        """Return a *verified* local answer, or None to signal escalation.

        Any text the model produced is recorded in ``_drafts`` even when we reject
        it, so a later Fireworks failure can fall back to it rather than to "".
        """
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
                deadline=deadline,
            )
        except Exception as exc:  # noqa: BLE001 - any local failure escalates
            logger.warning("Local model failed on task %s: %s; escalating", task.task_id, exc)
            return None

        if answer:
            self._record_draft(plan.index, answer)

        # A truncated answer is a fragment, not an answer, and a fragment can still
        # satisfy a verifier. Escalate on truncation before judging the content.
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

    def _run_local_phase(
        self,
        plans: list[_Plan],
        deadline: float,
        pool: ThreadPoolExecutor,
        futures: dict[int, Future[str]],
    ) -> None:
        """Answer as many tasks as the local budget allows, escalating the rest.

        Cheapest-first, by token cap: most of a task's local cost is a fixed prompt
        prefill, but the decode tail scales with the cap, so the cheap categories
        fit the most tasks — and therefore save the most Fireworks tokens — into a
        fixed budget.

        A task is only started if the budget can plausibly *finish* it, estimated
        from what tasks have actually cost so far. Abandoning a generation part-way
        is pure waste: the prefill is already paid and yields nothing.

        Every task that isn't answered locally — verification failed, generation
        failed, or the budget didn't reach it — is submitted to ``pool`` **the
        moment we know**, not banked until the phase ends. Escalations used to wait
        for the whole (then 300s) phase to finish, which stacked the entire
        Fireworks tail into the back half of the container's life. Now it overlaps:
        llama.cpp releases the GIL while decoding, so those calls make real progress
        alongside the local work.
        """
        budget_end = min(deadline, time.monotonic() + self._config.local_budget)
        ordered = sorted(plans, key=lambda p: (p.policy.max_tokens, p.index))
        durations: list[float] = []
        answered = 0

        for position, plan in enumerate(ordered):
            remaining = budget_end - time.monotonic()
            # Until we've measured this box, assume a task costs the floor we saw
            # on 2 vCPUs; after that, trust the running mean.
            expected = sum(durations) / len(durations) if durations else _LOCAL_TASK_FLOOR
            if remaining < expected:
                logger.info(
                    "Local budget spent after %d/%d task(s) (%.0fs left, ~%.0fs needed); "
                    "the remaining %d go to Fireworks",
                    len(durations),
                    len(plans),
                    remaining,
                    expected,
                    len(ordered) - position,
                )
                for skipped in ordered[position:]:
                    futures[skipped.index] = pool.submit(
                        self._answer_guarded, skipped, deadline
                    )
                break

            started = time.monotonic()
            # Never let one task overrun what's left of the phase.
            timeout = min(float(self._config.local_task_timeout), remaining)
            answer = self._try_local(plan, timeout, budget_end)
            durations.append(time.monotonic() - started)

            if answer is not None:
                self._publish(plan.index, answer)
                answered += 1
            else:
                futures[plan.index] = pool.submit(self._answer_guarded, plan, deadline)

        logger.info(
            "Local phase: %d/%d eligible task(s) answered locally in %.0fs (0 tokens)",
            answered,
            len(plans),
            sum(durations),
        )

    # -- Fireworks phase -----------------------------------------------------

    def _answer_via_fireworks(self, plan: _Plan, deadline: float) -> str:
        """Try the category's tier, then the other one. Bounded by ``deadline``.

        The deadline goes all the way down into the client, which clamps each
        attempt's HTTP timeout to the time actually left and stops retrying when
        there is none. Without that, this ladder (2 tiers x N attempts x
        request_timeout) is the longest tail in the agent.
        """
        primary_tier = plan.policy.tier
        alt_tier = Tier.LARGE if primary_tier is Tier.SMALL else Tier.SMALL

        for tier in (primary_tier, alt_tier):
            if time.monotonic() >= deadline:
                logger.warning(
                    "Out of time before the %s tier for task %s",
                    tier.value,
                    plan.task.task_id,
                )
                break
            model = self._model_for(tier)
            try:
                answer = self._client.complete(
                    model=model,
                    system=plan.system,
                    user=plan.task.prompt,
                    max_tokens=plan.policy.max_tokens,
                    temperature=plan.policy.temperature,
                    deadline=deadline,
                )
                if answer:
                    return answer
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Task %s failed on %s model %s: %s",
                    plan.task.task_id,
                    tier.value,
                    model,
                    exc,
                )
            # If the two tiers resolve to the same model, don't retry it.
            if self._model_for(alt_tier) == self._model_for(primary_tier):
                break

        return _FALLBACK_ANSWER

    def _answer_guarded(self, plan: _Plan, deadline: float) -> str:
        """Fireworks answer for ``plan``, never returning empty if we can help it.

        Precedence: a Fireworks answer, then any local draft we already hold, then
        one unverified local generation. An empty answer is graded wrong with
        certainty, so it is strictly worse than an unverified guess — the only
        thing that can beat "no answer" is *some* answer.
        """
        if time.monotonic() < deadline:
            answer = self._answer_via_fireworks(plan, deadline)
            if answer:
                return answer
            logger.error("Task %s produced no Fireworks answer", plan.task.task_id)
        else:
            logger.warning("Time budget exhausted; skipping task %s", plan.task.task_id)

        with self._results_lock:
            draft = self._drafts.get(plan.index)
        if draft:
            logger.info("Task %s falling back to the unverified local draft", plan.task.task_id)
            return draft

        return self._last_resort_local(plan, deadline)

    def _last_resort_local(self, plan: _Plan, deadline: float) -> str:
        """One unverified local generation, when Fireworks gave us nothing at all.

        Only reachable when the API is failing outright — which is exactly when
        *every* task takes this path at once. They then queue on the single
        llama.cpp context at ~45s each, so the deadline has to bound the wait as
        well as the generation; checking the clock here and then blocking
        indefinitely on the lock is what pushed the container past 10 minutes.
        ``complete_with_usage`` re-checks ``deadline`` once it holds the lock and
        raises rather than start a generation whose time has passed.
        """
        if self._local is None:
            return _FALLBACK_ANSWER
        remaining = deadline - time.monotonic()
        if remaining < _LOCAL_TASK_FLOOR:
            return _FALLBACK_ANSWER
        if not self._fits_local(plan):
            return _FALLBACK_ANSWER

        logger.warning(
            "Task %s: no Fireworks answer; attempting an unverified local answer",
            plan.task.task_id,
        )
        timeout = min(float(self._config.local_task_timeout), remaining)
        try:
            answer, _ = self._local.complete_with_usage(
                system=plan.system,
                user=plan.task.prompt,
                max_tokens=min(plan.policy.max_tokens, self._config.local_max_tokens),
                temperature=plan.policy.temperature,
                timeout=timeout,
                deadline=deadline,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Last-resort local answer failed for %s: %s", plan.task.task_id, exc)
            return _FALLBACK_ANSWER
        return answer or _FALLBACK_ANSWER

    def _fits_local(self, plan: _Plan) -> bool:
        """False if the prompt is too long to prefill inside the local timeout.

        Local cost is dominated by prefill, and prefill cannot be interrupted (see
        ``local_model._generate_bounded``), so ``local_task_timeout`` does not bound
        it. Bounding the input is the only bound that holds.
        """
        size = len(plan.system) + len(plan.task.prompt)
        if size <= self._config.local_max_prompt_chars:
            return True
        logger.info(
            "Task %s prompt is %d chars (> %d); too big to prefill locally in time",
            plan.task.task_id,
            size,
            self._config.local_max_prompt_chars,
        )
        return False

    # -- entry point ---------------------------------------------------------

    def run(self, tasks: list[Task]) -> list[dict]:
        """Answer every task and return result dicts in input order."""
        started = self._started_at if self._started_at is not None else time.monotonic()
        deadline = started + self._config.time_budget

        plans: list[_Plan] = []
        for index, task in enumerate(tasks):
            if not task.prompt.strip():
                self._publish(index, _FALLBACK_ANSWER)
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

        # Only categories with a real verifier may be answered locally — and only
        # if the prompt is small enough to prefill inside the local timeout.
        local_plans = (
            [p for p in plans if p.policy.local_ok and self._fits_local(p)]
            if self._local is not None
            else []
        )
        local_indexes = {p.index for p in local_plans}
        api_plans = [p for p in plans if p.index not in local_indexes]

        if plans:
            logger.info(
                "Routing %d task(s): %d local-eligible (%s), %d to Fireworks",
                len(plans),
                len(local_plans),
                ", ".join(sorted({p.category.value for p in local_plans})) or "none",
                len(api_plans),
            )

        workers = max(1, min(self._config.max_concurrency, len(plans) or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Fireworks work starts now and runs *alongside* the local phase:
            # llama.cpp releases the GIL while decoding, so these calls make real
            # progress instead of queueing behind it.
            futures: dict[int, Future[str]] = {
                p.index: pool.submit(self._answer_guarded, p, deadline) for p in api_plans
            }

            # Local phase: serialized and CPU-bound, so it runs in this thread. It
            # submits its own escalations into `futures` as they arise.
            self._run_local_phase(local_plans, deadline, pool, futures)

            for index, future in futures.items():
                self._publish(index, future.result())

        return self.results_for(tasks)
