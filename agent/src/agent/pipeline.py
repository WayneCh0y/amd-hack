"""Orchestration: turn a list of tasks into a list of answers.

For each task we classify it (no tokens), look up its policy, pick the right
model tier, and call Fireworks. Tasks run concurrently within a bounded thread
pool so large batches finish inside the runtime budget. Robustness guarantees:

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

from .categories import Category, Tier, policy_for
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


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str


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
    ):
        self._config = config
        self._client = client
        self._selector = selector
        # When present, tasks are answered locally first (zero Fireworks tokens)
        # and only escalated to the API when the local answer fails verification.
        self._local = local

    def _model_for(self, tier: Tier) -> str:
        return self._selector.small() if tier is Tier.SMALL else self._selector.large()

    def _answer_one(self, task: Task, deadline: float) -> str:
        if not task.prompt.strip():
            return _FALLBACK_ANSWER
        if time.monotonic() >= deadline:
            logger.warning("Time budget exhausted; skipping task %s", task.task_id)
            return _FALLBACK_ANSWER

        category = classify(task.prompt)
        policy = policy_for(category)
        system = system_prompt_for(category)

        # 1) Local-first: answer with the bundled model at zero Fireworks cost.
        #    Keep it only if it passes verification; otherwise fall through to
        #    the API. This is where the token savings come from.
        if self._local is not None:
            local_answer = self._try_local(task, category, system, policy)
            if local_answer is not None:
                return local_answer

        # 2) Escalate to Fireworks.
        return self._answer_via_fireworks(task, system, policy)

    def _try_local(self, task: Task, category: Category, system: str, policy) -> str | None:
        """Return a trusted local answer, or None to signal escalation."""
        try:
            answer = self._local.complete(
                system=system,
                user=task.prompt,
                max_tokens=policy.max_tokens,
                temperature=policy.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - any local failure escalates
            logger.warning("Local model failed on task %s: %s; escalating", task.task_id, exc)
            return None

        if answer and is_trustworthy(category, task.prompt, answer):
            return answer
        logger.info(
            "Local answer for task %s (%s) failed verification; escalating",
            task.task_id,
            category.value,
        )
        return None

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

    def run(self, tasks: list[Task]) -> list[dict]:
        """Process all tasks concurrently and return ordered result dicts."""
        deadline = time.monotonic() + self._config.time_budget
        results: list[dict | None] = [None] * len(tasks)

        def work(index_task: tuple[int, Task]) -> tuple[int, str]:
            index, task = index_task
            return index, self._answer_one(task, deadline)

        workers = min(self._config.max_concurrency, max(1, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, answer in pool.map(work, enumerate(tasks)):
                results[index] = {"task_id": tasks[index].task_id, "answer": answer}

        # results is fully populated (map covers every index), but guard anyway.
        return [
            r if r is not None else {"task_id": tasks[i].task_id, "answer": _FALLBACK_ANSWER}
            for i, r in enumerate(results)
        ]
