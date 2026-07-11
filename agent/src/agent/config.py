"""Runtime configuration, loaded and validated from environment variables.

The judging harness injects ``FIREWORKS_API_KEY``, ``FIREWORKS_BASE_URL`` and
``ALLOWED_MODELS`` at evaluation time. Per the competition rules we read these
purely from the environment and never hardcode keys, URLs or model IDs.

Everything else (paths, timeouts, concurrency) has a sensible default but stays
overridable by env so nothing is baked into the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Required environment variable {name!r} is missing or empty. "
            "It is injected by the judging harness at evaluation time."
        )
    return value


def _split_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.split(",")]
    models = [m for m in models if m]
    if not models:
        raise ConfigError("ALLOWED_MODELS did not contain any model IDs.")
    return models


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of everything the agent needs to run."""

    api_key: str
    base_url: str
    allowed_models: list[str] = field(default_factory=list)

    input_path: str = "/input/tasks.json"
    output_path: str = "/output/results.json"

    # Per-request timeout (seconds). Kept comfortably under the 30s/request limit.
    request_timeout: int = 25
    # Number of tasks processed in parallel. Threads are fine: work is IO-bound.
    max_concurrency: int = 8
    # Retries per request on transient errors (429/5xx/timeouts). Each attempt can
    # cost a full `request_timeout`, and the pipeline tries two model tiers, so the
    # worst case per task is 2 * (max_retries + 1) * request_timeout. At the old
    # value of 2 that was 150s for ONE task — "excessive retries" is exactly what
    # the guide lists as a cause of TIMEOUT. Attempts are also deadline-clamped now
    # (see fireworks_client), so this is a second line of defence rather than the
    # only one.
    max_retries: int = 1
    # Soft wall-clock budget (seconds): past this, no new inference is *started*
    # and outstanding work is cut short. It is not a hard stop — see hard_budget.
    time_budget: int = 420
    # Hard wall-clock budget (seconds). At this point the watchdog writes whatever
    # answers exist and exits 0, no matter what the pipeline is doing.
    #
    # This is the load-bearing guarantee against the `TIMEOUT` status, which scores
    # zero even when most answers were already in hand. Everything below it (the
    # deadline-aware retries, the bounded local phase) exists to make sure the
    # watchdog rarely fires; the watchdog exists because "rarely" is not "never" —
    # a llama.cpp prefill cannot be interrupted from Python, so no amount of
    # in-pipeline bookkeeping can bound a hung generation.
    #
    # 510s leaves ~90s of margin under the 10-minute (600s) limit — for the
    # container's own startup, which the grader's clock includes but ours doesn't,
    # and for a grading box slower than the one we measured. The margin is close to
    # free: a healthy 19-task run finishes in ~200s, so these budgets only bind when
    # something has already gone wrong, and the failure they trade against (TIMEOUT,
    # which scores zero) is far worse than the handful of answers a lower ceiling
    # might cost.
    hard_budget: int = 510
    # Reasoning effort for models that support it (e.g. gpt-oss). Lower effort
    # spends far fewer hidden "thinking" tokens — the main token-efficiency lever
    # when the allowed models reason by default. Empty string disables the param.
    # If a model rejects it, the client transparently retries without it.
    reasoning_effort: str = "low"

    # Local answering (zero Fireworks tokens), restricted to the categories that
    # have a real verifier — code and summarization; see categories.LOCAL_OK.
    # Auto-disabled at startup if the weights aren't present, so the agent
    # degrades gracefully to Fireworks-only.
    local_enabled: bool = True
    # Wall-clock ceiling for the whole local phase. Local generation is serialized
    # (one llama.cpp context), so the phase costs the SUM over tasks, not the max.
    # Measured on 2 vCPU / 4 GB — the grading box's shape — a task costs ~45s,
    # dominated by prompt prefill rather than by answer length.
    #
    # The budget makes that safe without having to predict the grading hardware:
    # we answer as many eligible tasks locally as it allows and escalate the rest,
    # so a slower CPU costs Fireworks tokens rather than a TIMEOUT (the failure
    # that scores zero). The phase now runs *concurrently* with the Fireworks
    # calls, so it no longer delays them.
    # sized so that even a fully-spent local phase leaves the Fireworks phase far
    # more room than it needs (a healthy Fireworks answer takes ~3s). At ~45s per
    # task this answers ~3 tasks for free; the old 300s bought maybe 3 more at the
    # cost of pushing every escalation into the back half of the container's life.
    local_budget: int = 150
    # Backstop for a single local generation. Above the measured per-task cost so
    # it does not fire on healthy tasks: a truncated answer is escalated, which
    # means we pay the local time AND the Fireworks tokens — the worst outcome.
    local_task_timeout: int = 45
    # Output-token ceiling for local answers. The per-category caps are sized for
    # Fireworks (1024 for code); locally that is ~2 minutes of decode. 384 leaves
    # room for a full function or a summary without truncating.
    local_max_tokens: int = 384
    # Prompt-size ceiling for local answering, in characters (~4 chars/token).
    #
    # Local cost is dominated by *prefill*, which scales with the prompt and which
    # we cannot interrupt: llama.cpp yields no token — and so gives us no chance to
    # check the clock — until the whole prompt has been evaluated. `local_task_timeout`
    # therefore does not bound prefill, and a long summarization passage can blow
    # through it unchecked. Bounding the input is the only bound that actually
    # holds. Oversized prompts skip the local phase and go straight to Fireworks.
    local_max_prompt_chars: int = 6000

    @classmethod
    def from_env(cls) -> "Config":
        """Build a ``Config`` from the process environment, validating as we go."""
        return cls(
            api_key=_require("FIREWORKS_API_KEY"),
            base_url=_require("FIREWORKS_BASE_URL"),
            allowed_models=_split_models(_require("ALLOWED_MODELS")),
            input_path=os.environ.get("INPUT_PATH", "/input/tasks.json").strip()
            or "/input/tasks.json",
            output_path=os.environ.get("OUTPUT_PATH", "/output/results.json").strip()
            or "/output/results.json",
            request_timeout=_env_int("REQUEST_TIMEOUT", 25),
            max_concurrency=max(1, _env_int("MAX_CONCURRENCY", 8)),
            max_retries=max(0, _env_int("MAX_RETRIES", 1)),
            time_budget=max(30, _env_int("TIME_BUDGET", 420)),
            hard_budget=max(30, _env_int("HARD_TIME_BUDGET", 510)),
            reasoning_effort=os.environ.get("REASONING_EFFORT", "low").strip(),
            local_enabled=_env_bool("LOCAL_MODEL_ENABLED", True),
            local_budget=max(0, _env_int("LOCAL_TIME_BUDGET", 150)),
            local_task_timeout=max(1, _env_int("LOCAL_TASK_TIMEOUT", 45)),
            local_max_tokens=max(1, _env_int("LOCAL_MAX_TOKENS", 384)),
            local_max_prompt_chars=max(1, _env_int("LOCAL_MAX_PROMPT_CHARS", 6000)),
        )
