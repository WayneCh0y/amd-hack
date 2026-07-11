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
    # Retries per request on transient errors (429/5xx/timeouts).
    max_retries: int = 2
    # Overall wall-clock budget (seconds). Kept under the 10-min hard limit so
    # we always have time to write results.json before the container is killed.
    time_budget: int = 540
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
    local_budget: int = 300
    # Backstop for a single local generation. Above the measured per-task cost so
    # it does not fire on healthy tasks: a truncated answer is escalated, which
    # means we pay the local time AND the Fireworks tokens — the worst outcome.
    local_task_timeout: int = 60
    # Output-token ceiling for local answers. The per-category caps are sized for
    # Fireworks (1024 for code); locally that is ~2 minutes of decode. 384 leaves
    # room for a full function or a summary without truncating.
    local_max_tokens: int = 384

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
            max_retries=max(0, _env_int("MAX_RETRIES", 2)),
            time_budget=max(30, _env_int("TIME_BUDGET", 540)),
            reasoning_effort=os.environ.get("REASONING_EFFORT", "low").strip(),
            local_enabled=_env_bool("LOCAL_MODEL_ENABLED", True),
            local_budget=max(0, _env_int("LOCAL_TIME_BUDGET", 300)),
            local_task_timeout=max(1, _env_int("LOCAL_TASK_TIMEOUT", 60)),
            local_max_tokens=max(1, _env_int("LOCAL_MAX_TOKENS", 384)),
        )
