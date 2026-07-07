"""Wrapper around the Fireworks AI chat-completions API.

Fireworks is OpenAI-compatible, so we use the official ``openai`` client pointed
at ``FIREWORKS_BASE_URL``. Every inference request goes through this base URL —
that is a hard competition requirement: calls that bypass it are not recorded by
the judging proxy and score zero tokens.

The client also:
  * retries transient failures with bounded exponential backoff, and
  * accumulates token usage across all calls (thread-safe) for reporting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from openai import OpenAI

from .config import Config


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


class TokenMeter:
    """Thread-safe accumulator of token usage across concurrent requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = Usage()

    def add(self, usage: Usage) -> None:
        with self._lock:
            self._total = self._total + usage

    @property
    def total(self) -> Usage:
        with self._lock:
            return self._total


class FireworksClient:
    """Minimal chat-completion client with retries and token metering."""

    def __init__(self, config: Config):
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
            max_retries=0,  # we manage retries ourselves for backoff control
        )
        self.meter = TokenMeter()
        # Models that rejected `reasoning_effort`; we stop sending it to them.
        self._no_reasoning_param: set[str] = set()
        self._no_reasoning_lock = threading.Lock()

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """Run one chat completion and return the assistant text."""
        text, _ = self._run(
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return text

    def complete_with_usage(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> tuple[str, Usage]:
        """Like :meth:`complete` but also returns this call's token usage.

        Used by the benchmark harness to measure per-call cost per model.
        """
        return self._run(
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _run(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, Usage]:
        """Single completion with retries; returns (text, usage).

        Raises the last exception if all attempts fail; callers decide how to
        fall back. Token usage from successful calls is always metered.
        """
        last_exc: Exception | None = None
        attempts = self._config.max_retries + 1

        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **self._reasoning_kwargs(model),
                )
                usage = self._usage_of(response)
                self.meter.add(usage)
                content = response.choices[0].message.content
                return (content or "").strip(), usage
            except Exception as exc:  # noqa: BLE001 - retry on any transient error
                # If the model rejected `reasoning_effort`, drop it and retry
                # immediately (doesn't consume a backoff attempt).
                if self._maybe_disable_reasoning(model, exc):
                    continue
                last_exc = exc
                if attempt < attempts - 1:
                    # Exponential backoff: 0.5s, 1s, 2s, ... capped at 4s.
                    time.sleep(min(0.5 * (2**attempt), 4.0))

        assert last_exc is not None
        raise last_exc

    def _reasoning_kwargs(self, model: str) -> dict:
        effort = self._config.reasoning_effort
        if not effort:
            return {}
        with self._no_reasoning_lock:
            if model in self._no_reasoning_param:
                return {}
        return {"reasoning_effort": effort}

    def _maybe_disable_reasoning(self, model: str, exc: Exception) -> bool:
        """Return True if ``exc`` is an unsupported-``reasoning_effort`` error.

        Records the model so future calls omit the param, and signals the caller
        to retry without counting it as a backoff attempt.
        """
        if "reasoning_effort" not in str(exc):
            return False
        with self._no_reasoning_lock:
            if model in self._no_reasoning_param:
                return False  # already dropped; a different error, don't loop
            self._no_reasoning_param.add(model)
        return True

    @staticmethod
    def _usage_of(response) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
