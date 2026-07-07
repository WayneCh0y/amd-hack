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

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """Run one chat completion and return the assistant text.

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
                )
                self._record_usage(response)
                content = response.choices[0].message.content
                return (content or "").strip()
            except Exception as exc:  # noqa: BLE001 - retry on any transient error
                last_exc = exc
                if attempt < attempts - 1:
                    # Exponential backoff: 0.5s, 1s, 2s, ... capped at 4s.
                    time.sleep(min(0.5 * (2**attempt), 4.0))

        assert last_exc is not None
        raise last_exc

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.meter.add(
            Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        )
