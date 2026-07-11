"""Bundled local model (llama.cpp / GGUF), run on CPU at zero Fireworks cost.

The single biggest ranking lever for Track 1 is answering tasks *locally*: only
tokens routed through ``FIREWORKS_BASE_URL`` count toward the token score, so a
task the local model answers correctly costs **zero** scored tokens. This module
wraps a bundled 2-3B 4-bit GGUF model via ``llama-cpp-python`` so it can run on
the 4 GB RAM / 2 vCPU grading box with no GPU and no external runtime.

Design notes:
  * **Lazy load.** The (~1.9 GB) weights load on first use, not at import time,
    so importing the package — and the Fireworks-only code path and unit tests —
    never needs the model file present.
  * **Single context, serialized.** llama.cpp is not safe for concurrent
    generation on one context, and the box only has 2 vCPUs anyway, so every
    generation is guarded by a lock. Local calls are effectively sequential.
  * **Deterministic.** Fixed seed + temperature 0 by default: these are
    objective tasks and reproducibility helps when comparing runs.
  * **Same call shape as FireworksClient.** ``complete(system, user, ...)`` so
    the pipeline can try local first and escalate to Fireworks symmetrically.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default location the Docker image bundles the weights at. Overridable by env
# so dev machines (which download into agent/models/) and CI can point elsewhere.
_DEFAULT_MODEL_PATH = "/models/model.gguf"


class LocalModelError(RuntimeError):
    """Raised when the local model can't be loaded or run."""


@dataclass(frozen=True)
class LocalUsage:
    """Local token counts. These do NOT count toward the competition score;
    tracked only for benchmarking (latency / sizing).

    ``finish_reason`` is ``"stop"`` for a complete answer, ``"length"`` when the
    token cap cut it off, and ``"timeout"`` when the wall-clock cap did. The two
    latter mean the text is a *fragment*: callers must escalate rather than trust
    it, since a truncated answer can still look structurally valid.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def truncated(self) -> bool:
        return self.finish_reason in ("length", "timeout")


class LocalModel:
    """Lazy-loaded, thread-safe wrapper around a GGUF chat model on CPU."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        seed: int = 0,
    ):
        self._model_path = model_path or os.environ.get(
            "LOCAL_MODEL_PATH", _DEFAULT_MODEL_PATH
        )
        # Context window: keep modest to bound RAM. Big enough for the longest
        # Track 1 prompts (summarization inputs); env-overridable.
        self._n_ctx = n_ctx or _env_int("LOCAL_N_CTX", 4096)
        # Match the grading box (2 vCPU). More threads than cores just thrashes.
        self._n_threads = n_threads or _env_int("LOCAL_N_THREADS", 2)
        self._seed = seed

        self._llm = None  # loaded lazily
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    @property
    def model_path(self) -> str:
        return self._model_path

    def available(self) -> bool:
        """True if the weights file exists (does not load it)."""
        return os.path.isfile(self._model_path)

    def load(self) -> None:
        """Load the weights if not already loaded. Idempotent, thread-safe."""
        if self._llm is not None:
            return
        with self._load_lock:
            if self._llm is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise LocalModelError(
                    "llama-cpp-python is not installed. Install it with "
                    "`pip install llama-cpp-python` (prebuilt CPU wheels: "
                    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu)."
                ) from exc
            if not self.available():
                raise LocalModelError(
                    f"Local model weights not found at {self._model_path!r}. "
                    "Set LOCAL_MODEL_PATH or place the GGUF file there."
                )
            logger.info(
                "Loading local model %s (n_ctx=%d, n_threads=%d)",
                self._model_path,
                self._n_ctx,
                self._n_threads,
            )
            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=0,  # CPU only
                seed=self._seed,
                verbose=False,
            )
            logger.info("Local model loaded.")

    # -- inference -----------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> str:
        """Run one chat completion locally and return the assistant text."""
        text, _ = self.complete_with_usage(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        return text

    def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> tuple[str, LocalUsage]:
        """Like :meth:`complete` but also returns local token usage.

        ``timeout`` bounds wall-clock generation. CPU decoding on the 2-vCPU
        grading box runs at single-digit tokens/sec, so an uncapped generation
        can run for minutes and eat the whole container budget. The clock starts
        when *this* call begins decoding, not when it queued behind ``_gen_lock``.
        """
        self.load()
        assert self._llm is not None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # llama.cpp shares one context: serialize generations.
        with self._gen_lock:
            if timeout and timeout > 0:
                return self._generate_bounded(messages, max_tokens, temperature, timeout)
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=self._seed,
            )

        choice = response["choices"][0]
        content = (choice.get("message", {}).get("content") or "").strip()
        usage = response.get("usage") or {}
        return content, LocalUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            finish_reason=choice.get("finish_reason") or "stop",
        )

    def _generate_bounded(
        self, messages: list[dict], max_tokens: int, temperature: float, timeout: float
    ) -> tuple[str, LocalUsage]:
        """Generate with a wall-clock ceiling, by streaming and cutting the stream.

        ``create_chat_completion`` takes no ``stopping_criteria`` (only the raw
        ``create_completion`` does), and a blocking call cannot be cancelled from
        another thread — it runs in C. Streaming is the one interruption point the
        chat API gives us: we check the clock between tokens and close the
        generator, which unwinds llama.cpp's sampling loop.

        Caller must already hold ``_gen_lock``.
        """
        deadline = time.monotonic() + timeout
        stream = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=self._seed,
            stream=True,
        )

        parts: list[str] = []
        completion_tokens = 0
        finish_reason = "stop"
        try:
            for chunk in stream:
                choice = chunk["choices"][0]
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    parts.append(piece)
                    completion_tokens += 1  # llama.cpp yields one token per chunk
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if time.monotonic() >= deadline:
                    finish_reason = "timeout"
                    break
        finally:
            stream.close()

        return "".join(parts).strip(), LocalUsage(
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
            finish_reason=finish_reason,
        )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
