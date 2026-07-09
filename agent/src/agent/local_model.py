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
    tracked only for benchmarking (latency / sizing)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
    ) -> str:
        """Run one chat completion locally and return the assistant text."""
        text, _ = self.complete_with_usage(
            system=system, user=user, max_tokens=max_tokens, temperature=temperature
        )
        return text

    def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> tuple[str, LocalUsage]:
        """Like :meth:`complete` but also returns local token usage."""
        self.load()
        assert self._llm is not None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # llama.cpp shares one context: serialize generations.
        with self._gen_lock:
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
        )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
