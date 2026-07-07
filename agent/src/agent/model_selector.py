"""Pick which allowed model to use for a task, without hardcoding model IDs.

The exact ``ALLOWED_MODELS`` list is only published on launch day, so instead of
naming models we infer a rough *capability score* from the parameter count
encoded in each model ID (e.g. ``...-8b-...``, ``...-70b``, ``...-235b-a22b``,
``mixtral-8x7b``). Smaller models are cheaper (fewer tokens, faster) and go to
simple tasks; larger models go to hard ones.

The selector degrades gracefully: with a single allowed model both tiers return
it, and models with no parseable size get a neutral score so they are never
wrongly treated as the smallest or largest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Neutral score for models whose size we cannot parse. Sits between typical
# "small" (~7-8B) and "large" (~70B+) models so an unknown never dominates a tier.
_UNKNOWN_SIZE = 30.0

# "8x7b" style mixture-of-experts: total params ~= experts * expert_size.
_MOE_RE = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)
# Plain "<number>b" size marker (llama-8b, 70b, 235b, 120b, ...).
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)

# Substrings that mark "reasoning" models, which emit costly hidden thinking
# tokens. We avoid them for the cheap tier when a plain model is available.
_REASONING_MARKERS = ("r1", "reasoning", "thinking", "qwq", "-o1", "reasoner")


def _parse_size(model_id: str) -> float:
    """Best-effort parameter count (in billions) inferred from the model ID."""
    text = model_id.lower()

    sizes: list[float] = []
    for experts, expert_size in _MOE_RE.findall(text):
        sizes.append(int(experts) * float(expert_size))
    # Also collect plain "<n>b" markers (covers total-param counts like 235b).
    for match in _SIZE_RE.findall(text):
        sizes.append(float(match))

    if not sizes:
        return _UNKNOWN_SIZE
    # Use the largest marker found: for "235b-a22b" the total (235) reflects
    # capability better than the active-expert count (22).
    return max(sizes)


def _is_reasoning(model_id: str) -> bool:
    text = model_id.lower()
    return any(marker in text for marker in _REASONING_MARKERS)


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    size: float
    reasoning: bool
    order: int  # original position in ALLOWED_MODELS, for stable tie-breaks


class ModelSelector:
    """Chooses a small (cheap) or large (capable) model from the allowed list."""

    def __init__(self, allowed_models: list[str]):
        if not allowed_models:
            raise ValueError("allowed_models must be non-empty")
        self._models = [
            ModelInfo(
                model_id=m,
                size=_parse_size(m),
                reasoning=_is_reasoning(m),
                order=i,
            )
            for i, m in enumerate(allowed_models)
        ]

    def small(self) -> str:
        """Smallest model; prefer a non-reasoning one to keep token cost low."""
        best = min(
            self._models,
            key=lambda m: (m.size, m.reasoning, m.order),
        )
        return best.model_id

    def large(self) -> str:
        """Largest model; prefer non-reasoning on ties to avoid extra tokens."""
        best = max(
            self._models,
            key=lambda m: (m.size, not m.reasoning, -m.order),
        )
        return best.model_id

    def describe(self) -> str:
        """Human-readable summary for startup logging."""
        parts = [
            f"{m.model_id} (~{m.size:g}B{', reasoning' if m.reasoning else ''})"
            for m in sorted(self._models, key=lambda m: m.size)
        ]
        return "; ".join(parts)
