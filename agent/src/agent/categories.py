"""Task categories and their per-category inference policy.

Each of the eight capability areas maps to a :class:`CategoryPolicy` that decides
the model tier (cheap ``small`` vs capable ``large``), an output-token cap, and a
sampling temperature. This module is the single place to tune the
accuracy-vs-token trade-off: widen a cap or bump a tier here if a category misses
the accuracy gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    CODE_GEN = "code_gen"
    LOGIC = "logic"


class Tier(str, Enum):
    SMALL = "small"
    LARGE = "large"


@dataclass(frozen=True)
class CategoryPolicy:
    tier: Tier
    max_tokens: int
    temperature: float = 0.0
    # May the bundled local model answer this category at all? See LOCAL_OK below:
    # this is the accuracy gate's load-bearing switch, not a token optimisation.
    local_ok: bool = False


# Which categories the local model is allowed to answer.
#
# This list is empirical, and the earlier "try everything locally" default is what
# failed the 80% accuracy gate. Measured on the participant guide's own practice
# tasks (2 vCPU / 4 GB, Qwen2.5-3B Q4), the local model scored 4/8. It was wrong in
# exactly the ways a local verifier cannot catch:
#
#   * factual   — "Canberra is near the Australian Alps" (a mountain range, asked
#                 for a body of water). Fluent, well-formed, wrong.
#   * math      — invented a step ("Tuesday sells 36 + 60 = 96"), answered 108
#                 where the answer is 144. Passed the old `_has_number` check.
#   * sentiment — bare "Negative" on a mixed review, no justification. Passed the
#                 old `_has_label` check.
#   * ner       — exceeded the 45 s local timeout and produced nothing.
#
# There is no cheap, sound way to detect a confidently-wrong-but-well-formed
# answer without ground truth, and the evaluation prompts are unseen. So rather
# than pretend to verify those categories, we simply don't answer them locally.
#
# The three categories below stay local because they scored 100% AND they are the
# only ones where a verifier can check something real (see verifiers.py): code can
# be parsed and executed, and a summary can be checked against the length/format
# constraint the prompt states and against the source text.
LOCAL_OK: frozenset[Category] = frozenset(
    {Category.CODE_GEN, Category.CODE_DEBUG, Category.SUMMARIZATION}
)

# Simple, short-answer tasks go to the cheap model with tight caps. Tasks that
# need multi-step correctness (math, logic, code) go to the capable model with
# more room. Temperature is 0 everywhere: these are objective tasks and
# determinism helps both accuracy and reproducibility.
# NOTE on caps: max_tokens is a *ceiling*, not a target — a concise model stops
# early and only spends what it needs. But reasoning-capable models spend hidden
# "thinking" tokens before the visible answer, so a cap that is too tight can
# truncate the answer to empty (observed with a 48-token sentiment cap). We keep
# generous floors so an answer always survives; `reasoning_effort=low` (see
# config/client) is what actually keeps token spend down.
_BASE_POLICIES: dict[Category, CategoryPolicy] = {
    Category.SENTIMENT: CategoryPolicy(Tier.SMALL, max_tokens=256),
    Category.NER: CategoryPolicy(Tier.SMALL, max_tokens=512),
    Category.FACTUAL: CategoryPolicy(Tier.SMALL, max_tokens=512),
    Category.SUMMARIZATION: CategoryPolicy(Tier.SMALL, max_tokens=512),
    Category.MATH: CategoryPolicy(Tier.LARGE, max_tokens=1024),
    Category.LOGIC: CategoryPolicy(Tier.LARGE, max_tokens=1024),
    Category.CODE_DEBUG: CategoryPolicy(Tier.LARGE, max_tokens=1024),
    Category.CODE_GEN: CategoryPolicy(Tier.LARGE, max_tokens=1024),
}

POLICIES: dict[Category, CategoryPolicy] = {
    category: CategoryPolicy(
        tier=policy.tier,
        max_tokens=policy.max_tokens,
        temperature=policy.temperature,
        local_ok=category in LOCAL_OK,
    )
    for category, policy in _BASE_POLICIES.items()
}

# Fallback used if a category is somehow missing from POLICIES. Never local: an
# unknown category is exactly the case we have no verifier for.
DEFAULT_POLICY = CategoryPolicy(Tier.LARGE, max_tokens=512, local_ok=False)


def policy_for(category: Category) -> CategoryPolicy:
    return POLICIES.get(category, DEFAULT_POLICY)
