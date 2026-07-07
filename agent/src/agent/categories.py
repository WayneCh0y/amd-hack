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


# Simple, short-answer tasks go to the cheap model with tight caps. Tasks that
# need multi-step correctness (math, logic, code) go to the capable model with
# more room. Temperature is 0 everywhere: these are objective tasks and
# determinism helps both accuracy and reproducibility.
POLICIES: dict[Category, CategoryPolicy] = {
    Category.SENTIMENT: CategoryPolicy(Tier.SMALL, max_tokens=48),
    Category.NER: CategoryPolicy(Tier.SMALL, max_tokens=256),
    Category.FACTUAL: CategoryPolicy(Tier.SMALL, max_tokens=300),
    Category.SUMMARIZATION: CategoryPolicy(Tier.SMALL, max_tokens=256),
    Category.MATH: CategoryPolicy(Tier.LARGE, max_tokens=512),
    Category.LOGIC: CategoryPolicy(Tier.LARGE, max_tokens=512),
    Category.CODE_DEBUG: CategoryPolicy(Tier.LARGE, max_tokens=640),
    Category.CODE_GEN: CategoryPolicy(Tier.LARGE, max_tokens=640),
}

# Fallback used if a category is somehow missing from POLICIES.
DEFAULT_POLICY = CategoryPolicy(Tier.LARGE, max_tokens=512)


def policy_for(category: Category) -> CategoryPolicy:
    return POLICIES.get(category, DEFAULT_POLICY)
