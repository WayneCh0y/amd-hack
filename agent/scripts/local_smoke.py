#!/usr/bin/env python3
"""Smoke test for the bundled local model.

Loads the GGUF via LocalModel and runs one prompt per capability category, so you
can eyeball whether a 2-3B local model answers each well enough to keep off
Fireworks (zero scored tokens). Prints answer + local token counts + latency.

Usage:
    python scripts/local_smoke.py [path/to/model.gguf]

If no path is given, uses $LOCAL_MODEL_PATH, else the single .gguf under models/.
"""

from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.local_model import LocalModel  # noqa: E402
from agent.prompts import system_prompt_for  # noqa: E402
from agent.router import classify  # noqa: E402


def _resolve_model_path(argv: list[str]) -> str | None:
    import os

    if len(argv) > 1:
        return argv[1]
    if os.environ.get("LOCAL_MODEL_PATH"):
        return os.environ["LOCAL_MODEL_PATH"]
    ggufs = sorted((ROOT / "models").glob("*.gguf"))
    return str(ggufs[0]) if ggufs else None


# One representative prompt per category (short, so the smoke test is quick).
PROMPTS = [
    "What is the capital of Australia?",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
    "Classify the sentiment: 'The plot dragged, but the acting was superb.'",
    "Summarise in one sentence: The mitochondria is the powerhouse of the cell, "
    "generating most of the cell's supply of ATP used as chemical energy.",
    "Extract the named entities: 'Satya Nadella announced in Seattle that "
    "Microsoft will acquire the startup in March 2025.'",
    "This Python function should return the factorial but has a bug:\n"
    "def fact(n):\n    r = 0\n    for i in range(1, n+1):\n        r *= i\n    return r",
    "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops "
    "definitely Lazzies? Answer yes or no and explain briefly.",
    "Write a Python function is_palindrome(s) that returns True if s is a "
    "palindrome, ignoring case and non-alphanumeric characters.",
]


def main() -> int:
    path = _resolve_model_path(sys.argv)
    if not path:
        print("No model path given and no .gguf found under models/.", file=sys.stderr)
        return 1

    model = LocalModel(model_path=path)
    print(f"Loading {path} ...", file=sys.stderr)
    t0 = time.monotonic()
    model.load()
    print(f"Loaded in {time.monotonic() - t0:.1f}s\n", file=sys.stderr)

    for prompt in PROMPTS:
        category = classify(prompt)
        system = system_prompt_for(category)
        t0 = time.monotonic()
        text, usage = model.complete_with_usage(
            system=system, user=prompt, max_tokens=512, temperature=0.0
        )
        dt = time.monotonic() - t0
        print("=" * 72)
        print(f"[{category.value}] {prompt.splitlines()[0][:60]}")
        print(f"  ({dt:.1f}s, local tokens: prompt={usage.prompt_tokens} "
              f"completion={usage.completion_tokens})")
        print("-" * 72)
        print(text)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
