#!/usr/bin/env python3
"""Measure accuracy x token cost per model per category.

This is the launch-day decision tool: point it at the real ALLOWED_MODELS and it
tells you, per category, which models clear an accuracy bar and how many tokens
each spends — so you can pick the cheapest model that passes (or decide a single
model is best). No guessing about routing strategy.

Usage:
    python scripts/benchmark.py                       # all models in ALLOWED_MODELS
    python scripts/benchmark.py --models m1,m2        # only these
    python scripts/benchmark.py --categories math,code_gen
    python scripts/benchmark.py --threshold 0.8 --reasoning low --out bench.json

Credentials come from the environment or agent/.env (same as run_local.py).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model accuracy/token benchmark")
    p.add_argument("--models", help="comma-separated model IDs (default: ALLOWED_MODELS)")
    p.add_argument("--categories", help="comma-separated category filter")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="accuracy bar for the recommendation (default 0.7)")
    p.add_argument("--reasoning", help="override REASONING_EFFORT (e.g. low/medium/high/'')")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default=str(ROOT / "benchmark_results.json"))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _load_dotenv(ROOT / ".env")
    if args.models:
        os.environ["ALLOWED_MODELS"] = args.models
    if args.reasoning is not None:
        os.environ["REASONING_EFFORT"] = args.reasoning

    from agent.categories import Category, policy_for
    from agent.config import Config, ConfigError
    from agent.fireworks_client import FireworksClient
    from agent.prompts import system_prompt_for
    from benchmark.checkers import verify
    from benchmark.dataset import TASKS

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    models = list(config.allowed_models)
    tasks = TASKS
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",")}
        tasks = [t for t in tasks if t.category.value in wanted]
    if not tasks:
        print("No tasks match the category filter.", file=sys.stderr)
        return 1

    client = FireworksClient(config)
    print(f"Benchmarking {len(models)} model(s) x {len(tasks)} task(s) "
          f"(reasoning_effort={config.reasoning_effort or 'off'})\n", file=sys.stderr)

    # One record per (model, task).
    def evaluate(job: tuple[str, object]) -> dict:
        model, task = job
        policy = policy_for(task.category)
        start = time.monotonic()
        try:
            answer, usage = client.complete_with_usage(
                model=model,
                system=system_prompt_for(task.category),
                user=task.prompt,
                max_tokens=policy.max_tokens,
                temperature=policy.temperature,
            )
            correct = verify(task.check, answer)
            total = usage.total_tokens
            error = ""
        except Exception as exc:  # noqa: BLE001
            answer, correct, total, error = "", False, 0, str(exc)[:100]
        return {
            "model": model, "task": task.id, "category": task.category.value,
            "correct": correct, "tokens": total,
            "latency": round(time.monotonic() - start, 2), "error": error,
        }

    jobs = [(m, t) for m in models for t in tasks]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        records = list(pool.map(evaluate, jobs))

    _report(records, models, tasks, args.threshold)

    pathlib.Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nRaw records written to {args.out}", file=sys.stderr)
    return 0


def _report(records, models, tasks, threshold: float) -> None:
    from agent.categories import Category

    categories = []
    for t in tasks:
        if t.category not in categories:
            categories.append(t.category)

    # Aggregate: (model, category) -> [correct...], [tokens...]
    agg_correct: dict = defaultdict(list)
    agg_tokens: dict = defaultdict(list)
    for r in records:
        key = (r["model"], r["category"])
        agg_correct[key].append(r["correct"])
        agg_tokens[key].append(r["tokens"])

    def short(model: str) -> str:
        return model.split("/")[-1][:18]

    col_w = max(12, *(len(short(m)) + 2 for m in models))
    cat_w = max(14, *(len(c.value) for c in categories))

    def header(title: str) -> None:
        print(f"\n{title}")
        print("category".ljust(cat_w) + "".join(short(m).rjust(col_w) for m in models))

    # Accuracy grid.
    header("ACCURACY  (correct / total)")
    for c in categories:
        row = c.value.ljust(cat_w)
        for m in models:
            vals = agg_correct[(m, c.value)]
            n = sum(vals)
            row += f"{n}/{len(vals)}".rjust(col_w)
        print(row)

    # Token grid (avg total tokens per task).
    header("AVG TOKENS / task")
    for c in categories:
        row = c.value.ljust(cat_w)
        for m in models:
            toks = agg_tokens[(m, c.value)]
            avg = round(sum(toks) / len(toks)) if toks else 0
            row += str(avg).rjust(col_w)
        print(row)

    # Per-category recommendation: cheapest model at/above the accuracy bar.
    print(f"\nRECOMMENDATION per category (accuracy >= {threshold:.0%}, cheapest tokens)")
    for c in categories:
        best = None  # (tokens, model, acc)
        for m in models:
            vals = agg_correct[(m, c.value)]
            acc = sum(vals) / len(vals) if vals else 0.0
            avg = sum(agg_tokens[(m, c.value)]) / len(vals) if vals else 0.0
            if acc >= threshold and (best is None or avg < best[0]):
                best = (avg, m, acc)
        if best:
            print(f"  {c.value.ljust(cat_w)} -> {short(best[1]).ljust(20)} "
                  f"({best[2]:.0%}, {round(best[0])} tok)")
        else:
            print(f"  {c.value.ljust(cat_w)} -> NONE cleared the bar "
                  f"(raise threshold, tune prompts, or widen max_tokens)")

    # Best single model: passes the most categories, tie-break on total tokens.
    print(f"\nBEST SINGLE MODEL (for a no-routing strategy)")
    ranked = []
    for m in models:
        passed = sum(
            1 for c in categories
            if (sum(agg_correct[(m, c.value)]) / len(agg_correct[(m, c.value)])) >= threshold
        )
        total_tokens = sum(sum(agg_tokens[(m, c.value)]) for c in categories)
        ranked.append((passed, -total_tokens, m, total_tokens))
    ranked.sort(reverse=True)
    for passed, _, m, total_tokens in ranked:
        print(f"  {short(m).ljust(20)} passes {passed}/{len(categories)} cats, "
              f"{total_tokens} tokens total")


if __name__ == "__main__":
    sys.exit(main())
