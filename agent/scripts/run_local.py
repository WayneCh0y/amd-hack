#!/usr/bin/env python3
"""Local development harness.

Runs the full agent pipeline against a tasks file on your machine, using the
same code path as the container entrypoint. Reads Fireworks credentials from the
environment (or a local .env file, which must NOT be committed or baked into the
image). Prints where results were written and the total token usage.

Usage:
    python scripts/run_local.py [tasks.json] [results.json]

Environment (required to actually call Fireworks):
    FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _load_dotenv(path: pathlib.Path) -> None:
    """Minimal .env loader for local dev (no dependency on python-dotenv)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    _load_dotenv(ROOT / ".env")

    input_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "examples" / "tasks.json")
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "results.local.json")

    os.environ["INPUT_PATH"] = input_path
    os.environ["OUTPUT_PATH"] = output_path

    from agent.main import main as agent_main

    code = agent_main()

    if code == 0 and os.path.exists(output_path):
        results = json.loads(pathlib.Path(output_path).read_text(encoding="utf-8"))
        print(f"\nWrote {len(results)} result(s) to {output_path}", file=sys.stderr)
        for r in results:
            answer = (r.get("answer") or "").replace("\n", " ")
            preview = answer[:100] + ("…" if len(answer) > 100 else "")
            print(f"  [{r.get('task_id')}] {preview}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
