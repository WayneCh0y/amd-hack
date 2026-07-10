"""Capture a REAL run of the agent on the sample tasks -> demo/trace.json.

The Streamlit demo can't load the bundled 3B model (free-tier RAM), so the
"Benchmark story" tab replays a trace captured here instead. This script runs the
agent's actual local-first -> escalate loop (same modules as the container) and
records, per task: the category, whether the local model's answer was kept or
escalated, the real Fireworks token cost, and the final answer.

Run it where the local model works — i.e. **inside the Docker image**, which
builds llama.cpp for a portable AVX2 baseline (a bare Windows/mac host often
SIGILLs on the prebuilt wheel). Example (Git Bash, from the repo root):

    docker build -t amd-track1:dev agent
    MSYS_NO_PATHCONV=1 docker run --rm \
      -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
      -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
      -e ALLOWED_MODELS="$ALLOWED_MODELS" \
      -e LOCAL_MODEL_PATH=/models/model.gguf \
      -v "$PWD:/work" --entrypoint python amd-track1:dev \
      /work/demo/capture_trace.py --out /work/demo/trace.json

Then commit demo/trace.json. The demo picks it up automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "agent" / "src"))

from agent.categories import Tier, policy_for  # noqa: E402
from agent.config import Config  # noqa: E402
from agent.fireworks_client import FireworksClient  # noqa: E402
from agent.local_model import LocalModel  # noqa: E402
from agent.model_selector import ModelSelector  # noqa: E402
from agent.prompts import system_prompt_for  # noqa: E402
from agent.router import classify  # noqa: E402
from agent.verifiers import is_trustworthy  # noqa: E402


def _tasks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def capture(tasks: list[dict]) -> list[dict]:
    cfg = Config.from_env()
    client = FireworksClient(cfg)
    selector = ModelSelector(cfg.allowed_models)

    local = LocalModel()
    if local.available():
        local.load()
    else:
        print("WARNING: local weights absent — trace will be Fireworks-only", file=sys.stderr)
        local = None

    rows: list[dict] = []
    for task in tasks:
        prompt = task.get("prompt", "")
        category = classify(prompt)
        policy = policy_for(category)
        system = system_prompt_for(category)

        source = "fireworks"
        fw_tokens = 0
        final = ""
        local_answer = ""

        # 1) local-first
        if local is not None:
            try:
                local_answer = local.complete(
                    system=system, user=prompt,
                    max_tokens=policy.max_tokens, temperature=policy.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{task['task_id']}: local error {exc}", file=sys.stderr)
                local_answer = ""
            if local_answer and is_trustworthy(category, prompt, local_answer):
                source, final = "local", local_answer

        # 2) escalate on failure
        if source != "local":
            model = selector.small() if policy.tier is Tier.SMALL else selector.large()
            answer, usage = client.complete_with_usage(
                model=model, system=system, user=prompt,
                max_tokens=policy.max_tokens, temperature=policy.temperature,
            )
            final, fw_tokens = answer, usage.total_tokens

        rows.append({
            "task_id": task.get("task_id"),
            "prompt": prompt,
            "category": category.value,
            "tier": "small" if policy.tier is Tier.SMALL else "large",
            "source": source,
            "fireworks_tokens": fw_tokens,
            "local_answer": local_answer,
            "final_answer": final,
        })
        print(f"{task.get('task_id')}: {category.value} -> {source} ({fw_tokens} tokens)",
              file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=str(_REPO / "agent" / "examples" / "tasks.json"))
    ap.add_argument("--out", default=str(_REPO / "demo" / "trace.json"))
    args = ap.parse_args()

    rows = capture(_tasks(Path(args.tasks)))
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(r["fireworks_tokens"] for r in rows)
    local_kept = sum(1 for r in rows if r["source"] == "local")
    print(f"\nWrote {args.out}: {len(rows)} tasks, {local_kept} local (0 tokens), "
          f"{total} Fireworks tokens total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
