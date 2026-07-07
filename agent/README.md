# Track 1 — General-Purpose AI Agent

A batch AI agent for the AMD Developer Hackathon (ACT II), Track 1. It reads a
list of natural-language tasks, solves each one via Fireworks AI, and writes the
answers — then exits. It is optimised for the competition's scoring: **pass the
accuracy gate, then minimise total tokens**.

## How it works

```
/input/tasks.json
      │  read + validate
      ▼
  normalize_tasks ──► router.classify (0 tokens, heuristic)
      │                     │
      │                     ▼
      │              categories.policy  (model tier + max_tokens + temp)
      │                     │
      │                     ▼
      │              model_selector    (smallest/largest of ALLOWED_MODELS)
      │                     │
      │                     ▼
      │              FireworksClient   (OpenAI-compatible, via FIREWORKS_BASE_URL)
      ▼
/output/results.json   [{ "task_id", "answer" }, ...]
```

Key design choices (see `../.claude/memory/track1-project.md` for the full rationale):

- **Heuristic router** — classifies each prompt into one of the 8 categories with
  keyword/pattern rules, spending **zero tokens** on classification.
- **Size-based model routing** — parses parameter counts from the model IDs in
  `ALLOWED_MODELS` (only known on launch day) and sends simple tasks to the
  smallest model, hard tasks to a larger one. No model IDs are hardcoded.
- **Per-category caps** — tight `max_tokens` and `temperature=0` per category
  (`categories.py`) keep completions short and deterministic.
- **Robust** — bounded concurrency to fit the 10-min budget, cross-tier retry on
  failure, a wall-clock deadline guard, and atomic output so `results.json` is
  always valid and complete.

## Layout

```
src/agent/       config, fireworks_client, model_selector, router,
                 categories, prompts, pipeline, main
tests/           router + selector unit tests
scripts/         run_local.py — local end-to-end harness
examples/        tasks.json — one task per capability category
Dockerfile       linux/amd64 image, entrypoint python -m agent.main
```

## Local development

```bash
cd agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest

# Unit tests (no credentials needed)
PYTHONPATH=src python -m pytest tests/ -q

# End-to-end against the sample tasks (needs real Fireworks credentials)
cp .env.example .env    # then edit .env with a dev key + real model IDs
python scripts/run_local.py            # uses examples/tasks.json
python scripts/run_local.py path/to/tasks.json out.json
```

`run_local.py` prints total token usage and a preview of each answer.

## Build, test, and submit the Docker image

The judging VM runs **linux/amd64**. Build for that platform explicitly
(required on Apple Silicon; harmless elsewhere):

```bash
cd agent

# Build + push in one step to a PUBLIC registry (Docker Hub or GHCR).
docker buildx build --platform linux/amd64 \
  -t docker.io/<your-user>/amd-track1:latest --push .
# or GHCR:
docker buildx build --platform linux/amd64 \
  -t ghcr.io/<your-user>/amd-track1:latest --push .
```

Make the pushed image **public** (Docker Hub: repo → Settings → Make public;
GHCR: package → Package settings → Change visibility → Public).

### Test the container the way the harness runs it

```bash
mkdir -p /tmp/out
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$PWD/examples":/input \
  -v /tmp/out:/output \
  <your-image>
cat /tmp/out/results.json    # must be valid JSON, one answer per task_id
echo "exit code: $?"          # must be 0
```

## Competition compliance checklist

- [x] Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from env only.
- [x] No hardcoded keys, base URLs, or model IDs; no `.env` baked into the image.
- [x] All inference goes through `FIREWORKS_BASE_URL` (OpenAI-compatible client).
- [x] Only uses models from `ALLOWED_MODELS`.
- [x] Reads `/input/tasks.json`, writes valid `/output/results.json` (`task_id` + `answer`).
- [x] Exit 0 on success, non-zero on fatal error; wall-clock budget < 10 min.
- [x] `linux/amd64` image; small base (`python:3.12-slim`), well under 10 GB.

## Launch-day tuning

Model IDs never need editing — they flow entirely from `ALLOWED_MODELS`. Two
knobs matter, in order of impact:

1. **`reasoning_effort`** (config default `low`, env `REASONING_EFFORT`) — the
   biggest token lever. 2026 Fireworks models reason by default; low effort cut
   completion tokens ~27% in testing and prevents the "all budget spent on
   hidden thinking, empty answer" failure. The client auto-drops the param for
   any model that doesn't accept it. Raise to `medium` only if math/logic
   accuracy suffers.
2. **`src/agent/categories.py`** — widen a category's `max_tokens` or bump its
   tier `SMALL`→`LARGE` if it misses the accuracy gate. Caps are ceilings, not
   targets; keep them generous so an answer is never truncated to empty.

Always run `scripts/run_local.py` on launch day and eyeball the answers — watch
for models that print their reasoning *into* the answer (verbose, token-heavy)
versus those that return only the final answer (preferred).
