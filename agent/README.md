# Track 1 — General-Purpose AI Agent

A batch AI agent for the AMD Developer Hackathon (ACT II), Track 1. It reads a
list of natural-language tasks, solves each one, and writes the answers — then
exits. It is optimised for the competition's scoring: **pass the accuracy gate,
then minimise total tokens**.

The agent is **local-first**: a bundled 2–3B model (`Qwen2.5-3B-Instruct`, 4-bit
GGUF, run on CPU via `llama-cpp-python`) answers each task at **zero Fireworks
tokens**. Only tokens routed through `FIREWORKS_BASE_URL` count toward the score,
so every task answered locally is free. Fireworks is used only as an escalation
path when a local answer fails verification.

## How it works

```
/input/tasks.json
      │  read + validate
      ▼
  normalize_tasks ──► router.classify (0 tokens, heuristic)
      │                     │
      │                     ▼
      │              categories.policy  (max_tokens + temp + which verifier)
      │                     │
      │      ┌──────────────┴───────────────┐
      │      ▼                               │
      │  LocalModel  (Qwen 3B, CPU)          │  1) local-first, 0 Fireworks tokens
      │      │                               │
      │      ▼                               │
      │  verifiers.is_trustworthy?           │
      │      │ yes → keep local answer       │
      │      │ no  ▼                          │
      │  model_selector + FireworksClient    │  2) escalate only on failure
      │      (smallest/largest of ALLOWED_MODELS, via FIREWORKS_BASE_URL)
      ▼
/output/results.json   [{ "task_id", "answer" }, ...]
```

Key design choices (see `../.claude/memory/` for the full rationale, esp.
`track1-hybrid-architecture.md` and `track1-local-model-results.md`):

- **Local-first answering** — the bundled model answers every task first at zero
  Fireworks cost. On the 40-task dev benchmark it scored 36/40 (90%), passing all
  8 categories — enough to nearly clear the gate with `ZERO_API_CALLS`.
- **Verify, then escalate** — `verifiers.py` decides keep-vs-escalate. Because
  eval prompts are unseen (no ground truth), verifiers are *conservative*: they
  reject only clear generation failures (empty, refusal, math-without-a-number,
  code-without-code, unlabeled sentiment) and keep everything else. Failures
  escalate to Fireworks.
- **Heuristic router** — classifies each prompt into one of the 8 categories with
  keyword/pattern rules, spending **zero tokens**; the category selects the
  prompt template, `max_tokens` cap, and which verifier runs.
- **Size-based model routing** (escalation only) — parses parameter counts from
  the model IDs in `ALLOWED_MODELS` (only known on launch day) and sends simple
  tasks to the smallest model, hard tasks to a larger one. No model IDs hardcoded.
- **Robust** — graceful degradation to Fireworks-only if the weights are absent,
  bounded concurrency, cross-tier retry, a wall-clock deadline guard, and atomic
  output so `results.json` is always valid and complete.

## Layout

```
src/agent/       config, fireworks_client, local_model, verifiers,
                 model_selector, router, categories, prompts, pipeline, main
tests/           router, pipeline, verifiers, local_model unit tests
scripts/         run_local.py — end-to-end harness (Fireworks)
                 local_smoke.py — eyeball local answers, one per category
                 benchmark.py [--local] — per-model/-category accuracy × tokens
benchmark/       labelled dataset + deterministic checkers (dev-only)
models/          bundled GGUF weights (git-ignored; downloaded, see below)
Dockerfile       multi-stage linux/amd64 image (AVX2 llama.cpp + model + agent)
```

## Local development

```bash
cd agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest

# Unit tests (no credentials, no model weights needed)
PYTHONPATH=src python -m pytest tests/ -q

# End-to-end against the sample tasks (needs real Fireworks credentials)
cp .env.example .env    # then edit .env with a dev key + real model IDs
python scripts/run_local.py            # uses examples/tasks.json
python scripts/run_local.py path/to/tasks.json out.json
```

`run_local.py` prints total token usage and a preview of each answer.

### The bundled local model

Download the GGUF weights once (git-ignored; ~1.8 GB) — the Docker build copies
them into the image:

```bash
python -c "from huggingface_hub import hf_hub_download as d; \
  d('bartowski/Qwen2.5-3B-Instruct-GGUF','Qwen2.5-3B-Instruct-Q4_K_M.gguf', local_dir='models')"
```

**Running the model locally uses `llama-cpp-python`, which must match your CPU.**
The prebuilt PyPI/CPU wheel assumes AVX-512 and SIGILLs on CPUs without it (e.g.
12th/13th-gen Intel). A native MinGW build on Windows also fails (old SDK
headers). **The reliable path is Docker** — it builds `llama-cpp-python` from
source with an AVX2 baseline (see `Dockerfile`) that runs on any modern x86-64.

```bash
# Build the runtime image (compiles llama.cpp — a few minutes the first time).
docker build -t amd-track1:dev .

# Per-category local benchmark inside the container (uses the 40-task harness).
# Windows Git Bash: prefix MSYS_NO_PATHCONV=1 and use C:/... paths for -v.
docker run --rm -v "$PWD:/app" \
  -e LOCAL_MODEL_PATH=/app/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -e LOCAL_N_THREADS=8 \
  --entrypoint python amd-track1:dev \
  scripts/benchmark.py --local --out /app/benchmark_results.local.json
```

`benchmark.py --local` prints an accuracy grid, avg local tokens per category,
and a per-category recommendation — the data that decides what stays local vs.
escalates. (`scripts/local_smoke.py <model.gguf>` runs one prompt per category
for a quick eyeball.)

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
- [x] `linux/amd64` image; `python:3.12-slim` base + AVX2 llama.cpp + ~1.8 GB
      model ≈ 2.3 GB, well under 10 GB.
- [x] Local model answers count toward accuracy at **zero** Fireworks tokens; a
      local-only run is a valid `ZERO_API_CALLS` result, not a failure.

## Launch-day tuning

Model IDs never need editing — they flow entirely from `ALLOWED_MODELS`. Knobs
that matter, in order of impact:

0. **`LOCAL_MODEL_ENABLED`** (default `true`) — local-first answering is the
   biggest token lever (local answers cost zero scored tokens). Set `false` to
   force Fireworks-only. The agent also auto-disables local if the weights are
   missing. `LOCAL_N_THREADS` (default 2, matching the grading box) trades speed
   for CPU; on the real 2-vCPU box, confirm each request stays < 30 s.
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
