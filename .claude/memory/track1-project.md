---
name: track1-project
description: What the amd-hack repo is — AMD Hackathon Track 1 batch AI agent, its hard constraints and scoring model
metadata:
  type: project
---

This repo (`amd-hack`) is a submission for **AMD Developer Hackathon (ACT II) — Track 1: General-Purpose AI Agent**. Wayne is doing Track 1 only.

**What it is:** a batch AI agent (NOT a chatbot) shipped as a Docker image. On startup it reads `/input/tasks.json`, solves each NL prompt across 8 capability categories (factual, math, sentiment, summarization, NER, code debugging, logical reasoning, code generation) using **Fireworks AI** models, writes `/output/results.json` (`[{task_id, answer}]`), and exits 0.

**Scoring (two-stage):** (1) accuracy gate via LLM-Judge — below threshold = dropped; (2) among passers, ranked ascending by **total tokens** recorded by the Fireworks proxy. So the goal is *enough accuracy at minimum token cost*. This is an inference-orchestration problem, not model training.

**Hard constraints (from `documentation/Participant Guide_ AMD Developer Hackathon (ACT II).pdf`):**
- Read `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from env ONLY — never hardcode keys/URLs/model IDs. `ALLOWED_MODELS` is comma-separated, published launch day.
- ALL inference must route through `FIREWORKS_BASE_URL` (Fireworks is OpenAI-compatible → use `openai` client) or tokens aren't recorded (scores zero). Only models in `ALLOWED_MODELS` are allowed.
- Runtime ≤ 10 min total, < 30 s/request; English only; exit 0 success / non-zero failure; valid JSON output with an answer per task_id.
- Image: `linux/amd64` (`docker buildx build --platform linux/amd64 --push`), publicly pullable, ≤ 10 GB, ready < 60 s. No `.env` in image. Don't hardcode/cache answers (unseen variants).

**Design decisions:** auto size-based model routing (parse `Nb` sizes from model IDs → small tier for simple tasks, large tier for hard tasks, graceful single-model fallback); heuristic (0-token) task router; per-category `max_tokens` caps + temperature 0.

**Status: implemented & verified** under `agent/` (Python, OpenAI client). Modules in `agent/src/agent/`: `config`, `model_selector`, `router`, `categories`, `prompts`, `fireworks_client`, `pipeline`, `main`. 18 unit tests pass (`agent/tests/`); full pipeline verified end-to-end against a local mock OpenAI server — correct routing per category, token metering, atomic JSON write, exit 1 on config/malformed-input errors, exit 0 on success. Launch-day tuning lives in `categories.py` (max_tokens/tier per category). Build: `docker buildx build --platform linux/amd64 -t <registry>/amd-track1:latest --push agent/`. Local run: `agent/scripts/run_local.py`.
