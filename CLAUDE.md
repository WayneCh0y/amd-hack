# CLAUDE.md — amd-hack (AMD Developer Hackathon, Track 1)

## What this repo is
A submission for **AMD Developer Hackathon (ACT II) — Track 1: General-Purpose AI Agent**.
A **batch** AI agent (not a chatbot), shipped as a Docker image: read `/input/tasks.json` →
solve each prompt via **Fireworks AI** → write `/output/results.json` → exit 0.

Source of truth for requirements: `documentation/Participant Guide_ AMD Developer Hackathon (ACT II).pdf`
and `documentation/track-1.md`.

## Non-negotiable constraints
- Read `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` **from env only** — never hardcode
  keys, base URLs, or model IDs. `ALLOWED_MODELS` (comma-separated) is published on launch day.
- **All** inference must go through `FIREWORKS_BASE_URL` (Fireworks is OpenAI-compatible → `openai` client)
  or tokens aren't recorded → scores zero. Only use models listed in `ALLOWED_MODELS`.
- Output `[{ "task_id", "answer" }]` as valid JSON, one entry per input task_id.
- Runtime ≤ 10 min total, < 30 s/request; English only; exit 0 on success, non-zero on failure.
- Image: `linux/amd64`, publicly pullable, ≤ 10 GB, ready < 60 s. No `.env` bundled in the image.
- Don't hardcode or cache answers — evaluation uses unseen prompt variants.

## Scoring model (drives all design choices)
1. Accuracy gate (LLM-Judge) — below threshold = removed from leaderboard.
2. Token efficiency — passers ranked by **fewest total tokens**. Goal: *enough accuracy, minimum tokens*.

## Conventions
- Python. Keep modules small and single-purpose (see the `agent/` layout in the plan).
- Model IDs never appear in code — everything is derived from `ALLOWED_MODELS` at runtime.
- Prefer heuristics over extra LLM calls; cap `max_tokens` per category; temperature 0 for objective tasks.

## Memory
Project memory lives in `.claude/memory/` (indexed by `.claude/memory/MEMORY.md`). Read it at session start.
