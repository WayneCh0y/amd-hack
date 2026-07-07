---
name: track1-benchmark-harness
description: The launch-day measurement tool for Track 1 — how to run it and what the first results showed
metadata:
  type: project
---

Built a **benchmark harness** (dev-only, excluded from image) to pick the model/routing strategy from data, not guesswork — the answer to "is regex routing enough / should I train a router". Part of [[track1-project]]; complements [[track1-launch-day-tuning]].

**Files (under `agent/`):** `benchmark/dataset.py` (24 labelled tasks, 3 per category, some plainly-worded to defeat keyword routing), `benchmark/checkers.py` (deterministic scoring: keywords/numeric/label/entities + **real code execution** for code tasks), `scripts/benchmark.py` (runner). Client gained `complete_with_usage()` for per-call tokens.

**Run:** `.venv/bin/python scripts/benchmark.py --models "<ids>" --threshold 0.67` (creds from `.env`). Prints ACCURACY grid, AVG-TOKENS grid, cheapest-passing model per category, and best single model. Raw records → `benchmark_results.json`.

**Gotcha fixed:** code-exec timeout used `signal.alarm`, which only works on the main thread; the runner is multi-threaded, so every code check falsely failed until guarded with `threading.current_thread() is main_thread()`.

**First results (2026-07-07, this account's 5 working models — NOT the competition's; kimi-k2p5 returns HTTP 500, flux is image-only):**
- **gpt-oss-120b is the best single model: passes 8/8 categories at the lowest single-model token total (~5210 over 24 tasks).** Simple safe strategy = gpt-oss-120b for everything + reasoning_effort=low.
- Per-category cheapest (routing frontier): factual→deepseek-v4-pro (112 tok), summarization & code_debug→glm-5p2 (137/236), code_gen→deepseek-v4-pro (259); gpt-oss wins math/sentiment/ner/logic. Optimal per-category routing ≈ 1529 tok/set vs gpt-oss single ≈ 1737 → only **~12% savings**, and it needs accurate classification (misroute = accuracy-gate risk).
- Sentiment is the discriminator (only gpt-oss got 3/3; the mixed-review task splits models).

**Takeaway:** with an all-frontier model menu, routing buys little (~12%) and single-model gpt-oss is the pragmatic choice; the big levers stay reasoning_effort + output minimization. IF the real launch-day list has a genuinely small/cheap model that still passes easy categories, routing savings will be far larger — so re-run this harness on the real ALLOWED_MODELS and read the recommendation before deciding.
