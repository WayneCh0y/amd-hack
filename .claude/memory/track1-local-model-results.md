---
name: track1-local-model-results
description: Track 1 local-model benchmark — Qwen2.5-3B-Instruct Q4 scores 36/40 (90%) on CPU, nearly clearing the gate at zero Fireworks tokens; plus the dev-setup gotchas
metadata:
  type: project
---

Answered the blocking open question in [[track1-hybrid-architecture]]: benchmarked a bundled local model per category (2026-07-09). Chosen local model: **Qwen2.5-3B-Instruct Q4_K_M GGUF** (bartowski repo, ~1.8 GB) via **llama-cpp-python**. Ran the existing 40-task harness with `scripts/benchmark.py --local` (new mode) inside the Docker image.

**Result: 36/40 = 90%, passes 8/8 categories (≥80% each).** Per-category (correct/5, avg tokens): math 5/5, logic 5/5, code_debug 5/5, code_gen 5/5 (all 100%); factual 4/5 (67 tok), sentiment 4/5, summarization 4/5, ner 4/5 (all 80%). The 4 misses were an ambiguous mixed-sentiment review (sent3), NER completeness (ner1), a factual-recency item (fact4 "most moons" — Saturn vs Jupiter), and long-text summarization keyword coverage (sum4) — i.e. hard/ambiguous instances, not category-wide weakness.

**This REFUTES the architecture doc's pre-data hypothesis** (it guessed math/logic/code-gen would "lean Fireworks"). In reality those are the local model's STRONGEST categories (100%); the weaker ones are factual/sentiment/summarization/ner. **Implication:** a 3B local model alone nearly clears the 16/19 (84.2%) gate at **zero Fireworks tokens** — the best possible ranking outcome. **Caveat:** our checkers are deterministic keyword/numeric/code checks on OUR 40 tasks; the real gate is an LLM judge on 19 UNSEEN tasks, so 90%-here ≠ guaranteed pass. Margin over the gate is ~1 task — thin given judge non-determinism.

**Verifier caveat for wiring escalation:** in production we have NO ground-truth (unseen prompts), so local verifiers can only check FORMAT/SANITY (empty answer, malformed label set, non-parseable code, no number in a math answer) + confidence — they can't catch a confidently-wrong-but-well-formed answer. So escalation mainly rescues generation failures, not subtle wrongness; the local model's raw accuracy carries most of the load.

**Latency:** 0.1–13.3 s/task at LOCAL_N_THREADS=8 on a 13600KF. MUST re-verify < 30 s/request and ≤ 10 min total on the real 2-vCPU grading box (slower) before trusting local-only.

**Dev-setup gotchas (Windows) — see also [[track1-project]]:**
- 13th-gen Intel (13600KF) has AVX2 but NOT AVX-512 → the stock `llama-cpp-python` CPU wheel SIGILLs (0xc000001d illegal instruction).
- Native MinGW source build fails: Strawberry's Windows headers lack `THREAD_POWER_THROTTLING_STATE` (llama.cpp uses it). Windows MAX_PATH also breaks sdist extraction (fix: `TMP=C:\t`).
- **Working path = Docker.** `agent/Dockerfile` is multi-stage: builder compiles `llama-cpp-python==0.3.33` from source with `--no-binary` + `CMAKE_ARGS=-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF`; runtime is python:3.12-slim + **libgomp1** (required — llama.cpp built with -fopenmp) + the wheel + openai. Image ~466 MB without the model.
- Bind-mount for the in-container benchmark needs PowerShell or `MSYS_NO_PATHCONV=1` + `C:/...` path (Git Bash mangles `-v /c/...:/app`).
