---
name: track1-local-model-results
description: Track 1 local-model benchmark — our own harness scored Qwen2.5-3B at 90%, but the guide's practice tasks scored it at 50%; the 90% was an artifact of lenient self-written checkers
metadata:
  type: project
---

> ⚠️ **The headline number in this memory was wrong and it cost us a submission.** The 36/40 (90%) below was measured with **our own keyword/numeric checkers on our own 40 synthetic tasks**. On the participant guide's **published practice tasks**, the same model scored **4/8 (50%)** — and trusting the 90% is what produced `ACCURACY_GATE_FAILED`. See [[track1-accuracy-gate-postmortem]]. **Lesson: never benchmark a model against checkers you wrote for tasks you wrote.** A lenient checker measures your checker, not the model. The real gate is an LLM judge on unseen prompts.

Benchmarked a bundled local model per category (2026-07-09). Chosen local model: **Qwen2.5-3B-Instruct Q4_K_M GGUF** (bartowski repo, ~1.8 GB) via **llama-cpp-python**. Ran the existing 40-task harness with `scripts/benchmark.py --local` inside the Docker image.

**Result (self-scored, ~unreliable): 36/40 = 90%.** Per-category (correct/5): math 5/5, logic 5/5, code_debug 5/5, code_gen 5/5; factual 4/5, sentiment 4/5, summarization 4/5, ner 4/5.

**What actually held up.** Of the four "100%" categories, only **code_debug and code_gen** survived contact with the practice tasks — and those, plus **summarization**, are now the only categories answered locally (they're also the only ones with a verifier that checks something real: code is *executed*; a summary is checked against the stated length constraint). The self-benchmark's "100%" on **math and logic was an illusion** — on the practice math task the model invented a step and answered 108 (truth: 144), which the numeric checker happily accepted.

**Verifier caveat (this is the load-bearing one):** in production there is NO ground truth (unseen prompts), so a verifier can only check FORMAT/SANITY. It **cannot catch a confidently-wrong-but-well-formed answer** — and that is the local model's dominant failure mode, not generation failure. Do not build escalation logic that assumes otherwise; instead, only answer locally where wrongness is *detectable*.

**Latency:** 0.1–13.3 s/task at LOCAL_N_THREADS=8 on a 13600KF. MUST re-verify < 30 s/request and ≤ 10 min total on the real 2-vCPU grading box (slower) before trusting local-only.

**Dev-setup gotchas (Windows) — see also [[track1-project]]:**
- 13th-gen Intel (13600KF) has AVX2 but NOT AVX-512 → the stock `llama-cpp-python` CPU wheel SIGILLs (0xc000001d illegal instruction).
- Native MinGW source build fails: Strawberry's Windows headers lack `THREAD_POWER_THROTTLING_STATE` (llama.cpp uses it). Windows MAX_PATH also breaks sdist extraction (fix: `TMP=C:\t`).
- **Working path = Docker.** `agent/Dockerfile` is multi-stage: builder compiles `llama-cpp-python==0.3.33` from source with `--no-binary` + `CMAKE_ARGS=-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF`; runtime is python:3.12-slim + **libgomp1** (required — llama.cpp built with -fopenmp) + the wheel + openai. Image ~466 MB without the model.
- Bind-mount for the in-container benchmark needs PowerShell or `MSYS_NO_PATHCONV=1` + `C:/...` path (Git Bash mangles `-v /c/...:/app`).
