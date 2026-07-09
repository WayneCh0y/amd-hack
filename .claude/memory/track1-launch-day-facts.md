---
name: track1-launch-day-facts
description: Track 1 published facts — the real ALLOWED_MODELS list, 80% gate over 19 tasks, and the zero-token local-model strategy
metadata:
  type: project
---

Official Track 1 facts published via Discord + confirmed in the Participant Guide PDF (as of 2026-07-09). These replace the earlier "unknown until launch day" placeholders in [[track1-project]] and [[track1-benchmark-harness]].

**Real `ALLOWED_MODELS` (still read from env, never hardcode):** `minimax-m3`, `kimi-k2p7-code`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it-nvfp4`. Re-run the benchmark harness against THESE on the real proxy to pick routing/model. Note the size-parsing selector ("Nb") won't parse these IDs cleanly (gemma-4-31b/26b, kimi, minimax) — verify tier assignment doesn't silently misroute.

**Accuracy gate = 80%, and there are exactly 19 fixed tasks.** Every score is n/19: 16/19 = 84.2% passes, 15/19 = 78.9% fails. **Need ≥ 16 of 19 correct.** The LLM judge is non-deterministic run-to-run (same image can wobble a point), so build margin above 16 — don't sit on the line.

**Local models are a first-class, zero-token strategy (the biggest ranking lever).** A bundled local model's answers count FULLY toward accuracy; only tokens through `FIREWORKS_BASE_URL` count toward the token score. A local correct answer = zero Fireworks tokens = best possible rank. `flagged: ZERO_API_CALLS` is explicitly NOT a failure. **Why:** ranking is fewest-tokens-among-passers, so every task answered locally is free. **How to apply:** consider answering easy/deterministic categories locally and escalating to Fireworks only when the local model is likely wrong; if a small local model clears 16/19 alone, call Fireworks zero times.

**Grading env constraints for a local model:** 4 GB RAM, 2 vCPU. A 2B–3B 4-bit quantized model fits comfortably; 7B 4-bit fills all RAM leaving no room for agent code. **No Ollama/runtime is pre-installed** — bundle weights + runtime in the image (10 GB compressed limit still applies).

**Ops:** submissions rate-limited to **10/hour per team** (test locally first). The registry pull counter (GitHub Packages/Docker Hub) shows whether graders have pulled your image yet.
