# Project Memory Index

- [Track 1 Project](track1-project.md) — what amd-hack is: AMD Hackathon Track 1 batch AI agent, constraints, scoring, design decisions
- [Track 1 Launch-Day Facts](track1-launch-day-facts.md) — published: real ALLOWED_MODELS list, 80% gate over 19 tasks (need ≥16), zero-token local-model strategy, grading env limits
- [Track 1 Accuracy-Gate Post-Mortem](track1-accuracy-gate-postmortem.md) — **why we got ACCURACY_GATE_FAILED and the fix**: shape checks can't see wrongness, so only answer locally where a verifier checks something true (code→execute, summary→stated constraint); 4/8 → 8/8, −44% tokens
- [Track 1 Timeout Post-Mortem](track1-timeout-postmortem.md) — **why we then got TIMEOUT and the fix**: a budget checked only before starting work is not a budget; unbounded waits on the llama.cpp lock + a deadline-blind retry ladder; watchdog force-writes results and exits 0
- [Track 1 Hybrid Architecture](track1-hybrid-architecture.md) — target design (documentation/architecture-1.md); its "verify everything locally" decision is REFUTED — see the post-mortem
- [Track 1 Launch-Day Tuning](track1-launch-day-tuning.md) — real-Fireworks lessons: reasoning_effort is the main token lever, watch for empty answers & in-content reasoning
- [Track 1 Benchmark Harness](track1-benchmark-harness.md) — scripts/benchmark.py measures accuracy×tokens per model per category; run it on real ALLOWED_MODELS on launch day
- [Track 1 Local Model Results](track1-local-model-results.md) — Qwen2.5-3B on CPU: our harness said 90%, the guide's practice tasks said 50%; the gap was lenient self-written checkers. Docker/AVX2 build setup + gotchas
- [Presentation Manim Setup](presentation-manim-setup.md) — video deck in presentation/slides.py (manim + manim-slides); render commands, global-Python-not-agent-venv, PyAV-encodes-not-ffmpeg, teardown gotchas
