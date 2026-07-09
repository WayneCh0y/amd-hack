# Project Memory Index

- [Track 1 Project](track1-project.md) — what amd-hack is: AMD Hackathon Track 1 batch AI agent, constraints, scoring, design decisions
- [Track 1 Launch-Day Facts](track1-launch-day-facts.md) — published: real ALLOWED_MODELS list, 80% gate over 19 tasks (need ≥16), zero-token local-model strategy, grading env limits
- [Track 1 Hybrid Architecture](track1-hybrid-architecture.md) — target design (documentation/architecture-1.md): local 2-3B answers + per-category verify + escalate to Fireworks; one model not a separate router; no training
- [Track 1 Launch-Day Tuning](track1-launch-day-tuning.md) — real-Fireworks lessons: reasoning_effort is the main token lever, watch for empty answers & in-content reasoning
- [Track 1 Benchmark Harness](track1-benchmark-harness.md) — scripts/benchmark.py measures accuracy×tokens per model per category; run it on real ALLOWED_MODELS on launch day
- [Track 1 Local Model Results](track1-local-model-results.md) — Qwen2.5-3B-Instruct Q4 scores 36/40 (90%) on CPU via llama-cpp-python; nearly clears the gate at zero tokens; Docker/AVX2 build setup + gotchas
- [Presentation Manim Setup](presentation-manim-setup.md) — video deck in presentation/slides.py (manim + manim-slides); render commands, global-Python-not-agent-venv, PyAV-encodes-not-ffmpeg, teardown gotchas
