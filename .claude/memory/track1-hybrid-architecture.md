---
name: track1-hybrid-architecture
description: Track 1 target architecture — local-first hybrid (local 2-3B answers + verify + escalate to Fireworks), documented in documentation/architecture-1.md
metadata:
  type: project
---

Target architecture for the Track 1 agent, decided 2026-07-09 and written up in `documentation/architecture-1.md` (with a mermaid flow diagram). Supersedes the pure Fireworks-only routing design in [[track1-project]]. Driven by the published facts in [[track1-launch-day-facts]] (80% gate = ≥16/19; local answers cost zero Fireworks tokens).

**Core principle:** ranking is fewest-tokens-among-passers and only tokens through `FIREWORKS_BASE_URL` count, so every task answered locally AND verified is free. Optimize for that.

**Decisions (Wayne proposed a lightweight-router → local-vs-Fireworks split; refined to):**
1. **One local model, two jobs — NOT a separate router model.** A single bundled 2–3B 4-bit instruct model does zero-shot category classification AND answers light tasks. 7B fills all 4GB RAM. **No trained router, no trained answerer** — unseen prompt variants make training on our synthetic phrasing an overfitting trap; off-the-shelf instruct models handle the 8 generic categories.
2. **Category selects policy, not final escalation** — it picks prompt template + max_tokens cap + which verifier runs. Difficulty is per-instance, not per-category, so category-only escalation misroutes easy-instances-of-hard-categories (wasted tokens) and hard-instances-of-easy-categories (wrong answers).
3. **Escalate on verification/confidence, not category.** Try local → verify → escalate on fail. Verifiers: math→recompute/numeric check; code→execute vs asserts; sentiment/NER→format+label-set validation; factual/summary/logic→confidence/self-consistency. Captures easy instances of every category for free; spends Fireworks tokens only where local falls short.
4. **The local-vs-Fireworks map is data-driven** — benchmark the chosen 2–3B model per category (`benchmark/dataset.py`, no submission slot) before wiring defaults. Hypothesis: sentiment/NER/factual/summarization go local; math/logic/code-gen lean Fireworks.

**Why the regex router is NOT the thing to rebuild:** a misroute only mis-selects template/cap/verifier, not the answer, so it's low-stakes; the real leverage is moving tasks off Fireworks entirely, not perfecting classification. **How to apply:** keep the heuristic (or zero-shot local) classifier as a cheap first pass; put effort into the local answerer + verifiers + escalation.

**Blocking open question / next action:** how well does a specific 2–3B model answer each of the 8 categories? Benchmark to build the real per-category pass/fail map before implementing.
