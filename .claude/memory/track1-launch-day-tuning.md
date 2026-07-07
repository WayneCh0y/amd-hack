---
name: track1-launch-day-tuning
description: Hard-won tuning lessons for Track 1 from real Fireworks testing — reasoning tokens are the main lever
metadata:
  type: project
---

Lessons from a **real Fireworks run** of the Track 1 agent (see [[track1-project]]), on 2026-07-07 with the account's available models (deepseek-v4-pro, glm-5p1, glm-5p2, gpt-oss-120b, kimi-k2p5, kimi-k2p6; flux is image-only — exclude). These matter most on launch day when the real `ALLOWED_MODELS` appears.

**1. Reasoning tokens are the biggest token-efficiency lever.** 2026 Fireworks models reason by default. On `gpt-oss-120b`, default effort spent the entire `max_tokens` budget on hidden thinking and returned **empty content** for a tight (48-token) sentiment cap. Setting `reasoning_effort=low` produced a clean answer AND cut completion tokens ~27% (batch total 2127→1831). `none` is NOT supported (400); valid values are low/medium/high. The agent now sends `reasoning_effort` (config default "low", env `REASONING_EFFORT`) and transparently retries without it if a model rejects the param (`fireworks_client._maybe_disable_reasoning`).

**2. Two model families behave differently.** `gpt-oss-120b` keeps reasoning in a separate channel and returns only the clean final answer → great. `glm-5p1` dumps its "1. Analyze the Request… 2. Formulate…" scaffolding INTO `content`, which both inflates tokens and gets truncated mid-thought → unusable for our purposes. Prefer models that separate reasoning for any tier. The size-based selector's reasoning de-prioritization only catches IDs containing r1/qwq/thinking — it does NOT catch glm's behavior, so eyeball outputs on launch day.

**3. Never let caps truncate the answer to empty.** Raised `max_tokens` floors in `categories.py` (sentiment 48→256, others 512, hard tasks 1024). Caps are ceilings, not targets — a concise model stops early — but too-tight caps + hidden reasoning = empty answer = fails the accuracy gate = removed from leaderboard. Err high.

**Launch-day checklist:** (a) confirm which allowed models are chat vs image; (b) run `scripts/run_local.py` on `examples/tasks.json`, eyeball all 8 answers for correctness AND for in-content reasoning scaffolding; (c) keep `reasoning_effort=low` unless accuracy drops on math/logic (then try medium for just those); (d) if a category misses the gate, bump its tier SMALL→LARGE in `categories.py`.
