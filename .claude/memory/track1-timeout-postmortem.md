---
name: track1-timeout-postmortem
description: Track 1 post-mortem — why the agent got TIMEOUT after the accuracy fix, and the deadline/watchdog changes that bound every wait
metadata:
  type: project
---

The submission after the accuracy fix returned **`TIMEOUT`** (checked 2026-07-12) — the container did not finish inside 10 minutes, so it scored zero. Fixed the same day. Sits alongside [[track1-accuracy-gate-postmortem]]; both were caused by the local model, in opposite ways.

**Root cause: the run had no deadline, only a "may I start?" check.** `time_budget` was tested *before* dispatching work and never again, so anything already in flight ran as long as it liked:

1. **The regression.** The accuracy fix added `_last_resort_local` (one unverified local generation when Fireworks returns nothing). It checked the clock *before* queueing on `LocalModel._gen_lock` — a plain unbounded `with self._gen_lock:`. Local generations serialize on one llama.cpp context at ~45 s each, so when the API is degraded **every** task takes this path and they pile up behind the lock; nothing re-checks the clock after the wait. `ThreadPoolExecutor.__exit__` then waits for all of them. This is what turned a *scored* failure into an unscored one.
2. **The retry ladder was deadline-blind.** 2 tiers × (`max_retries` 2 + 1) × `request_timeout` 25 s = **150 s for a single task**, and a task could start at t=539 s against a 540 s budget. ("Excessive retries" is literally what the guide lists under `TIMEOUT`.)
3. **`local_task_timeout` never bounded prefill.** It works by checking the clock *between streamed tokens*, but llama.cpp yields no token until the whole prompt is evaluated — and prefill is the dominant cost on 2 vCPU. A long summarization passage blows through the cap unchecked. **Prefill cannot be interrupted from Python.**
4. **Escalations were banked until the local phase ended**, stacking the entire Fireworks tail into the back half of the container's life.

**The fix, in layers:**
- **Watchdog (`main._start_watchdog`) — the actual guarantee.** Daemon thread; at `hard_budget` (510 s) it writes whatever answers exist and `os._exit(0)`s from under the running threads. `os._exit` is deliberate: a clean shutdown joins workers, and a thread stuck in llama.cpp's C code will not join. Armed **before** the model load, so a stalled load still produces a file. Needed because #3 means no amount of bookkeeping can *guarantee* a generation returns.
- **A partial results file always beats a missing one.** Unanswered tasks just grade wrong; `TIMEOUT` discards the answers you did get. `Pipeline.results_for()` publishes answers as they land and is always schema-valid.
- **Deadlines are enforced *inside* every wait**, not just before it: the Fireworks client clamps each attempt's HTTP timeout to time remaining and stops retrying/backing off when there's none; `LocalModel` acquires `_gen_lock` with a timeout and **re-checks the deadline once it holds the lock**.
- **Bound the input, since you can't bound prefill:** `local_max_prompt_chars` (6000) keeps oversized prompts out of the local phase entirely.
- Local failures escalate to the pool **the moment they're known**, overlapping with the remaining local work.
- Budgets: `time_budget` 540→**420** (soft), `hard_budget` **510**, `local_budget` 300→**150**, `max_retries` 2→**1**, `local_task_timeout` 60→**45**.

**Also fixed: a local-model load crash used to kill the container** (exit non-zero → `RUNTIME_ERROR`). `_init_local_model` caught only `LocalModelError`, but loading a GGUF drops into C — a bad ISA raises `OSError` (0xc000001d/SIGILL) and 1.9 GB on a 4 GB box can raise `MemoryError`. It now catches `Exception` and degrades to Fireworks-only. **The local model is an optimisation; nothing about it is worth failing the run for.**

**How to apply:** a wall-clock budget you only check before starting work is not a budget. Every blocking wait (lock, HTTP, subprocess) needs the deadline passed *into* it. And when a dependency can't be interrupted at all — C extensions, llama.cpp prefill — the only real answer is an out-of-band watchdog that writes output and force-exits.

**Timing after the fix** (19 tasks): healthy ≈ 200 s (load ~40 s + local phase ≤150 s, overlapped); dead-API worst case terminates at the 420 s soft deadline; watchdog backstop at 510 s; ~90 s margin under the 600 s kill. Verified: black-hole API → exit 0 at the deadline; forced watchdog → exit 0 with a complete file; real API happy path → 8/8 in 3.0 s.
