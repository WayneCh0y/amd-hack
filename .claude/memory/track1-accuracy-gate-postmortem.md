---
name: track1-accuracy-gate-postmortem
description: Track 1 post-mortem — why the local-first agent got ACCURACY_GATE_FAILED, and the verified-subset fix that took it from 4/8 to 8/8 on the practice tasks
metadata:
  type: project
---

The first real Track 1 submission ("Local-First Batch Agent") returned **`ACCURACY_GATE_FAILED`** (2026-07-11). Diagnosed and fixed the same day. This supersedes decision #3 of [[track1-hybrid-architecture]] and corrects the optimism in [[track1-local-model-results]].

**What `ACCURACY_GATE_FAILED` rules out — and why that's the whole diagnosis.** The status means the image pulled, the container exited 0, `results.json` was schema-valid, only allowed models were called, and it ran in time. Every piece of plumbing was verified correct (confirmed by running the container at `--cpus=2 --memory=4g` and calling the real API: the Fireworks leg answered 8/8 in 2.8 s). So the bug could only be *answer quality* — i.e. the local model. Don't go looking for routing/format/client bugs when you see this status.

**Root cause: shape checks cannot see wrongness.** The agent tried every category on the bundled Qwen2.5-3B and kept the answer unless it was empty/malformed. `verifiers.is_trustworthy` "verified" math by *contains a number*, sentiment by *contains a label*, and factual/summary/logic by *is non-empty*. On the guide's own practice tasks the local model scored **4/8**, and every miss passed its verifier: answered 108 (truth: 144), said Canberra is near the "Australian Alps" (asked for a body of *water*), replied bare "Negative" to a mixed review, and timed out on NER → shipped `""`. **Without ground truth, a well-formed wrong answer is indistinguishable from a right one.** The eval prompts are unseen, so there is no ground truth, ever.

**The fix — answer locally only where a verifier can check something TRUE** (`categories.LOCAL_OK`):
- `code_gen` / `code_debug` → **execute the code in a subprocess**. Doesn't parse / throws on import / never terminates = objectively broken.
- `summarization` → the prompt *states* the constraint ("in exactly one sentence", "in 50 words"); check the answer against it, and against the source (must compress, must not be a verbatim copy).
- `factual` / `math` / `sentiment` / `ner` / `logic` → **no sound check exists → never answered locally.** Straight to Fireworks.

**Result (8 practice tasks, in-container, 2 vCPU / 4 GB):** 4/8 → **8/8**, tokens 1,881 (all-Fireworks) → **1,049 (−44%)**, 95 s, exit 0. Local answers 3 tasks at zero tokens, all passing verification.

**Why: token efficiency is a *ranking*; accuracy is a *gate*.** Optimising the ranking at the expense of the gate scores zero. The old design spent 300 s of CPU to *lower* accuracy — the worst possible trade. Local inference is only free **if the answer is right**.

**How to apply / other traps fixed:**
- **Never ship `""`.** An empty answer is wrong with certainty; an unverified local draft is only *likely* wrong. Precedence: Fireworks → local draft (already generated, free) → one unverified local gen → `""`.
- **Prompts were losing answers too:** FACTUAL said "as few words as the question needs" and dropped the second half of a two-part question; SENTIMENT offered no "Mixed" label and made the justification optional (the guide's category is *label **and** justify*). Both fixed.
- Local costs **~31–45 s/task on 2 vCPU** (not the 15 s the config assumed), so the local budget only ever covers a handful of tasks — "answer everything locally" was never reachable on this hardware regardless.
- The local + Fireworks phases now **overlap** (llama.cpp releases the GIL while decoding), so the local phase is off the critical path.
