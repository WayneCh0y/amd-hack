# Architecture 1 — Hybrid Local-First Agent (Track 1)

> Proposed architecture for the Track 1 agent. Goal: clear the **80% gate (≥ 16 / 19 tasks)**
> while spending the **fewest Fireworks tokens**, by answering as many tasks as possible with a
> bundled **local model at zero Fireworks-token cost** and escalating to the Fireworks API only
> when a local answer can't be trusted.
>
> Related: [track-1.md](track-1.md) (Part 0 — Launch-Day Facts). Supersedes the pure
> Fireworks-only routing design as the target architecture.

## Core principle

Ranking is *fewest tokens among gate-passers*, and **only tokens routed through
`FIREWORKS_BASE_URL` count**. A local model that answers correctly = **zero** Fireworks tokens =
the best possible ranking outcome (`flagged: ZERO_API_CALLS` is a valid, non-failing result). So
every task we can answer locally *and verify* is free — the whole design optimizes for that.

## Key design decisions

1. **One local model, two jobs — no separate router model.** On the 4 GB RAM / 2 vCPU grading
   box a second model is dead weight. A single bundled **2–3B 4-bit instruct model** does both
   category classification (zero-shot, one-line prompt) *and* answering the light tasks. A 7B
   4-bit model fills all RAM and leaves no room for agent code, so stay at 2–3B. **No trained
   router and no trained answerer** — the eval uses unseen prompt variants, so training on our own
   synthetic phrasing risks overfitting for little gain. Off-the-shelf instruct models already
   handle these 8 generic categories.

2. **Category selects policy, not the final escalation.** The category tag chooses the prompt
   template, the `max_tokens` cap, and *which verifier* runs. It is **not** the sole trigger for
   going to Fireworks, because difficulty lives in the *instance*, not the category ("2 + 2" and a
   4-step projection are both "math"). Routing purely by category would send easy instances of
   hard categories to Fireworks (wasted tokens) and keep hard instances of easy categories local
   (wrong answers).

3. **~~Escalate on verification/confidence, not on category.~~ → REFUTED. Escalate on category;
   answer locally only where a verifier can check something real.**

   > **This decision was wrong, and it is what failed the 80% accuracy gate** (submission status
   > `ACCURACY_GATE_FAILED`). The original plan was "try every category locally, verify, escalate
   > on failure." The flaw: **without ground truth, a 'verifier' can only check shape, and shape
   > cannot see wrongness.** Measured on the participant guide's own practice tasks (2 vCPU / 4 GB,
   > Qwen2.5-3B Q4), the local model scored **4/8** and every miss sailed through its verifier:
   >
   > | Task | Local answer | Verifier said | Truth |
   > |---|---|---|---|
   > | factual | "Canberra; near the **Australian Alps**" | ✅ non-empty | ❌ asked for a *body of water* |
   > | math | invented a step, answered **108** | ✅ "has a number" | ❌ 144 |
   > | sentiment | bare **"Negative"** | ✅ "has a label" | ❌ mixed review, no justification |
   > | NER | *(timed out at 45 s)* | — | ❌ empty answer shipped |
   >
   > A well-formed wrong answer is *indistinguishable* from a well-formed right one at zero cost.
   > Pretending otherwise converted the local model's ~50% accuracy directly into the submission's
   > accuracy.

   The corrected rule: **a category is answered locally only if we can verify something true about
   the answer.** Two checks qualify, and only two:
   - **code generation / debugging** → extract the code and *actually execute it* in a subprocess.
     Code that doesn't parse, throws on import, or never terminates is objectively broken. ✅ local
   - **summarization** → the prompt *states* the constraint ("in exactly one sentence", "in 50
     words"), so we can check the answer against it, and against the source (a summary must
     compress, and must not be a verbatim copy). ✅ local
   - **factual / math / sentiment / NER / logic** → no sound ground-truth-free check exists.
     **Not answered locally at all.** ❌ Fireworks

   Escalating a good answer wastes tokens; keeping a wrong one loses the gate — and the gate is
   binary. The asymmetry is not close, so we don't gamble on the unverifiable categories.

4. **The local-vs-Fireworks map is data-driven, not assumed.** ✅ **Resolved** — see the table in
   "Measured results" below. Note that the *first* benchmark (36/40 on our own synthetic tasks,
   with our own keyword/numeric checkers) badly overstated local accuracy: lenient checkers on
   self-written tasks are not a proxy for an LLM judge on unseen ones. The practice tasks published
   in the guide were the first honest signal.

## Flow

The two phases **overlap**: the Fireworks calls are IO-bound and the local phase is CPU-bound
(llama.cpp releases the GIL while decoding), so the local phase costs almost no extra wall clock.

```mermaid
flowchart TD
    A[/input/tasks.json/] --> B[Classify category<br/>regex heuristic, 0 tokens]
    B --> C{Category has a<br/>real verifier?}

    C -->|"factual · math · sentiment<br/>NER · logic"| D[Fireworks API<br/>via FIREWORKS_BASE_URL<br/>tokens spent here only]

    C -->|"code_gen · code_debug<br/>summarization"| E[Local 3B model<br/>0 Fireworks tokens]
    E --> F{Verify}
    F -->|code| F1[Execute in a subprocess:<br/>parses? runs? terminates?]
    F -->|summary| F2[Obeys the stated length<br/>constraint? compresses?<br/>not a copy?]

    F1 --> G{Pass?}
    F2 --> G
    G -->|Yes| H[Keep local answer<br/>0 tokens]
    G -->|No / truncated / budget spent| D

    D --> I{Got an answer?}
    I -->|Yes| J[Collect result]
    I -->|No — API down| K[Fall back to the local draft,<br/>never to an empty string]
    H --> J
    K --> J
    J --> L[/output/results.json/]
    L --> M[Exit 0]
```

## Measured results

Guide's 8 practice tasks, in-container, `--cpus=2 --memory=4g` (the grading box's shape):

| Configuration | Accuracy | Fireworks tokens | Runtime |
|---|---|---|---|
| Local-first everywhere (the version that failed the gate) | **4/8 (50%)** | — | 300 s (budget-capped) |
| All-Fireworks | 8/8 | 1,881 | 3 s |
| **Verified subset (current)** | **8/8 (100%)** | **1,049 (−44%)** | 95 s |

The verified subset answers summarization + both code tasks locally at **zero tokens**, all three
passing verification, and routes the other five to Fireworks.

## Never ship an empty answer

`_FALLBACK_ANSWER = ""` used to be the response to any failure. An empty answer is graded **wrong
with certainty**; an unverified local draft is merely *likely* wrong — strictly better. So the
precedence is now: Fireworks answer → the local draft we already generated (free) → one unverified
local generation → only then `""`. In the failed run, an NER task timed out locally and shipped an
empty string; that path no longer exists.

## What this changes vs. the pre-mortem implementation

- The regex router is **not** the accuracy bottleneck — a misroute mis-selects a template and a
  cap, not the answer. It now also selects *which backend*, so a misroute into a local-eligible
  category is the one costly case; the verifiers catch it (a "summary" of a math question won't
  respect a constraint that isn't there... but it will be *kept*, so keep the router's
  local-eligible patterns tight).
- Local inference costs **~31–45 s/task on 2 vCPU**, so the local budget only ever covers a handful
  of tasks. This is fine now that local handles a *subset* by design, but it means "answer
  everything locally" was never reachable on this hardware anyway.

## Constraints this must respect

- Local model sized for **4 GB RAM / 2 vCPU** (2–3B, 4-bit). **No Ollama/runtime pre-installed** —
  bundle weights + runtime in the image; **≤ 10 GB compressed**; container **ready < 60 s**;
  **< 30 s per request**; **≤ 10 min** total; English only; exit 0 on success.
- Read `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` from env only; route all
  Fireworks calls through `FIREWORKS_BASE_URL`; only call models in `ALLOWED_MODELS`.

## The lesson, stated plainly

**Token efficiency is a ranking; accuracy is a gate.** Optimising the ranking at the expense of the
gate scores zero, and that is precisely the trade the original local-first design made — it spent
300 s of CPU to *lower* the submission's accuracy. Local inference is only free if the answer is
right, so the local model may only be trusted where being wrong is *detectable*. Everywhere else,
paying tokens is the cheap option.
