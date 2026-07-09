# AMD Developer Hackathon (Track 1) Notes

---

# Part 0 — Launch-Day Facts (published, confirmed)

> These override the earlier "unknown until launch day" placeholders. Source: official Discord clarifications + Participant Guide PDF.

## Allowed models (`ALLOWED_MODELS`)

These are the exact model IDs. Still read them from the env var at runtime — do **not** hardcode — but these are what launch day published:

- `minimax-m3`
- `kimi-k2p7-code`
- `gemma-4-31b-it`
- `gemma-4-26b-a4b-it`
- `gemma-4-31b-it-nvfp4`

## Scoring specifics

- **Accuracy gate = 80%.** Below it, you're excluded from the leaderboard regardless of token count.
- **Exactly 19 fixed tasks.** Every score is `n/19`, so the visible percentages are just arithmetic: `16/19 = 84.2%` (passes), `15/19 = 78.9%` (**fails** the 80% gate). **You need ≥ 16 of 19 correct.**
- The **LLM judge is not perfectly deterministic** run-to-run — the same image can score slightly differently. Build margin above 16/19; don't sit exactly on the line.
- Passers are ranked ascending by **total tokens recorded by the Fireworks proxy** (fewest wins).

## Local models are a first-class strategy (biggest lever)

- A container may answer tasks with a **bundled local model**; those answers **count fully toward accuracy**.
- **Only tokens routed through `FIREWORKS_BASE_URL` count toward the token score.** A local model that answers correctly uses **zero Fireworks tokens — the best possible ranking outcome.**
- `flagged: ZERO_API_CALLS` (a local-only run makes no proxy calls) is **not a failure** — it's an explicitly valid strategy.
- **Grading environment: 4 GB RAM, 2 vCPU.** A 2B–3B 4-bit quantized model fits comfortably; a 7B 4-bit model fills the whole RAM budget, leaving no room for agent code. Size accordingly.
- **No Ollama or model runtime is pre-installed** — bundle the model weights + runtime directly in the Docker image. The 10 GB compressed image limit still applies.
- Implication: the ideal design answers the easy/deterministic categories locally (0 tokens) and only escalates to Fireworks when the local model is likely wrong — or, if a small local model clears 16/19 on its own, calls Fireworks **zero** times.

## Operational limits

- **Submissions are rate-limited to 10 per hour per team.** Test locally before spending a slot.
- Your registry's **download/pull counter** (GitHub Packages, Docker Hub) indicates whether the graders have pulled your image yet — useful while the backlog clears.
- General (all tracks): container ready < 60 s, < 30 s per request, English only, exit 0 on success.

---

# Part 1 — Understanding the Challenge

## What is the overarching idea?

The goal of Track 1 is **not** to build a chatbot or conversational assistant.

Instead, the challenge is to build an **AI agent that automatically processes a batch of natural language tasks**. The judging system will provide your program with a list of prompts, your agent must solve every prompt, save the results, and then terminate.

Conceptually, the workflow looks like this:

```text
Evaluator
    │
    │ provides
    ▼
tasks.json
    │
    ▼
Your Docker Container
    │
    ├── Read tasks
    ├── Process each prompt
    ├── Call Fireworks AI models
    └── Save responses
    ▼
results.json
```

Unlike a normal chatbot:

- There is **no frontend**.
- There is **no API endpoint**.
- There is **no user interaction**.

Everything happens automatically inside a Docker container.

---

## Input Format

When the container starts, it will receive a JSON file at:

```text
/input/tasks.json
```

Example:

```json
[
  {
    "task_id": "t1",
    "prompt": "What is Newton's First Law?"
  },
  {
    "task_id": "t2",
    "prompt": "Summarise the following article..."
  }
]
```

---

## Output Format

Before exiting, the container must produce:

```text
/output/results.json
```

Example:

```json
[
  {
    "task_id": "t1",
    "answer": "Newton's First Law states that..."
  },
  {
    "task_id": "t2",
    "answer": "The article discusses..."
  }
]
```

The judging system will read this file to evaluate your submission.

---

## Capability Areas

The agent should handle all eight categories:

1. Factual knowledge
2. Mathematical reasoning
3. Sentiment classification
4. Text summarisation
5. Named Entity Recognition (NER)
6. Code debugging
7. Logical reasoning
8. Code generation

Importantly, the evaluation uses **unseen prompts**, meaning that hardcoding answers or memorizing datasets will not work.

---

## Scoring

The competition uses a two-stage scoring system.

### Stage 1 — Accuracy

Your submission must first pass an accuracy threshold.

If it does not reach this threshold, it is removed from the leaderboard.

---

### Stage 2 — Token Efficiency

Once the accuracy threshold has been met, submissions are ranked according to:

> **Total number of tokens consumed.**

Lower token usage results in a higher ranking.

This means the objective is **not simply to maximize accuracy**, but rather to achieve **sufficiently high accuracy while minimizing token consumption**.

---

## Fireworks AI Requirements

Your container must obtain the following values from environment variables supplied by the judging harness:

```text
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

Your application should **never hardcode**:

- API keys
- Base URLs
- Model names

Every inference request must go through the supplied `FIREWORKS_BASE_URL`.

---

## Submission Process

The submission consists of a **Docker image**.

Typical workflow:

```text
Develop locally
        ↓
Build Docker image
        ↓
Test locally
        ↓
Push image to a public registry
        ↓
Submit image reference
```

The image:

- must be publicly accessible
- must support `linux/amd64`
- must complete execution within the competition limits

---

# Part 2 — Recommended Development Strategy

## Should I fine-tune a model?

Probably **not**.

The evaluation spans many unrelated tasks:

- mathematics
- coding
- reasoning
- summarization
- factual knowledge
- NER
- sentiment analysis

Fine-tuning generally improves performance within a narrow domain, whereas this competition intentionally covers many domains.

Furthermore, the leaderboard prioritizes **token efficiency**, not simply benchmark accuracy.

For these reasons, engineering a smart inference pipeline is likely to produce greater gains than training a specialized model.

---

## Think of the Challenge as an AI Systems Problem

Instead of training a new model, consider building an intelligent system around existing models.

Conceptually:

```text
Incoming Prompt
        │
        ▼
Task Classifier
        │
        ├── Mathematics
        ├── Coding
        ├── Summarization
        ├── Sentiment
        ├── NER
        └── General Knowledge
```

Once the task type is identified:

```text
Simple Task?
        │
       Yes
        │
        ▼
Small / Cheap Model

Complex Task?
        │
       Yes
        │
        ▼
Larger / More Capable Model
```

This routing strategy can significantly reduce token usage while maintaining competitive accuracy.

---

## Suggested Project Structure

```text
agent/
│
├── main.py
├── firework_client.py
├── router.py
├── prompts.py
├── requirements.txt
├── Dockerfile
│
├── input/
│     tasks.json
│
└── output/
      results.json
```

---

## Suggested Execution Flow

```text
Container Starts
        │
        ▼
Read tasks.json
        │
        ▼
For each task
        │
        ▼
Determine task category
        │
        ▼
Select appropriate model
        │
        ▼
Generate response
        │
        ▼
Store result
        │
        ▼
Write results.json
        │
        ▼
Exit
```

---

## Development Roadmap

### Phase 1 — Build a Working Baseline

Implement the complete pipeline:

- Read `tasks.json`
- Call a Fireworks model
- Produce `results.json`
- Package everything into Docker

The goal is simply to satisfy the competition interface.

---

### Phase 2 — Build a Lightweight Task Router

Identify which category each prompt belongs to.

Possible categories include:

- Summarization
- Coding
- Mathematics
- Sentiment
- NER
- General Question Answering

The router can be based on lightweight heuristics or a small model.

---

### Phase 3 — Optimize Prompting

Reduce unnecessary token usage by:

- Writing concise system prompts
- Avoiding unnecessary reasoning requests
- Limiting output length where appropriate
- Reusing prompt templates

Small prompt improvements can noticeably reduce total token consumption.

---

### Phase 4 — Optimize Model Selection

When the list of allowed Fireworks models becomes available:

- Benchmark each model.
- Measure accuracy.
- Measure token consumption.
- Route different categories to different models if beneficial.

For example:

- Smaller model for sentiment or NER.
- Larger model for mathematical reasoning or code generation.

---

### Phase 5 — Measure and Iterate

Run representative workloads through your system.

Track:

- Accuracy
- Latency
- Token usage

Refine the routing logic and prompting strategy to achieve the best balance between performance and efficiency.

---

# Overall Recommendation

This hackathon is best approached as an **AI systems engineering challenge**, rather than a machine learning challenge.

Rather than spending time fine-tuning a single model, focus on building an efficient inference pipeline that intelligently routes tasks, minimizes token usage, and consistently produces accurate answers.

A well-engineered orchestration system is likely to outperform a single, heavily customized model under the competition's scoring rules.