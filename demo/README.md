# Demo app — Streamlit showcase for the Track 1 agent

`streamlit_app.py` (at the repo root) is an interactive showcase of the batch
agent. It's **not** the graded deliverable — that's the Docker image
(`documentation/docker-submission.md`). The demo reuses the agent's *real* code
(router, per-category policy, prompts, Fireworks client) so what you see mirrors
the container.

- **Live pipeline tab** — router classification (0 tokens) + the real Fireworks
  escalation call, with live token counts.
- **Benchmark story tab** — replays a captured container run (`demo/trace.json`)
  if present; otherwise shows the live router classification of the sample tasks.

## Deploy to Streamlit Community Cloud (~2 min, free)

1. Push this repo to GitHub (already at `github.com/WayneCh0y/amd-hack`) and make
   it public (or grant the deploy access).
2. Go to **https://share.streamlit.io** → sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `WayneCh0y/amd-hack`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. **Advanced settings → Secrets** — paste (TOML):
   ```toml
   FIREWORKS_API_KEY = "sk-..."
   FIREWORKS_BASE_URL = "https://.../inference/v1"
   ALLOWED_MODELS = "gemma-4-31b-it,gemma-4-26b-a4b-it,kimi-k2p7-code"
   ```
   (Router works without these; secrets only enable the live Fireworks call.)
5. **Deploy.** You'll get a public URL like
   `https://<something>.streamlit.app` — that's the "Demo Application URL" for
   the submission form.

> Streamlit installs from the repo-root `requirements.txt` (streamlit + openai).
> `llama-cpp-python` is intentionally excluded — the free tier can't run the 3B
> model, and the demo doesn't load it.

## Optional: capture a real benchmark trace

To make the Benchmark tab replay a real container run (local answers + verdicts +
true token cost), run `capture_trace.py` **inside the Docker image** (a bare
Windows/mac host usually SIGILLs on the prebuilt llama.cpp wheel):

```bash
docker build -t amd-track1:dev agent
MSYS_NO_PATHCONV=1 docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -e LOCAL_MODEL_PATH=/models/model.gguf \
  -v "$PWD:/work" --entrypoint python amd-track1:dev \
  /work/demo/capture_trace.py --out /work/demo/trace.json
```

Commit `demo/trace.json`; the deployed app picks it up on the next redeploy.

## Run locally

```bash
cd /c/Me/Projects/amd-hack
agent/.venv/Scripts/python -m streamlit run streamlit_app.py
```
