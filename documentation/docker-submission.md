# Docker Registry & Submission Guide (Track 1)

How to build the agent image, push it to a **public** registry, verify it, and
get the image reference you paste into the submission. The deliverable for Track
1 is a single publicly-pullable `linux/amd64` Docker image (≤ 10 GB) — not a repo.

> Commands assume **Windows + Git Bash** (the `MSYS_NO_PATHCONV=1` prefix and
> `C:/...` paths on `-v` mounts stop Git Bash from mangling container paths). On
> macOS/Linux, drop that prefix and use normal paths.

---

## 0. Prerequisites (once)

1. **Docker Desktop running.** Check: `docker info` prints a server section.
2. **Model weights present.** The GGUF is git-ignored, so on a fresh checkout
   re-download it — the build copies it into the image:
   ```bash
   cd /c/Me/Projects/amd-hack/agent
   ls models/*.gguf || python -c "from huggingface_hub import hf_hub_download as d; \
     d('bartowski/Qwen2.5-3B-Instruct-GGUF','Qwen2.5-3B-Instruct-Q4_K_M.gguf', local_dir='models')"
   ```
3. **A registry account** — Docker Hub (simplest) or GitHub Container Registry
   (GHCR). Pick one and follow that section below.

---

## Option A — Docker Hub (recommended)

### A1. Log in
```bash
docker login
# username = your Docker Hub ID; password = an access token
# (Docker Hub → Account Settings → Personal access tokens → Generate)
```

### A2. Build + push (linux/amd64)
Replace `<dockerhub-user>` with your Docker Hub username.
```bash
cd /c/Me/Projects/amd-hack/agent
docker buildx build --platform linux/amd64 \
  -t docker.io/<dockerhub-user>/amd-track1:v1 \
  -t docker.io/<dockerhub-user>/amd-track1:latest \
  --push .
```
- Building on an amd64 PC = native, no emulation. First build compiles llama.cpp
  (a few minutes); later builds are cached.
- Two tags: a **versioned** `:v1` (submit this — it's immutable) plus `:latest`.

### A3. Make it public
Docker Hub → your `amd-track1` repo → **Settings** → **Make public**.

### A4. Your image reference
```
docker.io/<dockerhub-user>/amd-track1:v1
```
(You can also write it as `<dockerhub-user>/amd-track1:v1` — Docker Hub is the default registry.)

---

## Option B — GitHub Container Registry (GHCR)

### B1. Log in
```bash
# Create a classic PAT with scope: write:packages  (GitHub → Settings →
# Developer settings → Personal access tokens → Tokens (classic))
echo "<your-PAT>" | docker login ghcr.io -u <github-user> --password-stdin
```

### B2. Build + push
```bash
cd /c/Me/Projects/amd-hack/agent
docker buildx build --platform linux/amd64 \
  -t ghcr.io/<github-user>/amd-track1:v1 \
  -t ghcr.io/<github-user>/amd-track1:latest \
  --push .
```

### B3. Make it public
GitHub → your profile → **Packages** → `amd-track1` → **Package settings** →
**Change visibility** → **Public**.

### B4. Your image reference
```
ghcr.io/<github-user>/amd-track1:v1
```

---

## 1. Verify the pushed image (do this BEFORE submitting)

Pull it **fresh** (proves the public image is complete and self-contained), then
run it exactly the way the grader will — `/input/tasks.json` in,
`/output/results.json` out, only the three env vars. Use **real** Fireworks
credentials so any escalation path is exercised.

```bash
IMAGE=docker.io/<dockerhub-user>/amd-track1:v1     # or your ghcr.io/... ref

docker pull "$IMAGE"

mkdir -p /c/t/sub-out
MSYS_NO_PATHCONV=1 docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "C:/Me/Projects/amd-hack/agent/examples:/input" \
  -v "C:/t/sub-out:/output" \
  "$IMAGE"

echo "exit: $?"                    # must be 0
cat /c/t/sub-out/results.json      # valid JSON, one { task_id, answer } per input
```
Watch the final log line: `... tokens ... total=N` is your token count (ideally
near 0 — local answers are free). If exit is 0 and the JSON looks right, submit.

---

## 2. Submit
Paste the **image reference** (e.g. `docker.io/<dockerhub-user>/amd-track1:v1`)
into the submission form / evaluation channel. Confirm from the Participant Guide
+ Discord whether the reference goes on the lablab.ai form or a separate
evaluation bot/endpoint (the "10 submissions/hour" limit suggests a separate
endpoint).

---

## 3. Iterating / re-submitting
- Bump the version tag each rebuild (`:v2`, `:v3`, …) and submit that — never
  reuse a tag you've already submitted, so evaluations stay reproducible.
- **Submissions are rate-limited to 10/hour per team.** The registry's
  **pull count** shows when the graders have fetched your image.

---

## Compliance recap (already satisfied by this image)
- `linux/amd64`, publicly pullable, **2.02 GB** (< 10 GB), ready in seconds.
- Reads `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` from env
  only; no keys/URLs/model IDs hardcoded; no `.env` baked in.
- Reads `/input/tasks.json`, writes valid `/output/results.json`, exits 0.

## Troubleshooting
- **`docker buildx` not found** → `docker buildx version`; Docker Desktop ships
  it. If missing: `docker buildx create --use`.
- **`denied: requested access to the resource is denied`** → not logged in, or
  the `-t` namespace doesn't match your username.
- **Build fails at `COPY models/...`** → the GGUF isn't in the build context; run
  step 0.2.
- **Graders can't pull** → the image is still private (step A3 / B3).
- **`no match for platform`** → always build with `--platform linux/amd64`.
