# Track 1 agent image. Judging VM runs linux/amd64 — build with:
#   docker buildx build --platform linux/amd64 -t <registry>/amd-track1:latest --push .
#
# Two stages:
#   1) builder — compiles llama-cpp-python from source with an AVX2 baseline so
#      the bundled local model runs on any modern x86-64 server (AMD/Intel)
#      WITHOUT requiring AVX-512. GGML_NATIVE=OFF keeps the build-host's exact
#      ISA out of the binary (portability > a few % speed).
#   2) runtime — a slim image with just the built wheel + openai + agent source.
#      No compiler toolchain ships in the final image, so it stays small and
#      ready quickly.

# ---- Stage 1: build the llama-cpp-python wheel (AVX2, no AVX-512) ------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# Portable CPU baseline. AVX2/FMA/F16C are present on all current server CPUs
# (AMD EPYC/Ryzen included); AVX-512 is NOT universal, so it stays off.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF"

# --no-binary forces a source build with the flags above; otherwise pip would
# grab the prebuilt manylinux wheel (which may assume AVX-512 and SIGILL).
RUN pip wheel --no-cache-dir --no-binary llama-cpp-python \
        --wheel-dir /wheels "llama-cpp-python==0.3.33"

# ---- Stage 2: slim runtime --------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime dep of the compiled llama.cpp: it's built with OpenMP (-fopenmp), so
# the final image needs libgomp even though the compiler toolchain does not ship.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the locally built llama-cpp-python wheel (+ its deps) and the API client.
COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-cache-dir /wheels/*.whl \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /wheels

# Application source only (see .dockerignore for what is excluded).
COPY src/ ./src/

# ---- Bundle the local model weights -----------------------------------------
# Local-first answering is wired in (see architecture-1.md): the agent answers
# with this model at zero Fireworks-token cost and escalates only on failure.
# The GGUF is git-ignored (GBs) but must be present in the build context.
COPY models/Qwen2.5-3B-Instruct-Q4_K_M.gguf /models/model.gguf

ENV PYTHONPATH=/app/src \
    LOCAL_MODEL_PATH=/models/model.gguf \
    LOCAL_N_THREADS=2

# Read tasks from /input, write results to /output (created if absent at runtime).
ENTRYPOINT ["python", "-m", "agent.main"]
