"""Container entrypoint for the Track 1 agent.

Flow: load config from env -> read /input/tasks.json -> run the pipeline ->
write /output/results.json atomically -> log token usage -> exit 0. Any fatal
error is logged and the process exits non-zero, as the rules require.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time

from .config import Config, ConfigError
from .fireworks_client import FireworksClient
from .local_model import LocalModel, LocalModelError
from .model_selector import ModelSelector
from .pipeline import Pipeline, normalize_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("agent")


def _init_local_model(config: Config) -> LocalModel | None:
    """Build and warm the bundled local model, or return None to run
    Fireworks-only. Loading eagerly here (a) surfaces load errors before task
    processing and (b) keeps the first task's latency clean. Any failure —
    disabled, weights absent, or load error — degrades gracefully to the API.
    """
    if not config.local_enabled:
        logger.info("Local model disabled by config; using Fireworks only.")
        return None
    candidate = LocalModel()
    if not candidate.available():
        logger.warning(
            "Local model enabled but weights not found at %s; using Fireworks only.",
            candidate.model_path,
        )
        return None
    try:
        candidate.load()
    except LocalModelError as exc:
        logger.warning("Local model failed to load (%s); using Fireworks only.", exc)
        return None
    logger.info("Local model ready: %s", candidate.model_path)
    return candidate


def _read_tasks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at {path!r}, got {type(data).__name__}."
        )
    return data


def _write_results(path: str, results: list[dict]) -> None:
    """Write results.json atomically so a crash never leaves partial output."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        # Best-effort cleanup of the temp file on failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main() -> int:
    start = time.monotonic()
    try:
        config = Config.from_env()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    selector = ModelSelector(config.allowed_models)
    logger.info("Allowed models: %s", selector.describe())

    try:
        raw_tasks = _read_tasks(config.input_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to read tasks from %s: %s", config.input_path, exc)
        return 1

    tasks = normalize_tasks(raw_tasks)
    logger.info("Loaded %d task(s)", len(tasks))

    client = FireworksClient(config)
    local = _init_local_model(config)
    pipeline = Pipeline(config, client, selector, local=local)

    try:
        results = pipeline.run(tasks)
    except Exception as exc:  # noqa: BLE001 - never crash before writing output
        logger.exception("Pipeline error: %s", exc)
        # Still emit a valid, complete results file with empty answers.
        results = [{"task_id": t.task_id, "answer": ""} for t in tasks]

    try:
        _write_results(config.output_path, results)
    except OSError as exc:
        logger.error("Failed to write results to %s: %s", config.output_path, exc)
        return 1

    usage = client.meter.total
    elapsed = time.monotonic() - start
    logger.info(
        "Done: %d result(s) in %.1fs | tokens prompt=%d completion=%d total=%d",
        len(results),
        elapsed,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
