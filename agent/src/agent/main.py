"""Container entrypoint for the Track 1 agent.

Flow: load config from env -> read /input/tasks.json -> run the pipeline ->
write /output/results.json atomically -> log token usage -> exit 0. Any fatal
error is logged and the process exits non-zero, as the rules require.

A **watchdog** guards the whole run. The grader kills the container at 10 minutes
and reports ``TIMEOUT``, which scores zero — even if every answer was already
computed and we simply never got to write the file. That failure is unacceptable
and, worse, not fully preventable from inside the pipeline: a llama.cpp prefill
runs in C and cannot be interrupted from Python, so no amount of deadline
bookkeeping can *guarantee* a generation returns. So we don't rely on the pipeline
finishing. At ``hard_budget`` the watchdog writes whatever answers exist and exits
0 from under the running threads.

A partial results file always beats a missing one: unanswered tasks are simply
graded wrong, whereas ``TIMEOUT`` throws away the answers we did get.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time

from .config import Config, ConfigError
from .fireworks_client import FireworksClient
from .local_model import LocalModel
from .model_selector import ModelSelector
from .pipeline import Pipeline, Task, normalize_tasks

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

    "Any failure" is meant literally, hence the broad except. Loading a GGUF drops
    into C: an ISA the grading CPU doesn't implement raises ``OSError``
    (0xc000001d / SIGILL), and 1.9 GB of weights on a 4 GB box can raise
    ``MemoryError``. Neither is a ``LocalModelError``, so both used to escape this
    function and kill the container with a traceback — exit non-zero,
    ``RUNTIME_ERROR``, scored zero. The local model is an *optimisation*; nothing
    about it is worth failing the run for, since Fireworks alone answers every task.
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
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("Local model failed to load (%r); using Fireworks only.", exc)
        return None
    logger.info("Local model ready: %s", candidate.model_path)
    return candidate


def _start_watchdog(
    config: Config, started_at: float, pipeline: Pipeline, tasks: list[Task]
) -> None:
    """Write results and exit 0 at ``hard_budget``, whatever the pipeline is doing.

    Runs as a daemon thread so it never keeps the process alive on the happy path.
    When it fires it calls ``os._exit``, which is deliberate: a clean shutdown would
    join the worker threads, and a thread stuck inside llama.cpp's C code will not
    join — that is precisely the hang we are escaping. The results file is written
    atomically (``os.replace``) before we exit, so cutting the process down mid-flight
    cannot corrupt or truncate it.
    """
    deadline = started_at + config.hard_budget

    def _fire() -> None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 1.0))

        results = pipeline.results_for(tasks)
        answered = sum(1 for r in results if r["answer"])
        logger.error(
            "Hard time budget (%ds) reached; writing %d/%d answered task(s) and exiting.",
            config.hard_budget,
            answered,
            len(results),
        )
        try:
            _write_results(config.output_path, results)
        except OSError as exc:
            logger.error("Watchdog could not write results to %s: %s", config.output_path, exc)
            sys.stderr.flush()
            os._exit(1)
        sys.stderr.flush()
        os._exit(0)

    threading.Thread(target=_fire, name="watchdog", daemon=True).start()
    logger.info("Watchdog armed: results will be written by T+%ds.", config.hard_budget)


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
    # started=process start: the model load below is part of the container's
    # wall-clock budget, so the deadline has to count it.
    pipeline = Pipeline(config, client, selector, local=None, started_at=start)

    # Armed before the model loads, so that even a stalled load ends in a written
    # results file rather than a TIMEOUT.
    _start_watchdog(config, start, pipeline, tasks)
    pipeline.attach_local(_init_local_model(config))

    try:
        results = pipeline.run(tasks)
    except Exception as exc:  # noqa: BLE001 - never crash before writing output
        logger.exception("Pipeline error: %s", exc)
        # Salvage whatever the pipeline had answered before it broke, rather than
        # discarding it: a wrong answer and a missing one score the same, so
        # anything we already hold is free upside.
        results = pipeline.results_for(tasks)

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
