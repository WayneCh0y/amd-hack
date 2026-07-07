"""Track 1 general-purpose AI agent for the AMD Developer Hackathon.

A batch pipeline that reads tasks from ``/input/tasks.json``, solves each prompt
via Fireworks AI, and writes ``/output/results.json``. See ``main.py`` for the
entrypoint and ``pipeline.py`` for orchestration.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
