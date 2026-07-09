"""Tests for the local-model wrapper that don't require the weights file.

These cover the cheap-to-verify contract: availability detection and a clear
error when the GGUF is missing. Actual inference is exercised by
scripts/local_smoke.py and scripts/benchmark.py --local (they need the ~2 GB
weights, so they live outside the unit-test suite).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.local_model import LocalModel, LocalModelError  # noqa: E402


def test_available_false_for_missing_file():
    model = LocalModel(model_path="/definitely/not/here.gguf")
    assert model.available() is False


def test_load_raises_clear_error_when_weights_missing():
    model = LocalModel(model_path="/definitely/not/here.gguf")
    with pytest.raises(LocalModelError):
        model.load()


def test_model_path_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL_PATH", "/some/env/path.gguf")
    assert LocalModel().model_path == "/some/env/path.gguf"
