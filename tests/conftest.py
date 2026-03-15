"""Shared test fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test outputs (logs, results, etc.)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
