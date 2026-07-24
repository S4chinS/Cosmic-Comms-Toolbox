"""Skip the GUI test suite when PySide6 isn't installed (headless installs)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed; skipping GUI tests")
