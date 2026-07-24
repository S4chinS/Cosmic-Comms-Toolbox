"""Unit tests for free-to-roll visualization attitude helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Repo layout: <repo_root>/src, <repo_root>/tests

from cosmic_toolbox_gui.tabs.visualization_tab import VisualizationTabMixin


def test_free_to_roll_preserves_body_x_axis():
    base_axes = np.identity(3, dtype=float)
    los = np.array([0.2, 0.5, 0.7], dtype=float)

    rolled_axes = VisualizationTabMixin._free_to_roll_axes_from_base_and_los(
        base_axes,
        los,
    )

    assert np.allclose(rolled_axes[:, 0], base_axes[:, 0])


def test_free_to_roll_places_los_in_body_xz_plane():
    base_axes = np.identity(3, dtype=float)
    los = np.array([0.2, 0.5, 0.7], dtype=float)

    rolled_axes = VisualizationTabMixin._free_to_roll_axes_from_base_and_los(
        base_axes,
        los,
    )
    los_hat = los / np.linalg.norm(los)
    los_body = np.array(
        [
            np.dot(los_hat, rolled_axes[:, 0]),
            np.dot(los_hat, rolled_axes[:, 1]),
            np.dot(los_hat, rolled_axes[:, 2]),
        ],
        dtype=float,
    )

    assert abs(float(los_body[1])) < 1e-6
    assert float(los_body[2]) >= 0.0
