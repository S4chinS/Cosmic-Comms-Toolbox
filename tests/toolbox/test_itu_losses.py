"""Regression tests for ITU atmospheric loss array shaping."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Repo layout: <repo_root>/src, <repo_root>/tests

from cosmic_toolbox import itu_losses


class _FakeQuantity:
    def __init__(self, value: float):
        self.value = value


def test_estimate_slant_path_loss_returns_1d_total_for_single_elevation(monkeypatch):
    def _fake_attenuation_slant_path(**_kwargs):
        return _FakeQuantity(0.42)

    monkeypatch.setattr(
        itu_losses.itur,
        "atmospheric_attenuation_slant_path",
        _fake_attenuation_slant_path,
    )

    losses = itu_losses.estimate_slant_path_loss(
        frequency_GHz=2.2,
        elevations_deg=[30.0],
        lat_deg=0.0,
        lon_deg=0.0,
        altitude_m=0.0,
    )

    assert isinstance(losses, np.ndarray)
    assert losses.shape == (1,)
    assert np.allclose(losses, [0.42])


def test_estimate_slant_path_loss_returns_1d_contributions_for_single_elevation(monkeypatch):
    def _fake_attenuation_slant_path(**_kwargs):
        return (
            _FakeQuantity(0.10),
            _FakeQuantity(0.20),
            _FakeQuantity(0.30),
            _FakeQuantity(0.40),
            _FakeQuantity(1.00),
        )

    monkeypatch.setattr(
        itu_losses.itur,
        "atmospheric_attenuation_slant_path",
        _fake_attenuation_slant_path,
    )

    losses, contributions = itu_losses.estimate_slant_path_loss(
        frequency_GHz=2.2,
        elevations_deg=[30.0],
        lat_deg=0.0,
        lon_deg=0.0,
        altitude_m=0.0,
        return_contributions=True,
    )

    assert isinstance(losses, np.ndarray)
    assert losses.shape == (1,)
    assert np.allclose(losses, [1.00])
    assert contributions is not None
    assert set(contributions) == {"gaseous", "cloud", "rain", "scintillation", "total"}
    for values in contributions.values():
        assert isinstance(values, np.ndarray)
        assert values.shape == (1,)
