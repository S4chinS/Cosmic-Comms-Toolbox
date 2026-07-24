"""Unit tests for ITU power flux density helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# Repo layout: <repo_root>/src, <repo_root>/tests

from cosmic_toolbox import pfd_math


def test_occupied_bandwidth_uses_rolloff():
    assert pfd_math.occupied_bandwidth_hz(150e6, 0.25) == 187.5e6


def test_x_band_limit_matches_piecewise_mask():
    elevations = np.array([0.0, 5.0, 10.0, 25.0, 90.0], dtype=float)
    limits = pfd_math.itu_surface_pfd_limit_dBW_per_m2(
        elevations,
        frequency_MHz=8400.0,
    )
    assert np.allclose(limits, np.array([-150.0, -150.0, -147.5, -140.0, -140.0]))


def test_ka_band_limit_matches_piecewise_mask():
    elevations = np.array([0.0, 5.0, 10.0, 25.0, 90.0], dtype=float)
    limits = pfd_math.itu_surface_pfd_limit_dBW_per_m2(
        elevations,
        frequency_MHz=26000.0,
    )
    assert np.allclose(limits, np.array([-115.0, -115.0, -112.5, -105.0, -105.0]))


def test_actual_pfd_matches_manual_equation():
    occupied_bw_hz = pfd_math.occupied_bandwidth_hz(10e6, 0.25)
    directional_eirp = pfd_math.directional_eirp_dBW(
        tx_power_dBw=10.0,
        tx_losses_dB=2.0,
        tx_backoff_dB=0.0,
        antenna_gain_dBi=np.array([35.0], dtype=float),
    )
    actual = pfd_math.pfd_at_4khz_dBW_per_m2(
        directional_eirp_dBW=directional_eirp,
        slant_range_m=np.array([1.0e6], dtype=float),
        occupied_bandwidth_hz=occupied_bw_hz,
    )
    manual = (
        43.0
        - 10.0 * np.log10(4.0 * np.pi * (1.0e6**2))
        - 10.0 * np.log10(occupied_bw_hz / 4000.0)
    )
    assert np.allclose(actual, np.array([manual], dtype=float))


def test_ka_band_reference_bandwidth_is_1mhz():
    spec = pfd_math.itu_surface_pfd_mask_spec(26000.0)
    assert spec.reference_bandwidth_hz == 1_000_000.0


def test_actual_pfd_matches_manual_equation_at_1mhz():
    occupied_bw_hz = pfd_math.occupied_bandwidth_hz(10e6, 0.25)
    directional_eirp = pfd_math.directional_eirp_dBW(
        tx_power_dBw=10.0,
        tx_losses_dB=2.0,
        tx_backoff_dB=0.0,
        antenna_gain_dBi=np.array([35.0], dtype=float),
    )
    actual = pfd_math.pfd_at_reference_bandwidth_dBW_per_m2(
        directional_eirp_dBW=directional_eirp,
        slant_range_m=np.array([1.0e6], dtype=float),
        occupied_bandwidth_hz=occupied_bw_hz,
        reference_bandwidth_hz=1_000_000.0,
    )
    manual = (
        43.0
        - 10.0 * np.log10(4.0 * np.pi * (1.0e6**2))
        - 10.0 * np.log10(occupied_bw_hz / 1_000_000.0)
    )
    assert np.allclose(actual, np.array([manual], dtype=float))


def test_limit_rejects_unsupported_band():
    with pytest.raises(ValueError, match="only implemented"):
        pfd_math.itu_surface_pfd_limit_dBW_per_m2(
            np.array([10.0], dtype=float),
            frequency_MHz=4000.0,
        )
