"""Unit tests for the lightweight link-budget math helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import link_budget_math  # type: ignore[import-not-found]


def test_slant_range_decreases_with_elevation():
    elevations = np.array([5.0, 30.0, 60.0, 85.0])
    ranges = link_budget_math.slant_range_km(
        elevations, ground_altitude_km=0.1, satellite_altitude_km=550.0
    )
    assert np.all(np.diff(ranges) < 0)  # higher elevation → shorter range


def test_select_modcod_respects_margin():
    esn0 = [12.0]
    no_margin = link_budget_math.select_modcod(esn0, margin_dB=0.0)[0]
    large_margin = link_budget_math.select_modcod(esn0, margin_dB=20.0)[0]
    assert no_margin  # should select a MODCOD
    assert large_margin == {}  # link should fail when margin is excessive


def test_calculate_link_budget_returns_data_rates():
    elevations = np.linspace(10.0, 80.0, 5)
    antenna_gain = np.full_like(elevations, 8.0)
    atmos_loss = np.zeros_like(elevations)

    result = link_budget_math.calculate_link_budget(
        elevations_deg=elevations,
        antenna_gains_dBi=antenna_gain,
        atmospheric_losses_dB=atmos_loss,
        tx_power_dBw=8.0,
        tx_boresight_gain_dBi=11.0,
        tx_losses_dB=1.0,
        tx_backoff_dB=0.5,
        frequency_GHz=8.2,
        satellite_altitude_km=550.0,
        ground_altitude_m=200.0,
        receiver_G_T_dB_K=28.0,
        receiver_losses_dB=0.5,
        symbol_rate_sps=150e6,
        implementation_loss_dB=1.0,
        margin_dB=2.0,
    )

    assert "data_rate_mbps" in result
    assert np.any(result["data_rate_mbps"] > 0.0)
    assert len(result["modcod_names"]) == len(elevations)


def test_integrate_data_volume_trapz():
    time = np.array([0.0, 10.0, 20.0])
    data_rate = np.array([100.0, 200.0, 0.0])  # Mbps
    volume = link_budget_math.integrate_data_volume_gb(time, data_rate)
    assert volume == np.trapezoid(data_rate, time) / 1000.0


def test_build_parameter_rows_captures_inputs_and_calculations():
    elevations = np.linspace(5.0, 50.0, 5)
    antenna_gain = np.full_like(elevations, 8.0)
    atmos_loss = np.full_like(elevations, 0.5)

    results = link_budget_math.calculate_link_budget(
        elevations_deg=elevations,
        antenna_gains_dBi=antenna_gain,
        atmospheric_losses_dB=atmos_loss,
        tx_power_dBw=8.0,
        tx_boresight_gain_dBi=10.0,
        tx_losses_dB=1.0,
        tx_backoff_dB=0.5,
        frequency_GHz=8.2,
        satellite_altitude_km=550.0,
        ground_altitude_m=200.0,
        receiver_G_T_dB_K=28.0,
        receiver_losses_dB=0.5,
        symbol_rate_sps=150e6,
        implementation_loss_dB=1.0,
        margin_dB=2.0,
    )

    contributions = {
        "gaseous": np.full_like(elevations, 0.2),
        "cloud": np.full_like(elevations, 0.1),
        "rain": np.full_like(elevations, 0.05),
        "scintillation": np.full_like(elevations, 0.02),
        "total": atmos_loss,
    }

    rows = link_budget_math.build_parameter_rows(
        elevations_deg=elevations,
        results=results,
        evaluation_elevation_deg=25.0,
        elevation_lower_bound_deg=5.0,
        elevation_upper_bound_deg=90.0,
        tx_frequency_GHz=8.2,
        tx_power_dBw=8.0,
        tx_losses_dB=1.0,
        tx_boresight_gain_dBi=10.0,
        tx_backoff_dB=0.5,
        symbol_rate_sps=150e6,
        receiver_G_T_dB_K=28.0,
        implementation_loss_dB=1.0,
        margin_dB=2.0,
        rolloff=0.25,
        polarization_loss_dB=0.1,
        satellite_altitude_km=550.0,
        atmospheric_breakdown_dB=contributions,
    )

    expected_prefix = [
        "TX Frequency",
        "TX Power",
        "TX Losses",
        "TX Antenna Boresight Gain",
        "EIRP",
    ]
    assert [row.parameter for row in rows[:5]] == expected_prefix
    assert rows[-1].parameter == "Surplus Link Margin"
