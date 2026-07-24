"""Unit tests for the lightweight link-budget math helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Repo layout: <repo_root>/src, <repo_root>/tests

from cosmic_toolbox import link_budget_math


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


def test_calculate_link_budget_supports_qpsk_viterbi_fixed_mode():
    elevations = np.array([30.0], dtype=float)
    result = link_budget_math.calculate_link_budget(
        elevations_deg=elevations,
        antenna_gains_dBi=np.array([8.0], dtype=float),
        atmospheric_losses_dB=np.array([0.0], dtype=float),
        tx_power_dBw=20.0,
        tx_boresight_gain_dBi=8.0,
        tx_losses_dB=0.0,
        tx_backoff_dB=0.0,
        frequency_GHz=2.2,
        satellite_altitude_km=550.0,
        ground_altitude_m=0.0,
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        receiver_losses_dB=0.0,
        polarization_loss_dB=0.0,
        symbol_rate_sps=1e6,
        implementation_loss_dB=0.0,
        margin_dB=0.0,
        fixed_modcod_name=link_budget_math.QPSK_VITERBI_MODE_NAME,
    )

    assert result["modcod_names"] == [link_budget_math.QPSK_VITERBI_MODE_NAME]
    assert np.isclose(
        float(result["spectral_efficiency_bits_per_symbol"][0]),
        link_budget_math.QPSK_VITERBI_BITS_PER_SYMBOL,
    )
    assert np.isclose(
        float(result["required_EsN0_dB"][0]),
        link_budget_math.QPSK_VITERBI_REQUIRED_ESN0_DB,
    )


def test_build_parameter_rows_formats_qpsk_viterbi_display_fields():
    elevations = np.array([30.0], dtype=float)
    results = link_budget_math.calculate_link_budget(
        elevations_deg=elevations,
        antenna_gains_dBi=np.array([8.0], dtype=float),
        atmospheric_losses_dB=np.array([0.0], dtype=float),
        tx_power_dBw=20.0,
        tx_boresight_gain_dBi=8.0,
        tx_losses_dB=0.0,
        tx_backoff_dB=0.0,
        frequency_GHz=2.2,
        satellite_altitude_km=550.0,
        ground_altitude_m=0.0,
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        receiver_losses_dB=0.0,
        polarization_loss_dB=0.0,
        symbol_rate_sps=1e6,
        implementation_loss_dB=0.0,
        margin_dB=0.0,
        fixed_modcod_name=link_budget_math.QPSK_VITERBI_MODE_NAME,
    )
    rows = link_budget_math.build_parameter_rows(
        elevations_deg=elevations,
        results=results,
        evaluation_elevation_deg=30.0,
        elevation_lower_bound_deg=30.0,
        elevation_upper_bound_deg=30.0,
        tx_frequency_GHz=2.2,
        tx_power_dBw=20.0,
        tx_losses_dB=0.0,
        tx_boresight_gain_dBi=8.0,
        tx_backoff_dB=0.0,
        symbol_rate_sps=1e6,
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        implementation_loss_dB=0.0,
        margin_dB=0.0,
        rolloff=0.25,
        polarization_loss_dB=0.0,
        satellite_altitude_km=550.0,
    )
    row_values = {row.parameter: row.value for row in rows}

    assert row_values["Modulation Type"] == "QPSK"
    assert row_values["Modulation Order"] == "4"
    assert row_values["Coding Rate"] == "0.437"


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
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        receiver_losses_dB=0.5,
        polarization_loss_dB=0.1,
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


def test_calculate_link_budget_applies_polarization_loss():
    elevations = np.array([25.0], dtype=float)
    antenna_gain = np.array([10.0], dtype=float)
    atmos_loss = np.array([0.0], dtype=float)

    no_polarization_loss = link_budget_math.calculate_link_budget(
        elevations_deg=elevations,
        antenna_gains_dBi=antenna_gain,
        atmospheric_losses_dB=atmos_loss,
        tx_power_dBw=8.0,
        tx_boresight_gain_dBi=10.0,
        tx_losses_dB=1.0,
        tx_backoff_dB=0.0,
        frequency_GHz=8.2,
        satellite_altitude_km=550.0,
        ground_altitude_m=0.0,
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        receiver_losses_dB=0.0,
        polarization_loss_dB=0.0,
        symbol_rate_sps=150e6,
        implementation_loss_dB=1.0,
        margin_dB=2.0,
    )
    with_polarization_loss = link_budget_math.calculate_link_budget(
        elevations_deg=elevations,
        antenna_gains_dBi=antenna_gain,
        atmospheric_losses_dB=atmos_loss,
        tx_power_dBw=8.0,
        tx_boresight_gain_dBi=10.0,
        tx_losses_dB=1.0,
        tx_backoff_dB=0.0,
        frequency_GHz=8.2,
        satellite_altitude_km=550.0,
        ground_altitude_m=0.0,
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        receiver_losses_dB=0.0,
        polarization_loss_dB=0.5,
        symbol_rate_sps=150e6,
        implementation_loss_dB=1.0,
        margin_dB=2.0,
    )

    no_pol_rx = float(no_polarization_loss["received_carrier_power_dBw"][0])
    with_pol_rx = float(with_polarization_loss["received_carrier_power_dBw"][0])
    no_pol_cn0 = float(no_polarization_loss["c_to_n0_dBHz"][0])
    with_pol_cn0 = float(with_polarization_loss["c_to_n0_dBHz"][0])

    assert np.isclose(no_pol_rx - with_pol_rx, 0.5)
    assert np.isclose(no_pol_cn0 - with_pol_cn0, 0.5)


def test_optimize_ccm_per_pass_returns_no_link_when_infeasible():
    time = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    es_n0 = np.array([-20.0, -20.0, -20.0, -20.0], dtype=float)

    result = link_budget_math.optimize_ccm_per_pass(
        time_seconds=time,
        es_n0_dB=es_n0,
        margin_dB=3.0,
        symbol_rate_sps=150e6,
        offset_step_s=10.0,
        max_offset_s=30.0,
    )

    assert result.modcod_name == "No Link"
    assert result.volume_gb == 0.0
    assert np.all(result.rate_mbps == 0.0)


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
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        receiver_losses_dB=0.5,
        polarization_loss_dB=0.1,
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
        elevation_upper_bound_deg=float(np.max(elevations)),
        tx_frequency_GHz=8.2,
        tx_power_dBw=8.0,
        tx_losses_dB=1.0,
        tx_boresight_gain_dBi=10.0,
        tx_backoff_dB=0.5,
        symbol_rate_sps=150e6,
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        implementation_loss_dB=1.0,
        margin_dB=2.0,
        rolloff=0.25,
        polarization_loss_dB=0.1,
        satellite_altitude_km=550.0,
        atmospheric_breakdown_dB=contributions,
    )

    expected_prefix = [
        "Earth Radius",
        "Spacecraft Altitude",
        "Elevation Angle",
        "Elevation Lower Bound",
        "Elevation Upper Bound",
    ]
    assert [row.parameter for row in rows[:5]] == expected_prefix
    assert rows[-1].parameter == "PFD Margin"
