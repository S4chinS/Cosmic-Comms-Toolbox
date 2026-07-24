from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from cosmic_toolbox import link_budget_math
from cosmic_toolbox.models import GroundStationConfig, PassStatistic
from cosmic_toolbox.services import antenna_pattern
from cosmic_toolbox.services.scenario_data_volume import (
    AccessSeries,
    ScenarioDataVolumeEvaluator,
    compute_pass_downlink_volumes_gbit,
    compute_total_and_per_orbit_gbit,
    filter_passes_by_max_elevation,
)

_LUT_PATH = antenna_pattern.default_synthesized_lut_path()


def _make_access_series() -> AccessSeries:
    time_seconds = np.array([0.0, 10.0, 20.0], dtype=float)
    zeros = np.zeros((3, 3), dtype=float)
    return AccessSeries(
        time_seconds=time_seconds,
        station_series={"Alpha": np.array([5.0, 20.0, 5.0], dtype=float)},
        orbit_period_s=100.0,
        altitude_km=np.array([290.0, 290.0, 290.0], dtype=float),
        sat_ecef_m=zeros,
        body_x_ecef=zeros,
        body_y_ecef=zeros,
        body_z_ecef=zeros,
    )


def test_filter_passes_by_max_elevation_uses_inclusive_bounds() -> None:
    start = datetime(2028, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    passes = [
        PassStatistic(
            index=1,
            aos=start,
            los=start + timedelta(seconds=10),
            duration_minutes=1.0,
            max_elevation_deg=10.0,
            station_name="Alpha",
        ),
        PassStatistic(
            index=2,
            aos=start,
            los=start + timedelta(seconds=10),
            duration_minutes=1.0,
            max_elevation_deg=70.0,
            station_name="Alpha",
        ),
        PassStatistic(
            index=3,
            aos=start,
            los=start + timedelta(seconds=10),
            duration_minutes=1.0,
            max_elevation_deg=71.0,
            station_name="Alpha",
        ),
    ]

    filtered = filter_passes_by_max_elevation(passes, 10.0, 70.0)

    assert [item.index for item in filtered] == [1, 2]


def test_filter_passes_by_max_elevation_skips_non_finite_values() -> None:
    start = datetime(2028, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    passes = [
        PassStatistic(
            index=1,
            aos=start,
            los=start + timedelta(seconds=10),
            duration_minutes=1.0,
            max_elevation_deg=float("nan"),
            station_name="Alpha",
        ),
        PassStatistic(
            index=2,
            aos=start,
            los=start + timedelta(seconds=10),
            duration_minutes=1.0,
            max_elevation_deg=20.0,
            station_name="Alpha",
        ),
    ]

    filtered = filter_passes_by_max_elevation(passes, 10.0, 70.0)

    assert [item.index for item in filtered] == [2]


def test_compute_pass_downlink_volumes_gbit_uses_station_specific_series() -> None:
    access_series = _make_access_series()
    start = datetime(2028, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    passes = [
        PassStatistic(
            index=1,
            aos=start,
            los=start + timedelta(seconds=20),
            duration_minutes=20.0 / 60.0,
            max_elevation_deg=20.0,
            station_name="Alpha",
        )
    ]
    station_rates = {
        "Alpha": np.array([100.0, 100.0, 100.0], dtype=float),
    }

    volumes = compute_pass_downlink_volumes_gbit(
        access_series=access_series,
        station_rate_lookup=station_rates,
        passes=passes,
        scenario_start_time=start,
    )

    assert len(volumes) == 1
    assert volumes[0] == 2.0


def test_compute_total_and_per_orbit_gbit_matches_ui_aggregation() -> None:
    access_series = _make_access_series()
    pass_volumes_gbit = [2.0, 1.0]

    total_gbit, per_orbit_gbit = compute_total_and_per_orbit_gbit(
        access_series=access_series,
        pass_volumes_gbit=pass_volumes_gbit,
    )

    expected_total = sum(pass_volumes_gbit)
    expected_per_orbit = expected_total / ((20.0 - 0.0) / 100.0)

    assert total_gbit == expected_total
    assert per_orbit_gbit == expected_per_orbit


def _make_geometric_access_series(elevations: np.ndarray) -> AccessSeries:
    """Synthetic scenario with a real satellite geometry over a station region."""
    n = elevations.size
    time_seconds = np.arange(n, dtype=float) * 10.0
    # Satellite on a short ECEF arc at ~400 km altitude over the equator.
    theta = np.linspace(-0.05, 0.05, n)
    radius_m = (6378.137 + 400.0) * 1000.0
    sat_ecef_m = np.column_stack(
        [
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            np.zeros(n),
        ]
    )
    # Orthonormal body axes (identity) repeated for every sample.
    body_x = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    body_y = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    body_z = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    return AccessSeries(
        time_seconds=time_seconds,
        station_series={"Alpha": elevations.astype(float)},
        orbit_period_s=5400.0,
        altitude_km=np.full(n, 400.0, dtype=float),
        sat_ecef_m=sat_ecef_m,
        body_x_ecef=body_x,
        body_y_ecef=body_y,
        body_z_ecef=body_z,
    )


def test_in_view_masking_matches_full_timeline_evaluation() -> None:
    """The in-view optimisation must be numerically identical to evaluating the
    full timeline and zeroing below-horizon samples afterwards."""
    pytest.importorskip("itur")
    if not _LUT_PATH.exists():
        pytest.skip("Antenna LUT resource not available")

    elevations = np.array(
        [-10.0, -1.0, 0.0, 2.0, 15.0, 60.0, 30.0, 5.0, -3.0, -20.0], dtype=float
    )
    access = _make_geometric_access_series(elevations)
    station = GroundStationConfig(
        name="Alpha", latitude_deg=0.0, longitude_deg=0.0, altitude_m=0.0
    )
    contact_elevation_deg = 10.0

    evaluator = ScenarioDataVolumeEvaluator(
        access_series=access,
        station_lookup={"Alpha": station},
        scenario_start_time=datetime(2030, 1, 1, tzinfo=timezone.utc),
        comms_pointing_mode="prograde_pointing",
        comms_pointing_aoa_limit_deg=5.0,
        contact_elevation_deg=contact_elevation_deg,
    )

    # Reference: full-timeline gain + budget, then zero below-horizon (old behaviour).
    lut = antenna_pattern.load_spherical_gain_lut(_LUT_PATH.expanduser().resolve())
    full_gains, _az, _el, _roll = antenna_pattern.evaluate_station_gain_series(
        lut=lut,
        station=station,
        sat_ecef_m=access.sat_ecef_m,
        body_x_ecef=access.body_x_ecef,
        body_y_ecef=access.body_y_ecef,
        body_z_ecef=access.body_z_ecef,
        pointing_mode="prograde_pointing",
        max_aoa_deg=5.0,
        steering_active_mask=elevations >= contact_elevation_deg,
    )
    sanitized = np.clip(elevations, 0.0, 90.0)
    loss_grid, loss_values = evaluator._get_timeseries_atmospheric_loss_curve(
        station=station,
        frequency_GHz=8.2,
        unavailability_percent=0.1,
        min_elev_deg=0.1,
        max_elev_deg=90.0,
    )
    full_losses = np.interp(
        np.maximum(sanitized, float(loss_grid[0])),
        loss_grid,
        loss_values,
        left=float(loss_values[0]),
        right=float(loss_values[-1]),
    )
    ref = link_budget_math.calculate_link_budget(
        elevations_deg=sanitized,
        antenna_gains_dBi=full_gains,
        atmospheric_losses_dB=full_losses,
        tx_power_dBw=3.0,
        tx_boresight_gain_dBi=float(np.max(lut.gain_dbi_grid)),
        tx_losses_dB=2.0,
        tx_backoff_dB=0.0,
        frequency_GHz=8.2,
        satellite_altitude_km=access.altitude_km,
        ground_altitude_m=0.0,
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        receiver_losses_dB=0.0,
        polarization_loss_dB=0.1,
        symbol_rate_sps=10e6,
        implementation_loss_dB=1.0,
        margin_dB=0.0,
        fixed_modcod_name=None,
    )
    ref_rates = np.asarray(ref["data_rate_mbps"], dtype=float)
    ref_rates[elevations <= 0.0] = 0.0

    # New path: gain series is restricted to in-view samples internally.
    gains = evaluator._get_station_antenna_gain_series(
        station=station, antenna_lut_path=_LUT_PATH
    )
    in_view = elevations > 0.0
    assert np.allclose(gains[in_view], full_gains[in_view])
    assert np.all(gains[~in_view] == 0.0)

    new_rates = evaluator._evaluate_station_data_rate(
        station=station,
        elevations_deg=elevations,
        altitude_km=access.altitude_km,
        antenna_gains_dBi=gains,
        frequency_GHz=8.2,
        symbol_rate_sps=10e6,
        tx_power_dBw=3.0,
        tx_gain_dBi=float(np.max(lut.gain_dbi_grid)),
        tx_losses_dB=2.0,
        tx_backoff_dB=0.0,
        rx_antenna_gain_dBi=34.0,
        receiver_noise_figure_dB=1.5,
        sky_background_temperature_K=50.0,
        rx_losses_dB=0.0,
        polarization_loss_dB=0.1,
        implementation_loss_dB=1.0,
        margin_dB=0.0,
        unavailability_percent=0.1,
        fixed_modcod_name=None,
    )

    assert new_rates.shape == elevations.shape
    assert np.allclose(new_rates, ref_rates)
