"""Regression coverage for cached access recompute."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
import types

import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
import pytest

sys.modules.setdefault("orekitdata", types.SimpleNamespace())
sys.modules.setdefault(
    "cosmic_toolbox.services.access_analysis",
    types.SimpleNamespace(run_access_analysis=lambda *_args, **_kwargs: None),
)

from cosmic_toolbox.models import (
    AnalysisConfig,
    AnalysisOptions,
    GroundStationConfig,
    OrbitConfig,
    PropagatedEphemeris,
    PropagationConfig,
    ScenarioConfig,
)
from cosmic_toolbox.services.cached_access_recompute import derive_access_results_from_ephemeris
from cosmic_toolbox_gui.main_window import GroundStationApp


@pytest.fixture(scope="session", autouse=True)
def _qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_config(
    *,
    comms_pointing_mode: str = "prograde_pointing",
    comms_pointing_aoa_limit_deg: float = 5.0,
    contact_elevation_deg: float = 10.0,
) -> AnalysisConfig:
    start = datetime(2026, 3, 9, tzinfo=timezone.utc)
    end = start + timedelta(minutes=4)
    return AnalysisConfig(
        ground_station=None,
        orbit=OrbitConfig(
            semi_major_axis_km=6878.137,
            eccentricity=0.0,
            inclination_deg=97.0,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
        ),
        propagation=PropagationConfig(
            propagator_type="brouwer_lyddane",
            min_elevation_deg=0.0,
            sample_step_seconds=60.0,
            contact_elevation_deg=contact_elevation_deg,
            enable_contact_attitude_switching=False,
            comms_pointing_mode=comms_pointing_mode,
            comms_pointing_aoa_limit_deg=comms_pointing_aoa_limit_deg,
        ),
        scenario=ScenarioConfig(start_time=start, end_time=end),
        options=AnalysisOptions(compute_ground_station_passes=True),
    )


def _make_ephemeris() -> PropagatedEphemeris:
    start = datetime(2026, 3, 9, tzinfo=timezone.utc)
    earth_radius_m = 6_378_137.0
    orbit_radius_m = earth_radius_m + 500_000.0
    angles_deg = [-40.0, -10.0, 0.0, 10.0, 40.0]
    timeline_seconds = np.array([0.0, 60.0, 120.0, 180.0, 240.0], dtype=np.float64)
    speed_mps = 7_500.0
    n = len(angles_deg)

    timestamps_unix = np.array(
        [(start + timedelta(seconds=s)).timestamp() for s in timeline_seconds],
        dtype=np.float64,
    )
    thetas = np.radians(np.array(angles_deg, dtype=np.float64))
    xs_m = orbit_radius_m * np.cos(thetas)
    ys_m = orbit_radius_m * np.sin(thetas)
    vxs_mps = -speed_mps * np.sin(thetas)
    vys_mps = speed_mps * np.cos(thetas)
    zeros = np.zeros(n, dtype=np.float64)

    eci_pos_km = np.column_stack([xs_m / 1000.0, ys_m / 1000.0, zeros])
    eci_vel_km_s = np.column_stack([vxs_mps / 1000.0, vys_mps / 1000.0, zeros])
    ecef_pos_km = eci_pos_km.copy()
    ecef_vel_km_s = eci_vel_km_s.copy()
    body_x_ecef = np.tile([1.0, 0.0, 0.0], (n, 1))
    body_y_ecef = np.tile([0.0, 1.0, 0.0], (n, 1))
    body_z_ecef = np.tile([0.0, 0.0, 1.0], (n, 1))
    gt_lat_deg = np.zeros(n, dtype=np.float64)
    gt_lon_deg = np.array(angles_deg, dtype=np.float64)

    return PropagatedEphemeris(
        timeline_seconds=timeline_seconds,
        orbit_period_seconds=5400.0,
        orbital_altitude_km=np.full(n, 500.0, dtype=np.float64),
        timestamps_unix=timestamps_unix,
        eci_pos_km=eci_pos_km,
        eci_vel_km_s=eci_vel_km_s,
        ecef_pos_km=ecef_pos_km,
        ecef_vel_km_s=ecef_vel_km_s,
        body_x_ecef=body_x_ecef,
        body_y_ecef=body_y_ecef,
        body_z_ecef=body_z_ecef,
        gt_lat_deg=gt_lat_deg,
        gt_lon_deg=gt_lon_deg,
    )


def test_derive_access_from_ephemeris_detects_single_pass():
    config = _make_config()
    ephemeris = _make_ephemeris()
    station = GroundStationConfig("Equator", 0.0, 0.0, 0.0)

    result = derive_access_results_from_ephemeris(
        ephemeris=ephemeris,
        config=config,
        stations=[station],
    )

    assert list(result.station_elevation_series) == ["Equator"]
    assert len(result.passes) == 1
    assert result.passes[0].station_name == "Equator"
    assert result.summary.total_passes == 1
    elevations = result.station_elevation_series["Equator"]
    assert elevations[0] < 0.0
    assert elevations[2] > 80.0
    assert elevations[-1] < 0.0


def test_station_swap_reuses_same_ephemeris():
    config = _make_config()
    ephemeris = _make_ephemeris()
    station_a = GroundStationConfig("A", 0.0, 0.0, 0.0)
    station_b = GroundStationConfig("B", 0.0, 25.0, 0.0)

    result_a = derive_access_results_from_ephemeris(
        ephemeris=ephemeris,
        config=config,
        stations=[station_a],
    )
    result_b = derive_access_results_from_ephemeris(
        ephemeris=ephemeris,
        config=config,
        stations=[station_b],
    )

    assert np.array_equal(ephemeris.timeline_seconds, [0.0, 60.0, 120.0, 180.0, 240.0])
    assert list(result_a.station_elevation_series) == ["A"]
    assert list(result_b.station_elevation_series) == ["B"]
    assert not np.array_equal(
        result_a.station_elevation_series["A"], result_b.station_elevation_series["B"]
    )


def test_ephemeris_signature_ignores_pointing_changes_and_recompute_stays_enabled():
    window = GroundStationApp.__new__(GroundStationApp)
    base = _make_config(
        comms_pointing_mode="prograde_pointing",
        comms_pointing_aoa_limit_deg=5.0,
        contact_elevation_deg=10.0,
    )
    pointing_changed = _make_config(
        comms_pointing_mode="constrained_aoa",
        comms_pointing_aoa_limit_deg=20.0,
        contact_elevation_deg=25.0,
    )

    assert window._ephemeris_signature_for_config(base) == window._ephemeris_signature_for_config(
        pointing_changed
    )
    assert window._is_cached_recompute_supported_for_config(base) is True


def test_cached_access_ui_state_transitions():
    window = GroundStationApp.__new__(GroundStationApp)
    window.analysis_cache_status_label = QLabel()
    window.ground_station_recompute_hint_label = QLabel()
    window.refresh_cached_access_button = QPushButton()
    window.export_cached_package_button = QPushButton()
    window.import_cached_package_button = QPushButton()
    window._cached_recompute_message = "Derived outputs stale."

    window._last_ephemeris = object()
    window._is_dirty = False
    window._derived_outputs_stale = True
    window._update_cached_access_ui_state()

    assert window.refresh_cached_access_button.isEnabled() is True
    assert "Derived outputs stale." in window.analysis_cache_status_label.text()

    window._derived_outputs_stale = False
    window._update_cached_access_ui_state()
    assert "Ephemeris cached" in window.analysis_cache_status_label.text()
    assert window.refresh_cached_access_button.isEnabled() is False

    window._is_dirty = True
    window._update_cached_access_ui_state()
    assert "Full rerun required" in window.analysis_cache_status_label.text()
    assert window.refresh_cached_access_button.isEnabled() is False
