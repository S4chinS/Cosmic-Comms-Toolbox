"""Headless coverage for the shared defaults / helpers unified across the repo.

Exercises the single-source-of-truth link-budget defaults, the promoted
orbit/pass-geometry helpers, the station-by-name loader, the widened facade,
and (importantly) the cached-access fast path — which previously had no
headless test because its only coverage lived in the PySide6-gated GUI suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from cosmic_toolbox import ToolboxFacade, orbit_utils
from cosmic_toolbox.link_budget_defaults import load_link_budget_defaults
from cosmic_toolbox.models import (
    AnalysisConfig,
    AnalysisOptions,
    GroundStationConfig,
    OrbitConfig,
    PropagatedEphemeris,
    PropagationConfig,
    ScenarioConfig,
)
from cosmic_toolbox.services.cached_access_recompute import (
    derive_access_results_from_ephemeris,
)
from cosmic_toolbox.services.pass_geometry import (
    elevation_deg_geocentric,
    interpolate_zero_crossing_time,
    resolve_positive_rate_bounds,
)
from cosmic_toolbox.services.station_importer import (
    StationImportError,
    load_station_by_name,
)


# ── Canonical link-budget defaults ──────────────────────────────────────────
def test_link_budget_defaults_values_and_cache():
    d = load_link_budget_defaults()
    # Consensus values chosen when the 26 scattered sources were unified.
    assert d.frequency_GHz == 2.212
    assert d.tx_power_dBw == 0.0
    assert d.tx_losses_dB == 2.0
    assert d.gs_gt_dBK == 12.0
    assert d.symbol_rate_Msps == 2.0
    assert d.margin_dB == 6.0
    assert d.unavailability_percent == 0.1
    assert d.rolloff == 0.25
    assert d.fixed_modcod_name == "QPSK 1/2"
    # Derived properties.
    assert d.symbol_rate_sps == 2_000_000.0
    assert d.frequency_MHz == pytest.approx(2212.0)
    # Cached: same object back for the default path.
    assert load_link_budget_defaults() is d
    # Facade exposes the identical loader.
    assert ToolboxFacade.link_budget_defaults() is d


# ── Orbit utils ─────────────────────────────────────────────────────────────
def test_sso_inclination_known_value():
    # 600 km circular SSO is ~97.79 deg (retrograde).
    inc = orbit_utils.sso_inclination_deg(altitude_km=600.0, eccentricity=0.0)
    assert inc == pytest.approx(97.79, abs=0.02)


def test_sso_inclination_km_and_m_consistent():
    # The solve must be scale-consistent: a 550 km orbit resolves regardless.
    inc = orbit_utils.sso_inclination_deg(altitude_km=550.0)
    assert 96.0 < inc < 98.5


@pytest.mark.parametrize(
    "alt, ecc",
    [(-10.0, 0.0), (600.0, 1.5), (600.0, -0.1), (float("nan"), 0.0)],
)
def test_sso_inclination_returns_none_for_invalid(alt, ecc):
    assert orbit_utils.sso_inclination_deg(altitude_km=alt, eccentricity=ecc) is None


def test_orbital_period_leo_range():
    # ~500 km LEO period is ~5670 s.
    period = orbit_utils.orbital_period_s(altitude_km=500.0)
    assert 5600.0 < period < 5750.0


# ── Pass geometry ───────────────────────────────────────────────────────────
def test_elevation_overhead_and_below_horizon():
    stn = np.array([6_378_137.0, 0.0, 0.0])
    overhead = np.array([[7_000_000.0, 0.0, 0.0]])
    assert elevation_deg_geocentric(overhead, stn)[0] == pytest.approx(90.0)
    behind = np.array([[-7_000_000.0, 0.0, 0.0]])
    assert elevation_deg_geocentric(behind, stn)[0] < 0.0


def test_elevation_scale_invariant_km_vs_m():
    stn_m = np.array([6_378_137.0, 0.0, 0.0])
    sat_m = np.array([[6_378_137.0 + 500_000.0, 300_000.0, 0.0]])
    el_m = elevation_deg_geocentric(sat_m, stn_m)
    el_km = elevation_deg_geocentric(sat_m / 1000.0, stn_m / 1000.0)
    assert el_m == pytest.approx(el_km)


def test_interpolate_zero_crossing_midpoint():
    t = interpolate_zero_crossing_time(t0=0.0, y0=-1.0, t1=2.0, y1=1.0)
    assert t == pytest.approx(1.0)


def test_resolve_positive_rate_bounds_trims_to_positive_span():
    time_axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    rates = np.array([-1.0, 1.0, 2.0, 1.0, -1.0])
    bounds = resolve_positive_rate_bounds(
        time_axis=time_axis, rates=rates, start_sec=0.0, end_sec=4.0
    )
    assert bounds is not None
    lo, hi = bounds
    assert 0.0 < lo < 1.0
    assert 3.0 < hi < 4.0


# ── Station-by-name loader (single source of truth for coordinates) ─────────
def test_load_station_known_and_unknown():
    stn = load_station_by_name("Longyearbyen, NO")
    assert stn.latitude_deg == pytest.approx(78.2315)
    assert stn.longitude_deg == pytest.approx(15.4111)
    assert stn.altitude_m == pytest.approx(484.0)
    # Distinct site — must not collide with Longyearbyen.
    ny = load_station_by_name("Ny-Alesund, NO")
    assert ny.latitude_deg == pytest.approx(78.9231)
    with pytest.raises(StationImportError):
        load_station_by_name("Nowhere, XX")


def test_facade_load_station_matches_importer():
    assert (
        ToolboxFacade.load_station("Stockholm, SE").latitude_deg
        == load_station_by_name("Stockholm, SE").latitude_deg
    )


def test_facade_default_antenna_lut_loads():
    lut = ToolboxFacade.load_antenna_lut()
    assert lut.gain_dbi_grid.ndim == 2


# ── Cached access fast path (headless — this is the regression the audit
#    flagged as untested outside the GUI suite) ───────────────────────────────
def _make_config() -> AnalysisConfig:
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
            contact_elevation_deg=10.0,
            enable_contact_attitude_switching=False,
            comms_pointing_mode="prograde_pointing",
            comms_pointing_aoa_limit_deg=5.0,
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
    return PropagatedEphemeris(
        timeline_seconds=timeline_seconds,
        orbit_period_seconds=5400.0,
        orbital_altitude_km=np.full(n, 500.0, dtype=np.float64),
        timestamps_unix=timestamps_unix,
        eci_pos_km=eci_pos_km,
        eci_vel_km_s=eci_vel_km_s,
        ecef_pos_km=eci_pos_km.copy(),
        ecef_vel_km_s=eci_vel_km_s.copy(),
        body_x_ecef=np.tile([1.0, 0.0, 0.0], (n, 1)),
        body_y_ecef=np.tile([0.0, 1.0, 0.0], (n, 1)),
        body_z_ecef=np.tile([0.0, 0.0, 1.0], (n, 1)),
        gt_lat_deg=np.zeros(n, dtype=np.float64),
        gt_lon_deg=np.array(angles_deg, dtype=np.float64),
    )


def test_derive_access_fast_path_headless():
    result = derive_access_results_from_ephemeris(
        ephemeris=_make_ephemeris(),
        config=_make_config(),
        stations=[GroundStationConfig("Equator", 0.0, 0.0, 0.0)],
    )
    assert list(result.station_elevation_series) == ["Equator"]
    assert len(result.passes) == 1
    elevations = result.station_elevation_series["Equator"]
    assert elevations[0] < 0.0
    assert elevations[2] > 80.0
    assert elevations[-1] < 0.0
