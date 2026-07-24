"""Tests for cached trajectory scenario package import/export."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from cosmic_toolbox.models import PropagatedEphemeris
from cosmic_toolbox.services.scenario_package_io import (
    export_cached_trajectory_package,
    import_cached_trajectory_package,
)


def _make_ephemeris() -> PropagatedEphemeris:
    start = datetime(2026, 3, 9, tzinfo=timezone.utc)
    n = 3
    timestamps_unix = np.array(
        [(start + timedelta(seconds=i * 60)).timestamp() for i in range(n)],
        dtype=np.float64,
    )
    eci_pos_km = np.array(
        [[7000.0 + i, 10.0 * i, 20.0 * i] for i in range(n)], dtype=np.float64
    )
    eci_vel_km_s = np.array(
        [[0.1 * i, 7.5, -0.2 * i] for i in range(n)], dtype=np.float64
    )
    ecef_pos_km = np.array(
        [[6990.0 + i, 11.0 * i, 21.0 * i] for i in range(n)], dtype=np.float64
    )
    ecef_vel_km_s = np.array(
        [[0.3 * i, 7.4, -0.1 * i] for i in range(n)], dtype=np.float64
    )
    body_x_ecef = np.tile([1.0, 0.0, 0.0], (n, 1))
    body_y_ecef = np.tile([0.0, 1.0, 0.0], (n, 1))
    body_z_ecef = np.tile([0.0, 0.0, 1.0], (n, 1))
    gt_lat_deg = np.array([float(i) for i in range(n)], dtype=np.float64)
    gt_lon_deg = np.array([float(i * 2) for i in range(n)], dtype=np.float64)
    return PropagatedEphemeris(
        timeline_seconds=np.array([0.0, 60.0, 120.0], dtype=np.float64),
        orbit_period_seconds=5400.0,
        mean_ic_report="Mean IC: applied",
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
        orbital_altitude_km=np.array([500.0, 500.1, 500.2], dtype=np.float64),
    )


def test_scenario_package_roundtrip():
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ephemeris = _make_ephemeris()
        settings = {
            "orbit": {"altitude_km": 250.0},
            "scenario": {"start_time_utc": "2026-03-09T00:00:00+00:00"},
        }
        stations = [
            {
                "name": "A",
                "latitude_deg": 1.0,
                "longitude_deg": 2.0,
                "altitude_m": 3.0,
            }
        ]
        enabled = ["A"]

        npz_path, sidecar_path = export_cached_trajectory_package(
            ephemeris=ephemeris,
            settings=settings,
            stations=stations,
            enabled_station_names=enabled,
            output_path=tmp_path / "case.npz",
        )

        assert npz_path.exists()
        assert sidecar_path.exists()

        # import via npz path
        imported = import_cached_trajectory_package(npz_path)

        assert imported.settings == settings
        assert imported.stations == stations
        assert imported.enabled_station_names == enabled
        assert imported.ephemeris.orbit_period_seconds == ephemeris.orbit_period_seconds
        assert imported.ephemeris.mean_ic_report == ephemeris.mean_ic_report
        assert np.array_equal(imported.ephemeris.timeline_seconds, ephemeris.timeline_seconds)
        assert imported.ephemeris.orbital_altitude_km == pytest.approx(ephemeris.orbital_altitude_km)

        # Check sample index 1 using numpy arrays
        assert imported.ephemeris.eci_pos_km[1, 1] == pytest.approx(ephemeris.eci_pos_km[1, 1])
        assert imported.ephemeris.ecef_pos_km[1, 1] == pytest.approx(ephemeris.ecef_pos_km[1, 1])
        assert imported.ephemeris.ecef_vel_km_s[1, 0] == pytest.approx(ephemeris.ecef_vel_km_s[1, 0])
        assert imported.ephemeris.body_x_ecef[1, 0] == pytest.approx(ephemeris.body_x_ecef[1, 0])

        # Check ground track at index 2
        assert imported.ephemeris.gt_lat_deg[2] == pytest.approx(ephemeris.gt_lat_deg[2])
        assert imported.ephemeris.gt_lon_deg[2] == pytest.approx(ephemeris.gt_lon_deg[2])


def test_import_via_sidecar_path():
    """import_cached_trajectory_package should also accept the .toolbox.json path."""
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ephemeris = _make_ephemeris()
        _, sidecar_path = export_cached_trajectory_package(
            ephemeris=ephemeris,
            settings={"orbit": {}, "scenario": {}},
            stations=[],
            enabled_station_names=[],
            output_path=tmp_path / "case.npz",
        )
        imported = import_cached_trajectory_package(sidecar_path)
        assert imported.ephemeris.timestamps_unix.shape[0] == 3


def test_import_requires_matching_sidecar():
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ephemeris = _make_ephemeris()
        npz_path, sidecar_path = export_cached_trajectory_package(
            ephemeris=ephemeris,
            settings={"orbit": {}, "scenario": {}},
            stations=[],
            enabled_station_names=[],
            output_path=tmp_path / "case.npz",
        )
        sidecar_path.unlink()

        with pytest.raises(FileNotFoundError):
            import_cached_trajectory_package(npz_path)


def test_legacy_schema_v1_import():
    """Schema-version-1 packages (OEM + fat JSON) should still load correctly."""
    from cosmic_toolbox.services.scenario_package_io import (
        _isoformat_utc,
        _LEGACY_SCHEMA_VERSION,
    )

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ephemeris = _make_ephemeris()
        n = ephemeris.timestamps_unix.shape[0]

        # Build OEM content from numpy arrays
        oem_lines = [
            "CCSDS_OEM_VERS = 2.0",
            "META_START",
            "REF_FRAME = EME2000",
            "META_STOP",
            "",
        ]
        for i in range(n):
            ts = datetime.fromtimestamp(float(ephemeris.timestamps_unix[i]), tz=timezone.utc)
            oem_lines.append(
                f"{_isoformat_utc(ts)} "
                f"{ephemeris.eci_pos_km[i,0]:.9f} {ephemeris.eci_pos_km[i,1]:.9f} {ephemeris.eci_pos_km[i,2]:.9f} "
                f"{ephemeris.eci_vel_km_s[i,0]:.12f} {ephemeris.eci_vel_km_s[i,1]:.12f} {ephemeris.eci_vel_km_s[i,2]:.12f}"
            )
        oem_path = tmp_path / "leg.oem"
        oem_path.write_text("\n".join(oem_lines), encoding="ascii")

        # Build v1 fat JSON sidecar using raw dicts
        sv_list = [
            {
                "timestamp_utc": _isoformat_utc(
                    datetime.fromtimestamp(float(ephemeris.timestamps_unix[i]), tz=timezone.utc)
                ),
                "eci_x_km": float(ephemeris.eci_pos_km[i, 0]),
                "eci_y_km": float(ephemeris.eci_pos_km[i, 1]),
                "eci_z_km": float(ephemeris.eci_pos_km[i, 2]),
                "eci_vx_km_s": float(ephemeris.eci_vel_km_s[i, 0]),
                "eci_vy_km_s": float(ephemeris.eci_vel_km_s[i, 1]),
                "eci_vz_km_s": float(ephemeris.eci_vel_km_s[i, 2]),
                "ecef_x_km": float(ephemeris.ecef_pos_km[i, 0]),
                "ecef_y_km": float(ephemeris.ecef_pos_km[i, 1]),
                "ecef_z_km": float(ephemeris.ecef_pos_km[i, 2]),
                "ecef_vx_km_s": float(ephemeris.ecef_vel_km_s[i, 0]),
                "ecef_vy_km_s": float(ephemeris.ecef_vel_km_s[i, 1]),
                "ecef_vz_km_s": float(ephemeris.ecef_vel_km_s[i, 2]),
                "body_x_ecef_x": float(ephemeris.body_x_ecef[i, 0]),
                "body_x_ecef_y": float(ephemeris.body_x_ecef[i, 1]),
                "body_x_ecef_z": float(ephemeris.body_x_ecef[i, 2]),
                "body_y_ecef_x": float(ephemeris.body_y_ecef[i, 0]),
                "body_y_ecef_y": float(ephemeris.body_y_ecef[i, 1]),
                "body_y_ecef_z": float(ephemeris.body_y_ecef[i, 2]),
                "body_z_ecef_x": float(ephemeris.body_z_ecef[i, 0]),
                "body_z_ecef_y": float(ephemeris.body_z_ecef[i, 1]),
                "body_z_ecef_z": float(ephemeris.body_z_ecef[i, 2]),
            }
            for i in range(n)
        ]
        gt_list = [
            {
                "timestamp_utc": _isoformat_utc(
                    datetime.fromtimestamp(float(ephemeris.timestamps_unix[i]), tz=timezone.utc)
                ),
                "latitude_deg": float(ephemeris.gt_lat_deg[i]),
                "longitude_deg": float(ephemeris.gt_lon_deg[i]),
                "x_km": float(ephemeris.ecef_pos_km[i, 0]),
                "y_km": float(ephemeris.ecef_pos_km[i, 1]),
                "z_km": float(ephemeris.ecef_pos_km[i, 2]),
            }
            for i in range(n)
        ]

        v1_payload = {
            "schema_version": _LEGACY_SCHEMA_VERSION,
            "settings": {"orbit": {}, "scenario": {}},
            "stations": [],
            "enabled_station_names": [],
            "ephemeris": {
                "timeline_seconds": ephemeris.timeline_seconds.tolist(),
                "orbit_period_seconds": ephemeris.orbit_period_seconds,
                "mean_ic_report": ephemeris.mean_ic_report,
                "ground_track": gt_list,
                "state_vectors": sv_list,
                "orbital_altitude_km": ephemeris.orbital_altitude_km.tolist(),
                "semi_major_axis_km": [],
                "perigee_altitude_km": [],
                "apogee_altitude_km": [],
                "eccentricity": [],
                "inclination_deg": [],
                "argument_of_perigee_deg": [],
                "orbital_period_series_s": [],
                "true_anomaly_deg": [],
                "angle_of_attack_deg": [],
            },
        }
        sidecar_path = tmp_path / "leg.toolbox.json"
        sidecar_path.write_text(json.dumps(v1_payload), encoding="utf-8")

        imported = import_cached_trajectory_package(oem_path)
        assert imported.ephemeris.timestamps_unix.shape[0] == 3
        assert imported.ephemeris.ecef_pos_km[1, 1] == pytest.approx(float(ephemeris.ecef_pos_km[1, 1]))
