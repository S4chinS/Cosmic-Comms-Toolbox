"""Tests for the antenna-pattern LUT helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

# Repo layout: <repo_root>/src, <repo_root>/tests

from cosmic_toolbox.models import GroundStationConfig
from cosmic_toolbox.services import antenna_pattern
from cosmic_toolbox import link_budget_math


def _write_test_lut(tmp_path: Path, *, grid: np.ndarray) -> Path:
    path = tmp_path / "test_lut.npz"
    np.savez_compressed(
        path,
        az_deg=np.array([0.0, 180.0, 360.0], dtype=float),
        el_deg=np.array([-90.0, 0.0, 90.0], dtype=float),
        gain_dbi_grid=grid,
    )
    return path


def test_load_spherical_gain_lut_rejects_missing_keys():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "invalid_lut.npz"
        np.savez_compressed(path, az_deg=np.array([0.0, 360.0], dtype=float))

        with pytest.raises(KeyError):
            antenna_pattern.load_spherical_gain_lut(path)

def test_spherical_gain_lut_wraps_azimuth():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_test_lut(
            Path(tmp),
            grid=np.array(
                [
                    [0.0, 1.0, 2.0],
                    [10.0, 11.0, 12.0],
                    [20.0, 21.0, 22.0],
                ],
                dtype=float,
            ),
        )
        lut = antenna_pattern.load_spherical_gain_lut(path)

        wrapped = lut.gain_dbi(np.array([360.0, 540.0]), np.array([0.0, 0.0]))
        direct = lut.gain_dbi(np.array([0.0, 180.0]), np.array([0.0, 0.0]))

        assert np.allclose(wrapped, direct)


def test_evaluate_station_gain_series_uses_body_z_boresight():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_test_lut(
            Path(tmp),
            grid=np.array(
                [
                    [-20.0, -10.0, 5.0],
                    [-20.0, -10.0, 5.0],
                    [-20.0, -10.0, 5.0],
                ],
                dtype=float,
            ),
        )
        lut = antenna_pattern.load_spherical_gain_lut(path)
        station = GroundStationConfig(
            name="NorthPole",
            latitude_deg=90.0,
            longitude_deg=0.0,
            altitude_m=0.0,
        )
        gs_ecef = antenna_pattern.station_ecef_m(station)
        sat_ecef = np.array([gs_ecef - np.array([0.0, 0.0, 1000.0], dtype=float)])
        body_x = np.array([[1.0, 0.0, 0.0]], dtype=float)
        body_y = np.array([[0.0, 1.0, 0.0]], dtype=float)
        body_z = np.array([[0.0, 0.0, 1.0]], dtype=float)

        gains_dbi, az_deg, el_deg, roll_deg = antenna_pattern.evaluate_station_gain_series(
            lut=lut,
            station=station,
            sat_ecef_m=sat_ecef,
            body_x_ecef=body_x,
            body_y_ecef=body_y,
            body_z_ecef=body_z,
            pointing_mode="prograde_pointing",
        )

        assert np.allclose(az_deg, [0.0])
        assert np.allclose(el_deg, [90.0])
        assert np.allclose(gains_dbi, [5.0])
        assert np.allclose(roll_deg, [0.0])


def test_gain_series_feeds_vector_link_budget():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_test_lut(
            Path(tmp),
            grid=np.array(
                [
                    [-15.0, 0.0, 5.0],
                    [-15.0, 0.0, 5.0],
                    [-15.0, 0.0, 5.0],
                ],
                dtype=float,
            ),
        )
        lut = antenna_pattern.load_spherical_gain_lut(path)
        station = GroundStationConfig(
            name="NorthPole",
            latitude_deg=90.0,
            longitude_deg=0.0,
            altitude_m=0.0,
        )
        gs_ecef = antenna_pattern.station_ecef_m(station)
        sat_ecef = np.array(
            [
                gs_ecef - np.array([0.0, 0.0, 1000.0], dtype=float),
                gs_ecef - np.array([1000.0, 0.0, 0.0], dtype=float),
            ]
        )
        body_x = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        body_y = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        body_z = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float)
        gains_dbi, _az_deg, _el_deg, _roll_deg = antenna_pattern.evaluate_station_gain_series(
            lut=lut,
            station=station,
            sat_ecef_m=sat_ecef,
            body_x_ecef=body_x,
            body_y_ecef=body_y,
            body_z_ecef=body_z,
            pointing_mode="prograde_pointing",
        )

        results = link_budget_math.calculate_link_budget(
            elevations_deg=np.array([45.0, 45.0], dtype=float),
            antenna_gains_dBi=gains_dbi,
            atmospheric_losses_dB=np.array([0.5, 0.5], dtype=float),
            tx_power_dBw=3.0,
            tx_boresight_gain_dBi=5.0,
            tx_losses_dB=1.0,
            tx_backoff_dB=0.0,
            frequency_GHz=8.2,
            satellite_altitude_km=np.array([550.0, 550.0], dtype=float),
            ground_altitude_m=0.0,
            rx_antenna_gain_dBi=34.0,
            receiver_noise_figure_dB=1.5,
            sky_background_temperature_K=50.0,
            receiver_losses_dB=0.0,
            polarization_loss_dB=0.1,
            symbol_rate_sps=150e6,
            implementation_loss_dB=1.0,
            margin_dB=3.0,
        )

        assert gains_dbi.shape == (2,)
        assert np.all(np.isfinite(results["es_to_n0_dB"]))
        assert results["pointing_loss_dB"].shape == (2,)


def test_free_roll_about_body_x_points_boresight_toward_station():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_test_lut(
            Path(tmp),
            grid=np.array(
                [
                    [-15.0, -15.0, -15.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 5.0, 5.0],
                ],
                dtype=float,
            ),
        )
        lut = antenna_pattern.load_spherical_gain_lut(path)
        station = GroundStationConfig(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_m=0.0,
        )
        gs_ecef = antenna_pattern.station_ecef_m(station)
        sat_ecef = np.array([gs_ecef + np.array([0.0, -1000.0, 0.0], dtype=float)])
        body_x = np.array([[1.0, 0.0, 0.0]], dtype=float)
        body_y = np.array([[0.0, 1.0, 0.0]], dtype=float)
        body_z = np.array([[0.0, 0.0, 1.0]], dtype=float)

        _fixed_gains, _fixed_az, fixed_el, fixed_roll = antenna_pattern.evaluate_station_gain_series(
            lut=lut,
            station=station,
            sat_ecef_m=sat_ecef,
            body_x_ecef=body_x,
            body_y_ecef=body_y,
            body_z_ecef=body_z,
            pointing_mode="prograde_pointing",
        )
        _rolled_gains, _rolled_az, rolled_el, rolled_roll = antenna_pattern.evaluate_station_gain_series(
            lut=lut,
            station=station,
            sat_ecef_m=sat_ecef,
            body_x_ecef=body_x,
            body_y_ecef=body_y,
            body_z_ecef=body_z,
            pointing_mode="free_to_roll",
        )

        assert np.allclose(fixed_el, [0.0])
        assert np.allclose(fixed_roll, [0.0])
        assert np.allclose(rolled_el, [90.0])
        assert np.allclose(rolled_roll, [90.0])


def test_constrained_aoa_limits_body_x_deviation():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_test_lut(
            Path(tmp),
            grid=np.array(
                [
                    [-15.0, -15.0, -15.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 5.0, 5.0],
                ],
                dtype=float,
            ),
        )
        lut = antenna_pattern.load_spherical_gain_lut(path)
        station = GroundStationConfig(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_m=0.0,
        )
        gs_ecef = antenna_pattern.station_ecef_m(station)
        sat_ecef = np.array([gs_ecef + np.array([1000.0, -1000.0, 0.0], dtype=float)])
        body_x = np.array([[1.0, 0.0, 0.0]], dtype=float)
        body_y = np.array([[0.0, 1.0, 0.0]], dtype=float)
        body_z = np.array([[0.0, 0.0, 1.0]], dtype=float)

        _gains, _az, el_deg, roll_deg = antenna_pattern.evaluate_station_gain_series(
            lut=lut,
            station=station,
            sat_ecef_m=sat_ecef,
            body_x_ecef=body_x,
            body_y_ecef=body_y,
            body_z_ecef=body_z,
            pointing_mode="constrained_aoa",
            max_aoa_deg=10.0,
        )

        assert np.allclose(roll_deg, [0.0])
        assert np.allclose(el_deg, [55.0], atol=1e-6)


def test_steering_active_mask_only_applies_during_comms():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_test_lut(
            Path(tmp),
            grid=np.array(
                [
                    [-15.0, -15.0, -15.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 5.0, 5.0],
                ],
                dtype=float,
            ),
        )
        lut = antenna_pattern.load_spherical_gain_lut(path)
        station = GroundStationConfig(
            name="Equator",
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_m=0.0,
        )
        gs_ecef = antenna_pattern.station_ecef_m(station)
        sat_ecef = np.array(
            [
                gs_ecef + np.array([0.0, -1000.0, 0.0], dtype=float),
                gs_ecef + np.array([0.0, -1000.0, 0.0], dtype=float),
            ]
        )
        body_x = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        body_y = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        body_z = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float)

        _gains, _az, el_deg, roll_deg = antenna_pattern.evaluate_station_gain_series(
            lut=lut,
            station=station,
            sat_ecef_m=sat_ecef,
            body_x_ecef=body_x,
            body_y_ecef=body_y,
            body_z_ecef=body_z,
            pointing_mode="free_to_roll",
            steering_active_mask=np.array([False, True], dtype=bool),
        )

        assert np.allclose(el_deg, [0.0, 90.0])
        assert np.allclose(roll_deg, [0.0, 90.0])
