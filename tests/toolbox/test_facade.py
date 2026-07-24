"""Behavioural tests for :class:`cosmic_toolbox.facade.ToolboxFacade`."""

from __future__ import annotations

import numpy as np

from cosmic_toolbox import (
    AnalysisCancelled,
    CancelToken,
    NullProgress,
    ProgressReporter,
    ToolboxFacade,
)
from cosmic_toolbox.analyses.orbit_summary import (
    compute_orbit_averaged_orbit_summary,
    compute_series_stats,
)


def test_null_progress_is_a_protocol_instance():
    np_ = NullProgress()
    assert isinstance(np_, ProgressReporter)
    np_.report(0.5, "still going")  # no-op, must not raise


def test_cancel_token_round_trip():
    token = CancelToken()
    assert not token.cancelled
    token.cancel()
    assert token.cancelled
    try:
        token.check()
    except AnalysisCancelled:
        return
    raise AssertionError("CancelToken.check() should have raised AnalysisCancelled")


def test_orbit_summary_facade_smoke():
    """The facade orbit-summary helper should match the analysis function directly."""

    times_s = np.linspace(0.0, 6000.0, 600)
    inst = {
        "orbital_altitude_km": [550.0 + 0.001 * t for t in times_s],
        "semi_major_axis_km": [6928.0] * times_s.size,
        "perigee_altitude_km": [549.0] * times_s.size,
        "apogee_altitude_km": [551.0] * times_s.size,
        "eccentricity": [0.0001] * times_s.size,
        "inclination_deg": [97.5] * times_s.size,
        "argument_of_perigee_deg": [(t * 0.01) % 360.0 for t in times_s],
        "true_anomaly_deg": [(t * 0.05) % 360.0 for t in times_s],
        "orbital_period_s": [5800.0] * times_s.size,
        "angle_of_attack_deg": [0.0] * times_s.size,
    }

    facade_out = ToolboxFacade(progress=NullProgress()).build_orbit_summary(
        times_s=times_s, instantaneous=inst
    )
    direct_avg, direct_meta = compute_orbit_averaged_orbit_summary(times_s=times_s, inst=inst)
    direct_stats = compute_series_stats(direct_avg)

    assert facade_out["averaged"].keys() == direct_avg.keys()
    assert facade_out["meta"] == direct_meta
    assert facade_out["stats"].keys() == direct_stats.keys()


def test_pfd_compliance_backoff_via_facade():
    elevations = np.linspace(5.0, 90.0, 18)
    backoff = ToolboxFacade().pfd_compliance_backoff_db(
        elevations_deg=elevations,
        sat_altitude_km=550.0,
        ground_altitude_m=10.0,
        tx_power_dBw=3.0,
        tx_losses_dB=2.0,
        antenna_gain_dBi=5.4,
        frequency_GHz=8.2,
        symbol_rate_sps=300e6,
        rolloff=0.25,
    )
    assert backoff >= 0.0
