"""Single public entry point for the toolbox (used by GUI, scripts, notebooks).

The facade is intentionally thin: it forwards to the appropriate
``cosmic_toolbox.services`` / ``cosmic_toolbox.analyses`` helpers and threads a
:class:`ProgressReporter` and :class:`CancelToken` through every operation.

No Qt or plotting imports allowed in this module.
"""

from __future__ import annotations

from typing import List

import numpy as np

from cosmic_toolbox.models import (
    AnalysisConfig,
    AnalysisResult,
    DerivedAccessResult,
    GroundStationConfig,
    PropagatedEphemeris,
)
from cosmic_toolbox.services.cancel import CancelToken
from cosmic_toolbox.services.progress import NullProgress, ProgressReporter


class ToolboxFacade:
    """Stable, GUI-agnostic surface for analyses provided by ``cosmic_toolbox``.

    Args:
        progress: Default progress reporter used by long-running operations
            when the caller doesn't supply one. Defaults to a no-op.
        cancel: Default cooperative cancel token. Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> None:
        self._progress = progress if progress is not None else NullProgress()
        self._cancel = cancel

    # ── Access analysis ────────────────────────────────────────────────
    def run_access(
        self,
        config: AnalysisConfig,
        stations: List[GroundStationConfig] | None = None,
        *,
        progress: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> AnalysisResult:
        """Propagate the orbit and compute multi-station access statistics."""

        from cosmic_toolbox.services.access_analysis import run_access_analysis

        return run_access_analysis(
            config,
            stations,
            progress=progress if progress is not None else self._progress,
            cancel=cancel if cancel is not None else self._cancel,
        )

    def derive_access_from_ephemeris(
        self,
        *,
        ephemeris: PropagatedEphemeris,
        config: AnalysisConfig,
        stations: List[GroundStationConfig] | None = None,
    ) -> DerivedAccessResult:
        """Derive station-dependent access outputs from a cached ephemeris."""

        from cosmic_toolbox.services.cached_access_recompute import (
            derive_access_results_from_ephemeris,
        )

        return derive_access_results_from_ephemeris(
            ephemeris=ephemeris, config=config, stations=stations
        )

    # ── Link budget ────────────────────────────────────────────────────
    def run_link_budget(self, inputs):  # type: ignore[no-untyped-def]
        """Run the standalone link-budget pipeline (CLI/script convenience).

        Accepts a :class:`cosmic_toolbox.tools.link_budget_profile.LinkBudgetInputs`
        and returns the timing dictionary produced by ``run_pipeline``.
        """

        from cosmic_toolbox.tools.link_budget_profile import LossCache, run_pipeline

        return run_pipeline(inputs, LossCache())

    # ── Data volume ────────────────────────────────────────────────────
    def run_data_volume(
        self,
        *,
        result: AnalysisResult,
        architecture,  # type: ignore[no-untyped-def]
        stations: List[GroundStationConfig],
        scenario_start_time,  # datetime
        comms_pointing_mode: str = "prograde_pointing",
        comms_pointing_aoa_limit_deg: float = 5.0,
        contact_elevation_deg: float = 10.0,
        filtered_passes=None,
    ):
        """Evaluate one architecture against an :class:`AnalysisResult`."""

        from cosmic_toolbox.services.scenario_data_volume import (
            ScenarioDataVolumeEvaluator,
            build_access_series,
        )

        series = build_access_series(result)
        evaluator = ScenarioDataVolumeEvaluator(
            access_series=series,
            station_lookup={s.name: s for s in stations},
            scenario_start_time=scenario_start_time,
            comms_pointing_mode=comms_pointing_mode,
            comms_pointing_aoa_limit_deg=comms_pointing_aoa_limit_deg,
            contact_elevation_deg=contact_elevation_deg,
        )
        return evaluator.evaluate_architecture(
            architecture=architecture,
            filtered_passes=list(filtered_passes) if filtered_passes is not None else list(result.passes),
        )

    # ── Orbit summary ──────────────────────────────────────────────────
    def build_orbit_summary(self, *, times_s, instantaneous):  # type: ignore[no-untyped-def]
        """Compute orbit-averaged summary series and stats."""

        from cosmic_toolbox.analyses.orbit_summary import (
            compute_orbit_averaged_orbit_summary,
            compute_series_stats,
        )

        averaged, meta = compute_orbit_averaged_orbit_summary(
            times_s=np.asarray(times_s, dtype=float),
            inst=instantaneous,
        )
        stats = compute_series_stats(averaged)
        return {"averaged": averaged, "meta": meta, "stats": stats}

    # ── PFD ────────────────────────────────────────────────────────────
    def pfd_compliance_backoff_db(self, **kwargs) -> float:
        """Convenience wrapper around :func:`analyses.pfd.compliance_backoff_db`."""

        from cosmic_toolbox.analyses.pfd import compliance_backoff_db

        return compliance_backoff_db(**kwargs)

    # ── Shared inputs (defaults, stations, antenna LUTs, ITU losses) ────
    # These are the read-only helpers every analysis script needs; exposing
    # them here means scripts import the facade instead of reaching into
    # cosmic_toolbox.services.* internals.
    @staticmethod
    def link_budget_defaults(path=None):
        """Canonical link-budget defaults (see resources/link_budget_defaults.yaml)."""

        from cosmic_toolbox.link_budget_defaults import load_link_budget_defaults

        return load_link_budget_defaults(path)

    @staticmethod
    def load_station(name: str) -> GroundStationConfig:
        """Canonical :class:`GroundStationConfig` for a database station name."""

        from cosmic_toolbox.services.station_importer import load_station_by_name

        return load_station_by_name(name)

    @staticmethod
    def load_stations_from_file(path) -> List[GroundStationConfig]:
        """Load ground stations from a CSV/Excel file (name-only or full format)."""

        from cosmic_toolbox.services.station_importer import (
            load_ground_stations_from_file,
        )

        return load_ground_stations_from_file(path)

    @staticmethod
    def default_antenna_lut_path():
        """Path to the bundled default synthesized antenna LUT."""

        from cosmic_toolbox.services.antenna_pattern import (
            default_synthesized_lut_path,
        )

        return default_synthesized_lut_path()

    @staticmethod
    def load_antenna_lut(path=None):
        """Load a spherical antenna-gain LUT (defaults to the bundled synthesized LUT)."""

        from cosmic_toolbox.services.antenna_pattern import (
            default_synthesized_lut_path,
            load_spherical_gain_lut,
        )

        return load_spherical_gain_lut(
            path if path is not None else default_synthesized_lut_path()
        )

    @staticmethod
    def estimate_slant_path_loss(*args, **kwargs):
        """ITU-R slant-path atmospheric loss (see :func:`itu_losses.estimate_slant_path_loss`)."""

        from cosmic_toolbox.itu_losses import estimate_slant_path_loss

        return estimate_slant_path_loss(*args, **kwargs)


__all__ = ["ToolboxFacade"]
