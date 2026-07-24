"""Qt workers that drive long-running toolbox operations on background threads."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from cosmic_toolbox import ToolboxFacade
from cosmic_toolbox.models import AnalysisConfig, GroundStationConfig, PropagatedEphemeris  # noqa: F401
from cosmic_toolbox.services.cancel import AnalysisCancelled, CancelToken
from cosmic_toolbox.services.progress import ProgressReporter


class QtProgressReporter:
    """Adapter from :class:`ProgressReporter` to a Qt ``Signal``."""

    def __init__(self, signal) -> None:  # type: ignore[no-untyped-def]
        self._signal = signal

    def report(self, fraction: float, message: str | None = None) -> None:
        percent = max(0.0, min(1.0, float(fraction))) * 100.0
        self._signal.emit(percent)


class AnalysisWorker(QObject):
    """Background worker that executes the access analysis and emits progress."""

    progress = Signal(float)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        config: AnalysisConfig,
        stations: list[GroundStationConfig],
        *,
        facade: ToolboxFacade | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._stations = stations
        self._facade = facade or ToolboxFacade()
        self._cancel = CancelToken()

    def request_cancel(self) -> None:
        """Cooperatively cancel the in-flight analysis."""

        self._cancel.cancel()

    def run(self) -> None:
        reporter: ProgressReporter = QtProgressReporter(self.progress)
        try:
            result = self._facade.run_access(
                self._config,
                self._stations,
                progress=reporter,
                cancel=self._cancel,
            )
        except AnalysisCancelled:
            self.error.emit("Analysis cancelled by user.")
            return
        except Exception as exc:  # pragma: no cover - GUI execution path
            self.error.emit(str(exc))
            raise
        self.finished.emit(result)


class CachedRecomputeWorker(QObject):
    """Background worker that recomputes station access from a cached ephemeris."""

    # Carries (AnalysisResult, AnalysisConfig, list[GroundStationConfig], sv_arrays | None)
    finished: Signal = Signal(object, object, object, object)
    error: Signal = Signal(str)

    def __init__(
        self,
        ephemeris: PropagatedEphemeris,
        config: AnalysisConfig,
        stations: list[GroundStationConfig],
    ) -> None:
        super().__init__()
        self._ephemeris = ephemeris
        self._config = config
        self._stations = stations

    def run(self) -> None:
        try:
            from cosmic_toolbox.services.cached_access_recompute import (
                analysis_result_from_components,
                derive_access_results_from_ephemeris,
            )

            eph = self._ephemeris
            has_arrays = (
                eph.ecef_pos_km is not None
                and eph.ecef_vel_km_s is not None
                and eph.body_x_ecef is not None
            )
            sat_ecef_m: np.ndarray | None
            sat_ecef_v_mps: np.ndarray | None
            if has_arrays:
                sat_ecef_m = np.asarray(eph.ecef_pos_km) * 1000.0
                sat_ecef_v_mps = np.asarray(eph.ecef_vel_km_s) * 1000.0
                sv_arrays: dict | None = {
                    "sat_ecef_m": sat_ecef_m,
                    "body_x_ecef": eph.body_x_ecef,
                    "body_y_ecef": eph.body_y_ecef,
                    "body_z_ecef": eph.body_z_ecef,
                }
            else:
                sat_ecef_m = sat_ecef_v_mps = None
                sv_arrays = None

            derived_access = derive_access_results_from_ephemeris(
                ephemeris=eph,
                config=self._config,
                stations=self._stations if self._stations else None,
                _prebuilt_sat_ecef_m=sat_ecef_m,
                _prebuilt_sat_ecef_v_mps=sat_ecef_v_mps,
            )
            result = analysis_result_from_components(
                ephemeris=eph,
                derived_access=derived_access,
            )
            self.finished.emit(result, self._config, self._stations, sv_arrays)
        except Exception as exc:
            self.error.emit(str(exc))


__all__ = [
    "AnalysisWorker",
    "CachedRecomputeWorker",
    "QtProgressReporter",
]
