"""Cosmic Comms Toolbox: Orekit-backed access, link-budget, and data-volume analyses.

The public surface is:

- :class:`cosmic_toolbox.facade.ToolboxFacade` (preferred entry point)
- :mod:`cosmic_toolbox.models` (typed dataclasses)
- :mod:`cosmic_toolbox.services` (lower-level helpers)
- :mod:`cosmic_toolbox.analyses` (pure-compute helpers lifted from the GUI)

This package never imports PySide6, pyqtgraph, matplotlib, cartopy, or moderngl.
"""

from __future__ import annotations

from cosmic_toolbox.facade import ToolboxFacade
from cosmic_toolbox.models import (
    AnalysisConfig,
    AnalysisOptions,
    AnalysisResult,
    AnalysisSummary,
    DerivedAccessResult,
    GroundStationConfig,
    OrbitConfig,
    PassStatistic,
    PropagatedEphemeris,
    PropagationConfig,
    ScenarioConfig,
    StationSummary,
)
from cosmic_toolbox.services.cancel import AnalysisCancelled, CancelToken
from cosmic_toolbox.services.progress import NullProgress, ProgressReporter

__all__ = [
    "ToolboxFacade",
    "AnalysisCancelled",
    "AnalysisConfig",
    "AnalysisOptions",
    "AnalysisResult",
    "AnalysisSummary",
    "CancelToken",
    "DerivedAccessResult",
    "GroundStationConfig",
    "NullProgress",
    "OrbitConfig",
    "PassStatistic",
    "ProgressReporter",
    "PropagatedEphemeris",
    "PropagationConfig",
    "ScenarioConfig",
    "StationSummary",
]
