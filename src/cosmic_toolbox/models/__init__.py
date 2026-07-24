"""Shared dataclasses for configs and analysis results."""

from __future__ import annotations

from cosmic_toolbox.models.configs import (
    AnalysisConfig,
    AnalysisOptions,
    GroundStationConfig,
    OrbitConfig,
    PropagationConfig,
    ScenarioConfig,
)
from cosmic_toolbox.models.results import (
    AnalysisResult,
    AnalysisSummary,
    DerivedAccessResult,
    GroundTrackPoint,
    PassStatistic,
    PropagatedEphemeris,
    StateVectorSample,
    StationSummary,
    analysis_result_sample_count,
    propagated_ephemeris_sample_count,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisOptions",
    "AnalysisResult",
    "AnalysisSummary",
    "DerivedAccessResult",
    "GroundStationConfig",
    "GroundTrackPoint",
    "OrbitConfig",
    "PassStatistic",
    "PropagatedEphemeris",
    "PropagationConfig",
    "ScenarioConfig",
    "StateVectorSample",
    "StationSummary",
    "analysis_result_sample_count",
    "propagated_ephemeris_sample_count",
]
