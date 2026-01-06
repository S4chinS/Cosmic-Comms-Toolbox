"""Link-budget timing utilities shared by CLI and tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import time
import numpy as np

from src.models import GroundStationConfig
from src.itu_losses import estimate_slant_path_loss
from src import link_budget_math


@dataclass(frozen=True)
class LinkBudgetInputs:
    frequency_GHz: float = 8.2
    tx_power_dBw: float = 3.0
    tx_boresight_gain_dBi: float = 5.41
    tx_losses_dB: float = 2.0
    tx_backoff_dB: float = 0.0
    antenna_gain_dBi: float = 5.41
    receiver_G_T_dB_K: float = 26.0
    receiver_losses_dB: float = 0.0
    symbol_rate_Msps: float = 300.0
    implementation_loss_dB: float = 1.0
    margin_dB: float = 3.0
    satellite_altitude_km: float = 550.0
    min_gs_elevation_deg: float = 5.0
    gs_elevation_deg: float = 60.0
    unavailability_percent: float = 0.1
    polarization_loss_dB: float = 0.1
    rolloff: float = 0.25


BASE_STATION = GroundStationConfig(
    name="Svalbard (Ny-Alesund)",
    latitude_deg=78.92,
    longitude_deg=11.93,
    altitude_m=30.0,
)


@dataclass
class LossCache:
    key: Optional[tuple] = None
    losses: Optional[np.ndarray] = None
    contributions: Optional[Dict[str, np.ndarray]] = None

    def build_key(
        self,
        *,
        frequency: float,
        plot_lower_bound: float,
        unavailability: float,
        num_samples: int,
    ) -> tuple:
        return (
            round(float(frequency), 6),
            round(float(plot_lower_bound), 4),
            round(float(unavailability), 6),
            round(float(BASE_STATION.latitude_deg), 6),
            round(float(BASE_STATION.longitude_deg), 6),
            round(float(BASE_STATION.altitude_m), 3),
            int(num_samples),
        )

    def get(
        self,
        key: tuple,
        shape: Tuple[int, ...],
    ) -> Optional[Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]]:
        if self.key == key and self.losses is not None and self.losses.shape == shape:
            return self.losses, self.contributions
        return None

    def update(
        self,
        key: tuple,
        losses: np.ndarray,
        contributions: Optional[Dict[str, np.ndarray]],
    ) -> None:
        self.key = key
        self.losses = losses
        self.contributions = contributions


def _generate_elevation_grid(inputs: LinkBudgetInputs) -> np.ndarray:
    min_el = min(inputs.min_gs_elevation_deg, 90.0)
    lower = max(0.0, min_el)
    return np.linspace(lower, 90.0, 1000)


def run_pipeline(
    inputs: LinkBudgetInputs, cache: Optional[LossCache]
) -> Dict[str, float]:
    elevations = _generate_elevation_grid(inputs)
    symbol_rate_sps = inputs.symbol_rate_Msps * 1e6

    timings: Dict[str, float] = {}

    cache_key = None
    cached_losses = None
    if cache is not None:
        cache_key = cache.build_key(
            frequency=inputs.frequency_GHz,
            plot_lower_bound=float(elevations[0]),
            unavailability=inputs.unavailability_percent,
            num_samples=elevations.size,
        )
        cached = cache.get(cache_key, elevations.shape)
        if cached:
            atmospheric_losses, contribution_breakdown = cached
            timings["itu_loss_seconds"] = 0.0
            cached_losses = True

    if not cached_losses:
        start = time.perf_counter()
        loss_result = estimate_slant_path_loss(
            frequency_GHz=inputs.frequency_GHz,
            elevations_deg=elevations,
            lat_deg=BASE_STATION.latitude_deg,
            lon_deg=BASE_STATION.longitude_deg,
            altitude_m=BASE_STATION.altitude_m,
            unavailability_percent=inputs.unavailability_percent,
            return_contributions=True,
        )
        timings["itu_loss_seconds"] = time.perf_counter() - start

        if isinstance(loss_result, tuple):
            atmospheric_losses, contribution_breakdown = loss_result
        else:
            atmospheric_losses = loss_result
            contribution_breakdown = None
        if cache is not None and cache_key is not None:
            cache.update(cache_key, atmospheric_losses, contribution_breakdown)

    start = time.perf_counter()
    results = link_budget_math.calculate_link_budget(
        elevations_deg=elevations,
        antenna_gains_dBi=np.full_like(elevations, inputs.antenna_gain_dBi),
        atmospheric_losses_dB=atmospheric_losses,
        tx_power_dBw=inputs.tx_power_dBw,
        tx_boresight_gain_dBi=inputs.tx_boresight_gain_dBi,
        tx_losses_dB=inputs.tx_losses_dB,
        tx_backoff_dB=inputs.tx_backoff_dB,
        frequency_GHz=inputs.frequency_GHz,
        satellite_altitude_km=inputs.satellite_altitude_km,
        ground_altitude_m=BASE_STATION.altitude_m,
        receiver_G_T_dB_K=inputs.receiver_G_T_dB_K,
        receiver_losses_dB=inputs.receiver_losses_dB,
        symbol_rate_sps=symbol_rate_sps,
        implementation_loss_dB=inputs.implementation_loss_dB,
        margin_dB=inputs.margin_dB,
    )
    timings["link_math_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    link_budget_math.build_parameter_rows(
        elevations_deg=elevations,
        results=results,
        evaluation_elevation_deg=min(
            90.0, max(inputs.gs_elevation_deg, inputs.min_gs_elevation_deg)
        ),
        elevation_lower_bound_deg=inputs.min_gs_elevation_deg,
        elevation_upper_bound_deg=90.0,
        tx_frequency_GHz=inputs.frequency_GHz,
        tx_power_dBw=inputs.tx_power_dBw,
        tx_losses_dB=inputs.tx_losses_dB,
        tx_boresight_gain_dBi=inputs.tx_boresight_gain_dBi,
        tx_backoff_dB=inputs.tx_backoff_dB,
        symbol_rate_sps=symbol_rate_sps,
        receiver_G_T_dB_K=inputs.receiver_G_T_dB_K,
        implementation_loss_dB=inputs.implementation_loss_dB,
        margin_dB=inputs.margin_dB,
        rolloff=inputs.rolloff,
        polarization_loss_dB=inputs.polarization_loss_dB,
        satellite_altitude_km=inputs.satellite_altitude_km,
        atmospheric_breakdown_dB=contribution_breakdown,
    )
    timings["table_seconds"] = time.perf_counter() - start

    timings["total_seconds"] = sum(timings.values())
    return timings


def profile_parameters(
    parameters: Iterable[Tuple[str, Callable[[LinkBudgetInputs], LinkBudgetInputs]]],
    *,
    use_cache: bool,
) -> List[Tuple[str, Dict[str, float]]]:
    results = []
    for name, mutator in parameters:
        cache = LossCache() if use_cache else None
        if cache is not None:
            run_pipeline(BASE_INPUTS, cache)
        updated = mutator(BASE_INPUTS)
        timings = run_pipeline(updated, cache)
        results.append((name, timings))
    return results


BASE_INPUTS = LinkBudgetInputs()

PARAMETER_MUTATORS: List[Tuple[str, Callable[[LinkBudgetInputs], LinkBudgetInputs]]] = [
    ("Frequency", lambda p: replace(p, frequency_GHz=p.frequency_GHz + 0.1)),
    ("TX power", lambda p: replace(p, tx_power_dBw=p.tx_power_dBw + 0.5)),
    (
        "TX gain",
        lambda p: replace(p, tx_boresight_gain_dBi=p.tx_boresight_gain_dBi + 0.2),
    ),
    ("TX losses", lambda p: replace(p, tx_losses_dB=p.tx_losses_dB + 0.2)),
    ("TX backoff", lambda p: replace(p, tx_backoff_dB=p.tx_backoff_dB + 0.2)),
    ("Antenna gain", lambda p: replace(p, antenna_gain_dBi=p.antenna_gain_dBi + 0.2)),
    ("Receiver G/T", lambda p: replace(p, receiver_G_T_dB_K=p.receiver_G_T_dB_K + 0.2)),
    (
        "Receiver losses",
        lambda p: replace(p, receiver_losses_dB=p.receiver_losses_dB + 0.2),
    ),
    ("Symbol rate", lambda p: replace(p, symbol_rate_Msps=p.symbol_rate_Msps + 10)),
    (
        "Implementation loss",
        lambda p: replace(p, implementation_loss_dB=p.implementation_loss_dB + 0.2),
    ),
    ("Margin", lambda p: replace(p, margin_dB=p.margin_dB + 0.2)),
    (
        "Satellite altitude",
        lambda p: replace(p, satellite_altitude_km=p.satellite_altitude_km + 10),
    ),
    (
        "Min GS elevation",
        lambda p: replace(p, min_gs_elevation_deg=p.min_gs_elevation_deg + 1),
    ),
    ("GS elevation", lambda p: replace(p, gs_elevation_deg=p.gs_elevation_deg + 1)),
    (
        "Unavailability",
        lambda p: replace(p, unavailability_percent=p.unavailability_percent + 0.1),
    ),
    (
        "Polarization loss",
        lambda p: replace(p, polarization_loss_dB=p.polarization_loss_dB + 0.05),
    ),
    ("Roll-off", lambda p: replace(p, rolloff=p.rolloff + 0.05)),
]


def main() -> None:
    header = f"{'Parameter':<22}  {'ITU (ms)':>10}  {'Link math (ms)':>14}  {'Table (ms)':>11}  {'Total (ms)':>11}"
    print(header)
    print("-" * len(header))
    for name, timings in profile_parameters(PARAMETER_MUTATORS, use_cache=True):
        print(
            f"{name:<22}  "
            f"{timings['itu_loss_seconds'] * 1e3:10.2f}  "
            f"{timings['link_math_seconds'] * 1e3:14.2f}  "
            f"{timings['table_seconds'] * 1e3:11.2f}  "
            f"{timings['total_seconds'] * 1e3:11.2f}"
        )


__all__ = [
    "LinkBudgetInputs",
    "LossCache",
    "BASE_INPUTS",
    "PARAMETER_MUTATORS",
    "profile_parameters",
    "run_pipeline",
    "main",
]
