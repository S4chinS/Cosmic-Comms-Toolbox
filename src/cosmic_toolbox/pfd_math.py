"""Power flux density helpers for ITU-style pass compliance checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_LOWER_BAND_MIN_MHZ = 1525.0
_LOWER_BAND_MAX_MHZ = 2300.0
_X_BAND_MIN_MHZ = 8025.0
_X_BAND_MAX_MHZ = 8500.0
_KA_BAND_MIN_MHZ = 25500.0
_KA_BAND_MAX_MHZ = 27000.0
_REFERENCE_BANDWIDTH_4KHZ_HZ = 4000.0
_REFERENCE_BANDWIDTH_1MHZ_HZ = 1_000_000.0


@dataclass(frozen=True)
class PfdMaskSpec:
    """Frequency-dependent ITU PFD mask definition."""

    low_limit_dB: float
    high_limit_dB: float
    reference_bandwidth_hz: float


def _to_numpy(values: Sequence[float] | float) -> np.ndarray:
    return np.asarray(values, dtype=float)


def occupied_bandwidth_hz(symbol_rate_sps: float, rolloff: float) -> float:
    """Return the occupied bandwidth used for PFD normalization."""
    if not np.isfinite(symbol_rate_sps) or symbol_rate_sps <= 0.0:
        raise ValueError("Symbol rate must be positive for PFD calculation.")
    if not np.isfinite(rolloff) or rolloff < 0.0:
        raise ValueError("Rolloff must be finite and non-negative for PFD calculation.")
    return float(symbol_rate_sps) * (1.0 + float(rolloff))


def directional_eirp_dBW(
    *,
    tx_power_dBw: float,
    tx_losses_dB: float,
    tx_backoff_dB: float,
    antenna_gain_dBi: Sequence[float] | float,
) -> np.ndarray:
    """Return the directional EIRP using power delivered to the antenna input."""
    gains = _to_numpy(antenna_gain_dBi)
    return (
        float(tx_power_dBw)
        - float(tx_losses_dB)
        - float(tx_backoff_dB)
        + gains
    )


def pfd_at_4khz_dBW_per_m2(
    *,
    directional_eirp_dBW: Sequence[float] | float,
    slant_range_m: Sequence[float] | float,
    occupied_bandwidth_hz: float,
) -> np.ndarray:
    """Convert directional EIRP into power flux density normalized to 4 kHz."""
    return pfd_at_reference_bandwidth_dBW_per_m2(
        directional_eirp_dBW=directional_eirp_dBW,
        slant_range_m=slant_range_m,
        occupied_bandwidth_hz=occupied_bandwidth_hz,
        reference_bandwidth_hz=_REFERENCE_BANDWIDTH_4KHZ_HZ,
    )


def pfd_at_reference_bandwidth_dBW_per_m2(
    *,
    directional_eirp_dBW: Sequence[float] | float,
    slant_range_m: Sequence[float] | float,
    occupied_bandwidth_hz: float,
    reference_bandwidth_hz: float,
) -> np.ndarray:
    """Convert directional EIRP into PFD normalized to a reference bandwidth."""
    eirp = _to_numpy(directional_eirp_dBW)
    ranges_m = _to_numpy(slant_range_m)
    if eirp.shape != ranges_m.shape:
        raise ValueError("Directional EIRP and slant range arrays must have matching shapes.")
    if not np.all(np.isfinite(ranges_m)) or np.any(ranges_m <= 0.0):
        raise ValueError("Slant range must be finite and strictly positive for PFD calculation.")
    if not np.isfinite(occupied_bandwidth_hz) or occupied_bandwidth_hz <= 0.0:
        raise ValueError("Occupied bandwidth must be positive for PFD calculation.")
    if not np.isfinite(reference_bandwidth_hz) or reference_bandwidth_hz <= 0.0:
        raise ValueError("Reference bandwidth must be positive for PFD calculation.")
    spreading_loss_dB = 10.0 * np.log10(4.0 * np.pi * np.square(ranges_m))
    bandwidth_normalization_dB = 10.0 * np.log10(
        float(occupied_bandwidth_hz) / float(reference_bandwidth_hz)
    )
    return eirp - spreading_loss_dB - bandwidth_normalization_dB


def itu_surface_pfd_mask_spec(frequency_MHz: float) -> PfdMaskSpec:
    """Return the ITU PFD mask and reference bandwidth for a supported band."""
    freq_mhz = float(frequency_MHz)
    if _LOWER_BAND_MIN_MHZ <= freq_mhz <= _LOWER_BAND_MAX_MHZ:
        return PfdMaskSpec(
            low_limit_dB=-154.0,
            high_limit_dB=-144.0,
            reference_bandwidth_hz=_REFERENCE_BANDWIDTH_4KHZ_HZ,
        )
    if _X_BAND_MIN_MHZ <= freq_mhz <= _X_BAND_MAX_MHZ:
        return PfdMaskSpec(
            low_limit_dB=-150.0,
            high_limit_dB=-140.0,
            reference_bandwidth_hz=_REFERENCE_BANDWIDTH_4KHZ_HZ,
        )
    if _KA_BAND_MIN_MHZ <= freq_mhz <= _KA_BAND_MAX_MHZ:
        return PfdMaskSpec(
            low_limit_dB=-115.0,
            high_limit_dB=-105.0,
            reference_bandwidth_hz=_REFERENCE_BANDWIDTH_1MHZ_HZ,
        )
    raise ValueError(
        "PFD limits are only implemented for 1525-2300 MHz, 8025-8500 MHz, and 25500-27000 MHz."
    )


def itu_surface_pfd_limit_dBW_per_m2_4khz(
    elevation_deg: Sequence[float] | float,
    *,
    frequency_MHz: float,
) -> np.ndarray:
    """Return the ITU surface PFD limit for the supported S- and X-band masks."""
    spec = itu_surface_pfd_mask_spec(frequency_MHz)
    if spec.reference_bandwidth_hz != _REFERENCE_BANDWIDTH_4KHZ_HZ:
        raise ValueError("Requested frequency uses a non-4 kHz PFD mask.")
    return itu_surface_pfd_limit_dBW_per_m2(elevation_deg, frequency_MHz=frequency_MHz)


def itu_surface_pfd_limit_dBW_per_m2(
    elevation_deg: Sequence[float] | float,
    *,
    frequency_MHz: float,
) -> np.ndarray:
    """Return the ITU surface PFD limit for the supported frequency masks."""
    elevations = _to_numpy(elevation_deg)
    if not np.all(np.isfinite(elevations)):
        raise ValueError("Elevation samples must be finite for PFD limit evaluation.")
    if np.any((elevations < 0.0) | (elevations > 90.0)):
        raise ValueError("Elevation samples must lie within 0 to 90 degrees for PFD limits.")
    spec = itu_surface_pfd_mask_spec(frequency_MHz)

    limits = np.empty_like(elevations, dtype=float)
    low_mask = elevations <= 5.0
    mid_mask = (elevations > 5.0) & (elevations < 25.0)
    high_mask = elevations >= 25.0
    limits[low_mask] = spec.low_limit_dB
    limits[mid_mask] = spec.low_limit_dB + 0.5 * (elevations[mid_mask] - 5.0)
    limits[high_mask] = spec.high_limit_dB
    return limits
