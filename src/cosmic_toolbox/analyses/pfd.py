"""PFD compliance helpers (toolbox; no Qt)."""

from __future__ import annotations

import numpy as np

from cosmic_toolbox import link_budget_math
from cosmic_toolbox import pfd_math


def compliance_backoff_db(
    *,
    elevations_deg: np.ndarray,
    sat_altitude_km: float,
    ground_altitude_m: float,
    tx_power_dBw: float,
    tx_losses_dB: float,
    antenna_gain_dBi: float,
    frequency_GHz: float,
    symbol_rate_sps: float,
    rolloff: float,
) -> float:
    """TX back-off (dB) so worst-case surface PFD meets the ITU mask (same as link-budget tab)."""

    freq_mhz = frequency_GHz * 1000.0
    spec = pfd_math.itu_surface_pfd_mask_spec(freq_mhz)
    occ_bw_hz = symbol_rate_sps * (1.0 + rolloff)
    bw_norm_dB = 10.0 * np.log10(occ_bw_hz / spec.reference_bandwidth_hz)
    slant_km = link_budget_math.slant_range_km(
        elevations_deg, ground_altitude_m / 1000.0, sat_altitude_km
    )
    slant_m = slant_km * 1000.0
    spreading_dB = 10.0 * np.log10(4.0 * np.pi * slant_m**2)
    eirp_zero = tx_power_dBw - tx_losses_dB + antenna_gain_dBi
    pfd_zero = eirp_zero - spreading_dB - bw_norm_dB
    pfd_limit = pfd_math.itu_surface_pfd_limit_dBW_per_m2(
        elevations_deg, frequency_MHz=freq_mhz
    )
    required = float(np.max(pfd_zero - pfd_limit))
    return max(0.0, required)


__all__ = ["compliance_backoff_db"]
