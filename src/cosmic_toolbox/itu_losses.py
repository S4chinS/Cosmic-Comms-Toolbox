"""
Thin wrapper around itu-rpy for atmospheric attenuation estimation.

Inputs are intentionally primitive so the UI can pass textbox/dropdown values
directly without intermediate objects.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from astropy import units as u
import warnings

try:
    import itur
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'itur' package is required for slant-path attenuation calculations. "
        "Install it via `pip install itu-rpy`."
    ) from exc

warnings.filterwarnings(
    "ignore",
    message=(
        "The approximated method to compute the gaseous attenuation "
        "in recommendation ITU-P 676-11 is only recommended for elevation "
        "angles between 5 and 90 degrees"
    ),
    category=RuntimeWarning,
    module=r"itur\..*",
)
# np.where evaluates both branches before selecting, so h**b can overflow,
# and sqrt/power can produce invalid values for low elevations even when the
# affected branch is not the one ultimately used. These are spurious evaluation
# artefacts inside the itur library; the selected results are correct.
warnings.filterwarnings(
    "ignore",
    message=r"overflow encountered in (scalar )?power",
    category=RuntimeWarning,
    module=r"itur\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in (sqrt|power)",
    category=RuntimeWarning,
    module=r"itur\..*",
)

# Minimum elevation passed to the ITU-R models.  The scintillation formula in
# ITU-R P.618 divides by sin(el)**1.2, which diverges to infinity as el -> 0.
# Values below this threshold are clamped; the ITU models are not physically
# meaningful below the horizon anyway.
_ITU_MIN_ELEVATION_DEG = 0.5


def estimate_slant_path_loss(
    frequency_GHz: float,
    elevations_deg: Sequence[float],
    lat_deg: float,
    lon_deg: float,
    altitude_m: float,
    unavailability_percent: float = 0.1,
    antenna_diameter_m: float = 1.0,
    return_contributions: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]]:
    """
    Use ITU-R models to estimate atmospheric attenuation vs elevation angle.

    Args:
        frequency_GHz: RF frequency.
        elevations_deg: Elevation angles in degrees.
        lat_deg / lon_deg: Ground-station coordinates in degrees.
        altitude_m: Ground-station altitude above sea level (meters).
        unavailability_percent: Link unavailability target (default 0.1%).
        antenna_diameter_m: Antenna diameter assumption for rain-scatter calcs.
    """
    elevations = np.asarray(elevations_deg, dtype=float)
    if elevations.ndim != 1:
        raise ValueError("Elevation input must be one-dimensional.")
    if frequency_GHz <= 0:
        raise ValueError("Frequency must be positive.")
    # Clamp to the ITU minimum: the scintillation and rain models in P.618
    # divide by sin(el)**n, so el <= 0 produces NaN/inf.  Sub-horizon values
    # have no physical meaning for a slant-path link budget.
    elevations = np.maximum(elevations, _ITU_MIN_ELEVATION_DEG)

    kwargs = {
        "lat": lat_deg,
        "lon": lon_deg,
        "f": frequency_GHz * u.GHz,
        "el": elevations * u.deg,
        "p": unavailability_percent,
        "D": antenna_diameter_m * u.m,
    }
    if return_contributions:
        kwargs["return_contributions"] = True

    def _call(**call_kwargs):
        try:
            return itur.atmospheric_attenuation_slant_path(
                **call_kwargs, h_station=altitude_m * u.m
            )
        except TypeError as exc:  # pragma: no cover - depends on installed itu-rpy
            if "h_station" in str(exc):
                return itur.atmospheric_attenuation_slant_path(**call_kwargs)
            raise

    contributions: Optional[Dict[str, np.ndarray]] = None
    use_contributions = return_contributions
    result = None

    if use_contributions:
        try:
            result = _call(**kwargs)
        except TypeError as exc:  # pragma: no cover - depends on installed itu-rpy
            if "return_contributions" in str(exc):
                kwargs.pop("return_contributions", None)
                use_contributions = False
                result = _call(**kwargs)
            else:
                raise
    else:
        result = _call(**kwargs)

    if use_contributions and isinstance(result, tuple):
        a_g, a_c, a_r, a_s, a_t = result
        contributions = {
            "gaseous": np.atleast_1d(np.asarray(a_g.value, dtype=float)),
            "cloud": np.atleast_1d(np.asarray(a_c.value, dtype=float)),
            "rain": np.atleast_1d(np.asarray(a_r.value, dtype=float)),
            "scintillation": np.atleast_1d(np.asarray(a_s.value, dtype=float)),
            "total": np.atleast_1d(np.asarray(a_t.value, dtype=float)),
        }
        losses_value = contributions["total"]
    else:
        # result is already the total attenuation quantity
        losses_value = np.atleast_1d(np.asarray(result.value, dtype=float))

    if not return_contributions:
        return losses_value
    return losses_value, contributions
