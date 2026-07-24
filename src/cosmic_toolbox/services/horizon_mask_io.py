"""Load azimuth-elevation horizon mask profiles.

Handles two common per-station horizon-mask CSV formats:

**Simple format** (e.g. StationA.csv)::

    Azimuth,Horizon
    0,4
    1,3.25
    ...

**Header-block format** (e.g. StationB.csv)::

    # comment lines ...
    Antenna   = IG1
    Latitude  = 68.3
    ...
    Azimuth\tHorizon
    0\t2.11
    1\t2.11
    ...

Both styles must contain exactly 360 rows after the ``Azimuth``/``Horizon``
header, one per integer azimuth degree (0–359).  The returned array is indexed
directly by integer azimuth degree::

    mask = load_horizon_mask("Inuvik.csv")
    min_elev = mask[azimuth_deg_int]          # scalar lookup
    min_elev_arr = np.interp(az_deg, np.arange(360), mask, period=360)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np


def load_horizon_mask(path: str | Path) -> np.ndarray:
    """Return a (360,) float64 array of minimum horizon elevations in degrees.

    Index ``i`` is the minimum elevation (deg) above which a satellite must
    be visible at azimuth ``i`` degrees (measured clockwise from North).

    Raises
    ------
    ValueError
        If the file cannot be parsed or does not contain 360 data rows.
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Horizon mask file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Find the header row containing both "Azimuth" and "Horizon" keywords.
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.search(r"(?i)\bazimuth\b", stripped) and re.search(
            r"(?i)\bhorizon\b", stripped
        ):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Could not find an 'Azimuth'/'Horizon' header line in {path.name}"
        )

    data_lines = lines[header_idx + 1 :]
    elevations: list[float] = []
    for line in data_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept comma, tab, or whitespace as delimiter.
        parts = re.split(r"[,\t\s]+", line, maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            elevations.append(float(parts[1]))
        except ValueError:
            continue

    if len(elevations) != 360:
        raise ValueError(
            f"Expected 360 azimuth rows in {path.name}, got {len(elevations)}"
        )

    return np.array(elevations, dtype=np.float64)


@lru_cache(maxsize=32)
def _cached_load(path_str: str) -> np.ndarray:
    """LRU-cached wrapper so repeated calls within a run don't re-read disk."""
    return load_horizon_mask(path_str)


def horizon_elevations_at_azimuths(
    azimuth_deg: np.ndarray, mask_path: str | Path
) -> np.ndarray:
    """Vectorised lookup: return the horizon elevation (deg) for each azimuth.

    Uses linear interpolation with 360° wrap-around so fractional azimuths
    are handled smoothly.

    Parameters
    ----------
    azimuth_deg:
        1-D array of azimuths in degrees (any range, will be wrapped to 0–360).
    mask_path:
        Path to the horizon mask CSV file.

    Returns
    -------
    np.ndarray
        Same shape as *azimuth_deg*, values in degrees.
    """
    mask = _cached_load(str(Path(mask_path).expanduser().resolve()))
    az = np.asarray(azimuth_deg, dtype=np.float64) % 360.0
    indices = np.arange(360, dtype=np.float64)
    # np.interp does NOT support period= in all NumPy versions; handle wrap
    # manually by appending the first point at 360° for continuity.
    return np.interp(az, np.append(indices, 360.0), np.append(mask, mask[0]))


def builtin_mask_path(station_name: str) -> Path | None:
    """Return a bundled horizon mask path for a known station name, if any.

    Tries a case-insensitive prefix/substring match against the resource
    directory.  Returns ``None`` if no match is found (no masks are bundled
    by default — drop your own CSVs into ``resources/horizon_masks/`` to use
    this lookup).
    """
    resources = Path(__file__).resolve().parents[1] / "resources" / "horizon_masks"
    if not resources.is_dir():
        return None
    needle = station_name.strip().lower()
    for csv_file in resources.glob("*.csv"):
        if needle in csv_file.stem.lower():
            return csv_file
    return None
