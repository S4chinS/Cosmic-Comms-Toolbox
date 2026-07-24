"""Utilities for loading ground-station definitions from CSV/Excel files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd

from cosmic_toolbox.models import GroundStationConfig

REQUIRED_COLUMNS = ["name", "latitude", "longitude", "altitude"]

_DB_PATH = Path(__file__).parents[1] / "resources" / "ground_station_database.csv"


@dataclass
class StationImportError(Exception):
    """Raised when the ground-station file cannot be parsed."""

    message: str

    def __str__(self) -> str:
        return self.message


@lru_cache(maxsize=1)
def _load_database() -> dict[str, dict]:
    """Load ground_station_database.csv keyed by name.  LRU-cached for the process lifetime."""
    try:
        df = pd.read_csv(_DB_PATH, keep_default_na=False)
    except FileNotFoundError:
        return {}
    return {
        str(row["name"]).strip(): row.to_dict()
        for _, row in df.iterrows()
    }


def _resolve_mask(db_row: dict) -> str | None:
    """Convert a relative horizon_mask value from the database to an absolute path."""
    rel = str(db_row.get("horizon_mask", "")).strip()
    if not rel:
        return None
    abs_path = _DB_PATH.parent / rel
    return str(abs_path) if abs_path.exists() else None


def available_station_names() -> List[str]:
    """Names of every station in the master ground-station database."""

    return sorted(_load_database().keys())


def load_station_by_name(name: str) -> GroundStationConfig:
    """Return the canonical :class:`GroundStationConfig` for a database name.

    Single source of truth for station coordinates — scripts should call this
    instead of hardcoding lat/lon/alt, so a site is defined in exactly one
    place.  Raises :class:`StationImportError` if the name is unknown.
    """

    db = _load_database()
    row = db.get(name.strip())
    if row is None:
        raise StationImportError(
            f"Unknown ground station {name!r}. "
            f"Known stations: {', '.join(available_station_names())}"
        )
    return GroundStationConfig(
        name=str(row["name"]).strip(),
        latitude_deg=float(row["latitude"]),
        longitude_deg=float(row["longitude"]),
        altitude_m=float(row["altitude"]),
        supplier=str(row.get("supplier", "")).strip(),
        horizon_mask_path=_resolve_mask(row),
    )


def load_ground_stations_from_file(path: str | Path) -> List[GroundStationConfig]:
    """Load stations from a CSV or Excel file.

    Supports two CSV formats:

    * **Name-only** — a single ``name`` column; each entry is resolved against
      the master ``ground_station_database.csv``, which supplies lat/lon/alt,
      supplier, and horizon mask automatically.

    * **Full** — ``name, latitude, longitude, altitude`` columns (original
      behaviour).  Supplier and horizon mask are enriched from the database
      when the station name matches an entry.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise StationImportError(f"File not found: {file_path}")

    df = _read_dataframe(file_path)
    cols_lower = {col.lower() for col in df.columns}

    if "name" in cols_lower and not (cols_lower >= {"latitude", "longitude", "altitude"}):
        return _load_name_only(df)
    return _load_full_format(df)


def _load_name_only(df: pd.DataFrame) -> List[GroundStationConfig]:
    """Resolve a name-only CSV against the master database."""
    name_col = next(c for c in df.columns if c.lower() == "name")
    db = _load_database()
    stations: List[GroundStationConfig] = []
    for i, raw_name in enumerate(df[name_col]):
        name = str(raw_name).strip()
        if not name:
            continue
        if name not in db:
            raise StationImportError(
                f"Row {i + 1}: '{name}' not found in the ground station database."
            )
        row = db[name]
        try:
            stations.append(
                GroundStationConfig(
                    name=name,
                    latitude_deg=float(row["latitude"]),
                    longitude_deg=float(row["longitude"]),
                    altitude_m=float(row["altitude"]),
                    supplier=str(row.get("supplier", "")).strip(),
                    horizon_mask_path=_resolve_mask(row),
                )
            )
        except (TypeError, ValueError) as exc:
            raise StationImportError(
                f"Database entry for '{name}' has invalid numeric data: {exc}"
            ) from exc

    if not stations:
        raise StationImportError("No rows were found in the provided file.")
    return stations


def _load_full_format(df: pd.DataFrame) -> List[GroundStationConfig]:
    """Load a full-format CSV (name, latitude, longitude, altitude).

    Supplier and horizon mask are auto-populated from the database when the
    station name exactly matches a database entry.
    """
    normalized_columns = {col.lower(): col for col in df.columns}
    missing = [col for col in REQUIRED_COLUMNS if col not in normalized_columns]
    if missing:
        raise StationImportError(
            f"Missing required columns: {', '.join(missing)}. "
            "Expected columns: name, latitude, longitude, altitude."
        )

    db = _load_database()
    stations: List[GroundStationConfig] = []
    for i, row in df.iterrows():
        try:
            name = str(row[normalized_columns["name"]]).strip()
            db_row = db.get(name, {})
            stations.append(
                GroundStationConfig(
                    name=name,
                    latitude_deg=float(row[normalized_columns["latitude"]]),
                    longitude_deg=float(row[normalized_columns["longitude"]]),
                    altitude_m=float(row[normalized_columns["altitude"]]),
                    supplier=str(db_row.get("supplier", "")).strip(),
                    horizon_mask_path=_resolve_mask(db_row) if db_row else None,
                )
            )
        except (TypeError, ValueError) as exc:
            raise StationImportError(
                f"Invalid numeric value in row {i + 1}: {exc}"
            ) from exc

    if not stations:
        raise StationImportError("No rows were found in the provided file.")
    return stations


def _read_dataframe(file_path: Path) -> pd.DataFrame:
    """Return a pandas DataFrame from CSV or Excel input."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, keep_default_na=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise StationImportError(
        f"Unsupported file type '{suffix}'. Please select CSV or Excel."
    )
