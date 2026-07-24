"""Import/export helpers for cached trajectory scenario packages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from cosmic_toolbox.models import PropagatedEphemeris


SCENARIO_PACKAGE_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1


@dataclass
class ImportedScenarioPackage:
    """Parsed cached-trajectory scenario package."""

    ephemeris: PropagatedEphemeris
    settings: dict[str, Any]
    stations: list[dict[str, Any]]
    enabled_station_names: list[str]
    npz_path: Path
    sidecar_path: Path


# ---------------------------------------------------------------------------
# Shared datetime helpers
# ---------------------------------------------------------------------------

def _isoformat_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _parse_utc_datetime(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty UTC datetime string.")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# NPZ write / read  (schema version 2)
# ---------------------------------------------------------------------------

def _sidecar_path_for_npz(npz_path: Path) -> Path:
    return npz_path.with_suffix(".toolbox.json")


def _write_ephemeris_npz(ephemeris: PropagatedEphemeris, output_path: Path) -> None:
    """Serialize a PropagatedEphemeris to a compressed NPZ file."""
    if ephemeris.ecef_pos_km is None or ephemeris.timestamps_unix is None:
        raise ValueError("Cannot export NPZ: ephemeris numpy arrays are not populated.")

    def _arr(seq: np.ndarray) -> np.ndarray:
        a = np.asarray(seq, dtype=np.float64)
        return a if a.size else np.empty(0, dtype=np.float64)

    n = ephemeris.ecef_pos_km.shape[0]
    body_x = ephemeris.body_x_ecef if ephemeris.body_x_ecef is not None else np.full((n, 3), np.nan)
    body_y = ephemeris.body_y_ecef if ephemeris.body_y_ecef is not None else np.full((n, 3), np.nan)
    body_z = ephemeris.body_z_ecef if ephemeris.body_z_ecef is not None else np.full((n, 3), np.nan)
    gt_lat = ephemeris.gt_lat_deg if ephemeris.gt_lat_deg is not None else np.full(n, np.nan)
    gt_lon = ephemeris.gt_lon_deg if ephemeris.gt_lon_deg is not None else np.full(n, np.nan)
    eci_pos = ephemeris.eci_pos_km if ephemeris.eci_pos_km is not None else np.full((n, 3), np.nan)
    eci_vel = ephemeris.eci_vel_km_s if ephemeris.eci_vel_km_s is not None else np.full((n, 3), np.nan)
    ecef_vel = ephemeris.ecef_vel_km_s if ephemeris.ecef_vel_km_s is not None else np.full((n, 3), np.nan)

    # timeline_s is omitted — it is exactly derivable as
    # ``timestamps_unix - timestamps_unix[0]`` with no accuracy loss.
    # The reader re-derives it with a backward-compat fallback when loading
    # older files that stored it explicitly.
    np.savez_compressed(
        output_path,
        timestamps_unix=ephemeris.timestamps_unix,
        eci_pos_km=eci_pos,
        eci_vel_km_s=eci_vel,
        ecef_pos_km=ephemeris.ecef_pos_km,
        ecef_vel_km_s=ecef_vel,
        body_x_ecef=body_x,
        body_y_ecef=body_y,
        body_z_ecef=body_z,
        gt_lat_deg=gt_lat,
        gt_lon_deg=gt_lon,
        orbit_period_seconds=np.array([ephemeris.orbit_period_seconds], dtype=np.float64),
        altitude_km=_arr(ephemeris.orbital_altitude_km),
        semi_major_axis_km=_arr(ephemeris.semi_major_axis_km),
        perigee_altitude_km=_arr(ephemeris.perigee_altitude_km),
        apogee_altitude_km=_arr(ephemeris.apogee_altitude_km),
        eccentricity=_arr(ephemeris.eccentricity),
        inclination_deg=_arr(ephemeris.inclination_deg),
        argument_of_perigee_deg=_arr(ephemeris.argument_of_perigee_deg),
        orbital_period_s=_arr(ephemeris.orbital_period_series_s),
        true_anomaly_deg=_arr(ephemeris.true_anomaly_deg),
        angle_of_attack_deg=_arr(ephemeris.angle_of_attack_deg),
    )


def _read_ephemeris_npz(
    npz_path: Path,
    orbit_period_seconds: float,
    mean_ic_report: str | None,
) -> PropagatedEphemeris:
    """Deserialize a PropagatedEphemeris from a compressed NPZ file."""
    with np.load(npz_path) as data:
        if "orbit_period_seconds" in data:
            orbit_period_seconds = float(data["orbit_period_seconds"][0])

        # Support both old key name (eci_vel_km_s) and legacy (eci_vel_km_s was written as eci_vel_km_s)
        # The old writer used "eci_vel_km_s" but some files may have "eci_vel_km_s" from the old format.
        timestamps_unix = data["timestamps_unix"].copy()
        eci_pos_km = data["eci_pos_km"].copy()
        eci_vel_km_s = data["eci_vel_km_s"].copy()
        ecef_pos_km = data["ecef_pos_km"].copy()
        ecef_vel_km_s = data["ecef_vel_km_s"].copy()
        body_x_ecef = data["body_x_ecef"].copy()
        body_y_ecef = data["body_y_ecef"].copy()
        body_z_ecef = data["body_z_ecef"].copy()
        gt_lat_deg = data["gt_lat_deg"].copy()
        gt_lon_deg = data["gt_lon_deg"].copy()
        # timeline_s is no longer stored (new files) — derive it exactly from
        # timestamps_unix.  Old files also carry timestamps_unix, so this path
        # is unconditional and correct for both old and new files.
        timeline = timestamps_unix - timestamps_unix[0]
        altitude = data["altitude_km"].copy()
        sma = data["semi_major_axis_km"].copy()
        perigee = data["perigee_altitude_km"].copy()
        apogee = data["apogee_altitude_km"].copy()
        ecc = data["eccentricity"].copy()
        inc = data["inclination_deg"].copy()
        aop = data["argument_of_perigee_deg"].copy()
        orb_per = data["orbital_period_s"].copy()
        true_anom = data["true_anomaly_deg"].copy()
        aoa = data["angle_of_attack_deg"].copy()

        return PropagatedEphemeris(
            timeline_seconds=timeline,
            orbit_period_seconds=orbit_period_seconds,
            mean_ic_report=mean_ic_report,
            timestamps_unix=timestamps_unix,
            eci_pos_km=eci_pos_km,
            eci_vel_km_s=eci_vel_km_s,
            ecef_pos_km=ecef_pos_km,
            ecef_vel_km_s=ecef_vel_km_s,
            body_x_ecef=body_x_ecef,
            body_y_ecef=body_y_ecef,
            body_z_ecef=body_z_ecef,
            gt_lat_deg=gt_lat_deg,
            gt_lon_deg=gt_lon_deg,
            orbital_altitude_km=altitude,
            semi_major_axis_km=sma,
            perigee_altitude_km=perigee,
            apogee_altitude_km=apogee,
            eccentricity=ecc,
            inclination_deg=inc,
            argument_of_perigee_deg=aop,
            orbital_period_series_s=orb_per,
            true_anomaly_deg=true_anom,
            angle_of_attack_deg=aoa,
        )


# ---------------------------------------------------------------------------
# Legacy deserialization helpers (schema version 1 — read-only)
# ---------------------------------------------------------------------------

def _legacy_deserialize_ephemeris(payload: dict[str, Any]) -> PropagatedEphemeris:
    """Build a PropagatedEphemeris from a v1 JSON payload using columnar arrays."""
    sv_items: list[dict[str, Any]] = payload.get("state_vectors", [])
    gt_items: list[dict[str, Any]] = payload.get("ground_track", [])
    n = len(sv_items)

    if n > 0:
        timestamps_unix = np.array(
            [_parse_utc_datetime(item["timestamp_utc"]).timestamp() for item in sv_items],
            dtype=np.float64,
        )
        eci_pos_km = np.array(
            [[item["eci_x_km"], item["eci_y_km"], item["eci_z_km"]] for item in sv_items],
            dtype=np.float64,
        )
        eci_vel_km_s = np.array(
            [[item["eci_vx_km_s"], item["eci_vy_km_s"], item["eci_vz_km_s"]] for item in sv_items],
            dtype=np.float64,
        )
        ecef_pos_km = np.array(
            [[item["ecef_x_km"], item["ecef_y_km"], item["ecef_z_km"]] for item in sv_items],
            dtype=np.float64,
        )
        ecef_vel_km_s = np.array(
            [[item["ecef_vx_km_s"], item["ecef_vy_km_s"], item["ecef_vz_km_s"]] for item in sv_items],
            dtype=np.float64,
        )
        body_x_ecef = np.array(
            [[item["body_x_ecef_x"], item["body_x_ecef_y"], item["body_x_ecef_z"]] for item in sv_items],
            dtype=np.float64,
        )
        body_y_ecef = np.array(
            [[item["body_y_ecef_x"], item["body_y_ecef_y"], item["body_y_ecef_z"]] for item in sv_items],
            dtype=np.float64,
        )
        body_z_ecef = np.array(
            [[item["body_z_ecef_x"], item["body_z_ecef_y"], item["body_z_ecef_z"]] for item in sv_items],
            dtype=np.float64,
        )
    else:
        timestamps_unix = None
        eci_pos_km = eci_vel_km_s = ecef_pos_km = ecef_vel_km_s = None
        body_x_ecef = body_y_ecef = body_z_ecef = None

    if gt_items:
        gt_lat_deg = np.array([float(item["latitude_deg"]) for item in gt_items], dtype=np.float64)
        gt_lon_deg = np.array([float(item["longitude_deg"]) for item in gt_items], dtype=np.float64)
    else:
        gt_lat_deg = gt_lon_deg = None

    return PropagatedEphemeris(
        timeline_seconds=[float(v) for v in payload["timeline_seconds"]],
        orbit_period_seconds=float(payload["orbit_period_seconds"]),
        mean_ic_report=payload.get("mean_ic_report"),
        timestamps_unix=timestamps_unix,
        eci_pos_km=eci_pos_km,
        eci_vel_km_s=eci_vel_km_s,
        ecef_pos_km=ecef_pos_km,
        ecef_vel_km_s=ecef_vel_km_s,
        body_x_ecef=body_x_ecef,
        body_y_ecef=body_y_ecef,
        body_z_ecef=body_z_ecef,
        gt_lat_deg=gt_lat_deg,
        gt_lon_deg=gt_lon_deg,
        orbital_altitude_km=[float(v) for v in payload.get("orbital_altitude_km", [])],
        semi_major_axis_km=[float(v) for v in payload.get("semi_major_axis_km", [])],
        perigee_altitude_km=[float(v) for v in payload.get("perigee_altitude_km", [])],
        apogee_altitude_km=[float(v) for v in payload.get("apogee_altitude_km", [])],
        eccentricity=[float(v) for v in payload.get("eccentricity", [])],
        inclination_deg=[float(v) for v in payload.get("inclination_deg", [])],
        argument_of_perigee_deg=[float(v) for v in payload.get("argument_of_perigee_deg", [])],
        orbital_period_series_s=[float(v) for v in payload.get("orbital_period_series_s", [])],
        true_anomaly_deg=[float(v) for v in payload.get("true_anomaly_deg", [])],
        angle_of_attack_deg=[float(v) for v in payload.get("angle_of_attack_deg", [])],
    )


def _legacy_import(sidecar_path: Path, oem_path: Path) -> ImportedScenarioPackage:
    """Load a schema-version-1 package (OEM + fat JSON sidecar)."""
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    settings = payload.get("settings")
    stations = payload.get("stations")
    enabled_station_names = payload.get("enabled_station_names")
    ephemeris_payload = payload.get("ephemeris")
    if not isinstance(settings, dict):
        raise ValueError("Scenario package is missing a valid settings payload.")
    if not isinstance(stations, list):
        raise ValueError("Scenario package is missing its station list.")
    if not isinstance(enabled_station_names, list):
        raise ValueError("Scenario package is missing enabled_station_names.")
    if not isinstance(ephemeris_payload, dict):
        raise ValueError("Scenario package is missing its ephemeris payload.")
    ephemeris = _legacy_deserialize_ephemeris(ephemeris_payload)
    return ImportedScenarioPackage(
        ephemeris=ephemeris,
        settings=settings,
        stations=stations,
        enabled_station_names=[str(name) for name in enabled_station_names],
        npz_path=oem_path,
        sidecar_path=sidecar_path,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_cached_trajectory_package(
    *,
    ephemeris: PropagatedEphemeris,
    settings: dict[str, Any],
    stations: list[dict[str, Any]],
    enabled_station_names: list[str],
    output_path: Path,
) -> tuple[Path, Path]:
    """Write a compressed NPZ ephemeris file plus a lean JSON sidecar.

    Returns ``(npz_path, sidecar_path)``.
    """
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = _sidecar_path_for_npz(output_path)

    _write_ephemeris_npz(ephemeris, output_path)

    lean_sidecar = {
        "schema_version": SCENARIO_PACKAGE_SCHEMA_VERSION,
        "orbit_period_seconds": float(ephemeris.orbit_period_seconds),
        "mean_ic_report": ephemeris.mean_ic_report,
        "settings": settings,
        "stations": stations,
        "enabled_station_names": list(enabled_station_names),
    }
    sidecar_path.write_text(json.dumps(lean_sidecar, indent=2), encoding="utf-8")
    return output_path, sidecar_path


def import_cached_trajectory_package(path: str | Path) -> ImportedScenarioPackage:
    """Load a cached trajectory package.

    Accepts a ``.npz`` file, a ``.toolbox.json`` sidecar, or (for legacy
    packages) a ``.oem`` file.  Schema-version-1 packages (OEM + fat JSON
    sidecar) are still supported for backward compatibility.
    """
    import_path = Path(path).expanduser().resolve()

    # --- resolve npz / oem / sidecar paths from whatever the user selected ---
    if import_path.suffix.lower() == ".npz":
        npz_path = import_path
        sidecar_path = _sidecar_path_for_npz(npz_path)
        oem_path = None
    elif import_path.suffix.lower() == ".oem":
        oem_path = import_path
        sidecar_path = import_path.with_suffix(".toolbox.json")
        npz_path = import_path.with_suffix(".npz")
    elif import_path.name.endswith(".toolbox.json"):
        sidecar_path = import_path
        stem = import_path.name[: -len(".toolbox.json")]
        npz_path = import_path.with_name(stem + ".npz")
        oem_path = import_path.with_name(stem + ".oem")
    else:
        raise ValueError(
            "Select a .npz trajectory file, its matching .toolbox.json sidecar, "
            "or a legacy .oem file."
        )

    if not sidecar_path.exists():
        raise FileNotFoundError(f"Scenario sidecar not found: {sidecar_path}")

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", -1))

    # --- legacy path: schema version 1 (OEM + fat JSON) ---
    if schema_version == _LEGACY_SCHEMA_VERSION:
        resolved_oem = oem_path if oem_path is not None else npz_path.with_suffix(".oem")
        if not resolved_oem.exists():
            raise FileNotFoundError(f"Legacy OEM file not found: {resolved_oem}")
        return _legacy_import(sidecar_path, resolved_oem)

    # --- current path: schema version 2 (NPZ + lean JSON) ---
    if schema_version != SCENARIO_PACKAGE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported scenario package schema version: {schema_version!r}"
        )

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ trajectory file not found: {npz_path}")

    settings = payload.get("settings")
    stations = payload.get("stations")
    enabled_station_names = payload.get("enabled_station_names")
    if not isinstance(settings, dict):
        raise ValueError("Scenario package is missing a valid settings payload.")
    if not isinstance(stations, list):
        raise ValueError("Scenario package is missing its station list.")
    if not isinstance(enabled_station_names, list):
        raise ValueError("Scenario package is missing enabled_station_names.")

    ephemeris = _read_ephemeris_npz(
        npz_path,
        orbit_period_seconds=float(payload.get("orbit_period_seconds", 0.0)),
        mean_ic_report=payload.get("mean_ic_report"),
    )

    return ImportedScenarioPackage(
        ephemeris=ephemeris,
        settings=settings,
        stations=stations,
        enabled_station_names=[str(name) for name in enabled_station_names],
        npz_path=npz_path,
        sidecar_path=sidecar_path,
    )
