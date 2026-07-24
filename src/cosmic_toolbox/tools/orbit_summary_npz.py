from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OrbitSummaryNPZ:
    time_seconds: np.ndarray
    time_hours: np.ndarray
    instantaneous: dict[str, np.ndarray]
    orbit_averaged: dict[str, np.ndarray]
    meta: dict[str, Any] | None


def _as_1d_float(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr.reshape((-1,))


def save_orbit_summary_npz(
    path: Path,
    *,
    time_seconds: np.ndarray,
    time_hours: np.ndarray | None = None,
    instantaneous: dict[str, np.ndarray] | None = None,
    orbit_averaged: dict[str, np.ndarray] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save an Orbit Summary NPZ with flattened keys.

    Stored keys:
    - time_seconds, time_hours
    - instantaneous__<name> for each instantaneous series
    - orbit_averaged__<name> for each orbit-averaged series
    - meta_json (optional) as a JSON string
    """
    path = Path(path)
    inst = instantaneous or {}
    avg = orbit_averaged or {}

    t_s = _as_1d_float(time_seconds)
    t_h = _as_1d_float(time_hours) if time_hours is not None else (t_s / 3600.0)

    payload: dict[str, Any] = {
        "time_seconds": t_s,
        "time_hours": t_h,
    }
    for k, v in inst.items():
        payload[f"instantaneous__{k}"] = _as_1d_float(v)
    for k, v in avg.items():
        payload[f"orbit_averaged__{k}"] = _as_1d_float(v)
    if meta is not None:
        payload["meta_json"] = np.array([json.dumps(meta)], dtype=object)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_orbit_summary_npz(path: Path) -> OrbitSummaryNPZ:
    """Load an Orbit Summary NPZ written by save_orbit_summary_npz()."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    def _get_time(name: str) -> np.ndarray:
        if name in data:
            return _as_1d_float(data[name])
        raise KeyError(f"Missing required key in NPZ: {name}")

    t_s = _get_time("time_seconds")
    t_h = _as_1d_float(data["time_hours"]) if "time_hours" in data else (t_s / 3600.0)

    instantaneous: dict[str, np.ndarray] = {}
    orbit_averaged: dict[str, np.ndarray] = {}
    for k in data.files:
        if k.startswith("instantaneous__"):
            instantaneous[k.split("__", 1)[1]] = _as_1d_float(data[k])
        elif k.startswith("orbit_averaged__"):
            orbit_averaged[k.split("__", 1)[1]] = _as_1d_float(data[k])

    meta: dict[str, Any] | None = None
    if "meta_json" in data:
        raw = data["meta_json"]
        try:
            s = raw.item() if hasattr(raw, "item") else str(raw)
            meta = json.loads(s)
        except Exception as exc:
            raise ValueError("Invalid meta_json in NPZ (failed to parse JSON)") from exc

    return OrbitSummaryNPZ(
        time_seconds=t_s,
        time_hours=t_h,
        instantaneous=instantaneous,
        orbit_averaged=orbit_averaged,
        meta=meta,
    )

