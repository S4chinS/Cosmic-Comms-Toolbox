"""Canonical link-budget defaults, loaded from one bundled YAML file.

``resources/link_budget_defaults.yaml`` is the single source of truth for
RF / link-budget parameter defaults across the core, GUI, web app, and the
analysis scripts.  Consumers call :func:`load_link_budget_defaults` (cached)
and read attributes; anything that needs a different value overrides it
explicitly at its own call site so the deviation stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

import yaml

from cosmic_toolbox.paths import package_resources_root


@dataclass(frozen=True)
class LinkBudgetDefaults:
    """Typed view of ``resources/link_budget_defaults.yaml``."""

    # Carrier
    frequency_GHz: float
    uplink_frequency_MHz: float
    # Transmitter
    tx_power_dBw: float
    tx_losses_dB: float
    tx_backoff_dB: float
    tx_boresight_gain_dBi: float
    # Receiver — fixed G/T or three-parameter breakdown
    gs_gt_dBK: float
    rx_antenna_gain_dBi: float
    receiver_noise_figure_dB: float
    sky_background_temperature_K: float
    receiver_losses_dB: float
    # Channel / waveform
    symbol_rate_Msps: float
    rolloff: float
    operating_mode: str
    fixed_modcod_name: str
    # Losses and margin
    polarization_loss_dB: float
    implementation_loss_dB: float
    margin_dB: float
    unavailability_percent: float
    # Geometry
    satellite_altitude_km: float
    min_elevation_deg: float

    @property
    def symbol_rate_sps(self) -> float:
        return float(self.symbol_rate_Msps) * 1e6

    @property
    def frequency_MHz(self) -> float:
        return float(self.frequency_GHz) * 1000.0


def default_defaults_path() -> Path:
    """Path of the bundled canonical defaults YAML."""

    return package_resources_root() / "link_budget_defaults.yaml"


def _load(path: Path) -> LinkBudgetDefaults:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Link-budget defaults file is not a mapping: {path}")
    known = {f.name for f in fields(LinkBudgetDefaults)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in link-budget defaults file {path}: {sorted(unknown)}"
        )
    missing = known - set(raw)
    if missing:
        raise ValueError(
            f"Missing keys in link-budget defaults file {path}: {sorted(missing)}"
        )
    return LinkBudgetDefaults(**raw)


@lru_cache(maxsize=None)
def _load_cached(resolved: str) -> LinkBudgetDefaults:
    return _load(Path(resolved))


def load_link_budget_defaults(path: Path | str | None = None) -> LinkBudgetDefaults:
    """Load the canonical defaults (cached per file path).

    Args:
        path: Optional alternative YAML with the same schema; defaults to the
            bundled ``resources/link_budget_defaults.yaml``.
    """

    target = Path(path) if path is not None else default_defaults_path()
    return _load_cached(str(target.resolve()))
