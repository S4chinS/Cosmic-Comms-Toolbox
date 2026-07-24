"""Filesystem helpers for locating bundled resources and analysis outputs."""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from pathlib import Path

# One canonical run-directory timestamp format for the whole repo.  Scripts
# must use run_dir()/outputs_subdir() rather than inventing their own scheme so
# outputs sort consistently and land in one place.
RUN_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def package_resources_root() -> Path:
    """Return a filesystem path to bundled ``cosmic_toolbox.resources``.

    Falls back to the in-tree location when running from an editable
    install or directly from source.
    """

    try:
        ref = resources.files("cosmic_toolbox.resources")
        p = Path(str(ref))
        if p.exists():
            return p
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    return Path(__file__).resolve().parent / "resources"


def repo_root() -> Path:
    """Repository root (parent of ``src/``) for scripts that still need it."""

    # cosmic_toolbox/paths.py -> cosmic_toolbox -> src -> repo
    return Path(__file__).resolve().parents[2]


def outputs_root() -> Path:
    """Canonical ``<repo>/outputs`` directory, created on first use.

    Every script writes here (directly or via :func:`run_dir` /
    :func:`outputs_subdir`) instead of re-deriving its own path or using a
    CWD-relative ``"outputs"`` string that only works from the repo root.
    """

    root = repo_root() / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize(token: str) -> str:
    """Make a filename-safe token (lowercase, no spaces/separators)."""

    safe = "".join(c if c.isalnum() else "_" for c in token.strip().lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def run_dir(
    name: str,
    *,
    timestamp: datetime | None = None,
    suffix: str | None = None,
) -> Path:
    """Create and return a timestamped run directory under ``outputs/``.

    Produces ``outputs/<name>_<YYYYMMDD_HHMMSS>[_<suffix>]/`` so every run's
    artifacts (plots, CSVs, NPZs, reports) stay grouped together with a single,
    sortable timestamp format.

    Args:
        name: Analysis name, e.g. ``"annual_contact"``.
        timestamp: Run time; defaults to now.
        suffix: Optional config tag, e.g. ``"600km_2msym"``.
    """

    ts = (timestamp or datetime.now()).strftime(RUN_TIMESTAMP_FMT)
    parts = [_sanitize(name), ts]
    if suffix:
        parts.append(_sanitize(suffix))
    directory = outputs_root() / "_".join(parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def outputs_subdir(category: str) -> Path:
    """Return (and create) a stable categorized folder under ``outputs/``.

    For loose, non-run artifacts that should not scatter across the outputs
    root, e.g. ``outputs_subdir("plots")`` or ``outputs_subdir("cache")``.
    """

    directory = outputs_root() / _sanitize(category)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
