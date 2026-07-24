"""Progress reporting abstraction for long-running analysis (GUI-agnostic)."""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class ProgressReporter(Protocol):
    """Report normalized progress in ``[0, 1]`` and an optional status message."""

    def report(self, fraction: float, message: str | None = None) -> None:
        """``fraction`` is in ``[0, 1]``; ``message`` may be ``None``."""


class NullProgress:
    """No-op progress reporter."""

    def report(self, fraction: float, message: str | None = None) -> None:
        return


def progress_from_legacy_callback(
    callback: Callable[[float], None] | None,
) -> ProgressReporter | None:
    """Wrap a legacy ``progress_callback(percent: float)`` (0–100) as :class:`ProgressReporter`."""

    if callback is None:
        return None

    class _Legacy:
        def report(self, fraction: float, message: str | None = None) -> None:
            p = max(0.0, min(1.0, float(fraction))) * 100.0
            callback(p)

    return _Legacy()


def legacy_callback_from_progress(
    reporter: ProgressReporter | None,
) -> Callable[[float], None] | None:
    """Adapt :class:`ProgressReporter` to ``progress_callback(percent: float)``."""

    if reporter is None:
        return None

    def _cb(percent: float) -> None:
        reporter.report(float(percent) / 100.0, None)

    return _cb


__all__ = [
    "ProgressReporter",
    "NullProgress",
    "progress_from_legacy_callback",
    "legacy_callback_from_progress",
]
