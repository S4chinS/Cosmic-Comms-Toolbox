"""Cooperative cancellation token for long-running analysis."""

from __future__ import annotations


class CancelToken:
    """Thread-safe cooperative cancel flag (GUI may set from another thread)."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check(self) -> None:
        """Raise :class:`AnalysisCancelled` if cancelled."""

        if self._cancelled:
            raise AnalysisCancelled()


class AnalysisCancelled(Exception):
    """Raised when a run is cancelled via :class:`CancelToken`."""


__all__ = ["CancelToken", "AnalysisCancelled"]
