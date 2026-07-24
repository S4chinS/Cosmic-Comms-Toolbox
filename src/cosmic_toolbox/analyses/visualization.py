"""Pure geometry helpers for globe / frame transforms (no Qt)."""


def ecef_to_globe_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert ECEF/ITRF coordinates to the globe widget plotting frame (+90° Z from ECEF)."""

    return (-y, x, z)


__all__ = ["ecef_to_globe_coords"]
