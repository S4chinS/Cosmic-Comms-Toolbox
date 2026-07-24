from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import moderngl
import numpy as np
from matplotlib import image as mpl_image
from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from cosmic_toolbox_gui.constants import (
    EARTH_DAYMAP_FILE,
    EARTH_NIGHTMAP_FILE,
    EARTH_SPECULAR_FILE,
    STARFIELD_FILE,
)

logger = logging.getLogger(__name__)

_PRELOADED_IMAGES: dict[Path, np.ndarray | None] = {}
_TEXTURE_LOAD_ORDER: list[tuple[str, Path]] = [
    ("Earth day map", EARTH_DAYMAP_FILE),
    ("Earth night map", EARTH_NIGHTMAP_FILE),
    ("Earth specular map", EARTH_SPECULAR_FILE),
    ("Starfield", STARFIELD_FILE),
]


def preload_globe_textures(
    progress_callback: Callable[[str, float], None] | None = None
) -> None:
    """Eagerly load globe textures so later widget init is instant."""

    total = len(_TEXTURE_LOAD_ORDER)
    if total == 0:
        return
    for index, (label, path) in enumerate(_TEXTURE_LOAD_ORDER, start=1):
        progress = (index - 1) / total
        if progress_callback:
            progress_callback(f"Loading textures: {label}", progress)
        if path in _PRELOADED_IMAGES:
            if progress_callback:
                progress_callback(f"Loading textures: {label}", index / total)
            continue
        _load_image(path)  # caches internally
        if progress_callback:
            progress_callback(f"Loading textures: {label}", index / total)
    if progress_callback:
        progress_callback("Textures ready", 1.0)


@dataclass(frozen=True)
class MeshBuffers:
    """Container for shared vertex/index buffers."""

    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    vertex_count: int
    index_element_size: int


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / max(aspect, 1e-6)
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    real_up = np.cross(right, forward)
    view = np.identity(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = real_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot = np.identity(4, dtype=np.float32)
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    return rot


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot = np.identity(4, dtype=np.float32)
    rot[1, 1] = c
    rot[1, 2] = -s
    rot[2, 1] = s
    rot[2, 2] = c
    return rot


def _translation(vec: Iterable[float]) -> np.ndarray:
    mat = np.identity(4, dtype=np.float32)
    mat[:3, 3] = np.array(tuple(vec)[:3], dtype=np.float32)
    return mat


def _scale(value: float) -> np.ndarray:
    mat = np.identity(4, dtype=np.float32)
    mat[0, 0] = value
    mat[1, 1] = value
    mat[2, 2] = value
    return mat


def _gl_bytes(mat: np.ndarray) -> bytes:
    return np.asarray(mat, dtype=np.float32).T.tobytes()


def ecef_sun_direction_to_globe(x: float, y: float, z: float) -> np.ndarray:
    """Convert sun direction from ECEF to globe coordinates with shader correction.

    Applies +90° Z-rotation to align with globe coordinate system.

    Args:
        x: ECEF X component of sun direction
        y: ECEF Y component of sun direction
        z: ECEF Z component of sun direction (+ = north, - = south)

    Returns:
        np.ndarray: Sun direction vector in globe coordinates
    """
    return np.array([-y, x, z], dtype=np.float32)


def _rotate_vec_z(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a 3D vector about +Z by angle_rad."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float32)


def _solar_declination_rad(day_of_year: int) -> float:
    """Approximate solar declination (radians) for a given day-of-year."""
    # Simple model: δ ≈ 23.44° * sin(2π*(N-81)/365)
    return math.radians(23.44) * math.sin(2.0 * math.pi * (float(day_of_year) - 81.0) / 365.0)


def _subsolar_longitude_rad(now_utc: datetime) -> float:
    """Approximate subsolar longitude (radians, east-positive) from UTC time."""
    seconds = (
        now_utc.hour * 3600
        + now_utc.minute * 60
        + now_utc.second
        + now_utc.microsecond / 1_000_000.0
    )
    frac = seconds / 86400.0
    # At 12:00 UTC, subsolar longitude ~ 0° (Greenwich meridian).
    lon_deg = 180.0 - 360.0 * frac
    # Wrap to [-180, 180)
    lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0
    return math.radians(lon_deg)


def _load_image(path: Path, *, cache_result: bool = True) -> np.ndarray | None:
    if cache_result and path in _PRELOADED_IMAGES:
        return _PRELOADED_IMAGES[path]
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        data = mpl_image.imread(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read image: {path}") from exc
    array = np.asarray(data)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if cache_result:
        _PRELOADED_IMAGES[path] = array
    return array


def _generate_sphere(
    radius: float,
    theta_segments: int,
    phi_segments: int,
    *,
    invert_normals: bool = False,
    flattening: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sphere or oblate ellipsoid mesh.

    Args:
        radius: Equatorial radius
        theta_segments: Number of longitudinal segments
        phi_segments: Number of latitudinal segments
        invert_normals: If True, flip normals inward
        flattening: Earth flattening factor (0 for sphere, 1/298.257 for WGS84)

    Returns:
        Tuple of (vertices, indices) arrays
    """
    vertices: list[list[float]] = []
    indices: list[int] = []

    # Calculate polar radius for oblate ellipsoid
    polar_scale = 1.0 - flattening

    for j in range(phi_segments + 1):
        v = j / phi_segments
        phi = math.pi * v
        for i in range(theta_segments + 1):
            u = i / theta_segments
            theta = 2 * math.pi * u

            # Generate ellipsoid coordinates
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * polar_scale * math.cos(phi)

            # Calculate proper normals for ellipsoid
            # For ellipsoid, normals are not radial - need to account for scaling
            nx = math.sin(phi) * math.cos(theta) / radius
            ny = math.sin(phi) * math.sin(theta) / radius
            nz = math.cos(phi) / (radius * polar_scale)
            # Normalize the normal vector
            n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= n_len
            ny /= n_len
            nz /= n_len

            if invert_normals:
                nx, ny, nz = -nx, -ny, -nz

            # Texture coordinates based on spherical coordinates
            lon = math.atan2(x, y)
            # For lat calculation, use the actual position
            r_xy = math.sqrt(x * x + y * y)
            lat = math.atan2(z, r_xy)
            tex_u = 0.5 - lon / (2 * math.pi)
            tex_v = 0.5 - lat / math.pi
            vertices.append([x, y, z, nx, ny, nz, tex_u, tex_v])

    for j in range(phi_segments):
        for i in range(theta_segments):
            a = j * (theta_segments + 1) + i
            b = a + theta_segments + 1
            if invert_normals:
                indices.extend([a, b + 1, b, a, a + 1, b + 1])
            else:
                indices.extend([a, b, b + 1, a, b + 1, a + 1])
    vertices_arr = np.array(vertices, dtype=np.float32)
    indices_arr = np.array(indices, dtype=np.uint32)
    return vertices_arr, indices_arr


def _polyline_to_segments(points: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) polyline into (2*(N-1), 3) vertex pairs for GL_LINES."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    segments = np.empty((2 * (pts.shape[0] - 1), 3), dtype=np.float32)
    segments[0::2] = pts[:-1]
    segments[1::2] = pts[1:]
    return segments


def _generate_graticule(
    radius: float,
    *,
    parallels_deg: Iterable[float] = (-60.0, -30.0, 30.0, 60.0),
    meridian_step_deg: float = 30.0,
    segments: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate lat/lon graticule line geometry as GL_LINES vertex pairs.

    Returns a tuple ``(equator, grid)`` of ``(M, 3)`` float32 arrays where the
    equator is kept separate so it can be drawn brighter than the faint
    parallels and meridians.  Uses +Z as the polar axis to match the textured
    sphere mesh and the ground-track overlay.
    """
    r = float(radius)

    # Equator (lat = 0): circle in the XY plane.
    a = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    equator_pts = np.column_stack(
        [r * np.cos(a), r * np.sin(a), np.zeros_like(a)]
    ).astype(np.float32)
    equator = _polyline_to_segments(equator_pts)

    grid_chunks: list[np.ndarray] = []

    # Parallels (latitude circles).
    for lat_deg in parallels_deg:
        lat = math.radians(float(lat_deg))
        ring_r = r * math.cos(lat)
        z = r * math.sin(lat)
        ring = np.column_stack(
            [ring_r * np.cos(a), ring_r * np.sin(a), np.full_like(a, z)]
        ).astype(np.float32)
        grid_chunks.append(_polyline_to_segments(ring))

    # Meridians (longitude half-circles from pole to pole).
    phi = np.linspace(-math.pi / 2.0, math.pi / 2.0, segments + 1)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    lon = 0.0
    while lon < 360.0:
        ang = math.radians(lon)
        meridian = np.column_stack(
            [
                r * cos_phi * math.cos(ang),
                r * cos_phi * math.sin(ang),
                r * sin_phi,
            ]
        ).astype(np.float32)
        grid_chunks.append(_polyline_to_segments(meridian))
        lon += float(meridian_step_deg)

    if grid_chunks:
        grid = np.vstack(grid_chunks).astype(np.float32)
    else:
        grid = np.zeros((0, 3), dtype=np.float32)
    return equator, grid


_COASTLINE_LONLAT_CACHE: list[np.ndarray] | None = None


def _load_coastline_polylines() -> list[np.ndarray]:
    """Load Natural Earth 110m coastline polylines as (N, 2) lon/lat arrays.

    Reuses the same cartopy Natural Earth dataset already used by the ground-track
    map.  Cached at module scope so multiple GlobeWidget instances share one load.
    Raises if cartopy/the dataset is unavailable so the failure is visible rather
    than silently degraded.
    """
    global _COASTLINE_LONLAT_CACHE
    if _COASTLINE_LONLAT_CACHE is not None:
        return _COASTLINE_LONLAT_CACHE

    from cartopy import feature as cfeature

    polylines: list[np.ndarray] = []
    for geom in cfeature.COASTLINE.with_scale("110m").geometries():
        geom_type = getattr(geom, "geom_type", "")
        if geom_type == "LineString":
            lines = [geom]
        elif geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue
        for line in lines:
            coords = np.asarray(line.coords, dtype=np.float64)
            if coords.ndim == 2 and coords.shape[0] >= 2:
                polylines.append(coords[:, :2])

    _COASTLINE_LONLAT_CACHE = polylines
    return polylines


def _generate_coastlines(radius: float) -> np.ndarray:
    """Coastline geometry as GL_LINES vertex pairs (M, 3) on a sphere of `radius`.

    Uses +Z as the polar axis and longitude measured from +X toward +Y, matching
    the graticule and textured sphere so coastlines align with the grid and rotate
    with the same Earth model matrix.
    """
    r = float(radius)
    chunks: list[np.ndarray] = []
    for lonlat in _load_coastline_polylines():
        lon = np.radians(lonlat[:, 0])
        lat = np.radians(lonlat[:, 1])
        cos_lat = np.cos(lat)
        pts = np.column_stack(
            [
                r * cos_lat * np.cos(lon),
                r * cos_lat * np.sin(lon),
                r * np.sin(lat),
            ]
        ).astype(np.float32)
        chunks.append(_polyline_to_segments(pts))
    if chunks:
        return np.vstack(chunks).astype(np.float32)
    return np.zeros((0, 3), dtype=np.float32)


def _generate_cube(size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cube mesh centered at the origin.

    Vertex format matches `_generate_sphere`: position(3), normal(3), uv(2).
    """
    s = float(size) / 2.0
    # 6 faces × 4 vertices each (unique normals/UVs per face)
    # Each vertex: [x, y, z, nx, ny, nz, u, v]
    v: list[list[float]] = []
    i: list[int] = []

    def _add_face(
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
        p3: tuple[float, float, float],
        n: tuple[float, float, float],
    ) -> None:
        base = len(v)
        nx, ny, nz = n
        v.extend(
            [
                [p0[0], p0[1], p0[2], nx, ny, nz, 0.0, 0.0],
                [p1[0], p1[1], p1[2], nx, ny, nz, 1.0, 0.0],
                [p2[0], p2[1], p2[2], nx, ny, nz, 1.0, 1.0],
                [p3[0], p3[1], p3[2], nx, ny, nz, 0.0, 1.0],
            ]
        )
        # Two triangles per face (counter-clockwise winding)
        i.extend([base + 0, base + 1, base + 2, base + 0, base + 2, base + 3])

    # +X
    _add_face(
        (s, -s, -s),
        (s, s, -s),
        (s, s, s),
        (s, -s, s),
        (1.0, 0.0, 0.0),
    )
    # -X
    _add_face(
        (-s, s, -s),
        (-s, -s, -s),
        (-s, -s, s),
        (-s, s, s),
        (-1.0, 0.0, 0.0),
    )
    # +Y
    _add_face(
        (-s, s, -s),
        (-s, s, s),
        (s, s, s),
        (s, s, -s),
        (0.0, 1.0, 0.0),
    )
    # -Y
    _add_face(
        (-s, -s, -s),
        (s, -s, -s),
        (s, -s, s),
        (-s, -s, s),
        (0.0, -1.0, 0.0),
    )
    # +Z
    _add_face(
        (-s, -s, s),
        (s, -s, s),
        (s, s, s),
        (-s, s, s),
        (0.0, 0.0, 1.0),
    )
    # -Z
    _add_face(
        (-s, s, -s),
        (s, s, -s),
        (s, -s, -s),
        (-s, -s, -s),
        (0.0, 0.0, -1.0),
    )

    return np.asarray(v, dtype=np.float32), np.asarray(i, dtype=np.uint32)


def _generate_box(
    *,
    dims: tuple[float, float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a box mesh with arbitrary dimensions and centroid.

    Vertex format matches `_generate_sphere`: position(3), normal(3), uv(2).
    """
    dx, dy, dz = (float(dims[0]), float(dims[1]), float(dims[2]))
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

    v: list[list[float]] = []
    i: list[int] = []

    def _add_face(
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
        p3: tuple[float, float, float],
        n: tuple[float, float, float],
    ) -> None:
        base = len(v)
        nx, ny, nz = n
        v.extend(
            [
                [p0[0], p0[1], p0[2], nx, ny, nz, 0.0, 0.0],
                [p1[0], p1[1], p1[2], nx, ny, nz, 1.0, 0.0],
                [p2[0], p2[1], p2[2], nx, ny, nz, 1.0, 1.0],
                [p3[0], p3[1], p3[2], nx, ny, nz, 0.0, 1.0],
            ]
        )
        i.extend([base + 0, base + 1, base + 2, base + 0, base + 2, base + 3])

    # Define corners about the centroid.
    x0, x1 = (cx - hx, cx + hx)
    y0, y1 = (cy - hy, cy + hy)
    z0, z1 = (cz - hz, cz + hz)

    # +X
    _add_face((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (1.0, 0.0, 0.0))
    # -X
    _add_face((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (-1.0, 0.0, 0.0))
    # +Y
    _add_face((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (0.0, 1.0, 0.0))
    # -Y
    _add_face((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0.0, -1.0, 0.0))
    # +Z
    _add_face((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0.0, 0.0, 1.0))
    # -Z
    _add_face((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0), (0.0, 0.0, -1.0))

    return np.asarray(v, dtype=np.float32), np.asarray(i, dtype=np.uint32)


# Baseline visual dimensions for the simplified spacecraft mesh (meters).
# Body cuboid + 2 symmetric solar-panel cuboids; values are purely cosmetic.
_BODY_XYZ_M: tuple[float, float, float] = (1.0, 0.75, 0.75)
_SOLAR_XYZ_M: tuple[float, float, float] = (1.0, 1.2, 0.01)


def _load_baseline_spacecraft_dims() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the hard-coded baseline spacecraft dimensions used for rendering."""
    return _BODY_XYZ_M, _SOLAR_XYZ_M


def _generate_simplified_spacecraft_mesh() -> tuple[np.ndarray, np.ndarray, float]:
    """Simplified baseline geometry (body cuboid + 2 solar panel cuboids).

    Returned mesh is normalized to fit within [-0.5, +0.5] in local units.
    """
    body_xyz_m, solar_xyz_m = _load_baseline_spacecraft_dims()
    _, by, _ = body_xyz_m
    _, sy, _ = solar_xyz_m
    y_off = 0.5 * float(by) + 0.5 * float(sy)

    boxes = [
        (body_xyz_m, (0.0, 0.0, 0.0)),
        (solar_xyz_m, (0.0, -y_off, 0.0)),
        (solar_xyz_m, (0.0, +y_off, 0.0)),
    ]

    verts_all: list[np.ndarray] = []
    idx_all: list[np.ndarray] = []
    base = 0
    for dims, center in boxes:
        v, idx = _generate_box(dims=tuple(dims), center=center)
        verts_all.append(v)
        idx_all.append(idx + base)
        base += int(v.shape[0])

    verts = np.vstack(verts_all).astype(np.float32)
    idxs = np.concatenate(idx_all).astype(np.uint32)

    pos = verts[:, 0:3].astype(np.float64)
    extent_m = float(np.max(np.abs(pos))) if pos.size else 1.0
    if not np.isfinite(extent_m) or extent_m <= 0.0:
        extent_m = 1.0
    verts[:, 0:3] = (pos / (2.0 * extent_m)).astype(np.float32)
    local_extent = float(np.max(np.abs(verts[:, 0:3]))) if verts.size else 0.5
    return verts, idxs, local_extent


class OrbitCamera:
    """Minimal orbit-style camera with yaw/pitch control."""

    def __init__(
        self,
        *,
        radius_km: float,
        height_km: float,
        view_angle_deg: float,
    ) -> None:
        self._radius_base = radius_km
        self._height_base = height_km
        self.view_angle_deg = view_angle_deg
        self._zoom_scale = 1.0
        self.azimuth_rad = 0.0
        self._focus_point = np.zeros(3, dtype=np.float32)
        self._orbit_offset = np.zeros(3, dtype=np.float32)
        self._base_distance = math.sqrt(radius_km**2 + height_km**2)
        self._base_pitch = math.atan2(height_km, radius_km)
        self.pitch_rad = self._base_pitch
        self._pitch_limits = (
            math.radians(-89.5),
            math.radians(89.5),
        )
        self._update_position()

    def _current_distance(self) -> float:
        return self._base_distance * self._zoom_scale

    def _clamp_pitch(self) -> None:
        self.pitch_rad = float(
            np.clip(self.pitch_rad, self._pitch_limits[0], self._pitch_limits[1])
        )

    def _update_position(self) -> None:
        self._clamp_pitch()
        distance = self._current_distance()
        cos_pitch = math.cos(self.pitch_rad)
        offset = np.array(
            [
                distance * cos_pitch * math.cos(self.azimuth_rad),
                distance * cos_pitch * math.sin(self.azimuth_rad),
                distance * math.sin(self.pitch_rad),
            ],
            dtype=np.float32,
        )
        self._orbit_offset = offset
        self.position = self._focus_point + offset

    def set_focus_point(self, target: Iterable[float]) -> None:
        coords = tuple(target)[:3]
        if len(coords) < 3:
            raise ValueError("Focus point must provide 3 coordinates")
        self._focus_point = np.array(coords, dtype=np.float32)
        self._update_position()

    @property
    def focus_point(self) -> np.ndarray:
        return self._focus_point.copy()

    def set_rotation(self, angle_rad: float) -> None:
        self.azimuth_rad = angle_rad
        self._update_position()

    def rotate_by(self, delta_rad: float) -> None:
        self.azimuth_rad = (self.azimuth_rad + delta_rad) % (2 * math.pi)
        self._update_position()

    def set_pitch(self, angle_rad: float) -> None:
        self.pitch_rad = angle_rad
        self._update_position()

    def tilt_by(self, delta_rad: float) -> None:
        self.pitch_rad += delta_rad
        self._update_position()

    def zoom_by(self, factor: float) -> None:
        self._zoom_scale = float(
            np.clip(self._zoom_scale * factor, 0.25, 4.0)  # type: ignore[arg-type]
        )
        self._update_position()

    def reset_zoom(self) -> None:
        self._zoom_scale = 1.0
        self._update_position()

    def reset_pitch(self) -> None:
        self.pitch_rad = self._base_pitch
        self._update_position()

    def align_north(self) -> None:
        """Re-align roll so that world north stays 'up'."""
        # Roll is implicitly constrained by using the world up vector.
        # A call here simply ensures limits are enforced and position refreshed.
        self._update_position()

    def view_matrix(self) -> np.ndarray:
        eye = self.position
        target = self._focus_point
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return _look_at(eye, target, up)


class GlobeWidget(QOpenGLWidget):
    """ModernGL-based textured globe with optional overlays."""

    def __init__(self, parent=None, radius_km: float = 6378.0) -> None:
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.setMouseTracking(True)
        self._ctx: moderngl.Context | None = None
        self._radius_km = radius_km
        self._camera = OrbitCamera(
            radius_km=radius_km * 3.0,
            height_km=radius_km * 0.3,
            view_angle_deg=30.0,
        )
        self._projection = np.identity(4, dtype=np.float32)
        self._earth_rotation_deg = 0.0
        self._earth_rotation_mode = "ECI"
        # Earth rendering mode: "wireframe" (lat/lon graticule) or "textured".
        self._earth_render_mode = "wireframe"
        self._graticule_equator_vbo: moderngl.Buffer | None = None
        self._graticule_grid_vbo: moderngl.Buffer | None = None
        self._graticule_equator_count = 0
        self._graticule_grid_count = 0
        self._coastline_vbo: moderngl.Buffer | None = None
        self._coastline_count = 0
        # Wireframe palette (ported from conops-visualization buildEarthWireframe).
        self._wireframe_equator_color = (1.0, 1.0, 1.0, 0.28)
        self._wireframe_grid_color = (0.63, 0.63, 0.67, 0.16)
        # Coastlines: faint cyan derived from the ground-track map colour (#3cc3ff).
        self._wireframe_coastline_color = (0.235, 0.765, 1.0, 0.45)
        # Opaque body fill so the far-side grid/coastlines are occluded.
        self._wireframe_opaque = True
        self._wireframe_fill_color = (0.035, 0.040, 0.050, 1.0)
        self._wireframe_clear_color = (0.020, 0.020, 0.024, 1.0)
        self._wireframe_terminator_color = (0.961, 0.647, 0.141, 0.55)
        self._wireframe_show_terminator = False
        self._terminator_vbo: moderngl.Buffer | None = None
        self._terminator_count = 0
        self._track_vbo: moderngl.Buffer | None = None
        self._track_vertex_count = 0
        self._pending_track_coords: np.ndarray | None = None
        self._pending_track_clear = False
        self._sensor_outline_vbo: moderngl.Buffer | None = None
        self._sensor_outline_vertex_count = 0
        self._pending_sensor_outline_coords: np.ndarray | None = None
        self._pending_sensor_outline_clear = False
        self._link_vbo: moderngl.Buffer | None = None
        self._link_vertex_count = 0
        self._satellite_model = np.identity(4, dtype=np.float32)
        self._satellite_visible = False
        # Satellite marker scale (km) and local extent for triad sizing.
        self._satellite_scale_km = 500.0
        self._satellite_local_extent = 0.5
        self._earth_texture = None
        self._night_texture = None
        self._cloud_texture = None
        self._specular_texture = None
        self._cloud_rotation_deg = 0.0
        self._night_blend_softness = 0.25
        self._starfield_texture = None
        self._mesh_buffers: dict[str, MeshBuffers] = {}
        self._programs = {}
        self._vaos = {}
        self._fallback_textures: dict[str, moderngl.Texture] = {}
        self._mouse_last_pos = QPoint()
        self._sun_color = (1.0, 0.98, 0.92)
        self._sun_brightness = 1.0
        self._day_night_enabled = True
        self._uniform_lighting = False
        self._ambient_color = (0.03, 0.05, 0.08)
        self._ambient_intensity = 0.15
        self._twilight_strength = 0.4
        self._twilight_exponent = 2.2
        self._specular_intensity = 0.12
        self._specular_exponent = 64.0
        self._light_model = "directional"
        self._sun_distance_km = radius_km * 400.0
        self._point_light_falloff = (1e-3, 2e-6)
        self._momentum_velocity = np.zeros(2, dtype=np.float32)
        self._momentum_decay = 0.92
        self._momentum_timer = QTimer(self)
        self._momentum_timer.setInterval(16)
        self._momentum_timer.timeout.connect(self._apply_momentum)
        self._arrow_vbo: moderngl.Buffer | None = None
        self._arrow_vertex_count = 0
        self._pending_arrow_vertices: np.ndarray | None = None
        self._arrow_color = (1.0, 0.55, 0.0, 1.0)
        self._sensor_outline_color = (1.0, 0.2, 0.75, 1.0)
        self._triad_vbos: dict[str, moderngl.Buffer | None] = {"x": None, "y": None, "z": None}
        self._triad_vertex_counts: dict[str, int] = {"x": 0, "y": 0, "z": 0}
        self._pending_triad_vertices: dict[str, np.ndarray | None] = {"x": None, "y": None, "z": None}
        self._sun_datetime: datetime | None = None

    def sizeHint(self) -> QSize:  # pragma: no cover - Qt hook
        return QSize(450, 450)

    # ------------------------------------------------------------------
    # Qt / ModernGL lifecycle hooks
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:  # pragma: no cover - GPU init
        self._ctx = moderngl.create_context(require=330)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._compile_programs()
        self._build_meshes()
        self._load_textures()
        self._build_vaos()

    def resizeGL(self, width: int, height: int) -> None:  # pragma: no cover - Qt hook
        if self._ctx is None:
            return
        self._bind_default_framebuffer(width, height)
        aspect = width / max(height, 1)
        self._projection = _perspective(
            self._camera.view_angle_deg,
            aspect,
            near=100.0,
            far=self._radius_km * 20.0,
        )

    def paintGL(self) -> None:  # pragma: no cover - Qt hook
        if self._ctx is None:
            return
        self._bind_default_framebuffer(self.width(), self.height())
        wireframe = self._earth_render_mode == "wireframe"
        if wireframe:
            self._ctx.clear(*self._wireframe_clear_color)
        else:
            self._ctx.clear(0.02, 0.02, 0.04, 1.0)
        view = self._camera.view_matrix()
        self._flush_pending_track()
        self._flush_pending_sensor_outline()
        self._flush_pending_arrow()
        self._flush_pending_triad()
        if wireframe:
            if self._wireframe_opaque:
                self._draw_wireframe_fill(view)
            self._draw_wireframe_earth(view)
            self._draw_terminator(view)
        else:
            self._draw_starfield(view)
            self._draw_earth(view)
        self._draw_track(view)
        self._draw_sensor_outline(view)
        self._draw_arrow(view)
        self._draw_satellite(view)
        self._draw_triad(view)
        self._draw_link(view)

    def _bind_default_framebuffer(self, width: int, height: int) -> None:
        if self._ctx is None:
            return
        framebuffer = self._ctx.detect_framebuffer()
        framebuffer.use()
        dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        width_px = max(int(width * dpr), 1)
        height_px = max(int(height * dpr), 1)
        self._ctx.viewport = (0, 0, width_px, height_px)

    def _flush_pending_track(self) -> None:
        if self._ctx is None:
            return
        if self._pending_track_clear:
            if self._track_vbo is not None:
                self._track_vbo.release()
            self._track_vbo = None
            self._track_vertex_count = 0
            self._vaos.pop("track", None)
            self._pending_track_clear = False
            logger.debug("Track buffer cleared")
        if self._pending_track_coords is not None:
            data = self._pending_track_coords
            flat = data.ravel()
            if self._track_vbo is None or self._track_vbo.size != flat.nbytes:
                if self._track_vbo is not None:
                    self._track_vbo.release()
                self._track_vbo = self._ctx.buffer(flat.tobytes())
            else:
                self._track_vbo.write(flat.tobytes())
            self._track_vertex_count = len(data)
            if len(data) >= 2:
                last = data[-1]
                prev = data[-2]
                direction = last - prev
                self.update_direction_arrow(tuple(last), tuple(direction))
            else:
                self.update_direction_arrow(None, None)
            self._vaos.pop("track", None)
            self._pending_track_coords = None
            logger.debug("Track buffer flushed (%d points)", self._track_vertex_count)

    def _flush_pending_sensor_outline(self) -> None:
        if self._ctx is None:
            return
        if self._pending_sensor_outline_clear:
            if self._sensor_outline_vbo is not None:
                self._sensor_outline_vbo.release()
            self._sensor_outline_vbo = None
            self._sensor_outline_vertex_count = 0
            self._vaos.pop("sensor_outline", None)
            self._pending_sensor_outline_clear = False
        if self._pending_sensor_outline_coords is not None:
            data = self._pending_sensor_outline_coords
            flat = data.ravel()
            recreated_buffer = False
            if self._sensor_outline_vbo is None or self._sensor_outline_vbo.size != flat.nbytes:
                if self._sensor_outline_vbo is not None:
                    self._sensor_outline_vbo.release()
                self._sensor_outline_vbo = self._ctx.buffer(flat.tobytes())
                recreated_buffer = True
            else:
                self._sensor_outline_vbo.orphan(flat.nbytes)
                self._sensor_outline_vbo.write(flat.tobytes())
            self._sensor_outline_vertex_count = len(data)
            if recreated_buffer:
                self._vaos.pop("sensor_outline", None)
            self._pending_sensor_outline_coords = None
            logger.debug(
                "Sensor outline buffer flushed (%d points)",
                self._sensor_outline_vertex_count,
            )

    def _flush_pending_arrow(self) -> None:
        if self._ctx is None or self._pending_arrow_vertices is None:
            return
        data = self._pending_arrow_vertices
        self._pending_arrow_vertices = None
        if data.size == 0:
            if self._arrow_vbo is not None:
                self._arrow_vbo.release()
            self._arrow_vbo = None
            self._arrow_vertex_count = 0
            self._vaos.pop("arrow", None)
            return
        flat = data.astype(np.float32).ravel()
        if self._arrow_vbo is None or self._arrow_vbo.size != flat.nbytes:
            if self._arrow_vbo is not None:
                self._arrow_vbo.release()
            self._arrow_vbo = self._ctx.buffer(flat.tobytes())
        else:
            self._arrow_vbo.write(flat.tobytes())
        self._arrow_vertex_count = len(data)
        self._vaos.pop("arrow", None)

    def _flush_pending_triad(self) -> None:
        if self._ctx is None:
            return
        any_pending = any(self._pending_triad_vertices[axis] is not None for axis in ("x", "y", "z"))
        if not any_pending:
            return
        for axis in ("x", "y", "z"):
            data = self._pending_triad_vertices[axis]
            if data is None:
                continue
            self._pending_triad_vertices[axis] = None
            if data.size == 0:
                if self._triad_vbos[axis] is not None:
                    self._triad_vbos[axis].release()
                self._triad_vbos[axis] = None
                self._triad_vertex_counts[axis] = 0
                self._vaos.pop(f"triad_{axis}", None)
                continue
            flat = data.astype(np.float32).ravel()
            buf = self._triad_vbos[axis]
            if buf is None or buf.size != flat.nbytes:
                if buf is not None:
                    buf.release()
                self._triad_vbos[axis] = self._ctx.buffer(flat.tobytes())
            else:
                buf.write(flat.tobytes())
            self._triad_vertex_counts[axis] = len(data)
            self._vaos.pop(f"triad_{axis}", None)

    # ------------------------------------------------------------------
    # Public API for UI mixins
    # ------------------------------------------------------------------
    def set_frame_rotation(
        self, mode: str, angle_deg: float, *, request_repaint: bool = True
    ) -> None:
        """Update Earth rotation for the supplied reference frame."""
        self._earth_rotation_mode = mode
        self._earth_rotation_deg = angle_deg
        self._cloud_rotation_deg = angle_deg
        if request_repaint:
            self.update()

    def set_earth_render_mode(self, mode: str) -> None:
        """Switch the Earth between wireframe (graticule) and textured rendering."""
        if mode not in {"wireframe", "textured"}:
            raise ValueError("Unsupported earth render mode: %s" % mode)
        self._earth_render_mode = mode
        self.update()

    def set_wireframe_terminator_enabled(self, enabled: bool) -> None:
        """Toggle the sun-terminator great circle in wireframe mode."""
        self._wireframe_show_terminator = bool(enabled)
        self.update()

    def set_wireframe_opaque(self, enabled: bool) -> None:
        """Toggle the opaque Earth body fill in wireframe mode.

        When enabled the globe is solid and only near-side grid/coastlines are
        visible; when disabled the wireframe is see-through.
        """
        self._wireframe_opaque = bool(enabled)
        self.update()

    def set_light_model(self, mode: str) -> None:
        """Switch between directional (sun-like) and point light models."""
        if mode not in {"directional", "point"}:
            raise ValueError("Unsupported light model: %s" % mode)
        self._light_model = mode
        self.update()

    def set_day_night_enabled(self, enabled: bool) -> None:
        """Enable or disable the day/night shading pipeline."""
        self._day_night_enabled = bool(enabled)
        self.update()

    def set_uniform_lighting(self, enabled: bool) -> None:
        """Force a fully lit appearance (used by mission tab)."""
        self._uniform_lighting = bool(enabled)
        self.update()

    def set_sun_brightness(self, value: float) -> None:
        """Adjust overall sun intensity multiplier."""
        self._sun_brightness = float(np.clip(value, 0.05, 5.0))
        self.update()

    def set_sun_datetime(
        self, timestamp: datetime | None, *, request_repaint: bool = True
    ) -> None:
        """Set the datetime used for sun/season lighting (UTC)."""
        if timestamp is None:
            self._sun_datetime = None
        else:
            self._sun_datetime = timestamp.astimezone(timezone.utc)
        if request_repaint:
            self.update()

    def update_track(self, coords_km: Optional[np.ndarray]) -> None:
        if coords_km is None or len(coords_km) == 0:
            self._pending_track_coords = None
            self._pending_track_clear = True
            logger.debug("Clearing track buffer")
        else:
            coords = np.asarray(coords_km, dtype=np.float32)
            logger.debug("Accepting track buffer with %d points", coords.shape[0])
            self._pending_track_coords = coords.copy()
            self._pending_track_clear = False
        self.update()

    def update_sensor_cone_outline(
        self,
        coords_km: Optional[np.ndarray],
        colors_rgba: Optional[np.ndarray] = None,
        *,
        request_repaint: bool = True,
    ) -> None:
        if coords_km is None or len(coords_km) == 0:
            self._pending_sensor_outline_coords = None
            self._pending_sensor_outline_clear = True
            logger.debug("Clearing sensor outline buffer")
        else:
            coords = np.asarray(coords_km, dtype=np.float32)
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError("Sensor outline coordinates must have shape (N, 3)")
            if colors_rgba is None:
                color = np.asarray(self._sensor_outline_color, dtype=np.float32).reshape(1, 4)
                colors = np.repeat(color, coords.shape[0], axis=0)
            else:
                colors = np.asarray(colors_rgba, dtype=np.float32)
                if colors.shape != (coords.shape[0], 4):
                    raise ValueError("Sensor outline colors must have shape (N, 4)")
            packed = np.empty((coords.shape[0], 7), dtype=np.float32)
            packed[:, :3] = coords
            packed[:, 3:] = colors
            logger.debug("Accepting sensor outline buffer with %d points", coords.shape[0])
            self._pending_sensor_outline_coords = packed
            self._pending_sensor_outline_clear = False
        if request_repaint:
            self.update()

    def radius_km(self) -> float:
        return float(self._radius_km)

    @staticmethod
    def _arrow_vertices(
        base: np.ndarray,
        direction: np.ndarray,
        *,
        length: float,
        head_length_ratio: float = 0.22,
        head_width_ratio: float = 0.09,
    ) -> np.ndarray:
        """Return a simple arrow (lines) as 6 vertices for rendering as LINES."""
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return np.zeros((0, 3), dtype=np.float32)
        dir_norm = direction.astype(np.float32) / norm
        tip = base + dir_norm * float(length)
        side = np.cross(dir_norm, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if float(np.linalg.norm(side)) < 1e-6:
            side = np.cross(dir_norm, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        side /= max(float(np.linalg.norm(side)), 1e-6)
        head_length = float(length) * float(head_length_ratio)
        head_width = float(length) * float(head_width_ratio)
        left = tip - dir_norm * head_length + side * head_width
        right = tip - dir_norm * head_length - side * head_width
        return np.array([base, tip, tip, left, tip, right], dtype=np.float32)

    def update_satellite_position(
        self,
        position_km: Optional[tuple[float, float, float]],
        *,
        body_axes: np.ndarray | None = None,
        show_body_axes: bool | None = None,
        request_repaint: bool = True,
    ) -> None:
        """Update satellite actor position, optional cube orientation, and optional body triad.

        Args:
            position_km: Satellite position in globe coordinates (km).
            body_axes: Optional 3x3 matrix whose columns are body X/Y/Z unit axes in world coords.
            show_body_axes: If True, render the RGB body triad; if False hide it.
                Defaults to True when body_axes is provided, otherwise False.
        """
        if position_km is None:
            self._satellite_model = np.identity(4, dtype=np.float32)
            self._satellite_visible = False
            for axis in ("x", "y", "z"):
                self._pending_triad_vertices[axis] = np.zeros((0, 3), dtype=np.float32)
            if request_repaint:
                self.update()
            return

        rot = np.identity(4, dtype=np.float32)
        axes = None if body_axes is None else np.asarray(body_axes, dtype=np.float32)
        if axes is not None and axes.shape == (3, 3):
            # Columns are the world-space directions of the model's local X/Y/Z axes.
            rot[:3, 0] = axes[:, 0]
            rot[:3, 1] = axes[:, 1]
            rot[:3, 2] = axes[:, 2]
        scale_km = float(getattr(self, "_satellite_scale_km", 200.0) or 200.0)
        self._satellite_model = _translation(position_km) @ rot @ _scale(scale_km)
        self._satellite_visible = True

        show = bool(body_axes is not None) if show_body_axes is None else bool(show_body_axes)
        if show and axes is not None and axes.shape == (3, 3):
            base = np.asarray(position_km, dtype=np.float32)
            # Scale triad so it remains visible relative to the rendered spacecraft
            # (solar panels extend in +/-Y and would otherwise occlude the Y axis line).
            local_extent = float(getattr(self, "_satellite_local_extent", 0.5) or 0.5)
            triad_scale = 1.0 / 3.0
            length = triad_scale * max(900.0, scale_km * local_extent * 12.0)
            self._pending_triad_vertices["x"] = self._arrow_vertices(
                base, axes[:, 0], length=length
            )
            self._pending_triad_vertices["y"] = self._arrow_vertices(
                base, axes[:, 1], length=length
            )
            self._pending_triad_vertices["z"] = self._arrow_vertices(
                base, axes[:, 2], length=length
            )
        else:
            for axis in ("x", "y", "z"):
                self._pending_triad_vertices[axis] = np.zeros((0, 3), dtype=np.float32)
        if request_repaint:
            self.update()

    def update_link_segment(
        self,
        start_km: Optional[tuple[float, float, float]],
        end_km: Optional[tuple[float, float, float]],
        *,
        request_repaint: bool = True,
    ) -> None:
        if self._ctx is None:
            return
        if start_km is None or end_km is None:
            if self._link_vbo is not None:
                self._link_vbo.release()
            self._link_vbo = None
            self._link_vertex_count = 0
        else:
            data = np.array([start_km, end_km], dtype=np.float32).ravel()
            if self._link_vbo is None:
                self._link_vbo = self._ctx.buffer(data.tobytes())
            else:
                self._link_vbo.orphan(len(data) * 4)
                self._link_vbo.write(data.tobytes())
            self._link_vertex_count = 2
        self._vaos.pop("link", None)
        if request_repaint:
            self.update()

    def update_direction_arrow(
        self,
        position_km: Optional[tuple[float, float, float]],
        direction_km: Optional[tuple[float, float, float]],
    ) -> None:
        if position_km is None or direction_km is None:
            self._pending_arrow_vertices = np.zeros((0, 3), dtype=np.float32)
            self.update()
            return
        base = np.asarray(position_km, dtype=np.float32)
        direction = np.asarray(direction_km, dtype=np.float32)
        vertices = self._arrow_vertices(base, direction, length=800.0)
        self._pending_arrow_vertices = vertices
        self.update()

    def set_focus_point(
        self,
        position_km: Optional[tuple[float, float, float]],
        *,
        request_repaint: bool = True,
    ) -> None:
        """Recenter the camera orbit target."""
        target = (0.0, 0.0, 0.0) if position_km is None else position_km
        self._camera.set_focus_point(target)
        if request_repaint:
            self.update()

    def reset_camera(self) -> None:
        self._camera.set_focus_point((0.0, 0.0, 0.0))
        self._camera.set_rotation(0.0)
        self._camera.reset_pitch()
        self._camera.reset_zoom()
        self.update()

    def _current_sun_direction(self) -> np.ndarray:
        if self._sun_datetime is not None:
            now = self._sun_datetime.astimezone(timezone.utc)
        else:
            now = datetime.now(timezone.utc)
        day_of_year = int(now.timetuple().tm_yday)
        decl_rad = _solar_declination_rad(day_of_year)
        subsolar_lon_rad = _subsolar_longitude_rad(now)

        # Sun direction in ECEF (unit vector Earth->Sun), east-positive longitude.
        x = math.cos(decl_rad) * math.cos(subsolar_lon_rad)
        y = math.cos(decl_rad) * math.sin(subsolar_lon_rad)
        z = math.sin(decl_rad)

        # Convert to globe coords (ECEF -> globe).
        # Here x,y,z represent Earth->Sun in ECEF.
        dir_vec = ecef_sun_direction_to_globe(x, y, z)

        mode = str(getattr(self, "_earth_rotation_mode", "ECI") or "ECI")
        earth_rot_deg = float(getattr(self, "_earth_rotation_deg", 0.0) or 0.0)
        # In ECI mode, Earth geometry is rotated ECEF->ECI by `earth_rotation_deg`,
        # so rotate the sun direction by the same transform to express it in ECI world.
        if mode == "ECI":
            dir_vec = _rotate_vec_z(dir_vec, math.radians(earth_rot_deg))

        # Shader uses `resolved_light = normalize(-light_dir)`.
        # We want resolved_light to point from surface towards the Sun (Earth->Sun),
        # so we provide light_dir as the opposite (Sun->Earth).
        dir_vec = -dir_vec
        dir_vec /= np.linalg.norm(dir_vec)
        return dir_vec

    def _sun_world_position(self, direction: np.ndarray) -> np.ndarray:
        return direction * self._sun_distance_km

    def _active_sun_direction(self) -> np.ndarray:
        if self._day_night_enabled:
            return self._current_sun_direction()
        # Default direction (no Z negation for default since it's arbitrary)
        x, y, z = 0.35, -0.45, 0.7
        default_dir = np.array([-y, x, z], dtype=np.float32)
        return default_dir / np.linalg.norm(default_dir)

    @staticmethod
    def _vec3_tuple(vec: Iterable[float]) -> tuple[float, float, float]:
        arr = np.asarray(tuple(vec)[:3], dtype=np.float32)
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    def _sun_color_scaled(self) -> tuple[float, float, float]:
        brightness = float(np.clip(self._sun_brightness, 0.05, 5.0))
        return tuple(float(max(0.0, c * brightness)) for c in self._sun_color)

    def _twilight_strength_scaled(self, base: float = 1.0) -> float:
        brightness = float(np.clip(self._sun_brightness, 0.05, 5.0))
        return float(base) / brightness

    # ------------------------------------------------------------------
    # Mouse interaction (orbit controls)
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - Qt hook
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_last_pos = event.pos()
            self._stop_momentum()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - Qt hook
        if event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.pos() - self._mouse_last_pos
            self._mouse_last_pos = event.pos()
            rotation_sensitivity = 0.005
            yaw_delta = -delta.x() * rotation_sensitivity
            pitch_delta = delta.y() * rotation_sensitivity * 0.6
            self._camera.rotate_by(yaw_delta)
            self._camera.tilt_by(pitch_delta)
            self._momentum_velocity = np.array(
                [yaw_delta, pitch_delta], dtype=np.float32
            )
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - Qt hook
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._start_momentum_if_needed():
                self._camera.align_north()
                self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # pragma: no cover - Qt hook
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta != 0:
            step = delta / 120.0
            factor = math.exp(-0.12 * step)
            self._camera.zoom_by(factor)
            self.update()
        super().wheelEvent(event)

    def _stop_momentum(self) -> None:
        if self._momentum_timer.isActive():
            self._momentum_timer.stop()
        self._momentum_velocity[:] = 0.0

    def _start_momentum_if_needed(self) -> bool:
        if np.linalg.norm(self._momentum_velocity) < 1e-5:
            return False
        if not self._momentum_timer.isActive():
            self._momentum_timer.start()
        return True

    def _apply_momentum(self) -> None:
        if np.linalg.norm(self._momentum_velocity) < 1e-5:
            self._stop_momentum()
            self._camera.align_north()
            self.update()
            return
        yaw_delta, pitch_delta = self._momentum_velocity
        self._camera.rotate_by(float(yaw_delta))
        self._camera.tilt_by(float(pitch_delta))
        self._momentum_velocity *= self._momentum_decay
        self.update()

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------
    def _compile_programs(self) -> None:
        assert self._ctx is not None
        textured_vs = """
            #version 330
            uniform mat4 mvp;
            uniform mat4 model;
            in vec3 in_pos;
            in vec3 in_normal;
            in vec2 in_uv;
            out vec2 v_uv;
            out vec3 v_normal;
            out vec3 v_world_pos;
            void main() {
                vec4 world = model * vec4(in_pos, 1.0);
                gl_Position = mvp * vec4(in_pos, 1.0);
                v_uv = in_uv;
                v_normal = mat3(model) * in_normal;
                v_world_pos = world.xyz;
            }
        """
        textured_fs = """
            #version 330
            uniform sampler2D tex;
            uniform sampler2D night_tex;
            uniform sampler2D specular_map;
            uniform vec3 light_dir;
            uniform vec3 light_pos;
            uniform vec3 camera_pos;
            uniform vec3 sun_color;
            uniform vec3 ambient_color;
            uniform vec2 point_falloff;
            uniform float ambient_intensity;
            uniform float min_light;
            uniform float opacity;
            uniform float twilight_strength;
            uniform float twilight_exponent;
            uniform float specular_intensity;
            uniform float specular_exponent;
            uniform bool use_alpha;
            uniform bool use_night_map;
            uniform bool use_specular_map;
            uniform bool fade_alpha_with_light;
            uniform float alpha_fade_min;
            uniform float night_edge_softness;
            uniform int light_type; // 0=directional, 1=point
            uniform bool uniform_lighting;
            in vec2 v_uv;
            in vec3 v_normal;
            in vec3 v_world_pos;
            out vec4 fragColor;
            void main() {
                vec3 normal = normalize(v_normal);
                vec3 view_dir = normalize(camera_pos - v_world_pos);
                vec3 resolved_light = normalize(-light_dir);
                float attenuation = 1.0;
                if (!uniform_lighting && light_type == 1) {
                    vec3 light_vec = light_pos - v_world_pos;
                    float dist = length(light_vec);
                    if (dist > 0.0) {
                        resolved_light = light_vec / dist;
                    }
                    attenuation = 1.0 / max(1.0 + point_falloff.x * dist + point_falloff.y * dist * dist, 1e-4);
                }
                float lambert = uniform_lighting ? 1.0 : max(dot(normal, resolved_light), 0.0);
                float twilight_input = uniform_lighting ? 0.0 : max(dot(normal, -resolved_light), 0.0);
                float diffuse = max(lambert, min_light) * attenuation;
                float twilight = pow(twilight_input, twilight_exponent) * twilight_strength;
                float light_mix = diffuse + twilight;
                vec4 day_sample = texture(tex, v_uv);
                vec3 base_color = day_sample.rgb;
                if (use_night_map) {
                    vec3 night_sample = texture(night_tex, v_uv).rgb;
                    float day_mix = smoothstep(0.0, max(night_edge_softness, 0.0001), lambert);
                    base_color = mix(night_sample, base_color, day_mix);
                }
                vec3 ambient = ambient_color * ambient_intensity;
                vec3 color = base_color * (ambient + sun_color * light_mix);
                if (specular_intensity > 0.0 && !uniform_lighting) {
                    vec3 half_vec = normalize(resolved_light + view_dir);
                    float spec = pow(max(dot(normal, half_vec), 0.0), specular_exponent) * specular_intensity;
                    float spec_factor = 1.0;
                    if (use_specular_map) {
                        spec_factor = texture(specular_map, v_uv).r;
                    }
                    color += sun_color * (spec * spec_factor);
                }
                float alpha = use_alpha ? day_sample.a : 1.0;
                if (fade_alpha_with_light) {
                    float fade = max(lambert, alpha_fade_min);
                    alpha *= fade;
                }
                fragColor = vec4(clamp(color, 0.0, 1.0), alpha * opacity);
            }
        """
        color_vs = """
            #version 330
            uniform mat4 mvp;
            in vec3 in_pos;
            void main() {
                gl_Position = mvp * vec4(in_pos, 1.0);
            }
        """
        color_fs = """
            #version 330
            uniform vec4 color;
            out vec4 fragColor;
            void main() {
                fragColor = color;
            }
        """
        color_vertex_vs = """
            #version 330
            uniform mat4 mvp;
            in vec3 in_pos;
            in vec4 in_color;
            out vec4 v_color;
            void main() {
                gl_Position = mvp * vec4(in_pos, 1.0);
                v_color = in_color;
            }
        """
        color_vertex_fs = """
            #version 330
            in vec4 v_color;
            out vec4 fragColor;
            void main() {
                fragColor = v_color;
            }
        """
        self._programs["textured"] = self._ctx.program(
            vertex_shader=textured_vs,
            fragment_shader=textured_fs,
        )
        self._programs["color"] = self._ctx.program(
            vertex_shader=color_vs,
            fragment_shader=color_fs,
        )
        self._programs["color_vertex"] = self._ctx.program(
            vertex_shader=color_vertex_vs,
            fragment_shader=color_vertex_fs,
        )

    def _build_meshes(self) -> None:
        assert self._ctx is not None
        # WGS84 flattening: 1/298.257223563
        wgs84_flattening = 1.0 / 298.257223563

        # Earth with WGS84 oblate ellipsoid shape
        sphere_data = _generate_sphere(
            self._radius_km,
            theta_segments=180,
            phi_segments=90,
            flattening=wgs84_flattening,
        )
        self._mesh_buffers["earth"] = self._create_mesh_buffers(*sphere_data)

        # Starfield (spherical background)
        starfield_data = _generate_sphere(
            self._radius_km * 12.0,
            theta_segments=90,
            phi_segments=45,
            invert_normals=True,
        )
        self._mesh_buffers["starfield"] = self._create_mesh_buffers(*starfield_data)

        # Satellite (simplified rideshare geometry: body + two solar panels)
        sat_v, sat_i, sat_extent = _generate_simplified_spacecraft_mesh()
        self._mesh_buffers["satellite"] = self._create_mesh_buffers(sat_v, sat_i)
        if np.isfinite(sat_extent) and float(sat_extent) > 1e-6:
            self._satellite_local_extent = float(sat_extent)

        # Wireframe Earth graticule (lat/lon grid drawn as GL_LINES pairs).
        equator, grid = _generate_graticule(self._radius_km)
        if equator.size:
            self._graticule_equator_vbo = self._ctx.buffer(equator.tobytes())
            self._graticule_equator_count = int(equator.shape[0])
        if grid.size:
            self._graticule_grid_vbo = self._ctx.buffer(grid.tobytes())
            self._graticule_grid_count = int(grid.shape[0])

        # Coastline outline (Natural Earth 110m via cartopy).  Treated as an
        # optional overlay: if the dataset cannot be loaded we log and skip it
        # rather than failing the whole globe.
        try:
            coastlines = _generate_coastlines(self._radius_km)
        except Exception:
            logger.warning("Coastline outline unavailable; skipping", exc_info=True)
            coastlines = np.zeros((0, 3), dtype=np.float32)
        if coastlines.size:
            self._coastline_vbo = self._ctx.buffer(coastlines.tobytes())
            self._coastline_count = int(coastlines.shape[0])

    def _create_mesh_buffers(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
    ) -> MeshBuffers:
        assert self._ctx is not None
        vbo = self._ctx.buffer(vertices.tobytes())
        ibo = self._ctx.buffer(indices.tobytes())
        return MeshBuffers(
            vbo=vbo,
            ibo=ibo,
            vertex_count=len(indices),
            index_element_size=indices.dtype.itemsize,
        )

    def _load_textures(self) -> None:
        assert self._ctx is not None
        daymap = _load_image(EARTH_DAYMAP_FILE)
        if daymap is not None:
            self._earth_texture = self._ctx.texture(
                daymap.shape[1::-1],
                daymap.shape[2],
                daymap.tobytes(),
            )
            self._earth_texture.build_mipmaps()
            self._earth_texture.filter = (
                moderngl.LINEAR_MIPMAP_LINEAR,
                moderngl.LINEAR,
            )
        else:
            self._earth_texture = self._create_solid_texture((11, 42, 63))
        nightmap = _load_image(EARTH_NIGHTMAP_FILE)
        if nightmap is not None:
            self._night_texture = self._ctx.texture(
                nightmap.shape[1::-1],
                nightmap.shape[2],
                nightmap.tobytes(),
            )
            self._night_texture.build_mipmaps()
            self._night_texture.filter = (
                moderngl.LINEAR_MIPMAP_LINEAR,
                moderngl.LINEAR,
            )
        else:
            self._night_texture = None
        self._cloud_texture = None
        specular = _load_image(EARTH_SPECULAR_FILE)
        if specular is not None:
            self._specular_texture = self._ctx.texture(
                specular.shape[1::-1],
                specular.shape[2],
                specular.tobytes(),
            )
            self._specular_texture.build_mipmaps()
            self._specular_texture.filter = (
                moderngl.LINEAR_MIPMAP_LINEAR,
                moderngl.LINEAR,
            )
        else:
            self._specular_texture = None
        starfield = _load_image(STARFIELD_FILE)
        if starfield is not None:
            self._starfield_texture = self._ctx.texture(
                starfield.shape[1::-1],
                starfield.shape[2],
                starfield.tobytes(),
            )
            self._starfield_texture.build_mipmaps()
            self._starfield_texture.filter = (
                moderngl.LINEAR_MIPMAP_LINEAR,
                moderngl.LINEAR,
            )
        else:
            self._starfield_texture = self._create_solid_texture((5, 5, 15))
        self._fallback_textures["satellite"] = self._create_solid_texture((255, 176, 0))

    def _build_vaos(self) -> None:
        assert self._ctx is not None
        textured = self._programs["textured"]
        layout = "3f 3f 2f"
        attrs = ("in_pos", "in_normal", "in_uv")
        self._vaos["earth"] = self._ctx.vertex_array(
            textured,
            [(self._mesh_buffers["earth"].vbo, layout, *attrs)],
            self._mesh_buffers["earth"].ibo,
            index_element_size=self._mesh_buffers["earth"].index_element_size,
        )
        self._vaos["starfield"] = self._ctx.vertex_array(
            textured,
            [(self._mesh_buffers["starfield"].vbo, layout, *attrs)],
            self._mesh_buffers["starfield"].ibo,
            index_element_size=self._mesh_buffers["starfield"].index_element_size,
        )
        self._vaos["satellite"] = self._ctx.vertex_array(
            textured,
            [(self._mesh_buffers["satellite"].vbo, layout, *attrs)],
            self._mesh_buffers["satellite"].ibo,
            index_element_size=self._mesh_buffers["satellite"].index_element_size,
        )
        # Solid fill of the Earth body for wireframe mode: reuse the earth mesh
        # positions with the flat `color` program (skip the normal+uv attributes).
        self._vaos["earth_fill"] = self._ctx.vertex_array(
            self._programs["color"],
            [(self._mesh_buffers["earth"].vbo, "3f 5x4", "in_pos")],
            self._mesh_buffers["earth"].ibo,
            index_element_size=self._mesh_buffers["earth"].index_element_size,
        )

    def _create_solid_texture(self, color: tuple[int, ...]) -> moderngl.Texture:
        assert self._ctx is not None
        components = len(color)
        data = np.array(color, dtype=np.uint8).reshape(1, 1, components)
        texture = self._ctx.texture(
            (1, 1),
            components,
            data.tobytes(),
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture
    def _draw_starfield(self, view: np.ndarray) -> None:
        if (
            "starfield" not in self._vaos
            or self._starfield_texture is None
            or self._ctx is None
        ):
            return
        prog = self._programs["textured"]
        model = np.identity(4, dtype=np.float32)
        mvp = self._projection @ view @ model
        camera_pos = self._camera.position
        prog["mvp"].write(_gl_bytes(mvp))
        prog["model"].write(_gl_bytes(model))
        prog["light_dir"].value = (0.0, 0.0, 1.0)
        prog["light_pos"].value = (0.0, 0.0, 0.0)
        prog["camera_pos"].value = self._vec3_tuple(camera_pos)
        prog["sun_color"].value = (1.0, 1.0, 1.0)
        prog["ambient_color"].value = (1.0, 1.0, 1.0)
        prog["ambient_intensity"].value = 1.0
        prog["twilight_strength"].value = 0.0
        prog["twilight_exponent"].value = 1.0
        prog["specular_intensity"].value = 0.0
        prog["specular_exponent"].value = 1.0
        prog["point_falloff"].value = (0.0, 0.0)
        prog["light_type"].value = 0
        prog["use_specular_map"].value = False
        prog["specular_map"].value = 0
        prog["fade_alpha_with_light"].value = False
        prog["alpha_fade_min"].value = 0.0
        prog["uniform_lighting"].value = False
        prog["min_light"].value = 1.0
        prog["opacity"].value = 1.0
        prog["use_alpha"].value = False
        prog["use_night_map"].value = False
        prog["night_edge_softness"].value = 0.0
        prog["night_tex"].value = 0
        prog["tex"].value = 0
        self._starfield_texture.use(location=0)
        self._vaos["starfield"].render()

    def _draw_earth(self, view: np.ndarray) -> None:
        if "earth" not in self._vaos or self._earth_texture is None:
            return
        prog = self._programs["textured"]
        # Only rotate Earth geometry in inertial mode.
        if self._earth_rotation_mode == "ECI":
            model = _rotation_z(math.radians(self._earth_rotation_deg))
        else:
            model = np.identity(4, dtype=np.float32)
        mvp = self._projection @ view @ model
        sun_dir = self._active_sun_direction()
        sun_pos = self._sun_world_position(sun_dir)
        camera_pos = self._camera.position
        prog["mvp"].write(_gl_bytes(mvp))
        prog["model"].write(_gl_bytes(model))
        prog["light_dir"].value = self._vec3_tuple(sun_dir)
        prog["light_pos"].value = self._vec3_tuple(sun_pos)
        prog["camera_pos"].value = self._vec3_tuple(camera_pos)
        prog["sun_color"].value = self._sun_color_scaled()
        prog["ambient_color"].value = self._ambient_color
        prog["ambient_intensity"].value = float(self._ambient_intensity)
        prog["twilight_strength"].value = self._twilight_strength_scaled(
            self._twilight_strength
        )
        prog["twilight_exponent"].value = float(self._twilight_exponent)
        prog["specular_intensity"].value = float(self._specular_intensity)
        prog["specular_exponent"].value = float(self._specular_exponent)
        prog["point_falloff"].value = self._point_light_falloff
        prog["light_type"].value = 0 if self._light_model == "directional" else 1
        use_spec_map = self._specular_texture is not None
        prog["use_specular_map"].value = use_spec_map
        if use_spec_map:
            self._specular_texture.use(location=2)
            prog["specular_map"].value = 2
        else:
            prog["specular_map"].value = 0
        prog["fade_alpha_with_light"].value = False
        prog["alpha_fade_min"].value = 0.0
        prog["uniform_lighting"].value = self._uniform_lighting
        prog["min_light"].value = 0.2
        prog["opacity"].value = 1.0
        prog["use_alpha"].value = False
        prog["tex"].value = 0
        prog["night_edge_softness"].value = self._night_blend_softness
        use_night = self._night_texture is not None and self._day_night_enabled
        prog["use_night_map"].value = use_night
        if use_night:
            self._night_texture.use(location=1)
            prog["night_tex"].value = 1
        else:
            prog["night_tex"].value = 0
        prog["tex"].value = 0
        self._earth_texture.use(location=0)
        self._vaos["earth"].render()

    def _wireframe_model_matrix(self) -> np.ndarray:
        """Model matrix for the graticule, matching the textured Earth rotation."""
        if self._earth_rotation_mode == "ECI":
            return _rotation_z(math.radians(self._earth_rotation_deg))
        return np.identity(4, dtype=np.float32)

    def _draw_wireframe_fill(self, view: np.ndarray) -> None:
        """Draw the opaque Earth body so far-side grid/coastlines are occluded."""
        if self._ctx is None or "earth_fill" not in self._vaos:
            return
        prog = self._programs["color"]
        # Shrink slightly (uniform) so the radius-exact graticule/coastlines sit
        # just outside the fill surface and never z-fight with it.
        scale = np.identity(4, dtype=np.float32)
        scale[0, 0] = scale[1, 1] = scale[2, 2] = 0.997
        model = self._wireframe_model_matrix() @ scale
        mvp = self._projection @ view @ model
        prog["mvp"].write(_gl_bytes(mvp))
        prog["color"].value = self._wireframe_fill_color
        self._vaos["earth_fill"].render()

    def _draw_wireframe_earth(self, view: np.ndarray) -> None:
        if self._ctx is None:
            return
        prog = self._programs["color"]
        model = self._wireframe_model_matrix()
        mvp = self._projection @ view @ model
        prog["mvp"].write(_gl_bytes(mvp))
        self._ctx.line_width = 1

        groups = (
            (
                "graticule_grid",
                self._graticule_grid_vbo,
                self._graticule_grid_count,
                self._wireframe_grid_color,
            ),
            (
                "coastlines",
                self._coastline_vbo,
                self._coastline_count,
                self._wireframe_coastline_color,
            ),
            (
                "graticule_equator",
                self._graticule_equator_vbo,
                self._graticule_equator_count,
                self._wireframe_equator_color,
            ),
        )
        for key, vbo, count, color in groups:
            if vbo is None or count == 0:
                continue
            vao = self._vaos.get(key)
            if vao is None:
                vao = self._build_line_vao(key, vbo)
                self._vaos[key] = vao
            prog["color"].value = color
            vao.render(mode=moderngl.LINES, vertices=count)

    def _draw_terminator(self, view: np.ndarray) -> None:
        if self._ctx is None or not self._wireframe_show_terminator:
            return
        sun_dir = self._active_sun_direction()
        norm = float(np.linalg.norm(sun_dir))
        if not np.isfinite(norm) or norm <= 1e-6:
            return
        s = (sun_dir / norm).astype(np.float32)
        helper = (
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(s[2])) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        u = np.cross(s, helper)
        u /= max(float(np.linalg.norm(u)), 1e-6)
        w = np.cross(s, u)
        w /= max(float(np.linalg.norm(w)), 1e-6)
        a = np.linspace(0.0, 2.0 * math.pi, 129)
        pts = (
            self._radius_km
            * (np.outer(np.cos(a), u) + np.outer(np.sin(a), w))
        ).astype(np.float32)
        segments = _polyline_to_segments(pts)
        if segments.shape[0] == 0:
            return
        flat = segments.ravel()
        if self._terminator_vbo is None or self._terminator_vbo.size != flat.nbytes:
            if self._terminator_vbo is not None:
                self._terminator_vbo.release()
            self._terminator_vbo = self._ctx.buffer(flat.tobytes())
            self._vaos.pop("terminator", None)
        else:
            self._terminator_vbo.write(flat.tobytes())
        self._terminator_count = int(segments.shape[0])
        vao = self._vaos.get("terminator")
        if vao is None:
            vao = self._build_line_vao("terminator", self._terminator_vbo)
            self._vaos["terminator"] = vao
        # Terminator is computed in world coordinates already (no Earth rotation).
        prog = self._programs["color"]
        mvp = self._projection @ view @ np.identity(4, dtype=np.float32)
        prog["mvp"].write(_gl_bytes(mvp))
        prog["color"].value = self._wireframe_terminator_color
        self._ctx.line_width = 1
        vao.render(mode=moderngl.LINES, vertices=self._terminator_count)

    def _draw_clouds(self, view: np.ndarray) -> None:
        if (
            "clouds" not in self._vaos
            or self._cloud_texture is None
            or self._ctx is None
        ):
            return
        prog = self._programs["textured"]
        if self._earth_rotation_mode == "ECI":
            model = _rotation_z(math.radians(self._cloud_rotation_deg))
        else:
            model = np.identity(4, dtype=np.float32)
        mvp = self._projection @ view @ model
        sun_dir = self._active_sun_direction()
        sun_pos = self._sun_world_position(sun_dir)
        camera_pos = self._camera.position
        prog["mvp"].write(_gl_bytes(mvp))
        prog["model"].write(_gl_bytes(model))
        prog["light_dir"].value = self._vec3_tuple(sun_dir)
        prog["light_pos"].value = self._vec3_tuple(sun_pos)
        prog["camera_pos"].value = self._vec3_tuple(camera_pos)
        prog["sun_color"].value = self._sun_color_scaled()
        prog["ambient_color"].value = self._ambient_color
        prog["ambient_intensity"].value = float(self._ambient_intensity * 0.4)
        prog["twilight_strength"].value = self._twilight_strength_scaled(
            self._twilight_strength * 0.5
        )
        prog["twilight_exponent"].value = float(self._twilight_exponent)
        prog["specular_intensity"].value = float(self._specular_intensity * 0.35)
        prog["specular_exponent"].value = float(self._specular_exponent)
        prog["point_falloff"].value = self._point_light_falloff
        prog["light_type"].value = 0 if self._light_model == "directional" else 1
        prog["use_specular_map"].value = False
        prog["specular_map"].value = 0
        prog["fade_alpha_with_light"].value = True
        prog["alpha_fade_min"].value = 0.0
        prog["uniform_lighting"].value = self._uniform_lighting
        prog["min_light"].value = 0.2
        prog["opacity"].value = 0.65
        prog["use_alpha"].value = True
        prog["use_night_map"].value = False
        prog["night_edge_softness"].value = 0.0
        prog["night_tex"].value = 0
        prog["tex"].value = 0
        self._cloud_texture.use(location=0)
        self._vaos["clouds"].render()

    def _build_line_vao(self, key: str, buffer: moderngl.Buffer) -> moderngl.VertexArray:
        prog = self._programs["color"]
        return self._ctx.vertex_array(
            prog,
            [(buffer, "3f", "in_pos")],
        )

    def _build_vertex_color_line_vao(
        self, key: str, buffer: moderngl.Buffer
    ) -> moderngl.VertexArray:
        prog = self._programs["color_vertex"]
        return self._ctx.vertex_array(
            prog,
            [(buffer, "3f 4f", "in_pos", "in_color")],
        )

    def _draw_track(self, view: np.ndarray) -> None:
        if (
            self._track_vbo is None
            or self._track_vertex_count == 0
            or self._ctx is None
        ):
            return
        vao = self._vaos.get("track")
        if vao is None:
            vao = self._build_line_vao("track", self._track_vbo)
            self._vaos["track"] = vao
        prog = self._programs["color"]
        model = np.identity(4, dtype=np.float32)
        mvp = self._projection @ view @ model
        prog["mvp"].write(_gl_bytes(mvp))
        prog["color"].value = (1.0, 0.0, 0.0, 1.0)
        self._ctx.line_width = 3
        vao.render(mode=moderngl.LINE_STRIP, vertices=self._track_vertex_count)

    def _draw_sensor_outline(self, view: np.ndarray) -> None:
        if (
            self._sensor_outline_vbo is None
            or self._sensor_outline_vertex_count == 0
            or self._ctx is None
        ):
            return
        vao = self._vaos.get("sensor_outline")
        if vao is None:
            vao = self._build_vertex_color_line_vao("sensor_outline", self._sensor_outline_vbo)
            self._vaos["sensor_outline"] = vao
        prog = self._programs["color_vertex"]
        model = np.identity(4, dtype=np.float32)
        mvp = self._projection @ view @ model
        prog["mvp"].write(_gl_bytes(mvp))
        self._ctx.disable(moderngl.CULL_FACE)
        vao.render(mode=moderngl.TRIANGLES, vertices=self._sensor_outline_vertex_count)
        self._ctx.enable(moderngl.CULL_FACE)

    def _draw_arrow(self, view: np.ndarray) -> None:
        if (
            self._arrow_vbo is None
            or self._arrow_vertex_count == 0
            or self._ctx is None
        ):
            return
        vao = self._vaos.get("arrow")
        if vao is None:
            vao = self._build_line_vao("arrow", self._arrow_vbo)
            self._vaos["arrow"] = vao
        prog = self._programs["color"]
        model = np.identity(4, dtype=np.float32)
        mvp = self._projection @ view @ model
        prog["mvp"].write(_gl_bytes(mvp))
        prog["color"].value = self._arrow_color
        self._ctx.line_width = 4
        vao.render(mode=moderngl.LINES, vertices=self._arrow_vertex_count)

    def _draw_triad(self, view: np.ndarray) -> None:
        if self._ctx is None:
            return
        # Draw the triad as an overlay so the spacecraft geometry doesn't occlude it.
        self._ctx.disable(moderngl.DEPTH_TEST)
        prog = self._programs["color"]
        model = np.identity(4, dtype=np.float32)
        mvp = self._projection @ view @ model
        prog["mvp"].write(_gl_bytes(mvp))
        self._ctx.line_width = 4
        axis_specs = (
            ("x", (1.0, 0.0, 0.0, 1.0)),
            ("y", (0.0, 1.0, 0.0, 1.0)),
            ("z", (0.0, 0.0, 1.0, 1.0)),
        )
        for axis, color in axis_specs:
            vbo = self._triad_vbos.get(axis)
            count = int(self._triad_vertex_counts.get(axis, 0))
            if vbo is None or count == 0:
                continue
            vao_key = f"triad_{axis}"
            vao = self._vaos.get(vao_key)
            if vao is None:
                vao = self._build_line_vao(vao_key, vbo)
                self._vaos[vao_key] = vao
            prog["color"].value = color
            vao.render(mode=moderngl.LINES, vertices=count)
        self._ctx.enable(moderngl.DEPTH_TEST)

    def _draw_satellite(self, view: np.ndarray) -> None:
        if (
            not self._satellite_visible
            or "satellite" not in self._vaos
            or self._ctx is None
        ):
            return
        prog = self._programs["textured"]
        model = self._satellite_model
        mvp = self._projection @ view @ model
        sun_dir = self._active_sun_direction()
        sun_pos = self._sun_world_position(sun_dir)
        camera_pos = self._camera.position
        prog["mvp"].write(_gl_bytes(mvp))
        prog["model"].write(_gl_bytes(model))
        prog["light_dir"].value = self._vec3_tuple(sun_dir)
        prog["light_pos"].value = self._vec3_tuple(sun_pos)
        prog["camera_pos"].value = self._vec3_tuple(camera_pos)
        prog["sun_color"].value = self._sun_color_scaled()
        prog["ambient_color"].value = self._ambient_color
        prog["ambient_intensity"].value = float(self._ambient_intensity)
        prog["twilight_strength"].value = self._twilight_strength_scaled(
            self._twilight_strength * 0.2
        )
        prog["twilight_exponent"].value = float(self._twilight_exponent)
        prog["specular_intensity"].value = float(self._specular_intensity * 1.5)
        prog["specular_exponent"].value = float(self._specular_exponent)
        prog["point_falloff"].value = self._point_light_falloff
        prog["light_type"].value = 0 if self._light_model == "directional" else 1
        prog["use_specular_map"].value = False
        prog["specular_map"].value = 0
        prog["fade_alpha_with_light"].value = False
        prog["alpha_fade_min"].value = 0.0
        prog["uniform_lighting"].value = self._uniform_lighting
        prog["min_light"].value = 0.8
        prog["opacity"].value = 1.0
        prog["use_alpha"].value = False
        prog["use_night_map"].value = False
        prog["night_edge_softness"].value = 0.0
        prog["night_tex"].value = 0
        prog["tex"].value = 0
        texture = self._fallback_textures.get("satellite")
        if texture is None:
            texture = self._create_solid_texture((255, 176, 0))
            self._fallback_textures["satellite"] = texture
        texture.use(location=0)
        # Render satellite without face culling so we don't "lose" faces if a
        # transform temporarily produces a mirrored basis (or if mesh winding
        # differs across faces). Depth test still ensures correct occlusion.
        self._ctx.disable(moderngl.CULL_FACE)
        self._vaos["satellite"].render()
        self._ctx.enable(moderngl.CULL_FACE)

    def _draw_link(self, view: np.ndarray) -> None:
        if self._link_vbo is None or self._link_vertex_count == 0 or self._ctx is None:
            return
        self._ctx.disable(moderngl.DEPTH_TEST)
        vao = self._vaos.get("link")
        if vao is None:
            vao = self._build_line_vao("link", self._link_vbo)
            self._vaos["link"] = vao
        prog = self._programs["color"]
        model = np.identity(4, dtype=np.float32)
        mvp = self._projection @ view @ model
        prog["mvp"].write(_gl_bytes(mvp))
        prog["color"].value = (0.0, 0.902, 0.451, 1.0)
        self._ctx.line_width = 4
        vao.render(mode=moderngl.LINES, vertices=self._link_vertex_count)
        self._ctx.enable(moderngl.DEPTH_TEST)

