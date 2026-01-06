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

from src.ui.constants import (
    EARTH_CLOUDS_FILE,
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
    ("Earth cloud map", EARTH_CLOUDS_FILE),
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

    Applies +90° Z-rotation to align with globe coordinate system and negates Z
    to compensate for the shader's light direction negation.

    Args:
        x: ECEF X component of sun direction
        y: ECEF Y component of sun direction
        z: ECEF Z component of sun direction (+ = north, - = south)

    Returns:
        np.ndarray: Sun direction vector in globe coordinates
    """
    return np.array([-y, x, -z], dtype=np.float32)


def _load_image(path: Path, *, cache_result: bool = True) -> np.ndarray | None:
    if cache_result and path in _PRELOADED_IMAGES:
        return _PRELOADED_IMAGES[path]
    if not path.exists():
        if cache_result:
            _PRELOADED_IMAGES[path] = None
        return None
    try:
        data = mpl_image.imread(path)
    except Exception:
        if cache_result:
            _PRELOADED_IMAGES[path] = None
        return None
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
        self._track_vbo: moderngl.Buffer | None = None
        self._track_vertex_count = 0
        self._pending_track_coords: np.ndarray | None = None
        self._pending_track_clear = False
        self._link_vbo: moderngl.Buffer | None = None
        self._link_vertex_count = 0
        self._satellite_model = np.identity(4, dtype=np.float32)
        self._satellite_visible = False
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
        self._ctx.clear(0.02, 0.02, 0.04, 1.0)
        view = self._camera.view_matrix()
        self._flush_pending_track()
        self._flush_pending_arrow()
        self._draw_starfield(view)
        self._draw_earth(view)
        self._draw_clouds(view)
        self._draw_track(view)
        self._draw_arrow(view)
        self._draw_satellite(view)
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

    # ------------------------------------------------------------------
    # Public API for UI mixins
    # ------------------------------------------------------------------
    def set_frame_rotation(self, mode: str, angle_deg: float) -> None:
        """Update Earth rotation for the supplied reference frame."""
        self._earth_rotation_mode = mode
        self._earth_rotation_deg = angle_deg
        self._cloud_rotation_deg = angle_deg
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

    def set_sun_datetime(self, timestamp: datetime | None) -> None:
        """Set the datetime used for sun/season lighting (UTC)."""
        if timestamp is None:
            self._sun_datetime = None
        else:
            self._sun_datetime = timestamp.astimezone(timezone.utc)
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

    def update_satellite_position(
        self, position_km: Optional[tuple[float, float, float]]
    ) -> None:
        if position_km is None:
            self._satellite_model = np.identity(4, dtype=np.float32)
            self._satellite_visible = False
        else:
            self._satellite_model = _translation(position_km) @ _scale(200.0)
            self._satellite_visible = True
        self.update()

    def update_link_segment(
        self,
        start_km: Optional[tuple[float, float, float]],
        end_km: Optional[tuple[float, float, float]],
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
        norm = np.linalg.norm(direction)
        if norm < 1e-3:
            self._pending_arrow_vertices = np.zeros((0, 3), dtype=np.float32)
            self.update()
            return
        dir_norm = direction / norm
        length = 800.0
        tip = base + dir_norm * length
        side = np.cross(dir_norm, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if np.linalg.norm(side) < 1e-3:
            side = np.cross(dir_norm, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        side /= np.linalg.norm(side)
        head_length = length * 0.2
        head_width = length * 0.08
        left = tip - dir_norm * head_length + side * head_width
        right = tip - dir_norm * head_length - side * head_width
        vertices = np.array(
            [base, tip, tip, left, tip, right],
            dtype=np.float32,
        )
        self._pending_arrow_vertices = vertices
        self.update()

    def set_focus_point(
        self, position_km: Optional[tuple[float, float, float]]
    ) -> None:
        """Recenter the camera orbit target."""
        target = (0.0, 0.0, 0.0) if position_km is None else position_km
        self._camera.set_focus_point(target)
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
        day_of_year = now.timetuple().tm_yday
        seconds = (
            now.hour * 3600
            + now.minute * 60
            + now.second
            + now.microsecond / 1_000_000.0
        )
        year_fraction = ((day_of_year - 80) + seconds / 86400.0) / 365.2422
        solar_longitude = (2.0 * math.pi * year_fraction) % (2.0 * math.pi)
        axial_tilt = math.radians(23.44)
        x = math.cos(solar_longitude)
        y = math.sin(solar_longitude) * math.cos(axial_tilt)
        z = math.sin(solar_longitude) * math.sin(axial_tilt)
        dir_vec = ecef_sun_direction_to_globe(x, y, z)
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
        self._programs["textured"] = self._ctx.program(
            vertex_shader=textured_vs,
            fragment_shader=textured_fs,
        )
        self._programs["color"] = self._ctx.program(
            vertex_shader=color_vs,
            fragment_shader=color_fs,
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

        # Clouds with same flattening as Earth
        clouds_data = _generate_sphere(
            self._radius_km * 1.01,
            theta_segments=180,
            phi_segments=90,
            flattening=wgs84_flattening,
        )
        self._mesh_buffers["clouds"] = self._create_mesh_buffers(*clouds_data)

        # Satellite (spherical representation)
        sat_data = _generate_sphere(
            1.0,
            theta_segments=24,
            phi_segments=12,
        )
        self._mesh_buffers["satellite"] = self._create_mesh_buffers(*sat_data)

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
        clouds = _load_image(EARTH_CLOUDS_FILE)
        if clouds is not None:
            if clouds.shape[2] == 3:
                alpha = np.full((*clouds.shape[:2], 1), 255, dtype=np.uint8)
                clouds = np.concatenate([clouds, alpha], axis=-1)
            self._cloud_texture = self._ctx.texture(
                clouds.shape[1::-1],
                clouds.shape[2],
                clouds.tobytes(),
            )
            self._cloud_texture.build_mipmaps()
            self._cloud_texture.filter = (
                moderngl.LINEAR_MIPMAP_LINEAR,
                moderngl.LINEAR,
            )
        else:
            self._cloud_texture = self._create_solid_texture((255, 255, 255, 0))
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
        self._vaos["clouds"] = self._ctx.vertex_array(
            textured,
            [(self._mesh_buffers["clouds"].vbo, layout, *attrs)],
            self._mesh_buffers["clouds"].ibo,
            index_element_size=self._mesh_buffers["clouds"].index_element_size,
        )
        self._vaos["satellite"] = self._ctx.vertex_array(
            textured,
            [(self._mesh_buffers["satellite"].vbo, layout, *attrs)],
            self._mesh_buffers["satellite"].ibo,
            index_element_size=self._mesh_buffers["satellite"].index_element_size,
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
        model = _rotation_z(math.radians(self._earth_rotation_deg))
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

    def _draw_clouds(self, view: np.ndarray) -> None:
        if (
            "clouds" not in self._vaos
            or self._cloud_texture is None
            or self._ctx is None
        ):
            return
        prog = self._programs["textured"]
        model = _rotation_z(math.radians(self._cloud_rotation_deg))
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
        self._vaos["satellite"].render()

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

