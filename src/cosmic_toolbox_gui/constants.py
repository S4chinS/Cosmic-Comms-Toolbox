"""Shared UI constants for the Cosmic Comms Toolbox."""

from __future__ import annotations

from pathlib import Path

from cosmic_toolbox.paths import package_resources_root
from cosmic_toolbox.models import GroundStationConfig

HIST_BIN_OPTIONS = [1, 5, 15, 60]  # seconds
STATION_COLOR_PALETTE = [
    "#2E8B57",
    "#1E90FF",
    "#FF8C00",
    "#8A2BE2",
    "#DC143C",
    "#20B2AA",
    "#FF1493",
    "#708090",
]
MAP_OCEAN_COLOR = "#020b1b"
MAP_LAND_COLOR = "#0c2a3f"
MAP_COASTLINE_COLOR = "#3cc3ff"
MAP_GRID_COLOR = "#3cc3ff"
DEFAULT_STATION_DIR = package_resources_root() / "groundstation_list"
DEFAULT_MANUAL_STATION = GroundStationConfig(
    name="Svalbard (Ny-Alesund)",
    latitude_deg=78.92,
    longitude_deg=11.93,
    altitude_m=30.0,
)
# GUI-only textures live in the GUI resource directory.
TEXTURE_DIR = Path(__file__).resolve().parent / "resources" / "textures"
EARTH_DAYMAP_FILE = TEXTURE_DIR / "8k_earth_daymap.jpg"
EARTH_NIGHTMAP_FILE = TEXTURE_DIR / "8k_earth_nightmap.jpg"
EARTH_CLOUDS_FILE = TEXTURE_DIR / "8k_earth_clouds.png"
EARTH_SPECULAR_FILE = TEXTURE_DIR / "8k_earth_specular_map.tif"
STARFIELD_FILE = TEXTURE_DIR / "8k_stars_milky_way.jpg"
EARTH_ROTATION_RATE_RAD_PER_SEC = 7.2921159e-5

