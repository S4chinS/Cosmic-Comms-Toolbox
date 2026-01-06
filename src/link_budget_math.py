"""
Lightweight link budget math utilities.

All functions are pure and require callers to supply every parameter explicitly.
This keeps the calculations easy to reason about and simple to drive from the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Physical constants
EARTH_RADIUS_KM = 6378.137
BOLTZMANN_CONSTANT_DB = 228.6  # k in dB (J/K)

# DVB-S2 MODCOD reference table (subset from ETSI EN 302 307-2)
MODCOD_TABLE: Tuple[Tuple[str, float, float], ...] = (
    ("QPSK 1/4", 0.490243, -2.35),
    ("QPSK 1/3", 0.656448, -1.24),
    ("QPSK 2/5", 0.789412, -0.30),
    ("QPSK 1/2", 0.988858, 1.00),
    ("QPSK 3/5", 1.188304, 2.23),
    ("QPSK 2/3", 1.322253, 3.10),
    ("QPSK 3/4", 1.487473, 4.03),
    ("QPSK 4/5", 1.587196, 4.68),
    ("QPSK 5/6", 1.654663, 5.18),
    ("QPSK 8/9", 1.766451, 6.20),
    ("QPSK 9/10", 1.788612, 6.42),
    ("8PSK 3/5", 1.779991, 5.50),
    ("8PSK 2/3", 1.980636, 6.62),
    ("8PSK 3/4", 2.228124, 7.91),
    ("8PSK 5/6", 2.478562, 9.35),
    ("8PSK 8/9", 2.646012, 10.69),
    ("8PSK 9/10", 2.679207, 10.98),
    ("16APSK 2/3", 2.637201, 8.97),
    ("16APSK 3/4", 2.966728, 10.21),
    ("16APSK 4/5", 3.165623, 11.03),
    ("16APSK 5/6", 3.300184, 11.61),
    ("16APSK 8/9", 3.523143, 12.89),
    ("16APSK 9/10", 3.567342, 13.13),
    ("32APSK 3/4", 3.703295, 12.73),
    ("32APSK 4/5", 3.951571, 13.64),
    ("32APSK 5/6", 4.119540, 14.28),
    ("32APSK 8/9", 4.397854, 15.69),
    ("32APSK 9/10", 4.453027, 16.05),
)


@dataclass(frozen=True)
class ParameterRow:
    """Row entry used for the GUI link-budget table."""

    parameter: str
    value: str
    unit: str


def to_numpy(values: Sequence[float]) -> np.ndarray:
    """Convert any sequence to a 1-D NumPy array of floats."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")
    return arr


def slant_range_km(
    elevation_deg: Sequence[float],
    ground_altitude_km: float,
    satellite_altitude_km: Sequence[float] | float,
) -> np.ndarray:
    """
    Compute the slant range between a ground station and satellite.

    Args:
        elevation_deg: Elevation angles in degrees.
        ground_altitude_km: Ground station altitude above sea level.
        satellite_altitude_km: Satellite altitude(s) above sea level.
    """
    elevations = to_numpy(elevation_deg)
    altitude_sat_km = np.asarray(satellite_altitude_km, dtype=float)
    if altitude_sat_km.ndim == 0:
        altitude_sat_km = np.full_like(elevations, float(altitude_sat_km))
    if altitude_sat_km.shape != elevations.shape:
        raise ValueError("Satellite altitude array must match elevation array length.")

    sin_el = np.sin(np.radians(elevations))
    cos_el = np.cos(np.radians(elevations))
    term = (EARTH_RADIUS_KM + ground_altitude_km) * sin_el

    return -term + np.sqrt(
        (EARTH_RADIUS_KM + altitude_sat_km) ** 2
        - (EARTH_RADIUS_KM + ground_altitude_km) ** 2 * cos_el**2
    )


def free_space_path_loss_dB(
    slant_range_km: Sequence[float], frequency_GHz: float
) -> np.ndarray:
    """Return free-space path loss in dB."""
    slant = to_numpy(slant_range_km)
    if frequency_GHz <= 0:
        raise ValueError("Frequency must be positive.")
    return 20 * np.log10(slant) + 20 * np.log10(frequency_GHz) + 92.45


def select_modcod(
    esn0_dB: Sequence[float],
    margin_dB: float,
    modcod_table: Tuple[Tuple[str, float, float], ...] = MODCOD_TABLE,
) -> List[dict]:
    """
    Select the best MODCOD for each Es/N0 value after subtracting the link margin.

    Returns a list of dictionaries with modcod info or None if no mode closes.
    """
    esn0 = to_numpy(esn0_dB) - margin_dB
    ideal_thresholds = np.array([entry[2] for entry in modcod_table])
    bits_per_symbol = np.array([entry[1] for entry in modcod_table])

    result: List[dict] = []
    for value in esn0:
        valid_idx = np.where(ideal_thresholds <= value)[0]
        if valid_idx.size == 0:
            result.append({})
            continue
        best_idx = valid_idx[np.argmax(bits_per_symbol[valid_idx])]
        modcod_name, bits, threshold = modcod_table[best_idx]
        result.append(
            {
                "modcod": modcod_name,
                "bits_per_symbol": bits,
                "required_EsN0_dB": threshold,
                "margin_to_threshold_dB": value - threshold,
            }
        )
    return result


def calculate_link_budget(
    elevations_deg: Sequence[float],
    antenna_gains_dBi: Sequence[float],
    atmospheric_losses_dB: Sequence[float],
    *,
    tx_power_dBw: float,
    tx_boresight_gain_dBi: float,
    tx_losses_dB: float,
    tx_backoff_dB: float,
    frequency_GHz: float,
    satellite_altitude_km: Sequence[float] | float,
    ground_altitude_m: float,
    receiver_G_T_dB_K: float,
    receiver_losses_dB: float,
    symbol_rate_sps: float,
    implementation_loss_dB: float,
    margin_dB: float,
) -> dict:
    """
    Core link-budget routine that performs scalar/vector calculations.

    Returns a dictionary with intermediate terms and MODCOD/data rate info.
    """
    elevations = to_numpy(elevations_deg)
    antenna_gains = to_numpy(antenna_gains_dBi)
    atmospheric_losses = to_numpy(atmospheric_losses_dB)

    if not (len(elevations) == len(antenna_gains) == len(atmospheric_losses)):
        raise ValueError(
            "Elevation, antenna gain, and atmospheric loss arrays must match."
        )

    pointing_loss_dB = tx_boresight_gain_dBi - antenna_gains
    eirp_boresight_dBw = (
        tx_power_dBw + tx_boresight_gain_dBi - tx_losses_dB - tx_backoff_dB
    )
    eirp_actual_dBw = eirp_boresight_dBw - pointing_loss_dB

    slant_km = slant_range_km(
        elevations,
        ground_altitude_m / 1000.0,
        satellite_altitude_km,
    )
    fspl_dB = free_space_path_loss_dB(slant_km, frequency_GHz)

    p_rx_c_dBw = eirp_actual_dBw - fspl_dB - atmospheric_losses - receiver_losses_dB
    c_t_dBW_K = receiver_G_T_dB_K + p_rx_c_dBw
    c_n0_ideal_dBHz = c_t_dBW_K + BOLTZMANN_CONSTANT_DB
    c_n0_dBHz = c_n0_ideal_dBHz - implementation_loss_dB
    es_n0_dB = c_n0_dBHz - 10 * np.log10(symbol_rate_sps)

    modcod_info = select_modcod(es_n0_dB, margin_dB)
    spectral_efficiency = np.array(
        [entry.get("bits_per_symbol", 0.0) if entry else 0.0 for entry in modcod_info]
    )
    modcod_names = [
        entry.get("modcod", "No Link") if entry else "No Link" for entry in modcod_info
    ]
    required_esn0 = np.array(
        [
            entry.get("required_EsN0_dB", np.nan) if entry else np.nan
            for entry in modcod_info
        ]
    )
    margin_to_required = es_n0_dB - required_esn0

    data_rate_bps = spectral_efficiency * symbol_rate_sps
    data_rate_mbps = data_rate_bps / 1e6

    return {
        "eirp_actual_dBw": eirp_actual_dBw,
        "pointing_loss_dB": pointing_loss_dB,
        "slant_range_km": slant_km,
        "free_space_path_loss_dB": fspl_dB,
        "atmospheric_loss_dB": atmospheric_losses,
        "received_carrier_power_dBw": p_rx_c_dBw,
        "c_to_t_dBW_per_K": c_t_dBW_K,
        "c_to_n0_dBHz": c_n0_dBHz,
        "es_to_n0_dB": es_n0_dB,
        "modcod_names": modcod_names,
        "spectral_efficiency_bits_per_symbol": spectral_efficiency,
        "data_rate_mbps": data_rate_mbps,
        "required_EsN0_dB": required_esn0,
        "margin_to_required_EsN0_dB": margin_to_required,
    }


def integrate_data_volume_gb(
    time_seconds: Sequence[float], data_rate_mbps: Sequence[float]
) -> float:
    """
    Integrate the transmitted data volume (result in Gigabits).

    Args:
        time_seconds: Time axis in seconds.
        data_rate_mbps: Data rate profile in Mbps.
    """
    t = to_numpy(time_seconds)
    rates = to_numpy(data_rate_mbps)
    if len(t) != len(rates):
        raise ValueError("Time and data-rate arrays must match in length.")
    return float(np.trapezoid(rates, t) / 1000.0)


def _format_value(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return "—"
    return f"{value:.{decimals}f}"


def _format_negative(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    magnitude = abs(float(value))
    return f"-{magnitude:.{decimals}f}"


def _select_index(elevations: np.ndarray, evaluation_deg: float) -> int:
    if elevations.size == 0:
        raise ValueError("Elevation array cannot be empty.")
    deltas = np.abs(elevations - evaluation_deg)
    return int(np.argmin(deltas))


def build_parameter_rows(
    *,
    elevations_deg: Sequence[float],
    results: dict,
    evaluation_elevation_deg: float,
    elevation_lower_bound_deg: float,
    elevation_upper_bound_deg: float,
    tx_frequency_GHz: float,
    tx_power_dBw: float,
    tx_losses_dB: float,
    tx_boresight_gain_dBi: float,
    tx_backoff_dB: float,
    symbol_rate_sps: float,
    receiver_G_T_dB_K: float,
    implementation_loss_dB: float,
    margin_dB: float,
    rolloff: float,
    polarization_loss_dB: float,
    satellite_altitude_km: float,
    atmospheric_breakdown_dB: Dict[str, np.ndarray] | None = None,
) -> List[ParameterRow]:
    """
    Build the table rows shown in the GUI for a single evaluation elevation.
    """

    elevations = to_numpy(elevations_deg)
    index = _select_index(elevations, evaluation_elevation_deg)

    def _sample_vector(key: str) -> float:
        data = np.asarray(results[key])
        if data.ndim == 0:
            return float(data)
        return float(data[index])

    eval_elevation = float(elevations[index])
    slant_km = _sample_vector("slant_range_km")
    fspl_dB = _sample_vector("free_space_path_loss_dB")
    atm_loss_dB = _sample_vector("atmospheric_loss_dB")
    pointing_loss_dB = _sample_vector("pointing_loss_dB")
    c_n0_dBHz = _sample_vector("c_to_n0_dBHz")
    modcod_name = results["modcod_names"][index]
    spectral_efficiency = _sample_vector("spectral_efficiency_bits_per_symbol")
    required_esn0_dB = _sample_vector("required_EsN0_dB")

    eirp_boresight_dBw = (
        tx_power_dBw + tx_boresight_gain_dBi - tx_losses_dB - tx_backoff_dB
    )
    total_propagation_loss_dB = (
        fspl_dB + atm_loss_dB + pointing_loss_dB + polarization_loss_dB
    )
    rx_power_dBw = eirp_boresight_dBw - total_propagation_loss_dB

    modulation_parts = modcod_name.split()
    modulation_type = modulation_parts[0] if modulation_parts else "No Link"
    coding_rate = modulation_parts[1] if len(modulation_parts) > 1 else "—"

    def _component_value(name: str) -> float | None:
        if not atmospheric_breakdown_dB or name not in atmospheric_breakdown_dB:
            return None
        return float(np.asarray(atmospheric_breakdown_dB[name])[index])

    def _modulation_order(modulation: str) -> str:
        if modulation in ("No", "No Link"):
            return "—"
        digits = "".join(ch for ch in modulation if ch.isdigit())
        if digits:
            return digits
        mapping = {"BPSK": "2", "QPSK": "4"}
        return mapping.get(modulation.upper(), "—")

    modulation_order = _modulation_order(modulation_type)

    has_link = (
        modcod_name != "No Link"
        and not np.isnan(required_esn0_dB)
        and spectral_efficiency > 0.0
    )
    spectral_efficiency_dB = None
    if spectral_efficiency > 0.0:
        spectral_efficiency_dB = 10 * np.log10(spectral_efficiency)

    tx_symbol_limit_Msym = symbol_rate_sps / 1e6

    if has_link:
        max_symbol_rate_dBHz = c_n0_dBHz - required_esn0_dB - margin_dB
        max_symbol_rate_Hz = 10 ** (max_symbol_rate_dBHz / 10.0)
        actual_symbol_rate_Hz = min(max_symbol_rate_Hz, symbol_rate_sps)
        actual_symbol_rate_dBHz = 10 * np.log10(actual_symbol_rate_Hz)
        actual_symbol_rate_Msym = actual_symbol_rate_Hz / 1e6
        spectral_efficiency_value = spectral_efficiency
        max_info_rate_dBHz = None
        max_info_rate_Mbps = None
        occupied_bw_MHz = None

        if spectral_efficiency > 0.0:
            max_info_rate_dBHz = actual_symbol_rate_dBHz + spectral_efficiency_dB
            max_info_rate_Hz = 10 ** (max_info_rate_dBHz / 10.0)
            max_info_rate_Mbps = max_info_rate_Hz / 1e6
            occupied_bw_MHz = actual_symbol_rate_Msym * (1.0 + rolloff)

        actual_symbol_rate_dBHz = (
            10 * np.log10(symbol_rate_sps)
            if actual_symbol_rate_Hz == symbol_rate_sps
            else actual_symbol_rate_dBHz
        )
        actual_es_n0_dB = c_n0_dBHz - actual_symbol_rate_dBHz
        link_margin_dB = actual_es_n0_dB - required_esn0_dB
        surplus_margin_dB = link_margin_dB - margin_dB
    else:
        max_symbol_rate_dBHz = None
        actual_symbol_rate_Msym = None
        spectral_efficiency_value = None
        max_info_rate_dBHz = None
        max_info_rate_Mbps = None
        occupied_bw_MHz = None
        link_margin_dB = None
        surplus_margin_dB = None

    row_values: Dict[str, ParameterRow] = {}

    def _set(label: str, value: str, unit: str) -> None:
        row_values[label] = ParameterRow(label, value, unit)

    _set("TX Frequency", _format_value(tx_frequency_GHz), "GHz")
    _set("TX Power", _format_value(tx_power_dBw), "dBW")
    _set("TX Losses", _format_negative(tx_losses_dB), "dB")
    _set("TX Antenna Boresight Gain", _format_value(tx_boresight_gain_dBi), "dBi")
    _set("EIRP", _format_value(eirp_boresight_dBw), "dBW")
    _set("Spacecraft Altitude", _format_value(satellite_altitude_km, 0), "km")
    _set("GS Elevation Angle", _format_value(eval_elevation, 1), "deg")
    _set("Elevation Lower Bound", _format_value(elevation_lower_bound_deg, 1), "deg")
    _set("Elevation Upper Bound", _format_value(elevation_upper_bound_deg, 1), "deg")
    _set("Slant Range", _format_value(slant_km, 0), "km")
    _set("Free Space Path Loss (FSPL)", _format_value(-fspl_dB), "dB")

    for label, key in [
        ("Gaseous Loss", "gaseous"),
        ("Cloud Loss", "cloud"),
        ("Rain Loss", "rain"),
        ("Scintillation Loss", "scintillation"),
    ]:
        comp_val = _component_value(key)
        _set(label, _format_negative(comp_val) if comp_val is not None else "—", "dB")

    _set("Total Atmospheric Loss", _format_value(-atm_loss_dB), "dB")
    _set("Polarization Loss", _format_negative(polarization_loss_dB), "dB")
    _set("TX Antenna Pointing Loss", _format_negative(pointing_loss_dB), "dB")
    _set("Total Propagation Losses", _format_value(-total_propagation_loss_dB), "dB")
    _set("Received Signal Power", _format_value(rx_power_dBw), "dBW")
    _set("Ground Station G/T", _format_value(receiver_G_T_dB_K), "dB/K")
    _set("Implementation Loss", _format_negative(implementation_loss_dB), "dB")
    _set("C/N0", _format_value(c_n0_dBHz), "dBHz")
    _set("Modulation Type", modulation_type, "")
    _set("Modulation Order", modulation_order, "")
    _set("Coding Rate", coding_rate, "")
    _set("Required Es/N0", _format_value(required_esn0_dB), "dB")
    _set("Required Link Margin", _format_value(margin_dB), "dB")
    _set("Maximum Symbol Rate", _format_value(max_symbol_rate_dBHz), "dBHz")
    _set("TX Symbol Limit", _format_value(tx_symbol_limit_Msym), "Msym/s")
    _set("Channel Symbol Rate", _format_value(actual_symbol_rate_Msym), "Msym/s")
    _set("Spectral Efficiency", _format_value(spectral_efficiency_value), "bit/sym")
    _set("Maximum Information Rate", _format_value(max_info_rate_dBHz), "dBHz")
    _set("Roll-off Factor", _format_value(rolloff), "")
    _set("Occupied Bandwidth", _format_value(occupied_bw_MHz), "MHz")
    _set("Max. Information Rate", _format_value(max_info_rate_Mbps), "Mbps")
    _set("Link Margin", _format_value(link_margin_dB), "dB")
    _set("Surplus Link Margin", _format_value(surplus_margin_dB), "dB")

    desired_order = [
        "TX Frequency",
        "TX Power",
        "TX Losses",
        "TX Antenna Boresight Gain",
        "EIRP",
        "Spacecraft Altitude",
        "GS Elevation Angle",
        "Elevation Lower Bound",
        "Elevation Upper Bound",
        "Slant Range",
        "Free Space Path Loss (FSPL)",
        "Gaseous Loss",
        "Cloud Loss",
        "Rain Loss",
        "Scintillation Loss",
        "Total Atmospheric Loss",
        "Polarization Loss",
        "TX Antenna Pointing Loss",
        "Total Propagation Losses",
        "Received Signal Power",
        "Ground Station G/T",
        "Implementation Loss",
        "C/N0",
        "Modulation Type",
        "Modulation Order",
        "Coding Rate",
        "Required Es/N0",
        "Required Link Margin",
        "Maximum Symbol Rate",
        "TX Symbol Limit",
        "Channel Symbol Rate",
        "Spectral Efficiency",
        "Maximum Information Rate",
        "Roll-off Factor",
        "Occupied Bandwidth",
        "Max. Information Rate",
        "Link Margin",
        "Required Link Margin",
        "Surplus Link Margin",
    ]

    rows: List[ParameterRow] = []
    for label in desired_order:
        if label in row_values:
            rows.append(row_values[label])
        else:
            rows.append(ParameterRow(label, "—", ""))

    return rows
