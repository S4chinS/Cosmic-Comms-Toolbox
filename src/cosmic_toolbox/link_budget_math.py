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
_NOISE_FIGURE_REF_TEMP_K = 290.0  # IEEE/ITU noise-figure reference temperature

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

QPSK_VITERBI_MODE_NAME = "QPSK Viterbi (k=7,r=1/2)"
QPSK_VITERBI_BITS_PER_SYMBOL = 1.0
QPSK_VITERBI_REQUIRED_ESN0_DB = 4.5

# Custom QPSK r=7/8 mode (non-DVB-S2).
# Es/N0 = Eb/N0 + 10*log10(bits_per_symbol) = 6.5 + 10*log10(2 * 7/8) = 8.93 dB
QPSK_78_MODE_NAME = "QPSK 7/8"
QPSK_78_BITS_PER_SYMBOL = 1.75  # 2 bits/symbol * rate 7/8
QPSK_78_REQUIRED_ESN0_DB = 8.93

FIXED_LINK_MODE_TABLE: Tuple[Tuple[str, float, float], ...] = MODCOD_TABLE + (
    (
        QPSK_VITERBI_MODE_NAME,
        QPSK_VITERBI_BITS_PER_SYMBOL,
        QPSK_VITERBI_REQUIRED_ESN0_DB,
    ),
    (
        QPSK_78_MODE_NAME,
        QPSK_78_BITS_PER_SYMBOL,
        QPSK_78_REQUIRED_ESN0_DB,
    ),
)


@dataclass(frozen=True)
class ParameterRow:
    """Row entry used for the GUI link-budget table."""

    parameter: str
    value: str
    unit: str
    color: str | None = None
    is_calc: bool = False


def to_numpy(values: Sequence[float]) -> np.ndarray:
    """Convert any sequence to a 1-D NumPy array of floats."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")
    return arr


def system_noise_temperature_K(
    atmospheric_losses_dB: np.ndarray,
    *,
    receiver_noise_figure_dB: float,
    sky_background_temperature_K: float,
    atmospheric_physical_temperature_K: float = 290.0,
) -> np.ndarray:
    """
    Compute system noise temperature at each sample in the pass.

    Sky brightness temperature is coupled to atmospheric absorption: a lossy
    atmosphere both attenuates the signal and re-radiates thermal noise,
    raising the effective antenna noise temperature.

    Args:
        atmospheric_losses_dB: One-way atmospheric loss (positive, dB) per sample.
        receiver_noise_figure_dB: LNA/receiver noise figure in dB.
        sky_background_temperature_K: Cold-sky background temperature (K).
            Typical value: 50 K for clear sky; 290 K pointing at warm ground.
        atmospheric_physical_temperature_K: Physical temperature of the absorbing
            atmosphere (K). Defaults to 290 K.
    """
    atm_loss_linear = 10.0 ** (atmospheric_losses_dB / 10.0)
    t_brightness = (
        atmospheric_physical_temperature_K * (1.0 - 1.0 / atm_loss_linear)
        + sky_background_temperature_K / atm_loss_linear
    )
    t_receiver = _NOISE_FIGURE_REF_TEMP_K * (
        10.0 ** (receiver_noise_figure_dB / 10.0) - 1.0
    )
    return t_brightness + t_receiver


def system_g_over_t_dB_K(
    atmospheric_losses_dB: np.ndarray,
    *,
    rx_antenna_gain_dBi: float,
    receiver_noise_figure_dB: float,
    sky_background_temperature_K: float,
    atmospheric_physical_temperature_K: float = 290.0,
) -> np.ndarray:
    """
    Return system G/T in dB/K as an elevation-varying vector.

    G/T varies with elevation because atmospheric absorption raises the sky
    brightness temperature seen by the antenna.
    """
    t_sys = system_noise_temperature_K(
        atmospheric_losses_dB,
        receiver_noise_figure_dB=receiver_noise_figure_dB,
        sky_background_temperature_K=sky_background_temperature_K,
        atmospheric_physical_temperature_K=atmospheric_physical_temperature_K,
    )
    return rx_antenna_gain_dBi - 10.0 * np.log10(t_sys)


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
    Select the best MODCOD for each Es/N0 value after subtracting the link margin (VCM).

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


def _modcod_index_by_name(
    modcod_name: str,
    modcod_table: Tuple[Tuple[str, float, float], ...] = FIXED_LINK_MODE_TABLE,
) -> int:
    """Return table index for the given MODCOD name. Raises ValueError if not found."""
    for i, (name, _bits, _thresh) in enumerate(modcod_table):
        if name == modcod_name:
            return i
    raise ValueError(f"MODCOD not found: {modcod_name!r}")


def select_modcod_fixed(
    esn0_dB: Sequence[float],
    margin_dB: float,
    modcod_name: str,
    modcod_table: Tuple[Tuple[str, float, float], ...] = FIXED_LINK_MODE_TABLE,
) -> List[dict]:
    """
    Apply a single MODCOD: link closes only where Es/N0 >= threshold + margin.

    Returns a list of dicts with modcod info where link closes, else empty dict.
    """
    idx = _modcod_index_by_name(modcod_name, modcod_table)
    modcod_name, bits, threshold = modcod_table[idx]
    esn0 = to_numpy(esn0_dB) - margin_dB
    result: List[dict] = []
    for value in esn0:
        if value >= threshold:
            result.append(
                {
                    "modcod": modcod_name,
                    "bits_per_symbol": bits,
                    "required_EsN0_dB": threshold,
                    "margin_to_threshold_dB": value - threshold,
                }
            )
        else:
            result.append({})
    return result


def required_esn0_for_mode(
    modcod_name: str,
    modcod_table: Tuple[Tuple[str, float, float], ...] = FIXED_LINK_MODE_TABLE,
) -> float:
    """Return the required Es/N0 threshold for a fixed link mode."""
    idx = _modcod_index_by_name(modcod_name, modcod_table)
    return float(modcod_table[idx][2])


@dataclass(frozen=True)
class CcmOptimizerResult:
    """Result of per-pass CCM optimization."""

    start_offset_s: float
    end_offset_s: float
    modcod_name: str
    volume_gb: float
    rate_mbps: np.ndarray  # full pass length: rate in window, 0 outside


def optimize_ccm_per_pass(
    time_seconds: Sequence[float],
    es_n0_dB: Sequence[float],
    margin_dB: float,
    symbol_rate_sps: float,
    offset_step_s: float = 10.0,
    max_offset_s: float = 300.0,
    modcod_table: Tuple[Tuple[str, float, float], ...] = MODCOD_TABLE,
) -> CcmOptimizerResult:
    """
    For a single pass, find (start_offset, end_offset, MODCOD) that maximize data volume.

    Offsets are in seconds: transmit window is [AOS + start_offset_s, LOS - end_offset_s].
    Valid windows require AOS + start_offset_s < LOS - end_offset_s.
    Uses grid search over offsets and MODCOD list. Returns the best result.
    """
    t = to_numpy(time_seconds)
    esn0 = to_numpy(es_n0_dB)
    if len(t) != len(esn0):
        raise ValueError("time_seconds and es_n0_dB must have the same length.")
    if len(t) == 0:
        raise ValueError("Pass must have at least one sample.")
    t0 = float(t[0])
    t1 = float(t[-1])
    duration = t1 - t0
    if duration <= 0:
        raise ValueError("Pass duration must be positive.")

    best_volume = -1.0
    best_start = 0.0
    best_end = 0.0
    best_modcod_name = "No Link"
    best_bits = 0.0

    step = max(1e-6, float(offset_step_s))
    max_off = max(0.0, float(max_offset_s))

    start_offsets = np.arange(0.0, min(max_off, duration - step) + 1e-9, step)
    end_offsets = np.arange(0.0, min(max_off, duration - step) + 1e-9, step)

    for start_off in start_offsets:
        for end_off in end_offsets:
            t_start = t0 + start_off
            t_end = t1 - end_off
            if t_start >= t_end:
                continue
            window_mask = (t >= t_start) & (t <= t_end)
            if not np.any(window_mask):
                continue
            min_esn0 = float(np.min(esn0[window_mask]))
            for modcod_name, bits, threshold in modcod_table:
                if min_esn0 < threshold + margin_dB:
                    continue
                rate_bps = bits * symbol_rate_sps
                rate_mbps = rate_bps / 1e6
                dur_win = t_end - t_start
                vol_gb = rate_mbps * dur_win / 1000.0
                if vol_gb > best_volume:
                    best_volume = vol_gb
                    best_start = start_off
                    best_end = end_off
                    best_modcod_name = modcod_name
                    best_bits = bits
    if best_volume < 0.0:
        return CcmOptimizerResult(
            start_offset_s=0.0,
            end_offset_s=0.0,
            modcod_name="No Link",
            volume_gb=0.0,
            rate_mbps=np.zeros_like(t, dtype=float),
        )

    rate_mbps_const = (best_bits * symbol_rate_sps) / 1e6
    t_start = t0 + best_start
    t_end = t1 - best_end
    window_mask = (t >= t_start) & (t <= t_end)
    rate_series = np.zeros_like(t, dtype=float)
    rate_series[window_mask] = rate_mbps_const

    return CcmOptimizerResult(
        start_offset_s=best_start,
        end_offset_s=best_end,
        modcod_name=best_modcod_name,
        volume_gb=best_volume,
        rate_mbps=rate_series,
    )


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
    rx_antenna_gain_dBi: float = 0.0,
    receiver_noise_figure_dB: float = 0.0,
    sky_background_temperature_K: float | None = None,
    receiver_losses_dB: float = 0.0,
    polarization_loss_dB: float = 0.0,
    symbol_rate_sps: float,
    implementation_loss_dB: float,
    margin_dB: float,
    fixed_modcod_name: str | None = None,
    gs_gt_dBK: float | None = None,
) -> dict:
    """
    Core link-budget routine that performs scalar/vector calculations.

    Receiver noise quality can be specified either via three physically
    meaningful inputs (rx_antenna_gain_dBi, receiver_noise_figure_dB,
    sky_background_temperature_K) or directly as a fixed ground-station G/T
    figure (gs_gt_dBK, dB/K).  When gs_gt_dBK is provided it takes precedence
    and the three individual receiver parameters are ignored.

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

    p_rx_c_dBw = (
        eirp_actual_dBw
        - fspl_dB
        - atmospheric_losses
        - polarization_loss_dB
        - receiver_losses_dB
    )
    if gs_gt_dBK is not None:
        receiver_G_T_dB_K = np.full_like(atmospheric_losses, float(gs_gt_dBK))
        t_sys = np.full_like(atmospheric_losses, np.nan)
    else:
        if sky_background_temperature_K is None:
            raise ValueError(
                "calculate_link_budget requires sky_background_temperature_K when "
                "gs_gt_dBK is not supplied (there is no physically meaningful default)."
            )
        t_sys = system_noise_temperature_K(
            atmospheric_losses,
            receiver_noise_figure_dB=receiver_noise_figure_dB,
            sky_background_temperature_K=sky_background_temperature_K,
        )
        receiver_G_T_dB_K = rx_antenna_gain_dBi - 10.0 * np.log10(t_sys)
    c_t_dBW_K = receiver_G_T_dB_K + p_rx_c_dBw
    c_n0_ideal_dBHz = c_t_dBW_K + BOLTZMANN_CONSTANT_DB
    c_n0_dBHz = c_n0_ideal_dBHz - implementation_loss_dB
    es_n0_dB = c_n0_dBHz - 10 * np.log10(symbol_rate_sps)

    if fixed_modcod_name is not None:
        modcod_info = select_modcod_fixed(
            es_n0_dB, margin_dB, fixed_modcod_name
        )
    else:
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
        "system_noise_temperature_K": t_sys,
        "receiver_G_T_dB_K": receiver_G_T_dB_K,
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
    rx_antenna_gain_dBi: float = 0.0,
    receiver_noise_figure_dB: float = 0.0,
    sky_background_temperature_K: float | None = None,
    implementation_loss_dB: float = 0.0,
    margin_dB: float = 0.0,
    rolloff: float = 0.25,
    polarization_loss_dB: float = 0.0,
    satellite_altitude_km: float = 0.0,
    atmospheric_breakdown_dB: Dict[str, np.ndarray] | None = None,
    gs_gt_dBK: float | None = None,
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
    receiver_G_T_dB_K = _sample_vector("receiver_G_T_dB_K")
    system_noise_temp_K = _sample_vector("system_noise_temperature_K")
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

    if modcod_name == QPSK_VITERBI_MODE_NAME:
        modulation_type = "QPSK"
        coding_rate = _format_value(QPSK_VITERBI_BITS_PER_SYMBOL / 2.0, 3)
    else:
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

    def _set(
        label: str,
        value: str,
        unit: str,
        color: str | None = None,
        *,
        is_calc: bool = False,
    ) -> None:
        row_values[label] = ParameterRow(label, value, unit, color, is_calc)

    def _format_rate_row(rate_mbps: float | None) -> tuple[str, str]:
        if rate_mbps is None:
            return ("—", "")
        rate = float(rate_mbps)
        if not np.isfinite(rate):
            return ("—", "")
        rate_bps = rate * 1e6
        unit_name = "bit"
        unit_scale = 1.0
        for candidate_name, candidate_scale in (
            ("bit", 1.0),
            ("kbit", 1e3),
            ("Mbit", 1e6),
            ("Gbit", 1e9),
            ("Tbit", 1e12),
        ):
            if abs(rate_bps) >= candidate_scale:
                unit_name = candidate_name
                unit_scale = candidate_scale
        return (_format_value(rate_bps / unit_scale), f"{unit_name}/s")

    # ── Input rows (values entered directly by the user) ──────────────────
    _set("Earth Radius", _format_value(EARTH_RADIUS_KM, 3), "km")
    _set("Spacecraft Altitude", _format_value(satellite_altitude_km, 0), "km")
    _set("Elevation Angle", _format_value(eval_elevation, 1), "deg")
    _set("Elevation Lower Bound", _format_value(elevation_lower_bound_deg, 1), "deg")
    _set("Elevation Upper Bound", _format_value(elevation_upper_bound_deg, 1), "deg")
    _set("TX Frequency", _format_value(tx_frequency_GHz), "GHz")
    _set("TX Power", _format_value(tx_power_dBw), "dBW")
    _set("TX Losses", _format_negative(tx_losses_dB), "dB")
    _set("TX Antenna Boresight Gain", _format_value(tx_boresight_gain_dBi), "dBi")
    _set("Polarization Loss", _format_negative(polarization_loss_dB), "dB")
    if gs_gt_dBK is not None:
        _set("Ground Station G/T", _format_value(gs_gt_dBK), "dB/K")
    else:
        _set("Sky Background Temperature", _format_value(sky_background_temperature_K, 0), "K")
        _set("Receiver Noise Figure", _format_value(receiver_noise_figure_dB), "dB")
        _set("Rx Antenna Gain", _format_value(rx_antenna_gain_dBi), "dBi")
    _set("Implementation Loss", _format_negative(implementation_loss_dB), "dB")
    _set("Required Link Margin", _format_value(margin_dB), "dB")
    _set("TX Symbol Limit", _format_value(tx_symbol_limit_Msym), "Msym/s")
    _set("Roll-off Factor", _format_value(rolloff), "")

    # ── Calc rows (values derived from the link-budget calculation) ────────
    _set("EIRP", _format_value(eirp_boresight_dBw), "dBW", is_calc=True)
    _set("Slant Range", _format_value(slant_km, 0), "km", is_calc=True)
    _set("Free Space Path Loss (FSPL)", _format_value(-fspl_dB), "dB", is_calc=True)

    for label, key in [
        ("Gaseous Loss", "gaseous"),
        ("Cloud Loss", "cloud"),
        ("Rain Loss", "rain"),
        ("Scintillation Loss", "scintillation"),
    ]:
        comp_val = _component_value(key)
        _set(
            label,
            _format_negative(comp_val) if comp_val is not None else "—",
            "dB",
            is_calc=True,
        )

    _set("Total Atmospheric Loss", _format_value(-atm_loss_dB), "dB", is_calc=True)
    _set("TX Antenna Pointing Loss", _format_negative(pointing_loss_dB), "dB", is_calc=True)
    _set("Total Propagation Losses", _format_value(-total_propagation_loss_dB), "dB", is_calc=True)
    _set("Received Signal Power", _format_value(rx_power_dBw), "dBW", is_calc=True)
    if gs_gt_dBK is None:
        _set("System Noise Temperature", _format_value(system_noise_temp_K, 0), "K", is_calc=True)
    _set("Ground Station G/T", _format_value(receiver_G_T_dB_K), "dB/K", is_calc=True)
    _set("C/N0", _format_value(c_n0_dBHz), "dBHz", is_calc=True)
    _set("Modulation Type", modulation_type, "", is_calc=True)
    _set("Modulation Order", modulation_order, "", is_calc=True)
    _set("Coding Rate", coding_rate, "", is_calc=True)
    _set("Required Es/N0", _format_value(required_esn0_dB), "dB", is_calc=True)
    _set("Maximum Symbol Rate", _format_value(max_symbol_rate_dBHz), "dBHz", is_calc=True)
    _set("Channel Symbol Rate", _format_value(actual_symbol_rate_Msym), "Msym/s", is_calc=True)
    _set("Spectral Efficiency", _format_value(spectral_efficiency_value), "bit/sym", is_calc=True)
    _set("Maximum Information Rate", _format_value(max_info_rate_dBHz), "dBHz", is_calc=True)
    _set("Occupied Bandwidth", _format_value(occupied_bw_MHz), "MHz", is_calc=True)
    max_info_rate_value, max_info_rate_unit = _format_rate_row(max_info_rate_Mbps)
    _set("Max. Information Rate", max_info_rate_value, max_info_rate_unit, is_calc=True)
    _set("Link Margin", _format_value(link_margin_dB), "dB", is_calc=True)
    _set("Surplus Link Margin", _format_value(surplus_margin_dB), "dB", is_calc=True)

    # PFD rows — computed at the evaluation elevation.  ITU PFD limits are only
    # defined for specific frequency bands; outside those bands the mask spec
    # lookup raises and PFD is legitimately "N/A".  Any other error (bad symbol
    # rate, slant range, elevation, etc.) is a real problem and must surface
    # rather than being masked as "N/A".
    from cosmic_toolbox import pfd_math as _pfd_math

    _pfd_freq_mhz = tx_frequency_GHz * 1000.0
    try:
        _pfd_spec = _pfd_math.itu_surface_pfd_mask_spec(_pfd_freq_mhz)
    except ValueError:
        _pfd_spec = None

    if _pfd_spec is None:
        _set("PFD", "N/A", "", is_calc=True)
        _set("PFD Limit", "N/A", "", is_calc=True)
        _set("PFD Margin", "N/A", "", is_calc=True)
    else:
        _occ_bw_hz = float(symbol_rate_sps) * (1.0 + float(rolloff))
        _slant_m = slant_km * 1000.0
        _spreading_dB = 10.0 * np.log10(4.0 * np.pi * _slant_m**2)
        _bw_norm_dB = 10.0 * np.log10(_occ_bw_hz / _pfd_spec.reference_bandwidth_hz)
        _eirp_eval = _sample_vector("eirp_actual_dBw")
        _pfd_val = _eirp_eval - _spreading_dB - _bw_norm_dB
        _pfd_limit_val = float(
            _pfd_math.itu_surface_pfd_limit_dBW_per_m2(
                [eval_elevation], frequency_MHz=_pfd_freq_mhz
            )[0]
        )
        _pfd_margin_val = _pfd_limit_val - _pfd_val
        _ref_bw_khz = _pfd_spec.reference_bandwidth_hz / 1000.0
        _pfd_unit = f"dBW/m\u00b2/{_ref_bw_khz:.0f} kHz"
        _pfd_color = "#1b5e20" if _pfd_margin_val >= 0.0 else "#7f0000"
        _set("PFD", _format_value(_pfd_val), _pfd_unit, is_calc=True)
        _set("PFD Limit", _format_value(_pfd_limit_val), _pfd_unit, is_calc=True)
        _set("PFD Margin", _format_value(_pfd_margin_val), "dB", color=_pfd_color, is_calc=True)

    desired_order = [
        "Earth Radius",
        "Spacecraft Altitude",
        "Elevation Angle",
        "Elevation Lower Bound",
        "Elevation Upper Bound",
        "TX Frequency",
        "TX Power",
        "TX Losses",
        "TX Antenna Boresight Gain",
        "EIRP",
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
        "Sky Background Temperature",
        "Receiver Noise Figure",
        "Rx Antenna Gain",
        "System Noise Temperature",
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
        "Surplus Link Margin",
        "PFD",
        "PFD Limit",
        "PFD Margin",
    ]

    rows: List[ParameterRow] = []
    for label in desired_order:
        if label in row_values:
            rows.append(row_values[label])
        else:
            rows.append(ParameterRow(label, "—", ""))

    return rows
