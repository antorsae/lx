#!/usr/bin/env python3
"""
Generate a synthetic CDSL design from Juan's baffleless driver measurements.

The generator intentionally works from the already processed HDF5 files so the
synthetic system uses the same gating, smoothing, phase, and angle convention as
the measured LX521 system pages.
"""

from __future__ import annotations

import csv
import html
import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import javaobj
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import least_squares
from plotly.subplots import make_subplots

from directivity_calculations import DirectivityCalculator


FS_HZ = 48_000.0
FREQ_MIN = 70.0
FREQ_MAX = 20_000.0
COMMON_FREQ = np.geomspace(FREQ_MIN, FREQ_MAX, 1200)
ANGLES = [0, 15, 30, 45, 60, 75, 90]
PLOT_FREQS = [125, 250, 500, 1000, 2000, 4000, 8000, 12000]
TARGET_SPL_DB = 76.0
FLATNESS_FIT_BAND_HZ = (160.0, 18_000.0)
FLATNESS_PRIMARY_BAND_HZ = "200-10k"
FLATNESS_PRIMARY_SMOOTHING = "one_third_octave"
FLATNESS_TARGET_PEAK_TO_PEAK_DB = 0.70
FLATNESS_TARGET_RMS_DB = 0.25
FLAT_EQ_PRUNE_DB = 0.25

JUAN_HDF5 = Path("output/data/polar_data_juan_baffleless.h5")
LX521_HDF5 = Path("output/data/polar_data_lx521_system.h5")
SYNTHETIC_HDF5 = Path("output/data/polar_data_juan_cdsl_synthetic.h5")
OUTPUT_ROOT = Path("output/juan-baffleless-cdsl")
DOCS_ROOT = Path("docs/juan-baffleless-cdsl")
DOCS_PAGE = Path("docs/juan-baffleless-cdsl.html")


@dataclass
class Biquad:
    type: str
    fc: float
    q: float = 0.707
    gain_db: float = 0.0
    source: str = "eq"

    def manifest(self) -> Dict:
        return {
            "type": self.type,
            "fc_hz": round(self.fc, 3),
            "q": round(self.q, 4),
            "gain_db": round(self.gain_db, 3),
            "source": self.source,
        }


@dataclass
class DriverBand:
    role: str
    driver: str
    passband: Tuple[float, float]
    source: str
    rationale: str
    filters: List[Biquad] = field(default_factory=list)
    gain_db: float = 0.0
    delay_ms: float = 0.0
    polarity: int = 1

    @property
    def lo(self) -> float:
        return self.passband[0]

    @property
    def hi(self) -> float:
        return self.passband[1]


DESIGN: List[DriverBand] = [
    DriverBand(
        role="Low dipole / boundary high-pass",
        driver="L26RO4Y",
        passband=(70.0, 160.0),
        source="measurements/juan/L26RO4Y POLARES EN BAFFLE CILINDRICO VER NOTA",
        rationale=(
            "Highest measured low-band SPL with controlled 60 deg attenuation; "
            "note this is the cylindrical-baffle capture, not fully nude."
        ),
    ),
    DriverBand(
        role="Lower mid dipole",
        driver="L22MG (nude)",
        passband=(160.0, 800.0),
        source="measurements/juan/SEAS L22MG NUDE MIC ON AXIS",
        rationale="Nude front/rear data covers the low-mid bridge with usable output below the GRS start frequency.",
    ),
    DriverBand(
        role="Planar dipole mid",
        driver="GRS PT6816",
        passband=(800.0, 2400.0),
        source="measurements/juan/GRS PT6816 A MIC ON AXIS",
        rationale="Juan's notes identify it as the best constant-directivity and front/rear symmetry reference.",
    ),
    DriverBand(
        role="Narrowing upper-mid CDSL beam",
        driver="SS10F8414G10",
        passband=(2400.0, 10000.0),
        source="measurements/juan/ScanSpeak 10F8414G10",
        rationale=(
            "Best balanced-search compromise above 2 kHz: much stronger 30-to-60 deg separation than ND25/GRS, "
            "with Juan's notes rating rear directivity control better than 8424. "
            "L10NEO remains a serious alternate because raw REW THD is not worse than the ScanSpeak pair."
        ),
    ),
    DriverBand(
        role="Top octave dipole fill",
        driver="ND25FW4 (nude 18mm)",
        passband=(10000.0, 20000.0),
        source="measurements/juan/DAYTON ND25FW4 ANIDADOS 18 MM NUDE",
        rationale=(
            "Used only above the 2-10 kHz crosstalk-critical band to extend the top octave "
            "without relaxing the 2-10 kHz 30-to-60 deg target."
        ),
    ),
]

CROSSOVERS = [
    {"frequency_hz": 160.0, "type": "LR4", "low_driver": "L26RO4Y", "high_driver": "L22MG (nude)"},
    {"frequency_hz": 800.0, "type": "LR4", "low_driver": "L22MG (nude)", "high_driver": "GRS PT6816"},
    {"frequency_hz": 2400.0, "type": "LR4", "low_driver": "GRS PT6816", "high_driver": "SS10F8414G10"},
    {"frequency_hz": 10000.0, "type": "LR4", "low_driver": "SS10F8414G10", "high_driver": "ND25FW4 (nude 18mm)"},
]

DRIVER_META = {
    "L26RO4Y": {
        "source": "measurements/juan/L26RO4Y POLARES EN BAFFLE CILINDRICO VER NOTA",
        "rationale": "Best measured low-frequency candidate, but captured in a cylindrical baffle.",
        "risk_penalty": 0.5,
    },
    "L22MG (nude)": {
        "source": "measurements/juan/SEAS L22MG NUDE MIC ON AXIS",
        "rationale": "Strong low-mid bridge with nude front/rear data and useful coverage below 500 Hz.",
        "risk_penalty": 0.1,
    },
    "GRS PT6816": {
        "source": "measurements/juan/GRS PT6816 A MIC ON AXIS",
        "rationale": "Best local constant-directivity and front/rear symmetry reference.",
        "risk_penalty": 0.45,
    },
    "L10NEO": {
        "source": "measurements/juan/POLARES L10NEO",
        "rationale": "Strongest front-side 30-to-60 degree separation from about 2.5-8 kHz.",
        "risk_penalty": 0.35,
    },
    "SS10F8414G10": {
        "source": "measurements/juan/ScanSpeak 10F8414G10",
        "rationale": "Good directivity order with local notes rating distortion close to the 8424.",
        "risk_penalty": 0.2,
    },
    "SS10F8424G00": {
        "source": "measurements/juan/ScanSpeak 10F8424G00",
        "rationale": "Best local distortion/SPL notes, traded against weaker rear/directivity control.",
        "risk_penalty": 0.3,
    },
    "MU10RB-SL": {
        "source": "measurements/juan/SEAS MU10RB SL POLARES",
        "rationale": "Included as a measured alternative, but Juan's notes rank its distortion worst in the upper-mid comparison.",
        "risk_penalty": 1.2,
    },
    "ND25FW4 (nude 18mm)": {
        "source": "measurements/juan/DAYTON ND25FW4 ANIDADOS 18 MM NUDE",
        "rationale": "Treble extension candidate; measured 2-10 kHz pattern is wider than the 10 cm candidates.",
        "risk_penalty": 0.55,
    },
}

DRIVER_AUDIT_DRIVERS = [
    "L22MG (nude)",
    "GRS PT6816",
    "L10NEO",
    "SS10F8414G10",
    "SS10F8424G00",
    "MU10RB-SL",
    "ND25FW4 (nude 18mm)",
]

DRIVER_AUDIT_BANDS = [
    ("650-2000 Hz", 650.0, 2000.0),
    ("2-7 kHz", 2000.0, 7000.0),
    ("2-10 kHz", 2000.0, 10000.0),
]

DISTORTION_AUDIT_FILES = [
    {
        "driver": "L10NEO",
        "sample": "A",
        "path": Path("measurements/juan/POLARES L10NEO/FRONTALES/SEAS L10NEO A 0 F.mdat"),
    },
    {
        "driver": "SS10F8414G10",
        "sample": "single",
        "path": Path("measurements/juan/ScanSpeak 10F8414G10/SS10F8414G10 0 F.mdat"),
    },
    {
        "driver": "SS10F8424G00",
        "sample": "sn074",
        "path": Path("measurements/juan/ScanSpeak 10F8424G00/FRONT/SS10F8424G00 0 F sn 074.mdat"),
    },
    {
        "driver": "SS10F8424G00",
        "sample": "SN086",
        "path": Path("measurements/juan/ScanSpeak 10F8424G00/PAIR MATCHING SPL/SS10F8424G00 0 F SN 086.mdat"),
    },
    {
        "driver": "MU10RB-SL",
        "sample": "A",
        "path": Path("measurements/juan/SEAS MU10RB SL POLARES/FRONT/SEAS MU10RBSL A 0 F.mdat"),
    },
]

DISTORTION_AUDIT_BANDS = [
    ("1-7 kHz", 1000.0, 7000.0),
    ("2-7 kHz", 2000.0, 7000.0),
    ("2-10 kHz", 2000.0, 10000.0),
    ("7-10 kHz", 7000.0, 10000.0),
]

ROLE_NAMES = {
    4: ["Low", "Lower mid", "Upper/directivity band", "Top"],
    5: ["Low", "Lower mid", "Mid", "High/directivity band", "Top"],
}


def ensure_dirs(root: Path) -> None:
    for sub in [
        root,
        root / "static_plots/core",
        root / "static_plots/polar",
        root / "interactive/polar",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def load_hdf5(path: Path) -> Tuple[Dict, Dict]:
    data: Dict[str, Dict] = {}
    with h5py.File(path, "r") as h5:
        cfg = {
            "gate_left_ms": float(h5.attrs.get("gate_left_ms", 0.0)),
            "gate_right_ms": float(h5.attrs.get("gate_right_ms", 0.0)),
            "smoothing": int(h5.attrs.get("smoothing", 0)),
            "smoothing_str": str(h5.attrs.get("smoothing_str", "None")),
        }
        for driver in h5.keys():
            group = h5[driver]
            freqs = np.array(group["frequencies"])
            entry = {
                "frequencies": freqs,
                "has_rear": bool(group.attrs.get("has_rear", False)),
                "angles": {},
                "rear_angles": {},
            }
            for group_name, target in [("angles", "angles"), ("rear_angles", "rear_angles")]:
                if group_name not in group:
                    continue
                for angle_str in group[group_name].keys():
                    angle_group = group[group_name][angle_str]
                    entry[target][int(angle_str)] = {
                        "magnitude": np.array(angle_group["magnitude"], dtype=float),
                        "phase": np.array(angle_group["phase"], dtype=float),
                    }
            data[driver] = entry
    return data, cfg


def db_to_pressure(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, db / 20.0)


def pressure_to_db(pressure: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(pressure), floor))


def interp_complex(driver_data: Dict, side: str, angle: int, freq: np.ndarray) -> np.ndarray:
    angle_key = "rear_angles" if side == "R" else "angles"
    if angle not in driver_data.get(angle_key, {}):
        return np.zeros_like(freq, dtype=complex)

    raw_freq = driver_data["frequencies"]
    raw = driver_data[angle_key][angle]
    x = np.log(raw_freq)
    xi = np.log(freq)

    mag = np.interp(xi, x, raw["magnitude"], left=np.nan, right=np.nan)
    phase_rad = np.unwrap(np.deg2rad(raw["phase"]))
    phase = np.interp(xi, x, phase_rad, left=np.nan, right=np.nan)
    valid = np.isfinite(mag) & np.isfinite(phase)
    out = np.zeros_like(freq, dtype=complex)
    out[valid] = db_to_pressure(mag[valid]) * np.exp(1j * phase[valid])
    return out


def _normalize_biquad(b: Iterable[float], a: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    b_arr = np.asarray(list(b), dtype=float)
    a_arr = np.asarray(list(a), dtype=float)
    return b_arr / a_arr[0], a_arr / a_arr[0]


def biquad_coefficients(flt: Biquad, fs: float = FS_HZ) -> Tuple[np.ndarray, np.ndarray]:
    fc = min(max(float(flt.fc), 5.0), fs * 0.49)
    q = max(float(flt.q), 0.05)
    w0 = 2.0 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    if flt.type == "lowpass":
        b = [(1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    elif flt.type == "highpass":
        b = [(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    elif flt.type == "peaking":
        amp = 10 ** (flt.gain_db / 40.0)
        b = [1 + alpha * amp, -2 * cos_w0, 1 - alpha * amp]
        a = [1 + alpha / amp, -2 * cos_w0, 1 - alpha / amp]
    elif flt.type in {"lowshelf", "highshelf"}:
        amp = 10 ** (flt.gain_db / 40.0)
        sqrt_amp = math.sqrt(amp)
        shelf_slope = max(float(flt.q), 0.05)
        shelf_alpha = sin_w0 / 2.0 * math.sqrt((amp + 1.0 / amp) * (1.0 / shelf_slope - 1.0) + 2.0)
        if flt.type == "lowshelf":
            b = [
                amp * ((amp + 1) - (amp - 1) * cos_w0 + 2 * sqrt_amp * shelf_alpha),
                2 * amp * ((amp - 1) - (amp + 1) * cos_w0),
                amp * ((amp + 1) - (amp - 1) * cos_w0 - 2 * sqrt_amp * shelf_alpha),
            ]
            a = [
                (amp + 1) + (amp - 1) * cos_w0 + 2 * sqrt_amp * shelf_alpha,
                -2 * ((amp - 1) + (amp + 1) * cos_w0),
                (amp + 1) + (amp - 1) * cos_w0 - 2 * sqrt_amp * shelf_alpha,
            ]
        else:
            b = [
                amp * ((amp + 1) + (amp - 1) * cos_w0 + 2 * sqrt_amp * shelf_alpha),
                -2 * amp * ((amp - 1) + (amp + 1) * cos_w0),
                amp * ((amp + 1) + (amp - 1) * cos_w0 - 2 * sqrt_amp * shelf_alpha),
            ]
            a = [
                (amp + 1) - (amp - 1) * cos_w0 + 2 * sqrt_amp * shelf_alpha,
                2 * ((amp - 1) - (amp + 1) * cos_w0),
                (amp + 1) - (amp - 1) * cos_w0 - 2 * sqrt_amp * shelf_alpha,
            ]
    else:
        raise ValueError(f"Unsupported biquad type: {flt.type}")

    return _normalize_biquad(b, a)


def filter_response(filters: Iterable[Biquad], freq: np.ndarray) -> np.ndarray:
    z1 = np.exp(-1j * 2.0 * np.pi * freq / FS_HZ)
    z2 = z1 * z1
    response = np.ones_like(freq, dtype=complex)
    for flt in filters:
        b, a = biquad_coefficients(flt)
        response *= (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)
    return response


def add_lr4(filters: List[Biquad], kind: str, fc: float, source: str) -> None:
    for _ in range(2):
        filters.append(Biquad(kind, fc, q=0.7071, source=source))


def cascaded_lr4_filters(
    stage_index: int,
    xovers: List[float],
    *,
    include_boundary_highpass: bool = True,
    source_prefix: str = "LR4",
) -> List[Biquad]:
    filters: List[Biquad] = []
    if include_boundary_highpass:
        add_lr4(filters, "highpass", FREQ_MIN, f"{source_prefix} global boundary high-pass")
    for upstream_idx, fc in enumerate(xovers[:stage_index]):
        label = f"{source_prefix} cascaded upstream high-pass"
        if upstream_idx == stage_index - 1:
            label = f"{source_prefix} branch high-pass"
        add_lr4(filters, "highpass", fc, label)
    if stage_index < len(xovers):
        add_lr4(filters, "lowpass", xovers[stage_index], f"{source_prefix} branch low-pass")
    return filters


def add_crossover_filters(design: List[DriverBand]) -> None:
    xovers = [band.hi for band in design[:-1]]
    for idx, band in enumerate(design):
        band.filters.extend(cascaded_lr4_filters(idx, xovers))


def crossover_manifest(drivers: List[str], xovers: List[float]) -> List[Dict]:
    return [
        {
            "frequency_hz": float(fc),
            "type": "LR4",
            "low_driver": drivers[idx],
            "high_driver": drivers[idx + 1],
        }
        for idx, fc in enumerate(xovers)
    ]


def make_design(drivers: List[str], xovers: List[float]) -> List[DriverBand]:
    edges = [FREQ_MIN, *xovers, FREQ_MAX]
    roles = ROLE_NAMES[len(drivers)]
    design = []
    for idx, driver in enumerate(drivers):
        meta = DRIVER_META[driver]
        design.append(
            DriverBand(
                role=roles[idx],
                driver=driver,
                passband=(float(edges[idx]), float(edges[idx + 1])),
                source=meta["source"],
                rationale=meta["rationale"],
            )
        )
    return design


def interp_magnitude(driver_data: Dict, side: str, angle: int, freq: np.ndarray) -> np.ndarray:
    angle_key = "rear_angles" if side == "R" else "angles"
    if angle not in driver_data.get(angle_key, {}):
        return np.full_like(freq, np.nan, dtype=float)
    raw_freq = driver_data["frequencies"]
    raw = driver_data[angle_key][angle]["magnitude"]
    return np.interp(np.log(freq), np.log(raw_freq), raw, left=np.nan, right=np.nan)


def precompute_search_data(data: Dict[str, Dict]) -> Dict[str, Dict]:
    pre = {}
    for driver, driver_data in data.items():
        front_abs = {angle: interp_magnitude(driver_data, "F", angle, COMMON_FREQ) for angle in ANGLES}
        rear_abs = {angle: interp_magnitude(driver_data, "R", angle, COMMON_FREQ) for angle in ANGLES}
        f0 = front_abs[0]
        pre[driver] = {
            "front_abs": front_abs,
            "rear_abs": rear_abs,
            "front_norm": {angle: front_abs[angle] - f0 for angle in ANGLES},
            "rear_norm": {angle: rear_abs[angle] - f0 for angle in ANGLES},
            "valid": np.isfinite(f0),
        }
    return pre


def passband_weight(stage_index: int, xovers: List[float], freq: np.ndarray = COMMON_FREQ) -> np.ndarray:
    filters = cascaded_lr4_filters(stage_index, xovers, source_prefix="search LR4")
    return np.abs(filter_response(filters, freq))


def synthesize_search_norm(pre: Dict[str, Dict], drivers: List[str], xovers: List[float]) -> Dict:
    edges = [FREQ_MIN, *xovers, FREQ_MAX]
    weights = []
    for idx, driver in enumerate(drivers):
        w = passband_weight(idx, xovers)
        w = np.where(pre[driver]["valid"], w, 0.0)
        weights.append(w)
    weights_arr = np.vstack(weights)
    denom = np.sum(weights_arr, axis=0)
    valid = denom > 1e-8

    out = {"front": {}, "rear": {}, "valid": valid, "weights": weights_arr}
    for side_name, norm_key in [("front", "front_norm"), ("rear", "rear_norm")]:
        for angle in ANGLES:
            linear = np.zeros_like(COMMON_FREQ, dtype=float)
            for idx, driver in enumerate(drivers):
                norm_db = pre[driver][norm_key][angle]
                ratio = np.nan_to_num(db_to_pressure(norm_db), nan=0.0, posinf=0.0, neginf=0.0)
                linear += weights_arr[idx] * ratio
            arr = np.full_like(COMMON_FREQ, np.nan, dtype=float)
            arr[valid] = pressure_to_db(linear[valid] / denom[valid])
            out[side_name][angle] = arr
    return out


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 100.0
    return float(np.sqrt(np.mean(values * values)))


def search_xover_mismatch(pre: Dict[str, Dict], drivers: List[str], xovers: List[float]) -> float:
    errors = []
    for idx, fc in enumerate(xovers):
        low = pre[drivers[idx]]
        high = pre[drivers[idx + 1]]
        mask = (COMMON_FREQ >= fc / 2 ** (1 / 6)) & (COMMON_FREQ <= fc * 2 ** (1 / 6))
        if not np.any(mask):
            continue
        for key in ["front_norm", "rear_norm"]:
            for angle in [30, 60, 75, 90]:
                diff = low[key][angle][mask] - high[key][angle][mask]
                errors.append(diff)
    if not errors:
        return 50.0
    return rms(np.concatenate(errors))


def frequency_validity_penalty(pre: Dict[str, Dict], drivers: List[str], xovers: List[float]) -> float:
    edges = [FREQ_MIN, *xovers, FREQ_MAX]
    penalty = 0.0
    for idx, driver in enumerate(drivers):
        lo = edges[idx] * 1.08
        hi = edges[idx + 1] / 1.08 if edges[idx + 1] < FREQ_MAX else FREQ_MAX / 1.05
        mask = (COMMON_FREQ >= lo) & (COMMON_FREQ <= hi)
        if not np.any(mask):
            penalty += 5.0
            continue
        valid_fraction = float(np.mean(pre[driver]["valid"][mask]))
        penalty += (1.0 - valid_fraction) * 8.0
    return penalty


def candidate_prior_penalty(drivers: List[str]) -> float:
    penalty = 0.0
    for idx, driver in enumerate(drivers):
        penalty += DRIVER_META[driver]["risk_penalty"] * (1.15 if idx >= 2 else 0.65)
    if "ND25FW4 (nude 18mm)" not in drivers:
        penalty += 0.5
    if "L26RO4Y" in drivers:
        penalty += 0.25
    return penalty


def score_candidate(pre: Dict[str, Dict], drivers: List[str], xovers: List[float]) -> Dict:
    synth = synthesize_search_norm(pre, drivers, xovers)
    valid = synth["valid"]
    all_mask = valid & (COMMON_FREQ >= 80) & (COMMON_FREQ <= 18000)
    high_mask = valid & (COMMON_FREQ >= 2000) & (COMMON_FREQ <= 10000)

    ideal = {
        15: 20 * np.log10(np.cos(np.deg2rad(15))),
        30: 20 * np.log10(np.cos(np.deg2rad(30))),
        45: 20 * np.log10(np.cos(np.deg2rad(45))),
        60: 20 * np.log10(np.cos(np.deg2rad(60))),
        75: 20 * np.log10(np.cos(np.deg2rad(75))),
    }

    front_errors = []
    rear_errors = []
    for angle, target in ideal.items():
        front_errors.append(synth["front"][angle][all_mask] - target)
        rear_errors.append(synth["rear"][angle][all_mask] - target)
    dipole_front = rms(np.concatenate(front_errors))
    dipole_rear = rms(np.concatenate(rear_errors))

    front90 = synth["front"][90][all_mask]
    rear90 = synth["rear"][90][all_mask]
    null_penalty = rms(np.maximum(front90 + 18.0, 0.0)) + 0.6 * rms(np.maximum(rear90 + 18.0, 0.0))
    rear0_penalty = rms(synth["rear"][0][all_mask] - synth["front"][0][all_mask])

    sep = synth["front"][30] - synth["front"][60]
    sep_med = float(np.nanmedian(sep[high_mask])) if np.any(high_mask) else 0.0
    sep_p10 = float(np.nanpercentile(sep[high_mask], 10)) if np.any(high_mask) else 0.0
    front30_med = float(np.nanmedian(synth["front"][30][high_mask])) if np.any(high_mask) else -99.0

    high_penalty = (
        1.1 * max(0.0, 8.0 - sep_med)
        + 0.8 * max(0.0, 5.5 - sep_p10)
        + 0.5 * max(0.0, -5.0 - front30_med)
        - 0.22 * min(sep_med, 12.0)
    )

    xover_penalty = search_xover_mismatch(pre, drivers, xovers)
    validity_penalty = frequency_validity_penalty(pre, drivers, xovers)
    prior_penalty = candidate_prior_penalty(drivers)

    score = (
        1.25 * dipole_front
        + 0.75 * dipole_rear
        + 0.65 * null_penalty
        + 0.45 * rear0_penalty
        + 0.45 * xover_penalty
        + high_penalty
        + validity_penalty
        + prior_penalty
    )

    return {
        "score": round(float(score), 4),
        "drivers": drivers,
        "xovers": [float(x) for x in xovers],
        "ways": len(drivers),
        "dipole_front_rms_db": round(dipole_front, 3),
        "dipole_rear_rms_db": round(dipole_rear, 3),
        "null_penalty_db": round(null_penalty, 3),
        "rear0_rms_db": round(rear0_penalty, 3),
        "xover_mismatch_rms_db": round(xover_penalty, 3),
        "validity_penalty": round(validity_penalty, 3),
        "prior_penalty": round(prior_penalty, 3),
        "sep_30_60_median_2_10k_db": round(sep_med, 3),
        "sep_30_60_p10_2_10k_db": round(sep_p10, 3),
        "front30_median_2_10k_db": round(front30_med, 3),
    }


def iter_candidate_specs() -> Iterable[Tuple[List[str], List[float]]]:
    low = "L26RO4Y"
    lower_mid = "L22MG (nude)"
    mid_candidates = ["GRS PT6816", "L10NEO", "SS10F8414G10", "SS10F8424G00", "MU10RB-SL"]
    high_candidates = ["L10NEO", "SS10F8414G10", "SS10F8424G00", "GRS PT6816", "MU10RB-SL"]
    top_candidates = ["ND25FW4 (nude 18mm)", "GRS PT6816", "SS10F8424G00", "SS10F8414G10"]

    for x1 in [120.0, 160.0, 200.0]:
        for x2 in [650.0, 800.0, 1000.0]:
            for x3 in [2000.0, 2400.0, 3000.0]:
                for x4 in [7000.0, 8000.0, 10000.0, 12000.0]:
                    for mid in mid_candidates:
                        for high in high_candidates:
                            for top in top_candidates:
                                drivers = [low, lower_mid, mid, high, top]
                                if len(set(drivers)) != len(drivers):
                                    continue
                                yield drivers, [x1, x2, x3, x4]

    upper_candidates = ["GRS PT6816", "L10NEO", "SS10F8414G10", "SS10F8424G00", "MU10RB-SL"]
    for x1 in [120.0, 160.0, 200.0]:
        for x2 in [650.0, 800.0, 1000.0, 1200.0]:
            for x3 in [2400.0, 3200.0, 4500.0, 7000.0, 10000.0]:
                for upper in upper_candidates:
                    for top in top_candidates:
                        drivers = [low, lower_mid, upper, top]
                        if len(set(drivers)) != len(drivers):
                            continue
                        yield drivers, [x1, x2, x3]


def optimize_design_search(data: Dict[str, Dict]) -> List[Dict]:
    pre = precompute_search_data(data)
    results = []
    for drivers, xovers in iter_candidate_specs():
        if any(driver not in pre for driver in drivers):
            continue
        results.append(score_candidate(pre, drivers, xovers))
    results.sort(key=lambda row: row["score"])
    return results


def choose_final_candidate(search_results: List[Dict]) -> Tuple[Dict, Dict]:
    for idx, row in enumerate(search_results, start=1):
        if (
            row["sep_30_60_median_2_10k_db"] >= 8.0
            and row["sep_30_60_p10_2_10k_db"] >= 5.8
            and row["dipole_front_rms_db"] <= 3.2
            and row["xover_mismatch_rms_db"] <= 9.5
            and row["validity_penalty"] <= 0.2
        ):
            selected = dict(row)
            selected["balanced_rank"] = idx
            return selected, {
                "method": "constrained CDSL selection",
                "reason": (
                    "Selected the lowest composite-score candidate that also clears "
                    "8 dB median and 5.8 dB 10th-percentile SPL30-SPL60 from 2-10 kHz, "
                    "while keeping front dipole RMS <=3.2 dB and crossover mismatch <=9.5 dB."
                ),
                "fallback_used": False,
            }

    selected = dict(search_results[0])
    selected["balanced_rank"] = 1
    return selected, {
        "method": "balanced fallback",
        "reason": "No candidate met the CDSL separation constraints, so the lowest composite score was selected.",
        "fallback_used": True,
    }


def with_rank(row: Dict, rank: int, role: str, note: str) -> Dict:
    result = dict(row)
    result["balanced_rank"] = rank
    result["finalist_role"] = role
    result["note"] = note
    return result


def has_both_scanspeaks(row: Dict) -> bool:
    drivers = set(row["drivers"])
    return "SS10F8424G00" in drivers and "SS10F8414G10" in drivers


def exactly_one_scanspeak(row: Dict) -> bool:
    drivers = set(row["drivers"])
    return ("SS10F8424G00" in drivers) ^ ("SS10F8414G10" in drivers)


def same_candidate(a: Dict, b: Dict) -> bool:
    return a["drivers"] == b["drivers"] and a["xovers"] == b["xovers"]


def choose_recommended_candidate(search_results: List[Dict]) -> Tuple[Dict, Dict]:
    for idx, row in enumerate(search_results, start=1):
        if (
            not has_both_scanspeaks(row)
            and row["validity_penalty"] <= 0.2
            and row["dipole_front_rms_db"] <= 2.5
            and row["xover_mismatch_rms_db"] <= 9.5
        ):
            return with_rank(
                row,
                idx,
                "Recommended balanced",
                (
                    "Primary recommendation: best balanced score that avoids splitting two near-identical "
                    "ScanSpeak 10 cm drivers into adjacent bands."
                ),
            ), {
                "method": "balanced recommendation with duplicate-driver guard",
                "reason": (
                    "Selected the best balanced candidate that avoids using both near-identical "
                    "SS10F8424G00 and SS10F8414G10 in adjacent passbands. This favors dipole consistency, "
                    "crossover continuity, and build simplicity over the most aggressive 30-to-60 dB target."
                ),
                "fallback_used": False,
            }

    selected = with_rank(search_results[0], 1, "Recommended fallback", "No non-duplicated candidate passed the balanced guard.")
    return selected, {
        "method": "balanced fallback",
        "reason": "No non-duplicated candidate passed the balanced guard, so the lowest composite score was selected.",
        "fallback_used": True,
    }


def first_candidate(search_results: List[Dict], predicate, role: str, note: str) -> Optional[Dict]:
    for idx, row in enumerate(search_results, start=1):
        if predicate(row):
            return with_rank(row, idx, role, note)
    return None


def choose_finalists(search_results: List[Dict]) -> Tuple[Dict, Dict, List[Dict]]:
    recommended, selection_info = choose_recommended_candidate(search_results)
    constrained, constrained_info = choose_final_candidate(search_results)
    constrained = with_rank(
        constrained,
        constrained.get("balanced_rank", search_results.index(next(r for r in search_results if same_candidate(r, constrained))) + 1),
        "CTC-constrained",
        "Experimental option: clears the 2-10 kHz 30-to-60 constraint, but uses both ScanSpeak 10 cm variants.",
    )

    l10 = first_candidate(
        search_results,
        lambda row: (
            "L10NEO" in row["drivers"]
            and row["sep_30_60_median_2_10k_db"] >= 7.5
            and row["xover_mismatch_rms_db"] <= 9.5
            and row["validity_penalty"] <= 0.2
        ),
        "L10NEO alternate",
        "Best L10NEO-flavored candidate with strong 30-to-60 separation and acceptable crossover mismatch.",
    )
    single_scan = first_candidate(
        search_results,
        lambda row: (
            row["ways"] == 4
            and exactly_one_scanspeak(row)
            and "GRS PT6816" in row["drivers"]
            and row["validity_penalty"] <= 0.2
        ),
        "Simple 4-way",
        "Simplest high-confidence 4-way variant using one ScanSpeak 10 cm driver plus GRS.",
    )

    finalists: List[Dict] = []
    for candidate in [recommended, constrained, l10, single_scan]:
        if candidate is None:
            continue
        if any(same_candidate(candidate, existing) for existing in finalists):
            continue
        finalists.append(candidate)

    selection_info["ctc_constrained_selection"] = constrained_info
    return recommended, selection_info, finalists


def plot_search_results(search_results: List[Dict], root: Path, selected_candidate: Optional[Dict] = None) -> None:
    top = []
    if selected_candidate is not None:
        top.append(selected_candidate)
    for row in search_results:
        if selected_candidate is not None and row["drivers"] == selected_candidate["drivers"] and row["xovers"] == selected_candidate["xovers"]:
            continue
        top.append(row)
        if len(top) >= 20:
            break
    labels = [
        f"{'recommended ' if idx == 0 and selected_candidate is not None else ''}{row['ways']}w: {' / '.join(row['drivers'][2:])}\n"
        f"{' / '.join(f'{x:g}' for x in row['xovers'])} Hz"
        for idx, row in enumerate(top)
    ]
    scores = [row["score"] for row in top]
    colors = ["#0f766e" if idx == 0 else "#94a3b8" for idx in range(len(top))]
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(np.arange(len(top))[::-1], scores[::-1], color=colors[::-1])
    ax.set_yticks(np.arange(len(top))[::-1], labels[::-1], fontsize=8)
    ax.set_xlabel("Composite score (lower is better)")
    ax.set_title("Recommended CDSL candidate plus best balanced candidates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "static_plots/core/cdsl_search_top_candidates.png", dpi=180)
    plt.close(fig)


def octave_smooth(freq: np.ndarray, values: np.ndarray, fraction: float = 3.0) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    logf = np.log2(freq)
    half_width = 1.0 / (2.0 * fraction)
    for idx, center in enumerate(logf):
        mask = np.abs(logf - center) <= half_width
        out[idx] = np.nanmedian(values[mask]) if np.any(mask) else values[idx]
    return out


def octave_mean_matrix(freq: np.ndarray, fraction: float) -> np.ndarray:
    logf = np.log2(freq)
    half_width = 1.0 / (2.0 * fraction)
    window = np.abs(logf[:, None] - logf[None, :]) <= half_width
    counts = np.maximum(window.sum(axis=1, keepdims=True), 1)
    return window.astype(float) / counts


def eq_centers_for_band(lo: float, hi: float) -> List[float]:
    if hi <= 220:
        return [85, 120, 165]
    if hi <= 800:
        return [220, 330, 500]
    if hi <= 3000:
        return [800, 1200, 1800]
    if hi <= 12000:
        return [2800, 4000, 6000, 8500]
    return [11000, 14000, 18000]


def add_auto_eq_and_gains(data: Dict[str, Dict], design: List[DriverBand]) -> None:
    for band in design:
        driver_data = data[band.driver]
        raw = interp_complex(driver_data, "F", 0, COMMON_FREQ)
        measured_db = pressure_to_db(raw)
        core_lo = max(band.lo * 1.12, FREQ_MIN)
        core_hi = min(band.hi / 1.12, FREQ_MAX)
        if band.hi >= FREQ_MAX:
            core_hi = FREQ_MAX / 1.05
        core = (COMMON_FREQ >= core_lo) & (COMMON_FREQ <= core_hi) & np.isfinite(measured_db)
        smooth_db = octave_smooth(COMMON_FREQ, measured_db, fraction=3.0)
        target = float(np.nanmedian(smooth_db[core])) if np.any(core) else TARGET_SPL_DB

        for center in eq_centers_for_band(band.lo, band.hi):
            if not (band.lo * 1.02 <= center <= band.hi / 1.02 or band.hi >= FREQ_MAX and center <= band.hi):
                continue
            desired = target - float(np.interp(np.log(center), np.log(COMMON_FREQ), smooth_db))
            desired = float(np.clip(desired, -6.0, 6.0))
            if abs(desired) >= 0.75:
                band.filters.append(Biquad("peaking", center, q=1.0, gain_db=desired, source="auto-EQ"))

        eq_only = [flt for flt in band.filters if flt.source == "auto-EQ"]
        xover_only = [flt for flt in band.filters if flt.source != "auto-EQ"]
        corrected = raw * filter_response(eq_only, COMMON_FREQ) * filter_response(xover_only, COMMON_FREQ)
        corrected_db = pressure_to_db(corrected)
        gain_region = (COMMON_FREQ >= core_lo) & (COMMON_FREQ <= core_hi)
        median_after = float(np.nanmedian(corrected_db[gain_region])) if np.any(gain_region) else TARGET_SPL_DB
        band.gain_db = TARGET_SPL_DB - median_after


def set_initial_gains(data: Dict[str, Dict], design: List[DriverBand]) -> None:
    for band in design:
        raw = interp_complex(data[band.driver], "F", 0, COMMON_FREQ) * filter_response(band.filters, COMMON_FREQ)
        measured_db = pressure_to_db(raw)
        core_lo = max(band.lo * 1.15, FREQ_MIN)
        core_hi = min(band.hi / 1.15, FREQ_MAX)
        if band.hi >= FREQ_MAX:
            core_hi = FREQ_MAX / 1.05
        core = (COMMON_FREQ >= core_lo) & (COMMON_FREQ <= core_hi) & np.isfinite(measured_db)
        median_after = float(np.nanmedian(measured_db[core])) if np.any(core) else TARGET_SPL_DB
        band.gain_db = TARGET_SPL_DB - median_after


def driver_transfer(band: DriverBand, freq: np.ndarray, *, include_delay: bool = True) -> np.ndarray:
    h = filter_response(band.filters, freq) * db_to_pressure(np.asarray(band.gain_db))
    if band.polarity < 0:
        h *= -1.0
    if include_delay and band.delay_ms:
        h *= np.exp(-1j * 2.0 * np.pi * freq * (band.delay_ms / 1000.0))
    return h


def contribution(
    data: Dict[str, Dict],
    band: DriverBand,
    side: str,
    angle: int,
    freq: np.ndarray,
    *,
    include_delay: bool = True,
) -> np.ndarray:
    raw = interp_complex(data[band.driver], side, angle, freq)
    return raw * driver_transfer(band, freq, include_delay=include_delay)


def optimize_delays(data: Dict[str, Dict], design: List[DriverBand]) -> None:
    design[0].delay_ms = 0.0
    design[0].polarity = 1
    delay_grid = np.linspace(-1.5, 1.5, 601)

    for idx in range(1, len(design)):
        low = design[idx - 1]
        high = design[idx]
        fc = high.lo
        band_mask = (COMMON_FREQ >= fc / math.sqrt(2.0)) & (COMMON_FREQ <= fc * math.sqrt(2.0))
        low_sig = contribution(data, low, "F", 0, COMMON_FREQ)[band_mask]
        high_base = contribution(data, high, "F", 0, COMMON_FREQ, include_delay=False)[band_mask]
        f = COMMON_FREQ[band_mask]

        best = (float("inf"), 0.0, 1)
        for polarity in (1, -1):
            signed = high_base * polarity
            for delay_ms in delay_grid:
                delayed = signed * np.exp(-1j * 2.0 * np.pi * f * (delay_ms / 1000.0))
                summed_db = pressure_to_db(low_sig + delayed)
                score = float(np.sqrt(np.mean((summed_db - TARGET_SPL_DB) ** 2)))
                if score < best[0]:
                    best = (score, float(delay_ms), polarity)

        high.delay_ms = best[1]
        high.polarity = best[2]


def synthesize_system(data: Dict[str, Dict], design: List[DriverBand]) -> Dict:
    system = {"freq": COMMON_FREQ, "front": {}, "rear": {}, "driver_contributions": {}}
    for side_name, side in [("front", "F"), ("rear", "R")]:
        for angle in ANGLES:
            total = np.zeros_like(COMMON_FREQ, dtype=complex)
            for band in design:
                total += contribution(data, band, side, angle, COMMON_FREQ)
            system[side_name][angle] = total

    for band in design:
        system["driver_contributions"][band.driver] = contribution(data, band, "F", 0, COMMON_FREQ)
    return system


def synthesize_front0(data: Dict[str, Dict], design: List[DriverBand]) -> np.ndarray:
    total = np.zeros_like(COMMON_FREQ, dtype=complex)
    for band in design:
        total += contribution(data, band, "F", 0, COMMON_FREQ)
    return total


def synthesize_front0_from_raw(raw_front0: Dict[str, np.ndarray], design: List[DriverBand]) -> np.ndarray:
    total = np.zeros_like(COMMON_FREQ, dtype=complex)
    for band in design:
        total += raw_front0[band.driver] * driver_transfer(band, COMMON_FREQ)
    return total


def flatness_summary(values_db: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label, lo, hi in [
        ("80-18k", 80.0, 18_000.0),
        ("200-10k", 200.0, 10_000.0),
        ("2-10k", 2_000.0, 10_000.0),
    ]:
        mask = (COMMON_FREQ >= lo) & (COMMON_FREQ <= hi) & np.isfinite(values_db)
        if not np.any(mask):
            continue
        band = values_db[mask]
        median = float(np.nanmedian(band))
        err = band - median
        out[label] = {
            "median_db": round(median, 3),
            "min_error_db": round(float(np.nanmin(err)), 3),
            "max_error_db": round(float(np.nanmax(err)), 3),
            "peak_to_peak_db": round(float(np.nanmax(err) - np.nanmin(err)), 3),
            "rms_error_db": round(float(np.sqrt(np.nanmean(err * err))), 3),
        }
    return out


def flatness_report(data: Dict[str, Dict], design: List[DriverBand]) -> Dict[str, Dict[str, Dict[str, float]]]:
    front0 = pressure_to_db(synthesize_front0(data, design))
    return {
        "raw": flatness_summary(front0),
        "one_sixth_octave": flatness_summary(octave_smooth(COMMON_FREQ, front0, 6.0)),
        "one_third_octave": flatness_summary(octave_smooth(COMMON_FREQ, front0, 3.0)),
    }


def flat_eq_candidate_specs(design: List[DriverBand]) -> List[Tuple[DriverBand, Biquad]]:
    centers = {
        "L26RO4Y": [("lowshelf", 95.0, 1.0), ("peaking", 110.0, 1.0)],
        "L22MG (nude)": [("peaking", 150.0, 1.0), ("peaking", 220.0, 1.0), ("peaking", 330.0, 1.0), ("peaking", 500.0, 1.0), ("peaking", 620.0, 1.0)],
        "SS10F8414G10": [("peaking", 700.0, 1.0), ("peaking", 900.0, 1.0), ("peaking", 1150.0, 1.0), ("peaking", 1450.0, 1.0), ("peaking", 1800.0, 1.0)],
        "SS10F8424G00": [("peaking", 700.0, 1.0), ("peaking", 950.0, 1.0), ("peaking", 1300.0, 1.0), ("peaking", 1800.0, 1.0), ("peaking", 2400.0, 1.0)],
        "L10NEO": [("peaking", 900.0, 1.0), ("peaking", 1300.0, 1.0), ("peaking", 1800.0, 1.0), ("peaking", 2600.0, 1.0), ("peaking", 4200.0, 1.0)],
        "GRS PT6816": [("peaking", 2200.0, 1.0), ("peaking", 2800.0, 1.0), ("peaking", 3500.0, 1.0), ("peaking", 4500.0, 1.0), ("peaking", 5700.0, 1.0), ("peaking", 7200.0, 1.0), ("peaking", 9000.0, 1.0), ("peaking", 11000.0, 1.0)],
        "ND25FW4 (nude 18mm)": [("peaking", 12000.0, 1.0), ("peaking", 14000.0, 1.0), ("peaking", 16500.0, 1.0), ("highshelf", 18000.0, 1.0)],
    }
    candidates: List[Tuple[DriverBand, Biquad]] = []
    for band in design:
        for filter_type, center, q in centers.get(band.driver, []):
            if band.lo * 0.75 <= center <= band.hi * 1.25:
                flt = Biquad(filter_type, center, q=q, gain_db=0.0, source="flat-EQ")
                candidates.append((band, flt))
    return candidates


def flat_eq_key(candidate: Tuple[DriverBand, Biquad]) -> str:
    band, flt = candidate
    return f"{band.driver}|{flt.type}|{flt.fc:.3f}|{flt.q:.4f}"


def restore_filter_selection(
    design: List[DriverBand],
    base_filters: Dict[str, List[Biquad]],
    selected: List[Tuple[DriverBand, Biquad]],
) -> None:
    for band in design:
        band.filters = list(base_filters[band.driver])
    for band, flt in selected:
        band.filters.append(flt)


def flatness_primary_stats(report: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    return report[FLATNESS_PRIMARY_SMOOTHING][FLATNESS_PRIMARY_BAND_HZ]


def flatness_constraints_met(report: Dict[str, Dict[str, Dict[str, float]]]) -> bool:
    stats = flatness_primary_stats(report)
    return (
        stats["peak_to_peak_db"] <= FLATNESS_TARGET_PEAK_TO_PEAK_DB
        and stats["rms_error_db"] <= FLATNESS_TARGET_RMS_DB
    )


def flatness_selection_score(report: Dict[str, Dict[str, Dict[str, float]]], filter_count: int) -> float:
    primary = flatness_primary_stats(report)
    sixth = report["one_sixth_octave"][FLATNESS_PRIMARY_BAND_HZ]
    raw = report["raw"][FLATNESS_PRIMARY_BAND_HZ]
    excess_p2p = max(0.0, primary["peak_to_peak_db"] - FLATNESS_TARGET_PEAK_TO_PEAK_DB)
    excess_rms = max(0.0, primary["rms_error_db"] - FLATNESS_TARGET_RMS_DB)
    return float(
        10.0 * excess_p2p
        + 12.0 * excess_rms
        + 0.65 * primary["rms_error_db"]
        + 0.20 * sixth["rms_error_db"]
        + 0.04 * raw["rms_error_db"]
        + 0.025 * filter_count
    )


def optimize_flatness_selection(
    data: Dict[str, Dict],
    design: List[DriverBand],
    selected: List[Tuple[DriverBand, Biquad]],
    base_filters: Dict[str, List[Biquad]],
    base_gains: np.ndarray,
    *,
    initial_filter_gains: Optional[Dict[str, float]] = None,
    max_nfev: int = 50,
) -> Dict:
    restore_filter_selection(design, base_filters, selected)
    fit_lo, fit_hi = FLATNESS_FIT_BAND_HZ
    fit_freq = COMMON_FREQ[(COMMON_FREQ >= fit_lo) & (COMMON_FREQ <= fit_hi)][::4]
    raw_front0 = {band.driver: interp_complex(data[band.driver], "F", 0, fit_freq) for band in design}
    smooth_third = octave_mean_matrix(fit_freq, 3.0)
    smooth_sixth = octave_mean_matrix(fit_freq, 6.0)
    weights = np.ones_like(fit_freq)
    weights[(fit_freq < 220.0) | (fit_freq > 12_000.0)] = 0.5

    def set_params(params: np.ndarray) -> None:
        for idx, band in enumerate(design):
            band.gain_db = float(base_gains[idx] + params[idx])
        offset = len(design)
        for idx, (_, flt) in enumerate(selected):
            flt.gain_db = float(params[offset + idx])

    def residual(params: np.ndarray) -> np.ndarray:
        set_params(params)
        total = np.zeros_like(fit_freq, dtype=complex)
        for band in design:
            total += raw_front0[band.driver] * driver_transfer(band, fit_freq)
        raw = pressure_to_db(total)
        one_third = smooth_third @ raw
        one_sixth = smooth_sixth @ raw
        return np.concatenate(
            [
                0.9 * (one_third - TARGET_SPL_DB) * weights,
                0.3 * (one_sixth - TARGET_SPL_DB) * weights,
                0.08 * (raw - TARGET_SPL_DB) * weights,
                0.03 * params[: len(design)],
                0.06 * params[len(design) :],
            ]
        )

    params0 = np.zeros(len(design) + len(selected), dtype=float)
    if initial_filter_gains:
        for idx, candidate in enumerate(selected):
            params0[len(design) + idx] = initial_filter_gains.get(flat_eq_key(candidate), 0.0)
    lower = np.concatenate([np.full(len(design), -8.0), np.full(len(selected), -8.0)])
    upper = np.concatenate([np.full(len(design), 8.0), np.full(len(selected), 8.0)])
    result = least_squares(residual, params0, bounds=(lower, upper), max_nfev=max_nfev)
    set_params(result.x)
    report = flatness_report(data, design)
    return {
        "report": report,
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "filter_gains": {flat_eq_key(candidate): round(candidate[1].gain_db, 6) for candidate in selected},
        "selected": selected,
        "score": flatness_selection_score(report, len(selected)),
        "constraints_met": flatness_constraints_met(report),
    }


def optimize_system_flatness(data: Dict[str, Dict], design: List[DriverBand]) -> Dict:
    before = flatness_report(data, design)
    candidates = flat_eq_candidate_specs(design)
    base_filters = {band.driver: list(band.filters) for band in design}
    base_gains = np.asarray([band.gain_db for band in design], dtype=float)

    full = optimize_flatness_selection(
        data,
        design,
        candidates,
        base_filters,
        base_gains,
        max_nfev=90,
    )
    full_gains = dict(full["filter_gains"])
    ranked = sorted(candidates, key=lambda candidate: abs(full_gains.get(flat_eq_key(candidate), 0.0)), reverse=True)

    best_record: Optional[Dict] = None
    best_unconstrained: Optional[Dict] = None
    for count in range(len(ranked) + 1):
        selected = ranked[:count]
        record = optimize_flatness_selection(
            data,
            design,
            selected,
            base_filters,
            base_gains,
            initial_filter_gains=full_gains,
            max_nfev=45,
        )
        if best_unconstrained is None or record["score"] < best_unconstrained["score"]:
            best_unconstrained = record
        if record["constraints_met"]:
            best_record = record
            break

    if best_record is None:
        best_record = best_unconstrained if best_unconstrained is not None else full

    selected = list(best_record["selected"])
    selected_gains = dict(best_record["filter_gains"])
    pruned = [candidate for candidate in selected if abs(selected_gains.get(flat_eq_key(candidate), 0.0)) >= FLAT_EQ_PRUNE_DB]
    if len(pruned) < len(selected):
        pruned_record = optimize_flatness_selection(
            data,
            design,
            pruned,
            base_filters,
            base_gains,
            initial_filter_gains=selected_gains,
            max_nfev=60,
        )
        if pruned_record["constraints_met"] or not best_record["constraints_met"]:
            best_record = pruned_record

    restore_filter_selection(design, base_filters, list(best_record["selected"]))
    for idx, band in enumerate(design):
        band.gain_db = float(base_gains[idx])
    final_record = optimize_flatness_selection(
        data,
        design,
        list(best_record["selected"]),
        base_filters,
        base_gains,
        initial_filter_gains=best_record["filter_gains"],
        max_nfev=80,
    )
    kept = list(final_record["selected"])
    after = final_record["report"]

    return {
        "target_db": TARGET_SPL_DB,
        "fit_domain_hz": list(FLATNESS_FIT_BAND_HZ),
        "primary_constraint": {
            "smoothing": FLATNESS_PRIMARY_SMOOTHING,
            "band": FLATNESS_PRIMARY_BAND_HZ,
            "max_peak_to_peak_db": FLATNESS_TARGET_PEAK_TO_PEAK_DB,
            "max_rms_error_db": FLATNESS_TARGET_RMS_DB,
        },
        "constraint_met": flatness_constraints_met(after),
        "optimizer_cost": round(float(final_record["optimizer_cost"]), 3),
        "optimizer_nfev": int(final_record["optimizer_nfev"]),
        "filters_tested": len(candidates),
        "filters_kept": len(kept),
        "full_fit_filters_needed": sum(1 for gain in full_gains.values() if abs(gain) >= FLAT_EQ_PRUNE_DB),
        "method": (
            "sparse least-squares: solve full candidate pool, rank by fitted correction magnitude, "
            "then keep the smallest ranked set that satisfies the smoothed summed-response constraint"
        ),
        "before": before,
        "after": after,
    }


def matrix_from_system(system: Dict, side_name: str = "front") -> np.ndarray:
    return np.column_stack([pressure_to_db(system[side_name][angle]) for angle in ANGLES])


def load_lx521(data: Dict[str, Dict]) -> Dict:
    driver = next(iter(data.keys()))
    entry = data[driver]
    out = {"freq": COMMON_FREQ, "front": {}, "rear": {}, "driver": driver}
    for side_name, side in [("front", "F"), ("rear", "R")]:
        for angle in ANGLES:
            out[side_name][angle] = interp_complex(entry, side, angle, COMMON_FREQ)
    return out


def metric_30_60(curves: Dict[int, np.ndarray]) -> np.ndarray:
    return pressure_to_db(curves[30]) - pressure_to_db(curves[60])


def rear_front_delta(system_like: Dict, angle: int = 0) -> np.ndarray:
    return pressure_to_db(system_like["rear"][angle]) - pressure_to_db(system_like["front"][angle])


def side_null_delta(system_like: Dict) -> np.ndarray:
    return pressure_to_db(system_like["front"][90]) - pressure_to_db(system_like["front"][0])


def upper_mid_side_feature(system_like: Dict) -> Dict[str, float]:
    rel90 = pressure_to_db(system_like["front"][90]) - pressure_to_db(system_like["front"][0])
    mask = (COMMON_FREQ >= 1600.0) & (COMMON_FREQ <= 3200.0) & np.isfinite(rel90)
    if not np.any(mask):
        return {}
    local_freq = COMMON_FREQ[mask]
    local_rel = rel90[mask]
    min_idx = int(np.nanargmin(local_rel))
    max_idx = int(np.nanargmax(local_rel))
    return {
        "search_band_hz": [1600.0, 3200.0],
        "min_f90_minus_f0_db": round(float(local_rel[min_idx]), 3),
        "min_frequency_hz": round(float(local_freq[min_idx]), 1),
        "max_f90_minus_f0_db": round(float(local_rel[max_idx]), 3),
        "max_frequency_hz": round(float(local_freq[max_idx]), 1),
        "interpretation": (
            "The dark feature at 90 degrees is a side null, not an off-axis SPL peak. "
            "It is dominated by the upper-mid crossover/driver transition and should not be filled with on-axis EQ unless the design goal changes."
        ),
    }


def band_stats(freq: np.ndarray, values: np.ndarray, lo: float, hi: float) -> Dict[str, float]:
    mask = (freq >= lo) & (freq <= hi) & np.isfinite(values)
    if not np.any(mask):
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    vals = values[mask]
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


def cosine_target_db(angle: int) -> float:
    return 20.0 * np.log10(max(np.cos(np.deg2rad(float(angle))), 1e-6))


def measured_driver_directivity_audit(data: Dict[str, Dict]) -> List[Dict]:
    rows: List[Dict] = []
    audit_angles = [0, 15, 30, 45, 60, 75]
    for driver in DRIVER_AUDIT_DRIVERS:
        if driver not in data:
            continue
        driver_data = data[driver]
        front = {angle: interp_magnitude(driver_data, "F", angle, COMMON_FREQ) for angle in ANGLES}
        rear = {angle: interp_magnitude(driver_data, "R", angle, COMMON_FREQ) for angle in ANGLES}
        for band_label, lo, hi in DRIVER_AUDIT_BANDS:
            mask = (COMMON_FREQ >= lo) & (COMMON_FREQ <= hi) & np.isfinite(front[0])
            if not np.any(mask):
                continue

            front_errors = []
            rear_errors = []
            for angle in audit_angles:
                target = cosine_target_db(angle)
                front_errors.append(front[angle][mask] - front[0][mask] - target)
                if np.any(np.isfinite(rear[0][mask])):
                    rear_errors.append(rear[angle][mask] - rear[0][mask] - target)

            sep = front[30][mask] - front[60][mask]
            rear0_delta = rear[0][mask] - front[0][mask]
            side_null = front[90][mask] - front[0][mask]
            rows.append(
                {
                    "driver": driver,
                    "band": band_label,
                    "front_dipole_rms_db": round(rms(np.concatenate(front_errors)), 3),
                    "rear_dipole_rms_db": round(rms(np.concatenate(rear_errors)), 3) if rear_errors else float("nan"),
                    "sep_30_60_median_db": round(float(np.nanmedian(sep)), 3),
                    "sep_30_60_p10_db": round(float(np.nanpercentile(sep, 10)), 3),
                    "rear0_minus_front0_median_db": round(float(np.nanmedian(rear0_delta)), 3),
                    "front90_minus_front0_median_db": round(float(np.nanmedian(side_null)), 3),
                }
            )
    return rows


def read_rew_mdat_measurement(path: Path):
    with path.open("rb") as f:
        unmarshaller = javaobj.JavaObjectUnmarshaller(f, use_numpy_arrays=True)
        measurement = None
        for _ in range(5):
            measurement = unmarshaller.readObject(ignore_remaining_data=True)
    return measurement


def distortion_thd_audit() -> List[Dict]:
    rows: List[Dict] = []
    for spec in DISTORTION_AUDIT_FILES:
        path = spec["path"]
        if not path.exists():
            rows.append(
                {
                    "driver": spec["driver"],
                    "sample": spec["sample"],
                    "band": "missing",
                    "source_file": str(path),
                    "has_distortion_data": False,
                }
            )
            continue

        measurement = read_rew_mdat_measurement(path)
        distortion = getattr(measurement, "distortionData", None)
        harm_data = getattr(distortion, "harmData", None) if distortion is not None else None
        if distortion is None or harm_data is None or len(harm_data) < 2:
            rows.append(
                {
                    "driver": spec["driver"],
                    "sample": spec["sample"],
                    "band": "missing",
                    "source_file": str(path),
                    "measurement": str(getattr(measurement, "shortDesc", "")),
                    "has_distortion_data": False,
                }
            )
            continue

        freq = np.asarray(distortion.freqs, dtype=float)
        thd_db = np.asarray(harm_data[0], dtype=float)
        fundamental_spl = np.asarray(harm_data[1], dtype=float)
        thd_percent = 100.0 * np.power(10.0, thd_db / 20.0)

        for band_label, lo, hi in DISTORTION_AUDIT_BANDS:
            mask = (freq >= lo) & (freq <= hi) & np.isfinite(thd_db) & np.isfinite(fundamental_spl)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "driver": spec["driver"],
                    "sample": spec["sample"],
                    "band": band_label,
                    "source_file": str(path),
                    "measurement": str(getattr(measurement, "shortDesc", "")),
                    "has_distortion_data": True,
                    "fundamental_spl_median_db": round(float(np.nanmedian(fundamental_spl[mask])), 3),
                    "fundamental_spl_p10_db": round(float(np.nanpercentile(fundamental_spl[mask], 10)), 3),
                    "fundamental_spl_p90_db": round(float(np.nanpercentile(fundamental_spl[mask], 90)), 3),
                    "thd_median_db": round(float(np.nanmedian(thd_db[mask])), 3),
                    "thd_p90_db": round(float(np.nanpercentile(thd_db[mask], 90)), 3),
                    "thd_median_percent": round(float(np.nanmedian(thd_percent[mask])), 4),
                    "thd_p90_percent": round(float(np.nanpercentile(thd_percent[mask], 90)), 4),
                    "frequency_points": int(np.sum(mask)),
                }
            )
    return rows


def distortion_level_notes(rows: List[Dict]) -> List[str]:
    by_driver = {
        row["driver"]: row
        for row in rows
        if row.get("band") == "2-7 kHz" and row.get("has_distortion_data")
    }
    notes: List[str] = [
        "Raw REW .mdat THD was extracted from each driver's front 0-degree measurement; harmData[0] is treated as REW's THD trace and harmData[1] as the fundamental SPL trace.",
    ]
    l10 = by_driver.get("L10NEO")
    if l10:
        comparisons = []
        for driver in ["SS10F8414G10", "SS10F8424G00", "MU10RB-SL"]:
            peers = [row for row in rows if row.get("band") == "2-7 kHz" and row.get("driver") == driver and row.get("has_distortion_data")]
            if not peers:
                continue
            peer = min(peers, key=lambda row: abs(row["fundamental_spl_median_db"] - l10["fundamental_spl_median_db"]))
            delta = l10["fundamental_spl_median_db"] - peer["fundamental_spl_median_db"]
            comparisons.append(f"{delta:+.1f} dB vs {driver} {peer['sample']}")
        if comparisons:
            notes.append(
                "In the 2-7 kHz THD band, L10NEO's median fundamental SPL is "
                f"{l10['fundamental_spl_median_db']:.1f} dB ({', '.join(comparisons)}), so it was not measured at a lower level than the ScanSpeak references."
            )
        notes.append(
            f"L10NEO 2-7 kHz median THD is {l10['thd_median_percent']:.3f}% ({l10['thd_median_db']:.1f} dB), which is not worse than the available 8414/8424 raw REW THD traces."
        )
    notes.append("These are not perfectly level-matched sweeps; distortion rankings should be treated as measured-at-level evidence, not normalized maximum-SPL evidence.")
    return notes


def combine_0_180(front_matrix: np.ndarray, rear_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    front_angles = np.asarray(ANGLES)
    rear_angles = 180 - np.asarray(ANGLES[::-1])
    combined_angles = np.concatenate([front_angles, rear_angles[1:]])
    combined = np.concatenate([front_matrix, rear_matrix[:, ::-1][:, 1:]], axis=1)
    return combined_angles, combined


def plot_filter_transfer(design: List[DriverBand], root: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for band in design:
        h = driver_transfer(band, COMMON_FREQ)
        ax.semilogx(COMMON_FREQ, pressure_to_db(h), label=band.driver)
    for xo in CROSSOVERS:
        ax.axvline(xo["frequency_hz"], color="#94a3b8", linestyle="--", linewidth=0.9)
    ax.set_title("Synthetic CDSL per-driver IIR transfer functions")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Transfer magnitude (dB)")
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_ylim(-60, 18)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(root / "static_plots/core/cdsl_filter_transfer.png", dpi=180)
    plt.close(fig)


def plot_crossover_regions(system: Dict, design: List[DriverBand], root: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for band in design:
        ax.semilogx(COMMON_FREQ, pressure_to_db(system["driver_contributions"][band.driver]), label=band.driver, alpha=0.82)
    ax.semilogx(COMMON_FREQ, pressure_to_db(system["front"][0]), color="black", linewidth=2.2, label="Synthetic sum 0 deg")
    for xo in CROSSOVERS:
        ax.axvline(xo["frequency_hz"], color="#64748b", linestyle="--", linewidth=0.9)
        ax.text(xo["frequency_hz"], TARGET_SPL_DB + 8, f'{xo["frequency_hz"]:.0f} Hz', rotation=90, va="top", ha="right", fontsize=8)
    ax.set_title("Driver acoustic contributions and LR4 crossover regions")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB)")
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_ylim(TARGET_SPL_DB - 34, TARGET_SPL_DB + 12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(root / "static_plots/core/cdsl_crossover_regions.png", dpi=180)
    plt.close(fig)


def plot_freq_response(system: Dict, root: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for angle in ANGLES:
        width = 2.4 if angle == 0 else 1.2
        ax.semilogx(COMMON_FREQ, pressure_to_db(system["front"][angle]), label=f"F{angle} deg", linewidth=width)
    ax.set_title("Synthetic CDSL front frequency response by angle")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB)")
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_ylim(TARGET_SPL_DB - 36, TARGET_SPL_DB + 12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(root / "static_plots/core/cdsl_freq_response_angles.png", dpi=180)
    plt.close(fig)


def plot_contours(system: Dict, root: Path) -> None:
    front = matrix_from_system(system, "front")
    rear = matrix_from_system(system, "rear")
    angles, absolute = combine_0_180(front, rear)
    normalized = absolute - absolute[:, [0]]

    for name, matrix, levels, cmap, label in [
        ("absolute", absolute, np.linspace(TARGET_SPL_DB - 34, TARGET_SPL_DB + 10, 23), "viridis", "SPL (dB)"),
        ("normalized", normalized, np.linspace(-30, 3, 23), "magma", "Relative SPL to front 0 deg (dB)"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        contour = ax.contourf(COMMON_FREQ, angles, matrix.T, levels=levels, cmap=cmap, extend="both")
        ax.axhline(90, color="white", linewidth=0.9, alpha=0.8)
        for xo in CROSSOVERS:
            ax.axvline(xo["frequency_hz"], color="white", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlim(FREQ_MIN, FREQ_MAX)
        ax.set_ylim(0, 180)
        ax.set_title(f"Synthetic CDSL 0-180 deg contour ({name})")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Angle: front 0-90, rear 90-180 (deg)")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(label)
        fig.tight_layout()
        fig.savefig(root / f"static_plots/core/cdsl_contour_{name}.png", dpi=180)
        plt.close(fig)


def directivity_metrics(system: Dict) -> Dict:
    front = matrix_from_system(system, "front")
    calc = DirectivityCalculator(COMMON_FREQ, np.asarray(ANGLES), front)
    return {
        "di": calc.calculate_directivity_index(),
        "beam_3": calc.calculate_beamwidth(-3.0),
        "beam_6": calc.calculate_beamwidth(-6.0),
        "beam_10": calc.calculate_beamwidth(-10.0),
    }


def plot_di_beam(system_metrics: Dict, lx_metrics: Dict, root: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].semilogx(COMMON_FREQ, system_metrics["di"], label="Synthetic CDSL", linewidth=2)
    axes[0].semilogx(COMMON_FREQ, lx_metrics["di"], label="LX521 measured", linewidth=1.6, alpha=0.8)
    axes[0].set_ylabel("DI (dB)")
    axes[0].set_title("Directivity Index")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].semilogx(COMMON_FREQ, system_metrics["beam_6"], label="CDSL -6 dB", linewidth=2)
    axes[1].semilogx(COMMON_FREQ, system_metrics["beam_10"], label="CDSL -10 dB", linewidth=1.5)
    axes[1].semilogx(COMMON_FREQ, lx_metrics["beam_6"], label="LX521 -6 dB", linewidth=1.6, alpha=0.8)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Beamwidth (deg)")
    axes[1].set_title("Horizontal beamwidth")
    axes[1].set_ylim(0, 190)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(root / "static_plots/core/cdsl_di_beamwidth.png", dpi=180)
    plt.close(fig)

    for key, ylabel, fname in [
        ("di", "DI (dB)", "cdsl_vs_lx521_di.png"),
        ("beam_6", "-6 dB beamwidth (deg)", "cdsl_vs_lx521_beamwidth.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.semilogx(COMMON_FREQ, system_metrics[key], label="Synthetic CDSL", linewidth=2)
        ax.semilogx(COMMON_FREQ, lx_metrics[key], label="LX521 measured", linewidth=1.7, alpha=0.85)
        ax.set_xlim(FREQ_MIN, FREQ_MAX)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / f"static_plots/core/{fname}", dpi=180)
        plt.close(fig)


def plot_30_60(system: Dict, lx: Dict, root: Path) -> Tuple[np.ndarray, np.ndarray]:
    cdsl_metric = metric_30_60(system["front"])
    lx_metric = metric_30_60(lx["front"])
    ideal = 20.0 * np.log10(np.cos(np.deg2rad(30.0)) / np.cos(np.deg2rad(60.0)))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.semilogx(COMMON_FREQ, cdsl_metric, label="Synthetic CDSL", linewidth=2)
    ax.semilogx(COMMON_FREQ, lx_metric, label="LX521 measured", linewidth=1.7, alpha=0.85)
    ax.axhline(ideal, color="#64748b", linestyle="--", linewidth=1.0, label=f"Cosine dipole {ideal:.2f} dB")
    ax.axvspan(2000, 10000, color="#ccfbf1", alpha=0.28, label="2-10 kHz target band")
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL30 - SPL60 (dB)")
    ax.set_title("30-to-60 degree separation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(root / "static_plots/core/cdsl_30_vs_60_metric.png", dpi=180)
    fig.savefig(root / "static_plots/core/cdsl_vs_lx521_30_vs_60.png", dpi=180)
    plt.close(fig)
    return cdsl_metric, lx_metric


def polar_curve(system_like: Dict, freq_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    idx = int(np.argmin(np.abs(COMMON_FREQ - freq_hz)))
    values_front = np.array([pressure_to_db(system_like["front"][a][idx]) for a in ANGLES])
    values_rear = np.array([pressure_to_db(system_like["rear"][a][idx]) for a in ANGLES])

    right_angles = np.asarray(ANGLES)
    rear_angles = 180 - np.asarray(ANGLES[::-1])
    half_angles = np.concatenate([right_angles, rear_angles[1:]])
    half_values = np.concatenate([values_front, values_rear[::-1][1:]])
    full_angles = np.concatenate([half_angles, 360 - half_angles[-2:0:-1]])
    full_values = np.concatenate([half_values, half_values[-2:0:-1]])
    full_values = full_values - np.max(full_values)
    return np.deg2rad(full_angles), full_values


def plot_polar(system: Dict, lx: Dict, root: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), subplot_kw={"projection": "polar"})
    for ax, freq_hz in zip(axes.ravel(), PLOT_FREQS):
        theta, db = polar_curve(system, freq_hz)
        ax.plot(theta, db, linewidth=2, color="#0f766e")
        ax.set_title(f"{freq_hz:g} Hz")
        ax.set_rlim(-30, 0)
        ax.set_rticks([-30, -20, -10, 0])
        ax.grid(True, alpha=0.35)
    fig.suptitle("Synthetic CDSL normalized circular polars", y=1.02)
    fig.tight_layout()
    fig.savefig(root / "static_plots/polar/cdsl_polar_circular.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    compare_freqs = [500, 1000, 2000, 4000, 8000, 12000]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={"projection": "polar"})
    for ax, freq_hz in zip(axes.ravel(), compare_freqs):
        theta, db = polar_curve(system, freq_hz)
        theta_lx, db_lx = polar_curve(lx, freq_hz)
        ax.plot(theta, db, linewidth=2, color="#0f766e", label="CDSL")
        ax.plot(theta_lx, db_lx, linewidth=1.6, color="#7c3aed", alpha=0.8, label="LX521")
        ax.set_title(f"{freq_hz:g} Hz")
        ax.set_rlim(-30, 0)
        ax.set_rticks([-30, -20, -10, 0])
        ax.grid(True, alpha=0.35)
    axes.ravel()[0].legend(loc="lower left", bbox_to_anchor=(-0.28, -0.2))
    fig.suptitle("Synthetic CDSL vs measured LX521 normalized polars", y=1.02)
    fig.tight_layout()
    fig.savefig(root / "static_plots/polar/cdsl_vs_lx521_polar_circular.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_plotly_pages(system: Dict, lx: Dict, metrics: Dict, lx_metrics: Dict, cdsl_30_60: np.ndarray, lx_30_60: np.ndarray, root: Path) -> None:
    freq = COMMON_FREQ

    fig = go.Figure()
    for band in DESIGN:
        fig.add_trace(go.Scatter(x=freq, y=pressure_to_db(driver_transfer(band, freq)), name=band.driver))
    fig.update_xaxes(type="log", title="Frequency (Hz)")
    fig.update_yaxes(title="Transfer magnitude (dB)", range=[-60, 18])
    fig.update_layout(title="Synthetic CDSL IIR filter transfer", template="plotly_white")
    fig.write_html(root / "interactive/cdsl_filter_transfer.html", include_plotlyjs="cdn")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Frequency response", "30-to-60 degree separation"))
    for angle in ANGLES:
        fig.add_trace(go.Scatter(x=freq, y=pressure_to_db(system["front"][angle]), name=f"F{angle} deg"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq, y=cdsl_30_60, name="CDSL 30-60", line=dict(width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=freq, y=lx_30_60, name="LX521 30-60", line=dict(width=1.5, dash="dot")), row=2, col=1)
    fig.update_xaxes(type="log", title="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title="SPL (dB)", row=1, col=1)
    fig.update_yaxes(title="SPL30 - SPL60 (dB)", row=2, col=1)
    fig.update_layout(title="Synthetic CDSL design explorer", template="plotly_white", height=820)
    fig.write_html(root / "interactive/cdsl_design_explorer.html", include_plotlyjs="cdn")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Directivity Index", "-6 dB beamwidth"))
    fig.add_trace(go.Scatter(x=freq, y=metrics["di"], name="CDSL DI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq, y=lx_metrics["di"], name="LX521 DI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq, y=metrics["beam_6"], name="CDSL -6 dB"), row=2, col=1)
    fig.add_trace(go.Scatter(x=freq, y=lx_metrics["beam_6"], name="LX521 -6 dB"), row=2, col=1)
    fig.update_xaxes(type="log", title="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title="DI (dB)", row=1, col=1)
    fig.update_yaxes(title="Beamwidth (deg)", row=2, col=1)
    fig.update_layout(title="Synthetic CDSL directivity dashboard", template="plotly_white", height=760)
    fig.write_html(root / "interactive/cdsl_directivity_dashboard.html", include_plotlyjs="cdn")
    fig.write_html(root / "interactive/cdsl_vs_lx521_dashboard.html", include_plotlyjs="cdn")

    polar_fig = go.Figure()
    first = True
    buttons = []
    trace_count_per_freq = 2
    for freq_hz in PLOT_FREQS:
        theta, db = polar_curve(system, freq_hz)
        theta_lx, db_lx = polar_curve(lx, freq_hz)
        polar_fig.add_trace(go.Scatterpolar(theta=np.rad2deg(theta), r=db, name=f"CDSL {freq_hz:g} Hz", visible=first))
        polar_fig.add_trace(go.Scatterpolar(theta=np.rad2deg(theta_lx), r=db_lx, name=f"LX521 {freq_hz:g} Hz", visible=first))
        visible = [False] * (len(PLOT_FREQS) * trace_count_per_freq)
        base = PLOT_FREQS.index(freq_hz) * trace_count_per_freq
        visible[base] = True
        visible[base + 1] = True
        buttons.append({"label": f"{freq_hz:g} Hz", "method": "update", "args": [{"visible": visible}, {"title": f"Polar response at {freq_hz:g} Hz"}]})
        first = False
    polar_fig.update_layout(
        title=f"Polar response at {PLOT_FREQS[0]:g} Hz",
        template="plotly_white",
        polar=dict(radialaxis=dict(range=[-30, 0])),
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.02, "y": 1.12}],
    )
    polar_fig.write_html(root / "interactive/polar/cdsl_polar_explorer.html", include_plotlyjs="cdn")


def write_synthetic_hdf5(system: Dict, config_info: Dict, output_path: Path = SYNTHETIC_HDF5) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["gate_left_ms"] = config_info.get("gate_left_ms", 0.0)
        h5.attrs["gate_right_ms"] = config_info.get("gate_right_ms", 0.0)
        h5.attrs["smoothing"] = config_info.get("smoothing", 0)
        h5.attrs["smoothing_str"] = config_info.get("smoothing_str", "None")
        h5.attrs["synthetic_source"] = "generate_cdsl_design.py"

        group = h5.create_group("CDSL Synthetic SUM")
        group.attrs["driver_name"] = "CDSL Synthetic SUM"
        group.attrs["has_rear"] = True
        group.create_dataset("frequencies", data=COMMON_FREQ)

        for group_name, side_name in [("angles", "front"), ("rear_angles", "rear")]:
            angle_group = group.create_group(group_name)
            for angle in ANGLES:
                ag = angle_group.create_group(str(angle))
                values = system[side_name][angle]
                ag.create_dataset("magnitude", data=pressure_to_db(values))
                ag.create_dataset("phase", data=np.rad2deg(np.angle(values)))
                ag.attrs["unit"] = "dB SPL"
                ag.attrs["smoothing"] = config_info.get("smoothing_str", "None")
                ag.attrs["timing_corrected"] = False
                ag.attrs["timing_offset_ms"] = 0.0


def format_hz(value: float) -> str:
    return f"{value:.0f}"


def filter_topology_summary(design: List[DriverBand]) -> Dict:
    xovers = [band.hi for band in design[:-1]]
    diagram_lines = [
        "Input",
        f"  +-- LR4 HP {format_hz(FREQ_MIN)} Hz (2 biquads, global boundary)",
    ]
    for idx, band in enumerate(design[:-1]):
        fc = xovers[idx]
        indent = "  " * (idx + 2)
        diagram_lines.append(f"{indent}+-- split @ {format_hz(fc)} Hz")
        diagram_lines.append(f"{indent}    +-- LR4 LP {format_hz(fc)} Hz (2 biquads) -> {band.driver}")
        if idx < len(design) - 2:
            diagram_lines.append(f"{indent}    +-- LR4 HP {format_hz(fc)} Hz (2 biquads) -> next split")
        else:
            diagram_lines.append(f"{indent}    +-- LR4 HP {format_hz(fc)} Hz (2 biquads) -> {design[-1].driver}")

    stages = []
    total_effective_lr4 = 0
    total_eq = 0
    for idx, band in enumerate(design):
        global_hp = sum(1 for flt in band.filters if "global boundary high-pass" in flt.source)
        upstream_hp = sum(1 for flt in band.filters if "cascaded upstream high-pass" in flt.source)
        branch_hp = sum(1 for flt in band.filters if "branch high-pass" in flt.source)
        branch_lp = sum(1 for flt in band.filters if "branch low-pass" in flt.source)
        lr4 = sum(1 for flt in band.filters if flt.source.startswith("LR4"))
        flat_eq = sum(1 for flt in band.filters if flt.source == "flat-EQ")
        total_effective_lr4 += lr4
        total_eq += flat_eq
        stages.append(
            {
                "stage": idx + 1,
                "driver": band.driver,
                "passband_hz": [band.lo, band.hi],
                "global_boundary_hp_biquads": global_hp,
                "inherited_upstream_hp_biquads": upstream_hp,
                "own_branch_hp_biquads": branch_hp,
                "own_branch_lp_biquads": branch_lp,
                "effective_lr4_biquads": lr4,
                "flat_eq_biquads": flat_eq,
                "effective_total_biquads": len(band.filters),
            }
        )

    shared_tree_lr4 = 2 + 4 * len(xovers)
    return {
        "architecture": "cascaded LR4 split tree with shared high-pass carryover",
        "diagram": "\n".join(diagram_lines),
        "stages": stages,
        "summary": {
            "shared_tree_lr4_biquads": shared_tree_lr4,
            "standalone_channel_lr4_biquads": total_effective_lr4,
            "flat_eq_biquads": total_eq,
            "shared_tree_total_biquads_with_eq": shared_tree_lr4 + total_eq,
            "standalone_channel_total_biquads": total_effective_lr4 + total_eq,
        },
        "notes": [
            "Shared-tree count assumes the DSP can route a high-pass bus into the next split stage.",
            "Standalone-channel count is what is exported per driver when each output channel must contain all inherited upstream filters.",
            "Gain, delay, and polarity are not counted as biquads.",
        ],
    }


def write_exports(
    root: Path,
    design: List[DriverBand],
    metrics_rows: List[Dict],
    driver_audit_rows: List[Dict],
    distortion_audit_rows: List[Dict],
    flatness_info: Dict,
    side_feature_info: Dict,
    config_info: Dict,
    search_results: List[Dict],
    selected_candidate: Dict,
    selection_info: Dict,
    finalists: List[Dict],
) -> Dict:
    topology = filter_topology_summary(design)
    manifest = {
        "title": "Juan Baffleless Synthetic CDSL Design",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_hdf5": {
            "juan_baffleless": str(JUAN_HDF5),
            "lx521_system": str(LX521_HDF5),
            "synthetic_output": str(SYNTHETIC_HDF5),
        },
        "measurement_conditions": config_info,
        "frequency_grid_hz": {"min": FREQ_MIN, "max": FREQ_MAX, "points": int(len(COMMON_FREQ))},
        "angles_degrees": ANGLES,
        "target_spl_db": TARGET_SPL_DB,
        "crossovers": CROSSOVERS,
        "search": {
            "candidate_count": len(search_results),
            "score_direction": "lower is better",
            "selection": selection_info,
            "selected_candidate": selected_candidate,
            "finalists": finalists,
            "top_candidates": search_results[:25],
            "score_terms": {
                "dipole_front_rms_db": "front normalized polar fit to cosine dipole for 15-75 degrees",
                "dipole_rear_rms_db": "rear normalized polar fit to cosine dipole for 15-75 degrees",
                "null_penalty_db": "penalty for weak 90 degree nulls",
                "rear0_rms_db": "rear 0 degree magnitude symmetry relative to front 0 degree",
                "xover_mismatch_rms_db": "adjacent-driver normalized polar mismatch around each LR4 crossover",
                "sep_30_60_median_2_10k_db": "median front SPL30-SPL60 from 2-10 kHz",
                "prior_penalty": "small local evidence penalty for distortion/SPL uncertainty or known caveats",
            },
        },
        "driver_directivity_audit": driver_audit_rows,
        "distortion_thd_audit": distortion_audit_rows,
        "distortion_level_notes": distortion_level_notes(distortion_audit_rows),
        "filter_topology": topology,
        "flatness_optimization": flatness_info,
        "upper_mid_side_feature": side_feature_info,
        "local_evidence_notes": [
            "Juan's screenshot notes rank 8424 distortion/SPL best, 8414 close behind, and MU10 worst among the 8424/8414/MU10 upper-mid comparison.",
            "Raw REW .mdat THD extraction adds L10NEO to the comparison and does not support describing L10NEO as worse-distortion than the ScanSpeak pair.",
            "The same notes flag 8424 rear-side directivity and high-angle order as weaker than 8414, especially above 2 kHz.",
            "GRS is treated as the measured dipole/directivity-order reference, but it does not provide the largest 30-to-60 separation above 2 kHz.",
            "L10NEO remains a high-separation alternate; it is not selected in the balanced primary due to the composite directivity/crossover score, not because of worse raw THD evidence.",
        ],
        "mounting_geometry_notes": [
            "The synthetic HDF5 is a horizontal 0-180 degree sum of separately suspended driver measurements; it does not model vertical-plane lobing from center-to-center spacing.",
            "Non-GRS cone/dome drivers are assumed approximately axisymmetric when reasoning about vertical behavior, but the GRS planar is not axisymmetric and needs measured or modeled vertical data.",
            "A first-order vertical lobing model is feasible if driver coordinates and acoustic-center offsets are supplied, but it would only cover source interference, not pseudo-baffle diffraction.",
            "The pseudo-baffle/diffraction from a vertical mounting spine, neighboring motor structures, wiring, and driver frames is not identifiable from the current individual-driver HDF5 data; it needs CAD/BEM or assembly measurements.",
            "Recommendation: use this synthetic model for driver/filter shortlisting, then measure the actual stacked fixture horizontally and vertically before treating the crossover as final.",
        ],
        "validation_warnings": [
            "L26RO4Y source is a 25 cm deep / 32 cm diameter cylindrical-baffle measurement, not a nude baffleless capture.",
            "Processed phase was reconstructed from separately loaded measurements after per-measurement impulse peak alignment; synthetic complex summation is therefore preliminary.",
            "Rear polarity convention is inherited from the measurement files and existing 0-180 mapping; it has not been independently validated with raw impulse polarity.",
            "Absolute SPL and maximum-SPL claims remain limited by the available local evidence; raw THD is extracted for available 0-degree REW files but not normalized to matched drive level.",
            "DI and beamwidth figures match the existing repo convention and use front-horizontal data only, not a full 3D power response.",
            "Vertical-plane beaming and pseudo-baffle diffraction from the final stacked mounting geometry are not modeled.",
        ],
        "generated_files": {
            "synthetic_hdf5": str(SYNTHETIC_HDF5),
            "html_report": str(DOCS_PAGE),
            "asset_root": str(DOCS_ROOT),
            "driver_directivity_audit_csv": str(root / "driver_directivity_audit.csv"),
            "distortion_thd_audit_csv": str(root / "distortion_thd_audit.csv"),
        },
        "drivers": [
            {
                "role": band.role,
                "driver": band.driver,
                "passband_hz": [band.lo, band.hi],
                "source": band.source,
                "gain_db": round(band.gain_db, 3),
                "delay_ms": round(band.delay_ms, 4),
                "polarity": band.polarity,
                "rationale": band.rationale,
                "filters": [flt.manifest() for flt in band.filters],
            }
            for band in design
        ],
    }
    (root / "design_manifest.json").write_text(json.dumps(manifest, indent=2))

    (root / "candidate_search_results.json").write_text(json.dumps(search_results, indent=2))

    filters = {
        "_topology": topology,
        "_drivers": [band.driver for band in design],
        "drivers": {
        band.driver: {
            "gain_db": round(band.gain_db, 3),
            "delay_ms": round(band.delay_ms, 4),
            "polarity": band.polarity,
            "filters": [flt.manifest() for flt in band.filters],
        }
        for band in design
        },
    }
    (root / "cdsl_filters.json").write_text(json.dumps(filters, indent=2))

    yaml_lines = [
        "# Preliminary CamillaDSP-style filter export",
        "# Topology: cascaded LR4 split tree; later drivers include upstream high-pass stages in this per-channel export.",
        "# Diagram:",
    ]
    yaml_lines.extend(f"# {line}" for line in topology["diagram"].splitlines())
    yaml_lines.append("filters:")
    for band in design:
        safe = band.driver.replace(" ", "_").replace("(", "").replace(")", "")
        for idx, flt in enumerate(band.filters, start=1):
            yaml_lines.extend(
                [
                    f"  {safe}_{idx}:",
                    "    type: Biquad",
                    f"    parameters: {{type: {flt.type}, freq: {flt.fc:.3f}, q: {flt.q:.4f}, gain: {flt.gain_db:.3f}}}",
                ]
            )
    (root / "cdsl_camilladsp.yaml").write_text("\n".join(yaml_lines) + "\n")

    dsp_lines = [
        "# Preliminary Hypex-style filter listing generated from the synthetic CDSL model",
        "# Topology: cascaded LR4 split tree; later drivers include upstream high-pass stages in this per-channel export.",
        "# Diagram:",
    ]
    dsp_lines.extend(f"# {line}" for line in topology["diagram"].splitlines())
    for band in design:
        dsp_lines.append(f"\n[{band.driver}]")
        dsp_lines.append(f"Gain={band.gain_db:.3f} dB")
        dsp_lines.append(f"Delay={band.delay_ms:.4f} ms")
        dsp_lines.append(f"Polarity={band.polarity}")
        for idx, flt in enumerate(band.filters, start=1):
            dsp_lines.append(
                f"Filter{idx}={flt.type}, Fc={flt.fc:.3f}, Q={flt.q:.4f}, Gain={flt.gain_db:.3f}, Source={flt.source}"
            )
    (root / "cdsl_hypex.dsp").write_text("\n".join(dsp_lines) + "\n")

    with (root / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics_rows)

    if driver_audit_rows:
        with (root / "driver_directivity_audit.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(driver_audit_rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(driver_audit_rows)

    if distortion_audit_rows:
        fieldnames = sorted({key for row in distortion_audit_rows for key in row.keys()})
        with (root / "distortion_thd_audit.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(distortion_audit_rows)

    return manifest


def link(path: str, text: str) -> str:
    return f'<a href="{html.escape(path)}">{html.escape(text)}</a>'


def clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def render_report(root: Path, manifest: Dict, metrics_rows: List[Dict], cdsl_stats: Dict, lx_stats: Dict) -> str:
    stack_text = ", ".join(
        f"{band.driver} {band.lo:.0f}-{band.hi:.0f} Hz" if band.hi < FREQ_MAX else f"{band.driver} above {band.lo:.0f} Hz"
        for band in DESIGN
    )
    xo_text = ", ".join(f"{xo['frequency_hz']:.0f}" for xo in CROSSOVERS)
    driver_rows = []
    for band in DESIGN:
        driver_href = f"juan-baffleless/interactive/{band.driver}_freq_response_angles.html"
        if band.driver == "ND25FW4 (nude 18mm)":
            driver_href = "juan-baffleless/interactive/ND25FW4 (nude 18mm)_freq_response_angles.html"
        driver_rows.append(
            f"""
            <tr>
                <td>{html.escape(band.role)}</td>
                <td>{link(driver_href, band.driver)}</td>
                <td>{band.lo:.0f}-{band.hi:.0f}</td>
                <td>{band.gain_db:+.2f}</td>
                <td>{band.delay_ms:+.3f}</td>
                <td>{'normal' if band.polarity > 0 else 'inverted'}</td>
                <td>{html.escape(band.rationale)}</td>
            </tr>
            """
        )

    filter_rows = []
    for band in DESIGN:
        for flt in band.filters:
            filter_rows.append(
                f"""
                <tr>
                    <td>{html.escape(band.driver)}</td>
                    <td>{html.escape(flt.source)}</td>
                    <td>{html.escape(flt.type)}</td>
                    <td>{flt.fc:.1f}</td>
                    <td>{flt.q:.3f}</td>
                    <td>{flt.gain_db:+.2f}</td>
                </tr>
                """
            )

    metric_summary = "".join(
        f"""
        <tr>
            <td>{row['band']}</td>
            <td>{float(row['cdsl_30_60_median_db']):.2f}</td>
            <td>{float(row['lx521_30_60_median_db']):.2f}</td>
            <td>{float(row['cdsl_di_mean_db']):.2f}</td>
            <td>{float(row['cdsl_beam6_median_deg']):.0f}</td>
        </tr>
        """
        for row in metrics_rows
    )
    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in manifest["validation_warnings"])
    finalist_rows = "".join(
        f"""
        <tr>
            <td>{html.escape(row['finalist_role'])}</td>
            <td>{row.get('balanced_rank', idx + 1)}</td>
            <td>{row['ways']}</td>
            <td>{html.escape(' / '.join(row['drivers']))}</td>
            <td>{html.escape(' / '.join(f'{x:g}' for x in row['xovers']))}</td>
            <td>{float(row['score']):.2f}</td>
            <td>{float(row['dipole_front_rms_db']):.2f}</td>
            <td>{float(row['xover_mismatch_rms_db']):.2f}</td>
            <td>{float(row['sep_30_60_median_2_10k_db']):.2f}</td>
            <td>{html.escape(row['note'])}</td>
        </tr>
        """
        for idx, row in enumerate(manifest["search"]["finalists"])
    )
    audit_rows = "".join(
        f"""
        <tr>
            <td>{html.escape(row['driver'])}</td>
            <td>{html.escape(row['band'])}</td>
            <td>{float(row['front_dipole_rms_db']):.2f}</td>
            <td>{float(row['rear_dipole_rms_db']):.2f}</td>
            <td>{float(row['sep_30_60_median_db']):.2f}</td>
            <td>{float(row['sep_30_60_p10_db']):.2f}</td>
            <td>{float(row['rear0_minus_front0_median_db']):.2f}</td>
            <td>{float(row['front90_minus_front0_median_db']):.2f}</td>
        </tr>
        """
        for row in manifest["driver_directivity_audit"]
    )
    evidence_notes = "".join(f"<li>{html.escape(item)}</li>" for item in manifest["local_evidence_notes"])
    distortion_notes = "".join(f"<li>{html.escape(item)}</li>" for item in manifest["distortion_level_notes"])
    distortion_rows = "".join(
        f"""
        <tr>
            <td>{html.escape(row['driver'])}</td>
            <td>{html.escape(row['sample'])}</td>
            <td>{html.escape(row['band'])}</td>
            <td>{float(row['fundamental_spl_median_db']):.2f}</td>
            <td>{float(row['fundamental_spl_p10_db']):.2f}</td>
            <td>{float(row['fundamental_spl_p90_db']):.2f}</td>
            <td>{float(row['thd_median_db']):.2f}</td>
            <td>{float(row['thd_median_percent']):.3f}%</td>
            <td>{float(row['thd_p90_percent']):.3f}%</td>
        </tr>
        """
        for row in manifest["distortion_thd_audit"]
        if row.get("band") in {"2-7 kHz", "2-10 kHz"} and row.get("has_distortion_data")
    )
    mounting_notes = "".join(f"<li>{html.escape(item)}</li>" for item in manifest["mounting_geometry_notes"])
    flatness_rows = "".join(
        f"""
        <tr>
            <td>{html.escape(stage)}</td>
            <td>{html.escape(smoothing)}</td>
            <td>{html.escape(band)}</td>
            <td>{stats['median_db']:.2f}</td>
            <td>{stats['min_error_db']:.2f}</td>
            <td>{stats['max_error_db']:.2f}</td>
            <td>{stats['peak_to_peak_db']:.2f}</td>
            <td>{stats['rms_error_db']:.2f}</td>
        </tr>
        """
        for stage, smoothing_data in [
            ("Before flat-EQ", manifest["flatness_optimization"]["before"]),
            ("After flat-EQ", manifest["flatness_optimization"]["after"]),
        ]
        for smoothing, band_data in smoothing_data.items()
        for band, stats in band_data.items()
    )
    side_feature = manifest.get("upper_mid_side_feature", {})
    topology = manifest["filter_topology"]
    topology_summary = topology["summary"]
    topology_rows = "".join(
        f"""
        <tr>
            <td>{row['stage']}</td>
            <td>{html.escape(row['driver'])}</td>
            <td>{row['passband_hz'][0]:.0f}-{row['passband_hz'][1]:.0f}</td>
            <td>{row['global_boundary_hp_biquads']}</td>
            <td>{row['inherited_upstream_hp_biquads']}</td>
            <td>{row['own_branch_hp_biquads']}</td>
            <td>{row['own_branch_lp_biquads']}</td>
            <td>{row['effective_lr4_biquads']}</td>
            <td>{row['flat_eq_biquads']}</td>
            <td>{row['effective_total_biquads']}</td>
        </tr>
        """
        for row in topology["stages"]
    )
    topology_notes = "".join(f"<li>{html.escape(note)}</li>" for note in topology["notes"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Juan Baffleless Synthetic CDSL Design</title>
    <link rel="stylesheet" href="assets/css/styles.css">
    <style>
        :root {{
            --primary: #0f766e;
            --primary-dark: #115e59;
            --highlight-bg: #ccfbf1;
            --highlight-border: #0f766e;
            --highlight-text: #115e59;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin: 1.25rem 0;
        }}
        .metric-card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}
        .metric-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary-dark);
            line-height: 1.2;
        }}
        .metric-card .label {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.92rem;
        }}
        th, td {{
            border: 1px solid var(--border);
            padding: 0.55rem 0.65rem;
            vertical-align: top;
        }}
        th {{
            background: #f1f5f9;
            text-align: left;
        }}
        .asset-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1rem;
        }}
        .asset-grid img {{
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: white;
        }}
        .note {{
            color: var(--text-muted);
            font-size: 0.95rem;
        }}
        pre.topology-diagram {{
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            overflow-x: auto;
            line-height: 1.45;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Juan Baffleless Synthetic CDSL Design</h1>
        <p>Filtered synthetic sum from Juan driver measurements, compared with LX521.4</p>
        <a href="index.html" class="back-link">Back to Main Page</a>
    </header>

    <main>
        <div class="card">
            <div class="card-header">
                <h2>Executive Summary</h2>
                <div class="subtitle">5-way LR4 synthetic CDSL, generated from complex HDF5 measurements</div>
            </div>
            <div class="card-body">
                <div class="highlight">
                    <strong>Recommended balanced design:</strong> {html.escape(stack_text)}.
                </div>
                <div class="summary-grid">
                    <div class="metric-card"><div class="value">{cdsl_stats['median']:.1f} dB</div><div class="label">CDSL median SPL30-SPL60, 2-10 kHz</div></div>
                    <div class="metric-card"><div class="value">{lx_stats['median']:.1f} dB</div><div class="label">LX521 measured median SPL30-SPL60, 2-10 kHz</div></div>
                    <div class="metric-card"><div class="value">LR4</div><div class="label">Topology at {html.escape(xo_text)} Hz</div></div>
                    <div class="metric-card"><div class="value">0.5 / 3.0 ms</div><div class="label">Same HDF5 gate condition as Juan LX521 data</div></div>
                </div>
                <p class="note">
                    The model sums measured complex pressure by angle and side after explicit digital biquad filters, gain, polarity, and delay.
                    SPL calibration, physical driver offsets, cabinet diffraction, and distortion under equalized drive remain assumptions.
                    The recommended stack is from {manifest['search']['candidate_count']} evaluated 4-way/5-way LR4 combinations.
                    Selection method: {html.escape(manifest['search']['selection']['method'])}.
                </p>
            </div>
        </div>

        <h2 class="section-title">Finalists & Candidate Search</h2>
        <div class="card"><div class="card-body">
            <p class="note">
                The search is not just the preliminary stack. It tries L10NEO, both ScanSpeak 10F variants, GRS, MU10, and ND25 across multiple LR4 crossover grids.
                Lower score is better; the score favors cosine/dipole-like front and rear polars, strong side nulls, adjacent-driver pattern match near crossovers,
                measured-frequency validity, and larger 2-10 kHz SPL30-SPL60 separation.
                {html.escape(manifest['search']['selection']['reason'])}
            </p>
            <table>
                <thead><tr><th>Finalist</th><th>Balanced Rank</th><th>Ways</th><th>Drivers</th><th>XOs Hz</th><th>Score</th><th>Front Dipole RMS</th><th>XO Mismatch</th><th>30-60 dB</th><th>Note</th></tr></thead>
                <tbody>{finalist_rows}</tbody>
            </table>
            <div class="asset-grid">
                <img src="juan-baffleless-cdsl/static_plots/core/cdsl_search_top_candidates.png" alt="Top CDSL candidate search results">
            </div>
            <p>{link('juan-baffleless-cdsl/candidate_search_results.json', 'Download full candidate search JSON')}</p>
        </div></div>

        <h2 class="section-title">Driver Tradeoff Audit</h2>
        <div class="card"><div class="card-body">
            <p class="note">
                These rows are measured-driver diagnostics from Juan's baffleless HDF5 data before synthetic crossover summation.
                They explain the upper-band tradeoff: GRS is closest to a clean dipole shape, L10NEO and the 10F ScanSpeaks provide more 30-to-60 separation,
                and the dual-ScanSpeak split is kept as an experimental finalist because it adds a crossover between two near-identical radiators.
            </p>
            <table>
                <thead><tr><th>Driver</th><th>Band</th><th>Front Dipole RMS</th><th>Rear Dipole RMS</th><th>30-60 Median</th><th>30-60 P10</th><th>Rear0-Front0</th><th>Front90-Front0</th></tr></thead>
                <tbody>{audit_rows}</tbody>
            </table>
            <p class="note">
                Distortion and SPL evidence comes from <code>measurements/juan/UMs desnudos/contexto.txt</code> and its associated screenshots:
            </p>
            <ul>{evidence_notes}</ul>
            <p>{link('juan-baffleless-cdsl/driver_directivity_audit.csv', 'Download measured-driver directivity audit CSV')}</p>
        </div></div>

        <h2 class="section-title">Raw REW THD / SPL Audit</h2>
        <div class="card"><div class="card-body">
            <p class="note">
                THD and measurement level are extracted directly from the front 0-degree REW <code>.mdat</code> files, not from screenshots.
                The table shows each driver's fundamental SPL during the distortion sweep plus REW's stored THD trace.
                This avoids comparing a low-level distortion sweep against a high-level one.
            </p>
            <table>
                <thead><tr><th>Driver</th><th>Sample</th><th>Band</th><th>Fund SPL Med</th><th>Fund SPL P10</th><th>Fund SPL P90</th><th>THD Med dB</th><th>THD Med</th><th>THD P90</th></tr></thead>
                <tbody>{distortion_rows}</tbody>
            </table>
            <ul>{distortion_notes}</ul>
            <p>{link('juan-baffleless-cdsl/distortion_thd_audit.csv', 'Download raw REW THD / SPL audit CSV')}</p>
        </div></div>

        <h2 class="section-title">Chosen Drivers</h2>
        <div class="card"><div class="card-body">
            <table>
                <thead><tr><th>Role</th><th>Driver</th><th>Passband Hz</th><th>Gain dB</th><th>Delay ms</th><th>Polarity</th><th>Rationale</th></tr></thead>
                <tbody>{''.join(driver_rows)}</tbody>
            </table>
        </div></div>

        <h2 class="section-title">DSP / IIR Filters</h2>
        <div class="card"><div class="card-body">
            <h3>Cascaded Filter Topology</h3>
            <p class="note">
                Architecture: {html.escape(topology['architecture'])}. The higher branch of each split carries the previous high-pass stages into the next split,
                so later drivers include the first-stage and intermediate high-pass filters before their own branch filters.
            </p>
            <pre class="topology-diagram">{html.escape(topology['diagram'])}</pre>
            <div class="summary-grid">
                <div class="metric-card"><div class="value">{topology_summary['shared_tree_lr4_biquads']}</div><div class="label">Shared-tree LR4 biquads</div></div>
                <div class="metric-card"><div class="value">{topology_summary['standalone_channel_lr4_biquads']}</div><div class="label">Per-channel exported LR4 biquads</div></div>
                <div class="metric-card"><div class="value">{topology_summary['flat_eq_biquads']}</div><div class="label">Flat-EQ biquads</div></div>
                <div class="metric-card"><div class="value">{topology_summary['shared_tree_total_biquads_with_eq']}</div><div class="label">Shared-tree total with EQ</div></div>
            </div>
            <table>
                <thead><tr><th>Stage</th><th>Driver</th><th>Passband Hz</th><th>Global HP</th><th>Inherited HP</th><th>Own HP</th><th>Own LP</th><th>LR4 Total</th><th>Flat-EQ</th><th>Effective Total</th></tr></thead>
                <tbody>{topology_rows}</tbody>
            </table>
            <ul>{topology_notes}</ul>
        </div></div>

        <div class="card"><div class="card-body">
            <p class="note">
                Crossovers are LR4 cascades using two Q=0.7071 biquads per LR4 edge.
                The <code>flat-EQ</code> filters are a sparse summed-response least-squares fit assigned per driver.
                They use {manifest['flatness_optimization']['filters_kept']} active correction filters out of {manifest['flatness_optimization']['filters_tested']} candidates;
                the unconstrained full fit would keep about {manifest['flatness_optimization']['full_fit_filters_needed']} filters above the pruning threshold.
                This is still a simulated equalization pass, not a final measured-room/prototype EQ.
            </p>
            <table>
                <thead><tr><th>Driver</th><th>Source</th><th>Type</th><th>Fc Hz</th><th>Q</th><th>Gain dB</th></tr></thead>
                <tbody>{''.join(filter_rows)}</tbody>
            </table>
            <ul class="link-list">
                <li>{link('juan-baffleless-cdsl/cdsl_filters.json', 'Filter JSON')}</li>
                <li>{link('juan-baffleless-cdsl/cdsl_camilladsp.yaml', 'CamillaDSP-style YAML')}</li>
                <li>{link('juan-baffleless-cdsl/cdsl_hypex.dsp', 'Hypex-style DSP listing')}</li>
                <li>{link('juan-baffleless-cdsl/design_manifest.json', 'Design manifest')}</li>
            </ul>
        </div></div>

        <h2 class="section-title">Flatness Check</h2>
        <div class="card"><div class="card-body">
            <p class="note">
                Target flatness is evaluated on the synthetic front 0-degree sum. The raw gated response still contains narrow features that should not be over-corrected from this preliminary phase model.
                The more meaningful target here is the smoothed trend before prototype measurement.
            </p>
            <table>
                <thead><tr><th>Stage</th><th>Smoothing</th><th>Band</th><th>Median dB</th><th>Min Err</th><th>Max Err</th><th>Peak-Peak</th><th>RMS Err</th></tr></thead>
                <tbody>{flatness_rows}</tbody>
            </table>
        </div></div>

        <h2 class="section-title">Model Risks</h2>
        <div class="card"><div class="card-body">
            <ul>{warning_items}</ul>
        </div></div>

        <h2 class="section-title">Mounting Geometry</h2>
        <div class="card"><div class="card-body">
            <p class="note">
                The suspended-driver data is most reliable for choosing drivers and horizontal crossover behavior.
                It is not a complete model of the final physical stack. Vertically aligned drivers can create vertical lobing,
                and a narrow support spine or neighboring driver motors can behave like a thin diffracting baffle.
            </p>
            <ul>{mounting_notes}</ul>
        </div></div>

        <h2 class="section-title">Synthetic Acoustic Sum</h2>
        <div class="asset-grid">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_crossover_regions.png" alt="CDSL crossover regions">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_freq_response_angles.png" alt="CDSL frequency response by angle">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_filter_transfer.png" alt="CDSL filter transfer">
            <img src="juan-baffleless-cdsl/static_plots/polar/cdsl_polar_circular.png" alt="CDSL polar circular plots">
        </div>

        <h2 class="section-title">Directivity Results</h2>
        <div class="asset-grid">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_contour_normalized.png" alt="CDSL normalized contour">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_contour_absolute.png" alt="CDSL absolute contour">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_di_beamwidth.png" alt="CDSL DI and beamwidth">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_30_vs_60_metric.png" alt="CDSL 30 vs 60 metric">
        </div>
        <div class="card"><div class="card-body">
            <p class="note">
                Upper-mid side-feature diagnostic: in the 1.6-3.2 kHz region, front 90-degree response reaches
                <strong>{side_feature.get('min_f90_minus_f0_db', float('nan')):.1f} dB</strong> relative to front 0 degrees at
                <strong>{side_feature.get('min_frequency_hz', float('nan')):.0f} Hz</strong>.
                This is a side null, not a 90-degree SPL peak. Filling it would reduce the dipole null and the intended CDSL separation.
            </p>
        </div></div>

        <h2 class="section-title">30-vs-60 Metric</h2>
        <div class="card"><div class="card-body">
            <p>
                Metric definition: <strong>Delta30-60 = SPL30 - SPL60</strong>. A cosine dipole gives 4.77 dB;
                higher values above 2 kHz indicate more separation between the near-ear and far-ear angles for CDSL/crosstalk-cancellation use.
            </p>
            <table>
                <thead><tr><th>Band</th><th>CDSL median dB</th><th>LX521 median dB</th><th>CDSL mean DI dB</th><th>CDSL median -6 dB beamwidth</th></tr></thead>
                <tbody>{metric_summary}</tbody>
            </table>
            <p>{link('juan-baffleless-cdsl/metrics.csv', 'Download metrics CSV')}</p>
        </div></div>

        <h2 class="section-title">Comparison To LX521</h2>
        <div class="asset-grid">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_vs_lx521_di.png" alt="CDSL vs LX521 DI">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_vs_lx521_beamwidth.png" alt="CDSL vs LX521 beamwidth">
            <img src="juan-baffleless-cdsl/static_plots/core/cdsl_vs_lx521_30_vs_60.png" alt="CDSL vs LX521 30 vs 60">
            <img src="juan-baffleless-cdsl/static_plots/polar/cdsl_vs_lx521_polar_circular.png" alt="CDSL vs LX521 polar">
        </div>
        <p style="margin-top: 1rem;">{link('lx521-system.html', 'Open Juan LX521.4 measured system page')}</p>

        <h2 class="section-title">Interactive Views</h2>
        <div class="card"><div class="card-body">
            <ul class="link-list">
                <li>{link('juan-baffleless-cdsl/interactive/cdsl_design_explorer.html', 'CDSL design explorer')}</li>
                <li>{link('juan-baffleless-cdsl/interactive/cdsl_filter_transfer.html', 'Filter transfer dashboard')}</li>
                <li>{link('juan-baffleless-cdsl/interactive/cdsl_directivity_dashboard.html', 'Directivity dashboard')}</li>
                <li>{link('juan-baffleless-cdsl/interactive/cdsl_vs_lx521_dashboard.html', 'CDSL vs LX521 dashboard')}</li>
                <li>{link('juan-baffleless-cdsl/interactive/polar/cdsl_polar_explorer.html', 'Polar explorer')}</li>
            </ul>
        </div></div>

        <h2 class="section-title">Reproducibility</h2>
        <div class="card"><div class="card-body">
            <p><strong>Generated:</strong> {html.escape(manifest['generated_at_utc'])}</p>
            <p><strong>Inputs:</strong> {html.escape(str(JUAN_HDF5))}, {html.escape(str(LX521_HDF5))}</p>
            <p><strong>Synthetic HDF5:</strong> {html.escape(str(SYNTHETIC_HDF5))}</p>
            <p><strong>Smoothing:</strong> {html.escape(manifest['measurement_conditions']['smoothing_str'])}; <strong>gate:</strong> {manifest['measurement_conditions']['gate_left_ms']} ms / {manifest['measurement_conditions']['gate_right_ms']} ms.</p>
            <p class="note">This is a synthetic, equalized pressure sum. It is not a replacement for a measured prototype with final driver spacing and baffle geometry.</p>
        </div></div>
    </main>

    <footer>
        <p><a href="https://github.com/antorsae/lx">Source Code</a> | <a href="index.html">Back to Main Page</a></p>
    </footer>
</body>
</html>
"""


def copy_to_docs(output_root: Path, docs_root: Path) -> None:
    if docs_root.exists():
        shutil.rmtree(docs_root)
    shutil.copytree(output_root, docs_root)


def main() -> None:
    global DESIGN, CROSSOVERS

    if not JUAN_HDF5.exists():
        raise FileNotFoundError(f"Missing {JUAN_HDF5}")
    if not LX521_HDF5.exists():
        raise FileNotFoundError(f"Missing {LX521_HDF5}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    ensure_dirs(OUTPUT_ROOT)
    juan_data, juan_cfg = load_hdf5(JUAN_HDF5)
    lx_data, lx_cfg = load_hdf5(LX521_HDF5)

    search_results = optimize_design_search(juan_data)
    if not search_results:
        raise RuntimeError("No valid candidate designs were scored")

    best, selection_info, finalists = choose_finalists(search_results)
    DESIGN = make_design(best["drivers"], best["xovers"])
    CROSSOVERS = crossover_manifest(best["drivers"], best["xovers"])

    missing = [band.driver for band in DESIGN if band.driver not in juan_data]
    if missing:
        raise RuntimeError(f"Missing required drivers in {JUAN_HDF5}: {missing}")

    plot_search_results(search_results, OUTPUT_ROOT, best)
    add_crossover_filters(DESIGN)
    set_initial_gains(juan_data, DESIGN)
    optimize_delays(juan_data, DESIGN)
    flatness_info = optimize_system_flatness(juan_data, DESIGN)

    system = synthesize_system(juan_data, DESIGN)
    lx = load_lx521(lx_data)
    side_feature_info = upper_mid_side_feature(system)

    system_metrics = directivity_metrics(system)
    lx_metrics = directivity_metrics(lx)

    plot_filter_transfer(DESIGN, OUTPUT_ROOT)
    plot_crossover_regions(system, DESIGN, OUTPUT_ROOT)
    plot_freq_response(system, OUTPUT_ROOT)
    plot_contours(system, OUTPUT_ROOT)
    plot_di_beam(system_metrics, lx_metrics, OUTPUT_ROOT)
    cdsl_30_60, lx_30_60 = plot_30_60(system, lx, OUTPUT_ROOT)
    plot_polar(system, lx, OUTPUT_ROOT)
    write_plotly_pages(system, lx, system_metrics, lx_metrics, cdsl_30_60, lx_30_60, OUTPUT_ROOT)
    write_synthetic_hdf5(system, juan_cfg)

    bands = [
        ("70-200 Hz", 70, 200),
        ("200-800 Hz", 200, 800),
        ("800-2500 Hz", 800, 2500),
        ("2-10 kHz", 2000, 10000),
        ("10-20 kHz", 10000, 20000),
    ]
    metrics_rows: List[Dict] = []
    for label, lo, hi in bands:
        cdsl_stats = band_stats(COMMON_FREQ, cdsl_30_60, lo, hi)
        lx_stats = band_stats(COMMON_FREQ, lx_30_60, lo, hi)
        di_stats = band_stats(COMMON_FREQ, system_metrics["di"], lo, hi)
        beam_stats = band_stats(COMMON_FREQ, system_metrics["beam_6"], lo, hi)
        rear0_stats = band_stats(COMMON_FREQ, rear_front_delta(system, 0), lo, hi)
        null_stats = band_stats(COMMON_FREQ, side_null_delta(system), lo, hi)
        metrics_rows.append(
            {
                "band": label,
                "cdsl_30_60_mean_db": round(cdsl_stats["mean"], 3),
                "cdsl_30_60_median_db": round(cdsl_stats["median"], 3),
                "lx521_30_60_mean_db": round(lx_stats["mean"], 3),
                "lx521_30_60_median_db": round(lx_stats["median"], 3),
                "cdsl_di_mean_db": round(di_stats["mean"], 3),
                "cdsl_beam6_median_deg": round(beam_stats["median"], 3),
                "cdsl_r0_minus_f0_median_db": round(rear0_stats["median"], 3),
                "cdsl_f90_minus_f0_median_db": round(null_stats["median"], 3),
            }
        )

    driver_audit_rows = measured_driver_directivity_audit(juan_data)
    distortion_audit_rows = distortion_thd_audit()
    manifest = write_exports(
        OUTPUT_ROOT,
        DESIGN,
        metrics_rows,
        driver_audit_rows,
        distortion_audit_rows,
        flatness_info,
        side_feature_info,
        juan_cfg,
        search_results,
        best,
        selection_info,
        finalists,
    )
    report = render_report(
        OUTPUT_ROOT,
        manifest,
        metrics_rows,
        band_stats(COMMON_FREQ, cdsl_30_60, 2000, 10000),
        band_stats(COMMON_FREQ, lx_30_60, 2000, 10000),
    )
    copy_to_docs(OUTPUT_ROOT, DOCS_ROOT)
    DOCS_PAGE.write_text(clean_text(report))

    print(f"Wrote {OUTPUT_ROOT}")
    print(f"Wrote {DOCS_ROOT}")
    print(f"Wrote {DOCS_PAGE}")


if __name__ == "__main__":
    main()
