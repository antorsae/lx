#!/usr/bin/env python3
"""Slice and audit every released pause-and-bury magnet STL.

This is deliberately an offline, non-printer-contact pipeline.  It invokes
only Bambu Studio's local slicer CLI, parses the resulting plain G-code, and
writes hash-backed JSON/CSV/Markdown evidence.  It cannot upload or start a
print.

The CAD producer owns ``captive_magnet_release_catalog.json``.  In particular,
the catalog records magnet stations *after* the exact front-face-down STL
export transform.  This consumer therefore never imports build123d/OCC and is
safe to run on the macOS slicing host.

Pause markers are measured from the actual G-code layer schedule.  For each
site, local cavity toolpaths identify the final fully open layer and the first
boundary-contraction/interior-deposition layer.  The CAD bury plane is only a
bounded consistency datum; CAD heights alone are never emitted as pause
instructions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import dataclasses
import datetime as dt
import fnmatch
import html
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET

from bambu_3mf_audit import (
    Bambu3MFAuditError,
    Matrix4 as BambuMatrix4,
    audit_bambu_3mf,
    transform_point as transform_bambu_point,
    transform_vector as transform_bambu_vector,
    validate_bed_fit as validate_bambu_bed_fit,
    validate_result_bbox as validate_bambu_result_bbox,
)
from lx521_baffle.magnet_contract import DEFAULT_SPEC as DEFAULT_MAGNET_SPEC

from lx521_baffle.print_contract import (
    FrontDownContractError,
    RELEASE_ACOUSTIC_PRINT_CONTRACT,
    validate_front_down_transform,
    validate_print_sidecar,
)
from json_schema_subset import (
    JsonSchemaSubsetError,
    validate_json_schema,
)
from lx521_baffle.io import pretty_json_bytes, sha256_bytes, sha256_file


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = PROJECT_ROOT / "review" / "captive_magnet_release_catalog.json"
CATALOG_SCHEMA = PROJECT_ROOT / "captive_magnet_release_catalog.schema.json"
DEFAULT_PROFILE = PROJECT_ROOT / "captive_magnet_slicing_profile.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "review" / "captive_magnet_slice_audit"
SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
OVERSIZE_COVERED_STATUS = (
    "not_p2s_printable__cavity_covered_by_exact_split")
EXPECTED_RELEASE_ARTIFACT_COUNT = 58
EXPECTED_RELEASE_MAGNET_COUNT = 94
CANONICAL_MANIFEST_FILENAMES = (
    "captive_magnet_pause_manifest.json",
    "captive_magnet_pause_manifest.csv",
    "CAPTIVE_MAGNET_PAUSE_MANIFEST.md",
)
PLACED_3MF_FILENAME = "audited_slice_project.3mf"
READY_3MF_FILENAME = "ready_to_print.gcode.3mf"
FACADE_SOURCE = (SCRIPT_DIR / "slice_captive_magnets.py").resolve()
AUDIT_SOURCE_FILES = (
    FACADE_SOURCE,
    (SCRIPT_DIR / "release_validation.py").resolve(),
    (SCRIPT_DIR / "gcode_analysis.py").resolve(),
    (SCRIPT_DIR / "artifact_emit.py").resolve(),
    (SCRIPT_DIR / "bambu_3mf_audit.py").resolve(),
    (PROJECT_ROOT / "src/lx521_baffle/magnet_contract.py").resolve(),
    (PROJECT_ROOT / "src/lx521_baffle/print_contract.py").resolve(),
    (SCRIPT_DIR / "json_schema_subset.py").resolve(),
    (PROJECT_ROOT / "src/lx521_baffle/io.py").resolve(),
)


def _audit_source_hashes() -> dict[str, str]:
    """Hash every executable owner bound into cache and release evidence."""
    return {str(path): sha256_file(path) for path in AUDIT_SOURCE_FILES}


PRINT_INSERTION_DIRECTION_XYZ = (0.0, 0.0, -1.0)
PRINT_INSERTION_INSTRUCTION = (
    "insert vertically downward from above the paused part (+Z side) "
    "through the open loading chimney along print -Z"
)

FLOAT_EPS = 1.0e-6
LAYER_EPS = 2.0e-4
SITE_SAMPLE_STEP_MM = 0.10
ARC_TESSELLATION_STEP_MM = 0.20
MAX_ARC_TESSELLATION_SEGMENTS = 8192
ARC_RADIUS_TOLERANCE_MM = 0.05
LAST_OPEN_INTERIOR_PATH_LIMIT_MM = 0.20
CLOSING_BOUNDARY_INSET_MM = 0.03
CLOSING_BOUNDARY_REOPEN_TOLERANCE_MM = 0.03
FALLBACK_LINE_WIDTH_MM = 0.45
# A physically continuous 0.42-mm variable-width bead can tolerate a small gap
# between sampled segment centrelines.  Anything larger than one bead plus
# the two 0.05-mm half-sample offsets is a real break, not G-code segmentation.
RETAINING_PATH_CONNECTIVITY_GAP_MM = 0.52
RETAINING_TRACK_DEDUP_TOLERANCE_MM = 0.012
# Classify a transverse retaining traversal by the bead edge that forms the
# cavity boundary, rather than by a broad centreline band.  Across all current
# release slices the real boundary edge is within 0.0271 mm of nominal; 0.06
# mm leaves measurement/slicer margin while excluding V1's surrounding-body
# hairpin, whose nearest edge is 0.254 mm behind the cavity boundary.
TRANSVERSE_CAVITY_EDGE_TOLERANCE_MM = 0.06
TRANSVERSE_SAME_PATH_EDGE_RETURN_BIN_LIMIT = 3
# Bambu's one-path adaptive bead reaches 0.661027 mm on the legacy V1 inner
# skin.  Width alone never grants release: the same stage must still prove one
# physical traversal and a path-width-aware D5 x 2 loading aperture.
# The 0.52-mm dual-nozzle skins raise the inset-under-curved-face wedge
# family (stock/slim curved receivers 0.57..0.70 physical, Obi-Wan
# shoulder/ring carriers 0.67..0.78) past Arachne's default two-bead split
# threshold; the pinned min_bead_width=0.40 override forces those wedges to
# one adaptive bead, whose boundary width may legitimately reach the wedge
# maximum.  0.78 is that geometric ceiling, not an allowance for arbitrary
# fat beads.
TRANSVERSE_RETAINING_BEAD_WIDTH_RANGE_MM = (0.42, 0.78)
AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM = (0.42, 0.65)
# Arachne reports the resolved variable-width bead rather than simply echoing
# the 0.42-mm nominal outer-wall setting.  Angled transverse/interface paths
# in the pinned release reach 0.415656 mm, whereas axial rings merely serialize
# as 0.419996 mm.  Keep the CAD/profile contract at 0.42 mm and give each
# topology only the lower-side margin its measured output requires.
TRANSVERSE_RETAINING_BEAD_LOWER_WIDTH_TOLERANCE_MM = 0.005
AXIAL_RETAINING_BEAD_LOWER_WIDTH_TOLERANCE_MM = 0.000005
# The four constants above are the pinned 0.4-mm-nozzle-lane qualification.
# Another nozzle lane pins its own acceptance through the profile's
# requirements.retaining_bead_acceptance object; prepare_profiles() installs
# the active lane into this shared mapping exactly once per CLI invocation
# (each run resolves exactly one profile) and the effective profile contract
# records the installed values so every audit output names the acceptance it
# was judged against.  gcode_analysis reads this mapping, never the raw
# constants, at its band-gated call sites.
RETAINING_BEAD_ACCEPTANCE: dict[str, Any] = {
    "transverse_width_range_mm": TRANSVERSE_RETAINING_BEAD_WIDTH_RANGE_MM,
    "axial_width_range_mm": AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM,
    "transverse_lower_width_tolerance_mm": (
        TRANSVERSE_RETAINING_BEAD_LOWER_WIDTH_TOLERANCE_MM),
    "axial_lower_width_tolerance_mm": (
        AXIAL_RETAINING_BEAD_LOWER_WIDTH_TOLERANCE_MM),
}


def _resolve_retaining_bead_acceptance(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one lane's retaining-bead acceptance from requirements."""
    defaults: dict[str, Any] = {
        "transverse_width_range_mm": (
            TRANSVERSE_RETAINING_BEAD_WIDTH_RANGE_MM),
        "axial_width_range_mm": AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM,
        "transverse_lower_width_tolerance_mm": (
            TRANSVERSE_RETAINING_BEAD_LOWER_WIDTH_TOLERANCE_MM),
        "axial_lower_width_tolerance_mm": (
            AXIAL_RETAINING_BEAD_LOWER_WIDTH_TOLERANCE_MM),
    }
    requirements = config.get("requirements")
    supplied = (
        requirements.get("retaining_bead_acceptance")
        if isinstance(requirements, Mapping) else None)
    if supplied is None:
        return defaults
    if not isinstance(supplied, Mapping) or set(supplied) != set(defaults):
        raise AuditError(
            "requirements.retaining_bead_acceptance must define exactly: "
            + ", ".join(sorted(defaults)))
    resolved: dict[str, Any] = {}
    for key in ("transverse_width_range_mm", "axial_width_range_mm"):
        pair = supplied[key]
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes))
                or len(pair) != 2):
            raise AuditError(
                f"retaining_bead_acceptance.{key} must be [low, high]")
        low = _float(pair[0], f"retaining_bead_acceptance.{key}[0]")
        high = _float(pair[1], f"retaining_bead_acceptance.{key}[1]")
        if not 0.0 < low < high:
            raise AuditError(
                f"retaining_bead_acceptance.{key} must satisfy 0 < low < high")
        resolved[key] = (low, high)
    for key in ("transverse_lower_width_tolerance_mm",
                "axial_lower_width_tolerance_mm"):
        value = _float(supplied[key], f"retaining_bead_acceptance.{key}")
        if value < 0.0:
            raise AuditError(
                f"retaining_bead_acceptance.{key} must be non-negative")
        resolved[key] = value
    return resolved
ANNULAR_COMPONENT_SEAM_WIDTH_MARGIN_MM = 0.04
ANNULAR_COMPONENT_SEAM_SEARCH_RAYS = 2.0
EVIDENCE_CELL_PX = 218
EVIDENCE_MARGIN_MM = 4.0
# Keep repository overrides fail-closed.  Bambu Studio accepts unknown JSON
# keys without reporting that they are inert, so every authority we intentionally
# inject must be registered here as a setting the audit understands and checks.
PROFILE_OVERRIDE_KEYS = {
    "machine": frozenset({
        "machine_pause_gcode",
        # The 0.6-mm lane runs the high-flow hotend: pin the nozzle
        # volume type so every preset resolves its High Flow variant
        # column (temps, pressure advance, volumetric ceilings) and the
        # device-side nozzle check matches at send time.
        "default_nozzle_volume_type",
    }),
    "process": frozenset({
        # The 0.6-mm lane bases its process on a stock 0.6-nozzle preset and
        # pins the release-wide 0.16/0.20 layer schedule through overrides so
        # both nozzle lanes share one Z stack and one pause table.
        "layer_height",
        "initial_layer_print_height",
        # Deterministic single-bead retaining skins: each lane pins the
        # Arachne minimum bead so the inset/curved-face skin wedges can never
        # split into sub-floor bead pairs (0.40 on the 0.4 lane, 0.51 on the
        # 0.6 lane).
        "min_bead_width",
        "wall_loops",
        "top_shell_layers",
        "bottom_shell_layers",
        "outer_wall_speed",
        "curr_bed_type",
        "sparse_infill_pattern",
        "sparse_infill_density",
        "wall_generator",
        "enable_support",
        "support_on_build_plate_only",
        "support_critical_regions_only",
        "support_remove_small_overhang",
        "precise_outer_wall",
        "detect_thin_wall",
        "ensure_vertical_shell_thickness",
        "detect_narrow_internal_solid_infill",
        "elefant_foot_compensation",
        "xy_hole_compensation",
    }),
    "filament": frozenset({
        "nozzle_temperature",
        "nozzle_temperature_initial_layer",
        "fan_max_speed",
        "overhang_fan_speed",
        "filament_max_volumetric_speed",
        "textured_plate_temp",
        "textured_plate_temp_initial_layer",
    }),
}
RELEASE_SITE_GEOMETRY_MM = {
    "magnet_diameter_mm": DEFAULT_MAGNET_SPEC.magnet_diameter_mm,
    "magnet_depth_mm": DEFAULT_MAGNET_SPEC.magnet_depth_mm,
    "cavity_diameter_mm": DEFAULT_MAGNET_SPEC.cavity_diameter_mm,
    "cavity_depth_mm": DEFAULT_MAGNET_SPEC.cavity_depth_mm,
    "face_skin_mm": DEFAULT_MAGNET_SPEC.face_skin_mm,
    "inner_skin_mm": DEFAULT_MAGNET_SPEC.inner_skin_mm,
    "captive_land_mm": DEFAULT_MAGNET_SPEC.captive_land_mm,
    "interface_gap_mm": DEFAULT_MAGNET_SPEC.interface_gap_mm,
    "roof_angle_deg": DEFAULT_MAGNET_SPEC.roof_angle_deg,
    "minimum_retaining_path_mm": DEFAULT_MAGNET_SPEC.retaining_path_mm,
}
# Derived per-interface separations: base is two qualified face skins plus
# the 0.05-mm solid spacing standoff; the curved carrier stations add their
# 0.14-mm cavity-face inset and the Obi-Wan shoulder/ring interfaces add
# their 0.15-mm inset.  Deriving from the contract keeps this map correct
# across skin respins (0.45 -> 0.52 dual-nozzle).
_PAIRED_BASE_MM = DEFAULT_MAGNET_SPEC.paired_face_separation_mm
PAIRED_MAGNET_FACE_SEPARATION_MM = {
    None: _PAIRED_BASE_MM,
    "standard_straight": _PAIRED_BASE_MM,
    "standard_curved": round(_PAIRED_BASE_MM + 0.14, 9),
    "shoulder": round(_PAIRED_BASE_MM + 0.15, 9),
    "ring": round(_PAIRED_BASE_MM + 0.15, 9),
}
SUPPORT_PROCESS_KEYS = (
    "enable_support",
    "support_on_build_plate_only",
    "support_critical_regions_only",
    "support_remove_small_overhang",
)
SUPPORTED_ARTIFACT_MATCHES = (
    {
        "state": "floor_stand",
        "variant": "Obi-Wan",
        "part": "obiwan_core_2_of_2_um_carrier",
    },
    {
        "state": "no_floor_stand",
        "variant": "Obi-Wan",
        "part": "obiwan_core_2_of_2_um_carrier",
    },
    {
        "state": "floor_stand",
        "variant": "Obi-Wan-split",
        "part": "obiwan_optional_lm_keyed_1_of_2_bottom",
    },
    {
        "state": "no_floor_stand",
        "variant": "Obi-Wan-split",
        "part": "obiwan_optional_lm_keyed_1_of_2_bottom",
    },
    {
        "state": "floor_stand",
        "variant": "Obi-Wan-split",
        "part": "obiwan_optional_lm_keyed_2_of_2_top",
    },
    {
        "state": "no_floor_stand",
        "variant": "Obi-Wan-split",
        "part": "obiwan_optional_lm_keyed_2_of_2_top",
    },
)
DUCT_SUPPORT_BLOCKER_MATCHES = SUPPORTED_ARTIFACT_MATCHES
EXPECTED_LM_SPLIT_SEAM_Y_MM = 172.481
EXPECTED_KEYED_LM_DUCT_REGION_NAMES = {
    "floor_stand": frozenset({
        "um_route_lumen",
        "t_route_lumen",
        "floor_lm_lane_lumen",
        "floor_um_lane_lumen",
        "floor_um_feed_relief",
        "floor_t_lane_lumen",
        "floor_t_feed_relief",
    }),
    "no_floor_stand": frozenset({
        "um_route_lumen",
        "t_route_lumen",
        "lm_internal_and_rear_exit_lumen",
        "no_floor_lm_entry_bore",
        "no_floor_t_entry_bore",
        "no_floor_um_entry_bore",
        "no_floor_t_entry_vestibule",
        "no_floor_um_entry_vestibule",
    }),
}
EXPECTED_UM_CARRIER_DUCT_REGION_NAMES = frozenset({
    "um_carrier_t_route_lumen",
})
MAGNET_INSERTION_PARK_BEGIN = "; MAGNET_INSERTION_PARK_BEGIN"
MAGNET_INSERTION_PARK_END = "; MAGNET_INSERTION_PARK_END"
MAGNET_INSERTION_PAUSE_COMMAND = "M400 U1"
MAGNET_INSERTION_CUSTOM_GCODE_TYPE = "Custom"


class AuditError(RuntimeError):
    """A release-blocking catalog, profile, slice, or toolpath error."""


def _canonical_json(data: Any) -> bytes:
    return (json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode()


_sha256_bytes = sha256_bytes


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pretty_json_bytes(data, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label} must be finite, got {value!r}")
    return result


def _vec3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AuditError(f"{label} must contain three numbers")
    return tuple(_float(item, label) for item in value)  # type: ignore[return-value]


def _unit_xy(value: Sequence[float], label: str) -> tuple[float, float]:
    x, y = float(value[0]), float(value[1])
    length = math.hypot(x, y)
    if length <= 1.0e-8:
        raise AuditError(f"{label} has no XY projection")
    return x / length, y / length


def _unit3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    x, y, z = _vec3(value, label)
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1.0e-8:
        raise AuditError(f"{label} is zero length")
    return x / length, y / length, z / length


def _lookup(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    """Return one catalog field without manufacturing a safety default."""
    if key not in mapping:
        raise AuditError(f"{label} is required")
    return mapping[key]


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return clean.strip("._-") or "artifact"


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _requires_duct_support_blocker(
    state: str, variant: str, part: str,
) -> bool:
    identity = {"state": state, "variant": variant, "part": part}
    return any(identity == match for match in DUCT_SUPPORT_BLOCKER_MATCHES)


def _expected_duct_region_names(
    *, state: str, variant: str, part: str,
) -> frozenset[str]:
    if (variant == "Obi-Wan"
            and part == "obiwan_core_2_of_2_um_carrier"):
        return EXPECTED_UM_CARRIER_DUCT_REGION_NAMES
    if (variant == "Obi-Wan-split"
            and part in {
                "obiwan_optional_lm_keyed_1_of_2_bottom",
                "obiwan_optional_lm_keyed_2_of_2_top",
            }):
        names = EXPECTED_KEYED_LM_DUCT_REGION_NAMES.get(state)
        if names is not None:
            return names
    raise AuditError(
        f"{state}:{variant}:{part}: no duct collision inventory authority")


def _normalize_duct_collision_contract(
    value: Any, *, artifact_id: str, state: str,
    variant: str, part: str, modifier_clearance_mm: float,
) -> dict[str, Any]:
    """Validate the OCC-free centerline authority used by the G-code gate."""
    if not isinstance(value, Mapping):
        raise AuditError(
            f"{artifact_id}: support blocker lacks a duct collision contract")
    if (value.get("schema_version") != 1
            or value.get("coordinate_space")
            != "authoritative_source_mm"):
        raise AuditError(
            f"{artifact_id}: duct collision contract has the wrong schema "
            "or coordinate space")
    split_half: str | None = None
    seam_y: float | None = None
    owner: str | None = None
    if part == "obiwan_core_2_of_2_um_carrier":
        owner = value.get("owner")
        if owner != "um_carrier":
            raise AuditError(
                f"{artifact_id}: UM carrier collision contract has the "
                "wrong owner")
        if "split_half" in value or "split_seam_y_mm" in value:
            raise AuditError(
                f"{artifact_id}: unsplit UM carrier contract contains "
                "keyed-split fields")
    else:
        split_half = value.get("split_half")
        expected_half = (
            "bottom" if part.endswith("1of2_bottom") else "top")
        if split_half != expected_half:
            raise AuditError(
                f"{artifact_id}: duct collision contract split half is "
                f"{split_half!r}, expected {expected_half!r}")
        seam_y = _float(
            value.get("split_seam_y_mm"),
            f"{artifact_id} duct collision split seam")
        if not math.isclose(
                seam_y, EXPECTED_LM_SPLIT_SEAM_Y_MM,
                abs_tol=1.0e-9, rel_tol=0.0):
            raise AuditError(
                f"{artifact_id}: duct collision split seam {seam_y:g} "
                f"differs from R6F authority "
                f"{EXPECTED_LM_SPLIT_SEAM_Y_MM:g}")
    contract_clearance = _float(
        value.get("modifier_clearance_mm"),
        f"{artifact_id} duct collision modifier clearance")
    if (contract_clearance <= 0.0
            or not math.isclose(
                contract_clearance, modifier_clearance_mm,
                abs_tol=1.0e-9, rel_tol=0.0)):
        raise AuditError(
            f"{artifact_id}: duct collision clearance differs from the "
            "support-blocker clearance")
    raw_regions = value.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise AuditError(
            f"{artifact_id}: duct collision contract has no regions")
    regions: list[dict[str, Any]] = []
    names: set[str] = set()
    total_points = 0
    for index, raw in enumerate(raw_regions):
        label = f"{artifact_id} duct collision region {index}"
        if not isinstance(raw, Mapping):
            raise AuditError(f"{label} is not an object")
        name = raw.get("name")
        if (not isinstance(name, str) or not name
                or name in names):
            raise AuditError(f"{label} has an invalid or duplicate name")
        names.add(name)
        if raw.get("kind") != "polyline_tube":
            raise AuditError(f"{label} is not a polyline_tube")
        radius = _float(raw.get("radius_mm"), f"{label} radius")
        if radius <= 0.0:
            raise AuditError(f"{label} radius must be positive")
        raw_points = raw.get("points_xyz_mm")
        if not isinstance(raw_points, list) or not raw_points:
            raise AuditError(f"{label} has no centerline points")
        points = [
            list(_vec3(point, f"{label} point {point_index}"))
            for point_index, point in enumerate(raw_points)
        ]
        total_points += len(points)
        if total_points > 200_000:
            raise AuditError(
                f"{artifact_id}: duct collision contract is unbounded")
        regions.append({
            "name": name,
            "kind": "polyline_tube",
            "radius_mm": radius,
            "points_xyz_mm": points,
        })
    expected_names = _expected_duct_region_names(
        state=state, variant=variant, part=part)
    if names != expected_names:
        raise AuditError(
            f"{artifact_id}: duct collision region inventory is incomplete: "
            f"expected={sorted(expected_names)}, actual={sorted(names)}")
    normalized = {
        "schema_version": 1,
        "coordinate_space": "authoritative_source_mm",
        "modifier_clearance_mm": contract_clearance,
        "regions": regions,
    }
    if owner is not None:
        normalized["owner"] = owner
    else:
        normalized["split_half"] = split_half
        normalized["split_seam_y_mm"] = seam_y
    return normalized


def _normalize_duct_support_blocker(
    *, artifact_id: str, state: str, variant: str, stl: Path,
    stl_sha256: str, part: str,
    source_to_stl_matrix: tuple[tuple[float, ...], ...],
) -> dict[str, Any]:
    """Bind the generated no-support volume to its exact printable STL."""
    blocker = (
        stl.parent.parent / "support_blockers"
        / f"{stl.stem}.support_blocker.stl"
    ).resolve()
    binding = blocker.with_suffix(".json")
    if not blocker.is_file() or not binding.is_file():
        raise AuditError(
            f"{artifact_id}: generated duct support blocker or binding is "
            f"missing: {blocker}, {binding}")
    payload = _load_json(binding)
    if not isinstance(payload, Mapping):
        raise AuditError(
            f"{artifact_id}: duct support-blocker binding is not an object")
    if (payload.get("schema_version") != 1
            or payload.get("kind") != "bambu_support_blocker"
            or payload.get("purpose")
            != "forbid_support_inside_functional_ducts"
            or payload.get("part") != part):
        raise AuditError(
            f"{artifact_id}: duct support-blocker binding has the wrong "
            "schema, purpose, or part")
    if Path(str(payload.get("support_blocker", ""))).name != blocker.name:
        raise AuditError(
            f"{artifact_id}: blocker binding names a different modifier STL")
    if Path(str(payload.get("main_stl", ""))).name != stl.name:
        raise AuditError(
            f"{artifact_id}: blocker binding names a different printable STL")
    blocker_sha256 = sha256_file(blocker)
    if (payload.get("main_stl_sha256") != stl_sha256
            or payload.get("support_blocker_sha256") != blocker_sha256):
        raise AuditError(
            f"{artifact_id}: support-blocker binding hashes differ from the "
            "release meshes")
    binding_matrix = _matrix4(
        payload.get("source_to_stl_matrix"),
        f"{artifact_id} support-blocker source-to-STL transform")
    if binding_matrix != source_to_stl_matrix:
        raise AuditError(
            f"{artifact_id}: support blocker and printable STL use different "
            "source-to-STL transforms")
    modifier_clearance = _float(
        payload.get("modifier_clearance_mm"),
        f"{artifact_id} support-blocker clearance")
    if modifier_clearance <= 0.0:
        raise AuditError(
            f"{artifact_id}: support-blocker clearance must be positive")
    duct_collision_contract = _normalize_duct_collision_contract(
        payload.get("duct_collision_contract"),
        artifact_id=artifact_id,
        state=state,
        variant=variant,
        part=part,
        modifier_clearance_mm=modifier_clearance,
    )
    return {
        "support_blocker": blocker,
        "support_blocker_sha256": blocker_sha256,
        "support_blocker_binding": binding,
        "support_blocker_binding_sha256": sha256_file(binding),
        "duct_collision_contract": duct_collision_contract,
    }


class PresetResolver:
    """Resolve Bambu system ``inherits``/``include`` chains exactly once."""

    def __init__(
        self,
        vendor_root: Path,
        *,
        extra_presets: Sequence[Path] = (),
    ):
        self.vendor_root = vendor_root.resolve()
        if not self.vendor_root.is_dir():
            raise AuditError(f"Bambu preset root does not exist: {vendor_root}")
        self.by_name: dict[str, list[Path]] = {}
        self.raw: dict[Path, dict[str, Any]] = {}
        indexed_paths = {
            path.resolve() for path in self.vendor_root.rglob("*.json")
        }
        for path in extra_presets:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                raise AuditError(
                    f"additional Bambu preset does not exist: {resolved}")
            indexed_paths.add(resolved)
        for path in sorted(indexed_paths):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            self.raw[path.resolve()] = data
            name = data.get("name")
            if isinstance(name, str) and name:
                self.by_name.setdefault(name, []).append(path.resolve())
        self.dependencies: set[Path] = set()

    def _find_named(self, name: str, type_hint: str | None, origin: Path) -> Path:
        candidates = list(self.by_name.get(name, ()))
        if type_hint:
            typed = [p for p in candidates
                     if self.raw[p].get("type") in (None, type_hint)]
            if typed:
                candidates = typed
        same_dir = [p for p in candidates if p.parent == origin.parent]
        if len(same_dir) == 1:
            return same_dir[0]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise AuditError(
                f"Bambu preset dependency {name!r} referenced by {origin} "
                f"was not found below {self.vendor_root}")
        raise AuditError(
            f"ambiguous Bambu preset dependency {name!r}: "
            + ", ".join(str(path) for path in candidates))

    def resolve(self, path: Path) -> dict[str, Any]:
        return self._resolve(path.resolve(), ())

    @staticmethod
    def _merge_values(
        base: Mapping[str, Any],
        overlay: Mapping[str, Any],
        *,
        origin: Path,
    ) -> dict[str, Any]:
        """Apply Bambu's per-slot ``nil`` inheritance while flattening."""
        merged = dict(base)
        for key, value in overlay.items():
            if key in ("inherits", "include"):
                continue
            if isinstance(value, list) and any(
                    str(item).strip().lower() == "nil" for item in value):
                parent = merged.get(key)
                value = [
                    parent[index]
                    if (str(item).strip().lower() == "nil"
                        and isinstance(parent, list)
                        and index < len(parent))
                    else item
                    for index, item in enumerate(value)
                ]
            elif isinstance(value, str) and value.strip().lower() == "nil":
                if key in merged:
                    continue
            merged[key] = copy.deepcopy(value)
        return merged

    def _resolve(self, path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        if path in stack:
            chain = " -> ".join(item.name for item in (*stack, path))
            raise AuditError(f"Bambu preset inheritance cycle: {chain}")
        try:
            child = self.raw[path]
        except KeyError as exc:
            raise AuditError(f"unindexed Bambu preset: {path}") from exc
        self.dependencies.add(path)
        type_hint = child.get("type")
        merged: dict[str, Any] = {}
        parent = child.get("inherits")
        if isinstance(parent, str) and parent.strip():
            parent_path = self._find_named(parent.strip(), type_hint, path)
            merged = self._merge_values(
                merged,
                self._resolve(parent_path, (*stack, path)),
                origin=parent_path)
        includes = child.get("include", [])
        if isinstance(includes, str):
            includes = [includes]
        if includes is None:
            includes = []
        if not isinstance(includes, list):
            raise AuditError(f"invalid include list in {path}: {includes!r}")
        for name in includes:
            if not isinstance(name, str) or not name:
                raise AuditError(f"invalid include name in {path}: {name!r}")
            include_path = self._find_named(name, type_hint, path)
            merged = self._merge_values(
                merged,
                self._resolve(include_path, (*stack, path)),
                origin=include_path)
        return self._merge_values(merged, child, origin=path)


def _default_bambu_system_root(vendor: str) -> Path:
    return (Path.home() / "Library" / "Application Support" / "BambuStudio"
            / "system" / vendor)


def _find_user_filament_preset(name: str) -> Path:
    """Resolve one exact saved Bambu filament preset on the local Mac."""
    if not isinstance(name, str) or not name.strip():
        raise AuditError("user_filament_preset must be a non-empty name")
    user_root = (
        Path.home() / "Library" / "Application Support" / "BambuStudio"
        / "user"
    )
    candidates = sorted(
        path.resolve()
        for path in user_root.glob(f"*/filament/{name.strip()}.json")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise AuditError(
            f"saved Bambu filament preset {name!r} resolved to "
            f"{len(candidates)} files below {user_root}; expected exactly one")
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(
            f"cannot read saved Bambu filament preset {candidates[0]}") from exc
    if not isinstance(payload, Mapping) or payload.get("name") != name.strip():
        raise AuditError(
            f"saved Bambu filament preset identity differs from {name!r}")
    return candidates[0]


def _find_bambu_binary(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("BAMBU_STUDIO_BIN")
    if env:
        candidates.append(Path(env).expanduser())
    found = shutil.which("bambu-studio") or shutil.which("BambuStudio")
    if found:
        candidates.append(Path(found))
    candidates.extend((
        Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio"),
        Path("/Applications/Bambu Studio.app/Contents/MacOS/BambuStudio"),
    ))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise AuditError("Bambu Studio CLI was not found; set BAMBU_STUDIO_BIN")


def _scalar(profile: Mapping[str, Any], key: str, label: str) -> float:
    value = profile.get(key)
    if isinstance(value, list):
        if not value:
            raise AuditError(f"resolved {label}.{key} is empty")
        value = value[0]
    return _float(value, f"resolved {label}.{key}")


def _boolish(value: Any) -> bool:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "enabled")
    return False


def _profile_value_equal(actual: Any, expected: Any) -> bool:
    """Compare Bambu JSON values without weakening vector cardinality."""
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_profile_value_equal(a, e)
                    for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(actual, list) and len(actual) == 1:
        # Bambu serializes single-extruder machine scalars as one-element
        # vectors (observed: default_nozzle_volume_type=['High Flow'] echoed
        # for the scalar 'High Flow' override).  Unwrap only for scalar
        # expectations; vector expectations keep strict cardinality above.
        return _profile_value_equal(actual[0], expected)
    if isinstance(expected, bool):
        return _boolish(actual) is expected
    if isinstance(expected, (int, float)):
        try:
            return math.isclose(
                _float(actual, "profile value"), float(expected),
                abs_tol=1.0e-9, rel_tol=0.0)
        except AuditError:
            return False
    return str(actual) == str(expected)


def _apply_profile_overrides(
    resolved: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Apply repository authority only after preset inheritance is flat."""
    result = {
        key: copy.deepcopy(dict(value)) for key, value in resolved.items()
    }
    if not isinstance(overrides, Mapping):
        raise AuditError(f"{label} must be an object")
    unexpected = sorted(set(overrides) - set(result))
    if unexpected:
        raise AuditError(
            f"{label} contains unsupported profile sections: {unexpected}")
    for section, values in overrides.items():
        if not isinstance(values, Mapping):
            raise AuditError(f"{label}.{section} must be an object")
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise AuditError(f"{label}.{section} has an invalid key")
            allowed = PROFILE_OVERRIDE_KEYS.get(section, frozenset())
            if key not in allowed:
                raise AuditError(
                    f"{label}.{section}.{key} is not a registered, "
                    "toolpath-audited Bambu setting")
            result[section][key] = copy.deepcopy(value)
    return result


def _assert_profile_overrides(
    resolved: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for section, values in overrides.items():
        for key, expected in values.items():
            actual = resolved[section].get(key)
            if not _profile_value_equal(actual, expected):
                raise AuditError(
                    f"{label} was not applied: {section}.{key} "
                    f"is {actual!r}, expected {expected!r}")


def _percent(value: Any, label: str) -> float:
    if isinstance(value, list):
        if not value:
            raise AuditError(f"{label} is empty")
        value = value[0]
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1]
    return _float(text, label)


def _validate_support_override_policy(config: Mapping[str, Any]) -> None:
    """Pin support off globally and on only for its exact safe allowlist."""
    catalog_mode = config.get("catalog_mode", "release")
    if catalog_mode not in {"release", "auxiliary"}:
        raise AuditError(
            "slicing profile catalog_mode must be release or auxiliary")
    requirements = config.get("requirements")
    if (not isinstance(requirements, Mapping)
            or requirements.get("support_enabled") is not False):
        raise AuditError(
            "requirements.support_enabled must explicitly be false by "
            "default")
    repo_overrides = config.get("repo_overrides")
    if not isinstance(repo_overrides, Mapping):
        raise AuditError("slicing profile repo_overrides must be an object")
    base_process = repo_overrides.get("process")
    if not isinstance(base_process, Mapping):
        raise AuditError("slicing profile repo_overrides.process is required")
    for key in SUPPORT_PROCESS_KEYS:
        if key not in base_process or _boolish(base_process[key]):
            raise AuditError(
                f"repo_overrides.process.{key} must explicitly be 0")

    rules = config.get("artifact_overrides")
    if not isinstance(rules, list):
        raise AuditError("artifact_overrides must be an array")
    actual_matches = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise AuditError(f"artifact_overrides[{index}] must be an object")
        process = rule.get("process", {})
        if not isinstance(process, Mapping):
            raise AuditError(
                f"artifact_overrides[{index}].process must be an object")
        present = set(process).intersection(SUPPORT_PROCESS_KEYS)
        if not present:
            continue
        if present != set(SUPPORT_PROCESS_KEYS):
            raise AuditError(
                f"artifact_overrides[{index}] must set all support keys")
        support_values = tuple(
            _boolish(process[key]) for key in SUPPORT_PROCESS_KEYS)
        required_value = catalog_mode == "release"
        if any(value != required_value for value in support_values):
            raise AuditError(
                f"artifact_overrides[{index}] support keys must all be "
                f"{1 if required_value else 0} in {catalog_mode} mode")
        match = rule.get("match")
        if not isinstance(match, Mapping):
            raise AuditError(
                f"artifact_overrides[{index}].match must be an object")
        actual_matches.append(dict(match))
    actual_match_keys = sorted(
        tuple(sorted(item.items())) for item in actual_matches)
    required_match_keys = sorted(
        tuple(sorted(item.items())) for item in SUPPORTED_ARTIFACT_MATCHES)
    if catalog_mode == "release" and actual_match_keys != required_match_keys:
        raise AuditError(
            "support overrides must target exactly the floor/no-floor "
            "Obi-Wan keyed LM split and UM carrier artifacts")
    if catalog_mode == "auxiliary" and actual_match_keys:
        # Auxiliary artifacts in this workflow are support-free by contract.
        # Explicit all-zero per-object rules are harmless but unnecessary;
        # the ready-project emitter pins the same four keys on every normal
        # object.  Rejecting a truthy rule above is the safety boundary.
        return


def _magnet_insertion_pause_policy(
    config: Mapping[str, Any], machine: Mapping[str, Any],
) -> dict[str, float | str | bool]:
    """Validate the physical, P2S-specific magnet insertion pause motion."""
    raw = config.get("magnet_insertion_pause")
    if not isinstance(raw, Mapping):
        raise AuditError("magnet_insertion_pause must be an object")
    park_z = _float(raw.get("park_z_mm"), "magnet insertion park Z")
    speed = _float(
        raw.get("z_travel_speed_mm_s"), "magnet insertion Z travel speed")
    bounds = config.get("machine_bounds_mm")
    if not isinstance(bounds, Mapping):
        raise AuditError("machine_bounds_mm must be an object")
    z_bounds = bounds.get("z")
    if (not isinstance(z_bounds, list) or len(z_bounds) != 2):
        raise AuditError("machine_bounds_mm.z must contain [min, max]")
    z_min = _float(z_bounds[0], "machine minimum Z")
    z_max = _float(z_bounds[1], "machine maximum Z")
    if not (z_min < park_z < z_max):
        raise AuditError(
            f"magnet insertion park Z {park_z:g} must be strictly inside "
            f"the P2S Z envelope {z_min:g}..{z_max:g}")
    max_z_speed = _scalar(machine, "machine_max_speed_z", "machine")
    if speed <= 0.0 or speed > max_z_speed + 1.0e-9:
        raise AuditError(
            f"magnet insertion Z travel speed {speed:g} mm/s exceeds the "
            f"P2S maximum {max_z_speed:g} mm/s")
    return {
        "custom_gcode_type": MAGNET_INSERTION_CUSTOM_GCODE_TYPE,
        "pause_command": MAGNET_INSERTION_PAUSE_COMMAND,
        "park_z_mm": park_z,
        "z_travel_speed_mm_s": speed,
        "z_travel_feedrate_mm_min": speed * 60.0,
        "restore_exact_pause_z": True,
        "xy_motion": "none",
    }


def _effective_profile_contract(
    resolved: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    enforced_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and summarize the exact settings sent to Bambu Studio."""
    req = config["requirements"]
    checks = {
        "nozzle diameter": (
            _scalar(resolved["machine"], "nozzle_diameter", "machine"),
            _float(req["nozzle_diameter_mm"], "required nozzle diameter")),
        "layer height": (
            _scalar(resolved["process"], "layer_height", "process"),
            _float(req["layer_height_mm"], "required layer height")),
        "first layer height": (
            _scalar(
                resolved["process"], "initial_layer_print_height", "process"),
            _float(
                req["first_layer_height_mm"], "required first layer height")),
        "outer wall width": (
            _scalar(resolved["process"], "outer_wall_line_width", "process"),
            _float(
                req["outer_wall_line_width_mm"], "required outer wall width")),
        "inner wall width": (
            _scalar(resolved["process"], "inner_wall_line_width", "process"),
            _float(
                req["inner_wall_line_width_mm"], "required inner wall width")),
    }
    for item, (actual, expected) in checks.items():
        if not math.isclose(actual, expected, abs_tol=1.0e-8):
            raise AuditError(
                f"resolved Bambu {item} {actual:g} != required {expected:g}")
    _assert_profile_overrides(
        resolved, enforced_overrides, label="repository profile override")
    process = resolved["process"]
    filament = resolved["filament"]
    machine = resolved["machine"]
    magnet_insertion_pause = _magnet_insertion_pause_policy(config, machine)
    wall_generator = str(process.get("wall_generator", "")).lower()
    required_wall_generator = str(req["wall_generator"]).lower()
    if required_wall_generator != "arachne":
        raise AuditError(
            "repository slicing requirements must pin wall_generator to "
            "'arachne'")
    if wall_generator != required_wall_generator:
        raise AuditError(
            f"resolved wall generator {wall_generator!r} is not "
            f"{required_wall_generator!r}")
    support_enabled = _boolish(process.get("enable_support"))
    support_on_build_plate_only = _boolish(
        process.get("support_on_build_plate_only"))
    support_critical_regions_only = _boolish(
        process.get("support_critical_regions_only"))
    support_remove_small_overhang = _boolish(
        process.get("support_remove_small_overhang"))
    if support_enabled and not (
            support_on_build_plate_only and support_critical_regions_only
            and support_remove_small_overhang):
        raise AuditError(
            "support-enabled artifact profiles must also enable "
            "support_on_build_plate_only and "
            "support_critical_regions_only and "
            "support_remove_small_overhang")
    if not support_enabled and (
            support_on_build_plate_only or support_critical_regions_only
            or support_remove_small_overhang):
        raise AuditError(
            "support scope flags must be disabled when enable_support is 0")
    model = machine.get("printer_model")
    if model != req["printer_model"]:
        raise AuditError(
            f"resolved printer model {model!r} is not "
            f"{req['printer_model']!r}")
    if str(machine.get("machine_pause_gcode", "")).strip() != "M400 U1":
        raise AuditError(
            "resolved machine_pause_gcode must be exactly 'M400 U1'")
    lane_acceptance = _resolve_retaining_bead_acceptance(config)
    return {
        "printer_model": model,
        "nozzle_diameter_mm": checks["nozzle diameter"][0],
        "retaining_bead_acceptance": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in lane_acceptance.items()
        },
        "layer_height_mm": checks["layer height"][0],
        "first_layer_height_mm": checks["first layer height"][0],
        "wall_generator": wall_generator,
        "outer_wall_line_width_mm": checks["outer wall width"][0],
        "inner_wall_line_width_mm": checks["inner wall width"][0],
        "wall_loops": int(round(_scalar(process, "wall_loops", "process"))),
        "top_shell_layers": int(round(
            _scalar(process, "top_shell_layers", "process"))),
        "bottom_shell_layers": int(round(
            _scalar(process, "bottom_shell_layers", "process"))),
        "outer_wall_speed_mm_s": _scalar(
            process, "outer_wall_speed", "process"),
        "bed_type": process.get("curr_bed_type"),
        "sparse_infill_pattern": process.get("sparse_infill_pattern"),
        "sparse_infill_density_percent": _percent(
            process.get("sparse_infill_density"),
            "resolved process.sparse_infill_density"),
        "precise_outer_wall": _boolish(process.get("precise_outer_wall")),
        "detect_thin_wall": _boolish(process.get("detect_thin_wall")),
        "ensure_vertical_shell_thickness": process.get(
            "ensure_vertical_shell_thickness"),
        "detect_narrow_internal_solid_infill": _boolish(
            process.get("detect_narrow_internal_solid_infill")),
        "elefant_foot_compensation_mm": _scalar(
            process, "elefant_foot_compensation", "process"),
        "xy_hole_compensation_mm": _scalar(
            process, "xy_hole_compensation", "process"),
        "xy_hole_compensation_policy": req.get(
            "xy_hole_compensation_policy"),
        "support_enabled": support_enabled,
        "support_on_build_plate_only": support_on_build_plate_only,
        "support_critical_regions_only": support_critical_regions_only,
        "support_remove_small_overhang": support_remove_small_overhang,
        "arc_fitting_enabled": _boolish(process.get("enable_arc_fitting")),
        "machine_pause_gcode": machine["machine_pause_gcode"],
        "magnet_insertion_pause": magnet_insertion_pause,
        "nozzle_temperature_c": _scalar(
            filament, "nozzle_temperature", "filament"),
        "nozzle_temperature_initial_layer_c": _scalar(
            filament, "nozzle_temperature_initial_layer", "filament"),
        "fan_max_speed_percent": _scalar(
            filament, "fan_max_speed", "filament"),
        "overhang_fan_speed_percent": _scalar(
            filament, "overhang_fan_speed", "filament"),
        "filament_max_volumetric_speed_mm3_s": _scalar(
            filament, "filament_max_volumetric_speed", "filament"),
        "textured_plate_temp_c": _scalar(
            filament, "textured_plate_temp", "filament"),
        "textured_plate_temp_initial_layer_c": _scalar(
            filament, "textured_plate_temp_initial_layer", "filament"),
        "filament": filament.get("name"),
    }


def _parse_bambu_studio_version(help_output: str) -> str:
    matches = re.findall(
        r"(?m)^BambuStudio-([0-9]+(?:\.[0-9]+){3}):\s*$", help_output)
    if len(matches) != 1:
        raise AuditError(
            "Bambu Studio --help did not contain exactly one "
            "BambuStudio-XX.XX.XX.XX banner")
    return matches[0]


def prepare_profiles(
    config_path: Path,
    output_dir: Path,
    *,
    system_root: Path | None,
    bambu_binary: Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1:
        raise AuditError(f"unsupported slicing profile schema in {config_path}")
    _validate_support_override_policy(config)
    root = (system_root or _default_bambu_system_root(config["vendor"])).resolve()
    user_filament_name = config.get("user_filament_preset")
    user_filament_path = (
        _find_user_filament_preset(user_filament_name)
        if user_filament_name is not None else None
    )
    if user_filament_path is not None:
        expected_user_sha = config.get("user_filament_preset_sha256")
        if (not isinstance(expected_user_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_user_sha)
                or sha256_file(user_filament_path) != expected_user_sha):
            raise AuditError(
                "saved user filament preset differs from the hash pinned "
                f"by {config_path}")
    resolver = PresetResolver(
        root,
        extra_presets=(
            (user_filament_path,) if user_filament_path is not None else ()
        ),
    )
    sources = {
        "machine": root / config["machine_preset"],
        "process": root / config["process_preset"],
        "filament": (
            user_filament_path
            if user_filament_path is not None
            else root / config["filament_preset"]
        ),
    }
    for label, path in sources.items():
        if path.resolve() not in resolver.raw:
            raise AuditError(f"configured {label} preset was not found: {path}")
    flattened = {
        label: resolver.resolve(path) for label, path in sources.items()
    }
    repo_overrides = config.get("repo_overrides")
    if not isinstance(repo_overrides, Mapping) or not repo_overrides:
        raise AuditError("slicing profile must define non-empty repo_overrides")
    resolved = _apply_profile_overrides(
        flattened, repo_overrides, label="repo_overrides")
    # Install this lane's retaining-bead acceptance before any audit reads
    # the shared mapping.  Each CLI invocation resolves exactly one profile,
    # so the install is a one-shot lane selection, not a mutable setting.
    RETAINING_BEAD_ACCEPTANCE.update(
        _resolve_retaining_bead_acceptance(config))
    effective = _effective_profile_contract(
        resolved, config, repo_overrides)

    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, data in resolved.items():
        path = profile_dir / f"resolved_{label}.json"
        _write_json(path, data)
        paths[label] = path
    dependency_records = []
    for path in sorted(resolver.dependencies):
        try:
            dependency_path = str(path.relative_to(root))
            dependency_scope = "system_vendor_root"
        except ValueError:
            dependency_path = str(path)
            dependency_scope = "saved_user_preset"
        dependency_records.append({
            "path": dependency_path,
            "scope": dependency_scope,
            "sha256": sha256_file(path),
        })
    try:
        version_run = subprocess.run(
            [str(bambu_binary), "--help"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30, check=False)
        if version_run.returncode != 0:
            raise AuditError(
                f"Bambu Studio --help exited {version_run.returncode}")
        version = _parse_bambu_studio_version(version_run.stdout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuditError(f"cannot execute Bambu Studio CLI: {exc}") from exc
    required_version = config.get("required_bambu_studio_version")
    if not isinstance(required_version, str) or not required_version:
        raise AuditError("required_bambu_studio_version is missing")
    if version != required_version:
        raise AuditError(
            f"Bambu Studio version {version!r} != required "
            f"{required_version!r}")
    audit_sources = _audit_source_hashes()
    identity = {
        "backend": "BambuStudio",
        "binary": str(bambu_binary),
        "binary_sha256": sha256_file(bambu_binary),
        "version": version,
        "required_version": required_version,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "system_vendor_root": str(root),
        "source_presets": {
            key: {
                "path": str(value.resolve()),
                "sha256": sha256_file(value),
            } for key, value in sources.items()
        },
        "resolution_dependencies": dependency_records,
        "resolved_profiles": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "repo_overrides": copy.deepcopy(repo_overrides),
        "effective": effective,
        "machine_bounds_mm": config["machine_bounds_mm"],
        "audit_sources": audit_sources,
    }
    identity["profile_set_sha256"] = _sha256_bytes(_canonical_json({
        key: record["sha256"]
        for key, record in identity["resolved_profiles"].items()
    }))
    _write_json(profile_dir / "profile_provenance.json", identity)
    return {
        "config": config,
        "identity": identity,
        "paths": paths,
        "resolved": resolved,
        "enforced_overrides": copy.deepcopy(repo_overrides),
        # Keep the original scalar for existing diagnostic consumers, while
        # binding every pure-Python authority used by the slice audit.
        "audit_script_sha256": audit_sources[str(FACADE_SOURCE)],
        "audit_source_sha256": audit_sources,
    }


def _artifact_profile_bundle(
    artifact: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize any exact artifact override as a hash-bound profile set."""
    rules = profile_bundle["config"].get("artifact_overrides", [])
    if not isinstance(rules, list):
        raise AuditError("artifact_overrides must be an array")
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise AuditError(f"artifact_overrides[{index}] must be an object")
        match = rule.get("match")
        if not isinstance(match, Mapping) or not match:
            raise AuditError(
                f"artifact_overrides[{index}].match must be a non-empty object")
        if any(key not in artifact or artifact[key] != expected
               for key, expected in match.items()):
            continue
        payload = {key: value for key, value in rule.items() if key != "match"}
        if not payload:
            raise AuditError(
                f"artifact_overrides[{index}] has no profile values")
        matches.append((index, payload))
    if len(matches) > 1:
        raise AuditError(
            f"{artifact['id']}: multiple artifact profile overrides match: "
            f"{[index for index, _payload in matches]}")
    if not matches:
        return dict(profile_bundle)

    index, override = matches[0]
    resolved = _apply_profile_overrides(
        profile_bundle["resolved"], override,
        label=f"artifact_overrides[{index}]")
    enforced = copy.deepcopy(profile_bundle["enforced_overrides"])
    for section, values in override.items():
        section_values = enforced.setdefault(section, {})
        if not isinstance(values, Mapping):
            raise AuditError(
                f"artifact_overrides[{index}].{section} must be an object")
        section_values.update(copy.deepcopy(dict(values)))
    effective = _effective_profile_contract(
        resolved, profile_bundle["config"], enforced)

    profile_dir = output_dir / "slice_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for section, data in resolved.items():
        path = profile_dir / f"resolved_{section}.json"
        _write_json(path, data)
        paths[section] = path
    identity = copy.deepcopy(profile_bundle["identity"])
    identity["effective"] = effective
    identity["artifact_override"] = {
        "rule_index": index,
        "match": copy.deepcopy(rules[index]["match"]),
        "values": copy.deepcopy(override),
    }
    identity["resolved_profiles"] = {
        section: {"path": str(path), "sha256": sha256_file(path)}
        for section, path in paths.items()
    }
    identity["profile_set_sha256"] = _sha256_bytes(_canonical_json({
        section: record["sha256"]
        for section, record in identity["resolved_profiles"].items()
    }))
    return {
        **profile_bundle,
        "identity": identity,
        "paths": paths,
        "resolved": resolved,
        "enforced_overrides": enforced,
    }


@dataclasses.dataclass(frozen=True)
class MeshFacts:
    triangle_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.bounds_min, self.bounds_max))  # type: ignore[return-value]


def inspect_stl(path: Path) -> MeshFacts:
    """Read binary or ASCII STL bounds without trimesh/OCC."""
    data = path.read_bytes()
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    count = 0
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + n * 50:
            offset = 84
            for _ in range(n):
                values = struct.unpack_from("<9f", data, offset + 12)
                for index in range(0, 9, 3):
                    for axis in range(3):
                        value = float(values[index + axis])
                        if not math.isfinite(value):
                            raise AuditError(f"non-finite vertex in {path}")
                        mins[axis] = min(mins[axis], value)
                        maxs[axis] = max(maxs[axis], value)
                count += 1
                offset += 50
            return MeshFacts(count, tuple(mins), tuple(maxs))  # type: ignore[arg-type]
    text = data.decode("ascii", errors="ignore")
    for match in re.finditer(
            r"(?im)^\s*vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            text):
        values = tuple(float(match.group(i)) for i in range(1, 4))
        for axis, value in enumerate(values):
            if not math.isfinite(value):
                raise AuditError(f"non-finite vertex in {path}")
            mins[axis] = min(mins[axis], value)
            maxs[axis] = max(maxs[axis], value)
        count += 1
    if count == 0 or count % 3:
        raise AuditError(f"{path} is not a recognized STL")
    return MeshFacts(count // 3, tuple(mins), tuple(maxs))  # type: ignore[arg-type]


PARAMETER_MODIFIER_PROCESS_KEYS = frozenset({
    "sparse_infill_density",
    "sparse_infill_pattern",
})


def _profile_relative_path(
    profile_bundle: Mapping[str, Any],
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuditError(f"{label} must stay below the profile directory")
    config_path = Path(profile_bundle["identity"]["config_path"]).resolve()
    return (config_path.parent / relative).resolve()


def _parameter_modifiers_for_artifact(
    artifact: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Resolve and hash-check exact Bambu parameter-modifier contracts."""
    rules = profile_bundle["config"].get("parameter_modifiers", [])
    if not isinstance(rules, list):
        raise AuditError("parameter_modifiers must be an array")
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise AuditError(f"parameter_modifiers[{index}] must be an object")
        match = rule.get("match")
        if not isinstance(match, Mapping) or not match:
            raise AuditError(
                f"parameter_modifiers[{index}].match must be a non-empty "
                "object")
        if all(artifact.get(key) == expected
               for key, expected in match.items()):
            matches.append((index, rule))
    if len(matches) > 1:
        raise AuditError(
            f"{artifact['id']}: multiple parameter modifiers match: "
            f"{[index for index, _rule in matches]}")
    if not matches:
        return ()

    index, rule = matches[0]
    contract_path = _profile_relative_path(
        profile_bundle, rule.get("contract"),
        label=f"parameter_modifiers[{index}].contract")
    contract = _load_json(contract_path)
    expected_match = dict(rule["match"])
    if (contract.get("schema_version") != 1
            or contract.get("kind") != "bambu_parameter_modifier"
            or contract.get("subtype") != "modifier_part"
            or contract.get("artifact_match") != expected_match):
        raise AuditError(
            f"parameter modifier contract identity is invalid: "
            f"{contract_path}")
    process = contract.get("process")
    if (not isinstance(process, Mapping)
            or set(process) != PARAMETER_MODIFIER_PROCESS_KEYS
            or process.get("sparse_infill_density") != "100%"
            or process.get("sparse_infill_pattern") != "zig-zag"):
        raise AuditError(
            f"{contract_path}: bridge/root modifier must pin exactly "
            "100% zig-zag infill")

    source_stl = _profile_relative_path(
        profile_bundle, contract.get("source_stl"),
        label=f"{contract_path}.source_stl")
    print_sidecar = _profile_relative_path(
        profile_bundle, contract.get("print_sidecar"),
        label=f"{contract_path}.print_sidecar")
    modifier_stl = _profile_relative_path(
        profile_bundle, contract.get("modifier_stl"),
        label=f"{contract_path}.modifier_stl")
    for path, expected_sha, label in (
        (source_stl, contract.get("source_stl_sha256"), "source STL"),
        (print_sidecar, contract.get("print_sidecar_sha256"), "print sidecar"),
        (modifier_stl, contract.get("modifier_stl_sha256"), "modifier STL"),
    ):
        if (not path.is_file() or not isinstance(expected_sha, str)
                or sha256_file(path) != expected_sha):
            raise AuditError(
                f"{contract_path}: {label} is missing or hash-mismatched")
    artifact_stl = Path(artifact["stl"])
    artifact_sidecar = Path(artifact["print_sidecar"])
    if (sha256_file(artifact_stl) != contract["source_stl_sha256"]
            or sha256_file(artifact_sidecar)
            != contract["print_sidecar_sha256"]):
        raise AuditError(
            f"{artifact['id']}: parameter modifier is bound to different "
            "release geometry")
    modifier_mesh = inspect_stl(modifier_stl)
    if modifier_mesh.triangle_count != int(contract.get(
            "triangle_count", -1)):
        raise AuditError(
            f"{contract_path}: modifier triangle count is inconsistent")
    return ({
        "rule_index": index,
        "match": expected_match,
        "contract": contract_path,
        "contract_sha256": sha256_file(contract_path),
        "role": contract.get("role"),
        "path": modifier_stl,
        "sha256": sha256_file(modifier_stl),
        "triangle_count": modifier_mesh.triangle_count,
        "process": dict(process),
    },)


def _validate_parameter_modifier_coverage(
    artifacts: Sequence[Mapping[str, Any]],
    profile_bundle: Mapping[str, Any],
) -> None:
    """Require every configured modifier rule to bind one release artifact."""
    rules = profile_bundle["config"].get("parameter_modifiers", [])
    if not isinstance(rules, list):
        raise AuditError("parameter_modifiers must be an array")
    matched_ids: set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping) or not isinstance(
                rule.get("match"), Mapping):
            raise AuditError(f"parameter_modifiers[{index}] is invalid")
        match = rule["match"]
        matches = [
            artifact for artifact in artifacts
            if all(artifact.get(key) == expected
                   for key, expected in match.items())
        ]
        if len(matches) != 1:
            raise AuditError(
                f"parameter_modifiers[{index}] match {dict(match)!r} "
                f"resolved to {len(matches)} catalog artifacts; expected "
                "exactly one")
        artifact_id = str(matches[0]["id"])
        if artifact_id in matched_ids:
            raise AuditError(
                f"multiple parameter modifiers target {artifact_id}")
        matched_ids.add(artifact_id)
        _parameter_modifiers_for_artifact(matches[0], profile_bundle)


def _validate_profile_artifact_scope(
    artifacts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    """Reject use of a material profile outside its explicit artifact scope."""
    scope = config.get("artifact_scope")
    if scope is None:
        return
    if not isinstance(scope, list) or not scope:
        raise AuditError("artifact_scope must be a non-empty array")
    matches = []
    for index, match in enumerate(scope):
        if not isinstance(match, Mapping) or not match:
            raise AuditError(
                f"artifact_scope[{index}] must be a non-empty object")
        matches.append(dict(match))
    violations = [
        str(artifact.get("id", "<unknown>"))
        for artifact in artifacts
        if not any(all(artifact.get(key) == expected
                       for key, expected in match.items())
                   for match in matches)
    ]
    if violations:
        raise AuditError(
            "slicing profile is not authorized for artifact(s): "
            + ", ".join(violations))


def _matrix4(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    if (not isinstance(value, list) or len(value) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in value)):
        raise AuditError(f"{label} must be a 4x4 matrix")
    return tuple(tuple(_float(v, label) for v in row) for row in value)


def _transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> tuple[float, float, float]:
    result = []
    for row in matrix[:3]:
        result.append(sum(row[i] * point[i] for i in range(3)) + row[3])
    return tuple(result)  # type: ignore[return-value]


def _transform_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    result = []
    for row in matrix[:3]:
        result.append(sum(row[i] * vector[i] for i in range(3)))
    return tuple(result)  # type: ignore[return-value]


def _site_in_bambu_bed_space(
    site: Mapping[str, Any], matrix: BambuMatrix4,
) -> dict[str, Any]:
    """Apply Bambu's audited Rz+XY arrangement to one print-space station.

    The catalog remains authoritative in exported-STL coordinates.  Bambu may
    rotate and center that front-down STL in the bed plane, so all G-code ROIs,
    radial axes, and evidence overlays must use the full archived affine—not a
    translation inferred from an axis-aligned bounding box.
    """
    transformed = dict(site)
    original: dict[str, list[float]] = {}
    for key in (
            "print_cavity_center_xyz_mm",
            "print_seated_magnet_center_xyz_mm",
            "print_actual_face_xyz_mm"):
        if key not in site:
            continue
        value = _vec3(site[key], key)
        original[key] = list(value)
        transformed[key] = transform_bambu_point(matrix, value)
    for key in (
            "print_material_inward_xyz",
            "print_marked_pole_axis_xyz",
            "print_insertion_direction_xyz"):
        if key not in site:
            continue
        value = _vec3(site[key], key)
        original[key] = list(value)
        transformed[key] = transform_bambu_vector(matrix, value)
    transformed["catalog_stl_print_space"] = original
    return transformed


def _site_mapping(site: Mapping[str, Any]) -> dict[str, Any]:
    facts = site.get("facts")
    if facts is None:
        facts = {}
    if not isinstance(facts, dict):
        raise AuditError("site.facts must be an object")
    merged = dict(facts)
    merged.update(site)
    return merged


def _source_site_contract(site: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical exact source-space contract used for split equivalence."""
    closure = str(site.get("closure_kind", ""))
    contract: dict[str, Any] = {
        "name": str(site.get("name", "")),
        "closure_kind": closure,
        "cavity_center_xyz_mm": list(_vec3(
            _lookup(site, "cavity_center_xyz_mm", "cavity_center_xyz"),
            "source cavity center")),
        "seated_magnet_center_xyz_mm": list(_vec3(
            _lookup(
                site, "seated_magnet_center_xyz_mm",
                "seated_magnet_center_xyz"),
            "source seated magnet center")),
        "marked_pole_axis_xyz": list(_vec3(
            site["marked_pole_axis_xyz"], "source marked pole axis")),
        "insertion_direction_xyz": list(_vec3(_required(
            site, "insertion_direction_xyz",
            "source insertion direction"), "source insertion direction")),
        "installed_marked_pole_axis_xyz": list(_vec3(_required(
            site, "installed_marked_pole_axis_xyz",
            "installed marked pole axis"), "installed marked pole axis")),
        "cavity_bury_roof_start_print_z_mm": _float(_lookup(
            site, "cavity_bury_roof_start_print_z_mm",
            "roof_start_print_z_mm", "bury_plane_print_z_mm"),
            "source bury plane"),
        "roof_apex_print_z_mm": _float(
            site["roof_apex_print_z_mm"], "source roof apex"),
        "magnet_diameter_mm": _float(_required(
            site, "magnet_diameter_mm", "source magnet diameter"),
            "source magnet diameter"),
        "magnet_depth_mm": _float(_required(
            site, "magnet_depth_mm", "source magnet depth"),
            "source magnet depth"),
        "cavity_diameter_mm": _float(_required(
            site, "cavity_diameter_mm", "source cavity diameter"),
            "source cavity diameter"),
        "cavity_depth_mm": _float(_required(
            site, "cavity_depth_mm", "source cavity depth"),
            "source cavity depth"),
        "face_skin_mm": _float(_required(
            site, "face_skin_mm", "source face skin"), "source face skin"),
        "inner_skin_mm": _float(_required(
            site, "inner_skin_mm", "source inner skin"),
            "source inner skin"),
        "roof_angle_deg": _float(_required(
            site, "roof_angle_deg", "source roof angle"),
            "source roof angle"),
        "polarity_instruction": str(_required(
            site, "polarity_instruction", "source polarity instruction")),
        "magnet_count": int(_float(_required(
            site, "magnet_count", "source magnet count"),
            "source magnet count")),
        "structural_load_credit_n": _float(_required(
            site, "structural_load_credit_n",
            "source structural load credit"),
            "source structural load credit"),
    }
    for key in (
            "minimum_retaining_path_mm", "captive_land_mm",
            "interface_gap_mm", "paired_magnet_face_separation_mm"):
        contract[key] = _float(
            _required(site, key, f"source {key}"), f"source {key}")
    if "interface_kind" in site:
        contract["interface_kind"] = str(site["interface_kind"])
        contract["carrier_cavity_face_inset_mm"] = _float(
            _required(
                site, "carrier_cavity_face_inset_mm",
                "source carrier cavity-face inset"),
            "source carrier cavity-face inset")
    for key in ("roof_height_mm", "side_wall_margin_mm"):
        if key in site:
            contract[key] = _float(site[key], f"source {key}")
    if closure == "transverse_gable_45deg":
        contract["actual_face_xyz_mm"] = list(_vec3(
            site["actual_face_xyz_mm"], "source actual face"))
        contract["material_inward_xyz"] = list(_vec3(
            site["material_inward_xyz"], "source material inward"))
    return contract


def _validate_cavity_audit_proxies(
    artifacts: Sequence[dict[str, Any]],
) -> None:
    """Bind every declared oversized site to one exact same-state split site."""
    by_id = {artifact["id"]: artifact for artifact in artifacts}
    if len(by_id) != len(artifacts):
        raise AuditError("duplicate artifact IDs prevent proxy validation")
    for artifact in artifacts:
        mode = artifact.get("p2s_printability")
        declared = artifact.get("cavity_audit_proxies")
        if mode not in (None, "not_printable_oversize"):
            raise AuditError(
                f"{artifact['id']}: unsupported p2s_printability {mode!r}")
        if mode != "not_printable_oversize":
            if declared not in (None, []):
                raise AuditError(
                    f"{artifact['id']}: cavity proxies are allowed only for "
                    "an explicitly oversized non-P2S artifact")
            artifact["cavity_audit_proxies"] = []
            continue
        if not isinstance(declared, list) or not declared:
            raise AuditError(
                f"{artifact['id']}: oversized artifact lacks cavity proxies")
        source_sites = {site["name"]: site for site in artifact["sites"]}
        if len(source_sites) != len(artifact["sites"]):
            raise AuditError(f"{artifact['id']}: duplicate source site names")
        clean = []
        seen_sources: set[str] = set()
        seen_targets: set[tuple[str, str]] = set()
        for index, raw in enumerate(declared):
            if not isinstance(raw, dict):
                raise AuditError(
                    f"{artifact['id']}: proxy {index} is not an object")
            source_name = str(raw.get("site", ""))
            target_id = str(raw.get("artifact_id", ""))
            target_name = str(raw.get("proxy_site", ""))
            if not source_name or not target_id or not target_name:
                raise AuditError(
                    f"{artifact['id']}: proxy {index} lacks site/artifact/site")
            if source_name in seen_sources:
                raise AuditError(
                    f"{artifact['id']}: site {source_name} has multiple proxies")
            if (target_id, target_name) in seen_targets:
                raise AuditError(
                    f"{artifact['id']}: proxy target is reused: "
                    f"{target_id}/{target_name}")
            source = source_sites.get(source_name)
            target = by_id.get(target_id)
            if source is None or target is None:
                raise AuditError(
                    f"{artifact['id']}: unresolved proxy "
                    f"{source_name} -> {target_id}/{target_name}")
            if (target["state"] != artifact["state"]
                    or target["variant"] != "Obi-Wan-split"):
                raise AuditError(
                    f"{artifact['id']}: proxy {target_id} is not a same-state "
                    "Obi-Wan-split artifact")
            if target.get("p2s_printability") == "not_printable_oversize":
                raise AuditError(
                    f"{artifact['id']}: proxy {target_id} is itself declared "
                    "not P2S-printable")
            target_matches = [
                site for site in target["sites"]
                if site["name"] == target_name
            ]
            if len(target_matches) != 1:
                raise AuditError(
                    f"{artifact['id']}: proxy site {target_id}/{target_name} "
                    "is missing or ambiguous")
            proxy_site = target_matches[0]
            if source["source_contract_sha256"] != proxy_site[
                    "source_contract_sha256"]:
                raise AuditError(
                    f"{artifact['id']}: source-space cavity contract differs "
                    f"for {source_name} and {target_id}/{target_name}")
            seen_sources.add(source_name)
            seen_targets.add((target_id, target_name))
            clean.append({
                "site": source_name,
                "artifact_id": target_id,
                "proxy_site": target_name,
                "source_contract_sha256": source["source_contract_sha256"],
            })
        if seen_sources != set(source_sites):
            missing = sorted(set(source_sites) - seen_sources)
            raise AuditError(
                f"{artifact['id']}: not every monolith site has one proxy: "
                f"{missing}")
        artifact["cavity_audit_proxies"] = clean


def _print_field(
    site: Mapping[str, Any],
    artifact: Mapping[str, Any],
    name: str,
    *,
    vector: bool = False,
) -> tuple[float, float, float]:
    aliases = (
        f"print_{name}", f"stl_{name}", f"print_space_{name}",
    )
    explicit: tuple[float, float, float] | None = None
    print_space = site.get("print_space")
    if isinstance(print_space, dict):
        value = _lookup(print_space, name, f"{name}_mm", *aliases)
        if value is not None:
            explicit = _vec3(value, f"site print_space.{name}")
    value = _lookup(site, *aliases)
    if value is not None and explicit is None:
        explicit = _vec3(value, f"site {name}")
    source = _lookup(site, name, f"{name}_mm")
    matrix_value = _lookup(
        artifact, "source_to_stl_matrix", "print_transform_matrix",
        "source_to_print_matrix")
    if source is not None and matrix_value is not None:
        matrix = _matrix4(matrix_value, "artifact source-to-STL transform")
        source_vec = _vec3(source, f"site source {name}")
        transformed = (_transform_vector(matrix, source_vec) if vector
                       else _transform_point(matrix, source_vec))
        if explicit is not None and any(
                abs(a - b) > 1.0e-6
                for a, b in zip(explicit, transformed)):
            raise AuditError(
                f"site {site.get('name', '<unnamed>')} print-space {name} "
                f"{explicit} disagrees with source-to-STL transform "
                f"{transformed}")
        return explicit or transformed
    if explicit is not None:
        return explicit
    raise AuditError(
        f"site {site.get('name', '<unnamed>')} lacks post-export print-space {name}")


def normalize_catalog(
    catalog_path: Path, *, enforce_release_inventory: bool = True,
) -> dict[str, Any]:
    try:
        catalog_bytes = catalog_path.read_bytes()
        data = json.loads(catalog_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {catalog_path}: {exc}") from exc
    catalog_input_sha256 = _sha256_bytes(catalog_bytes)
    if not isinstance(data, dict):
        raise AuditError("release catalog root must be an object")
    root_required = {
        "schema_version", "schema_sha256", "catalog_kind", "generated_by",
        "print_contract", "geometry", "inventory", "exclusions",
        "source_revision", "artifacts",
    }
    missing_root = sorted(root_required - set(data))
    if missing_root:
        raise AuditError(
            f"release catalog lacks required root fields: {missing_root}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise AuditError(
            f"release catalog schema must be {SCHEMA_VERSION}, got "
            f"{data.get('schema_version')!r}")
    if data.get("catalog_kind") != "released_pause_and_bury_captive_magnets":
        raise AuditError("release catalog_kind is not the production contract")
    if not CATALOG_SCHEMA.is_file():
        raise AuditError(f"catalog schema is missing: {CATALOG_SCHEMA}")
    try:
        schema_bytes = CATALOG_SCHEMA.read_bytes()
        schema = json.loads(schema_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(
            f"cannot read catalog schema {CATALOG_SCHEMA}: {exc}") from exc
    catalog_schema_sha256 = _sha256_bytes(schema_bytes)
    if data.get("schema_sha256") != catalog_schema_sha256:
        raise AuditError("release catalog is stale against its JSON schema")
    try:
        validate_json_schema(data, schema)
    except JsonSchemaSubsetError as exc:
        raise AuditError(
            f"release catalog violates its JSON schema: {exc}") from exc
    source_revision = data.get("source_revision")
    if (not isinstance(source_revision, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_revision)):
        raise AuditError(
            "release source_revision must be one non-null 64-hex digest")
    if data.get("print_contract") != RELEASE_ACOUSTIC_PRINT_CONTRACT:
        raise AuditError(
            "release print_contract must exactly require every released "
            "acoustic part front-face-down with in-bed Z rotation only")
    if not isinstance(data.get("geometry"), dict):
        raise AuditError("release geometry must be an object")
    if not isinstance(data.get("inventory"), dict):
        raise AuditError("release inventory must be an object")
    exclusions = data.get("exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        raise AuditError("release exclusions must be a non-empty list")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AuditError("release catalog must contain a non-empty artifacts list")
    normalized = []
    seen = set()
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, dict):
            raise AuditError(f"artifact {index} must be an object")
        artifact_required = {
            "id", "state", "variant", "part", "stl", "stl_sha256",
            "print_sidecar", "print_sidecar_sha256", "source_files",
            "source_file_sha256",
            "print_orientation", "rotation_deg", "source_to_stl_matrix",
            "sites",
        }
        missing_artifact = sorted(artifact_required - set(raw))
        if missing_artifact:
            raise AuditError(
                f"artifact {index} lacks required fields: {missing_artifact}")
        state = str(raw.get("state", "shared"))
        variant = str(raw.get("variant", "unknown"))
        part = str(_lookup(raw, "part", "name") or "").strip()
        if not part:
            raise AuditError(f"artifact {index} lacks part/name")
        artifact_id = str(raw.get("id") or f"{state}:{variant}:{part}")
        if artifact_id in seen:
            raise AuditError(f"duplicate artifact id {artifact_id!r}")
        seen.add(artifact_id)
        orientation = str(raw.get("print_orientation", ""))
        if orientation != "front_face_down":
            raise AuditError(
                f"{artifact_id}: print_orientation must be front_face_down, "
                f"got {orientation!r}")
        try:
            source_to_stl_matrix = validate_front_down_transform(
                raw, label=f"catalog artifact {artifact_id}")
        except FrontDownContractError as exc:
            raise AuditError(str(exc)) from exc
        rotation = {
            "x": _float(raw["rotation_deg"]["x"], "catalog X rotation"),
            "z": _float(raw["rotation_deg"]["z"], "catalog Z rotation"),
        }
        stl_value = _lookup(raw, "stl", "stl_path", "path")
        if not isinstance(stl_value, str):
            raise AuditError(f"{artifact_id}: missing STL path")
        stl = _resolve_path(stl_value, catalog_path.parent)
        stl_sha = str(raw["stl_sha256"])
        print_sidecar_value = raw["print_sidecar"]
        print_sidecar_sha = str(raw["print_sidecar_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", stl_sha):
            raise AuditError(f"{artifact_id}: invalid catalog STL SHA-256")
        if (not isinstance(print_sidecar_value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", print_sidecar_sha)):
            raise AuditError(
                f"{artifact_id}: invalid print-transform authority binding")
        print_sidecar = _resolve_path(
            print_sidecar_value, catalog_path.parent)
        sites_raw = raw.get("sites")
        if not isinstance(sites_raw, list) or not sites_raw:
            raise AuditError(f"{artifact_id}: no magnet sites")
        sites = []
        for site_index, raw_site in enumerate(sites_raw):
            if not isinstance(raw_site, dict):
                raise AuditError(f"{artifact_id}: site {site_index} is not an object")
            site = _site_mapping(raw_site)
            site_name = str(site.get("name") or f"site_{site_index + 1}")
            closure = str(site.get("closure_kind", ""))
            if closure not in (
                    "transverse_gable_45deg",
                    "axis_parallel_conical_45deg",
                    "axis_opposed_conical_45deg"):
                raise AuditError(
                    f"{artifact_id}/{site_name}: unsupported closure {closure!r}")
            bury = _lookup(
                site, "cavity_bury_roof_start_print_z_mm",
                "roof_start_print_z_mm", "bury_plane_print_z_mm")
            apex = _lookup(site, "roof_apex_print_z_mm")
            if bury is None or apex is None:
                raise AuditError(
                    f"{artifact_id}/{site_name}: missing bury plane or roof apex")
            cavity_center = _print_field(
                site, raw, "cavity_center_xyz_mm")
            seated_center = _print_field(
                site, raw, "seated_magnet_center_xyz_mm")
            marked_axis = _print_field(
                site, raw, "marked_pole_axis_xyz", vector=True)
            insertion_direction = _print_field(
                site, raw, "insertion_direction_xyz", vector=True)
            if any(not math.isclose(
                    actual, expected, abs_tol=1.0e-9, rel_tol=0.0)
                    for actual, expected in zip(
                        insertion_direction,
                        PRINT_INSERTION_DIRECTION_XYZ,
                        strict=True)):
                raise AuditError(
                    f"{artifact_id}/{site_name}: insertion direction must "
                    "be print [0, 0, -1] (vertically downward from +Z); "
                    f"got {insertion_direction}")
            source_contract = _source_site_contract(site)
            for key, expected in RELEASE_SITE_GEOMETRY_MM.items():
                actual = _float(_required(
                    site, key, f"{artifact_id}/{site_name}: {key}"), key)
                if not math.isclose(actual, expected, abs_tol=1.0e-9):
                    raise AuditError(
                        f"{artifact_id}/{site_name}: {key} must be "
                        f"{expected:.3f}, got {actual:.6f}")
            interface_kind = site.get("interface_kind")
            if interface_kind is not None:
                interface_kind = str(interface_kind)
            interface_profile = site.get("interface_profile")
            if interface_profile is not None:
                interface_profile = str(interface_profile)
            is_obiwan = variant.startswith("Obi-Wan")
            expected_interface_kind = None
            if is_obiwan:
                if interface_profile is not None:
                    raise AuditError(
                        f"{artifact_id}/{site_name}: interface_profile is "
                        "reserved for standard/slim pairs")
                expected_interface_kind = (
                    "shoulder" if site_name.startswith("lm_lower_")
                    else "ring")
                if interface_kind != expected_interface_kind:
                    raise AuditError(
                        f"{artifact_id}/{site_name}: interface_kind must be "
                        f"{expected_interface_kind!r}, got {interface_kind!r}")
            elif interface_kind is not None:
                raise AuditError(
                    f"{artifact_id}/{site_name}: interface_kind is reserved "
                    "for Obi-Wan carrier/wing pairs")
            elif interface_profile not in (
                    None, "standard_straight", "standard_curved"):
                raise AuditError(
                    f"{artifact_id}/{site_name}: unsupported standard/slim "
                    f"interface_profile {interface_profile!r}")
            separation_profile = (
                interface_kind if is_obiwan else interface_profile)
            expected_pair_separation = PAIRED_MAGNET_FACE_SEPARATION_MM[
                separation_profile]
            actual_pair_separation = _float(_required(
                site, "paired_magnet_face_separation_mm",
                f"{artifact_id}/{site_name}: paired magnet-face separation"),
                "paired magnet-face separation")
            if not math.isclose(
                    actual_pair_separation, expected_pair_separation,
                    abs_tol=1.0e-9, rel_tol=0.0):
                raise AuditError(
                    f"{artifact_id}/{site_name}: paired magnet-face "
                    f"separation must be {expected_pair_separation:.3f}, "
                    f"got {actual_pair_separation:.6f}")
            if is_obiwan:
                expected_inset = (
                    0.15 if expected_interface_kind in {"ring", "shoulder"}
                    else 0.0)
                actual_inset = _float(_required(
                    site, "carrier_cavity_face_inset_mm",
                    f"{artifact_id}/{site_name}: carrier cavity-face inset"),
                    "carrier cavity-face inset")
                if not math.isclose(
                        actual_inset, expected_inset,
                        abs_tol=1.0e-9, rel_tol=0.0):
                    raise AuditError(
                        f"{artifact_id}/{site_name}: carrier cavity-face "
                        f"inset must be {expected_inset:.3f}, got "
                        f"{actual_inset:.6f}")
            polarity = str(_required(
                site, "polarity_instruction",
                f"{artifact_id}/{site_name}: polarity instruction")).strip()
            if not polarity:
                raise AuditError(
                    f"{artifact_id}/{site_name}: polarity instruction is empty")
            magnet_count = int(_float(_required(
                site, "magnet_count",
                f"{artifact_id}/{site_name}: magnet count"),
                "magnet count"))
            if magnet_count != 1:
                raise AuditError(
                    f"{artifact_id}/{site_name}: magnet_count must be 1")
            structural_credit = _float(_required(
                site, "structural_load_credit_n",
                f"{artifact_id}/{site_name}: structural load credit"),
                "structural load credit")
            if abs(structural_credit) > 1.0e-12:
                raise AuditError(
                    f"{artifact_id}/{site_name}: magnets receive zero "
                    "structural-load credit")
            installed_axis = _vec3(_required(
                site, "installed_marked_pole_axis_xyz",
                f"{artifact_id}/{site_name}: installed marked-pole axis"),
                "installed marked-pole axis")
            normalized_site: dict[str, Any] = {
                "name": site_name,
                "closure_kind": closure,
                "cavity_bury_roof_start_print_z_mm": _float(bury, "bury plane"),
                "roof_apex_print_z_mm": _float(apex, "roof apex"),
                "print_cavity_center_xyz_mm": cavity_center,
                "print_seated_magnet_center_xyz_mm": seated_center,
                "print_marked_pole_axis_xyz": marked_axis,
                "print_insertion_direction_xyz": insertion_direction,
                "magnet_diameter_mm": source_contract["magnet_diameter_mm"],
                "magnet_depth_mm": source_contract["magnet_depth_mm"],
                "cavity_diameter_mm": source_contract["cavity_diameter_mm"],
                "cavity_depth_mm": source_contract["cavity_depth_mm"],
                "face_skin_mm": source_contract["face_skin_mm"],
                "inner_skin_mm": source_contract["inner_skin_mm"],
                "roof_angle_deg": source_contract["roof_angle_deg"],
                "captive_land_mm": source_contract["captive_land_mm"],
                "interface_gap_mm": source_contract["interface_gap_mm"],
                "paired_magnet_face_separation_mm": source_contract[
                    "paired_magnet_face_separation_mm"],
                "minimum_retaining_path_mm": source_contract[
                    "minimum_retaining_path_mm"],
                "polarity_instruction": polarity,
                "installed_marked_pole_axis_xyz": installed_axis,
                "magnet_count": magnet_count,
                "structural_load_credit_n": structural_credit,
                "expected_pause_marker_z_mm": (
                    None if site.get("expected_pause_marker_z_mm") is None
                    else _float(site["expected_pause_marker_z_mm"],
                                "expected pause marker")),
                "source_contract": source_contract,
                "source_contract_sha256": _sha256_bytes(
                    _canonical_json(source_contract)),
            }
            if interface_kind is not None:
                normalized_site.update({
                    "interface_kind": interface_kind,
                    "carrier_cavity_face_inset_mm": source_contract[
                        "carrier_cavity_face_inset_mm"],
                })
            if closure == "transverse_gable_45deg":
                normalized_site["print_actual_face_xyz_mm"] = _print_field(
                    site, raw, "actual_face_xyz_mm")
                normalized_site["print_material_inward_xyz"] = _print_field(
                    site, raw, "material_inward_xyz", vector=True)
            sites.append(normalized_site)
        source_values = raw.get("source_files")
        if (not isinstance(source_values, list) or not source_values
                or any(not isinstance(value, str) or not value
                       for value in source_values)):
            raise AuditError(f"{artifact_id}: source_files must be non-empty")
        source_hash_values = raw.get("source_file_sha256")
        if not isinstance(source_hash_values, dict):
            raise AuditError(
                f"{artifact_id}: source_file_sha256 must be an object")
        if (len(set(source_values)) != len(source_values)
                or set(source_hash_values) != set(source_values)):
            raise AuditError(
                f"{artifact_id}: source_file_sha256 must bind every exact "
                "source_files entry once")
        source_files = []
        source_file_sha256: dict[Path, str] = {}
        for value in source_values:
            digest = source_hash_values.get(value)
            if (not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise AuditError(
                    f"{artifact_id}: invalid source hash for {value!r}")
            source_path = _resolve_path(value, catalog_path.parent)
            if source_path in source_file_sha256:
                raise AuditError(
                    f"{artifact_id}: source_files entries alias the same "
                    f"resolved path: {source_path}")
            source_files.append(source_path)
            source_file_sha256[source_path] = digest
        auxiliary_bindings: dict[str, Any] = {}
        for path_key, hash_key in (
                ("transaction_manifest", "transaction_manifest_sha256"),
                ("facts", "facts_sha256"),
                ("stage_manifest", "stage_manifest_sha256")):
            path_value = raw.get(path_key)
            hash_value = raw.get(hash_key)
            if path_value is None and hash_value is None:
                continue
            if (not isinstance(path_value, str) or not path_value
                    or not isinstance(hash_value, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", hash_value)):
                raise AuditError(
                    f"{artifact_id}: invalid {path_key} path/hash binding")
            auxiliary_bindings[path_key] = _resolve_path(
                path_value, catalog_path.parent)
            auxiliary_bindings[hash_key] = hash_value
        if variant in ("Obi-Wan-Flat", "Obi-Wan-Graded") and not {
                "transaction_manifest", "transaction_manifest_sha256",
                "facts", "facts_sha256"} <= set(auxiliary_bindings):
            raise AuditError(
                f"{artifact_id}: flat/graded artifact lacks facts/transaction "
                "manifest hash bindings")
        if variant in ("Obi-Wan", "Obi-Wan-split") and not {
                "stage_manifest", "stage_manifest_sha256"
        } <= set(auxiliary_bindings):
            raise AuditError(
                f"{artifact_id}: Obi-Wan artifact lacks staged-build manifest "
                "hash binding")
        support_blocker_binding: dict[str, Any] = {}
        if _requires_duct_support_blocker(state, variant, part):
            support_blocker_binding = _normalize_duct_support_blocker(
                artifact_id=artifact_id,
                state=state,
                variant=variant,
                stl=stl,
                stl_sha256=stl_sha,
                part=part,
                source_to_stl_matrix=source_to_stl_matrix,
            )
        normalized.append({
            "id": artifact_id,
            "state": state,
            "variant": variant,
            "part": part,
            "stl": stl,
            "stl_catalog_value": stl_value,
            "stl_catalog_sha256": stl_sha,
            "print_sidecar": print_sidecar,
            "print_sidecar_catalog_value": print_sidecar_value,
            "print_sidecar_sha256": print_sidecar_sha,
            "print_orientation": orientation,
            "rotation_deg": rotation,
            "source_to_stl_matrix": source_to_stl_matrix,
            "sites": sites,
            "source_files": source_files,
            "source_file_sha256": source_file_sha256,
            "catalog_source_revision": source_revision,
            "catalog_sha256": catalog_input_sha256,
            "p2s_printability": raw.get("p2s_printability"),
            "cavity_audit_proxies": raw.get("cavity_audit_proxies"),
            "catalog_record": raw,
            **auxiliary_bindings,
            **support_blocker_binding,
        })
    _validate_cavity_audit_proxies(normalized)
    inventory = data["inventory"]
    declared_artifacts = inventory.get("artifact_count")
    declared_magnets = inventory.get("magnet_count")
    actual_magnets = sum(
        int(site["magnet_count"])
        for artifact in normalized for site in artifact["sites"])
    if declared_artifacts != len(normalized) or declared_magnets != actual_magnets:
        raise AuditError(
            "catalog inventory does not match its normalized contents: "
            f"declared={declared_artifacts}/{declared_magnets}, "
            f"actual={len(normalized)}/{actual_magnets}")
    if enforce_release_inventory and (
            len(normalized) != EXPECTED_RELEASE_ARTIFACT_COUNT
            or actual_magnets != EXPECTED_RELEASE_MAGNET_COUNT):
        raise AuditError(
            "production release inventory must remain "
            f"{EXPECTED_RELEASE_ARTIFACT_COUNT} artifacts / "
            f"{EXPECTED_RELEASE_MAGNET_COUNT} captive stations; got "
            f"{len(normalized)} / {actual_magnets}")
    result = dict(data)
    result["artifacts"] = normalized
    result["_catalog_sha256"] = catalog_input_sha256
    result["_catalog_schema_sha256"] = catalog_schema_sha256
    if sha256_file(catalog_path) != catalog_input_sha256:
        raise AuditError("release catalog changed while it was normalized")
    if sha256_file(CATALOG_SCHEMA) != catalog_schema_sha256:
        raise AuditError("release catalog schema changed while it was normalized")
    return result


def _validate_artifact_bindings(artifact: Mapping[str, Any]) -> None:
    """Reject any STL or print-transform authority changed after cataloging."""
    stl: Path = artifact["stl"]
    authority: Path = artifact["print_sidecar"]
    if not stl.is_file():
        raise AuditError(f"{artifact['id']}: STL is missing: {stl}")
    if sha256_file(stl) != artifact["stl_catalog_sha256"]:
        raise AuditError(f"{artifact['id']}: STL hash differs from catalog")
    if not authority.is_file():
        raise AuditError(
            f"{artifact['id']}: print-transform authority is missing: "
            f"{authority}")
    if sha256_file(authority) != artifact["print_sidecar_sha256"]:
        raise AuditError(
            f"{artifact['id']}: print-transform authority hash differs "
            "from catalog")
    try:
        sidecar = validate_print_sidecar(stl, authority)
    except (FrontDownContractError, OSError) as exc:
        raise AuditError(
            f"{artifact['id']}: invalid adjacent print sidecar: {exc}") from exc
    if sidecar.get("stl_sha256") != artifact["stl_catalog_sha256"]:
        raise AuditError(
            f"{artifact['id']}: sidecar STL hash differs from catalog")
    if sidecar.get("print_orientation") != artifact["print_orientation"]:
        raise AuditError(
            f"{artifact['id']}: sidecar orientation differs from catalog")
    sidecar_rotation = sidecar.get("rotation_deg")
    if sidecar_rotation != artifact["rotation_deg"]:
        raise AuditError(
            f"{artifact['id']}: sidecar rotation differs from catalog")
    try:
        sidecar_matrix = _matrix4(
            sidecar.get("source_to_stl_matrix"),
            f"{artifact['id']} sidecar source-to-STL transform")
    except AuditError as exc:
        raise AuditError(
            f"{artifact['id']}: invalid sidecar transform: {exc}") from exc
    if sidecar_matrix != artifact["source_to_stl_matrix"]:
        raise AuditError(
            f"{artifact['id']}: sidecar matrix differs from catalog")
    source_bindings = artifact.get("source_file_sha256")
    if (not isinstance(source_bindings, Mapping)
            or set(source_bindings) != set(artifact["source_files"])):
        raise AuditError(
            f"{artifact['id']}: normalized source hash bindings are incomplete")
    for source in artifact["source_files"]:
        expected = source_bindings[source]
        if not source.is_file():
            raise AuditError(
                f"{artifact['id']}: bound source file is missing: {source}")
        if sha256_file(source) != expected:
            raise AuditError(
                f"{artifact['id']}: source hash differs from catalog: "
                f"{source}")
    for path_key, hash_key in (
            ("transaction_manifest", "transaction_manifest_sha256"),
            ("facts", "facts_sha256"),
            ("stage_manifest", "stage_manifest_sha256")):
        path = artifact.get(path_key)
        expected = artifact.get(hash_key)
        if path is None and expected is None:
            continue
        if (not isinstance(path, Path) or not path.is_file()
                or sha256_file(path) != expected):
            raise AuditError(
                f"{artifact['id']}: {path_key} hash differs from catalog")
    blocker_keys = {
        "support_blocker", "support_blocker_sha256",
        "support_blocker_binding", "support_blocker_binding_sha256",
        "duct_collision_contract",
    }
    requires_blocker = _requires_duct_support_blocker(
        str(artifact["state"]), str(artifact["variant"]),
        str(artifact["part"]))
    present_blocker_keys = blocker_keys.intersection(artifact)
    if requires_blocker and present_blocker_keys != blocker_keys:
        raise AuditError(
            f"{artifact['id']}: duct support blocker binding is "
            "incomplete")
    if not requires_blocker and present_blocker_keys:
        raise AuditError(
            f"{artifact['id']}: unexpected support-blocker binding")
    if requires_blocker:
        blocker = artifact["support_blocker"]
        binding = artifact["support_blocker_binding"]
        if (not isinstance(blocker, Path) or not blocker.is_file()
                or sha256_file(blocker)
                != artifact["support_blocker_sha256"]):
            raise AuditError(
                f"{artifact['id']}: support-blocker STL hash differs from "
                "its release binding")
        if (not isinstance(binding, Path) or not binding.is_file()
                or sha256_file(binding)
                != artifact["support_blocker_binding_sha256"]):
            raise AuditError(
                f"{artifact['id']}: support-blocker metadata hash differs "
                "from its release binding")
        payload = _load_json(binding)
        if (not isinstance(payload, Mapping)
                or payload.get("main_stl_sha256")
                != artifact["stl_catalog_sha256"]
                or payload.get("support_blocker_sha256")
                != artifact["support_blocker_sha256"]
                or Path(str(payload.get("main_stl", ""))).name != stl.name
                or Path(str(payload.get("support_blocker", ""))).name
                != blocker.name
                or _matrix4(
                    payload.get("source_to_stl_matrix"),
                    f"{artifact['id']} support-blocker transform")
                != artifact["source_to_stl_matrix"]
                or _normalize_duct_collision_contract(
                    payload.get("duct_collision_contract"),
                    artifact_id=str(artifact["id"]),
                    state=str(artifact["state"]),
                    variant=str(artifact["variant"]),
                    part=str(artifact["part"]),
                    modifier_clearance_mm=_float(
                        payload.get("modifier_clearance_mm"),
                        f"{artifact['id']} support-blocker clearance"),
                ) != artifact["duct_collision_contract"]):
            raise AuditError(
                f"{artifact['id']}: support-blocker metadata no longer "
                "binds the staged printable and modifier meshes")


def _copy_hash_bound_file(
    source: Path, destination: Path, expected_sha256: str, label: str,
) -> None:
    """Copy one release input and prove both sides stayed byte-identical."""
    if not source.is_file():
        raise AuditError(f"{label} is missing: {source}")
    if sha256_file(source) != expected_sha256:
        raise AuditError(f"{label} changed before immutable staging")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise AuditError(f"{label} staging copy differs from release bytes")
    temporary.replace(destination)
    destination.chmod(0o444)
    if sha256_file(source) != expected_sha256:
        raise AuditError(f"{label} changed while it was being staged")


def _source_snapshot(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for artifact in artifacts:
        for path in artifact["source_files"]:
            key = str(path)
            if key in snapshot:
                if snapshot[key] != artifact["source_file_sha256"][path]:
                    raise AuditError(
                        f"conflicting catalog hashes for shared source: {path}")
                continue
            snapshot[key] = artifact["source_file_sha256"][path]
    return snapshot


def _stage_release_inputs(
    catalog_path: Path,
    artifacts: Sequence[Mapping[str, Any]],
    stage_root: Path,
    *,
    expected_catalog_sha256: str | None = None,
    expected_catalog_schema_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze the exact catalog/STL/sidecar bytes consumed by the slicer.

    The normalized catalog still owns semantic validation, while every Bambu
    process reads a private, read-only STL copy.  This prevents a concurrent
    remote promotion from mixing two release revisions within one audit.
    """
    stage_root.mkdir(parents=True, exist_ok=True)
    catalog_sha = sha256_file(catalog_path)
    if (expected_catalog_sha256 is not None
            and catalog_sha != expected_catalog_sha256):
        raise AuditError("release catalog changed before immutable staging")
    staged_catalog = stage_root / "captive_magnet_release_catalog.json"
    _copy_hash_bound_file(
        catalog_path, staged_catalog, catalog_sha, "release catalog")
    schema_sha = sha256_file(CATALOG_SCHEMA)
    if (expected_catalog_schema_sha256 is not None
            and schema_sha != expected_catalog_schema_sha256):
        raise AuditError(
            "release catalog schema changed before immutable staging")
    staged_schema = stage_root / CATALOG_SCHEMA.name
    _copy_hash_bound_file(
        CATALOG_SCHEMA, staged_schema, schema_sha, "release catalog schema")
    staged_artifacts: list[dict[str, Any]] = []
    originals: list[Mapping[str, Any]] = []
    for index, artifact in enumerate(sorted(
            artifacts, key=lambda item: item["id"])):
        _validate_artifact_bindings(artifact)
        originals.append(artifact)
        artifact_dir = stage_root / "artifacts" / (
            f"{index:03d}_{_slug(artifact['id'])}")
        staged_stl = artifact_dir / artifact["stl"].name
        staged_sidecar = artifact_dir / artifact["print_sidecar"].name
        _copy_hash_bound_file(
            artifact["stl"], staged_stl,
            artifact["stl_catalog_sha256"],
            f"{artifact['id']} STL")
        _copy_hash_bound_file(
            artifact["print_sidecar"], staged_sidecar,
            artifact["print_sidecar_sha256"],
            f"{artifact['id']} print sidecar")
        staged = dict(artifact)
        staged["release_stl"] = artifact["stl"]
        staged["release_print_sidecar"] = artifact["print_sidecar"]
        staged["stl"] = staged_stl
        staged["print_sidecar"] = staged_sidecar
        if "support_blocker" in artifact:
            staged_blocker = (
                artifact_dir / "modifiers" / artifact["support_blocker"].name)
            staged_blocker_binding = (
                artifact_dir / "modifiers"
                / artifact["support_blocker_binding"].name)
            _copy_hash_bound_file(
                artifact["support_blocker"], staged_blocker,
                artifact["support_blocker_sha256"],
                f"{artifact['id']} support-blocker STL")
            _copy_hash_bound_file(
                artifact["support_blocker_binding"],
                staged_blocker_binding,
                artifact["support_blocker_binding_sha256"],
                f"{artifact['id']} support-blocker binding")
            staged["release_support_blocker"] = artifact[
                "support_blocker"]
            staged["release_support_blocker_binding"] = artifact[
                "support_blocker_binding"]
            staged["support_blocker"] = staged_blocker
            staged["support_blocker_binding"] = staged_blocker_binding
        staged_sources: list[Path] = []
        staged_source_hashes: dict[Path, str] = {}
        for source_index, source in enumerate(artifact["source_files"]):
            expected = artifact["source_file_sha256"][source]
            staged_source = (
                artifact_dir / "sources"
                / f"{source_index:02d}_{source.name}")
            _copy_hash_bound_file(
                source, staged_source, expected,
                f"{artifact['id']} source {source.name}")
            staged_sources.append(staged_source)
            staged_source_hashes[staged_source] = expected
        staged["release_source_files"] = artifact["source_files"]
        staged["source_files"] = staged_sources
        staged["source_file_sha256"] = staged_source_hashes
        for path_key, hash_key in (
                ("transaction_manifest", "transaction_manifest_sha256"),
                ("facts", "facts_sha256"),
                ("stage_manifest", "stage_manifest_sha256")):
            if path_key not in artifact:
                continue
            source = artifact[path_key]
            destination = artifact_dir / "metadata" / source.name
            _copy_hash_bound_file(
                source, destination, artifact[hash_key],
                f"{artifact['id']} {path_key}")
            staged[f"release_{path_key}"] = source
            staged[path_key] = destination
        _validate_artifact_bindings(staged)
        staged_artifacts.append(staged)
        # Detect a replacement that raced either copy.
        _validate_artifact_bindings(artifact)
    if sha256_file(catalog_path) != catalog_sha:
        raise AuditError("release catalog changed during immutable staging")
    return {
        "catalog_path": staged_catalog,
        "catalog_sha256": catalog_sha,
        "catalog_schema_path": staged_schema,
        "catalog_schema_sha256": schema_sha,
        "artifacts": staged_artifacts,
        "original_artifacts": originals,
        "source_files": _source_snapshot(originals),
    }


def _verify_staged_release_inputs(
    staged: Mapping[str, Any], original_catalog_path: Path,
) -> None:
    """Recheck staged and live authorities immediately before publication."""
    expected_catalog_sha = staged["catalog_sha256"]
    if sha256_file(staged["catalog_path"]) != expected_catalog_sha:
        raise AuditError("immutable staged catalog changed during slicing")
    if sha256_file(original_catalog_path) != expected_catalog_sha:
        raise AuditError("release catalog changed during slicing")
    if (sha256_file(staged["catalog_schema_path"])
            != staged["catalog_schema_sha256"]
            or sha256_file(CATALOG_SCHEMA)
            != staged["catalog_schema_sha256"]):
        raise AuditError("release catalog schema changed during slicing")
    for artifact in staged["artifacts"]:
        _validate_artifact_bindings(artifact)
    for artifact in staged["original_artifacts"]:
        _validate_artifact_bindings(artifact)
    for raw_path, expected_sha in staged["source_files"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise AuditError(
                f"release source changed during slicing: {path}")


def _verify_profile_inputs(
    profile_bundle: Mapping[str, Any], bambu: Path,
) -> None:
    """Reject profile or slicer-binary drift across a long audit run."""
    identity = profile_bundle["identity"]
    expected_audit_sources = profile_bundle.get("audit_source_sha256")
    if not isinstance(expected_audit_sources, Mapping):
        raise AuditError("captive-magnet audit source hashes are missing")
    current_audit_sources = _audit_source_hashes()
    if current_audit_sources != dict(expected_audit_sources):
        raise AuditError("captive-magnet audit source changed during slicing")
    if sha256_file(bambu) != identity["binary_sha256"]:
        raise AuditError("Bambu Studio binary changed during slicing")
    for label, path in profile_bundle["paths"].items():
        expected = identity["resolved_profiles"][label]["sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditError(
                f"resolved {label} profile changed during slicing")
    config_path = Path(identity["config_path"])
    if (not config_path.is_file()
            or sha256_file(config_path) != identity["config_sha256"]):
        raise AuditError("slicing-profile contract changed during slicing")
    for record in identity["source_presets"].values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AuditError(f"Bambu source preset changed during slicing: {path}")
    vendor_root = Path(identity["system_vendor_root"])
    for record in identity["resolution_dependencies"]:
        path = vendor_root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AuditError(
                f"Bambu preset dependency changed during slicing: {path}")


def _require_record_file(
    path_value: Any, digest_value: Any, label: str,
) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise AuditError(f"{label} path is missing")
    if (not isinstance(digest_value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest_value)):
        raise AuditError(f"{label} SHA-256 is missing or invalid")
    path = Path(path_value)
    if not path.is_file() or path.stat().st_size == 0:
        raise AuditError(f"{label} file is missing or empty: {path}")
    if sha256_file(path) != digest_value:
        raise AuditError(f"{label} file differs from its recorded SHA-256")
    return path


def _validate_complete_release(
    catalog: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]] = (),
    *,
    enforce_expected_inventory: bool = True,
    require_ready_projects: bool = False,
) -> None:
    """Require exact, passing, hash-backed coverage before publication."""
    # Local import keeps the owner DAG acyclic: artifact emission depends on
    # release validation, while completeness only needs this one record view.
    from artifact_emit import _pause_groups

    if failures:
        raise AuditError(
            f"canonical manifest publication blocked by {len(failures)} "
            "slice exception(s)")
    artifacts = catalog.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AuditError("canonical publication requires a normalized catalog")
    inventory = catalog.get("inventory", {})
    expected_artifact_count = len(artifacts)
    expected_magnet_count = sum(
        int(site["magnet_count"])
        for artifact in artifacts for site in artifact["sites"])
    if (inventory.get("artifact_count") != expected_artifact_count
            or inventory.get("magnet_count") != expected_magnet_count):
        raise AuditError("catalog inventory changed before publication")
    if enforce_expected_inventory and (
            expected_artifact_count != EXPECTED_RELEASE_ARTIFACT_COUNT
            or expected_magnet_count != EXPECTED_RELEASE_MAGNET_COUNT):
        raise AuditError(
            "canonical publication requires the exact production inventory "
            f"of {EXPECTED_RELEASE_ARTIFACT_COUNT} artifacts / "
            f"{EXPECTED_RELEASE_MAGNET_COUNT} stations")
    expected_by_id = {artifact["id"]: artifact for artifact in artifacts}
    if len(expected_by_id) != expected_artifact_count:
        raise AuditError("canonical catalog contains duplicate artifact ids")
    record_ids = [str(record.get("id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise AuditError("canonical slice records contain duplicate artifact ids")
    if set(record_ids) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(record_ids))
        unexpected = sorted(set(record_ids) - set(expected_by_id))
        raise AuditError(
            "canonical slice coverage is not exact; "
            f"missing={missing}, unexpected={unexpected}")

    covered_station_count = 0
    pause_station_count = 0
    for record in records:
        artifact = expected_by_id[record["id"]]
        if record.get("input", {}).get(
                "stl_sha256") != artifact["stl_catalog_sha256"]:
            raise AuditError(
                f"{record['id']}: sliced STL hash differs from catalog")
        expected_source_hashes = {
            str(path): digest
            for path, digest in artifact["source_file_sha256"].items()
        }
        actual_source_hashes = {
            item.get("path"): item.get("sha256")
            for item in record.get("input", {}).get("source_files", ())
            if isinstance(item, Mapping)
        }
        if actual_source_hashes != expected_source_hashes:
            raise AuditError(
                f"{record['id']}: source-file evidence differs from catalog")
        expected_sites = {site["name"] for site in artifact["sites"]}
        if artifact.get("p2s_printability") == "not_printable_oversize":
            if (record.get("audit_mode") != "exact_split_proxy_coverage"
                    or record.get("status") != OVERSIZE_COVERED_STATUS
                    or record.get("cavity_audit_coverage", {}).get(
                        "pass") is not True):
                raise AuditError(
                    f"{record['id']}: oversized monolith lacks passing exact "
                    "split coverage")
            covered = {
                item.get("site")
                for item in record.get(
                    "cavity_audit_coverage", {}).get("sites", ())
                if isinstance(item, Mapping)
            }
            if covered != expected_sites:
                raise AuditError(
                    f"{record['id']}: oversized station coverage is incomplete")
            if _pause_groups(record):
                raise AuditError(
                    f"{record['id']}: oversized monolith emitted a fake pause")
            covered_station_count += len(covered)
            continue

        if (record.get("audit_mode") != "actual_p2s_slice"
                or record.get("status") != "pass"):
            raise AuditError(
                f"{record['id']}: actual P2S slice did not pass")
        site_records = record.get("sites")
        if not isinstance(site_records, list):
            raise AuditError(f"{record['id']}: site audit list is missing")
        actual_sites = [
            item.get("site", {}).get("name")
            for item in site_records if isinstance(item, Mapping)
        ]
        if (len(actual_sites) != len(set(actual_sites))
                or set(actual_sites) != expected_sites):
            raise AuditError(
                f"{record['id']}: actual station coverage is not exact")
        evidence = record.get("evidence", {})
        _require_record_file(
            evidence.get("svg"), evidence.get("svg_sha256"),
            f"{record['id']} SVG evidence")
        png = evidence.get("png", {})
        if not isinstance(png, Mapping):
            raise AuditError(f"{record['id']}: PNG evidence record is missing")
        _require_record_file(
            png.get("path"), png.get("sha256"),
            f"{record['id']} PNG evidence")
        slicer = record.get("slicer", {})
        _require_record_file(
            slicer.get("gcode"), slicer.get("gcode_sha256"),
            f"{record['id']} G-code")
        _require_record_file(
            slicer.get("result_json"), slicer.get("result_sha256"),
            f"{record['id']} Bambu result")
        _require_record_file(
            slicer.get("project_3mf"), slicer.get("project_3mf_sha256"),
            f"{record['id']} audited Bambu 3MF")
        if require_ready_projects:
            ready = slicer.get("ready_project", {})
            if not isinstance(ready, Mapping) or ready.get("status") != "pass":
                raise AuditError(
                    f"{record['id']}: authoritative publication requires a "
                    "passing ready-to-print project")
            if not isinstance(ready.get("output_fingerprint"), str) or not re.fullmatch(
                    r"[0-9a-f]{64}", ready["output_fingerprint"]):
                raise AuditError(
                    f"{record['id']}: ready-to-print output fingerprint is "
                    "missing or invalid")
            if Path(str(ready.get("project_3mf", ""))).name != READY_3MF_FILENAME:
                raise AuditError(
                    f"{record['id']}: primary ready project must be named "
                    f"{READY_3MF_FILENAME}")
            for path_key, hash_key, item in (
                    ("custom_gcodes_json", "custom_gcodes_sha256",
                     "custom magnet park/pause/restore JSON"),
                    ("result_json", "result_sha256", "ready Bambu result"),
                    ("gcode", "gcode_sha256", "ready G-code"),
                    ("project_3mf", "project_3mf_sha256",
                     "ready-to-print 3MF")):
                _require_record_file(
                    ready.get(path_key), ready.get(hash_key),
                    f"{record['id']} {item}")
        pause_count = sum(
            int(group["magnet_count"]) for group in _pause_groups(record))
        if pause_count != len(expected_sites):
            raise AuditError(
                f"{record['id']}: pause coverage does not equal its stations")
        pause_station_count += pause_count
        covered_station_count += len(actual_sites)
    if covered_station_count != expected_magnet_count:
        raise AuditError(
            "canonical station coverage total differs from catalog: "
            f"{covered_station_count} != {expected_magnet_count}")
    expected_pause_count = sum(
        len(artifact["sites"]) for artifact in artifacts
        if artifact.get("p2s_printability") != "not_printable_oversize")
    if pause_station_count != expected_pause_count:
        raise AuditError(
            "canonical pause total differs from printable catalog stations: "
            f"{pause_station_count} != {expected_pause_count}")


def _validate_manifest_bundle(
    paths: Mapping[str, Path],
    *,
    expected_artifact_count: int,
    expected_magnet_count: int,
    enforce_release_polarity_contract: bool,
) -> None:
    if expected_artifact_count <= 0 or expected_magnet_count <= 0:
        raise AuditError("staged manifest inventory expectation is invalid")
    if {path.name for path in paths.values()} != set(
            CANONICAL_MANIFEST_FILENAMES):
        raise AuditError("staged manifest bundle has unexpected filenames")
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise AuditError(f"staged {label} manifest is missing or empty")
    manifest = _load_json(paths["json"])
    if manifest.get("authoritative") is not True:
        raise AuditError("staged JSON manifest is not authoritative")
    summary = manifest.get("summary", {})
    if (summary.get("failed_artifacts") != 0
            or summary.get("requested_artifact_count")
            != expected_artifact_count
            or summary.get("catalog_artifact_count")
            != expected_artifact_count
            or summary.get("catalog_magnet_station_count")
            != expected_magnet_count):
        raise AuditError("staged JSON manifest lacks complete passing coverage")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise AuditError("staged JSON manifest lacks artifact records")
    for record in artifacts:
        if (not isinstance(record, Mapping)
                or record.get("audit_mode") != "actual_p2s_slice"):
            continue
        ready = record.get("slicer", {}).get("ready_project", {})
        if (not isinstance(ready, Mapping)
                or ready.get("status") != "pass"
                or not isinstance(ready.get("output_fingerprint"), str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}", ready["output_fingerprint"])
                or Path(str(ready.get("project_3mf", ""))).name
                != READY_3MF_FILENAME):
            raise AuditError(
                f"{record.get('id', '<unknown>')}: staged JSON artifact is "
                "not bound to a ready-to-print project")
    if summary.get("ready_project_count") != summary.get(
            "sliced_artifact_count"):
        raise AuditError(
            "staged JSON ready-project count differs from sliced artifacts")
    groups = manifest.get("pause_groups")
    if not isinstance(groups, list) or not groups:
        raise AuditError("staged JSON manifest contains no pause groups")
    for group in groups:
        if (not isinstance(group, Mapping)
                or group.get("print_insertion_direction_xyz")
                != list(PRINT_INSERTION_DIRECTION_XYZ)
                or group.get("insertion_instruction")
                != PRINT_INSERTION_INSTRUCTION):
            raise AuditError(
                "staged JSON manifest lacks the exact print -Z insertion "
                "contract")
        if (group.get("ready_project") is not True
                or not isinstance(
                    group.get("ready_project_output_fingerprint"), str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    group["ready_project_output_fingerprint"])
                or Path(str(group.get("audited_bambu_3mf", ""))).name
                != READY_3MF_FILENAME):
            raise AuditError(
                "staged JSON manifest pause group is not bound to a "
                "ready-to-print project")
        _require_record_file(
            group.get("audited_bambu_3mf"),
            group.get("audited_bambu_3mf_sha256"),
            f"{group.get('artifact_id', '<unknown>')} arranged Bambu 3MF")
        _float(
            group.get("bambu_arrange_rz_degrees"),
            f"{group.get('artifact_id', '<unknown>')} arrange Rz")
    with paths["csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AuditError("staged CSV manifest contains no pause rows")
    for row in rows:
        if (row.get("print_insertion_direction_xyz") != "[0.0,0.0,-1.0]"
                or row.get("insertion_instruction")
                != PRINT_INSERTION_INSTRUCTION):
            raise AuditError(
                "staged CSV manifest lacks the exact print -Z insertion "
                "contract")
        if (row.get("ready_project") != "True"
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    row.get("ready_project_output_fingerprint", ""))
                or Path(row.get("audited_bambu_3mf", "")).name
                != READY_3MF_FILENAME):
            raise AuditError(
                "staged CSV manifest row is not bound to a ready-to-print "
                "project")
        _require_record_file(
            row.get("audited_bambu_3mf"),
            row.get("audited_bambu_3mf_sha256"),
            f"{row.get('artifact_id', '<unknown>')} CSV arranged Bambu 3MF")
        _float(
            row.get("bambu_arrange_rz_degrees"),
            f"{row.get('artifact_id', '<unknown>')} CSV arrange Rz")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    if "# Captive-magnet pause manifest" not in markdown:
        raise AuditError("staged Markdown manifest has the wrong document kind")
    required_texts = [
        "print_insertion_direction_xyz = [0, 0, -1]",
        "## Audited Bambu arrangements",
        READY_3MF_FILENAME,
    ]
    if enforce_release_polarity_contract:
        required_texts.append("unpaired coupon1 regression station")
    for required_text in required_texts:
        if required_text not in markdown:
            raise AuditError(
                "staged Markdown manifest omits required insertion/polarity "
                f"instruction: {required_text}")


__all__ = tuple(
    name for name in globals()
    if name != "__all__" and not name.startswith("__"))
