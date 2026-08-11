#!/usr/bin/env python3
"""Build and audit local-only Obi-Wan 01+02+03+04 P2S plates.

This is packaging, not CAD generation.  Four already-released front-down
STLs are translated into one deterministic multi-shell STL, then loaded into
Bambu Studio as one printable assembly with four normal volumes and the three
released duct support blockers.  The fixed disposition is intentionally not
auto-arranged or rotated.  Both floor-stand states are explicit, hash-bound
variants so a plate can never silently consume geometry or blockers from the
other state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _canonical_import_root in (
        PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)

import artifact_emit as emit
from bambu_3mf_audit import (
    Bambu3MFAuditError,
    mesh_bounds,
    read_stl_triangles,
    validate_bed_fit,
    validate_result_bbox,
)
import composite_bambu_3mf_audit as composite_audit
from composite_bambu_3mf_audit import audit_bambu_composite_3mf
from check_manifold import stl_diagnostics
from gcode_analysis import (
    audit_support_toolpaths_vs_ducts,
    parse_gcode,
)
from lx521_baffle.io import pretty_json_bytes, sha256_bytes, sha256_file
from lx521_baffle.print_contract import (
    FrontDownContractError,
    validate_print_sidecar,
)
import release_validation as captive


ROOT = PROJECT_ROOT
PLATE_OUTPUT_DIR = ROOT / "build" / "print_plates" / "obiwan"
DEFAULT_PROFILE = ROOT / "captive_magnet_slicing_profile_petg_gf_06hf.json"
DEFAULT_RELEASE_CATALOG = ROOT / "review" / "captive_magnet_release_catalog.json"
DEFAULT_RELEASE_AUDIT = ROOT / "review" / "captive_magnet_slice_audit"
PAUSE_Z_MM = 5.96
MINIMUM_PART_GAP_MM = 2.0
MACHINE_BOUNDS_MM = {
    "x": (0.0, 256.0),
    "y": (0.0, 256.0),
    "z": (0.0, 256.0),
}
SUPPORT_KEYS = (
    "enable_support",
    "support_on_build_plate_only",
    "support_critical_regions_only",
    "support_remove_small_overhang",
)
STRICT_MESH_KEYS = (
    "open",
    "winding",
    "over_shared",
    "degenerate",
    "duplicates",
    "nonfinite",
    "zero_volume",
    "negative_volume",
    "component_error",
)


class ComboPlateError(RuntimeError):
    """The locked composite-plate contract did not pass."""


@dataclass(frozen=True)
class PlatePart:
    friendly_name: str
    source_stl: Path
    translation_mm: tuple[float, float, float]
    artifact_id: str | None = None
    support_blocker: Path | None = None

    @property
    def staged_name(self) -> str:
        return f"{self.friendly_name}.stl"


@dataclass(frozen=True)
class ComboPlateVariant:
    state: str
    label: str
    plate_name: str
    expected_triangle_count: int
    sparse_infill_density_percent: float
    sparse_infill_pattern: str
    parts: tuple[PlatePart, ...]


# The 04 crescent moved +0.124/+0.273 mm along the exact 02-to-04
# nearest-point direction: the land-derived R113.94 keyed-top fairing ate
# 0.113 mm of the former 2.0-mm plate gap (measured 1.887), and this nudge
# restores ~2.19 mm while keeping the crescent 5 mm inside the 256 bed.
LOCKED_TRANSLATIONS_MM = (
    (2.697, 34.319, 0.0),
    (27.025, 2.010, 0.0),
    (71.034, 28.412, 0.0),
    (108.802, 209.132, 0.0),
)


def _variant(
    *,
    state: str,
    label: str,
    plate_name: str,
    expected_triangle_count: int,
    sparse_infill_density_percent: float,
    sparse_infill_pattern: str,
) -> ComboPlateVariant:
    bottom_suffix = (
        "no_floor_stand" if state == "no_floor_stand" else "floor_stand"
    )
    identities = (
        (
            f"obiwan_01_LM_bottom_keyed_1_of_2_{bottom_suffix}",
            "obiwan_optional_lm_keyed_1_of_2_bottom",
            f"{state}:Obi-Wan-split:"
            "obiwan_optional_lm_keyed_1_of_2_bottom",
        ),
        (
            "obiwan_02_LM_top_keyed_2_of_2",
            "obiwan_optional_lm_keyed_2_of_2_top",
            f"{state}:Obi-Wan-split:"
            "obiwan_optional_lm_keyed_2_of_2_top",
        ),
        (
            "obiwan_03_UM_carrier_1_of_1",
            "obiwan_core_2_of_2_um_carrier",
            f"{state}:Obi-Wan:obiwan_core_2_of_2_um_carrier",
        ),
        (
            "obiwan_04_T_tweeter_crescent_1_of_1",
            "obiwan_addon_tweeter_crescent",
            None,
        ),
    )
    parts = []
    for identity, translation in zip(
            identities, LOCKED_TRANSLATIONS_MM, strict=True):
        friendly_name, stem, artifact_id = identity
        support_blocker = (
            ROOT / "build" / state / "support_blockers"
            / f"{stem}.support_blocker.stl"
            if artifact_id is not None else None
        )
        parts.append(PlatePart(
            friendly_name=friendly_name,
            source_stl=ROOT / "build" / state / "stl" / f"{stem}.stl",
            translation_mm=translation,
            artifact_id=artifact_id,
            support_blocker=support_blocker,
        ))
    return ComboPlateVariant(
        state=state,
        label=label,
        plate_name=plate_name,
        expected_triangle_count=expected_triangle_count,
        sparse_infill_density_percent=sparse_infill_density_percent,
        sparse_infill_pattern=sparse_infill_pattern,
        parts=tuple(parts),
    )


VARIANTS = {
    "no_floor_stand": _variant(
        state="no_floor_stand",
        label="no-floor",
        plate_name=(
            "obiwan_01_02_03_04_LM_UM_combo_no_floor_stand"
        ),
        expected_triangle_count=63_004,
        sparse_infill_density_percent=40.0,
        sparse_infill_pattern="gyroid",
    ),
    "floor_stand": _variant(
        state="floor_stand",
        label="floor-stand",
        plate_name=(
            "obiwan_01_02_03_04_LM_UM_combo_floor_stand"
        ),
        expected_triangle_count=165_848,
        sparse_infill_density_percent=100.0,
        sparse_infill_pattern="zig-zag",
    ),
}


def _activate_variant(variant: ComboPlateVariant) -> None:
    global ACTIVE_VARIANT
    global PLATE_NAME, PLATE_STL, PLATE_MANIFEST
    global DEFAULT_WORKSPACE, EXPECTED_TRIANGLE_COUNT, PARTS
    ACTIVE_VARIANT = variant
    PLATE_NAME = variant.plate_name
    PLATE_STL = PLATE_OUTPUT_DIR / f"{PLATE_NAME}.stl"
    PLATE_MANIFEST = PLATE_OUTPUT_DIR / f"{PLATE_NAME}.plate.json"
    DEFAULT_WORKSPACE = (
        ROOT / "review" / "to_print_slice_workspace"
        / "composite" / PLATE_NAME
    )
    EXPECTED_TRIANGLE_COUNT = variant.expected_triangle_count
    PARTS = variant.parts


@dataclass(frozen=True)
class ComboPlateAPI:
    """Stand-state-bound API consumed by the shelf and focused tests."""

    variant: ComboPlateVariant

    @property
    def PLATE_NAME(self) -> str:
        return self.variant.plate_name

    @property
    def PARTS(self) -> tuple[PlatePart, ...]:
        return self.variant.parts

    @property
    def EXPECTED_TRIANGLE_COUNT(self) -> int:
        return self.variant.expected_triangle_count

    @property
    def SPARSE_INFILL_DENSITY_PERCENT(self) -> float:
        return self.variant.sparse_infill_density_percent

    @property
    def SPARSE_INFILL_PATTERN(self) -> str:
        return self.variant.sparse_infill_pattern

    @property
    def PLATE_STL(self) -> Path:
        return PLATE_OUTPUT_DIR / f"{self.PLATE_NAME}.stl"

    @property
    def PLATE_MANIFEST(self) -> Path:
        return PLATE_OUTPUT_DIR / f"{self.PLATE_NAME}.plate.json"

    @property
    def DEFAULT_WORKSPACE(self) -> Path:
        return (
            ROOT / "review" / "to_print_slice_workspace"
            / "composite" / self.PLATE_NAME
        )

    def activate(self) -> None:
        _activate_variant(self.variant)

    def validate_source_bundle(
        self,
        stl: Path | None = None,
        manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        self.activate()
        return validate_source_bundle(
            stl or self.PLATE_STL,
            manifest_path or self.PLATE_MANIFEST,
        )

    def build_or_validate_ready_plate(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.activate()
        kwargs.setdefault("workspace", self.DEFAULT_WORKSPACE)
        return build_or_validate_ready_plate(**kwargs)


def get_variant(state: str) -> ComboPlateAPI:
    try:
        return ComboPlateAPI(VARIANTS[state.lower()])
    except KeyError as exc:
        raise ComboPlateError(
            f"unknown composite plate stand state {state!r}") from exc


_activate_variant(VARIANTS["no_floor_stand"])


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComboPlateError(f"cannot read {label} {path}: {exc}") from exc


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pretty_json_bytes(data, allow_nan=False))
    temporary.replace(path)


def _canonical_json(data: Any) -> bytes:
    return (json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode("utf-8")


def _binary_stl_records(path: Path) -> tuple[bytes, ...]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ComboPlateError(f"{path} is too short to be a binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if count <= 0 or len(data) != 84 + 50 * count:
        raise ComboPlateError(
            f"{path} must be a non-empty, exact-length binary STL")
    return tuple(
        data[84 + 50 * index:84 + 50 * (index + 1)]
        for index in range(count)
    )


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _translate_stl_record(
    record: bytes,
    translation: Sequence[float],
) -> bytes:
    if len(record) != 50 or len(translation) != 3:
        raise ComboPlateError("invalid STL record or translation")
    offsets = tuple(_float32(float(value)) for value in translation)
    values = list(struct.unpack_from("<9f", record, 12))
    for vertex in range(3):
        for axis in range(3):
            index = vertex * 3 + axis
            values[index] = _float32(
                _float32(values[index]) + offsets[axis])
    result = bytearray(record)
    struct.pack_into("<9f", result, 12, *values)
    return bytes(result)


def _strict_source_mesh(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ComboPlateError(f"missing released source STL: {path}")
    facts = stl_diagnostics(path)
    failures = {
        key: facts.get(key)
        for key in STRICT_MESH_KEYS
        if int(facts.get(key, 0)) != 0
    }
    if failures:
        raise ComboPlateError(
            f"{path.name}: released source STL failed strict topology: "
            f"{failures}")
    return facts


def _part_footprint(part: PlatePart):
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ComboPlateError(
            "Shapely is required for exact composite-plate gap validation"
        ) from exc
    dx, dy, _dz = part.translation_mm
    polygons = []
    for triangle in read_stl_triangles(part.source_stl):
        polygon = Polygon(tuple(
            (point[0] + dx, point[1] + dy) for point in triangle))
        if polygon.area > 1.0e-10:
            polygons.append(polygon)
    if not polygons:
        raise ComboPlateError(f"{part.source_stl} has no projected footprint")
    footprint = unary_union(polygons)
    if footprint.is_empty or not footprint.is_valid:
        raise ComboPlateError(
            f"{part.source_stl.name}: projected footprint is invalid")
    return footprint


def _packing_facts() -> tuple[dict[str, Any], dict[str, Any]]:
    footprints = {
        part.friendly_name: _part_footprint(part) for part in PARTS
    }
    pairwise = []
    minimum_gap = math.inf
    for index, left in enumerate(PARTS):
        for right in PARTS[index + 1:]:
            left_shape = footprints[left.friendly_name]
            right_shape = footprints[right.friendly_name]
            distance = float(left_shape.distance(right_shape))
            intersects = bool(left_shape.intersects(right_shape))
            if intersects or distance < MINIMUM_PART_GAP_MM:
                raise ComboPlateError(
                    f"locked placement collides or has insufficient gap: "
                    f"{left.friendly_name} vs {right.friendly_name}, "
                    f"distance={distance:.6f} mm")
            minimum_gap = min(minimum_gap, distance)
            pairwise.append({
                "left": left.friendly_name,
                "right": right.friendly_name,
                "xy_gap_mm": distance,
                "intersects": intersects,
            })
    return ({
        "minimum_required_xy_gap_mm": MINIMUM_PART_GAP_MM,
        "minimum_actual_xy_gap_mm": minimum_gap,
        "pairwise": pairwise,
    }, footprints)


def _combined_expected_triangles():
    return tuple(
        triangle
        for part in PARTS
        for triangle in composite_audit.translated_float32_triangles(
            read_stl_triangles(part.source_stl), part.translation_mm)
    )


def build_source_bundle(
    *,
    output_stl: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Write the deterministic translated STL union and its scoped contract."""
    output_stl = output_stl or PLATE_STL
    manifest_path = manifest_path or PLATE_MANIFEST
    part_records = []
    translated_records: list[bytes] = []
    for part in PARTS:
        sidecar = part.source_stl.with_suffix(".print.json")
        try:
            validate_print_sidecar(part.source_stl, sidecar)
        except FrontDownContractError as exc:
            raise ComboPlateError(
                f"{part.friendly_name}: invalid source print authority: {exc}"
            ) from exc
        mesh = _strict_source_mesh(part.source_stl)
        source_records = _binary_stl_records(part.source_stl)
        translated_records.extend(
            _translate_stl_record(record, part.translation_mm)
            for record in source_records
        )
        blocker_record = None
        if part.support_blocker is not None:
            if not part.support_blocker.is_file():
                raise ComboPlateError(
                    f"missing released support blocker: "
                    f"{part.support_blocker}")
            blocker_record = {
                "path": _relative(part.support_blocker),
                "sha256": sha256_file(part.support_blocker),
                "triangle_count": len(_binary_stl_records(
                    part.support_blocker)),
            }
        part_records.append({
            "friendly_name": part.friendly_name,
            "source_stl": _relative(part.source_stl),
            "source_stl_sha256": sha256_file(part.source_stl),
            "source_print_sidecar": _relative(sidecar),
            "source_print_sidecar_sha256": sha256_file(sidecar),
            "catalog_artifact_id": part.artifact_id,
            "translation_mm": list(part.translation_mm),
            "triangle_count": len(source_records),
            "mesh_diagnostics": mesh,
            "support_blocker": blocker_record,
        })
    if len(translated_records) != EXPECTED_TRIANGLE_COUNT:
        raise ComboPlateError(
            f"composite triangle count {len(translated_records)} != "
            f"{EXPECTED_TRIANGLE_COUNT}")

    output_stl.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"LX521 Obi-Wan {ACTIVE_VARIANT.label} 01+02+03+04 locked P2S plate"
        .encode("ascii")
    ).ljust(80, b"\0")
    payload = (
        header
        + struct.pack("<I", len(translated_records))
        + b"".join(translated_records)
    )
    temporary = output_stl.with_suffix(output_stl.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output_stl)

    expected = _combined_expected_triangles()
    actual = read_stl_triangles(output_stl)
    try:
        exact_error = composite_audit.validate_triangle_soup_equivalence(
            expected, actual, tolerance_mm=0.0)
    except Bambu3MFAuditError as exc:
        raise ComboPlateError(
            f"generated composite STL is not the exact translated union: {exc}"
        ) from exc
    bounds = mesh_bounds(actual)
    clearances = validate_bed_fit(bounds, MACHINE_BOUNDS_MM)
    packing, _footprints = _packing_facts()
    manifest = {
        "schema_version": 1,
        "manifest_kind": "lx521_locked_composite_print_plate",
        "name": PLATE_NAME,
        "stand_state": ACTIVE_VARIANT.state,
        "source_policy": (
            f"translation-only concatenation of four released "
            f"{ACTIVE_VARIANT.state} front-down STLs; no CAD or structure "
            "regeneration"),
        "stl": output_stl.name,
        "stl_sha256": sha256_file(output_stl),
        "stl_bytes": output_stl.stat().st_size,
        "triangle_count": len(actual),
        "expected_disconnected_printable_part_count": len(PARTS),
        "composite_topology_policy": (
            "each source STL passes the ordinary strict manifold gate; the "
            "bundle is accepted only as their exact, non-intersecting "
            "translated triangle concatenation"),
        "mesh_max_abs_error_mm": exact_error,
        "bounds_mm": bounds.as_dict(),
        "bed_clearances_mm": {
            axis: list(values) for axis, values in clearances.items()
        },
        "packing": packing,
        "parts": part_records,
        "print_profile": {
            "sparse_infill_density_percent": (
                ACTIVE_VARIANT.sparse_infill_density_percent
            ),
            "sparse_infill_pattern": (
                ACTIVE_VARIANT.sparse_infill_pattern
            ),
        },
        "support_policy": {
            "enabled": True,
            "global_and_object_fields": {
                key: "1" for key in SUPPORT_KEYS
            },
            "duct_blocker_count": 3,
            "support_toolpath_vs_duct_collision_gate": "mandatory",
        },
        "magnet_pause": {
            "pause_z_mm": PAUSE_Z_MM,
            "magnet_count": 6,
            "sites": [
                "lm_lower_left",
                "lm_lower_right",
                "lm_upper_left",
                "lm_upper_right",
                "um_left",
                "um_right",
            ],
        },
    }
    manifest["source_bundle_fingerprint"] = sha256_bytes(_canonical_json({
        "name": manifest["name"],
        "stand_state": manifest["stand_state"],
        "stl_sha256": manifest["stl_sha256"],
        "parts": manifest["parts"],
        "packing": manifest["packing"],
        "print_profile": manifest["print_profile"],
        "support_policy": manifest["support_policy"],
        "magnet_pause": manifest["magnet_pause"],
    }))
    _write_json(manifest_path, manifest)
    validate_source_bundle(output_stl, manifest_path)
    return manifest


def validate_source_bundle(
    stl: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the intentional multi-body STL without relaxing global policy."""
    stl = stl or PLATE_STL
    manifest_path = manifest_path or PLATE_MANIFEST
    payload = _read_json(manifest_path, "composite plate manifest")
    if (not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or payload.get("manifest_kind")
            != "lx521_locked_composite_print_plate"
            or payload.get("name") != PLATE_NAME
            or payload.get("stand_state") != ACTIVE_VARIANT.state):
        raise ComboPlateError("composite plate manifest identity is invalid")
    if (not stl.is_file()
            or payload.get("stl") != stl.name
            or payload.get("stl_sha256") != sha256_file(stl)
            or payload.get("stl_bytes") != stl.stat().st_size):
        raise ComboPlateError(
            "composite plate STL does not match its manifest")
    if payload.get("print_profile") != {
        "sparse_infill_density_percent": (
            ACTIVE_VARIANT.sparse_infill_density_percent
        ),
        "sparse_infill_pattern": ACTIVE_VARIANT.sparse_infill_pattern,
    }:
        raise ComboPlateError(
            "composite plate infill contract crossed stand states")
    records = payload.get("parts")
    if not isinstance(records, list) or len(records) != len(PARTS):
        raise ComboPlateError("composite plate part inventory is incomplete")
    for part, record in zip(PARTS, records, strict=True):
        if not isinstance(record, Mapping):
            raise ComboPlateError("composite plate part record is invalid")
        expected = {
            "friendly_name": part.friendly_name,
            "source_stl": _relative(part.source_stl),
            "source_stl_sha256": sha256_file(part.source_stl),
            "source_print_sidecar": _relative(
                part.source_stl.with_suffix(".print.json")),
            "source_print_sidecar_sha256": sha256_file(
                part.source_stl.with_suffix(".print.json")),
            "catalog_artifact_id": part.artifact_id,
            "translation_mm": list(part.translation_mm),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ComboPlateError(
                    f"{part.friendly_name}: composite source contract drifted "
                    f"at {key}")
        try:
            validate_print_sidecar(part.source_stl)
        except FrontDownContractError as exc:
            raise ComboPlateError(
                f"{part.friendly_name}: source sidecar failed: {exc}") from exc
        _strict_source_mesh(part.source_stl)
        blocker = record.get("support_blocker")
        if part.support_blocker is None:
            if blocker is not None:
                raise ComboPlateError(
                    f"{part.friendly_name}: unexpected support blocker")
        elif (not isinstance(blocker, Mapping)
              or blocker.get("path") != _relative(part.support_blocker)
              or blocker.get("sha256") != sha256_file(
                  part.support_blocker)):
            raise ComboPlateError(
                f"{part.friendly_name}: support blocker binding drifted")

    actual = read_stl_triangles(stl)
    if (len(actual) != EXPECTED_TRIANGLE_COUNT
            or payload.get("triangle_count") != EXPECTED_TRIANGLE_COUNT):
        raise ComboPlateError("composite plate triangle count drifted")
    try:
        composite_audit.validate_triangle_soup_equivalence(
            _combined_expected_triangles(), actual, tolerance_mm=0.0)
    except Bambu3MFAuditError as exc:
        raise ComboPlateError(
            f"composite STL is not the exact translated source union: {exc}"
        ) from exc
    bounds = mesh_bounds(actual)
    validate_bed_fit(bounds, MACHINE_BOUNDS_MM)
    packing, _footprints = _packing_facts()
    if not math.isclose(
            float(payload.get("packing", {}).get(
                "minimum_actual_xy_gap_mm", -1.0)),
            packing["minimum_actual_xy_gap_mm"],
            abs_tol=1.0e-9, rel_tol=0.0):
        raise ComboPlateError("composite plate packing evidence drifted")
    return dict(payload)


def _local_slice_guard() -> None:
    if platform.system() != "Darwin":
        raise ComboPlateError(
            "the Obi-Wan composite plate may be sliced only on the local Mac")
    hostname = socket.gethostname().split(".", 1)[0].lower()
    if hostname == "osado" or hostname.startswith("osado-"):
        raise ComboPlateError("refusing to slice the composite plate on osado")
    execution = os.environ.get("LX_CAD_EXECUTION", "").strip().lower()
    if execution in {"remote", "remote-worker"}:
        raise ComboPlateError(
            f"refusing local slicing under LX_CAD_EXECUTION={execution}")


def _materialize(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ComboPlateError(f"cannot stage missing input {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == sha256_file(source):
        return
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _normalized_artifacts(
    release_catalog: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        catalog = captive.normalize_catalog(release_catalog)
    except captive.AuditError as exc:
        raise ComboPlateError(
            f"cannot normalize captive-magnet release catalog: {exc}") from exc
    by_id = {artifact["id"]: artifact for artifact in catalog["artifacts"]}
    selected = {}
    for part in PARTS:
        if part.artifact_id is None:
            continue
        artifact = by_id.get(part.artifact_id)
        if artifact is None:
            raise ComboPlateError(
                f"missing release artifact {part.artifact_id}")
        if Path(artifact["stl"]).resolve() != part.source_stl.resolve():
            raise ComboPlateError(
                f"{part.friendly_name}: release artifact source drifted")
        if Path(artifact.get("support_blocker", "")).resolve() != (
                part.support_blocker.resolve()):
            raise ComboPlateError(
                f"{part.friendly_name}: release support blocker drifted")
        if not isinstance(artifact.get("duct_collision_contract"), Mapping):
            raise ComboPlateError(
                f"{part.friendly_name}: release duct contract is missing")
        selected[part.artifact_id] = dict(artifact)
    return catalog, selected


def _write_assemble_list(
    path: Path,
    *,
    staged_parts: Mapping[str, Path],
    staged_blockers: Mapping[str, Path],
    staged_modifiers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    objects = []
    for part in PARTS:
        def record(
            mesh: Path,
            subtype: str,
            print_params: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            dx, dy, dz = part.translation_mm
            payload = {
                "path": str(mesh.resolve()),
                "subtype": subtype,
                "count": 1,
                "filaments": [1],
                "assemble_index": [1],
                "pos_x": [dx],
                "pos_y": [dy],
                "pos_z": [dz],
            }
            if print_params:
                payload["print_params"] = dict(print_params)
            return payload
        objects.append(record(
            staged_parts[part.friendly_name], "normal_part"))
        if part.support_blocker is not None:
            objects.append(record(
                staged_blockers[part.friendly_name], "support_blocker"))
        for modifier in staged_modifiers.get(part.friendly_name, ()):
            objects.append(record(
                Path(modifier["path"]), "modifier_part",
                modifier["process"]))
    _write_json(path, {
        "plates": [{
            "plate_name": PLATE_NAME,
            "need_arrange": False,
            "objects": objects,
            "assembled_params": [{
                "assemble_index": 1,
                "print_params": {
                    **{key: "1" for key in SUPPORT_KEYS},
                    "sparse_infill_density": (
                        f"{ACTIVE_VARIANT.sparse_infill_density_percent:g}%"
                    ),
                    "sparse_infill_pattern": (
                        ACTIVE_VARIANT.sparse_infill_pattern
                    ),
                },
            }],
        }],
    })


def _write_custom_gcodes(
    path: Path,
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    profile_bundle: Mapping[str, Any],
) -> None:
    site_names = []
    pause_values = set()
    for part in PARTS:
        if part.artifact_id is None:
            continue
        for site in artifacts[part.artifact_id]["sites"]:
            site_names.append(str(site["name"]))
            pause_values.add(float(site["expected_pause_marker_z_mm"]))
    expected_sites = [
        "lm_lower_left",
        "lm_lower_right",
        "lm_upper_left",
        "lm_upper_right",
        "um_left",
        "um_right",
    ]
    if site_names != expected_sites or pause_values != {PAUSE_Z_MM}:
        raise ComboPlateError(
            "composite magnet site or pause inventory drifted")
    pause_policy = profile_bundle["identity"]["effective"].get(
        "magnet_insertion_pause")
    if not isinstance(pause_policy, Mapping):
        raise ComboPlateError(
            "resolved profile lacks the magnet insertion pause policy")
    group = {
        "pause_marker_z_mm": PAUSE_Z_MM,
        "sites": site_names,
        "magnet_count": len(site_names),
    }
    _write_json(path, {
        "mode": "SingleExtruder",
        "gcodes": [{
            "type": emit.MAGNET_INSERTION_CUSTOM_GCODE_TYPE,
            "print_z": PAUSE_Z_MM,
            "color": "",
            "extruder": 1,
            "extra": emit._magnet_pause_program(group, pause_policy),
        }],
    })


def _translation_matrix(
    translation: Sequence[float],
) -> tuple[tuple[float, float, float, float], ...]:
    dx, dy, dz = (float(value) for value in translation)
    return (
        (1.0, 0.0, 0.0, dx),
        (0.0, 1.0, 0.0, dy),
        (0.0, 0.0, 1.0, dz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrix_multiply(left, right):
    return tuple(tuple(
        sum(float(left[row][index]) * float(right[index][column])
            for index in range(4))
        for column in range(4)
    ) for row in range(4))


def _support_coverage(
    gcode: Path,
    footprints: Mapping[str, Any],
) -> tuple[dict[str, int], Any]:
    try:
        from shapely.geometry import Point
        from shapely.prepared import prep
    except ImportError as exc:
        raise ComboPlateError(
            "Shapely is required for support-footprint validation") from exc
    prepared = {name: prep(shape) for name, shape in footprints.items()}
    counts = {part.friendly_name: 0 for part in PARTS}
    parsed = parse_gcode(
        gcode, retain_regions=None,
        retain_feature_prefixes=("support",))
    for layer in parsed.layers:
        for segment in layer.segments:
            if not segment.feature.lower().startswith("support"):
                continue
            midpoint = Point(
                (segment.x0 + segment.x1) / 2.0,
                (segment.y0 + segment.y1) / 2.0,
            )
            for name, shape in prepared.items():
                if shape.covers(midpoint):
                    counts[name] += 1
    for part in PARTS[:3]:
        if counts[part.friendly_name] <= 0:
            raise ComboPlateError(
                f"{part.friendly_name}: no support toolpath reaches its "
                "footprint; floating cantilever risk")
    tweeter = PARTS[3].friendly_name
    if counts[tweeter] != 0:
        raise ComboPlateError(
            f"{tweeter}: unexpected support toolpath under the tweeter")
    return counts, parsed


def _result_object(result_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _read_json(result_path, "Bambu result")
    if not isinstance(result, Mapping) or result.get("return_code") != 0:
        raise ComboPlateError("Bambu result is not Success")
    plates = result.get("sliced_plates")
    if not isinstance(plates, list) or len(plates) != 1:
        raise ComboPlateError("composite slice must contain exactly one plate")
    plate = plates[0]
    objects = plate.get("objects") if isinstance(plate, Mapping) else None
    if not isinstance(objects, list) or len(objects) != 1:
        raise ComboPlateError(
            "composite slice must contain exactly one printable object")
    obj = objects[0]
    if (int(plate.get("triangle_count", -1)) != EXPECTED_TRIANGLE_COUNT
            or int(obj.get("triangle_count", -1))
            != EXPECTED_TRIANGLE_COUNT):
        raise ComboPlateError(
            "Bambu triangle count differs from the composite STL")
    if plate.get("warning_message") not in ("", None):
        raise ComboPlateError(
            f"Bambu plate warning: {plate.get('warning_message')}")
    return dict(plate), dict(obj)


def _discovery_record(release_audit: Path, artifact_id: str) -> dict[str, Any]:
    path = (
        release_audit / "slices" / captive._slug(artifact_id)
        / "captive_magnet_slice_audit.json")
    record = _read_json(path, f"{artifact_id} authoritative slice audit")
    if (not isinstance(record, Mapping)
            or record.get("id") != artifact_id
            or record.get("status") != "pass"):
        raise ComboPlateError(
            f"{artifact_id}: authoritative discovery audit is not passing")
    return dict(record)


def validate_ready_plate(
    *,
    workspace: Path,
    profile_bundle: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    staged_parts: Mapping[str, Path],
    staged_blockers: Mapping[str, Path],
    staged_modifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    release_audit: Path,
) -> dict[str, Any]:
    """Run every promotion gate against the final pause-bearing project."""
    ready = workspace / "ready"
    project = ready / f"{PLATE_NAME}.gcode.3mf"
    gcode = ready / "plate_1.gcode"
    result_path = ready / "result.json"
    for path in (project, gcode, result_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ComboPlateError(f"missing composite slice output: {path}")
    source_manifest = validate_source_bundle(PLATE_STL, PLATE_MANIFEST)
    try:
        project_audit = audit_bambu_composite_3mf(
            project,
            PLATE_STL,
            normal_part_stls=[
                (staged_parts[part.friendly_name], part.translation_mm)
                for part in PARTS
            ],
            support_blocker_stls=[
                (staged_blockers[part.friendly_name], part.translation_mm)
                for part in PARTS if part.support_blocker is not None
            ],
            parameter_modifier_stls=[
                (
                    Path(modifier["path"]),
                    part.translation_mm,
                    modifier["process"],
                )
                for part in PARTS
                for modifier in staged_modifiers.get(
                    part.friendly_name, ())
            ],
        )
    except Bambu3MFAuditError as exc:
        raise ComboPlateError(
            f"composite project/STL equivalence failed: {exc}") from exc
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    matrix_error = max(
        abs(project_audit.stl_to_bed_matrix[row][column]
            - identity[row][column])
        for row in range(4) for column in range(4)
    )
    if matrix_error > 2.0e-6:
        raise ComboPlateError(
            "composite build transform moved or rotated the locked plate")
    clearances = validate_bed_fit(
        project_audit.transformed_actual_mesh_bounds,
        profile_bundle["identity"]["machine_bounds_mm"])
    plate, obj = _result_object(result_path)
    bbox = obj.get("bbox")
    if not isinstance(bbox, Mapping):
        raise ComboPlateError("Bambu result lacks the composite object bbox")
    try:
        validate_result_bbox(
            bbox, project_audit.source_bounds,
            project_audit.stl_to_bed_matrix)
    except Bambu3MFAuditError as exc:
        raise ComboPlateError(
            f"Bambu result/3MF placement mismatch: {exc}") from exc

    try:
        archive = emit._validate_ready_project_archive(
            project, gcode, expected_pause_z=[PAUSE_Z_MM],
            profile_bundle=profile_bundle)
    except captive.AuditError as exc:
        raise ComboPlateError(
            f"ready-project archive audit failed: {exc}") from exc
    if len(archive.get("object_support_overrides", ())) != 1:
        raise ComboPlateError(
            "composite project must contain one support-pinned object")

    cavity_records = {}
    parsed_for_pause = None
    for part in PARTS:
        if part.artifact_id is None:
            continue
        artifact = artifacts[part.artifact_id]
        try:
            parsed, cavity = emit._validate_ready_cavity_toolpaths(
                artifact=artifact,
                discovery_record=_discovery_record(
                    release_audit, part.artifact_id),
                gcode=gcode,
                stl_to_bed_matrix=_translation_matrix(
                    part.translation_mm),
            )
        except captive.AuditError as exc:
            raise ComboPlateError(
                f"{part.friendly_name}: captive-cavity audit failed: {exc}"
            ) from exc
        parsed_for_pause = parsed_for_pause or parsed
        cavity_records[part.friendly_name] = cavity
    if parsed_for_pause is None:
        raise ComboPlateError("composite plate has no captive-cavity audit")
    try:
        pause_before_extrusion = emit._assert_pauses_precede_layer_extrusion(
            parsed_for_pause, archive["gcode_pause_events"])
    except captive.AuditError as exc:
        raise ComboPlateError(
            f"magnet pause ordering failed: {exc}") from exc

    _packing, footprints = _packing_facts()
    support_coverage, support_parsed = _support_coverage(
        gcode, footprints)
    support_summary = emit._support_toolpath_summary(gcode)
    if support_summary["support_feature_blocks"] <= 0:
        raise ComboPlateError(
            "support is enabled but no support feature blocks were emitted")
    profile_errors = emit._validate_actual_gcode_profile(
        support_parsed, profile_bundle)
    if profile_errors:
        raise ComboPlateError(
            "composite G-code profile mismatch: "
            + "; ".join(profile_errors))

    duct_records = {}
    outer = project_audit.stl_to_bed_matrix
    for part in PARTS:
        if part.artifact_id is None:
            continue
        artifact = artifacts[part.artifact_id]
        try:
            duct = audit_support_toolpaths_vs_ducts(
                gcode=gcode,
                contract=artifact["duct_collision_contract"],
                source_to_stl_matrix=artifact["source_to_stl_matrix"],
                stl_to_bed_matrix=_matrix_multiply(
                    outer, _translation_matrix(part.translation_mm)),
            )
        except captive.AuditError as exc:
            raise ComboPlateError(
                f"{part.friendly_name}: support-vs-duct gate failed: {exc}"
            ) from exc
        if duct.get("status") != "pass" or duct.get("collision_count") != 0:
            raise ComboPlateError(
                f"{part.friendly_name}: support enters a functional duct")
        duct_records[part.friendly_name] = duct

    static_validation = emit._validate_with_gcode_skill(
        gcode, ready, profile_bundle)
    if static_validation.get("ok") is not True:
        raise ComboPlateError(
            "composite G-code static validation did not pass")
    effective = profile_bundle["identity"]["effective"]
    if (effective.get("sparse_infill_density_percent")
            != ACTIVE_VARIANT.sparse_infill_density_percent
            or effective.get("sparse_infill_pattern")
            != ACTIVE_VARIANT.sparse_infill_pattern
            or effective.get("support_enabled") is not True):
        raise ComboPlateError(
            f"{ACTIVE_VARIANT.state} composite effective profile is not "
            f"{ACTIVE_VARIANT.sparse_infill_density_percent:g}% "
            f"{ACTIVE_VARIANT.sparse_infill_pattern} with support")
    modifier_count = sum(
        len(modifiers) for modifiers in staged_modifiers.values())
    petg_gf_core = (
        profile_bundle["config"].get("user_filament_preset")
        == "TINMORRY PETG-GF Profile @BBL P2S"
    )
    if petg_gf_core:
        expected_modifier_count = (
            1 if ACTIVE_VARIANT.state == "no_floor_stand" else 0)
        # TINMORRY PETG-GF is exclusive to the 0.6-mm high-flow lane: six
        # 0.62-mm walls give the same ~3.7-mm structural shell the former
        # 0.4-mm lane built from eight 0.45-mm walls.
        if (modifier_count != expected_modifier_count
                or project_audit.parameter_modifier_count
                != expected_modifier_count
                or effective.get("wall_loops") != 6
                or effective.get("nozzle_diameter_mm") != 0.6
                or effective.get("filament")
                != "TINMORRY PETG-GF Profile @BBL P2S"):
            raise ComboPlateError(
                "structural core plate must use six 0.62-mm walls on the "
                "0.6-mm high-flow lane and the saved TINMORRY PETG-GF "
                "profile; no-floor additionally requires one audited "
                "100%-solid bridge/root modifier")
    record = {
        "schema_version": 1,
        "audit_kind": "lx521_locked_composite_print_plate",
        "name": PLATE_NAME,
        "stand_state": ACTIVE_VARIANT.state,
        "status": "pass",
        "local_only": True,
        "source_manifest": _relative(PLATE_MANIFEST),
        "source_manifest_sha256": sha256_file(PLATE_MANIFEST),
        "source_bundle_fingerprint": source_manifest[
            "source_bundle_fingerprint"],
        "project_3mf": _relative(project),
        "project_3mf_sha256": sha256_file(project),
        "gcode": _relative(gcode),
        "gcode_sha256": sha256_file(gcode),
        "result_json": _relative(result_path),
        "result_sha256": sha256_file(result_path),
        "project_stl_equivalence": project_audit.as_record(),
        "bed_clearances_mm": {
            axis: list(values) for axis, values in clearances.items()
        },
        "result": {
            "triangle_count": int(plate["triangle_count"]),
            "bbox": dict(bbox),
            "estimated_print_time_seconds": plate.get(
                "total_predication"),
            "filaments": plate.get("filaments"),
        },
        "archive_audit": archive,
        "captive_cavity_audit": cavity_records,
        "pause_before_first_layer_extrusion": pause_before_extrusion,
        "support_toolpaths": support_summary,
        "support_midpoints_inside_part_footprints": support_coverage,
        "duct_support_toolpath_audit": {
            "status": "pass",
            "collision_count": 0,
            "parts": duct_records,
        },
        "gcode_static_validation": static_validation,
        "profile_effective": dict(effective),
        "parameter_modifiers": [{
            "part": part.friendly_name,
            "contract": _relative(Path(modifier["contract"])),
            "contract_sha256": modifier["contract_sha256"],
            "modifier_stl": _relative(Path(modifier["path"])),
            "modifier_stl_sha256": modifier["sha256"],
            "process": dict(modifier["process"]),
        } for part in PARTS for modifier in staged_modifiers.get(
            part.friendly_name, ())],
    }
    _write_json(ready / "plate_audit.json", record)
    preview = ready / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(project) as archive_file:
            for member in ("Metadata/top_1.png", "Metadata/plate_1.png"):
                data = archive_file.read(member)
                (preview / Path(member).name).write_bytes(data)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ComboPlateError(
            f"cannot extract composite project previews: {exc}") from exc
    return record


def _prepare_slice(
    *,
    workspace: Path,
    profile_path: Path,
    release_catalog: Path,
    system_root: Path | None,
    bambu_binary: str | None,
) -> dict[str, Any]:
    _local_slice_guard()
    build_source_bundle(
        output_stl=PLATE_STL,
        manifest_path=PLATE_MANIFEST,
    )
    try:
        bambu = captive._find_bambu_binary(bambu_binary)
        base_profile = captive.prepare_profiles(
            profile_path, workspace / "base_profile",
            system_root=system_root, bambu_binary=bambu)
    except captive.AuditError as exc:
        raise ComboPlateError(str(exc)) from exc
    catalog, artifacts = _normalized_artifacts(release_catalog)
    try:
        captive._validate_profile_artifact_scope(
            list(artifacts.values()), base_profile["config"])
    except captive.AuditError as exc:
        raise ComboPlateError(str(exc)) from exc
    first_artifact = artifacts[PARTS[0].artifact_id]  # type: ignore[index]
    try:
        profile_bundle = captive._artifact_profile_bundle(
            first_artifact, base_profile, workspace / "composite_profile")
    except captive.AuditError as exc:
        raise ComboPlateError(
            f"cannot prepare composite profile: {exc}") from exc
    effective = profile_bundle["identity"]["effective"]
    if (effective.get("support_enabled") is not True
            or effective.get("sparse_infill_density_percent")
            != ACTIVE_VARIANT.sparse_infill_density_percent
            or effective.get("sparse_infill_pattern")
            != ACTIVE_VARIANT.sparse_infill_pattern):
        raise ComboPlateError(
            f"the {ACTIVE_VARIANT.state} keyed-LM profile is not support-on "
            f"{ACTIVE_VARIANT.sparse_infill_density_percent:g}% "
            f"{ACTIVE_VARIANT.sparse_infill_pattern}")
    for section_values in profile_bundle["enforced_overrides"].values():
        if not isinstance(section_values, Mapping):
            continue
    process = profile_bundle["resolved"]["process"]
    for key in SUPPORT_KEYS:
        if not captive._boolish(process.get(key)):
            raise ComboPlateError(
                f"resolved composite profile does not pin {key}=1")

    inputs = workspace / "inputs"
    staged_parts = {}
    staged_blockers = {}
    staged_modifiers: dict[str, tuple[dict[str, Any], ...]] = {}
    for part in PARTS:
        staged = inputs / part.staged_name
        _materialize(part.source_stl, staged)
        staged_parts[part.friendly_name] = staged
        if part.support_blocker is not None:
            staged_blocker = inputs / part.support_blocker.name
            _materialize(part.support_blocker, staged_blocker)
            staged_blockers[part.friendly_name] = staged_blocker
        modifiers = []
        if part.artifact_id is not None:
            try:
                resolved_modifiers = captive._parameter_modifiers_for_artifact(
                    artifacts[part.artifact_id], profile_bundle)
            except captive.AuditError as exc:
                raise ComboPlateError(
                    f"{part.friendly_name}: parameter modifier failed: "
                    f"{exc}") from exc
            for modifier in resolved_modifiers:
                source_modifier = Path(modifier["path"])
                staged_modifier = (
                    inputs / "parameter_modifiers" / source_modifier.name)
                _materialize(source_modifier, staged_modifier)
                modifiers.append({**modifier, "path": staged_modifier})
        staged_modifiers[part.friendly_name] = tuple(modifiers)
    ready = workspace / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    assemble_list = ready / "bambu_assemble_list.json"
    custom_gcodes = ready / "custom_gcodes.json"
    _write_assemble_list(
        assemble_list, staged_parts=staged_parts,
        staged_blockers=staged_blockers,
        staged_modifiers=staged_modifiers)
    _write_custom_gcodes(
        custom_gcodes, artifacts=artifacts,
        profile_bundle=profile_bundle)
    command = emit._bambu_command(
        bambu, PLATE_STL, ready, profile_bundle,
        project_filename=f"{PLATE_NAME}.gcode.3mf",
        custom_gcodes=custom_gcodes,
        assemble_list=assemble_list,
    )
    arrange_index = command.index("--arrange")
    command[arrange_index + 1] = "0"
    _write_json(ready / "dry_run_command.json", command)
    fingerprint = sha256_bytes(_canonical_json({
        "source_manifest_sha256": sha256_file(PLATE_MANIFEST),
        "release_catalog_sha256": sha256_file(release_catalog),
        "profile_set_sha256": profile_bundle[
            "identity"]["profile_set_sha256"],
        "bambu_binary_sha256": profile_bundle["identity"]["binary_sha256"],
        "assemble_list_sha256": sha256_file(assemble_list),
        "parameter_modifiers": [{
            "contract_sha256": modifier["contract_sha256"],
            "modifier_stl_sha256": modifier["sha256"],
            "process": modifier["process"],
        } for modifiers in staged_modifiers.values()
          for modifier in modifiers],
        "custom_gcodes_sha256": sha256_file(custom_gcodes),
        "command": command,
    }))
    return {
        "workspace": workspace,
        "ready": ready,
        "bambu": bambu,
        "profile_bundle": profile_bundle,
        "release_catalog": catalog,
        "artifacts": artifacts,
        "staged_parts": staged_parts,
        "staged_blockers": staged_blockers,
        "staged_modifiers": staged_modifiers,
        "assemble_list": assemble_list,
        "custom_gcodes": custom_gcodes,
        "command": command,
        "fingerprint": fingerprint,
    }


def _cache_matches(prepared: Mapping[str, Any]) -> bool:
    ready = Path(prepared["ready"])
    fingerprint_path = ready / "slice_fingerprint.json"
    project = ready / f"{PLATE_NAME}.gcode.3mf"
    gcode = ready / "plate_1.gcode"
    result = ready / "result.json"
    if not all(path.is_file() for path in (
            fingerprint_path, project, gcode, result)):
        return False
    prior = _read_json(fingerprint_path, "composite slice fingerprint")
    return isinstance(prior, Mapping) and all(
        prior.get(key) == value for key, value in {
            "fingerprint": prepared["fingerprint"],
            "project_3mf_sha256": sha256_file(project),
            "gcode_sha256": sha256_file(gcode),
            "result_sha256": sha256_file(result),
        }.items())


def build_or_validate_ready_plate(
    *,
    workspace: Path | None = None,
    profile_path: Path = DEFAULT_PROFILE,
    release_catalog: Path = DEFAULT_RELEASE_CATALOG,
    release_audit: Path = DEFAULT_RELEASE_AUDIT,
    system_root: Path | None = None,
    bambu_binary: str | None = None,
    allow_slice: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    workspace = workspace or DEFAULT_WORKSPACE
    prepared = _prepare_slice(
        workspace=workspace,
        profile_path=profile_path,
        release_catalog=release_catalog,
        system_root=system_root,
        bambu_binary=bambu_binary,
    )
    if dry_run:
        return {
            "name": PLATE_NAME,
            "dry_run": True,
            "command": prepared["command"],
            "fingerprint": prepared["fingerprint"],
        }
    reused = _cache_matches(prepared)
    ready = Path(prepared["ready"])
    project = ready / f"{PLATE_NAME}.gcode.3mf"
    gcode = ready / "plate_1.gcode"
    result = ready / "result.json"
    fingerprint_path = ready / "slice_fingerprint.json"
    if not reused:
        if not allow_slice:
            make_target = (
                "obiwan_floor_combo_plate"
                if ACTIVE_VARIANT.state == "floor_stand"
                else "obiwan_no_floor_combo_plate"
            )
            raise ComboPlateError(
                "composite project is missing or stale; run "
                f"make {make_target}")
        for stale in (
                project, gcode, result, fingerprint_path,
                ready / "bambu_studio.log"):
            stale.unlink(missing_ok=True)
        timeout = int(prepared["profile_bundle"]["config"][
            "slicing"]["timeout_seconds"])
        run = subprocess.run(
            prepared["command"],
            cwd=ready,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        (ready / "bambu_studio.log").write_text(
            run.stdout, encoding="utf-8", errors="replace")
        if run.returncode != 0:
            raise ComboPlateError(
                f"Bambu Studio exited {run.returncode}; see "
                f"{ready / 'bambu_studio.log'}")
        if not all(path.is_file() for path in (project, gcode, result)):
            raise ComboPlateError(
                "Bambu Studio did not create the composite project, G-code, "
                "and result.json")
        try:
            emit._encode_ready_project_custom_gcode_newlines(project)
            emit._inject_ready_project_object_support(
                project, enabled=True)
        except captive.AuditError as exc:
            raise ComboPlateError(
                f"cannot finalize composite project metadata: {exc}") from exc
        _write_json(fingerprint_path, {
            "fingerprint": prepared["fingerprint"],
            "command": prepared["command"],
            "project_3mf_sha256": sha256_file(project),
            "gcode_sha256": sha256_file(gcode),
            "result_sha256": sha256_file(result),
        })
    try:
        emit._validate_bambu_slicer_log(
            ready / "bambu_studio.log",
            artifact_id=PLATE_NAME,
            phase="local composite ready-project slice")
    except captive.AuditError as exc:
        raise ComboPlateError(
            f"composite Bambu log validation failed: {exc}") from exc
    audit = validate_ready_plate(
        workspace=workspace,
        profile_bundle=prepared["profile_bundle"],
        artifacts=prepared["artifacts"],
        staged_parts=prepared["staged_parts"],
        staged_blockers=prepared["staged_blockers"],
        staged_modifiers=prepared["staged_modifiers"],
        release_audit=release_audit,
    )
    audit["slice_reused"] = reused
    _write_json(ready / "plate_audit.json", audit)
    return {
        "name": PLATE_NAME,
        "source_stl": PLATE_STL,
        "source_manifest": PLATE_MANIFEST,
        "project": project,
        "gcode": gcode,
        "result": result,
        "audit_path": ready / "plate_audit.json",
        "audit": audit,
        "profile_effective": prepared[
            "profile_bundle"]["identity"]["effective"],
        "reused": reused,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default="no_floor_stand")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source-only", action="store_true",
        help="build and validate only the translated composite STL")
    mode.add_argument(
        "--dry-run", action="store_true",
        help="write and report the exact local Bambu command without slicing")
    mode.add_argument(
        "--slice-missing", action="store_true",
        help="locally slice when the current audited cache is absent or stale")
    mode.add_argument(
        "--validate-only", action="store_true",
        help="validate the existing current composite project without slicing")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--release-catalog", type=Path, default=DEFAULT_RELEASE_CATALOG)
    parser.add_argument(
        "--release-audit", type=Path, default=DEFAULT_RELEASE_AUDIT)
    parser.add_argument("--bambu-studio")
    parser.add_argument("--bambu-system-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--variant", choices=sorted(VARIANTS), default="no_floor_stand")
    selected, _remaining = selector.parse_known_args(arguments)
    _activate_variant(VARIANTS[selected.variant])
    args = build_parser().parse_args(arguments)
    if args.source_only:
        manifest = build_source_bundle(
            output_stl=PLATE_STL,
            manifest_path=PLATE_MANIFEST,
        )
        print(
            f"{ACTIVE_VARIANT.label} composite STL ready: {PLATE_STL} "
            f"({manifest['triangle_count']} triangles)")
        return 0
    result = build_or_validate_ready_plate(
        workspace=args.workspace.expanduser().resolve(),
        profile_path=args.profile.expanduser().resolve(),
        release_catalog=args.release_catalog.expanduser().resolve(),
        release_audit=args.release_audit.expanduser().resolve(),
        system_root=(
            args.bambu_system_root.expanduser().resolve()
            if args.bambu_system_root else None),
        bambu_binary=args.bambu_studio,
        allow_slice=bool(args.slice_missing),
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            f"{ACTIVE_VARIANT.label} composite plate ready: "
            f"{result['project']} "
            f"(reused={result['reused']})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
            ComboPlateError,
            captive.AuditError,
            Bambu3MFAuditError,
            OSError,
            subprocess.SubprocessError,
    ) as exc:
        print(
            f"Obi-Wan {ACTIVE_VARIANT.label} composite plate failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
