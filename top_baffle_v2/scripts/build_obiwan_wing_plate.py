#!/usr/bin/env python3
"""Build and audit local-only Obi-Wan flat/graded B-wing P2S plates.

This is packaging, not CAD generation.  The four released front-face-down
wing STLs are rigidly rotated about Z and translated into one deterministic
multi-shell STL.  Bambu Studio then slices that locked bundle as one object
with the released support-off wing profile and one six-magnet pause.
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
    audit_bambu_3mf,
    validate_bed_fit,
    validate_result_bbox,
)
from check_manifold import stl_diagnostics
from gcode_analysis import parse_gcode
from lx521_baffle.io import pretty_json_bytes, sha256_bytes, sha256_file
from lx521_baffle.print_contract import (
    FrontDownContractError,
    validate_print_sidecar,
)
import release_validation as captive


ROOT = PROJECT_ROOT
PLATE_OUTPUT_DIR = ROOT / "build" / "print_plates" / "obiwan"
DEFAULT_PROFILE = ROOT / "captive_magnet_slicing_profile.json"
DEFAULT_RELEASE_CATALOG = ROOT / "review" / "captive_magnet_release_catalog.json"
DEFAULT_RELEASE_AUDIT = ROOT / "review" / "captive_magnet_slice_audit"
PAUSE_Z_MM = 5.96
MINIMUM_PART_GAP_MM = 3.5
MINIMUM_BED_EDGE_MM = 3.5
# Bambu recenters this nearly full-bed composite through decimal 3MF component
# transforms.  At the 250-mm coordinate scale the measured float32/XML
# round-trip reaches 1.056e-5 mm; 2e-5 mm remains an exact-mesh identity gate
# (20 nm, with triangle-soup equality still required).
WING_PLATE_3MF_MESH_TOLERANCE_MM = 2.0e-5
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


class WingPlateError(RuntimeError):
    """A locked flat/graded wing-plate contract did not pass."""


@dataclass(frozen=True)
class PlatePart:
    friendly_name: str
    source_stl: Path
    artifact_id: str
    side: str
    role: str
    rz_degrees: float
    translation_xy_mm: tuple[float, float]

    @property
    def matrix4(self) -> tuple[tuple[float, float, float, float], ...]:
        angle = math.radians(self.rz_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        tx, ty = self.translation_xy_mm
        return (
            (cosine, -sine, 0.0, tx),
            (sine, cosine, 0.0, ty),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )


EXPECTED_SITE_NAMES = (
    "lm_lower_left",
    "lm_upper_left",
    "um_left",
    "lm_lower_right",
    "lm_upper_right",
    "um_right",
)

LOCKED_PLACEMENTS_BY_VARIANT = {
    # flat and graded share the same four print-space footprints. Their left LM+UM
    # upper footprint has a 90-degree source-space clocking, so it needs its
    # own rigid transform to retain the common packed layout. The lower-root
    # placements are the locally re-optimized G1 tangent-root arrangement.
    "flat": (
        (58.724273577150, (51.156885879315, 86.406547312980)),
        (-172.921230563781, (248.183921832446, 252.320989480231)),
        (-89.203304927416, (202.926578585501, 167.723983347523)),
        (110.702555432118, (231.314810940662, 67.458534877173)),
    ),
    "graded": (
        (58.724273577150, (51.156885879315, 86.406547312980)),
        (-82.921230563781, (50.756431596864, 227.804382692246)),
        (-89.203304927416, (202.926578585501, 167.723983347523)),
        (110.702555432118, (231.314810940662, 67.458534877173)),
    ),
}


@dataclass(frozen=True)
class WingPlateVariant:
    slug: str
    label: str
    plate_name: str
    expected_triangle_count: int
    parts: tuple[PlatePart, ...]


def _variant(
    slug: str,
    label: str,
    slots: tuple[str, str, str, str],
    expected_triangle_count: int,
) -> WingPlateVariant:
    identities = (
        ("left", "lm_lower", "1_of_2_lm_lower", "LM_lower_left_1_of_2"),
        (
            "left", "lm_um_upper", "2_of_2_lm_um_upper",
            "LM_UM_upper_left_2_of_2",
        ),
        ("right", "lm_lower", "1_of_2_lm_lower", "LM_lower_right_1_of_2"),
        (
            "right", "lm_um_upper", "2_of_2_lm_um_upper",
            "LM_UM_upper_right_2_of_2",
        ),
    )
    parts = []
    for slot, identity, placement in zip(
            slots, identities, LOCKED_PLACEMENTS_BY_VARIANT[slug], strict=True):
        side, role, source_suffix, friendly_suffix = identity
        rz_degrees, translation = placement
        stem = f"obiwan_wing_{slug}_{side}_split2_{source_suffix}"
        parts.append(PlatePart(
            f"obiwan_{slot}_split2_{slug}_wing_{friendly_suffix}",
            ROOT / f"build/wings/{slug}/stl/{stem}.stl",
            f"shared:Obi-Wan-{label}:{stem}",
            side,
            role,
            rz_degrees,
            translation,
        ))
    return WingPlateVariant(
        slug=slug,
        label=label,
        plate_name=f"obiwan_{slug}_wings_split2_combo",
        expected_triangle_count=expected_triangle_count,
        parts=tuple(parts),
    )


# The triangle counts are the sum of the four released split2 meshes, pinned
# so an unintended mesh change cannot reach a plate.  They therefore have to
# move in the same commit as any deliberate wing regeneration: the graded
# count fell from 2_169_008 to 734_014 when the uncut rim was cut, and until
# it was moved this builder failed while the previous, phantom-walled plate
# STL stayed on disk and stayed hard-linked onto the shelf.
VARIANTS = {
    "flat": _variant("flat", "Flat", ("05", "06", "08", "09"), 17_836),
    "graded": _variant("graded", "Graded", ("11", "12", "14", "15"), 734_014),
}


def _activate_variant(variant: WingPlateVariant) -> None:
    global ACTIVE_VARIANT
    global PLATE_NAME, PLATE_STL, PLATE_MANIFEST, PLATE_PREVIEW
    global DEFAULT_WORKSPACE, EXPECTED_TRIANGLE_COUNT, PARTS
    ACTIVE_VARIANT = variant
    PLATE_NAME = variant.plate_name
    PLATE_STL = PLATE_OUTPUT_DIR / f"{PLATE_NAME}.stl"
    PLATE_MANIFEST = PLATE_OUTPUT_DIR / f"{PLATE_NAME}.plate.json"
    PLATE_PREVIEW = PLATE_OUTPUT_DIR / f"{PLATE_NAME}.layout.png"
    DEFAULT_WORKSPACE = (
        ROOT / "review" / "to_print_slice_workspace"
        / "composite" / PLATE_NAME
    )
    EXPECTED_TRIANGLE_COUNT = variant.expected_triangle_count
    PARTS = variant.parts


@dataclass(frozen=True)
class WingPlateAPI:
    """Variant-bound API consumed by the shelf and focused tests."""

    variant: WingPlateVariant

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
    def PLATE_STL(self) -> Path:
        return PLATE_OUTPUT_DIR / f"{self.PLATE_NAME}.stl"

    @property
    def PLATE_MANIFEST(self) -> Path:
        return PLATE_OUTPUT_DIR / f"{self.PLATE_NAME}.plate.json"

    @property
    def PLATE_PREVIEW(self) -> Path:
        return PLATE_OUTPUT_DIR / f"{self.PLATE_NAME}.layout.png"

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


def get_variant(slug: str) -> WingPlateAPI:
    try:
        return WingPlateAPI(VARIANTS[slug.lower()])
    except KeyError as exc:
        raise WingPlateError(f"unknown wing-plate variant {slug!r}") from exc


_activate_variant(VARIANTS["graded"])


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WingPlateError(f"cannot read {label} {path}: {exc}") from exc


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
        raise WingPlateError(f"{path} is too short to be a binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if count <= 0 or len(data) != 84 + 50 * count:
        raise WingPlateError(
            f"{path} must be a non-empty, exact-length binary STL")
    return tuple(
        data[84 + 50 * index:84 + 50 * (index + 1)]
        for index in range(count)
    )


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _transform_xy(
    x: float,
    y: float,
    matrix: Sequence[Sequence[float]],
    *,
    translate: bool,
) -> tuple[float, float]:
    tx = float(matrix[0][3]) if translate else 0.0
    ty = float(matrix[1][3]) if translate else 0.0
    return (
        _float32(float(matrix[0][0]) * float(x)
                 + float(matrix[0][1]) * float(y) + tx),
        _float32(float(matrix[1][0]) * float(x)
                 + float(matrix[1][1]) * float(y) + ty),
    )


def _transform_stl_record(record: bytes, part: PlatePart) -> bytes:
    if len(record) != 50:
        raise WingPlateError("invalid binary STL triangle record")
    matrix = part.matrix4
    result = bytearray(record)
    nx, ny, nz = struct.unpack_from("<3f", record, 0)
    transformed_normal = (*_transform_xy(
        nx, ny, matrix, translate=False), _float32(nz))
    struct.pack_into("<3f", result, 0, *transformed_normal)
    values = list(struct.unpack_from("<9f", record, 12))
    for vertex in range(3):
        index = vertex * 3
        x, y = _transform_xy(
            values[index], values[index + 1], matrix, translate=True)
        values[index] = x
        values[index + 1] = y
        values[index + 2] = _float32(values[index + 2])
    struct.pack_into("<9f", result, 12, *values)
    return bytes(result)


def _strict_source_mesh(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise WingPlateError(f"missing released source STL: {path}")
    facts = stl_diagnostics(path)
    failures = {
        key: facts.get(key)
        for key in STRICT_MESH_KEYS
        if int(facts.get(key, 0)) != 0
    }
    if failures:
        raise WingPlateError(
            f"{path.name}: released source STL failed strict topology: "
            f"{failures}")
    return facts


def _shapely_matrix(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [
        float(matrix[0][0]), float(matrix[0][1]),
        float(matrix[1][0]), float(matrix[1][1]),
        float(matrix[0][3]), float(matrix[1][3]),
    ]


def _authoritative_footprints() -> dict[str, Any]:
    os.environ.setdefault("LX_ROUTING_PROFILE", "obiwan")
    os.environ.setdefault("LX_STAND_FOOT", "0")
    try:
        from shapely import affinity
        from lx521_baffle.obiwan.wings import (
            wing_two_piece_print_plan_parts,
        )
    except ImportError as exc:
        raise WingPlateError(
            "Shapely and the canonical wing plan are required") from exc

    footprints = {}
    for part in PARTS:
        plan = wing_two_piece_print_plan_parts(
            ACTIVE_VARIANT.slug, part.side)[part.role]
        sidecar = _read_json(
            part.source_stl.with_suffix(".print.json"),
            f"{part.friendly_name} print sidecar",
        )
        source_to_stl = sidecar.get("source_to_stl_matrix")
        if (not isinstance(source_to_stl, list)
                or len(source_to_stl) != 4):
            raise WingPlateError(
                f"{part.friendly_name}: source-to-STL matrix is invalid")
        stl_plan = affinity.affine_transform(
            plan, _shapely_matrix(source_to_stl))
        footprint = affinity.affine_transform(
            stl_plan, _shapely_matrix(part.matrix4))
        if footprint.is_empty or not footprint.is_valid:
            raise WingPlateError(
                f"{part.friendly_name}: transformed print footprint is invalid")
        footprints[part.friendly_name] = footprint
    return footprints


def _packing_facts(
    footprints: Mapping[str, Any],
) -> dict[str, Any]:
    pairwise = []
    minimum_gap = math.inf
    for index, left in enumerate(PARTS):
        for right in PARTS[index + 1:]:
            left_shape = footprints[left.friendly_name]
            right_shape = footprints[right.friendly_name]
            intersection = float(left_shape.intersection(right_shape).area)
            distance = float(left_shape.distance(right_shape))
            if intersection > 1.0e-9 or distance < MINIMUM_PART_GAP_MM:
                raise WingPlateError(
                    "locked wing placement collides or has insufficient gap: "
                    f"{left.friendly_name} vs {right.friendly_name}, "
                    f"intersection={intersection:.9f} mm2, "
                    f"distance={distance:.6f} mm")
            minimum_gap = min(minimum_gap, distance)
            pairwise.append({
                "left": left.friendly_name,
                "right": right.friendly_name,
                "xy_gap_mm": distance,
                "intersection_area_mm2": intersection,
            })

    edge_records = []
    minimum_edge = math.inf
    for part in PARTS:
        minimum_x, minimum_y, maximum_x, maximum_y = (
            footprints[part.friendly_name].bounds)
        clearances = {
            "x_low": float(minimum_x),
            "x_high": float(256.0 - maximum_x),
            "y_low": float(minimum_y),
            "y_high": float(256.0 - maximum_y),
        }
        part_minimum = min(clearances.values())
        if part_minimum < MINIMUM_BED_EDGE_MM:
            raise WingPlateError(
                f"{part.friendly_name}: footprint bed-edge clearance "
                f"{part_minimum:.6f} mm < {MINIMUM_BED_EDGE_MM:.3f} mm")
        minimum_edge = min(minimum_edge, part_minimum)
        edge_records.append({
            "part": part.friendly_name,
            "clearance_mm": clearances,
        })
    return {
        "minimum_required_xy_gap_mm": MINIMUM_PART_GAP_MM,
        "minimum_actual_xy_gap_mm": minimum_gap,
        "minimum_required_bed_edge_mm": MINIMUM_BED_EDGE_MM,
        "minimum_actual_bed_edge_mm": minimum_edge,
        "pairwise": pairwise,
        "bed_edges": edge_records,
    }


def _validate_mesh_witnesses(
    footprints: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    try:
        import numpy as np
        from shapely import contains_xy
    except ImportError as exc:
        raise WingPlateError(
            "NumPy and Shapely are required for mesh-footprint validation"
        ) from exc

    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    records = {}
    for part in PARTS:
        source = np.fromfile(part.source_stl, dtype=dtype, offset=84)
        if len(source) == 0:
            raise WingPlateError(f"{part.source_stl} has no triangles")
        vertices = source["vertices"].astype(np.float64)
        matrix = np.asarray(part.matrix4, dtype=np.float64)
        transformed = (
            vertices[:, :, :2] @ matrix[:2, :2].T
            + matrix[:2, 3]
        )
        flattened = transformed.reshape(-1, 2)
        centroids = transformed.mean(axis=1)
        guard = footprints[part.friendly_name].buffer(0.005)
        outside_vertices = int(np.count_nonzero(~contains_xy(
            guard, flattened[:, 0], flattened[:, 1])))
        outside_centroids = int(np.count_nonzero(~contains_xy(
            guard, centroids[:, 0], centroids[:, 1])))
        if outside_vertices or outside_centroids:
            raise WingPlateError(
                f"{part.friendly_name}: transformed STL leaves its "
                "authoritative footprint; "
                f"vertices={outside_vertices}, centroids={outside_centroids}")
        records[part.friendly_name] = {
            "triangle_count": int(len(source)),
            "vertices_outside_0_005mm_guard": outside_vertices,
            "triangle_centroids_outside_0_005mm_guard": outside_centroids,
        }
    return records


def _render_layout(
    path: Path,
    footprints: Mapping[str, Any],
    packing: Mapping[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise WingPlateError(
            "Matplotlib is required for the wing-plate review image") from exc

    colors = ("#3f8fc5", "#e48f28", "#56a764", "#c95d73")
    fig, axis = plt.subplots(figsize=(9.0, 9.0), dpi=180)
    axis.add_patch(Rectangle(
        (0.0, 0.0), 256.0, 256.0,
        facecolor="#f2f2f2", edgecolor="#242424", linewidth=1.4,
    ))
    for part, color in zip(PARTS, colors, strict=True):
        footprint = footprints[part.friendly_name]
        x_values, y_values = footprint.exterior.xy
        axis.fill(
            x_values, y_values, color=color, alpha=0.72,
            edgecolor=color, linewidth=1.0,
            label=part.friendly_name.split("_of_16_", 1)[0],
        )
    axis.text(
        128.0, 128.0,
        f"minimum part gap: "
        f"{packing['minimum_actual_xy_gap_mm']:.2f} mm\n"
        f"minimum bed edge: "
        f"{packing['minimum_actual_bed_edge_mm']:.2f} mm",
        ha="center", va="center", fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#555", "alpha": 0.92},
    )
    axis.set_title(
        f"Obi-Wan {ACTIVE_VARIANT.label} B wings — "
        "locked front-face-down P2S plate",
        fontsize=12, weight="bold",
    )
    axis.set_xlim(-4.0, 260.0)
    axis.set_ylim(-4.0, 260.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Bambu bed X (mm)")
    axis.set_ylabel("Bambu bed Y (mm)")
    axis.grid(True, color="#d2d2d2", linewidth=0.45)
    axis.legend(loc="center", fontsize=7.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _validate_exact_payload(stl: Path) -> None:
    actual = _binary_stl_records(stl)
    expected_count = sum(
        len(_binary_stl_records(part.source_stl)) for part in PARTS)
    if len(actual) != expected_count:
        raise WingPlateError(
            f"composite STL has {len(actual)} triangles, "
            f"expected {expected_count}")
    index = 0
    for part in PARTS:
        for source_record in _binary_stl_records(part.source_stl):
            expected = _transform_stl_record(source_record, part)
            if actual[index] != expected:
                raise WingPlateError(
                    f"composite STL differs at triangle {index} "
                    f"from {part.friendly_name}")
            index += 1


def build_source_bundle(
    *,
    output_stl: Path = PLATE_STL,
    manifest_path: Path = PLATE_MANIFEST,
    preview_path: Path = PLATE_PREVIEW,
) -> dict[str, Any]:
    """Write the deterministic rigidly packed STL and its scoped contract."""
    part_records = []
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_stl.with_suffix(output_stl.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write((
            f"LX521 Obi-Wan {ACTIVE_VARIANT.label} B-wing locked P2S plate"
        ).encode("ascii").ljust(80, b"\0"))
        stream.write(struct.pack("<I", EXPECTED_TRIANGLE_COUNT))
        actual_count = 0
        for part in PARTS:
            sidecar = part.source_stl.with_suffix(".print.json")
            try:
                validate_print_sidecar(part.source_stl, sidecar)
            except FrontDownContractError as exc:
                raise WingPlateError(
                    f"{part.friendly_name}: invalid print authority: {exc}"
                ) from exc
            mesh = _strict_source_mesh(part.source_stl)
            source_records = _binary_stl_records(part.source_stl)
            for record in source_records:
                stream.write(_transform_stl_record(record, part))
            actual_count += len(source_records)
            part_records.append({
                "friendly_name": part.friendly_name,
                "source_stl": _relative(part.source_stl),
                "source_stl_sha256": sha256_file(part.source_stl),
                "source_print_sidecar": _relative(sidecar),
                "source_print_sidecar_sha256": sha256_file(sidecar),
                "catalog_artifact_id": part.artifact_id,
                "side": part.side,
                "role": part.role,
                "rz_degrees": part.rz_degrees,
                "translation_xy_mm": list(part.translation_xy_mm),
                "stl_to_plate_matrix": [
                    list(row) for row in part.matrix4
                ],
                "triangle_count": len(source_records),
                "mesh_diagnostics": mesh,
            })
    if actual_count != EXPECTED_TRIANGLE_COUNT:
        temporary.unlink(missing_ok=True)
        raise WingPlateError(
            f"composite triangle count {actual_count} != "
            f"{EXPECTED_TRIANGLE_COUNT}")
    temporary.replace(output_stl)
    _validate_exact_payload(output_stl)

    footprints = _authoritative_footprints()
    packing = _packing_facts(footprints)
    witnesses = _validate_mesh_witnesses(footprints)
    _render_layout(preview_path, footprints, packing)
    mesh = captive.inspect_stl(output_stl)
    clearances = validate_bed_fit(_bounds3d(mesh), MACHINE_BOUNDS_MM)
    manifest = {
        "schema_version": 1,
        "manifest_kind": "lx521_locked_composite_print_plate",
        "name": PLATE_NAME,
        "source_policy": (
            "rigid Rz plus XY placement of four released front-face-down "
            "STLs; no CAD, BREP, support structure, or acoustic geometry "
            "regeneration"),
        "stl": output_stl.name,
        "stl_sha256": sha256_file(output_stl),
        "stl_bytes": output_stl.stat().st_size,
        "triangle_count": mesh.triangle_count,
        "expected_disconnected_printable_part_count": len(PARTS),
        "composite_topology_policy": (
            "each released source passes strict manifold validation and the "
            "bundle is their exact, non-intersecting rigid transform"),
        "bounds_mm": {
            "minimum_mm": list(mesh.bounds_min),
            "maximum_mm": list(mesh.bounds_max),
            "size_mm": list(mesh.size),
        },
        "bed_clearances_mm": {
            axis: list(values) for axis, values in clearances.items()
        },
        "packing": packing,
        "mesh_footprint_witnesses": witnesses,
        "parts": part_records,
        "preview": _relative(preview_path),
        "preview_sha256": sha256_file(preview_path),
        "support_policy": {
            "enabled": False,
            "global_and_object_fields": {
                key: "0" for key in SUPPORT_KEYS
            },
            "support_blocker_count": 0,
            "support_toolpath_gate": "no support feature blocks permitted",
        },
        "magnet_pause": {
            "pause_z_mm": PAUSE_Z_MM,
            "magnet_count": 6,
            "sites": list(EXPECTED_SITE_NAMES),
        },
    }
    manifest["source_bundle_fingerprint"] = sha256_bytes(_canonical_json({
        "name": manifest["name"],
        "stl_sha256": manifest["stl_sha256"],
        "parts": manifest["parts"],
        "packing": manifest["packing"],
        "support_policy": manifest["support_policy"],
        "magnet_pause": manifest["magnet_pause"],
    }))
    _write_json(manifest_path, manifest)
    validate_source_bundle(output_stl, manifest_path)
    return manifest


def _bounds3d(mesh):
    from bambu_3mf_audit import Bounds3D
    return Bounds3D(mesh.bounds_min, mesh.bounds_max)


def validate_source_bundle(
    stl: Path = PLATE_STL,
    manifest_path: Path = PLATE_MANIFEST,
) -> dict[str, Any]:
    payload = _read_json(manifest_path, "wing plate manifest")
    if (not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or payload.get("manifest_kind")
            != "lx521_locked_composite_print_plate"
            or payload.get("name") != PLATE_NAME):
        raise WingPlateError("wing plate manifest identity is invalid")
    if (not stl.is_file()
            or payload.get("stl") != stl.name
            or payload.get("stl_sha256") != sha256_file(stl)
            or payload.get("stl_bytes") != stl.stat().st_size):
        raise WingPlateError("wing plate STL does not match its manifest")
    records = payload.get("parts")
    if not isinstance(records, list) or len(records) != len(PARTS):
        raise WingPlateError("wing plate part inventory is incomplete")
    for part, record in zip(PARTS, records, strict=True):
        if not isinstance(record, Mapping):
            raise WingPlateError("wing plate part record is invalid")
        expected = {
            "friendly_name": part.friendly_name,
            "source_stl": _relative(part.source_stl),
            "source_stl_sha256": sha256_file(part.source_stl),
            "source_print_sidecar": _relative(
                part.source_stl.with_suffix(".print.json")),
            "source_print_sidecar_sha256": sha256_file(
                part.source_stl.with_suffix(".print.json")),
            "catalog_artifact_id": part.artifact_id,
            "rz_degrees": part.rz_degrees,
            "translation_xy_mm": list(part.translation_xy_mm),
            "stl_to_plate_matrix": [
                list(row) for row in part.matrix4
            ],
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise WingPlateError(
                    f"{part.friendly_name}: source contract drifted at {key}")
        try:
            validate_print_sidecar(part.source_stl)
        except FrontDownContractError as exc:
            raise WingPlateError(
                f"{part.friendly_name}: source sidecar failed: {exc}") from exc
        _strict_source_mesh(part.source_stl)
    _validate_exact_payload(stl)
    mesh = captive.inspect_stl(stl)
    if (mesh.triangle_count != EXPECTED_TRIANGLE_COUNT
            or payload.get("triangle_count") != EXPECTED_TRIANGLE_COUNT):
        raise WingPlateError("wing plate triangle count drifted")
    validate_bed_fit(_bounds3d(mesh), MACHINE_BOUNDS_MM)
    packing = _packing_facts(_authoritative_footprints())
    recorded = payload.get("packing")
    if not isinstance(recorded, Mapping):
        raise WingPlateError("wing plate packing evidence is missing")
    for key in (
            "minimum_actual_xy_gap_mm",
            "minimum_actual_bed_edge_mm"):
        if not math.isclose(
                float(recorded.get(key, -1.0)),
                float(packing[key]),
                abs_tol=1.0e-9, rel_tol=0.0):
            raise WingPlateError(f"wing plate packing drifted at {key}")
    return dict(payload)


def _local_slice_guard() -> None:
    if platform.system() != "Darwin":
        raise WingPlateError(
            f"the {ACTIVE_VARIANT.label} wing plate may be sliced only "
            "on the local Mac")
    hostname = socket.gethostname().split(".", 1)[0].lower()
    if hostname == "osado" or hostname.startswith("osado-"):
        raise WingPlateError(
            f"refusing to slice the {ACTIVE_VARIANT.label} wing plate "
            "on osado")
    execution = os.environ.get("LX_CAD_EXECUTION", "").strip().lower()
    if execution in {"remote", "remote-worker"}:
        raise WingPlateError(
            f"refusing local slicing under LX_CAD_EXECUTION={execution}")


def _materialize(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise WingPlateError(f"cannot stage missing input {source}")
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
        raise WingPlateError(
            f"cannot normalize captive-magnet release catalog: {exc}") from exc
    by_id = {artifact["id"]: artifact for artifact in catalog["artifacts"]}
    selected = {}
    for part in PARTS:
        artifact = by_id.get(part.artifact_id)
        if artifact is None:
            raise WingPlateError(f"missing release artifact {part.artifact_id}")
        if Path(artifact["stl"]).resolve() != part.source_stl.resolve():
            raise WingPlateError(
                f"{part.friendly_name}: release source STL drifted")
        if artifact["stl_catalog_sha256"] != sha256_file(part.source_stl):
            raise WingPlateError(
                f"{part.friendly_name}: release source hash drifted")
        if "support_blocker" in artifact:
            raise WingPlateError(
                f"{part.friendly_name}: wing unexpectedly requires support")
        selected[part.artifact_id] = dict(artifact)
    return catalog, selected


def _write_assemble_list(path: Path, staged_stl: Path) -> None:
    _write_json(path, {
        "plates": [{
            "plate_name": PLATE_NAME,
            "need_arrange": False,
            "objects": [{
                "path": str(staged_stl.resolve()),
                "subtype": "normal_part",
                "count": 1,
                "filaments": [1],
                "assemble_index": [1],
                "pos_x": [0.0],
                "pos_y": [0.0],
                "pos_z": [0.0],
            }],
            "assembled_params": [{
                "assemble_index": 1,
                "print_params": {
                    **{key: "0" for key in SUPPORT_KEYS},
                    "sparse_infill_density": "30%",
                    "sparse_infill_pattern": "gyroid",
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
    names = []
    pauses = set()
    for part in PARTS:
        for site in artifacts[part.artifact_id]["sites"]:
            names.append(str(site["name"]))
            pauses.add(float(site["expected_pause_marker_z_mm"]))
    if tuple(names) != EXPECTED_SITE_NAMES or pauses != {PAUSE_Z_MM}:
        raise WingPlateError(
            "wing plate magnet-site or pause inventory drifted")
    pause_policy = profile_bundle["identity"]["effective"].get(
        "magnet_insertion_pause")
    if not isinstance(pause_policy, Mapping):
        raise WingPlateError(
            "resolved profile lacks the magnet insertion pause policy")
    group = {
        "pause_marker_z_mm": PAUSE_Z_MM,
        "sites": names,
        "magnet_count": len(names),
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


def _matrix_multiply(left, right):
    return tuple(tuple(
        sum(float(left[row][index]) * float(right[index][column])
            for index in range(4))
        for column in range(4)
    ) for row in range(4))


def _discovery_record(release_audit: Path, artifact_id: str) -> dict[str, Any]:
    path = (
        release_audit / "slices" / captive._slug(artifact_id)
        / "captive_magnet_slice_audit.json")
    record = _read_json(path, f"{artifact_id} authoritative slice audit")
    if (not isinstance(record, Mapping)
            or record.get("id") != artifact_id
            or record.get("status") != "pass"):
        raise WingPlateError(
            f"{artifact_id}: authoritative discovery audit is not passing")
    return dict(record)


def _result_object(result_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _read_json(result_path, "Bambu result")
    if not isinstance(result, Mapping) or result.get("return_code") != 0:
        raise WingPlateError("Bambu result is not Success")
    plates = result.get("sliced_plates")
    if not isinstance(plates, list) or len(plates) != 1:
        raise WingPlateError("wing slice must contain exactly one plate")
    plate = plates[0]
    objects = plate.get("objects") if isinstance(plate, Mapping) else None
    if not isinstance(objects, list) or len(objects) != 1:
        raise WingPlateError(
            "wing slice must contain exactly one printable object")
    obj = objects[0]
    if (int(plate.get("triangle_count", -1)) != EXPECTED_TRIANGLE_COUNT
            or int(obj.get("triangle_count", -1))
            != EXPECTED_TRIANGLE_COUNT):
        raise WingPlateError(
            "Bambu triangle count differs from the composite STL")
    if plate.get("warning_message") not in ("", None):
        raise WingPlateError(
            f"Bambu plate warning: {plate.get('warning_message')}")
    return dict(plate), dict(obj)


def validate_ready_plate(
    *,
    workspace: Path,
    profile_bundle: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    staged_stl: Path,
    release_audit: Path,
) -> dict[str, Any]:
    ready = workspace / "ready"
    project = ready / f"{PLATE_NAME}.gcode.3mf"
    gcode = ready / "plate_1.gcode"
    result_path = ready / "result.json"
    for path in (project, gcode, result_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise WingPlateError(f"missing wing slice output: {path}")
    source_manifest = validate_source_bundle(PLATE_STL, PLATE_MANIFEST)
    try:
        project_audit = audit_bambu_3mf(
            project,
            staged_stl,
            mesh_tolerance_mm=WING_PLATE_3MF_MESH_TOLERANCE_MM,
        )
    except Bambu3MFAuditError as exc:
        raise WingPlateError(
            f"wing project/STL equivalence failed: {exc}") from exc
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
        raise WingPlateError(
            "Bambu moved or rotated the locked wing plate")
    clearances = validate_bed_fit(
        project_audit.transformed_actual_mesh_bounds,
        profile_bundle["identity"]["machine_bounds_mm"])
    plate, obj = _result_object(result_path)
    bbox = obj.get("bbox")
    if not isinstance(bbox, Mapping):
        raise WingPlateError("Bambu result lacks the wing object bbox")
    try:
        validate_result_bbox(
            bbox, project_audit.source_bounds,
            project_audit.stl_to_bed_matrix)
    except Bambu3MFAuditError as exc:
        raise WingPlateError(
            f"Bambu result/3MF placement mismatch: {exc}") from exc
    try:
        archive = emit._validate_ready_project_archive(
            project, gcode, expected_pause_z=[PAUSE_Z_MM],
            profile_bundle=profile_bundle)
    except captive.AuditError as exc:
        raise WingPlateError(
            f"wing ready-project archive audit failed: {exc}") from exc
    overrides = archive.get("object_support_overrides", ())
    if len(overrides) != 1 or any(
            str(overrides[0].get(key)) != "0" for key in SUPPORT_KEYS):
        raise WingPlateError(
            "wing project does not pin all four object support fields off")

    cavity_records = {}
    parsed_for_pause = None
    outer = project_audit.stl_to_bed_matrix
    for part in PARTS:
        artifact = artifacts[part.artifact_id]
        try:
            parsed, cavity = emit._validate_ready_cavity_toolpaths(
                artifact=artifact,
                discovery_record=_discovery_record(
                    release_audit, part.artifact_id),
                gcode=gcode,
                stl_to_bed_matrix=_matrix_multiply(
                    outer, part.matrix4),
            )
        except captive.AuditError as exc:
            raise WingPlateError(
                f"{part.friendly_name}: captive-cavity audit failed: {exc}"
            ) from exc
        parsed_for_pause = parsed_for_pause or parsed
        cavity_records[part.friendly_name] = cavity
    if parsed_for_pause is None:
        raise WingPlateError("wing plate has no captive-cavity audit")
    try:
        pause_before_extrusion = emit._assert_pauses_precede_layer_extrusion(
            parsed_for_pause, archive["gcode_pause_events"])
    except captive.AuditError as exc:
        raise WingPlateError(
            f"wing magnet pause ordering failed: {exc}") from exc

    parsed_profile = parse_gcode(gcode, retain_regions=())
    support_summary = emit._support_toolpath_summary(gcode)
    if (support_summary["support_feature_blocks"] != 0
            or support_summary["support_interface_feature_blocks"] != 0):
        raise WingPlateError(
            "support-disabled wing plate emitted support toolpaths")
    profile_errors = emit._validate_actual_gcode_profile(
        parsed_profile, profile_bundle)
    if profile_errors:
        raise WingPlateError(
            "wing G-code profile mismatch: " + "; ".join(profile_errors))
    static_validation = emit._validate_with_gcode_skill(
        gcode, ready, profile_bundle)
    if static_validation.get("ok") is not True:
        raise WingPlateError(
            "wing G-code static validation did not pass")
    effective = profile_bundle["identity"]["effective"]
    if (effective.get("sparse_infill_density_percent") != 30.0
            or effective.get("sparse_infill_pattern") != "gyroid"
            or effective.get("support_enabled") is not False):
        raise WingPlateError(
            "wing effective profile is not support-off 30% gyroid")

    archive["duct_support_toolpath_audit"] = {
        "status": "pass",
        "collision_count": 0,
        "gate": "support_disabled_no_support_feature_blocks",
    }
    record = {
        "schema_version": 1,
        "audit_kind": (
            f"lx521_locked_{ACTIVE_VARIANT.slug}_wing_print_plate"),
        "name": PLATE_NAME,
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
            "estimated_print_time_seconds": plate.get("total_predication"),
            "filaments": plate.get("filaments"),
        },
        "archive_audit": archive,
        "captive_cavity_audit": cavity_records,
        "pause_before_first_layer_extrusion": pause_before_extrusion,
        "support_toolpaths": support_summary,
        "duct_support_toolpath_audit": archive[
            "duct_support_toolpath_audit"],
        "gcode_static_validation": static_validation,
        "profile_effective": dict(effective),
    }
    _write_json(ready / "plate_audit.json", record)
    preview = ready / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(project) as archive_file:
            for member in ("Metadata/top_1.png", "Metadata/plate_1.png"):
                (preview / Path(member).name).write_bytes(
                    archive_file.read(member))
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise WingPlateError(
            f"cannot extract wing project previews: {exc}") from exc
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
        preview_path=PLATE_PREVIEW,
    )
    try:
        bambu = captive._find_bambu_binary(bambu_binary)
        base_profile = captive.prepare_profiles(
            profile_path, workspace / "base_profile",
            system_root=system_root, bambu_binary=bambu)
    except captive.AuditError as exc:
        raise WingPlateError(str(exc)) from exc
    catalog, artifacts = _normalized_artifacts(release_catalog)
    first = artifacts[PARTS[0].artifact_id]
    try:
        profile_bundle = captive._artifact_profile_bundle(
            first, base_profile, workspace / "composite_profile")
    except captive.AuditError as exc:
        raise WingPlateError(
            f"cannot prepare wing profile: {exc}") from exc
    effective = profile_bundle["identity"]["effective"]
    if (effective.get("support_enabled") is not False
            or effective.get("sparse_infill_density_percent") != 30.0
            or effective.get("sparse_infill_pattern") != "gyroid"):
        raise WingPlateError(
            "released wing profile is not support-off 30% gyroid")
    process = profile_bundle["resolved"]["process"]
    if any(captive._boolish(process.get(key)) for key in SUPPORT_KEYS):
        raise WingPlateError(
            "resolved wing profile does not pin all support fields off")

    inputs = workspace / "inputs"
    staged_stl = inputs / PLATE_STL.name
    _materialize(PLATE_STL, staged_stl)
    ready = workspace / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    assemble_list = ready / "bambu_assemble_list.json"
    custom_gcodes = ready / "custom_gcodes.json"
    _write_assemble_list(assemble_list, staged_stl)
    _write_custom_gcodes(
        custom_gcodes, artifacts=artifacts,
        profile_bundle=profile_bundle)
    command = emit._bambu_command(
        bambu, staged_stl, ready, profile_bundle,
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
        "staged_stl": staged_stl,
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
    prior = _read_json(fingerprint_path, "wing slice fingerprint")
    return isinstance(prior, Mapping) and all(
        prior.get(key) == value for key, value in {
            "fingerprint": prepared["fingerprint"],
            "project_3mf_sha256": sha256_file(project),
            "gcode_sha256": sha256_file(gcode),
            "result_sha256": sha256_file(result),
        }.items())


def build_or_validate_ready_plate(
    *,
    workspace: Path = DEFAULT_WORKSPACE,
    profile_path: Path = DEFAULT_PROFILE,
    release_catalog: Path = DEFAULT_RELEASE_CATALOG,
    release_audit: Path = DEFAULT_RELEASE_AUDIT,
    system_root: Path | None = None,
    bambu_binary: str | None = None,
    allow_slice: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
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
            raise WingPlateError(
                "wing project is missing or stale; run "
                "make obiwan_graded_wing_plate")
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
            raise WingPlateError(
                f"Bambu Studio exited {run.returncode}; see "
                f"{ready / 'bambu_studio.log'}")
        if not all(path.is_file() for path in (project, gcode, result)):
            raise WingPlateError(
                "Bambu Studio did not create the wing project, G-code, "
                "and result.json")
        try:
            emit._encode_ready_project_custom_gcode_newlines(project)
            emit._inject_ready_project_object_support(
                project, enabled=False)
        except captive.AuditError as exc:
            raise WingPlateError(
                f"cannot finalize wing project metadata: {exc}") from exc
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
            phase=(
                f"local {ACTIVE_VARIANT.label} wing ready-project slice"))
    except captive.AuditError as exc:
        raise WingPlateError(
            f"wing Bambu log validation failed: {exc}") from exc
    audit = validate_ready_plate(
        workspace=workspace,
        profile_bundle=prepared["profile_bundle"],
        artifacts=prepared["artifacts"],
        staged_stl=prepared["staged_stl"],
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
        "--variant", choices=sorted(VARIANTS),
        default=ACTIVE_VARIANT.slug,
        help="wing acoustic variant to package")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source-only", action="store_true",
        help="build and validate only the rigidly packed composite STL")
    mode.add_argument(
        "--dry-run", action="store_true",
        help="write and report the exact local Bambu command without slicing")
    mode.add_argument(
        "--slice-missing", action="store_true",
        help="locally slice when the current audited cache is absent or stale")
    mode.add_argument(
        "--validate-only", action="store_true",
        help="validate the existing current wing project without slicing")
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
        "--variant", choices=sorted(VARIANTS), default="graded")
    selected, _remaining = selector.parse_known_args(arguments)
    _activate_variant(VARIANTS[selected.variant])
    args = build_parser().parse_args(arguments)
    if args.source_only:
        manifest = build_source_bundle(
            output_stl=PLATE_STL,
            manifest_path=PLATE_MANIFEST,
            preview_path=PLATE_PREVIEW,
        )
        print(
            f"{ACTIVE_VARIANT.label} wing plate STL ready: {PLATE_STL} "
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
            f"{ACTIVE_VARIANT.label} wing plate ready: {result['project']} "
            f"(reused={result['reused']})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
            WingPlateError,
            captive.AuditError,
            Bambu3MFAuditError,
            OSError,
            subprocess.SubprocessError,
    ) as exc:
        print(
            f"Obi-Wan {ACTIVE_VARIANT.label} wing plate failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
