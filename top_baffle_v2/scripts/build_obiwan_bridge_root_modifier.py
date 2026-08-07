#!/usr/bin/env python3
"""Build the local-solid process modifier for Obi-Wan no-floor 01a.

This is slicer geometry, not printable CAD.  The convex source-space prism
covers the complete shallow bridge, its soft LM transition, and both lower LM
driver bosses.  It is transformed by the released 01a front-down matrix so the
same modifier can be assembled with either the individual part or a translated
multi-part plate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    text = str(import_root)
    if text not in sys.path:
        sys.path.insert(0, text)

from lx521_baffle.io import pretty_json_bytes, sha256_file
from lx521_baffle.print_contract import (
    FrontDownContractError,
    validate_print_sidecar,
)


PART = "lx521_top_obiwan_optional_lm_keyed_1of2_bottom"
ARTIFACT_MATCH = {
    "state": "no_floor_stand",
    "variant": "Obi-Wan-split",
    "part": PART,
}
DEFAULT_STL = (
    PROJECT_ROOT / "build/no_floor_stand/stl" / f"{PART}.stl"
)
DEFAULT_SIDECAR = DEFAULT_STL.with_suffix(".print.json")
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "build/no_floor_stand/process_modifiers"
    / f"{PART}.bridge_root_100pct.modifier.stl"
)
DEFAULT_CONTRACT = DEFAULT_OUTPUT.with_suffix(".json")

# Source XY is deliberately inside the released carrier envelope.  The W62
# bridge is covered exactly through its straight core; the widening upper half
# reaches the complete lower-ring load path and both lower LM pilot axes at
# (+/-52.375, 110.265).  Empty ducts and the acoustic opening remain empty:
# a Bambu modifier changes parameters only where it intersects printable solid.
SOURCE_PLAN_XY = (
    (-31.0, 14.0),
    (31.0, 14.0),
    (31.0, 70.0),
    (70.0, 110.0),
    (70.0, 125.0),
    (-70.0, 125.0),
    (-70.0, 110.0),
    (-31.0, 70.0),
)
SOURCE_Z_MM = (5.3, 18.3)
PROCESS_SETTINGS = {
    "sparse_infill_density": "100%",
    "sparse_infill_pattern": "zig-zag",
}


class ModifierError(RuntimeError):
    """The modifier could not be generated or validated."""


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _transform(
    matrix: Sequence[Sequence[float]],
    point: Sequence[float],
) -> tuple[float, float, float]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ModifierError("source_to_stl_matrix must be 4x4")
    if len(point) != 3:
        raise ModifierError("point must contain three coordinates")
    result = tuple(
        sum(float(matrix[row][column]) * float(point[column])
            for column in range(3))
        + float(matrix[row][3])
        for row in range(3)
    )
    if not all(math.isfinite(value) for value in result):
        raise ModifierError("transformed modifier coordinate is not finite")
    return result


def _normal(
    triangle: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    a, b, c = triangle
    ab = tuple(float(b[index]) - float(a[index]) for index in range(3))
    ac = tuple(float(c[index]) - float(a[index]) for index in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(value * value for value in cross))
    if length <= 1.0e-9:
        raise ModifierError("modifier contains a degenerate triangle")
    return tuple(value / length for value in cross)


def _source_triangles() -> tuple[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    ...,
]:
    count = len(SOURCE_PLAN_XY)
    lower = tuple((x, y, SOURCE_Z_MM[0]) for x, y in SOURCE_PLAN_XY)
    upper = tuple((x, y, SOURCE_Z_MM[1]) for x, y in SOURCE_PLAN_XY)
    triangles = []
    # CCW top, reversed bottom.
    for index in range(1, count - 1):
        triangles.append((lower[0], lower[index + 1], lower[index]))
        triangles.append((upper[0], upper[index], upper[index + 1]))
    for index in range(count):
        following = (index + 1) % count
        triangles.append((
            lower[index], lower[following], upper[following]))
        triangles.append((
            lower[index], upper[following], upper[index]))
    return tuple(triangles)


def _binary_stl(
    triangles: Iterable[Sequence[Sequence[float]]],
) -> bytes:
    records = []
    for triangle in triangles:
        if len(triangle) != 3:
            raise ModifierError("modifier triangle must have three vertices")
        normal = _normal(triangle)
        coordinates = [
            float(coordinate)
            for point in triangle
            for coordinate in point
        ]
        records.append(struct.pack(
            "<12fH", *normal, *coordinates, 0))
    header = b"Obi-Wan 01a PETG-GF bridge/root 100% modifier".ljust(
        80, b"\0")
    return header + struct.pack("<I", len(records)) + b"".join(records)


def _bounds(
    triangles: Iterable[Sequence[Sequence[float]]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    points = [point for triangle in triangles for point in triangle]
    return (
        tuple(min(float(point[axis]) for point in points)
              for axis in range(3)),
        tuple(max(float(point[axis]) for point in points)
              for axis in range(3)),
    )


def build_modifier(
    *,
    source_stl: Path = DEFAULT_STL,
    sidecar_path: Path = DEFAULT_SIDECAR,
    output_stl: Path = DEFAULT_OUTPUT,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, object]:
    try:
        sidecar = validate_print_sidecar(source_stl, sidecar_path)
    except FrontDownContractError as exc:
        raise ModifierError(str(exc)) from exc
    if sidecar.get("part") != PART:
        raise ModifierError(
            f"print sidecar names {sidecar.get('part')!r}, expected {PART!r}")
    matrix = sidecar["source_to_stl_matrix"]
    source_triangles = _source_triangles()
    print_triangles = tuple(
        tuple(_transform(matrix, point) for point in triangle)
        for triangle in source_triangles
    )
    payload = _binary_stl(print_triangles)
    _write_atomic(output_stl, payload)
    if len(payload) != 84 + len(print_triangles) * 50:
        raise ModifierError("binary STL length does not match triangle count")
    source_bounds = _bounds(source_triangles)
    print_bounds = _bounds(print_triangles)
    if abs(print_bounds[0][2]) > 2.0e-4:
        raise ModifierError(
            f"modifier front face is not on the bed: Z={print_bounds[0][2]}")
    try:
        relative_stl = output_stl.resolve().relative_to(PROJECT_ROOT)
        relative_source = source_stl.resolve().relative_to(PROJECT_ROOT)
        relative_sidecar = sidecar_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ModifierError(
            "modifier inputs and output must stay inside the project") from exc
    contract: dict[str, object] = {
        "schema_version": 1,
        "kind": "bambu_parameter_modifier",
        "generated_by": "scripts/build_obiwan_bridge_root_modifier.py",
        "artifact_match": ARTIFACT_MATCH,
        "role": "bridge_root_local_solid",
        "subtype": "modifier_part",
        "source_stl": str(relative_source),
        "source_stl_sha256": sha256_file(source_stl),
        "print_sidecar": str(relative_sidecar),
        "print_sidecar_sha256": sha256_file(sidecar_path),
        "modifier_stl": str(relative_stl),
        "modifier_stl_sha256": sha256_file(output_stl),
        "triangle_count": len(print_triangles),
        "source_plan_xy_mm": [list(point) for point in SOURCE_PLAN_XY],
        "source_z_mm": list(SOURCE_Z_MM),
        "source_bounds_mm": {
            "minimum": list(source_bounds[0]),
            "maximum": list(source_bounds[1]),
        },
        "print_bounds_mm": {
            "minimum": list(print_bounds[0]),
            "maximum": list(print_bounds[1]),
        },
        "process": PROCESS_SETTINGS,
        "policy": (
            "The volume changes infill only where it intersects printable "
            "01a material; ducts, bores, and acoustic openings remain void."
        ),
    }
    _write_atomic(
        contract_path,
        pretty_json_bytes(contract, allow_nan=False),
    )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stl", type=Path, default=DEFAULT_STL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    contract = build_modifier(
        source_stl=args.source_stl.expanduser().resolve(),
        sidecar_path=args.sidecar.expanduser().resolve(),
        output_stl=args.output.expanduser().resolve(),
        contract_path=args.contract.expanduser().resolve(),
    )
    print(json.dumps({
        "modifier_stl": contract["modifier_stl"],
        "modifier_stl_sha256": contract["modifier_stl_sha256"],
        "triangle_count": contract["triangle_count"],
        "process": contract["process"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ModifierError, OSError, ValueError) as exc:
        print(f"Obi-Wan structural modifier failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
