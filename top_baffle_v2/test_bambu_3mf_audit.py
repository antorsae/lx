#!/usr/bin/env python3
"""Pure synthetic regressions for the Bambu 3MF transform/mesh audit."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import zipfile

import pytest

from bambu_3mf_audit import (
    Bambu3MFAuditError,
    Matrix4,
    audit_bambu_3mf,
    expected_bambu_result_bbox,
    validate_bed_fit,
    validate_result_bbox,
)


Triangle = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


TRIANGLES: tuple[Triangle, ...] = (
    ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.4, 1.1, 1.2), (2.0, 0.0, 0.0)),
    ((2.0, 0.0, 0.0), (0.4, 1.1, 1.2), (0.0, 3.0, 0.0)),
    ((0.0, 3.0, 0.0), (0.4, 1.1, 1.2), (0.0, 0.0, 0.0)),
)


def _matrix(
    linear: tuple[tuple[float, float, float], ...],
    translation: tuple[float, float, float] = (10.0, 20.0, 0.0),
) -> Matrix4:
    return (
        (*linear[0], translation[0]),
        (*linear[1], translation[1]),
        (*linear[2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rz(degrees: float, translation=(10.0, 20.0, 0.0)) -> Matrix4:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return _matrix(
        ((cosine, -sine, 0.0),
         (sine, cosine, 0.0),
         (0.0, 0.0, 1.0)),
        translation,
    )


def _serialize_3mf_matrix(matrix: Matrix4) -> str:
    # Inverse of bambu_3mf_audit._parse_3mf_transform: 3MF stores a
    # row-vector 3x4 matrix as m00 m01 ... m32.
    values = (
        matrix[0][0], matrix[1][0], matrix[2][0],
        matrix[0][1], matrix[1][1], matrix[2][1],
        matrix[0][2], matrix[1][2], matrix[2][2],
        matrix[0][3], matrix[1][3], matrix[2][3],
    )
    return " ".join(f"{value:.12g}" for value in values)


def _write_stl(path: Path, *, ascii_stl: bool) -> None:
    if ascii_stl:
        lines = ["solid synthetic"]
        for triangle in TRIANGLES:
            lines.extend(("  facet normal 0 0 0", "    outer loop"))
            lines.extend(
                f"      vertex {x:.9g} {y:.9g} {z:.9g}"
                for x, y, z in triangle)
            lines.extend(("    endloop", "  endfacet"))
        lines.append("endsolid synthetic")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return
    payload = bytearray(b"synthetic fixture".ljust(80, b"\0"))
    payload.extend(struct.pack("<I", len(TRIANGLES)))
    for triangle in TRIANGLES:
        coordinates = [coordinate for point in triangle for coordinate in point]
        payload.extend(struct.pack("<12fH", 0.0, 0.0, 0.0,
                                   *coordinates, 0))
    path.write_bytes(bytes(payload))


def _write_3mf(
    path: Path,
    *,
    build_matrix: Matrix4,
    vertex_delta_mm: float = 0.0,
    multiple_items: bool = False,
    multiple_components: bool = False,
    omit_last_triangle: bool = False,
    reorder_triangle_soup: bool = False,
) -> None:
    offset = (1.0, 1.5, 0.6)
    source_triangles = list(TRIANGLES)
    if reorder_triangle_soup:
        source_triangles = [tuple(reversed(triangle))
                            for triangle in reversed(source_triangles)]

    vertices: list[tuple[float, float, float]] = []
    vertex_indices: dict[tuple[float, float, float], int] = {}
    indexed_triangles: list[tuple[int, int, int]] = []
    for triangle in source_triangles:
        indices = []
        for source_point in triangle:
            point = tuple(source_point[axis] - offset[axis]
                          for axis in range(3))
            if point not in vertex_indices:
                vertex_indices[point] = len(vertices)
                vertices.append(point)
            indices.append(vertex_indices[point])
        indexed_triangles.append(tuple(indices))  # type: ignore[arg-type]
    if vertex_delta_mm:
        first = list(vertices[0])
        first[0] += vertex_delta_mm
        vertices[0] = tuple(first)  # type: ignore[assignment]
    if omit_last_triangle:
        indexed_triangles.pop()

    vertex_xml = "\n".join(
        f'     <vertex x="{x:.12g}" y="{y:.12g}" z="{z:.12g}"/>'
        for x, y, z in vertices)
    triangle_xml = "\n".join(
        f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>'
        for a, b, c in indexed_triangles)
    object_model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources>
  <object id="1" type="model">
   <mesh>
    <vertices>
{vertex_xml}
    </vertices>
    <triangles>
{triangle_xml}
    </triangles>
   </mesh>
  </object>
 </resources>
</model>
'''
    component_matrix = _matrix(
        ((1.0, 0.0, 0.0),
         (0.0, 1.0, 0.0),
         (0.0, 0.0, 1.0)),
        offset,
    )
    component = (
        '    <component p:path="/3D/Objects/object_1.model" objectid="1" '
        f'transform="{_serialize_3mf_matrix(component_matrix)}"/>')
    components = component + ("\n" + component if multiple_components else "")
    item = (
        f'  <item objectid="2" transform="{_serialize_3mf_matrix(build_matrix)}" '
        'printable="1"/>')
    items = item + ("\n" + item if multiple_items else "")
    root_model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
 xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">
 <resources>
  <object id="2" type="model">
   <components>
{components}
   </components>
  </object>
 </resources>
 <build>
{items}
 </build>
</model>
'''
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("3D/3dmodel.model", root_model)
        archive.writestr("3D/Objects/object_1.model", object_model)


def _fixture(
    tmp_path: Path,
    *,
    ascii_stl: bool = False,
    build_matrix: Matrix4 | None = None,
    **project_options,
) -> tuple[Path, Path]:
    stl = tmp_path / ("part_ascii.stl" if ascii_stl else "part.stl")
    project = tmp_path / "part.3mf"
    _write_stl(stl, ascii_stl=ascii_stl)
    _write_3mf(project, build_matrix=build_matrix or _rz(30.0),
               **project_options)
    return project, stl


@pytest.mark.parametrize("ascii_stl", [False, True])
def test_accepts_exact_single_object_rz_and_binds_mesh(
    tmp_path: Path, ascii_stl: bool,
) -> None:
    project, stl = _fixture(
        tmp_path, ascii_stl=ascii_stl, reorder_triangle_soup=True)
    audit = audit_bambu_3mf(project, stl)

    assert audit.triangle_count == len(TRIANGLES)
    assert audit.component_depth == 1
    assert audit.mesh_max_abs_error_mm < 1.0e-6
    assert audit.rigid_rz.determinant == pytest.approx(1.0, abs=2.0e-9)
    assert audit.rigid_rz.rz_degrees == pytest.approx(30.0, abs=1.0e-8)
    assert audit.stl_to_bed_matrix[2] == pytest.approx((0.0, 0.0, 1.0, 0.0))
    assert audit.transformed_actual_mesh_bounds.minimum[2] == pytest.approx(0.0)

    bbox = expected_bambu_result_bbox(
        audit.source_bounds, audit.stl_to_bed_matrix)
    assert validate_result_bbox(
        bbox, audit.source_bounds, audit.stl_to_bed_matrix) == bbox
    clearances = validate_bed_fit(
        audit.transformed_actual_mesh_bounds,
        {"x": [0.0, 256.0], "y": [0.0, 256.0], "z": [0.0, 256.0]},
    )
    assert all(low >= 0.0 and high >= 0.0
               for low, high in clearances.values())


def test_accepts_only_sub_tolerance_mesh_roundoff(tmp_path: Path) -> None:
    project, stl = _fixture(tmp_path, vertex_delta_mm=5.0e-7)
    audit = audit_bambu_3mf(project, stl)
    assert audit.mesh_max_abs_error_mm == pytest.approx(5.0e-7, abs=1.0e-7)


@pytest.mark.parametrize(
    "matrix",
    [
        _matrix(((1.01, 0.0, 0.0), (0.0, 1.01, 0.0), (0.0, 0.0, 1.0))),
        _matrix(((1.0, 0.02, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        _matrix(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        _matrix(((1.0, 0.0, 0.0),
                 (0.0, math.cos(math.radians(4.0)),
                  -math.sin(math.radians(4.0))),
                 (0.0, math.sin(math.radians(4.0)),
                  math.cos(math.radians(4.0))))),
        _rz(0.0, translation=(10.0, 20.0, 0.01)),
    ],
    ids=("scale", "shear", "reflection", "tilt", "z-translation"),
)
def test_rejects_non_rz_or_non_unit_build_transforms(
    tmp_path: Path, matrix: Matrix4,
) -> None:
    project, stl = _fixture(tmp_path, build_matrix=matrix)
    with pytest.raises(Bambu3MFAuditError):
        audit_bambu_3mf(project, stl)


@pytest.mark.parametrize("delta", [5.0e-5, -8.0e-5])
def test_rejects_changed_mesh(tmp_path: Path, delta: float) -> None:
    project, stl = _fixture(tmp_path, vertex_delta_mm=delta)
    with pytest.raises(Bambu3MFAuditError, match="mesh differs"):
        audit_bambu_3mf(project, stl)


def test_rejects_changed_triangle_count(tmp_path: Path) -> None:
    project, stl = _fixture(tmp_path, omit_last_triangle=True)
    with pytest.raises(Bambu3MFAuditError, match="triangle count"):
        audit_bambu_3mf(project, stl)


def test_rejects_multiple_build_items(tmp_path: Path) -> None:
    project, stl = _fixture(tmp_path, multiple_items=True)
    with pytest.raises(Bambu3MFAuditError, match="exactly one build item"):
        audit_bambu_3mf(project, stl)


def test_rejects_multiple_components(tmp_path: Path) -> None:
    project, stl = _fixture(tmp_path, multiple_components=True)
    with pytest.raises(Bambu3MFAuditError, match="exactly one component"):
        audit_bambu_3mf(project, stl)


def test_result_bbox_and_actual_mesh_bed_fit_fail_closed(tmp_path: Path) -> None:
    project, stl = _fixture(tmp_path)
    audit = audit_bambu_3mf(project, stl)
    bbox = expected_bambu_result_bbox(
        audit.source_bounds, audit.stl_to_bed_matrix)
    bbox["width"] += 0.01
    with pytest.raises(Bambu3MFAuditError, match="does not match"):
        validate_result_bbox(
            bbox, audit.source_bounds, audit.stl_to_bed_matrix)

    with pytest.raises(Bambu3MFAuditError, match="exceeds machine x bounds"):
        validate_bed_fit(
            audit.transformed_actual_mesh_bounds,
            {"x": [0.0, 10.1], "y": [0.0, 256.0], "z": [0.0, 256.0]},
        )
