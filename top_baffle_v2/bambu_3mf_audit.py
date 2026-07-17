#!/usr/bin/env python3
"""Pure-Python, fail-closed audit of a single-object Bambu 3MF project.

The Bambu CLI may translate and rotate an arranged object around print Z even
when ``--orient 0`` is used.  Its ``result.json`` object bounding box is the
axis-aligned box of the *transformed source bounding box*, not necessarily the
tight bounds of the transformed mesh.  Comparing those dimensions directly to
the unrotated STL dimensions therefore confuses an allowed bed-plane rotation
with scaling.

This module binds the exported 3MF back to the staged STL instead.  It resolves
the single component chain, reconstructs the mesh in original STL coordinates,
compares the complete triangle soup, and then proves that the build transform
is a proper, unit-scale Rz plus XY translation.  No OCC, slicer, or third-party
geometry package is imported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import posixpath
import re
import struct
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET
import zipfile


Point3 = tuple[float, float, float]
Triangle = tuple[Point3, Point3, Point3]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


# Bambu serializes imported float32 mesh coordinates through decimal XML.  A
# 10-nanometre coordinate tolerance covers the observed round-trip maximum
# (4.55e-6 mm across small, 210-mm, and high-triangle-count production parts)
# while remaining far below any printable or CAD modelling tolerance.
DEFAULT_MESH_TOLERANCE_MM = 1.0e-5
DEFAULT_TRANSFORM_TOLERANCE = 2.0e-6
DEFAULT_BBOX_TOLERANCE_MM = 2.0e-4
DEFAULT_BED_Z_TOLERANCE_MM = 2.0e-2


class Bambu3MFAuditError(ValueError):
    """Raised whenever the archive cannot prove the required print contract."""


@dataclass(frozen=True)
class Bounds3D:
    minimum: Point3
    maximum: Point3

    @property
    def size(self) -> Point3:
        return tuple(
            self.maximum[index] - self.minimum[index]
            for index in range(3)
        )  # type: ignore[return-value]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "minimum_mm": list(self.minimum),
            "maximum_mm": list(self.maximum),
            "size_mm": list(self.size),
        }


@dataclass(frozen=True)
class RigidRzFacts:
    determinant: float
    orthonormal_max_error: float
    rz_degrees: float


@dataclass(frozen=True)
class Bambu3MFAudit:
    project_3mf: Path
    staged_stl: Path
    root_object_id: int
    component_depth: int
    triangle_count: int
    mesh_max_abs_error_mm: float
    source_bounds: Bounds3D
    transformed_actual_mesh_bounds: Bounds3D
    stl_to_bed_matrix: Matrix4
    rigid_rz: RigidRzFacts

    def as_record(self) -> dict[str, object]:
        return {
            "project_3mf": str(self.project_3mf),
            "staged_stl": str(self.staged_stl),
            "root_object_id": self.root_object_id,
            "component_depth": self.component_depth,
            "triangle_count": self.triangle_count,
            "mesh_max_abs_error_mm": self.mesh_max_abs_error_mm,
            "source_bounds": self.source_bounds.as_dict(),
            "transformed_actual_mesh_bounds": (
                self.transformed_actual_mesh_bounds.as_dict()),
            "stl_to_bed_matrix": [list(row) for row in self.stl_to_bed_matrix],
            "rigid_rz": {
                "determinant": self.rigid_rz.determinant,
                "orthonormal_max_error": (
                    self.rigid_rz.orthonormal_max_error),
                "rz_degrees": self.rigid_rz.rz_degrees,
            },
        }


IDENTITY4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


_ASCII_VERTEX_RE = re.compile(
    r"(?im)^\s*vertex\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise Bambu3MFAuditError(f"{label} is not finite")
    return value


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if _local_name(child.tag) == name]


def _attribute(element: ET.Element, name: str) -> str | None:
    for key, value in element.attrib.items():
        if _local_name(key) == name:
            return value
    return None


def _matrix_multiply(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column]
                for inner in range(4))
            for column in range(4)
        )
        for row in range(4)
    )  # type: ignore[return-value]


def transform_point(matrix: Matrix4, point: Sequence[float]) -> Point3:
    if len(point) != 3:
        raise Bambu3MFAuditError("point must have exactly three coordinates")
    result = tuple(
        sum(matrix[row][column] * float(point[column])
            for column in range(3)) + matrix[row][3]
        for row in range(3)
    )
    return tuple(_finite(value, "transformed coordinate")
                 for value in result)  # type: ignore[return-value]


def transform_vector(matrix: Matrix4, vector: Sequence[float]) -> Point3:
    if len(vector) != 3:
        raise Bambu3MFAuditError("vector must have exactly three coordinates")
    result = tuple(
        sum(matrix[row][column] * float(vector[column])
            for column in range(3))
        for row in range(3)
    )
    return tuple(_finite(value, "transformed vector coordinate")
                 for value in result)  # type: ignore[return-value]


def _parse_3mf_transform(value: str | None, label: str) -> Matrix4:
    """Parse a 3MF 3x4 affine into conventional column-vector form.

    3MF serializes ``m00 m01 m02 m10 ... m32`` for a row-vector affine.
    Transposing the linear portion gives the conventional matrix used by this
    module: ``bed_point = matrix @ [source_point, 1]``.
    """
    if value is None or not value.strip():
        return IDENTITY4
    fields = value.split()
    if len(fields) != 12:
        raise Bambu3MFAuditError(
            f"{label} must contain exactly 12 affine values")
    numbers = tuple(_finite(float(field), label) for field in fields)
    return (
        (numbers[0], numbers[3], numbers[6], numbers[9]),
        (numbers[1], numbers[4], numbers[7], numbers[10]),
        (numbers[2], numbers[5], numbers[8], numbers[11]),
        (0.0, 0.0, 0.0, 1.0),
    )


def read_stl_triangles(path: Path | str) -> tuple[Triangle, ...]:
    """Read a binary or ASCII STL as an exact triangle soup."""
    path = Path(path)
    data = path.read_bytes()
    triangles: list[Triangle] = []
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + triangle_count * 50:
            offset = 84
            for triangle_index in range(triangle_count):
                values = struct.unpack_from("<9f", data, offset + 12)
                points = []
                for vertex_index in range(3):
                    point = tuple(
                        _finite(values[vertex_index * 3 + axis],
                                f"{path}: triangle {triangle_index} vertex")
                        for axis in range(3)
                    )
                    points.append(point)
                triangles.append(tuple(points))  # type: ignore[arg-type]
                offset += 50
            if not triangles:
                raise Bambu3MFAuditError(f"{path} contains no STL triangles")
            return tuple(triangles)

    text = data.decode("ascii", errors="strict")
    vertices = [
        tuple(_finite(float(match.group(axis)), f"{path}: ASCII vertex")
              for axis in range(1, 4))
        for match in _ASCII_VERTEX_RE.finditer(text)
    ]
    if not vertices or len(vertices) % 3:
        raise Bambu3MFAuditError(
            f"{path} is not a recognized non-empty binary or ASCII STL")
    for index in range(0, len(vertices), 3):
        triangles.append(tuple(vertices[index:index + 3]))  # type: ignore[arg-type]
    return tuple(triangles)


def mesh_bounds(triangles: Sequence[Triangle]) -> Bounds3D:
    if not triangles:
        raise Bambu3MFAuditError("cannot measure an empty triangle soup")
    points = [point for triangle in triangles for point in triangle]
    minimum = tuple(min(point[axis] for point in points) for axis in range(3))
    maximum = tuple(max(point[axis] for point in points) for axis in range(3))
    return Bounds3D(minimum, maximum)  # type: ignore[arg-type]


def transform_mesh_bounds(
    triangles: Sequence[Triangle], matrix: Matrix4,
) -> Bounds3D:
    return mesh_bounds(tuple(
        tuple(transform_point(matrix, point) for point in triangle)
        for triangle in triangles
    ))


def transform_aabb(bounds: Bounds3D, matrix: Matrix4) -> Bounds3D:
    corners = [
        transform_point(matrix, (x, y, z))
        for x in (bounds.minimum[0], bounds.maximum[0])
        for y in (bounds.minimum[1], bounds.maximum[1])
        for z in (bounds.minimum[2], bounds.maximum[2])
    ]
    minimum = tuple(min(point[axis] for point in corners)
                    for axis in range(3))
    maximum = tuple(max(point[axis] for point in corners)
                    for axis in range(3))
    return Bounds3D(minimum, maximum)  # type: ignore[arg-type]


def expected_bambu_result_bbox(
    source_bounds: Bounds3D, stl_to_bed_matrix: Matrix4,
) -> dict[str, float]:
    """Return Bambu's conservative transformed-source-AABB representation."""
    bounds = transform_aabb(source_bounds, stl_to_bed_matrix)
    return {
        "x": bounds.minimum[0],
        "y": bounds.minimum[1],
        "z": bounds.minimum[2],
        "width": bounds.size[0],
        "depth": bounds.size[1],
        "height": bounds.size[2],
    }


def validate_result_bbox(
    result_bbox: Mapping[str, object],
    source_bounds: Bounds3D,
    stl_to_bed_matrix: Matrix4,
    *,
    tolerance_mm: float = DEFAULT_BBOX_TOLERANCE_MM,
) -> dict[str, float]:
    """Fail unless result.json matches the archived affine and source AABB."""
    tolerance_mm = _finite(tolerance_mm, "bbox tolerance")
    if tolerance_mm < 0.0:
        raise Bambu3MFAuditError("bbox tolerance must be non-negative")
    expected = expected_bambu_result_bbox(
        source_bounds, stl_to_bed_matrix)
    for key, expected_value in expected.items():
        if key not in result_bbox:
            raise Bambu3MFAuditError(f"result bbox is missing {key!r}")
        try:
            actual_value = _finite(float(result_bbox[key]),
                                   f"result bbox {key}")
        except (TypeError, ValueError) as error:
            raise Bambu3MFAuditError(
                f"result bbox {key!r} is not numeric") from error
        if abs(actual_value - expected_value) > tolerance_mm:
            raise Bambu3MFAuditError(
                f"result bbox {key}={actual_value:.9f} does not match "
                f"archived affine expectation {expected_value:.9f} "
                f"within {tolerance_mm:.6g} mm")
    return expected


def validate_bed_fit(
    actual_mesh_bounds: Bounds3D,
    machine_bounds_mm: Mapping[str, Sequence[float]],
    *,
    tolerance_mm: float = 1.0e-4,
) -> dict[str, tuple[float, float]]:
    """Check the tight transformed mesh bounds against rectangular limits.

    Start/end G-code, purge paths, and conservative result bounding boxes are
    deliberately not used.  The returned values are the low/high clearances.
    """
    tolerance_mm = _finite(tolerance_mm, "bed-fit tolerance")
    if tolerance_mm < 0.0:
        raise Bambu3MFAuditError("bed-fit tolerance must be non-negative")
    clearances: dict[str, tuple[float, float]] = {}
    for axis_index, axis in enumerate(("x", "y", "z")):
        values = machine_bounds_mm.get(axis)
        if values is None or len(values) != 2:
            raise Bambu3MFAuditError(
                f"machine bounds {axis!r} must contain [minimum, maximum]")
        low = _finite(values[0], f"machine {axis} minimum")
        high = _finite(values[1], f"machine {axis} maximum")
        if high <= low:
            raise Bambu3MFAuditError(
                f"machine {axis} bounds are not increasing")
        actual_low = actual_mesh_bounds.minimum[axis_index]
        actual_high = actual_mesh_bounds.maximum[axis_index]
        low_clearance = actual_low - low
        high_clearance = high - actual_high
        if low_clearance < -tolerance_mm or high_clearance < -tolerance_mm:
            raise Bambu3MFAuditError(
                f"transformed mesh exceeds machine {axis} bounds: "
                f"actual=[{actual_low:.6f}, {actual_high:.6f}], "
                f"limit=[{low:.6f}, {high:.6f}]")
        clearances[axis] = (low_clearance, high_clearance)
    return clearances


def validate_rigid_rz_affine(
    matrix: Matrix4,
    *,
    tolerance: float = DEFAULT_TRANSFORM_TOLERANCE,
) -> RigidRzFacts:
    """Prove a finite, proper, unit-scale Rz plus XY translation affine."""
    tolerance = _finite(tolerance, "transform tolerance")
    if tolerance < 0.0:
        raise Bambu3MFAuditError("transform tolerance must be non-negative")
    for row in range(4):
        for column in range(4):
            _finite(matrix[row][column], "build affine")
    for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        if abs(actual - expected) > tolerance:
            raise Bambu3MFAuditError(
                "build transform is not an affine 4x4 matrix")

    linear = tuple(tuple(matrix[row][column] for column in range(3))
                   for row in range(3))
    gram_errors = []
    for left in range(3):
        for right in range(3):
            value = sum(linear[row][left] * linear[row][right]
                        for row in range(3))
            gram_errors.append(abs(value - (1.0 if left == right else 0.0)))
    orthonormal_error = max(gram_errors)
    if orthonormal_error > tolerance:
        raise Bambu3MFAuditError(
            "build transform has scale or shear: "
            f"orthonormal residual {orthonormal_error:.9g}")

    determinant = (
        linear[0][0] * (linear[1][1] * linear[2][2]
                        - linear[1][2] * linear[2][1])
        - linear[0][1] * (linear[1][0] * linear[2][2]
                          - linear[1][2] * linear[2][0])
        + linear[0][2] * (linear[1][0] * linear[2][1]
                          - linear[1][1] * linear[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise Bambu3MFAuditError(
            "build transform is not a proper det+1 rotation: "
            f"det={determinant:.9g}")

    forbidden_tilt = (
        matrix[0][2], matrix[1][2], matrix[2][0], matrix[2][1])
    if any(abs(value) > tolerance for value in forbidden_tilt):
        raise Bambu3MFAuditError(
            "build transform tilts the front face; only Rz is permitted")
    if abs(matrix[2][2] - 1.0) > tolerance:
        raise Bambu3MFAuditError(
            "build transform flips or scales print Z")
    if abs(matrix[2][3]) > tolerance:
        raise Bambu3MFAuditError(
            f"build transform changes Z by {matrix[2][3]:.9g} mm")

    rz_degrees = math.degrees(math.atan2(matrix[1][0], matrix[0][0]))
    return RigidRzFacts(
        determinant=determinant,
        orthonormal_max_error=orthonormal_error,
        rz_degrees=rz_degrees,
    )


def _normalize_member(current_member: str, value: str) -> str:
    value = value.replace("\\", "/")
    if value.startswith("/"):
        normalized = posixpath.normpath(value.lstrip("/"))
    else:
        normalized = posixpath.normpath(
            posixpath.join(posixpath.dirname(current_member), value))
    path = PurePosixPath(normalized)
    if normalized in ("", ".") or path.is_absolute() or ".." in path.parts:
        raise Bambu3MFAuditError(
            f"unsafe or empty 3MF component path {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class _ModelDocument:
    member: str
    root: ET.Element
    objects: Mapping[int, ET.Element]


class _BambuPackage:
    def __init__(self, path: Path):
        self.path = path
        try:
            self._zip = zipfile.ZipFile(path, "r")
        except (OSError, zipfile.BadZipFile) as error:
            raise Bambu3MFAuditError(
                f"{path} is not a readable 3MF ZIP archive") from error
        self._members = set(self._zip.namelist())
        self._documents: dict[str, _ModelDocument] = {}

    def close(self) -> None:
        self._zip.close()

    def document(self, member: str) -> _ModelDocument:
        member = _normalize_member("", member)
        if member in self._documents:
            return self._documents[member]
        if member not in self._members:
            raise Bambu3MFAuditError(
                f"3MF archive is missing model member {member!r}")
        try:
            root = ET.fromstring(self._zip.read(member))
        except (ET.ParseError, KeyError, OSError) as error:
            raise Bambu3MFAuditError(
                f"cannot parse 3MF model member {member!r}") from error
        if _local_name(root.tag) != "model":
            raise Bambu3MFAuditError(f"{member!r} is not a 3MF model")
        unit = root.attrib.get("unit", "millimeter").strip().lower()
        if unit not in ("millimeter", "millimetre"):
            raise Bambu3MFAuditError(
                f"{member!r} uses unsupported unit {unit!r}; millimeters required")
        resources = _children(root, "resources")
        if len(resources) != 1:
            raise Bambu3MFAuditError(
                f"{member!r} must contain exactly one resources element")
        objects: dict[int, ET.Element] = {}
        for element in _children(resources[0], "object"):
            try:
                object_id = int(element.attrib["id"])
            except (KeyError, ValueError) as error:
                raise Bambu3MFAuditError(
                    f"{member!r} has an object with invalid id") from error
            if object_id in objects:
                raise Bambu3MFAuditError(
                    f"{member!r} repeats object id {object_id}")
            objects[object_id] = element
        document = _ModelDocument(member, root, objects)
        self._documents[member] = document
        return document

    def resolve_object(
        self,
        document: _ModelDocument,
        object_id: int,
        active: frozenset[tuple[str, int]] = frozenset(),
    ) -> tuple[tuple[Triangle, ...], int, int]:
        key = (document.member, object_id)
        if key in active:
            raise Bambu3MFAuditError("cyclic 3MF component graph")
        element = document.objects.get(object_id)
        if element is None:
            raise Bambu3MFAuditError(
                f"{document.member!r} has no object id {object_id}")
        meshes = _children(element, "mesh")
        component_groups = _children(element, "components")
        if len(meshes) == 1 and not component_groups:
            return self._parse_mesh(meshes[0], document.member), 0, 1
        if len(component_groups) != 1 or meshes:
            raise Bambu3MFAuditError(
                f"object {object_id} in {document.member!r} must contain "
                "exactly one mesh or one component group")
        components = _children(component_groups[0], "component")
        if len(components) != 1:
            raise Bambu3MFAuditError(
                f"object {object_id} in {document.member!r} must contain "
                "exactly one component")
        component = components[0]
        try:
            child_id = int(component.attrib["objectid"])
        except (KeyError, ValueError) as error:
            raise Bambu3MFAuditError("3MF component has invalid objectid") from error
        path_value = _attribute(component, "path")
        child_document = (
            self.document(_normalize_member(document.member, path_value))
            if path_value else document
        )
        child_triangles, child_depth, leaf_count = self.resolve_object(
            child_document, child_id, active | {key})
        transform = _parse_3mf_transform(
            component.attrib.get("transform"),
            f"component {object_id}->{child_id} transform")
        transformed = tuple(
            tuple(transform_point(transform, point) for point in triangle)
            for triangle in child_triangles
        )
        return transformed, child_depth + 1, leaf_count

    @staticmethod
    def _parse_mesh(mesh: ET.Element, member: str) -> tuple[Triangle, ...]:
        vertex_groups = _children(mesh, "vertices")
        triangle_groups = _children(mesh, "triangles")
        if len(vertex_groups) != 1 or len(triangle_groups) != 1:
            raise Bambu3MFAuditError(
                f"mesh in {member!r} needs exactly one vertices and triangles group")
        vertices: list[Point3] = []
        for index, vertex in enumerate(_children(vertex_groups[0], "vertex")):
            try:
                point = tuple(
                    _finite(float(vertex.attrib[axis]),
                            f"{member}: vertex {index} {axis}")
                    for axis in ("x", "y", "z")
                )
            except (KeyError, ValueError) as error:
                raise Bambu3MFAuditError(
                    f"mesh in {member!r} has an invalid vertex") from error
            vertices.append(point)  # type: ignore[arg-type]
        if not vertices:
            raise Bambu3MFAuditError(f"mesh in {member!r} has no vertices")
        triangles: list[Triangle] = []
        for index, triangle in enumerate(_children(triangle_groups[0], "triangle")):
            try:
                indices = tuple(int(triangle.attrib[key])
                                for key in ("v1", "v2", "v3"))
            except (KeyError, ValueError) as error:
                raise Bambu3MFAuditError(
                    f"mesh in {member!r} has an invalid triangle") from error
            if len(set(indices)) != 3:
                raise Bambu3MFAuditError(
                    f"mesh in {member!r} triangle {index} repeats a vertex")
            if any(vertex_index < 0 or vertex_index >= len(vertices)
                   for vertex_index in indices):
                raise Bambu3MFAuditError(
                    f"mesh in {member!r} triangle {index} has an invalid index")
            triangles.append(tuple(vertices[vertex_index]
                                   for vertex_index in indices))  # type: ignore[arg-type]
        if not triangles:
            raise Bambu3MFAuditError(f"mesh in {member!r} has no triangles")
        return tuple(triangles)


def _validate_mesh_equivalence(
    staged: Sequence[Triangle],
    reconstructed: Sequence[Triangle],
    *,
    tolerance_mm: float,
) -> float:
    if len(staged) != len(reconstructed):
        raise Bambu3MFAuditError(
            "3MF triangle count differs from staged STL: "
            f"{len(reconstructed)} != {len(staged)}")

    # Assign every reconstructed vertex to one and only one staged-STL vertex
    # within the strict coordinate tolerance.  A spatial hash avoids fragile
    # float sorting: a sub-tolerance change near zero must not reorder (0,0,0)
    # after (0,3,0), while two genuinely distinct source vertices within the
    # tolerance are intentionally rejected as ambiguous rather than merged.
    source_points = sorted(set(
        point for triangle in staged for point in triangle))
    point_ids = {point: index for index, point in enumerate(source_points)}
    staged_soup = Counter(
        tuple(sorted(point_ids[point] for point in triangle))
        for triangle in staged)
    max_error = 0.0
    mapped_cache: dict[Point3, int] = {}
    if tolerance_mm == 0.0:
        def map_point(point: Point3) -> int:
            try:
                return point_ids[point]
            except KeyError as error:
                raise Bambu3MFAuditError(
                    f"3MF mesh vertex {point!r} is absent from staged STL") from error
    else:
        cell = tolerance_mm
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for source_id, point in enumerate(source_points):
            key = tuple(math.floor(value / cell) for value in point)
            buckets.setdefault(key, []).append(source_id)  # type: ignore[arg-type]

        def map_point(point: Point3) -> int:
            nonlocal max_error
            cached = mapped_cache.get(point)
            if cached is not None:
                return cached
            key = tuple(math.floor(value / cell) for value in point)
            candidates: list[tuple[int, float]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for source_id in buckets.get(
                                (key[0] + dx, key[1] + dy, key[2] + dz), ()):
                            source = source_points[source_id]
                            error = max(abs(point[axis] - source[axis])
                                        for axis in range(3))
                            if error <= tolerance_mm:
                                candidates.append((source_id, error))
            if not candidates:
                raise Bambu3MFAuditError(
                    "3MF mesh differs from staged STL: vertex "
                    f"{point!r} has no match within {tolerance_mm:.9g} mm")
            candidate_ids = {source_id for source_id, _error in candidates}
            if len(candidate_ids) != 1:
                raise Bambu3MFAuditError(
                    "staged STL contains vertices too close for an unambiguous "
                    f"{tolerance_mm:.9g}-mm mesh comparison")
            source_id, error = candidates[0]
            max_error = max(max_error, error)
            mapped_cache[point] = source_id
            return source_id

    reconstructed_soup = Counter(
        tuple(sorted(map_point(point) for point in triangle))
        for triangle in reconstructed)
    if reconstructed_soup != staged_soup:
        missing = sum((staged_soup - reconstructed_soup).values())
        added = sum((reconstructed_soup - staged_soup).values())
        raise Bambu3MFAuditError(
            "3MF canonical triangle soup differs from staged STL: "
            f"missing={missing}, added={added}")
    return max_error


def audit_bambu_3mf(
    project_3mf: Path | str,
    staged_stl: Path | str,
    *,
    mesh_tolerance_mm: float = DEFAULT_MESH_TOLERANCE_MM,
    transform_tolerance: float = DEFAULT_TRANSFORM_TOLERANCE,
    bed_z_tolerance_mm: float = DEFAULT_BED_Z_TOLERANCE_MM,
) -> Bambu3MFAudit:
    """Audit one Bambu-exported, single-component 3MF against its staged STL."""
    project_path = Path(project_3mf)
    stl_path = Path(staged_stl)
    mesh_tolerance_mm = _finite(mesh_tolerance_mm, "mesh tolerance")
    bed_z_tolerance_mm = _finite(bed_z_tolerance_mm, "bed-Z tolerance")
    if mesh_tolerance_mm < 0.0 or bed_z_tolerance_mm < 0.0:
        raise Bambu3MFAuditError("audit tolerances must be non-negative")
    staged_triangles = read_stl_triangles(stl_path)
    source_bounds = mesh_bounds(staged_triangles)
    if abs(source_bounds.minimum[2]) > bed_z_tolerance_mm:
        raise Bambu3MFAuditError(
            "front-down staged STL does not sit at Z=0: "
            f"min Z={source_bounds.minimum[2]:.9g} mm")

    package = _BambuPackage(project_path)
    try:
        root_document = package.document("3D/3dmodel.model")
        builds = _children(root_document.root, "build")
        if len(builds) != 1:
            raise Bambu3MFAuditError(
                "root 3MF model must contain exactly one build element")
        items = _children(builds[0], "item")
        if len(items) != 1:
            raise Bambu3MFAuditError(
                "3MF must contain exactly one build item")
        item = items[0]
        printable = item.attrib.get("printable", "1").strip().lower()
        if printable in ("0", "false", "no"):
            raise Bambu3MFAuditError("the sole 3MF build item is not printable")
        if printable not in ("1", "true", "yes"):
            raise Bambu3MFAuditError(
                f"the build item has invalid printable={printable!r}")
        try:
            root_object_id = int(item.attrib["objectid"])
        except (KeyError, ValueError) as error:
            raise Bambu3MFAuditError("3MF build item has invalid objectid") from error
        reconstructed, component_depth, leaf_count = package.resolve_object(
            root_document, root_object_id)
        if component_depth < 1 or leaf_count != 1:
            raise Bambu3MFAuditError(
                "3MF build item must resolve through one component chain "
                "to exactly one object mesh")
        stl_to_bed = _parse_3mf_transform(
            item.attrib.get("transform"), "build item transform")
    finally:
        package.close()

    mesh_max_error = _validate_mesh_equivalence(
        staged_triangles, reconstructed,
        tolerance_mm=mesh_tolerance_mm)
    rigid_rz = validate_rigid_rz_affine(
        stl_to_bed, tolerance=transform_tolerance)
    transformed_bounds = transform_mesh_bounds(staged_triangles, stl_to_bed)
    if abs(transformed_bounds.minimum[2]) > bed_z_tolerance_mm:
        raise Bambu3MFAuditError(
            "archived build transform does not leave the front-down mesh on "
            f"the bed: min Z={transformed_bounds.minimum[2]:.9g} mm")

    return Bambu3MFAudit(
        project_3mf=project_path,
        staged_stl=stl_path,
        root_object_id=root_object_id,
        component_depth=component_depth,
        triangle_count=len(staged_triangles),
        mesh_max_abs_error_mm=mesh_max_error,
        source_bounds=source_bounds,
        transformed_actual_mesh_bounds=transformed_bounds,
        stl_to_bed_matrix=stl_to_bed,
        rigid_rz=rigid_rz,
    )


__all__ = [
    "Bambu3MFAudit",
    "Bambu3MFAuditError",
    "Bounds3D",
    "DEFAULT_BBOX_TOLERANCE_MM",
    "DEFAULT_BED_Z_TOLERANCE_MM",
    "DEFAULT_MESH_TOLERANCE_MM",
    "DEFAULT_TRANSFORM_TOLERANCE",
    "Matrix4",
    "RigidRzFacts",
    "audit_bambu_3mf",
    "expected_bambu_result_bbox",
    "mesh_bounds",
    "read_stl_triangles",
    "transform_aabb",
    "transform_mesh_bounds",
    "transform_point",
    "transform_vector",
    "validate_bed_fit",
    "validate_result_bbox",
    "validate_rigid_rz_affine",
]
