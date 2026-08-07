"""Scoped Bambu 3MF audit for an intentional translated multi-STL bundle.

The canonical captive-magnet auditor remains single-normal-part and
hash-stable.  This module composes its pure parsing/equivalence primitives for
the separately packaged Obi-Wan plate without broadening that release policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Mapping, Sequence

from bambu_3mf_audit import (
    Bambu3MFAuditError,
    Bounds3D,
    DEFAULT_BED_Z_TOLERANCE_MM,
    DEFAULT_MESH_TOLERANCE_MM,
    DEFAULT_TRANSFORM_TOLERANCE,
    Matrix4,
    RigidRzFacts,
    Triangle,
    _BambuPackage,
    _bambu_model_part_records,
    _children,
    _finite,
    _parse_3mf_transform,
    _resolve_root_component,
    _validate_mesh_equivalence,
    mesh_bounds,
    read_stl_triangles,
    transform_mesh_bounds,
    validate_rigid_rz_affine,
)


@dataclass(frozen=True)
class BambuComposite3MFAudit:
    """Evidence for one intentionally multi-volume printable assembly."""

    project_3mf: Path
    staged_stl: Path
    root_object_id: int
    triangle_count: int
    mesh_max_abs_error_mm: float
    source_bounds: Bounds3D
    transformed_actual_mesh_bounds: Bounds3D
    stl_to_bed_matrix: Matrix4
    rigid_rz: RigidRzFacts
    normal_part_triangle_counts: tuple[int, ...]
    support_blocker_triangle_counts: tuple[int, ...]
    normal_part_names: tuple[str, ...]
    support_blocker_names: tuple[str, ...]
    parameter_modifier_triangle_counts: tuple[int, ...] = ()
    parameter_modifier_names: tuple[str, ...] = ()
    parameter_modifier_settings: tuple[
        tuple[tuple[str, str], ...], ...
    ] = ()

    @property
    def support_blocker_count(self) -> int:
        return len(self.support_blocker_names)

    @property
    def parameter_modifier_count(self) -> int:
        return len(self.parameter_modifier_names)

    def as_record(self) -> dict[str, object]:
        return {
            "project_3mf": str(self.project_3mf),
            "staged_stl": str(self.staged_stl),
            "root_object_id": self.root_object_id,
            "triangle_count": self.triangle_count,
            "mesh_max_abs_error_mm": self.mesh_max_abs_error_mm,
            "source_bounds": self.source_bounds.as_dict(),
            "transformed_actual_mesh_bounds": (
                self.transformed_actual_mesh_bounds.as_dict()),
            "stl_to_bed_matrix": [
                list(row) for row in self.stl_to_bed_matrix
            ],
            "rigid_rz": {
                "determinant": self.rigid_rz.determinant,
                "orthonormal_max_error": (
                    self.rigid_rz.orthonormal_max_error),
                "rz_degrees": self.rigid_rz.rz_degrees,
            },
            "normal_part_triangle_counts": list(
                self.normal_part_triangle_counts),
            "support_blocker_triangle_counts": list(
                self.support_blocker_triangle_counts),
            "normal_part_names": list(self.normal_part_names),
            "support_blocker_names": list(self.support_blocker_names),
            "support_blocker_count": self.support_blocker_count,
            "parameter_modifier_triangle_counts": list(
                self.parameter_modifier_triangle_counts),
            "parameter_modifier_names": list(
                self.parameter_modifier_names),
            "parameter_modifier_settings": [
                dict(settings)
                for settings in self.parameter_modifier_settings
            ],
            "parameter_modifier_count": self.parameter_modifier_count,
        }


def translated_float32_triangles(
    triangles: Sequence[Triangle],
    translation: Sequence[float],
) -> tuple[Triangle, ...]:
    """Apply a locked float32 translation exactly as the binary STL writer."""
    if len(translation) != 3:
        raise Bambu3MFAuditError(
            "a composite-part translation must contain exactly three values")
    as_float32 = lambda value: struct.unpack(
        "<f", struct.pack("<f", value))[0]
    dx, dy, dz = (
        as_float32(_finite(float(value), "composite-part translation"))
        for value in translation
    )
    return tuple(
        tuple((as_float32(as_float32(point[0]) + dx),
               as_float32(as_float32(point[1]) + dy),
               as_float32(as_float32(point[2]) + dz))
              for point in triangle)
        for triangle in triangles
    )


def validate_triangle_soup_equivalence(
    expected: Sequence[Triangle],
    actual: Sequence[Triangle],
    *,
    tolerance_mm: float,
) -> float:
    """Expose the canonical exact triangle-soup comparator for plate sources."""
    return _validate_mesh_equivalence(
        expected, actual, tolerance_mm=tolerance_mm)


def audit_bambu_composite_3mf(
    project_3mf: Path | str,
    staged_stl: Path | str,
    *,
    normal_part_stls: Sequence[
        tuple[Path | str, Sequence[float]]
    ],
    support_blocker_stls: Sequence[
        tuple[Path | str, Sequence[float]]
    ] = (),
    parameter_modifier_stls: Sequence[
        tuple[
            Path | str,
            Sequence[float],
            Mapping[str, object],
        ]
    ] = (),
    mesh_tolerance_mm: float = DEFAULT_MESH_TOLERANCE_MM,
    transform_tolerance: float = DEFAULT_TRANSFORM_TOLERANCE,
    bed_z_tolerance_mm: float = DEFAULT_BED_Z_TOLERANCE_MM,
) -> BambuComposite3MFAudit:
    """Audit one-object 3MF assembly against an exact translated STL bundle."""
    project_path = Path(project_3mf)
    stl_path = Path(staged_stl)
    if not normal_part_stls:
        raise Bambu3MFAuditError(
            "a composite 3MF audit requires at least one normal part")
    mesh_tolerance_mm = _finite(mesh_tolerance_mm, "mesh tolerance")
    bed_z_tolerance_mm = _finite(bed_z_tolerance_mm, "bed-Z tolerance")
    if mesh_tolerance_mm < 0.0 or bed_z_tolerance_mm < 0.0:
        raise Bambu3MFAuditError("audit tolerances must be non-negative")

    normal_records = tuple(
        (Path(path), tuple(float(value) for value in translation))
        for path, translation in normal_part_stls
    )
    blocker_records = tuple(
        (Path(path), tuple(float(value) for value in translation))
        for path, translation in support_blocker_stls
    )
    modifier_records = tuple(
        (
            Path(path),
            tuple(float(value) for value in translation),
            {str(key): str(value) for key, value in settings.items()},
        )
        for path, translation, settings in parameter_modifier_stls
    )
    normal_names = tuple(path.name for path, _translation in normal_records)
    blocker_names = tuple(path.name for path, _translation in blocker_records)
    modifier_names = tuple(
        path.name for path, _translation, _settings in modifier_records)
    if len(normal_names) != len(set(normal_names)):
        raise Bambu3MFAuditError(
            "composite normal-part STL basenames must be unique")
    if len(blocker_names) != len(set(blocker_names)):
        raise Bambu3MFAuditError(
            "composite support-blocker STL basenames must be unique")
    if len(modifier_names) != len(set(modifier_names)):
        raise Bambu3MFAuditError(
            "composite parameter-modifier STL basenames must be unique")

    staged_triangles = read_stl_triangles(stl_path)
    source_bounds = mesh_bounds(staged_triangles)
    if abs(source_bounds.minimum[2]) > bed_z_tolerance_mm:
        raise Bambu3MFAuditError(
            "front-down composite STL does not sit at Z=0: "
            f"min Z={source_bounds.minimum[2]:.9g} mm")

    expected_normals = {
        path.name: translated_float32_triangles(
            read_stl_triangles(path), translation)
        for path, translation in normal_records
    }
    expected_blockers = {
        path.name: translated_float32_triangles(
            read_stl_triangles(path), translation)
        for path, translation in blocker_records
    }
    expected_modifiers = {
        path.name: translated_float32_triangles(
            read_stl_triangles(path), translation)
        for path, translation, _settings in modifier_records
    }
    expected_combined = tuple(
        triangle
        for name in normal_names
        for triangle in expected_normals[name]
    )
    staged_error = validate_triangle_soup_equivalence(
        expected_combined, staged_triangles,
        tolerance_mm=mesh_tolerance_mm)

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
                "composite 3MF must contain exactly one build item")
        item = items[0]
        printable = item.attrib.get("printable", "1").strip().lower()
        if printable not in ("1", "true", "yes"):
            raise Bambu3MFAuditError(
                "the composite 3MF build item is not explicitly printable")
        try:
            root_object_id = int(item.attrib["objectid"])
        except (KeyError, ValueError) as error:
            raise Bambu3MFAuditError(
                "composite 3MF build item has invalid objectid") from error

        part_records = _bambu_model_part_records(
            package, root_object_id)
        expected_subtypes = {"normal_part"}
        if blocker_records:
            expected_subtypes.add("support_blocker")
        if modifier_records:
            expected_subtypes.add("modifier_part")
        if set(part_records) != expected_subtypes:
            raise Bambu3MFAuditError(
                "composite 3MF part subtypes differ from its declared "
                "normal/modifier inventory")
        actual_normal_names = tuple(
            Path(source).name
            for _part_id, source, _metadata
            in part_records["normal_part"]
        )
        if Counter(actual_normal_names) != Counter(normal_names):
            raise Bambu3MFAuditError(
                "composite 3MF normal_part inventory differs from its sources")
        actual_blocker_names = tuple(
            Path(source).name
            for _part_id, source, _metadata
            in part_records.get("support_blocker", ())
        )
        if Counter(actual_blocker_names) != Counter(blocker_names):
            raise Bambu3MFAuditError(
                "composite 3MF support_blocker inventory differs from sources")
        actual_modifier_names = tuple(
            Path(source).name
            for _part_id, source, _metadata
            in part_records.get("modifier_part", ())
        )
        if Counter(actual_modifier_names) != Counter(modifier_names):
            raise Bambu3MFAuditError(
                "composite 3MF modifier_part inventory differs from sources")

        reconstructed_normals: list[Triangle] = []
        part_errors = [staged_error]
        normal_triangle_counts = []
        for part_id, source, _metadata in part_records["normal_part"]:
            name = Path(source).name
            reconstructed, _depth = _resolve_root_component(
                package, root_document, root_object_id, part_id)
            part_errors.append(validate_triangle_soup_equivalence(
                expected_normals[name], reconstructed,
                tolerance_mm=mesh_tolerance_mm))
            reconstructed_normals.extend(reconstructed)
            normal_triangle_counts.append(len(expected_normals[name]))
        part_errors.append(validate_triangle_soup_equivalence(
            staged_triangles, reconstructed_normals,
            tolerance_mm=mesh_tolerance_mm))

        blocker_triangle_counts = []
        for part_id, source, _metadata in part_records.get(
                "support_blocker", ()):
            name = Path(source).name
            reconstructed, _depth = _resolve_root_component(
                package, root_document, root_object_id, part_id)
            part_errors.append(validate_triangle_soup_equivalence(
                expected_blockers[name], reconstructed,
                tolerance_mm=mesh_tolerance_mm))
            blocker_triangle_counts.append(len(expected_blockers[name]))
        modifier_expected_by_name = {
            path.name: settings
            for path, _translation, settings in modifier_records
        }
        modifier_triangle_counts = []
        modifier_settings = []
        for part_id, source, metadata in part_records.get(
                "modifier_part", ()):
            name = Path(source).name
            expected_settings = modifier_expected_by_name[name]
            for key, expected in expected_settings.items():
                if metadata.get(key) != expected:
                    raise Bambu3MFAuditError(
                        f"composite modifier_part {name} has "
                        f"{key}={metadata.get(key)!r}, expected "
                        f"{expected!r}")
            reconstructed, _depth = _resolve_root_component(
                package, root_document, root_object_id, part_id)
            part_errors.append(validate_triangle_soup_equivalence(
                expected_modifiers[name], reconstructed,
                tolerance_mm=mesh_tolerance_mm))
            modifier_triangle_counts.append(
                len(expected_modifiers[name]))
            modifier_settings.append(
                tuple(sorted(expected_settings.items())))
        stl_to_bed = _parse_3mf_transform(
            item.attrib.get("transform"), "composite build item transform")
    finally:
        package.close()

    rigid_rz = validate_rigid_rz_affine(
        stl_to_bed, tolerance=transform_tolerance)
    transformed_bounds = transform_mesh_bounds(staged_triangles, stl_to_bed)
    if abs(transformed_bounds.minimum[2]) > bed_z_tolerance_mm:
        raise Bambu3MFAuditError(
            "archived composite transform does not leave the plate bundle "
            f"on the bed: min Z={transformed_bounds.minimum[2]:.9g} mm")
    return BambuComposite3MFAudit(
        project_3mf=project_path,
        staged_stl=stl_path,
        root_object_id=root_object_id,
        triangle_count=len(staged_triangles),
        mesh_max_abs_error_mm=max(part_errors),
        source_bounds=source_bounds,
        transformed_actual_mesh_bounds=transformed_bounds,
        stl_to_bed_matrix=stl_to_bed,
        rigid_rz=rigid_rz,
        normal_part_triangle_counts=tuple(normal_triangle_counts),
        support_blocker_triangle_counts=tuple(blocker_triangle_counts),
        normal_part_names=actual_normal_names,
        support_blocker_names=actual_blocker_names,
        parameter_modifier_triangle_counts=tuple(
            modifier_triangle_counts),
        parameter_modifier_names=actual_modifier_names,
        parameter_modifier_settings=tuple(modifier_settings),
    )


__all__ = [
    "BambuComposite3MFAudit",
    "audit_bambu_composite_3mf",
    "translated_float32_triangles",
    "validate_triangle_soup_equivalence",
]
