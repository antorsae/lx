#!/usr/bin/env python3
"""Export one STEP-first Stock/Slim ``vase_TEBM35C10-4`` artifact set.

The authoritative BREP and STEP remain in installed/source coordinates.  The
STL alone receives the release-wide X180 front-face-down transform.  A small,
isolated one-artifact captive-magnet catalog binds that exact STL and its four
station datums for the local-only Bambu slicing target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))

from build123d import (
    Pos,
    Rot,
    export_brep,
    export_step,
    export_stl,
    import_brep,
)

from export_piece_stls import (
    OBIWAN_MESH_ANGULAR_TOLERANCE,
    OBIWAN_MESH_TOLERANCE_MM,
    _canonicalize_transform_zeros,
    _remove_collapsed_apex_facets,
    _strict_mesh_facts,
    _validate_binary_stl,
    _write_print_transform_sidecar,
)
from export_steps import FIXED_TIMESTAMP, validate_step_transaction
from lx521_baffle.io import pretty_json_bytes, sha256_file
from lx521_baffle.magnets import DEFAULT_SPEC, NOMINAL_PAIRED_FACE_SEPARATION_MM
from lx521_baffle.print_contract import (
    RELEASE_ACOUSTIC_PRINT_CONTRACT,
    validate_print_sidecar,
)
from lx521_baffle.proud.vase_tebm35c10_4 import (
    PART_NAME,
    T_MAGNET_FACE_X_MM,
    VaseTEBMProfile,
    build_model,
    design_facts,
    vase_profile,
)


DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "build" / PART_NAME
CATALOG_SCHEMA = PROJECT_ROOT / "captive_magnet_release_catalog.schema.json"
CATALOG_SCHEMA_VERSION = 1
ARTIFACT_STATE = "shared"
SOURCE_REVISION_ENV = "LX_CAD_SOURCE_SHA256"

SOURCE_FILES = (
    "src/lx521_baffle/base.py",
    "src/lx521_baffle/cables.py",
    "src/lx521_baffle/geom.py",
    "src/lx521_baffle/io.py",
    "src/lx521_baffle/magnet_contract.py",
    "src/lx521_baffle/magnets.py",
    "src/lx521_baffle/print_contract.py",
    "src/lx521_baffle/stl_export.py",
    "src/lx521_baffle/proud/b.py",
    "src/lx521_baffle/proud/b2.py",
    "src/lx521_baffle/proud/b2_split.py",
    "src/lx521_baffle/proud/vase_tebm35c10_4.py",
    "scripts/check_manifold.py",
    "scripts/export_piece_stls.py",
    "scripts/export_steps.py",
    "scripts/export_vase_tebm35c10_4.py",
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(pretty_json_bytes(payload, allow_nan=False))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.parent.resolve())


def _source_bindings(output: Path) -> tuple[list[str], dict[str, str]]:
    paths = tuple(PROJECT_ROOT / relative for relative in SOURCE_FILES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing source provenance: {missing}")
    values = [_relative(path, output) for path in paths]
    return values, {
        value: sha256_file(path) for value, path in zip(values, paths, strict=True)
    }


def _source_revision(source_hashes: Mapping[str, str]) -> str:
    explicit = os.environ.get(SOURCE_REVISION_ENV, "").strip()
    if explicit:
        if re.fullmatch(r"[0-9a-f]{64}", explicit) is None:
            raise RuntimeError(
                f"{SOURCE_REVISION_ENV} must be a 64-hex digest")
        return explicit
    encoded = json.dumps(
        dict(sorted(source_hashes.items())),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transform_point(
    matrix: Sequence[Sequence[float]], point: Sequence[float],
) -> list[float]:
    return [
        sum(float(matrix[row][column]) * float(point[column])
            for column in range(3)) + float(matrix[row][3])
        for row in range(3)
    ]


def _transform_vector(
    matrix: Sequence[Sequence[float]], vector: Sequence[float],
) -> list[float]:
    return [
        sum(float(matrix[row][column]) * float(vector[column])
            for column in range(3))
        for row in range(3)
    ]


def _site_record(tools, matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    record = dict(tools.facts())
    record.update({
        "installed_marked_pole_axis_xyz": list(tools.pair_axis_xyz),
        "polarity_instruction": (
            "marked/N pole points OUT from the vase along "
            "installed_marked_pole_axis_xyz; verify the future mating "
            "piece uses the opposite interface-facing pole before burial"
        ),
        "magnet_count": 1,
        "structural_load_credit_n": 0.0,
        "interface_profile": "standard_straight",
        "carrier_cavity_face_inset_mm": 0.0,
        "outer_surface_face_xy_mm": list(tools.interface_datum_xyz[:2]),
        "paired_magnet_face_separation_mm": round(
            NOMINAL_PAIRED_FACE_SEPARATION_MM, 9),
    })
    point_fields = (
        "actual_face_xyz_mm",
        "cavity_center_xyz_mm",
        "seated_magnet_center_xyz_mm",
    )
    vector_fields = (
        "marked_pole_axis_xyz",
        "insertion_direction_xyz",
        "material_inward_xyz",
    )
    record["print_space"] = {
        **{key: _transform_point(matrix, record[key])
           for key in point_fields},
        **{key: _transform_vector(matrix, record[key])
           for key in vector_fields},
    }
    return record


def _validate_native_round_trip(path: Path, *, expected_volume: float) -> None:
    loaded = import_brep(str(path))
    solids = list(loaded.solids())
    if not loaded.is_valid or len(solids) != 1:
        raise RuntimeError(f"BREP round trip is not one valid solid: {path}")
    if not math.isclose(
        loaded.volume, expected_volume, rel_tol=1.0e-10, abs_tol=1.0e-5,
    ):
        raise RuntimeError(
            f"BREP volume changed on round trip: {loaded.volume} vs "
            f"{expected_volume}")


def _export_native(shape, *, brep: Path, step: Path) -> None:
    for path, suffix in ((brep, ".brep"), (step, ".step")):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.stem}.{os.getpid()}.tmp{suffix}")
        try:
            if suffix == ".brep":
                export_brep(shape, str(temporary))
                if temporary.stat().st_size < 1024:
                    raise RuntimeError(f"temporary BREP is truncated: {path}")
                _validate_native_round_trip(
                    temporary, expected_volume=float(shape.volume))
            else:
                export_step(shape, str(temporary), timestamp=FIXED_TIMESTAMP)
                validate_step_transaction(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _export_print_stl(
    shape,
    stl: Path,
    profile: VaseTEBMProfile,
) -> tuple[dict[str, Any], object]:
    oriented = Rot(X=180.0) * shape
    oriented_bbox = oriented.bounding_box()
    size = oriented_bbox.size
    if max(size.X, size.Y, size.Z) > 256.0 + 1.0e-6:
        raise RuntimeError(
            f"{PART_NAME} exceeds the P2S envelope: "
            f"{size.X:.3f} x {size.Y:.3f} x {size.Z:.3f} mm")
    moved = Pos(
        -oriented_bbox.min.X,
        -oriented_bbox.min.Y,
        -oriented_bbox.min.Z,
    ) * oriented
    stl.parent.mkdir(parents=True, exist_ok=True)
    temporary = stl.with_name(f".{stl.stem}.{os.getpid()}.tmp.stl")
    try:
        export_stl(
            moved,
            str(temporary),
            tolerance=OBIWAN_MESH_TOLERANCE_MM,
            angular_tolerance=OBIWAN_MESH_ANGULAR_TOLERANCE,
        )
        _validate_binary_stl(temporary)
        canonicalized = _canonicalize_transform_zeros(temporary)
        collapsed = _remove_collapsed_apex_facets(temporary)
        mesh = _strict_mesh_facts(temporary)
        mesh["transform_zero_coordinates_canonicalized"] = canonicalized
        mesh["collapsed_apex_facets_removed"] = collapsed
        volume_error = abs(float(mesh["signed_volume"]) - float(shape.volume))
        mesh["brep_volume_abs_error_mm3"] = volume_error
        mesh["brep_volume_relative_error"] = volume_error / float(shape.volume)
        if mesh["brep_volume_relative_error"] > 5.0e-4:
            raise RuntimeError(
                "STL tessellation volume differs from BREP by more than "
                f"0.05%: {mesh['brep_volume_relative_error']:.6%}")
        temporary.replace(stl)
    finally:
        temporary.unlink(missing_ok=True)
    _write_print_transform_sidecar(
        stl,
        name=PART_NAME,
        variant=profile.release_variant,
        z_rotation_deg=0.0,
        oriented_bbox=oriented_bbox,
        mesh_facts=mesh,
        mesh_tolerance_mm=OBIWAN_MESH_TOLERANCE_MM,
        mesh_angular_tolerance=OBIWAN_MESH_ANGULAR_TOLERANCE,
    )
    return mesh, oriented_bbox


def _catalog(
    *, output: Path, stl: Path, magnet_tools: Sequence[object],
    profile: VaseTEBMProfile,
) -> dict[str, Any]:
    sidecar_path = stl.with_suffix(".print.json")
    sidecar = validate_print_sidecar(stl, sidecar_path)
    matrix = sidecar["source_to_stl_matrix"]
    source_files, source_hashes = _source_bindings(output)
    sites = [_site_record(tools, matrix) for tools in magnet_tools]
    artifact_id = (
        f"{ARTIFACT_STATE}:{profile.release_variant}:{PART_NAME}")
    global_geometry = DEFAULT_SPEC.facts()
    global_geometry.pop("paired_magnet_face_separation_mm", None)
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "schema_sha256": sha256_file(CATALOG_SCHEMA),
        "catalog_kind": "released_pause_and_bury_captive_magnets",
        "generated_by": Path(__file__).name,
        "source_revision": _source_revision(source_hashes),
        "print_contract": dict(RELEASE_ACOUSTIC_PRINT_CONTRACT),
        "geometry": {
            **global_geometry,
            "nominal_magnet": "D5.0 x 2.0 mm disc",
            "paired_magnet_face_separation_by_interface_profile_mm": {
                "standard_straight": NOMINAL_PAIRED_FACE_SEPARATION_MM,
            },
            "glue": False,
            "external_access_opening": False,
            "internal_support_material": False,
            "structural_load_credit_n": 0.0,
        },
        "inventory": {
            "artifact_count": 1,
            "magnet_count": len(sites),
            "count_semantics": (
                f"isolated optional {profile.key} BMR-vase artifact; not an "
                "amendment to "
                "the protected 64-artifact release inventory"
            ),
            "family_counts": {
                profile.release_variant: {
                    "artifact_count": 1,
                    "magnet_count": len(sites),
                },
            },
            "families": [profile.release_variant],
        },
        "exclusions": [{
            "path": "review/captive_magnet_release_catalog.json",
            "reason": (
                "the protected production release remains independent; this "
                "catalog is consumed only with --auxiliary-catalog"
            ),
        }],
        "artifacts": [{
            "id": artifact_id,
            "state": ARTIFACT_STATE,
            "variant": profile.release_variant,
            "part": PART_NAME,
            "stl": _relative(stl, output),
            "stl_sha256": sha256_file(stl),
            "print_sidecar": _relative(sidecar_path, output),
            "print_sidecar_sha256": sha256_file(sidecar_path),
            "print_orientation": "front_face_down",
            "rotation_deg": sidecar["rotation_deg"],
            "source_to_stl_matrix": matrix,
            "sites": sites,
            "source_files": source_files,
            "source_file_sha256": source_hashes,
        }],
    }


def export_artifacts(
    output_root: Path,
    profile: str | VaseTEBMProfile = "stock",
) -> dict[str, Path]:
    spec = vase_profile(profile)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "brep": output_root / f"{PART_NAME}.brep",
        "step": output_root / f"{PART_NAME}.step",
        "stl": output_root / f"{PART_NAME}.stl",
        "facts": output_root / f"{PART_NAME}.facts.json",
        "catalog": output_root / f"{PART_NAME}.catalog.json",
        "manifest": output_root / "cad_manifest.json",
        "stamp": output_root / ".stamp_cad_validated",
    }
    model = build_model(spec)
    _export_native(model.solid, brep=paths["brep"], step=paths["step"])
    mesh, oriented_bbox = _export_print_stl(
        model.solid, paths["stl"], spec)
    sidecar = paths["stl"].with_suffix(".print.json")

    source_bbox = model.solid.bounding_box()
    facts = {
        "schema_version": 1,
        "generated_by": Path(__file__).name,
        "design": design_facts(spec),
        "native_geometry": {
            "valid": bool(model.solid.is_valid),
            "solid_count": len(model.solid.solids()),
            "volume_mm3": float(model.solid.volume),
            "bounds_mm": {
                "minimum": [source_bbox.min.X, source_bbox.min.Y,
                            source_bbox.min.Z],
                "maximum": [source_bbox.max.X, source_bbox.max.Y,
                            source_bbox.max.Z],
                "size": [source_bbox.size.X, source_bbox.size.Y,
                         source_bbox.size.Z],
            },
        },
        "print_geometry": {
            "bounds_size_mm": [oriented_bbox.size.X, oriented_bbox.size.Y,
                               oriented_bbox.size.Z],
            "p2s_256mm_fit": True,
            "support_enabled": False,
            "mesh": mesh,
        },
        "magnet_sites": [tools.facts() for tools in model.magnet_tools],
        "files": {
            "brep": {"path": paths["brep"].name,
                     "sha256": sha256_file(paths["brep"])},
            "step": {"path": paths["step"].name,
                     "sha256": sha256_file(paths["step"])},
            "stl": {"path": paths["stl"].name,
                    "sha256": sha256_file(paths["stl"])},
            "print_sidecar": {"path": sidecar.name,
                              "sha256": sha256_file(sidecar)},
        },
    }
    _atomic_json(paths["facts"], facts)
    _atomic_json(
        paths["catalog"],
        _catalog(output=paths["catalog"], stl=paths["stl"],
                 magnet_tools=model.magnet_tools, profile=spec),
    )

    # Validate the isolated catalog with the same strict consumer, changing
    # only the protected release-inventory count gate.
    from release_validation import (
        _validate_artifact_bindings,
        normalize_catalog,
    )

    normalized = normalize_catalog(
        paths["catalog"], enforce_release_inventory=False)
    for artifact in normalized["artifacts"]:
        _validate_artifact_bindings(artifact)

    manifest_files = tuple(
        path for key, path in paths.items()
        if key not in {"manifest", "stamp"}
    ) + (sidecar,)
    _atomic_json(paths["manifest"], {
        "schema_version": 1,
        "artifact": (
            f"{ARTIFACT_STATE}:{spec.release_variant}:{PART_NAME}"),
        "profile": spec.key,
        "files": [{
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        } for path in sorted(manifest_files)],
        "validation": {
            "native_brep_round_trip": "pass",
            "step_transaction": "pass",
            "stl_strict_two_manifold": "pass",
            "stl_brep_volume_relative_error": mesh[
                "brep_volume_relative_error"],
            "front_down_print_contract": "pass",
            "auxiliary_catalog_bindings": "pass",
        },
    })
    paths["stamp"].write_text(
        sha256_file(paths["manifest"]) + "\n", encoding="ascii")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("stock", "slim"), default="stock")
    parser.add_argument(
        "--output-root", type=Path)
    args = parser.parse_args()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (DEFAULT_OUTPUT_PARENT / args.profile).resolve()
    )
    paths = export_artifacts(output_root, args.profile)
    print(json.dumps(
        {key: str(path) for key, path in paths.items()},
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
