#!/usr/bin/env python3
"""Export the candidate STEP-first ``obiwan_bmr_crescent_TEBM35C10-4`` set.

This mirrors ``export_vase_tebm35c10_4.py``: one isolated, transactional
artifact family in its own build child, deliberately outside the Obi-Wan
stage manifest, the release inventory and ``to_print``.  The authoritative
BREP and STEP stay in installed/source coordinates and only the STL receives
the release-wide X180 front-face-down transform.

Unlike the vase there is no captive-magnet catalog: this part carries no
magnets, exactly like the released ND25FW-4 crescent it is an alternative to.
Nothing here is release-authorized; the facts payload carries the candidate
flags and the exporter refuses to pretend otherwise.
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

# The Obi-Wan carriers refuse to build outside a live guard, and this part
# reaches them through the released crescent.  Re-exec before importing any
# CAD so an unguarded invocation never starts a build it cannot finish.
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
from lx521_baffle.print_contract import (
    RELEASE_ACOUSTIC_PRINT_CONTRACT,
    validate_print_sidecar,
)
from lx521_baffle.obiwan.bmr_crescent import (
    MAGNET_COUNT,
    PART_NAME,
    PHYSICAL_MEASURE_REQUIRED,
    RELEASE_AUTHORIZED,
    RELEASE_VARIANT,
    build_model,
    design_facts,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "build" / "bmr_crescent_TEBM35C10-4"
ARTIFACT_STATE = "shared"
BED_LIMIT_MM = 256.0
SOURCE_REVISION_ENV = "LX_CAD_SOURCE_SHA256"

SOURCE_FILES = (
    "src/lx521_baffle/base.py",
    "src/lx521_baffle/cables.py",
    "src/lx521_baffle/geom.py",
    "src/lx521_baffle/io.py",
    "src/lx521_baffle/print_contract.py",
    "src/lx521_baffle/stl_export.py",
    "src/lx521_baffle/proud/b.py",
    "src/lx521_baffle/proud/b2.py",
    "src/lx521_baffle/proud/v1.py",
    "src/lx521_baffle/proud/vase_tebm35c10_4.py",
    "src/lx521_baffle/obiwan/attachments.py",
    "src/lx521_baffle/obiwan/bmr_crescent.py",
    "src/lx521_baffle/obiwan/carriers.py",
    "src/lx521_baffle/obiwan/joints.py",
    "scripts/check_manifold.py",
    "scripts/export_bmr_crescent.py",
    "scripts/export_piece_stls.py",
    "scripts/export_steps.py",
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
        value: sha256_file(path)
        for value, path in zip(values, paths, strict=True)
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
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{suffix}")
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


def _export_print_stl(shape, stl: Path) -> tuple[dict[str, Any], object]:
    oriented = Rot(X=180.0) * shape
    oriented_bbox = oriented.bounding_box()
    size = oriented_bbox.size
    if max(size.X, size.Y, size.Z) > BED_LIMIT_MM + 1.0e-6:
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
        variant=RELEASE_VARIANT,
        z_rotation_deg=0.0,
        oriented_bbox=oriented_bbox,
        mesh_facts=mesh,
        mesh_tolerance_mm=OBIWAN_MESH_TOLERANCE_MM,
        mesh_angular_tolerance=OBIWAN_MESH_ANGULAR_TOLERANCE,
    )
    return mesh, oriented_bbox


def _qualification() -> dict[str, Any]:
    """The candidate gate this part must not be printed for use without."""
    return {
        "release_authorized": RELEASE_AUTHORIZED,
        "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
        "status": "candidate_not_release_authorized",
        "counts_against_release_inventory": False,
        "in_obiwan_stage_manifest": False,
        "in_to_print": False,
        "open_items": [
            "TEBM35C10-4 flange/basket/depth measured on the actual driver, "
            "not taken from the published envelope",
            "back-to-back 2.40 mm partition printed and pressure/rattle "
            "checked with both drivers fitted",
            "M2 x 4 heat-set installation in both D66 lands without "
            "breakthrough into the opposite pocket",
            "UM-to-crescent two-screw joint re-proven at this part's hanging "
            "mass, which is well above the released ND25FW-4 crescent's",
            "free T cable dressed to both -Y lead outlets without pinch "
            "behind the crescent",
        ],
    }


def export_artifacts(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "brep": output_root / f"{PART_NAME}.brep",
        "step": output_root / f"{PART_NAME}.step",
        "stl": output_root / f"{PART_NAME}.stl",
        "facts": output_root / f"{PART_NAME}.facts.json",
        "manifest": output_root / "cad_manifest.json",
        "stamp": output_root / ".stamp_cad_validated",
    }
    model = build_model()
    solid = model.solid
    _export_native(solid, brep=paths["brep"], step=paths["step"])
    mesh, oriented_bbox = _export_print_stl(solid, paths["stl"])
    sidecar = paths["stl"].with_suffix(".print.json")
    validate_print_sidecar(paths["stl"], sidecar)

    source_files, source_hashes = _source_bindings(paths["facts"])
    source_bbox = solid.bounding_box()
    facts = {
        "schema_version": 1,
        "generated_by": Path(__file__).name,
        "artifact": f"{ARTIFACT_STATE}:{RELEASE_VARIANT}:{PART_NAME}",
        "qualification": _qualification(),
        "print_contract": dict(RELEASE_ACOUSTIC_PRINT_CONTRACT),
        "magnet_count": MAGNET_COUNT,
        "stand_state": os.environ.get("LX_STAND_FOOT", ""),
        "stand_state_independent": True,
        "design": design_facts(),
        "native_geometry": {
            "valid": bool(solid.is_valid),
            "solid_count": len(solid.solids()),
            "volume_mm3": float(solid.volume),
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
        "source_files": source_files,
        "source_file_sha256": source_hashes,
        "source_revision": _source_revision(source_hashes),
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

    manifest_files = tuple(
        path for key, path in paths.items()
        if key not in {"manifest", "stamp"}
    ) + (sidecar,)
    _atomic_json(paths["manifest"], {
        "schema_version": 1,
        "artifact": f"{ARTIFACT_STATE}:{RELEASE_VARIANT}:{PART_NAME}",
        "release_authorized": RELEASE_AUTHORIZED,
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
            "one_solid": "pass",
        },
    })
    paths["stamp"].write_text(
        sha256_file(paths["manifest"]) + "\n", encoding="ascii")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else DEFAULT_OUTPUT_ROOT.resolve()
    )
    paths = export_artifacts(output_root)
    print(json.dumps(
        {key: str(path) for key, path in paths.items()},
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
