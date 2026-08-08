#!/usr/bin/env python3
"""Export both candidate STEP-first Obi-Wan TEBM35C10-4 BMR pods.

This mirrors ``export_vase_tebm35c10_4.py``: one isolated, transactional
artifact family in its own build child, deliberately outside the Obi-Wan
stage manifest, the release inventory and ``to_print``.  The authoritative
BREPs and STEPs stay in installed/source coordinates and only the STLs
receive the release-wide X180 front-face-down transform.

Two variants share that child directory and every gate:

* ``coaxial`` -- ``obiwan_bmr_crescent_TEBM35C10-4``, the two BMRs stacked
  back to back on one axis, 50.2 mm deep, with two captive magnets on its one
  outward D66 land; and
* ``opposed`` -- ``obiwan_bmr_crescent_opposed_TEBM35C10-4``, the qualified
  vase's side-by-side layout on the same crescent mount, 25.1 mm deep and much
  taller, with all four of the vase's captive magnets.

Both carry captive D5 x 2 stations, and neither is wired into the released
captive-magnet catalog or the sliced pause events: the stations are recorded
in each part's own facts payload, and wiring them up is part of the
qualification each variant still owes.  Nothing here is release-authorized;
the facts payload carries the candidate flags and the exporter refuses to
pretend otherwise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from lx521_baffle.obiwan import bmr_crescent, bmr_crescent_opposed


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "build" / "bmr_crescent_TEBM35C10-4"
ARTIFACT_STATE = "shared"
BED_LIMIT_MM = 256.0
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
    "src/lx521_baffle/proud/v1.py",
    "src/lx521_baffle/proud/vase_tebm35c10_4.py",
    "src/lx521_baffle/obiwan/attachments.py",
    "src/lx521_baffle/obiwan/bmr_crescent.py",
    "src/lx521_baffle/obiwan/bmr_crescent_opposed.py",
    "src/lx521_baffle/obiwan/bmr_pod.py",
    "src/lx521_baffle/obiwan/carriers.py",
    "src/lx521_baffle/obiwan/closure_webs.py",
    "src/lx521_baffle/obiwan/joints.py",
    "src/lx521_baffle/obiwan/route.py",
    "src/lx521_baffle/obiwan/wings.py",
    "scripts/check_manifold.py",
    "scripts/export_bmr_crescent.py",
    "scripts/export_piece_stls.py",
    "scripts/export_steps.py",
)

# Both parts are candidates and neither is release-authorized, so these lists
# are the gate, not a formality: they say what has not been proven yet.  The
# shared entries are the ones the two variants owe for the same reason; each
# variant then adds what only its own arrangement raises.
_SHARED_OPEN_ITEMS = (
    "TEBM35C10-4 flange/basket/depth measured on the actual driver, "
    "not taken from the published envelope",
    "M2 x 4 heat-set installation in every D66 land without breakthrough "
    "into a pocket or a magnet cavity",
    "the captive D5 x 2 stations wired up for release: a catalog entry and "
    "the sliced pause events, neither of which exists while the part is a "
    "candidate, then a pull test on the printed land",
    "the T cable threaded for real: out of the UM's declared mouth, into "
    "the Ø6.00 mate-face entry and on to both drivers, with both drivers "
    "fitted and the pod screwed down",
)
_SHARED_CLOSED_ITEMS = (
    "the inherited M4 ND25FW-4 faceplate clamp passages, which carried no "
    "fastener in either variant, are resolved by deletion: the silhouette "
    "they existed to preserve is gone",
    "the open window between the pod and the UM collar is resolved by the "
    "flush junction skirt, which lands on the released crescent's own seam "
    "and is checked head on against the staged collar in both stand states",
    "both external Ø4.6 driver lead outlets are resolved by deletion: the "
    "cable now enters once, on the UM mate face, so no opening reaches the "
    "assembled exterior at all",
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


def _export_print_stl(
    shape, stl: Path, *, part_name: str, release_variant: str,
) -> tuple[dict[str, Any], object]:
    oriented = Rot(X=180.0) * shape
    oriented_bbox = oriented.bounding_box()
    size = oriented_bbox.size
    if max(size.X, size.Y, size.Z) > BED_LIMIT_MM + 1.0e-6:
        raise RuntimeError(
            f"{part_name} exceeds the P2S envelope: "
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
        name=part_name,
        variant=release_variant,
        z_rotation_deg=0.0,
        oriented_bbox=oriented_bbox,
        mesh_facts=mesh,
        mesh_tolerance_mm=OBIWAN_MESH_TOLERANCE_MM,
        mesh_angular_tolerance=OBIWAN_MESH_ANGULAR_TOLERANCE,
    )
    return mesh, oriented_bbox


_COAXIAL_OPEN_ITEMS = (
    "back-to-back 2.40 mm partition printed and pressure/rattle checked "
    "with both drivers fitted; front-face-down it is also a D42.9 "
    "unsupported span over the front pocket, and it carries one Ø4.60 pass",
    "UM-to-pod two-screw joint re-proven at this part's hanging mass, which "
    "is still above the released ND25FW-4 crescent's",
    "the junction skirt loaded at that hanging mass; its section at the ears "
    "is chosen to keep the qualified half-lap governing, but no skirt has "
    "been printed or loaded",
    "an acoustic opinion on the dropped axis: the coaxial pair now sits "
    "86.413 mm from the MU10 axis instead of the released 102.112 mm, and "
    "no measurement or model backs that spacing",
    "the two captive stations on the front land: they sit 15.70 mm below "
    "the released tweeter axis on a land the vase never had to hold a "
    "cantilevered 50.2 mm stack from",
)
_OPPOSED_OPEN_ITEMS = (
    "the two 1.20 mm blind walls printed and pressure/rattle checked with "
    "both drivers fitted; each is a D42.9 unsupported span, the lower one "
    "printed as the last layers over an open pocket",
    "UM-to-pod two-screw joint re-proven at this part's hanging mass on a "
    "far longer arm than either the coaxial pod or the released crescent: "
    "the upper driver axis stands 80.294 mm above the half-lap line",
    "the junction skirt and the 43.88 mm waist between the two lands loaded "
    "at that hanging moment; the skirt's ear section is chosen to keep the "
    "qualified half-lap governing, but nothing has been printed or loaded",
    "an acoustic opinion on both axes: the lower BMR sits 86.413 mm from "
    "the MU10 axis and the upper one 135.713 mm, and no measurement or "
    "model backs either spacing on an Obi-Wan collar",
    "all four captive stations, and specifically the upper pair, which sit "
    "on the land furthest from the only mount this part has",
)


def _qualification(
    variant: "_Variant", open_items: tuple[str, ...],
) -> dict[str, Any]:
    """The candidate gate this part must not be printed for use without."""
    module = variant.module
    return {
        "release_authorized": module.RELEASE_AUTHORIZED,
        "physical_measure_required": module.PHYSICAL_MEASURE_REQUIRED,
        "status": "candidate_not_release_authorized",
        "counts_against_release_inventory": False,
        "in_obiwan_stage_manifest": False,
        "in_to_print": False,
        "in_captive_magnet_release_catalog": False,
        "magnet_release_wiring_note": (
            f"the part buries {module.MAGNET_COUNT} captive D5 x 2 stations "
            "and is deliberately absent from the released catalog, the "
            "release_validation counts and the slicing profiles; the "
            "stations are recorded under design.magnets instead"),
        "open_items": list(_SHARED_OPEN_ITEMS) + list(open_items),
        "closed_items": list(_SHARED_CLOSED_ITEMS),
    }


@dataclass(frozen=True)
class _Variant:
    """One candidate pod: its module and the qualification only it owes."""

    key: str
    module: Any
    open_items: tuple[str, ...]


VARIANTS = {
    variant.key: variant for variant in (
        _Variant("coaxial", bmr_crescent, _COAXIAL_OPEN_ITEMS),
        _Variant("opposed", bmr_crescent_opposed, _OPPOSED_OPEN_ITEMS),
    )
}


def export_artifacts(output_root: Path, variant: _Variant) -> dict[str, Path]:
    module = variant.module
    part_name = module.PART_NAME
    release_variant = module.RELEASE_VARIANT
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "brep": output_root / f"{part_name}.brep",
        "step": output_root / f"{part_name}.step",
        "stl": output_root / f"{part_name}.stl",
        "facts": output_root / f"{part_name}.facts.json",
        "manifest": output_root / f"cad_manifest_{variant.key}.json",
        "stamp": output_root / f".stamp_cad_validated_{variant.key}",
    }
    model = module.build_model()
    solid = model.solid
    # One outer shell plus one sealed void per captive station.  The cavities
    # are buried by construction, so a differing count means either a station
    # vanished into a bore or something else sealed itself in.
    if len(solid.shells()) != 1 + module.MAGNET_COUNT:
        raise RuntimeError(
            f"{part_name}: expected one outer shell plus "
            f"{module.MAGNET_COUNT} buried captive cavities, got "
            f"{len(solid.shells())} shells")
    _export_native(solid, brep=paths["brep"], step=paths["step"])
    mesh, oriented_bbox = _export_print_stl(
        solid, paths["stl"],
        part_name=part_name, release_variant=release_variant)
    sidecar = paths["stl"].with_suffix(".print.json")
    validate_print_sidecar(paths["stl"], sidecar)

    source_files, source_hashes = _source_bindings(paths["facts"])
    source_bbox = solid.bounding_box()
    facts = {
        "schema_version": 1,
        "generated_by": Path(__file__).name,
        "artifact": f"{ARTIFACT_STATE}:{release_variant}:{part_name}",
        "qualification": _qualification(variant, variant.open_items),
        "print_contract": dict(RELEASE_ACOUSTIC_PRINT_CONTRACT),
        "magnet_count": module.MAGNET_COUNT,
        "stand_state": os.environ.get("LX_STAND_FOOT", ""),
        "stand_state_independent": True,
        "design": module.design_facts(model.magnet_tools),
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
        "artifact": f"{ARTIFACT_STATE}:{release_variant}:{part_name}",
        "variant": variant.key,
        "release_authorized": module.RELEASE_AUTHORIZED,
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
            # One outer shell plus one sealed void per captive station.  The
            # cavities are buried by construction, so a differing count means
            # either a station vanished or something else sealed itself in.
            "captive_magnet_voids": module.MAGNET_COUNT,
            "shell_count": len(solid.shells()),
        },
    })
    paths["stamp"].write_text(
        sha256_file(paths["manifest"]) + "\n", encoding="ascii")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--variant", action="append", choices=sorted(VARIANTS),
        help="export only this variant; repeatable (default: both)")
    args = parser.parse_args()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else DEFAULT_OUTPUT_ROOT.resolve()
    )
    selected = args.variant or sorted(VARIANTS)
    written = {
        key: {name: str(path) for name, path in
              export_artifacts(output_root, VARIANTS[key]).items()}
        for key in selected
    }
    print(json.dumps(written, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
