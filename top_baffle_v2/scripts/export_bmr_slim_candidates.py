#!/usr/bin/env python3
"""Export one explicit TEBM35C10-4 BMR-slim candidate as STEP-first CAD.

The BMR-slim topology is intentionally CAD-only and unqualified.  It is kept
out of the protected release/slicing catalogs until a real driver, magnet and
cable fit has been checked.  Run one variant per process because Proud and
Obi-Wan routing profiles are mutually exclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))

from build123d import export_brep, export_step, import_brep

from export_steps import FIXED_TIMESTAMP, validate_step_transaction
from lx521_baffle.io import pretty_json_bytes, sha256_file
from lx521_baffle.tebm35c10_4_land import BMR_SLIM_LAND


SOURCE_FILES = (
    "src/lx521_baffle/assembly.py",
    "src/lx521_baffle/base.py",
    "src/lx521_baffle/cables.py",
    "src/lx521_baffle/geom.py",
    "src/lx521_baffle/io.py",
    "src/lx521_baffle/magnet_contract.py",
    "src/lx521_baffle/magnets.py",
    "src/lx521_baffle/tebm35c10_4_land.py",
    "src/lx521_baffle/proud/b.py",
    "src/lx521_baffle/proud/b2.py",
    "src/lx521_baffle/proud/b2_split.py",
    "src/lx521_baffle/proud/vase_tebm35c10_4.py",
    "src/lx521_baffle/proud/vase_tebm35c10_4_stock_bmr_slim.py",
    "src/lx521_baffle/proud/vase_tebm35c10_4_slim_bmr_slim.py",
    "src/lx521_baffle/obiwan/attachments.py",
    "src/lx521_baffle/obiwan/bmr_pod.py",
    "src/lx521_baffle/obiwan/bmr_crescent.py",
    "src/lx521_baffle/obiwan/bmr_crescent_opposed.py",
    "src/lx521_baffle/obiwan/bmr_crescent_bmr_slim.py",
    "src/lx521_baffle/obiwan/bmr_crescent_opposed_bmr_slim.py",
    "src/lx521_baffle/obiwan/carriers.py",
    "src/lx521_baffle/obiwan/closure_webs.py",
    "src/lx521_baffle/obiwan/joints.py",
    "src/lx521_baffle/obiwan/route.py",
    "src/lx521_baffle/obiwan/wings.py",
    "scripts/export_bmr_slim_candidates.py",
    "scripts/export_steps.py",
)


VARIANTS = {
    "proud-stock": {
        "routing_profile": "proud",
        "module": (
            "lx521_baffle.proud.vase_tebm35c10_4_stock_bmr_slim"),
        "relative_output": (
            "build/bmr_slim_TEBM35C10-4/proud/stock/"
            "vase_TEBM35C10-4.step"),
    },
    "proud-slim": {
        "routing_profile": "proud",
        "module": (
            "lx521_baffle.proud.vase_tebm35c10_4_slim_bmr_slim"),
        "relative_output": (
            "build/bmr_slim_TEBM35C10-4/proud/slim/"
            "vase_TEBM35C10-4.step"),
    },
    "obiwan-coaxial": {
        "routing_profile": "obiwan",
        "module": "lx521_baffle.obiwan.bmr_crescent_bmr_slim",
        "relative_output": (
            "build/bmr_slim_TEBM35C10-4/"
            "obiwan_bmr_slim_crescent_TEBM35C10-4.step"),
    },
    "obiwan-opposed": {
        "routing_profile": "obiwan",
        "module": (
            "lx521_baffle.obiwan.bmr_crescent_opposed_bmr_slim"),
        "relative_output": (
            "build/bmr_slim_TEBM35C10-4/"
            "obiwan_bmr_slim_crescent_opposed_TEBM35C10-4.step"),
    },
}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.parent.resolve())


def _source_bindings(output: Path) -> tuple[list[str], dict[str, str]]:
    paths = tuple(PROJECT_ROOT / relative for relative in SOURCE_FILES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing BMR-slim source provenance: {missing}")
    values = [_relative(path, output) for path in paths]
    return values, {
        value: sha256_file(path)
        for value, path in zip(values, paths, strict=True)
    }


def _source_revision(source_hashes: dict[str, str]) -> str:
    encoded = json.dumps(
        dict(sorted(source_hashes.items())),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _export_native(solid, step_path: Path) -> tuple[Path, Path]:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    brep_path = step_path.with_suffix(".brep")
    temporary_step = step_path.with_name(
        f".{step_path.stem}.{os.getpid()}.tmp.step")
    temporary_brep = brep_path.with_name(
        f".{brep_path.stem}.{os.getpid()}.tmp.brep")
    try:
        export_brep(solid, str(temporary_brep))
        round_trip = import_brep(str(temporary_brep))
        source_shell_count = len(solid.shells())
        source_volume = float(solid.volume)
        volume_relative_error = abs(
            float(round_trip.volume) - source_volume) / source_volume
        if (not round_trip.is_valid or len(round_trip.solids()) != 1
                or len(round_trip.shells()) != source_shell_count
                or volume_relative_error > 1.0e-9):
            raise RuntimeError("BMR-slim BREP round-trip validation failed")
        export_step(
            solid, str(temporary_step), timestamp=FIXED_TIMESTAMP)
        validate_step_transaction(temporary_step)
        temporary_brep.replace(brep_path)
        temporary_step.replace(step_path)
    finally:
        temporary_step.unlink(missing_ok=True)
        temporary_brep.unlink(missing_ok=True)
    return brep_path, step_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    required_profile = spec["routing_profile"]
    actual_profile = os.environ.get("LX_ROUTING_PROFILE", "")
    if actual_profile != required_profile:
        raise SystemExit(
            f"{args.variant} requires LX_ROUTING_PROFILE={required_profile}; "
            f"got {actual_profile!r}")

    module = importlib.import_module(spec["module"])
    if module.LAND_TOPOLOGY is not BMR_SLIM_LAND:
        raise RuntimeError(
            f"{args.variant}: wrapper no longer selects canonical BMR-slim")
    if (module.RELEASE_AUTHORIZED is not False
            or module.PHYSICAL_MEASURE_REQUIRED is not True):
        raise RuntimeError(
            f"{args.variant}: candidate qualification flags regressed")
    model = module.build_model()
    solid = model.solid
    solids = list(solid.solids())
    expected_shells = 1 + module.MAGNET_COUNT
    if (not solid.is_valid or len(solids) != 1
            or len(solid.shells()) != expected_shells):
        raise RuntimeError(
            f"{module.PART_NAME}: expected one valid solid and "
            f"{expected_shells} shells; got valid={solid.is_valid}, "
            f"solids={len(solids)}, shells={len(solid.shells())}")

    step_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (PROJECT_ROOT / spec["relative_output"]).resolve()
    )
    brep_path, step_path = _export_native(solid, step_path)
    bounds = solid.bounding_box()
    design = (
        module.design_facts(model.magnet_tools)
        if required_profile == "obiwan"
        else module.design_facts()
    )
    topology = design.get("land_topology", design.get("pod", {}))
    if (topology.get("key") != "bmr-slim"
            or design.get("variant") != module.VARIANT
            or design.get("release_variant") != module.RELEASE_VARIANT
            or design.get("release_authorized") is not False
            or design.get("physical_measure_required") is not True):
        raise RuntimeError(
            f"{args.variant}: wrapper/design identity contract regressed")
    if required_profile == "proud":
        if model.land_topology is not BMR_SLIM_LAND:
            raise RuntimeError(
                f"{args.variant}: built model lost BMR-slim topology")
        expected_profile = args.variant.removeprefix("proud-")
        if design.get("profile") != expected_profile:
            raise RuntimeError(
                f"{args.variant}: expected {expected_profile} envelope")
        driver = design["tebm35c10_4"]
        if abs(float(driver["axis_pitch_mm"]) - 49.3) > 1.0e-9:
            raise RuntimeError(f"{args.variant}: driver pitch moved")
    else:
        placement = design["axis_placement"]
        if abs(float(placement["preserved_axis_y_mm"])
               - 452.494193004) > 1.0e-9:
            raise RuntimeError(f"{args.variant}: acoustic axis moved")
    facts_path = step_path.with_suffix(".facts.json")
    source_files, source_hashes = _source_bindings(facts_path)
    facts = {
        "schema_version": 1,
        "generated_by": Path(__file__).name,
        "variant": args.variant,
        "status": "candidate_not_release_authorized",
        "release_authorized": False,
        "physical_measure_required": True,
        "design": design,
        "native_geometry": {
            "valid": True,
            "solid_count": 1,
            "shell_count": len(solid.shells()),
            "volume_mm3": float(solid.volume),
            "bounds_mm": {
                "minimum": [bounds.min.X, bounds.min.Y, bounds.min.Z],
                "maximum": [bounds.max.X, bounds.max.Y, bounds.max.Z],
                "size": [bounds.size.X, bounds.size.Y, bounds.size.Z],
            },
        },
        "source_files": source_files,
        "source_file_sha256": source_hashes,
        "source_revision": _source_revision(source_hashes),
        "files": {
            "brep": {
                "path": brep_path.name,
                "sha256": sha256_file(brep_path),
            },
            "step": {
                "path": step_path.name,
                "sha256": sha256_file(step_path),
            },
        },
    }
    _atomic_bytes(
        facts_path, pretty_json_bytes(facts, allow_nan=False))
    print(f"[export_bmr_slim_candidates] wrote {step_path}")


if __name__ == "__main__":
    main()
