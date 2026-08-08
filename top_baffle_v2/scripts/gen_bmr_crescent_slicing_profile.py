#!/usr/bin/env python3
"""Derive one isolated candidate BMR-crescent slicing profile.

This is ``gen_vase_tebm35c10_4_slicing_profile.py`` for the two candidate
Obi-Wan BMR pods, and it deliberately makes the same material choice.  The
base ``captive_magnet_slicing_profile.json`` (Bambu PLA Tough+, six walls,
30% gyroid) is the profile for magnet-bearing parts that are not the
structural core; ``captive_magnet_slicing_profile_petg_gf.json`` exists only
for the three parts that *are* -- the two LM keyed halves and the UM carrier
-- and it is scoped to exactly those by ``artifact_scope``.  A crescent hangs
off the UM carrier's already-qualified M3 half-lap; it is the hanging part,
not the joint, exactly as the released ND25FW-4 crescent it replaces is not in
the PETG-GF scope either.  So both candidates take the base profile unchanged,
walls and infill included, and differ from it only in the four fields the vase
also changes.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "captive_magnet_slicing_profile.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "build" / "bmr_crescent_TEBM35C10-4"
SUPPORT_KEYS = (
    "enable_support",
    "support_on_build_plate_only",
    "support_critical_regions_only",
    "support_remove_small_overhang",
)


@dataclass(frozen=True)
class _Variant:
    """One candidate pod's catalog identity, restated for the slicer."""

    key: str
    release_variant: str
    part: str


VARIANTS = {
    variant.key: variant for variant in (
        _Variant(
            "coaxial",
            "Obiwan-TEBM35C10-4-BMR-crescent",
            "obiwan_bmr_crescent_TEBM35C10-4",
        ),
        _Variant(
            "opposed",
            "Obiwan-TEBM35C10-4-BMR-crescent-opposed",
            "obiwan_bmr_crescent_opposed_TEBM35C10-4",
        ),
    )
}


def default_output(variant: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{VARIANTS[variant].part}.slicing_profile.json"


def generate(base: Path, output: Path, variant: str = "coaxial") -> dict:
    try:
        spec = VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown BMR crescent variant {variant!r}") from exc
    payload = json.loads(base.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported base slicing profile: {base}")
    result = copy.deepcopy(payload)
    result["catalog_mode"] = "auxiliary"
    result["generated_from"] = os.path.relpath(base, output.parent)
    result["artifact_scope"] = [{
        "state": "shared",
        "variant": spec.release_variant,
        "part": spec.part,
    }]
    # The six release-only support exceptions belong to the Obi-Wan LM split
    # and must never leak into a support-free candidate pod.  Ready-project
    # generation still writes all four explicit zeroes globally and onto the
    # normal 3MF object.
    result["artifact_overrides"] = []
    process = result.get("repo_overrides", {}).get("process", {})
    if any(str(process.get(key)) != "0" for key in SUPPORT_KEYS):
        raise RuntimeError(
            "base profile no longer pins all four support fields to zero")
    requirements = result.get("requirements", {})
    if requirements.get("support_enabled") is not False:
        raise RuntimeError("base profile no longer requires support off")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if json.loads(temporary.read_text(encoding="utf-8")) != result:
            raise RuntimeError("slicing-profile round trip failed")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def pause_layer_z(profile: dict, closing_plane_mm: float) -> float:
    """The first printed layer strictly above a cavity's closing plane.

    A pause is only ever *published* from the sliced G-code, but the layer
    ladder is fully determined by this profile, so the Z a correct slice must
    land on is predictable from CAD alone.  The delivery validator compares
    the two, which is what turns the pause into a regression rather than a
    number read off whatever the slicer happened to do.
    """
    requirements = profile.get("requirements", {})
    try:
        first = float(requirements["first_layer_height_mm"])
        layer = float(requirements["layer_height_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "slicing profile does not pin the layer ladder") from exc
    if first <= 0.0 or layer <= 0.0:
        raise RuntimeError(f"non-positive layer ladder: {first}/{layer}")
    plane = float(closing_plane_mm)
    if plane < first - 1.0e-9:
        return round(first, 9)
    index = math.floor((plane - first) / layer + 1.0e-6) + 1
    return round(first + index * layer, 9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--variant", choices=tuple(VARIANTS), default="coaxial")
    args = parser.parse_args()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output(args.variant).resolve()
    )
    generate(args.base.expanduser().resolve(), output, args.variant)
    print(output)


if __name__ == "__main__":
    main()
