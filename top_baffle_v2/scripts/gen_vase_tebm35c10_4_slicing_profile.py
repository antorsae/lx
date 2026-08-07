#!/usr/bin/env python3
"""Derive one isolated Stock/Slim TEBM-vase slicing profile."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "captive_magnet_slicing_profile.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "build" / "vase_TEBM35C10-4"
    / "stock" / "vase_TEBM35C10-4.slicing_profile.json"
)
VARIANTS = {
    "stock": "Stock-TEBM35C10-4",
    "slim": "Slim-TEBM35C10-4",
}
SUPPORT_KEYS = (
    "enable_support",
    "support_on_build_plate_only",
    "support_critical_regions_only",
    "support_remove_small_overhang",
)


def generate(base: Path, output: Path, profile: str = "stock") -> dict:
    try:
        variant = VARIANTS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown TEBM vase profile {profile!r}") from exc
    payload = json.loads(base.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported base slicing profile: {base}")
    result = copy.deepcopy(payload)
    result["catalog_mode"] = "auxiliary"
    result["generated_from"] = os.path.relpath(base, output.parent)
    result["artifact_scope"] = [{
        "state": "shared",
        "variant": variant,
        "part": "vase_TEBM35C10-4",
    }]
    # The six release-only support exceptions must never leak into this
    # support-free duct-bearing vase.  Ready-project generation still writes
    # all four explicit zeroes globally and onto the normal 3MF object.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profile", choices=tuple(VARIANTS), default="stock")
    args = parser.parse_args()
    generate(
        args.base.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.profile,
    )
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
