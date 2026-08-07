#!/usr/bin/env python3
"""Fast contracts for the local Obi-Wan floor/no-floor core plates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT / "scripts"):
    text = str(import_root)
    if text not in sys.path:
        sys.path.insert(0, text)

import build_obiwan_combo_plate as plate


EXPECTED = {
    "no_floor_stand": {
        "plate_name": (
            "obiwan_01_02_03_04_LM_UM_combo_no_floor_stand"
        ),
        "bottom_name": (
            "obiwan_01_LM_bottom_keyed_1_of_2_no_floor_stand"
        ),
        "triangle_count": 63_008,
        "make_slug": "no_floor",
        "infill": (40.0, "gyroid"),
    },
    "floor_stand": {
        "plate_name": (
            "obiwan_01_02_03_04_LM_UM_combo_floor_stand"
        ),
        "bottom_name": (
            "obiwan_01_LM_bottom_keyed_1_of_2_floor_stand"
        ),
        "triangle_count": 165_892,
        "make_slug": "floor",
        "infill": (100.0, "zig-zag"),
    },
}

SHARED_NAMES = (
    "obiwan_02_LM_top_keyed_2_of_2",
    "obiwan_03_UM_carrier_1_of_1",
    "obiwan_04_T_tweeter_crescent_1_of_1",
)
PETG_GF_PROFILE = ROOT / "captive_magnet_slicing_profile_petg_gf.json"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_variant(state: str) -> None:
    api = plate.get_variant(state)
    api.activate()
    expected = EXPECTED[state]
    expected_names = (expected["bottom_name"], *SHARED_NAMES)
    check(
        api.PLATE_NAME == expected["plate_name"],
        f"{state}: plate identity drifted",
    )
    check(
        api.EXPECTED_TRIANGLE_COUNT == expected["triangle_count"],
        f"{state}: expected triangle count drifted",
    )
    check(
        tuple(part.friendly_name for part in api.PARTS) == expected_names,
        f"{state}: four-part friendly inventory or ordering drifted",
    )
    check(
        tuple(part.translation_mm for part in api.PARTS)
        == plate.LOCKED_TRANSLATIONS_MM,
        f"{state}: locked translation-only disposition drifted",
    )
    for part in api.PARTS:
        check(
            f"/build/{state}/" in str(part.source_stl),
            f"{part.friendly_name}: source is not bound to {state}",
        )
        if part.artifact_id is not None:
            check(
                part.artifact_id.startswith(f"{state}:"),
                f"{part.friendly_name}: release identity crossed stand states",
            )
            check(
                part.support_blocker is not None
                and f"/build/{state}/support_blockers/" in str(
                    part.support_blocker),
                f"{part.friendly_name}: blocker crossed stand states",
            )

    contract = api.validate_source_bundle()
    check(
        contract["stand_state"] == state
        and contract["name"] == api.PLATE_NAME,
        f"{state}: source manifest identity drifted",
    )
    check(
        contract["triangle_count"] == expected["triangle_count"]
        and contract["expected_disconnected_printable_part_count"] == 4,
        f"{state}: exact four-part triangle contract drifted",
    )
    check(
        contract["packing"]["minimum_actual_xy_gap_mm"]
        >= plate.MINIMUM_PART_GAP_MM,
        f"{state}: locked plate no longer has the required part gap",
    )
    check(
        contract["support_policy"]["duct_blocker_count"] == 3
        and set(contract["support_policy"][
            "global_and_object_fields"].values()) == {"1"},
        f"{state}: support/blocker source contract drifted",
    )
    check(
        (
            contract["print_profile"]["sparse_infill_density_percent"],
            contract["print_profile"]["sparse_infill_pattern"],
        ) == expected["infill"],
        f"{state}: authoritative infill contract drifted",
    )
    check(
        contract["magnet_pause"]["magnet_count"] == 6
        and contract["magnet_pause"]["pause_z_mm"] == plate.PAUSE_Z_MM,
        f"{state}: six-magnet pause contract drifted",
    )

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    check(
        plate.DEFAULT_PROFILE == PETG_GF_PROFILE,
        "combined core plates must default to the structural PETG-GF profile",
    )
    profile = json.loads(PETG_GF_PROFILE.read_text(encoding="utf-8"))
    check(
        profile["user_filament_preset"]
        == "TINMORRY PETG-GF Profile @BBL P2S"
        and profile["repo_overrides"]["process"]["wall_loops"] == "8",
        "combined core profile must pin saved TINMORRY PETG-GF and 8 walls",
    )
    make_slug = expected["make_slug"]
    for target in (
        f"obiwan_{make_slug}_combo_plate_source",
        f"obiwan_{make_slug}_combo_plate",
        f"obiwan_{make_slug}_combo_plate_validate",
        f"obiwan_{make_slug}_combo_plate_to_print",
    ):
        check(
            target in makefile,
            f"{target} is absent from the first-class Make graph",
        )
    check(
        makefile.count('--profile "$(PETG_GF_PROFILE)"') >= 7,
        "standalone 01a and both core-plate recipes must use PETG-GF",
    )
    print(f"Obi-Wan {state} four-piece local plate: all checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=tuple(EXPECTED), action="append",
        help="stand state to check; repeat for both (default: both)")
    args = parser.parse_args()
    for state in args.variant or tuple(EXPECTED):
        check_variant(state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, ValueError) as exc:
        print(f"test_obiwan_combo_plates.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
