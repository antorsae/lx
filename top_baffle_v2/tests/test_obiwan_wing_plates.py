#!/usr/bin/env python3
"""Fast contract checks for local Obi-Wan flat/graded B-wing plates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT / "scripts"):
    text = str(import_root)
    if text not in sys.path:
        sys.path.insert(0, text)

import build_obiwan_wing_plate as plate


EXPECTED = {
    "flat": {
        "plate_name": "obiwan_flat_wings_split2_combo",
        "names": (
            "obiwan_05_split2_flat_wing_LM_lower_left_1_of_2",
            "obiwan_06_split2_flat_wing_LM_UM_upper_left_2_of_2",
            "obiwan_08_split2_flat_wing_LM_lower_right_1_of_2",
            "obiwan_09_split2_flat_wing_LM_UM_upper_right_2_of_2",
        ),
    },
    "graded": {
        "plate_name": "obiwan_graded_wings_split2_combo",
        "names": (
            "obiwan_11_split2_graded_wing_LM_lower_left_1_of_2",
            "obiwan_12_split2_graded_wing_LM_UM_upper_left_2_of_2",
            "obiwan_14_split2_graded_wing_LM_lower_right_1_of_2",
            "obiwan_15_split2_graded_wing_LM_UM_upper_right_2_of_2",
        ),
    },
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_variant(slug: str) -> None:
    api = plate.get_variant(slug)
    api.activate()
    expected = EXPECTED[slug]
    check(
        api.PLATE_NAME == expected["plate_name"],
        f"{slug}: plate identity drifted",
    )
    check(
        tuple(part.friendly_name for part in api.PARTS) == expected["names"],
        f"{slug}: four-part friendly inventory or ordering drifted",
    )
    check(
        tuple((part.side, part.role) for part in api.PARTS) == (
            ("left", "lm_lower"),
            ("left", "lm_um_upper"),
            ("right", "lm_lower"),
            ("right", "lm_um_upper"),
        ),
        f"{slug}: wing source-role bindings drifted",
    )
    for part in api.PARTS:
        matrix = part.matrix4
        check(
            matrix[2] == (0.0, 0.0, 1.0, 0.0)
            and matrix[3] == (0.0, 0.0, 0.0, 1.0),
            f"{part.friendly_name}: placement is not front-face-down Rz+XY",
        )
        xx = matrix[0][0] ** 2 + matrix[0][1] ** 2
        yy = matrix[1][0] ** 2 + matrix[1][1] ** 2
        dot = matrix[0][0] * matrix[1][0] + matrix[0][1] * matrix[1][1]
        determinant = (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )
        check(
            math.isclose(xx, 1.0, abs_tol=1.0e-12)
            and math.isclose(yy, 1.0, abs_tol=1.0e-12)
            and math.isclose(dot, 0.0, abs_tol=1.0e-12)
            and math.isclose(determinant, 1.0, abs_tol=1.0e-12),
            f"{part.friendly_name}: placement is not a rigid proper rotation",
        )

    packing = plate._packing_facts(plate._authoritative_footprints())
    check(
        packing["minimum_actual_xy_gap_mm"]
        >= plate.MINIMUM_PART_GAP_MM,
        "locked plate no longer has the required inter-part gap",
    )
    check(
        packing["minimum_actual_bed_edge_mm"]
        >= plate.MINIMUM_BED_EDGE_MM,
        "locked plate no longer has the required bed-edge clearance",
    )

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        f"obiwan_{slug}_wing_plate_source",
        f"obiwan_{slug}_wing_plate",
        f"obiwan_{slug}_wing_plate_validate",
    ):
        check(
            target in makefile,
            f"{target} is absent from the first-class Make graph",
        )
    check(
        "--slice-missing" in makefile
        and "scripts/build_obiwan_wing_plate.py" in makefile,
        "Make does not route the wing plate through its local builder",
    )

    check(api.PLATE_MANIFEST.is_file(), "generated plate manifest is absent")
    manifest = json.loads(api.PLATE_MANIFEST.read_text(encoding="utf-8"))
    check(
        tuple(record["friendly_name"] for record in manifest["parts"])
        == expected["names"],
        "generated manifest part inventory drifted",
    )
    check(
        manifest["support_policy"]["enabled"] is False
        and set(manifest["support_policy"]["global_and_object_fields"].values())
        == {"0"},
        "generated source contract does not pin all support fields off",
    )
    check(
        manifest["magnet_pause"]["magnet_count"] == 6
        and manifest["magnet_pause"]["pause_z_mm"] == plate.PAUSE_Z_MM,
        "six-magnet pause contract drifted",
    )
    print(f"Obi-Wan {slug.title()} four-wing local plate: all checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("flat", "graded"), action="append",
        help="variant to check; repeat to check both (default: both)")
    args = parser.parse_args()
    for slug in args.variant or ("flat", "graded"):
        check_variant(slug)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, ValueError) as exc:
        print(f"test_obiwan_wing_plates.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
