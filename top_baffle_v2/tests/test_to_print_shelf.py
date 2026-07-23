#!/usr/bin/env python3
"""Pure-Python contract checks for the friendly P2S delivery shelf."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import sys

import build_to_print_shelf as shelf


ROOT = PROJECT_ROOT


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    resolver_cases = {
        "floor_stand/stl/part.stl": "build/floor_stand/stl/part.stl",
        "no_floor_stand/stl/part.stl": "build/no_floor_stand/stl/part.stl",
        "wings/ac/stl/part.stl": "build/wings/ac/stl/part.stl",
        "build/wings/ac/stl/part.stl": "build/wings/ac/stl/part.stl",
    }
    for historical, canonical in resolver_cases.items():
        check(
            shelf._resolve_shelf_source_relative(
                historical, "compatibility probe").as_posix() == canonical,
            f"legacy shelf source did not resolve read-only: {historical}",
        )

    raw, entries = shelf._catalog_entries(shelf.DEFAULT_CATALOG)
    release, by_id = shelf._release_artifacts(shelf.DEFAULT_RELEASE_CATALOG)
    shelf._bind_entries_to_release(entries, by_id)

    check(raw["printer"] == "Bambu Lab P2S", "catalog printer drift")
    check(len(entries) == 39, "shelf must contain exactly 39 physical files")
    families = {
        family: sum(entry["family"] == family for entry in entries)
        for family in shelf.EXPECTED_FAMILY_COUNTS
    }
    check(families == {"stock": 11, "slim": 11, "obiwan": 17},
          f"unexpected family counts: {families}")
    magnetic = [entry for entry in entries if entry.get("catalog_artifact_id")]
    check(len(magnetic) == 30, "expected 30 audited magnet projects")
    check(len(entries) - len(magnetic) == 9,
          "expected 9 locally sliced non-magnet projects")
    blocker_artifacts = [
        entry["artifact"] for entry in magnetic
        if "support_blocker" in entry["artifact"]
    ]
    check(len(blocker_artifacts) == 1,
          "exactly one shelf project must carry the duct support blocker")
    blocker = Path(blocker_artifacts[0]["support_blocker"])
    check(blocker.name ==
          "lx521_top_obiwan_optional_lm_keyed_1of2_bottom.support_blocker.stl",
          "unexpected shelf support-blocker identity")
    check(blocker.is_file(), "shelf support-blocker STL is missing")

    names = {entry["name"] for entry in entries}
    for required in (
        "stock_01a_of_10_LM_bottom_1_of_3_no_floor_stand",
        "stock_01b_of_10_LM_bottom_1_of_3_floor_stand",
        "stock_03_of_10_LM_mid_right_3_of_3",
        "slim_01a_of_10_LM_bottom_1_of_3_no_floor_stand",
        "slim_03_of_10_LM_mid_right_3_of_3",
        "obiwan_01a_of_16_LM_bottom_keyed_1_of_2_no_floor_stand",
        "obiwan_02_of_16_LM_top_keyed_2_of_2",
        "obiwan_03_of_16_UM_carrier_1_of_1",
        "obiwan_04_of_16_T_tweeter_crescent_1_of_1",
        "obiwan_16_of_16_Ae_wing_UM_right_3_of_3",
    ):
        check(required in names, f"missing required friendly name: {required}")
    for forbidden in (
            "stock_03a_of_10_LM_mid_right_3_of_3_no_floor_stand",
            "stock_03b_of_10_LM_mid_right_3_of_3_floor_stand",
            "slim_03a_of_10_LM_mid_right_3_of_3_no_floor_stand",
            "slim_03b_of_10_LM_mid_right_3_of_3_floor_stand",
            "obiwan_02a_of_16_LM_top_keyed_2_of_2_no_floor_stand",
            "obiwan_02b_of_16_LM_top_keyed_2_of_2_floor_stand",
            "obiwan_03a_of_16_UM_carrier_1_of_1_no_floor_stand",
            "obiwan_03b_of_16_UM_carrier_1_of_1_floor_stand"):
        check(forbidden not in names, f"stale state duplicate remains: {forbidden}")
    check(not any("LM_top_keyed_1_of_2" in name for name in names),
          "Obi-Wan keyed LM top must be 2 of 2")
    check(not any("core_1of2_lm_carrier" in entry["source_stl"]
                  for entry in entries),
          "P2S-oversize Obi-Wan monolith leaked into shelf")
    check(not any(any(token in entry["source_stl"].lower()
                          for token in shelf.UNPRINTABLE_OR_LEGACY_TOKENS)
                  for entry in entries),
          "legacy/dormant output leaked into shelf")

    for entry in entries:
        source = Path(entry["source_path"])
        check(source.is_file(), f"missing source STL {source}")
        stl, project = shelf._delivery_paths(shelf.DEFAULT_SHELF, entry)
        check(stl.name == f"{entry['name']}.stl", "friendly STL name drift")
        check(project.name == f"{entry['name']}.gcode.3mf",
              "friendly P2S project name drift")
        if entry.get("catalog_artifact_id"):
            artifact = by_id[entry["catalog_artifact_id"]]
            check(Path(artifact["stl"]).resolve() == source.resolve(),
                  f"release source mismatch for {entry['name']}")

    check(release["inventory"]["artifact_count"] == 56,
          "unexpected canonical captive-magnet release inventory")
    print("to_print shelf catalog: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, shelf.ShelfError) as exc:
        print(f"test_to_print_shelf.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
