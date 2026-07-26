#!/usr/bin/env python3
"""Pure-Python contract checks for the friendly P2S delivery shelf."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess

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
    def make_dry_run(target: str) -> str:
        run = subprocess.run(
            ["make", "-nB", target],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        check(run.returncode == 0,
              f"make -nB {target} failed: {run.stdout[-2000:]}")
        check("scripts/remote_cad.py" not in run.stdout
              and "osado.lan" not in run.stdout
              and not any(line.lstrip().startswith("ssh ")
                          for line in run.stdout.splitlines()),
              f"{target} must not dispatch CAD or slicing to osado")
        return run.stdout

    shelf_dry_run = make_dry_run("to_print")
    check("--emit-ready-projects" not in shelf_dry_run
          and "scripts/slice_captive_magnets.py" not in shelf_dry_run,
          "to_print must consume existing release projects, not implicitly "
          "run the heavyweight captive-magnet release slicer")
    combo_dry_run = make_dry_run("obiwan_combo_plate")
    check(0 <= combo_dry_run.find("--dry-run")
          < combo_dry_run.find("--slice-missing"),
          "composite artifact must dry-run before local slicing")
    promotion_dry_run = make_dry_run("obiwan_combo_plate_to_print")
    shelf_command = promotion_dry_run.rfind(
        "scripts/build_to_print_shelf.py")
    check(shelf_command >= 0
          and "--validate-only" in promotion_dry_run[shelf_command:]
          and f'--only "{shelf.combo.PLATE_NAME}"'
          in promotion_dry_run[shelf_command:],
          "targeted composite promotion must disable slicing and cross the "
          "existing complete-shelf validation barrier")

    concrete_targets = (
        "build/print_plates/obiwan/"
        f"{shelf.combo.PLATE_NAME}.stl",
        "review/to_print_slice_workspace/composite/"
        f"{shelf.combo.PLATE_NAME}/ready/"
        f"{shelf.combo.PLATE_NAME}.gcode.3mf",
        "to_print/obiwan/3mf/"
        f"{shelf.combo.PLATE_NAME}.gcode.3mf",
    )
    for target in concrete_targets:
        output = make_dry_run(target)
        check("build_obiwan_combo_plate.py" in output,
              f"{target} is not backed by the composite artifact graph")
        database = subprocess.run(
            ["make", "-np", target],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        check(database.returncode == 0,
              f"cannot inspect Make record for {target}")
        marker = f"\n{target}:"
        start = database.stdout.find(marker)
        found = start >= 0
        if not found and database.stdout.startswith(f"{target}:"):
            start = -1
            found = True
        check(found, f"missing concrete Make record for {target}")
        record = database.stdout[start + 1:].split("\n\n", 1)[0]
        check("#  Phony target" not in record and ".stamp_" in record,
              f"{target} must be a non-phony stamp-backed artifact")

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
    check(len(entries) == 48, "shelf must contain exactly 48 entries")
    families = {
        family: sum(entry["family"] == family for entry in entries)
        for family in shelf.EXPECTED_FAMILY_COUNTS
    }
    check(families == {"stock": 11, "slim": 11, "obiwan": 26},
          f"unexpected family counts: {families}")
    magnetic = [entry for entry in entries if shelf._is_magnet_entry(entry)]
    check(len(magnetic) == 39, "expected 39 audited magnet projects")
    check(len(entries) - len(magnetic) == 9,
          "expected 9 locally sliced non-magnet projects")
    canonical_magnetic = [
        entry for entry in entries if entry.get("catalog_artifact_id")
    ]
    blocker_artifacts = [
        entry["artifact"] for entry in canonical_magnetic
        if "support_blocker" in entry["artifact"]
    ]
    check(len(blocker_artifacts) == 4,
          "all four support-enabled shelf projects must carry duct blockers")
    blocker_names = {
        (
            artifact["state"],
            Path(artifact["support_blocker"]).name,
        )
        for artifact in blocker_artifacts
    }
    check(blocker_names == {
        (
            "no_floor_stand",
            "lx521_top_obiwan_optional_lm_keyed_1of2_bottom."
            "support_blocker.stl",
        ),
        (
            "floor_stand",
            "lx521_top_obiwan_optional_lm_keyed_1of2_bottom."
            "support_blocker.stl",
        ),
        (
            "no_floor_stand",
            "lx521_top_obiwan_optional_lm_keyed_2of2_top."
            "support_blocker.stl",
        ),
        (
            "no_floor_stand",
            "lx521_top_obiwan_core_2of2_um_carrier."
            "support_blocker.stl",
        ),
    }, "unexpected shelf support-blocker inventory")
    check(all(Path(artifact["support_blocker"]).is_file()
              for artifact in blocker_artifacts),
          "a shelf support-blocker STL is missing")
    release_blockers = [
        artifact for artifact in by_id.values()
        if "support_blocker" in artifact
    ]
    check(len(release_blockers) == 6,
          "all six state-specific supported artifacts need blockers")

    names = {entry["name"] for entry in entries}
    wing_names = {
        entry["name"] for entry in entries
        if entry["family"] == "obiwan"
        and entry["selection"].startswith(("Ac_wings_", "Ae_wings_"))
    }
    check(wing_names == {
        "obiwan_05a_of_16_Ac_wing_LM_lower_left_1_of_3",
        "obiwan_06a_of_16_Ac_wing_LM_upper_left_2_of_3",
        "obiwan_07a_of_16_Ac_wing_UM_left_3_of_3",
        "obiwan_08a_of_16_Ac_wing_LM_lower_right_1_of_3",
        "obiwan_09a_of_16_Ac_wing_LM_upper_right_2_of_3",
        "obiwan_10a_of_16_Ac_wing_UM_right_3_of_3",
        "obiwan_05b_of_16_Ac_wing_LM_lower_left_1_of_2",
        "obiwan_06b_of_16_Ac_wing_LM_UM_upper_left_2_of_2",
        "obiwan_08b_of_16_Ac_wing_LM_lower_right_1_of_2",
        "obiwan_09b_of_16_Ac_wing_LM_UM_upper_right_2_of_2",
        "obiwan_11a_of_16_Ae_wing_LM_lower_left_1_of_3",
        "obiwan_12a_of_16_Ae_wing_LM_upper_left_2_of_3",
        "obiwan_13a_of_16_Ae_wing_UM_left_3_of_3",
        "obiwan_14a_of_16_Ae_wing_LM_lower_right_1_of_3",
        "obiwan_15a_of_16_Ae_wing_LM_upper_right_2_of_3",
        "obiwan_16a_of_16_Ae_wing_UM_right_3_of_3",
        "obiwan_11b_of_16_Ae_wing_LM_lower_left_1_of_2",
        "obiwan_12b_of_16_Ae_wing_LM_UM_upper_left_2_of_2",
        "obiwan_14b_of_16_Ae_wing_LM_lower_right_1_of_2",
        "obiwan_15b_of_16_Ae_wing_LM_UM_upper_right_2_of_2",
    }, "Ac/Ae A/B left/right wing shelf names drifted")
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
        "obiwan_01a_02_03_04_LM_UM_1_of_1",
        "obiwan_16a_of_16_Ae_wing_UM_right_3_of_3",
        "obiwan_11b_of_16_Ae_wing_LM_lower_left_1_of_2",
        "obiwan_12b_of_16_Ae_wing_LM_UM_upper_left_2_of_2",
        "obiwan_14b_of_16_Ae_wing_LM_lower_right_1_of_2",
        "obiwan_15b_of_16_Ae_wing_LM_UM_upper_right_2_of_2",
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
        if entry.get("composite_plate"):
            check(entry["name"] == shelf.combo.PLATE_NAME,
                  "unexpected composite shelf entry")
            contract = shelf.combo.validate_source_bundle(
                source, Path(entry["source_contract_path"]))
            check(contract["triangle_count"] == 56688,
                  "composite source triangle count drift")
            check(len(entry["composite_artifacts"]) == 3,
                  "composite source lacks three captive release bindings")

    check(release["inventory"]["artifact_count"] == 64,
          "unexpected canonical captive-magnet release inventory")
    manifest = json.loads(
        (shelf.DEFAULT_SHELF / "release_manifest.json").read_text(
            encoding="utf-8"))
    gate = manifest.get("project_stl_equivalence_gate")
    check(isinstance(gate, dict) and gate.get("status") == "pass",
          "shelf manifest lacks a passing project/STL equivalence gate")
    check(gate.get("required_pair_count") == 48
          and gate.get("passing_pair_count") == 48
          and len(gate.get("entries", ())) == 48,
          "shelf promotion did not cross a 48/48 equivalence gate")
    manifest_records = {
        record["name"]: record for record in manifest["entries"]
    }
    check(set(manifest_records) == names,
          "shelf manifest names differ from the catalog")
    combo_record = manifest_records[shelf.combo.PLATE_NAME]
    check(combo_record["project_kind"]
          == "local_composite_captive_magnet_slice",
          "composite project kind drift")
    check(combo_record["magnet_insertions"] == 6,
          "composite project must pause for six magnets")
    check(combo_record["placement_audit"]["normal_part_count"] == 4
          and combo_record["placement_audit"]["support_blocker_count"] == 3,
          "composite project must carry four parts and three duct blockers")
    combo_duct = combo_record["archive_audit"][
        "duct_support_toolpath_audit"]
    check(combo_duct["status"] == "pass"
          and combo_duct["collision_count"] == 0
          and len(combo_duct["parts"]) == 3,
          "composite support-vs-duct collision gate is not passing")
    combo_profile = combo_record["profile_effective"]
    check(all(combo_profile[key] is True for key in (
        "support_enabled",
        "support_on_build_plate_only",
        "support_critical_regions_only",
        "support_remove_small_overhang",
    )), "composite project does not pin all four support fields")
    object_support = combo_record["archive_audit"][
        "object_support_overrides"]
    check(len(object_support) == 1
          and all(object_support[0][key] == "1" for key in (
              "enable_support",
              "support_on_build_plate_only",
              "support_critical_regions_only",
              "support_remove_small_overhang",
          )), "composite object-level support fields are not all pinned")
    support_coverage = combo_record["archive_audit"][
        "support_midpoints_inside_part_footprints"]
    check(support_coverage[
              "obiwan_03_of_16_UM_carrier_1_of_1"] > 0,
          "composite UM carrier has floating-cantilever risk")
    check(support_coverage[
              "obiwan_04_of_16_T_tweeter_crescent_1_of_1"] == 0,
          "composite tweeter unexpectedly receives support")
    check(manifest["inventory"]["magnet_project_count"] == 39
          and manifest["inventory"]["non_magnet_project_count"] == 9
          and manifest["inventory"]["magnet_insertions"] == 62,
          "shelf inventory does not include the composite alternative")
    for entry in entries:
        record = manifest_records[entry["name"]]
        delivered_stl = ROOT / record["delivered_stl"]
        delivered_project = ROOT / record["p2s_project"]
        check(shelf._sha256(delivered_stl)
              == record["delivered_stl_sha256"],
              f"delivered STL hash drift: {entry['name']}")
        check(shelf._sha256(delivered_project)
              == record["p2s_project_sha256"],
              f"delivered project hash drift: {entry['name']}")
    print("to_print shelf catalog: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, shelf.ShelfError) as exc:
        print(f"test_to_print_shelf.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
