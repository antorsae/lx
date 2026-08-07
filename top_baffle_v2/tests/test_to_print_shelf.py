#!/usr/bin/env python3
"""Pure-Python contract checks for the friendly P2S delivery shelf."""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
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
    remote_contract_only = (
        os.environ.get("LX_CAD_EXECUTION") == "remote-worker"
    )
    graph_env = os.environ.copy()
    # Pure dependency-graph inspection is allowed on a remote Linux worker,
    # but the Makefile accepts this escape hatch only while GNU Make itself is
    # in -n/--just-print mode *and* every exceptional recursive-Make recipe is
    # neutralized by /usr/bin/true.  No slicer recipe can execute.
    graph_env["LX_BAMBU_GRAPH_ONLY"] = "1"

    def make_graph(target: str, *flags: str) -> str:
        ignored_missing: list[str] = []
        for _attempt in range(8):
            command = ["make", *flags]
            if remote_contract_only:
                command.append("-k")
            command.append("SHELL=/usr/bin/true")
            for missing in ignored_missing:
                command.extend(("-o", missing))
            command.append(target)
            run = subprocess.run(
                command,
                cwd=ROOT,
                env=graph_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if run.returncode == 0:
                return run.stdout
            missing_paths = re.findall(
                r"No rule to make target '([^']+)'", run.stdout)
            check(remote_contract_only and missing_paths,
                  f"make {' '.join(flags)} {target} failed: "
                  f"{run.stdout[-2000:]}")
            allowed_root = (
                ROOT / "review/captive_magnet_slice_audit/slices"
            ).resolve()
            fatal_lines = [
                line for line in run.stdout.splitlines() if "***" in line
            ]
            check(all("No rule to make target" in line
                      for line in fatal_lines),
                  f"unexpected remote graph error for {target}: "
                  f"{run.stdout[-2000:]}")
            shelf_manifest = (ROOT / "to_print/release_manifest.json").resolve()
            new_missing = []
            for raw in missing_paths:
                path = Path(raw).resolve()
                # The shelf's release manifest is excluded from remote
                # snapshots (a local shelf refresh rewrites it and must not
                # bust the remote make cache), so a remote worker tree
                # legitimately lacks it.
                check((path.name == "captive_magnet_slice_audit.json"
                       and path.is_relative_to(allowed_root))
                      or path == shelf_manifest,
                      f"remote graph has an unexpected missing input: {raw}")
                if raw not in ignored_missing:
                    new_missing.append(raw)
            check(new_missing,
                  f"remote graph made no progress for {target}: "
                  f"{run.stdout[-2000:]}")
            ignored_missing.extend(new_missing)
        raise AssertionError(
            f"remote graph required too many missing-input passes: {target}")

    def make_dry_run(target: str) -> str:
        output = make_graph(target, "-nB")
        check("scripts/remote_cad.py" not in output
              and "osado.lan" not in output
              and not any(line.lstrip().startswith("ssh ")
                          for line in output.splitlines()),
              f"{target} must not dispatch CAD or slicing to osado")
        return output

    shelf_dry_run = make_dry_run("to_print")
    check("--emit-ready-projects" not in shelf_dry_run
          and "scripts/slice_captive_magnets.py" not in shelf_dry_run,
          "to_print must consume existing release projects, not implicitly "
          "run the heavyweight captive-magnet release slicer")
    for target in (
        "obiwan_no_floor_petg_gf_01a",
        "obiwan_floor_petg_gf_01b",
    ):
        structural_build = make_dry_run(target)
        check(
            0 <= structural_build.find("--dry-run")
            < structural_build.find("--emit-ready-projects"),
            f"{target} must dry-run before local structural slicing",
        )
    for target, friendly_name in (
        (
            "obiwan_no_floor_petg_gf_01a_to_print",
            "obiwan_01_LM_bottom_keyed_1_of_2_no_floor_stand",
        ),
        (
            "obiwan_floor_petg_gf_01b_to_print",
            "obiwan_01_LM_bottom_keyed_1_of_2_floor_stand",
        ),
    ):
        structural_promotion = make_dry_run(target)
        structural_shelf_command = structural_promotion.rfind(
            "scripts/build_to_print_shelf.py")
        check(
            structural_shelf_command >= 0
            and "--validate-only"
            in structural_promotion[structural_shelf_command:]
            and f'--only "{friendly_name}"'
            in structural_promotion[structural_shelf_command:],
            f"targeted PETG-GF promotion {target} must disable slicing and "
            "cross the complete-shelf validation barrier",
        )
    for make_slug, api in (
            ("no_floor", shelf.NO_FLOOR_COMBO_PLATE),
            ("floor", shelf.FLOOR_COMBO_PLATE)):
        combo_dry_run = make_dry_run(
            f"obiwan_{make_slug}_combo_plate")
        check(0 <= combo_dry_run.find("--dry-run")
              < combo_dry_run.find("--slice-missing"),
              f"{make_slug}: composite artifact must dry-run before slicing")
        promotion_dry_run = make_dry_run(
            f"obiwan_{make_slug}_combo_plate_to_print")
        shelf_command = promotion_dry_run.rfind(
            "scripts/build_to_print_shelf.py")
        check(shelf_command >= 0
              and "--validate-only" in promotion_dry_run[shelf_command:]
              and f'--only "{api.PLATE_NAME}"'
              in promotion_dry_run[shelf_command:],
              f"targeted {make_slug} composite promotion must disable "
              "slicing and cross the complete-shelf validation barrier")
    for slug in ("ac", "ae"):
        wing_plate_dry_run = make_dry_run(
            f"obiwan_{slug}_wing_plate")
        check(0 <= wing_plate_dry_run.find("--dry-run")
              < wing_plate_dry_run.find("--slice-missing"),
              f"{slug}: wing-plate artifact must dry-run before local slicing")
    for slug, api in (
            ("ac", shelf.AC_WING_PLATE),
            ("ae", shelf.AE_WING_PLATE)):
        wing_promotion_dry_run = make_dry_run(
            f"obiwan_{slug}_wing_plate_to_print")
        wing_shelf_command = wing_promotion_dry_run.rfind(
            "scripts/build_to_print_shelf.py")
        check(wing_shelf_command >= 0
              and "--validate-only"
              in wing_promotion_dry_run[wing_shelf_command:]
              and f'--only "{api.PLATE_NAME}"'
              in wing_promotion_dry_run[wing_shelf_command:],
              f"targeted {slug} wing-plate promotion must disable slicing "
              "and cross the complete-shelf validation barrier")

    plate_apis = (
        shelf.NO_FLOOR_COMBO_PLATE,
        shelf.FLOOR_COMBO_PLATE,
        shelf.AC_WING_PLATE,
        shelf.AE_WING_PLATE,
    )
    concrete_targets = tuple(
        target
        for api in plate_apis
        for target in (
            f"build/print_plates/obiwan/{api.PLATE_NAME}.stl",
            "review/to_print_slice_workspace/composite/"
            f"{api.PLATE_NAME}/ready/{api.PLATE_NAME}.gcode.3mf",
            f"to_print/obiwan/3mf/{api.PLATE_NAME}.gcode.3mf",
        )
    )
    for target in concrete_targets:
        output = make_dry_run(target)
        expected_builder = (
            "build_obiwan_wing_plate.py"
            if any(api.PLATE_NAME in target for api in (
                shelf.AC_WING_PLATE, shelf.AE_WING_PLATE))
            else "build_obiwan_combo_plate.py"
        )
        check(expected_builder in output,
              f"{target} is not backed by the composite artifact graph")
        database_output = make_graph(target, "-np")
        marker = f"\n{target}:"
        start = database_output.find(marker)
        found = start >= 0
        if not found and database_output.startswith(f"{target}:"):
            start = -1
            found = True
        check(found, f"missing concrete Make record for {target}")
        record = database_output[start + 1:].split("\n\n", 1)[0]
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

    if remote_contract_only:
        raw = json.loads(
            shelf.DEFAULT_CATALOG.read_text(encoding="utf-8"))
        raw_entries = raw.get("entries") if isinstance(raw, dict) else None
        check(raw.get("schema_version") == 1
              and raw.get("catalog_kind") == "lx521_p2s_print_shelf"
              and raw.get("printer") == "Bambu Lab P2S"
              and isinstance(raw_entries, list),
              "remote shelf catalog header is invalid")
        check(len(raw_entries) == 51,
              "remote shelf catalog must contain exactly 51 entries")
        names = []
        families = {family: 0 for family in shelf.EXPECTED_FAMILY_COUNTS}
        for index, entry in enumerate(raw_entries):
            check(isinstance(entry, dict),
                  f"remote catalog entry {index} is not an object")
            name = entry.get("name")
            family = entry.get("family")
            source = Path(str(entry.get("source_stl", "")))
            check(isinstance(name, str) and shelf.NAME_RE.fullmatch(name),
                  f"remote catalog entry {index} has an invalid name")
            check(family in families,
                  f"remote catalog entry {index} has an invalid family")
            check(not source.is_absolute() and ".." not in source.parts
                  and source.suffix.lower() == ".stl",
                  f"remote catalog entry {index} has an unsafe source")
            check(not any(token in source.as_posix().lower()
                          for token in shelf.UNPRINTABLE_OR_LEGACY_TOKENS),
                  f"remote catalog entry {index} selects legacy geometry")
            names.append(name)
            families[family] += 1
        check(len(set(names)) == len(names),
              "remote shelf catalog contains duplicate names")
        check(families == {"stock": 11, "slim": 11, "obiwan": 29},
              f"remote shelf family counts drifted: {families}")
        check(sum(shelf._is_magnet_entry(entry)
                  for entry in raw_entries) == 42,
              "remote shelf magnet-project count drifted")
        check(sum(entry.get("composite_plate") is not None
                  for entry in raw_entries) == 4,
              "remote shelf composite-plate count drifted")
        release, by_id = shelf._release_artifacts(
            shelf.DEFAULT_RELEASE_CATALOG)
        check(len(by_id) == 58
              and release["inventory"]["artifact_count"] == 58,
              "remote captive-magnet release inventory drifted")
        referenced = {
            entry["catalog_artifact_id"] for entry in raw_entries
            if entry.get("catalog_artifact_id")
        }
        check(referenced <= set(by_id),
              "remote shelf references an unknown release artifact")
        release_blockers = [
            artifact for artifact in by_id.values()
            if "support_blocker" in artifact
        ]
        check(len(release_blockers) == 6
              and all(Path(artifact["support_blocker"]).is_file()
                      for artifact in release_blockers),
              "remote release lacks all six duct support blockers")
        print(
            "to_print remote contracts: neutralized Make graph, 51-entry "
            "shelf catalog, and 58-artifact release catalog pass; "
            "project/STL equivalence remains local-only"
        )
        return 0

    raw, entries = shelf._catalog_entries(shelf.DEFAULT_CATALOG)
    release, by_id = shelf._release_artifacts(shelf.DEFAULT_RELEASE_CATALOG)
    shelf._bind_entries_to_release(entries, by_id)

    check(raw["printer"] == "Bambu Lab P2S", "catalog printer drift")
    check(
        shelf.COMPOSITE_SPECS[
            shelf.NO_FLOOR_COMBO_PLATE.PLATE_NAME]["profile_path"]
        == shelf.PETG_GF_PROFILE
        and shelf.COMPOSITE_SPECS[
            shelf.FLOOR_COMBO_PLATE.PLATE_NAME]["profile_path"]
        == shelf.PETG_GF_PROFILE,
        "both combined core plates must use the structural PETG-GF profile",
    )
    check(
        "profile_path" not in shelf.COMPOSITE_SPECS[
            shelf.AC_WING_PLATE.PLATE_NAME]
        and "profile_path" not in shelf.COMPOSITE_SPECS[
            shelf.AE_WING_PLATE.PLATE_NAME],
        "Ac/Ae wing plates must remain on the standard non-GF profile",
    )
    check(
        set(shelf.PETG_GF_RELEASE_AUDITS) == {
            shelf.PETG_GF_01A_ARTIFACT_ID,
            shelf.PETG_GF_01B_ARTIFACT_ID,
        },
        "both standalone keyed LM bottoms must use structural PETG-GF audits",
    )
    check(len(entries) == 51, "shelf must contain exactly 51 entries")
    families = {
        family: sum(entry["family"] == family for entry in entries)
        for family in shelf.EXPECTED_FAMILY_COUNTS
    }
    check(families == {"stock": 11, "slim": 11, "obiwan": 29},
          f"unexpected family counts: {families}")
    magnetic = [entry for entry in entries if shelf._is_magnet_entry(entry)]
    check(len(magnetic) == 42, "expected 42 audited magnet projects")
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
            "obiwan_optional_lm_keyed_1_of_2_bottom."
            "support_blocker.stl",
        ),
        (
            "floor_stand",
            "obiwan_optional_lm_keyed_1_of_2_bottom."
            "support_blocker.stl",
        ),
        (
            "no_floor_stand",
            "obiwan_optional_lm_keyed_2_of_2_top."
            "support_blocker.stl",
        ),
        (
            "no_floor_stand",
            "obiwan_core_2_of_2_um_carrier."
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
        "obiwan_05_split3_Ac_wing_LM_lower_left_1_of_3",
        "obiwan_06_split3_Ac_wing_LM_upper_left_2_of_3",
        "obiwan_07_split3_Ac_wing_UM_left_3_of_3",
        "obiwan_08_split3_Ac_wing_LM_lower_right_1_of_3",
        "obiwan_09_split3_Ac_wing_LM_upper_right_2_of_3",
        "obiwan_10_split3_Ac_wing_UM_right_3_of_3",
        "obiwan_05_split2_Ac_wing_LM_lower_left_1_of_2",
        "obiwan_06_split2_Ac_wing_LM_UM_upper_left_2_of_2",
        "obiwan_08_split2_Ac_wing_LM_lower_right_1_of_2",
        "obiwan_09_split2_Ac_wing_LM_UM_upper_right_2_of_2",
        "obiwan_ac_wings_split2_combo",
        "obiwan_11_split3_Ae_wing_LM_lower_left_1_of_3",
        "obiwan_12_split3_Ae_wing_LM_upper_left_2_of_3",
        "obiwan_13_split3_Ae_wing_UM_left_3_of_3",
        "obiwan_14_split3_Ae_wing_LM_lower_right_1_of_3",
        "obiwan_15_split3_Ae_wing_LM_upper_right_2_of_3",
        "obiwan_16_split3_Ae_wing_UM_right_3_of_3",
        "obiwan_11_split2_Ae_wing_LM_lower_left_1_of_2",
        "obiwan_12_split2_Ae_wing_LM_UM_upper_left_2_of_2",
        "obiwan_14_split2_Ae_wing_LM_lower_right_1_of_2",
        "obiwan_15_split2_Ae_wing_LM_UM_upper_right_2_of_2",
        "obiwan_ae_wings_split2_combo",
    }, "Ac/Ae A/B left/right wing shelf names drifted")
    for required in (
        "stock_01_LM_bottom_1_of_3_no_floor_stand",
        "stock_01_LM_bottom_1_of_3_floor_stand",
        "stock_03_LM_mid_right_3_of_3",
        "slim_01_LM_bottom_1_of_3_no_floor_stand",
        "slim_03_LM_mid_right_3_of_3",
        "obiwan_01_LM_bottom_keyed_1_of_2_no_floor_stand",
        "obiwan_01_LM_bottom_keyed_1_of_2_floor_stand",
        "obiwan_02_LM_top_keyed_2_of_2",
        "obiwan_03_UM_carrier_1_of_1",
        "obiwan_04_T_tweeter_crescent_1_of_1",
        "obiwan_01_02_03_04_LM_UM_combo_no_floor_stand",
        "obiwan_01_02_03_04_LM_UM_combo_floor_stand",
        "obiwan_ac_wings_split2_combo",
        "obiwan_16_split3_Ae_wing_UM_right_3_of_3",
        "obiwan_11_split2_Ae_wing_LM_lower_left_1_of_2",
        "obiwan_12_split2_Ae_wing_LM_UM_upper_left_2_of_2",
        "obiwan_14_split2_Ae_wing_LM_lower_right_1_of_2",
        "obiwan_15_split2_Ae_wing_LM_UM_upper_right_2_of_2",
        "obiwan_ae_wings_split2_combo",
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
    check(not any("core_1_of_2_lm_carrier" in entry["source_stl"]
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
            module = entry["composite_spec"]["module"]
            check(entry["name"] in shelf.COMPOSITE_SPECS,
                  "unexpected composite shelf entry")
            contract = module.validate_source_bundle(
                source, Path(entry["source_contract_path"]))
            expected_bindings = (
                3 if entry["name"] in {
                    shelf.NO_FLOOR_COMBO_PLATE.PLATE_NAME,
                    shelf.FLOOR_COMBO_PLATE.PLATE_NAME,
                } else 4
            )
            check(contract["triangle_count"]
                  == module.EXPECTED_TRIANGLE_COUNT,
                  f"{entry['name']}: composite source triangle count drift")
            check(len(entry["composite_artifacts"]) == expected_bindings,
                  f"{entry['name']}: captive release bindings are incomplete")

    check(release["inventory"]["artifact_count"] == 58,
          "unexpected canonical captive-magnet release inventory")
    manifest = json.loads(
        (shelf.DEFAULT_SHELF / "release_manifest.json").read_text(
            encoding="utf-8"))
    gate = manifest.get("project_stl_equivalence_gate")
    check(isinstance(gate, dict) and gate.get("status") == "pass",
          "shelf manifest lacks a passing project/STL equivalence gate")
    check(gate.get("required_pair_count") == 51
          and gate.get("passing_pair_count") == 51
          and len(gate.get("entries", ())) == 51,
          "shelf promotion did not cross a 51/51 equivalence gate")
    manifest_records = {
        record["name"]: record for record in manifest["entries"]
    }
    check(set(manifest_records) == names,
          "shelf manifest names differ from the catalog")
    for label, api, expected_infill in (
            (
                "no-floor", shelf.NO_FLOOR_COMBO_PLATE,
                (40.0, "gyroid"),
            ),
            (
                "floor-stand", shelf.FLOOR_COMBO_PLATE,
                (100.0, "zig-zag"),
            )):
        combo_record = manifest_records[api.PLATE_NAME]
        check(combo_record["project_kind"]
              == "local_composite_captive_magnet_slice",
              f"{label}: composite project kind drift")
        check(combo_record["magnet_insertions"] == 6,
              f"{label}: composite project must pause for six magnets")
        check(combo_record["placement_audit"]["normal_part_count"] == 4
              and combo_record["placement_audit"][
                  "support_blocker_count"] == 3,
              f"{label}: composite must carry four parts and three blockers")
        combo_duct = combo_record["archive_audit"][
            "duct_support_toolpath_audit"]
        check(combo_duct["status"] == "pass"
              and combo_duct["collision_count"] == 0
              and len(combo_duct["parts"]) == 3,
              f"{label}: support-vs-duct collision gate is not passing")
        combo_profile = combo_record["profile_effective"]
        check((
            combo_profile["sparse_infill_density_percent"],
            combo_profile["sparse_infill_pattern"],
        ) == expected_infill,
              f"{label}: combined-plate infill profile drifted")
        check(all(combo_profile[key] is True for key in (
            "support_enabled",
            "support_on_build_plate_only",
            "support_critical_regions_only",
            "support_remove_small_overhang",
        )), f"{label}: project does not pin all support fields")
        object_support = combo_record["archive_audit"][
            "object_support_overrides"]
        check(len(object_support) == 1
              and all(object_support[0][key] == "1" for key in (
                  "enable_support",
                  "support_on_build_plate_only",
                  "support_critical_regions_only",
                  "support_remove_small_overhang",
              )), f"{label}: object support fields are not all pinned")
        support_coverage = combo_record["archive_audit"][
            "support_midpoints_inside_part_footprints"]
        check(support_coverage[
                  "obiwan_03_UM_carrier_1_of_1"] > 0,
              f"{label}: UM carrier has floating-cantilever risk")
        check(support_coverage[
                  "obiwan_04_T_tweeter_crescent_1_of_1"] == 0,
              f"{label}: tweeter unexpectedly receives support")
        check(
            combo_profile["filament"]
            == "TINMORRY PETG-GF Profile @BBL P2S"
            and combo_profile["wall_loops"] == 8,
            f"{label}: combined core plate must use PETG-GF and eight walls",
        )
        expected_modifier_count = 1 if label == "no-floor" else 0
        check(
            combo_record["placement_audit"]["parameter_modifier_count"]
            == expected_modifier_count,
            f"{label}: bridge/root parameter-modifier count drifted",
        )
    structural_01a = manifest_records[
        "obiwan_01_LM_bottom_keyed_1_of_2_no_floor_stand"]
    check(
        structural_01a["profile_effective"]["filament"]
        == "TINMORRY PETG-GF Profile @BBL P2S"
        and structural_01a["profile_effective"]["wall_loops"] == 8
        and structural_01a["placement_audit"]["parameter_modifier_count"] == 1,
        "standalone no-floor 01a must use PETG-GF, eight walls, and one "
        "100%-solid bridge/root modifier",
    )
    structural_01b = manifest_records[
        "obiwan_01_LM_bottom_keyed_1_of_2_floor_stand"]
    check(
        structural_01b["profile_effective"]["filament"]
        == "TINMORRY PETG-GF Profile @BBL P2S"
        and structural_01b["profile_effective"]["wall_loops"] == 8
        and structural_01b["profile_effective"][
            "sparse_infill_density_percent"] == 100.0
        and structural_01b["profile_effective"][
            "sparse_infill_pattern"] == "zig-zag"
        and structural_01b["placement_audit"][
            "parameter_modifier_count"] == 0,
        "standalone floor 01b must use PETG-GF, eight walls, global 100% "
        "zig-zag, and no local parameter modifier",
    )
    for label, api in (
            ("Ac", shelf.AC_WING_PLATE),
            ("Ae", shelf.AE_WING_PLATE)):
        wing_record = manifest_records[api.PLATE_NAME]
        check(wing_record["project_kind"] == "local_locked_wing_plate_slice"
              and wing_record["magnet_insertions"] == 6,
              f"{label} wing-plate project identity or pause drifted")
        check(wing_record["placement_audit"]["normal_part_count"] == 4
              and wing_record["placement_audit"][
                  "support_blocker_count"] == 0,
              f"{label} wing plate must contain four parts and no blockers")
        wing_duct = wing_record["archive_audit"][
            "duct_support_toolpath_audit"]
        check(wing_duct["status"] == "pass"
              and wing_duct["collision_count"] == 0,
              f"{label} wing-plate support-toolpath gate is not passing")
        wing_profile = wing_record["profile_effective"]
        check(
            wing_profile["filament"] == "Bambu PLA Tough+ @BBL P2S"
            and wing_profile["wall_loops"] == 6,
            f"{label} wing plate must remain PLA Tough+ with six walls",
        )
        check(all(wing_profile[key] is False for key in (
            "support_enabled",
            "support_on_build_plate_only",
            "support_critical_regions_only",
            "support_remove_small_overhang",
        )), f"{label} wing plate does not pin all support fields off")
        wing_object_support = wing_record["archive_audit"][
            "object_support_overrides"]
        check(len(wing_object_support) == 1
              and all(wing_object_support[0][key] == "0" for key in (
                  "enable_support",
                  "support_on_build_plate_only",
                  "support_critical_regions_only",
                  "support_remove_small_overhang",
              )), f"{label} object support fields are not all pinned off")
        wing_packing = wing_record["archive_audit"]["source_packing"]
        check(wing_packing["minimum_actual_xy_gap_mm"] >= 3.5
              and wing_packing["minimum_actual_bed_edge_mm"] >= 3.5,
              f"{label} wing-plate packing clearance drifted")
        check(len(wing_record["archive_audit"][
                  "captive_cavity_audit"]) == 4,
              f"{label} wing plate lacks four cavity-toolpath audits")
    for wing_name in wing_names:
        wing_profile = manifest_records[wing_name]["profile_effective"]
        check(
            wing_profile["filament"] == "Bambu PLA Tough+ @BBL P2S"
            and wing_profile["wall_loops"] == 6,
            f"{wing_name}: wing/shoulder material must remain non-GF PLA",
        )
    check(manifest["inventory"]["magnet_project_count"] == 42
          and manifest["inventory"]["non_magnet_project_count"] == 9
          and manifest["inventory"]["magnet_insertions"] == 80,
          "shelf inventory does not include all four plate alternatives")
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
