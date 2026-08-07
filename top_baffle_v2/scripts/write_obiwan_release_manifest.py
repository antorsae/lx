"""Write deterministic, hash-backed provenance for one Obi-Wan R6F state.

The release consists of many independently generated STEP, STL and PNG
files.  A timestamp alone cannot prove that a directory is one coherent
floor/no-floor build, so this manifest binds every required R6F artifact to
the exact generator sources and explicit stand state that produced it.

This module intentionally imports no CAD package and is safe to import from
the strict release checker.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)

from lx521_baffle.print_contract import sidecar_path_for_stl, validate_print_sidecar
from lx521_baffle.io import sha256_file


ROOT = PROJECT_ROOT
FORMAT_VERSION = 12
QUALIFICATION_RECORD = ROOT / "docs/obiwan_physical_qualification.md"
EXPECTED_KEYED_LM_DUCT_REGION_NAMES = {
    False: {
        "um_route_lumen",
        "t_route_lumen",
        "lm_internal_and_rear_exit_lumen",
        "no_floor_lm_entry_bore",
        "no_floor_t_entry_bore",
        "no_floor_um_entry_bore",
        "no_floor_t_entry_vestibule",
        "no_floor_um_entry_vestibule",
    },
    True: {
        "um_route_lumen",
        "t_route_lumen",
        "floor_lm_lane_lumen",
        "floor_um_lane_lumen",
        "floor_um_feed_relief",
        "floor_t_lane_lumen",
        "floor_t_feed_relief",
    },
}
EXPECTED_UM_CARRIER_DUCT_REGION_NAMES = {
    "um_carrier_t_route_lumen",
}
SUPPORT_BLOCKER_STEMS = (
    "obiwan_core_2_of_2_um_carrier",
    "obiwan_optional_lm_keyed_1_of_2_bottom",
    "obiwan_optional_lm_keyed_2_of_2_top",
)

REQUIRED_REFERENCE_PATHS = (
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_driver.stl",
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_driver_STL_notes.md",
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_Datasheet.pdf",
    ROOT.parent / "E0022_W22EX001.stp",
    QUALIFICATION_RECORD,
)


def generation_source_paths() -> tuple[Path, ...]:
    """Complete deterministic source set for the R6F release artifacts."""
    paths = set((ROOT / "src/lx521_baffle").rglob("*.py"))
    paths.update((ROOT / "scripts").glob("*.py"))
    paths.update((ROOT / "tests").glob("test_*.py"))
    required_generators = (
        ROOT / "cad-remote-requirements.lock",
        ROOT / "Makefile",
    )
    missing = [
        path for path in (*required_generators, *REQUIRED_REFERENCE_PATHS)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "required release source/reference(s) missing: "
            + ", ".join(str(path) for path in missing))
    paths.update(required_generators)
    paths.update(REQUIRED_REFERENCE_PATHS)
    return tuple(sorted(paths))


def source_hashes() -> dict[str, str]:
    records = {}
    for path in generation_source_paths():
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = "../" + path.relative_to(ROOT.parent).as_posix()
        records[name] = sha256_file(path)
    return records


def native_stage_record(
        state_dir: Path, stand_foot: bool, *,
        require_active_environment: bool = False) -> dict:
    """Validate and bind the native BREP transaction behind the artifacts."""
    from export_obiwan_staged import load_stage_manifest

    path = state_dir / ".obiwan_stage" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing required Obi-Wan native stage manifest: {path}")
    payload = load_stage_manifest(
        path, stand_foot=stand_foot,
        require_active_environment=require_active_environment)
    return {
        "path": ".obiwan_stage/manifest.json",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_sha256": payload["source_sha256"],
        "schema_version": payload["schema_version"],
        "guard_policy": payload["guard_policy"],
        "runtime_identity": payload["runtime_identity"],
    }


def expected_artifact_names(stand_foot: bool) -> tuple[str, ...]:
    review = (
        "obiwan_split.step",
        "obiwan_lm_split.step",
        "obiwan_attachments.step",
        "obiwan_assembled.step",
        "um_fit.step",
        "baffle_cable_routing_proud.png",
        "baffle_cable_routing_obiwan.png",
        "baffle_variants_drivers.png",
        "baffle_b1_drivers.png",
        "baffle_b2_drivers.png",
    )
    obiwan_stls = [
        "stl/obiwan_core_1_of_2_lm_carrier.stl",
        "stl/obiwan_core_2_of_2_um_carrier.stl",
        "stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl",
        "stl/obiwan_optional_lm_keyed_2_of_2_top.stl",
        "stl/obiwan_addon_tweeter_crescent.stl",
    ]
    # Print orientation is part of the release, not an informal slicer note.
    # Bind every installed Obi-Wan STL to the hash-backed X180/front-down plus
    # optional in-bed-Z transform emitted by export_piece_stls.py.  Include
    # the nonmagnetic tweeter crescent as well so all visible installed pieces
    # retain the same build-plate texture contract.
    obiwan_print_sidecars = [
        name.removesuffix(".stl") + ".print.json"
        for name in obiwan_stls
    ]
    strength_reports = (
        "obiwan_integrated_floor_strength.json",
        "obiwan_integrated_floor_strength.md",
    ) if stand_foot else ()
    support_blockers = tuple(
        f"support_blockers/{stem}.support_blocker.{suffix}"
        for stem in SUPPORT_BLOCKER_STEMS
        for suffix in ("stl", "json")
    )
    coupons = [
        f"stl/lx521_coupon_{name}.stl"
        for name in (
            "1_fit_plate",
            "2_fit_key",
            "3_fish_entry",
            "4_um_outlet_proud",
            "5_fish_ts_dive",
            "6_fish_foot",
            "7_recess_seat",
            "8_fish_ts_oval_proud",
            "9_um_faston_clocking",
            "12_obiwan_closed_bore_bump",
        )
    ]
    coupon_print_sidecars = [
        name.removesuffix(".stl") + ".print.json"
        for name in coupons
    ]
    return tuple(sorted((
        *review, *obiwan_stls, *obiwan_print_sidecars,
        *strength_reports, *support_blockers,
        *coupons, *coupon_print_sidecars)))


def qualification_record(stand_foot: bool) -> dict:
    """Current fail-closed, configuration-specific physical gate."""
    record = {
        "status": (
            "pending_physical_fit_and_structural_proof"
            if stand_foot else "pending_physical_fit"),
        "release_authorized": False,
        "physical_measure_required": True,
        "authorization_scope": "all_shipped_lm_print_forms",
        "record": QUALIFICATION_RECORD.relative_to(ROOT).as_posix(),
        "record_sha256": sha256_file(QUALIFICATION_RECORD),
        "reason": (
            "MU reference omits terminals; modeled 12 mm pull has "
            "zero positive release overtravel margin"
            + ("; integral floor geometry has only an analytical screen "
               "until its material-specific proof and creep tests pass"
               if stand_foot else "")),
        "configurations": {
            "monolithic_lm": {
                "release_authorized": False,
                "status": (
                    "pending_physical_fit_and_floor_load_test"
                    if stand_foot else "pending_physical_fit"),
            },
            "optional_lm_keyed_split": {
                "release_authorized": False,
                "status": "pending_two_pin_fit_and_installed_load_test",
                "registration_structural_load_credit_n": 0.0,
                "required_evidence": (
                    "front-face-down print proving both complete D1.60 pin "
                    "toolpaths, the exterior support lands, and the right "
                    "round and left X-relieved socket walls; simultaneous "
                    "straight insertion, actual U22 fit, full seating, "
                    "assembled front coplanarity, cable pull-through/route "
                    "continuity, and driver-installed 1g/3g/5g proof"),
            },
        },
    }
    if stand_foot:
        record["integral_floor_stand"] = {
            "analytical_report": "obiwan_integrated_floor_strength.json",
            "physical_status": "pending_proof_creep_and_anti_tip",
            "free_standing_use_authorized": False,
            "required_evidence": (
                "100% local-solid production print, 2x 24 h proof at 35 C, "
                "1.5x 168 h creep test, connector pull test and installed "
                "anti-tip/anchor verification"),
        }
    return record


def build_manifest(state_dir: Path, stand_foot: bool) -> dict:
    expected_state = "floor_stand" if stand_foot else "no_floor_stand"
    if state_dir.name != expected_state:
        raise RuntimeError(
            f"LX_STAND_FOOT selects {expected_state}, not {state_dir.name}")
    stage = native_stage_record(
        state_dir, stand_foot, require_active_environment=True)
    stage_mtime_ns = (
        state_dir / stage["path"]).stat().st_mtime_ns
    expected_names = expected_artifact_names(stand_foot)
    artifacts = {}
    for name in expected_names:
        path = state_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing required R6F artifact: {path}")
        if path.stat().st_mtime_ns < stage_mtime_ns:
            raise RuntimeError(
                f"R6F artifact predates its native stage transaction: "
                f"{path}")
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    # The manifest must never bless a self-consistent but stale or malformed
    # orientation sidecar. Every printable STL in this Obi-Wan transaction has
    # exactly one adjacent, hash-bound front-down authority record.
    stl_names = tuple(
        name for name in expected_names
        if name.startswith("stl/") and name.endswith(".stl"))
    expected_sidecars = {
        name.removesuffix(".stl") + ".print.json" for name in stl_names
    }
    actual_sidecars = {
        name for name in expected_names if name.endswith(".print.json")
    }
    if actual_sidecars != expected_sidecars:
        raise RuntimeError(
            "Obi-Wan manifest print-sidecar inventory is not exactly adjacent: "
            f"missing={sorted(expected_sidecars - actual_sidecars)} "
            f"extra={sorted(actual_sidecars - expected_sidecars)}")
    for name in stl_names:
        validate_print_sidecar(state_dir / name)
    for stem in SUPPORT_BLOCKER_STEMS:
        blocker_relative = (
            f"support_blockers/{stem}.support_blocker.stl")
        binding_relative = blocker_relative.removesuffix(".stl") + ".json"
        binding = json.loads(
            (state_dir / binding_relative).read_text(encoding="utf-8"))
        main_stl = state_dir / "stl" / f"{stem}.stl"
        blocker = state_dir / blocker_relative
        if (binding.get("schema_version") != 1
                or binding.get("kind") != "bambu_support_blocker"
                or binding.get("purpose")
                != "forbid_support_inside_functional_ducts"
                or binding.get("main_stl_sha256") != sha256_file(main_stl)
                or binding.get("support_blocker_sha256")
                != sha256_file(blocker)):
            raise RuntimeError(
                f"{stem}: duct support-blocker binding is stale")
        sidecar = json.loads(
            sidecar_path_for_stl(main_stl).read_text(encoding="utf-8"))
        if (binding.get("source_to_stl_matrix")
                != sidecar.get("source_to_stl_matrix")):
            raise RuntimeError(
                f"{stem}: duct support blocker uses a different print "
                "transform from its keyed LM part")
        collision = binding.get("duct_collision_contract")
        is_um_carrier = stem == "obiwan_core_2_of_2_um_carrier"
        expected_regions = (
            EXPECTED_UM_CARRIER_DUCT_REGION_NAMES
            if is_um_carrier
            else EXPECTED_KEYED_LM_DUCT_REGION_NAMES[stand_foot])
        if (not isinstance(collision, dict)
                or collision.get("schema_version") != 1
                or collision.get("coordinate_space")
                != "authoritative_source_mm"
                or (
                    collision.get("owner") != "um_carrier"
                    if is_um_carrier
                    else collision.get("split_seam_y_mm") != 172.481)
                or {
                    region.get("name")
                    for region in collision.get("regions", ())
                    if isinstance(region, dict)
                } != expected_regions):
            raise RuntimeError(
                f"{stem}: duct collision contract is missing or malformed")
    return {
        "format_version": FORMAT_VERSION,
        "variant": "Obi-Wan",
        "routing_revision": "R6F",
        "routing_profile": "obiwan",
        "state": expected_state,
        "stand_foot": stand_foot,
        "qualification": qualification_record(stand_foot),
        "sources": source_hashes(),
        "native_stage": stage,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args()
    mode = os.environ.get("LX_STAND_FOOT")
    if mode not in {"0", "1"}:
        parser.error("LX_STAND_FOOT must be explicitly 0 or 1")
    if os.environ.get("LX_ROUTING_PROFILE") != "obiwan":
        parser.error("LX_ROUTING_PROFILE must be obiwan")
    state_dir = args.state_dir.resolve()
    manifest = build_manifest(state_dir, mode == "1")
    output = state_dir / "obiwan_release_manifest.json"
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.tmp.json")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        if json.loads(temporary.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("temporary release manifest round-trip failed")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[obiwan manifest] wrote {output}")


if __name__ == "__main__":
    main()
