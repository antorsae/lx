"""Write deterministic, hash-backed provenance for one V1LF R6F state.

The release consists of many independently generated STEP, STL and PNG
files.  A timestamp alone cannot prove that a directory is one coherent
floor/no-floor build, so this manifest binds every required R6F artifact to
the exact generator sources and explicit stand state that produced it.

This module intentionally imports no CAD package and is safe to import from
the strict release checker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORMAT_VERSION = 7
QUALIFICATION_RECORD = ROOT / "V1LF_PHYSICAL_QUALIFICATION.md"

REQUIRED_REFERENCE_PATHS = (
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_driver.stl",
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_driver_STL_notes.md",
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_Datasheet.pdf",
    ROOT.parent / "E0022_W22EX001.stp",
    QUALIFICATION_RECORD,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def generation_source_paths() -> tuple[Path, ...]:
    """Complete deterministic source set for the R6F release artifacts."""
    paths = set(ROOT.glob("top_baffle_nd25fw4*.py"))
    required_generators = tuple(ROOT / name for name in (
            "export_steps.py",
            "export_v1lf_staged.py",
            "export_piece_stls.py",
            "export_coupon.py",
            "gen_cable_routing.py",
            "gen_driver_overlay.py",
            "run_memory_guarded.py",
            "check_manifold.py",
            "test_clearances.py",
            "test_v1lf_r6f.py",
            "test_remote_cad.py",
            "write_v1lf_release_manifest.py",
            "remote_cad.py",
            "cad-remote-requirements.lock",
            "Makefile"))
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
    from export_v1lf_staged import load_stage_manifest

    path = state_dir / ".v1lf_stage" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing required V1LF native stage manifest: {path}")
    payload = load_stage_manifest(
        path, stand_foot=stand_foot,
        require_active_environment=require_active_environment)
    return {
        "path": ".v1lf_stage/manifest.json",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_sha256": payload["source_sha256"],
        "schema_version": payload["schema_version"],
        "guard_policy": payload["guard_policy"],
        "runtime_identity": payload["runtime_identity"],
    }


def expected_artifact_names(stand_foot: bool) -> tuple[str, ...]:
    review = (
        "top_baffle_nd25fw4_v1lf_split.step",
        "top_baffle_nd25fw4_v1lf_lm_split.step",
        "top_baffle_nd25fw4_v1lf_attachments.step",
        "top_baffle_nd25fw4_v1lf_assembled.step",
        "top_baffle_nd25fw4_um_fit.step",
        "baffle_cable_routing_proud.png",
        "baffle_cable_routing_v1lf.png",
        "baffle_variants_drivers.png",
        "baffle_b1_drivers.png",
        "baffle_b2_drivers.png",
    )
    v1lf_stls = [
        "stl/lx521_top_v1lf_core_1of2_lm_carrier.stl",
        "stl/lx521_top_v1lf_core_2of2_um_carrier.stl",
        "stl/lx521_top_v1lf_optional_lm_keyed_1of2_bottom.stl",
        "stl/lx521_top_v1lf_optional_lm_keyed_2of2_top.stl",
        "stl/lx521_top_v1lf_addon_tweeter_crescent.stl",
    ]
    strength_reports = (
        "v1lf_integrated_floor_strength.json",
        "v1lf_integrated_floor_strength.md",
    ) if stand_foot else ()
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
            "12_v1lf_closed_bore_bump",
        )
    ]
    return tuple(sorted((
        *review, *v1lf_stls, *strength_reports, *coupons)))


def qualification_record(stand_foot: bool) -> dict:
    """Current fail-closed, configuration-specific physical gate."""
    record = {
        "status": (
            "pending_physical_fit_and_structural_proof"
            if stand_foot else "pending_physical_fit"),
        "release_authorized": False,
        "physical_measure_required": True,
        "authorization_scope": "all_shipped_lm_print_forms",
        "record": QUALIFICATION_RECORD.name,
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
                "status": "pending_key_fit_and_installed_load_test",
                "registration_structural_load_credit_n": 0.0,
                "required_evidence": (
                    "printed key/socket coupon, cable pull-through, and "
                    "driver-installed 1g/3g/5g proof"),
            },
        },
    }
    if stand_foot:
        record["integral_floor_stand"] = {
            "analytical_report": "v1lf_integrated_floor_strength.json",
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
    artifacts = {}
    for name in expected_artifact_names(stand_foot):
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
    return {
        "format_version": FORMAT_VERSION,
        "variant": "V1LF",
        "routing_revision": "R6F",
        "routing_profile": "v1lf",
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
    if os.environ.get("LX_ROUTING_PROFILE") != "v1lf":
        parser.error("LX_ROUTING_PROFILE must be v1lf")
    state_dir = args.state_dir.resolve()
    manifest = build_manifest(state_dir, mode == "1")
    output = state_dir / "v1lf_release_manifest.json"
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
    print(f"[v1lf manifest] wrote {output}")


if __name__ == "__main__":
    main()
