#!/usr/bin/env python3
"""Fail-closed validation for one sliced candidate BMR-crescent delivery.

This is ``validate_vase_tebm35c10_4_delivery.py`` for the two candidate
Obi-Wan BMR pods, plus the two things a candidate delivery owes that the
qualified vase does not:

* the slicing profile is re-derived from the base profile here and the
  delivery is rejected unless the file on disk is byte-identical to it and to
  the hash the exporter bound into the artifact facts, so a hand-edited
  profile cannot reach a printer through this path; and
* every pause Z is checked against the CAD closing plane pushed through the
  profile's own layer ladder, so the pause is a prediction the slice has to
  meet rather than whatever Z the slicer happened to choose.

The delivery is also required to stay a candidate: the artifact must be
absent from the released catalog and its facts must still say so.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from gen_bmr_crescent_slicing_profile import (
    DEFAULT_BASE,
    VARIANTS,
    generate,
    pause_layer_z,
)
from lx521_baffle.io import sha256_file
from release_validation import _validate_artifact_bindings, normalize_catalog


RELEASE_CATALOG = PROJECT_ROOT / "review" / "captive_magnet_release_catalog.json"
SUPPORT_KEYS = (
    "enable_support",
    "support_on_build_plate_only",
    "support_critical_regions_only",
    "support_remove_small_overhang",
)
EXPECTED_SITES = {"coaxial": 2, "opposed": 4}
INSERTION_DIRECTION_XYZ = (0.0, 0.0, -1.0)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _pin_profile(profile: Path, facts: dict, variant: str) -> str:
    """Re-derive the profile and refuse anything that is not byte-identical.

    The re-derivation lands beside the real file rather than in a scratch
    directory: the profile records its base as a path relative to its own
    parent, so comparing bytes is only meaningful from the same directory.
    """
    digest = sha256_file(profile)
    expected = profile.with_name(f".{profile.name}.rederived.tmp")
    try:
        generate(DEFAULT_BASE, expected, variant)
        if expected.read_bytes() != profile.read_bytes():
            raise RuntimeError(
                f"{profile} is not what gen_bmr_crescent_slicing_profile.py "
                f"derives for {variant!r} from {DEFAULT_BASE.name}")
    finally:
        expected.unlink(missing_ok=True)
    recorded = facts["delivery"]["slicing_profile"]
    if recorded.get("sha256") != digest:
        raise RuntimeError(
            "artifact facts are bound to a different slicing profile")
    if recorded.get("path") != profile.name:
        raise RuntimeError("artifact facts name a different profile file")
    return digest


def _expected_pause_groups(facts: dict, profile_payload: dict) -> list[dict]:
    """The CAD pause plan, recomputed here rather than trusted from facts."""
    plan = facts["delivery"]["pause_manifest"]["groups"]
    if not isinstance(plan, list) or not plan:
        raise RuntimeError("artifact facts carry no pause plan")
    groups = []
    for group in plan:
        plane = float(group["cavity_bury_roof_start_plane_z_mm"])
        expected_z = pause_layer_z(profile_payload, plane)
        if not math.isclose(
                expected_z, float(group["expected_pause_marker_z_mm"]),
                abs_tol=1.0e-9):
            raise RuntimeError(
                f"facts pause Z {group['expected_pause_marker_z_mm']} is not "
                f"this profile's first layer above {plane}")
        groups.append({
            "cavity_bury_roof_start_plane_z_mm": plane,
            "expected_pause_marker_z_mm": expected_z,
            "magnet_count": int(group["magnet_count"]),
            "sites": [str(name) for name in group["sites"]],
        })
    return sorted(groups, key=lambda item: item["expected_pause_marker_z_mm"])


def _assert_still_a_candidate(artifact_id: str, facts: dict) -> None:
    qualification = facts["qualification"]
    for key in ("release_authorized", "counts_against_release_inventory",
                "in_obiwan_stage_manifest", "in_to_print",
                "in_captive_magnet_release_catalog"):
        if qualification.get(key) is not False:
            raise RuntimeError(f"delivery claims release status via {key}")
    if qualification.get("has_captive_magnet_pause_delivery") is not True:
        raise RuntimeError("facts do not record the pause delivery")
    if RELEASE_CATALOG.is_file():
        released = _load(RELEASE_CATALOG)
        if any(str(entry.get("id")) == artifact_id
               for entry in released.get("artifacts", ())):
            raise RuntimeError(
                f"{artifact_id} has entered the released catalog; this "
                "candidate delivery validator is no longer the right gate")


def validate(
    *, catalog: Path, facts: Path, profile: Path, audit: Path, project: Path,
    gcode: Path, variant: str = "coaxial",
) -> dict[str, object]:
    try:
        spec = VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown BMR crescent variant {variant!r}") from exc
    artifact_id = f"shared:{spec.release_variant}:{spec.part}"
    expected_sites = EXPECTED_SITES[variant]

    normalized = normalize_catalog(catalog, enforce_release_inventory=False)
    if len(normalized["artifacts"]) != 1:
        raise RuntimeError("candidate catalog is no longer one artifact")
    entry = normalized["artifacts"][0]
    if entry["id"] != artifact_id or len(entry["sites"]) != expected_sites:
        raise RuntimeError("candidate catalog identity/site inventory drifted")
    _validate_artifact_bindings(entry)

    facts_payload = _load(facts)
    if facts_payload.get("artifact") != artifact_id:
        raise RuntimeError("artifact facts describe a different part")
    _assert_still_a_candidate(artifact_id, facts_payload)
    profile_sha256 = _pin_profile(profile, facts_payload, variant)
    profile_payload = _load(profile)
    expected_groups = _expected_pause_groups(facts_payload, profile_payload)
    if sum(group["magnet_count"] for group in expected_groups) != (
            expected_sites):
        raise RuntimeError("the pause plan does not cover every station")

    record = _load(audit)
    if (record.get("id") != artifact_id
            or record.get("status") != "pass"
            or record.get("audit_mode") != "actual_p2s_slice"):
        raise RuntimeError("actual P2S candidate slice did not pass")
    # A re-export moves the STL but leaves the promoted project and its audit
    # agreeing with each other, so hash-binding the slice to the catalog's
    # current mesh is the only thing that stops a stale delivery revalidating.
    sliced_stl_sha256 = str(record.get("input", {}).get("stl_sha256", ""))
    if sliced_stl_sha256 != entry["stl_catalog_sha256"]:
        raise RuntimeError(
            "the promoted project was sliced from a different STL than the "
            "catalog binds; re-run the slicing target")
    if len(record.get("sites", ())) != expected_sites:
        raise RuntimeError(
            f"slice audit does not cover all {expected_sites} magnet sites")
    for item in record["sites"]:
        if not all(item.get(key) is True for key in (
                "loading_aperture_pass", "retaining_paths_pass",
                "roof_progression_pass", "regression_pass")):
            raise RuntimeError(
                "magnet site failed toolpath audit: "
                f"{item.get('site', {}).get('name')}")
        site = item.get("site", {})
        if not str(site.get("polarity_instruction", "")).strip():
            raise RuntimeError(f"{site.get('name')}: no polarity instruction")
        pole = site.get("installed_marked_pole_axis_xyz")
        if (not isinstance(pole, list) or len(pole) != 3
                or not math.isclose(
                    math.fsum(value * value for value in pole), 1.0,
                    abs_tol=1.0e-9)):
            raise RuntimeError(
                f"{site.get('name')}: marked-pole axis is not a unit vector")
        direction = site.get("print_insertion_direction_xyz")
        if (not isinstance(direction, list) or any(
                not math.isclose(actual, want, abs_tol=1.0e-9)
                for actual, want in zip(
                    direction, INSERTION_DIRECTION_XYZ, strict=False))):
            raise RuntimeError(
                f"{site.get('name')}: insertion is not straight down in print "
                f"space: {direction}")

    slicer = record.get("slicer", {})
    effective = slicer.get("effective_config", {})
    if any(str(effective.get(key)) != "0" for key in SUPPORT_KEYS):
        raise RuntimeError("G-code does not pin all four support fields off")
    ready = slicer.get("ready_project", {})
    if ready.get("status") != "pass":
        raise RuntimeError("ready G-code 3MF did not pass packaging audit")
    archive = ready.get("archive_audit", {})
    project_settings = archive.get("enforced_project_settings", {})
    if any(str(project_settings.get(key)) != "0" for key in SUPPORT_KEYS):
        raise RuntimeError("3MF project does not pin all four support fields")
    object_overrides = archive.get("object_support_overrides")
    if not isinstance(object_overrides, list) or len(object_overrides) != 1:
        raise RuntimeError("3MF lacks one exact normal-object support override")
    if any(str(object_overrides[0].get(key)) != "0" for key in SUPPORT_KEYS):
        raise RuntimeError("3MF normal object does not pin support off")

    support_summary = ready.get("support_toolpaths", {})
    if not isinstance(support_summary, dict) or any(
            int(value) != 0 for value in support_summary.values()):
        raise RuntimeError("support toolpaths exist in a support-free pod")
    support_gate = ready.get("duct_support_toolpath_audit", {})
    if (support_gate.get("status") != "pass"
            or support_gate.get("gate")
            != "support_disabled_no_support_feature_blocks"
            or int(support_gate.get("support_extrusion_segments_checked", -1))
            != 0
            or int(support_gate.get("collision_count", -1)) != 0):
        raise RuntimeError("support gate did not prove support disabled")

    mesh = ready.get("bambu_3mf_audit", {})
    if (int(mesh.get("triangle_count", 0)) <= 0
            or float(mesh.get("mesh_max_abs_error_mm", 1.0)) > 1.0e-5
            or int(mesh.get("support_blocker_count", -1)) != 0
            or int(mesh.get("parameter_modifier_count", -1)) != 0):
        raise RuntimeError("3MF/STL equivalence or modifier inventory failed")

    pause_events = archive.get("gcode_pause_events")
    if not isinstance(pause_events, list) or len(pause_events) != len(
            expected_groups):
        raise RuntimeError(
            f"the {expected_sites} cavities must produce exactly "
            f"{len(expected_groups)} insertion pause(s), got "
            f"{len(pause_events) if isinstance(pause_events, list) else '?'}")
    park_z = float(profile_payload["magnet_insertion_pause"]["park_z_mm"])
    observed = []
    for event, group in zip(
            sorted(pause_events, key=lambda item: float(item["z_mm"])),
            expected_groups, strict=True):
        actual_z = float(event["z_mm"])
        if not math.isclose(
                actual_z, group["expected_pause_marker_z_mm"], abs_tol=0.001):
            raise RuntimeError(
                f"pause Z {actual_z} is not the first layer above the "
                f"{group['cavity_bury_roof_start_plane_z_mm']} mm closing "
                f"plane ({group['expected_pause_marker_z_mm']})")
        if not math.isclose(float(event["park_z_mm"]), park_z, abs_tol=1.0e-9):
            raise RuntimeError(
                f"pause parks at {event['park_z_mm']}, not the profile's "
                f"{park_z}")
        if not math.isclose(
                float(event["restore_z_mm"]), actual_z, abs_tol=1.0e-9):
            raise RuntimeError("pause does not restore its exact layer Z")
        if str(event.get("command", "")).strip() != "M400 U1":
            raise RuntimeError(
                f"pause command is {event.get('command')!r}, not 'M400 U1'")
        if not (int(event["park_command_line_number"])
                < int(event["command_line_number"])
                < int(event["restore_command_line_number"])):
            raise RuntimeError("pause is not park, then pause, then restore")
        observed.append({
            "pause_marker_z_mm": actual_z,
            "park_z_mm": park_z,
            "magnet_count": group["magnet_count"],
            "sites": group["sites"],
        })

    if not project.is_file() or not gcode.is_file():
        raise RuntimeError("ready 3MF or extracted G-code is missing")
    if ready.get("project_3mf_sha256") != sha256_file(project):
        raise RuntimeError("ready-project hash differs from slice audit")
    if ready.get("gcode_sha256") != sha256_file(gcode):
        raise RuntimeError("ready G-code hash differs from slice audit")
    if not zipfile.is_zipfile(project):
        raise RuntimeError("ready project is not a valid 3MF/ZIP archive")
    with zipfile.ZipFile(project) as bundle:
        members = set(bundle.namelist())
    required_members = {
        "Metadata/project_settings.config",
        "Metadata/model_settings.config",
        "Metadata/custom_gcode_per_layer.xml",
        "Metadata/plate_1.gcode",
    }
    if not required_members <= members:
        raise RuntimeError(
            f"ready 3MF lacks members: {sorted(required_members - members)}")

    return {
        "artifact": artifact_id,
        "variant": variant,
        "release_status": "candidate_not_release_authorized",
        "catalog_sites": expected_sites,
        "slice_status": "pass",
        "support_fields_global": "all_zero",
        "support_fields_object": "all_zero",
        "support_toolpaths": "none",
        "stl_3mf_equivalence": "pass",
        "slicing_profile": str(profile),
        "slicing_profile_sha256": profile_sha256,
        "slicing_profile_rederived": "identical",
        "filament": profile_payload["filament"],
        "magnet_pause_groups": len(observed),
        "pause_events": observed,
        "project": str(project),
        "project_sha256": sha256_file(project),
        "gcode": str(gcode),
        "gcode_sha256": sha256_file(gcode),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--gcode", type=Path, required=True)
    parser.add_argument(
        "--review", type=Path,
        help="also write this JSON audit record for the review shelf")
    parser.add_argument(
        "--variant", choices=tuple(VARIANTS), default="coaxial")
    args = parser.parse_args()
    result = validate(
        catalog=args.catalog.expanduser().resolve(),
        facts=args.facts.expanduser().resolve(),
        profile=args.profile.expanduser().resolve(),
        audit=args.audit.expanduser().resolve(),
        project=args.project.expanduser().resolve(),
        gcode=args.gcode.expanduser().resolve(),
        variant=args.variant,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.review is not None:
        review = args.review.expanduser().resolve()
        review.parent.mkdir(parents=True, exist_ok=True)
        temporary = review.with_name(f".{review.name}.tmp")
        try:
            temporary.write_text(text + "\n", encoding="utf-8")
            temporary.replace(review)
        finally:
            temporary.unlink(missing_ok=True)
    print(text)


if __name__ == "__main__":
    main()
