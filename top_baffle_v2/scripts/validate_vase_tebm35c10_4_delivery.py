#!/usr/bin/env python3
"""Fail-closed validation for one sliced Stock/Slim TEBM vase delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from lx521_baffle.io import sha256_file
from release_validation import _validate_artifact_bindings, normalize_catalog


ARTIFACT_IDS = {
    "stock": "shared:Stock-TEBM35C10-4:vase_TEBM35C10-4",
    "slim": "shared:Slim-TEBM35C10-4:vase_TEBM35C10-4",
}
SUPPORT_KEYS = (
    "enable_support",
    "support_on_build_plate_only",
    "support_critical_regions_only",
    "support_remove_small_overhang",
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def validate(
    *, catalog: Path, audit: Path, project: Path, gcode: Path,
    profile: str = "stock",
) -> dict[str, object]:
    try:
        artifact_id = ARTIFACT_IDS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown TEBM vase profile {profile!r}") from exc
    normalized = normalize_catalog(catalog, enforce_release_inventory=False)
    if len(normalized["artifacts"]) != 1:
        raise RuntimeError("auxiliary catalog is no longer one artifact")
    artifact = normalized["artifacts"][0]
    if artifact["id"] != artifact_id or len(artifact["sites"]) != 4:
        raise RuntimeError("auxiliary catalog identity/site inventory drifted")
    _validate_artifact_bindings(artifact)

    record = _load(audit)
    if (record.get("id") != artifact_id
            or record.get("status") != "pass"
            or record.get("audit_mode") != "actual_p2s_slice"):
        raise RuntimeError("actual P2S vase slice did not pass")
    if len(record.get("sites", ())) != 4:
        raise RuntimeError("slice audit does not cover all four magnet sites")
    for item in record["sites"]:
        if not all(item.get(key) is True for key in (
                "loading_aperture_pass", "retaining_paths_pass",
                "roof_progression_pass", "regression_pass")):
            raise RuntimeError(
                "magnet site failed toolpath audit: "
                f"{item.get('site', {}).get('name')}")

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
        raise RuntimeError("support toolpaths exist in the duct-bearing vase")
    duct_gate = ready.get("duct_support_toolpath_audit", {})
    if (duct_gate.get("status") != "pass"
            or duct_gate.get("gate")
            != "support_disabled_no_support_feature_blocks"
            or int(duct_gate.get("support_extrusion_segments_checked", -1))
            != 0
            or int(duct_gate.get("collision_count", -1)) != 0):
        raise RuntimeError("support-vs-duct gate did not prove support disabled")

    mesh = ready.get("bambu_3mf_audit", {})
    if (int(mesh.get("triangle_count", 0)) <= 0
            or float(mesh.get("mesh_max_abs_error_mm", 1.0)) > 1.0e-5
            or int(mesh.get("support_blocker_count", -1)) != 0
            or int(mesh.get("parameter_modifier_count", -1)) != 0):
        raise RuntimeError("3MF/STL equivalence or modifier inventory failed")
    pause_events = archive.get("gcode_pause_events")
    if not isinstance(pause_events, list) or len(pause_events) != 1:
        raise RuntimeError(
            "the four same-height cavities must share one insertion pause")

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
        "catalog_sites": 4,
        "slice_status": "pass",
        "support_fields_global": "all_zero",
        "support_fields_object": "all_zero",
        "support_toolpaths": "none",
        "stl_3mf_equivalence": "pass",
        "magnet_pause_groups": 1,
        "project": str(project),
        "project_sha256": sha256_file(project),
        "gcode": str(gcode),
        "gcode_sha256": sha256_file(gcode),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--gcode", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=tuple(ARTIFACT_IDS), default="stock")
    args = parser.parse_args()
    result = validate(
        catalog=args.catalog.expanduser().resolve(),
        audit=args.audit.expanduser().resolve(),
        project=args.project.expanduser().resolve(),
        gcode=args.gcode.expanduser().resolve(),
        profile=args.profile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
