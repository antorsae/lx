#!/usr/bin/env python3
"""Publish the complete 0.6-mm high-flow lane onto the P2S shelf.

Two sources feed ``to_print/<family>/3mf_06hf/``:

* every shelf entry bound to a release-catalog artifact republishes its
  *already audited* 0.6-mm ready project from
  ``review/captive_magnet_slice_audit_06hf/slices/<slug>/ready/``;
* every plain entry without captive magnets (LM bottoms, mids, the tweeter
  crescent) is sliced here under the pinned 0.6-mm PLA Basic profile and
  validated with the canonical shelf builder's own result/mesh/bed gates --
  there is no pause to embed, so a profile- and hash-bound plain slice is
  the complete deliverable.

Still outside this lane view: the two PLA wing combo plates (0.4-mm lane
deliveries), the two auxiliary BMR crescent candidates, and the four
``lane: 06hf`` structural PETG-GF entries -- the canonical builder already
delivers those directly into ``3mf_06hf`` from the PETG-GF 0.6 outputs.

Fail-closed: every artifact-bound entry must have a passing 0.6 audit and
every plain slice must pass the result/mesh/bed gates, or nothing is
published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import build_to_print_shelf as canonical
from build_to_print_shelf import ShelfError

DEFAULT_SHELF = PROJECT_ROOT / "to_print"
DEFAULT_AUDIT = PROJECT_ROOT / "review" / "captive_magnet_slice_audit_06hf"
LANE_PROFILE = PROJECT_ROOT / "captive_magnet_slicing_profile_06hf.json"
LANE_WORKSPACE = PROJECT_ROOT / "review" / "to_print_slice_workspace_06hf"
LANE = "0.6mm_high_flow"
MANIFEST_SCHEMA_VERSION = 2
# Shipping the process under a name that exists in no Bambu Studio install
# means the GUI's remembered "same-name preset conflict -> use installed"
# choice can never substitute base values for the project's pinned ones:
# with no installed preset to collide with, Studio keeps the embedded
# settings and the panel shows the real Arachne/0.16 configuration.
LOCKED_PROCESS_ID = "LX521 captive 0.6HF (locked - do not re-slice)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(source: Path, family_dir: Path, name: str) -> Path:
    """Copy one ready project, lock its process identity, and freeze it.

    The G-code member is verified byte-identical to the audited source, the
    process preset id is renamed to LOCKED_PROCESS_ID inside
    project_settings.config only, and the delivered file is made read-only
    so a Bambu Studio session can never save its own modified state back
    over the audited shelf copy (observed: a GUI save stripped the embedded
    G-code and baked classic/0.18 into the shelf file).
    """
    family_dir.mkdir(parents=True, exist_ok=True)
    destination = family_dir / f"{name}_06hf.gcode.3mf"
    if destination.exists():
        destination.chmod(stat.S_IWUSR | stat.S_IRUSR)
        destination.unlink()
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if "Metadata/plate_1.gcode" not in names:
            raise ShelfError(
                f"{name}: audited source project lacks embedded G-code")
        source_gcode = archive.read("Metadata/plate_1.gcode")
        payloads = {
            member.filename: archive.read(member.filename)
            for member in members
        }
    settings = json.loads(
        payloads["Metadata/project_settings.config"].decode("utf-8"))
    settings["print_settings_id"] = LOCKED_PROCESS_ID
    payloads["Metadata/project_settings.config"] = (
        json.dumps(settings, indent=4, ensure_ascii=False).encode("utf-8"))
    with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.writestr(member.filename, payloads[member.filename])
    with zipfile.ZipFile(destination) as archive:
        if archive.read("Metadata/plate_1.gcode") != source_gcode:
            raise ShelfError(
                f"{name}: delivered G-code diverged from the audited source")
    destination.chmod(
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return destination


def _slice_plain_entry(
    *,
    entry: dict,
    bundle: dict,
    bambu: Path,
) -> tuple[Path, dict]:
    """Slice one magnet-free entry under the pinned 0.6 bundle."""
    name = entry["name"]
    source = PROJECT_ROOT / "build" / entry["source_stl"]
    if not source.is_file():
        raise ShelfError(f"{name}: missing source STL {source}")
    workspace = LANE_WORKSPACE / "non_magnet" / name
    workspace.mkdir(parents=True, exist_ok=True)
    project = workspace / f"{name}.gcode.3mf"
    gcode = workspace / "plate_1.gcode"
    result = workspace / "result.json"
    fingerprint_path = workspace / "slice_fingerprint.json"
    command = canonical.captive._bambu_command(
        bambu, source, workspace, bundle, project_filename=project.name)
    fingerprint = canonical._non_magnet_fingerprint(
        source=source, sidecar=source.with_suffix(".print.json"),
        command=command, profile_bundle=bundle)
    reused = canonical._non_magnet_cache_matches(
        fingerprint_path, fingerprint, gcode, result, project)
    if not reused:
        for stale in (gcode, result, project, fingerprint_path):
            stale.unlink(missing_ok=True)
        run = subprocess.run(
            command, cwd=workspace, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(bundle["config"]["slicing"]["timeout_seconds"]),
            check=False, env={**os.environ, "LC_ALL": "C"})
        (workspace / "bambu_studio.log").write_text(
            run.stdout, encoding="utf-8", errors="replace")
        if run.returncode != 0:
            raise ShelfError(
                f"{name}: Bambu Studio exited {run.returncode}; see "
                f"{workspace / 'bambu_studio.log'}")
        if not all(path.is_file() for path in (gcode, result, project)):
            raise ShelfError(
                f"{name}: Bambu did not create plate_1.gcode, result.json, "
                "and the project")
        canonical.captive._inject_ready_project_object_support(
            project, enabled=False)
        canonical._write_json(fingerprint_path, {"fingerprint": fingerprint})
    validation = canonical._validate_result(
        label=name, stl=source, project=project, result_path=result,
        profile_bundle=bundle, artifact=None)
    return project, {
        "reused": bool(reused),
        "rz_degrees": validation["rz_degrees"],
        "triangle_count": validation["triangle_count"],
        "bed_clearances_mm": validation["bed_clearances_mm"],
        "source_workspace": str(
            project.relative_to(PROJECT_ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish the complete 0.6-mm lane onto the shelf")
    parser.add_argument("--shelf", type=Path, default=DEFAULT_SHELF)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args(argv)

    shelf = args.shelf.resolve()
    audit_root = args.audit_output.resolve()
    catalog = json.loads(
        (shelf / "catalog.json").read_text(encoding="utf-8"))

    pause_manifest = audit_root / "captive_magnet_pause_manifest.json"
    if not pause_manifest.is_file():
        raise SystemExit(
            f"missing authoritative 0.6-lane pause manifest: {pause_manifest}\n"
            "run 'make bambu_slice_release_06hf' first")
    provenance = json.loads(
        (audit_root / "profiles" / "profile_provenance.json")
        .read_text(encoding="utf-8"))

    audited = []
    plain = []
    skipped: dict[str, str] = {}
    for entry in catalog["entries"]:
        if entry.get("lane") == "06hf":
            skipped[entry["name"]] = (
                "PETG-GF structural delivery; the canonical shelf builder "
                "publishes it into 3mf_06hf from the PETG-GF 0.6 outputs")
        elif "auxiliary_delivery" in entry:
            skipped[entry["name"]] = (
                "auxiliary candidate delivery; 0.4-mm lane only for now")
        elif "composite_plate" in entry:
            skipped[entry["name"]] = (
                "PLA wing combo plate; 0.4-mm lane delivery")
        elif "catalog_artifact_id" in entry:
            audited.append(entry)
        else:
            plain.append(entry)

    failures: list[str] = []
    audited_records = []
    for entry in audited:
        artifact_id = entry["catalog_artifact_id"]
        slug = artifact_id.replace(":", "_")
        slice_dir = audit_root / "slices" / slug
        ready = slice_dir / "ready" / "ready_to_print.gcode.3mf"
        audit_json = slice_dir / "captive_magnet_slice_audit.json"
        if not ready.is_file() or not audit_json.is_file():
            failures.append(f"{artifact_id}: missing ready project or audit")
            continue
        audit = json.loads(audit_json.read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            failures.append(
                f"{artifact_id}: audit status {audit.get('status')!r}")
            continue
        audited_records.append((entry, ready, audit_json))

    if failures:
        print("0.6-lane shelf publication refused:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    bambu = canonical.captive._find_bambu_binary(None)
    bundle = canonical._profile_bundle(
        workspace=LANE_WORKSPACE / "profile",
        profile_path=LANE_PROFILE,
        bambu=bambu,
        system_root=None,
    )

    manifest_entries = []
    for entry, ready, audit_json in audited_records:
        destination = _publish(
            ready, shelf / entry["family"] / "3mf_06hf", entry["name"])
        manifest_entries.append({
            "name": entry["name"],
            "family": entry["family"],
            "state": entry["state"],
            "kind": "audited_captive_magnet_project",
            "catalog_artifact_id": entry["catalog_artifact_id"],
            "project": str(destination.relative_to(shelf)),
            "project_sha256": _sha256(destination),
            "source_ready_project": str(ready.relative_to(PROJECT_ROOT)),
            "source_audit": str(audit_json.relative_to(PROJECT_ROOT)),
        })
    for entry in plain:
        project, detail = _slice_plain_entry(
            entry=entry, bundle=bundle, bambu=bambu)
        destination = _publish(
            project, shelf / entry["family"] / "3mf_06hf", entry["name"])
        manifest_entries.append({
            "name": entry["name"],
            "family": entry["family"],
            "state": entry["state"],
            "kind": "plain_slice_no_magnets",
            "project": str(destination.relative_to(shelf)),
            "project_sha256": _sha256(destination),
            **detail,
        })

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lane": LANE,
        "printer": "Bambu Lab P2S 0.6 nozzle (high flow)",
        "profile_config_sha256": provenance.get("config_sha256"),
        "profile_config_path": provenance.get("config_path"),
        "plain_slice_profile_sha256": bundle["identity"]["config_sha256"],
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
        "excluded_entries": {
            name: reason for name, reason in sorted(skipped.items())
        },
    }
    manifest_path = shelf / "catalog_06hf.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=False) + "\n",
        encoding="utf-8")
    audited_count = sum(
        1 for record in manifest_entries
        if record["kind"] == "audited_captive_magnet_project")
    plain_count = len(manifest_entries) - audited_count
    print(f"published {len(manifest_entries)} 0.6-lane projects "
          f"({audited_count} pause-bearing, {plain_count} plain); "
          f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ShelfError as exc:
        print(f"0.6-lane shelf failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
