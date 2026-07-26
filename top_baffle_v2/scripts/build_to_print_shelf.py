#!/usr/bin/env python3
"""Build the small, printer-facing LX521 P2S delivery shelf.

``to_print/`` is deliberately a view over the canonical CAD release rather
than a second source of geometry.  It gives the files a human-useful order,
hard-links the small printable source STLs and ready Bambu projects, and
records a hash-backed local manifest.  It never contacts a printer.

Magnet-bearing files reuse the already audited pause-bearing projects from
``scripts/slice_captive_magnets.py``.  The nine non-magnet pieces are sliced locally
with precisely the same pinned P2S profile and receive an equivalent 3MF/mesh
/G-code audit, except that they must have no pause marker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET
import zipfile

from bambu_3mf_audit import (
    Bambu3MFAuditError,
    audit_bambu_3mf,
    validate_bed_fit as validate_bambu_bed_fit,
    validate_result_bbox as validate_bambu_result_bbox,
)
import build_obiwan_combo_plate as combo
from lx521_baffle.print_contract import FrontDownContractError, validate_print_sidecar
from lx521_baffle.io import pretty_json_bytes, sha256_bytes, sha256_file
import slice_captive_magnets as captive


ROOT = PROJECT_ROOT
DEFAULT_SHELF = ROOT / "to_print"
DEFAULT_CATALOG = DEFAULT_SHELF / "catalog.json"
DEFAULT_RELEASE_CATALOG = ROOT / "review" / "captive_magnet_release_catalog.json"
DEFAULT_RELEASE_AUDIT = ROOT / "review" / "captive_magnet_slice_audit"
DEFAULT_PROFILE = ROOT / "captive_magnet_slicing_profile.json"
LEGACY_SHELF_SOURCE_ROOTS = {
    "floor_stand": Path("build/floor_stand"),
    "no_floor_stand": Path("build/no_floor_stand"),
    "wings": Path("build/wings"),
}

EXPECTED_FAMILY_COUNTS = {"stock": 11, "slim": 11, "obiwan": 26}
EXPECTED_ENTRY_COUNT = sum(EXPECTED_FAMILY_COUNTS.values())
EXPECTED_MAGNET_PROJECT_COUNT = 39
EXPECTED_NON_MAGNET_PROJECT_COUNT = 9
NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
UNPRINTABLE_OR_LEGACY_TOKENS = (
    "core_1of2_lm_carrier", "c7", "v0", "coupon", "grommet",
    "lx521_top_v1_4of4_vase",
)


class ShelfError(RuntimeError):
    """A printer-shelf mapping, source, or slice contract failed."""


def _is_magnet_entry(entry: Mapping[str, Any]) -> bool:
    return bool(
        entry.get("catalog_artifact_id")
        or entry.get("composite_plate"))


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _canonical_json(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return sha256_bytes(data)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pretty_json_bytes(data, allow_nan=False))
    temporary.replace(path)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShelfError(f"cannot read {label} {path}: {exc}") from exc


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_shelf_source_relative(value: Any, label: str) -> Path:
    """Map one frozen pre-Stage-3 shelf source to its canonical read path."""
    if not isinstance(value, str) or not value:
        raise ShelfError(f"{label}.source_stl must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ShelfError(f"{label}.source_stl must stay inside this repository")
    # ``to_print/catalog.json`` is a frozen historical delivery.  Resolve its
    # pre-Stage-3 source roots read-only at the shelf-consumption boundary;
    # never rewrite the protected catalog or teach current producers two
    # authorities for the same generated tree.
    if relative.parts and relative.parts[0] in LEGACY_SHELF_SOURCE_ROOTS:
        relative = (
            LEGACY_SHELF_SOURCE_ROOTS[relative.parts[0]]
            / Path(*relative.parts[1:]))
    return relative


def _require_relative_stl(value: Any, label: str) -> Path:
    relative = _resolve_shelf_source_relative(value, label)
    source = (ROOT / relative).resolve()
    if source.suffix.lower() != ".stl" or not source.is_file():
        raise ShelfError(f"{label}.source_stl is missing or is not an STL: {source}")
    return source


def _catalog_entries(catalog_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _read_json(catalog_path, "shelf catalog")
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ShelfError("shelf catalog must be schema_version 1")
    if raw.get("catalog_kind") != "lx521_p2s_print_shelf":
        raise ShelfError("shelf catalog_kind is not lx521_p2s_print_shelf")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise ShelfError("shelf catalog entries must be an array")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    family_counts: dict[str, int] = {}
    core_slots: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ShelfError(f"catalog entry {index} must be an object")
        entry = dict(raw_entry)
        label = f"catalog entry {index}"
        name = entry.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ShelfError(f"{label}.name must contain only letters, digits, and _")
        if name in names:
            raise ShelfError(f"duplicate shelf file name {name!r}")
        names.add(name)
        family = entry.get("family")
        if family not in EXPECTED_FAMILY_COUNTS:
            raise ShelfError(f"{label}.family must be one of {sorted(EXPECTED_FAMILY_COUNTS)}")
        family_counts[family] = family_counts.get(family, 0) + 1
        for key in ("logical_slot", "state", "selection", "description"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ShelfError(f"{label}.{key} must be a non-empty string")
        lowered_source = str(entry.get("source_stl", "")).lower()
        if any(token in lowered_source for token in UNPRINTABLE_OR_LEGACY_TOKENS):
            raise ShelfError(f"{label} selects excluded legacy/oversized geometry")
        entry["source_path"] = _require_relative_stl(entry.get("source_stl"), label)
        composite_plate = entry.get("composite_plate")
        if composite_plate is None:
            sidecar = entry["source_path"].with_suffix(".print.json")
            try:
                entry["print_sidecar"] = validate_print_sidecar(
                    entry["source_path"], sidecar)
            except FrontDownContractError as exc:
                raise ShelfError(
                    f"{name}: invalid front-down print sidecar: {exc}") from exc
            entry["source_contract_path"] = sidecar
        else:
            if (name != combo.PLATE_NAME
                    or family != "obiwan"
                    or entry["state"] != "no_floor_stand"
                    or entry["selection"] != "core_plate_alternative"
                    or not isinstance(composite_plate, Mapping)):
                raise ShelfError(
                    f"{name}: unsupported composite-plate catalog entry")
            manifest_value = composite_plate.get("manifest")
            if not isinstance(manifest_value, str) or not manifest_value:
                raise ShelfError(
                    f"{name}.composite_plate.manifest must be a path")
            manifest_relative = Path(manifest_value)
            if (manifest_relative.is_absolute()
                    or ".." in manifest_relative.parts):
                raise ShelfError(
                    f"{name}: composite manifest must stay in the repository")
            manifest_path = (ROOT / manifest_relative).resolve()
            if (composite_plate.get("builder")
                    != "scripts/build_obiwan_combo_plate.py"
                    or composite_plate.get("magnet_insertions") != 6):
                raise ShelfError(
                    f"{name}: composite builder/pause contract drifted")
            expected_replacements = [
                part.friendly_name for part in combo.PARTS
            ]
            if composite_plate.get("replaces") != expected_replacements:
                raise ShelfError(
                    f"{name}: composite replacement inventory drifted")
            try:
                entry["print_sidecar"] = combo.validate_source_bundle(
                    entry["source_path"], manifest_path)
            except combo.ComboPlateError as exc:
                raise ShelfError(
                    f"{name}: invalid composite source contract: {exc}") from exc
            entry["source_contract_path"] = manifest_path
        artifact_id = entry.get("catalog_artifact_id")
        if artifact_id is not None and (not isinstance(artifact_id, str)
                                        or not artifact_id):
            raise ShelfError(f"{name}.catalog_artifact_id must be a string when set")
        normalized.append(entry)
        if entry["selection"] == "core":
            core_slots.setdefault((family, entry["logical_slot"]), []).append(entry)
    # Exactly one state-dependent lower part remains in each family.  Every
    # later core slot is canonicalized to one shared STL/project, so a user
    # can never select a floor/no-floor duplicate for a part that mates to
    # both lower states.
    for (family, slot), slot_entries in core_slots.items():
        if slot == "01":
            if (len(slot_entries) != 2
                    or {entry["state"] for entry in slot_entries}
                    != {"no_floor_stand", "floor_stand"}):
                raise ShelfError(
                    f"{family} core slot 01 must contain exactly the two "
                    "state-specific lower parts")
        elif len(slot_entries) != 1 or slot_entries[0]["state"] != "shared":
            raise ShelfError(
                f"{family} core slot {slot} must be one canonical shared part")
    if len(normalized) != EXPECTED_ENTRY_COUNT:
        raise ShelfError(
            f"shelf has {len(normalized)} entries, expected {EXPECTED_ENTRY_COUNT}")
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise ShelfError(
            f"shelf family counts {family_counts} != {EXPECTED_FAMILY_COUNTS}")
    magnetic = [entry for entry in normalized if _is_magnet_entry(entry)]
    if len(magnetic) != EXPECTED_MAGNET_PROJECT_COUNT:
        raise ShelfError(
            f"shelf has {len(magnetic)} magnet projects, expected "
            f"{EXPECTED_MAGNET_PROJECT_COUNT}")
    if len(normalized) - len(magnetic) != EXPECTED_NON_MAGNET_PROJECT_COUNT:
        raise ShelfError("shelf non-magnet project count is wrong")
    return raw, normalized


def _release_artifacts(release_catalog: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        catalog = captive.normalize_catalog(release_catalog)
    except captive.AuditError as exc:
        raise ShelfError(f"released captive-magnet catalog is invalid: {exc}") from exc
    by_id = {artifact["id"]: artifact for artifact in catalog["artifacts"]}
    return catalog, by_id


def _bind_entries_to_release(
    entries: Sequence[dict[str, Any]],
    artifact_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    for entry in entries:
        if entry.get("composite_plate") is not None:
            bound = []
            manifest_parts = entry["print_sidecar"].get("parts")
            if not isinstance(manifest_parts, list):
                raise ShelfError(
                    f"{entry['name']}: composite manifest has no parts")
            for part, record in zip(combo.PARTS, manifest_parts, strict=True):
                if part.artifact_id is None:
                    if record.get("catalog_artifact_id") is not None:
                        raise ShelfError(
                            f"{part.friendly_name}: unexpected release binding")
                    continue
                artifact = artifact_by_id.get(part.artifact_id)
                if artifact is None:
                    raise ShelfError(
                        f"{entry['name']}: unknown release artifact "
                        f"{part.artifact_id}")
                if (Path(artifact["stl"]).resolve()
                        != part.source_stl.resolve()
                        or artifact["stl_catalog_sha256"]
                        != sha256_file(part.source_stl)
                        or Path(artifact.get(
                            "support_blocker", "")).resolve()
                        != part.support_blocker.resolve()):
                    raise ShelfError(
                        f"{part.friendly_name}: composite release binding "
                        "differs from the canonical artifact")
                bound.append(dict(artifact))
            if len(bound) != 3:
                raise ShelfError(
                    f"{entry['name']}: expected three captive release bindings")
            entry["composite_artifacts"] = bound
            continue
        artifact_id = entry.get("catalog_artifact_id")
        if artifact_id is None:
            continue
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            raise ShelfError(f"{entry['name']}: unknown release artifact {artifact_id}")
        if entry["source_path"].resolve() != Path(artifact["stl"]).resolve():
            raise ShelfError(
                f"{entry['name']}: source STL does not match {artifact_id}")
        source_sha = _sha256(entry["source_path"])
        if source_sha != artifact["stl_catalog_sha256"]:
            raise ShelfError(
                f"{entry['name']}: source STL hash differs from release catalog")
        if _sha256(entry["source_path"].with_suffix(".print.json")) != artifact["print_sidecar_sha256"]:
            raise ShelfError(
                f"{entry['name']}: source print sidecar hash differs from release catalog")
        entry["artifact"] = dict(artifact)


def _delivery_paths(shelf: Path, entry: Mapping[str, Any]) -> tuple[Path, Path]:
    family_root = shelf / str(entry["family"])
    return (
        family_root / "stl" / f"{entry['name']}.stl",
        family_root / "3mf" / f"{entry['name']}.gcode.3mf",
    )


def _workspace_root(shelf: Path) -> Path:
    """Keep slicer cache out of the user-facing delivery tree.

    ``to_print`` is intended to be browseable directly in Finder or Bambu
    Studio.  A hidden workspace there still makes stale projects discoverable
    and previously let retired 02a/02b and 03a/03b variants linger beside the
    canonical release.  Review/cache material belongs under ``review/``.
    """
    return (shelf.parent / "review" / f"{shelf.name}_slice_workspace").resolve()


def _migrate_legacy_workspace(shelf: Path, workspace: Path) -> None:
    """Move the former in-shelf cache once, preserving valid local slices."""
    legacy = shelf / ".slice_workspace"
    if not legacy.exists() and not legacy.is_symlink():
        return
    if not legacy.is_dir() or legacy.is_symlink():
        raise ShelfError(f"legacy slice workspace is not a real directory: {legacy}")
    if workspace.exists() or workspace.is_symlink():
        return
    workspace.parent.mkdir(parents=True, exist_ok=True)
    legacy.replace(workspace)


def _remove_known_directory(path: Path) -> None:
    """Remove one explicitly-owned internal directory, never a broad path."""
    if path.is_symlink() or path.is_file():
        raise ShelfError(f"expected managed directory, found file/symlink: {path}")
    if path.is_dir():
        shutil.rmtree(path)


def _prune_delivery_view(shelf: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    """Make the managed delivery folders an exact view of the shelf catalog.

    This runs only after every desired file has been materialized and audited,
    so a failed build never deletes the previous usable shelf.  ``stl`` and
    ``3mf`` are project-owned artifact-only directories; stale files there are
    retired canonical revisions, not user documents.
    """
    expected_by_root: dict[Path, set[Path]] = {}
    for entry in entries:
        stl, project = _delivery_paths(shelf, entry)
        expected_by_root.setdefault(stl.parent.absolute(), set()).add(stl.absolute())
        expected_by_root.setdefault(project.parent.absolute(), set()).add(
            project.absolute())
    for root, expected in expected_by_root.items():
        if not root.exists():
            raise ShelfError(f"managed delivery directory disappeared: {root}")
        if not root.is_dir() or root.is_symlink():
            raise ShelfError(f"managed delivery path is not a directory: {root}")
        for candidate in root.iterdir():
            if candidate.is_dir() and not candidate.is_symlink():
                raise ShelfError(
                    f"unexpected nested directory in managed delivery view: {candidate}")
            if candidate.absolute() not in expected:
                candidate.unlink()
    # Finder metadata is neither a source artifact nor a printable file.
    for metadata in shelf.rglob(".DS_Store"):
        metadata.unlink(missing_ok=True)


def _prune_workspace(workspace: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    """Retire obsolete non-magnet caches after a successful shelf rebuild."""
    non_magnet_root = workspace / "non_magnet"
    if not non_magnet_root.is_dir() or non_magnet_root.is_symlink():
        return
    expected = {
        str(entry["name"])
        for entry in entries
        if not _is_magnet_entry(entry)
    }
    for candidate in non_magnet_root.iterdir():
        if candidate.name in expected:
            continue
        _remove_known_directory(candidate)


def _retire_legacy_workspace(shelf: Path) -> None:
    """Remove the obsolete cache location after the new cache is proven valid."""
    legacy = shelf / ".slice_workspace"
    if legacy.exists() or legacy.is_symlink():
        _remove_known_directory(legacy)


def _link_or_copy(source: Path, destination: Path) -> str:
    """Materialize ordinary local files while avoiding a duplicate payload."""
    source = source.resolve()
    if not source.is_file():
        raise ShelfError(f"cannot materialize missing source {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            same = destination.samefile(source)
        except OSError:
            same = False
        if same:
            return "hardlink"
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _ready_source(
    release_audit: Path,
    artifact_id: str,
) -> tuple[Path, Path, Path, Path]:
    slug = captive._slug(artifact_id)
    ready_dir = release_audit / "slices" / slug / "ready"
    return (
        ready_dir / captive.READY_3MF_FILENAME,
        ready_dir / "plate_1.gcode",
        ready_dir / "result.json",
        release_audit / "slices" / slug / "captive_magnet_slice_audit.json",
    )


def _expected_pause_z(
    artifact: Mapping[str, Any],
    audit_path: Path,
) -> list[float]:
    """Read actual G-code pause planes, never a CAD-only estimate."""
    audit = _read_json(audit_path, f"{artifact['id']} slice audit")
    if not isinstance(audit, Mapping) or audit.get("id") != artifact["id"]:
        raise ShelfError(f"{artifact['id']}: slice audit identity mismatch")
    if audit.get("status") != "pass":
        raise ShelfError(f"{artifact['id']}: slice audit is not passing")
    sites = audit.get("sites")
    if not isinstance(sites, list):
        raise ShelfError(f"{artifact['id']}: slice audit lacks site records")
    expected_names = {site["name"] for site in artifact["sites"]}
    actual_names: set[str] = set()
    pauses: set[float] = set()
    for record in sites:
        if not isinstance(record, Mapping):
            raise ShelfError(f"{artifact['id']}: malformed slice-audit site")
        site = record.get("site")
        actual = record.get("actual")
        if not isinstance(site, Mapping) or not isinstance(actual, Mapping):
            raise ShelfError(f"{artifact['id']}: slice-audit site lacks actual pause")
        name = site.get("name")
        value = actual.get("bambu_studio_pause_marker_z_mm")
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            raise ShelfError(f"{artifact['id']}: invalid G-code pause record")
        actual_names.add(name)
        pauses.add(float(value))
    if actual_names != expected_names:
        raise ShelfError(f"{artifact['id']}: slice-audit site set differs from release")
    if not pauses:
        raise ShelfError(f"{artifact['id']}: release artifact has no pause planes")
    return sorted(pauses)


def _validate_result(
    *,
    label: str,
    stl: Path,
    project: Path,
    result_path: Path,
    profile_bundle: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    mesh = captive.inspect_stl(stl)
    if abs(mesh.bounds_min[2]) > 0.02:
        raise ShelfError(f"{label}: front-down STL does not sit on Z=0")
    result = _read_json(result_path, f"{label} Bambu result")
    if not isinstance(result, Mapping) or result.get("return_code") != 0:
        raise ShelfError(f"{label}: Bambu result is not Success")
    plates = result.get("sliced_plates")
    if not isinstance(plates, list) or len(plates) != 1:
        raise ShelfError(f"{label}: expected exactly one sliced plate")
    objects = plates[0].get("objects") if isinstance(plates[0], Mapping) else None
    if not isinstance(objects, list) or len(objects) != 1:
        raise ShelfError(f"{label}: expected exactly one sliced object")
    if (int(plates[0].get("triangle_count", -1)) != mesh.triangle_count
            or int(objects[0].get("triangle_count", -1)) != mesh.triangle_count):
        raise ShelfError(f"{label}: Bambu triangle count differs from STL")
    bbox = objects[0].get("bbox")
    if not isinstance(bbox, Mapping):
        raise ShelfError(f"{label}: Bambu result does not contain an object bbox")
    support_blockers = (
        (Path(artifact["support_blocker"]),)
        if artifact is not None and "support_blocker" in artifact else ()
    )
    try:
        audit = audit_bambu_3mf(
            project, stl, support_blocker_stls=support_blockers)
        validate_bambu_result_bbox(bbox, audit.source_bounds, audit.stl_to_bed_matrix)
        clearances = validate_bambu_bed_fit(
            audit.transformed_actual_mesh_bounds,
            profile_bundle["identity"]["machine_bounds_mm"])
    except Bambu3MFAuditError as exc:
        raise ShelfError(f"{label}: 3MF mesh/placement audit failed: {exc}") from exc
    return {
        "triangle_count": mesh.triangle_count,
        "mesh_max_abs_error_mm": audit.mesh_max_abs_error_mm,
        "rz_degrees": audit.rigid_rz.rz_degrees,
        "bed_clearances_mm": clearances,
        "stl_to_bed_matrix": [
            list(row) for row in audit.stl_to_bed_matrix
        ],
        "support_blocker_count": audit.support_blocker_count,
    }


def _parse_project_archive(
    *,
    label: str,
    project: Path,
    plain_gcode: Path,
    profile_bundle: Mapping[str, Any],
    expected_pause_z: Sequence[float] | None,
) -> dict[str, Any]:
    if expected_pause_z is not None:
        try:
            return captive._validate_ready_project_archive(
                project, plain_gcode, expected_pause_z=expected_pause_z,
                profile_bundle=profile_bundle)
        except captive.AuditError as exc:
            raise ShelfError(f"{label}: ready-project archive audit failed: {exc}") from exc

    required = (
        "Metadata/project_settings.config",
        "Metadata/model_settings.config",
        "Metadata/plate_1.gcode",
    )
    try:
        with zipfile.ZipFile(project) as archive:
            names = archive.namelist()
            if archive.testzip() is not None:
                raise ShelfError(f"{label}: 3MF ZIP is corrupt")
            for member in required:
                if names.count(member) != 1:
                    raise ShelfError(f"{label}: expected exactly one {member}")
            settings_bytes = archive.read(required[0])
            model_settings = archive.read(required[1])
            embedded_gcode = archive.read(required[2])
            custom_xml = (archive.read("Metadata/custom_gcode_per_layer.xml")
                          if "Metadata/custom_gcode_per_layer.xml" in names
                          else b"")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ShelfError(f"{label}: cannot inspect 3MF archive: {exc}") from exc
    if embedded_gcode != plain_gcode.read_bytes():
        raise ShelfError(f"{label}: embedded G-code differs from plate_1.gcode")
    try:
        settings = json.loads(settings_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShelfError(f"{label}: project settings are not JSON") from exc
    if not isinstance(settings, Mapping):
        raise ShelfError(f"{label}: project settings are not an object")
    enforced: dict[str, Any] = {}
    for section, values in profile_bundle["enforced_overrides"].items():
        for key, expected in values.items():
            actual = settings.get(key)
            if not captive._profile_value_equal(actual, expected):
                raise ShelfError(
                    f"{label}: embedded setting {key}={actual!r} differs from "
                    f"the resolved {section} profile")
            enforced[key] = actual
    try:
        object_support_overrides = (
            captive._validate_ready_project_object_support(
                model_settings, project_3mf=project, enabled=False))
    except captive.AuditError as exc:
        raise ShelfError(
            f"{label}: object support policy is not pinned off: {exc}") from exc
    if custom_xml:
        try:
            root = ET.fromstring(custom_xml)
        except ET.ParseError as exc:
            raise ShelfError(f"{label}: custom G-code XML is invalid") from exc
        layers = [element for element in root.iter()
                  if element.tag.rsplit("}", 1)[-1] == "layer"]
        if layers:
            raise ShelfError(f"{label}: non-magnet project unexpectedly embeds a pause")
    parsed = captive.parse_gcode(plain_gcode, retain_regions=())
    errors = captive._validate_actual_gcode_profile(parsed, profile_bundle)
    if errors:
        raise ShelfError(f"{label}: G-code profile mismatch: {'; '.join(errors)}")
    pause_policy = profile_bundle["identity"]["effective"].get(
        "magnet_insertion_pause")
    if not isinstance(pause_policy, Mapping):
        raise ShelfError(f"{label}: profile has no magnet insertion pause policy")
    pauses = captive._gcode_pause_events(plain_gcode, pause_policy)
    if pauses:
        raise ShelfError(f"{label}: non-magnet project unexpectedly contains pauses")
    return {
        "project_settings_sha256": _sha256_bytes(settings_bytes),
        "model_settings_sha256": _sha256_bytes(model_settings),
        "object_support_overrides": object_support_overrides,
        "embedded_gcode_sha256": _sha256_bytes(embedded_gcode),
        "custom_gcode_xml_sha256": _sha256_bytes(custom_xml),
        "enforced_project_settings": enforced,
        "pause_z_mm": [],
        "gcode_pause_events": [],
    }


def _non_magnet_workspace(shelf: Path, entry: Mapping[str, Any]) -> Path:
    return _workspace_root(shelf) / "non_magnet" / str(entry["name"])


def _non_magnet_fingerprint(
    *,
    source: Path,
    sidecar: Path,
    command: Sequence[str],
    profile_bundle: Mapping[str, Any],
) -> str:
    return _sha256_bytes(_canonical_json({
        "source_stl_sha256": _sha256(source),
        "source_print_sidecar_sha256": _sha256(sidecar),
        "profile_set_sha256": profile_bundle["identity"]["profile_set_sha256"],
        "bambu_binary_sha256": profile_bundle["identity"]["binary_sha256"],
        "command": list(command),
        "object_support_policy": {
            key: "0" for key in captive.SUPPORT_PROCESS_KEYS
        },
    }))


def _non_magnet_cache_matches(
    fingerprint_path: Path,
    fingerprint: str,
    gcode: Path,
    result: Path,
    project: Path,
) -> bool:
    if not all(path.is_file() for path in (fingerprint_path, gcode, result, project)):
        return False
    prior = _read_json(fingerprint_path, "non-magnet slice fingerprint")
    return isinstance(prior, Mapping) and all(
        prior.get(key) == value for key, value in {
            "fingerprint": fingerprint,
            "gcode_sha256": _sha256(gcode),
            "result_sha256": _sha256(result),
            "project_3mf_sha256": _sha256(project),
        }.items())


def _relocated_non_magnet_cache_matches(
    *,
    fingerprint_path: Path,
    fingerprint: str,
    source: Path,
    sidecar: Path,
    command: Sequence[str],
    profile_bundle: Mapping[str, Any],
    workspace: Path,
    legacy_workspace: Path,
    gcode: Path,
    result: Path,
    project: Path,
) -> bool:
    """Accept one cache migration only when the command differs by its root.

    The old fingerprint deliberately included absolute Bambu paths.  Moving
    its cache out of the shelf must not silently accept different slicing
    inputs, but it also should not force nine needless re-slices when every
    input and every generated byte is unchanged.
    """
    if not all(path.is_file() for path in (fingerprint_path, gcode, result, project)):
        return False
    prior = _read_json(fingerprint_path, "legacy non-magnet slice fingerprint")
    prior_command = prior.get("command") if isinstance(prior, Mapping) else None
    if (not isinstance(prior_command, list)
            or any(not isinstance(value, str) for value in prior_command)):
        return False
    marker = "<slice-workspace>"
    normalize = lambda values, root: [
        value.replace(str(root.resolve()), marker) for value in values
    ]
    if normalize(prior_command, legacy_workspace) != normalize(command, workspace):
        return False
    legacy_fingerprint = _non_magnet_fingerprint(
        source=source, sidecar=sidecar, command=prior_command,
        profile_bundle=profile_bundle)
    expected_hashes = {
        "gcode_sha256": _sha256(gcode),
        "result_sha256": _sha256(result),
        "project_3mf_sha256": _sha256(project),
    }
    if (prior.get("fingerprint") != legacy_fingerprint
            or any(prior.get(key) != value
                   for key, value in expected_hashes.items())):
        return False
    _write_json(fingerprint_path, {
        "fingerprint": fingerprint,
        "command": list(command),
        **expected_hashes,
    })
    return True


def _slice_non_magnet(
    *,
    shelf: Path,
    entry: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
    bambu: Path,
    allow_slice: bool,
) -> tuple[Path, Path, Path, bool]:
    source = Path(entry["source_path"])
    workspace = _non_magnet_workspace(shelf, entry)
    workspace.mkdir(parents=True, exist_ok=True)
    project = workspace / f"{entry['name']}.gcode.3mf"
    gcode = workspace / "plate_1.gcode"
    result = workspace / "result.json"
    fingerprint_path = workspace / "slice_fingerprint.json"
    command = captive._bambu_command(
        bambu, source, workspace, profile_bundle,
        project_filename=project.name)
    fingerprint = _non_magnet_fingerprint(
        source=source, sidecar=source.with_suffix(".print.json"),
        command=command, profile_bundle=profile_bundle)
    reused = _non_magnet_cache_matches(
        fingerprint_path, fingerprint, gcode, result, project)
    if not reused:
        reused = _relocated_non_magnet_cache_matches(
            fingerprint_path=fingerprint_path, fingerprint=fingerprint,
            source=source, sidecar=source.with_suffix(".print.json"),
            command=command, profile_bundle=profile_bundle,
            workspace=_workspace_root(shelf),
            legacy_workspace=(shelf / ".slice_workspace").resolve(),
            gcode=gcode, result=result, project=project)
    if not reused:
        if not allow_slice:
            raise ShelfError(
                f"{entry['name']}: no current non-magnet P2S project; "
                "run make to_print")
        for stale in (gcode, result, project, fingerprint_path):
            stale.unlink(missing_ok=True)
        run = subprocess.run(
            command, cwd=workspace, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(profile_bundle["config"]["slicing"]["timeout_seconds"]),
            check=False, env={**os.environ, "LC_ALL": "C"})
        (workspace / "bambu_studio.log").write_text(
            run.stdout, encoding="utf-8", errors="replace")
        if run.returncode != 0:
            raise ShelfError(
                f"{entry['name']}: Bambu Studio exited {run.returncode}; "
                f"see {workspace / 'bambu_studio.log'}")
        if not all(path.is_file() for path in (gcode, result, project)):
            raise ShelfError(
                f"{entry['name']}: Bambu did not create plate_1.gcode, "
                "result.json, and .gcode.3mf")
        try:
            captive._inject_ready_project_object_support(
                project, enabled=False)
        except captive.AuditError as exc:
            raise ShelfError(
                f"{entry['name']}: could not pin object support off: "
                f"{exc}") from exc
        _write_json(fingerprint_path, {
            "fingerprint": fingerprint,
            "command": command,
            "gcode_sha256": _sha256(gcode),
            "result_sha256": _sha256(result),
            "project_3mf_sha256": _sha256(project),
        })
    return project, gcode, result, reused


def _profile_bundle(
    *,
    workspace: Path,
    profile_path: Path,
    bambu: Path,
    system_root: Path | None,
) -> dict[str, Any]:
    return captive.prepare_profiles(
        profile_path, workspace,
        system_root=system_root, bambu_binary=bambu)


def _validate_magnet_project(
    *,
    entry: Mapping[str, Any],
    release_audit: Path,
    base_profile: Mapping[str, Any],
    profile_workspace: Path,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact = entry["artifact"]
    ready_project, gcode, result, audit_path = _ready_source(
        release_audit, artifact["id"])
    if not all(path.is_file() for path in (ready_project, gcode, result, audit_path)):
        raise ShelfError(
            f"{entry['name']}: audited ready project is missing for {artifact['id']}; "
            "run make bambu_slice_release")
    project_profile = captive._artifact_profile_bundle(
        artifact, base_profile,
        profile_workspace / captive._slug(artifact["id"]))
    archive = _parse_project_archive(
        label=str(entry["name"]), project=ready_project, plain_gcode=gcode,
        profile_bundle=project_profile,
        expected_pause_z=_expected_pause_z(artifact, audit_path))
    placement = _validate_result(
        label=str(entry["name"]), stl=Path(entry["source_path"]),
        project=ready_project, result_path=result,
        profile_bundle=project_profile, artifact=artifact)
    support_enabled = bool(
        project_profile["identity"]["effective"]["support_enabled"])
    support_toolpaths = captive._support_toolpath_summary(gcode)
    if support_enabled:
        contract = artifact.get("duct_collision_contract")
        if not isinstance(contract, Mapping):
            raise ShelfError(
                f"{entry['name']}: support-enabled project lacks a "
                "hash-bound duct collision contract")
        try:
            duct_audit = captive.audit_support_toolpaths_vs_ducts(
                gcode=gcode,
                contract=contract,
                source_to_stl_matrix=artifact["source_to_stl_matrix"],
                stl_to_bed_matrix=tuple(
                    tuple(float(value) for value in row)
                    for row in placement["stl_to_bed_matrix"]),
            )
        except captive.AuditError as exc:
            raise ShelfError(
                f"{entry['name']}: support-vs-duct collision gate failed: "
                f"{exc}") from exc
    else:
        if any(support_toolpaths.values()):
            raise ShelfError(
                f"{entry['name']}: support-disabled project contains "
                "support feature blocks")
        duct_audit = {
            "status": "pass",
            "gate": "support_disabled_no_support_feature_blocks",
            "collision_count": 0,
        }
    archive["support_toolpaths"] = support_toolpaths
    archive["duct_support_toolpath_audit"] = duct_audit
    return (
        ready_project, gcode, result, archive, placement,
        dict(project_profile["identity"]["effective"]),
    )


def _validate_non_magnet_project(
    *,
    entry: Mapping[str, Any],
    project: Path,
    gcode: Path,
    result: Path,
    profile_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    archive = _parse_project_archive(
        label=str(entry["name"]), project=project, plain_gcode=gcode,
        profile_bundle=profile_bundle, expected_pause_z=None)
    placement = _validate_result(
        label=str(entry["name"]), stl=Path(entry["source_path"]),
        project=project, result_path=result, profile_bundle=profile_bundle,
        artifact=None)
    support_toolpaths = captive._support_toolpath_summary(gcode)
    if any(support_toolpaths.values()):
        raise ShelfError(
            f"{entry['name']}: non-magnet support-disabled project contains "
            "support feature blocks")
    archive["support_toolpaths"] = support_toolpaths
    archive["duct_support_toolpath_audit"] = {
        "status": "pass",
        "gate": "support_disabled_no_support_feature_blocks",
        "collision_count": 0,
    }
    return archive, placement


def _validate_composite_project(
    *,
    entry: Mapping[str, Any],
    shelf: Path,
    profile_path: Path,
    release_catalog: Path,
    release_audit: Path,
    system_root: Path | None,
    bambu: Path,
    allow_slice: bool,
) -> tuple[
    Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any], bool
]:
    if entry.get("name") != combo.PLATE_NAME:
        raise ShelfError("unknown composite shelf project")
    try:
        result = combo.build_or_validate_ready_plate(
            workspace=(
                _workspace_root(shelf) / "composite" / combo.PLATE_NAME),
            profile_path=profile_path,
            release_catalog=release_catalog,
            release_audit=release_audit,
            system_root=system_root,
            bambu_binary=str(bambu),
            allow_slice=allow_slice,
        )
    except combo.ComboPlateError as exc:
        raise ShelfError(
            f"{entry['name']}: composite project audit failed: {exc}") from exc
    audit = result["audit"]
    project_equivalence = audit["project_stl_equivalence"]
    archive = dict(audit["archive_audit"])
    archive["support_toolpaths"] = audit["support_toolpaths"]
    archive["duct_support_toolpath_audit"] = audit[
        "duct_support_toolpath_audit"]
    archive["captive_cavity_audit"] = audit["captive_cavity_audit"]
    archive["pause_before_first_layer_extrusion"] = audit[
        "pause_before_first_layer_extrusion"]
    archive["support_midpoints_inside_part_footprints"] = audit[
        "support_midpoints_inside_part_footprints"]
    placement = {
        "triangle_count": int(project_equivalence["triangle_count"]),
        "mesh_max_abs_error_mm": float(
            project_equivalence["mesh_max_abs_error_mm"]),
        "rz_degrees": float(project_equivalence["rigid_rz"]["rz_degrees"]),
        "bed_clearances_mm": audit["bed_clearances_mm"],
        "stl_to_bed_matrix": project_equivalence["stl_to_bed_matrix"],
        "support_blocker_count": int(
            project_equivalence["support_blocker_count"]),
        "normal_part_count": len(
            project_equivalence["normal_part_names"]),
    }
    return (
        Path(result["project"]),
        Path(result["gcode"]),
        Path(result["result"]),
        archive,
        placement,
        dict(result["profile_effective"]),
        bool(result["reused"]),
    )


def build_shelf(
    *,
    shelf: Path,
    catalog_path: Path,
    release_catalog: Path,
    release_audit: Path,
    profile_path: Path,
    allow_slice: bool,
    system_root: Path | None,
    bambu_binary: str | None,
    only_names: set[str] | None = None,
) -> dict[str, Any]:
    catalog_raw, entries = _catalog_entries(catalog_path)
    release, artifacts = _release_artifacts(release_catalog)
    _bind_entries_to_release(entries, artifacts)
    selected_entries = entries
    prior_manifest: dict[str, Any] | None = None
    if only_names:
        entries_by_name = {str(entry["name"]): entry for entry in entries}
        unknown = sorted(only_names - set(entries_by_name))
        if unknown:
            raise ShelfError(
                "--only names are not present in the shelf catalog: "
                + ", ".join(unknown))
        selected_entries = [
            entry for entry in entries if entry["name"] in only_names]
        manifest_path = shelf / "release_manifest.json"
        prior = _read_json(manifest_path, "existing shelf release manifest")
        if (not isinstance(prior, dict)
                or prior.get("schema_version") != 1
                or prior.get("manifest_kind") != "lx521_p2s_print_shelf"
                or not isinstance(prior.get("entries"), list)):
            raise ShelfError(
                "targeted shelf refresh requires one valid existing "
                "to_print/release_manifest.json")
        prior_names = {
            record.get("name") for record in prior["entries"]
            if isinstance(record, Mapping)
        }
        expected_names = {entry["name"] for entry in entries}
        if prior_names != expected_names:
            raise ShelfError(
                "targeted shelf refresh requires an existing complete "
                f"{EXPECTED_ENTRY_COUNT}-entry manifest")
        prior_manifest = dict(prior)
    workspace = _workspace_root(shelf)
    _migrate_legacy_workspace(shelf, workspace)
    try:
        bambu = captive._find_bambu_binary(bambu_binary)
    except captive.AuditError as exc:
        raise ShelfError(str(exc)) from exc
    base_profile = _profile_bundle(
        workspace=workspace, profile_path=profile_path, bambu=bambu,
        system_root=system_root)
    # Validate every source/project pair before touching any managed shelf
    # STL or 3MF.  Targeted refreshes still cross the complete equivalence barrier;
    # ``selected_entries`` controls only the later promotion step.
    validated: list[dict[str, Any]] = []
    for entry in entries:
        source = Path(entry["source_path"])
        source_sidecar = Path(entry["source_contract_path"])
        artifact = entry.get("artifact")
        composite_plate = entry.get("composite_plate")
        if composite_plate is not None:
            (project, gcode, result, archive, placement,
             profile_effective, reused) = _validate_composite_project(
                entry=entry,
                shelf=shelf,
                profile_path=profile_path,
                release_catalog=release_catalog,
                release_audit=release_audit,
                system_root=system_root,
                bambu=bambu,
                allow_slice=allow_slice,
            )
            project_kind = "local_composite_captive_magnet_slice"
        elif artifact is not None:
            (project, gcode, result, archive, placement,
             profile_effective) = _validate_magnet_project(
                entry=entry, release_audit=release_audit,
                base_profile=base_profile,
                profile_workspace=workspace / "magnet_profiles")
            project_kind = "audited_captive_magnet_reuse"
            reused = True
        else:
            project, gcode, result, reused = _slice_non_magnet(
                shelf=shelf, entry=entry, profile_bundle=base_profile,
                bambu=bambu, allow_slice=allow_slice)
            archive, placement = _validate_non_magnet_project(
                entry=entry, project=project, gcode=gcode, result=result,
                profile_bundle=base_profile)
            project_kind = "local_non_magnet_slice"
            profile_effective = base_profile["identity"]["effective"]
        validated.append({
            "entry": entry,
            "source": source,
            "source_sidecar": source_sidecar,
            "project": project,
            "gcode": gcode,
            "record": {
                "name": entry["name"],
                "family": entry["family"],
                "logical_slot": entry["logical_slot"],
                "state": entry["state"],
                "selection": entry["selection"],
                "description": entry["description"],
                "catalog_artifact_id": entry.get("catalog_artifact_id"),
                "magnet_insertions": (
                    int(composite_plate["magnet_insertions"])
                    if composite_plate is not None
                    else len(artifact["sites"]) if artifact else 0),
                "source_stl": _relative(source),
                "source_stl_sha256": _sha256(source),
                "source_print_sidecar": _relative(source_sidecar),
                "source_print_sidecar_sha256": _sha256(source_sidecar),
                "composite_plate": (
                    dict(composite_plate)
                    if composite_plate is not None else None),
                "project_source": _relative(project),
                "gcode_source": _relative(gcode),
                "gcode_sha256": _sha256(gcode),
                "project_kind": project_kind,
                "slice_reused": reused,
                "profile_effective": profile_effective,
                "archive_audit": archive,
                "placement_audit": placement,
            },
        })
    if len(validated) != EXPECTED_ENTRY_COUNT:
        raise ShelfError(
            "project/STL equivalence gate did not inspect every shelf entry")
    equivalence_entries = []
    for item in validated:
        record = item["record"]
        placement = record["placement_audit"]
        if (not isinstance(placement.get("triangle_count"), int)
                or placement["triangle_count"] <= 0
                or float(placement.get(
                    "mesh_max_abs_error_mm", float("inf"))) > 0.02):
            raise ShelfError(
                f"{record['name']}: project/STL equivalence evidence is "
                "missing or out of tolerance")
        duct_audit = record["archive_audit"].get(
            "duct_support_toolpath_audit")
        if (not isinstance(duct_audit, Mapping)
                or duct_audit.get("status") != "pass"
                or duct_audit.get("collision_count") != 0):
            raise ShelfError(
                f"{record['name']}: support/duct safety gate is not passing")
        equivalence_entries.append({
            "name": record["name"],
            "source_stl_sha256": record["source_stl_sha256"],
            "project_sha256": _sha256(item["project"]),
            "triangle_count": placement["triangle_count"],
            "mesh_max_abs_error_mm": placement[
                "mesh_max_abs_error_mm"],
            "support_blocker_count": placement[
                "support_blocker_count"],
        })
    equivalence_gate = {
        "status": "pass",
        "required_pair_count": EXPECTED_ENTRY_COUNT,
        "passing_pair_count": len(equivalence_entries),
        "mesh_tolerance_mm": 0.02,
        "promotion_started_after_complete_gate": True,
        "entries": equivalence_entries,
    }

    records: list[dict[str, Any]] = []
    selected_names = {entry["name"] for entry in selected_entries}
    for item in validated:
        record = item["record"]
        if record["name"] not in selected_names:
            continue
        entry = item["entry"]
        source = item["source"]
        project = item["project"]
        stl_destination, project_destination = _delivery_paths(shelf, entry)
        stl_delivery = _link_or_copy(source, stl_destination)
        project_delivery = _link_or_copy(project, project_destination)
        if _sha256(project_destination) != _sha256(project):
            raise ShelfError(f"{entry['name']}: delivered 3MF hash mismatch")
        if _sha256(stl_destination) != _sha256(source):
            raise ShelfError(f"{entry['name']}: delivered STL hash mismatch")
        record.update({
            "delivered_stl": _relative(stl_destination),
            "delivered_stl_sha256": _sha256(stl_destination),
            "stl_delivery": stl_delivery,
            "p2s_project": _relative(project_destination),
            "p2s_project_sha256": _sha256(project_destination),
            "project_delivery": project_delivery,
        })
        records.append(record)
    if prior_manifest is not None:
        refreshed = {record["name"]: record for record in records}
        records = [
            refreshed.get(record["name"], dict(record))
            for record in prior_manifest["entries"]
        ]

    manifest = {
        "schema_version": 1,
        "manifest_kind": "lx521_p2s_print_shelf",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(
            microsecond=0).isoformat().replace("+00:00", "Z"),
        "shelf_catalog": {
            "path": _relative(catalog_path),
            "sha256": _sha256(catalog_path),
            "selection_rules": catalog_raw["selection_rules"],
        },
        "release_catalog": {
            "path": _relative(release_catalog),
            "sha256": _sha256(release_catalog),
            "source_revision": release["source_revision"],
        },
        "slicer": base_profile["identity"],
        "project_stl_equivalence_gate": equivalence_gate,
        "inventory": {
            "entry_count": len(records),
            "family_counts": EXPECTED_FAMILY_COUNTS,
            "magnet_project_count": sum(
                int(record["magnet_insertions"]) > 0 for record in records),
            "non_magnet_project_count": sum(
                int(record["magnet_insertions"]) == 0 for record in records),
            "magnet_insertions": sum(record["magnet_insertions"] for record in records),
        },
        "entries": records,
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json({
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }))
    # All new files have been materialized and validated at this point.  Make
    # the delivery tree exact only now, then retire cache entries that belong
    # to superseded friendly names.
    if prior_manifest is None:
        _prune_delivery_view(shelf, entries)
        _prune_workspace(workspace, entries)
        _retire_legacy_workspace(shelf)
    _write_json(shelf / "release_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shelf", type=Path, default=DEFAULT_SHELF)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--release-catalog", type=Path,
                        default=DEFAULT_RELEASE_CATALOG)
    parser.add_argument("--release-audit", type=Path,
                        default=DEFAULT_RELEASE_AUDIT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--bambu-studio")
    parser.add_argument("--bambu-system-root", type=Path)
    parser.add_argument(
        "--slice-missing", action="store_true",
        help="slice missing or stale non-magnet P2S projects locally")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="require a complete current shelf without slicing anything")
    parser.add_argument(
        "--only", action="append", metavar="NAME",
        help="refresh and validate only this friendly shelf entry while "
             "preserving the other records; repeat for multiple entries")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.slice_missing and args.validate_only:
        raise ShelfError("--slice-missing and --validate-only are mutually exclusive")
    shelf = args.shelf.expanduser().resolve()
    shelf.mkdir(parents=True, exist_ok=True)
    manifest = build_shelf(
        shelf=shelf,
        catalog_path=args.catalog.expanduser().resolve(),
        release_catalog=args.release_catalog.expanduser().resolve(),
        release_audit=args.release_audit.expanduser().resolve(),
        profile_path=args.profile.expanduser().resolve(),
        allow_slice=bool(args.slice_missing),
        system_root=(args.bambu_system_root.expanduser().resolve()
                     if args.bambu_system_root else None),
        bambu_binary=args.bambu_studio,
        only_names=set(args.only or ()),
    )
    inventory = manifest["inventory"]
    print(
        f"P2S shelf ready: {inventory['entry_count']} files "
        f"({inventory['magnet_project_count']} pause-bearing, "
        f"{inventory['non_magnet_project_count']} without magnets)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ShelfError, captive.AuditError, OSError) as exc:
        print(f"to_print shelf failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
