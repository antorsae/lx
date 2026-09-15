#!/usr/bin/env python3
"""Slice and publish the Obi-Wan wing combo plates in TINMORRY PETG-GF.

Both split2 wing plates -- flat and graded -- on the 0.6 mm high-flow
nozzle, four pieces and one six-magnet pause each.

These are ordinary ready-to-print deliveries, unlike the structural core
next door in ``3mf_06hf_petg-gf_pla``.  The difference is support: the wings
print support-off by their own contract, so this lane's profile carries no
support recipe and therefore no PLA interface filament.  One filament means
the Bambu CLI slices it normally, instead of mapping a second filament to a
nozzle that does not exist.

The wing plates and the structural core each get their own profile, scoped
by ``artifact_scope`` to the artifacts they may print, and the wing plate
builder now enforces that scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import build_obiwan_wing_plate as wing_plate

PROFILE = (
    PROJECT_ROOT / "captive_magnet_slicing_profile_petg_gf_wings_06hf.json")
RELEASE_CATALOG = PROJECT_ROOT / "review" / "captive_magnet_release_catalog.json"
RELEASE_AUDIT = PROJECT_ROOT / "review" / "captive_magnet_slice_audit_06hf"
WORKSPACE = PROJECT_ROOT / "review" / "wing_plate_petg_gf_06hf"
from delivery_contract import PETG_WING_DIRECTORY

DEFAULT_OUTPUT = PROJECT_ROOT / "to_print" / "obiwan" / PETG_WING_DIRECTORY
EXPECTED_PAUSE_Z_MM = 5.96
EXPECTED_MAGNETS = 6
MODEL_FILAMENT = "TINMORRY PETG-GF Profile @BBL P2S"
LOCKED_PROCESS_ID = "LX521 ObiWan PETG-GF wings 0.6HF (locked - do not re-slice)"


class WingPetgError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(source: Path, destination: Path) -> None:
    """Copy the audited project, lock its process identity, freeze it.

    Same treatment the 0.6 shelf gives its own deliveries: the process is
    renamed to something no Bambu Studio install carries, so the GUI can
    never answer a same-name preset conflict by substituting installed
    values for the audited ones, and the file is made read-only so a Studio
    session cannot save its modified state back over it.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.chmod(stat.S_IWUSR | stat.S_IRUSR)
        destination.unlink()
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        payloads = {m.filename: archive.read(m.filename) for m in members}
    source_gcode = payloads["Metadata/plate_1.gcode"]
    settings = json.loads(
        payloads["Metadata/project_settings.config"].decode("utf-8"))
    settings["print_settings_id"] = LOCKED_PROCESS_ID
    payloads["Metadata/project_settings.config"] = json.dumps(
        settings, indent=4, ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.writestr(member.filename, payloads[member.filename])
    with zipfile.ZipFile(destination) as archive:
        if archive.read("Metadata/plate_1.gcode") != source_gcode:
            raise WingPetgError(
                f"{destination.name}: delivered G-code diverged from the "
                "audited source")
    destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _validate(audit: dict, gcode: Path, *, label: str) -> dict:
    """The wing contract, re-checked against what was actually emitted."""
    if audit.get("status") != "pass":
        raise WingPetgError(f"{label}: audit status {audit.get('status')!r}")
    supports = audit["support_toolpaths"]
    if (supports["support_feature_blocks"] != 0
            or supports["support_interface_feature_blocks"] != 0):
        raise WingPetgError(
            f"{label}: support-off wing plate emitted support toolpaths")
    pauses = audit["pause_before_first_layer_extrusion"]
    if len(pauses) != 1 or not pauses[0]["pass"]:
        raise WingPetgError(
            f"{label}: expected exactly one pause preceding its layer's "
            f"extrusion, got {pauses}")
    if abs(float(pauses[0]["z_mm"]) - EXPECTED_PAUSE_Z_MM) > 1.0e-6:
        raise WingPetgError(
            f"{label}: pause at Z={pauses[0]['z_mm']}, expected "
            f"{EXPECTED_PAUSE_Z_MM}")
    sites = [
        site["site"]
        for record in audit["captive_cavity_audit"].values()
        for site in record
    ]
    if len(set(sites)) != EXPECTED_MAGNETS:
        raise WingPetgError(
            f"{label}: {len(set(sites))} magnet sites, expected "
            f"{EXPECTED_MAGNETS}")

    text = gcode.read_text(encoding="utf-8", errors="replace")
    config = {}
    for line in text.splitlines():
        if line.startswith("; ") and " = " in line:
            key, _, value = line[2:].partition(" = ")
            config.setdefault(key.strip(), value.strip())
    # The whole point of this lane: one filament, the vendor's own figures,
    # and the high-flow variant the hotend actually is.
    for key, expected in (
            ("nozzle_temperature", "260"),
            ("filament_flow_ratio", "0.93"),
            ("filament_max_volumetric_speed", "12"),
            ("filament_retraction_length", "1"),
            ("nozzle_volume_type", "High Flow"),
            ("nozzle_diameter", "0.6")):
        if config.get(key) != expected:
            raise WingPetgError(
                f"{label}: G-code {key}={config.get(key)!r}, expected "
                f"{expected!r}")
    if MODEL_FILAMENT not in config.get("filament_settings_id", ""):
        raise WingPetgError(
            f"{label}: G-code filament is "
            f"{config.get('filament_settings_id')!r}")
    if "," in config.get("filament_settings_id", ""):
        raise WingPetgError(
            f"{label}: more than one filament loaded; this lane is "
            "single-filament by design")
    return {
        "magnets": len(set(sites)),
        "pause_z_mm": float(pauses[0]["z_mm"]),
        "triangle_count": int(audit["result"]["triangle_count"]),
    }


def _build_one(slug: str, output: Path) -> dict:
    api = wing_plate.get_variant(slug)
    api.activate()
    workspace = WORKSPACE / api.PLATE_NAME
    try:
        result = api.build_or_validate_ready_plate(
            workspace=workspace,
            profile_path=PROFILE,
            release_catalog=RELEASE_CATALOG,
            release_audit=RELEASE_AUDIT,
            system_root=None,
            bambu_binary=None,
            allow_slice=True,
        )
    except wing_plate.WingPlateError as exc:
        raise WingPetgError(f"{api.PLATE_NAME}: {exc}") from exc
    project = Path(result["project"])
    facts = _validate(
        result["audit"], Path(result["gcode"]), label=api.PLATE_NAME)
    destination = output / f"{api.PLATE_NAME}_06hf_petg-gf.gcode.3mf"
    _publish(project, destination)
    facts.update({
        "name": api.PLATE_NAME,
        "project": str(destination.relative_to(PROJECT_ROOT)),
        "project_sha256": _sha256(destination),
        "source_audit": str(
            Path(result["audit_path"]).relative_to(PROJECT_ROOT)),
        "reused": bool(result["reused"]),
    })
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variant", choices=sorted(wing_plate.VARIANTS), action="append",
        default=None, help="limit to one wing profile (default: both)")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    records = [
        _build_one(slug, output)
        for slug in (args.variant or sorted(wing_plate.VARIANTS))
    ]
    for record in records:
        print(f"{record['name']}: {record['project']} "
              f"(pause Z={record['pause_z_mm']} mm, {record['magnets']} "
              f"magnets, {record['triangle_count']} triangles)")
    manifest = output / "wing_plates.json"
    manifest.write_text(
        json.dumps({
            "kind": "petg_gf_wing_plates_ready_to_print",
            "lane": "0.6mm_high_flow",
            "filament": MODEL_FILAMENT,
            "support": "off by wing contract; single filament, no interface",
            "profile": str(PROFILE.relative_to(PROJECT_ROOT)),
            "plates": records,
        }, indent=1) + "\n", encoding="utf-8")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WingPetgError, wing_plate.WingPlateError) as exc:
        print(f"PETG-GF wing plates failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
