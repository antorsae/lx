#!/usr/bin/env python3
"""Emit GUI-sliceable PETG-GF projects that Bambu's CLI cannot slice itself.

The structural core wants the part in TINMORRY PETG-GF and the support
interface in Bambu PLA Basic, because a PETG interface welds to a PETG part
while PLA peels off at a zero Z gap.  Bambu Studio's CLI cannot deliver that
on this one-nozzle P2S: every structural path loads an assemble list (the
support blockers and the bridge/root modifier require one) and on that path
Studio resolves ``filament_map`` to ``1,0``, assigning the second filament to
a nozzle that does not exist.  It then prints the interface in the model
filament and still reports success, which is why
``_validate_actual_gcode_profile`` now refuses that G-code outright.

Only *slicing* is broken.  Exporting the same plate as a project -- the same
Bambu invocation with ``--slice`` removed -- works, and the project keeps
both filament presets, every process override, the locked placements, the
support blockers and modifier, and the six-magnet pause.  So this writes the
plate the owner opens in Bambu Studio, assigns PLA to its AMS slot, and
slices there.

The projects carry no G-code and are therefore not shelf deliverables, so
they live in ``3mf_06hf_petg-cf_pla`` rather than ``3mf_06hf``: same 0.6-mm
high-flow lane, named for the material pair it is sliced with, and separate
so nothing can confuse a project that still has to be sliced with an
audited, ready-to-print lane output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import build_obiwan_combo_plate as combo
import artifact_emit as emit

PETG_PROFILE = PROJECT_ROOT / "captive_magnet_slicing_profile_petg_gf_06hf.json"
RELEASE_CATALOG = PROJECT_ROOT / "review" / "captive_magnet_release_catalog.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "to_print" / "obiwan" / "3mf_06hf_petg-cf_pla")
WORKSPACE = PROJECT_ROOT / "review" / "petg_gui_project_workspace"
EXPECTED_PAUSE_Z_MM = 5.96
EXPECTED_MAGNETS = 6
# A process name that exists in no Bambu Studio install.  A project whose
# print_settings_id matches an installed preset gets the *installed* values:
# opening one shipped under the stock "0.18mm Balanced Quality @BBL P2S 0.6
# nozzle" name showed support disabled, tree(auto), Default/Default support
# filaments and a 0.18 mm top Z gap, none of which are what the project
# actually carries.  With no preset to collide with, Studio keeps ours.
GUI_PROCESS_ID = "LX521 ObiWan PETG-GF core 0.6HF (GUI)"
EXPECTED_NOZZLE_VOLUME = "High Flow"
MODEL_FILAMENT = "TINMORRY PETG-GF Profile @BBL P2S"
INTERFACE_FILAMENT = "Bambu PLA Basic @BBL P2S 0.6 nozzle"

# Standalone PETG-GF projects: everything Obi-Wan except the wings.  The
# audited singles carry captive-magnet stations and stage through the
# release-catalog dry-run (per-part profile overrides, blockers, pause);
# the plain singles have no stations and load their STL directly on the
# base 30%-gyroid, support-free process.
AUDITED_SINGLES = (
    ("obiwan_01_LM_bottom_keyed_1_of_2_no_floor_stand",
     "no_floor_stand:Obi-Wan-split:obiwan_optional_lm_keyed_1_of_2_bottom"),
    ("obiwan_01_LM_bottom_keyed_1_of_2_floor_stand",
     "floor_stand:Obi-Wan-split:obiwan_optional_lm_keyed_1_of_2_bottom"),
    ("obiwan_02_LM_top_keyed_2_of_2",
     "no_floor_stand:Obi-Wan-split:obiwan_optional_lm_keyed_2_of_2_top"),
    ("obiwan_03_UM_carrier_1_of_1",
     "no_floor_stand:Obi-Wan:obiwan_core_2_of_2_um_carrier"),
)
PLAIN_SINGLES = (
    ("obiwan_04_T_tweeter_crescent_1_of_1",
     "build/no_floor_stand/stl/obiwan_addon_tweeter_crescent.stl"),
    ("obiwan_NL8_service_lid_1_of_1",
     "build/floor_stand/stl/obiwan_addon_nl8_service_lid.stl"),
)


class GuiProjectError(RuntimeError):
    pass


def _export_command(prepared: dict, destination_name: str) -> list[str]:
    """The prepared slice command, turned into a project export.

    Dropping ``--slice`` is the whole trick: everything Studio does before
    slicing -- loading the assemble list, applying both filaments, placing
    the objects, attaching the custom G-code -- is exactly what the project
    needs, and none of it touches the broken filament map.
    """
    command = list(prepared["command"])
    try:
        index = command.index("--slice")
    except ValueError as exc:
        raise GuiProjectError(
            "prepared command has no --slice to remove") from exc
    del command[index:index + 2]
    try:
        export = command.index("--export-3mf")
    except ValueError as exc:
        raise GuiProjectError("prepared command has no --export-3mf") from exc
    command[export + 1] = destination_name
    return command


def _finalize_project_settings(project: Path) -> list[str]:
    """Make the project's own process survive being opened in the GUI.

    Two edits.  The process is renamed off the stock preset name so Studio
    cannot substitute an installed preset's values for the ones the project
    carries -- see GUI_PROCESS_ID.  And every loaded filament is given this
    printer's only nozzle: an un-sliced export records ``filament_map:
    ["1"]``, one entry for two filaments, because the map is a slicing
    product, so a complete map means the picker opens with both filaments
    assigned to nozzle 1 rather than showing the second one as unmapped.
    """
    member = "Metadata/project_settings.config"
    with zipfile.ZipFile(project) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    settings = json.loads(entries[member])
    filaments = settings.get("filament_settings_id")
    if not isinstance(filaments, list) or not filaments:
        raise GuiProjectError(f"{project}: project declares no filaments")
    settings["filament_map"] = ["1"] * len(filaments)
    settings["print_settings_id"] = GUI_PROCESS_ID
    # Studio falls back to the first entry of extruder_variant_list, so an
    # exported project claims "Standard" on a machine whose own
    # default_nozzle_volume_type is High Flow.  That is the wrong hardware --
    # it picks the Standard column of every per-variant filament value,
    # including the volumetric ceiling -- so pin the machine's own default.
    default_volume = settings.get("default_nozzle_volume_type")
    if not isinstance(default_volume, list) or not default_volume:
        raise GuiProjectError(
            f"{project}: machine declares no default_nozzle_volume_type")
    if default_volume[0] != EXPECTED_NOZZLE_VOLUME:
        raise GuiProjectError(
            f"{project}: machine default nozzle is {default_volume[0]!r}, "
            f"expected {EXPECTED_NOZZLE_VOLUME!r}")
    settings["nozzle_volume_type"] = list(default_volume)
    entries[member] = (
        json.dumps(settings, indent=4) + "\n").encode("utf-8")
    temporary = project.with_suffix(".3mf.tmp")
    with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    temporary.replace(project)
    return list(filaments)


def _validate(
    project: Path,
    *,
    label: str,
    expected_infill: str,
    expected_pattern: str,
    expected_parts: int = 4,
) -> dict:
    """Refuse to hand over a project that lost anything on the way out.

    The infill expectations are per plate: the no-floor plate prints 40%
    gyroid with a 100%-solid modifier through the bridge/root, while the
    floor plate keeps the integral floor's global 100% zig-zag.  Both are
    values a substituted stock preset silently replaces with 15%.
    """
    with zipfile.ZipFile(project) as archive:
        names = set(archive.namelist())
        for required in (
                "Metadata/project_settings.config",
                "Metadata/model_settings.config",
                "Metadata/custom_gcode_per_layer.xml",
                "3D/3dmodel.model"):
            if required not in names:
                raise GuiProjectError(f"{label}: project lacks {required}")
        settings = json.loads(
            archive.read("Metadata/project_settings.config"))
        models = archive.read("Metadata/model_settings.config").decode("utf-8")
        pause_xml = archive.read("Metadata/custom_gcode_per_layer.xml")
        if "Metadata/plate_1.gcode" in names:
            raise GuiProjectError(
                f"{label}: project carries G-code and would look like an "
                "audited shelf delivery")

    filaments = settings.get("filament_settings_id")
    if filaments != [MODEL_FILAMENT, INTERFACE_FILAMENT]:
        raise GuiProjectError(
            f"{label}: filaments are {filaments!r}, expected the PETG-GF "
            "model filament and the PLA interface")
    for key, expected in (
            ("support_filament", "1"),
            ("support_interface_filament", "2"),
            ("enable_support", "1"),
            ("top_shell_layers", "10"),
            ("ironing_type", "top")):
        if str(settings.get(key)) != expected:
            raise GuiProjectError(
                f"{label}: {key}={settings.get(key)!r}, expected {expected!r}")
    if settings.get("filament_map") != ["1", "1"]:
        raise GuiProjectError(
            f"{label}: filament_map={settings.get('filament_map')!r}, "
            "expected both filaments on nozzle 1")
    if settings.get("nozzle_volume_type") != [EXPECTED_NOZZLE_VOLUME]:
        raise GuiProjectError(
            f"{label}: nozzle_volume_type="
            f"{settings.get('nozzle_volume_type')!r}, expected "
            f"[{EXPECTED_NOZZLE_VOLUME!r}] to match the installed hotend")
    if settings.get("print_settings_id") != GUI_PROCESS_ID:
        raise GuiProjectError(
            f"{label}: process ships as "
            f"{settings.get('print_settings_id')!r}; under a name Bambu "
            "Studio has installed, it replaces every value below with the "
            "installed preset's")
    for key, expected in (
            ("support_type", "normal(auto)"),
            ("support_style", "snug"),
            ("support_top_z_distance", "0"),
            ("support_interface_spacing", "0"),
            ("support_on_build_plate_only", "1"),
            ("support_object_xy_distance", "0.7"),
            ("sparse_infill_density", expected_infill),
            ("sparse_infill_pattern", expected_pattern)):
        if str(settings.get(key)) != expected:
            raise GuiProjectError(
                f"{label}: {key}={settings.get(key)!r}, expected {expected!r}")

    root = ElementTree.fromstring(pause_xml)
    layers = [
        layer for layer in root.iter()
        if layer.tag.rsplit("}", 1)[-1] == "layer"
    ]
    if len(layers) != 1:
        raise GuiProjectError(
            f"{label}: project carries {len(layers)} custom G-code layers, "
            "expected exactly the one magnet pause")
    pause_z = float(layers[0].get("top_z", "nan"))
    if abs(pause_z - EXPECTED_PAUSE_Z_MM) > 1.0e-6:
        raise GuiProjectError(
            f"{label}: magnet pause is at Z={pause_z}, expected "
            f"{EXPECTED_PAUSE_Z_MM}")
    program = layers[0].get("extra", "")
    magnets = program.count(",") + 1 if "Insert" in program else 0
    if f"Insert {EXPECTED_MAGNETS} magnet" not in program:
        raise GuiProjectError(
            f"{label}: pause does not announce {EXPECTED_MAGNETS} magnets")
    if "M400" not in program:
        raise GuiProjectError(f"{label}: pause program has no M400 park")

    counts = {
        subtype: models.count(f'subtype="{subtype}"')
        for subtype in ("normal_part", "support_blocker", "modifier_part")
    }
    if counts["normal_part"] != expected_parts:
        raise GuiProjectError(
            f"{label}: project holds {counts['normal_part']} parts, "
            f"expected {expected_parts}")
    if counts["support_blocker"] != 3:
        raise GuiProjectError(
            f"{label}: project holds {counts['support_blocker']} duct "
            "blockers, expected three")
    return {
        "filaments": filaments,
        "pause_z_mm": pause_z,
        "magnets": EXPECTED_MAGNETS,
        "parts": counts,
    }


def _validate_single(project: Path, *, label: str, expects: dict) -> dict:
    """Single-part flavour of _validate, driven by per-part expectations."""
    with zipfile.ZipFile(project) as archive:
        names = set(archive.namelist())
        for required in (
                "Metadata/project_settings.config",
                "Metadata/model_settings.config",
                "3D/3dmodel.model"):
            if required not in names:
                raise GuiProjectError(f"{label}: project lacks {required}")
        settings = json.loads(
            archive.read("Metadata/project_settings.config"))
        models = archive.read("Metadata/model_settings.config").decode("utf-8")
        pause_xml = (
            archive.read("Metadata/custom_gcode_per_layer.xml")
            if "Metadata/custom_gcode_per_layer.xml" in names else None)
        if "Metadata/plate_1.gcode" in names:
            raise GuiProjectError(
                f"{label}: project carries G-code and would look like an "
                "audited shelf delivery")
    filaments = settings.get("filament_settings_id")
    expected_filaments = (
        [MODEL_FILAMENT, INTERFACE_FILAMENT]
        if expects["support"] else [MODEL_FILAMENT])
    if filaments != expected_filaments:
        raise GuiProjectError(
            f"{label}: filaments are {filaments!r}, "
            f"expected {expected_filaments!r}")
    if settings.get("filament_map") != ["1"] * len(filaments):
        raise GuiProjectError(
            f"{label}: filament_map={settings.get('filament_map')!r}")
    if settings.get("nozzle_volume_type") != [EXPECTED_NOZZLE_VOLUME]:
        raise GuiProjectError(
            f"{label}: nozzle_volume_type="
            f"{settings.get('nozzle_volume_type')!r}")
    if settings.get("print_settings_id") != GUI_PROCESS_ID:
        raise GuiProjectError(
            f"{label}: process ships as "
            f"{settings.get('print_settings_id')!r}")
    checks = [
        ("enable_support", "1" if expects["support"] else "0"),
        ("sparse_infill_density", expects["infill"]),
        ("sparse_infill_pattern", expects["pattern"]),
    ]
    if expects["support"]:
        checks += [
            ("support_filament", "1"),
            ("support_interface_filament", "2"),
            ("support_on_build_plate_only", "1"),
        ]
    for key, expected in checks:
        if str(settings.get(key)) != expected:
            raise GuiProjectError(
                f"{label}: {key}={settings.get(key)!r}, "
                f"expected {expected!r}")
    magnets = int(expects["magnets"])
    pause_z = None
    if magnets:
        if pause_xml is None:
            raise GuiProjectError(f"{label}: magnet pause layer is missing")
        root = ElementTree.fromstring(pause_xml)
        layers = [
            layer for layer in root.iter()
            if layer.tag.rsplit("}", 1)[-1] == "layer"]
        if len(layers) != 1:
            raise GuiProjectError(
                f"{label}: {len(layers)} custom G-code layers, expected 1")
        pause_z = float(layers[0].get("top_z", "nan"))
        if abs(pause_z - EXPECTED_PAUSE_Z_MM) > 1.0e-6:
            raise GuiProjectError(
                f"{label}: magnet pause at Z={pause_z}")
        program = layers[0].get("extra", "")
        if f"Insert {magnets} magnet(s)" not in program:
            raise GuiProjectError(
                f"{label}: pause does not announce {magnets} magnet(s)")
        if "M400" not in program:
            raise GuiProjectError(f"{label}: pause program has no M400 park")
    elif pause_xml is not None:
        root = ElementTree.fromstring(pause_xml)
        layers = [
            layer for layer in root.iter()
            if layer.tag.rsplit("}", 1)[-1] == "layer"]
        if layers:
            raise GuiProjectError(
                f"{label}: unexpected custom G-code layers on a part "
                "with no magnet stations")
    counts = {
        subtype: models.count(f'subtype="{subtype}"')
        for subtype in ("normal_part", "support_blocker", "modifier_part")
    }
    if counts["normal_part"] != 1:
        raise GuiProjectError(f"{label}: part inventory {counts}")
    if counts["modifier_part"] != expects.get("modifiers", 0):
        raise GuiProjectError(
            f"{label}: {counts['modifier_part']} modifiers, "
            f"expected {expects.get('modifiers', 0)}")
    if counts["support_blocker"] != expects["blockers"]:
        raise GuiProjectError(
            f"{label}: {counts['support_blocker']} blockers, "
            f"expected {expects['blockers']}")
    return {
        "filaments": filaments,
        "pause_z_mm": pause_z,
        "magnets": magnets,
        "parts": counts,
    }


def _resolved_process_expectations(process_path: Path) -> tuple[str, str, bool]:
    resolved = json.loads(process_path.read_text(encoding="utf-8"))
    def scalar(key):
        value = resolved.get(key)
        return value[0] if isinstance(value, list) and value else value
    support = str(scalar("enable_support")) == "1"
    return (str(scalar("sparse_infill_density")),
            str(scalar("sparse_infill_pattern")), support)


def _pause_policy(profiles_dir: Path) -> dict:
    provenance = json.loads(
        (profiles_dir / "profile_provenance.json").read_text(
            encoding="utf-8"))
    policy = provenance.get("effective", {}).get("magnet_insertion_pause")
    if not isinstance(policy, dict):
        raise GuiProjectError(
            "resolved PETG-GF profile lacks the magnet insertion pause "
            "policy")
    return policy


def _remap_assemble_paths(assemble: Path) -> None:
    """Point staged object paths back at the durable build/ originals.

    The dry-run stages its inputs in a TemporaryDirectory that no longer
    exists by the time this export runs.
    """
    payload = json.loads(assemble.read_text(encoding="utf-8"))
    candidates = {}
    for state in ("floor_stand", "no_floor_stand"):
        for sub in ("stl", "support_blockers", "process_modifiers"):
            root = PROJECT_ROOT / "build" / state / sub
            if root.is_dir():
                for path in root.glob("*.stl"):
                    candidates.setdefault(f"{state}/{path.name}", path)
                    candidates.setdefault(path.name, path)
    for plate in payload.get("plates", ()):
        for obj in plate.get("objects", ()):
            name = Path(obj["path"]).name
            replacement = candidates.get(name)
            if replacement is None or not replacement.is_file():
                raise GuiProjectError(
                    f"{assemble}: no durable source for {name}")
            obj["path"] = str(replacement)
    assemble.write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _dry_run_single(name: str, artifact_id: str, workspace: Path) -> dict:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/slice_captive_magnets.py"),
         "--catalog", str(RELEASE_CATALOG),
         "--profile", str(PETG_PROFILE),
         "--output", str(workspace),
         "--jobs", "1", "--dry-run", "--only", artifact_id],
        cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
        env={**os.environ, "LC_ALL": "C",
             "LX_ROUTING_PROFILE": "obiwan"})
    (workspace / "dry_run.log").write_text(
        run.stdout, encoding="utf-8", errors="replace")
    if run.returncode != 0:
        raise GuiProjectError(
            f"{name}: dry-run staging exited {run.returncode}; see "
            f"{workspace / 'dry_run.log'}")
    records = json.loads(
        (workspace / "dry_run_commands.json").read_text(
            encoding="utf-8"))["records"]
    if len(records) != 1 or records[0]["id"] != artifact_id:
        raise GuiProjectError(f"{name}: dry-run staged {len(records)} records")
    return records[0]


def _catalog_sites(artifact_id: str) -> list[dict]:
    catalog = json.loads(RELEASE_CATALOG.read_text(encoding="utf-8"))
    for row in catalog["artifacts"]:
        if row["id"] == artifact_id:
            return list(row.get("sites", ()))
    raise GuiProjectError(f"{artifact_id}: not in the release catalog")


def _build_single_audited(name: str, artifact_id: str, output: Path) -> dict:
    workspace = WORKSPACE / "singles" / name
    record = _dry_run_single(name, artifact_id, workspace)
    slug = artifact_id.replace(":", "_")
    slice_dir = workspace / "slices" / slug
    assemble = slice_dir / "bambu_assemble_list.json"
    _remap_assemble_paths(assemble)
    command = list(record["command"])
    sites = _catalog_sites(artifact_id)
    if sites:
        pause_values = {
            float(site["expected_pause_marker_z_mm"]) for site in sites}
        if pause_values != {EXPECTED_PAUSE_Z_MM}:
            raise GuiProjectError(
                f"{name}: pause markers {sorted(pause_values)}")
        group = {
            "pause_marker_z_mm": EXPECTED_PAUSE_Z_MM,
            "sites": [str(site["name"]) for site in sites],
            "magnet_count": len(sites),
        }
        custom = slice_dir / "custom_gcodes.json"
        custom.write_text(json.dumps({
            "mode": "SingleExtruder",
            "gcodes": [{
                "type": emit.MAGNET_INSERTION_CUSTOM_GCODE_TYPE,
                "print_z": EXPECTED_PAUSE_Z_MM,
                "color": "",
                "extruder": 1,
                "extra": emit._magnet_pause_program(
                    group, _pause_policy(workspace / "profiles")),
            }],
        }, indent=1) + "\n", encoding="utf-8")
        command.extend(("--load-custom-gcodes", str(custom)))
    export_dir = workspace / "gui_project"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    project_name = f"{name}_GUI.3mf"
    command = _export_command({"command": command}, project_name)
    outputdir = command.index("--outputdir")
    command[outputdir + 1] = str(export_dir)
    run = subprocess.run(
        command, cwd=export_dir, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
        env={**os.environ, "LC_ALL": "C"})
    (export_dir / "bambu_studio.log").write_text(
        run.stdout, encoding="utf-8", errors="replace")
    project = export_dir / project_name
    if run.returncode != 0 or not project.is_file():
        raise GuiProjectError(
            f"{name}: Bambu exited {run.returncode}; see "
            f"{export_dir / 'bambu_studio.log'}")
    _finalize_project_settings(project)
    payload = json.loads(assemble.read_text(encoding="utf-8"))
    blockers = sum(
        1 for plate in payload.get("plates", ())
        for obj in plate.get("objects", ())
        if obj.get("subtype") == "support_blocker")
    modifiers = sum(
        1 for plate in payload.get("plates", ())
        for obj in plate.get("objects", ())
        if obj.get("subtype") == "modifier_part")
    infill, pattern, support = _resolved_process_expectations(
        slice_dir / "slice_profile" / "resolved_process.json")
    facts = _validate_single(project, label=name, expects={
        "support": support, "infill": infill, "pattern": pattern,
        "magnets": len(sites), "blockers": blockers,
        "modifiers": modifiers,
    })
    output.mkdir(parents=True, exist_ok=True)
    destination = output / project_name
    shutil.copy2(project, destination)
    facts["project"] = str(destination.relative_to(PROJECT_ROOT))
    facts["name"] = name
    facts["kind"] = "single"
    return facts


def _build_single_plain(
        name: str, stl_relative: str, output: Path,
        template_command: list[str], profiles_dir: Path) -> dict:
    workspace = WORKSPACE / "singles" / name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    stl = PROJECT_ROOT / stl_relative
    if not stl.is_file():
        raise GuiProjectError(f"{name}: missing source STL {stl}")
    project_name = f"{name}_GUI.3mf"
    settings = ";".join(str(profiles_dir / part) for part in (
        "resolved_machine.json", "resolved_process.json"))
    command = [
        template_command[0], "--debug", "2", "--arrange", "1",
        "--orient", "0", "--allow-rotations=0",
        "--export-3mf", project_name,
        "--load-settings", settings,
        "--load-filaments", str(profiles_dir / "resolved_filament.json"),
        "--outputdir", str(workspace),
        str(stl),
    ]
    run = subprocess.run(
        command, cwd=workspace, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
        env={**os.environ, "LC_ALL": "C"})
    (workspace / "bambu_studio.log").write_text(
        run.stdout, encoding="utf-8", errors="replace")
    project = workspace / project_name
    if run.returncode != 0 or not project.is_file():
        raise GuiProjectError(
            f"{name}: Bambu exited {run.returncode}; see "
            f"{workspace / 'bambu_studio.log'}")
    _finalize_project_settings(project)
    infill, pattern, support = _resolved_process_expectations(
        profiles_dir / "resolved_process.json")
    if support:
        raise GuiProjectError(
            f"{name}: base process enables support; the plain single "
            "path expects the support-free base")
    facts = _validate_single(project, label=name, expects={
        "support": False, "infill": infill, "pattern": pattern,
        "magnets": 0, "blockers": 0,
    })
    output.mkdir(parents=True, exist_ok=True)
    destination = output / project_name
    shutil.copy2(project, destination)
    facts["project"] = str(destination.relative_to(PROJECT_ROOT))
    facts["name"] = name
    facts["kind"] = "single"
    return facts


def _build_one(slug: str, output: Path) -> dict:
    api = combo.get_variant(slug)
    api.activate()
    workspace = WORKSPACE / api.PLATE_NAME
    prepared = combo._prepare_slice(
        workspace=workspace,
        profile_path=PETG_PROFILE,
        release_catalog=RELEASE_CATALOG,
        system_root=None,
        bambu_binary=None,
    )
    export_dir = workspace / "gui_project"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    name = f"{api.PLATE_NAME}_GUI.3mf"
    command = _export_command(prepared, name)
    outputdir = command.index("--outputdir")
    command[outputdir + 1] = str(export_dir)
    run = subprocess.run(
        command, cwd=export_dir, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
        env={**os.environ, "LC_ALL": "C"})
    (export_dir / "bambu_studio.log").write_text(
        run.stdout, encoding="utf-8", errors="replace")
    project = export_dir / name
    if run.returncode != 0 or not project.is_file():
        raise GuiProjectError(
            f"{api.PLATE_NAME}: Bambu exited {run.returncode}; see "
            f"{export_dir / 'bambu_studio.log'}")
    _finalize_project_settings(project)
    facts = _validate(
        project,
        label=api.PLATE_NAME,
        expected_infill=(
            f"{api.variant.sparse_infill_density_percent:g}%"),
        expected_pattern=api.variant.sparse_infill_pattern,
        expected_parts=len(api.PARTS),
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / name
    shutil.copy2(project, destination)
    facts["project"] = str(destination.relative_to(PROJECT_ROOT))
    facts["name"] = api.PLATE_NAME
    return facts


README = """# PETG-GF structural plates for GUI slicing

These are Bambu Studio **projects**, not sliced deliveries: they carry no
G-code, and they are not part of the audited `3mf_06hf` shelf. Same 0.6-mm
high-flow lane, kept separate because they still have to be sliced.

Bambu's CLI cannot slice them. Every structural plate loads an assemble list
(the duct blockers and the bridge/root modifier need one) and on that path
Studio maps the second filament to nozzle 0, which does not exist, then
prints the support interface in the model filament and reports success. The
repo now refuses that G-code, so these plates are handed over as projects
instead. See `docs/PRINTING.md` for the full finding.

## What is already set

* filament 1 = `TINMORRY PETG-GF Profile @BBL P2S`, filament 2 =
  `Bambu PLA Basic @BBL P2S 0.6 nozzle`, both mapped to the single nozzle
* supports on, printed in filament 1, interface in filament 2, zero top Z
  gap, snug, `support_on_build_plate_only`
* 10 top shell layers and ironing on top surfaces
* the four core pieces at their locked placements, all three duct blockers,
  the 100%-solid bridge/root modifier
* the six-magnet pause at Z = 5.96 mm, with its park/restore program

## What to do

1. Open the project in Bambu Studio 02.07.01.62.
2. Assign filament 1 to the **AMS-HT** slot holding TINMORRY PETG-GF, and
   filament 2 to the **AMS** slot holding PLA (slot 2 or 3).
3. Slice. Confirm before printing:
   * the support **interface** is filament 2 and the support **body** is
     filament 1 -- if both come out filament 1 the mapping did not take;
   * the pause still sits at Z = 5.96 mm and announces six magnets;
   * a prime tower is placed clear of all four parts.
4. Insert the six magnets at the pause, then resume.

Do not print these alongside the individual 01/02/03/04 files, and do not
mix the two stand states.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variant", choices=sorted(combo.VARIANTS), action="append",
        default=None, help="limit to one stand state (default: both)")
    args = parser.parse_args(argv)

    slugs = args.variant or sorted(combo.VARIANTS)
    records = []
    for slug in slugs:
        record = _build_one(slug, args.output.resolve())
        record["kind"] = "plate"
        records.append(record)
        print(f"{record['name']}: {record['project']} "
              f"(pause Z={record['pause_z_mm']} mm, "
              f"{record['magnets']} magnets, "
              f"{record['parts']['normal_part']} parts, "
              f"{record['parts']['support_blocker']} blockers)")
    template_command: list[str] | None = None
    profiles_dir: Path | None = None
    for name, artifact_id in AUDITED_SINGLES:
        record = _build_single_audited(
            name, artifact_id, args.output.resolve())
        records.append(record)
        if template_command is None:
            staged = json.loads(
                (WORKSPACE / "singles" / name
                 / "dry_run_commands.json").read_text(encoding="utf-8"))
            template_command = list(staged["records"][0]["command"])
            profiles_dir = WORKSPACE / "singles" / name / "profiles"
        print(f"{record['name']}: {record['project']} "
              f"({record['magnets']} magnets, "
              f"{record['parts']['support_blocker']} blockers)")
    if template_command is None or profiles_dir is None:
        raise GuiProjectError(
            "plain singles need an audited single's resolved profiles")
    for name, stl_relative in PLAIN_SINGLES:
        record = _build_single_plain(
            name, stl_relative, args.output.resolve(),
            template_command, profiles_dir)
        records.append(record)
        print(f"{record['name']}: {record['project']} (plain, no magnets)")
    (args.output.resolve() / "README.md").write_text(README, encoding="utf-8")
    manifest = args.output.resolve() / "gui_projects.json"
    manifest.write_text(
        json.dumps({
            "kind": "petg_gf_gui_projects_no_gcode",
            "reason": (
                "Bambu CLI maps a second filament to nozzle 0 on the "
                "assemble-list path; slice these in the GUI"),
            "model_filament": MODEL_FILAMENT,
            "interface_filament": INTERFACE_FILAMENT,
            "projects": records,
        }, indent=1) + "\n", encoding="utf-8")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GuiProjectError, combo.ComboPlateError) as exc:
        print(f"PETG GUI projects failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
