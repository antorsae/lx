"""H2C native profile resolution and explicit, centralized process mapping."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

from ..io import sha256_file
from ..print_policy import policy as shared_policy, role_settings

ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / "print_policy_h2c.json"
BBL = Path.home() / "Library/Application Support/BambuStudio/system/BBL"
META = {"name", "type", "from", "instantiation", "setting_id", "filament_id", "inherits",
        "include", "description", "version", "compatible_printers", "compatible_prints",
        "compatible_printers_condition", "compatible_prints_condition", "filament_settings_id",
        "filament_ingredients_safe", "filament_emission_safe", "filament_contact_safe"}


def policy():
    return json.loads(POLICY.read_text())


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def resolve_profiles(lane: str, directory: Path):
    from release_validation import PresetResolver, _standalone_cli_config
    cfg = policy()
    resolver = PresetResolver(BBL)
    machine = _standalone_cli_config("machine", resolver.resolve(BBL / cfg["machine_preset"]))
    machine["default_nozzle_volume_type"] = cfg["nozzle_volume_type"]
    machine["nozzle_volume_type"] = cfg["nozzle_volume_type"]
    process = _standalone_cli_config("process", resolver.resolve(BBL / cfg["process_preset"]))
    for key,value in cfg['native_process_defaults'].items():process.setdefault(key,value)
    # Quality settings are shared; machine speeds and retraction vectors retain
    # H2C's four entries (left/right x Standard/HF).
    legacy = json.loads((ROOT / "captive_magnet_slicing_profile_petg_gf_06hf.json").read_text())
    for key, val in legacy["repo_overrides"]["process"].items():
        if key.startswith("wipe_tower") or key == "curr_bed_type": continue
        shape = process.get(key)
        if isinstance(val, list) and isinstance(shape, list) and len(shape) == 4 and len(val) == 2:
            val = val * 2
        process[key] = val
    process.update(shared_policy()["support"])
    process.update(support_bottom_z_distance="0", support_bottom_interface_spacing="0",
        curr_bed_type=cfg["plate"], bed_temperature_formula="by_first_filament",
        prime_tower_width="50", prime_tower_brim_width="2", enable_prime_tower="1",
        brim_type="outer_only", brim_width="5", print_sequence="by layer",
        support_on_build_plate_only="0", support_critical_regions_only="0",
        support_remove_small_overhang="0", enable_support="1")
    lane_cfg = cfg["lanes"][lane]
    filaments = []
    for kind in ("model", "interface"):
        material = _standalone_cli_config("filament", resolver.resolve(BBL / lane_cfg[kind + "_preset"]))
        for key, value in lane_cfg[kind + "_overrides"].items():
            native = material.get(key)
            if isinstance(native, list) and not isinstance(value, list):
                value = [value] * len(native)
            material[key] = value
        material["compatible_printers"] = [machine["name"]]
        if kind == "model": material["name"] = lane_cfg["model_name"]
        material["filament_settings_id"] = [material["name"]]
        # A dual-material plate is one physical temperature, even after swaps.
        for k in ("eng_plate_temp", "eng_plate_temp_initial_layer"):
            material[k] = [str(cfg["bed_temperature_c"])]
        filaments.append(material)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in zip(("machine", "process", "model", "interface"), (machine, process, *filaments)):
        write_json(directory / (name + ".json"), data)
    for path in resolver.dependencies:
        target = directory / "system_sources" / path.relative_to(BBL)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(directory / "provenance.json", dict(policy_sha256=sha256_file(POLICY),
        shared_policy_sha256=sha256_file(ROOT / "print_policy.json"),
        native_sources={str(p.relative_to(BBL)): sha256_file(p) for p in sorted(resolver.dependencies)}))
    return machine, process, filaments


def project_settings(bundle, lane: str, role: str, *, brim=5, tower=(270, 255)):
    cfg = policy()
    machine, process, materials = bundle
    result = {k:deepcopy(v) for layer in (machine, process) for k,v in layer.items() if k not in META}
    # Preserve both nozzle-variant columns per material. Lists of metadata
    # such as AMS drying modes stay one active entry per material instead.
    for key in (set(materials[0]) | set(materials[1])) - META:
        a, b = [m.get(key) for m in materials]
        if a is None or b is None:
            if key in {"first_x_layer_fan_speed", "pre_start_fan_time"}:
                a, b = a or ["0"], b or ["0"]  # Native PrintConfig defaults.
            else: raise ValueError(f"Asymmetric H2C material key {key}: {a!r}, {b!r}")
        if isinstance(a, list):
            if not isinstance(b, list): raise ValueError(key)
            if key.startswith("filament_dev_ams_drying_") or len(a) > 2 or len(b) > 2:
                a, b = a[:1], b[:1]
            # Some native values vary by hotend; expand a scalar side only.
            columns = max(len(a), len(b))
            result[key] = (a * columns if len(a) == 1 else a) + (b * columns if len(b) == 1 else b)
        elif a == b:
            result[key] = a
        else:
            raise ValueError(f"Conflicting scalar material key {key}: {a!r}, {b!r}")
    result.update(role_settings(role))
    result.update(printer_settings_id=machine["name"],
        print_settings_id=f"LX521 H2C 0.6HF {lane} {role}",
        filament_settings_id=[m["name"] for m in materials],
        filament_ids=[m["filament_id"] for m in materials],
        filament_self_index=["1", "1", "2", "2"],
        filament_colour=cfg["lanes"][lane]["colours"],
        default_filament_colour=cfg["lanes"][lane]["colours"],
        filament_map_mode="Manual", filament_map=["1", "2"],
        filament_map_2=["1", "2"], filament_nozzle_map=["1", "2"],
        filament_volume_map=["1", "1"], nozzle_volume_type=["High Flow", "High Flow"],
        extruder_nozzle_stats=["High Flow#1", "High Flow#1"],
        enable_filament_dynamic_map="0", flush_into_objects="0", flush_into_infill="0", flush_into_support="0",
        # Kept for native serialization; physically separate nozzles use the
        # H2C priming volumes and do not flush model/PLA through each other.
        flush_volumes_matrix=["0", "280", "280", "0"], flush_volumes_vector=["140"] * 4,
        wipe_tower_x=[str(tower[0])], wipe_tower_y=[str(tower[1])], brim_width=str(brim))
    result["compatible_printers"] = [machine["name"]]
    result.update(cfg.get('role_overrides',{}).get(role,{}))
    result.update(cfg.get('lane_role_overrides',{}).get(lane,{}).get(role,{}))
    return result


def job_process(bundle, settings):
    """A standalone CLI process with the same effective per-job policy."""
    process = deepcopy(bundle[1])
    for key in process:
        if key in settings and key not in META:
            process[key] = deepcopy(settings[key])
    # Explicit process overrides can be absent from a sparse inherited
    # native preset. Include them so --load-settings cannot reset the 3MF.
    keys=set(policy()['native_process_defaults'])
    keys.update(k for r in policy().get('role_overrides',{}).values() for k in r)
    keys.update(k for lane in policy().get('lane_role_overrides',{}).values() for r in lane.values() for k in r)
    for key in keys:
        if key in settings:process[key]=deepcopy(settings[key])
    process.update(name=settings["print_settings_id"], print_settings_id=settings["print_settings_id"],
        compatible_printers=[bundle[0]["name"]], compatible_printers_condition="")
    return process


def job_settings(bundle, lane, role, name, **kwargs):
    settings = project_settings(bundle, lane, role, **kwargs)
    settings.update(policy().get('part_process_overrides', {}).get(name, {}))
    return settings
