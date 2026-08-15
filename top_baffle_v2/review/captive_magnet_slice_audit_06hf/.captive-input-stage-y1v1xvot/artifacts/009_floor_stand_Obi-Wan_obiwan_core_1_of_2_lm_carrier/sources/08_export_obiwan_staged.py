"""Guarded native-BREP staging for Obi-Wan R6F release artifacts.

The canonical LM carrier remains monolithic.  The same finalized LM BREP is
also subdivided into a mutually-exclusive top/bottom hidden-keyed print
option, so route lumens and every state-specific interface remain identical
to the one-piece source.

The local macOS profile cannot reliably build and hollow the face-rich LM
carrier in one OCC process.  A lightweight, stdlib-only parent therefore
runs its outer blank, every exact route-cutter group, and final functional
recuts in separate guarded descendants there.  On the osado large-memory
profile the same carrier is built directly in one guarded worker; the 21
cutter-group count remains a checked geometry contract, not an execution
split.  Other printed parts and review proxies are serialized to native BREP
before a final import-only STEP assembly worker.

Public commands::

    python export_obiwan_staged.py stage --manifest STATE/.obiwan_stage/manifest.json
    python export_obiwan_staged.py step --manifest ... --kind split --output ...

Every direct invocation is wrapped by ``run_memory_guarded.py``.  Descendants
inherit that live SID/PGID and never establish a second guard or process
group, so the selected local/remote memory profile remains authoritative.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import subprocess
import sys
import tempfile
import time

import run_memory_guarded as memory_guard
from lx521_baffle.io import pretty_json_bytes, sha256_file


SCRIPT = Path(__file__).resolve()
ROOT = PROJECT_ROOT
SCHEMA_VERSION = 7
EXPECTED_LM_CUTTER_GROUP_COUNT = 21
FIXED_TIMESTAMP = "2020-01-01T00:00:00"
WORKER_HEADROOM_MIB = 3200.0
RUNTIME_DISTRIBUTIONS = ("build123d", "cadquery-ocp", "numpy", "shapely")

REFERENCE_INPUTS = (
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_driver.stl",
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_driver_STL_notes.md",
    ROOT.parent / "linkwitz" / "H1658-04_MU10RB-SL_Datasheet.pdf",
    ROOT.parent / "E0022_W22EX001.stp",
)
SOURCE_INPUTS = tuple(sorted((
    *(ROOT / "src/lx521_baffle").rglob("*.py"),
    ROOT / "scripts/export_steps.py",
)))


def _runtime_identity() -> dict:
    packages = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = "missing"
    return {"python": sys.version, "packages": packages}


def _guard_policy() -> dict:
    worker_headroom = (
        WORKER_HEADROOM_MIB if memory_guard.MIN_FREE_MB else 0.0)
    # The stage policy describes the heavy build tier that produces staged
    # geometry.  A light-declared reader's own admission tier must not alter
    # the fingerprint, or every light consumer would reject a valid stage.
    return {
        "memory_profile": memory_guard.MEMORY_PROFILE,
        "max_process_tree_rss_mib": memory_guard.HEAVY_MAX_RSS_MB,
        "min_immediately_reclaimable_mib": memory_guard.MIN_FREE_MB,
        "guard_slots": memory_guard.HEAVY_GUARD_SLOTS,
        "worker_launch_headroom_mib": worker_headroom,
        "aggregate_cgroup_max_mib": memory_guard.CGROUP_MEMORY_MAX_MIB,
    }


def _validate_guard_policy_record(policy: object) -> dict:
    expected_keys = {
        "memory_profile", "max_process_tree_rss_mib",
        "min_immediately_reclaimable_mib", "guard_slots",
        "worker_launch_headroom_mib", "aggregate_cgroup_max_mib",
    }
    if not isinstance(policy, dict) or set(policy) != expected_keys:
        raise RuntimeError("Obi-Wan stage guard-policy record is malformed")
    profile_name = policy["memory_profile"]
    if profile_name not in memory_guard.MEMORY_PROFILES:
        raise RuntimeError("Obi-Wan stage memory-profile record is unknown")
    profile = memory_guard.MEMORY_PROFILES[profile_name]
    maximum = policy["max_process_tree_rss_mib"]
    floor = policy["min_immediately_reclaimable_mib"]
    slots = policy["guard_slots"]
    headroom = policy["worker_launch_headroom_mib"]
    aggregate = policy["aggregate_cgroup_max_mib"]
    expected_headroom = WORKER_HEADROOM_MIB if floor else 0.0
    if (type(maximum) is not int or not 0 < maximum <= profile["max_rss_mb"]
            or type(floor) is not int or floor < profile["min_free_mb"]
            or type(slots) is not int
            or not 1 <= slots <= profile["max_guard_slots"]
            or not isinstance(headroom, (int, float))
            or isinstance(headroom, bool)
            or not math.isfinite(float(headroom))
            or float(headroom) != expected_headroom):
        raise RuntimeError("Obi-Wan stage guard-policy limits are impossible")
    if profile_name == "local-macos":
        if slots != 1 or aggregate is not None:
            raise RuntimeError("local stage claims a remote aggregate cgroup")
    else:
        if (aggregate != profile["max_rss_mb"]
                or maximum * slots + floor > aggregate):
            raise RuntimeError("remote stage guard/cgroup budget is impossible")
    return policy


if __name__ == "__main__":
    memory_guard.reexec_under_guard(SCRIPT)


PRINT_PART_SPECS = {
    "core_lm_carrier": {
        "filename": "core_lm_carrier.brep",
        "label": "core_lm_carrier",
        "stl_name": "obiwan_core_1_of_2_lm_carrier",
        "group": "lm",
    },
    "core_um_carrier": {
        "filename": "core_um_carrier.brep",
        "label": "core_um_carrier",
        "stl_name": "obiwan_core_2_of_2_um_carrier",
        "group": "um",
    },
    "optional_lm_keyed_1_of_2_bottom": {
        "filename": "optional_lm_keyed_1_of_2_bottom.brep",
        "label": "optional_lm_keyed_1_of_2_bottom",
        "stl_name": "obiwan_optional_lm_keyed_1_of_2_bottom",
        "group": "lm_split",
    },
    "optional_lm_keyed_2_of_2_top": {
        "filename": "optional_lm_keyed_2_of_2_top.brep",
        "label": "optional_lm_keyed_2_of_2_top",
        "stl_name": "obiwan_optional_lm_keyed_2_of_2_top",
        "group": "lm_split",
    },
    "addon_tweeter_crescent": {
        "filename": "addon_tweeter_crescent.brep",
        "label": "addon_tweeter_crescent",
        "stl_name": "obiwan_addon_tweeter_crescent",
        "group": "tweeter",
    },
}


REVIEW_PART_SPECS = {
    "reference_mu10_body": {
        "filename": "review_reference_mu10_body.brep",
        "label": (
            "REFERENCE_MU10_D98_D80_D60_BODY_TERMINALS_OMITTED_"
            "PHYSICAL_CHECK_REQUIRED"),
        "group": "static",
    },
    "reference_terminal_carrier": {
        "filename": "review_reference_terminal_carrier.brep",
        "label": "REFERENCE_terminal_carrier_proxy_clock_283deg",
        "group": "static",
    },
    "keep_clear_removal_envelope": {
        "filename": "review_keep_clear_removal_envelope.brep",
        "label": "KEEP_CLEAR_Faston_outboard_removal_envelope",
        "group": "static",
    },
    "keep_clear_faston_pull_sweep_1": {
        "filename": "review_keep_clear_faston_pull_sweep_1.brep",
        "label": "KEEP_CLEAR_faston_flag_pull_sweep_1_12mm",
        "group": "static",
    },
    "keep_clear_faston_pull_sweep_2": {
        "filename": "review_keep_clear_faston_pull_sweep_2.brep",
        "label": "KEEP_CLEAR_faston_flag_pull_sweep_2_12mm",
        "group": "static",
    },
    "reference_terminal_tab_1": {
        "filename": "review_reference_terminal_tab_1.brep",
        "label": "REFERENCE_terminal_tab_1",
        "group": "static",
    },
    "reference_faston_receptacle_1": {
        "filename": "review_reference_faston_receptacle_1.brep",
        "label": "REFERENCE_faston_receptacle_1",
        "group": "static",
    },
    "reference_terminal_tab_2": {
        "filename": "review_reference_terminal_tab_2.brep",
        "label": "REFERENCE_terminal_tab_2",
        "group": "static",
    },
    "reference_faston_receptacle_2": {
        "filename": "review_reference_faston_receptacle_2.brep",
        "label": "REFERENCE_faston_receptacle_2",
        "group": "static",
    },
    "keep_clear_faston_boot_1": {
        "filename": "review_keep_clear_faston_boot_1.brep",
        "label": "KEEP_CLEAR_faston_insulation_boot_1",
        "group": "static",
    },
    "keep_clear_faston_boot_2": {
        "filename": "review_keep_clear_faston_boot_2.brep",
        "label": "KEEP_CLEAR_faston_insulation_boot_2",
        "group": "static",
    },
    "reference_um_cable": {
        "filename": "review_reference_um_cable.brep",
        "label": (
            "REFERENCE_UM_D7_LM_printed_cover_then_free_behind_UM_"
            "R15_R20_Faston_handoff"),
        "group": "um_service",
    },
    "reference_y_bundle": {
        "filename": "review_reference_y_bundle.brep",
        "label": "REFERENCE_obiwan_Y_breakout_bundle_heatshrink",
        "group": "um_service",
    },
    "reference_y_terminal_lead_1": {
        "filename": "review_reference_y_terminal_lead_1.brep",
        "label": (
            "REFERENCE_obiwan_Y_breakout_terminal_lead_1_heatshrink"),
        "group": "um_service",
    },
    "reference_y_terminal_lead_2": {
        "filename": "review_reference_y_terminal_lead_2.brep",
        "label": (
            "REFERENCE_obiwan_Y_breakout_terminal_lead_2_heatshrink"),
        "group": "um_service",
    },
    "reference_lm_cable": {
        "filename": "review_reference_lm_cable.brep",
        "label": "REFERENCE_LM_D7p8_short_free_span_no_micro_duct",
        "group": "lm_cable",
    },
    "reference_ts_cable": {
        "filename": "review_reference_ts_cable.brep",
        "label": (
            "REFERENCE_TS_D5p2_LM_UM_printed_then_free_behind_tweeter"),
        "group": "ts_cable",
    },
    "state_hardware": {
        "filename": "review_state_hardware.brep",
        "label": None,
        "group": "hardware",
    },
}


CORE_KEYS = ("core_lm_carrier", "core_um_carrier")
OPTIONAL_LM_SPLIT_KEYS = (
    "optional_lm_keyed_1_of_2_bottom",
    "optional_lm_keyed_2_of_2_top",
)
ATTACHMENT_KEYS_BASE = (
    "addon_tweeter_crescent",
)
REVIEW_KEYS = tuple(REVIEW_PART_SPECS)


def _stand_foot() -> bool:
    text = os.environ.get("LX_STAND_FOOT", "1")
    if text not in {"0", "1"}:
        raise RuntimeError("LX_STAND_FOOT must be 0 or 1")
    return text == "1"


def _require_obiwan_profile() -> None:
    profile = os.environ.get("LX_ROUTING_PROFILE", "obiwan")
    if profile != "obiwan":
        raise RuntimeError(
            "Obi-Wan staging requires LX_ROUTING_PROFILE=obiwan")
    os.environ["LX_ROUTING_PROFILE"] = "obiwan"


def _state_name(stand_foot: bool) -> str:
    return "floor_stand" if stand_foot else "no_floor_stand"


def _large_host_execution() -> bool:
    """Use whole-part construction only in the remote large-host workflow."""
    return (
        os.environ.get("LX_CAD_MEMORY_PROFILE") == "osado-512g"
        and os.environ.get("LX_CAD_ALLOW_PARALLEL") == "1"
    )


_sha256_file = sha256_file


def _source_fingerprint(
        stand_foot: bool, *, runtime_identity: dict | None = None,
        guard_policy: dict | None = None) -> str:
    """Hash every geometry source plus the CAD-kernel/runtime identity."""
    runtime_identity = runtime_identity or _runtime_identity()
    guard_policy = guard_policy or _guard_policy()
    digest = hashlib.sha256()
    digest.update(
        f"lx521-obiwan-release-native-brep-stage-v{SCHEMA_VERSION}\0"
        .encode("ascii"))
    digest.update(b"\0floor=" + (b"1" if stand_foot else b"0"))
    digest.update(b"\0profile=obiwan")
    digest.update(b"\0runtime=")
    digest.update(json.dumps(
        runtime_identity, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    digest.update(b"\0guard-policy=")
    digest.update(json.dumps(
        guard_policy, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    paths = [
        SCRIPT,
        ROOT / "scripts/run_memory_guarded.py",
        *SOURCE_INPUTS,
        *REFERENCE_INPUTS,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"required Obi-Wan stage input is missing: {path}")
        try:
            identity = path.relative_to(ROOT.parent).as_posix()
        except ValueError:
            identity = str(path.resolve())
        digest.update(b"\0" + identity.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_bytes(pretty_json_bytes(payload, allow_nan=True))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_brep_transaction(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 256:
        raise RuntimeError(f"native BREP transaction is truncated: {path}")


def _expected_print_keys(stand_foot: bool) -> tuple[str, ...]:
    keys = [*CORE_KEYS, *OPTIONAL_LM_SPLIT_KEYS]
    keys.extend(ATTACHMENT_KEYS_BASE)
    return tuple(keys)


def _expected_review_keys(stand_foot: bool) -> tuple[str, ...]:
    """State-local review set after deleting separate floor hardware."""
    return tuple(
        key for key in REVIEW_KEYS
        if not (stand_foot and key == "state_hardware")
    )


def _record_for(path: Path, label: str | None, relative_to: Path) -> dict:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "label": label,
    }


def _resolved_record_path(manifest_path: Path, record: dict) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe staged BREP path: {relative}")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    if path.parent != root:
        raise RuntimeError(f"staged BREP escapes manifest directory: {path}")
    return path


def load_stage_manifest(
        path: Path, *, stand_foot: bool | None = None,
        require_active_environment: bool = True,
        require_current_sources: bool = True) -> dict:
    """Validate source/policy/runtime provenance and every staged BREP hash.

    Portable release verification uses the recorded runtime and guard policy
    to recompute provenance. Build/export consumers additionally require those
    records to match their active worker environment. A consumer of staged
    geometry that is provably independent of a changed source may disable the
    broad repository-level source fingerprint; state, identity, transaction,
    byte-count, and per-BREP hash checks remain mandatory.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if stand_foot is None:
        stand_foot = _stand_foot()
        _require_obiwan_profile()
    else:
        stand_foot = bool(stand_foot)
    expected_state = _state_name(stand_foot)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported Obi-Wan stage manifest: {path}")
    if payload.get("state") != expected_state:
        raise RuntimeError(
            f"Obi-Wan stage state {payload.get('state')!r} != {expected_state!r}")
    if payload.get("stand_foot") is not stand_foot:
        raise RuntimeError("Obi-Wan stage stand-foot flag mismatch")
    if payload.get("routing_profile") != "obiwan":
        raise RuntimeError("Obi-Wan stage routing-profile mismatch")
    if payload.get("lm_cutter_group_count") != EXPECTED_LM_CUTTER_GROUP_COUNT:
        raise RuntimeError("Obi-Wan stage cutter-group count mismatch")
    guard_policy = _validate_guard_policy_record(payload.get("guard_policy"))
    runtime_identity = payload.get("runtime_identity")
    if (not isinstance(runtime_identity, dict)
            or set(runtime_identity) != {"python", "packages"}
            or not isinstance(runtime_identity.get("python"), str)
            or not isinstance(runtime_identity.get("packages"), dict)
            or set(runtime_identity["packages"]) != set(
                RUNTIME_DISTRIBUTIONS)
            or not all(isinstance(value, str)
                       for value in runtime_identity["packages"].values())):
        raise RuntimeError("Obi-Wan stage runtime-identity record is malformed")
    source_sha256 = payload.get("source_sha256")
    if (not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in source_sha256)):
        raise RuntimeError("Obi-Wan stage source fingerprint is malformed")
    if require_current_sources:
        expected_source = _source_fingerprint(
            stand_foot, runtime_identity=runtime_identity,
            guard_policy=guard_policy)
        if source_sha256 != expected_source:
            raise RuntimeError(
                "Obi-Wan stage is stale for the current CAD sources")
    if require_active_environment:
        if guard_policy != _guard_policy():
            raise RuntimeError("Obi-Wan stage guard-policy mismatch")
        if runtime_identity != _runtime_identity():
            raise RuntimeError("Obi-Wan stage runtime-identity mismatch")

    expected_sections = {
        "parts": {
            key: PRINT_PART_SPECS[key]
            for key in _expected_print_keys(stand_foot)
        },
        "review_parts": {
            key: REVIEW_PART_SPECS[key]
            for key in _expected_review_keys(stand_foot)
        },
    }
    for section, expected_specs in expected_sections.items():
        records = payload.get(section)
        if (not isinstance(records, dict)
                or set(records) != set(expected_specs)):
            raise RuntimeError(
                f"Obi-Wan stage {section} set mismatch: "
                f"{sorted(records or {})} != {sorted(expected_specs)}")
        for key, record in records.items():
            if (not isinstance(record, dict)
                    or set(record) != {"path", "bytes", "sha256", "label"}):
                raise RuntimeError(f"malformed Obi-Wan stage record: {key}")
            spec = expected_specs[key]
            if (record["path"] != spec["filename"]
                    or record["label"] != spec["label"]
                    or not isinstance(record["bytes"], int)
                    or record["bytes"] < 256
                    or not isinstance(record["sha256"], str)
                    or len(record["sha256"]) != 64
                    or any(character not in "0123456789abcdef"
                           for character in record["sha256"])):
                raise RuntimeError(
                    f"Obi-Wan stage identity record mismatch: {key}")
            staged = _resolved_record_path(path, record)
            _validate_brep_transaction(staged)
            if staged.stat().st_size != record.get("bytes"):
                raise RuntimeError(f"staged BREP size mismatch: {key}")
            if _sha256_file(staged) != record.get("sha256"):
                raise RuntimeError(f"staged BREP hash mismatch: {key}")
    return payload


def staged_part_paths(manifest_path: Path, payload: dict | None = None) -> dict:
    payload = payload or load_stage_manifest(manifest_path)
    return {
        key: _resolved_record_path(Path(manifest_path), record)
        for key, record in payload["parts"].items()
    }


def _wait_for_worker_headroom(label: str) -> None:
    """Wait for the proven launch envelope without changing guard policy."""
    import run_memory_guarded as memory_guard

    if memory_guard.MIN_FREE_MB == 0:
        return
    deadline = time.monotonic() + 60.0
    while True:
        free_mib = memory_guard._free_memory_mib()
        if free_mib is None:
            raise RuntimeError(
                f"cannot measure memory before isolated {label} worker")
        if free_mib >= WORKER_HEADROOM_MIB:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"only {free_mib:.0f} MiB immediately reclaimable; "
                f"refusing to launch isolated {label} worker")
        time.sleep(0.5)


def _run_worker(label: str, arguments: list[str]) -> None:
    _wait_for_worker_headroom(label)
    started = time.monotonic()
    exit_code = 0
    print(f"[obiwan-stage] worker start: {label}", flush=True)
    try:
        subprocess.run(
            [sys.executable, str(SCRIPT), "worker", *arguments],
            check=True,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as exc:
        exit_code = exc.returncode
        raise
    finally:
        print(
            "[obiwan-stage-profile] "
            + json.dumps({
                "schema_version": 1,
                "label": label,
                "wall_seconds": time.monotonic() - started,
                "exit_code": exit_code,
                "stand_foot": _stand_foot(),
            }, sort_keys=True, separators=(",", ":")),
            flush=True,
        )


def _run_large_host_workers(output_dir: Path) -> None:
    """Build independent LM and non-LM stage branches concurrently.

    Profiled Osado runs showed the direct LM branch and the print/review
    branch using only about 3.4 GiB combined peak RSS while running
    sequentially for roughly 343 seconds.  They share no output paths and
    meet only when the manifest is assembled, so let the existing guarded
    process tree execute both branches at once on the large host.
    """
    jobs = (
        ("LM direct full carrier and optional split",
         ["lm-direct", "--output-dir", str(output_dir)]),
        ("all non-LM print/review groups",
         ["groups-direct", "--output-dir", str(output_dir)]),
    )
    with ThreadPoolExecutor(
            max_workers=len(jobs),
            thread_name_prefix="obiwan-stage") as executor:
        futures = [
            executor.submit(_run_worker, label, arguments)
            for label, arguments in jobs
        ]
        for future in futures:
            future.result()


def _require_guarded_worker() -> None:
    memory_guard.require_guarded_build(
        "Obi-Wan stage worker escaped the active memory guard")


def _export_brep_transaction(shape, output: Path,
                             *, require_single_solid: bool) -> None:
    from build123d import export_brep

    solids = list(shape.solids())
    if (not shape.is_valid or not solids
            or any(solid.volume <= 0.01 for solid in solids)
            or (require_single_solid and len(solids) != 1)):
        raise RuntimeError(
            f"invalid staged shape for {output.name}: valid={shape.is_valid} "
            f"volumes={[solid.volume for solid in solids]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.{time.time_ns()}.tmp.brep")
    try:
        if not export_brep(shape, str(temporary)):
            raise RuntimeError(f"failed to export native BREP: {output}")
        _validate_brep_transaction(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _worker_count(output: Path) -> None:
    from lx521_baffle.obiwan.route import route_inner_cutter_group_count

    count = int(route_inner_cutter_group_count("lm"))
    _atomic_json(output, {"lm_cutter_group_count": count})


def _worker_lm_outer(output: Path) -> None:
    from lx521_baffle.obiwan.carriers import lm_carrier_outer_blank

    _export_brep_transaction(
        lm_carrier_outer_blank(), output, require_single_solid=True)


def _worker_lm_cut(input_path: Path, output: Path, index: int) -> None:
    from build123d import import_brep
    from lx521_baffle.obiwan.carriers import apply_lm_route_cutter

    part = apply_lm_route_cutter(import_brep(str(input_path)), index)
    _export_brep_transaction(part, output, require_single_solid=True)


def _worker_lm_finalize(input_path: Path, output: Path) -> None:
    from build123d import import_brep
    from lx521_baffle.obiwan.carriers import finalize_lm_carrier

    part = finalize_lm_carrier(
        import_brep(str(input_path)), routes_already_cut=True)
    _export_brep_transaction(part, output, require_single_solid=True)


def _worker_lm_direct(output_dir: Path) -> None:
    """Build one final LM, then derive both optional split prints from it."""
    if not _large_host_execution():
        raise RuntimeError(
            "direct LM construction requires the osado large-host workflow")
    from lx521_baffle.obiwan.carriers import lm_carrier
    from lx521_baffle.obiwan.lm_split import lm_carrier_split_parts

    lm = lm_carrier()
    _export_brep_transaction(
        lm, output_dir / PRINT_PART_SPECS["core_lm_carrier"]["filename"],
        require_single_solid=True)
    split_parts = lm_carrier_split_parts(lm)
    if set(split_parts) != set(OPTIONAL_LM_SPLIT_KEYS):
        raise RuntimeError(
            "optional LM split part set mismatch: "
            f"{sorted(split_parts)} != {sorted(OPTIONAL_LM_SPLIT_KEYS)}")
    for key, shape in split_parts.items():
        _export_brep_transaction(
            shape, output_dir / PRINT_PART_SPECS[key]["filename"],
            require_single_solid=True)


def _worker_lm_split(input_path: Path, output_dir: Path) -> None:
    """Derive the optional pair from an already-finalized native LM BREP."""
    from build123d import import_brep
    from lx521_baffle.obiwan.lm_split import lm_carrier_split_parts

    parts = lm_carrier_split_parts(import_brep(str(input_path)))
    if set(parts) != set(OPTIONAL_LM_SPLIT_KEYS):
        raise RuntimeError(
            "optional LM split part set mismatch: "
            f"{sorted(parts)} != {sorted(OPTIONAL_LM_SPLIT_KEYS)}")
    for key, shape in parts.items():
        _export_brep_transaction(
            shape, output_dir / PRINT_PART_SPECS[key]["filename"],
            require_single_solid=True)


def _worker_print_group(group: str, output_dir: Path) -> None:
    if group == "um":
        from lx521_baffle.obiwan.carriers import um_carrier
        parts = {"core_um_carrier": um_carrier()}
    elif group == "tweeter":
        from lx521_baffle.obiwan.attachments import tweeter_crescent
        parts = {"addon_tweeter_crescent": tweeter_crescent()}
    else:
        raise RuntimeError(f"unknown Obi-Wan print group: {group}")

    expected = {
        key for key, spec in PRINT_PART_SPECS.items()
        if spec["group"] == group
    }
    if set(parts) != expected:
        raise RuntimeError(
            f"Obi-Wan {group} part set {sorted(parts)} != {sorted(expected)}")
    for key, shape in parts.items():
        _export_brep_transaction(
            shape, output_dir / PRINT_PART_SPECS[key]["filename"],
            require_single_solid=True)


def _worker_review_group(group: str, output_dir: Path) -> None:
    if group == "static":
        from lx521_baffle.um_fit import (
            faston_boot_proxy_parts,
            faston_pull_sweep_parts,
            faston_proxy_parts,
            mu10_body_keepout,
            removal_envelope,
            terminal_carrier_proxy,
        )
        fastons = faston_proxy_parts()
        boots = faston_boot_proxy_parts()
        pull_sweeps = faston_pull_sweep_parts()
        parts = {
            "reference_mu10_body": mu10_body_keepout(include_flange=True),
            "reference_terminal_carrier": terminal_carrier_proxy(),
            "keep_clear_removal_envelope": removal_envelope(),
            "keep_clear_faston_pull_sweep_1": (
                pull_sweeps["faston_flag_pull_sweep_1"]),
            "keep_clear_faston_pull_sweep_2": (
                pull_sweeps["faston_flag_pull_sweep_2"]),
            "reference_terminal_tab_1": fastons["terminal_tab_1"],
            "reference_faston_receptacle_1": (
                fastons["faston_receptacle_1"]),
            "reference_terminal_tab_2": fastons["terminal_tab_2"],
            "reference_faston_receptacle_2": (
                fastons["faston_receptacle_2"]),
            "keep_clear_faston_boot_1": boots["faston_insulation_boot_1"],
            "keep_clear_faston_boot_2": boots["faston_insulation_boot_2"],
        }
    elif group == "um_service":
        from lx521_baffle.um_fit import (
            rear_cable_envelope,
            obiwan_y_breakout_boot_parts,
        )
        y_parts = obiwan_y_breakout_boot_parts()
        parts = {
            "reference_um_cable": rear_cable_envelope("obiwan"),
            "reference_y_bundle": (
                y_parts["obiwan_Y_breakout_bundle_heatshrink"]),
            "reference_y_terminal_lead_1": (
                y_parts["obiwan_Y_breakout_terminal_lead_1_heatshrink"]),
            "reference_y_terminal_lead_2": (
                y_parts["obiwan_Y_breakout_terminal_lead_2_heatshrink"]),
        }
    elif group == "lm_cable":
        from lx521_baffle.um_fit import obiwan_lm_cable_envelope
        parts = {"reference_lm_cable": obiwan_lm_cable_envelope()}
    elif group == "ts_cable":
        from lx521_baffle.um_fit import obiwan_ts_cable_envelope
        parts = {"reference_ts_cable": obiwan_ts_cable_envelope()}
    elif group == "hardware":
        if _stand_foot():
            raise RuntimeError(
                "integral floor Obi-Wan has no separate support hardware")
        from lx521_baffle.obiwan.bridge import (
            bridge_fastener_head_envelopes,
        )
        hardware = bridge_fastener_head_envelopes()
        parts = {"state_hardware": hardware}
    else:
        raise RuntimeError(f"unknown Obi-Wan review group: {group}")

    expected = {
        key for key, spec in REVIEW_PART_SPECS.items()
        if spec["group"] == group
    }
    if set(parts) != expected:
        raise RuntimeError(
            f"Obi-Wan {group} review set {sorted(parts)} != {sorted(expected)}")
    for key, shape in parts.items():
        _export_brep_transaction(
            shape, output_dir / REVIEW_PART_SPECS[key]["filename"],
            require_single_solid=False)


def _worker_all_groups_direct(output_dir: Path) -> None:
    """Build all non-LM staged parts in one large-host OCC lifecycle."""
    if not _large_host_execution():
        raise RuntimeError(
            "whole-stage group construction requires osado large-host mode")
    groups = ["um", "tweeter"]
    for group in groups:
        _worker_print_group(group, output_dir)
    review_groups = ["static", "um_service", "lm_cable", "ts_cable"]
    if not _stand_foot():
        review_groups.append("hardware")
    for group in review_groups:
        _worker_review_group(group, output_dir)


def _worker_step(manifest_path: Path, kind: str, output: Path) -> None:
    from build123d import Compound, export_step, import_brep

    payload = load_stage_manifest(manifest_path)
    stand_foot = bool(payload["stand_foot"])
    part_records = payload["parts"]
    review_records = payload["review_parts"]

    attachment_keys = list(ATTACHMENT_KEYS_BASE)
    if kind == "split":
        keys = list(CORE_KEYS)
        root_label = (
            "lx521_obiwan_r6f_core_2piece_floor" if stand_foot else
            "lx521_obiwan_r6f_core_2piece_no_floor_fused_solid_web")
    elif kind == "lm_split":
        keys = list(OPTIONAL_LM_SPLIT_KEYS)
        root_label = (
            "lx521_obiwan_r6f_optional_lm_keyed_split_floor"
            if stand_foot else
            "lx521_obiwan_r6f_optional_lm_keyed_split_no_floor")
    elif kind == "attachments":
        keys = attachment_keys
        root_label = (
            "lx521_obiwan_r6f_optional_addons_floor_integrated_mount"
            if stand_foot else
            "lx521_obiwan_r6f_optional_addons_no_floor")
    elif kind == "assembled":
        keys = [*CORE_KEYS, *attachment_keys]
        root_label = (
            "lx521_obiwan_r6f_assembled_floor" if stand_foot else
            "lx521_obiwan_r6f_assembled_no_floor_fused_solid_web")
    else:
        raise RuntimeError(f"unknown Obi-Wan STEP kind: {kind}")

    children = []
    for key in keys:
        record = part_records[key]
        shape = import_brep(str(_resolved_record_path(
            manifest_path, record)))
        shape.label = record["label"]
        children.append(shape)
    if kind == "assembled":
        for key in _expected_review_keys(stand_foot):
            record = review_records[key]
            shape = import_brep(str(_resolved_record_path(
                manifest_path, record)))
            label = record["label"]
            if key == "state_hardware":
                label = "KEEP_CLEAR_four_stock_bridge_M5_heads"
            shape.label = label
            children.append(shape)

    assembly = Compound(children=children)
    assembly.label = root_label
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.{time.time_ns()}.tmp.step")
    try:
        export_step(assembly, str(temporary), timestamp=FIXED_TIMESTAMP)
        from export_steps import validate_step_transaction
        validate_step_transaction(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[obiwan-stage] wrote {kind} STEP {output}", flush=True)


def _stage(manifest_path: Path) -> None:
    stand_foot = _stand_foot()
    _require_obiwan_profile()
    runtime_identity = _runtime_identity()
    guard_policy = _guard_policy()
    source_sha256 = _source_fingerprint(
        stand_foot, runtime_identity=runtime_identity,
        guard_policy=guard_policy)
    if manifest_path.is_file():
        try:
            payload = load_stage_manifest(manifest_path)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                json.JSONDecodeError):
            pass
        else:
            if payload["source_sha256"] == source_sha256:
                print(f"[obiwan-stage] reused {manifest_path}", flush=True)
                return

    stage_dir = manifest_path.parent
    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f".{stage_dir.name}.build-", dir=stage_dir.parent) as tmp:
        temporary_dir = Path(tmp)
        count_path = temporary_dir / "cutter_count.json"
        _run_worker("LM cutter-count", [
            "count", "--output", str(count_path)])
        count = json.loads(count_path.read_text(
            encoding="utf-8"))["lm_cutter_group_count"]
        if count != EXPECTED_LM_CUTTER_GROUP_COUNT:
            raise RuntimeError(
                f"Obi-Wan exporter expects {EXPECTED_LM_CUTTER_GROUP_COUNT} "
                f"LM cutter groups, geometry reports {count}")

        lm_final = temporary_dir / PRINT_PART_SPECS[
            "core_lm_carrier"]["filename"]
        if _large_host_execution():
            # The large remote host does not need the macOS-era 25-process
            # outer/cutter/finalize transaction chain.  Keep the exact group
            # count above as a release contract while constructing the same
            # final carrier in one OCC address space.  The independent LM and
            # non-LM branches run concurrently and join before manifesting.
            print(
                "[obiwan-stage] execution=parallel direct-full (osado)",
                flush=True)
            _run_large_host_workers(temporary_dir)
        else:
            print(
                "[obiwan-stage] LM execution=segmented (local)",
                flush=True)
            previous = temporary_dir / "lm_outer.brep"
            _run_worker("LM outer", [
                "lm-outer", "--output", str(previous)])
            for index in range(count):
                current = temporary_dir / f"lm_cut_{index:02d}.brep"
                _run_worker(f"LM cutter {index + 1}/{count}", [
                    "lm-cut", "--input", str(previous),
                    "--output", str(current), "--index", str(index)])
                previous.unlink()
                previous = current
            _run_worker("LM finalization", [
                "lm-final", "--input", str(previous),
                "--output", str(lm_final)])
            previous.unlink()
            _run_worker("optional LM keyed split", [
                "lm-split", "--input", str(lm_final),
                "--output-dir", str(temporary_dir)])

        if not _large_host_execution():
            print_groups = ["um", "tweeter"]
            for group in print_groups:
                _run_worker(f"print group {group}", [
                    "print-group", "--group", group,
                    "--output-dir", str(temporary_dir)])
            review_groups = [
                "static", "um_service", "lm_cable", "ts_cable"]
            if not stand_foot:
                review_groups.append("hardware")
            for group in review_groups:
                _run_worker(f"review group {group}", [
                    "review-group", "--group", group,
                    "--output-dir", str(temporary_dir)])

        print_records = {}
        for key in _expected_print_keys(stand_foot):
            spec = PRINT_PART_SPECS[key]
            source = temporary_dir / spec["filename"]
            _validate_brep_transaction(source)
            print_records[key] = _record_for(
                source, spec["label"], temporary_dir)
        review_records = {}
        for key in _expected_review_keys(stand_foot):
            spec = REVIEW_PART_SPECS[key]
            source = temporary_dir / spec["filename"]
            _validate_brep_transaction(source)
            review_records[key] = _record_for(
                source, spec["label"], temporary_dir)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "state": _state_name(stand_foot),
            "stand_foot": stand_foot,
            "routing_profile": "obiwan",
            "source_sha256": source_sha256,
            "lm_cutter_group_count": count,
            "guard_policy": guard_policy,
            "runtime_identity": runtime_identity,
            "parts": print_records,
            "review_parts": review_records,
        }

        stage_dir.mkdir(parents=True, exist_ok=True)
        expected_filenames = {
            record["path"] for section in (print_records, review_records)
            for record in section.values()
        }
        for filename in sorted(expected_filenames):
            (temporary_dir / filename).replace(stage_dir / filename)
        _atomic_json(manifest_path, payload)
        for stale in stage_dir.glob("*.brep"):
            if stale.name not in expected_filenames:
                stale.unlink()
    load_stage_manifest(manifest_path)
    print(f"[obiwan-stage] wrote {manifest_path}", flush=True)


def _step(manifest_path: Path, kind: str, output: Path) -> None:
    load_stage_manifest(manifest_path)
    _run_worker(f"{kind} STEP assembly", [
        "step", "--manifest", str(manifest_path),
        "--kind", kind, "--output", str(output)])


def _worker(args) -> None:
    _require_guarded_worker()
    _require_obiwan_profile()
    if args.worker_kind == "count":
        _worker_count(args.output)
    elif args.worker_kind == "lm-outer":
        _worker_lm_outer(args.output)
    elif args.worker_kind == "lm-cut":
        _worker_lm_cut(args.input, args.output, args.index)
    elif args.worker_kind == "lm-final":
        _worker_lm_finalize(args.input, args.output)
    elif args.worker_kind == "lm-direct":
        _worker_lm_direct(args.output_dir)
    elif args.worker_kind == "lm-split":
        _worker_lm_split(args.input, args.output_dir)
    elif args.worker_kind == "print-group":
        _worker_print_group(args.group, args.output_dir)
    elif args.worker_kind == "review-group":
        _worker_review_group(args.group, args.output_dir)
    elif args.worker_kind == "groups-direct":
        _worker_all_groups_direct(args.output_dir)
    elif args.worker_kind == "step":
        _worker_step(args.manifest, args.kind, args.output)
    else:
        raise AssertionError(args.worker_kind)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--manifest", required=True, type=Path)
    step = commands.add_parser("step")
    step.add_argument("--manifest", required=True, type=Path)
    step.add_argument(
        "--kind", required=True,
        choices=("split", "lm_split", "attachments", "assembled"))
    step.add_argument("--output", required=True, type=Path)

    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    workers = worker.add_subparsers(dest="worker_kind", required=True)
    count = workers.add_parser("count", help=argparse.SUPPRESS)
    count.add_argument("--output", required=True, type=Path)
    lm_outer = workers.add_parser("lm-outer", help=argparse.SUPPRESS)
    lm_outer.add_argument("--output", required=True, type=Path)
    lm_cut = workers.add_parser("lm-cut", help=argparse.SUPPRESS)
    lm_cut.add_argument("--input", required=True, type=Path)
    lm_cut.add_argument("--output", required=True, type=Path)
    lm_cut.add_argument("--index", required=True, type=int)
    lm_final = workers.add_parser("lm-final", help=argparse.SUPPRESS)
    lm_final.add_argument("--input", required=True, type=Path)
    lm_final.add_argument("--output", required=True, type=Path)
    lm_direct = workers.add_parser("lm-direct", help=argparse.SUPPRESS)
    lm_direct.add_argument("--output-dir", required=True, type=Path)
    lm_split = workers.add_parser("lm-split", help=argparse.SUPPRESS)
    lm_split.add_argument("--input", required=True, type=Path)
    lm_split.add_argument("--output-dir", required=True, type=Path)
    print_group = workers.add_parser(
        "print-group", help=argparse.SUPPRESS)
    print_group.add_argument(
        "--group", required=True,
        choices=("um", "tweeter"))
    print_group.add_argument("--output-dir", required=True, type=Path)
    review_group = workers.add_parser(
        "review-group", help=argparse.SUPPRESS)
    review_group.add_argument(
        "--group", required=True,
        choices=("static", "um_service", "lm_cable", "ts_cable",
                 "hardware"))
    review_group.add_argument("--output-dir", required=True, type=Path)
    groups_direct = workers.add_parser(
        "groups-direct", help=argparse.SUPPRESS)
    groups_direct.add_argument("--output-dir", required=True, type=Path)
    step_worker = workers.add_parser("step", help=argparse.SUPPRESS)
    step_worker.add_argument("--manifest", required=True, type=Path)
    step_worker.add_argument(
        "--kind", required=True,
        choices=("split", "lm_split", "attachments", "assembled"))
    step_worker.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "stage":
        _stage(args.manifest)
    elif args.command == "step":
        _step(args.manifest, args.kind, args.output)
    elif args.command == "worker":
        _worker(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
