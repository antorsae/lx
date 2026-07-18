#!/usr/bin/env python3
"""Slice and audit every released pause-and-bury magnet STL.

This is deliberately an offline, non-printer-contact pipeline.  It invokes
only Bambu Studio's local slicer CLI, parses the resulting plain G-code, and
writes hash-backed JSON/CSV/Markdown evidence.  It cannot upload or start a
print.

The CAD producer owns ``captive_magnet_release_catalog.json``.  In particular,
the catalog records magnet stations *after* the exact front-face-down STL
export transform.  This consumer therefore never imports build123d/OCC and is
safe to run on the macOS slicing host.

Pause markers are measured from the actual G-code layer schedule.  For each
site, local cavity toolpaths identify the final fully open layer and the first
boundary-contraction/interior-deposition layer.  The CAD bury plane is only a
bounded consistency datum; CAD heights alone are never emitted as pause
instructions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

from bambu_3mf_audit import (
    Bambu3MFAuditError,
    Matrix4 as BambuMatrix4,
    audit_bambu_3mf,
    transform_point as transform_bambu_point,
    transform_vector as transform_bambu_vector,
    validate_bed_fit as validate_bambu_bed_fit,
    validate_result_bbox as validate_bambu_result_bbox,
)

from front_down_contract import (
    FrontDownContractError,
    RELEASE_ACOUSTIC_PRINT_CONTRACT,
    validate_front_down_transform,
    validate_print_sidecar,
)
from json_schema_subset import (
    JsonSchemaSubsetError,
    validate_json_schema,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = SCRIPT_DIR / "review" / "captive_magnet_release_catalog.json"
CATALOG_SCHEMA = SCRIPT_DIR / "captive_magnet_release_catalog.schema.json"
DEFAULT_PROFILE = SCRIPT_DIR / "captive_magnet_slicing_profile.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "review" / "captive_magnet_slice_audit"
SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
OVERSIZE_COVERED_STATUS = (
    "not_p2s_printable__cavity_covered_by_exact_split")
EXPECTED_RELEASE_ARTIFACT_COUNT = 56
EXPECTED_RELEASE_MAGNET_COUNT = 102
CANONICAL_MANIFEST_FILENAMES = (
    "captive_magnet_pause_manifest.json",
    "captive_magnet_pause_manifest.csv",
    "CAPTIVE_MAGNET_PAUSE_MANIFEST.md",
)
PLACED_3MF_FILENAME = "audited_slice_project.3mf"
AUDIT_SOURCE_FILES = (
    Path(__file__).resolve(),
    (SCRIPT_DIR / "bambu_3mf_audit.py").resolve(),
    (SCRIPT_DIR / "front_down_contract.py").resolve(),
    (SCRIPT_DIR / "json_schema_subset.py").resolve(),
)
PRINT_INSERTION_DIRECTION_XYZ = (0.0, 0.0, -1.0)
PRINT_INSERTION_INSTRUCTION = (
    "insert vertically downward from above the paused part (+Z side) "
    "through the open loading chimney along print -Z"
)

FLOAT_EPS = 1.0e-6
LAYER_EPS = 2.0e-4
SITE_SAMPLE_STEP_MM = 0.10
ARC_TESSELLATION_STEP_MM = 0.20
MAX_ARC_TESSELLATION_SEGMENTS = 8192
ARC_RADIUS_TOLERANCE_MM = 0.05
LAST_OPEN_INTERIOR_PATH_LIMIT_MM = 0.20
CLOSING_BOUNDARY_INSET_MM = 0.03
CLOSING_BOUNDARY_REOPEN_TOLERANCE_MM = 0.03
FALLBACK_LINE_WIDTH_MM = 0.45
# A physically continuous 0.42-mm Classic bead can tolerate a small gap
# between sampled segment centrelines.  Anything larger than one bead plus
# the two 0.05-mm half-sample offsets is a real break, not G-code segmentation.
RETAINING_PATH_CONNECTIVITY_GAP_MM = 0.52
EVIDENCE_CELL_PX = 218
EVIDENCE_MARGIN_MM = 4.0
RELEASE_SITE_GEOMETRY_MM = {
    "magnet_diameter_mm": 5.0,
    "magnet_depth_mm": 2.0,
    "cavity_diameter_mm": 5.20,
    "cavity_depth_mm": 2.10,
    "face_skin_mm": 0.45,
    "inner_skin_mm": 0.45,
    "captive_land_mm": 3.00,
    "interface_gap_mm": 0.05,
    "paired_magnet_face_separation_mm": 0.95,
    "roof_angle_deg": 45.0,
    "classic_retaining_path_mm": 0.42,
}


class AuditError(RuntimeError):
    """A release-blocking catalog, profile, slice, or toolpath error."""


def _canonical_json(data: Any) -> bytes:
    return (json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        data, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label} must be finite, got {value!r}")
    return result


def _vec3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AuditError(f"{label} must contain three numbers")
    return tuple(_float(item, label) for item in value)  # type: ignore[return-value]


def _unit_xy(value: Sequence[float], label: str) -> tuple[float, float]:
    x, y = float(value[0]), float(value[1])
    length = math.hypot(x, y)
    if length <= 1.0e-8:
        raise AuditError(f"{label} has no XY projection")
    return x / length, y / length


def _unit3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    x, y, z = _vec3(value, label)
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1.0e-8:
        raise AuditError(f"{label} is zero length")
    return x / length, y / length, z / length


def _lookup(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    """Return one catalog field without manufacturing a safety default."""
    if key not in mapping:
        raise AuditError(f"{label} is required")
    return mapping[key]


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return clean.strip("._-") or "artifact"


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


class PresetResolver:
    """Resolve Bambu system ``inherits``/``include`` chains exactly once."""

    def __init__(self, vendor_root: Path):
        self.vendor_root = vendor_root.resolve()
        if not self.vendor_root.is_dir():
            raise AuditError(f"Bambu preset root does not exist: {vendor_root}")
        self.by_name: dict[str, list[Path]] = {}
        self.raw: dict[Path, dict[str, Any]] = {}
        for path in sorted(self.vendor_root.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            self.raw[path.resolve()] = data
            name = data.get("name")
            if isinstance(name, str) and name:
                self.by_name.setdefault(name, []).append(path.resolve())
        self.dependencies: set[Path] = set()

    def _find_named(self, name: str, type_hint: str | None, origin: Path) -> Path:
        candidates = list(self.by_name.get(name, ()))
        if type_hint:
            typed = [p for p in candidates
                     if self.raw[p].get("type") in (None, type_hint)]
            if typed:
                candidates = typed
        same_dir = [p for p in candidates if p.parent == origin.parent]
        if len(same_dir) == 1:
            return same_dir[0]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise AuditError(
                f"Bambu preset dependency {name!r} referenced by {origin} "
                f"was not found below {self.vendor_root}")
        raise AuditError(
            f"ambiguous Bambu preset dependency {name!r}: "
            + ", ".join(str(path) for path in candidates))

    def resolve(self, path: Path) -> dict[str, Any]:
        return self._resolve(path.resolve(), ())

    def _resolve(self, path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        if path in stack:
            chain = " -> ".join(item.name for item in (*stack, path))
            raise AuditError(f"Bambu preset inheritance cycle: {chain}")
        try:
            child = self.raw[path]
        except KeyError as exc:
            raise AuditError(f"unindexed Bambu preset: {path}") from exc
        self.dependencies.add(path)
        type_hint = child.get("type")
        merged: dict[str, Any] = {}
        parent = child.get("inherits")
        if isinstance(parent, str) and parent.strip():
            parent_path = self._find_named(parent.strip(), type_hint, path)
            merged.update(self._resolve(parent_path, (*stack, path)))
        includes = child.get("include", [])
        if isinstance(includes, str):
            includes = [includes]
        if includes is None:
            includes = []
        if not isinstance(includes, list):
            raise AuditError(f"invalid include list in {path}: {includes!r}")
        for name in includes:
            if not isinstance(name, str) or not name:
                raise AuditError(f"invalid include name in {path}: {name!r}")
            include_path = self._find_named(name, type_hint, path)
            merged.update(self._resolve(include_path, (*stack, path)))
        merged.update({key: value for key, value in child.items()
                       if key not in ("inherits", "include")})
        return merged


def _default_bambu_system_root(vendor: str) -> Path:
    return (Path.home() / "Library" / "Application Support" / "BambuStudio"
            / "system" / vendor)


def _find_bambu_binary(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("BAMBU_STUDIO_BIN")
    if env:
        candidates.append(Path(env).expanduser())
    found = shutil.which("bambu-studio") or shutil.which("BambuStudio")
    if found:
        candidates.append(Path(found))
    candidates.extend((
        Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio"),
        Path("/Applications/Bambu Studio.app/Contents/MacOS/BambuStudio"),
    ))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise AuditError("Bambu Studio CLI was not found; set BAMBU_STUDIO_BIN")


def _scalar(profile: Mapping[str, Any], key: str, label: str) -> float:
    value = profile.get(key)
    if isinstance(value, list):
        if not value:
            raise AuditError(f"resolved {label}.{key} is empty")
        value = value[0]
    return _float(value, f"resolved {label}.{key}")


def _boolish(value: Any) -> bool:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "enabled")
    return False


def prepare_profiles(
    config_path: Path,
    output_dir: Path,
    *,
    system_root: Path | None,
    bambu_binary: Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1:
        raise AuditError(f"unsupported slicing profile schema in {config_path}")
    root = (system_root or _default_bambu_system_root(config["vendor"])).resolve()
    resolver = PresetResolver(root)
    sources = {
        "machine": root / config["machine_preset"],
        "process": root / config["process_preset"],
        "filament": root / config["filament_preset"],
    }
    for label, path in sources.items():
        if path.resolve() not in resolver.raw:
            raise AuditError(f"configured {label} preset was not found: {path}")
    resolved = {label: resolver.resolve(path) for label, path in sources.items()}
    req = config["requirements"]
    checks = {
        "nozzle diameter": (
            _scalar(resolved["machine"], "nozzle_diameter", "machine"),
            _float(req["nozzle_diameter_mm"], "required nozzle diameter")),
        "layer height": (
            _scalar(resolved["process"], "layer_height", "process"),
            _float(req["layer_height_mm"], "required layer height")),
        "first layer height": (
            _scalar(resolved["process"], "initial_layer_print_height", "process"),
            _float(req["first_layer_height_mm"], "required first layer height")),
        "outer wall width": (
            _scalar(resolved["process"], "outer_wall_line_width", "process"),
            _float(req["outer_wall_line_width_mm"], "required outer wall width")),
        "inner wall width": (
            _scalar(resolved["process"], "inner_wall_line_width", "process"),
            _float(req["inner_wall_line_width_mm"], "required inner wall width")),
    }
    for label, (actual, expected) in checks.items():
        if not math.isclose(actual, expected, abs_tol=1.0e-8):
            raise AuditError(
                f"resolved Bambu {label} {actual:g} != required {expected:g}")
    wall_generator = str(resolved["process"].get("wall_generator", "")).lower()
    if wall_generator != str(req["wall_generator"]).lower():
        raise AuditError(
            f"resolved wall generator {wall_generator!r} is not Classic")
    if _boolish(resolved["process"].get("enable_support")):
        raise AuditError("resolved process enables support; captive cavities forbid it")
    model = resolved["machine"].get("printer_model")
    if model != req["printer_model"]:
        raise AuditError(f"resolved printer model {model!r} is not {req['printer_model']!r}")

    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, data in resolved.items():
        path = profile_dir / f"resolved_{label}.json"
        _write_json(path, data)
        paths[label] = path
    dependency_records = []
    for path in sorted(resolver.dependencies):
        dependency_records.append({
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        })
    try:
        version_run = subprocess.run(
            [str(bambu_binary), "--help"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30, check=False)
        version_line = version_run.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuditError(f"cannot execute Bambu Studio CLI: {exc}") from exc
    audit_sources = {
        str(path): sha256_file(path) for path in AUDIT_SOURCE_FILES
    }
    identity = {
        "backend": "BambuStudio",
        "binary": str(bambu_binary),
        "binary_sha256": sha256_file(bambu_binary),
        "version": version_line,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "system_vendor_root": str(root),
        "source_presets": {
            key: {
                "path": str(value.resolve()),
                "sha256": sha256_file(value),
            } for key, value in sources.items()
        },
        "resolution_dependencies": dependency_records,
        "resolved_profiles": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "effective": {
            "printer_model": model,
            "nozzle_diameter_mm": checks["nozzle diameter"][0],
            "layer_height_mm": checks["layer height"][0],
            "first_layer_height_mm": checks["first layer height"][0],
            "wall_generator": wall_generator,
            "outer_wall_line_width_mm": checks["outer wall width"][0],
            "inner_wall_line_width_mm": checks["inner wall width"][0],
            "support_enabled": False,
            "arc_fitting_enabled": _boolish(
                resolved["process"].get("enable_arc_fitting")),
            "filament": resolved["filament"].get("name"),
        },
        "machine_bounds_mm": config["machine_bounds_mm"],
        "audit_sources": audit_sources,
    }
    identity["profile_set_sha256"] = _sha256_bytes(_canonical_json({
        key: record["sha256"]
        for key, record in identity["resolved_profiles"].items()
    }))
    _write_json(profile_dir / "profile_provenance.json", identity)
    return {
        "config": config,
        "identity": identity,
        "paths": paths,
        "resolved": resolved,
        # Keep the original scalar for existing diagnostic consumers, while
        # binding every pure-Python authority used by the slice audit.
        "audit_script_sha256": audit_sources[str(Path(__file__).resolve())],
        "audit_source_sha256": audit_sources,
    }


@dataclasses.dataclass(frozen=True)
class MeshFacts:
    triangle_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.bounds_min, self.bounds_max))  # type: ignore[return-value]


def inspect_stl(path: Path) -> MeshFacts:
    """Read binary or ASCII STL bounds without trimesh/OCC."""
    data = path.read_bytes()
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    count = 0
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + n * 50:
            offset = 84
            for _ in range(n):
                values = struct.unpack_from("<9f", data, offset + 12)
                for index in range(0, 9, 3):
                    for axis in range(3):
                        value = float(values[index + axis])
                        if not math.isfinite(value):
                            raise AuditError(f"non-finite vertex in {path}")
                        mins[axis] = min(mins[axis], value)
                        maxs[axis] = max(maxs[axis], value)
                count += 1
                offset += 50
            return MeshFacts(count, tuple(mins), tuple(maxs))  # type: ignore[arg-type]
    text = data.decode("ascii", errors="ignore")
    for match in re.finditer(
            r"(?im)^\s*vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            text):
        values = tuple(float(match.group(i)) for i in range(1, 4))
        for axis, value in enumerate(values):
            if not math.isfinite(value):
                raise AuditError(f"non-finite vertex in {path}")
            mins[axis] = min(mins[axis], value)
            maxs[axis] = max(maxs[axis], value)
        count += 1
    if count == 0 or count % 3:
        raise AuditError(f"{path} is not a recognized STL")
    return MeshFacts(count // 3, tuple(mins), tuple(maxs))  # type: ignore[arg-type]


def _matrix4(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    if (not isinstance(value, list) or len(value) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in value)):
        raise AuditError(f"{label} must be a 4x4 matrix")
    return tuple(tuple(_float(v, label) for v in row) for row in value)


def _transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> tuple[float, float, float]:
    result = []
    for row in matrix[:3]:
        result.append(sum(row[i] * point[i] for i in range(3)) + row[3])
    return tuple(result)  # type: ignore[return-value]


def _transform_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    result = []
    for row in matrix[:3]:
        result.append(sum(row[i] * vector[i] for i in range(3)))
    return tuple(result)  # type: ignore[return-value]


def _site_in_bambu_bed_space(
    site: Mapping[str, Any], matrix: BambuMatrix4,
) -> dict[str, Any]:
    """Apply Bambu's audited Rz+XY arrangement to one print-space station.

    The catalog remains authoritative in exported-STL coordinates.  Bambu may
    rotate and center that front-down STL in the bed plane, so all G-code ROIs,
    radial axes, and evidence overlays must use the full archived affine—not a
    translation inferred from an axis-aligned bounding box.
    """
    transformed = dict(site)
    original: dict[str, list[float]] = {}
    for key in (
            "print_cavity_center_xyz_mm",
            "print_seated_magnet_center_xyz_mm",
            "print_actual_face_xyz_mm"):
        if key not in site:
            continue
        value = _vec3(site[key], key)
        original[key] = list(value)
        transformed[key] = transform_bambu_point(matrix, value)
    for key in (
            "print_material_inward_xyz",
            "print_marked_pole_axis_xyz",
            "print_insertion_direction_xyz"):
        if key not in site:
            continue
        value = _vec3(site[key], key)
        original[key] = list(value)
        transformed[key] = transform_bambu_vector(matrix, value)
    transformed["catalog_stl_print_space"] = original
    return transformed


def _site_mapping(site: Mapping[str, Any]) -> dict[str, Any]:
    facts = site.get("facts")
    if facts is None:
        facts = {}
    if not isinstance(facts, dict):
        raise AuditError("site.facts must be an object")
    merged = dict(facts)
    merged.update(site)
    return merged


def _source_site_contract(site: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical exact source-space contract used for split equivalence."""
    closure = str(site.get("closure_kind", ""))
    contract: dict[str, Any] = {
        "name": str(site.get("name", "")),
        "closure_kind": closure,
        "cavity_center_xyz_mm": list(_vec3(
            _lookup(site, "cavity_center_xyz_mm", "cavity_center_xyz"),
            "source cavity center")),
        "seated_magnet_center_xyz_mm": list(_vec3(
            _lookup(
                site, "seated_magnet_center_xyz_mm",
                "seated_magnet_center_xyz"),
            "source seated magnet center")),
        "marked_pole_axis_xyz": list(_vec3(
            site["marked_pole_axis_xyz"], "source marked pole axis")),
        "insertion_direction_xyz": list(_vec3(_required(
            site, "insertion_direction_xyz",
            "source insertion direction"), "source insertion direction")),
        "installed_marked_pole_axis_xyz": list(_vec3(_required(
            site, "installed_marked_pole_axis_xyz",
            "installed marked pole axis"), "installed marked pole axis")),
        "cavity_bury_roof_start_print_z_mm": _float(_lookup(
            site, "cavity_bury_roof_start_print_z_mm",
            "roof_start_print_z_mm", "bury_plane_print_z_mm"),
            "source bury plane"),
        "roof_apex_print_z_mm": _float(
            site["roof_apex_print_z_mm"], "source roof apex"),
        "magnet_diameter_mm": _float(_required(
            site, "magnet_diameter_mm", "source magnet diameter"),
            "source magnet diameter"),
        "magnet_depth_mm": _float(_required(
            site, "magnet_depth_mm", "source magnet depth"),
            "source magnet depth"),
        "cavity_diameter_mm": _float(_required(
            site, "cavity_diameter_mm", "source cavity diameter"),
            "source cavity diameter"),
        "cavity_depth_mm": _float(_required(
            site, "cavity_depth_mm", "source cavity depth"),
            "source cavity depth"),
        "face_skin_mm": _float(_required(
            site, "face_skin_mm", "source face skin"), "source face skin"),
        "inner_skin_mm": _float(_required(
            site, "inner_skin_mm", "source inner skin"),
            "source inner skin"),
        "roof_angle_deg": _float(_required(
            site, "roof_angle_deg", "source roof angle"),
            "source roof angle"),
        "polarity_instruction": str(_required(
            site, "polarity_instruction", "source polarity instruction")),
        "magnet_count": int(_float(_required(
            site, "magnet_count", "source magnet count"),
            "source magnet count")),
        "structural_load_credit_n": _float(_required(
            site, "structural_load_credit_n",
            "source structural load credit"),
            "source structural load credit"),
    }
    for key in (
            "classic_retaining_path_mm", "captive_land_mm",
            "interface_gap_mm", "paired_magnet_face_separation_mm"):
        contract[key] = _float(
            _required(site, key, f"source {key}"), f"source {key}")
    for key in ("roof_height_mm", "side_wall_margin_mm"):
        if key in site:
            contract[key] = _float(site[key], f"source {key}")
    if closure == "transverse_gable_45deg":
        contract["actual_face_xyz_mm"] = list(_vec3(
            site["actual_face_xyz_mm"], "source actual face"))
        contract["material_inward_xyz"] = list(_vec3(
            site["material_inward_xyz"], "source material inward"))
    return contract


def _validate_cavity_audit_proxies(
    artifacts: Sequence[dict[str, Any]],
) -> None:
    """Bind every declared oversized site to one exact same-state split site."""
    by_id = {artifact["id"]: artifact for artifact in artifacts}
    if len(by_id) != len(artifacts):
        raise AuditError("duplicate artifact IDs prevent proxy validation")
    for artifact in artifacts:
        mode = artifact.get("p2s_printability")
        declared = artifact.get("cavity_audit_proxies")
        if mode not in (None, "not_printable_oversize"):
            raise AuditError(
                f"{artifact['id']}: unsupported p2s_printability {mode!r}")
        if mode != "not_printable_oversize":
            if declared not in (None, []):
                raise AuditError(
                    f"{artifact['id']}: cavity proxies are allowed only for "
                    "an explicitly oversized non-P2S artifact")
            artifact["cavity_audit_proxies"] = []
            continue
        if not isinstance(declared, list) or not declared:
            raise AuditError(
                f"{artifact['id']}: oversized artifact lacks cavity proxies")
        source_sites = {site["name"]: site for site in artifact["sites"]}
        if len(source_sites) != len(artifact["sites"]):
            raise AuditError(f"{artifact['id']}: duplicate source site names")
        clean = []
        seen_sources: set[str] = set()
        seen_targets: set[tuple[str, str]] = set()
        for index, raw in enumerate(declared):
            if not isinstance(raw, dict):
                raise AuditError(
                    f"{artifact['id']}: proxy {index} is not an object")
            source_name = str(raw.get("site", ""))
            target_id = str(raw.get("artifact_id", ""))
            target_name = str(raw.get("proxy_site", ""))
            if not source_name or not target_id or not target_name:
                raise AuditError(
                    f"{artifact['id']}: proxy {index} lacks site/artifact/site")
            if source_name in seen_sources:
                raise AuditError(
                    f"{artifact['id']}: site {source_name} has multiple proxies")
            if (target_id, target_name) in seen_targets:
                raise AuditError(
                    f"{artifact['id']}: proxy target is reused: "
                    f"{target_id}/{target_name}")
            source = source_sites.get(source_name)
            target = by_id.get(target_id)
            if source is None or target is None:
                raise AuditError(
                    f"{artifact['id']}: unresolved proxy "
                    f"{source_name} -> {target_id}/{target_name}")
            if (target["state"] != artifact["state"]
                    or target["variant"] != "Obi-Wan-split"):
                raise AuditError(
                    f"{artifact['id']}: proxy {target_id} is not a same-state "
                    "Obi-Wan-split artifact")
            if target.get("p2s_printability") == "not_printable_oversize":
                raise AuditError(
                    f"{artifact['id']}: proxy {target_id} is itself declared "
                    "not P2S-printable")
            target_matches = [
                site for site in target["sites"]
                if site["name"] == target_name
            ]
            if len(target_matches) != 1:
                raise AuditError(
                    f"{artifact['id']}: proxy site {target_id}/{target_name} "
                    "is missing or ambiguous")
            proxy_site = target_matches[0]
            if source["source_contract_sha256"] != proxy_site[
                    "source_contract_sha256"]:
                raise AuditError(
                    f"{artifact['id']}: source-space cavity contract differs "
                    f"for {source_name} and {target_id}/{target_name}")
            seen_sources.add(source_name)
            seen_targets.add((target_id, target_name))
            clean.append({
                "site": source_name,
                "artifact_id": target_id,
                "proxy_site": target_name,
                "source_contract_sha256": source["source_contract_sha256"],
            })
        if seen_sources != set(source_sites):
            missing = sorted(set(source_sites) - seen_sources)
            raise AuditError(
                f"{artifact['id']}: not every monolith site has one proxy: "
                f"{missing}")
        artifact["cavity_audit_proxies"] = clean


def _print_field(
    site: Mapping[str, Any],
    artifact: Mapping[str, Any],
    name: str,
    *,
    vector: bool = False,
) -> tuple[float, float, float]:
    aliases = (
        f"print_{name}", f"stl_{name}", f"print_space_{name}",
    )
    explicit: tuple[float, float, float] | None = None
    print_space = site.get("print_space")
    if isinstance(print_space, dict):
        value = _lookup(print_space, name, f"{name}_mm", *aliases)
        if value is not None:
            explicit = _vec3(value, f"site print_space.{name}")
    value = _lookup(site, *aliases)
    if value is not None and explicit is None:
        explicit = _vec3(value, f"site {name}")
    source = _lookup(site, name, f"{name}_mm")
    matrix_value = _lookup(
        artifact, "source_to_stl_matrix", "print_transform_matrix",
        "source_to_print_matrix")
    if source is not None and matrix_value is not None:
        matrix = _matrix4(matrix_value, "artifact source-to-STL transform")
        source_vec = _vec3(source, f"site source {name}")
        transformed = (_transform_vector(matrix, source_vec) if vector
                       else _transform_point(matrix, source_vec))
        if explicit is not None and any(
                abs(a - b) > 1.0e-6
                for a, b in zip(explicit, transformed)):
            raise AuditError(
                f"site {site.get('name', '<unnamed>')} print-space {name} "
                f"{explicit} disagrees with source-to-STL transform "
                f"{transformed}")
        return explicit or transformed
    if explicit is not None:
        return explicit
    raise AuditError(
        f"site {site.get('name', '<unnamed>')} lacks post-export print-space {name}")


def normalize_catalog(
    catalog_path: Path, *, enforce_release_inventory: bool = True,
) -> dict[str, Any]:
    try:
        catalog_bytes = catalog_path.read_bytes()
        data = json.loads(catalog_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {catalog_path}: {exc}") from exc
    catalog_input_sha256 = _sha256_bytes(catalog_bytes)
    if not isinstance(data, dict):
        raise AuditError("release catalog root must be an object")
    root_required = {
        "schema_version", "schema_sha256", "catalog_kind", "generated_by",
        "print_contract", "geometry", "inventory", "exclusions",
        "source_revision", "artifacts",
    }
    missing_root = sorted(root_required - set(data))
    if missing_root:
        raise AuditError(
            f"release catalog lacks required root fields: {missing_root}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise AuditError(
            f"release catalog schema must be {SCHEMA_VERSION}, got "
            f"{data.get('schema_version')!r}")
    if data.get("catalog_kind") != "released_pause_and_bury_captive_magnets":
        raise AuditError("release catalog_kind is not the production contract")
    if not CATALOG_SCHEMA.is_file():
        raise AuditError(f"catalog schema is missing: {CATALOG_SCHEMA}")
    try:
        schema_bytes = CATALOG_SCHEMA.read_bytes()
        schema = json.loads(schema_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(
            f"cannot read catalog schema {CATALOG_SCHEMA}: {exc}") from exc
    catalog_schema_sha256 = _sha256_bytes(schema_bytes)
    if data.get("schema_sha256") != catalog_schema_sha256:
        raise AuditError("release catalog is stale against its JSON schema")
    try:
        validate_json_schema(data, schema)
    except JsonSchemaSubsetError as exc:
        raise AuditError(
            f"release catalog violates its JSON schema: {exc}") from exc
    source_revision = data.get("source_revision")
    if (not isinstance(source_revision, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_revision)):
        raise AuditError(
            "release source_revision must be one non-null 64-hex digest")
    if data.get("print_contract") != RELEASE_ACOUSTIC_PRINT_CONTRACT:
        raise AuditError(
            "release print_contract must exactly require every released "
            "acoustic part front-face-down with in-bed Z rotation only")
    if not isinstance(data.get("geometry"), dict):
        raise AuditError("release geometry must be an object")
    if not isinstance(data.get("inventory"), dict):
        raise AuditError("release inventory must be an object")
    exclusions = data.get("exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        raise AuditError("release exclusions must be a non-empty list")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AuditError("release catalog must contain a non-empty artifacts list")
    normalized = []
    seen = set()
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, dict):
            raise AuditError(f"artifact {index} must be an object")
        artifact_required = {
            "id", "state", "variant", "part", "stl", "stl_sha256",
            "print_sidecar", "print_sidecar_sha256", "source_files",
            "source_file_sha256",
            "print_orientation", "rotation_deg", "source_to_stl_matrix",
            "sites",
        }
        missing_artifact = sorted(artifact_required - set(raw))
        if missing_artifact:
            raise AuditError(
                f"artifact {index} lacks required fields: {missing_artifact}")
        state = str(raw.get("state", "shared"))
        variant = str(raw.get("variant", "unknown"))
        part = str(_lookup(raw, "part", "name") or "").strip()
        if not part:
            raise AuditError(f"artifact {index} lacks part/name")
        artifact_id = str(raw.get("id") or f"{state}:{variant}:{part}")
        if artifact_id in seen:
            raise AuditError(f"duplicate artifact id {artifact_id!r}")
        seen.add(artifact_id)
        orientation = str(raw.get("print_orientation", ""))
        if orientation != "front_face_down":
            raise AuditError(
                f"{artifact_id}: print_orientation must be front_face_down, "
                f"got {orientation!r}")
        try:
            source_to_stl_matrix = validate_front_down_transform(
                raw, label=f"catalog artifact {artifact_id}")
        except FrontDownContractError as exc:
            raise AuditError(str(exc)) from exc
        rotation = {
            "x": _float(raw["rotation_deg"]["x"], "catalog X rotation"),
            "z": _float(raw["rotation_deg"]["z"], "catalog Z rotation"),
        }
        stl_value = _lookup(raw, "stl", "stl_path", "path")
        if not isinstance(stl_value, str):
            raise AuditError(f"{artifact_id}: missing STL path")
        stl = _resolve_path(stl_value, catalog_path.parent)
        stl_sha = str(raw["stl_sha256"])
        print_sidecar_value = raw["print_sidecar"]
        print_sidecar_sha = str(raw["print_sidecar_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", stl_sha):
            raise AuditError(f"{artifact_id}: invalid catalog STL SHA-256")
        if (not isinstance(print_sidecar_value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", print_sidecar_sha)):
            raise AuditError(
                f"{artifact_id}: invalid print-transform authority binding")
        print_sidecar = _resolve_path(
            print_sidecar_value, catalog_path.parent)
        sites_raw = raw.get("sites")
        if not isinstance(sites_raw, list) or not sites_raw:
            raise AuditError(f"{artifact_id}: no magnet sites")
        sites = []
        for site_index, raw_site in enumerate(sites_raw):
            if not isinstance(raw_site, dict):
                raise AuditError(f"{artifact_id}: site {site_index} is not an object")
            site = _site_mapping(raw_site)
            site_name = str(site.get("name") or f"site_{site_index + 1}")
            closure = str(site.get("closure_kind", ""))
            if closure not in (
                    "transverse_gable_45deg",
                    "axis_parallel_conical_45deg",
                    "axis_opposed_conical_45deg"):
                raise AuditError(
                    f"{artifact_id}/{site_name}: unsupported closure {closure!r}")
            bury = _lookup(
                site, "cavity_bury_roof_start_print_z_mm",
                "roof_start_print_z_mm", "bury_plane_print_z_mm")
            apex = _lookup(site, "roof_apex_print_z_mm")
            if bury is None or apex is None:
                raise AuditError(
                    f"{artifact_id}/{site_name}: missing bury plane or roof apex")
            cavity_center = _print_field(
                site, raw, "cavity_center_xyz_mm")
            seated_center = _print_field(
                site, raw, "seated_magnet_center_xyz_mm")
            marked_axis = _print_field(
                site, raw, "marked_pole_axis_xyz", vector=True)
            insertion_direction = _print_field(
                site, raw, "insertion_direction_xyz", vector=True)
            if any(not math.isclose(
                    actual, expected, abs_tol=1.0e-9, rel_tol=0.0)
                    for actual, expected in zip(
                        insertion_direction,
                        PRINT_INSERTION_DIRECTION_XYZ,
                        strict=True)):
                raise AuditError(
                    f"{artifact_id}/{site_name}: insertion direction must "
                    "be print [0, 0, -1] (vertically downward from +Z); "
                    f"got {insertion_direction}")
            source_contract = _source_site_contract(site)
            for key, expected in RELEASE_SITE_GEOMETRY_MM.items():
                actual = _float(_required(
                    site, key, f"{artifact_id}/{site_name}: {key}"), key)
                if not math.isclose(actual, expected, abs_tol=1.0e-9):
                    raise AuditError(
                        f"{artifact_id}/{site_name}: {key} must be "
                        f"{expected:.3f}, got {actual:.6f}")
            polarity = str(_required(
                site, "polarity_instruction",
                f"{artifact_id}/{site_name}: polarity instruction")).strip()
            if not polarity:
                raise AuditError(
                    f"{artifact_id}/{site_name}: polarity instruction is empty")
            magnet_count = int(_float(_required(
                site, "magnet_count",
                f"{artifact_id}/{site_name}: magnet count"),
                "magnet count"))
            if magnet_count != 1:
                raise AuditError(
                    f"{artifact_id}/{site_name}: magnet_count must be 1")
            structural_credit = _float(_required(
                site, "structural_load_credit_n",
                f"{artifact_id}/{site_name}: structural load credit"),
                "structural load credit")
            if abs(structural_credit) > 1.0e-12:
                raise AuditError(
                    f"{artifact_id}/{site_name}: magnets receive zero "
                    "structural-load credit")
            installed_axis = _vec3(_required(
                site, "installed_marked_pole_axis_xyz",
                f"{artifact_id}/{site_name}: installed marked-pole axis"),
                "installed marked-pole axis")
            normalized_site: dict[str, Any] = {
                "name": site_name,
                "closure_kind": closure,
                "cavity_bury_roof_start_print_z_mm": _float(bury, "bury plane"),
                "roof_apex_print_z_mm": _float(apex, "roof apex"),
                "print_cavity_center_xyz_mm": cavity_center,
                "print_seated_magnet_center_xyz_mm": seated_center,
                "print_marked_pole_axis_xyz": marked_axis,
                "print_insertion_direction_xyz": insertion_direction,
                "magnet_diameter_mm": source_contract["magnet_diameter_mm"],
                "magnet_depth_mm": source_contract["magnet_depth_mm"],
                "cavity_diameter_mm": source_contract["cavity_diameter_mm"],
                "cavity_depth_mm": source_contract["cavity_depth_mm"],
                "face_skin_mm": source_contract["face_skin_mm"],
                "inner_skin_mm": source_contract["inner_skin_mm"],
                "roof_angle_deg": source_contract["roof_angle_deg"],
                "captive_land_mm": source_contract["captive_land_mm"],
                "interface_gap_mm": source_contract["interface_gap_mm"],
                "paired_magnet_face_separation_mm": source_contract[
                    "paired_magnet_face_separation_mm"],
                "classic_retaining_path_mm": source_contract[
                    "classic_retaining_path_mm"],
                "polarity_instruction": polarity,
                "installed_marked_pole_axis_xyz": installed_axis,
                "magnet_count": magnet_count,
                "structural_load_credit_n": structural_credit,
                "expected_pause_marker_z_mm": (
                    None if site.get("expected_pause_marker_z_mm") is None
                    else _float(site["expected_pause_marker_z_mm"],
                                "expected pause marker")),
                "source_contract": source_contract,
                "source_contract_sha256": _sha256_bytes(
                    _canonical_json(source_contract)),
            }
            if closure == "transverse_gable_45deg":
                normalized_site["print_actual_face_xyz_mm"] = _print_field(
                    site, raw, "actual_face_xyz_mm")
                normalized_site["print_material_inward_xyz"] = _print_field(
                    site, raw, "material_inward_xyz", vector=True)
            sites.append(normalized_site)
        source_values = raw.get("source_files")
        if (not isinstance(source_values, list) or not source_values
                or any(not isinstance(value, str) or not value
                       for value in source_values)):
            raise AuditError(f"{artifact_id}: source_files must be non-empty")
        source_hash_values = raw.get("source_file_sha256")
        if not isinstance(source_hash_values, dict):
            raise AuditError(
                f"{artifact_id}: source_file_sha256 must be an object")
        if (len(set(source_values)) != len(source_values)
                or set(source_hash_values) != set(source_values)):
            raise AuditError(
                f"{artifact_id}: source_file_sha256 must bind every exact "
                "source_files entry once")
        source_files = []
        source_file_sha256: dict[Path, str] = {}
        for value in source_values:
            digest = source_hash_values.get(value)
            if (not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise AuditError(
                    f"{artifact_id}: invalid source hash for {value!r}")
            source_path = _resolve_path(value, catalog_path.parent)
            if source_path in source_file_sha256:
                raise AuditError(
                    f"{artifact_id}: source_files entries alias the same "
                    f"resolved path: {source_path}")
            source_files.append(source_path)
            source_file_sha256[source_path] = digest
        auxiliary_bindings: dict[str, Any] = {}
        for path_key, hash_key in (
                ("transaction_manifest", "transaction_manifest_sha256"),
                ("facts", "facts_sha256"),
                ("stage_manifest", "stage_manifest_sha256")):
            path_value = raw.get(path_key)
            hash_value = raw.get(hash_key)
            if path_value is None and hash_value is None:
                continue
            if (not isinstance(path_value, str) or not path_value
                    or not isinstance(hash_value, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", hash_value)):
                raise AuditError(
                    f"{artifact_id}: invalid {path_key} path/hash binding")
            auxiliary_bindings[path_key] = _resolve_path(
                path_value, catalog_path.parent)
            auxiliary_bindings[hash_key] = hash_value
        if variant in ("Obi-Wan-Ac", "Obi-Wan-Ae") and not {
                "transaction_manifest", "transaction_manifest_sha256",
                "facts", "facts_sha256"} <= set(auxiliary_bindings):
            raise AuditError(
                f"{artifact_id}: Ac/Ae artifact lacks facts/transaction "
                "manifest hash bindings")
        if variant in ("Obi-Wan", "Obi-Wan-split") and not {
                "stage_manifest", "stage_manifest_sha256"
        } <= set(auxiliary_bindings):
            raise AuditError(
                f"{artifact_id}: Obi-Wan artifact lacks staged-build manifest "
                "hash binding")
        normalized.append({
            "id": artifact_id,
            "state": state,
            "variant": variant,
            "part": part,
            "stl": stl,
            "stl_catalog_value": stl_value,
            "stl_catalog_sha256": stl_sha,
            "print_sidecar": print_sidecar,
            "print_sidecar_catalog_value": print_sidecar_value,
            "print_sidecar_sha256": print_sidecar_sha,
            "print_orientation": orientation,
            "rotation_deg": rotation,
            "source_to_stl_matrix": source_to_stl_matrix,
            "sites": sites,
            "source_files": source_files,
            "source_file_sha256": source_file_sha256,
            "catalog_source_revision": source_revision,
            "catalog_sha256": catalog_input_sha256,
            "p2s_printability": raw.get("p2s_printability"),
            "cavity_audit_proxies": raw.get("cavity_audit_proxies"),
            "catalog_record": raw,
            **auxiliary_bindings,
        })
    _validate_cavity_audit_proxies(normalized)
    inventory = data["inventory"]
    declared_artifacts = inventory.get("artifact_count")
    declared_magnets = inventory.get("magnet_count")
    actual_magnets = sum(
        int(site["magnet_count"])
        for artifact in normalized for site in artifact["sites"])
    if declared_artifacts != len(normalized) or declared_magnets != actual_magnets:
        raise AuditError(
            "catalog inventory does not match its normalized contents: "
            f"declared={declared_artifacts}/{declared_magnets}, "
            f"actual={len(normalized)}/{actual_magnets}")
    if enforce_release_inventory and (
            len(normalized) != EXPECTED_RELEASE_ARTIFACT_COUNT
            or actual_magnets != EXPECTED_RELEASE_MAGNET_COUNT):
        raise AuditError(
            "production release inventory must remain "
            f"{EXPECTED_RELEASE_ARTIFACT_COUNT} artifacts / "
            f"{EXPECTED_RELEASE_MAGNET_COUNT} captive stations; got "
            f"{len(normalized)} / {actual_magnets}")
    result = dict(data)
    result["artifacts"] = normalized
    result["_catalog_sha256"] = catalog_input_sha256
    result["_catalog_schema_sha256"] = catalog_schema_sha256
    if sha256_file(catalog_path) != catalog_input_sha256:
        raise AuditError("release catalog changed while it was normalized")
    if sha256_file(CATALOG_SCHEMA) != catalog_schema_sha256:
        raise AuditError("release catalog schema changed while it was normalized")
    return result


def _validate_artifact_bindings(artifact: Mapping[str, Any]) -> None:
    """Reject any STL or print-transform authority changed after cataloging."""
    stl: Path = artifact["stl"]
    authority: Path = artifact["print_sidecar"]
    if not stl.is_file():
        raise AuditError(f"{artifact['id']}: STL is missing: {stl}")
    if sha256_file(stl) != artifact["stl_catalog_sha256"]:
        raise AuditError(f"{artifact['id']}: STL hash differs from catalog")
    if not authority.is_file():
        raise AuditError(
            f"{artifact['id']}: print-transform authority is missing: "
            f"{authority}")
    if sha256_file(authority) != artifact["print_sidecar_sha256"]:
        raise AuditError(
            f"{artifact['id']}: print-transform authority hash differs "
            "from catalog")
    try:
        sidecar = validate_print_sidecar(stl, authority)
    except (FrontDownContractError, OSError) as exc:
        raise AuditError(
            f"{artifact['id']}: invalid adjacent print sidecar: {exc}") from exc
    if sidecar.get("stl_sha256") != artifact["stl_catalog_sha256"]:
        raise AuditError(
            f"{artifact['id']}: sidecar STL hash differs from catalog")
    if sidecar.get("print_orientation") != artifact["print_orientation"]:
        raise AuditError(
            f"{artifact['id']}: sidecar orientation differs from catalog")
    sidecar_rotation = sidecar.get("rotation_deg")
    if sidecar_rotation != artifact["rotation_deg"]:
        raise AuditError(
            f"{artifact['id']}: sidecar rotation differs from catalog")
    try:
        sidecar_matrix = _matrix4(
            sidecar.get("source_to_stl_matrix"),
            f"{artifact['id']} sidecar source-to-STL transform")
    except AuditError as exc:
        raise AuditError(
            f"{artifact['id']}: invalid sidecar transform: {exc}") from exc
    if sidecar_matrix != artifact["source_to_stl_matrix"]:
        raise AuditError(
            f"{artifact['id']}: sidecar matrix differs from catalog")
    source_bindings = artifact.get("source_file_sha256")
    if (not isinstance(source_bindings, Mapping)
            or set(source_bindings) != set(artifact["source_files"])):
        raise AuditError(
            f"{artifact['id']}: normalized source hash bindings are incomplete")
    for source in artifact["source_files"]:
        expected = source_bindings[source]
        if not source.is_file():
            raise AuditError(
                f"{artifact['id']}: bound source file is missing: {source}")
        if sha256_file(source) != expected:
            raise AuditError(
                f"{artifact['id']}: source hash differs from catalog: "
                f"{source}")
    for path_key, hash_key in (
            ("transaction_manifest", "transaction_manifest_sha256"),
            ("facts", "facts_sha256"),
            ("stage_manifest", "stage_manifest_sha256")):
        path = artifact.get(path_key)
        expected = artifact.get(hash_key)
        if path is None and expected is None:
            continue
        if (not isinstance(path, Path) or not path.is_file()
                or sha256_file(path) != expected):
            raise AuditError(
                f"{artifact['id']}: {path_key} hash differs from catalog")


def _copy_hash_bound_file(
    source: Path, destination: Path, expected_sha256: str, label: str,
) -> None:
    """Copy one release input and prove both sides stayed byte-identical."""
    if not source.is_file():
        raise AuditError(f"{label} is missing: {source}")
    if sha256_file(source) != expected_sha256:
        raise AuditError(f"{label} changed before immutable staging")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise AuditError(f"{label} staging copy differs from release bytes")
    temporary.replace(destination)
    destination.chmod(0o444)
    if sha256_file(source) != expected_sha256:
        raise AuditError(f"{label} changed while it was being staged")


def _source_snapshot(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for artifact in artifacts:
        for path in artifact["source_files"]:
            key = str(path)
            if key in snapshot:
                if snapshot[key] != artifact["source_file_sha256"][path]:
                    raise AuditError(
                        f"conflicting catalog hashes for shared source: {path}")
                continue
            snapshot[key] = artifact["source_file_sha256"][path]
    return snapshot


def _stage_release_inputs(
    catalog_path: Path,
    artifacts: Sequence[Mapping[str, Any]],
    stage_root: Path,
    *,
    expected_catalog_sha256: str | None = None,
    expected_catalog_schema_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze the exact catalog/STL/sidecar bytes consumed by the slicer.

    The normalized catalog still owns semantic validation, while every Bambu
    process reads a private, read-only STL copy.  This prevents a concurrent
    remote promotion from mixing two release revisions within one audit.
    """
    stage_root.mkdir(parents=True, exist_ok=True)
    catalog_sha = sha256_file(catalog_path)
    if (expected_catalog_sha256 is not None
            and catalog_sha != expected_catalog_sha256):
        raise AuditError("release catalog changed before immutable staging")
    staged_catalog = stage_root / "captive_magnet_release_catalog.json"
    _copy_hash_bound_file(
        catalog_path, staged_catalog, catalog_sha, "release catalog")
    schema_sha = sha256_file(CATALOG_SCHEMA)
    if (expected_catalog_schema_sha256 is not None
            and schema_sha != expected_catalog_schema_sha256):
        raise AuditError(
            "release catalog schema changed before immutable staging")
    staged_schema = stage_root / CATALOG_SCHEMA.name
    _copy_hash_bound_file(
        CATALOG_SCHEMA, staged_schema, schema_sha, "release catalog schema")
    staged_artifacts: list[dict[str, Any]] = []
    originals: list[Mapping[str, Any]] = []
    for index, artifact in enumerate(sorted(
            artifacts, key=lambda item: item["id"])):
        _validate_artifact_bindings(artifact)
        originals.append(artifact)
        artifact_dir = stage_root / "artifacts" / (
            f"{index:03d}_{_slug(artifact['id'])}")
        staged_stl = artifact_dir / artifact["stl"].name
        staged_sidecar = artifact_dir / artifact["print_sidecar"].name
        _copy_hash_bound_file(
            artifact["stl"], staged_stl,
            artifact["stl_catalog_sha256"],
            f"{artifact['id']} STL")
        _copy_hash_bound_file(
            artifact["print_sidecar"], staged_sidecar,
            artifact["print_sidecar_sha256"],
            f"{artifact['id']} print sidecar")
        staged = dict(artifact)
        staged["release_stl"] = artifact["stl"]
        staged["release_print_sidecar"] = artifact["print_sidecar"]
        staged["stl"] = staged_stl
        staged["print_sidecar"] = staged_sidecar
        staged_sources: list[Path] = []
        staged_source_hashes: dict[Path, str] = {}
        for source_index, source in enumerate(artifact["source_files"]):
            expected = artifact["source_file_sha256"][source]
            staged_source = (
                artifact_dir / "sources"
                / f"{source_index:02d}_{source.name}")
            _copy_hash_bound_file(
                source, staged_source, expected,
                f"{artifact['id']} source {source.name}")
            staged_sources.append(staged_source)
            staged_source_hashes[staged_source] = expected
        staged["release_source_files"] = artifact["source_files"]
        staged["source_files"] = staged_sources
        staged["source_file_sha256"] = staged_source_hashes
        for path_key, hash_key in (
                ("transaction_manifest", "transaction_manifest_sha256"),
                ("facts", "facts_sha256"),
                ("stage_manifest", "stage_manifest_sha256")):
            if path_key not in artifact:
                continue
            source = artifact[path_key]
            destination = artifact_dir / "metadata" / source.name
            _copy_hash_bound_file(
                source, destination, artifact[hash_key],
                f"{artifact['id']} {path_key}")
            staged[f"release_{path_key}"] = source
            staged[path_key] = destination
        _validate_artifact_bindings(staged)
        staged_artifacts.append(staged)
        # Detect a replacement that raced either copy.
        _validate_artifact_bindings(artifact)
    if sha256_file(catalog_path) != catalog_sha:
        raise AuditError("release catalog changed during immutable staging")
    return {
        "catalog_path": staged_catalog,
        "catalog_sha256": catalog_sha,
        "catalog_schema_path": staged_schema,
        "catalog_schema_sha256": schema_sha,
        "artifacts": staged_artifacts,
        "original_artifacts": originals,
        "source_files": _source_snapshot(originals),
    }


def _verify_staged_release_inputs(
    staged: Mapping[str, Any], original_catalog_path: Path,
) -> None:
    """Recheck staged and live authorities immediately before publication."""
    expected_catalog_sha = staged["catalog_sha256"]
    if sha256_file(staged["catalog_path"]) != expected_catalog_sha:
        raise AuditError("immutable staged catalog changed during slicing")
    if sha256_file(original_catalog_path) != expected_catalog_sha:
        raise AuditError("release catalog changed during slicing")
    if (sha256_file(staged["catalog_schema_path"])
            != staged["catalog_schema_sha256"]
            or sha256_file(CATALOG_SCHEMA)
            != staged["catalog_schema_sha256"]):
        raise AuditError("release catalog schema changed during slicing")
    for artifact in staged["artifacts"]:
        _validate_artifact_bindings(artifact)
    for artifact in staged["original_artifacts"]:
        _validate_artifact_bindings(artifact)
    for raw_path, expected_sha in staged["source_files"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise AuditError(
                f"release source changed during slicing: {path}")


def _verify_profile_inputs(
    profile_bundle: Mapping[str, Any], bambu: Path,
) -> None:
    """Reject profile or slicer-binary drift across a long audit run."""
    identity = profile_bundle["identity"]
    expected_audit_sources = profile_bundle.get("audit_source_sha256")
    if not isinstance(expected_audit_sources, Mapping):
        raise AuditError("captive-magnet audit source hashes are missing")
    current_audit_sources = {
        str(path): sha256_file(path) for path in AUDIT_SOURCE_FILES
    }
    if current_audit_sources != dict(expected_audit_sources):
        raise AuditError("captive-magnet audit source changed during slicing")
    if sha256_file(bambu) != identity["binary_sha256"]:
        raise AuditError("Bambu Studio binary changed during slicing")
    for label, path in profile_bundle["paths"].items():
        expected = identity["resolved_profiles"][label]["sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditError(
                f"resolved {label} profile changed during slicing")
    config_path = Path(identity["config_path"])
    if (not config_path.is_file()
            or sha256_file(config_path) != identity["config_sha256"]):
        raise AuditError("slicing-profile contract changed during slicing")
    for record in identity["source_presets"].values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AuditError(f"Bambu source preset changed during slicing: {path}")
    vendor_root = Path(identity["system_vendor_root"])
    for record in identity["resolution_dependencies"]:
        path = vendor_root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise AuditError(
                f"Bambu preset dependency changed during slicing: {path}")


@dataclasses.dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    e_delta: float
    feature: str
    line_width: float | None
    line_number: int
    z0: float = 0.0
    z1: float = 0.0

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclasses.dataclass
class Layer:
    z: float
    layer_height: float | None
    segments: list[Segment]
    line_number: int


@dataclasses.dataclass
class ParsedGcode:
    layers: list[Layer]
    movement_commands: int
    arc_commands: int
    extrusion_moves: int
    temperature_commands: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    config: dict[str, str]


_ARG_RE = re.compile(
    r"(?:^|\s)([XYZEFIJKR])"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def _segment_intersects_regions(
    x0: float, y0: float, x1: float, y1: float,
    regions: Sequence[tuple[float, float, float, float]] | None,
) -> bool:
    if regions is None:
        return True
    low_x, high_x = min(x0, x1), max(x0, x1)
    low_y, high_y = min(y0, y1), max(y0, y1)
    return any(
        low_x <= region[2] and high_x >= region[0]
        and low_y <= region[3] and high_y >= region[1]
        for region in regions
    )


def parse_gcode(
    path: Path,
    *,
    retain_regions: Sequence[tuple[float, float, float, float]] | None = None,
) -> ParsedGcode:
    """Parse Bambu G-code and retain only requested local extrusion paths.

    G2/G3 arcs are tessellated while streaming so I/J arc fitting cannot hide
    a circular retaining wall or loading obstruction.  Production calls pass
    cavity ROIs, bounding retained memory independently of overall part size;
    all arc subpoints still contribute to global motion bounds.
    """
    layers: list[Layer] = []
    pending_change = False
    current: Layer | None = None
    x = y = z = e = 0.0
    xyz_absolute = True
    e_absolute = False
    feature = "Undefined"
    line_width: float | None = None
    movement = arcs = extrusion = temperatures = 0
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    config: dict[str, str] = {}
    in_config = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if line == "; CONFIG_BLOCK_START":
                in_config = True
                continue
            if line == "; CONFIG_BLOCK_END":
                in_config = False
                continue
            if in_config and line.startswith("; ") and " = " in line:
                key, value = line[2:].split(" = ", 1)
                config[key.strip()] = value.strip()
                continue
            if line == "; CHANGE_LAYER":
                pending_change = True
                continue
            if pending_change and line.startswith("; Z_HEIGHT:"):
                layer_z = _float(line.split(":", 1)[1], "G-code layer Z")
                current = Layer(layer_z, None, [], line_number)
                layers.append(current)
                pending_change = False
                continue
            # Bambu repeats ``LAYER_HEIGHT`` inside individual Bridge feature
            # blocks, where it describes the bridge-flow bead rather than the
            # scheduled layer.  Only the first value immediately following a
            # CHANGE_LAYER/Z_HEIGHT header is authoritative for the layer
            # schedule; never let later feature metadata overwrite it.
            if (current is not None and current.layer_height is None
                    and line.startswith("; LAYER_HEIGHT:")):
                current.layer_height = _float(
                    line.split(":", 1)[1], "G-code layer height")
                continue
            if line.startswith("; FEATURE:"):
                feature = line.split(":", 1)[1].strip()
                continue
            if line.startswith("; LINE_WIDTH:"):
                try:
                    line_width = float(line.split(":", 1)[1])
                except ValueError:
                    line_width = None
                continue
            command = line.split(";", 1)[0].strip()
            if not command:
                continue
            token = command.split(None, 1)[0]
            if token == "G90.1":
                raise AuditError(
                    f"{path}:{line_number}: absolute arc-centre mode G90.1 "
                    "is unsupported; expected relative I/J")
            if token == "G91.1":
                # Explicitly confirms the normal relative-I/J convention.
                continue
            if token == "G90":
                xyz_absolute = True
                continue
            if token == "G91":
                xyz_absolute = False
                continue
            if token == "M82":
                e_absolute = True
                continue
            if token == "M83":
                e_absolute = False
                continue
            if token == "G92":
                args = {key: float(value) for key, value in _ARG_RE.findall(command)}
                x = args.get("X", x)
                y = args.get("Y", y)
                z = args.get("Z", z)
                e = args.get("E", e)
                continue
            if token in ("M104", "M109", "M140", "M190"):
                temperatures += 1
            if token not in ("G0", "G1", "G2", "G3"):
                continue
            movement += 1
            args = {key: float(value) for key, value in _ARG_RE.findall(command)}
            old_x, old_y, old_z, old_e = x, y, z, e
            if xyz_absolute:
                x = args.get("X", x)
                y = args.get("Y", y)
                z = args.get("Z", z)
            else:
                x += args.get("X", 0.0)
                y += args.get("Y", 0.0)
                z += args.get("Z", 0.0)
            if "E" in args:
                if e_absolute:
                    e = args["E"]
                    e_delta = e - old_e
                else:
                    e_delta = args["E"]
                    e += e_delta
            else:
                e_delta = 0.0
            if token in ("G2", "G3"):
                arcs += 1
                if "R" in args:
                    raise AuditError(
                        f"{path}:{line_number}: radius-encoded {token} is "
                        "unsupported; expected relative I/J")
                if "I" not in args and "J" not in args:
                    raise AuditError(
                        f"{path}:{line_number}: {token} lacks relative I/J")
                center_x = old_x + args.get("I", 0.0)
                center_y = old_y + args.get("J", 0.0)
                start_radius = math.hypot(old_x - center_x, old_y - center_y)
                end_radius = math.hypot(x - center_x, y - center_y)
                if start_radius <= FLOAT_EPS:
                    raise AuditError(
                        f"{path}:{line_number}: {token} has zero I/J radius")
                same_endpoint = math.hypot(x - old_x, y - old_y) <= 1.0e-4
                if (not same_endpoint
                        and abs(end_radius - start_radius)
                        > ARC_RADIUS_TOLERANCE_MM):
                    raise AuditError(
                        f"{path}:{line_number}: {token} start/end radii differ "
                        f"by {abs(end_radius - start_radius):.4f} mm")
                start_angle = math.atan2(
                    old_y - center_y, old_x - center_x)
                end_angle = math.atan2(y - center_y, x - center_x)
                if same_endpoint:
                    sweep = -2.0 * math.pi if token == "G2" else 2.0 * math.pi
                    end_radius = start_radius
                elif token == "G2":
                    magnitude = (start_angle - end_angle) % (2.0 * math.pi)
                    sweep = -(magnitude or 2.0 * math.pi)
                else:
                    magnitude = (end_angle - start_angle) % (2.0 * math.pi)
                    sweep = magnitude or 2.0 * math.pi
                mean_radius = (start_radius + end_radius) / 2.0
                planar_length = abs(sweep) * mean_radius
                path_length = math.hypot(planar_length, z - old_z)
                count = max(
                    1, int(math.ceil(path_length / ARC_TESSELLATION_STEP_MM)))
                if count > MAX_ARC_TESSELLATION_SEGMENTS:
                    raise AuditError(
                        f"{path}:{line_number}: {token} needs {count} "
                        "tessellation segments; refusing unbounded arc")
                prior_x, prior_y, prior_z = old_x, old_y, old_z
                for index in range(1, count + 1):
                    t = index / count
                    angle = start_angle + sweep * t
                    radius = start_radius + (end_radius - start_radius) * t
                    next_x = center_x + radius * math.cos(angle)
                    next_y = center_y + radius * math.sin(angle)
                    next_z = old_z + (z - old_z) * t
                    if index == count:
                        next_x, next_y, next_z = x, y, z
                    for axis, value in enumerate((next_x, next_y, next_z)):
                        mins[axis] = min(mins[axis], value)
                        maxs[axis] = max(maxs[axis], value)
                    if (current is not None and e_delta > 1.0e-8
                            and math.hypot(
                                next_x - prior_x,
                                next_y - prior_y) > 1.0e-8
                            and _segment_intersects_regions(
                                prior_x, prior_y, next_x, next_y,
                                retain_regions)):
                        current.segments.append(Segment(
                            prior_x, prior_y, next_x, next_y,
                            e_delta / count, feature, line_width, line_number,
                            prior_z, next_z))
                    prior_x, prior_y, prior_z = next_x, next_y, next_z
                if e_delta > 1.0e-8 and planar_length > 1.0e-8:
                    extrusion += 1
            else:
                for axis, value in enumerate((old_x, old_y, old_z)):
                    mins[axis] = min(mins[axis], value)
                    maxs[axis] = max(maxs[axis], value)
                for axis, value in enumerate((x, y, z)):
                    mins[axis] = min(mins[axis], value)
                    maxs[axis] = max(maxs[axis], value)
                if (current is not None and e_delta > 1.0e-8
                        and math.hypot(x - old_x, y - old_y) > 1.0e-8):
                    if _segment_intersects_regions(
                            old_x, old_y, x, y, retain_regions):
                        current.segments.append(Segment(
                            old_x, old_y, x, y, e_delta, feature,
                            line_width, line_number, old_z, z))
                    extrusion += 1
    if not layers:
        raise AuditError(f"no Bambu CHANGE_LAYER/Z_HEIGHT records in {path}")
    if not movement or not extrusion or not temperatures:
        raise AuditError(
            f"invalid G-code {path}: moves={movement}, extrusion={extrusion}, "
            f"temperatures={temperatures}")
    return ParsedGcode(
        layers, movement, arcs, extrusion, temperatures,
        tuple(mins), tuple(maxs), config)  # type: ignore[arg-type]


def _validate_actual_gcode_profile(
    parsed: ParsedGcode,
    profile_bundle: Mapping[str, Any],
) -> list[str]:
    """Verify slicer output, not merely requested profile inputs."""
    errors = []
    expected = profile_bundle["identity"]["effective"]
    actual_fields = {
        "layer_height": expected["layer_height_mm"],
        "initial_layer_print_height": expected["first_layer_height_mm"],
        "outer_wall_line_width": expected["outer_wall_line_width_mm"],
        "inner_wall_line_width": expected["inner_wall_line_width_mm"],
    }
    for key, expected_value in actual_fields.items():
        value = parsed.config.get(key)
        if value is None:
            errors.append(f"G-code CONFIG_BLOCK lacks {key}")
            continue
        try:
            actual = float(value)
        except ValueError:
            errors.append(f"G-code {key} is not numeric: {value!r}")
            continue
        if not math.isclose(actual, expected_value, abs_tol=1.0e-8):
            errors.append(
                f"G-code {key}={actual:g} != resolved {expected_value:g}")
    if parsed.config.get("wall_generator", "").lower() != "classic":
        errors.append(
            f"G-code wall_generator={parsed.config.get('wall_generator')!r}, "
            "expected classic")
    if _boolish(parsed.config.get("enable_support")):
        errors.append("G-code enables support")
    actual_arc_fitting = _boolish(parsed.config.get("enable_arc_fitting"))
    if actual_arc_fitting != bool(expected.get("arc_fitting_enabled")):
        errors.append(
            f"G-code enable_arc_fitting={actual_arc_fitting} != resolved "
            f"{bool(expected.get('arc_fitting_enabled'))}")
    first_height = expected["first_layer_height_mm"]
    if not math.isclose(parsed.layers[0].z, first_height, abs_tol=0.001):
        errors.append(
            f"first actual layer Z={parsed.layers[0].z:.3f}, "
            f"expected {first_height:.3f}")
    for layer in parsed.layers[1:]:
        if (layer.layer_height is not None
                and not math.isclose(
                    layer.layer_height, expected["layer_height_mm"],
                    abs_tol=0.001)):
            errors.append(
                f"layer at Z={layer.z:.3f} has height "
                f"{layer.layer_height:.3f}, expected "
                f"{expected['layer_height_mm']:.3f}")
            break
    return errors


def _layer_at_or_below(layers: Sequence[Layer], value: float) -> Layer:
    matches = [layer for layer in layers if layer.z <= value + LAYER_EPS]
    if not matches:
        raise AuditError(f"no sliced layer at or below {value:.3f} mm")
    return matches[-1]


def _layer_above(layers: Sequence[Layer], value: float) -> Layer:
    for layer in layers:
        if layer.z > value + LAYER_EPS:
            return layer
    raise AuditError(f"no sliced layer above {value:.3f} mm")


def _layer_at_or_above(layers: Sequence[Layer], value: float) -> Layer:
    for layer in layers:
        if layer.z >= value - LAYER_EPS:
            return layer
    raise AuditError(f"no sliced layer at or above {value:.3f} mm")


def _cavity_retain_regions(
    sites: Sequence[Mapping[str, Any]],
    placement_xy: tuple[float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    """Bound retained G-code segments to the evidence ROI of every site."""
    regions = []
    for site in sites:
        cx, cy, _ = site["print_cavity_center_xyz_mm"]
        cx += placement_xy[0]
        cy += placement_xy[1]
        half = site["cavity_diameter_mm"] / 2.0 + EVIDENCE_MARGIN_MM
        regions.append((cx - half, cy - half, cx + half, cy + half))
    return tuple(regions)


def _seated_magnet_print_z_bounds(
    site: Mapping[str, Any],
) -> tuple[float, float]:
    """Exact vertical bounds of a fully seated cylindrical magnet.

    A cylinder's support extent along print Z combines its axial half-depth
    and the projection of its circular radius perpendicular to that axis.
    This works for both transverse coupon-style discs and the axial V0 site.
    """
    center_z = _vec3(
        site["print_seated_magnet_center_xyz_mm"],
        "seated magnet center")[2]
    axis = _unit3(site["print_marked_pole_axis_xyz"], "magnet axis")
    axis_z = max(-1.0, min(1.0, axis[2]))
    axial_extent = abs(axis_z) * site["magnet_depth_mm"] / 2.0
    radial_extent = (
        math.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        * site["magnet_diameter_mm"] / 2.0
    )
    extent = axial_extent + radial_extent
    return center_z - extent, center_z + extent


def _retaining_stage_pass(
    site: Mapping[str, Any], stage: str, metrics: Mapping[str, Any],
) -> bool:
    """Require real retaining extrusion throughout every open stage.

    At the very first circular-cradle layer the theoretical chord begins at
    zero width, so demanding the representative layer's 3-mm span there
    would reject valid circular bottoms.  It must nevertheless contain both
    axial skin paths.  Representative and last-open layers retain the full
    physically validated threshold.
    """
    retaining = metrics["retaining_paths"]
    if (stage == "lowest_open"
            and site["closure_kind"] == "transverse_gable_45deg"):
        return (
            retaining["interface_skin_path_length_mm"] >= 0.20
            and retaining["inner_skin_path_length_mm"] >= 0.20
            and retaining[
                "interface_skin_longest_contiguous_span_mm"] >= 0.10
            and retaining[
                "inner_skin_longest_contiguous_span_mm"] >= 0.10
        )
    return bool(retaining["pass"])


def _sample_segment(segment: Segment, step: float = SITE_SAMPLE_STEP_MM) -> Iterator[tuple[float, float, float]]:
    length = segment.length
    count = max(1, int(math.ceil(length / step)))
    weight = length / count
    for index in range(count):
        t = (index + 0.5) / count
        yield (
            segment.x0 + t * (segment.x1 - segment.x0),
            segment.y0 + t * (segment.y1 - segment.y0),
            weight,
        )


def _longest_connected_v_span(
    points: Sequence[tuple[float, float]],
) -> float:
    """Return the longest V span of one connected extrusion component.

    Connectivity is measured in the local wall plane, not only after
    projecting onto V.  Thus two fragments at different U positions cannot
    masquerade as one wall merely because their V ranges overlap.  A small
    spatial hash keeps this linear for normal sampled toolpaths.
    """
    if not points:
        return 0.0
    gap = RETAINING_PATH_CONNECTIVITY_GAP_MM
    gap2 = gap * gap
    parents = list(range(len(points)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        a_root, b_root = find(a), find(b)
        if a_root != b_root:
            parents[b_root] = a_root

    cells: dict[tuple[int, int], list[int]] = {}
    for index, (u, v) in enumerate(points):
        cell = (math.floor(u / gap), math.floor(v / gap))
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                for other in cells.get((cell[0] + du, cell[1] + dv), ()):
                    ou, ov = points[other]
                    if (u - ou) ** 2 + (v - ov) ** 2 <= gap2:
                        union(index, other)
        cells.setdefault(cell, []).append(index)

    extents: dict[int, list[float]] = {}
    for index, (_u, v) in enumerate(points):
        bounds = extents.setdefault(find(index), [v, v])
        bounds[0] = min(bounds[0], v)
        bounds[1] = max(bounds[1], v)
    return max(high - low for low, high in extents.values())


def _largest_circular_sample_gap(
    angles: Sequence[float], radius: float,
) -> float:
    """Return the largest uncovered arc between annular path samples."""
    if len(angles) < 2 or radius <= 0.0:
        return math.inf
    ordered = sorted(float(angle) % (2.0 * math.pi) for angle in angles)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + 2.0 * math.pi - ordered[-1])
    return max(gaps) * radius


def _toolpath_metrics(
    layer: Layer,
    site: Mapping[str, Any],
    placement_xy: tuple[float, float],
) -> dict[str, Any]:
    cx, cy, _ = site["print_cavity_center_xyz_mm"]
    cx += placement_xy[0]
    cy += placement_xy[1]
    radius = site["cavity_diameter_mm"] / 2.0
    magnet_radius = site.get("magnet_diameter_mm", 5.0) / 2.0
    interior = 0.0
    boundary_distances: list[float] = []
    local_segments: list[Segment] = []
    wall_a = wall_b = 0.0
    wall_a_v: list[float] = []
    wall_b_v: list[float] = []
    wall_a_uv: list[tuple[float, float]] = []
    wall_b_uv: list[tuple[float, float]] = []
    roi = radius + EVIDENCE_MARGIN_MM
    closure = site["closure_kind"]
    if closure == "transverse_gable_45deg":
        fx, fy, _ = site["print_actual_face_xyz_mm"]
        fx += placement_xy[0]
        fy += placement_xy[1]
        ux, uy = _unit_xy(site["print_material_inward_xyz"], "material inward")
        vx, vy = -uy, ux
        face_skin = site["face_skin_mm"]
        cavity_depth = site["cavity_depth_mm"]
        inner_skin = site["inner_skin_mm"]
        first_wall_center = face_skin / 2.0
        second_wall_center = face_skin + cavity_depth + inner_skin / 2.0
        wall_band = max(0.28, min(face_skin * 0.75, 0.38))
        central_chord_half = min(1.0, magnet_radius / 2.0)
        positive_free_edges: list[float] = []
        negative_free_edges: list[float] = []
        interface_cavity_edges: list[float] = []
        inner_cavity_edges: list[float] = []
        for segment in layer.segments:
            if min(segment.x0, segment.x1) > cx + roi or max(segment.x0, segment.x1) < cx - roi:
                continue
            if min(segment.y0, segment.y1) > cy + roi or max(segment.y0, segment.y1) < cy - roi:
                continue
            local_segments.append(segment)
            path_width = (
                segment.line_width
                if segment.line_width is not None and segment.line_width > 0.0
                else FALLBACK_LINE_WIDTH_MM)
            half_path = path_width / 2.0
            for x, y, weight in _sample_segment(segment):
                dx, dy = x - fx, y - fy
                u = dx * ux + dy * uy
                v = (x - cx) * vx + (y - cy) * vy
                if (face_skin + 0.05 < u < face_skin + cavity_depth - 0.05
                        and abs(v) < radius - 0.05):
                    interior += weight
                if (face_skin + 0.12 <= u <= face_skin + cavity_depth - 0.12
                        and abs(v) <= radius + 1.0):
                    boundary_distances.append(abs(v))
                    if v >= 0.0:
                        positive_free_edges.append(v - half_path)
                    else:
                        negative_free_edges.append(v + half_path)
                if abs(u - first_wall_center) <= wall_band and abs(v) <= radius + 0.35:
                    wall_a += weight
                    wall_a_v.append(v)
                    wall_a_uv.append((u, v))
                    if abs(v) <= central_chord_half:
                        interface_cavity_edges.append(u + half_path)
                if abs(u - second_wall_center) <= wall_band and abs(v) <= radius + 0.35:
                    wall_b += weight
                    wall_b_v.append(v)
                    wall_b_uv.append((u, v))
                    if abs(v) <= central_chord_half:
                        inner_cavity_edges.append(u - half_path)
        span_a = max(wall_a_v) - min(wall_a_v) if wall_a_v else 0.0
        span_b = max(wall_b_v) - min(wall_b_v) if wall_b_v else 0.0
        contiguous_span_a = _longest_connected_v_span(wall_a_uv)
        contiguous_span_b = _longest_connected_v_span(wall_b_uv)
        free_transverse_diameter = (
            min(positive_free_edges) - max(negative_free_edges)
            if positive_free_edges and negative_free_edges else None)
        free_axial_slot = (
            min(inner_cavity_edges) - max(interface_cavity_edges)
            if interface_cavity_edges and inner_cavity_edges else None)
        loading_aperture = {
            "interior_extrusion_path_length_mm": interior,
            "free_transverse_diameter_mm": free_transverse_diameter,
            "free_axial_slot_width_mm": free_axial_slot,
            "central_chord_half_width_mm": central_chord_half,
        }
        retaining = {
            "kind": "two_axial_skin_paths",
            "interface_skin_path_length_mm": wall_a,
            "interface_skin_transverse_span_mm": span_a,
            "interface_skin_longest_contiguous_span_mm": contiguous_span_a,
            "inner_skin_path_length_mm": wall_b,
            "inner_skin_transverse_span_mm": span_b,
            "inner_skin_longest_contiguous_span_mm": contiguous_span_b,
            "connectivity_gap_limit_mm": RETAINING_PATH_CONNECTIVITY_GAP_MM,
            "pass": contiguous_span_a >= 3.0 and contiguous_span_b >= 3.0,
        }
    else:
        annulus_length = 0.0
        annulus_angles: list[float] = []
        radial_free_edges: list[float] = []
        for segment in layer.segments:
            if min(segment.x0, segment.x1) > cx + roi or max(segment.x0, segment.x1) < cx - roi:
                continue
            if min(segment.y0, segment.y1) > cy + roi or max(segment.y0, segment.y1) < cy - roi:
                continue
            local_segments.append(segment)
            path_width = (
                segment.line_width
                if segment.line_width is not None and segment.line_width > 0.0
                else FALLBACK_LINE_WIDTH_MM)
            half_path = path_width / 2.0
            for x, y, weight in _sample_segment(segment):
                radial = math.hypot(x - cx, y - cy)
                if radial < radius - 0.05:
                    interior += weight
                if radial <= radius + 1.0:
                    boundary_distances.append(radial)
                    radial_free_edges.append(radial - half_path)
                if radius - 0.20 <= radial <= radius + 0.80:
                    annulus_length += weight
                    annulus_angles.append(math.atan2(y - cy, x - cx))
        # A complete circumference is not required to be one continuous G-code
        # segment, but adjacent segments must cover the complete circumference;
        # summing unrelated fragments is not evidence of a printable cradle.
        largest_gap = _largest_circular_sample_gap(annulus_angles, radius)
        free_radial_diameter = (
            2.0 * min(radial_free_edges) if radial_free_edges else None)
        loading_aperture = {
            "interior_extrusion_path_length_mm": interior,
            "free_radial_diameter_mm": free_radial_diameter,
            "free_axial_slot_width_mm": None,
        }
        retaining = {
            "kind": "annular_open_cavity_path",
            "annular_path_length_mm": annulus_length,
            "sample_count": len(annulus_angles),
            "largest_uncovered_arc_mm": (
                largest_gap if math.isfinite(largest_gap) else None),
            "connectivity_gap_limit_mm": RETAINING_PATH_CONNECTIVITY_GAP_MM,
            "pass": (
                annulus_length >= math.pi * radius
                and largest_gap <= RETAINING_PATH_CONNECTIVITY_GAP_MM
            ),
        }
    return {
        "z_mm": layer.z,
        "gcode_line_number": layer.line_number,
        "local_extrusion_segment_count": len(local_segments),
        "roof_interior_path_length_mm": interior,
        "opening_half_width_path_mm": (
            min(boundary_distances) if boundary_distances else None),
        "loading_aperture": loading_aperture,
        "retaining_paths": retaining,
        "segments": local_segments,
    }


def _loading_aperture_pass(
    site: Mapping[str, Any], metrics: Mapping[str, Any],
) -> tuple[bool, str]:
    """Prove the nominal D5x2 disc can enter on the last-open layer."""
    aperture = metrics["loading_aperture"]
    interior = aperture["interior_extrusion_path_length_mm"]
    if site["closure_kind"] == "transverse_gable_45deg":
        diameter = aperture["free_transverse_diameter_mm"]
        slot = aperture["free_axial_slot_width_mm"]
        diameter_pass = (
            diameter is not None
            and diameter >= site["magnet_diameter_mm"] - LAYER_EPS)
        slot_pass = (
            slot is not None
            and slot >= site["magnet_depth_mm"] - LAYER_EPS)
    else:
        diameter = aperture["free_radial_diameter_mm"]
        slot = None
        diameter_pass = (
            diameter is not None
            and diameter >= site["magnet_diameter_mm"] - LAYER_EPS)
        slot_pass = True
    interior_pass = interior <= LAST_OPEN_INTERIOR_PATH_LIMIT_MM + LAYER_EPS
    passed = interior_pass and diameter_pass and slot_pass
    return passed, (
        f"interior path={interior:.3f} mm "
        f"(limit {LAST_OPEN_INTERIOR_PATH_LIMIT_MM:.3f}); "
        f"free diameter={diameter}; free axial slot={slot}; "
        f"required D={site['magnet_diameter_mm']:.3f}, "
        f"depth={site['magnet_depth_mm']:.3f} mm")


def _roof_progression_pass(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[bool, str]:
    last_value = metrics["last_fully_open"]["roof_interior_path_length_mm"]
    first_value = metrics["first_closing_pause"]["roof_interior_path_length_mm"]
    sealed_value = metrics["fully_sealed"]["roof_interior_path_length_mm"]
    last_boundary = metrics["last_fully_open"]["opening_half_width_path_mm"]
    first_boundary = metrics["first_closing_pause"]["opening_half_width_path_mm"]
    sealed_boundary = metrics["fully_sealed"]["opening_half_width_path_mm"]
    # The first 0.16-mm 45-degree roof strip is narrower than a 0.42-mm
    # Classic wall.  Its centreline can therefore remain outside the nominal
    # cavity interior even though Preview has begun closing the roof.  The
    # robust sliced fact is that the nearest roof-boundary path moves inward
    # on the first-closing layer and continues inward by the sealed layer.
    boundary_pass = (
        last_boundary is not None and first_boundary is not None
        and sealed_boundary is not None
        and first_boundary <= last_boundary - 0.03
        and sealed_boundary <= first_boundary - 0.03)
    # Interior deposition is a useful secondary confirmation once the roof
    # has sealed; it is deliberately not required on the sub-line-width first
    # strip.
    sealed_pass = sealed_value >= max(last_value, first_value) + 0.03
    passed = boundary_pass and sealed_pass
    return passed, (
        f"boundary last={last_boundary}, first={first_boundary}, "
        f"sealed={sealed_boundary} mm; interior path last={last_value:.3f}, "
        f"first={first_value:.3f}, sealed={sealed_value:.3f} mm")


def _discover_actual_closure_layers(
    layers: Sequence[Layer],
    site: Mapping[str, Any],
    placement_xy: tuple[float, float],
) -> tuple[dict[str, Layer], dict[str, dict[str, Any]], dict[str, Any]]:
    """Discover the first roof-closing layer from sliced toolpaths.

    The CAD bury plane is a consistency datum, not the layer selector.  The
    selector first finds the widest, loadable chimney/cavity toolpath, then
    scans every following scheduled layer.  The first boundary contraction,
    new cavity-interior deposition, or loss of the loading aperture is the
    actual closing onset.  Every scheduled layer in the preceding loadable
    run must remain indistinguishable from the fully open reference.

    This deliberately fails closed if the closing signature is missing,
    ambiguous, or reopens.  The CAD bury plane is checked as a bounded
    consistency datum against the actual toolpath onset, but it never selects
    the pause layer: a sub-line-width roof strip may not acquire a printable
    Classic centreline until one or two scheduled layers after the exact CAD
    boundary.  A pause can therefore never be manufactured from nominal CAD Z
    alone.
    """
    if not layers:
        raise AuditError("cannot discover cavity closure without sliced layers")
    bury = float(site["cavity_bury_roof_start_print_z_mm"])
    apex = float(site["roof_apex_print_z_mm"])
    center_z = float(site["print_cavity_center_xyz_mm"][2])
    if site["closure_kind"] == "transverse_gable_45deg":
        bottom = center_z - site["cavity_diameter_mm"] / 2.0
    else:
        bottom = center_z - site["cavity_depth_mm"] / 2.0
    lowest = _layer_at_or_above(layers, bottom)
    sealed = _layer_above(layers, apex)
    scan_start = _layer_at_or_above(layers, center_z)
    scan_layers = [
        layer for layer in layers
        if scan_start.z - LAYER_EPS <= layer.z <= sealed.z + LAYER_EPS
    ]
    if len(scan_layers) < 3:
        raise AuditError(
            f"{site['name']}: too few actual layers to discover roof closure")

    entries: list[dict[str, Any]] = []
    for layer in scan_layers:
        metrics = _toolpath_metrics(layer, site, placement_xy)
        aperture_pass, aperture_detail = _loading_aperture_pass(site, metrics)
        entries.append({
            "layer": layer,
            "metrics": metrics,
            "aperture_pass": aperture_pass,
            "aperture_detail": aperture_detail,
            "boundary": metrics["opening_half_width_path_mm"],
            "interior": metrics["roof_interior_path_length_mm"],
        })

    loadable_boundaries = [
        float(entry["boundary"])
        for entry in entries
        if entry["aperture_pass"] and entry["boundary"] is not None
    ]
    if not loadable_boundaries:
        raise AuditError(
            f"{site['name']}: G-code has no fully loadable open-cavity layer")
    open_boundary = max(loadable_boundaries)
    full_open_indices = [
        index for index, entry in enumerate(entries)
        if (entry["aperture_pass"]
            and entry["boundary"] is not None
            and open_boundary - float(entry["boundary"])
            < CLOSING_BOUNDARY_INSET_MM - LAYER_EPS)
    ]
    if not full_open_indices:
        raise AuditError(
            f"{site['name']}: no stable fully open G-code boundary")
    first_open_index = full_open_indices[0]
    open_interior = min(
        float(entries[index]["interior"])
        for index in full_open_indices
    )

    def closing_reasons(entry: Mapping[str, Any]) -> list[str]:
        reasons = []
        boundary = entry["boundary"]
        if boundary is None:
            reasons.append("opening boundary disappeared")
        elif (open_boundary - float(boundary)
              >= CLOSING_BOUNDARY_INSET_MM - LAYER_EPS):
            reasons.append(
                f"boundary inset {open_boundary - float(boundary):.3f} mm")
        if (float(entry["interior"]) - open_interior
                >= CLOSING_BOUNDARY_INSET_MM - LAYER_EPS):
            reasons.append(
                "new cavity-interior extrusion "
                f"{float(entry['interior']) - open_interior:.3f} mm")
        # A missing sampled chord/slot is not itself evidence of obstruction:
        # seam placement can temporarily leave no segment crossing the exact
        # probe line even while the full D5 aperture remains unchanged.  Count
        # aperture loss only when the G-code supplies a finite, measured free
        # dimension that is actually below the nominal magnet envelope.
        aperture = entry["metrics"]["loading_aperture"]
        free_diameter = (
            aperture["free_transverse_diameter_mm"]
            if site["closure_kind"] == "transverse_gable_45deg"
            else aperture["free_radial_diameter_mm"])
        free_slot = (
            aperture["free_axial_slot_width_mm"]
            if site["closure_kind"] == "transverse_gable_45deg"
            else None)
        measured_aperture_blocked = (
            (free_diameter is not None
             and float(free_diameter)
             < site["magnet_diameter_mm"] - LAYER_EPS)
            or (free_slot is not None
                and float(free_slot)
                < site["magnet_depth_mm"] - LAYER_EPS))
        if measured_aperture_blocked:
            reasons.append("nominal D5 loading aperture is no longer open")
        return reasons

    closing_index: int | None = None
    onset_reasons: list[str] = []
    for index in range(first_open_index + 1, len(entries)):
        reasons = closing_reasons(entries[index])
        if reasons:
            closing_index = index
            onset_reasons = reasons
            break
    if closing_index is None:
        raise AuditError(
            f"{site['name']}: no roof-closing signature was found in G-code")
    if closing_index <= first_open_index:
        raise AuditError(
            f"{site['name']}: roof closure has no preceding fully open layer")

    prior_entries = entries[first_open_index:closing_index]
    prior_failures = []
    for entry in prior_entries:
        reasons = closing_reasons(entry)
        if reasons:
            prior_failures.append(
                f"Z={entry['layer'].z:.3f}: " + "; ".join(reasons))
    if prior_failures:
        raise AuditError(
            f"{site['name']}: earlier scheduled cavity layers already close: "
            + " | ".join(prior_failures))

    last_entry = entries[closing_index - 1]
    first_entry = entries[closing_index]
    if first_entry["boundary"] is None:
        raise AuditError(
            f"{site['name']}: first actual closing layer has no auditable "
            "roof boundary")
    cad_consistency_tolerance = max(
        float(site.get("classic_retaining_path_mm", FALLBACK_LINE_WIDTH_MM)),
        LAYER_EPS)
    cad_bury_directly_bracketed = (
        last_entry["layer"].z <= bury + LAYER_EPS
        and first_entry["layer"].z > bury + LAYER_EPS)
    if (first_entry["layer"].z <= bury + LAYER_EPS
            or abs(last_entry["layer"].z - bury)
            > cad_consistency_tolerance + LAYER_EPS
            or abs(first_entry["layer"].z - bury)
            > cad_consistency_tolerance + LAYER_EPS):
        raise AuditError(
            f"{site['name']}: actual G-code closing onset "
            f"{first_entry['layer'].z:.3f} mm must begin above, and remain "
            f"within one Classic path width of, CAD bury plane {bury:.3f} "
            f"mm (tolerance {cad_consistency_tolerance:.3f} mm); last actual "
            f"open layer is {last_entry['layer'].z:.3f} mm")

    # A qualified 45-degree roof never reopens.  Inspect every remaining
    # boundary through the first fully sealed evidence layer, not merely the
    # first and last snapshots.
    previous_boundary = float(first_entry["boundary"])
    for entry in entries[closing_index + 1:]:
        boundary = entry["boundary"]
        if boundary is None:
            raise AuditError(
                f"{site['name']}: roof boundary vanished at "
                f"Z={entry['layer'].z:.3f} mm")
        boundary = float(boundary)
        if boundary > (
                previous_boundary
                + CLOSING_BOUNDARY_REOPEN_TOLERANCE_MM + LAYER_EPS):
            raise AuditError(
                f"{site['name']}: roof reopens in G-code at "
                f"Z={entry['layer'].z:.3f} mm "
                f"({previous_boundary:.3f} -> {boundary:.3f} mm)")
        previous_boundary = boundary

    representative_entry = prior_entries[len(prior_entries) // 2]
    selected = {
        "lowest_open": lowest,
        "representative_open": representative_entry["layer"],
        "last_fully_open": last_entry["layer"],
        "first_closing_pause": first_entry["layer"],
        "fully_sealed": sealed,
    }
    metrics_by_z = {
        round(float(entry["layer"].z), 6): entry["metrics"]
        for entry in entries
    }
    selected_metrics = {
        key: metrics_by_z.get(
            round(float(layer.z), 6),
            _toolpath_metrics(layer, site, placement_xy),
        )
        for key, layer in selected.items()
    }
    discovery = {
        "method": "earliest_actual_gcode_roof_closing_signature",
        "open_reference_boundary_half_width_mm": open_boundary,
        "open_reference_interior_path_length_mm": open_interior,
        "boundary_inset_threshold_mm": CLOSING_BOUNDARY_INSET_MM,
        "examined_layer_z_mm": [entry["layer"].z for entry in entries],
        "proven_fully_open_layer_z_mm": [
            entry["layer"].z for entry in prior_entries],
        "all_prior_scheduled_open_layers_pass": True,
        "first_closing_layer_z_mm": first_entry["layer"].z,
        "first_closing_signature": onset_reasons,
        "cad_bury_plane_bracketed": cad_bury_directly_bracketed,
        "cad_bury_plane_consistent_with_toolpath": True,
        "cad_bury_plane_consistency_tolerance_mm": (
            cad_consistency_tolerance),
    }
    return selected, selected_metrics, discovery


def _color_for_feature(feature: str) -> str:
    value = feature.lower()
    if "outer wall" in value:
        return "#1769aa"
    if "inner wall" in value:
        return "#00a878"
    if "bridge" in value:
        return "#e4572e"
    if "infill" in value:
        return "#775da6"
    if "gap" in value:
        return "#f3a712"
    return "#66717e"


def _render_evidence_svg(
    path: Path,
    artifact: Mapping[str, Any],
    site_records: Sequence[Mapping[str, Any]],
    placement_xy: tuple[float, float],
) -> None:
    stages = (
        "lowest_open", "representative_open", "last_fully_open",
        "first_closing_pause", "fully_sealed")
    width = EVIDENCE_CELL_PX * len(stages)
    header = 74
    height = header + EVIDENCE_CELL_PX * len(site_records)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="12" y="25" font-family="sans-serif" font-size="17" font-weight="bold">{html.escape(artifact["id"])} — Bambu P2S 0.4 / 0.16 Classic</text>',
        '<text x="12" y="49" font-family="sans-serif" font-size="12">front face down · local sliced G-code toolpaths · red box = nominal cavity plan</text>',
    ]
    for column, stage in enumerate(stages):
        x = column * EVIDENCE_CELL_PX + EVIDENCE_CELL_PX / 2
        elements.append(
            f'<text x="{x:.1f}" y="69" text-anchor="middle" font-family="sans-serif" font-size="11">{stage.replace("_", " ")}</text>')
    for row, record in enumerate(site_records):
        site = record["site"]
        cx, cy, _ = site["print_cavity_center_xyz_mm"]
        cx += placement_xy[0]
        cy += placement_xy[1]
        radius = site["cavity_diameter_mm"] / 2.0
        world_half = radius + EVIDENCE_MARGIN_MM
        scale = (EVIDENCE_CELL_PX - 28.0) / (2.0 * world_half)
        y0 = header + row * EVIDENCE_CELL_PX
        for column, stage in enumerate(stages):
            x0 = column * EVIDENCE_CELL_PX
            metrics = record["layer_metrics"][stage]
            elements.append(
                f'<rect x="{x0 + 1}" y="{y0 + 1}" width="{EVIDENCE_CELL_PX - 2}" height="{EVIDENCE_CELL_PX - 2}" fill="#fbfcfd" stroke="#d7dde4"/>')
            def map_xy(x: float, y: float) -> tuple[float, float]:
                return (
                    x0 + EVIDENCE_CELL_PX / 2 + (x - cx) * scale,
                    y0 + EVIDENCE_CELL_PX / 2 - (y - cy) * scale,
                )
            for segment in metrics.pop("segments"):
                sx0, sy0 = map_xy(segment.x0, segment.y0)
                sx1, sy1 = map_xy(segment.x1, segment.y1)
                stroke = _color_for_feature(segment.feature)
                width_px = max(0.75, (segment.line_width or 0.42) * scale * 0.55)
                elements.append(
                    f'<line x1="{sx0:.2f}" y1="{sy0:.2f}" x2="{sx1:.2f}" y2="{sy1:.2f}" stroke="{stroke}" stroke-width="{width_px:.2f}" stroke-linecap="round"/>')
            # Draw nominal open cavity projection in the print plane.
            if site["closure_kind"] == "transverse_gable_45deg":
                fx, fy, _ = site["print_actual_face_xyz_mm"]
                fx += placement_xy[0]
                fy += placement_xy[1]
                ux, uy = _unit_xy(site["print_material_inward_xyz"], "material inward")
                vx, vy = -uy, ux
                u0 = site["face_skin_mm"]
                u1 = u0 + site["cavity_depth_mm"]
                corners = []
                for u, v in ((u0, -radius), (u1, -radius),
                             (u1, radius), (u0, radius)):
                    corners.append(map_xy(fx + u * ux + v * vx,
                                          fy + u * uy + v * vy))
                points = " ".join(f"{x:.2f},{y:.2f}" for x, y in corners)
                elements.append(
                    f'<polygon points="{points}" fill="none" stroke="#d62828" stroke-width="1.2" stroke-dasharray="4 3"/>')
            else:
                pcx, pcy = map_xy(cx, cy)
                elements.append(
                    f'<circle cx="{pcx:.2f}" cy="{pcy:.2f}" r="{radius * scale:.2f}" fill="none" stroke="#d62828" stroke-width="1.2" stroke-dasharray="4 3"/>')
            elements.append(
                f'<text x="{x0 + 8}" y="{y0 + 17}" font-family="monospace" font-size="10">{html.escape(site["name"])} Z={metrics["z_mm"]:.2f}</text>')
            elements.append(
                f'<text x="{x0 + 8}" y="{y0 + EVIDENCE_CELL_PX - 7}" font-family="monospace" font-size="9">roof interior={metrics["roof_interior_path_length_mm"]:.2f} mm</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _svg_to_png(svg: Path, png: Path) -> dict[str, Any]:
    if not svg.is_file() or svg.stat().st_size == 0:
        raise AuditError(f"SVG evidence is missing or empty: {svg}")
    # Never accept a renderer's stale output from a prior audit.
    png.unlink(missing_ok=True)
    commands = []
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        commands.append([rsvg, "-o", str(png), str(svg)])
    magick = shutil.which("magick")
    if magick:
        commands.append([magick, str(svg), str(png)])
    convert = shutil.which("convert")
    if convert:
        commands.append([convert, str(svg), str(png)])
    errors = []
    for command in commands:
        png.unlink(missing_ok=True)
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        if run.returncode == 0 and png.is_file() and png.stat().st_size:
            return {"path": str(png), "sha256": sha256_file(png),
                    "renderer": command[0]}
        errors.append(f"{' '.join(command)}: {run.stdout[-1000:]}")
    png.unlink(missing_ok=True)
    detail = "; ".join(errors or ["no SVG-to-PNG renderer found"])
    raise AuditError(f"fresh PNG evidence could not be rendered: {detail}")


def _gcode_tool_path() -> Path | None:
    candidates = sorted((Path.home() / ".codex" / "plugins" / "cache"
                         / "text-to-cad" / "cad").glob(
        "*/skills/gcode/scripts/gcode_tool.py"), reverse=True)
    return candidates[0] if candidates else None


def _validate_with_gcode_skill(
    gcode: Path,
    out_dir: Path,
    profile_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    tool = _gcode_tool_path()
    if tool is None:
        return {"ok": None, "reason": "gcode skill validator not installed"}
    effective = profile_bundle["identity"]["effective"]
    bounds = profile_bundle["identity"]["machine_bounds_mm"]
    filament = profile_bundle["resolved"]["filament"]
    nozzle_temp = _scalar(filament, "nozzle_temperature", "filament")
    bed_temp = _scalar(filament, "eng_plate_temp", "filament")
    wrapper = {
        # The skill validator's schema currently enumerates Orca/Prusa/Cura.
        # This value is only a validation-schema compatibility field: actual
        # slicing provenance remains BambuStudio in every manifest record.
        "backend": "orcaslicer",
        "native_config": str(profile_bundle["paths"]["machine"]),
        "machine": {
            "name": effective["printer_model"],
            "bed_size_mm": [bounds["x"][1], bounds["y"][1]],
            "z_height_mm": bounds["z"][1],
            "motion_bounds_mm": bounds,
        },
        "filament": {
            "type": effective["filament"],
            "nozzle_temp_c": nozzle_temp,
            "bed_temp_c": bed_temp,
        },
    }
    wrapper_path = out_dir / "gcode_validation_profile.json"
    _write_json(wrapper_path, wrapper)
    run = subprocess.run(
        [sys.executable, str(tool), "validate", "--gcode", str(gcode),
         "--profile", str(wrapper_path), "--json"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False)
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError:
        result = {"ok": False, "returncode": run.returncode,
                  "raw_output": run.stdout[-4000:]}
    _write_json(out_dir / "gcode_skill_validation.json", result)
    return result


def _cached_slice_matches(
    prior: Mapping[str, Any], *, fingerprint: str, stl: Path,
    gcode: Path, result_path: Path, project_3mf: Path,
) -> bool:
    """Reuse only hash-bound slicer outputs, never merely their input key."""
    required = {
        "fingerprint": fingerprint,
        "stl_sha256": sha256_file(stl),
        "gcode_sha256": sha256_file(gcode),
        "result_sha256": sha256_file(result_path),
        "project_3mf_sha256": sha256_file(project_3mf),
    }
    return all(prior.get(key) == value for key, value in required.items())


def _source_hashes(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.is_file():
            records.append({"path": str(path), "sha256": sha256_file(path)})
        else:
            records.append({"path": str(path), "sha256": None,
                            "error": "missing source"})
    return records


def _artifact_fingerprint(
    artifact: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
    catalog_sha: str,
) -> str:
    stl = artifact["stl"]
    payload = {
        "catalog_sha256": catalog_sha,
        "catalog_source_revision": artifact["catalog_source_revision"],
        "catalog_record": artifact["catalog_record"],
        "stl_sha256": sha256_file(stl),
        "print_sidecar_sha256": sha256_file(artifact["print_sidecar"]),
        "source_file_sha256": sorted(
            artifact["catalog_record"]["source_file_sha256"].items()),
        "transaction_manifest_sha256": artifact.get(
            "transaction_manifest_sha256"),
        "facts_sha256": artifact.get("facts_sha256"),
        "stage_manifest_sha256": artifact.get("stage_manifest_sha256"),
        "profile_set_sha256": profile_bundle["identity"]["profile_set_sha256"],
        "bambu_binary_sha256": profile_bundle["identity"]["binary_sha256"],
        "audit_source_sha256": sorted(
            profile_bundle["audit_source_sha256"].items()),
    }
    return _sha256_bytes(_canonical_json(payload))


def _bambu_command(
    bambu: Path,
    stl: Path,
    output: Path,
    profile_bundle: Mapping[str, Any],
) -> list[str]:
    settings = ";".join(str(profile_bundle["paths"][key])
                        for key in ("machine", "process"))
    return [
        str(bambu), "--debug", "2", "--slice", "0", "--arrange", "1",
        "--orient", "0", "--export-3mf", PLACED_3MF_FILENAME,
        "--load-settings", settings,
        "--load-filaments", str(profile_bundle["paths"]["filament"]),
        "--outputdir", str(output), str(stl),
    ]


def _slice_one(
    artifact: Mapping[str, Any],
    *,
    output_root: Path,
    profile_bundle: Mapping[str, Any],
    bambu: Path,
    catalog_sha: str,
    reuse: bool,
    dry_run: bool,
) -> dict[str, Any]:
    stl: Path = artifact["stl"]
    release_stl: Path = artifact.get("release_stl", stl)
    _validate_artifact_bindings(artifact)
    mesh = inspect_stl(stl)
    if abs(mesh.bounds_min[2]) > 0.02:
        raise AuditError(
            f"{artifact['id']}: front-down STL must sit at Z=0; "
            f"min Z={mesh.bounds_min[2]:.4f}")
    bounds = profile_bundle["identity"]["machine_bounds_mm"]
    if mesh.size[0] > bounds["x"][1] - bounds["x"][0] + 1e-4 \
            or mesh.size[1] > bounds["y"][1] - bounds["y"][0] + 1e-4 \
            or mesh.size[2] > bounds["z"][1] - bounds["z"][0] + 1e-4:
        raise AuditError(
            f"{artifact['id']}: {mesh.size} mm exceeds P2S 256-mm envelope")
    slug = _slug(artifact["id"])
    out_dir = output_root / "slices" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _artifact_fingerprint(artifact, profile_bundle, catalog_sha)
    fingerprint_path = out_dir / "slice_fingerprint.json"
    gcode = out_dir / "plate_1.gcode"
    result_path = out_dir / "result.json"
    project_3mf = out_dir / PLACED_3MF_FILENAME
    reused = False
    if (reuse and fingerprint_path.is_file() and gcode.is_file()
            and result_path.is_file() and project_3mf.is_file()):
        prior = _load_json(fingerprint_path)
        if isinstance(prior, dict) and _cached_slice_matches(
                prior, fingerprint=fingerprint, stl=stl,
                gcode=gcode, result_path=result_path,
                project_3mf=project_3mf):
            reused = True
    command = _bambu_command(bambu, stl, out_dir, profile_bundle)
    if dry_run:
        return {"id": artifact["id"], "dry_run": True, "command": command,
                "fingerprint": fingerprint}
    if not reused:
        for stale in (gcode, result_path, project_3mf):
            stale.unlink(missing_ok=True)
        run = subprocess.run(
            command, cwd=out_dir, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(profile_bundle["config"]["slicing"]["timeout_seconds"]),
            check=False,
            env={**os.environ, "LC_ALL": "C"})
        (out_dir / "bambu_studio.log").write_text(
            run.stdout, encoding="utf-8", errors="replace")
        if run.returncode != 0:
            raise AuditError(
                f"{artifact['id']}: Bambu Studio exited {run.returncode}; "
                f"see {out_dir / 'bambu_studio.log'}")
        if (not gcode.is_file() or not result_path.is_file()
                or not project_3mf.is_file()):
            raise AuditError(
                f"{artifact['id']}: Bambu Studio did not create "
                "plate_1.gcode/result.json/audited 3MF")
        _write_json(fingerprint_path, {
            "fingerprint": fingerprint,
            "command": command,
            "stl_sha256": sha256_file(stl),
            "gcode_sha256": sha256_file(gcode),
            "result_sha256": sha256_file(result_path),
            "project_3mf_sha256": sha256_file(project_3mf),
        })
    result_json = _load_json(result_path)
    if result_json.get("return_code") != 0:
        raise AuditError(f"{artifact['id']}: slicer result is not Success: {result_json}")
    plates = result_json.get("sliced_plates")
    if not isinstance(plates, list) or len(plates) != 1:
        raise AuditError(f"{artifact['id']}: expected exactly one sliced plate")
    objects = plates[0].get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise AuditError(f"{artifact['id']}: expected exactly one sliced object")
    if int(plates[0].get("triangle_count", -1)) != mesh.triangle_count:
        raise AuditError(
            f"{artifact['id']}: sliced plate triangle count differs from "
            f"staged STL ({plates[0].get('triangle_count')} != "
            f"{mesh.triangle_count})")
    if int(objects[0].get("triangle_count", -1)) != mesh.triangle_count:
        raise AuditError(
            f"{artifact['id']}: sliced object triangle count differs from "
            f"staged STL ({objects[0].get('triangle_count')} != "
            f"{mesh.triangle_count})")
    bbox = objects[0].get("bbox")
    if not isinstance(bbox, dict):
        raise AuditError(f"{artifact['id']}: missing Bambu object bbox")
    try:
        project_audit = audit_bambu_3mf(project_3mf, stl)
        expected_bbox = validate_bambu_result_bbox(
            bbox, project_audit.source_bounds,
            project_audit.stl_to_bed_matrix)
        bed_clearances = validate_bambu_bed_fit(
            project_audit.transformed_actual_mesh_bounds, bounds)
    except Bambu3MFAuditError as exc:
        raise AuditError(
            f"{artifact['id']}: Bambu 3MF placement/mesh audit failed: "
            f"{exc}") from exc
    if project_audit.triangle_count != mesh.triangle_count:
        raise AuditError(
            f"{artifact['id']}: archived 3MF triangle count differs from "
            "staged STL")
    slicer_sites = [
        _site_in_bambu_bed_space(site, project_audit.stl_to_bed_matrix)
        for site in artifact["sites"]
    ]
    # All local geometry now carries Bambu's full audited Rz+XY transform.
    # The legacy additive placement parameter remains zero so the cavity
    # toolpath routines can stay independent of the 3MF parser.
    placement = (0.0, 0.0)
    parsed = parse_gcode(
        gcode,
        retain_regions=_cavity_retain_regions(slicer_sites, placement),
    )
    site_records = []
    errors = _validate_actual_gcode_profile(parsed, profile_bundle)
    for site in slicer_sites:
        selected, metrics, closure_discovery = (
            _discover_actual_closure_layers(parsed.layers, site, placement))
        roof_pass, roof_detail = _roof_progression_pass(metrics)
        retaining_stage_pass = {
            stage: _retaining_stage_pass(site, stage, metrics[stage])
            for stage in (
                "lowest_open", "representative_open", "last_fully_open")
        }
        retaining_pass = all(retaining_stage_pass.values())
        aperture_pass, aperture_detail = _loading_aperture_pass(
            site, metrics["last_fully_open"])
        magnet_bottom_z, magnet_top_z = _seated_magnet_print_z_bounds(site)
        seated_below_last_open = (
            selected["last_fully_open"].z - magnet_top_z)
        seated_below_first_closing = (
            selected["first_closing_pause"].z - magnet_top_z)
        seated_clearance_pass = (
            seated_below_last_open >= -LAYER_EPS
            and seated_below_first_closing >= 0.02 - LAYER_EPS
        )
        diametric_clearance = (
            site["cavity_diameter_mm"] - site["magnet_diameter_mm"])
        axial_clearance = (
            site["cavity_depth_mm"] - site["magnet_depth_mm"])
        insertion_fit_pass = (
            diametric_clearance >= 0.19
            and axial_clearance >= 0.09
        )
        expected = site.get("expected_pause_marker_z_mm")
        actual_pause = selected["first_closing_pause"].z
        regression_pass = expected is None or math.isclose(
            actual_pause, expected, abs_tol=0.001)
        if not roof_pass:
            errors.append(f"{site['name']}: no first-closing roof progression ({roof_detail})")
        if not retaining_pass:
            failed_stages = [
                stage for stage, passed in retaining_stage_pass.items()
                if not passed
            ]
            errors.append(
                f"{site['name']}: retaining paths missing at open stage(s) "
                + ", ".join(failed_stages))
        if not aperture_pass:
            errors.append(
                f"{site['name']}: last-open loading aperture rejects the "
                f"nominal magnet ({aperture_detail})")
        if not seated_clearance_pass:
            errors.append(
                f"{site['name']}: fully seated magnet top Z "
                f"{magnet_top_z:.3f} is not below the completed last-open "
                f"layer Z {selected['last_fully_open'].z:.3f} and clear of "
                f"first-closing Z {actual_pause:.3f}")
        if not insertion_fit_pass:
            errors.append(
                f"{site['name']}: nominal magnet cannot be dropped/seated; "
                f"diametric clearance={diametric_clearance:.3f} mm, "
                f"axial clearance={axial_clearance:.3f} mm")
        if not regression_pass:
            errors.append(
                f"{site['name']}: pause regression {actual_pause:.2f} != {expected:.2f}")
        stage_records = {}
        for key, layer in selected.items():
            clean_metrics = {k: v for k, v in metrics[key].items() if k != "segments"}
            stage_records[key] = clean_metrics
        site_records.append({
            "site": site,
            "actual": {
                "lowest_open_layer_z_mm": selected["lowest_open"].z,
                "representative_open_layer_z_mm": selected["representative_open"].z,
                "last_completely_open_layer_z_mm": selected["last_fully_open"].z,
                "cavity_bury_roof_start_plane_z_mm": site[
                    "cavity_bury_roof_start_print_z_mm"],
                "first_closing_layer_z_mm": actual_pause,
                "bambu_studio_pause_marker_z_mm": actual_pause,
                "fully_sealed_inspection_layer_z_mm": selected["fully_sealed"].z,
            },
            "layer_metrics": metrics,
            "layer_evidence": stage_records,
            "roof_progression_pass": roof_pass,
            "roof_progression_detail": roof_detail,
            "retaining_paths_pass": retaining_pass,
            "retaining_paths_stage_pass": retaining_stage_pass,
            "loading_aperture_pass": aperture_pass,
            "loading_aperture_detail": aperture_detail,
            "seated_magnet": {
                "print_center_xyz_mm": list(
                    site["print_seated_magnet_center_xyz_mm"]),
                "print_bottom_z_mm": magnet_bottom_z,
                "print_top_z_mm": magnet_top_z,
                "below_last_open_layer_mm": seated_below_last_open,
                "below_first_closing_layer_mm": seated_below_first_closing,
                "clearance_pass": seated_clearance_pass,
            },
            "insertion_fit": {
                "diametric_clearance_mm": diametric_clearance,
                "axial_clearance_mm": axial_clearance,
                "pass": insertion_fit_pass,
            },
            "regression_expected_z_mm": expected,
            "regression_pass": regression_pass,
            "closure_discovery": closure_discovery,
        })
    evidence_svg = out_dir / "captive_toolpath_evidence.svg"
    evidence_svg.unlink(missing_ok=True)
    evidence_artifact = dict(artifact)
    evidence_artifact["sites"] = slicer_sites
    _render_evidence_svg(
        evidence_svg, evidence_artifact, site_records, placement)
    if not evidence_svg.is_file() or evidence_svg.stat().st_size == 0:
        raise AuditError(
            f"{artifact['id']}: fresh SVG evidence was not created")
    evidence_png = out_dir / "captive_toolpath_evidence.png"
    png_record = _svg_to_png(evidence_svg, evidence_png)
    # Renderer consumes/removes segment objects from metrics.  Only serializable
    # stage evidence remains in the final record.
    for record in site_records:
        record.pop("layer_metrics", None)
    skill_validation = _validate_with_gcode_skill(
        gcode, out_dir, profile_bundle)
    if skill_validation.get("ok") is not True:
        errors.append(
            "plain G-code static validation did not return an explicit pass")
    sources = _source_hashes(artifact.get(
        "release_source_files", artifact["source_files"]))
    missing_sources = [item["path"] for item in sources if item["sha256"] is None]
    if missing_sources:
        errors.append("missing source files: " + ", ".join(missing_sources))
    project_audit_record = project_audit.as_record()
    project_audit_record.pop("staged_stl", None)
    project_audit_record["audited_release_stl"] = str(release_stl)
    project_audit_record["audited_stl_sha256"] = sha256_file(stl)
    record = {
        "id": artifact["id"],
        "state": artifact["state"],
        "variant": artifact["variant"],
        "part": artifact["part"],
        "print_orientation": artifact["print_orientation"],
        "audit_mode": "actual_p2s_slice",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "reused_slice": reused,
        "command": command,
        "fingerprint": fingerprint,
        "input": {
            "stl": str(release_stl),
            "stl_sha256": sha256_file(stl),
            "sliced_from_immutable_stage": (
                release_stl.resolve() != stl.resolve()),
            "triangle_count": mesh.triangle_count,
            "bounds_min_mm": mesh.bounds_min,
            "bounds_max_mm": mesh.bounds_max,
            "size_mm": mesh.size,
            "source_files": sources,
        },
        "slicer": {
            "result_json": str(result_path),
            "result_sha256": sha256_file(result_path),
            "project_3mf": str(project_3mf),
            "project_3mf_sha256": sha256_file(project_3mf),
            "gcode": str(gcode),
            "gcode_sha256": sha256_file(gcode),
            "bambu_3mf_audit": project_audit_record,
            "bambu_expected_result_bbox": expected_bbox,
            "actual_mesh_bed_clearance_mm": {
                axis: list(values) for axis, values in bed_clearances.items()
            },
            "sliced_bbox": bbox,
            "layer_count": len(parsed.layers),
            "movement_commands": parsed.movement_commands,
            "arc_commands": parsed.arc_commands,
            "extrusion_moves": parsed.extrusion_moves,
            "temperature_commands": parsed.temperature_commands,
            "gcode_bounds_min_mm": parsed.bounds_min,
            "gcode_bounds_max_mm": parsed.bounds_max,
            "effective_config": {
                key: parsed.config.get(key) for key in (
                    "layer_height", "initial_layer_print_height",
                    "wall_generator", "outer_wall_line_width",
                    "inner_wall_line_width", "enable_support",
                    "enable_arc_fitting")
            },
            "gcode_skill_validation": skill_validation,
        },
        "sites": site_records,
        "evidence": {
            "svg": str(evidence_svg),
            "svg_sha256": sha256_file(evidence_svg),
            "png": png_record,
        },
    }
    _write_json(out_dir / "captive_magnet_slice_audit.json", record)
    return record


def _oversize_proxy_coverage_record(
    artifact: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
    profile_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Cover an unprintable monolith only through exact passing split sites."""
    stl: Path = artifact["stl"]
    release_stl: Path = artifact.get("release_stl", stl)
    errors: list[str] = []
    _validate_artifact_bindings(artifact)
    mesh = inspect_stl(stl)
    bounds = profile_bundle["identity"]["machine_bounds_mm"]
    limits = tuple(
        bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z"))
    exceeds = tuple(
        size > limit + 1.0e-4 for size, limit in zip(mesh.size, limits))
    if not (exceeds[0] or exceeds[1]):
        errors.append(
            "catalog declares this artifact P2S-oversize, but its front-down "
            f"XY footprint {mesh.size[:2]} fits {limits[:2]}; contract is stale")
    if exceeds[2]:
        errors.append(
            f"artifact also exceeds P2S Z: {mesh.size[2]:.3f} > "
            f"{limits[2]:.3f} mm")
    coverage = []
    for proxy in artifact["cavity_audit_proxies"]:
        proxy_record = records_by_id.get(proxy["artifact_id"])
        if proxy_record is None:
            errors.append(
                f"{proxy['site']}: proxy artifact was not successfully sliced: "
                f"{proxy['artifact_id']}")
            continue
        if (proxy_record.get("audit_mode") != "actual_p2s_slice"
                or proxy_record.get("status") != "pass"):
            errors.append(
                f"{proxy['site']}: proxy did not pass a normal P2S slice: "
                f"{proxy['artifact_id']} status={proxy_record.get('status')}")
            continue
        matches = [
            record for record in proxy_record.get("sites", ())
            if record.get("site", {}).get("name") == proxy["proxy_site"]
        ]
        if len(matches) != 1:
            errors.append(
                f"{proxy['site']}: proxy audit site is missing or ambiguous: "
                f"{proxy['artifact_id']}/{proxy['proxy_site']}")
            continue
        site_record = matches[0]
        if site_record["site"].get("source_contract_sha256") != proxy[
                "source_contract_sha256"]:
            errors.append(
                f"{proxy['site']}: sliced proxy source contract hash drifted")
            continue
        if not all((
                site_record.get("retaining_paths_pass") is True,
                site_record.get("loading_aperture_pass") is True,
                site_record.get("seated_magnet", {}).get(
                    "clearance_pass") is True,
                site_record.get("insertion_fit", {}).get("pass") is True,
                site_record.get("regression_pass") is True,
        )):
            errors.append(
                f"{proxy['site']}: proxy site did not pass every cavity gate")
            continue
        evidence = proxy_record.get("evidence", {})
        png = evidence.get("png", {})
        coverage.append({
            "site": proxy["site"],
            "source_contract_sha256": proxy["source_contract_sha256"],
            "proxy_artifact_id": proxy["artifact_id"],
            "proxy_site": proxy["proxy_site"],
            "proxy_stl_sha256": proxy_record["input"]["stl_sha256"],
            "proxy_gcode_sha256": proxy_record["slicer"]["gcode_sha256"],
            "proxy_site_audit_sha256": _sha256_bytes(
                _canonical_json(site_record)),
            "proxy_evidence_svg_sha256": evidence.get("svg_sha256"),
            "proxy_evidence_png_sha256": png.get("sha256"),
            "pause_marker_z_mm": site_record["actual"][
                "bambu_studio_pause_marker_z_mm"],
        })
    expected_sites = {site["name"] for site in artifact["sites"]}
    covered_sites = {item["site"] for item in coverage}
    if covered_sites != expected_sites:
        errors.append(
            "exact split coverage is incomplete; missing="
            + ",".join(sorted(expected_sites - covered_sites)))
    sources = _source_hashes(artifact.get(
        "release_source_files", artifact["source_files"]))
    missing_sources = [item["path"] for item in sources
                       if item["sha256"] is None]
    if missing_sources:
        errors.append("missing source files: " + ", ".join(missing_sources))
    status = OVERSIZE_COVERED_STATUS if not errors else "fail"
    return {
        "id": artifact["id"],
        "state": artifact["state"],
        "variant": artifact["variant"],
        "part": artifact["part"],
        "print_orientation": artifact["print_orientation"],
        "audit_mode": "exact_split_proxy_coverage",
        "status": status,
        "errors": errors,
        "p2s_printable": False,
        "input": {
            "stl": str(release_stl),
            "stl_sha256": sha256_file(stl),
            "sliced_from_immutable_stage": (
                release_stl.resolve() != stl.resolve()),
            "triangle_count": mesh.triangle_count,
            "bounds_min_mm": mesh.bounds_min,
            "bounds_max_mm": mesh.bounds_max,
            "size_mm": mesh.size,
            "source_files": sources,
        },
        "p2s_bed_fit": {
            "pass": False,
            "machine_limits_mm": limits,
            "exceeds_axes": {
                axis: value for axis, value in zip(("x", "y", "z"), exceeds)
            },
            "policy": (
                "no virtual bed, scaling, tilting, clipping, G-code, or fake "
                "pause group for this canonical monolith"),
        },
        "source_site_contracts": [
            {
                "site": site["name"],
                "sha256": site["source_contract_sha256"],
            }
            for site in artifact["sites"]
        ],
        "cavity_audit_coverage": {
            "pass": not errors,
            "method": "exact_same_state_keyed_split_p2s_gcode",
            "sites": coverage,
        },
    }


def _pause_groups(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (record.get("audit_mode") != "actual_p2s_slice"
            or record.get("status") != "pass"):
        return []
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for site_record in record.get("sites", ()):
        z = float(site_record["actual"]["bambu_studio_pause_marker_z_mm"])
        grouped.setdefault(z, []).append(site_record)
    result = []
    for index, (z, sites) in enumerate(sorted(grouped.items()), 1):
        insertion_directions = [
            _vec3(_required(
                item["site"], "print_insertion_direction_xyz",
                f"{record.get('id', '<unknown>')}/"
                f"{item['site'].get('name', '<unnamed>')}: print insertion "
                "direction"), "print insertion direction")
            for item in sites
        ]
        if any(any(not math.isclose(
                actual, expected, abs_tol=1.0e-9, rel_tol=0.0)
                for actual, expected in zip(
                    direction, PRINT_INSERTION_DIRECTION_XYZ, strict=True))
                for direction in insertion_directions):
            raise AuditError(
                f"{record.get('id', '<unknown>')}: pause group {index} has "
                f"an unsafe insertion direction: {insertion_directions}")
        result.append({
            "group": index,
            "pause_marker_z_mm": z,
            "sites": [item["site"]["name"] for item in sites],
            "magnet_count": len(sites),
            "last_completely_open_layer_z_mm": max(
                item["actual"]["last_completely_open_layer_z_mm"] for item in sites),
            "cavity_bury_roof_start_plane_z_mm": sorted({
                item["actual"]["cavity_bury_roof_start_plane_z_mm"] for item in sites}),
            "first_closing_layer_z_mm": z,
            "print_insertion_direction_xyz": list(
                PRINT_INSERTION_DIRECTION_XYZ),
            "insertion_instruction": PRINT_INSERTION_INSTRUCTION,
            "minimum_seated_below_last_open_layer_mm": min(
                item["seated_magnet"]["below_last_open_layer_mm"]
                for item in sites),
            "minimum_seated_below_first_closing_layer_mm": min(
                item["seated_magnet"]["below_first_closing_layer_mm"]
                for item in sites),
            "polarity": [{
                "site": item["site"]["name"],
                "print_marked_pole_axis_xyz": item["site"]["print_marked_pole_axis_xyz"],
                "installed_marked_pole_axis_xyz": item["site"].get(
                    "installed_marked_pole_axis_xyz"),
                "instruction": item["site"]["polarity_instruction"],
            } for item in sites],
        })
    return result


def _require_record_file(
    path_value: Any, digest_value: Any, label: str,
) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise AuditError(f"{label} path is missing")
    if (not isinstance(digest_value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest_value)):
        raise AuditError(f"{label} SHA-256 is missing or invalid")
    path = Path(path_value)
    if not path.is_file() or path.stat().st_size == 0:
        raise AuditError(f"{label} file is missing or empty: {path}")
    if sha256_file(path) != digest_value:
        raise AuditError(f"{label} file differs from its recorded SHA-256")
    return path


def _validate_complete_release(
    catalog: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]] = (),
    *,
    enforce_expected_inventory: bool = True,
) -> None:
    """Require exact, passing, hash-backed coverage before publication."""
    if failures:
        raise AuditError(
            f"canonical manifest publication blocked by {len(failures)} "
            "slice exception(s)")
    artifacts = catalog.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AuditError("canonical publication requires a normalized catalog")
    inventory = catalog.get("inventory", {})
    expected_artifact_count = len(artifacts)
    expected_magnet_count = sum(
        int(site["magnet_count"])
        for artifact in artifacts for site in artifact["sites"])
    if (inventory.get("artifact_count") != expected_artifact_count
            or inventory.get("magnet_count") != expected_magnet_count):
        raise AuditError("catalog inventory changed before publication")
    if enforce_expected_inventory and (
            expected_artifact_count != EXPECTED_RELEASE_ARTIFACT_COUNT
            or expected_magnet_count != EXPECTED_RELEASE_MAGNET_COUNT):
        raise AuditError(
            "canonical publication requires the exact production inventory "
            f"of {EXPECTED_RELEASE_ARTIFACT_COUNT} artifacts / "
            f"{EXPECTED_RELEASE_MAGNET_COUNT} stations")
    expected_by_id = {artifact["id"]: artifact for artifact in artifacts}
    if len(expected_by_id) != expected_artifact_count:
        raise AuditError("canonical catalog contains duplicate artifact ids")
    record_ids = [str(record.get("id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise AuditError("canonical slice records contain duplicate artifact ids")
    if set(record_ids) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(record_ids))
        unexpected = sorted(set(record_ids) - set(expected_by_id))
        raise AuditError(
            "canonical slice coverage is not exact; "
            f"missing={missing}, unexpected={unexpected}")

    covered_station_count = 0
    pause_station_count = 0
    for record in records:
        artifact = expected_by_id[record["id"]]
        if record.get("input", {}).get(
                "stl_sha256") != artifact["stl_catalog_sha256"]:
            raise AuditError(
                f"{record['id']}: sliced STL hash differs from catalog")
        expected_source_hashes = {
            str(path): digest
            for path, digest in artifact["source_file_sha256"].items()
        }
        actual_source_hashes = {
            item.get("path"): item.get("sha256")
            for item in record.get("input", {}).get("source_files", ())
            if isinstance(item, Mapping)
        }
        if actual_source_hashes != expected_source_hashes:
            raise AuditError(
                f"{record['id']}: source-file evidence differs from catalog")
        expected_sites = {site["name"] for site in artifact["sites"]}
        if artifact.get("p2s_printability") == "not_printable_oversize":
            if (record.get("audit_mode") != "exact_split_proxy_coverage"
                    or record.get("status") != OVERSIZE_COVERED_STATUS
                    or record.get("cavity_audit_coverage", {}).get(
                        "pass") is not True):
                raise AuditError(
                    f"{record['id']}: oversized monolith lacks passing exact "
                    "split coverage")
            covered = {
                item.get("site")
                for item in record.get(
                    "cavity_audit_coverage", {}).get("sites", ())
                if isinstance(item, Mapping)
            }
            if covered != expected_sites:
                raise AuditError(
                    f"{record['id']}: oversized station coverage is incomplete")
            if _pause_groups(record):
                raise AuditError(
                    f"{record['id']}: oversized monolith emitted a fake pause")
            covered_station_count += len(covered)
            continue

        if (record.get("audit_mode") != "actual_p2s_slice"
                or record.get("status") != "pass"):
            raise AuditError(
                f"{record['id']}: actual P2S slice did not pass")
        site_records = record.get("sites")
        if not isinstance(site_records, list):
            raise AuditError(f"{record['id']}: site audit list is missing")
        actual_sites = [
            item.get("site", {}).get("name")
            for item in site_records if isinstance(item, Mapping)
        ]
        if (len(actual_sites) != len(set(actual_sites))
                or set(actual_sites) != expected_sites):
            raise AuditError(
                f"{record['id']}: actual station coverage is not exact")
        evidence = record.get("evidence", {})
        _require_record_file(
            evidence.get("svg"), evidence.get("svg_sha256"),
            f"{record['id']} SVG evidence")
        png = evidence.get("png", {})
        if not isinstance(png, Mapping):
            raise AuditError(f"{record['id']}: PNG evidence record is missing")
        _require_record_file(
            png.get("path"), png.get("sha256"),
            f"{record['id']} PNG evidence")
        slicer = record.get("slicer", {})
        _require_record_file(
            slicer.get("gcode"), slicer.get("gcode_sha256"),
            f"{record['id']} G-code")
        _require_record_file(
            slicer.get("result_json"), slicer.get("result_sha256"),
            f"{record['id']} Bambu result")
        _require_record_file(
            slicer.get("project_3mf"), slicer.get("project_3mf_sha256"),
            f"{record['id']} audited Bambu 3MF")
        pause_count = sum(
            int(group["magnet_count"]) for group in _pause_groups(record))
        if pause_count != len(expected_sites):
            raise AuditError(
                f"{record['id']}: pause coverage does not equal its stations")
        pause_station_count += pause_count
        covered_station_count += len(actual_sites)
    if covered_station_count != expected_magnet_count:
        raise AuditError(
            "canonical station coverage total differs from catalog: "
            f"{covered_station_count} != {expected_magnet_count}")
    expected_pause_count = sum(
        len(artifact["sites"]) for artifact in artifacts
        if artifact.get("p2s_printability") != "not_printable_oversize")
    if pause_station_count != expected_pause_count:
        raise AuditError(
            "canonical pause total differs from printable catalog stations: "
            f"{pause_station_count} != {expected_pause_count}")


def _write_manifest_bundle(
    output: Path,
    catalog_path: Path,
    catalog: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]] = (),
) -> dict[str, Path]:
    all_groups = []
    for record in records:
        for group in _pause_groups(record):
            all_groups.append({
                "artifact_id": record["id"],
                "state": record["state"],
                "variant": record["variant"],
                "part": record["part"],
                "print_orientation": record["print_orientation"],
                "stl": record["input"]["stl"],
                "stl_sha256": record["input"]["stl_sha256"],
                "gcode": record["slicer"]["gcode"],
                "gcode_sha256": record["slicer"]["gcode_sha256"],
                "audited_bambu_3mf": record["slicer"]["project_3mf"],
                "audited_bambu_3mf_sha256": record["slicer"][
                    "project_3mf_sha256"],
                "bambu_arrange_rz_degrees": record["slicer"][
                    "bambu_3mf_audit"]["rigid_rz"]["rz_degrees"],
                **group,
            })
    actual_slice_records = [
        record for record in records
        if record.get("audit_mode") == "actual_p2s_slice"
    ]
    oversize_records = [
        record for record in records
        if record.get("audit_mode") == "exact_split_proxy_coverage"
    ]
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "authoritative": True,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "safety_boundary": (
            "local slicing and static inspection only; no printer upload, "
            "MQTT, FTPS, or print start"),
        "catalog": {
            "path": str(catalog_path),
            "sha256": catalog["_catalog_sha256"],
            "source_revision": catalog.get("source_revision"),
        },
        "profile": profile_bundle["identity"],
        "summary": {
            "catalog_artifact_count": catalog.get(
                "inventory", {}).get("artifact_count"),
            "catalog_magnet_station_count": catalog.get(
                "inventory", {}).get("magnet_count"),
            "requested_artifact_count": len(records) + len(failures),
            "sliced_artifact_count": len(actual_slice_records),
            "p2s_oversize_artifact_count": len(oversize_records),
            "p2s_oversize_exact_split_covered": sum(
                record.get("status") == OVERSIZE_COVERED_STATUS
                for record in oversize_records),
            "pause_group_count": len(all_groups),
            "magnet_count": sum(group["magnet_count"] for group in all_groups),
            "p2s_pause_magnet_count": sum(
                group["magnet_count"] for group in all_groups),
            "oversize_proxy_covered_site_count": sum(
                len(record.get("cavity_audit_coverage", {}).get("sites", ()))
                for record in oversize_records),
            "passed_artifacts": sum(
                record.get("status") in ("pass", OVERSIZE_COVERED_STATUS)
                for record in records),
            "failed_artifacts": (
                sum(record.get("status") not in (
                    "pass", OVERSIZE_COVERED_STATUS)
                    for record in records)
                + len(failures)),
        },
        "slice_failures": list(failures),
        "pause_groups": all_groups,
        "artifacts": records,
    }
    json_path = output / "captive_magnet_pause_manifest.json"
    _write_json(json_path, manifest)
    csv_path = output / "captive_magnet_pause_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "artifact_id", "state", "variant", "part", "print_orientation",
            "group", "sites", "magnet_count",
            "last_completely_open_layer_z_mm",
            "cavity_bury_roof_start_plane_z_mm",
            "first_closing_layer_z_mm", "pause_marker_z_mm",
            "minimum_seated_below_last_open_layer_mm",
            "minimum_seated_below_first_closing_layer_mm",
            "print_insertion_direction_xyz", "insertion_instruction",
            "stl", "stl_sha256", "gcode", "gcode_sha256",
            "audited_bambu_3mf", "audited_bambu_3mf_sha256",
            "bambu_arrange_rz_degrees", "polarity")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in all_groups:
            writer.writerow({
                **{key: group.get(key) for key in fields},
                "sites": "; ".join(group["sites"]),
                "cavity_bury_roof_start_plane_z_mm": "; ".join(
                    f"{value:.2f}" for value in group[
                        "cavity_bury_roof_start_plane_z_mm"]),
                "print_insertion_direction_xyz": json.dumps(
                    group["print_insertion_direction_xyz"],
                    separators=(",", ":")),
                "polarity": json.dumps(group["polarity"], separators=(",", ":")),
            })
    md_path = output / "CAPTIVE_MAGNET_PAUSE_MANIFEST.md"
    lines = [
        "# Captive-magnet pause manifest",
        "",
        ("Authoritative for the exact STL and profile hashes below. This run "
         "used Bambu Lab P2S, 0.4 mm nozzle, 0.16 mm High Quality, Classic "
         "walls, and Bambu PLA Tough+. All parts are front-face-down."),
        "",
        "This audit did not contact a printer and did not upload or start a print.",
        "",
        "## Insertion procedure",
        "",
        "1. Print each part front-face-down; do not auto-orient it.",
        "2. Add the Bambu Studio pause marker at the exact **first-closing** Z listed below.",
        ("3. At each pause, insert the listed number of D5 x 2 mm magnets "
         "vertically downward from above (+Z side) along print `-Z` "
         "(`print_insertion_direction_xyz = [0, 0, -1]`), with the marked "
         "pole oriented exactly as specified."),
        "4. Ensure every magnet is fully seated below the completed layer and cannot rise into the toolhead path.",
        "5. Resume printing. Polarity cannot be corrected after the roof buries the magnet.",
        "",
        "## Exact pauses",
        "",
        "| State | Variant / part | Pause Z | Last open | Seated margin | Magnets / sites | Insertion | Polarity |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for group in all_groups:
        polarity = "<br>".join(
            (f"`{item['site']}`: marked pole → "
             f"`{item['print_marked_pole_axis_xyz']}` in print coordinates; "
             f"{item['instruction']}")
            for item in group["polarity"])
        lines.append(
            f"| {group['state']} | {group['variant']} / `{group['part']}` | "
            f"**{group['pause_marker_z_mm']:.2f} mm** | "
            f"{group['last_completely_open_layer_z_mm']:.2f} mm | "
            f"{group['minimum_seated_below_last_open_layer_mm']:.2f} mm | "
            f"{group['magnet_count']} / {', '.join(group['sites'])} | "
            f"`{group['print_insertion_direction_xyz']}`: "
            f"{group['insertion_instruction']} | {polarity} |")
    placement_groups = {
        group["artifact_id"]: group for group in all_groups
    }
    lines.extend((
        "",
        "## Audited Bambu arrangements",
        "",
        ("Every listed 3MF was exported by the same Bambu slice invocation, "
         "hash-bound to the staged STL, and audited as an exact mesh with "
         "only a proper unit-scale rotation about print Z plus XY placement."),
        "",
        "| State | Variant / part | Arrange Rz | Audited 3MF | SHA-256 |",
        "|---|---|---:|---|---|",
    ))
    for group in placement_groups.values():
        lines.append(
            f"| {group['state']} | {group['variant']} / "
            f"`{group['part']}` | "
            f"{group['bambu_arrange_rz_degrees']:.6f} deg | "
            f"`{group['audited_bambu_3mf']}` | "
            f"`{group['audited_bambu_3mf_sha256']}` |")
    if oversize_records:
        lines.extend((
            "",
            "## Explicitly not P2S-printable",
            "",
            ("These canonical monoliths exceed the P2S bed in their mandatory "
             "front-face-down orientation. They have no generated monolith "
             "G-code and no pause group. Their cavity evidence comes only "
             "from the exact same-state keyed split prints listed below."),
            "",
            "| State | Canonical part | Front-down size | Coverage status | Exact split proxies |",
            "|---|---|---|---|---|",
        ))
        for record in oversize_records:
            proxies = record.get("cavity_audit_coverage", {}).get("sites", ())
            proxy_text = ", ".join(
                f"`{item['site']}` → `{item['proxy_artifact_id']}`"
                for item in proxies) or "none"
            size = " × ".join(
                f"{float(value):.2f}"
                for value in record["input"]["size_mm"])
            lines.append(
                f"| {record['state']} | `{record['part']}` | {size} mm | "
                f"`{record['status']}` | {proxy_text} |")
    lines.extend((
        "",
        "## Profile and evidence",
        "",
        f"- Catalog SHA-256: `{manifest['catalog']['sha256']}`",
        f"- Resolved profile-set SHA-256: `{manifest['profile']['profile_set_sha256']}`",
        f"- Bambu Studio binary SHA-256: `{manifest['profile']['binary_sha256']}`",
        f"- Artifacts: {manifest['summary']['passed_artifacts']} passed, "
        f"{manifest['summary']['failed_artifacts']} failed",
        ("- Each printable artifact directory under `slices/` contains the "
         "hash-bound arranged Bambu 3MF, plain G-code, Bambu `result.json`, "
         "static validator output, and five-layer SVG/PNG toolpath evidence "
         "for every cavity."),
        "",
        "The JSON file is the machine-readable authority; this Markdown and the CSV are derived views.",
        "",
    ))
    if failures:
        lines.extend(("## Slice failures", ""))
        for failure in failures:
            lines.append(f"- `{failure['id']}`: {failure['error']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _validate_manifest_bundle(paths: Mapping[str, Path]) -> None:
    if {path.name for path in paths.values()} != set(
            CANONICAL_MANIFEST_FILENAMES):
        raise AuditError("staged manifest bundle has unexpected filenames")
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise AuditError(f"staged {label} manifest is missing or empty")
    manifest = _load_json(paths["json"])
    if manifest.get("authoritative") is not True:
        raise AuditError("staged JSON manifest is not authoritative")
    summary = manifest.get("summary", {})
    if (summary.get("failed_artifacts") != 0
            or summary.get("requested_artifact_count")
            != EXPECTED_RELEASE_ARTIFACT_COUNT
            or summary.get("catalog_artifact_count")
            != EXPECTED_RELEASE_ARTIFACT_COUNT
            or summary.get("catalog_magnet_station_count")
            != EXPECTED_RELEASE_MAGNET_COUNT):
        raise AuditError("staged JSON manifest lacks complete passing coverage")
    groups = manifest.get("pause_groups")
    if not isinstance(groups, list) or not groups:
        raise AuditError("staged JSON manifest contains no pause groups")
    for group in groups:
        if (not isinstance(group, Mapping)
                or group.get("print_insertion_direction_xyz")
                != list(PRINT_INSERTION_DIRECTION_XYZ)
                or group.get("insertion_instruction")
                != PRINT_INSERTION_INSTRUCTION):
            raise AuditError(
                "staged JSON manifest lacks the exact print -Z insertion "
                "contract")
        _require_record_file(
            group.get("audited_bambu_3mf"),
            group.get("audited_bambu_3mf_sha256"),
            f"{group.get('artifact_id', '<unknown>')} arranged Bambu 3MF")
        _float(
            group.get("bambu_arrange_rz_degrees"),
            f"{group.get('artifact_id', '<unknown>')} arrange Rz")
    with paths["csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AuditError("staged CSV manifest contains no pause rows")
    for row in rows:
        if (row.get("print_insertion_direction_xyz") != "[0.0,0.0,-1.0]"
                or row.get("insertion_instruction")
                != PRINT_INSERTION_INSTRUCTION):
            raise AuditError(
                "staged CSV manifest lacks the exact print -Z insertion "
                "contract")
        _require_record_file(
            row.get("audited_bambu_3mf"),
            row.get("audited_bambu_3mf_sha256"),
            f"{row.get('artifact_id', '<unknown>')} CSV arranged Bambu 3MF")
        _float(
            row.get("bambu_arrange_rz_degrees"),
            f"{row.get('artifact_id', '<unknown>')} CSV arrange Rz")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    if "# Captive-magnet pause manifest" not in markdown:
        raise AuditError("staged Markdown manifest has the wrong document kind")
    for required_text in (
            "print_insertion_direction_xyz = [0, 0, -1]",
            "## Audited Bambu arrangements",
            "audited_slice_project.3mf",
            "provisional unpaired V0 convention",
            "unpaired coupon1 regression station"):
        if required_text not in markdown:
            raise AuditError(
                "staged Markdown manifest omits required insertion/polarity "
                f"instruction: {required_text}")


def _transactional_publish_bundle(
    paths: Mapping[str, Path], destination: Path,
) -> dict[str, Path]:
    """Replace the canonical three-file set, rolling back any process error."""
    destination.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(
        prefix=".captive-manifest-backup-", dir=destination))
    targets = {key: destination / path.name for key, path in paths.items()}
    backups: dict[str, Path] = {}
    installed: list[str] = []
    retain_backup = False
    try:
        for key, target in targets.items():
            if target.exists():
                backup = backup_dir / target.name
                os.replace(target, backup)
                backups[key] = backup
        for key, staged_path in paths.items():
            os.replace(staged_path, targets[key])
            installed.append(key)
    except Exception as exc:
        restore_errors = []
        # A backup can replace a newly installed file atomically.  Only a
        # target that did not exist before this transaction needs deletion.
        for key in installed:
            if key in backups:
                continue
            try:
                targets[key].unlink(missing_ok=True)
            except Exception as remove_exc:  # pragma: no cover - catastrophic FS
                restore_errors.append(str(remove_exc))
        for key, backup in backups.items():
            try:
                os.replace(backup, targets[key])
            except Exception as restore_exc:  # pragma: no cover - catastrophic FS
                restore_errors.append(str(restore_exc))
        detail = ("; rollback errors: " + "; ".join(restore_errors)
                  if restore_errors else "")
        retain_backup = bool(restore_errors)
        raise AuditError(
            f"canonical manifest transaction failed: {exc}{detail}"
            + (f"; retained backups at {backup_dir}"
               if retain_backup else "")) from exc
    finally:
        if not retain_backup:
            shutil.rmtree(backup_dir, ignore_errors=True)
    return targets


def write_manifests(
    output: Path,
    catalog_path: Path,
    catalog: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]] = (),
) -> dict[str, Path]:
    """Validate, stage, then transactionally publish canonical manifests."""
    _validate_complete_release(catalog, records, failures)
    if sha256_file(catalog_path) != catalog.get("_catalog_sha256"):
        raise AuditError("release catalog changed before manifest publication")
    if sha256_file(CATALOG_SCHEMA) != catalog.get(
            "_catalog_schema_sha256"):
        raise AuditError(
            "release catalog schema changed before manifest publication")
    for artifact in catalog["artifacts"]:
        _validate_artifact_bindings(artifact)
    _verify_profile_inputs(
        profile_bundle, Path(profile_bundle["identity"]["binary"]))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".captive-manifest-stage-", dir=output) as directory:
        staged_paths = _write_manifest_bundle(
            Path(directory), catalog_path, catalog, profile_bundle,
            records, failures)
        _validate_manifest_bundle(staged_paths)
        _validate_complete_release(catalog, records, failures)
        if (sha256_file(catalog_path) != catalog["_catalog_sha256"]
                or sha256_file(CATALOG_SCHEMA)
                != catalog["_catalog_schema_sha256"]):
            raise AuditError(
                "release catalog authority changed during manifest staging")
        for artifact in catalog["artifacts"]:
            _validate_artifact_bindings(artifact)
        _verify_profile_inputs(
            profile_bundle, Path(profile_bundle["identity"]["binary"]))
        return _transactional_publish_bundle(staged_paths, output)


def _filter_artifacts(
    artifacts: Sequence[Mapping[str, Any]], patterns: Sequence[str],
) -> list[Mapping[str, Any]]:
    if not patterns:
        return list(artifacts)
    selected = []
    for artifact in artifacts:
        haystacks = (artifact["id"], artifact["part"], artifact["variant"],
                     str(artifact["stl"]))
        if any(fnmatch.fnmatch(value, pattern)
               for pattern in patterns for value in haystacks):
            selected.append(artifact)
    if not selected:
        raise AuditError(f"--only patterns matched no artifacts: {patterns}")
    return selected


def _authoritative_run_requested(
    only_patterns: Sequence[str], dry_run: bool,
) -> bool:
    """Only an unfiltered, executed release audit may publish pauses."""
    return not only_patterns and not dry_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Bambu P2S captive-magnet slicing and pause audit")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bambu-studio")
    parser.add_argument("--bambu-system-root", type=Path)
    parser.add_argument("--only", action="append", default=[],
                        help="glob against artifact id/part/variant/STL; repeatable")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel local Bambu Studio processes (default 1)")
    parser.add_argument("--no-reuse", action="store_true",
                        help="ignore content-addressed completed slices")
    parser.add_argument("--prepare-profiles-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="write resolved profiles and print commands only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise AuditError("--jobs must be positive")
    authoritative_request = _authoritative_run_requested(
        args.only, args.dry_run)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bambu = _find_bambu_binary(args.bambu_studio)
    profile_bundle = prepare_profiles(
        args.profile.expanduser().resolve(), output,
        system_root=(args.bambu_system_root.expanduser().resolve()
                     if args.bambu_system_root else None),
        bambu_binary=bambu)
    if args.prepare_profiles_only:
        print(output / "profiles" / "profile_provenance.json")
        return 0
    catalog_path = args.catalog.expanduser().resolve()
    catalog = normalize_catalog(catalog_path)
    selected = _filter_artifacts(catalog["artifacts"], args.only)
    catalog_by_id = {
        artifact["id"]: artifact for artifact in catalog["artifacts"]
    }
    # Selecting an oversized monolith implicitly selects every exact split
    # dependency needed to prove its cavities.  This never selects a virtual,
    # scaled, tilted, or clipped version of the monolith itself.
    selected_by_id = {artifact["id"]: artifact for artifact in selected}
    for artifact in tuple(selected):
        for proxy in artifact.get("cavity_audit_proxies", ()):
            selected_by_id[proxy["artifact_id"]] = catalog_by_id[
                proxy["artifact_id"]]
    requested_artifacts = sorted(
        selected_by_id.values(), key=lambda item: item["id"])
    with tempfile.TemporaryDirectory(
            prefix=".captive-input-stage-", dir=output) as stage_directory:
        staged = _stage_release_inputs(
            catalog_path, requested_artifacts, Path(stage_directory),
            expected_catalog_sha256=catalog["_catalog_sha256"],
            expected_catalog_schema_sha256=catalog[
                "_catalog_schema_sha256"])
        artifacts = staged["artifacts"]
        oversized = [
            artifact for artifact in artifacts
            if artifact.get("p2s_printability") == "not_printable_oversize"
        ]
        slice_targets = [
            artifact for artifact in artifacts if artifact not in oversized
        ]
        catalog_sha = staged["catalog_sha256"]
        records: list[Mapping[str, Any]] = []
        failures: list[dict[str, str]] = []

        def work(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
            return _slice_one(
                artifact, output_root=output, profile_bundle=profile_bundle,
                bambu=bambu, catalog_sha=catalog_sha,
                reuse=not args.no_reuse, dry_run=args.dry_run)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.jobs) as pool:
            future_map = {
                pool.submit(work, artifact): artifact
                for artifact in slice_targets
            }
            for future in concurrent.futures.as_completed(future_map):
                artifact = future_map[future]
                try:
                    record = future.result()
                    records.append(record)
                    print(
                        f"{record['id']}: "
                        f"{record.get('status', 'dry-run')}", flush=True)
                except Exception as exc:  # keep auditing independent parts
                    failures.append({
                        "id": artifact["id"], "error": str(exc)})
                    print(
                        f"{artifact['id']}: ERROR: {exc}",
                        file=sys.stderr, flush=True)

        _verify_staged_release_inputs(staged, catalog_path)
        _verify_profile_inputs(profile_bundle, bambu)
        if args.dry_run:
            _write_json(output / "dry_run_commands.json", {
                "authoritative": False,
                "canonical_pause_manifest_published": False,
                "catalog_sha256": catalog_sha,
                "profile": profile_bundle["identity"],
                "records": records,
                "oversize_not_sliced": [{
                    "id": artifact["id"],
                    "p2s_printable": False,
                    "proxy_artifact_ids": sorted({
                        proxy["artifact_id"]
                        for proxy in artifact["cavity_audit_proxies"]
                    }),
                    "policy": "no monolith G-code or pause group",
                } for artifact in oversized],
                "failures": failures,
            })
            return 1 if failures else 0

        records_by_id = {record["id"]: record for record in records}
        for artifact in oversized:
            records.append(_oversize_proxy_coverage_record(
                artifact, records_by_id, profile_bundle))
        records.sort(key=lambda item: item["id"])
        # Oversize coverage reads proxy evidence after the first immutable
        # verification, so verify every authority once more before any output
        # is eligible to become canonical.
        _verify_staged_release_inputs(staged, catalog_path)
        _verify_profile_inputs(profile_bundle, bambu)
        failed_records = [
            record for record in records
            if record.get("status") not in (
                "pass", OVERSIZE_COVERED_STATUS)
        ]

        if not authoritative_request:
            # Dry runs returned above, so the remaining non-authoritative
            # mode is necessarily an explicit --only subset.
            if not args.only:
                raise AuditError(
                    "internal authority classification is inconsistent")
            # A filtered audit is useful diagnostics but can never be mistaken
            # for the release-wide pause authority.
            subset_path = output / "subset_slice_results.json"
            _write_json(subset_path, {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "authoritative": False,
                "canonical_pause_manifest_published": False,
                "catalog_sha256": catalog_sha,
                "requested_patterns": list(args.only),
                "requested_artifact_ids": [
                    artifact["id"] for artifact in artifacts],
                "records": records,
                "failures": failures,
                "pause_groups": [],
                "note": (
                    "Subset audits never publish pause instructions; run the "
                    "complete unfiltered release audit for authority."),
            })
            print(
                f"subset (non-authoritative): {subset_path}\n"
                "canonical pause manifests were not modified")
            return 1 if failures or failed_records else 0

        if failures or failed_records:
            stamp = dt.datetime.now(dt.timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ")
            failure_path = (
                output / "failed_runs" / f"failed_slice_{stamp}.json")
            _write_json(failure_path, {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "authoritative": False,
                "canonical_pause_manifest_published": False,
                "catalog_sha256": catalog_sha,
                "records": records,
                "failures": failures,
                "pause_groups": [],
            })
            print(
                f"release audit failed: {failure_path}\n"
                "canonical pause manifests were not modified",
                file=sys.stderr)
            return 1

        _validate_complete_release(catalog, records, failures)
        paths = write_manifests(
            output, catalog_path, catalog, profile_bundle, records, failures)
        print("\n".join(
            f"{key}: {path}" for key, path in paths.items()))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
