"""Pure-Python validation for the released front-face-down transform.

Released acoustic parts rotate the installed CAD frame 180 degrees about X
so the acoustic/front face lies on the bed, then optionally rotate only about
the bed-normal Z axis. Translation may place the result at the STL origin,
but no X/Y tilt or reflection is permitted.

This module deliberately imports no CAD package so the remote catalog
producer and local, no-OCC slicing audit enforce the same contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PRINT_SIDECAR_SCHEMA_VERSION = 1

# One immutable, pure-Python authority for the release-wide build-plate
# direction.  The captive-magnet catalog contains only the magnet-bearing
# subset, but its root contract deliberately states the broader acoustic
# release rule: every released acoustic part uses the same front datum.
# Production coupons follow the same transform through their exporter.
RELEASE_ACOUSTIC_PRINT_CONTRACT = {
    "scope": "every_released_acoustic_part",
    "orientation": "front_face_down",
    "allowed_additional_rotation": "in_bed_Z_only",
    "reason": "consistent acoustic-front build-plate texture",
    "slicer": "Bambu Studio P2S 0.4 mm / 0.16 mm Arachne",
    "pause_authority": "actual sliced G-code first-closing layer",
    "printer_contact": "forbidden; offline slicing only",
    "oversize_policy": (
        "never scale, tilt, clip, or virtual-bed an oversized STL; "
        "the Obi-Wan LM monolith is explicitly not P2S-printable and "
        "its cavity G-code is covered only by exact same-state "
        "keyed split artifacts"
    ),
}


class FrontDownContractError(ValueError):
    """Raised when release metadata does not encode X180 plus Z-only."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrontDownContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FrontDownContractError(f"{label} must be a finite number")
    return result


def _vec3(value: Any, label: str) -> tuple[float, float, float]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != 3):
        raise FrontDownContractError(f"{label} must contain three numbers")
    return tuple(
        _finite_number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path_for_stl(stl_path: str | os.PathLike[str]) -> Path:
    """Return the one adjacent print-authority path for an STL."""
    path = Path(stl_path)
    if path.suffix.lower() != ".stl":
        raise FrontDownContractError(
            f"print authority requires an STL path, got {path}")
    return path.with_suffix(".print.json")


def validate_front_down_transform(
    record: Mapping[str, Any],
    *,
    label: str = "print transform",
    tolerance: float = 1.0e-9,
) -> tuple[tuple[float, float, float, float], ...]:
    """Validate and return a canonical numeric source-to-STL matrix.

    ``record`` is either a ``*.print.json`` payload or a wing
    ``print_transform`` object. A ``front_face_down`` string alone is not
    sufficient: the linear transform must equal
    ``Rz(z_rotation) * Rx(180 degrees)`` and its only other freedom is a
    finite XYZ translation.
    """
    if record.get("print_orientation") != "front_face_down":
        raise FrontDownContractError(
            f"{label}: print_orientation must be front_face_down")

    rotation = record.get("rotation_deg")
    if not isinstance(rotation, Mapping) or set(rotation) != {"x", "z"}:
        raise FrontDownContractError(
            f"{label}: rotation_deg must contain only x and z")
    x_angle = _finite_number(rotation["x"], f"{label}.rotation_deg.x")
    z_angle = _finite_number(rotation["z"], f"{label}.rotation_deg.z")
    if not math.isclose(x_angle, 180.0, abs_tol=tolerance, rel_tol=0.0):
        raise FrontDownContractError(
            f"{label}: X rotation must be exactly 180 degrees")

    raw_matrix = record.get("source_to_stl_matrix")
    if (not isinstance(raw_matrix, Sequence)
            or isinstance(raw_matrix, (str, bytes))
            or len(raw_matrix) != 4):
        raise FrontDownContractError(f"{label}: matrix must be 4x4")
    rows = []
    for row_index, raw_row in enumerate(raw_matrix):
        if (not isinstance(raw_row, Sequence)
                or isinstance(raw_row, (str, bytes))
                or len(raw_row) != 4):
            raise FrontDownContractError(f"{label}: matrix must be 4x4")
        rows.append(tuple(
            _finite_number(value, f"{label}.matrix[{row_index}][{column}]")
            for column, value in enumerate(raw_row)
        ))
    matrix = tuple(rows)

    radians = math.radians(z_angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    expected_linear = (
        (cosine, sine, 0.0),
        (sine, -cosine, 0.0),
        (0.0, 0.0, -1.0),
    )
    for row in range(3):
        for column in range(3):
            if not math.isclose(
                    matrix[row][column], expected_linear[row][column],
                    abs_tol=tolerance, rel_tol=0.0):
                raise FrontDownContractError(
                    f"{label}: matrix is not X180 plus Z-only rotation")
    for column, expected in enumerate((0.0, 0.0, 0.0, 1.0)):
        if not math.isclose(
                matrix[3][column], expected,
                abs_tol=tolerance, rel_tol=0.0):
            raise FrontDownContractError(
                f"{label}: invalid homogeneous matrix bottom row")

    if "stl_origin_translation_mm" in record:
        translation = _vec3(
            record["stl_origin_translation_mm"],
            f"{label}.stl_origin_translation_mm")
        for row, expected in enumerate(translation):
            if not math.isclose(
                    matrix[row][3], expected,
                    abs_tol=tolerance, rel_tol=0.0):
                raise FrontDownContractError(
                    f"{label}: matrix translation disagrees with metadata")
    return matrix


def _validate_print_sidecar_payload(
    stl_path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
    tolerance: float,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "part",
        "stl",
        "stl_bytes",
        "stl_sha256",
        "print_orientation",
        "source_to_stl_matrix",
        "rotation_deg",
        "pre_translation_bbox_min_mm",
        "stl_origin_translation_mm",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise FrontDownContractError(
            f"{label}: missing required field(s): {', '.join(missing)}")
    if (type(payload.get("schema_version")) is not int
            or payload["schema_version"] != PRINT_SIDECAR_SCHEMA_VERSION):
        raise FrontDownContractError(
            f"{label}: schema_version must be "
            f"{PRINT_SIDECAR_SCHEMA_VERSION}")
    part = payload.get("part")
    if not isinstance(part, str) or not part.strip():
        raise FrontDownContractError(f"{label}: part must be a nonempty string")
    if part != stl_path.stem:
        raise FrontDownContractError(
            f"{label}: part {part!r} does not match STL stem "
            f"{stl_path.stem!r}")
    if payload.get("stl") != stl_path.name:
        raise FrontDownContractError(
            f"{label}: stl must be the exact basename {stl_path.name!r}")
    size = payload.get("stl_bytes")
    if type(size) is not int or size != stl_path.stat().st_size:
        raise FrontDownContractError(
            f"{label}: stl_bytes does not match {stl_path.name}")
    digest = payload.get("stl_sha256")
    if (not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise FrontDownContractError(
            f"{label}: stl_sha256 must be a lowercase SHA-256 digest")
    if digest != _sha256_file(stl_path):
        raise FrontDownContractError(
            f"{label}: stl_sha256 does not match {stl_path.name}")

    validate_front_down_transform(
        payload, label=label, tolerance=tolerance)
    bbox_minimum = _vec3(
        payload["pre_translation_bbox_min_mm"],
        f"{label}.pre_translation_bbox_min_mm")
    translation = _vec3(
        payload["stl_origin_translation_mm"],
        f"{label}.stl_origin_translation_mm")
    for axis, (minimum, offset) in enumerate(
            zip(bbox_minimum, translation, strict=True)):
        if not math.isclose(
                minimum + offset, 0.0,
                abs_tol=tolerance, rel_tol=0.0):
            raise FrontDownContractError(
                f"{label}: origin translation does not negate "
                f"pre-translation bbox minimum on axis {axis}")
    return dict(payload)


def validate_print_sidecar(
    stl_path: str | os.PathLike[str],
    sidecar_path: str | os.PathLike[str] | None = None,
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Validate one adjacent, hash-bound front-down print sidecar."""
    stl = Path(stl_path)
    expected_sidecar = sidecar_path_for_stl(stl)
    sidecar = (expected_sidecar if sidecar_path is None
               else Path(sidecar_path))
    if sidecar.resolve() != expected_sidecar.resolve():
        raise FrontDownContractError(
            f"{sidecar}: print sidecar must be adjacent to {stl.name} "
            f"as {expected_sidecar.name}")
    if not stl.is_file():
        raise FrontDownContractError(f"missing STL for print authority: {stl}")
    if not sidecar.is_file():
        raise FrontDownContractError(
            f"missing print authority for {stl.name}: {sidecar}")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontDownContractError(
            f"{sidecar}: unreadable print sidecar: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrontDownContractError(
            f"{sidecar}: print sidecar must contain a JSON object")
    return _validate_print_sidecar_payload(
        stl, payload, label=str(sidecar), tolerance=tolerance)


def write_print_sidecar(
    stl_path: str | os.PathLike[str],
    *,
    part: str,
    transform: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write the canonical adjacent print-authority sidecar."""
    stl = Path(stl_path)
    sidecar = sidecar_path_for_stl(stl)
    if not stl.is_file():
        raise FrontDownContractError(f"missing STL for print authority: {stl}")
    if not isinstance(transform, Mapping):
        raise FrontDownContractError("print transform must be an object")
    transform_fields = (
        "print_orientation",
        "source_to_stl_matrix",
        "rotation_deg",
        "pre_translation_bbox_min_mm",
        "stl_origin_translation_mm",
    )
    missing_transform = [
        field for field in transform_fields if field not in transform]
    if missing_transform:
        raise FrontDownContractError(
            "print transform lacks required field(s): "
            + ", ".join(missing_transform))
    payload: dict[str, Any] = {
        "schema_version": PRINT_SIDECAR_SCHEMA_VERSION,
        "part": part,
        "stl": stl.name,
        "stl_bytes": stl.stat().st_size,
        "stl_sha256": _sha256_file(stl),
        **{field: transform[field] for field in transform_fields},
    }
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise FrontDownContractError(
                "print sidecar extra fields must be an object")
        overlap = sorted(set(extra) & set(payload))
        if overlap:
            raise FrontDownContractError(
                "print sidecar extra fields override authority field(s): "
                + ", ".join(overlap))
        payload.update(extra)
    try:
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        round_trip = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise FrontDownContractError(
            f"{sidecar}: print sidecar is not JSON-serializable: {exc}") from exc
    _validate_print_sidecar_payload(
        stl, round_trip, label=str(sidecar), tolerance=1.0e-9)

    temporary = sidecar.with_name(
        f".{sidecar.stem}.{os.getpid()}.tmp.json")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        if json.loads(temporary.read_text(encoding="utf-8")) != round_trip:
            raise FrontDownContractError(
                f"{sidecar}: temporary print sidecar round trip failed")
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    validate_print_sidecar(stl, sidecar)
    return sidecar
