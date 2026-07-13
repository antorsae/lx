"""Strict release sweep for every exported STL and V1LF manifest.

The primary contract remains explicit: every undirected mesh edge is
shared by exactly two oppositely wound triangles.  The checker also
rejects malformed binary length, repeated/zero-area triangles, duplicate
facets and stale state-incompatible V1LF artifacts.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

# The exact edge/component sweep can consume substantial memory on dense
# release meshes even though it does not build CAD. Direct CLI use therefore
# receives the same process-tree cap and host-memory floor as Make.
if __name__ == "__main__":
    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        guard = Path(__file__).with_name("run_memory_guarded.py")
        raise SystemExit(subprocess.run(
            [sys.executable, str(guard), "--", sys.executable,
             str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False).returncode)

import math
import hashlib
import json
import struct
from collections import Counter

from PIL import Image

from write_v1lf_release_manifest import (
    FORMAT_VERSION,
    QUALIFICATION_RECORD,
    expected_artifact_names,
    native_stage_record,
    sha256_file,
    source_hashes,
)


def stl_diagnostics(path: Path) -> dict[str, int | float]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path}: shorter than binary STL header")
    triangles = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + triangles * 50
    if len(data) != expected:
        raise ValueError(
            f"{path}: binary length {len(data)} != expected {expected}")

    undirected: Counter = Counter()
    orientation_balance: Counter = Counter()
    facets: Counter = Counter()
    degenerate = 0
    nonfinite = 0
    signed_volume = 0.0
    offset = 84
    for _ in range(triangles):
        raw = struct.unpack_from("<9f", data, offset + 12)
        tri = tuple(tuple(raw[i:i + 3]) for i in (0, 3, 6))
        if not all(math.isfinite(value) for vertex in tri for value in vertex):
            nonfinite += 1
            degenerate += 1
            offset += 50
            continue
        facets[tuple(sorted(tri))] += 1
        if len(set(tri)) < 3:
            degenerate += 1
        else:
            ab = tuple(tri[1][i] - tri[0][i] for i in range(3))
            ac = tuple(tri[2][i] - tri[0][i] for i in range(3))
            cross = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            if math.sqrt(sum(value * value for value in cross)) <= 1e-9:
                degenerate += 1
            signed_volume += (
                tri[0][0] * (tri[1][1] * tri[2][2] - tri[1][2] * tri[2][1])
                - tri[0][1] * (tri[1][0] * tri[2][2] - tri[1][2] * tri[2][0])
                + tri[0][2] * (tri[1][0] * tri[2][1] - tri[1][1] * tri[2][0])
            ) / 6.0
        for a, b in ((0, 1), (1, 2), (2, 0)):
            start, stop = tri[a], tri[b]
            if start == stop:
                # Already counted as a degenerate facet. A one-vertex
                # frozenset is not an edge and must not enter winding checks.
                continue
            edge = (start, stop) if start < stop else (stop, start)
            undirected[edge] += 1
            orientation_balance[edge] += 1 if (start, stop) == edge else -1
        offset += 50

    open_edges = sum(count == 1 for count in undirected.values())
    over_shared = sum(count > 2 for count in undirected.values())
    winding_errors = 0
    for edge, count in undirected.items():
        if count != 2:
            continue
        if orientation_balance[edge] != 0:
            winding_errors += 1
    duplicate_facets = sum(count - 1 for count in facets.values() if count > 1)
    del facets, orientation_balance

    # One printable STL must be one connected surface shell. Reuse the
    # canonical edge keys already resident in memory and union their vertices.
    parent = {}

    def find(vertex):
        parent.setdefault(vertex, vertex)
        root = vertex
        while parent[root] != root:
            root = parent[root]
        while parent[vertex] != vertex:
            vertex, parent[vertex] = parent[vertex], root
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in undirected:
        union(a, b)
    components = len({find(vertex) for vertex in parent}) if parent else 0
    return {
        "triangles": triangles,
        "open": open_edges,
        "over_shared": over_shared,
        "winding": winding_errors,
        "degenerate": degenerate,
        "duplicates": duplicate_facets,
        "nonfinite": nonfinite,
        "signed_volume": signed_volume,
        "zero_volume": int(abs(signed_volume) <= 1e-6),
        "negative_volume": int(signed_volume < -1e-6),
        "components": components,
        "component_error": int(components != 1),
    }


def stl_edge_parity(path: Path) -> tuple[int, int, int]:
    """Compatibility API: (triangles, exact open, exact over-shared)."""
    facts = stl_diagnostics(path)
    return facts["triangles"], facts["open"], facts["over_shared"]


def _contains_bytes(path: Path, needle: bytes) -> bool:
    """Streaming token search for large, line-wrapped STEP files.

    OpenCascade may wrap a long PRODUCT label in the middle of its quoted
    token.  Fold only physical CR/LF record boundaries so contract checks
    still see the semantic STEP label without loading a multi-GB file.
    """
    overlap = max(0, len(needle) - 1)
    carry = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            data = carry + chunk.replace(b"\r", b"").replace(b"\n", b"")
            if needle in data:
                return True
            carry = data[-overlap:] if overlap else b""
    return False


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _step_has_trailer(path: Path) -> bool:
    """Reject a large but interrupted/truncated Part 21 file."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - 4096))
        tail = stream.read()
    return tail.rstrip().endswith(b"END-ISO-10303-21;")


def _png_has_trailer(path: Path) -> bool:
    """A valid PNG transaction must end with its complete IEND chunk."""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 12))
        return stream.read() == b"\x00\x00\x00\x00IEND\xaeB`\x82"


def _png_diagnostics(path: Path) -> dict:
    """Decode pixels and parsed text metadata, not just raw PNG bytes."""
    with Image.open(path) as image:
        metadata = dict(image.info)
        image.verify()
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        rgba.load()
        width, height = rgba.size
        digest = hashlib.sha256()
        digest.update(struct.pack(">II", width, height))
        digest.update(b"RGBA")
        digest.update(rgba.tobytes())
        sample = rgba.copy()
        sample.thumbnail((256, 256))
        pixels = tuple(sample.getdata())
    count = max(len(pixels), 1)
    nonwhite = sum(
        1 for r, g, b, a in pixels
        if a > 8 and min(r, g, b) < 248)
    chromatic = sum(
        1 for r, g, b, a in pixels
        if a > 8 and max(r, g, b) - min(r, g, b) > 18)
    return {
        "width": width,
        "height": height,
        "title": metadata.get("Title"),
        "description": metadata.get("Description"),
        "pixel_sha256": digest.hexdigest(),
        "nonwhite_fraction": nonwhite / count,
        "chromatic_fraction": chromatic / count,
    }


def _state_token_error(path: Path, expected: bytes, opposite: bytes,
                       description: str) -> str | None:
    has_expected = _contains_bytes(path, expected)
    has_opposite = _contains_bytes(path, opposite)
    if not has_expected or has_opposite:
        return (
            f"{path.parent.name}: {description} state token mismatch "
            f"(expected={has_expected}, opposite={has_opposite})")
    return None


def _release_manifest_errors(state_dir: Path) -> list[str]:
    """Verify one exact, source-bound floor/no-floor artifact transaction."""
    state = state_dir.name
    stand_foot = state == "floor_stand"
    path = state_dir / "v1lf_release_manifest.json"
    if not path.is_file():
        return [f"{state}: missing hash-backed V1LF release manifest"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{state}: unreadable V1LF release manifest: {exc}"]
    errors = []
    expected_header = {
        "format_version": FORMAT_VERSION,
        "variant": "V1LF",
        "routing_revision": "R6F",
        "routing_profile": "v1lf",
        "state": state,
        "stand_foot": stand_foot,
    }
    for key, expected in expected_header.items():
        if data.get(key) != expected:
            errors.append(
                f"{state}: manifest {key}={data.get(key)!r}, "
                f"expected {expected!r}")
    expected_qualification = {
        "status": "pending_physical_fit",
        "release_authorized": False,
        "physical_measure_required": True,
        "record": QUALIFICATION_RECORD.name,
        "record_sha256": sha256_file(QUALIFICATION_RECORD),
        "reason": (
            "MU reference omits terminals; modeled 12 mm pull has "
            "zero positive release overtravel margin"),
    }
    if data.get("qualification") != expected_qualification:
        errors.append(
            f"{state}: manifest physical qualification record/status is stale")
    try:
        current_sources = source_hashes()
    except OSError as exc:
        current_sources = None
        errors.append(
            f"{state}: cannot verify mandatory release references: {exc}")
    if (current_sources is not None
            and data.get("sources") != current_sources):
        errors.append(f"{state}: manifest generator-source hashes are stale")
    try:
        current_stage = native_stage_record(state_dir, stand_foot)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            json.JSONDecodeError) as exc:
        current_stage = None
        errors.append(
            f"{state}: invalid native-stage provenance: {exc}")
    if (current_stage is not None
            and data.get("native_stage") != current_stage):
        errors.append(f"{state}: manifest native-stage record is stale")
    expected_names = set(expected_artifact_names(stand_foot))
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append(f"{state}: manifest artifacts is not an object")
        return errors
    actual_names = set(artifacts)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing:
            errors.append(
                f"{state}: manifest omits artifacts: {', '.join(missing)}")
        if extra:
            errors.append(
                f"{state}: manifest has extra artifacts: {', '.join(extra)}")
    for name in sorted(expected_names & actual_names):
        artifact = state_dir / name
        record = artifacts.get(name)
        if not artifact.is_file():
            errors.append(f"{state}: manifested artifact is missing: {name}")
            continue
        if not isinstance(record, dict):
            errors.append(f"{state}: malformed manifest record: {name}")
            continue
        if record.get("bytes") != artifact.stat().st_size:
            errors.append(f"{state}: manifested size mismatch: {name}")
        if record.get("sha256") != sha256_file(artifact):
            errors.append(f"{state}: manifested hash mismatch: {name}")
    return errors


def _review_artifact_errors(state_dir: Path) -> list[str]:
    state = state_dir.name
    required = {
        "top_baffle_nd25fw4_v1lf_split.step",
        "top_baffle_nd25fw4_v1lf_attachments.step",
        "top_baffle_nd25fw4_v1lf_assembled.step",
        "top_baffle_nd25fw4_um_fit.step",
        "baffle_cable_routing_proud.png",
        "baffle_cable_routing_v1lf.png",
        "baffle_variants_drivers.png",
        "baffle_b1_drivers.png",
        "baffle_b2_drivers.png",
    }
    errors = []
    for name in sorted(required):
        path = state_dir / name
        if not path.is_file():
            errors.append(f"{state}: missing R6F review artifact: {name}")
            continue
        if path.suffix == ".step":
            if path.stat().st_size < 1024:
                errors.append(f"{state}: truncated STEP artifact: {name}")
                continue
            with path.open("rb") as stream:
                header = stream.read(32)
            if not header.startswith(b"ISO-10303-21;"):
                errors.append(f"{state}: invalid STEP signature: {name}")
            elif not _step_has_trailer(path):
                errors.append(f"{state}: incomplete STEP trailer: {name}")
        else:
            with path.open("rb") as stream:
                header = stream.read(24)
            if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append(f"{state}: invalid PNG signature: {name}")
                continue
            if not _png_has_trailer(path):
                errors.append(f"{state}: incomplete PNG trailer: {name}")
            try:
                png = _png_diagnostics(path)
            except Exception as exc:
                errors.append(f"{state}: PNG decode/CRC failure {name}: {exc}")
                continue
            width, height = png["width"], png["height"]
            min_width = 1600 if "routing" in name else 1000
            if width < min_width or height < 1200:
                errors.append(
                    f"{state}: undersized review PNG {name}: "
                    f"{width}x{height}")
            expected_mode = "1" if state == "floor_stand" else "0"
            if "drivers" in name:
                token = f"LX521_OVERLAY_R6F_{state}"
                expected_description = (
                    f"{token}; LX_STAND_FOOT={expected_mode}")
            else:
                profile, revision = (
                    ("V1LF", "R6F") if "v1lf" in name
                    else ("PROUD", "R6P"))
                token = f"LX521_{profile}_{revision}_{state}"
                expected_description = (
                    f"{token}; LX_STAND_FOOT={expected_mode}; "
                    f"LX_ROUTING_PROFILE={profile.lower()}"
                    + ("; LX_V1LF_SIDE_SECTION=roof_to_bore_solid_backfill"
                       if profile == "V1LF" else ""))
            if (png["title"] != token
                    or png["description"] != expected_description):
                errors.append(
                    f"{state}: parsed PNG state/profile metadata mismatch: "
                    f"{name}")
            if png["nonwhite_fraction"] < 0.005:
                errors.append(f"{state}: blank/near-white review PNG: {name}")
            if png["chromatic_fraction"] < 0.001:
                errors.append(f"{state}: review PNG lacks route/driver color: {name}")

    attachments = state_dir / "top_baffle_nd25fw4_v1lf_attachments.step"
    split = state_dir / "top_baffle_nd25fw4_v1lf_split.step"
    assembled = state_dir / "top_baffle_nd25fw4_v1lf_assembled.step"
    if attachments.is_file():
        has_support = _contains_bytes(
            attachments, b"addon_mount_floor_support")
        if has_support != (state == "floor_stand"):
            errors.append(
                f"{state}: attachments STEP has wrong floor-support state")
        token_error = _state_token_error(
            attachments,
            (b"lx521_v1lf_r6f_required_floor_support_and_optional_addons_floor"
             if state == "floor_stand"
             else b"lx521_v1lf_r6f_optional_addons_no_floor"),
            (b"lx521_v1lf_r6f_optional_addons_no_floor"
             if state == "floor_stand"
             else
             b"lx521_v1lf_r6f_required_floor_support_and_optional_addons_floor"),
            "attachments STEP")
        if token_error:
            errors.append(token_error)
        for removed in (
                b"addon_um_grommet_half_a",
                b"addon_um_grommet_half_b"):
            if _contains_bytes(attachments, removed):
                errors.append(
                    f"{state}: attachments STEP retains removed V1LF "
                    f"grommet child {removed.decode('ascii')}")
    if split.is_file():
        token_error = _state_token_error(
            split,
            (b"lx521_v1lf_r6f_core_2piece_floor"
             if state == "floor_stand"
             else b"lx521_v1lf_r6f_core_2piece_no_floor_fused_solid_web"),
            (b"lx521_v1lf_r6f_core_2piece_no_floor_fused_solid_web"
             if state == "floor_stand"
             else b"lx521_v1lf_r6f_core_2piece_floor"),
            "core STEP")
        if token_error:
            errors.append(token_error)
    if assembled.is_file():
        free_lm_token = b"REFERENCE_LM_D7p8_short_free_span_no_micro_duct"
        obsolete_lm_token = b"REFERENCE_LM_D7p8_integral_270deg_lead"
        if (not _contains_bytes(assembled, free_lm_token)
                or _contains_bytes(assembled, obsolete_lm_token)):
            errors.append(
                f"{state}: assembled STEP lacks the free-LM/no-micro-duct "
                "contract or retains the obsolete integral-lead label")
        current_route_tokens = (
            b"REFERENCE_UM_D7_LM_printed_cover_then_free_behind_UM_"
            b"R15_R20_Faston_handoff",
            b"REFERENCE_TS_D5p2_LM_UM_printed_then_free_behind_tweeter",
        )
        obsolete_route_tokens = (
            b"REFERENCE_UM_D7_buried_route_plus_R15_R20_flag_Faston_handoff",
            b"REFERENCE_TS_D5p2_crown_crossover_route",
        )
        missing_route_tokens = [
            token for token in current_route_tokens
            if not _contains_bytes(assembled, token)
        ]
        retained_route_tokens = [
            token for token in obsolete_route_tokens
            if _contains_bytes(assembled, token)
        ]
        if missing_route_tokens or retained_route_tokens:
            errors.append(
                f"{state}: assembled STEP has stale V1LF UM/T route labels "
                f"(missing={[token.decode('ascii') for token in missing_route_tokens]}, "
                f"obsolete={[token.decode('ascii') for token in retained_route_tokens]})")
        for token in (
                b"KEEP_CLEAR_faston_flag_pull_sweep_1_12mm",
                b"KEEP_CLEAR_faston_flag_pull_sweep_2_12mm"):
            if not _contains_bytes(assembled, token):
                errors.append(
                    f"{state}: assembled STEP lacks independent Faston "
                    f"pull sweep {token.decode('ascii')}")
        expected = (b"KEEP_CLEAR_three_floor_support"
                    if state == "floor_stand"
                    else b"KEEP_CLEAR_four_stock_bridge")
        opposite = (b"KEEP_CLEAR_four_stock_bridge"
                    if state == "floor_stand"
                    else b"KEEP_CLEAR_three_floor_support")
        if (not _contains_bytes(assembled, expected)
                or _contains_bytes(assembled, opposite)):
            errors.append(
                f"{state}: assembled STEP lacks state-specific hardware")
        token_error = _state_token_error(
            assembled,
            (b"lx521_v1lf_r6f_assembled_floor"
             if state == "floor_stand"
             else b"lx521_v1lf_r6f_assembled_no_floor_fused_solid_web"),
            (b"lx521_v1lf_r6f_assembled_no_floor_fused_solid_web"
             if state == "floor_stand"
             else b"lx521_v1lf_r6f_assembled_floor"),
            "assembled STEP")
        if token_error:
            errors.append(token_error)
        for removed in (
                b"addon_um_grommet_half_a",
                b"addon_um_grommet_half_b"):
            if _contains_bytes(assembled, removed):
                errors.append(
                    f"{state}: assembled STEP retains removed V1LF "
                    f"grommet child {removed.decode('ascii')}")
    fit_step = state_dir / "top_baffle_nd25fw4_um_fit.step"
    if fit_step.is_file():
        for removed in (
                b"v1lf_um_grommet_half_a_TPU_PRINT_PART",
                b"v1lf_um_grommet_half_b_TPU_PRINT_PART"):
            if _contains_bytes(fit_step, removed):
                errors.append(
                    f"{state}: UM-fit STEP retains removed V1LF "
                    f"grommet child {removed.decode('ascii')}")
    errors.extend(_release_manifest_errors(state_dir))
    return errors


def _v1lf_manifest_errors(root: Path) -> list[str]:
    state = root.parent.name
    if state not in {"floor_stand", "no_floor_stand"}:
        return []
    actual = {path.name for path in root.glob("lx521_top_v1lf_*.stl")}
    expected = {
        "lx521_top_v1lf_core_1of2_lm_carrier.stl",
        "lx521_top_v1lf_core_2of2_um_carrier.stl",
        "lx521_top_v1lf_addon_tweeter_crescent.stl",
    }
    if state == "floor_stand":
        expected.add("lx521_top_v1lf_addon_mount_floor_support.stl")
    errors = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{state}: missing V1LF artifacts: {', '.join(missing)}")
    if extra:
        errors.append(f"{state}: stale/extra V1LF artifacts: {', '.join(extra)}")
    forbidden_v1lf = {
        "lx521_top_v1lf_addon_um_grommet_half_a.stl",
        "lx521_top_v1lf_addon_um_grommet_half_b.stl",
        "lx521_coupon_10_um_split_grommet_half_a.stl",
        "lx521_coupon_11_um_split_grommet_half_b.stl",
        "lx521_coupon_14_v1lf_grommet_receiver.stl",
    }
    stale_v1lf = sorted(
        path.name for path in root.iterdir()
        if path.name in forbidden_v1lf)
    if stale_v1lf:
        errors.append(
            f"{state}: removed V1LF grommet artifacts remain: "
            f"{', '.join(stale_v1lf)}")
    stale_stage = sorted(
        path.name for path in (root.parent / ".v1lf_stage").glob(
            "addon_um_grommet_half_*.brep"))
    if stale_stage:
        errors.append(
            f"{state}: removed V1LF staged grommet BREPs remain: "
            f"{', '.join(stale_stage)}")
    expected_coupons = {
        Path(name).name
        for name in expected_artifact_names(state == "floor_stand")
        if name.startswith("stl/lx521_coupon_")
    }
    actual_coupons = {path.name for path in root.glob("lx521_coupon_*.stl")}
    if actual_coupons != expected_coupons:
        missing = sorted(expected_coupons - actual_coupons)
        extra = sorted(actual_coupons - expected_coupons)
        if missing:
            errors.append(f"{state}: missing coupons: {', '.join(missing)}")
        if extra:
            errors.append(f"{state}: stale/extra coupons: {', '.join(extra)}")
    forbidden_coupon = root / "lx521_coupon_12_v1lf_open_bore_jump.stl"
    if forbidden_coupon.exists():
        errors.append(f"{state}: stale open-window coupon remains")
    common_release = {
        "lx521_top_base_1of4_bottom.stl",
        "lx521_top_base_2of4_mid_left.stl",
        "lx521_top_base_3of4_mid_right.stl",
        "lx521_top_base_4of4_vase_b2.stl",
        "lx521_top_addonA_1of4_shoulder_top_left.stl",
        "lx521_top_addonA_2of4_shoulder_top_right.stl",
        "lx521_top_addonA_3of4_shoulder_bottom_left.stl",
        "lx521_top_addonA_4of4_shoulder_bottom_right.stl",
        "lx521_top_addonB1_1of2_wing_left.stl",
        "lx521_top_addonB1_2of2_wing_right.stl",
        "lx521_top_proud_addon_um_grommet_half_a.stl",
        "lx521_top_proud_addon_um_grommet_half_b.stl",
        "lx521_top_c7base_1of4_bottom.stl",
        "lx521_top_c7base_2of4_mid_left.stl",
        "lx521_top_c7base_3of4_mid_right.stl",
        "lx521_top_c7base_4of4_vase_b2.stl",
        "lx521_top_v0_4of4_vase.stl",
        "lx521_top_v1_4of4_vase.stl",
        "lx521_top_v1addonA_shoulder_top_left.stl",
        "lx521_top_v1addonA_shoulder_top_right.stl",
        "lx521_top_v1addonA_shoulder_bottom_left.stl",
        "lx521_top_v1addonA_shoulder_bottom_right.stl",
        "lx521_top_v1addonB1_wing_left.stl",
        "lx521_top_v1addonB1_wing_right.stl",
        "lx521_top_v1l_1of4_bottom.stl",
        "lx521_top_v1l_2of4_mid_left.stl",
        "lx521_top_v1l_3of4_mid_right.stl",
        "lx521_top_v1l_4of4_vase_b2.stl",
        "lx521_top_v1l_addon_um_grommet_half_a.stl",
        "lx521_top_v1l_addon_um_grommet_half_b.stl",
    }
    complete_expected = common_release | expected | expected_coupons
    if state == "floor_stand":
        complete_expected |= {
            "lx521_polar_base_1of2_base.stl",
            "lx521_polar_base_2of2_rotor.stl",
        }
    complete_actual = {path.name for path in root.glob("*.stl")}
    if complete_actual != complete_expected:
        missing = sorted(complete_expected - complete_actual)
        extra = sorted(complete_actual - complete_expected)
        if missing:
            errors.append(
                f"{state}: missing release STLs: {', '.join(missing)}")
        if extra:
            errors.append(
                f"{state}: stale/extra release STLs: {', '.join(extra)}")
    errors.extend(_review_artifact_errors(root.parent))
    return errors


def main() -> int:
    arguments = list(sys.argv[1:])
    require_release_authorized = False
    if "--require-release-authorized" in arguments:
        arguments.remove("--require-release-authorized")
        require_release_authorized = True
    unknown_options = [arg for arg in arguments if arg.startswith("-")]
    if unknown_options:
        print(
            "unknown option(s): " + ", ".join(unknown_options),
            file=sys.stderr)
        return 2
    roots = ([Path(arg) for arg in arguments] if arguments else
             [Path(__file__).parent / state / "stl"
              for state in ("floor_stand", "no_floor_stand")])
    files = sorted(path for root in roots if root.is_dir()
                   for path in root.glob("*.stl"))
    if not files:
        print("no STLs found", file=sys.stderr)
        return 1

    bad = 0
    good = 0
    for path in files:
        try:
            facts = stl_diagnostics(path)
        except ValueError as exc:
            bad += 1
            print(f"  DEFECT {exc}")
            continue
        defects = sum(int(facts[key]) for key in (
            "open", "over_shared", "winding", "degenerate", "duplicates",
            "nonfinite", "zero_volume", "negative_volume",
            "component_error"))
        status = "ok" if not defects else "DEFECT"
        bad += bool(defects)
        good += not defects
        print(
            f"  {status:6s} {path.parent.parent.name}/{path.name}: "
            f"{facts['triangles']} tris, {facts['open']} open, "
            f"{facts['over_shared']} over-shared, "
            f"{facts['winding']} winding, {facts['degenerate']} degenerate, "
            f"{facts['duplicates']} duplicate, {facts['nonfinite']} nonfinite, "
            f"{facts['components']} component(s), "
            f"signed volume {facts['signed_volume']:.2f} mm3")

    manifest_errors = [error for root in roots
                       for error in _v1lf_manifest_errors(root)]
    if require_release_authorized:
        for root in roots:
            manifest_path = root.parent / "v1lf_release_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(
                    encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                manifest_errors.append(
                    f"{root.parent.name}: cannot prove physical release "
                    f"authorization: {exc}")
                continue
            qualification = manifest.get("qualification")
            if (not isinstance(qualification, dict)
                    or qualification.get("release_authorized") is not True):
                manifest_errors.append(
                    f"{root.parent.name}: physical qualification is pending; "
                    "candidate artifacts must not be released")
    for error in manifest_errors:
        print(f"  DEFECT {error}")
    bad += len(manifest_errors)
    if len(roots) == 2:
        state_dirs = {root.parent.name: root.parent for root in roots}
        floor = state_dirs.get("floor_stand")
        no_floor = state_dirs.get("no_floor_stand")
        if floor and no_floor:
            for name in (
                    "top_baffle_nd25fw4_v1lf_split.step",
                    "top_baffle_nd25fw4_v1lf_attachments.step",
                    "top_baffle_nd25fw4_v1lf_assembled.step",
                    "top_baffle_nd25fw4_um_fit.step"):
                a, b = floor / name, no_floor / name
                if (a.is_file() and b.is_file()
                        and a.stat().st_size == b.stat().st_size
                        and _sha256(a) == _sha256(b)):
                    bad += 1
                    print(
                        f"  DEFECT floor/no-floor artifacts are identical: "
                        f"{name}")
            for name in (
                    "baffle_cable_routing_proud.png",
                    "baffle_cable_routing_v1lf.png",
                    "baffle_variants_drivers.png",
                    "baffle_b1_drivers.png",
                    "baffle_b2_drivers.png"):
                a, b = floor / name, no_floor / name
                if a.is_file() and b.is_file():
                    try:
                        identical = (_png_diagnostics(a)["pixel_sha256"]
                                     == _png_diagnostics(b)["pixel_sha256"])
                    except Exception:
                        # The state-local review check already reports the
                        # decode failure; do not hide it with a second crash.
                        identical = False
                    if identical:
                        bad += 1
                        print(
                            "  DEFECT floor/no-floor decoded PNG pixels are "
                            f"identical: {name}")
            for name in (
                    "stl/lx521_top_v1lf_core_1of2_lm_carrier.stl",
                    "stl/lx521_coupon_12_v1lf_closed_bore_bump.stl"):
                a, b = floor / name, no_floor / name
                if (a.is_file() and b.is_file()
                        and a.stat().st_size == b.stat().st_size
                        and _sha256(a) == _sha256(b)):
                    bad += 1
                    print(
                        "  DEFECT state-dependent meshes are identical: "
                        f"{name}")
            for state_dir in (floor, no_floor):
                v1lf = state_dir / "baffle_cable_routing_v1lf.png"
                proud = state_dir / "baffle_cable_routing_proud.png"
                if v1lf.is_file() and proud.is_file():
                    try:
                        identical = (
                            _png_diagnostics(v1lf)["pixel_sha256"]
                            == _png_diagnostics(proud)["pixel_sha256"])
                    except Exception:
                        identical = False
                else:
                    identical = False
                if identical:
                    bad += 1
                    print(
                        f"  DEFECT {state_dir.name}: proud and V1LF routing "
                        "PNGs are identical")
    print(f"{good}/{len(files)} STLs strict-manifold; "
          f"manifest errors {len(manifest_errors)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
