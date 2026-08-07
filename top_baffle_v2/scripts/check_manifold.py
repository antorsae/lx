"""Strict release sweep for every exported STL and Obi-Wan manifest.

The primary contract remains explicit: every undirected mesh edge is
shared by exactly two oppositely wound triangles.  The checker also
rejects malformed binary length, repeated/zero-area triangles, duplicate
facets and stale state-incompatible Obi-Wan artifacts.  A printable solid may
have one outward boundary plus inward-wound, fully nested cavity boundaries;
those are voids in one material body, not disconnected printable bodies.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import sys

# The exact edge/component sweep can consume substantial memory on dense
# release meshes even though it does not build CAD. Direct CLI use therefore
# receives the same process-tree cap and host-memory floor as Make.
if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))

import math
import hashlib
import json
import struct
from collections import Counter

from lx521_baffle.print_contract import (
    FrontDownContractError,
    validate_print_sidecar,
)
from lx521_baffle.io import sha256_file
from write_obiwan_release_manifest import (
    FORMAT_VERSION,
    QUALIFICATION_RECORD,
    expected_artifact_names,
    native_stage_record,
    qualification_record,
    source_hashes,
)


FLOOR_POLAR_SIDECAR_EXCLUSIONS = frozenset({
    "lx521_polar_base_1of2_base.stl",
    "lx521_polar_base_2of2_rotor.stl",
})
WING_SLUGS = ("flat", "graded")
WING_SIDES = ("left", "right")
WING_ROLES = ("lm_lower", "lm_upper", "um")
WING_TWO_PIECE_ROLES = ("lm_lower", "lm_um_upper")
EXPECTED_NONPOLAR_STATE_STL_COUNT = 39
EXPECTED_WING_STL_COUNT = 10


def expected_wing_stl_names(slug: str) -> frozenset[str]:
    if slug not in WING_SLUGS:
        raise ValueError(f"unknown released wing slug: {slug}")
    three_piece = {
        f"obiwan_wing_{slug}_{side}_{order}_of_3_{role}.stl"
        for side in WING_SIDES
        for order, role in enumerate(WING_ROLES, start=1)
    }
    two_piece = {
        f"obiwan_wing_{slug}_{side}_split2_{order}_of_2_{role}.stl"
        for side in WING_SIDES
        for order, role in enumerate(WING_TWO_PIECE_ROLES, start=1)
    }
    names = frozenset(three_piece | two_piece)
    if len(names) != EXPECTED_WING_STL_COUNT:
        raise RuntimeError(
            f"{slug}: released wing inventory count drifted to {len(names)}")
    return names


def _print_sidecar_inventory_errors(
        root: Path, expected_stl_names: set[str] | frozenset[str], *,
        excluded_stl_names: set[str] | frozenset[str] = frozenset(),
        actual_name_prefixes: tuple[str, ...] | None = None,
        label: str | None = None) -> list[str]:
    """Validate one exact STL-to-adjacent-sidecar release inventory."""
    scope = label or root.as_posix()
    expected_stls = set(expected_stl_names)
    excluded = set(excluded_stl_names)
    errors = []
    unknown_exclusions = sorted(excluded - expected_stls)
    if unknown_exclusions:
        errors.append(
            f"{scope}: print-sidecar exclusions are not released STLs: "
            f"{', '.join(unknown_exclusions)}")
    expected_sidecars = {
        Path(name).with_suffix(".print.json").name
        for name in expected_stls - excluded
    }
    actual_sidecars = {
        path.name for path in root.glob("*.print.json")
        if (path.is_file()
            and (actual_name_prefixes is None
                 or path.name.startswith(actual_name_prefixes)))
    }
    missing = sorted(expected_sidecars - actual_sidecars)
    extra = sorted(actual_sidecars - expected_sidecars)
    if missing:
        errors.append(
            f"{scope}: missing adjacent print sidecars: {', '.join(missing)}")
    if extra:
        errors.append(
            f"{scope}: stale/extra print sidecars: {', '.join(extra)}")
    for stl_name in sorted(expected_stls - excluded):
        stl = root / stl_name
        sidecar = stl.with_suffix(".print.json")
        # Missing release STLs and sidecars are reported by their exact
        # inventory gates. Validate only complete pairs so one absence does
        # not obscure the remaining independent defects.
        if not stl.is_file() or not sidecar.is_file():
            continue
        try:
            validate_print_sidecar(stl)
        except (FrontDownContractError, OSError) as exc:
            errors.append(
                f"{scope}: invalid print sidecar for {stl_name}: {exc}")
    return errors


def _wing_print_sidecar_errors(root: Path, slug: str) -> list[str]:
    """Gate exactly six front-down STL/sidecar pairs for one flat/graded wing."""
    expected = expected_wing_stl_names(slug)
    actual_stls = {path.name for path in root.glob("*.stl") if path.is_file()}
    errors = []
    missing = sorted(expected - actual_stls)
    extra = sorted(actual_stls - expected)
    if missing:
        errors.append(
            f"build/wings/{slug}: missing wing STLs: {', '.join(missing)}")
    if extra:
        errors.append(
            f"build/wings/{slug}: stale/extra wing STLs: {', '.join(extra)}")
    errors.extend(_print_sidecar_inventory_errors(
        root, expected, label=f"build/wings/{slug}"))
    return errors


def _triangle_vertices(data: bytes, triangle: int):
    """Return the three exact float32 vertices from one binary STL facet."""
    raw = struct.unpack_from("<9f", data, 84 + triangle * 50 + 12)
    return tuple(tuple(raw[i:i + 3]) for i in (0, 3, 6))


def _triangle_signed_volume(tri) -> float:
    return (
        tri[0][0] * (tri[1][1] * tri[2][2] - tri[1][2] * tri[2][1])
        - tri[0][1] * (tri[1][0] * tri[2][2] - tri[1][2] * tri[2][0])
        + tri[0][2] * (tri[1][0] * tri[2][1] - tri[1][1] * tri[2][0])
    ) / 6.0


def _solid_angle(tri, point) -> float:
    """Signed solid angle of ``tri`` at ``point`` (Van Oosterom-Strackee)."""
    vectors = tuple(tuple(vertex[i] - point[i] for i in range(3))
                    for vertex in tri)
    a, b, c = vectors
    lengths = tuple(math.sqrt(sum(value * value for value in vector))
                    for vector in vectors)
    if min(lengths) <= 1e-12:
        # A nested cavity witness must never lie on the outer boundary.
        return math.nan
    cross_bc = (
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    )
    numerator = sum(a[i] * cross_bc[i] for i in range(3))
    denominator = (
        lengths[0] * lengths[1] * lengths[2]
        + sum(a[i] * b[i] for i in range(3)) * lengths[2]
        + sum(b[i] * c[i] for i in range(3)) * lengths[0]
        + sum(c[i] * a[i] for i in range(3)) * lengths[1]
    )
    return 2.0 * math.atan2(numerator, denominator)


def stl_diagnostics(path: Path) -> dict[str, object]:
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
        tri = _triangle_vertices(data, (offset - 84) // 50)
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
            signed_volume += _triangle_signed_volume(tri)
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

    # Reuse the canonical edge keys already resident in memory and union their
    # vertices into connected *boundary* components. A solid with buried
    # cavities legitimately has more than one boundary component: exactly one
    # outward-wound outer shell and zero or more inward-wound nested shells.
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
    roots = {find(vertex) for vertex in parent}
    components = len(roots) if parent else 0

    component_stats = {
        root: {
            "signed_volume": 0.0,
            "triangles": 0,
            "minimum": [math.inf, math.inf, math.inf],
            "maximum": [-math.inf, -math.inf, -math.inf],
            "witness": None,
        }
        for root in roots
    }
    for triangle in range(triangles):
        tri = _triangle_vertices(data, triangle)
        if not all(math.isfinite(value) for vertex in tri for value in vertex):
            continue
        if tri[0] not in parent:
            # A fully collapsed facet has no real edge and therefore no
            # boundary component. It is already rejected as degenerate.
            continue
        root = find(tri[0])
        stats = component_stats[root]
        stats["signed_volume"] += _triangle_signed_volume(tri)
        stats["triangles"] += 1
        if stats["witness"] is None:
            stats["witness"] = tuple(
                sum(vertex[axis] for vertex in tri) / 3.0
                for axis in range(3))
        for vertex in tri:
            for axis in range(3):
                stats["minimum"][axis] = min(
                    stats["minimum"][axis], vertex[axis])
                stats["maximum"][axis] = max(
                    stats["maximum"][axis], vertex[axis])

    volume_epsilon = 1e-6
    positive_roots = [
        root for root, stats in component_stats.items()
        if stats["signed_volume"] > volume_epsilon
    ]
    negative_roots = [
        root for root, stats in component_stats.items()
        if stats["signed_volume"] < -volume_epsilon
    ]
    zero_roots = [
        root for root, stats in component_stats.items()
        if abs(stats["signed_volume"]) <= volume_epsilon
    ]
    outer_root = positive_roots[0] if len(positive_roots) == 1 else None
    nested_void_roots = []
    nonnested_void_roots = []
    if outer_root is not None:
        outer = component_stats[outer_root]
        bbox_epsilon = 1e-6
        candidates = []
        for root in negative_roots:
            stats = component_stats[root]
            bbox_nested = all(
                stats["minimum"][axis]
                > outer["minimum"][axis] + bbox_epsilon
                and stats["maximum"][axis]
                < outer["maximum"][axis] - bbox_epsilon
                for axis in range(3))
            if bbox_nested and stats["witness"] is not None:
                candidates.append(root)
            else:
                nonnested_void_roots.append(root)

        # Generalized winding/solid-angle containment is streamed over the
        # outer shell. This rejects a reversed loose shell merely located
        # inside the outer bounding box without retaining dense triangles.
        angles = {root: 0.0 for root in candidates}
        for triangle in range(triangles):
            tri = _triangle_vertices(data, triangle)
            if not all(math.isfinite(value)
                       for vertex in tri for value in vertex):
                continue
            if tri[0] not in parent:
                continue
            if find(tri[0]) != outer_root:
                continue
            for root in candidates:
                angles[root] += _solid_angle(
                    tri, component_stats[root]["witness"])
        for root in candidates:
            angle = angles[root]
            if math.isfinite(angle) and abs(angle) > 2.0 * math.pi:
                nested_void_roots.append(root)
            else:
                nonnested_void_roots.append(root)

    disconnected_material_components = max(0, len(positive_roots) - 1)
    component_error = int(
        outer_root is None
        or bool(zero_roots)
        or bool(nonnested_void_roots)
        or len(nested_void_roots) != len(negative_roots)
    )
    ordered_component_volumes = list(sorted(
        (float(stats["signed_volume"])
         for stats in component_stats.values()),
        reverse=True))
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
        "outer_components": len(positive_roots),
        "nested_void_components": len(nested_void_roots),
        "nonnested_void_components": len(nonnested_void_roots),
        "zero_volume_components": len(zero_roots),
        "disconnected_material_components": disconnected_material_components,
        "component_signed_volumes": ordered_component_volumes,
        "component_error": component_error,
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
    from PIL import Image

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
    path = state_dir / "obiwan_release_manifest.json"
    if not path.is_file():
        return [f"{state}: missing hash-backed Obi-Wan release manifest"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{state}: unreadable Obi-Wan release manifest: {exc}"]
    errors = []
    expected_header = {
        "format_version": FORMAT_VERSION,
        "variant": "Obi-Wan",
        "routing_revision": "R6F",
        "routing_profile": "obiwan",
        "state": state,
        "stand_foot": stand_foot,
    }
    for key, expected in expected_header.items():
        if data.get(key) != expected:
            errors.append(
                f"{state}: manifest {key}={data.get(key)!r}, "
                f"expected {expected!r}")
    expected_qualification = qualification_record(stand_foot)
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


def _floor_strength_report_errors(state_dir: Path) -> list[str]:
    """Validate the hash-bound analytical screen, not merely its presence."""
    state = state_dir.name
    json_path = state_dir / "obiwan_integrated_floor_strength.json"
    markdown_path = state_dir / "obiwan_integrated_floor_strength.md"
    if state != "floor_stand":
        stale = [path.name for path in (json_path, markdown_path)
                 if path.exists()]
        return ([f"{state}: stale floor-strength report(s): "
                 + ", ".join(stale)] if stale else [])
    missing = [path.name for path in (json_path, markdown_path)
               if not path.is_file()]
    if missing:
        return [f"{state}: missing integral-floor strength report(s): "
                + ", ".join(missing)]
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{state}: unreadable integral-floor strength JSON: {exc}"]
    from lx521_baffle.obiwan.floor_strength import (
        integral_floor_strength_facts,
    )
    # Compare the report in its actual wire format.  The analytical facts
    # intentionally use tuples for immutable dimensions, while JSON decodes
    # those arrays as lists; comparing the raw Python objects falsely marks
    # every nested tuple-bearing field stale even when the serialized values
    # are byte-for-byte current.
    expected = json.loads(json.dumps(integral_floor_strength_facts()))
    errors = []
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(
                f"{state}: floor-strength analytical field is stale: {key}")
    if set(payload) != {*expected, "production_geometry"}:
        errors.append(
            f"{state}: floor-strength JSON field set is malformed")
    geometry = payload.get("production_geometry")
    records = geometry.get("artifacts") if isinstance(geometry, dict) else None
    expected_steps = (
        "obiwan_split.step",
        "obiwan_lm_split.step",
    )
    if (not isinstance(records, list) or len(records) != 2
            or not isinstance(geometry.get("derivation"), str)):
        errors.append(
            f"{state}: floor-strength production-geometry binding malformed")
        records = []
    by_name = {
        Path(record.get("path", "")).name: record
        for record in records if isinstance(record, dict)
    }
    for name in expected_steps:
        step = state_dir / name
        record = by_name.get(name)
        if (not step.is_file() or not isinstance(record, dict)
                or record.get("bytes") != step.stat().st_size
                or record.get("sha256") != sha256_file(step)):
            errors.append(
                f"{state}: floor-strength STEP hash binding failed: {name}")
    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    if ("not FEA or physical qualification" not in markdown
            or "Free-standing lateral tip threshold" not in markdown
            or "PENDING" not in markdown
            or any(record.get("sha256", "") not in markdown
                   for record in records if isinstance(record, dict))):
        errors.append(
            f"{state}: floor-strength Markdown omits required limitations "
            "or geometry hashes")
    return errors


def _review_artifact_errors(state_dir: Path) -> list[str]:
    state = state_dir.name
    required = {
        "obiwan_split.step",
        "obiwan_lm_split.step",
        "obiwan_attachments.step",
        "obiwan_assembled.step",
        "um_fit.step",
        "baffle_cable_routing_proud.png",
        "baffle_cable_routing_obiwan.png",
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
                profile_token, profile_slug, revision = (
                    ("Obi-Wan", "obiwan", "R6F") if "obiwan" in name
                    else ("PROUD", "proud", "R6P"))
                token = f"LX521_{profile_token}_{revision}_{state}"
                expected_description = (
                    f"{token}; LX_STAND_FOOT={expected_mode}; "
                    f"LX_ROUTING_PROFILE={profile_slug}"
                    + ("; LX_OBIWAN_VIEWS=front_xy,route_depth"
                       "; LX_OBIWAN_CONTENT=LM_UM_routes_only"
                       "; LX_OBIWAN_TERMINAL_SERVICE_OVERLAY=0"
                       "; LX_OBIWAN_SEPARATE_FLOOR_SUPPORT=0"
                       if profile_slug == "obiwan" else ""))
            if (png["title"] != token
                    or png["description"] != expected_description):
                errors.append(
                    f"{state}: parsed PNG state/profile metadata mismatch: "
                    f"{name}")
            if png["nonwhite_fraction"] < 0.005:
                errors.append(f"{state}: blank/near-white review PNG: {name}")
            if png["chromatic_fraction"] < 0.001:
                errors.append(f"{state}: review PNG lacks route/driver color: {name}")

    attachments = state_dir / "obiwan_attachments.step"
    split = state_dir / "obiwan_split.step"
    lm_split = state_dir / "obiwan_lm_split.step"
    assembled = state_dir / "obiwan_assembled.step"
    if attachments.is_file():
        has_support = _contains_bytes(
            attachments, b"addon_mount_floor_support")
        if has_support:
            errors.append(
                f"{state}: attachments STEP retains deleted floor-support "
                "child")
        token_error = _state_token_error(
            attachments,
            (b"lx521_obiwan_r6f_optional_addons_floor_integrated_mount"
             if state == "floor_stand"
             else b"lx521_obiwan_r6f_optional_addons_no_floor"),
            (b"lx521_obiwan_r6f_optional_addons_no_floor"
             if state == "floor_stand"
             else
             b"lx521_obiwan_r6f_optional_addons_floor_integrated_mount"),
            "attachments STEP")
        if token_error:
            errors.append(token_error)
        if _contains_bytes(
                attachments,
                b"lx521_obiwan_r6f_required_floor_support_and_optional_addons_floor"):
            errors.append(
                f"{state}: attachments STEP retains obsolete required-"
                "floor-support root label")
        for removed in (
                b"addon_um_grommet_half_a",
                b"addon_um_grommet_half_b"):
            if _contains_bytes(attachments, removed):
                errors.append(
                    f"{state}: attachments STEP retains removed Obi-Wan "
                    f"grommet child {removed.decode('ascii')}")
    if split.is_file():
        token_error = _state_token_error(
            split,
            (b"lx521_obiwan_r6f_core_2piece_floor"
             if state == "floor_stand"
             else b"lx521_obiwan_r6f_core_2piece_no_floor_fused_solid_web"),
            (b"lx521_obiwan_r6f_core_2piece_no_floor_fused_solid_web"
             if state == "floor_stand"
             else b"lx521_obiwan_r6f_core_2piece_floor"),
            "core STEP")
        if token_error:
            errors.append(token_error)
        for optional in (
                b"optional_lm_keyed_1_of_2_bottom",
                b"optional_lm_keyed_2_of_2_top"):
            if _contains_bytes(split, optional):
                errors.append(
                    f"{state}: canonical core STEP contains mutually "
                    f"exclusive LM split child {optional.decode('ascii')}")
    if lm_split.is_file():
        token_error = _state_token_error(
            lm_split,
            (b"lx521_obiwan_r6f_optional_lm_keyed_split_floor"
             if state == "floor_stand" else
             b"lx521_obiwan_r6f_optional_lm_keyed_split_no_floor"),
            (b"lx521_obiwan_r6f_optional_lm_keyed_split_no_floor"
             if state == "floor_stand" else
             b"lx521_obiwan_r6f_optional_lm_keyed_split_floor"),
            "optional LM split STEP")
        if token_error:
            errors.append(token_error)
        for child in (
                b"optional_lm_keyed_1_of_2_bottom",
                b"optional_lm_keyed_2_of_2_top"):
            if not _contains_bytes(lm_split, child):
                errors.append(
                    f"{state}: optional LM split STEP lacks child "
                    f"{child.decode('ascii')}")
        for canonical in (b"core_lm_carrier", b"core_um_carrier"):
            if _contains_bytes(lm_split, canonical):
                errors.append(
                    f"{state}: optional LM split STEP contains canonical "
                    f"core child {canonical.decode('ascii')}")
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
                f"{state}: assembled STEP has stale Obi-Wan UM/T route labels "
                f"(missing={[token.decode('ascii') for token in missing_route_tokens]}, "
                f"obsolete={[token.decode('ascii') for token in retained_route_tokens]})")
        for token in (
                b"KEEP_CLEAR_faston_flag_pull_sweep_1_12mm",
                b"KEEP_CLEAR_faston_flag_pull_sweep_2_12mm"):
            if not _contains_bytes(assembled, token):
                errors.append(
                    f"{state}: assembled STEP lacks independent Faston "
                    f"pull sweep {token.decode('ascii')}")
        support_hardware = _contains_bytes(
            assembled, b"KEEP_CLEAR_three_floor_support")
        bridge_hardware = _contains_bytes(
            assembled, b"KEEP_CLEAR_four_stock_bridge")
        if support_hardware:
            errors.append(
                f"{state}: assembled STEP retains deleted floor-support "
                "hardware")
        if state == "floor_stand":
            if bridge_hardware:
                errors.append(
                    f"{state}: assembled STEP contains no-floor bridge "
                    "hardware")
        elif not bridge_hardware:
            errors.append(
                f"{state}: assembled STEP lacks stock-bridge hardware")
        token_error = _state_token_error(
            assembled,
            (b"lx521_obiwan_r6f_assembled_floor"
             if state == "floor_stand"
             else b"lx521_obiwan_r6f_assembled_no_floor_fused_solid_web"),
            (b"lx521_obiwan_r6f_assembled_no_floor_fused_solid_web"
             if state == "floor_stand"
             else b"lx521_obiwan_r6f_assembled_floor"),
            "assembled STEP")
        if token_error:
            errors.append(token_error)
        for removed in (
                b"addon_mount_floor_support",
                b"addon_um_grommet_half_a",
                b"addon_um_grommet_half_b"):
            if _contains_bytes(assembled, removed):
                errors.append(
                    f"{state}: assembled STEP retains removed Obi-Wan "
                    f"child {removed.decode('ascii')}")
        for optional in (
                b"optional_lm_keyed_1_of_2_bottom",
                b"optional_lm_keyed_2_of_2_top"):
            if _contains_bytes(assembled, optional):
                errors.append(
                    f"{state}: canonical assembled STEP contains mutually "
                    f"exclusive LM split child {optional.decode('ascii')}")
    fit_step = state_dir / "um_fit.step"
    if fit_step.is_file():
        for removed in (
                b"obiwan_um_grommet_half_a_TPU_PRINT_PART",
                b"obiwan_um_grommet_half_b_TPU_PRINT_PART"):
            if _contains_bytes(fit_step, removed):
                errors.append(
                    f"{state}: UM-fit STEP retains removed Obi-Wan "
                    f"grommet child {removed.decode('ascii')}")
    errors.extend(_floor_strength_report_errors(state_dir))
    errors.extend(_release_manifest_errors(state_dir))
    return errors


def _obiwan_manifest_errors(
        root: Path, *, obiwan_only: bool = False) -> list[str]:
    state = root.parent.name
    if (root.name == "stl" and state in WING_SLUGS
            and root.parent.parent.name == "wings"):
        return _wing_print_sidecar_errors(root, state)
    if state not in {"floor_stand", "no_floor_stand"}:
        return []
    actual = {path.name for path in root.glob("obiwan_*.stl")}
    expected = {
        "obiwan_core_1_of_2_lm_carrier.stl",
        "obiwan_core_2_of_2_um_carrier.stl",
        "obiwan_optional_lm_keyed_1_of_2_bottom.stl",
        "obiwan_optional_lm_keyed_2_of_2_top.stl",
        "obiwan_addon_tweeter_crescent.stl",
    }
    errors = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{state}: missing Obi-Wan artifacts: {', '.join(missing)}")
    if extra:
        errors.append(f"{state}: stale/extra Obi-Wan artifacts: {', '.join(extra)}")
    # Retired artifacts keep the exact names they were removed under, so a
    # warm tree that still carries one is still caught.
    forbidden_obiwan = {
        "lx521_top_obiwan_addon_mount_floor_support.stl",
        "lx521_top_obiwan_addon_um_grommet_half_a.stl",
        "lx521_top_obiwan_addon_um_grommet_half_b.stl",
        "lx521_coupon_10_um_split_grommet_half_a.stl",
        "lx521_coupon_11_um_split_grommet_half_b.stl",
        "lx521_coupon_14_obiwan_grommet_receiver.stl",
    }
    stale_obiwan = sorted(
        path.name for path in root.iterdir()
        if path.name in forbidden_obiwan)
    if stale_obiwan:
        errors.append(
            f"{state}: removed Obi-Wan artifacts remain: "
            f"{', '.join(stale_obiwan)}")
    stale_stage = sorted(
        path.name for path in (root.parent / ".obiwan_stage").glob(
            "addon_um_grommet_half_*.brep"))
    if stale_stage:
        errors.append(
            f"{state}: removed Obi-Wan staged grommet BREPs remain: "
            f"{', '.join(stale_stage)}")
    stale_floor_stage = [
        name for name in (
            "addon_mount_floor_support.brep",
            "review_state_hardware.brep",
        )
        if (state == "floor_stand"
            and (root.parent / ".obiwan_stage" / name).exists())
    ]
    if stale_floor_stage:
        errors.append(
            f"{state}: deleted floor-support staged BREPs remain: "
            f"{', '.join(stale_floor_stage)}")
    expected_coupons = {
        Path(name).name
        for name in expected_artifact_names(state == "floor_stand")
        if (name.startswith("stl/lx521_coupon_")
            and name.endswith(".stl"))
    }
    actual_coupons = {path.name for path in root.glob("lx521_coupon_*.stl")}
    if actual_coupons != expected_coupons:
        missing = sorted(expected_coupons - actual_coupons)
        extra = sorted(actual_coupons - expected_coupons)
        if missing:
            errors.append(f"{state}: missing coupons: {', '.join(missing)}")
        if extra:
            errors.append(f"{state}: stale/extra coupons: {', '.join(extra)}")
    forbidden_coupon = root / "lx521_coupon_12_obiwan_open_bore_jump.stl"
    if forbidden_coupon.exists():
        errors.append(f"{state}: stale open-window coupon remains")
    common_release = {
        "stock_1_of_4_bottom.stl",
        "stock_2_of_4_mid_left.stl",
        "stock_3_of_4_mid_right.stl",
        "stock_4_of_4_vase_b2.stl",
        "stock_shoulder_1_of_4_top_left.stl",
        "stock_shoulder_2_of_4_top_right.stl",
        "stock_shoulder_3_of_4_bottom_left.stl",
        "stock_shoulder_4_of_4_bottom_right.stl",
        "stock_wing_1_of_2_left.stl",
        "stock_wing_2_of_2_right.stl",
        "stock_um_grommet_half_a.stl",
        "stock_um_grommet_half_b.stl",
        "slim_shoulder_2_of_4_top_left.stl",
        "slim_shoulder_4_of_4_top_right.stl",
        "slim_shoulder_1_of_4_bottom_left.stl",
        "slim_shoulder_3_of_4_bottom_right.stl",
        "slim_wing_1_of_2_left.stl",
        "slim_wing_2_of_2_right.stl",
        "slim_1_of_4_bottom.stl",
        "slim_2_of_4_mid_left.stl",
        "slim_3_of_4_mid_right.stl",
        "slim_4_of_4_vase_b2.stl",
        "slim_um_grommet_half_a.stl",
        "slim_um_grommet_half_b.stl",
    }
    complete_expected = expected | expected_coupons
    if not obiwan_only:
        complete_expected |= common_release
    if state == "floor_stand" and not obiwan_only:
        complete_expected |= {
            "lx521_polar_base_1of2_base.stl",
            "lx521_polar_base_2of2_rotor.stl",
        }
    complete_actual = {
        path.name for path in root.glob("*.stl")
        if (not obiwan_only
            or path.name.startswith((
                "obiwan_", "lx521_coupon_")))
    }
    if complete_actual != complete_expected:
        missing = sorted(complete_expected - complete_actual)
        extra = sorted(complete_actual - complete_expected)
        if missing:
            errors.append(
                f"{state}: missing release STLs: {', '.join(missing)}")
        if extra:
            errors.append(
                f"{state}: stale/extra release STLs: {', '.join(extra)}")
    polar_exclusions = (
        FLOOR_POLAR_SIDECAR_EXCLUSIONS
        if state == "floor_stand" and not obiwan_only else frozenset()
    )
    if (not obiwan_only
            and len(complete_expected - set(polar_exclusions))
            != EXPECTED_NONPOLAR_STATE_STL_COUNT):
        errors.append(
            f"{state}: nonpolar release STL count drifted from "
            f"{EXPECTED_NONPOLAR_STATE_STL_COUNT} to "
            f"{len(complete_expected - set(polar_exclusions))}")
    errors.extend(_print_sidecar_inventory_errors(
        root, complete_expected, excluded_stl_names=polar_exclusions,
        actual_name_prefixes=(
            ("obiwan_", "lx521_coupon_")
            if obiwan_only else None),
        label=state))
    errors.extend(_review_artifact_errors(root.parent))
    return errors


def main() -> int:
    arguments = list(sys.argv[1:])
    require_release_authorized = False
    obiwan_only = False
    stl_only = False
    metadata_only = False
    if "--require-release-authorized" in arguments:
        arguments.remove("--require-release-authorized")
        require_release_authorized = True
    if "--obiwan-only" in arguments:
        arguments.remove("--obiwan-only")
        obiwan_only = True
    if "--stl-only" in arguments:
        arguments.remove("--stl-only")
        stl_only = True
    if "--metadata-only" in arguments:
        arguments.remove("--metadata-only")
        metadata_only = True
    if stl_only and metadata_only:
        print(
            "--stl-only and --metadata-only are mutually exclusive",
            file=sys.stderr)
        return 2
    if stl_only and (require_release_authorized or obiwan_only):
        print(
            "--stl-only cannot be combined with manifest options",
            file=sys.stderr)
        return 2
    unknown_options = [arg for arg in arguments if arg.startswith("-")]
    if unknown_options:
        print(
            "unknown option(s): " + ", ".join(unknown_options),
            file=sys.stderr)
        return 2
    default_roots = [
        PROJECT_ROOT / "build" / state / "stl"
        for state in ("floor_stand", "no_floor_stand")
    ]
    if stl_only:
        roots = []
        files = sorted(Path(arg) for arg in arguments)
    else:
        roots = ([Path(arg) for arg in arguments]
                 if arguments else default_roots)
        files = ([] if metadata_only else sorted(
            path for root in roots if root.is_dir()
            for path in root.glob("*.stl")))
    if not files and not metadata_only:
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
            f"{facts['components']} boundary component(s) "
            f"({facts['nested_void_components']} nested void), "
            f"signed volume {facts['signed_volume']:.2f} mm3")

    manifest_errors = ([] if stl_only else [
        error for root in roots
        for error in _obiwan_manifest_errors(root, obiwan_only=obiwan_only)])
    if require_release_authorized:
        for root in roots:
            manifest_path = root.parent / "obiwan_release_manifest.json"
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
                continue
            configurations = qualification.get("configurations")
            if (qualification.get("authorization_scope")
                    != "all_shipped_lm_print_forms"
                    or not isinstance(configurations, dict)
                    or not configurations
                    or any(
                        not isinstance(record, dict)
                        or record.get("release_authorized") is not True
                        for record in configurations.values())):
                manifest_errors.append(
                    f"{root.parent.name}: every shipped LM print form must "
                    "be independently authorized before release")
    for error in manifest_errors:
        print(f"  DEFECT {error}")
    bad += len(manifest_errors)
    if not stl_only and len(roots) == 2:
        state_dirs = {root.parent.name: root.parent for root in roots}
        floor = state_dirs.get("floor_stand")
        no_floor = state_dirs.get("no_floor_stand")
        if floor and no_floor:
            for name in (
                    "obiwan_split.step",
                    "obiwan_lm_split.step",
                    "obiwan_attachments.step",
                    "obiwan_assembled.step",
                    "um_fit.step"):
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
                    "baffle_cable_routing_obiwan.png",
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
                    "stl/obiwan_core_1_of_2_lm_carrier.stl",
                    "stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl",
                    "stl/lx521_coupon_12_obiwan_closed_bore_bump.stl"):
                a, b = floor / name, no_floor / name
                if (a.is_file() and b.is_file()
                        and a.stat().st_size == b.stat().st_size
                        and _sha256(a) == _sha256(b)):
                    bad += 1
                    print(
                        "  DEFECT state-dependent meshes are identical: "
                        f"{name}")
            for state_dir in (floor, no_floor):
                obiwan = state_dir / "baffle_cable_routing_obiwan.png"
                proud = state_dir / "baffle_cable_routing_proud.png"
                if obiwan.is_file() and proud.is_file():
                    try:
                        identical = (
                            _png_diagnostics(obiwan)["pixel_sha256"]
                            == _png_diagnostics(proud)["pixel_sha256"])
                    except Exception:
                        identical = False
                else:
                    identical = False
                if identical:
                    bad += 1
                    print(
                        f"  DEFECT {state_dir.name}: proud and Obi-Wan routing "
                        "PNGs are identical")
    print(f"{good}/{len(files)} STLs strict-manifold; "
          f"manifest errors {len(manifest_errors)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
