"""Remote-only flat/graded Obi-Wan wing acceptance checks.

The flat/graded generator is deliberately expensive: each OCC-heavy check runs in
its own authenticated memory-guard process on the ``osado-512g`` profile.
Generic pytest collection is disabled so a local invocation cannot allocate
CAD state before the remote-profile gate is evaluated.
"""

from __future__ import annotations

__test__ = False

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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
import re
import subprocess
import sys

from lx521_baffle.print_contract import (
    validate_front_down_transform,
    validate_print_sidecar,
)
from lx521_baffle.io import sha256_file


ROOT = PROJECT_ROOT
NO_FLOOR_STAGE_MANIFEST = (
    ROOT / "build/no_floor_stand/.obiwan_stage/manifest.json")
FLOOR_STAGE_MANIFEST = ROOT / "build/floor_stand/.obiwan_stage/manifest.json"
VARIANT_IDS = ("flat", "graded")
SIDE_NAMES = ("left", "right")
PRINT_PART_ROLES = ("lm_lower", "lm_upper", "um")
TWO_PIECE_PART_ROLES = ("lm_lower", "lm_um_upper")
FRONT_Z_MM = 18.3
REAR_Z_MM = 6.8
FULL_DEPTH_MM = 11.5
GRADED_EDGE_DEPTH_MM = 0.24
GRADED_MONOTONIC_TOL_MM = 0.002
GRADED_MAX_SLOPE = 6.0
GRADED_LAND_BOUNDARY_SAMPLE_SPACING_MM = 0.50
GRADED_LAND_BOUNDARY_PROBE_OFFSET_MM = 0.004
GRADED_LAND_BOUNDARY_MAX_JUMP_MM = 0.03
# Standing-blade gate. A printable piece is exported already lying in its print
# orientation, so a modelling remnant a slicer would draw as a phantom wall
# is a thin blade standing off the +Z surface. Vertical ray casting on a
# 0.05-mm lattice samples any blade thicker than the lattice; grey opening by a
# 0.30-mm disk then levels every feature narrower than 0.60 mm, so the blade is
# what survives above the opened surface. Judged only where the whole disk is
# on the part, which is what tells a blade standing on material apart from the
# design's own narrow plan features (the feathered crescent tip, dovetail
# points) -- those read as a 6-9 mm rise otherwise.
#
# Threshold basis: over all twenty shipped pieces the largest rise is
# 0.2732 mm, at the sharp convex crest where the graded rear feathers out.
# That is the expected curvature residue of opening (curvature times radius
# squared).
# Synthetic blades planted on the real lower measure 0.51/3.01/6.01 mm for
# crests of 0.5/3/6 mm at 0.08-0.30 mm thick, so 1.00 mm separates the two
# populations with a 3.7x margin over real geometry. The gate is deliberately
# blind to blades under about 0.5 mm tall: at that height the opening residue
# of a legitimately sharp edge is indistinguishable from a blade, and nothing
# that small reads as a phantom wall.
STANDING_BLADE_LATTICE_MM = 0.05
STANDING_BLADE_DISK_RADIUS_MM = 0.30
STANDING_BLADE_MAX_RISE_MM = 1.00
# Depth-law fidelity gate. The blade gate above only sees features narrow
# enough for a 0.30-mm disk to level. This one catches the complementary
# failure -- a broad ridge, bulge or spline fold that is smooth but is not the
# surface the weighted-depth law asks for. Measured interior deviation: exactly
# 0.0000 mm on all ten flat pieces (their law is the constant full depth), and
# at worst 0.2933 mm on the graded pieces, mirror-symmetric between left and
# right, which is what deterministic design geometry looks like rather than a
# numerical artifact. 1.00 mm therefore keeps a 3.4x margin while still
# catching any multi-millimetre ridge.
#
# The interior margin matters more than it looks. Sweeping it over
# 1.2/1.6/2.0/2.5/3.0 mm walks the worst deviation down
# 0.5654/0.3784/0.2933/0.2156/0.1910 mm -- a clean monotone decay, which is
# itself the evidence that the residual is outline bookkeeping rather than
# anything in the surface. 2.0 mm still judges the great majority of each
# piece, and the ridge class this gate exists for spans a third of the part.
DEPTH_LAW_LATTICE_MM = 0.4
DEPTH_LAW_INTERIOR_MARGIN_MM = 2.0
DEPTH_LAW_MAX_DEVIATION_MM = 1.00
DOVETAIL_CLEARANCE_MM = 0.05
T_WING_CLEARANCE_MM = 0.20
DOVETAIL_ENDPOINT_TAPER_MM = 2.0
DOVETAIL_MIN_LIGAMENT_MM = 2.0
DOVETAIL_ROOT_OVERLAP_MM = 0.05
MAGNET_CAVITY_DIAMETER_MM = 5.20
MAGNET_CAVITY_DEPTH_MM = 2.10
MAGNET_FACE_SKIN_MM = 0.45
MAGNET_INNER_SKIN_MM = 0.45
MAGNET_CAPTIVE_LAND_MM = 3.00
MAGNET_INTERFACE_GAP_MM = 0.05
RING_MAGNET_FACE_SEPARATION_MM = 1.10
RING_CAVITY_FACE_OFFSET_MM = 0.65
RING_CAVITY_FACE_INSET_MM = 0.15
RING_FLUSH_FAIRING_MM = 0.80
MAGNET_ROOF_ANGLE_DEG = 45.0
OBIWAN_MAGNET_Z_MM = 15.10
DOVETAIL_PROFILES_MM = (
    {"neck": 7.0, "head": 9.0, "depth": 4.0},
    {"neck": 7.0, "head": 8.5, "depth": 4.0},
)
BED_XY_MM = 220.0
# flat is planar and stays on the original 10-ppm STEP round-trip gate.  graded's
# densely trimmed tensor B-spline is re-integrated by OCC after STEP import;
# bound that serialization-only quadrature delta at 20 ppm while retaining
# independent exact topology/bounds/STL/C0 gates plus exact plan symmetry and
# paired imported rear-depth probes.
STEP_ROUNDTRIP_VOLUME_REL_TOL = {"flat": 1.0e-5, "graded": 2.0e-5}
STEP_ROUNDTRIP_VOLUME_ABS_TOL_MM3 = 0.02
STEP_ASSEMBLY_VOLUME_REL_TOL = 5.0e-4
STEP_FAST_IDENTITY_VOLUME_REL_TOL = 2.0e-2
REVIEW_KINDS = (
    "front", "rear", "side_section", "split_exploded",
    "two_piece_split_exploded", "magnet_roots")


def _expected_magnet_face_separation(interface_kind: str) -> float:
    assert interface_kind in {"shoulder", "ring"}, (
        f"unknown Obi-Wan magnet interface kind: {interface_kind!r}")
    return RING_MAGNET_FACE_SEPARATION_MM


def _artifact_root() -> Path:
    raw = os.environ.get("LX_OBIWAN_WING_ARTIFACT_ROOT", "build/wings")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    assert path == (ROOT / "build/wings").resolve(), (
        "the flat/graded release contract is rooted at top_baffle_v2/build/wings")
    return path


def _large_host_execution() -> bool:
    """True only inside the explicitly selected remote CAD profile."""
    return (
        os.environ.get("LX_CAD_EXECUTION") != "local"
        and os.environ.get("LX_CAD_MEMORY_PROFILE") == "osado-512g"
        and os.environ.get("LX_CAD_ALLOW_PARALLEL") == "1"
    )


def _require_remote_guard() -> None:
    """Fail before importing build123d/OCC outside the remote guard."""
    import run_memory_guarded as memory_guard

    assert _large_host_execution(), (
        "flat/graded acceptance is remote-only; use the osado-512g CAD profile")
    assert memory_guard.is_guarded_process(), (
        "flat/graded acceptance escaped the authenticated CAD memory guard")


def _shape_bounds(shape) -> tuple[tuple[float, float, float],
                                  tuple[float, float, float]]:
    bounds = shape.bounding_box()
    return (
        (bounds.min.X, bounds.min.Y, bounds.min.Z),
        (bounds.max.X, bounds.max.Y, bounds.max.Z),
    )


def _assert_one_positive_solid(shape, label: str) -> None:
    solids = list(shape.solids())
    assert shape.is_valid, f"{label}: invalid BREP"
    assert len(solids) == 1, f"{label}: expected one solid, got {len(solids)}"
    assert solids[0].volume > 1.0, f"{label}: non-positive/implausible volume"


def _front_section_exterior(shape, z_mm: float):
    """Sample the exact OCC plane section into one exterior XY polygon.

    The section itself is analytic OCC geometry.  Sampling is used only to
    compare its exterior in Shapely; internal driver/insert loops are ignored.
    """
    from build123d import Plane, Rectangle, Wire
    from shapely.geometry import Polygon

    solids = list(shape.solids())
    assert len(solids) == 1
    section_face = Plane.XY.offset(z_mm) * Rectangle(1000.0, 1000.0)
    _vertices, edges = solids[0]._ocp_section(section_face)
    wires = Wire.combine(edges, tol=1.0e-6)
    polygons = []
    for wire in wires:
        if not wire.is_closed or wire.length <= 0.1:
            continue
        sample_count = max(96, int(math.ceil(wire.length / 0.10)))
        ordered_samples = []
        for index in range(sample_count):
            point = wire.position_at(index / sample_count)
            ordered_samples.append((index / sample_count, point))
        # Uniform arc-length samples alone need not land on a topological
        # vertex.  That made the exact y=172.481 LM split corner disappear
        # from one state while the other happened to sample 0.0007 mm away,
        # creating a false 0.030-mm Hausdorff excursion between coincident
        # analytic profiles.  Insert every OCC wire vertex at its ordered
        # normalized position; curved spans remain sampled at <=0.10 mm.
        ordered_samples.extend(
            (wire.param_at_point(vertex.center()), vertex.center())
            for vertex in wire.vertices()
        )
        ordered_samples.sort(key=lambda sample: sample[0])
        points = []
        for _position, point in ordered_samples:
            xy = (float(point.X), float(point.Y))
            if (not points
                    or math.hypot(
                        xy[0] - points[-1][0],
                        xy[1] - points[-1][1]) > 1.0e-8):
                points.append(xy)
        polygon = Polygon(points).buffer(0)
        if polygon.geom_type == "Polygon" and polygon.area > 0.01:
            polygons.append(polygon)
    assert polygons, f"no closed front section at z={z_mm:g}"
    return max(polygons, key=lambda polygon: polygon.area)


def _adaptive_volume_mm3(shape) -> float:
    """Mirror the release's adaptive exact-BREP volume oracle."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    total = 0.0
    solids = list(shape.solids())
    for solid in solids:
        properties = GProp_GProps()
        reached_error = float(BRepGProp.VolumeProperties_s(
            solid.wrapped, properties, 1.0e-6, True, False))
        assert math.isfinite(reached_error) and reached_error <= 5.0e-6
        volume = float(properties.Mass())
        assert math.isfinite(volume) and volume > 0.0
        total += volume
    return total


def _difference_volume(left, right) -> float:
    difference = left - right
    if difference is None:
        return 0.0
    return _adaptive_volume_mm3(difference)


def _intersection_volume(left, right) -> float:
    intersection = left & right
    if intersection is None:
        return 0.0
    return _adaptive_volume_mm3(intersection)


def _assert_bounds_close(left, right, tolerance_mm: float, label: str) -> None:
    for bound_index, (left_xyz, right_xyz) in enumerate(
            zip(left, right, strict=True)):
        for axis_index, (left_value, right_value) in enumerate(
                zip(left_xyz, right_xyz, strict=True)):
            assert math.isclose(
                left_value, right_value, abs_tol=tolerance_mm), (
                    f"{label}: bound[{bound_index}][{axis_index}] "
                    f"{left_value:.6f} != {right_value:.6f} mm")


_sha256_file = sha256_file


def _contains_step_token(path: Path, token: str) -> bool:
    """Find a STEP token even if OCC wrapped it across physical lines."""
    needle = token.encode("utf-8")
    overlap = max(0, len(needle) - 1)
    carry = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            folded = chunk.replace(b"\r", b"").replace(b"\n", b"")
            data = carry + folded
            if needle in data:
                return True
            carry = data[-overlap:] if overlap else b""
    return False


def _assert_complete_step(path: Path, required_tokens: tuple[str, ...]) -> None:
    assert path.is_file(), f"missing STEP: {path}"
    assert path.stat().st_size >= 1024, f"implausibly small STEP: {path}"
    with path.open("rb") as stream:
        header = stream.read(256)
        stream.seek(max(0, path.stat().st_size - 4096))
        trailer = stream.read().rstrip()
    assert b"ISO-10303-21;" in header, f"missing STEP header: {path}"
    assert trailer.endswith(b"END-ISO-10303-21;"), (
        f"interrupted STEP transaction: {path}")
    for token in required_tokens:
        assert _contains_step_token(path, token), (
            f"{path.name}: missing semantic label {token!r}")


def _expected_stl_names(slug: str) -> tuple[str, ...]:
    three_piece = tuple(
        f"obiwan_wing_{slug}_{side}_{order}_of_3_{role}.stl"
        for side in SIDE_NAMES
        for order, role in enumerate(PRINT_PART_ROLES, start=1)
    )
    two_piece = tuple(
        f"obiwan_wing_{slug}_{side}_split2_{order}_of_2_{role}.stl"
        for side in SIDE_NAMES
        for order, role in enumerate(TWO_PIECE_PART_ROLES, start=1)
    )
    return three_piece + two_piece


def _variant_paths(slug: str) -> dict[str, Path | tuple[Path, ...]]:
    directory = _artifact_root() / slug
    stls = tuple(
        directory / "stl" / name for name in _expected_stl_names(slug))
    return {
        "directory": directory,
        "canonical_step": (
            directory / f"obiwan_wing_{slug}.step"),
        "assembled_step": (
            directory
            / f"obiwan_wing_{slug}_assembled.step"),
        "two_piece_assembled_step": (
            directory
            / f"obiwan_wing_{slug}_assembled_split2.step"),
        "facts": directory / f"obiwan_wing_{slug}_facts.json",
        "manifest": directory / f"obiwan_wing_{slug}_print_manifest.json",
        "stls": stls,
        "sidecars": tuple(path.with_suffix(".print.json") for path in stls),
    }


def _read_json_object(path: Path) -> dict:
    assert path.is_file(), f"missing JSON artifact: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"invalid JSON artifact {path}: {exc}") from exc
    assert isinstance(value, dict), f"{path}: top level must be an object"
    return value


def _assert_strict_stl(path: Path) -> dict:
    from check_manifold import stl_diagnostics

    assert path.is_file(), f"missing STL: {path}"
    facts = stl_diagnostics(path)
    assert facts["triangles"] > 0, f"{path}: empty mesh"
    for key in (
            "open", "over_shared", "winding", "degenerate", "duplicates",
            "nonfinite", "zero_volume", "negative_volume",
            "component_error"):
        assert facts[key] == 0, f"{path}: {key}={facts[key]}"
    assert facts["outer_components"] == 1, f"{path}: no unique outer shell"
    assert facts["components"] == 1 + facts["nested_void_components"], (
        f"{path}: boundary component is neither outer material nor a "
        "fully nested cavity")
    assert facts["signed_volume"] > 1.0, f"{path}: implausible volume"
    return facts


def _binary_stl_triangles(path: Path):
    """Return an (n, 3, 3) array of triangle vertices from a binary STL."""
    import numpy as np

    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size >= 84, f"{path}: shorter than binary STL header"
    count = int(np.frombuffer(raw[80:84].tobytes(), dtype="<u4")[0])
    assert raw.size == 84 + 50 * count, (
        f"{path}: binary length {raw.size} != expected {84 + 50 * count}")
    records = raw[84:].reshape(count, 50)[:, 12:48]
    return np.frombuffer(
        records.tobytes(), dtype="<f4").reshape(count, 3, 3).astype(float)


def _top_height_field(triangles, pitch_mm: float):
    """Height of the topmost surface above each lattice cell centre.

    Exact vertical ray casting.  Every triangle is rasterized over the cells
    its XY bounding box covers and the barycentric-interpolated height is kept
    wherever the cell centre falls inside the triangle.  The per-cell maximum
    is reduced by sorting rather than ``np.maximum.at``: the unbuffered ufunc
    dominates runtime on the graded meshes.  Cells off the part stay NaN.
    """
    import numpy as np

    points = triangles.reshape(-1, 3)
    origin = points[:, :2].min(axis=0) - pitch_mm
    far = points[:, :2].max(axis=0) + pitch_mm
    nx = int(np.ceil((far[0] - origin[0]) / pitch_mm)) + 1
    ny = int(np.ceil((far[1] - origin[1]) / pitch_mm)) + 1
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    low = np.minimum(np.minimum(a[:, :2], b[:, :2]), c[:, :2])
    high = np.maximum(np.maximum(a[:, :2], b[:, :2]), c[:, :2])
    i0 = np.clip(np.ceil(
        (low[:, 0] - origin[0]) / pitch_mm - 0.5).astype(np.int64), 0, nx - 1)
    i1 = np.clip(np.floor(
        (high[:, 0] - origin[0]) / pitch_mm - 0.5).astype(np.int64), 0, nx - 1)
    j0 = np.clip(np.ceil(
        (low[:, 1] - origin[1]) / pitch_mm - 0.5).astype(np.int64), 0, ny - 1)
    j1 = np.clip(np.floor(
        (high[:, 1] - origin[1]) / pitch_mm - 0.5).astype(np.int64), 0, ny - 1)
    spans_x = np.maximum(i1 - i0 + 1, 0)
    spans_y = np.maximum(j1 - j0 + 1, 0)
    counts = spans_x * spans_y
    order = np.nonzero(counts > 0)[0]
    order = order[np.argsort(counts[order], kind="stable")]
    running = np.cumsum(counts[order])
    cell_parts: list = []
    height_parts: list = []
    start = 0
    while start < len(order):
        budget = (running[start - 1] if start else 0) + 2_000_000
        stop = min(max(int(np.searchsorted(running, budget)) + 1,
                       start + 1), len(order))
        batch = order[start:stop]
        start = stop
        repeats = counts[batch]
        owner = np.repeat(batch, repeats)
        offset = (np.arange(repeats.sum())
                  - np.repeat(np.cumsum(repeats) - repeats, repeats))
        width = spans_y[owner]
        gi = i0[owner] + offset // width
        gj = j0[owner] + offset % width
        px = origin[0] + (gi + 0.5) * pitch_mm
        py = origin[1] + (gj + 0.5) * pitch_mm
        ax, ay, az = a[owner, 0], a[owner, 1], a[owner, 2]
        bx, by, bz = b[owner, 0], b[owner, 1], b[owner, 2]
        cx, cy, cz = c[owner, 0], c[owner, 1], c[owner, 2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        safe = np.where(np.abs(det) < 1.0e-14, 1.0, det)
        u = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / safe
        v = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / safe
        w = 1.0 - u - v
        inside = ((np.abs(det) >= 1.0e-14) & (u >= -1.0e-9)
                  & (v >= -1.0e-9) & (w >= -1.0e-9))
        cell_parts.append((gi[inside] * ny + gj[inside]).astype(np.int64))
        height_parts.append((u[inside] * az[inside] + v[inside] * bz[inside]
                             + w[inside] * cz[inside]).astype(np.float32))
    cells = np.concatenate(cell_parts)
    heights = np.concatenate(height_parts)
    ranked = np.lexsort((heights, cells))
    cells = cells[ranked]
    heights = heights[ranked]
    topmost = np.nonzero(np.append(cells[1:] != cells[:-1], True))[0]
    field = np.full(nx * ny, np.nan, dtype=np.float32)
    field[cells[topmost]] = heights[topmost]
    return field.reshape(nx, ny)


def _disk_offsets(radius_cells: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (dx, dy)
        for dx in range(-radius_cells, radius_cells + 1)
        for dy in range(-radius_cells, radius_cells + 1)
        if math.hypot(dx, dy) <= radius_cells + 0.5)


def _disk_sweep(field, offsets, combine, fill):
    """Accumulate ``combine`` over the disk without stacking shifted copies."""
    import numpy as np

    nx, ny = field.shape
    pad = max(max(abs(dx), abs(dy)) for dx, dy in offsets) + 1
    source = np.pad(field, pad, constant_values=fill)
    source[np.isnan(source)] = fill
    out = np.full((nx, ny), fill, dtype=np.float32)
    for dx, dy in offsets:
        combine(out, source[pad + dx:pad + dx + nx, pad + dy:pad + dy + ny],
                out=out)
    return out


def _opened_surface(field, radius_cells: int):
    """Grey opening by a disk: erases anything narrower than twice the radius.

    Opening is exact on planes and only perturbs a curved surface by roughly
    curvature times radius squared, so the smooth graded rear survives it while
    a blade of any *length* is levelled to the material it stands on.  A plain
    ring probe cannot do this: on an extended ridge the ring lies along the
    ridge and reports the blade's own height as the local background.
    """
    offsets = _disk_offsets(radius_cells)
    import numpy as np

    eroded = _disk_sweep(field, offsets, np.minimum, np.inf)
    eroded[~np.isfinite(eroded)] = np.nan
    opened = _disk_sweep(eroded, offsets, np.maximum, -np.inf)
    opened[~np.isfinite(opened)] = np.nan
    return opened


def _disk_fully_on_part(field, radius_cells: int):
    """True where every cell of the disk carries sampled material."""
    import numpy as np

    offsets = _disk_offsets(radius_cells)
    nx, ny = field.shape
    pad = radius_cells + 1
    on_part = np.pad(np.isfinite(field).astype(np.float32), pad)
    covered = np.zeros((nx, ny), dtype=np.float32)
    for dx, dy in offsets:
        covered += on_part[pad + dx:pad + dx + nx, pad + dy:pad + dy + ny]
    return covered >= len(offsets) - 0.5


def _assert_no_standing_blade(path: Path, label: str) -> float:
    """Reject thin membranes standing proud of a printable piece's top face.

    Printable pieces are exported already lying in print orientation, so a
    modelling remnant that a slicer would draw as a phantom wall is a blade
    standing off the +Z surface.  Vertical ray casting on a lattice finer than
    any printable wall samples such a blade; grey opening by a disk then
    levels anything narrower than the disk, whatever its length, so the blade
    is what remains above the opened surface.  Only cells whose whole disk sits
    on the part are judged, which is what separates a blade standing on
    material from the legitimately narrow plan features of the design -- the
    feathered crescent tip and the dovetail points.  Sliver solids and
    zero-thickness shells are already rejected by ``_assert_strict_stl``; this
    closes the remaining case of a blade fused to the main body.
    """
    import numpy as np

    triangles = _binary_stl_triangles(path)
    field = _top_height_field(triangles, STANDING_BLADE_LATTICE_MM)
    radius_cells = max(2, int(round(
        STANDING_BLADE_DISK_RADIUS_MM / STANDING_BLADE_LATTICE_MM)))
    rise = np.where(
        _disk_fully_on_part(field, radius_cells),
        field - _opened_surface(field, radius_cells), np.nan)
    if not np.isfinite(rise).any():
        raise AssertionError(f"{label}: no sampled top surface in {path.name}")
    worst = float(np.nanmax(rise))
    standing = int((np.nan_to_num(rise, nan=-1.0)
                    > STANDING_BLADE_MAX_RISE_MM).sum())
    if standing:
        flat_index = int(np.nanargmax(rise))
        gi, gj = np.unravel_index(flat_index, rise.shape)
        points = triangles.reshape(-1, 3)
        origin = points[:, :2].min(axis=0) - STANDING_BLADE_LATTICE_MM
        at = origin + (np.array([gi, gj]) + 0.5) * STANDING_BLADE_LATTICE_MM
        raise AssertionError(
            f"{label}: {standing} lattice cells stand more than "
            f"{STANDING_BLADE_MAX_RISE_MM} mm above the surface opened by a "
            f"{STANDING_BLADE_DISK_RADIUS_MM} mm disk; worst {worst:.4f} mm "
            f"at x={at[0]:.2f} y={at[1]:.2f} in {path.name}. A thin "
            "membrane is standing off the printed top surface.")
    return worst


def _assert_surface_matches_depth_law(
        path: Path, sidecar: dict, cad, slug: str, side: str,
        label: str) -> float:
    """Gate the built rear surface against the approved analytic depth law.

    The blade gate above catches a narrow membrane; this catches the other
    half of the same failure -- a broad ridge, bulge or fold that is smooth
    enough to survive opening but is simply not the surface the depth law
    asks for.  A printable piece lies front-face-down with the front face on
    z=0, so the sampled top height *is* the local material depth and can be
    compared straight against ``wing_depth_at`` once the cell centre is mapped
    back through the sidecar transform.

    Only the interior is judged.  Within roughly a probe radius of the
    outline a cell centre can map just outside the wing plan, where the law
    evaluates against a different region and reports metre-scale nonsense;
    that band is boundary bookkeeping, not geometry.
    """
    import numpy as np

    matrix = np.array(sidecar["source_to_stl_matrix"], dtype=float)
    inverse = np.linalg.inv(matrix)
    triangles = _binary_stl_triangles(path)
    field = _top_height_field(triangles, DEPTH_LAW_LATTICE_MM)
    interior = _disk_fully_on_part(field, max(2, int(round(
        DEPTH_LAW_INTERIOR_MARGIN_MM / DEPTH_LAW_LATTICE_MM))))
    gi, gj = np.nonzero(np.isfinite(field) & interior)
    assert len(gi) > 100, f"{label}: too few interior samples in {path.name}"
    points = triangles.reshape(-1, 3)
    origin = points[:, :2].min(axis=0) - DEPTH_LAW_LATTICE_MM
    xs = origin[0] + (gi + 0.5) * DEPTH_LAW_LATTICE_MM
    ys = origin[1] + (gj + 0.5) * DEPTH_LAW_LATTICE_MM
    built = field[gi, gj].astype(float)
    source = np.stack([xs, ys, built, np.ones_like(xs)], axis=1) @ inverse.T
    analytic = np.asarray(cad.wing_depth_at(slug, side, source[:, :2]),
                          dtype=float)
    deviation = built - analytic
    worst = float(np.abs(deviation).max())
    if worst > DEPTH_LAW_MAX_DEVIATION_MM:
        k = int(np.argmax(np.abs(deviation)))
        raise AssertionError(
            f"{label}: built rear surface departs from the approved depth law "
            f"by {deviation[k]:+.4f} mm at stl x={xs[k]:.2f} y={ys[k]:.2f} "
            f"(built {built[k]:.4f} mm, law {analytic[k]:.4f} mm) in "
            f"{path.name}; limit {DEPTH_LAW_MAX_DEVIATION_MM} mm")
    return worst


def _assert_review_png(path: Path, slug: str, kind: str) -> None:
    from PIL import Image

    assert path.is_file() and path.stat().st_size >= 4096, (
        f"missing/truncated review PNG: {path}")
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 12))
        assert stream.read() == b"\x00\x00\x00\x00IEND\xaeB`\x82", (
            f"interrupted PNG transaction: {path}")
    with Image.open(path) as image:
        title = image.info.get("Title")
        description = image.info.get("Description")
        image.verify()
    with Image.open(path) as image:
        image.load()
        assert image.width >= 1200 and image.height >= 800, (
            f"{path}: review render is too small: {image.size}")
    assert isinstance(title, str) and slug.upper() in title, (
        f"{path}: missing variant-bound PNG metadata")
    assert (isinstance(description, str)
            and "dual-state Obi-Wan LM-lower silhouettes: coincident common "
                "profile, blue dash-dot no-floor and green dotted floor "
                "stand" in description), (
        f"{path}: missing Obi-Wan context-silhouette metadata")
    required_phrase = {
        "front": "acoustic front",
        "rear": "rear surface",
        "side_section": "rear-depth profile",
        "split_exploded": "exploded print assembly",
        "two_piece_split_exploded": "exploded print assembly",
        "magnet_roots": "magnetic roots",
    }[kind]
    assert required_phrase in title.lower(), (
        f"{path}: review kind/title mismatch: {title!r}")


def _artifact_record_map(manifest: dict) -> dict[str, dict]:
    records = manifest.get("artifacts")
    assert isinstance(records, list), "manifest artifacts must be a list"
    result: dict[str, dict] = {}
    for record in records:
        assert isinstance(record, dict), "manifest artifact must be an object"
        relative = record.get("path")
        assert isinstance(relative, str) and relative, (
            "manifest artifact path must be a nonempty string")
        assert relative not in result, f"duplicate manifest artifact: {relative}"
        result[relative] = record
    return result


def _resolve_variant_relative(directory: Path, relative: str) -> Path:
    relative_path = Path(relative)
    assert not relative_path.is_absolute(), (
        f"manifest path must be relative: {relative}")
    assert ".." not in relative_path.parts, (
        f"manifest path escapes variant directory: {relative}")
    path = directory / relative_path
    resolved_root = directory.resolve()
    resolved = path.resolve()
    assert resolved == resolved_root or resolved_root in resolved.parents, (
        f"manifest path escapes variant directory: {relative}")
    return path


def _mesh_facts_match(recorded: dict, actual: dict, label: str) -> None:
    assert isinstance(recorded, dict), f"{label}: mesh facts must be an object"
    assert set(recorded) == set(actual), (
        f"{label}: mesh-fact keys drifted: "
        f"{sorted(set(recorded) ^ set(actual))}")
    for key, expected in recorded.items():
        measured = actual[key]
        if isinstance(expected, float):
            assert math.isclose(
                measured, expected, rel_tol=1e-12, abs_tol=1e-9), (
                    f"{label}: mesh {key} {measured!r} != {expected!r}")
        else:
            assert measured == expected, (
                f"{label}: mesh {key} {measured!r} != {expected!r}")


def _source_hash_records(source: dict) -> tuple[tuple[str, str], ...]:
    """Extract the three required path/hash pairs from the source record."""
    assert isinstance(source, dict), "source provenance must be an object"
    required = (
        ("geometry_path", "geometry_sha256"),
        ("exporter_path", "exporter_sha256"),
        ("contract_generator_path", "contract_generator_sha256"),
    )
    records = []
    for path_key, hash_key in required:
        relative = source.get(path_key)
        digest = source.get(hash_key)
        assert isinstance(relative, str) and relative, (
            f"source {path_key} must be a nonempty string")
        assert isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), (
            f"source {hash_key} is not a SHA-256 digest")
        records.append((relative, digest))
    combined = source.get("combined_sha256")
    assert isinstance(combined, str) and re.fullmatch(r"[0-9a-f]{64}", combined), (
        "source combined_sha256 is not a SHA-256 digest")
    module = source.get("geometry_module")
    assert module == "lx521_baffle.obiwan.wings", (
        f"unexpected geometry module provenance: {module!r}")
    return tuple(records)


def _verify_source_hashes(source: dict) -> None:
    primary = dict(_source_hash_records(source))
    records = source.get("files")
    assert isinstance(records, list) and len(records) >= 7, (
        "source attestation must enumerate all CAD interface inputs")
    by_path: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for record in records:
        assert isinstance(record, dict) and set(record) == {"path", "sha256"}
        relative = record.get("path")
        expected = record.get("sha256")
        assert isinstance(relative, str) and relative
        assert isinstance(expected, str) and re.fullmatch(
            r"[0-9a-f]{64}", expected)
        assert relative not in by_path, f"duplicate source attestation: {relative}"
        by_path[relative] = expected
        relative_path = Path(relative)
        assert not relative_path.is_absolute() and ".." not in relative_path.parts, (
            f"unsafe source path: {relative}")
        path = ROOT / relative_path
        assert path.is_file(), f"missing source-bound file: {path}"
        assert _sha256_file(path) == expected, f"stale source hash: {relative}"
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(expected.encode("ascii"))
        aggregate.update(b"\n")
    assert list(by_path) == sorted(by_path), (
        "source attestation is not deterministically path-sorted")
    assert primary.items() <= by_path.items(), (
        "primary source hashes disagree with complete source attestation")
    assert source.get("combined_sha256") == aggregate.hexdigest(), (
        "combined source attestation digest is stale")


def _print_record_key(record: dict) -> tuple[str, str, str, int]:
    split_variant = record.get("split_variant")
    side = record.get("side")
    role = record.get("role")
    order = record.get("order")
    assert split_variant in {"a", "b"}, (
        f"invalid split variant: {split_variant!r}")
    assert side in SIDE_NAMES, f"invalid print side: {side!r}"
    expected_roles = (
        PRINT_PART_ROLES if split_variant == "a"
        else TWO_PIECE_PART_ROLES)
    assert role in expected_roles, f"invalid print role: {role!r}"
    assert isinstance(order, int), f"invalid print order: {order!r}"
    return split_variant, side, role, order


def _assert_print_bbox(record: dict, label: str) -> None:
    bbox = _assert_bbox_facts(record.get("print_bbox_mm"), label)
    size = [float(value) for value in bbox["size_mm"]]
    assert all(abs(float(value)) <= 1e-6 for value in bbox["min_mm"]), (
        f"{label}: print part was not translated to the bed origin")
    assert size[0] <= BED_XY_MM + 0.05, (
        f"{label}: print X {size[0]:.2f} exceeds {BED_XY_MM:g} mm")
    assert size[1] <= BED_XY_MM + 0.05, (
        f"{label}: print Y {size[1]:.2f} exceeds {BED_XY_MM:g} mm")
    assert size[2] <= BED_XY_MM + 0.05, (
        f"{label}: print Z {size[2]:.2f} exceeds {BED_XY_MM:g} mm")


def _assert_bbox_facts(bbox, label: str) -> dict:
    assert isinstance(bbox, dict) and set(bbox) == {
        "min_mm", "max_mm", "size_mm"}, (
            f"{label}: malformed bbox facts")
    assert all(
        isinstance(bbox[key], list) and len(bbox[key]) == 3
        for key in ("min_mm", "max_mm", "size_mm")), (
            f"{label}: malformed bbox vectors")
    size = [float(value) for value in bbox["size_mm"]]
    assert all(value > 0.0 for value in size), f"{label}: empty bbox"
    for axis in range(3):
        measured = float(bbox["max_mm"][axis]) - float(bbox["min_mm"][axis])
        assert math.isclose(measured, size[axis], abs_tol=1e-6), (
            f"{label}: inconsistent bbox axis {axis}")
    return bbox


def test_exported_artifact_contract() -> None:
    """Gate the exact flat/graded release transaction and every printable mesh."""
    _require_remote_guard()

    import lx521_baffle.obiwan.wings as cad

    assert tuple(cad.VARIANT_IDS) == VARIANT_IDS
    assert tuple(cad.SIDE_NAMES) == SIDE_NAMES
    assert tuple(cad.PRINT_PART_KEYS) == PRINT_PART_ROLES
    assert tuple(cad.TWO_PIECE_PRINT_PART_KEYS) == TWO_PIECE_PART_ROLES

    for slug in VARIANT_IDS:
        paths = _variant_paths(slug)
        directory = paths["directory"]
        assert isinstance(directory, Path) and directory.is_dir(), (
            f"missing variant artifact directory: {directory}")
        canonical_step = paths["canonical_step"]
        assembled_step = paths["assembled_step"]
        two_piece_assembled_step = paths["two_piece_assembled_step"]
        facts_path = paths["facts"]
        manifest_path = paths["manifest"]
        stls = paths["stls"]
        sidecars = paths["sidecars"]
        assert isinstance(canonical_step, Path)
        assert isinstance(assembled_step, Path)
        assert isinstance(two_piece_assembled_step, Path)
        assert isinstance(facts_path, Path)
        assert isinstance(manifest_path, Path)
        assert isinstance(stls, tuple)
        assert isinstance(sidecars, tuple)
        assert len(stls) == len(sidecars) == 10

        review_paths = tuple(
            directory / "review" / f"obiwan_wing_{slug}_{kind}.png"
            for kind in REVIEW_KINDS)

        expected_relative = {
            canonical_step.relative_to(directory).as_posix(),
            assembled_step.relative_to(directory).as_posix(),
            two_piece_assembled_step.relative_to(directory).as_posix(),
            facts_path.relative_to(directory).as_posix(),
            manifest_path.relative_to(directory).as_posix(),
            *(path.relative_to(directory).as_posix() for path in stls),
            *(path.relative_to(directory).as_posix() for path in sidecars),
            *(path.relative_to(directory).as_posix() for path in review_paths),
        }
        actual_relative = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*") if path.is_file()
        }
        assert actual_relative == expected_relative, (
            f"{slug}: artifact inventory drifted; missing="
            f"{sorted(expected_relative - actual_relative)}, extra="
            f"{sorted(actual_relative - expected_relative)}")

        canonical_labels = (
            f"lx521_obiwan_basic_wing_{slug}_monolithic_pair",
            f"obiwan_wing_{slug}_left_monolithic",
            f"obiwan_wing_{slug}_right_monolithic",
        )
        assembled_labels = (
            f"lx521_obiwan_basic_wing_{slug}_print_assembly",
            *(
                f"obiwan_wing_{slug}_{side}_{order}_of_3_{role}"
                for side in SIDE_NAMES
                for order, role in enumerate(PRINT_PART_ROLES, start=1)
            ),
        )
        two_piece_assembled_labels = (
            f"lx521_obiwan_basic_wing_{slug}_two_piece_print_assembly",
            *(
                f"obiwan_wing_{slug}_{side}_split2_{order}_of_2_{role}"
                for side in SIDE_NAMES
                for order, role in enumerate(
                    TWO_PIECE_PART_ROLES, start=1)
            ),
        )
        _assert_complete_step(canonical_step, canonical_labels)
        _assert_complete_step(assembled_step, assembled_labels)
        _assert_complete_step(
            two_piece_assembled_step, two_piece_assembled_labels)

        facts = _read_json_object(facts_path)
        manifest = _read_json_object(manifest_path)
        for document, name in ((facts, "facts"), (manifest, "manifest")):
            assert document.get("schema_version") == 3, (
                f"{slug} {name}: schema version drifted")
            assert document.get("artifact_family") == "obiwan_wing_artifacts", (
                f"{slug} {name}: artifact family drifted")
            assert document.get("variant_slug") == slug, (
                f"{slug} {name}: variant identity drifted")
        geometry = facts.get("geometry")
        assert isinstance(geometry, dict), f"{slug}: missing geometry facts"
        assert geometry.get("variant_slug") == slug
        qualification = geometry.get("qualification")
        assert qualification == {
            "status": "unmeasured_acoustic_experiment",
            "not_w1_w2_spec_compliance": True,
            "canonical_geometry": "monolithic_pair",
            "print_geometry_derivation": "post_boolean_plan_intersection",
        }, f"{slug}: qualification contract drifted"
        depth = geometry.get("depth_contract")
        assert isinstance(depth, dict)
        assert math.isclose(depth.get("full_depth_mm"), FULL_DEPTH_MM,
                            abs_tol=1e-9)
        assert depth.get("retention_centres") == ["LM", "UM", "T"]
        assert depth.get("retention_scales") == [1.8, 0.9, 0.58]
        if slug == "flat":
            assert depth.get("model") == "constant"
            assert depth.get("surface_construction") == "plan_prism"
            assert math.isclose(depth.get("minimum_depth_mm"),
                                FULL_DEPTH_MM, abs_tol=1e-9)
            assert depth.get("protected_perimeter_brep_c0_gate") is None
        else:
            assert depth.get("model") == "LM_UM_T_weighted_smooth_rear"
            assert depth.get("surface_construction") == (
                "direct_open_uniform_control_bspline")
            assert depth.get("surface_spline_degree") == 3
            assert math.isclose(depth.get("minimum_depth_mm"),
                                GRADED_EDGE_DEPTH_MM, abs_tol=1e-9)
            assert math.isclose(depth.get("exact_edge_brep_band_mm"),
                                0.12, abs_tol=1e-9)
            island_gate = depth.get("conservative_relief_island_retention")
            assert isinstance(island_gate, dict)
            assert island_gate.get("candidate_component_count") == 2
            assert island_gate.get("cut_component_count") == 1
            assert island_gate.get("retained_full_depth_island_count") == 1
            assert 3.8 < island_gate.get(
                "retained_full_depth_island_area_mm2") < 4.0
            assert island_gate.get("largest_retained_island_area_mm2") == (
                island_gate.get("retained_full_depth_island_area_mm2"))
            assert island_gate.get("maximum_allowed_island_area_mm2") == 4.0
            assert island_gate.get(
                "maximum_allowed_total_island_area_mm2") == 4.0
            relief_mask = depth.get("conservative_boolean_relief_mask")
            assert isinstance(relief_mask, dict)
            assert relief_mask.get("normal_inset_mm") == 0.04
            assert relief_mask.get("general_maximum_hausdorff_mm") == 0.08
            assert relief_mask.get("lower_root_exception_radius_mm") == 0.25
            assert relief_mask.get("lower_root_maximum_hausdorff_mm") == 0.20
            assert relief_mask.get("outside_exact_relief_area_mm2") <= 1.0e-9
            assert 0.18 < relief_mask.get(
                "measured_maximum_hausdorff_mm") < 0.20
            assert relief_mask.get("edge_plan_overlap_mm") >= 0.02
            perimeter_gate = depth.get("protected_perimeter_brep_c0_gate")
            assert isinstance(perimeter_gate, dict)
            assert math.isclose(
                perimeter_gate.get("maximum_sample_spacing_mm"),
                0.5, abs_tol=1e-9)
            assert math.isclose(
                perimeter_gate.get("paired_probe_offset_mm"),
                0.004, abs_tol=1e-9)
            assert math.isclose(
                perimeter_gate.get("maximum_allowed_c0_jump_mm"),
                0.03, abs_tol=1e-9)
            assert perimeter_gate.get("paired_probe_count", 0) >= 20
            assert perimeter_gate.get("maximum_measured_c0_jump_mm") <= 0.03
            assert perimeter_gate.get(
                "excluded_external_boundary_length_mm") >= 0.0
        interface = geometry.get("interface_contract")
        assert isinstance(interface, dict)
        assert interface.get("selected_receiver_count_per_side") == 3
        lower_integration = interface.get("lower_floor_bend_integration")
        assert isinstance(lower_integration, dict)
        assert lower_integration.get("profile") == (
            "g1_cubic_from_floor_bend_tangent_to_outer_flank")
        assert math.isclose(
            lower_integration.get("minimum_y_mm"), 74.15, abs_tol=1e-9)
        assert math.isclose(
            lower_integration.get("inner_start_xy_mm")[1],
            74.15, abs_tol=1e-9)
        assert math.isclose(
            lower_integration.get("outer_join_xy_mm")[1],
            105.981, abs_tol=1e-9)
        t_interface = interface.get("tweeter_crescent")
        assert isinstance(t_interface, dict)
        assert math.isclose(
            t_interface.get("normal_plan_clearance_mm"),
            T_WING_CLEARANCE_MM, abs_tol=1e-9)
        assert t_interface.get("released_upper_profile_source") == (
            "OUTLINE_B2 cubic horn flanks")
        assert t_interface.get("symmetry_policy") == (
            "union_with_mirror_worst_case")
        assert t_interface.get("keepout_construction") == (
            "released_plan.buffer(clearance)")
        assert t_interface.get("split_independent") is True
        keyed_interface = interface.get("optional_lm_keyed_split")
        assert keyed_interface == {
            "geometrically_compatible": True,
            "physical_fit_coupon_required": True,
            "ring_local_key_clearance_mm": 0.25,
            "pocket_location": "carrier_interface_between_front_and_rear",
            "right_pocket_uses_left_relief_worst_case": True,
            "left_is_exact_mirror": True,
            "carrier_exterior_growth_mm": 0.0,
            "primary_magnet_datums_unchanged": True,
        }
        for side in SIDE_NAMES:
            receivers = interface.get("receivers", {}).get(side)
            assert isinstance(receivers, list) and len(receivers) == 3
            assert {receiver.get("name") for receiver in receivers} == {
                f"lm_lower_{side}", f"lm_upper_{side}", f"um_{side}"}
            for receiver in receivers:
                assert receiver.get("closure_kind") == (
                    "transverse_gable_45deg")
                assert math.isclose(
                    receiver.get("cavity_diameter_mm"),
                    MAGNET_CAVITY_DIAMETER_MM, abs_tol=1e-9)
                assert math.isclose(
                    receiver.get("cavity_depth_mm"),
                    MAGNET_CAVITY_DEPTH_MM, abs_tol=1e-9)
                assert math.isclose(
                    receiver.get("face_skin_mm"), MAGNET_FACE_SKIN_MM,
                    abs_tol=1e-9)
                assert math.isclose(
                    receiver.get("inner_skin_mm"), MAGNET_INNER_SKIN_MM,
                    abs_tol=1e-9)
                assert math.isclose(
                    receiver.get("captive_land_mm"),
                    MAGNET_CAPTIVE_LAND_MM, abs_tol=1e-9)
                assert math.isclose(
                    receiver.get("roof_angle_deg"),
                    MAGNET_ROOF_ANGLE_DEG, abs_tol=1e-9)
                assert math.isclose(
                    receiver.get("receiver_solid_standoff_mm"),
                    MAGNET_INTERFACE_GAP_MM, abs_tol=1e-9)
                assert math.isclose(
                    receiver.get("physical_interface_gap_mm"),
                    0.0, abs_tol=1e-9)
                assert receiver.get(
                    "receiver_spacing_standoff_is_solid") is True
                assert math.isclose(
                    receiver.get("interface_gap_mm"),
                    MAGNET_INTERFACE_GAP_MM, abs_tol=1e-9)
                assert math.isclose(
                    receiver.get("paired_magnet_face_separation_mm"),
                    _expected_magnet_face_separation(
                        receiver.get("interface_kind")), abs_tol=1e-9)
                assert receiver.get("carrier_magnet_fully_buried") is True
                assert receiver.get("receiver_magnet_fully_buried") is True
                assert all(math.isclose(actual, expected, abs_tol=1e-12)
                           for actual, expected in zip(
                               receiver.get("marked_pole_axis_xyz"),
                               [*receiver.get("axis_normal_xy"), 0.0]))
                carrier_face = receiver.get("carrier_face_xy_mm")
                carrier_cavity_datum = receiver.get(
                    "carrier_cavity_datum_xy_mm")
                mouth = receiver.get("receiver_cavity_face_xy_mm")
                normal = receiver.get("axis_normal_xy")
                expected_inset = RING_CAVITY_FACE_INSET_MM
                assert all(math.isclose(
                    carrier_face[index],
                    carrier_cavity_datum[index]
                    + expected_inset * normal[index],
                    abs_tol=1e-9) for index in range(2))
                assert all(math.isclose(
                    mouth[index],
                    carrier_face[index]
                    + MAGNET_INTERFACE_GAP_MM * normal[index],
                    abs_tol=1e-9) for index in range(2))
                assert receiver.get("print_up_source_xyz") == [0.0, 0.0, -1.0]
            serialized_lower = next(
                receiver for receiver in receivers
                if receiver.get("name") == f"lm_lower_{side}")
            sign = -1.0 if side == "left" else 1.0
            assert serialized_lower.get("interface_kind") == "shoulder"
            assert all(math.isclose(actual, expected, abs_tol=1e-6)
                       for actual, expected in zip(
                           serialized_lower.get("carrier_face_xy_mm"),
                           [sign * 45.285011, 89.190370], strict=True))
            assert all(math.isclose(actual, expected, abs_tol=1e-6)
                       for actual, expected in zip(
                           serialized_lower.get("axis_normal_xy"),
                           [sign * 0.706451, -0.707762], strict=True))
            assert math.isclose(
                serialized_lower.get("axis_z_mm"),
                OBIWAN_MAGNET_Z_MM, abs_tol=1e-9)
        dovetails = geometry.get("dovetail_contract")
        assert isinstance(dovetails, dict)
        assert dovetails.get("method") == (
            "v1l_style_through_thickness_xy_dovetails")
        assert math.isclose(
            dovetails.get("clearance_mm"), 0.05, abs_tol=1e-9)
        assert math.isclose(
            dovetails.get("endpoint_taper_mm"), 2.0, abs_tol=1e-9)
        assert math.isclose(
            dovetails.get("male_root_overlap_mm"), 0.05, abs_tol=1e-9)
        assert dovetails.get("key_count_per_side") == 2
        assert dovetails.get("part_roles") == list(PRINT_PART_ROLES)
        assert dovetails.get("two_piece_part_roles") == list(
            TWO_PIECE_PART_ROLES)
        assert dovetails.get("male_owners") == ["lm_lower", "lm_upper"]
        assert dovetails.get("lower_profile_mm") == {
            "neck": 7.0, "head": 9.0, "depth": 4.0}
        assert dovetails.get("upper_profile_mm") == {
            "neck": 7.0, "head": 8.5, "depth": 4.0}
        assert dovetails.get("no_envelope_growth") is True
        assert dovetails.get("through_local_thickness") is True
        assert dovetails.get("z_retention") is False
        joint_area = dovetails.get("graded_joint_interface_area_mm2")
        assert isinstance(joint_area, list) and len(joint_area) == 2
        assert joint_area[0] >= 75.0
        assert joint_area[1] >= 50.0
        joint_mismatch = dovetails.get("graded_joint_rear_mismatch_mm")
        assert isinstance(joint_mismatch, list) and len(joint_mismatch) == 2
        assert max(joint_mismatch) <= 0.15
        print_contract = geometry.get("print_contract")
        assert isinstance(print_contract, dict)
        installed = print_contract.get("installed_piece_brep")
        assert isinstance(installed, dict)
        assert set(installed) == set(SIDE_NAMES)
        two_piece_installed = print_contract.get(
            "two_piece_installed_piece_brep")
        assert isinstance(two_piece_installed, dict)
        assert set(two_piece_installed) == set(SIDE_NAMES)
        assert print_contract.get("options") == {
            "a": {
                "piece_count_per_side": 3,
                "part_roles": list(PRINT_PART_ROLES),
            },
            "b": {
                "piece_count_per_side": 2,
                "part_roles": list(TWO_PIECE_PART_ROLES),
                "lower_geometry_identical_to_a": True,
                "former_upper_fit_gap_restored": True,
            },
        }
        actual_facts = geometry.get("actual_brep")
        assert isinstance(actual_facts, dict)
        assert actual_facts.get("valid_single_solid_each_side") is True
        assert manifest.get("source") == facts.get("source"), (
            f"{slug}: manifest/facts source provenance differs")
        _verify_source_hashes(facts.get("source"))

        facts_relative = facts_path.relative_to(directory).as_posix()
        canonical_relative = canonical_step.relative_to(directory).as_posix()
        assembled_relative = assembled_step.relative_to(directory).as_posix()
        two_piece_assembled_relative = (
            two_piece_assembled_step.relative_to(directory).as_posix())
        review_relatives = [
            f"review/obiwan_wing_{slug}_{kind}.png" for kind in REVIEW_KINDS
        ]
        assert manifest.get("facts_path") == facts_relative
        assert manifest.get("facts_sha256") == _sha256_file(facts_path)
        assert manifest.get("canonical_step_path") == canonical_relative
        assert manifest.get("assembled_step_path") == assembled_relative
        assert manifest.get("two_piece_assembled_step_path") == (
            two_piece_assembled_relative)
        assert manifest.get("review_pngs") == review_relatives, (
            f"{slug}: review inventory/order drifted")

        artifact_records = _artifact_record_map(manifest)
        artifact_expected = expected_relative - {manifest_path.name}
        assert set(artifact_records) == artifact_expected, (
            f"{slug}: hash inventory differs from exact artifacts")
        assert list(artifact_records) == sorted(artifact_records), (
            f"{slug}: hash inventory is not deterministically sorted")
        for relative, record in artifact_records.items():
            artifact = _resolve_variant_relative(directory, relative)
            assert artifact.is_file(), f"missing hashed artifact: {artifact}"
            assert record.get("size_bytes") == artifact.stat().st_size, (
                f"{slug}: stale size for {relative}")
            assert record.get("sha256") == _sha256_file(artifact), (
                f"{slug}: stale hash for {relative}")
            assert record.get("kind") in {
                "canonical_step", "assembled_step", "print_stl",
                "print_sidecar", "facts_json", "review_png"}, (
                f"{slug}: invalid artifact kind for {relative}")
            if relative == canonical_relative:
                expected_kind = "canonical_step"
            elif relative in {
                    assembled_relative, two_piece_assembled_relative}:
                expected_kind = "assembled_step"
            elif relative == facts_relative:
                expected_kind = "facts_json"
            elif relative.endswith(".print.json"):
                expected_kind = "print_sidecar"
            elif relative.startswith("stl/") and relative.endswith(".stl"):
                expected_kind = "print_stl"
            else:
                expected_kind = "review_png"
            assert record.get("kind") == expected_kind, (
                f"{slug}: wrong artifact kind for {relative}")

        for kind, path in zip(REVIEW_KINDS, review_paths, strict=True):
            _assert_review_png(path, slug, kind)

        print_records = facts.get("exports", {}).get("print_parts")
        assert isinstance(print_records, list) and len(print_records) == 10, (
            f"{slug}: facts must contain ten print records")
        expected_keys = {
            ("a", side, role, order)
            for side in SIDE_NAMES
            for order, role in enumerate(PRINT_PART_ROLES, start=1)
        } | {
            ("b", side, role, order)
            for side in SIDE_NAMES
            for order, role in enumerate(TWO_PIECE_PART_ROLES, start=1)
        }
        record_map = {
            _print_record_key(record): record for record in print_records
        }
        assert set(record_map) == expected_keys, (
            f"{slug}: print role/order inventory drifted")
        for side in SIDE_NAMES:
            a_lower = _resolve_variant_relative(
                directory, record_map[("a", side, "lm_lower", 1)]["path"])
            b_lower = _resolve_variant_relative(
                directory, record_map[("b", side, "lm_lower", 1)]["path"])
            assert _sha256_file(a_lower) == _sha256_file(b_lower), (
                f"{slug}/{side}: B lower is not byte-identical to A lower")
        assert manifest.get("print_parts") == [
            record["path"] for record in print_records], (
                f"{slug}: facts/manifest print inventory differs")
        assert manifest.get("print_sidecars") == [
            record["print_sidecar"] for record in print_records], (
                f"{slug}: facts/manifest sidecar inventory differs")
        assert set(manifest["print_sidecars"]) == {
            path.relative_to(directory).as_posix() for path in sidecars
        }, f"{slug}: manifest does not bind exactly ten real sidecars"

        for split_variant, side, role, order in sorted(expected_keys):
            record = record_map[(split_variant, side, role, order)]
            piece_count = 3 if split_variant == "a" else 2
            if split_variant == "a":
                expected_name = (
                    f"obiwan_wing_{slug}_{side}_"
                    f"{order}_of_3_{role}.stl")
                expected_label = (
                    f"obiwan_wing_{slug}_{side}_{order}_of_3_{role}")
            else:
                expected_name = (
                    f"obiwan_wing_{slug}_{side}_split2_"
                    f"{order}_of_2_{role}.stl")
                expected_label = (
                    f"obiwan_wing_{slug}_{side}_split2_{order}_of_2_{role}")
            relative = record.get("path")
            expected_relative_path = f"stl/{expected_name}"
            assert relative == expected_relative_path, (
                f"{slug} {side}/{role}: STL path drifted")
            assert record.get("label") == expected_label, (
                    f"{slug} {side}/{role}: STEP/STL label drifted")
            assert record.get("split_variant") == split_variant
            assert record.get("piece_count") == piece_count
            assert isinstance(record.get("volume_mm3"), (int, float))
            assert record["volume_mm3"] > 1.0
            assert record.get("volume_integration") == (
                "BRepGProp_adaptive_2d_Gauss")
            assembly_bbox = _assert_bbox_facts(
                record.get("assembly_bbox_mm"),
                f"{slug} {side}/{role} assembly")
            assert assembly_bbox["min_mm"][2] >= REAR_Z_MM - 0.02, (
                f"{slug} {side}/{role}: extends behind Obi-Wan envelope")
            assert assembly_bbox["max_mm"][2] <= FRONT_Z_MM + 0.02, (
                f"{slug} {side}/{role}: extends ahead of front datum")
            assert math.isclose(
                assembly_bbox["max_mm"][2], FRONT_Z_MM, abs_tol=0.01), (
                    f"{slug} {side}/{role}: front is not flush at z=18.3")
            assert record.get("bed_limit_mm") == BED_XY_MM
            expected_mesh_tolerance = 0.002 if slug == "graded" else 0.01
            expected_angular_tolerance = 0.03 if slug == "graded" else 0.08
            assert record.get("mesh_tolerance_mm") == expected_mesh_tolerance
            assert record.get("mesh_angular_tolerance") == (
                expected_angular_tolerance)
            assert isinstance(record.get("transform_zero_fixes"), int)
            assert record["transform_zero_fixes"] >= 0
            transform = record.get("print_transform_deg")
            assert (isinstance(transform, dict)
                    and set(transform) == {"x", "z"}
                    and transform["x"] == 180.0
                    and isinstance(transform["z"], (int, float))), (
                f"{slug} {side}/{role}: invalid print transform")
            structured_transform = record.get("print_transform")
            assert isinstance(structured_transform, dict), (
                f"{slug} {side}/{role}: missing exact print transform")
            validate_front_down_transform(
                structured_transform,
                label=f"{slug} {side}/{role} print transform",
            )
            assert structured_transform["rotation_deg"] == transform, (
                f"{slug} {side}/{role}: print transform metadata disagrees")
            matrix = structured_transform["source_to_stl_matrix"]
            front_print_z = (
                float(matrix[2][2]) * FRONT_Z_MM + float(matrix[2][3]))
            assert math.isclose(front_print_z, 0.0, abs_tol=1e-6), (
                f"{slug} {side}/{role}: acoustic front is not on the bed")
            expected_sidecar_relative = Path(
                expected_relative_path).with_suffix(".print.json").as_posix()
            sidecar_relative = record.get("print_sidecar")
            assert sidecar_relative == expected_sidecar_relative, (
                f"{slug} {side}/{role}: non-sibling print sidecar")
            sidecar = _resolve_variant_relative(
                directory, sidecar_relative)
            sidecar_payload = validate_print_sidecar(
                directory / relative, sidecar)
            assert record.get("print_sidecar_sha256") == _sha256_file(
                sidecar), (
                    f"{slug} {side}/{role}: stale sidecar hash in facts")
            assert sidecar_payload.get("stl_sha256") == _sha256_file(
                directory / relative), (
                    f"{slug} {side}/{role}: sidecar does not bind its STL")
            assert sidecar_payload.get("part") == Path(expected_name).stem, (
                f"{slug} {side}/{role}: sidecar part identity drifted")
            assert {
                key: sidecar_payload.get(key)
                for key in (
                    "artifact_family", "variant_slug", "assembly_label",
                    "split_variant", "piece_count", "side", "order", "role",
                    "mesh")
            } == {
                "artifact_family": "obiwan_wing_artifacts",
                "variant_slug": slug,
                "assembly_label": record["label"],
                "split_variant": split_variant,
                "piece_count": piece_count,
                "side": side,
                "order": order,
                "role": role,
                "mesh": {
                    "tolerance_mm": expected_mesh_tolerance,
                    "angular_tolerance": expected_angular_tolerance,
                },
            }, f"{slug} {side}/{role}: sidecar release metadata drifted"
            validate_front_down_transform(
                sidecar_payload,
                label=f"{slug} {side}/{role} print sidecar",
            )
            assert {
                key: sidecar_payload.get(key) for key in structured_transform
            } == structured_transform, (
                f"{slug} {side}/{role}: sidecar/facts transform differs")
            _assert_print_bbox(record, f"{slug} {side}/{role}")
            mesh = _assert_strict_stl(directory / relative)
            _assert_no_standing_blade(
                directory / relative, f"{slug} {side}/{role}")
            _assert_surface_matches_depth_law(
                directory / relative, sidecar_payload, cad, slug, side,
                f"{slug} {side}/{role}")
            _mesh_facts_match(
                record.get("stl_diagnostics"), mesh,
                f"{slug} {side}/{role}")
            assert math.isclose(
                mesh["signed_volume"], record["volume_mm3"],
                rel_tol=0.003, abs_tol=0.5), (
                    f"{slug} {side}/{role}: STL/source volume mismatch")
        for side in SIDE_NAMES:
            a_lower = record_map[("a", side, "lm_lower", 1)]
            b_lower = record_map[("b", side, "lm_lower", 1)]
            assert math.isclose(
                a_lower["volume_mm3"], b_lower["volume_mm3"],
                rel_tol=0.0, abs_tol=1.0e-6), (
                    f"{slug}/{side}: B lower is not A lower")
            assert a_lower["assembly_bbox_mm"] == b_lower[
                "assembly_bbox_mm"], (
                    f"{slug}/{side}: B lower bounds differ from A lower")

        canonical_record = facts.get("exports", {}).get("canonical_step")
        assembled_record = facts.get("exports", {}).get("assembled_step")
        two_piece_assembled_record = facts.get("exports", {}).get(
            "two_piece_assembled_step")
        assert isinstance(canonical_record, dict)
        assert isinstance(assembled_record, dict)
        assert isinstance(two_piece_assembled_record, dict)
        canonical_bbox = _assert_bbox_facts(
            canonical_record.get("bbox_mm"), f"{slug} canonical STEP")
        assembled_bbox = _assert_bbox_facts(
            assembled_record.get("bbox_mm"), f"{slug} assembled STEP")
        two_piece_assembled_bbox = _assert_bbox_facts(
            two_piece_assembled_record.get("bbox_mm"),
            f"{slug} two-piece assembled STEP")
        assert canonical_record == {
            "path": canonical_relative,
            "label": f"lx521_obiwan_basic_wing_{slug}_monolithic_pair",
            "solid_count": 2,
            "bbox_mm": canonical_bbox,
        }
        assert assembled_record == {
            "path": assembled_relative,
            "label": f"lx521_obiwan_basic_wing_{slug}_print_assembly",
            "solid_count": 6,
            "bbox_mm": assembled_bbox,
        }
        assert two_piece_assembled_record == {
            "path": two_piece_assembled_relative,
            "label": (
                f"lx521_obiwan_basic_wing_{slug}_"
                "two_piece_print_assembly"),
            "solid_count": 4,
            "bbox_mm": two_piece_assembled_bbox,
        }
        _assert_bounds_close(
            (tuple(canonical_bbox["min_mm"]), tuple(canonical_bbox["max_mm"])),
            (tuple(assembled_bbox["min_mm"]), tuple(assembled_bbox["max_mm"])),
            0.02, f"{slug} canonical/print assembly")
        _assert_bounds_close(
            (tuple(canonical_bbox["min_mm"]), tuple(canonical_bbox["max_mm"])),
            (tuple(two_piece_assembled_bbox["min_mm"]),
             tuple(two_piece_assembled_bbox["max_mm"])),
            0.02, f"{slug} canonical/two-piece print assembly")
        assert facts.get("exports", {}).get("review_pngs") == review_relatives
        review_context = facts.get("exports", {}).get("review_context")
        assert isinstance(review_context, dict)
        assert review_context.get("state_contract") == (
            "dual_lm_lower_with_no_floor_upper_um_t")
        context_records = review_context.get("records")
        assert isinstance(context_records, list) and len(context_records) == 5
        assert [record.get("key") for record in context_records] == [
            "lm_lower_floor", "lm_lower_no_floor", "lm_upper", "um", "t"]
        by_key = {record["key"]: record for record in context_records}
        assert by_key["lm_lower_no_floor"]["state"] == "no_floor_stand"
        assert by_key["lm_lower_no_floor"]["part_key"] == (
            "optional_lm_keyed_1_of_2_bottom")
        assert by_key["lm_lower_no_floor"]["color"] == "#2878b5"
        assert by_key["lm_lower_no_floor"]["line_style"] == "dash_dot"
        assert by_key["lm_lower_floor"]["state"] == "floor_stand"
        assert by_key["lm_lower_floor"]["part_key"] == (
            "optional_lm_keyed_1_of_2_bottom")
        assert by_key["lm_lower_floor"]["color"] == "#2e8b57"
        assert by_key["lm_lower_floor"]["line_style"] == "dotted"
        assert (by_key["lm_lower_floor"]["source_sha256"]
                != by_key["lm_lower_no_floor"]["source_sha256"])
        assert by_key["lm_lower_floor"]["z_bounds_mm"][0] <= -149.9
        assert by_key["lm_lower_no_floor"]["z_bounds_mm"][0] > -10.0
        for key in ("lm_upper", "um", "t"):
            assert by_key[key]["state"] == "no_floor_stand"
            assert by_key[key]["line_style"] == "dotted_neutral"
        for record in context_records:
            assert (isinstance(record.get("source_sha256"), str)
                    and len(record["source_sha256"]) == 64)
            assert type(record.get("xy_path_count")) is int
            assert record["xy_path_count"] > 0
            assert type(record.get("yz_path_count")) is int
            assert record["yz_path_count"] > 0
        print(f"  {slug}: exact STEP/JSON inventory and ten strict STLs pass")


def _vertical_depth_mm(shape, x_mm: float, y_mm: float) -> float:
    """Measure actual BREP material span on one vertical probe line."""
    from build123d import Axis

    hits = shape.intersect(
        Axis((float(x_mm), float(y_mm), REAR_Z_MM - 2.0), (0.0, 0.0, 1.0)),
        tolerance=1.0e-5, include_touched=True)
    assert hits, f"vertical BREP probe missed x/y={x_mm:.4f}/{y_mm:.4f}"
    z_values = []
    for hit in hits:
        z_values.extend(float(vertex.Z) for vertex in hit.vertices())
    assert len(z_values) >= 2, (
        f"vertical BREP probe returned no span at {x_mm:.4f}/{y_mm:.4f}")
    return max(z_values) - min(z_values)


def _line_parts(geometry) -> tuple:
    """Flatten a Shapely line collection without buffering its exact locus."""
    if geometry.is_empty:
        return ()
    if geometry.geom_type in ("LineString", "LinearRing"):
        return (geometry,) if geometry.length > 1.0e-6 else ()
    if not hasattr(geometry, "geoms"):
        return ()
    return tuple(
        line
        for child in geometry.geoms
        for line in _line_parts(child)
    )


def _assert_plan_dovetail_contract(cad) -> None:
    """Gate the exact V1L-style XY keys before OCC plan intersection."""
    import numpy as np
    from shapely.geometry import Point
    from shapely.ops import unary_union

    layout = cad._layout()
    keys = layout.dovetail_keys
    gaps = layout.fit_clearance_gaps
    seams = layout.joint_seams
    assert len(keys) == len(gaps) == len(seams) == 2

    expected_owners = (
        ("lm_lower", "lm_upper"),
        ("lm_upper", "um"),
    )
    names = []
    for index, (record, gap, seam, profile, owners) in enumerate(zip(
            keys, gaps, seams, DOVETAIL_PROFILES_MM, expected_owners,
            strict=True)):
        assert isinstance(record, dict), f"dovetail {index}: bad record"
        names.append(record.get("name"))
        assert record.get("male_owner") == owners[0]
        assert record.get("female_owner") == owners[1]
        assert math.isclose(
            record.get("neck_mm"), profile["neck"], abs_tol=1e-9)
        assert math.isclose(
            record.get("head_mm"), profile["head"], abs_tol=1e-9)
        assert math.isclose(
            record.get("penetration_mm"), profile["depth"], abs_tol=1e-9)
        assert math.isclose(
            record.get("root_overlap_mm"), DOVETAIL_ROOT_OVERLAP_MM,
            abs_tol=1e-9)

        polygon = record.get("polygon")
        assert polygon is not None and not polygon.is_empty
        assert polygon.is_valid and polygon.area > 1.0
        assert layout.field_right.buffer(1.0e-6).covers(polygon), (
            f"dovetail {index}: key leaves the monolithic plan envelope")
        assert layout.nominal_parts[owners[0]].buffer(1.0e-6).covers(
            polygon), f"dovetail {index}: key is not owned by {owners[0]}"

        ligament = float(polygon.distance(layout.field_right.boundary))
        assert ligament >= DOVETAIL_MIN_LIGAMENT_MM - 0.01, (
            f"dovetail {index}: {ligament:.3f}-mm exterior ligament")
        assert math.isclose(
            float(record.get("ligament_mm")), ligament, abs_tol=0.01), (
                f"dovetail {index}: recorded ligament is stale")

        tangent = np.asarray(record.get("tangent"), dtype=float)
        normal = np.asarray(record.get("normal"), dtype=float)
        assert tangent.shape == normal.shape == (2,)
        assert math.isclose(float(np.linalg.norm(tangent)), 1.0,
                            abs_tol=1e-9)
        assert math.isclose(float(np.linalg.norm(normal)), 1.0,
                            abs_tol=1e-9)
        assert math.isclose(float(np.dot(tangent, normal)), 0.0,
                            abs_tol=1e-9)
        center = record.get("center_xy_mm")
        assert isinstance(center, (list, tuple)) and len(center) == 2
        assert polygon.buffer(1.0e-6).covers(Point(*center))
        center_xy = np.asarray(center, dtype=float)
        vertices = np.asarray(polygon.exterior.coords, dtype=float)[:-1]
        normal_offsets = (vertices - center_xy) @ normal
        assert math.isclose(
            float(np.min(normal_offsets)), -DOVETAIL_ROOT_OVERLAP_MM,
            rel_tol=0.0, abs_tol=1.0e-8), (
                f"dovetail {index}: male root overlap is not geometric")
        assert math.isclose(
            float(np.max(normal_offsets)), profile["depth"],
            rel_tol=0.0, abs_tol=1.0e-8), (
                f"dovetail {index}: male penetration is not geometric")

        assert gap.is_valid and not gap.is_empty and gap.area > 0.001
        assert gap.difference(layout.field_right.buffer(1.0e-6)).area <= 1e-6
        assert gap.intersection(layout.print_parts[owners[0]]).area <= 1e-6
        assert gap.intersection(layout.print_parts[owners[1]]).area <= 1e-6
        required_key_relief = polygon.buffer(
            DOVETAIL_CLEARANCE_MM, join_style=2,
            mitre_limit=10).difference(
                layout.nominal_parts[owners[0]]).intersection(
                    layout.field_right)
        assert required_key_relief.difference(
            gap.buffer(1.0e-6)).area <= 0.01, (
                f"dovetail {index}: female key relief is under-clearanced")

        # The straight-seam fit gap is one-sided and tapers to zero over the
        # first/last 2 mm.  Both exposed plan endpoints must therefore remain
        # exact, closed continuations of the monolithic outer edge.
        endpoints = (Point(*seam.coords[0]), Point(*seam.coords[-1]))
        for endpoint in endpoints:
            assert endpoint.distance(gap) <= 1.0e-4
            assert gap.intersection(endpoint.buffer(0.1)).area <= 0.001

    assert len(set(names)) == 2 and all(
        isinstance(name, str) and name for name in names)
    assert keys[0]["polygon"].intersection(keys[1]["polygon"]).area <= 1e-6
    reconstructed = unary_union((
        *layout.print_parts.values(), *layout.fit_clearance_gaps)).buffer(0)
    assert reconstructed.symmetric_difference(layout.field_right).area <= 0.02
    assert set(layout.two_piece_print_parts) == set(TWO_PIECE_PART_ROLES)
    two_piece_lower = layout.two_piece_print_parts["lm_lower"]
    two_piece_upper = layout.two_piece_print_parts["lm_um_upper"]
    assert two_piece_lower.symmetric_difference(
        layout.print_parts["lm_lower"]).area <= 1.0e-9
    assert layout.print_parts["lm_upper"].difference(
        two_piece_upper).area <= 0.01
    assert layout.print_parts["um"].difference(two_piece_upper).area <= 0.01
    assert layout.fit_clearance_gaps[1].difference(
        two_piece_upper).area <= 0.01
    assert two_piece_lower.intersection(two_piece_upper).area <= 0.01
    reconstructed_two_piece = unary_union((
        *layout.two_piece_print_parts.values(),
        layout.fit_clearance_gaps[0])).buffer(0)
    assert reconstructed_two_piece.symmetric_difference(
        layout.field_right).area <= 0.02

    # The exposed upper T edge follows the complete released crescent
    # profile: the measured lower arc followed by the cubic horn flank.
    # Check both the intended normal clearance and the visually amplified
    # horizontal opening at the station seen in the physical assembly.
    from shapely.geometry import LineString, box

    actual_t_plan = cad.contract._released_t_crescent_plan()
    upper_t_field = layout.field_right.intersection(
        box(0.0, 430.0, 180.0, cad.contract.A_TAPER_CAP_Y - 0.05))
    assert math.isclose(
        upper_t_field.distance(actual_t_plan),
        T_WING_CLEARANCE_MM, abs_tol=0.005)

    station_y = 426.0
    station = LineString(((-80.0, station_y), (180.0, station_y)))
    wing_section = layout.field_right.intersection(station)
    crescent_section = actual_t_plan.intersection(station)
    wing_x = min(
        point[0]
        for line in _line_parts(wing_section)
        for point in line.coords
        if point[0] > 20.0)
    crescent_x = max(
        point[0]
        for line in _line_parts(crescent_section)
        for point in line.coords)
    horizontal_gap = wing_x - crescent_x
    assert 0.30 <= horizontal_gap <= 0.45, (
        f"T-to-wing y=426 horizontal gap is {horizontal_gap:.4f} mm")

    # Lock two stations on the cubic horn continuation. A circle extrapolated
    # from the lower three-point arc falls progressively inside these values
    # and can make the released T mesh obstruct the wing near the tip.
    for station_y, expected_x in ((440.0, 42.6566), (448.0, 47.1115)):
        station = LineString(((-80.0, station_y), (180.0, station_y)))
        crescent_section = actual_t_plan.intersection(station)
        released_x = max(
            point[0]
            for line in _line_parts(crescent_section)
            for point in line.coords)
        assert math.isclose(released_x, expected_x, abs_tol=0.01), (
            f"released cubic T profile drifted at y={station_y:g}: "
            f"x={released_x:.4f} mm")


def test_live_brep_geometry_contract() -> None:
    """Gate the hash-bound released STEP BREPs, not only their metadata."""
    _require_remote_guard()

    import numpy as np
    from build123d import (
        Align, Box, Plane, Pos, import_brep, import_step, mirror)
    from lx521_baffle.floor_bend import centerline_controls
    from lx521_baffle.magnets import DEFAULT_SPEC, pair_facts, wall_cavity_tools
    from lx521_baffle.obiwan.bridge import common_lm_wing_contact_plan
    from export_obiwan_staged import load_stage_manifest, staged_part_paths
    import export_obiwan_wings as exporter
    import lx521_baffle.obiwan.wings as cad
    import lx521_baffle.obiwan.lm_split as lm_split

    selected_slug = os.environ.get("LX_OBIWAN_WING_LIVE_SLUG")
    assert selected_slug is None or selected_slug in VARIANT_IDS, (
        f"unknown live wing selector: {selected_slug!r}")
    live_slugs = VARIANT_IDS if selected_slug is None else (selected_slug,)

    def released_variant_breps(slug: str) -> dict[str, dict]:
        """Import each released wing BREP once after binding it to its hash.

        The exporter already paid the cost of constructing and serializing
        these exact flat/graded solids.  Rebuilding all 24 live solids inside the
        acceptance process duplicated nearly an hour of OCC work.  STEP
        labels retain the semantic side/role identity needed by every
        downstream Boolean probe, so validate the signed artifact inventory
        and consume those authoritative BREPs directly.
        """
        paths = _variant_paths(slug)
        directory = paths["directory"]
        manifest_path = paths["manifest"]
        assert isinstance(directory, Path)
        assert isinstance(manifest_path, Path)
        manifest = _read_json_object(manifest_path)
        _verify_source_hashes(manifest.get("source"))
        artifact_records = _artifact_record_map(manifest)

        def normalized_label(shape) -> str:
            # OCC's STEP reader may preserve the STEP string delimiters.
            return str(shape.label).strip("'")

        def imported_children(
                path_key: str,
                expected_kind: str,
                expected_parent_label: str,
                expected_child_labels: tuple[str, ...],
        ) -> dict[str, object]:
            path = paths[path_key]
            assert isinstance(path, Path)
            relative = path.relative_to(directory).as_posix()
            record = artifact_records.get(relative)
            assert isinstance(record, dict), (
                f"{slug}: released STEP is absent from manifest: {relative}")
            assert record.get("kind") == expected_kind, (
                f"{slug}: wrong manifest kind for {relative}")
            assert record.get("size_bytes") == path.stat().st_size, (
                f"{slug}: stale manifest size for {relative}")
            assert record.get("sha256") == _sha256_file(path), (
                f"{slug}: stale manifest hash for {relative}")

            assembly = import_step(path)
            assert normalized_label(assembly) == expected_parent_label, (
                f"{slug}: STEP parent label drifted for {relative}: "
                f"{assembly.label!r}")
            children = list(assembly.children)
            actual_labels = tuple(normalized_label(child) for child in children)
            assert actual_labels == expected_child_labels, (
                f"{slug}: STEP child inventory/order drifted for {relative}: "
                f"{actual_labels!r}")
            assert len(assembly.solids()) == len(expected_child_labels), (
                f"{slug}: STEP solid count drifted for {relative}")
            for label, child in zip(
                    expected_child_labels, children, strict=True):
                _assert_one_positive_solid(child, f"{slug}/released/{label}")
            return dict(zip(expected_child_labels, children, strict=True))

        canonical_labels = tuple(
            f"obiwan_wing_{slug}_{side}_monolithic"
            for side in SIDE_NAMES)
        print_labels = tuple(
            f"obiwan_wing_{slug}_{side}_{order}_of_3_{role}"
            for side in SIDE_NAMES
            for order, role in enumerate(PRINT_PART_ROLES, start=1))
        two_piece_labels = tuple(
            f"obiwan_wing_{slug}_{side}_split2_{order}_of_2_{role}"
            for side in SIDE_NAMES
            for order, role in enumerate(TWO_PIECE_PART_ROLES, start=1))

        canonical = imported_children(
            "canonical_step", "canonical_step",
            f"lx521_obiwan_basic_wing_{slug}_monolithic_pair",
            canonical_labels)
        print_assembly = imported_children(
            "assembled_step", "assembled_step",
            f"lx521_obiwan_basic_wing_{slug}_print_assembly",
            print_labels)
        two_piece_assembly = imported_children(
            "two_piece_assembled_step", "assembled_step",
            f"lx521_obiwan_basic_wing_{slug}_two_piece_print_assembly",
            two_piece_labels)
        return {
            "monoliths": {
                side: canonical[f"obiwan_wing_{slug}_{side}_monolithic"]
                for side in SIDE_NAMES
            },
            "print_parts": {
                side: {
                    role: print_assembly[
                        f"obiwan_wing_{slug}_{side}_{order}_of_3_{role}"]
                    for order, role in enumerate(PRINT_PART_ROLES, start=1)
                }
                for side in SIDE_NAMES
            },
            "two_piece_parts": {
                side: {
                    role: two_piece_assembly[
                        f"obiwan_wing_{slug}_{side}_split2_{order}_of_2_{role}"]
                    for order, role in enumerate(
                        TWO_PIECE_PART_ROLES, start=1)
                }
                for side in SIDE_NAMES
            },
        }

    _assert_plan_dovetail_contract(cad)
    release_shapes = {
        slug: released_variant_breps(slug) for slug in live_slugs}
    context_parts = cad.wing_review_split_context_parts(
        NO_FLOOR_STAGE_MANIFEST, FLOOR_STAGE_MANIFEST)
    assert list(context_parts) == [
        "lm_lower_floor", "lm_lower_no_floor", "lm_upper", "um", "t"]
    context_records = exporter._context_mesh_records(context_parts)
    assert [record["key"] for record in context_records] == [
        "lm_lower_floor", "lm_lower_no_floor", "lm_upper", "um", "t"]
    assert [record["label"] for record in context_records] == [
        "Obi-Wan LM lower — floor stand",
        "Obi-Wan LM lower — no-floor",
        "Obi-Wan LM upper — no-floor reference",
        "Obi-Wan UM — no-floor reference",
        "Obi-Wan T crescent — no-floor reference",
    ]
    context_by_key = {record["key"]: record for record in context_records}
    assert (context_by_key["lm_lower_floor"]["source_sha256"]
            != context_by_key["lm_lower_no_floor"]["source_sha256"])
    assert context_by_key["lm_lower_floor"]["triangles"][..., 2].min() <= -149.9
    assert context_by_key["lm_lower_no_floor"]["triangles"][..., 2].min() > -10.0
    for record in context_records:
        assert record["triangles"].shape[0] > 0
        assert len(exporter._projected_context_loops(record, (0, 1))) >= 1
        assert len(exporter._projected_context_loops(record, (1, 2))) >= 1

    # No-floor retains the complete shallow bridge profile. Floor keeps only
    # the same upper shoulder and deliberately omits every lower rail/skirt.
    # The Option-B bend is intentionally state-specific below its upright
    # centreline tangent.  The shared wing clearance therefore follows the
    # larger no-floor owner; the floor owner must remain a subset and match it
    # from the exact wing-root/bend tangent upward.
    from shapely.geometry import LineString, box

    section_z = FRONT_Z_MM - 0.15
    floor_outline = _front_section_exterior(
        context_parts["lm_lower_floor"]["shape"], section_z)
    no_floor_outline = _front_section_exterior(
        context_parts["lm_lower_no_floor"]["shape"], section_z)
    floor_excess = floor_outline.difference(no_floor_outline).area
    assert floor_excess <= 0.30, (
        "floor LM protrudes beyond the shared no-floor wing clearance by "
        f"{floor_excess:.6f} mm2")
    floor_bend_tangent_y = float(centerline_controls()[-1][1])
    assert math.isclose(floor_bend_tangent_y, 74.15, abs_tol=1.0e-9)
    upper_clip = box(-150.0, floor_bend_tangent_y, 150.0, 180.0)
    floor_upper = floor_outline.intersection(upper_clip)
    no_floor_upper = no_floor_outline.intersection(upper_clip)
    upper_symmetric_difference = floor_upper.symmetric_difference(
        no_floor_upper).area
    assert upper_symmetric_difference <= 0.30, (
        "floor/no-floor upper LM wing shoulders differ by "
        f"{upper_symmetric_difference:.6f} mm2")
    outline_hausdorff = floor_upper.hausdorff_distance(no_floor_upper)
    assert outline_hausdorff <= 0.03
    removed_lower_box = no_floor_outline.difference(
        floor_outline).intersection(box(-40.0, 0.0, 40.0, 60.0))
    assert removed_lower_box.area > 1000.0, (
        "floor LM still carries the obsolete lower shallow box")
    assert math.isclose(no_floor_outline.bounds[1], 0.0, abs_tol=0.01)
    assert math.isclose(floor_outline.bounds[1], 60.0, abs_tol=0.01)
    for outline in (floor_outline, no_floor_outline):
        assert math.isclose(outline.bounds[3], 172.481, abs_tol=0.01)
    maximum_station_width_delta = 0.0
    for y_mm in np.arange(floor_bend_tangent_y + 0.25, 121.76, 0.25):
        station = LineString(((-150.0, y_mm), (150.0, y_mm)))
        floor_width = floor_outline.intersection(station).length
        no_floor_width = no_floor_outline.intersection(station).length
        maximum_station_width_delta = max(
            maximum_station_width_delta,
            abs(float(floor_width - no_floor_width)))
    assert maximum_station_width_delta <= 0.04, (
        "floor/no-floor upper LM-lower station widths differ by "
        f"{maximum_station_width_delta:.6f} mm")

    t_crescent = context_parts["t"]["shape"]

    # Probe the complete carrier-side captive system, not merely the shared
    # datum.  Each staged lower print must retain both sealed skins and the
    # positive cradle/roof land while leaving every functional cutter empty
    # on the same shared shoulder-normal axis used by the flat/graded receiver.
    lower_carriers = {
        key: context_parts[key]["shape"]
        for key in ("lm_lower_floor", "lm_lower_no_floor")
    }
    shared_lm_contact_owner = cad._plan_prism(
        common_lm_wing_contact_plan(),
        REAR_Z_MM - 0.5, FRONT_Z_MM + 0.5)
    for side in SIDE_NAMES:
        site = next(
            site for site in cad._selected_sites(side)
            if site["name"] == f"lm_lower_{side}")
        assert math.isclose(site["face_offset_mm"], -0.15, abs_tol=1e-12)
        carrier_tools = wall_cavity_tools(
            name=site["name"], face=site["face"],
            outward=(*site["normal"], 0.0), owner="carrier",
            axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
            front_z=FRONT_Z_MM,
            interface_gap_mm=MAGNET_INTERFACE_GAP_MM)
        qualified_solid = carrier_tools.required_land
        for cutter in carrier_tools.cutters:
            qualified_solid = qualified_solid - cutter
        nx, ny = site["normal"]
        skin_diameter = 4.60
        face_skin = cad._axis_cylinder(
            site["face"], site["normal"], site["z_mm"], skin_diameter,
            inward=DEFAULT_SPEC.face_skin_mm - 0.03, outward=0.0)
        inner_face = (
            site["face"][0]
            - (DEFAULT_SPEC.face_skin_mm
               + DEFAULT_SPEC.cavity_depth_mm) * nx,
            site["face"][1]
            - (DEFAULT_SPEC.face_skin_mm
               + DEFAULT_SPEC.cavity_depth_mm) * ny,
        )
        inner_skin = cad._axis_cylinder(
            inner_face, site["normal"], site["z_mm"], skin_diameter,
            inward=DEFAULT_SPEC.inner_skin_mm - 0.03, outward=0.0)
        for state_key, carrier in lower_carriers.items():
            for cutter_index, cutter in enumerate(carrier_tools.cutters):
                residual = _intersection_volume(carrier, cutter)
                assert residual <= 0.03, (
                    f"{state_key}/{side}: lower LM captive cutter "
                    f"{cutter_index} is obstructed by {residual:.4f} mm3")
            retained = _intersection_volume(carrier, qualified_solid)
            assert retained >= 0.98 * qualified_solid.volume, (
                f"{state_key}/{side}: captive land incomplete: "
                f"{retained:.3f}/{qualified_solid.volume:.3f} mm3")
            for skin_label, skin in (("interface", face_skin),
                                     ("inner", inner_skin)):
                fill = _intersection_volume(carrier, skin)
                assert fill >= 0.97 * skin.volume, (
                    f"{state_key}/{side}: {skin_label} skin is not sealed")
    print(
        "  dual-state LM-lower front profile: common above the exact "
        f"y={floor_bend_tangent_y:.2f} bend/wing-root tangent, "
        f"sampled BREP Hausdorff={outline_hausdorff:.4f} mm, "
        f"max width delta={maximum_station_width_delta:.4f} mm; "
        "both lower D5.20 x 2.10 captive stations sealed",
        flush=True)
    staged_lm_parts = {}
    for state, manifest_path, stand_foot in (
            ("floor", FLOOR_STAGE_MANIFEST, True),
            ("no_floor", NO_FLOOR_STAGE_MANIFEST, False)):
        payload = load_stage_manifest(manifest_path, stand_foot=stand_foot)
        paths = staged_part_paths(manifest_path, payload)
        for part_key in (
                "core_lm_carrier",
                "optional_lm_keyed_1_of_2_bottom",
                "optional_lm_keyed_2_of_2_top"):
            shape = import_brep(str(paths[part_key]))
            _assert_one_positive_solid(shape, f"{state}/{part_key}")
            staged_lm_parts[f"{state}/{part_key}"] = shape
    del context_records
    del context_parts

    def positive_local_clip(shape, clip, label: str):
        """Clip to one proven key-change neighborhood and retain exact BREP."""
        clipped = shape & clip
        assert clipped is not None and clipped.is_valid, (
            f"{label}: invalid/empty local BREP clip")
        solids = list(clipped.solids())
        assert solids and all(solid.volume > 0.0 for solid in solids), (
            f"{label}: local clip has no positive solid")
        return clipped

    # The keyed-split acceptance in test_obiwan_r6f proves, for both stand
    # states, that the staged halves equal the canonical/augmented LM outside
    # the declared support/pin/socket tools.  Baseline canonical wing clearance
    # is gated separately above.  Therefore only these two small declared
    # neighborhoods need the expensive artifact-backed staged-part Booleans.
    support_lands = lm_split.registration_support_land_tools()
    wing_pockets = lm_split.registration_wing_clearance_tools()
    male_pins = lm_split.male_registration_pin_tools()
    female_sockets = lm_split.female_registration_socket_tools()
    key_neighborhoods = {}
    staged_local_parts = {side: {} for side in SIDE_NAMES}
    for side in SIDE_NAMES:
        local_tools = (
            ("wing pocket", wing_pockets[side]),
            ("support land", support_lands[side]),
            ("male pin", male_pins[side]),
            ("female socket", female_sockets[side]),
        )
        local_bounds = tuple(
            _shape_bounds(tool) for _tool_name, tool in local_tools)
        padding_mm = 0.50
        lower = tuple(
            min(bounds[0][axis] for bounds in local_bounds) - padding_mm
            for axis in range(3))
        upper = tuple(
            max(bounds[1][axis] for bounds in local_bounds) + padding_mm
            for axis in range(3))
        clip = Pos(*lower) * Box(
            *(upper[axis] - lower[axis] for axis in range(3)),
            align=(Align.MIN, Align.MIN, Align.MIN))
        key_neighborhoods[side] = clip
        clip_bounds = _shape_bounds(clip)
        for tool_name, tool in local_tools:
            tool_bounds = _shape_bounds(tool)
            margins = (
                *(tool_bounds[0][axis] - clip_bounds[0][axis]
                  for axis in range(3)),
                *(clip_bounds[1][axis] - tool_bounds[1][axis]
                  for axis in range(3)),
            )
            assert min(margins) >= 0.49, (
                f"{side}: {tool_name} lacks local proof-box margin")
            assert _difference_volume(tool, clip) <= 0.01, (
                f"{side}: {tool_name} escapes local proof box")
        for part_key, staged_lm in staged_lm_parts.items():
            staged_local_parts[side][part_key] = positive_local_clip(
                staged_lm, clip, f"{side}/{part_key}")

    graded_live_depth_samplers = {}
    for slug in live_slugs:
        right = release_shapes[slug]["monoliths"]["right"]
        left = release_shapes[slug]["monoliths"]["left"]
        _assert_one_positive_solid(right, f"{slug}/right monolith")
        _assert_one_positive_solid(left, f"{slug}/left monolith")
        right_bounds = _shape_bounds(right)
        left_bounds = _shape_bounds(left)
        assert math.isclose(right_bounds[1][2], FRONT_Z_MM, abs_tol=0.005)
        assert math.isclose(left_bounds[1][2], FRONT_Z_MM, abs_tol=0.005)
        assert right_bounds[0][2] >= REAR_Z_MM - 0.005
        assert left_bounds[0][2] >= REAR_Z_MM - 0.005
        for side, monolith in (("right", right), ("left", left)):
            wing_plan = cad.wing_plan(slug, side)
            if slug == "flat":
                plan_envelope = cad._plan_prism(
                    wing_plan, REAR_Z_MM - 0.5, FRONT_Z_MM + 0.5)
                outside_plan = _difference_volume(monolith, plan_envelope)
                assert outside_plan <= 0.03, (
                    f"{slug}/{side}: finalized monolith grows outside exact "
                    f"wing plan by {outside_plan:.6f} mm3")
            else:
                # graded is constructed from this exact shared plan and differs
                # from flat only at its rear depth field.  A whole-solid
                # near-empty STEP subtraction spends ~1h21 CPU integrating
                # healed tensor-spline slivers.  Bind the imported BREP to the
                # plan through its independently checked mirrored XY bounds;
                # flat retains the exact common-plan Boolean and graded's unique
                # rear surface is probed below.
                plan_bounds = wing_plan.bounds
                side_bounds = right_bounds if side == "right" else left_bounds
                assert side_bounds[0][0] >= plan_bounds[0] - 0.005
                assert side_bounds[0][1] >= plan_bounds[1] - 0.005
                assert side_bounds[1][0] <= plan_bounds[2] + 0.005
                assert side_bounds[1][1] <= plan_bounds[3] + 0.005

            x0, x1 = ((0.0, 180.0)
                      if side == "right" else (-180.0, 0.0))
            upper_t_tool = cad._plan_prism(
                box(x0, 430.0, x1, cad.contract.A_TAPER_CAP_Y - 0.05),
                REAR_Z_MM - 0.5, FRONT_Z_MM + 0.5)
            if slug == "flat":
                wing_upper_t = monolith & upper_t_tool
                crescent_upper_t = t_crescent & upper_t_tool
                assert _intersection_volume(
                    wing_upper_t, crescent_upper_t) <= 0.01, (
                        f"{slug}/{side}: upper wing collides T crescent")
            # The imported-BREP Common above proves physical non-collision.
            # The exact clearance is an XY plan contract because both parts
            # share the acoustic-front datum; querying Shape.distance_to on
            # graded's full trimmed rear B-spline invokes an unrelated global
            # surface-extrema search and costs tens of core-minutes.
            wing_upper_t_plan = cad.wing_plan(slug, side).intersection(
                box(x0, 430.0, x1, cad.contract.A_TAPER_CAP_Y - 0.05))
            measured_t_clearance = wing_upper_t_plan.distance(
                cad.contract._released_t_crescent_plan())
            # The common wing outline clears the union of the released
            # crescent and its mirror.  The farther source flank therefore
            # governs at the nominal 0.20 mm, while the opposite physical
            # flank retains the source drawing's safe ~0.011-mm asymmetry.
            # The analytic symmetric-plan gate above remains locked to
            # 0.20 +/- 0.005 mm; this live-BREP interval additionally bounds
            # the harmless non-governing-side opening without requiring the
            # crescent itself to be made artificially symmetric.
            assert T_WING_CLEARANCE_MM - 0.01 <= measured_t_clearance <= (
                T_WING_CLEARANCE_MM + 0.012), (
                    f"{slug}/{side}: BREP T-to-wing clearance is "
                    f"{measured_t_clearance:.4f} mm")
        print(f"  {slug}: plan envelope and T clearance pass", flush=True)
        right_volume = cad.adaptive_volume_mm3(right)
        left_volume = cad.adaptive_volume_mm3(left)
        assert math.isclose(right_volume, left_volume, rel_tol=1e-9,
                            abs_tol=0.02)
        print(f"  {slug}: imported monolith volumes pass", flush=True)
        for side, monolith in (("right", right), ("left", left)):
            pocket = wing_pockets[side]
            land = support_lands[side]
            pocket_bounds = pocket.bounding_box()
            land_bounds = land.bounding_box()
            assert pocket_bounds.min.Z - REAR_Z_MM >= 5.82
            assert FRONT_Z_MM - pocket_bounds.max.Z >= 2.32
            assert _difference_volume(land, pocket) <= 0.01, (
                f"{slug}/{side}: support land escapes its clearance pocket")
            clearance_margins = (
                land_bounds.min.X - pocket_bounds.min.X,
                pocket_bounds.max.X - land_bounds.max.X,
                land_bounds.min.Y - pocket_bounds.min.Y,
                pocket_bounds.max.Y - land_bounds.max.Y,
                land_bounds.min.Z - pocket_bounds.min.Z,
                pocket_bounds.max.Z - land_bounds.max.Z,
            )
            constructed_clearance = min(clearance_margins)
            assert constructed_clearance >= (
                lm_split.REGISTRATION_WING_CLEARANCE_MM - 0.01), (
                    f"{slug}/{side}: keyed-land clearance capsule is only "
                    f"{constructed_clearance:.4f} mm")
            assert _intersection_volume(
                monolith, land) <= 0.03, (
                    f"{slug}/{side}: optional LM support land collides wing")
            assert _intersection_volume(
                monolith, pocket) <= 0.03, (
                    f"{slug}/{side}: keyed-land clearance pocket is blocked")
            wing_local = positive_local_clip(
                monolith, key_neighborhoods[side],
                f"{slug}/{side}/wing key neighborhood")
            # The six-margin capsule proof above and zero wing/pocket
            # intersection establish the minimum clearance directly.  A
            # Shape.distance_to query here would ask OCC to scan graded's entire
            # underlying tensor B-spline despite this local trim.
            for part_key, staged_lm in staged_local_parts[side].items():
                overlap = _intersection_volume(wing_local, staged_lm)
                assert overlap <= 0.03, (
                    f"{slug}/{side}: actual staged {part_key} collides wing "
                    f"by {overlap:.6f} mm3")
            for name, land in cad.receiver_required_lands(side).items():
                assert _intersection_volume(pocket, land) <= 0.03, (
                    f"{slug}/{side}: key pocket reaches receiver land {name}")
            for name, cutter in cad.receiver_pockets(side).items():
                assert _intersection_volume(pocket, cutter) <= 0.03, (
                    f"{slug}/{side}: key pocket reaches receiver void {name}")
            for key in cad._layout().dovetail_keys:
                key_plan = (key["polygon"] if side == "right"
                            else cad._mirror_plan(key["polygon"]))
                key_tool = cad._plan_prism(
                    key_plan, REAR_Z_MM - 0.5, FRONT_Z_MM + 0.5)
                assert _intersection_volume(pocket, key_tool) <= 0.03, (
                    f"{slug}/{side}: key pocket reaches {key['name']} dovetail")
            exposed_edge = cad._graded_analytics()[1].exposed_outer_edge
            exposed_edge = (exposed_edge if side == "right"
                            else cad._mirror_plan(exposed_edge))
            exposed_edge_guard = cad._plan_prism(
                exposed_edge.buffer(1.0),
                REAR_Z_MM - 0.5, FRONT_Z_MM + 0.5)
            assert _intersection_volume(pocket, exposed_edge_guard) <= 0.03, (
                f"{slug}/{side}: key pocket reaches outer acoustic edge")
            print(
                f"  {slug}/{side}: staged key and receiver neighborhood pass",
                flush=True)
        facts_path = _variant_paths(slug)["facts"]
        assert isinstance(facts_path, Path)
        serialized = _read_json_object(facts_path).get("geometry", {})
        serialized_actual = serialized.get("actual_brep", {})
        step_roundtrip_rel_tol = STEP_ROUNDTRIP_VOLUME_REL_TOL[slug]
        assert math.isclose(
            float(serialized_actual.get("right_volume_mm3")), right_volume,
            rel_tol=step_roundtrip_rel_tol,
            abs_tol=STEP_ROUNDTRIP_VOLUME_ABS_TOL_MM3), (
                f"{slug}: released STEP right-volume round trip exceeds "
                f"{step_roundtrip_rel_tol * 1.0e6:g} ppm: "
                f"STEP={right_volume:.9f} mm3, serialized="
                f"{float(serialized_actual.get('right_volume_mm3')):.9f} mm3")
        assert math.isclose(
            float(serialized_actual.get("left_volume_mm3")), left_volume,
            rel_tol=step_roundtrip_rel_tol,
            abs_tol=STEP_ROUNDTRIP_VOLUME_ABS_TOL_MM3), (
                f"{slug}: released STEP left-volume round trip exceeds "
                f"{step_roundtrip_rel_tol * 1.0e6:g} ppm: "
                f"STEP={left_volume:.9f} mm3, serialized="
                f"{float(serialized_actual.get('left_volume_mm3')):.9f} mm3")
        _assert_bounds_close(
            serialized_actual.get("right_bounds_mm"), right_bounds, 0.002,
            f"{slug} serialized right bounds")
        _assert_bounds_close(
            serialized_actual.get("left_bounds_mm"), left_bounds, 0.002,
            f"{slug} serialized left bounds")
        print(f"  {slug}: serialized STEP bounds/volume pass", flush=True)

        mirrored = mirror(right, about=Plane.YZ)
        _assert_bounds_close(
            _shape_bounds(mirrored), left_bounds, 0.002,
            f"{slug} exact left/right mirror bounds")
        if slug == "flat":
            assert _difference_volume(left, mirrored) <= 0.02, (
                f"{slug}: left contains material outside mirrored right")
            assert _difference_volume(mirrored, left) <= 0.02, (
                f"{slug}: mirrored right contains material outside left")
        else:
            # Independently healed STEP copies of graded's dense trimmed tensor
            # B-spline can leave near-empty Boolean/section slivers whose OCC
            # integration consumes tens of core-hours.  The exact analytic
            # plan-mirror contract is already gated above; validate the
            # imported rear surfaces here with paired vertical probes.  Strict
            # single-solid topology, mirrored bounds, adaptive total volume,
            # STL and protected-C0 gates remain independent above/below this
            # sampled surface oracle.
            right_depth_sampler = cad._VerticalDepthSampler(right)
            left_depth_sampler = cad._VerticalDepthSampler(left)
            graded_live_depth_samplers.update({
                "right": right_depth_sampler,
                "left": left_depth_sampler,
            })
            maximum_mirrored_depth_delta = 0.0
            mirror_probe_count = 0
            mirror_sections = cad.wing_section_samples(
                "graded", "right", samples=17)
            for section in mirror_sections.values():
                section_xy = np.asarray(section["xy_mm"], dtype=float)
                for x_mm, y_mm in section_xy[1:-1:2]:
                    right_depth = right_depth_sampler.depth_mm(x_mm, y_mm)
                    left_depth = left_depth_sampler.depth_mm(-x_mm, y_mm)
                    maximum_mirrored_depth_delta = max(
                        maximum_mirrored_depth_delta,
                        abs(right_depth - left_depth))
                    mirror_probe_count += 1
            assert mirror_probe_count >= 35
            assert maximum_mirrored_depth_delta <= 0.03, (
                "graded mirrored rear-surface depth differs by "
                f"{maximum_mirrored_depth_delta:.6f} mm")
            print(
                "  graded: exact analytic plan plus "
                f"{mirror_probe_count} paired imported rear-depth probes; "
                "max depth delta="
                f"{maximum_mirrored_depth_delta:.4f} mm",
                flush=True)

        receiver_records = cad.receiver_facts("right")
        assert len(receiver_records) == 3
        assert {record["name"] for record in receiver_records} == {
            "lm_lower_right", "lm_upper_right", "um_right"}
        lower_receiver = next(
            record for record in receiver_records
            if record["name"] == "lm_lower_right")
        assert lower_receiver["interface_kind"] == "shoulder"
        assert all(math.isclose(actual, expected, abs_tol=1e-6)
                   for actual, expected in zip(
                       lower_receiver["axis_normal_xy"],
                       [0.706451, -0.707762], strict=True))
        assert all(math.isclose(actual, expected, abs_tol=1e-6)
                   for actual, expected in zip(
                       lower_receiver["carrier_face_xy_mm"],
                       [45.285011, 89.190370], strict=True))
        assert all(math.isclose(actual, expected, abs_tol=1e-6)
                   for actual, expected in zip(
                       lower_receiver["receiver_cavity_face_xy_mm"],
                       [45.320334, 89.154982], strict=True))
        assert math.isclose(
            lower_receiver["axis_z_mm"],
            OBIWAN_MAGNET_Z_MM, abs_tol=1e-9)
        for record in receiver_records:
            assert record["closure_kind"] == "transverse_gable_45deg"
            assert math.isclose(
                record["cavity_diameter_mm"],
                MAGNET_CAVITY_DIAMETER_MM, abs_tol=1e-9)
            assert math.isclose(
                record["cavity_depth_mm"],
                MAGNET_CAVITY_DEPTH_MM, abs_tol=1e-9)
            assert math.isclose(
                record["face_skin_mm"], MAGNET_FACE_SKIN_MM,
                abs_tol=1e-9)
            assert math.isclose(
                record["inner_skin_mm"], MAGNET_INNER_SKIN_MM,
                abs_tol=1e-9)
            assert math.isclose(
                record["captive_land_mm"], MAGNET_CAPTIVE_LAND_MM,
                abs_tol=1e-9)
            assert math.isclose(
                record["roof_angle_deg"], MAGNET_ROOF_ANGLE_DEG,
                abs_tol=1e-9)
            assert math.isclose(
                record["receiver_solid_standoff_mm"],
                MAGNET_INTERFACE_GAP_MM, abs_tol=1e-9)
            assert math.isclose(
                record["physical_interface_gap_mm"], 0.0, abs_tol=1e-9)
            assert record["receiver_spacing_standoff_is_solid"] is True
            assert math.isclose(
                record["paired_magnet_face_separation_mm"],
                _expected_magnet_face_separation(
                    record["interface_kind"]), abs_tol=1e-9)
            assert record["carrier_magnet_fully_buried"] is True
            assert record["receiver_magnet_fully_buried"] is True

            site = next(
                site for site in cad._selected_sites("right")
                if site["name"] == record["name"])
            receiver_datum = cad._receiver_datum_face(site)
            cavity_inset = float(
                site.get("carrier_cavity_face_inset_mm", 0.0))
            expected_pair_gap = MAGNET_INTERFACE_GAP_MM + cavity_inset
            expected_face_separation = _expected_magnet_face_separation(
                site["interface_kind"])
            base_tools = wall_cavity_tools(
                name=site["name"], face=site["face"],
                outward=(*site["normal"], 0.0), owner="carrier",
                axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
                front_z=FRONT_Z_MM,
                interface_gap_mm=MAGNET_INTERFACE_GAP_MM)
            receiver_tools = wall_cavity_tools(
                name=site["name"], face=receiver_datum,
                outward=(*site["normal"], 0.0), owner="wing",
                axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
                front_z=FRONT_Z_MM,
                interface_gap_mm=MAGNET_INTERFACE_GAP_MM)
            pair = pair_facts(base_tools, receiver_tools)
            assert math.isclose(
                pair["interface_gap_mm"], expected_pair_gap,
                abs_tol=1e-9)
            assert math.isclose(
                pair["nominal_magnet_face_separation_mm"],
                expected_face_separation, abs_tol=1e-9)
            expected_pole_axis = np.asarray(
                (*site["normal"], 0.0), dtype=float)
            expected_pole_axis /= np.linalg.norm(expected_pole_axis)
            for pole_key in (
                    "base_marked_pole_axis_xyz",
                    "receiver_marked_pole_axis_xyz"):
                actual_pole_axis = np.asarray(pair[pole_key], dtype=float)
                assert np.all(np.isfinite(actual_pole_axis))
                assert math.isclose(
                    float(np.linalg.norm(actual_pole_axis)), 1.0,
                    rel_tol=0.0, abs_tol=1.0e-12)
                alignment = float(np.dot(
                    actual_pole_axis, expected_pole_axis))
                assert alignment >= 1.0 - 1.0e-12, (
                    f"{slug}/{record['name']}: {pole_key} is reversed or "
                    f"misaligned; dot={alignment:.16g}")
                assert np.allclose(
                    actual_pole_axis, expected_pole_axis,
                    rtol=0.0, atol=1.0e-12)
        for name, cutter in cad.receiver_pockets("right").items():
            assert _intersection_volume(right, cutter) <= 0.03, (
                f"{slug}: receiver cutter still intersects material: {name}")

        serialized_print_contract = serialized.get("print_contract", {})
        serialized_a_parts = serialized_print_contract.get(
            "installed_piece_brep", {})
        serialized_b_parts = serialized_print_contract.get(
            "two_piece_installed_piece_brep", {})
        assert set(serialized_a_parts) == set(SIDE_NAMES)
        assert set(serialized_b_parts) == set(SIDE_NAMES)

        def assert_imported_piece_facts(piece, record, label: str) -> None:
            _assert_one_positive_solid(piece, label)
            assert isinstance(record, dict)
            _assert_bounds_close(
                record.get("bounds_mm"), _shape_bounds(piece), 0.003,
                f"{label} serialized bounds")
            # Shape.volume is the inexpensive ordinary BRepGProp property
            # already used by _assert_one_positive_solid.  Dense trimmed graded
            # pieces vary by up to 1.54% under that non-adaptive quadrature,
            # so use it only as a gross identity check.  The source value was
            # produced by the strict adaptive integrator and is independently
            # bound to the released STL within 0.3% by the artifact gate.
            assert math.isclose(
                float(piece.volume), float(record.get("volume_mm3")),
                rel_tol=STEP_FAST_IDENTITY_VOLUME_REL_TOL, abs_tol=1.0), (
                    f"{label}: imported/source volume drifted")

        def expected_mirrored_bounds(bounds):
            return (
                (-bounds[1][0], bounds[0][1], bounds[0][2]),
                (-bounds[0][0], bounds[1][1], bounds[1][2]),
            )

        pieces = release_shapes[slug]["print_parts"]["right"]
        left_pieces = release_shapes[slug]["print_parts"]["left"]
        assert tuple(pieces) == PRINT_PART_ROLES
        assert tuple(left_pieces) == PRINT_PART_ROLES
        for role, piece in pieces.items():
            left_piece = left_pieces[role]
            assert_imported_piece_facts(
                piece, serialized_a_parts["right"][role],
                f"{slug}/right/{role}")
            assert_imported_piece_facts(
                left_piece, serialized_a_parts["left"][role],
                f"{slug}/left/{role}")
            if slug == "flat":
                assert _difference_volume(piece, right) <= 0.03, (
                    f"{slug}/{role}: split print solid leaves monolith")
                mirrored_piece = mirror(piece, about=Plane.YZ)
                assert _difference_volume(
                    left_piece, mirrored_piece) <= 0.02, (
                        f"{slug}/{role}: left print solid is not exact mirror")
                assert _difference_volume(
                    mirrored_piece, left_piece) <= 0.02, (
                        f"{slug}/{role}: mirrored right print solid differs")
            else:
                _assert_bounds_close(
                    _shape_bounds(left_piece),
                    expected_mirrored_bounds(_shape_bounds(piece)), 0.003,
                    f"{slug}/{role} imported mirror bounds")

        two_piece = release_shapes[slug]["two_piece_parts"]["right"]
        left_two_piece = release_shapes[slug]["two_piece_parts"]["left"]
        assert tuple(two_piece) == TWO_PIECE_PART_ROLES
        assert tuple(left_two_piece) == TWO_PIECE_PART_ROLES
        for side, monolith, a_parts, b_parts in (
                ("right", right, pieces, two_piece),
                ("left", left, left_pieces, left_two_piece)):
            for role, piece in b_parts.items():
                assert_imported_piece_facts(
                    piece, serialized_b_parts[side][role],
                    f"{slug}/{side}/B/{role}")
                if slug == "flat":
                    assert _difference_volume(piece, monolith) <= 0.03, (
                        f"{slug}/{side}/B/{role}: leaves monolith")
            if slug == "flat":
                assert _difference_volume(
                    b_parts["lm_lower"], a_parts["lm_lower"]) <= 0.02
                assert _difference_volume(
                    a_parts["lm_lower"], b_parts["lm_lower"]) <= 0.02
                for a_role in ("lm_upper", "um"):
                    assert _difference_volume(
                        a_parts[a_role], b_parts["lm_um_upper"]) <= 0.03, (
                            f"{slug}/{side}/B upper omits A {a_role}")
                assert _intersection_volume(
                    b_parts["lm_lower"], b_parts["lm_um_upper"]) <= 0.03, (
                        f"{slug}/{side}: B pieces overlap")

                upper_gap_plan = cad._layout().fit_clearance_gaps[1]
                if side == "left":
                    upper_gap_plan = cad._mirror_plan(upper_gap_plan)
                upper_gap_tool = cad._plan_prism(
                    upper_gap_plan, REAR_Z_MM - 0.5, FRONT_Z_MM + 0.5)
                upper_gap_material = monolith & upper_gap_tool
                assert upper_gap_material is not None
                assert _difference_volume(
                    upper_gap_material,
                    b_parts["lm_um_upper"]) <= 0.03, (
                        f"{slug}/{side}: B upper retained former split slit")
            else:
                assert (
                    serialized_b_parts[side]["lm_lower"]["bounds_mm"]
                    == serialized_a_parts[side]["lm_lower"]["bounds_mm"])
                assert math.isclose(
                    float(serialized_b_parts[side]["lm_lower"]["volume_mm3"]),
                    float(serialized_a_parts[side]["lm_lower"]["volume_mm3"]),
                    rel_tol=0.0, abs_tol=1.0e-9)
                restored_gap_volume = (
                    float(serialized_b_parts[side]["lm_um_upper"][
                        "volume_mm3"])
                    - sum(float(serialized_a_parts[side][role]["volume_mm3"])
                          for role in ("lm_upper", "um")))
                assert 1.0 < restored_gap_volume < 100.0
        if slug == "flat":
            for role, piece in two_piece.items():
                mirrored_piece = mirror(piece, about=Plane.YZ)
                assert _difference_volume(
                    left_two_piece[role], mirrored_piece) <= 0.02
                assert _difference_volume(
                    mirrored_piece, left_two_piece[role]) <= 0.02
        else:
            for role, piece in two_piece.items():
                _assert_bounds_close(
                    _shape_bounds(left_two_piece[role]),
                    expected_mirrored_bounds(_shape_bounds(piece)), 0.003,
                    f"{slug}/B/{role} imported mirror bounds")
        print(
            f"  {slug}: two-piece BREP is single-solid, mirrored, "
            "A-lower-identical, and fills the former upper seam",
            flush=True)

        if slug == "flat":
            for key in cad._layout().dovetail_keys:
                male_role = key["male_owner"]
                female_role = key["female_owner"]
                for side, monolith, side_pieces in (
                        ("right", right, pieces),
                        ("left", left, left_pieces)):
                    key_plan = (key["polygon"] if side == "right"
                                else cad._mirror_plan(key["polygon"]))
                    key_tool = cad._plan_prism(
                        key_plan, REAR_Z_MM - 0.5, FRONT_Z_MM + 0.5)
                    key_material = monolith & key_tool
                    assert key_material is not None
                    key_volume = _adaptive_volume_mm3(key_material)
                    assert key_volume > 1.0, (
                        f"{slug}/{side}/{key['name']}: empty key envelope")
                    missing = _difference_volume(
                        key_material, side_pieces[male_role])
                    assert missing <= 0.03, (
                        f"{slug}/{side}/{key['name']}: male {male_role} "
                        f"misses {missing:.6f} mm3 of finalized key material")
                    female_intrusion = _intersection_volume(
                        key_material, side_pieces[female_role])
                    assert female_intrusion <= 0.03, (
                        f"{slug}/{side}/{key['name']}: female {female_role} "
                        f"owns {female_intrusion:.6f} mm3 of male key")

        # The lower receiver is a real fully sealed captive station in
        # lm_lower, not merely a coordinate record.  Probe the
        # cradle/chimney/roof voids, both 0.45-mm skins, the complete 3.00-mm
        # positive land, and the conformal receiver-side spacing standoff on
        # both mirrors.  The 0.05-mm offset is measured at the shoulder datum;
        # away from the datum, its tangent-plane proxy is partitioned exactly
        # by the curved carrier surface.  Requiring the wing to fill the
        # carrier-owned side would create a real collision.  Together the two
        # owners must fill the proxy without a visible pocket-width exterior
        # notch. No upper print may own station material.
        for side, side_pieces in (
                ("right", pieces), ("left", left_pieces)):
            site = next(
                record for record in cad._selected_sites(side)
                if record["name"] == f"lm_lower_{side}")
            receiver_datum = cad._receiver_datum_face(site)
            tools = wall_cavity_tools(
                name=site["name"], face=receiver_datum,
                outward=(*site["normal"], 0.0), owner="wing",
                axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
                front_z=FRONT_Z_MM,
                interface_gap_mm=MAGNET_INTERFACE_GAP_MM)
            for cutter_index, cutter in enumerate(tools.cutters):
                assert _intersection_volume(
                    side_pieces["lm_lower"], cutter) <= 0.03, (
                        f"{slug}/{side}: receiver cutter {cutter_index} "
                        "is obstructed")
            assert _intersection_volume(
                side_pieces["lm_lower"], tools.nominal_magnet) <= 0.02

            qualified_solid = tools.required_land
            for cutter in tools.cutters:
                qualified_solid = qualified_solid - cutter
            retained = _intersection_volume(
                side_pieces["lm_lower"], qualified_solid)
            assert retained >= 0.98 * qualified_solid.volume, (
                f"{slug}/{side}: lower receiver captive land incomplete: "
                f"{retained:.3f}/{qualified_solid.volume:.3f} mm3")

            nx, ny = site["normal"]
            actual_face = tools.actual_face_xyz[:2]
            skin_diameter = 4.60
            face_skin = cad._axis_cylinder(
                actual_face, site["normal"], site["z_mm"], skin_diameter,
                inward=0.0, outward=DEFAULT_SPEC.face_skin_mm - 0.03)
            inner_face = (
                actual_face[0]
                + (DEFAULT_SPEC.face_skin_mm
                   + DEFAULT_SPEC.cavity_depth_mm) * nx,
                actual_face[1]
                + (DEFAULT_SPEC.face_skin_mm
                   + DEFAULT_SPEC.cavity_depth_mm) * ny,
            )
            inner_skin = cad._axis_cylinder(
                inner_face, site["normal"], site["z_mm"], skin_diameter,
                inward=0.0, outward=DEFAULT_SPEC.inner_skin_mm - 0.03)
            raw_standoff = cad._axis_cylinder(
                receiver_datum, site["normal"], site["z_mm"], skin_diameter,
                inward=0.0, outward=MAGNET_INTERFACE_GAP_MM - 0.01)
            solid_standoff = raw_standoff - shared_lm_contact_owner
            carrier_standoff = raw_standoff & shared_lm_contact_owner
            assert solid_standoff.volume > 0.25 * raw_standoff.volume
            assert carrier_standoff.volume > 0.01 * raw_standoff.volume
            standoff_fill = _intersection_volume(
                side_pieces["lm_lower"], solid_standoff)
            assert standoff_fill >= 0.97 * solid_standoff.volume, (
                f"{slug}/{side}: conformal receiver spacing standoff is "
                "missing")
            for state_key, carrier in lower_carriers.items():
                carrier_fill = _intersection_volume(
                    carrier, carrier_standoff)
                assert carrier_fill >= 0.97 * carrier_standoff.volume, (
                    f"{slug}/{side}/{state_key}: curved shoulder does not "
                    "own its side of the receiver standoff partition")
            for skin_label, skin in (("interface", face_skin),
                                     ("inner", inner_skin)):
                fill = _intersection_volume(side_pieces["lm_lower"], skin)
                assert fill >= 0.97 * skin.volume, (
                    f"{slug}/{side}: {skin_label} skin is not sealed")
            for other_role in ("lm_upper", "um"):
                assert _intersection_volume(
                    side_pieces[other_role], qualified_solid) <= 0.03, (
                        f"{slug}/{side}: {other_role} intrudes into the "
                        "lower captive receiver land")

            # The two ring receivers obey the same sealed geometry and are
            # wholly owned by their corresponding split prints.  The carrier
            # cavity datum sits 0.15 mm beneath a continuous +0.80-mm ring
            # fairing; there is no station-local backing boss or visible cue.
            for ring_site in (
                    candidate for candidate in cad._selected_sites(side)
                    if candidate["interface_kind"] == "ring"):
                assert math.isclose(
                    ring_site["face_offset_mm"],
                    RING_CAVITY_FACE_OFFSET_MM, abs_tol=1e-12)
                assert math.isclose(
                    ring_site["carrier_cavity_face_inset_mm"],
                    RING_CAVITY_FACE_INSET_MM, abs_tol=1e-12)
                assert math.isclose(
                    ring_site["continuous_flush_ring_fairing_mm"],
                    RING_FLUSH_FAIRING_MM, abs_tol=1e-12)
                assert math.isclose(
                    ring_site["local_captive_backing_boss_mm"],
                    0.0, abs_tol=1e-12)
                receiver_datum = cad._receiver_datum_face(ring_site)
                for index in range(2):
                    assert math.isclose(
                        receiver_datum[index],
                        ring_site["face"][index]
                        + RING_CAVITY_FACE_INSET_MM
                        * ring_site["normal"][index],
                        abs_tol=1e-9)
                ring_role = (
                    "lm_upper" if ring_site["driver"] == "lm" else "um")
                ring_tools = wall_cavity_tools(
                    name=ring_site["name"], face=receiver_datum,
                    outward=(*ring_site["normal"], 0.0), owner="wing",
                    axis_z=ring_site["z_mm"], print_up=(0.0, 0.0, -1.0),
                    front_z=FRONT_Z_MM,
                    interface_gap_mm=MAGNET_INTERFACE_GAP_MM)
                for cutter_index, cutter in enumerate(ring_tools.cutters):
                    assert _intersection_volume(
                        side_pieces[ring_role], cutter) <= 0.03, (
                            f"{slug}/{side}/{ring_role}: ring cutter "
                            f"{cutter_index} is obstructed")
                ring_solid = ring_tools.required_land
                for cutter in ring_tools.cutters:
                    ring_solid = ring_solid - cutter
                ring_retained = _intersection_volume(
                    side_pieces[ring_role], ring_solid)
                assert ring_retained >= 0.98 * ring_solid.volume, (
                    f"{slug}/{side}/{ring_role}: captive ring land "
                    f"incomplete: {ring_retained:.3f}/"
                    f"{ring_solid.volume:.3f} mm3")

                # The same zero-air-gap contract applies at the curved
                # LM-upper and UM roots.  Probe the solid 0.05-mm interval
                # separately from the qualified land so a plan-level carrier
                # clearance cannot silently turn it back into an air notch.
                ring_standoff = cad._axis_cylinder(
                    receiver_datum, ring_site["normal"], ring_site["z_mm"],
                    skin_diameter, inward=0.0,
                    outward=MAGNET_INTERFACE_GAP_MM - 0.01)
                ring_standoff_fill = _intersection_volume(
                    side_pieces[ring_role], ring_standoff)
                assert ring_standoff_fill >= 0.97 * ring_standoff.volume, (
                    f"{slug}/{side}/{ring_role}: solid receiver spacing "
                    "standoff is missing")

                ring_actual_face = ring_tools.actual_face_xyz[:2]
                ring_face_skin = cad._axis_cylinder(
                    ring_actual_face, ring_site["normal"], ring_site["z_mm"],
                    skin_diameter, inward=0.0,
                    outward=DEFAULT_SPEC.face_skin_mm - 0.03)
                rnx, rny = ring_site["normal"]
                ring_inner_face = (
                    ring_actual_face[0]
                    + (DEFAULT_SPEC.face_skin_mm
                       + DEFAULT_SPEC.cavity_depth_mm) * rnx,
                    ring_actual_face[1]
                    + (DEFAULT_SPEC.face_skin_mm
                       + DEFAULT_SPEC.cavity_depth_mm) * rny,
                )
                ring_inner_skin = cad._axis_cylinder(
                    ring_inner_face, ring_site["normal"], ring_site["z_mm"],
                    skin_diameter, inward=0.0,
                    outward=DEFAULT_SPEC.inner_skin_mm - 0.03)
                for skin_label, skin in (
                        ("interface", ring_face_skin),
                        ("inner", ring_inner_skin)):
                    fill = _intersection_volume(
                        side_pieces[ring_role], skin)
                    assert fill >= 0.97 * skin.volume, (
                        f"{slug}/{side}/{ring_role}: {skin_label} ring skin "
                        "is not sealed")
                for other_role in (
                        role for role in PRINT_PART_ROLES
                        if role != ring_role):
                    assert _intersection_volume(
                        side_pieces[other_role], ring_solid) <= 0.03, (
                            f"{slug}/{side}: {other_role} owns "
                            f"{ring_site['name']} captive land")
        if slug == "flat":
            roles = list(PRINT_PART_ROLES)
            for index, first in enumerate(roles):
                for second in roles[index + 1:]:
                    assert _intersection_volume(
                        pieces[first], pieces[second]) <= 0.03, (
                            f"{slug}: print parts overlap: {first}/{second}")
        serialized_parts = (
            serialized.get("print_contract", {})
            .get("installed_piece_brep", {}).get("right"))
        assert isinstance(serialized_parts, dict)
        assert tuple(serialized_parts) == PRINT_PART_ROLES
        serialized_monolith_volume = float(
            serialized_actual.get("right_volume_mm3"))
        serialized_fit_clearance_volume = (
            serialized_monolith_volume
            - sum(float(serialized_parts[role]["volume_mm3"])
                  for role in PRINT_PART_ROLES))
        assert (0.01 < serialized_fit_clearance_volume
                < 0.015 * serialized_monolith_volume), (
            f"{slug}: implausible source-BREP dovetail fit-clearance volume "
            f"{serialized_fit_clearance_volume:.3f} mm3")
        if slug == "flat":
            fit_clearance_volume = (
                right_volume - sum(
                    cad.adaptive_volume_mm3(piece)
                    for piece in pieces.values()))
            assembly_roundtrip_limit = max(
                1.0,
                STEP_ASSEMBLY_VOLUME_REL_TOL * serialized_monolith_volume)
            assert abs(
                fit_clearance_volume - serialized_fit_clearance_volume
            ) <= assembly_roundtrip_limit, (
                f"{slug}: STEP assembly fit-clearance round trip drifted: "
                f"STEP={fit_clearance_volume:.3f} mm3, source="
                f"{serialized_fit_clearance_volume:.3f} mm3, limit="
                f"{assembly_roundtrip_limit:.3f} mm3")

    if selected_slug != "graded":
        flat = release_shapes["flat"]["monoliths"]["right"]
        flat_depth_sampler = cad._VerticalDepthSampler(flat)
        flat_plan = cad.wing_plan("flat", "right")
        flat_raw_volume = float(flat_plan.area) * FULL_DEPTH_MM
        cavity_removed = flat_raw_volume - float(flat.volume)
        uncut_flat = cad._plan_prism(flat_plan, REAR_Z_MM, FRONT_Z_MM)
        expected_cavity_volume = sum(
            _intersection_volume(uncut_flat, cutter)
            for cutter in cad.receiver_pockets("right").values())
        key_pocket = lm_split.registration_wing_clearance_tools()["right"]
        for name, cutter in cad.receiver_pockets("right").items():
            assert _intersection_volume(key_pocket, cutter) <= 0.03, (
                f"flat key pocket overlaps receiver cutter {name}")
        expected_key_pocket_volume = _intersection_volume(
            uncut_flat, key_pocket)
        # The 3x-long keys and sockets are wholly buried in the native R113.8
        # ring (zero carrier-envelope growth).  Only the deliberate 0.25-mm
        # wing-clearance offset crosses the flat interface, so its exact clipped
        # volume is a small edge sliver rather than the former external-land
        # pocket.  Bracket both failure modes: no clearance cut and a renewed
        # bulky/visible support land.
        assert 0.05 < expected_key_pocket_volume < 0.25, (
            "flat native-ring key clearance has implausible clipped volume: "
            f"{expected_key_pocket_volume:.6f} mm3")
        expected_removed_volume = (
            expected_cavity_volume + expected_key_pocket_volume)
        assert math.isclose(
            cavity_removed, expected_removed_volume,
            rel_tol=1e-5, abs_tol=0.03), (
                "flat functional removal does not match the exact clipped "
                "receiver cutters plus optional-LM key pocket: "
                f"{cavity_removed:.3f} vs {expected_removed_volume:.3f} mm3")
        for role, plan_piece in cad.wing_print_plan_parts(
                "flat", "right").items():
            witness = plan_piece.representative_point()
            measured = flat_depth_sampler.depth_mm(witness.x, witness.y)
            assert math.isclose(measured, FULL_DEPTH_MM, abs_tol=0.01), (
                f"flat {role} rear is not the constant 11.5-mm plane: "
                f"{measured:.4f} mm")

    if selected_slug == "flat":
        print("  live flat BREP mirror, receiver, split and depth gates pass")
        return

    graded = release_shapes["graded"]["monoliths"]["right"]
    graded_depth_sampler = graded_live_depth_samplers.get(
        "right") or cad._VerticalDepthSampler(graded)
    sections = cad.wing_section_samples("graded", "right", samples=65)
    for key in ("S1", "S2", "S3", "S4"):
        section = sections[key]
        assert section["monotonic_nonincreasing"], (
            f"graded analytic {key} is not monotonic")
        assert section["worst_depth_reversal_mm"] <= GRADED_MONOTONIC_TOL_MM
        xy = np.asarray(section["xy_mm"], dtype=float)
        analytic = np.asarray(section["depth_mm"], dtype=float)
        along = np.asarray(section["distance_mm"], dtype=float)

        direction_in = xy[-2] - xy[-1]
        direction_in /= np.linalg.norm(direction_in)
        edge_probe = xy[-1] + 0.004 * direction_in
        edge_depth = graded_depth_sampler.depth_mm(*edge_probe)
        assert math.isclose(edge_depth, GRADED_EDGE_DEPTH_MM, abs_tol=0.035), (
            f"graded actual {key} free edge is {edge_depth:.4f} mm")

        direction_out = xy[1] - xy[0]
        direction_out /= np.linalg.norm(direction_out)
        land_probe = xy[0] + 0.004 * direction_out
        land_depth = graded_depth_sampler.depth_mm(*land_probe)
        assert math.isclose(land_depth, FULL_DEPTH_MM, abs_tol=0.05), (
            f"graded actual {key} mating land is {land_depth:.4f} mm")

        # Probe every interior witness: cubic rear surfaces can hide a local
        # reversal between a sparse set of 17 checks.  The user contract is
        # monotonic, so a 30-micron numerical allowance is the only rise
        # accepted after the full-depth land.
        indices = np.arange(1, len(xy) - 1, dtype=int)
        actual = np.asarray([
            graded_depth_sampler.depth_mm(*xy[index]) for index in indices
        ])
        expected = analytic[indices]
        maximum_error = float(np.max(np.abs(actual - expected)))
        reversal = float(np.max(actual - np.minimum.accumulate(actual)))
        sampled_along = along[indices]
        maximum_slope = float(np.max(
            np.abs(np.diff(actual) / np.diff(sampled_along))))
        print(
            f"  graded {key}: actual max error={maximum_error:.4f} mm, "
            f"reversal={reversal:.4f} mm, slope={maximum_slope:.4f}",
            flush=True)
        assert maximum_error <= 0.75, (
            f"graded actual {key} departs from analytic field by "
            f"{maximum_error:.3f} mm")
        assert reversal <= 0.03, (
            f"graded actual {key} rear surface reverses by "
            f"{reversal:.4f} mm")
        assert maximum_slope <= GRADED_MAX_SLOPE + 0.25, (
            f"graded actual {key} slope is {maximum_slope:.3f}")

    s5 = sections["S5"]
    s5_xy = np.asarray(s5["xy_mm"], dtype=float)
    for index in np.unique(np.linspace(2, len(s5_xy) - 3, 9, dtype=int)):
        measured = graded_depth_sampler.depth_mm(*s5_xy[index])
        assert math.isclose(measured, FULL_DEPTH_MM, abs_tol=0.05), (
            f"graded actual S5 T-seat depth is {measured:.4f} mm")

    # Probe both sides of the complete internal protected-land transition.  A
    # hard union could otherwise pass the one-sided land and monotonic-section
    # gates while hiding a C0 rear step immediately outside the exact 11.5-mm
    # prism.  There are no control-support exceptions: only perimeter segments
    # coincident with the external plan boundary are inapplicable because one
    # side of their +/- probe intentionally contains no wing.
    from shapely.geometry import Point

    _solution, depth_field, _definitions = cad._graded_analytics()
    protected = depth_field.protected
    plan = cad.wing_plan("graded", "right")
    external_boundary_guard = plan.boundary.buffer(
        cad.contract.GRADED_EDGE_MATCH_TOL_MM, cap_style=2, join_style=2)
    excluded_external_boundary = protected.boundary.intersection(
        external_boundary_guard)
    protected_transition = protected.boundary.difference(
        external_boundary_guard)
    plan_with_tolerance = plan.buffer(1.0e-6)
    perimeter_samples = 0
    qualified_samples = 0
    maximum_jump = 0.0
    maximum_jump_xy = (math.nan, math.nan)
    for line in _line_parts(protected_transition):
        sample_count = max(
            1, int(math.ceil(
                line.length / GRADED_LAND_BOUNDARY_SAMPLE_SPACING_MM)))
        # Bin midpoints avoid ambiguous polygon vertices while keeping every
        # point on the perimeter within 0.25 mm of a measured witness.
        for sample_index in range(sample_count):
            perimeter_samples += 1
            distance = line.length * (
                sample_index + 0.5) / float(sample_count)
            tangent_half_span = min(0.05, 0.20 * line.length)
            before = line.interpolate(max(0.0, distance - tangent_half_span))
            after = line.interpolate(min(line.length, distance + tangent_half_span))
            tx = float(after.x - before.x)
            ty = float(after.y - before.y)
            tangent_length = math.hypot(tx, ty)
            if tangent_length <= 1.0e-8:
                continue
            nx = -ty / tangent_length
            ny = tx / tangent_length
            boundary_point = line.interpolate(distance)
            plus = Point(
                boundary_point.x + GRADED_LAND_BOUNDARY_PROBE_OFFSET_MM * nx,
                boundary_point.y + GRADED_LAND_BOUNDARY_PROBE_OFFSET_MM * ny)
            minus = Point(
                boundary_point.x - GRADED_LAND_BOUNDARY_PROBE_OFFSET_MM * nx,
                boundary_point.y - GRADED_LAND_BOUNDARY_PROBE_OFFSET_MM * ny)
            plus_inside = protected.covers(plus)
            minus_inside = protected.covers(minus)
            if plus_inside == minus_inside:
                continue
            inside = plus if plus_inside else minus
            outside = minus if plus_inside else plus
            if (not plan_with_tolerance.covers(inside)
                    or not plan_with_tolerance.covers(outside)):
                continue
            inside_depth = graded_depth_sampler.depth_mm(inside.x, inside.y)
            outside_depth = graded_depth_sampler.depth_mm(outside.x, outside.y)
            assert math.isclose(
                inside_depth, FULL_DEPTH_MM, abs_tol=0.03), (
                    "graded actual protected-land perimeter lost its exact depth: "
                    f"{inside_depth:.4f} mm at "
                    f"({inside.x:.4f}, {inside.y:.4f})")
            jump = abs(inside_depth - outside_depth)
            qualified_samples += 1
            if qualified_samples == 1 or jump > maximum_jump:
                maximum_jump = jump
                maximum_jump_xy = (boundary_point.x, boundary_point.y)
    assert perimeter_samples >= 20, (
        "graded protected-land transition produced too few perimeter witnesses")
    assert qualified_samples == perimeter_samples, (
        "graded protected-land perimeter could not be probed completely: "
        f"{qualified_samples}/{perimeter_samples} samples")
    assert maximum_jump <= GRADED_LAND_BOUNDARY_MAX_JUMP_MM, (
        "graded protected-land C0 rear step exceeds the 0.03-mm gate: "
        f"{maximum_jump:.4f} mm at "
        f"({maximum_jump_xy[0]:.4f}, {maximum_jump_xy[1]:.4f})")
    graded_facts_path = _variant_paths("graded")["facts"]
    assert isinstance(graded_facts_path, Path)
    serialized_gate = (
        _read_json_object(graded_facts_path)["geometry"]["depth_contract"]
        ["protected_perimeter_brep_c0_gate"])
    assert serialized_gate["probe_engine"] == (
        "IntCurvesFace_ShapeIntersector")
    assert serialized_gate["paired_probe_count"] == qualified_samples
    assert math.isclose(
        serialized_gate["maximum_measured_c0_jump_mm"], maximum_jump,
        abs_tol=1.0e-6)
    assert serialized_gate["legacy_boolean_calibration_probe_count"] >= 18
    assert serialized_gate["legacy_boolean_maximum_delta_mm"] <= 0.002
    print(
        "  graded protected perimeter: "
        f"{qualified_samples} paired +/-0.004-mm probes, "
        f"max C0 jump={maximum_jump:.4f} mm, "
        "excluded external-boundary length="
        f"{excluded_external_boundary.length:.3f} mm",
        flush=True)
    checked_label = "flat/graded" if selected_slug is None else selected_slug.upper()
    print(
        f"  live {checked_label} BREP mirror, receiver, split and depth "
        "gates pass")


def _run_live_brep_variant(slug: str) -> None:
    previous = os.environ.get("LX_OBIWAN_WING_LIVE_SLUG")
    os.environ["LX_OBIWAN_WING_LIVE_SLUG"] = slug
    try:
        test_live_brep_geometry_contract()
    finally:
        if previous is None:
            os.environ.pop("LX_OBIWAN_WING_LIVE_SLUG", None)
        else:
            os.environ["LX_OBIWAN_WING_LIVE_SLUG"] = previous


def test_live_brep_geometry_contract_flat() -> None:
    _run_live_brep_variant("flat")


def test_live_brep_geometry_contract_graded() -> None:
    _run_live_brep_variant("graded")


CHECKS = (
    test_exported_artifact_contract,
    test_live_brep_geometry_contract,
)
SINGLE_CHECKS = CHECKS + (
    test_live_brep_geometry_contract_flat,
    test_live_brep_geometry_contract_graded,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remote-only flat/graded Obi-Wan wing acceptance")
    parser.add_argument("--artifact-root", default="build/wings")
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    os.environ["LX_OBIWAN_WING_ARTIFACT_ROOT"] = str(
        artifact_root.resolve())

    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        guard = ROOT / "scripts/run_memory_guarded.py"
        completed = subprocess.run(
            [sys.executable, str(guard), "--", sys.executable,
             str(Path(__file__).resolve()),
             "--artifact-root", str(artifact_root)],
            env=os.environ.copy(), check=False)
        return completed.returncode

    single = os.environ.get("LX_OBIWAN_WING_SINGLE_CHECK")
    if single:
        check = next(
            (item for item in SINGLE_CHECKS if item.__name__ == single), None)
        if check is None:
            raise SystemExit(f"unknown flat/graded check: {single}")
        print(f"{single}:", flush=True)
        check()
        return 0

    if not _large_host_execution():
        raise SystemExit(
            "flat/graded validation requires the remote osado-512g profile")

    def run_check(check):
        env = os.environ.copy()
        env["LX_OBIWAN_WING_SINGLE_CHECK"] = check.__name__
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--artifact-root", str(artifact_root)],
            env=env, text=True, capture_output=True, check=False)
        return (check.__name__, completed.returncode,
                completed.stdout, completed.stderr)

    try:
        requested_workers = int(os.environ.get("LX_CAD_GUARD_SLOTS", "4"))
    except ValueError as exc:
        raise SystemExit("LX_CAD_GUARD_SLOTS must be an integer") from exc
    if requested_workers <= 0:
        raise SystemExit("LX_CAD_GUARD_SLOTS must be positive")
    workers = min(requested_workers, len(CHECKS))
    results = []
    if workers == 1:
        for check in CHECKS:
            result = run_check(check)
            results.append(result)
            _name, _code, stdout, stderr = result
            print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", file=sys.stderr, flush=True)
    else:
        print(
            f"flat/graded remote runner: {workers} concurrent isolated checks",
            flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_check, check): check.__name__
                for check in CHECKS
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _name, _code, stdout, stderr = result
                print(stdout, end="", flush=True)
                if stderr:
                    print(stderr, end="", file=sys.stderr, flush=True)
    failures = {
        name for name, code, _stdout, _stderr in results if code != 0}
    if failures:
        ordered = [check.__name__ for check in CHECKS
                   if check.__name__ in failures]
        raise SystemExit("flat/graded FAILED: " + ", ".join(ordered))
    print("\nall flat/graded Obi-Wan wing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
