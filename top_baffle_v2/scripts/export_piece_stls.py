"""Export print-ready STLs: the four B2 baffle pieces plus the six
attachment pieces that turn the B2 set into variant A-comp or B1.

Run:  python export_piece_stls.py
Each part is rotated front-face-down, translated so its bounding box starts
at the origin, and written to stl/<name>.stl.  Only an in-bed Z rotation may
follow the common X180 process orientation; no released piece is printed on
an acoustic edge or rear face.
Exits nonzero if any bed-targeted piece stops fitting.  The retained
monolithic floor-state Obi-Wan LM is an explicit large-format reference; its
two optional keyed replacement prints remain strictly bed-checked.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)

# Direct CLI use gets the same process-tree guard policy as Make recipes.
# Imports from an already guarded test/export process do not
# re-exec, and ordinary module imports remain side-effect free.
if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))

from build123d import (
    Plane,
    Pos,
    Rot,
    Text,
    export_stl,
    extrude,
    import_brep,
    import_step,
)
from lx521_baffle.print_contract import (
    front_down_transform_record,
    sidecar_path_for_stl,
    write_print_sidecar,
)
from lx521_baffle.io import pretty_json_bytes, sha256_file
from lx521_baffle.stl_export import (
    BinaryStlLayoutError,
    canonicalize_near_zero_stl_coordinates,
    stl_topology_defects,
    validate_binary_stl_length,
)

BED_MM = 256.0
OBIWAN_OPTIONAL_LM_SPLIT_BED_MM = 220.0
DEFAULT_MESH_TOLERANCE_MM = 0.05
DEFAULT_MESH_ANGULAR_TOLERANCE = 0.20
# Obi-Wan's 0.45-mm captive skins, 0.8-mm flush route shells and narrow
# complementary closure faces need a finer deterministic tessellation than
# the broad legacy baffles. At 0.05 mm OCC can leave the narrowest valid BREP
# faces without triangulation, yielding an open STL even though the native
# carrier is one valid solid. flat uses this same 0.01/0.08 release class;
# apply it to every Obi-Wan core piece so split and canonical meshes share one
# contract.
OBIWAN_MESH_TOLERANCE_MM = 0.01
OBIWAN_MESH_ANGULAR_TOLERANCE = 0.08

# A rigid OCC Location can leave mathematically-zero coordinates as tiny
# nonzero values on only one of two adjacent face triangulations.  Binary
# STL stores vertices face-by-face, so that harmless transform noise becomes
# an exact-edge seam to downstream slicers/checkers.  Front-down plus in-bed
# Z placement can leave a shared datum at about 1e-7 mm on adjacent faces,
# differing by only a few femtometres.  This 0.2-nm threshold is
# still 250,000 times below the 0.05 mm mesh deflection; it canonicalizes
# rigid-transform roundoff only and is neither vertex welding nor CAD repair.
STL_TRANSFORM_ZERO_EPSILON_MM = 2.0e-7

# In-bed rotations preserve the common front-down process direction.
BED_ROT_Z = {
    "obiwan_core_1_of_2_lm_carrier": 28.0,
    "obiwan_optional_lm_keyed_1_of_2_bottom": 26.0,
    "obiwan_optional_lm_keyed_2_of_2_top": 45.0,
}


def _validate_binary_stl(path: Path) -> None:
    try:
        validate_binary_stl_length(path)
    except BinaryStlLayoutError as exc:
        if exc.truncated_header:
            raise RuntimeError(
                f"temporary STL is truncated: {path}") from None
        raise RuntimeError(
            "temporary STL transaction invalid: "
            f"triangles={exc.triangle_count} "
            f"bytes={exc.actual_bytes} expected={exc.expected_bytes}"
        ) from None


def _canonicalize_transform_zeros(
        path: Path, epsilon_mm: float = STL_TRANSFORM_ZERO_EPSILON_MM) -> int:
    """Write sub-nanometre vertex roundoff as one exact +0.0 value.

    OCC triangulates each BREP face independently.  After a rigid Location,
    the same shared vertex can consequently arrive in binary STL as 0.0 on
    one face and roughly 1e-15 mm on its neighbour.  The source BREP is still
    closed, but an exact edge-parity sweep correctly sees two different
    floating-point edges.  Canonicalizing only coordinates within this tiny
    neighbourhood preserves the mesh and makes the rigid transform exact.
    """
    if epsilon_mm <= 0.0:
        raise ValueError("STL transform-zero epsilon must be positive")
    try:
        return canonicalize_near_zero_stl_coordinates(path, epsilon_mm)
    except BinaryStlLayoutError as exc:
        if exc.truncated_header:
            raise RuntimeError(
                f"temporary STL is truncated: {path}") from None
        raise RuntimeError(
            "temporary STL transaction invalid: "
            f"triangles={exc.triangle_count} "
            f"bytes={exc.actual_bytes} expected={exc.expected_bytes}"
        ) from None


def _remove_collapsed_apex_facets(path: Path) -> int:
    """Drop only exact zero-area facets collapsed onto a real mesh edge.

    A mathematically sharp conical cavity apex is a valid BREP vertex, but
    OCC may serialize one triangle there with its second and third vertices
    identical.  That facet has exactly zero area and contributes no surface;
    retaining it merely counts its remaining edge twice.  Removing the
    collapsed record preserves every nonzero triangle and the exact CAD
    surface.  The subsequent strict topology gate must still prove the result
    closed, two-manifold, consistently wound, and free of every defect.

    Collinear triangles whose three vertices are distinct are deliberately
    untouched and remain a hard failure in ``_strict_mesh_facts``.
    """
    data = path.read_bytes()
    if len(data) < 84:
        raise RuntimeError(f"temporary STL is truncated: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * triangles
    if len(data) != expected:
        raise RuntimeError(
            f"temporary STL transaction invalid: triangles={triangles} "
            f"bytes={len(data)} expected={expected}")

    kept = []
    removed = 0
    for triangle in range(triangles):
        offset = 84 + 50 * triangle
        record = data[offset:offset + 50]
        coordinates = struct.unpack_from("<9f", record, 12)
        vertices = tuple(
            tuple(coordinates[3 * index:3 * index + 3])
            for index in range(3))
        if len(set(vertices)) < 3:
            removed += 1
        else:
            kept.append(record)
    if removed:
        repaired = bytearray(data[:80])
        repaired.extend(struct.pack("<I", len(kept)))
        repaired.extend(b"".join(kept))
        path.write_bytes(repaired)
        _validate_binary_stl(path)
    return removed


def _strict_mesh_facts(path: Path) -> dict[str, int | float]:
    """Run the release edge contract before atomically promoting an STL."""
    # Keep the authoritative implementation in one place.  The import is
    # deliberately lazy so ordinary use of this generator does not load the
    # release checker until a completed temporary mesh exists.
    facts, defects = stl_topology_defects(path)
    if defects:
        raise RuntimeError(
            f"temporary STL fails strict manifold contract: {path.name}: "
            f"{defects}; triangles={facts['triangles']} "
            f"components={facts['components']}")
    return facts


def _report_null_triangulation_faces(shape) -> None:
    """Emit bounded native-face facts when OCC skips a valid BREP face."""
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    missing = []
    for index, face in enumerate(shape.faces()):
        triangulation = BRep_Tool.Triangulation_s(
            face.wrapped, TopLoc_Location())
        if triangulation is not None:
            continue
        bbox = face.bounding_box()
        edge_lengths = [float(edge.length) for edge in face.edges()]
        missing.append({
            "face": index,
            "type": str(face.geom_type),
            "valid": bool(face.is_valid),
            "area_mm2": float(face.area),
            "bounds_mm": (
                (bbox.min.X, bbox.min.Y, bbox.min.Z),
                (bbox.max.X, bbox.max.Y, bbox.max.Z),
            ),
            "edges": len(edge_lengths),
            "edge_min_mm": min(edge_lengths, default=0.0),
            "edge_max_mm": max(edge_lengths, default=0.0),
        })
    print(
        "OCC null-triangulation face diagnostics: "
        + json.dumps(missing, sort_keys=True),
        file=sys.stderr,
    )


def _modifier_mesh_facts(path: Path) -> dict[str, int | float]:
    """Validate a possibly disconnected support-blocker STL."""
    from check_manifold import stl_diagnostics

    facts = stl_diagnostics(path)
    defect_keys = (
        "open", "over_shared", "winding", "degenerate", "duplicates",
        "nonfinite", "zero_volume", "negative_volume",
    )
    defects = {key: facts[key] for key in defect_keys if facts[key]}
    if defects or int(facts.get("components", 0)) < 1:
        raise RuntimeError(
            f"temporary support blocker fails mesh contract: {path.name}: "
            f"{defects}; components={facts.get('components')}")
    return facts


_sha256_file = sha256_file


def _write_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(pretty_json_bytes(payload, allow_nan=True))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _export_duct_support_blocker(
        *, out_dir: Path, name: str, main_stl_path: Path,
        source_bbox, bed_rotation_deg: float) -> str:
    """Export and hash-bind one carrier-owned duct support blocker."""
    from obiwan_support_blocker import (
        DUCT_SUPPORT_BLOCKER_CLEARANCE_MM,
        duct_support_blocker,
    )

    part_key = name.removeprefix("obiwan_")
    blocker, collision_contract = duct_support_blocker(part_key)
    blocker = Rot(X=180.0) * blocker
    if bed_rotation_deg:
        blocker = Rot(Z=bed_rotation_deg) * blocker
    blocker = Pos(
        -source_bbox.min.X, -source_bbox.min.Y, -source_bbox.min.Z,
    ) * blocker
    blocker_dir = out_dir.parent / "support_blockers"
    blocker_path = blocker_dir / f"{name}.support_blocker.stl"
    blocker_dir.mkdir(parents=True, exist_ok=True)
    temporary = blocker_path.with_name(
        f".{blocker_path.stem}.{os.getpid()}.tmp.stl")
    try:
        export_stl(
            blocker, str(temporary), tolerance=0.05,
            angular_tolerance=0.2)
        _validate_binary_stl(temporary)
        _canonicalize_transform_zeros(temporary)
        _remove_collapsed_apex_facets(temporary)
        blocker_mesh = _modifier_mesh_facts(temporary)
        temporary.replace(blocker_path)
    finally:
        temporary.unlink(missing_ok=True)

    main_sidecar = json.loads(
        sidecar_path_for_stl(main_stl_path).read_text(encoding="utf-8"))
    binding_path = blocker_path.with_suffix(".json")
    _write_atomic_json(binding_path, {
        "schema_version": 1,
        "kind": "bambu_support_blocker",
        "purpose": "forbid_support_inside_functional_ducts",
        "part": name,
        "main_stl": f"../stl/{main_stl_path.name}",
        "main_stl_sha256": _sha256_file(main_stl_path),
        "support_blocker": blocker_path.name,
        "support_blocker_sha256": _sha256_file(blocker_path),
        "source_to_stl_matrix": main_sidecar["source_to_stl_matrix"],
        "modifier_clearance_mm": DUCT_SUPPORT_BLOCKER_CLEARANCE_MM,
        "duct_collision_contract": collision_contract,
        "mesh": blocker_mesh,
    })
    return (
        f"  support blocker {blocker_path.name}"
        f" ({blocker_mesh['components']} closed components)")


def _write_print_transform_sidecar(
        path: Path, *, name: str, variant: str, z_rotation_deg: float,
        oriented_bbox, mesh_facts: dict[str, int | float],
        mesh_tolerance_mm: float,
        mesh_angular_tolerance: float) -> None:
    """Bind one STL to its exact front-down rigid source transform."""
    transform = front_down_transform_record(
        [
            float(oriented_bbox.min.X),
            float(oriented_bbox.min.Y),
            float(oriented_bbox.min.Z),
        ],
        z_rotation_deg=z_rotation_deg,
    )
    write_print_sidecar(
        path,
        part=name,
        transform=transform,
        extra={
            "variant_export": variant,
            "mesh": mesh_facts,
            "mesh_tolerance_mm": mesh_tolerance_mm,
            "mesh_angular_tolerance": mesh_angular_tolerance,
        },
    )


def _unlink_print_pair(path: Path) -> None:
    """Remove one generated STL and its adjacent print-contract sidecar."""
    if path.name.endswith(".print.json"):
        stl = path.with_name(
            path.name.removesuffix(".print.json") + ".stl")
    elif path.suffix.lower() == ".stl":
        stl = path
    else:
        raise ValueError(f"not a generated STL or print sidecar: {path}")
    stl.unlink(missing_ok=True)
    sidecar_path_for_stl(stl).unlink(missing_ok=True)


def _routing_rev():
    return {"proud": "R6P", "obiwan": "R6F"}.get(
        os.environ.get("LX_ROUTING_PROFILE", "proud"), "R6P")

# safe embossing anchors (flat local rear, clear of pockets/thin zones)
# (x, y, rot_deg, font, short_label) -- verified flat-rear spots. The
# TOP shoulders are crescent-tapered to a knife over most of their
# rear: only the full-thickness corner past the horn tips fits text,
# with a shortened family+position code at font 2.6.
EMBOSS_XY = {
    # The old (70, 40) rear-plane site became empty when the hard floor
    # corner was replaced by the rearward Option-B bend.  This centered band
    # is on the retained vertical tangent: the tallest V1L label spans
    # y=74.63..77.55, leaving ~0.45 mm above the R41 wall endpoint (74.15).
    # No-floor holds the full-depth plane up to V1L's ramp start (78.0), so
    # its local rear is flat.  The floor state's path-length ramp is still
    # climbing here and falls 0.140 mm across the label, so the recess runs
    # 0.438 mm at the bottom of the glyphs to 0.298 mm at the top -- an
    # accepted trade for reaching full depth at the horizontal tangent.
    # test_emboss_driver_keepouts binds both states.
    # Anchors are keyed by the piece-name tail, which is unique across both
    # proud products: stock and slim number their shoulders differently, but
    # a shoulder always ends in its corner and a base piece in its role.
    "_1_of_4_bottom": (0.0, 76.0, 0.0, 4.0, False),
    # Use the compact V1L-2 family/position code here. The optional routing
    # suffix put one inward glyph stroke into the guarded R95 opening and
    # appeared as the reported duct-out bite in the exported STL.
    "_2_of_4_mid_left": (-98.0, 160.0, 0.0, 4.0, True),
    "_3_of_4_mid_right": (95.0, 160.0, 0.0, 4.0, False),
    "_4_of_4_vase_b2": (0.0, 320.6, 0.0, 4.0, False),
    "_top_left": (-44.0, 412.0, 0.0, 3.2, False),
    "_top_right": (44.0, 412.0, 0.0, 3.2, False),
    "_bottom_left": (-55.0, 345.0, 90.0, 3.2, False),
    "_bottom_right": (55.0, 345.0, 90.0, 3.2, False),
    "_wing_1_of_2_left": (-71.0, 388.0, -73.5, 4.0, False),
    "_wing_2_of_2_right": (71.0, 388.0, 73.5, 4.0, False),
    "_core_1_of_2_lm_carrier": (0.0, 300.0, 0.0, 3.0, False),
    "_core_2_of_2_um_carrier": (0.0, 408.0, 0.0, 2.2, False),
}

# The embossed provenance code stays the established design vocabulary even
# though the filenames are now product-first: a printed B2-1 or V1A-SBL part
# must remain identifiable against every earlier print and photo.
EMBOSS_CODES = {
    "stock_1_of_4_bottom": "B2-1",
    "stock_2_of_4_mid_left": "B2-2",
    "stock_3_of_4_mid_right": "B2-3",
    "stock_4_of_4_vase_b2": "B2-4",
    "stock_shoulder_1_of_4_top_left": "A-1",
    "stock_shoulder_2_of_4_top_right": "A-2",
    "stock_shoulder_3_of_4_bottom_left": "A-3",
    "stock_shoulder_4_of_4_bottom_right": "A-4",
    "stock_wing_1_of_2_left": "B1-1",
    "stock_wing_2_of_2_right": "B1-2",
    "slim_1_of_4_bottom": "V1L-1",
    "slim_2_of_4_mid_left": "V1L-2",
    "slim_3_of_4_mid_right": "V1L-3",
    "slim_4_of_4_vase_b2": "V1L-4",
    "slim_shoulder_1_of_4_bottom_left": "V1A-SBL",
    "slim_shoulder_2_of_4_top_left": "V1A-STL",
    "slim_shoulder_3_of_4_bottom_right": "V1A-SBR",
    "slim_shoulder_4_of_4_top_right": "V1A-STR",
    "slim_wing_1_of_2_left": "V1B1-WL",
    "slim_wing_2_of_2_right": "V1B1-WR",
}


def _label(name):
    """Short provenance code, e.g. B2-1, V1L-3, V1A-STL, V1B1-WL."""
    if "obiwan_core_1_of_2_lm" in name:
        return f"Obi-Wan-LM {_routing_rev()}"
    if "obiwan_core_2_of_2_um" in name:
        return f"Obi-Wan-UM {_routing_rev()}"
    try:
        code = EMBOSS_CODES[name]
    except KeyError:
        raise SystemExit(f"no emboss provenance code for {name}") from None
    return f"{code} {_routing_rev()}"


def _emboss(solid, name):
    """Recessed 0.4 mm ID text on the piece's hidden rear face,
    centered on a safe flat spot, mirrored to read correctly when
    looking AT the rear; rotated 90 deg on narrow pieces."""
    from build123d import Rot, mirror

    # R6F is deliberately material-minimal. Identity stays in filenames
    # and STEP labels; no rear engraving is allowed to partition a thin
    # tunnel cover or the fused front-flush bridge web.
    if "obiwan_" in name or "grommet" in name:
        return solid

    for suffix, (ax, ay, rot, font, short) in EMBOSS_XY.items():
        if name.endswith(suffix):
            break
    else:
        raise SystemExit(f"no emboss anchor for {name}")
    # rear z AT the anchor, not the piece's global min (Obi-Wan pad
    # buttons and the stand foot both undercut the plate rear)
    from build123d import Cylinder

    pin = solid & (Pos(ax, ay, 0.0) * Cylinder(0.6, 400.0))
    if pin is None or not pin.volume:
        raise SystemExit(f"emboss anchor for {name} is off the solid")
    zr = pin.bounding_box().min.Z
    if zr < -1.0:
        zr = 0.0  # foot piece: the plate rear, not the foot tip
    label = _label(name)
    if short:
        label = label.split(" ")[0]
    txt = mirror(Text(label, font_size=font), Plane.YZ)
    cutter = extrude(Pos(ax, ay, zr - 0.4) * Rot(Z=rot) * txt, amount=0.8)
    inter = solid & cutter
    carved = inter.volume if inter is not None else 0.0
    if carved < 0.42 * cutter.volume:
        raise SystemExit(
            f"emboss on {name} not fully in material "
            f"({carved:.0f}/{cutter.volume:.0f} mm3) -- move the anchor")
    return solid - cutter
OUT_DIR = PROJECT_ROOT / "build/floor_stand/stl"


# slicer-friendly names: <group>_<print order>_<part>
STL_NAMES = {
    "piece_bottom": "stock_1_of_4_bottom",
    "piece_mid_left": "stock_2_of_4_mid_left",
    "piece_mid_right": "stock_3_of_4_mid_right",
    "piece_top_b2": "stock_4_of_4_vase_b2",
    "attach_a_shoulder_top_left": "stock_shoulder_1_of_4_top_left",
    "attach_a_shoulder_top_right": "stock_shoulder_2_of_4_top_right",
    "attach_a_shoulder_bottom_left": "stock_shoulder_3_of_4_bottom_left",
    "attach_a_shoulder_bottom_right": "stock_shoulder_4_of_4_bottom_right",
    "attach_b1_wing_left": "stock_wing_1_of_2_left",
    "attach_b1_wing_right": "stock_wing_2_of_2_right",
}

# Slim receivers are numbered in the shelf's bottom-left/top-left/bottom-right
# /top-right order; the stock set keeps its own established 1..4 ordering.
V1_ATTACHMENT_STL_NAMES = {
    "attach_v1a_shoulder_bottom_left": "slim_shoulder_1_of_4_bottom_left",
    "attach_v1a_shoulder_top_left": "slim_shoulder_2_of_4_top_left",
    "attach_v1a_shoulder_bottom_right": "slim_shoulder_3_of_4_bottom_right",
    "attach_v1a_shoulder_top_right": "slim_shoulder_4_of_4_top_right",
    "attach_v1b1_wing_left": "slim_wing_1_of_2_left",
    "attach_v1b1_wing_right": "slim_wing_2_of_2_right",
}

PROUD_STEP_PIECE_LABELS = (
    "piece_bottom",
    "piece_mid_left",
    "piece_mid_right",
    "piece_top_b2",
)


def _load_proud_step_parts(
        path: Path, *, stand_mode: str) -> dict[str, object]:
    """Load the four physical print children from an authoritative STEP.

    STEP is already a required release artifact for every proud-family split.
    Meshing those exact children avoids rebuilding the same Boolean tree in a
    second process and gives OCC one deterministic serialization boundary
    before face triangulation.  Labels and the stand-state envelope are
    checked fail-closed so a stale or wrong-state STEP cannot feed an STL.
    """
    if not path.is_file():
        raise RuntimeError(f"STEP mesh source does not exist: {path}")
    assembly = import_step(str(path))
    children = tuple(assembly.children)
    labels = tuple(child.label for child in children)
    if len(labels) != len(set(labels)):
        raise RuntimeError(
            f"STEP mesh source has duplicate child labels: {path}: {labels}")
    parts = dict(zip(labels, children, strict=True))
    expected = set(PROUD_STEP_PIECE_LABELS)
    if set(parts) != expected:
        raise RuntimeError(
            f"STEP mesh source must contain exactly {sorted(expected)}; "
            f"got {sorted(parts)} from {path}")
    for label, part in parts.items():
        solids = tuple(part.solids())
        if (not part.is_valid or len(solids) != 1
                or solids[0].volume <= 0.01):
            raise RuntimeError(
                f"STEP mesh source child {label!r} is not one valid solid: "
                f"valid={part.is_valid} volumes="
                f"{[solid.volume for solid in solids]}")

    # Only the bottom child differs categorically between the stand states.
    # The released no-floor baffle is 18.3 mm deep; the floor child extends
    # 150 mm behind the same datum.  This broad classification deliberately
    # avoids binding the gate to tessellation or sub-millimetre tolerances.
    bottom_depth = float(parts["piece_bottom"].bounding_box().size.Z)
    source_has_floor = bottom_depth > 50.0
    expected_has_floor = stand_mode == "1"
    if source_has_floor != expected_has_floor:
        raise RuntimeError(
            f"STEP mesh source stand-state mismatch for {path}: "
            f"LX_STAND_FOOT={stand_mode}, bottom depth={bottom_depth:.3f} mm")
    print(
        f"[step-mesh-source] loaded {path} "
        f"({len(parts)} labeled solids; bottom depth {bottom_depth:.3f} mm)",
        flush=True,
    )
    return parts


def _large_host_execution() -> bool:
    """Allow whole-variant meshing only in the guarded osado workflow."""
    return (
        os.environ.get("LX_CAD_EXECUTION") != "local"
        and os.environ.get("LX_CAD_MEMORY_PROFILE") == "osado-512g"
        and os.environ.get("LX_CAD_ALLOW_PARALLEL") == "1"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUT_DIR,
                    help="directory for the STLs (default: stl/)")
    ap.add_argument("--variant", choices=("b2", "v1", "v1l", "obiwan"),
                    default="b2",
                    help="b2: base pieces + attachments; v1: the slim "
                         "shoulder/wing receivers (the slim vase and base "
                         "pieces ship from v1l)")
    ap.add_argument(
        "--source-step", type=Path,
        help="mesh labeled proud-family pieces from this authoritative "
             "split STEP instead of rebuilding their CAD")
    ap.add_argument(
        "--obiwan-part",
        choices=("lm", "lm_split", "um", "tweeter"),
        help="export one staged R6F group; omit on osado to mesh the whole "
             "state in one guarded process")
    ap.add_argument(
        "--obiwan-key",
        choices=(
            "core_lm_carrier",
            "core_um_carrier",
            "optional_lm_keyed_1_of_2_bottom",
            "optional_lm_keyed_2_of_2_top",
            "addon_tweeter_crescent",
        ),
        help="export exactly one staged R6F print part; intended for focused "
             "remote iteration without remeshing an unaffected group")
    ap.add_argument(
        "--obiwan-stage-manifest", type=Path,
        help="hash-verified native-BREP stage manifest produced by "
             "export_obiwan_staged.py; required for every Obi-Wan export")
    ap.add_argument(
        "--v1l-piece",
        choices=("piece_bottom", "piece_mid_left", "piece_mid_right",
                 "piece_top_b2", "grommet"),
        help="build one V1L split piece or its terminal grommet pair; "
             "serial use releases OCC geometry between groups and "
             "reduces the local process-tree RSS peak")
    args = ap.parse_args()
    if args.obiwan_part and args.variant != "obiwan":
        ap.error("--obiwan-part requires --variant obiwan")
    if args.obiwan_key and args.variant != "obiwan":
        ap.error("--obiwan-key requires --variant obiwan")
    if args.obiwan_key and args.obiwan_part:
        ap.error("--obiwan-key and --obiwan-part are mutually exclusive")
    if args.obiwan_stage_manifest and args.variant != "obiwan":
        ap.error("--obiwan-stage-manifest requires --variant obiwan")
    if args.v1l_piece and args.variant != "v1l":
        ap.error("--v1l-piece requires --variant v1l")
    if args.source_step and args.variant == "obiwan":
        ap.error("--source-step is for proud-family split STEP files only")
    stand_mode = os.environ.get("LX_STAND_FOOT", "1")
    if stand_mode not in {"0", "1"}:
        ap.error("LX_STAND_FOOT must be 0 or 1")
    if (args.variant == "obiwan" and args.obiwan_part is None
            and args.obiwan_key is None
            and not _large_host_execution()):
        ap.error(
            "local --variant obiwan requires one --obiwan-part or "
            "--obiwan-key so every OCC group runs in a fresh guarded process")
    if args.variant == "obiwan" and args.obiwan_stage_manifest is None:
        ap.error(
            "--variant obiwan requires --obiwan-stage-manifest; direct "
            "monolithic carrier generation is intentionally disabled")
    profile = "obiwan" if args.variant == "obiwan" else "proud"
    os.environ["LX_ROUTING_PROFILE"] = profile
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    step_parts = (
        _load_proud_step_parts(args.source_step, stand_mode=stand_mode)
        if args.source_step is not None else None
    )
    if args.variant == "obiwan":
        from export_obiwan_staged import (
            PRINT_PART_SPECS,
            load_stage_manifest,
            staged_part_paths,
        )

        payload = load_stage_manifest(args.obiwan_stage_manifest)
        staged = staged_part_paths(args.obiwan_stage_manifest, payload)
        keys = tuple(
            key for key, spec in PRINT_PART_SPECS.items()
            if key in staged and (
                args.obiwan_key is None or key == args.obiwan_key) and (
                args.obiwan_part is None
                or spec["group"] == args.obiwan_part))
        if not keys:
            ap.error(
                f"the staged {payload['state']} manifest has no "
                f"{args.obiwan_part!r} print group")
        parts = {
            PRINT_PART_SPECS[key]["stl_name"]: import_brep(str(staged[key]))
            for key in keys
        }
        # Prune artifacts owned by superseded Obi-Wan generators.  In
        # particular this removes the deleted external R14 raceway and
        # right alignment link from incremental output directories.
        for legacy in (
                *out_dir.glob("obiwan_[1-4]of4_*.stl"),
                *out_dir.glob("obiwan_[1-4]of4_*.print.json")):
            _unlink_print_pair(legacy)
        expected = {
            "obiwan_core_1_of_2_lm_carrier.stl",
            "obiwan_core_2_of_2_um_carrier.stl",
            "obiwan_optional_lm_keyed_1_of_2_bottom.stl",
            "obiwan_optional_lm_keyed_2_of_2_top.stl",
            "obiwan_addon_tweeter_crescent.stl",
        }
        for legacy in (
                *out_dir.glob("obiwan_addon_*.stl"),
                *out_dir.glob("obiwan_addon_*.print.json")):
            stl_name = (
                legacy.name.removesuffix(".print.json") + ".stl"
                if legacy.name.endswith(".print.json") else legacy.name)
            if stl_name not in expected:
                _unlink_print_pair(legacy)
    elif args.variant == "v1l":
        parts = {}
        if args.v1l_piece != "grommet":
            if step_parts is None:
                from lx521_baffle.proud.v1l_split import pieces_v1l
                physical_parts = pieces_v1l(only=args.v1l_piece)
            else:
                selected = (
                    PROUD_STEP_PIECE_LABELS
                    if args.v1l_piece is None else (args.v1l_piece,)
                )
                physical_parts = {
                    key: step_parts[key] for key in selected
                }
            parts.update({
                STL_NAMES[k].replace("stock_", "slim_", 1): v
                for k, v in physical_parts.items()
            })
        if args.v1l_piece in (None, "grommet"):
            from lx521_baffle.um_fit import split_grommet_parts
            parts.update({
                f"slim_{name}": solid
                for name, solid in split_grommet_parts("v1l").items()
            })
    elif args.variant == "v1":
        # The V1 split STEP remains the geometry authority the slim receivers
        # are cut against, but only its attachments are released as prints:
        # the slim vase ships from the V1L set.
        from lx521_baffle.proud.v1_attachments import v1_attachments
        parts = {V1_ATTACHMENT_STL_NAMES[k]: v
                 for k, v in v1_attachments().items()}
    else:
        from lx521_baffle.proud.attachments import attachments
        if step_parts is None:
            from lx521_baffle.proud.b2_split import pieces
            parts = dict(pieces())
        else:
            parts = dict(step_parts)
        parts.update(attachments())
        parts = {STL_NAMES[k]: v for k, v in parts.items()}
        from lx521_baffle.um_fit import split_grommet_parts
        parts.update({f"stock_{name}": solid
                      for name, solid in split_grommet_parts("proud").items()})
    misfits = []
    for name, solid in parts.items():
        source_solids = list(solid.solids())
        if (not solid.is_valid or len(source_solids) != 1
                or source_solids[0].volume <= 0.01):
            raise RuntimeError(
                f"{name}: expected exactly one valid source solid; "
                f"valid={solid.is_valid} volumes="
                f"{[item.volume for item in source_solids]}")
        solid = _emboss(solid, name)
        embossed_solids = list(solid.solids())
        if (not solid.is_valid or len(embossed_solids) != 1
                or embossed_solids[0].volume <= 0.01):
            raise RuntimeError(
                f"{name}: emboss/finalization damaged source topology")
        # One print-side contract for the complete release: put the acoustic
        # front on the build plate.  Besides making the visible texture
        # consistent, this is the process direction used by every captive
        # loading chimney.  The former Obi-Wan X26/X90 packing rotations made
        # those chimneys run sideways; the optional floor LM lower still fits
        # the 220-mm envelope front-down (about 218.7 x 175.9 mm).
        solid = Rot(X=180.0) * solid
        canonical_lm_large_format = (
            "obiwan_core_1_of_2_lm_carrier" in name)
        bed_rotation = next((angle for key, angle in BED_ROT_Z.items()
                             if key in name), 0.0)
        orientation = " @ X180deg front-down"
        if bed_rotation:
            solid = Rot(Z=bed_rotation) * solid
            orientation += f" @ Z{bed_rotation:g}deg"
        bb = solid.bounding_box()
        size = bb.size
        bed_limit = (
            OBIWAN_OPTIONAL_LM_SPLIT_BED_MM
            if "obiwan_optional_lm_keyed_" in name else BED_MM)
        fits = (
            canonical_lm_large_format
            or (size.X <= bed_limit and size.Y <= bed_limit
                and size.Z <= bed_limit)
        )
        if not fits:
            misfits.append(name)
        bed_status = (
            "LARGE-FORMAT CANONICAL (split option is bed-checked)"
            if canonical_lm_large_format
            else ("OK" if fits else "DOES NOT FIT")
        )
        moved = Pos(-bb.min.X, -bb.min.Y, -bb.min.Z) * solid
        moved_solids = list(moved.solids())
        if (not moved.is_valid or len(moved_solids) != 1
                or moved_solids[0].volume <= 0.01):
            raise RuntimeError(f"{name}: print transform damaged topology")
        path = out_dir / f"{name}.stl"
        # Keep the last known-good mesh intact if OCC/export is interrupted.
        # The temporary name still ends in .stl so build123d selects the
        # correct writer; os.replace via Path.replace is atomic on this
        # same-directory transaction.
        temporary = path.with_name(
            f".{path.stem}.{os.getpid()}.tmp.stl")
        canonicalized_zeros = 0
        collapsed_apex_facets = 0
        mesh_tolerance = (
            OBIWAN_MESH_TOLERANCE_MM
            if args.variant == "obiwan" else DEFAULT_MESH_TOLERANCE_MM)
        mesh_angular_tolerance = (
            OBIWAN_MESH_ANGULAR_TOLERANCE
            if args.variant == "obiwan"
            else DEFAULT_MESH_ANGULAR_TOLERANCE)
        try:
            export_stl(
                moved, str(temporary), tolerance=mesh_tolerance,
                angular_tolerance=mesh_angular_tolerance)
            _validate_binary_stl(temporary)
            # X-axis print transforms can express exact zero coordinates as
            # face-local floating-point noise. Keep all other meshes
            # byte-for-byte faithful to the ordinary OCC export.
            canonicalized_zeros = _canonicalize_transform_zeros(temporary)
            collapsed_apex_facets = _remove_collapsed_apex_facets(temporary)
            try:
                mesh_facts = _strict_mesh_facts(temporary)
            except RuntimeError:
                _report_null_triangulation_faces(moved)
                raise
            mesh_facts["collapsed_apex_facets_removed"] = (
                collapsed_apex_facets)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        _write_print_transform_sidecar(
            path, name=name, variant=args.variant,
            z_rotation_deg=bed_rotation, oriented_bbox=bb,
            mesh_facts=mesh_facts,
            mesh_tolerance_mm=mesh_tolerance,
            mesh_angular_tolerance=mesh_angular_tolerance)
        support_blocker_note = ""
        if args.variant == "obiwan":
            from obiwan_support_blocker import (
                DUCT_SUPPORT_BLOCKER_STL_NAMES,
            )
        if (args.variant == "obiwan"
                and name in DUCT_SUPPORT_BLOCKER_STL_NAMES):
            support_blocker_note = _export_duct_support_blocker(
                out_dir=out_dir,
                name=name,
                main_stl_path=path,
                source_bbox=bb,
                bed_rotation_deg=bed_rotation,
            )
        print(
            f"{name:22s} {size.X:7.2f} x {size.Y:7.2f} x {size.Z:5.2f} mm  "
            f"volume {solid.volume / 1000.0:7.1f} cm3  "
            f"bed fit <= {bed_limit:g}: "
            f"{bed_status}"
            f"{orientation}"
            f"  mesh {mesh_facts['triangles']} tris/strict"
            f"  transform-zero fixes {canonicalized_zeros}"
            f"  collapsed-apex facets {collapsed_apex_facets}"
            f"  -> {path.name}{support_blocker_note}"
        )
    if misfits:
        sys.exit("ERROR: piece(s) exceed their configured bed envelope: "
                 + ", ".join(misfits))


if __name__ == "__main__":
    main()
