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
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

# Direct CLI use gets the same process-tree guard policy as Make recipes.
# Imports from an already guarded test/export process do not
# re-exec, and ordinary module imports remain side-effect free.
if __name__ == "__main__":
    import subprocess
    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        guard = Path(__file__).with_name("run_memory_guarded.py")
        raise SystemExit(subprocess.run(
            [sys.executable, str(guard), "--", sys.executable,
             str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False).returncode)

from build123d import Plane, Pos, Rot, Text, export_stl, extrude, import_brep
from front_down_contract import sidecar_path_for_stl, write_print_sidecar

BED_MM = 256.0
OBIWAN_OPTIONAL_LM_SPLIT_BED_MM = 220.0
DEFAULT_MESH_TOLERANCE_MM = 0.05
DEFAULT_MESH_ANGULAR_TOLERANCE = 0.20
# Obi-Wan's 0.45-mm captive skins, 0.8-mm flush route shells and narrow
# complementary closure faces need a finer deterministic tessellation than
# the broad legacy baffles. At 0.05 mm OCC can leave the narrowest valid BREP
# faces without triangulation, yielding an open STL even though the native
# carrier is one valid solid. Ac uses this same 0.01/0.08 release class;
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
    "obiwan_core_1of2_lm_carrier": 28.0,
    "obiwan_optional_lm_keyed_1of2_bottom": 26.0,
    "obiwan_optional_lm_keyed_2of2_top": 45.0,
}


def _validate_binary_stl(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(84)
    if len(header) != 84:
        raise RuntimeError(f"temporary STL is truncated: {path}")
    triangles = struct.unpack_from("<I", header, 80)[0]
    expected = 84 + 50 * triangles
    if triangles < 1 or path.stat().st_size != expected:
        raise RuntimeError(
            f"temporary STL transaction invalid: triangles={triangles} "
            f"bytes={path.stat().st_size} expected={expected}")


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
    data = bytearray(path.read_bytes())
    if len(data) < 84:
        raise RuntimeError(f"temporary STL is truncated: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * triangles
    if len(data) != expected:
        raise RuntimeError(
            f"temporary STL transaction invalid: triangles={triangles} "
            f"bytes={len(data)} expected={expected}")
    changed = 0
    for triangle in range(triangles):
        vertex_base = 84 + 50 * triangle + 12
        for coordinate in range(9):
            offset = vertex_base + 4 * coordinate
            value = struct.unpack_from("<f", data, offset)[0]
            if value != 0.0 and abs(value) <= epsilon_mm:
                struct.pack_into("<f", data, offset, 0.0)
                changed += 1
    if changed:
        path.write_bytes(data)
    return changed


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
    from check_manifold import stl_diagnostics

    facts = stl_diagnostics(path)
    defect_keys = (
        "open", "over_shared", "winding", "degenerate", "duplicates",
        "nonfinite", "zero_volume", "negative_volume", "component_error",
    )
    defects = {key: facts[key] for key in defect_keys if facts[key]}
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _export_no_floor_lm_support_blocker(
        *, out_dir: Path, name: str, main_stl_path: Path,
        source_bbox, bed_rotation_deg: float) -> str:
    """Export and hash-bind the no-floor duct support-blocker modifier."""
    from top_baffle_nd25fw4_obiwan_lm_split import LM_SPLIT_SEAM_Y
    from top_baffle_nd25fw4_obiwan_route import (
        no_floor_lm_bottom_support_blocker,
    )

    blocker = no_floor_lm_bottom_support_blocker(LM_SPLIT_SEAM_Y)
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
        "purpose": "forbid_support_inside_no_floor_lm_um_t_ducts",
        "part": name,
        "main_stl": f"../stl/{main_stl_path.name}",
        "main_stl_sha256": _sha256_file(main_stl_path),
        "support_blocker": blocker_path.name,
        "support_blocker_sha256": _sha256_file(blocker_path),
        "source_to_stl_matrix": main_sidecar["source_to_stl_matrix"],
        "modifier_clearance_mm": 0.25,
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
    angle = math.radians(float(z_rotation_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    # Rz(angle) * Rx(180).  Translation then drops the oriented BREP's
    # bounding-box minimum to STL origin.
    matrix = (
        (cosine, sine, 0.0, -float(oriented_bbox.min.X)),
        (sine, -cosine, 0.0, -float(oriented_bbox.min.Y)),
        (0.0, 0.0, -1.0, -float(oriented_bbox.min.Z)),
        (0.0, 0.0, 0.0, 1.0),
    )
    transform = {
        "print_orientation": "front_face_down",
        "source_to_stl_matrix": [list(row) for row in matrix],
        "rotation_deg": {"x": 180.0, "z": float(z_rotation_deg)},
        "pre_translation_bbox_min_mm": [
            float(oriented_bbox.min.X), float(oriented_bbox.min.Y),
            float(oriented_bbox.min.Z),
        ],
        "stl_origin_translation_mm": [
            -float(oriented_bbox.min.X), -float(oriented_bbox.min.Y),
            -float(oriented_bbox.min.Z),
        ],
    }
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
    "1of4_bottom": (70.0, 40.0, 0.0, 4.0, False),
    # Use the compact V1L-2 family/position code here. The optional routing
    # suffix put one inward glyph stroke into the guarded R95 opening and
    # appeared as the reported duct-out bite in the exported STL.
    "2of4_mid_left": (-98.0, 160.0, 0.0, 4.0, True),
    "3of4_mid_right": (95.0, 160.0, 0.0, 4.0, False),
    "4of4_vase": (0.0, 320.6, 0.0, 4.0, False),
    "shoulder_top_left": (-44.0, 412.0, 0.0, 3.2, False),
    "shoulder_top_right": (44.0, 412.0, 0.0, 3.2, False),
    "shoulder_bottom_left": (-55.0, 345.0, 90.0, 3.2, False),
    "shoulder_bottom_right": (55.0, 345.0, 90.0, 3.2, False),
    "wing_left": (-71.0, 388.0, -73.5, 4.0, False),
    "wing_right": (71.0, 388.0, 73.5, 4.0, False),
    "core_1of2_lm_carrier": (0.0, 300.0, 0.0, 3.0, False),
    "core_2of2_um_carrier": (0.0, 408.0, 0.0, 2.2, False),
}


def _label(name):
    """Short provenance code, e.g. B2-1, V1L-3, V1A-TL, B1-WL."""
    if "obiwan_core_1of2_lm" in name:
        return f"Obi-Wan-LM {_routing_rev()}"
    if "obiwan_core_2of2_um" in name:
        return f"Obi-Wan-UM {_routing_rev()}"
    n = name.replace("lx521_top_", "")
    fam = {"base": "B2", "c7base": "C7", "v1l": "V1L", "obiwan": "Obi-Wan",
           "v1": "V1", "addonA": "A", "addonB1": "B1", "v1addonA": "V1A",
           "v1addonB1": "V1B1"}
    head = n.split("_")[0]
    code = fam.get(head, head.upper())
    if "of4" in n or "of2" in n:
        code += "-" + n.split("_")[1][0]
    else:
        parts = n.split("_")
        code += "-" + "".join(w[0] for w in parts[1:]).upper()
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
        if suffix in name:
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
OUT_DIR = Path(__file__).parent / "stl"


# slicer-friendly names: <group>_<print order>_<part>
STL_NAMES = {
    "piece_bottom": "lx521_top_base_1of4_bottom",
    "piece_mid_left": "lx521_top_base_2of4_mid_left",
    "piece_mid_right": "lx521_top_base_3of4_mid_right",
    "piece_top_b2": "lx521_top_base_4of4_vase_b2",
    "attach_a_shoulder_top_left": "lx521_top_addonA_1of4_shoulder_top_left",
    "attach_a_shoulder_top_right": "lx521_top_addonA_2of4_shoulder_top_right",
    "attach_a_shoulder_bottom_left": "lx521_top_addonA_3of4_shoulder_bottom_left",
    "attach_a_shoulder_bottom_right": "lx521_top_addonA_4of4_shoulder_bottom_right",
    "attach_b1_wing_left": "lx521_top_addonB1_1of2_wing_left",
    "attach_b1_wing_right": "lx521_top_addonB1_2of2_wing_right",
}


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
    ap.add_argument("--variant", choices=("b2", "c7", "v0", "v1", "v1l", "obiwan"),
                    default="b2",
                    help="b2: base pieces + attachments; c7: the four "
                         "LM-knife-taper base pieces (attachments and "
                         "piece_top are shared with b2)")
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
            "optional_lm_keyed_1of2_bottom",
            "optional_lm_keyed_2of2_top",
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
                *out_dir.glob("lx521_top_obiwan_[1-4]of4_*.stl"),
                *out_dir.glob("lx521_top_obiwan_[1-4]of4_*.print.json")):
            _unlink_print_pair(legacy)
        expected = {
            "lx521_top_obiwan_core_1of2_lm_carrier.stl",
            "lx521_top_obiwan_core_2of2_um_carrier.stl",
            "lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl",
            "lx521_top_obiwan_optional_lm_keyed_2of2_top.stl",
            "lx521_top_obiwan_addon_tweeter_crescent.stl",
        }
        for legacy in (
                *out_dir.glob("lx521_top_obiwan_addon_*.stl"),
                *out_dir.glob("lx521_top_obiwan_addon_*.print.json")):
            stl_name = (
                legacy.name.removesuffix(".print.json") + ".stl"
                if legacy.name.endswith(".print.json") else legacy.name)
            if stl_name not in expected:
                _unlink_print_pair(legacy)
    elif args.variant == "v1l":
        from top_baffle_nd25fw4_v1l_split import pieces_v1l
        parts = {}
        if args.v1l_piece != "grommet":
            parts.update({
                STL_NAMES[k].replace("lx521_top_base_", "lx521_top_v1l_"): v
                for k, v in pieces_v1l(only=args.v1l_piece).items()
            })
        if args.v1l_piece in (None, "grommet"):
            from top_baffle_nd25fw4_um_fit import split_grommet_parts
            parts.update({
                f"lx521_top_v1l_addon_{name}": solid
                for name, solid in split_grommet_parts("v1l").items()
            })
    elif args.variant == "v1":
        from top_baffle_nd25fw4_v1_attachments import v1_attachments
        from top_baffle_nd25fw4_v1_split import pieces_v1
        parts = {"lx521_top_v1_4of4_vase": pieces_v1()["piece_top_b2"]}
        parts.update({k.replace("attach_v1a_", "lx521_top_v1addonA_")
                       .replace("attach_v1b1_", "lx521_top_v1addonB1_"): v
                      for k, v in v1_attachments().items()})
    elif args.variant == "v0":
        from top_baffle_nd25fw4_v0_split import pieces_v0
        parts = {"lx521_top_v0_4of4_vase":
                 pieces_v0()["piece_top_b2"]}
    elif args.variant == "c7":
        from top_baffle_nd25fw4_c7_split import pieces_c7
        parts = {STL_NAMES[k].replace("lx521_top_base_", "lx521_top_c7base_"):
                 v for k, v in pieces_c7().items()}
    else:
        from top_baffle_nd25fw4_attachments import attachments
        from top_baffle_nd25fw4_b2_split import pieces

        parts = dict(pieces())
        parts.update(attachments())
        parts = {STL_NAMES[k]: v for k, v in parts.items()}
        from top_baffle_nd25fw4_um_fit import split_grommet_parts
        parts.update({f"lx521_top_proud_addon_{name}": solid
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
            "obiwan_core_1of2_lm_carrier" in name)
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
        if (args.variant == "obiwan" and stand_mode == "0"
                and name ==
                "lx521_top_obiwan_optional_lm_keyed_1of2_bottom"):
            support_blocker_note = _export_no_floor_lm_support_blocker(
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
