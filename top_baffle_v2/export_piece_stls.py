"""Export print-ready STLs: the four B2 baffle pieces plus the six
attachment pieces that turn the B2 set into variant A-comp or B1.

Run:  python export_piece_stls.py
Each part is translated so its bounding box starts at the origin (still
lying flat, thickness along Z, front face up) and written to stl/<name>.stl.
Exits nonzero if any piece stops fitting the print bed.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

# Direct CLI use gets the same process-tree and free-memory protection as
# Make recipes. Imports from an already guarded test/export process do not
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

BED_MM = 256.0
V1LF_OPTIONAL_LM_SPLIT_BED_MM = 220.0

# A rigid OCC Location can leave mathematically-zero coordinates as tiny
# nonzero values on only one of two adjacent face triangulations.  Binary
# STL stores vertices face-by-face, so that harmless transform noise becomes
# an exact-edge seam to downstream slicers/checkers.  OCC's X45-plus-origin
# placement leaves the no-floor bridge datum at about 1e-7 mm on the two
# adjacent faces, differing by only 7.1e-15 mm.  This 0.2-nm threshold is
# still 250,000 times below the 0.05 mm mesh deflection; it canonicalizes
# rigid-transform roundoff only and is neither vertex welding nor CAD repair.
STL_TRANSFORM_ZERO_EPSILON_MM = 2.0e-7

# Slicer orientations for intentionally skeletal R6F pieces whose
# global-axis bounding boxes exceed the bed even though a supported
# rotation can fit the complete three-dimensional envelope.
BED_ROT_Z = {
    "v1lf_core_1of2_lm_carrier": 28.0,
    "v1lf_optional_lm_keyed_1of2_bottom": 26.0,
    "v1lf_optional_lm_keyed_2of2_top": 45.0,
    "v1lf_addon_mount_floor_support": 70.0,
}
V1LF_NO_FLOOR_LM_TILT_X_DEG = 45.0


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


def _routing_rev():
    return {"proud": "R6P", "v1lf": "R6F"}.get(
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
    if "v1lf_core_1of2_lm" in name:
        return f"V1LF-LM {_routing_rev()}"
    if "v1lf_core_2of2_um" in name:
        return f"V1LF-UM {_routing_rev()}"
    n = name.replace("lx521_top_", "")
    fam = {"base": "B2", "c7base": "C7", "v1l": "V1L", "v1lf": "V1LF",
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
    if "v1lf_" in name or "grommet" in name:
        return solid

    for suffix, (ax, ay, rot, font, short) in EMBOSS_XY.items():
        if suffix in name:
            break
    else:
        raise SystemExit(f"no emboss anchor for {name}")
    # rear z AT the anchor, not the piece's global min (V1LF pad
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
    ap.add_argument("--variant", choices=("b2", "c7", "v0", "v1", "v1l", "v1lf"),
                    default="b2",
                    help="b2: base pieces + attachments; c7: the four "
                         "LM-knife-taper base pieces (attachments and "
                         "piece_top are shared with b2)")
    ap.add_argument(
        "--v1lf-part",
        choices=("lm", "lm_split", "um", "support", "tweeter"),
        help="export one staged R6F group; omit on osado to mesh the whole "
             "state in one guarded process")
    ap.add_argument(
        "--v1lf-stage-manifest", type=Path,
        help="hash-verified native-BREP stage manifest produced by "
             "export_v1lf_staged.py; required for every V1LF export")
    ap.add_argument(
        "--v1l-piece",
        choices=("piece_bottom", "piece_mid_left", "piece_mid_right",
                 "piece_top_b2", "grommet"),
        help="build one V1L split piece or its terminal grommet pair; "
             "serial use releases OCC geometry between groups and "
             "protects the macOS free-memory floor")
    args = ap.parse_args()
    if args.v1lf_part and args.variant != "v1lf":
        ap.error("--v1lf-part requires --variant v1lf")
    if args.v1lf_stage_manifest and args.variant != "v1lf":
        ap.error("--v1lf-stage-manifest requires --variant v1lf")
    if args.v1l_piece and args.variant != "v1l":
        ap.error("--v1l-piece requires --variant v1l")
    stand_mode = os.environ.get("LX_STAND_FOOT", "1")
    if stand_mode not in {"0", "1"}:
        ap.error("LX_STAND_FOOT must be 0 or 1")
    if (args.variant == "v1lf" and args.v1lf_part is None
            and not _large_host_execution()):
        ap.error(
            "local --variant v1lf requires one --v1lf-part so every OCC "
            "group runs in a fresh guarded process")
    if args.variant == "v1lf" and args.v1lf_stage_manifest is None:
        ap.error(
            "--variant v1lf requires --v1lf-stage-manifest; direct "
            "monolithic carrier generation is intentionally disabled")
    if (args.variant == "v1lf" and args.v1lf_part == "support"
            and stand_mode == "0"):
        ap.error("the floor-support add-on does not exist in no-floor state")
    profile = "v1lf" if args.variant == "v1lf" else "proud"
    os.environ["LX_ROUTING_PROFILE"] = profile
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.variant == "v1lf":
        from export_v1lf_staged import (
            PRINT_PART_SPECS,
            load_stage_manifest,
            staged_part_paths,
        )

        payload = load_stage_manifest(args.v1lf_stage_manifest)
        staged = staged_part_paths(args.v1lf_stage_manifest, payload)
        keys = tuple(
            key for key, spec in PRINT_PART_SPECS.items()
            if key in staged and (
                args.v1lf_part is None
                or spec["group"] == args.v1lf_part))
        if not keys:
            ap.error(
                f"the staged {payload['state']} manifest has no "
                f"{args.v1lf_part!r} print group")
        parts = {
            PRINT_PART_SPECS[key]["stl_name"]: import_brep(str(staged[key]))
            for key in keys
        }
        # Prune artifacts owned by superseded V1LF generators.  In
        # particular this removes the deleted external R14 raceway and
        # right alignment link from incremental output directories.
        for legacy in out_dir.glob("lx521_top_v1lf_[1-4]of4_*.stl"):
            legacy.unlink()
        expected = {
            "lx521_top_v1lf_core_1of2_lm_carrier.stl",
            "lx521_top_v1lf_core_2of2_um_carrier.stl",
            "lx521_top_v1lf_optional_lm_keyed_1of2_bottom.stl",
            "lx521_top_v1lf_optional_lm_keyed_2of2_top.stl",
            "lx521_top_v1lf_addon_tweeter_crescent.stl",
        }
        if stand_mode == "1":
            expected.add("lx521_top_v1lf_addon_mount_floor_support.stl")
        for legacy in out_dir.glob("lx521_top_v1lf_addon_*.stl"):
            if legacy.name not in expected:
                legacy.unlink()
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
        no_floor_lm_tilt = (
            "v1lf_core_1of2_lm_carrier" in name
            and os.environ.get("LX_STAND_FOOT", "1") == "0")
        bed_rotation = next((angle for key, angle in BED_ROT_Z.items()
                             if key in name), 0.0)
        orientation = ""
        if no_floor_lm_tilt:
            solid = Rot(X=V1LF_NO_FLOOR_LM_TILT_X_DEG) * solid
            orientation = f" @ X{V1LF_NO_FLOOR_LM_TILT_X_DEG:g}deg"
        elif bed_rotation:
            solid = Rot(Z=bed_rotation) * solid
            orientation = f" @ Z{bed_rotation:g}deg"
        bb = solid.bounding_box()
        size = bb.size
        bed_limit = (
            V1LF_OPTIONAL_LM_SPLIT_BED_MM
            if "v1lf_optional_lm_keyed_" in name else BED_MM)
        fits = (size.X <= bed_limit and size.Y <= bed_limit
                and size.Z <= bed_limit)
        if not fits:
            misfits.append(name)
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
        try:
            export_stl(
                moved, str(temporary), tolerance=0.05,
                angular_tolerance=0.2)
            _validate_binary_stl(temporary)
            # The exact-float seam is specific to the no-floor LM's X tilt
            # print transform.  Keep every other mesh byte-for-byte faithful
            # to the ordinary OCC export and let the strict contract reject
            # any unrelated topology defect.
            if no_floor_lm_tilt:
                canonicalized_zeros = _canonicalize_transform_zeros(temporary)
            mesh_facts = _strict_mesh_facts(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(
            f"{name:22s} {size.X:7.2f} x {size.Y:7.2f} x {size.Z:5.2f} mm  "
            f"volume {solid.volume / 1000.0:7.1f} cm3  "
            f"bed fit <= {bed_limit:g}: "
            f"{'OK' if fits else 'DOES NOT FIT'}"
            f"{orientation}"
            f"  mesh {mesh_facts['triangles']} tris/strict"
            f"  transform-zero fixes {canonicalized_zeros}"
            f"  -> {path.name}"
        )
    if misfits:
        sys.exit("ERROR: piece(s) exceed their configured bed envelope: "
                 + ", ".join(misfits))


if __name__ == "__main__":
    main()
