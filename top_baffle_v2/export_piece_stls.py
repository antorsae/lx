"""Export print-ready STLs: the four B2 baffle pieces plus the six
attachment pieces that turn the B2 set into variant A-comp or B1.

Run:  python export_piece_stls.py
Each part is translated so its bounding box starts at the origin (still
lying flat, thickness along Z, front face up) and written to stl/<name>.stl.
Exits nonzero if any piece stops fitting the print bed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from build123d import Plane, Pos, Text, export_stl, extrude

from top_baffle_nd25fw4_attachments import attachments
from top_baffle_nd25fw4_b2_split import pieces

BED_MM = 256.0
ROUTING_REV = "R4"  # bump when the common routing changes

# safe embossing anchors (flat local rear, clear of pockets/thin zones)
EMBOSS_XY = {
    "1of4_bottom": (70.0, 40.0),
    "2of4_mid_left": (-95.0, 160.0),
    "3of4_mid_right": (95.0, 160.0),
    "4of4_vase": (0.0, 320.6),
}


def _label(name):
    """Short provenance code, e.g. B2-1, V1L-3, V1A-TL, B1-WL."""
    n = name.replace("lx521_top_", "")
    fam = {"base": "B2", "c7base": "C7", "v1l": "V1L", "v1": "V1",
           "addonA": "A", "addonB1": "B1", "v1addonA": "V1A",
           "v1addonB1": "V1B1"}
    head = n.split("_")[0]
    code = fam.get(head, head.upper())
    if "of4" in n or "of2" in n:
        code += "-" + n.split("_")[1][0]
    else:
        parts = n.split("_")
        code += "-" + "".join(w[0] for w in parts[1:]).upper()
    return f"{code} {ROUTING_REV}"


def _emboss(solid, name):
    """Recessed 0.4 mm ID text on the piece's hidden rear face,
    centered on a safe flat spot, mirrored to read correctly when
    looking AT the rear; rotated 90 deg on narrow pieces."""
    from build123d import Rot, mirror

    for suffix, (ax, ay) in EMBOSS_XY.items():
        if suffix in name:
            break
    else:
        c = solid.bounding_box().center()
        ax, ay = c.X, c.Y
    bb = solid.bounding_box()
    rot = 90.0 if (bb.size.X < 34.0) else 0.0
    zr = bb.min.Z
    txt = mirror(Text(_label(name), font_size=4.0), Plane.YZ)
    cutter = extrude(Pos(ax, ay, zr - 0.4) * Rot(Z=rot) * txt, amount=0.8)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUT_DIR,
                    help="directory for the STLs (default: stl/)")
    ap.add_argument("--variant", choices=("b2", "c7", "v0", "v1", "v1l"), default="b2",
                    help="b2: base pieces + attachments; c7: the four "
                         "LM-knife-taper base pieces (attachments and "
                         "piece_top are shared with b2)")
    args = ap.parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.variant == "v1l":
        from top_baffle_nd25fw4_v1l_split import pieces_v1l
        parts = {STL_NAMES[k].replace("lx521_top_base_", "lx521_top_v1l_"):
                 v for k, v in pieces_v1l().items()}
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
        parts = dict(pieces())
        parts.update(attachments())
        parts = {STL_NAMES[k]: v for k, v in parts.items()}
    misfits = []
    for name, solid in parts.items():
        solid = _emboss(solid, name)
        bb = solid.bounding_box()
        size = bb.size
        fits = size.X <= BED_MM and size.Y <= BED_MM
        if not fits:
            misfits.append(name)
        moved = Pos(-bb.min.X, -bb.min.Y, -bb.min.Z) * solid
        path = out_dir / f"{name}.stl"
        export_stl(moved, str(path), tolerance=0.05, angular_tolerance=0.2)
        print(
            f"{name:22s} {size.X:7.2f} x {size.Y:7.2f} x {size.Z:5.2f} mm  "
            f"volume {solid.volume / 1000.0:7.1f} cm3  "
            f"bed fit: {'OK' if fits else 'DOES NOT FIT'}  -> {path.name}"
        )
    if misfits:
        sys.exit(f"ERROR: piece(s) exceed the {BED_MM:.0f} mm bed: "
                 + ", ".join(misfits))


if __name__ == "__main__":
    main()
