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

from build123d import Pos, export_stl

from top_baffle_nd25fw4_attachments import attachments
from top_baffle_nd25fw4_b2_split import pieces

BED_MM = 256.0
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
        from top_baffle_nd25fw4_v1_split import pieces_v1
        parts = {"lx521_top_v1_4of4_vase": pieces_v1()["piece_top_b2"]}
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
