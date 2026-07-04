"""Export print-ready STLs: the four B2 baffle pieces plus the four
attachment pieces that turn the B2 set into variant A-comp or B1.

Run:  python export_piece_stls.py
Each part is translated so its bounding box starts at the origin (still
lying flat, thickness along Z, front face up) and written to stl/<name>.stl.
"""

from __future__ import annotations

import argparse
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
    out_dir = ap.parse_args().outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = dict(pieces())
    parts.update(attachments())
    parts = {STL_NAMES[k]: v for k, v in parts.items()}
    for name, solid in parts.items():
        bb = solid.bounding_box()
        size = bb.size
        fits = size.X <= BED_MM and size.Y <= BED_MM
        moved = Pos(-bb.min.X, -bb.min.Y, -bb.min.Z) * solid
        path = out_dir / f"{name}.stl"
        export_stl(moved, str(path), tolerance=0.05, angular_tolerance=0.2)
        print(
            f"{name:22s} {size.X:7.2f} x {size.Y:7.2f} x {size.Z:5.2f} mm  "
            f"volume {solid.volume / 1000.0:7.1f} cm3  "
            f"bed fit: {'OK' if fits else 'DOES NOT FIT'}  -> {path.name}"
        )


if __name__ == "__main__":
    main()
