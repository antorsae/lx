"""Export the 20-minute print-calibration coupon (lx521_coupon.stl).

One plate + one loose key exercising every fit that matters before
committing kilograms of filament:

  * dovetail: a female seam-B pocket (grown by the working CLEARANCE_MM
    = 0.10) in the plate edge + a loose male key -- tune X-Y hole
    compensation until it slides snugly by hand;
  * O6.4 x 7.0 bore (W22 M5 heat-set) and O4.6 x 4.0 bore (10F M3
    heat-set) -- test insert setting;
  * the V1 UPPER-POCKET WALL SECTION: a ledge replicating the site-2
    geometry exactly -- 8.2 mm local wall, horizontal O5.4 x 1.0 pin
    pocket with the 1.2 front wall and 1.6 floor wall. If this chips
    or prints porous, revisit before trusting 8 of them.

Print flat (plate face down), same profile as the real pieces.
"""

from __future__ import annotations

from pathlib import Path

from build123d import (Box, Cylinder, Plane, Polyline, Pos, Rot, Wire,
                       export_stl, extrude, make_face)

from top_baffle_nd25fw4 import THICKNESS_MM
from top_baffle_nd25fw4_b2_split import CLEARANCE_MM, _grown, _trapezoid_up

NECK, HEAD, DEPTH = 10.0, 14.0, 6.0   # seam-B key proportions


def _poly_prism(poly, h):
    pts = list(poly.exterior.coords)
    face = make_face(Wire(Polyline(*pts).edges()))
    return extrude(Plane.XY.offset(-0.5) * face, amount=h + 1.0)


def coupon() -> list:
    t = THICKNESS_MM
    plate = Pos(0.0, 20.0, t / 2.0) * Box(92.0, 40.0, t)
    # female dovetail pocket in the y=40 edge (grown like the real seams)
    pocket = _grown(_trapezoid_up(-28.0, 40.0 - DEPTH, NECK, HEAD, DEPTH))
    plate -= _poly_prism(pocket, t)
    # heat-set bores from the front (z=t face)
    plate -= Pos(0.0, 20.0, t - 3.5) * Cylinder(3.2, 7.2)
    plate -= Pos(14.0, 20.0, t - 2.0) * Cylinder(2.3, 4.2)
    # V1 upper-pocket wall section: ledge with local rear at z=10.1
    # (8.2 wall), O5.4 x 1.0 pocket at zc=14.4 bored from the y=0 edge:
    # front wall 18.3-(14.4+2.7)=1.2, floor wall 14.4-2.7-10.1=1.6
    plate -= Pos(28.0, 5.0, 10.1 / 2.0 - 0.5) * Box(24.0, 12.0, 10.1 + 1.0)
    plate -= (Pos(28.0, 0.55, 14.4) * Rot(X=90.0) * Cylinder(2.7, 3.1))
    # loose male key (no clearance -- the pocket carries it), parked in
    # its own clear cell right of the fishing grid
    male = _poly_prism(_trapezoid_up(90.0, -92.0, NECK, HEAD, DEPTH), t)
    male += Pos(90.0, -98.0, t / 2.0) * Box(20.0, 12.0, t)
    return [plate, male]


def _fishing_blocks():
    """Real duct geometry carved from region boxes -- fishing rehearsal
    before the real pieces: (A) the no-foot entry cluster with the twin
    T ramps, feeders and the O6.8 Y-step; (C) the UM window bend + exit
    (the hardest pull); (D) the TS notch dive; (B) a stand-foot R14
    elbow pair. Print with 2 walls / ~8 % infill -- these are practice
    holes, not structure."""
    import importlib
    import os
    import sys

    # (region box, placement of the region's LOWER-LEFT corner). Blocks
    # are laid on a grid below the plate (plate occupies y 0..40), each
    # in its own 44 x 44 cell with >=6 mm gaps -- no fusion.
    regions = {
        "0": [((-24.0, 42.0, 14.0, 70.0), (-90.0, -60.0)),    # entry cluster
              ((-2.0, 296.0, 24.0, 330.0), (-30.0, -60.0)),   # UM window bend
              ((-44.0, 390.0, 0.0, 434.0), (30.0, -60.0))],   # TS notch dive
        "1": [((-20.0, 2.0, 20.0, 32.0), (-90.0, -120.0))],   # foot R14 elbow
    }
    blocks = []
    for mode, regs in regions.items():
        os.environ["LX_STAND_FOOT"] = mode
        for m in ("top_baffle_nd25fw4", "top_baffle_nd25fw4_cables"):
            if m in sys.modules:
                importlib.reload(sys.modules[m])
            else:
                importlib.import_module(m)
        cab = sys.modules["top_baffle_nd25fw4_cables"]
        cutters = cab.cable_cutters()
        z0 = -22.0 if mode == "1" else 0.0
        for (x0, y0, x1, y1), (px, py) in regs:
            blk = Pos((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                      (z0 + THICKNESS_MM) / 2.0) * Box(
                x1 - x0, y1 - y0, THICKNESS_MM - z0)
            for c in cutters:
                blk -= c
            # move region's lower-left (x0,y0,rear) to the grid cell (px,py)
            blocks.append(Pos(px - x0, py - y0, -z0) * blk)
    return blocks


def main() -> None:
    solids = coupon() + _fishing_blocks()
    part = solids[0]
    for s in solids[1:]:
        part += s
    for d in ("floor_stand", "no_floor_stand"):
        out = Path(__file__).parent / d / "stl" / "lx521_coupon.stl"
        out.parent.mkdir(parents=True, exist_ok=True)
        export_stl(part, str(out), tolerance=0.05, angular_tolerance=0.2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
