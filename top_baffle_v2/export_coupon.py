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
    # loose male key (no clearance -- the pocket carries it)
    male = _poly_prism(_trapezoid_up(28.0, -26.0, NECK, HEAD, DEPTH), t)
    male += Pos(28.0, -32.0, t / 2.0) * Box(20.0, 12.0, t)
    return [plate, male]


def main() -> None:
    solids = coupon()
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
