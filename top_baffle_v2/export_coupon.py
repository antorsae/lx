"""Export the print-calibration coupon as SIX separate, print-ready
STL files (lx521_coupon_*.stl) -- one body per file, each laid flat on
the bed so the slicer places it without reorientation.

  1 fit_plate    dovetail female pocket (grown by the working
                 CLEARANCE_MM = 0.10) + O6.4 x 7.0 (W22) and O4.6 x 4.0
                 (10F) heat-set bores opening UP + the V1 upper-pocket
                 wall section (8.2 wall, O5.4 x 1.0 pin pocket: 1.2
                 front / 1.6 floor walls)
  2 fit_key      loose male dovetail key (no clearance) -- tune X-Y
                 hole compensation until it slides snug into the plate
  3 fish_entry   the no-foot entry cluster: twin T ramps, feeders and
                 the O6.8 Y-step
  4 fish_um_bend the UM window bend + exit -- the hardest single pull
  5 fish_ts_dive the TS notch dive
  6 fish_foot    a stand-foot R14 elbow pair

Fit pieces: same profile as the real parts. Fishing blocks: 2 walls /
~8 % infill (practice holes, not structure). Dry-fish 3-6 before
committing kilograms.
"""

from __future__ import annotations

from pathlib import Path

from build123d import (Box, Cylinder, Plane, Polyline, Pos, Rot, Wire,
                       export_stl, extrude, make_face)

from shapely import box as sbox
from shapely.ops import unary_union

from top_baffle_nd25fw4 import THICKNESS_MM
from top_baffle_nd25fw4_b2_split import CLEARANCE_MM, _grown, _trapezoid_up

NECK, HEAD, DEPTH = 10.0, 14.0, 6.0   # seam-B key proportions


def _poly_prism(poly, h):
    pts = list(poly.exterior.coords)
    face = make_face(Wire(Polyline(*pts).edges()))
    return extrude(Plane.XY.offset(-0.5) * face, amount=h + 1.0)


def _fit_pieces() -> dict:
    t = THICKNESS_MM
    plate = Pos(0.0, 20.0, t / 2.0) * Box(92.0, 40.0, t)
    # female dovetail pocket in the y=40 edge (grown like the real seams)
    pocket = _grown(_trapezoid_up(-28.0, 40.0 - DEPTH, NECK, HEAD, DEPTH))
    plate -= _poly_prism(pocket, t)
    # heat-set bores from the top face (open UP for easy insert setting)
    plate -= Pos(0.0, 20.0, t - 3.5) * Cylinder(3.2, 7.2)
    plate -= Pos(14.0, 20.0, t - 2.0) * Cylinder(2.3, 4.2)
    # V1 upper-pocket wall section: ledge with local rear at z=10.1
    # (8.2 wall), O5.4 x 1.0 pocket at zc=14.4 bored from the y=0 edge:
    # front wall 18.3-(14.4+2.7)=1.2, floor wall 14.4-2.7-10.1=1.6
    plate -= Pos(28.0, 5.0, 10.1 / 2.0 - 0.5) * Box(24.0, 12.0, 10.1 + 1.0)
    plate -= (Pos(28.0, 0.55, 14.4) * Rot(X=90.0) * Cylinder(2.7, 3.1))
    # one clean extrusion (trapezoid head + handle rect), z 0..t
    key2d = unary_union([_trapezoid_up(0.0, 0.0, NECK, HEAD, DEPTH),
                         sbox(-10.0, -12.0, 10.0, 0.01)])
    kface = make_face(Wire(Polyline(*key2d.exterior.coords).edges()))
    male = extrude(Plane.XY * kface, amount=t)
    return {"1_fit_plate": plate, "2_fit_key": male}


def _fishing_pieces() -> dict:
    """Real duct geometry carved from a region box, one body per hard
    fishing spot."""
    import importlib
    import os
    import sys

    # name -> (LX_STAND_FOOT, region box, rear z0)
    specs = {
        "3_fish_entry": ("0", (-24.0, 42.0, 14.0, 70.0), 0.0),
        "4_fish_um_bend": ("0", (-2.0, 296.0, 24.0, 330.0), 0.0),
        "5_fish_ts_dive": ("0", (-44.0, 390.0, 0.0, 434.0), 0.0),
        "6_fish_foot": ("1", (-20.0, 2.0, 20.0, 32.0), -22.0),
    }
    out = {}
    for name, (mode, (x0, y0, x1, y1), z0) in specs.items():
        os.environ["LX_STAND_FOOT"] = mode
        for m in ("top_baffle_nd25fw4", "top_baffle_nd25fw4_cables"):
            if m in sys.modules:
                importlib.reload(sys.modules[m])
            else:
                importlib.import_module(m)
        cab = sys.modules["top_baffle_nd25fw4_cables"]
        blk = Pos((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                  (z0 + THICKNESS_MM) / 2.0) * Box(
            x1 - x0, y1 - y0, THICKNESS_MM - z0)
        for c in cab.cable_cutters():
            blk -= c
        out[name] = blk
    return out


def _lay_flat(solid):
    """Rotate onto the flattest footprint (min Z height), preferring the
    native pose on ties, then drop bbox-min to the origin."""
    best_h = solid.bounding_box().size.Z
    best = solid
    for rx, ry in ((90.0, 0.0), (0.0, 90.0), (180.0, 0.0)):
        cand = Rot(X=rx, Y=ry) * solid
        h = cand.bounding_box().size.Z
        if h < best_h - 0.5:
            best_h, best = h, cand
    bb = best.bounding_box()
    return Pos(-bb.min.X, -bb.min.Y, -bb.min.Z) * best


def main() -> None:
    pieces = {**_fit_pieces(), **_fishing_pieces()}
    for d in ("floor_stand", "no_floor_stand"):
        stl_dir = Path(__file__).parent / d / "stl"
        stl_dir.mkdir(parents=True, exist_ok=True)
        old = stl_dir / "lx521_coupon.stl"
        if old.exists():
            old.unlink()
        for name, solid in pieces.items():
            out = stl_dir / f"lx521_coupon_{name}.stl"
            export_stl(_lay_flat(solid), str(out),
                       tolerance=0.05, angular_tolerance=0.2)
            print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
