"""Export the print-calibration coupon as EIGHT separate, print-ready
STL files (lx521_coupon_*.stl) -- one body per file, each laid flat on
the bed so the slicer places it without reorientation.

  1 fit_plate    dovetail female pocket (grown by the working
                 CLEARANCE_MM) + O6.4 x 6.8 (W22) and O4.6 x 4.0
                 (10F) heat-set bores opening UP + the V1 upper-pocket
                 wall section (8.2 wall, O5.4 x 1.0 pin pocket: 1.2
                 front / 1.6 floor walls)
  2 fit_key      loose male dovetail key (no clearance) -- tune X-Y
                 hole compensation until it slides snug into the plate
  3 fish_entry   the no-foot entry cluster: twin T ramps, feeders and
                 the O6.8 Y-step
  4 fish_um_exit the top of the UM arc + its straight-back exit bore
                 (round-5; the old window-bend region carries no duct)
  5 fish_ts_dive the TS notch dive at full 18.3 depth (B2/C7/V0
                 family; the duct is the round-5 oval there too)
  6 fish_foot    a stand-foot R14 elbow pair
  7 recess_seat  V1LF: a U22REX/P-SL recess-seat SECTOR (~46 deg of
                 seat arc) with ~25 mm of through-void inboard of the
                 D190 cutout edge, the 270-deg insert bore over its
                 rear pad, and the LM exit. METHOD -- you never lower
                 the driver into the block: sit the driver CONE-UP on
                 its magnet on the table and FLIP THE BLOCK front-
                 face-down onto it, so the seat drops over the flange
                 edge (the through-void clears the cone/surround; the
                 motor never matters). Straightedge across block face
                 vs flange top = flushness; rotate the driver so a
                 flange hole meets the block's pilot for a REAL
                 M5 x 12 clamp test into the insert-on-pad stack.
                 Verify seat depth = true flange-edge thickness
                 (owner measured 6.0) BEFORE printing pieces
  8 fish_um_oval V1LF: the whole vase run -- morph, lane, crest, notch
                 under the MU10 recess, exit morph -- the worst pull
                 in the project; dry-fish both tweeter pairs here.
                 The block also carries the LEFT arc of the MU10 seat
                 with the cutout void inboard: same flip-onto-driver
                 method as block 7 (MU10 cone-up on its magnet;
                 measure vs its 5.4 datasheet / 4.0 measured flange!)

Fit pieces: same profile as the real parts. Fishing blocks: 2 walls /
~8 % infill (practice holes, not structure). Dry-fish 3-8 before
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
    plate -= Pos(0.0, 20.0, t - 3.2) * Cylinder(3.2, 7.2)  # O6.4 x 6.8
    plate -= Pos(14.0, 20.0, t - 2.0) * Cylinder(2.3, 4.2)
    # V1 upper-pocket wall section: ledge with local rear at z=10.1
    # (8.2 wall), O5.4 x 1.0 pocket at zc=14.4 bored from the y=0 edge:
    # front wall 18.3-(14.4+2.7)=1.2, floor wall 14.4-2.7-10.1=1.6
    plate -= Pos(28.0, 5.0, 10.1 / 2.0 - 0.5) * Box(24.0, 12.0, 10.1 + 1.0)
    plate -= (Pos(28.0, 0.55, 14.4) * Rot(X=90.0) * Cylinder(2.7, 3.1))
    # one clean extrusion, z 0..t: trapezoid + a grip handle on the
    # HEAD (wide) side, so when the key drops into the plate's edge
    # slot the handle sticks OUT past the edge (a neck-side handle would
    # ram into the plate body -- it cannot then be inserted).
    key2d = unary_union([_trapezoid_up(0.0, 0.0, NECK, HEAD, DEPTH),
                         sbox(-10.0, DEPTH - 0.01, 10.0, DEPTH + 12.0)])
    kface = make_face(Wire(Polyline(*key2d.exterior.coords).edges()))
    male = extrude(Plane.XY * kface, amount=t)
    return {"1_fit_plate": plate, "2_fit_key": male}


def _fishing_pieces() -> dict:
    """Real duct geometry carved from a region box, one body per hard
    fishing spot."""
    import importlib
    import os
    import sys

    # name -> (LX_STAND_FOOT, region box, rear z0, V1LF-flush state)
    specs = {
        "3_fish_entry": ("0", (-24.0, 42.0, 14.0, 70.0), 0.0, False),
        "4_fish_um_exit": ("1", (70.0, 266.0, 102.0, 298.0), 0.0, False),
        "5_fish_ts_dive": ("0", (-44.0, 390.0, 0.0, 434.0), 0.0, False),
        "6_fish_foot": ("1", (-20.0, 2.0, 20.0, 32.0), -22.0, False),
        "7_recess_seat": ("1", (-40.0, 78.0, 40.0, 131.0), 0.0, True),
        "8_fish_um_oval": ("1", (-58.0, 318.0, 2.0, 430.0), 0.0, True),
    }
    out = {}
    for name, (mode, (x0, y0, x1, y1), z0, flush) in specs.items():
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
        if flush:  # replicate the V1LF state: thin field with pad
            # reliefs, vase slab, flange recesses, deepened pilots --
            # AND the driver cutouts, which in the real pieces come
            # from baffle_face, not from any cutter (without them the
            # blocks had no hole to take a driver: the flange could
            # never reach its seat)
            import top_baffle_nd25fw4_flush as fl
            import top_baffle_nd25fw4_v1 as v1
            from top_baffle_nd25fw4 import L22_CUTOUT, UM_CUTOUT
            for ccx, ccy, ccd in (L22_CUTOUT, UM_CUTOUT):
                blk -= Pos(ccx, ccy, THICKNESS_MM / 2.0) * Cylinder(
                    ccd / 2.0, THICKNESS_MM + 4.0)
            for c in (list(fl.v1lf_field_cutters())
                      + list(v1.field_cutters())
                      + fl.recess_cutters() + fl.deep_pilot_cutters()):
                blk -= c
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
        for stale in ("lx521_coupon.stl", "lx521_coupon_4_fish_um_bend.stl"):
            old = stl_dir / stale
            if old.exists():
                old.unlink()
        for name, solid in pieces.items():
            out = stl_dir / f"lx521_coupon_{name}.stl"
            export_stl(_lay_flat(solid), str(out),
                       tolerance=0.05, angular_tolerance=0.2)
            print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
