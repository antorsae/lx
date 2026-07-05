"""Internal cable ducts -- four routes, one per driver voice.

Each route is up to three OVERLAPPING cutters; the mains are strictly
planar because every inline z-eased spline variant broke OCC booleans:

  MAINS (planar spline sweeps, common to both stand-foot states):
    LM (blue):  2 x 2.5 mm^2 twisted pair, O8.5 at mid-plane z=9.15;
        straight plan line drifting x=-8.6 -> -10.5 past the 270-deg
        W22 pilot (2.3 mm wall).
    UM (green): twisted 2 x 2.0 mm^2 pair (~O7.0 bundle), O7.8 DEEP
        at z=5.7: R26 fan fillet, tangent line onto ONE constant arc
        r=100.7 hugging the D190 rim (wall 1.8), passing UNDER the M5
        heat-set W22 pilots (floor z=11.3, 1.7 mm no-load membrane --
        same construction as the T ducts under the 10F ring), then the
        exit tail. Entirely inside the C7 full-depth core, so ALL
        piece variants share one routing. 2 x 2.5 mm^2 no longer fits.
    T1/T2 (yellow/red): 2 x AWG24 each, O3.8, in TWO planar mains
        joined by a straight z-step bore at the seam-B elbow corridor
        (T_STEP, 5.7 -> 3.7). LOWER (z=5.7): fan line tangent onto the
        r=110 arc -- fully buried in the C7 core, no ribs. UPPER
        (z=3.7, vase unchanged): flank lane under the 10F heat-set
        bores (D4.6 x 7.0: 5.7 mm floor), R20 crest fillet, head-on
        pierce of the D78.5 scallop rim (faces the tweeter pole gap).

  ENTRIES (state-dependent): with STAND_FOOT the mains continue down
    the plate into packed foot lanes (FOOT_LANES: 90-deg R14 vertical-
    plane elbows, then rearward to the connector channel's step face
    at z=-99). Without it, straight oblique ramp bores (BIG_RAMPS /
    T_RAMP) pierce the rear face inside the support plate's D20 window:
    LM/UM breakouts at (-/+5.2, 60.5), T1/T2 ovals at (+/-3.85, 52.2).

  EXITS (both states): straight oblique bores (EXIT_RAMPS) dive from
    the planar mains through the driver-cutout walls; openings center
    at z~6.2, in the rear quarter that the drivers' baskets do not
    occlude. The T mains already run at ~1/4 depth and exit head-on.

The ducts cross the glue seams; cables are laid/fished through each
piece's open segments during assembly. Clearances (duct-duct, W22
pilots, magnet pockets, foot-lane webs) are checked by
test_clearances.py (make check).

NOTE: the STAND_FOOT entry knots feed the same interpolating spline as
the shared main, which shifts the shared tail by <0.05 mm between the
two flag states -- far under the 0.10 seam clearance, but the mid/top
STLs are not byte-identical between floor_stand/ and no_floor_stand/.
"""

from __future__ import annotations

import math

from build123d import Circle, Cylinder, Plane, Pos, Rot, Spline, sweep

from top_baffle_nd25fw4 import STAND_FOOT

LM_Y = 200.981
UM_Y = 366.081

R_T = 124.0
# Duct plane depth per route. The T uppers run at z=3.7 (rear skin
# 1.8, roof 5.6); the D4.6 x 7.0 10F heat-set bores (floor z=11.3)
# keep a solid 5.7 mm floor above them at the lane crossings.
# The O8.5 LM duct runs at mid-plane (skins 4.9 both sides).
DUCT_Z = {"lm": 9.15, "um": 5.7, "t1": 5.7, "t2": 5.7}  # below-seam
# planes; above seam B the ducts STEP to the FRONT half (T 10.7,
# UM 12.3) so thin vases can share the FRONT plane (V1 flush mount)
EDGE_OFFSET = 6.16  # duct center 5.9 mm inside the flare/chamfer walls

# LM carries 2 x 2.5 mm^2 (twisted pair ~O7.8) -> O8.5 duct, mid-plane.
# UM carries a TWISTED pair in ONE round bore: O8.6 at mid-plane -- the
# max its route allows (enabled by T arcs at r=124, the right seam-B
# dovetail at cx=21.5 with the T elbow at x=33, and slimmer seam-A
# dovetails). Comfortable for twisted 2 x 2.0 mm^2 (bundle ~O7);
# 2 x 2.5 mm^2 twisted (~O7.8) is snug. T = 2 x AWG24.
CABLE_D = {"lm": 8.5, "um": 7.8, "t1": 3.8, "t2": 3.8}

# Mains are planar constant-z sweeps; entries and exits are separate
# straight oblique bores that OVERLAP the mains (see cable_cutters) --
# a continuous void without 3D-eased splines, which OCC can't boolean
# reliably against this solid.


def _with_z(pts2d, knots):
    """Attach z to 2D route points by cumulative path length, piecewise-
    linear between (s, z) knots (the spline smooths knot corners)."""
    out, s_acc = [], 0.0
    for i, p in enumerate(pts2d):
        if i:
            s_acc += math.dist(p, pts2d[i - 1])
        z = knots[-1][1]
        for (s0, z0), (s1, z1) in zip(knots, knots[1:]):
            if s_acc <= s1:
                f = 0.0 if s1 == s0 else (s_acc - s0) / (s1 - s0)
                z = z0 + f * (z1 - z0)
                break
        out.append((p[0], p[1], z))
    return out


def _arc(r, thetas_deg):
    return [
        (r * math.cos(math.radians(t)), LM_Y + r * math.sin(math.radians(t)))
        for t in thetas_deg
    ]


def _flare_offset_r(y):
    """Duct x on the right flank, EDGE_OFFSET inside the flare edge."""
    return (0.29752 * y - 55.888) - EDGE_OFFSET


def _chamfer_offset_r(y):
    """Duct x on the right flank, EDGE_OFFSET inside the chamfer edge."""
    return -(0.88592 * y - 369.25) / 0.46375


def _line_arc_line(p0, d0, p1, d1, r, step_mm=6.0, step_deg=6.0):
    """Dense points along: straight from p0 (heading d0) -> tangent
    fillet of radius r -> straight to p1 (arriving with heading d1).
    Both headings must be unit vectors along the travel direction."""
    x0, y0 = p0
    # corner = intersection of the two rays
    den = d0[0] * d1[1] - d0[1] * d1[0]
    t = ((p1[0] - x0) * d1[1] - (p1[1] - y0) * d1[0]) / den
    cx, cy = x0 + d0[0] * t, y0 + d0[1] * t
    cos_turn = d0[0] * d1[0] + d0[1] * d1[1]
    half = math.acos(max(-1.0, min(1.0, cos_turn))) / 2.0
    off = r * math.tan(half)
    t1 = (cx - d0[0] * off, cy - d0[1] * off)
    t2 = (cx + d1[0] * off, cy + d1[1] * off)
    cw = d0[0] * d1[1] - d0[1] * d1[0] < 0
    n0 = (d0[1], -d0[0]) if cw else (-d0[1], d0[0])
    ctr = (t1[0] + n0[0] * r, t1[1] + n0[1] * r)
    pts = []
    run = math.dist(p0, t1)
    for i in range(max(1, int(run / step_mm))):
        s = i * run / max(1, int(run / step_mm))
        pts.append((x0 + d0[0] * s, y0 + d0[1] * s))
    a0 = math.atan2(t1[1] - ctr[1], t1[0] - ctr[0])
    a1 = math.atan2(t2[1] - ctr[1], t2[0] - ctr[0])
    sweep = a1 - a0
    if cw and sweep > 0:
        sweep -= 2 * math.pi
    if not cw and sweep < 0:
        sweep += 2 * math.pi
    n = max(2, int(abs(math.degrees(sweep)) / step_deg))
    for i in range(n + 1):
        a = a0 + sweep * i / n
        pts.append((ctr[0] + r * math.cos(a), ctr[1] + r * math.sin(a)))
    run2 = math.dist(t2, p1)
    for i in range(1, max(1, int(run2 / step_mm))):
        s = i * run2 / max(1, int(run2 / step_mm))
        pts.append((t2[0] + d1[0] * s, t2[1] + d1[1] * s))
    return pts


def _elbow_points():
    """Exact R12 fillet turning to vertical at x=33 (center (45, 317)),
    then a gentle lean onto the flare lane. Walls: 3.2 mm to the waist
    kink corner, 4.7 mm to the chamfer edge, 2.45 mm to the (moved)
    seam-B tab pocket at 28.65."""
    cx, cy, r = 45.0, 317.0, 12.0
    # the transition arrives tangent to the fillet at its 250-deg point
    pts = []
    for a in range(250, 173, -7):  # 250 deg .. 180 deg, CW to vertical
        pts.append((cx + r * math.cos(math.radians(a)),
                    cy + r * math.sin(math.radians(a))))
    # lean out onto the flare lane
    pts += [(33.1, 319.5), (33.8, 322.8), (34.9, 326.3), (35.9, 330.0)]
    return pts


UM_STEP = ((8.6, 306.5, 5.7), (6.5, 316.5, 12.55))  # UM's own
# seam z-step bore (O8.6) to the front-half vase plane
T_STEP = ((34.5, 311.0, 5.7), (33.2, 316.2, 10.7))  # rises to the
# front-half vase plane through the open seam-B elbow corridor
# bore stepping the T main 5.7 -> 3.7 through the open seam-B elbow
# z-step (2.3 wall to the waist kink and the seam-B pocket)


def _t1_route():
    # LOWER T main (planar z=5.7): fan line tangent onto the r=110 arc
    # -- fully inside the C7 core (>=20 from the taper edges at the
    # pinch), 9.3 outboard of the UM rim arc, over the W22 pilots with
    # the z membrane. Ends at the seam-B z-step bore (T_STEP).
    if STAND_FOOT:
        line = ([(13.9, 22.0), (14.4, 27.0), (15.4, 33.0)]
                + _line_arc_line((16.0, 38.0), (0.0, 1.0),
                                 (82.97, 128.74), (0.6570, 0.7540), 15.0))
    else:
        line = _line_arc_line((5.5, 51.5), (1.0, 0.0),
                              (82.97, 128.74), (0.6570, 0.7540), 12.0)
    return (
        line
        + _arc(110.0, [-41.05, -30, -18, -6, 6, 18, 30, 42, 50, 58, 64])
        + [(44.0, 302.0), (40.0, 304.5), (37.0, 307.5), (35.0, 310.5),
           (34.0, 313.0)]
    )


def _t1_upper():
    # UPPER T main (planar z=3.7, vase geometry unchanged): from the
    # old elbow lean-out up the flank lane, crest, crescent tail
    return (
        [(33.1, 319.5), (33.8, 322.8), (34.9, 326.3), (35.9, 330.0)]
        + [(_flare_offset_r(y), y) for y in (334, 342, 352, 360)]
        # R20 fillet at the crest (tangent to both offset lines)
        + [(48.94, 373.05), (49.68, 379.89), (48.11, 386.62),
           (44.34, 392.41), (38.94, 396.41)]
        + [(_chamfer_offset_r(y), y) for y in (400, 403, 406)]
        # dive to the middle of the crescent band (>=2.2 mm from the neck
        # corner (10.08, 418.18)), climb it and pierce the D78.5 scallop
        # rim head-on at ~(3.3, 429.8). No z-rise here: this opening faces
        # the tweeter POLE GAP, which stays open front-to-back (the
        # face-to-face pair clamps the faces, not the bore), so it cannot
        # be occluded -- and 3D-rising this curved tail breaks OCC.
        + [(16.0, 409.0), (11.8, 412.0), (7.5, 415.0), (5.0, 418.0),
           (3.6, 421.5), (3.2, 425.5), (3.3, 430.0), (3.5, 433.5)]
    )


# Without the stand foot the baffle bolts to the stock support via the
# four pass-throughs, and all four cables must pass a D20 hole in the
# support plate: center (0, 60) -- horizontally centered, top edge
# tangent to the line joining the two upper screws (y=70). Best-effort
# packing (two D9.3/9.4 bores alone span 19.75 of the 20):
#   LM/UM breakouts side by side, fully inside the window: steep ~55 deg
#     straight ramps crossing z=0 at (-/+5.2, 60.5), tips tucked inside
#     their mains at (-/+6.1, 66.9, 9.15). Surface lip between the two
#     openings is ~1.0 (grows past 1.5 within 3 mm of depth).
#   T1/T2 breakouts at the window's lower edge (ovals centered
#     (+/-3.85, 52.2), far tips ~0.8 mm past the rim; the floppy AWG24
#     pairs duck in): ~64 deg ramps, tips at (+/-5.5, 51.5, 3.7) where
#     the main runs straight to an R12 fillet onto the fan line.
SUPPORT_WINDOW = (0.0, 60.0, 20.0)  # cx, cy, D of the support-plate hole
# (tip overshoots the main's start (5.5, 51.5, 3.7) by 15% along the
# same axis so the ramp lances THROUGH the main tube -- ~4 mm of shared
# void; p0 pulled left so the ramp's plan direction is nearly along the
# main: the fishing bend at the tip is one ~66 deg turn, no compound S)
T_RAMP = ((1.0, 53.4, -6.4), (6.175, 51.215, 7.515))


# LM/UM entries: straight oblique ramps (p0 behind the rear face ->
# tip inside the planar main), O0.8 over their ducts.
BIG_RAMPS = {
    "lm": ((-4.6, 56.0, -6.4), (-8.0, 68.5, 9.15)),
    "um": ((4.8, 55.5, -6.4), (4.2, 67.0, 5.7)),
}
# Lowered exits (drivers occlude the cutout walls from mid-depth toward
# the front): straight oblique bores DIVING 6-10 deg from the planar
# mains through the rim walls; openings center at z~6.2 (the 1/4-depth
# target clamped by the 1.8 mm rear lip). T ducts already sit at ~1/4
# depth (z~1/4) natively.
EXIT_RAMPS = {
    "lm": ((-9.9, 86.0, 9.15), (-10.6, 119.0, 5.2), 9.3),
    "um": ((5.3, 320.5, 12.55), (2.95, 332.4, 12.0), 8.6),
}


# STAND_FOOT lanes: (x, duct z, run height y_f, elbow radius, bore D).
# Lane packing note: each pair of descent curves CROSSES in the (y,z)
# plane (the plate->foot turn), so the 1.5 mm webs must come from Dx
# alone: 8.45 + 10.85 + 8.5 mm of span inside the 38-wide tongue.
# R14 is the LARGEST elbow radius whose tube wraps the plate/foot
# inner corner (18.3, z=0) with >=1.4 mm clearance while staying
# inside plate+foot -- there is no corner rib to tunnel through.
FOOT_LANES = {
    "lm": (-5.45, 9.15, 10.5, 14.0, 9.3),
    "um": (5.4, 5.7, 10.5, 14.0, 8.6),
    "t1": (13.9, 5.7, 5.5, 14.0, 4.6),
    "t2": (-13.9, 5.7, 5.5, 14.0, 4.6),
}


def route_points(name):
    """Planar duct centerline (z=DUCT_Z), starting where the entry ramp
    has merged into the plane (y=72)."""
    if name == "lm":
        # Planar z=9.15; the line drifts from the x=-8.6 entry column to
        # x=-10.5 past the 270-deg W22 insert bore (D7.8: 2.3 mm wall).
        # The lowered exit is a straight oblique bore (EXIT_RAMPS).
        lead = ([(-5.45, 30.0, 9.15), (-6.2, 38.0, 9.15),
                 (-7.3, 48.0, 9.15), (-7.7, 58.0, 9.15)]
                if STAND_FOOT else [])
        return lead + [(-8.0, 68.0, 9.15), (-9.0, 78.0, 9.15),
                       (-10.0, 90.0, 9.15), (-10.5, 98.0, 9.15),
                       (-10.5, 103.0, 9.15)]
    if name == "um":
        # O7.8 bore, DEEP at z=5.7: the M5 heat-set pilots (floor
        # z=11.3) let the duct pass UNDER the whole W22 pilot ring with
        # a 1.7 mm no-load membrane, so it hugs the D190 rim on one
        # constant arc r=100.7 (rim wall 1.8) -- entirely inside the C7
        # full-depth core, no taper interaction anywhere. The main
        # starts at (4, 60) so the fan keeps >=8.6 to the T mains
        # (UM-T z split is only 2.0 now). Fully PLANAR; the exit is a
        # near-level oblique bore (EXIT_RAMPS).
        return (
            _with_z(
                ([(5.4, 30.0), (5.0, 38.0), (4.4, 47.0), (4.1, 56.0)]
                 if STAND_FOOT else [])
                + _line_arc_line((4.0, 60.0), (0.0, 1.0),
                               (58.63, 119.11), (0.81310, 0.58210), 26.0)
                + _arc(100.7, [-54.39, -42, -30, -18, -6, 6, 18, 30,
                               42, 54, 64, 72, 78])
                + [(16.0, 300.9), (11.5, 303.2), (8.6, 306.5)],
                [(0, 5.7), (9999, 5.7)])
            + [tuple(UM_STEP[0][i] + f * (UM_STEP[1][i] - UM_STEP[0][i])
                     for i in range(3)) for f in (0.34, 0.67, 1.0)]
            + [(6.3, 318.5, 12.55), (5.3, 320.5, 12.55),
               (4.9, 322.0, 12.55)]
        )
    if name == "t1":
        step = [tuple(T_STEP[0][i] + f * (T_STEP[1][i] - T_STEP[0][i])
                      for i in range(3))
                for f in (0.25, 0.5, 0.75, 1.0)]  # straight, like the bore
        return (_with_z(_t1_route(), [(0, 5.7), (9999, 5.7)])
                + step
                + _with_z(_t1_upper(), [(0, 10.7), (9999, 10.7)]))
    if name == "t2":
        return [(-p[0], *p[1:]) for p in route_points("t1")]
    raise ValueError(name)


def _entry_ramp(p0, p1, dia):
    """Straight oblique bore from outside the rear face (z<0) into the
    duct plane, along the arbitrary 3D axis p0 -> p1. Crosses z=0 at the
    rear breakout and overlaps the planar duct where it reaches z~5.5."""
    d = tuple(b - a for a, b in zip(p0, p1))
    length = math.dist(p0, p1)
    azimuth = math.degrees(math.atan2(d[1], d[0]))
    polar = math.degrees(math.acos(d[2] / length))
    mid = tuple((a + b) / 2.0 for a, b in zip(p0, p1))
    return (
        Pos(*mid)
        * Rot(Z=azimuth - 90.0)
        * Rot(X=-polar)
        * Cylinder(dia / 2.0, length)
    )


def cable_cutters():
    cutters = []
    for name in ("lm",):
        dia = CABLE_D[name]
        path = Spline(*route_points(name))
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(dia / 2.0)
        cutters.append(sweep(section, path=path))
    # UM: lower main + seam z-step bore + front-half vase main
    um_pts = route_points("um")
    for pts in (um_pts[:-6], um_pts[-3:]):
        path = Spline(*pts)
        section = (Plane(origin=path @ 0, z_dir=path % 0)
                   * Circle(CABLE_D["um"] / 2.0))
        cutters.append(sweep(section, path=path))
    cutters.append(_entry_ramp(UM_STEP[0], UM_STEP[1], CABLE_D["um"] + 0.8))
    for sign in (1.0, -1.0):  # T: lower + upper mains + seam-B z-step
        dia = CABLE_D["t1"]
        for pts in (_with_z(_t1_route(), [(0, 5.7), (9999, 5.7)]),
                    _with_z(_t1_upper(), [(0, 10.7), (9999, 10.7)])):
            path = Spline(*[(sign * x, y, z) for x, y, z in pts])
            section = (Plane(origin=path @ 0, z_dir=path % 0)
                       * Circle(dia / 2.0))
            cutters.append(sweep(section, path=path))
        p0 = (sign * T_STEP[0][0], *T_STEP[0][1:])
        p1 = (sign * T_STEP[1][0], *T_STEP[1][1:])
        cutters.append(_entry_ramp(p0, p1, 4.6))
    # LM/UM raised exits: straight oblique bores through the rim walls
    # (driver-side breakouts -- needed with or without the stand foot)
    for name, (p0, p1, dia) in EXIT_RAMPS.items():
        cutters.append(_entry_ramp(p0, p1, dia))
    if STAND_FOOT:
        # 90-deg vertical-plane elbows + runs along the foot, packed to
        # fit the 38-wide connector tongue, exiting through the channel
        # step face (z=-99, ~40 mm short of the NL8 panel). One planar
        # (constant-x) spline per route, O0.8 over its duct.
        for x, z_d, y_f, r, dia in FOOT_LANES.values():
            y_c, z_c = y_f + r, z_d - r
            pts = [(x, y_c + 7.5, z_d), (x, y_c + 4.0, z_d)]
            for a in range(0, 91, 15):
                pts.append((x, y_c - r * math.sin(math.radians(a)),
                            z_c + r * math.cos(math.radians(a))))
            # run the tongue, exiting through the channel step face z=-99
            for z in (z_c - 10.0, z_c - 30.0, -60.0, -85.0, -103.0):
                pts.append((x, y_f, z))
            path = Spline(*pts)
            section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(dia / 2.0)
            cutters.append(sweep(section, path=path))
        return cutters
    # T entries: straight oblique ramps (see T_RAMP)
    for sign in (1.0, -1.0):
        p0 = (sign * T_RAMP[0][0], T_RAMP[0][1], T_RAMP[0][2])
        p1 = (sign * T_RAMP[1][0], T_RAMP[1][1], T_RAMP[1][2])
        cutters.append(_entry_ramp(p0, p1, 4.6))
    # LM/UM entries: steep straight ramps into the D20 support window
    for name, (p0, p1) in BIG_RAMPS.items():
        cutters.append(_entry_ramp(p0, p1, CABLE_D[name] + 0.8))
    return cutters
