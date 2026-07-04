"""Internal cable ducts -- four routes, one per driver voice:

  LM (blue):  2 x 2.5 mm^2 twisted pair, O8.5 bore. Straight plan line at
      x=-8.6 (2.25 mm to the 270-deg W22 pilot). One continuous 3D
      spline: z eases 18 deg at the rear-face pierce -> ~9 deg -> exit
      into the D190 rim with the opening centered at exit_z (basket-ring
      occlusion sits at mid-depth).
  UM (green): twisted pair (2 x 2.0 mm^2 comfortable, 2 x 2.5 snug),
      O8.6 bore, one continuous 3D spline: eased entry like LM, R26 fan
      fillet, tangent line to the r=115.5 arc, ring-crossing between the
      30- and 90-deg W22 pilots, lane at r~102.2, one R28.8 exit arc
      entering the D82 rim near-radially, opening at exit_z.
  T1/T2 (yellow/red): 2 x AWG24 each, O3.8 planar ducts at z=3.7 (the
      flank lanes pass under the 10F insert bores, D5.8 x 11: the 1.7 mm
      floor above the duct roof is a no-load membrane -- see the note in
      top_baffle_nd25fw4.py). Straight fan line tangent
      to the r=124 arc, outboard transition, exact R12 elbow at x=33,
      flank lane, R20 crest fillet, head-on pierce of the D78.5 scallop
      rim (that opening faces the tweeter pole gap -- open front-to-back,
      so no z-rise is needed). Entries: straight 19-deg oblique ramps
      (O4.6) -- every z-eased spline variant broke OCC booleans here.

Rear breakouts sit at y~47 between the bridge screws (hidden by the
bridge): x = -16 (T2), -8.6 (LM), +4 (UM), +16 (T1). The ducts cross the
glue seams; cables are laid/fished through each piece's open segments
during assembly. All plan clearances (pilots, holes, seam tabs, edges,
duct-duct) and z walls are verified by the analytic suite + solid probes.
"""

from __future__ import annotations

import math

from build123d import Circle, Cylinder, Plane, Pos, Rot, Spline, sweep

from top_baffle_nd25fw4 import STAND_FOOT

LM_Y = 200.981
UM_Y = 366.081

R_GREEN = 115.5
R_T = 124.0
# Duct plane depth per route. The T ducts run at z=3.7 (rear skin 1.8,
# roof 5.6) so the D5.8 x 11 10F insert bores (floor z=7.3) keep a
# 1.7 mm no-load membrane above them at the lane crossings.
# The O8.5 LM duct runs at mid-plane (skins 4.9 both sides).
DUCT_Z = {"lm": 9.15, "um": 9.15, "t1": 3.7, "t2": 3.7}
# Exit openings sit at EXIT_Z_FRAC of the thickness (REAR quarter: the
# drivers' baskets occlude the cutout walls from mid-depth toward the
# front), clamped so the oblique opening keeps a >= 1.8 mm rear lip.
EXIT_Z_FRAC = 0.25
THICK = 18.3


def exit_z(name):
    return max(EXIT_Z_FRAC * THICK, 1.8 + CABLE_D[name] / 2.0 + 0.15)
EDGE_OFFSET = 6.16  # duct center 5.9 mm inside the flare/chamfer walls

# LM carries 2 x 2.5 mm^2 (twisted pair ~O7.8) -> O8.5 duct, mid-plane.
# UM carries a TWISTED pair in ONE round bore: O8.6 at mid-plane -- the
# max its route allows (enabled by T arcs at r=124, the right seam-B
# dovetail at cx=21.5 with the T elbow at x=33, and slimmer seam-A
# dovetails). Comfortable for twisted 2 x 2.0 mm^2 (bundle ~O7);
# 2 x 2.5 mm^2 twisted (~O7.8) is snug. T = 2 x AWG24.
CABLE_D = {"lm": 8.5, "um": 8.6, "t1": 3.8, "t2": 3.8}
ENTRY_X = {"lm": -8.6, "um": 4.0, "t1": 16.0, "t2": -16.0}

# Each duct is ONE continuous 3D spline: no separate entry-ramp bores.
# z follows a smooth eased profile along the route (piecewise knots in
# cumulative path length, smoothed by the spline): 18 deg only at the
# rear-face pierce (compact oval), easing to ~9 deg, then <1 deg drift.
# Breakouts (z=0 crossings) sit at y~47: x = -16, -8.6, +4, +16.


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


def _t1_route():
    # one dead-straight line, exactly tangent to the r=124 arc at
    # theta=-47.3 (14.2 mm from the (20,70) hole, 8.4 from the UM arc).
    # This main duct is strictly PLANAR (z=3.7): planar paths sweep with
    # a fixed frame, which the tight R12 elbow needs. The z-eased entry
    # is a separate overlapping bore (see _t_entry_pts).
    if STAND_FOOT:
        # drift to the packed foot lane (x=13.85), then a planar R15
        # fillet from vertical x=16 onto the tangent fan line
        line = ([(13.9, 22.0), (14.4, 27.0), (15.4, 33.0)]
                + _line_arc_line((16.0, 38.0), (0.0, 1.0),
                                 (84.055, 109.874), (0.7339, 0.6783), 15.0))
    else:
        # from the ramp tip at the support window's edge (5.5, 51.5):
        # dead straight rightward, then ONE exact R12 fillet tangent
        # onto the fan line at (24.3, 54.7) -- single bend, easy fishing
        line = _line_arc_line((5.5, 51.5), (1.0, 0.0),
                              (84.055, 109.874), (0.7339, 0.6783), 12.0)
    return (
        line
        + _arc(R_T, [-47.3, -36, -24, -12, 0, 12, 24, 36, 41])
        # transition to the waist corridor (>=8.2 outboard of the UM
        # spiral), easing onto the elbow fillet's 250-deg tangent line
        + [(80.5, 289.5), (70.5, 294.2), (67.22, 296.14),
           (59.7, 298.88), (53.12, 301.27), (46.54, 303.67)]
        # approach line + exact R12 fillet into the flare lane, sampled
        # densely so the spline tracks the true geometry. The outline kink
        # caps this elbow at ~R12; seam B cuts through it, so cables are
        # laid in through the open seam face during assembly.
        + _elbow_points()
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
T_RAMP = ((1.0, 53.4, -6.4), (6.175, 51.215, 5.215))


# LM/UM entries: straight oblique ramps (p0 behind the rear face ->
# tip inside the planar main), O0.8 over their ducts.
BIG_RAMPS = {
    "lm": ((-4.6, 56.0, -6.4), (-6.1, 66.9, 9.15)),
    "um": ((4.6, 56.0, -6.4), (6.1, 66.9, 9.15)),
}
# Lowered exits (drivers occlude the cutout walls from mid-depth toward
# the front): straight oblique bores DIVING 6-10 deg from the planar
# mains through the rim walls; openings center at z~6.2 (the 1/4-depth
# target clamped by the 1.8 mm rear lip). T ducts already sit at ~1/4
# depth (z~1/4) natively.
EXIT_RAMPS = {
    "lm": ((-9.9, 86.0, 9.15), (-10.6, 119.0, 5.2), 9.3),
    "um": ((10.2, 308.5, 9.15), (2.95, 332.4, 4.9), 9.4),
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
    "um": (5.4, 9.15, 10.5, 14.0, 9.4),
    "t1": (13.9, 3.7, 5.5, 14.0, 4.6),
    "t2": (-13.9, 3.7, 5.5, 14.0, 4.6),
}


def route_points(name):
    """Planar duct centerline (z=DUCT_Z), starting where the entry ramp
    has merged into the plane (y=72)."""
    if name == "lm":
        # Planar z=9.15; the line drifts from the x=-8.6 entry column to
        # x=-10.5 past the 270-deg W22 insert bore (D7.8: 2.3 mm wall).
        # The lowered exit is a straight oblique bore (EXIT_RAMPS).
        lead = ([(-5.45, 30.0, 9.15), (-6.2, 38.0, 9.15),
                 (-7.3, 48.0, 9.15)] if STAND_FOOT else [])
        return lead + [(-8.6, 63.0, 9.15), (-9.3, 76.0, 9.15),
                       (-10.1, 89.0, 9.15), (-10.5, 98.0, 9.15),
                       (-10.5, 103.0, 9.15)]
    if name == "um":
        # O8.6 bore, mid-plane: inner right arc, ring-crossing between
        # the 30- and 90-deg W22 pilots, lane at r~102.2 (rim wall 2.9),
        # then ONE continuous R28.8 exit arc (center (32.7, 328.1))
        # tangent to the lane, entering the D82 rim near-radially at
        # (4, 325.3) -- >=10.8 from the 90-deg pilot, 2.4 mm wall to the
        # seam-B tab corner. No intermediate straights or S-bends.
        spiral = [(46, 112.0), (52, 107.5), (58, 104.3), (63, 102.6),
                  (68, 102.25), (72, 102.2)]
        # main duct: fully PLANAR at z=9.15 (any inline 3D easing
        # corrupts OCC booleans); the raised exit is a straight oblique
        # bore (EXIT_RAMPS).
        return (
            _with_z(
                ([(5.4, 30.0), (5.0, 38.0), (4.4, 47.0)]
                 if STAND_FOOT else [])
                + _line_arc_line((4.0, 58.0), (0.0, 1.0),
                               (46.98, 95.49), (0.9135, 0.4067), 26.0)
                + _arc(R_GREEN, [-66, -54, -42, -30, -18, -6, 6, 18, 30, 40])
                + [(r * math.cos(math.radians(t)),
                    LM_Y + r * math.sin(math.radians(t))) for t, r in spiral]
                + [(27.78, 299.43), (23.76, 300.73), (20.49, 302.02),
                   (16.14, 304.53), (12.29, 307.75), (9.07, 311.60)],
                [(0, 9.15), (9999, 9.15)])
            + [(6.56, 315.95, 9.15), (5.3, 319.5, 9.15),
               (4.9, 321.5, 9.15)]
        )
    if name == "t1":
        return _with_z(_t1_route(), [(0, 3.7), (9999, 3.7)])
    if name == "t2":
        return [(-p[0], *p[1:])
                for p in _with_z(_t1_route(), [(0, 3.7), (9999, 3.7)])]
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
    for name in ("lm", "um", "t1", "t2"):
        dia = CABLE_D[name]
        path = Spline(*route_points(name))
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(dia / 2.0)
        cutters.append(sweep(section, path=path))
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
