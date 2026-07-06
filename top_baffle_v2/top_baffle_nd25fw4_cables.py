"""Internal cable ducts -- THREE routes (round-4 "front-datum" layout).

All mains are strictly planar (inline z-eased splines break OCC
booleans) and ALL of them now run in the FRONT half of the plate, so
thin variants (V1 vase, V1L bottom/mids) can share the FRONT plane
with the LM section and mount flush:

  MAINS (planar spline sweeps, common to both stand-foot states):
    LM (blue):  2 x 2.5 mm^2 twisted pair, O8.2 at z=12.55; straight
        plan line drifting x=-8.6 -> -10.5 past the 270-deg W22 pilot
        (plan-clear 3.0).
    UM (green): twisted 2 x 2.0 mm^2 pair (~O7.0 bundle), O7.8 at
        z=12.55 END-TO-END (no step): R26 fan fillet, tangent onto ONE
        arc r=119.5 -- OUTSIDE the W22 pilot ring (radial 7.65) --
        then a straight diagonal to the seam-B crossing at x~7 and the
        vase tail to the D82 exit. The RIGHT side carries only this
        route now.
    TS (gold): BOTH tweeter pairs (2 x 2xAWG24, side by side ~5.2)
        share ONE O6.0 duct at z=11.5 END-TO-END up the LEFT flank:
        feeder across the bottom strip (the T1 pair crosses over from
        the right entry), tangent onto the r=114 arc (outside the W22
        ring), the left vase flank lane 5.1 inside the walls, a crest
        transition, and the threaded dive through the notch corridor
        (D82 rim vs chamfer edge -- O6.0 is the LARGEST duct that
        corridor admits) to a SINGLE head-on pierce of the D78.5
        scallop rim at (-3.3, ~430). Both tweeter cables emerge there
        and dress to their tweeters through the open scallop void.
        The 10F pilot pattern is rotated to (58,148,238,328) so the
        left pair clears the lane and the dive (the right pair faces
        no ducts at all).

  ENTRIES (state-dependent): with STAND_FOOT the mains continue down
    the plate into packed foot lanes (FOOT_LANES: 90-deg R14 vertical-
    plane elbows, then rearward to the connector channel's step face
    at z=-99). Without it, straight oblique ramp bores (BIG_RAMPS /
    T_RAMP) pierce the rear face inside the support plate's D20
    window: LM/UM breakouts at (-/+5.2, 60.5); twin T ramp ovals at
    (+3.8, 52.2) / (-3.1, 52.7) into the strip feeders (t1f z=3.7 /
    t2f z=9.5; far lips up to ~1.4 past the window rim -- the floppy
    AWG24 pairs duck in).

  EXITS (both states): straight near-level oblique bores (EXIT_RAMPS)
    from the planar mains through the driver-cutout walls; openings
    center at z~12, through the basket spoke-window zone of each
    driver. The TS main exits head-on through the scallop rim (the
    tweeter pole-gap void cannot be occluded).

The ducts cross the glue seams; cables are laid/fished through each
piece's open segments during assembly. Clearances (duct-duct, pilots,
magnet pockets, seam keys, notch corridor, foot-lane webs) are checked
by test_clearances.py (make check).

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

# Duct plane depth per route -- ALL in the front half (front-datum):
# the O8.2 LM window is 6.85..18.25 (the uniform-11.5 binder), UM 8.65..16.45,
# TS 8.5..14.5. Floors stay >=1.6 above the thin-family rear plane
# (z=6.8 -- V1 vase and V1L mids alike).
DUCT_Z = {"lm": 12.55, "um": 12.55, "ts": 11.5}
LANE_OFFSET = 5.11  # TS lane center this far inside the vase wall lines

# LM carries 2 x 2.5 mm^2 (twisted ~O7.8) -> O8.2. UM carries a twisted
# 2 x 2.0 mm^2 pair (~O7.0) -> O7.8 (2 x 2.5 no longer fits). TS
# carries BOTH tweeter pairs side by side (~5.2 across) -> O6.0, the
# largest bore the notch corridor between the D82 rim and the vase
# chamfer admits.
CABLE_D = {"lm": 8.2, "um": 7.8, "ts": 6.0}  # LM O8.2: the
# uniform-11.5 window binder; ~0.4 over the twisted 2x2.5 bundle --
# snug fishing (leader + lube), chosen over dropping to 2x2.0 wire


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


def _wall_x(y):
    """|x| of the vase outline at height y (flare, then chamfer)."""
    if y <= 391.709:
        return 38.113 + 0.29752 * (y - 315.947)
    return 60.654 - 1.9108 * (y - 391.709)


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
    swp = a1 - a0
    if cw and swp > 0:
        swp -= 2 * math.pi
    if not cw and swp < 0:
        swp += 2 * math.pi
    n = max(2, int(abs(math.degrees(swp)) / step_deg))
    for i in range(n + 1):
        a = a0 + swp * i / n
        pts.append((ctr[0] + r * math.cos(a), ctr[1] + r * math.sin(a)))
    run2 = math.dist(t2, p1)
    for i in range(1, max(1, int(run2 / step_mm))):
        s = i * run2 / max(1, int(run2 / step_mm))
        pts.append((t2[0] + d1[0] * s, t2[1] + d1[1] * s))
    return pts


# The T pairs cross the bottom strip below the mains (t1f at z=3.7,
# t2f at z=9.5) -- the strip stays full-depth (18.3) in every variant
# (it carries the foot/bridge), and down there the feeders pass the
# LM/UM columns with >=7.3 (3D), which no front-half z could do. A
# short O6.8 z-step bore rises to the shared main west of the LM
# column.
TS_STEP = ((-14.0, 55.0, 3.7), (-16.8, 58.8, 11.5))  # O6.8 bore
# (both pair-channels merge into it; their caps bury inside)


def _t1_feeder():
    """T1's pair: right entry across the strip to the z-step (z=3.7)."""
    # foot lead rounds the down-the-plate corner at R~7.5 (was a sharp
    # R2.3 elbow -- caught by test_route_smoothness)
    lead = ([(13.9, 27.0), (14.0, 34.0), (13.9, 41.0), (13.2, 45.5),
             (11.5, 48.9), (8.9, 51.0)]
            if STAND_FOOT else [(13.9, 50.0)])
    # runs 1.9 past the z-step mouth so its end cap is buried inside
    # the TS_STEP bore (coincident caps poison OCC booleans)
    return lead + [(5.8, 51.9), (0.0, 51.8), (-6.0, 52.8),
                   (-10.0, 53.6), (-14.0, 55.0), (-15.8, 55.5)]


def _t2_feeder():
    """T2's pair: left entry to the same z-step (z=9.5)."""
    # T2 always runs at z=9.5 -- 5.8 of z-separation from t1f (z=3.7)
    # wherever their plans shadow or converge (coplanar near-tangent
    # pipes poison OCC booleans) -- and merges into the O6.8 step at
    # its upper half. End knots bury their caps in the step, staggered
    # from t1f's.
    if STAND_FOOT:
        return [(-14.9, 27.0), (-14.9, 48.0), (-15.2, 53.0),
                (-15.4, 56.2)]
    return [(-6.3, 51.7), (-10.0, 53.2), (-13.5, 54.8), (-15.5, 55.9)]


def _ts_route():
    """The shared O6.0 tweeter duct (planar z=11.5): left tangent+arc
    r=114, the left vase flank lane, crest transition, notch-corridor
    dive, single scallop pierce at (-3.3, ~430)."""
    # straight run to the arc tangent point: keep the knots SPARSE --
    # long runs of collinear spline knots degenerate OCC's pipe frame
    line = [(-16.8, 58.8), (-48.0, 88.9)]
    return (
        line
        + _arc(114.0, [-133.99, -142, -152, -162, -172, -182, -192,
                       -202, -212, -222, -232, -242])
        + [(-45.0, 306.0), (-38.0, 310.5), (-34.5, 314.0),
           (-33.4, 317.5), (-33.3, 320.5), (-34.1, 324.0),
           (-35.3, 327.5), (-36.6, 330.8)]
        + [(-(_wall_x(y) - LANE_OFFSET), y)
           for y in (338, 346, 354, 362, 370, 378, 384)]
        + [(-53.3, 386.5), (-52.2, 389.5), (-50.2, 392.0),
           (-47.6, 393.3), (-45.0, 394.5), (-42.5, 395.6), (-40.5, 396.9)]
        + [(-36.0, 399.3), (-31.0, 402.0), (-25.6, 404.8), (-20.3, 407.5),
           (-15.5, 410.0), (-10.4, 412.8), (-5.5, 417.5), (-4.3, 420.5),
           (-3.5, 424.0), (-3.3, 427.0), (-3.3, 430.0), (-3.4, 433.0)]
    )


# Without the stand foot the baffle bolts to the stock support via the
# four pass-throughs, and all cables must pass a D20 hole in the
# support plate: center (0, 60). Packing: LM/UM breakouts side by side
# (steep ramps crossing z=0 at (-/+5.2, 60.5)); twin O4.6 T ramps
# breaking out at (+3.8, 52.2) / (-3.1, 52.7) into the strip feeders,
# far lips up to ~1.4 past the rim (the floppy AWG24 pairs duck in).
SUPPORT_WINDOW = (0.0, 60.0, 20.0)  # cx, cy, D of the support-plate hole
T_RAMP = ((1.0, 53.4, -6.4), (6.0, 51.3, 5.0))  # right pair
T_RAMP_L = ((-1.0, 53.4, -6.4), (-6.3, 51.7, 9.5))  # left pair,
# lancing the raised t2 feeder (z=9.5); breakout ~(-3.1, 52.7)


# LM/UM entries: straight oblique ramps (p0 behind the rear face ->
# tip inside the planar main), O0.8 over their ducts.
BIG_RAMPS = {
    "lm": ((-4.6, 56.0, -6.4), (-8.0, 68.5, 12.55)),
    "um": ((2.0, 56.0, -6.4), (8.0, 60.0, 12.55)),  # 69-deg ramp
    # lancing the fan arc; breakout (4.0, 57.4) inside the D20 window
}
# Near-level exits through the driver-cutout walls; openings center at
# z~12, in each basket's spoke-window zone (with the shallow front-half
# mains a rear-quarter opening is no longer reachable; align the
# opening with a basket window at assembly). TS exits head-on through
# the scallop rim -- no ramp needed.
EXIT_RAMPS = {
    "lm": ((-9.9, 86.0, 12.55), (-10.6, 119.0, 12.0), 9.0),
    "um": ((5.3, 320.5, 12.55), (2.95, 332.4, 12.0), 8.6),
}


# STAND_FOOT lanes: (x, duct z, run height y_f, elbow radius, bore D).
# Four lanes persist (one per NL8 pair); the two T lanes feed the
# shared duct via the feeder/lead. Webs come from Dx alone.
FOOT_LANES = {
    "lm": (-5.45, 12.55, 10.5, 14.0, 9.0),
    "um": (5.4, 12.55, 10.5, 14.0, 8.6),
    "t1": (13.9, 3.7, 5.5, 14.0, 4.6),
    "t2": (-14.9, 9.5, 5.5, 14.0, 4.6),  # outboard: at z=9.5
    # the old x lands 7.1 from the LM lead (need 7.65)
}


def route_points(name):
    """Planar duct centerline (z=DUCT_Z[name]) for the three routes."""
    if name == "t1f":
        return _with_z(_t1_feeder(), [(0, 3.7), (9999, 3.7)])
    if name == "t2f":
        return _with_z(_t2_feeder(), [(0, 9.5), (9999, 9.5)])
    if name == "lm":
        # Planar z=12.55; the line drifts from the x=-8.6 entry column
        # to x=-10.5 past the 270-deg W22 insert bore (plan-clear 3.0).
        # The exit is a near-level oblique bore (EXIT_RAMPS).
        lead = ([(-5.45, 30.0, 12.55), (-6.2, 38.0, 12.55),
                 (-7.3, 48.0, 12.55), (-7.7, 58.0, 12.55)]
                if STAND_FOOT else [])
        return lead + [(-8.0, 68.0, 12.55), (-9.0, 78.0, 12.55),
                       (-10.0, 90.0, 12.55), (-10.5, 98.0, 12.55),
                       (-10.5, 103.0, 12.55)]
    if name == "um":
        # O7.8 at z=12.55 END-TO-END: fan, tangent onto r=119.5 OUTSIDE
        # the W22 pilot ring, arc, diagonal to the seam-B crossing at
        # x~7 (clear of the right B-pocket by 1.5), vase tail to the
        # D82 exit bore (EXIT_RAMPS). The right side has no other duct.
        # The fan fillet is EXPLICIT: the r=119.5 tangent line passes
        # through the old (4, 60) anchor, which degenerates
        # _line_arc_line into a backward stub (spline cusp -> inverted
        # pipe shell -> poisoned booleans; the 2026-07-06 OOM).
        fan = [(30.0 + 26.0 * math.cos(math.radians(a)),
                46.1 + 26.0 * math.sin(math.radians(a)))
               for a in (180, 172, 164, 156, 148, 140, 132, 124)]
        return _with_z(
            ([(5.4, 30.0), (4.9, 37.0), (4.35, 43.0)]
             if STAND_FOOT else [])
            + fan + [(40.9, 84.6)]
            + _arc(119.5, [-56.29, -44, -32, -20, -8, 4, 12, 20])
            # R50 fillet off the arc onto a straight diagonal TANGENT
            # to a r=13.5 keep-out about the 30-deg pilot (smooth by
            # construction; no kink); then one R~12 window bend between
            # the 90-deg pilot and the right B-key (cx=28), and the
            # vase tail. Guarded by test_route_smoothness.
            + [(108.01, 251.03), (102.45, 258.63), (95.57, 265.04),
               (87.60, 270.06)]
            + [(61.76, 283.11), (35.92, 296.16)]
            + [(18.5, 304.7), (14.5, 306.9), (11.6, 309.1), (9.7, 311.7),
               (8.5, 314.6), (7.4, 317.4), (6.2, 320.0), (4.9, 322.0)],
            [(0, 12.55), (9999, 12.55)])
    if name == "ts":
        return _with_z(_ts_route(), [(0, 11.5), (9999, 11.5)])
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
    for name in ("lm", "um", "ts"):
        dia = CABLE_D[name]
        path = Spline(*route_points(name))
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(dia / 2.0)
        cutters.append(sweep(section, path=path))
    # the two O3.8 pair-feeders in the strip + the z-step bore
    for fname in ("t1f", "t2f"):
        path = Spline(*route_points(fname))
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(1.9)
        cutters.append(sweep(section, path=path))
    cutters.append(_entry_ramp(TS_STEP[0], TS_STEP[1], 6.8))
    # near-level exits through the driver-cutout walls
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
    # twin T entry ramps lancing into the O3.8 strip feeders
    cutters.append(_entry_ramp(T_RAMP[0], T_RAMP[1], 4.6))
    cutters.append(_entry_ramp(T_RAMP_L[0], T_RAMP_L[1], 4.6))
    # LM/UM entries: steep straight ramps into the D20 support window
    for name, (p0, p1) in BIG_RAMPS.items():
        cutters.append(_entry_ramp(p0, p1, CABLE_D[name] + 0.8))
    return cutters
