"""Internal cable ducts -- THREE routes (round-5 "flush-ready" layout).

All mains are strictly planar (inline z-eased splines break OCC
booleans) EXCEPT the TS vase stretch, which dips as a flattened oval
under the MU10RB-SL flange seat (V1LF flush mounting); everything
runs in the FRONT half of the plate so thin variants (V1 vase, V1L
bottom/mids) share the FRONT plane and mount flush. Round-5 keeps
every duct clear of the two flange-recess rings (see the flush
module): the U22 ring is plan r<=110.6 about (0, LM_Y) with its seat
at z=12.3; the MU10 ring r<=49.3 about (0, UM_Y), seat z=14.3.

  MAINS (common to both stand-foot states):
    LM (blue):  2 x 2.5 mm^2 twisted pair, O8.2 at z=12.55; straight
        plan line drifting x=-8.6 -> -10.5, ending at y=85 -- clear of
        the U22 recess ring (surface-to-rim 1.75) -- where it exits
        straight back.
    UM (green): twisted 2 x 2.0 mm^2 pair (~O7.0 bundle), O7.8 at
        z=12.55 END-TO-END: R26 fan fillet, tangent onto ONE arc
        r=119.5 -- OUTSIDE the W22 pilot ring (radial 7.65) and the
        U22 recess rim (radial 5.0) -- riding the arc all the way to
        +49 deg, exiting straight back at (78.4, 291.2) under the
        right chamfer (3.2 from the outline). The old diagonal-to-
        seam-B tail lived entirely UNDER the U22 flange ring and died
        with round-5.
    TS (gold): BOTH tweeter pairs (2 x 2xAWG24, side by side ~5.2)
        share ONE duct up the LEFT flank: O6.0 at z=11.5 across the
        bottom and the r=116.5 arc (outside the W22 ring AND recess
        rim), then -- through the vase lane, crest transition and
        notch corridor, which all run under the MU10 flange seat --
        a flattened W6.6 x H4.4 oval at zc=10.45 (TS_OVAL: 1.65 under
        the seat, 1.45 over the thin rear plane), morphing back to
        O6.0 z=11.5 for the head-on pierce of the D78.5 scallop rim
        at (-3.3, ~430). Both tweeter cables emerge there and dress
        to their tweeters through the open scallop void. The 10F
        pilot pattern (58,148,238,328) keeps the left pair clear of
        the lane and the dive; the oval additionally threads the
        (148, 238) pilot keep-outs, which now reach z=10.3.

  ENTRIES (state-dependent): with STAND_FOOT the mains continue down
    the plate into packed foot lanes (FOOT_LANES: 90-deg R14 vertical-
    plane elbows, then rearward to the connector channel's step face
    at z=-99). Without it, straight oblique ramp bores (BIG_RAMPS /
    T_RAMP) pierce the rear face inside the support plate's D20
    window: LM/UM breakouts at (-/+5.2, 60.5); twin T ramp ovals at
    (+3.8, 52.2) / (-3.1, 52.7) into the strip feeders (t1f z=3.7 /
    t2f z=9.5; far lips up to ~1.4 past the window rim -- the floppy
    AWG24 pairs duck in).

  EXITS (both states): straight-back bores (EXIT_RAMPS) from the mains
    to the REAR face, opening BEHIND each driver so the cable comes out
    the back and plugs into the rear terminals -- NOT into the cutout
    cavity (which fouled the basket on assembly). LM exits below its
    cutout at y=85 (just below the U22 flange edge at y~90.7 -- the
    recess ring pushed it 13 lower); UM exits at the end of its arc
    under the right chamfer (in C7 that bore emerges through the
    bevel slope -- functional, rear-facing). The TS main exits head-on
    through the scallop rim into the face-to-face tweeter pole gap.

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

from build123d import (
    Circle,
    Cylinder,
    Plane,
    Polyline,
    Pos,
    Rot,
    Spline,
    Wire,
    loft,
    make_face,
    sweep,
)

from top_baffle_nd25fw4 import STAND_FOOT

LM_Y = 200.981
UM_Y = 366.081

# Duct plane depth per route -- ALL in the front half (front-datum):
# LM O8.2 spans z 8.45..16.65, UM 8.65..16.45, TS 8.5..14.5 (round) /
# 8.25..12.65 (vase oval). Floors stay >=1.6 above the thin-family
# rear plane (z=6.8) except the oval's 1.45 -- the price of passing
# under the MU10 flange seat (z=14.3) with 1.65 to spare above.
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

# TS vase stretch: flattened oval under the MU10RB-SL flange seat
# (V1LF recess floor z=14.3). W6.6 x H4.4 at zc=10.45: 1.65 under the
# seat, 1.45 over the thin rear plane (z=6.8), and W/2=3.3 threads the
# notch corridor (>=1.84 to the D82 rim) and the vase lane (1.81 to
# the outline). The two AWG24 pairs ride side by side (~5.2 x 2.6).
# Morphs are y-parameterized (the vase path is y-monotone there):
# round O6.0 z=11.5 below y=316 and above y=425.5, full oval between
# y=330 and y=417.5, linear in the morph bands. The tube is fully
# dropped before it comes within 1.6 of the ring rim on both sides
# (rim crossings at plan dist 49.3 about (0, UM_Y): y~334 / ~415.5),
# and is round again before piercing the scallop rim (~429).
TS_OVAL = {"w2": 3.3, "h2": 2.2, "zc": 10.45,
           "y_in": (316.0, 330.0), "y_out": (417.5, 425.5)}


def ts_section(y):
    """(w2, h2, zc) of the TS duct section at height y: the oval drop
    under the MU10 flange seat, round O6.0 z=11.5 elsewhere. The duct
    cutter and route_points z both derive from this single law."""
    rw2, rh2, rzc = CABLE_D["ts"] / 2.0, CABLE_D["ts"] / 2.0, DUCT_Z["ts"]
    (yi0, yi1), (yo0, yo1) = TS_OVAL["y_in"], TS_OVAL["y_out"]
    ow2, oh2, ozc = TS_OVAL["w2"], TS_OVAL["h2"], TS_OVAL["zc"]
    if y <= yi0 or y >= yo1:
        return rw2, rh2, rzc
    if yi1 <= y <= yo0:
        return ow2, oh2, ozc
    if y < yi1:                       # entry morph
        f = (y - yi0) / (yi1 - yi0)
    else:                             # exit morph (f runs 1 -> 0)
        f = (yo1 - y) / (yo1 - yo0)
    return (rw2 + f * (ow2 - rw2), rh2 + f * (oh2 - rh2),
            rzc + f * (ozc - rzc))


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
    """The shared tweeter duct: left tangent+arc r=116.5 (outside the
    W22 pilot ring AND the U22 recess rim, surface-to-rim 2.9), the
    left vase flank lane, crest transition (bowed 1 mm outboard of
    the deepened 148-deg MU10 pilot), notch-corridor dive, single
    scallop pierce at (-3.3, ~430). Section per ts_section(): O6.0
    z=11.5 outside the vase, W6.6 x H4.4 oval at zc=10.45 under the
    MU10 flange seat."""
    # straight run to the arc tangent point: keep the knots SPARSE --
    # long runs of collinear spline knots degenerate OCC's pipe frame
    line = [(-16.8, 58.8), (-48.9, 88.0)]
    return (
        line
        + _arc(116.5, [-133.99, -142, -152, -162, -172, -182, -192,
                       -202, -212, -222, -232, -242])
        + [(-46.5, 307.0), (-38.6, 311.0), (-34.5, 314.0),
           (-33.4, 317.5), (-33.3, 320.5), (-34.1, 324.0),
           (-35.3, 327.5), (-36.6, 330.8)]
        + [(-(_wall_x(y) - LANE_OFFSET), y)
           for y in (338, 346, 354, 362, 370, 378, 384)]
        + [(-53.3, 386.5), (-52.2, 389.5), (-50.2, 392.0),
           (-47.6, 393.3), (-45.0, 394.5), (-42.7, 395.7), (-40.7, 397.0)]
        + [(-38.0, 399.0), (-31.0, 402.0), (-25.6, 404.8), (-20.3, 407.5),
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
# Driver-cable exits: straight-back bores from the main to the REAR
# face (z 12.55 -> -2), opening BEHIND the driver so the cable comes
# out the back and plugs into the rear terminals -- NOT into the driver
# cutout cavity (the old exits dumped the cable at z~12 inside the hole,
# fouling the basket). Round-5: both mains now END clear of the U22
# flange-recess ring, so the exits moved with them. LM opens at y=85,
# ~6 below the U22 flange edge on the rear face (in V1L that is mid-
# ramp; the bore pierces the sloped rear). UM opens at the top of its
# arc, (78.4, 291.2), under the right chamfer -- in C7 the bore
# emerges through the rear bevel slope (rear-facing, functional). The
# MU10 cable dresses from there up the rear face to its terminals.
EXIT_RAMPS = {
    "lm": ((-10.5, 84.0, 12.55), (-10.5, 84.0, -2.0), 9.0),
    "um": ((78.39, 291.17, 12.55), (78.39, 291.17, -2.0), 7.8),
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
        # to x=-10.5 past the 270-deg W22 insert bore (plan-clear 3.0),
        # ending at y=85: the old y=98 tail ran UNDER the U22 flange
        # recess ring (surface 8 mm inside the rim); at y=85 the duct
        # surface clears the rim by 1.75. Straight-back exit there.
        lead = ([(-5.45, 30.0, 12.55), (-6.2, 38.0, 12.55),
                 (-7.3, 48.0, 12.55), (-7.7, 58.0, 12.55)]
                if STAND_FOOT else [])
        # knots stay SPARSE (~9 mm) -- tighter near-collinear knots
        # degenerate OCC's pipe frame (MakeSolid fails). y=84: the O9
        # exit bore (fatter than the duct) needs 1.6 to the recess rim
        return lead + [(-8.0, 68.0, 12.55), (-9.3, 76.0, 12.55),
                       (-10.5, 84.0, 12.55)]
    if name == "um":
        # O7.8 at z=12.55 END-TO-END: fan, tangent onto r=119.5 OUTSIDE
        # the W22 pilot ring (radial 7.65) and the U22 recess rim
        # (surface-to-rim 5.0), riding the SAME arc to +49 deg where it
        # exits straight back (EXIT_RAMPS) 3.2 inside the outline. The
        # old R50-fillet/diagonal/window-bend tail to (11.6, 309.1) ran
        # its whole length UNDER the U22 flange ring (plan dist
        # 102..109 vs rim 110.6) -- removed in round-5. The right side
        # carries only this route.
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
            + _arc(119.5, [-56.29, -44, -32, -20, -8, 4, 12, 20,
                           28, 36, 43, 49]),
            [(0, 12.55), (9999, 12.55)])
    if name == "ts":
        return [(x, y, ts_section(y)[2]) for x, y in _ts_route()]
    raise ValueError(name)





def _ts_cutter():
    """The TS duct as ONE ruled loft: vertical elliptical sections
    every ~2.4 mm along the interpolating spline of the route, sized
    and positioned by ts_section(y). Outside the vase every section
    is the O6.0 circle at z=11.5, so the solid matches the old pipe
    sweep there to within the chord sag (<0.03 on the r=116.5 arc).
    ONE solid because joining coaxial tubes (round sweep + oval loft)
    leaves near-tangent sliver faces at the join -- the exposed-duct/
    open-STL-edge failure class. Sections are 24-gons (inscribed:
    bore ~1% under nominal -- wall margins stay conservative)."""
    path = Spline(*route_points("ts"))
    n_st = max(6, int(path.length / 2.4))
    secs = []
    for i in range(n_st + 1):
        t = i / n_st
        px, py, _pz = tuple(path @ t)
        tx, ty, _tz = tuple(path % t)
        nrm = math.hypot(tx, ty)
        tx, ty = tx / nrm, ty / nrm
        w2, h2, zc = ts_section(py)
        # plane y-axis = z_dir x x_dir = +Z for any plan tangent, so
        # the profile's v coordinate is GLOBAL z (C7 station idiom)
        pl = Plane(origin=(px, py, 0.0), x_dir=(-ty, tx, 0.0),
                   z_dir=(tx, ty, 0.0))
        pts = [(w2 * math.cos(a), zc + h2 * math.sin(a))
               for a in (2.0 * math.pi * k / 24.0 for k in range(24))]
        pts.append(pts[0])
        secs.append(pl * make_face(Wire(Polyline(*pts).edges())))
    return loft(secs, ruled=True)


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


# Duct crossings of the glue seams: (x, y, z, duct r). A vertical
# relief cylinder (r+1.0, bore-height) at each crossing widens the
# mouth LATERALLY by 1 mm on both mating faces -- a funnel for the
# cross-seam fishing hand-off. Lateral only: z skins are untouched
# (the planar routing already guarantees vertical alignment). Nearest
# grown key pocket stays >=3.7 clear (seam B TS crossing binds).
SEAM_CROSSINGS = (
    (87.88, 120.0, 12.55, 3.9),    # UM x seam A (r=119.5 arc)
    (-83.75, 120.0, 11.5, 3.0),    # TS x seam A (r=116.5 arc)
    (-33.89, 315.95, 11.5, 3.0),   # TS x seam B (UM no longer reaches B)
)


def seam_relief_cutters():
    # squashed SPHERES (plan r+1.0, z-semi r-0.4): smooth everywhere,
    # so no wall/cap grazing against the bore -- flat-capped cylinders
    # left near-tangent slivers the mesher dropped (open STL edges
    # that read as an exposed duct; the c7 mid_right bug), and the
    # foot-mode spline's <0.05 lateral shift made it intermittent.
    # Poles stay inside the bore (z-semi < r); z skins untouched.
    from build123d import Sphere, scale
    return [Pos(x, y, z) * scale(Sphere(r + 1.0),
                                 (1.0, 1.0, (r - 0.4) / (r + 1.0)))
            for x, y, z, r in SEAM_CROSSINGS]


def cable_cutters():
    cutters = []
    cutters += seam_relief_cutters()
    for name in ("lm", "um"):
        dia = CABLE_D[name]
        path = Spline(*route_points(name))
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(dia / 2.0)
        cutters.append(sweep(section, path=path))
    cutters.append(_ts_cutter())
    # the two O3.8 pair-feeders in the strip + the z-step bore
    for fname in ("t1f", "t2f"):
        path = Spline(*route_points(fname))
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(1.9)
        cutters.append(sweep(section, path=path))
    cutters.append(_entry_ramp(TS_STEP[0], TS_STEP[1], 6.8))
    # straight-back exits to the rear face (behind each driver)
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
