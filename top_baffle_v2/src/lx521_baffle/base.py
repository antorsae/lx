"""LX521.4 top baffle, modified for face-to-face Dayton ND25FW-4 tweeters.

Geometry source: exact vector paths extracted from the 1:1 drawing
"plano top baffle con anidados V2.pdf" (lines + cubic beziers, mm).
Coordinate frame: X = horizontal from baffle centerline, Y = vertical from
the baffle bottom edge, Z = through-thickness (front face at Z=THICKNESS).

Cross-checked against the drawing's printed dimensions:
  overall 304.80 x 468.31, bottom 152.40, top opening 98.34 / 121.84,
  scallop cut from circles D78.50 / D102.11 centered ~(0, 483.05),
  step 8.7, prong tip 12.7, shelf 24.11.

Thickness defaults to 18.3 mm, the stock LX521.4 top-baffle thickness used
throughout this repo (lx521_l22mg_baffle/geometry.py).
"""

from __future__ import annotations

import os
from math import cos, radians, sin

from build123d import (
    Bezier,
    Circle,
    Cylinder,
    Face,
    Line,
    Plane,
    Polyline,
    Pos,
    ThreePointArc,
    Wire,
    extrude,
    loft,
    make_face,
)

from .geom import smoothstep01 as _smoothstep

THICKNESS_MM = 18.3

# Tweeter-pair clamp holes: drawing shows D4.0 at (+/-32.56, 451.24); printed
# at 4.4 mm so an M4 machine screw passes an FDM part without reaming.
TWEETER_HOLE_D_MM = 4.4
TWEETER_HOLE_XY = [(-32.56, 451.24), (32.56, 451.24)]

# Lower-mid cutout from the drawing; upper-mid aligned to the stock LX521.4
# baffle ("lx521 baffle metric.dxf", UM at 368.3 with LM at 203.2): in the
# LM-aligned frame the UM center is 366.081, i.e. 5.857 below the V2 drawing.
L22_CUTOUT = (0.0, 200.981, 190.0)
UM_CUTOUT = (0.0, 366.081, 82.0)
UM_ALIGN_DY_MM = 371.938 - 366.081  # drawing UM minus stock-aligned UM

# Variant A's tweeter section (and the perimeter above the neck) moves down
# with the UM so the T/UM relationship of the V2 drawing is preserved.
TWEETER_DROP_A_MM = UM_ALIGN_DY_MM

# Blind driver-mounting pilots, cut PILOT_DEPTH_MM into the front face
# (z = THICKNESS_MM) only -- the rear face stays closed.
PILOT_DEPTH_MM = THICKNESS_MM / 2.0  # 9.15

# Upper mid: production SEAS MU10RB-SL (H1658-04), 4-hole D89.5
# pattern. Mounted with brass M3 x 3 heat-set inserts in D4.6 x 4.0
# bores. M3 screws pass the flange holes; the inboard wall to the D82
# cutout is 1.45. The reference/acoustic mesh intentionally omits the
# electrical terminals, so terminal fit is represented separately by
# lx521_baffle.um_fit and still requires a hardware trial.
UM_PILOT_D_MM = 4.6
UM_PILOT_PCD_MM = 89.5
UM_PILOT_ANGLES_DEG = (58.0, 148.0, 238.0, 328.0)  # rotated:
# The terminal carrier is clocked BETWEEN the lower screws: the allowed
# gap is 238..328 deg and its exact midpoint axis is 283 deg.
# The left pair also clears the proud-family shared tweeter lane.
UM_TERMINAL_GAP_DEG = (238.0, 328.0)
UM_TERMINAL_CLOCK_DEG = 283.0
UM_PILOT_DEPTH_MM = 4.0

# Lower mid: production SEAS U22REX/P-SL -- 6 x D5.0 flange holes
# (D8.8 head recess) on D209.5 pitch, cross-checked against the
# E0022_W22EX001 reference STEP used to establish the mounting template.
# Mounted with BRASS
# HEAT-SET inserts M5 x 5.8 long x D6.3 OD (D7.1 maximum head): the
# unchanged total bore is D6.4 x 6.8 (5.8 insert + 1.0 melt room), with a
# D6.5 x 2.0 entry relief at the insertion face.  The extra 0.05 mm radial
# lead exists only at the mouth; the remaining 4.8 mm stays D6.4.  M5 screws
# pass the D5.0 flange holes natively; heads
# seat in the D8.8 recesses.  Stock/Slim and Obi-Wan share the same
# 0/60/...300-degree clock.  That puts two inserts in ``piece_bottom``,
# keeps the LM terminal axis vertical, and removes the former +Y bore from
# the narrow seam-C land.  Floor z=11.5; every bore remains regression-gated
# against the complete front-half duct set.
M5_INSERT_BODY_D_MM = 6.4
M5_INSERT_ENTRY_D_MM = 6.5
M5_INSERT_ENTRY_DEPTH_MM = 2.0
L22_PILOT_D_MM = M5_INSERT_BODY_D_MM
L22_PILOT_PCD_MM = 209.5
L22_PILOT_ANGLES_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
L22_PILOT_DEPTH_MM = 6.8

# STAND_FOOT: piece_bottom carries a fused stand foot (see the split
# module). With the foot there is no bridge, so the four bridge screw
# pass-throughs (and their countersinks) are omitted, and the cable
# ducts route down through the foot instead of breaking the rear face.
# Driven by the LX_STAND_FOOT env var (default ON) so the Makefile can
# emit both artifact sets (build/floor_stand/ and build/no_floor_stand/).
STAND_FOOT = os.environ.get("LX_STAND_FOOT", "1") != "0"

# Four bridge mounting points, per the measured bridge (40.0 x 50.0
# pattern, rows 20/70 above the bottom edge). Fastener: M5 machine
# screw from the BRIDGE (behind) into a BRASS HEAT-SET insert -- the
# SAME insert as the W22/LM (bore D_L22_PILOT x L22_PILOT_DEPTH), but
# bored BLIND from the REAR face (z=0), i.e. the OPPOSITE side from the
# front-mounted driver inserts. no-stand only (the stand foot replaces
# the bridge). See BRIDGE_INSERT_* below (aliased to the L22 pilot).
BRIDGE_INSERT_D_MM = L22_PILOT_D_MM        # same insert as the W22/LM
BRIDGE_INSERT_DEPTH_MM = L22_PILOT_DEPTH_MM
BRIDGE_HOLE_XY = [
    (-20.0, 20.0),
    (20.0, 20.0),
    (-20.0, 70.0),
    (20.0, 70.0),
]

# Bottom-corner pass-through holes: centers 10 mm above the bottom edge and
# 10 mm inboard of the bottom corners (+/-76.2, 0). Sized for an M5 machine
# screw to THREAD-FORM through the full 18.3 mm (D4.5 ~ 60% engagement;
# pre-run the screw once to form the threads).
# Selectable: set CORNER_HOLES_ENABLED = True to cut them (default OFF).
CORNER_HOLES_ENABLED = False
CORNER_HOLE_D_MM = 4.5
CORNER_HOLE_XY = [
    (-66.2, 10.0),
    (66.2, 10.0),
]


def m5_insert_bore_cutter(
    center_xy: tuple[float, float],
    *,
    opening_z: float,
    total_depth: float,
    opening_side: str,
    overshoot: float = 0.20,
):
    """Return the shared stepped M5 heat-set-insert bore cutter.

    ``opening_side='+z'`` serves front/driver-seat insertion: the physical
    mouth is ``opening_z`` and the blind floor is ``opening_z-total_depth``.
    ``opening_side='-z'`` serves rear insertion and reverses that interval.
    In either direction the first 2.0 mm is D6.5 and all remaining depth is
    D6.4.  ``overshoot`` lies outside the physical host and therefore never
    changes the requested total depth.
    """
    depth = float(total_depth)
    extra = float(overshoot)
    if depth < M5_INSERT_ENTRY_DEPTH_MM:
        raise ValueError(
            "M5 insert bore depth must contain the complete entry relief")
    if extra < 0.0:
        raise ValueError("M5 insert bore overshoot cannot be negative")
    x, y = map(float, center_xy)
    mouth = float(opening_z)
    if opening_side == "+z":
        body_z0, body_z1 = mouth - depth, mouth + extra
        entry_z0, entry_z1 = mouth - M5_INSERT_ENTRY_DEPTH_MM, mouth + extra
    elif opening_side == "-z":
        body_z0, body_z1 = mouth - extra, mouth + depth
        entry_z0, entry_z1 = mouth - extra, mouth + M5_INSERT_ENTRY_DEPTH_MM
    else:
        raise ValueError("M5 insert bore opening_side must be '+z' or '-z'")

    def cylinder(diameter: float, z0: float, z1: float):
        return Pos(x, y, (z0 + z1) / 2.0) * Cylinder(
            diameter / 2.0, z1 - z0)

    body = cylinder(M5_INSERT_BODY_D_MM, body_z0, body_z1)
    entry = cylinder(M5_INSERT_ENTRY_D_MM, entry_z0, entry_z1)
    cutter = body.fuse(entry).clean()
    cutter.label = "m5_heat_set_bore_D6p5x2_then_D6p4"
    return cutter

# Outer boundary, exactly as drawn. ("L", start, end) straight segments and
# ("C", start, ctrl1, ctrl2, end) cubic beziers; consecutive collinear top-edge
# segments from the source path are merged.
OUTLINE = [
    ("C", (-57.149, 371.938), (-57.149, 372.638), (-57.135, 373.334), (-57.110, 374.027)),
    ("L", (-57.110, 374.027), (-57.048, 409.062)),
    ("L", (-57.048, 409.062), (-60.918, 439.046)),
    ("L", (-60.918, 439.046), (-36.811, 439.046)),
    ("L", (-36.811, 439.046), (-36.811, 447.736)),
    ("C", (-36.811, 447.736), (-42.416, 453.459), (-46.699, 460.483), (-49.161, 468.314)),
    ("L", (-49.161, 468.314), (-36.468, 468.314)),
    ("C", (-36.468, 468.314), (-35.847, 466.854), (-35.182, 465.235), (-34.388, 463.861)),
    ("C", (-34.388, 463.861), (-27.556, 452.048), (-14.742, 443.808), (0.001, 443.804)),
    ("C", (0.001, 443.804), (14.946, 443.800), (27.970, 452.113), (34.681, 464.367)),
    ("C", (34.681, 464.367), (35.293, 465.485), (35.977, 467.123), (36.483, 468.314)),
    ("L", (36.483, 468.314), (49.177, 468.314)),
    ("C", (49.177, 468.314), (46.712, 460.477), (42.425, 453.449), (36.813, 447.723)),
    ("L", (36.813, 447.723), (36.813, 439.046)),
    ("L", (36.813, 439.046), (60.921, 439.046)),
    ("L", (60.921, 439.046), (57.046, 409.062)),
    ("L", (57.046, 409.062), (57.111, 374.071)),
    ("C", (57.111, 374.071), (57.137, 373.363), (57.151, 372.652), (57.151, 371.938)),
    ("L", (57.151, 371.938), (57.151, 305.981)),
    ("L", (57.151, 305.981), (152.401, 256.120)),
    ("L", (152.401, 256.120), (76.201, 0.0)),
    ("L", (76.201, 0.0), (-76.199, 0.0)),
    ("L", (-76.199, 0.0), (-152.401, 256.155)),
    ("L", (-152.401, 256.155), (-57.151, 306.016)),
    ("C", (-57.151, 306.016), (-57.151, 328.002), (-57.149, 349.952), (-57.149, 371.938)),
]


def _aligned_outline():
    """Variant-A outline with the flare/shelf/tweeter perimeter (everything
    above the neck, y >= 409) lowered by UM_ALIGN_DY_MM; the neck verticals
    absorb the shift."""
    outline = []
    for seg in OUTLINE:
        pts = [
            (x, y - UM_ALIGN_DY_MM) if y >= 409.0 else (x, y)
            for x, y in seg[1:]
        ]
        outline.append((seg[0], *pts))
    return outline


def outline_face(outline=OUTLINE) -> Face:
    edges = []
    for seg in outline:
        if seg[0] == "L":
            edges.append(Line(seg[1], seg[2]).edge())
        elif seg[0] == "A":
            edges.append(ThreePointArc(*seg[1:]).edge())
        else:
            edges.append(Bezier(*seg[1:]).edge())
    return make_face(Wire(edges))


def baffle_face(outline=OUTLINE, tweeter_drop_mm: float = 0.0) -> Face:
    face = outline_face(outline)
    for cx, cy, dia in (L22_CUTOUT, UM_CUTOUT):
        face -= Pos(cx, cy) * Circle(dia / 2.0)
    # (bridge mounts are now blind REAR heat-set bores, cut in the 3D
    # solid -- no through-hole in the 2D face)
    if CORNER_HOLES_ENABLED:
        for cx, cy in CORNER_HOLE_XY:
            face -= Pos(cx, cy) * Circle(CORNER_HOLE_D_MM / 2.0)
    for cx, cy in TWEETER_HOLE_XY:
        face -= Pos(cx, cy - tweeter_drop_mm) * Circle(TWEETER_HOLE_D_MM / 2.0)
    return face


def _pilot_centers(center_xy, pcd: float, angles_deg) -> list[tuple[float, float]]:
    cx, cy = center_xy
    r = pcd / 2.0
    return [(cx + r * cos(radians(a)), cy + r * sin(radians(a))) for a in angles_deg]


# --- Tweeter-crescent rear taper ------------------------------------------
# The horseshoe that carries the face-to-face tweeter pair thins from the
# REAR (the front face stays a full plane): 18.3 at the scallop bottom,
# 4.0 at the clamp pass-throughs, feathering to ~0.4 at the horn tips.
# Thickness follows the arc angle about the scallop center through two C1
# smoothstep segments (zero slope at the bottom blend AND at the clamp
# ring, so the rear tweeter faceplate gets a locally flat seat). Cut as a
# loft of radial sections: full depth from r0 out to the knee r_k (which
# covers the D102.11 arc joint at r~51.05), then a smoothstep radial fade
# of the cut back to 0 by r_f. The fade carries the SAME taper into the
# crescent's outboard neighbours (the A-comp top shoulders and B1 wings,
# which sit just outside the arc), so their rear faces stay flush across
# the detachable interface; the fade completes at r_f=62.0, inside the vertical
# flank's top corner (r~62.4), so the outboard flank/top edges -- and
# the chamfer walls at larger r -- return to full 18.3 depth.
CRESCENT_SCALLOP_CY = 483.05     # scallop center, un-dropped drawing frame
CRESCENT_TAPER_T_HOLES_MM = 4.0
CRESCENT_TAPER_T_TIP_MM = 0.4    # feather floor: protects the front face
CRESCENT_TAPER_THETA_DEG = (-90.0, -44.33, -16.68)  # bottom, clamps, tips
CRESCENT_TAPER_R_MM = (36.0, 51.5, 62.0)  # inner, knee, fade-out
# Thin V1/V1L hosts retain a broad symmetric thickness shelf through the
# detachable upper-magnet band.  At least 9.30 mm of material leaves the
# common Z=15.10 captive land 0.45 mm behind the sculpted rear surface.  The
# shelf ends before the clamp-seat control point and is applied on both sides
# of the crescent; it is not a station-local pad or pocket-shaped cue.
CRESCENT_TAPER_MAGNET_BAND_THETA_DEG = -64.0
CRESCENT_TAPER_MAGNET_BAND_MIN_THICKNESS_MM = 9.30


def _crescent_taper_depth(theta_deg: float,
                          front_mm: float = THICKNESS_MM) -> float:
    """Rear cut depth at arc angle theta (0 at the bottom of the scallop,
    front-4 at the clamp holes, front-0.4 past the horn tips). front_mm
    lets thinned variants (V1) keep the SAME seat/tip thicknesses on a
    thinner slab."""
    th_b, th_h, th_e = CRESCENT_TAPER_THETA_DEG
    d_h = front_mm - CRESCENT_TAPER_T_HOLES_MM
    d_e = front_mm - CRESCENT_TAPER_T_TIP_MM
    if theta_deg <= th_h:
        original = d_h * _smoothstep(
            (theta_deg - th_b) / (th_h - th_b))
        th_m = CRESCENT_TAPER_MAGNET_BAND_THETA_DEG
        magnet_limit = max(
            0.0,
            front_mm - CRESCENT_TAPER_MAGNET_BAND_MIN_THICKNESS_MM,
        )
        original_at_magnet = d_h * _smoothstep(
            (th_m - th_b) / (th_h - th_b))
        if original_at_magnet <= magnet_limit + 1.0e-9:
            return original
        if theta_deg <= th_m:
            return magnet_limit * _smoothstep(
                (theta_deg - th_b) / (th_m - th_b))
        return magnet_limit + (d_h - magnet_limit) * _smoothstep(
            (theta_deg - th_m) / (th_h - th_m))
    return d_h + (d_e - d_h) * _smoothstep((theta_deg - th_h) / (th_e - th_h))


def _crescent_taper_cutters(tweeter_drop_mm: float,
                            front_mm: float = THICKNESS_MM,
                            rear_mm: float = 0.0):
    cy = CRESCENT_SCALLOP_CY - tweeter_drop_mm
    r0, r_k, r_f = CRESCENT_TAPER_R_MM
    n_fade = 8
    cutters = []
    for sign in (1.0, -1.0):
        sections = []
        for i in range(20):
            th = -87.0 + 4.0 * i  # -87 deg .. -11 deg
            d = _crescent_taper_depth(th, front_mm - rear_mm)
            # section profile in (r, z): floor at z=-0.5, cut depth = d for
            # r0..r_k, then smoothstep-faded d->0 across r_k..r_f
            pts = [(r0, rear_mm - 0.5), (r_f, rear_mm - 0.5)]
            for k in range(n_fade + 1):
                rr = r_f - (r_f - r_k) * k / n_fade   # r_f .. r_k
                frac = (rr - r_k) / (r_f - r_k)        # 1 .. 0
                pts.append((rr, rear_mm + d * (1.0 - _smoothstep(frac))))
            pts.append((r0, rear_mm + d))
            pts.append((r0, rear_mm - 0.5))
            a = radians(th)
            pl = Plane(
                origin=(0.0, cy, 0.0),
                x_dir=(sign * cos(a), sin(a), 0.0),
                z_dir=(sin(a), -sign * cos(a), 0.0),
            )
            sections.append(pl * make_face(Wire(Polyline(*pts).edges())))
        cutters.append(loft(sections))
    return cutters


def baffle_solid(outline=OUTLINE, tweeter_drop_mm: float = 0.0,
                 crescent_front_mm: float = THICKNESS_MM,
                 crescent_rear_mm: float = 0.0):
    part = extrude(Plane.XY * baffle_face(outline, tweeter_drop_mm), amount=THICKNESS_MM)
    pilots = [
        (UM_CUTOUT[:2], UM_PILOT_PCD_MM, UM_PILOT_ANGLES_DEG, UM_PILOT_D_MM,
         UM_PILOT_DEPTH_MM),
    ]
    for center_xy, pcd, angles, dia, depth in pilots:
        for px, py in _pilot_centers(center_xy, pcd, angles):
            part -= Pos(px, py, THICKNESS_MM - depth / 2.0) * Cylinder(
                dia / 2.0, depth
            )
    for px, py in _pilot_centers(
            L22_CUTOUT[:2], L22_PILOT_PCD_MM, L22_PILOT_ANGLES_DEG):
        part -= m5_insert_bore_cutter(
            (px, py),
            opening_z=THICKNESS_MM,
            total_depth=L22_PILOT_DEPTH_MM,
            opening_side="+z",
        )
    for cutter in _crescent_taper_cutters(tweeter_drop_mm,
                                          crescent_front_mm,
                                          crescent_rear_mm):
        part -= cutter
    # Bridge mounting: M5 heat-set inserts (same bore as the W22/LM),
    # BLIND from the REAR face (z=0..depth) -- the opposite side from the
    # front-mounted driver inserts. The bridge screws in from behind.
    for cx, cy in (BRIDGE_HOLE_XY if not STAND_FOOT else []):
        part -= m5_insert_bore_cutter(
            (cx, cy),
            opening_z=0.0,
            total_depth=BRIDGE_INSERT_DEPTH_MM,
            opening_side="-z",
        )
    return part


OUTLINE_ALIGNED = _aligned_outline()


def gen_step():
    part = baffle_solid(OUTLINE_ALIGNED, tweeter_drop_mm=TWEETER_DROP_A_MM)
    part.label = "lx521_4_top_baffle"
    return part
