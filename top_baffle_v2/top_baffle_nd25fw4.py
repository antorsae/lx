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

# Upper mid: Scan-Speak 10F/8424G00 -- 4 x D3.80 flange holes on pitch
# D89.5 (datasheet 10f-8424g00.pdf). Mounted with BRASS HEAT-SET
# inserts M3 x 3 long x D5 OD: bore D4.6 x 4.0 (3.0 insert + 1.0 melt
# room; 4 x ~250 N pull-out vs the 0.43 kg 10F). M3 screws pass the
# D3.8 flange holes natively; inboard wall to the D82 is 1.45. The
# SHORT bores (floor z=14.3) let the vase-side ducts run in the FRONT
# half (T lanes z=10.7, roof 12.6: 1.7 clear of the floors where the
# lanes cross the ring) -- which is what allows variant V1 to thin the
# vase from the REAR and mount FRONT-FLUSH with the LM section.
UM_PILOT_D_MM = 4.6
UM_PILOT_PCD_MM = 89.5
UM_PILOT_ANGLES_DEG = (58.0, 148.0, 238.0, 328.0)  # rotated:
# the LEFT pair clears the shared T duct's flank lane and notch dive;
# the right pair faces no ducts at all (round-4 layout)
UM_PILOT_DEPTH_MM = 4.0

# Lower mid: SEAS W22EX001 -- 6 x D5.0 flange holes (D8.8 head recess) on
# pitch D209.5 (measured from E0022_W22EX001.stp). Mounted with BRASS
# HEAT-SET inserts M5 x 5.8 long x D6.3 OD: bore D6.4 x 6.8 (5.8
# insert + 1.0 melt room; manufacturer's recommended hole), set with a
# soldering iron. M5 screws pass the D5.0 flange holes natively; heads
# seat in the D8.8 recesses. Pattern aligned VERTICALLY (30/90/...330
# deg). Floor z=11.5; the ring is plan-clear of every front-half duct
# (LM keeps 3.05 to the 270-deg bore, seam C clears the 90-deg bore by
# 2.25) -- checked by the suite.
L22_PILOT_D_MM = 6.4
L22_PILOT_PCD_MM = 209.5
L22_PILOT_ANGLES_DEG = (30.0, 90.0, 150.0, 210.0, 270.0, 330.0)
L22_PILOT_DEPTH_MM = 6.8

# STAND_FOOT: piece_bottom carries a fused stand foot (see the split
# module). With the foot there is no bridge, so the four bridge screw
# pass-throughs (and their countersinks) are omitted, and the cable
# ducts route down through the foot instead of breaking the rear face.
# Driven by the LX_STAND_FOOT env var (default ON) so the Makefile can
# emit both artifact sets (floor_stand/ and no_floor_stand/).
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
# the glue joint; the fade completes at r_f=62.0, inside the vertical
# flank's top corner (r~62.4), so the outboard flank/top edges -- and
# the chamfer walls at larger r -- return to full 18.3 depth.
CRESCENT_SCALLOP_CY = 483.05     # scallop center, un-dropped drawing frame
CRESCENT_TAPER_T_HOLES_MM = 4.0
CRESCENT_TAPER_T_TIP_MM = 0.4    # feather floor: protects the front face
CRESCENT_TAPER_THETA_DEG = (-90.0, -44.33, -16.68)  # bottom, clamps, tips
CRESCENT_TAPER_R_MM = (36.0, 51.5, 62.0)  # inner, knee, fade-out


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 3.0 * t * t - 2.0 * t * t * t


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
        return d_h * _smoothstep((theta_deg - th_b) / (th_h - th_b))
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
        (L22_CUTOUT[:2], L22_PILOT_PCD_MM, L22_PILOT_ANGLES_DEG, L22_PILOT_D_MM,
         L22_PILOT_DEPTH_MM),
    ]
    for center_xy, pcd, angles, dia, depth in pilots:
        for px, py in _pilot_centers(center_xy, pcd, angles):
            part -= Pos(px, py, THICKNESS_MM - depth / 2.0) * Cylinder(
                dia / 2.0, depth
            )
    for cutter in _crescent_taper_cutters(tweeter_drop_mm,
                                          crescent_front_mm,
                                          crescent_rear_mm):
        part -= cutter
    # Bridge mounting: M5 heat-set inserts (same bore as the W22/LM),
    # BLIND from the REAR face (z=0..depth) -- the opposite side from the
    # front-mounted driver inserts. The bridge screws in from behind.
    for cx, cy in (BRIDGE_HOLE_XY if not STAND_FOOT else []):
        part -= Pos(cx, cy, BRIDGE_INSERT_DEPTH_MM / 2.0) * Cylinder(
            BRIDGE_INSERT_D_MM / 2.0, BRIDGE_INSERT_DEPTH_MM)
    return part


OUTLINE_ALIGNED = _aligned_outline()


def gen_step():
    part = baffle_solid(OUTLINE_ALIGNED, tweeter_drop_mm=TWEETER_DROP_A_MM)
    part.label = "lx521_4_top_baffle_nd25fw4"
    return part
