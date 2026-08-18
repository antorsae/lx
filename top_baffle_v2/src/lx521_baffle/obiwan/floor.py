"""Integral floor-state stem, foot, connector and buried continuations.

The floor-state LM carrier owns this geometry.  There is no mating support,
yoke, rail set or support fastener.  The outer floor envelope is a solid
W64 rectangle from the baffle front to the retained NL8 rear plane; only the
three buried cable lumens and the necessary connector service scoop are
subtracted.

This module intentionally does not import ``obiwan`` so the
carrier can add the floor body without an import cycle.
"""

from __future__ import annotations

import math

from build123d import (
    Bezier,
    Box,
    Circle,
    Cylinder,
    Face,
    Line,
    Part,
    Plane,
    Polyline,
    Pos,
    RectangleRounded,
    Rot,
    ThreePointArc,
    Wire,
    extrude,
    fillet,
    loft,
    make_face,
    sweep,
)

from ..base import L22_CUTOUT, STAND_FOOT, THICKNESS_MM
from ..cables import (
    LM_DUCT_OUT_REAR_Z_MM,
    LM_EXIT_BEND_R_MM,
    lm_exit_handoff_points,
    lm_exit_handoff_spec,
)
from ..flush import PAD_FACE_Z
from ..floor_bend import (
    BEND_MIN_CENTERLINE_RADIUS_MM,
    BEND_VERTICAL_HANDLE_MM,
    FUSION_OVERLAP_MM as FLOOR_BEND_FUSION_OVERLAP_MM,
    bend_facts,
    bent_wall_prism,
    canonical_lane_controls,
    sampled_minimum_radius,
)
from .floor_strength import (
    FLOOR_Y_MM,
    FOOT_FRONT_Z_MM,
    FOOT_HEIGHT_MM,
    FOOT_REAR_Z_MM,
    FOOT_WIDTH_MM,
    LM_AXIS_Y_MM,
)


# --- NL8 boss, service trough and snap lid (concept sign-off 2026-08-17)
# The rear is no longer a flat W64 panel: a 38.4-square mating face
# (NL8MPRXX flange 38.16 + 0.12/side lip, all four corners r4.2) crowns a
# body whose crest holds level over the connector barrel (true penetration
# 25.85 mm, vendor-STEP contact plane 7.15 behind the D-face tip) and
# falls as one cosine to the foot.  Window and service cavity merge into
# ONE trough ending in a flat duct-entry wall at Z=-77 -- exactly 73 mm
# from the face, the toe of the rise -- so the Faston flags get ~47 mm of
# open run.  A same-solid lid clips over the trough on four hidden
# cantilever fingers.
BOSS_TOP_W_MM = 38.4
NL8_CENTER_Y_MM = BOSS_TOP_W_MM / 2.0          # square face, derived
BOSS_FLANGE_T_MM = 5.6                         # insert seat 4.0 + 1.6 roof
BOSS_CREST_HOLD_Z_MM = -120.0
BOSS_FALL_SPAN_MM = 44.0
BOSS_WIDTH_EASE_END_Z_MM = -56.0               # approved trumpet span
# The loft runs 7.85 past the bend's horizontal tangent for a deep,
# OCC-stable fusion; its last section's bottom edge is raised to 1.2 so
# the end cap hides above the bend's rising underside (0.76 there)
# instead of leaving a downward sliver.
BOSS_END_Z_MM = -58.0
BOSS_END_LIFT_MM = 1.2
BOSS_FILLET_FADE_SPAN_MM = 14.0
PANEL_INNER_Z_MM = FOOT_REAR_Z_MM + BOSS_FLANGE_T_MM
PANEL_T_MM = PANEL_INNER_Z_MM - FOOT_REAR_Z_MM
PANEL_H_MM = BOSS_TOP_W_MM
NL8_CUTOUT_D_MM = 31.0
NL8_SCREW_D_MM = 4.6            # M3 heat-set insert bore, 4.0 deep
NL8_SCREW_PITCH_MM = 29.2
NL8_INSERT_L_MM = 4.0

TROUGH_HALF_W_MM = 16.55
TROUGH_FLOOR_Y_MM = 3.6
TROUGH_END_WALL_Z_MM = -77.0
TROUGH_CORNER_R_MM = 2.0
TROUGH_SPRING_Z_MM = -124.5
TROUGH_CROWN_Z_MM = -141.0
TROUGH_R_SPRING_MM = 8.0
TROUGH_R_CROWN_MM = 4.0

LID_T_MM = 1.8                                 # 3 beads at 0.6
LID_CLEARANCE_MM = 0.2
LID_STEP_RING_OFF_MM = 1.0                     # rebate in the crest roof
LID_REAR_OVERLAP_OFF_MM = 0.8
LID_PAD_SPANS_Z_MM = ((-111.0, -105.0), (-93.0, -87.0))
LID_CLIP_SPANS_Z_MM = ((-120.0, -112.5), (-86.5, -79.5))
LID_BLOCK_SPANS_X_MM = ((-7.5, -5.0), (5.0, 7.5))
LID_HOOK_DROP_MM = 6.0                         # hook top below the seat
LID_POCKET_PRELOAD_MM = 0.05

# The enclosed cavity is now exactly the barrel chamber: flush with the
# trough walls, under a 3.6 roof, ending where the open trough takes over.
SERVICE_CAVITY_Z_MM = (PANEL_INNER_Z_MM, BOSS_CREST_HOLD_Z_MM)
SERVICE_CAVITY_X_MM = (-TROUGH_HALF_W_MM, TROUGH_HALF_W_MM)
SERVICE_CAVITY_Y_MM = (3.6, 34.8)

# A full-depth W64 stem is the governing printed load member.  The soft
# shoulders enter the lower LM cap at the D190 tangent without entering the
# acoustic opening.  The rear face remains at z=0 rather than growing behind
# the approved concept envelope.
STEM_Z_MM = (0.0, THICKNESS_MM)
STEM_HALF_WIDTH_MM = FOOT_WIDTH_MM / 2.0
STEM_SHOULDER_HALF_WIDTH_MM = 58.0
STEM_SHOULDER_START_Y_MM = 68.0
STEM_TOP_Y_MM = LM_AXIS_Y_MM - L22_CUTOUT[2] / 2.0
STEM_SHOULDER_SAMPLES = 32
FLOOR_BEND_VERTICAL_TANGENT_Y_MM = bend_facts()[
    "vertical_tangent_xyz_mm"][1]
FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM = bend_facts()[
    "horizontal_tangent_xyz_mm"][2]
FLOOR_BEND_UPRIGHT_START_Y_MM = (
    FLOOR_BEND_VERTICAL_TANGENT_Y_MM - FLOOR_BEND_FUSION_OVERLAP_MM)
FLOOR_BEND_REAR_FLAT_END_Z_MM = (
    FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM + FLOOR_BEND_FUSION_OVERLAP_MM)

FLOOR_LANE_BEND_R_MM = BEND_MIN_CENTERLINE_RADIUS_MM
FLOOR_LANE_SERVICE_START_Z_MM = TROUGH_END_WALL_Z_MM - 1.0
# The body-only UM/T sweep stops before the annular feed.  After fusion, the
# globally phased owner cutter reaches 2.0 mm backward through this temporary
# solid bridge, yielding 1.2 mm of final lumen overlap.  This avoids duplicate
# coincident sweeps at the thin feed wall without reaching into Option B's
# convex transition.  A short forward continuation is retained only in the
# dependency-light installed-centerline drawing.
FLOOR_LANE_INSTALLED_PREVIEW_FORWARD_MM = 3.0
FLOOR_LANE_PREFUSION_HANDOFF_GAP_MM = 0.8
FLOOR_ROUTE_OWNER_BACKREACH_MM = 2.0
FLOOR_LANE_EFFECTIVE_OVERLAP_MM = (
    FLOOR_ROUTE_OWNER_BACKREACH_MM
    - FLOOR_LANE_PREFUSION_HANDOFF_GAP_MM)
FLOOR_LANE_UM_END_HANDLE_MM = 35.0
FLOOR_LANE_T_END_HANDLE_MM = 32.0
FLOOR_LM_EXIT_HANDOFF = lm_exit_handoff_spec(
    12.55, STEM_Z_MM[0], LM_DUCT_OUT_REAR_Z_MM)
# UM/T need clearance behind their complete printed cover envelopes, not just
# their nominal lumens.  That clearance is now internal: a 0.45-mm rear skin
# closes the former visible lower-stem mouths while keeping the same robust
# Boolean margin at the hidden handoff plane.
FLOOR_FEED_MOUTH_SHELL_MM = 0.8
FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM = 0.30
FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM = 0.05
FLOOR_REAR_FACE_SKIN_MM = 0.45
FLOOR_FEED_MOUTH_RELIEF_Z_MM = (FLOOR_REAR_FACE_SKIN_MM, PAD_FACE_Z)
# The floor-state T bundle enters from the service cavity, so it can use a
# left-side handoff that stays wholly clear of the new direct LM continuation.
# This is intentionally independent of the no-floor stock/slim rear entry.
FLOOR_T_ROUTE_FEED_XY = (-26.0, 82.0)
FLOOR_T_ROUTE_FEED_BEARING_DEG = 116.0
FLOOR_LANE_SPECS = {
    "lm": {
        "x_mm": 0.0,
        "floor_y_mm": 10.5,
        "stem_z_mm": 12.55,
        "diameter_mm": 9.0,
        # The floor lane reaches the shared stock/slim rear-face mouth through
        # the common R14 handoff.  ``mouth_xyz_mm`` is the installed datum;
        # ``feed_xyz_mm`` is only the tangent cutter overtravel outside the
        # printed rear face.
        "mouth_xyz_mm": FLOOR_LM_EXIT_HANDOFF["face"],
        "feed_xyz_mm": FLOOR_LM_EXIT_HANDOFF["rear_end"],
        "handoff_mode": "buried_r14_rear_exit",
    },
    "um": {
        "x_mm": 12.0,
        "floor_y_mm": 10.5,
        "stem_z_mm": 12.55,
        "diameter_mm": 8.2,
        "feed_xyz_mm": (8.0, 82.0, PAD_FACE_Z),
        "feed_bearing_deg": 65.0,
        "handoff_mode": "buried_route_overlap",
    },
    "t": {
        "x_mm": -12.0,
        # y=7.5 keeps the complete D6 opening inside the service cavity's
        # y>=4 boundary.  The old y=5.5 station clipped the connector cap.
        "floor_y_mm": 7.5,
        "stem_z_mm": 6.20,
        "diameter_mm": 6.0,
        "feed_xyz_mm": (
            FLOOR_T_ROUTE_FEED_XY[0],
            FLOOR_T_ROUTE_FEED_XY[1],
            PAD_FACE_Z,
        ),
        "feed_bearing_deg": FLOOR_T_ROUTE_FEED_BEARING_DEG,
        "handoff_mode": "buried_route_overlap",
    },
}


def _require_guarded_build() -> None:
    import run_memory_guarded as memory_guard
    memory_guard.require_guarded_build(
        "integral Obi-Wan floor geometry requires run_memory_guarded.py")


def _plan_face(points):
    return Face(Wire(Polyline(*points).edges()))


def _quadratic(p0, p1, p2, count):
    out = []
    for index in range(count + 1):
        u = index / count
        out.append((
            (1.0 - u) ** 2 * p0[0]
            + 2.0 * (1.0 - u) * u * p1[0]
            + u ** 2 * p2[0],
            (1.0 - u) ** 2 * p0[1]
            + 2.0 * (1.0 - u) * u * p1[1]
            + u ** 2 * p2[1],
        ))
    return out


def integral_stem_plan_points():
    """Closed XY outline with symmetric quadratic shoulder integration."""
    right = _quadratic(
        (STEM_HALF_WIDTH_MM, STEM_SHOULDER_START_Y_MM),
        (STEM_HALF_WIDTH_MM, STEM_TOP_Y_MM - 3.0),
        (STEM_SHOULDER_HALF_WIDTH_MM, STEM_TOP_Y_MM),
        STEM_SHOULDER_SAMPLES)
    left = [(-x, y) for x, y in reversed(right)]
    return (
        (-STEM_HALF_WIDTH_MM, FLOOR_Y_MM),
        (STEM_HALF_WIDTH_MM, FLOOR_Y_MM),
        (STEM_HALF_WIDTH_MM, STEM_SHOULDER_START_Y_MM),
        *right[1:],
        *left,
        (-STEM_HALF_WIDTH_MM, FLOOR_Y_MM),
    )


def _stem_prism():
    face = _plan_face(integral_stem_plan_points())
    full = Pos(0.0, 0.0, STEM_Z_MM[0]) * extrude(
        face, amount=STEM_Z_MM[1] - STEM_Z_MM[0])
    clip_height = STEM_TOP_Y_MM - FLOOR_BEND_UPRIGHT_START_Y_MM + 2.0
    clip = Pos(
        0.0,
        (FLOOR_BEND_UPRIGHT_START_Y_MM + STEM_TOP_Y_MM) / 2.0,
        sum(STEM_Z_MM) / 2.0,
    ) * Box(
        2.0 * STEM_SHOULDER_HALF_WIDTH_MM + 4.0,
        clip_height,
        STEM_Z_MM[1] - STEM_Z_MM[0] + 2.0,
    )
    return (full & clip).clean()


# --- boss body ---------------------------------------------------------

def _boss_ease(u: float) -> float:
    return (1.0 - math.cos(math.pi * min(1.0, max(0.0, u)))) / 2.0


def boss_height_mm(z: float) -> float:
    if z <= BOSS_CREST_HOLD_Z_MM:
        return BOSS_TOP_W_MM
    return FOOT_HEIGHT_MM + (BOSS_TOP_W_MM - FOOT_HEIGHT_MM) * (
        1.0 - _boss_ease((z - BOSS_CREST_HOLD_Z_MM) / BOSS_FALL_SPAN_MM))


def boss_width_mm(z: float) -> float:
    return BOSS_TOP_W_MM + (FOOT_WIDTH_MM - BOSS_TOP_W_MM) * _boss_ease(
        (z - FOOT_REAR_Z_MM)
        / (BOSS_WIDTH_EASE_END_Z_MM - FOOT_REAR_Z_MM))


def _boss_section(z: float, y0: float = 0.0):
    w, h = boss_width_mm(z), boss_height_mm(z)
    fade = _boss_ease((BOSS_END_Z_MM - z) / BOSS_FILLET_FADE_SPAN_MM)
    r = fade * min(
        4.2 + 3.8 * _boss_ease((z - FOOT_REAR_Z_MM) / 40.0),
        h / 3.0, w / 4.0)
    half = w / 2.0
    wire = (Polyline((half, y0), (-half, y0), (-half, h))
            + Line((-half, h), (half, h)) + Line((half, h), (half, y0)))
    face = make_face(wire)
    # all four corners alike: the NL8 flange is rounded on all four
    corners = [v for v in face.vertices() if abs(abs(v.X) - half) < 1e-6
               and (abs(v.Y - h) < 1e-6 or abs(v.Y - y0) < 1e-6)]
    if r > 0.2 and corners:
        face = fillet(corners, r)
    return Plane.XY.offset(z) * face


def _boss_prism():
    """The uncut boss loft; also the blank the lid is carved from."""
    sections = [_boss_section(FOOT_REAR_Z_MM + 3.0 * i) for i in range(31)]
    sections.append(_boss_section(BOSS_END_Z_MM, y0=BOSS_END_LIFT_MM))
    return loft(sections, ruled=True)


# --- trough plan -------------------------------------------------------

def _trough_plan_points():
    u_half = TROUGH_HALF_W_MM
    t_off = TROUGH_R_SPRING_MM / math.tan(math.radians(67.5))
    sp_side = TROUGH_SPRING_Z_MM + t_off
    sp_flank = t_off / math.sqrt(2.0)
    crown_c = TROUGH_CROWN_Z_MM + TROUGH_R_CROWN_MM / math.sin(
        math.radians(45.0))
    lf0 = (-(u_half - sp_flank), TROUGH_SPRING_Z_MM - sp_flank)
    lf1 = (-TROUGH_R_CROWN_MM / math.sqrt(2.0),
           TROUGH_CROWN_Z_MM + TROUGH_R_CROWN_MM / math.sqrt(2.0))
    pts = [lf0, lf1]
    for i in range(1, 16):
        a = math.radians(225.0 + 90.0 * i / 16.0)
        pts.append((TROUGH_R_CROWN_MM * math.cos(a),
                    crown_c + TROUGH_R_CROWN_MM * math.sin(a)))
    pts += [(-lf1[0], lf1[1]), (-lf0[0], lf0[1])]
    for i in range(1, 12):
        a = math.radians(-45.0 + 45.0 * i / 12.0)
        pts.append((u_half - TROUGH_R_SPRING_MM
                    + TROUGH_R_SPRING_MM * math.cos(a),
                    sp_side + TROUGH_R_SPRING_MM * math.sin(a)))
    pts += [(u_half, sp_side),
            (u_half, TROUGH_END_WALL_Z_MM - TROUGH_CORNER_R_MM)]
    for i in range(1, 8):
        a = math.radians(90.0 * i / 8.0)
        pts.append((u_half - TROUGH_CORNER_R_MM
                    + TROUGH_CORNER_R_MM * math.cos(a),
                    TROUGH_END_WALL_Z_MM - TROUGH_CORNER_R_MM
                    + TROUGH_CORNER_R_MM * math.sin(a)))
    pts += [(u_half - TROUGH_CORNER_R_MM, TROUGH_END_WALL_Z_MM),
            (-(u_half - TROUGH_CORNER_R_MM), TROUGH_END_WALL_Z_MM)]
    for i in range(1, 8):
        a = math.radians(90.0 + 90.0 * i / 8.0)
        pts.append((-(u_half - TROUGH_CORNER_R_MM)
                    + TROUGH_CORNER_R_MM * math.cos(a),
                    TROUGH_END_WALL_Z_MM - TROUGH_CORNER_R_MM
                    + TROUGH_CORNER_R_MM * math.sin(a)))
    pts += [(-u_half, TROUGH_END_WALL_Z_MM - TROUGH_CORNER_R_MM),
            (-u_half, sp_side)]
    for i in range(1, 12):
        a = math.radians(180.0 + 45.0 * i / 12.0)
        pts.append((-(u_half - TROUGH_R_SPRING_MM)
                    + TROUGH_R_SPRING_MM * math.cos(a),
                    sp_side + TROUGH_R_SPRING_MM * math.sin(a)))
    pts.append(pts[0])
    clean = [pts[0]]
    for q in pts[1:]:
        if (abs(q[0] - clean[-1][0]) > 5e-4
                or abs(q[1] - clean[-1][1]) > 5e-4):
            clean.append(q)
    if clean[-1] != clean[0]:
        clean.append(clean[0])
    return clean


def _trough_plan_offset(off_rear: float, off_front: float,
                        jog_z: float = -119.0, blend: float = 3.0):
    pts = _trough_plan_points()[:-1]
    n = len(pts)
    area = sum(pts[i][0] * pts[(i + 1) % n][1]
               - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
    sgn = 1.0 if area > 0 else -1.0
    out = []
    for i in range(n):
        px, pz = pts[i]
        ax, az = pts[i - 1]
        bx, bz = pts[(i + 1) % n]
        tx, tz = bx - ax, bz - az
        ln = math.hypot(tx, tz) or 1.0
        nx, nz = sgn * tz / ln, -sgn * tx / ln
        off = off_front + (off_rear - off_front) * _boss_ease(
            (jog_z - pz) / blend)
        out.append((px + off * nx, pz + off * nz))
    out.append(out[0])
    return out


def _trough_prism(outline):
    return (Pos(0.0, 75.0, 0.0) * Rot(90, 0, 0)
            * extrude(make_face(Polyline(outline)), 75.0 - TROUGH_FLOOR_Y_MM))


def lid_seat_y_mm(z: float) -> float:
    return boss_height_mm(z) - LID_T_MM


def _lid_seat_pads():
    pads = []
    for z0, z1 in LID_PAD_SPANS_Z_MM:
        for sx in (-1, 1):
            prof = [(lid_seat_y_mm(z0 + u * (z1 - z0) / 6.0),
                     z0 + u * (z1 - z0) / 6.0) for u in range(7)]
            prof += [(lid_seat_y_mm(z1) - 2.5, z1 - 2.5),
                     (lid_seat_y_mm(z0) - 2.5, z0)]
            face = Plane.YZ.offset(
                min(sx * (TROUGH_HALF_W_MM - 2.5),
                    sx * TROUGH_HALF_W_MM)) * make_face(
                Polyline(prof + [prof[0]]))
            pads.append(extrude(face, 2.5) & _trough_prism(
                _trough_plan_points()))
    return pads


def _lid_nose_blocks():
    blocks = []
    bt = lid_seat_y_mm(TROUGH_END_WALL_Z_MM)
    for xa, xb in LID_BLOCK_SPANS_X_MM:
        face = Plane.YZ.offset(xa) * make_face(Polyline([
            (bt, -78.2), (bt, -76.6), (bt - 1.6, -76.6), (bt, -78.2)]))
        blocks.append(extrude(face, xb - xa)
                      & Pos(0, 40.0, -110.0) * Box(40.0, 80.0, 70.0))
    return blocks


def _lid_clip_geo(z0: float, z1: float):
    zc = (z0 + z1) / 2.0
    ys = lid_seat_y_mm(zc)
    return zc, ys, ys - LID_HOOK_DROP_MM


def _lid_pocket_cutters():
    cutters = []
    for z0, z1 in LID_CLIP_SPANS_Z_MM:
        zc, ys, yh = _lid_clip_geo(z0, z1)
        yp = yh - LID_POCKET_PRELOAD_MM
        z0p, z1p = z0 - 0.6, z1 + 0.6
        for sx in (-1, 1):
            face = Plane.YZ.offset(
                min(sx * (TROUGH_HALF_W_MM - 0.05),
                    sx * (TROUGH_HALF_W_MM + 1.55))) * make_face(
                Polyline([(yp, z0p), (yp, z1p), (yp - 3.5, z1p),
                          (yp - 3.5, z0p + 3.5), (yp, z0p)]))
            cutters.append(extrude(face, 1.6))
    return cutters


def _lid_step_cutter():
    ring = (_trough_prism(_trough_plan_offset(LID_STEP_RING_OFF_MM, -0.01))
            - _trough_prism(_trough_plan_points()))
    band = Pos(0.0, 57.4 + (BOSS_TOP_W_MM - LID_T_MM), -110.0) * Box(
        60.0, 114.8, 90.0)
    return ring & band


def _trough_cutter():
    cutter = _trough_prism(_trough_plan_points())
    for keep in (*_lid_seat_pads(), *_lid_nose_blocks()):
        cutter -= keep
    return cutter


def _boss_cavity_cutter():
    cy = NL8_CENTER_Y_MM
    return loft([
        Plane.XY.offset(PANEL_INNER_Z_MM) * Pos(0.0, cy)
        * RectangleRounded(31.4, 31.4, 15.6),
        Plane.XY.offset(PANEL_INNER_Z_MM + 8.0) * Pos(0.0, cy)
        * RectangleRounded(2 * TROUGH_HALF_W_MM,
                           SERVICE_CAVITY_Y_MM[1] - SERVICE_CAVITY_Y_MM[0],
                           6.0),
        Plane.XY.offset(BOSS_CREST_HOLD_Z_MM) * Pos(0.0, cy)
        * RectangleRounded(2 * TROUGH_HALF_W_MM,
                           SERVICE_CAVITY_Y_MM[1] - SERVICE_CAVITY_Y_MM[0],
                           6.0),
    ], ruled=True)


def _boss_panel_cutters():
    cutters = [
        Pos(0.0, NL8_CENTER_Y_MM,
            FOOT_REAR_Z_MM + BOSS_FLANGE_T_MM / 2.0)
        * Cylinder(NL8_CUTOUT_D_MM / 2.0, BOSS_FLANGE_T_MM + 2.0),
    ]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            cutters.append(
                Pos(sx * NL8_SCREW_PITCH_MM / 2.0,
                    NL8_CENTER_Y_MM + sy * NL8_SCREW_PITCH_MM / 2.0,
                    FOOT_REAR_Z_MM + NL8_INSERT_L_MM / 2.0)
                * Cylinder(NL8_SCREW_D_MM / 2.0, NL8_INSERT_L_MM))
    return tuple(cutters)


def floor_service_lid():
    """The snap-in trough lid, carved from the same boss loft."""
    _require_guarded_build()
    if not STAND_FOOT:
        raise RuntimeError("no-floor Obi-Wan has no service lid")
    blank = _boss_prism()
    lid = blank & _trough_prism(_trough_plan_offset(
        LID_REAR_OVERLAP_OFF_MM, -LID_CLEARANCE_MM))
    below = Plane.YZ.offset(-20.0) * make_face(Polyline(
        [(lid_seat_y_mm(-144.0 + i), -144.0 + i) for i in range(0, 70)]
        + [(0.0, -74.0), (0.0, -144.0), (lid_seat_y_mm(-144.0), -144.0)]))
    lid -= extrude(below, 40.0)
    for z0, z1 in LID_PAD_SPANS_Z_MM:
        zc = (z0 + z1) / 2.0
        for sx in (-1, 1):
            lid += Pos(sx * (TROUGH_HALF_W_MM - 0.05),
                       lid_seat_y_mm(zc) + 0.8, zc) * Box(0.4, 1.2, 0.6)
    for z0, z1 in LID_CLIP_SPANS_Z_MM:
        zc, ys, yh = _lid_clip_geo(z0, z1)
        for sx in (-1, 1):
            blade = Plane.YZ.offset(
                min(sx * (TROUGH_HALF_W_MM - 1.4),
                    sx * (TROUGH_HALF_W_MM - 0.2))) * make_face(
                Polyline([(yh - 1.4, z0), (ys + 0.4, z0),
                          (ys + 0.4, z1), (yh - 1.4, z1), (yh - 1.4, z0)]))
            lid += extrude(blade, 1.2)
            hook = Plane.XY.offset(z0 + 0.6) * make_face(Polyline(
                [(sx * (TROUGH_HALF_W_MM - 0.2), yh),
                 (sx * (TROUGH_HALF_W_MM + 0.4), yh - 0.16),
                 (sx * (TROUGH_HALF_W_MM - 0.2), yh - 1.36),
                 (sx * (TROUGH_HALF_W_MM - 0.2), yh)]))
            lid += extrude(hook, (z1 - 1.8) - (z0 + 0.6))
            cut_top = lid_seat_y_mm(z1) - 0.05
            lid -= (Pos(sx * (TROUGH_HALF_W_MM - 0.8),
                        (cut_top + yh - 1.5) / 2.0, z1)
                    * Rot(0, 45, 0)
                    * Box(1.71, cut_top - (yh - 1.5), 1.71))
    lid = lid.clean()
    solids = tuple(lid.solids())
    if (not lid.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "floor service lid must be one valid solid; "
            f"valid={lid.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def integrated_floor_addition():
    """Uncut one-solid floor body to fuse into the floor LM outer blank."""
    _require_guarded_build()
    if not STAND_FOOT:
        raise RuntimeError("no-floor Obi-Wan has no integral floor addition")
    # The boss loft owns the whole rear: its sections carry the foot
    # cross-section (trumpet 38.4 -> W64) from the flange to 5 mm past
    # the bend's horizontal tangent, so no separate flat foot box exists.
    bend = bent_wall_prism(FOOT_WIDTH_MM)
    boss = _boss_prism()
    body = boss.fuse(
        bend, _stem_prism(),
        *_lid_seat_pads(), *_lid_nose_blocks()).clean()
    solids = tuple(body.solids())
    if (not body.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "integral floor addition must be one valid solid; "
            f"valid={body.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def _floor_lane_entry_components(name: str):
    """Connector line plus direct Option-B cubic for one floor trunk."""
    try:
        spec = FLOOR_LANE_SPECS[name]
    except KeyError as exc:
        raise ValueError(name) from exc
    controls = _floor_lane_bezier_points(name)
    line_start = (
        spec["x_mm"],
        spec["floor_y_mm"],
        FLOOR_LANE_SERVICE_START_Z_MM,
    )
    return line_start, controls


def _floor_lane_bezier_points(name: str):
    """G1 cubic from the straight foot lane to its selected handoff."""
    spec = FLOOR_LANE_SPECS[name]
    canonical = canonical_lane_controls(
        spec["x_mm"], spec["floor_y_mm"], spec["stem_z_mm"])
    if name == "lm":
        endpoint = FLOOR_LM_EXIT_HANDOFF["start"]
        tangent = FLOOR_LM_EXIT_HANDOFF["plan_tangent"]
        handle = BEND_VERTICAL_HANDLE_MM
    elif spec["handoff_mode"] == "buried_route_overlap":
        endpoint = spec["feed_xyz_mm"]
        bearing = math.radians(spec["feed_bearing_deg"])
        tangent = (math.cos(bearing), math.sin(bearing), 0.0)
        handle = (
            FLOOR_LANE_UM_END_HANDLE_MM
            if name == "um" else FLOOR_LANE_T_END_HANDLE_MM)
    else:
        raise ValueError(f"unsupported floor handoff for {name}")
    return (
        canonical[0],
        canonical[1],
        tuple(endpoint[index] - handle * tangent[index]
              for index in range(3)),
        tuple(endpoint),
    )


def _floor_lm_rear_port_components():
    """Shared R14 descent through the released lower-ring rear face."""
    handoff = FLOOR_LM_EXIT_HANDOFF
    return (
        handoff["start"],
        handoff["arc_mid"],
        handoff["face"],
        handoff["rear_end"],
    )


def _floor_lm_rear_port_edges():
    exit_start, exit_mid, face, rear_end = (
        _floor_lm_rear_port_components())
    return (
        ThreePointArc(exit_start, exit_mid, face),
        Line(face, rear_end),
    )


def _floor_lane_overlap_end(name: str):
    spec = FLOOR_LANE_SPECS[name]
    feed = spec["feed_xyz_mm"]
    bearing = math.radians(spec["feed_bearing_deg"])
    return (
        feed[0] + FLOOR_LANE_INSTALLED_PREVIEW_FORWARD_MM * math.cos(bearing),
        feed[1] + FLOOR_LANE_INSTALLED_PREVIEW_FORWARD_MM * math.sin(bearing),
        feed[2],
    )


def floor_lane_path(name: str):
    """Floor-body cutter path following the Option-B wall transition.

    UM/T stop on their authoritative cubic 0.8 mm before the feed; the later
    2.0-mm annular owner-cutter backreach creates the final overlap through
    ordinary solid body material.  LM reaches its existing lower-ring R14
    outlet directly from the same long tangent cubic.
    """
    try:
        spec = FLOOR_LANE_SPECS[name]
    except KeyError as exc:
        raise ValueError(name) from exc
    line_start, controls = _floor_lane_entry_components(name)
    edges = [Line(line_start, controls[0])]
    if spec["handoff_mode"] == "buried_route_overlap":
        edges.append(Bezier(*_prefusion_cubic_controls(name)))
    else:
        edges.append(Bezier(*controls))
        edges.extend(_floor_lm_rear_port_edges())
    return Wire(edges)


def _cubic_point(points, u: float):
    p0, p1, p2, p3 = points
    return tuple(
        (1.0 - u) ** 3 * p0[index]
        + 3.0 * (1.0 - u) ** 2 * u * p1[index]
        + 3.0 * (1.0 - u) * u ** 2 * p2[index]
        + u ** 3 * p3[index]
        for index in range(3))


def _left_cubic_controls(points, u: float):
    """Exact De Casteljau controls for the cubic interval [0,u]."""
    p0, p1, p2, p3 = points

    def lerp(left, right):
        return tuple(
            (1.0 - u) * left[index] + u * right[index]
            for index in range(3))

    a = lerp(p0, p1)
    b = lerp(p1, p2)
    c = lerp(p2, p3)
    d = lerp(a, b)
    e = lerp(b, c)
    endpoint = lerp(d, e)
    return p0, a, d, endpoint


def _prefusion_cubic_controls(name: str):
    """Truncate an UM/T cubic at the exact pre-fusion feed setback."""
    cubic = _floor_lane_bezier_points(name)
    feed = cubic[-1]
    target = FLOOR_LANE_PREFUSION_HANDOFF_GAP_MM
    lower = 0.50
    upper = 1.0
    if math.dist(_cubic_point(cubic, lower), feed) <= target:
        raise RuntimeError(
            f"{name} floor-lane cubic is too short for its handoff gap")
    for _index in range(60):
        parameter = 0.5 * (lower + upper)
        if math.dist(_cubic_point(cubic, parameter), feed) > target:
            lower = parameter
        else:
            upper = parameter
    controls = _left_cubic_controls(cubic, 0.5 * (lower + upper))
    endpoint_gap = math.dist(controls[-1], feed)
    if not math.isclose(endpoint_gap, target, abs_tol=1.0e-9):
        raise RuntimeError(
            f"{name} floor-lane handoff gap drifted to {endpoint_gap:.9f}")
    return controls


def floor_lane_control_points(name: str):
    """Dependency-light preview of the installed continuous centerline.

    UM/T include a short forward visual continuation from the unchanged feed;
    the actual body-only cutter stops 0.8 mm early and the annular owner cutter
    supplies the final 1.2-mm backreaching overlap. LM stays buried until its
    final lower-ring R14 turn, then continues through the clean rear port.
    """
    try:
        spec = FLOOR_LANE_SPECS[name]
    except KeyError as exc:
        raise ValueError(name) from exc
    line_start, controls = _floor_lane_entry_components(name)
    points = [line_start, controls[0]]
    if spec["handoff_mode"] == "buried_route_overlap":
        points.extend(
            _cubic_point(controls, index / 64.0)
            for index in range(1, 65))
        points.append(_floor_lane_overlap_end(name))
    else:
        points.extend(
            _cubic_point(controls, index / 64.0)
            for index in range(1, 65))
        points.extend(lm_exit_handoff_points(
            spec["stem_z_mm"], STEM_Z_MM[0], n=32,
            rear_end_z_mm=LM_DUCT_OUT_REAR_Z_MM)[1:])
    return tuple(points)


def _floor_lane_cutter(name: str):
    spec = FLOOR_LANE_SPECS[name]
    path = floor_lane_path(name)
    section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(
        spec["diameter_mm"] / 2.0)
    return sweep(section, path=path)


def _floor_feed_mouth_relief(name: str):
    """Shallow rear counter-relief outside one UM/T cover envelope."""
    spec = FLOOR_LANE_SPECS[name]
    if spec["handoff_mode"] != "buried_route_overlap":
        raise ValueError(f"{name} has no annular-route feed mouth")
    radius = (
        spec["diameter_mm"] / 2.0
        + FLOOR_FEED_MOUTH_SHELL_MM
        + FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM
        + FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM)
    z0, z1 = FLOOR_FEED_MOUTH_RELIEF_Z_MM
    feed = spec["feed_xyz_mm"]
    return Pos(feed[0], feed[1], (z0 + z1) / 2.0) * Cylinder(
        radius, z1 - z0)


def integrated_floor_feature_group(index: int):
    """Build only one bounded cutter group to avoid retaining all sweeps."""
    _require_guarded_build()
    if not STAND_FOOT:
        raise RuntimeError("no-floor Obi-Wan has no integral floor features")
    if index == 0:
        return "connector_service_cavity", (_boss_cavity_cutter(),)
    if index == 1:
        return "connector_panel_holes", _boss_panel_cutters()
    lane_names = ("lm", "um", "t")
    if 2 <= index < 2 + len(lane_names):
        name = lane_names[index - 2]
        cutters = [_floor_lane_cutter(name)]
        if FLOOR_LANE_SPECS[name]["handoff_mode"] == "buried_route_overlap":
            cutters.append(_floor_feed_mouth_relief(name))
        return f"floor_lane_{name}", tuple(cutters)
    if index == 2 + len(lane_names):
        return "service_trough_and_lid_seats", (
            _trough_cutter(), _lid_step_cutter(), *_lid_pocket_cutters())
    raise IndexError(index)


def apply_integrated_floor_feature_group(part, index: int):
    if index < 0 or index >= integrated_floor_feature_group_count():
        raise IndexError(index)
    label, cutters = integrated_floor_feature_group(index)
    for cutter in cutters:
        part -= cutter
    part = part.clean()
    solids = tuple(part.solids())
    if (not part.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            f"{label}: integral floor cutter damaged LM; "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def integrated_floor_feature_group_count() -> int:
    return 6 if STAND_FOOT else 0


def integrated_floor_facts() -> dict:
    """Dependency-light dimensions shared by tests, drawings and manifests."""
    lanes = {}
    for name, spec in FLOOR_LANE_SPECS.items():
        entry_min_radius, entry_min_parameter = sampled_minimum_radius(
            _floor_lane_bezier_points(name), samples=20_000)
        lanes[name] = {
            **spec,
            "bend_radius_mm": entry_min_radius,
            "bend_min_parameter": entry_min_parameter,
            "entry_controls_xyz_mm": _floor_lane_bezier_points(name),
            "service_start_z_mm": FLOOR_LANE_SERVICE_START_Z_MM,
            "route_overlap_mm": (
                FLOOR_LANE_EFFECTIVE_OVERLAP_MM
                if spec["handoff_mode"] == "buried_route_overlap" else 0.0),
            "prefusion_handoff_gap_mm": (
                FLOOR_LANE_PREFUSION_HANDOFF_GAP_MM
                if spec["handoff_mode"] == "buried_route_overlap" else 0.0),
            "owner_cutter_backreach_mm": (
                FLOOR_ROUTE_OWNER_BACKREACH_MM
                if spec["handoff_mode"] == "buried_route_overlap" else 0.0),
            "rear_mouth_relief_radius_mm": (
                spec["diameter_mm"] / 2.0
                + FLOOR_FEED_MOUTH_SHELL_MM
                + FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM
                + FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM
                if spec["handoff_mode"] == "buried_route_overlap" else 0.0),
            "rear_mouth_relief_z_mm": (
                FLOOR_FEED_MOUTH_RELIEF_Z_MM
                if spec["handoff_mode"] == "buried_route_overlap" else None),
            "rear_face_skin_mm": (
                FLOOR_REAR_FACE_SKIN_MM
                if spec["handoff_mode"] == "buried_route_overlap" else None),
            "preview_points": floor_lane_control_points(name),
        }
        if name == "lm":
            lanes[name].update({
                "exit_bend_radius_mm": LM_EXIT_BEND_R_MM,
                "rear_face_mouth_xyz_mm": FLOOR_LM_EXIT_HANDOFF["face"],
                "rear_face_angle_deg_from_normal": (
                    FLOOR_LM_EXIT_HANDOFF[
                        "face_angle_deg_from_rear_normal"]),
                "external_tangent_xyz": (
                    FLOOR_LM_EXIT_HANDOFF["face_tangent"]),
            })
    return {
        "ownership": "floor_core_lm_and_optional_keyed_bottom",
        "separate_floor_support_exists": False,
        "floor_y_mm": FLOOR_Y_MM,
        "lm_axis_y_mm": LM_AXIS_Y_MM,
        "lm_axis_to_floor_mm": LM_AXIS_Y_MM - FLOOR_Y_MM,
        "foot_width_mm": FOOT_WIDTH_MM,
        "foot_height_mm": FOOT_HEIGHT_MM,
        "foot_z_mm": (FOOT_REAR_Z_MM, FOOT_FRONT_Z_MM),
        "stem_z_mm": STEM_Z_MM,
        "stem_top_y_mm": STEM_TOP_Y_MM,
        "stem_shoulder_half_width_mm": STEM_SHOULDER_HALF_WIDTH_MM,
        "root_fillet_r_mm": None,
        "floor_bend": bend_facts(),
        "rear_flat_end_z_mm": FLOOR_BEND_REAR_FLAT_END_Z_MM,
        "upright_start_y_mm": FLOOR_BEND_UPRIGHT_START_Y_MM,
        "panel_z_mm": (FOOT_REAR_Z_MM, PANEL_INNER_Z_MM),
        "panel_height_mm": PANEL_H_MM,
        "boss": {
            "top_w_mm": BOSS_TOP_W_MM,
            "crest_hold_z_mm": BOSS_CREST_HOLD_Z_MM,
            "fall_span_mm": BOSS_FALL_SPAN_MM,
            "flange_t_mm": BOSS_FLANGE_T_MM,
            "insert_bore_d_mm": NL8_SCREW_D_MM,
            "insert_depth_mm": NL8_INSERT_L_MM,
            "trough_half_w_mm": TROUGH_HALF_W_MM,
            "trough_floor_y_mm": TROUGH_FLOOR_Y_MM,
            "duct_entry_wall_z_mm": TROUGH_END_WALL_Z_MM,
            "lid_thickness_mm": LID_T_MM,
            "lid_shutline_clearance_mm": LID_CLEARANCE_MM,
            "lid_pad_spans_z_mm": LID_PAD_SPANS_Z_MM,
            "lid_clip_spans_z_mm": LID_CLIP_SPANS_Z_MM,
            "lid_pocket_preload_mm": LID_POCKET_PRELOAD_MM,
        },
        "nl8_center_y_mm": NL8_CENTER_Y_MM,
        "nl8_cutout_d_mm": NL8_CUTOUT_D_MM,
        "nl8_screw_d_mm": NL8_SCREW_D_MM,
        "nl8_screw_pitch_mm": NL8_SCREW_PITCH_MM,
        "service_cavity_xyz_mm": (
            SERVICE_CAVITY_X_MM,
            SERVICE_CAVITY_Y_MM,
            SERVICE_CAVITY_Z_MM,
        ),
        "floor_lane_count": len(FLOOR_LANE_SPECS),
        "floor_lanes": lanes,
        "feature_group_count": integrated_floor_feature_group_count(),
    }
