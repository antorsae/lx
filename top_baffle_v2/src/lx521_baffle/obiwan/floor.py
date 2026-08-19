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
    Rectangle,
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

from ..base import (
    L22_CUTOUT,
    STAND_FOOT,
    THICKNESS_MM,
    m5_insert_bore_cutter,
)
from ..cables import (
    FLOOR_LM_DUCT_OUT_Y_MM,
    LM_DUCT_OUT_REAR_Z_MM,
    LM_DUCT_OUT_X_MM,
    LM_EXIT_BEND_R_MM,
    lm_exit_handoff_points,
    lm_exit_handoff_spec,
)
from ..flush import PAD_FACE_Z
from ..floor_bend import (
    BEND_VERTICAL_HANDLE_MM,
    FUSION_OVERLAP_MM as FLOOR_BEND_FUSION_OVERLAP_MM,
    bend_facts,
    canonical_lane_controls,
    centerline_controls,
    cubic_point,
    sampled_minimum_radius,
)
from .floor_strength import (
    FLOOR_BEND_KW,
    FLOOR_FOOT_FLAT_MM,
    FLOOR_Y_MM,
    FOOT_FRONT_Z_MM,
    FOOT_HEIGHT_MM,
    FOOT_REAR_Z_MM,
    FOOT_WIDTH_MM,
    LM_AXIS_Y_MM,
)


# --- NL8 boss, underside service bay and snap lid ---------------------
# The rear is no longer a flat W64 panel: a 38.4-square mating face
# (NL8MPRXX flange 38.16 + 0.12/side lip, all four corners r4.2) crowns a
# body whose crest holds level over the connector barrel (true penetration
# 25.85 mm, vendor-STEP contact plane 7.15 behind the D-face tip) and
# falls as one cosine to the foot.  The service opening moved from the
# dome to the foot's UNDERSIDE (2026-08-19): the same crown-to-end-wall
# plan is cut upward from the flat floor face, the dome stays unbroken,
# and a flat recessed lid clips in from below on four hidden cantilever
# fingers.  The duct-entry wall stays at Z=-77.
BOSS_TOP_W_MM = 38.4
NL8_CENTER_Y_MM = BOSS_TOP_W_MM / 2.0          # square face, derived
BOSS_FLANGE_T_MM = 5.6                         # insert seat 4.0 + 1.6 roof
BOSS_CREST_HOLD_Z_MM = -120.0
BOSS_FALL_SPAN_MM = 44.0
# Full re-span: ONE cosine ease carries the whole 38.4 -> 64.0 widening
# over the complete wall path -- flange face, straight run, and the
# entire bend arc -- reaching 64.0 exactly at the vertical tangent into
# the stem.  Nothing completes at the junction: width and corner radius
# are both continuous there, the lower stand reads as one narrowing
# waist wrapping around the arc, and the corner fillets stay at full
# radius wherever the wall leans, fading to a sharp edge only where it
# is perpendicular.
# The boss loft and the bend loft BUTT at the horizontal tangent with
# identical cross-sections: a clean planar union interface, no lateral
# coincident surfaces (which shed sub-mm3 boolean debris into the keyed
# split mass audit) and no proud overlap.
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

# The service opening lives on the foot's UNDERSIDE: the same crown/
# spring/straight/end-wall plan that proved printable as the old top
# trough is cut upward from the flat floor face instead, so the dome
# above stays completely unbroken (which also retired the top lip's
# razor-fin rabbet).  The bay ceiling clears the tallest lane mouth
# (LM D9 at y 10.5 tops at 15.0) by 0.6 while keeping >=2.4 mm of roof
# under the dome at the shallowest station.  A 1.0-mm perimeter rebate
# seats a flat lid recessed 0.1 below the floor plane: the frame rails
# carry the stand, the floor presses the lid into its seat when the
# stand is upright, and four cantilever fingers retain it in handling.
BAY_CEILING_Y_MM = 15.6
BAY_REBATE_OFF_MM = 1.0
LID_T_MM = 1.8                                 # 3 beads at 0.6
LID_RECESS_Y_MM = 0.1
BAY_REBATE_DEPTH_Y_MM = LID_T_MM + LID_RECESS_Y_MM
LID_CLEARANCE_MM = 0.2
LID_CLIP_SPANS_Z_MM = ((-120.0, -112.5), (-86.5, -79.5))
LID_POCKET_X_MM = (16.5, 18.1)
LID_POCKET_CEILING_Y_MM = 8.6                  # hook bearing plane
LID_POCKET_FLOOR_Y_MM = 5.1
LID_POCKET_PRELOAD_MM = 0.05
LID_BLADE_T_MM = 1.2
LID_NAIL_GROOVE_HALF_X_MM = 4.0
LID_NAIL_GROOVE_DEPTH_MM = 0.8

# M5 floor anchor: the stand bolts down through the owner's base from
# below.  The shared stepped insert bore (D6.5 x 2.0 entry + D6.4 body,
# 6.8 deep -- the same insert type as the driver pilots and wing bores)
# opens flush with the floor face on the centre axis, as far forward as
# the 130-mm flat allows (rim 0.75 behind the z=-20 lift-off).  The LM
# lane ramps from its 10.5 mouth height to 12.55 across a buried cosine
# ramp (min R 44) and runs raised over the anchor: 1.25 of web above the
# bore and 1.25 of roof under the foot's top face, both two perimeters.
# Interior top wall carried forward from the barrel chamber's 3.6 roof
# when the dead block between the chamber and the entry wall was hollowed.
HOLLOW_TOP_WALL_MM = 3.6

# Two anchors: the front one as far forward as the flat allows, and a
# rear one as close to the lid as the LM lane's climb permits -- the lane
# leaves its 10.5 mouth immediately behind the entry wall and reaches the
# raised 12.55 run in 13 mm (cosine ramp, min R 18.8), so the rear bore's
# rim sits at z=-65, 12 mm forward of the bay's end wall.
FLOOR_M5_ANCHOR_Z_MM = (-61.75, -24.0)
FLOOR_M5_ANCHOR_DEPTH_MM = 6.8
FLOOR_LM_RAISED_Y_MM = 12.55
FLOOR_LM_RAMP_Z_MM = (-78.0, -65.0)
FLOOR_LM_RAMP_HANDLE_MM = 5.2

# The enclosed cavity is now exactly the barrel chamber: flush with the
# bay walls, ending where the open underside bay takes over.
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
# The Obi-Wan stand rides the shortened long-foot bend (130-mm flat,
# lift-off z=-20); Stock/Slim keep the shared 75-mm span via the
# parameter defaults.  See FLOOR_BEND_KW in floor_strength.py.
FLOOR_BEND_VERTICAL_TANGENT_Y_MM = bend_facts(**FLOOR_BEND_KW)[
    "vertical_tangent_xyz_mm"][1]
FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM = bend_facts(**FLOOR_BEND_KW)[
    "horizontal_tangent_xyz_mm"][2]
FLOOR_BEND_UPRIGHT_START_Y_MM = (
    FLOOR_BEND_VERTICAL_TANGENT_Y_MM - FLOOR_BEND_FUSION_OVERLAP_MM)
FLOOR_BEND_REAR_FLAT_END_Z_MM = (
    FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM + FLOOR_BEND_FUSION_OVERLAP_MM)

FLOOR_LANE_BEND_R_MM = bend_facts(**FLOOR_BEND_KW)[
    "minimum_centerline_radius_mm"]
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
# The mouth sits at the unified Obi-Wan 17.8-mm aperture clearance (see
# cables.py): with the old shared 10-mm outlet the R14 turn's cutter broke
# through the driver-recess floor between the pilot bores.
FLOOR_LM_EXIT_HANDOFF = lm_exit_handoff_spec(
    12.55, STEM_Z_MM[0], LM_DUCT_OUT_REAR_Z_MM,
    face_xy_mm=(LM_DUCT_OUT_X_MM, FLOOR_LM_DUCT_OUT_Y_MM))
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


def _bend_arc_lengths():
    controls = centerline_controls(**FLOOR_BEND_KW)
    points = [cubic_point(controls, index / 400.0) for index in range(401)]
    cumulative = [0.0]
    for left, right in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(left, right))
    return tuple(cumulative)


_BEND_ARC_CUMULATIVE = _bend_arc_lengths()
BOSS_PATH_STRAIGHT_MM = (
    FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM - FOOT_REAR_Z_MM)
BOSS_PATH_TOTAL_MM = BOSS_PATH_STRAIGHT_MM + _BEND_ARC_CUMULATIVE[-1]


def _boss_path_width_mm(s: float) -> float:
    return BOSS_TOP_W_MM + (FOOT_WIDTH_MM - BOSS_TOP_W_MM) * _boss_ease(
        s / BOSS_PATH_TOTAL_MM)


def boss_width_mm(z: float) -> float:
    return _boss_path_width_mm(
        min(z - FOOT_REAR_Z_MM, BOSS_PATH_STRAIGHT_MM))


def _boss_section_w_h_r(z: float):
    w, h = boss_width_mm(z), boss_height_mm(z)
    r = min(
        4.2 + 3.8 * _boss_ease((z - FOOT_REAR_Z_MM) / 40.0),
        h / 3.0, w / 4.0)
    return w, h, r


def _boss_bottom_r_mm(z: float, r_top: float, h: float, w: float) -> float:
    """Bottom-corner radius: tight through the underside bay span.

    The full-respan radius applies where the wall leans (the dome and the
    bend); the foot's BOTTOM corners instead cap at 2.4 through the bay
    span so the lid rebate's x=17.55 wall stays buried inside the corner
    arc (the full radius crossed it and shed tangential slivers).  The
    radius blends from the NL8 flange's face rounding behind the crown and
    back up to the shared respan value at the horizontal tangent, keeping
    the boss/bend butt joint's sections identical.
    """
    if z <= -146.0:
        raw = r_top
    elif z <= -141.0:
        raw = r_top + (2.4 - r_top) * _boss_ease((z + 146.0) / 5.0)
    elif z <= -50.0:
        raw = 2.4
    else:
        raw = 2.4 + (r_top - 2.4) * _boss_ease((z + 50.0) / 30.0)
    return min(raw, h / 3.0, w / 4.0)


def _boss_section(z: float, y0: float = 0.0):
    w, h, r = _boss_section_w_h_r(z)
    rb = _boss_bottom_r_mm(z, r, h, w)
    half = w / 2.0
    wire = (Polyline((half, y0), (-half, y0), (-half, h))
            + Line((-half, h), (half, h)) + Line((half, h), (half, y0)))
    face = make_face(wire)
    top = [v for v in face.vertices() if abs(abs(v.X) - half) < 1e-6
           and abs(v.Y - h) < 1e-6]
    if r > 0.2 and top:
        face = fillet(top, r)
    bottom = [v for v in face.vertices() if abs(abs(v.X) - half) < 1e-6
              and abs(v.Y - y0) < 1e-6]
    if rb > 0.2 and bottom:
        face = fillet(bottom, rb)
    return Plane.XY.offset(z) * face


def _bend_taper_w_r(u: float):
    arc = _BEND_ARC_CUMULATIVE[
        min(len(_BEND_ARC_CUMULATIVE) - 1, round(u * 400))]
    w = _boss_path_width_mm(BOSS_PATH_STRAIGHT_MM + arc)
    r = (FOOT_HEIGHT_MM / 3.0) * (
        1.0 - _boss_ease(arc / _BEND_ARC_CUMULATIVE[-1]))
    return w, min(r, w / 4.0)


def _tapered_bend_loft():
    """The Option-B wall as perpendicular sections along its centerline.

    Constant 18.3 normal thickness like ``bent_wall_prism``, but the
    width carries the trumpet's final millimetre (63.0 -> 64.0) and the
    corner radius fades 2.0 -> sharp around the arc, so the boss taper
    only completes where the wall is perpendicular.
    """
    controls = centerline_controls(**FLOOR_BEND_KW)

    def frame(u: float):
        point = cubic_point(controls, u)
        # The Option-B endpoints are exactly horizontal/vertical; the
        # analytic tangents keep the first section coplanar with the
        # boss loft's butt face so the union dissolves the interface.
        if u <= 0.0:
            return (point[1], point[2]), (0.0, 1.0)
        if u >= 1.0:
            return (point[1], point[2]), (1.0, 0.0)
        step = 1.0e-4
        rear = cubic_point(controls, max(0.0, u - step))
        fore = cubic_point(controls, min(1.0, u + step))
        ty, tz = fore[1] - rear[1], fore[2] - rear[2]
        norm = math.hypot(ty, tz)
        return (point[1], point[2]), (ty / norm, tz / norm)

    def section(origin_yz, tangent_yz, w, r):
        plane = Plane(
            origin=(0.0, origin_yz[0], origin_yz[1]),
            x_dir=(1.0, 0.0, 0.0),
            z_dir=(0.0, tangent_yz[0], tangent_yz[1]))
        if r > 0.15:
            return plane * RectangleRounded(w, FOOT_HEIGHT_MM, r)
        return plane * Rectangle(w, FOOT_HEIGHT_MM)

    sections = []
    for index in range(33):
        u = index / 32.0
        origin, tangent = frame(u)
        w, r = _bend_taper_w_r(u)
        sections.append(section(origin, tangent, w, r))
    (y1, z1), t1 = frame(1.0)
    for fore in (FLOOR_BEND_FUSION_OVERLAP_MM / 2.0,
                 FLOOR_BEND_FUSION_OVERLAP_MM):
        sections.append(section(
            (y1 + t1[0] * fore, z1 + t1[1] * fore), t1,
            FOOT_WIDTH_MM, 0.0))
    shape = loft(sections, ruled=True)
    # The ruled chords between rotated sections overshoot the exact
    # Y=0/Z=18.3 datum planes by nanometres; the coplanarity gates demand
    # exact, so clip to the analytic envelope like bent_wall_prism kept.
    return shape & Pos(0.0, 60.0, -70.85) * Box(80.0, 120.0, 178.3)


def _boss_prism():
    """The uncut boss loft; also the blank the lid is carved from."""
    count = int(BOSS_PATH_STRAIGHT_MM // 3.0)
    sections = [_boss_section(FOOT_REAR_Z_MM + 3.0 * i)
                for i in range(count + 1)]
    sections.append(_boss_section(FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM))
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


def _bay_prism(outline, y0: float, y1: float):
    """The plan outline as a vertical prism spanning y0..y1."""
    return (Pos(0.0, y1, 0.0) * Rot(90, 0, 0)
            * extrude(make_face(Polyline(outline)), y1 - y0))


def _bay_cutter():
    """Underside service bay plus the lid's perimeter rebate ring.

    The bay reuses the crown/spring/straight/end-wall plan unchanged (its
    flank and spring angles are what made the old top trough printable in
    the front-face-down orientation; as the print-roof of an underside bay
    they work identically).  The rebate ring sinks the flat lid 0.1 below
    the floor plane so the frame rails seat the stand deterministically.
    """
    bay = _bay_prism(_trough_plan_points(), -1.0, BAY_CEILING_Y_MM)
    rebate = _bay_prism(
        _trough_plan_offset(BAY_REBATE_OFF_MM, BAY_REBATE_OFF_MM),
        -1.0, BAY_REBATE_DEPTH_Y_MM)
    return bay.fuse(rebate).clean()


def _lid_pocket_cutters():
    """Blind finger pockets in both bay side walls.

    The pocket's z0 leg keeps the 45-degree slope: z0 is the print-top of
    a side cavity in the front-face-down orientation, exactly like the old
    top-trough pockets it mirrors.
    """
    x0, x1 = LID_POCKET_X_MM
    cutters = []
    for z0, z1 in LID_CLIP_SPANS_Z_MM:
        z0p, z1p = z0 - 0.6, z1 + 0.6
        for sx in (-1, 1):
            face = Plane.YZ.offset(
                min(sx * x0, sx * x1)) * make_face(
                Polyline([
                    (LID_POCKET_CEILING_Y_MM, z0p),
                    (LID_POCKET_CEILING_Y_MM, z1p),
                    (LID_POCKET_FLOOR_Y_MM, z1p),
                    (LID_POCKET_FLOOR_Y_MM, z0p + 3.5),
                    (LID_POCKET_CEILING_Y_MM, z0p)]))
            cutters.append(extrude(face, x1 - x0))
    return cutters


def _floor_m5_anchor_cutters():
    """Vertical stepped M5 insert bores, mouths flush with the floor face."""
    return tuple(
        Pos(0.0, 0.0, z) * Rot(-90, 0, 0)
        * m5_insert_bore_cutter(
            (0.0, 0.0), opening_z=0.0,
            total_depth=FLOOR_M5_ANCHOR_DEPTH_MM,
            opening_side="-z")
        for z in FLOOR_M5_ANCHOR_Z_MM)


def _hollow_end_z_mm() -> float:
    """Where the dome-following interior ceiling meets the bay ceiling."""
    lo, hi = BOSS_CREST_HOLD_Z_MM, TROUGH_END_WALL_Z_MM
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if boss_height_mm(mid) - HOLLOW_TOP_WALL_MM > BAY_CEILING_Y_MM:
            lo = mid
        else:
            hi = mid
    return lo


def _boss_cavity_cutter():
    """Barrel chamber plus the hollowed run to the duct-entry wall.

    Forward of the crest hold the interior used to be dead solid between
    the bay ceiling and the dome; it carries no meaningful load (the
    strength root is the bend tangent 60 mm forward), so the chamber now
    continues as one cavity whose ceiling tracks the falling dome at a
    constant 3.6 top wall, walls flush with the bay's +-16.55, until the
    ceiling meets the bay ceiling and the existing (thinner) roof takes
    over toward the entry wall.
    """
    cy = NL8_CENTER_Y_MM
    y_lo = SERVICE_CAVITY_Y_MM[0]
    sections = [
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
    ]
    end_z = _hollow_end_z_mm()
    count = max(2, int((end_z - BOSS_CREST_HOLD_Z_MM) // 3.0))
    for index in range(1, count + 1):
        z = (BOSS_CREST_HOLD_Z_MM
             + (end_z - BOSS_CREST_HOLD_Z_MM) * index / count)
        top = boss_height_mm(z) - HOLLOW_TOP_WALL_MM
        height = top - y_lo
        sections.append(
            Plane.XY.offset(z) * Pos(0.0, (y_lo + top) / 2.0)
            * RectangleRounded(2 * TROUGH_HALF_W_MM, height,
                               min(6.0, height / 2.0 - 0.1)))
    return loft(sections, ruled=True)


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
    """The snap-in underside bay lid: a flat recessed plate on four fingers.

    The plate drops into the perimeter rebate from below with 0.2 shutline
    clearance and 0.8 of ledge overlap all around; standing, the floor
    presses it into its seat.  Four cantilever blades rise along the bay
    walls into the blind pockets, hooks bearing upward with 0.05 preload,
    mirroring the proven top-lid finger geometry.  A 45-degree nail groove
    across the front edge (facing the floor, invisible in service) opens
    it.
    """
    _require_guarded_build()
    if not STAND_FOOT:
        raise RuntimeError("no-floor Obi-Wan has no service lid")
    inset = BAY_REBATE_OFF_MM - LID_CLEARANCE_MM
    lid = _bay_prism(_trough_plan_offset(inset, inset),
                     LID_RECESS_Y_MM, LID_RECESS_Y_MM + LID_T_MM)
    plate_top = LID_RECESS_Y_MM + LID_T_MM
    hook_bearing = LID_POCKET_CEILING_Y_MM + LID_POCKET_PRELOAD_MM
    for z0, z1 in LID_CLIP_SPANS_Z_MM:
        for sx in (-1, 1):
            blade_out = TROUGH_HALF_W_MM - LID_CLEARANCE_MM
            blade = Plane.YZ.offset(
                min(sx * blade_out,
                    sx * (blade_out - LID_BLADE_T_MM))) * make_face(
                Polyline([(plate_top - 1.4, z0), (hook_bearing - 0.1, z0),
                          (hook_bearing - 0.1, z1), (plate_top - 1.4, z1),
                          (plate_top - 1.4, z0)]))
            lid += extrude(blade, LID_BLADE_T_MM)
            hook = Plane.XY.offset(z0 + 0.6) * make_face(Polyline(
                [(sx * blade_out, hook_bearing),
                 (sx * (blade_out + 0.6), hook_bearing - 0.16),
                 (sx * blade_out, hook_bearing - 1.36),
                 (sx * blade_out, hook_bearing)]))
            lid += extrude(hook, (z1 - 1.8) - (z0 + 0.6))
            release = Pos(
                sx * (TROUGH_HALF_W_MM - 0.8),
                (plate_top - 0.05 + hook_bearing - 1.5) / 2.0, z1) \
                * Rot(0, 45, 0) * Box(
                    1.71, (hook_bearing - 1.5) - (plate_top - 0.05), 1.71)
            lid -= release
    lid -= (Pos(0.0, LID_RECESS_Y_MM,
                TROUGH_END_WALL_Z_MM + BAY_REBATE_OFF_MM - LID_CLEARANCE_MM)
            * Rot(45, 0, 0)
            * Box(2.0 * LID_NAIL_GROOVE_HALF_X_MM,
                  2.0 * LID_NAIL_GROOVE_DEPTH_MM,
                  2.0 * LID_NAIL_GROOVE_DEPTH_MM))
    # Emit in a front-face-down-compatible frame: the shared X180 export
    # contract then lands the flat lid outer-face-down with the fingers
    # up, instead of standing a 1.8-mm plate 65 mm tall on its end.
    lid = (Rot(-90, 0, 0) * lid).clean()
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
    bend = _tapered_bend_loft()
    boss = _boss_prism()
    body = boss.fuse(bend, _stem_prism()).clean()
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
    lane_y = (FLOOR_LM_RAISED_Y_MM if name == "lm"
              else spec["floor_y_mm"])
    canonical = canonical_lane_controls(
        spec["x_mm"], lane_y, spec["stem_z_mm"],
        **FLOOR_BEND_KW)
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


def _lm_ramp_controls():
    """Buried cosine ramp lifting the LM run over the M5 floor anchor."""
    spec = FLOOR_LANE_SPECS["lm"]
    x = spec["x_mm"]
    low = spec["floor_y_mm"]
    z0, z1 = FLOOR_LM_RAMP_Z_MM
    handle = FLOOR_LM_RAMP_HANDLE_MM
    return ((x, low, z0), (x, low, z0 + handle),
            (x, FLOOR_LM_RAISED_Y_MM, z1 - handle),
            (x, FLOOR_LM_RAISED_Y_MM, z1))


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
    if name == "lm":
        # The ramp's rear tangent is horizontal at the mouth itself, so
        # the straight low segment vanished when the rear anchor pulled
        # the climb back to the entry wall.
        edges = [Bezier(*_lm_ramp_controls()),
                 Line(_lm_ramp_controls()[-1], controls[0])]
    else:
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
    if name == "lm":
        ramp = _lm_ramp_controls()
        points = [ramp[0]]
        points.extend(
            _cubic_point(ramp, index / 16.0) for index in range(1, 17))
        points.append(controls[0])
    else:
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
        return "underside_service_bay_lid_seats_and_anchors", (
            _bay_cutter(), *_lid_pocket_cutters(),
            *_floor_m5_anchor_cutters())
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
        "floor_bend": bend_facts(**FLOOR_BEND_KW),
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
            "bay_half_w_mm": TROUGH_HALF_W_MM,
            "bay_side": "underside",
            "bay_ceiling_y_mm": BAY_CEILING_Y_MM,
            "bay_rebate_off_mm": BAY_REBATE_OFF_MM,
            "duct_entry_wall_z_mm": TROUGH_END_WALL_Z_MM,
            "lid_thickness_mm": LID_T_MM,
            "lid_recess_y_mm": LID_RECESS_Y_MM,
            "lid_shutline_clearance_mm": LID_CLEARANCE_MM,
            "lid_clip_spans_z_mm": LID_CLIP_SPANS_Z_MM,
            "lid_pocket_preload_mm": LID_POCKET_PRELOAD_MM,
            "m5_floor_anchor": {
                "centers_xz_mm": tuple(
                    (0.0, z) for z in FLOOR_M5_ANCHOR_Z_MM),
                "bore_depth_mm": FLOOR_M5_ANCHOR_DEPTH_MM,
                "opens": "floor_face_flush",
                "lm_raised_y_mm": FLOOR_LM_RAISED_Y_MM,
                "web_above_bore_mm": (
                    FLOOR_LM_RAISED_Y_MM - 4.5
                    - FLOOR_M5_ANCHOR_DEPTH_MM),
                "roof_above_lane_mm": (
                    FOOT_HEIGHT_MM - (FLOOR_LM_RAISED_Y_MM + 4.5)),
            },
            "full_respan": {
                "path_total_mm": BOSS_PATH_TOTAL_MM,
                "root_width_mm": _boss_path_width_mm(
                    BOSS_PATH_STRAIGHT_MM),
                "completes_at": "vertical_tangent",
            },
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
