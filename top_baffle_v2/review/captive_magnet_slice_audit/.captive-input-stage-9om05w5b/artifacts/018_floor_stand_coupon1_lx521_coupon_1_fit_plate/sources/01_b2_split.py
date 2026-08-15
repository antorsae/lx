"""Print-split of variant B2 (see :mod:`lx521_baffle.proud.b2`).

The one-piece baffle (304.8 x 453.5 x 18.3 mm) exceeds a 256 mm print bed,
so it is split into four flat pieces joined by through-thickness dovetail
keys:

  seam A  y = 120   full-width horizontal cut (crosses the D190 cutout, so
                    two ~58 mm land segments are actually joined)
  seam B  y = 315.95  horizontal cut exactly through B2's waist kinks, so
                    the seam hides in the outline crease and BOTH pieces get
                    obtuse corners there (top foot ~107 deg vs the flare,
                    mids ~152 deg vs the chamfer) -- no brittle knife-tips
  seam C  x = -5.6  vertical cut between seams A and B (mostly inside the
                    D190 cutout; ~20 mm of real joint above the cutout).
                    Its offset leaves the centerline inside mid_right for
                    the hidden radial seam-B M3 fastener.

Pieces (all fit a 256 x 256 bed front-face-down):
  piece_bottom     ~250.6 x 125 mm   (male dovetails up into the mids)
  piece_mid_left   ~146.7 x 202 mm   (male dovetail up into the top)
  piece_mid_right  ~162.0 x 202 mm   (male dovetails up + into mid_left)
  piece_top_b2     ~121.3 x 138 mm   (the whole B2 mini-vase + crescent)

Female sides are opened up by CLEARANCE_MM (0.05, a snug fit --
tune X-Y hole compensation on the coupon first) via a mitred offset,
so every mating surface has a uniform assembly gap.

One M3 x 20 socket-cap screw enters radially from the hidden upper wall of
the LM cutout, crosses ``piece_mid_right``, and engages a D4.6 x 4.0 heat-set
receiver opening on the vase's seam-B face.  Its D6.2 x 3.4 head pocket is
fully recessed and becomes inaccessible/hidden when the W22 is installed.
"""

from __future__ import annotations

import math

from build123d import (
    Box,
    Cylinder,
    Part,
    Plane,
    Polyline,
    Pos,
    Rot,
    Shell,
    Solid,
    Wire,
    extrude,
    make_face,
)
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from ..assembly import ordered_labeled_compound
from ..base import L22_CUTOUT, STAND_FOOT, THICKNESS_MM, baffle_solid
from ..floor_bend import (
    BEND_HORIZONTAL_ENDPOINT_RADIUS_MM,
    BEND_MIN_CENTERLINE_RADIUS_MM,
    BEND_REAR_SPAN_MM,
    BEND_RISE_MM,
    FUSION_OVERLAP_MM as FLOOR_BEND_FUSION_OVERLAP_MM,
    WALL_THICKNESS_MM as FLOOR_BEND_WALL_THICKNESS_MM,
    bend_facts,
    bent_wall_lateral_hermite,
    centerline_controls as floor_bend_centerline_controls,
    cubic_derivatives as floor_bend_cubic_derivatives,
)
from .b import (
    TWEETER_DROP_MM,
    apply_magnet_base_cavities,
)
from .b2 import OUTLINE_B2
from ..cables import ROUTING_PROFILE, cable_cutters
from ..cables import TS_ROUTE_CAPTIVE

PRINT_ORIENTATION = "front-face-down"

if ROUTING_PROFILE != "proud":
    raise RuntimeError(
        "B2/V1/V1L pieces require LX_ROUTING_PROFILE=proud; "
        "Obi-Wan has a separate skeletal builder"
    )

CLEARANCE_MM = 0.05

# Largest boundary displacement the Option-B tangent join caps may disagree
# by before the two owners are refused for sewing.  1e-5 mm is two orders
# below OCC's sewing tolerance and far below any print process.
JOIN_CAP_EDGE_TOLERANCE_MM = 1.0e-5

# Integral stand on piece_bottom (STAND_FOOT flag lives in the base module --
# it also removes the bridge pass-throughs and selects the floor ducts).  The
# old horizontal slab plus vertical plate and its hard 90-degree corner are
# replaced by the shared Option-B constant-thickness tangent wall.  The rear
# straight foot and tapered connector tongue retain their released envelope.
FOOT_THICK = 18.3         # vertical height of the foot slab
FOOT_DEPTH_REAR = 150.0   # behind the rear face (z=0)
FLANK_SLOPE = 0.29752     # dx/dy of the lower flanks
# The foot tapers in plan CONTINUOUSLY -- one straight line per side
# from the strip corners (+/-81.6 at the plate) to 38 wide at the panel
# inner face (z=-146), whose last 4 mm carry a minimal panel wall
# (38 x 44 x 4) for a Neutrik NL8MPXX-BAG: D31 cutout at (0, 20.5) +
# 4 x D3.2 on the 29.2 x 29.2 pattern. The center is channeled to a
# 4.0 floor between side rails so the D30.5 body fits; the four duct
# runs (packed at x -13.9/-5.45/+5.4/+13.9) exit through the channel's
# step face at z=-99, leaving ~40 mm of open channel to the connector
# tabs.
TONGUE_HALF_W = 19.0
PANEL_T = 4.0
PANEL_H = 44.0
CHANNEL_HALF_W = 17.0   # rails 2.0 thick; >=1.75 around the D30.5 body
CHANNEL_FLOOR = 4.0
CHANNEL_STEP_Z = -99.0
NL8_CENTER_Y = 20.5
NL8_CUTOUT_D = 31.0
NL8_SCREW_D = 3.2
NL8_SCREW_PITCH = 29.2

FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM = (
    floor_bend_centerline_controls()[0][2])
FLOOR_BEND_VERTICAL_TANGENT_Y_MM = (
    floor_bend_centerline_controls()[-1][1])
FLOOR_BEND_REAR_FLAT_END_Z_MM = (
    FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM
    + FLOOR_BEND_FUSION_OVERLAP_MM)
FLOOR_BEND_UPRIGHT_START_Y_MM = FLOOR_BEND_VERTICAL_TANGENT_Y_MM

# The rear foot and upright lower flank have different lateral envelopes.
# Match both exactly at the Option-B tangencies, including their side slopes,
# so the curved wall widens continuously rather than ending in the former
# diagonal constant-width clipping wedge.
FLOOR_PLAN_FRONT_Z_MM = 0.5
FLOOR_PLAN_FRONT_HALF_WIDTH_MM = 82.0
FLOOR_PLAN_REAR_Z_MM = -FOOT_DEPTH_REAR + PANEL_T
FLOOR_PLAN_REAR_HALF_WIDTH_MM = TONGUE_HALF_W
RIGHT_FLANK_BOTTOM_XY_MM = (76.201, 0.0)
RIGHT_FLANK_TOP_XY_MM = (152.401, 256.120)
LEFT_FLANK_BOTTOM_XY_MM = (-76.199, 0.0)
LEFT_FLANK_TOP_XY_MM = (-152.401, 256.155)


def _linear_value_and_slope(
        start, end, coordinate: float) -> tuple[float, float]:
    span = float(end[1]) - float(start[1])
    if abs(span) <= 1.0e-12:
        raise RuntimeError("lateral envelope datum has zero coordinate span")
    slope = (float(end[0]) - float(start[0])) / span
    return float(start[0]) + slope * (float(coordinate) - start[1]), slope


_floor_bend_controls = floor_bend_centerline_controls()
_floor_bend_start_derivative = floor_bend_cubic_derivatives(
    _floor_bend_controls, 0.0)[0]
_floor_bend_end_derivative = floor_bend_cubic_derivatives(
    _floor_bend_controls, 1.0)[0]
_rear_half_width, _rear_plan_dx_dz = _linear_value_and_slope(
    (FLOOR_PLAN_REAR_HALF_WIDTH_MM, FLOOR_PLAN_REAR_Z_MM),
    (FLOOR_PLAN_FRONT_HALF_WIDTH_MM, FLOOR_PLAN_FRONT_Z_MM),
    FLOOR_BEND_HORIZONTAL_TANGENT_Z_MM,
)
_upright_right_x, _upright_right_dx_dy = _linear_value_and_slope(
    RIGHT_FLANK_BOTTOM_XY_MM,
    RIGHT_FLANK_TOP_XY_MM,
    FLOOR_BEND_VERTICAL_TANGENT_Y_MM,
)
_upright_left_x, _upright_left_dx_dy = _linear_value_and_slope(
    LEFT_FLANK_BOTTOM_XY_MM,
    LEFT_FLANK_TOP_XY_MM,
    FLOOR_BEND_VERTICAL_TANGENT_Y_MM,
)
FLOOR_BEND_LATERAL_ENVELOPE = {
    "profile": "cubic_hermite_matches_rear_plan_and_lower_flanks",
    "rear_x_mm": (-_rear_half_width, _rear_half_width),
    "upright_x_mm": (_upright_left_x, _upright_right_x),
    "rear_dx_du": (
        -_rear_plan_dx_dz * _floor_bend_start_derivative[2],
        _rear_plan_dx_dz * _floor_bend_start_derivative[2],
    ),
    "upright_dx_du": (
        _upright_left_dx_dy * _floor_bend_end_derivative[1],
        _upright_right_dx_dy * _floor_bend_end_derivative[1],
    ),
    "rear_side_slope_dx_dz": (
        -_rear_plan_dx_dz, _rear_plan_dx_dz),
    "upright_side_slope_dx_dy": (
        _upright_left_dx_dy, _upright_right_dx_dy),
}

if not math.isclose(
        FOOT_THICK, FLOOR_BEND_WALL_THICKNESS_MM, abs_tol=1.0e-12):
    raise RuntimeError("proud foot thickness drifted from shared floor bend")

SEAM_A_Y = 120.0
SEAM_B_Y = 315.95  # exactly at B2's waist kinks -> obtuse seam corners

# Dovetail keys: (center along seam, neck width, head width, depth).
# Seam A sits at y=120 so its keys at +-89 live in the 16 mm-wide
# FULL-DEPTH window between the T arc (r=110, crossing at x~72.6) and
# the legacy knife-taper boundary (x~92) -- keys clear of every duct AND fully
# out of the taper in all variants. Seam-B teeth occupy the clear
# lands between the proud R6P return corridor and the waist kink.
# TWO per mid (was one each) so neither half pivots on a single
# tab. Spread across the flank band, straddling the duct
# crossings (TS at x=-80, UM at x=+88).
# inner teeth SHALLOW (depth 3) and at +-66 so the wall to the LM
# cutout (which bulges to x=57 at the seam-band top) stays ~7 mm, not
# ~1.5; outer teeth (+-103, depth 5) are the main anchor.  The positive
# seam-B tooth is centred at x=29 rather than x=30 so its grown female
# pocket stays clear of the complete lower-right captive-magnet sealing
# land.  This is a broad registration-key placement change, not a local
# magnet-shaped patch, and leaves the acoustic/front/rear envelope unchanged.
DOVETAILS_A = [(-103.0, 6.0, 7.0, 5.0), (-66.0, 7.0, 9.0, 3.0),
               (66.0, 7.0, 9.0, 3.0), (103.0, 6.0, 7.0, 5.0)]
DOVETAILS_B = [(-19.0, 10.0, 14.0, 6.0), (29.0, 10.0, 14.0, 6.0)]
# ONE tooth in the ~20 mm mid-mid neck (per user preference), centered
# and beefed (neck 7, head 8.5). mid_right carries the tab.
DOVETAILS_C = [(305.0, 7.0, 8.5, 4.0)]
SEAM_C_X = -5.6

# One radial M3 seam fastener supplements the two ordinary seam-B dovetails.
# Its socket-head access opens only into the LM driver cutout, so the screw is
# serviceable before the W22 is installed and completely hidden afterward.
# The common z=12.55 axis is the mid-plane of the thinnest V1/V1L section;
# Stock keeps the same datum so every proud-family mid/vase remains
# interchangeable.  The vase owns the same D4.6 x 4.0 M3 x 3 heat-set bore
# used by the released Obi-Wan joints.  A standard M3 socket-cap head is at
# most about D5.5 x H3.0; D6.2 x 3.4 provides radial and axial print clearance.
SEAM_B_M3_AXIS_X_MM = 0.0
SEAM_B_M3_AXIS_Z_MM = 12.55
SEAM_B_M3_CLEARANCE_D_MM = 3.4
SEAM_B_M3_HEAD_CLEARANCE_D_MM = 6.2
SEAM_B_M3_HEAD_RECESS_DEPTH_MM = 3.4
SEAM_B_M3_INSERT_BORE_D_MM = 4.6
SEAM_B_M3_INSERT_DEPTH_MM = 4.0
SEAM_B_M3_CUTTER_OVERSHOOT_MM = 0.20
SEAM_B_M3_ENTRY_Y_MM = L22_CUTOUT[1] + L22_CUTOUT[2] / 2.0
SEAM_B_M3_HEAD_SEAT_Y_MM = (
    SEAM_B_M3_ENTRY_Y_MM + SEAM_B_M3_HEAD_RECESS_DEPTH_MM
)
SEAM_B_M3_VASE_FACE_Y_MM = SEAM_B_Y + CLEARANCE_MM
SEAM_B_M3_RECOMMENDED_SCREW_LENGTH_MM = 20.0
SEAM_B_M3_INSERT_ENGAGEMENT_MM = (
    SEAM_B_M3_RECOMMENDED_SCREW_LENGTH_MM
    - (SEAM_B_M3_VASE_FACE_Y_MM - SEAM_B_M3_HEAD_SEAT_Y_MM)
)
SEAM_B_M3_INSERT_TIP_MARGIN_MM = (
    SEAM_B_M3_INSERT_DEPTH_MM - SEAM_B_M3_INSERT_ENGAGEMENT_MM
)

XMAX = 200.0  # anything beyond the baffle outline


def _trapezoid_up(cx: float, y: float, neck: float, head: float, depth: float) -> Polygon:
    return Polygon(
        [
            (cx - neck / 2.0, y),
            (cx + neck / 2.0, y),
            (cx + head / 2.0, y + depth),
            (cx - head / 2.0, y + depth),
        ]
    )


def _trapezoid_down(cx: float, y: float, neck: float, head: float, depth: float) -> Polygon:
    return Polygon(
        [
            (cx - neck / 2.0, y),
            (cx + neck / 2.0, y),
            (cx + head / 2.0, y - depth),
            (cx - head / 2.0, y - depth),
        ]
    )


def _below_region(seam_y: float, dovetails, tabs_up: bool = True) -> Polygon:
    """tabs_up: the below piece carries up-tabs (stock). Otherwise the
    below piece carries POCKETS (down-trapezoid notches) and the piece
    above inherits down-tabs via the same complement/grow machinery --
    growing a notched polygon shrinks the notches, which lands the
    0.05 clearance on the male side instead of the female (same fit)."""
    below = box(-XMAX, -20.0, XMAX, seam_y)
    if tabs_up:
        return unary_union(
            [below] + [_trapezoid_up(cx, seam_y, n, h, d)
                       for cx, n, h, d in dovetails])
    for cx, n, h, d in dovetails:
        below = below.difference(_trapezoid_down(cx, seam_y, n, h, d))
    return below


def _above_region(seam_y: float, dovetails) -> Polygon:
    """Exact complement of _below_region within the part's extent (used to
    CUT male-side pieces -- OCC's common op is flaky on the ducted solid,
    subtraction is robust)."""
    above = box(-XMAX, seam_y, XMAX, 600.0)
    for cx, n, h, d in dovetails:
        above = above.difference(_trapezoid_up(cx, seam_y, n, h, d))
    return above


def _right_region() -> Polygon:
    s = SEAM_C_X
    parts = [box(s, 50.0, XMAX, 400.0)]
    for cy, neck, head, depth in DOVETAILS_C:
        parts.append(Polygon([
            (s, cy - neck / 2.0),
            (s, cy + neck / 2.0),
            (s - depth, cy + head / 2.0),
            (s - depth, cy - head / 2.0),
        ]))
    return unary_union(parts)


def _prism(poly: Polygon):
    pts = list(poly.exterior.coords)
    face = make_face(Wire(Polyline(*pts).edges()))
    return extrude(Plane.XY.offset(-1.0) * face, amount=THICKNESS_MM + 2.0)


def _grown(poly: Polygon) -> Polygon:
    return poly.buffer(CLEARANCE_MM, join_style=2, mitre_limit=10.0)


def _y_axis_cylinder(
    diameter_mm: float,
    y0_mm: float,
    y1_mm: float,
):
    """Cylinder on the shared radial +Y fastener axis."""
    if y1_mm <= y0_mm:
        raise ValueError("radial cutter requires y1 > y0")
    return (
        Pos(
            SEAM_B_M3_AXIS_X_MM,
            (y0_mm + y1_mm) / 2.0,
            SEAM_B_M3_AXIS_Z_MM,
        )
        * Rot(90.0, 0.0, 0.0)
        * Cylinder(diameter_mm / 2.0, y1_mm - y0_mm)
    )


def seam_b_m3_mid_right_cutter():
    """Hidden M3 shaft passage plus recessed socket-head access."""
    y0 = SEAM_B_M3_ENTRY_Y_MM - SEAM_B_M3_CUTTER_OVERSHOOT_MM
    shaft = _y_axis_cylinder(
        SEAM_B_M3_CLEARANCE_D_MM,
        y0,
        SEAM_B_Y + SEAM_B_M3_CUTTER_OVERSHOOT_MM,
    )
    head = _y_axis_cylinder(
        SEAM_B_M3_HEAD_CLEARANCE_D_MM,
        y0,
        SEAM_B_M3_HEAD_SEAT_Y_MM,
    )
    cutter = shaft.fuse(head).clean()
    cutter.label = "seam_B_hidden_M3_clearance_and_head_access"
    return cutter


def seam_b_m3_vase_insert_cutter():
    """Blind D4.6 x 4.0 receiver opening only on the vase seam face."""
    cutter = _y_axis_cylinder(
        SEAM_B_M3_INSERT_BORE_D_MM,
        SEAM_B_M3_VASE_FACE_Y_MM - SEAM_B_M3_CUTTER_OVERSHOOT_MM,
        SEAM_B_M3_VASE_FACE_Y_MM + SEAM_B_M3_INSERT_DEPTH_MM,
    )
    cutter.label = "seam_B_M3_heat_set_receiver_D4p6x4"
    return cutter


def _floor_strip_prism():
    """Historic lower-flank strip extruded through the rear-foot depth."""
    w0 = 76.2
    w1 = 76.2 + FLANK_SLOPE * FOOT_THICK
    strip = make_face(Wire(Polyline(
        (-w0, 0.0), (w0, 0.0), (w1, FOOT_THICK),
        (-w1, FOOT_THICK), (-w0, 0.0)
    ).edges()))
    return extrude(
        Plane.XY.offset(-FOOT_DEPTH_REAR - 1.0) * strip,
        amount=FOOT_DEPTH_REAR + 1.0)


def _floor_plan_prism(*, y_min_mm: float, y_max_mm: float):
    """Tapered XZ foot plan extruded through a requested Y interval."""
    if y_max_mm <= y_min_mm:
        raise ValueError("floor plan prism needs a positive Y span")
    h = TONGUE_HALF_W
    panel_inner_z = -FOOT_DEPTH_REAR + PANEL_T
    plan = make_face(Wire(Polyline(
        (-82.0, 0.5), (82.0, 0.5), (h, panel_inner_z),
        (h, -FOOT_DEPTH_REAR), (-h, -FOOT_DEPTH_REAR),
        (-h, panel_inner_z), (-82.0, 0.5)
    ).edges()))
    # The plane normal is world -Y.  Start at the requested upper station and
    # extrude down to the lower one, matching the established foot builder.
    return extrude(
        Plane((0.0, y_max_mm, 0.0),
              x_dir=(1.0, 0.0, 0.0), z_dir=(0.0, -1.0, 0.0)) * plan,
        amount=y_max_mm - y_min_mm)


def _floor_panel():
    return Pos(
        0.0,
        PANEL_H / 2.0,
        -FOOT_DEPTH_REAR + PANEL_T / 2.0,
    ) * Box(2.0 * TONGUE_HALF_W, PANEL_H, PANEL_T)


def _apply_floor_connector_cutters(part):
    """Retain the NL8 panel/service geometry behind the new bend."""
    panel_inner_z = -FOOT_DEPTH_REAR + PANEL_T
    part -= Pos(
        0.0,
        (CHANNEL_FLOOR + PANEL_H + 6.0) / 2.0,
        (CHANNEL_STEP_Z + panel_inner_z) / 2.0,
    ) * Box(
        2.0 * CHANNEL_HALF_W,
        PANEL_H + 6.0 - CHANNEL_FLOOR,
        CHANNEL_STEP_Z - panel_inner_z,
    )
    panel_mid_z = -FOOT_DEPTH_REAR + PANEL_T / 2.0
    holes = [(0.0, NL8_CENTER_Y, NL8_CUTOUT_D)] + [
        (sx * NL8_SCREW_PITCH / 2.0,
         NL8_CENTER_Y + sy * NL8_SCREW_PITCH / 2.0,
         NL8_SCREW_D)
        for sx in (1.0, -1.0) for sy in (1.0, -1.0)
    ]
    for cx, cy, diameter in holes:
        part -= Pos(cx, cy, panel_mid_z) * Cylinder(
            diameter / 2.0, PANEL_T + 2.0)
    return part


def _option_b_floor_bottom(ducted_bottom, ducts, *, shape_cuts=(),
                           wall_thickness_law=None):
    """Substitute the lower hard corner with the complete Option-B wall.

    ``wall_thickness_law`` is V1L's path-length rear-thickness ramp, which
    carries on through the bend; stock passes None and keeps the released
    constant 18.3-mm wall.
    """
    lateral = FLOOR_BEND_LATERAL_ENVELOPE
    wall = bent_wall_lateral_hermite(
        rear_left_x_mm=lateral["rear_x_mm"][0],
        rear_right_x_mm=lateral["rear_x_mm"][1],
        upright_left_x_mm=lateral["upright_x_mm"][0],
        upright_right_x_mm=lateral["upright_x_mm"][1],
        rear_left_dx_du=lateral["rear_dx_du"][0],
        rear_right_dx_du=lateral["rear_dx_du"][1],
        upright_left_dx_du=lateral["upright_dx_du"][0],
        upright_right_dx_du=lateral["upright_dx_du"][1],
        thickness_law=wall_thickness_law,
    )

    # The analytic bend ends with an exact vertical tangent and the complete
    # z=0..18.3 wall section at this station.  Join there without Boolean
    # overlap: the two qualified owners are sewn by their shared boundary.
    lower_cut_y = FLOOR_BEND_VERTICAL_TANGENT_Y_MM
    lower_cut = Pos(
        0.0, lower_cut_y / 2.0, THICKNESS_MM / 2.0,
    ) * Box(2.0 * XMAX, lower_cut_y, THICKNESS_MM + 2.0)
    retained_upright = ducted_bottom - lower_cut

    old_foot = (_floor_strip_prism() & _floor_plan_prism(
        y_min_mm=-1.0, y_max_mm=FOOT_THICK + 1.0)).clean()
    flat_length = (
        FLOOR_BEND_REAR_FLAT_END_Z_MM + FOOT_DEPTH_REAR)
    rear_flat_clip = Pos(
        0.0,
        FOOT_THICK / 2.0,
        (-FOOT_DEPTH_REAR + FLOOR_BEND_REAR_FLAT_END_Z_MM) / 2.0,
    ) * Box(
        2.0 * XMAX,
        FOOT_THICK + 2.0,
        flat_length,
    )
    rear_flat = (old_foot & rear_flat_clip).clean()

    # Keep every cable Boolean local to the lower floor owner. Applying a
    # full-length ruled cutter after it had been fused to the upper baffle
    # made OCC re-partition a distant LM tunnel face around y=90 mm. The
    # resulting BREP passed is_valid but could not produce a closed STEP/STL.
    # Here floor_body stops at the bend tangent (~74 mm), so the same cutters
    # cannot touch the already-qualified upper tunnel topology.
    floor_body = wall.fuse(rear_flat, _floor_panel())
    # A split variant can own rear-side shaping at the vertical tangent, so
    # a taper may already be fully active by y=74.15.  Apply
    # the same pre-duct cutters to the curved lower owner so both sides of the
    # sewn join expose the identical section; below the tangent those global
    # cutters naturally leave the rearward-moving Option-B wall.
    for shape_cutter in shape_cuts:
        floor_body -= shape_cutter
    floor_body = _apply_floor_connector_cutters(floor_body)
    for duct in ducts:
        floor_body -= duct

    # Boolean fusion reprocesses remote faces even when the only intersection
    # is this tangent station. Preserve both owners exactly: remove their
    # coincident planar join caps, sew the remaining oriented faces, and make
    # the one closed solid. Every duct boundary at the join comes from the
    # same cutter, so the upper and lower wire sets are identical.
    join_y = FLOOR_BEND_VERTICAL_TANGENT_Y_MM

    def exterior_faces_without_join(shape, owner):
        exterior = []
        join_faces = []
        for face in shape.faces():
            bbox = face.bounding_box()
            if (math.isclose(bbox.min.Y, join_y, abs_tol=1.0e-5)
                    and math.isclose(bbox.max.Y, join_y, abs_tol=1.0e-5)):
                join_faces.append(face)
            else:
                exterior.append(face)
        if not join_faces:
            raise RuntimeError(
                f"Option-B {owner} owner has no y={join_y:g} join cap")
        return (
            exterior,
            sum(face.area for face in join_faces),
            sum(edge.length for face in join_faces for edge in face.edges()),
            len(join_faces),
        )

    (upright_faces, upright_join_area, upright_join_perimeter,
     upright_join_count) = exterior_faces_without_join(
        retained_upright, "upright")
    (floor_faces, floor_join_area, floor_join_perimeter,
     floor_join_count) = exterior_faces_without_join(floor_body, "floor")
    # The two owners re-trim the SAME duct openings in different Boolean
    # contexts, so their cap areas differ by a few parts in 1e6 even where
    # the boundaries coincide analytically.  Bind that to an edge
    # DISPLACEMENT rather than a bare area, so the gate keeps its meaning
    # whatever the cap's size: an area mismatch is admissible only if it can
    # be explained by moving the whole cap boundary by less than
    # JOIN_CAP_EDGE_TOLERANCE_MM, which is two orders below OCC's sewing
    # tolerance.  The V1L floor bottom measures a 0.001613 mm2 difference
    # over a 497.205 mm cap perimeter: 3.2e-6 mm of displacement, against a
    # 0.004972 mm2 budget.
    join_perimeter = max(upright_join_perimeter, floor_join_perimeter)
    if abs(upright_join_area - floor_join_area) > (
            JOIN_CAP_EDGE_TOLERANCE_MM * join_perimeter):
        raise RuntimeError(
            "Option-B tangent join caps disagree: "
            f"upright={upright_join_area:.9f} mm2/{upright_join_count} faces, "
            f"floor={floor_join_area:.9f} mm2/{floor_join_count} faces, "
            f"perimeter={join_perimeter:.6f} mm -> "
            f"{abs(upright_join_area - floor_join_area) / join_perimeter:.3e}"
            " mm of edge displacement")
    body = Solid(Shell([*upright_faces, *floor_faces]))
    solids = tuple(body.solids())
    if (not body.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "Option-B proud floor bottom must be one valid solid; "
            f"valid={body.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def proud_floor_bend_facts() -> dict:
    """Public geometry record shared by tests and release documentation."""
    facts = bend_facts()
    facts.update({
        "family": "stock_slim_proud",
        "rear_depth_mm": FOOT_DEPTH_REAR,
        "rear_flat_end_z_mm": FLOOR_BEND_REAR_FLAT_END_Z_MM,
        "upright_start_y_mm": FLOOR_BEND_UPRIGHT_START_Y_MM,
        "plan_taper_half_width_rear_mm": TONGUE_HALF_W,
        "minimum_centerline_radius_mm": BEND_MIN_CENTERLINE_RADIUS_MM,
        "horizontal_endpoint_radius_mm": (
            BEND_HORIZONTAL_ENDPOINT_RADIUS_MM),
        "nominal_rear_span_mm": BEND_REAR_SPAN_MM,
        "nominal_rise_mm": BEND_RISE_MM,
        "lateral_envelope": FLOOR_BEND_LATERAL_ENVELOPE,
    })
    return facts


def pieces(outline=OUTLINE_B2, tweeter_drop_mm: float = TWEETER_DROP_MM,
           shape_cuts=(), shape_adds=(), magnet_cavities=True,
           crescent_front_mm=None, crescent_rear_mm=0.0,
           seam_b_dovetails=None, seam_b_tabs_up: bool = True,
           um_handoff_key: str = "proud",
           only: str | None = None,
           cable_routes=None,
           cable_y_range=None,
           ts_route_key: str = TS_ROUTE_CAPTIVE,
           floor_wall_thickness_law=None) -> dict:
    """Split the (optionally re-shaped) baffle into the four print
    pieces. ``shape_cuts``/``shape_adds`` are applied before the ducts
    are cut, and the ducts then re-cut through any added material.
    Obi-Wan is a separate two-carrier R6F core and never passes through
    this four-piece builder.
    ``um_handoff_key`` is explicit so V1L can select its rear-plane axis
    handoff while every other proud-family caller keeps the default.
    ``only`` constructs one requested split solid without retaining the
    other three OCC trees; guarded validation/export uses this to reduce the
    local process-tree RSS peak. ``cable_routes``
    may further omit cutters spatially disjoint from that one piece; the
    default still builds the complete cable set. ``cable_y_range`` trims
    the high-face-count TS loft on its original section grid for a known
    split band. ``ts_route_key`` defaults to the shared stock/slim captive-
    land keepout; explicit diagnostic callers may still request the legacy
    unnudged centerline. ``floor_wall_thickness_law`` lets a variant carry
    its rear-thickness ramp on through the Option-B bend; the default keeps
    the released constant-thickness stand."""
    piece_order = (
        "piece_bottom", "piece_mid_left", "piece_mid_right", "piece_top_b2")
    if only is not None and only not in piece_order:
        raise ValueError(f"unknown split piece {only!r}; choose {piece_order}")
    requested = set(piece_order if only is None else (only,))

    baffle = baffle_solid(outline, tweeter_drop_mm,
                          crescent_front_mm or THICKNESS_MM,
                          crescent_rear_mm)
    for add in shape_adds:
        baffle += add
    for cutter in shape_cuts:
        baffle -= cutter
    ducts = cable_cutters(
        um_handoff_key=um_handoff_key,
        route_names=cable_routes,
        ts_y_range=cable_y_range,
        ts_route_key=ts_route_key)  # internal cable ducts (LM/UM/T)
    # Preserve one pre-duct geometry authority for the integral floor bottom.
    # Its floor wall is fused first and every duct is then subtracted once.
    # The former cut -> fuse -> re-cut order produced coincident tunnel faces
    # around y=90 mm which were BREP-valid but failed both strict STL and STEP
    # round-trip. Upper pieces and the no-floor bottom retain the ordinary
    # single-cut baffle below.
    baffle_with_ducts = baffle
    for duct in ducts:
        baffle_with_ducts = baffle_with_ducts - duct

    below_a = _below_region(SEAM_A_Y, DOVETAILS_A)
    result = {}

    if "piece_bottom" in requested:
        upper_region = _prism(_above_region(SEAM_A_Y, DOVETAILS_A))
        if STAND_FOOT:
            ducted_bottom = baffle_with_ducts - upper_region
            bottom = _option_b_floor_bottom(
                ducted_bottom, ducts,
                shape_cuts=shape_cuts,
                wall_thickness_law=floor_wall_thickness_law)
        else:
            bottom = baffle_with_ducts - upper_region
        result["piece_bottom"] = bottom

    upper_requested = requested - {"piece_bottom"}
    if upper_requested:
        below_b = _below_region(
            SEAM_B_Y, seam_b_dovetails or DOVETAILS_B,
            tabs_up=seam_b_tabs_up)
        rest = baffle_with_ducts - _prism(_grown(below_a))
        if "piece_top_b2" in upper_requested:
            top = rest - _prism(_grown(below_b))
            if magnet_cavities:
                top = apply_magnet_base_cavities(top)
            top -= seam_b_m3_vase_insert_cutter()
            result["piece_top_b2"] = top
        mids_requested = upper_requested & {
            "piece_mid_left", "piece_mid_right"}
        if mids_requested:
            mids = rest & _prism(below_b)
            right_c = _right_region()
            if "piece_mid_right" in mids_requested:
                mid_right = mids & _prism(right_c)
                mid_right -= seam_b_m3_mid_right_cutter()
                result["piece_mid_right"] = mid_right
            if "piece_mid_left" in mids_requested:
                result["piece_mid_left"] = mids - _prism(_grown(right_c))

    return {name: result[name] for name in piece_order if name in result}


def gen_step():
    return ordered_labeled_compound(
        pieces(), label="lx521_4_top_baffle_b2_split")
