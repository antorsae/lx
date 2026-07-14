"""Optional no-extra-fastener two-print split of the finalized V1LF LM.

The authoritative ``core_lm_carrier`` remains a single printable solid.
This module derives a mutually-exclusive top/bottom print option from that
*final* carrier, after its route lumens, insert bores, magnet pockets and
state-specific bridge/support interfaces have all been cut.

The seam is 28.5 mm below the LM centre.  That location makes both halves fit
a conservative 220 mm square print envelope after an in-plane rotation and
keeps the seam away from every LM insert and magnet axis.  Both buried cable
passages necessarily cross any useful top/bottom ring seam; deriving the
halves from the final BREP preserves their exact open lumen sections.

One concealed keyed male/female registration feature sits wholly inside the
right-hand R110.6..R113 carrier lip.  It aligns the loose halves without
another screw or any growth beyond the monolithic envelope.  It receives
zero installed structural-load credit: the LM driver flange and its normal
fasteners bridge the seam in service, and the split configuration remains
pending physical print/fit/load qualification.
"""

from __future__ import annotations

from build123d import Compound, Part
from math import atan2, degrees, sqrt

from shapely.affinity import translate
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from top_baffle_nd25fw4_cables import ROUTING_PROFILE
from top_baffle_nd25fw4_v1lf import (
    LM_CORE_R,
    LM_RECESS_R,
    L22_CUTOUT,
    _fuse_attached,
    _plan_prism,
    lm_carrier,
)


if ROUTING_PROFILE != "v1lf":
    raise RuntimeError(
        "the optional V1LF LM split requires LX_ROUTING_PROFILE=v1lf")


LM_SPLIT_TARGET_BED_MM = 220.0
LM_SPLIT_SEAM_OFFSET_Y = -28.5
LM_SPLIT_SEAM_Y = L22_CUTOUT[1] + LM_SPLIT_SEAM_OFFSET_Y
# The carrier faces are a closed planar butt joint.  A clearance kerf here
# would cut straight through both buried cable covers. Registration clearance
# is confined to one blind socket inside the right-hand structural lip.
LM_SPLIT_GAP_MM = 0.0

# A single straight, rounded tongue provides the requested internal joint.
# Its centreline follows the local tangent of the right-hand lip, so the top
# half has one unambiguous straight-pull assembly motion.  Male and socket
# both stay outside the R110.6 driver recess, inside the R113 outer envelope,
# and above/outboard of both buried route crowns.  The female relief remains
# blind to both radial faces.
REGISTRATION_CENTER_R_MM = 111.80
REGISTRATION_TONGUE_WIDTH_MM = 0.80
REGISTRATION_TONGUE_S_MM = (-1.00, 3.50)
REGISTRATION_MALE_Z = (12.00, 16.60)
REGISTRATION_SOCKET_PLAN_CLEAR_MM = 0.12
REGISTRATION_SOCKET_END_CLEAR_MM = 0.25
REGISTRATION_SOCKET_Z_CLEAR_MM = 0.15
REGISTRATION_DRIVER_FLANGE_R_MM = 110.52


def _registration_basis():
    """Return point, radial unit and straight-pull tangent unit vectors."""
    cx, cy = L22_CUTOUT[:2]
    dy = LM_SPLIT_SEAM_Y - cy
    if abs(dy) >= REGISTRATION_CENTER_R_MM:
        raise RuntimeError("V1LF LM registration radius misses the seam")
    dx = sqrt(REGISTRATION_CENTER_R_MM ** 2 - dy ** 2)
    radial = (dx / REGISTRATION_CENTER_R_MM,
              dy / REGISTRATION_CENTER_R_MM)
    tangent = (-radial[1], radial[0])
    return (cx + dx, LM_SPLIT_SEAM_Y), radial, tangent


def male_registration_key_plan():
    """One straight rounded tongue wholly inside the existing LM lip."""
    point, _, tangent = _registration_basis()
    radius = REGISTRATION_TONGUE_WIDTH_MM / 2.0
    # Pull the LineString endpoints inward by the cap radius so the finished
    # capsule occupies exactly the specified S interval.
    endpoints = []
    for s in (REGISTRATION_TONGUE_S_MM[0] + radius,
              REGISTRATION_TONGUE_S_MM[1] - radius):
        endpoints.append((
            point[0] + s * tangent[0],
            point[1] + s * tangent[1],
        ))
    plan = LineString(endpoints).buffer(
        radius, resolution=32, cap_style=1, join_style=1)
    if plan.geom_type != "Polygon" or not plan.is_valid:
        raise RuntimeError(
            "V1LF LM male registration key must be one plan polygon")
    return plan


def female_registration_socket_plan():
    """Printable female relief swept along the straight-pull axis."""
    male = male_registration_key_plan()
    _, _, tangent = _registration_basis()
    plan = unary_union((
        male,
        translate(
            male,
            xoff=REGISTRATION_SOCKET_END_CLEAR_MM * tangent[0],
            yoff=REGISTRATION_SOCKET_END_CLEAR_MM * tangent[1],
        ),
    )).buffer(
        REGISTRATION_SOCKET_PLAN_CLEAR_MM,
        resolution=24, cap_style=1, join_style=1)
    if plan.geom_type != "Polygon" or not plan.is_valid:
        raise RuntimeError(
            "V1LF LM female registration socket must be one plan polygon")
    return plan


def male_registration_key_tool():
    return _plan_prism(
        male_registration_key_plan(), *REGISTRATION_MALE_Z)


def female_registration_socket_tool():
    return _plan_prism(
        female_registration_socket_plan(),
        REGISTRATION_MALE_Z[0] - REGISTRATION_SOCKET_Z_CLEAR_MM,
        REGISTRATION_MALE_Z[1] + REGISTRATION_SOCKET_Z_CLEAR_MM)


def registration_fit_facts() -> dict:
    """Exact design facts for the one concealed male/female registration."""
    male = male_registration_key_plan()
    socket = female_registration_socket_plan()
    center = Point(*L22_CUTOUT[:2])
    socket_radii = tuple(
        center.distance(Point(float(x), float(y)))
        for x, y in socket.exterior.coords)
    point, radial, tangent = _registration_basis()
    top_material = box(
        -400.0, LM_SPLIT_SEAM_Y, 400.0, 600.0).difference(socket)
    insertion_overlaps = []
    # In top-half coordinates the stationary bottom tongue moves opposite
    # the separation direction.  This sampled contract proves that one
    # straight tangential motion clears the female mouth with no undercut.
    for step in range(41):
        offset = 5.0 * step / 40.0
        moving_male = translate(
            male, xoff=-offset * tangent[0],
            yoff=-offset * tangent[1])
        insertion_overlaps.append(moving_male.intersection(
            top_material).area)
    return {
        "split_kind": "optional_top_bottom_hidden_keyed_registration",
        "seam_y_mm": LM_SPLIT_SEAM_Y,
        "seam_offset_from_lm_center_mm": LM_SPLIT_SEAM_OFFSET_Y,
        "assembly_gap_mm": LM_SPLIT_GAP_MM,
        "buried_route_joint": "closed_zero_gap_planar_butt",
        "registration_pair_count": 1,
        "registration_side": "right",
        "registration_is_keyed": True,
        "registration_form": "single_straight_rounded_tongue",
        "registration_center_xy_mm": point,
        "registration_radial_unit": radial,
        "registration_insertion_unit": tangent,
        "registration_insertion_angle_deg": degrees(atan2(
            tangent[1], tangent[0])),
        "assembly_motion": (
            "top_half_approaches_along_negative_registration_axis"),
        "registration_tongue_width_mm": REGISTRATION_TONGUE_WIDTH_MM,
        "male_plan_area_mm2": male.area,
        "male_z_height_mm": (
            REGISTRATION_MALE_Z[1] - REGISTRATION_MALE_Z[0]),
        "male_volume_mm3": male.area * (
            REGISTRATION_MALE_Z[1] - REGISTRATION_MALE_Z[0]),
        "engagement_depth_mm": (
            REGISTRATION_TONGUE_S_MM[1]),
        "socket_plan_clearance_mm": (
            REGISTRATION_SOCKET_PLAN_CLEAR_MM),
        "socket_end_clearance_mm": REGISTRATION_SOCKET_END_CLEAR_MM,
        "socket_z_clearance_mm": REGISTRATION_SOCKET_Z_CLEAR_MM,
        "registered_plan_play_mm": (
            2.0 * REGISTRATION_SOCKET_PLAN_CLEAR_MM),
        "registered_z_play_mm": (
            2.0 * REGISTRATION_SOCKET_Z_CLEAR_MM),
        "socket_inner_wall_mm": min(socket_radii) - LM_RECESS_R,
        "socket_outer_wall_mm": LM_CORE_R - max(socket_radii),
        "driver_radial_clearance_mm": (
            min(socket_radii) - REGISTRATION_DRIVER_FLANGE_R_MM),
        "sampled_insertion_max_plan_overlap_mm2": max(
            insertion_overlaps),
        "envelope_growth_mm": 0.0,
        "target_square_bed_mm": LM_SPLIT_TARGET_BED_MM,
        "installed_structural_load_credit_n": 0.0,
        "standalone_retention_credit_n": 0.0,
        "z_registration": "assemble_front_faces_on_flat_datum_then_driver",
        "physical_coupon_required": True,
        "physical_load_qualification_required": True,
    }


def _one_solid(shape, label: str):
    shape = shape.clean()
    solids = list(shape.solids())
    if (not shape.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            f"{label}: expected one valid solid, got valid={shape.is_valid} "
            f"volumes={[solid.volume for solid in solids]}")
    return Part([solids[0]])


def lm_carrier_split_parts(source=None) -> dict:
    """Derive the optional bottom/top pair from one final LM carrier BREP."""
    carrier = source if source is not None else lm_carrier()
    carrier = _one_solid(carrier, "source LM carrier")
    clip_z = (-100.0, 100.0)
    bottom_region = box(
        -400.0, -400.0, 400.0,
        LM_SPLIT_SEAM_Y - LM_SPLIT_GAP_MM / 2.0)
    top_region = box(
        -400.0, LM_SPLIT_SEAM_Y + LM_SPLIT_GAP_MM / 2.0,
        400.0, 600.0)
    bottom = _one_solid(
        carrier & _plan_prism(bottom_region, *clip_z),
        "optional LM bottom base")
    top = _one_solid(
        carrier & _plan_prism(top_region, *clip_z),
        "optional LM top base")

    male_tool = male_registration_key_tool()
    # The key is source ownership reassigned to the bottom half, never new
    # material. Requiring the complete tool inside the final carrier catches
    # any future route/magnet/seat change that would nick the concealed key.
    outside_source = male_tool - carrier
    outside_volume = 0.0 if outside_source is None else sum(
        solid.volume for solid in outside_source.solids())
    if outside_volume > 0.02:
        raise RuntimeError(
            "male registration key escaped the monolithic LM by "
            f"{outside_volume:.6f} mm3")
    bottom = _fuse_attached(
        bottom, male_tool, "optional LM concealed male registration key")

    socket_tool = female_registration_socket_tool()
    top -= socket_tool

    bottom = _one_solid(bottom, "optional LM keyed bottom")
    top = _one_solid(top, "optional LM keyed top")
    collision = bottom & top
    collision_volume = 0.0 if collision is None else sum(
        solid.volume for solid in collision.solids())
    if collision_volume > 0.02:
        raise RuntimeError(
            "optional LM keyed halves overlap by "
            f"{collision_volume:.6f} mm3")
    return {
        "optional_lm_keyed_1of2_bottom": bottom,
        "optional_lm_keyed_2of2_top": top,
    }


def gen_step():
    children = []
    for label, solid in lm_carrier_split_parts().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_v1lf_r6f_optional_lm_keyed_split"
    return assembly
