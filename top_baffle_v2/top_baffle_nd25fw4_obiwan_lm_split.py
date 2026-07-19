"""Optional no-extra-fastener two-print split of the finalized Obi-Wan LM.

The authoritative ``core_lm_carrier`` remains a single printable solid.
This module derives a mutually-exclusive top/bottom print option from that
*final* carrier, after its route lumens, insert bores, captive magnet cavities and
state-specific no-floor bridge or integral-floor geometry have all been cut.

The seam is 28.5 mm below the LM centre.  That location makes both halves fit
a conservative 220 mm square print envelope in the common front-face-down
orientation, using only an in-bed Z rotation where required.  The seam stays
away from every LM insert and magnet axis.  Both buried cable
passages necessarily cross any useful top/bottom ring seam; deriving the
halves from the final BREP preserves their exact open lumen sections.

Two concealed cylindrical pins sit symmetrically on small exterior support
lands outside the LM driver recess and point normal to the horizontal seam
(world +Y).  The lands merge into the R110.6..R113 carrier lip and preserve a
0.05 mm plan clearance outside that recess.  A local roughly 1.40 mm perimeter
growth is required: putting the longer/wider sockets inward would collide the
hash-pinned W22 flange.  The right socket is round; the left
socket has the minimum X relief needed to avoid over-constraining the roughly
218.37 mm pin spacing.  This round-and-diamond constraint pattern registers
both halves without the binding risk of two tight round sockets, another
screw, or a driver-flange collision.  The required exterior lands create only
the documented local perimeter growth; the rest of the carrier is unchanged.
It receives zero installed structural-load credit: the LM driver flange and
its normal fasteners bridge the seam in service, and the split configuration
remains pending physical print/fit/load qualification.
"""

from __future__ import annotations

from build123d import Align, Box, Compound, Cylinder, Part, Pos, Rot
from math import pi, sqrt

from shapely.geometry import box

from top_baffle_nd25fw4_cables import ROUTING_PROFILE
from top_baffle_nd25fw4_obiwan import (
    CORE_REAR_Z,
    LM_CORE_R,
    LM_RECESS_R,
    LM_SEAT_Z,
    L22_CUTOUT,
    THICKNESS_MM,
    _fuse_attached,
    _plan_prism,
    lm_carrier,
)


if ROUTING_PROFILE != "obiwan":
    raise RuntimeError(
        "the optional Obi-Wan LM split requires LX_ROUTING_PROFILE=obiwan")


LM_SPLIT_TARGET_BED_MM = 220.0
LM_SPLIT_SEAM_OFFSET_Y = -28.5
LM_SPLIT_SEAM_Y = L22_CUTOUT[1] + LM_SPLIT_SEAM_OFFSET_Y
# The carrier faces are a closed planar butt joint.  A clearance kerf here
# would cut straight through both buried cable covers. Registration clearance
# is confined to two blind sockets inside the left/right structural lips.
LM_SPLIT_GAP_MM = 0.0

# Two identical cylindrical male pins replace the former one-sided tangential
# tongue.  Their axes are normal to the horizontal split seam: world +Y from
# bottom into top.  The pins are D1.60 (four nominal 0.40-mm nozzle widths) and
# engage 2.40 mm, three times the former 0.80-mm engagement.  That longer,
# wider fit cannot retain the required wall inside the bare 2.40-mm radial
# carrier lip, so each socket is housed by a tiny exterior land outside the LM
# driver recess.  This changes the perimeter locally by roughly 1.40 mm but
# avoids the installed-driver volume; an inward land was rejected after it
# intersected the hash-pinned W22 STEP.  The legacy 0.12-mm per-side fit
# clearance and 0.25-mm blind-end clearance are retained.  One
# socket is round and fully locating; the opposite socket is relieved by
# 0.06 mm along X only.  This classic round-and-diamond arrangement tolerates
# up to +/-0.30 mm relative pin-pitch error instead of making two widely
# spaced round fits fight one another, while preserving at least 0.50 mm of
# local socket wall and 0.50 mm behind each blind end.
REGISTRATION_PIN_DIAMETER_MM = 1.60
REGISTRATION_PIN_ENGAGEMENT_MM = 2.40
REGISTRATION_PIN_ROOT_OVERLAP_MM = 0.50
REGISTRATION_CENTER_Z_MM = 14.30
REGISTRATION_SOCKET_RADIAL_CLEAR_MM = 0.12
REGISTRATION_SOCKET_END_CLEAR_MM = 0.25
REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM = 0.06
REGISTRATION_SOCKET_BOOLEAN_OVERTRAVEL_MM = 0.05
REGISTRATION_MIN_RADIAL_WALL_MM = 0.50
REGISTRATION_SUPPORT_END_WALL_MM = 0.50
REGISTRATION_SUPPORT_RECESS_CLEAR_MM = 0.05
REGISTRATION_DRIVER_FLANGE_R_MM = 110.52
REGISTRATION_WING_CLEARANCE_MM = 0.25


def _registration_abs_x_mm() -> float:
    """Place both lands wholly outside the driver-recess authority."""
    _, cy = L22_CUTOUT[:2]
    support_start_y = (
        LM_SPLIT_SEAM_Y - REGISTRATION_PIN_ROOT_OVERLAP_MM)
    support_length = (
        REGISTRATION_PIN_ROOT_OVERLAP_MM
        + REGISTRATION_PIN_ENGAGEMENT_MM
        + REGISTRATION_SOCKET_END_CLEAR_MM
        + REGISTRATION_SUPPORT_END_WALL_MM)
    support_end_y = support_start_y + support_length
    dy = support_end_y - cy
    support_half_x = (
        REGISTRATION_PIN_DIAMETER_MM / 2.0
        + REGISTRATION_SOCKET_RADIAL_CLEAR_MM
        + REGISTRATION_MIN_RADIAL_WALL_MM
        + REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM)
    inner_authority_r = (
        LM_RECESS_R + REGISTRATION_SUPPORT_RECESS_CLEAR_MM)
    if abs(dy) >= inner_authority_r:
        raise RuntimeError(
            "Obi-Wan LM registration support misses the driver recess")
    x = sqrt(inner_authority_r ** 2 - dy ** 2) + support_half_x
    if x <= support_half_x:
        raise RuntimeError(
            "Obi-Wan LM registration support has no symmetric placement")
    return x


def registration_pin_centers_xyz():
    """Return symmetric left/right pin centres on the horizontal seam."""
    cx = L22_CUTOUT[0]
    x = _registration_abs_x_mm()
    return {
        "left": (cx - x, LM_SPLIT_SEAM_Y, REGISTRATION_CENTER_Z_MM),
        "right": (cx + x, LM_SPLIT_SEAM_Y, REGISTRATION_CENTER_Z_MM),
    }


def _y_axis_cylinder(x, start_y, z, radius, length):
    """Cylinder whose minimum axial face is at ``start_y`` and points +Y."""
    return (
        Pos(x, start_y, z)
        * Rot(X=-90.0)
        * Cylinder(
            radius, length,
            align=(Align.CENTER, Align.CENTER, Align.MIN)))


def _y_axis_capsule(x, start_y, z, radius, length, x_relief):
    """Blind +Y tool with an X-relieved capsule cross-section."""
    if x_relief <= 0.0:
        return _y_axis_cylinder(x, start_y, z, radius, length)
    left = _y_axis_cylinder(
        x - x_relief, start_y, z, radius, length)
    right = _y_axis_cylinder(
        x + x_relief, start_y, z, radius, length)
    bridge = (
        Pos(x, start_y + length / 2.0, z)
        * Box(
            2.0 * x_relief, length, 2.0 * radius,
            align=(Align.CENTER, Align.CENTER, Align.CENTER)))
    return left.fuse(bridge, right).clean()


def male_registration_pin_tools() -> dict:
    """Two identical symmetric cylindrical pins, both pointing world +Y."""
    radius = REGISTRATION_PIN_DIAMETER_MM / 2.0
    start_y = LM_SPLIT_SEAM_Y - REGISTRATION_PIN_ROOT_OVERLAP_MM
    length = (REGISTRATION_PIN_ROOT_OVERLAP_MM
              + REGISTRATION_PIN_ENGAGEMENT_MM)
    return {
        side: _y_axis_cylinder(x, start_y, z, radius, length)
        for side, (x, _, z) in registration_pin_centers_xyz().items()
    }


def female_registration_socket_tools() -> dict:
    """One round locator plus one minimally X-relieved blind socket."""
    radius = (REGISTRATION_PIN_DIAMETER_MM / 2.0
              + REGISTRATION_SOCKET_RADIAL_CLEAR_MM)
    start_y = (LM_SPLIT_SEAM_Y
               - REGISTRATION_SOCKET_BOOLEAN_OVERTRAVEL_MM)
    length = (REGISTRATION_PIN_ENGAGEMENT_MM
              + REGISTRATION_SOCKET_END_CLEAR_MM
              + REGISTRATION_SOCKET_BOOLEAN_OVERTRAVEL_MM)
    tools = {}
    for side, (x, _, z) in registration_pin_centers_xyz().items():
        relief = (REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM
                  if side == "left" else 0.0)
        tools[side] = _y_axis_capsule(
            x, start_y, z, radius, length, relief)
    return tools


def registration_support_land_tools() -> dict:
    """Local socket lands that grow outward from the existing LM lip."""
    radius = (
        REGISTRATION_PIN_DIAMETER_MM / 2.0
        + REGISTRATION_SOCKET_RADIAL_CLEAR_MM
        + REGISTRATION_MIN_RADIAL_WALL_MM)
    start_y = LM_SPLIT_SEAM_Y - REGISTRATION_PIN_ROOT_OVERLAP_MM
    length = (
        REGISTRATION_PIN_ROOT_OVERLAP_MM
        + REGISTRATION_PIN_ENGAGEMENT_MM
        + REGISTRATION_SOCKET_END_CLEAR_MM
        + REGISTRATION_SUPPORT_END_WALL_MM)
    tools = {}
    for side, (x, _, z) in registration_pin_centers_xyz().items():
        relief = (
            REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM
            if side == "left" else 0.0)
        tools[side] = _y_axis_capsule(
            x, start_y, z, radius, length, relief)
    return tools


def registration_wing_clearance_tools() -> dict:
    """Symmetric Ac/Ae interface pockets around either support-land form.

    The left socket land is 0.06 mm wider in X than the round right land.  Both
    wing pockets deliberately use that worst-case relieved capsule so the
    finalized right wing can remain the exact mirror authority for the left.
    The 0.25-mm offset is applied radially and at both axial ends; the pocket
    remains wholly between the acoustic front and rear faces.
    """
    clearance = REGISTRATION_WING_CLEARANCE_MM
    radius = (
        REGISTRATION_PIN_DIAMETER_MM / 2.0
        + REGISTRATION_SOCKET_RADIAL_CLEAR_MM
        + REGISTRATION_MIN_RADIAL_WALL_MM
        + clearance)
    start_y = (
        LM_SPLIT_SEAM_Y - REGISTRATION_PIN_ROOT_OVERLAP_MM - clearance)
    length = (
        REGISTRATION_PIN_ROOT_OVERLAP_MM
        + REGISTRATION_PIN_ENGAGEMENT_MM
        + REGISTRATION_SOCKET_END_CLEAR_MM
        + REGISTRATION_SUPPORT_END_WALL_MM
        + 2.0 * clearance)
    return {
        side: _y_axis_capsule(
            x, start_y, z, radius, length,
            REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM)
        for side, (x, _, z) in registration_pin_centers_xyz().items()
    }


def registration_support_land_tool():
    """Compatibility aggregate for the two driver-clear exterior lands."""
    return Compound(children=list(registration_support_land_tools().values()))


def male_registration_key_tool():
    """Compatibility aggregate for the two cylindrical male pins."""
    return Compound(children=list(male_registration_pin_tools().values()))


def female_registration_socket_tool():
    """Compatibility aggregate for the two blind female sockets."""
    return Compound(children=list(
        female_registration_socket_tools().values()))


def registration_augmented_carrier(source):
    """Add only the two driver-clear exterior registration lands."""
    carrier = _one_solid(source, "source LM carrier")
    for side, support in registration_support_land_tools().items():
        carrier = _fuse_attached(
            carrier, support,
            f"optional LM concealed {side} registration support land")
    return _one_solid(carrier, "registration-augmented LM carrier")


def registration_fit_facts() -> dict:
    """Exact pure-math design facts for the exterior-land two-pin registration."""
    centers = registration_pin_centers_xyz()
    cx, cy = L22_CUTOUT[:2]
    x = _registration_abs_x_mm()
    support_start_y = (
        LM_SPLIT_SEAM_Y - REGISTRATION_PIN_ROOT_OVERLAP_MM)
    socket_depth = (REGISTRATION_PIN_ENGAGEMENT_MM
                    + REGISTRATION_SOCKET_END_CLEAR_MM)
    pin_r = REGISTRATION_PIN_DIAMETER_MM / 2.0
    round_r = pin_r + REGISTRATION_SOCKET_RADIAL_CLEAR_MM
    relieved_half_x = (
        round_r + REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM)
    support_r = round_r + REGISTRATION_MIN_RADIAL_WALL_MM
    support_half_x = (
        support_r + REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM)
    support_length = (
        REGISTRATION_PIN_ROOT_OVERLAP_MM
        + REGISTRATION_PIN_ENGAGEMENT_MM
        + REGISTRATION_SOCKET_END_CLEAR_MM
        + REGISTRATION_SUPPORT_END_WALL_MM)
    support_end_y = support_start_y + support_length
    support_z_min = REGISTRATION_CENTER_Z_MM - support_r
    support_z_max = REGISTRATION_CENTER_Z_MM + support_r
    if support_z_min <= LM_SEAT_Z:
        raise RuntimeError(
            "Obi-Wan LM registration land reaches into the driver recess")
    if support_z_min <= CORE_REAR_Z:
        raise RuntimeError(
            "Obi-Wan LM registration land reaches the rear exterior")
    if support_z_max >= THICKNESS_MM:
        raise RuntimeError(
            "Obi-Wan LM registration land reaches the front exterior")

    # The support cross-sections are exact 0.50-mm offsets of their socket
    # cross-sections.  The far-end relieved land is the point closest to the
    # driver recess; the root outer point governs local plan-envelope growth.
    support_root_dy = support_start_y - cy
    support_end_dy = support_end_y - cy
    recess_clearance = sqrt(
        (x - support_half_x) ** 2 + support_end_dy ** 2
    ) - LM_RECESS_R
    outer_land_r = sqrt(
        (x + support_half_x) ** 2 + support_root_dy ** 2)
    plan_outline_growth = outer_land_r - LM_CORE_R
    driver_flange_clearance = (
        LM_RECESS_R + recess_clearance
        - REGISTRATION_DRIVER_FLANGE_R_MM)
    min_wall = min(
        REGISTRATION_MIN_RADIAL_WALL_MM,
        REGISTRATION_SUPPORT_END_WALL_MM)
    if min_wall < REGISTRATION_MIN_RADIAL_WALL_MM:
        raise RuntimeError(
            "Obi-Wan LM relieved registration socket leaves only "
            f"{min_wall:.3f} mm radial wall")
    pin_length = (REGISTRATION_PIN_ROOT_OVERLAP_MM
                  + REGISTRATION_PIN_ENGAGEMENT_MM)
    round_area = pi * round_r ** 2
    relieved_area = (
        round_area
        + 4.0 * round_r * REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM)
    return {
        "split_kind": "optional_top_bottom_hidden_two_pin_registration",
        "seam_y_mm": LM_SPLIT_SEAM_Y,
        "seam_offset_from_lm_center_mm": LM_SPLIT_SEAM_OFFSET_Y,
        "assembly_gap_mm": LM_SPLIT_GAP_MM,
        "buried_route_joint": "closed_zero_gap_planar_butt",
        "registration_pair_count": 2,
        "registration_sides": ("left", "right"),
        "registration_is_keyed": True,
        "registration_form": (
            "two_symmetric_cylindrical_pins_round_plus_relief_sockets"),
        "registration_centers_xyz_mm": centers,
        "registration_center_spacing_mm": 2.0 * x,
        "registration_symmetry_error_mm": abs(
            centers["left"][0] + centers["right"][0] - 2.0 * cx),
        "registration_axis_world_xyz": (0.0, 1.0, 0.0),
        "registration_axis_normal_to_horizontal_seam": True,
        "assembly_motion": (
            "top_half_approaches_along_negative_world_y"),
        "pin_diameter_mm": REGISTRATION_PIN_DIAMETER_MM,
        "pin_root_overlap_mm": REGISTRATION_PIN_ROOT_OVERLAP_MM,
        "male_pin_length_mm": pin_length,
        "male_total_volume_mm3": (
            2.0 * pi * pin_r ** 2 * pin_length),
        "engagement_depth_mm": REGISTRATION_PIN_ENGAGEMENT_MM,
        "socket_round_diameter_mm": 2.0 * round_r,
        "socket_radial_clearance_mm": (
            REGISTRATION_SOCKET_RADIAL_CLEAR_MM),
        "socket_end_clearance_mm": REGISTRATION_SOCKET_END_CLEAR_MM,
        "socket_blind_depth_mm": socket_depth,
        "round_socket_side": "right",
        "relieved_socket_side": "left",
        "relieved_socket_x_extra_each_side_mm": (
            REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM),
        "relieved_socket_x_span_mm": 2.0 * relieved_half_x,
        "round_socket_cross_section_area_mm2": round_area,
        "relieved_socket_cross_section_area_mm2": relieved_area,
        "registered_round_diametral_play_mm": (
            2.0 * REGISTRATION_SOCKET_RADIAL_CLEAR_MM),
        "relative_pin_pitch_error_capacity_mm": (
            2.0 * REGISTRATION_SOCKET_RADIAL_CLEAR_MM
            + REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM),
        "round_socket_inner_wall_mm": REGISTRATION_MIN_RADIAL_WALL_MM,
        "round_socket_outer_wall_mm": REGISTRATION_MIN_RADIAL_WALL_MM,
        "relieved_socket_inner_wall_mm": REGISTRATION_MIN_RADIAL_WALL_MM,
        "relieved_socket_outer_wall_mm": REGISTRATION_MIN_RADIAL_WALL_MM,
        "minimum_socket_radial_wall_mm": min_wall,
        "socket_blind_end_wall_mm": REGISTRATION_SUPPORT_END_WALL_MM,
        "support_land_length_mm": support_length,
        "support_land_z_range_mm": (support_z_min, support_z_max),
        "support_land_clearance_above_rear_mm": (
            support_z_min - CORE_REAR_Z),
        "support_land_clearance_below_front_mm": (
            THICKNESS_MM - support_z_max),
        "support_land_driver_recess_plan_clearance_mm": recess_clearance,
        "support_land_driver_flange_plan_clearance_mm": (
            driver_flange_clearance),
        "wing_interface_clearance_mm": REGISTRATION_WING_CLEARANCE_MM,
        "wing_clearance_compatible_variants": ("ac", "ae"),
        "wing_clearance_pocket_between_front_and_rear": True,
        "exterior_support_land": True,
        "support_land_plan_outline_growth_mm": plan_outline_growth,
        "inward_support_land_rejected_for_driver_collision": True,
        "two_round_socket_design_rejected": True,
        "tolerance_strategy": (
            "right_round_locator_left_x_relief_round_and_diamond"),
        "binding_drawback": (
            "two round sockets across the wide pitch can bind from XY scale "
            "or shrink error; the left X relief removes only that redundant "
            "constraint while retaining Z and angular registration"),
        "nominal_nozzle_diameter_mm": 0.40,
        "pin_nominal_nozzle_width_count": (
            REGISTRATION_PIN_DIAMETER_MM / 0.40),
        "pin_and_socket_slicer_gate_required": True,
        "printability_drawback": (
            "the horizontal D1.6 pins are four nominal 0.4-mm nozzle widths "
            "but have no load credit; verify both pin toolpaths, the local "
            "support lands and the minimum socket walls before release"),
        "envelope_growth_mm": plan_outline_growth,
        "target_square_bed_mm": LM_SPLIT_TARGET_BED_MM,
        "floor_bottom_print_rotation_x_deg": 180.0,
        "print_orientation": "front_face_down_all_pieces",
        "floor_bottom_in_bed_rotation_deg": 26.0,
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
    carrier = registration_augmented_carrier(carrier)
    # Floor mode owns the complete z=-150 connector foot.  The former
    # +/-100 clip silently truncated it before the optional split.
    clip_z = (-200.0, 100.0)
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

    # Each pin is ownership reassigned to the bottom half from the augmented
    # optional-split carrier.  Requiring every complete tool inside that
    # carrier catches any future route/magnet/seat change that would nick a
    # concealed pin.
    for side, male_tool in male_registration_pin_tools().items():
        outside_source = male_tool - carrier
        outside_volume = 0.0 if outside_source is None else sum(
            solid.volume for solid in outside_source.solids())
        if outside_volume > 0.02:
            raise RuntimeError(
                f"{side} male registration pin escaped the monolithic LM by "
                f"{outside_volume:.6f} mm3")
        bottom = _fuse_attached(
            bottom, male_tool,
            f"optional LM concealed {side} male registration pin")

    for socket_tool in female_registration_socket_tools().values():
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
    assembly.label = "lx521_obiwan_r6f_optional_lm_keyed_split"
    return assembly
