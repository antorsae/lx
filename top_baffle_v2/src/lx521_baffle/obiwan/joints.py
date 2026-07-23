"""Obi-Wan LM/UM and tweeter joint plans, solids and load contracts."""

from __future__ import annotations

import math

from pathlib import Path

from build123d import (
    Box,
    Compound,
    Cylinder,
    Face,
    Part,
    Polyline,
    Pos,
    Rot,
    Wire,
    extrude,
)

from shapely.geometry import LineString, Point, Polygon, box

from shapely.geometry.polygon import orient

from shapely.ops import unary_union

from ..assembly import ordered_labeled_compound

from ..base import (
    L22_CUTOUT,
    L22_PILOT_D_MM,
    STAND_FOOT,
    THICKNESS_MM,
    UM_CUTOUT,
    UM_PILOT_ANGLES_DEG,
    UM_PILOT_DEPTH_MM,
    UM_PILOT_D_MM,
    UM_TERMINAL_CLOCK_DEG,
)

from ..flush import (
    LM_BORE_DEPTH_MM,
    LM_PILOT_XY,
    LM_RECESS_R,
    LM_SEAT_Z,
    PAD_D_MM,
    PAD_FACE_Z,
    UM_PILOT_XY,
    UM_PAD_D_MM,
    UM_PAD_FLOOR_MM,
    UM_RECESS_R,
    UM_SEAT_Z,
)

from . import route

from .route import (
    CROSSOVER_T_Z,
    TS_CUTTER_R,
    TUNNEL_ROOF_SKIN,
    lm_rear_exit_port_cutter,
    no_floor_rear_entry_transition_cutters,
    route_inner_cutter_group,
    route_inner_cutter_group_count,
    route_inner_cutters,
    route_outer_covers,
)

from .bridge import (
    floor_wing_contact_profile_addition,
    fused_bridge_tail,
)

from .floor import (
    apply_integrated_floor_feature_group,
    integrated_floor_addition,
    integrated_floor_feature_group_count,
)

from ..magnets import (
    CAPTIVE_LAND_MM,
    CAVITY_DEPTH_MM,
    CAVITY_DIAMETER_MM,
    FACE_SKIN_MM,
    INNER_SKIN_MM,
    INTERFACE_GAP_MM,
    wall_cavity_tools,
)


from .carriers import (
    CORE_CENTER_SPACING,
    CORE_REAR_Z,
    CORE_RING_GAP,
    JOINT_BORE_REAR_OVERSHOOT,
    JOINT_BOSS_D,
    JOINT_CLEARANCE_BORE_D,
    JOINT_CLEARANCE_BORE_TOP_Z,
    JOINT_CONTACT_LEVER_FACTOR,
    JOINT_DESIGN_MASS_KG,
    JOINT_EAR_X,
    JOINT_EAR_Y,
    JOINT_FUNCTIONAL_BOSS_D,
    JOINT_INSERT_BORE_D,
    JOINT_INSERT_BORE_Z,
    JOINT_INSERT_DEPTH_MM,
    JOINT_INSERT_FRONT_FLOOR_MM,
    JOINT_M3_SHEAR_ALLOW_MPA,
    JOINT_M3_TENSION_ALLOW_MPA,
    JOINT_NECK_D,
    JOINT_PLAN_LEVER_MM,
    JOINT_PLA_CREEP_ALLOW_MPA,
    JOINT_PLA_SHORT_ALLOW_MPA,
    JOINT_REAR_LEVER_MM,
    JOINT_RECEIVER_RADIAL_CLEAR,
    JUNCTION_WEB_EAR_CHORD_INSET,
    JUNCTION_WEB_EAR_CLEAR,
    JUNCTION_WEB_LENS_FUSION_MM,
    JUNCTION_WEB_MIN_LENS_AREA_MM2,
    JUNCTION_WEB_OWNER_OVERLAP,
    JUNCTION_WEB_SAMPLES,
    JUNCTION_WEB_SEAM_GAP,
    JUNCTION_WEB_Z,
    LM_BASE_MAGNET_FACE_X,
    LM_BASE_MAGNET_Y,
    LM_BASE_MAGNET_Z,
    LM_CORE_R,
    LM_JOINT_Z,
    LM_STRUCT_SPOKE_W,
    LM_T_CLOSURE_HANDOFF_RADIAL_INSET_MM,
    LM_T_CLOSURE_HANDOFF_RADIAL_OUTSET_MM,
    LM_T_CLOSURE_HANDOFF_RELIEF_MM,
    LM_UM_REAR_BACKFILL_ARC_HALF_SPAN_DEG,
    LM_UM_REAR_BACKFILL_CENTER_ANGLES_DEG,
    LM_UM_REAR_BACKFILL_CENTER_R,
    LM_UM_REAR_BACKFILL_RADIAL_WIDTH_MM,
    LM_UM_REAR_BACKFILL_Z,
    LM_UM_WEB_BLEND_START_X,
    LM_UM_WEB_HALF_WIDTH,
    LM_VISIBLE_RING_R,
    OBIWAN_MAGNET_Z_MM,
    PRINT_ORIENTATION,
    SEAT_MEMBRANE_LIP_OVERLAP,
    SEAT_MEMBRANE_T,
    SIDE_EAR_D,
    SIDE_EAR_IN,
    SIDE_EAR_OUT,
    SIDE_INTERFACE_GAP,
    SIDE_MAGNET_ANGLES,
    SIDE_MAGNET_CAPTIVE_LAND,
    SIDE_MAGNET_D,
    SIDE_MAGNET_DEPTH,
    SIDE_MAGNET_FACE_OFFSET,
    SIDE_MAGNET_FACE_SKIN,
    SIDE_MAGNET_INNER_SKIN,
    SIDE_MAGNET_POCKET_D,
    SIDE_MAGNET_Z,
    SIDE_RING_CAVITY_FACE_INSET_MM,
    SIDE_RING_CAVITY_FACE_OFFSET_MM,
    SIDE_RING_CAVITY_RECESS_CLEAR_MM,
    SIDE_RING_FAIRING_FUSION_OVERLAP_MM,
    SIDE_RING_FLUSH_FAIRING_MM,
    STRUCT_CREEP_ALLOW_MPA,
    STRUCT_DESIGN_MASS_KG,
    STRUCT_SHORT_ALLOW_MPA,
    TWEETER_ADDON_JOINT_Z,
    TWEETER_CORE_BORE_TOP_Z,
    TWEETER_CORE_JOINT_Z,
    TWEETER_JOINT_BOSS_D,
    TWEETER_JOINT_CLEAR,
    TWEETER_JOINT_FUNCTIONAL_BOSS_D,
    TWEETER_JOINT_HOLE_D,
    TWEETER_JOINT_INSERT_BORE_D,
    TWEETER_JOINT_INSERT_BORE_Z,
    TWEETER_JOINT_INSERT_DEPTH_MM,
    TWEETER_JOINT_INSERT_FRONT_FLOOR_MM,
    TWEETER_JOINT_NECK_D,
    TWEETER_JOINT_X,
    TWEETER_JOINT_Y,
    T_CRESCENT_ARC_CENTER,
    T_CRESCENT_ARC_R,
    T_UM_CABLE_MOUTH_HALF_WIDTH,
    T_UM_WEB_BLEND_START_X,
    T_UM_WEB_OUTER_X,
    UM_CORE_R,
    UM_INSERT_BOSS_D,
    UM_JOINT_EAR_SPOKE_CLEAR_MM,
    UM_JOINT_TUNNEL_LIGAMENT,
    UM_JOINT_Z,
    UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z,
    UM_PILOT_RECESS_CLOSURE_LAND_DEPTH_MM,
    UM_PILOT_RECESS_CLOSURE_LAND_EXPANSION_MM,
    UM_PILOT_SPOKE_TANGENTIAL_OFFSETS_DEG,
    UM_STRUCT_SPOKE_W,
    UM_T_FAIRING_CUSP_HALF_WIDTH_MM,
    UM_T_REAR_BACKFILL_ARC_HALF_SPAN_DEG,
    UM_T_REAR_BACKFILL_CENTER_ANGLES_DEG,
    UM_T_REAR_BACKFILL_CENTER_R,
    UM_T_REAR_BACKFILL_RADIAL_WIDTH_MM,
    UM_T_REAR_BACKFILL_Z,
    UM_VISIBLE_RING_R,
    _cut_lm_mount_holes,
    _cylinder_at,
    _ensure_shell_contained,
    _lm_t_closure_handoff_cutters,
    _minimal_ring_blank,
    _no_floor_cover_remainders,
    _plan_polygon_components,
    _plan_prism,
    _polar_xy,
    _radial_spoke,
    _radial_spoke_plan,
    _require_guarded_build,
    _side_ring_fairing,
    _subtract_plan_prisms,
    _um_pilot_recess_closure_land,
    apply_lm_route_cutter,
    carrier_spoke_load_facts,
    core_parts,
    finalize_lm_carrier,
    gen_step,
    lm_carrier,
    lm_carrier_outer_blank,
    side_ring_outer_plan,
    um_carrier,
    um_pilot_spoke_z_segments,
)

from .closure_webs import (
    _bounded_plan_lenses,
    _circle_branch_slope,
    _circle_branch_y,
    _closure_owner_bands,
    _cubic_point,
    _curve_band,
    _enforce_junction_plan_ownership,
    _junction_closure_web,
    _lm_um_rear_recess_backfill,
    _lm_um_rear_recess_backfill_plan,
    _mirrored_reversed,
    _owned_lens_addition,
    _partition_lens_components,
    _path_band,
    _path_owner_bands,
    _printable_lens_components,
    _t_crescent_boundary_y,
    _tangent_blend_to_boss,
    _terminal_fit_drains,
    _um_t_rear_recess_backfill,
    _um_t_rear_recess_backfill_plan,
    junction_closure_polygons,
    lm_um_closure_polygons,
    t_um_closure_polygons,
)


def joint_ear_polygon(owner: str, x: float, clearance: float = 0.0,
                      *, boss_d: float = JOINT_BOSS_D):
    """Exact rounded half-lap footprint for visual/tests/booleans.

    ``owner`` chooses the ring touched by the narrow neck.  A positive
    clearance buffers the *complete* outline for the mating receiver.
    """
    if owner == "lm":
        center, radius = L22_CUTOUT[:2], LM_CORE_R
    elif owner == "um":
        center, radius = UM_CUTOUT[:2], UM_CORE_R
    else:
        raise ValueError(owner)
    dx, dy = x - center[0], JOINT_EAR_Y - center[1]
    length = math.hypot(dx, dy)
    contact = (center[0] + radius * dx / length,
               center[1] + radius * dy / length)
    boss = Point(x, JOINT_EAR_Y).buffer(boss_d / 2.0,
                                        resolution=32)
    neck = Point(*contact).buffer(JOINT_NECK_D / 2.0, resolution=32)
    polygon = boss.union(neck).convex_hull
    return polygon.buffer(clearance, resolution=16) if clearance else polygon

def tweeter_joint_polygon(x: float, clearance: float = 0.0,
                          *, boss_d: float = TWEETER_JOINT_BOSS_D):
    """Compact direct ear footprint from the UM ring to the crescent."""
    center = UM_CUTOUT[:2]
    dx, dy = x - center[0], TWEETER_JOINT_Y - center[1]
    length = math.hypot(dx, dy)
    contact = (center[0] + UM_CORE_R * dx / length,
               center[1] + UM_CORE_R * dy / length)
    boss = Point(x, TWEETER_JOINT_Y).buffer(
        boss_d / 2.0, resolution=32)
    neck = Point(*contact).buffer(
        TWEETER_JOINT_NECK_D / 2.0, resolution=32)
    polygon = boss.union(neck).convex_hull
    return polygon.buffer(clearance, resolution=16) if clearance else polygon

def _joint_ear(owner: str, x: float, z_span, clearance: float = 0.0):
    return _plan_prism(joint_ear_polygon(owner, x, clearance), *z_span)

def _supported_plan_components(plan, support):
    """Keep whole printable components that overlap their massive owner."""
    kept = [
        piece for piece in _plan_polygon_components(plan)
        if piece.intersection(support).area > 1.0e-8
    ]
    if not kept:
        raise RuntimeError("ownership clipping detached every plan component")
    return unary_union(kept).buffer(0)

def _owned_joint_ear_plan(owner: str, x: float):
    """Diagnostic closure-clipped LM/UM ear plan.

    This is the footprint contributed by the full-depth closure-web base
    before the functional Z-owned joint is restored.  It is deliberately not
    the printable insert/bore authority: using it for a finished ear splits
    the cylindrical wall between the two independently printed carriers.
    """
    record = junction_closure_polygons()["lm_um"]
    blocked = unary_union((
        record["target"].difference(record[owner]),
        record["terminal_drain"],
    )).buffer(0)
    plan = joint_ear_polygon(owner, x).difference(blocked).buffer(0)
    center = L22_CUTOUT[:2] if owner == "lm" else UM_CUTOUT[:2]
    radius = LM_CORE_R if owner == "lm" else UM_CORE_R
    support = unary_union((
        Point(*center).buffer(radius, resolution=128),
        record[owner],
    )).buffer(0)
    return _supported_plan_components(plan, support)

def _complete_joint_ear_plan(owner: str, x: float,
                             clearance: float = 0.0):
    """Complete standalone LM/UM ear footprint, never split by the web seam."""
    return joint_ear_polygon(
        owner, x, clearance, boss_d=JOINT_FUNCTIONAL_BOSS_D)

def _complete_joint_ear(owner: str, x: float):
    """Complete Z-owned functional ear for one independently printed carrier."""
    if owner not in {"lm", "um"}:
        raise ValueError(owner)
    z_span = LM_JOINT_Z if owner == "lm" else UM_JOINT_Z
    return _plan_prism(_complete_joint_ear_plan(owner, x), *z_span)

def _owned_tweeter_joint_plan(owner: str, x: float,
                              clearance: float = 0.0):
    """Diagnostic closure-clipped T--UM base-ear footprint.

    The result describes only material contributed by the full-depth closure
    web.  It is deliberately not the printable bore/insert authority: using
    this plan for a finished half-ear lets the plan seam bisect the receiver.
    """
    if owner not in {"um", "tweeter"}:
        raise ValueError(owner)
    record = junction_closure_polygons()["t_um"]
    blocked = unary_union((
        record["target"].difference(record[owner]),
        record["terminal_drain"],
    )).buffer(0)
    if clearance > 0.0:
        return _owned_tweeter_joint_plan(owner, x).buffer(
            clearance, resolution=16).difference(blocked).buffer(0)
    plan = tweeter_joint_polygon(x, clearance).difference(blocked).buffer(0)
    if owner == "um":
        support = unary_union((
            Point(*UM_CUTOUT[:2]).buffer(UM_CORE_R, resolution=128),
            record["um"],
        )).buffer(0)
    else:
        support = unary_union((
            Point(*T_CRESCENT_ARC_CENTER).buffer(
                T_CRESCENT_ARC_R, resolution=128),
            record["tweeter"],
        )).buffer(0)
    return _supported_plan_components(plan, support)

def _complete_tweeter_joint_ear_plan(owner: str, x: float,
                                      clearance: float = 0.0):
    """Complete standalone T--UM ear, never split by closure ownership."""
    if owner not in {"um", "tweeter"}:
        raise ValueError(owner)
    return tweeter_joint_polygon(
        x, clearance, boss_d=TWEETER_JOINT_FUNCTIONAL_BOSS_D)

def _complete_tweeter_joint_ear(owner: str, x: float):
    """Complete Z-owned functional ear for one independently printed part."""
    if owner not in {"um", "tweeter"}:
        raise ValueError(owner)
    z_span = (TWEETER_CORE_JOINT_Z if owner == "um"
              else TWEETER_ADDON_JOINT_Z)
    return _plan_prism(
        _complete_tweeter_joint_ear_plan(owner, x), *z_span)

def joint_load_facts():
    """Upper-joint direct, moment-contact and M3 screen at 1/3/5g."""
    gravity = 9.80665
    full_half_thickness = min(
        LM_JOINT_Z[1] - LM_JOINT_Z[0],
        UM_JOINT_Z[1] - UM_JOINT_Z[0],
        TWEETER_CORE_JOINT_Z[1] - TWEETER_CORE_JOINT_Z[0],
        TWEETER_ADDON_JOINT_Z[1] - TWEETER_ADDON_JOINT_Z[0],
    )
    # The outward-shifted complete functional ears do not intersect the T
    # lumen (the exact cutter remains 0.445 mm away). Therefore the true
    # minimum axial ear thickness is 5.4 mm; the nearby 5.35-mm route-to-front
    # ligament is reported separately and receives no fictitious ear penalty.
    minimum_half_thickness = full_half_thickness
    pair_thickness = 2.0 * minimum_half_thickness
    neck_width = min(JOINT_NECK_D, TWEETER_JOINT_NECK_D)
    # The LM--UM interface is now insert-fastened, so its governing printed
    # annulus is set by the D4.6 receiver rather than the D3.4 screw passage.
    # Evaluate both reinforced D9.8 functional bosses separately for moment
    # contact.  In particular, the crescent half is governed by its D4.6
    # insert receiver, not by the core half's smaller D3.4 screw passage.
    lm_um_net_width = JOINT_FUNCTIONAL_BOSS_D - JOINT_INSERT_BORE_D
    tweeter_net_width = (
        TWEETER_JOINT_FUNCTIONAL_BOSS_D
        - TWEETER_JOINT_INSERT_BORE_D)
    net_width = min(lm_um_net_width, tweeter_net_width)
    bearing_width = min(
        JOINT_INSERT_BORE_D, TWEETER_JOINT_HOLE_D)
    neck_area = neck_width * pair_thickness
    net_area = net_width * pair_thickness
    bearing_area = bearing_width * pair_thickness
    bolt_shear_area = 2.0 * math.pi * (3.0 / 2.0) ** 2
    m3_tensile_area = math.pi * (2.53 / 2.0) ** 2
    contact_levers = {
        "lm_um": (JOINT_FUNCTIONAL_BOSS_D
                  * JOINT_CONTACT_LEVER_FACTOR),
        "um_tweeter": (TWEETER_JOINT_FUNCTIONAL_BOSS_D
                       * JOINT_CONTACT_LEVER_FACTOR),
    }
    contact_lever = min(contact_levers.values())
    moment_lever = math.hypot(
        JOINT_PLAN_LEVER_MM, JOINT_REAR_LEVER_MM)
    # Moment contact is governed by the single weaker ear in each interface.
    net_area_per_ear = {
        "lm_um": lm_um_net_width * minimum_half_thickness,
        "um_tweeter": tweeter_net_width * minimum_half_thickness,
    }

    def force(g_load):
        return JOINT_DESIGN_MASS_KG * gravity * g_load

    def facts(g_load, allowable):
        f = force(g_load)
        stresses = {
            "neck": f / neck_area,
            "net": f / net_area,
            "bearing": f / bearing_area,
        }
        return stresses, min(allowable / value for value in stresses.values())

    def moment_facts(g_load, allowable):
        moment = force(g_load) * moment_lever
        interfaces = {}
        for name in ("lm_um", "um_tweeter"):
            contact_force = moment / (2.0 * contact_levers[name])
            contact_stress = contact_force / net_area_per_ear[name]
            interfaces[name] = {
                "contact_lever_mm": contact_levers[name],
                "net_area_per_ear_mm2": net_area_per_ear[name],
                "contact_force_per_ear_n": contact_force,
                "contact_stress_mpa": contact_stress,
                "contact_sf": allowable / contact_stress,
            }
        governing_name = min(
            interfaces, key=lambda name: interfaces[name]["contact_sf"])
        governing = interfaces[governing_name]
        contact_force_per_ear = max(
            facts["contact_force_per_ear_n"]
            for facts in interfaces.values())
        m3_tension_stress = contact_force_per_ear / m3_tensile_area
        return {
            "moment_nmm": moment,
            "governing_interface": governing_name,
            "contact_force_per_ear_n": contact_force_per_ear,
            "contact_stress_mpa": governing["contact_stress_mpa"],
            "contact_sf": governing["contact_sf"],
            "m3_tension_stress_mpa": m3_tension_stress,
            "m3_tension_sf": (
                JOINT_M3_TENSION_ALLOW_MPA / m3_tension_stress),
            "lm_um_insert_pullout_required_n": (
                interfaces["lm_um"]["contact_force_per_ear_n"]),
            "um_tweeter_insert_pullout_required_n": (
                interfaces["um_tweeter"]["contact_force_per_ear_n"]),
            "interfaces": interfaces,
        }

    stress_1g, sf_1g = facts(1.0, JOINT_PLA_CREEP_ALLOW_MPA)
    stress_3g, sf_3g = facts(3.0, JOINT_PLA_SHORT_ALLOW_MPA)
    stress_5g, sf_5g = facts(5.0, JOINT_PLA_SHORT_ALLOW_MPA)
    moment_1g = moment_facts(1.0, JOINT_PLA_CREEP_ALLOW_MPA)
    moment_3g = moment_facts(3.0, JOINT_PLA_SHORT_ALLOW_MPA)
    moment_5g = moment_facts(5.0, JOINT_PLA_SHORT_ALLOW_MPA)
    return {
        "design_mass_kg": JOINT_DESIGN_MASS_KG,
        "creep_allow_mpa": JOINT_PLA_CREEP_ALLOW_MPA,
        "short_allow_mpa": JOINT_PLA_SHORT_ALLOW_MPA,
        "m3_shear_allow_mpa": JOINT_M3_SHEAR_ALLOW_MPA,
        "m3_tension_allow_mpa": JOINT_M3_TENSION_ALLOW_MPA,
        "plan_lever_mm": JOINT_PLAN_LEVER_MM,
        "rear_lever_mm": JOINT_REAR_LEVER_MM,
        "resultant_moment_lever_mm": moment_lever,
        "contact_lever_mm": contact_lever,
        "lm_um_functional_boss_d_mm": JOINT_FUNCTIONAL_BOSS_D,
        "um_tweeter_functional_boss_d_mm": (
            TWEETER_JOINT_FUNCTIONAL_BOSS_D),
        "lm_um_net_width_mm": lm_um_net_width,
        "um_tweeter_net_width_mm": tweeter_net_width,
        "minimum_half_thickness_mm": minimum_half_thickness,
        "full_half_thickness_mm": full_half_thickness,
        "nearby_route_front_ligament_mm": UM_JOINT_TUNNEL_LIGAMENT,
        "lm_um_clearance_bore_d_mm": JOINT_CLEARANCE_BORE_D,
        "lm_um_insert_receiver_d_mm": JOINT_INSERT_BORE_D,
        "lm_um_insert_receiver_depth_mm": JOINT_INSERT_DEPTH_MM,
        "lm_um_insert_front_floor_mm": JOINT_INSERT_FRONT_FLOOR_MM,
        "lm_um_axial_gap_mm": UM_JOINT_Z[0] - LM_JOINT_Z[1],
        "lm_um_bore_overlap_mm": (
            JOINT_CLEARANCE_BORE_TOP_Z - JOINT_INSERT_BORE_Z[0]),
        "lm_um_standalone_ear_ownership_required": True,
        "lm_um_full_360_wall_required": True,
        "lm_um_cross_owner_material_allowed": False,
        "lm_um_insert_pullout_qualification_required": True,
        "um_tweeter_clearance_bore_d_mm": TWEETER_JOINT_HOLE_D,
        "um_tweeter_insert_receiver_d_mm": TWEETER_JOINT_INSERT_BORE_D,
        "um_tweeter_insert_receiver_depth_mm": (
            TWEETER_JOINT_INSERT_DEPTH_MM),
        "um_tweeter_insert_front_floor_mm": (
            TWEETER_JOINT_INSERT_FRONT_FLOOR_MM),
        "um_tweeter_axial_gap_mm": (
            TWEETER_ADDON_JOINT_Z[0] - TWEETER_CORE_JOINT_Z[1]),
        "um_tweeter_bore_overlap_mm": (
            TWEETER_CORE_BORE_TOP_Z
            - TWEETER_JOINT_INSERT_BORE_Z[0]),
        "um_tweeter_standalone_ear_ownership_required": True,
        "um_tweeter_full_360_wall_required": True,
        "um_tweeter_cross_owner_material_allowed": False,
        "um_tweeter_insert_pullout_qualification_required": True,
        "neck_area_mm2": neck_area,
        "net_area_mm2": net_area,
        "bearing_area_mm2": bearing_area,
        "pla_stress_1g_mpa": stress_1g,
        "pla_stress_3g_mpa": stress_3g,
        "pla_stress_5g_mpa": stress_5g,
        "pla_sf_1g_creep": sf_1g,
        "pla_sf_3g": sf_3g,
        "pla_sf_5g": sf_5g,
        "m3_shear_stress_5g_mpa": force(5.0) / bolt_shear_area,
        "m3_shear_sf_5g": (
            JOINT_M3_SHEAR_ALLOW_MPA / (force(5.0) / bolt_shear_area)),
        "moment_1g": moment_1g,
        "moment_3g": moment_3g,
        "moment_5g": moment_5g,
        "magnet_load_credit_n": 0.0,
    }

def _complete_joint_receiver_notch(part, ear_owner: str, x: float):
    """Remove the full opposing ear so the functional boss has one owner.

    The notch spans the complete 0.20-mm axial air gap as well as the opposing
    ear.  This prevents either full-depth closure web from surviving across
    the nominal gap or bisecting the standalone bore/insert annulus.
    """
    if ear_owner == "lm":
        z0 = LM_JOINT_Z[0] - JOINT_RECEIVER_RADIAL_CLEAR
        z1 = UM_JOINT_Z[0]
    elif ear_owner == "um":
        z0 = LM_JOINT_Z[1]
        z1 = UM_JOINT_Z[1] + JOINT_RECEIVER_RADIAL_CLEAR
    else:
        raise ValueError(ear_owner)
    return _subtract_plan_prisms(
        part,
        _complete_joint_ear_plan(
            ear_owner, x, JOINT_RECEIVER_RADIAL_CLEAR),
        z0, z1)

def _fuse_attached(part, addition, label: str):
    """Fuse one required solid, always unifying same-domain faces."""
    before = part.volume
    added = addition.volume
    volume_tol = max(0.05, (before + added) * 1e-6)
    combined = part.fuse(addition).clean()
    solids = list(combined.solids())
    if (len(solids) == 1
            and combined.is_valid
            and solids[0].volume > 0.01
            and combined.volume >= before - volume_tol
            and combined.volume > before + min(0.05, added * 1e-4)
            and combined.volume <= before + added + volume_tol):
        return Part([solids[0]])
    raise RuntimeError(
        f"{label}: addition is detached or fusion failed; expected bounded "
        f"growth from {before:.3f} + {added:.3f} mm3; "
        f"valid={combined.is_valid} volumes="
        f"{[solid.volume for solid in combined.solids()]} bboxes="
        f"{[((solid.bounding_box().min.X, solid.bounding_box().min.Y,
              solid.bounding_box().min.Z),
             (solid.bounding_box().max.X, solid.bounding_box().max.Y,
              solid.bounding_box().max.Z))
            for solid in combined.solids()]}")

def _apply_complete_lm_um_joint(part, owner: str):
    """Restore one carrier's complete ear and cut its standalone fastener void."""
    if owner not in {"lm", "um"}:
        raise ValueError(owner)
    opposing = "um" if owner == "lm" else "lm"
    for x in JOINT_EAR_X:
        part = _complete_joint_receiver_notch(part, opposing, x)
    for x in JOINT_EAR_X:
        part = _fuse_attached(
            part, _complete_joint_ear(owner, x),
            f"{owner.upper()} complete standalone joint ear {x:+.1f}")
        if owner == "lm":
            part -= _cylinder_at(
                x, JOINT_EAR_Y, JOINT_CLEARANCE_BORE_D / 2.0,
                CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                JOINT_CLEARANCE_BORE_TOP_Z)
        else:
            part -= _cylinder_at(
                x, JOINT_EAR_Y, JOINT_INSERT_BORE_D / 2.0,
                *JOINT_INSERT_BORE_Z)
    return part

def _complete_tweeter_joint_receiver_notch(
        part, ear_owner: str, x: float):
    """Remove one complete opposing T ear plus the 0.20-mm axial gap."""
    if ear_owner == "um":
        z0 = TWEETER_CORE_JOINT_Z[0] - TWEETER_JOINT_CLEAR
        z1 = TWEETER_ADDON_JOINT_Z[0]
    elif ear_owner == "tweeter":
        z0 = TWEETER_CORE_JOINT_Z[1]
        z1 = TWEETER_ADDON_JOINT_Z[1] + TWEETER_JOINT_CLEAR
    else:
        raise ValueError(ear_owner)
    return _subtract_plan_prisms(
        part,
        _complete_tweeter_joint_ear_plan(
            ear_owner, x, TWEETER_JOINT_CLEAR),
        z0, z1)

def _apply_complete_um_tweeter_joint(part, owner: str):
    """Restore complete standalone T ears and their owner-specific voids."""
    if owner not in {"um", "tweeter"}:
        raise ValueError(owner)
    opposing = "tweeter" if owner == "um" else "um"
    for x in TWEETER_JOINT_X:
        part = _complete_tweeter_joint_receiver_notch(part, opposing, x)
    for x in TWEETER_JOINT_X:
        part = _fuse_attached(
            part, _complete_tweeter_joint_ear(owner, x),
            f"{owner.upper()} complete standalone tweeter joint ear "
            f"{x:+.1f}")
        # Preserve the rear-driven screw passage across the true 0.20-mm
        # inter-part gap.  On the crescent it then widens into the D4.6 blind
        # heat-set receiver; the larger cutter deliberately starts at the
        # rear owner's z=12.2 termination for a 0.35-mm service overlap.
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_BORE_TOP_Z)
        if owner == "tweeter":
            part -= _cylinder_at(
                x, TWEETER_JOINT_Y,
                TWEETER_JOINT_INSERT_BORE_D / 2.0,
                *TWEETER_JOINT_INSERT_BORE_Z)
    return part
