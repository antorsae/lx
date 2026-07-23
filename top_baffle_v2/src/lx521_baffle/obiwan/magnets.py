"""Obi-Wan side-magnet sites, cavity cuts and retained-land checks."""

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


def side_magnet_sites(driver: str | None = None):
    """Return side-interface records with exact mating faces/normals."""
    records = []
    for key, center, radius in (
            ("lm", L22_CUTOUT[:2], LM_CORE_R),
            ("um", UM_CUTOUT[:2], UM_CORE_R)):
        if driver is not None and key != driver:
            continue
        for angle in SIDE_MAGNET_ANGLES[key]:
            a = math.radians(angle)
            normal = (math.cos(a), math.sin(a))
            face_offset = SIDE_MAGNET_FACE_OFFSET[key]
            face_r = radius + face_offset
            face = (center[0] + face_r * normal[0],
                    center[1] + face_r * normal[1])
            visible_radius = (
                LM_VISIBLE_RING_R if key == "lm" else UM_VISIBLE_RING_R)
            outer_surface_face = (
                center[0] + visible_radius * normal[0],
                center[1] + visible_radius * normal[1],
            )
            side = "left" if normal[0] < 0 else "right"
            vertical = "upper" if normal[1] >= 0 else "lower"
            records.append({
                "name": (f"{key}_{vertical}_{side}"
                         if key == "lm" else f"{key}_{side}"),
                "driver": key,
                "angle_deg": angle, "normal": normal,
                "face": face, "center": center, "radius": radius,
                "outer_surface_face": outer_surface_face,
                "outer_surface_radius_mm": visible_radius,
                "clock_from_top_deg": 90.0 - angle,
                "face_offset_mm": face_offset,
                "carrier_cavity_face_inset_mm": (
                    visible_radius - face_r),
                "z_mm": SIDE_MAGNET_Z[key],
                "interface_kind": "ring",
                "magnet_fully_buried": True,
                "local_captive_backing_boss_mm": 0.0,
                "continuous_flush_ring_fairing_mm": (
                    visible_radius - radius),
                "proud_ear_added": False,
            })
        if key == "lm":
            for side, face_x, normal_x, angle in (
                    ("left", -LM_BASE_MAGNET_FACE_X, -1.0, 180.0),
                    ("right", LM_BASE_MAGNET_FACE_X, 1.0, 0.0)):
                records.append({
                    "name": f"lm_lower_{side}",
                    "driver": "lm",
                    "angle_deg": angle,
                    "normal": (normal_x, 0.0),
                    "face": (face_x, LM_BASE_MAGNET_Y),
                    "outer_surface_face": (face_x, LM_BASE_MAGNET_Y),
                    # These compatibility fields describe the horizontal
                    # base-side datum; receiver construction must key from
                    # interface_kind rather than treating it as an R113 arc.
                    "center": (0.0, LM_BASE_MAGNET_Y),
                    "radius": LM_BASE_MAGNET_FACE_X,
                    "clock_from_top_deg": 90.0 - angle,
                    "face_offset_mm": 0.0,
                    "carrier_cavity_face_inset_mm": 0.0,
                    "z_mm": LM_BASE_MAGNET_Z,
                    "interface_kind": "base_side",
                    "magnet_fully_buried": True,
                    "local_captive_backing_boss_mm": 0.0,
                    "continuous_flush_ring_fairing_mm": 0.0,
                    "proud_ear_added": False,
                })
    return records

def _axis_cylinder(face, normal, zc: float, diameter: float,
                   inward: float, outward: float):
    """Cylinder along an XY wall normal, measured about ``face``."""
    angle = math.degrees(math.atan2(normal[1], normal[0]))
    length = inward + outward
    shift = (outward - inward) / 2.0
    return (Pos(face[0], face[1], zc)
            * Rot(Z=angle) * Rot(Y=90) * Pos(0.0, 0.0, shift)
            * Cylinder(diameter / 2.0, length))

def _cut_side_magnet_pockets(part, driver: str):
    """Reassert sealed coupon-style cavities after every possible union."""
    for site in side_magnet_sites(driver):
        tools = wall_cavity_tools(
            name=site["name"],
            face=site["face"],
            outward=(*site["normal"], 0.0),
            owner="carrier",
            axis_z=site["z_mm"],
            print_up=(0.0, 0.0, -1.0),
            front_z=THICKNESS_MM,
            interface_gap_mm=SIDE_INTERFACE_GAP,
        )
        for cutter in tools.cutters:
            part -= cutter
        # Clean after each complete cradle/chimney/roof group.  Feeding all
        # six tools into one deferred clean left OCC with same-domain seam
        # fragments at the straight LM side faces and an invalid final shell.
        part = part.clean()
        solids = list(part.solids())
        if (not part.is_valid or len(solids) != 1
                or solids[0].volume <= 0.01):
            raise RuntimeError(
                f"{site['name']} captive cavity invalidated {driver} "
                f"carrier: valid={part.is_valid} "
                f"volumes={[solid.volume for solid in solids]}")
    return part

def _verify_side_magnet_lands(part, driver: str,
                              interface_kinds: set[str] | None = None):
    """Fail unless the immutable carrier already owns every captive land.

    Lower-LM base sites and the four ring sites follow the same rule: no
    station may silently fuse missing material.  Ring lands are contained by
    their continuous smooth fairings; lower lands are contained by the broad
    base shoulder.  Magnets remain completely buried without a local cue.
    """
    for site in side_magnet_sites(driver):
        if (interface_kinds is not None
                and site["interface_kind"] not in interface_kinds):
            continue
        tools = wall_cavity_tools(
            name=site["name"],
            face=site["face"],
            outward=(*site["normal"], 0.0),
            owner="carrier",
            axis_z=site["z_mm"],
            print_up=(0.0, 0.0, -1.0),
            front_z=THICKNESS_MM,
            interface_gap_mm=SIDE_INTERFACE_GAP,
        )
        missing = tools.required_land - part
        missing_volume = sum(solid.volume for solid in missing.solids())
        if missing_volume > 0.01:
            raise RuntimeError(
                f"{site['name']} immutable {site['interface_kind']} host "
                f"misses {missing_volume:.4f} mm3 of captive land")
    return part
