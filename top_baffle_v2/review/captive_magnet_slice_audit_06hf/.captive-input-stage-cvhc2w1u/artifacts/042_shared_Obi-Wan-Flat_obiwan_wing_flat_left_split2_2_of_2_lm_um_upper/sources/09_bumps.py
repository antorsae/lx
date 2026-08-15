"""Obi-Wan covered bumps, burial webs, cover ownership and shell builders."""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from build123d import (
    Box,
    Compound,
    Cylinder,
    Face,
    Plane,
    Polyline,
    Pos,
    Rectangle,
    Sphere,
    Wire,
    extrude,
    loft,
    make_face,
)

from shapely.geometry import LineString, Point, box

from shapely.geometry.polygon import orient

from shapely.ops import unary_union

from ..base import (
    BRIDGE_HOLE_XY,
    L22_CUTOUT,
    L22_PILOT_D_MM,
    STAND_FOOT,
    UM_CUTOUT,
    UM_PILOT_DEPTH_MM,
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
    UM_SEAT_Z,
)

from ..cables import (
    LM_DUCT_OUT_CLEARANCE_FROM_APERTURE_MM,
    LM_DUCT_OUT_REAR_Z_MM,
    LM_DUCT_OUT_X_MM,
    LM_DUCT_OUT_Y_MM,
    LM_EXIT_BEND_R_MM,
    LM_EXIT_D_MM,
    LM_EXIT_MIN_BEND_R_MM,
    OBIWAN_NO_FLOOR_ENTRY_WINDOW_CENTER_XY,
    OBIWAN_NO_FLOOR_ENTRY_WINDOW_D_MM,
    OBIWAN_NO_FLOOR_LM_ENTRY_XY,
    OBIWAN_NO_FLOOR_T_ENTRY_XY,
    OBIWAN_NO_FLOOR_UM_ENTRY_XY,
    lm_exit_handoff_points,
    lm_exit_handoff_spec,
)

from .floor import (
    FLOOR_LANE_SPECS,
    FLOOR_T_ROUTE_FEED_BEARING_DEG,
    FLOOR_T_ROUTE_FEED_XY,
    STEM_HALF_WIDTH_MM,
    STEM_SHOULDER_HALF_WIDTH_MM,
    STEM_TOP_Y_MM,
    STEM_Z_MM,
)


from .route import (
    ANCHOR_LEG_W,
    ANCHOR_SECTION_SPACING,
    ANCHOR_START_OVERLAP,
    BOOLEAN_CLEARANCE_MARGIN,
    BRIDGE_LENGTH,
    BUMP_BACKFILL_TUBE_OVERLAP,
    BURIAL_WEB_LATERAL_OVERLAP,
    BURIAL_WEB_OWNER_INSET,
    BURIAL_WEB_TUBE_OVERLAP,
    CABLE_D_EST,
    CABLE_R_EST,
    CENTRAL_MAIN_FEED_START_BEARING_DEG,
    CENTRAL_MAIN_FEED_START_HANDLE,
    CENTRAL_MAIN_FEED_XY,
    CENTRAL_T_FEED_RISE_LENGTH,
    CENTRAL_T_FEED_START_BEARING_DEG,
    CENTRAL_T_FEED_START_HANDLE,
    CENTRAL_T_FEED_XY,
    CORE_REAR_Z,
    CROSSOVER_ANGLE_DEG,
    CROSSOVER_HALF_LENGTH,
    CROSSOVER_LEG_OMIT_RADIUS,
    CROSSOVER_MAIN_S,
    CROSSOVER_MIN_CLEARANCE,
    CROSSOVER_TARGET_CLEARANCE,
    CROSSOVER_TS_S,
    CROSSOVER_T_Z,
    CROSSOVER_UM_Z,
    CROSSOVER_XY,
    CUTTER_R,
    CUTTER_SIDES,
    CUTTER_SPLIT_OVERLAP,
    DUCT_D,
    DUCT_R,
    FLOOR_FEED_CUTTER_EXTENSION,
    FLOOR_MAIN_FEED_START_BEARING_DEG,
    FLOOR_MAIN_FEED_START_HANDLE,
    FLOOR_MAIN_FEED_XY,
    FLOOR_STEM_CORE_BOUNDS,
    FLOOR_T_FEED_END_HANDLE,
    FLOOR_T_FEED_RISE_LENGTH,
    FLOOR_T_FEED_START_BEARING_DEG,
    FLOOR_T_FEED_START_HANDLE,
    FLOOR_T_FEED_XY,
    INSERT_COVER_CLEAR,
    LM_ARC_LENGTH,
    LM_BRIDGE_START,
    LM_CABLE_D_EST,
    LM_CORE_R,
    LM_ENTRY_LENGTH,
    LM_EXIT_TUBE_SECTION_SPACING_MM,
    LM_EXTERNAL_LEAD_END,
    LM_EXTERNAL_LEAD_END_Z,
    LM_EXTERNAL_LEAD_LENGTH_MM,
    LM_INTERNAL_CENTER_Z_MM,
    LM_INTERNAL_CUTTER_GROUP_COUNT,
    LM_INTERNAL_DUCT_D_MM,
    LM_INTERNAL_DUCT_R,
    LM_INTERNAL_FRONT_SKIN_MM,
    LM_INTERNAL_JUNCTION_OVERTRAVEL_MM,
    LM_INTERNAL_PLAN_LENGTH_MM,
    LM_INTERNAL_PORT_OVERLAP_MM,
    LM_INTERNAL_REAR_SKIN_MM,
    LM_INTERNAL_ROUTE_LENGTH_MM,
    LM_MAIN_BUMP_Z,
    LM_MAIN_CUTTER_SEGMENT_COUNT,
    LM_REAR_HANDOFF_CENTER_Z,
    LM_REAR_HANDOFF_PLAN_BEARING_DEG,
    LM_REAR_HANDOFF_SPEC,
    LM_REAR_PORT_CLEARANCE_FROM_APERTURE_MM,
    LM_REAR_PORT_D_MM,
    LM_REAR_PORT_INNER_DEPTH_MM,
    LM_REAR_PORT_INNER_Z,
    LM_REAR_PORT_PREFUSION_MM,
    LM_REAR_PORT_R,
    LM_REAR_PORT_REAR_OVERTRAVEL_MM,
    LM_REAR_PORT_REAR_Z,
    LM_REAR_PORT_SEAT_CLEAR_MM,
    LM_REAR_PORT_XY,
    LM_ROUTE_ARC_START_DEG,
    LM_ROUTE_CUTTER_GROUP_COUNT,
    LM_ROUTE_END_DEG,
    LM_ROUTE_OWNER_CLEARANCE,
    LM_ROUTE_START_DEG,
    LM_SEAT_MEMBRANE_BOTTOM_Z,
    LM_TS_BUMP_Z,
    LM_T_CUTTER_SEGMENT_COUNT,
    LM_VISIBLE_RING_R,
    MAIN_ANCHOR_KEEPOUTS,
    MAIN_BRIDGE_END_HANDLE_MM,
    MAIN_BRIDGE_START_HANDLE_MM,
    MAIN_LM_ROUTE_R,
    MAIN_OUTER_R,
    MAIN_TRENCH_CENTER_Z,
    MAIN_UM_ENTRY_S,
    NO_FLOOR_BRIDGE_CORE_BOUNDS,
    NO_FLOOR_BRIDGE_ROUTE_BOUNDS,
    NO_FLOOR_ENTRY_BORE_DEPTH_MM,
    NO_FLOOR_ENTRY_BORE_REAR_OVERTRAVEL_MM,
    NO_FLOOR_ENTRY_VESTIBULE_REAR_SKIN_MM,
    NO_FLOOR_FEED_CUTTER_EXTENSION,
    NO_FLOOR_FEED_END_HANDLE,
    NO_FLOOR_FEED_REAR_Z,
    NO_FLOOR_LM_DUCT_OUT_CLEARANCE_FROM_APERTURE_MM,
    NO_FLOOR_LM_DUCT_OUT_Y_MM,
    NO_FLOOR_LM_ENTRY_BORE_INNER_Z_MM,
    NO_FLOOR_LM_ENTRY_END_BEARING_DEG,
    NO_FLOOR_LM_ENTRY_END_HANDLE_MM,
    NO_FLOOR_LM_ENTRY_RELIEF_RADIAL_MM,
    NO_FLOOR_LM_ENTRY_RELIEF_REAR_SKIN_MM,
    NO_FLOOR_LM_ENTRY_START_BEARING_DEG,
    NO_FLOOR_LM_ENTRY_START_HANDLE_MM,
    NO_FLOOR_LM_EXIT_PLAN_BEARING_DEG,
    NO_FLOOR_LM_FEED_XY,
    NO_FLOOR_MAIN_ENTRY_END_HANDLE_MM,
    NO_FLOOR_MAIN_ENTRY_JOIN_BEARING_DEG,
    NO_FLOOR_MAIN_ENTRY_JOIN_IN_HANDLE_MM,
    NO_FLOOR_MAIN_ENTRY_JOIN_OUT_HANDLE_MM,
    NO_FLOOR_MAIN_ENTRY_JOIN_XY,
    NO_FLOOR_MAIN_ENTRY_START_BEARING_DEG,
    NO_FLOOR_MAIN_ENTRY_START_HANDLE_MM,
    NO_FLOOR_MAIN_FEED_RISE_LENGTH,
    NO_FLOOR_MAIN_FEED_START_Z,
    NO_FLOOR_MAIN_FEED_XY,
    NO_FLOOR_MAIN_PAD_BUMP_RELIEF,
    NO_FLOOR_MAIN_RECESS_CLEARANCE_Z,
    NO_FLOOR_RECESS_SKIN_BOOLEAN_MARGIN_MM,
    NO_FLOOR_RING_ENTRY_BURIAL_OVERLAP_MM,
    NO_FLOOR_SERVICE_PATCH_BOUNDS,
    NO_FLOOR_SERVICE_PATCH_MARGIN_MM,
    NO_FLOOR_SERVICE_PATCH_RELEASE_MODE,
    NO_FLOOR_T_ENTRY_END_HANDLE_MM,
    NO_FLOOR_T_ENTRY_JOIN_BEARING_DEG,
    NO_FLOOR_T_ENTRY_JOIN_IN_HANDLE_MM,
    NO_FLOOR_T_ENTRY_JOIN_OUT_HANDLE_MM,
    NO_FLOOR_T_ENTRY_JOIN_XY,
    NO_FLOOR_T_ENTRY_RELIEF_RADIAL_MM,
    NO_FLOOR_T_ENTRY_RELIEF_REAR_SKIN_MM,
    NO_FLOOR_T_ENTRY_START_BEARING_DEG,
    NO_FLOOR_T_ENTRY_START_HANDLE_MM,
    NO_FLOOR_T_FEED_RISE_LENGTH,
    NO_FLOOR_T_FEED_START_Z,
    NO_FLOOR_T_FEED_XY,
    NO_FLOOR_T_PAD_BUMP_RELIEF,
    NO_FLOOR_UM_ENTRY_CAP_RELIEF_HALF_LENGTH_MM,
    NO_FLOOR_UM_ENTRY_CAP_RELIEF_RADIAL_INSET_MM,
    ROUTE_LENGTH,
    ROUTE_SPLIT_GAP,
    SIDE_WALL,
    STANDARD_CUTTER_EXTENSION,
    THROAT_LENGTH,
    TRENCH_CENTER_Z,
    TS_ADDON_SUPPORT_MIN_Y,
    TS_BRIDGE_END_HANDLE_MM,
    TS_BRIDGE_LENGTH,
    TS_BRIDGE_START_HANDLE_MM,
    TS_CABLE_D_EST,
    TS_CORE_END_S,
    TS_CUTTER_R,
    TS_CUTTER_SIDES,
    TS_DUCT_D,
    TS_DUCT_R,
    TS_ENTRY_LENGTH,
    TS_FREE_CABLE_REAR_CLEARANCE,
    TS_FREE_CABLE_Z,
    TS_HANDOFF_LENGTH,
    TS_LM_ARC_LENGTH,
    TS_LM_ARC_START_DEG,
    TS_LM_BRIDGE_START,
    TS_LM_ROUTE_END_DEG,
    TS_LM_ROUTE_R,
    TS_LM_ROUTE_START_DEG,
    TS_OUTER_R,
    TS_ROUTE_LENGTH,
    TS_SIDE_WALL,
    TS_TRENCH_CENTER_Z,
    TS_TWEETER_FLUSH_R,
    TS_TWEETER_MOUTH,
    TS_UM_ARC_LENGTH,
    TS_UM_ARC_START,
    TS_UM_CENTER_Z,
    TS_UM_CORE_COVER_END_R,
    TS_UM_CORE_COVER_END_S,
    TS_UM_ENTRY_S,
    TS_UM_HANDOFF_START,
    TS_UM_ROUTE_END_DEG,
    TS_UM_ROUTE_R,
    TS_UM_ROUTE_START_DEG,
    TS_UM_Z_TRANSITION_LENGTH_MM,
    TUBE_SECTION_SIDES,
    TUBE_SECTION_SPACING,
    TUNNEL_CABLE_CLEAR,
    TUNNEL_FLOOR_SKIN,
    TUNNEL_FUSE_OVERLAP,
    TUNNEL_ROOF_SKIN,
    TUNNEL_SKIN,
    T_ANCHOR_KEEPOUTS,
    T_LM_TRENCH_CENTER_Z,
    UM_CORE_R,
    UM_ENTRY_ANGLE_DEG,
    UM_ENTRY_POINT,
    UM_ENTRY_R,
    UM_MOUTH_POINT,
    UM_MOUTH_R,
    UM_MOUTH_TANGENT,
    UM_MOUTH_Z,
    UM_PILOT_FLOOR_Z,
    UM_ROUTE_CUTTER_GROUP_COUNT,
    UM_SEAT_MEMBRANE_BOTTOM_Z,
    UM_TERMINAL_ARC_ENTRY_BEARING_DEG,
    UM_TERMINAL_ARC_SIDE,
    UM_TERMINAL_PLAN_BEND_R,
    UM_TS_BUMP_Z,
    UM_T_CUTTER_MOUTH_OVERSHOOT,
    _C1,
    _C2,
    _CROSS_GEOM,
    _CROSS_MAIN_TANGENT,
    _CROSS_TS_TANGENT,
    _LM_INTERNAL_ENTRY_XY,
    _LM_INTERNAL_EXIT_XY,
    _LM_INTERNAL_PLAN,
    _LM_INTERNAL_PLAN_S,
    _LM_PILOT_BY_ANGLE,
    _MAIN_ARC,
    _MAIN_ARC_START,
    _MAIN_ARC_TANGENT,
    _MAIN_BRIDGE,
    _MAIN_FEED,
    _MAIN_PLAN,
    _MAIN_PLAN_S,
    _MAIN_THROAT,
    _MAIN_THROAT_PHI,
    _MAIN_THROAT_TANGENT,
    _TS_BRIDGE,
    _TS_C1,
    _TS_C2,
    _TS_COVER_END_INDEX,
    _TS_COVER_END_PREV,
    _TS_FEED,
    _TS_FEED_START_DIRECTION,
    _TS_H1,
    _TS_H2,
    _TS_HANDOFF,
    _TS_LM_ARC,
    _TS_LM_ARC_START,
    _TS_LM_TANGENT,
    _TS_PLAN,
    _TS_PLAN_S,
    _TS_POST_INDICES,
    _TS_RADII_FROM_UM,
    _TS_UM_ARC,
    _TS_UM_END_TANGENT,
    _TS_UM_START_TANGENT,
    _UM_ARC_END_RADIUS,
    _UM_ARC_SIGN,
    _UM_ARC_START_ANGLE,
    _UM_ARC_START_RADIUS,
    _UM_ENTRY_DIRECTION,
    _UM_ENTRY_TO_MOUTH,
    _UM_ENTRY_TO_MOUTH_PERP,
    _UM_PILOT_BY_ANGLE,
    _UM_PLAN_BEND_CENTER,
    _UM_TERMINAL_U,
    _UM_TERMINAL_V,
    _arc,
    _bearing_unit,
    _cosine01,
    _cubic,
    _floor_t_handoff_owner_plan,
    _join,
    _local_max,
    _local_min,
    _main_xyz,
    _no_floor_lm_complete_cutter_path,
    _plan_lengths,
    _plan_tangent,
    _polar,
    _polygon_prism,
    _q,
    _r0,
    _r1,
    _require_guarded_build,
    _resample,
    _rotate90,
    _sampled_centerline_surface_wall,
    _smooth01,
    _station_near,
    _ts_xyz,
    _two_cubic_fan,
    lm_cable_points,
    lm_complete_duct_points,
    lm_internal_duct_cutter_points,
    lm_internal_duct_points,
    lm_rear_handoff_points,
    route_cable_points,
    route_facts,
    route_inner_cutter_group,
    route_inner_cutter_group_count,
    route_inner_cutters,
    route_material_plan,
    route_plan_containment_facts,
    ts_cable_points,
)

from .rear_entry import (
    RearEntryBore,
    RearEntryVestibule,
    _crop_path_interval,
    _global_suffix_first_section,
    _outside_path_halfspace,
    _round_tube,
    _round_tube_from_global_sections,
    _round_tube_global_segment,
    _round_tube_global_suffix,
    _sampled_arc_station,
    _slice_points,
    _tube_section_points,
    _z_axis_bore,
    lm_rear_exit_port_cutter,
    no_floor_lm_bottom_support_blocker,
    no_floor_lm_internal_cutter,
    no_floor_rear_entry_bore_cutters,
    no_floor_rear_entry_bores,
    no_floor_rear_entry_cap_relief_cutters,
    no_floor_rear_entry_transition_cutters,
    no_floor_rear_entry_vestibule_cutters,
    no_floor_rear_entry_vestibules,
)


@dataclass(frozen=True)
class CoveredBump:
    """One continuously skinned Z bypass; it has no omitted span."""

    name: str
    station: float
    low_z: float
    half_length: float

def _named_bumps(plan, records):
    return tuple(CoveredBump(name, _station_near(plan, point), low, half)
                 for name, point, low, half in records)

MAIN_COVERED_BUMPS = _named_bumps(_MAIN_PLAN, (
    ("lm_pilot_300", _LM_PILOT_BY_ANGLE[300.0],
     LM_MAIN_BUMP_Z - NO_FLOOR_MAIN_PAD_BUMP_RELIEF, 34.0),
    ("lm_pilot_0", _LM_PILOT_BY_ANGLE[0.0],
     LM_MAIN_BUMP_Z - 0.40, 32.0),
    ("lm_pilot_60", _LM_PILOT_BY_ANGLE[60.0],
     LM_MAIN_BUMP_Z - 0.40, 32.0),
))

T_COVERED_BUMPS = _named_bumps(_TS_PLAN, (
    ("lm_pilot_240", _LM_PILOT_BY_ANGLE[240.0],
     LM_TS_BUMP_Z - NO_FLOOR_T_PAD_BUMP_RELIEF, 32.0),
    ("lm_pilot_180", _LM_PILOT_BY_ANGLE[180.0],
     LM_TS_BUMP_Z - 0.40, 32.0),
    ("lm_pilot_120", _LM_PILOT_BY_ANGLE[120.0],
     LM_TS_BUMP_Z - 0.40, 32.0),
    # The compact D20 no-floor entry shifts the upstream stationing by a few
    # millimetres.  Extend this existing smooth 328-degree relief by
    # 3 mm so the T cover retains the qualified clearance to the +X UM joint
    # ear as it approaches the same pilot; low Z and all native ownership stay
    # unchanged.
    ("um_pilot_328", _UM_PILOT_BY_ANGLE[328.0], UM_TS_BUMP_Z, 28.0),
    ("um_pilot_58", _UM_PILOT_BY_ANGLE[58.0], UM_TS_BUMP_Z, 28.0),
))

def _central_rear_feed_rise(stations, start_z, nominal_z, rise_length):
    """State-owned rear-face mouth to buried layer with a shallow ramp.

    The integral floor continuation arrives tangent to the XY feed bearing,
    so floor mode uses a quintic zero-slope/zero-curvature rise at both ends.
    That makes the connector continuation and annular route G2 in Z while
    retaining the common feed mouth.  The no-floor bridge starts each oblique
    sweep just behind the rear skin and uses its explicit Z bore for the
    external mouth.  In both states the owner supplies the surrounding wall;
    no separate external raceway is added.
    """
    stations = np.asarray(stations, dtype=float)
    u = np.clip(stations / rise_length, 0.0, 1.0)
    rise = nominal_z - start_z
    if STAND_FOOT:
        return start_z + rise * _smooth01(u)
    return start_z + rise * (1.0 - (1.0 - u) ** 3)

def _no_floor_burial_guard_stations(stations, xy, outer_radius):
    """Return the planar-patch and full-depth-ring handoff stations."""
    stations = np.asarray(stations, dtype=float)
    xy = np.asarray(xy, dtype=float)
    x0, y0, x1, y1 = NO_FLOOR_SERVICE_PATCH_BOUNDS
    dx = np.maximum.reduce((x0 - xy[:, 0], np.zeros(len(xy)), xy[:, 0] - x1))
    dy = np.maximum.reduce((y0 - xy[:, 1], np.zeros(len(xy)), xy[:, 1] - y1))
    overlaps_patch = np.hypot(dx, dy) <= float(outer_radius) + 1.0e-9
    indices = np.flatnonzero(overlaps_patch)
    if not len(indices):
        raise RuntimeError("no-floor cable feed no longer crosses service patch")
    patch_guard_end = float(stations[indices[-1]])

    # At this crossing the circular LM route owner begins supplying material
    # below the z=5.3 bridge crop.  Hold the route level for one further skin
    # width so the quintic descent starts with zero slope inside that deep
    # owner.  Requiring the *whole* outer cover to enter first is needlessly
    # conservative: T would then have only 11.85 mm to reach its mandatory
    # first insert bypass, violating the R14 cable-bend contract.  The exact
    # final-BREP rear-skin spine test proves this coupled handoff stays closed.
    ring_entry_center_radius = LM_VISIBLE_RING_R
    radial = np.linalg.norm(
        xy - np.asarray(L22_CUTOUT[:2], dtype=float), axis=1)
    ring_entries = np.flatnonzero(
        (stations >= patch_guard_end)
        & (radial <= ring_entry_center_radius + 1.0e-7))
    if not len(ring_entries):
        raise RuntimeError(
            "no-floor cable feed no longer enters the full-depth LM ring owner")
    ring_entry_station = float(stations[ring_entries[0]])
    guard_end = max(
        patch_guard_end,
        ring_entry_station + NO_FLOOR_RING_ENTRY_BURIAL_OVERLAP_MM,
    )
    return {
        "patch_guard_end_station_mm": patch_guard_end,
        "ring_deep_owner_entry_station_mm": ring_entry_station,
        "ring_entry_center_radius_mm": ring_entry_center_radius,
        "ring_entry_overlap_mm": NO_FLOOR_RING_ENTRY_BURIAL_OVERLAP_MM,
        "guard_end_station_mm": guard_end,
    }

def _no_floor_service_patch_burial_profile(
        stations, xy, outer_radius, buried_z,
        release_station, release_z, *, protect_lm_recess=False,
        recess_clearance_z=None, direct_to_release=False):
    """Authoritative prefix Z that keeps the conduit off the rear patch.

    ``NO_FLOOR_SERVICE_PATCH_BOUNDS`` is the four-insert rectangle plus the
    requested 6-mm margin.  Expanding that rectangle by the conduit outer
    radius protects the visible service patch.  The positive bridge-tail
    owner is planar at z=5.3.  Both routes begin their rearward descent only
    after their complete covers clear that patch.  T goes directly to its
    first pilot-bypass depth; UM first reaches a recess-safe Z at the R110.6
    crossing, then joins its first LM-pilot bypass with a zero-slope quintic
    suffix.  The retained positive cover makes each portion below z=5.3 a
    closed hidden belly rather than an open exterior slot.
    """
    stations = np.asarray(stations, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if STAND_FOOT:
        return np.full_like(stations, np.nan)
    guard = _no_floor_burial_guard_stations(stations, xy, outer_radius)
    guard_end = guard["guard_end_station_mm"]
    release_station = float(release_station)
    flat_guard_end = (
        guard["patch_guard_end_station_mm"]
        if protect_lm_recess else guard_end)
    if release_station <= flat_guard_end:
        raise RuntimeError(
            "first insert bypass no longer follows the buried ring handoff")
    profile = np.full_like(stations, np.nan)
    guarded = stations <= flat_guard_end
    profile[guarded] = float(buried_z)

    if protect_lm_recess:
        if direct_to_release:
            transition = (
                (stations > flat_guard_end)
                & (stations <= release_station))
            u = ((stations[transition] - flat_guard_end)
                 / (release_station - flat_guard_end))
            profile[transition] = (
                float(buried_z)
                + (float(release_z) - float(buried_z)) * _cosine01(u))
            return profile
        if recess_clearance_z is None:
            recess_clearance_z = NO_FLOOR_MAIN_RECESS_CLEARANCE_Z
        recess_clearance_z = float(recess_clearance_z)
        radial = np.linalg.norm(
            xy - np.asarray(L22_CUTOUT[:2], dtype=float), axis=1)
        recess_entries = np.flatnonzero(
            (stations > flat_guard_end)
            & (radial <= LM_RECESS_R + 1.0e-7))
        if not len(recess_entries):
            raise RuntimeError("UM feed no longer enters the LM flange recess")
        recess_entry = float(stations[recess_entries[0]])
        if release_station <= recess_entry:
            raise RuntimeError(
                "first UM insert bypass precedes the protected recess entry")
        descent = (
            (stations > flat_guard_end) & (stations <= recess_entry))
        u = ((stations[descent] - flat_guard_end)
             / (recess_entry - flat_guard_end))
        profile[descent] = (
            float(buried_z)
            + (recess_clearance_z - float(buried_z))
            * _cosine01(u))
        suffix = (
            (stations > recess_entry) & (stations <= release_station))
        u = ((stations[suffix] - recess_entry)
             / (release_station - recess_entry))
        profile[suffix] = (
            recess_clearance_z
            + (float(release_z) - recess_clearance_z)
            * _smooth01(u))
    else:
        transition = (
            (stations > guard_end) & (stations <= release_station))
        u = ((stations[transition] - guard_end)
             / (release_station - guard_end))
        profile[transition] = (
            float(buried_z)
            + (float(release_z) - float(buried_z)) * _smooth01(u))
    return profile

@dataclass(frozen=True)
class BumpBackfillSpec:
    """One local solid roof-to-bore saddle in its final owner carrier."""

    name: str
    route_name: str
    owner: str
    station: float
    route_xyz: tuple[float, float, float]
    pilot_xy: tuple[float, float]
    route_outer_radius: float
    pilot_support_radius: float
    bottom_z: float
    top_z: float

def _interp_xyz(stations, points, station):
    return tuple(float(np.interp(station, stations, points[:, axis]))
                 for axis in range(3))

def bump_backfill_specs():
    """Authoritative eight solid-backed insert-crossing records."""
    main_s, main = _main_xyz(0.35)
    ts_s, ts = _ts_xyz(0.35)
    specs = []
    for route_name, bumps, stations, points, outer_radius in (
            ("UM", MAIN_COVERED_BUMPS, main_s, main, MAIN_OUTER_R),
            ("T", T_COVERED_BUMPS, ts_s, ts, TS_OUTER_R)):
        for bump in bumps:
            angle = float(bump.name.rsplit("_", 1)[-1])
            is_lm = bump.name.startswith("lm_pilot_")
            pilot = (_LM_PILOT_BY_ANGLE[angle] if is_lm
                     else _UM_PILOT_BY_ANGLE[angle])
            route_xyz = _interp_xyz(stations, points, bump.station)
            top_z = (LM_SEAT_Z - LM_BORE_DEPTH_MM if is_lm
                     else UM_SEAT_Z - UM_PILOT_DEPTH_MM)
            support_radius = (PAD_D_MM / 2.0 if is_lm
                              else UM_PAD_D_MM / 2.0)
            owner = "lm" if is_lm else "um"
            specs.append(BumpBackfillSpec(
                name=bump.name,
                route_name=route_name,
                owner=owner,
                station=bump.station,
                route_xyz=route_xyz,
                pilot_xy=tuple(map(float, pilot)),
                route_outer_radius=outer_radius,
                pilot_support_radius=support_radius,
                bottom_z=(route_xyz[2] + outer_radius
                          - BUMP_BACKFILL_TUBE_OVERLAP),
                top_z=top_z,
            ))
    return tuple(specs)

def _bump_backfill(spec, clearance=0.0):
    """Build a full-width roof saddle, retaining only real hardware voids."""
    route_disk = Point(*spec.route_xyz[:2]).buffer(
        spec.route_outer_radius + clearance, resolution=32)
    pilot_disk = Point(*spec.pilot_xy).buffer(
        spec.pilot_support_radius + clearance, resolution=32)
    plan = unary_union((route_disk, pilot_disk)).convex_hull
    fill = _polygon_prism(
        plan, spec.bottom_z - clearance, spec.top_z + clearance)
    owner_cutout = L22_CUTOUT if spec.owner == "lm" else UM_CUTOUT
    fill -= Pos(owner_cutout[0], owner_cutout[1], 0.0) * Cylinder(
        owner_cutout[2] / 2.0, 100.0)
    # Backfills share the same native radial ownership as their carrier.
    # In floor mode the 300/240-degree route centers begin at legacy R114;
    # their convex-hull saddles must not project a pointed cap beyond the
    # minimal LM R113 outline.  Keep the roof-to-bore fill inside the owner
    # and let the short exterior entry remain free cable as specified.
    fill = (
        _lm_ring_outer_crop(fill)
        if spec.owner == "lm"
        else _um_owner_crop(fill))
    fill = fill.clean()
    solids = tuple(fill.solids())
    if (not fill.is_valid or not solids
            or any(solid.volume <= 0.01 for solid in solids)):
        raise RuntimeError(
            f"{spec.name}: invalid solid bump backfill; "
            f"valid={fill.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return solids

def bump_backfill_components(owner, route_name=None):
    """Yield the exact local saddles owned by one printed carrier."""
    _require_guarded_build()
    if owner not in {"lm", "um"}:
        raise ValueError(owner)
    for spec in bump_backfill_specs():
        if spec.owner != owner:
            continue
        if route_name is not None and spec.route_name != route_name:
            continue
        yield from _bump_backfill(spec)

def _anchor_leg(
        points, outer_radius, sign, anchor_base_z, clearance=0.0):
    """Loft one fallback side leg for an upper-owner round cover.

    Current production LM and UM low runs use
    ``_burial_web_components`` instead: leaving the omega space open exposes
    longitudinal pockets beside insert bypasses.  This helper remains for
    bounded route experiments that explicitly disable the full burial web.
    Every retained fallback leg overlaps its round cover and carrier by
    positive volume; no route/base tangency is used as a union.
    """
    points = np.asarray(points, dtype=float)
    # The leg is only a local carrier-to-cover web; cable clearance is owned
    # by the round inner/outer tube.  Resample it independently so OCC does
    # not retain hundreds of B-spline section poles for every low route run.
    # A ruled 5-mm web follows the same authoritative centerline endpoints
    # and Z profile while materially reducing Boolean peak memory.
    source_s = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    section_s = np.linspace(
        0.0, source_s[-1],
        max(2, int(math.ceil(
            source_s[-1] / ANCHOR_SECTION_SPACING))) + 1)
    points = np.column_stack(tuple(
        np.interp(section_s, source_s, points[:, axis])
        for axis in range(3)))
    sections = []
    leg_width = ANCHOR_LEG_W + 2.0 * clearance
    center_offset = sign * (outer_radius - ANCHOR_LEG_W / 2.0)
    for index in range(len(points)):
        before = points[max(0, index - 1), :2]
        after = points[min(len(points) - 1, index + 1), :2]
        tangent = after - before
        tangent /= np.linalg.norm(tangent)
        normal = np.asarray((-tangent[1], tangent[0]))
        p = points[index]
        bottom = min(float(p[2]), anchor_base_z - 0.20) - clearance
        offsets = (
            center_offset - leg_width / 2.0,
            center_offset + leg_width / 2.0,
        )
        q0 = p[:2] + offsets[0] * normal
        q1 = p[:2] + offsets[1] * normal
        corners = (
            (float(q0[0]), float(q0[1]), bottom),
            (float(q1[0]), float(q1[1]), bottom),
            (float(q1[0]), float(q1[1]),
             anchor_base_z + TUNNEL_FUSE_OVERLAP + clearance),
            (float(q0[0]), float(q0[1]),
             anchor_base_z + TUNNEL_FUSE_OVERLAP + clearance),
            (float(q0[0]), float(q0[1]), bottom),
        )
        sections.append(Face(Wire(Polyline(*corners).edges())))
    leg = loft(sections, ruled=True).clean()
    solids = list(leg.solids())
    if not leg.is_valid or len(solids) != 1:
        raise RuntimeError(
            "route anchor leg must be one valid loft; "
            f"valid={leg.is_valid} solids={len(solids)}")
    return solids[0]

def _point_runs(points, keep):
    """Contiguous kept point runs without entering excluded hardware."""
    runs = []
    start = None
    for index, value in enumerate(keep):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(keep) - 1):
            stop = index + 1 if value else index
            if stop - start >= 2:
                runs.append(points[start:stop])
            start = None
    return runs

def _point_runs_with_boundary_overlap(points, keep, allowed):
    """Kept runs plus one safe station into naturally buried material.

    A loft ending at the last low station can leave a triangular pocket
    between its ruled end face and the first naturally buried tube section.
    Extend one station at either end, but never cross an unsupported owner
    boundary or the explicit T/UM crossover exclusion.
    """
    points = np.asarray(points, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    allowed = np.asarray(allowed, dtype=bool)
    runs = []
    start = None
    for index, value in enumerate(keep):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(keep) - 1):
            stop = index + 1 if value else index
            expanded_start = start - 1 if start > 0 and allowed[start - 1] else start
            expanded_stop = (
                stop + 1 if stop < len(points) and allowed[stop] else stop)
            if expanded_stop - expanded_start >= 2:
                runs.append(points[expanded_start:expanded_stop])
            start = None
    return runs

def _support_plan_mask(points, support_domains):
    """Centerline stations backed by actual carrier/add-on material.

    Anchor fins are useful only where they join a low tube to a seat
    membrane or structural lip. They are forbidden on free-span handoffs;
    the closed circular tube alone crosses those gaps.
    """
    points = np.asarray(points, dtype=float)
    if isinstance(support_domains, str):
        support_domains = (support_domains,)
    supported = np.zeros(len(points), dtype=bool)
    for domain in support_domains:
        if domain == "lm":
            radial = np.linalg.norm(
                points[:, :2] - np.asarray(L22_CUTOUT[:2]), axis=1)
            supported |= ((radial >= L22_CUTOUT[2] / 2.0)
                          & (radial <= LM_CORE_R))
        elif domain == "um":
            radial = np.linalg.norm(
                points[:, :2] - np.asarray(UM_CUTOUT[:2]), axis=1)
            supported |= ((radial >= UM_CUTOUT[2] / 2.0)
                          & (radial <= UM_CORE_R))
        elif domain == "tweeter":
            supported |= points[:, 1] >= TS_ADDON_SUPPORT_MIN_Y
        else:
            raise ValueError(f"unknown anchor support domain: {domain}")
    return supported

def _anchor_keep_mask(
        points, outer_radius, *, anchor_base_z=LM_SEAT_MEMBRANE_BOTTOM_Z,
        omit_crossover=False, clearance=0.0, hardware_keepouts=(),
        support_domains=("lm",)):
    points = np.asarray(points, dtype=float)
    keep = (points[:, 2] + outer_radius
            < anchor_base_z + ANCHOR_START_OVERLAP)
    keep &= _support_plan_mask(points, support_domains)
    if omit_crossover:
        keep &= (np.linalg.norm(points[:, :2] - CROSSOVER_XY, axis=1)
                 > CROSSOVER_LEG_OMIT_RADIUS + clearance)
    for center, solid_radius in hardware_keepouts:
        keep &= (np.linalg.norm(points[:, :2] - center, axis=1)
                 > solid_radius + outer_radius + INSERT_COVER_CLEAR
                 + clearance)
    return keep

def _burial_web_masks(
        points, outer_radius, *, anchor_base_z=LM_SEAT_MEMBRANE_BOTTOM_Z,
        omit_crossover=False, clearance=0.0, support_domains=("lm",)):
    """Return low-section and safe-domain masks for a closed burial web."""
    points = np.asarray(points, dtype=float)
    allowed = _support_plan_mask(points, support_domains)
    if omit_crossover:
        allowed &= (
            np.linalg.norm(points[:, :2] - CROSSOVER_XY, axis=1)
            > CROSSOVER_LEG_OMIT_RADIUS + clearance)
    # A crown-only threshold leaves the conduit shoulders below the carrier
    # for much of each Z rise. Keep the full-width web until the tube centre
    # plane itself has positive overlap with the carrier; only then is the
    # complete upper half naturally buried rather than merely the crown.
    low = points[:, 2] < anchor_base_z + ANCHOR_START_OVERLAP
    return low & allowed, allowed

def _burial_web_owner_plan(support_domains):
    """Return the native XY owner used to bound positive burial webs.

    The web is auxiliary backing inside material that already belongs to a
    carrier.  Building its transverse sections inside that owner avoids
    creating microscopic faces when the finished cover is cropped at a
    tangent or near-tangent ring boundary.  LM uses the actual exposed
    R113.8-side/R113-cusp outline; UM uses its native structural R51.7 disk.
    """
    if isinstance(support_domains, str):
        support_domains = (support_domains,)
    plans = []
    for domain in support_domains:
        if domain == "lm":
            plans.append(_lm_positive_owner_plan())
        elif domain == "um":
            plans.append(Point(*UM_CUTOUT[:2]).buffer(
                UM_CORE_R, resolution=256))
        elif domain == "tweeter":
            # The crescent support domain is not radially cropped.  No
            # production full-width burial web currently uses it.
            return None
        else:
            raise ValueError(f"unknown burial-web owner domain: {domain}")
    if not plans:
        raise ValueError("burial web requires at least one owner domain")
    # The support masks use exact circular radii while Shapely represents the
    # same owners with fine polygons.  A 1-um query buffer prevents a station
    # on an exact structural datum from being rejected by polygon sagitta;
    # the 0.05-mm production inset remains fifty times larger.
    return unary_union(plans).buffer(0).buffer(0.001)

def _owner_ray_limit(owner_plan, point, direction):
    """Distance from an interior point to one owner boundary along a ray."""
    if owner_plan is None:
        return math.inf
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    if not owner_plan.covers(Point(float(point[0]), float(point[1]))):
        raise RuntimeError(
            "burial-web center station lies outside its printed owner: "
            f"point={tuple(map(float, point))}")
    low = 0.0
    high = 2.0 * (LM_VISIBLE_RING_R + UM_CORE_R) + 50.0
    if owner_plan.covers(Point(*map(float, point + high * direction))):
        raise RuntimeError("burial-web owner ray did not reach a boundary")
    for _ in range(48):
        middle = (low + high) / 2.0
        probe = point + middle * direction
        if owner_plan.covers(Point(float(probe[0]), float(probe[1]))):
            low = middle
        else:
            high = middle
    return low

def _burial_web(
        points, outer_radius, anchor_base_z, clearance=0.0,
        owner_plan=None):
    """Full conduit-width web from below centerline to the seat membrane.

    The circular outer cover still defines the minimum rear bump.  This
    front-side web replaces the two narrow omega legs on owner-supported low
    runs, closing the otherwise visible pockets without increasing rear
    depth.  Both LM-owned UM/T runs and the UM-owned T run use this solid web.
    """
    points = np.asarray(points, dtype=float)
    source_s = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    section_s = np.linspace(
        0.0, source_s[-1],
        max(2, int(math.ceil(
            source_s[-1] / ANCHOR_SECTION_SPACING))) + 1)
    points = np.column_stack(tuple(
        np.interp(section_s, source_s, points[:, axis])
        for axis in range(3)))
    sections = []
    # Extend slightly beyond the circumscribed conduit half-width. The tube
    # and web use independently phased ruled sections, so exact-R endpoints
    # can otherwise meet only tangentially at a section corner.
    half_width = outer_radius + BURIAL_WEB_LATERAL_OVERLAP + clearance
    top_z = anchor_base_z + TUNNEL_FUSE_OVERLAP + clearance
    for index, point in enumerate(points):
        before = points[max(0, index - 1), :2]
        after = points[min(len(points) - 1, index + 1), :2]
        tangent = after - before
        tangent /= np.linalg.norm(tangent)
        normal = np.asarray((-tangent[1], tangent[0]))
        minus_width = min(
            half_width,
            max(0.0, _owner_ray_limit(
                owner_plan, point[:2], -normal)
                - BURIAL_WEB_OWNER_INSET - clearance))
        plus_width = min(
            half_width,
            max(0.0, _owner_ray_limit(
                owner_plan, point[:2], normal)
                - BURIAL_WEB_OWNER_INSET - clearance))
        if min(minus_width, plus_width) <= 0.10:
            raise RuntimeError(
                "burial-web owner leaves a degenerate transverse section: "
                f"point={tuple(map(float, point[:2]))} "
                f"widths=({minus_width:.6f}, {plus_width:.6f})")
        q0 = point[:2] - minus_width * normal
        q1 = point[:2] + plus_width * normal
        # The web begins just behind the center plane, overlapping the upper
        # half of the round tube. This closes both lateral shoulders; the tube
        # still extends much farther rearward to center-R, so rear depth is
        # unchanged.
        bottom_z = (
            float(point[2]) - BURIAL_WEB_TUBE_OVERLAP - clearance)
        corners = (
            (float(q0[0]), float(q0[1]), bottom_z),
            (float(q1[0]), float(q1[1]), bottom_z),
            (float(q1[0]), float(q1[1]), top_z),
            (float(q0[0]), float(q0[1]), top_z),
            (float(q0[0]), float(q0[1]), bottom_z),
        )
        sections.append(Face(Wire(Polyline(*corners).edges())))
    web = loft(sections, ruled=True).clean()
    solids = tuple(web.solids())
    if (not web.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "route burial web must be one valid loft before recuts; "
            f"valid={web.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return solids[0]

def _burial_web_components(
        points, outer_radius, *, anchor_base_z=LM_SEAT_MEMBRANE_BOTTOM_Z,
        omit_crossover=False, clearance=0.0,
        support_domains=("lm",)):
    """Closed full-width owner burial webs."""
    points = np.asarray(points, dtype=float)
    keep, allowed = _burial_web_masks(
        points, outer_radius, anchor_base_z=anchor_base_z,
        omit_crossover=omit_crossover, clearance=clearance,
        support_domains=support_domains)
    components = []
    owner_plan = _burial_web_owner_plan(support_domains)
    for run in _point_runs_with_boundary_overlap(points, keep, allowed):
        web = _burial_web(
            run, outer_radius, anchor_base_z, clearance,
            owner_plan=owner_plan)
        solids = tuple(web.solids())
        if (not web.is_valid or any(solid.volume <= 0.01 for solid in solids)):
            raise RuntimeError(
                "route burial web failed; "
                f"valid={web.is_valid} volumes="
                f"{[solid.volume for solid in solids]}")
        components.extend(solids)
    return tuple(components)

def _anchored_cover_components(
        points, outer_radius, *, anchor_base_z=LM_SEAT_MEMBRANE_BOTTOM_Z,
        omit_crossover=False, clearance=0.0, hardware_keepouts=(),
        support_domains=("lm",), full_burial_web=False):
    """Round cover plus either minimum legs or a closed burial web."""
    points = np.asarray(points, dtype=float)
    supports = (
        _burial_web_components(
            points, outer_radius, anchor_base_z=anchor_base_z,
            omit_crossover=omit_crossover, clearance=clearance,
            support_domains=support_domains)
        if full_burial_web else
        _anchor_leg_components(
            points, outer_radius, anchor_base_z=anchor_base_z,
            omit_crossover=omit_crossover, clearance=clearance,
            hardware_keepouts=hardware_keepouts,
            support_domains=support_domains)
    )
    return (
        _round_tube(points, outer_radius + clearance),
        *supports,
    )

def _anchor_leg_components(
        points, outer_radius, *, anchor_base_z=LM_SEAT_MEMBRANE_BOTTOM_Z,
        omit_crossover=False, clearance=0.0, hardware_keepouts=(),
        support_domains=("lm",)):
    """Return upper-owner local legs where a full burial web is not asked."""
    points = np.asarray(points, dtype=float)
    components = []
    # A leg is necessary only before a round cover would lose robust
    # positive overlap with the z=6.8 carrier. Begin while 0.4 mm of
    # overlap remains; omit the complete pad/head footprint so the closed
    # tube alone makes the shortest hardware bridge.
    keep = _anchor_keep_mask(
        points, outer_radius, anchor_base_z=anchor_base_z,
        omit_crossover=omit_crossover,
        clearance=clearance, hardware_keepouts=hardware_keepouts,
        support_domains=support_domains)
    runs = _point_runs(points, keep)
    for run in runs:
        components.extend((
            _anchor_leg(
                run, outer_radius, -1.0, anchor_base_z, clearance),
            _anchor_leg(
                run, outer_radius, 1.0, anchor_base_z, clearance),
        ))
    return tuple(components)

def _extended_points(points, extension_mm=STANDARD_CUTTER_EXTENSION):
    points = np.asarray(points, dtype=float)
    first = points[1] - points[0]
    last = points[-1] - points[-2]
    first /= np.linalg.norm(first)
    last /= np.linalg.norm(last)
    return np.vstack((
        points[0] - extension_mm * first,
        points,
        points[-1] + extension_mm * last,
    ))

def _owner_cutter_extension(owner):
    """Return the state-authoritative endpoint overshoot for one owner."""
    if owner not in ("lm", "um"):
        raise ValueError(owner)
    return (FLOOR_FEED_CUTTER_EXTENSION if STAND_FOOT
            else NO_FLOOR_FEED_CUTTER_EXTENSION)

def _owner_cutter_points(points, owner):
    """Return the exact cutter path used by one printed route owner.

    Floor feeds start inside the state-owned stem and retain the 2.0-mm
    overshoot. No-floor feeds terminate in explicit rear-normal entry bores,
    so their swept cutters start exactly at the functional datum. Keeping this
    state choice in one helper is essential: changing an extension changes
    the global ruled-section phase along the complete path, not only its caps.
    """
    extension = _owner_cutter_extension(owner)
    if extension <= 0.0:
        return np.asarray(points, dtype=float).copy()
    return _extended_points(points, extension)

def _um_owner_crop(shape, *, cutter=False):
    """Crop T material to the native UM owner and its open cutter mouth."""
    radius = UM_CORE_R + (
        UM_T_CUTTER_MOUTH_OVERSHOOT if cutter else 0.0)
    # build123d cylinders are Z-centered by default; z=0 spans -50..+50.
    cylinder = Pos(UM_CUTOUT[0], UM_CUTOUT[1], 0.0) * Cylinder(radius, 100.0)
    return shape & cylinder

def _lm_positive_owner_plan():
    """Actual exposed LM outline: R113.8 sides, structural R113 cusp.

    The carrier's continuous side fairing deliberately stops at the existing
    LM--UM interface cusp.  Giving a route cover a full R113.8 circular owner
    there creates an unsupported annular lens that the final complementary
    ownership recut can detach as a Boolean sliver.  Use the same plan rule as
    the carrier blank: visible R113.8 on exposed sides and native R113 in the
    upper interface region.
    """
    center = np.asarray(L22_CUTOUT[:2], dtype=float)
    structural = Point(*center).buffer(LM_CORE_R, resolution=256)
    visible = Point(*center).buffer(LM_VISIBLE_RING_R, resolution=256)
    exposed_side = box(
        center[0] - LM_VISIBLE_RING_R - 1.0,
        center[1] - LM_VISIBLE_RING_R - 1.0,
        center[0] + LM_VISIBLE_RING_R + 1.0,
        center[1] + LM_CORE_R,
    )
    return structural.union(visible.intersection(exposed_side)).buffer(0)

def _lm_ring_outer_crop(shape, *, cutter=False):
    """Crop the LM lumen at R113 and its printed cover at visible R113.8."""
    if cutter:
        owner = Pos(L22_CUTOUT[0], L22_CUTOUT[1], 0.0) * Cylinder(
            LM_CORE_R + CUTTER_SPLIT_OVERLAP, 100.0)
    else:
        center = np.asarray(L22_CUTOUT[:2], dtype=float)
        structural = Pos(*center, 0.0) * Cylinder(LM_CORE_R, 100.0)
        visible = Pos(*center, 0.0) * Cylinder(LM_VISIBLE_RING_R, 100.0)
        # ``Box`` is centre-aligned in direct build123d construction.  Place
        # it at the centre of the Shapely fairing clip used by
        # ``side_ring_outer_plan``; treating the clip's lower-left corner as
        # a build123d origin silently retained the R113.8 fairing on only one
        # side of the LM ring and clipped the outward UM route on the other.
        clip_x0 = center[0] - LM_VISIBLE_RING_R - 1.0
        clip_y0 = center[1] - LM_VISIBLE_RING_R - 1.0
        clip_x1 = center[0] + LM_VISIBLE_RING_R + 1.0
        clip_y1 = center[1] + LM_CORE_R
        clip = Pos(
            (clip_x0 + clip_x1) / 2.0,
            (clip_y0 + clip_y1) / 2.0,
            0.0,
        ) * Box(
            clip_x1 - clip_x0,
            clip_y1 - clip_y0,
            100.0,
        )
        owner = structural.fuse(visible & clip).clean()
    return shape & owner

def _lm_state_tail_crop(shape, *, cutter=False):
    """Crop route material to the integral stem or no-floor bridge owner.

    The floor carrier's rear feed mouths are native openings at z=5.3.  Its
    positive round covers therefore begin at that plane, while the negative
    lumen cutters retain the complete z=0..5.3 owner overlap needed to open
    the already-solid integral stem.  The no-floor bridge itself begins at
    z=5.3, so both of its domains use the same bound.
    """
    if STAND_FOOT:
        bounds = FLOOR_STEM_CORE_BOUNDS
        z0 = STEM_Z_MM[0] if cutter else PAD_FACE_Z
    else:
        bounds = NO_FLOOR_BRIDGE_ROUTE_BOUNDS
        # Outside the exact four-insert +6-mm flat patch the UM feed descends
        # behind the LM flange recess.  Retain both its positive cover and
        # matching lumen below z=5.3 so that transition is a closed hidden
        # conduit belly, not a clipped rear-open slot.  The path itself stays
        # at z=10.2/9.1 until the full cover clears the protected flat patch.
        z0 = 0.0
    plan = box(*bounds)
    if STAND_FOOT:
        # The rectangular stem owns the bulk of both floor continuations.
        # The left T lane alone exits its x=-32 boundary before it reaches
        # the LM ring, so fuse its intentionally narrow owner corridor into
        # the crop.  This makes the floor cutter and annular route overlap by
        # positive volume instead of leaving a capped hidden sliver.
        t_handoff = _floor_t_handoff_owner_plan()
        if t_handoff is None:
            raise RuntimeError("floor T handoff owner plan is unavailable")
        plan = unary_union((plan, t_handoff)).buffer(0)
    owner = _polygon_prism(plan, z0, 100.0)
    return shape & owner

def _lm_printed_owner_crop(shape, *, cutter=False):
    """Restrict one LM route solid to actual printed LM material.

    Positive covers stop 0.05 mm inside the visible R113.8 fairing except for
    the central feed span owned by the integral floor stem or no-floor bridge.
    Negative cutters receive only the required structural-R113 mouth
    overshoot and never enter the free-cable span toward UM.
    """
    ring_component = _lm_ring_outer_crop(shape, cutter=cutter)
    components = []
    if ring_component is not None:
        components.extend(
            solid for solid in ring_component.solids()
            if solid.volume > 1e-9)
    tail_component = _lm_state_tail_crop(shape, cutter=cutter)
    if tail_component is not None:
        components.extend(
            solid for solid in tail_component.solids()
            if solid.volume > 1e-9)
    if not components:
        return None
    combined = components[0].fuse(*components[1:]).clean()
    solids = tuple(combined.solids())
    if (not combined.is_valid or not solids
            or any(solid.volume <= 0.01 for solid in solids)):
        raise RuntimeError(
            "LM printed owner-domain crop failed; "
            f"valid={combined.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return combined

def _main_printed_outer_from_full(full_tube):
    """Keep the LM-owned main cover through its flush R113.8 mouth."""
    return _lm_printed_owner_crop(full_tube)

def _t_flush_owner_components(shape):
    """Return the exact LM+UM core domain for the printed T route."""
    ts_points = ts_cable_points(1.8)
    lm_source = _crop_path_interval(
        shape, ts_points, TS_ROUTE_LENGTH, 0.0, TS_UM_ENTRY_S)
    lm = _lm_ring_outer_crop(lm_source)
    # The path crosses R51.7 exactly once. The radial owner crop alone is the
    # authoritative native butt mouth; a preceding tangent station trim cuts
    # away a small curved shell cap that still lies inside the UM owner.
    um = _um_owner_crop(shape)
    core_pieces = []
    for component in (lm, um):
        if component is not None:
            core_pieces.extend(
                solid for solid in component.solids()
                if solid.volume > 1e-9)
    tail = _lm_state_tail_crop(shape)
    if tail is not None:
        core_pieces.extend(
            solid for solid in tail.solids()
            if solid.volume > 1e-9)
    core = (None if not core_pieces else
            core_pieces[0].fuse(*core_pieces[1:]).clean())
    return (("core", core),)

def _t_flush_owner_crop(shape, owner_filter=None):
    """Keep selected native T-shell owners, including no-floor web."""
    pieces = []
    for owner_name, cropped in _t_flush_owner_components(shape):
        if owner_filter is not None and owner_name != owner_filter:
            continue
        if cropped is not None:
            pieces.extend(
                solid for solid in cropped.solids()
                if solid.volume > 1e-9)
    if not pieces:
        return None
    return pieces[0].fuse(*pieces[1:]).clean()

def _t_owner_phase_shell(shape):
    """Crop T material and subtract the core production-phased lumen."""
    pieces = []
    ts_points = ts_cable_points(1.8)
    for owner_name, owned in _t_flush_owner_components(shape):
        if owned is None:
            continue
        cutter = _round_tube(
            _owner_cutter_points(ts_points, "lm"), TS_CUTTER_R)
        owned = (owned - cutter).clean()
        pieces.extend(
            solid for solid in owned.solids()
            if solid.volume > 1e-9)
    if not pieces:
        return None
    return pieces[0].fuse(*pieces[1:]).clean()

def _required_solids(shape, label):
    if shape is None or not shape.is_valid:
        raise RuntimeError(f"{label}: crop returned no valid shape")
    solids = tuple(shape.solids())
    if not solids or any(solid.volume <= 0.01 for solid in solids):
        raise RuntimeError(
            f"{label}: expected positive solids, got "
            f"{[solid.volume for solid in solids]}")
    return solids

def _owned_crop_solids(shape, label, *, allow_empty=False):
    """Return the positive solids that remain in one carrier domain.

    Anchor legs are generated from the complete smooth route so their run
    endpoints stay independent of the LM/UM split.  A complete leg can
    therefore lie on the other carrier and legitimately crop to nothing.
    That empty ownership result is not a failed Boolean; any non-empty,
    invalid or sliver-producing crop remains a hard source-geometry error.
    """
    source_solids = tuple(shape.solids()) if shape is not None else ()
    if not source_solids:
        if allow_empty:
            return ()
        raise RuntimeError(f"{label}: mandatory source cover is empty")
    cropped = _um_owner_crop(shape)
    solids = tuple(cropped.solids()) if cropped is not None else ()
    if not solids:
        if allow_empty:
            return ()
        raise RuntimeError(f"{label}: mandatory round cover has no owner crop")
    if not cropped.is_valid or any(solid.volume <= 0.01 for solid in solids):
        raise RuntimeError(
            f"{label}: invalid owned crop; valid={cropped.is_valid} "
            f"volumes={[solid.volume for solid in solids]}")
    return solids

def _fused_cover_group(solids, label):
    """Unify one tube and all of its positive-overlap anchor/housing solids."""
    solids = tuple(solids)
    if not solids:
        raise RuntimeError(f"{label}: empty cover group")
    grouped = solids[0].fuse(*solids[1:]).clean()
    result = tuple(grouped.solids())
    if (not grouped.is_valid or len(result) != 1
            or result[0].volume <= 0.01):
        raise RuntimeError(
            f"{label}: cover group must be one valid solid; "
            f"valid={grouped.is_valid} volumes="
            f"{[solid.volume for solid in result]}")
    return result[0]

def route_outer_covers(which):
    """Yield continuous cover additions one at a time for bounded memory."""
    _require_guarded_build()
    if which not in ("lm", "um"):
        raise ValueError(which)
    main = route_cable_points(1.8)
    ts = ts_cable_points(1.8)
    if which == "lm":
        main_parts = []
        for index, component in enumerate(
                _anchored_cover_components(
                    main, MAIN_OUTER_R,
                    hardware_keepouts=MAIN_ANCHOR_KEEPOUTS,
                    full_burial_web=True)):
            # The no-floor bridge is already a solid z=5.3..18.3 body and
            # needs only its lumen before the ring.  Both states end the
            # negative lumen 0.05 mm inside structural R113 and the positive
            # cover 0.05 mm inside the visible R113.8 fairing; the native
            # carrier supplies the uninterrupted exterior and the later span
            # is free cable.
            # Only component 0 is the continuous round tube. Every printed
            # main-route component ends at the visible R113.8 LM boundary; the
            # remainder is intentionally free cable behind the UM.
            ring_component = _lm_ring_outer_crop(component)
            if index == 0:
                component = _main_printed_outer_from_full(component)
            else:
                component = ring_component
            if component is None:
                if index > 0:
                    continue
                raise RuntimeError("LM main cover has no R113.8 owner crop")
            solids = tuple(component.solids())
            if (not component.is_valid
                    or any(solid.volume <= 0.01 for solid in solids)):
                raise RuntimeError(
                    f"LM main visible-R113.8 component {index} invalid; "
                    f"valid={component.is_valid} volumes="
                    f"{[solid.volume for solid in solids]}")
            main_parts.extend(solids)
        main_parts.extend(bump_backfill_components("lm", "UM"))
        cover = _fused_cover_group(main_parts, "LM main cover")
        del main_parts
        yield cover
        t_parts = []
        for index, component in enumerate(
                _anchored_cover_components(
                    ts, TS_OUTER_R, omit_crossover=True,
                    hardware_keepouts=T_ANCHOR_KEEPOUTS,
                    full_burial_web=True)):
            component = _crop_path_interval(
                component, ts, TS_ROUTE_LENGTH, 0.0, TS_UM_ENTRY_S)
            # The T outer cover must use exactly the same LM ownership
            # domain as its cutter.  In the floor state the left T entry
            # bends out of the rectangular stem before it reaches the ring; a
            # ring-only crop leaves a real 0.8-mm shell sliver uncovered at
            # z=5.3..6.8.  ``_lm_printed_owner_crop`` is the authoritative
            # union of the native LM ring and the state-specific tail/corridor
            # (the no-floor bridge or the floor-stem handoff), without adding
            # any exterior silhouette outside actual printed material.
            component = _lm_printed_owner_crop(component)
            solids = tuple(component.solids()) if component is not None else ()
            if not solids:
                if index > 0:
                    continue
                raise RuntimeError("LM T round cover has no R113.8 owner crop")
            if (not component.is_valid
                    or any(solid.volume <= 0.01 for solid in solids)):
                raise RuntimeError(
                    f"LM T flush cover component {index} invalid; "
                    f"valid={component.is_valid} volumes="
                    f"{[solid.volume for solid in solids]}")
            t_parts.extend(solids)
        t_parts.extend(bump_backfill_components("lm", "T"))
        cover = _fused_cover_group(t_parts, "LM T cover")
        del t_parts
        yield cover
    else:
        t_parts = []
        for index, component in enumerate(
                _anchored_cover_components(
                    ts, TS_OUTER_R,
                    anchor_base_z=UM_SEAT_MEMBRANE_BOTTOM_Z,
                    omit_crossover=True,
                    hardware_keepouts=T_ANCHOR_KEEPOUTS,
                    support_domains=("um",),
                    full_burial_web=True)):
            t_parts.extend(_owned_crop_solids(
                component, f"UM T cover component {index}",
                allow_empty=index > 0))
        t_parts.extend(bump_backfill_components("um", "T"))
        cover = _fused_cover_group(t_parts, "UM T cover")
        del t_parts
        yield cover

def _contract_outside_halfspace(points, at_start):
    """Authoritative oriented trim beyond one functional route mouth."""
    index = 0 if at_start else -1
    neighbor = 1 if at_start else -2
    tangent = points[neighbor] - points[index]
    if not at_start:
        tangent = -tangent
    tangent = tangent / np.linalg.norm(tangent)
    outward = -tangent if at_start else tangent
    origin = np.asarray(points[index], dtype=float) + 0.05 * outward
    face = Plane(
        origin=tuple(map(float, origin)),
        z_dir=tuple(map(float, outward))) * Rectangle(200.0, 200.0)
    return extrude(face, amount=20.0)

def required_assembled_shell_components(route_name, normal_wall_mm=None):
    """Exact full nominal shell contract, independent of build helpers.

    This deliberately does *not* reuse anchor-leg or production owner-crop
    construction. The contract is the exact shell within the printed owners,
    minus every physical cable envelope and trimmed at intentional mouths.
    Tests subtract final LM+UM BREPs from the result, so any ownership gap,
    opened roof, failed fusion or under-thickness wall remains as positive
    missing volume. At an isolated section the required normal wall is
    exactly outer_radius-inner_radius; crossover clearance is checked against
    the free physical UM cable independently.
    """
    _require_guarded_build()
    if route_name == "LM":
        if STAND_FOOT:
            return ()
        inner_radius = LM_INTERNAL_DUCT_R
        outer_radius = inner_radius + (
            TUNNEL_SKIN if normal_wall_mm is None else normal_wall_mm)
        points = lm_internal_duct_points(1.2)
        shell = _round_tube(points, outer_radius)
        shell -= _round_tube(points, inner_radius)
        lm_entry = next(
            bore for bore in no_floor_rear_entry_bores()
            if bore.name == "lm")
        shell -= _z_axis_bore(
            lm_entry.xy, lm_entry.radius_mm,
            lm_entry.rear_z_mm, lm_entry.inner_z_mm)
        shell -= lm_rear_exit_port_cutter()
        shell = shell.clean()
        shells = tuple(shell.solids())
        if (not shell.is_valid or not shells
                or any(solid.volume <= 0.01 for solid in shells)):
            raise RuntimeError(
                "no-floor LM internal shell contract failed; "
                f"valid={shell.is_valid} volumes="
                f"{[solid.volume for solid in shells]}")
        return shells
    if route_name == "UM":
        points = route_cable_points(1.8)
        inner_radius = CUTTER_R
    elif route_name == "T":
        points = ts_cable_points(1.8)
        inner_radius = TS_CUTTER_R
    else:
        raise ValueError(route_name)
    outer_radius = inner_radius + (
        TUNNEL_SKIN if normal_wall_mm is None else normal_wall_mm)
    outer = _round_tube(points, outer_radius)
    # Both route contracts stop at their last printed owner. The physical
    # centerlines continue so bend and service clearance remain testable.
    if route_name == "UM":
        outer = _main_printed_outer_from_full(outer)
        shell = outer
    else:
        # LM/UM use the state-authoritative core cutter phase. Subtract that
        # owner-specific lumen before joining the butt-mouth domains;
        # otherwise the no-floor contract is globally re-phased.
        shell = _t_owner_phase_shell(outer)
        if shell is None:
            raise RuntimeError("T flush-owner shell contract is empty")
    # Trim only material beyond the two oriented mouth planes.  The former
    # endpoint spheres exempted roughly one full tube radius at precisely the
    # outlet locations that must remain enclosed.  A 0.05-mm outward
    # overshoot avoids a coincident cap while retaining the complete shell up
    # to each functional mouth plane.
    trims = (
        _contract_outside_halfspace(points, True),
        _contract_outside_halfspace(points, False))
    # Stream the nominal buried-route voids. Holding all complete route BREPs
    # beside the outer shell exceeded the release memory floor even though
    # each individual Boolean is comfortably bounded.
    cutter_specs = [
        (_owner_cutter_points(route_cable_points(1.8), "lm"), CUTTER_R),
    ]
    if route_name == "UM":
        cutter_specs.append((
            _owner_cutter_points(ts_cable_points(1.8), "lm"),
            TS_CUTTER_R))
    for cutter_points, cutter_radius in cutter_specs:
        cutter = _round_tube(cutter_points, cutter_radius)
        next_shell = shell - cutter
        del shell, cutter
        shell = next_shell
    # The rear-normal entry/exit bores are part of the final exterior-mouth
    # contract, not hidden omissions in the shell.  Remove them explicitly
    # so the final-BREP shell audit agrees with the manufactured parts.
    if not STAND_FOOT:
        # LM production owns one globally phased D9 entry/tunnel/R14 cutter.
        # Its start passes between the tightly packed UM/T entries, so a
        # separately lofted exit suffix is not the complete manufactured
        # cross-route void.  Subtract that exact one-piece cutter here too.
        shell = shell - no_floor_lm_internal_cutter()
        # T and UM likewise use the same pre-fused transition tools as
        # production; subtracting their overlapping bore/vestibule/relief
        # members sequentially can produce a valid-looking signed-complement
        # BREP instead of the intended void.
        for _entry_name, transition in (
                no_floor_rear_entry_transition_cutters()):
            shell = shell - transition
    else:
        shell = shell - lm_rear_exit_port_cutter()
    for trim in trims:
        shell = shell - trim
    shell = shell.clean()
    shells = tuple(shell.solids())
    if (not shell.is_valid or not shells
            or any(solid.volume <= 0.01 for solid in shells)):
        raise RuntimeError(
            f"{route_name}: required shell contract failed; "
            f"valid={shell.is_valid} volumes="
            f"{[solid.volume for solid in shells]}")
    return shells

def required_assembled_shell_segment_components(
        route_name, segment_index, segment_count, normal_wall_mm=None):
    """Exact globally phased subset of the nominal shell contract.

    Every bounded worker lofts consecutive faces from the same global section
    grid used by ``_round_tube(full_path)``.  Cutter intervals likewise use
    exact consecutive global faces.  Adjacent outer intervals share their
    boundary section, so their union is the authoritative full ruled loft,
    without locally re-phased approximations or seam exemptions.
    """
    _require_guarded_build()
    if not 0 <= segment_index < segment_count:
        raise ValueError((segment_index, segment_count))
    if route_name == "LM":
        raise ValueError("LM lead is a free cable and has no printed shell")
    if route_name == "UM":
        points = route_cable_points(1.8)
        inner_radius = CUTTER_R
    elif route_name == "T":
        points = ts_cable_points(1.8)
        inner_radius = TS_CUTTER_R
    else:
        raise ValueError(route_name)
    outer_radius = inner_radius + (
        TUNNEL_SKIN if normal_wall_mm is None else normal_wall_mm)
    outer_sections = _tube_section_points(points)
    outer_edge_count = len(outer_sections) - 1
    if segment_count > outer_edge_count:
        raise ValueError((segment_count, outer_edge_count))
    first_edge = outer_edge_count * segment_index // segment_count
    last_edge_exclusive = (
        outer_edge_count * (segment_index + 1) // segment_count)
    shell = _round_tube_from_global_sections(
        outer_sections, outer_radius, first_edge, last_edge_exclusive)

    if route_name == "UM":
        pieces = []
        ring_piece = _lm_ring_outer_crop(shell)
        if ring_piece is not None:
            pieces.extend(
                solid for solid in ring_piece.solids()
                if solid.volume > 1e-9)
        if not STAND_FOOT:
            bridge_owner = _polygon_prism(
                box(*NO_FLOOR_BRIDGE_ROUTE_BOUNDS), PAD_FACE_Z, 100.0)
            bridge_piece = shell & bridge_owner
            if bridge_piece is not None:
                pieces.extend(
                    solid for solid in bridge_piece.solids()
                    if solid.volume > 1e-9)
        if not pieces:
            raise RuntimeError(
                f"UM segment {segment_index} has no printed owner")
        shell = pieces[0].fuse(*pieces[1:]).clean()

    selected_outer = outer_sections[first_edge:last_edge_exclusive + 1]
    outer_circumradius = outer_radius / math.cos(
        math.pi / TUBE_SECTION_SIDES)
    outer_min = selected_outer.min(axis=0) - outer_circumradius
    outer_max = selected_outer.max(axis=0) + outer_circumradius

    def subtract_bounded_cutter(base, cutter_points, cutter_radius):
        """Subtract only globally phased cutter edges near this segment."""
        cutter_sections = _tube_section_points(cutter_points)
        cutter_circumradius = cutter_radius / math.cos(
            math.pi / TUBE_SECTION_SIDES)
        edge_min = np.minimum(
            cutter_sections[:-1], cutter_sections[1:]) - cutter_circumradius
        edge_max = np.maximum(
            cutter_sections[:-1], cutter_sections[1:]) + cutter_circumradius
        intersects = np.all(edge_max >= outer_min, axis=1) & np.all(
            edge_min <= outer_max, axis=1)
        edge_indices = np.flatnonzero(intersects)
        if not len(edge_indices):
            return base
        padded = intersects.copy()
        for offset in (1, 2):
            padded[offset:] |= intersects[:-offset]
            padded[:-offset] |= intersects[offset:]
        edge_indices = np.flatnonzero(padded)
        split_at = np.flatnonzero(np.diff(edge_indices) > 1) + 1
        for run in np.split(edge_indices, split_at):
            void = _round_tube_from_global_sections(
                cutter_sections, cutter_radius,
                int(run[0]), int(run[-1]) + 1)
            base = base - void
            del void
        return base

    # The full authoritative contract uses these exact extended-path cutter
    # lofts.  Select every cutter edge whose radius-expanded AABB can meet the
    # bounded outer interval, then include two neighbor edges so the temporary
    # cutter cap is strictly outside that possible intersection. T has only
    # the mandatory core owner.
    if route_name == "T":
        owner_pieces = []
        for owner_name in ("core",):
            owned = _t_flush_owner_crop(shell, owner_name)
            if owned is None:
                continue
            owned = subtract_bounded_cutter(
                owned,
                _owner_cutter_points(
                    ts_cable_points(1.8), "lm"),
                TS_CUTTER_R)
            owner_pieces.extend(
                solid for solid in owned.solids()
                if solid.volume > 1e-9)
        if not owner_pieces:
            raise RuntimeError(
                f"T segment {segment_index} has no printed owner")
        shell = owner_pieces[0].fuse(*owner_pieces[1:]).clean()

    cutter_specs = [
        (_owner_cutter_points(route_cable_points(1.8), "lm"), CUTTER_R),
    ]
    if route_name == "UM":
        cutter_specs.append((
            _owner_cutter_points(ts_cable_points(1.8), "lm"),
            TS_CUTTER_R))
    for cutter_points, cutter_radius in cutter_specs:
        shell = subtract_bounded_cutter(
            shell, cutter_points, cutter_radius)
    # Apply the same production-owned entry cutters to every bounded shard.
    # This is set-correct and avoids both a fragile segment-zero exception and
    # any independent re-phasing of the tightly packed D20 entry cluster.
    if not STAND_FOOT:
        shell = shell - no_floor_lm_internal_cutter()
        for _entry_name, transition in (
                no_floor_rear_entry_transition_cutters()):
            shell = shell - transition
    else:
        shell = shell - lm_rear_exit_port_cutter()

    # Only the complete route's two ends are intentional shell mouths.
    if segment_index == 0:
        shell = shell - _contract_outside_halfspace(points, True)
    if segment_index == segment_count - 1:
        shell = shell - _contract_outside_halfspace(points, False)
    shell = shell.clean()
    shells = tuple(shell.solids())
    if (not shell.is_valid or not shells
            or any(solid.volume <= 1e-9 for solid in shells)):
        raise RuntimeError(
            f"{route_name}: required shell segment {segment_index}/"
            f"{segment_count} failed; valid={shell.is_valid} volumes="
            f"{[solid.volume for solid in shells]}")
    return shells

def required_handoff_shell_components(route_name, owner_filter=None):
    """Obi-Wan route exits are native flush mouths with no printed handoff."""
    _require_guarded_build()
    if route_name in ("LM", "UM", "T"):
        return ()
    raise ValueError(route_name)
