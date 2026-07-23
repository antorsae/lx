"""Obi-Wan rear-entry bores, vestibules, support blocker and tube builders."""

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


@dataclass(frozen=True)
class RearEntryBore:
    """One circular rear-face port with an intentional buried-route overlap."""

    name: str
    xy: tuple[float, float]
    radius_mm: float
    rear_z_mm: float
    inner_z_mm: float

@dataclass(frozen=True)
class RearEntryVestibule:
    """Hidden spherical transition between one Z bore and swept lumen."""

    name: str
    xy: tuple[float, float]
    radius_mm: float
    center_z_mm: float

def _z_axis_bore(xy, radius_mm: float, z0_mm: float, z1_mm: float):
    """Return a Z-axis cylindrical cutter with explicit absolute bounds."""
    if z1_mm <= z0_mm:
        raise ValueError((z0_mm, z1_mm))
    return Pos(
        float(xy[0]), float(xy[1]), (z0_mm + z1_mm) / 2.0,
    ) * Cylinder(radius_mm, z1_mm - z0_mm)

def no_floor_rear_entry_bores():
    """Return all three full rear-normal entries in the no-floor bridge.

    The horizontal core routes own all buried cable travel.  These are only
    the short circular external mouths that make those routes usable from the
    rear face.  Every bore positively overlaps its corresponding lumen.
    """
    if STAND_FOOT:
        return ()
    z0 = NO_FLOOR_FEED_REAR_Z - NO_FLOOR_ENTRY_BORE_REAR_OVERTRAVEL_MM
    z1 = NO_FLOOR_FEED_REAR_Z + NO_FLOOR_ENTRY_BORE_DEPTH_MM
    return (
        RearEntryBore(
            "lm", tuple(map(float, NO_FLOOR_LM_FEED_XY)),
            LM_INTERNAL_DUCT_R, z0,
            NO_FLOOR_LM_ENTRY_BORE_INNER_Z_MM),
        RearEntryBore(
            "t", tuple(map(float, NO_FLOOR_T_FEED_XY)), TS_CUTTER_R,
            z0, z1),
        RearEntryBore(
            "um", tuple(map(float, NO_FLOOR_MAIN_FEED_XY)), CUTTER_R,
            z0, z1),
    )

def no_floor_rear_entry_bore_cutters():
    """Build the no-floor rear entry cutters, one per cable bundle.

    The visible LM/T mouths remain exactly D9/D6.  Buried 0.005-mm radial
    reliefs begin 0.10 mm behind the rear surface so the cylinders and
    horizontal lofts do not leave unmeshable cap slivers.  The reliefs are
    five microns—far below process resolution—and never reach an exterior
    face or change the D20 port contract.
    """
    _require_guarded_build()
    cutters = []
    for bore in no_floor_rear_entry_bores():
        cutter = _z_axis_bore(
            bore.xy, bore.radius_mm, bore.rear_z_mm, bore.inner_z_mm)
        relief_spec = {
            "lm": (NO_FLOOR_LM_ENTRY_RELIEF_RADIAL_MM,
                   NO_FLOOR_LM_ENTRY_RELIEF_REAR_SKIN_MM),
            "t": (NO_FLOOR_T_ENTRY_RELIEF_RADIAL_MM,
                  NO_FLOOR_T_ENTRY_RELIEF_REAR_SKIN_MM),
        }.get(bore.name)
        if relief_spec is not None:
            radial_relief, rear_skin = relief_spec
            relief_z0 = PAD_FACE_Z + rear_skin
            relief = _z_axis_bore(
                bore.xy,
                bore.radius_mm + radial_relief,
                relief_z0, bore.inner_z_mm)
            cutter = cutter.fuse(relief).clean()
            solids = tuple(cutter.solids())
            if (not cutter.is_valid or len(solids) != 1
                    or not solids[0].is_valid):
                raise RuntimeError(
                    f"buried {bore.name.upper()} entry relief failed; "
                    f"valid={cutter.is_valid} volumes="
                    f"{[solid.volume for solid in solids]}")
            cutter = solids[0]
        cutters.append(cutter)
    return tuple(cutters)

def no_floor_rear_entry_vestibules():
    """Return the hidden rounded UM/T bore-to-sweep transitions.

    Each sphere shares the nominal lumen radius and is centered on the first
    swept-route station.  Its rear pole remains 0.10 mm behind the exterior
    rear face, while the matching Z bore reaches through its lower half.  The
    resulting union opens the nominal circular throat without widening any
    visible D20-cluster mouth or creating an exterior raceway.  UM's tiny
    octagonal corner remainder is removed separately by the route-phased cap
    relief below.
    """
    if STAND_FOOT:
        return ()
    return (
        RearEntryVestibule(
            "t", tuple(map(float, NO_FLOOR_T_FEED_XY)), TS_CUTTER_R,
            NO_FLOOR_T_FEED_START_Z),
        RearEntryVestibule(
            "um", tuple(map(float, NO_FLOOR_MAIN_FEED_XY)), CUTTER_R,
            NO_FLOOR_MAIN_FEED_START_Z),
    )

def no_floor_rear_entry_vestibule_cutters():
    """Build hidden spherical UM/T entry transitions."""
    _require_guarded_build()
    return tuple(
        Pos(*vestibule.xy, vestibule.center_z_mm)
        * Sphere(vestibule.radius_mm)
        for vestibule in no_floor_rear_entry_vestibules()
    )

def lm_rear_exit_port_cutter():
    """D9/R14 outlet with a short tangent overlap into the buried lane."""
    _require_guarded_build()
    if not STAND_FOOT:
        # The no-floor R14 outlet is the terminal suffix of one continuous
        # D9 entry/tunnel/exit loft.  Select its exact globally phased
        # sections instead of independently re-lofting the same centerline;
        # the latter differs by ~1 mm3 at octagonal section seams and is not
        # the production void.
        points, handoff_station = _no_floor_lm_complete_cutter_path()
        return _round_tube_global_suffix(
            points, LM_REAR_PORT_R, handoff_station,
            section_spacing_mm=LM_EXIT_TUBE_SECTION_SPACING_MM)
    handoff = lm_rear_handoff_points(0.5)
    tangent_start = np.asarray(
        LM_REAR_HANDOFF_SPEC["plan_tangent"], dtype=float)
    points = np.vstack((
        handoff[0] - LM_REAR_PORT_PREFUSION_MM * tangent_start,
        handoff,
    ))
    return _round_tube(
        points, LM_REAR_PORT_R,
        section_spacing_mm=LM_EXIT_TUBE_SECTION_SPACING_MM)

def no_floor_lm_bottom_support_blocker(
        max_y_mm: float, clearance_mm: float = 0.25):
    """Modifier volume that forbids slicer support inside lower LM lumens.

    The support-blocker is an auxiliary slicer object, never printable model
    geometry.  It follows all three no-floor ducts through the optional lower
    half and grows 0.25 mm into their walls so tessellation cannot leave a
    support sliver.  Its prefix caps stop at ``max_y_mm`` and therefore do not
    enlarge the keyed bottom's assembled/plate envelope.
    """
    _require_guarded_build()
    if STAND_FOOT:
        raise RuntimeError(
            "the internal-duct support blocker is no-floor-only")
    if clearance_mm <= 0.0:
        raise ValueError("support-blocker clearance must be positive")

    def prefix(points, radius):
        points = np.asarray(points, dtype=float)
        target_y = max_y_mm - radius - clearance_mm
        crossings = np.flatnonzero(points[:, 1] >= target_y)
        if not len(crossings):
            return points
        end = int(crossings[0])
        if end == 0:
            raise RuntimeError("support-blocker route begins above split")
        p0, p1 = points[end - 1], points[end]
        fraction = ((target_y - p0[1])
                    / max(p1[1] - p0[1], 1.0e-12))
        crossing = p0 + np.clip(fraction, 0.0, 1.0) * (p1 - p0)
        return np.vstack((points[:end], crossing))

    tools = []
    for points, radius in (
            (lm_internal_duct_cutter_points(1.0), LM_INTERNAL_DUCT_R),
            (route_cable_points(1.0), CUTTER_R),
            (ts_cable_points(1.0), TS_CUTTER_R)):
        clipped = prefix(points, radius)
        tools.append(_round_tube(clipped, radius + clearance_mm))
    for bore in no_floor_rear_entry_bores():
        tools.append(_z_axis_bore(
            bore.xy, bore.radius_mm + clearance_mm,
            bore.rear_z_mm - clearance_mm,
            bore.inner_z_mm + clearance_mm))
    for vestibule in no_floor_rear_entry_vestibules():
        tools.append(
            Pos(*vestibule.xy, vestibule.center_z_mm)
            * Sphere(vestibule.radius_mm + clearance_mm))
    tools.extend(no_floor_rear_entry_cap_relief_cutters(clearance_mm))
    tools.append(_z_axis_bore(
        LM_REAR_PORT_XY, LM_REAR_PORT_R + clearance_mm,
        LM_REAR_PORT_REAR_Z - LM_REAR_PORT_REAR_OVERTRAVEL_MM
        - clearance_mm,
        LM_REAR_PORT_INNER_Z + clearance_mm))
    return Compound(children=tools)

def _slice_points(points, total, start, stop, spacing_mm=1.8):
    """Exact station-preserving subpath with interpolated end planes.

    Ruled polygonal lofts are sensitive to their section stations.  The old
    implementation resampled an already authoritative path, so an owner
    cover and its nominal cutter used shifted octagons over every bend and
    could erase a nominal 0.8-mm wall.  Preserve every original interior
    station and add only the two requested interval endpoints.
    """
    points = np.asarray(points, dtype=float)
    source_s = np.linspace(0.0, total, len(points))
    if not 0.0 <= start < stop <= total:
        raise ValueError((start, stop, total))
    interior = source_s[(source_s > start + 1e-9)
                        & (source_s < stop - 1e-9)]
    stations = np.concatenate(([start], interior, [stop]))
    if len(stations) < 3:
        stations = np.asarray((start, (start + stop) / 2.0, stop))
    return np.column_stack(tuple(
        np.interp(stations, source_s, points[:, axis])
        for axis in range(3)))

def _outside_path_halfspace(segment, at_start):
    """Large oriented cutter beyond one exact path-interval endpoint."""
    segment = np.asarray(segment, dtype=float)
    if at_start:
        endpoint = segment[0]
        tangent = segment[1] - segment[0]
        outward = -tangent
    else:
        endpoint = segment[-1]
        tangent = segment[-1] - segment[-2]
        outward = tangent
    outward /= np.linalg.norm(outward)
    origin = endpoint + 0.02 * outward
    face = Plane(
        origin=tuple(map(float, origin)),
        z_dir=tuple(map(float, outward))) * Rectangle(300.0, 300.0)
    return extrude(face, amount=300.0)

def _crop_path_interval(shape, full_points, total, start, stop):
    """Crop a full-path BREP by station planes without re-lofting it."""
    segment = _slice_points(full_points, total, start, stop)
    if start > 1e-9:
        shape = shape - _outside_path_halfspace(segment, True)
    if stop < total - 1e-9:
        shape = shape - _outside_path_halfspace(segment, False)
    return shape

def _tube_section_points(points, spacing_mm=None):
    """Return the globally phased ruled-loft center sections for a path."""
    points = np.asarray(points, dtype=float)
    spacing_mm = (
        TUBE_SECTION_SPACING if spacing_mm is None else float(spacing_mm))
    if spacing_mm <= 0.0:
        raise ValueError("tube section spacing must be positive")
    source_s = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    section_s = np.linspace(
        0.0, source_s[-1],
        max(2, int(math.ceil(source_s[-1] / spacing_mm))) + 1)
    return np.column_stack(tuple(
        np.interp(section_s, source_s, points[:, axis])
        for axis in range(3)))

def _round_tube_from_global_sections(
        section_points, radius, first_section=0, last_section=None):
    """Loft an exact consecutive subset of globally oriented sections."""
    section_points = np.asarray(section_points, dtype=float)
    if last_section is None:
        last_section = len(section_points) - 1
    if not 0 <= first_section < last_section < len(section_points):
        raise ValueError((first_section, last_section, len(section_points)))
    sides = TUBE_SECTION_SIDES
    circumradius = radius / math.cos(math.pi / sides)
    sections = []
    for index in range(first_section, last_section + 1):
        point = section_points[index]
        before = section_points[max(0, index - 1)]
        after = section_points[min(len(section_points) - 1, index + 1)]
        tangent = after - before
        tangent /= np.linalg.norm(tangent)
        if abs(tangent[2]) < 0.92:
            x_dir = np.asarray((-tangent[1], tangent[0], 0.0))
            x_dir /= np.linalg.norm(x_dir)
        else:
            x_dir = np.asarray((1.0, 0.0, 0.0))
        plane = Plane(
            origin=tuple(map(float, point)),
            x_dir=tuple(map(float, x_dir)),
            z_dir=tuple(map(float, tangent)),
        )
        polygon = [
            (circumradius * math.cos(
                2.0 * math.pi * k / sides + math.pi / sides),
             circumradius * math.sin(
                2.0 * math.pi * k / sides + math.pi / sides))
            for k in range(sides)
        ]
        polygon.append(polygon[0])
        sections.append(
            plane * make_face(Wire(Polyline(*polygon).edges())))
    tube = loft(sections, ruled=True).clean()
    solids = tuple(tube.solids())
    if (not tube.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "round tube loft must be one valid solid; "
            f"valid={tube.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return solids[0]

def _round_tube(points, radius, section_spacing_mm=None):
    """BREP-stable circumscribed polygon tube along a 3-D centerline.

    OCC's Frenet pipe reports a valid solid for these long compound curves
    but classifies its interior backwards in cylinder Booleans (intersection
    returned the cylinder complement). A 5.5-mm ruled section loft produces
    correctly classified, closed solids without OCC's pathological global
    B-spline/surface intersection.  General routes retain the globally phased
    5.5-mm grid; the fishing-critical LM R14 handoff explicitly requests a
    2.0-mm grid (0.036-mm centerline sagitta). Circumscription preserves the
    requested minimum round
    radius at every section; the octagon reduces Boolean face count
    without reducing the inscribed duct diameter.
    """
    return _round_tube_from_global_sections(
        _tube_section_points(points, section_spacing_mm), radius)

def no_floor_rear_entry_cap_relief_cutters(clearance_mm: float = 0.0):
    """Return the hidden route-phased UM start-cap relief.

    The exact D8.2 spherical vestibule remains the minimum functional throat.
    This 1.2-mm-long octagonal micro-extension removes only the corners left
    by the ruled sweep's planar first section.  It is clipped to retain the
    same 0.10-mm rear skin and stays at least 0.805 mm from the buried D9 LM
    entry cutter in the release geometry.
    """
    _require_guarded_build()
    if STAND_FOOT:
        return ()
    if clearance_mm < 0.0:
        raise ValueError("cap-relief clearance cannot be negative")
    points = np.asarray(route_cable_points(0.20), dtype=float)
    tangent = points[1] - points[0]
    tangent /= np.linalg.norm(tangent)
    half_length = (
        NO_FLOOR_UM_ENTRY_CAP_RELIEF_HALF_LENGTH_MM + clearance_mm)
    radius = (
        CUTTER_R - NO_FLOOR_UM_ENTRY_CAP_RELIEF_RADIAL_INSET_MM
        + clearance_mm)
    relief = _round_tube(
        np.vstack((points[0] - half_length * tangent,
                   points[0] + half_length * tangent)),
        radius)
    clip_z0 = (
        PAD_FACE_Z + NO_FLOOR_ENTRY_VESTIBULE_REAR_SKIN_MM
        - clearance_mm)
    clip = _polygon_prism(
        box(-250.0, -50.0, 250.0, 500.0), clip_z0, 50.0)
    relief = (relief & clip).clean()
    solids = tuple(relief.solids())
    if (not relief.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "no-floor UM entry cap relief failed; "
            f"valid={relief.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return (solids[0],)

def no_floor_rear_entry_transition_cutters():
    """Return the fused T/UM rear-entry tools used by every BREP owner.

    Each rear-normal bore overlaps its spherical vestibule; UM additionally
    overlaps the short route-phased cap relief.  Sequential subtraction can
    leave an equal-radius shared edge and a signed-complement BREP.  Fusing
    each exact tool set first preserves the same void while giving OCC one
    closed subtraction solid.  LM's separate D9 cutter already owns its bore.
    """
    _require_guarded_build()
    if STAND_FOOT:
        return ()
    bore_by_name = {
        entry.name: cutter
        for entry, cutter in zip(
            no_floor_rear_entry_bores(),
            no_floor_rear_entry_bore_cutters(), strict=True)
    }
    vestibule_by_name = {
        entry.name: cutter
        for entry, cutter in zip(
            no_floor_rear_entry_vestibules(),
            no_floor_rear_entry_vestibule_cutters(), strict=True)
    }
    um_reliefs = tuple(no_floor_rear_entry_cap_relief_cutters())
    transitions = []
    for entry_name in ("t", "um"):
        tools = [bore_by_name[entry_name], vestibule_by_name[entry_name]]
        if entry_name == "um":
            tools.extend(um_reliefs)
        transition = tools[0].fuse(*tools[1:]).clean()
        solids = tuple(transition.solids())
        if (not transition.is_valid or len(solids) != 1
                or not solids[0].is_valid or solids[0].volume <= 0.01):
            raise RuntimeError(
                f"{entry_name.upper()} rear-entry transition tool failed; "
                f"valid={transition.is_valid} volumes="
                f"{[solid.volume for solid in solids]}")
        transitions.append((entry_name, solids[0]))
    return tuple(transitions)

def no_floor_lm_internal_cutter():
    """Return the one-piece D9 LM entry/tunnel/R14-exit production cutter.

    The three equal-diameter branches must remain one Boolean object: cutting
    them independently leaves coincident end-cap faces at the two T-junctions.
    Keeping this construction public also lets final-BREP contracts deduct
    exactly this intentional void when probing the neighbouring UM/T skins.
    """
    _require_guarded_build()
    if STAND_FOOT:
        raise RuntimeError("the internal LM cutter is no-floor-only")
    points, _handoff_station = _no_floor_lm_complete_cutter_path()
    tunnel = _round_tube(
        points, LM_INTERNAL_DUCT_R,
        section_spacing_mm=LM_EXIT_TUBE_SECTION_SPACING_MM)
    tunnel = tunnel.fuse(
        no_floor_rear_entry_bore_cutters()[0],
    ).clean()
    solids = tuple(tunnel.solids())
    if (not tunnel.is_valid or len(solids) != 1
            or solids[0].volume <= 1e-9):
        raise RuntimeError(
            "no-floor LM internal D9 cutter failed; "
            f"valid={tunnel.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return tunnel

def _round_tube_global_segment(points, radius, index, count):
    """Return one exact consecutive edge group of a full ruled tube."""
    sections = _tube_section_points(points)
    edge_count = len(sections) - 1
    if not 0 <= index < count or count > edge_count:
        raise ValueError((index, count, edge_count))
    first_edge = edge_count * index // count
    last_edge_exclusive = edge_count * (index + 1) // count
    return _round_tube_from_global_sections(
        sections, radius, first_edge, last_edge_exclusive)

def _sampled_arc_station(points, parameter_total, parameter_station):
    """Map one route-plan parameter to the sampled 3-D arc station.

    Route constants such as ``LM_ENTRY_LENGTH`` are measured in the XY plan,
    while Z-first insert bypasses make the sampled 3-D centerline longer.
    Global ruled-section selectors consume true 3-D arc length, so passing a
    plan station directly starts a suffix early whenever a bypass lies in the
    preceding interval.
    """
    points = np.asarray(points, dtype=float)
    if not 0.0 <= parameter_station <= parameter_total:
        raise ValueError((parameter_station, parameter_total))
    parameters = np.linspace(0.0, parameter_total, len(points))
    index = int(np.searchsorted(
        parameters, parameter_station, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)
    span = parameters[index + 1] - parameters[index]
    fraction = (parameter_station - parameters[index]) / max(span, 1e-12)
    cumulative = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(
            np.diff(points, axis=0), axis=1))))
    return float(
        cumulative[index]
        + fraction * (cumulative[index + 1] - cumulative[index]))

def _global_suffix_first_section(
        points, start_station, section_spacing_mm=None):
    """Return the first authoritative section retained by a route suffix."""
    points = np.asarray(points, dtype=float)
    sections = _tube_section_points(points, section_spacing_mm)
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if not 0.0 <= start_station <= total:
        raise ValueError((start_station, total))
    section_stations = np.linspace(0.0, total, len(sections))
    containing_edge = int(np.searchsorted(
        section_stations, start_station, side="right") - 1)
    containing_edge = min(containing_edge, len(sections) - 2)
    return max(0, containing_edge - 2)

def _round_tube_global_suffix(
        points, radius, start_station, section_spacing_mm=None):
    """Return a suffix on the full path's authoritative section phase.

    Selecting consecutive global sections avoids both failure modes of a
    tangent half-space crop on a looping route: retained upstream islands and
    locally re-phased octagons that erode the nominal wall. Two predecessor
    edges provide more than one outer diameter of positive overlap before the
    later owner-domain crop, while the explicit arc-length station cannot
    alias to a spatially nearby point elsewhere on a looping route.
    """
    sections = _tube_section_points(points, section_spacing_mm)
    first = _global_suffix_first_section(
        points, start_station, section_spacing_mm)
    return _round_tube_from_global_sections(
        sections, radius, first, len(sections) - 1)
