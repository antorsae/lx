"""Obi-Wan junction closure bands, lenses, webs and plan ownership."""

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


def _circle_branch_y(center, radius: float, x: float, *, upper: bool):
    """Return one exact circle branch, rejecting out-of-domain samples."""
    dx = x - center[0]
    radicand = radius * radius - dx * dx
    if radicand < -1.0e-9:
        raise ValueError(
            f"x={x:g} lies outside circle {center!r}, R={radius:g}")
    dy = math.sqrt(max(0.0, radicand))
    return center[1] + (dy if upper else -dy)

def _curve_band(xs, lower_ys, upper_ys):
    """One valid sampled band between two ordered tangent curves."""
    if not (len(xs) == len(lower_ys) == len(upper_ys)):
        raise ValueError("curve-band sample lengths differ")
    if len(xs) < 3 or any(
            upper <= lower
            for lower, upper in zip(lower_ys, upper_ys)):
        raise ValueError("curve band must have positive height everywhere")
    points = list(zip(xs, lower_ys))
    points.extend(zip(reversed(xs), reversed(upper_ys)))
    polygon = Polygon(points).buffer(0)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 1.0e-6:
        raise RuntimeError("closure curve band is not a valid positive plan")
    return polygon

def _path_band(lower_points, upper_points):
    """One valid band between paired, possibly non-vertical 2-D paths."""
    if len(lower_points) != len(upper_points) or len(lower_points) < 3:
        raise ValueError("closure path sample lengths differ")
    if any(math.dist(lower, upper) <= 1.0e-5
           for lower, upper in zip(lower_points, upper_points)):
        raise ValueError("closure path band must have positive width")
    points = list(lower_points)
    points.extend(reversed(upper_points))
    polygon = Polygon(points).buffer(0)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 1.0e-6:
        raise RuntimeError("closure path band is not a valid positive plan")
    return polygon

def _path_owner_bands(lower_points, upper_points, overlap_weights):
    """Split a tangent path band with a normal 0.05-mm owner seam.

    The 0.40-mm fusion overlap tapers to zero only after both paths have
    entered the rounded ear boss.  Consequently every exposed boundary is a
    circle/Bezier tangent; the terminal construction chord is buried inside
    existing ear material instead of appearing as a vertical facet.
    """
    if not (len(lower_points) == len(upper_points)
            == len(overlap_weights)):
        raise ValueError("closure owner path sample lengths differ")
    half_gap = JUNCTION_WEB_SEAM_GAP / 2.0
    lower_inner = []
    lower_seam = []
    upper_seam = []
    upper_inner = []
    for lower, upper, weight in zip(
            lower_points, upper_points, overlap_weights):
        dx = upper[0] - lower[0]
        dy = upper[1] - lower[1]
        width = math.hypot(dx, dy)
        if width <= JUNCTION_WEB_SEAM_GAP + 1.0e-5:
            raise ValueError("closure owner band is narrower than its seam")
        nx, ny = dx / width, dy / width
        seam = ((lower[0] + upper[0]) / 2.0,
                (lower[1] + upper[1]) / 2.0)
        overlap = JUNCTION_WEB_OWNER_OVERLAP * weight
        lower_inner.append(
            (lower[0] - overlap * nx, lower[1] - overlap * ny))
        lower_seam.append(
            (seam[0] - half_gap * nx, seam[1] - half_gap * ny))
        upper_seam.append(
            (seam[0] + half_gap * nx, seam[1] + half_gap * ny))
        upper_inner.append(
            (upper[0] + overlap * nx, upper[1] + overlap * ny))
    return (
        _path_band(lower_inner, lower_seam),
        _path_band(upper_seam, upper_inner),
        _path_band(lower_points, upper_points),
    )

def _circle_branch_slope(center, radius: float, x: float, *, upper: bool):
    """Exact dy/dx for one circle branch."""
    dx = x - center[0]
    root = math.sqrt(max(0.0, radius * radius - dx * dx))
    if root <= 1.0e-9:
        raise ValueError("circle tangent is vertical at requested station")
    return (-dx / root) if upper else (dx / root)

def _cubic_point(p0, p1, p2, p3, t: float):
    one = 1.0 - t
    return (
        one ** 3 * p0[0] + 3.0 * one ** 2 * t * p1[0]
        + 3.0 * one * t ** 2 * p2[0] + t ** 3 * p3[0],
        one ** 3 * p0[1] + 3.0 * one ** 2 * t * p1[1]
        + 3.0 * one * t ** 2 * p2[1] + t ** 3 * p3[1],
    )

def _tangent_blend_to_boss(y_at_start, slope_at_start, *,
                           start_x: float, boss_x: float, boss_y: float,
                           boss_radius: float, boss_upper: bool,
                           samples: int):
    """C1 cubic from an exact ring branch into a hidden ear-boss chord."""
    end_x = boss_x + boss_radius - JUNCTION_WEB_EAR_CHORD_INSET
    boss_dx = end_x - boss_x
    boss_root = math.sqrt(max(
        0.0, boss_radius * boss_radius - boss_dx * boss_dx))
    end_y = boss_y + (boss_root if boss_upper else -boss_root)
    end_slope = (-boss_dx / boss_root if boss_upper
                 else boss_dx / boss_root)
    span = end_x - start_x
    if span <= 1.0:
        raise ValueError("junction tangent blend has no useful run")
    # X-handle lengths preserve each exact dy/dx while keeping the Bezier
    # monotone in X.  The shorter boss handle avoids a bulb immediately
    # before the steep circular ear tangent.
    start_handle_x = 0.30 * span
    end_handle_x = 0.12 * span
    p0 = (start_x, y_at_start)
    p1 = (start_x + start_handle_x,
          y_at_start + start_handle_x * slope_at_start)
    p3 = (end_x, end_y)
    p2 = (end_x - end_handle_x,
          end_y - end_handle_x * end_slope)
    return [
        _cubic_point(p0, p1, p2, p3, index / samples)
        for index in range(samples + 1)
    ]

def _mirrored_reversed(points):
    return [(-x, y) for x, y in reversed(points)]

def _lm_um_rear_recess_backfill_plan():
    """Symmetric C1 rear crescents closing route-pinched recess slivers.

    A buffered circular centerline gives tangent circular sides and round
    ends.  Its outer 0.25 mm overlaps the immutable R110.6 lip; its inner
    edge stops at R109.55, well outside the D190 driver opening.  The part's
    normal final route-cutter pass remains authoritative if a future route
    profile ever intersects this conservative backing land.
    """
    half = LM_UM_REAR_BACKFILL_ARC_HALF_SPAN_DEG
    pieces = []
    for center_angle in LM_UM_REAR_BACKFILL_CENTER_ANGLES_DEG:
        samples = [
            _polar_xy(
                L22_CUTOUT[:2], LM_UM_REAR_BACKFILL_CENTER_R,
                center_angle - half + 2.0 * half * index / 48.0)
            for index in range(49)
        ]
        pieces.append(LineString(samples).buffer(
            LM_UM_REAR_BACKFILL_RADIAL_WIDTH_MM / 2.0,
            resolution=32, cap_style=1, join_style=1))
    return unary_union(pieces).buffer(0)

def _lm_um_rear_recess_backfill():
    return _plan_prism(
        _lm_um_rear_recess_backfill_plan(), *LM_UM_REAR_BACKFILL_Z)

def _um_t_rear_recess_backfill_plan():
    """Symmetric rear crescents closing the T-cover/UM-lip blind pocket."""
    half = UM_T_REAR_BACKFILL_ARC_HALF_SPAN_DEG
    pieces = []
    for center_angle in UM_T_REAR_BACKFILL_CENTER_ANGLES_DEG:
        samples = [
            _polar_xy(
                UM_CUTOUT[:2], UM_T_REAR_BACKFILL_CENTER_R,
                center_angle - half + 2.0 * half * index / 48.0)
            for index in range(49)
        ]
        pieces.append(LineString(samples).buffer(
            UM_T_REAR_BACKFILL_RADIAL_WIDTH_MM / 2.0,
            resolution=32, cap_style=1, join_style=1))
    return unary_union(pieces).buffer(0)

def _um_t_rear_recess_backfill():
    return _plan_prism(
        _um_t_rear_recess_backfill_plan(), *UM_T_REAR_BACKFILL_Z)

def _bounded_plan_lenses(material, window, *, max_area: float = 20.0):
    """Return only small bounded holes in one independent local silhouette."""
    merged = unary_union(material).buffer(0)
    pieces = ((merged,) if merged.geom_type == "Polygon"
              else tuple(merged.geoms))
    lenses = []
    for piece in pieces:
        if piece.geom_type != "Polygon":
            continue
        for ring in piece.interiors:
            lens = Polygon(ring)
            if (0.01 < lens.area <= max_area
                    and window.covers(lens.representative_point())):
                lenses.append(lens)
    return (unary_union(lenses).buffer(0) if lenses
            else Polygon())

def _owned_lens_addition(lenses, support, opposing_plan):
    """Fill a lens through one qualified extrusion-path-wide fusion land."""
    if lenses.is_empty:
        return lenses
    overlap = lenses.buffer(
        JUNCTION_WEB_LENS_FUSION_MM, join_style=1).intersection(support)
    return unary_union((lenses, overlap)).difference(
        opposing_plan.buffer(JUNCTION_WEB_SEAM_GAP / 2.0, join_style=1)
    ).buffer(0)

def _printable_lens_components(geometry):
    """Drop route-keepout Boolean dust that cannot form a wall path."""
    kept = [
        piece for piece in _plan_polygon_components(geometry)
        if piece.area >= JUNCTION_WEB_MIN_LENS_AREA_MM2
    ]
    return (unary_union(kept).buffer(0) if kept else Polygon())

def _partition_lens_components(lenses, first_support, second_support):
    """Assign every complete bounded lens to exactly one printable owner."""
    merged = unary_union(lenses).buffer(0)
    if merged.is_empty:
        return Polygon(), Polygon()
    pieces = ((merged,) if merged.geom_type == "Polygon"
              else tuple(merged.geoms))
    first = []
    second = []
    for lens in pieces:
        if lens.is_empty or lens.area <= 1.0e-8:
            continue
        probe = lens.buffer(0.06, join_style=1)
        first_contact = probe.intersection(first_support).area
        second_contact = probe.intersection(second_support).area
        (first if first_contact >= second_contact else second).append(lens)
    return (
        unary_union(first).buffer(0) if first else Polygon(),
        unary_union(second).buffer(0) if second else Polygon(),
    )

def _terminal_fit_drains(xs, y: float, boss_d: float):
    """Continue a 0.05-mm fit seam through each rounded boss perimeter."""
    drain_half = JUNCTION_WEB_SEAM_GAP / 2.0
    pieces = []
    for x in xs:
        sign = -1.0 if x < 0.0 else 1.0
        start = abs(x) + boss_d / 2.0 \
            - JUNCTION_WEB_EAR_CHORD_INSET - 0.02
        end = abs(x) + boss_d / 2.0 + 0.05
        pieces.append(LineString((
            (sign * start, y),
            (sign * end, y),
        )).buffer(drain_half, cap_style=2, join_style=1))
    return unary_union(pieces).buffer(0)

def _closure_owner_bands(xs, lower_ys, upper_ys):
    """Split one closure band into complementary full-depth plan owners."""
    half_gap = JUNCTION_WEB_SEAM_GAP / 2.0
    seam = [0.5 * (lower + upper)
            for lower, upper in zip(lower_ys, upper_ys)]
    lower_owner = _curve_band(
        xs,
        [value - JUNCTION_WEB_OWNER_OVERLAP for value in lower_ys],
        [value - half_gap for value in seam],
    )
    upper_owner = _curve_band(
        xs,
        [value + half_gap for value in seam],
        [value + JUNCTION_WEB_OWNER_OVERLAP for value in upper_ys],
    )
    target = _curve_band(xs, lower_ys, upper_ys)
    return lower_owner, upper_owner, target

def lm_um_closure_polygons():
    """Return complementary full-depth LM/UM junction plan owners.

    The central boundaries follow the exact LM-top and UM-bottom circles.
    Beyond x=+/-20, paired C1 cubic curves enter a chord 0.40 mm inside each
    D9 ear boss while matching both the ring and boss-circle tangents.  Thus
    no constant-X construction cap is exposed.  The owner seam follows the
    path midline, while the 0.40-mm fusion overlap tapers only inside the ear.
    Existing complementary receiver recuts establish clearance in the
    opposing Z half. Route cutters are applied after these solids are fused,
    so only their intentional cable passages can reopen the junction.
    """
    from .joints import joint_ear_polygon

    blend_samples = JUNCTION_WEB_SAMPLES // 3
    central_samples = JUNCTION_WEB_SAMPLES // 2
    start_x = LM_UM_WEB_BLEND_START_X
    lower_right = _tangent_blend_to_boss(
        _circle_branch_y(L22_CUTOUT[:2], LM_CORE_R, start_x, upper=True),
        _circle_branch_slope(
            L22_CUTOUT[:2], LM_CORE_R, start_x, upper=True),
        start_x=start_x, boss_x=abs(JOINT_EAR_X[1]),
        boss_y=JOINT_EAR_Y, boss_radius=JOINT_BOSS_D / 2.0,
        boss_upper=False, samples=blend_samples)
    upper_right = _tangent_blend_to_boss(
        _circle_branch_y(UM_CUTOUT[:2], UM_CORE_R, start_x, upper=False),
        _circle_branch_slope(
            UM_CUTOUT[:2], UM_CORE_R, start_x, upper=False),
        start_x=start_x, boss_x=abs(JOINT_EAR_X[1]),
        boss_y=JOINT_EAR_Y, boss_radius=JOINT_BOSS_D / 2.0,
        boss_upper=True, samples=blend_samples)
    central_xs = [
        -start_x + 2.0 * start_x * index / central_samples
        for index in range(central_samples + 1)
    ]
    central_lower = [
        (x, _circle_branch_y(L22_CUTOUT[:2], LM_CORE_R, x, upper=True))
        for x in central_xs
    ]
    central_upper = [
        (x, _circle_branch_y(UM_CUTOUT[:2], UM_CORE_R, x, upper=False))
        for x in central_xs
    ]
    lower = (_mirrored_reversed(lower_right)
             + central_lower[1:-1] + lower_right)
    upper = (_mirrored_reversed(upper_right)
             + central_upper[1:-1] + upper_right)
    taper = [
        1.0 - (3.0 * (index / blend_samples) ** 2
               - 2.0 * (index / blend_samples) ** 3)
        for index in range(blend_samples + 1)
    ]
    weights = list(reversed(taper)) + [1.0] * (central_samples - 1) + taper
    lm_plan, um_plan, target = _path_owner_bands(lower, upper, weights)
    lm_disk = Point(*L22_CUTOUT[:2]).buffer(LM_CORE_R, resolution=128)
    um_disk = Point(*UM_CUTOUT[:2]).buffer(UM_CORE_R, resolution=128)
    silhouette = unary_union((lm_disk, um_disk, target)).buffer(0)
    lm_ears = unary_union([
        joint_ear_polygon("lm", x).difference(
            um_plan.intersection(target)) for x in JOINT_EAR_X
    ]).buffer(0)
    um_ears = unary_union([
        joint_ear_polygon("um", x).difference(
            lm_plan.intersection(target)) for x in JOINT_EAR_X
    ]).buffer(0)
    window = Polygon(((-45.0, 306.0), (45.0, 306.0),
                      (45.0, 330.0), (-45.0, 330.0)))
    lm_lenses = _bounded_plan_lenses((silhouette, lm_ears), window)
    um_lenses = _bounded_plan_lenses((silhouette, um_ears), window)
    lm_lens_add = _owned_lens_addition(
        lm_lenses, unary_union((lm_disk, lm_ears, lm_plan)), um_plan)
    um_lens_add = _owned_lens_addition(
        um_lenses, unary_union((um_disk, um_ears, um_plan)), lm_plan)
    lm_plan = unary_union((lm_plan, lm_lens_add)).buffer(0)
    um_plan = unary_union((um_plan, um_lens_add)).buffer(0)
    target = unary_union((target, lm_lenses, um_lenses)).buffer(0)
    ear_keepout = unary_union([
        joint_ear_polygon(owner, x, JUNCTION_WEB_EAR_CLEAR)
        for owner in ("lm", "um")
        for x in JOINT_EAR_X
    ])
    # Do not erase the ear neighbourhood from both owners here.  Each
    # full-depth web is fused first; the existing complementary half-lap
    # receiver cuts then remove only the opposing ear in its own Z half.
    # Subtracting a shared plan keepout at this stage leaves the visible
    # triangular moat that this closure is specifically intended to remove.
    fit_seam = target.difference(
        unary_union((lm_plan, um_plan))).buffer(0)
    terminal_drain = _terminal_fit_drains(
        JOINT_EAR_X, JOINT_EAR_Y, JOINT_BOSS_D)
    audit_domain = unary_union((target, terminal_drain)).buffer(0)
    return {
        "lm": lm_plan,
        "um": um_plan,
        "target": target,
        "audit_domain": audit_domain,
        "fit_seam": fit_seam,
        "terminal_drain": terminal_drain,
        "terminal_chords": unary_union((
            LineString((lower[0], upper[0])),
            LineString((lower[-1], upper[-1])),
        )),
        "closure_lenses": unary_union((lm_lenses, um_lenses)).buffer(0),
        "ear_keepout": ear_keepout,
    }

def _t_crescent_boundary_y(x: float):
    """Mirrored lower B2 crescent-arc branch at one world x station."""
    # The released left arc is the mirror of the right within 0.002 mm;
    # using one mirrored authority makes the two printable closure halves
    # exactly symmetric and remains inside the measured B2 outline error.
    mirrored_x = abs(x)
    return _circle_branch_y(
        T_CRESCENT_ARC_CENTER, T_CRESCENT_ARC_R,
        mirrored_x, upper=False)

def t_um_closure_polygons():
    """Return complementary UM/crescent closure plans around the T mouth."""
    from .joints import tweeter_joint_polygon

    um_pieces = []
    crescent_pieces = []
    target_pieces = []
    terminal_chords = []
    ear_keepouts = []
    for sign in (-1.0, 1.0):
        exact_samples = JUNCTION_WEB_SAMPLES // 4
        blend_samples = JUNCTION_WEB_SAMPLES // 3
        exact_xs = [
            T_UM_CABLE_MOUTH_HALF_WIDTH
            + (T_UM_WEB_BLEND_START_X - T_UM_CABLE_MOUTH_HALF_WIDTH)
            * index / exact_samples
            for index in range(exact_samples + 1)
        ]
        start_x = T_UM_WEB_BLEND_START_X
        lower_blend = _tangent_blend_to_boss(
            _circle_branch_y(UM_CUTOUT[:2], UM_CORE_R, start_x,
                             upper=True),
            _circle_branch_slope(
                UM_CUTOUT[:2], UM_CORE_R, start_x, upper=True),
            start_x=start_x, boss_x=abs(TWEETER_JOINT_X[1]),
            boss_y=TWEETER_JOINT_Y,
            boss_radius=TWEETER_JOINT_BOSS_D / 2.0,
            boss_upper=False, samples=blend_samples)
        upper_blend = _tangent_blend_to_boss(
            _t_crescent_boundary_y(start_x),
            _circle_branch_slope(
                T_CRESCENT_ARC_CENTER, T_CRESCENT_ARC_R,
                start_x, upper=False),
            start_x=start_x, boss_x=abs(TWEETER_JOINT_X[1]),
            boss_y=TWEETER_JOINT_Y,
            boss_radius=TWEETER_JOINT_BOSS_D / 2.0,
            boss_upper=True, samples=blend_samples)
        lower = [
            (x, _circle_branch_y(UM_CUTOUT[:2], UM_CORE_R, x,
                                 upper=True))
            for x in exact_xs
        ][:-1] + lower_blend
        upper = [(x, _t_crescent_boundary_y(x)) for x in exact_xs][:-1] \
            + upper_blend
        taper = [
            1.0 - (3.0 * (index / blend_samples) ** 2
                   - 2.0 * (index / blend_samples) ** 3)
            for index in range(blend_samples + 1)
        ]
        weights = [1.0] * exact_samples + taper
        if sign < 0.0:
            lower = _mirrored_reversed(lower)
            upper = _mirrored_reversed(upper)
            weights = list(reversed(weights))
        um_plan, crescent_plan, target = _path_owner_bands(
            lower, upper, weights)
        ear_x = sign * abs(TWEETER_JOINT_X[1])
        keepout = tweeter_joint_polygon(
            ear_x, TWEETER_JOINT_CLEAR)
        # As at LM--UM, fuse both full-depth web owners first and let the
        # existing complementary Z-half receiver cuts establish the fit
        # clearance.  A double-sided plan keepout would leave a visible moat.
        um_pieces.append(um_plan)
        crescent_pieces.append(crescent_plan)
        target_pieces.append(target)
        terminal_chords.append(LineString((lower[-1], upper[-1]))
                               if sign > 0.0 else
                               LineString((lower[0], upper[0])))
        ear_keepouts.append(keepout)
    target = unary_union(target_pieces).buffer(0)
    um_disk = Point(*UM_CUTOUT[:2]).buffer(UM_CORE_R, resolution=128)
    crescent_disk = Point(*T_CRESCENT_ARC_CENTER).buffer(
        T_CRESCENT_ARC_R, resolution=128)
    silhouette = unary_union((um_disk, crescent_disk, target)).buffer(0)
    raw_ears = unary_union([
        tweeter_joint_polygon(x) for x in TWEETER_JOINT_X
    ]).buffer(0)
    um_plan = unary_union(um_pieces).buffer(0)
    tweeter_plan = unary_union(crescent_pieces).buffer(0)
    core_ears = raw_ears.difference(
        tweeter_plan.intersection(target)).buffer(0)
    addon_ears = raw_ears.difference(
        um_plan.intersection(target)).buffer(0)
    window = Polygon(((-35.0, 408.0), (35.0, 408.0),
                      (35.0, 429.0), (-35.0, 429.0)))
    route_keepout = Polygon((
        (-T_UM_CABLE_MOUTH_HALF_WIDTH, 412.0),
        (T_UM_CABLE_MOUTH_HALF_WIDTH, 412.0),
        (T_UM_CABLE_MOUTH_HALF_WIDTH, 425.0),
        (-T_UM_CABLE_MOUTH_HALF_WIDTH, 425.0),
    ))
    raw_um_lenses = _printable_lens_components(
        _bounded_plan_lenses(
            (silhouette, core_ears), window).difference(route_keepout))
    raw_tweeter_lenses = _printable_lens_components(
        _bounded_plan_lenses(
            (silhouette, addon_ears), window).difference(route_keepout))
    um_support = unary_union((um_disk, core_ears, um_plan)).buffer(0)
    tweeter_support = unary_union(
        (crescent_disk, addon_ears, tweeter_plan)).buffer(0)
    # The same geometric hole can be discovered from both half-lap states.
    # Allocate each whole connected lens once; independently adding both
    # detections would make the printable owners collide at the front plane.
    um_lenses, tweeter_lenses = _partition_lens_components(
        (raw_um_lenses, raw_tweeter_lenses),
        um_support, tweeter_support)
    um_lens_add = _owned_lens_addition(
        um_lenses, um_support, tweeter_plan)
    tweeter_lens_add = _owned_lens_addition(
        tweeter_lenses, tweeter_support, um_plan)
    um_plan = unary_union((um_plan, um_lens_add)).difference(
        route_keepout).buffer(0)
    tweeter_plan = unary_union(
        (tweeter_plan, tweeter_lens_add)).difference(
            route_keepout).buffer(0)
    target = unary_union(
        (target, um_lenses, tweeter_lenses)).buffer(0)
    # The two natural cusp bands terminate where the UM and crescent outlines
    # meet; between them the outlines overlap rather than enclose another
    # positive-height band.  The central rectangle is therefore only the
    # explicit free-cable mouth keepout, not material deleted from a fictive
    # closure web.
    fit_seam = target.difference(
        unary_union((um_plan, tweeter_plan))).buffer(0)
    # The normal 0.05-mm parting seam terminates inside each rounded boss.
    # Continue it only far enough to reach the boss perimeter, so the seam is
    # externally vented at every Z instead of becoming a sealed front-plane
    # island when the blind add-on receiver is inactive.  This is a fit seam,
    # not a second cable opening; the only service mouth remains central.
    terminal_drain = _terminal_fit_drains(
        TWEETER_JOINT_X, TWEETER_JOINT_Y, TWEETER_JOINT_BOSS_D)
    audit_domain = unary_union((target, terminal_drain)).buffer(0)
    return {
        "um": um_plan,
        "tweeter": tweeter_plan,
        "target": target,
        "audit_domain": audit_domain,
        "fit_seam": fit_seam,
        "terminal_drain": terminal_drain,
        # Planning keepout only.  It is not accepted as an opening by the
        # closure tests: the exact route-cutter/mouth BREP is qualified by the
        # dedicated floor/no-floor native-mouth checks.
        "route_keepout": route_keepout,
        "terminal_chords": unary_union(terminal_chords),
        "closure_lenses": unary_union(
            (um_lenses, tweeter_lenses)).buffer(0),
        "ear_keepout": unary_union(ear_keepouts).buffer(0),
    }

def junction_closure_polygons():
    """Analytic plan authority shared by CAD, tests, and routing views."""
    return {"lm_um": lm_um_closure_polygons(),
            "t_um": t_um_closure_polygons()}

def _junction_closure_web(junction: str, owner: str):
    plan = junction_closure_polygons()[junction][owner]
    if plan.is_empty or plan.area <= 0.01:
        raise RuntimeError(f"empty {junction}/{owner} closure plan")
    return _plan_prism(plan, *JUNCTION_WEB_Z)

def _enforce_junction_plan_ownership(part, junction: str, owner: str):
    """Remove every opposing full-depth owner only inside the shared target.

    Released ring/crescent sources predate the closure split and may already
    occupy part of the local cusp.  This final complementary mask prevents
    that legacy material (or a Z-half ear) from colliding with the other
    independently printable owner, without cutting any material outside the
    common closure envelope.
    """
    record = junction_closure_polygons()[junction]
    if owner not in record:
        raise RuntimeError(f"no opposing owner for {junction}/{owner}")
    # Remove the opposing owner *and* the real 0.05-mm fit seam.  Subtracting
    # only the opposing material lets a legacy ring blank silently refill the
    # seam and can turn it into a bounded cavity when the half-lap ear closes.
    relief = record["target"].difference(record[owner]).buffer(0)
    if "terminal_drain" in record:
        relief = unary_union((relief, record["terminal_drain"])).buffer(0)
    return _subtract_plan_prisms(part, relief, *JUNCTION_WEB_Z)
