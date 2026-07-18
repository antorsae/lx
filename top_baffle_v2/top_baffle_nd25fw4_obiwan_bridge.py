"""Universal Obi-Wan LM-lower front profile and no-floor bridge load screen.

The four stock bridge XY datums are immutable.  In no-floor mode they are
carried by one solid, shallow web fused directly into the LM carrier.  The
web occupies exactly the deepest existing LM carrier envelope (the z=5.3
insert-pad rear datum through the z=18.3 front): it is flush with the front,
has no rear X-frame or separate depth ribs, and exposes four blind insert
bores from the rear.

Floor and no-floor carriers share one exact wing-contact outline: the union
of the historic integral-floor stem and no-floor bridge outlines.  The
no-floor web owns that complete shallow plan.  Floor mode adds only its
missing shoulder delta through the normal z=6.8..18.3 wing-contact depth, so
the deep floor leg and the four no-floor blind inserts remain state-specific.
"""

from __future__ import annotations

from functools import lru_cache
import math

from build123d import Compound, Cylinder, Face, Part, Polyline, Pos, Wire, extrude
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from top_baffle_nd25fw4 import (
    BRIDGE_HOLE_XY,
    BRIDGE_INSERT_D_MM,
    BRIDGE_INSERT_DEPTH_MM,
    L22_CUTOUT,
    L22_PILOT_D_MM,
    THICKNESS_MM,
)
from top_baffle_nd25fw4_flush import LM_RECESS_R, PAD_FACE_Z
from top_baffle_nd25fw4_obiwan_route import (
    DUCT_D,
    LM_ROUTE_ARC_START_DEG,
    LM_ROUTE_R,
    NO_FLOOR_FEED_CUTTER_EXTENSION,
    NO_FLOOR_FEED_END_HANDLE,
    NO_FLOOR_FEED_START_BEARING_DEG,
    NO_FLOOR_FEED_START_HANDLE,
    NO_FLOOR_MAIN_FEED_XY,
    NO_FLOOR_T_FEED_XY,
    TS_DUCT_D,
    TS_LM_ARC_START_DEG,
)
from top_baffle_nd25fw4_obiwan_floor import integral_stem_plan_points

# The web matches the deepest existing LM carrier datum: the six insert-pad
# rear faces at z=5.3. It never reaches the old global z=0 sheet and adds no
# depth beyond the LM carrier envelope. A 6.8-mm blind insert leaves a 6.2-mm
# solid front floor. The 13.0-mm depth is fixed by that existing envelope;
# strength is screened below against the route-cut net section, not a nominal
# opening-free rectangle.
BRIDGE_WEB_REAR_Z = PAD_FACE_Z
BRIDGE_WEB_FRONT_Z = THICKNESS_MM
BRIDGE_FACE_Z = (BRIDGE_WEB_REAR_Z, BRIDGE_WEB_FRONT_Z)
BRIDGE_WEB_T = BRIDGE_WEB_FRONT_Z - BRIDGE_WEB_REAR_Z
BRIDGE_BORE_FLOOR_MM = BRIDGE_WEB_T - BRIDGE_INSERT_DEPTH_MM

# Solid panel around the unchanged 40 x 50 hole pattern. Its lower rounded
# rectangle is the immutable insert-bearing core; filled cubic shoulders blend
# its vertical sides tangentially into the LM outer ring. The two rear feed
# lumens now pass through the upper panel, so both diameters are conservatively
# removed from the complete 62-mm core even though the cubic shoulders make
# every real horizontal cut wider. No strength credit is taken for the thin
# tunnel skins themselves.
BRIDGE_WEB_X = (-31.0, 31.0)
BRIDGE_WEB_Y = (14.0, 90.25)
BRIDGE_WEB_CORNER_R = 4.0
BRIDGE_WEB_WIDTH = BRIDGE_WEB_X[1] - BRIDGE_WEB_X[0]
BRIDGE_WEB_HEIGHT = BRIDGE_WEB_Y[1] - BRIDGE_WEB_Y[0]
BRIDGE_WEB_TUNNEL_DEDUCTION_MM = DUCT_D + TS_DUCT_D
BRIDGE_WEB_NET_WIDTH_MM = (
    BRIDGE_WEB_WIDTH - BRIDGE_WEB_TUNNEL_DEDUCTION_MM)

# Screen the first uninterrupted horizontal ligament above the immutable top
# insert pair. The 0.05-mm offset avoids crediting a tangent to the D6.4 blind
# bores. Both rear-facing cutter extensions already cross this section, so the
# full 14.2-mm lumen deduction applies. A separate sampled outline/corridor
# audit below proves that the real soft-blend section is wider than this 47.8
# mm lower bound.
BRIDGE_ROUTE_SECTION_Y_MM = (
    max(float(y) for _x, y in BRIDGE_HOLE_XY)
    + L22_PILOT_D_MM / 2.0 + 0.05)
BRIDGE_ROUTE_SECTION_Y_RANGE = (
    BRIDGE_ROUTE_SECTION_Y_MM, BRIDGE_WEB_Y[1])
BRIDGE_ROUTE_SECTION_SAMPLE_MM = 0.01

# An annular-sector union spreads the shallow web into the lower LM structural
# lip. It is not a rear rib: its physical z=5.3..18.3 faces remain coplanar
# with the web/carrier envelope. The existing annular LM lip itself begins at
# z=6.8, however, so the load screen credits only the real 11.5-mm monolithic
# overlap—not the cradle's unsupported rear 1.5 mm. Every printed route that
# can cross this lower sector is still conservatively deducted.
BRIDGE_FUSION_CRADLE_CENTER_DEG = 270.0
BRIDGE_FUSION_CRADLE_SPAN_DEG = 68.0
BRIDGE_FUSION_CRADLE_INNER_R = LM_RECESS_R
BRIDGE_FUSION_CRADLE_OUTER_R = LM_RECESS_R + 2.4
BRIDGE_FUSION_CRADLE_Z = BRIDGE_FACE_Z
BRIDGE_FUSION_INTERFACE_Z = (6.8, BRIDGE_WEB_FRONT_Z)
BRIDGE_FUSION_INTERFACE_T = (
    BRIDGE_FUSION_INTERFACE_Z[1] - BRIDGE_FUSION_INTERFACE_Z[0])
BRIDGE_FUSION_TUNNEL_DEDUCTION_MM = DUCT_D + TS_DUCT_D

# The wing itself occupies z=6.8..18.3.  Both stand states must therefore
# expose exactly the same external LM-lower plan through this complete depth.
# A 0.10-mm inward strip makes the floor-only compatibility addition a robust
# positive-volume union without changing the shared exterior silhouette.
LM_WING_CONTACT_Z = BRIDGE_FUSION_INTERFACE_Z
LM_WING_CONTACT_FUSION_OVERLAP_MM = 0.10

# The old panel met the annular cradle through a short rounded top edge. Its
# two near-vertical corners read as a separate rounded square. The finished
# silhouette now leaves each x=+/-31 side with a vertical tangent at y=60,
# then follows one sampled cubic to a tangency on the R113 LM outer circle.
# The symmetric tangencies sit at 270+/-46 degrees. Filling the complete
# lower-circle region between those curves creates a continuous, gently
# flared load path; it adds no rear depth and does not move any insert axis.
# The lower LM magnets now live on the straight W64 base sides, so this
# shoulder tangency is independent of every captive magnet cavity.
BRIDGE_BLEND_START_Y = 60.0
BRIDGE_BLEND_RING_OFFSET_DEG = 45.5
BRIDGE_BLEND_RIGHT_ANGLE_DEG = (
    BRIDGE_FUSION_CRADLE_CENTER_DEG + BRIDGE_BLEND_RING_OFFSET_DEG)
BRIDGE_BLEND_LEFT_ANGLE_DEG = (
    BRIDGE_FUSION_CRADLE_CENTER_DEG - BRIDGE_BLEND_RING_OFFSET_DEG)
BRIDGE_BLEND_START_HANDLE_MM = 24.0
BRIDGE_BLEND_END_HANDLE_MM = 40.0
BRIDGE_BLEND_SAMPLES = 96


def _circle_xy(radius: float, angle_deg: float):
    angle = math.radians(angle_deg)
    return (
        L22_CUTOUT[0] + radius * math.cos(angle),
        L22_CUTOUT[1] + radius * math.sin(angle),
    )


BRIDGE_BLEND_RIGHT_TANGENCY = _circle_xy(
    BRIDGE_FUSION_CRADLE_OUTER_R, BRIDGE_BLEND_RIGHT_ANGLE_DEG)
BRIDGE_BLEND_LEFT_TANGENCY = _circle_xy(
    BRIDGE_FUSION_CRADLE_OUTER_R, BRIDGE_BLEND_LEFT_ANGLE_DEG)

# The old 45.06-mm R113 chord preceded the filled cubic shoulders and the two
# bridge-entry lumens; it is no longer the physical horizontal section. The
# member screen instead uses the opening-aware 62 - 8.2 - 6.0 = 47.8-mm core
# lower bound at the first ligament above the top insert pair. This is smaller
# than the exact sampled soft-outline net section and therefore conservative.
BRIDGE_GOVERNING_NECK_WIDTH_MM = BRIDGE_WEB_NET_WIDTH_MM

# Deliberately derated PLA Tough+ structural model.
BRIDGE_DESIGN_MASS_KG = 4.0
BRIDGE_DESIGN_Y_CG = 230.0
BRIDGE_REAR_CG_MM = 70.0
BRIDGE_INSERT_PULLOUT_N = 600.0
BRIDGE_SHORT_ALLOW_MPA = 18.0
BRIDGE_CREEP_ALLOW_MPA = 8.0
BRIDGE_MIN_MEMBER_SF_5G = 1.05
BRIDGE_MIN_FUSION_SF_5G = 1.40


def _plan_prism(polygon, z0: float, z1: float):
    """Extrude a Shapely plan while preserving any intentional openings."""
    if polygon.geom_type != "Polygon":
        pieces = [_plan_prism(poly, z0, z1) for poly in polygon.geoms]
        return Compound(children=pieces)
    polygon = orient(polygon, sign=1.0)
    outer = Wire(Polyline(*[
        (float(x), float(y)) for x, y in polygon.exterior.coords
    ]).edges())
    holes = [
        Wire(Polyline(*[(float(x), float(y)) for x, y in ring.coords]).edges())
        for ring in polygon.interiors
    ]
    return Pos(0.0, 0.0, z0) * extrude(Face(outer, holes), amount=z1 - z0)


def bridge_solid_web_plan():
    """Opening-free insert-bearing core retained inside the soft outline."""
    x0, x1 = BRIDGE_WEB_X
    y0, y1 = BRIDGE_WEB_Y
    r = BRIDGE_WEB_CORNER_R
    return box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(
        r, resolution=32, cap_style=1, join_style=1)


def _cubic_xy(p0, p1, p2, p3, count: int):
    points = []
    for index in range(count + 1):
        u = index / count
        v = 1.0 - u
        points.append((
            v ** 3 * p0[0]
            + 3.0 * v ** 2 * u * p1[0]
            + 3.0 * v * u ** 2 * p2[0]
            + u ** 3 * p3[0],
            v ** 3 * p0[1]
            + 3.0 * v ** 2 * u * p1[1]
            + 3.0 * v * u ** 2 * p2[1]
            + u ** 3 * p3[1],
        ))
    return points


def bridge_soft_blend_plan():
    """Filled symmetric cubic transition from the panel into the LM ring."""
    start = (BRIDGE_WEB_X[1], BRIDGE_BLEND_START_Y)
    end = BRIDGE_BLEND_RIGHT_TANGENCY
    angle = math.radians(BRIDGE_BLEND_RIGHT_ANGLE_DEG)
    tangent = (-math.sin(angle), math.cos(angle))
    right = _cubic_xy(
        start,
        (start[0], start[1] + BRIDGE_BLEND_START_HANDLE_MM),
        (end[0] - BRIDGE_BLEND_END_HANDLE_MM * tangent[0],
         end[1] - BRIDGE_BLEND_END_HANDLE_MM * tangent[1]),
        end,
        BRIDGE_BLEND_SAMPLES,
    )
    left = [(-x, y) for x, y in right]
    angles = [
        BRIDGE_BLEND_RIGHT_ANGLE_DEG
        + (BRIDGE_BLEND_LEFT_ANGLE_DEG - BRIDGE_BLEND_RIGHT_ANGLE_DEG)
        * index / (2 * BRIDGE_BLEND_SAMPLES)
        for index in range(2 * BRIDGE_BLEND_SAMPLES + 1)
    ]
    lower_ring_arc = [
        _circle_xy(BRIDGE_FUSION_CRADLE_OUTER_R, angle_deg)
        for angle_deg in angles
    ]
    outline = [left[0], right[0]]
    outline.extend(right[1:])
    outline.extend(lower_ring_arc[1:])
    outline.extend(reversed(left[:-1]))
    plan = Polygon(outline)
    if not plan.is_valid or plan.is_empty or len(plan.interiors):
        raise RuntimeError(
            "soft bridge blend must be one valid opening-free polygon")
    return plan


def bridge_fusion_cradle_plan():
    """Same-depth annular-sector spreader inside the LM structural lip."""
    half = BRIDGE_FUSION_CRADLE_SPAN_DEG / 2.0
    start = BRIDGE_FUSION_CRADLE_CENTER_DEG - half
    stop = BRIDGE_FUSION_CRADLE_CENTER_DEG + half
    angles = [start + (stop - start) * i / 128.0 for i in range(129)]
    cx, cy = L22_CUTOUT[:2]
    outer = [(
        cx + BRIDGE_FUSION_CRADLE_OUTER_R * math.cos(math.radians(a)),
        cy + BRIDGE_FUSION_CRADLE_OUTER_R * math.sin(math.radians(a)),
    ) for a in angles]
    inner = [(
        cx + BRIDGE_FUSION_CRADLE_INNER_R * math.cos(math.radians(a)),
        cy + BRIDGE_FUSION_CRADLE_INNER_R * math.sin(math.radians(a)),
    ) for a in reversed(angles)]
    return Polygon((*outer, *inner))


def native_bridge_face_plan():
    """Historic no-floor bridge outline before universal-profile growth."""
    plan = unary_union((
        bridge_solid_web_plan(),
        bridge_soft_blend_plan(),
        bridge_fusion_cradle_plan(),
    ))
    if plan.geom_type != "Polygon":
        raise RuntimeError(
            "front-flush bridge plan must be one opening-free polygon; "
            f"type={plan.geom_type}")
    # The soft blend and cradle sample the same circular boundary at
    # different phase increments. Their union can therefore leave only
    # chordal numerical pinholes (tens of microns squared) along what is
    # physically one solid face. The bridge contract is explicitly
    # opening-free: fill those sampling artifacts, but fail if any real
    # opening ever appears.
    if plan.interiors:
        hole_area = sum(Polygon(ring).area for ring in plan.interiors)
        if hole_area > 0.10:
            raise RuntimeError(
                "front-flush bridge plan developed a real opening: "
                f"interior_area={hole_area:.6f} mm2")
        plan = Polygon(plan.exterior)
    if not plan.is_valid or plan.is_empty or plan.interiors:
        raise RuntimeError(
            "front-flush bridge plan pinhole repair failed")
    return plan


def common_lm_wing_contact_plan():
    """State-independent LM-lower outline used by both carriers and wings.

    The union removes no existing material: it preserves the broad no-floor
    cubic shoulder and the floor stem's complete Y=0 lower tongue.  It is a
    single opening-free exterior plan; state-specific holes, lumens and deep
    floor geometry are applied after this common interface is established.
    """
    floor_stem = Polygon(integral_stem_plan_points()).buffer(0)
    plan = unary_union((native_bridge_face_plan(), floor_stem)).buffer(0)
    if plan.geom_type != "Polygon" or not plan.is_valid or plan.is_empty:
        raise RuntimeError(
            "universal LM wing-contact union must be one valid polygon")
    # The two historic shoulders cross twice per side, enclosing one small
    # transition pocket on each side.  A universal *exterior* is represented
    # by the filled outer boundary; the normal R110.6 driver-seat cutter later
    # removes both pockets because they lie wholly inside that flange recess.
    # Guard the expected two-pocket topology so this exterior operation cannot
    # silently hide a future opening outside the driver keepout.
    if plan.interiors:
        pocket_areas = tuple(Polygon(ring).area for ring in plan.interiors)
        if (len(pocket_areas) != 2
                or not all(18.0 <= area <= 19.0 for area in pocket_areas)):
            raise RuntimeError(
                "universal LM wing-contact union developed unexpected "
                f"pockets: {pocket_areas}")
        plan = Polygon(plan.exterior)
    if not plan.is_valid or plan.is_empty or plan.interiors:
        raise RuntimeError(
            "universal LM wing-contact exterior must be opening-free")
    return plan


def bridge_face_plan():
    """Complete universal no-floor front web and wing-contact silhouette."""
    return common_lm_wing_contact_plan()


def floor_wing_contact_profile_addition_plan():
    """Plan delta that grows the floor stem to the universal front profile.

    The inward owner offset provides a positive fusion strip.  The addition
    remains inside the universal exterior and is one connected perimeter
    polygon.  Its sole interior is intentional: that region is already solid
    floor-stem material, not a cavity in the finished carrier.
    """
    floor_stem = Polygon(integral_stem_plan_points()).buffer(0)
    retained_owner = floor_stem.buffer(
        -LM_WING_CONTACT_FUSION_OVERLAP_MM, join_style=1)
    plan = common_lm_wing_contact_plan().difference(
        retained_owner).buffer(0)
    if plan.geom_type != "Polygon" or not plan.is_valid or plan.is_empty:
        raise RuntimeError(
            "floor LM wing-contact delta must be one valid polygon")
    if len(plan.interiors) != 1:
        raise RuntimeError(
            "floor LM wing-contact delta must retain exactly one existing-"
            f"stem owner opening; count={len(plan.interiors)}")
    owner_opening = Polygon(plan.interiors[0])
    if owner_opening.symmetric_difference(retained_owner).area > 1e-6:
        raise RuntimeError(
            "floor LM wing-contact delta owner opening drifted from the "
            "positive-overlap stem datum")
    return plan


def floor_wing_contact_profile_addition():
    """Solid floor-profile delta through the wing's z=6.8..18.3 depth.

    The integral W64 stem/foot remains the sole deep floor load path.
    """
    plan = floor_wing_contact_profile_addition_plan()
    addition = _plan_prism(plan, *LM_WING_CONTACT_Z).clean()
    solids = list(addition.solids())
    if (not addition.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "floor LM wing-contact delta must be one valid solid; "
            f"valid={addition.is_valid} solids={len(solids)}")
    return Part([solids[0]])


def _no_floor_feed_lumen_plan():
    """Conservative plan projection of both no-floor bridge feed lumens."""
    sqrt_half = math.sqrt(0.5)

    def entry(feed_xy, start_bearing_deg, arc_angle_deg,
              end_direction, radius):
        feed = tuple(map(float, feed_xy))
        start_direction = (
            math.cos(math.radians(start_bearing_deg)),
            math.sin(math.radians(start_bearing_deg)),
        )
        arc = _circle_xy(LM_ROUTE_R, arc_angle_deg)
        points = [(
            feed[0] - start_direction[0] * NO_FLOOR_FEED_CUTTER_EXTENSION,
            feed[1] - start_direction[1] * NO_FLOOR_FEED_CUTTER_EXTENSION,
        )]
        points.extend(_cubic_xy(
            feed,
            (feed[0] + start_direction[0] * NO_FLOOR_FEED_START_HANDLE,
             feed[1] + start_direction[1] * NO_FLOOR_FEED_START_HANDLE),
            (
                arc[0] - end_direction[0] * NO_FLOOR_FEED_END_HANDLE,
                arc[1] - end_direction[1] * NO_FLOOR_FEED_END_HANDLE,
            ),
            arc,
            800,
        ))
        # A round buffer slightly overbounds the real 24/32-sided cutter
        # sections, which is appropriate for a load-bearing deduction.
        return LineString(points).buffer(
            radius, resolution=32, cap_style=1, join_style=1)

    main = entry(
        NO_FLOOR_MAIN_FEED_XY,
        NO_FLOOR_FEED_START_BEARING_DEG,
        LM_ROUTE_ARC_START_DEG,
        (sqrt_half, sqrt_half),
        DUCT_D / 2.0,
    )
    tweeter = entry(
        NO_FLOOR_T_FEED_XY,
        180.0 - NO_FLOOR_FEED_START_BEARING_DEG,
        TS_LM_ARC_START_DEG,
        (-sqrt_half, sqrt_half),
        TS_DUCT_D / 2.0,
    )
    return unary_union((main, tweeter))


@lru_cache(maxsize=1)
def bridge_route_section_facts():
    """Exact sampled horizontal cuts through the route-bearing bridge web."""
    face = bridge_face_plan()
    lumen = _no_floor_feed_lumen_plan()
    net = face.difference(lumen)
    y0, y1 = BRIDGE_ROUTE_SECTION_Y_RANGE
    count = int(math.ceil((y1 - y0) / BRIDGE_ROUTE_SECTION_SAMPLE_MM))
    samples = [
        y0 + (y1 - y0) * index / count
        for index in range(count + 1)
    ]

    def cut(y):
        line = LineString(((-200.0, y), (200.0, y)))
        gross = float(line.intersection(face).length)
        net_width = float(line.intersection(net).length)
        return gross, gross - net_width, net_width

    cuts = [(y, *cut(y)) for y in samples]
    minimum = min(cuts, key=lambda item: item[3])
    root = cut(y1)
    if minimum[3] + 1e-6 < BRIDGE_GOVERNING_NECK_WIDTH_MM:
        raise RuntimeError(
            "route-cut bridge section fell below structural lower bound: "
            f"sample={minimum[3]:.3f} mm "
            f"bound={BRIDGE_GOVERNING_NECK_WIDTH_MM:.3f} mm")
    return {
        "sample_step_max_mm": BRIDGE_ROUTE_SECTION_SAMPLE_MM,
        "y_range_mm": BRIDGE_ROUTE_SECTION_Y_RANGE,
        "minimum_y_mm": minimum[0],
        "minimum_gross_width_mm": minimum[1],
        "minimum_lumen_deduction_mm": minimum[2],
        "minimum_net_width_mm": minimum[3],
        "design_net_width_mm": BRIDGE_GOVERNING_NECK_WIDTH_MM,
        "root_y_mm": y1,
        "root_gross_width_mm": root[0],
        "root_lumen_deduction_mm": root[1],
        "root_net_width_mm": root[2],
    }


def fused_bridge_tail():
    """Return the one-piece shallow web before fusion into the LM ring."""
    tail = _plan_prism(bridge_face_plan(), *BRIDGE_FACE_Z)
    for x, y in BRIDGE_HOLE_XY:
        tail -= Pos(
            x, y, BRIDGE_WEB_REAR_Z + BRIDGE_INSERT_DEPTH_MM / 2.0
        ) * Cylinder(BRIDGE_INSERT_D_MM / 2.0, BRIDGE_INSERT_DEPTH_MM)
    tail = tail.clean()
    solids = list(tail.solids())
    if (not tail.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "front-flush bridge web must be one valid solid; "
            f"valid={tail.is_valid} solids={len(solids)}")
    return Part([solids[0]])


def bridge_insert_envelopes():
    """Four immutable blind insert bores opening at the LM rear datum."""
    return Compound(children=[
        Pos(x, y, BRIDGE_WEB_REAR_Z + BRIDGE_INSERT_DEPTH_MM / 2.0)
        * Cylinder(BRIDGE_INSERT_D_MM / 2.0, BRIDGE_INSERT_DEPTH_MM)
        for x, y in BRIDGE_HOLE_XY
    ])


def bridge_fastener_head_envelopes():
    """Conservative D10 x 3 screw-head approach immediately behind web."""
    return Compound(children=[
        Pos(x, y, BRIDGE_WEB_REAR_Z - 1.5) * Cylinder(5.0, 3.0)
        for x, y in BRIDGE_HOLE_XY
    ])


def _rectangular_section(width: float, thickness: float):
    """Biaxial section properties of one net solid-web strip."""
    return {
        "area_mm2": width * thickness,
        "in_plane_section_modulus_mm3": thickness * width ** 2 / 6.0,
        "rear_section_modulus_mm3": width * thickness ** 2 / 6.0,
    }


def bridge_fusion_interface_facts():
    mean_r = (
        BRIDGE_FUSION_CRADLE_INNER_R + BRIDGE_FUSION_CRADLE_OUTER_R
    ) / 2.0
    gross_width = mean_r * math.radians(BRIDGE_FUSION_CRADLE_SPAN_DEG)
    effective_width = gross_width - BRIDGE_FUSION_TUNNEL_DEDUCTION_MM
    section = _rectangular_section(
        effective_width, BRIDGE_FUSION_INTERFACE_T)
    return {
        "span_deg": BRIDGE_FUSION_CRADLE_SPAN_DEG,
        "gross_arc_width_mm": gross_width,
        "deducted_um_tunnel_count": 1,
        "deducted_um_tunnel_width_mm": DUCT_D,
        "deducted_t_tunnel_width_mm": TS_DUCT_D,
        "deducted_tunnel_width_mm": BRIDGE_FUSION_TUNNEL_DEDUCTION_MM,
        "effective_width_mm": effective_width,
        "interface_z": BRIDGE_FUSION_INTERFACE_Z,
        "interface_height_mm": BRIDGE_FUSION_INTERFACE_T,
        **section,
    }


def bridge_load_facts():
    """Four-hole reactions plus a biaxial shallow-solid-web PLA screen."""
    gravity = 9.80665
    ys = [float(y) for _x, y in BRIDGE_HOLE_XY]
    ybar = sum(ys) / len(ys)
    sum_sq = sum((y - ybar) ** 2 for y in ys)

    def force(g_load: float):
        return BRIDGE_DESIGN_MASS_KG * gravity * g_load

    def normal_insert(g_load: float):
        f = force(g_load)
        moment = f * (BRIDGE_DESIGN_Y_CG - ybar)
        return max(f / len(ys) + abs(moment * (y - ybar) / sum_sq)
                   for y in ys)

    def rear_insert(g_load: float):
        moment = force(g_load) * BRIDGE_REAR_CG_MM
        return max(abs(moment * (y - ybar) / sum_sq) for y in ys)

    def combined_insert(g_load: float):
        f = force(g_load)
        normal_moment = f * (BRIDGE_DESIGN_Y_CG - ybar)
        rear_moment = f * BRIDGE_REAR_CG_MM
        return max(math.hypot(
            f / len(ys) + abs(normal_moment * (y - ybar) / sum_sq),
            abs(rear_moment * (y - ybar) / sum_sq),
        ) for y in ys)

    root_y = BRIDGE_WEB_Y[1]
    governing_section_y = BRIDGE_ROUTE_SECTION_Y_MM
    member_normal_lever = BRIDGE_DESIGN_Y_CG - governing_section_y
    root_normal_lever = BRIDGE_DESIGN_Y_CG - root_y
    web_section = _rectangular_section(
        BRIDGE_GOVERNING_NECK_WIDTH_MM, BRIDGE_WEB_T)
    fusion = bridge_fusion_interface_facts()

    def stress(g_load: float, section, lever):
        f = force(g_load)
        in_plane = (
            f * lever / section["in_plane_section_modulus_mm3"])
        rear = f * BRIDGE_REAR_CG_MM / section["rear_section_modulus_mm3"]
        return in_plane, rear, in_plane + rear

    member = {
        g: stress(g, web_section, member_normal_lever)
        for g in (1.0, 3.0, 5.0)
    }
    fusion_stress = {
        g: stress(g, fusion, root_normal_lever)
        for g in (1.0, 3.0, 5.0)
    }
    fusion_shear = {
        g: force(g) / fusion["area_mm2"] for g in (1.0, 3.0, 5.0)
    }
    return {
        "design_mass_kg": BRIDGE_DESIGN_MASS_KG,
        "design_y_cg_mm": BRIDGE_DESIGN_Y_CG,
        "rear_cg_mm": BRIDGE_REAR_CG_MM,
        "root_y_mm": root_y,
        "governing_section_y_mm": governing_section_y,
        "normal_root_lever_mm": root_normal_lever,
        "member_normal_lever_mm": member_normal_lever,
        "fusion_normal_root_lever_mm": root_normal_lever,
        "creep_allow_mpa": BRIDGE_CREEP_ALLOW_MPA,
        "short_allow_mpa": BRIDGE_SHORT_ALLOW_MPA,
        "insert_pullout_n": BRIDGE_INSERT_PULLOUT_N,
        "web_rear_z_mm": BRIDGE_WEB_REAR_Z,
        "web_front_z_mm": BRIDGE_WEB_FRONT_Z,
        "web_depth_mm": BRIDGE_WEB_T,
        "rear_depth_protrusion_mm": 0.0,
        "gross_web_width_mm": BRIDGE_WEB_WIDTH,
        "deducted_central_tunnel_width_mm": BRIDGE_WEB_TUNNEL_DEDUCTION_MM,
        "net_web_width_mm": BRIDGE_WEB_NET_WIDTH_MM,
        "governing_neck_width_mm": BRIDGE_GOVERNING_NECK_WIDTH_MM,
        "route_section": bridge_route_section_facts(),
        **web_section,
        "group_ybar_mm": ybar,
        "group_sum_sq_mm2": sum_sq,
        "normal_insert_1g_n": normal_insert(1.0),
        "normal_insert_3g_n": normal_insert(3.0),
        "normal_insert_5g_n": normal_insert(5.0),
        "rear_moment_insert_1g_n": rear_insert(1.0),
        "rear_moment_insert_3g_n": rear_insert(3.0),
        "rear_moment_insert_5g_n": rear_insert(5.0),
        "combined_insert_1g_n": combined_insert(1.0),
        "combined_insert_3g_n": combined_insert(3.0),
        "combined_insert_5g_n": combined_insert(5.0),
        "insert_sf_3g": BRIDGE_INSERT_PULLOUT_N / combined_insert(3.0),
        "insert_sf_5g": BRIDGE_INSERT_PULLOUT_N / combined_insert(5.0),
        "member_in_plane_stress_1g_mpa": member[1.0][0],
        "member_in_plane_stress_3g_mpa": member[3.0][0],
        "member_in_plane_stress_5g_mpa": member[5.0][0],
        "member_rear_stress_1g_mpa": member[1.0][1],
        "member_rear_stress_3g_mpa": member[3.0][1],
        "member_rear_stress_5g_mpa": member[5.0][1],
        "member_stress_1g_mpa": member[1.0][2],
        "member_stress_3g_mpa": member[3.0][2],
        "member_stress_5g_mpa": member[5.0][2],
        "member_sf_1g_creep": BRIDGE_CREEP_ALLOW_MPA / member[1.0][2],
        "member_sf_3g": BRIDGE_SHORT_ALLOW_MPA / member[3.0][2],
        "member_sf_5g": BRIDGE_SHORT_ALLOW_MPA / member[5.0][2],
        "fusion_interface": fusion,
        "fusion_stress_1g_mpa": fusion_stress[1.0][2],
        "fusion_stress_3g_mpa": fusion_stress[3.0][2],
        "fusion_stress_5g_mpa": fusion_stress[5.0][2],
        "fusion_sf_1g_creep": (
            BRIDGE_CREEP_ALLOW_MPA / fusion_stress[1.0][2]),
        "fusion_sf_3g": BRIDGE_SHORT_ALLOW_MPA / fusion_stress[3.0][2],
        "fusion_sf_5g": BRIDGE_SHORT_ALLOW_MPA / fusion_stress[5.0][2],
        "fusion_shear_stress_1g_mpa": fusion_shear[1.0],
        "fusion_shear_stress_3g_mpa": fusion_shear[3.0],
        "fusion_shear_stress_5g_mpa": fusion_shear[5.0],
        "fusion_shear_sf_1g": BRIDGE_CREEP_ALLOW_MPA / fusion_shear[1.0],
        "fusion_shear_sf_3g": BRIDGE_SHORT_ALLOW_MPA / fusion_shear[3.0],
        "fusion_shear_sf_5g": BRIDGE_SHORT_ALLOW_MPA / fusion_shear[5.0],
        "magnet_load_credit_n": 0.0,
    }


def bridge_plan_facts():
    center = L22_CUTOUT[:2]
    vectors = [(x - center[0], y - center[1]) for x, y in BRIDGE_HOLE_XY]
    face_plan = bridge_face_plan()
    native_plan = native_bridge_face_plan()
    floor_stem_plan = Polygon(integral_stem_plan_points()).buffer(0)
    raw_union = unary_union((native_plan, floor_stem_plan)).buffer(0)
    transition_pocket_areas = tuple(
        float(Polygon(ring).area) for ring in raw_union.interiors)
    web_plan = bridge_solid_web_plan()
    cradle_plan = bridge_fusion_cradle_plan()
    blend_plan = bridge_soft_blend_plan()
    return {
        "holes": tuple(tuple(map(float, p)) for p in BRIDGE_HOLE_XY),
        "vectors_from_lm": tuple(vectors),
        "radii_from_lm": tuple(math.hypot(x, y) for x, y in vectors),
        "pattern_width_mm": 40.0,
        "pattern_height_mm": 50.0,
        "pattern_center_offset_mm": center[1] - 45.0,
        "face_plan_area_mm2": float(face_plan.area),
        "face_exterior_area_mm2": float(Polygon(face_plan.exterior).area),
        "face_opening_count": len(face_plan.interiors),
        "face_outline": tuple(
            (float(x), float(y)) for x, y in face_plan.exterior.coords),
        "universal_wing_contact_profile": True,
        "universal_wing_contact_bounds": tuple(map(float, face_plan.bounds)),
        "universal_wing_contact_area_mm2": float(face_plan.area),
        "native_bridge_bounds": tuple(map(float, native_plan.bounds)),
        "native_floor_stem_bounds": tuple(map(
            float, floor_stem_plan.bounds)),
        "floor_profile_added_area_mm2": float(
            face_plan.difference(floor_stem_plan).area),
        "no_floor_profile_added_area_mm2": float(
            face_plan.difference(native_plan).area),
        "transition_pocket_fill_count": len(transition_pocket_areas),
        "transition_pocket_fill_area_mm2": sum(
            transition_pocket_areas),
        "wing_contact_z": LM_WING_CONTACT_Z,
        "wing_contact_fusion_overlap_mm": (
            LM_WING_CONTACT_FUSION_OVERLAP_MM),
        "solid_web_plan_area_mm2": float(web_plan.area),
        "solid_web_bounds": tuple(map(float, web_plan.bounds)),
        "solid_web_width_mm": BRIDGE_WEB_WIDTH,
        "solid_web_height_mm": BRIDGE_WEB_HEIGHT,
        "solid_web_corner_radius_mm": BRIDGE_WEB_CORNER_R,
        "soft_blend_plan_area_mm2": float(blend_plan.area),
        "soft_blend_start_y_mm": BRIDGE_BLEND_START_Y,
        "soft_blend_ring_offset_deg": BRIDGE_BLEND_RING_OFFSET_DEG,
        "soft_blend_right_tangency": tuple(map(
            float, BRIDGE_BLEND_RIGHT_TANGENCY)),
        "soft_blend_left_tangency": tuple(map(
            float, BRIDGE_BLEND_LEFT_TANGENCY)),
        "soft_blend_start_handle_mm": BRIDGE_BLEND_START_HANDLE_MM,
        "soft_blend_end_handle_mm": BRIDGE_BLEND_END_HANDLE_MM,
        "governing_neck_width_mm": BRIDGE_GOVERNING_NECK_WIDTH_MM,
        "route_section": bridge_route_section_facts(),
        "web_z": BRIDGE_FACE_Z,
        "rear_insert_entry_z_mm": BRIDGE_WEB_REAR_Z,
        "insert_front_floor_mm": BRIDGE_BORE_FLOOR_MM,
        "rear_rib_depth_mm": 0.0,
        "fusion_cradle_plan_area_mm2": float(cradle_plan.area),
        "fusion_cradle_span_deg": BRIDGE_FUSION_CRADLE_SPAN_DEG,
        "fusion_cradle_z": BRIDGE_FUSION_CRADLE_Z,
        "fusion_interface_z": BRIDGE_FUSION_INTERFACE_Z,
        "fusion_cradle_outline": tuple(
            (float(x), float(y)) for x, y in cradle_plan.exterior.coords),
    }
