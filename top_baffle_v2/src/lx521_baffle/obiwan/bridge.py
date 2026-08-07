"""Universal Obi-Wan LM-lower front profile and no-floor bridge load screen.

The four stock bridge XY datums are immutable.  In no-floor mode they are
carried by one solid, shallow web fused directly into the LM carrier.  The
web occupies exactly the deepest existing LM carrier envelope (the z=5.3
insert-pad rear datum through the z=18.3 front): it is flush with the front,
has no rear X-frame or separate depth ribs, and exposes four blind insert
bores from the rear.

No-floor owns the complete shallow bridge outline.  Floor mode retains only
the same upper shoulder where it meets the LM ring.  The lower magnet pair is
located on that shared shoulder, so floor mode needs neither a lower rail nor
a thin fusion skirt around the absent pre-Option-B stem.
"""

from __future__ import annotations

from functools import lru_cache
import math

from build123d import Compound, Cylinder, Face, Part, Polyline, Pos, Wire, extrude
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ..base import (
    BRIDGE_HOLE_XY,
    BRIDGE_INSERT_DEPTH_MM,
    L22_CUTOUT,
    M5_INSERT_ENTRY_D_MM,
    THICKNESS_MM,
    m5_insert_bore_cutter,
)
from ..flush import LM_RECESS_R, PAD_FACE_Z
from .route import (
    DUCT_D,
    LM_INTERNAL_DUCT_D_MM,
    TS_DUCT_D,
    lm_internal_duct_cutter_points,
    route_cable_points,
    ts_cable_points,
)
from .floor import integral_stem_plan_points

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
# its vertical sides tangentially into the LM outer ring. The three D20-packed
# rear feed lumens pass through the upper panel, so all three diameters are
# conservatively removed from the complete 62-mm core even though they do not
# govern at one identical horizontal station and the cubic shoulders make
# every real cut wider. No strength credit is taken for tunnel skins.
BRIDGE_WEB_X = (-31.0, 31.0)
BRIDGE_WEB_Y = (14.0, 90.25)
BRIDGE_WEB_CORNER_R = 4.0
BRIDGE_WEB_WIDTH = BRIDGE_WEB_X[1] - BRIDGE_WEB_X[0]
BRIDGE_WEB_HEIGHT = BRIDGE_WEB_Y[1] - BRIDGE_WEB_Y[0]
BRIDGE_WEB_TUNNEL_DEDUCTION_MM = (
    LM_INTERNAL_DUCT_D_MM + DUCT_D + TS_DUCT_D)
BRIDGE_WEB_NET_WIDTH_MM = (
    BRIDGE_WEB_WIDTH - BRIDGE_WEB_TUNNEL_DEDUCTION_MM)

# Screen the first uninterrupted horizontal ligament above the immutable top
# insert pair. The 0.05-mm offset avoids crediting a tangent to the D6.5 entry
# reliefs. All three rear-facing lumens are conservatively deducted across the
# complete core even though their actual curved centerlines do not coincide at
# one section. A separate sampled outline/corridor audit below proves the real
# soft-blend section is at least this 38.8-mm lower bound.
BRIDGE_ROUTE_SECTION_Y_MM = (
    max(float(y) for _x, y in BRIDGE_HOLE_XY)
    + M5_INSERT_ENTRY_D_MM / 2.0 + 0.05)
BRIDGE_ROUTE_SECTION_Y_RANGE = (
    BRIDGE_ROUTE_SECTION_Y_MM, BRIDGE_WEB_Y[1])
BRIDGE_ROUTE_SECTION_SAMPLE_MM = 0.01

# The member safety-factor screen must account for the three lumens at their
# real X/Z locations.  Treating every diameter as a full-depth vertical strip
# double-counts absent void and makes the unchanged 2.0/1.5/1.05 acceptance
# thresholds internally impossible.  The sampled model below remains
# conservative: every round/oblique lumen is replaced by an axis-aligned
# bounding rectangle, expanded beyond the sampled cutter envelope.
BRIDGE_MEMBER_SECTION_SAMPLE_MM = 0.05
BRIDGE_MEMBER_CENTERLINE_STEP_MM = 0.02
BRIDGE_MEMBER_VOID_BOUNDING_MARGIN_MM = 0.05

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
BRIDGE_FUSION_TUNNEL_DEDUCTION_MM = BRIDGE_WEB_TUNNEL_DEDUCTION_MM

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
# member screen instead uses the opening-aware
# 62 - 9.0 - 8.2 - 6.0 = 38.8-mm core lower bound at the first ligament above
# the top insert pair. This is smaller than the exact sampled soft-outline net
# section and therefore conservative.
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


def bridge_soft_blend_right_controls():
    """Exact right-side cubic controls for the lower LM shoulder."""
    start = (BRIDGE_WEB_X[1], BRIDGE_BLEND_START_Y)
    end = BRIDGE_BLEND_RIGHT_TANGENCY
    angle = math.radians(BRIDGE_BLEND_RIGHT_ANGLE_DEG)
    tangent = (-math.sin(angle), math.cos(angle))
    return (
        start,
        (start[0], start[1] + BRIDGE_BLEND_START_HANDLE_MM),
        (end[0] - BRIDGE_BLEND_END_HANDLE_MM * tangent[0],
         end[1] - BRIDGE_BLEND_END_HANDLE_MM * tangent[1]),
        end,
    )


def bridge_soft_blend_frame(
        parameter: float, side: str = "right",
        ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the shoulder point and outward unit normal at a cubic station."""
    u = float(parameter)
    if not 0.0 <= u <= 1.0:
        raise ValueError("soft-blend parameter must be in [0, 1]")
    if side not in {"left", "right"}:
        raise ValueError("soft-blend side must be 'left' or 'right'")
    p0, p1, p2, p3 = bridge_soft_blend_right_controls()
    v = 1.0 - u
    point = (
        v ** 3 * p0[0] + 3.0 * v ** 2 * u * p1[0]
        + 3.0 * v * u ** 2 * p2[0] + u ** 3 * p3[0],
        v ** 3 * p0[1] + 3.0 * v ** 2 * u * p1[1]
        + 3.0 * v * u ** 2 * p2[1] + u ** 3 * p3[1],
    )
    derivative = (
        3.0 * (
            v ** 2 * (p1[0] - p0[0])
            + 2.0 * v * u * (p2[0] - p1[0])
            + u ** 2 * (p3[0] - p2[0])),
        3.0 * (
            v ** 2 * (p1[1] - p0[1])
            + 2.0 * v * u * (p2[1] - p1[1])
            + u ** 2 * (p3[1] - p2[1])),
    )
    length = math.hypot(*derivative)
    if length <= 1.0e-12:
        raise RuntimeError("soft-blend frame encountered a zero tangent")
    # The right curve runs upward with carrier material to its left, making
    # the clockwise normal the physical outward interface direction.
    outward = (derivative[1] / length, -derivative[0] / length)
    if side == "left":
        return (-point[0], point[1]), (-outward[0], outward[1])
    return point, outward


def bridge_soft_blend_plan():
    """Filled symmetric cubic transition from the panel into the LM ring."""
    right = _cubic_xy(
        *bridge_soft_blend_right_controls(), BRIDGE_BLEND_SAMPLES)
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
    """Return only the upper floor-state wing-contact shoulder.

    The Option-B wall removed the old front-depth stem below its upright
    tangent.  Subtracting an inward offset of that *historic* stem over the
    whole plan therefore exported the nominal 0.10-mm fusion allowance as a
    visible rectangular perimeter box.  The lower magnet pair now lives on
    the shared curved shoulder, so the addition starts at ``y=60`` and keeps
    no material at all below that tangent.
    """
    floor_stem = Polygon(integral_stem_plan_points()).buffer(0)
    retained_owner = floor_stem.buffer(
        -LM_WING_CONTACT_FUSION_OVERLAP_MM, join_style=1)
    upper_clip = box(
        -200.0,
        BRIDGE_BLEND_START_Y,
        200.0,
        200.0,
    )
    plan = common_lm_wing_contact_plan().difference(
        retained_owner).intersection(upper_clip).buffer(0)
    if plan.geom_type != "Polygon" or not plan.is_valid or plan.is_empty:
        raise RuntimeError(
            "floor LM wing-contact delta must be one valid polygon")
    if plan.interiors:
        raise RuntimeError(
            "floor LM wing-contact delta must remain open below the curved "
            f"stand; openings={len(plan.interiors)}")
    forbidden_lower = box(-200.0, 0.0, 200.0, BRIDGE_BLEND_START_Y)
    if plan.intersection(forbidden_lower).area > 1.0e-8:
        raise RuntimeError(
            "floor LM wing-contact delta added material below the shared "
            "upper shoulder")
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


@lru_cache(maxsize=1)
def _no_floor_feed_lumen_records():
    """Sample the route-owned bridge centerlines once for all load screens."""
    records = (
        ("lm", lm_internal_duct_cutter_points(
            BRIDGE_MEMBER_CENTERLINE_STEP_MM),
         LM_INTERNAL_DUCT_D_MM / 2.0),
        ("um", route_cable_points(BRIDGE_MEMBER_CENTERLINE_STEP_MM),
         DUCT_D / 2.0),
        ("t", ts_cable_points(BRIDGE_MEMBER_CENTERLINE_STEP_MM),
         TS_DUCT_D / 2.0),
    )
    normalized = []
    for name, points, radius in records:
        coordinates = tuple(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in points)
        if len(coordinates) < 2:
            raise RuntimeError(
                f"no-floor bridge {name} lumen centerline is empty")
        normalized.append((name, coordinates, float(radius)))
    return tuple(normalized)


def _no_floor_feed_lumen_plan():
    """Conservative plan projection of all three no-floor bridge lumens."""
    lumens = []
    for _name, points, radius in _no_floor_feed_lumen_records():
        lumens.append(LineString(
            [(x, y) for x, y, _z in points]).buffer(
                radius, resolution=32, cap_style=1, join_style=1))
    return unary_union(lumens)


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


def _rectangular_area_moments(x0, x1, z0, z1):
    """Area, first moments and origin second moments of one X/Z box."""
    width = x1 - x0
    height = z1 - z0
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("bridge member section contains an empty rectangle")
    area = width * height
    center_x = (x0 + x1) / 2.0
    center_z = (z0 + z1) / 2.0
    return {
        "area_mm2": area,
        "first_x_mm3": area * center_x,
        "first_z_mm3": area * center_z,
        "second_about_z_origin_mm4": (
            height * (x1 ** 3 - x0 ** 3) / 3.0),
        "second_about_x_origin_mm4": (
            width * (z1 ** 3 - z0 ** 3) / 3.0),
    }


def _bridge_member_section_at(y_mm, lumen_records):
    """Conservative X/Z section using route-envelope bounding rectangles."""
    gross = _rectangular_area_moments(
        BRIDGE_WEB_X[0], BRIDGE_WEB_X[1],
        BRIDGE_WEB_REAR_Z, BRIDGE_WEB_FRONT_Z)
    totals = dict(gross)
    rectangles = []
    cut = LineString(((BRIDGE_WEB_X[0], y_mm),
                      (BRIDGE_WEB_X[1], y_mm)))
    margin = BRIDGE_MEMBER_VOID_BOUNDING_MARGIN_MM

    for name, points, radius in lumen_records:
        plan = LineString([(x, y) for x, y, _z in points]).buffer(
            radius + margin, resolution=32, cap_style=1, join_style=1)
        intersection = cut.intersection(plan)
        if intersection.is_empty:
            continue
        segments = (
            (intersection,) if intersection.geom_type == "LineString"
            else tuple(
                item for item in getattr(intersection, "geoms", ())
                if item.geom_type == "LineString" and item.length > 1e-9)
        )
        near_z = [
            z for _x, center_y, z in points
            if abs(center_y - y_mm) <= (
                radius + margin + BRIDGE_MEMBER_CENTERLINE_STEP_MM)
        ]
        if not near_z:
            raise RuntimeError(
                f"bridge {name} plan cut has no sampled Z witness")
        z0 = max(
            BRIDGE_WEB_REAR_Z,
            min(near_z) - radius - margin)
        z1 = min(
            BRIDGE_WEB_FRONT_Z,
            max(near_z) + radius + margin)
        for segment in segments:
            x0 = max(BRIDGE_WEB_X[0], float(segment.bounds[0]))
            x1 = min(BRIDGE_WEB_X[1], float(segment.bounds[2]))
            if x1 - x0 <= 1e-9:
                continue
            rectangle = box(x0, z0, x1, z1)
            for prior_name, prior in rectangles:
                overlap = rectangle.intersection(prior).area
                if overlap > 1e-7:
                    raise RuntimeError(
                        "bridge member bounding rectangles overlap; "
                        f"cannot subtract independently: {prior_name}/{name} "
                        f"area={overlap:.6f} mm2")
            rectangles.append((name, rectangle))
            removed = _rectangular_area_moments(x0, x1, z0, z1)
            for key, value in removed.items():
                totals[key] -= value

    area = totals["area_mm2"]
    if area <= 0.0:
        raise RuntimeError("bridge member bounding model removed all material")
    centroid_x = totals["first_x_mm3"] / area
    centroid_z = totals["first_z_mm3"] / area
    second_about_z = (
        totals["second_about_z_origin_mm4"] - area * centroid_x ** 2)
    second_about_x = (
        totals["second_about_x_origin_mm4"] - area * centroid_z ** 2)
    extreme_x = max(
        centroid_x - BRIDGE_WEB_X[0],
        BRIDGE_WEB_X[1] - centroid_x)
    extreme_z = max(
        centroid_z - BRIDGE_WEB_REAR_Z,
        BRIDGE_WEB_FRONT_Z - centroid_z)
    in_plane_modulus = second_about_z / extreme_x
    rear_modulus = second_about_x / extreme_z
    if in_plane_modulus <= 0.0 or rear_modulus <= 0.0:
        raise RuntimeError("bridge member bounding model has invalid modulus")

    force_1g = BRIDGE_DESIGN_MASS_KG * 9.80665
    normal_lever = BRIDGE_DESIGN_Y_CG - y_mm
    in_plane_stress = force_1g * normal_lever / in_plane_modulus
    rear_stress = force_1g * BRIDGE_REAR_CG_MM / rear_modulus
    return {
        "section_y_mm": float(y_mm),
        "normal_lever_mm": float(normal_lever),
        "lumen_bounding_rectangle_count": len(rectangles),
        "lumen_names": tuple(name for name, _rectangle in rectangles),
        "area_mm2": area,
        "centroid_x_mm": centroid_x,
        "centroid_z_mm": centroid_z,
        "in_plane_section_modulus_mm3": in_plane_modulus,
        "rear_section_modulus_mm3": rear_modulus,
        "in_plane_stress_1g_mpa": in_plane_stress,
        "rear_stress_1g_mpa": rear_stress,
        "combined_stress_1g_mpa": in_plane_stress + rear_stress,
    }


@lru_cache(maxsize=1)
def bridge_member_section_facts():
    """Worst sampled conservative X/Z member section across the bridge."""
    y0, y1 = BRIDGE_ROUTE_SECTION_Y_RANGE
    count = int(math.ceil(
        (y1 - y0) / BRIDGE_MEMBER_SECTION_SAMPLE_MM))
    records = _no_floor_feed_lumen_records()
    sections = [
        _bridge_member_section_at(
            y0 + (y1 - y0) * index / count, records)
        for index in range(count + 1)
    ]
    governing = max(
        sections, key=lambda item: item["combined_stress_1g_mpa"])
    return {
        "model": "sampled_axis_aligned_lumen_bounding_rectangles",
        "sample_step_max_mm": (y1 - y0) / count,
        "centerline_step_max_mm": BRIDGE_MEMBER_CENTERLINE_STEP_MM,
        "void_bounding_margin_mm": BRIDGE_MEMBER_VOID_BOUNDING_MARGIN_MM,
        "sample_count": len(sections),
        **governing,
    }


def fused_bridge_tail():
    """Return the one-piece shallow web before fusion into the LM ring."""
    tail = _plan_prism(bridge_face_plan(), *BRIDGE_FACE_Z)
    for x, y in BRIDGE_HOLE_XY:
        tail -= m5_insert_bore_cutter(
            (x, y),
            opening_z=BRIDGE_WEB_REAR_Z,
            total_depth=BRIDGE_INSERT_DEPTH_MM,
            opening_side="-z",
        )
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
        m5_insert_bore_cutter(
            (x, y),
            opening_z=BRIDGE_WEB_REAR_Z,
            total_depth=BRIDGE_INSERT_DEPTH_MM,
            opening_side="-z",
            overshoot=0.0,
        )
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
        "deducted_lm_tunnel_count": 1,
        "deducted_lm_tunnel_width_mm": LM_INTERNAL_DUCT_D_MM,
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
    member_section = bridge_member_section_facts()
    governing_section_y = member_section["section_y_mm"]
    member_normal_lever = member_section["normal_lever_mm"]
    root_normal_lever = BRIDGE_DESIGN_Y_CG - root_y
    web_section = {
        key: member_section[key] for key in (
            "area_mm2", "in_plane_section_modulus_mm3",
            "rear_section_modulus_mm3")
    }
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
        "member_section": member_section,
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
        "universal_wing_contact_profile": False,
        "no_floor_wing_contact_profile": "common_bridge_plus_floor_stem",
        "floor_wing_contact_profile": "upper_common_shoulder_only",
        "floor_exposed_perimeter_box": False,
        "floor_lower_magnet_rails": False,
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
        "floor_profile_min_y_mm": float(
            floor_wing_contact_profile_addition_plan().bounds[1]),
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
