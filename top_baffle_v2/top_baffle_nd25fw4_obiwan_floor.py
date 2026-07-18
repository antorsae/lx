"""Integral floor-state stem, foot, connector and buried continuations.

The floor-state LM carrier owns this geometry.  There is no mating support,
yoke, rail set or support fastener.  The outer floor envelope is a solid
W64 rectangle from the baffle front to the retained NL8 rear plane; only the
three buried cable lumens and the necessary connector service scoop are
subtracted.

This module intentionally does not import ``top_baffle_nd25fw4_obiwan`` so the
carrier can add the floor body without an import cycle.
"""

from __future__ import annotations

import math

from build123d import (
    Bezier,
    Box,
    Circle,
    Cylinder,
    Face,
    Line,
    Part,
    Plane,
    Polyline,
    Pos,
    ThreePointArc,
    Wire,
    extrude,
    sweep,
)

from top_baffle_nd25fw4 import L22_CUTOUT, STAND_FOOT, THICKNESS_MM
from top_baffle_nd25fw4_flush import PAD_FACE_Z
from top_baffle_nd25fw4_obiwan_floor_strength import (
    FLOOR_Y_MM,
    FOOT_FRONT_Z_MM,
    FOOT_HEIGHT_MM,
    FOOT_REAR_Z_MM,
    FOOT_WIDTH_MM,
    LM_AXIS_Y_MM,
    ROOT_FILLET_R_MM,
)


PANEL_INNER_Z_MM = -146.0
PANEL_T_MM = PANEL_INNER_Z_MM - FOOT_REAR_Z_MM
PANEL_H_MM = 44.0
NL8_CENTER_Y_MM = 22.0
NL8_CUTOUT_D_MM = 31.0
NL8_SCREW_D_MM = 3.2
NL8_SCREW_PITCH_MM = 29.2

SERVICE_CAVITY_Z_MM = (PANEL_INNER_Z_MM - 0.20, -104.0)
SERVICE_CAVITY_X_MM = (-18.0, 18.0)
SERVICE_CAVITY_Y_MM = (4.0, 48.0)

# A full-depth W64 stem is the governing printed load member.  The soft
# shoulders enter the lower LM cap at the D190 tangent without entering the
# acoustic opening.  The rear face remains at z=0 rather than growing behind
# the approved concept envelope.
STEM_Z_MM = (0.0, THICKNESS_MM)
STEM_HALF_WIDTH_MM = FOOT_WIDTH_MM / 2.0
STEM_SHOULDER_HALF_WIDTH_MM = 58.0
STEM_SHOULDER_START_Y_MM = 68.0
STEM_TOP_Y_MM = LM_AXIS_Y_MM - L22_CUTOUT[2] / 2.0
STEM_SHOULDER_SAMPLES = 32
ROOT_FUSION_OVERLAP_MM = 0.10

FLOOR_LANE_BEND_R_MM = 14.0
FLOOR_LANE_SERVICE_START_Z_MM = -108.0
# The body-only UM/T sweep stops before the annular feed.  After fusion, the
# globally phased owner cutter reaches 8 mm backward through this temporary
# solid bridge, yielding 7.2 mm of final lumen overlap.  This avoids duplicate
# coincident sweeps at the thin feed wall.  A short forward continuation is
# retained only in the dependency-light installed-centerline drawing.
FLOOR_LANE_INSTALLED_PREVIEW_FORWARD_MM = 3.0
FLOOR_LANE_PREFUSION_HANDOFF_GAP_MM = 0.8
FLOOR_ROUTE_OWNER_BACKREACH_MM = 8.0
FLOOR_LANE_EFFECTIVE_OVERLAP_MM = (
    FLOOR_ROUTE_OWNER_BACKREACH_MM
    - FLOOR_LANE_PREFUSION_HANDOFF_GAP_MM)
FLOOR_LANE_BEZIER_START_HANDLE_MM = 20.0
FLOOR_LANE_BEZIER_END_HANDLE_MM = 25.0
# UM/T enter the integral stem through native rear mouths at z=5.3.  The
# floor body must be absent behind the complete printed cover envelope, not
# merely behind the nominal lumen.  A 0.05-mm Boolean allowance is applied
# outside the 0.30-mm final-BREP contract probe; it does not enlarge the
# functional lumen in front of the mouth plane.
FLOOR_FEED_MOUTH_SHELL_MM = 0.8
FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM = 0.30
FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM = 0.05
FLOOR_FEED_MOUTH_RELIEF_Z_MM = (-0.20, PAD_FACE_Z)
# The free LM lead does not need to approach the centered UM/T feed mouths.
# Its second R14 turn begins directly where the first R14 turn ends, so the
# D9 lane reaches its rear mouth at y=38.5 without crossing either long
# floor-body lane.  The former y=82 endpoint grazed both feed skins; merely
# moving it to y=74.5 instead crossed the complete UM/T approach cubics.
FLOOR_LM_FLOAT_FEED_Y_MM = 38.5
FLOOR_LANE_SPECS = {
    "lm": {
        "x_mm": 0.0,
        "floor_y_mm": 10.5,
        "stem_z_mm": 12.55,
        "diameter_mm": 9.0,
        # The short LM lead intentionally floats from this rear-facing
        # opening; it does not join either printed annular route.
        "feed_xyz_mm": (0.0, FLOOR_LM_FLOAT_FEED_Y_MM, -6.0),
        "handoff_mode": "rear_open_float",
    },
    "um": {
        "x_mm": 12.0,
        "floor_y_mm": 10.5,
        "stem_z_mm": 12.55,
        "diameter_mm": 8.2,
        "feed_xyz_mm": (8.0, 82.0, PAD_FACE_Z),
        "feed_bearing_deg": 65.0,
        "handoff_mode": "buried_route_overlap",
    },
    "t": {
        "x_mm": -12.0,
        # y=7.5 keeps the complete D6 opening inside the service cavity's
        # y>=4 boundary.  The old y=5.5 station clipped the connector cap.
        "floor_y_mm": 7.5,
        "stem_z_mm": 6.20,
        "diameter_mm": 6.0,
        "feed_xyz_mm": (-8.0, 82.0, PAD_FACE_Z),
        "feed_bearing_deg": 115.0,
        "handoff_mode": "buried_route_overlap",
    },
}


def _require_guarded_build() -> None:
    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        raise RuntimeError(
            "integral Obi-Wan floor geometry requires run_memory_guarded.py")


def _plan_face(points):
    return Face(Wire(Polyline(*points).edges()))


def _quadratic(p0, p1, p2, count):
    out = []
    for index in range(count + 1):
        u = index / count
        out.append((
            (1.0 - u) ** 2 * p0[0]
            + 2.0 * (1.0 - u) * u * p1[0]
            + u ** 2 * p2[0],
            (1.0 - u) ** 2 * p0[1]
            + 2.0 * (1.0 - u) * u * p1[1]
            + u ** 2 * p2[1],
        ))
    return out


def integral_stem_plan_points():
    """Closed XY outline with symmetric quadratic shoulder integration."""
    right = _quadratic(
        (STEM_HALF_WIDTH_MM, STEM_SHOULDER_START_Y_MM),
        (STEM_HALF_WIDTH_MM, STEM_TOP_Y_MM - 3.0),
        (STEM_SHOULDER_HALF_WIDTH_MM, STEM_TOP_Y_MM),
        STEM_SHOULDER_SAMPLES)
    left = [(-x, y) for x, y in reversed(right)]
    return (
        (-STEM_HALF_WIDTH_MM, FLOOR_Y_MM),
        (STEM_HALF_WIDTH_MM, FLOOR_Y_MM),
        (STEM_HALF_WIDTH_MM, STEM_SHOULDER_START_Y_MM),
        *right[1:],
        *left,
        (-STEM_HALF_WIDTH_MM, FLOOR_Y_MM),
    )


def _stem_prism():
    face = _plan_face(integral_stem_plan_points())
    return Pos(0.0, 0.0, STEM_Z_MM[0]) * extrude(
        face, amount=STEM_Z_MM[1] - STEM_Z_MM[0])


def _root_fillet_prism():
    """True R12 internal YZ fillet, extruded across the complete W64 root."""
    radius = ROOT_FILLET_R_MM
    center_y = FOOT_HEIGHT_MM + radius
    center_z = STEM_Z_MM[0] - radius
    arc = []
    for index in range(25):
        angle = -0.5 * math.pi * index / 24.0
        arc.append((
            center_y + radius * math.sin(angle),
            center_z + radius * math.cos(angle),
        ))
    # Local Plane.YZ face coordinates are (world Y, world Z).
    # Retain the external analytic R12 surface while growing the hidden
    # fusion edges 0.10 mm into both owners.  Face-only contact between the
    # former fillet, foot and stem was needlessly fragile in OCC booleans.
    overlap = ROOT_FUSION_OVERLAP_MM
    points = [
        (FOOT_HEIGHT_MM - overlap, center_z),
        (FOOT_HEIGHT_MM - overlap, STEM_Z_MM[0] + overlap),
        (center_y, STEM_Z_MM[0] + overlap),
        (center_y, STEM_Z_MM[0]),
        *arc[1:],
        (FOOT_HEIGHT_MM - overlap, center_z),
    ]
    # Plane.YZ's positive extrusion direction is world -X.  Start at the
    # +X side so the R12 prism spans exactly -32..+32 rather than producing
    # a one-sided -96..-32 wing.
    face = Pos(FOOT_WIDTH_MM / 2.0, 0.0, 0.0) * (
        Plane.YZ * _plan_face(points))
    return extrude(face, amount=FOOT_WIDTH_MM)


def integrated_floor_addition():
    """Uncut one-solid floor body to fuse into the floor LM outer blank."""
    _require_guarded_build()
    if not STAND_FOOT:
        raise RuntimeError("no-floor Obi-Wan has no integral floor addition")
    foot = Pos(
        0.0,
        FLOOR_Y_MM + FOOT_HEIGHT_MM / 2.0,
        (FOOT_REAR_Z_MM + FOOT_FRONT_Z_MM) / 2.0,
    ) * Box(
        FOOT_WIDTH_MM,
        FOOT_HEIGHT_MM,
        FOOT_FRONT_Z_MM - FOOT_REAR_Z_MM,
    )
    panel = Pos(
        0.0, PANEL_H_MM / 2.0,
        (FOOT_REAR_Z_MM + PANEL_INNER_Z_MM) / 2.0,
    ) * Box(FOOT_WIDTH_MM, PANEL_H_MM, PANEL_T_MM)
    body = foot.fuse(_stem_prism(), _root_fillet_prism(), panel).clean()
    solids = tuple(body.solids())
    if (not body.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "integral floor addition must be one valid solid; "
            f"valid={body.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def _floor_lane_entry_points(name: str):
    """Connector line and exact R14 quarter-turn defining points."""
    try:
        spec = FLOOR_LANE_SPECS[name]
    except KeyError as exc:
        raise ValueError(name) from exc
    x = spec["x_mm"]
    floor_y = spec["floor_y_mm"]
    stem_z = spec["stem_z_mm"]
    radius = FLOOR_LANE_BEND_R_MM
    center_y = floor_y + radius
    center_z = stem_z - radius
    line_start = (x, floor_y, FLOOR_LANE_SERVICE_START_Z_MM)
    arc_start = (x, floor_y, center_z)
    arc_mid = (
        x,
        center_y - radius / math.sqrt(2.0),
        center_z + radius / math.sqrt(2.0),
    )
    arc_end = (x, center_y, stem_z)
    return line_start, arc_start, arc_mid, arc_end


def _floor_lane_bezier_points(name: str):
    """G1 cubic from the R14 entry to a buried annular-route feed."""
    spec = FLOOR_LANE_SPECS[name]
    if spec["handoff_mode"] != "buried_route_overlap":
        raise ValueError(f"{name} does not have an annular-route Bezier")
    p0 = _floor_lane_entry_points(name)[-1]
    p3 = spec["feed_xyz_mm"]
    bearing = math.radians(spec["feed_bearing_deg"])
    tangent = (math.cos(bearing), math.sin(bearing), 0.0)
    p1 = (
        p0[0],
        p0[1] + FLOOR_LANE_BEZIER_START_HANDLE_MM,
        p0[2],
    )
    p2 = tuple(
        p3[index] - FLOOR_LANE_BEZIER_END_HANDLE_MM * tangent[index]
        for index in range(3))
    return p0, p1, p2, p3


def _floor_lane_overlap_end(name: str):
    spec = FLOOR_LANE_SPECS[name]
    feed = spec["feed_xyz_mm"]
    bearing = math.radians(spec["feed_bearing_deg"])
    return (
        feed[0] + FLOOR_LANE_INSTALLED_PREVIEW_FORWARD_MM * math.cos(bearing),
        feed[1] + FLOOR_LANE_INSTALLED_PREVIEW_FORWARD_MM * math.sin(bearing),
        feed[2],
    )


def floor_lane_path(name: str):
    """Floor-body cutter path with true R14 arcs and G1 joins.

    UM/T stop on their authoritative cubic 0.8 mm before the feed; the later
    8-mm annular owner-cutter backreach creates the final overlap through
    ordinary solid body material.  LM instead makes a second exact R14 turn
    through the rear face before the centered y=82 UM/T feeds, leaving the
    deliberately short terminal lead free rather than recreating a
    micro-duct.
    """
    try:
        spec = FLOOR_LANE_SPECS[name]
    except KeyError as exc:
        raise ValueError(name) from exc
    line_start, arc_start, arc_mid, arc_end = _floor_lane_entry_points(name)
    edges = [
        Line(line_start, arc_start),
        ThreePointArc(arc_start, arc_mid, arc_end),
    ]
    if spec["handoff_mode"] == "rear_open_float":
        radius = FLOOR_LANE_BEND_R_MM
        exit_start = (
            spec["x_mm"], spec["feed_xyz_mm"][1] - radius,
            spec["stem_z_mm"])
        exit_center = (
            spec["x_mm"], exit_start[1], exit_start[2] - radius)
        exit_mid = (
            spec["x_mm"],
            exit_center[1] + radius / math.sqrt(2.0),
            exit_center[2] + radius / math.sqrt(2.0),
        )
        exit_end = (
            spec["x_mm"],
            exit_start[1] + radius,
            exit_start[2] - radius,
        )
        if math.dist(arc_end, exit_start) > 1.0e-9:
            edges.append(Line(arc_end, exit_start))
        edges.extend((
            ThreePointArc(exit_start, exit_mid, exit_end),
            Line(exit_end, spec["feed_xyz_mm"]),
        ))
    else:
        edges.append(Bezier(*_prefusion_cubic_controls(name)))
    return Wire(edges)


def _cubic_point(points, u: float):
    p0, p1, p2, p3 = points
    return tuple(
        (1.0 - u) ** 3 * p0[index]
        + 3.0 * (1.0 - u) ** 2 * u * p1[index]
        + 3.0 * (1.0 - u) * u ** 2 * p2[index]
        + u ** 3 * p3[index]
        for index in range(3))


def _left_cubic_controls(points, u: float):
    """Exact De Casteljau controls for the cubic interval [0,u]."""
    p0, p1, p2, p3 = points

    def lerp(left, right):
        return tuple(
            (1.0 - u) * left[index] + u * right[index]
            for index in range(3))

    a = lerp(p0, p1)
    b = lerp(p1, p2)
    c = lerp(p2, p3)
    d = lerp(a, b)
    e = lerp(b, c)
    endpoint = lerp(d, e)
    return p0, a, d, endpoint


def _prefusion_cubic_controls(name: str):
    """Truncate an UM/T cubic at the exact pre-fusion feed setback."""
    cubic = _floor_lane_bezier_points(name)
    feed = cubic[-1]
    target = FLOOR_LANE_PREFUSION_HANDOFF_GAP_MM
    lower = 0.50
    upper = 1.0
    if math.dist(_cubic_point(cubic, lower), feed) <= target:
        raise RuntimeError(
            f"{name} floor-lane cubic is too short for its handoff gap")
    for _index in range(60):
        parameter = 0.5 * (lower + upper)
        if math.dist(_cubic_point(cubic, parameter), feed) > target:
            lower = parameter
        else:
            upper = parameter
    controls = _left_cubic_controls(cubic, 0.5 * (lower + upper))
    endpoint_gap = math.dist(controls[-1], feed)
    if not math.isclose(endpoint_gap, target, abs_tol=1.0e-9):
        raise RuntimeError(
            f"{name} floor-lane handoff gap drifted to {endpoint_gap:.9f}")
    return controls


def floor_lane_control_points(name: str):
    """Dependency-light preview of the installed continuous centerline.

    UM/T include a short forward visual continuation from the unchanged feed;
    the actual body-only cutter stops 0.8 mm early and the annular owner cutter
    supplies the final 7.2-mm backreaching overlap.
    """
    try:
        spec = FLOOR_LANE_SPECS[name]
    except KeyError as exc:
        raise ValueError(name) from exc
    line_start, arc_start, _arc_mid, arc_end = _floor_lane_entry_points(name)
    radius = FLOOR_LANE_BEND_R_MM
    center_y = spec["floor_y_mm"] + radius
    center_z = spec["stem_z_mm"] - radius
    points = [line_start, arc_start]
    for index in range(1, 17):
        angle = math.pi - 0.5 * math.pi * index / 16.0
        points.append((
            spec["x_mm"],
            center_y + radius * math.cos(angle),
            center_z + radius * math.sin(angle),
        ))
    if spec["handoff_mode"] == "rear_open_float":
        exit_start = (
            spec["x_mm"], spec["feed_xyz_mm"][1] - radius,
            spec["stem_z_mm"])
        if math.dist(points[-1], exit_start) > 1.0e-9:
            points.append(exit_start)
        exit_center_z = spec["stem_z_mm"] - radius
        for index in range(1, 17):
            angle = 0.5 * math.pi * (1.0 - index / 16.0)
            points.append((
                spec["x_mm"],
                exit_start[1] + radius * math.cos(angle),
                exit_center_z + radius * math.sin(angle),
            ))
        points.append(spec["feed_xyz_mm"])
    else:
        bezier = _floor_lane_bezier_points(name)
        points.extend(
            _cubic_point(bezier, index / 32.0)
            for index in range(1, 33))
        points.append(_floor_lane_overlap_end(name))
    return tuple(points)


def _floor_lane_cutter(name: str):
    spec = FLOOR_LANE_SPECS[name]
    path = floor_lane_path(name)
    section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(
        spec["diameter_mm"] / 2.0)
    return sweep(section, path=path)


def _floor_feed_mouth_relief(name: str):
    """Shallow rear counter-relief outside one UM/T cover envelope."""
    spec = FLOOR_LANE_SPECS[name]
    if spec["handoff_mode"] != "buried_route_overlap":
        raise ValueError(f"{name} has no annular-route feed mouth")
    radius = (
        spec["diameter_mm"] / 2.0
        + FLOOR_FEED_MOUTH_SHELL_MM
        + FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM
        + FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM)
    z0, z1 = FLOOR_FEED_MOUTH_RELIEF_Z_MM
    feed = spec["feed_xyz_mm"]
    return Pos(feed[0], feed[1], (z0 + z1) / 2.0) * Cylinder(
        radius, z1 - z0)


def integrated_floor_feature_group(index: int):
    """Build only one bounded cutter group to avoid retaining all sweeps."""
    _require_guarded_build()
    if not STAND_FOOT:
        raise RuntimeError("no-floor Obi-Wan has no integral floor features")
    if index == 0:
        cavity = Pos(
            0.0,
            sum(SERVICE_CAVITY_Y_MM) / 2.0,
            sum(SERVICE_CAVITY_Z_MM) / 2.0,
        ) * Box(
            SERVICE_CAVITY_X_MM[1] - SERVICE_CAVITY_X_MM[0],
            SERVICE_CAVITY_Y_MM[1] - SERVICE_CAVITY_Y_MM[0],
            SERVICE_CAVITY_Z_MM[1] - SERVICE_CAVITY_Z_MM[0],
        )
        return "connector_service_cavity", (cavity,)
    if index == 1:
        panel_cutters = [
            Pos(0.0, NL8_CENTER_Y_MM,
                (FOOT_REAR_Z_MM + PANEL_INNER_Z_MM) / 2.0)
            * Cylinder(NL8_CUTOUT_D_MM / 2.0, PANEL_T_MM + 2.0),
        ]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                panel_cutters.append(
                    Pos(
                        sx * NL8_SCREW_PITCH_MM / 2.0,
                        NL8_CENTER_Y_MM
                        + sy * NL8_SCREW_PITCH_MM / 2.0,
                        (FOOT_REAR_Z_MM + PANEL_INNER_Z_MM) / 2.0,
                    ) * Cylinder(
                        NL8_SCREW_D_MM / 2.0, PANEL_T_MM + 2.0))
        return "connector_panel_holes", tuple(panel_cutters)
    lane_names = ("lm", "um", "t")
    if 2 <= index < 2 + len(lane_names):
        name = lane_names[index - 2]
        cutters = [_floor_lane_cutter(name)]
        if FLOOR_LANE_SPECS[name]["handoff_mode"] == "buried_route_overlap":
            cutters.append(_floor_feed_mouth_relief(name))
        return f"floor_lane_{name}", tuple(cutters)
    raise IndexError(index)


def apply_integrated_floor_feature_group(part, index: int):
    if index < 0 or index >= integrated_floor_feature_group_count():
        raise IndexError(index)
    label, cutters = integrated_floor_feature_group(index)
    for cutter in cutters:
        part -= cutter
    part = part.clean()
    solids = tuple(part.solids())
    if (not part.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            f"{label}: integral floor cutter damaged LM; "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def integrated_floor_feature_group_count() -> int:
    return 5 if STAND_FOOT else 0


def integrated_floor_facts() -> dict:
    """Dependency-light dimensions shared by tests, drawings and manifests."""
    lanes = {}
    for name, spec in FLOOR_LANE_SPECS.items():
        lanes[name] = {
            **spec,
            "bend_radius_mm": FLOOR_LANE_BEND_R_MM,
            "service_start_z_mm": FLOOR_LANE_SERVICE_START_Z_MM,
            "route_overlap_mm": (
                FLOOR_LANE_EFFECTIVE_OVERLAP_MM
                if spec["handoff_mode"] == "buried_route_overlap" else 0.0),
            "prefusion_handoff_gap_mm": (
                FLOOR_LANE_PREFUSION_HANDOFF_GAP_MM
                if spec["handoff_mode"] == "buried_route_overlap" else 0.0),
            "owner_cutter_backreach_mm": (
                FLOOR_ROUTE_OWNER_BACKREACH_MM
                if spec["handoff_mode"] == "buried_route_overlap" else 0.0),
            "rear_mouth_relief_radius_mm": (
                spec["diameter_mm"] / 2.0
                + FLOOR_FEED_MOUTH_SHELL_MM
                + FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM
                + FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM
                if spec["handoff_mode"] == "buried_route_overlap" else 0.0),
            "rear_mouth_relief_z_mm": (
                FLOOR_FEED_MOUTH_RELIEF_Z_MM
                if spec["handoff_mode"] == "buried_route_overlap" else None),
            "preview_points": floor_lane_control_points(name),
        }
    return {
        "ownership": "floor_core_lm_and_optional_keyed_bottom",
        "separate_floor_support_exists": False,
        "floor_y_mm": FLOOR_Y_MM,
        "lm_axis_y_mm": LM_AXIS_Y_MM,
        "lm_axis_to_floor_mm": LM_AXIS_Y_MM - FLOOR_Y_MM,
        "foot_width_mm": FOOT_WIDTH_MM,
        "foot_height_mm": FOOT_HEIGHT_MM,
        "foot_z_mm": (FOOT_REAR_Z_MM, FOOT_FRONT_Z_MM),
        "stem_z_mm": STEM_Z_MM,
        "stem_top_y_mm": STEM_TOP_Y_MM,
        "stem_shoulder_half_width_mm": STEM_SHOULDER_HALF_WIDTH_MM,
        "root_fillet_r_mm": ROOT_FILLET_R_MM,
        "panel_z_mm": (FOOT_REAR_Z_MM, PANEL_INNER_Z_MM),
        "panel_height_mm": PANEL_H_MM,
        "nl8_center_y_mm": NL8_CENTER_Y_MM,
        "nl8_cutout_d_mm": NL8_CUTOUT_D_MM,
        "nl8_screw_d_mm": NL8_SCREW_D_MM,
        "nl8_screw_pitch_mm": NL8_SCREW_PITCH_MM,
        "service_cavity_xyz_mm": (
            SERVICE_CAVITY_X_MM,
            SERVICE_CAVITY_Y_MM,
            SERVICE_CAVITY_Z_MM,
        ),
        "floor_lane_count": len(FLOOR_LANE_SPECS),
        "floor_lanes": lanes,
        "feature_group_count": integrated_floor_feature_group_count(),
    }
