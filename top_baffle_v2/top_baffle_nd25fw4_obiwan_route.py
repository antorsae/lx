"""Final Obi-Wan R6F Z-first core tunnels plus free rear cable spans.

The UM and tweeter voids are cut from continuous swept outer-cover solids.
Every insert bypass is a smooth local Z excursion inside that closed cover;
full-width longitudinal webs and local roof-to-bore saddles leave no trapped
shoulder cavities. The short LM terminal lead intentionally floats behind the
carrier with no printed micro-duct. Its existing centerline is preserved by a
minimal rear-open clearance relief through whichever LM-lower owner overlaps
that free span; the relief is a subtraction, never a conduit or cover.
The tweeter route crosses the physical UM cable once near the LM crown,
near orthogonally, with T above UM in +Z and explicit cable/cover clearance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from build123d import (
    Cylinder,
    Face,
    Plane,
    Polyline,
    Pos,
    Rectangle,
    Wire,
    extrude,
    loft,
    make_face,
)
from shapely.geometry import LineString, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from top_baffle_nd25fw4 import (
    L22_CUTOUT,
    L22_PILOT_D_MM,
    STAND_FOOT,
    UM_CUTOUT,
    UM_PILOT_DEPTH_MM,
)
from top_baffle_nd25fw4_flush import (
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
from top_baffle_nd25fw4_obiwan_floor import (
    STEM_HALF_WIDTH_MM,
    STEM_TOP_Y_MM,
    STEM_Z_MM,
)


def _require_guarded_build():
    """Fail closed before any public route factory starts OCC work."""
    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        raise RuntimeError(
            "Obi-Wan route geometry must run through run_memory_guarded.py")

# Physical and nominal diameters from the design contract.
DUCT_D = 8.2
DUCT_R = DUCT_D / 2.0
CUTTER_R = DUCT_R
CUTTER_SIDES = 32
CABLE_D_EST = 7.0
CABLE_R_EST = CABLE_D_EST / 2.0
LM_CABLE_D_EST = 7.8
# The free LM lead is not enclosed by a printed conduit, but the universal
# no-floor web and integral floor stem both overlap its physical centerline.
# A 0.06-mm radial allowance clears the independent physical-radius +0.05-mm
# witness while retaining 0.01 mm of Boolean margin. The centerline is fixed.
LM_FREE_LEAD_RELIEF_RADIAL_CLEAR_MM = 0.06
LM_FREE_LEAD_RELIEF_R = (
    LM_CABLE_D_EST / 2.0 + LM_FREE_LEAD_RELIEF_RADIAL_CLEAR_MM)

TS_DUCT_D = 6.0
TS_DUCT_R = TS_DUCT_D / 2.0
TS_CUTTER_R = TS_DUCT_R
TS_CUTTER_SIDES = 24
TS_CABLE_D_EST = 5.2

# Two complete 0.4-mm extrusion widths are the minimum reliable
# non-load-bearing skin.  The coupon remains the physical print gate.
TUNNEL_SKIN = 0.80
TUNNEL_FLOOR_SKIN = TUNNEL_SKIN  # compatibility/reporting name
SIDE_WALL = TUNNEL_SKIN
TS_SIDE_WALL = TUNNEL_SKIN
TUNNEL_CABLE_CLEAR = 0.05
TUNNEL_FUSE_OVERLAP = 0.55
MAIN_OUTER_R = CUTTER_R + TUNNEL_SKIN
TS_OUTER_R = TS_CUTTER_R + TUNNEL_SKIN
TUNNEL_ROOF_SKIN = 0.85  # 0.05 mm avoids a tangent seat-plane union
TUBE_SECTION_SPACING = 5.5
TUBE_SECTION_SIDES = 8
ANCHOR_SECTION_SPACING = 5.0
BURIAL_WEB_TUBE_OVERLAP = TUNNEL_FUSE_OVERLAP
BURIAL_WEB_LATERAL_OVERLAP = TUNNEL_FUSE_OVERLAP
LM_MAIN_CUTTER_SEGMENT_COUNT = 8
LM_T_CUTTER_SEGMENT_COUNT = 12
LM_ROUTE_CUTTER_GROUP_COUNT = (
    LM_MAIN_CUTTER_SEGMENT_COUNT + LM_T_CUTTER_SEGMENT_COUNT)
# The UM carrier retains only the buried tweeter route. The main UM cable
# leaves its flush LM mouth and runs freely behind the UM carrier.
UM_ROUTE_CUTTER_GROUP_COUNT = 1

CORE_REAR_Z = 6.8
ANCHOR_LEG_W = TUNNEL_SKIN
LM_SEAT_MEMBRANE_BOTTOM_Z = LM_SEAT_Z - TUNNEL_ROOF_SKIN
UM_SEAT_MEMBRANE_BOTTOM_Z = UM_SEAT_Z - TUNNEL_ROOF_SKIN
ANCHOR_START_OVERLAP = 0.40
LM_CORE_R = 113.0
UM_CORE_R = 51.7
ROUTE_SPLIT_GAP = 0.20
CUTTER_SPLIT_OVERLAP = 1.0
UM_T_CUTTER_MOUTH_OVERSHOOT = 2.20
TS_ADDON_SUPPORT_MIN_Y = 416.0
# The raw tweeter crescent is recut outside the UM carrier at R51.9. The
# printed D6 cover ends at the native R51.7 UM edge; only its cutter extends
# 2.2 mm through that mouth. The cable then crosses the 0.2-mm gap freely, so
# neither owner emits a point horn, tongue, socket, or crescent tunnel.
TS_TWEETER_FLUSH_R = UM_CORE_R + ROUTE_SPLIT_GAP
# Nominal buried layers. At the physical crossover T rises and UM dips so the
# two cable envelopes retain at least 0.80 mm clearance. UM is already free
# cable there; this is not a claim for a printed separator web.
TRENCH_CENTER_Z = LM_SEAT_Z - CUTTER_R - TUNNEL_ROOF_SKIN
TS_TRENCH_CENTER_Z = LM_SEAT_Z - TS_CUTTER_R - TUNNEL_ROOF_SKIN
TS_UM_CENTER_Z = UM_SEAT_Z - TS_CUTTER_R - TUNNEL_ROOF_SKIN
# The free T suffix must pass behind—not through—the acoustic crescent. Its
# physical D5.2 envelope therefore tops out 0.4 mm behind the core rear face.
TS_FREE_CABLE_REAR_CLEARANCE = 0.40
TS_FREE_CABLE_Z = (
    CORE_REAR_Z - TS_CABLE_D_EST / 2.0 - TS_FREE_CABLE_REAR_CLEARANCE)
CROSSOVER_MIN_CLEARANCE = 0.80
CROSSOVER_TARGET_CLEARANCE = 1.00
CROSSOVER_T_Z = TS_UM_CENTER_Z - 0.5
CROSSOVER_UM_Z = (
    CROSSOVER_T_Z - CUTTER_R - TS_CUTTER_R
    - CROSSOVER_TARGET_CLEARANCE)
CROSSOVER_HALF_LENGTH = 17.0
CROSSOVER_LEG_OMIT_RADIUS = CUTTER_R + TS_OUTER_R + 1.0

# The insert solids are bypassed below, leaving a covered rear bump.
INSERT_COVER_CLEAR = 0.40
BOOLEAN_CLEARANCE_MARGIN = 0.05
LM_MAIN_BUMP_Z = (
    PAD_FACE_Z - INSERT_COVER_CLEAR - MAIN_OUTER_R
    - BOOLEAN_CLEARANCE_MARGIN)
LM_TS_BUMP_Z = (
    PAD_FACE_Z - INSERT_COVER_CLEAR - TS_OUTER_R
    - BOOLEAN_CLEARANCE_MARGIN)
# These two no-floor routes approach their pads on a diagonal, so the exact
# swept-cover distance is slightly smaller than the axial design equation.
# Keep the extra Z relief local to only the two responsible bypasses; the
# floor carrier retains its independently hardware-controlled bump depths.
NO_FLOOR_MAIN_PAD_BUMP_RELIEF = 0.20
NO_FLOOR_T_PAD_BUMP_RELIEF = 0.25
UM_PILOT_FLOOR_Z = (
    UM_SEAT_Z - UM_PILOT_DEPTH_MM - UM_PAD_FLOOR_MM)
UM_TS_BUMP_Z = (
    UM_PILOT_FLOOR_Z - INSERT_COVER_CLEAR - TS_OUTER_R
    - BOOLEAN_CLEARANCE_MARGIN)
# Every named insert bypass gets a local full-width solid saddle from the
# conduit roof to the blind-bore floor.  The saddle overlaps the existing
# round cover by 0.55 mm but never extends below it, so it closes the old
# 0.45..0.85-mm trapped cavity without increasing rear bump depth.
BUMP_BACKFILL_TUBE_OVERLAP = TUNNEL_FUSE_OVERLAP

# Plan intent from concept preview v3.
LM_ROUTE_R = 104.75
LM_ROUTE_START_DEG = 300.0
LM_ROUTE_ARC_START_DEG = 315.0
LM_ROUTE_END_DEG = 58.0

# Both state-owned solids carry the same central rear-facing feed mouths:
# the integral floor stem in floor mode and the fused bridge web otherwise.
# This deletes the former lateral R114/support-hardware detour while keeping
# the proven symmetric shallow rise into the LM annulus.
NO_FLOOR_MAIN_FEED_XY = np.asarray((8.0, 82.0), dtype=float)
NO_FLOOR_T_FEED_XY = np.asarray((-8.0, 82.0), dtype=float)
CENTRAL_MAIN_FEED_XY = NO_FLOOR_MAIN_FEED_XY
CENTRAL_T_FEED_XY = NO_FLOOR_T_FEED_XY
NO_FLOOR_FEED_REAR_Z = PAD_FACE_Z
NO_FLOOR_MAIN_FEED_RISE_LENGTH = 24.0
NO_FLOOR_T_FEED_RISE_LENGTH = 27.5
NO_FLOOR_FEED_CUTTER_EXTENSION = 8.0
STANDARD_CUTTER_EXTENSION = 1.5
# Dependency-neutral copy of the bridge's immutable insert-bearing core.
# bridge.py imports this route module, so the equality is bound by tests
# rather than a circular source import.
NO_FLOOR_BRIDGE_CORE_BOUNDS = (-31.0, 14.0, 31.0, 90.25)
FLOOR_STEM_CORE_BOUNDS = (
    -STEM_HALF_WIDTH_MM, 0.0,
    STEM_HALF_WIDTH_MM, STEM_TOP_Y_MM)
NO_FLOOR_FEED_START_HANDLE = 32.0
# Splay the two internal entry handles away from the R94 W22 rear body while
# keeping their rear mouths symmetric. The mirrored 65/115-degree bearings
# retain the complete covers inside the existing bridge/ring material.
NO_FLOOR_FEED_START_BEARING_DEG = 65.0
NO_FLOOR_FEED_END_HANDLE = 26.0

UM_ENTRY_ANGLE_DEG = 283.0
UM_MOUTH_R = 40.5
UM_MOUTH_Z = 2.70
UM_TERMINAL_PLAN_BEND_R = 15.0
# Tangent-preserving UM crown bridge handles. The selected R15 quarter-circle
# entry clears every joint-ear envelope and keeps the complete cover outside
# the D190 acoustic opening; the compact cubic retains R15.04 minimum.
MAIN_BRIDGE_START_HANDLE_MM = 11.0
MAIN_BRIDGE_END_HANDLE_MM = 17.0

TS_LM_ROUTE_START_DEG = 240.0
TS_LM_ARC_START_DEG = 225.0
TS_LM_ROUTE_END_DEG = 120.0
TS_UM_ROUTE_START_DEG = 315.0
TS_UM_ROUTE_END_DEG = 60.0
TS_UM_ROUTE_R = 46.35
# Tangent-handle lengths for the shortest practical crown bridge that keeps
# the fixed arc endpoints, remains comfortably above the R14 bend floor and
# crosses the UM route essentially orthogonally.
TS_BRIDGE_START_HANDLE_MM = 65.0
TS_BRIDGE_END_HANDLE_MM = 60.0

LM_LEAD_ANGLE_DEG = 269.5
LM_LEAD_OUTER_R = 114.0
LM_LEAD_INNER_R = 93.85
LM_LEAD_LENGTH = LM_LEAD_OUTER_R - LM_LEAD_INNER_R
LM_FREE_CABLE_OUTER_Z = 0.40
LM_FREE_CABLE_INNER_Z = 3.80
LM_FREE_CABLE_RISE_START_R = 103.0
LM_FREE_CABLE_REAR_CLEARANCE = (
    PAD_FACE_Z - LM_FREE_CABLE_OUTER_Z - LM_CABLE_D_EST / 2.0)


def _polar(center, radius, angle_deg):
    angle = math.radians(angle_deg)
    return np.asarray((
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
    ), dtype=float)


def _cubic(u, p0, p1, p2, p3):
    u = np.asarray(u, dtype=float)[..., None]
    return ((1.0 - u) ** 3 * p0
            + 3.0 * (1.0 - u) ** 2 * u * p1
            + 3.0 * (1.0 - u) * u ** 2 * p2
            + u ** 3 * p3)


def _join(*arrays):
    out = []
    for i, array in enumerate(arrays):
        array = np.asarray(array, dtype=float)
        out.append(array if i == len(arrays) - 1 else array[:-1])
    return np.vstack(out)


def _arc(center, radius, start_deg, stop_deg, count):
    angles = np.radians(np.linspace(start_deg, stop_deg, count))
    return np.column_stack((
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ))


def _plan_lengths(plan):
    return np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(plan, axis=0), axis=1))))


def _resample(plan, spacing_mm):
    source_s = _plan_lengths(plan)
    count = max(2, int(math.ceil(source_s[-1] / spacing_mm)))
    stations = np.linspace(0.0, source_s[-1], count + 1)
    xy = np.column_stack((
        np.interp(stations, source_s, plan[:, 0]),
        np.interp(stations, source_s, plan[:, 1]),
    ))
    return stations, xy


def _smooth01(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return 10.0 * x ** 3 - 15.0 * x ** 4 + 6.0 * x ** 5


def _local_min(z, stations, center, low, half):
    d = np.abs(stations - center)
    active = d < half
    target = low + (z[active] - low) * _smooth01(d[active] / half)
    z[active] = np.minimum(z[active], target)


def _local_max(z, stations, center, high, half):
    d = np.abs(stations - center)
    active = d < half
    target = high + (z[active] - high) * _smooth01(d[active] / half)
    z[active] = np.maximum(z[active], target)


def _station_near(plan, point):
    index = int(np.argmin(np.linalg.norm(plan - np.asarray(point), axis=1)))
    return float(_plan_lengths(plan)[index])


# -- main UM path -----------------------------------------------------
_MAIN_FEED = CENTRAL_MAIN_FEED_XY.copy()
_MAIN_ARC_START = _polar(
    L22_CUTOUT[:2], LM_ROUTE_R, LM_ROUTE_ARC_START_DEG)
_MAIN_FEED_START_DIRECTION = np.asarray((
        math.cos(math.radians(NO_FLOOR_FEED_START_BEARING_DEG)),
        math.sin(math.radians(NO_FLOOR_FEED_START_BEARING_DEG)),
    ))
_MAIN_ENTRY = _cubic(
    np.linspace(0.0, 1.0, 801),
    _MAIN_FEED,
    _MAIN_FEED + _MAIN_FEED_START_DIRECTION * NO_FLOOR_FEED_START_HANDLE,
    _MAIN_ARC_START - np.asarray((math.sqrt(0.5), math.sqrt(0.5))) * (
        NO_FLOOR_FEED_END_HANDLE),
    _MAIN_ARC_START,
)
_MAIN_ARC = _arc(
    L22_CUTOUT[:2], LM_ROUTE_R,
    LM_ROUTE_ARC_START_DEG, 360.0 + LM_ROUTE_END_DEG, 4001)
LM_BRIDGE_START = _MAIN_ARC[-1]
UM_MOUTH_POINT = _polar(UM_CUTOUT[:2], UM_MOUTH_R, UM_ENTRY_ANGLE_DEG)
_UM_TERMINAL_U = np.asarray((
    math.cos(math.radians(UM_ENTRY_ANGLE_DEG)),
    math.sin(math.radians(UM_ENTRY_ANGLE_DEG)),
))
_UM_TERMINAL_V = np.asarray((-_UM_TERMINAL_U[1], _UM_TERMINAL_U[0]))
# Exact planar R15 quarter-circle.  The endpoint is immutable on the 283°
# terminal axis, while the selected approach makes its outlet tangent the
# clockwise 193° circumference direction.  The free R20 bend then travels
# away from (rather than through) the known D60 motor/283° strut envelope.
# The equivalent mouth-to-entry chord is 328° and ``+1`` selects the outward
# circle center.  D190, joint-ear, pilot-bore, and shell constraints are
# independently rechecked below and in ``test_obiwan_r6f.py``.
UM_TERMINAL_ARC_ENTRY_BEARING_DEG = 328.0
UM_TERMINAL_ARC_SIDE = 1.0
_UM_ENTRY_DIRECTION = np.asarray((
    math.cos(math.radians(UM_TERMINAL_ARC_ENTRY_BEARING_DEG)),
    math.sin(math.radians(UM_TERMINAL_ARC_ENTRY_BEARING_DEG)),
))
UM_ENTRY_POINT = (
    UM_MOUTH_POINT
    + UM_TERMINAL_PLAN_BEND_R * math.sqrt(2.0) * _UM_ENTRY_DIRECTION)
_UM_ENTRY_TO_MOUTH = UM_MOUTH_POINT - UM_ENTRY_POINT
_UM_ENTRY_TO_MOUTH_PERP = np.asarray((
    -_UM_ENTRY_TO_MOUTH[1], _UM_ENTRY_TO_MOUTH[0]))
_UM_ENTRY_TO_MOUTH_PERP /= np.linalg.norm(_UM_ENTRY_TO_MOUTH_PERP)
_UM_PLAN_BEND_CENTER = (
    (UM_ENTRY_POINT + UM_MOUTH_POINT) / 2.0
    + UM_TERMINAL_ARC_SIDE
    * UM_TERMINAL_PLAN_BEND_R / math.sqrt(2.0)
    * _UM_ENTRY_TO_MOUTH_PERP)
_UM_ARC_START_RADIUS = (
    (UM_ENTRY_POINT - _UM_PLAN_BEND_CENTER)
    / UM_TERMINAL_PLAN_BEND_R)
_UM_ARC_END_RADIUS = (
    (UM_MOUTH_POINT - _UM_PLAN_BEND_CENTER)
    / UM_TERMINAL_PLAN_BEND_R)


def _rotate90(vector):
    return np.asarray((-vector[1], vector[0]))


_UM_ARC_SIGN = (1.0 if float(np.dot(
    _rotate90(_UM_ARC_START_RADIUS), _UM_ARC_END_RADIUS)) > 0.0 else -1.0)
UM_ENTRY_R = float(np.linalg.norm(UM_ENTRY_POINT - np.asarray(UM_CUTOUT[:2])))
_MAIN_ARC_TANGENT = np.asarray((
    -math.sin(math.radians(LM_ROUTE_END_DEG)),
    math.cos(math.radians(LM_ROUTE_END_DEG)),
))
_MAIN_THROAT_TANGENT = (
    _UM_ARC_SIGN * _rotate90(_UM_ARC_START_RADIUS))
UM_MOUTH_TANGENT = tuple(map(
    float, _UM_ARC_SIGN * _rotate90(_UM_ARC_END_RADIUS)))
_C1 = (
    LM_BRIDGE_START + MAIN_BRIDGE_START_HANDLE_MM * _MAIN_ARC_TANGENT)
_C2 = (
    UM_ENTRY_POINT - MAIN_BRIDGE_END_HANDLE_MM * _MAIN_THROAT_TANGENT)
_MAIN_BRIDGE = _cubic(
    np.linspace(0.0, 1.0, 2001),
    LM_BRIDGE_START, _C1, _C2, UM_ENTRY_POINT)
_MAIN_THROAT_PHI = np.linspace(0.0, _UM_ARC_SIGN * math.pi / 2.0, 801)
_UM_ARC_START_ANGLE = math.atan2(
    _UM_ARC_START_RADIUS[1], _UM_ARC_START_RADIUS[0])
_MAIN_THROAT = np.asarray([
    _UM_PLAN_BEND_CENTER
    + UM_TERMINAL_PLAN_BEND_R * np.asarray((
        math.cos(_UM_ARC_START_ANGLE + phi),
        math.sin(_UM_ARC_START_ANGLE + phi),
    ))
    for phi in _MAIN_THROAT_PHI
])
_MAIN_PLAN = _join(_MAIN_ENTRY, _MAIN_ARC, _MAIN_BRIDGE, _MAIN_THROAT)
_MAIN_PLAN_S = _plan_lengths(_MAIN_PLAN)
ROUTE_LENGTH = float(_MAIN_PLAN_S[-1])
LM_ENTRY_LENGTH = float(_plan_lengths(_MAIN_ENTRY)[-1])
LM_ARC_LENGTH = float(_plan_lengths(_MAIN_ARC)[-1])
BRIDGE_LENGTH = float(_plan_lengths(_MAIN_BRIDGE)[-1])
THROAT_LENGTH = UM_TERMINAL_PLAN_BEND_R * math.pi / 2.0
MAIN_UM_ENTRY_S = ROUTE_LENGTH - THROAT_LENGTH


# -- tweeter path -----------------------------------------------------
_TS_FEED = CENTRAL_T_FEED_XY.copy()
_TS_LM_ARC_START = _polar(
    L22_CUTOUT[:2], LM_ROUTE_R, TS_LM_ARC_START_DEG)
_TS_FEED_START_DIRECTION = np.asarray((
        math.cos(math.radians(180.0 - NO_FLOOR_FEED_START_BEARING_DEG)),
        math.sin(math.radians(180.0 - NO_FLOOR_FEED_START_BEARING_DEG)),
    ))
_TS_ENTRY = _cubic(
    np.linspace(0.0, 1.0, 801),
    _TS_FEED,
    _TS_FEED + _TS_FEED_START_DIRECTION * NO_FLOOR_FEED_START_HANDLE,
    _TS_LM_ARC_START - np.asarray((-math.sqrt(0.5), math.sqrt(0.5))) * (
        NO_FLOOR_FEED_END_HANDLE),
    _TS_LM_ARC_START,
)
_TS_LM_ARC = _arc(
    L22_CUTOUT[:2], LM_ROUTE_R,
    TS_LM_ARC_START_DEG, TS_LM_ROUTE_END_DEG, 4001)
TS_LM_BRIDGE_START = _TS_LM_ARC[-1]
TS_UM_ARC_START = _polar(
    UM_CUTOUT[:2], TS_UM_ROUTE_R, TS_UM_ROUTE_START_DEG)
_TS_LM_TANGENT = np.asarray((
    math.sin(math.radians(TS_LM_ROUTE_END_DEG)),
    -math.cos(math.radians(TS_LM_ROUTE_END_DEG)),
))
_TS_UM_START_TANGENT = np.asarray((
    -math.sin(math.radians(TS_UM_ROUTE_START_DEG)),
    math.cos(math.radians(TS_UM_ROUTE_START_DEG)),
))
_TS_C1 = (
    TS_LM_BRIDGE_START + TS_BRIDGE_START_HANDLE_MM * _TS_LM_TANGENT)
_TS_C2 = (
    TS_UM_ARC_START - TS_BRIDGE_END_HANDLE_MM * _TS_UM_START_TANGENT)
_TS_BRIDGE = _cubic(
    np.linspace(0.0, 1.0, 3001),
    TS_LM_BRIDGE_START, _TS_C1, _TS_C2, TS_UM_ARC_START)
_TS_UM_ARC = _arc(
    UM_CUTOUT[:2], TS_UM_ROUTE_R,
    TS_UM_ROUTE_START_DEG, 360.0 + TS_UM_ROUTE_END_DEG, 4001)
TS_UM_HANDOFF_START = _TS_UM_ARC[-1]
TS_TWEETER_MOUTH = np.asarray((0.0, 430.0))
_TS_UM_END_TANGENT = np.asarray((
    -math.sin(math.radians(TS_UM_ROUTE_END_DEG)),
    math.cos(math.radians(TS_UM_ROUTE_END_DEG)),
))
_TS_H1 = TS_UM_HANDOFF_START + 8.0 * _TS_UM_END_TANGENT
_TS_H2 = np.asarray((0.0, 414.0))
_TS_HANDOFF = _cubic(
    np.linspace(0.0, 1.0, 1201),
    TS_UM_HANDOFF_START, _TS_H1, _TS_H2, TS_TWEETER_MOUTH)
_TS_PLAN = _join(_TS_ENTRY, _TS_LM_ARC, _TS_BRIDGE, _TS_UM_ARC, _TS_HANDOFF)
_TS_PLAN_S = _plan_lengths(_TS_PLAN)
TS_ROUTE_LENGTH = float(_TS_PLAN_S[-1])
TS_ENTRY_LENGTH = float(_plan_lengths(_TS_ENTRY)[-1])
TS_LM_ARC_LENGTH = float(_plan_lengths(_TS_LM_ARC)[-1])
TS_BRIDGE_LENGTH = float(_plan_lengths(_TS_BRIDGE)[-1])
TS_UM_ARC_LENGTH = float(_plan_lengths(_TS_UM_ARC)[-1])
TS_HANDOFF_LENGTH = float(_plan_lengths(_TS_HANDOFF)[-1])
TS_UM_ENTRY_S = TS_ROUTE_LENGTH - TS_UM_ARC_LENGTH - TS_HANDOFF_LENGTH
TS_CORE_END_S = TS_ROUTE_LENGTH - TS_HANDOFF_LENGTH

# Find the upper native R51.7 mouth so UM cover generation can stop as soon
# as the route leaves the ring. The final owner crop is exact at R51.7; no
# printed cover or free-span conduit is generated above this station.
TS_UM_CORE_COVER_END_R = UM_CORE_R
_TS_RADII_FROM_UM = np.linalg.norm(
    _TS_PLAN - np.asarray(UM_CUTOUT[:2]), axis=1)
_TS_POST_INDICES = np.flatnonzero(
    (_TS_PLAN_S >= TS_CORE_END_S)
    & (_TS_RADII_FROM_UM >= TS_UM_CORE_COVER_END_R))
if not len(_TS_POST_INDICES):
    raise RuntimeError("T route never reaches the upper UM flush radius")
_TS_COVER_END_INDEX = int(_TS_POST_INDICES[0])
_TS_COVER_END_PREV = max(0, _TS_COVER_END_INDEX - 1)
_r0 = float(_TS_RADII_FROM_UM[_TS_COVER_END_PREV])
_r1 = float(_TS_RADII_FROM_UM[_TS_COVER_END_INDEX])
_q = ((TS_UM_CORE_COVER_END_R - _r0) / max(_r1 - _r0, 1e-12))
TS_UM_CORE_COVER_END_S = float(
    _TS_PLAN_S[_TS_COVER_END_PREV]
    + _q * (_TS_PLAN_S[_TS_COVER_END_INDEX]
            - _TS_PLAN_S[_TS_COVER_END_PREV]))


# Exact plan crossing and local directions.
_CROSS_GEOM = LineString(_MAIN_PLAN).intersection(LineString(_TS_PLAN))
if _CROSS_GEOM.geom_type != "Point":
    raise RuntimeError(f"Obi-Wan crown routes must cross once, got {_CROSS_GEOM}")
CROSSOVER_XY = np.asarray(_CROSS_GEOM.coords[0], dtype=float)
CROSSOVER_MAIN_S = float(LineString(_MAIN_PLAN).project(_CROSS_GEOM))
CROSSOVER_TS_S = float(LineString(_TS_PLAN).project(_CROSS_GEOM))


def _plan_tangent(plan, station):
    s = _plan_lengths(plan)
    i = int(np.searchsorted(s, station))
    i = min(max(i, 2), len(plan) - 3)
    tangent = plan[i + 2] - plan[i - 2]
    return tangent / np.linalg.norm(tangent)


_CROSS_MAIN_TANGENT = _plan_tangent(_MAIN_PLAN, CROSSOVER_MAIN_S)
_CROSS_TS_TANGENT = _plan_tangent(_TS_PLAN, CROSSOVER_TS_S)
CROSSOVER_ANGLE_DEG = math.degrees(math.acos(np.clip(
    abs(float(np.dot(_CROSS_MAIN_TANGENT, _CROSS_TS_TANGENT))),
    -1.0, 1.0)))


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


_LM_PILOT_BY_ANGLE = dict(zip(
    (0.0, 60.0, 120.0, 180.0, 240.0, 300.0), LM_PILOT_XY))
_UM_PILOT_BY_ANGLE = dict(zip((58.0, 148.0, 238.0, 328.0), UM_PILOT_XY))

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
    ("um_pilot_328", _UM_PILOT_BY_ANGLE[328.0], UM_TS_BUMP_Z, 22.0),
    ("um_pilot_58", _UM_PILOT_BY_ANGLE[58.0], UM_TS_BUMP_Z, 28.0),
))

MAIN_ANCHOR_KEEPOUTS = tuple((np.asarray(xy), 5.0)
                             for xy in LM_PILOT_XY)
T_ANCHOR_KEEPOUTS = (
    *tuple((np.asarray(xy), 5.0) for xy in LM_PILOT_XY),
    # The captive UM magnet at 50.5 degrees sits between the continuous T tube
    # and the 58-degree insert boss. The round tube retains >2.4 mm beyond
    # its 0.8-mm grown pocket keepout, but the optional membrane anchor fin
    # would enter that pocket. Grow only this anchor-mask exclusion; the
    # insert backfill and continuous closed tube remain unchanged.
    *tuple((np.asarray(xy),
            5.4 if angle == 58.0 else UM_PAD_D_MM / 2.0)
           for angle, xy in zip((58.0, 148.0, 238.0, 328.0), UM_PILOT_XY)),
)


def _central_rear_feed_rise(stations, nominal_z, rise_length):
    """State-owned rear-face mouth to buried layer with a shallow ramp.

    The integral floor continuation arrives tangent to the XY feed bearing,
    so floor mode uses a quintic zero-slope/zero-curvature rise at both ends.
    That makes the connector continuation and annular route G2 in Z while
    retaining the common feed mouth.  The already-released no-floor bridge
    keeps its cubic ease-out and therefore its exact geometry.  In both
    states the owner supplies the surrounding wall; no separate external
    raceway is added.
    """
    stations = np.asarray(stations, dtype=float)
    u = np.clip(stations / rise_length, 0.0, 1.0)
    rise = nominal_z - NO_FLOOR_FEED_REAR_Z
    if STAND_FOOT:
        return NO_FLOOR_FEED_REAR_Z + rise * _smooth01(u)
    return NO_FLOOR_FEED_REAR_Z + rise * (1.0 - (1.0 - u) ** 3)


def _main_xyz(spacing_mm):
    stations, xy = _resample(_MAIN_PLAN, spacing_mm)
    z = _central_rear_feed_rise(
        stations, TRENCH_CENTER_Z, NO_FLOOR_MAIN_FEED_RISE_LENGTH)
    final_bump = next(jump for jump in MAIN_COVERED_BUMPS
                      if jump.name == "lm_pilot_60")
    for jump in MAIN_COVERED_BUMPS:
        if jump is final_bump:
            continue
        _local_min(z, stations, jump.station, jump.low_z,
                   jump.half_length)
    # One C2 piecewise profile owns the last LM-pilot bypass, crossover
    # layer and nearby 283-degree terminal descent. This avoids profile
    # intersections and therefore avoids curvature cusps.
    pilot_center = final_bump.station
    pilot_low = final_bump.low_z
    pilot_half = final_bump.half_length
    ramp_start = pilot_center - pilot_half
    descending = (stations >= ramp_start) & (stations <= pilot_center)
    u = ((stations[descending] - ramp_start)
         / (pilot_center - ramp_start))
    z[descending] = (TRENCH_CENTER_Z
                     + (pilot_low - TRENCH_CENTER_Z) * _smooth01(u))
    rising = (stations > pilot_center) & (stations <= CROSSOVER_MAIN_S)
    u = ((stations[rising] - pilot_center)
         / (CROSSOVER_MAIN_S - pilot_center))
    z[rising] = (pilot_low
                 + (CROSSOVER_UM_Z - pilot_low) * _smooth01(u))
    terminal = stations > CROSSOVER_MAIN_S
    u = ((stations[terminal] - CROSSOVER_MAIN_S)
         / (ROUTE_LENGTH - CROSSOVER_MAIN_S))
    z[terminal] = (CROSSOVER_UM_Z
                   + (UM_MOUTH_Z - CROSSOVER_UM_Z) * _smooth01(u))
    return stations, np.column_stack((xy, z))


def _ts_xyz(spacing_mm):
    stations, xy = _resample(_TS_PLAN, spacing_mm)
    # The UM seat is 2 mm higher than the LM seat.  Raise the nominal T
    # tunnel smoothly after it enters the UM ring.
    transition = _smooth01((stations - TS_UM_ENTRY_S) / 24.0)
    nominal_z = TS_TRENCH_CENTER_Z + transition * (
        TS_UM_CENTER_Z - TS_TRENCH_CENTER_Z)
    feed_z = _central_rear_feed_rise(
        stations, TS_TRENCH_CENTER_Z,
        NO_FLOOR_T_FEED_RISE_LENGTH)
    # The rear-feed rise is complete long before the UM transition.
    z = np.where(
        stations < NO_FLOOR_T_FEED_RISE_LENGTH, feed_z, nominal_z)
    for bump in T_COVERED_BUMPS:
        _local_min(z, stations, bump.station, bump.low_z,
                   bump.half_length)
    _local_max(z, stations, CROSSOVER_TS_S,
               CROSSOVER_T_Z, CROSSOVER_HALF_LENGTH)
    # Leave the last 58-degree bore bypass at zero slope, then continue its
    # rearward motion smoothly so the cable is already behind the crescent at
    # the native upper-UM mouth. The suffix stays at that Z as free cable.
    free_start = next(
        bump.station for bump in T_COVERED_BUMPS
        if bump.name == "um_pilot_58")
    free_start_z = float(np.interp(free_start, stations, z))
    descent = ((stations >= free_start)
               & (stations <= TS_UM_CORE_COVER_END_S))
    u = ((stations[descent] - free_start)
         / (TS_UM_CORE_COVER_END_S - free_start))
    z[descent] = (free_start_z
                  + (TS_FREE_CABLE_Z - free_start_z) * _smooth01(u))
    z[stations > TS_UM_CORE_COVER_END_S] = TS_FREE_CABLE_Z
    return stations, np.column_stack((xy, z))


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


def _polygon_prism(polygon, z0, z1):
    """One absolute-Z prism used by compact local backfill saddles."""
    polygon = orient(polygon, sign=1.0)
    outer = Wire(Polyline(*[
        (float(x), float(y)) for x, y in polygon.exterior.coords
    ]).edges())
    holes = [
        Wire(Polyline(*[(float(x), float(y)) for x, y in ring.coords]).edges())
        for ring in polygon.interiors
    ]
    return Pos(0.0, 0.0, z0) * extrude(
        Face(outer, holes), amount=z1 - z0)


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


def route_cable_points(spacing_mm: float = 0.5):
    """Complete UM centerline from LM feed to the 283-degree mouth."""
    return _main_xyz(spacing_mm)[1]


def ts_cable_points(spacing_mm: float = 0.5):
    """Complete tweeter centerline including crossover and handoff."""
    return _ts_xyz(spacing_mm)[1]


def lm_cable_points(spacing_mm: float = 0.5):
    """Short reference-only LM lead floating behind the 269.5-degree lane."""
    count = max(2, int(math.ceil(LM_LEAD_LENGTH / spacing_mm)))
    radii = np.linspace(LM_LEAD_OUTER_R, LM_LEAD_INNER_R, count + 1)
    angle = math.radians(LM_LEAD_ANGLE_DEG)
    xy = np.column_stack((
        L22_CUTOUT[0] + radii * math.cos(angle),
        L22_CUTOUT[1] + radii * math.sin(angle),
    ))
    # No printed micro-duct remains. A 0.5-degree plan dogleg and 0.4-mm rear
    # center height place the lead between the two centered bridge-duct exits
    # with 1.0 mm clearance below the deepest z=5.3 carrier datum at its outer
    # station. A gentle circular rise begins at R103 before the R95/D190 mouth.
    # Where the universal LM-lower owners overlap the inner rise, the separate
    # rear-open subtractive relief keeps this same D7.8 envelope clear.
    travel = np.clip(
        LM_FREE_CABLE_RISE_START_R - radii, 0.0,
        LM_FREE_CABLE_RISE_START_R - LM_LEAD_INNER_R)
    rise_length = LM_FREE_CABLE_RISE_START_R - LM_LEAD_INNER_R
    rise_height = LM_FREE_CABLE_INNER_Z - LM_FREE_CABLE_OUTER_Z
    rise_radius = (rise_length ** 2 + rise_height ** 2) / (2.0 * rise_height)
    z = (LM_FREE_CABLE_OUTER_Z + rise_radius
         - np.sqrt(np.maximum(rise_radius ** 2 - travel ** 2, 0.0)))
    return np.column_stack((xy, z))


def lm_free_lead_relief_cutter():
    """Continuous rear-open clearance for the un-ducted D7.8 LM lead.

    This is deliberately a bare subtractive relief around the immutable
    reference centerline. Its circular envelope crosses the rear exterior
    plane in both stand states, so it cannot become a sealed micro-duct.
    """
    _require_guarded_build()
    return _round_tube(lm_cable_points(1.0), LM_FREE_LEAD_RELIEF_R)


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


def _tube_section_points(points):
    """Return the globally phased ruled-loft center sections for a path."""
    points = np.asarray(points, dtype=float)
    source_s = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    section_s = np.linspace(
        0.0, source_s[-1],
        max(2, int(math.ceil(source_s[-1] / TUBE_SECTION_SPACING))) + 1)
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


def _round_tube(points, radius):
    """BREP-stable circumscribed polygon tube along a 3-D centerline.

    OCC's Frenet pipe reports a valid solid for these long compound curves
    but classifies its interior backwards in cylinder Booleans (intersection
    returned the cylinder complement). A 5.5-mm ruled section loft produces
    correctly classified, closed solids without OCC's pathological global
    B-spline/surface intersection. At the LM route's R20 its maximum
    centerline chord sagitta is 0.191 mm, inside the 0.20-mm radial cable
    clearance. Circumscription preserves the requested minimum round
    radius at every section; the octagon reduces Boolean face count
    without reducing the inscribed duct diameter.
    """
    return _round_tube_from_global_sections(
        _tube_section_points(points), radius)


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


def _global_suffix_first_section(points, start_station):
    """Return the first authoritative section retained by a route suffix."""
    points = np.asarray(points, dtype=float)
    sections = _tube_section_points(points)
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if not 0.0 <= start_station <= total:
        raise ValueError((start_station, total))
    section_stations = np.linspace(0.0, total, len(sections))
    containing_edge = int(np.searchsorted(
        section_stations, start_station, side="right") - 1)
    containing_edge = min(containing_edge, len(sections) - 2)
    return max(0, containing_edge - 2)


def _round_tube_global_suffix(points, radius, start_station):
    """Return a suffix on the full path's authoritative section phase.

    Selecting consecutive global sections avoids both failure modes of a
    tangent half-space crop on a looping route: retained upstream islands and
    locally re-phased octagons that erode the nominal wall. Two predecessor
    edges provide more than one outer diameter of positive overlap before the
    later owner-domain crop, while the explicit arc-length station cannot
    alias to a spatially nearby point elsewhere on a looping route.
    """
    sections = _tube_section_points(points)
    first = _global_suffix_first_section(points, start_station)
    return _round_tube_from_global_sections(
        sections, radius, first, len(sections) - 1)


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


def _burial_web(
        points, outer_radius, anchor_base_z, clearance=0.0):
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
        q0 = point[:2] - half_width * normal
        q1 = point[:2] + half_width * normal
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
    for run in _point_runs_with_boundary_overlap(points, keep, allowed):
        web = _burial_web(run, outer_radius, anchor_base_z, clearance)
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
    return NO_FLOOR_FEED_CUTTER_EXTENSION


def _owner_cutter_points(points, owner):
    """Return the exact cutter path used by one printed route owner.

    The central feed starts inside the state-owned stem/bridge, so both core
    carriers use the longer 8-mm cutter overshoot. Keeping this owner choice
    in one helper is essential:
    changing the extension changes the global ruled-section phase along the
    complete path, not only its endpoint caps.
    """
    return _extended_points(points, _owner_cutter_extension(owner))


def _um_owner_crop(shape, *, cutter=False):
    """Crop T material to the native UM owner and its open cutter mouth."""
    radius = UM_CORE_R + (
        UM_T_CUTTER_MOUTH_OVERSHOOT if cutter else 0.0)
    # build123d cylinders are Z-centered by default; z=0 spans -50..+50.
    cylinder = Pos(UM_CUTOUT[0], UM_CUTOUT[1], 0.0) * Cylinder(radius, 100.0)
    return shape & cylinder


def _lm_ring_outer_crop(shape, *, cutter=False):
    """LM-owned T material ending at the native R113 carrier boundary."""
    radius = LM_CORE_R + (CUTTER_SPLIT_OVERLAP if cutter else 0.0)
    owner = Pos(L22_CUTOUT[0], L22_CUTOUT[1], 0.0) * Cylinder(
        radius, 100.0)
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
        bounds = NO_FLOOR_BRIDGE_CORE_BOUNDS
        z0 = PAD_FACE_Z
    owner = _polygon_prism(box(*bounds), z0, 100.0)
    return shape & owner


def _lm_printed_owner_crop(shape, *, cutter=False):
    """Restrict one LM route solid to actual printed LM material.

    Positive covers stop at the native R113 ring except for the central feed
    span owned by the integral floor stem or no-floor bridge.  Negative
    cutters receive only the required owner-mouth overshoot and never enter
    the free-cable span toward UM.
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
    """Keep only the LM-owned main cover and its native R113 mouth."""
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
            # needs only its lumen before R113. Floor keeps the legacy R114
            # centerline but also ends printed material at the native R113
            # carrier boundary; the short lateral entry span is free cable.
            # Only component 0 is the continuous round tube. Every printed
            # main-route component ends at the native R113 LM boundary; the
            # remainder is intentionally free cable behind the UM.
            ring_component = _lm_ring_outer_crop(component)
            if index == 0:
                component = _main_printed_outer_from_full(component)
            else:
                component = ring_component
            if component is None:
                if index > 0:
                    continue
                raise RuntimeError("LM main cover has no R113 owner crop")
            solids = tuple(component.solids())
            if (not component.is_valid
                    or any(solid.volume <= 0.01 for solid in solids)):
                raise RuntimeError(
                    f"LM main native-R113 component {index} invalid; "
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
            component = _lm_ring_outer_crop(component)
            solids = tuple(component.solids()) if component is not None else ()
            if not solids:
                if index > 0:
                    continue
                raise RuntimeError("LM T round cover has no R113 owner crop")
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


def route_inner_cutter_group(which, index):
    """Build one independent nominal cutter group for bounded workers."""
    _require_guarded_build()
    if which not in ("lm", "um"):
        raise ValueError(which)
    count = route_inner_cutter_group_count(which)
    if not 0 <= index < count:
        raise ValueError((which, index))
    if which == "um" and index == 0:
        ts_points = ts_cable_points(1.8)
        ts = _round_tube(
            _owner_cutter_points(ts_points, "um"),
            TS_CUTTER_R)
        return _required_solids(
            _um_owner_crop(ts, cutter=True), f"{which} T cutter")
    if which == "lm" and index < LM_MAIN_CUTTER_SEGMENT_COUNT:
        main = _round_tube_global_segment(
            _owner_cutter_points(route_cable_points(1.8), "lm"),
            CUTTER_R,
            index, LM_MAIN_CUTTER_SEGMENT_COUNT)
        cropped = _lm_printed_owner_crop(main, cutter=True)
        if cropped is None:
            return ()
        solids = tuple(cropped.solids())
        if not cropped.is_valid or any(
                solid.volume <= 1e-9 for solid in solids):
            raise RuntimeError(
                f"LM main cutter segment {index} crop failed; "
                f"valid={cropped.is_valid} volumes="
                f"{[solid.volume for solid in solids]}")
        return solids
    t_group_start = LM_MAIN_CUTTER_SEGMENT_COUNT
    t_group_stop = t_group_start + LM_T_CUTTER_SEGMENT_COUNT
    if which == "lm" and t_group_start <= index < t_group_stop:
        # Each worker subtracts an exact consecutive subset of the globally
        # phased full T cutter. Their union is identical to the full loft,
        # but no worker must retain that face-rich BREP beside the LM carrier.
        ts = _round_tube_global_segment(
            _owner_cutter_points(ts_cable_points(1.8), "lm"),
            TS_CUTTER_R,
            index - t_group_start, LM_T_CUTTER_SEGMENT_COUNT)
        cropped = _lm_printed_owner_crop(ts, cutter=True)
        if cropped is None:
            return ()
        solids = tuple(cropped.solids())
        if not cropped.is_valid or any(
                solid.volume <= 1e-9 for solid in solids):
            raise RuntimeError(
                f"LM T cutter segment {index - t_group_start} crop failed; "
                f"valid={cropped.is_valid} volumes="
                f"{[solid.volume for solid in solids]}")
        return solids
    raise AssertionError((which, index))


def route_inner_cutter_group_count(which):
    if which == "lm":
        return LM_ROUTE_CUTTER_GROUP_COUNT
    if which == "um":
        return UM_ROUTE_CUTTER_GROUP_COUNT
    raise ValueError(which)


def route_inner_cutters(which):
    """Yield overshot nominal cutters serially for one carrier owner."""
    _require_guarded_build()
    for index in range(route_inner_cutter_group_count(which)):
        yield from route_inner_cutter_group(which, index)


def route_material_plan():
    """Analytic assembled XY design outline used by plan containment.

    This union is derived directly from the carrier and swept-cover design
    parameters rather than reconstructed from a clearance formula inside a
    test.  Its erosion/normal-distance result is a plan-intent check only;
    ``required_assembled_shell_components`` and the final-carrier Boolean
    tests are authoritative for the manufactured BREP. Insert bypasses remain
    in plan because their clearance is resolved in Z.
    """
    _require_guarded_build()
    lm_annulus = Point(*L22_CUTOUT[:2]).buffer(
        113.0, resolution=96).difference(
            Point(*L22_CUTOUT[:2]).buffer(
                L22_CUTOUT[2] / 2.0, resolution=96))
    um_annulus = Point(*UM_CUTOUT[:2]).buffer(
        UM_CORE_R, resolution=96).difference(
            Point(*UM_CUTOUT[:2]).buffer(
                UM_CUTOUT[2] / 2.0, resolution=96))
    lm_domain = Point(*L22_CUTOUT[:2]).buffer(LM_CORE_R, resolution=96)
    um_domain = Point(*UM_CUTOUT[:2]).buffer(UM_CORE_R, resolution=96)
    tail_domain = box(*(
        FLOOR_STEM_CORE_BOUNDS if STAND_FOOT
        else NO_FLOOR_BRIDGE_CORE_BOUNDS))
    main_domain = unary_union((lm_domain, tail_domain))
    t_domain = unary_union((lm_domain, um_domain, tail_domain))
    main_cover = LineString(_MAIN_PLAN).buffer(
        MAIN_OUTER_R, resolution=48, cap_style=1, join_style=1
    ).intersection(main_domain)
    # The T cover is printed only in the mandatory LM/UM core. Its short
    # suffix behind the tweeter crescent is a freely dressed cable.
    t_cover = LineString(_TS_PLAN).buffer(
        TS_OUTER_R, resolution=48, cap_style=1, join_style=1
    ).intersection(t_domain)
    bases = [lm_annulus, um_annulus, main_cover, t_cover]
    bases.append(tail_domain)
    return unary_union(bases)


def route_plan_containment_facts():
    """Eroded-outline and exact normal-wall facts for printed owner spans."""
    outline = route_material_plan()
    boundary = outline.boundary
    results = {}
    for name, points, inner_r, skin in (
            ("UM", route_cable_points(0.35), CUTTER_R, TUNNEL_SKIN),
            ("T", ts_cable_points(0.35), TS_CUTTER_R, TUNNEL_SKIN)):
        xy = np.asarray(points, dtype=float)[:, :2]
        if name == "UM":
            printed = np.linalg.norm(
                xy - np.asarray(L22_CUTOUT[:2]), axis=1) <= LM_CORE_R
        else:
            printed = (
                (np.linalg.norm(
                    xy - np.asarray(L22_CUTOUT[:2]), axis=1) <= LM_CORE_R)
                | (np.linalg.norm(
                    xy - np.asarray(UM_CUTOUT[:2]), axis=1) <= UM_CORE_R))
        if not STAND_FOOT:
            x0, y0, x1, y1 = NO_FLOOR_BRIDGE_CORE_BOUNDS
            printed |= ((xy[:, 0] >= x0) & (xy[:, 0] <= x1)
                        & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
        # Exclude five outer radii along each printed run at a functional
        # open mouth. A shallow circle crossing needs more path length than
        # one radius before the complete normal wall fits inside the owner;
        # exact final-BREP mouth tests independently police the excluded cap.
        trim = max(
            3, int(math.ceil(5.0 * (inner_r + skin) / 0.35)))
        runs = [run for run in _point_runs(xy, printed)
                if len(run) > 2 * trim + 1]
        interiors = [LineString(run[trim:-trim]) for run in runs]
        interior_xy = np.vstack([run[trim:-trim] for run in runs])
        eroded = outline.buffer(-(inner_r + skin - 0.03), resolution=48)
        distances = [Point(float(x), float(y)).distance(boundary)
                     for x, y in interior_xy]
        eroded_witness = eroded.buffer(0.04)
        outside = [
            (float(x), float(y)) for x, y in interior_xy
            if not eroded_witness.covers(Point(float(x), float(y)))]
        results[name] = {
            "contained": all(
                eroded_witness.covers(interior)
                for interior in interiors),
            "min_normal_wall_mm": float(min(distances) - inner_r),
            "uncontained_point_count": len(outside),
            "first_uncontained_xy": None if not outside else outside[0],
        }
    return results


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
    # Stream the three independent nominal voids. Holding all complete route
    # BREPs beside the outer shell exceeded the release memory floor even
    # though each individual Boolean is comfortably bounded.
    cutter_specs = [
        (_owner_cutter_points(route_cable_points(1.8), "lm"), CUTTER_R),
        (_extended_points(lm_cable_points(1.2)), CUTTER_R),
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
                box(*NO_FLOOR_BRIDGE_CORE_BOUNDS), PAD_FACE_Z, 100.0)
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
        (_extended_points(lm_cable_points(1.2)), CUTTER_R),
    ]
    if route_name == "UM":
        cutter_specs.append((
            _owner_cutter_points(ts_cable_points(1.8), "lm"),
            TS_CUTTER_R))
    for cutter_points, cutter_radius in cutter_specs:
        shell = subtract_bounded_cutter(
            shell, cutter_points, cutter_radius)

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


def route_facts():
    main_s, main = _main_xyz(0.20)
    ts_s, ts = _ts_xyz(0.20)
    lm_free = lm_cable_points(0.20)
    main_cross_z = float(np.interp(CROSSOVER_MAIN_S, main_s, main[:, 2]))
    ts_cross_z = float(np.interp(CROSSOVER_TS_S, ts_s, ts[:, 2]))
    void_web = ts_cross_z - main_cross_z - CUTTER_R - TS_CUTTER_R
    physical_gap = (ts_cross_z - main_cross_z
                    - CABLE_R_EST - TS_CABLE_D_EST / 2.0)
    free_um_to_t_cover_gap = (
        ts_cross_z - main_cross_z - CABLE_R_EST - TS_OUTER_R)
    lm_relief_lower_z = lm_free[:, 2] - LM_FREE_LEAD_RELIEF_R
    lm_relief_upper_z = lm_free[:, 2] + LM_FREE_LEAD_RELIEF_R
    lm_relief_floor_rear_open_margin = (
        STEM_Z_MM[0] - float(lm_relief_lower_z.max()))
    lm_relief_no_floor_rear_open_margin = (
        PAD_FACE_Z - float(lm_relief_lower_z.max()))
    lm_relief_seat_membrane_margin = (
        LM_SEAT_MEMBRANE_BOTTOM_Z - float(lm_relief_upper_z.max()))

    # Length of the short plan overlap where the two nominal voids are
    # closer than the sum of radii.  This is diagnostic; Z separation is
    # the authoritative clearance at those stations.
    ts_line = LineString(ts[:, :2])
    near = np.asarray([
        ts_line.distance(Point(float(x), float(y)))
        <= CUTTER_R + TS_CUTTER_R
        for x, y in main[:, :2]
    ])
    overlap_length = float(np.sum(np.diff(main_s)[near[:-1] & near[1:]]))

    def anchor_facts(points, outer_radius, **kwargs):
        points = np.asarray(points, dtype=float)
        keep = _anchor_keep_mask(points, outer_radius, **kwargs)
        anchor_base_z = kwargs.get(
            "anchor_base_z", LM_SEAT_MEMBRANE_BOTTOM_Z)
        anchor_top_z = anchor_base_z + TUNNEL_FUSE_OVERLAP
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        run_length = float(np.sum(lengths[keep[:-1] & keep[1:]]))
        heights = np.maximum(
            anchor_top_z - np.minimum(
                points[:, 2], anchor_base_z - 0.20), 0.0)
        average_heights = (heights[:-1] + heights[1:]) / 2.0
        added_volume = float(np.sum(
            lengths[keep[:-1] & keep[1:]]
            * average_heights[keep[:-1] & keep[1:]]
            * ANCHOR_LEG_W * 2.0))
        return {
            "run_length_mm": run_length,
            "route_fraction": run_length / max(
                float(np.sum(lengths)), 1e-9),
            "estimated_added_volume_mm3": added_volume,
        }

    def burial_facts(points, outer_radius, **kwargs):
        points = np.asarray(points, dtype=float)
        keep, allowed = _burial_web_masks(points, outer_radius, **kwargs)
        anchor_base_z = kwargs.get(
            "anchor_base_z", LM_SEAT_MEMBRANE_BOTTOM_Z)
        top_z = anchor_base_z + TUNNEL_FUSE_OVERLAP
        run_length = 0.0
        gross_volume = 0.0
        run_count = 0
        for run in _point_runs_with_boundary_overlap(points, keep, allowed):
            lengths = np.linalg.norm(np.diff(run, axis=0), axis=1)
            bottoms = run[:, 2] - BURIAL_WEB_TUBE_OVERLAP
            heights = np.maximum(top_z - bottoms, 0.0)
            run_length += float(np.sum(lengths))
            gross_volume += float(np.sum(
                lengths * (heights[:-1] + heights[1:]) / 2.0
                * 2.0 * (
                    outer_radius + BURIAL_WEB_LATERAL_OVERLAP)))
            run_count += 1
        return {
            "run_count": run_count,
            "run_length_mm": run_length,
            "full_width_mm": 2.0 * (
                outer_radius + BURIAL_WEB_LATERAL_OVERLAP),
            "tube_center_overlap_mm": BURIAL_WEB_TUBE_OVERLAP,
            "lateral_fusion_overlap_mm": BURIAL_WEB_LATERAL_OVERLAP,
            # This gross loft volume deliberately overstates final growth:
            # the overlapping round upper half and existing carrier/backfill
            # are not subtracted. It is therefore a conservative material-
            # budget increment until the pinned remote artifact is measured.
            "estimated_growth_upper_bound_mm3": gross_volume,
        }

    anchor_records = {}
    burial_records = {
        "UM": burial_facts(
            main, MAIN_OUTER_R, support_domains=("lm",)),
        "T": burial_facts(
            ts, TS_OUTER_R, omit_crossover=True,
            support_domains=("lm",)),
    }
    um_burial_records = {
        "T": burial_facts(
            ts, TS_OUTER_R,
            anchor_base_z=UM_SEAT_MEMBRANE_BOTTOM_Z,
            omit_crossover=True, support_domains=("um",)),
    }
    main_feed_keep, main_feed_allowed = _burial_web_masks(
        main, MAIN_OUTER_R, support_domains=("lm",))
    t_feed_keep, t_feed_allowed = _burial_web_masks(
        ts, TS_OUTER_R, omit_crossover=True, support_domains=("lm",))
    backfills = bump_backfill_specs()
    backfill_records = tuple({
        "name": spec.name,
        "route": spec.route_name,
        "owner": spec.owner,
        "route_xyz": spec.route_xyz,
        "pilot_xy": spec.pilot_xy,
        "bottom_z_mm": spec.bottom_z,
        "bore_floor_z_mm": spec.top_z,
        "filled_height_mm": spec.top_z - spec.bottom_z,
        "tube_overlap_mm": BUMP_BACKFILL_TUBE_OVERLAP,
    } for spec in backfills)

    plan = route_plan_containment_facts()
    return {
        "length_mm": ROUTE_LENGTH,
        "ts_length_mm": TS_ROUTE_LENGTH,
        "plan_containment": plan,
        "min_plan_normal_wall_mm": min(
            record["min_normal_wall_mm"] for record in plan.values()),
        "lm_roof_mm": LM_SEAT_Z - TRENCH_CENTER_Z - CUTTER_R,
        "bridge_side_wall_mm": TUNNEL_SKIN,
        "ts_lm_roof_mm": LM_SEAT_Z - TS_TRENCH_CENTER_Z - TS_CUTTER_R,
        "ts_um_roof_mm": UM_SEAT_Z - TS_UM_CENTER_Z - TS_CUTTER_R,
        "ts_bridge_side_wall_mm": TUNNEL_SKIN,
        "tunnel_floor_skin_mm": TUNNEL_SKIN,
        "tunnel_floor_fuse_overlap_mm": TUNNEL_FUSE_OVERLAP,
        "lm_seat_membrane_mm": TUNNEL_ROOF_SKIN,
        "um_seat_membrane_mm": TUNNEL_ROOF_SKIN,
        "open_bore_jump_count": 0,
        "covered_bump_names": tuple(
            bump.name
            for bump in MAIN_COVERED_BUMPS + T_COVERED_BUMPS),
        "solid_backfill_count": len(backfill_records),
        "solid_backfill_names": tuple(
            record["name"] for record in backfill_records),
        "solid_backfill_records": backfill_records,
        "solid_backfill_tube_overlap_mm": BUMP_BACKFILL_TUBE_OVERLAP,
        "solid_backfill_added_rear_depth_mm": 0.0,
        "solid_backfill_floor_hardware_exceptions": (),
        "lm_burial_webs": burial_records,
        "lm_burial_web_count": sum(
            record["run_count"] for record in burial_records.values()),
        "lm_burial_web_growth_upper_bound_mm3": sum(
            record["estimated_growth_upper_bound_mm3"]
            for record in burial_records.values()),
        "lm_burial_web_floor_hardware_clear_d_mm": None,
        "um_burial_webs": um_burial_records,
        "um_burial_web_count": sum(
            record["run_count"] for record in um_burial_records.values()),
        "um_burial_web_growth_upper_bound_mm3": sum(
            record["estimated_growth_upper_bound_mm3"]
            for record in um_burial_records.values()),
        "functional_lm_feed_count": 2,
        "functional_lm_feed_mode": (
            "integrated_stem_rear_face_shallow_rise" if STAND_FOOT
            else "bridge_rear_face_shallow_rise"),
        "functional_lm_feed_points": (
            tuple(map(float, main[0])), tuple(map(float, ts[0]))),
        "central_owner_feed_xy": (
            tuple(map(float, NO_FLOOR_MAIN_FEED_XY)),
            tuple(map(float, NO_FLOOR_T_FEED_XY))),
        "central_owner_feed_rear_z_mm": NO_FLOOR_FEED_REAR_Z,
        "central_owner_feed_rise_lengths_mm": (
            NO_FLOOR_MAIN_FEED_RISE_LENGTH,
            NO_FLOOR_T_FEED_RISE_LENGTH),
        "functional_lm_feed_web_omitted": bool(
            (not main_feed_keep[0] and not main_feed_allowed[0]
                and not t_feed_keep[0] and not t_feed_allowed[0])),
        "main_min_z_mm": float(main[:, 2].min()),
        "ts_min_z_mm": float(ts[:, 2].min()),
        "main_max_rear_protrusion_mm": max(
            0.0, CORE_REAR_Z - float((main[:, 2] - MAIN_OUTER_R).min())),
        "ts_max_rear_protrusion_mm": max(
            0.0, CORE_REAR_Z - float((ts[:, 2] - TS_OUTER_R).min())),
        "crossover_xy": tuple(map(float, CROSSOVER_XY)),
        "crossover_angle_deg": CROSSOVER_ANGLE_DEG,
        "main_bridge_start_handle_mm": MAIN_BRIDGE_START_HANDLE_MM,
        "main_bridge_end_handle_mm": MAIN_BRIDGE_END_HANDLE_MM,
        "t_bridge_start_handle_mm": TS_BRIDGE_START_HANDLE_MM,
        "t_bridge_end_handle_mm": TS_BRIDGE_END_HANDLE_MM,
        "crossover_main_z_mm": main_cross_z,
        "crossover_t_z_mm": ts_cross_z,
        "crossover_nominal_void_gap_mm": void_web,
        "crossover_free_um_to_t_cover_gap_mm": free_um_to_t_cover_gap,
        "crossover_physical_gap_mm": physical_gap,
        "crossover_plan_overlap_mm": overlap_length,
        "terminal_clock_deg": UM_ENTRY_ANGLE_DEG,
        "terminal_plan_bend_radius_mm": UM_TERMINAL_PLAN_BEND_R,
        "terminal_mouth_tangent": "clockwise_circumferential_body_clear",
        "um_terminal_reference_opening_radius_mm": UM_CUTOUT[2] / 2.0,
        "um_printed_owner": "lm_only",
        "um_carrier_main_duct": False,
        "um_terminal_lead_mode": "free_from_lm_r113_mouth_behind_um",
        "um_telescoping_handoff_count": 0,
        "t_handoff_mode": "lm_um_core_then_free_behind_tweeter",
        "t_lower_lm_flush_radius_mm": LM_CORE_R,
        "t_lower_um_flush_radius_mm": UM_CORE_R,
        "t_upper_um_flush_radius_mm": UM_CORE_R,
        "t_crescent_clear_radius_mm": TS_TWEETER_FLUSH_R,
        "t_tweeter_printed_duct": False,
        "t_free_cable_z_mm": TS_FREE_CABLE_Z,
        "t_free_cable_rear_clearance_mm": TS_FREE_CABLE_REAR_CLEARANCE,
        "t_telescoping_handoff_count": 0,
        "um_bump_z_mm": UM_TS_BUMP_Z,
        "um_pilot_bump_names": ("um_pilot_328", "um_pilot_58"),
        "anchor_legs": anchor_records,
        "printed_lm_tunnel_count": 0,
        "lm_lead_mode": "short_free_span_rear_open_relief_no_micro_duct",
        "lm_free_lead_relief_kind": "subtractive_rear_open_not_a_duct",
        "lm_free_lead_relief_radial_clearance_mm": (
            LM_FREE_LEAD_RELIEF_RADIAL_CLEAR_MM),
        "lm_free_lead_relief_radius_mm": LM_FREE_LEAD_RELIEF_R,
        "lm_free_lead_relief_floor_rear_open_margin_mm": (
            lm_relief_floor_rear_open_margin),
        "lm_free_lead_relief_no_floor_rear_open_margin_mm": (
            lm_relief_no_floor_rear_open_margin),
        "lm_free_lead_relief_seat_membrane_margin_mm": (
            lm_relief_seat_membrane_margin),
        "lm_free_lead_relief_rear_open_both_states": bool(
            lm_relief_floor_rear_open_margin > 0.0
            and lm_relief_no_floor_rear_open_margin > 0.0),
        "lm_free_cable_outer_z_mm": LM_FREE_CABLE_OUTER_Z,
        "lm_free_cable_inner_z_mm": LM_FREE_CABLE_INNER_Z,
        "lm_free_cable_rear_clearance_mm": LM_FREE_CABLE_REAR_CLEARANCE,
    }
