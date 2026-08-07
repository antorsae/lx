"""Final Obi-Wan R6F Z-first core tunnels and gradual cable ports.

The UM and tweeter voids are cut from continuous swept outer-cover solids.
Every insert bypass is a smooth local Z excursion inside that closed cover;
full-width longitudinal webs and local roof-to-bore saddles leave no trapped
shoulder cavities.  The no-floor bridge has three explicit rear-normal entry
bores packed wholly inside its one D20 support opening: LM above, T lower-left
and UM lower-right.  Each bore overlaps its buried tunnel by positive volume
rather than merely touching it at the rear plane.  The LM lead exits through
one continuous R14/D9 handoff at the shared rear-face outlet; its external
span follows the face tangent and is never allowed to sweep sideways through
the visible lower ring.  The LM-owned UM and T lumens run at the outside of
the structural ring, with their complete envelopes buried by a 0.05-mm solid
safety land beneath the continuous R113.8 exterior, so neither route carves a
tell-tale groove.
The tweeter route crosses the physical UM cable once near the LM crown,
near orthogonally, with T above UM in +Z and explicit cable/cover clearance.
"""

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


def _require_guarded_build():
    """Fail closed before any public route factory starts OCC work."""
    import run_memory_guarded as memory_guard
    memory_guard.require_guarded_build(
        "Obi-Wan route geometry must run through run_memory_guarded.py")

# Physical and nominal diameters from the design contract.
DUCT_D = 8.2
DUCT_R = DUCT_D / 2.0
CUTTER_R = DUCT_R
CUTTER_SIDES = 32
CABLE_D_EST = 7.0
CABLE_R_EST = CABLE_D_EST / 2.0
LM_CABLE_D_EST = 7.8
# The lower LM pair is D7.8 estimated cable inside the established D9 outlet.
# Its R14 centerline stays wholly below the acoustic aperture and intersects
# only the rear face, so it cannot recreate the former visible-ring bite.
LM_REAR_PORT_D_MM = LM_EXIT_D_MM
LM_REAR_PORT_R = LM_REAR_PORT_D_MM / 2.0
LM_REAR_PORT_REAR_OVERTRAVEL_MM = 0.20
LM_REAR_PORT_SEAT_CLEAR_MM = 0.20
# Floor mode shares the established z=12.55 R14 lane.  No-floor mode begins
# the same R14 handoff at z=13.0, where its shallow bridge retains the full
# 0.80-mm front skin.  Both reach the exact rear-face mouth and continue
# outside along the face tangent.
LM_REAR_PORT_INNER_DEPTH_MM = 1.80
LM_EXTERNAL_LEAD_LENGTH_MM = 18.0

# The no-floor piece carries the same LM cable bundle as Stock/Slim from the
# exact Stock/Slim rear-entry datum to Obi-Wan's established lower-ring exit.
# D9 provides 0.60 mm radial clearance around the estimated D7.8 bundle.  Its
# z=13.0 center leaves a complete 0.80-mm front skin in the shallow 5.3..18.3
# bridge while retaining 3.20 mm of material behind the horizontal lumen.
LM_INTERNAL_DUCT_D_MM = 9.0
LM_INTERNAL_DUCT_R = LM_INTERNAL_DUCT_D_MM / 2.0
LM_INTERNAL_CENTER_Z_MM = 13.0
LM_INTERNAL_FRONT_SKIN_MM = 0.80
LM_INTERNAL_REAR_SKIN_MM = (
    LM_INTERNAL_CENTER_Z_MM - LM_INTERNAL_DUCT_R - PAD_FACE_Z)
LM_INTERNAL_PORT_OVERLAP_MM = 0.80
LM_INTERNAL_JUNCTION_OVERTRAVEL_MM = 2.00
# The independently applied rear-port cutter reaches back along the R14
# arc's exact horizontal entry tangent.  This ordinary same-axis overlap
# prevents OCC from leaving a coincident end-cap seam where the buried lane
# meets the gradual outlet; it does not add a corner or move either datum.
LM_REAR_PORT_PREFUSION_MM = 2.00
LM_EXIT_TUBE_SECTION_SPACING_MM = 2.0

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
# Burial-web lofts used to extend a uniform 0.55 mm past the conduit on both
# sides and rely on a later radial Boolean to trim them back to the carrier.
# Once the LM-owned UM/T passages moved out to the R113/R113.8 shell, that
# operation left 0.002..0.03-mm-wide B-spline sliver faces at oblique route
# stations.  Those faces are valid BREP topology but cannot be triangulated
# reliably.  Keep every generated web section analytically inside its native
# printed owner instead.  The 0.05-mm inset is far below one extrusion width,
# while the existing carrier material still supplies the complete exterior
# skin; it is a CAD robustness allowance, not an exterior groove or air gap.
BURIAL_WEB_OWNER_INSET = 0.05
LM_MAIN_CUTTER_SEGMENT_COUNT = 8
LM_T_CUTTER_SEGMENT_COUNT = 12
LM_INTERNAL_CUTTER_GROUP_COUNT = 1
LM_ROUTE_CUTTER_GROUP_COUNT = (
    LM_MAIN_CUTTER_SEGMENT_COUNT + LM_T_CUTTER_SEGMENT_COUNT
    + LM_INTERNAL_CUTTER_GROUP_COUNT)
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
# Preserve the already-qualified floor-state rise.  Only the no-floor 01a
# route needs the longer, gentler rise after its lowered LM-recess crossing.
TS_UM_Z_TRANSITION_LENGTH_MM = 24.0 if STAND_FOOT else 44.0
# The ruled production lumen is an octagon circumscribed around the requested
# D8.2 circle.  Only no-floor 01a exposes the LM flange recess directly above
# this route, so lower that state's nominal UM lane by the exact corner delta;
# floor Obi-Wan retains its independently qualified circular-radius datum.
NO_FLOOR_RECESS_SKIN_BOOLEAN_MARGIN_MM = 0.05
MAIN_TRENCH_CENTER_Z = (
    TRENCH_CENTER_Z if STAND_FOOT else
    LM_SEAT_Z
    - CUTTER_R / math.cos(math.pi / TUBE_SECTION_SIDES)
    - TUNNEL_ROOF_SKIN
    - NO_FLOOR_RECESS_SKIN_BOOLEAN_MARGIN_MM)
T_LM_TRENCH_CENTER_Z = (
    TS_TRENCH_CENTER_Z if STAND_FOOT else
    LM_SEAT_Z
    - TS_CUTTER_R / math.cos(math.pi / TUBE_SECTION_SIDES)
    - TUNNEL_ROOF_SKIN
    - NO_FLOOR_RECESS_SKIN_BOOLEAN_MARGIN_MM)
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

# The functional lumens run at the outside of the structural LM ring.  A
# literal tangent at R113/R113.8 made valid CAD but left independently meshed
# Boolean faces with nonconformal seams.  Bury both complete conduit envelopes
# by 0.05 mm instead: the lumen ends at R112.95, its positive cover at R113.75,
# and the native carrier continues unbroken to visible R113.8.  Thus the
# user-visible result has no groove and gains a conservative 0.85-mm exterior
# skin; the 0.05-mm land is solid carrier material, not an air gap.
LM_VISIBLE_RING_R = LM_CORE_R + TUNNEL_SKIN
LM_ROUTE_OWNER_CLEARANCE = 0.05
MAIN_LM_ROUTE_R = LM_CORE_R - CUTTER_R - LM_ROUTE_OWNER_CLEARANCE
TS_LM_ROUTE_R = LM_CORE_R - TS_CUTTER_R - LM_ROUTE_OWNER_CLEARANCE
LM_ROUTE_START_DEG = 300.0
LM_ROUTE_ARC_START_DEG = 315.0
LM_ROUTE_END_DEG = 58.0

# The no-floor bridge accepts LM/T/UM through one D20 support opening.  The
# floor stem retains its own internal handoff locations because cables enter
# it from the NL8 service cavity rather than the bridge rear face.  Every later
# path derives from the selected state datum, so no generated STL can silently
# mix the two systems.
NO_FLOOR_MAIN_FEED_XY = np.asarray(OBIWAN_NO_FLOOR_UM_ENTRY_XY, dtype=float)
NO_FLOOR_T_FEED_XY = np.asarray(OBIWAN_NO_FLOOR_T_ENTRY_XY, dtype=float)
NO_FLOOR_LM_FEED_XY = np.asarray(OBIWAN_NO_FLOOR_LM_ENTRY_XY, dtype=float)
FLOOR_MAIN_FEED_XY = np.asarray((8.0, 82.0), dtype=float)
FLOOR_T_FEED_XY = np.asarray(FLOOR_T_ROUTE_FEED_XY, dtype=float)
CENTRAL_MAIN_FEED_XY = (
    FLOOR_MAIN_FEED_XY if STAND_FOOT else NO_FLOOR_MAIN_FEED_XY)
CENTRAL_T_FEED_XY = (
    FLOOR_T_FEED_XY if STAND_FOOT else NO_FLOOR_T_FEED_XY)
NO_FLOOR_FEED_REAR_Z = PAD_FACE_Z
# A centerline tangent to the rear plane can appear as a closed or slivered
# mouth after slicing.  These short Z-axis bores establish full circular
# external entries, then overlap the first portion of each buried route.
NO_FLOOR_ENTRY_BORE_REAR_OVERTRAVEL_MM = 0.25
NO_FLOOR_ENTRY_BORE_DEPTH_MM = 5.00
NO_FLOOR_LM_ENTRY_RELIEF_RADIAL_MM = 0.005
NO_FLOOR_LM_ENTRY_RELIEF_REAR_SKIN_MM = 0.10
NO_FLOOR_T_ENTRY_RELIEF_RADIAL_MM = 0.005
NO_FLOOR_T_ENTRY_RELIEF_REAR_SKIN_MM = 0.10
# The long swept lumens use circumscribed octagonal sections.  A nominal
# same-radius sphere opens the full round throat but cannot quite contain an
# octagon corner behind the sweep's first section.  Extend only UM's phased
# octagonal cutter by 0.60 mm each way, inset 0.01 mm radially so its exact
# distance to the buried D9 LM entry remains 0.805 mm.  T's smaller probe is
# already completely open after its nominal sphere.
NO_FLOOR_UM_ENTRY_CAP_RELIEF_HALF_LENGTH_MM = 0.60
NO_FLOOR_UM_ENTRY_CAP_RELIEF_RADIAL_INSET_MM = 0.010
NO_FLOOR_ENTRY_VESTIBULE_REAR_SKIN_MM = TUNNEL_SKIN
NO_FLOOR_LM_ENTRY_BORE_INNER_Z_MM = (
    LM_INTERNAL_CENTER_Z_MM + LM_INTERNAL_PORT_OVERLAP_MM)
# The complete *outer* conduits, not only their nominal lumens, must sit at or
# above the planar z=5.3 rear face.  This leaves the ordinary 0.80-mm solid
# wall over each void while making the bridge face exactly coplanar everywhere
# except the three intentional circular entry bores.
NO_FLOOR_MAIN_FEED_START_Z = PAD_FACE_Z + MAIN_OUTER_R
NO_FLOOR_T_FEED_START_Z = PAD_FACE_Z + TS_OUTER_R
NO_FLOOR_MAIN_FEED_RISE_LENGTH = 24.0
NO_FLOOR_T_FEED_RISE_LENGTH = 27.5
FLOOR_T_FEED_RISE_LENGTH = 45.0
CENTRAL_T_FEED_RISE_LENGTH = (
    FLOOR_T_FEED_RISE_LENGTH if STAND_FOOT
    else NO_FLOOR_T_FEED_RISE_LENGTH)
# The no-floor entries now have explicit rear-normal bores with positive
# buried overlap. Extending either swept cutter behind its own entry is both
# redundant and unsafe.  Floor UM/T still need to reach 0.8 mm behind their
# fixed feed datums to meet the prefused lane cutters.  A 2.0-mm backreach
# leaves 1.2 mm of positive axial overlap while keeping the complete cutter
# inside Option B's vertical-tangent owner; the historical 8-mm backreach
# reached into the convex transition and detached the T handoff shell.
NO_FLOOR_FEED_CUTTER_EXTENSION = 0.0
FLOOR_FEED_CUTTER_EXTENSION = 2.0
STANDARD_CUTTER_EXTENSION = 1.5
# Dependency-neutral copy of the bridge's immutable insert-bearing core.
# bridge.py imports this route module, so the equality is bound by tests
# rather than a circular source import.
NO_FLOOR_BRIDGE_CORE_BOUNDS = (-31.0, 14.0, 31.0, 90.25)
# The visible rear service patch is the four-insert datum rectangle expanded
# by the requested 6 mm in XY.  A route center remains on its fully buried Z
# layer until its complete outer conduit has cleared this rectangle *and* its
# centerline has entered the closed LM-ring owner.  Releasing in the short
# bridge-to-ring overlap leaves the tail crop at z=5.3 while the lumen has
# already descended below that plane, producing a real rear-open slot.  One
# skin-width of travel inside the ring makes the ownership handoff robust
# before a smooth suffix returns to the established ring layer.
NO_FLOOR_SERVICE_PATCH_MARGIN_MM = 6.0
NO_FLOOR_SERVICE_PATCH_BOUNDS = (
    min(float(x) for x, _y in BRIDGE_HOLE_XY)
    - NO_FLOOR_SERVICE_PATCH_MARGIN_MM,
    min(float(y) for _x, y in BRIDGE_HOLE_XY)
    - NO_FLOOR_SERVICE_PATCH_MARGIN_MM,
    max(float(x) for x, _y in BRIDGE_HOLE_XY)
    + NO_FLOOR_SERVICE_PATCH_MARGIN_MM,
    max(float(y) for _x, y in BRIDGE_HOLE_XY)
    + NO_FLOOR_SERVICE_PATCH_MARGIN_MM,
)
NO_FLOOR_RING_ENTRY_BURIAL_OVERLAP_MM = TUNNEL_SKIN
NO_FLOOR_SERVICE_PATCH_RELEASE_MODE = (
    "hold_through_flat_service_patch_then_descend_behind_lm_recess")
# The no-floor UM feed cannot remain on its z=10.2 service-patch layer as it
# enters the R110.6 flange recess: the real circumscribed-octagon D8.2 cutter
# would open a window through that recessed front surface.  Once the complete
# R4.9 outer cover has cleared the specified four-insert rectangle +6 mm, a
# half-cosine descent reaches this center Z at the R110.6 plan crossing.  The
# resulting three-dimensional centerline remains above R14 while retaining a
# conservative 0.85-mm wall to the flange recess.  Its closed positive cover
# is allowed to form the existing hidden rear conduit belly outside the flat
# service patch; it is never an open raceway.
NO_FLOOR_MAIN_RECESS_CLEARANCE_Z = 6.70
# The one-piece bridge has a real soft-blend shoulder up to the LM lower
# tangent.  The stock/slim-aligned entries use that upper material before
# they enter the native R113 annulus, so route shells must retain it instead
# of being artificially truncated at the rectangular insert-bearing core.
NO_FLOOR_BRIDGE_ROUTE_BOUNDS = (
    -STEM_SHOULDER_HALF_WIDTH_MM, 14.0,
    STEM_SHOULDER_HALF_WIDTH_MM, STEM_TOP_Y_MM)
FLOOR_STEM_CORE_BOUNDS = (
    -STEM_HALF_WIDTH_MM, 0.0,
    STEM_HALF_WIDTH_MM, STEM_TOP_Y_MM)
# The three tightly packed D20 mouths diverge through the solid bridge, never
# through exterior raceways.  T passes on the inboard/right side of the
# upper-left bridge insert and UM on the inboard/left side of the upper-right
# insert, matching the requested blue/red/green topology.  Each two-cubic fan
# then reaches its unchanged radially flush LM-ring arc with a G1 tangent.
# Rounded, audited control datums retain at least R14 for both feeds, keep at
# least 0.80 mm between each complete route cover and the D6.4 insert
# envelopes, and preserve the requested three-port D20 entry packing.
NO_FLOOR_MAIN_ENTRY_JOIN_XY = (12.9, 78.0)
NO_FLOOR_MAIN_ENTRY_START_BEARING_DEG = 34.75
NO_FLOOR_MAIN_ENTRY_JOIN_BEARING_DEG = 70.0
NO_FLOOR_MAIN_ENTRY_START_HANDLE_MM = 10.0
NO_FLOOR_MAIN_ENTRY_JOIN_IN_HANDLE_MM = 10.0
NO_FLOOR_MAIN_ENTRY_JOIN_OUT_HANDLE_MM = 20.0
NO_FLOOR_MAIN_ENTRY_END_HANDLE_MM = 15.0

NO_FLOOR_T_ENTRY_JOIN_XY = (-12.0, 74.0)
NO_FLOOR_T_ENTRY_START_BEARING_DEG = 126.0
NO_FLOOR_T_ENTRY_JOIN_BEARING_DEG = 122.0
NO_FLOOR_T_ENTRY_START_HANDLE_MM = 6.5
NO_FLOOR_T_ENTRY_JOIN_IN_HANDLE_MM = 6.5
NO_FLOOR_T_ENTRY_JOIN_OUT_HANDLE_MM = 23.0
NO_FLOOR_T_ENTRY_END_HANDLE_MM = 13.3

# Rotate only no-floor Obi-Wan's R14 LM bend plane.  The exact common rear
# mouth stays fixed, but the buried tangent start moves right/up so the blue T
# feed has a useful internal corridor.  The red D9 cubic remains G1 at both
# ends and has a sampled R19.6 minimum; Stock, Slim and floor Obi-Wan retain
# the shared historical +Y (90-degree) handoff plane.
NO_FLOOR_LM_EXIT_PLAN_BEARING_DEG = 120.0
# Leave the immutable D20 mouth normal to its upper lobe before turning toward
# the fixed outlet.  The former 105-degree immediate left sweep approached T
# within 0.757 mm at y~=64 after T's prefix was raised to seal its rear skin.
# A +Y tangent restores the exact ruled-sweep wall without moving either port,
# changing a diameter, or altering the R14 outlet tangent.
NO_FLOOR_LM_ENTRY_START_BEARING_DEG = 90.0
# The bridge-to-R14 span is short.  After the recess-safe outlet moved 0.7 mm
# lower, retaining the old 8-mm outlet handle over-controlled the cubic and
# reduced its bend to R12.1.  A 4/6-mm handle pair is monotone and restores a
# sampled R18+ turn without moving either D20 entry or D9 outlet datum.
NO_FLOOR_LM_ENTRY_START_HANDLE_MM = 4.0
NO_FLOOR_LM_ENTRY_END_BEARING_DEG = NO_FLOOR_LM_EXIT_PLAN_BEARING_DEG
NO_FLOOR_LM_ENTRY_END_HANDLE_MM = 6.0

# Floor cables enter from the service cavity rather than the bridge, retaining
# the already-qualified centered handoff geometry.
# Floor UM and T enter from different service-side lanes. Their handles are
# independently tuned so both keep a >=R14 continuous path after the T lane
# moved left to clear the direct LM continuation.
FLOOR_MAIN_FEED_START_HANDLE = 32.0
FLOOR_T_FEED_START_HANDLE = 20.0
FLOOR_T_FEED_END_HANDLE = 22.5
FLOOR_MAIN_FEED_START_BEARING_DEG = 65.0
FLOOR_T_FEED_START_BEARING_DEG = FLOOR_T_ROUTE_FEED_BEARING_DEG
CENTRAL_MAIN_FEED_START_HANDLE = (
    FLOOR_MAIN_FEED_START_HANDLE if STAND_FOOT
    else NO_FLOOR_MAIN_ENTRY_START_HANDLE_MM)
CENTRAL_T_FEED_START_HANDLE = (
    FLOOR_T_FEED_START_HANDLE if STAND_FOOT
    else NO_FLOOR_T_ENTRY_START_HANDLE_MM)
CENTRAL_MAIN_FEED_START_BEARING_DEG = (
    FLOOR_MAIN_FEED_START_BEARING_DEG if STAND_FOOT
    else NO_FLOOR_MAIN_ENTRY_START_BEARING_DEG)
CENTRAL_T_FEED_START_BEARING_DEG = (
    FLOOR_T_FEED_START_BEARING_DEG if STAND_FOOT
    else NO_FLOOR_T_ENTRY_START_BEARING_DEG)
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
TS_LM_ARC_START_DEG = TS_LM_ROUTE_START_DEG
TS_LM_ROUTE_END_DEG = 120.0
TS_UM_ROUTE_START_DEG = 315.0
TS_UM_ROUTE_END_DEG = 60.0
TS_UM_ROUTE_R = 46.35
# Tangent-handle lengths for the shortest practical crown bridge that keeps
# the fixed arc endpoints, remains comfortably above the R14 bend floor and
# crosses the UM route essentially orthogonally.
# The radially flush R110 lower arc would otherwise send its old long tangent
# through the left LM--UM M3 ear.  Turn inward sooner, then take a longer
# crown-side handle: the bridge clears both complete ears by >4 mm, retains a
# >R26 plan bend and crosses UM at about 83 degrees.
TS_BRIDGE_START_HANDLE_MM = 30.0
TS_BRIDGE_END_HANDLE_MM = 80.0

# The no-floor lower piece needs more recess-skin room than the deeper floor
# stem.  The earlier 17.1-mm datum protected only the planar z=18.3 face; the
# real circumscribed-octagon D9/R14 cutter still grazed the R110.6 flange
# recess and exposed a small window.  Moving the same X=-10.5 mouth just
# 0.7 mm lower retains D9 and R14 while leaving >0.8 mm to both the planar
# front and recessed flange surfaces.  Floor mode intentionally keeps the
# common Stock/Slim 10-mm outlet.
NO_FLOOR_LM_DUCT_OUT_CLEARANCE_FROM_APERTURE_MM = 17.8
NO_FLOOR_LM_DUCT_OUT_Y_MM = (
    L22_CUTOUT[1] - L22_CUTOUT[2] / 2.0
    - NO_FLOOR_LM_DUCT_OUT_CLEARANCE_FROM_APERTURE_MM)
LM_REAR_PORT_CLEARANCE_FROM_APERTURE_MM = (
    LM_DUCT_OUT_CLEARANCE_FROM_APERTURE_MM if STAND_FOOT
    else NO_FLOOR_LM_DUCT_OUT_CLEARANCE_FROM_APERTURE_MM)
LM_REAR_PORT_XY = np.asarray(
    (
        LM_DUCT_OUT_X_MM,
        LM_DUCT_OUT_Y_MM if STAND_FOOT else NO_FLOOR_LM_DUCT_OUT_Y_MM,
    ),
    dtype=float,
)
LM_REAR_PORT_REAR_Z = STEM_Z_MM[0] if STAND_FOOT else PAD_FACE_Z
LM_REAR_HANDOFF_CENTER_Z = (
    FLOOR_LANE_SPECS["lm"]["stem_z_mm"]
    if STAND_FOOT else LM_INTERNAL_CENTER_Z_MM)
LM_REAR_HANDOFF_PLAN_BEARING_DEG = (
    90.0 if STAND_FOOT else NO_FLOOR_LM_EXIT_PLAN_BEARING_DEG)
LM_REAR_HANDOFF_SPEC = lm_exit_handoff_spec(
    LM_REAR_HANDOFF_CENTER_Z,
    LM_REAR_PORT_REAR_Z,
    LM_DUCT_OUT_REAR_Z_MM,
    LM_EXIT_BEND_R_MM,
    LM_REAR_HANDOFF_PLAN_BEARING_DEG,
    face_xy_mm=LM_REAR_PORT_XY,
)
# Compatibility/reporting name: the former cylinder's inner endpoint is now
# the tangent start of the gradual handoff.
LM_REAR_PORT_INNER_Z = LM_REAR_HANDOFF_SPEC["start"][2]
LM_EXTERNAL_LEAD_END = tuple(
    LM_REAR_HANDOFF_SPEC["face"][index]
    + LM_EXTERNAL_LEAD_LENGTH_MM
    * LM_REAR_HANDOFF_SPEC["face_tangent"][index]
    for index in range(3))
LM_EXTERNAL_LEAD_END_Z = LM_EXTERNAL_LEAD_END[2]


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


def _bearing_unit(bearing_deg):
    angle = math.radians(bearing_deg)
    return np.asarray((math.cos(angle), math.sin(angle)), dtype=float)


def _two_cubic_fan(
        start, join, end, *, start_bearing_deg, join_bearing_deg,
        end_bearing_deg, start_handle_mm, join_in_handle_mm,
        join_out_handle_mm, end_handle_mm):
    """Two G1 cubic spans used by one insert-clearing D20 entry fan."""
    start = np.asarray(start, dtype=float)
    join = np.asarray(join, dtype=float)
    end = np.asarray(end, dtype=float)
    start_tangent = _bearing_unit(start_bearing_deg)
    join_tangent = _bearing_unit(join_bearing_deg)
    end_tangent = _bearing_unit(end_bearing_deg)
    first = _cubic(
        np.linspace(0.0, 1.0, 1201),
        start,
        start + start_handle_mm * start_tangent,
        join - join_in_handle_mm * join_tangent,
        join,
    )
    second = _cubic(
        np.linspace(0.0, 1.0, 1601),
        join,
        join + join_out_handle_mm * join_tangent,
        end - end_handle_mm * end_tangent,
        end,
    )
    return _join(first, second)


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


def _sampled_centerline_surface_wall(
        first, first_radius: float, second, second_radius: float) -> float:
    """Minimum sampled 3-D wall between two round cable lumens.

    The packed no-floor feeds intentionally use different Z lanes after their
    common rear-entry plane.  A plan-only projection therefore reports a
    false collision even when the actual circular lumens retain a solid wall.
    Route facts use the already dense 0.20-mm production samples; final BREP
    tests independently measure the swept solids themselves.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if (first.ndim != 2 or second.ndim != 2
            or first.shape[1] != 3 or second.shape[1] != 3
            or not len(first) or not len(second)):
        raise ValueError("centerline wall inputs must be non-empty Nx3 arrays")
    best_squared = math.inf
    # Bound peak memory while retaining vectorized Euclidean distances.
    for start in range(0, len(first), 256):
        delta = first[start:start + 256, None, :] - second[None, :, :]
        best_squared = min(
            best_squared,
            float(np.min(np.einsum("ijk,ijk->ij", delta, delta))))
    return (
        math.sqrt(best_squared) - float(first_radius) - float(second_radius))


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


def _cosine01(x):
    """Zero-slope easing with less peak curvature than minimum-jerk."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * x)


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
    L22_CUTOUT[:2], MAIN_LM_ROUTE_R, LM_ROUTE_ARC_START_DEG)
if STAND_FOOT:
    _MAIN_FEED_START_DIRECTION = _bearing_unit(
        CENTRAL_MAIN_FEED_START_BEARING_DEG)
    _MAIN_ENTRY = _cubic(
        np.linspace(0.0, 1.0, 801),
        _MAIN_FEED,
        (_MAIN_FEED + _MAIN_FEED_START_DIRECTION
         * CENTRAL_MAIN_FEED_START_HANDLE),
        (_MAIN_ARC_START - _bearing_unit(45.0)
         * NO_FLOOR_FEED_END_HANDLE),
        _MAIN_ARC_START,
    )
else:
    _MAIN_ENTRY = _two_cubic_fan(
        _MAIN_FEED,
        NO_FLOOR_MAIN_ENTRY_JOIN_XY,
        _MAIN_ARC_START,
        start_bearing_deg=NO_FLOOR_MAIN_ENTRY_START_BEARING_DEG,
        join_bearing_deg=NO_FLOOR_MAIN_ENTRY_JOIN_BEARING_DEG,
        end_bearing_deg=45.0,
        start_handle_mm=NO_FLOOR_MAIN_ENTRY_START_HANDLE_MM,
        join_in_handle_mm=NO_FLOOR_MAIN_ENTRY_JOIN_IN_HANDLE_MM,
        join_out_handle_mm=NO_FLOOR_MAIN_ENTRY_JOIN_OUT_HANDLE_MM,
        end_handle_mm=NO_FLOOR_MAIN_ENTRY_END_HANDLE_MM,
    )
_MAIN_ARC = _arc(
    L22_CUTOUT[:2], MAIN_LM_ROUTE_R,
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
    L22_CUTOUT[:2], TS_LM_ROUTE_R, TS_LM_ARC_START_DEG)
_TS_FEED_START_DIRECTION = _bearing_unit(
    CENTRAL_T_FEED_START_BEARING_DEG)
if STAND_FOOT:
    _TS_ENTRY = _cubic(
        np.linspace(0.0, 1.0, 801),
        _TS_FEED,
        _TS_FEED + _TS_FEED_START_DIRECTION * CENTRAL_T_FEED_START_HANDLE,
        _TS_LM_ARC_START
        - _bearing_unit(TS_LM_ARC_START_DEG - 90.0)
        * FLOOR_T_FEED_END_HANDLE,
        _TS_LM_ARC_START,
    )
else:
    _TS_ENTRY = _two_cubic_fan(
        _TS_FEED,
        NO_FLOOR_T_ENTRY_JOIN_XY,
        _TS_LM_ARC_START,
        start_bearing_deg=NO_FLOOR_T_ENTRY_START_BEARING_DEG,
        join_bearing_deg=NO_FLOOR_T_ENTRY_JOIN_BEARING_DEG,
        end_bearing_deg=TS_LM_ARC_START_DEG - 90.0,
        start_handle_mm=NO_FLOOR_T_ENTRY_START_HANDLE_MM,
        join_in_handle_mm=NO_FLOOR_T_ENTRY_JOIN_IN_HANDLE_MM,
        join_out_handle_mm=NO_FLOOR_T_ENTRY_JOIN_OUT_HANDLE_MM,
        end_handle_mm=NO_FLOOR_T_ENTRY_END_HANDLE_MM,
    )
_TS_LM_ARC = _arc(
    L22_CUTOUT[:2], TS_LM_ROUTE_R,
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


# -- no-floor LM path ------------------------------------------------
# Both endpoints are immutable project datums.  A shallow tangent cubic keeps
# the D9 lumen centered in the solid bridge and avoids a polygonal kink at
# either rear-normal access bore.  Floor mode already owns a different buried
# lane in the integral stem; no duplicate internal tunnel is created there.
_LM_INTERNAL_ENTRY_XY = NO_FLOOR_LM_FEED_XY.copy()
_LM_INTERNAL_EXIT_XY = np.asarray(
    LM_REAR_HANDOFF_SPEC["start"][:2], dtype=float)
_LM_INTERNAL_PLAN = _cubic(
    np.linspace(0.0, 1.0, 801),
    _LM_INTERNAL_ENTRY_XY,
    (_LM_INTERNAL_ENTRY_XY
     + NO_FLOOR_LM_ENTRY_START_HANDLE_MM
     * _bearing_unit(NO_FLOOR_LM_ENTRY_START_BEARING_DEG)),
    (_LM_INTERNAL_EXIT_XY
     - NO_FLOOR_LM_ENTRY_END_HANDLE_MM
     * _bearing_unit(NO_FLOOR_LM_ENTRY_END_BEARING_DEG)),
    _LM_INTERNAL_EXIT_XY,
)
_LM_INTERNAL_PLAN_S = _plan_lengths(_LM_INTERNAL_PLAN)
LM_INTERNAL_PLAN_LENGTH_MM = float(_LM_INTERNAL_PLAN_S[-1])
LM_INTERNAL_ROUTE_LENGTH_MM = (
    LM_INTERNAL_PLAN_LENGTH_MM
    + LM_EXIT_BEND_R_MM * LM_REAR_HANDOFF_SPEC["face_angle_rad"]
    + LM_REAR_HANDOFF_SPEC["external_length_to_rear_end_mm"]
)


def _floor_t_handoff_owner_plan():
    """Narrow bridge from the left floor feed into the LM-ring owner.

    The floor T datum at (-26, 82) initially turns left while it rises into
    the LM ring.  That intentional bend crosses x=-32 before it reaches the
    R113 owner circle, so the rectangular stem crop and the ring crop alone
    leave a real, uncropped 4.37-mm-wide slice of lumen.  Own only that
    local route-following corridor; it deliberately does not enlarge the
    physical floor stem or its exterior silhouette.
    """
    if not STAND_FOOT:
        return None
    radial = np.linalg.norm(
        _TS_PLAN - np.asarray(L22_CUTOUT[:2]), axis=1)
    entering_ring = np.flatnonzero(radial <= LM_CORE_R)
    if not len(entering_ring):
        raise RuntimeError("floor T handoff never reaches the LM owner ring")
    end = int(entering_ring[0])
    if end < 2:
        raise RuntimeError("floor T handoff owner corridor is degenerate")
    return LineString(_TS_PLAN[:end + 1]).buffer(
        TS_OUTER_R + CUTTER_SPLIT_OVERLAP + 0.05,
        resolution=32, cap_style=1, join_style=1)

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






_LM_PILOT_BY_ANGLE = dict(zip(
    (0.0, 60.0, 120.0, 180.0, 240.0, 300.0), LM_PILOT_XY))
_UM_PILOT_BY_ANGLE = dict(zip((58.0, 148.0, 238.0, 328.0), UM_PILOT_XY))


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








def _main_xyz(spacing_mm):
    stations, xy = _resample(_MAIN_PLAN, spacing_mm)
    z = _central_rear_feed_rise(
        stations,
        NO_FLOOR_FEED_REAR_Z if STAND_FOOT
        else NO_FLOOR_MAIN_FEED_START_Z,
        MAIN_TRENCH_CENTER_Z, NO_FLOOR_MAIN_FEED_RISE_LENGTH)
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
    z[descending] = (MAIN_TRENCH_CENTER_Z
                     + (pilot_low - MAIN_TRENCH_CENTER_Z) * _smooth01(u))
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
    burial_profile = _no_floor_service_patch_burial_profile(
        stations, xy, MAIN_OUTER_R,
        NO_FLOOR_MAIN_FEED_START_Z,
        MAIN_COVERED_BUMPS[0].station,
        MAIN_COVERED_BUMPS[0].low_z,
        protect_lm_recess=True)
    active = np.isfinite(burial_profile)
    z[active] = burial_profile[active]
    return stations, np.column_stack((xy, z))


def _ts_xyz(spacing_mm):
    stations, xy = _resample(_TS_PLAN, spacing_mm)
    # The UM seat is 2 mm higher than the LM seat.  Raise the nominal T
    # tunnel smoothly after it enters the UM ring.
    transition = _smooth01(
        (stations - TS_UM_ENTRY_S) / TS_UM_Z_TRANSITION_LENGTH_MM)
    nominal_z = T_LM_TRENCH_CENTER_Z + transition * (
        TS_UM_CENTER_Z - T_LM_TRENCH_CENTER_Z)
    feed_z = _central_rear_feed_rise(
        stations,
        NO_FLOOR_FEED_REAR_Z if STAND_FOOT else NO_FLOOR_T_FEED_START_Z,
        T_LM_TRENCH_CENTER_Z,
        CENTRAL_T_FEED_RISE_LENGTH)
    # The rear-feed rise is complete long before the UM transition.
    z = np.where(
        stations < CENTRAL_T_FEED_RISE_LENGTH, feed_z, nominal_z)
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
    burial_profile = _no_floor_service_patch_burial_profile(
        stations, xy, TS_OUTER_R,
        NO_FLOOR_T_FEED_START_Z,
        T_COVERED_BUMPS[0].station,
        T_COVERED_BUMPS[0].low_z,
        protect_lm_recess=True,
        recess_clearance_z=T_LM_TRENCH_CENTER_Z,
        direct_to_release=True)
    active = np.isfinite(burial_profile)
    z[active] = burial_profile[active]
    return stations, np.column_stack((xy, z))








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






def route_cable_points(spacing_mm: float = 0.5):
    """Complete UM centerline from LM feed to the 283-degree mouth."""
    return _main_xyz(spacing_mm)[1]


def ts_cable_points(spacing_mm: float = 0.5):
    """Complete tweeter centerline including crossover and handoff."""
    return _ts_xyz(spacing_mm)[1]


def lm_internal_duct_points(spacing_mm: float = 0.5):
    """Buried no-floor D9 LM centerline from entry to the R14 start."""
    if STAND_FOOT:
        return np.empty((0, 3), dtype=float)
    _stations, xy = _resample(_LM_INTERNAL_PLAN, spacing_mm)
    z = np.full((len(xy), 1), LM_INTERNAL_CENTER_Z_MM, dtype=float)
    return np.column_stack((xy, z))


def lm_rear_handoff_points(spacing_mm: float = 0.5):
    """Shared R14 LM handoff through this state's exact rear-face mouth."""
    if spacing_mm <= 0.0:
        raise ValueError("LM handoff spacing must be positive")
    arc_count = max(
        12,
        int(math.ceil(
            LM_EXIT_BEND_R_MM
            * LM_REAR_HANDOFF_SPEC["face_angle_rad"]
            / spacing_mm)),
    )
    return np.asarray(lm_exit_handoff_points(
        LM_REAR_HANDOFF_CENTER_Z,
        LM_REAR_PORT_REAR_Z,
        n=arc_count,
        rear_end_z_mm=LM_DUCT_OUT_REAR_Z_MM,
        radius_mm=LM_EXIT_BEND_R_MM,
        plan_bearing_deg=LM_REAR_HANDOFF_PLAN_BEARING_DEG,
        face_xy_mm=LM_REAR_PORT_XY,
    ), dtype=float)


def lm_complete_duct_points(spacing_mm: float = 0.5):
    """Complete printed LM centerline, including the gradual rear handoff."""
    if STAND_FOOT:
        from .floor import floor_lane_control_points
        return np.asarray(floor_lane_control_points("lm"), dtype=float)
    planar = lm_internal_duct_points(spacing_mm)
    handoff = lm_rear_handoff_points(spacing_mm)
    if np.linalg.norm(planar[-1] - handoff[0]) > 1.0e-9:
        raise RuntimeError("LM planar route no longer meets its R14 handoff")
    return np.vstack((planar, handoff[1:]))


def _no_floor_lm_complete_cutter_path(spacing_mm: float = 1.2):
    """Return the one authoritative D9 cutter path and handoff station."""
    if STAND_FOOT:
        raise RuntimeError("the complete no-floor LM cutter is no-floor-only")
    planar = lm_internal_duct_points(spacing_mm)
    points = lm_complete_duct_points(spacing_mm)
    start_tangent = points[1] - points[0]
    start_tangent /= np.linalg.norm(start_tangent)
    points = np.vstack((
        points[0] - LM_INTERNAL_JUNCTION_OVERTRAVEL_MM * start_tangent,
        points,
    ))
    # The inserted overtravel is section zero; the last planar point is the
    # exact G1 start of the shared R14 handoff.
    handoff_index = len(planar)
    handoff_station = float(np.linalg.norm(
        np.diff(points[:handoff_index + 1], axis=0), axis=1).sum())
    return points, handoff_station


def lm_internal_duct_cutter_points(spacing_mm: float = 0.5):
    """D9 planar cutter centerline with entry-side Boolean overtravel.

    The functional datums remain the first/last points returned by
    ``lm_internal_duct_points``.  Extending the Boolean loft 2 mm beyond each
    one makes the perpendicular entry bore cross the tube interior rather than
    meet an OCC end cap at a coincident T-junction.  The exit is a tangent R14
    continuation and therefore needs no hidden straight overtravel.
    """
    points = lm_internal_duct_points(spacing_mm)
    if len(points) < 2:
        raise RuntimeError("no-floor LM internal centerline is empty")
    start_tangent = points[1] - points[0]
    start_tangent /= np.linalg.norm(start_tangent)
    return np.vstack((
        points[0] - LM_INTERNAL_JUNCTION_OVERTRAVEL_MM * start_tangent,
        points,
    ))




















def lm_cable_points(spacing_mm: float = 0.5):
    """Reference-only free LM lead continuing along the mouth tangent."""
    if spacing_mm <= 0.0:
        raise ValueError("LM cable spacing must be positive")
    start = np.asarray(LM_REAR_HANDOFF_SPEC["face"], dtype=float)
    end = np.asarray(LM_EXTERNAL_LEAD_END, dtype=float)
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(distance / spacing_mm)))
    u = np.linspace(0.0, 1.0, count + 1)[:, None]
    return start[None, :] + u * (end - start)[None, :]




















































































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
    lm_internal_index = t_group_stop
    if which == "lm" and index == lm_internal_index:
        if STAND_FOOT:
            return ()
        return tuple(no_floor_lm_internal_cutter().solids())
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
    lm_owner_plan = _lm_positive_owner_plan()
    lm_annulus = lm_owner_plan.difference(
        Point(*L22_CUTOUT[:2]).buffer(
            L22_CUTOUT[2] / 2.0, resolution=96))
    um_annulus = Point(*UM_CUTOUT[:2]).buffer(
        UM_CORE_R, resolution=96).difference(
            Point(*UM_CUTOUT[:2]).buffer(
                UM_CUTOUT[2] / 2.0, resolution=96))
    lm_domain = lm_owner_plan
    um_domain = Point(*UM_CUTOUT[:2]).buffer(UM_CORE_R, resolution=96)
    tail_domain = box(*(
        FLOOR_STEM_CORE_BOUNDS if STAND_FOOT
        else NO_FLOOR_BRIDGE_ROUTE_BOUNDS))
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
            x0, y0, x1, y1 = NO_FLOOR_BRIDGE_ROUTE_BOUNDS
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










def route_facts():
    main_s, main = _main_xyz(0.20)
    ts_s, ts = _ts_xyz(0.20)
    lm_internal = lm_internal_duct_points(0.20)
    lm_free = lm_cable_points(0.20)
    main_cross_z = float(np.interp(CROSSOVER_MAIN_S, main_s, main[:, 2]))
    ts_cross_z = float(np.interp(CROSSOVER_TS_S, ts_s, ts[:, 2]))
    void_web = ts_cross_z - main_cross_z - CUTTER_R - TS_CUTTER_R
    physical_gap = (ts_cross_z - main_cross_z
                    - CABLE_R_EST - TS_CABLE_D_EST / 2.0)
    free_um_to_t_cover_gap = (
        ts_cross_z - main_cross_z - CABLE_R_EST - TS_OUTER_R)
    rear_entry_ports = tuple({
        "name": bore.name,
        "xy_mm": bore.xy,
        "diameter_mm": 2.0 * bore.radius_mm,
        "rear_z_mm": bore.rear_z_mm,
        "inner_z_mm": bore.inner_z_mm,
    } for bore in no_floor_rear_entry_bores())
    rear_entry_vestibules = tuple({
        "name": vestibule.name,
        "xy_mm": vestibule.xy,
        "diameter_mm": 2.0 * vestibule.radius_mm,
        "center_z_mm": vestibule.center_z_mm,
        "rear_skin_mm": (
            vestibule.center_z_mm - vestibule.radius_mm - PAD_FACE_Z),
    } for vestibule in no_floor_rear_entry_vestibules())
    if len(lm_internal):
        lm_internal_line = LineString(lm_internal[:, :2])
        lm_to_um_plan_lumen_wall = (
            lm_internal_line.distance(LineString(main[:, :2]))
            - LM_INTERNAL_DUCT_R - CUTTER_R)
        lm_to_t_plan_lumen_wall = (
            lm_internal_line.distance(LineString(ts[:, :2]))
            - LM_INTERNAL_DUCT_R - TS_CUTTER_R)
        lm_to_um_lumen_wall = _sampled_centerline_surface_wall(
            lm_internal, LM_INTERNAL_DUCT_R, main, CUTTER_R)
        lm_to_t_lumen_wall = _sampled_centerline_surface_wall(
            lm_internal, LM_INTERNAL_DUCT_R, ts, TS_CUTTER_R)
    else:
        lm_to_um_plan_lumen_wall = None
        lm_to_t_plan_lumen_wall = None
        lm_to_um_lumen_wall = None
        lm_to_t_lumen_wall = None

    def service_patch_record(
            points, stations, inner_radius, outer_radius, release_station,
            *, protect_lm_recess=False):
        if STAND_FOOT:
            return None
        guard = _no_floor_burial_guard_stations(
            stations, points[:, :2], outer_radius)
        flat_guard_end = (
            guard["patch_guard_end_station_mm"]
            if protect_lm_recess else guard["guard_end_station_mm"])
        guarded = stations <= flat_guard_end + 1.0e-9
        return {
            **guard,
            "flat_face_guard_end_station_mm": float(flat_guard_end),
            "release_station_mm": float(release_station),
            "transition_length_mm": float(
                release_station - flat_guard_end),
            "protects_lm_recess": bool(protect_lm_recess),
            "min_lumen_rear_skin_mm": float(np.min(
                points[guarded, 2] - inner_radius - PAD_FACE_Z)),
            "min_outer_cover_rear_clearance_mm": float(np.min(
                points[guarded, 2] - outer_radius - PAD_FACE_Z)),
        }

    service_patch_routes = None if STAND_FOOT else {
        "um": service_patch_record(
            main, main_s, CUTTER_R, MAIN_OUTER_R,
            MAIN_COVERED_BUMPS[0].station,
            protect_lm_recess=True),
        "t": service_patch_record(
            ts, ts_s, TS_CUTTER_R, TS_OUTER_R,
            T_COVERED_BUMPS[0].station,
            protect_lm_recess=True),
    }

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
        "lm_roof_mm": LM_SEAT_Z - MAIN_TRENCH_CENTER_Z - CUTTER_R,
        "bridge_side_wall_mm": TUNNEL_SKIN,
        "ts_lm_roof_mm": LM_SEAT_Z - T_LM_TRENCH_CENTER_Z - TS_CUTTER_R,
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
        # Legacy field names below refer to the two long UM/T routes owned by
        # the LM carrier.  The explicit all-cable fields include the new LM
        # tunnel and prevent it from disappearing from release inventory.
        "functional_lm_feed_count": 2,
        "functional_lm_feed_mode": (
            "integrated_stem_rear_face_shallow_rise" if STAND_FOOT
            else "bridge_rear_normal_entry_then_buried_rise"),
        "functional_lm_feed_points": (
            (tuple(map(float, main[0])), tuple(map(float, ts[0])))
            if STAND_FOOT else (
                (*tuple(map(float, NO_FLOOR_MAIN_FEED_XY)), PAD_FACE_Z),
                (*tuple(map(float, NO_FLOOR_T_FEED_XY)), PAD_FACE_Z))),
        "buried_route_start_points": (
            tuple(map(float, main[0])), tuple(map(float, ts[0]))),
        "no_floor_route_start_z_mm": (
            None if STAND_FOOT else (
                NO_FLOOR_MAIN_FEED_START_Z,
                NO_FLOOR_T_FEED_START_Z)),
        "no_floor_route_rear_skin_mm": (
            None if STAND_FOOT else (
                NO_FLOOR_MAIN_FEED_START_Z - CUTTER_R - PAD_FACE_Z,
                NO_FLOOR_T_FEED_START_Z - TS_CUTTER_R - PAD_FACE_Z)),
        "no_floor_entry_route_overlap_mm": (
            None if STAND_FOOT else (
                NO_FLOOR_ENTRY_BORE_DEPTH_MM + PAD_FACE_Z
                - NO_FLOOR_MAIN_FEED_START_Z,
                NO_FLOOR_ENTRY_BORE_DEPTH_MM + PAD_FACE_Z
                - NO_FLOOR_T_FEED_START_Z)),
        "central_owner_feed_xy": (
            tuple(map(float, CENTRAL_MAIN_FEED_XY)),
            tuple(map(float, CENTRAL_T_FEED_XY))),
        "central_owner_feed_state": "floor" if STAND_FOOT else "no_floor",
        "central_owner_feed_rear_z_mm": NO_FLOOR_FEED_REAR_Z,
        "central_owner_feed_rise_lengths_mm": (
            NO_FLOOR_MAIN_FEED_RISE_LENGTH,
            CENTRAL_T_FEED_RISE_LENGTH),
        "no_floor_service_patch_margin_mm": (
            None if STAND_FOOT else NO_FLOOR_SERVICE_PATCH_MARGIN_MM),
        "no_floor_service_patch_bounds_mm": (
            None if STAND_FOOT else NO_FLOOR_SERVICE_PATCH_BOUNDS),
        "no_floor_service_patch_release_mode": (
            None if STAND_FOOT else NO_FLOOR_SERVICE_PATCH_RELEASE_MODE),
        "no_floor_service_patch_routes": service_patch_routes,
        "no_floor_rear_entry_ports": rear_entry_ports,
        "no_floor_rear_entry_vestibules": rear_entry_vestibules,
        "no_floor_um_entry_cap_relief_half_length_mm": (
            None if STAND_FOOT
            else NO_FLOOR_UM_ENTRY_CAP_RELIEF_HALF_LENGTH_MM),
        "no_floor_um_entry_cap_relief_radial_inset_mm": (
            None if STAND_FOOT
            else NO_FLOOR_UM_ENTRY_CAP_RELIEF_RADIAL_INSET_MM),
        "no_floor_entry_vestibule_rear_skin_mm": (
            None if STAND_FOOT
            else NO_FLOOR_ENTRY_VESTIBULE_REAR_SKIN_MM),
        "no_floor_entry_layout": (
            None if STAND_FOOT
            else "d20_lm_top_t_lower_left_um_lower_right"),
        "no_floor_entry_window_center_xy_mm": (
            None if STAND_FOOT else tuple(map(
                float, OBIWAN_NO_FLOOR_ENTRY_WINDOW_CENTER_XY))),
        "no_floor_entry_window_diameter_mm": (
            None if STAND_FOOT
            else OBIWAN_NO_FLOOR_ENTRY_WINDOW_D_MM),
        "no_floor_lm_entry_buried_relief_radial_mm": (
            None if STAND_FOOT
            else NO_FLOOR_LM_ENTRY_RELIEF_RADIAL_MM),
        "no_floor_lm_entry_buried_relief_rear_skin_mm": (
            None if STAND_FOOT
            else NO_FLOOR_LM_ENTRY_RELIEF_REAR_SKIN_MM),
        "no_floor_t_entry_buried_relief_radial_mm": (
            None if STAND_FOOT
            else NO_FLOOR_T_ENTRY_RELIEF_RADIAL_MM),
        "no_floor_t_entry_buried_relief_rear_skin_mm": (
            None if STAND_FOOT
            else NO_FLOOR_T_ENTRY_RELIEF_REAR_SKIN_MM),
        "functional_all_cable_feed_count": (
            2 if STAND_FOOT else 3),
        "functional_all_cable_feed_names": (
            ("um", "t") if STAND_FOOT else ("lm", "um", "t")),
        "no_floor_lm_entry_xy_mm": (
            None if STAND_FOOT
            else tuple(map(float, NO_FLOOR_LM_FEED_XY))),
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
        "lm_visible_ring_radius_mm": LM_VISIBLE_RING_R,
        "lm_route_owner_clearance_mm": LM_ROUTE_OWNER_CLEARANCE,
        "main_lm_route_radius_mm": MAIN_LM_ROUTE_R,
        "t_lm_route_radius_mm": TS_LM_ROUTE_R,
        "main_lm_lumen_outer_radius_mm": MAIN_LM_ROUTE_R + CUTTER_R,
        "t_lm_lumen_outer_radius_mm": TS_LM_ROUTE_R + TS_CUTTER_R,
        "main_lm_cover_outer_radius_mm": MAIN_LM_ROUTE_R + MAIN_OUTER_R,
        "t_lm_cover_outer_radius_mm": TS_LM_ROUTE_R + TS_OUTER_R,
        "lm_ring_min_exterior_skin_mm": min(
            LM_VISIBLE_RING_R - (MAIN_LM_ROUTE_R + CUTTER_R),
            LM_VISIBLE_RING_R - (TS_LM_ROUTE_R + TS_CUTTER_R)),
        # The carrier's native R113.8 fairing remains continuous outside the
        # buried covers.  Cover-to-owner clearance is solid material, so it
        # cannot be reported as the former open exterior groove.
        "lm_ring_route_groove_mm": 0.0,
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
        "burial_web_owner_inset_mm": BURIAL_WEB_OWNER_INSET,
        "printed_lm_tunnel_count": 0 if STAND_FOOT else 1,
        "lm_lead_mode": (
            "integral_floor_lane_then_r14_d9_rear_handoff"
            if STAND_FOOT else
            "d20_cluster_entry_buried_d9_tunnel_then_r14_rear_handoff"),
        "lm_internal_diameter_mm": (
            None if STAND_FOOT else LM_INTERNAL_DUCT_D_MM),
        "lm_internal_center_z_mm": (
            None if STAND_FOOT else LM_INTERNAL_CENTER_Z_MM),
        "lm_internal_route_length_mm": (
            None if STAND_FOOT else LM_INTERNAL_ROUTE_LENGTH_MM),
        "lm_internal_front_skin_mm": (
            None if STAND_FOOT else LM_INTERNAL_FRONT_SKIN_MM),
        "lm_internal_rear_skin_mm": (
            None if STAND_FOOT else LM_INTERNAL_REAR_SKIN_MM),
        "lm_internal_to_um_lumen_wall_mm": lm_to_um_lumen_wall,
        "lm_internal_to_t_lumen_wall_mm": lm_to_t_lumen_wall,
        "lm_internal_lumen_wall_measurement": "sampled_3d_centerline_surface",
        "lm_internal_lumen_wall_sample_spacing_mm": 0.20,
        "lm_internal_to_um_plan_lumen_wall_mm": lm_to_um_plan_lumen_wall,
        "lm_internal_to_t_plan_lumen_wall_mm": lm_to_t_plan_lumen_wall,
        "lm_rear_exit_kind": "continuous_r14_d9_rear_face_handoff",
        "lm_rear_exit_bend_radius_mm": LM_EXIT_BEND_R_MM,
        "lm_rear_exit_min_qualified_radius_mm": LM_EXIT_MIN_BEND_R_MM,
        "lm_rear_port_xy_mm": tuple(map(float, LM_REAR_PORT_XY)),
        "lm_rear_port_clearance_from_aperture_mm": (
            LM_REAR_PORT_CLEARANCE_FROM_APERTURE_MM),
        "lm_rear_port_diameter_mm": LM_REAR_PORT_D_MM,
        "lm_rear_port_rear_z_mm": LM_REAR_PORT_REAR_Z,
        "lm_rear_port_inner_z_mm": LM_REAR_PORT_INNER_Z,
        "lm_rear_handoff_start_xyz_mm": LM_REAR_HANDOFF_SPEC["start"],
        "lm_rear_handoff_plan_bearing_deg": (
            LM_REAR_HANDOFF_SPEC["plan_bearing_deg"]),
        "lm_rear_handoff_plan_tangent_xyz": (
            LM_REAR_HANDOFF_SPEC["plan_tangent"]),
        "lm_rear_face_mouth_xyz_mm": LM_REAR_HANDOFF_SPEC["face"],
        "lm_rear_face_angle_deg_from_normal": (
            LM_REAR_HANDOFF_SPEC["face_angle_deg_from_rear_normal"]),
        "lm_rear_face_tangent_xyz": LM_REAR_HANDOFF_SPEC["face_tangent"],
        "lm_external_cable_end_z_mm": LM_EXTERNAL_LEAD_END_Z,
        "lm_external_cable_follows_handoff_tangent": bool(
            np.allclose(
                (lm_free[-1] - lm_free[0])
                / np.linalg.norm(lm_free[-1] - lm_free[0]),
                LM_REAR_HANDOFF_SPEC["face_tangent"], atol=1e-12)),
    }


# Compatibility facade: implementations live in cohesive Stage 4 owners.
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

from .bumps import (
    BumpBackfillSpec,
    CoveredBump,
    MAIN_COVERED_BUMPS,
    T_COVERED_BUMPS,
    _anchor_keep_mask,
    _anchor_leg,
    _anchor_leg_components,
    _anchored_cover_components,
    _bump_backfill,
    _burial_web,
    _burial_web_components,
    _burial_web_masks,
    _burial_web_owner_plan,
    _central_rear_feed_rise,
    _contract_outside_halfspace,
    _extended_points,
    _fused_cover_group,
    _interp_xyz,
    _lm_positive_owner_plan,
    _lm_printed_owner_crop,
    _lm_ring_outer_crop,
    _lm_state_tail_crop,
    _main_printed_outer_from_full,
    _named_bumps,
    _no_floor_burial_guard_stations,
    _no_floor_service_patch_burial_profile,
    _owned_crop_solids,
    _owner_cutter_extension,
    _owner_cutter_points,
    _owner_ray_limit,
    _point_runs,
    _point_runs_with_boundary_overlap,
    _required_solids,
    _support_plan_mask,
    _t_flush_owner_components,
    _t_flush_owner_crop,
    _t_owner_phase_shell,
    _um_owner_crop,
    bump_backfill_components,
    bump_backfill_specs,
    required_assembled_shell_components,
    required_assembled_shell_segment_components,
    required_handoff_shell_components,
    route_outer_covers,
)
