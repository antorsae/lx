"""Extreme V1LF R6F barebone: two flush driver carriers and nothing else.

The previous V1LF was a rear-thinned copy of the complete B2 outline.
This module starts from the irreducible interfaces instead:

* LM carrier: D190 opening, D221.2 flush seat, R113.0 outside.
* UM carrier: D82 opening, D98.6 flush seat, R51.7 outside.
* two rounded M3 through-bolted half-lap ears establish 165.100 mm;
* six fully buried captive magnet interfaces provide four LM and two UM
  alignment sites;
  the original upper LM pair stays at +/-26 degrees from top while the lower
  LM pair mates horizontally through the shared W64 base sides;
* floor mode owns a full-depth integral W64 stem and rectangular floor foot;
  no-floor mode owns a shallow front-flush four-hole solid web;
* the D8.2 UM path is buried only in LM and exits flush at R113 before its
  free span behind UM; D6 is buried only in LM/UM before floating behind the
  crescent; the short LM lead floats in a minimal rear-open relief without a
  printed micro-duct;
* two compact direct half-lap ears attach the optional tweeter crescent.
* tangent-blended LM--UM and T--UM cusp closures are full-depth solids,
  split in plan between their independently printed owners; only the central
  T free-cable mouth and the functional route cuts remain open.

The outer lips are only 2.4 mm beyond the flange-recess radii. The LM
rear insert pads, driver pilots and flush seats remain unchanged. The
only the tweeter crescent is a separate printed add-on.  The floor stem,
foot, NL8 panel and buried continuations are monolithic LM geometry;
there is no floor-support add-on or support hardware.
the no-floor bridge web is the sole core exception: it is
mandatory, monolithic LM-core geometry in that state.
The printed cable passage is irreducible core geometry only inside its LM/UM
owners.  Every continuation beyond the flush owner mouth is free cable.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


# A direct module invocation used to enter both full carrier builds before a
# guard existed. Establish the authenticated outer guard before importing OCC.
if __name__ == "__main__":
    import run_memory_guarded as _memory_guard

    if not _memory_guard.is_guarded_process():
        raise SystemExit(_memory_guard.main([
            sys.executable, str(Path(__file__).resolve())]))

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
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def _require_guarded_build():
    """Reject accidental in-process carrier construction outside the guard."""
    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        raise RuntimeError(
            "V1LF carrier construction requires run_memory_guarded.py; "
            "use Make or the staged V1LF exporter")


from top_baffle_nd25fw4 import (
    L22_CUTOUT,
    L22_PILOT_D_MM,
    STAND_FOOT,
    THICKNESS_MM,
    UM_CUTOUT,
    UM_PILOT_DEPTH_MM,
    UM_PILOT_D_MM,
    UM_TERMINAL_CLOCK_DEG,
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
    UM_RECESS_R,
    UM_SEAT_Z,
)
import top_baffle_nd25fw4_v1lf_route as route
from top_baffle_nd25fw4_v1lf_route import (
    CROSSOVER_T_Z,
    TS_CUTTER_R,
    TUNNEL_ROOF_SKIN,
    lm_free_lead_relief_cutter,
    route_inner_cutter_group,
    route_inner_cutter_group_count,
    route_inner_cutters,
    route_outer_covers,
)
from top_baffle_nd25fw4_v1lf_bridge import (
    floor_wing_contact_profile_addition,
    fused_bridge_tail,
)
from top_baffle_nd25fw4_v1lf_floor import (
    apply_integrated_floor_feature_group,
    integrated_floor_addition,
    integrated_floor_feature_group_count,
)
from captive_magnets import (
    CAPTIVE_LAND_MM,
    CAVITY_DEPTH_MM,
    CAVITY_DIAMETER_MM,
    FACE_SKIN_MM,
    INNER_SKIN_MM,
    INTERFACE_GAP_MM,
    wall_cavity_tools,
)

PRINT_ORIENTATION = "front-face-down"
CORE_REAR_Z = 6.8
LM_CORE_R = LM_RECESS_R + 2.4       # 113.0
UM_CORE_R = UM_RECESS_R + 2.4       # 51.7
CORE_CENTER_SPACING = UM_CUTOUT[1] - L22_CUTOUT[1]  # 165.100
CORE_RING_GAP = CORE_CENTER_SPACING - LM_CORE_R - UM_CORE_R  # 0.400

# The old carrier blank left full 5.5/7.5-mm annular slabs beneath the
# flush seats. R6F retains only two full extrusion widths as a continuous
# seat membrane; the narrow outer lip, insert bosses and short radial spokes
# carry load. Tunnel roofs locally merge into this membrane.
SEAT_MEMBRANE_T = TUNNEL_ROOF_SKIN
SEAT_MEMBRANE_LIP_OVERLAP = 0.40
LM_STRUCT_SPOKE_W = 4.0
UM_INSERT_BOSS_D = UM_PAD_D_MM
UM_STRUCT_SPOKE_W = 3.2
STRUCT_DESIGN_MASS_KG = 4.0
STRUCT_CREEP_ALLOW_MPA = 8.0
STRUCT_SHORT_ALLOW_MPA = 18.0

# Exactly two LM and one UM magnets per physical side. There are no rear
# attachment bores. The upper LM pair remains radial at +/-26 degrees from the
# top centreline (world polar 116/64 degrees). The lower pair is no longer in
# the R113 lip: it is mirrored through the straight sides of the shared W64
# lower tongue at x=+/-32, y=18, with horizontal outward normals. Both lower
# sites use the common z=12.55 datum, so floor/no-floor carriers and the Ac/Ae
# lm_lower prints share one exact receiver axis.  D5x2 magnets are completely
# captive in D5.20 x 2.10 cavities behind 0.45-mm skins, with a printable
# circular cradle and self-supporting 45-degree roof.  The lower pair stays on
# the exact W64 side faces.  The four 2.40-mm ring lips need a local 0.60-mm
# outward backing boss to contain the qualified 3.00-mm land without entering
# either flange recess.  These are sealed interface pads, never exposed magnet
# ears.  Magnets carry alignment/anti-rattle load only.  The monolithic W64
# stem/root carries floor load; the four stock bridge holes carry no-floor
# load.
SIDE_EAR_D = 7.8
SIDE_EAR_IN = 0.7
SIDE_EAR_OUT = 3.3
SIDE_MAGNET_D = 5.0
SIDE_MAGNET_POCKET_D = CAVITY_DIAMETER_MM
SIDE_MAGNET_DEPTH = CAVITY_DEPTH_MM
SIDE_MAGNET_CAPTIVE_LAND = CAPTIVE_LAND_MM
SIDE_MAGNET_FACE_SKIN = FACE_SKIN_MM
SIDE_MAGNET_INNER_SKIN = INNER_SKIN_MM
SIDE_MAGNET_Z = {
    "lm": 12.55,
    # Raised within the R51.7 lip so the complete captive envelope keeps at
    # least 1.1 mm from the conservative upper-T cover envelope.  The local
    # +0.60-mm backing makes room for both qualified 0.45-mm axial skins; the
    # inward skin ends exactly at the immutable R49.3 recess datum.
    "um": 15.10,
}
SIDE_INTERFACE_GAP = INTERFACE_GAP_MM
SIDE_MAGNET_ANGLES = {
    # The upper pair is one degree crownward of the adjacent 120/60-degree W22
    # inserts, retaining its exact 2.251-mm D5.2 captive-envelope to D9.6-boss
    # edge gap. Lower LM sites are explicit base-side records below rather
    # than fake polar positions on the R113 ring.
    "lm": (116.0, 64.0),
    # Symmetric top-side sites clear the T arc, D8 insert pads and direct
    # tweeter ears; the former 45-degree site cut through the T cover.
    "um": (129.5, 50.5),
}
SIDE_MAGNET_FACE_OFFSET = {
    # The immutable flange-recess leaves only a 2.40-mm radial lip.  Shift
    # the local interface face outward by the missing 0.60 mm so the entire
    # qualified 3.00-mm captive land ends exactly at, never inside, the
    # recess.  This is a local backing boss, not an exposed magnet ear.
    "lm": CAPTIVE_LAND_MM - (LM_CORE_R - LM_RECESS_R),
    "um": CAPTIVE_LAND_MM - (UM_CORE_R - UM_RECESS_R),
}
LM_BASE_MAGNET_FACE_X = 32.0
LM_BASE_MAGNET_Y = 18.0
LM_BASE_MAGNET_Z = 12.55

# Two compact Z-axis bolted *rounded ears*.  The old rectangular tabs
# projected into the MU flange seating region.  Each replacement is a
# D9 bolt boss convex-hulled to a D4 contact neck on its owning ring.
# The complete footprint (not only its M3 axis) clears both recess discs.
# LM owns the rear half and UM owns the front half; their 0.20 mm Z air
# gap prevents the two prints fusing during assembly.
# A 7.00-mm outward shift from the concept nominal clears the buried T route
# and the complementary half-lap receiver envelope. Neither rounded boss nor
# neck is enlarged; the UM half-lap remains clear by Z separation.
JOINT_EAR_X = (-32.0, 32.0)
JOINT_EAR_Y = 315.770102
JOINT_HOLE_D = 3.4
JOINT_BOSS_D = 9.0
JOINT_NECK_D = 4.0
JOINT_BORE_REAR_OVERSHOOT = 0.01
LM_JOINT_Z = (CORE_REAR_Z, 12.20)
UM_JOINT_Z = (12.40, THICKNESS_MM)
UM_JOINT_TUNNEL_LIGAMENT = THICKNESS_MM - (
    CROSSOVER_T_Z + TS_CUTTER_R)
JOINT_RECEIVER_RADIAL_CLEAR = 0.10

# Direct UM-to-tweeter half-laps.  They supersede the deleted long
# side-magnet knees while keeping the optional crescent independently
# printable.
# Rear-driven M3 screws pass only through the core half; the crescent
# owns blind insert receivers so its acoustic front stays uninterrupted.
TWEETER_JOINT_X = (-24.0, 24.0)
TWEETER_JOINT_Y = 421.5
TWEETER_JOINT_BOSS_D = 9.0
TWEETER_JOINT_NECK_D = 4.0
TWEETER_JOINT_HOLE_D = 3.4
TWEETER_JOINT_INSERT_BORE_D = 4.6
TWEETER_CORE_JOINT_Z = (CORE_REAR_Z, 12.20)
TWEETER_ADDON_JOINT_Z = (12.40, THICKNESS_MM)
TWEETER_JOINT_CLEAR = 0.10
# Keep the rear M3 through-bore continuous into the add-on's blind insert
# receiver, but do not terminate its cutter on a real 0.20/0.16-mm Bambu
# layer.  The former +0.30 endpoint was exactly world Z=12.50; an OCC section
# through that coincident cap retained one mirrored bore and closed the other.
# The extra 0.05 mm remains inside the existing 0.35-mm axial overlap with
# the add-on receiver and makes the Z=12.50 print layer unambiguously open.
TWEETER_CORE_BORE_TOP_Z = TWEETER_CORE_JOINT_Z[1] + 0.35

# Full-depth, plan-split closure webs replace the two pairs of open cusp
# islands at the LM--UM and UM--tweeter junctions.  These are deliberately
# not thin front skins: every owner occupies the complete V1LF depth so a
# front-face-down print has solid material behind the coplanar front face.
# The complementary owners stop across the same 0.05-mm assembly clearance
# used by the captive side interfaces. Existing Z-split half-lap ears remain
# separate: both web plans first reach the ear neighbourhood, then the normal
# complementary Z-half receiver recuts establish fit clearance. The receiver
# footprint record below is diagnostic only; it is never subtracted from both
# plan owners.
JUNCTION_WEB_Z = (CORE_REAR_Z, THICKNESS_MM)
JUNCTION_WEB_SEAM_GAP = SIDE_INTERFACE_GAP
JUNCTION_WEB_OWNER_OVERLAP = 0.40
JUNCTION_WEB_EAR_CLEAR = JOINT_RECEIVER_RADIAL_CLEAR
JUNCTION_WEB_SAMPLES = 96
JUNCTION_WEB_EAR_CHORD_INSET = 0.40
JUNCTION_WEB_LENS_FUSION_MM = 0.45
JUNCTION_WEB_MIN_LENS_AREA_MM2 = 0.05
# The handoff recut must use the same R3.00 globally phased lumen as the
# production route cutter.  Any radial oversize removes qualified shell at
# the R113 owner mouth (the former +0.18 left only 0.62 mm of the required
# 0.80-mm cover).  Axial/radial locality comes from the annular crop below,
# not from enlarging the cable void.
LM_T_CLOSURE_HANDOFF_RELIEF_MM = 0.0
LM_T_CLOSURE_HANDOFF_RADIAL_INSET_MM = 2.0
LM_T_CLOSURE_HANDOFF_RADIAL_OUTSET_MM = 4.0
LM_UM_WEB_BLEND_START_X = 20.0
T_UM_WEB_BLEND_START_X = 14.0

# The buried left-hand T cover approaches the LM lip closely enough to pinch
# off a 1.58-mm2 sliver of the otherwise rear-open D221.2 flange recess at
# the 103.1..106.3-degree crown sector.  That is not cable lumen: the closest
# D6 cover surface remains 0.27 mm away and the D6 cutter remains 1.07 mm
# away.  Close it with a solid, round-ended rear crescent that overlaps the
# R110.6 lip and ends exactly at the underside of the existing seat membrane.
# Mirror the crescent so the released carrier remains symmetric at the
# structural LM--UM crown.  Nothing is added in the flange seating depth.
LM_UM_REAR_BACKFILL_CENTER_ANGLES_DEG = (75.3, 104.7)
LM_UM_REAR_BACKFILL_CENTER_R = 110.20
LM_UM_REAR_BACKFILL_ARC_HALF_SPAN_DEG = 6.0
LM_UM_REAR_BACKFILL_RADIAL_WIDTH_MM = 1.30
LM_UM_REAR_BACKFILL_Z = (
    CORE_REAR_Z, LM_SEAT_Z - SEAT_MEMBRANE_T)

# The T cover's lower-right UM approach similarly pinches blind pockets
# against the inner R49.3 lip.  They migrate from R49.30 at z=11.70 inward to
# R48.16 at z=12.50 as the round cover section closes.  Exact route-section
# checks keep every pocket at least 1.006 mm outside the functional R3 lumen.
# Two mirrored rear crescents span the complete possible cover/lip pinch band
# (R46.8..50.2) without entering the D82 opening or the UM flange seat.  The
# later route cutter remains authoritative wherever the land meets its lumen.
UM_T_REAR_BACKFILL_CENTER_ANGLES_DEG = (231.7, 308.3)
UM_T_REAR_BACKFILL_CENTER_R = 48.50
UM_T_REAR_BACKFILL_ARC_HALF_SPAN_DEG = 5.0
UM_T_REAR_BACKFILL_RADIAL_WIDTH_MM = 3.40
UM_T_REAR_BACKFILL_Z = (
    CORE_REAR_Z, UM_SEAT_Z - SEAT_MEMBRANE_T)

# The lower junction closes continuously between the two M3 half-laps. At
# the upper junction the two positive-height cusp bands close up to the
# central outline overlap; the existing route cutter remains authoritative
# for the D5.2 free-cable handoff. Its 6.0-mm plan keepout prevents either
# side web from intruding into that service region.
LM_UM_WEB_HALF_WIDTH = abs(JOINT_EAR_X[1])
T_UM_CABLE_MOUTH_HALF_WIDTH = 6.0
T_UM_WEB_OUTER_X = abs(TWEETER_JOINT_X[1])

# Exact B2 right-hand crescent arc through the released three-point arc
# (36.813,432.866), (24.570,423.478), (10.081,418.176).  The web uses the
# mirrored lower branch as its upper tangent boundary; no material is added
# outside the existing B2/V1LF envelope.
T_CRESCENT_ARC_CENTER = (-0.016809359544911025, 468.21906343086)
T_CRESCENT_ARC_R = 51.05167922220417

# Conservative screen for both two-ear interfaces. The datasheet/printing
# ledger gives 0.43 kg UM + 0.20 kg tweeter pair; 0.85 kg retains margin for
# the skeletal UM carrier, crescent, wire and fasteners. The bridge/support,
# not these upper joints, carries the complete 4 kg baffle. Magnets remain
# absent from every load path.
JOINT_DESIGN_MASS_KG = 0.85
JOINT_PLAN_LEVER_MM = 120.0
JOINT_REAR_LEVER_MM = 70.0
JOINT_CONTACT_LEVER_FACTOR = 0.75
JOINT_PLA_CREEP_ALLOW_MPA = 8.0
JOINT_PLA_SHORT_ALLOW_MPA = 18.0
JOINT_M3_SHEAR_ALLOW_MPA = 100.0
JOINT_M3_TENSION_ALLOW_MPA = 100.0


def _cylinder_at(cx: float, cy: float, radius: float,
                 z0: float, z1: float):
    return Pos(cx, cy, (z0 + z1) / 2.0) * Cylinder(radius, z1 - z0)


def _polar_xy(center, radius, angle_deg):
    a = math.radians(angle_deg)
    return (center[0] + radius * math.cos(a),
            center[1] + radius * math.sin(a))


def _plan_prism(polygon, z0: float, z1: float):
    """Extrude a Shapely Polygon/MultiPolygon, preserving every hole."""
    if polygon.geom_type != "Polygon":
        return Compound(children=[
            _plan_prism(piece, z0, z1) for piece in polygon.geoms
        ])
    # Force +Z face normals; unary unions and buffers otherwise return mixed
    # winding and silently mirror nominal z0..z1 members below z0.
    polygon = orient(polygon, sign=1.0)
    outer = Wire(Polyline(*[
        (float(x), float(y)) for x, y in polygon.exterior.coords
    ]).edges())
    holes = [
        Wire(Polyline(*[(float(x), float(y)) for x, y in ring.coords]).edges())
        for ring in polygon.interiors
        # Shapely unions can retain zero-area numerical rings (~1e-17 mm2).
        # Passing those through OCC produces an invalid prism even though the
        # printable plan is topologically solid.  Preserve every physical
        # hole and discard only sub-nanometric Boolean residue.
        if Polygon(ring).area > 1.0e-8
    ]
    face = Face(outer, holes)
    return Pos(0.0, 0.0, z0) * extrude(face, amount=z1 - z0)


def _plan_polygon_components(geometry):
    """Yield only positive-area polygon members from a Shapely result."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        if geometry.area > 1.0e-8:
            yield geometry
        return
    for child in geometry.geoms:
        yield from _plan_polygon_components(child)


def _subtract_plan_prisms(part, geometry, z0: float, z1: float):
    """Subtract a possibly disconnected plan as ordinary solid booleans."""
    pieces = list(_plan_polygon_components(geometry))
    if not pieces:
        raise RuntimeError("plan subtraction contains no positive-area polygon")
    for piece in pieces:
        part -= _plan_prism(piece, z0, z1)
    return part


def _minimal_ring_blank(center, cut_radius, recess_radius, outer_radius,
                        seat_z):
    """Load-bearing outer lip plus the minimum continuous seat membrane."""
    cx, cy = center
    lip = _cylinder_at(cx, cy, outer_radius, CORE_REAR_Z, THICKNESS_MM)
    lip -= _cylinder_at(
        cx, cy, recess_radius, CORE_REAR_Z - 0.2, THICKNESS_MM + 0.2)
    membrane = _cylinder_at(
        cx, cy, recess_radius + SEAT_MEMBRANE_LIP_OVERLAP,
        seat_z - SEAT_MEMBRANE_T, seat_z)
    membrane -= _cylinder_at(
        cx, cy, cut_radius, seat_z - SEAT_MEMBRANE_T - 0.2,
        seat_z + 0.2)
    blank = lip.fuse(membrane).clean()
    solids = list(blank.solids())
    if (not blank.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "minimal carrier lip/membrane blank must be one valid solid; "
            f"valid={blank.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def _radial_spoke(center, pilot_xy, contact_radius, width, z0, z1):
    dx = pilot_xy[0] - center[0]
    dy = pilot_xy[1] - center[1]
    length = math.hypot(dx, dy)
    contact = (
        center[0] + contact_radius * dx / length,
        center[1] + contact_radius * dy / length,
    )
    plan = LineString((pilot_xy, contact)).buffer(
        width / 2.0, resolution=24, cap_style=1, join_style=1)
    return _plan_prism(plan, z0, z1)


def carrier_spoke_load_facts():
    """Conservative direct-shear screen for the retained insert spokes."""
    gravity = 9.80665
    lm_area = (len(LM_PILOT_XY) * LM_STRUCT_SPOKE_W
               * (LM_SEAT_Z - PAD_FACE_Z))
    um_floor = UM_SEAT_Z - UM_PILOT_DEPTH_MM - UM_PAD_FLOOR_MM
    um_area = (len(UM_PILOT_XY) * UM_STRUCT_SPOKE_W
               * (UM_SEAT_Z - um_floor))

    def record(area, g_load, allowable):
        stress = STRUCT_DESIGN_MASS_KG * gravity * g_load / area
        return stress, allowable / stress

    lm_1g, lm_sf_1g = record(lm_area, 1.0, STRUCT_CREEP_ALLOW_MPA)
    lm_3g, lm_sf_3g = record(lm_area, 3.0, STRUCT_SHORT_ALLOW_MPA)
    lm_5g, lm_sf_5g = record(lm_area, 5.0, STRUCT_SHORT_ALLOW_MPA)
    um_1g, um_sf_1g = record(um_area, 1.0, STRUCT_CREEP_ALLOW_MPA)
    um_3g, um_sf_3g = record(um_area, 3.0, STRUCT_SHORT_ALLOW_MPA)
    um_5g, um_sf_5g = record(um_area, 5.0, STRUCT_SHORT_ALLOW_MPA)
    return {
        "design_mass_kg": STRUCT_DESIGN_MASS_KG,
        "creep_allow_mpa": STRUCT_CREEP_ALLOW_MPA,
        "short_allow_mpa": STRUCT_SHORT_ALLOW_MPA,
        "lm_spoke_area_mm2": lm_area,
        "um_spoke_area_mm2": um_area,
        "lm_stress_1g_mpa": lm_1g,
        "lm_stress_3g_mpa": lm_3g,
        "lm_stress_5g_mpa": lm_5g,
        "um_stress_1g_mpa": um_1g,
        "um_stress_3g_mpa": um_3g,
        "um_stress_5g_mpa": um_5g,
        "lm_sf_1g": lm_sf_1g,
        "lm_sf_3g": lm_sf_3g,
        "lm_sf_5g": lm_sf_5g,
        "um_sf_1g": um_sf_1g,
        "um_sf_3g": um_sf_3g,
        "um_sf_5g": um_sf_5g,
    }


def joint_ear_polygon(owner: str, x: float, clearance: float = 0.0):
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
    boss = Point(x, JOINT_EAR_Y).buffer(JOINT_BOSS_D / 2.0,
                                        resolution=32)
    neck = Point(*contact).buffer(JOINT_NECK_D / 2.0, resolution=32)
    polygon = boss.union(neck).convex_hull
    return polygon.buffer(clearance, resolution=16) if clearance else polygon


def tweeter_joint_polygon(x: float, clearance: float = 0.0):
    """Compact direct ear footprint from the UM ring to the crescent."""
    center = UM_CUTOUT[:2]
    dx, dy = x - center[0], TWEETER_JOINT_Y - center[1]
    length = math.hypot(dx, dy)
    contact = (center[0] + UM_CORE_R * dx / length,
               center[1] + UM_CORE_R * dy / length)
    boss = Point(x, TWEETER_JOINT_Y).buffer(
        TWEETER_JOINT_BOSS_D / 2.0, resolution=32)
    neck = Point(*contact).buffer(
        TWEETER_JOINT_NECK_D / 2.0, resolution=32)
    polygon = boss.union(neck).convex_hull
    return polygon.buffer(clearance, resolution=16) if clearance else polygon


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
    """Fill a lens through one Classic-wall-wide owning fusion land."""
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
    """Rounded LM/UM ear plan, excluding unsupported outboard islands."""
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


def _owned_joint_ear(owner: str, x: float, z_span):
    return _plan_prism(_owned_joint_ear_plan(owner, x), *z_span)


def _joint_receiver_plan(owner: str, x: float):
    """Clear only the opposing ear material that is actually printable."""
    current_owner = "lm" if owner == "um" else "um"
    record = junction_closure_polygons()["lm_um"]
    return _owned_joint_ear_plan(owner, x).buffer(
        JOINT_RECEIVER_RADIAL_CLEAR, resolution=16).difference(
            record[current_owner].intersection(record["target"])).buffer(0)


def _owned_tweeter_joint_plan(owner: str, x: float,
                              clearance: float = 0.0):
    """One T half-lap footprint clipped to its full-depth plan owner."""
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
            side = "left" if normal[0] < 0 else "right"
            vertical = "upper" if normal[1] >= 0 else "lower"
            records.append({
                "name": (f"{key}_{vertical}_{side}"
                         if key == "lm" else f"{key}_{side}"),
                "driver": key,
                "angle_deg": angle, "normal": normal,
                "face": face, "center": center, "radius": radius,
                "clock_from_top_deg": 90.0 - angle,
                "face_offset_mm": face_offset,
                "z_mm": SIDE_MAGNET_Z[key],
                "interface_kind": "ring",
                "magnet_fully_buried": True,
                "local_captive_backing_boss_mm": face_offset,
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
                    # These compatibility fields describe the horizontal
                    # base-side datum; receiver construction must key from
                    # interface_kind rather than treating it as an R113 arc.
                    "center": (0.0, LM_BASE_MAGNET_Y),
                    "radius": LM_BASE_MAGNET_FACE_X,
                    "clock_from_top_deg": 90.0 - angle,
                    "face_offset_mm": 0.0,
                    "z_mm": LM_BASE_MAGNET_Z,
                    "interface_kind": "base_side",
                    "magnet_fully_buried": True,
                    "local_captive_backing_boss_mm": 0.0,
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


def _add_side_magnet_ears(part, driver: str,
                          interface_kinds: set[str] | None = None):
    """Add only the exact local land required by each captive station.

    Existing carrier material absorbs the land at the lower LM sites.  The
    four 2.40-mm ring lips gain only the 0.60-mm local boss needed to reach
    the helper's qualified 3.00-mm land; magnets remain completely buried.
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
        part = _ensure_shell_contained(
            part, tools.required_land,
            f"{site['name']} captive-magnet minimum land")
        if site["interface_kind"] == "ring":
            # The qualified 3.00-mm land ends exactly on the immutable
            # circular flange-seat datum.  Give the later authoritative
            # recess cutter one helper-epsilon of sacrificial material to
            # cross instead of asking OCC to subtract at a tangent/coplanar
            # terminal face.  Translation toward material-inward extends
            # only the construction-side end; the installed interface face
            # and its 0.60-mm local boss do not move.  The final recess recut
            # removes this overlap completely, so released geometry retains
            # the exact recess radius and the full 0.45-mm inner skin.
            overlap = tools.spec.boolean_epsilon_mm
            inward = tools.material_inward_xyz
            sacrificial_land = (
                Pos(*(overlap * component for component in inward))
                * tools.required_land
            )
            part = _ensure_shell_contained(
                part, sacrificial_land,
                f"{site['name']} recess-recut construction overlap")
    return part


def joint_load_facts():
    """Upper-joint direct, moment-contact and M3 screen at 1/3/5g."""
    gravity = 9.80665
    full_half_thickness = min(
        LM_JOINT_Z[1] - LM_JOINT_Z[0],
        UM_JOINT_Z[1] - UM_JOINT_Z[0],
        TWEETER_CORE_JOINT_Z[1] - TWEETER_CORE_JOINT_Z[0],
        TWEETER_ADDON_JOINT_Z[1] - TWEETER_ADDON_JOINT_Z[0],
    )
    tunneled_half_thickness = min(
        full_half_thickness, UM_JOINT_TUNNEL_LIGAMENT)
    pair_thickness = full_half_thickness + tunneled_half_thickness
    neck_width = min(JOINT_NECK_D, TWEETER_JOINT_NECK_D)
    net_width = min(
        JOINT_BOSS_D - JOINT_HOLE_D,
        TWEETER_JOINT_BOSS_D - TWEETER_JOINT_HOLE_D,
    )
    bearing_width = min(JOINT_HOLE_D, TWEETER_JOINT_HOLE_D)
    neck_area = neck_width * pair_thickness
    net_area = net_width * pair_thickness
    bearing_area = bearing_width * pair_thickness
    bolt_shear_area = 2.0 * math.pi * (3.0 / 2.0) ** 2
    m3_tensile_area = math.pi * (2.53 / 2.0) ** 2
    contact_lever = (min(JOINT_BOSS_D, TWEETER_JOINT_BOSS_D)
                     * JOINT_CONTACT_LEVER_FACTOR)
    moment_lever = math.hypot(
        JOINT_PLAN_LEVER_MM, JOINT_REAR_LEVER_MM)
    # Moment contact is governed by the single weaker tunneled ear.
    net_area_per_ear = net_width * tunneled_half_thickness

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
        contact_force_per_ear = moment / (2.0 * contact_lever)
        contact_stress = contact_force_per_ear / net_area_per_ear
        m3_tension_stress = contact_force_per_ear / m3_tensile_area
        return {
            "moment_nmm": moment,
            "contact_force_per_ear_n": contact_force_per_ear,
            "contact_stress_mpa": contact_stress,
            "contact_sf": allowable / contact_stress,
            "m3_tension_stress_mpa": m3_tension_stress,
            "m3_tension_sf": (
                JOINT_M3_TENSION_ALLOW_MPA / m3_tension_stress),
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
        "minimum_half_thickness_mm": tunneled_half_thickness,
        "full_half_thickness_mm": full_half_thickness,
        "tunneled_half_thickness_mm": tunneled_half_thickness,
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


def _cut_lm_mount_holes(part):
    """All six W22 sites remain ordinary blind driver inserts."""
    for x, y in LM_PILOT_XY:
        part -= _cylinder_at(
            x, y, L22_PILOT_D_MM / 2.0,
            LM_SEAT_Z - LM_BORE_DEPTH_MM, LM_SEAT_Z + 0.15)
    return part


def _receiver_notch(part, owner: str, x: float, z_span):
    """Cut a receiver while preserving the current ring's web ownership."""
    z0, z1 = z_span
    notch_plan = _joint_receiver_plan(owner, x)
    return _subtract_plan_prisms(
        part, notch_plan,
        z0 - JOINT_RECEIVER_RADIAL_CLEAR,
        z1 + JOINT_RECEIVER_RADIAL_CLEAR)


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


def _ensure_shell_contained(part, shell, label: str):
    """Fuse only a positive final-shell remainder after all recuts."""
    missing = shell - part
    missing_volume = sum(solid.volume for solid in missing.solids())
    if missing_volume <= 0.01:
        return part
    return _fuse_attached(part, shell, label)


def lm_carrier_outer_blank():
    """Solid LM carrier/tail before the one streamed tunnel-cutter pass."""
    _require_guarded_build()
    cx, cy, cut_d = L22_CUTOUT
    part = _minimal_ring_blank(
        (cx, cy), cut_d / 2.0, LM_RECESS_R, LM_CORE_R, LM_SEAT_Z)
    part = _fuse_attached(
        part, _lm_um_rear_recess_backfill(),
        "symmetric solid rear LM--UM recess-island backfill")

    # Six minimum bosses and short radial spokes replace the deleted annular
    # slab. Each spoke reaches the structural outer lip; the 0.85-mm seat
    # membrane is not credited as the sole insert load path.
    for px, py in LM_PILOT_XY:
        boss = _cylinder_at(
            px, py, PAD_D_MM / 2.0, PAD_FACE_Z, LM_SEAT_Z)
        spoke = _radial_spoke(
            (cx, cy), (px, py), LM_RECESS_R + 0.25,
            LM_STRUCT_SPOKE_W, PAD_FACE_Z, LM_SEAT_Z)
        part = _fuse_attached(
            part, boss.fuse(spoke).clean(), "LM insert boss/spoke")
    part = _cut_lm_mount_holes(part)

    # LM owns the lower full-depth half of the tangent-blended LM--UM
    # closure.  Fuse it before route hollowing so any intentional cable
    # passages are cut from solid material rather than hidden behind a skin.
    part = _fuse_attached(
        part, _junction_closure_web("lm_um", "lm"),
        "LM-owned full-depth LM--UM closure web")
    # Resolve legacy ring material against the complementary plan before the
    # Z-half ears exist; a late mask can slice their outboard tips free.
    part = _enforce_junction_plan_ownership(part, "lm_um", "lm")

    # Establish every small attachment union before carving the long
    # swept tunnel cutters.
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "um", x, UM_JOINT_Z)
    for x in JOINT_EAR_X:
        part += _owned_joint_ear("lm", x, LM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    # One continuous outer sweep per route is fused before the nominal voids
    # are cut.  This keeps every Z bump covered and avoids the old fragmented
    # coplanar rear-floor topology.
    for index, cover in enumerate(route_outer_covers("lm")):
        part = _fuse_attached(
            part, cover, f"LM closed tunnel cover component {index}")

    # Floor features are cut only from their massive owner before that body
    # touches the thin covers.  Each buried floor lane deliberately stops
    # short of its feed during this pre-fusion cut; the normal 8-mm owner
    # cutter below opens the final G1 overlap through solid body material.
    # This avoids both coincident lumen walls and the destructive late T-lane
    # subtraction that otherwise detaches OCC's unrelated main-cover branch.
    if STAND_FOOT:
        floor_body = integrated_floor_addition()
        for index in range(integrated_floor_feature_group_count()):
            floor_body = apply_integrated_floor_feature_group(
                floor_body, index)
        part = _fuse_attached(
            part, floor_body,
            "integral floor stem/foot/NL8 body")
        del floor_body
        part = _fuse_attached(
            part, floor_wing_contact_profile_addition(),
            "universal LM-lower floor wing-contact shoulder")

    # In no-floor mode fuse the shallow solid bridge web before hollowing
    # the combined carrier. Hollowing the carrier and web independently and
    # then fusing their coincident tunnel walls can make OCC discard an
    # otherwise valid
    # 0.8-mm branch.  One cutter pass is set-equivalent, cheaper, and leaves
    # no same-domain internal faces.
    if not STAND_FOOT:
        part = _fuse_attached(
            part, fused_bridge_tail(), "fused no-floor solid bridge web")

    # Both state bodies now own the exact x=+/-32 lower datums.  Add/qualify
    # every captive land only after that massive owner is present: doing so
    # avoids both detached lower tangent solids and making the subsequent
    # large body union operate on the four small ring-land splitters.  Ring
    # interfaces retain only their explicit 0.60-mm local outward boss;
    # lower lands point entirely inward and add no proud material.
    part = _add_side_magnet_ears(part, "lm")

    part = part.clean()
    solids = list(part.solids())
    if (not part.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            f"LM outer blank failed: valid={part.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def apply_lm_route_cutter(part, index):
    """Apply one exact LM cutter group to an imported native outer BREP."""
    _require_guarded_build()
    for cutter in route_inner_cutter_group("lm", index):
        part -= cutter
    part = part.clean()
    solids = list(part.solids())
    if (not part.is_valid or len(solids) != 1
            or solids[0].volume <= 1e-9):
        raise RuntimeError(
            f"LM cutter group {index} detached geometry: "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in part.solids()]}")
    return Part([solids[0]])


def _lm_t_closure_handoff_cutters():
    """Exact phased T lumen through the LM-owned junction-web material.

    The ordinary LM cutter is intentionally cropped to R113 plus its 1-mm
    mouth overtravel.  The new full-depth LM--UM closure owns a small amount
    of material beyond that radial domain, so the radial crop alone leaves a
    thin cap across the otherwise open T handoff.  Reuse the globally phased
    production group with 0.18 mm local mouth relief and clip it to the short
    R111..R117 radial shell; this opens only the intended lumen without
    extending a cover horn or cutting the opposing UM owner.  The relief
    remains inside the existing 0.20-mm middle-wall witness and leaves the
    normal flush-mouth shell intact.
    """
    points = route._owner_cutter_points(
        route.ts_cable_points(1.8), "lm")
    sections = route._tube_section_points(points)
    inside = [
        math.hypot(
            float(point[0]) - L22_CUTOUT[0],
            float(point[1]) - L22_CUTOUT[1]) <= LM_CORE_R
        for point in sections
    ]
    crossings = [
        index for index in range(1, len(inside))
        if inside[index] != inside[index - 1]
    ]
    if not crossings:
        raise RuntimeError("LM T route has no R113 handoff crossing")
    crossing_edge = crossings[-1] - 1
    edge_count = len(sections) - 1
    group_count = route.LM_T_CUTTER_SEGMENT_COUNT
    handoff_group = min(
        group_count - 1,
        crossing_edge * group_count // edge_count)
    handoff_segment = route._round_tube_global_segment(
        points, route.TS_CUTTER_R + LM_T_CLOSURE_HANDOFF_RELIEF_MM,
        handoff_group, group_count)
    outer = _cylinder_at(
        *L22_CUTOUT[:2],
        LM_CORE_R + LM_T_CLOSURE_HANDOFF_RADIAL_OUTSET_MM,
        -50.0, 50.0)
    inner = _cylinder_at(
        *L22_CUTOUT[:2],
        LM_CORE_R - LM_T_CLOSURE_HANDOFF_RADIAL_INSET_MM,
        -50.0, 50.0)
    # Keep only a short shell across the native R113 mouth.  Subtracting the
    # whole global group is set-correct but turns a local flush-mouth repair
    # into an unnecessarily expensive large-carrier Boolean.
    local = ((handoff_segment & outer) - inner).clean()
    solids = tuple(
        solid for solid in local.solids() if solid.volume > 1.0e-8)
    if not local.is_valid or not solids:
        raise RuntimeError(
            "LM T closure handoff cutter missed its full-depth web")
    return solids


def finalize_lm_carrier(part, *, routes_already_cut=False):
    """Hollow and functionally recut one native-BREP LM outer blank."""
    _require_guarded_build()
    cx, cy, cut_d = L22_CUTOUT

    def clean_stage(candidate, label: str):
        candidate = candidate.clean()
        stage_solids = list(candidate.solids())
        if (not candidate.is_valid or len(stage_solids) != 1
                or stage_solids[0].volume <= 0.01):
            raise RuntimeError(
                f"LM {label} invalidated carrier: "
                f"valid={candidate.is_valid} "
                f"volumes={[solid.volume for solid in stage_solids]}")
        return Part([stage_solids[0]])

    if not routes_already_cut:
        for index in range(route_inner_cutter_group_count("lm")):
            part = apply_lm_route_cutter(part, index)
    # The short D7.8 LM lead remains a free cable, not a printed tunnel. The
    # universal no-floor web and integral floor stem nevertheless overlap its
    # immutable centerline, so remove the route authority's minimal rear-open
    # clearance before any captive cavity or functional-interface recut.
    part -= lm_free_lead_relief_cutter()
    part = clean_stage(part, "rear-open free-LM-lead relief")
    # Cut the transverse captive stations while their qualified backing
    # lands still belong to the continuous outer blank.  Recutting the
    # circular flange seat first leaves the ring lands tangent to that seat;
    # OCC can then retain a zero-width same-domain seam when the gable group
    # is subtracted.  The two functional driver cylinders immediately below
    # remain authoritative and remove any construction overlap/fill after
    # the cavities have been established.
    part = _cut_side_magnet_pockets(part, "lm")
    # Reassert the functional driver interfaces after every cover union;
    # no crossover/anchor material may re-enter the flange seat.
    part -= _cylinder_at(cx, cy, LM_RECESS_R,
                         LM_SEAT_Z, THICKNESS_MM + 0.5)
    part = clean_stage(part, "flange-recess recut")
    part -= _cylinder_at(cx, cy, cut_d / 2.0,
                         CORE_REAR_Z - 12.0, THICKNESS_MM + 1.0)
    part = clean_stage(part, "driver-opening recut")
    part = _cut_lm_mount_holes(part)
    part = clean_stage(part, "mount-hole recut")
    # Covers are fused after the first ownership mask.  Reassert the exact
    # opposing-owner relief, fit seam and boss-perimeter drains once all
    # covers/cutters are final so no later route shell can refill them.
    part = _enforce_junction_plan_ownership(part, "lm_um", "lm")
    part = clean_stage(part, "final LM--UM closure ownership recut")
    # The ownership mask above may restore source-owned web material outside
    # the legacy R114 negative-cutter crop.  Reopen the exact T lumen through
    # that local web so the user-visible handoff is flush and uncapped.
    for cutter in _lm_t_closure_handoff_cutters():
        part -= cutter
    part = clean_stage(part, "final LM T closure handoff lumen recut")
    # One final pass is sufficient here: all route, magnet, driver and mount
    # Booleans are already complete. The former consecutive duplicate pass
    # changed no geometry and made every LM build pay for four extra cuts.
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "um", x, UM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    part = clean_stage(part, "final joint receiver recut")

    part = part.clean()
    solids = list(part.solids())
    if (not part.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            f"LM carrier finalization failed: valid={part.is_valid} "
            f"solids={len(solids)}")
    return Part([solids[0]])


def lm_carrier():
    _require_guarded_build()
    return finalize_lm_carrier(lm_carrier_outer_blank())


def um_carrier():
    _require_guarded_build()
    cx, cy, cut_d = UM_CUTOUT
    part = _minimal_ring_blank(
        (cx, cy), cut_d / 2.0, UM_RECESS_R, UM_CORE_R, UM_SEAT_Z)
    part = _fuse_attached(
        part, _um_t_rear_recess_backfill(),
        "symmetric solid rear T-cover/UM-recess backfill")

    um_boss_floor = (
        UM_SEAT_Z - UM_PILOT_DEPTH_MM - UM_PAD_FLOOR_MM)
    for px, py in UM_PILOT_XY:
        boss = _cylinder_at(
            px, py, UM_INSERT_BOSS_D / 2.0,
            um_boss_floor, UM_SEAT_Z)
        spoke = _radial_spoke(
            (cx, cy), (px, py), UM_RECESS_R + 0.25,
            UM_STRUCT_SPOKE_W, um_boss_floor, UM_SEAT_Z)
        part = _fuse_attached(
            part, boss.fuse(spoke).clean(), "UM insert boss/spoke")

    # UM owns the upper half of the LM--UM closure and the lower half of the
    # T--UM closure.  Both are complete z=6.8..18.3 solids; the final route
    # cutter pass alone is allowed to open an intentional cable mouth.
    part = _fuse_attached(
        part, _junction_closure_web("lm_um", "um"),
        "UM-owned full-depth LM--UM closure web")
    part = _fuse_attached(
        part, _junction_closure_web("t_um", "um"),
        "UM-owned full-depth T--UM closure web")
    # Mask only the pre-ear ring/web blank.  The subsequently added Z-half
    # ears retain their complete rounded footprints and remain independently
    # printable without detached slivers.
    part = _enforce_junction_plan_ownership(part, "lm_um", "um")
    part = _enforce_junction_plan_ownership(part, "t_um", "um")

    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "lm", x, LM_JOINT_Z)
    for x in JOINT_EAR_X:
        part += _owned_joint_ear("um", x, UM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    part = _add_side_magnet_ears(part, "um")

    # Rear half of the direct crescent joints.  The complementary upper
    # half is an add-on and is removed from the core with a 0.1-mm plan
    # receiver clearance plus the established 0.2-mm axial split.
    for x in TWEETER_JOINT_X:
        part = _subtract_plan_prisms(
            part,
            _owned_tweeter_joint_plan("tweeter", x, TWEETER_JOINT_CLEAR),
            TWEETER_ADDON_JOINT_Z[0] - TWEETER_JOINT_CLEAR,
            TWEETER_ADDON_JOINT_Z[1] + 0.2)
        part += _plan_prism(
            _owned_tweeter_joint_plan("um", x), *TWEETER_CORE_JOINT_Z)
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_BORE_TOP_Z)
    for px, py in UM_PILOT_XY:
        part -= _cylinder_at(px, py, UM_PILOT_D_MM / 2.0,
                             UM_SEAT_Z - UM_PILOT_DEPTH_MM,
                             UM_SEAT_Z + 0.15)
    for index, cover in enumerate(route_outer_covers("um")):
        part = _fuse_attached(
            part, cover, f"UM closed tunnel cover component {index}")
    for cutter in route_inner_cutters("um"):
        part -= cutter
    part -= _cylinder_at(cx, cy, UM_RECESS_R,
                         UM_SEAT_Z, THICKNESS_MM + 0.5)
    part -= _cylinder_at(cx, cy, cut_d / 2.0,
                         CORE_REAR_Z - 12.0, THICKNESS_MM + 1.0)
    for px, py in UM_PILOT_XY:
        part -= _cylinder_at(px, py, UM_PILOT_D_MM / 2.0,
                             UM_SEAT_Z - UM_PILOT_DEPTH_MM,
                             UM_SEAT_Z + 0.15)
    # Tunnel covers cross both compact joint neighborhoods. One final pass
    # after every cover/cutter and driver recut is authoritative; the former
    # consecutive duplicate passes only repeated identical OCC subtractions.
    part = _enforce_junction_plan_ownership(part, "lm_um", "um")
    part = _enforce_junction_plan_ownership(part, "t_um", "um")
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "lm", x, LM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    for x in TWEETER_JOINT_X:
        part = _subtract_plan_prisms(
            part,
            _owned_tweeter_joint_plan("tweeter", x, TWEETER_JOINT_CLEAR),
            TWEETER_ADDON_JOINT_Z[0] - TWEETER_JOINT_CLEAR,
            TWEETER_ADDON_JOINT_Z[1] + 0.2)
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_BORE_TOP_Z)
    part = _cut_side_magnet_pockets(part, "um")
    part = part.clean()
    solids = list(part.solids())
    if (not part.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            f"UM carrier finalization failed: valid={part.is_valid} "
            f"solids={len(solids)} volumes="
            f"{[solid.volume for solid in solids]} bboxes="
            f"{[((solid.bounding_box().min.X, solid.bounding_box().min.Y,
                  solid.bounding_box().min.Z),
                 (solid.bounding_box().max.X, solid.bounding_box().max.Y,
                  solid.bounding_box().max.Z)) for solid in solids]}")
    return Part([solids[0]])


def core_parts():
    return {
        "core_lm_carrier": lm_carrier(),
        "core_um_carrier": um_carrier(),
    }


def gen_step():
    children = []
    for label, solid in core_parts().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    state = "floor" if STAND_FOOT else "no_floor_fused_solid_web"
    assembly.label = f"lx521_v1lf_r6f_extreme_barebone_core_{state}"
    return assembly


if __name__ == "__main__":
    for name, solid in core_parts().items():
        bb = solid.bounding_box().size
        print(name, f"{bb.X:.2f} x {bb.Y:.2f} x {bb.Z:.2f} mm",
              f"{solid.volume / 1000.0:.2f} cm3", "valid", solid.is_valid)
