"""Extreme Obi-Wan R6F barebone: two flush driver carriers and nothing else.

The previous Obi-Wan was a rear-thinned copy of the complete B2 outline.
This module starts from the irreducible interfaces instead:

* LM carrier: D190 opening, D221.2 flush seat, R113.0 structural lip with a
  smooth R113.80 exterior ring.
* UM carrier: D82 opening, D98.6 flush seat, R51.7 structural lip with a
  smooth R52.50 exterior ring.
* two rounded M3 insert-fastened half-lap ears establish 165.100 mm;
* six fully buried captive magnet interfaces provide four LM and two UM
  alignment sites;
  the original upper LM pair stays at +/-26 degrees from top while the lower
  LM pair mates horizontally through the shared W64 base sides;
* floor mode owns a full-depth integral W64 stem and rectangular floor foot;
  no-floor mode owns a shallow front-flush four-hole solid web;
* the D8.2 UM path is buried only in LM and exits flush at R113 before its
  free span behind UM; D6 is buried only in LM/UM before floating behind the
  crescent; the no-floor UM/T bundle enters through explicit rear-normal
  bores and the LM lead exits through one continuous R14/D9 rear handoff;
* two compact direct half-lap ears attach the optional tweeter crescent.
* tangent-blended LM--UM and T--UM cusp closures are full-depth solids,
  split in plan between their independently printed owners; only the central
  T free-cable mouth and the functional route cuts remain open.

The structural outer lips are only 2.4 mm beyond the flange-recess radii.
A continuous 0.80 mm exterior ring fairing hides the four radial captive
magnet stations without a local boss, flat, notch, or other position cue. The
LM rear insert pads, driver pilots and flush seats remain unchanged. Only
the tweeter crescent is a separate printed add-on.  The floor stem,
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
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def _require_guarded_build():
    """Reject accidental in-process carrier construction outside the guard."""
    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        raise RuntimeError(
            "Obi-Wan carrier construction requires run_memory_guarded.py; "
            "use Make or the staged Obi-Wan exporter")


from top_baffle_nd25fw4 import (
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
import top_baffle_nd25fw4_obiwan_route as route
from top_baffle_nd25fw4_obiwan_route import (
    CROSSOVER_T_Z,
    TS_CUTTER_R,
    TUNNEL_ROOF_SKIN,
    lm_rear_exit_port_cutter,
    no_floor_rear_entry_bores,
    no_floor_rear_entry_bore_cutters,
    no_floor_rear_entry_cap_relief_cutters,
    no_floor_rear_entry_vestibules,
    no_floor_rear_entry_vestibule_cutters,
    route_inner_cutter_group,
    route_inner_cutter_group_count,
    route_inner_cutters,
    route_outer_covers,
)
from top_baffle_nd25fw4_obiwan_bridge import (
    floor_wing_contact_profile_addition,
    fused_bridge_tail,
)
from top_baffle_nd25fw4_obiwan_floor import (
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
# The D5 x 2 captive stack is 3.00 mm deep, while the structural lip is only
# 2.40 mm.  A local flat pad was visibly proud of the circular ring.  Replace
# it with one continuous cylindrical fairing around each exposed carrier
# side.  Only the pre-existing LM--UM and T--UM cusp/service regions retain
# the structural radius;
# every magnet station shares one exact exterior radius, so its location is
# impossible to infer from the outer silhouette.  The cavity datum remains
# 0.15 mm below that surface; the extra skin is intentional and keeps the
# rectangular D5.20+walls land wholly inside the circular fairing even on the
# tighter UM radius.
SIDE_RING_CAVITY_RECESS_CLEAR_MM = 0.05
SIDE_RING_CAVITY_FACE_OFFSET_MM = (
    CAPTIVE_LAND_MM - (LM_CORE_R - LM_RECESS_R)
    + SIDE_RING_CAVITY_RECESS_CLEAR_MM
)
SIDE_RING_FLUSH_FAIRING_MM = 0.80
LM_VISIBLE_RING_R = LM_CORE_R + SIDE_RING_FLUSH_FAIRING_MM
UM_VISIBLE_RING_R = UM_CORE_R + SIDE_RING_FLUSH_FAIRING_MM
SIDE_RING_CAVITY_FACE_INSET_MM = (
    SIDE_RING_FLUSH_FAIRING_MM - SIDE_RING_CAVITY_FACE_OFFSET_MM)
SIDE_RING_FAIRING_FUSION_OVERLAP_MM = 0.10
# Preserve the pre-existing central T cable mouth and T--UM closure geometry.
# The two UM magnet stations lie well outboard at |x| ~= 33.4 mm, so stopping
# only the annular fairing inside this |x| <= 14 mm cusp window cannot create
# a pocket-location cue.
UM_T_FAIRING_CUSP_HALF_WIDTH_MM = 14.0
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
# The 238-degree UM driver-mount boss lies beside the left LM--UM M3 ear.
# A radial spoke at that one site fused into the ear and appeared as an
# unexplained rectangular block next to the heat-set receiver.  Keep the
# driver bolt pattern and its required radial load path, but turn only the
# ear-height section of this spoke 5 degrees toward the free side.  Its lower
# radial root remains only below the UM ear start so the old support does not
# leave a small sealed lower junction void.  The ear-height plan gap is at
# least 0.80 mm; all other mount supports remain exactly radial.
UM_PILOT_SPOKE_TANGENTIAL_OFFSETS_DEG = {238.0: 5.0}
UM_JOINT_EAR_SPOKE_CLEAR_MM = 0.80
# The clear upper support leaves a tiny crescent between the driver recess
# wall and the historic radial contact.  Close it from inside the recess with
# a 0.45-mm terminal land: it is only 1.20 mm deep from the recess rim and
# never reaches the external ring surface or the M3 ear.
UM_PILOT_RECESS_CLOSURE_LAND_EXPANSION_MM = 0.45
UM_PILOT_RECESS_CLOSURE_LAND_DEPTH_MM = 1.20
STRUCT_DESIGN_MASS_KG = 4.0
STRUCT_CREEP_ALLOW_MPA = 8.0
STRUCT_SHORT_ALLOW_MPA = 18.0

# Exactly two LM and one UM magnets per physical side. There are no rear
# attachment bores. The upper LM pair remains radial at +/-26 degrees from the
# top centreline (world polar 116/64 degrees). The lower pair is no longer in
# the R113 lip: it is mirrored through the straight sides of the shared W64
# lower tongue at x=+/-32, y=18, with horizontal outward normals. Both lower
# sites use the same front-biased z=15.10 datum as the upper LM and UM sites,
# so floor/no-floor carriers and the Ac/Ae split prints share one exact
# receiver axis and closure plane. D5x2 magnets are completely
# captive in D5.20 x 2.10 cavities behind 0.45-mm skins, with a printable
# circular cradle and self-supporting 45-degree roof.  The lower pair stays on
# the exact W64 side faces.  The four radial stations sit inside continuous
# smooth R113.80/R52.50 exterior rings.  Their cavity datums are 0.65 mm
# outside the structural lip and 0.15 mm below the visible surface, so the
# qualified 3.00-mm land clears the flange recess while no local backing pad
# breaks the circular silhouette.  Magnets carry alignment/anti-rattle load
# only.  The monolithic W64 stem/root carries floor load; the four stock
# bridge holes carry no-floor load.
SIDE_EAR_D = 7.8
SIDE_EAR_IN = 0.7
SIDE_EAR_OUT = 3.3
SIDE_MAGNET_D = 5.0
SIDE_MAGNET_POCKET_D = CAVITY_DIAMETER_MM
SIDE_MAGNET_DEPTH = CAVITY_DEPTH_MM
SIDE_MAGNET_CAPTIVE_LAND = CAPTIVE_LAND_MM
SIDE_MAGNET_FACE_SKIN = FACE_SKIN_MM
SIDE_MAGNET_INNER_SKIN = INNER_SKIN_MM
OBIWAN_MAGNET_Z_MM = 15.10
SIDE_MAGNET_Z = {
    "lm": OBIWAN_MAGNET_Z_MM,
    "um": OBIWAN_MAGNET_Z_MM,
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
    # Cavity construction datums, not visible interface faces.  Both are
    # recessed 0.15 mm below their continuous circular outer surfaces.
    "lm": SIDE_RING_CAVITY_FACE_OFFSET_MM,
    "um": SIDE_RING_CAVITY_FACE_OFFSET_MM,
}
LM_BASE_MAGNET_FACE_X = 32.0
LM_BASE_MAGNET_Y = 18.0
LM_BASE_MAGNET_Z = OBIWAN_MAGNET_Z_MM

# Two compact Z-axis fastened *rounded ears*.  The old rectangular tabs
# projected into the MU flange seating region.  Each replacement is a
# D9 base-web boss convex-hulled to a D4 contact neck on its owning ring;
# the final Z-owned functional boss is locally D9.8 as documented below.
# The complete footprint (not only its M3 axis) clears both recess discs.
# LM owns the complete rear ear and rear-driven screw-clearance passage. UM
# owns the complete front ear and a rear-opening blind M3 heat-set receiver.
# Neither functional boss is allowed to inherit the full-depth closure-web
# plan seam: the non-owning carrier receives the complete opposing-ear notch,
# so each insert/bore can be prepared in its individual print. Their 0.20 mm
# Z air gap prevents the two prints fusing during assembly.
# A 7.00-mm outward shift from the concept nominal clears the buried T route
# and the complementary half-lap receiver envelope.  The D4 neck and D9
# closure-base boss stay nominal; only the Z-owned functional cylinder grows
# locally to D9.8, and the two halves remain clear by Z separation.
JOINT_EAR_X = (-32.0, 32.0)
JOINT_EAR_Y = 315.770102
JOINT_CLEARANCE_BORE_D = 3.4
JOINT_INSERT_BORE_D = 4.6
JOINT_INSERT_DEPTH_MM = 4.0
# The closure-web base plan keeps the compact historical D9 teardrop. The
# complete Z-owned functional ear locally grows only its cylindrical boss to
# D9.8: a D4.6 heat-set receiver inside D9 would reduce the existing 5g
# moment-contact screen below its accepted factor. The extra 0.40-mm radius is
# local to the two axial half-ears and remains clear of both flange seats,
# cable routes, and all captive-magnet lands.
JOINT_BOSS_D = 9.0
JOINT_FUNCTIONAL_BOSS_D = 9.8
JOINT_NECK_D = 4.0
JOINT_BORE_REAR_OVERSHOOT = 0.01
LM_JOINT_Z = (CORE_REAR_Z, 12.20)
UM_JOINT_Z = (12.40, THICKNESS_MM)
# The original radial 238-degree support closes the lower receiver perimeter,
# but must hand over to the deflected support before the UM's standalone M3
# ear begins.  The two support plans never coexist at one Z plane: a radial
# lower section runs to this datum and the clear deflected section starts
# there.  That avoids both the visible ear bridge and the sealed wedge that a
# lower-Z fan of both supports would create.  The 0.01-mm offset resolves to
# a full 0.16-mm front-down Bambu layer separation: the last radial layer is
# at z=12.34 and the first ear layer at z=12.50.
UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z = UM_JOINT_Z[0] - 0.01
JOINT_CLEARANCE_BORE_TOP_Z = LM_JOINT_Z[1] + 0.35
JOINT_INSERT_BORE_Z = (
    LM_JOINT_Z[1], UM_JOINT_Z[0] + JOINT_INSERT_DEPTH_MM)
JOINT_INSERT_FRONT_FLOOR_MM = THICKNESS_MM - JOINT_INSERT_BORE_Z[1]
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
# As at LM--UM, the full-depth closure web keeps the compact historical D9
# footprint, while each independently printed functional half-ear restores a
# complete D9.8 boss.  The crescent half contains a D4.6 heat-set receiver;
# evaluating or printing that receiver inside D9 would miss the accepted 5g
# moment-contact screen even if its wall were not split by the plan seam.
TWEETER_JOINT_FUNCTIONAL_BOSS_D = 9.8
TWEETER_JOINT_NECK_D = 4.0
TWEETER_JOINT_HOLE_D = 3.4
TWEETER_JOINT_INSERT_BORE_D = 4.6
TWEETER_JOINT_INSERT_DEPTH_MM = 4.0
TWEETER_CORE_JOINT_Z = (CORE_REAR_Z, 12.20)
TWEETER_ADDON_JOINT_Z = (12.40, THICKNESS_MM)
TWEETER_JOINT_CLEAR = 0.10
TWEETER_JOINT_INSERT_BORE_Z = (
    TWEETER_CORE_JOINT_Z[1],
    TWEETER_ADDON_JOINT_Z[0] + TWEETER_JOINT_INSERT_DEPTH_MM,
)
TWEETER_JOINT_INSERT_FRONT_FLOOR_MM = (
    THICKNESS_MM - TWEETER_JOINT_INSERT_BORE_Z[1])
# Keep the rear M3 through-bore continuous into the add-on's blind insert
# receiver, but do not terminate its cutter on a real 0.20/0.16-mm Bambu
# layer.  The former +0.30 endpoint was exactly world Z=12.50; an OCC section
# through that coincident cap retained one mirrored bore and closed the other.
# The extra 0.05 mm remains inside the existing 0.35-mm axial overlap with
# the add-on receiver and makes the Z=12.50 print layer unambiguously open.
TWEETER_CORE_BORE_TOP_Z = TWEETER_CORE_JOINT_Z[1] + 0.35

# Full-depth, plan-split closure webs replace the two pairs of open cusp
# islands at the LM--UM and UM--tweeter junctions.  These are deliberately
# not thin front skins: every owner occupies the complete Obi-Wan depth so a
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
T_UM_WEB_BLEND_START_X = UM_T_FAIRING_CUSP_HALF_WIDTH_MM

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
# outside the existing B2/Obi-Wan envelope.
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


def side_ring_outer_plan(driver: str):
    """Visible ring plan with its fairing stopped outside interface cusps.

    The fairing is continuous around every exposed side and every magnet
    station.  The pre-existing LM--UM and T--UM cusp/service regions retain
    the structural radii, preventing independently printed owners from
    acquiring an overlapping annular lens or a capped cable mouth.
    """
    if driver == "lm":
        center = L22_CUTOUT[:2]
        nominal_r = LM_CORE_R
        visible_r = LM_VISIBLE_RING_R
        clip = box(
            center[0] - visible_r - 1.0,
            center[1] - visible_r - 1.0,
            center[0] + visible_r + 1.0,
            center[1] + nominal_r,
        )
    elif driver == "um":
        center = UM_CUTOUT[:2]
        nominal_r = UM_CORE_R
        visible_r = UM_VISIBLE_RING_R
        clip = box(
            center[0] - visible_r - 1.0,
            center[1] - nominal_r,
            center[0] + visible_r + 1.0,
            center[1] + visible_r + 1.0,
        )
        top_cusp = box(
            center[0] - UM_T_FAIRING_CUSP_HALF_WIDTH_MM,
            center[1],
            center[0] + UM_T_FAIRING_CUSP_HALF_WIDTH_MM,
            center[1] + visible_r + 1.0,
        )
    else:
        raise ValueError(driver)
    nominal = Point(*center).buffer(nominal_r, resolution=256)
    fairing = Point(*center).buffer(
        visible_r, resolution=256).intersection(clip)
    if driver == "um":
        fairing = fairing.difference(top_cusp)
    return nominal.union(fairing).buffer(0)


def _side_ring_fairing(driver: str):
    """Exact cylindrical fairing with a complementary cusp stop."""
    if driver == "lm":
        center = L22_CUTOUT[:2]
        nominal_r = LM_CORE_R
        visible_r = LM_VISIBLE_RING_R
        clip_plan = box(
            center[0] - visible_r - 1.0,
            center[1] - visible_r - 1.0,
            center[0] + visible_r + 1.0,
            center[1] + nominal_r,
        )
    elif driver == "um":
        center = UM_CUTOUT[:2]
        nominal_r = UM_CORE_R
        visible_r = UM_VISIBLE_RING_R
        clip_plan = box(
            center[0] - visible_r - 1.0,
            center[1] - nominal_r,
            center[0] + visible_r + 1.0,
            center[1] + visible_r + 1.0,
        )
        top_cusp = box(
            center[0] - UM_T_FAIRING_CUSP_HALF_WIDTH_MM,
            center[1],
            center[0] + UM_T_FAIRING_CUSP_HALF_WIDTH_MM,
            center[1] + visible_r + 1.0,
        )
    else:
        raise ValueError(driver)
    fairing = _cylinder_at(
        *center, visible_r, CORE_REAR_Z, THICKNESS_MM)
    fairing -= _cylinder_at(
        *center,
        nominal_r - SIDE_RING_FAIRING_FUSION_OVERLAP_MM,
        CORE_REAR_Z - 0.2,
        THICKNESS_MM + 0.2,
    )
    fairing &= _plan_prism(
        clip_plan, CORE_REAR_Z - 0.2, THICKNESS_MM + 0.2)
    if driver == "um":
        fairing -= _plan_prism(
            top_cusp, CORE_REAR_Z - 0.2, THICKNESS_MM + 0.2)
    fairing = fairing.clean()
    solids = list(fairing.solids())
    if (not fairing.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            f"{driver} smooth ring fairing must be one valid solid; "
            f"valid={fairing.is_valid} "
            f"volumes={[solid.volume for solid in solids]}")
    return Part([solids[0]])


def _radial_spoke_plan(center, pilot_xy, contact_radius, width,
                       tangent_offset_deg: float = 0.0):
    """Plan for a structural driver-mount spoke.

    Most supports are precisely radial.  The sole nearby LM--UM interface
    support uses a tiny tangential correction so it remains an independent
    driver-mount load path rather than accidentally welding itself to the M3
    heat-set ear.
    """
    dx = pilot_xy[0] - center[0]
    dy = pilot_xy[1] - center[1]
    angle = math.atan2(dy, dx) + math.radians(tangent_offset_deg)
    contact = (
        center[0] + contact_radius * math.cos(angle),
        center[1] + contact_radius * math.sin(angle),
    )
    return LineString((pilot_xy, contact)).buffer(
        width / 2.0, resolution=24, cap_style=1, join_style=1)


def _radial_spoke(center, pilot_xy, contact_radius, width, z0, z1,
                  tangent_offset_deg: float = 0.0):
    return _plan_prism(
        _radial_spoke_plan(
            center, pilot_xy, contact_radius, width,
            tangent_offset_deg),
        z0, z1)


def um_pilot_spoke_z_segments(pilot_angle: float, z0: float, z1: float):
    """Return non-overlapping (tangent offset, z0, z1) UM support spans.

    The 238-degree support changes its plan exactly below the adjacent
    standalone M3 ear.  It must be a Z handoff, not a lower-Z union of two
    diverging spokes: that fan would enclose a non-printable internal wedge
    against the UM recess boundary.
    """
    if z1 <= z0:
        raise ValueError("UM pilot support span must have positive depth")
    tangent_offset = UM_PILOT_SPOKE_TANGENTIAL_OFFSETS_DEG.get(
        pilot_angle, 0.0)
    if not tangent_offset:
        return ((0.0, z0, z1),)
    transition = UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z
    if not z0 < transition < z1:
        raise ValueError(
            "UM pilot spoke transition must lie inside its support span")
    return (
        (0.0, z0, transition),
        (tangent_offset, transition, z1),
    )


def _um_pilot_recess_closure_land(center, pilot_xy):
    """Small hidden outer-recess land closing the clear-spoke crescent.

    This is deliberately a terminal slice of the former radial load path,
    expanded only by one extrusion width.  It sits wholly below the driver
    flange seat and within the recess/ring interface; it is not an external
    boss, tab, or cue for either the driver insert or the adjacent M3 ear.
    """
    radial = _radial_spoke_plan(
        center, pilot_xy, UM_RECESS_R + 0.25, UM_STRUCT_SPOKE_W)
    outer = Point(*center).buffer(UM_RECESS_R + 0.25, resolution=128)
    inner = Point(*center).buffer(
        UM_RECESS_R - UM_PILOT_RECESS_CLOSURE_LAND_DEPTH_MM,
        resolution=128)
    land = radial.buffer(
        UM_PILOT_RECESS_CLOSURE_LAND_EXPANSION_MM,
        resolution=24).intersection(outer).difference(inner).buffer(0)
    if land.is_empty or not land.is_valid or land.area <= 0.01:
        raise RuntimeError("UM pilot recess closure land is invalid")
    return land


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


def joint_ear_polygon(owner: str, x: float, clearance: float = 0.0,
                      *, boss_d: float = JOINT_BOSS_D):
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
    boss = Point(x, JOINT_EAR_Y).buffer(boss_d / 2.0,
                                        resolution=32)
    neck = Point(*contact).buffer(JOINT_NECK_D / 2.0, resolution=32)
    polygon = boss.union(neck).convex_hull
    return polygon.buffer(clearance, resolution=16) if clearance else polygon


def tweeter_joint_polygon(x: float, clearance: float = 0.0,
                          *, boss_d: float = TWEETER_JOINT_BOSS_D):
    """Compact direct ear footprint from the UM ring to the crescent."""
    center = UM_CUTOUT[:2]
    dx, dy = x - center[0], TWEETER_JOINT_Y - center[1]
    length = math.hypot(dx, dy)
    contact = (center[0] + UM_CORE_R * dx / length,
               center[1] + UM_CORE_R * dy / length)
    boss = Point(x, TWEETER_JOINT_Y).buffer(
        boss_d / 2.0, resolution=32)
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
    """Diagnostic closure-clipped LM/UM ear plan.

    This is the footprint contributed by the full-depth closure-web base
    before the functional Z-owned joint is restored.  It is deliberately not
    the printable insert/bore authority: using it for a finished ear splits
    the cylindrical wall between the two independently printed carriers.
    """
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


def _complete_joint_ear_plan(owner: str, x: float,
                             clearance: float = 0.0):
    """Complete standalone LM/UM ear footprint, never split by the web seam."""
    return joint_ear_polygon(
        owner, x, clearance, boss_d=JOINT_FUNCTIONAL_BOSS_D)


def _complete_joint_ear(owner: str, x: float):
    """Complete Z-owned functional ear for one independently printed carrier."""
    if owner not in {"lm", "um"}:
        raise ValueError(owner)
    z_span = LM_JOINT_Z if owner == "lm" else UM_JOINT_Z
    return _plan_prism(_complete_joint_ear_plan(owner, x), *z_span)


def _owned_tweeter_joint_plan(owner: str, x: float,
                              clearance: float = 0.0):
    """Diagnostic closure-clipped T--UM base-ear footprint.

    The result describes only material contributed by the full-depth closure
    web.  It is deliberately not the printable bore/insert authority: using
    this plan for a finished half-ear lets the plan seam bisect the receiver.
    """
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


def _complete_tweeter_joint_ear_plan(owner: str, x: float,
                                      clearance: float = 0.0):
    """Complete standalone T--UM ear, never split by closure ownership."""
    if owner not in {"um", "tweeter"}:
        raise ValueError(owner)
    return tweeter_joint_polygon(
        x, clearance, boss_d=TWEETER_JOINT_FUNCTIONAL_BOSS_D)


def _complete_tweeter_joint_ear(owner: str, x: float):
    """Complete Z-owned functional ear for one independently printed part."""
    if owner not in {"um", "tweeter"}:
        raise ValueError(owner)
    z_span = (TWEETER_CORE_JOINT_Z if owner == "um"
              else TWEETER_ADDON_JOINT_Z)
    return _plan_prism(
        _complete_tweeter_joint_ear_plan(owner, x), *z_span)


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


def joint_load_facts():
    """Upper-joint direct, moment-contact and M3 screen at 1/3/5g."""
    gravity = 9.80665
    full_half_thickness = min(
        LM_JOINT_Z[1] - LM_JOINT_Z[0],
        UM_JOINT_Z[1] - UM_JOINT_Z[0],
        TWEETER_CORE_JOINT_Z[1] - TWEETER_CORE_JOINT_Z[0],
        TWEETER_ADDON_JOINT_Z[1] - TWEETER_ADDON_JOINT_Z[0],
    )
    # The outward-shifted complete functional ears do not intersect the T
    # lumen (the exact cutter remains 0.445 mm away). Therefore the true
    # minimum axial ear thickness is 5.4 mm; the nearby 5.35-mm route-to-front
    # ligament is reported separately and receives no fictitious ear penalty.
    minimum_half_thickness = full_half_thickness
    pair_thickness = 2.0 * minimum_half_thickness
    neck_width = min(JOINT_NECK_D, TWEETER_JOINT_NECK_D)
    # The LM--UM interface is now insert-fastened, so its governing printed
    # annulus is set by the D4.6 receiver rather than the D3.4 screw passage.
    # Evaluate both reinforced D9.8 functional bosses separately for moment
    # contact.  In particular, the crescent half is governed by its D4.6
    # insert receiver, not by the core half's smaller D3.4 screw passage.
    lm_um_net_width = JOINT_FUNCTIONAL_BOSS_D - JOINT_INSERT_BORE_D
    tweeter_net_width = (
        TWEETER_JOINT_FUNCTIONAL_BOSS_D
        - TWEETER_JOINT_INSERT_BORE_D)
    net_width = min(lm_um_net_width, tweeter_net_width)
    bearing_width = min(
        JOINT_INSERT_BORE_D, TWEETER_JOINT_HOLE_D)
    neck_area = neck_width * pair_thickness
    net_area = net_width * pair_thickness
    bearing_area = bearing_width * pair_thickness
    bolt_shear_area = 2.0 * math.pi * (3.0 / 2.0) ** 2
    m3_tensile_area = math.pi * (2.53 / 2.0) ** 2
    contact_levers = {
        "lm_um": (JOINT_FUNCTIONAL_BOSS_D
                  * JOINT_CONTACT_LEVER_FACTOR),
        "um_tweeter": (TWEETER_JOINT_FUNCTIONAL_BOSS_D
                       * JOINT_CONTACT_LEVER_FACTOR),
    }
    contact_lever = min(contact_levers.values())
    moment_lever = math.hypot(
        JOINT_PLAN_LEVER_MM, JOINT_REAR_LEVER_MM)
    # Moment contact is governed by the single weaker ear in each interface.
    net_area_per_ear = {
        "lm_um": lm_um_net_width * minimum_half_thickness,
        "um_tweeter": tweeter_net_width * minimum_half_thickness,
    }

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
        interfaces = {}
        for name in ("lm_um", "um_tweeter"):
            contact_force = moment / (2.0 * contact_levers[name])
            contact_stress = contact_force / net_area_per_ear[name]
            interfaces[name] = {
                "contact_lever_mm": contact_levers[name],
                "net_area_per_ear_mm2": net_area_per_ear[name],
                "contact_force_per_ear_n": contact_force,
                "contact_stress_mpa": contact_stress,
                "contact_sf": allowable / contact_stress,
            }
        governing_name = min(
            interfaces, key=lambda name: interfaces[name]["contact_sf"])
        governing = interfaces[governing_name]
        contact_force_per_ear = max(
            facts["contact_force_per_ear_n"]
            for facts in interfaces.values())
        m3_tension_stress = contact_force_per_ear / m3_tensile_area
        return {
            "moment_nmm": moment,
            "governing_interface": governing_name,
            "contact_force_per_ear_n": contact_force_per_ear,
            "contact_stress_mpa": governing["contact_stress_mpa"],
            "contact_sf": governing["contact_sf"],
            "m3_tension_stress_mpa": m3_tension_stress,
            "m3_tension_sf": (
                JOINT_M3_TENSION_ALLOW_MPA / m3_tension_stress),
            "lm_um_insert_pullout_required_n": (
                interfaces["lm_um"]["contact_force_per_ear_n"]),
            "um_tweeter_insert_pullout_required_n": (
                interfaces["um_tweeter"]["contact_force_per_ear_n"]),
            "interfaces": interfaces,
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
        "lm_um_functional_boss_d_mm": JOINT_FUNCTIONAL_BOSS_D,
        "um_tweeter_functional_boss_d_mm": (
            TWEETER_JOINT_FUNCTIONAL_BOSS_D),
        "lm_um_net_width_mm": lm_um_net_width,
        "um_tweeter_net_width_mm": tweeter_net_width,
        "minimum_half_thickness_mm": minimum_half_thickness,
        "full_half_thickness_mm": full_half_thickness,
        "nearby_route_front_ligament_mm": UM_JOINT_TUNNEL_LIGAMENT,
        "lm_um_clearance_bore_d_mm": JOINT_CLEARANCE_BORE_D,
        "lm_um_insert_receiver_d_mm": JOINT_INSERT_BORE_D,
        "lm_um_insert_receiver_depth_mm": JOINT_INSERT_DEPTH_MM,
        "lm_um_insert_front_floor_mm": JOINT_INSERT_FRONT_FLOOR_MM,
        "lm_um_axial_gap_mm": UM_JOINT_Z[0] - LM_JOINT_Z[1],
        "lm_um_bore_overlap_mm": (
            JOINT_CLEARANCE_BORE_TOP_Z - JOINT_INSERT_BORE_Z[0]),
        "lm_um_standalone_ear_ownership_required": True,
        "lm_um_full_360_wall_required": True,
        "lm_um_cross_owner_material_allowed": False,
        "lm_um_insert_pullout_qualification_required": True,
        "um_tweeter_clearance_bore_d_mm": TWEETER_JOINT_HOLE_D,
        "um_tweeter_insert_receiver_d_mm": TWEETER_JOINT_INSERT_BORE_D,
        "um_tweeter_insert_receiver_depth_mm": (
            TWEETER_JOINT_INSERT_DEPTH_MM),
        "um_tweeter_insert_front_floor_mm": (
            TWEETER_JOINT_INSERT_FRONT_FLOOR_MM),
        "um_tweeter_axial_gap_mm": (
            TWEETER_ADDON_JOINT_Z[0] - TWEETER_CORE_JOINT_Z[1]),
        "um_tweeter_bore_overlap_mm": (
            TWEETER_CORE_BORE_TOP_Z
            - TWEETER_JOINT_INSERT_BORE_Z[0]),
        "um_tweeter_standalone_ear_ownership_required": True,
        "um_tweeter_full_360_wall_required": True,
        "um_tweeter_cross_owner_material_allowed": False,
        "um_tweeter_insert_pullout_qualification_required": True,
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


def _complete_joint_receiver_notch(part, ear_owner: str, x: float):
    """Remove the full opposing ear so the functional boss has one owner.

    The notch spans the complete 0.20-mm axial air gap as well as the opposing
    ear.  This prevents either full-depth closure web from surviving across
    the nominal gap or bisecting the standalone bore/insert annulus.
    """
    if ear_owner == "lm":
        z0 = LM_JOINT_Z[0] - JOINT_RECEIVER_RADIAL_CLEAR
        z1 = UM_JOINT_Z[0]
    elif ear_owner == "um":
        z0 = LM_JOINT_Z[1]
        z1 = UM_JOINT_Z[1] + JOINT_RECEIVER_RADIAL_CLEAR
    else:
        raise ValueError(ear_owner)
    return _subtract_plan_prisms(
        part,
        _complete_joint_ear_plan(
            ear_owner, x, JOINT_RECEIVER_RADIAL_CLEAR),
        z0, z1)


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


def _apply_complete_lm_um_joint(part, owner: str):
    """Restore one carrier's complete ear and cut its standalone fastener void."""
    if owner not in {"lm", "um"}:
        raise ValueError(owner)
    opposing = "um" if owner == "lm" else "lm"
    for x in JOINT_EAR_X:
        part = _complete_joint_receiver_notch(part, opposing, x)
    for x in JOINT_EAR_X:
        part = _fuse_attached(
            part, _complete_joint_ear(owner, x),
            f"{owner.upper()} complete standalone joint ear {x:+.1f}")
        if owner == "lm":
            part -= _cylinder_at(
                x, JOINT_EAR_Y, JOINT_CLEARANCE_BORE_D / 2.0,
                CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                JOINT_CLEARANCE_BORE_TOP_Z)
        else:
            part -= _cylinder_at(
                x, JOINT_EAR_Y, JOINT_INSERT_BORE_D / 2.0,
                *JOINT_INSERT_BORE_Z)
    return part


def _complete_tweeter_joint_receiver_notch(
        part, ear_owner: str, x: float):
    """Remove one complete opposing T ear plus the 0.20-mm axial gap."""
    if ear_owner == "um":
        z0 = TWEETER_CORE_JOINT_Z[0] - TWEETER_JOINT_CLEAR
        z1 = TWEETER_ADDON_JOINT_Z[0]
    elif ear_owner == "tweeter":
        z0 = TWEETER_CORE_JOINT_Z[1]
        z1 = TWEETER_ADDON_JOINT_Z[1] + TWEETER_JOINT_CLEAR
    else:
        raise ValueError(ear_owner)
    return _subtract_plan_prisms(
        part,
        _complete_tweeter_joint_ear_plan(
            ear_owner, x, TWEETER_JOINT_CLEAR),
        z0, z1)


def _apply_complete_um_tweeter_joint(part, owner: str):
    """Restore complete standalone T ears and their owner-specific voids."""
    if owner not in {"um", "tweeter"}:
        raise ValueError(owner)
    opposing = "tweeter" if owner == "um" else "um"
    for x in TWEETER_JOINT_X:
        part = _complete_tweeter_joint_receiver_notch(part, opposing, x)
    for x in TWEETER_JOINT_X:
        part = _fuse_attached(
            part, _complete_tweeter_joint_ear(owner, x),
            f"{owner.upper()} complete standalone tweeter joint ear "
            f"{x:+.1f}")
        # Preserve the rear-driven screw passage across the true 0.20-mm
        # inter-part gap.  On the crescent it then widens into the D4.6 blind
        # heat-set receiver; the larger cutter deliberately starts at the
        # rear owner's z=12.2 termination for a 0.35-mm service overlap.
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_BORE_TOP_Z)
        if owner == "tweeter":
            part -= _cylinder_at(
                x, TWEETER_JOINT_Y,
                TWEETER_JOINT_INSERT_BORE_D / 2.0,
                *TWEETER_JOINT_INSERT_BORE_Z)
    return part


def _ensure_shell_contained(part, shell, label: str):
    """Fuse only a positive final-shell remainder after all recuts."""
    missing = shell - part
    missing_volume = sum(solid.volume for solid in missing.solids())
    if missing_volume <= 0.01:
        return part
    return _fuse_attached(part, shell, label)


def _no_floor_cover_remainders(cover, bridge, label: str):
    """Drop only cover volume already owned by the solid no-floor bridge.

    The bridge and each swept cover intentionally describe the same material
    above the rear plane.  Asking OCC to fuse either complete, deeply
    overlapping BREP after the other can leave invalid same-domain internal
    faces even though the mathematical union is one solid.  Remove only the
    subset that is already inside the *actual bridge solid*, beginning 0.55
    mm above its rear face.  The retained rear band therefore overlaps the
    bridge by positive volume, while any conduit shoulder outside the bridge
    silhouette remains untouched.  Since the discarded set is a strict
    subset of ``bridge``, this is exactly set-equivalent to the full union.
    """
    if STAND_FOOT:
        return tuple(cover.solids())
    bridge_box = bridge.bounding_box()
    trim_z0 = PAD_FACE_Z + route.TUNNEL_FUSE_OVERLAP
    trim_z1 = bridge_box.max.Z + 1.0
    trim = Pos(
        (bridge_box.min.X + bridge_box.max.X) / 2.0,
        (bridge_box.min.Y + bridge_box.max.Y) / 2.0,
        (trim_z0 + trim_z1) / 2.0,
    ) * Box(
        bridge_box.max.X - bridge_box.min.X + 2.0,
        bridge_box.max.Y - bridge_box.min.Y + 2.0,
        trim_z1 - trim_z0,
    )
    redundant = (bridge & trim).clean()
    remainder = (cover - redundant).clean()
    solids = tuple(remainder.solids())
    if (not remainder.is_valid or not solids
            or any(solid.volume <= 0.01 for solid in solids)):
        raise RuntimeError(
            f"{label}: bridge-owned cover reduction failed; "
            f"valid={remainder.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return solids


def lm_carrier_outer_blank():
    """Solid LM carrier/tail before the one streamed tunnel-cutter pass."""
    _require_guarded_build()
    cx, cy, cut_d = L22_CUTOUT
    part = _minimal_ring_blank(
        (cx, cy), cut_d / 2.0, LM_RECESS_R, LM_CORE_R, LM_SEAT_Z)
    part = _fuse_attached(
        part, _side_ring_fairing("lm"),
        "continuous smooth LM side-ring fairing")
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
    part = _apply_complete_lm_um_joint(part, "lm")

    # In no-floor mode fuse the massive shallow bridge before adding the thin
    # route covers.  Fusing it afterward makes the R113.8-flush cover and the
    # bridge shoulder share several nearly coincident faces; OCC can return a
    # single-volume but invalid shell.  The one streamed cutter pass remains
    # after every union, so this ordering change is set-equivalent and still
    # leaves no coincident internal tunnel walls.
    bridge = None
    if not STAND_FOOT:
        bridge = fused_bridge_tail()
        part = _fuse_attached(
            part, bridge, "fused no-floor solid bridge web")

    # One continuous outer sweep per route is fused before the nominal voids
    # are cut.  This keeps every Z bump covered and avoids the old fragmented
    # coplanar rear-floor topology.
    for index, cover in enumerate(route_outer_covers("lm")):
        remainders = (
            _no_floor_cover_remainders(
                cover, bridge, f"LM closed tunnel cover component {index}")
            if bridge is not None else (cover,))
        for subindex, remainder in enumerate(remainders):
            part = _fuse_attached(
                part, Part([remainder]),
                f"LM closed tunnel cover component {index}.{subindex}")

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

    # Both state bodies now own the exact x=+/-32 lower datums.  Add/qualify
    # every captive land only after that massive owner is present: doing so
    # avoids both detached lower tangent solids and making the subsequent
    # large body union operate on the four small ring-land splitters.  Ring
    # interfaces are contained by one continuous smooth R113.80 ring;
    # lower lands point entirely inward and add no proud material.
    part = _verify_side_magnet_lands(part, "lm")

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
        significant = [
            solid for solid in stage_solids if solid.volume > 1.0e-4]
        dust = [
            solid for solid in stage_solids if solid.volume <= 1.0e-4]
        if (len(significant) == 1 and significant[0].is_valid
                and sum(solid.volume for solid in dust) <= 1.0e-4):
            # OCC can leave an invalid, disconnected numerical needle where
            # two already-overlapping void cutters meet.  It is neither part
            # of the printable carrier nor large enough to tessellate.  Keep
            # the independently valid functional body and still fail closed
            # for every material component above 0.0001 mm3.
            return Part([significant[0]])
        if (not candidate.is_valid or len(stage_solids) != 1
                or stage_solids[0].volume <= 0.01):
            stage_bounds = [(
                solid.bounding_box().min.X,
                solid.bounding_box().min.Y,
                solid.bounding_box().min.Z,
                solid.bounding_box().max.X,
                solid.bounding_box().max.Y,
                solid.bounding_box().max.Z,
            ) for solid in stage_solids]
            raise RuntimeError(
                f"LM {label} invalidated carrier: "
                f"valid={candidate.is_valid} "
                f"volumes={[solid.volume for solid in stage_solids]} "
                f"bounds={stage_bounds}")
        return Part([stage_solids[0]])

    if not routes_already_cut:
        for index in range(route_inner_cutter_group_count("lm")):
            part = apply_lm_route_cutter(part, index)
    # No-floor UM/T feeds use a rear-normal bore, a hidden spherical
    # vestibule, and (for UM) one short octagonal start-cap relief.  These
    # volumes intentionally overlap.  Subtracting them sequentially can leave
    # an invalid equal-radius shared edge whose occurrence depends on distant
    # LM face numbering.  Fuse the exact tools first, validate that union, and
    # make one Boolean cut per cable.  Dimensions and the resulting void set
    # are unchanged; only the intermediate topology is removed.
    if not STAND_FOOT:
        bore_by_name = {
            entry.name: cutter
            for entry, cutter in zip(
                no_floor_rear_entry_bores(),
                no_floor_rear_entry_bore_cutters(), strict=True)
            if entry.name != "lm"
        }
        vestibule_by_name = {
            entry.name: cutter
            for entry, cutter in zip(
                no_floor_rear_entry_vestibules(),
                no_floor_rear_entry_vestibule_cutters(), strict=True)
        }
        um_reliefs = tuple(no_floor_rear_entry_cap_relief_cutters())
        for entry_name in ("t", "um"):
            tools = [bore_by_name[entry_name],
                     vestibule_by_name[entry_name]]
            if entry_name == "um":
                tools.extend(um_reliefs)
            transition = tools[0].fuse(*tools[1:]).clean()
            transition_solids = list(transition.solids())
            if (not transition.is_valid or len(transition_solids) != 1
                    or not transition_solids[0].is_valid
                    or transition_solids[0].volume <= 0.01):
                raise RuntimeError(
                    f"LM {entry_name.upper()} entry transition tool invalid: "
                    f"valid={transition.is_valid} volumes="
                    f"{[solid.volume for solid in transition_solids]}")
            part -= transition
            part = clean_stage(
                part, f"{entry_name.upper()} fused rear entry transition")
    # The D7.8 LM pair remains external after leaving the carrier.  Floor mode
    # recuts only the shared R14/D9 handoff already owned by its integral lane;
    # the curve reaches the rear face without the superseded side relief that
    # cut a visible crescent bite into the lower ring.
    if STAND_FOOT:
        part -= lm_rear_exit_port_cutter()
        part = clean_stage(part, "LM gradual R14 rear cable exit")
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
    part = _apply_complete_lm_um_joint(part, "lm")
    part = clean_stage(part, "final standalone LM joint recut")

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
        part, _side_ring_fairing("um"),
        "continuous smooth UM side-ring fairing")
    part = _fuse_attached(
        part, _um_t_rear_recess_backfill(),
        "symmetric solid rear T-cover/UM-recess backfill")

    um_boss_floor = (
        UM_SEAT_Z - UM_PILOT_DEPTH_MM - UM_PAD_FLOOR_MM)
    for pilot_angle, (px, py) in zip(UM_PILOT_ANGLES_DEG, UM_PILOT_XY):
        boss = _cylinder_at(
            px, py, UM_INSERT_BOSS_D / 2.0,
            um_boss_floor, UM_SEAT_Z)
        support = boss
        for tangent_offset, support_z0, support_z1 in (
                um_pilot_spoke_z_segments(
                    pilot_angle, um_boss_floor, UM_SEAT_Z)):
            support = support.fuse(_radial_spoke(
                (cx, cy), (px, py), UM_RECESS_R + 0.25,
                UM_STRUCT_SPOKE_W, support_z0, support_z1,
                tangent_offset)).clean()
        if pilot_angle in UM_PILOT_SPOKE_TANGENTIAL_OFFSETS_DEG:
            # Fill only the upper clear-spoke crescent.  The land is fully
            # buried inside the UM driver recess and stops 0.80+ mm clear of
            # the standalone M3 ear, so it cannot recreate the unwanted
            # visible ear bridge.
            support = support.fuse(_plan_prism(
                _um_pilot_recess_closure_land((cx, cy), (px, py)),
                UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z,
                UM_SEAT_Z)).clean()
        part = _fuse_attached(
            part, support, "UM insert boss/spoke")

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

    part = _apply_complete_lm_um_joint(part, "um")
    part = _verify_side_magnet_lands(part, "um")

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
    part = _cut_side_magnet_pockets(part, "um")
    # The final ownership masks intentionally govern the full-depth closure
    # web, not the Z-owned functional boss. Restore the complete UM receiver
    # after those masks and after every positive cover union. Then reopen the
    # route lumen: the nearby T cutter is authoritative and must not be
    # refilled by the late ear fusion.
    part = _apply_complete_lm_um_joint(part, "um")
    for cutter in route_inner_cutters("um"):
        part -= cutter
    # Route cuts and receiver cuts are both negative, but repeat the receiver
    # cylinders explicitly so this final construction order remains obvious
    # and robust to future positive route-shell repairs.
    for x in JOINT_EAR_X:
        part -= _cylinder_at(
            x, JOINT_EAR_Y, JOINT_INSERT_BORE_D / 2.0,
            *JOINT_INSERT_BORE_Z)
    # The full-depth T--UM plan mask is a closure-web authority only.  Restore
    # the complete rear functional ears after every positive cover union and
    # every ownership mask, then reopen the authoritative route and screw
    # passage so neither can be refilled by this final fusion.
    part = _apply_complete_um_tweeter_joint(part, "um")
    for cutter in route_inner_cutters("um"):
        part -= cutter
    for x in TWEETER_JOINT_X:
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_BORE_TOP_Z)
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
    assembly.label = f"lx521_obiwan_r6f_extreme_barebone_core_{state}"
    return assembly


if __name__ == "__main__":
    for name, solid in core_parts().items():
        bb = solid.bounding_box().size
        print(name, f"{bb.X:.2f} x {bb.Y:.2f} x {bb.Z:.2f} mm",
              f"{solid.volume / 1000.0:.2f} cm3", "valid", solid.is_valid)
