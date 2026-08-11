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
  LM pair follows the shared curved bridge shoulder above the floor bend;
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
from pathlib import Path


# A direct module invocation used to enter both full carrier builds before a
# guard existed. Establish the authenticated outer guard before importing OCC.
if __name__ == "__main__":
    import run_memory_guarded as _memory_guard
    _memory_guard.reexec_under_guard(Path(__file__))

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

def _require_guarded_build():
    """Reject accidental in-process carrier construction outside the guard."""
    import run_memory_guarded as memory_guard
    memory_guard.require_guarded_build(
        "Obi-Wan carrier construction requires run_memory_guarded.py; "
        "use Make or the staged Obi-Wan exporter")


from ..base import (
    L22_CUTOUT,
    STAND_FOOT,
    THICKNESS_MM,
    UM_CUTOUT,
    UM_PILOT_ANGLES_DEG,
    UM_PILOT_DEPTH_MM,
    UM_PILOT_D_MM,
    UM_TERMINAL_CLOCK_DEG,
    m5_insert_bore_cutter,
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
    bridge_soft_blend_frame,
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

PRINT_ORIENTATION = "front-face-down"
CORE_REAR_Z = 6.8
LM_CORE_R = LM_RECESS_R + 2.4       # 113.0
UM_CORE_R = UM_RECESS_R + 2.4       # 51.7
# The D5 x 2 captive stack is 3.14 mm deep (dual 0.4/0.6-nozzle 0.52-mm
# skins), while the structural lip is only 2.40 mm.  A local flat pad was
# visibly proud of the circular ring.  Replace it with one continuous
# cylindrical fairing around each exposed carrier side.  Only the
# pre-existing LM--UM and T--UM cusp/service regions retain the structural
# radius;
# every magnet station shares one exact exterior radius, so its location is
# impossible to infer from the outer silhouette.  The cavity datum remains
# 0.15 mm below that surface; the extra skin is intentional and keeps the
# rectangular D5.20+walls land wholly inside the circular fairing even on the
# tighter UM radius.  The fairing depth is derived, not literal: a deeper
# captive land pushes the cavity datum outward, and the visible radius must
# follow it or the designed 0.15-mm inset silently collapses (the former
# 0.80 literal left 0.01 mm at the 3.14-mm land and the ring land check
# refused the chord-bulge corners).
SIDE_RING_CAVITY_RECESS_CLEAR_MM = 0.05
SIDE_RING_CAVITY_FACE_OFFSET_MM = (
    CAPTIVE_LAND_MM - (LM_CORE_R - LM_RECESS_R)
    + SIDE_RING_CAVITY_RECESS_CLEAR_MM
)
SIDE_RING_CAVITY_INTENDED_INSET_MM = 0.15
SIDE_RING_FLUSH_FAIRING_MM = (
    SIDE_RING_CAVITY_FACE_OFFSET_MM + SIDE_RING_CAVITY_INTENDED_INSET_MM)
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
# top centreline (world polar 116/64 degrees). The lower pair sits at the exact
# midpoint of the common soft shoulder above the Option-B tangent. Its local
# normal is almost 45 degrees, and the cavity datum is inset 0.15 mm from the
# uninterrupted shoulder surface. This removes every lower rail while keeping
# floor/no-floor carriers and flat/graded split prints on one exact interface. All
# sites use the same front-biased z=15.10 datum. D5x2 magnets are completely
# captive in D5.20 x 2.10 cavities behind 0.45-mm skins, with a printable
# circular cradle and self-supporting 45-degree roof. The four ring stations
# sit inside continuous
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
    # edge gap. Lower LM sites are explicit shoulder records below rather
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
LM_SHOULDER_MAGNET_PARAMETER = 0.50
(
    LM_SHOULDER_MAGNET_OUTER_FACE_RIGHT,
    LM_SHOULDER_MAGNET_NORMAL_RIGHT,
) = bridge_soft_blend_frame(LM_SHOULDER_MAGNET_PARAMETER, "right")
LM_SHOULDER_MAGNET_CAVITY_FACE_INSET_MM = (
    SIDE_RING_CAVITY_FACE_INSET_MM)
LM_SHOULDER_MAGNET_FACE_RIGHT = tuple(
    point - LM_SHOULDER_MAGNET_CAVITY_FACE_INSET_MM * normal
    for point, normal in zip(
        LM_SHOULDER_MAGNET_OUTER_FACE_RIGHT,
        LM_SHOULDER_MAGNET_NORMAL_RIGHT,
        strict=True,
    )
)
LM_SHOULDER_MAGNET_ANGLE_RIGHT_DEG = (
    math.degrees(math.atan2(
        LM_SHOULDER_MAGNET_NORMAL_RIGHT[1],
        LM_SHOULDER_MAGNET_NORMAL_RIGHT[0],
    )) % 360.0
)
LM_SHOULDER_MAGNET_Z = OBIWAN_MAGNET_Z_MM

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
















































































def _cut_lm_mount_holes(part):
    """All six W22 sites remain ordinary blind driver inserts."""
    for x, y in LM_PILOT_XY:
        part -= m5_insert_bore_cutter(
            (x, y),
            opening_z=LM_SEAT_Z,
            total_depth=LM_BORE_DEPTH_MM,
            opening_side="+z",
            overshoot=0.15,
        )
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
    # short of its feed during this pre-fusion cut; the 2.0-mm owner
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
        bounds = [(
            solid.bounding_box().min.X,
            solid.bounding_box().min.Y,
            solid.bounding_box().min.Z,
            solid.bounding_box().max.X,
            solid.bounding_box().max.Y,
            solid.bounding_box().max.Z,
        ) for solid in solids]
        raise RuntimeError(
            f"LM cutter group {index} detached geometry: "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in solids]} bounds={bounds}")
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
    # No-floor UM/T feeds use the route owner's shared, pre-fused transition
    # tools. Dimensions and the resulting void set are unchanged; only the
    # invalid equal-radius intermediate topology is removed.
    if not STAND_FOOT:
        for entry_name, transition in no_floor_rear_entry_transition_cutters():
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
    state = "floor" if STAND_FOOT else "no_floor_fused_solid_web"
    return ordered_labeled_compound(
        core_parts(),
        label=f"lx521_obiwan_r6f_extreme_barebone_core_{state}",
    )


if __name__ == "__main__":
    for name, solid in core_parts().items():
        bb = solid.bounding_box().size
        print(name, f"{bb.X:.2f} x {bb.Y:.2f} x {bb.Z:.2f} mm",
              f"{solid.volume / 1000.0:.2f} cm3", "valid", solid.is_valid)


# Compatibility facade: implementations live in cohesive Stage 4 owners.
from .closure_webs import (
    _bounded_plan_lenses,
    _circle_branch_slope,
    _circle_branch_y,
    _closure_owner_bands,
    _cubic_point,
    _curve_band,
    _enforce_junction_plan_ownership,
    _junction_closure_web,
    _lm_um_rear_recess_backfill,
    _lm_um_rear_recess_backfill_plan,
    _mirrored_reversed,
    _owned_lens_addition,
    _partition_lens_components,
    _path_band,
    _path_owner_bands,
    _printable_lens_components,
    _t_crescent_boundary_y,
    _tangent_blend_to_boss,
    _terminal_fit_drains,
    _um_t_rear_recess_backfill,
    _um_t_rear_recess_backfill_plan,
    junction_closure_polygons,
    lm_um_closure_polygons,
    t_um_closure_polygons,
)

from .joints import (
    _apply_complete_lm_um_joint,
    _apply_complete_um_tweeter_joint,
    _complete_joint_ear,
    _complete_joint_ear_plan,
    _complete_joint_receiver_notch,
    _complete_tweeter_joint_ear,
    _complete_tweeter_joint_ear_plan,
    _complete_tweeter_joint_receiver_notch,
    _fuse_attached,
    _joint_ear,
    _owned_joint_ear_plan,
    _owned_tweeter_joint_plan,
    _supported_plan_components,
    joint_ear_polygon,
    joint_load_facts,
    tweeter_joint_polygon,
)

from .magnets import (
    _axis_cylinder,
    _cut_side_magnet_pockets,
    _verify_side_magnet_lands,
    side_magnet_sites,
)
