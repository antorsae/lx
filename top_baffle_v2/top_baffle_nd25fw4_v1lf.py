"""Extreme V1LF R6F barebone: two flush driver carriers and nothing else.

The previous V1LF was a rear-thinned copy of the complete B2 outline.
This module starts from the irreducible interfaces instead:

* LM carrier: D190 opening, D221.2 flush seat, R113.0 outside.
* UM carrier: D82 opening, D98.6 flush seat, R51.7 outside.
* two rounded M3 through-bolted half-lap ears establish 165.100 mm;
* six flush magnet interfaces provide four LM and two UM alignment sites;
  the original upper LM pair stays at +/-26 degrees from top while the lower
  LM pair mates horizontally through the shared W64 base sides;
* floor mode owns a full-depth integral W64 stem and rectangular floor foot;
  no-floor mode owns a shallow front-flush four-hole solid web;
* the D8.2 UM path is buried only in LM and exits flush at R113 before its
  free span behind UM; D6 is buried only in LM/UM before floating behind the
  crescent; the short LM lead also floats without a printed micro-duct;
* two compact direct half-lap ears attach the optional tweeter crescent.

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
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import orient


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
from top_baffle_nd25fw4_v1lf_route import (
    CROSSOVER_T_Z,
    TS_CUTTER_R,
    TUNNEL_ROOF_SKIN,
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
# lm_lower prints share one exact receiver axis. All four LM sites are flush
# with their owning surfaces, with no ear or proud material. The UM pair uses
# flush 129.5/50.5-degree sites at z=15.1 in its R51.7 lip to clear the upper
# T cover, so neither carrier has a proud magnet ear. D5x2 magnets use
# D5.2 x 2.2 axis-normal pockets; the extra 0.2-mm depth is adhesive allowance,
# so each magnet must be
# held flush while bonding rather than bottomed. They carry alignment/anti-
# rattle load only. The monolithic W64 stem/root carries floor load; the four
# stock bridge holes carry no-floor load.
SIDE_EAR_D = 7.8
SIDE_EAR_IN = 0.7
SIDE_EAR_OUT = 3.3
SIDE_MAGNET_D = 5.0
SIDE_MAGNET_POCKET_D = 5.2
SIDE_MAGNET_DEPTH = 2.2
SIDE_MAGNET_Z = {
    "lm": 12.55,
    # Raised within the R51.7 lip so the flush pocket keeps at least 1.1 mm
    # from the conservative upper-T cover envelope while retaining 0.6 mm
    # front skin. The inward radial pocket floor is 0.2 mm at R49.3.
    "um": 15.10,
}
SIDE_INTERFACE_GAP = 0.20
SIDE_MAGNET_ANGLES = {
    # The upper pair is one degree crownward of the adjacent 120/60-degree W22
    # inserts, retaining its exact 2.251-mm D5.2-pocket-capsule to D9.6-boss
    # edge gap. Lower LM sites are explicit base-side records below rather
    # than fake polar positions on the R113 ring.
    "lm": (116.0, 64.0),
    # Symmetric top-side sites clear the T arc, D8 insert pads and direct
    # tweeter ears; the former 45-degree site cut through the T cover.
    "um": (129.5, 50.5),
}
SIDE_MAGNET_FACE_OFFSET = {
    "lm": 0.0,
    "um": 0.0,
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
    ]
    face = Face(outer, holes)
    return Pos(0.0, 0.0, z0) * extrude(face, amount=z1 - z0)


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


def _joint_ear(owner: str, x: float, z_span, clearance: float = 0.0):
    return _plan_prism(joint_ear_polygon(owner, x, clearance), *z_span)


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
                "flush_buried": face_offset == 0.0,
                "proud_ear_added": face_offset > 0.0,
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
                    "flush_buried": True,
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
    """Reassert final surface-normal pockets after every possible union."""
    for site in side_magnet_sites(driver):
        part -= _axis_cylinder(
            site["face"], site["normal"], site["z_mm"],
            SIDE_MAGNET_POCKET_D, SIDE_MAGNET_DEPTH, 1.0)
    return part


def _add_side_magnet_ears(part, driver: str):
    for site in side_magnet_sites(driver):
        normal = site["normal"]
        face_offset = site["face_offset_mm"]
        if face_offset > 0.0:
            # Retained as a generic fallback for any future proud interface;
            # both present V1LF carriers deliberately use zero offset.
            wall = (site["face"][0] - face_offset * normal[0],
                    site["face"][1] - face_offset * normal[1])
            ear = _axis_cylinder(
                wall, normal, site["z_mm"], SIDE_EAR_D,
                SIDE_EAR_IN, face_offset)
            part += ear
    return _cut_side_magnet_pockets(part, driver)


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
    """Cut the buffered complementary rounded ear from the other ring."""
    z0, z1 = z_span
    notch = _joint_ear(owner, x,
                       (z0 - JOINT_RECEIVER_RADIAL_CLEAR,
                        z1 + JOINT_RECEIVER_RADIAL_CLEAR),
                       JOINT_RECEIVER_RADIAL_CLEAR)
    return part - notch


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
        f"{[solid.volume for solid in combined.solids()]}")


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

    # Establish every small attachment union before carving the long
    # swept tunnel cutters.
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "um", x, UM_JOINT_Z)
    for x in JOINT_EAR_X:
        part += _joint_ear("lm", x, LM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    part = _add_side_magnet_ears(part, "lm")

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


def finalize_lm_carrier(part, *, routes_already_cut=False):
    """Hollow and functionally recut one native-BREP LM outer blank."""
    _require_guarded_build()
    cx, cy, cut_d = L22_CUTOUT
    if not routes_already_cut:
        for index in range(route_inner_cutter_group_count("lm")):
            part = apply_lm_route_cutter(part, index)
    # Reassert the functional driver interfaces after every cover union;
    # no crossover/anchor material may re-enter the flange seat.
    part -= _cylinder_at(cx, cy, LM_RECESS_R,
                         LM_SEAT_Z, THICKNESS_MM + 0.5)
    part -= _cylinder_at(cx, cy, cut_d / 2.0,
                         CORE_REAR_Z - 12.0, THICKNESS_MM + 1.0)
    part = _cut_lm_mount_holes(part)
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "um", x, UM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    # Recut the complementary UM half-laps after every route Boolean.
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "um", x, UM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    part = _cut_side_magnet_pockets(part, "lm")

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

    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "lm", x, LM_JOINT_Z)
    for x in JOINT_EAR_X:
        part += _joint_ear("um", x, UM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    part = _add_side_magnet_ears(part, "um")

    # Rear half of the direct crescent joints.  The complementary upper
    # half is an add-on and is removed from the core with a 0.1-mm plan
    # receiver clearance plus the established 0.2-mm axial split.
    for x in TWEETER_JOINT_X:
        part -= _plan_prism(
            tweeter_joint_polygon(x, TWEETER_JOINT_CLEAR),
            TWEETER_ADDON_JOINT_Z[0] - TWEETER_JOINT_CLEAR,
            TWEETER_ADDON_JOINT_Z[1] + 0.2)
        part += _plan_prism(
            tweeter_joint_polygon(x), *TWEETER_CORE_JOINT_Z)
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_JOINT_Z[1] + 0.3)
    for px, py in UM_PILOT_XY:
        part -= _cylinder_at(px, py, UM_PILOT_D_MM / 2.0,
                             UM_SEAT_Z - UM_PILOT_DEPTH_MM,
                             UM_SEAT_Z + 0.15)
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "lm", x, LM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    for x in TWEETER_JOINT_X:
        part -= _plan_prism(
            tweeter_joint_polygon(x, TWEETER_JOINT_CLEAR),
            TWEETER_ADDON_JOINT_Z[0] - TWEETER_JOINT_CLEAR,
            TWEETER_ADDON_JOINT_Z[1] + 0.2)
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_JOINT_Z[1] + 0.3)

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
    # Tunnel covers cross both compact joint neighborhoods. Re-cut every
    # complementary half-lap and bolt/insert approach after those unions so
    # the cable skin cannot silently refill an assembly interface.
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "lm", x, LM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    for x in TWEETER_JOINT_X:
        part -= _plan_prism(
            tweeter_joint_polygon(x, TWEETER_JOINT_CLEAR),
            TWEETER_ADDON_JOINT_Z[0] - TWEETER_JOINT_CLEAR,
            TWEETER_ADDON_JOINT_Z[1] + 0.2)
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_JOINT_Z[1] + 0.3)

    # Recut both families of assembly interfaces after the final route
    # Booleans; all cable exits are native flush mouths with no telescoping
    # collars to reassert.
    for x in JOINT_EAR_X:
        part = _receiver_notch(part, "lm", x, LM_JOINT_Z)
        part -= _cylinder_at(x, JOINT_EAR_Y, JOINT_HOLE_D / 2.0,
                             CORE_REAR_Z - JOINT_BORE_REAR_OVERSHOOT,
                             THICKNESS_MM + 0.2)
    for x in TWEETER_JOINT_X:
        part -= _plan_prism(
            tweeter_joint_polygon(x, TWEETER_JOINT_CLEAR),
            TWEETER_ADDON_JOINT_Z[0] - TWEETER_JOINT_CLEAR,
            TWEETER_ADDON_JOINT_Z[1] + 0.2)
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_JOINT_Z[1] + 0.3)
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
