"""Separate R6F add-ons for the extreme two-collar V1LF core.

Nothing in this module is required to manufacture the LM or UM driver
carrier. The floor support remains a separate print but is mandatory in
the loaded floor-state assembly; pick the other functions as needed:

* the required open-rail floor/NL8 support in floor mode,
* the original tapered tweeter crescent on two compact direct ears.

Six Ø5×2 side magnets sit face-flush in Ø5.2×2.2 pockets--four on LM and
two on UM--and remain alignment/anti-rattle interfaces with no structural
load credit. The no-floor front-flush solid bridge web is monolithic LM-core
geometry; the floor support owns three heat-sets on rotated lower W22 axes.
"""

from __future__ import annotations

import math

from build123d import (
    Box,
    Compound,
    Cylinder,
    Part,
    Plane,
    Polyline,
    Pos,
    Rot,
    Wire,
    extrude,
    make_face,
)
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from top_baffle_nd25fw4 import (
    BRIDGE_HOLE_XY,
    L22_CUTOUT,
    STAND_FOOT,
    THICKNESS_MM,
    UM_CUTOUT,
)
from top_baffle_nd25fw4_cables import ROUTING_PROFILE
from top_baffle_nd25fw4_v1lf import (
    SIDE_INTERFACE_GAP,
    LM_RECESS_R,
    LM_STRUCT_SPOKE_W,
    STRUCT_MOUNT_CLEAR_D,
    STRUCT_MOUNT_KEY_D,
    STRUCT_MOUNT_KEY_Z,
    TWEETER_ADDON_JOINT_Z,
    TWEETER_CORE_JOINT_Z,
    TWEETER_JOINT_CLEAR,
    TWEETER_JOINT_INSERT_BORE_D,
    TWEETER_JOINT_X,
    TWEETER_JOINT_Y,
    _require_guarded_build,
    _plan_prism,
    structural_mount_sites,
    tweeter_joint_polygon,
)
from top_baffle_nd25fw4_v1lf_route import (
    FLOOR_SUPPORT_INSERT_BOSS_D,
    FLOOR_SUPPORT_INSERT_BOSS_FRONT_Z,
    FLOOR_SUPPORT_INSERT_REAR_Z,
    route_outer_covers,
    support_floor_clearance_cutters,
)
from top_baffle_nd25fw4_v1lf_bridge import bridge_plan_facts

if ROUTING_PROFILE != "v1lf":
    raise RuntimeError(
        "V1LF add-ons require LX_ROUTING_PROFILE=v1lf (R6F)"
    )

NL8_PANEL_W = 38.0
NL8_PANEL_H = 44.0
NL8_PANEL_T = 4.0
NL8_CUTOUT_D = 31.0
NL8_SCREW_D = 3.2
NL8_SCREW_PITCH = 29.2
NL8_CENTER_Y = 22.0
FOOT_RAIL_PANEL_TOP_Z = -146.0
# Do not rely on a coplanar rail end/panel face for the structural
# union.  Three 0.20-mm layers of positive overlap survive meshing and
# slicing while remaining inside the panel's existing Z envelope.
FOOT_RAIL_PANEL_OVERLAP = 0.60

# The rear support yoke stays wholly outside all three cable paths.
SUPPORT_YOKE_R = 126.0
# The 10-mm ribbed yoke is the minimum section already credited by the
# structural screen.  The former 14-mm sweep made the floor-support print
# exceed a 256-mm bed even at its optimum rotation, without increasing
# the governing 10-mm transfer-member capacity.
SUPPORT_MEMBER_W = 10.0
SUPPORT_FLANGE_Z = (0.60, FLOOR_SUPPORT_INSERT_BOSS_FRONT_Z)
SUPPORT_RIB_W = 4.0
SUPPORT_RIB_REAR_Z = -11.40
# Two full 0.20-mm layers of overlap keep the ribs joined to the face
# flange after slicing.  A token CAD-only overlap can disappear in the
# generated toolpath even when OCC reports one valid solid.
SUPPORT_RIB_FLANGE_OVERLAP = 0.40
# Three positive keyed/bolted carrier interfaces reuse the D9.6 rear pads
# at the lower W22 axes. The carrier has D5.5 clearances at those sites;
# a long driver-side M5 screw enters a rear-installed support heat-set.
# No rear screw head, front nut or double-used carrier insert is assumed.
STRUCT_LUG_D = 20.0
# The 300-degree UM route and 240/180-degree tweeter route cross their
# reused floor-support axes below the flange. A 14-mm channel puts the inner
# edges of its 4-mm ribs only 3 mm off-axis, directly inside both the
# insert boss and the floor keepout. At 20 mm the ribs start 6 mm
# off-axis, outside the local insert boss while their outer edges land on the
# existing D20 lug.  The tangential route can still notch the rib end
# caps locally; the uninterrupted 20 x 6 flange carries that short
# near-support transition before the full U-section resumes outboard.
STRUCT_ARM_W = 20.0
STRUCT_CLEAR_D = STRUCT_MOUNT_CLEAR_D
STRUCT_RECEIVER_D = STRUCT_MOUNT_KEY_D + 0.4
STRUCT_RECEIVER_FLOOR_Z = STRUCT_MOUNT_KEY_Z[0]
STRUCT_INSERT_D = 6.4
STRUCT_INSERT_BOSS_D = FLOOR_SUPPORT_INSERT_BOSS_D
STRUCT_INSERT_REAR_Z = FLOOR_SUPPORT_INSERT_REAR_Z
STRUCT_INSERT_DEPTH = 6.8
STRUCT_INSERT_FRONT_Z = STRUCT_INSERT_REAR_Z + STRUCT_INSERT_DEPTH
STRUCT_INSERT_HARDWARE_LENGTH = 5.8
STRUCT_INSERT_HARDWARE_FRONT_Z = (
    STRUCT_INSERT_REAR_Z + STRUCT_INSERT_HARDWARE_LENGTH)
STRUCT_DRIVER_HEAD_D = 10.0
STRUCT_DRIVER_HEAD_H = 3.0
STRUCT_DRIVER_HEAD_Z = (THICKNESS_MM, THICKNESS_MM + STRUCT_DRIVER_HEAD_H)
STRUCT_SCREW_SHANK_D = 5.0

# Conservative structural model.  The repository's installed-driver
# budget is 3.2 kg; 4.0 kg includes printed parts, wiring and hardware.
# Bambu publishes 65 MPa XY flexural strength for PLA Tough+.  The
# 18/8 MPa allowables below deliberately derate for FDM anisotropy,
# stress concentrations and sustained-load creep at room temperature.
STRUCT_DESIGN_MASS_KG = 4.0
STRUCT_DESIGN_Y_CG = 230.0
STRUCT_REAR_CG_MM = 70.0
STRUCT_INSERT_PULLOUT_N = 600.0
def _u_section_modulus_mm3():
    """Actual narrowest 10x6 flange plus two 4x12 rear webs."""
    flange_width = 10.0
    flange_t = SUPPORT_FLANGE_Z[1] - SUPPORT_FLANGE_Z[0]
    web_width = 2.0 * SUPPORT_RIB_W
    web_height = SUPPORT_FLANGE_Z[0] - SUPPORT_RIB_REAR_Z
    web_area = web_width * web_height
    flange_area = flange_width * flange_t
    web_z = web_height / 2.0
    flange_z = web_height + flange_t / 2.0
    centroid = ((web_area * web_z + flange_area * flange_z)
                / (web_area + flange_area))
    inertia = (
        web_width * web_height ** 3 / 12.0
        + web_area * (web_z - centroid) ** 2
        + flange_width * flange_t ** 3 / 12.0
        + flange_area * (flange_z - centroid) ** 2
    )
    return inertia / max(centroid, web_height + flange_t - centroid)


STRUCT_SECTION_MODULUS_MM3 = _u_section_modulus_mm3()
STRUCT_MEMBER_SPAN_MM = 110.0
STRUCT_SHORT_ALLOW_MPA = 18.0
STRUCT_CREEP_ALLOW_MPA = 8.0


def _cylinder_at(x, y, radius, z0, z1):
    return Pos(x, y, (z0 + z1) / 2.0) * Cylinder(radius, z1 - z0)


def _channel_plan(points, width):
    return LineString(points).buffer(
        width / 2.0, resolution=24, cap_style=1, join_style=1)


def _channel_rib_plan(points, width):
    """Twin/perimeter ribs as one exact planar channel outline."""
    outer = _channel_plan(points, width)
    inner_radius = width / 2.0 - SUPPORT_RIB_W
    if inner_radius <= 0.0:
        return outer
    inner = LineString(points).buffer(
        inner_radius, resolution=24, cap_style=1, join_style=1)
    return outer.difference(inner)


def _polar_lm(radius: float, angle_deg: float):
    angle = math.radians(angle_deg)
    return (L22_CUTOUT[0] + radius * math.cos(angle),
            L22_CUTOUT[1] + radius * math.sin(angle))


def _support_backbone(z0: float, z1: float):
    """Single-extrusion flange and rib networks for the floor support."""
    p = lambda angle: _polar_lm(SUPPORT_YOKE_R, angle)

    angles = list(range(180, 361, 10))
    yoke = [p(angle) for angle in angles]
    flange_members = [_channel_plan(yoke, SUPPORT_MEMBER_W)]
    rib_members = [_channel_rib_plan(yoke, SUPPORT_MEMBER_W)]

    # Three threaded load paths from the LM carrier into the yoke. Their
    # D20 lugs are part of the same flange plan, not later coplanar fuses.
    for site in structural_mount_sites():
        xy = site["xy"]
        target = p(site["angle_deg"])
        flange_members.extend((
            _channel_plan((xy, target), STRUCT_ARM_W),
            Point(*xy).buffer(STRUCT_LUG_D / 2.0, resolution=32),
        ))
        rib_members.append(_channel_rib_plan((xy, target), STRUCT_ARM_W))

    # Both floor-transfer arms and their crossbar are also incorporated
    # before extrusion, eliminating the old z=0.6 shared-face network.
    transfers = (
        (p(260.0), (-30.0, 14.0), 10.0),
        (p(280.0), (30.0, 14.0), 10.0),
        ((-30.0, 14.0), (30.0, 14.0), 12.0),
    )
    for start, stop, width in transfers:
        flange_members.append(_channel_plan((start, stop), width))
        rib_members.append(_channel_rib_plan((start, stop), width))

    flange_plan = unary_union(flange_members)
    rib_plan = unary_union(rib_members)
    part = _plan_prism(flange_plan, z0, z1)
    ribs = _plan_prism(
        rib_plan, SUPPORT_RIB_REAR_Z,
        z0 + SUPPORT_RIB_FLANGE_OVERLAP)
    part = part.fuse(ribs).clean()
    insert_boss_plan = unary_union([
        Point(*site["xy"]).buffer(STRUCT_INSERT_BOSS_D / 2.0,
                                   resolution=32)
        for site in structural_mount_sites()
    ])
    part = part.fuse(_plan_prism(
        insert_boss_plan, STRUCT_INSERT_REAR_Z, z1)).clean()
    return part


def _finish_structural_mounts(part):
    for site in structural_mount_sites():
        x, y = site["xy"]
        # Rear-installed D6.4 heat-set cavity with a D5.5 front screw
        # throat. The 0.20-mm axial overlap avoids a Boolean membrane.
        part -= _cylinder_at(
            x, y, STRUCT_INSERT_D / 2.0,
            STRUCT_INSERT_REAR_Z - 0.2, STRUCT_INSERT_FRONT_Z)
        part -= _cylinder_at(
            x, y, STRUCT_CLEAR_D / 2.0,
            STRUCT_INSERT_FRONT_Z - 0.2, SUPPORT_FLANGE_Z[1] + 0.4)
        part -= _cylinder_at(
            x, y, STRUCT_RECEIVER_D / 2.0,
            STRUCT_RECEIVER_FLOOR_Z, SUPPORT_FLANGE_Z[1] + 0.4)
    return part


def _finalize_support(part, label: str):
    """Require source booleans to yield one valid support without repair."""
    finished = part.clean()
    solids = list(finished.solids())
    if (finished.is_valid and len(solids) == 1
            and solids[0].volume > 0.01):
        return Part([solids[0]])
    diagnostics = []
    for solid in solids:
        bounds = solid.bounding_box()
        diagnostics.append({
            "volume_mm3": solid.volume,
            "min": (bounds.min.X, bounds.min.Y, bounds.min.Z),
            "max": (bounds.max.X, bounds.max.Y, bounds.max.Z),
        })
    raise RuntimeError(
        f"{label}: support finalization failed without mesh/volume repair; "
        f"valid={finished.is_valid} solids={diagnostics}")


def _fuse_required(part, addition, label):
    """Positive-growth one-solid fusion; never discard detached pieces."""
    before = part.volume
    added = addition.volume
    volume_tol = max(0.05, (before + added) * 1e-6)
    combined = part.fuse(addition).clean()
    solids = list(combined.solids())
    if (combined.is_valid and len(solids) == 1
            and solids[0].volume > 0.01
            and combined.volume > before + min(0.05, added * 1e-4)
            and combined.volume <= before + added + volume_tol):
        return Part([solids[0]])
    raise RuntimeError(
        f"{label}: required fusion failed; valid={combined.is_valid} "
        f"volumes={[solid.volume for solid in combined.solids()]}")


def structural_spoke_clearance_cutters():
    """Exact support reliefs for the three carrier-owned radial spokes.

    The D10 round receiver clears each D9.6 locating key, but the key's real
    outward load spoke also occupies the support-flange layer.  Keep these
    source-owned cutters public so the final-BREP acceptance test can compare
    the support against the *intentionally relieved* socket envelope rather
    than a fictitious unrelieved D20 annulus.
    """
    _require_guarded_build()
    cx, cy = L22_CUTOUT[:2]
    cutters = []
    for site in structural_mount_sites():
        x, y = site["xy"]
        dx, dy = x - cx, y - cy
        length = math.hypot(dx, dy)
        contact = (
            cx + (LM_RECESS_R + 0.25) * dx / length,
            cy + (LM_RECESS_R + 0.25) * dy / length,
        )
        plan = LineString(((x, y), contact)).buffer(
            LM_STRUCT_SPOKE_W / 2.0 + SIDE_INTERFACE_GAP,
            resolution=24, cap_style=1, join_style=1)
        cutters.append(_plan_prism(
            plan, STRUCT_MOUNT_KEY_Z[0] - SIDE_INTERFACE_GAP,
            SUPPORT_FLANGE_Z[1] + SIDE_INTERFACE_GAP))
    return tuple(cutters)


def _clear_buried_route_floors(part):
    # The resampled/clearance sweeps below provide the specified 0.4-mm air
    # gap, but their independently phased octagons can miss a tiny corner of
    # the exact manufactured cover. Subtract the exact final cover first so
    # clearance construction can never leave a print/print collision.
    for cover in route_outer_covers("lm"):
        part -= cover
    for cutter in support_floor_clearance_cutters():
        part -= cutter
    # Each lower LM locating boss has a real radial load spoke. The D10
    # circular receiver already clears the D9.6 boss; extend that receiver by
    # the spoke's exact plan plus 0.2 mm so the support yoke cannot overlap the
    # three floor-state spokes while retaining the rest of the key socket.
    for cutter in structural_spoke_clearance_cutters():
        part -= cutter
    return part


def support_plan_geometry(stand_foot: bool):
    """Shared plan facts for CAD and the routing-sheet overlay."""
    if not stand_foot:
        return {
            "yoke": [],
            "structural_mounts": [],
            "bridge_holes": list(BRIDGE_HOLE_XY),
            "floor_rails": [],
            "fused_bridge": bridge_plan_facts(),
        }
    angles = list(range(180, 361, 10))
    return {
        "yoke": [_polar_lm(SUPPORT_YOKE_R, angle) for angle in angles],
        "structural_mounts": [site["xy"] for site in structural_mount_sites()],
        "bridge_holes": list(BRIDGE_HOLE_XY),
        "floor_rails": [(-30.0, 14.0), (30.0, 14.0)],
        "fused_bridge": None,
    }


def structural_load_facts():
    """Deterministic bolt-group and U-channel screening calculation."""
    gravity = 9.80665
    ys = [site["xy"][1] for site in structural_mount_sites()]
    ybar = sum(ys) / len(ys)
    sum_sq = sum((y - ybar) ** 2 for y in ys)

    def worst_normal_insert(g_load: float):
        force = STRUCT_DESIGN_MASS_KG * gravity * g_load
        moment = force * (STRUCT_DESIGN_Y_CG - ybar)
        return max(force / len(ys) + abs(moment * (y - ybar) / sum_sq)
                   for y in ys)

    def rear_moment_insert(g_load: float):
        moment = (STRUCT_DESIGN_MASS_KG * gravity * g_load
                  * STRUCT_REAR_CG_MM)
        return max(abs(moment * (y - ybar) / sum_sq) for y in ys)

    def combined_insert(g_load: float):
        force = STRUCT_DESIGN_MASS_KG * gravity * g_load
        normal_moment = force * (STRUCT_DESIGN_Y_CG - ybar)
        rear_moment = force * STRUCT_REAR_CG_MM
        return max(math.hypot(
            force / len(ys)
            + abs(normal_moment * (y - ybar) / sum_sq),
            abs(rear_moment * (y - ybar) / sum_sq))
            for y in ys)

    # One outer path conservatively receives 95% of the resultant; a
    # simply-supported 110-mm U-channel gives Mmax = F*L/4.
    def member_moment(g_load):
        return (STRUCT_DESIGN_MASS_KG * gravity * g_load * 0.95
                * STRUCT_MEMBER_SPAN_MM / 4.0)

    stress_1g = member_moment(1.0) / STRUCT_SECTION_MODULUS_MM3
    stress_3g = member_moment(3.0) / STRUCT_SECTION_MODULUS_MM3
    stress_5g = member_moment(5.0) / STRUCT_SECTION_MODULUS_MM3
    return {
        "design_mass_kg": STRUCT_DESIGN_MASS_KG,
        "design_y_cg_mm": STRUCT_DESIGN_Y_CG,
        "rear_cg_mm": STRUCT_REAR_CG_MM,
        "creep_allow_mpa": STRUCT_CREEP_ALLOW_MPA,
        "short_allow_mpa": STRUCT_SHORT_ALLOW_MPA,
        "insert_pullout_n": STRUCT_INSERT_PULLOUT_N,
        "member_span_mm": STRUCT_MEMBER_SPAN_MM,
        "group_ybar_mm": ybar,
        "group_sum_sq_mm2": sum_sq,
        "normal_insert_1g_n": worst_normal_insert(1.0),
        "normal_insert_3g_n": worst_normal_insert(3.0),
        "normal_insert_5g_n": worst_normal_insert(5.0),
        "rear_moment_insert_1g_n": rear_moment_insert(1.0),
        "rear_moment_insert_3g_n": rear_moment_insert(3.0),
        "rear_moment_insert_5g_n": rear_moment_insert(5.0),
        "combined_insert_1g_n": combined_insert(1.0),
        "combined_insert_3g_n": combined_insert(3.0),
        "combined_insert_5g_n": combined_insert(5.0),
        "insert_sf_3g": (STRUCT_INSERT_PULLOUT_N
                          / combined_insert(3.0)),
        "insert_sf_5g": (STRUCT_INSERT_PULLOUT_N
                          / combined_insert(5.0)),
        "actual_section_mm3": STRUCT_SECTION_MODULUS_MM3,
        "required_section_1g_mm3": (
            member_moment(1.0) / STRUCT_CREEP_ALLOW_MPA),
        "required_section_3g_mm3": (
            member_moment(3.0) / STRUCT_SHORT_ALLOW_MPA),
        "required_section_5g_mm3": (
            member_moment(5.0) / STRUCT_SHORT_ALLOW_MPA),
        "member_stress_1g_mpa": stress_1g,
        "member_stress_3g_mpa": stress_3g,
        "member_stress_5g_mpa": stress_5g,
        "member_sf_static_creep": STRUCT_CREEP_ALLOW_MPA / stress_1g,
        "member_sf_3g": STRUCT_SHORT_ALLOW_MPA / stress_3g,
        "member_sf_5g": STRUCT_SHORT_ALLOW_MPA / stress_5g,
        "magnet_load_credit_n": 0.0,
    }


def structural_hardware_proxies():
    """Support insert, long M5 shank and driver-side head envelopes."""
    _require_guarded_build()
    children = []
    for site in structural_mount_sites():
        x, y = site["xy"]
        children.extend((
            _cylinder_at(x, y, STRUCT_INSERT_D / 2.0,
                         STRUCT_INSERT_REAR_Z,
                         STRUCT_INSERT_HARDWARE_FRONT_Z),
            _cylinder_at(x, y, STRUCT_SCREW_SHANK_D / 2.0,
                         STRUCT_INSERT_HARDWARE_FRONT_Z,
                         STRUCT_DRIVER_HEAD_Z[0]),
            _cylinder_at(x, y, STRUCT_DRIVER_HEAD_D / 2.0,
                         *STRUCT_DRIVER_HEAD_Z),
        ))
    return Compound(children=children)


def _foot_rail(sign: float):
    """One open floor rail, tapered from x=+/-30 at the baffle to the
    38 mm connector panel.  Built in XZ, then extruded along Y."""
    panel_joint_z = FOOT_RAIL_PANEL_TOP_Z - FOOT_RAIL_PANEL_OVERLAP
    pts = [(sign * 36.0, 4.0), (sign * 24.0, 4.0),
           (sign * 13.0, panel_joint_z),
           (sign * 19.0, panel_joint_z),
           (sign * 36.0, 4.0)]
    # Reverse the left polygon so both wires have a consistent face.
    if sign < 0:
        pts = list(reversed(pts))
    face = make_face(Wire(Polyline(*pts).edges()))
    return extrude(Plane.XZ * face, amount=-18.3)


def floor_support():
    """Open twin-rail floor add-on with a minimal NL8 panel and two
    rear truss arms to the LM carrier.  It replaces the old solid foot."""
    _require_guarded_build()
    z0, z1 = SUPPORT_FLANGE_Z
    part = _support_backbone(z0, z1)
    for member in (_foot_rail(-1.0), _foot_rail(1.0)):
        part = part.fuse(member)

    panel_center_z = FOOT_RAIL_PANEL_TOP_Z - NL8_PANEL_T / 2.0
    panel = Pos(0, NL8_PANEL_H / 2.0, panel_center_z) * Box(
        NL8_PANEL_W, NL8_PANEL_H, NL8_PANEL_T)
    panel -= _cylinder_at(0.0, NL8_CENTER_Y, NL8_CUTOUT_D / 2.0,
                          -151.0, -145.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            panel -= _cylinder_at(
                sx * NL8_SCREW_PITCH / 2.0,
                NL8_CENTER_Y + sy * NL8_SCREW_PITCH / 2.0,
                NL8_SCREW_D / 2.0, -151.0, -145.0)
    part = part.fuse(panel)
    part = _clear_buried_route_floors(part)
    part = _finish_structural_mounts(part)
    return _finalize_support(part, "floor support")


def tweeter_crescent():
    """The original V1 rear-tapered tweeter crescent, cropped out of
    the mandatory baffle and returned with two compact direct half-laps."""
    _require_guarded_build()
    from top_baffle_nd25fw4_v1 import v1_solid
    from top_baffle_nd25fw4_v1lf import UM_CORE_R

    raw = v1_solid()
    # Crop above the last flare fragments so the acoustic crescent is
    # one connected body rather than a multi-island remnant.
    crop = Pos(0.0, 434.75, 10.0) * Box(150.0, 37.5, 25.0)
    cropped = (raw & crop).clean()
    cropped_solids = list(cropped.solids())
    if (not cropped.is_valid or len(cropped_solids) != 1
            or cropped_solids[0].volume <= 0.01):
        raise RuntimeError(
            "tweeter crop must produce exactly one crescent; "
            f"valid={cropped.is_valid} volumes="
            f"{[solid.volume for solid in cropped.solids()]}")
    part = Part([cropped_solids[0]])

    # Remove the core-owned rear half of each direct ear from the raw
    # crescent, then add its complementary front half and blind M3
    # heat-set receiver.  No bolt or head pierces the acoustic front.
    for x in TWEETER_JOINT_X:
        part -= _plan_prism(
            tweeter_joint_polygon(x, TWEETER_JOINT_CLEAR),
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_JOINT_Z[1] + TWEETER_JOINT_CLEAR)
        part = _fuse_required(
            part,
            _plan_prism(tweeter_joint_polygon(x), *TWEETER_ADDON_JOINT_Z),
            f"tweeter rounded ear {x:+.1f}")
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_INSERT_BORE_D / 2.0,
            TWEETER_ADDON_JOINT_Z[0] - 0.2,
            TWEETER_ADDON_JOINT_Z[0] + 4.0)

    # Keep only the acoustic crescent outside the UM ownership domain. The T
    # cable leaves its flush upper-UM mouth and floats behind this add-on; no
    # printed arc, point horn, trench or hidden crescent suffix remains.
    part -= _cylinder_at(UM_CUTOUT[0], UM_CUTOUT[1], UM_CORE_R + 0.20,
                         6.7, 20.0)
    part = part.clean()
    solids = list(part.solids())
    if (not part.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "tweeter finalization must retain every required feature; "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in part.solids()]}")
    return Part([solids[0]])


def v1lf_attachments():
    _require_guarded_build()
    out = {}
    if STAND_FOOT:
        out["addon_mount_floor_support"] = floor_support()
    out["addon_tweeter_crescent"] = tweeter_crescent()
    return out


def gen_step():
    _require_guarded_build()
    children = []
    for label, solid in v1lf_attachments().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = (
        "lx521_v1lf_r6f_required_floor_support_and_optional_addons_floor"
        if STAND_FOOT else
        "lx521_v1lf_r6f_optional_addons_no_floor")
    return assembly
