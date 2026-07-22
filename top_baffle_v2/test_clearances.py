"""Analytic clearance regression suite for the top-baffle geometry.

Checks the proud/V1L clearances the README/module comments promise. The
final Obi-Wan R6F source and OCC acceptance gates live in
``test_obiwan_r6f.py`` so superseded route experiments cannot define release.

  * duct-duct 3D centerline separation >= r_a + r_b + 1.5 (both
    LX_STAND_FOOT states; planar MAINS only -- the entry-ramp mouths
    intentionally converge at the support window / foot lanes)
  * every W22 pilot bore vs every duct, in plan (or fully z-separated)
  * foot-lane packing webs >= 1.5 (Dx alone, per the packing note)
  * captive magnet cavities: receiver wall to the shoulder/wing chamfer
    mating face, rear-crescent-taper wall behind cavity cradles, front-face
    wall, and 3D clearance to the T ducts
  * variant outlines still splice (variant_outline raises on unmatched
    anchors)

Run:  python test_clearances.py          (or `make check`)
"""

from __future__ import annotations

# This suite owns a fresh-process guarded runner below. Generic pytest must
# not accumulate its OCC checks in one unbounded interpreter.
__test__ = False

import importlib
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SAMPLES_N = 2500      # per route; ~0.3 mm spacing on the longest route
SAMPLING_SLACK = 0.2  # sampled minima can overestimate true minima

_ROUTE_CACHE: dict[tuple[bool, str, str], dict[str, np.ndarray]] = {}


def _large_host_execution() -> bool:
    """True only for the guarded high-memory remote worker profile."""
    return (
        os.environ.get("LX_CAD_EXECUTION") != "local"
        and os.environ.get("LX_CAD_MEMORY_PROFILE") == "osado-512g"
        and os.environ.get("LX_CAD_ALLOW_PARALLEL") == "1"
    )


def _routes(stand_foot: bool, routing_profile: str = "proud",
            um_handoff_key: str = "proud") -> dict[str, np.ndarray]:
    """Sampled centerlines keyed by stand state, profile, and UM tail.

    ``um_handoff_key`` is intentionally independent of
    ``LX_ROUTING_PROFILE``: V1L remains in the subtractive proud family
    but selects its own terminal-axis handoff.  Obi-Wan still owns its
    separate integral route and therefore rejects a keyed proud tail.
    """
    if routing_profile != "proud" and um_handoff_key != "proud":
        raise ValueError("keyed UM handoffs apply only to profile proud")
    key = (stand_foot, routing_profile, um_handoff_key)
    current = sys.modules.get("top_baffle_nd25fw4_cables")
    normalized = (current is not None
                  and current.ROUTING_PROFILE == routing_profile
                  and bool(current.STAND_FOOT) == stand_foot)
    if key in _ROUTE_CACHE and normalized:
        return _ROUTE_CACHE[key]
    os.environ["LX_STAND_FOOT"] = "1" if stand_foot else "0"
    os.environ["LX_ROUTING_PROFILE"] = routing_profile
    for name in ("top_baffle_nd25fw4", "top_baffle_nd25fw4_cables"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)
    cab = sys.modules["top_baffle_nd25fw4_cables"]
    if key in _ROUTE_CACHE:
        return _ROUTE_CACHE[key]
    from build123d import Spline

    out = {}
    for name in ("lm", "um", "ts"):
        if name in {"lm", "um"}:
            pts = cab.route_centerline_points(
                name, spacing_mm=0.35,
                um_handoff_key=um_handoff_key)
            out[name] = np.asarray(pts, dtype=float)
        else:
            path = Spline(*cab.route_points(name))
            pts = [path @ (i / SAMPLES_N) for i in range(SAMPLES_N + 1)]
            out[name] = np.array([[p.X, p.Y, p.Z] for p in pts])
    for name in ("t1f", "t2f"):
        path = Spline(*cab.route_points(name))
        pts = [path @ (i / SAMPLES_N) for i in range(SAMPLES_N + 1)]
        out[name] = np.array([[p.X, p.Y, p.Z] for p in pts])
    _ROUTE_CACHE[key] = out
    return out


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    best = math.inf
    for i in range(0, len(a), 400):
        d = np.linalg.norm(a[i:i + 400, None, :] - b[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def _min_three_point_radius(points: np.ndarray) -> float:
    """Parameterization-independent sampled curvature radius."""
    points = np.asarray(points, dtype=float)
    a = points[1:-1] - points[:-2]
    b = points[2:] - points[1:-1]
    c = points[2:] - points[:-2]
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    radii = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
             * np.linalg.norm(c, axis=1)
             / np.maximum(2.0 * cross, 1e-12))
    curved = radii[cross > 1e-8]
    return math.inf if len(curved) == 0 else float(np.min(curved))


def _cab(stand_foot: bool = True, routing_profile: str = "proud",
         um_handoff_key: str = "proud"):
    """Return a module explicitly normalized to the requested profile."""
    _routes(stand_foot, routing_profile, um_handoff_key)
    return sys.modules["top_baffle_nd25fw4_cables"]


def _section_at(name, y, z):
    """(w2, h2, zc) of a duct at height y: TS follows its oval law
    (round/oval/morph per ts_section), everything else is the round
    CABLE_D bore centered at the sampled z."""
    if name == "ts":
        return _cab().ts_section(y)
    r = _cab().CABLE_D.get(name, 3.8) / 2.0
    return r, r, z


def _ts_floor_skin(name, h2):
    """Floor-skin rule: 1.6 everywhere except the TS oval span (1.45
    by design -- the price of passing under the MU10 flange seat)."""
    return 1.4 if (name == "ts" and h2 < 2.9) else 1.6


def test_duct_duct_separation():
    from top_baffle_nd25fw4_cables import CABLE_D
    from top_baffle_nd25fw4 import (
        UM_CUTOUT,
        UM_PILOT_ANGLES_DEG,
        UM_PILOT_D_MM,
        UM_PILOT_PCD_MM,
        _pilot_centers,
    )

    names = ("lm", "um", "ts", "t1f", "t2f")
    merged = ({"ts", "t1f"}, {"ts", "t2f"}, {"t1f", "t2f"})
    cab = _cab(True, "proud")
    for stand_foot in (True, False):
        route_sets = (
            ("standard", _routes(stand_foot, "proud")),
            ("V1L", _routes(stand_foot, "proud",
                            cab.UM_V1L_HANDOFF_KEY)),
        )
        for variant, routes in route_sets:
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if {a, b} in merged:
                        continue  # they merge at the z-step by design
                    r_a = CABLE_D.get(a, 3.8)
                    r_b = CABLE_D.get(b, 3.8)
                    required = (r_a + r_b) / 2.0 + 1.5
                    measured = _min_dist(routes[a], routes[b])
                    print(f"  {variant} duct-duct {a}-{b} "
                          f"(foot={stand_foot}): {measured:.2f} >= "
                          f"{required:.2f} + slack")
                    assert measured >= required + SAMPLING_SLACK, (
                        f"{variant} {a}-{b} separation {measured:.2f} < "
                        f"{required + SAMPLING_SLACK:.2f} "
                        f"(foot={stand_foot})")


def test_duct_vs_w22_pilots():
    from top_baffle_nd25fw4 import (
        L22_CUTOUT, L22_PILOT_ANGLES_DEG, L22_PILOT_D_MM, L22_PILOT_DEPTH_MM,
        L22_PILOT_PCD_MM, THICKNESS_MM, _pilot_centers)
    from top_baffle_nd25fw4_cables import CABLE_D, DUCT_Z

    pilots = _pilot_centers(L22_CUTOUT[:2], L22_PILOT_PCD_MM,
                            L22_PILOT_ANGLES_DEG)
    pilot_floor_z = THICKNESS_MM - L22_PILOT_DEPTH_MM
    cab = _cab(True, "proud")
    for stand_foot in (True, False):
        route_sets = (
            ("standard", _routes(stand_foot, "proud")),
            ("V1L", _routes(stand_foot, "proud",
                            cab.UM_V1L_HANDOFF_KEY)),
        )
        for variant, routes in route_sets:
            for name, pts in routes.items():
                duct_top_z = (DUCT_Z.get(name, 3.7)
                              + CABLE_D.get(name, 3.8) / 2.0)
                if duct_top_z <= pilot_floor_z - 1.5:
                    continue  # T ducts pass fully below the pilot floor
                required = (L22_PILOT_D_MM / 2.0
                            + CABLE_D.get(name, 3.8) / 2.0 + 1.5)
                for px, py in pilots:
                    measured = float(np.min(np.hypot(
                        pts[:, 0] - px, pts[:, 1] - py)))
                    assert measured >= required + SAMPLING_SLACK, (
                        f"{variant} {name} vs W22 pilot "
                        f"({px:.1f},{py:.1f}): plan {measured:.2f} < "
                        f"{required + SAMPLING_SLACK:.2f}")
        print(f"  W22 pilots vs standard/V1L LM/UM clearances OK "
              f"(foot={stand_foot})")


def test_foot_lane_webs():
    from top_baffle_nd25fw4_cables import FOOT_LANES

    lanes = sorted((x, dia) for x, _z, _y, _r, dia in FOOT_LANES.values())
    for (x0, d0), (x1, d1) in zip(lanes, lanes[1:]):
        web = (x1 - x0) - (d0 + d1) / 2.0
        print(f"  foot lanes x={x0:+.2f}/{x1:+.2f}: web {web:.2f}")
        assert web >= 1.499, f"foot-lane web {web:.3f} < 1.5"


def _chamfer_plane():
    """B2's chamfer edge (the shoulder/wing mating face), as (P0, inward
    unit normal) -- the wall the top magnet receiver must not pierce."""
    from top_baffle_nd25fw4_b import B2_RIGHT_SEGS

    (_, p_apex, p_crest) = B2_RIGHT_SEGS[1]  # ("L", apex, crest)
    ux, uy = p_crest[0] - p_apex[0], p_crest[1] - p_apex[1]
    norm = math.hypot(ux, uy)
    n = (-uy / norm, ux / norm)  # rotate left: points into the shoulder
    return p_apex, n


def test_magnet_top_site_walls():
    from top_baffle_nd25fw4 import (
        CRESCENT_SCALLOP_CY, THICKNESS_MM, _crescent_taper_depth)
    from top_baffle_nd25fw4_b import (
        MAG_CAVITY_D_MM, MAG_CAVITY_DEPTH_MM, MAG_FACE_SKIN_MM,
        MAG_INNER_SKIN_MM, MAG_INTERFACE_GAP_MM, MAG_LAND_DEPTH_MM,
        MAGNET_D_MM, MAGNET_SITES, MAGNET_T_MM,
        TWEETER_DROP_MM)

    # One fit standard applies to every generated variant: the purchased
    # magnet remains D5 x 2 while all base/receiver cavities are D5.2 x 2.1,
    # buried between one printable 0.45 mm extrusion at each axial face.
    assert MAGNET_D_MM == 5.0
    assert MAGNET_T_MM == 2.0
    assert MAG_CAVITY_D_MM == 5.2
    assert MAG_CAVITY_DEPTH_MM == 2.10
    assert MAG_FACE_SKIN_MM == MAG_INNER_SKIN_MM == 0.45
    assert MAG_INTERFACE_GAP_MM == 0.05
    assert math.isclose(
        MAG_CAVITY_DEPTH_MM - MAGNET_T_MM, 0.10, abs_tol=1e-12)
    assert math.isclose(MAG_LAND_DEPTH_MM, 3.0, abs_tol=1e-12)
    assert math.isclose(
        2.0 * MAG_FACE_SKIN_MM + MAG_INTERFACE_GAP_MM,
        0.95, abs_tol=1e-12)

    x, y, nx, ny, _pin, zc = MAGNET_SITES[1]
    p0, nc = _chamfer_plane()

    def wall(px, py):
        return (px - p0[0]) * nc[0] + (py - p0[1]) * nc[1]

    # Receiver cavity begins behind its 0.45-mm qualified face skin and the
    # solid 0.05-mm spacing standoff. Its down-arc bottom corner is the closest approach
    # to the chamfer mating face (at z = zc).
    r = MAG_CAVITY_D_MM / 2.0
    tx, ty = -ny, nx  # up-arc tangent
    receiver_far = (MAG_INTERFACE_GAP_MM + MAG_FACE_SKIN_MM
                    + MAG_CAVITY_DEPTH_MM)
    bx = x + receiver_far * nx - r * tx
    by = y + receiver_far * ny - r * ty
    w = wall(bx, by)
    print(f"  receiver bottom corner -> chamfer face: {w:.2f} mm")
    assert w >= 1.0, f"receiver wall to chamfer face {w:.2f} < 1.0"

    # rear-crescent-taper walls behind the cavity floors (cavity mouths
    # sit at r < r_k, i.e. in the full-depth zone -- no radial fade)
    cy_dropped = CRESCENT_SCALLOP_CY - TWEETER_DROP_MM
    theta = math.degrees(math.atan2(y - cy_dropped, x))
    cut = _crescent_taper_depth(theta)
    for label, dia in (("receiver", MAG_CAVITY_D_MM),
                       ("base cavity", MAG_CAVITY_D_MM)):
        floor_z = zc - dia / 2.0
        w = floor_z - cut
        print(f"  {label} floor z={floor_z:.2f} vs taper cut {cut:.2f}: "
              f"wall {w:.2f} mm")
        assert w >= 1.4, f"{label} rear taper wall {w:.2f} < 1.4"

    w_front = THICKNESS_MM - (zc + MAG_CAVITY_D_MM / 2.0)
    print(f"  receiver front-face wall: {w_front:.2f} mm")
    # The common front-biased plane deliberately leaves the helper's full
    # 0.60-mm transverse margin ahead of the D5.20 cradle.  This is three
    # 0.20-mm layers and remains outside every cavity cutter.
    assert w_front >= 0.60 - 1.0e-9, (
        f"front wall {w_front:.2f} < 0.60")
    # Both cavities are closed after their insertion pause.  There is no
    # adhesive allowance or post-print access to either magnet.


def test_magnet_cavities_vs_t_ducts():
    from top_baffle_nd25fw4_b import (
        MAG_CAVITY_D_MM, MAG_CAVITY_DEPTH_MM, MAG_FACE_SKIN_MM,
        MAG_INTERFACE_GAP_MM, MAGNET_SITES)
    from top_baffle_nd25fw4_cables import CABLE_D

    t_r = CABLE_D["ts"] / 2.0
    cavity_r = MAG_CAVITY_D_MM / 2.0
    for stand_foot in (True, False):
        t1 = _routes(stand_foot)["ts"]
        for x, y, nx, ny, _pin, zc in MAGNET_SITES:
            # Combined captive-void envelope.  Solid face skins are not part
            # of the void, but they offset both cavities from the interface.
            base_far = MAG_FACE_SKIN_MM + MAG_CAVITY_DEPTH_MM
            receiver_far = (MAG_INTERFACE_GAP_MM + MAG_FACE_SKIN_MM
                            + MAG_CAVITY_DEPTH_MM)
            a = np.array([x - base_far * nx,
                          y - base_far * ny, zc])
            b = np.array([x + receiver_far * nx,
                          y + receiver_far * ny, zc])
            ab = b - a
            t = np.clip((t1 - a) @ ab / (ab @ ab), 0.0, 1.0)
            d = np.linalg.norm(t1 - (a + t[:, None] * ab), axis=1)
            clear = float(d.min()) - cavity_r - t_r
            print(f"  magnet site ({x:.1f},{y:.1f}) vs TS duct "
                  f"(foot={stand_foot}): {clear:.2f} mm")
            assert clear >= 0.8, (
                f"magnet cavity ({x},{y}) to T duct {clear:.2f} < 0.8")


def test_ts_captive_keepout_nudge():
    """Every stock/slim variant retains D6 and the lower-left land.

    One short, smooth positive-X detour moves the duct inward by 0.60 mm;
    exact BREP screening leaves the complete helper land untouched while
    preserving every duct section.
    """
    from build123d import Spline
    from top_baffle_nd25fw4_b import MAGNET_SITES

    cab = _cab(True, "proud")
    standard = cab._ts_route(cab.TS_ROUTE_STANDARD)
    protected = cab._ts_route(cab.TS_ROUTE_CAPTIVE)
    assert len(standard) == len(protected)

    for (x0, y0), (x1, y1) in zip(standard, protected):
        assert math.isclose(y1, y0, abs_tol=1.0e-12)
        expected = cab._ts_captive_nudge_mm(y0)
        assert math.isclose(x1 - x0, expected, abs_tol=1.0e-12)
        assert 0.0 <= expected <= cab.TS_CAPTIVE_NUDGE_MAX_MM
        assert cab.ts_section(y1) == cab.ts_section(y0)

    protected_by_y = {y: x for x, y in protected}
    standard_by_y = {y: x for x, y in standard}
    for y, expected in cab.TS_CAPTIVE_NUDGE_KNOTS:
        assert math.isclose(
            protected_by_y[y] - standard_by_y[y], expected,
            abs_tol=1.0e-12)

    _x, _y, lower_nx, _lower_ny, _pin, _zc = MAGNET_SITES[0]
    restored_normal_web = (
        cab.TS_CAPTIVE_NUDGE_MAX_MM * lower_nx)
    assert restored_normal_web >= 0.57

    path = Spline(*cab.route_points(
        "ts", ts_route_key=cab.TS_ROUTE_CAPTIVE))
    points = np.asarray([
        tuple(path @ (index / SAMPLES_N))
        for index in range(SAMPLES_N + 1)
    ])
    assert _min_three_point_radius(points) >= 4.5
    print("  stock/slim TS captive keepout: D6 unchanged; "
          f"normal nudge={restored_normal_web:.3f} mm")


def test_standard_captive_magnet_contract():
    """Pair geometry, one closure plane, and flush solid receiver skin."""
    from captive_magnets import pair_facts, wall_cavity_tools
    from top_baffle_nd25fw4 import THICKNESS_MM
    from top_baffle_nd25fw4_b import (
        BASE_CAVITY_FACE_INSET_MM, B2_RIGHT_SEGS,
        LOWER_INTERFACE_DATUM_DEVIATION_MAX_MM,
        MAGNET_QUALIFIED_LAND_WIDTH_MM, MAGNET_SITES,
        STANDARD_MAGNET_Z_MM,
        UPPER_INTERFACE_CURVE_DEVIATION_AREA_MM2,
        UPPER_INTERFACE_CURVE_DEVIATION_MAX_MM)
    from top_baffle_nd25fw4_v1 import REAR_MM, V1_MAGNET_ZC

    for index, (x, y, nx, ny, _pin, zc) in enumerate(MAGNET_SITES):
        assert math.isclose(zc, STANDARD_MAGNET_Z_MM, abs_tol=1e-12)
        for side, sx in (("right", 1.0), ("left", -1.0)):
            normal_length = math.hypot(nx, ny)
            outward = (sx * nx / normal_length, ny / normal_length, 0.0)
            inset = BASE_CAVITY_FACE_INSET_MM[index]
            base_kwargs = {
                "face": (
                    sx * x - inset * outward[0],
                    y - inset * outward[1],
                    zc,
                ),
                "outward": outward,
                "print_up": (0.0, 0.0, -1.0),
                "bed_datum": (0.0, 0.0, THICKNESS_MM),
            }
            receiver_kwargs = {
                "face": (sx * x, y, zc),
                "outward": (sx * nx, ny, 0.0),
                "print_up": (0.0, 0.0, -1.0),
                "bed_datum": (0.0, 0.0, THICKNESS_MM),
            }
            base = wall_cavity_tools(
                name=f"standard_{index}_{side}_base", owner="base",
                **base_kwargs)
            receiver = wall_cavity_tools(
                name=f"standard_{index}_{side}_receiver",
                owner="receiver", **receiver_kwargs)
            pair = pair_facts(base, receiver)
            assert math.isclose(
                pair["interface_gap_mm"], 0.05 + inset, abs_tol=1e-9)
            assert math.isclose(
                pair["nominal_magnet_face_separation_mm"],
                0.95 + inset, abs_tol=1e-9)
            assert np.allclose(
                pair["base_marked_pole_axis_xyz"],
                pair["receiver_marked_pole_axis_xyz"], atol=1.0e-12)
            assert math.isclose(
                base.roof_start_print_z_mm, 5.80, abs_tol=1e-9)
            assert math.isclose(
                receiver.roof_start_print_z_mm, 5.80, abs_tol=1e-9)
            assert math.isclose(
                base.required_land.bounding_box().min.Z,
                9.45, abs_tol=0.01)
            assert math.isclose(
                base.required_land.bounding_box().max.Z,
                THICKNESS_MM, abs_tol=0.01)
            # The 0.05-mm receiver offset is solid material, not a local
            # exterior gap cutter.  Receiver cavity tools are therefore the
            # same three cradle/chimney/roof cutters as the carrier.
            assert len(receiver.cutters) == 3

    # V1/V1L use the exact same common source/closure plane.
    thin_print_height = THICKNESS_MM - REAR_MM
    for index, ((x, y, nx, ny, _pin, _zc), zc) in enumerate(
            zip(MAGNET_SITES, V1_MAGNET_ZC)):
        tools = wall_cavity_tools(
            name=f"v1_{index}_right_base",
            face=(
                x - BASE_CAVITY_FACE_INSET_MM[index] * nx,
                y - BASE_CAVITY_FACE_INSET_MM[index] * ny,
                zc,
            ), outward=(nx, ny, 0.0), owner="base",
            print_up=(0.0, 0.0, -1.0),
            bed_datum=(0.0, 0.0, THICKNESS_MM))
        assert tools.required_min_part_top_print_z_mm <= (
            thin_print_height + 1.0e-9)
        assert math.isclose(zc, STANDARD_MAGNET_Z_MM, abs_tol=1e-12)

    # Derive the two tiny datum adaptations from the released outline rather
    # than accepting magic margins.  The upper numerical integration is the
    # exact area between the real ThreePointArc and the tangent land.
    lower_a, lower_b = B2_RIGHT_SEGS[2][1:]
    lx, ly, lnx, lny, _pin, _zc = MAGNET_SITES[0]
    line_dx, line_dy = (
        lower_b[0] - lower_a[0], lower_b[1] - lower_a[1])
    line_len = math.hypot(line_dx, line_dy)
    exact_n = (line_dy / line_len, -line_dx / line_len)
    normal_len = math.hypot(lnx, lny)
    normal = (lnx / normal_len, lny / normal_len)
    tangent = (-normal[1], normal[0])
    half = MAGNET_QUALIFIED_LAND_WIDTH_MM / 2.0

    def line_boundary_u(t):
        px = lx + t * tangent[0]
        py = ly + t * tangent[1]
        signed = ((px - lower_a[0]) * exact_n[0]
                  + (py - lower_a[1]) * exact_n[1])
        return -signed / (normal[0] * exact_n[0]
                          + normal[1] * exact_n[1])

    lower_excess = max(line_boundary_u(-half), line_boundary_u(half))
    assert math.isclose(
        lower_excess, LOWER_INTERFACE_DATUM_DEVIATION_MAX_MM,
        abs_tol=1.0e-6)

    _arc, p1, p2, p3 = B2_RIGHT_SEGS[0]

    def circle_three(a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c
        det = 2.0 * (
            ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = (((ax * ax + ay * ay) * (by - cy)
               + (bx * bx + by * by) * (cy - ay)
               + (cx * cx + cy * cy) * (ay - by)) / det)
        uy = (((ax * ax + ay * ay) * (cx - bx)
               + (bx * bx + by * by) * (ax - cx)
               + (cx * cx + cy * cy) * (bx - ax)) / det)
        return ux, uy, math.hypot(ax - ux, ay - uy)

    cx, cy, radius = circle_three(p1, p2, p3)
    ux, uy, unx, uny, _pin, _uz = MAGNET_SITES[1]
    upper_n_len = math.hypot(unx, uny)
    upper_n = (unx / upper_n_len, uny / upper_n_len)
    upper_t = (-upper_n[1], upper_n[0])

    def arc_boundary_u(t):
        ax = ux + t * upper_t[0] - cx
        ay = uy + t * upper_t[1] - cy
        linear = upper_n[0] * ax + upper_n[1] * ay
        constant = ax * ax + ay * ay - radius * radius
        return -linear + math.sqrt(linear * linear - constant)

    upper_boss = max(-arc_boundary_u(-half), -arc_boundary_u(half))
    assert math.isclose(
        upper_boss, UPPER_INTERFACE_CURVE_DEVIATION_MAX_MM,
        abs_tol=1.0e-6)

    intervals = 2000
    step = 2.0 * half / intervals
    area_sum = 0.0
    for index in range(intervals + 1):
        t = -half + index * step
        height = max(0.0, -arc_boundary_u(t))
        weight = 1 if index in (0, intervals) else (4 if index % 2 else 2)
        area_sum += weight * height
    boss_area = area_sum * step / 3.0
    assert math.isclose(
        boss_area, UPPER_INTERFACE_CURVE_DEVIATION_AREA_MM2,
        abs_tol=1.0e-6)


def test_standard_local_magnet_backing():
    """Magnet booleans may change only fully internal cavity volume.

    The old regression intentionally allowed a 0.75-mm rear cap and a local
    upper tangent boss.  Those additions were visible in the sliced exterior.
    The magnet-free production host is now the immutable acoustic envelope:
    every helper-required land must already fit inside it, the final part may
    not grow at all, and every subtraction must be explained by one of the
    helper's own cavity cutters.  Together these gates reject a front or
    rear breakout, a station-shaped exterior boss/flat, and an ad-hoc relief.
    """
    import gc

    from top_baffle_nd25fw4 import baffle_solid
    from top_baffle_nd25fw4_a_comp import OUTLINE_A_COMP
    from top_baffle_nd25fw4_attachments import _box
    from top_baffle_nd25fw4_b import (
        TWEETER_DROP_MM, _apply_magnets)
    from top_baffle_nd25fw4_b1 import OUTLINE_B1
    from top_baffle_nd25fw4_b2 import OUTLINE_B2
    from top_baffle_nd25fw4_b2_split import pieces
    from top_baffle_nd25fw4_cables import TS_ROUTE_CAPTIVE
    from top_baffle_nd25fw4_v1 import (
        REAR_MM, V1_MAGNET_ZC, field_cutters)
    from top_baffle_nd25fw4_v1_attachments import _v1_base

    def volume(shape):
        return 0.0 if shape is None else float(shape.volume)

    family_planes = {"stock": [], "slim": []}

    def audit(label, family, host, owner, **magnet_kwargs):
        host = host.clean()
        host_solids = len(host.solids())
        assert host.is_valid and host_solids > 0, f"{label}: invalid host"
        final, records = _apply_magnets(host, owner, **magnet_kwargs)
        assert final.is_valid, f"{label}: invalid final captive BREP"
        assert len(final.solids()) == host_solids, (
            f"{label}: captive operation changed material-component count")
        assert len(records) == 4, f"{label}: expected four magnet sites"

        growth = volume(final - host)
        assert growth < 0.02, (
            f"{label}: magnet treatment grew the acoustic exterior by "
            f"{growth:.4f} mm3")

        # Removing the helper cutters from the host-minus-final delta must
        # leave nothing.  This catches local tangent reliefs/dents as well as
        # positive boxes caught by the growth check above.
        unexplained_removal = (host - final).clean()
        for tools in records:
            family_planes[family].append((
                float(tools.seated_magnet_center_xyz[2]),
                float(tools.roof_start_print_z_mm),
                f"{label}/{tools.name}",
            ))
            missing_land = volume(tools.required_land - host)
            assert missing_land < 0.02, (
                f"{label}/{tools.name}: helper land leaves the unmodified "
                f"host by {missing_land:.4f} mm3")

            qualified_solid = tools.required_land
            for cutter in tools.cutters:
                qualified_solid = (qualified_solid - cutter).clean()
                unexplained_removal = (
                    unexplained_removal - cutter).clean()
                obstruction = volume(final & cutter)
                assert obstruction < 0.03, (
                    f"{label}/{tools.name}: cutter remains obstructed by "
                    f"{obstruction:.4f} mm3")
            retained = volume(final & qualified_solid)
            assert retained >= 0.98 * volume(qualified_solid), (
                f"{label}/{tools.name}: only {retained:.3f}/"
                f"{volume(qualified_solid):.3f} mm3 of the sealed land "
                "remains")
            assert volume(final & tools.nominal_magnet) < 0.02, (
                f"{label}/{tools.name}: nominal D5x2 magnet is obstructed")

        unexplained = volume(unexplained_removal)
        assert unexplained < 0.05, (
            f"{label}: {unexplained:.4f} mm3 of non-cavity local relief "
            "changes the exterior")
        before, after = host.bounding_box(), final.bounding_box()
        assert (
            after.min.X >= before.min.X - 0.02
            and after.min.Y >= before.min.Y - 0.02
            and after.min.Z >= before.min.Z - 0.02
            and after.max.X <= before.max.X + 0.02
            and after.max.Y <= before.max.Y + 0.02
            and after.max.Z <= before.max.Z + 0.02
        ), f"{label}: captive treatment changed the exterior bounds"
        print(f"  {label}: four contained lands; no exterior growth/relief")

    stock_b2 = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM)
    audit(
        "stock B2 split top", "stock",
        pieces(only="piece_top_b2", magnet_cavities=False)["piece_top_b2"],
        "base")
    audit(
        "stock A receiver aggregate", "stock",
        ((baffle_solid(OUTLINE_A_COMP, TWEETER_DROP_MM) - stock_b2)
         & _box(303.0, 500.0)),
        "receiver")
    audit(
        "stock B1 receiver aggregate", "stock",
        ((baffle_solid(OUTLINE_B1, TWEETER_DROP_MM) - stock_b2)
         & _box(307.8, 500.0)),
        "receiver")
    del stock_b2
    gc.collect()

    slim_kwargs = {
        "site_zc": V1_MAGNET_ZC,
        "lower_rear_caps": False,
        "name_prefix": "v1",
    }
    audit(
        "slim V1 split top", "slim",
        pieces(
            only="piece_top_b2", magnet_cavities=False,
            shape_cuts=field_cutters(), crescent_rear_mm=REAR_MM,
            ts_route_key=TS_ROUTE_CAPTIVE,
        )["piece_top_b2"],
        "base", **slim_kwargs)
    slim_b2 = _v1_base(OUTLINE_B2)
    audit(
        "slim A receiver aggregate", "slim",
        ((_v1_base(OUTLINE_A_COMP) - slim_b2) & _box(303.0, 500.0)),
        "receiver", **slim_kwargs)
    audit(
        "slim B1 receiver aggregate", "slim",
        (_v1_base(OUTLINE_B1) - slim_b2),
        "receiver", **slim_kwargs)

    for family, values in family_planes.items():
        assert values, family
        source_z, roof_z, _owner = values[0]
        drift = [
            value for value in values
            if (not math.isclose(value[0], source_z, abs_tol=1.0e-9)
                or not math.isclose(value[1], roof_z, abs_tol=1.0e-9))
        ]
        assert not drift, (
            f"{family}: every base/receiver magnet must share one source-Z "
            f"and one closure plane; reference={(source_z, roof_z)}, "
            f"drift={drift}")

    del slim_b2
    gc.collect()


def test_individual_attachment_magnet_burial():
    """Every released A/B1 receiver print buries its own magnet(s).

    Aggregate-wing checks can miss a cavity whose qualified land is later
    clipped by the top/bottom shoulder split.  Compare each final individual
    print to the exact same generator with magnet operations disabled.  The
    treatment may only subtract its declared cavity/gap tools, must preserve
    both acoustic face slabs, and may neither grow nor expose the magnet.
    """
    import gc

    from build123d import Box, Pos
    from captive_magnets import wall_cavity_tools
    from top_baffle_nd25fw4 import THICKNESS_MM
    from top_baffle_nd25fw4_attachments import attachments
    from top_baffle_nd25fw4_b import MAGNET_SITES
    from top_baffle_nd25fw4_v1 import V1_MAGNET_ZC
    from top_baffle_nd25fw4_v1_attachments import v1_attachments

    def volume(shape):
        return 0.0 if shape is None else float(shape.volume)

    def tools_for(index, side, zc, family):
        x, y, nx, ny, _pin, _released_z = MAGNET_SITES[index]
        sign = -1.0 if side == "left" else 1.0
        raw_nx, raw_ny = sign * nx, ny
        normal_length = math.hypot(raw_nx, raw_ny)
        outward = (raw_nx / normal_length, raw_ny / normal_length, 0.0)
        return wall_cavity_tools(
            name=f"{family}_{index}_{side}_individual_receiver",
            face=(sign * x, y, zc), outward=outward, owner="receiver",
            print_up=(0.0, 0.0, -1.0),
            bed_datum=(0.0, 0.0, THICKNESS_MM),
        )

    roles = {
        "attach_a_shoulder_top": (1,),
        "attach_a_shoulder_bottom": (0,),
        "attach_b1_wing": (0, 1),
        "attach_v1a_shoulder_top": (1,),
        "attach_v1a_shoulder_bottom": (0,),
        "attach_v1b1_wing": (0, 1),
    }
    face_guards = (
        Pos(0.0, 250.0, (17.75 + 18.31) / 2.0)
        * Box(400.0, 500.0, 18.31 - 17.75),
        Pos(0.0, 250.0, (6.75 + 9.40) / 2.0)
        * Box(400.0, 500.0, 9.40 - 6.75),
    )

    families = (
        ("stock", attachments(magnet_cavities=False), attachments(),
         tuple(site[-1] for site in MAGNET_SITES)),
        ("slim", v1_attachments(magnet_cavities=False), v1_attachments(),
         V1_MAGNET_ZC),
    )
    for family, hosts, finals, z_centres in families:
        assert set(hosts) == set(finals)
        for name, host in hosts.items():
            final = finals[name]
            role = next(prefix for prefix in roles if name.startswith(prefix))
            side = "left" if name.endswith("_left") else "right"
            selected = tuple(
                tools_for(index, side, z_centres[index], family)
                for index in roles[role]
            )
            assert host.is_valid and final.is_valid, name
            assert len(host.solids()) == len(final.solids()) == 1, name
            assert volume(final - host) < 0.02, (
                f"{family}/{name}: magnet operation grew the print")

            unexplained = (host - final).clean()
            for tools in selected:
                missing = volume(tools.required_land - host)
                assert missing < 0.02, (
                    f"{family}/{name}/{tools.name}: individual print lacks "
                    f"{missing:.4f} mm3 of qualified burial land")
                qualified = tools.required_land
                for cutter in tools.cutters:
                    qualified = (qualified - cutter).clean()
                    unexplained = (unexplained - cutter).clean()
                    assert volume(final & cutter) < 0.03, (
                        f"{family}/{name}/{tools.name}: cavity obstructed")
                assert volume(final & qualified) >= 0.98 * volume(qualified), (
                    f"{family}/{name}/{tools.name}: sealed land clipped by "
                    "the individual-print split")
                assert volume(final & tools.nominal_magnet) < 0.02, (
                    f"{family}/{name}/{tools.name}: magnet volume blocked")

            assert volume(unexplained) < 0.05, (
                f"{family}/{name}: undeclared pocket-shaped exterior relief "
                f"of {volume(unexplained):.4f} mm3")
            for guard in face_guards:
                assert volume((host - final) & guard) < 0.02, (
                    f"{family}/{name}: captive treatment marked an acoustic "
                    "front/rear surface")
            before, after = host.bounding_box(), final.bounding_box()
            assert all(
                actual >= expected - 0.02
                for actual, expected in (
                    (after.min.X, before.min.X),
                    (after.min.Y, before.min.Y),
                    (after.min.Z, before.min.Z),
                )
            )
            assert all(
                actual <= expected + 0.02
                for actual, expected in (
                    (after.max.X, before.max.X),
                    (after.max.Y, before.max.Y),
                    (after.max.Z, before.max.Z),
                )
            )
            print(
                f"  {family}/{name}: {len(selected)} fully buried station(s); "
                "front/rear envelopes unchanged")
        del hosts, finals
        gc.collect()


def test_v0_duct_corridor():
    """Variant V0's REAR bevel vs the ducts (z-containment, same rule
    as C7): the rear cut must stay below z_bottom - 1.6.  Its front-down
    captive stations require a full R3.20 x 5.60-mm axial land.  The invalid
    legacy plan sites were entirely outside the B2 flare.  The first mirrored
    correction connected both lands, but its left station failed the T-route
    rule and its right station visibly restored the rear knife bevel.  The
    final symmetric inboard sites fit the immutable post-bevel host without
    positive backing."""
    import top_baffle_nd25fw4_v0 as v0
    from top_baffle_nd25fw4 import (
        UM_CUTOUT,
        UM_PILOT_ANGLES_DEG,
        UM_PILOT_D_MM,
        UM_PILOT_PCD_MM,
        _pilot_centers,
    )
    from top_baffle_nd25fw4_cables import CABLE_D

    assert v0.PRINT_ORIENTATION == "front-face-down"
    assert set(v0.V0_LEGACY_MAGNET_SITES) == {"right", "left"}
    assert set(v0.V0_FIRST_CORRECTION_MAGNET_SITES) == {"right", "left"}
    assert set(v0.V0_MAGNET_SITES) == {"right", "left"}
    assert math.isclose(
        v0.V0_LEGACY_SITE_OUTSIDE_MM, 5.2630363132, abs_tol=1e-9)
    assert math.isclose(
        v0.V0_LEGACY_CAVITY_DETACHMENT_MM, 2.6630363132,
        abs_tol=1e-9)
    assert math.isclose(
        v0.V0_LEGACY_LAND_DETACHMENT_MM, 2.0630363132,
        abs_tol=1e-9)
    assert math.isclose(
        v0.V0_CAPTIVE_LAND_RADIUS_MM, 3.20, abs_tol=1e-12)
    assert math.isclose(
        v0.V0_LEGACY_TO_FIRST_SHIFT_MM, 8.6630363132, abs_tol=1e-9)
    assert math.isclose(
        math.dist(v0.V0_LEGACY_MAGNET_SITES["right"],
                  v0.V0_FIRST_CORRECTION_MAGNET_SITES["right"]),
        v0.V0_LEGACY_TO_FIRST_SHIFT_MM, abs_tol=1e-6)
    assert np.allclose(
        v0.V0_FIRST_CORRECTION_MAGNET_SITES["right"],
        (37.696679, 326.470436), atol=1e-6)
    assert np.allclose(
        v0.V0_FIRST_CORRECTION_MAGNET_SITES["left"],
        (-37.696679, 326.470436), atol=1e-6)
    assert np.allclose(
        v0.V0_MAGNET_SITES["right"],
        (6.690000, 321.290000), atol=1e-6)
    assert np.allclose(
        v0.V0_MAGNET_SITES["left"],
        (-6.690000, 321.290000), atol=1e-6)
    assert all(
        clearance > 28.0
        for clearance in v0.V0_MAGNET_LAND_OUTLINE_CLEARANCE_MM.values())
    assert math.isclose(
        v0.V0_TS_REQUIRED_CENTER_CLEARANCE_MM, 8.0, abs_tol=1e-12)
    from shapely.geometry import Point
    from top_baffle_nd25fw4_b2_split import (
        DOVETAILS_B, SEAM_B_Y, _below_region, _grown)
    top_female_keepout = _grown(_below_region(SEAM_B_Y, DOVETAILS_B))
    um_pilots = _pilot_centers(
        UM_CUTOUT[:2], UM_PILOT_PCD_MM, UM_PILOT_ANGLES_DEG)
    cutout_margins = {}
    pilot_margins = {}
    seam_margins = {}
    for side, (x, y) in v0.V0_MAGNET_SITES.items():
        cutout_margins[side] = (
            math.dist((x, y), UM_CUTOUT[:2])
            - UM_CUTOUT[2] / 2.0
            - v0.V0_CAPTIVE_LAND_RADIUS_MM)
        pilot_margins[side] = min(
            math.dist((x, y), center)
            - UM_PILOT_D_MM / 2.0
            - v0.V0_CAPTIVE_LAND_RADIUS_MM
            for center in um_pilots)
        split_clearance = (
            Point(x, y).distance(top_female_keepout)
            - v0.V0_CAPTIVE_LAND_RADIUS_MM)
        seam_margins[side] = split_clearance - 1.0
        assert cutout_margins[side] >= (
            v0.V0_KEEPOUT_QUALIFICATION_ALLOWANCE_MM - 1e-6), (
            f"V0 {side} captive land enters the D82 cutout by "
            f"{-cutout_margins[side]:.3f} mm")
        assert pilot_margins[side] >= (
            v0.V0_KEEPOUT_QUALIFICATION_ALLOWANCE_MM - 1e-6), (
            f"V0 {side} captive land/pilot margin only "
            f"{pilot_margins[side]:.3f} mm")
        assert seam_margins[side] >= (
            v0.V0_KEEPOUT_QUALIFICATION_ALLOWANCE_MM - 1e-6), (
            f"V0 {side} captive land only {split_clearance:.3f} mm "
            "from the grown seam-B female dovetail")
    assert all(1.087 < margin < 1.089 for margin in cutout_margins.values())
    assert 12.84 < pilot_margins["left"] < 12.86
    assert 25.65 < pilot_margins["right"] < 25.68
    assert 1.087 < seam_margins["left"] < 1.090
    assert 1.089 < seam_margins["right"] < 1.091
    assert math.isclose(
        v0.V0_CAPTIVE_MAGNET_CENTER_Z_MM, 4.10, abs_tol=1e-12)
    assert math.isclose(
        v0.V0_MAGNET_INWARD_SHIFT_MM, 3.10, abs_tol=1e-12)

    routes = _routes(True)
    rejected_left_ts_distance = min(
        math.dist(v0.V0_FIRST_CORRECTION_MAGNET_SITES["left"], (x, y))
        for x, y, _z in routes["ts"])
    final_ts_distances = {
        side: min(
            math.dist(site, (x, y)) for x, y, _z in routes["ts"])
        for side, site in v0.V0_MAGNET_SITES.items()
    }
    assert 2.60 < rejected_left_ts_distance < 2.61
    assert rejected_left_ts_distance < v0.V0_TS_REQUIRED_CENTER_CLEARANCE_MM
    assert all(
        distance >= (
            v0.V0_TS_REQUIRED_CENTER_CLEARANCE_MM
            + v0.V0_KEEPOUT_QUALIFICATION_ALLOWANCE_MM)
        for distance in final_ts_distances.values())
    assert 26.57 < final_ts_distances["left"] < 26.60
    assert 39.92 < final_ts_distances["right"] < 39.96

    for name, pts in routes.items():
        d1 = np.gradient(pts, axis=0)
        for (x, y, z), t in zip(pts, d1):
            if not 316.0 < y < 419.0:
                continue
            w2, h2, zc = _section_at(name, y, z)
            skin = _ts_floor_skin(name, h2)
            n = math.hypot(t[0], t[1]) or 1.0
            nx, ny = -t[1] / n, t[0] / n
            for o in (-w2, -0.7 * w2, 0.0, 0.7 * w2, w2):
                drop = h2 * math.sqrt(max(1.0 - (o / w2) ** 2, 0.0))
                allowed = zc - drop - skin
                cut = v0.rear_cut_at(x + o * nx, y + o * ny)
                assert cut <= allowed + 0.02, (
                    f"{name}: V0 rear cut {cut:.2f} > {allowed:.2f} "
                    f"at ({x + o * nx:.1f},{y + o * ny:.1f}) (o={o:+.1f})")
        r = max(CABLE_D.get(name, 3.8) / 2.0,
                _cab().TS_OVAL["w2"] if name == "ts" else 0.0)
        for side, (mx, my) in v0.V0_MAGNET_SITES.items():
            m = min(math.dist((mx, my), (x, y)) for x, y, _ in pts)
            required = v0.V0_CAPTIVE_LAND_RADIUS_MM + r + 1.5
            assert m >= (required
                         + v0.V0_KEEPOUT_QUALIFICATION_ALLOWANCE_MM
                         - 1e-6), (
                f"V0 {side} captive land ({mx},{my}) "
                f"{m:.2f} from {name}; need {required:.2f} + "
                f"{v0.V0_KEEPOUT_QUALIFICATION_ALLOWANCE_MM:.2f}")
    print("  V0 rear bevel: duct floors covered (5 lateral offsets); "
          f"front-down R{v0.V0_CAPTIVE_LAND_RADIUS_MM:.2f} captive "
          f"stations clear; rejected left TS={rejected_left_ts_distance:.3f}, "
          f"final TS={final_ts_distances} mm")


def test_v0_captive_geometry():
    """V0 cavities are closed and purely subtractive in the release host.

    This is deliberately a final-BREP gate, not another coordinate check.
    The exact magnet-free split top--after bevel, cable, and seam cuts--must
    already contain each complete land.  Cavity treatment may add nothing or
    mark either exterior face, and both 0.45-mm skins must survive.
    """
    import gc

    from build123d import Align, Box, Cylinder, Pos

    from generate_captive_magnet_catalog import _v0_sites
    from top_baffle_nd25fw4_b2_split import pieces
    import top_baffle_nd25fw4_v0 as v0
    import top_baffle_nd25fw4_v0_split as v0_split

    def volume(shape):
        return 0.0 if shape is None else float(shape.volume)

    host = pieces(
        only="piece_top_b2",
        shape_cuts=list(v0.slide_cutters()),
        magnet_cavities=False,
    )["piece_top_b2"].clean()
    assert host.is_valid and len(host.solids()) == 1
    assert not hasattr(v0, "v0_magnet_backing")

    final, tools_by_side = v0.apply_v0_magnets(host)
    assert final.is_valid and len(final.solids()) == 1
    assert host.volume - final.volume > 80.0, (
        "V0 captive Booleans removed no meaningful cavity volume")
    assert len(tools_by_side) == 2
    assert volume(final - host) < 0.02, (
        "V0 captive adaptation added a local backing or exterior cue")
    host_bb, final_bb = host.bounding_box(), final.bounding_box()
    assert (final_bb.min.X >= host_bb.min.X - 0.02
            and final_bb.max.X <= host_bb.max.X + 0.02
            and final_bb.min.Y >= host_bb.min.Y - 0.02
            and final_bb.max.Y <= host_bb.max.Y + 0.02
            and final_bb.min.Z >= host_bb.min.Z - 0.02
            and final_bb.max.Z <= host_bb.max.Z + 0.02), (
        "V0 cavity operation changed the immutable host bounds")

    unexplained_removal = (host - final).clean()
    rear_guard = Pos(0.0, 250.0, 0.20) * Box(400.0, 500.0, 0.40)
    front_guard = Pos(0.0, 250.0, 18.05) * Box(400.0, 500.0, 0.50)
    assert volume((host - final) & rear_guard) < 0.02, (
        "V0 cavity marked the rear knife-bevel surface")
    assert volume((host - final) & front_guard) < 0.02, (
        "V0 cavity marked the acoustic front surface")

    catalog = _v0_sites()
    assert set(catalog) == {"right", "left"}
    tool_by_name = {tools.name.rsplit("_", 1)[-1]: tools
                    for tools in tools_by_side}
    for side, (x, y) in v0.V0_MAGNET_SITES.items():
        tools = tool_by_name[side]
        assert tools.closure_kind == "axis_opposed_conical_45deg"
        assert np.allclose(tools.pair_axis_xyz, (0.0, 0.0, -1.0))
        assert np.allclose(tools.material_inward_xyz, (0.0, 0.0, 1.0))
        assert np.allclose(tools.print_frame.print_up, (0.0, 0.0, -1.0))
        assert np.allclose(tools.actual_face_xyz, (x, y, 0.0))
        assert np.allclose(
            tools.seated_magnet_center_xyz, (x, y, 4.10))
        assert math.isclose(tools.spec.face_skin_mm, 0.45, abs_tol=1e-12)
        assert math.isclose(tools.spec.inner_skin_mm, 0.45, abs_tol=1e-12)
        assert math.isclose(tools.spec.roof_angle_deg, 45.0, abs_tol=1e-12)
        assert np.allclose(
            catalog[side]["actual_face_xyz_mm"], (x, y, 0.0))
        assert np.allclose(
            catalog[side]["marked_pole_axis_xyz"], (0.0, 0.0, -1.0))

        missing_land = tools.required_land - host
        assert volume(missing_land) < 0.02, (
            f"V0 {side} immutable host lacks "
            f"{volume(missing_land):.4f} mm3 of captive land")
        assert volume(final & tools.nominal_magnet) < 0.02, (
            f"V0 {side} nominal D5x2 magnet was not actually subtracted")
        for index, cutter in enumerate(tools.cutters):
            assert volume(final & cutter) < 0.05, (
                f"V0 {side} cutter {index} remains in the final solid")
            unexplained_removal = (unexplained_removal - cutter).clean()

        # Probe material strictly inside both qualified axial skins.  These
        # checks fail if the cavity breaks out to the rear or through its
        # inner face even when a nominal-magnet intersection happens to pass.
        for label, z0 in (("rear", 0.05), ("inner", 5.20)):
            probe = Pos(x, y, z0) * Cylinder(
                2.40, 0.35,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
            missing_skin = probe - final
            assert volume(missing_skin) < 0.01, (
                f"V0 {side} {label} captive skin is open by "
                f"{volume(missing_skin):.4f} mm3")

    assert volume(unexplained_removal) < 0.05, (
        "V0 magnet treatment removed undeclared exterior material")

    # The split top imports the exact production applicator rather than
    # owning a second, potentially stale pocket implementation.
    assert v0_split.apply_v0_magnets is v0.apply_v0_magnets
    print(
        "  V0 captive BREP: immutable release host contains both R3.20 "
        "lands; subtractive-only voids, skins and datums verified")
    del host, final, tools_by_side
    gc.collect()


def test_v1_field():
    """V1/V1L front-flush fields contain their selected duct windows.

    V1 uses the standard proud UM tail; V1L uses the keyed 283-degree
    tail.  Both have material z=6.8..18.3, except for each intentional
    analytic R14 rear opening.
    """
    import top_baffle_nd25fw4_v1 as v1
    from top_baffle_nd25fw4 import THICKNESS_MM
    from top_baffle_nd25fw4_cables import CABLE_D

    cab = _cab(True, "proud")
    route_sets = (
        ("V1", _routes(True, "proud")),
        ("V1L", _routes(True, "proud", cab.UM_V1L_HANDOFF_KEY)),
    )
    for variant, routes in route_sets:
        for name, pts in routes.items():
            for x, y, z in pts:
                if not 96.0 < y < 434.0:
                    continue
                if name in {"lm", "um"} and z < 12.54:
                    # The selected R14 is an intentional rear opening;
                    # exact mouth and curvature tests cover it.
                    continue
                w2, h2, zc = _section_at(name, y, z)
                skin = _ts_floor_skin(name, h2)
                assert zc - h2 - skin >= v1.REAR_MM - 0.02, (
                    f"{variant} {name} floor {zc-h2:.2f} below rear at "
                    f"({x:.1f},{y:.1f})")
                assert zc + h2 + 1.6 <= THICKNESS_MM + 0.001, (
                    f"{variant} {name} roof {zc+h2:.2f} above front at "
                    f"({x:.1f},{y:.1f})")
    print("  V1/V1L front-flush: selected duct windows inside "
          "z 6.8..18.3 (TS oval floor on the 1.4 rule)")


def test_duct_vs_um_pilots():
    """The rotated MU10 pilot bores (D4.6 x 4.0 from the front, floor
    z=14.3) vs the shared TS duct (roof 14.5): they z-overlap, so plan
    clearance >= 2.3 + 3.0 + 1.5 is required everywhere."""
    from top_baffle_nd25fw4 import (UM_CUTOUT, UM_PILOT_ANGLES_DEG,
                                    UM_PILOT_D_MM, UM_PILOT_PCD_MM,
                                    _pilot_centers)
    from top_baffle_nd25fw4_cables import CABLE_D

    pilots = _pilot_centers(UM_CUTOUT[:2], UM_PILOT_PCD_MM,
                            UM_PILOT_ANGLES_DEG)
    cab = _cab(True, "proud")
    for stand_foot in (True, False):
        route_sets = (
            ("standard", _routes(stand_foot, "proud")),
            ("V1L", _routes(stand_foot, "proud",
                            cab.UM_V1L_HANDOFF_KEY)),
        )
        for variant, routes in route_sets:
            for name, pts in routes.items():
                if name == "ts":  # oval span widens w2 to 3.3 per point
                    w2 = np.array([_section_at(name, y, z)[0]
                                   for _x, y, z in pts])
                else:
                    w2 = CABLE_D.get(name, 3.8) / 2.0
                for px, py in pilots:
                    margin = (np.hypot(pts[:, 0] - px, pts[:, 1] - py)
                              - (UM_PILOT_D_MM / 2.0 + w2 + 1.5))
                    measured = float(np.min(margin))
                    assert measured >= SAMPLING_SLACK, (
                        f"{variant} {name} vs MU10 pilot "
                        f"({px:.1f},{py:.1f}): plan margin "
                        f"{measured:.2f} < {SAMPLING_SLACK:.2f}")
    print("  MU10 pilots vs standard/V1L proud-family ducts: plan "
          "clearances OK (section-aware)")


def test_route_smoothness():
    """Whole-centerline bend radii, including both outlet handoffs.

    Three-point circumradii are parameterization independent and do not
    discard the endpoints.  R6 explicitly requires the analytic elbow
    to appear in this test; the former suite stopped 20 samples before
    the separate 90-degree outlet cylinder.
    """
    FLOORS = {"lm": 13.9, "um": 12.5, "ts": 4.5,
              "t1f": 6.0, "t2f": 6.0}

    for stand_foot in (True, False):
        for name, pts in _routes(stand_foot).items():
            probe = pts if name in {"lm", "um"} else pts[20:-20]
            r_min = _min_three_point_radius(probe)
            assert r_min >= FLOORS[name], (
                f"{name} min bend radius {r_min:.1f} < {FLOORS[name]} "
                f"(foot={stand_foot})")
        print(f"  route smoothness OK (foot={stand_foot})")

    # Proud's full outlet must remain here. Obi-Wan's closed Z-first paths
    # are owned by the guarded final R6F suite in test_obiwan_r6f.py.
    cab = _cab(False, "proud")
    handoff_r = _min_three_point_radius(
        np.asarray(cab.um_handoff_points(n=160)))
    assert handoff_r >= 13.9, (
        f"proud UM outlet elbow {handoff_r:.3f} < R13.9")
    t = np.asarray(cab.UM_HANDOFF["proud"]["tangent"], dtype=float)
    s = cab._um_plan_spline()
    ts = np.asarray(tuple(s % 1.0), dtype=float)
    angle = math.degrees(math.acos(np.clip(np.dot(t, ts)
                                           / np.linalg.norm(ts), -1, 1)))
    assert angle <= 0.25, f"proud plan/elbow tangent error {angle:.3f} deg"
    print("  proud UM handoff: full R14 outlet included; G1 join <=0.25 deg")

    # V1L preserves every upstream R6P station but refits the terminal
    # span so its *physical* z=6.8 mouth reaches the 283-degree axis.
    # Check the keyed spline and analytic elbow independently as well as
    # their G1 join; sampling only the common/default route would let a
    # V1L-only kink escape this regression.
    key = cab.UM_V1L_HANDOFF_KEY
    plan = cab._um_plan_spline(um_handoff_key=key)
    n_plan = max(80, int(plan.length / 0.20))
    plan_pts = np.asarray(
        [tuple(plan @ (i / n_plan)) for i in range(n_plan + 1)])
    plan_r = _min_three_point_radius(plan_pts)
    elbow_pts = np.asarray(cab.um_handoff_points(
        n=240, um_handoff_key=key)[:-1])
    elbow_r = _min_three_point_radius(elbow_pts)
    assert plan_r >= 12.5, f"V1L UM plan bend {plan_r:.3f} < R12.5"
    assert elbow_r >= 13.9, f"V1L UM elbow {elbow_r:.3f} < R13.9"
    wanted = np.asarray(cab.UM_HANDOFF[key]["tangent"], dtype=float)
    actual = np.asarray(tuple(plan % 1.0), dtype=float)
    join_angle = math.degrees(math.acos(np.clip(
        np.dot(wanted, actual)
        / (np.linalg.norm(wanted) * np.linalg.norm(actual)), -1, 1)))
    assert join_angle <= 0.25, (
        f"V1L UM plan/elbow tangent error {join_angle:.3f} deg")
    print(f"  V1L UM terminal span: plan R{plan_r:.2f}, elbow "
          f"R{elbow_r:.2f}, G1 error {join_angle:.3f} deg")


def test_lm_exit_curvature_and_fishing():
    """Every Stock/Slim LM outlet is a traversable R14, never an L-bore.

    The standard and V1L pieces have different rear depths, so their R14
    arcs reach the surface at different angles.  Both must retain the same
    exact mouth XY, a G1 plan/arc/lead handoff, and a D9 terminal section for
    the estimated D7.8 cable.
    """
    for stand_foot in (False, True):
        cab = _cab(stand_foot, "proud")
        for label, key, expected_rear_z in (
                ("Stock", "proud", cab.LM_EXIT_STANDARD_REAR_FACE_Z_MM),
                ("Slim", cab.UM_V1L_HANDOFF_KEY,
                 cab.LM_EXIT_V1L_REAR_FACE_Z_MM)):
            spec = cab._lm_handoff_spec(key)
            arc_intervals = 320
            points = np.asarray(cab.lm_handoff_points(
                n=arc_intervals, um_handoff_key=key), dtype=float)
            elbow_r = _min_three_point_radius(
                points[:arc_intervals + 1])
            assert elbow_r >= cab.LM_EXIT_MIN_BEND_R_MM, (
                f"{label} LM outlet radius {elbow_r:.4f} mm")
            assert math.isclose(
                spec["radius_mm"], cab.LM_EXIT_BEND_R_MM,
                abs_tol=1e-12)
            assert np.allclose(
                spec["face"],
                (cab.LM_DUCT_OUT_X_MM, cab.LM_DUCT_OUT_Y_MM,
                 expected_rear_z), atol=1e-12)

            plan = cab._lm_plan_spline(um_handoff_key=key)
            plan_tangent = np.asarray(tuple(plan % 1.0), dtype=float)
            plan_tangent /= np.linalg.norm(plan_tangent)
            assert np.dot(plan_tangent, (0.0, 1.0, 0.0)) > 0.999999
            arc_face_tangent = (
                points[arc_intervals] - points[arc_intervals - 1])
            arc_face_tangent /= np.linalg.norm(arc_face_tangent)
            lead_tangent = (
                points[arc_intervals + 1] - points[arc_intervals])
            lead_tangent /= np.linalg.norm(lead_tangent)
            join_error = math.degrees(math.acos(np.clip(
                np.dot(arc_face_tangent, lead_tangent), -1.0, 1.0)))
            assert join_error <= 0.20, (
                f"{label} LM arc/lead tangent error {join_error:.4f} deg")

            section_points, section_radii = cab._lm_tube_sections(
                um_handoff_key=key, spacing_mm=1.0)
            assert any(
                np.allclose(point, spec["face"], atol=1e-9)
                for point in section_points)
            assert math.isclose(
                section_radii[-1] * 2.0, cab.LM_EXIT_D_MM,
                abs_tol=1e-12)
            assert (
                section_radii[-1] - 7.8 / 2.0 >= 0.59), (
                f"{label} LM terminal radial cable slack is too small")
            assert section_radii[-1] >= section_radii[-2]
            assert all(
                right + 1e-12 >= left
                for left, right in zip(
                    section_radii[:-1], section_radii[1:])), (
                f"{label} LM D8.2-to-D9 flare is not monotone")
            print(
                f"  {label} LM (foot={stand_foot}): R{elbow_r:.2f}, "
                f"rear angle {spec['face_angle_deg_from_rear_normal']:.2f} "
                "deg, D9 terminal")


def test_v1l_um_terminal_axis_handoff():
    """The V1L mid-right mouth is centered on the requested 283° axis.

    The routing sheet's white standard outlet is below the thin V1L
    rear plane.  Therefore this regression solves the R14 at z=6.8 and
    checks the real printed mouth, rather than merely clocking the
    nominal z=-2 rear endpoint.  It also proves the complete alternate
    tail remains wholly in piece_mid_right with printable seam walls.
    """
    cab = _cab(False, "proud")
    from top_baffle_nd25fw4 import (
        UM_CUTOUT, UM_PILOT_ANGLES_DEG, UM_PILOT_D_MM,
        UM_PILOT_PCD_MM, UM_TERMINAL_CLOCK_DEG, _pilot_centers)
    from top_baffle_nd25fw4_b2_split import SEAM_B_Y, SEAM_C_X

    key = cab.UM_V1L_HANDOFF_KEY
    spec = cab.UM_HANDOFF[key]
    rear = np.asarray(spec["rear_face_axis_point"], dtype=float)
    radial = rear[:2] - np.asarray(UM_CUTOUT[:2], dtype=float)
    station = float(np.linalg.norm(radial))
    bearing = math.degrees(math.atan2(radial[1], radial[0])) % 360.0
    assert abs(station - cab.UM_V1L_AXIS_STATION_MM) < 1e-8
    assert abs(bearing - UM_TERMINAL_CLOCK_DEG) < 1e-8
    assert abs(rear[2] - cab.UM_V1L_REAR_FACE_Z_MM) < 1e-9

    sx, sy, sz = spec["start"]
    tx, ty, _tz = spec["tangent"]
    radius = cab.UM_HANDOFF_R_MM
    cos_phi = 1.0 - ((sz - rear[2]) / radius)
    phi = math.acos(cos_phi)
    solved = np.asarray((
        sx + radius * math.sin(phi) * tx,
        sy + radius * math.sin(phi) * ty,
        sz - radius * (1.0 - math.cos(phi)),
    ))
    assert np.linalg.norm(solved - rear) < 1e-7, (
        f"V1L R14 misses physical rear-axis point by "
        f"{np.linalg.norm(solved-rear):.6g} mm")

    standard = cab.UM_HANDOFF["proud"]
    assert np.linalg.norm(
        np.asarray(spec["start"]) - np.asarray(standard["start"])) > 20.0
    assert standard["rear_end"] == (33.445854, 301.491571, -2.00)
    assert cab.route_points("um")[-1][:2] == standard["start"][:2]
    assert cab.route_points("um", um_handoff_key=key)[-1][:2] == \
        spec["start"][:2]

    pts = np.asarray(cab.route_centerline_points(
        "um", spacing_mm=0.20, um_handoff_key=key))
    anchor = np.asarray((61.76, 283.11))
    i0 = int(np.argmin(np.linalg.norm(pts[:, :2] - anchor, axis=1)))
    tail = pts[i0:]
    duct_r = cab.UM_HANDOFF_D_MM / 2.0
    seam_c_wall = float(np.min(tail[:, 0]) - duct_r - SEAM_C_X)
    seam_b_wall = float(SEAM_B_Y - np.max(tail[:, 1]) - duct_r)
    assert seam_c_wall >= 1.6, (
        f"V1L UM tail seam-C wall {seam_c_wall:.3f} < 1.6")
    assert seam_b_wall >= 1.6, (
        f"V1L UM tail seam-B wall {seam_b_wall:.3f} < 1.6")

    pilots = _pilot_centers(UM_CUTOUT[:2], UM_PILOT_PCD_MM,
                            UM_PILOT_ANGLES_DEG)
    pilot_need = UM_PILOT_D_MM / 2.0 + duct_r + 1.5
    pilot_margin = min(
        float(np.min(np.hypot(tail[:, 0] - px, tail[:, 1] - py)))
        - pilot_need for px, py in pilots)
    assert pilot_margin >= 0.0, (
        f"V1L UM terminal tail/pilot margin {pilot_margin:.3f} < 0")
    print(f"  V1L UM rear mouth: r={station:.3f} mm @ {bearing:.3f} deg; "
          f"mid-right walls C/B={seam_c_wall:.2f}/{seam_b_wall:.2f} mm; "
          f"pilot margin={pilot_margin:.2f} mm")


def test_um_eroded_outline_containment():
    """Exact normal-distance containment for both proud-family UM tails.

    This replaces the deleted ``outline_x - outlet_x`` dashboard
    expression.  That horizontal formula was not a distance normal to
    the sloped B2 edge and falsely passed the R5 side breach.  V1L's
    keyed terminal span is checked separately because it is no longer
    represented by the default proud centerline.
    """
    from shapely.geometry import LineString, Polygon
    from gen_driver_overlay import outline_polygon
    from top_baffle_nd25fw4_b2 import OUTLINE_B2

    poly = Polygon(outline_polygon(OUTLINE_B2, samples=256))
    assert poly.is_valid
    cab = _cab(False, "proud")
    need = cab.UM_HANDOFF_D_MM / 2.0 + 1.6
    eroded = poly.buffer(-need, resolution=64, join_style=2)
    for label, key in (("standard proud", "proud"),
                       ("V1L", cab.UM_V1L_HANDOFF_KEY)):
        pts = np.asarray(cab.route_centerline_points(
            "um", spacing_mm=0.25, um_handoff_key=key))
        route_line = LineString(pts[:, :2])
        # Cover the interpolated line, not only its sampled vertices: a
        # chord crossing a concavity must not hide between good points.
        assert eroded.covers(route_line), (
            f"{label} UM route leaves outline eroded by {need:.2f}")
        center_distance = route_line.distance(poly.boundary)
        wall = center_distance - cab.UM_HANDOFF_D_MM / 2.0
        print(f"  {label} UM route exact normal wall: {wall:.3f} mm")
        assert wall >= 1.6 - 0.01, (
            f"{label} UM exact outline wall {wall:.3f} < 1.6")


def test_ts_eroded_outline_containment():
    """The proud D6 tweeter spline and seam mouths stay inside B2.

    This catches the V1L mid-left bite class: sparse spline controls
    can bow outside the print even when every nominal knot is inside.
    The complete interpolated centerline is tested against an exact
    3.0 + 1.6 mm eroded outline, then every enlarged seam mouth is
    checked separately against the true boundary normal distance.
    """
    from shapely.geometry import LineString, Point, Polygon
    from gen_driver_overlay import outline_polygon
    from top_baffle_nd25fw4_b2 import OUTLINE_B2

    poly = Polygon(outline_polygon(OUTLINE_B2, samples=512))
    assert poly.is_valid
    cab = _cab(False, "proud")
    pts = np.asarray(cab.route_centerline_points("ts", spacing_mm=0.10))
    # The support/foot entry below y=96 is intentionally outside the
    # baffle field; the routed field through seam B ends at y=316.
    field = pts[(pts[:, 1] >= 96.0) & (pts[:, 1] <= 316.0)]
    route_line = LineString(field[:, :2])
    need = cab.CABLE_D["ts"] / 2.0 + 1.6
    # Round erosion matches the circular D6 envelope.  A mitred negative
    # buffer spuriously removes the safe inside of B2's concave waist.
    eroded = poly.buffer(-need, resolution=128, join_style=1)
    assert eroded.covers(route_line), (
        f"proud TS route leaves outline eroded by {need:.2f}")
    wall = route_line.distance(poly.boundary) - cab.CABLE_D["ts"] / 2.0
    print(f"  proud TS route exact normal wall: {wall:.3f} mm")
    assert wall >= 1.6 - 0.01

    for x, y, _z, radius, extra in cab.SEAM_CROSSINGS:
        wall = Point(x, y).distance(poly.boundary) - radius - extra
        assert wall >= 1.6 - 0.01, (
            f"seam relief at ({x:.3f},{y:.3f}) outer wall {wall:.3f}")


def test_bridge_inserts():
    """No-stand bridge heat-set inserts (rear blind bores, z 0..6.8):
    plan-clear of every duct that z-overlaps them (>= r_bore + r_duct
    + 1.5), and the front face stays intact above them."""
    from top_baffle_nd25fw4 import (BRIDGE_HOLE_XY, BRIDGE_INSERT_D_MM,
                                    BRIDGE_INSERT_DEPTH_MM)
    from top_baffle_nd25fw4_cables import CABLE_D

    r_ins = BRIDGE_INSERT_D_MM / 2.0
    pts = _routes(False)  # no-stand
    # include the strip feeders (they run at z=3.7/9.5 in the bottom)
    import importlib as il
    cab = sys.modules["top_baffle_nd25fw4_cables"]
    feeders = {n: np.array(cab.route_points(n)) for n in ("t1f", "t2f")}
    for name, arr in {**pts, **feeders}.items():
        r = CABLE_D.get(name, 3.8) / 2.0
        need = r_ins + r + 1.5
        zov = arr[(arr[:, 2] - r < BRIDGE_INSERT_DEPTH_MM)
                  & (arr[:, 2] + r > 0.0)]
        for cx, cy in BRIDGE_HOLE_XY:
            if len(zov) == 0:
                continue
            d = float(np.min(np.hypot(zov[:, 0] - cx, zov[:, 1] - cy)))
            assert d >= need, (f"bridge insert ({cx},{cy}) vs {name}: "
                               f"{d:.2f} < {need:.2f}")
    print("  bridge inserts: plan-clear of every z-overlapping duct")


def test_cutter_health():
    """Every cutter is valid and the new R6 UM wire subtracts cleanly.

    The TS ruled loft alone creates ~4,800 faces; replaying all fourteen
    sequential booleans twice here duplicates the actual STL export and
    can exhaust OCC memory.  Full-piece exports + manifold checks remain
    the end-to-end boolean gate; this fast test isolates the new cutter.
    """
    import gc
    import importlib as il
    import os as _os
    for mode in ("0", "1"):
        _os.environ["LX_STAND_FOOT"] = mode
        _os.environ["LX_ROUTING_PROFILE"] = "proud"
        for m in ("top_baffle_nd25fw4", "top_baffle_nd25fw4_cables"):
            il.reload(sys.modules[m]) if m in sys.modules \
                else il.import_module(m)
        cab = sys.modules["top_baffle_nd25fw4_cables"]
        base_mod = sys.modules["top_baffle_nd25fw4"]
        from top_baffle_nd25fw4_b import TWEETER_DROP_MM
        from top_baffle_nd25fw4_b2 import OUTLINE_B2
        base = base_mod.baffle_solid(OUTLINE_B2, TWEETER_DROP_MM)
        cutters = cab.cable_cutters()
        for i, c in enumerate(cutters):
            bb = c.bounding_box().size
            assert c.is_valid and c.volume > 0 and bb.X < 500, (
                f"cutter {i} sick (foot={mode}): valid={c.is_valid} "
                f"vol={c.volume/1000:.1f}")
        # seam reliefs are 0..2, LM is 3, R6 UM is index 4
        base -= cutters[4]
        assert base.is_valid and base.volume > 0, (
            f"R6 UM cutter poisoned base boolean (foot={mode})")
        v1l_tube = cab.um_tube(um_handoff_key=cab.UM_V1L_HANDOFF_KEY)
        assert (v1l_tube.is_valid and len(v1l_tube.solids()) == 1
                and v1l_tube.volume > 0), (
            f"V1L keyed UM cutter sick (foot={mode})")
        v1l_base = base_mod.baffle_solid(OUTLINE_B2, TWEETER_DROP_MM)
        v1l_base -= v1l_tube
        assert v1l_base.is_valid and v1l_base.volume > 0, (
            f"V1L keyed UM cutter poisoned base boolean (foot={mode})")
        print(f"  all cutters healthy + standard/V1L UM booleans clean "
              f"(foot={mode})")
        del base, v1l_base, v1l_tube, cab
        gc.collect()


def test_v1l_mid_right_terminal_duct_topology():
    """The keyed handoff is a real open bore in V1L mid_right only.

    Analytic centerline checks cannot catch an OCC subtraction that
    silently leaves a cap or invalid sliver.  Build the actual mid-right
    split through the low-memory single-piece path, then intersect the
    authoritative cutter and a small probe at the physical 283-degree
    rear mouth.
    """
    import gc
    from build123d import Pos, Sphere

    cab = _cab(False, "proud")
    from top_baffle_nd25fw4_v1l_split import pieces_v1l

    parts = pieces_v1l(only="piece_mid_right")
    assert set(parts) == {"piece_mid_right"}
    for name, part in parts.items():
        assert part.is_valid and len(part.solids()) == 1, (
            f"V1L {name} invalid or disconnected")

    mid = parts["piece_mid_right"]
    key = cab.UM_V1L_HANDOFF_KEY
    tube = cab.um_tube(um_handoff_key=key)
    collision = mid & tube
    collision_volume = 0.0 if collision is None else collision.volume
    assert collision_volume < 0.05, (
        f"V1L keyed UM cutter left {collision_volume:.4f} mm3 in mid_right")

    rear = cab.UM_HANDOFF[key]["rear_face_axis_point"]
    mouth_probe = Pos(*rear) * Sphere(0.50)
    mouth_collision = mid & mouth_probe
    mouth_volume = (0.0 if mouth_collision is None
                    else mouth_collision.volume)
    assert mouth_volume < 0.01, (
        f"V1L 283-degree rear mouth retained a cap "
        f"({mouth_volume:.4f} mm3)")
    print(f"  V1L split topology: valid low-memory mid_right; keyed bore "
          f"residual={collision_volume:.4f} mm3; mouth cap="
          f"{mouth_volume:.4f} mm3")
    del parts, mid, tube, collision, mouth_probe, mouth_collision
    gc.collect()


def _test_v1l_upper_dovetail_depth_profile(
        piece_name: str, dovetail_index: int):
    """One seam-B male key obeys the shared V1L/V1 rear plane.

    The low-memory per-piece exporter once applied only the LM-side
    field cutter to the two mids.  Because each male tooth projects
    above seam B into the vase, its projecting 6-mm plan region escaped
    that cutter and retained the stock 18.3-mm depth.  Probe the actual
    emitted geometry inside each tooth, away from the seam and edges.
    """
    import gc
    from build123d import Box, Pos

    _cab(False, "proud")
    from top_baffle_nd25fw4 import THICKNESS_MM
    from top_baffle_nd25fw4_b2_split import DOVETAILS_B, SEAM_B_Y
    from top_baffle_nd25fw4_v1 import REAR_MM
    from top_baffle_nd25fw4_v1l_split import pieces_v1l

    cx, neck, _head, depth = DOVETAILS_B[dovetail_index]
    part = pieces_v1l(
        only=piece_name, include_cables=False)[piece_name]
    probe = Pos(
        cx,
        SEAM_B_Y + depth / 2.0,
        THICKNESS_MM / 2.0,
    ) * Box(neck / 2.0, depth - 0.50, THICKNESS_MM + 2.0)
    tooth = part & probe
    assert tooth is not None and tooth.volume > 1.0, (
        f"V1L {piece_name} upper dovetail missing")
    rear_z = tooth.bounding_box().min.Z
    assert rear_z >= REAR_MM - 0.02, (
        f"V1L {piece_name} upper dovetail rear z={rear_z:.3f}; "
        f"must follow rear plane z={REAR_MM:.3f}")
    assert tooth.bounding_box().max.Z <= THICKNESS_MM + 0.02
    print(f"  V1L {piece_name} upper dovetail rear z={rear_z:.3f}")
    del part, probe, tooth
    gc.collect()


def test_v1l_upper_dovetail_depth_mid_left():
    _test_v1l_upper_dovetail_depth_profile("piece_mid_left", 0)


def test_v1l_upper_dovetail_depth_mid_right():
    _test_v1l_upper_dovetail_depth_profile("piece_mid_right", 1)


def test_seam_keys_vs_ducts():
    """Every seam dovetail (grown female pocket) must keep a wall to
    every duct crossing its seam -- the check that was missing when the
    deep reroute ran the UM arc straight through the old +-97 A-keys."""
    from top_baffle_nd25fw4_b2_split import (DOVETAILS_A, DOVETAILS_C,
                                             DOVETAILS_B, SEAM_A_Y,
                                             SEAM_B_Y, SEAM_C_X)
    from top_baffle_nd25fw4_cables import CABLE_D

    rects = []  # (x0, x1, y0, y1) grown pockets, both directions
    for cx, _n, h, d in DOVETAILS_A:
        rects.append((cx - h / 2 - 0.1, cx + h / 2 + 0.1,
                      SEAM_A_Y - d - 0.1, SEAM_A_Y + d + 0.1))
    for cx, _n, h, d in DOVETAILS_B:
        rects.append((cx - h / 2 - 0.1, cx + h / 2 + 0.1,
                      SEAM_B_Y - d - 0.1, SEAM_B_Y + d + 0.1))
    # R6F Obi-Wan is two rings with half-laps; it has no B2 seam keys.
    for cy, _n, h, d in DOVETAILS_C:
        rects.append((SEAM_C_X - d - 0.1, SEAM_C_X + d + 0.1,
                      cy - h / 2 - 0.1, cy + h / 2 + 0.1))
    cab = _cab(True, "proud")
    for stand_foot in (True, False):
        route_sets = (
            ("standard", _routes(stand_foot, "proud")),
            ("V1L", _routes(stand_foot, "proud",
                            cab.UM_V1L_HANDOFF_KEY)),
        )
        for variant, routes in route_sets:
            for name, pts in routes.items():
                r = CABLE_D.get(name, 3.8) / 2.0
                for x, y, _z in pts:
                    for x0, x1, y0, y1 in rects:
                        dx = max(x0 - x, 0.0, x - x1)
                        dy = max(y0 - y, 0.0, y - y1)
                        dd = (dx * dx + dy * dy) ** 0.5 - r
                        assert dd >= 1.4, (
                            f"{variant} {name} duct {dd:.2f} from seam "
                            f"key [{x0:.1f}..{x1:.1f}]x"
                            f"[{y0:.1f}..{y1:.1f}] at ({x:.1f},{y:.1f})")
    print("  seam keys: standard/V1L ducts keep >=1.4 to every pocket")


def test_c7_duct_corridor():
    """Variant C7's LM knife taper vs the ducts. The ducts sit at FIXED
    z from the rear face and the taper cuts the rear, so the criterion
    is plain z-interval containment for EVERY route: the local rear
    surface (18.3 - t) must stay below z_duct - r - skin. With the
    proud R6P front-half planar spans no ribs are needed or modeled."""
    import top_baffle_nd25fw4_c7 as c7
    from top_baffle_nd25fw4 import THICKNESS_MM
    from top_baffle_nd25fw4_cables import CABLE_D, DUCT_Z

    skin = 1.6
    routes = _routes(True)  # below-seam mains are state-independent
    for name, pts in routes.items():
        r = CABLE_D.get(name, 3.8) / 2.0
        z_d = DUCT_Z.get(name, 3.7)
        d1 = np.gradient(pts, axis=0)
        for (x, y, z), t in zip(pts, d1):
            if not 45.0 < y < 312.0:
                continue
            # FULL-WIDTH containment: the taper crosses bores laterally
            n = np.hypot(t[0], t[1])
            nx, ny = (-t[1] / n, t[0] / n) if n else (1.0, 0.0)
            for o in (-r, -0.7 * r, 0.0, 0.7 * r, r):
                rear_z = THICKNESS_MM - c7.thickness_at(x + o * nx,
                                                        y + o * ny)
                allowed = z_d - math.sqrt(max(r * r - o * o, 0.0)) - skin
                assert rear_z <= allowed + 0.02, (
                    f"{name}: rear z={rear_z:.2f} > {allowed:.2f} at "
                    f"({x:.1f},{y:.1f}) offset {o:+.1f} -- lateral breach")
    print("  C7 corridor (z-containment): all front-half mains stay "
          "buried under the taper")


def test_variant_outlines_splice():
    for name in ("top_baffle_nd25fw4_b1", "top_baffle_nd25fw4_b2",
                 "top_baffle_nd25fw4_a_comp"):
        mod = importlib.import_module(name)
        importlib.reload(mod)  # re-runs variant_outline's anchor checks
    print("  variant outlines splice cleanly")


def test_attachment_flushness():
    """Every wing/shoulder retains the calibrated B2 outline datum outside
    every captive-magnet station (the printed-wing 4.75-mm gap bug class):
    thin horizontal slices of each attachment,
    inner-face |x| vs the outline law. Straight flare/chamfer spans only."""
    _routes(True)  # normalize the reload state
    from build123d import Box, Pos

    from top_baffle_nd25fw4_attachments import attachments
    from top_baffle_nd25fw4_v1_attachments import v1_attachments
    wall_x = _cab()._wall_x

    # Surrounding attachment outlines mate at exact fit (variant - b2, no
    # grown clearance); the receiver's 0.05-mm cavity-datum offset remains
    # solid and does not alter this exterior.
    # Expected inner-face offsets vs the cables _wall_x LAW (calibrated
    # 2026-07-10; the law simplifies the true B2 outline): flare spans
    # sit at -0.089, upper-chamfer spans at -0.573 -- identically on
    # every piece, side and family. Any drift is the printed-wing
    # 4.75 mm gap bug class.
    # Start at y=330 so the probe remains on the straight calibrated span
    # represented by the simplified wall law.
    FLARE = ((330.0, 350.0, 370.0, 381.0), -0.089)
    CHAMF = ((396.0, 402.0, 408.0, 414.0), -0.573)
    probes = {"wing": (FLARE, CHAMF),
              "shoulder_bottom": (FLARE,),
              "shoulder_top": (CHAMF,)}
    for src in (attachments, v1_attachments):
        for name, solid in src().items():
            kind = next((k for k in probes if k in name), None)
            if kind is None:
                continue
            sgn = -1.0 if "left" in name else 1.0
            hits = 0
            for ys, law_off in probes[kind]:
                for y in ys:
                    slab = solid & (Pos(sgn * 100.0, y, 9.15)
                                    * Box(200.0, 0.6, 40.0))
                    if slab is None or getattr(slab, "volume", 0.0) < 1.0:
                        continue
                    bb = slab.bounding_box()
                    inner = bb.min.X if sgn > 0 else -bb.max.X
                    want = wall_x(y) + law_off
                    assert abs(inner - want) <= 0.03, (
                        f"{name} inner face at y={y}: |x|={inner:.3f} "
                        f"vs expected {want:.3f}")
                    hits += 1
            assert hits >= 3, f"{name}: only {hits} probe slices hit"
    print("  attachment outline datums remain flush through every captive "
          "station; the 0.05-mm receiver standoff is solid")


def test_v1l_service_envelope():
    """V1L's cable enters the terminal corridor; printed TPU must not.

    Zero cable/Faston overlap was the correct rule for the laterally
    offset standard proud outlet, but would contradict the requested
    on-axis V1L handoff.  The installed D7 cable must intentionally
    occupy that corridor while both split strain-relief halves remain
    outside it and preserve the cable bore.
    """
    _cab(False, "proud")
    import top_baffle_nd25fw4_um_fit as fit

    assert fit.PHYSICAL_MEASURE_REQUIRED
    assert fit.V1L_CABLE_REMOVAL_OVERLAP_INTENTIONAL
    assert (fit.REMOVAL_ENVELOPE_CABLE_POLICY["v1l"]
            == "intentional_terminal_handoff_overlap")
    assert fit.REMOVAL_ENVELOPE_GROMMET_POLICY["v1l"] == "must_clear"

    env = fit.removal_envelope()
    cable = fit.rear_cable_envelope("v1l")
    assert cable.is_valid and len(cable.solids()) == 1
    overlap = cable & env
    overlap_volume = 0.0 if overlap is None else overlap.volume
    assert overlap_volume > 10.0, (
        "V1L D7 cable no longer reaches the modeled terminal corridor")

    grommets = fit.split_grommet_parts("v1l")
    assert set(grommets) == {"um_grommet_half_a", "um_grommet_half_b"}
    for name, half in grommets.items():
        assert half.is_valid and len(half.solids()) == 1
        service_hit = half & env
        service_volume = (0.0 if service_hit is None
                          else service_hit.volume)
        assert service_volume < 0.01, (
            f"V1L {name} enters Faston removal volume "
            f"({service_volume:.4f} mm3)")
        cable_hit = half & cable
        cable_volume = 0.0 if cable_hit is None else cable_hit.volume
        assert cable_volume < 0.01, (
            f"V1L {name} pinches D7 cable by {cable_volume:.4f} mm3")
    halves_hit = (grommets["um_grommet_half_a"]
                  & grommets["um_grommet_half_b"])
    halves_volume = 0.0 if halves_hit is None else halves_hit.volume
    assert halves_volume < 0.01, (
        f"V1L split grommet halves overlap by {halves_volume:.4f} mm3")
    print(f"  V1L service semantics: cable/Faston intentional overlap="
          f"{overlap_volume:.2f} mm3; both TPU halves clear service, "
          "cable and each other (physical terminal fit still mandatory)")


def test_emboss_driver_keepouts():
    """Rear ID text must not nick a driver opening in exported STLs."""
    _cab(False, "proud")
    from build123d import Cylinder, Plane, Pos, Rot, Text, extrude, mirror
    from export_piece_stls import EMBOSS_XY, _label
    from top_baffle_nd25fw4 import L22_CUTOUT

    ax, ay, rot, font, short = EMBOSS_XY["2of4_mid_left"]
    label = _label("lx521_top_v1l_2of4_mid_left")
    if short:
        label = label.split(" ")[0]
    text_face = mirror(Text(label, font_size=font), Plane.YZ)
    cutter = extrude(Pos(ax, ay, 6.4) * Rot(Z=rot) * text_face, amount=0.8)
    # R95 opening plus a 1.5 mm no-engraving guard. The compact family/
    # position ID avoids recreating the reported apparent duct-out bite.
    keepout = (Pos(L22_CUTOUT[0], L22_CUTOUT[1], 6.0)
               * Cylinder(L22_CUTOUT[2] / 2.0 + 1.5, 2.0))
    collision = cutter & keepout
    assert collision is None or collision.volume < 1e-7, (
        f"V1L mid-left ID enters LM keepout by {collision.volume:.6f} mm3")
    print("  V1L mid-left rear ID clears the LM opening by >=1.5 mm")


def test_margin_dashboard():
    """Report-only: the tightest margins project-wide, sorted. Erosion
    shows up here before it becomes a red assert somewhere else."""
    from top_baffle_nd25fw4 import (L22_CUTOUT, L22_PILOT_ANGLES_DEG,
                                    L22_PILOT_D_MM, L22_PILOT_PCD_MM,
                                    UM_CUTOUT, UM_PILOT_ANGLES_DEG,
                                    UM_PILOT_D_MM, UM_PILOT_PCD_MM,
                                    _pilot_centers)
    from top_baffle_nd25fw4_cables import CABLE_D
    from captive_magnets import FACE_SKIN_MM, INNER_SKIN_MM
    from top_baffle_nd25fw4_b import (
        MAG_CAVITY_D_MM, STANDARD_MAGNET_Z_MM)

    entries = []
    routes = _routes(True)
    # duct-duct
    names = ("lm", "um", "ts", "t1f", "t2f")
    merged = ({"ts", "t1f"}, {"ts", "t2f"}, {"t1f", "t2f"})
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if {a, b} in merged:
                continue
            req = (CABLE_D.get(a, 3.8) + CABLE_D.get(b, 3.8)) / 2 + 1.5
            entries.append((_min_dist(routes[a], routes[b]) - req,
                            f"duct {a}-{b} separation"))
    # pilots (plan)
    for label, pilots, pd in (
            ("W22", _pilot_centers(L22_CUTOUT[:2], L22_PILOT_PCD_MM,
                                   L22_PILOT_ANGLES_DEG), L22_PILOT_D_MM),
            ("MU10", _pilot_centers(UM_CUTOUT[:2], UM_PILOT_PCD_MM,
                                   UM_PILOT_ANGLES_DEG), UM_PILOT_D_MM)):
        for name, pts in routes.items():
            req = pd / 2 + CABLE_D.get(name, 3.8) / 2 + 1.5
            for px, py in pilots:
                d = float(np.min(np.hypot(pts[:, 0] - px, pts[:, 1] - py)))
                entries.append((d - req, f"{name} vs {label} pilot "
                                f"({px:.0f},{py:.0f})"))
    # thin-family z-window skins (rear 6.8 / front 18.3, y>96) --
    # section-aware: the TS oval runs on the 1.4 floor rule
    for name, pts in routes.items():
        m = (pts[:, 1] > 96) & (pts[:, 1] < 434)
        if name == "um":
            m &= pts[:, 2] >= 12.54  # exclude intentional R14 rear opening
        if m.any():
            floors, roofs = [], []
            for _x, y, z in pts[m]:
                w2, h2, zc = _section_at(name, y, z)
                floors.append(zc - h2 - 6.8 - _ts_floor_skin(name, h2))
                roofs.append(18.3 - (zc + h2) - 1.6)
            entries.append((min(floors), f"{name} thin-family floor skin"))
            entries.append((min(roofs), f"{name} thin-family roof skin"))
    # smoothness headroom
    floors = {"lm": 25.0, "um": 12.5, "ts": 4.5,
              "t1f": 6.0, "t2f": 6.0}
    for name, pts in routes.items():
        probe = pts if name == "um" else pts[20:-20]
        entries.append((_min_three_point_radius(probe) - floors[name],
                        f"{name} bend radius over floor"))
    # Common stock/slim front-biased captive plane.  The acoustic face stays
    # unchanged and retains more than the qualified 0.45-mm Arachne skin;
    # the broad slim taper shelf retains the qualified inner skin without a
    # station-shaped backing feature.
    cavity_r = MAG_CAVITY_D_MM / 2.0
    entries.append((18.3 - (STANDARD_MAGNET_Z_MM + cavity_r)
                    - FACE_SKIN_MM,
                    "stock/slim captive front skin (-0.45 rule)"))
    entries.append((INNER_SKIN_MM - 0.42,
                    "stock/slim captive inner skin (-0.42 path)"))
    # R6F core facts and proud exact normal-distance outlet margin.
    import top_baffle_nd25fw4_flush as fl
    import top_baffle_nd25fw4_obiwan as core
    import top_baffle_nd25fw4_obiwan_route as vroute
    from shapely.geometry import LineString, Polygon
    from gen_driver_overlay import outline_polygon
    from top_baffle_nd25fw4_b2 import OUTLINE_B2

    cabm = _cab(True, "proud")
    ov = cabm.TS_OVAL
    entries.append((ov["zc"] - ov["h2"] - 6.8 - 1.4,
                    "TS oval thin-family floor (-1.4 rule)"))
    poly = Polygon(outline_polygon(OUTLINE_B2, samples=256))
    cab = _cab(False, "proud")
    pts = np.asarray(cab.route_centerline_points("um", spacing_mm=0.25))
    normal_wall = (LineString(pts[:, :2]).distance(poly.boundary)
                   - cab.UM_HANDOFF_D_MM / 2.0)
    entries.append((normal_wall - 1.6,
                    "proud UM exact normal outline wall (-1.6)"))
    v1l_pts = np.asarray(cab.route_centerline_points(
        "um", spacing_mm=0.25,
        um_handoff_key=cab.UM_V1L_HANDOFF_KEY))
    v1l_normal_wall = (LineString(v1l_pts[:, :2]).distance(poly.boundary)
                       - cab.UM_HANDOFF_D_MM / 2.0)
    entries.append((v1l_normal_wall - 1.6,
                    "V1L UM exact normal outline wall (-1.6)"))
    from top_baffle_nd25fw4_b2_split import SEAM_B_Y, SEAM_C_X
    anchor = np.asarray((61.76, 283.11))
    i0 = int(np.argmin(np.linalg.norm(v1l_pts[:, :2] - anchor, axis=1)))
    v1l_tail = v1l_pts[i0:]
    duct_r = cab.UM_HANDOFF_D_MM / 2.0
    entries.append((float(np.min(v1l_tail[:, 0]) - duct_r - SEAM_C_X - 1.6),
                    "V1L UM tail seam-C wall (-1.6)"))
    entries.append((float(SEAM_B_Y - np.max(v1l_tail[:, 1]) - duct_r - 1.6),
                    "V1L UM tail seam-B wall (-1.6)"))
    plan = cab._um_plan_spline(um_handoff_key=cab.UM_V1L_HANDOFF_KEY)
    n_plan = max(80, int(plan.length / 0.20))
    plan_pts = np.asarray(
        [tuple(plan @ (i / n_plan)) for i in range(n_plan + 1)])
    entries.append((_min_three_point_radius(plan_pts) - 12.5,
                    "V1L UM terminal-span bend over R12.5"))
    facts = vroute.route_facts()
    entries.append((facts["min_plan_normal_wall_mm"] - 0.8,
                    "Obi-Wan exact plan-normal tunnel wall (-0.8)"))
    entries.append((facts["lm_roof_mm"] - 0.8,
                    "Obi-Wan UM tunnel seat roof (-0.8)"))
    entries.append((facts["bridge_side_wall_mm"] - 0.8,
                    "Obi-Wan UM closed-cover skin (-0.8)"))
    entries.append((facts["ts_lm_roof_mm"] - 0.8,
                    "Obi-Wan tweeter tunnel roof (-0.8)"))
    entries.append((facts["ts_bridge_side_wall_mm"] - 0.8,
                    "Obi-Wan tweeter closed-cover skin (-0.8)"))
    entries.append((core.LM_CORE_R - fl.LM_RECESS_R - 2.4,
                    "Obi-Wan LM outer lip (-2.4 rule)"))
    entries.append((core.UM_CORE_R - fl.UM_RECESS_R - 2.4,
                    "Obi-Wan UM outer lip (-2.4 rule)"))
    entries.append((fl.LM_SEAT_Z - fl.PAD_FACE_Z - fl.LM_BORE_DEPTH_MM
                    - 0.75, "Obi-Wan insert stack over pad (-0.75 rule)"))
    entries.sort()
    print("  tightest margins (mm over the rule):")
    for m, label in entries[:15]:
        print(f"   {m:+6.2f}  {label}")
    assert entries[0][0] > -0.001, f"negative margin: {entries[0]}"


if __name__ == "__main__":
    checks = [
        test_foot_lane_webs,
        test_variant_outlines_splice,
        test_v1l_service_envelope,
        test_emboss_driver_keepouts,
        test_attachment_flushness,
        test_magnet_top_site_walls,
        test_magnet_cavities_vs_t_ducts,
        test_ts_captive_keepout_nudge,
        test_standard_captive_magnet_contract,
        test_standard_local_magnet_backing,
        test_individual_attachment_magnet_burial,
        test_duct_vs_w22_pilots,
        test_duct_duct_separation,
        test_duct_vs_um_pilots,
        test_c7_duct_corridor,
        test_seam_keys_vs_ducts,
        test_v0_duct_corridor,
        test_v0_captive_geometry,
        test_v1_field,
        test_route_smoothness,
        test_lm_exit_curvature_and_fishing,
        test_v1l_um_terminal_axis_handoff,
        test_um_eroded_outline_containment,
        test_ts_eroded_outline_containment,
        test_bridge_inserts,
        test_margin_dashboard,
        test_cutter_health,
        test_v1l_mid_right_terminal_duct_topology,
        test_v1l_upper_dovetail_depth_mid_left,
        test_v1l_upper_dovetail_depth_mid_right,
    ]
    single = os.environ.get("LX_CLEARANCE_SINGLE_CHECK")
    if single:
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            import subprocess

            guard = Path(__file__).with_name("run_memory_guarded.py")
            proc = subprocess.run(
                [sys.executable, str(guard), "--", sys.executable,
                 str(Path(__file__).resolve())],
                env=os.environ.copy())
            raise SystemExit(proc.returncode)
        check = next((c for c in checks if c.__name__ == single), None)
        if check is None:
            sys.exit(f"unknown single check: {single}")
        print(f"{check.__name__}:")
        try:
            check()
        except AssertionError as exc:
            print(f"  FAIL: {exc}")
            sys.exit(1)
        sys.exit(0)

    # OCC retains substantial native memory after large boolean trees.
    # Run each independent analytic/geometry check in a fresh process so
    # the complete suite is deterministic instead of order/OOM-sensitive.
    import subprocess
    guard = Path(__file__).with_name("run_memory_guarded.py")

    def run_check(check):
        env = os.environ.copy()
        env["LX_CLEARANCE_SINGLE_CHECK"] = check.__name__
        proc = subprocess.run(
            [sys.executable, str(guard), "--", sys.executable,
             str(Path(__file__).resolve())],
            env=env, text=True, capture_output=True)
        return check.__name__, proc.returncode, proc.stdout, proc.stderr

    try:
        requested_workers = int(os.environ.get("LX_CAD_GUARD_SLOTS", "4"))
    except ValueError as exc:
        raise SystemExit("LX_CAD_GUARD_SLOTS must be an integer") from exc
    if requested_workers <= 0:
        raise SystemExit("LX_CAD_GUARD_SLOTS must be positive")
    workers = (min(requested_workers, len(checks))
               if _large_host_execution() else 1)

    results = []
    if workers == 1:
        for check in checks:
            result = run_check(check)
            results.append(result)
            _name, _returncode, stdout, stderr = result
            print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", file=sys.stderr, flush=True)
    else:
        print(
            f"clearance remote runner: {workers} concurrent isolated checks",
            flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_check, check): check.__name__
                for check in checks
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _name, _returncode, stdout, stderr = result
                print(stdout, end="", flush=True)
                if stderr:
                    print(stderr, end="", file=sys.stderr, flush=True)

    failures = {name for name, returncode, _stdout, _stderr in results
                if returncode}
    failed = [check.__name__ for check in checks
              if check.__name__ in failures]
    if failed:
        sys.exit("\nFAILED: " + ", ".join(failed))
    print("\nall clearance checks passed")
