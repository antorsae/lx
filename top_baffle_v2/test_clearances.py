"""Analytic clearance regression suite for the top-baffle geometry.

Checks the proud/V1L clearances the README/module comments promise. The
final V1LF R6F source and OCC acceptance gates live in
``test_v1lf_r6f.py`` so superseded route experiments cannot define release.

  * duct-duct 3D centerline separation >= r_a + r_b + 1.5 (both
    LX_STAND_FOOT states; planar MAINS only -- the entry-ramp mouths
    intentionally converge at the support window / foot lanes)
  * every W22 pilot bore vs every duct, in plan (or fully z-separated)
  * foot-lane packing webs >= 1.5 (Dx alone, per the packing note)
  * magnet pockets: receiver wall to the shoulder/wing chamfer mating
    face, rear-crescent-taper wall behind pocket floors, front-face
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
    but selects its own terminal-axis handoff.  V1LF still owns its
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
        if name == "um":
            pts = cab.route_centerline_points(
                "um", spacing_mm=0.35,
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
        MAG_FLUSH_DEPTH_MM, MAG_PIN_BASE_DEPTH_MM,
        MAG_PIN_RECEIVER_DEPTH_MM, MAG_POCKET_D_MM, MAG_RECEIVER_D_MM,
        MAGNET_D_MM, MAGNET_SITES, MAGNET_T_MM, TWEETER_DROP_MM)

    # One fit standard applies to every generated variant: the purchased
    # magnet remains D5 x 2 while all base/receiver pockets are D5.2 x 2.2.
    # The extra 0.2 mm depth is adhesive allowance, not a bottoming datum.
    assert MAGNET_D_MM == 5.0
    assert MAGNET_T_MM == 2.0
    assert MAG_POCKET_D_MM == MAG_RECEIVER_D_MM == 5.2
    assert (MAG_FLUSH_DEPTH_MM == MAG_PIN_BASE_DEPTH_MM
            == MAG_PIN_RECEIVER_DEPTH_MM == 2.2)
    assert math.isclose(
        MAG_PIN_BASE_DEPTH_MM - MAGNET_T_MM, 0.2, abs_tol=1e-12)

    x, y, nx, ny, _pin, zc = MAGNET_SITES[1]
    p0, nc = _chamfer_plane()

    def wall(px, py):
        return (px - p0[0]) * nc[0] + (py - p0[1]) * nc[1]

    # receiver: D x depth bore along +n; its down-arc bottom corner is
    # the closest approach to the chamfer mating face (at z = zc)
    r = MAG_RECEIVER_D_MM / 2.0
    tx, ty = -ny, nx  # up-arc tangent
    bx = x + MAG_PIN_RECEIVER_DEPTH_MM * nx - r * tx
    by = y + MAG_PIN_RECEIVER_DEPTH_MM * ny - r * ty
    w = wall(bx, by)
    print(f"  receiver bottom corner -> chamfer face: {w:.2f} mm")
    assert w >= 1.0, f"receiver wall to chamfer face {w:.2f} < 1.0"

    # rear-crescent-taper walls behind the pocket floors (pocket mouths
    # sit at r < r_k, i.e. in the full-depth zone -- no radial fade)
    cy_dropped = CRESCENT_SCALLOP_CY - TWEETER_DROP_MM
    theta = math.degrees(math.atan2(y - cy_dropped, x))
    cut = _crescent_taper_depth(theta)
    for label, dia in (("receiver", MAG_RECEIVER_D_MM),
                       ("base pocket", MAG_POCKET_D_MM)):
        floor_z = zc - dia / 2.0
        w = floor_z - cut
        print(f"  {label} floor z={floor_z:.2f} vs taper cut {cut:.2f}: "
              f"wall {w:.2f} mm")
        assert w >= 1.4, f"{label} rear taper wall {w:.2f} < 1.4"

    w_front = THICKNESS_MM - (zc + MAG_RECEIVER_D_MM / 2.0)
    print(f"  receiver front-face wall: {w_front:.2f} mm")
    assert w_front >= 3.0, f"front wall {w_front:.2f} < 3.0"
    # Equal 2.2 mm base/receiver pockets include a 0.2 mm adhesive allowance.
    # Assembly must fixture each 2.0 mm magnet at the mating face; bottoming
    # it would recess the magnet and defeat the flush-interface contract.


def test_magnet_pockets_vs_t_ducts():
    from top_baffle_nd25fw4_b import (
        MAG_PIN_BASE_DEPTH_MM, MAG_PIN_RECEIVER_DEPTH_MM, MAG_RECEIVER_D_MM,
        MAGNET_SITES)
    from top_baffle_nd25fw4_cables import CABLE_D

    t_r = CABLE_D["ts"] / 2.0
    pocket_r = MAG_RECEIVER_D_MM / 2.0
    for stand_foot in (True, False):
        t1 = _routes(stand_foot)["ts"]
        for x, y, nx, ny, _pin, zc in MAGNET_SITES:
            # combined pocket envelope: base bore + receiver, one segment
            a = np.array([x - MAG_PIN_BASE_DEPTH_MM * nx,
                          y - MAG_PIN_BASE_DEPTH_MM * ny, zc])
            b = np.array([x + MAG_PIN_RECEIVER_DEPTH_MM * nx,
                          y + MAG_PIN_RECEIVER_DEPTH_MM * ny, zc])
            ab = b - a
            t = np.clip((t1 - a) @ ab / (ab @ ab), 0.0, 1.0)
            d = np.linalg.norm(t1 - (a + t[:, None] * ab), axis=1)
            clear = float(d.min()) - pocket_r - t_r
            print(f"  magnet site ({x:.1f},{y:.1f}) vs TS duct "
                  f"(foot={stand_foot}): {clear:.2f} mm")
            assert clear >= 0.8, (
                f"magnet pocket ({x},{y}) to T duct {clear:.2f} < 0.8")


def test_v0_duct_corridor():
    """Variant V0's REAR bevel vs the ducts (z-containment, same rule
    as C7): the rear cut must stay below z_bottom - 1.6. Pin pockets
    (rear face, z 0..2.2) must be plan-clear of every duct."""
    import top_baffle_nd25fw4_v0 as v0
    from top_baffle_nd25fw4_cables import CABLE_D

    for name, pts in _routes(True).items():
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
        for sx in (1, -1):
            for mx, my in v0.V0_MAGNET_SITES:
                m = min(math.dist((sx*mx, my), (x, y)) for x, y, _ in pts)
                assert m >= v0.MAG_POCKET_D_MM / 2.0 + r + 1.5, (
                    f"V0 pocket ({sx*mx},{my}) {m:.2f} from {name}")
    print("  V0 rear bevel: duct floors covered (5 lateral offsets); "
          "pockets clear")


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
                if name == "um" and z < 12.54:
                    # The selected R14 is the intentional rear opening;
                    # exact axis, containment and curvature tests cover it.
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
    """Whole-centerline bend radii, including the UM outlet handoff.

    Three-point circumradii are parameterization independent and do not
    discard the endpoints.  R6 explicitly requires the analytic elbow
    to appear in this test; the former suite stopped 20 samples before
    the separate 90-degree outlet cylinder.
    """
    FLOORS = {"lm": 25.0, "um": 12.5, "ts": 4.5,
              "t1f": 6.0, "t2f": 6.0}

    for stand_foot in (True, False):
        for name, pts in _routes(stand_foot).items():
            probe = pts if name == "um" else pts[20:-20]
            r_min = _min_three_point_radius(probe)
            assert r_min >= FLOORS[name], (
                f"{name} min bend radius {r_min:.1f} < {FLOORS[name]} "
                f"(foot={stand_foot})")
        print(f"  route smoothness OK (foot={stand_foot})")

    # Proud's full outlet must remain here. V1LF's closed Z-first paths
    # are owned by the guarded final R6F suite in test_v1lf_r6f.py.
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
    # R6F V1LF is two rings with half-laps; it has no B2 seam keys.
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
    """Every wing/shoulder must meet the B2 walls at exactly
    wall + CLEARANCE (the printed-wing 4.75 mm gap bug class): thin
    horizontal slices of each attachment, inner-face |x| vs the
    outline law. Straight flare/chamfer spans only."""
    _routes(True)  # normalize the reload state
    from build123d import Box, Pos

    from top_baffle_nd25fw4_attachments import attachments
    from top_baffle_nd25fw4_v1_attachments import v1_attachments
    wall_x = _cab()._wall_x

    # Attachments mate at EXACT fit (variant - b2, no grown clearance).
    # Expected inner-face offsets vs the cables _wall_x LAW (calibrated
    # 2026-07-10; the law simplifies the true B2 outline): flare spans
    # sit at -0.089, upper-chamfer spans at -0.573 -- identically on
    # every piece, side and family. Any drift is the printed-wing
    # 4.75 mm gap bug class.
    FLARE = ((320.0, 330.0, 350.0, 370.0, 381.0), -0.089)
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
    print("  attachment mating faces flush on the calibrated outline "
          "offsets (exact fit)")


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
    from top_baffle_nd25fw4_b import MAG_POCKET_D_MM

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
    # V1 upper-pocket walls (site2 zc=14.4, local rear ~10.1)
    pocket_r = MAG_POCKET_D_MM / 2.0
    entries.append((18.3 - (14.4 + pocket_r) - 1.0,
                    "V1 upper pocket front wall (-1.0 rule)"))
    entries.append((14.4 - pocket_r - 10.1 - 1.4,
                    "V1 upper pocket floor wall (-1.4 rule)"))
    # R6F core facts and proud exact normal-distance outlet margin.
    import top_baffle_nd25fw4_flush as fl
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as vroute
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
                    "V1LF exact plan-normal tunnel wall (-0.8)"))
    entries.append((facts["lm_roof_mm"] - 0.8,
                    "V1LF UM tunnel seat roof (-0.8)"))
    entries.append((facts["bridge_side_wall_mm"] - 0.8,
                    "V1LF UM closed-cover skin (-0.8)"))
    entries.append((facts["ts_lm_roof_mm"] - 0.8,
                    "V1LF tweeter tunnel roof (-0.8)"))
    entries.append((facts["ts_bridge_side_wall_mm"] - 0.8,
                    "V1LF tweeter closed-cover skin (-0.8)"))
    entries.append((core.LM_CORE_R - fl.LM_RECESS_R - 2.4,
                    "V1LF LM outer lip (-2.4 rule)"))
    entries.append((core.UM_CORE_R - fl.UM_RECESS_R - 2.4,
                    "V1LF UM outer lip (-2.4 rule)"))
    entries.append((fl.LM_SEAT_Z - fl.PAD_FACE_Z - fl.LM_BORE_DEPTH_MM
                    - 0.75, "V1LF insert stack over pad (-0.75 rule)"))
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
        test_magnet_pockets_vs_t_ducts,
        test_duct_vs_w22_pilots,
        test_duct_duct_separation,
        test_duct_vs_um_pilots,
        test_c7_duct_corridor,
        test_seam_keys_vs_ducts,
        test_v0_duct_corridor,
        test_v1_field,
        test_route_smoothness,
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
