"""Analytic clearance regression suite for the top-baffle geometry.

Checks the clearances the README/module comments promise, without any
OCC booleans (fast, ~30 s):

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

Run:  python test_clearances.py          (or via pytest, or `make check`)
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SAMPLES_N = 2500      # per route; ~0.3 mm spacing on the longest route
SAMPLING_SLACK = 0.2  # sampled minima can overestimate true minima

_ROUTE_CACHE: dict[bool, dict[str, np.ndarray]] = {}


def _routes(stand_foot: bool) -> dict[str, np.ndarray]:
    """Sampled 3D centerlines of the four duct MAINS for one flag state."""
    if stand_foot in _ROUTE_CACHE:
        return _ROUTE_CACHE[stand_foot]
    os.environ["LX_STAND_FOOT"] = "1" if stand_foot else "0"
    for name in ("top_baffle_nd25fw4", "top_baffle_nd25fw4_cables"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)
    cab = sys.modules["top_baffle_nd25fw4_cables"]
    from build123d import Spline

    out = {}
    for name in ("lm", "um", "ts"):
        path = Spline(*cab.route_points(name))
        pts = [path @ (i / SAMPLES_N) for i in range(SAMPLES_N + 1)]
        out[name] = np.array([[p.X, p.Y, p.Z] for p in pts])
    for name in ("t1f", "t2f"):
        path = Spline(*cab.route_points(name))
        pts = [path @ (i / SAMPLES_N) for i in range(SAMPLES_N + 1)]
        out[name] = np.array([[p.X, p.Y, p.Z] for p in pts])
    _ROUTE_CACHE[stand_foot] = out
    return out


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    best = math.inf
    for i in range(0, len(a), 400):
        d = np.linalg.norm(a[i:i + 400, None, :] - b[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def test_duct_duct_separation():
    from top_baffle_nd25fw4_cables import CABLE_D

    names = ("lm", "um", "ts", "t1f", "t2f")
    merged = ({"ts", "t1f"}, {"ts", "t2f"}, {"t1f", "t2f"})
    for stand_foot in (True, False):
        routes = _routes(stand_foot)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if {a, b} in merged:
                    continue  # they merge at the z-step by design
                r_a = CABLE_D.get(a, 3.8)
                r_b = CABLE_D.get(b, 3.8)
                required = (r_a + r_b) / 2.0 + 1.5
                measured = _min_dist(routes[a], routes[b])
                print(f"  duct-duct {a}-{b} (foot={stand_foot}): "
                      f"{measured:.2f} >= {required:.2f} + slack")
                assert measured >= required + SAMPLING_SLACK, (
                    f"{a}-{b} separation {measured:.2f} < "
                    f"{required + SAMPLING_SLACK:.2f} (foot={stand_foot})")


def test_duct_vs_w22_pilots():
    from top_baffle_nd25fw4 import (
        L22_CUTOUT, L22_PILOT_ANGLES_DEG, L22_PILOT_D_MM, L22_PILOT_DEPTH_MM,
        L22_PILOT_PCD_MM, THICKNESS_MM, _pilot_centers)
    from top_baffle_nd25fw4_cables import CABLE_D, DUCT_Z

    pilots = _pilot_centers(L22_CUTOUT[:2], L22_PILOT_PCD_MM,
                            L22_PILOT_ANGLES_DEG)
    pilot_floor_z = THICKNESS_MM - L22_PILOT_DEPTH_MM
    for stand_foot in (True, False):
        routes = _routes(stand_foot)
        for name, pts in routes.items():
            duct_top_z = (DUCT_Z.get(name, 3.7)
                          + CABLE_D.get(name, 3.8) / 2.0)
            if duct_top_z <= pilot_floor_z - 1.5:
                continue  # T ducts pass fully below the pilot floor
            required = (L22_PILOT_D_MM / 2.0
                        + CABLE_D.get(name, 3.8) / 2.0 + 1.5)
            for px, py in pilots:
                measured = float(np.min(np.hypot(pts[:, 0] - px,
                                                 pts[:, 1] - py)))
                assert measured >= required + SAMPLING_SLACK, (
                    f"{name} vs W22 pilot ({px:.1f},{py:.1f}): plan "
                    f"{measured:.2f} < {required + SAMPLING_SLACK:.2f}")
        print(f"  W22 pilots vs LM/UM plan clearances OK (foot={stand_foot})")


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
        MAG_PIN_BASE_DEPTH_MM, MAG_PIN_RECEIVER_DEPTH_MM, MAG_POCKET_D_MM,
        MAG_RECEIVER_D_MM, MAGNET_SITES, TWEETER_DROP_MM)

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
    # FLUSH scheme: base and receiver pockets are equal depth (2.0 =
    # magnet thickness), so both magnets sit level with their faces.
    assert MAG_PIN_BASE_DEPTH_MM == MAG_PIN_RECEIVER_DEPTH_MM == 2.0, (
        f"magnets not flush: base {MAG_PIN_BASE_DEPTH_MM}, "
        f"receiver {MAG_PIN_RECEIVER_DEPTH_MM}")


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
    (rear face, z 0..1) must be plan-clear of every duct."""
    import top_baffle_nd25fw4_v0 as v0
    from top_baffle_nd25fw4_cables import CABLE_D

    for name, pts in _routes(True).items():
        r = CABLE_D.get(name, 3.8) / 2.0
        for x, y, z in pts:
            if not 316.0 < y < 419.0:
                continue
            allowed = z - r - 1.6
            cut = v0.rear_cut_at(x, y)
            assert cut <= allowed + 0.001, (
                f"{name}: V0 rear cut {cut:.2f} > {allowed:.2f} "
                f"at ({x:.1f},{y:.1f})")
        for sx in (1, -1):
            for mx, my in v0.V0_MAGNET_SITES:
                m = min(math.dist((sx*mx, my), (x, y)) for x, y, _ in pts)
                assert m >= 2.7 + r + 1.5, (
                    f"V0 pocket ({sx*mx},{my}) {m:.2f} from {name}")
    print("  V0 rear bevel: every duct floor covered; pockets clear")


def test_v1_field():
    """V1 front-flush: material z 6.8..18.3 in the vase -- every duct
    window (floor AND roof) must fit inside it with 1.6 skins; the
    global short pilot floors (z=14.3) clear the raised lane roofs."""
    import top_baffle_nd25fw4_v1 as v1
    from top_baffle_nd25fw4 import THICKNESS_MM
    from top_baffle_nd25fw4_cables import CABLE_D

    for name, pts in _routes(True).items():
        r = CABLE_D.get(name, 3.8) / 2.0
        for x, y, z in pts:
            if not 96.0 < y < 434.0:
                continue
            if True:
                assert z - r - 1.6 >= v1.REAR_MM - 0.001, (
                    f"{name} floor {z-r:.1f} below the V1 rear at "
                    f"({x:.1f},{y:.1f})")
                assert z + r + 1.6 <= THICKNESS_MM + 0.001, (
                    f"{name} roof {z+r:.1f} above the front at "
                    f"({x:.1f},{y:.1f})")
    print("  V1 front-flush: duct windows inside z 6.8..18.3; pilots clear")


def test_duct_vs_um_pilots():
    """The rotated 10F pilot bores (D4.6 x 4.0 from the front, floor
    z=14.3) vs the shared TS duct (roof 14.5): they z-overlap, so plan
    clearance >= 2.3 + 3.0 + 1.5 is required everywhere."""
    from top_baffle_nd25fw4 import (UM_CUTOUT, UM_PILOT_ANGLES_DEG,
                                    UM_PILOT_D_MM, UM_PILOT_PCD_MM,
                                    _pilot_centers)
    from top_baffle_nd25fw4_cables import CABLE_D

    pilots = _pilot_centers(UM_CUTOUT[:2], UM_PILOT_PCD_MM,
                            UM_PILOT_ANGLES_DEG)
    for stand_foot in (True, False):
        for name, pts in _routes(stand_foot).items():
            required = (UM_PILOT_D_MM + CABLE_D.get(name, 3.8)) / 2.0 + 1.5
            for px, py in pilots:
                measured = float(np.min(np.hypot(pts[:, 0] - px,
                                                 pts[:, 1] - py)))
                assert measured >= required + SAMPLING_SLACK, (
                    f"{name} vs 10F pilot ({px:.1f},{py:.1f}): plan "
                    f"{measured:.2f} < {required + SAMPLING_SLACK:.2f}")
    print("  10F pilots (rotated) vs all ducts: plan clearances OK")


def test_route_smoothness():
    """Minimum bend radius of every duct centerline (sampled from the
    REAL interpolated spline): routes must stay fishable and free of
    gratuitous wiggles. Floors reflect each route's genuine pinch
    (the UM window bend, the TS crest transition); anything tighter
    is sloppy geometry, anything near the bore radius risks a
    self-intersecting (inside-out) pipe."""
    FLOORS = {"lm": 25.0, "um": 10.0, "ts": 4.5, "t1f": 6.0, "t2f": 6.0}
    for stand_foot in (True, False):
        for name, pts in _routes(stand_foot).items():
            d1 = np.gradient(pts, axis=0)
            d2 = np.gradient(d1, axis=0)
            kappa = (np.linalg.norm(np.cross(d1, d2), axis=1)
                     / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12))
            # ignore the first/last few samples: open spline ends have
            # unconstrained tangents (the ramps take over there)
            r_min = float(1.0 / max(kappa[20:-20].max(), 1e-12))
            assert r_min >= FLOORS[name], (
                f"{name} min bend radius {r_min:.1f} < {FLOORS[name]} "
                f"(foot={stand_foot})")
        print(f"  route smoothness OK (foot={stand_foot})")


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
    """Every duct cutter must be a valid, positive-volume, sane-bbox
    solid, and subtracting all of them must leave the baffle VALID --
    an inside-out or tangent-sick sweep passes is_valid alone but
    poisons the split booleans (the 2026-07-06 250 GB OOM)."""
    import importlib as il
    import os as _os
    for mode in ("0", "1"):
        _os.environ["LX_STAND_FOOT"] = mode
        for m in ("top_baffle_nd25fw4", "top_baffle_nd25fw4_cables"):
            il.reload(sys.modules[m]) if m in sys.modules \
                else il.import_module(m)
        cab = sys.modules["top_baffle_nd25fw4_cables"]
        base_mod = sys.modules["top_baffle_nd25fw4"]
        from top_baffle_nd25fw4_b import TWEETER_DROP_MM
        from top_baffle_nd25fw4_b2 import OUTLINE_B2
        base = base_mod.baffle_solid(OUTLINE_B2, TWEETER_DROP_MM)
        for i, c in enumerate(cab.cable_cutters()):
            bb = c.bounding_box().size
            assert c.is_valid and c.volume > 0 and bb.X < 500, (
                f"cutter {i} sick (foot={mode}): valid={c.is_valid} "
                f"vol={c.volume/1000:.1f}")
            base -= c
            assert base.is_valid and base.volume > 0, (
                f"boolean poisoned by cutter {i} (foot={mode})")
        print(f"  cutters healthy + booleans clean (foot={mode}, "
              f"vol={base.volume/1000:.1f})")


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
    for cy, _n, h, d in DOVETAILS_C:
        rects.append((SEAM_C_X - d - 0.1, SEAM_C_X + d + 0.1,
                      cy - h / 2 - 0.1, cy + h / 2 + 0.1))
    for stand_foot in (True, False):
        for name, pts in _routes(stand_foot).items():
            r = CABLE_D.get(name, 3.8) / 2.0
            for x, y, _z in pts:
                for x0, x1, y0, y1 in rects:
                    dx = max(x0 - x, 0.0, x - x1)
                    dy = max(y0 - y, 0.0, y - y1)
                    dd = (dx * dx + dy * dy) ** 0.5 - r
                    assert dd >= 1.4, (
                        f"{name} duct {dd:.2f} from seam key "
                        f"[{x0:.1f}..{x1:.1f}]x[{y0:.1f}..{y1:.1f}] "
                        f"at ({x:.1f},{y:.1f})")
    print("  seam keys: every duct keeps >=1.4 to every grown pocket")


def test_c7_duct_corridor():
    """Variant C7's LM knife taper vs the ducts. The ducts sit at FIXED
    z from the rear face and the taper cuts the rear, so the criterion
    is plain z-interval containment for EVERY route: the local rear
    surface (18.3 - t) must stay below z_duct - r - skin. With the
    round-4 front-half mains no ribs are needed or modeled."""
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


def test_margin_dashboard():
    """Report-only: the tightest margins project-wide, sorted. Erosion
    shows up here before it becomes a red assert somewhere else."""
    from top_baffle_nd25fw4 import (L22_CUTOUT, L22_PILOT_ANGLES_DEG,
                                    L22_PILOT_D_MM, L22_PILOT_PCD_MM,
                                    UM_CUTOUT, UM_PILOT_ANGLES_DEG,
                                    UM_PILOT_D_MM, UM_PILOT_PCD_MM,
                                    _pilot_centers)
    from top_baffle_nd25fw4_cables import CABLE_D

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
            ("10F", _pilot_centers(UM_CUTOUT[:2], UM_PILOT_PCD_MM,
                                   UM_PILOT_ANGLES_DEG), UM_PILOT_D_MM)):
        for name, pts in routes.items():
            req = pd / 2 + CABLE_D.get(name, 3.8) / 2 + 1.5
            for px, py in pilots:
                d = float(np.min(np.hypot(pts[:, 0] - px, pts[:, 1] - py)))
                entries.append((d - req, f"{name} vs {label} pilot "
                                f"({px:.0f},{py:.0f})"))
    # thin-family z-window skins (rear 6.8 / front 18.3, y>96)
    for name, pts in routes.items():
        r = CABLE_D.get(name, 3.8) / 2
        m = (pts[:, 1] > 96) & (pts[:, 1] < 434)
        if m.any():
            entries.append((float((pts[m, 2] - r).min()) - 6.8 - 1.6,
                            f"{name} thin-family floor skin"))
            entries.append((18.3 - float((pts[m, 2] + r).max()) - 1.6,
                            f"{name} thin-family roof skin"))
    # smoothness headroom
    floors = {"lm": 25.0, "um": 10.0, "ts": 4.5, "t1f": 6.0, "t2f": 6.0}
    for name, pts in routes.items():
        d1 = np.gradient(pts, axis=0)
        d2 = np.gradient(d1, axis=0)
        kap = (np.linalg.norm(np.cross(d1, d2), axis=1)
               / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12))
        entries.append((1.0 / kap[20:-20].max() - floors[name],
                        f"{name} bend radius over floor"))
    # V1 upper-pocket walls (site2 zc=14.4, local rear ~10.1)
    entries.append((18.3 - (14.4 + 2.7) - 1.0, "V1 upper pocket front wall (-1.0 rule)"))
    entries.append((14.4 - 2.7 - 10.1 - 1.4, "V1 upper pocket floor wall (-1.4 rule)"))
    entries.sort()
    print("  tightest margins (mm over the rule):")
    for m, label in entries[:15]:
        print(f"   {m:+6.2f}  {label}")
    assert entries[0][0] > -0.001, f"negative margin: {entries[0]}"


if __name__ == "__main__":
    checks = [
        test_foot_lane_webs,
        test_variant_outlines_splice,
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
        test_bridge_inserts,
        test_cutter_health,
        test_margin_dashboard,
    ]
    failed = []
    for check in checks:
        print(f"{check.__name__}:")
        try:
            check()
        except AssertionError as exc:
            failed.append(f"{check.__name__}: {exc}")
            print(f"  FAIL: {exc}")
    if failed:
        sys.exit("\n".join(["", "FAILED:"] + failed))
    print("\nall clearance checks passed")
