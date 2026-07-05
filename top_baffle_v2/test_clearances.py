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
    for name in ("lm", "um", "t1", "t2"):
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

    names = ("lm", "um", "t1", "t2")
    for stand_foot in (True, False):
        routes = _routes(stand_foot)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                required = (CABLE_D[a] + CABLE_D[b]) / 2.0 + 1.5
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
            duct_top_z = DUCT_Z[name] + CABLE_D[name] / 2.0
            if duct_top_z <= pilot_floor_z - 1.5:
                continue  # T ducts pass fully below the pilot floor
            required = L22_PILOT_D_MM / 2.0 + CABLE_D[name] / 2.0 + 1.5
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
    assert MAG_PIN_BASE_DEPTH_MM < MAG_PIN_RECEIVER_DEPTH_MM


def test_magnet_pockets_vs_t_ducts():
    from top_baffle_nd25fw4_b import (
        MAG_PIN_BASE_DEPTH_MM, MAG_PIN_RECEIVER_DEPTH_MM, MAG_RECEIVER_D_MM,
        MAGNET_SITES)
    from top_baffle_nd25fw4_cables import CABLE_D

    t_r = CABLE_D["t1"] / 2.0
    pocket_r = MAG_RECEIVER_D_MM / 2.0
    for stand_foot in (True, False):
        t1 = _routes(stand_foot)["t1"]
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
            print(f"  magnet site ({x:.1f},{y:.1f}) vs T1 duct "
                  f"(foot={stand_foot}): {clear:.2f} mm")
            assert clear >= 0.8, (
                f"magnet pocket ({x},{y}) to T duct {clear:.2f} < 0.8")


def test_seam_keys_vs_ducts():
    """Every seam dovetail (grown female pocket) must keep a wall to
    every duct crossing its seam -- the check that was missing when the
    deep reroute ran the UM arc straight through the old +-97 A-keys."""
    from top_baffle_nd25fw4_b2_split import (DOVETAIL_C, DOVETAILS_A,
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
    cy, _n, h, d = DOVETAIL_C
    rects.append((SEAM_C_X - d - 0.1, SEAM_C_X + d + 0.1,
                  cy - h / 2 - 0.1, cy + h / 2 + 0.1))
    for stand_foot in (True, False):
        for name, pts in _routes(stand_foot).items():
            r = CABLE_D[name] / 2.0
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
    is z-interval containment: the local rear surface (18.3 - t) must
    stay below z_duct - r - skin. The T mains (z=3.7) tolerate no cut
    at all and must be covered by their ribs wherever the taper bites;
    a rib keeps material to z = 3.7 + sqrt(5.4^2 - dx^2) >= 7.2 on-axis."""
    import top_baffle_nd25fw4_c7 as c7
    from top_baffle_nd25fw4 import THICKNESS_MM
    from top_baffle_nd25fw4_cables import CABLE_D, DUCT_Z

    skin = 1.6
    routes = _routes(True)  # below-seam mains are state-independent
    for name, pts in routes.items():
        z_floor_need = DUCT_Z[name] - CABLE_D[name] / 2.0 - skin
        for x, y, _z in pts:
            if not 45.0 < y < 312.0:
                continue
            rear_z = THICKNESS_MM - c7.thickness_at(x, y)
            if rear_z <= z_floor_need + 0.001:
                continue  # duct fully inside the tapered plate
            assert name in ("t1", "t2"), (
                f"{name}: rear surface z={rear_z:.2f} > allowed "
                f"{z_floor_need:.2f} at ({x:.1f},{y:.1f}) -- duct breaks "
                "out of the taper")
            assert (c7.RIB_Y_SPAN[0] - 0.1 <= y <= c7.RIB_Y_SPAN[1] + 0.1), (
                f"{name}: taper bites (rear z={rear_z:.2f}) at "
                f"({x:.1f},{y:.1f}) outside the rib span {c7.RIB_Y_SPAN}")
    print("  C7 corridor (z-containment): big mains stay buried; every "
          "T bite is inside the rib span")


def test_variant_outlines_splice():
    for name in ("top_baffle_nd25fw4_b1", "top_baffle_nd25fw4_b2",
                 "top_baffle_nd25fw4_a_comp"):
        mod = importlib.import_module(name)
        importlib.reload(mod)  # re-runs variant_outline's anchor checks
    print("  variant outlines splice cleanly")


if __name__ == "__main__":
    checks = [
        test_foot_lane_webs,
        test_variant_outlines_splice,
        test_magnet_top_site_walls,
        test_magnet_pockets_vs_t_ducts,
        test_duct_vs_w22_pilots,
        test_duct_duct_separation,
        test_c7_duct_corridor,
        test_seam_keys_vs_ducts,
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
