"""DRAFT visualization for the LM knife-edge taper (working name C7):
the lower-mid section keeps its full 18.3 mm around the W22 and then
thins REAR-SIDE (front face stays a full plane, crescent-taper style)
from 24 mm inside the flank/chamfer edges down to a ~0.5 mm knife at
the outline. Draft only -- no geometry code touched: this sheet is for
judging the concept before the duct rerouting work.

Shown: front view with iso-thickness contours + duct keep-in corridors
(t>=12.5 for the O8.5/8.6 mains, t>=7.5 for the O3.8 T ducts), the
CURRENT duct routes that violate them, indicative reroutes, and two
sections (LM axis y=201 and y=100) with the taper profile.

Kept full thickness: everything within the taper-start contour, the
bottom strip below y=60 (stand-foot / bridge interface), the cutout
rim, and the pilot ring (which keeps >=1.9 mm behind the 11-deep
bores at its worst point). Toward seam B the cut FADES OUT over
y=270..308 so the piece returns to the full 18.3 with a land
before the joint -- no step against the (full-thickness) vase piece,
and the seam-B dovetails keep their full section.

Run:  python gen_lm_knife_draft.py  ->  baffle_lm_knife_draft.png
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString, Point, Polygon, box

from gen_driver_overlay import draw, outline_polygon
from top_baffle_nd25fw4 import L22_CUTOUT, THICKNESS_MM
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2
from top_baffle_nd25fw4_cables import CABLE_D, route_points

T_FULL = THICKNESS_MM       # 18.3
T_EDGE = 0.5                # knife feather (protects the front skin)
W_TAPER = 19.0            # taper band width (matches top_baffle_nd25fw4_c7)
Y_KEEP_BOTTOM = 60.0        # flank taper starts above this (foot/bridge)
T_CORRIDOR_BIG = 15.05      # z-containment: mains at z=9.15 allow a rear cut of 3.25
T_CORRIDOR_T = 7.0          # min t for the O3.8 T ducts (rib covers the rest)
Y_REC0, Y_REC1 = 270.0, 308.0   # seam-B recovery (matches the C7 module)
                                # land before seam B (y=315.95)

LM_X, LM_Y, LM_D = L22_CUTOUT


def _flank_x(y):
    return 76.2 + 0.29752 * y


def _edge_x(y):
    """|x| of the LM-section outline at height y (flank, then chamfer)."""
    if y <= 256.12:
        return _flank_x(y)
    return 152.401 - 114.288 * (y - 256.12) / 59.827


# tapered edges: flanks (above y=60) + LM chamfers, both sides.
TAPER_EDGES = MultiLineString([
    [(_flank_x(Y_KEEP_BOTTOM), Y_KEEP_BOTTOM), (152.401, 256.120)],
    [(152.401, 256.120), (38.113, 315.947)],
    [(-_flank_x(Y_KEEP_BOTTOM), Y_KEEP_BOTTOM), (-152.401, 256.155)],
    [(-152.401, 256.155), (-38.122, 315.977)],
])


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 3 * s * s - 2 * s * s * s


def t_of_d(d):
    return T_EDGE + (T_FULL - T_EDGE) * smoothstep(np.asarray(d) / W_TAPER)


def d_of_t(t):
    """Invert t_of_d numerically (scalar)."""
    lo, hi = 0.0, W_TAPER
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_of_d(mid) < t else (lo, mid)
    return (lo + hi) / 2


def thickness_at(x, y):
    """Taper law by edge distance, times the seam-B recovery fade."""
    cut = T_FULL - float(t_of_d(Point(x, y).distance(TAPER_EDGES)))
    rec = 1.0 - float(smoothstep((y - Y_REC0) / (Y_REC1 - Y_REC0)))
    return T_FULL - cut * rec


def front_view(ax):
    draw(ax, OUTLINE_B2, TWEETER_DROP_MM,
         "LM knife-edge taper (C7 draft) — rear-side, front face intact",
         labels=False)
    poly = Polygon(outline_polygon(OUTLINE_B2)).buffer(0)
    poly = poly.difference(Point(LM_X, LM_Y).buffer(LM_D / 2.0))
    poly = poly.intersection(box(-200, -25, 200, 316.0))  # LM section only
    # thickness field on a grid (the seam-B recovery makes it 2D, not a
    # pure distance function)
    xs = np.arange(-156, 156, 2.0)
    ys = np.arange(-2, 318, 2.0)
    tt = np.full((len(ys), len(xs)), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if poly.contains(Point(x, y)):
                tt[j, i] = thickness_at(x, y)
    cut = T_FULL - tt
    cut[cut < 0.25] = np.nan  # keep the full-depth core clean
    ax.contourf(xs, ys, cut, levels=np.linspace(0.25, T_FULL, 12),
                cmap="Oranges", alpha=0.75, zorder=5)
    cs = ax.contour(xs, ys, tt, levels=[T_CORRIDOR_BIG, 18.0],
                    colors="0.2", linestyles=["--", "-"],
                    linewidths=[1.6, 1.2], zorder=7)
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([], [], color="0.2", ls="-", lw=1.2),
        Line2D([], [], color="0.2", ls="--", lw=1.6),
    ]
    proxy_labels = [
        "taper start (t=18)",
        f"t≥{T_CORRIDOR_BIG}: mains buried (rear cut ≤ 3.25)",
    ]
    # the (rerouted) common duct mains -- the T arcs cross the band on
    # their half-round ribs
    for name, color in (("lm", "tab:blue"), ("um", "tab:green"),
                        ("ts", "gold")):
        pts = np.array([p[:2] for p in route_points(name)])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=CABLE_D[name] * 0.7,
                alpha=0.30, zorder=6)
    # kept-full bottom strip
    ax.fill([-80, 80, 80, -80], [0, 0, Y_KEEP_BOTTOM, Y_KEEP_BOTTOM],
            fc="none", ec="0.4", hatch="\\\\\\", lw=0.0, zorder=5,
            alpha=0.35)
    ax.text(0, 30, "kept 18.3 (foot/bridge)", ha="center", fontsize=8,
            color="0.3", zorder=9)
    ax.annotate("knife edge t≈0.5 along the\nflank+chamfer outline",
                (-140, 230), (-165, 330), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.5"))
    ax.annotate("cut fades out y=270→308:\nfull-depth land at seam B\n"
                "(flush joint to the vase)", (-55, 300), (-160, 385),
                fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.5"))
    ax.plot([-200, 200], [315.95, 315.95], color="0.55", lw=0.7,
            ls=(0, (6, 3)), zorder=4)
    ax.text(148, 319, "seam B", fontsize=7, color="0.4", ha="right")
    ax.legend(proxies, proxy_labels, loc="lower left", fontsize=7.5,
              framealpha=0.95)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    ax.set_ylim(-15, 470)


def section(ax, y0, title):
    xs = np.linspace(-155, 155, 800)
    rim = math.sqrt(max((LM_D / 2) ** 2 - (y0 - LM_Y) ** 2, 0.0))
    xmax = _edge_x(y0)
    for sgn in (1, -1):
        seg = xs[(xs * sgn > rim) & (xs * sgn < xmax)]
        if not len(seg):
            continue
        t = np.array([thickness_at(x, y0) for x in seg])
        rear = T_FULL - t
        ax.fill(np.concatenate([seg, seg[::-1]]),
                np.concatenate([rear, np.full_like(seg, T_FULL)[::-1]]),
                fc="0.88", ec="0.35", lw=0.8)
    ax.axhline(T_FULL, color="0.25", lw=0.6)
    ax.text(0.01, 0.93, "front face (kept full plane)",
            transform=ax.transAxes, fontsize=7, color="0.35", va="top")
    # duct positions in this section: interpolate each route's crossing
    # of y=y0 (the raw knots can straddle it by >10 mm on the arcs)
    for name, color in (("lm", "tab:blue"), ("um", "tab:green"),
                        ("ts", "gold")):
        pts = np.array([p[:3] for p in route_points(name)])
        crossings = []
        for a, b in zip(pts, pts[1:]):
            if (a[1] - y0) * (b[1] - y0) <= 0 and a[1] != b[1]:
                f = (y0 - a[1]) / (b[1] - a[1])
                crossings.append(a + f * (b - a))
        for x, _, z in crossings[:2]:
            t_here = thickness_at(x, y0)
            need = CABLE_D[name] + 3.2
            ok = t_here >= need
            ax.add_patch(plt.Circle((x, T_FULL - t_here / 2), CABLE_D[name] / 2,
                                    fill=ok, fc=color, ec=color,
                                    ls="-" if ok else ":",
                                    alpha=0.75 if ok else 0.9, lw=1.2))
            if not ok:
                ax.text(x, T_FULL + 2.5, "✗", ha="center", fontsize=9,
                        color=color)
    ax.set_aspect("equal")
    ax.set_xlim(-155, 155)
    ax.set_ylim(-3, 26)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def taper_law(ax):
    d = np.linspace(0, 50, 200)
    ax.plot(d, t_of_d(d), color="0.2", lw=1.6)
    for t_lvl, label in ((T_CORRIDOR_BIG, "O8.6 corridor"),
                         (T_CORRIDOR_T, "T corridor")):
        dd = d_of_t(t_lvl)
        ax.plot([dd, dd, 0], [0, t_lvl, t_lvl], color="0.5", ls=":", lw=0.9)
        ax.annotate(f"{label}\nd≥{dd:.0f}", (dd, t_lvl), (dd + 3, t_lvl - 3),
                    fontsize=7, color="0.3")
    ax.set_xlabel("distance inside the edge d (mm)", fontsize=8)
    ax.set_ylabel("thickness t (mm)", fontsize=8)
    ax.set_title(f"taper law: smoothstep, {T_EDGE} → {T_FULL} over "
                 f"{W_TAPER:.0f} mm", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)


def removed_volume():
    poly = Polygon(outline_polygon(OUTLINE_B2)).buffer(0)
    poly = poly.difference(Point(LM_X, LM_Y).buffer(LM_D / 2.0))
    poly = poly.intersection(box(-200, -25, 200, 316.0))
    g = 2.5
    xs = np.arange(-155, 155, g)
    ys = np.arange(0, 320, g)
    vol = 0.0
    for x in xs:
        for y in ys:
            if poly.contains(Point(x, y)):
                vol += (T_FULL - thickness_at(x, y)) * g * g
    return vol / 1000.0


if __name__ == "__main__":
    fig = plt.figure(figsize=(15.5, 11.5), dpi=140)
    gs = fig.add_gridspec(4, 2, width_ratios=[330, 310],
                          height_ratios=[1, 1, 1, 1.15], hspace=0.55,
                          wspace=0.14)
    ax_front = fig.add_subplot(gs[:, 0])
    front_view(ax_front)
    section(fig.add_subplot(gs[0, 1]), 300.0,
            "section y=300 (recovery zone): taper closing to the "
            "seam-B land")
    section(fig.add_subplot(gs[1, 1]), LM_Y,
            "section y=201 (LM axis): dotted+✗ = duct no longer fits")
    section(fig.add_subplot(gs[2, 1]), 100.0,
            "section y=100: taper wedge at the lower flanks")
    taper_law(fig.add_subplot(gs[3, 1]))
    vol = removed_volume()
    fig.suptitle("C7 draft — LM section rear taper to a knife edge "
                 f"(removes ≈{vol:.0f} cm³ of solid volume from the "
                 "base pieces)", fontsize=12, y=0.985)
    fig.savefig("baffle_lm_knife_draft.png", bbox_inches="tight")
    print(f"wrote baffle_lm_knife_draft.png (removed volume ~{vol:.0f} cm3)")
