"""Render front-view PNGs of the baffle variants with all drivers overlaid
as dashed outer silhouettes:

  * SEAS W22EX001 lower mid, D221 flange, aligned by its 6 M5 pilots
  * Scan-Speak 10F/8424G00 upper mid, D97.5 flange, aligned by its 4 M3 pilots
  * ND25FW-4 tweeter pair, D104 faceplates, on the clamp-hole-derived axes
    (front tweeter = lower, on the front face; rear tweeter on standoffs)

Run:  python gen_driver_overlay.py
  ->  baffle_variants_drivers.png (A | B1 | B2)
      baffle_b1_drivers.png, baffle_b2_drivers.png (singles)
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from top_baffle_nd25fw4 import (
    BRIDGE_CSK_D_MM,
    BRIDGE_HOLE_D_MM,
    BRIDGE_HOLE_XY,
    CORNER_HOLE_D_MM,
    CORNER_HOLE_XY,
    CORNER_HOLES_ENABLED,
    STAND_FOOT,
    L22_CUTOUT,
    L22_PILOT_ANGLES_DEG,
    L22_PILOT_D_MM,
    L22_PILOT_PCD_MM,
    TWEETER_HOLE_D_MM,
    TWEETER_HOLE_XY,
    UM_CUTOUT,
    UM_PILOT_ANGLES_DEG,
    UM_PILOT_D_MM,
    UM_PILOT_PCD_MM,
    _pilot_centers,
)
from top_baffle_nd25fw4_a_comp import OUTLINE_A_COMP
from top_baffle_nd25fw4_b import MAGNET_D_MM, MAGNET_SITES, TWEETER_DROP_MM
from top_baffle_nd25fw4_b1 import OUTLINE_B1
from top_baffle_nd25fw4_b2 import OUTLINE_B2

W22_FLANGE_D_MM = 221.0   # from E0022_W22EX001.stp
TENF_FLANGE_D_MM = 97.5   # from 10f-8424g00.pdf
ND25_FACE_D_MM = 104.0    # from nd25fw-4-spec-sheet.pdf
# Tweeter axes per the V2 drawing (drop = 0): the lower tweeter faces
# FORWARD (faceplate on the front face, bolted through the clamp holes);
# the upper one faces rearward on standoffs.
T_FRONT_AXIS_Y = 483.78
T_REAR_AXIS_Y = 549.05

# Stock LX521.4 top baffle from "lx521 baffle metric.dxf" (centerline at
# x=152.4 in the DXF), aligned to the variants by the LM driver center:
# stock LM (0, 203.2) -> W22 pilots' center (0, 200.981), i.e. y - 2.219.
STOCK_DY = 200.981 - 203.2
STOCK_OUTLINE = [
    (-76.2, 0.0), (-152.4, 254.0), (-57.15, 304.8), (-57.15, 406.4),
    (-76.2, 558.8), (76.2, 558.8), (57.15, 406.4), (57.15, 304.8),
    (152.4, 254.0), (76.2, 0.0),
]
STOCK_HOLES = [  # (x, y, dia)
    (0.0, 203.2, 190.0),
    (0.0, 368.3, 82.0),
    (0.0, 450.85, 47.0),
    (0.0, 508.0, 47.0),
]

YMAX = T_REAR_AXIS_Y + ND25_FACE_D_MM / 2 + 15


def _arc_points(p1, p2, p3, samples=32):
    """Sample a circular arc through three points, from p1 to p3 via p2."""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    cx = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    cy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    r = np.hypot(x1 - cx, y1 - cy)
    a1, a2, a3 = (np.arctan2(y - cy, x - cx) for x, y in (p1, p2, p3))
    # sweep from a1 to a3 passing through a2
    ccw = (a2 - a1) % (2 * np.pi) < (a3 - a1) % (2 * np.pi)
    sweep = (a3 - a1) % (2 * np.pi) if ccw else -((a1 - a3) % (2 * np.pi))
    th = a1 + sweep * np.linspace(0.0, 1.0, samples, endpoint=False)
    return [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in th]


def outline_polygon(outline, samples=32):
    pts = []
    for seg in outline:
        if seg[0] == "L":
            pts.append(seg[1])
        elif seg[0] == "A":
            pts.extend(_arc_points(*seg[1:], samples=samples))
        else:
            p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in seg[1:])
            t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
            b = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
            pts.extend(map(tuple, b))
    pts.append(pts[0])
    return np.asarray(pts)


def draw(ax, outline, drop, title, labels=True, joint_outline=None):
    poly = outline_polygon(outline)
    ax.fill(poly[:, 0], poly[:, 1], color="0.85", zorder=1)
    ax.plot(poly[:, 0], poly[:, 1], color="0.25", lw=1.2, zorder=3)

    def circle(cx, cy, dia, **kw):
        th = np.linspace(0, 2 * np.pi, 181)
        ax.plot(cx + dia / 2 * np.cos(th), cy + dia / 2 * np.sin(th), **kw)

    for cx, cy, dia in (L22_CUTOUT, UM_CUTOUT):
        th = np.linspace(0, 2 * np.pi, 181)
        ax.fill(cx + dia / 2 * np.cos(th), cy + dia / 2 * np.sin(th), color="white", zorder=2)
        circle(cx, cy, dia, color="0.25", lw=1.0, zorder=3)
    if not STAND_FOOT:  # bridge screws (absent with the fused foot)
        for cx, cy in BRIDGE_HOLE_XY:
            circle(cx, cy, BRIDGE_HOLE_D_MM, color="0.25", lw=0.8, zorder=3)
        for cx, cy in BRIDGE_HOLE_XY:  # front-face countersinks (D10.4)
            circle(cx, cy, BRIDGE_CSK_D_MM, color="0.45", lw=0.6, zorder=3)
    if CORNER_HOLES_ENABLED:  # M5 thread-forming corner holes (D4.5)
        for cx, cy in CORNER_HOLE_XY:
            circle(cx, cy, CORNER_HOLE_D_MM, color="0.25", lw=0.8, zorder=3)
    for cx, cy in TWEETER_HOLE_XY:
        circle(cx, cy - drop, TWEETER_HOLE_D_MM, color="0.25", lw=0.8, zorder=3)
    for cx, cy in _pilot_centers(UM_CUTOUT[:2], UM_PILOT_PCD_MM, UM_PILOT_ANGLES_DEG):
        circle(cx, cy, UM_PILOT_D_MM, color="0.35", lw=0.8, zorder=3)
    for cx, cy in _pilot_centers(L22_CUTOUT[:2], L22_PILOT_PCD_MM, L22_PILOT_ANGLES_DEG):
        circle(cx, cy, L22_PILOT_D_MM, color="0.35", lw=0.8, zorder=3)

    # B2 base-piece boundary inside composite variants (attachment joint)
    if joint_outline is not None:
        jp = outline_polygon(joint_outline)
        ax.plot(jp[:, 0], jp[:, 1], color="0.55", lw=0.9, ls=(0, (4, 2, 1, 2)), zorder=5)

    # D5 x 2 magnets seen edge-on: discs sit in the flank walls with their
    # axes IN-PLANE (normal to the wall). Solid bar = base magnet in
    # piece_top_b2 (pin sites protrude 1 mm); lighter bar = the mating
    # attachment magnet (drawn only for composite variants).
    def mag_bar(px, py, nx, ny, a, b, alpha):
        tx, ty = -ny, nx  # wall tangent
        r = MAGNET_D_MM / 2.0
        xs = [px + tx * r + nx * a, px - tx * r + nx * a,
              px - tx * r + nx * b, px + tx * r + nx * b]
        ys = [py + ty * r + ny * a, py - ty * r + ny * a,
              py - ty * r + ny * b, py + ty * r + ny * b]
        ax.fill(xs, ys, color="tab:orange", alpha=alpha, lw=0.6,
                edgecolor="tab:orange", zorder=6)

    for mx, my, mnx, mny, pin, _zc in MAGNET_SITES:
        for sx in (1, -1):
            px, py, nx, ny = sx * mx, my, sx * mnx, mny
            if pin:
                mag_bar(px, py, nx, ny, -2.0, 1.0, 0.9)     # base, 1 mm proud
                if joint_outline is not None:
                    mag_bar(px, py, nx, ny, 1.2, 3.2, 0.45)  # attachment side
            else:
                mag_bar(px, py, nx, ny, -3.1, -0.1, 0.9)
                if joint_outline is not None:
                    mag_bar(px, py, nx, ny, 0.1, 3.1, 0.45)

    # stock LX521.4 baffle, dotted gray, LM-aligned
    sx = [p[0] for p in STOCK_OUTLINE] + [STOCK_OUTLINE[0][0]]
    sy = [p[1] + STOCK_DY for p in STOCK_OUTLINE] + [STOCK_OUTLINE[0][1] + STOCK_DY]
    ax.plot(sx, sy, color="0.45", lw=1.1, ls=(0, (1, 2)), zorder=5)
    for hx, hy, hdia in STOCK_HOLES:
        circle(hx, hy + STOCK_DY, hdia, color="0.45", lw=1.0, ls=(0, (1, 2)), zorder=5)

    circle(*L22_CUTOUT[:2], W22_FLANGE_D_MM, color="tab:blue", lw=1.5, ls=(0, (6, 4)), zorder=4)
    circle(*UM_CUTOUT[:2], TENF_FLANGE_D_MM, color="tab:red", lw=1.5, ls=(0, (6, 4)), zorder=4)
    circle(0, T_FRONT_AXIS_Y - drop, ND25_FACE_D_MM, color="tab:green", lw=1.5, ls=(0, (6, 4)), zorder=4)
    circle(0, T_REAR_AXIS_Y - drop, ND25_FACE_D_MM, color="tab:purple", lw=1.3, ls=(0, (2, 3)), zorder=4)

    if labels:
        ax.text(0, L22_CUTOUT[1] - W22_FLANGE_D_MM / 2 - 7, "W22EX001\nD221",
                color="tab:blue", ha="center", va="top", fontsize=8)
        ax.text(-115, UM_CUTOUT[1], "10F/8424G00\nD97.5",
                color="tab:red", ha="center", va="center", fontsize=8)
        ax.text(115, T_FRONT_AXIS_Y - drop, "ND25FW-4\nfront D104",
                color="tab:green", ha="center", va="center", fontsize=8)
        ax.text(-115, T_REAR_AXIS_Y - drop, "ND25FW-4\nrear D104",
                color="tab:purple", ha="center", va="center", fontsize=8)
        ax.text(112, 545, "stock\nLX521.4", color="0.45",
                ha="center", va="center", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlim(-170, 170)
    ax.set_ylim(-15, YMAX)
    ax.set_title(title, fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.4)


VARIANTS = [
    (OUTLINE_A_COMP, "A-comp (B2 pieces + shoulders)", OUTLINE_B2),
    (OUTLINE_B1, "B1 (B2 pieces + wings)", OUTLINE_B2),
    (OUTLINE_B2, "B2 (base pieces)", None),
]

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(16, 9.5), dpi=150)
    for ax, (outline, title, joint) in zip(axes, VARIANTS):
        draw(ax, outline, TWEETER_DROP_MM, title, labels=(ax is axes[0]), joint_outline=joint)
    fig.suptitle("LX521.4 top baffle variants - dashed driver silhouettes", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig("baffle_variants_drivers.png")
    plt.close(fig)
    print("wrote baffle_variants_drivers.png")

    for outline, title, name, joint in [
        (OUTLINE_B1, "Variant B1 - driver overlays", "baffle_b1_drivers.png", OUTLINE_B2),
        (OUTLINE_B2, "Variant B2 - driver overlays", "baffle_b2_drivers.png", None),
    ]:
        fig, ax = plt.subplots(figsize=(7, 11), dpi=150)
        draw(ax, outline, TWEETER_DROP_MM, title, joint_outline=joint)
        ax.set_xlabel("mm"); ax.set_ylabel("mm")
        fig.tight_layout()
        fig.savefig(name)
        plt.close(fig)
        print("wrote", name)
