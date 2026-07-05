"""Render the internal cable routing over the B2 baffle as a proper
orthographic view set. Faithful: samples the SAME build123d splines used
to cut the ducts (no re-smoothing) plus the straight ramp axes.

  front view (x-y)   duct mains + breakout/exit markers over the outline
  side view  (y-z)   full-height depth story, sharing y with the front
                     view: duct planes (LM/UM z=9.15, T z=3.7), entry
                     ramps / foot elbows, exit dives, blind pilot-bore
                     bands, the crescent rear taper (clamp-arc section),
                     and the stand foot with the NL8 channel
  top view   (x-z)   STAND_FOOT only, sharing x with the front view:
                     foot taper, packed lanes, channel and panel
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build123d import Spline

from gen_driver_overlay import draw
from top_baffle_nd25fw4 import (
    CRESCENT_SCALLOP_CY,
    L22_CUTOUT,
    L22_PILOT_ANGLES_DEG,
    L22_PILOT_D_MM,
    L22_PILOT_DEPTH_MM,
    L22_PILOT_PCD_MM,
    STAND_FOOT,
    THICKNESS_MM,
    UM_CUTOUT,
    UM_PILOT_ANGLES_DEG,
    UM_PILOT_D_MM,
    UM_PILOT_DEPTH_MM,
    UM_PILOT_PCD_MM,
    _crescent_taper_depth,
)
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2
from top_baffle_nd25fw4_cables import (
    BIG_RAMPS,
    CABLE_D,
    EXIT_RAMPS,
    FOOT_LANES,
    SUPPORT_WINDOW,
    T_RAMP,
    route_points,
)

STYLE = {
    "lm": ("tab:blue", "LM 2x2.5mm2, duct D8.5 (mid-plane)"),
    "um": ("tab:green", "UM twisted pair, duct D8.6 (mid-plane)"),
    "t1": ("gold", "T1 2xAWG24, duct D3.8"),
    "t2": ("tab:red", "T2 2xAWG24, duct D3.8"),
}
TOP_Y = 468.314 - TWEETER_DROP_MM       # B2 top edge (453.46)
TAPER_CY = CRESCENT_SCALLOP_CY - TWEETER_DROP_MM


def duct_xyz(name, n=400):
    path = Spline(*route_points(name))
    pts = [path @ (i / n) for i in range(n + 1)]
    return np.array([[p.X, p.Y, p.Z] for p in pts])


def breakout_xy(name):
    """Rear-face (z=0) crossing of the entry ramp -- where the cable
    emerges into the support plate's D20 window."""
    p0, p1 = _ramp(name)
    t = -p0[2] / (p1[2] - p0[2])
    return np.array([p0[0] + t * (p1[0] - p0[0]),
                     p0[1] + t * (p1[1] - p0[1])])


def _ramp(name):
    if name in BIG_RAMPS:
        return BIG_RAMPS[name]
    sign = 1.0 if name == "t1" else -1.0
    return (tuple(sign * v if i == 0 else v for i, v in enumerate(p))
            for p in T_RAMP)


def _foot_lane_yz(name):
    """(y, z) polyline of the foot elbow + run, as swept by the cutter."""
    x, z_d, y_f, r, dia = FOOT_LANES[name]
    y_c, z_c = y_f + r, z_d - r
    pts = [(y_c + 7.5, z_d), (y_c + 4.0, z_d)]
    for a in range(0, 91, 6):
        pts.append((y_c - r * math.sin(math.radians(a)),
                    z_c + r * math.cos(math.radians(a))))
    pts += [(y_f, z_c - 10.0), (y_f, -103.0)]
    return np.array(pts)


def _pilot_band_ys(center_y, pcd, angles):
    return sorted({round(center_y + pcd / 2.0 * math.sin(math.radians(a)), 2)
                   for a in angles})


def draw_side_view(ax):
    """Full-height y-z profile: plate slab, crescent rear taper, blind
    pilot bands, duct depth lines, entries and exit dives, stand foot."""
    # plate slab (front face at z=18.3, rear at z=0)
    ax.add_patch(plt.Rectangle((0, 0), THICKNESS_MM, TOP_Y, fc="0.90",
                               ec="0.35", lw=1.0, zorder=1))
    # crescent rear taper: section along the clamp arc (r=44 about the
    # scallop center) -- rear surface recedes to the 4.0 clamp seat and
    # the ~0.4 horn feather
    r_sec = 44.0
    wedge = [(0.0, TAPER_CY - r_sec)]
    th = -90.0
    while th < -10.0:
        y = TAPER_CY + r_sec * math.sin(math.radians(th))
        if y > TOP_Y:
            break
        wedge.append((_crescent_taper_depth(th), y))
        th += 1.5
    wedge.append((wedge[-1][0], TOP_Y))
    wedge.append((0.0, TOP_Y))
    ax.add_patch(plt.Polygon(wedge, closed=True, fc="white", ec="0.4",
                             ls="--", lw=0.9, zorder=2))
    if STAND_FOOT:  # the no-foot panel is too narrow for text
        ax.annotate("crescent rear taper\n(4.0 seat at the clamp ring)",
                    (14.3, 436.4), (-148, 470), fontsize=8, color="0.3",
                    arrowprops=dict(arrowstyle="-", color="0.5"))
    # blind pilot bores, front face only (depth 11)
    for cy, pcd, angles, dia, depth in (
            (L22_CUTOUT[1], L22_PILOT_PCD_MM, L22_PILOT_ANGLES_DEG,
             L22_PILOT_D_MM, L22_PILOT_DEPTH_MM),
            (UM_CUTOUT[1], UM_PILOT_PCD_MM, UM_PILOT_ANGLES_DEG,
             UM_PILOT_D_MM, UM_PILOT_DEPTH_MM)):
        for y in _pilot_band_ys(cy, pcd, angles):
            ax.add_patch(plt.Rectangle((THICKNESS_MM - depth, y - dia / 2),
                                       depth, dia, fc="white", ec="0.45",
                                       ls=":", lw=0.8, zorder=3))
    if STAND_FOOT:
        ax.annotate("blind pilot bores x11 deep\n(T ducts pass under the\n"
                    "10F ring: 1.7 floor)", (7.3, 334.4), (-148, 320),
                    fontsize=8, color="0.3",
                    arrowprops=dict(arrowstyle="-", color="0.5"))
    # duct mains (true z from the swept splines)
    for name, (color, _) in STYLE.items():
        pts = duct_xyz(name)
        ax.plot(pts[:, 2], pts[:, 1], color=color, lw=CABLE_D[name] * 0.8,
                alpha=0.55, solid_capstyle="round", zorder=6)
    # exit dives into the driver cutouts (both states)
    for name, (p0, p1, _dia) in EXIT_RAMPS.items():
        color = STYLE[name][0]
        ax.plot([p0[2], p1[2]], [p0[1], p1[1]], color=color, ls=":",
                lw=CABLE_D[name] * 0.6, alpha=0.8, zorder=7)
        ax.plot(p1[2], p1[1], marker="o", ms=6, mfc="white", mec=color,
                zorder=8)
    ax.plot(3.7, 433.5, marker="o", ms=5, mfc="white", mec=STYLE["t1"][0],
            zorder=8)  # T scallop-rim exits (head-on at duct depth)
    if STAND_FOOT:
        # foot slab, NL8 panel, connector channel, lanes
        ax.add_patch(plt.Rectangle((-150, 0), 150, 18.3, fc="0.90",
                                   ec="0.35", lw=1.0, zorder=1))
        ax.add_patch(plt.Rectangle((-150, 0), 4, 44, fc="0.62", ec="0.3",
                                   lw=0.8, zorder=3))
        ax.add_patch(plt.Rectangle((-146, 4), 47, 14.3, fc="white",
                                   ec="0.4", ls="--", lw=0.9, zorder=2))
        ax.add_patch(plt.Rectangle((-146, 5.25), 33, 30.5, fill=False,
                                   ec="0.35", ls=":", lw=1.0, zorder=4))
        ax.plot([-99, -99], [4, 18.3], color="0.4", lw=1.2, zorder=3)
        for name, (color, _) in STYLE.items():
            lane = _foot_lane_yz(name)
            ax.plot(lane[:, 1], lane[:, 0], color=color,
                    lw=FOOT_LANES[name][4] * 0.8, alpha=0.45,
                    solid_capstyle="round", zorder=6)
            ax.plot(-101, FOOT_LANES[name][2], marker="o", ms=5,
                    mfc="white", mec=color, zorder=8)
        ax.annotate("R14 elbows", (-5, 22), (-70, 40), fontsize=8,
                    color="0.3", arrowprops=dict(arrowstyle="-",
                                                 color="0.5"))
        ax.annotate("step-face outs (z=-99)\nNL8 panel at z=-150",
                    (-99, 30), (-145, 60), fontsize=8, color="0.3",
                    arrowprops=dict(arrowstyle="-", color="0.5"))
    else:
        # entry ramps through the support window (rear face z=0)
        for name in STYLE:
            p0, p1 = _ramp(name)
            color = STYLE[name][0]
            ax.plot([p0[2], p1[2]], [p0[1], p1[1]], color=color, ls=":",
                    lw=CABLE_D[name] * 0.6, alpha=0.8, zorder=7)
            bo = breakout_xy(name)
            ax.plot(0, bo[1], marker="s", ms=5, mfc=color, mec="0.2",
                    zorder=8)
    ax.axvline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_aspect("equal")
    ax.tick_params(labelleft=False)
    ax.set_xlabel("z (mm)")
    label = ("side view (y-z)\nfront face at z=18.3" if STAND_FOOT
             else "side\nview")
    ax.text(0.04, 0.99, label, transform=ax.transAxes, fontsize=9,
            color="0.25", va="top")
    ax.grid(True, lw=0.3, alpha=0.4)


def draw_foot_top_view(ax2):
    """Top view (x-z plan) of the STAND_FOOT foot, x-aligned with the
    front view above: taper to the 38-wide tongue, connector channel,
    NL8MPXX panel, and the four packed duct runs to their step-face
    exits."""
    from top_baffle_nd25fw4_b2_split import (
        CHANNEL_HALF_W, CHANNEL_STEP_Z, FLANK_SLOPE, FOOT_DEPTH_REAR,
        NL8_SCREW_PITCH, PANEL_T, TONGUE_HALF_W)
    h = TONGUE_HALF_W
    zp = -FOOT_DEPTH_REAR + PANEL_T  # panel inner face
    # plate band: piece_bottom's widest plan extent (at its y=170 tabs)
    w_plate = 76.2 + FLANK_SLOPE * 170.0
    ax2.add_patch(plt.Rectangle((-w_plate, 0), 2 * w_plate, 18.3,
                                fc="0.94", ec="0.5", lw=0.8, zorder=0))
    # foot plan: one continuous taper, strip corners -> 38-wide panel
    foot_xy = [(-81.64, 0), (81.64, 0), (h, zp),
               (h, -FOOT_DEPTH_REAR), (-h, -FOOT_DEPTH_REAR), (-h, zp)]
    ax2.add_patch(plt.Polygon(foot_xy, closed=True, fc="0.88", ec="0.35",
                              lw=1.2, zorder=1))
    # connector channel (floor 4.0 between 2.0 rails) + step face
    ax2.add_patch(plt.Rectangle((-CHANNEL_HALF_W, zp), 2 * CHANNEL_HALF_W,
                                CHANNEL_STEP_Z - zp, fc="white", ec="0.4",
                                ls="--", lw=0.9, zorder=2))
    ax2.plot([-CHANNEL_HALF_W, CHANNEL_HALF_W],
             [CHANNEL_STEP_Z, CHANNEL_STEP_Z], color="0.4", lw=1.2, zorder=3)
    # panel wall + NL8 body footprint + screw pass-throughs
    ax2.add_patch(plt.Rectangle((-h, -FOOT_DEPTH_REAR), 2 * h, PANEL_T,
                                fc="0.62", ec="0.3", lw=0.8, zorder=3))
    ax2.add_patch(plt.Rectangle((-15.25, zp), 30.5, 33, fill=False,
                                ec="0.35", ls=":", lw=1.0, zorder=4))
    for sx in (-NL8_SCREW_PITCH / 2, NL8_SCREW_PITCH / 2):
        ax2.plot(sx, -FOOT_DEPTH_REAR + PANEL_T / 2, marker="o", ms=3.5,
                 mfc="none", mec="0.2", zorder=5)
    # duct runs: dotted through the elbow dive, solid along the foot,
    # open ends 4 mm past the step face
    for name, (color, _) in STYLE.items():
        x, z_d, y_f, r, dia = FOOT_LANES[name]
        ax2.plot([x, x], [z_d, z_d - r], color=color, ls=":",
                 lw=dia * 0.6, alpha=0.55, zorder=6)
        ax2.plot([x, x], [z_d - r, CHANNEL_STEP_Z - 4.0], color=color,
                 lw=dia, alpha=0.55, solid_capstyle="round", zorder=6)
        ax2.plot(x, CHANNEL_STEP_Z - 2.0, marker="o", ms=6, mfc="white",
                 mec=color, zorder=8)
    ax2.annotate("step-face cable outs (z=-99)", (CHANNEL_HALF_W, -101),
                 (40, -95), fontsize=8, color="0.3",
                 arrowprops=dict(arrowstyle="-", color="0.5"))
    ax2.annotate("NL8MPXX body (D30.5 x 33)", (15.25, -130), (40, -132),
                 fontsize=8, color="0.3",
                 arrowprops=dict(arrowstyle="-", color="0.5"))
    ax2.annotate("panel 38 x 44 x 4\nD31 cutout + 4 x D3.2", (-h, -148),
                 (-125, -140), fontsize=8, color="0.3",
                 arrowprops=dict(arrowstyle="-", color="0.5"))
    ax2.set_aspect("equal")
    ax2.set_ylim(-160, 24)
    ax2.set_ylabel("z (mm)")
    ax2.text(0.02, 0.04, "foot - top view (x-z)", transform=ax2.transAxes,
             fontsize=9, color="0.25", va="bottom")
    ax2.grid(True, lw=0.3, alpha=0.4)


if __name__ == "__main__":
    if STAND_FOOT:
        fig = plt.figure(figsize=(10.6, 15.6), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[640, 190],
                              width_ratios=[340, 195], hspace=0.04,
                              wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_side = fig.add_subplot(gs[0, 1], sharey=ax)
        ax_top = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_leg = fig.add_subplot(gs[1, 1])
        ax_leg.axis("off")
    else:
        fig = plt.figure(figsize=(9.8, 12.6), dpi=150)
        gs = fig.add_gridspec(1, 2, width_ratios=[340, 55], wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_side = fig.add_subplot(gs[0, 1], sharey=ax)
        ax_top = ax_leg = None
    draw(ax, OUTLINE_B2, TWEETER_DROP_MM, "B2 - internal cable ducts (as modeled)")
    for name, (color, label) in STYLE.items():
        pts = duct_xyz(name)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=CABLE_D[name], alpha=0.55,
                solid_capstyle="round", zorder=7, label=label)
        ax.plot(*pts[-1][:2], marker="o", ms=7, mfc="white", mec=color, zorder=8)
        if STAND_FOOT:
            # elbow dive + run rearward through the foot (into the page),
            # exiting the channel step face just short of the NL8 panel
            x, _, y_f, _, _ = FOOT_LANES[name]
            ax.plot([pts[0][0], x], [pts[0][1], y_f], color=color, ls=":",
                    lw=CABLE_D[name] * 0.6, alpha=0.55, zorder=7)
            ax.plot(x, y_f, marker="v", ms=6, mfc=color, mec="0.2", zorder=8)
        else:
            bo = breakout_xy(name)
            ax.plot([bo[0], pts[0][0]], [bo[1], pts[0][1]], color=color,
                    lw=CABLE_D[name], alpha=0.55, solid_capstyle="round",
                    zorder=7)
            ax.plot(*bo, marker="s", ms=6, mfc=color, mec="0.2", zorder=8)
    if STAND_FOOT:
        # NL8MPXX panel at the foot's far end (z=-150), front projection
        ax.add_patch(plt.Rectangle((-19, 0), 38, 44, fill=False, ec="0.4",
                                   ls="--", lw=0.9, zorder=6))
        ax.add_patch(plt.Circle((0, 20.5), 15.5, fill=False, ec="0.4",
                                ls="--", lw=0.9, zorder=6))
        for sx in (-14.6, 14.6):
            for sy in (5.9, 35.1):
                ax.plot(sx, sy, marker="o", ms=3, mfc="none", mec="0.4",
                        zorder=6)
        ax.annotate("cables dive into the packed foot lanes\n"
                    "(see side and top views)", (17, 12), (46, 8),
                    fontsize=8, color="0.3",
                    arrowprops=dict(arrowstyle="-", color="0.5"))
        ax_leg.legend(*ax.get_legend_handles_labels(), loc="center",
                      fontsize=9, framealpha=0.9)
    else:
        # D20 window in the support plate: all four cables pass here
        wx, wy, wd = SUPPORT_WINDOW
        ax.add_patch(plt.Circle((wx, wy), wd / 2, fill=False, ec="0.25",
                                ls=":", lw=1.4, zorder=9))
        ax.annotate("support plate D20 window\n(top edge tangent to the\n"
                    "upper screw line y=70)", (wx + wd / 2, wy), (46, 40),
                    fontsize=8, color="0.3",
                    arrowprops=dict(arrowstyle="-", color="0.5"))
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    draw_side_view(ax_side)
    ax_side.set_xlim((-160, 26) if STAND_FOOT else (-14, 26))
    if STAND_FOOT:
        draw_foot_top_view(ax_top)
        ax_top.set_xlabel("mm")
    fig.tight_layout()
    fig.savefig("baffle_cable_routing.png")
    print("wrote baffle_cable_routing.png")
