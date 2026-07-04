"""Render the internal cable routing over the B2 baffle. Faithful: samples
the SAME build123d spline used to cut the ducts (no re-smoothing), plus the
straight plan track of each entry ramp from its rear breakout."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build123d import Spline

from gen_driver_overlay import draw
from top_baffle_nd25fw4 import STAND_FOOT
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2
from top_baffle_nd25fw4_cables import (
    BIG_RAMPS,
    CABLE_D,
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


def duct_xy(name, n=400):
    path = Spline(*route_points(name))
    return np.array([[(path @ (i / n)).X, (path @ (i / n)).Y] for i in range(n + 1)])


def breakout_xy(name):
    """Rear-face (z=0) crossing of the entry ramp -- where the cable
    emerges into the support plate's D20 window."""
    if name in BIG_RAMPS:
        p0, p1 = BIG_RAMPS[name]
    else:
        sign = 1.0 if name == "t1" else -1.0
        p0 = (sign * T_RAMP[0][0], T_RAMP[0][1], T_RAMP[0][2])
        p1 = (sign * T_RAMP[1][0], T_RAMP[1][1], T_RAMP[1][2])
    t = -p0[2] / (p1[2] - p0[2])
    return np.array([p0[0] + t * (p1[0] - p0[0]),
                     p0[1] + t * (p1[1] - p0[1])])


def draw_foot_top_view(ax2):
    """Top view (x-z plan) of the STAND_FOOT foot, x-aligned with the
    front view above: taper to the 38-wide tongue, connector channel,
    NL8MPXX panel, and the four packed duct runs to their step-face
    exits."""
    from top_baffle_nd25fw4_b2_split import (
        CHANNEL_HALF_W, CHANNEL_STEP_Z, FOOT_DEPTH_REAR, NL8_SCREW_PITCH,
        PANEL_T, TONGUE_HALF_W)
    h = TONGUE_HALF_W
    zp = -FOOT_DEPTH_REAR + PANEL_T  # panel inner face
    # plate band (piece_bottom silhouette from above) + corner rib
    ax2.add_patch(plt.Rectangle((-126, 0), 252, 18.3, fc="0.94", ec="0.5",
                                lw=0.8, zorder=0))
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
    ax2.text(0.01, 0.96, "foot - top view", transform=ax2.transAxes,
             fontsize=9, color="0.25", va="top")


def draw_foot_side_view(ax3):
    """Side view (y-z profile, looking along x), z-aligned with the top
    view: plate, foot slab, channel, NL8 body, panel, and the R14 duct
    dives (LM/UM at z=9.15 -> y=10.5, T at z=3.7 -> y=5.5)."""
    from top_baffle_nd25fw4_b2_split import (
        CHANNEL_FLOOR, CHANNEL_STEP_Z, FOOT_DEPTH_REAR,
        FOOT_THICK, NL8_CENTER_Y, NL8_CUTOUT_D, PANEL_H, PANEL_T)
    zp = -FOOT_DEPTH_REAR + PANEL_T
    ax3.add_patch(plt.Rectangle((0, 0), 58, 18.3, fc="0.94", ec="0.5",
                                lw=0.8, zorder=0))
    ax3.add_patch(plt.Rectangle((0, -FOOT_DEPTH_REAR), FOOT_THICK,
                                FOOT_DEPTH_REAR, fc="0.88", ec="0.35",
                                lw=1.0, zorder=1))
    # channel profile (void between floor 4.0 and the open top)
    ax3.add_patch(plt.Rectangle((CHANNEL_FLOOR, zp),
                                FOOT_THICK - CHANNEL_FLOOR,
                                CHANNEL_STEP_Z - zp, fc="white", ec="0.4",
                                ls="--", lw=0.9, zorder=2))
    # panel wall (rises to y=44) + NL8 body profile
    ax3.add_patch(plt.Rectangle((0, -FOOT_DEPTH_REAR), PANEL_H, PANEL_T,
                                fc="0.62", ec="0.3", lw=0.8, zorder=3))
    ax3.add_patch(plt.Rectangle((NL8_CENTER_Y - NL8_CUTOUT_D / 2 + 0.25,
                                 zp), NL8_CUTOUT_D - 0.5, 33, fill=False,
                                ec="0.35", ls=":", lw=1.0, zorder=4))
    for name, (color, _) in STYLE.items():
        x, z_d, y_f, r, dia = FOOT_LANES[name]
        y_c, z_c = y_f + r, z_d - r
        ys, zs = [54.0], [z_d]
        for a in range(0, 91, 6):
            ys.append(y_c - r * np.sin(np.radians(a)))
            zs.append(z_c + r * np.cos(np.radians(a)))
        ys.append(y_f)
        zs.append(CHANNEL_STEP_Z - 4.0)
        ax3.plot(ys, zs, color=color, lw=dia * 0.8, alpha=0.45,
                 solid_capstyle="round", zorder=6)
        ax3.plot(y_f, CHANNEL_STEP_Z - 2.0, marker="o", ms=5, mfc="white",
                 mec=color, zorder=8)
    ax3.text(30, -22, "R14 elbows", fontsize=7, color="0.3")
    ax3.set_aspect("equal")
    ax3.set_xlim(-4, 60)
    ax3.tick_params(labelleft=False)
    ax3.set_xlabel("y (mm)")
    ax3.text(0.04, 0.96, "side view", transform=ax3.transAxes,
             fontsize=9, color="0.25", va="top")


if __name__ == "__main__":
    if STAND_FOOT:
        fig = plt.figure(figsize=(9.6, 15.4), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[640, 195],
                              width_ratios=[360, 80], hspace=0.05,
                              wspace=0.06)
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
        fig.add_subplot(gs[0, 1]).axis("off")
    else:
        fig, ax = plt.subplots(figsize=(8, 12), dpi=150)
        ax2 = ax3 = None
    draw(ax, OUTLINE_B2, TWEETER_DROP_MM, "B2 - internal cable ducts (as modeled)")
    for name, (color, label) in STYLE.items():
        pts = duct_xy(name)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=CABLE_D[name], alpha=0.55,
                solid_capstyle="round", zorder=7, label=label)
        ax.plot(*pts[-1], marker="o", ms=7, mfc="white", mec=color, zorder=8)
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
        ax.annotate("foot lanes -> step-face exits (z=-99),\n"
                    "NL8MPXX panel at foot end (z=-150)", (17, 12), (46, 8),
                    fontsize=8, color="0.3",
                    arrowprops=dict(arrowstyle="-", color="0.5"))
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
    if STAND_FOOT:
        draw_foot_top_view(ax2)
        draw_foot_side_view(ax3)
        ax2.set_xlabel("mm")
    fig.tight_layout()
    fig.savefig("baffle_cable_routing.png")
    print("wrote baffle_cable_routing.png")
