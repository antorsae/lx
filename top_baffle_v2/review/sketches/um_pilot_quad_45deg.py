"""Sketch for (c): 45-degree-shifted UM pilot quad vs the release
geometry. Front view of the UM ring, to scale, from the real route
paths and closure plans."""
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from shapely.geometry import LineString, Point, box as shapely_box

sys.path.insert(0, "src")
import lx521_baffle.obiwan.carriers as ca            # noqa: E402
import lx521_baffle.obiwan.closure_webs as cw        # noqa: E402
import lx521_baffle.obiwan.route as rt               # noqa: E402

UMC = (0.0, 366.081)
PCD_R = 89.5 / 2.0
CUT_R, RECESS_R, CORE_R = 41.0, 49.3, 51.7
OLD = (58.0, 148.0, 238.0, 328.0)
NEW = (13.0, 103.0, 193.0, 283.0)
BORE_R = 4.6 / 2.0
BOSS_R = 8.0 / 2.0     # UM_INSERT_BOSS_D

t_pts = np.asarray(rt.ts_cable_points())
um_pts = np.asarray(rt.route_cable_points())

def seg_dist(pts, p0, p1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    t = np.clip(((pts - p0) @ d) / (d @ d), 0.0, 1.0)
    proj = p0[None, :] + t[:, None] * d[None, :]
    return float(np.min(np.linalg.norm(pts - proj, axis=1)))

fig, ax = plt.subplots(figsize=(10.8, 10.8), facecolor="white")
ax.set_aspect("equal")

INK = "#4a4a4a"
LM_FILL = "#d9d2c7"
UM_FILL = "#cdd6e0"
CABLE = "#d98200"
OK = "#2e7d32"
BAD = "#b00020"
WARN = "#b97400"

# UM ring band + flange recess + cutout
ax.add_patch(Circle(UMC, CORE_R, color=UM_FILL, zorder=1))
ax.add_patch(Circle(UMC, RECESS_R, color="#eceff3", zorder=2))
ax.add_patch(Circle(UMC, CUT_R, color="white", zorder=3))
for r, ls in ((CORE_R, "-"), (RECESS_R, "--"), (CUT_R, "-")):
    ax.add_patch(Circle(UMC, r, fill=False, ec=INK, lw=0.9, ls=ls,
                        zorder=4))
ax.add_patch(Circle(UMC, PCD_R, fill=False, ec="#888", lw=0.7,
                    ls=(0, (5, 3)), zorder=4))
ax.text(-56.5, 398.5, "PCD 89.5", fontsize=8, color="#666")
ax.annotate("", xy=(UMC[0] + PCD_R * math.cos(math.radians(137)),
                    UMC[1] + PCD_R * math.sin(math.radians(137))),
            xytext=(-52.0, 399.5),
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

# LM ring top arc for context
th = np.linspace(math.radians(35), math.radians(145), 200)
ax.plot(113.0 * np.cos(th), 200.981 + 113.0 * np.sin(th),
        color="#8a7a5f", lw=1.0)
ax.fill_between(113.0 * np.cos(th), 200.981 + 110.6 * np.sin(th) - 3.0,
                200.981 + 113.0 * np.sin(th), color=LM_FILL, alpha=0.55,
                lw=0)
ax.text(0, 306.3, "LM carrier crown", fontsize=8.5, ha="center",
        color="#5c5340")

# terminal gap wedge (SEAS: 238..328, axis 283)
ax.add_patch(Wedge(UMC, RECESS_R - 0.5, 238.0, 328.0, width=14.0,
                   color=WARN, alpha=0.14, zorder=3))
axis_pt = (UMC[0] + 38.0 * math.cos(math.radians(283)),
           UMC[1] + 38.0 * math.sin(math.radians(283)))
ax.annotate("SEAS terminal gap 238-328\n(Faston service, axis 283)",
            xy=axis_pt, xytext=(-69.0, 317.5), fontsize=8, color="#7a5b12",
            ha="left",
            arrowprops=dict(arrowstyle="->", color=WARN, lw=1.0),
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))

# free-cable mouth keepout at the T-UM junction
ax.add_patch(Rectangle((-6.0, 412.0), 12.0, 10.0, fill=False, ec="#888",
                       lw=0.8, ls=":"))
ax.text(0, 424.0, "T free-cable mouth |x|<=6", fontsize=7.5,
        ha="center", color="#666")

# cables with lumen bands
for pts, r, col in ((t_pts, 3.0, CABLE), (um_pts, 4.1, "#c9a227")):
    sel = pts[(pts[:, 1] > 300) & (pts[:, 1] < 432)]
    lum = LineString(sel[:, :2]).buffer(r)
    win = shapely_box(-72, 300, 72, 432)
    geom = lum.intersection(win)
    for g in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        if not g.is_empty and g.geom_type == "Polygon":
            ax.fill(*g.exterior.xy, color=col, alpha=0.24, zorder=5)
    ax.plot(sel[:, 0], sel[:, 1], color=col, lw=1.2, zorder=6)
ax.annotate("buried tweeter route\n(O6 lumen; threaded exactly\nbetween the 58/328 pilots)",
            xy=(44.3, 379.5), xytext=(48.5, 352.0), fontsize=8.5,
            color="#a36200",
            arrowprops=dict(arrowstyle="->", color=CABLE, lw=1.1),
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
ax.annotate("UM cable ->283 mouth\n(rear plane z<2.7)",
            xy=(27.0, 313.5), xytext=(45.0, 324.5), fontsize=8,
            color="#8a6d15",
            arrowprops=dict(arrowstyle="->", color="#c9a227", lw=1.0),
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

# magnet sites on the UM ring (wings)
for sx in (-33.5, 33.5):
    ax.add_patch(Circle((sx, 406.7), 2.6, fill=False, ec="#7c4dbe",
                        lw=1.2, zorder=7))
ax.annotate("wing magnet stations", xy=(-33.5, 404.4),
            xytext=(-68.5, 390.5), fontsize=8, color="#5c3d8f",
            arrowprops=dict(arrowstyle="->", color="#7c4dbe", lw=0.9),
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

# existing pilots
for ang in OLD:
    px = UMC[0] + PCD_R * math.cos(math.radians(ang))
    py = UMC[1] + PCD_R * math.sin(math.radians(ang))
    ax.add_patch(Circle((px, py), BOSS_R, fill=False, ec=INK, lw=1.0,
                        zorder=8))
    ax.add_patch(Circle((px, py), BORE_R, color="#9aa4ae", zorder=8))
    ax.text(px, py + 6.4, f"{ang:.0f}°", fontsize=7.5, ha="center",
            color=INK)

# proposed pilots with verdicts
VERDICT = {13.0: (BAD, "13° BLOCKED\nbore crosses the\ntweeter lumen"),
           103.0: (OK, "103° OK"),
           193.0: (OK, "193° OK"),
           283.0: (WARN, "283° CAUTION\nover the Faston\nservice envelope")}

for ang in NEW:
    px = UMC[0] + PCD_R * math.cos(math.radians(ang))
    py = UMC[1] + PCD_R * math.sin(math.radians(ang))
    col, lab = VERDICT[ang]
    ax.add_patch(Circle((px, py), BOSS_R, fill=False, ec=col, lw=1.6,
                        ls=(0, (4, 2)), zorder=9))
    ax.add_patch(Circle((px, py), BORE_R, color=col, alpha=0.75,
                        zorder=9))
    d = seg_dist(t_pts, (px, py, 9.0), (px, py, 14.3))
    off = {13.0: (56.5, 391.0), 103.0: (-51.0, 419.5),
           193.0: (-68.5, 356.0), 283.0: (-69.0, 331.0)}[ang]
    ax.annotate(lab + (f"\nT dist {d:.1f}" if ang in (13.0, 283.0) else ""),
                xy=(px, py), xytext=off, fontsize=8.5, color=col,
                ha="left" if off[0] > 0 else "left",
                arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

# 13-degree conflict zoom ring
p13 = (UMC[0] + PCD_R * math.cos(math.radians(13)),
       UMC[1] + PCD_R * math.sin(math.radians(13)))
ax.add_patch(Circle(p13, 7.5, fill=False, ec=BAD, lw=1.0, ls=":",
                    zorder=9))

ax.text(0, 366.081, "UM\nD82", fontsize=9, ha="center", va="center",
        color="#9aa4ae")
ax.set_xlim(-72, 72)
ax.set_ylim(300, 432)
ax.set_xlabel("x (mm)", fontsize=8)
ax.set_ylabel("y (mm)", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title("(c) 45°-shifted UM pilot quad on PCD 89.5 - existing "
             "(grey) vs proposed (colored by verdict), to scale",
             fontsize=10.5, pad=8)

out = "/Users/antor/.claude/jobs/4808081d/tmp/pilots_45deg_sketch.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
