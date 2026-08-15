"""One-off sketch of option (A): single vertical M2x8 at x=-20 in the
Obi-Wan LM-UM junction. Drawn from the real plan geometry."""
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from shapely.geometry import LineString, Point, box as shapely_box

sys.path.insert(0, "src")
import lx521_baffle.obiwan.carriers as ca            # noqa: E402
import lx521_baffle.obiwan.closure_webs as cw        # noqa: E402
import lx521_baffle.obiwan.route as rt               # noqa: E402

LM_C = (0.0, 200.981)
UM_C = (0.0, 366.081)
LM_CORE_R, LM_RECESS_R = 113.0, 110.6
UM_CORE_R, UM_RECESS_R = 51.7, 49.3
X0 = -20.0

polys = cw.lm_um_closure_polygons()
lm_plan, um_plan = polys["lm"], polys["um"]
backfill = cw._lm_um_rear_recess_backfill_plan()

# exact values on the x=-20 line
line = LineString([(X0, 300.0), (X0, 326.0)])
lm_hit = line.intersection(lm_plan)
um_hit = line.intersection(um_plan)
cres_hit = line.intersection(backfill)
LM_WEB_TOP = lm_hit.bounds[3]          # 315.28
UM_WEB_BOT = um_hit.bounds[1]          # 315.33
UM_WEB_TOP = um_hit.bounds[3]          # 318.90
CRES_LO, CRES_HI = cres_hit.bounds[1], cres_hit.bounds[3]
ARC_R1106 = LM_C[1] + math.sqrt(LM_RECESS_R**2 - X0*X0)   # 309.758
ARC_R113 = LM_C[1] + math.sqrt(LM_CORE_R**2 - X0*X0)      # 312.20
UM_LIP_LO = UM_C[1] - math.sqrt(UM_CORE_R**2 - X0*X0)     # 318.41
UM_LIP_HI = UM_C[1] - math.sqrt(UM_RECESS_R**2 - X0*X0)   # 321.02

# z structure
Z_REAR, Z_FRONT = 6.8, 18.3
LM_MEMB = (11.45, 12.3)     # LM_SEAT_Z - 0.85 .. LM_SEAT_Z
UM_MEMB = (13.45, 14.3)
Z_AXIS = 9.25  # counterbore O4.4 top lands exactly on the membrane underside

# screw stack
TIP_Y = UM_WEB_BOT + 0.10 + 2.5        # insert seated 0.10 in, full engage
SEAT_Y = TIP_Y - 8.0                   # M2x8 under-head
POCKET_TOP = UM_WEB_BOT + 2.9

t_pts = np.asarray(rt.ts_cable_points())
um_pts = np.asarray(rt.route_cable_points())

fig = plt.figure(figsize=(11.5, 12.6), facecolor="white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.16)

LM_FILL = "#d9d2c7"
UM_FILL = "#cdd6e0"
EDGE = "#4a4a4a"
INSERT_RED = "#c73a2f"
SCREW_GREEN = "#2e7d32"
CABLE = "#d98200"

# ---------------------------------------------------------------- panel A
axA = fig.add_subplot(gs[0])
axA.set_aspect("equal")
win = shapely_box(-48, 297, 48, 331)

def draw_poly(ax, geom, **kw):
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    for g in geoms:
        if g.is_empty or g.geom_type != "Polygon":
            continue
        ax.fill(*g.exterior.xy, **kw)

def arc(ax, c, r, **kw):
    th = np.linspace(0, 2*math.pi, 720)
    ax.plot(c[0] + r*np.cos(th), c[1] + r*np.sin(th), **kw)

# ring bands (annulus between recess and core radii)
lm_band = Point(LM_C).buffer(LM_CORE_R, resolution=256).difference(
    Point(LM_C).buffer(LM_RECESS_R, resolution=256))
um_band = Point(UM_C).buffer(UM_CORE_R, resolution=256).difference(
    Point(UM_C).buffer(UM_RECESS_R, resolution=256))
draw_poly(axA, lm_band.intersection(win), color=LM_FILL, zorder=1)
draw_poly(axA, um_band.intersection(win), color=UM_FILL, zorder=1)
draw_poly(axA, lm_plan.intersection(win), color=LM_FILL, zorder=2)
draw_poly(axA, um_plan.intersection(win), color=UM_FILL, zorder=2)
draw_poly(axA, backfill.intersection(win), color="#c4b8a4", zorder=3)

for r, c, ls in ((LM_CORE_R, LM_C, "-"), (LM_RECESS_R, LM_C, "--"),
                 (UM_CORE_R, UM_C, "-"), (UM_RECESS_R, UM_C, "--")):
    arc(axA, c, r, color=EDGE, lw=0.8, ls=ls, zorder=4)

# ears
for ex in (-32.0, 32.0):
    axA.add_patch(Circle((ex, 315.770102), 4.9, fill=False,
                         ec=EDGE, lw=1.0, zorder=5))
    axA.add_patch(Circle((ex, 315.770102), 1.7, fill=False,
                         ec=EDGE, lw=0.8, zorder=5))

# cables: centerline + lumen band
for pts, r in ((t_pts, 3.0), (um_pts, 4.1)):
    sel = pts[(pts[:, 1] > 292) & (pts[:, 1] < 331)]
    lum = LineString(sel[:, :2]).buffer(r)
    draw_poly(axA, lum.intersection(win), color=CABLE, alpha=0.28, zorder=6)
    axA.plot(sel[:, 0], sel[:, 1], color=CABLE, lw=1.2, zorder=7)
axA.text(-30.0, 298.2, "tweeter cable", fontsize=8, color="#a36200",
         bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
axA.text(36.2, 323.4, "UM cable\n(rear, z<2.7)", fontsize=7.5,
         color="#a36200", ha="left",
         bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
axA.annotate("right side blocked:\ntweeter route crosses seam\n(z 8.2-10.0, all states)",
             xy=(14.5, 317.2), xytext=(28.5, 300.4), fontsize=8.5,
             color="#b00020", ha="left",
             bbox=dict(fc="white", ec="none", alpha=0.75, pad=1),
             arrowprops=dict(arrowstyle="->", color="#b00020", lw=1.1))

# the M2 feature silhouettes at x=-20 (front view: Y-axis bore = strip)
axA.add_patch(Rectangle((X0-2.2, SEAT_Y-2.0), 4.4, 2.0,
              color=SCREW_GREEN, alpha=0.45, zorder=9))     # head zone
axA.add_patch(Rectangle((X0-1.2, SEAT_Y), 2.4, LM_WEB_TOP-SEAT_Y,
              color=SCREW_GREEN, alpha=0.85, zorder=9))     # clearance bore
axA.add_patch(Rectangle((X0-1.6, UM_WEB_BOT), 3.2, 2.9,
              color=INSERT_RED, alpha=0.9, zorder=9))       # insert pocket
axA.annotate("M2x8 at x=-20\n(bore 2.4 green /\ninsert 3.2 red)",
             xy=(X0, 312.0), xytext=(-45.5, 302.2), fontsize=8.5,
             ha="left",
             arrowprops=dict(arrowstyle="->", color="#222", lw=1.0))
axA.annotate("M3 half-lap ear (existing)", xy=(32, 320.6), xytext=(20, 327.8),
             fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="#222", lw=1.0))
axA.annotate("rear backfill crescent", xy=(-24.5, 309.2), xytext=(-46, 325.5),
             fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="#222", lw=1.0))
axA.text(0, 329.3, "UM carrier", ha="center", fontsize=10, color="#33475c")
axA.text(0, 298.6, "LM carrier", ha="center", fontsize=10, color="#5c5340")

axA.set_xlim(-48, 48)
axA.set_ylim(297, 331)
axA.set_xlabel("x (mm)", fontsize=8)
axA.set_ylabel("y (mm)", fontsize=8)
axA.tick_params(labelsize=7)
axA.set_title("A - front view of the LM-UM junction crown (to scale)",
              fontsize=10.5, pad=6)

# ---------------------------------------------------------------- panel B
axB = fig.add_subplot(gs[1])
axB.set_aspect("equal")

def rect(ax, y0, y1, z0, z1, **kw):
    ax.add_patch(Rectangle((z0, y0), z1-z0, y1-y0, **kw))

GAP_DRAW = 0.05
# LM material
rect(axB, CRES_LO, ARC_R1106, Z_REAR, LM_MEMB[0], color="#c4b8a4",
     ec=EDGE, lw=0.5)                                  # crescent (rear band)
rect(axB, ARC_R1106, LM_WEB_TOP, Z_REAR, Z_FRONT, color=LM_FILL,
     ec=EDGE, lw=0.5)                                  # lip + LM web
rect(axB, 296.0, ARC_R1106, LM_MEMB[0], LM_MEMB[1], color=LM_FILL,
     ec=EDGE, lw=0.4)                                  # seat membrane
# UM material
rect(axB, UM_WEB_BOT, UM_LIP_HI, Z_REAR, Z_FRONT, color=UM_FILL,
     ec=EDGE, lw=0.5)                                  # UM web + lip
rect(axB, UM_LIP_HI, 323.6, UM_MEMB[0], UM_MEMB[1], color=UM_FILL,
     ec=EDGE, lw=0.4)
axB.text(14.9, 322.6, "UM seat\nmembrane", fontsize=6.8, ha="left",
         color="#33475c")

# voids labels
axB.text(9.0, 306.4, "LM rear-open void\n(flange band, tool access)",
         fontsize=8, ha="center", color="#6b5f4d",
         bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
axB.text(15.3, 304.6, "LM flange recess\n(driver flange)", fontsize=8,
         ha="center", color="#6b5f4d",
         bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
axB.text(10.0, 322.6, "UM rear void", fontsize=8, ha="center",
         color="#33475c")
axB.text(2.9, 313.0, "open air\nbehind baffle", fontsize=8, ha="center",
         color="#888")

# T cable section at x=-20 (floor state): centerline ~ (y 299.4, z 8.45)
tc = t_pts[np.argmin(np.abs(t_pts[:, 0] - X0))]
axB.add_patch(Circle((tc[2], tc[1]), 3.0, color=CABLE, alpha=0.30))
axB.add_patch(Circle((tc[2], tc[1]), 3.8, fill=False, ec=CABLE,
                     lw=0.9, ls="--"))
axB.annotate("tweeter cable\n(lumen + cover)", xy=(tc[2]-2.0, tc[1]+2.4),
             xytext=(-3.2, 306.6), fontsize=8,
             bbox=dict(fc="white", ec="none", alpha=0.75, pad=1),
             arrowprops=dict(arrowstyle="->", color=CABLE, lw=1.0))

# counterbore notch (remove crescent material inside O4.5 around axis)
rect(axB, CRES_LO-0.01, SEAT_Y, Z_AXIS-2.2, Z_AXIS+2.2, color="white",
     ec="none", zorder=3)
# clearance bore
rect(axB, SEAT_Y, LM_WEB_TOP, Z_AXIS-1.2, Z_AXIS+1.2, color="white",
     ec="none", zorder=3)
# seam gap
rect(axB, LM_WEB_TOP, UM_WEB_BOT, Z_REAR, Z_FRONT, color="white",
     ec="none", zorder=3)
# insert pocket
rect(axB, UM_WEB_BOT, POCKET_TOP, Z_AXIS-1.6, Z_AXIS+1.6, color="white",
     ec="none", zorder=3)

# screw: head, shank, thread into insert
rect(axB, SEAT_Y-2.0, SEAT_Y, Z_AXIS-1.9, Z_AXIS+1.9,
     color=SCREW_GREEN, zorder=5)                       # head 3.8 x 2.0
rect(axB, SEAT_Y, TIP_Y, Z_AXIS-1.0, Z_AXIS+1.0,
     color=SCREW_GREEN, alpha=0.9, zorder=5)            # shank/thread
# insert (red) around the thread inside the pocket
rect(axB, TIP_Y-2.5, TIP_Y, Z_AXIS-1.6, Z_AXIS-1.0,
     color=INSERT_RED, zorder=6)
rect(axB, TIP_Y-2.5, TIP_Y, Z_AXIS+1.0, Z_AXIS+1.6,
     color=INSERT_RED, zorder=6)

# hex key path
axB.plot([Z_AXIS, Z_AXIS], [SEAT_Y-2.3, 303.7], color="#444", lw=2.2,
         zorder=7)
axB.plot([Z_AXIS, -1.5], [303.7, 303.7], color="#444", lw=2.2, zorder=7)
axB.annotate("M2 L-key from open rear\n(elbow clears cable cover by 0.5)",
             xy=(2.5, 303.7), xytext=(-3.2, 298.0), fontsize=8, ha="left",
             bbox=dict(fc="white", ec="none", alpha=0.75, pad=1),
             arrowprops=dict(arrowstyle="->", color="#444", lw=1.0))

# z datum lines + labels
for z, lab in ((6.8, "z=6.8 core rear"), (11.45, "11.45"),
               (12.3, "12.3 LM seat"), (14.3, "14.3 UM seat"),
               (18.3, "z=18.3 front face")):
    axB.axvline(z, color="#bbb", lw=0.5, zorder=0)
    axB.text(z, 296.4, lab, rotation=90, fontsize=6.5, ha="right",
             va="bottom", color="#777",
             bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))

# y dimension callouts (right margin)
def ydim(ax, y0, y1, z, text, off=0.4):
    ax.annotate("", xy=(z, y0), xytext=(z, y1),
                arrowprops=dict(arrowstyle="<->", color="#222", lw=0.8))
    ax.text(z+off, (y0+y1)/2, text, fontsize=7.5, va="center")

ydim(axB, SEAT_Y, LM_WEB_TOP, 20.2, f"bore {LM_WEB_TOP-SEAT_Y:.2f}")
ydim(axB, UM_WEB_BOT, POCKET_TOP, 20.2, "")
axB.text(20.6, 318.5, "pocket 2.90", fontsize=7.5)
ydim(axB, TIP_Y-2.5, TIP_Y, 23.0, "")
axB.text(23.4, 314.7, "insert 2.5", fontsize=7.5)
ydim(axB, SEAT_Y, TIP_Y, 26.4, "")
axB.text(26.8, 313.9, "M2x8\n(8.00)", fontsize=7.5)
ydim(axB, CRES_LO, SEAT_Y, 20.2, f"c'bore {SEAT_Y-CRES_LO:.2f}")

for y, lab in ((ARC_R1106, f"y={ARC_R1106:.2f} R110.6 wall"),
               (ARC_R113, f"{ARC_R113:.2f} R113 core"),
               (LM_WEB_TOP, f"{LM_WEB_TOP:.2f} seam (gap 0.05)"),
               (UM_LIP_LO, f"{UM_LIP_LO:.2f} UM core edge"),
               (UM_LIP_HI, f"{UM_LIP_HI:.2f} R49.3")):
    axB.plot([Z_REAR-0.7, Z_REAR], [y, y], color="#222", lw=0.6)
    axB.text(Z_REAR-0.95, y, lab, fontsize=6.8, ha="right", va="center")

axB.set_xlim(-3.5, 28.5)
axB.set_ylim(296, 324.5)
axB.set_xlabel("z (mm)  [rear <-> front]", fontsize=8)
axB.set_ylabel("y (mm)", fontsize=8)
axB.tick_params(labelsize=7)
axB.set_title("B - section through x = -20 (to scale; seam gap 0.05 true width)",
              fontsize=10.5, pad=6)

fig.suptitle("Obi-Wan LM-UM vertical tie, option (A): one M2x8 at x = -20",
             fontsize=13, y=0.985)
out = "/Users/antor/.claude/jobs/4808081d/tmp/option_a_sketch.svg"
fig.savefig(out, format="svg", bbox_inches="tight")
print("wrote", out)
print("stack: seat", round(SEAT_Y, 2), "tip", round(TIP_Y, 2),
      "bore", round(LM_WEB_TOP - SEAT_Y, 2),
      "cbore", round(SEAT_Y - CRES_LO, 2),
      "crescent", round(CRES_LO, 2), round(CRES_HI, 2))
