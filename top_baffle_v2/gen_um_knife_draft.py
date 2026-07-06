"""DRAFT: the super-minimalist UM vase ("V0"): the FRONT face slides
(C5-style single bevel, knife on the rear plane) from full 18.3 inside
the flange/pilot land down to a ~0.5 knife at the vase outline. The
REAR plane stays intact, so the T upper lanes (z=3.7) and the UM exit
tail keep their routing untouched -- no duct changes at all. Magnet
pin pockets bore NORMAL to the slide face; attachments (any depth)
scarf onto the slide. Draft only; no geometry code touched.

Keeps at full depth: flange+pilot disc (r<=49.3 about the UM), seam-B
band (y<=324) incl. the dovetail pockets and the UM exit corridor
(|x|<=14, y<=340), and a recovery before the crescent zone (y>=400,
which keeps its own rear taper + clamp seat).

Run: python gen_um_knife_draft.py -> baffle_um_knife_draft.png
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from top_baffle_nd25fw4 import UM_CUTOUT, THICKNESS_MM
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2
from top_baffle_nd25fw4_cables import route_points, CABLE_D
from gen_driver_overlay import draw

T, TE, W = THICKNESS_MM, 0.5, 12.0
UMX, UMY, UMD = UM_CUTOUT
WALLS = [((38.113, 315.947), (60.654, 391.709)),
         ((60.654, 391.709), (10.081, 418.176))]
ARC_C, ARC_R = (0.0, 468.193), 51.055

def S(u):
    u = min(max(u, 0.0), 1.0)
    return 3*u*u - 2*u*u*u

def d_wall(x, y):
    p = (abs(x), y)
    best = 1e9
    for a, b in WALLS:
        vx, vy = b[0]-a[0], b[1]-a[1]
        t = max(0, min(1, ((p[0]-a[0])*vx+(p[1]-a[1])*vy)/(vx*vx+vy*vy)))
        best = min(best, math.dist(p, (a[0]+t*vx, a[1]+t*vy)))
    best = min(best, abs(ARC_R - math.dist(p, ARC_C)))
    return best

def wall_x(y):
    """|x| of the vase outline at height y (flare, then chamfer)."""
    if y <= 391.709:
        return 38.113 + 0.29752 * (y - 315.947)
    return 60.654 - 1.9108 * (y - 391.709)


def inside(x, y):
    return 315.9 < y < 418 and abs(x) < wall_x(y) - 0.02


def front_cut(x, y):
    """FRONT-side removal depth (rear plane intact)."""
    if not (315.9 < y < 419):
        return 0.0
    target = TE + (T-TE)*S(d_wall(x, y)/W)
    keep = 0.0
    r = math.dist((x, y), (UMX, UMY))
    keep = max(keep, 1-S((r-49.3)/4.0))          # flange + M3 pilot land
    keep = max(keep, 1-S((y-324.0)/8.0))          # seam-B band + keys
    if abs(x) < 14:
        keep = max(keep, 1-S((y-332.0)/8.0))      # UM exit corridor
    keep = max(keep, S((y-400.0)/10.0))           # crescent recovery
    return (T - target) * (1.0 - keep)

fig = plt.figure(figsize=(15, 10.5), dpi=140)
gs = fig.add_gridspec(3, 2, width_ratios=[300, 300], hspace=0.5)
ax = fig.add_subplot(gs[:, 0])
draw(ax, OUTLINE_B2, TWEETER_DROP_MM,
     "V0 draft — UM vase FRONT-face slide to a rear-plane knife (C5 style)",
     labels=False)
xs = np.arange(-64, 64, 1.2); ys = np.arange(312, 422, 1.2)
cc = np.full((len(ys), len(xs)), np.nan)
for j, y in enumerate(ys):
    for i, x in enumerate(xs):
        c = front_cut(x, y)
        if c > 0.25 and inside(x, y) and \
           math.dist((x, y), (UMX, UMY)) > UMD/2:
            cc[j, i] = c
ax.contourf(xs, ys, cc, levels=np.linspace(0.25, T, 12), cmap="Oranges",
            alpha=0.8, zorder=5)
for name, col in (("ts", "gold"), ("um", "tab:green")):
    p = np.array([q[:2] for q in route_points(name)])
    ax.plot(p[:, 0], p[:, 1], color=col, lw=CABLE_D[name]*0.6, alpha=0.35,
            zorder=6)
for sx in (1, -1):  # magnet pins ON the slide (normal to the slide face)
    ax.plot(sx*47.0, 345.0, marker="o", ms=6, mfc="white", mec="0.2", zorder=8)
    ax.plot(sx*38.0, 400.0, marker="o", ms=6, mfc="white", mec="0.2", zorder=8)
ax.annotate("pin magnets bored NORMAL\nto the slide face; attachments\n"
            "scarf onto the slide", (47, 345), (75, 300), fontsize=8,
            color="0.25", arrowprops=dict(arrowstyle="-", color="0.5"))
ax.annotate("full-depth keeps: flange+pilot land,\nseam-B band + UM exit, "
            "crescent zone", (0, 366), (-160, 250), fontsize=8, color="0.25",
            arrowprops=dict(arrowstyle="-", color="0.5"))
ax.set_xlim(-170, 170); ax.set_ylim(290, 470)
ax.set_xlabel("mm"); ax.set_ylabel("mm")

for k, (y0, ttl) in enumerate(((391.0, "section y=391 (crest): slide to knife"),
                               (345.0, "section y=345 (lower triangles)"),
                               (366.1, "section y=366 (UM axis)"))):
    axs = fig.add_subplot(gs[k, 1])
    xs2 = np.linspace(-64, 64, 700)
    rim = math.sqrt(max((UMD/2)**2 - (y0-UMY)**2, 0))
    for sgn in (1, -1):
        seg = [x for x in xs2 if x*sgn > rim and inside(x, y0)]
        if not seg: continue
        seg = np.array(seg)
        front = np.array([T - front_cut(x, y0) for x in seg])
        axs.fill(np.concatenate([seg, seg[::-1]]),
                 np.concatenate([np.zeros_like(seg), front[::-1]]),
                 fc="0.88", ec="0.35", lw=0.8)
    axs.axhline(0, color="0.25", lw=0.8)
    axs.text(0.01, 0.06, "rear plane (kept intact: all ducts unchanged)",
             transform=axs.transAxes, fontsize=7, color="0.35")
    # ducts at true positions
    for name, col in (("ts", "gold"), ("um", "tab:green")):
        pts = [q for q in route_points(name)]
        for a, b in zip(pts, pts[1:]):
            if (a[1]-y0)*(b[1]-y0) <= 0 and a[1] != b[1]:
                f = (y0-a[1])/(b[1]-a[1])
                x = a[0]+f*(b[0]-a[0]); z = a[2]+f*(b[2]-a[2])
                axs.add_patch(plt.Circle((x, z), CABLE_D[name]/2, fc=col,
                                         ec=col, alpha=0.8))
    if abs(y0-366.1) < 1:  # M3 pilot band at the ring (45-deg bores off-plane)
        for sx in (1, -1):
            axs.add_patch(plt.Rectangle((sx*44.75-2.3, 11.3), 4.6, 7.0,
                                        fill=False, ec="0.45", ls=":", lw=0.8))
    axs.set_aspect("equal"); axs.set_xlim(-66, 66); axs.set_ylim(-2, 21)
    axs.set_title(ttl, fontsize=9); axs.grid(lw=0.3, alpha=0.4)
    axs.tick_params(labelsize=7)
fig.suptitle("V0: minimalist UM vase — front bevel (C5), rear plane and ALL "
             "duct routing untouched; flange/pilot land full depth",
             fontsize=12, y=0.99)
fig.tight_layout()
fig.savefig("baffle_um_knife_draft.png", bbox_inches="tight")
print("wrote baffle_um_knife_draft.png")
