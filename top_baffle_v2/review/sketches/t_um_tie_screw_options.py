"""T-UM tie: M2x8 / M2x12 / M2x16 compared in three views.

Bodies come from the released meshes (the previous round's bores filled
back in); the fastener stack for each option is drawn analytically with
the tip pinned where the tweeter duct allows it.
"""
import json
import math
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle, Circle
from pathlib import Path

sys.path.insert(0, "src")
import lx521_baffle.obiwan.carriers as ca            # noqa: E402


def load_world(stem, state="floor_stand"):
    base = Path("build") / state / "stl" / stem
    with open(base.with_suffix(".stl"), "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        rec = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tris = rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(float)
    d = json.load(open(str(base) + ".print.json"))
    t = np.asarray(d["stl_origin_translation_mm"], float)
    return (tris - t) * np.array([1.0, -1.0, -1.0])


def _bary(tris, u, v, iu, iv):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    d = ((b[:, iv] - c[:, iv]) * (a[:, iu] - c[:, iu])
         + (c[:, iu] - b[:, iu]) * (a[:, iv] - c[:, iv]))
    ok = np.abs(d) > 1e-12
    dd = np.where(ok, d, 1.0)
    w1 = ((b[:, iv] - c[:, iv]) * (u - c[:, iu])
          + (c[:, iu] - b[:, iu]) * (v - c[:, iv])) / dd
    w2 = ((c[:, iv] - a[:, iv]) * (u - c[:, iu])
          + (a[:, iu] - c[:, iu]) * (v - c[:, iv])) / dd
    w3 = 1.0 - w1 - w2
    h = ok & (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)
    return h, w1, w2, w3


def cross_y(tris, x, z):
    h, w1, w2, w3 = _bary(tris, x, z, 0, 2)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ys = np.sort(w1[h] * a[h][:, 1] + w2[h] * b[h][:, 1] + w3[h] * c[h][:, 1])
    return [(ys[i], ys[i + 1]) for i in range(0, len(ys) - 1, 2)]


def cross_z(tris, x, y):
    h, w1, w2, w3 = _bary(tris, x, y, 0, 1)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    zs = np.sort(w1[h] * a[h][:, 2] + w2[h] * b[h][:, 2] + w3[h] * c[h][:, 2])
    return [(zs[i], zs[i + 1]) for i in range(0, len(zs) - 1, 2)]


um, cres = load_world("obiwan_core_2_of_2_um_carrier"), \
    load_world("obiwan_addon_tweeter_crescent")
OLD_BORES_UM = [(409.036, 413.036, 2.2), (412.838, 417.770, 1.2)]
OLD_BORES_CR = [(417.220, 421.338, 1.6)]

X, Z = ca.T_UM_TIE_ABS_X, ca.T_UM_TIE_AXIS_Z
TIP = ca.T_UM_TIE_TIP_Y
MOUTH = ca.T_UM_TIE_INSERT_MOUTH_Y
BOTTOM = ca.T_UM_TIE_POCKET_BOTTOM_Y
CR_FACE, UM_FACE = ca.T_UM_TIE_CRES_FACE_Y, ca.T_UM_TIE_UM_FACE_Y
POCKET_D, KEY_D, CLR_D = (ca.T_UM_TIE_HEAD_POCKET_D,
                          ca.T_UM_TIE_KEY_BORE_D,
                          ca.T_UM_TIE_CLEARANCE_BORE_D)
HEAD_D = ca.T_UM_TIE_HEAD_D
CH_W = ca.T_UM_TIE_REAR_CHANNEL_W
CH_TOP = ca.T_UM_TIE_CHANNEL_TOP_Z
EDGE = max(hi for lo, hi in cross_y(cres, X, Z))

INK = "#3f3f3c"
UM_FILL = "#cdd6e0"
CR_FILL = "#ded3c6"
SCREW = "#2e7d32"
INSERT = "#c73a2f"
OK_C = "#2e7d32"
WARN_C = "#b97400"
BAD_C = "#b00020"

OPTIONS = []
for label, L, head_h in (("M2x8", 8, 2.0), ("M2x12", 12, 2.0),
                         ("M2x16", 16, 2.0)):
    seat = TIP + L
    top = seat + head_h
    key = EDGE - top
    if key < 0:
        color, verdict = BAD_C, f"head PROTRUDES {-key:.1f} mm"
    elif key < 1.5:
        color, verdict = WARN_C, "no key guidance"
    else:
        color, verdict = OK_C, f"head buried {key:.1f} mm"
    OPTIONS.append(dict(label=label, L=L, seat=seat, top=top, key=key,
                        color=color, verdict=verdict, head_h=head_h))

fig, axes = plt.subplots(3, 3, figsize=(13.4, 12.6), facecolor="white")

for col, opt in enumerate(OPTIONS):
    seat, top, key, color = opt["seat"], opt["top"], opt["key"], opt["color"]

    # ------------------------------------------------------------ FRONT
    ax = axes[0][col]
    ax.set_aspect("equal")
    for tris, fill in ((um, UM_FILL), (cres, CR_FILL)):
        ax.add_collection(PolyCollection(tris[:, :, :2], facecolors=fill,
                                         edgecolors="none", zorder=1))
    for sign in (-1, 1):
        if key >= 0:
            ax.add_patch(Circle((sign * X, EDGE), KEY_D / 2.0, color=color,
                                zorder=6))
        else:
            ax.add_patch(Circle((sign * X, top - opt["head_h"] / 2.0),
                                HEAD_D / 2.0, color=color, zorder=6))
    if key >= 0:
        cone = math.degrees(math.atan(KEY_D / max(key, 0.1)))
        caption = (f"O{KEY_D} hole; head {key:.1f} mm down\n"
                   f"visible only within {cone:.0f}\u00b0 of the axis")
    else:
        caption = "O3.8 head stands proud\nof the scallop edge"
    ax.text(0, 425.6, caption, fontsize=8, ha="center", color=color)
    ax.set_xlim(-24, 24)
    ax.set_ylim(424, 441)
    ax.set_title(f"{opt['label']}  -  {opt['verdict']}", fontsize=10.5,
                 color=color, pad=6)
    if col == 0:
        ax.set_ylabel("FRONT\ny (mm)", fontsize=8.5)
    ax.set_xlabel("x (mm)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)

    # ------------------------------------------------------------- SIDE
    ax = axes[1][col]
    ax.set_aspect("equal")
    STEP = 0.10
    z = 6.8 + STEP / 2.0
    while z < 18.3:
        for tris, fill, olds in ((um, UM_FILL, OLD_BORES_UM),
                                 (cres, CR_FILL, OLD_BORES_CR)):
            for lo, hi in cross_y(tris, X, z):
                ax.add_patch(Rectangle((z - STEP / 2.0, lo), STEP, hi - lo,
                                       color=fill, lw=0, zorder=1))
            for y0, y1, r in olds:
                if abs(z - 11.0) < r:
                    ax.add_patch(Rectangle((z - STEP / 2.0, y0), STEP,
                                           y1 - y0, color=fill, lw=0,
                                           zorder=1))
        z += STEP
    # cuts
    ax.add_patch(Rectangle((CH_TOP - 12.0, CR_FACE - 0.3), 12.0,
                           top + 0.3 - (CR_FACE - 0.3), color="white",
                           zorder=3))
    ax.add_patch(Rectangle((Z - POCKET_D / 2.0, seat), POCKET_D, top - seat,
                           color="white", zorder=3))
    if key > 0:
        ax.add_patch(Rectangle((Z - KEY_D / 2.0, top), KEY_D,
                               EDGE + 0.6 - top, color="white", zorder=3))
    ax.add_patch(Rectangle((Z - CLR_D / 2.0, CR_FACE - 0.3), CLR_D,
                           seat + 0.2 - (CR_FACE - 0.3), color="white",
                           zorder=3))
    ax.add_patch(Rectangle((Z - 1.6, BOTTOM), 3.2, MOUTH - BOTTOM,
                           color="white", zorder=3))
    ax.add_patch(Rectangle((Z - 1.2, MOUTH), 2.4, UM_FACE + 0.3 - MOUTH,
                           color="white", zorder=3))
    # hardware
    ax.add_patch(Rectangle((Z - HEAD_D / 2.0, seat), HEAD_D, opt["head_h"],
                           color=color, zorder=5))
    ax.add_patch(Rectangle((Z - 1.0, TIP), 2.0, seat - TIP, color=SCREW,
                           alpha=0.9, zorder=5))
    for side in (-1, 1):
        ax.add_patch(Rectangle((Z + side * 1.0, TIP), side * 0.6,
                               MOUTH - TIP, color=INSERT, zorder=6))
    ax.axhline(EDGE, color=BAD_C, lw=0.8, ls=(0, (5, 3)), zorder=7)
    ax.text(24.6, EDGE + 0.4, "scallop edge", fontsize=6.5, color=BAD_C,
            ha="right")
    ax.axhline(414.39, color="#a3670f", lw=0.8, ls=(0, (2, 2)), zorder=7)
    ax.text(24.6, 412.7, "tweeter duct (pins the tip)", fontsize=6.5,
            color="#a3670f", ha="right")
    ax.annotate("", xy=(21.0, TIP), xytext=(21.0, seat),
                arrowprops=dict(arrowstyle="<->", color=color, lw=0.9))
    ax.text(21.4, (TIP + seat) / 2.0, f"{opt['L']} mm", fontsize=7.5,
            color=color, va="center")
    ax.set_xlim(-1.0, 25.0)
    ax.set_ylim(410, 438)
    if col == 0:
        ax.set_ylabel("SIDE  (section x=+13)\ny (mm)", fontsize=8.5)
    ax.set_xlabel("z (mm)  [rear <-> front]", fontsize=7.5)
    ax.tick_params(labelsize=6.5)

    # -------------------------------------------------------------- TOP
    ax = axes[2][col]
    ax.set_aspect("equal")
    for xs in np.arange(-24.0, 24.01, 0.25):
        for lo, hi in cross_z(cres, xs, EDGE - 0.4):
            ax.add_patch(Rectangle((xs - 0.13, lo), 0.26, hi - lo,
                                   color=CR_FILL, lw=0, zorder=1))
    for sign in (-1, 1):
        ax.add_patch(Rectangle((sign * X - CH_W / 2.0, 6.4), CH_W,
                               CH_TOP - 6.4, fill=False, ec="#2f6f9f",
                               lw=1.0, hatch="///", zorder=4))
        if key >= 0:
            ax.add_patch(Circle((sign * X, Z), KEY_D / 2.0, color=color,
                                zorder=6))
        else:
            ax.add_patch(Circle((sign * X, Z), HEAD_D / 2.0, color=color,
                                zorder=6))
    ax.text(0, 19.4, "front face", fontsize=6.5, ha="center", color="#777")
    ax.text(0, 5.6, "rear (channels)", fontsize=6.5, ha="center",
            color="#2f6f9f")
    ax.set_xlim(-24, 24)
    ax.set_ylim(4.6, 20.4)
    if col == 0:
        ax.set_ylabel("TOP  (looking down\nthe screw axis)\nz (mm)",
                      fontsize=8.5)
    ax.set_xlabel("x (mm)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)

fig.suptitle("T-UM tie screw options - tip pinned by the tweeter duct, "
             "head must stay under the scallop edge",
             fontsize=13, y=0.975)
fig.tight_layout(rect=(0, 0, 1, 0.955))
out = "/Users/antor/.claude/jobs/4808081d/tmp/screw_options.png"
fig.savefig(out, dpi=150)
print("wrote", out)
for opt in OPTIONS:
    print(f"  {opt['label']:6} seat {opt['seat']:.2f} top {opt['top']:.2f} "
          f"key {opt['key']:+.2f}  {opt['verdict']}")
