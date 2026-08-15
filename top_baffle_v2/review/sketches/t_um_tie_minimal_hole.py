"""T-UM tie, minimal-hole revision: head laid in from the rear, only a
key passage reaches the scallop.

Front and rear views are the real projected silhouettes; the section is
reconstructed from the same meshes at x=+13 with the superseded bores of
the previous round filled back in, then the new stack drawn on top.
"""
import json
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
from pathlib import Path

sys.path.insert(0, "src")
import lx521_baffle.obiwan.carriers as ca            # noqa: E402

STATE = "floor_stand"


def load_world(stem):
    base = Path("build") / STATE / "stl" / stem
    with open(base.with_suffix(".stl"), "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        rec = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tris = rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(float)
    d = json.load(open(str(base) + ".print.json"))
    t = np.asarray(d["stl_origin_translation_mm"], float)
    return (tris - t) * np.array([1.0, -1.0, -1.0])


def cross_y(tris, x, z):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    d = ((b[:, 2] - c[:, 2]) * (a[:, 0] - c[:, 0])
         + (c[:, 0] - b[:, 0]) * (a[:, 2] - c[:, 2]))
    ok = np.abs(d) > 1e-12
    dd = np.where(ok, d, 1.0)
    w1 = ((b[:, 2] - c[:, 2]) * (x - c[:, 0])
          + (c[:, 0] - b[:, 0]) * (z - c[:, 2])) / dd
    w2 = ((c[:, 2] - a[:, 2]) * (x - c[:, 0])
          + (a[:, 0] - c[:, 0]) * (z - c[:, 2])) / dd
    w3 = 1.0 - w1 - w2
    h = ok & (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)
    ys = np.sort(w1[h] * a[h][:, 1] + w2[h] * b[h][:, 1] + w3[h] * c[h][:, 1])
    return [(ys[i], ys[i + 1]) for i in range(0, len(ys) - 1, 2)]


um, cres = load_world("obiwan_core_2_of_2_um_carrier"), \
    load_world("obiwan_addon_tweeter_crescent")
OLD_BORES = [(409.036, 413.036, 2.2), (412.838, 417.770, 1.2),
             (417.220, 421.338, 1.6)]

X = ca.T_UM_TIE_ABS_X
Z = ca.T_UM_TIE_AXIS_Z
SEAT = ca.T_UM_TIE_SEAT_Y
HEAD_TOP = ca.T_UM_TIE_HEAD_TOP_Y
POCKET_D = ca.T_UM_TIE_HEAD_POCKET_D
KEY_D = ca.T_UM_TIE_KEY_BORE_D
CLR_D = ca.T_UM_TIE_CLEARANCE_BORE_D
CH_LO, CH_HI = ca.T_UM_TIE_CHANNEL_LOW_Y, ca.T_UM_TIE_CHANNEL_HIGH_Y
CH_W, CH_TOP = ca.T_UM_TIE_REAR_CHANNEL_W, ca.T_UM_TIE_CHANNEL_TOP_Z
MOUTH, TIP = ca.T_UM_TIE_INSERT_MOUTH_Y, ca.T_UM_TIE_TIP_Y
BOTTOM = ca.T_UM_TIE_POCKET_BOTTOM_Y
UM_FACE, CR_FACE = ca.T_UM_TIE_UM_FACE_Y, ca.T_UM_TIE_CRES_FACE_Y

INK = "#3f3f3c"
UM_FILL = "#cdd6e0"
CR_FILL = "#ded3c6"
SCREW = "#2e7d32"
INSERT = "#c73a2f"
HOLE = "#b00020"
CHAN = "#2f6f9f"

fig = plt.figure(figsize=(12.2, 12.8), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02],
                      width_ratios=[1.0, 0.62], hspace=0.20, wspace=0.16)

band_top = {}
for sign in (-1, 1):
    runs = cross_y(cres, sign * X, Z)
    band_top[sign] = max(hi for lo, hi in runs) if runs else 431.16

# ------------------------------------------------------------- front view
axA = fig.add_subplot(gs[0, :])
axA.set_aspect("equal")
for tris, color in ((um, UM_FILL), (cres, CR_FILL)):
    axA.add_collection(PolyCollection(tris[:, :, :2], facecolors=color,
                                      edgecolors="none", zorder=1))
axA.plot([-52, 52], [ca.T_UM_TIE_SEAM_Y] * 2, color=INK, lw=0.7,
         ls=(0, (6, 4)), zorder=4)
for ex in (-24.0, 24.0):
    axA.add_patch(Circle((ex, 421.5), 4.9, fill=False, ec=INK, lw=1.0,
                         zorder=5))
axA.text(24.0, 415.4, "M3 half-lap ears", fontsize=7.5, ha="center",
         color=INK)
for sign in (-1, 1):
    axA.add_patch(Circle((sign * X, band_top[sign]), KEY_D / 2.0,
                         color=HOLE, zorder=8))
    axA.add_patch(Circle((sign * X, band_top[sign]), 4.4 / 2.0, fill=False,
                         ec=HOLE, lw=0.8, ls=(0, (3, 2)), alpha=0.55,
                         zorder=7))
axA.annotate(
    f"O{KEY_D} key passage - the only mark on any\nvisible surface "
    f"(dashed = the O4.4 bore this replaces)",
    xy=(X + 1.0, band_top[1]), xytext=(20.0, 442.5), fontsize=8.5,
    color=HOLE, arrowprops=dict(arrowstyle="->", color=HOLE, lw=1.2),
    bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.5))
axA.annotate("acoustic scallop (open)", xy=(0, 439.0), xytext=(-50.0, 445.5),
             fontsize=8.5, color="#6b6257",
             arrowprops=dict(arrowstyle="->", color="#6b6257", lw=1.0))
axA.text(0, 400.6, "UM carrier", ha="center", fontsize=9.5, color="#4a5a6b")
axA.text(-41.0, 431.0, "tweeter\ncrescent", ha="center", fontsize=9.5,
         color="#6b6257")
axA.text(-50.5, ca.T_UM_TIE_SEAM_Y + 0.8, "seam", fontsize=7.5, color=INK)
axA.set_xlim(-52, 52)
axA.set_ylim(398, 452)
axA.set_xlabel("x (mm)", fontsize=8)
axA.set_ylabel("y (mm)", fontsize=8)
axA.tick_params(labelsize=7)
axA.set_title("front view - real projected silhouette, to scale",
              fontsize=10.5, pad=6)

# ---------------------------------------------------------------- section
axB = fig.add_subplot(gs[1, 0])
axB.set_aspect("equal")
STEP = 0.08
for tris, color, tag in ((um, UM_FILL, "um"), (cres, CR_FILL, "cres")):
    z = 6.8 + STEP / 2.0
    while z < 18.3:
        for lo, hi in cross_y(tris, X, z):
            axB.add_patch(Rectangle((z - STEP / 2.0, lo), STEP, hi - lo,
                                    color=color, lw=0, zorder=1))
        bores = OLD_BORES[:2] if tag == "um" else OLD_BORES[2:]
        for y0, y1, r in bores:
            if abs(z - 11.0) < r:
                axB.add_patch(Rectangle((z - STEP / 2.0, y0), STEP, y1 - y0,
                                        color=color, lw=0, zorder=1))
        z += STEP

# cuts
axB.add_patch(Rectangle((CH_TOP - 12.0, CH_LO), 12.0, CH_HI - CH_LO,
                        color="white", zorder=3))          # rear channel
axB.add_patch(Rectangle((Z - POCKET_D / 2.0, SEAT), POCKET_D,
                        HEAD_TOP - SEAT, color="white", zorder=3))
axB.add_patch(Rectangle((Z - KEY_D / 2.0, HEAD_TOP), KEY_D,
                        band_top[1] + 1.2 - HEAD_TOP, color="white",
                        zorder=3))
axB.add_patch(Rectangle((Z - CLR_D / 2.0, CR_FACE - 0.3), CLR_D,
                        SEAT + 0.2 - (CR_FACE - 0.3), color="white",
                        zorder=3))
axB.add_patch(Rectangle((Z - 1.6, BOTTOM), 3.2, MOUTH - BOTTOM,
                        color="white", zorder=3))
axB.add_patch(Rectangle((Z - 1.2, MOUTH), 2.4, UM_FACE + 0.3 - MOUTH,
                        color="white", zorder=3))
# channel hatch to read as open to the rear
axB.add_patch(Rectangle((CH_TOP - 4.0, CH_LO), 4.0, CH_HI - CH_LO,
                        fill=False, ec=CHAN, lw=1.1, hatch="///", zorder=4))
axB.annotate("rear loading channel\n(open to the back of the\nbaffle - lay "
             "the screw in\nsideways, then slide down)",
             xy=(CH_TOP - 1.6, 420.5), xytext=(13.8, 412.6), fontsize=8,
             color=CHAN, arrowprops=dict(arrowstyle="->", color=CHAN, lw=1.1),
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
# hardware
axB.add_patch(Rectangle((Z - 1.9, SEAT), 3.8, 2.0, color=SCREW, zorder=5))
axB.add_patch(Rectangle((Z - 1.0, TIP), 2.0, SEAT - TIP, color=SCREW,
                        alpha=0.9, zorder=5))
for side in (-1, 1):
    axB.add_patch(Rectangle((Z + side * 1.0, TIP), side * 0.6, MOUTH - TIP,
                            color=INSERT, zorder=6))
axB.plot([Z, Z], [HEAD_TOP + 0.4, band_top[1] + 4.0], color="#444", lw=1.7,
         zorder=7)
axB.annotate("1.5 hex key", xy=(Z, band_top[1] + 2.0),
             xytext=(Z + 3.2, band_top[1] + 4.2), fontsize=8, color="#444",
             arrowprops=dict(arrowstyle="->", color="#444", lw=1.0))


def ydim(y0, y1, z, text):
    axB.annotate("", xy=(z, y0), xytext=(z, y1),
                 arrowprops=dict(arrowstyle="<->", color="#222", lw=0.8))
    axB.text(z + 0.35, (y0 + y1) / 2.0, text, fontsize=7.5, va="center")


ydim(SEAT, TIP, 20.6, "M2x8")
ydim(TIP, MOUTH, 18.8, "insert 2.5")
ydim(HEAD_TOP, band_top[1], 22.9, f"key bore {band_top[1] - HEAD_TOP:.1f}")
for y, lab in ((band_top[1], f"{band_top[1]:.1f} scallop edge"),
               (HEAD_TOP, f"{HEAD_TOP:.2f} head top"),
               (SEAT, f"{SEAT:.2f} head seat"),
               (CR_FACE, f"{CR_FACE:.2f} seam"),
               (BOTTOM, f"{BOTTOM:.2f} receiver floor")):
    axB.plot([5.6, 6.6], [y, y], color="#222", lw=0.6)
    axB.text(5.4, y, lab, fontsize=6.8, ha="right", va="center",
             bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.4))
for z, lab in ((6.8, "z=6.8 rear"), (18.3, "z=18.3 front")):
    axB.axvline(z, color="#bbb", lw=0.5, zorder=0)
    axB.text(z, 411.4, lab, rotation=90, fontsize=6.5, ha="right",
             va="bottom", color="#777")
axB.set_xlim(-1.5, 27.0)
axB.set_ylim(411, 437)
axB.set_xlabel("z (mm)   [rear <-> front]", fontsize=8)
axB.set_ylabel("y (mm)", fontsize=8)
axB.tick_params(labelsize=7)
axB.set_title(f"section through x = +{X:.0f}", fontsize=10.5, pad=6)

# ------------------------------------------------------------- rear detail
axC = fig.add_subplot(gs[1, 1])
axC.set_aspect("equal")
axC.add_collection(PolyCollection(cres[:, :, :2], facecolors=CR_FILL,
                                  edgecolors="none", zorder=1))
axC.add_collection(PolyCollection(um[:, :, :2], facecolors=UM_FILL,
                                  edgecolors="none", zorder=1))
for sign in (-1, 1):
    axC.add_patch(Rectangle((sign * X - CH_W / 2.0, CH_LO), CH_W,
                            CH_HI - CH_LO, facecolor="white", ec=CHAN,
                            lw=1.2, hatch="///", zorder=6))
    axC.add_patch(Circle((sign * X, band_top[sign]), KEY_D / 2.0,
                         color=HOLE, zorder=7))
axC.plot([-30, 30], [ca.T_UM_TIE_SEAM_Y] * 2, color=INK, lw=0.7,
         ls=(0, (6, 4)), zorder=5)
axC.annotate("the two rear channels\n(back of the baffle - never seen)",
             xy=(-X, 421.5), xytext=(-27.0, 412.0), fontsize=8, color=CHAN,
             arrowprops=dict(arrowstyle="->", color=CHAN, lw=1.1))
axC.set_xlim(-29, 29)
axC.set_ylim(409, 437)
axC.set_xlabel("x (mm)", fontsize=8)
axC.tick_params(labelsize=7)
axC.set_title("rear view detail", fontsize=10.5, pad=10)

fig.suptitle("T-UM tie, minimal-hole revision: M2x8 head laid in from the "
             f"rear; only a O{KEY_D} key passage reaches the scallop",
             fontsize=12.5, y=0.975)
out = "/Users/antor/.claude/jobs/4808081d/tmp/t_um_tie_minimal.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print(f"key bore O{KEY_D} vs previous O4.4 -> "
      f"{(KEY_D / 4.4) ** 2 * 100:.0f}% of the visible area")
