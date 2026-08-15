"""Reversed T-UM tie (b): what it looks like and what shows.

Front view is the real projected silhouette of the UM carrier and the
tweeter crescent; the section is reconstructed from the same meshes at
x=+13 with the superseded bores of the previous round filled back in,
then the new feature stack drawn analytically on top.
"""
import json
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

# Superseded round's bores at x=+-13, axis z=11 -- fill them back in so the
# section shows the body this design actually cuts into.
OLD = [(409.036, 413.036, 2.2), (412.838, 417.770, 1.2),
       (417.220, 421.338, 1.6)]

TIE_X = ca.T_UM_TIE_ABS_X
ZAX = ca.T_UM_TIE_AXIS_Z
SEAT = ca.T_UM_TIE_SEAT_Y
MOUTH = ca.T_UM_TIE_INSERT_MOUTH_Y
TIP = ca.T_UM_TIE_TIP_Y
BOTTOM = ca.T_UM_TIE_POCKET_BOTTOM_Y
UM_FACE = ca.T_UM_TIE_UM_FACE_Y
CR_FACE = ca.T_UM_TIE_CRES_FACE_Y

INK = "#3f3f3c"
UM_FILL = "#cdd6e0"
CR_FILL = "#ded3c6"
DUCT = "#e8c9a0"
SCREW = "#2e7d32"
INSERT = "#c73a2f"
HOLE = "#b00020"

fig = plt.figure(figsize=(11.0, 12.4), facecolor="white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.20)

# ---------------------------------------------------------------- front
axA = fig.add_subplot(gs[0])
axA.set_aspect("equal")
for tris, color in ((um, UM_FILL), (cres, CR_FILL)):
    axA.add_collection(PolyCollection(
        tris[:, :, :2], facecolors=color, edgecolors="none", zorder=1))

axA.plot([-52, 52], [ca.T_UM_TIE_SEAM_Y] * 2, color=INK, lw=0.7,
         ls=(0, (6, 4)), zorder=4)
axA.text(-50.5, ca.T_UM_TIE_SEAM_Y + 0.7, "UM / crescent seam",
         fontsize=7.5, color=INK)
for ex in (-24.0, 24.0):
    axA.add_patch(Circle((ex, 421.5), 4.9, fill=False, ec=INK, lw=1.0,
                         zorder=5))
axA.text(24.0, 415.6, "M3 half-lap ears", fontsize=7.5, ha="center",
         color=INK)
axA.add_patch(Rectangle((-6, 412), 12, 12, fill=False, ec="#777", lw=0.8,
                        ls=":", zorder=5))
axA.text(0, 410.4, "free-cable mouth", fontsize=7.5, ha="center",
         color="#777")

# breakout holes on the scallop edge
band_top = {}
for sign in (-1, 1):
    runs = cross_y(cres, sign * TIE_X, ZAX)
    band_top[sign] = max(hi for lo, hi in runs) if runs else 431.2
    axA.add_patch(Circle((sign * TIE_X, band_top[sign]),
                         ca.T_UM_TIE_CBORE_D / 2.0, color=HOLE, alpha=0.85,
                         zorder=8))
axA.annotate(
    f"the two O{ca.T_UM_TIE_CBORE_D} counterbore mouths\n"
    "(head sits ~8 mm inside) - this is\nall that shows, on the scallop edge",
    xy=(TIE_X, band_top[1]), xytext=(19.0, 441.0), fontsize=8.5, color=HOLE,
    arrowprops=dict(arrowstyle="->", color=HOLE, lw=1.2),
    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
axA.annotate("acoustic scallop (open)", xy=(0, 440.0), xytext=(-49.0, 446.0),
             fontsize=8.5, color="#6b6257",
             arrowprops=dict(arrowstyle="->", color="#6b6257", lw=1.0))
axA.text(0, 401.0, "UM carrier", ha="center", fontsize=9.5, color="#4a5a6b")
axA.text(-40.0, 431.0, "tweeter\ncrescent", ha="center", fontsize=9.5,
         color="#6b6257")
axA.set_xlim(-52, 52)
axA.set_ylim(398, 452)
axA.set_xlabel("x (mm)", fontsize=8)
axA.set_ylabel("y (mm)", fontsize=8)
axA.tick_params(labelsize=7)
axA.set_title("front view - real projected silhouette, to scale",
              fontsize=10.5, pad=6)

# -------------------------------------------------------------- section
axB = fig.add_subplot(gs[1])
axB.set_aspect("equal")
STEP = 0.08
for tris, color, tag in ((um, UM_FILL, "um"), (cres, CR_FILL, "cres")):
    z = 6.8 + STEP / 2.0
    while z < 18.3:
        for lo, hi in cross_y(tris, TIE_X, z):
            axB.add_patch(Rectangle((z - STEP / 2.0, lo), STEP, hi - lo,
                                    color=color, lw=0, zorder=1))
        # restore the superseded bores so the body reads as it really is
        if tag == "um":
            for y0, y1, r in OLD[:2]:
                if abs(z - ZAX + 1.0) < r:
                    axB.add_patch(Rectangle((z - STEP / 2.0, y0), STEP,
                                            y1 - y0, color=color, lw=0,
                                            zorder=1))
        else:
            y0, y1, r = OLD[2]
            if abs(z - ZAX + 1.0) < r:
                axB.add_patch(Rectangle((z - STEP / 2.0, y0), STEP, y1 - y0,
                                        color=color, lw=0, zorder=1))
        z += STEP

axB.text(15.6, 410.0, "buried tweeter\ncover fills the UM's\nrear recess - "
         "no corridor\nfor a head on this side", fontsize=8, color="#a3670f",
         ha="center",
         bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))

# new feature stack (cuts)
axB.add_patch(Rectangle((ZAX - 2.2, SEAT), 4.4, 12.5, color="white",
                        zorder=3))
axB.add_patch(Rectangle((ZAX - 1.2, CR_FACE - 0.3), 2.4,
                        SEAT + 0.2 - (CR_FACE - 0.3), color="white",
                        zorder=3))
axB.add_patch(Rectangle((ZAX - 1.6, BOTTOM), 3.2, MOUTH - BOTTOM,
                        color="white", zorder=3))
axB.add_patch(Rectangle((ZAX - 1.2, MOUTH), 2.4, UM_FACE + 0.3 - MOUTH,
                        color="white", zorder=3))
# screw + insert
axB.add_patch(Rectangle((ZAX - 1.9, SEAT), 3.8, 2.0, color=SCREW, zorder=5))
axB.add_patch(Rectangle((ZAX - 1.0, TIP), 2.0, SEAT - TIP, color=SCREW,
                        alpha=0.9, zorder=5))
for side in (-1, 1):
    axB.add_patch(Rectangle((ZAX + side * 1.0, TIP), side * 0.6,
                            MOUTH - TIP, color=INSERT, zorder=6))
axB.plot([ZAX, ZAX], [SEAT + 2.2, SEAT + 14.5], color="#444", lw=2.0,
         zorder=7)
axB.annotate("M2 hex key enters through\nthe scallop (open air)",
             xy=(ZAX, SEAT + 12.0), xytext=(ZAX + 3.4, SEAT + 13.5),
             fontsize=8, color="#444",
             arrowprops=dict(arrowstyle="->", color="#444", lw=1.0))


def ydim(y0, y1, z, text):
    axB.annotate("", xy=(z, y0), xytext=(z, y1),
                 arrowprops=dict(arrowstyle="<->", color="#222", lw=0.8))
    axB.text(z + 0.35, (y0 + y1) / 2.0, text, fontsize=7.5, va="center")


ydim(SEAT, TIP, 21.5, f"M2x{ca.T_UM_TIE_SCREW_L_MM:.0f} (8.00)")
ydim(TIP, MOUTH, 19.6, "insert 2.5")
ydim(SEAT, band_top[1], 23.9, f"counterbore {band_top[1] - SEAT:.2f}")
for y, lab in ((band_top[1], f"{band_top[1]:.1f} scallop edge (breakout)"),
               (SEAT, f"{SEAT:.2f} head seat"),
               (CR_FACE, f"{CR_FACE:.2f} seam (gap 0.05)"),
               (BOTTOM, f"{BOTTOM:.2f} receiver floor "
                        f"({ca.T_UM_TIE_RECEIVER_FLOOR_MM:.2f} left)")):
    axB.plot([6.1, 6.8], [y, y], color="#222", lw=0.6)
    axB.text(5.9, y, lab, fontsize=7, ha="right", va="center")
for z, lab in ((6.8, "z=6.8 rear"), (18.3, "z=18.3 front face")):
    axB.axvline(z, color="#bbb", lw=0.5, zorder=0)
    axB.text(z, 405.4, lab, rotation=90, fontsize=6.5, ha="right",
             va="bottom", color="#777")
axB.set_xlim(-1.0, 30.0)
axB.set_ylim(405, 440)
axB.set_xlabel("z (mm)   [rear <-> front]", fontsize=8)
axB.set_ylabel("y (mm)", fontsize=8)
axB.tick_params(labelsize=7)
axB.set_title(f"section through x = +{TIE_X:.0f} (to scale)", fontsize=10.5,
              pad=6)

fig.suptitle("Reversed T-UM tie: heads in the crescent, inserts in the UM "
             f"(2 x M2x8 at x = +/-{TIE_X:.0f}, axis z = {ZAX:.1f})",
             fontsize=12.5, y=0.975)
out = "/Users/antor/.claude/jobs/4808081d/tmp/t_um_tie_reversed.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print(f"band top {band_top[1]:.2f}  seat {SEAT:.2f}  cbore "
      f"{band_top[1] - SEAT:.2f}  tip {TIP:.2f}  floor "
      f"{ca.T_UM_TIE_RECEIVER_FLOOR_MM:.2f}")
