"""Proposal sheet for the C-series experiment add-ons (C1..C5) on the B2
base: front views with driver overlays plus a section row showing each
variant's distinguishing profile (edge treatment / width / waveguide).

C1  ultra-wide wings          -- dipole path-length hypothesis
C2  sharp wedge edge wrap     -- < > knife arris at mid-thickness
C3  rounded edge wrap (R9.15) -- edge profile at fixed outline
C4  single-vertex spike wings -- coherent vs decorrelated edge delays
C5  single-bevel edge wrap    -- /__ knife edge on one face plane
                                 (asymmetric, vs C2's symmetric < >)
C6  rear-thinned vase         -- NOT an add-on: an alternate
                                 piece_top_b2 print. Rear face feathers
                                 18.3 -> ~6 toward the UM walls (front
                                 plane intact, crescent-taper style);
                                 T ducts rerouted externally, magnet
                                 sites kept on local full-depth bosses.
                                 Tests SL's "ideally even thinner" in
                                 his own 4-10 kHz critical band.

All sections cut at the UM axis (y=366.08). The C2/C3/C5 wraps are
drawn schematically as plan bands (their differences live in the
section row); C1/C4 outlines are exact; C6's band is the treated
rear-face region.

Run:  python gen_c_variants.py  ->  baffle_C_variants_drivers.png
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gen_driver_overlay import draw
from top_baffle_nd25fw4 import THICKNESS_MM, UM_CUTOUT
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2

ORANGE = "#f2a541"

# B2 landmarks (post-drop frame)
WAIST = (38.113, 315.947)
CREST = (60.654, 391.709)
NOTCH = (10.081, 418.176)
HORN = (36.813, 432.866)
ARC_C = (0.0, 468.193)          # crescent arc center (dropped frame)
ARC_R = 51.055                  # D102.11 / 2
UM_Y = UM_CUTOUT[1]             # 366.081

# flare / chamfer outward unit normals (right side)
N_FLARE = (0.95846, -0.28517)
N_CHAMF = (0.46370, 0.88603)
WRAP_W = 10.0                   # C2/C3 wrap width in plan


def _arc_pts(a0_deg, a1_deg, n=24):
    return [(ARC_C[0] + ARC_R * math.cos(math.radians(a)),
             ARC_C[1] + ARC_R * math.sin(math.radians(a)))
            for a in np.linspace(a0_deg, a1_deg, n)]


def _inner_return():
    """B2 wall path from the horn corner back down to the waist kink."""
    th_horn = math.degrees(math.atan2(HORN[1] - ARC_C[1], HORN[0]))
    th_notch = math.degrees(math.atan2(NOTCH[1] - ARC_C[1], NOTCH[0]))
    return _arc_pts(th_horn, th_notch) + [NOTCH, CREST, WAIST]


def wing_poly(apex):
    """Wing between B2's wall and a straight-line outer flank via apex."""
    return [WAIST, apex, HORN] + _inner_return()[1:]


def wrap_poly(w=WRAP_W):
    """Band along the flare+chamfer walls, offset w outward (negative =
    inward), blunted 5 mm before the notch corner, mitred at the crest."""
    u_ch = ((CREST[0] - NOTCH[0]) / 57.078, (CREST[1] - NOTCH[1]) / 57.078)
    p_end = (NOTCH[0] + 5.0 * u_ch[0], NOTCH[1] + 5.0 * u_ch[1])
    a0 = (WAIST[0] + w * N_FLARE[0], WAIST[1] + w * N_FLARE[1])
    b0 = (p_end[0] + w * N_CHAMF[0], p_end[1] + w * N_CHAMF[1])
    u_fl = (0.28517, 0.95846)
    # miter: intersect offset flare line (a0 + t*u_fl) with offset
    # chamfer line (b0 + s*u_ch)
    den = u_fl[0] * u_ch[1] - u_fl[1] * u_ch[0]
    t = ((b0[0] - a0[0]) * u_ch[1] - (b0[1] - a0[1]) * u_ch[0]) / den
    miter = (a0[0] + t * u_fl[0], a0[1] + t * u_fl[1])
    return [WAIST, a0, miter, b0, p_end, CREST, WAIST]


def _mirror(poly):
    return [(-x, y) for x, y in poly]


def flare_x_at(y):
    return WAIST[0] + 0.29752 * (y - WAIST[1])


# ---------------------------------------------------------------- sections
def _slab(ax, x0, x1, z0=0.0, z1=THICKNESS_MM, fc="0.88", **kw):
    for s in (1, -1):
        ax.fill([s * x0, s * x1, s * x1, s * x0],
                [z0, z0, z1, z1], fc=fc, ec="0.35", lw=0.8, **kw)


def section_common(ax, x_wall):
    """B2 base material at the UM-axis section: cutout rim to the wall."""
    _slab(ax, UM_CUTOUT[2] / 2.0, x_wall)
    ax.axhline(0, color="0.7", lw=0.4, zorder=0)


def draw_section(ax, variant):
    x_wall = flare_x_at(UM_Y)          # B2 wall at the UM axis (~53.0)
    t = THICKNESS_MM
    if variant == "c1":
        section_common(ax, x_wall)
        _slab(ax, x_wall, 110.0, fc=ORANGE)
        ax.set_title("section y=366 (UM axis) — square edge at ±110",
                     fontsize=8)
    elif variant in ("c2", "c3"):
        section_common(ax, x_wall)
        xo = x_wall + WRAP_W
        for s in (1, -1):
            if variant == "c2":   # sharp < > wedge: knife arris at z=t/2
                pts = [(x_wall, 0), (xo, t / 2), (x_wall, t)]
            else:                 # full R9.15 roundover cap
                cap = [(xo - 9.15 + 9.15 * math.cos(math.radians(a)),
                        t / 2 + 9.15 * math.sin(math.radians(a)))
                       for a in np.linspace(-90, 90, 24)]
                pts = [(x_wall, 0)] + cap + [(x_wall, t)]
            ax.fill([s * x for x, _ in pts], [z for _, z in pts],
                    fc=ORANGE, ec="0.35", lw=0.8)
        label = "sharp < > wedge" if variant == "c2" else "R9.15 rounded"
        ax.set_title(f"section y=366 — {label} edge at ±{xo:.0f}",
                     fontsize=8)
    elif variant == "c4":
        section_common(ax, x_wall)
        _slab(ax, x_wall, 95.0, fc=ORANGE)
        ax.set_title("section y=366 — through the plan vertex (±95)",
                     fontsize=8)
    elif variant == "c5":
        section_common(ax, x_wall)
        xo = x_wall + WRAP_W
        # single bevel /__ : rear face runs flat to a knife edge at xo,
        # the front face ramps down to it (mirror of C2's mid-plane arris)
        for s in (1, -1):
            pts = [(x_wall, 0), (xo, 0), (x_wall, t)]
            ax.fill([s * x for x, _ in pts], [z for _, z in pts],
                    fc=ORANGE, ec="0.35", lw=0.8)
        ax.set_title("section y=366 — single bevel /__ , knife edge on "
                     "the rear plane (±63)", fontsize=8)
    elif variant == "c6":
        # alternate vase piece: rear face feathers from full depth at
        # the pilot ring to ~6 mm at the wall; front face untouched
        x_rim = UM_CUTOUT[2] / 2.0        # 41
        x_full = 44.0                     # taper start (pilot ring)
        z_wall = t - 6.0                  # rear surface height at wall
        for s in (1, -1):
            keep = [(x_rim, 0), (x_full, 0), (x_wall, z_wall),
                    (x_wall, t), (x_rim, t)]
            ax.fill([s * x for x, _ in keep], [z for _, z in keep],
                    fc="0.88", ec="0.35", lw=0.8)
            cut = [(x_full, 0), (x_wall, 0), (x_wall, z_wall)]
            ax.fill([s * x for x, _ in cut], [z for _, z in cut],
                    fc=ORANGE, ec="0.3", lw=0.8, alpha=0.7, hatch="///")
            th = np.linspace(0, 2 * math.pi, 40)
            ax.plot(s * (46.9 + 1.9 * np.cos(th)), 3.7 + 1.9 * np.sin(th),
                    color="0.25", ls=":", lw=0.9)
        ax.axhline(0, color="0.7", lw=0.4, zorder=0)
        ax.set_title("section y=366 — rear feather 18.3→6 at the wall; "
                     "T duct (dotted) rerouted", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-125, 125)
    ax.set_ylim(-6, 40)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)


VARIANTS = [
    ("c1", "C1 ultra-wide wings",
     "dipole path length:\npeak/null shift down",
     [wing_poly((110.0, 380.0))]),
    ("c2", "C2 sharp wedge edge wrap",
     "edge profile A/B:\n< > knife arris vs round/square",
     [wrap_poly()]),
    ("c3", "C3 rounded edge wrap",
     "edge profile A/B:\nR9.15 = λ/8 at 4.7 kHz",
     [wrap_poly()]),
    ("c4", "C4 single-vertex wings",
     "decorrelated edge delays\nvs B2's equidistant wall",
     [wing_poly((95.0, UM_Y))]),
    ("c5", "C5 single-bevel edge wrap",
     "asymmetric edge /__ :\none face flat, one beveled",
     [wrap_poly()]),
    ("c6", "C6 rear-thinned vase (alt. top piece)",
     "SL's 'ideally even thinner':\nrear feather in the 4–10 kHz band",
     []),
]

if __name__ == "__main__":
    n = len(VARIANTS)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 11), dpi=130,
                             height_ratios=[635, 170])
    for col, (key, title, hyp, polys) in enumerate(VARIANTS):
        ax = axes[0][col]
        draw(ax, OUTLINE_B2, TWEETER_DROP_MM, title, labels=False)
        for poly in polys:
            for p in (poly, _mirror(poly)):
                ax.fill([x for x, _ in p], [y for _, y in p], fc=ORANGE,
                        ec="0.3", lw=0.9, alpha=0.85, zorder=6)
        if key == "c6":
            # treated rear-face band along the vase walls (not an add-on)
            band = wrap_poly(-9.0)
            for p in (band, _mirror(band)):
                ax.fill([x for x, _ in p], [y for _, y in p], fc=ORANGE,
                        ec="0.3", lw=0.8, alpha=0.5, hatch="///", zorder=6)
            for sx in (1, -1):  # magnet sites kept on full-depth bosses
                ax.plot(sx * 40.0, 322.4, marker="o", ms=5, mfc="white",
                        mec="0.25", zorder=8)
        ax.text(0.5, -0.03, hyp, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="0.25")
        ax.tick_params(labelsize=7)
        draw_section(axes[1][col], key)
    axes[0][0].set_ylabel("mm", fontsize=8)
    axes[1][0].set_ylabel("z (mm)", fontsize=8)
    fig.suptitle("C-series experiment topologies on the B2 base — front "
                 "views and distinguishing sections (add-ons / treated "
                 "regions in orange; C6 is an alternate top piece, hatched "
                 "= rear-face treatment)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig("baffle_C_variants_drivers.png")
    print("wrote baffle_C_variants_drivers.png")
