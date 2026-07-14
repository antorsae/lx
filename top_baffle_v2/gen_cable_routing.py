"""Render separate proud/R6P and skeletal-V1LF/R6F routing sheets.

Faithful: both sheets sample the same complete centerlines used by the
subtractive proud cutters or integral V1LF printed-cover spans.  The proud
sheet shows the normal B2/C7/V0/V1 UM handoff plus the exact, clearly
labeled V1L-only alternate tail to its 283-degree rear-face aperture.
The V1LF sheet shows fully covered local Z bumps, their full-width burial
webs and solid roof-to-blind-bore backfill in nominal diametric sections, the short crown
crossover with T above UM, 0.8 mm minimum tube skins, the short free LM
lead (no micro-duct), the UM-cable cover owned only by LM, the T cover owned
only by LM/UM, their deliberately free cable handoffs, state-specific
support/feeds, six flush-buried LM/UM magnets, the free UM cable behind the
UM carrier, the 283 degree terminal clock and conservative Faston envelope.

  front view (x-y)   duct mains + breakout/exit markers over the outline
  side view  (y-z)   full-height depth story, sharing y with the front
                     view: exact printed-cover spans, free cable gaps,
                     state-specific feed rises, local Z bumps and the
                     free cable handoff behind UM
  top view   (x-z)   STAND_FOOT only, sharing x with the front view:
                     foot taper, packed lanes, channel and panel
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        guard = Path(__file__).with_name("run_memory_guarded.py")
        raise SystemExit(subprocess.run(
            [sys.executable, str(guard), "--", sys.executable,
             str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False).returncode)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
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
    UM_TERMINAL_CLOCK_DEG,
    _crescent_taper_depth,
)
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2
from top_baffle_nd25fw4_cables import (
    BIG_RAMPS,
    CABLE_D,
    EXIT_RAMPS,
    FOOT_LANES,
    ROUTING_PROFILE,
    ROUTING_REV,
    SUPPORT_WINDOW,
    T_RAMP,
    T_RAMP_L,
    UM_HANDOFF,
    UM_HANDOFF_D_MM,
    UM_V1L_AXIS_STATION_MM,
    UM_V1L_HANDOFF_KEY,
    UM_V1L_REAR_FACE_Z_MM,
    route_centerline_points,
    route_points,
)

STYLE = {
    "lm": ("tab:blue", "LM 2x2.5mm2, duct D8.2 (z=12.55)"),
    "um": ("tab:green", "UM D8.2 normal B2/C7/V0/V1 (z=12.55)"),
    "ts": ("gold", "T1+T2 shared, 2x(2xAWG24), duct D6.0 (z=11.5)"),
    "t1f": ("tab:red", "T pair feeders, D3.8 (z=3.7, strip)"),
    "t2f": ("tab:red", ""),
}
V1L_ALT_COLOR = "tab:purple"
V1L_ALT_LABEL = (
    f"UM D{UM_HANDOFF_D_MM:g} V1L only: "
    f"{UM_TERMINAL_CLOCK_DEG:g} deg alternate tail")
LEGACY_ROUTING_PNG = Path("baffle_cable_routing.png")


def _prune_legacy_routing_png():
    if LEGACY_ROUTING_PNG.exists():
        LEGACY_ROUTING_PNG.unlink()


def _save_routing_figure(fig, output, **kwargs):
    """Failure-atomic PNG with embedded profile/state provenance."""
    output = Path(output)
    state = "floor_stand" if STAND_FOOT else "no_floor_stand"
    token = f"LX521_{ROUTING_PROFILE.upper()}_{ROUTING_REV}_{state}"
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.tmp.png")
    metadata = {
        "Title": token,
        "Description": (
            f"{token}; LX_STAND_FOOT={int(STAND_FOOT)}; "
            f"LX_ROUTING_PROFILE={ROUTING_PROFILE}"
            + ("; LX_V1LF_SIDE_SECTION=roof_to_bore_solid_backfill"
               if ROUTING_PROFILE == "v1lf" else "")),
    }
    try:
        fig.savefig(temporary, metadata=metadata, **kwargs)
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                raise RuntimeError("temporary routing PNG has no pixels")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
TOP_Y = 468.314 - TWEETER_DROP_MM       # B2 top edge (453.46)
TAPER_CY = CRESCENT_SCALLOP_CY - TWEETER_DROP_MM


def duct_xyz(name, n=400):
    if name == "um":
        return np.array(route_centerline_points("um", spacing_mm=1.0))
    path = Spline(*route_points(name))
    pts = [path @ (i / n) for i in range(n + 1)]
    return np.array([[p.X, p.Y, p.Z] for p in pts])


def v1l_um_tail_xyz():
    """Exact modeled V1L UM tail, trimmed to its last shared plan knot."""
    pts = np.array(route_centerline_points(
        "um", spacing_mm=0.5, um_handoff_key=UM_V1L_HANDOFF_KEY))
    anchor_xy = np.array((61.76, 283.11))
    start = int(np.argmin(np.linalg.norm(pts[:, :2] - anchor_xy, axis=1)))
    return pts[start:]


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
    return T_RAMP if name == "t1f" else T_RAMP_L


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
    """Full-height y-z profile: plate, pilots, complete duct paths,
    rear handoffs and optional stand foot."""
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
        ax.annotate("blind heat-set pilot bores\n(rotated 10F pattern clears\n"
                    "the shared T duct in plan)", (7.3, 334.4), (-148, 320),
                    fontsize=8, color="0.3",
                    arrowprops=dict(arrowstyle="-", color="0.5"))
    # duct mains (true z from the swept splines)
    for name, (color, _) in STYLE.items():
        pts = duct_xyz(name)
        ax.plot(pts[:, 2], pts[:, 1], color=color, lw=CABLE_D.get(name, 3.8) * 0.8,
                alpha=0.55, solid_capstyle="round", zorder=6)
    # V1L keeps the proud main but substitutes this exact terminal tail.
    # Plot it separately so the standard R14 outlet remains unambiguous.
    v1l_tail = v1l_um_tail_xyz()
    v1l_spec = UM_HANDOFF[UM_V1L_HANDOFF_KEY]
    q = v1l_spec["rear_face_axis_point"]
    rear_end = v1l_spec["rear_end"]
    ax.plot(v1l_tail[:, 2], v1l_tail[:, 1], color=V1L_ALT_COLOR,
            lw=UM_HANDOFF_D_MM * 0.8, alpha=0.18,
            solid_capstyle="round", zorder=8)
    ax.plot(v1l_tail[:, 2], v1l_tail[:, 1], color=V1L_ALT_COLOR,
            lw=2.2, ls="--", alpha=0.95, zorder=9)
    ax.plot(q[2], q[1], marker="s", ms=6, mfc="white",
            mec=V1L_ALT_COLOR, mew=1.2, zorder=10)
    ax.plot(rear_end[2], rear_end[1], marker="v", ms=5,
            mfc=V1L_ALT_COLOR, mec=V1L_ALT_COLOR, zorder=10)
    ax.plot([UM_V1L_REAR_FACE_Z_MM, UM_V1L_REAR_FACE_Z_MM],
            [276.0, 315.95], color=V1L_ALT_COLOR, ls=":", lw=1.0,
            alpha=0.75, zorder=5)
    ax.annotate(
                f"V1L Q @ z={q[2]:.1f}\n"
                f"{UM_TERMINAL_CLOCK_DEG:g} deg / TPU seat",
                (q[2], q[1]), (0.96, 0.62),
                textcoords="axes fraction", fontsize=7.2,
                ha="right", va="center", color=V1L_ALT_COLOR,
                arrowprops=dict(arrowstyle="-", color=V1L_ALT_COLOR))
    # Separate legacy LM rear bore.  The UM outlet is already in the
    # complete R14 centerline plotted above.
    for name, (p0, p1, _dia) in EXIT_RAMPS.items():
        color = STYLE[name][0]
        ax.plot([p0[2], p1[2]], [p0[1], p1[1]], color=color, ls=":",
                lw=CABLE_D.get(name, 3.8) * 0.6, alpha=0.8, zorder=7)
        ax.plot(p1[2], p1[1], marker="o", ms=6, mfc="white", mec=color,
                zorder=8)
    ax.plot(11.5, 433.5, marker="o", ms=5, mfc="white", mec=STYLE["ts"][0],
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
            lane_key = {"t1f": "t1", "t2f": "t2"}.get(name, name)
            if lane_key not in FOOT_LANES:
                continue
            lane = _foot_lane_yz(lane_key)
            ax.plot(lane[:, 1], lane[:, 0], color=color,
                    lw=FOOT_LANES[lane_key][4] * 0.8, alpha=0.45,
                    solid_capstyle="round", zorder=6)
            ax.plot(-101, FOOT_LANES[lane_key][2], marker="o", ms=5,
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
            if name == "ts":
                continue  # shared main begins at TS_STEP, not rear face
            p0, p1 = _ramp(name)
            color = STYLE[name][0]
            ax.plot([p0[2], p1[2]], [p0[1], p1[1]], color=color, ls=":",
                    lw=CABLE_D.get(name, 3.8) * 0.6, alpha=0.8, zorder=7)
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
        lane_key = {"t1f": "t1", "t2f": "t2"}.get(name, name)
        if lane_key not in FOOT_LANES:
            continue
        x, z_d, y_f, r, dia = FOOT_LANES[lane_key]
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


def _radial_box_polygon(center, angle_deg, r0, r1, half_t, t_center=0.0):
    """XY corners of a radial/tangential service box."""
    a = math.radians(angle_deg)
    u = np.array((math.cos(a), math.sin(a)))
    v = np.array((-math.sin(a), math.cos(a)))
    c = np.array(center)
    return np.array([c + r0 * u + (t_center - half_t) * v,
                     c + r1 * u + (t_center - half_t) * v,
                     c + r1 * u + (t_center + half_t) * v,
                     c + r0 * u + (t_center + half_t) * v])


def draw_terminal_service(ax, compact=False):
    """Proud/V1L terminal gap and conservative service envelope."""
    from matplotlib.patches import Wedge
    from top_baffle_nd25fw4 import UM_TERMINAL_GAP_DEG
    from top_baffle_nd25fw4_um_fit import (
        FASTON_BOOT_RADIAL_L,
        FASTON_BOOT_TANGENTIAL_W,
        FASTON_PAIR_PITCH,
        FASTON_PULL_DISTANCE,
        TERMINAL_CONTACT_RADIUS,
    )

    center = UM_CUTOUT[:2]
    lo, hi = UM_TERMINAL_GAP_DEG
    ax.add_patch(Wedge(center, 70.0, lo, hi, width=18.0,
                       fc="tab:orange", ec="tab:orange", alpha=0.12,
                       lw=0.8, zorder=4))
    a = math.radians(UM_TERMINAL_CLOCK_DEG)
    ax.plot([center[0] + 20 * math.cos(a), center[0] + 70 * math.cos(a)],
            [center[1] + 20 * math.sin(a), center[1] + 70 * math.sin(a)],
            color="tab:orange", ls="--", lw=1.3, zorder=9)
    envelopes = []
    for off in (-FASTON_PAIR_PITCH / 2.0, FASTON_PAIR_PITCH / 2.0):
        env = _radial_box_polygon(
            center, UM_TERMINAL_CLOCK_DEG,
            TERMINAL_CONTACT_RADIUS,
            TERMINAL_CONTACT_RADIUS + FASTON_BOOT_RADIAL_L
            + FASTON_PULL_DISTANCE,
            FASTON_BOOT_TANGENTIAL_W / 2.0,
            off)
        envelopes.append(env)
        ax.add_patch(plt.Polygon(
            env, closed=True, fc="tab:red", ec="tab:red",
            alpha=0.10, ls=":", lw=1.2, zorder=5))
    if not compact:
        text_xy = (62, 350)
        ax.annotate(
                    f"terminals clock {UM_TERMINAL_CLOCK_DEG:g} deg\n"
                    "midway between screws 238/328\n"
                    f"red = per-tab {FASTON_PULL_DISTANCE:g} mm service "
                    "envelopes\nproxy hardware; physical dry-fit required",
                    envelopes[-1][2], text_xy, fontsize=7.5,
                    color="tab:orange",
                    arrowprops=dict(arrowstyle="-", color="tab:orange"))


def _polyline_interval(points, start_mm, end_mm):
    """Return a polyline interval with interpolated arc-length endpoints."""
    points = np.asarray(points, dtype=float)
    source_s = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    start_mm = float(np.clip(start_mm, 0.0, source_s[-1]))
    end_mm = float(np.clip(end_mm, start_mm, source_s[-1]))
    stations = np.concatenate((
        [start_mm], source_s[(source_s > start_mm) & (source_s < end_mm)],
        [end_mm]))
    return np.column_stack(tuple(
        np.interp(stations, source_s, points[:, axis])
        for axis in range(points.shape[1])))


def _draw_v1lf_terminal_service(
        ax, facts, route_handoff_xy, bundle_diameter):
    """Draw the free D82-to-terminal handoff and terminal service states."""
    from matplotlib.patches import Wedge
    from top_baffle_nd25fw4 import UM_TERMINAL_GAP_DEG
    from top_baffle_nd25fw4_um_fit import (
        FASTON_BOOT_RADIAL_L,
        FASTON_BOOT_TANGENTIAL_W,
        FASTON_BREAKOUT_BUNDLE_OD,
        FASTON_BREAKOUT_BUNDLE_OVERLAP_MM,
        FASTON_BREAKOUT_JUNCTION_LENGTH_MM,
        FASTON_BREAKOUT_LEAD_OD,
        FASTON_BREAKOUT_LENGTH,
        FASTON_LEAD_D,
        FASTON_LEAD_MIN_BEND_R,
        FASTON_LEAD_PULL_STATES_MM,
        FASTON_PAIR_PITCH,
        FASTON_PULL_DISTANCE,
        FASTON_RECEPTACLE_RADIAL_L,
        FASTON_RECEPTACLE_TANGENTIAL_W,
        FASTON_TAB_EXPOSED_L,
        FASTON_TAB_W,
        TERMINAL_CONTACT_RADIUS,
        V1LF_TERMINATED_HANDOFF_R,
        V1LF_TERMINATED_HANDOFF_STEPS,
        v1lf_terminal_lead_points,
        v1lf_terminal_lead_points_for_terminal_pull,
        v1lf_terminated_cable_points,
    )
    from top_baffle_nd25fw4_v1lf_route import UM_MOUTH_TANGENT

    center = UM_CUTOUT[:2]
    clock_deg = float(facts["terminal_clock_deg"])
    lo, hi = UM_TERMINAL_GAP_DEG
    ax.add_patch(Wedge(center, 70.0, lo, hi, width=18.0,
                       fc="tab:orange", ec="tab:orange", alpha=0.10,
                       lw=0.8, zorder=4))
    a = math.radians(clock_deg)
    radial = np.asarray((math.cos(a), math.sin(a)))
    tangent = np.asarray((-radial[1], radial[0]))
    ax.plot([center[0] + 20.0 * radial[0],
             center[0] + 72.0 * radial[0]],
            [center[1] + 20.0 * radial[1],
             center[1] + 72.0 * radial[1]],
            color="tab:orange", ls="--", lw=1.3, zorder=9)

    terminated = np.asarray(v1lf_terminated_cable_points(), dtype=float)
    # The entire UM-side span is free cable. The sampled route reaches the
    # D82 reference before this final handoff continues to the terminal axis;
    # neither interval is a small printed duct or a grommet.
    free_bundle = terminated[-(V1LF_TERMINATED_HANDOFF_STEPS + 1):]
    free_entry = np.vstack((
        np.asarray(route_handoff_xy, dtype=float),
        free_bundle[0, :2],
    ))
    ax.plot(free_entry[:, 0], free_entry[:, 1],
            color="white", lw=bundle_diameter + 1.2,
            solid_capstyle="butt", zorder=8)
    ax.plot(free_entry[:, 0], free_entry[:, 1],
            color=STYLE["um"][0], lw=bundle_diameter * 0.78,
            alpha=0.96, solid_capstyle="butt", zorder=9)
    ax.plot(free_bundle[:, 0], free_bundle[:, 1],
            color=STYLE["um"][0], lw=bundle_diameter * 0.78,
            alpha=0.92, solid_capstyle="round", zorder=9)
    ax.plot(*route_handoff_xy, marker="o", ms=5.0, mfc="white",
            mec=STYLE["um"][0], mew=1.0, zorder=13)
    ax.plot(*free_bundle[0, :2], marker=".", ms=3.5,
            color=STYLE["um"][0], zorder=13)

    lead_colors = {
        "terminal_lead_1": "tab:blue",
        "terminal_lead_2": "tab:red",
    }
    installed_leads = {
        name: np.asarray(points, dtype=float)
        for name, points in v1lf_terminal_lead_points().items()
    }

    # Installed tabs, receptacles and boots.  Each terminal keeps its own
    # polarity color; the receptacle is nested inside the boot outline.
    terminal_offsets = {
        1: -FASTON_PAIR_PITCH / 2.0,
        2: FASTON_PAIR_PITCH / 2.0,
    }
    for terminal_id, offset in terminal_offsets.items():
        color = lead_colors[f"terminal_lead_{terminal_id}"]
        tab = _radial_box_polygon(
            center, clock_deg,
            TERMINAL_CONTACT_RADIUS,
            TERMINAL_CONTACT_RADIUS + FASTON_TAB_EXPOSED_L,
            FASTON_TAB_W / 2.0, offset)
        receptacle = _radial_box_polygon(
            center, clock_deg,
            TERMINAL_CONTACT_RADIUS,
            TERMINAL_CONTACT_RADIUS + FASTON_RECEPTACLE_RADIAL_L,
            FASTON_RECEPTACLE_TANGENTIAL_W / 2.0, offset)
        boot = _radial_box_polygon(
            center, clock_deg,
            TERMINAL_CONTACT_RADIUS,
            TERMINAL_CONTACT_RADIUS + FASTON_BOOT_RADIAL_L,
            FASTON_BOOT_TANGENTIAL_W / 2.0, offset)
        ax.add_patch(plt.Polygon(
            tab, closed=True, fc="tab:orange", ec="tab:orange",
            alpha=0.48, lw=0.7, zorder=10))
        ax.add_patch(plt.Polygon(
            boot, closed=True, fill=False, ec=color,
            lw=1.4, zorder=11))
        ax.add_patch(plt.Polygon(
            receptacle, closed=True, fc=color, ec=color,
            alpha=0.24, lw=0.9, zorder=11))

        # Faint intermediate one-terminal states make the service motion
        # legible without implying that the opposite terminal also moves.
        for station_mm in FASTON_LEAD_PULL_STATES_MM[1:-1]:
            intermediate = np.asarray(
                v1lf_terminal_lead_points_for_terminal_pull(
                    terminal_id, station_mm)[f"terminal_lead_{terminal_id}"],
                dtype=float)
            ax.plot(intermediate[:, 0], intermediate[:, 1], color=color,
                    lw=FASTON_LEAD_D * 0.52, alpha=0.10,
                    solid_capstyle="round", zorder=9)

        pulled_lead = np.asarray(
            v1lf_terminal_lead_points_for_terminal_pull(
                terminal_id, FASTON_PULL_DISTANCE)[
                    f"terminal_lead_{terminal_id}"], dtype=float)
        ax.plot(pulled_lead[:, 0], pulled_lead[:, 1], color=color,
                lw=FASTON_LEAD_D * 0.74, alpha=0.82,
                ls=(0, (3.0, 1.5)), solid_capstyle="round", zorder=12)
        pulled_receptacle = _radial_box_polygon(
            center, clock_deg,
            TERMINAL_CONTACT_RADIUS + FASTON_PULL_DISTANCE,
            TERMINAL_CONTACT_RADIUS + FASTON_PULL_DISTANCE
            + FASTON_RECEPTACLE_RADIAL_L,
            FASTON_RECEPTACLE_TANGENTIAL_W / 2.0, offset)
        pulled_boot = _radial_box_polygon(
            center, clock_deg,
            TERMINAL_CONTACT_RADIUS + FASTON_PULL_DISTANCE,
            TERMINAL_CONTACT_RADIUS + FASTON_PULL_DISTANCE
            + FASTON_BOOT_RADIAL_L,
            FASTON_BOOT_TANGENTIAL_W / 2.0, offset)
        ax.add_patch(plt.Polygon(
            pulled_boot, closed=True, fill=False, ec=color,
            ls=(0, (3.0, 1.5)), lw=1.2, alpha=0.85, zorder=12))
        ax.add_patch(plt.Polygon(
            pulled_receptacle, closed=True, fill=False, ec=color,
            ls=":", lw=0.9, alpha=0.85, zorder=12))
        label_xy = pulled_boot[2] + (4.0 if terminal_id == 1 else -4.0) * tangent
        ax.text(label_xy[0], label_xy[1],
                f"T{terminal_id} pull {FASTON_PULL_DISTANCE:g}\n"
                f"T{2 if terminal_id == 1 else 1} installed",
                fontsize=6.4, color=color,
                ha="left" if terminal_id == 1 else "right",
                va="center", zorder=13)

    for name, lead in installed_leads.items():
        ax.plot(lead[:, 0], lead[:, 1], color=lead_colors[name],
                lw=FASTON_LEAD_D * 0.74, alpha=0.92,
                solid_capstyle="round", zorder=11)
        ax.plot(lead[-1, 0], lead[-1, 1], marker="s", ms=4.0,
                mfc="white", mec=lead_colors[name], zorder=13)

    # Installed positive-volume Y boot: OD8 incoming collar/tail plus two
    # OD4 branches.  Inner polarity lines retain the provisional D3.2 cable.
    free_s = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(
            np.diff(free_bundle, axis=0), axis=1))))
    boot_tail = _polyline_interval(
        free_bundle,
        max(0.0, free_s[-1] - FASTON_BREAKOUT_BUNDLE_OVERLAP_MM),
        free_s[-1])
    ax.plot(boot_tail[:, 0], boot_tail[:, 1], color="#5b4735",
            lw=FASTON_BREAKOUT_BUNDLE_OD, alpha=0.34,
            solid_capstyle="round", zorder=10)
    breakout = free_bundle[-1]
    ax.add_patch(plt.Circle(
        breakout[:2], FASTON_BREAKOUT_BUNDLE_OD / 2.0,
        fc="#5b4735", ec="#3b2d23", alpha=0.32, lw=0.8, zorder=10))
    for name, lead in installed_leads.items():
        branch = _polyline_interval(lead, 0.0, FASTON_BREAKOUT_LENGTH)
        ax.plot(branch[:, 0], branch[:, 1], color="#5b4735",
                lw=FASTON_BREAKOUT_LEAD_OD, alpha=0.34,
                solid_capstyle="round", zorder=10)
        ax.plot(branch[:, 0], branch[:, 1], color=lead_colors[name],
                lw=FASTON_LEAD_D * 0.72, alpha=0.88,
                solid_capstyle="round", zorder=11)

    tangent_bearing = math.degrees(math.atan2(
        UM_MOUTH_TANGENT[1], UM_MOUTH_TANGENT[0])) % 360.0
    ax.annotate(
        f"free R{facts['terminal_plan_bend_radius_mm']:.0f} cable crosses "
        f"D{2.0 * facts['um_terminal_reference_opening_radius_mm']:.0f} / "
        f"R{facts['um_terminal_reference_opening_radius_mm']:.0f} "
        "reference opening\n"
        f"continues to {clock_deg:g} deg terminal reference; "
        f"{tangent_bearing:.0f} deg tangent -> "
        f"free R{V1LF_TERMINATED_HANDOFF_R:.0f}; no grommet",
        route_handoff_xy, (126, 322), ha="right", fontsize=7.7,
        color="0.25", arrowprops=dict(arrowstyle="-", color="0.4"))
    ax.annotate(
        f"positive-volume Y: OD{FASTON_BREAKOUT_BUNDLE_OD:g} x "
        f"{FASTON_BREAKOUT_JUNCTION_LENGTH_MM:g} collar\n"
        f"two OD{FASTON_BREAKOUT_LEAD_OD:g} branches; provisional "
        f"D{FASTON_LEAD_D:g} / R{FASTON_LEAD_MIN_BEND_R:g}",
        breakout[:2], (126, 247), ha="right", fontsize=7.0,
        color="#5b4735", arrowprops=dict(arrowstyle="-", color="#5b4735"))
    ax.annotate(
        f"installed + independent {FASTON_PULL_DISTANCE:g} mm pull states\n"
        "both receptacles/boots shown; opposite Faston stays home\n"
        "proxy geometry — physical dry-fit remains mandatory",
        center + 67.0 * radial, (-128, 382), fontsize=7.2,
        color="tab:orange",
        arrowprops=dict(arrowstyle="-", color="tab:orange"))

    return {
        "free_bundle": free_bundle,
        "installed_leads": installed_leads,
        "lead_colors": lead_colors,
        "breakout": breakout,
        "handoff_radius_mm": V1LF_TERMINATED_HANDOFF_R,
        "lead_min_bend_radius_mm": FASTON_LEAD_MIN_BEND_R,
    }


def _draw_v1lf_bump_sections(
        parent, records, *, duct_specs, tunnel_skin,
        lm_pad_d, um_pad_d, lm_pilot_d, um_pilot_d,
        lm_seat_z, um_seat_z, stand_foot,
        insert_clear_d, insert_clear_z, shank_clear_d, shank_clear_z):
    """Draw two nominal diametric u-z sections from authoritative bump records.

    Unlike the neighboring station plot, each section cuts through the
    conduit axis and its crossed pilot axis. It shows nominal duct diameters
    and the exact vertical solid-backfill limits between the tube roof and
    blind-bore floor. The production loft uses circumscribed octagonal
    sections, so this deliberately does not claim to be an exact BREP slice.
    Floor mode overlays only the exact hardware-clearance cylinders removed
    by the CAD source.
    """
    records = tuple(records)

    def representative(route_name):
        candidates = [
            record for record in records
            if record["route"] == route_name
            and record["name"].startswith("lm_pilot_")
        ]
        if stand_foot:
            exception = [
                record for record in candidates
                if record["floor_hardware_clearance"]
            ]
            if exception:
                return exception[0]
        if candidates:
            return candidates[0]
        return next(record for record in records
                    if record["route"] == route_name)

    panel = parent.inset_axes((0.555, 0.505, 0.425, 0.465))
    panel.set_facecolor((1.0, 1.0, 1.0, 0.96))
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.7)
    panel.text(
        0.5, 0.965,
        "nominal local bump sections (u-z; exact Z limits)",
        ha="center", va="top", fontsize=6.8, weight="bold",
        color="#263746", transform=panel.transAxes)

    exceptions = []
    for index, route_name in enumerate(("UM", "T")):
        record = representative(route_name)
        duct_d, color = duct_specs[route_name]
        route_xyz = np.asarray(record["route_xyz"], dtype=float)
        pilot_xy = np.asarray(record["pilot_xy"], dtype=float)
        pilot_offset = float(np.linalg.norm(
            pilot_xy - route_xyz[:2]))
        is_lm = record["name"].startswith("lm_pilot_")
        support_r = (lm_pad_d if is_lm else um_pad_d) / 2.0
        pilot_d = lm_pilot_d if is_lm else um_pilot_d
        seat_z = lm_seat_z if is_lm else um_seat_z
        cutter_r = duct_d / 2.0
        outer_r = cutter_r + tunnel_skin
        bottom_z = float(record["bottom_z_mm"])
        bore_floor_z = float(record["bore_floor_z_mm"])

        section = panel.inset_axes((0.035 + 0.49 * index, 0.13,
                                    0.445, 0.715))
        section.set_facecolor("white")
        section.set_title(
            f"D{duct_d:g} {route_name}: {record['name']}",
            fontsize=5.8, pad=1.5, color=color)

        # The backfill is the vertical extrusion of the convex-hull saddle
        # through the route/pilot centre plane.  It overlaps the tube roof,
        # then remains solid all the way to the blind-bore floor.
        saddle_x0 = -outer_r
        saddle_x1 = pilot_offset + support_r
        section.add_patch(plt.Rectangle(
            (saddle_x0, bottom_z), saddle_x1 - saddle_x0,
            bore_floor_z - bottom_z,
            fc="#8799aa", ec="#405668", hatch="////", lw=0.65,
            alpha=0.72, zorder=2))
        section.add_patch(plt.Rectangle(
            (pilot_offset - support_r, bore_floor_z),
            2.0 * support_r, seat_z - bore_floor_z,
            fc="#aeb8c2", ec="#405668", lw=0.65, zorder=1))

        # Nominal diameter view of the circumscribed-octagon printed loft and
        # its duct clearance void. Drawing the annulus after the saddle keeps
        # the clearance void visually explicit while fused solids read as one.
        section.add_patch(plt.Circle(
            (0.0, route_xyz[2]), outer_r,
            fc="#8799aa", ec="#405668", lw=0.75, zorder=3))
        section.add_patch(plt.Circle(
            (0.0, route_xyz[2]), cutter_r,
            fc="white", ec=color, lw=1.0, zorder=5))

        # The crossed driver pilot is blind from the carrier face and ends
        # exactly at the record's authoritative bore-floor Z.
        section.add_patch(plt.Rectangle(
            (pilot_offset - pilot_d / 2.0, bore_floor_z),
            pilot_d, seat_z - bore_floor_z + 0.02,
            fc="white", ec="0.35", lw=0.65, zorder=4))

        if record["floor_hardware_clearance"]:
            exceptions.append(record["name"])
            for diameter, (z0, z1), hatch in (
                    (insert_clear_d, insert_clear_z, "xx"),
                    (shank_clear_d, shank_clear_z, "++")):
                section.add_patch(plt.Rectangle(
                    (pilot_offset - diameter / 2.0, z0),
                    diameter, z1 - z0,
                    fc="white", ec="tab:orange", hatch=hatch,
                    lw=0.75, alpha=0.92, zorder=6))

        x_margin = 0.8
        hardware_r = max(insert_clear_d, shank_clear_d) / 2.0
        x0 = min(-outer_r, pilot_offset - hardware_r) - x_margin
        x1 = max(saddle_x1, pilot_offset + hardware_r) + x_margin
        z0 = min(route_xyz[2] - outer_r,
                 insert_clear_z[0] if record["floor_hardware_clearance"]
                 else route_xyz[2] - outer_r) - 0.55
        z1 = max(seat_z, shank_clear_z[1]
                 if record["floor_hardware_clearance"] else seat_z) + 0.45
        section.set_xlim(x0, x1)
        section.set_ylim(z0, z1)
        section.set_aspect("equal", adjustable="box")
        section.tick_params(labelsize=4.2, pad=1.0, length=2.0)
        section.set_xlabel("local u (mm)", fontsize=4.8, labelpad=0.5)
        if index == 0:
            section.set_ylabel("z (mm)", fontsize=4.8,
                               labelpad=0.5)
        section.grid(True, lw=0.18, alpha=0.26)
        section.annotate(
            "duct clearance void", (0.0, route_xyz[2]),
            (x0 + 0.2, z0 + 0.3), fontsize=4.5, color=color,
            arrowprops=dict(arrowstyle="-", lw=0.45, color=color))
        section.annotate(
            f"{tunnel_skin:g} skin",
            (outer_r * 0.70,
             route_xyz[2] + outer_r * 0.70),
            (x0 + 0.2, z1 - 0.7), fontsize=4.5, color="#405668",
            arrowprops=dict(arrowstyle="-", lw=0.45,
                            color="#405668"))
        section.text(
            0.98, 0.03,
            f"solid roof -> bore floor\n"
            f"z {bottom_z:.2f}..{bore_floor_z:.2f}; "
            f"tube overlap {record['tube_overlap_mm']:.2f}",
            ha="right", va="bottom", fontsize=4.35,
            color="#263746", transform=section.transAxes,
            bbox=dict(fc="white", ec="none", alpha=0.78, pad=0.7))

    if exceptions:
        panel.text(
            0.5, 0.035,
            "orange hatch = exact floor-only hardware void: "
            f"D{insert_clear_d:g} z={insert_clear_z[0]:g}.."
            f"{insert_clear_z[1]:g}; D{shank_clear_d:g} "
            f"z={shank_clear_z[0]:g}..{shank_clear_z[1]:g}; "
            "surrounding backfill remains solid",
            ha="center", va="bottom", fontsize=4.65,
            color="#a94f09", transform=panel.transAxes)
    else:
        panel.text(
            0.5, 0.035,
            "no floor-support exception in this state: roof-to-bore "
            "backfill is continuous solid material",
            ha="center", va="bottom", fontsize=4.8,
            color="#263746", transform=panel.transAxes)


def render_v1lf():
    """Dedicated integral-routing sheet for the two-collar V1LF core."""
    from matplotlib.lines import Line2D
    from top_baffle_nd25fw4 import L22_CUTOUT
    from top_baffle_nd25fw4_flush import (
        LM_BORE_DEPTH_MM, LM_PILOT_XY, LM_RECESS_R, LM_SEAT_Z,
        PAD_D_MM, PAD_FACE_Z, UM_PAD_D_MM, UM_PILOT_XY, UM_RECESS_R,
        UM_SEAT_Z)
    from top_baffle_nd25fw4_v1lf import (
        CORE_REAR_Z, JOINT_EAR_X, JOINT_EAR_Y, LM_CORE_R,
        SIDE_MAGNET_DEPTH, SIDE_MAGNET_POCKET_D,
        STRUCT_MOUNT_INSERT_D,
        TWEETER_JOINT_HOLE_D, TWEETER_JOINT_X, TWEETER_JOINT_Y,
        UM_CORE_R, joint_ear_polygon, side_magnet_sites,
        tweeter_joint_polygon)
    from top_baffle_nd25fw4_v1lf_attachments import support_plan_geometry
    from top_baffle_nd25fw4_um_fit import (
        FASTON_BREAKOUT_LENGTH,
        w22_body_reference_facts,
    )
    from top_baffle_nd25fw4_v1lf_route import (
        CABLE_D_EST, DUCT_D, LM_CABLE_D_EST, MAIN_COVERED_BUMPS,
        FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_D,
        FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_Z,
        FLOOR_SUPPORT_BACKFILL_SHANK_CLEAR_D,
        FLOOR_SUPPORT_BACKFILL_SHANK_CLEAR_Z,
        LM_ARC_LENGTH, LM_ENTRY_LENGTH,
        LM_LEAD_ANGLE_DEG, LM_LEAD_LENGTH, ROUTE_LENGTH, SIDE_WALL,
        TRENCH_CENTER_Z,
        TS_ADDON_SUPPORT_MIN_Y, TS_CABLE_D_EST, TS_DUCT_D, T_COVERED_BUMPS,
        TS_ROUTE_LENGTH, TS_SIDE_WALL,
        TS_TWEETER_FLUSH_R, TS_UM_CORE_COVER_END_S, TS_UM_ENTRY_S,
        TUNNEL_FLOOR_SKIN, lm_cable_points,
        route_cable_points, route_facts, ts_cable_points)
    from gen_driver_overlay import outline_polygon
    from shapely.geometry import Point, Polygon, box

    fig = plt.figure(figsize=(14.0, 9.6), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.02, 0.98),
                          height_ratios=(0.77, 0.23),
                          wspace=0.18, hspace=0.30)
    ax = fig.add_subplot(gs[:, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_lm_side = fig.add_subplot(gs[1, 1])

    # State-specific support. Floor mode draws its required separate bolted
    # support; no-floor draws the mandatory shallow, front-flush solid web.
    support = support_plan_geometry(STAND_FOOT)
    support_color = "#65778a"
    if STAND_FOOT:
        yoke = np.asarray(support["yoke"], dtype=float)
        mounts = np.asarray(support["structural_mounts"], dtype=float)
        ax.plot(yoke[:, 0], yoke[:, 1], color=support_color, lw=9.0,
                alpha=0.30, solid_capstyle="round", zorder=0)
        for mount in mounts:
            target = yoke[np.argmin(np.linalg.norm(yoke - mount, axis=1))]
            ax.plot([mount[0], target[0]], [mount[1], target[1]],
                    color=support_color, lw=8.0, alpha=0.28,
                    solid_capstyle="round", zorder=0)
        rails = np.asarray(support["floor_rails"], dtype=float)
        for rail in rails:
            target = yoke[np.argmin(np.linalg.norm(yoke - rail, axis=1))]
            ax.plot([target[0], rail[0]], [target[1], rail[1]],
                    color=support_color, lw=7.0, alpha=0.28,
                    solid_capstyle="round", zorder=0)
        ax.plot(rails[:, 0], rails[:, 1], color=support_color, lw=9.0,
                alpha=0.30, solid_capstyle="round", zorder=0)
        support_name = "required floor/NL8 support add-on"
    else:
        bridge = np.asarray(support["bridge_holes"], dtype=float)
        bridge_facts = support["fused_bridge"]
        face = np.asarray(bridge_facts["face_outline"], dtype=float)
        yoke = face
        mounts = np.empty((0, 2), dtype=float)
        ax.fill(face[:, 0], face[:, 1], fc=support_color,
                ec="#31485f", lw=1.15, alpha=0.34, zorder=0)
        ax.plot(bridge[:, 0], bridge[:, 1], ls="none", marker="o",
                ms=5.5, mfc="white", mec="#31485f", mew=1.0,
                zorder=1)
        support_name = (
            "fused front-flush solid bridge web "
            "(carrier envelope z=5.3..18.3; no rear X/keel)")

    for mx, my in mounts:
        ax.add_patch(plt.Circle(
            (mx, my), STRUCT_MOUNT_INSERT_D / 2.0, fc="white",
            ec="#31485f", lw=1.2, zorder=8))

    # Mandatory core: two real annuli plus the paired lap tongues.
    for (cx, cy, cut_d), outer, recess, color in (
            (L22_CUTOUT, LM_CORE_R, LM_RECESS_R, "#b8bec7"),
            (UM_CUTOUT, UM_CORE_R, UM_RECESS_R, "#c7cdd5")):
        ax.add_patch(plt.Circle((cx, cy), outer, fc=color, ec="0.22",
                                lw=1.2, zorder=1))
        ax.add_patch(plt.Circle((cx, cy), recess, fill=False, ec="0.45",
                                ls="--", lw=0.9, zorder=2))
        ax.add_patch(plt.Circle((cx, cy), cut_d / 2.0, fc="white",
                                ec="0.25", lw=1.0, zorder=3))
    # Dimensioned terminal-less MU body reference: the D60 motor is the
    # governing obstruction that invalidated the former inward R14 cable.
    ax.add_patch(plt.Circle(
        UM_CUTOUT[:2], 40.0, fill=False, ec="#9b4d4d",
        ls=":", lw=0.9, alpha=0.75, zorder=4))
    ax.add_patch(plt.Circle(
        UM_CUTOUT[:2], 30.0, fc="#d98c8c", ec="#9b4d4d",
        lw=0.8, alpha=0.10, zorder=4))
    ax.annotate(
        "known terminal-less body\nD80 frame / D60 motor\n"
        "tabs still require physical dry-fit",
        (30.0, UM_CUTOUT[1]), (72, 395), fontsize=7.2,
        color="#9b4d4d",
        arrowprops=dict(arrowstyle="-", color="#9b4d4d"))

    # Conservative LM-driver obstruction and explicit source transform.
    # The render consumes the hash-pinned cached facts; it does not import
    # the W22 STEP or invoke OCC.
    w22_facts = w22_body_reference_facts()
    w22_keepout_r = max(
        step[0] for step in w22_facts["conservative_steps"])
    ax.add_patch(plt.Circle(
        L22_CUTOUT[:2], w22_keepout_r,
        fc=(0.84, 0.42, 0.37, 0.055), ec="#9a4037",
        lw=1.0, ls=(0, (5.0, 2.5)), zorder=3))
    ax.annotate(
        f"conservative W22 rear keepout D{2.0 * w22_keepout_r:g}\n"
        f"{w22_facts['provenance']['source_name']} "
        f"sha256 {w22_facts['source_step_sha256'][:10]}...\n"
        f"native +Y -> world +Z; Rot X="
        f"{w22_facts['native_to_world']['rotation']['degrees']:+g} deg; "
        f"front z={w22_facts['world_front_datum_z_mm']:g}",
        (-w22_keepout_r, L22_CUTOUT[1]), (126, 150),
        ha="right", fontsize=6.8, color="#9a4037",
        arrowprops=dict(arrowstyle="-", color="#9a4037"))
    # LM insert pads/pilots are explicit so the covered Z-bypasses and
    # insert-free crown can be read against the rotated R6F pattern.
    for i, (px, py) in enumerate(LM_PILOT_XY):
        crossed = i in (0, 1, 2, 3, 4, 5)
        edge = "tab:orange" if crossed else "0.50"
        ax.add_patch(plt.Circle((px, py), PAD_D_MM / 2.0,
                                fc="0.82", ec=edge, lw=1.0,
                                zorder=4))
        ax.add_patch(plt.Circle((px, py), L22_PILOT_D_MM / 2.0,
                                fc="white", ec=edge, lw=0.8,
                                zorder=5))
    for i, (px, py) in enumerate(UM_PILOT_XY):
        between_terminals = i in (2, 3)  # screws at 238/328 deg
        edge = "tab:orange" if between_terminals else "0.50"
        ax.add_patch(plt.Circle((px, py), UM_PILOT_D_MM / 2.0,
                                fc="white", ec=edge, lw=1.0,
                                zorder=5))

    for x in JOINT_EAR_X:
        for owner, color in (("lm", "#858d98"), ("um", "#a0a7b1")):
            ear = joint_ear_polygon(owner, x)
            ex, ey = ear.exterior.xy
            ax.fill(ex, ey, fc=color, ec="0.25", lw=0.8, zorder=2)
        ax.plot(x, JOINT_EAR_Y, marker="o", ms=4, mfc="white",
                mec="0.25", zorder=6)

    # Optional tweeter crescent on the current compact direct M3
    # half-laps.  The convex-hull polygons visibly include each minimal
    # ring-to-boss knee and rounded ear; no legacy long magnet knee exists.
    outline = Polygon(outline_polygon(OUTLINE_B2, samples=96))
    crescent = (outline.intersection(box(
                    -75.0, TS_ADDON_SUPPORT_MIN_Y, 75.0, 454.0))
                .difference(Point(*UM_CUTOUT[:2]).buffer(
                    TS_TWEETER_FLUSH_R, resolution=64)))
    for poly in ([crescent] if crescent.geom_type == "Polygon"
                 else crescent.geoms):
        xpoly, ypoly = poly.exterior.xy
        ax.fill(xpoly, ypoly, fc="#c9b6dc", ec="#76538f", lw=1.0,
                alpha=0.55, zorder=2)
    for x in TWEETER_JOINT_X:
        ear = tweeter_joint_polygon(x)
        ex, ey = ear.exterior.xy
        ax.fill(ex, ey, fc="#a88ac2", ec="#76538f", lw=1.0,
                alpha=0.75, zorder=4)
        ax.add_patch(plt.Circle(
            (x, TWEETER_JOINT_Y), TWEETER_JOINT_HOLE_D / 2.0,
            fc="white", ec="#76538f", lw=0.9, zorder=8))

    # Four LM and two UM radial magnets are all source-owned.  In particular,
    # retain both upper LM sites while the lower pair provides the new total;
    # the sheet derives its counts instead of baking them into annotations.
    magnet_sites = tuple(side_magnet_sites())
    magnet_counts = {
        driver: sum(site["driver"] == driver for site in magnet_sites)
        for driver in ("lm", "um")
    }
    if (magnet_counts != {"lm": 4, "um": 2}
            or len(magnet_sites) != sum(magnet_counts.values())
            or any(not site["flush_buried"]
                   or site["proud_ear_added"] for site in magnet_sites)
            or not math.isclose(SIDE_MAGNET_POCKET_D, 5.2)
            or not math.isclose(SIDE_MAGNET_DEPTH, 2.2)):
        raise RuntimeError(
            "V1LF routing sheet requires LM=4/UM=2 flush-buried sites "
            "with D5.2 x 2.2 pockets")
    magnet_count_text = (
        f"LM={magnet_counts['lm']} / UM={magnet_counts['um']} / "
        f"total={len(magnet_sites)}")
    for site in magnet_sites:
        nx, ny = site["normal"]
        face = np.asarray(site["face"])
        magnet_inner = face - SIDE_MAGNET_DEPTH * np.asarray((nx, ny))
        ax.plot([magnet_inner[0], face[0]], [magnet_inner[1], face[1]],
                color="tab:orange", lw=SIDE_MAGNET_POCKET_D,
                solid_capstyle="butt", zorder=7)
        if site["flush_buried"]:
            ax.plot(face[0], face[1], marker="|", ms=6.0,
                    color="#8a4b08", zorder=8)

    pts = np.array(route_cable_points(spacing_mm=0.45))
    lm_pts = np.array(lm_cable_points(spacing_mm=0.35))
    ts_pts = np.array(ts_cable_points(spacing_mm=0.45))
    facts = route_facts()
    station = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    ts_station = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(ts_pts, axis=0), axis=1))))

    def _span_mask(condition, *, start_at_zero=False):
        """Fill the one continuous sampled owner interval in condition."""
        indices = np.flatnonzero(condition)
        if not len(indices):
            raise RuntimeError("routing-sheet owner mask has no samples")
        result = np.zeros(len(condition), dtype=bool)
        first = 0 if start_at_zero else int(indices[0])
        result[first:int(indices[-1]) + 1] = True
        return result

    def _masked(values, mask):
        return np.ma.masked_where(~mask, values)

    def _radial_crossing(points, center, radius, crossing=-1):
        """Interpolate one exact circle crossing of a dense route polyline."""
        center = np.asarray(center, dtype=float)
        radii = np.linalg.norm(points[:, :2] - center, axis=1)
        candidates = np.flatnonzero(
            (radii[:-1] - radius) * (radii[1:] - radius) <= 0.0)
        if not len(candidates):
            raise RuntimeError(f"route does not cross R{radius:g}")
        index = int(candidates[crossing])
        relative = points[index, :2] - center
        delta = points[index + 1, :2] - points[index, :2]
        aa = float(np.dot(delta, delta))
        bb = 2.0 * float(np.dot(relative, delta))
        cc = float(np.dot(relative, relative) - radius ** 2)
        roots = np.roots((aa, bb, cc))
        valid = [float(root.real) for root in roots
                 if abs(float(root.imag)) <= 1e-8
                 and -1e-8 <= float(root.real) <= 1.0 + 1e-8]
        if not valid:
            # Dense centerlines make linear radius interpolation a safe
            # rendering fallback without changing any design coordinate.
            denominator = radii[index] - radii[index + 1]
            q = ((radii[index] - radius) / denominator
                 if abs(denominator) > 1e-12 else 0.5)
        else:
            q = min(valid, key=lambda value: abs(value - 0.5))
        return points[index] + np.clip(q, 0.0, 1.0) * (
            points[index + 1] - points[index])

    lm_center = np.asarray(L22_CUTOUT[:2], dtype=float)
    um_center = np.asarray(UM_CUTOUT[:2], dtype=float)
    main_lm_r = np.linalg.norm(pts[:, :2] - lm_center, axis=1)
    ts_lm_r = np.linalg.norm(ts_pts[:, :2] - lm_center, axis=1)
    ts_um_r = np.linalg.norm(ts_pts[:, :2] - um_center, axis=1)

    # Main UM bundle: only the LM carrier owns a printed cover. No-floor
    # starts in the bridge's rear-face lumen; floor starts at the first
    # legacy R113 ring mouth. Cable is free from the upper LM R113 mouth,
    # behind the complete UM carrier, through the D82 reference and onward.
    main_start_candidates = np.flatnonzero(
        main_lm_r <= facts["t_lower_lm_flush_radius_mm"] + 1e-6)
    if not len(main_start_candidates):
        raise RuntimeError("main route never enters the LM R113 carrier")
    main_printed = _span_mask(
        main_lm_r <= facts["t_lower_lm_flush_radius_mm"] + 1e-6,
        start_at_zero=not STAND_FOOT)
    main_free = ~main_printed
    main_lm_end_xyz = _radial_crossing(
        pts, lm_center, facts["t_lower_lm_flush_radius_mm"], -1)
    terminal_route_handoff = _radial_crossing(
        pts, um_center, facts["um_terminal_reference_opening_radius_mm"], -1)

    # T bundle has two printed owners only: LM and UM. Their native radial
    # crops are plain butt mouths; cable alone crosses the LM/UM ownership
    # gap and remains free after leaving UM, including behind the tweeter.
    # In no-floor mode the first span starts in the fused bridge web.
    t_lm_printed = _span_mask(
        (ts_station <= TS_UM_ENTRY_S + 1e-6)
        & (ts_lm_r <= facts["t_lower_lm_flush_radius_mm"] + 1e-6),
        start_at_zero=not STAND_FOOT)
    t_um_printed = _span_mask(
        (ts_station <= TS_UM_CORE_COVER_END_S + 1e-6)
        & (ts_um_r <= facts["t_upper_um_flush_radius_mm"] + 1e-6))
    t_printed = t_lm_printed | t_um_printed
    t_free = ~t_printed

    t_lm_end_xyz = _radial_crossing(
        ts_pts, lm_center, facts["t_lower_lm_flush_radius_mm"], -1)
    t_um_start_xyz = _radial_crossing(
        ts_pts, um_center, facts["t_lower_um_flush_radius_mm"], 0)
    t_um_end_xyz = _radial_crossing(
        ts_pts, um_center, facts["t_upper_um_flush_radius_mm"], -1)
    t_lm_end = t_lm_end_xyz[:2]
    t_um_start = t_um_start_xyz[:2]
    t_um_end = t_um_end_xyz[:2]
    t_lower_gap_mid = (t_lm_end + t_um_start) / 2.0
    t_lm_end_s = ts_station[np.argmin(
        np.linalg.norm(ts_pts - t_lm_end_xyz, axis=1))]
    t_um_start_s = ts_station[np.argmin(
        np.linalg.norm(ts_pts - t_um_start_xyz, axis=1))]
    t_um_end_s = ts_station[np.argmin(
        np.linalg.norm(ts_pts - t_um_end_xyz, axis=1))]
    t_lower_gap_station = float(np.mean((t_lm_end_s, t_um_start_s)))
    t_upper_free_station = float(np.mean((t_um_end_s, ts_station[-1])))
    t_upper_free_mid = np.asarray([
        np.interp(t_upper_free_station, ts_station, ts_pts[:, axis])
        for axis in range(3)
    ])

    # Dashed paths plus pale halos mean buried printed envelopes.  Solid
    # colored paths with a white keyline mean cable only. Butt caps keep
    # every owner boundary visibly flush instead of suggesting a horn.
    buried_dash = (0, (4.2, 2.4))
    ax.plot(_masked(pts[:, 0], main_printed),
            _masked(pts[:, 1], main_printed), color="0.30",
            lw=DUCT_D + 2.0 * SIDE_WALL, alpha=0.18,
            ls=buried_dash, dash_capstyle="butt", zorder=5)
    ax.plot(_masked(pts[:, 0], main_printed),
            _masked(pts[:, 1], main_printed), color=STYLE["um"][0],
            lw=DUCT_D, alpha=0.72, ls=buried_dash,
            dash_capstyle="butt", zorder=6)
    ax.plot(_masked(pts[:, 0], main_free),
            _masked(pts[:, 1], main_free), color="white",
            lw=CABLE_D_EST + 1.2, solid_capstyle="butt", zorder=6)
    ax.plot(_masked(pts[:, 0], main_free),
            _masked(pts[:, 1], main_free), color=STYLE["um"][0],
            lw=CABLE_D_EST, solid_capstyle="butt", zorder=7)
    ax.plot(lm_pts[:, 0], lm_pts[:, 1], color=STYLE["lm"][0],
            lw=LM_CABLE_D_EST, alpha=0.72, ls="-",
            solid_capstyle="round", zorder=6)
    ax.plot(_masked(ts_pts[:, 0], t_printed),
            _masked(ts_pts[:, 1], t_printed), color="0.30",
            lw=TS_DUCT_D + 2.0 * TS_SIDE_WALL, alpha=0.18,
            ls=buried_dash, dash_capstyle="butt", zorder=5)
    ax.plot(_masked(ts_pts[:, 0], t_printed),
            _masked(ts_pts[:, 1], t_printed), color=STYLE["ts"][0],
            lw=TS_DUCT_D, alpha=0.80, ls=buried_dash,
            dash_capstyle="butt", zorder=7)
    ax.plot(_masked(ts_pts[:, 0], t_free),
            _masked(ts_pts[:, 1], t_free), color="white",
            lw=TS_CABLE_D_EST + 1.2, solid_capstyle="butt", zorder=7)
    ax.plot(_masked(ts_pts[:, 0], t_free),
            _masked(ts_pts[:, 1], t_free), color=STYLE["ts"][0],
            lw=TS_CABLE_D_EST, alpha=0.96,
            solid_capstyle="butt", zorder=8)
    # Draw both free handoffs explicitly. The lower gap can fall between
    # 0.45-mm samples, while the upper interval now stays free all the way
    # from the UM butt mouth through the tweeter region.
    for start, stop, start_s, stop_s in (
            (t_lm_end_xyz, t_um_start_xyz, t_lm_end_s, t_um_start_s),
            (t_um_end_xyz, ts_pts[-1], t_um_end_s, ts_station[-1])):
        gap = _polyline_interval(ts_pts, start_s, stop_s)
        gap[0] = start
        gap[-1] = stop
        ax.plot(gap[:, 0], gap[:, 1], color="white",
                lw=TS_CABLE_D_EST + 1.2, solid_capstyle="butt", zorder=8)
        ax.plot(gap[:, 0], gap[:, 1], color=STYLE["ts"][0],
                lw=TS_CABLE_D_EST, solid_capstyle="butt", zorder=9)

    ax.plot(pts[0, 0], pts[0, 1], marker="s", ms=7, mfc="white",
            mec=STYLE["um"][0], zorder=8)
    ax.plot(lm_pts[[0, -1], 0], lm_pts[[0, -1], 1], ls="none",
            marker="o", ms=6, mfc="white", mec=STYLE["lm"][0],
            zorder=8)
    ax.plot(ts_pts[0, 0], ts_pts[0, 1], marker="s", ms=6,
            mfc="white", mec=STYLE["ts"][0], zorder=9)
    for mouth in (t_lm_end, t_um_start, t_um_end, ts_pts[-1, :2]):
        ax.plot(*mouth, marker="s", ms=4.8, mfc="white",
                mec=STYLE["ts"][0], mew=0.9, zorder=10)
    service = _draw_v1lf_terminal_service(
        ax, facts, terminal_route_handoff[:2], CABLE_D_EST)
    free_bundle = service["free_bundle"]
    terminal_leads = service["installed_leads"]
    lead_colors = service["lead_colors"]
    cross = np.asarray(facts["crossover_xy"], dtype=float)
    ax.plot(*cross, marker="x", ms=7, mew=1.5, color="#31485f",
            zorder=10)
    ax.annotate(
        f"crown crossover {facts['crossover_angle_deg']:.1f} deg\n"
        f"T upper z={facts['crossover_t_z_mm']:.2f}; "
        f"UM lower z={facts['crossover_main_z_mm']:.2f}\n"
        "UM cable is free here; gap to printed T cover="
        f"{facts['crossover_free_um_to_t_cover_gap_mm']:.2f} mm",
        cross, (-128, 335), fontsize=7.6, color="#31485f",
        arrowprops=dict(arrowstyle="-", color="#31485f"))
    jump = pts[np.argmin(np.abs(
        station - MAIN_COVERED_BUMPS[0].station)), :2]
    ax.annotate("all insert bypasses are fully covered\n"
                "full-width solid-webbed Z bumps; no rear cable windows",
                jump, (126, 198), ha="right", fontsize=8,
                color="#31485f",
                arrowprops=dict(arrowstyle="-", color="#31485f"))
    ax.annotate(f"LM lead: short free D{LM_CABLE_D_EST:.1f} cable\n"
                f"floating behind carrier at {LM_LEAD_ANGLE_DEG:.0f} deg; "
                "no micro-duct",
                lm_pts[len(lm_pts) // 2, :2], (-128, 101), fontsize=8,
                color=STYLE["lm"][0],
                arrowprops=dict(arrowstyle="-", color=STYLE["lm"][0]))
    ax.annotate(
        f"UM cable: printed D{DUCT_D:.1f} cover only in LM to R"
        f"{facts['t_lower_lm_flush_radius_mm']:.0f}\n"
        "then free behind UM to the 283 deg terminals; no UM rear duct",
        main_lm_end_xyz[:2], (126, 353), ha="right", fontsize=7.2,
        color="#1b5741",
        arrowprops=dict(arrowstyle="-", color="#1b5741"))
    ax.annotate(f"two-carrier mandatory core\nrounded bolted joint ears; "
                "LM may use optional hidden-key subdivision; "
                f"magnets {magnet_count_text}\n"
                f"all flush in D{SIDE_MAGNET_POCKET_D:.1f} x "
                f"{SIDE_MAGNET_DEPTH:.1f} pockets — no proud ears",
                (-85, 270), (-126, 248), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.45"))
    ax.annotate("optional tweeter piece + two minimal direct knees\n"
                "rounded M3 half-lap ears at x = +/-24, y = 421.5",
                (-24, 421.5), (-132, 438), fontsize=7.6, color="#76538f",
                arrowprops=dict(arrowstyle="-", color="#76538f"))
    ax.annotate("T cable remains free behind the optional tweeter\n"
                "no printed crescent conduit and no lower point horn",
                ts_pts[-1, :2], (72, 438), fontsize=7.6,
                color="#76538f",
                arrowprops=dict(arrowstyle="-", color="#76538f"))
    ax.annotate(
        f"T: LM R{facts['t_lower_lm_flush_radius_mm']:.0f} butt mouth -> "
        "free cable ->\n"
        f"UM R{facts['t_lower_um_flush_radius_mm']:.1f} butt mouth; "
        "no connecting horn",
        t_lower_gap_mid, (-126, 307), fontsize=7.2,
        color="#8a6510",
        arrowprops=dict(arrowstyle="-", color="#8a6510"))
    ax.annotate(
        f"upper T: printed through UM to R"
        f"{facts['t_upper_um_flush_radius_mm']:.1f} butt mouth\n"
        "then free behind tweeter; no tweeter-side printed cover",
        t_upper_free_mid[:2], (126, 414), ha="right", fontsize=7.2,
        color="#8a6510",
        arrowprops=dict(arrowstyle="-", color="#8a6510"))
    feed_points = np.asarray(facts["functional_lm_feed_points"], dtype=float)
    feed_mid = feed_points[:, :2].mean(axis=0)
    if STAND_FOOT:
        ax.annotate(
            "floor state: legacy lateral ring feeds retained\n"
            f"solid cable reaches the flush R"
            f"{facts['t_lower_lm_flush_radius_mm']:.0f} entry mouths",
            feed_mid, (-126, 66), fontsize=7.4, color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))
    else:
        feed_xy = np.asarray(facts["no_floor_bridge_feed_xy"], dtype=float)
        feed_z = facts["no_floor_bridge_feed_rear_z_mm"]
        ax.annotate(
            "no-floor rear bridge mouths (back face)\n"
            f"UM/T x={feed_xy[0, 0]:+g}/{feed_xy[1, 0]:+g}, "
            f"y={feed_xy[0, 1]:g}, z={feed_z:g}; shallow Z rises",
            feed_xy.mean(axis=0), (-126, 66), fontsize=7.4,
            color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))
    ax.annotate(f"{support_name}\nplan geometry from the CAD builder",
                yoke[len(yoke) // 2], (-132, 38), fontsize=7.6,
                color=support_color,
                arrowprops=dict(arrowstyle="-", color=support_color))
    if STAND_FOOT:
        ax.annotate("3 lower W22 axes: long driver screws\n"
                    "engage rear-installed support heat-sets",
                    mounts[1], (72, 50), fontsize=7.6, color="#31485f",
                    arrowprops=dict(arrowstyle="-", color="#31485f"))
    else:
        web_z = bridge_facts["web_z"]
        ax.annotate(
            "four rear-entry blind bridge inserts unchanged\n"
            f"40 x 50 pattern; solid web z={web_z[0]:g}..{web_z[1]:g}\n"
            f"soft cubic shoulders blend into R"
            f"{facts['t_lower_lm_flush_radius_mm']:.0f}; magnets carry 0 N",
                    bridge[2], (72, 50), fontsize=7.6, color="#31485f",
                    arrowprops=dict(arrowstyle="-", color="#31485f"))
    ax.set_aspect("equal")
    ax.set_xlim(-138, 138); ax.set_ylim(-5, 462)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.grid(True, lw=0.3, alpha=0.35)
    state_title = ("FLOOR: required bolted NL8 support add-on"
                   if STAND_FOOT else
                   "NO-FLOOR: soft-blend solid web + central rear feeds")
    ax.set_title(
        f"plan: printed envelopes dashed; free cable spans solid — "
        f"{state_title}", fontsize=10.5)

    legend = [
        Line2D([0], [0], color=STYLE["um"][0], lw=4, ls="--",
               label=(f"UM cable: printed D{DUCT_D:.1f} cover only in LM; "
                      "free behind UM to terminals")),
        Line2D([0], [0], color=STYLE["lm"][0], lw=4, ls="-",
               label=(f"LM: short free D{LM_CABLE_D_EST:.1f} lead; "
                      "no printed micro-duct")),
        Line2D([0], [0], color=STYLE["ts"][0], lw=4, ls="--",
               label=(f"T: printed D{TS_DUCT_D:.1f} cover only in LM/UM; "
                      "all ends flush/butt")),
        Line2D([0], [0], color=STYLE["ts"][0], lw=4, ls="-",
               label=(f"T: free D{TS_CABLE_D_EST:.1f} cable across the "
                      "LM/UM handoff and behind tweeter")),
        Line2D([0], [0], color="tab:orange", lw=5,
               label=(f"magnets {magnet_count_text}: D5x2 in "
                      f"D{SIDE_MAGNET_POCKET_D:.1f} x "
                      f"{SIDE_MAGNET_DEPTH:.1f} pockets; all flush, "
                      "do not bottom; alignment only")),
        Line2D([0], [0], color="#31485f", lw=4, ls="--",
               label=("full-width solid-webbed insert-bypass Z bumps; "
                      "zero open windows")),
        Line2D([0], [0], color=support_color, lw=6, alpha=0.45,
               label=support_name),
        Line2D([0], [0], color="tab:blue", lw=2.4,
               label="provisional terminal lead 1: D3.2 / R8-min"),
        Line2D([0], [0], color="tab:red", lw=2.4,
               label="provisional terminal lead 2: D3.2 / R8-min"),
    ]
    # Keep the legend outside the plan axes: the previous lower-right
    # placement hid the support and load-path callouts that this sheet is
    # specifically intended to communicate.
    fig.legend(handles=legend, loc="lower left",
               bbox_to_anchor=(0.055, 0.012), ncol=2,
               fontsize=7.2, framealpha=0.92)

    # Longitudinal side view along independent route stations. Pale hatched
    # bands exist only where a carrier owns printed material. Solid colored
    # segments are cable-only handoffs; local bypass skins remain continuous
    # inside every owned interval.
    cutter_r = DUCT_D / 2.0
    ax_side.axhspan(-14.5, CORE_REAR_Z, color="#e9f4fb", zorder=0)
    ax_side.axhspan(TRENCH_CENTER_Z + cutter_r, LM_SEAT_Z,
                    color="0.78", alpha=0.70, zorder=1)
    um_bottom = pts[:, 2] - cutter_r
    ax_side.fill_between(
        station, um_bottom - TUNNEL_FLOOR_SKIN, um_bottom,
        where=main_printed, interpolate=True, fc="#71869b", ec="#4d6277",
        hatch="////", alpha=0.26, zorder=2,
        label=f"printed conformal floor = {TUNNEL_FLOOR_SKIN:.1f} mm")
    ax_side.fill_between(station, pts[:, 2] - cutter_r,
                         pts[:, 2] + cutter_r, where=main_printed,
                         interpolate=True, color=STYLE["um"][0],
                         alpha=0.20, zorder=3,
                         label=f"D{DUCT_D:.1f} clearance envelope")
    ax_side.plot(_masked(station, main_printed),
                 _masked(pts[:, 2], main_printed),
                 color=STYLE["um"][0], lw=2.3, ls=buried_dash,
                 dash_capstyle="butt", zorder=5,
                 label=(f"UM-cable D{DUCT_D:.1f} printed LM span"))
    ax_side.plot(_masked(station, main_free),
                 _masked(pts[:, 2], main_free), color="white",
                 lw=3.8, solid_capstyle="butt", zorder=5)
    ax_side.plot(_masked(station, main_free),
                 _masked(pts[:, 2], main_free),
                 color=STYLE["um"][0], lw=2.5,
                 solid_capstyle="butt", zorder=6,
                 label=f"UM D{CABLE_D_EST:.1f} free behind UM")
    ts_r = TS_DUCT_D / 2.0
    ts_bottom = ts_pts[:, 2] - ts_r
    ax_side.fill_between(
        ts_station, ts_bottom - TUNNEL_FLOOR_SKIN, ts_bottom,
        where=t_printed, interpolate=True, fc="#9a7bb4", ec="#76538f",
        hatch="\\\\", alpha=0.20, zorder=2)
    ax_side.fill_between(ts_station, ts_pts[:, 2] - ts_r,
                         ts_pts[:, 2] + ts_r, where=t_printed,
                         interpolate=True, color=STYLE["ts"][0],
                         alpha=0.16, zorder=2,
                         label=f"D{TS_DUCT_D:.1f} T printed LM/UM cover")
    ax_side.plot(_masked(ts_station, t_printed),
                 _masked(ts_pts[:, 2], t_printed),
                 color=STYLE["ts"][0], lw=1.8, ls=buried_dash,
                 dash_capstyle="butt", zorder=4,
                 label=f"T D{TS_DUCT_D:.1f} printed LM/UM spans")
    ax_side.plot(_masked(ts_station, t_free),
                 _masked(ts_pts[:, 2], t_free), color="white",
                 lw=3.5, solid_capstyle="butt", zorder=5)
    ax_side.plot(_masked(ts_station, t_free),
                 _masked(ts_pts[:, 2], t_free),
                 color=STYLE["ts"][0], lw=2.2,
                 solid_capstyle="butt", zorder=6,
                 label=(f"T D{TS_CABLE_D_EST:.1f} LM/UM gap + "
                        "free behind tweeter"))

    for bump in MAIN_COVERED_BUMPS:
        mask = np.abs(station - bump.station) <= bump.half_length
        ax_side.plot(station[mask], pts[mask, 2], color="#31485f",
                     lw=2.8, alpha=0.55, zorder=7)
    for bump in T_COVERED_BUMPS:
        mask = np.abs(ts_station - bump.station) <= bump.half_length
        ax_side.plot(ts_station[mask], ts_pts[mask, 2],
                     color="#76538f", lw=2.4, alpha=0.55, zorder=7)

    pilot_floor_z = LM_SEAT_Z - LM_BORE_DEPTH_MM
    for idx, bump in enumerate(MAIN_COVERED_BUMPS):
        name, center, low_z = bump.name, bump.station, bump.low_z
        # Solid pad below a blind bore; the white upper portion is the
        # insert bore.  The D8.2 envelope passes completely beneath it.
        ax_side.add_patch(plt.Rectangle(
            (center - 4.8, PAD_FACE_Z), 9.6,
            LM_SEAT_Z - PAD_FACE_Z, fc="0.65", ec="0.30", lw=0.8,
            zorder=4))
        ax_side.add_patch(plt.Rectangle(
            (center - 3.2, pilot_floor_z), 6.4,
            LM_SEAT_Z - pilot_floor_z, fc="white", ec="0.38",
            lw=0.7, zorder=5))
        ax_side.text(center, LM_SEAT_Z + 0.45,
                     name.replace("lm_pilot_", "") + " deg pad",
                     ha="center", va="bottom", fontsize=7.5,
                     color="0.25")
        ax_side.annotate(
            f"covered Z bump\nD{CABLE_D_EST:g} z = "
            f"{low_z - CABLE_D_EST / 2.0:.1f}.."
            f"{low_z + CABLE_D_EST / 2.0:.1f}\n"
            f"needs {CORE_REAR_Z - (low_z - CABLE_D_EST / 2.0):.1f} "
            f"behind rear\npad rear = {PAD_FACE_Z:.1f}",
            (center, low_z), (center + (-28 if idx == 0 else 28), -1.3),
            ha="right" if idx == 0 else "left", fontsize=6.9,
            color="tab:orange",
            arrowprops=dict(arrowstyle="-", color="tab:orange"))

    for bump in T_COVERED_BUMPS:
        name, center, low_z = bump.name, bump.station, bump.low_z
        if not name.startswith("lm_pilot"):
            continue
        ax_side.add_patch(plt.Rectangle(
            (center - 4.8, PAD_FACE_Z), 9.6,
            LM_SEAT_Z - PAD_FACE_Z, fc="0.72", ec="0.38", lw=0.6,
            alpha=0.55, zorder=3))
        ax_side.text(center, -3.9,
                     name.rsplit("_", 1)[-1] + " deg covered T bump",
                     ha="center", va="bottom", fontsize=6.5,
                     color=STYLE["ts"][0], rotation=18)

    ax_side.axhline(CORE_REAR_Z, color="0.38", ls=":", lw=0.9,
                    zorder=2)
    ax_side.text(4.0, CORE_REAR_Z - 0.35,
                 "nominal ring-lip rear z=6.8; pads/web reach z=5.3",
                 va="top", fontsize=8, color="0.35")
    ax_side.axvline(LM_ENTRY_LENGTH + LM_ARC_LENGTH,
                    color="0.55", ls=":", lw=0.8)
    ax_side.text(LM_ENTRY_LENGTH + LM_ARC_LENGTH + 3.0, 14.15,
                 "free-cable crown span begins",
                 fontsize=7.5, color="0.35", ha="left", va="top")
    # Bind the printed/free transition to the upper native LM R113 mouth.
    main_lm_end_station = station[np.argmin(np.linalg.norm(
        pts - main_lm_end_xyz, axis=1))]
    ax_side.annotate(
        f"printed UM-cable cover ends at LM R"
        f"{facts['t_lower_lm_flush_radius_mm']:.0f}\n"
        "complete span behind UM to 283 deg is free; no rear duct/grommet",
        (main_lm_end_station, main_lm_end_xyz[2]),
        (station[-1] - 7.0, 1.0), fontsize=7.4,
        color="0.3", ha="right",
        arrowprops=dict(arrowstyle="-", color="0.4"))
    if STAND_FOOT:
        feed_side_text = "floor: legacy lateral LM-ring feed mouths"
    else:
        feed_side_text = (
            f"no-floor bridge rear mouths z="
            f"{facts['no_floor_bridge_feed_rear_z_mm']:g}; "
            "shallow rises into LM-owned printed layers")
    ax_side.annotate(
        feed_side_text, (0.0, min(pts[0, 2], ts_pts[0, 2])),
        (5.0, -11.8), fontsize=7.3, color="#31485f",
        arrowprops=dict(arrowstyle="-", color="#31485f"))
    ax_side.annotate(
        "T cable-only LM -> UM gap",
        (t_lower_gap_station, np.interp(
            t_lower_gap_station, ts_station, ts_pts[:, 2])),
        (t_lower_gap_station - 18.0, -9.5), fontsize=6.8,
        color="#8a6510",
        arrowprops=dict(arrowstyle="-", color="#8a6510"))
    ax_side.annotate(
        f"T free after UM R{facts['t_upper_um_flush_radius_mm']:.1f}; "
        "no tweeter-side printed cover",
        (t_upper_free_station, np.interp(
            t_upper_free_station, ts_station, ts_pts[:, 2])),
        (t_upper_free_station - 12.0, -6.7), fontsize=6.8,
        color="#8a6510", ha="right",
        arrowprops=dict(arrowstyle="-", color="#8a6510"))
    ax_side.text(4.0, 11.25,
                 f"roof = {facts['lm_roof_mm']:.1f} mm; "
                 f"floor = {facts['tunnel_floor_skin_mm']:.1f} mm",
                 fontsize=8, color="0.25")
    ax_side.set_xlim(0.0, max(ROUTE_LENGTH, TS_ROUTE_LENGTH) + 4.0)
    ax_side.set_ylim(-14.5, 14.8)
    ax_side.set_xlabel("distance along each route (independent stations, mm)")
    ax_side.set_ylabel("z (mm)")
    ax_side.grid(True, lw=0.3, alpha=0.35)
    ax_side.set_title("longitudinal side: dashed/hatched = printed cover; "
                      "solid = free cable\n"
                      "printed spans keep 0.8-mm skins + solid-backed Z bumps "
                      "(vertical scale exaggerated)",
                      fontsize=10)

    _draw_v1lf_bump_sections(
        ax_side, facts["solid_backfill_records"],
        duct_specs={
            "UM": (DUCT_D, STYLE["um"][0]),
            "T": (TS_DUCT_D, STYLE["ts"][0]),
        },
        tunnel_skin=TUNNEL_FLOOR_SKIN,
        lm_pad_d=PAD_D_MM,
        um_pad_d=UM_PAD_D_MM,
        lm_pilot_d=L22_PILOT_D_MM,
        um_pilot_d=UM_PILOT_D_MM,
        lm_seat_z=LM_SEAT_Z,
        um_seat_z=UM_SEAT_Z,
        stand_foot=STAND_FOOT,
        insert_clear_d=FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_D,
        insert_clear_z=FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_Z,
        shank_clear_d=FLOOR_SUPPORT_BACKFILL_SHANK_CLEAR_D,
        shank_clear_z=FLOOR_SUPPORT_BACKFILL_SHANK_CLEAR_Z,
    )

    service_inset = ax_side.inset_axes((0.62, 0.06, 0.36, 0.38))
    free_s = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(
            np.diff(free_bundle, axis=0), axis=1))))
    service_inset.plot(free_s, free_bundle[:, 2],
                       color=STYLE["um"][0], lw=2.0,
                       label=(f"D{CABLE_D_EST:g} "
                              f"R{service['handoff_radius_mm']:g}"))
    for name, lead in terminal_leads.items():
        lead_s = np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(
                np.diff(lead, axis=0), axis=1))))
        service_inset.plot(
            lead_s, lead[:, 2], color=lead_colors[name], lw=1.2)
        split_index = int(np.searchsorted(lead_s, FASTON_BREAKOUT_LENGTH))
        service_inset.plot(
            lead_s[split_index], lead[split_index, 2], marker=".",
            color=lead_colors[name], ms=3)
    service_inset.set_title("free service harness (not printed)", fontsize=6.8)
    service_inset.set_xlabel("lead station", fontsize=6)
    service_inset.set_ylabel("z", fontsize=6)
    service_inset.tick_params(labelsize=5.5)
    service_inset.grid(True, lw=0.2, alpha=0.3)

    # Separate true side profile of the intentionally free LM lead.
    lm_station = np.linspace(0.0, LM_LEAD_LENGTH, len(lm_pts))
    ax_lm_side.axhspan(-3.5, PAD_FACE_Z, color="#e9f4fb")
    ax_lm_side.axhspan(PAD_FACE_Z, CORE_REAR_Z, color="#d4dbe2",
                       alpha=0.42)
    lm_physical_r = LM_CABLE_D_EST / 2.0
    ax_lm_side.fill_between(lm_station,
                            lm_pts[:, 2] - lm_physical_r,
                            lm_pts[:, 2] + lm_physical_r,
                            color=STYLE["lm"][0], alpha=0.22)
    ax_lm_side.plot(lm_station, lm_pts[:, 2],
                    color=STYLE["lm"][0], lw=2.0, ls="-")
    ax_lm_side.axhline(PAD_FACE_Z, color="#31485f", ls="--", lw=0.8)
    ax_lm_side.axhline(CORE_REAR_Z, color="0.4", ls=":", lw=0.7)
    ax_lm_side.annotate(
        "deepest local pad/web rear z=5.3\nouter cable-top gap = 1.00 mm",
        (0.0, PAD_FACE_Z), (1.0, 8.2), fontsize=7.0, color="#31485f",
        arrowprops=dict(arrowstyle="-", color="#31485f"))
    ax_lm_side.annotate("free cable rises where rear clearance permits",
                        (LM_LEAD_LENGTH, lm_pts[-1, 2]), (6.2, -2.4),
                        fontsize=7.5, color="0.3",
                        arrowprops=dict(arrowstyle="-", color="0.4"))
    ax_lm_side.set_xlim(0.0, LM_LEAD_LENGTH)
    ax_lm_side.set_ylim(-3.5, 11.2)
    ax_lm_side.set_title(
        f"LM side: D{LM_CABLE_D_EST:.1f} short free span at "
        f"{LM_LEAD_ANGLE_DEG:.0f} deg — no printed micro-duct",
                         fontsize=9)
    ax_lm_side.set_xlabel("distance from outer feed toward D190 mouth (mm)")
    ax_lm_side.set_ylabel("z (mm)")
    ax_lm_side.tick_params(labelsize=7.5)
    ax_lm_side.grid(True, lw=0.25, alpha=0.3)

    state_heading = ("FLOOR / BOLTED NL8 SUPPORT ADD-ON" if STAND_FOOT
                     else "NO-FLOOR / SOFT-BLEND WEB / REAR BRIDGE FEEDS")
    fig.suptitle(
        "LX521 V1LF cable routing — minimum integral R6F core — "
        f"{state_heading}\nflush butt mouths + free cable gaps; "
        "no V1LF grommet; terminals/leads remain fit proxies pending dry-fit",
        fontsize=11.5, y=0.988)
    # Reserve a real title band: the two-line figure heading must not collide
    # with either subplot's own two-line title after tight bounding-box crop.
    fig.subplots_adjust(bottom=0.135, top=0.885)
    out = f"baffle_cable_routing_{ROUTING_PROFILE}.png"
    _prune_legacy_routing_png()
    _save_routing_figure(fig, out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    if ROUTING_PROFILE == "v1lf":
        render_v1lf()
        sys.exit(0)
    if STAND_FOOT:
        # The release checker requires review sheets >=1600 px wide.  Keep
        # both states at one explicit width so bbox/layout changes cannot
        # silently drop the no-floor sheet below that review resolution.
        fig = plt.figure(figsize=(10.8, 15.6), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[640, 190],
                              width_ratios=[340, 195], hspace=0.04,
                              wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_side = fig.add_subplot(gs[0, 1], sharey=ax)
        ax_top = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_leg = fig.add_subplot(gs[1, 1])
        ax_leg.axis("off")
    else:
        fig = plt.figure(figsize=(10.8, 12.6), dpi=150)
        gs = fig.add_gridspec(1, 2, width_ratios=[340, 55], wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_side = fig.add_subplot(gs[0, 1], sharey=ax)
        ax_top = ax_leg = None
    draw(ax, OUTLINE_B2, TWEETER_DROP_MM,
         f"Proud family {ROUTING_REV} - normal ducts + keyed V1L UM alternate")
    draw_terminal_service(ax)
    for name, (color, label) in STYLE.items():
        pts = duct_xyz(name)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=CABLE_D.get(name, 3.8), alpha=0.55,
                solid_capstyle="round", zorder=7, label=label or None)
        ax.plot(*pts[-1][:2], marker="o", ms=7, mfc="white", mec=color, zorder=8)
        if STAND_FOOT:
            # elbow dive + run rearward through the foot (into the page),
            # exiting the channel step face just short of the NL8 panel.
            # The shared TS main has no lane of its own (it starts at the
            # z-step); the pair feeders map onto the t1/t2 lanes.
            lane = {"t1f": "t1", "t2f": "t2"}.get(name, name)
            if lane not in FOOT_LANES:
                continue
            x, _, y_f, _, _ = FOOT_LANES[lane]
            ax.plot([pts[0][0], x], [pts[0][1], y_f], color=color, ls=":",
                    lw=CABLE_D.get(name, 3.8) * 0.6, alpha=0.55, zorder=7)
            ax.plot(x, y_f, marker="v", ms=6, mfc=color, mec="0.2", zorder=8)
        else:
            if name == "ts":
                continue
            bo = breakout_xy(name)
            ax.plot([bo[0], pts[0][0]], [bo[1], pts[0][1]], color=color,
                    lw=CABLE_D.get(name, 3.8), alpha=0.55, solid_capstyle="round",
                    zorder=7)
            ax.plot(*bo, marker="s", ms=6, mfc=color, mec="0.2", zorder=8)

    # Overlay only the exact diverging V1L terminal tail.  Q is the
    # physical aperture center on the z=6.8 rear face; the triangle is
    # the nominal outside centerline endpoint and must not be read as Q.
    v1l_tail = v1l_um_tail_xyz()
    v1l_spec = UM_HANDOFF[UM_V1L_HANDOFF_KEY]
    q = v1l_spec["rear_face_axis_point"]
    rear_end = v1l_spec["rear_end"]
    ax.plot(v1l_tail[:, 0], v1l_tail[:, 1], color=V1L_ALT_COLOR,
            lw=UM_HANDOFF_D_MM, alpha=0.18, solid_capstyle="round",
            zorder=9)
    ax.plot(v1l_tail[:, 0], v1l_tail[:, 1], color=V1L_ALT_COLOR,
            lw=2.2, ls="--", alpha=0.95, solid_capstyle="round",
            zorder=10, label=V1L_ALT_LABEL)
    ax.plot(q[0], q[1], marker="s", ms=7, mfc="white",
            mec=V1L_ALT_COLOR, mew=1.3, zorder=11)
    ax.plot(rear_end[0], rear_end[1], marker="v", ms=6,
            mfc=V1L_ALT_COLOR, mec=V1L_ALT_COLOR, zorder=11)
    ax.annotate(
        "V1L mid-right tail; top unchanged\n"
        f"Q=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.1f}), "
        f"r{UM_V1L_AXIS_STATION_MM:g} @ "
        f"{UM_TERMINAL_CLOCK_DEG:g} deg\n"
        f"outside=({rear_end[0]:.3f}, {rear_end[1]:.3f}, "
        f"{rear_end[2]:.1f})",
        (q[0], q[1]), (-148, 330), fontsize=7.2,
        color=V1L_ALT_COLOR,
        bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.2),
        arrowprops=dict(arrowstyle="-", color=V1L_ALT_COLOR))
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
    out = f"baffle_cable_routing_{ROUTING_PROFILE}.png"
    _prune_legacy_routing_png()
    _save_routing_figure(fig, out)
    print(f"wrote {out}")
