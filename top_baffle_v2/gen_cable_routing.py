"""Render separate proud/R6P and skeletal-Obi-Wan/R6F routing sheets.

Faithful: both sheets sample the same complete centerlines used by the
subtractive proud cutters or integral Obi-Wan printed-cover spans.  The proud
sheet shows the normal B2/C7/V0/V1 UM handoff plus the exact, clearly
labeled V1L-only alternate tail to its 283-degree rear-face aperture.
The Obi-Wan sheet shows fully covered local Z bumps, their full-width burial
webs and solid roof-to-blind-bore backfill, the short crown crossover with T
above UM, the short free LM lead (no micro-duct), deliberately free cable
handoffs, six fully buried LM/UM magnets, and the 283 degree terminal clock.
Each state is rendered in three actual orthographic projections.  Floor mode
shows the integral LM-owned stem/foot/connector body and its three buried
continuation lumens; no-floor mode shows the fused front-flush bridge web.

  front view (x-y)   duct mains + breakout/exit markers over the outline
  side view  (y-z)   true carrier/stem/foot depth projection with routes
  top view   (x-z)   true footprint/depth projection with routes
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
    profile_token = (
        "Obi-Wan" if ROUTING_PROFILE == "obiwan"
        else ROUTING_PROFILE.upper())
    token = f"LX521_{profile_token}_{ROUTING_REV}_{state}"
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.tmp.png")
    metadata = {
        "Title": token,
        "Description": (
            f"{token}; LX_STAND_FOOT={int(STAND_FOOT)}; "
            f"LX_ROUTING_PROFILE={ROUTING_PROFILE}"
            + ("; LX_OBIWAN_SIDE_SECTION=roof_to_bore_solid_backfill"
               "; LX_OBIWAN_VIEWS=front_xy,side_yz,top_xz"
               "; LX_OBIWAN_SEPARATE_FLOOR_SUPPORT=0"
               if ROUTING_PROFILE == "obiwan" else "")),
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


def _draw_obiwan_terminal_service(
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
        OBIWAN_TERMINATED_HANDOFF_R,
        OBIWAN_TERMINATED_HANDOFF_STEPS,
        obiwan_terminal_lead_points,
        obiwan_terminal_lead_points_for_terminal_pull,
        obiwan_terminated_cable_points,
    )
    from top_baffle_nd25fw4_obiwan_route import UM_MOUTH_TANGENT

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

    terminated = np.asarray(obiwan_terminated_cable_points(), dtype=float)
    # The entire UM-side span is free cable. The sampled route reaches the
    # D82 reference before this final handoff continues to the terminal axis;
    # neither interval is a small printed duct or a grommet.
    free_bundle = terminated[-(OBIWAN_TERMINATED_HANDOFF_STEPS + 1):]
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
        for name, points in obiwan_terminal_lead_points().items()
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
                obiwan_terminal_lead_points_for_terminal_pull(
                    terminal_id, station_mm)[f"terminal_lead_{terminal_id}"],
                dtype=float)
            ax.plot(intermediate[:, 0], intermediate[:, 1], color=color,
                    lw=FASTON_LEAD_D * 0.52, alpha=0.10,
                    solid_capstyle="round", zorder=9)

        pulled_lead = np.asarray(
            obiwan_terminal_lead_points_for_terminal_pull(
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
        f"free R{OBIWAN_TERMINATED_HANDOFF_R:.0f}; no grommet",
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
        "handoff_radius_mm": OBIWAN_TERMINATED_HANDOFF_R,
        "lead_min_bend_radius_mm": FASTON_LEAD_MIN_BEND_R,
    }


def render_obiwan():
    """Render true XY/YZ/XZ orthographic routing views for one Obi-Wan state."""
    from matplotlib.lines import Line2D
    from gen_driver_overlay import outline_polygon
    from shapely.geometry import Point, Polygon, box
    from top_baffle_nd25fw4 import (
        BRIDGE_HOLE_XY,
        BRIDGE_INSERT_D_MM,
        L22_CUTOUT,
        L22_PILOT_D_MM,
        THICKNESS_MM,
        UM_CUTOUT,
        UM_PILOT_D_MM,
    )
    from top_baffle_nd25fw4_flush import (
        LM_PILOT_XY,
        LM_RECESS_R,
        PAD_D_MM,
        UM_PILOT_XY,
        UM_RECESS_R,
    )
    from top_baffle_nd25fw4_obiwan import (
        CORE_REAR_Z,
        JUNCTION_WEB_Z,
        JOINT_EAR_X,
        JOINT_EAR_Y,
        LM_CORE_R,
        SIDE_MAGNET_DEPTH,
        SIDE_MAGNET_POCKET_D,
        TWEETER_JOINT_HOLE_D,
        TWEETER_JOINT_X,
        TWEETER_JOINT_Y,
        UM_CORE_R,
        _complete_joint_ear_plan,
        _complete_tweeter_joint_ear_plan,
        junction_closure_polygons,
        side_magnet_sites,
    )
    from top_baffle_nd25fw4_obiwan_bridge import (
        BRIDGE_FACE_Z,
        bridge_face_plan,
    )
    from top_baffle_nd25fw4_obiwan_floor import (
        FLOOR_LANE_SPECS,
        FOOT_FRONT_Z_MM,
        FOOT_HEIGHT_MM,
        FOOT_REAR_Z_MM,
        FOOT_WIDTH_MM,
        NL8_CENTER_Y_MM,
        NL8_CUTOUT_D_MM,
        NL8_SCREW_D_MM,
        NL8_SCREW_PITCH_MM,
        PANEL_H_MM,
        PANEL_INNER_Z_MM,
        ROOT_FILLET_R_MM,
        SERVICE_CAVITY_X_MM,
        SERVICE_CAVITY_Y_MM,
        SERVICE_CAVITY_Z_MM,
        STEM_HALF_WIDTH_MM,
        STEM_SHOULDER_HALF_WIDTH_MM,
        STEM_TOP_Y_MM,
        STEM_Z_MM,
        floor_lane_control_points,
        integral_stem_plan_points,
        integrated_floor_facts,
    )
    from top_baffle_nd25fw4_obiwan_route import (
        CABLE_D_EST,
        DUCT_D,
        LM_CABLE_D_EST,
        TS_ADDON_SUPPORT_MIN_Y,
        TS_CABLE_D_EST,
        TS_DUCT_D,
        TS_TWEETER_FLUSH_R,
        lm_cable_points,
        route_cable_points,
        route_facts,
        ts_cable_points,
    )

    route = np.asarray(route_cable_points(spacing_mm=0.45), dtype=float)
    t_route = np.asarray(ts_cable_points(spacing_mm=0.45), dtype=float)
    lm_lead = np.asarray(lm_cable_points(spacing_mm=0.35), dtype=float)
    facts = route_facts()
    floor_facts = integrated_floor_facts() if STAND_FOOT else None

    lm_center = np.asarray(L22_CUTOUT[:2], dtype=float)
    um_center = np.asarray(UM_CUTOUT[:2], dtype=float)
    main_in_lm = (
        np.linalg.norm(route[:, :2] - lm_center, axis=1)
        <= facts["t_lower_lm_flush_radius_mm"] + 1e-6)
    t_in_lm = (
        np.linalg.norm(t_route[:, :2] - lm_center, axis=1)
        <= facts["t_lower_lm_flush_radius_mm"] + 1e-6)
    t_in_um = (
        np.linalg.norm(t_route[:, :2] - um_center, axis=1)
        <= facts["t_upper_um_flush_radius_mm"] + 1e-6)
    if STAND_FOOT:
        main_entry_owner = (
            (np.abs(route[:, 0]) <= STEM_SHOULDER_HALF_WIDTH_MM)
            & (route[:, 1] <= STEM_TOP_Y_MM + 1e-6))
        t_entry_owner = (
            (np.abs(t_route[:, 0]) <= STEM_SHOULDER_HALF_WIDTH_MM)
            & (t_route[:, 1] <= STEM_TOP_Y_MM + 1e-6))
    else:
        bridge = bridge_face_plan()
        main_entry_owner = np.asarray([
            bridge.covers(Point(float(x), float(y)))
            for x, y in route[:, :2]
        ])
        t_entry_owner = np.asarray([
            bridge.covers(Point(float(x), float(y)))
            for x, y in t_route[:, :2]
        ])
    main_printed = main_in_lm | main_entry_owner
    t_printed = t_in_lm | t_in_um | t_entry_owner

    buried_dash = (0, (4.0, 2.2))
    route_colors = {
        "lm": STYLE["lm"][0],
        "um": STYLE["um"][0],
        "t": STYLE["ts"][0],
    }
    magnet_sites = tuple(side_magnet_sites())
    closure_plans = junction_closure_polygons()
    closure_colors = {
        "lm": "#6f9fba",
        "um": "#8eb8c9",
        "tweeter": "#b695cc",
    }
    if (sum(site["driver"] == "lm" for site in magnet_sites) != 4
            or sum(site["driver"] == "um" for site in magnet_sites) != 2
            or any(not site["magnet_fully_buried"]
                   for site in magnet_sites)
            or not math.isclose(SIDE_MAGNET_POCKET_D, 5.2)
            or not math.isclose(SIDE_MAGNET_DEPTH, 2.1)):
        raise RuntimeError(
            "routing view requires four LM and two UM captive D5.20x2.10 "
            "magnet cavities")

    def masked(values, mask):
        return np.ma.masked_where(~mask, values)

    def plot_owned_route(ax, points, printed, axes, color, duct_d,
                         cable_d, printed_label=None, free_label=None):
        a0, a1 = axes
        ax.plot(
            masked(points[:, a0], printed),
            masked(points[:, a1], printed),
            color="0.28", lw=duct_d + 1.6, alpha=0.16,
            ls=buried_dash, dash_capstyle="butt", zorder=6)
        ax.plot(
            masked(points[:, a0], printed),
            masked(points[:, a1], printed),
            color=color, lw=max(1.8, duct_d * 0.52), alpha=0.86,
            ls=buried_dash, dash_capstyle="butt", zorder=7,
            label=printed_label)
        free = ~printed
        ax.plot(
            masked(points[:, a0], free),
            masked(points[:, a1], free),
            color="white", lw=cable_d + 1.0,
            solid_capstyle="butt", zorder=7)
        ax.plot(
            masked(points[:, a0], free),
            masked(points[:, a1], free),
            color=color, lw=max(1.6, cable_d * 0.50),
            solid_capstyle="butt", zorder=8, label=free_label)

    def plot_floor_lanes(ax, axes):
        for name in ("lm", "um", "t"):
            points = np.asarray(floor_lane_control_points(name), dtype=float)
            spec = FLOOR_LANE_SPECS[name]
            color = route_colors[name]
            ax.plot(
                points[:, axes[0]], points[:, axes[1]],
                color="0.25", lw=spec["diameter_mm"] + 1.6,
                alpha=0.17, ls=buried_dash,
                dash_capstyle="round", zorder=5)
            ax.plot(
                points[:, axes[0]], points[:, axes[1]],
                color=color, lw=max(1.8, spec["diameter_mm"] * 0.48),
                alpha=0.92, ls=buried_dash,
                dash_capstyle="round", zorder=6)

    def draw_magnet_projection(ax, axes):
        """Project every exact carrier pocket axis into an orthographic view."""
        for site in magnet_sites:
            normal = np.asarray(site["normal"], dtype=float)
            face = np.asarray((
                site["face"][0], site["face"][1], site["z_mm"]),
                dtype=float)
            inner = face.copy()
            inner[:2] -= SIDE_MAGNET_DEPTH * normal
            projected = np.asarray((
                face[axes[0]] - inner[axes[0]],
                face[axes[1]] - inner[axes[1]],
            ))
            if np.linalg.norm(projected) <= 1e-9:
                # An axis normal to the projection plane is a true circular
                # D5.2 section, not an ambiguous point marker.  The mirrored
                # lower LM pockets coincide in YZ and are intentionally drawn
                # twice at the same exact datum.
                ax.add_patch(plt.Circle(
                    (face[axes[0]], face[axes[1]]),
                    SIDE_MAGNET_POCKET_D / 2.0,
                    fc="none", ec="tab:orange", lw=1.25, zorder=10))
                ax.plot(
                    face[axes[0]], face[axes[1]], marker=".", ms=2.6,
                    color="tab:orange", zorder=10.2)
            else:
                ax.plot(
                    [inner[axes[0]], face[axes[0]]],
                    [inner[axes[1]], face[axes[1]]],
                    color="tab:orange", lw=2.4, marker="o", ms=3.2,
                    solid_capstyle="butt", zorder=10)

    def draw_closure_projection(ax, axes):
        """Show the full-depth owned webs in true YZ/XZ projection."""
        for junction in ("lm_um", "t_um"):
            for owner in (("lm", "um") if junction == "lm_um"
                          else ("um", "tweeter")):
                plan = closure_plans[junction][owner]
                if plan.is_empty:
                    continue
                color = closure_colors[owner]
                pieces = ((plan,) if plan.geom_type == "Polygon"
                          else tuple(plan.geoms))
                for piece in pieces:
                    min_x, min_y, max_x, max_y = piece.bounds
                    if axes == (2, 1):
                        rect = plt.Rectangle(
                            (JUNCTION_WEB_Z[0], min_y),
                            JUNCTION_WEB_Z[1] - JUNCTION_WEB_Z[0],
                            max_y - min_y,
                            fc=color, ec=color, lw=0.65, alpha=0.24,
                            hatch="..", zorder=2.4)
                    elif axes == (0, 2):
                        rect = plt.Rectangle(
                            (min_x, JUNCTION_WEB_Z[0]),
                            max_x - min_x,
                            JUNCTION_WEB_Z[1] - JUNCTION_WEB_Z[0],
                            fc=color, ec=color, lw=0.65, alpha=0.24,
                            hatch="..", zorder=2.4)
                    else:
                        raise ValueError(axes)
                    ax.add_patch(rect)

    def draw_core_front(ax):
        for (cx, cy, cut_d), outer, recess, color in (
                (L22_CUTOUT, LM_CORE_R, LM_RECESS_R, "#b8bec7"),
                (UM_CUTOUT, UM_CORE_R, UM_RECESS_R, "#c7cdd5")):
            ax.add_patch(plt.Circle(
                (cx, cy), outer, fc=color, ec="#303943",
                lw=1.25, zorder=2))
            ax.add_patch(plt.Circle(
                (cx, cy), recess, fill=False, ec="0.48",
                ls="--", lw=0.8, zorder=3))
            ax.add_patch(plt.Circle(
                (cx, cy), cut_d / 2.0, fc="white", ec="#303943",
                lw=1.0, zorder=4))
        for px, py in LM_PILOT_XY:
            ax.add_patch(plt.Circle(
                (px, py), PAD_D_MM / 2.0, fc="#d7dce2",
                ec="tab:orange", lw=0.75, zorder=4))
            ax.add_patch(plt.Circle(
                (px, py), L22_PILOT_D_MM / 2.0, fc="white",
                ec="tab:orange", lw=0.75, zorder=5))
        for px, py in UM_PILOT_XY:
            ax.add_patch(plt.Circle(
                (px, py), UM_PILOT_D_MM / 2.0, fc="white",
                ec="tab:orange", lw=0.75, zorder=5))

        # Full-depth plan owners close the former red cusp islands.  Their
        # fine white boundaries are the complementary 0.05-mm assembly
        # seams, not shallow skins or rear cavities.
        for junction in ("lm_um", "t_um"):
            for owner in (("lm", "um") if junction == "lm_um"
                          else ("um", "tweeter")):
                plan = closure_plans[junction][owner]
                pieces = ((plan,) if plan.geom_type == "Polygon"
                          else tuple(plan.geoms))
                for poly in pieces:
                    if poly.is_empty:
                        continue
                    ex, ey = poly.exterior.xy
                    ax.fill(
                        ex, ey, fc=closure_colors[owner], ec="white",
                        lw=0.42, alpha=0.90, zorder=3.2)
        for x in JOINT_EAR_X:
            for owner, color in (("lm", "#858d98"), ("um", "#a0a7b1")):
                ear = _complete_joint_ear_plan(owner, x)
                ex, ey = ear.exterior.xy
                ax.fill(ex, ey, fc=color, ec="0.25", lw=0.7, zorder=3)
            ax.plot(x, JOINT_EAR_Y, marker="o", ms=3.8, mfc="white",
                    mec="0.25", zorder=6)

        outline = Polygon(outline_polygon(OUTLINE_B2, samples=96))
        crescent = (
            outline.intersection(
                box(-75.0, TS_ADDON_SUPPORT_MIN_Y, 75.0, 454.0))
            .difference(Point(*UM_CUTOUT[:2]).buffer(
                TS_TWEETER_FLUSH_R, resolution=64)))
        pieces = (
            (crescent,) if crescent.geom_type == "Polygon"
            else tuple(crescent.geoms))
        for poly in pieces:
            ex, ey = poly.exterior.xy
            ax.fill(ex, ey, fc="#c9b6dc", ec="#76538f", lw=0.9,
                    alpha=0.50, zorder=2)
        for x in TWEETER_JOINT_X:
            ear = _complete_tweeter_joint_ear_plan("tweeter", x)
            ex, ey = ear.exterior.xy
            ax.fill(ex, ey, fc="#a88ac2", ec="#76538f", lw=0.8,
                    alpha=0.72, zorder=4)
            ax.add_patch(plt.Circle(
                (x, TWEETER_JOINT_Y), TWEETER_JOINT_HOLE_D / 2.0,
                fc="white", ec="#76538f", lw=0.8, zorder=8))

        for site in magnet_sites:
            nx, ny = site["normal"]
            face = np.asarray(site["face"], dtype=float)
            inner = face - SIDE_MAGNET_DEPTH * np.asarray((nx, ny))
            ax.plot(
                [inner[0], face[0]], [inner[1], face[1]],
                color="tab:orange", lw=SIDE_MAGNET_POCKET_D,
                solid_capstyle="butt", zorder=8)

    fig = plt.figure(figsize=(16.0, 12.0), dpi=150)
    gs = fig.add_gridspec(
        2, 2, width_ratios=(1.18, 0.82), height_ratios=(1.0, 1.0),
        wspace=0.16, hspace=0.17)
    ax_front = fig.add_subplot(gs[:, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[1, 1])

    support_color = "#65778a"
    if STAND_FOOT:
        stem = np.asarray(integral_stem_plan_points(), dtype=float)
        ax_front.fill(
            stem[:, 0], stem[:, 1], fc=support_color,
            ec="#31485f", lw=1.15, alpha=0.50, zorder=1)
        ax_front.add_patch(plt.Rectangle(
            (-FOOT_WIDTH_MM / 2.0, 0.0),
            FOOT_WIDTH_MM, FOOT_HEIGHT_MM,
            fc=support_color, ec="#31485f", lw=1.0,
            alpha=0.58, zorder=1))
        ax_front.add_patch(plt.Rectangle(
            (-FOOT_WIDTH_MM / 2.0, 0.0),
            FOOT_WIDTH_MM, PANEL_H_MM,
            fill=False, ec="#31485f", lw=0.9,
            ls="--", zorder=2))
        ax_front.add_patch(plt.Circle(
            (0.0, NL8_CENTER_Y_MM), NL8_CUTOUT_D_MM / 2.0,
            fill=False, ec="#31485f", lw=0.9, ls=":", zorder=3))
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                ax_front.add_patch(plt.Circle(
                    (sx * NL8_SCREW_PITCH_MM / 2.0,
                     NL8_CENTER_Y_MM
                     + sy * NL8_SCREW_PITCH_MM / 2.0),
                    NL8_SCREW_D_MM / 2.0, fc="white",
                    ec="#31485f", lw=0.7, zorder=3))
        ax_front.add_patch(plt.Rectangle(
            (SERVICE_CAVITY_X_MM[0], SERVICE_CAVITY_Y_MM[0]),
            SERVICE_CAVITY_X_MM[1] - SERVICE_CAVITY_X_MM[0],
            SERVICE_CAVITY_Y_MM[1] - SERVICE_CAVITY_Y_MM[0],
            fill=False, ec="#8b5e34", lw=0.8, ls=":", zorder=3))
        plot_floor_lanes(ax_front, (0, 1))
    else:
        bridge_plan = bridge_face_plan()
        bx, by = bridge_plan.exterior.xy
        ax_front.fill(
            bx, by, fc=support_color, ec="#31485f",
            lw=1.15, alpha=0.48, zorder=1)
        for hx, hy in BRIDGE_HOLE_XY:
            ax_front.add_patch(plt.Circle(
                (hx, hy), BRIDGE_INSERT_D_MM / 2.0,
                fc="white", ec="#31485f", lw=0.8, zorder=3))

    draw_core_front(ax_front)
    plot_owned_route(
        ax_front, route, main_printed, (0, 1),
        route_colors["um"], DUCT_D, CABLE_D_EST,
        "UM: LM/stem-owned buried cover", "UM: free cable behind UM")
    plot_owned_route(
        ax_front, t_route, t_printed, (0, 1),
        route_colors["t"], TS_DUCT_D, TS_CABLE_D_EST,
        "T: LM/UM-owned buried cover", "T: free handoff")
    ax_front.plot(
        lm_lead[:, 0], lm_lead[:, 1],
        color=route_colors["lm"], lw=max(1.8, LM_CABLE_D_EST * 0.50),
        solid_capstyle="round", zorder=8,
        label="LM short lead floats; no micro-duct")
    _draw_obiwan_terminal_service(
        ax_front, facts,
        tuple(route[-1, :2]), CABLE_D_EST)
    if STAND_FOOT:
        ax_front.annotate(
            "LM-owned integral W64 stem + foot\n"
            "full depth z=0..18.3; no separate yoke, rail or support",
            (STEM_HALF_WIDTH_MM, 58.0), (128, 38),
            ha="right", fontsize=7.8, color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))
        ax_front.annotate(
            "rear NL8 panel/service region (hidden lines)\n"
            "three continuation lumens stay buried to central feeds",
            (0.0, 22.0), (-130, 42),
            fontsize=7.7, color="#8b5e34",
            arrowprops=dict(arrowstyle="-", color="#8b5e34"))
    else:
        ax_front.annotate(
            "fused front-flush bridge web\n"
            "unchanged 40 x 50 insert datums; no rear keel",
            (31.0, 58.0), (128, 40),
            ha="right", fontsize=7.8, color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))
    ax_front.annotate(
        "lower LM magnets: shared base sides\n"
        "x=+/-32, y=18, z=12.55; matching Ac/Ae LM-lower receivers",
        (32.0, 18.0), (104, 72), ha="right",
        fontsize=7.1, color="tab:orange",
        arrowprops=dict(arrowstyle="-", color="tab:orange"))
    ax_front.annotate(
        "all route bypass bumps are closed and solid-backed\n"
        "ordinary blind driver bores; zero open windows/cavities",
        (route[0, 0], route[0, 1]), (-130, 116),
        fontsize=7.4, color="#31485f",
        arrowprops=dict(arrowstyle="-", color="#31485f"))
    ax_front.annotate(
        "full-depth plan-split closure webs\n"
        "only central T free-cable mouth remains open",
        (0.0, 418.0), (-126, 438),
        fontsize=7.2, color=closure_colors["um"],
        arrowprops=dict(arrowstyle="-", color=closure_colors["um"]))
    ax_front.set_xlim(-140, 140)
    ax_front.set_ylim(-7, 462)
    ax_front.set_aspect("equal", adjustable="box")
    ax_front.set_xlabel("x (mm)")
    ax_front.set_ylabel("y (mm)")
    ax_front.set_title(
        "FRONT — XY orthographic\n"
        "dashed = buried printed lumen; solid = free cable")
    ax_front.grid(True, lw=0.28, alpha=0.28)

    # True YZ projection.  Carrier projections remain light so the depth
    # changes and cable ownership read clearly.
    for center_y, outer in (
            (L22_CUTOUT[1], LM_CORE_R),
            (UM_CUTOUT[1], UM_CORE_R)):
        ax_side.add_patch(plt.Rectangle(
            (CORE_REAR_Z, center_y - outer),
            THICKNESS_MM - CORE_REAR_Z, 2.0 * outer,
            fc="#c9cfd6", ec="#303943", lw=0.9,
            alpha=0.55, zorder=1))
    if STAND_FOOT:
        ax_side.add_patch(plt.Rectangle(
            (STEM_Z_MM[0], 0.0),
            STEM_Z_MM[1] - STEM_Z_MM[0], STEM_TOP_Y_MM,
            fc=support_color, ec="#31485f", lw=1.0,
            alpha=0.53, zorder=1))
        ax_side.add_patch(plt.Rectangle(
            (FOOT_REAR_Z_MM, 0.0),
            FOOT_FRONT_Z_MM - FOOT_REAR_Z_MM, FOOT_HEIGHT_MM,
            fc=support_color, ec="#31485f", lw=1.0,
            alpha=0.62, zorder=1))
        root_center_y = FOOT_HEIGHT_MM + ROOT_FILLET_R_MM
        root_center_z = STEM_Z_MM[0] - ROOT_FILLET_R_MM
        angles = np.linspace(0.0, -0.5 * math.pi, 40)
        root_y = root_center_y + ROOT_FILLET_R_MM * np.sin(angles)
        root_z = root_center_z + ROOT_FILLET_R_MM * np.cos(angles)
        root_poly = np.asarray((
            (root_center_z, FOOT_HEIGHT_MM),
            (STEM_Z_MM[0], FOOT_HEIGHT_MM),
            (STEM_Z_MM[0], root_center_y),
            *zip(root_z[1:], root_y[1:]),
            (root_center_z, FOOT_HEIGHT_MM),
        ), dtype=float)
        ax_side.fill(
            root_poly[:, 0], root_poly[:, 1],
            fc=support_color, ec="#31485f",
            lw=0.9, alpha=0.68, zorder=2)
        ax_side.add_patch(plt.Rectangle(
            (FOOT_REAR_Z_MM, 0.0),
            PANEL_INNER_Z_MM - FOOT_REAR_Z_MM, PANEL_H_MM,
            fc="#8797a7", ec="#31485f", lw=0.9,
            alpha=0.72, zorder=2))
        actual_cavity_top_y = min(
            FOOT_HEIGHT_MM, SERVICE_CAVITY_Y_MM[1])
        ax_side.add_patch(plt.Rectangle(
            (SERVICE_CAVITY_Z_MM[0], SERVICE_CAVITY_Y_MM[0]),
            SERVICE_CAVITY_Z_MM[1] - SERVICE_CAVITY_Z_MM[0],
            actual_cavity_top_y - SERVICE_CAVITY_Y_MM[0],
            fc="white", ec="#8b5e34", lw=0.9,
            hatch="//", alpha=0.92, zorder=3))
        ax_side.add_patch(plt.Rectangle(
            (SERVICE_CAVITY_Z_MM[0], SERVICE_CAVITY_Y_MM[0]),
            SERVICE_CAVITY_Z_MM[1] - SERVICE_CAVITY_Z_MM[0],
            SERVICE_CAVITY_Y_MM[1] - SERVICE_CAVITY_Y_MM[0],
            fill=False, ec="#8b5e34", lw=0.8,
            ls=":", zorder=4))
        ax_side.add_patch(plt.Rectangle(
            (FOOT_REAR_Z_MM - 0.4,
             NL8_CENTER_Y_MM - NL8_CUTOUT_D_MM / 2.0),
            PANEL_INNER_Z_MM - FOOT_REAR_Z_MM + 0.8,
            NL8_CUTOUT_D_MM,
            fc="white", ec="#31485f", lw=0.75, zorder=4))
        plot_floor_lanes(ax_side, (2, 1))
        ax_side.annotate(
            f"true R{ROOT_FILLET_R_MM:g} internal root",
            (-6.5, FOOT_HEIGHT_MM + 6.0), (-70, 58),
            fontsize=7.3, color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))
        ax_side.annotate(
            "rear NL8 panel + necessary service cavity\n"
            "connector axis y=22; panel z=-150..-146",
            (-148, NL8_CENTER_Y_MM), (-108, 75),
            fontsize=7.2, color="#8b5e34",
            arrowprops=dict(arrowstyle="-", color="#8b5e34"))
        ax_side.annotate(
            "three buried continuations\n"
            "LM D9 / UM D8.2 / shared T D6; R>=14 turn",
            (-55, 10.0), (-148, 126),
            fontsize=7.2, color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))
    else:
        bridge_min_x, bridge_min_y, bridge_max_x, bridge_max_y = (
            bridge.bounds)
        ax_side.add_patch(plt.Rectangle(
            (BRIDGE_FACE_Z[0], bridge_min_y),
            BRIDGE_FACE_Z[1] - BRIDGE_FACE_Z[0],
            bridge_max_y - bridge_min_y,
            fc=support_color, ec="#31485f", lw=1.0,
            alpha=0.58, zorder=2))
        ax_side.annotate(
            "front-flush bridge only\nz=5.3..18.3; no depth structure",
            (sum(BRIDGE_FACE_Z) / 2.0, 55.0), (-8, 118),
            fontsize=7.5, color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))

    plot_owned_route(
        ax_side, route, main_printed, (2, 1),
        route_colors["um"], DUCT_D, CABLE_D_EST)
    plot_owned_route(
        ax_side, t_route, t_printed, (2, 1),
        route_colors["t"], TS_DUCT_D, TS_CABLE_D_EST)
    ax_side.plot(
        lm_lead[:, 2], lm_lead[:, 1],
        color=route_colors["lm"], lw=2.0,
        solid_capstyle="round", zorder=8)
    draw_closure_projection(ax_side, (2, 1))
    draw_magnet_projection(ax_side, (2, 1))
    ax_side.annotate(
        "2x lower LM captive cavities\nD5.20 section; axes +/-X",
        (12.55, 18.0), (-70.0, 42.0), fontsize=6.9,
        color="tab:orange",
        arrowprops=dict(arrowstyle="-", color="tab:orange", lw=0.8))
    ax_side.axvline(0.0, color="0.20", lw=0.7, ls=":", zorder=0)
    ax_side.set_xlim(-158, 24)
    ax_side.set_ylim(-7, 462)
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.set_xlabel("z (mm)")
    ax_side.set_ylabel("y (mm)")
    ax_side.set_title(
        "SIDE — YZ orthographic\n"
        "actual depth, root, service cavity and Z-bypass paths")
    ax_side.grid(True, lw=0.28, alpha=0.28)

    # True XZ projection.  This is not a route-station plot: every line uses
    # its world x and z coordinates.
    ax_top.add_patch(plt.Rectangle(
        (-LM_CORE_R, CORE_REAR_Z),
        2.0 * LM_CORE_R, THICKNESS_MM - CORE_REAR_Z,
        fc="#c9cfd6", ec="#303943", lw=0.9,
        alpha=0.50, zorder=1))
    if STAND_FOOT:
        ax_top.add_patch(plt.Rectangle(
            (-STEM_SHOULDER_HALF_WIDTH_MM, STEM_Z_MM[0]),
            2.0 * STEM_SHOULDER_HALF_WIDTH_MM,
            STEM_Z_MM[1] - STEM_Z_MM[0],
            fc=support_color, ec="#31485f", lw=0.85,
            alpha=0.36, zorder=1))
        ax_top.add_patch(plt.Rectangle(
            (-FOOT_WIDTH_MM / 2.0, FOOT_REAR_Z_MM),
            FOOT_WIDTH_MM, FOOT_FRONT_Z_MM - FOOT_REAR_Z_MM,
            fc=support_color, ec="#31485f", lw=1.0,
            alpha=0.60, zorder=2))
        ax_top.add_patch(plt.Rectangle(
            (-FOOT_WIDTH_MM / 2.0, FOOT_REAR_Z_MM),
            FOOT_WIDTH_MM, PANEL_INNER_Z_MM - FOOT_REAR_Z_MM,
            fc="#8797a7", ec="#31485f", lw=0.8,
            alpha=0.72, zorder=3))
        ax_top.add_patch(plt.Rectangle(
            (-NL8_CUTOUT_D_MM / 2.0, FOOT_REAR_Z_MM - 0.4),
            NL8_CUTOUT_D_MM,
            PANEL_INNER_Z_MM - FOOT_REAR_Z_MM + 0.8,
            fc="white", ec="#31485f", lw=0.75, zorder=4))
        for sx in (-1.0, 1.0):
            ax_top.add_patch(plt.Rectangle(
                (sx * NL8_SCREW_PITCH_MM / 2.0
                 - NL8_SCREW_D_MM / 2.0,
                 FOOT_REAR_Z_MM - 0.4),
                NL8_SCREW_D_MM,
                PANEL_INNER_Z_MM - FOOT_REAR_Z_MM + 0.8,
                fc="white", ec="#31485f", lw=0.55, zorder=4))
        ax_top.add_patch(plt.Rectangle(
            (SERVICE_CAVITY_X_MM[0], SERVICE_CAVITY_Z_MM[0]),
            SERVICE_CAVITY_X_MM[1] - SERVICE_CAVITY_X_MM[0],
            SERVICE_CAVITY_Z_MM[1] - SERVICE_CAVITY_Z_MM[0],
            fc="white", ec="#8b5e34", lw=0.9,
            hatch="//", alpha=0.92, zorder=4))
        plot_floor_lanes(ax_top, (0, 2))
        ax_top.annotate(
            "W64 solid rectangular foot\n"
            "same depth as connector receptacle",
            (FOOT_WIDTH_MM / 2.0, -75.0), (112, -96),
            ha="right", fontsize=7.6, color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))
        ax_top.annotate(
            "rear panel + service cavity",
            (0.0, -125.0), (-106, -140),
            fontsize=7.3, color="#8b5e34",
            arrowprops=dict(arrowstyle="-", color="#8b5e34"))
    else:
        ax_top.add_patch(plt.Rectangle(
            (bridge_min_x, BRIDGE_FACE_Z[0]),
            bridge_max_x - bridge_min_x,
            BRIDGE_FACE_Z[1] - BRIDGE_FACE_Z[0],
            fc=support_color, ec="#31485f", lw=1.0,
            alpha=0.58, zorder=2))
        ax_top.annotate(
            "bridge web projection only",
            (31.0, sum(BRIDGE_FACE_Z) / 2.0), (106, -4),
            ha="right", fontsize=7.5, color="#31485f",
            arrowprops=dict(arrowstyle="-", color="#31485f"))

    plot_owned_route(
        ax_top, route, main_printed, (0, 2),
        route_colors["um"], DUCT_D, CABLE_D_EST)
    plot_owned_route(
        ax_top, t_route, t_printed, (0, 2),
        route_colors["t"], TS_DUCT_D, TS_CABLE_D_EST)
    ax_top.plot(
        lm_lead[:, 0], lm_lead[:, 2],
        color=route_colors["lm"], lw=2.0,
        solid_capstyle="round", zorder=8)
    draw_closure_projection(ax_top, (0, 2))
    draw_magnet_projection(ax_top, (0, 2))
    ax_top.axhline(0.0, color="0.20", lw=0.7, ls=":", zorder=0)
    ax_top.set_xlim(-120, 120)
    ax_top.set_ylim(-158 if STAND_FOOT else -8, 24)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("x (mm)")
    ax_top.set_ylabel("z (mm)")
    ax_top.set_title(
        "TOP — XZ orthographic\n"
        "true plan-depth footprint; buried lanes remain inside the solid")
    ax_top.grid(True, lw=0.28, alpha=0.28)

    legend = [
        Line2D([0], [0], color=route_colors["um"], lw=3, ls="--",
               label=f"UM D{DUCT_D:g} buried owner span"),
        Line2D([0], [0], color=route_colors["um"], lw=3,
               label=f"UM D{CABLE_D_EST:g} free cable"),
        Line2D([0], [0], color=route_colors["t"], lw=3, ls="--",
               label=f"T D{TS_DUCT_D:g} buried owner spans"),
        Line2D([0], [0], color=route_colors["t"], lw=3,
               label=f"T D{TS_CABLE_D_EST:g} free handoffs"),
        Line2D([0], [0], color=route_colors["lm"], lw=3,
               label=f"LM D{LM_CABLE_D_EST:g} short free lead"),
        Line2D([0], [0], color=support_color, lw=7, alpha=0.55,
               label=("integral LM-owned floor body"
                      if STAND_FOOT else "fused no-floor bridge web")),
        Line2D([0], [0], color=closure_colors["um"], lw=7, alpha=0.75,
               label="full-depth plan-split LM–UM / T–UM closure webs"),
        Line2D([0], [0], color="tab:orange", lw=3, marker="o",
               label="buried D5.20 x 2.10 cavities; lower LM at base"),
    ]
    if STAND_FOOT:
        legend.extend([
            Line2D([0], [0], color=route_colors["lm"], lw=3, ls="--",
                   label="integral floor LM lumen D9"),
            Line2D([0], [0], color=route_colors["um"], lw=3, ls="--",
                   label="integral floor UM lumen D8.2"),
            Line2D([0], [0], color=route_colors["t"], lw=3, ls="--",
                   label="integral floor shared-T lumen D6"),
        ])
    fig.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.035),
        ncol=3, fontsize=7.4, framealpha=0.94)

    state_heading = (
        "FLOOR — INTEGRAL W64 STEM/FOOT + REAR NL8"
        if STAND_FOOT else
        "NO-FLOOR — FUSED FRONT-FLUSH BRIDGE")
    fig.suptitle(
        "LX521 Obi-Wan cable routing — "
        f"{state_heading}\n"
        "true orthographic views; buried Z-preferred lanes; "
        "no separate floor support and no Obi-Wan grommets",
        fontsize=12.0, y=0.985)
    fig.text(
        0.995, 0.006,
        "LX_OBIWAN_VIEWS=front_xy,side_yz,top_xz | "
        "LX_OBIWAN_SEPARATE_FLOOR_SUPPORT=0",
        ha="right", va="bottom", fontsize=6.8, color="0.30")
    if STAND_FOOT:
        fig.text(
            0.01, 0.006,
            f"floor datum y={floor_facts['floor_y_mm']:g}; "
            f"LM axis-to-floor={floor_facts['lm_axis_to_floor_mm']:.3f} mm; "
            f"root R{floor_facts['root_fillet_r_mm']:g}",
            ha="left", va="bottom", fontsize=6.8, color="0.30")
    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.07, right=0.98)

    out = f"baffle_cable_routing_{ROUTING_PROFILE}.png"
    _prune_legacy_routing_png()
    _save_routing_figure(fig, out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    if ROUTING_PROFILE == "obiwan":
        render_obiwan()
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
