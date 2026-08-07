#!/usr/bin/env python3
"""Constraint-accurate review sheet for the no-floor Obi-Wan entries.

This is a review drawing, not production geometry.  Every outline, route,
entry, insert and Z datum comes from the released sources.  It documents two
separate questions:

* how the presently rear-open first portions of the UM/T runs could be closed
  by a printed rear saddle; and
* why the proposed direct T chord has no buried passage through the existing
  13-mm web, even though its complete OD7.6 plan envelope is inside the
  printed XY silhouette.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from top_baffle_nd25fw4 import (
    BRIDGE_HOLE_XY,
    BRIDGE_INSERT_D_MM,
    BRIDGE_INSERT_DEPTH_MM,
    L22_CUTOUT,
    THICKNESS_MM,
)
from top_baffle_nd25fw4_obiwan_bridge import bridge_face_plan
import top_baffle_nd25fw4_obiwan_route as route


REAR_Z = float(route.PAD_FACE_Z)
FRONT_Z = float(THICKNESS_MM)
LM_BORE_TOP_Z = float(route.NO_FLOOR_LM_ENTRY_BORE_INNER_Z_MM)
INSERT_BORE_TOP_Z = REAR_Z + float(BRIDGE_INSERT_DEPTH_MM)
DIRECT_WALL_MM = 0.44


def _route_station(points):
    points = np.asarray(points, dtype=float)
    return np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def _fill_polygon(ax, geometry, *, facecolor, edgecolor, alpha=1.0,
                  linewidth=1.0, zorder=0):
    polygons = ([geometry] if geometry.geom_type == "Polygon"
                else list(geometry.geoms))
    for polygon in polygons:
        xy = np.asarray(polygon.exterior.coords, dtype=float)
        ax.fill(xy[:, 0], xy[:, 1], fc=facecolor, ec=edgecolor,
                alpha=alpha, lw=linewidth, zorder=zorder)
        for ring in polygon.interiors:
            hole = np.asarray(ring.coords, dtype=float)
            ax.fill(hole[:, 0], hole[:, 1], fc="white", ec=edgecolor,
                    lw=linewidth, zorder=zorder + 0.1)


def _released_plan_material():
    """Plan projection of the bridge plus the structural LM annulus."""
    center = L22_CUTOUT[:2]
    outer = Point(*center).buffer(route.LM_CORE_R, resolution=384)
    opening = Point(*center).buffer(L22_CUTOUT[2] / 2.0, resolution=384)
    return unary_union((bridge_face_plan(), outer)).difference(opening)


def _direct_t_chord():
    """The user's red alternative, ending tangent to the released LM arc."""
    return LineString((
        tuple(map(float, route.NO_FLOOR_T_FEED_XY)),
        tuple(map(float, route._TS_LM_ARC_START)),
    ))


def _point_at(line, station):
    point = line.interpolate(float(station))
    return float(point.x), float(point.y)


def _plan_panel(ax):
    material = _released_plan_material()
    direct = _direct_t_chord()
    direct_outer = direct.buffer(
        route.TS_OUTER_R, resolution=64, cap_style=1, join_style=1)
    if not material.covers(direct_outer):
        outside = direct_outer.difference(material)
        raise RuntimeError(
            "direct T review chord unexpectedly leaves the XY silhouette: "
            f"outside_area={outside.area:.6f} mm2")

    _fill_polygon(
        ax, material, facecolor="#cfd5dc", edgecolor="#4f5964",
        linewidth=1.2, zorder=0)
    _fill_polygon(
        ax, direct_outer, facecolor="#ef9a9a", edgecolor="#c62828",
        alpha=0.26, linewidth=0.8, zorder=2)

    # Released T centerline and its existing LM-ring continuation.
    current = np.asarray(route._TS_ENTRY, dtype=float)
    ax.plot(current[:, 0], current[:, 1], color="#d89000", lw=2.4,
            label="released R14 T route", zorder=6)
    ax.plot(route._TS_LM_ARC[:, 0], route._TS_LM_ARC[:, 1],
            color="#d89000", lw=1.5, alpha=0.75, zorder=5)

    direct_xy = np.asarray(direct.coords, dtype=float)
    ax.plot(direct_xy[:, 0], direct_xy[:, 1], color="#c62828", lw=2.4,
            ls="--", label="requested direct T chord", zorder=8)

    lm_xy = np.asarray(route.NO_FLOOR_LM_FEED_XY, dtype=float)
    um_xy = np.asarray(route.NO_FLOOR_MAIN_FEED_XY, dtype=float)
    t_xy = np.asarray(route.NO_FLOOR_T_FEED_XY, dtype=float)
    upper_left = np.asarray((-20.0, 70.0), dtype=float)

    # Exact voids and the centerline exclusions they impose on a D6 route.
    lm_required = (
        route.LM_INTERNAL_DUCT_R + route.TS_CUTTER_R + DIRECT_WALL_MM)
    insert_required = BRIDGE_INSERT_D_MM / 2.0 + route.TS_OUTER_R
    obstacles = (
        (lm_xy, route.LM_INTERNAL_DUCT_R, lm_required,
         "LM D9 rear bore", "#6650a4"),
        (upper_left, BRIDGE_INSERT_D_MM / 2.0, insert_required,
         "blind insert D6.4", "#34495e"),
    )
    for xy, void_r, exclusion_r, label, color in obstacles:
        ax.add_patch(Circle(
            xy, exclusion_r, fc="none", ec=color, lw=1.0, ls=":",
            zorder=7))
        ax.add_patch(Circle(
            xy, void_r, fc="white", ec=color, lw=1.8, zorder=9))
        ax.annotate(label, xy, xytext=(8, 8), textcoords="offset points",
                    fontsize=7.8, color=color, zorder=11)

    # Other immutable mouths and the remaining bridge inserts provide scale.
    ax.add_patch(Circle(
        um_xy, route.CUTTER_R, fc="white", ec="#2b8c62", lw=1.8,
        zorder=9))
    ax.annotate("UM D8.2", um_xy, xytext=(8, -9),
                textcoords="offset points", fontsize=7.8,
                color="#2b8c62", zorder=11)
    ax.add_patch(Circle(
        t_xy, route.TS_CUTTER_R, fc="white", ec="#d89000", lw=1.8,
        zorder=10))
    ax.annotate("T D6", t_xy, xytext=(8, -8),
                textcoords="offset points", fontsize=7.8,
                color="#b77700", zorder=11)
    for xy in BRIDGE_HOLE_XY:
        if np.allclose(xy, upper_left):
            continue
        ax.add_patch(Circle(
            xy, BRIDGE_INSERT_D_MM / 2.0, fc="white", ec="#34495e",
            lw=1.0, zorder=4))

    lm_clear = direct.distance(Point(*lm_xy))
    insert_clear = direct.distance(Point(*upper_left))
    lm_station = direct.project(Point(*lm_xy))
    insert_station = direct.project(Point(*upper_left))
    collisions = (
        (_point_at(direct, lm_station),
         f"LM collision\n{lm_clear:.2f} mm < {lm_required:.2f} mm"),
        (_point_at(direct, insert_station),
         f"insert collision\n{insert_clear:.2f} mm < {insert_required:.2f} mm"),
    )
    for (x, y), label in collisions:
        ax.scatter([x], [y], marker="x", s=95, color="#9c1c1c",
                   lw=2.5, zorder=13)
        ax.annotate(
            label, (x, y), xytext=(12, -28), textcoords="offset points",
            fontsize=7.6, color="#8e1b1b",
            bbox=dict(fc="white", ec="#c62828", alpha=0.92), zorder=14)

    ax.text(
        -86.0, 136.0,
        "OD7.6 chord is fully inside the grey XY outline.\n"
        "Failure is the two internal voids—not the exterior edge.",
        fontsize=8.0, color="#1f4d3a", ha="left", va="top",
        bbox=dict(fc="#edf7f1", ec="#2b8c62", alpha=0.95), zorder=15)
    ax.annotate(
        "released turn moves away from the\nclosely packed rear mouths first",
        xy=(-12.0, 49.0), xytext=(-75.0, 42.0), fontsize=7.8,
        color="#9a6400", ha="left",
        arrowprops=dict(arrowstyle="->", color="#9a6400", lw=1.0),
        bbox=dict(fc="white", ec="#d89000", alpha=0.9), zorder=14)

    ax.set_title(
        "PLAN — direct chord stays inside the outline, but crosses two voids",
        fontsize=10.2, weight="bold")
    ax.set_xlim(-90, 45)
    ax.set_ylim(38, 145)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, color="#d7dbe0", lw=0.5)
    ax.legend(handles=(
        Line2D([0], [0], color="#d89000", lw=2.4,
               label="released R14 T route"),
        Line2D([0], [0], color="#c62828", lw=2.4, ls="--",
               label="requested direct T chord"),
    ), loc="lower right", fontsize=7.6, framealpha=0.95)

    return {
        "direct_length_mm": float(direct.length),
        "direct_outer_inside_xy": True,
        "lm_clearance_mm": float(lm_clear),
        "lm_required_mm": float(lm_required),
        "lm_station_mm": float(lm_station),
        "insert_clearance_mm": float(insert_clear),
        "insert_required_mm": float(insert_required),
        "insert_station_mm": float(insert_station),
    }


def _entry_cover_panel(ax):
    specs = (
        ("UM", np.asarray(route.route_cable_points(0.10)),
         route.CUTTER_R, route.MAIN_OUTER_R, "#2b8c62"),
        ("T", np.asarray(route.ts_cable_points(0.10)),
         route.TS_CUTTER_R, route.TS_OUTER_R, "#d89000"),
    )
    ax.axhspan(REAR_Z, FRONT_Z, color="#e1e5e9", zorder=0)
    ax.axhline(REAR_Z, color="#303840", lw=1.2)
    ax.axhline(FRONT_Z, color="#303840", lw=1.0)
    ax.text(46.0, REAR_Z + 0.25, "existing rear face z=5.3",
            fontsize=7.2, ha="right", va="bottom")
    max_projection = 0.0

    for label, points, lumen_r, outer_r, color in specs:
        station = _route_station(points)
        keep = station <= 45.0
        s = station[keep]
        z = points[keep, 2]
        lumen_lower = z - lumen_r
        shell_lower = z - outer_r
        max_projection = max(max_projection, REAR_Z - float(shell_lower.min()))
        ax.plot(s, z, color=color, lw=1.7, label=f"{label} centerline")
        ax.plot(s, lumen_lower, color=color, lw=1.1, ls="--")
        ax.fill_between(
            s, lumen_lower, REAR_Z, where=lumen_lower < REAR_Z,
            color="#d9534f", alpha=0.25, interpolate=True)
        ax.fill_between(
            s, shell_lower, lumen_lower, where=shell_lower < REAR_Z,
            color=color, alpha=0.28,
            label=f"{label} added rear saddle")

    ax.annotate(
        "red = present lumen is open to the rear\n"
        "beyond the intended circular mouth",
        xy=(8.0, 4.7), xytext=(16.0, 14.3), fontsize=7.7,
        arrowprops=dict(arrowstyle="->", color="#b23a3a"),
        bbox=dict(fc="white", ec="#b23a3a", alpha=0.92))
    ax.annotate(
        "colored shell = closed printed saddle\n"
        f"maximum added rear depth: {max_projection:.1f} mm",
        xy=(29.0, 2.8), xytext=(20.0, -2.0), fontsize=7.7,
        arrowprops=dict(arrowstyle="->", color="#52606d"),
        bbox=dict(fc="white", ec="#52606d", alpha=0.92))
    ax.set_title(
        "A — realistic closure of the exposed UM/T entry ramps",
        fontsize=10.0, weight="bold")
    ax.set_xlim(0, 47)
    ax.set_ylim(-3.2, 19.3)
    ax.set_xlabel("distance from rear mouth along released route (mm)")
    ax.set_ylabel("source z (mm)")
    ax.grid(True, color="#d7dbe0", lw=0.5)
    ax.legend(loc="upper right", fontsize=6.7, ncol=2, framealpha=0.95)


def _no_underpass_panel(ax, facts):
    lm_x = -7.5
    insert_x = 7.5
    body_w = 11.5

    for center in (lm_x, insert_x):
        ax.add_patch(Rectangle(
            (center - body_w / 2.0, REAR_Z), body_w,
            FRONT_Z - REAR_Z, fc="#dfe4e8", ec="#59636e", lw=1.0,
            zorder=0))

    # Actual rear-open vertical void spans.
    ax.add_patch(Rectangle(
        (lm_x - route.LM_INTERNAL_DUCT_R,
         route.NO_FLOOR_FEED_REAR_Z
         - route.NO_FLOOR_ENTRY_BORE_REAR_OVERTRAVEL_MM),
        2.0 * route.LM_INTERNAL_DUCT_R,
        LM_BORE_TOP_Z - (
            route.NO_FLOOR_FEED_REAR_Z
            - route.NO_FLOOR_ENTRY_BORE_REAR_OVERTRAVEL_MM),
        fc="white", ec="#6650a4", lw=1.8, zorder=3))
    ax.add_patch(Rectangle(
        (insert_x - BRIDGE_INSERT_D_MM / 2.0, REAR_Z),
        BRIDGE_INSERT_D_MM, INSERT_BORE_TOP_Z - REAR_Z,
        fc="white", ec="#34495e", lw=1.8, zorder=3))

    ax.text(lm_x, 9.6, "LM D9 bore\nz=5.05…13.80",
            color="#6650a4", fontsize=7.5, ha="center", va="center")
    ax.text(insert_x, 8.6, "blind insert D6.4\nz=5.30…12.10",
            color="#34495e", fontsize=7.5, ha="center", va="center")

    lm_front = FRONT_Z - LM_BORE_TOP_Z
    insert_front = FRONT_Z - INSERT_BORE_TOP_Z
    for x, z0, available in (
            (lm_x, LM_BORE_TOP_Z, lm_front),
            (insert_x, INSERT_BORE_TOP_Z, insert_front)):
        ax.annotate(
            "", xy=(x + 5.0, z0), xytext=(x + 5.0, FRONT_Z),
            arrowprops=dict(arrowstyle="<->", color="#303840", lw=1.0))
        ax.text(x + 5.4, (z0 + FRONT_Z) / 2.0,
                f"{available:.1f} mm\n< OD7.6",
                fontsize=7.0, ha="left", va="center")

    # What a Z-separated T passage would actually mean at the LM crossing.
    t_center_z = 1.0
    ax.add_patch(Circle(
        (lm_x, t_center_z), route.TS_OUTER_R,
        fc="#9fc5e8", ec="#1769aa", lw=1.4, zorder=4))
    ax.add_patch(Circle(
        (lm_x, t_center_z), route.TS_CUTTER_R,
        fc="white", ec="#d89000", lw=1.8, zorder=5))
    projection = REAR_Z - (t_center_z - route.TS_OUTER_R)
    ax.annotate(
        "", xy=(lm_x - 5.4, t_center_z - route.TS_OUTER_R),
        xytext=(lm_x - 5.4, REAR_Z),
        arrowprops=dict(arrowstyle="<->", color="#1769aa", lw=1.1))
    ax.text(lm_x - 5.8, (t_center_z - route.TS_OUTER_R + REAR_Z) / 2.0,
            f"{projection:.1f} mm\nrear growth", fontsize=7.1,
            color="#1769aa", ha="right", va="center")

    ax.axhline(REAR_Z, color="#303840", lw=1.2)
    ax.text(13.0, REAR_Z + 0.2, "existing rear plane", fontsize=7.2,
            ha="right", va="bottom")
    ax.text(
        0.0, -3.45,
        "The only Z-separated position is outside the existing part.  It "
        "would be a new rear raceway,\nnot a buried shortcut—and the LM "
        f"collision occurs only {facts['lm_station_mm']:.2f} mm from the T mouth.",
        fontsize=7.5, ha="center", va="bottom", color="#8e1b1b",
        bbox=dict(fc="#fff1f1", ec="#c62828", alpha=0.94))
    ax.set_title(
        "B — no hidden Z underpass exists inside the 13.0-mm bridge",
        fontsize=10.0, weight="bold")
    ax.set_xlim(-15.5, 15.5)
    ax.set_ylim(-4.0, 19.3)
    ax.set_xticks((lm_x, insert_x), ("LM crossing", "insert crossing"))
    ax.set_ylabel("source z (mm)")
    ax.grid(True, axis="y", color="#d7dbe0", lw=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("review/obiwan_entry_reroute_concept.png"))
    args = parser.parse_args()

    fig = plt.figure(figsize=(13.0, 8.8), dpi=170, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.16, 0.84))
    ax_plan = fig.add_subplot(gs[:, 0])
    ax_entry = fig.add_subplot(gs[0, 1])
    ax_under = fig.add_subplot(gs[1, 1])
    facts = _plan_panel(ax_plan)
    _entry_cover_panel(ax_entry)
    _no_underpass_panel(ax_under, facts)
    fig.suptitle(
        "LX521 Obi-Wan no-floor cable-entry review — corrected physical constraints",
        fontsize=13.0, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)
    for key, value in facts.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
