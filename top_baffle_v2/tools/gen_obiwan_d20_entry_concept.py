#!/usr/bin/env python3
"""Dimensioned D20 rear-entry concept for the Obi-Wan no-floor bridge.

Review-only.  The three released lumen diameters are packed inside the stock
support plate's D20 cable window in the requested order::

       LM
     T    UM

The ports are rear-normal.  Only the buried continuations fan tangentially
toward the released LM-ring route datums.  This script does not modify or
qualify production CAD.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from top_baffle_nd25fw4 import (
    BRIDGE_HOLE_XY,
    BRIDGE_INSERT_D_MM,
    L22_CUTOUT,
)
from top_baffle_nd25fw4_cables import SUPPORT_WINDOW
from top_baffle_nd25fw4_obiwan_bridge import bridge_face_plan
import top_baffle_nd25fw4_obiwan_route as route


COLORS = {"T": "#c97b00", "LM": "#6b4ca5", "UM": "#20845a"}
ORDER = ("LM", "T", "UM")
LUMEN_D = {
    "LM": 2.0 * float(route.LM_INTERNAL_DUCT_R),
    "T": 2.0 * float(route.TS_CUTTER_R),
    "UM": 2.0 * float(route.CUTTER_R),
}
LUMEN_R = {name: diameter / 2.0 for name, diameter in LUMEN_D.items()}
OUTER_R = {
    "LM": LUMEN_R["LM"] + float(route.TUNNEL_SKIN),
    "T": float(route.TS_OUTER_R),
    "UM": float(route.MAIN_OUTER_R),
}
REQUIRED_SHARED_WEB_MM = float(route.TUNNEL_SKIN)

# Coordinates are relative to the exact stock support-window centre.  This is
# the max-clearance horizontal-row packing with LM fixed on the vertical axis,
# rounded outward just enough to keep both limiting shared webs >= 0.80 mm.
PORT_LOCAL_XY = {
    "LM": np.asarray((0.00, 4.76), dtype=float),
    "T": np.asarray((-4.75, -4.09), dtype=float),
    "UM": np.asarray((3.17, -4.09), dtype=float),
}


@dataclass(frozen=True)
class RouteRecord:
    name: str
    port_xy: np.ndarray
    path: np.ndarray
    target_xy: np.ndarray
    min_plan_radius_mm: float


def _direction(bearing_deg: float) -> np.ndarray:
    angle = math.radians(bearing_deg)
    return np.asarray((math.cos(angle), math.sin(angle)), dtype=float)


def _bezier(p0, p1, p2, p3, count=1201):
    u = np.linspace(0.0, 1.0, count)[:, None]
    v = 1.0 - u
    return (v ** 3 * p0 + 3.0 * v ** 2 * u * p1
            + 3.0 * v * u ** 2 * p2 + u ** 3 * p3)


def _stations(points):
    return np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def _splice(path, station_mm):
    station = _stations(path)
    index = int(np.searchsorted(station, station_mm))
    lo = max(0, index - 5)
    hi = min(len(path) - 1, index + 5)
    tangent = path[hi] - path[lo]
    bearing = math.degrees(math.atan2(tangent[1], tangent[0])) % 360.0
    return index, np.asarray(path[index], dtype=float), bearing


def _min_three_point_radius(points):
    first = points[1:-1] - points[:-2]
    second = points[2:] - points[1:-1]
    chord = points[2:] - points[:-2]
    cross = np.abs(first[:, 0] * second[:, 1]
                   - first[:, 1] * second[:, 0])
    radius = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(chord, axis=1)
        / np.maximum(2.0 * cross, 1.0e-12))
    radius[cross < 1.0e-10] = 1.0e12
    return float(radius.min())


def _material_plan():
    center = L22_CUTOUT[:2]
    outer = Point(*center).buffer(route.LM_CORE_R, resolution=384)
    opening = Point(*center).buffer(L22_CUTOUT[2] / 2.0, resolution=384)
    return unary_union((bridge_face_plan(), outer)).difference(opening)


def _build_routes():
    window_center = np.asarray(SUPPORT_WINDOW[:2], dtype=float)
    ports = {
        name: window_center + PORT_LOCAL_XY[name]
        for name in ORDER
    }

    # Preserve the already-developed ring-side route tails.  Only their entry
    # throats are rebased to the D20 cluster.  The tangent-matched splice
    # stations were selected to retain R >= 14 mm in the replacement plans.
    t_index, t_join, t_bearing = _splice(route._TS_ENTRY, 50.0)
    t_throat = _bezier(
        ports["T"],
        ports["T"] + 12.0 * _direction(135.0),
        t_join - 12.0 * _direction(t_bearing),
        t_join,
    )
    t_path = np.vstack((t_throat[:-1], route._TS_ENTRY[t_index:]))

    um_index, um_join, um_bearing = _splice(route._MAIN_ENTRY, 95.0)
    um_throat = _bezier(
        ports["UM"],
        ports["UM"] + 28.0 * _direction(15.0),
        um_join - 32.0 * _direction(um_bearing),
        um_join,
    )
    um_path = np.vstack((um_throat[:-1], route._MAIN_ENTRY[um_index:]))

    lm_target = np.asarray(route._LM_INTERNAL_EXIT_XY, dtype=float)
    lm_path = _bezier(
        ports["LM"],
        ports["LM"] + 8.0 * _direction(140.0),
        lm_target - 8.0 * _direction(90.0),
        lm_target,
    )

    paths = {"LM": lm_path, "T": t_path, "UM": um_path}
    targets = {
        "LM": lm_target,
        "T": np.asarray(route._TS_LM_ARC_START, dtype=float),
        "UM": np.asarray(route._MAIN_ARC_START, dtype=float),
    }
    return tuple(RouteRecord(
        name=name,
        port_xy=ports[name],
        path=paths[name],
        target_xy=targets[name],
        min_plan_radius_mm=_min_three_point_radius(paths[name]),
    ) for name in ORDER)


def _validate(records, body):
    window_center = np.asarray(SUPPORT_WINDOW[:2], dtype=float)
    window_r = float(SUPPORT_WINDOW[2]) / 2.0
    facts = {}
    for record in records:
        local = record.port_xy - window_center
        rim = window_r - np.linalg.norm(local) - LUMEN_R[record.name]
        if rim < -1.0e-9:
            raise RuntimeError(
                f"{record.name} aperture leaves D20 by {-rim:.3f} mm")
        if LineString(record.path).difference(body).length > 1.0e-6:
            raise RuntimeError(f"{record.name} centerline leaves source body")
        if record.min_plan_radius_mm < 14.0 - 0.02:
            raise RuntimeError(
                f"{record.name} plan R {record.min_plan_radius_mm:.3f} < 14")
        facts[record.name] = {
            "local_xy_mm": tuple(map(float, local)),
            "world_xy_mm": tuple(map(float, record.port_xy)),
            "diameter_mm": LUMEN_D[record.name],
            "d20_rim_clearance_mm": float(rim),
            "minimum_plan_radius_mm": record.min_plan_radius_mm,
        }

    by_name = {record.name: record for record in records}
    pair_gaps = {}
    for left, right in (("LM", "T"), ("LM", "UM"), ("T", "UM")):
        gap = (
            LineString(by_name[left].path).distance(
                LineString(by_name[right].path))
            - LUMEN_R[left] - LUMEN_R[right]
        )
        if gap < REQUIRED_SHARED_WEB_MM - 0.01:
            raise RuntimeError(
                f"{left}/{right} shared web {gap:.3f} < "
                f"{REQUIRED_SHARED_WEB_MM:.3f}")
        pair_gaps[f"{left}_{right}"] = float(gap)
    facts["pair_shared_web_mm"] = pair_gaps
    return facts


def _fill_geometry(ax, geometry, facecolor, edgecolor, alpha, zorder):
    polygons = ([geometry] if geometry.geom_type == "Polygon"
                else list(geometry.geoms))
    for polygon in polygons:
        xy = np.asarray(polygon.exterior.coords, dtype=float)
        ax.fill(xy[:, 0], xy[:, 1], fc=facecolor, ec=edgecolor,
                alpha=alpha, lw=1.0, zorder=zorder)
        for ring in polygon.interiors:
            hole = np.asarray(ring.coords, dtype=float)
            ax.fill(hole[:, 0], hole[:, 1], fc="white", ec=edgecolor,
                    lw=1.0, zorder=zorder + 0.1)


def _packing_panel(ax, records, facts):
    window_r = float(SUPPORT_WINDOW[2]) / 2.0
    ax.add_patch(Circle(
        (0.0, 0.0), window_r, fc="#22272e", ec="#111111", lw=2.0,
        zorder=0))
    ax.add_patch(Circle(
        (0.0, 0.0), window_r, fc="none", ec="#111111", lw=1.4,
        ls="--", zorder=8))
    for record in records:
        name = record.name
        local = np.asarray(facts[name]["local_xy_mm"])
        ax.add_patch(Circle(
            local, LUMEN_R[name], fc="white", ec=COLORS[name],
            lw=2.4, zorder=4))
        ax.text(local[0], local[1], f"{name}\nD{LUMEN_D[name]:g}",
                ha="center", va="center", fontsize=10.0,
                weight="bold", color=COLORS[name], zorder=7)

    ax.annotate(
        "D20 support opening", xy=(0.0, 10.0), xytext=(0.0, 13.2),
        ha="center", fontsize=9.2, weight="bold",
        arrowprops=dict(arrowstyle="-[,widthB=3.8", lw=1.1,
                        color="#30343b"))
    ax.text(
        0.0, -14.3,
        "All three complete circular apertures remain inside D20.",
        ha="center", va="bottom", fontsize=8.0,
        bbox=dict(fc="white", ec="#4b5560", alpha=0.94))
    ax.set_title(
        "REAR FACE — required D20 packing",
        fontsize=10.7, weight="bold")
    ax.set_xlim(-15.5, 15.5)
    ax.set_ylim(-15.5, 15.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x relative to D20 centre (mm)")
    ax.set_ylabel("y relative to D20 centre (mm)")
    ax.grid(True, color="#d8dce1", lw=0.45)


def _plan_panel(ax, body, records):
    _fill_geometry(ax, body, "#d4d9df", "#505a64", 1.0, 0)
    window_center = np.asarray(SUPPORT_WINDOW[:2], dtype=float)
    ax.add_patch(Circle(
        window_center, SUPPORT_WINDOW[2] / 2.0,
        fc="#252a31", ec="#111111", lw=1.5, zorder=3))
    for xy in BRIDGE_HOLE_XY:
        ax.add_patch(Circle(
            xy, BRIDGE_INSERT_D_MM / 2.0, fc="white", ec="#34495e",
            lw=1.1, zorder=9))

    for record in records:
        name = record.name
        color = COLORS[name]
        line = LineString(record.path)
        lumen = line.buffer(
            LUMEN_R[name], resolution=48, cap_style=1, join_style=1)
        _fill_geometry(ax, lumen, color, color, 0.26, 4)
        ax.plot(record.path[:, 0], record.path[:, 1], color=color,
                lw=1.7, zorder=8)
        ax.add_patch(Circle(
            record.port_xy, LUMEN_R[name], fc="white", ec=color,
            lw=1.7, zorder=11))
        ax.text(record.port_xy[0], record.port_xy[1], name,
                ha="center", va="center", fontsize=6.6,
                weight="bold", color=color, zorder=12)
        ax.scatter([record.target_xy[0]], [record.target_xy[1]],
                   s=22, fc=color, ec="white", lw=0.6, zorder=10)

    ax.annotate(
        "three rear-normal mouths\ninside the same D20",
        window_center, xytext=(29.0, 51.0), fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color="#303840"),
        bbox=dict(fc="white", ec="#59636e", alpha=0.94), zorder=15)
    ax.annotate(
        "upper insert crossings remain\ncovered Z-underpasses",
        (-20.0, 70.0), xytext=(-84.0, 86.0), fontsize=7.4,
        color="#725b00",
        arrowprops=dict(arrowstyle="->", color="#8a6d00"),
        bbox=dict(fc="#fff8dd", ec="#ad8700", alpha=0.94), zorder=15)
    ax.text(
        0.0, 42.0,
        "rear-normal entry first; smooth buried fan begins behind the face",
        ha="center", va="center", fontsize=7.5,
        bbox=dict(fc="white", ec="#59636e", alpha=0.94), zorder=15)
    ax.set_title(
        "SOURCE PLAN — buried fan to released ring-route tails",
        fontsize=10.7, weight="bold")
    ax.set_xlim(-92, 92)
    ax.set_ylim(38, 145)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (mm)")
    ax.set_ylabel("world y (mm)")
    ax.grid(True, color="#d8dce1", lw=0.45)


def _facts_panel(ax, records, facts):
    ax.axis("off")
    rows = []
    for record in records:
        fact = facts[record.name]
        rows.append([
            record.name,
            f"D{fact['diameter_mm']:g}",
            (f"({fact['local_xy_mm'][0]:+.2f}, "
             f"{fact['local_xy_mm'][1]:+.2f})"),
            f"{fact['d20_rim_clearance_mm']:.3f}",
            f"{fact['minimum_plan_radius_mm']:.1f}",
        ])
    table = ax.table(
        cellText=rows,
        colLabels=("duct", "lumen", "centre in D20 (mm)",
                   "rim clearance", "min plan R"),
        cellLoc="center", colLoc="center", loc="upper center",
        bbox=(0.0, 0.70, 1.0, 0.29),
        colWidths=(0.10, 0.13, 0.28, 0.23, 0.18),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.55)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#6c757d")
        if row == 0:
            cell.set_facecolor("#e5e9ed")
            cell.set_text_props(weight="bold")

    gaps = facts["pair_shared_web_mm"]
    text = (
        "Shared material between lumen openings\n"
        f"LM–T: {gaps['LM_T']:.3f} mm\n"
        f"LM–UM: {gaps['LM_UM']:.3f} mm\n"
        f"T–UM: {gaps['T_UM']:.3f} mm\n\n"
        f"Required minimum: {REQUIRED_SHARED_WEB_MM:.2f} mm\n"
        "Limiting pair: LM–UM\n\n"
        "Interpretation\n"
        "• D20 is the only exterior cable opening.\n"
        "• LM is above; T lower-left; UM lower-right.\n"
        "• Entry axes are normal to the rear face.\n"
        "• Tangency belongs to the buried fan/ring handoff.\n"
        "• Insert crossings require the existing closed Z-bypass.\n\n"
        "Review concept only: full 3D BREP, support-free roofs, insertion "
        "access and slicer continuity remain to be qualified."
    )
    ax.text(
        0.04, 0.63, text, transform=ax.transAxes,
        ha="left", va="top", fontsize=7.8, linespacing=1.28,
        bbox=dict(fc="#f8f9fa", ec="#59636e", alpha=0.98,
                  boxstyle="round,pad=0.55"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("review/obiwan_d20_entry_concept.png"))
    args = parser.parse_args()

    body = _material_plan()
    records = _build_routes()
    facts = _validate(records, body)

    fig = plt.figure(figsize=(13.8, 8.4), dpi=175, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.15, 0.85),
                          height_ratios=(1.02, 0.98))
    ax_plan = fig.add_subplot(gs[:, 0])
    ax_pack = fig.add_subplot(gs[0, 1])
    ax_facts = fig.add_subplot(gs[1, 1])
    _plan_panel(ax_plan, body, records)
    _packing_panel(ax_pack, records, facts)
    _facts_panel(ax_facts, records, facts)
    fig.suptitle(
        "LX521 Obi-Wan no-floor — corrected D20 three-duct entry",
        fontsize=13.2, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(args.output)
    for name in ORDER:
        print(name, facts[name])
    print("pair_shared_web_mm", facts["pair_shared_web_mm"])


if __name__ == "__main__":
    main()
