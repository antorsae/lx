#!/usr/bin/env python3
"""Dimensioned plan comparison for alternative Obi-Wan rear entries.

Review-only: this script consumes released source datums and performs planar
envelope checks.  It does not alter production CAD.  Z bypasses at the LM
pilot pads and rear-mouth burial remain separate qualification items.
"""

from __future__ import annotations

import argparse
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
from top_baffle_nd25fw4_flush import LM_PILOT_XY, PAD_D_MM
from top_baffle_nd25fw4_obiwan_bridge import bridge_face_plan
import top_baffle_nd25fw4_obiwan_route as route


COLORS = {"T": "#c97b00", "LM": "#6b4ca5", "UM": "#20845a"}
LUMEN_R = {
    "T": route.TS_CUTTER_R,
    "LM": route.LM_INTERNAL_DUCT_R,
    "UM": route.CUTTER_R,
}
OUTER_R = {
    "T": route.TS_OUTER_R,
    "LM": route.LM_INTERNAL_DUCT_R + route.TUNNEL_SKIN,
    "UM": route.MAIN_OUTER_R,
}
TARGET = {
    "T": np.asarray(route._TS_LM_ARC_START, dtype=float),
    "LM": np.asarray(route._LM_INTERNAL_EXIT_XY, dtype=float),
    "UM": np.asarray(route._MAIN_ARC_START, dtype=float),
}
TARGET_BEARING = {"T": 135.0, "LM": 90.0, "UM": 45.0}


def material_plan():
    center = L22_CUTOUT[:2]
    outer = Point(*center).buffer(route.LM_CORE_R, resolution=384)
    opening = Point(*center).buffer(L22_CUTOUT[2] / 2.0, resolution=384)
    return unary_union((bridge_face_plan(), outer)).difference(opening)


def fill_geometry(ax, geometry, facecolor, edgecolor, alpha, zorder):
    polygons = ([geometry] if geometry.geom_type == "Polygon"
                else list(geometry.geoms))
    for polygon in polygons:
        xy = np.asarray(polygon.exterior.coords, dtype=float)
        ax.fill(xy[:, 0], xy[:, 1], fc=facecolor, ec=edgecolor,
                alpha=alpha, lw=0.9, zorder=zorder)
        for ring in polygon.interiors:
            hole = np.asarray(ring.coords, dtype=float)
            ax.fill(hole[:, 0], hole[:, 1], fc="white", ec=edgecolor,
                    lw=0.9, zorder=zorder + 0.1)


def bearing(line):
    p0 = np.asarray(line.coords[0], dtype=float)
    p1 = np.asarray(line.coords[-1], dtype=float)
    return math.degrees(math.atan2(*(p1 - p0)[::-1])) % 360.0


def mismatch_deg(actual, target):
    return abs((actual - target + 180.0) % 360.0 - 180.0)


def candidate_lines(entries):
    return {name: LineString((tuple(entries[name]), tuple(TARGET[name])))
            for name in ("T", "LM", "UM")}


def validate_candidate(name, entries, lines, body, required_pair_wall_mm):
    facts = {}
    for route_name, line in lines.items():
        envelope = line.buffer(
            OUTER_R[route_name], resolution=48,
            cap_style=1, join_style=1)
        outside = envelope.difference(body).area
        if outside > 1.0e-6:
            raise RuntimeError(
                f"{name} {route_name} leaves allowed plan by "
                f"{outside:.6f} mm2")
        insert_margin = min(
            line.distance(Point(*xy))
            - (BRIDGE_INSERT_D_MM / 2.0 + OUTER_R[route_name])
            for xy in BRIDGE_HOLE_XY)
        if insert_margin < -1.0e-6:
            raise RuntimeError(
                f"{name} {route_name} violates bridge insert by "
                f"{-insert_margin:.3f} mm")
        facts[route_name] = {
            "length_mm": float(line.length),
            "bearing_deg": bearing(line),
            "target_tangent_mismatch_deg": mismatch_deg(
                bearing(line), TARGET_BEARING[route_name]),
            "bridge_insert_outer_margin_mm": float(insert_margin),
        }

    pair_gaps = {}
    pair_margins = {}
    for left, right in (("T", "LM"), ("LM", "UM"), ("T", "UM")):
        distance = lines[left].distance(lines[right])
        raw_gap = distance - LUMEN_R[left] - LUMEN_R[right]
        margin = raw_gap - required_pair_wall_mm
        if margin < -1.0e-6:
            raise RuntimeError(
                f"{name} {left}/{right} lumen wall is short by "
                f"{-margin:.3f} mm")
        pair_gaps[f"{left}_{right}"] = float(raw_gap)
        pair_margins[f"{left}_{right}"] = float(margin)
    facts["required_pair_wall_mm"] = float(required_pair_wall_mm)
    facts["pair_raw_lumen_gap_mm"] = pair_gaps
    facts["pair_lumen_margin_after_required_wall_mm"] = pair_margins
    return facts


def draw_hardware(ax):
    for xy in BRIDGE_HOLE_XY:
        ax.add_patch(Circle(
            xy, BRIDGE_INSERT_D_MM / 2.0, fc="white", ec="#34495e",
            lw=1.0, zorder=8))
    for index in (4, 5):
        xy = LM_PILOT_XY[index]
        ax.add_patch(Circle(
            xy, PAD_D_MM / 2.0, fc="#fff2cc", ec="#ad7d00",
            lw=1.0, ls="--", zorder=7))
        ax.annotate(
            "existing LM pilot\nZ-bypass retained", xy,
            xytext=(0, 9), textcoords="offset points", ha="center",
            fontsize=6.6, color="#7a5900", zorder=12)


def draw_candidate(ax, title, entries, lines, body, facts, note):
    fill_geometry(ax, body, "#d4d9df", "#505a64", 1.0, 0)
    draw_hardware(ax)
    label_offsets = {
        "T": (5, -10),
        "LM": (-36, 10),
        "UM": (7, -14),
    }
    for name in ("T", "LM", "UM"):
        line = lines[name]
        envelope = line.buffer(
            OUTER_R[name], resolution=48, cap_style=1, join_style=1)
        fill_geometry(ax, envelope, COLORS[name], COLORS[name], 0.18, 3)
        xy = np.asarray(line.coords, dtype=float)
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS[name], lw=2.1,
                zorder=9)
        entry = np.asarray(entries[name], dtype=float)
        ax.add_patch(Circle(
            entry, LUMEN_R[name], fc="white", ec=COLORS[name],
            lw=1.8, zorder=11))
        ax.annotate(
            f"{name} ({entry[0]:.1f}, {entry[1]:.1f})",
            entry, xytext=label_offsets[name], textcoords="offset points",
            fontsize=7.0, color=COLORS[name], zorder=13)

    min_pair = min(facts["pair_raw_lumen_gap_mm"].values())
    ax.text(
        -86.0, 139.0,
        note + "\n"
        f"minimum raw plan gap between lumen envelopes: {min_pair:.2f} mm",
        fontsize=7.3, ha="left", va="top",
        bbox=dict(fc="white", ec="#5b6570", alpha=0.94), zorder=15)
    ax.set_title(title, fontsize=9.4, weight="bold")
    ax.set_xlim(-90, 90)
    ax.set_ylim(35, 145)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, color="#d7dbe0", lw=0.45)


def current_minimal_lines():
    entries = {
        "T": np.asarray((-32.0, 85.0)),
        "LM": np.asarray(route.NO_FLOOR_LM_FEED_XY, dtype=float),
        "UM": np.asarray(route.NO_FLOOR_MAIN_FEED_XY, dtype=float),
    }
    return entries, {
        "T": LineString((tuple(entries["T"]), tuple(TARGET["T"]))),
        "LM": LineString(np.asarray(route._LM_INTERNAL_PLAN, dtype=float)),
        "UM": LineString(np.asarray(route._MAIN_ENTRY, dtype=float)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("review/obiwan_entry_layout_alternatives.png"))
    args = parser.parse_args()
    body = material_plan()

    layouts = []
    minimal_entries, minimal_lines = current_minimal_lines()
    layouts.append((
        "A — minimal change (recommended first)",
        minimal_entries,
        minimal_lines,
        "Move only T; released LM and UM entries/routes stay unchanged.",
    ))
    symmetric_entries = {
        "T": np.asarray((-32.0, 85.0)),
        "LM": np.asarray((-8.0, 60.0)),
        "UM": np.asarray((32.0, 85.0)),
    }
    layouts.append((
        "B — robust symmetric fan",
        symmetric_entries,
        candidate_lines(symmetric_entries),
        "Straight fan; largest bridge-insert and inter-route margins.",
    ))
    compact_entries = {
        "T": np.asarray((-28.0, 80.0)),
        "LM": np.asarray((-8.0, 60.0)),
        "UM": np.asarray((28.0, 80.0)),
    }
    layouts.append((
        "C — compact upper fan",
        compact_entries,
        candidate_lines(compact_entries),
        "Mouths stay closer together; less shoulder/insert margin than B.",
    ))

    fig, axes = plt.subplots(
        1, 3, figsize=(15.2, 5.5), dpi=170, constrained_layout=True)
    all_facts = {}
    for ax, (title, entries, lines, note) in zip(axes, layouts):
        required_wall = 0.0 if title.startswith("A ") else route.TUNNEL_SKIN
        facts = validate_candidate(
            title, entries, lines, body, required_wall)
        all_facts[title] = facts
        draw_candidate(ax, title, entries, lines, body, facts, note)
    fig.suptitle(
        "LX521 Obi-Wan no-floor rear-entry alternatives — actual allowed plan",
        fontsize=13.0, weight="bold")
    fig.text(
        0.5, 0.008,
        "Plan screening only. Yellow pilot crossings retain the existing "
        "covered Z-bypass architecture. Rear-mouth burial remains a separate "
        "saddle/side-entry decision.",
        ha="center", va="bottom", fontsize=8.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)
    for layout_name, facts in all_facts.items():
        print(layout_name)
        for route_name in ("T", "LM", "UM"):
            print(f"  {route_name}: {facts[route_name]}")
        print(
            "  raw lumen gaps:", facts["pair_raw_lumen_gap_mm"])


if __name__ == "__main__":
    main()
