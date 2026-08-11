#!/usr/bin/env python3
"""Generate the measured Stock-vase TEBM35C10-4 layout concept.

This is a review drawing, not production CAD.  The released B2 outline and
UM datums come from the project sources.  TEBM envelope dimensions come from
the manufacturer outline drawing; the published cutout and mounting PCD come
from the Parts Express product listing.  Insert and magnet cavities are the
requested/released hardware envelopes, not slicer-qualified features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from gen_driver_overlay import outline_polygon
from lx521_baffle.base import THICKNESS_MM, UM_CUTOUT
from lx521_baffle.magnet_contract import (
    CAVITY_DEPTH_MM,
    CAVITY_DIAMETER_MM,
    FACE_SKIN_MM,
    MAGNET_DEPTH_MM,
    MAGNET_DIAMETER_MM,
)
from lx521_baffle.proud.b import (
    MAGNET_SITES,
    STANDARD_MAGNET_Z_MM,
    TWEETER_DROP_MM,
)
from lx521_baffle.proud.b2 import OUTLINE_B2
from lx521_baffle.proud.b2_split import SEAM_B_Y
from lx521_baffle.proud.vase_tebm35c10_4 import (
    TEBM_LAND_D_MM,
    T_MAGNET_FACE_X_MM,
    T_MAGNET_FLAT_EDGE_MARGIN_MM,
    T_MAGNET_FLAT_HALF_HEIGHT_MM,
)


# Released Stock geometry.
UM_AXIS_Y_MM = float(UM_CUTOUT[1])
UM_OPENING_D_MM = float(UM_CUTOUT[2])
UM_FLANGE_D_MM = 97.5
RELEASED_FRONT_T_AXIS_Y_MM = 483.78 - TWEETER_DROP_MM
RELEASED_REAR_T_AXIS_Y_MM = 549.05 - TWEETER_DROP_MM

# Tectonic Audio Labs TEBM35C10-4 outline drawing, rev. 1.0.
TEBM_NOMINAL_D_MM = 52.0
TEBM_MAX_D_MM = 54.0
TEBM_BASKET_D_MM = 43.6
TEBM_DEPTH_MM = 25.1
TEBM_MASS_G = 51.3
# Published product-listing interface dimensions.
TEBM_CUTOUT_D_MM = 1.69 * 25.4
TEBM_MOUNT_PCD_MM = 1.90 * 25.4
TEBM_MOUNT_HOLE_COUNT = 4

# Review assumptions.  These are intentionally visible in the PNG.
PRESERVED_FRONT_EDGE_GAP_MM = 2.10
BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM = 0.50
TEBM_BORE_D_MM = TEBM_CUTOUT_D_MM
M2_INSERT_BORE_D_MM = 3.2
M2_INSERT_DEPTH_MM = 4.0
M2_INSERTS_PER_DRIVER = TEBM_MOUNT_HOLE_COUNT
M2_INSERT_TOTAL = 2 * M2_INSERTS_PER_DRIVER
M2_INSERT_PROJECTED_OFFSET_MM = (
    TEBM_MOUNT_PCD_MM / 2.0 * math.sin(math.radians(45.0))
)
M2_INSERT_INNER_LIGAMENT_MM = (
    TEBM_MOUNT_PCD_MM / 2.0
    - M2_INSERT_BORE_D_MM / 2.0
    - TEBM_BORE_D_MM / 2.0
)
M2_INSERT_OUTER_LAND_MM = (
    TEBM_LAND_D_MM / 2.0
    - TEBM_MOUNT_PCD_MM / 2.0
    - M2_INSERT_BORE_D_MM / 2.0
)

# One released-size captive D5x2 station at the left and right outer edge of
# each T land.  Production CAD trims the D63 lands to planar side
# faces, including 0.10 mm beyond the captive helper's required half-height.
# The cavity axis is +/-X and retains the released 0.45-mm face skin.
T_MAGNETS_PER_DRIVER = 2
T_MAGNET_TOTAL = 2 * T_MAGNETS_PER_DRIVER
T_MAGNET_CENTER_X_MM = (
    T_MAGNET_FACE_X_MM
    - FACE_SKIN_MM
    - CAVITY_DEPTH_MM / 2.0
)
T_MAGNET_CAVITY_TO_BORE_LAND_MM = (
    T_MAGNET_CENTER_X_MM
    - CAVITY_DEPTH_MM / 2.0
    - TEBM_BORE_D_MM / 2.0
)
LOWER_T_AXIS_Y_MM = (
    UM_AXIS_Y_MM
    + UM_FLANGE_D_MM / 2.0
    + TEBM_MAX_D_MM / 2.0
    + PRESERVED_FRONT_EDGE_GAP_MM
)
# This cross-face condition is stricter than simply spacing two basket bores:
# one full D54 flange can meet the other driver's D43.6 maximum body because
# each 25.1-mm body crosses the opposite face of the 18.3-mm plate.
PAIR_AXIS_PITCH_MM = (
    TEBM_MAX_D_MM / 2.0
    + TEBM_BASKET_D_MM / 2.0
    + BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM
)
UPPER_T_AXIS_Y_MM = LOWER_T_AXIS_Y_MM + PAIR_AXIS_PITCH_MM
BORE_WEB_MM = PAIR_AXIS_PITCH_MM - TEBM_BORE_D_MM
FLANGE_PROJECTION_OVERLAP_MM = TEBM_MAX_D_MM - PAIR_AXIS_PITCH_MM
T_ZONE_DEPTH_MM = TEBM_DEPTH_MM
REAR_GROWTH_MM = T_ZONE_DEPTH_MM - THICKNESS_MM
REAR_T_MOUNT_Z_MM = -REAR_GROWTH_MM
OPPOSITE_FACE_PROTRUSION_MM = 0.0
ASSEMBLY_DEPTH_ENVELOPE_MM = T_ZONE_DEPTH_MM
# Grow only the rear face.  The exact released B2 flare crest provides a
# broad source datum; the sweep reaches full depth at the lower BMR opening.
# Quintic smootherstep gives zero slope and zero curvature at both ends, so
# neither junction contains a bevel/corner angle.
REAR_RAMP_START_Y_MM = 391.709
REAR_RAMP_END_Y_MM = LOWER_T_AXIS_Y_MM - TEBM_BORE_D_MM / 2.0
REAR_RAMP_LENGTH_MM = REAR_RAMP_END_Y_MM - REAR_RAMP_START_Y_MM


COLORS = {
    "ink": "#192433",
    "muted": "#6d7b8d",
    "grid": "#d9e0e8",
    "existing": "#9aa5b1",
    "proposal": "#dcecf5",
    "proposal_edge": "#2c5d79",
    "um": "#d1495b",
    "front": "#168b83",
    "rear": "#7654a6",
    "insert": "#f2b84b",
    "magnet": "#d48806",
    "warning": "#b15a12",
}


def _draw_geometry(ax, geometry, *, facecolor, edgecolor, alpha=1.0,
                   linewidth=1.2, linestyle="-", zorder=1):
    if geometry.is_empty:
        return
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        xy = np.asarray(polygon.exterior.coords, dtype=float)
        ax.fill(
            xy[:, 0], xy[:, 1], facecolor=facecolor, edgecolor=edgecolor,
            alpha=alpha, linewidth=linewidth, linestyle=linestyle,
            zorder=zorder,
        )
        for ring in polygon.interiors:
            hole = np.asarray(ring.coords, dtype=float)
            ax.fill(
                hole[:, 0], hole[:, 1], facecolor="white",
                edgecolor=edgecolor, linewidth=linewidth,
                linestyle=linestyle, zorder=zorder + 0.1,
            )


def _dimension_vertical(ax, x, y0, y1, label, *, color=None,
                        label_dx=2.0, zorder=30):
    color = color or COLORS["ink"]
    ax.annotate(
        "", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.25),
        zorder=zorder,
    )
    ax.text(
        x + label_dx, (y0 + y1) / 2.0, label, color=color,
        rotation=90, ha="left", va="center", fontsize=8.2,
        bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2),
        zorder=zorder + 1,
    )


def _dimension_horizontal(ax, y, x0, x1, label, *, color=None,
                          label_dy=1.6, zorder=30):
    color = color or COLORS["ink"]
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.25),
        zorder=zorder,
    )
    ax.text(
        (x0 + x1) / 2.0, y + label_dy, label, color=color,
        ha="center", va="bottom", fontsize=8.2,
        bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2),
        zorder=zorder + 1,
    )


def _smootherstep01(value):
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return value ** 3 * (value * (value * 6.0 - 15.0) + 10.0)


def _rear_surface_z(y):
    station = (
        (np.asarray(y, dtype=float) - REAR_RAMP_START_Y_MM)
        / REAR_RAMP_LENGTH_MM
    )
    return -REAR_GROWTH_MM * _smootherstep01(station)


def _mounting_hole_centers(axis_y, clock_deg):
    radius = TEBM_MOUNT_PCD_MM / 2.0
    return [
        (
            radius * math.cos(math.radians(clock_deg + 90.0 * index)),
            axis_y
            + radius * math.sin(math.radians(clock_deg + 90.0 * index)),
        )
        for index in range(TEBM_MOUNT_HOLE_COUNT)
    ]


def _validate_dimension_contract():
    magnet_minimum_skin = (
        T_MAGNET_FACE_X_MM
        - (T_MAGNET_CENTER_X_MM + CAVITY_DEPTH_MM / 2.0)
    )
    if not math.isclose(
            magnet_minimum_skin, FACE_SKIN_MM, abs_tol=1e-9):
        raise RuntimeError(
            "T magnet center no longer retains the planar face skin")
    if M2_INSERT_INNER_LIGAMENT_MM <= 0.0:
        raise RuntimeError("M2 insert bores intersect the T cutout")
    if M2_INSERT_OUTER_LAND_MM <= 0.0:
        raise RuntimeError("M2 insert bores escape the proposed T land")
    if T_MAGNET_CAVITY_TO_BORE_LAND_MM <= 0.0:
        raise RuntimeError("T magnet cavities intersect the T cutout")
    if BORE_WEB_MM <= 0.0 or REAR_RAMP_LENGTH_MM <= 0.0:
        raise RuntimeError("driver spacing or rear ramp is invalid")


def _driver_front(ax, axis_y, *, color, linestyle, clock_deg, label,
                  zorder=12):
    # The nominal/max rings describe the flange; the body and published
    # recommended cutout are intentionally separate.  Gold circles are the
    # requested blind insert bores on the published four-hole PCD.
    ax.add_patch(Circle(
        (0.0, axis_y), TEBM_MAX_D_MM / 2.0,
        fc="none", ec=color, lw=1.0, ls=(0, (2, 2)), zorder=zorder,
    ))
    ax.add_patch(Circle(
        (0.0, axis_y), TEBM_NOMINAL_D_MM / 2.0,
        fc="none", ec=color, lw=1.8, ls=linestyle, zorder=zorder,
    ))
    ax.add_patch(Circle(
        (0.0, axis_y), TEBM_BORE_D_MM / 2.0,
        fc="white", ec=COLORS["ink"], lw=1.15, zorder=zorder + 0.2,
    ))
    ax.add_patch(Circle(
        (0.0, axis_y), TEBM_BASKET_D_MM / 2.0,
        fc="none", ec=color, lw=0.9, ls=(0, (2, 2)), zorder=zorder + 0.3,
    ))
    for x, y in _mounting_hole_centers(axis_y, clock_deg):
        ax.add_patch(Circle(
            (x, y), M2_INSERT_BORE_D_MM / 2.0,
            fc=COLORS["insert"], ec=COLORS["ink"], lw=0.8,
            zorder=zorder + 0.5,
        ))
    ax.scatter([0.0], [axis_y], s=18, color=color, zorder=zorder + 1)
    ax.text(
        -36.0, axis_y, label, color=color, fontsize=8.4,
        ha="right", va="center", weight="bold", zorder=zorder + 2,
    )


def _draw_front_t_magnets(ax, axis_y, *, zorder=18):
    """Draw the +/-X captive cavities in their true front projection."""
    for sign in (-1.0, 1.0):
        center_x = sign * T_MAGNET_CENTER_X_MM
        cavity_x0 = center_x - CAVITY_DEPTH_MM / 2.0
        magnet_x0 = center_x - MAGNET_DEPTH_MM / 2.0
        ax.add_patch(Rectangle(
            (cavity_x0, axis_y - CAVITY_DIAMETER_MM / 2.0),
            CAVITY_DEPTH_MM, CAVITY_DIAMETER_MM,
            fc="#ffe1aa", ec=COLORS["magnet"], lw=1.1,
            zorder=zorder,
        ))
        ax.add_patch(Rectangle(
            (magnet_x0, axis_y - MAGNET_DIAMETER_MM / 2.0),
            MAGNET_DEPTH_MM, MAGNET_DIAMETER_MM,
            fc=COLORS["magnet"], ec=COLORS["ink"], lw=0.55,
            zorder=zorder + 0.2,
        ))


def _front_panel(ax):
    current = Polygon(outline_polygon(OUTLINE_B2)).buffer(0)
    current_vase = current.intersection(box(-90.0, SEAM_B_Y, 90.0, 560.0))

    preserved_lower = current_vase.intersection(
        box(-90.0, SEAM_B_Y, 90.0, 421.0))
    lands = unary_union((
        Point(0.0, LOWER_T_AXIS_Y_MM).buffer(
            TEBM_LAND_D_MM / 2.0, resolution=256),
        Point(0.0, UPPER_T_AXIS_Y_MM).buffer(
            TEBM_LAND_D_MM / 2.0, resolution=256),
    ))
    proposed = unary_union((preserved_lower, lands)).buffer(0)
    proposed = proposed.difference(Point(*UM_CUTOUT[:2]).buffer(
        UM_OPENING_D_MM / 2.0, resolution=256))
    for axis_y in (LOWER_T_AXIS_Y_MM, UPPER_T_AXIS_Y_MM):
        proposed = proposed.difference(Point(0.0, axis_y).buffer(
            TEBM_BORE_D_MM / 2.0, resolution=256))

    _draw_geometry(
        ax, current_vase, facecolor="#f2f4f6", edgecolor=COLORS["existing"],
        alpha=0.72, linewidth=1.1, linestyle=(0, (3, 3)), zorder=1,
    )
    _draw_geometry(
        ax, proposed, facecolor=COLORS["proposal"],
        edgecolor=COLORS["proposal_edge"], alpha=0.96,
        linewidth=1.7, zorder=3,
    )

    # Released UM remains unchanged.
    ax.add_patch(Circle(
        (0.0, UM_AXIS_Y_MM), UM_FLANGE_D_MM / 2.0,
        fc="none", ec=COLORS["um"], lw=1.8, ls=(0, (6, 3)), zorder=9,
    ))
    ax.add_patch(Circle(
        (0.0, UM_AXIS_Y_MM), UM_OPENING_D_MM / 2.0,
        fc="white", ec=COLORS["ink"], lw=1.1, zorder=8,
    ))
    ax.text(
        0.0, UM_AXIS_Y_MM, "UM UNCHANGED\naxis y=366.081\nopening Ø82 / flange Ø97.5",
        color=COLORS["um"], ha="center", va="center", fontsize=8.0,
        weight="bold", zorder=15,
    )

    _driver_front(
        ax, LOWER_T_AXIS_Y_MM, color=COLORS["front"], linestyle="-",
        clock_deg=45.0, label="LOWER / FRONT", zorder=12,
    )
    _driver_front(
        ax, UPPER_T_AXIS_Y_MM, color=COLORS["rear"],
        linestyle=(0, (4, 2)), clock_deg=-45.0,
        label="UPPER / REAR", zorder=12,
    )
    for axis_y in (LOWER_T_AXIS_Y_MM, UPPER_T_AXIS_Y_MM):
        _draw_front_t_magnets(ax, axis_y)

    # Existing captive-magnet station centers, retained as a feasibility cue.
    for mx, my, _nx, _ny, _pin, _zc in MAGNET_SITES:
        for sign in (-1.0, 1.0):
            ax.add_patch(Circle(
                (sign * mx, my), 2.6, fc="none", ec=COLORS["magnet"],
                lw=1.1, zorder=16,
            ))
    magnet_distance = math.dist(
        (MAGNET_SITES[1][0], MAGNET_SITES[1][1]),
        (0.0, LOWER_T_AXIS_Y_MM),
    )
    magnet_plan_gap = (
        magnet_distance - TEBM_BORE_D_MM / 2.0 - 2.6
    )
    ax.annotate(
        f"upper magnet station\n{magnet_plan_gap:.2f} mm to bore",
        xy=(MAGNET_SITES[1][0], MAGNET_SITES[1][1]),
        xytext=(-87.0, 414.5), color=COLORS["magnet"], fontsize=7.3,
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["magnet"], lw=1.0),
        bbox=dict(fc="white", ec=COLORS["magnet"], alpha=0.92, pad=2.0),
        zorder=25,
    )

    _dimension_vertical(
        ax, 42.0, UM_AXIS_Y_MM, LOWER_T_AXIS_Y_MM,
        f"{LOWER_T_AXIS_Y_MM - UM_AXIS_Y_MM:.2f} mm\n(2.10 mm face gap)",
        color=COLORS["front"],
    )
    _dimension_vertical(
        ax, 57.0, LOWER_T_AXIS_Y_MM, UPPER_T_AXIS_Y_MM,
        f"{PAIR_AXIS_PITCH_MM:.1f} mm pitch", color=COLORS["rear"],
    )
    ax.axhline(SEAM_B_Y, color=COLORS["muted"], lw=1.0, ls=":", zorder=20)
    ax.text(
        -87.0, SEAM_B_Y + 1.8, f"released seam B  y={SEAM_B_Y:.2f}",
        color=COLORS["muted"], fontsize=7.4, ha="left", va="bottom",
    )
    ax.text(
        -87.0, 531.5,
        "Grey = current Stock\n"
        "Blue = proposed Ø63 lands\n"
        "Gold = 4×/T M2 bores\n"
        "Orange = 2×/T D5×2 magnets",
        color=COLORS["muted"], fontsize=6.7, ha="left", va="top",
        bbox=dict(fc="white", ec=COLORS["grid"], alpha=0.92, pad=2.2),
    )

    ax.set_title(
        "FRONT VIEW\nexact released lower vase + measured BMR layout",
        fontsize=9.8, weight="bold", color=COLORS["ink"], pad=7.0)
    ax.set_xlim(-90.0, 90.0)
    ax.set_ylim(309.0, 535.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y from baffle datum (mm)")
    ax.grid(True, color=COLORS["grid"], lw=0.45, zorder=0)

    return {
        "upper_magnet_to_lower_bore_plan_gap_mm": magnet_plan_gap,
        "m2_insert_inner_ligament_mm": M2_INSERT_INNER_LIGAMENT_MM,
        "m2_insert_outer_land_mm": M2_INSERT_OUTER_LAND_MM,
        "t_magnet_cavity_to_bore_land_mm": T_MAGNET_CAVITY_TO_BORE_LAND_MM,
        "proposed_plan_bounds_mm": tuple(map(float, proposed.bounds)),
    }


def _driver_side_envelope(ax, axis_y, mount_z, direction, *, color,
                          label, zorder=8):
    # Conservative manufacturer envelope: the complete D43.6 maximum body is
    # used over the full 25.1-mm depth.  This intentionally does not take
    # credit for the visible basket taper in the undimensioned drawing.
    body_z0 = mount_z
    body_z1 = mount_z + direction * TEBM_DEPTH_MM
    left = min(body_z0, body_z1)
    width = abs(body_z1 - body_z0)
    ax.add_patch(Rectangle(
        (left, axis_y - TEBM_BASKET_D_MM / 2.0),
        width, TEBM_BASKET_D_MM,
        fc=color, ec=color, alpha=0.20, lw=1.4, zorder=zorder,
    ))
    ax.plot(
        [mount_z, mount_z],
        [axis_y - TEBM_MAX_D_MM / 2.0, axis_y + TEBM_MAX_D_MM / 2.0],
        color=color, lw=4.0, solid_capstyle="round", zorder=zorder + 3,
    )
    ax.axhline(axis_y, color=color, lw=0.8, ls=":", zorder=zorder + 2)
    label_x = body_z1 + (1.1 if direction > 0 else -1.1)
    ax.text(
        label_x, axis_y, label, color=color, fontsize=8.0,
        ha="left" if direction > 0 else "right", va="center",
        weight="bold", zorder=zorder + 4,
    )


def _side_panel(ax):
    y0 = REAR_RAMP_START_Y_MM - 5.0
    y1 = UPPER_T_AXIS_Y_MM + TEBM_LAND_D_MM / 2.0
    sweep_y = np.linspace(y0, y1, 600)
    sweep_z = _rear_surface_z(sweep_y)
    ax.fill_betweenx(
        sweep_y, sweep_z, THICKNESS_MM,
        facecolor=COLORS["proposal"], edgecolor=COLORS["proposal_edge"],
        linewidth=1.5, zorder=1,
    )
    ax.plot(
        sweep_z, sweep_y, color=COLORS["proposal_edge"], lw=2.0,
        zorder=3,
    )
    # Section through the two circular T openings.  The ramp is complete at
    # the lower opening edge, so both bores traverse the full 25.1-mm field.
    for axis_y in (LOWER_T_AXIS_Y_MM, UPPER_T_AXIS_Y_MM):
        ax.add_patch(Rectangle(
            (REAR_T_MOUNT_Z_MM - 0.15,
             axis_y - TEBM_BORE_D_MM / 2.0),
            T_ZONE_DEPTH_MM + 0.30, TEBM_BORE_D_MM,
            fc="white", ec=COLORS["proposal_edge"], lw=0.8,
            ls=(0, (2, 2)), zorder=3,
        ))

    # The front plane remains released.  The rear-facing unit moves onto the
    # smooth-grown rear face; both 25.1-mm bodies terminate flush.
    _driver_side_envelope(
        ax, LOWER_T_AXIS_Y_MM, THICKNESS_MM, -1.0,
        color=COLORS["front"], label="faces FRONT  →", zorder=6,
    )
    _driver_side_envelope(
        ax, UPPER_T_AXIS_Y_MM, REAR_T_MOUNT_Z_MM, +1.0,
        color=COLORS["rear"], label="←  faces REAR", zorder=6,
    )

    # Blind insert bores, shown in side projection.  Each gold rectangle is
    # the projection of two +/-X holes; the lower set opens from the front
    # face and the upper set opens from the grown rear face.
    for axis_y, open_z, direction in (
        (LOWER_T_AXIS_Y_MM, THICKNESS_MM, -1.0),
        (UPPER_T_AXIS_Y_MM, REAR_T_MOUNT_Z_MM, +1.0),
    ):
        bore_z1 = open_z + direction * M2_INSERT_DEPTH_MM
        for y_offset in (-M2_INSERT_PROJECTED_OFFSET_MM,
                         M2_INSERT_PROJECTED_OFFSET_MM):
            ax.add_patch(Rectangle(
                (min(open_z, bore_z1),
                 axis_y + y_offset - M2_INSERT_BORE_D_MM / 2.0),
                M2_INSERT_DEPTH_MM, M2_INSERT_BORE_D_MM,
                fc=COLORS["insert"], ec=COLORS["ink"], alpha=0.92,
                lw=0.75, zorder=15,
            ))

    # The left/right side-wall magnets overlay in this projection.  Dashed
    # rings identify that these are off the central section at x=+/-31.5.
    for axis_y in (LOWER_T_AXIS_Y_MM, UPPER_T_AXIS_Y_MM):
        ax.add_patch(Circle(
            (STANDARD_MAGNET_Z_MM, axis_y), CAVITY_DIAMETER_MM / 2.0,
            fc="#ffe1aa", ec=COLORS["magnet"], lw=1.0,
            ls=(0, (2, 2)), zorder=16,
        ))
        ax.add_patch(Circle(
            (STANDARD_MAGNET_Z_MM, axis_y), MAGNET_DIAMETER_MM / 2.0,
            fc=COLORS["magnet"], ec=COLORS["ink"], alpha=0.82,
            lw=0.55, zorder=16.2,
        ))
        ax.text(
            STANDARD_MAGNET_Z_MM - 0.4, axis_y + 4.0, "×2 sides",
            color=COLORS["magnet"], fontsize=6.3, ha="right", va="bottom",
            zorder=19,
        )

    ax.text(
        5.0, 391.0,
        "gold = blind M2 insert bores (2 holes/projected pair)\n"
        "orange = D5×2 side magnets (off-section, ×2 sides)",
        color=COLORS["warning"], fontsize=6.6, ha="left", va="bottom",
        bbox=dict(fc="white", ec=COLORS["warning"], alpha=0.94, pad=2.0),
        zorder=28,
    )

    # Governing cross-face maximum-envelope clearance.
    critical_y0 = LOWER_T_AXIS_Y_MM + TEBM_MAX_D_MM / 2.0
    critical_y1 = UPPER_T_AXIS_Y_MM - TEBM_BASKET_D_MM / 2.0
    _dimension_vertical(
        ax, THICKNESS_MM + 0.15, critical_y0, critical_y1,
        f"{critical_y1 - critical_y0:.1f} mm", color=COLORS["warning"],
        label_dx=0.7,
    )
    ax.annotate(
        "governing D54 flange ↔ D43.6 body",
        xy=(THICKNESS_MM, (critical_y0 + critical_y1) / 2.0),
        xytext=(23.0, 463.0), color=COLORS["warning"], fontsize=7.4,
        arrowprops=dict(arrowstyle="->", color=COLORS["warning"], lw=1.0),
        bbox=dict(fc="white", ec=COLORS["warning"], alpha=0.94, pad=2.0),
        zorder=25,
    )

    ax.plot(
        [0.0, 0.0], [y0, REAR_RAMP_START_Y_MM],
        color=COLORS["ink"], lw=1.0,
    )
    ax.axvline(THICKNESS_MM, color=COLORS["ink"], lw=1.0)
    ax.text(REAR_T_MOUNT_Z_MM, y1 + 2.2, "T REAR  z=-6.8",
            fontsize=7.5, ha="center")
    ax.text(THICKNESS_MM, y1 + 2.2, "FRONT  z=18.3", fontsize=7.5, ha="center")
    ax.annotate(
        "front", xy=(29.0, y0 + 3.0), xytext=(23.0, y0 + 3.0),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=COLORS["ink"]),
        ha="left", va="center", fontsize=7.5,
    )
    ramp_mid_y = (REAR_RAMP_START_Y_MM + REAR_RAMP_END_Y_MM) / 2.0
    ramp_mid_z = float(_rear_surface_z(ramp_mid_y))
    ax.annotate(
        f"C2 rear sweep\n{REAR_RAMP_LENGTH_MM:.1f} mm run / "
        f"{REAR_GROWTH_MM:.1f} mm growth\nzero slope + curvature at both ends",
        xy=(ramp_mid_z, ramp_mid_y), xytext=(-9.0, 399.0),
        fontsize=7.4, color=COLORS["proposal_edge"], ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["proposal_edge"], lw=1.0),
        bbox=dict(fc="white", ec=COLORS["proposal_edge"], alpha=0.94, pad=2.0),
        zorder=24,
    )

    ax.set_title("SIDE VIEW — smooth rear growth to a flush 25.1 mm T field",
                 fontsize=11.2, weight="bold", color=COLORS["ink"])
    ax.set_xlim(-10.0, 24.0)
    ax.set_ylim(y0 - 5.0, y1 + 9.0)
    ax.set_xlabel("z / acoustic depth (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, color=COLORS["grid"], lw=0.45, zorder=0)


def _top_panel(ax):
    # Looking downward along +Y; the two vertically separated drivers
    # superimpose in this view.  Colors show their opposite depth directions.
    ax.add_patch(Rectangle(
        (-TEBM_LAND_D_MM / 2.0, REAR_T_MOUNT_Z_MM),
        TEBM_LAND_D_MM, T_ZONE_DEPTH_MM,
        fc=COLORS["proposal"], ec=COLORS["proposal_edge"], lw=1.5,
        zorder=1,
    ))
    ax.add_patch(Rectangle(
        (-TEBM_LAND_D_MM / 2.0, REAR_T_MOUNT_Z_MM),
        TEBM_LAND_D_MM, REAR_GROWTH_MM,
        fc="#c7deeb", ec="none", alpha=0.92, zorder=2,
    ))
    ax.add_patch(Rectangle(
        (-TEBM_BASKET_D_MM / 2.0, REAR_T_MOUNT_Z_MM),
        TEBM_BASKET_D_MM, TEBM_DEPTH_MM,
        fc=COLORS["front"], ec=COLORS["front"], alpha=0.20, lw=1.2,
        zorder=4,
    ))
    ax.add_patch(Rectangle(
        (-TEBM_BASKET_D_MM / 2.0, REAR_T_MOUNT_Z_MM),
        TEBM_BASKET_D_MM, TEBM_DEPTH_MM,
        fc=COLORS["rear"], ec=COLORS["rear"], alpha=0.20, lw=1.2,
        zorder=5,
    ))
    ax.plot(
        [-TEBM_MAX_D_MM / 2.0, TEBM_MAX_D_MM / 2.0],
        [THICKNESS_MM, THICKNESS_MM], color=COLORS["front"], lw=4.0,
        solid_capstyle="round", zorder=8,
    )
    ax.plot(
        [-TEBM_MAX_D_MM / 2.0, TEBM_MAX_D_MM / 2.0],
        [REAR_T_MOUNT_Z_MM, REAR_T_MOUNT_Z_MM],
        color=COLORS["rear"], lw=4.0,
        solid_capstyle="round", zorder=8,
    )

    # Front-opening and rear-opening blind insert bores.  The top/bottom
    # drivers share these projected x offsets but occupy opposite faces.
    for x_offset in (-M2_INSERT_PROJECTED_OFFSET_MM,
                     M2_INSERT_PROJECTED_OFFSET_MM):
        ax.add_patch(Rectangle(
            (x_offset - M2_INSERT_BORE_D_MM / 2.0,
             THICKNESS_MM - M2_INSERT_DEPTH_MM),
            M2_INSERT_BORE_D_MM, M2_INSERT_DEPTH_MM,
            fc=COLORS["insert"], ec=COLORS["ink"], lw=0.7,
            zorder=12,
        ))
        ax.add_patch(Rectangle(
            (x_offset - M2_INSERT_BORE_D_MM / 2.0,
             REAR_T_MOUNT_Z_MM),
            M2_INSERT_BORE_D_MM, M2_INSERT_DEPTH_MM,
            fc=COLORS["insert"], ec=COLORS["ink"], lw=0.7,
            zorder=12,
        ))

    # The two y stations superimpose in top view.  Each side-wall pocket is
    # drawn as the true D5.20-transverse x 2.10-axial XZ projection.
    for sign in (-1.0, 1.0):
        center_x = sign * T_MAGNET_CENTER_X_MM
        ax.add_patch(Rectangle(
            (center_x - CAVITY_DEPTH_MM / 2.0,
             STANDARD_MAGNET_Z_MM - CAVITY_DIAMETER_MM / 2.0),
            CAVITY_DEPTH_MM, CAVITY_DIAMETER_MM,
            fc="#ffe1aa", ec=COLORS["magnet"], lw=1.0, zorder=14,
        ))
        ax.add_patch(Rectangle(
            (center_x - MAGNET_DEPTH_MM / 2.0,
             STANDARD_MAGNET_Z_MM - MAGNET_DIAMETER_MM / 2.0),
            MAGNET_DEPTH_MM, MAGNET_DIAMETER_MM,
            fc=COLORS["magnet"], ec=COLORS["ink"], lw=0.5, zorder=14.2,
        ))
    _dimension_vertical(
        ax, 30.2, REAR_T_MOUNT_Z_MM, THICKNESS_MM,
        f"{ASSEMBLY_DEPTH_ENVELOPE_MM:.1f} mm overall",
        color=COLORS["ink"], label_dx=1.0,
    )
    _dimension_horizontal(
        ax, -9.8, -TEBM_MAX_D_MM / 2.0, TEBM_MAX_D_MM / 2.0,
        "Ø54 max", color=COLORS["ink"], label_dy=0.8,
    )
    ax.text(
        0.0, 9.15,
        "released plate\n18.3 mm",
        color=COLORS["proposal_edge"], fontsize=8.0, ha="center",
        va="center", weight="bold", zorder=12,
    )
    ax.text(
        -31.5, 20.2, "lower/front face", color=COLORS["front"],
        fontsize=7.5, ha="left", va="center",
    )
    ax.text(
        -31.5, -7.8, "upper/rear face", color=COLORS["rear"],
        fontsize=7.5, ha="left", va="center",
    )
    ax.text(
        15.0, -3.4, "+6.8 mm smooth rear growth", color=COLORS["proposal_edge"],
        fontsize=6.9, ha="center", va="center", weight="bold", zorder=12,
    )
    ax.annotate(
        "D5×2 side magnets\n2 y stations overlap",
        xy=(T_MAGNET_CENTER_X_MM, STANDARD_MAGNET_Z_MM),
        xytext=(8.0, 3.2), color=COLORS["magnet"], fontsize=6.1,
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["magnet"], lw=0.8),
        bbox=dict(fc="white", ec=COLORS["magnet"], alpha=0.90, pad=1.5),
        zorder=16,
    )
    ax.set_title("TOP VIEW — vertical axes coincide at x=0",
                 fontsize=10.6, weight="bold", color=COLORS["ink"])
    ax.set_xlim(-37.0, 37.0)
    ax.set_ylim(-10.5, 22.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.grid(True, color=COLORS["grid"], lw=0.45, zorder=0)


def _facts_panel(ax, front_facts):
    ax.axis("off")
    ax.add_patch(Rectangle(
        (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
        fc="#f8fafc", ec=COLORS["grid"], lw=1.2, zorder=0,
    ))
    ax.text(
        0.04, 0.94, "DIMENSION + HARDWARE CONTRACT", transform=ax.transAxes,
        fontsize=10.5, weight="bold", color=COLORS["ink"], va="top",
    )
    measured = (
        "PUBLISHED DRIVER\n"
        "  Ø52 nominal / Ø54 max\n"
        "  basket envelope Ø43.6\n"
        f"  cutout Ø{TEBM_CUTOUT_D_MM:.3f}\n"
        f"  4 holes on Ø{TEBM_MOUNT_PCD_MM:.2f} PCD\n"
        "  depth 25.1 / mass 51.3 g\n\n"
        "RELEASED STOCK CAD\n"
        "  UM axis y=366.081\n"
        "  UM opening Ø82 / flange Ø97.5\n"
        "  plate thickness 18.3"
    )
    derived = (
        "LAYOUT / REAR GROWTH\n"
        f"  lower/front axis y={LOWER_T_AXIS_Y_MM:.3f}\n"
        f"  upper/rear axis y={UPPER_T_AXIS_Y_MM:.3f}\n"
        f"  center pitch {PAIR_AXIS_PITCH_MM:.1f}\n"
        f"  listed opening Ø{TEBM_BORE_D_MM:.3f}\n"
        f"  opening web {BORE_WEB_MM:.3f}\n"
        f"  projected flange overlap {FLANGE_PROJECTION_OVERLAP_MM:.1f}\n"
        f"  rear growth {REAR_GROWTH_MM:.1f}\n"
        f"  C2 ramp y={REAR_RAMP_START_Y_MM:.3f}..{REAR_RAMP_END_Y_MM:.3f}\n"
        f"  flush T-zone depth {ASSEMBLY_DEPTH_ENVELOPE_MM:.1f}"
    )
    ax.text(
        0.04, 0.85, measured, transform=ax.transAxes, fontsize=6.75,
        color=COLORS["ink"], va="top", family="monospace", linespacing=1.28,
    )
    ax.text(
        0.53, 0.85, derived, transform=ax.transAxes, fontsize=6.75,
        color=COLORS["ink"], va="top", family="monospace", linespacing=1.28,
    )
    hardware = (
        "REQUESTED INTERFACES\n"
        f"M2 inserts  4×/T · blind Ø{M2_INSERT_BORE_D_MM:.1f}×"
        f"{M2_INSERT_DEPTH_MM:.1f} · Ø{TEBM_MOUNT_PCD_MM:.2f} PCD · "
        f"{M2_INSERT_TOTAL} total\n"
        f"            inner ligament {M2_INSERT_INNER_LIGAMENT_MM:.2f} · "
        f"outer land {M2_INSERT_OUTER_LAND_MM:.2f}\n"
        f"T magnets   2×/T · D{MAGNET_DIAMETER_MM:.0f}×"
        f"{MAGNET_DEPTH_MM:.0f} in Ø{CAVITY_DIAMETER_MM:.2f}×"
        f"{CAVITY_DEPTH_MM:.2f} cavity · {T_MAGNET_TOTAL} total\n"
        f"            ±X outer edges · skin {FACE_SKIN_MM:.2f} · "
        f"z={STANDARD_MAGNET_Z_MM:.2f} · bore land "
        f"{T_MAGNET_CAVITY_TO_BORE_LAND_MM:.2f}"
    )
    ax.text(
        0.04, 0.42, hardware, transform=ax.transAxes, fontsize=6.45,
        color=COLORS["ink"], va="top", linespacing=1.28, wrap=True,
        bbox=dict(fc="#fff8e8", ec=COLORS["insert"], alpha=0.98, pad=3.0),
    )
    warning = (
        "NOT YET A CAD RELEASE\n"
        "• Cutout/PCD are listing-derived; caliper-check samples + terminals.\n"
        "• Ø3.2 is the requested insert envelope; thermal pilot needs a coupon.\n"
        "• 1.07-mm ligament, captive closure, receiver/polarity, cable route,\n"
        "  acoustics and BREP remain release gates."
    )
    ax.text(
        0.04, 0.205, warning, transform=ax.transAxes, fontsize=6.2,
        color=COLORS["warning"], va="top", linespacing=1.32,
        bbox=dict(fc="#fff7ed", ec="#e1a45e", alpha=0.98, pad=4.0),
    )


def generate(output: Path) -> dict[str, object]:
    _validate_dimension_contract()
    fig = plt.figure(figsize=(16.5, 10.8), dpi=160, facecolor="#eef3f7")
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.05, 0.82, 1.28),
        height_ratios=(1.18, 0.82), wspace=0.27, hspace=0.30,
    )
    front_ax = fig.add_subplot(grid[:, 0])
    side_ax = fig.add_subplot(grid[0, 1:])
    top_ax = fig.add_subplot(grid[1, 1])
    facts_ax = fig.add_subplot(grid[1, 2])

    front_facts = _front_panel(front_ax)
    _side_panel(side_ax)
    _top_panel(top_ax)
    _facts_panel(facts_ax, front_facts)

    fig.suptitle(
        "STOCK vase_TEBM35C10-4 — opposed two-driver interface concept",
        fontsize=17.0, weight="bold", color=COLORS["ink"], y=0.985,
    )
    fig.text(
        0.5, 0.955,
        "Lower T faces front; upper T faces rear.  UM position and released "
        "Stock lower-vase geometry remain unchanged; rear T field grows smoothly to 25.1 mm.  "
        "Each T receives four M2 inserts and two side magnets.",
        ha="center", va="top", fontsize=10.0, color=COLORS["muted"],
    )
    fig.text(
        0.012, 0.012,
        "Review-only PNG • local source inspection and published driver sources • "
        "no BREP/STL/3MF, slicing, or Osado resources",
        fontsize=7.4, color=COLORS["muted"], ha="left", va="bottom",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(
            temporary, bbox_inches="tight", facecolor=fig.get_facecolor(),
            metadata={
                "Title": "STOCK vase_TEBM35C10-4 measured concept",
                "Description": (
                    "Review-only opposed TEBM35C10-4 layout; released Stock "
                    "UM and B2 datums; manufacturer maximum envelopes."
                ),
            },
        )
        plt.close(fig)
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            image.load()
            if image.width < 2000 or image.height < 1200:
                raise RuntimeError(
                    f"concept PNG unexpectedly small: {image.size}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    facts = {
        "output": str(output),
        "released": {
            "plate_thickness_mm": THICKNESS_MM,
            "seam_b_y_mm": SEAM_B_Y,
            "um_axis_y_mm": UM_AXIS_Y_MM,
            "um_opening_d_mm": UM_OPENING_D_MM,
            "um_flange_d_mm": UM_FLANGE_D_MM,
            "current_front_t_axis_y_mm": RELEASED_FRONT_T_AXIS_Y_MM,
            "current_rear_t_axis_y_mm": RELEASED_REAR_T_AXIS_Y_MM,
        },
        "manufacturer": {
            "nominal_d_mm": TEBM_NOMINAL_D_MM,
            "max_d_mm": TEBM_MAX_D_MM,
            "basket_d_mm": TEBM_BASKET_D_MM,
            "published_cutout_d_mm": TEBM_CUTOUT_D_MM,
            "published_mount_pcd_mm": TEBM_MOUNT_PCD_MM,
            "published_mount_hole_count": TEBM_MOUNT_HOLE_COUNT,
            "depth_mm": TEBM_DEPTH_MM,
            "mass_g": TEBM_MASS_G,
        },
        "concept": {
            "lower_front_axis_y_mm": LOWER_T_AXIS_Y_MM,
            "upper_rear_axis_y_mm": UPPER_T_AXIS_Y_MM,
            "pair_axis_pitch_mm": PAIR_AXIS_PITCH_MM,
            "bore_d_mm": TEBM_BORE_D_MM,
            "bore_web_mm": BORE_WEB_MM,
            "flange_projection_overlap_mm": FLANGE_PROJECTION_OVERLAP_MM,
            "body_to_opposite_flange_clearance_mm": (
                BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM),
            "opposite_face_protrusion_mm": OPPOSITE_FACE_PROTRUSION_MM,
            "assembly_depth_envelope_mm": ASSEMBLY_DEPTH_ENVELOPE_MM,
            "t_zone_depth_mm": T_ZONE_DEPTH_MM,
            "rear_growth_mm": REAR_GROWTH_MM,
            "rear_t_mount_z_mm": REAR_T_MOUNT_Z_MM,
            "rear_ramp_start_y_mm": REAR_RAMP_START_Y_MM,
            "rear_ramp_end_y_mm": REAR_RAMP_END_Y_MM,
            "rear_ramp_length_mm": REAR_RAMP_LENGTH_MM,
            "rear_ramp_continuity": "C2 quintic smootherstep",
            "land_d_mm": TEBM_LAND_D_MM,
            "clocking_deg": {"lower_front": 45.0, "upper_rear": -45.0},
            "m2_insert_interface": {
                "bore_d_mm": M2_INSERT_BORE_D_MM,
                "blind_depth_mm": M2_INSERT_DEPTH_MM,
                "per_driver": M2_INSERTS_PER_DRIVER,
                "total": M2_INSERT_TOTAL,
                "pcd_mm": TEBM_MOUNT_PCD_MM,
                "inner_ligament_mm": M2_INSERT_INNER_LIGAMENT_MM,
                "outer_land_mm": M2_INSERT_OUTER_LAND_MM,
                "lower_front_bore_z_range_mm": (
                    THICKNESS_MM - M2_INSERT_DEPTH_MM,
                    THICKNESS_MM,
                ),
                "upper_rear_bore_z_range_mm": (
                    REAR_T_MOUNT_Z_MM,
                    REAR_T_MOUNT_Z_MM + M2_INSERT_DEPTH_MM,
                ),
            },
            "t_magnet_interface": {
                "nominal_d_mm": MAGNET_DIAMETER_MM,
                "nominal_depth_mm": MAGNET_DEPTH_MM,
                "cavity_d_mm": CAVITY_DIAMETER_MM,
                "cavity_depth_mm": CAVITY_DEPTH_MM,
                "face_skin_mm": FACE_SKIN_MM,
                "per_driver": T_MAGNETS_PER_DRIVER,
                "total": T_MAGNET_TOTAL,
                "axis": "+/-X",
                "center_x_mm": (-T_MAGNET_CENTER_X_MM,
                                T_MAGNET_CENTER_X_MM),
                "center_y_mm": (LOWER_T_AXIS_Y_MM, UPPER_T_AXIS_Y_MM),
                "center_z_mm": STANDARD_MAGNET_Z_MM,
                "interface_face_x_mm": (-T_MAGNET_FACE_X_MM,
                                         T_MAGNET_FACE_X_MM),
                "qualified_flat_height_mm": 2.0 * T_MAGNET_FLAT_HALF_HEIGHT_MM,
                "flat_edge_margin_mm": T_MAGNET_FLAT_EDGE_MARGIN_MM,
                "minimum_planar_outer_skin_mm": FACE_SKIN_MM,
                "cavity_to_bore_land_mm": (
                    T_MAGNET_CAVITY_TO_BORE_LAND_MM),
            },
            **front_facts,
        },
    }
    return facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "review" /
        "vase_TEBM35C10-4_front_side_top_concept.png",
    )
    args = parser.parse_args()
    facts = generate(args.output.resolve())
    print(json.dumps(facts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
