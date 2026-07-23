"""Dependency-free scalar geometry primitives shared by CAD modules."""

from __future__ import annotations

import math
from collections.abc import Sequence


def smoothstep01(value: float) -> float:
    """Return cubic smoothstep after clamping a scalar to ``[0, 1]``."""
    clamped = max(0.0, min(1.0, float(value)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def point_segment_distance(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    """Return 2-D Euclidean distance to a non-degenerate line segment."""
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    weight = max(
        0.0,
        min(
            1.0,
            (
                (point[0] - start[0]) * delta_x
                + (point[1] - start[1]) * delta_y
            )
            / (delta_x * delta_x + delta_y * delta_y),
        ),
    )
    closest = (
        start[0] + weight * delta_x,
        start[1] + weight * delta_y,
    )
    return math.dist(point, closest)
