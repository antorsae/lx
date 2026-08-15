"""Profile-neutral TEBM35C10-4 driver-land geometry.

The Proud vase and Obi-Wan candidates run under mutually exclusive routing
profiles, so neither family can import the other.  This module owns the small
piece of geometry they genuinely share: the conservative D63 prototype land
and the optional driver-following ``bmr-slim`` topology.

Both topologies keep the same side-magnet interface faces.  The full version
is a clipped D63 circle.  The slim version keeps a D56 driver ring, adds four
small M2 support pads, and grows only two local magnet lobes out to the D63
faces.  It is deliberately a candidate: the published driver drawing omits
the tolerances, lug detail, flange thickness and terminal envelope needed for
physical release.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from shapely.affinity import translate
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .magnet_contract import CAPTIVE_LAND_MM


# Published/distributor interface envelope, in millimetres.
TEBM_NOMINAL_D_MM = 52.0
TEBM_MAX_D_MM = 54.0
TEBM_BASKET_D_MM = 43.6
TEBM_DEPTH_MM = 25.1
TEBM_MASS_G = 51.3
TEBM_CUTOUT_D_MM = 1.69 * 25.4
TEBM_MOUNT_PCD_MM = 1.90 * 25.4
TEBM_MOUNT_HOLE_COUNT = 4

M2_INSERT_BORE_D_MM = 3.2
M2_INSERT_DEPTH_MM = 4.0
LOWER_T_MOUNT_CLOCK_DEG = 45.0
UPPER_T_MOUNT_CLOCK_DEG = -45.0
T_BLIND_BACK_WALL_THICKNESS_MM = 1.20
UPPER_T_BRANCH_D_MM = 4.60

# Shared side-magnet interface.  The 0.10-mm transverse margin makes the
# whole qualified 6.40-mm captive land sit on a real planar face.
T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM = 3.20
T_MAGNET_FLAT_EDGE_MARGIN_MM = 0.10
T_MAGNET_FLAT_HALF_HEIGHT_MM = (
    T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM + T_MAGNET_FLAT_EDGE_MARGIN_MM
)

FULL_LAND_D_MM = 63.0
FULL_LAND_R_MM = FULL_LAND_D_MM / 2.0
T_MAGNET_FACE_X_MM = math.sqrt(
    FULL_LAND_R_MM ** 2 - T_MAGNET_FLAT_HALF_HEIGHT_MM ** 2
)
T_MAGNET_TOTAL = 4

# The driver-following candidate keeps one millimetre radially beyond the
# conservative D54 envelope.  Tiny diagonal pads retain at least 2.50 mm of
# plastic outside each D3.2 M2 bore, and each side lobe contains the complete
# captive land plus a 0.20-mm radial construction margin.
BMR_SLIM_CORE_D_MM = 56.0
BMR_SLIM_CORE_R_MM = BMR_SLIM_CORE_D_MM / 2.0
BMR_SLIM_M2_WALL_MM = 2.50
BMR_SLIM_M2_BOSS_R_MM = M2_INSERT_BORE_D_MM / 2.0 + BMR_SLIM_M2_WALL_MM
BMR_SLIM_LOBE_ROOT_X_MM = BMR_SLIM_CORE_R_MM - 1.0
BMR_SLIM_LOBE_INNER_MARGIN_MM = 0.20
BMR_SLIM_LOBE_DEPTH_MM = CAPTIVE_LAND_MM + BMR_SLIM_LOBE_INNER_MARGIN_MM

PLAN_RESOLUTION = 96


@dataclass(frozen=True)
class BmrLandTopology:
    """One reusable TEBM land silhouette."""

    key: str
    profile: str
    core_d_mm: float
    parent_d_mm: float
    magnet_face_x_mm: float

    @property
    def core_r_mm(self) -> float:
        return self.core_d_mm / 2.0

    @property
    def plan_width_mm(self) -> float:
        return 2.0 * self.magnet_face_x_mm

    @property
    def magnet_inner_edge_x_mm(self) -> float:
        return self.magnet_face_x_mm - CAPTIVE_LAND_MM

    @property
    def conservative_driver_ligament_mm(self) -> float:
        return self.magnet_inner_edge_x_mm - TEBM_MAX_D_MM / 2.0

    @property
    def pocket_wall_mm(self) -> float:
        return self.core_r_mm - TEBM_CUTOUT_D_MM / 2.0

    @property
    def m2_radial_wall_mm(self) -> float:
        if self.key == "bmr-slim":
            return BMR_SLIM_M2_WALL_MM
        return self.core_r_mm - (
            TEBM_MOUNT_PCD_MM + M2_INSERT_BORE_D_MM) / 2.0


FULL_CIRCULAR_LAND = BmrLandTopology(
    key="full",
    profile="d63_circle_with_side_magnet_flats",
    core_d_mm=FULL_LAND_D_MM,
    parent_d_mm=FULL_LAND_D_MM,
    magnet_face_x_mm=T_MAGNET_FACE_X_MM,
)

BMR_SLIM_LAND = BmrLandTopology(
    key="bmr-slim",
    profile="d56_driver_ring_with_m2_pads_and_side_magnet_lobes",
    core_d_mm=BMR_SLIM_CORE_D_MM,
    parent_d_mm=FULL_LAND_D_MM,
    magnet_face_x_mm=T_MAGNET_FACE_X_MM,
)

LAND_TOPOLOGIES = {
    topology.key: topology
    for topology in (FULL_CIRCULAR_LAND, BMR_SLIM_LAND)
}


def land_topology(
    value: str | BmrLandTopology = "full",
) -> BmrLandTopology:
    if isinstance(value, BmrLandTopology):
        if any(value is topology for topology in LAND_TOPOLOGIES.values()):
            return value
        raise ValueError(
            "TEBM land topology objects must be one of the canonical shared "
            "instances; use 'full' or 'bmr-slim'")
    try:
        return LAND_TOPOLOGIES[value]
    except KeyError as exc:
        raise ValueError(f"unknown TEBM land topology {value!r}") from exc


def mount_centres() -> tuple[tuple[float, float], ...]:
    """The four shared land-local M2 insert centres."""
    radius = TEBM_MOUNT_PCD_MM / 2.0
    return tuple(
        (
            radius * math.cos(math.radians(
                LOWER_T_MOUNT_CLOCK_DEG + 90.0 * index)),
            radius * math.sin(math.radians(
                LOWER_T_MOUNT_CLOCK_DEG + 90.0 * index)),
        )
        for index in range(TEBM_MOUNT_HOLE_COUNT)
    )


@lru_cache(maxsize=None)
def local_land_plan(
    value: str | BmrLandTopology = "full",
):
    """Return one topology about a driver axis at the origin."""
    topology = land_topology(value)
    if topology.key == "full":
        disc = Point(0.0, 0.0).buffer(
            topology.core_r_mm, resolution=PLAN_RESOLUTION)
        return disc.intersection(box(
            -topology.magnet_face_x_mm,
            -4000.0,
            topology.magnet_face_x_mm,
            4000.0,
        ))

    ring = Point(0.0, 0.0).buffer(
        BMR_SLIM_CORE_R_MM, resolution=PLAN_RESOLUTION)
    pieces = [ring]
    for sign in (-1.0, 1.0):
        face_x = sign * topology.magnet_face_x_mm
        inner_x = sign * BMR_SLIM_LOBE_ROOT_X_MM
        pad = box(
            min(face_x, inner_x),
            -T_MAGNET_FLAT_HALF_HEIGHT_MM,
            max(face_x, inner_x),
            T_MAGNET_FLAT_HALF_HEIGHT_MM,
        )
        pieces.append(pad)

    pieces.extend(
        Point(x, y).buffer(
            BMR_SLIM_M2_BOSS_R_MM, resolution=PLAN_RESOLUTION)
        for x, y in mount_centres()
    )
    plan = unary_union(pieces).buffer(0)
    if plan.geom_type != "Polygon" or plan.interiors:
        raise RuntimeError(
            "the BMR-slim land must be one simple hole-free plan")
    return plan


def placed_land_plan(
    axis_y: float,
    value: str | BmrLandTopology = "full",
):
    return translate(local_land_plan(value), yoff=float(axis_y))


def topology_facts(
    value: str | BmrLandTopology = "full",
) -> dict[str, object]:
    topology = land_topology(value)
    plan = local_land_plan(topology)
    return {
        "key": topology.key,
        "profile": topology.profile,
        "core_d_mm": topology.core_d_mm,
        "parent_d_mm": topology.parent_d_mm,
        "max_plan_width_mm": topology.plan_width_mm,
        "plan_depth_mm": float(plan.bounds[3] - plan.bounds[1]),
        "plan_area_mm2": float(plan.area),
        "magnet_face_x_mm": topology.magnet_face_x_mm,
        "magnet_flat_height_mm": 2.0 * T_MAGNET_FLAT_HALF_HEIGHT_MM,
        "magnet_inner_edge_x_mm": topology.magnet_inner_edge_x_mm,
        "conservative_driver_ligament_mm": (
            topology.conservative_driver_ligament_mm),
        "pocket_wall_mm": topology.pocket_wall_mm,
        "m2_radial_wall_mm": topology.m2_radial_wall_mm,
        "physical_measure_required": True,
    }


__all__ = [
    "BMR_SLIM_CORE_D_MM",
    "BMR_SLIM_CORE_R_MM",
    "BMR_SLIM_LAND",
    "BMR_SLIM_LOBE_DEPTH_MM",
    "BMR_SLIM_M2_BOSS_R_MM",
    "BmrLandTopology",
    "FULL_CIRCULAR_LAND",
    "FULL_LAND_D_MM",
    "FULL_LAND_R_MM",
    "LAND_TOPOLOGIES",
    "LOWER_T_MOUNT_CLOCK_DEG",
    "M2_INSERT_BORE_D_MM",
    "M2_INSERT_DEPTH_MM",
    "T_BLIND_BACK_WALL_THICKNESS_MM",
    "TEBM_BASKET_D_MM",
    "TEBM_CUTOUT_D_MM",
    "TEBM_DEPTH_MM",
    "TEBM_MASS_G",
    "TEBM_MAX_D_MM",
    "TEBM_MOUNT_HOLE_COUNT",
    "TEBM_MOUNT_PCD_MM",
    "TEBM_NOMINAL_D_MM",
    "T_MAGNET_FACE_X_MM",
    "T_MAGNET_FLAT_EDGE_MARGIN_MM",
    "T_MAGNET_FLAT_HALF_HEIGHT_MM",
    "T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM",
    "T_MAGNET_TOTAL",
    "UPPER_T_BRANCH_D_MM",
    "UPPER_T_MOUNT_CLOCK_DEG",
    "land_topology",
    "local_land_plan",
    "mount_centres",
    "placed_land_plan",
    "topology_facts",
]
