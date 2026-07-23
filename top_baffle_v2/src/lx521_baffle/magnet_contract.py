"""Pure dimensional contract for the qualified captive-magnet topology.

This module intentionally imports no CAD kernel.  Geometry construction stays
in :mod:`captive_magnets`; release and slicer code may import these immutable
dimensions and derived facts without loading build123d or OCP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


MAGNET_DIAMETER_MM = 5.0
MAGNET_DEPTH_MM = 2.0
CAVITY_DIAMETER_MM = 5.20
CAVITY_DEPTH_MM = 2.10
FACE_SKIN_MM = 0.45
INNER_SKIN_MM = 0.45
INTERFACE_GAP_MM = 0.05
ROOF_ANGLE_DEG = 45.0
ROOF_PLANE_GRID_MM = 0.20
BOOLEAN_EPS_MM = 0.03
SIDE_WALL_MARGIN_MM = 0.60
MINIMUM_RETAINING_PATH_MM = 0.42

ROOF_HEIGHT_MM = (
    CAVITY_DIAMETER_MM / 2.0
    / math.tan(math.radians(ROOF_ANGLE_DEG))
)
CAPTIVE_LAND_MM = round(
    FACE_SKIN_MM + CAVITY_DEPTH_MM + INNER_SKIN_MM, 9)
NOMINAL_PAIRED_FACE_SEPARATION_MM = round(
    FACE_SKIN_MM + INTERFACE_GAP_MM + FACE_SKIN_MM, 9)


class CaptiveMagnetGeometryError(ValueError):
    """Raised when a request cannot realize the qualified topology."""


@dataclass(frozen=True)
class CaptiveMagnetSpec:
    """Parametric dimensions of one pause-and-bury disc cavity."""

    magnet_diameter_mm: float = MAGNET_DIAMETER_MM
    magnet_depth_mm: float = MAGNET_DEPTH_MM
    cavity_diameter_mm: float = CAVITY_DIAMETER_MM
    cavity_depth_mm: float = CAVITY_DEPTH_MM
    face_skin_mm: float = FACE_SKIN_MM
    inner_skin_mm: float = INNER_SKIN_MM
    interface_gap_mm: float = INTERFACE_GAP_MM
    roof_angle_deg: float = ROOF_ANGLE_DEG
    roof_plane_grid_mm: float = ROOF_PLANE_GRID_MM
    boolean_epsilon_mm: float = BOOLEAN_EPS_MM
    side_wall_margin_mm: float = SIDE_WALL_MARGIN_MM
    retaining_path_mm: float = MINIMUM_RETAINING_PATH_MM

    def __post_init__(self) -> None:
        positive = {
            "magnet_diameter_mm": self.magnet_diameter_mm,
            "magnet_depth_mm": self.magnet_depth_mm,
            "cavity_diameter_mm": self.cavity_diameter_mm,
            "cavity_depth_mm": self.cavity_depth_mm,
            "face_skin_mm": self.face_skin_mm,
            "inner_skin_mm": self.inner_skin_mm,
            "roof_angle_deg": self.roof_angle_deg,
            "roof_plane_grid_mm": self.roof_plane_grid_mm,
            "boolean_epsilon_mm": self.boolean_epsilon_mm,
            "side_wall_margin_mm": self.side_wall_margin_mm,
            "retaining_path_mm": self.retaining_path_mm,
        }
        bad = {key: value for key, value in positive.items() if value <= 0.0}
        if bad:
            raise CaptiveMagnetGeometryError(
                f"captive-magnet dimensions must be positive: {bad}")
        if self.interface_gap_mm < 0.0:
            raise CaptiveMagnetGeometryError(
                "interface_gap_mm must be non-negative")
        if self.cavity_diameter_mm < self.magnet_diameter_mm:
            raise CaptiveMagnetGeometryError(
                "cavity diameter cannot be smaller than the magnet")
        if self.cavity_depth_mm < self.magnet_depth_mm:
            raise CaptiveMagnetGeometryError(
                "cavity depth cannot be smaller than the magnet")
        if not 0.0 < self.roof_angle_deg <= 45.0:
            raise CaptiveMagnetGeometryError(
                "roof angle must be in (0, 45] degrees for self-support")
        if self.face_skin_mm + 1.0e-9 < self.retaining_path_mm:
            raise CaptiveMagnetGeometryError(
                "face skin is thinner than the qualified minimum wall path")
        if self.inner_skin_mm + 1.0e-9 < self.retaining_path_mm:
            raise CaptiveMagnetGeometryError(
                "inner skin is thinner than the qualified minimum wall path")

    @property
    def cavity_radius_mm(self) -> float:
        return self.cavity_diameter_mm / 2.0

    @property
    def roof_height_mm(self) -> float:
        return (
            self.cavity_radius_mm
            / math.tan(math.radians(self.roof_angle_deg))
        )

    @property
    def captive_land_mm(self) -> float:
        return round(
            self.face_skin_mm + self.cavity_depth_mm + self.inner_skin_mm,
            9,
        )

    @property
    def paired_face_separation_mm(self) -> float:
        return round(
            2.0 * self.face_skin_mm + self.interface_gap_mm,
            9,
        )

    def facts(self) -> dict[str, float]:
        return {
            "magnet_diameter_mm": self.magnet_diameter_mm,
            "magnet_depth_mm": self.magnet_depth_mm,
            "cavity_diameter_mm": self.cavity_diameter_mm,
            "cavity_depth_mm": self.cavity_depth_mm,
            "face_skin_mm": self.face_skin_mm,
            "inner_skin_mm": self.inner_skin_mm,
            "captive_land_mm": self.captive_land_mm,
            "interface_gap_mm": self.interface_gap_mm,
            "paired_magnet_face_separation_mm": self.paired_face_separation_mm,
            "roof_angle_deg": self.roof_angle_deg,
            "roof_height_mm": self.roof_height_mm,
            "roof_plane_grid_mm": self.roof_plane_grid_mm,
            "minimum_retaining_path_mm": self.retaining_path_mm,
        }


DEFAULT_SPEC = CaptiveMagnetSpec()


__all__ = [
    "MAGNET_DIAMETER_MM",
    "MAGNET_DEPTH_MM",
    "CAVITY_DIAMETER_MM",
    "CAVITY_DEPTH_MM",
    "FACE_SKIN_MM",
    "INNER_SKIN_MM",
    "INTERFACE_GAP_MM",
    "ROOF_ANGLE_DEG",
    "ROOF_PLANE_GRID_MM",
    "BOOLEAN_EPS_MM",
    "SIDE_WALL_MARGIN_MM",
    "MINIMUM_RETAINING_PATH_MM",
    "ROOF_HEIGHT_MM",
    "CAPTIVE_LAND_MM",
    "NOMINAL_PAIRED_FACE_SEPARATION_MM",
    "CaptiveMagnetGeometryError",
    "CaptiveMagnetSpec",
    "DEFAULT_SPEC",
]
