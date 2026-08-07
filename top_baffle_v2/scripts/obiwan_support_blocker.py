"""Slicer-only duct support blockers for support-enabled Obi-Wan carriers.

These volumes are never printable carrier geometry.  They reuse the
authoritative Obi-Wan lumen centerlines/crops, grow 0.25 mm into the
surrounding walls, and retain the carrier's exact route ownership crop.  The
optional LM blockers are additionally clipped at the keyed split plane.  A
compact analytic copy of the same physical lumen centerlines is emitted beside
each modifier so the final G-code can be checked for support/duct collisions
without importing OCC on the slicing workstation.
"""

from __future__ import annotations

from typing import Any, Iterable

from build123d import (
    Box,
    Circle,
    Compound,
    Cylinder,
    Plane,
    Pos,
    Sphere,
    sweep,
)

from lx521_baffle.base import STAND_FOOT
from lx521_baffle.obiwan import route
from lx521_baffle.obiwan.floor import (
    FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM,
    FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM,
    FLOOR_FEED_MOUTH_RELIEF_Z_MM,
    FLOOR_FEED_MOUTH_SHELL_MM,
    FLOOR_LANE_SPECS,
    floor_lane_control_points,
    floor_lane_path,
)
from lx521_baffle.obiwan.lm_split import LM_SPLIT_SEAM_Y
from lx521_baffle.obiwan.rear_entry import (
    _round_tube,
    _round_tube_global_segment,
    _z_axis_bore,
    no_floor_rear_entry_bores,
    no_floor_rear_entry_cap_relief_cutters,
    no_floor_rear_entry_vestibules,
)


DUCT_SUPPORT_BLOCKER_CLEARANCE_MM = 0.25
DUCT_SUPPORT_BLOCKER_BOOLEAN_MARGIN_MM = 0.05
DUCT_SUPPORT_BLOCKER_PART_KEYS = frozenset({
    "core_2_of_2_um_carrier",
    "optional_lm_keyed_1_of_2_bottom",
    "optional_lm_keyed_2_of_2_top",
})
DUCT_SUPPORT_BLOCKER_STL_NAMES = frozenset(
    f"obiwan_{key}" for key in DUCT_SUPPORT_BLOCKER_PART_KEYS)


def _require_guarded_build() -> None:
    import run_memory_guarded as memory_guard
    memory_guard.require_guarded_build(
        "Obi-Wan support-blocker geometry requires run_memory_guarded.py")


def _split_region(part_key: str):
    if part_key == "optional_lm_keyed_1_of_2_bottom":
        low_y, high_y = -400.0, LM_SPLIT_SEAM_Y
    elif part_key == "optional_lm_keyed_2_of_2_top":
        low_y, high_y = LM_SPLIT_SEAM_Y, 600.0
    else:
        raise ValueError(f"unsupported keyed LM blocker part {part_key!r}")
    return Pos(0.0, (low_y + high_y) / 2.0, 0.0) * Box(
        800.0, high_y - low_y, 500.0)


def _clipped_solids(shape, region, label: str):
    clipped = shape & region
    if clipped is None:
        return ()
    solids = tuple(
        solid for solid in clipped.solids() if solid.volume > 1.0e-6)
    if any(not solid.is_valid for solid in solids):
        raise RuntimeError(f"{label}: split-clipped blocker is invalid")
    return solids


def _lm_owner_route_blockers(
    *, clearance_mm: float, region,
) -> Iterable[Any]:
    specs = (
        (
            "um",
            route._owner_cutter_points(
                route.route_cable_points(1.0), "lm"),
            route.CUTTER_R + clearance_mm,
            route.LM_MAIN_CUTTER_SEGMENT_COUNT,
        ),
        (
            "t",
            route._owner_cutter_points(
                route.ts_cable_points(1.0), "lm"),
            route.TS_CUTTER_R + clearance_mm,
            route.LM_T_CUTTER_SEGMENT_COUNT,
        ),
    )
    for name, points, radius, group_count in specs:
        for index in range(group_count):
            segment = _round_tube_global_segment(
                points, radius, index, group_count)
            owned = route._lm_printed_owner_crop(segment, cutter=True)
            if owned is None:
                continue
            yield from _clipped_solids(
                owned, region, f"{name} owner route {index}")


def _um_owner_t_route_blockers(
    *, clearance_mm: float,
) -> Iterable[Any]:
    """Protect the only printed cable lumen owned by the UM carrier."""
    points = route._owner_cutter_points(
        route.ts_cable_points(1.0), "um")
    tube = _round_tube(points, route.TS_CUTTER_R + clearance_mm)
    owned = route._um_owner_crop(tube, cutter=True)
    solids = tuple(
        solid for solid in owned.solids() if solid.volume > 1.0e-6)
    if (not owned.is_valid or not solids
            or any(not solid.is_valid for solid in solids)):
        raise RuntimeError("UM owner T-route blocker is invalid")
    yield from solids


def _no_floor_entry_blockers(
    *, clearance_mm: float, region,
) -> Iterable[Any]:
    # Equal-radius route/bore/vestibule joins leave coincident faces in OCC's
    # otherwise-valid union and tessellate as over-shared edges.  Increasing
    # each successive overlapping tool by one small margin makes the Boolean
    # intersections unambiguous while preserving at least the declared
    # 0.25-mm blocker clearance everywhere.
    margin = DUCT_SUPPORT_BLOCKER_BOOLEAN_MARGIN_MM
    lm_points = route.lm_complete_duct_points(1.0)
    yield from _clipped_solids(
        _round_tube(
            lm_points, route.LM_INTERNAL_DUCT_R + clearance_mm),
        region, "no-floor LM complete duct")
    for bore in no_floor_rear_entry_bores():
        yield from _clipped_solids(
            _z_axis_bore(
                bore.xy, bore.radius_mm + clearance_mm + margin,
                bore.rear_z_mm - clearance_mm - margin,
                bore.inner_z_mm + clearance_mm + margin),
            region, f"no-floor {bore.name} entry bore")
    for vestibule in no_floor_rear_entry_vestibules():
        yield from _clipped_solids(
            Pos(*vestibule.xy, vestibule.center_z_mm)
            * Sphere(
                vestibule.radius_mm + clearance_mm + 2.0 * margin),
            region, f"no-floor {vestibule.name} vestibule")
    for index, relief in enumerate(
            no_floor_rear_entry_cap_relief_cutters(
                clearance_mm + 3.0 * margin)):
        yield from _clipped_solids(
            relief, region, f"no-floor entry cap relief {index}")


def _floor_lane_blockers(
    *, clearance_mm: float, region,
) -> Iterable[Any]:
    for name, spec in FLOOR_LANE_SPECS.items():
        path = floor_lane_path(name)
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(
            spec["diameter_mm"] / 2.0 + clearance_mm)
        yield from _clipped_solids(
            sweep(section, path=path),
            region, f"floor {name} lane")
        if spec["handoff_mode"] == "buried_route_overlap":
            radius = (
                spec["diameter_mm"] / 2.0
                + FLOOR_FEED_MOUTH_SHELL_MM
                + FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM
                + FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM
                + clearance_mm)
            z0, z1 = FLOOR_FEED_MOUTH_RELIEF_Z_MM
            feed = spec["feed_xyz_mm"]
            relief = Pos(
                feed[0], feed[1], (z0 + z1) / 2.0,
            ) * Cylinder(radius, z1 - z0 + 2.0 * clearance_mm)
            yield from _clipped_solids(
                relief,
                region, f"floor {name} feed relief")


def _fuse_components(components: Iterable[Any], label: str):
    """Fuse incrementally; OCC can invalidate the equivalent many-way union."""
    items = list(components)
    if not items:
        raise RuntimeError(f"{label}: blocker component group is empty")
    combined = items[0]
    for index, component in enumerate(items[1:], 1):
        combined = combined.fuse(component).clean()
        if (not combined.is_valid or not combined.solids()
                or any(not solid.is_valid for solid in combined.solids())):
            raise RuntimeError(
                f"{label}: blocker union became invalid at component {index}")
    return combined


def _polyline_region(
    name: str, points, radius_mm: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "polyline_tube",
        "radius_mm": float(radius_mm),
        "points_xyz_mm": [
            [float(value) for value in point] for point in points
        ],
    }


def _collision_regions() -> list[dict[str, Any]]:
    regions = [
        _polyline_region(
            "um_route_lumen", route.route_cable_points(0.75),
            route.CUTTER_R),
        _polyline_region(
            "t_route_lumen", route.ts_cable_points(0.75),
            route.TS_CUTTER_R),
    ]
    if STAND_FOOT:
        for name, spec in FLOOR_LANE_SPECS.items():
            regions.append(_polyline_region(
                f"floor_{name}_lane_lumen",
                floor_lane_control_points(name),
                spec["diameter_mm"] / 2.0))
            if spec["handoff_mode"] == "buried_route_overlap":
                z0, z1 = FLOOR_FEED_MOUTH_RELIEF_Z_MM
                feed = spec["feed_xyz_mm"]
                regions.append(_polyline_region(
                    f"floor_{name}_feed_relief",
                    (
                        (feed[0], feed[1], z0),
                        (feed[0], feed[1], z1),
                    ),
                    (
                        spec["diameter_mm"] / 2.0
                        + FLOOR_FEED_MOUTH_SHELL_MM
                        + FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM
                        + FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM
                    )))
    else:
        regions.append(_polyline_region(
            "lm_internal_and_rear_exit_lumen",
            route.lm_complete_duct_points(0.75),
            route.LM_INTERNAL_DUCT_R))
        for bore in no_floor_rear_entry_bores():
            regions.append(_polyline_region(
                f"no_floor_{bore.name}_entry_bore",
                (
                    (*bore.xy, bore.rear_z_mm),
                    (*bore.xy, bore.inner_z_mm),
                ),
                bore.radius_mm))
        for vestibule in no_floor_rear_entry_vestibules():
            regions.append(_polyline_region(
                f"no_floor_{vestibule.name}_entry_vestibule",
                ((
                    *vestibule.xy, vestibule.center_z_mm,
                ),),
                vestibule.radius_mm))
    return regions


def duct_support_blocker(
    part_key: str,
    clearance_mm: float = DUCT_SUPPORT_BLOCKER_CLEARANCE_MM,
):
    """Return one carrier-owned modifier and source-space duct contract."""
    _require_guarded_build()
    if part_key not in DUCT_SUPPORT_BLOCKER_PART_KEYS:
        raise ValueError(part_key)
    if clearance_mm <= 0.0:
        raise ValueError("support-blocker clearance must be positive")
    if part_key == "core_2_of_2_um_carrier":
        combined = _fuse_components(
            _um_owner_t_route_blockers(clearance_mm=clearance_mm),
            f"{part_key} UM owner T route")
        blocker_solids = tuple(combined.solids())
        if (not combined.is_valid or not blocker_solids
                or any(not solid.is_valid or solid.volume <= 1.0e-6
                       for solid in blocker_solids)):
            raise RuntimeError(
                f"{part_key}: fused duct support blocker is invalid")
        return Compound(children=blocker_solids), {
            "schema_version": 1,
            "coordinate_space": "authoritative_source_mm",
            "owner": "um_carrier",
            "modifier_clearance_mm": float(clearance_mm),
            "regions": [
                _polyline_region(
                    "um_carrier_t_route_lumen",
                    route.ts_cable_points(0.75),
                    route.TS_CUTTER_R),
            ],
        }

    region = _split_region(part_key)
    owner_components = list(_lm_owner_route_blockers(
        clearance_mm=clearance_mm, region=region))
    groups = [_fuse_components(
        owner_components,
        f"{part_key} LM owner routes",
    )]
    if STAND_FOOT:
        extra_components = list(_floor_lane_blockers(
            clearance_mm=clearance_mm, region=region))
        extra_label = f"{part_key} floor lanes"
    else:
        extra_components = list(_no_floor_entry_blockers(
            clearance_mm=clearance_mm, region=region))
        extra_label = f"{part_key} no-floor entries"
    if extra_components:
        groups.append(_fuse_components(extra_components, extra_label))
    combined = _fuse_components(
        groups, f"{part_key} complete duct blocker")
    blocker_solids = tuple(combined.solids())
    if (not combined.is_valid or not blocker_solids
            or any(not solid.is_valid or solid.volume <= 1.0e-6
                   for solid in blocker_solids)):
        raise RuntimeError(
            f"{part_key}: fused duct support blocker is invalid")
    contract = {
        "schema_version": 1,
        "coordinate_space": "authoritative_source_mm",
        "split_half": (
            "bottom" if part_key.endswith("1of2_bottom") else "top"),
        "split_seam_y_mm": float(LM_SPLIT_SEAM_Y),
        "modifier_clearance_mm": float(clearance_mm),
        "regions": _collision_regions(),
    }
    return Compound(children=blocker_solids), contract


# Compatibility name retained for focused callers and older test fixtures.
keyed_lm_duct_support_blocker = duct_support_blocker


__all__ = [
    "DUCT_SUPPORT_BLOCKER_BOOLEAN_MARGIN_MM",
    "DUCT_SUPPORT_BLOCKER_CLEARANCE_MM",
    "DUCT_SUPPORT_BLOCKER_PART_KEYS",
    "DUCT_SUPPORT_BLOCKER_STL_NAMES",
    "duct_support_blocker",
    "keyed_lm_duct_support_blocker",
]
