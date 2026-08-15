"""Reusable pause-and-bury geometry for captive D5 x 2 disc magnets.

This module is the production geometry authority derived from the physically
validated ``coupons/obiwan_ae_embed`` coupon.  It deliberately owns no baffle
outline, driver, route, insert, or backing-pad geometry.  Callers provide an
installed interface datum, and production geometry must already contain the
complete 3.14 mm captive land.  The helper never changes the exterior form.

Coordinate contract
-------------------

``outward`` always points from a base/carrier toward its mating
receiver/attachment.  ``face`` is the shared physical interface datum.  A
receiver cavity-face datum is placed at ``face + interface_gap * outward``;
the offset remains solid and produces a flush 0.57-mm physical receiver skin
(0.05-mm spacing standoff plus the qualified 0.52-mm skin), not a local air
notch.  The helper then derives the material-inward direction from ``owner``:

* ``base`` / ``carrier``: ``-outward``;
* ``receiver`` / ``attachment`` / ``wing``: ``+outward``.

``print_up`` is expressed in the installed/source coordinate system and
points toward later print layers.  ``bed_datum`` is any point on print Z=0.
For the normal front-face-down baffle orientation use ``print_up=(0,0,-1)``
and ``bed_datum=(0,0,front_z)`` (or pass ``front_z`` as a convenience).

The wall-normal helper requires the magnet axis to be perpendicular to the
print-up vector.  This is the exact coupon topology: circular cradle, open
upper half, full-width loading chimney, and a 45-degree gable roof.  The
separate axial helper handles a magnet axis parallel to print-up with a
45-degree conical roof.  It also handles an axis opposed to print-up by
placing the cone between the exterior skin and a deeper cavity.  That latter
layout is the front-face-down treatment: rear skin 0.00..0.52, cone
0.52..3.12, cavity 3.12..5.22, and inner skin through 5.74 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from build123d import (
    Align,
    Box,
    Cone,
    Cylinder,
    Face,
    Plane,
    Pos,
    Rot,
    Vector,
    Wire,
    extrude,
)
from .magnet_contract import (
    BOOLEAN_EPS_MM,
    CAPTIVE_LAND_MM,
    CAVITY_DEPTH_MM,
    CAVITY_DIAMETER_MM,
    DEFAULT_SPEC,
    FACE_SKIN_MM,
    INNER_SKIN_MM,
    INTERFACE_GAP_MM,
    MAGNET_DEPTH_MM,
    MAGNET_DIAMETER_MM,
    MINIMUM_RETAINING_PATH_MM,
    NOMINAL_PAIRED_FACE_SEPARATION_MM,
    ROOF_ANGLE_DEG,
    ROOF_HEIGHT_MM,
    ROOF_PLANE_GRID_MM,
    SIDE_WALL_MARGIN_MM,
    CaptiveMagnetGeometryError,
    CaptiveMagnetSpec,
)


@dataclass(frozen=True)
class PrintFrame:
    """Print build direction and bed plane expressed in source coordinates."""

    print_up: tuple[float, float, float]
    bed_datum: tuple[float, float, float]

    def height_mm(self, point: Sequence[float]) -> float:
        return _dot(_sub(_xyz(point), self.bed_datum), self.print_up)

    def facts(self) -> dict[str, list[float]]:
        return {
            "print_up_source_xyz": list(self.print_up),
            "bed_datum_source_xyz": list(self.bed_datum),
        }


@dataclass(frozen=True)
class CaptiveMagnetTools:
    """BREP tools and serializable datums for one captive magnet station."""

    name: str
    closure_kind: str
    owner: str
    interface_datum_xyz: tuple[float, float, float]
    actual_face_xyz: tuple[float, float, float]
    pair_axis_xyz: tuple[float, float, float]
    material_inward_xyz: tuple[float, float, float]
    print_frame: PrintFrame
    cavity_center_xyz: tuple[float, float, float]
    seated_magnet_center_xyz: tuple[float, float, float]
    raw_roof_start_print_z_mm: float
    roof_start_print_z_mm: float
    roof_apex_print_z_mm: float
    required_min_part_top_print_z_mm: float
    cutters: tuple[Any, ...]
    nominal_magnet: Any
    required_land: Any
    spec: CaptiveMagnetSpec

    def facts(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "closure_kind": self.closure_kind,
            "owner": self.owner,
            "interface_datum_xyz_mm": list(self.interface_datum_xyz),
            "actual_face_xyz_mm": list(self.actual_face_xyz),
            "pair_axis_xyz": list(self.pair_axis_xyz),
            "material_inward_xyz": list(self.material_inward_xyz),
            # The marked/N-pole vector is deliberately the same assembly
            # vector on both mates.  Mirrored sites therefore do not use the
            # same visible face blindly.
            "marked_pole_axis_xyz": list(self.pair_axis_xyz),
            # The loading chimney is open toward later print layers.  The
            # physical insertion motion is therefore the opposite vector:
            # from above the paused part down into the printed cradle.
            "insertion_direction_xyz": list(
                _scale(self.print_frame.print_up, -1.0)),
            "cavity_center_xyz_mm": list(self.cavity_center_xyz),
            "seated_magnet_center_xyz_mm": list(
                self.seated_magnet_center_xyz),
            "raw_roof_start_print_z_mm": self.raw_roof_start_print_z_mm,
            "cavity_bury_roof_start_print_z_mm": self.roof_start_print_z_mm,
            "roof_apex_print_z_mm": self.roof_apex_print_z_mm,
            "required_min_part_top_print_z_mm": (
                self.required_min_part_top_print_z_mm),
            "pause_marker_source": (
                "slice/G-code first closing layer; never CAD-only"),
            "magnet_seating": (
                "seat against the interface-side 0.45-mm skin; "
                "0.10-mm axial cavity allowance remains behind magnet"),
        }
        result.update(self.print_frame.facts())
        result.update(self.spec.facts())
        return result


def _xyz(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise CaptiveMagnetGeometryError(
            f"expected three coordinates, got {values!r}")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise CaptiveMagnetGeometryError(
            f"coordinates must be finite, got {values!r}")
    return result  # type: ignore[return-value]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(x) + float(y) for x, y in zip(a, b))  # type: ignore[return-value]


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(x) - float(y) for x, y in zip(a, b))  # type: ignore[return-value]


def _scale(a: Sequence[float], value: float) -> tuple[float, float, float]:
    return tuple(float(x) * float(value) for x in a)  # type: ignore[return-value]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _unit(values: Sequence[float], label: str) -> tuple[float, float, float]:
    vector = _xyz(values)
    length = math.sqrt(_dot(vector, vector))
    if length <= 1.0e-9:
        raise CaptiveMagnetGeometryError(f"{label} must be non-zero")
    return _scale(vector, 1.0 / length)


def _face_xyz(
    face: Sequence[float], axis_z: float | None,
) -> tuple[float, float, float]:
    if len(face) == 2:
        if axis_z is None:
            raise CaptiveMagnetGeometryError(
                "a 2D face requires axis_z")
        return (float(face[0]), float(face[1]), float(axis_z))
    result = _xyz(face)
    if axis_z is not None and not math.isclose(
            result[2], float(axis_z), abs_tol=1.0e-9):
        raise CaptiveMagnetGeometryError(
            f"face Z {result[2]} conflicts with axis_z {axis_z}")
    return result


def _print_frame(
    *,
    print_up: Sequence[float],
    bed_datum: Sequence[float] | None,
    front_z: float | None,
) -> PrintFrame:
    up = _unit(print_up, "print_up")
    if bed_datum is None:
        if front_z is None:
            raise CaptiveMagnetGeometryError(
                "provide bed_datum, or front_z for front-face-down printing")
        bed = (0.0, 0.0, float(front_z))
    else:
        bed = _xyz(bed_datum)
    return PrintFrame(up, bed)


def _snap_roof_start(
    raw_mm: float,
    spec: CaptiveMagnetSpec,
    explicit_mm: float | None,
) -> float:
    if explicit_mm is None:
        grid = spec.roof_plane_grid_mm
        return math.ceil((raw_mm - 1.0e-9) / grid) * grid
    value = float(explicit_mm)
    if value < raw_mm - 1.0e-6:
        raise CaptiveMagnetGeometryError(
            f"roof start {value:.4f} is below fully open cavity "
            f"height {raw_mm:.4f}")
    return value


def _owner_key(owner: str) -> str:
    value = str(owner).strip().lower()
    aliases = {
        "base": "base",
        "carrier": "base",
        "receiver": "receiver",
        "attachment": "receiver",
        "wing": "receiver",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise CaptiveMagnetGeometryError(
            f"unknown owner {owner!r}; expected base/carrier or "
            "receiver/attachment/wing") from exc


def wall_cavity_tools(
    *,
    name: str,
    face: Sequence[float],
    outward: Sequence[float],
    owner: str = "base",
    axis_z: float | None = None,
    print_up: Sequence[float] = (0.0, 0.0, -1.0),
    bed_datum: Sequence[float] | None = None,
    front_z: float | None = None,
    interface_gap_mm: float = INTERFACE_GAP_MM,
    roof_start_print_z_mm: float | None = None,
    spec: CaptiveMagnetSpec | None = None,
) -> CaptiveMagnetTools:
    """Return exact coupon-style tools for an XY wall-normal magnet.

    ``face`` is the base/carrier interface datum, even for a receiver.  The
    receiver's cavity-face datum is offset by ``interface_gap_mm`` along
    ``outward``.  The offset is deliberately *solid*: together with the
    receiver's 0.52-mm qualified face skin it gives a 0.57-mm physical skin
    behind the shared flush exterior.  This preserves the released magnet-
    face spacing without cutting a local 6.4-mm-wide exterior notch that
    reveals the pocket position.  No cavity cutter overshoots either axial
    skin or touches the shared interface surface.
    """

    spec = spec or CaptiveMagnetSpec(interface_gap_mm=interface_gap_mm)
    if not math.isclose(
            spec.interface_gap_mm, interface_gap_mm, abs_tol=1.0e-9):
        raise CaptiveMagnetGeometryError(
            "interface_gap_mm conflicts with the supplied spec")
    datum = _face_xyz(face, axis_z)
    pair_axis = _unit(outward, "outward")
    if abs(pair_axis[2]) > 1.0e-8:
        raise CaptiveMagnetGeometryError(
            "wall_cavity_tools requires an XY wall normal")
    frame = _print_frame(
        print_up=print_up, bed_datum=bed_datum, front_z=front_z)
    owner_key = _owner_key(owner)
    if owner_key == "base":
        actual_face = datum
        inward = _scale(pair_axis, -1.0)
    else:
        actual_face = _add(datum, _scale(pair_axis, interface_gap_mm))
        inward = pair_axis
    perpendicular = abs(_dot(inward, frame.print_up))
    if perpendicular > 1.0e-7:
        raise CaptiveMagnetGeometryError(
            "coupon gable requires magnet axis perpendicular to print_up; "
            f"abs(dot)={perpendicular:.6g}")

    # Plane local axes: +X material-inward, +Z print-up, +Y their
    # right-handed transverse direction.  This makes every XY wall angle use
    # exactly the same canonical coupon geometry.
    plane = Plane(
        origin=actual_face,
        x_dir=inward,
        z_dir=frame.print_up,
    )
    radius = spec.cavity_radius_mm
    cavity_center = _add(
        actual_face,
        _scale(inward, spec.face_skin_mm + spec.cavity_depth_mm / 2.0),
    )
    seated_center = _add(
        actual_face,
        _scale(inward, spec.face_skin_mm + spec.magnet_depth_mm / 2.0),
    )
    center_print_z = frame.height_mm(cavity_center)
    raw_roof_start = center_print_z + radius
    roof_start = _snap_roof_start(
        raw_roof_start, spec, roof_start_print_z_mm)
    roof_local_z = roof_start - center_print_z

    local_cradle = (
        Pos(spec.face_skin_mm, 0.0, 0.0)
        * Rot(Y=90.0)
        * Cylinder(
            radius,
            spec.cavity_depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    )
    local_chimney = Pos(
        spec.face_skin_mm, -radius, 0.0,
    ) * Box(
        spec.cavity_depth_mm,
        2.0 * radius,
        roof_local_z + spec.boolean_epsilon_mm,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    roof_wire = Wire.make_polygon(
        (
            Vector(spec.face_skin_mm, -radius, roof_local_z),
            Vector(spec.face_skin_mm, +radius, roof_local_z),
            Vector(
                spec.face_skin_mm,
                0.0,
                roof_local_z + spec.roof_height_mm,
            ),
        ),
        close=True,
    )
    local_roof = extrude(
        Face(roof_wire),
        amount=spec.cavity_depth_mm,
        dir=Vector(1.0, 0.0, 0.0),
    )
    local_magnet = (
        Pos(spec.face_skin_mm, 0.0, 0.0)
        * Rot(Y=90.0)
        * Cylinder(
            spec.magnet_diameter_mm / 2.0,
            spec.magnet_depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    )
    land_half_width = radius + spec.side_wall_margin_mm
    land_z0 = -radius - spec.side_wall_margin_mm
    land_z1 = (
        roof_local_z + spec.roof_height_mm + spec.inner_skin_mm
    )
    local_land = Pos(0.0, -land_half_width, land_z0) * Box(
        spec.captive_land_mm,
        2.0 * land_half_width,
        land_z1 - land_z0,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    local_cutters = [local_cradle, local_chimney, local_roof]
    roof_apex = roof_start + spec.roof_height_mm
    return CaptiveMagnetTools(
        name=str(name),
        closure_kind="transverse_gable_45deg",
        owner=owner_key,
        interface_datum_xyz=datum,
        actual_face_xyz=actual_face,
        pair_axis_xyz=pair_axis,
        material_inward_xyz=inward,
        print_frame=frame,
        cavity_center_xyz=cavity_center,
        seated_magnet_center_xyz=seated_center,
        raw_roof_start_print_z_mm=raw_roof_start,
        roof_start_print_z_mm=roof_start,
        roof_apex_print_z_mm=roof_apex,
        required_min_part_top_print_z_mm=roof_apex + spec.inner_skin_mm,
        cutters=tuple(plane * item for item in local_cutters),
        nominal_magnet=plane * local_magnet,
        required_land=plane * local_land,
        spec=spec,
    )


def _parallel_plane(
    origin: Sequence[float], print_up: Sequence[float],
) -> Plane:
    up = _unit(print_up, "print_up")
    seed = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)
    x_dir = _unit(_cross(seed, up), "axial local X")
    return Plane(origin=_xyz(origin), x_dir=x_dir, z_dir=up)


def axial_cavity_tools(
    *,
    name: str,
    face: Sequence[float],
    inward: Sequence[float],
    print_up: Sequence[float],
    bed_datum: Sequence[float] | None = None,
    front_z: float | None = None,
    pair_axis: Sequence[float] | None = None,
    roof_start_print_z_mm: float | None = None,
    spec: CaptiveMagnetSpec = DEFAULT_SPEC,
) -> CaptiveMagnetTools:
    """Return a vertically loadable cavity with a 45-degree conical roof.

    Two exact layouts are supported:

    * ``inward == print_up``: exterior skin, cavity, then closing cone;
    * ``inward == -print_up``: exterior skin, expanding cone, then the
      deeper cavity.  This second form keeps the host front-face-down while
      moving its rear-axis magnet centre 4.17 mm inward from the rear face.

    Oblique axes are rejected because neither layout is the qualified coupon
    topology.
    """

    datum = _xyz(face)
    inward_axis = _unit(inward, "inward")
    frame = _print_frame(
        print_up=print_up, bed_datum=bed_datum, front_z=front_z)
    parallel = _dot(inward_axis, frame.print_up)
    if abs(parallel) < 1.0 - 1.0e-7:
        raise CaptiveMagnetGeometryError(
            "axial cavity requires material inward parallel or opposed to "
            f"print_up; dot={parallel:.6g}")
    marked_axis = _unit(
        pair_axis if pair_axis is not None else _scale(inward_axis, -1.0),
        "pair_axis",
    )
    # Canonical local +Z is always material-inward.  Print order is +local Z
    # for the aligned layout and -local Z for the front-down layout.
    plane = _parallel_plane(datum, inward_axis)
    radius = spec.cavity_radius_mm
    face_print_z = frame.height_mm(datum)
    land_radius = radius + spec.side_wall_margin_mm

    if parallel > 0.0:
        cavity_z0 = spec.face_skin_mm
        cavity_center_local_z = cavity_z0 + spec.cavity_depth_mm / 2.0
        seated_center_local_z = cavity_z0 + spec.magnet_depth_mm / 2.0
        cavity_top = (
            face_print_z + spec.face_skin_mm + spec.cavity_depth_mm)
        raw_roof_start = cavity_top
        roof_start = _snap_roof_start(
            cavity_top, spec, roof_start_print_z_mm)
        roof_local_z = roof_start - face_print_z
        local_cavity = Pos(0.0, 0.0, cavity_z0) * Cylinder(
            radius, spec.cavity_depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        chimney_height = max(
            0.0,
            roof_local_z - (spec.face_skin_mm + spec.cavity_depth_mm),
        )
        local_chimney = Pos(
            0.0, 0.0, spec.face_skin_mm + spec.cavity_depth_mm,
        ) * Cylinder(
            radius,
            chimney_height + spec.boolean_epsilon_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        local_roof = Pos(0.0, 0.0, roof_local_z) * Cone(
            radius, 0.0, spec.roof_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        local_magnet = Pos(0.0, 0.0, cavity_z0) * Cylinder(
            spec.magnet_diameter_mm / 2.0, spec.magnet_depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        land_depth = (
            roof_local_z + spec.roof_height_mm + spec.inner_skin_mm)
        local_land = Cylinder(
            land_radius, land_depth,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        roof_apex = roof_start + spec.roof_height_mm
        required_top = roof_apex + spec.inner_skin_mm
        closure_kind = "axis_parallel_conical_45deg"
    else:
        # Front-face-down: printing advances toward the rear exterior.
        # Put the complete circular cavity deeper than the cone so it is open
        # when paused, then close toward the rear and finish the 0.45-mm skin.
        roof_apex_local_z = spec.face_skin_mm
        cavity_z0 = roof_apex_local_z + spec.roof_height_mm
        cavity_center_local_z = cavity_z0 + spec.cavity_depth_mm / 2.0
        # The vertical magnet is centred in its 0.10-mm axial allowance; this
        # is the explicit datum requested for a rear-axis site.
        seated_center_local_z = cavity_center_local_z
        local_roof = Pos(0.0, 0.0, roof_apex_local_z) * Cone(
            0.0, radius, spec.roof_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        local_cavity = Pos(0.0, 0.0, cavity_z0) * Cylinder(
            radius, spec.cavity_depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        # A zero-length nominal chimney is represented by the small Boolean
        # overlap only; it joins the circular cavity and cone robustly.
        local_chimney = Pos(0.0, 0.0, cavity_z0) * Cylinder(
            radius, spec.boolean_epsilon_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        local_magnet = Pos(
            0.0, 0.0,
            seated_center_local_z - spec.magnet_depth_mm / 2.0,
        ) * Cylinder(
            spec.magnet_diameter_mm / 2.0, spec.magnet_depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        land_depth = cavity_z0 + spec.cavity_depth_mm + spec.inner_skin_mm
        local_land = Cylinder(
            land_radius, land_depth,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        raw_roof_start = frame.height_mm(
            _add(datum, _scale(inward_axis, cavity_z0)))
        if (roof_start_print_z_mm is not None
                and not math.isclose(
                    float(roof_start_print_z_mm), raw_roof_start,
                    abs_tol=1.0e-6)):
            raise CaptiveMagnetGeometryError(
                "the opposed axial layout has an exact 0.45/45deg/cavity "
                f"stack; roof_start must be {raw_roof_start:.6f} mm")
        roof_start = raw_roof_start
        roof_apex = frame.height_mm(
            _add(datum, _scale(inward_axis, roof_apex_local_z)))
        required_top = face_print_z
        closure_kind = "axis_opposed_conical_45deg"

    cavity_center = _add(
        datum, _scale(inward_axis, cavity_center_local_z))
    seated_center = _add(
        datum, _scale(inward_axis, seated_center_local_z))
    return CaptiveMagnetTools(
        name=str(name),
        closure_kind=closure_kind,
        owner="direct",
        interface_datum_xyz=datum,
        actual_face_xyz=datum,
        pair_axis_xyz=marked_axis,
        material_inward_xyz=inward_axis,
        print_frame=frame,
        cavity_center_xyz=cavity_center,
        seated_magnet_center_xyz=seated_center,
        raw_roof_start_print_z_mm=raw_roof_start,
        roof_start_print_z_mm=roof_start,
        roof_apex_print_z_mm=roof_apex,
        required_min_part_top_print_z_mm=required_top,
        cutters=tuple(plane * item for item in (
            local_cavity, local_chimney, local_roof)),
        nominal_magnet=plane * local_magnet,
        required_land=plane * local_land,
        spec=spec,
    )


def _apply_tools(
    part: Any,
    tools: CaptiveMagnetTools,
) -> Any:
    """Subtract one qualified cavity without changing the host exterior.

    Production callers must provide an immutable host that already contains
    ``tools.required_land``.  Positive magnet-local additions are
    intentionally not supported: a backing pad, cap, boss, or bevel restore
    would reveal the pocket from an exterior surface.
    """
    missing = tools.required_land - part
    missing_volume = sum(float(solid.volume) for solid in missing.solids())
    if missing_volume > 0.02:
        raise CaptiveMagnetGeometryError(
            f"{tools.name}: immutable host misses "
            f"{missing_volume:.4f} mm3 of required captive land")

    result = part
    for cutter in tools.cutters:
        result = result - cutter
    return result.clean()


def apply_wall_cavity(
    part: Any,
    **kwargs: Any,
) -> tuple[Any, CaptiveMagnetTools]:
    """Subtract a wall cavity from an already-qualified immutable host."""

    tools = wall_cavity_tools(**kwargs)
    return _apply_tools(part, tools), tools


def apply_axial_cavity(
    part: Any,
    **kwargs: Any,
) -> tuple[Any, CaptiveMagnetTools]:
    """Subtract an axial cavity from an already-qualified immutable host."""

    tools = axial_cavity_tools(**kwargs)
    return _apply_tools(part, tools), tools


def pair_facts(
    base: CaptiveMagnetTools,
    receiver: CaptiveMagnetTools,
) -> Mapping[str, object]:
    """Return the coaxial-gap/polarity contract for a completed pair."""

    if base.owner != "base" or receiver.owner != "receiver":
        raise CaptiveMagnetGeometryError(
            "pair_facts expects base tools followed by receiver tools")
    axis_dot = _dot(base.pair_axis_xyz, receiver.pair_axis_xyz)
    if axis_dot < 1.0 - 1.0e-8:
        raise CaptiveMagnetGeometryError("paired magnet axes disagree")
    face_delta = _sub(receiver.actual_face_xyz, base.actual_face_xyz)
    gap = _dot(face_delta, base.pair_axis_xyz)
    transverse = _sub(face_delta, _scale(base.pair_axis_xyz, gap))
    coaxial_error = math.sqrt(_dot(transverse, transverse))
    if coaxial_error > 1.0e-6:
        raise CaptiveMagnetGeometryError(
            f"paired faces are not coaxial: error={coaxial_error:.6g} mm")
    return {
        "base_site": base.name,
        "receiver_site": receiver.name,
        "interface_gap_mm": gap,
        "nominal_magnet_face_separation_mm": (
            base.spec.face_skin_mm + gap + receiver.spec.face_skin_mm),
        "pair_axis_xyz": list(base.pair_axis_xyz),
        "base_marked_pole_axis_xyz": list(base.pair_axis_xyz),
        "receiver_marked_pole_axis_xyz": list(receiver.pair_axis_xyz),
        "polarity_instruction": (
            "marked/N pole vector points along pair axis in both pieces"),
    }


def design_facts() -> dict[str, object]:
    """Top-level serializable authority used by docs/manifests/tests."""

    return {
        "system": "pause_and_bury_captive_disc_magnet",
        "reference": "coupons/obiwan_ae_embed",
        "normal_print_orientation": "front_face_down",
        "wall_closure": "circular cradle + chimney + 45deg gable",
        "axial_closure": "circular cavity + chimney + 45deg cone",
        "pause_authority": "actual sliced G-code first closing layer",
        **DEFAULT_SPEC.facts(),
    }


__all__ = [
    "MAGNET_DIAMETER_MM",
    "MAGNET_DEPTH_MM",
    "CAVITY_DIAMETER_MM",
    "CAVITY_DEPTH_MM",
    "FACE_SKIN_MM",
    "INNER_SKIN_MM",
    "INTERFACE_GAP_MM",
    "ROOF_ANGLE_DEG",
    "ROOF_HEIGHT_MM",
    "CAPTIVE_LAND_MM",
    "NOMINAL_PAIRED_FACE_SEPARATION_MM",
    "CaptiveMagnetGeometryError",
    "CaptiveMagnetSpec",
    "CaptiveMagnetTools",
    "PrintFrame",
    "DEFAULT_SPEC",
    "wall_cavity_tools",
    "axial_cavity_tools",
    "apply_wall_cavity",
    "apply_axial_cavity",
    "pair_facts",
    "design_facts",
]
