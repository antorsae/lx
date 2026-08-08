"""Shared Option-B floor-to-upright transition geometry.

The released floor states use one constant-thickness bent wall instead of a
horizontal slab fused into a vertical plate at a hard 90-degree corner.  The
wall centreline is the selected 75 x 65 mm tangent cubic.  Its handles were
solved so the minimum centreline curvature radius is 41 mm without a
curvature reversal.

World axes follow the rest of the baffle project: X is left/right, Y is up
from the floor and Z is depth (positive toward the acoustic front).  The
default floor is Y=0 and the upright rear datum is Z=0.
"""

from __future__ import annotations

import math

from build123d import (
    Bezier,
    Edge,
    Face,
    Line,
    Part,
    Plane,
    Pos,
    Rectangle,
    Wire,
    extrude,
    loft,
    sweep,
)


WALL_THICKNESS_MM = 18.3
WALL_HALF_THICKNESS_MM = WALL_THICKNESS_MM / 2.0

BEND_REAR_SPAN_MM = 75.0
BEND_RISE_MM = 65.0
BEND_HORIZONTAL_HANDLE_MM = 67.5
BEND_VERTICAL_HANDLE_MM = 33.91836734693878

BEND_MIN_CENTERLINE_RADIUS_MM = 41.0002479308
BEND_MIN_RADIUS_PARAMETER = 0.507021
BEND_HORIZONTAL_ENDPOINT_RADIUS_MM = 219.8847352025
BEND_VERTICAL_ENDPOINT_RADIUS_MM = 230.0911111111

# Positive-volume overlap into the retained rear foot and upper upright.
# OCC's exact-offset surface does not produce a numerically useful common
# volume within the first ~2.8 mm of the nearly straight endpoint, even
# though the analytic profiles coincide there to microns.  A 3 mm owner
# overlap is therefore intentional.  It changes the ideal YZ envelope by
# less than 0.026 mm2, while avoiding a face-only or zero-solid Boolean.
FUSION_OVERLAP_MM = 3.0

# OCC's planar offset wire is geometrically valid and meshes locally, but its
# SurfaceOfExtrusion side faces are silently omitted by STEP translation.  Fit
# explicit B-spline boundary curves through the exact analytic parallel-offset
# samples instead. Interpolation retains every sample and both endpoint
# tangents; the dense grid keeps between-sample error far below print/process
# resolution while producing ordinary STEP-safe B-spline extrusion faces.
WALL_OFFSET_SPLINE_SAMPLES = 128
WALL_OFFSET_INTERPOLATION_TOLERANCE_MM = 1.0e-7

# A multisection sweep with 33 stations converges the Stock/Slim lateral
# envelope below 0.001 mm while retaining one six-face STEP-safe solid.  The
# odd count includes the exact cubic midpoint as well as both tangent ends.
LATERAL_HERMITE_SECTIONS = 33

# A variable-thickness wall cannot use the multisection SWEEP: its sections
# are off-centre from the spine, and OCC's pipe-shell relocates each one by
# projecting it back onto the spine, which put the built surface up to
# 0.741 mm off the requested law at mid-arc.  Lofting the same sections
# places every one exactly where it was built, so the law is realised to
# 1e-5 mm.  65 stations converge it to zero at double precision while the
# convex face stays on the exact parallel offset.
VARIABLE_THICKNESS_SECTIONS = 65

# Composite Simpson over |C'(u)| for the centreline arc length.  4096 even
# spans put the quadrature error on this cubic below 1e-11 mm, so the length
# is exact for every downstream use and stable across platforms.
ARC_LENGTH_SAMPLES = 4096


def centerline_controls(
    *,
    x_mm: float = 0.0,
    floor_y_mm: float = 0.0,
    upright_rear_z_mm: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    """Return the exact Option-B cubic controls in world XYZ."""
    horizontal_y = float(floor_y_mm) + WALL_HALF_THICKNESS_MM
    vertical_z = float(upright_rear_z_mm) + WALL_HALF_THICKNESS_MM
    horizontal_z = vertical_z - BEND_REAR_SPAN_MM
    vertical_y = horizontal_y + BEND_RISE_MM
    x = float(x_mm)
    return (
        (x, horizontal_y, horizontal_z),
        (x, horizontal_y,
         horizontal_z + BEND_HORIZONTAL_HANDLE_MM),
        (x, vertical_y - BEND_VERTICAL_HANDLE_MM, vertical_z),
        (x, vertical_y, vertical_z),
    )


def canonical_lane_controls(
    x_mm: float,
    floor_y_mm: float,
    upright_z_mm: float,
) -> tuple[tuple[float, float, float], ...]:
    """Translate the Option-B cubic to one constant-X duct lane.

    The lane has the same 75 x 65 mm side projection and therefore the same
    curvature contract as the wall centreline.  Callers may use a longer
    endpoint/tangent-specific cubic where a qualified upper route requires
    it; this canonical form remains the reference containment datum.
    """
    x = float(x_mm)
    floor_y = float(floor_y_mm)
    upright_z = float(upright_z_mm)
    horizontal_z = upright_z - BEND_REAR_SPAN_MM
    vertical_y = floor_y + BEND_RISE_MM
    return (
        (x, floor_y, horizontal_z),
        (x, floor_y, horizontal_z + BEND_HORIZONTAL_HANDLE_MM),
        (x, vertical_y - BEND_VERTICAL_HANDLE_MM, upright_z),
        (x, vertical_y, upright_z),
    )


def cubic_point(controls, parameter: float) -> tuple[float, float, float]:
    """Evaluate one cubic without importing OCC topology helpers."""
    p0, p1, p2, p3 = controls
    u = float(parameter)
    v = 1.0 - u
    return tuple(
        v ** 3 * p0[axis]
        + 3.0 * v ** 2 * u * p1[axis]
        + 3.0 * v * u ** 2 * p2[axis]
        + u ** 3 * p3[axis]
        for axis in range(3)
    )


def cubic_derivatives(controls, parameter: float):
    """Return first and second XYZ derivatives of one cubic."""
    p0, p1, p2, p3 = controls
    u = float(parameter)
    v = 1.0 - u
    first = tuple(
        3.0 * (
            v ** 2 * (p1[axis] - p0[axis])
            + 2.0 * v * u * (p2[axis] - p1[axis])
            + u ** 2 * (p3[axis] - p2[axis]))
        for axis in range(3)
    )
    second = tuple(
        6.0 * (
            v * (p2[axis] - 2.0 * p1[axis] + p0[axis])
            + u * (p3[axis] - 2.0 * p2[axis] + p1[axis]))
        for axis in range(3)
    )
    return first, second


def curvature_radius(controls, parameter: float) -> float:
    """Return the 3D curvature radius, or infinity on a straight station."""
    first, second = cubic_derivatives(controls, parameter)
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    speed = math.sqrt(sum(value * value for value in first))
    cross_norm = math.sqrt(sum(value * value for value in cross))
    if cross_norm <= 1.0e-15:
        return math.inf
    return speed ** 3 / cross_norm


def bezier_point(controls, parameter: float) -> tuple[float, float, float]:
    """Evaluate a Bezier of any degree at ``parameter``.

    Four control points delegate to :func:`cubic_point` so every released
    caller keeps its exact Bernstein arithmetic; the de Casteljau branch
    exists for the V1L floor lanes, which are quintics.
    """
    if len(controls) == 4:
        return cubic_point(controls, parameter)
    u = float(parameter)
    points = [tuple(float(value) for value in point) for point in controls]
    while len(points) > 1:
        points = [
            tuple(
                points[index][axis] * (1.0 - u) + points[index + 1][axis] * u
                for axis in range(3))
            for index in range(len(points) - 1)
        ]
    return points[0]


def centerline_arc_length(
    parameter: float = 1.0,
    *,
    controls=None,
    samples: int = ARC_LENGTH_SAMPLES,
) -> float:
    """Arc length of the Option-B centreline from u=0 to ``parameter``."""
    if samples < 2 or samples % 2:
        raise ValueError("Simpson arc length needs an even span count >= 2")
    end = float(parameter)
    if not 0.0 <= end <= 1.0:
        raise ValueError("centreline parameter must be in [0, 1]")
    if end == 0.0:
        return 0.0
    controls = centerline_controls() if controls is None else controls

    def speed(u: float) -> float:
        first, _second = cubic_derivatives(controls, u)
        return math.sqrt(sum(value * value for value in first))

    step = end / samples
    total = speed(0.0) + speed(end)
    for index in range(1, samples):
        total += (4.0 if index % 2 else 2.0) * speed(index * step)
    return total * step / 3.0


# The full sweep length, published so the V1L floor ramp can be parameterised
# by path length instead of by Y.
BEND_CENTERLINE_LENGTH_MM = centerline_arc_length()


def sampled_minimum_radius(controls=None, samples: int = 100_000):
    """Dependency-light numerical verification of the minimum radius."""
    if samples < 2:
        raise ValueError("minimum-radius sampling needs at least two spans")
    controls = centerline_controls() if controls is None else controls
    best_radius = math.inf
    best_parameter = 0.0
    for index in range(samples + 1):
        parameter = index / samples
        radius = curvature_radius(controls, parameter)
        if radius < best_radius:
            best_radius = radius
            best_parameter = parameter
    return best_radius, best_parameter


def centerline_wire(*, x_mm: float = 0.0):
    """Exact single-edge Option-B centreline."""
    return Wire(Bezier(*centerline_controls(x_mm=x_mm)).edge())


def bent_wall_prism(
    width_mm: float,
    *,
    center_x_mm: float = 0.0,
) -> Part:
    """Build a constant 18.3-mm Option-B wall across ``width_mm``.

    Build one butt-ended planar YZ profile from analytic parallel-offset
    samples, then extrude it in the explicit -X direction. This retains exact
    Y=0/18.3 and Z=0/18.3 tangent faces and avoids the non-serializable OCC
    offset-wire surfaces formerly emitted by STEP.
    """
    width = float(width_mm)
    if width <= 0.0:
        raise ValueError("bent-wall width must be positive")
    x_face = float(center_x_mm) + width / 2.0
    controls = centerline_controls(x_mm=x_face)
    positive_offset = []
    negative_offset = []
    positive_derivative = []
    negative_derivative = []
    parameters = []
    for index in range(WALL_OFFSET_SPLINE_SAMPLES + 1):
        parameter = index / WALL_OFFSET_SPLINE_SAMPLES
        parameters.append(parameter)
        point = cubic_point(controls, parameter)
        first, second = cubic_derivatives(controls, parameter)
        tangent_norm = math.hypot(first[1], first[2])
        if tangent_norm <= 1.0e-12:
            raise RuntimeError("Option-B offset encountered a zero tangent")
        tangent_norm_derivative = (
            first[1] * second[1] + first[2] * second[2]
        ) / tangent_norm
        normal_y = -first[2] / tangent_norm
        normal_z = first[1] / tangent_norm
        normal_y_derivative = -(
            second[2] * tangent_norm
            - first[2] * tangent_norm_derivative
        ) / tangent_norm ** 2
        normal_z_derivative = (
            second[1] * tangent_norm
            - first[1] * tangent_norm_derivative
        ) / tangent_norm ** 2
        offset = (
            0.0,
            WALL_HALF_THICKNESS_MM * normal_y,
            WALL_HALF_THICKNESS_MM * normal_z,
        )
        positive_offset.append(tuple(
            point[axis] + offset[axis] for axis in range(3)))
        negative_offset.append(tuple(
            point[axis] - offset[axis] for axis in range(3)))
        positive_derivative.append((
            0.0,
            first[1]
            + WALL_HALF_THICKNESS_MM * normal_y_derivative,
            first[2]
            + WALL_HALF_THICKNESS_MM * normal_z_derivative,
        ))
        negative_derivative.append((
            0.0,
            first[1]
            - WALL_HALF_THICKNESS_MM * normal_y_derivative,
            first[2]
            - WALL_HALF_THICKNESS_MM * normal_z_derivative,
        ))

    positive_edge = Edge.make_spline(
        positive_offset,
        tangents=(positive_derivative[0], positive_derivative[-1]),
        parameters=parameters,
        scale=False,
        tol=WALL_OFFSET_INTERPOLATION_TOLERANCE_MM,
    )
    negative_reversed = list(reversed(negative_offset))
    negative_edge = Edge.make_spline(
        negative_reversed,
        tangents=(
            tuple(-value for value in negative_derivative[-1]),
            tuple(-value for value in negative_derivative[0]),
        ),
        parameters=parameters,
        scale=False,
        tol=WALL_OFFSET_INTERPOLATION_TOLERANCE_MM,
    )
    outline = Wire((
        positive_edge,
        Line(positive_offset[-1], negative_offset[-1]),
        negative_edge,
        Line(negative_offset[0], positive_offset[0]),
    ))
    wall = extrude(
        Face(outline), amount=width, dir=(-1.0, 0.0, 0.0))
    solids = tuple(wall.solids())
    if (not wall.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "Option-B bent wall must be one valid solid; "
            f"valid={wall.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def _cubic_hermite(
    start_value: float,
    end_value: float,
    start_derivative: float,
    end_derivative: float,
    parameter: float,
) -> float:
    """Evaluate one scalar cubic Hermite span on ``0 <= parameter <= 1``."""
    u = float(parameter)
    if not 0.0 <= u <= 1.0:
        raise ValueError("Hermite parameter must be in [0, 1]")
    return (
        (2.0 * u ** 3 - 3.0 * u ** 2 + 1.0) * float(start_value)
        + (u ** 3 - 2.0 * u ** 2 + u) * float(start_derivative)
        + (-2.0 * u ** 3 + 3.0 * u ** 2) * float(end_value)
        + (u ** 3 - u ** 2) * float(end_derivative)
    )


def lateral_hermite_bounds(
    *,
    rear_left_x_mm: float,
    rear_right_x_mm: float,
    upright_left_x_mm: float,
    upright_right_x_mm: float,
    rear_left_dx_du: float,
    rear_right_dx_du: float,
    upright_left_dx_du: float,
    upright_right_dx_du: float,
    parameter: float,
) -> tuple[float, float]:
    """Return the left/right bounds of one widening-wall station."""
    return (
        _cubic_hermite(
            rear_left_x_mm,
            upright_left_x_mm,
            rear_left_dx_du,
            upright_left_dx_du,
            parameter,
        ),
        _cubic_hermite(
            rear_right_x_mm,
            upright_right_x_mm,
            rear_right_dx_du,
            upright_right_dx_du,
            parameter,
        ),
    )


def bent_wall_lateral_hermite(
    *,
    rear_left_x_mm: float,
    rear_right_x_mm: float,
    upright_left_x_mm: float,
    upright_right_x_mm: float,
    rear_left_dx_du: float,
    rear_right_dx_du: float,
    upright_left_dx_du: float,
    upright_right_dx_du: float,
    section_count: int | None = None,
    thickness_law=None,
) -> Part:
    """Build the Option-B wall with a smooth, widening lateral envelope.

    ``bent_wall_prism`` is intentionally constant in X and remains the right
    construction for Obi-Wan's W64 stem.  Stock/Slim are different: their
    rear connector foot narrows in X/Z while the lower baffle widens in X/Y.
    Joining two separately clipped constant-width bodies leaves a visible
    diagonal wedge at each side.  This single multisection sweep instead
    interpolates the left and right side boundaries independently, matching
    both endpoint positions and both existing endpoint slopes.

    Derivatives are with respect to the shared cubic parameter ``u``.  By
    default each section stays exactly 18.3 mm thick normal to the Option-B
    centreline; the first section is horizontal and the last is vertical,
    preserving the released rear-foot and upright tangent faces.

    ``thickness_law`` optionally makes the wall thickness a function of the
    cubic parameter, for the V1L floor bottom whose rear-thickness ramp runs
    on through the bend.  The section keeps its CONVEX edge on the exact
    ``WALL_HALF_THICKNESS_MM`` parallel offset and moves only the concave
    one, because the convex face is the floor-contact plane at one end and
    the front-flush plate face at the other -- neither may move.  Omitting
    the law reproduces the released constant-thickness solid exactly.
    """
    if section_count is None:
        section_count = (LATERAL_HERMITE_SECTIONS if thickness_law is None
                         else VARIABLE_THICKNESS_SECTIONS)
    if section_count < 5 or section_count % 2 == 0:
        raise ValueError(
            "lateral Hermite sweep needs an odd section count >= 5")
    if rear_right_x_mm <= rear_left_x_mm:
        raise ValueError("rear lateral bounds are reversed")
    if upright_right_x_mm <= upright_left_x_mm:
        raise ValueError("upright lateral bounds are reversed")

    controls = centerline_controls()
    sections = []
    sampled_bounds = []
    for index in range(section_count):
        parameter = index / (section_count - 1)
        point = cubic_point(controls, parameter)
        tangent, _second = cubic_derivatives(controls, parameter)
        left_x, right_x = lateral_hermite_bounds(
            rear_left_x_mm=rear_left_x_mm,
            rear_right_x_mm=rear_right_x_mm,
            upright_left_x_mm=upright_left_x_mm,
            upright_right_x_mm=upright_right_x_mm,
            rear_left_dx_du=rear_left_dx_du,
            rear_right_dx_du=rear_right_dx_du,
            upright_left_dx_du=upright_left_dx_du,
            upright_right_dx_du=upright_right_dx_du,
            parameter=parameter,
        )
        if right_x - left_x <= 1.0:
            raise RuntimeError(
                "Option-B lateral Hermite envelope collapsed at "
                f"u={parameter:.6f}: {left_x:.6f}..{right_x:.6f}")
        sampled_bounds.append((left_x, right_x))
        section_plane = Plane(
            origin=(
                (left_x + right_x) / 2.0,
                point[1],
                point[2],
            ),
            x_dir=(1.0, 0.0, 0.0),
            z_dir=tangent,
        )
        if thickness_law is None:
            sections.append(
                section_plane
                * Rectangle(right_x - left_x, WALL_THICKNESS_MM)
            )
            continue
        thickness = float(thickness_law(parameter))
        if not 0.0 < thickness <= WALL_THICKNESS_MM:
            raise ValueError(
                "Option-B wall thickness law left (0, 18.3] at "
                f"u={parameter:.6f}: {thickness:.6f}")
        # The section plane's local +Y is z_dir x x_dir, which points at the
        # centre of curvature -- the concave side.  Offsetting the centred
        # rectangle by half the thickness deficit therefore pins the convex
        # edge and lets the concave one lift away.
        sections.append(
            section_plane
            * Pos(0.0, thickness / 2.0 - WALL_HALF_THICKNESS_MM, 0.0)
            * Rectangle(right_x - left_x, thickness)
        )

    if any(
            sampled_bounds[index][0] > sampled_bounds[index - 1][0]
            or sampled_bounds[index][1] < sampled_bounds[index - 1][1]
            for index in range(1, len(sampled_bounds))):
        raise RuntimeError(
            "Option-B lateral Hermite envelope must widen monotonically")

    if thickness_law is None:
        wall = sweep(
            sections,
            path=centerline_wire(),
            multisection=True,
            is_frenet=True,
            clean=True,
        )
    else:
        # See VARIABLE_THICKNESS_SECTIONS: the loft honours the section
        # placement the law asked for, which the spine-projecting pipe shell
        # does not once the profiles stop being centred on the spine.  Both
        # end caps remain the exact planar first/last sections, so the
        # horizontal and vertical tangent joins are unchanged.
        wall = loft(sections, ruled=False)
    solids = tuple(wall.solids())
    if (not wall.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "Option-B lateral Hermite wall must be one valid solid; "
            f"valid={wall.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return Part([solids[0]])


def bend_facts() -> dict:
    """Stable analytic contract for manifests, drawings and tests."""
    controls = centerline_controls()
    return {
        "profile": "option_b_tangent_cubic",
        "wall_thickness_mm": WALL_THICKNESS_MM,
        "rear_span_mm": BEND_REAR_SPAN_MM,
        "rise_mm": BEND_RISE_MM,
        "horizontal_handle_mm": BEND_HORIZONTAL_HANDLE_MM,
        "vertical_handle_mm": BEND_VERTICAL_HANDLE_MM,
        "centerline_controls_xyz_mm": controls,
        "horizontal_tangent_xyz_mm": controls[0],
        "vertical_tangent_xyz_mm": controls[-1],
        "minimum_centerline_radius_mm": BEND_MIN_CENTERLINE_RADIUS_MM,
        "minimum_radius_parameter": BEND_MIN_RADIUS_PARAMETER,
        "horizontal_endpoint_radius_mm": (
            BEND_HORIZONTAL_ENDPOINT_RADIUS_MM),
        "vertical_endpoint_radius_mm": BEND_VERTICAL_ENDPOINT_RADIUS_MM,
        "curvature_reversals": 0,
        "fusion_overlap_mm": FUSION_OVERLAP_MM,
        "offset_spline_samples": WALL_OFFSET_SPLINE_SAMPLES,
        "offset_interpolation_tolerance_mm": (
            WALL_OFFSET_INTERPOLATION_TOLERANCE_MM),
        "lateral_hermite_sections": LATERAL_HERMITE_SECTIONS,
    }
