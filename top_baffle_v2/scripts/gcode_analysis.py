"""G-code parsing, profile metrics and closure-layer discovery."""

from __future__ import annotations

from release_validation import *


@dataclasses.dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    e_delta: float
    feature: str
    line_width: float | None
    line_number: int
    z0: float = 0.0
    z1: float = 0.0
    path_id: int = 0

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclasses.dataclass
class Layer:
    z: float
    layer_height: float | None
    segments: list[Segment]
    line_number: int
    first_extrusion_line_number: int | None = None


@dataclasses.dataclass
class ParsedGcode:
    layers: list[Layer]
    movement_commands: int
    arc_commands: int
    extrusion_moves: int
    temperature_commands: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    config: dict[str, str]


_ARG_RE = re.compile(
    r"(?:^|\s)([XYZEFIJKR])"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def _segment_intersects_regions(
    x0: float, y0: float, x1: float, y1: float,
    regions: Sequence[tuple[float, float, float, float]] | None,
) -> bool:
    if regions is None:
        return True
    low_x, high_x = min(x0, x1), max(x0, x1)
    low_y, high_y = min(y0, y1), max(y0, y1)
    return any(
        low_x <= region[2] and high_x >= region[0]
        and low_y <= region[3] and high_y >= region[1]
        for region in regions
    )


def parse_gcode(
    path: Path,
    *,
    retain_regions: Sequence[tuple[float, float, float, float]] | None = None,
    retain_feature_prefixes: Sequence[str] | None = None,
) -> ParsedGcode:
    """Parse Bambu G-code and retain only requested local extrusion paths.

    G2/G3 arcs are tessellated while streaming so I/J arc fitting cannot hide
    a circular retaining wall or loading obstruction.  Production calls pass
    cavity ROIs, bounding retained memory independently of overall part size;
    all arc subpoints still contribute to global motion bounds.
    """
    retained_prefixes = (
        tuple(value.strip().lower() for value in retain_feature_prefixes)
        if retain_feature_prefixes is not None else None)
    layers: list[Layer] = []
    pending_change = False
    current: Layer | None = None
    x = y = z = e = 0.0
    xyz_absolute = True
    e_absolute = False
    feature = "Undefined"
    line_width: float | None = None
    movement = arcs = extrusion = temperatures = 0
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    config: dict[str, str] = {}
    in_config = False
    next_path_id = 0
    active_path_id: int | None = None
    last_extrusion_end: tuple[float, float, float] | None = None

    def extrusion_path_id(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> int:
        nonlocal next_path_id, active_path_id, last_extrusion_end
        connected = (
            active_path_id is not None
            and last_extrusion_end is not None
            and math.dist(start, last_extrusion_end) <= 1.0e-5)
        if not connected:
            next_path_id += 1
            active_path_id = next_path_id
        last_extrusion_end = end
        return active_path_id
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if line == "; CONFIG_BLOCK_START":
                in_config = True
                continue
            if line == "; CONFIG_BLOCK_END":
                in_config = False
                continue
            if in_config and line.startswith("; ") and " = " in line:
                key, value = line[2:].split(" = ", 1)
                config[key.strip()] = value.strip()
                continue
            if line == "; CHANGE_LAYER":
                pending_change = True
                continue
            if pending_change and line.startswith("; Z_HEIGHT:"):
                layer_z = _float(line.split(":", 1)[1], "G-code layer Z")
                current = Layer(layer_z, None, [], line_number)
                layers.append(current)
                pending_change = False
                continue
            # Bambu repeats ``LAYER_HEIGHT`` inside individual Bridge feature
            # blocks, where it describes the bridge-flow bead rather than the
            # scheduled layer.  Only the first value immediately following a
            # CHANGE_LAYER/Z_HEIGHT header is authoritative for the layer
            # schedule; never let later feature metadata overwrite it.
            if (current is not None and current.layer_height is None
                    and line.startswith("; LAYER_HEIGHT:")):
                current.layer_height = _float(
                    line.split(":", 1)[1], "G-code layer height")
                continue
            if line.startswith("; FEATURE:"):
                feature = line.split(":", 1)[1].strip()
                continue
            if line.startswith("; LINE_WIDTH:"):
                try:
                    line_width = float(line.split(":", 1)[1])
                except ValueError:
                    line_width = None
                continue
            command = line.split(";", 1)[0].strip()
            if not command:
                continue
            token = command.split(None, 1)[0]
            if token == "G90.1":
                raise AuditError(
                    f"{path}:{line_number}: absolute arc-centre mode G90.1 "
                    "is unsupported; expected relative I/J")
            if token == "G91.1":
                # Explicitly confirms the normal relative-I/J convention.
                continue
            if token == "G90":
                xyz_absolute = True
                continue
            if token == "G91":
                xyz_absolute = False
                continue
            if token == "M82":
                e_absolute = True
                continue
            if token == "M83":
                e_absolute = False
                continue
            if token == "G92":
                args = {key: float(value) for key, value in _ARG_RE.findall(command)}
                x = args.get("X", x)
                y = args.get("Y", y)
                z = args.get("Z", z)
                e = args.get("E", e)
                continue
            if token in ("M104", "M109", "M140", "M190"):
                temperatures += 1
            if token not in ("G0", "G1", "G2", "G3"):
                continue
            movement += 1
            args = {key: float(value) for key, value in _ARG_RE.findall(command)}
            old_x, old_y, old_z, old_e = x, y, z, e
            if xyz_absolute:
                x = args.get("X", x)
                y = args.get("Y", y)
                z = args.get("Z", z)
            else:
                x += args.get("X", 0.0)
                y += args.get("Y", 0.0)
                z += args.get("Z", 0.0)
            if "E" in args:
                if e_absolute:
                    e = args["E"]
                    e_delta = e - old_e
                else:
                    e_delta = args["E"]
                    e += e_delta
            else:
                e_delta = 0.0
            if token in ("G2", "G3"):
                arcs += 1
                if "R" in args:
                    raise AuditError(
                        f"{path}:{line_number}: radius-encoded {token} is "
                        "unsupported; expected relative I/J")
                if "I" not in args and "J" not in args:
                    raise AuditError(
                        f"{path}:{line_number}: {token} lacks relative I/J")
                center_x = old_x + args.get("I", 0.0)
                center_y = old_y + args.get("J", 0.0)
                start_radius = math.hypot(old_x - center_x, old_y - center_y)
                end_radius = math.hypot(x - center_x, y - center_y)
                if start_radius <= FLOAT_EPS:
                    raise AuditError(
                        f"{path}:{line_number}: {token} has zero I/J radius")
                same_endpoint = math.hypot(x - old_x, y - old_y) <= 1.0e-4
                if (not same_endpoint
                        and abs(end_radius - start_radius)
                        > ARC_RADIUS_TOLERANCE_MM):
                    raise AuditError(
                        f"{path}:{line_number}: {token} start/end radii differ "
                        f"by {abs(end_radius - start_radius):.4f} mm")
                start_angle = math.atan2(
                    old_y - center_y, old_x - center_x)
                end_angle = math.atan2(y - center_y, x - center_x)
                if same_endpoint:
                    sweep = -2.0 * math.pi if token == "G2" else 2.0 * math.pi
                    end_radius = start_radius
                elif token == "G2":
                    magnitude = (start_angle - end_angle) % (2.0 * math.pi)
                    sweep = -(magnitude or 2.0 * math.pi)
                else:
                    magnitude = (end_angle - start_angle) % (2.0 * math.pi)
                    sweep = magnitude or 2.0 * math.pi
                mean_radius = (start_radius + end_radius) / 2.0
                planar_length = abs(sweep) * mean_radius
                path_length = math.hypot(planar_length, z - old_z)
                count = max(
                    1, int(math.ceil(path_length / ARC_TESSELLATION_STEP_MM)))
                if count > MAX_ARC_TESSELLATION_SEGMENTS:
                    raise AuditError(
                        f"{path}:{line_number}: {token} needs {count} "
                        "tessellation segments; refusing unbounded arc")
                if (current is not None and e_delta > 1.0e-8
                        and planar_length > 1.0e-8
                        and current.first_extrusion_line_number is None):
                    current.first_extrusion_line_number = line_number
                segment_path_id = 0
                if e_delta > 1.0e-8 and planar_length > 1.0e-8:
                    segment_path_id = extrusion_path_id(
                        (old_x, old_y, old_z), (x, y, z))
                else:
                    active_path_id = None
                    last_extrusion_end = None
                prior_x, prior_y, prior_z = old_x, old_y, old_z
                for index in range(1, count + 1):
                    t = index / count
                    angle = start_angle + sweep * t
                    radius = start_radius + (end_radius - start_radius) * t
                    next_x = center_x + radius * math.cos(angle)
                    next_y = center_y + radius * math.sin(angle)
                    next_z = old_z + (z - old_z) * t
                    if index == count:
                        next_x, next_y, next_z = x, y, z
                    for axis, value in enumerate((next_x, next_y, next_z)):
                        mins[axis] = min(mins[axis], value)
                        maxs[axis] = max(maxs[axis], value)
                    retain_feature = (
                        retained_prefixes is None
                        or feature.lower().startswith(retained_prefixes))
                    if (current is not None and e_delta > 1.0e-8
                            and math.hypot(
                                next_x - prior_x,
                                next_y - prior_y) > 1.0e-8
                            and retain_feature
                            and _segment_intersects_regions(
                                prior_x, prior_y, next_x, next_y,
                                retain_regions)):
                        current.segments.append(Segment(
                            prior_x, prior_y, next_x, next_y,
                            e_delta / count, feature, line_width, line_number,
                            prior_z, next_z, segment_path_id))
                    prior_x, prior_y, prior_z = next_x, next_y, next_z
                if e_delta > 1.0e-8 and planar_length > 1.0e-8:
                    extrusion += 1
                    if same_endpoint:
                        # A following full circle is another traversal even if
                        # it begins at the exact same coordinate.
                        active_path_id = None
                        last_extrusion_end = None
            else:
                for axis, value in enumerate((old_x, old_y, old_z)):
                    mins[axis] = min(mins[axis], value)
                    maxs[axis] = max(maxs[axis], value)
                for axis, value in enumerate((x, y, z)):
                    mins[axis] = min(mins[axis], value)
                    maxs[axis] = max(maxs[axis], value)
                if (current is not None and e_delta > 1.0e-8
                        and math.hypot(x - old_x, y - old_y) > 1.0e-8):
                    if current.first_extrusion_line_number is None:
                        current.first_extrusion_line_number = line_number
                    segment_path_id = extrusion_path_id(
                        (old_x, old_y, old_z), (x, y, z))
                    retain_feature = (
                        retained_prefixes is None
                        or feature.lower().startswith(retained_prefixes))
                    if (retain_feature and _segment_intersects_regions(
                            old_x, old_y, x, y, retain_regions)):
                        current.segments.append(Segment(
                            old_x, old_y, x, y, e_delta, feature,
                            line_width, line_number, old_z, z,
                            segment_path_id))
                    extrusion += 1
                elif math.hypot(x - old_x, y - old_y) > 1.0e-8:
                    active_path_id = None
                    last_extrusion_end = None
    if not layers:
        raise AuditError(f"no Bambu CHANGE_LAYER/Z_HEIGHT records in {path}")
    if not movement or not extrusion or not temperatures:
        raise AuditError(
            f"invalid G-code {path}: moves={movement}, extrusion={extrusion}, "
            f"temperatures={temperatures}")
    return ParsedGcode(
        layers, movement, arcs, extrusion, temperatures,
        tuple(mins), tuple(maxs), config)  # type: ignore[arg-type]


def _segment_distance_3d(
    first_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    second_end: Sequence[float],
) -> float:
    """Shortest Euclidean distance between two finite 3-D segments."""
    p0 = tuple(float(value) for value in first_start)
    p1 = tuple(float(value) for value in first_end)
    q0 = tuple(float(value) for value in second_start)
    q1 = tuple(float(value) for value in second_end)
    first_direction = tuple(
        p1[index] - p0[index] for index in range(3))
    second_direction = tuple(
        q1[index] - q0[index] for index in range(3))
    separation = tuple(
        p0[index] - q0[index] for index in range(3))

    def dot(left, right):
        return sum(left[index] * right[index] for index in range(3))

    def clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    first_length_sq = dot(first_direction, first_direction)
    second_length_sq = dot(second_direction, second_direction)
    small = 1.0e-12
    if first_length_sq <= small and second_length_sq <= small:
        return math.dist(p0, q0)
    if first_length_sq <= small:
        first_parameter = 0.0
        second_parameter = clamp01(
            dot(second_direction, separation) / second_length_sq)
    elif second_length_sq <= small:
        second_parameter = 0.0
        first_parameter = clamp01(
            -dot(first_direction, separation) / first_length_sq)
    else:
        first_second_dot = dot(first_direction, second_direction)
        first_separation_dot = dot(first_direction, separation)
        second_separation_dot = dot(second_direction, separation)
        denominator = (
            first_length_sq * second_length_sq
            - first_second_dot * first_second_dot)
        if denominator > small:
            first_parameter = clamp01((
                first_second_dot * second_separation_dot
                - first_separation_dot * second_length_sq
            ) / denominator)
        else:
            first_parameter = 0.0
        second_parameter = (
            first_second_dot * first_parameter + second_separation_dot
        ) / second_length_sq
        if second_parameter < 0.0:
            second_parameter = 0.0
            first_parameter = clamp01(
                -first_separation_dot / first_length_sq)
        elif second_parameter > 1.0:
            second_parameter = 1.0
            first_parameter = clamp01(
                (first_second_dot - first_separation_dot)
                / first_length_sq)
    delta = tuple(
        separation[index]
        + first_parameter * first_direction[index]
        - second_parameter * second_direction[index]
        for index in range(3))
    return math.sqrt(dot(delta, delta))


def _support_duct_capsules(
    contract: Mapping[str, Any],
    source_to_stl_matrix: BambuMatrix4,
    stl_to_bed_matrix: BambuMatrix4,
) -> list[dict[str, Any]]:
    split_half = contract.get("split_half")
    seam_y = (
        float(contract["split_seam_y_mm"])
        if split_half is not None else None)
    capsules: list[dict[str, Any]] = []
    for region in contract["regions"]:
        radius = float(region["radius_mm"])
        points = [
            tuple(float(value) for value in point)
            for point in region["points_xyz_mm"]
        ]
        pairs = (
            [(points[0], points[0])]
            if len(points) == 1 else list(zip(points[:-1], points[1:]))
        )
        for source_start, source_end in pairs:
            if (split_half == "bottom"
                    and min(source_start[1], source_end[1])
                    > float(seam_y) + radius):
                continue
            if (split_half == "top"
                    and max(source_start[1], source_end[1])
                    < float(seam_y) - radius):
                continue
            stl_start = transform_bambu_point(
                source_to_stl_matrix, source_start)
            stl_end = transform_bambu_point(
                source_to_stl_matrix, source_end)
            capsules.append({
                "region": str(region["name"]),
                "start": transform_bambu_point(
                    stl_to_bed_matrix, stl_start),
                "end": transform_bambu_point(
                    stl_to_bed_matrix, stl_end),
                "radius_mm": radius,
            })
    if not capsules:
        raise AuditError("duct collision contract produced no capsules")
    return capsules


def audit_support_toolpaths_vs_ducts(
    *,
    gcode: Path,
    contract: Mapping[str, Any],
    source_to_stl_matrix: BambuMatrix4,
    stl_to_bed_matrix: BambuMatrix4,
) -> dict[str, Any]:
    """Fail if any deposited support bead enters a functional cable lumen."""
    capsules = _support_duct_capsules(
        contract, source_to_stl_matrix, stl_to_bed_matrix)
    roi_margin = max(
        float(capsule["radius_mm"]) for capsule in capsules) + 1.0
    retain_regions = [
        (
            min(capsule["start"][0], capsule["end"][0]) - roi_margin,
            min(capsule["start"][1], capsule["end"][1]) - roi_margin,
            max(capsule["start"][0], capsule["end"][0]) + roi_margin,
            max(capsule["start"][1], capsule["end"][1]) + roi_margin,
        )
        for capsule in capsules
    ]
    parsed = parse_gcode(
        gcode, retain_regions=retain_regions,
        retain_feature_prefixes=("support",))

    cell_size = 8.0
    index: dict[tuple[int, int, int], set[int]] = {}

    def cell_range(low: float, high: float):
        return range(
            math.floor(low / cell_size),
            math.floor(high / cell_size) + 1)

    for capsule_index, capsule in enumerate(capsules):
        radius = float(capsule["radius_mm"])
        start = capsule["start"]
        end = capsule["end"]
        for ix in cell_range(
                min(start[0], end[0]) - radius,
                max(start[0], end[0]) + radius):
            for iy in cell_range(
                    min(start[1], end[1]) - radius,
                    max(start[1], end[1]) + radius):
                for iz in cell_range(
                        min(start[2], end[2]) - radius,
                        max(start[2], end[2]) + radius):
                    index.setdefault((ix, iy, iz), set()).add(
                        capsule_index)

    support_segments = 0
    candidate_checks = 0
    minimum_clearance = math.inf
    closest: dict[str, Any] | None = None
    collisions: list[dict[str, Any]] = []
    fallback_height = float(parsed.config.get("layer_height", 0.2))
    for layer in parsed.layers:
        layer_height = (
            float(layer.layer_height)
            if layer.layer_height is not None else fallback_height)
        for segment in layer.segments:
            if not segment.feature.lower().startswith("support"):
                continue
            support_segments += 1
            line_width = (
                float(segment.line_width)
                if segment.line_width is not None
                and segment.line_width > 0.0 else FALLBACK_LINE_WIDTH_MM)
            bead_radius = 0.5 * math.hypot(line_width, layer_height)
            start = (
                segment.x0, segment.y0,
                segment.z0 - layer_height / 2.0)
            end = (
                segment.x1, segment.y1,
                segment.z1 - layer_height / 2.0)
            candidates: set[int] = set()
            for ix in cell_range(
                    min(start[0], end[0]) - bead_radius,
                    max(start[0], end[0]) + bead_radius):
                for iy in cell_range(
                        min(start[1], end[1]) - bead_radius,
                        max(start[1], end[1]) + bead_radius):
                    for iz in cell_range(
                            min(start[2], end[2]) - bead_radius,
                            max(start[2], end[2]) + bead_radius):
                        candidates.update(index.get((ix, iy, iz), ()))
            for capsule_index in candidates:
                capsule = capsules[capsule_index]
                candidate_checks += 1
                distance = _segment_distance_3d(
                    start, end, capsule["start"], capsule["end"])
                clearance = (
                    distance - float(capsule["radius_mm"]) - bead_radius)
                evidence = {
                    "gcode_line_number": segment.line_number,
                    "layer_z_mm": layer.z,
                    "feature": segment.feature,
                    "line_width_mm": line_width,
                    "layer_height_mm": layer_height,
                    "duct_region": capsule["region"],
                    "centerline_distance_mm": distance,
                    "duct_radius_mm": capsule["radius_mm"],
                    "support_bead_radius_mm": bead_radius,
                    "bead_to_lumen_clearance_mm": clearance,
                }
                if clearance < minimum_clearance:
                    minimum_clearance = clearance
                    closest = evidence
                if clearance <= 0.0 and len(collisions) < 25:
                    collisions.append(evidence)
    if collisions:
        first = collisions[0]
        raise AuditError(
            "support toolpath enters a cable duct: "
            f"{first['duct_region']} at G-code line "
            f"{first['gcode_line_number']} has "
            f"{first['bead_to_lumen_clearance_mm']:.4f} mm clearance")
    return {
        "status": "pass",
        "gate": "support_extrusion_bead_vs_functional_duct_capsules",
        "support_extrusion_segments_checked": support_segments,
        "duct_capsule_segment_count": len(capsules),
        "candidate_distance_checks": candidate_checks,
        "collision_count": 0,
        "minimum_bead_to_lumen_clearance_mm": (
            minimum_clearance if math.isfinite(minimum_clearance) else None),
        "closest_approach": closest,
    }


def _validate_actual_gcode_profile(
    parsed: ParsedGcode,
    profile_bundle: Mapping[str, Any],
) -> list[str]:
    """Verify slicer output, not merely requested profile inputs."""
    errors = []
    expected = profile_bundle["identity"]["effective"]
    actual_fields = {
        "layer_height": expected["layer_height_mm"],
        "initial_layer_print_height": expected["first_layer_height_mm"],
        "outer_wall_line_width": expected["outer_wall_line_width_mm"],
        "inner_wall_line_width": expected["inner_wall_line_width_mm"],
        "wall_loops": expected["wall_loops"],
        "top_shell_layers": expected["top_shell_layers"],
        "bottom_shell_layers": expected["bottom_shell_layers"],
        "elefant_foot_compensation": expected[
            "elefant_foot_compensation_mm"],
        "xy_hole_compensation": expected["xy_hole_compensation_mm"],
    }
    for key, expected_value in actual_fields.items():
        value = parsed.config.get(key)
        if value is None:
            errors.append(f"G-code CONFIG_BLOCK lacks {key}")
            continue
        try:
            actual = float(value.strip().rstrip("%"))
        except ValueError:
            errors.append(f"G-code {key} is not numeric: {value!r}")
            continue
        if not math.isclose(actual, expected_value, abs_tol=1.0e-8):
            errors.append(
                f"G-code {key}={actual:g} != resolved {expected_value:g}")

    # Bambu serializes process vectors as comma-separated values.  Comparing
    # only the first process item would let a second speed silently retain an
    # unsafe value (for example outer_wall_speed=60,200).  Filament presets
    # also use vectors for standard/high-flow *variant* slots, but a one-tool
    # P2S G-code CONFIG_BLOCK serializes only the selected first slot.
    vector_fields = {
        "outer_wall_speed": ("process", "outer_wall_speed"),
        "nozzle_temperature": ("filament", "nozzle_temperature"),
        "nozzle_temperature_initial_layer": (
            "filament", "nozzle_temperature_initial_layer"),
        "fan_max_speed": ("filament", "fan_max_speed"),
        "overhang_fan_speed": ("filament", "overhang_fan_speed"),
        "filament_max_volumetric_speed": (
            "filament", "filament_max_volumetric_speed"),
        "textured_plate_temp": ("filament", "textured_plate_temp"),
        "textured_plate_temp_initial_layer": (
            "filament", "textured_plate_temp_initial_layer"),
    }
    for key, (section, profile_key) in vector_fields.items():
        raw_actual = parsed.config.get(key)
        if raw_actual is None:
            errors.append(f"G-code CONFIG_BLOCK lacks {key}")
            continue
        raw_expected = profile_bundle["resolved"][section].get(profile_key)
        expected_items = (
            list(raw_expected) if isinstance(raw_expected, list)
            else [raw_expected])
        actual_items = raw_actual.split(",")
        try:
            expected_vector = [
                float(str(value).strip().rstrip("%"))
                for value in expected_items
            ]
            actual_vector = [
                float(value.strip().rstrip("%"))
                for value in actual_items
            ]
        except (TypeError, ValueError):
            errors.append(
                f"G-code/resolved {key} vector is not numeric: "
                f"{raw_actual!r} / {raw_expected!r}")
            continue
        if len(actual_vector) == len(expected_vector):
            vector_pass = all(math.isclose(
                actual, required, abs_tol=1.0e-8, rel_tol=0.0)
                for actual, required in zip(
                    actual_vector, expected_vector, strict=True))
        elif len(actual_vector) == 1 and section == "filament":
            vector_pass = math.isclose(
                actual_vector[0], expected_vector[0],
                abs_tol=1.0e-8, rel_tol=0.0)
        elif len(actual_vector) == 1:
            # Bambu's CONFIG_BLOCK collapses identical per-tool vectors to one
            # scalar (the stock P2S output does this for 60,60 and 225,225).
            vector_pass = all(math.isclose(
                actual_vector[0], required, abs_tol=1.0e-8, rel_tol=0.0)
                for required in expected_vector)
        elif len(expected_vector) == 1:
            vector_pass = all(math.isclose(
                actual, expected_vector[0], abs_tol=1.0e-8, rel_tol=0.0)
                for actual in actual_vector)
        else:
            vector_pass = False
        if not vector_pass:
            errors.append(
                f"G-code {key}={actual_vector!r} != resolved "
                f"{expected_vector!r}")
    expected_wall_generator = str(expected["wall_generator"]).lower()
    if parsed.config.get("wall_generator", "").lower() != expected_wall_generator:
        errors.append(
            f"G-code wall_generator={parsed.config.get('wall_generator')!r}, "
            f"expected {expected_wall_generator}")
    support_fields = {
        "enable_support": expected["support_enabled"],
        "support_on_build_plate_only": expected[
            "support_on_build_plate_only"],
        "support_critical_regions_only": expected[
            "support_critical_regions_only"],
        "support_remove_small_overhang": expected[
            "support_remove_small_overhang"],
    }
    for key, expected_value in support_fields.items():
        if key not in parsed.config:
            errors.append(f"G-code CONFIG_BLOCK lacks {key}")
            continue
        actual_value = _boolish(parsed.config[key])
        if actual_value is not expected_value:
            errors.append(
                f"G-code {key}={actual_value} != resolved "
                f"{expected_value}")
    expected_pattern = str(expected["sparse_infill_pattern"]).lower()
    actual_pattern = parsed.config.get("sparse_infill_pattern", "").lower()
    if actual_pattern != expected_pattern:
        errors.append(
            f"G-code sparse_infill_pattern={actual_pattern!r} != resolved "
            f"{expected_pattern!r}")
    if parsed.config.get("curr_bed_type") != "Textured PEI Plate":
        errors.append("G-code curr_bed_type is not Textured PEI Plate")
    density = parsed.config.get("sparse_infill_density")
    if density is None:
        errors.append("G-code CONFIG_BLOCK lacks sparse_infill_density")
    else:
        try:
            actual_density = float(density.strip().rstrip("%"))
        except ValueError:
            errors.append(
                f"G-code sparse_infill_density is not numeric: {density!r}")
        else:
            if not math.isclose(
                    actual_density,
                    expected["sparse_infill_density_percent"],
                    abs_tol=1.0e-8):
                errors.append(
                    "G-code sparse_infill_density="
                    f"{actual_density:g}% != resolved "
                    f"{expected['sparse_infill_density_percent']:g}%")
    for key, expected_value in (
            ("precise_outer_wall", True),
            ("detect_thin_wall", True),
            ("detect_narrow_internal_solid_infill", True)):
        if _boolish(parsed.config.get(key)) is not expected_value:
            errors.append(f"G-code {key} is not enabled")
    if parsed.config.get("ensure_vertical_shell_thickness") != "enabled":
        errors.append(
            "G-code ensure_vertical_shell_thickness is not enabled")
    if parsed.config.get("machine_pause_gcode", "").strip() != "M400 U1":
        errors.append("G-code machine_pause_gcode is not exactly 'M400 U1'")
    actual_arc_fitting = _boolish(parsed.config.get("enable_arc_fitting"))
    if actual_arc_fitting != bool(expected.get("arc_fitting_enabled")):
        errors.append(
            f"G-code enable_arc_fitting={actual_arc_fitting} != resolved "
            f"{bool(expected.get('arc_fitting_enabled'))}")
    first_height = expected["first_layer_height_mm"]
    if not math.isclose(parsed.layers[0].z, first_height, abs_tol=0.001):
        errors.append(
            f"first actual layer Z={parsed.layers[0].z:.3f}, "
            f"expected {first_height:.3f}")
    for layer in parsed.layers[1:]:
        if (layer.layer_height is not None
                and not math.isclose(
                    layer.layer_height, expected["layer_height_mm"],
                    abs_tol=0.001)):
            errors.append(
                f"layer at Z={layer.z:.3f} has height "
                f"{layer.layer_height:.3f}, expected "
                f"{expected['layer_height_mm']:.3f}")
            break
    return errors


def _layer_at_or_below(layers: Sequence[Layer], value: float) -> Layer:
    matches = [layer for layer in layers if layer.z <= value + LAYER_EPS]
    if not matches:
        raise AuditError(f"no sliced layer at or below {value:.3f} mm")
    return matches[-1]


def _layer_above(layers: Sequence[Layer], value: float) -> Layer:
    for layer in layers:
        if layer.z > value + LAYER_EPS:
            return layer
    raise AuditError(f"no sliced layer above {value:.3f} mm")


def _layer_at_or_above(layers: Sequence[Layer], value: float) -> Layer:
    for layer in layers:
        if layer.z >= value - LAYER_EPS:
            return layer
    raise AuditError(f"no sliced layer at or above {value:.3f} mm")


def _cavity_retain_regions(
    sites: Sequence[Mapping[str, Any]],
    placement_xy: tuple[float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    """Bound retained G-code segments to the evidence ROI of every site."""
    regions = []
    for site in sites:
        cx, cy, _ = site["print_cavity_center_xyz_mm"]
        cx += placement_xy[0]
        cy += placement_xy[1]
        half = site["cavity_diameter_mm"] / 2.0 + EVIDENCE_MARGIN_MM
        regions.append((cx - half, cy - half, cx + half, cy + half))
    return tuple(regions)


def _seated_magnet_print_z_bounds(
    site: Mapping[str, Any],
) -> tuple[float, float]:
    """Exact vertical bounds of a fully seated cylindrical magnet.

    A cylinder's support extent along print Z combines its axial half-depth
    and the projection of its circular radius perpendicular to that axis.
    This works for both transverse coupon-style discs and axial sites.
    """
    center_z = _vec3(
        site["print_seated_magnet_center_xyz_mm"],
        "seated magnet center")[2]
    axis = _unit3(site["print_marked_pole_axis_xyz"], "magnet axis")
    axis_z = max(-1.0, min(1.0, axis[2]))
    axial_extent = abs(axis_z) * site["magnet_depth_mm"] / 2.0
    radial_extent = (
        math.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        * site["magnet_diameter_mm"] / 2.0
    )
    extent = axial_extent + radial_extent
    return center_z - extent, center_z + extent


def _retaining_stage_pass(
    site: Mapping[str, Any], stage: str, metrics: Mapping[str, Any],
) -> bool:
    """Require real retaining extrusion throughout every open stage.

    At the very first circular-cradle layer the theoretical chord begins at
    zero width, so demanding the representative layer's 3-mm span there
    would reject valid circular bottoms.  It must nevertheless contain both
    axial skin paths.  Representative and last-open layers retain the full
    physically validated threshold.
    """
    retaining = metrics["retaining_paths"]
    if (stage == "lowest_open"
            and site["closure_kind"] == "transverse_gable_45deg"):
        # At fat-nozzle line widths the emerging sliver's surroundings may
        # resolve as diagonal solid fill crossing the skin bands rather than
        # as along-wall runs (observed on the 0.6-mm lane's thin wing UM
        # band: 7.7 mm of in-band extrusion with a 0.06-mm along-wall span).
        # Dense crossing coverage retains the magnet at least as strongly as
        # one discrete wall, so accept either form; an actually open band
        # still fails both terms.
        def _band_ok(prefix: str) -> bool:
            length = retaining[f"{prefix}_skin_path_length_mm"]
            contiguous = retaining[
                f"{prefix}_skin_longest_contiguous_span_mm"]
            return length >= 0.20 and (contiguous >= 0.10 or length >= 2.0)

        return _band_ok("interface") and _band_ok("inner")
    if stage == "lowest_open":
        # At the first axial cradle layer the circular floor intersects the
        # cavity as top/gap paths, not yet as the mature medial annulus seen at
        # representative and last-open layers.  Require real continuous ring
        # material here, while reserving strict bounded-bead multiplicity for
        # the two mature open stages.
        gap = retaining.get("largest_uncovered_arc_mm")
        return (
            retaining.get("annular_path_length_mm", 0.0)
            >= math.pi * site["cavity_diameter_mm"] / 2.0
            and isinstance(gap, (int, float))
            and math.isfinite(float(gap))
            and float(gap) <= RETAINING_PATH_CONNECTIVITY_GAP_MM)
    return bool(retaining["pass"])


def _sample_segment(segment: Segment, step: float = SITE_SAMPLE_STEP_MM) -> Iterator[tuple[float, float, float]]:
    length = segment.length
    count = max(1, int(math.ceil(length / step)))
    weight = length / count
    for index in range(count):
        t = (index + 0.5) / count
        yield (
            segment.x0 + t * (segment.x1 - segment.x0),
            segment.y0 + t * (segment.y1 - segment.y0),
            weight,
        )


def _longest_connected_v_span(
    points: Sequence[tuple[float, float]],
) -> float:
    """Return the longest V span of one connected extrusion component.

    Connectivity is measured in the local wall plane, not only after
    projecting onto V.  Thus two fragments at different U positions cannot
    masquerade as one wall merely because their V ranges overlap.  A small
    spatial hash keeps this linear for normal sampled toolpaths.
    """
    if not points:
        return 0.0
    gap = RETAINING_PATH_CONNECTIVITY_GAP_MM
    gap2 = gap * gap
    parents = list(range(len(points)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        a_root, b_root = find(a), find(b)
        if a_root != b_root:
            parents[b_root] = a_root

    cells: dict[tuple[int, int], list[int]] = {}
    for index, (u, v) in enumerate(points):
        cell = (math.floor(u / gap), math.floor(v / gap))
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                for other in cells.get((cell[0] + du, cell[1] + dv), ()):
                    ou, ov = points[other]
                    if (u - ou) ** 2 + (v - ov) ** 2 <= gap2:
                        union(index, other)
        cells.setdefault(cell, []).append(index)

    extents: dict[int, list[float]] = {}
    for index, (_u, v) in enumerate(points):
        bounds = extents.setdefault(find(index), [v, v])
        bounds[0] = min(bounds[0], v)
        bounds[1] = max(bounds[1], v)
    return max(high - low for low, high in extents.values())


def _single_classic_track_summary(
    *,
    track_samples: Sequence[tuple[float, int, str, float, int]],
    required_bins: Sequence[int],
    expected_center_mm: float,
    allowed_width_range_mm: tuple[float, float],
    lower_width_tolerance_mm: float,
) -> dict[str, Any]:
    """Count extrusion tracks independently in every occupied scan bin.

    Path identity comes from extrusion/travel continuity in the raw G-code,
    not spatial clustering.  Thus coincident duplicate traversals and the two
    real contours observed around U=0.210 and U=0.240 mm remain two paths.
    """
    target = RELEASE_SITE_GEOMETRY_MM["minimum_retaining_path_mm"]
    by_bin: dict[int, list[tuple[float, str, float, int]]] = {}
    for position, bin_index, feature, width, path_id in track_samples:
        by_bin.setdefault(bin_index, []).append(
            (position, feature, width, path_id))
    centers_by_bin: dict[int, list[float]] = {}
    for bin_index, samples in by_bin.items():
        by_path: dict[int, list[tuple[float, str, float, int]]] = {}
        for sample in samples:
            by_path.setdefault(sample[3], []).append(sample)
        centers = []
        for _path_id, path_samples in sorted(by_path.items()):
            clusters: list[list[tuple[float, str, float, int]]] = []
            for sample in sorted(path_samples, key=lambda value: value[0]):
                if (not clusters
                        or sample[0] - clusters[-1][0][0]
                        > RETAINING_TRACK_DEDUP_TOLERANCE_MM):
                    clusters.append([sample])
                else:
                    clusters[-1].append(sample)
            centers.extend(
                sum(sample[0] for sample in cluster) / len(cluster)
                for cluster in clusters)
        centers_by_bin[bin_index] = sorted(centers)
    unique_features = sorted({
        str(sample[2]).strip() for sample in track_samples
    })
    unique_widths = sorted({
        round(float(sample[3]), 6) for sample in track_samples
    })
    effective_lower_width_mm = (
        allowed_width_range_mm[0]
        - lower_width_tolerance_mm)
    width_pass = bool(unique_widths) and all(
        effective_lower_width_mm - FLOAT_EPS <= value
        <= allowed_width_range_mm[1] + FLOAT_EPS
        for value in unique_widths)
    feature_pass = bool(unique_features) and all(
        value.lower() == "outer wall" for value in unique_features)
    missing_bins = sorted(set(required_bins) - set(centers_by_bin))
    per_bin_counts = {
        str(index): len(centers_by_bin.get(index, ()))
        for index in required_bins
    }
    max_count = max(per_bin_counts.values(), default=0)
    exactly_one_per_bin = (
        not missing_bins
        and all(value == 1 for value in per_bin_counts.values()))
    centered_pass = exactly_one_per_bin and all(
        abs(centers_by_bin[index][0] - expected_center_mm) <= 0.12
        for index in required_bins)
    single_pass = (
        exactly_one_per_bin
        and centered_pass
        and width_pass
        and feature_pass
    )
    return {
        "classic_target_line_width_mm": target,
        "allowed_single_bead_width_range_mm": list(
            allowed_width_range_mm),
        "lower_width_tolerance_mm": lower_width_tolerance_mm,
        "effective_minimum_bead_width_mm": effective_lower_width_mm,
        "observed_line_widths_mm": unique_widths,
        "observed_features": unique_features,
        "track_identity": "raw_gcode_extrusion_travel_continuity",
        "same_path_coordinate_dedup_tolerance_mm": (
            RETAINING_TRACK_DEDUP_TOLERANCE_MM),
        "expected_center_mm": expected_center_mm,
        "required_scan_bins": list(required_bins),
        "missing_scan_bins": missing_bins,
        "observed_track_centers_mm_by_bin": {
            str(index): centers for index, centers
            in sorted(centers_by_bin.items())
        },
        "path_count_by_scan_bin": per_bin_counts,
        "estimated_path_count": max_count,
        "exactly_one_path_per_scan_bin": exactly_one_per_bin,
        "centered_path_pass": centered_pass,
        "line_width_pass": width_pass,
        "outer_wall_only_pass": feature_pass,
        "single_classic_path_pass": single_pass,
    }


def _clustered_track_crossings(
    track_samples: Sequence[tuple[float, int, str, float, int]],
) -> dict[int, list[tuple[float, int, float, tuple[str, ...]]]]:
    """Cluster coincident segment intersections without losing path identity."""
    by_bin: dict[int, list[tuple[float, str, float, int]]] = {}
    for position, bin_index, feature, width, path_id in track_samples:
        by_bin.setdefault(bin_index, []).append(
            (position, feature, width, path_id))
    result: dict[int, list[tuple[float, int, float, tuple[str, ...]]]] = {}
    for bin_index, samples in by_bin.items():
        by_path: dict[int, list[tuple[float, str, float, int]]] = {}
        for sample in samples:
            by_path.setdefault(sample[3], []).append(sample)
        crossings: list[tuple[float, int, float, tuple[str, ...]]] = []
        for path_id, path_samples in sorted(by_path.items()):
            clusters: list[list[tuple[float, str, float, int]]] = []
            for sample in sorted(path_samples, key=lambda value: value[0]):
                if (not clusters
                        or sample[0] - clusters[-1][0][0]
                        > RETAINING_TRACK_DEDUP_TOLERANCE_MM):
                    clusters.append([sample])
                else:
                    clusters[-1].append(sample)
            crossings.extend((
                sum(sample[0] for sample in cluster) / len(cluster),
                path_id,
                max(float(sample[2]) for sample in cluster),
                tuple(sorted({str(sample[1]).strip() for sample in cluster})),
            ) for cluster in clusters)
        result[bin_index] = sorted(crossings)
    return result


def _single_transverse_classic_track_summary(
    *,
    track_samples: Sequence[tuple[float, int, str, float, int]],
    nearby_track_samples: Sequence[tuple[float, int, str, float, int]],
    required_bins: Sequence[int],
    expected_center_mm: float,
    allowed_width_range_mm: tuple[float, float],
    material_side_sign: int,
) -> dict[str, Any]:
    """Prove one cavity wall and reject nearby overlapping traversals.

    The primary candidate is selected by its physical cavity-facing bead edge.
    A second bead cannot evade the exact-one rule merely by falling just
    outside that classifier.  Every nearby longitudinal crossing whose bead
    footprint overlaps the primary is therefore audited too.  The sole
    permitted topology is Bambu's measured same-path surrounding-body hairpin:
    one Outer-wall return in at most three contiguous bins anchored to one edge
    of the scan window, on the material side of the cavity wall.
    """
    if material_side_sign not in (-1, 1):
        raise AuditError("transverse material-side sign must be -1 or +1")
    summary = _single_classic_track_summary(
        track_samples=track_samples,
        required_bins=required_bins,
        expected_center_mm=expected_center_mm,
        allowed_width_range_mm=allowed_width_range_mm,
        lower_width_tolerance_mm=RETAINING_BEAD_ACCEPTANCE[
            "transverse_lower_width_tolerance_mm"])
    primary = _clustered_track_crossings(track_samples)
    nearby = _clustered_track_crossings(nearby_track_samples)

    overlapping_extras: dict[
        int, list[tuple[float, int, float, tuple[str, ...]]]] = {}
    for bin_index in required_bins:
        primary_crossings = primary.get(bin_index, ())
        for crossing in nearby.get(bin_index, ()):
            matched_primary = any(
                crossing[1] == candidate[1]
                and abs(crossing[0] - candidate[0])
                <= RETAINING_TRACK_DEDUP_TOLERANCE_MM + FLOAT_EPS
                for candidate in primary_crossings)
            if matched_primary:
                continue
            bead_overlaps_primary = any(
                abs(crossing[0] - candidate[0])
                <= (crossing[2] + candidate[2]) / 2.0 + FLOAT_EPS
                for candidate in primary_crossings)
            if bead_overlaps_primary:
                overlapping_extras.setdefault(bin_index, []).append(crossing)

    extra_bins = sorted(overlapping_extras)
    one_extra_per_bin = all(
        len(overlapping_extras[index]) == 1 for index in extra_bins)
    edge_anchored = not extra_bins
    if extra_bins:
        contiguous = extra_bins == list(range(extra_bins[0], extra_bins[-1] + 1))
        edge_anchored = contiguous and (
            extra_bins[0] == required_bins[0]
            or extra_bins[-1] == required_bins[-1])
    bounded = (
        len(extra_bins) <= TRANSVERSE_SAME_PATH_EDGE_RETURN_BIN_LIMIT
        and one_extra_per_bin
        and edge_anchored)
    same_path = all(
        len(primary.get(index, ())) == 1
        and overlapping_extras[index][0][1] == primary[index][0][1]
        for index in extra_bins)
    material_side = all(
        material_side_sign * (
            overlapping_extras[index][0][0] - primary[index][0][0]) > 0.0
        for index in extra_bins)
    outer_wall_only = all(
        all(feature.lower() == "outer wall" for feature in crossing[3])
        for crossings in overlapping_extras.values() for crossing in crossings)
    nearby_duplicate_guard_pass = (
        bounded and same_path and material_side and outer_wall_only)
    single_pass = (
        summary["single_classic_path_pass"]
        and nearby_duplicate_guard_pass)
    max_nearby_overlap_count = max((
        len(primary.get(index, ()))
        + len(overlapping_extras.get(index, ()))
        for index in required_bins), default=0)
    summary.update({
        "track_identity": (
            "cavity_edge_primary_plus_nearby_overlap_duplicate_guard"),
        "nearby_overlapping_extra_scan_bins": extra_bins,
        "nearby_overlapping_extra_centers_mm_by_bin": {
            str(index): [crossing[0] for crossing in crossings]
            for index, crossings in sorted(overlapping_extras.items())
        },
        "nearby_overlapping_extra_path_ids_by_bin": {
            str(index): [crossing[1] for crossing in crossings]
            for index, crossings in sorted(overlapping_extras.items())
        },
        "nearby_overlapping_maximum_crossings_per_scan_bin": (
            max_nearby_overlap_count),
        "allowed_same_path_edge_return_scan_bins": (
            TRANSVERSE_SAME_PATH_EDGE_RETURN_BIN_LIMIT),
        "nearby_one_extra_per_bin_pass": one_extra_per_bin,
        "nearby_edge_return_contiguous_and_anchored_pass": edge_anchored,
        "nearby_edge_return_same_raw_path_pass": same_path,
        "nearby_edge_return_material_side_pass": material_side,
        "nearby_edge_return_outer_wall_only_pass": outer_wall_only,
        "nearby_duplicate_guard_pass": nearby_duplicate_guard_pass,
        "single_classic_path_pass": single_pass,
    })
    return summary


def _transverse_track_intersections(
    segments: Sequence[tuple[float, float, float, float, str, float, int]],
    *,
    scanlines_v_mm: Sequence[float],
) -> list[tuple[float, int, str, float, int]]:
    """Intersect every nearby longitudinal bead with fixed scan lines."""
    intersections = []
    for bin_index, scan_v in enumerate(scanlines_v_mm):
        for u0, v0, u1, v1, feature, width, path_id in segments:
            delta_v = v1 - v0
            if abs(delta_v) <= FLOAT_EPS:
                continue
            t = (scan_v - v0) / delta_v
            if t < -FLOAT_EPS or t > 1.0 + FLOAT_EPS:
                continue
            u = u0 + max(0.0, min(1.0, t)) * (u1 - u0)
            intersections.append((u, bin_index, feature, width, path_id))
    return intersections


def _cavity_boundary_track_intersections(
    track_samples: Sequence[tuple[float, int, str, float, int]],
    *,
    expected_cavity_edge_mm: float,
    cavity_edge_direction: int,
) -> list[tuple[float, int, str, float, int]]:
    """Select beads whose physical edge forms the nominal cavity boundary."""
    if cavity_edge_direction not in (-1, 1):
        raise AuditError("transverse cavity-edge direction must be -1 or +1")
    intersections = []
    for u, bin_index, feature, width, path_id in track_samples:
        cavity_edge = u + cavity_edge_direction * width / 2.0
        if (abs(cavity_edge - expected_cavity_edge_mm)
                <= TRANSVERSE_CAVITY_EDGE_TOLERANCE_MM + FLOAT_EPS):
            intersections.append((u, bin_index, feature, width, path_id))
    return intersections


def _annular_track_intersections(
    segments: Sequence[tuple[float, float, float, float, str, float, int]],
    *,
    center_xy: tuple[float, float],
    expected_radius_mm: float,
    skin_thickness_mm: float,
    ray_count: int = 72,
) -> list[tuple[float, int, str, float, int]]:
    """Intersect actual toolpath segments with fixed rays, not angle bins."""
    cx, cy = center_xy
    intersections = []
    for bin_index in range(ray_count):
        angle = (bin_index + 0.5) * 2.0 * math.pi / ray_count
        dx, dy = math.cos(angle), math.sin(angle)
        for x0, y0, x1, y1, feature, width, path_id in segments:
            px, py = x0 - cx, y0 - cy
            sx, sy = x1 - x0, y1 - y0
            denominator = sx * dy - sy * dx
            if abs(denominator) <= FLOAT_EPS:
                continue
            t = -(px * dy - py * dx) / denominator
            if t < -FLOAT_EPS or t > 1.0 + FLOAT_EPS:
                continue
            ix = px + max(0.0, min(1.0, t)) * sx
            iy = py + max(0.0, min(1.0, t)) * sy
            radial = ix * dx + iy * dy
            if radial <= 0.0:
                continue
            if (abs(radial - expected_radius_mm)
                    <= skin_thickness_mm / 2.0 + 0.04):
                intersections.append(
                    (radial, bin_index, feature, width, path_id))
    return intersections


def _single_annular_classic_track_summary(
    *,
    track_samples: Sequence[tuple[float, int, str, float, int]],
    required_bins: Sequence[int],
    expected_center_mm: float,
    allowed_width_range_mm: tuple[float, float],
    allowed_anomaly_bins: int = 2,
    allowed_component_paths: int = 2,
    center_tolerance_mm: float = 0.16,
    component_points: Mapping[
        int, Sequence[tuple[float, float, float]]] | None = None,
) -> dict[str, Any]:
    """Prove one annular bead, optionally split into two complementary arcs."""
    summary = _single_classic_track_summary(
        track_samples=track_samples,
        required_bins=required_bins,
        expected_center_mm=expected_center_mm,
        allowed_width_range_mm=allowed_width_range_mm,
        lower_width_tolerance_mm=RETAINING_BEAD_ACCEPTANCE[
            "axial_lower_width_tolerance_mm"])
    expected_bins = tuple(range(len(required_bins)))
    if tuple(required_bins) != expected_bins:
        raise AuditError(
            "annular scan bins must be the contiguous zero-based ray ring")
    ray_count = len(required_bins)
    if ray_count < 3:
        raise AuditError("annular scan requires at least three rays")
    samples_by_bin: dict[
        int, list[tuple[float, int, str, float, int]]] = {}
    for sample in track_samples:
        samples_by_bin.setdefault(sample[1], []).append(sample)
    unique_path_ids = sorted({sample[4] for sample in track_samples})
    component_bins = {
        path_id: {
            sample[1] for sample in track_samples if sample[4] == path_id
        }.intersection(required_bins)
        for path_id in unique_path_ids
    }

    def cyclic_interval(
        bins: set[int],
    ) -> tuple[bool, set[int]]:
        """Return whether bins form one cyclic run and its occupied endpoints."""
        if not bins:
            return False, set()
        if len(bins) == ray_count:
            return True, set()
        starts = {
            index for index in bins
            if (index - 1) % ray_count not in bins
        }
        ends = {
            index for index in bins
            if (index + 1) % ray_count not in bins
        }
        return len(starts) == 1 and len(ends) == 1, starts | ends

    component_intervals = {
        path_id: cyclic_interval(bins)
        for path_id, bins in component_bins.items()
    }
    occupied_bins = sorted(set(samples_by_bin).intersection(required_bins))
    missing_bins = sorted(set(required_bins) - set(occupied_bins))
    per_bin_counts = {
        str(bin_index): len(summary[
            "observed_track_centers_mm_by_bin"].get(str(bin_index), ()))
        for bin_index in required_bins
    }
    multiple_crossing_bins = sorted(
        bin_index for bin_index in required_bins
        if per_bin_counts[str(bin_index)] > 1)
    anomaly_bins = sorted(set(missing_bins) | set(multiple_crossing_bins))
    bounded_anomaly_count_pass = len(anomaly_bins) <= allowed_anomaly_bins
    coverage_pass = len(occupied_bins) >= ray_count - allowed_anomaly_bins
    bounded_component_path_count_pass = (
        1 <= len(unique_path_ids) <= allowed_component_paths)
    component_cyclic_contiguous_pass = (
        bounded_component_path_count_pass
        and all(value[0] for value in component_intervals.values()))

    cross_component_overlap_bins: set[int] = set()
    if len(unique_path_ids) == 2:
        cross_component_overlap_bins = (
            component_bins[unique_path_ids[0]]
            & component_bins[unique_path_ids[1]])
    component_exclusive_bins = {
        path_id: bins - set().union(*(
            other_bins for other_id, other_bins in component_bins.items()
            if other_id != path_id
        )) if len(component_bins) > 1 else set(bins)
        for path_id, bins in component_bins.items()
    }
    if len(unique_path_ids) == 1:
        complementary_coverage_pass = coverage_pass
    elif len(unique_path_ids) == 2:
        complementary_coverage_pass = (
            coverage_pass
            and not cross_component_overlap_bins
            and all(
                len(component_exclusive_bins[path_id]) >= 18
                for path_id in unique_path_ids))
    else:
        complementary_coverage_pass = False

    endpoint_bins = set().union(*(
        endpoints for _contiguous, endpoints in component_intervals.values()
    )) if component_intervals else set()
    if (len(unique_path_ids) == 1 and not endpoint_bins
            and multiple_crossing_bins):
        # A full single component has no occupancy gap from which to infer its
        # raw seam.  A bounded local double crossing is itself the seam datum.
        endpoint_bins.update(multiple_crossing_bins)

    def cyclic_distance(a: int, b: int) -> int:
        delta = abs(a - b)
        return min(delta, ray_count - delta)

    anomaly_endpoint_local_pass = (
        not anomaly_bins
        or (bool(endpoint_bins) and all(
            any(cyclic_distance(index, endpoint) <= 1
                for endpoint in endpoint_bins)
            for index in anomaly_bins)))
    if len(unique_path_ids) == 1 and anomaly_bins:
        anomalies_one_run, _unused = cyclic_interval(set(anomaly_bins))
        anomaly_endpoint_local_pass = (
            anomaly_endpoint_local_pass and anomalies_one_run)

    component_junctions: list[dict[str, Any]] = []
    if (len(unique_path_ids) == 2
            and not cross_component_overlap_bins):
        owner_by_bin = {
            index: next((
                path_id for path_id in unique_path_ids
                if index in component_bins[path_id]
            ), None)
            for index in required_bins
        }
        seen_junctions: set[tuple[int, int, int, int]] = set()
        for from_bin in required_bins:
            from_path = owner_by_bin[from_bin]
            if from_path is None:
                continue
            to_bin = (from_bin + 1) % ray_count
            steps = 1
            while owner_by_bin[to_bin] is None and steps < ray_count:
                to_bin = (to_bin + 1) % ray_count
                steps += 1
            to_path = owner_by_bin[to_bin]
            if to_path is None or to_path == from_path:
                continue
            key = (from_bin, to_bin, int(from_path), int(to_path))
            if key in seen_junctions:
                continue
            seen_junctions.add(key)
            candidates = []
            angle_a = (from_bin + 0.5) * 2.0 * math.pi / ray_count
            angle_b = (to_bin + 0.5) * 2.0 * math.pi / ray_count

            def angle_distance(a: float, b: float) -> float:
                delta = abs((a - b) % (2.0 * math.pi))
                return min(delta, 2.0 * math.pi - delta)

            search_angle = (
                ANNULAR_COMPONENT_SEAM_SEARCH_RAYS
                * 2.0 * math.pi / ray_count)
            geometry_a = [
                point for point in (component_points or {}).get(
                    int(from_path), ())
                if angle_distance(math.atan2(point[1], point[0]), angle_a)
                <= search_angle + FLOAT_EPS
            ]
            geometry_b = [
                point for point in (component_points or {}).get(
                    int(to_path), ())
                if angle_distance(math.atan2(point[1], point[0]), angle_b)
                <= search_angle + FLOAT_EPS
            ]
            measurement_basis = "annular_component_near_endpoint_samples"
            for point_a in geometry_a:
                for point_b in geometry_b:
                    distance = math.hypot(
                        point_b[0] - point_a[0],
                        point_b[1] - point_a[1])
                    width_limit = (point_a[2] + point_b[2]) / 2.0
                    limit = min(
                        RETAINING_PATH_CONNECTIVITY_GAP_MM,
                        width_limit + ANNULAR_COMPONENT_SEAM_WIDTH_MARGIN_MM)
                    candidates.append((distance, limit))
            if not candidates:
                measurement_basis = "ray_intersection_fallback"
                from_samples = [
                    sample for sample in track_samples
                    if sample[1] == from_bin and sample[4] == from_path
                ]
                to_samples = [
                    sample for sample in track_samples
                    if sample[1] == to_bin and sample[4] == to_path
                ]
                for sample_a in from_samples:
                    for sample_b in to_samples:
                        radial_a, radial_b = sample_a[0], sample_b[0]
                        distance = math.sqrt(max(
                            0.0,
                            radial_a * radial_a + radial_b * radial_b
                            - 2.0 * radial_a * radial_b
                            * math.cos(angle_b - angle_a)))
                        width_limit = (
                            float(sample_a[3]) + float(sample_b[3])) / 2.0
                        limit = min(
                            RETAINING_PATH_CONNECTIVITY_GAP_MM,
                            width_limit
                            + ANNULAR_COMPONENT_SEAM_WIDTH_MARGIN_MM)
                        candidates.append((distance, limit))
            if not candidates:
                distance, limit = math.inf, 0.0
            else:
                distance, limit = min(
                    candidates, key=lambda value: value[0])
            component_junctions.append({
                "from_path_id": from_path,
                "from_ray_bin": from_bin,
                "to_path_id": to_path,
                "to_ray_bin": to_bin,
                "intervening_ray_step_count": steps,
                "measurement_basis": measurement_basis,
                "nearest_centerline_distance_mm": distance,
                "allowed_centerline_distance_mm": limit,
                "pass": distance <= limit + FLOAT_EPS,
            })
    component_seam_continuity_pass = (
        len(unique_path_ids) == 1
        or (len(component_junctions) == 2
            and all(record["pass"] for record in component_junctions)))

    centered_pass = bool(track_samples) and all(
        abs(sample[0] - expected_center_mm) <= center_tolerance_mm
        for sample in track_samples)
    single_pass = (
        bounded_component_path_count_pass
        and component_cyclic_contiguous_pass
        and complementary_coverage_pass
        and component_seam_continuity_pass
        and bounded_anomaly_count_pass
        and anomaly_endpoint_local_pass
        and centered_pass
        and summary["line_width_pass"]
        and summary["outer_wall_only_pass"])
    summary.update({
        "track_identity": (
            "bounded_complementary_raw_gcode_annular_path_components"),
        "observed_path_ids": unique_path_ids,
        "estimated_path_count": len(unique_path_ids),
        "allowed_component_path_count": allowed_component_paths,
        "bounded_component_path_count_pass": (
            bounded_component_path_count_pass),
        "component_occupied_ray_bins": {
            str(path_id): sorted(component_bins[path_id])
            for path_id in unique_path_ids
        },
        "component_exclusive_ray_bin_count": {
            str(path_id): len(component_exclusive_bins[path_id])
            for path_id in unique_path_ids
        },
        "minimum_exclusive_ray_bins_per_split_component": 18,
        "cross_component_overlap_ray_bins": sorted(
            cross_component_overlap_bins),
        "component_cyclic_contiguous_pass": (
            component_cyclic_contiguous_pass),
        "complementary_component_coverage_pass": (
            complementary_coverage_pass),
        "component_seam_width_margin_mm": (
            ANNULAR_COMPONENT_SEAM_WIDTH_MARGIN_MM),
        "component_seam_search_ray_radius": (
            ANNULAR_COMPONENT_SEAM_SEARCH_RAYS),
        "component_seam_junctions": component_junctions,
        "component_seam_continuity_pass": (
            component_seam_continuity_pass),
        "required_minimum_occupied_ray_bins": (
            len(required_bins) - allowed_anomaly_bins),
        "occupied_ray_bin_count": len(occupied_bins),
        "missing_scan_bins": missing_bins,
        "annular_coverage_pass": coverage_pass,
        "allowed_local_seam_anomaly_bins": allowed_anomaly_bins,
        "combined_missing_or_multiple_anomaly_bins": anomaly_bins,
        "bounded_combined_anomaly_count_pass": bounded_anomaly_count_pass,
        "component_interval_endpoint_ray_bins": sorted(endpoint_bins),
        "anomaly_endpoint_local_pass": anomaly_endpoint_local_pass,
        "multiple_crossing_ray_bins": multiple_crossing_bins,
        "multiple_crossing_ray_bin_path_ids": {
            str(bin_index): sorted({
                sample[4] for sample in samples_by_bin[bin_index]})
            for bin_index in multiple_crossing_bins
        },
        "local_seam_anomaly_pass": (
            bounded_anomaly_count_pass and anomaly_endpoint_local_pass),
        "unique_annular_path_pass": len(unique_path_ids) == 1,
        "center_tolerance_mm": center_tolerance_mm,
        "centered_path_pass": centered_pass,
        "single_classic_path_pass": single_pass,
    })
    return summary


def _largest_circular_sample_gap(
    angles: Sequence[float], radius: float,
) -> float:
    """Return the largest uncovered arc between annular path samples."""
    if len(angles) < 2 or radius <= 0.0:
        return math.inf
    ordered = sorted(float(angle) % (2.0 * math.pi) for angle in angles)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + 2.0 * math.pi - ordered[-1])
    return max(gaps) * radius


def _toolpath_metrics(
    layer: Layer,
    site: Mapping[str, Any],
    placement_xy: tuple[float, float],
) -> dict[str, Any]:
    cx, cy, _ = site["print_cavity_center_xyz_mm"]
    cx += placement_xy[0]
    cy += placement_xy[1]
    radius = site["cavity_diameter_mm"] / 2.0
    magnet_radius = site.get("magnet_diameter_mm", 5.0) / 2.0
    interior = 0.0
    boundary_distances: list[float] = []
    local_segments: list[Segment] = []
    wall_a = wall_b = 0.0
    wall_a_v: list[float] = []
    wall_b_v: list[float] = []
    wall_a_uv: list[tuple[float, float]] = []
    wall_b_uv: list[tuple[float, float]] = []
    wall_a_track_segments: list[
        tuple[float, float, float, float, str, float, int]] = []
    wall_b_track_segments: list[
        tuple[float, float, float, float, str, float, int]] = []
    roi = radius + EVIDENCE_MARGIN_MM
    closure = site["closure_kind"]
    if closure == "transverse_gable_45deg":
        fx, fy, _ = site["print_actual_face_xyz_mm"]
        fx += placement_xy[0]
        fy += placement_xy[1]
        ux, uy = _unit_xy(site["print_material_inward_xyz"], "material inward")
        vx, vy = -uy, ux
        face_skin = site["face_skin_mm"]
        cavity_depth = site["cavity_depth_mm"]
        inner_skin = site["inner_skin_mm"]
        first_wall_center = face_skin / 2.0
        second_wall_center = face_skin + cavity_depth + inner_skin / 2.0
        wall_band = max(0.28, min(face_skin * 0.75, 0.38))
        central_chord_half = min(1.0, magnet_radius / 2.0)
        track_bin_count = max(1, int(math.ceil(
            2.0 * central_chord_half / SITE_SAMPLE_STEP_MM)))
        positive_free_edges: list[float] = []
        negative_free_edges: list[float] = []
        interface_cavity_edges: list[float] = []
        inner_cavity_edges: list[float] = []
        for segment in layer.segments:
            if min(segment.x0, segment.x1) > cx + roi or max(segment.x0, segment.x1) < cx - roi:
                continue
            if min(segment.y0, segment.y1) > cy + roi or max(segment.y0, segment.y1) < cy - roi:
                continue
            local_segments.append(segment)
            path_width = (
                segment.line_width
                if segment.line_width is not None and segment.line_width > 0.0
                else FALLBACK_LINE_WIDTH_MM)
            half_path = path_width / 2.0
            dx0, dy0 = segment.x0 - fx, segment.y0 - fy
            dx1, dy1 = segment.x1 - fx, segment.y1 - fy
            u0, v0 = dx0 * ux + dy0 * uy, (
                segment.x0 - cx) * vx + (segment.y0 - cy) * vy
            u1, v1 = dx1 * ux + dy1 * uy, (
                segment.x1 - cx) * vx + (segment.y1 - cy) * vy
            longitudinal = abs(v1 - v0) >= abs(u1 - u0)
            for x, y, weight in _sample_segment(segment):
                dx, dy = x - fx, y - fy
                u = dx * ux + dy * uy
                v = (x - cx) * vx + (y - cy) * vy
                if (face_skin + 0.05 < u < face_skin + cavity_depth - 0.05
                        and abs(v) < radius - 0.05):
                    interior += weight
                if (face_skin + 0.12 <= u <= face_skin + cavity_depth - 0.12
                        and abs(v) <= radius + 1.0):
                    boundary_distances.append(abs(v))
                    if v >= 0.0:
                        positive_free_edges.append(v - half_path)
                    else:
                        negative_free_edges.append(v + half_path)
                if abs(u - first_wall_center) <= wall_band and abs(v) <= radius + 0.35:
                    wall_a += weight
                    wall_a_v.append(v)
                    wall_a_uv.append((u, v))
                    if abs(v) <= central_chord_half:
                        interface_cavity_edges.append(u + half_path)
                if abs(u - second_wall_center) <= wall_band and abs(v) <= radius + 0.35:
                    wall_b += weight
                    wall_b_v.append(v)
                    wall_b_uv.append((u, v))
                    if abs(v) <= central_chord_half:
                        inner_cavity_edges.append(u - half_path)
            # Keep the duplicate guard scoped to the designed 0.45-mm skin
            # centre band (plus 0.04 mm numeric/toolpath margin).  Structural
            # perimeter legs deeper in the surrounding body are not extra
            # retaining-wall traversals, even when normal FDM bead footprints
            # touch.  A true near-duplicate just outside the 0.06-mm cavity-
            # edge classifier remains inside this broader skin band and is
            # audited below.
            if (longitudinal and min(u0, u1) <= (
                    first_wall_center + face_skin / 2.0 + 0.04)
                    and max(u0, u1) >= (
                        first_wall_center - face_skin / 2.0 - 0.04)):
                wall_a_track_segments.append((
                    u0, v0, u1, v1, segment.feature, path_width,
                    segment.path_id))
            if (longitudinal and min(u0, u1) <= (
                    second_wall_center + inner_skin / 2.0 + 0.04)
                    and max(u0, u1) >= (
                        second_wall_center - inner_skin / 2.0 - 0.04)):
                wall_b_track_segments.append((
                    u0, v0, u1, v1, segment.feature, path_width,
                    segment.path_id))
        span_a = max(wall_a_v) - min(wall_a_v) if wall_a_v else 0.0
        span_b = max(wall_b_v) - min(wall_b_v) if wall_b_v else 0.0
        contiguous_span_a = _longest_connected_v_span(wall_a_uv)
        contiguous_span_b = _longest_connected_v_span(wall_b_uv)
        required_track_bins = tuple(range(track_bin_count))
        scanlines = tuple(
            -central_chord_half + (index + 0.5) * SITE_SAMPLE_STEP_MM
            for index in required_track_bins)
        wall_a_nearby_tracks = _transverse_track_intersections(
            wall_a_track_segments, scanlines_v_mm=scanlines)
        wall_b_nearby_tracks = _transverse_track_intersections(
            wall_b_track_segments, scanlines_v_mm=scanlines)
        wall_a_tracks = _cavity_boundary_track_intersections(
            wall_a_nearby_tracks,
            expected_cavity_edge_mm=face_skin,
            cavity_edge_direction=1)
        wall_b_tracks = _cavity_boundary_track_intersections(
            wall_b_nearby_tracks,
            expected_cavity_edge_mm=face_skin + cavity_depth,
            cavity_edge_direction=-1)
        interface_single_path = _single_transverse_classic_track_summary(
            track_samples=wall_a_tracks,
            nearby_track_samples=wall_a_nearby_tracks,
            required_bins=required_track_bins,
            expected_center_mm=first_wall_center,
            allowed_width_range_mm=RETAINING_BEAD_ACCEPTANCE[
                "transverse_width_range_mm"],
            material_side_sign=-1)
        inner_single_path = _single_transverse_classic_track_summary(
            track_samples=wall_b_tracks,
            nearby_track_samples=wall_b_nearby_tracks,
            required_bins=required_track_bins,
            expected_center_mm=second_wall_center,
            allowed_width_range_mm=RETAINING_BEAD_ACCEPTANCE[
                "transverse_width_range_mm"],
            material_side_sign=1)
        for summary, boundary, direction in (
            (interface_single_path, face_skin, 1),
            (inner_single_path, face_skin + cavity_depth, -1),
        ):
            summary.update({
                "candidate_selection": "cavity_facing_bead_edge",
                "expected_cavity_edge_mm": boundary,
                "cavity_edge_direction": direction,
                "cavity_edge_tolerance_mm": (
                    TRANSVERSE_CAVITY_EDGE_TOLERANCE_MM),
            })
        free_transverse_diameter = (
            min(positive_free_edges) - max(negative_free_edges)
            if positive_free_edges and negative_free_edges else None)
        free_axial_slot = (
            min(inner_cavity_edges) - max(interface_cavity_edges)
            if interface_cavity_edges and inner_cavity_edges else None)
        loading_aperture = {
            "interior_extrusion_path_length_mm": interior,
            "free_transverse_diameter_mm": free_transverse_diameter,
            "free_axial_slot_width_mm": free_axial_slot,
            "central_chord_half_width_mm": central_chord_half,
        }
        retaining = {
            "kind": "two_axial_skin_paths",
            "interface_skin_path_length_mm": wall_a,
            "interface_skin_transverse_span_mm": span_a,
            "interface_skin_longest_contiguous_span_mm": contiguous_span_a,
            "inner_skin_path_length_mm": wall_b,
            "inner_skin_transverse_span_mm": span_b,
            "inner_skin_longest_contiguous_span_mm": contiguous_span_b,
            "interface_skin_single_path": interface_single_path,
            "inner_skin_single_path": inner_single_path,
            "single_classic_path_pass": (
                interface_single_path["single_classic_path_pass"]
                and inner_single_path["single_classic_path_pass"]),
            "connectivity_gap_limit_mm": RETAINING_PATH_CONNECTIVITY_GAP_MM,
            "pass": (
                contiguous_span_a >= 3.0
                and contiguous_span_b >= 3.0
                and interface_single_path["single_classic_path_pass"]
                and inner_single_path["single_classic_path_pass"]),
        }
    else:
        annulus_length = 0.0
        annulus_angles: list[float] = []
        axial_skin = site.get("face_skin_mm", 0.45)
        expected_ring_center = radius + axial_skin / 2.0
        annulus_component_points: dict[
            int, list[tuple[float, float, float]]] = {}
        annulus_track_segments: list[
            tuple[float, float, float, float, str, float, int]] = []
        radial_free_edges: list[float] = []
        for segment in layer.segments:
            if min(segment.x0, segment.x1) > cx + roi or max(segment.x0, segment.x1) < cx - roi:
                continue
            if min(segment.y0, segment.y1) > cy + roi or max(segment.y0, segment.y1) < cy - roi:
                continue
            local_segments.append(segment)
            path_width = (
                segment.line_width
                if segment.line_width is not None and segment.line_width > 0.0
                else FALLBACK_LINE_WIDTH_MM)
            half_path = path_width / 2.0
            annulus_hit = False
            for x, y, weight in _sample_segment(segment):
                radial = math.hypot(x - cx, y - cy)
                if radial < radius - 0.05:
                    interior += weight
                if radial <= radius + 1.0:
                    boundary_distances.append(radial)
                    radial_free_edges.append(radial - half_path)
                if radius - 0.20 <= radial <= radius + 0.80:
                    annulus_hit = True
                    annulus_length += weight
                    annulus_angles.append(math.atan2(y - cy, x - cx))
                if (abs(radial - expected_ring_center)
                        <= axial_skin / 2.0 + 0.04):
                    annulus_component_points.setdefault(
                        segment.path_id, []).append((
                            x - cx, y - cy, path_width))
            annulus_track_segments.append((
                segment.x0, segment.y0, segment.x1, segment.y1,
                segment.feature, path_width, segment.path_id))
        # A complete circumference is not required to be one continuous G-code
        # segment, but adjacent segments must cover the complete circumference;
        # summing unrelated fragments is not evidence of a printable cradle.
        largest_gap = _largest_circular_sample_gap(annulus_angles, radius)
        annulus_tracks = _annular_track_intersections(
            annulus_track_segments, center_xy=(cx, cy),
            expected_radius_mm=expected_ring_center,
            skin_thickness_mm=axial_skin)
        annular_single_path = _single_annular_classic_track_summary(
            track_samples=annulus_tracks,
            required_bins=tuple(range(72)),
            expected_center_mm=expected_ring_center,
            allowed_width_range_mm=RETAINING_BEAD_ACCEPTANCE[
                "axial_width_range_mm"],
            component_points=annulus_component_points)
        free_radial_diameter = (
            2.0 * min(radial_free_edges) if radial_free_edges else None)
        loading_aperture = {
            "interior_extrusion_path_length_mm": interior,
            "free_radial_diameter_mm": free_radial_diameter,
            "free_axial_slot_width_mm": None,
        }
        retaining = {
            "kind": "annular_open_cavity_path",
            "annular_path_length_mm": annulus_length,
            "sample_count": len(annulus_angles),
            "largest_uncovered_arc_mm": (
                largest_gap if math.isfinite(largest_gap) else None),
            "annular_single_path": annular_single_path,
            "single_classic_path_pass": annular_single_path[
                "single_classic_path_pass"],
            "connectivity_gap_limit_mm": RETAINING_PATH_CONNECTIVITY_GAP_MM,
            "pass": (
                annulus_length >= math.pi * radius
                and largest_gap <= RETAINING_PATH_CONNECTIVITY_GAP_MM
                and annular_single_path["single_classic_path_pass"]
            ),
        }
    return {
        "z_mm": layer.z,
        "gcode_line_number": layer.line_number,
        "local_extrusion_segment_count": len(local_segments),
        "roof_interior_path_length_mm": interior,
        "opening_half_width_path_mm": (
            min(boundary_distances) if boundary_distances else None),
        "loading_aperture": loading_aperture,
        "retaining_paths": retaining,
        "segments": local_segments,
    }


def _loading_aperture_pass(
    site: Mapping[str, Any], metrics: Mapping[str, Any],
) -> tuple[bool, str]:
    """Prove the nominal D5x2 disc can enter on the last-open layer."""
    aperture = metrics["loading_aperture"]
    interior = aperture["interior_extrusion_path_length_mm"]
    if site["closure_kind"] == "transverse_gable_45deg":
        diameter = aperture["free_transverse_diameter_mm"]
        slot = aperture["free_axial_slot_width_mm"]
        diameter_pass = (
            diameter is not None
            and diameter >= site["magnet_diameter_mm"] - LAYER_EPS)
        slot_pass = (
            slot is not None
            and slot >= site["magnet_depth_mm"] - LAYER_EPS)
    else:
        diameter = aperture["free_radial_diameter_mm"]
        slot = None
        diameter_pass = (
            diameter is not None
            and diameter >= site["magnet_diameter_mm"] - LAYER_EPS)
        slot_pass = True
    interior_pass = interior <= LAST_OPEN_INTERIOR_PATH_LIMIT_MM + LAYER_EPS
    passed = interior_pass and diameter_pass and slot_pass
    return passed, (
        f"interior path={interior:.3f} mm "
        f"(limit {LAST_OPEN_INTERIOR_PATH_LIMIT_MM:.3f}); "
        f"free diameter={diameter}; free axial slot={slot}; "
        f"required D={site['magnet_diameter_mm']:.3f}, "
        f"depth={site['magnet_depth_mm']:.3f} mm")


def _roof_progression_pass(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[bool, str]:
    last_value = metrics["last_fully_open"]["roof_interior_path_length_mm"]
    first_value = metrics["first_closing_pause"]["roof_interior_path_length_mm"]
    sealed_value = metrics["fully_sealed"]["roof_interior_path_length_mm"]
    last_boundary = metrics["last_fully_open"]["opening_half_width_path_mm"]
    first_boundary = metrics["first_closing_pause"]["opening_half_width_path_mm"]
    sealed_boundary = metrics["fully_sealed"]["opening_half_width_path_mm"]
    # The first 0.16-mm 45-degree roof strip is narrower than a nominal
    # 0.42-mm wall.  Its centreline can therefore remain outside the nominal
    # cavity interior even though Preview has begun closing the roof.  The
    # robust sliced fact is that the nearest roof-boundary path moves inward
    # on the first-closing layer and continues inward by the sealed layer.
    boundary_pass = (
        last_boundary is not None and first_boundary is not None
        and sealed_boundary is not None
        and first_boundary <= last_boundary - 0.03
        and sealed_boundary <= first_boundary - 0.03)
    # Interior deposition is a useful secondary confirmation once the roof
    # has sealed; it is deliberately not required on the sub-line-width first
    # strip.
    sealed_pass = sealed_value >= max(last_value, first_value) + 0.03
    passed = boundary_pass and sealed_pass
    return passed, (
        f"boundary last={last_boundary}, first={first_boundary}, "
        f"sealed={sealed_boundary} mm; interior path last={last_value:.3f}, "
        f"first={first_value:.3f}, sealed={sealed_value:.3f} mm")


def _discover_actual_closure_layers(
    layers: Sequence[Layer],
    site: Mapping[str, Any],
    placement_xy: tuple[float, float],
) -> tuple[dict[str, Layer], dict[str, dict[str, Any]], dict[str, Any]]:
    """Discover the first roof-closing layer from sliced toolpaths.

    The CAD bury plane is a consistency datum, not the layer selector.  The
    selector first finds the widest, loadable chimney/cavity toolpath, then
    scans every following scheduled layer.  The first boundary contraction,
    new cavity-interior deposition, or loss of the loading aperture is the
    actual closing onset.  Every scheduled layer in the preceding loadable
    run must remain indistinguishable from the fully open reference.

    This deliberately fails closed if the closing signature is missing,
    ambiguous, or reopens.  The CAD bury plane is checked as a bounded
    consistency datum against the actual toolpath onset, but it never selects
    the pause layer: a sub-line-width roof strip may not acquire a printable
    Arachne centreline until one or two scheduled layers after the exact CAD
    boundary.  A pause can therefore never be manufactured from nominal CAD Z
    alone.
    """
    if not layers:
        raise AuditError("cannot discover cavity closure without sliced layers")
    bury = float(site["cavity_bury_roof_start_print_z_mm"])
    apex = float(site["roof_apex_print_z_mm"])
    center_z = float(site["print_cavity_center_xyz_mm"][2])
    if site["closure_kind"] == "transverse_gable_45deg":
        bottom = center_z - site["cavity_diameter_mm"] / 2.0
    else:
        bottom = center_z - site["cavity_depth_mm"] / 2.0
    lowest = _layer_at_or_above(layers, bottom)
    sealed = _layer_above(layers, apex)
    scan_start = _layer_at_or_above(layers, center_z)
    scan_layers = [
        layer for layer in layers
        if scan_start.z - LAYER_EPS <= layer.z <= sealed.z + LAYER_EPS
    ]
    if len(scan_layers) < 3:
        raise AuditError(
            f"{site['name']}: too few actual layers to discover roof closure")

    entries: list[dict[str, Any]] = []
    for layer in scan_layers:
        metrics = _toolpath_metrics(layer, site, placement_xy)
        aperture_pass, aperture_detail = _loading_aperture_pass(site, metrics)
        entries.append({
            "layer": layer,
            "metrics": metrics,
            "aperture_pass": aperture_pass,
            "aperture_detail": aperture_detail,
            "boundary": metrics["opening_half_width_path_mm"],
            "interior": metrics["roof_interior_path_length_mm"],
        })

    loadable_boundaries = [
        float(entry["boundary"])
        for entry in entries
        if entry["aperture_pass"] and entry["boundary"] is not None
    ]
    if not loadable_boundaries:
        raise AuditError(
            f"{site['name']}: G-code has no fully loadable open-cavity layer")
    open_boundary = max(loadable_boundaries)
    full_open_indices = [
        index for index, entry in enumerate(entries)
        if (entry["aperture_pass"]
            and entry["boundary"] is not None
            and open_boundary - float(entry["boundary"])
            < CLOSING_BOUNDARY_INSET_MM - LAYER_EPS)
    ]
    if not full_open_indices:
        raise AuditError(
            f"{site['name']}: no stable fully open G-code boundary")
    first_open_index = full_open_indices[0]
    open_interior = min(
        float(entries[index]["interior"])
        for index in full_open_indices
    )

    def closing_reasons(entry: Mapping[str, Any]) -> list[str]:
        reasons = []
        boundary = entry["boundary"]
        if boundary is None:
            reasons.append("opening boundary disappeared")
        elif (open_boundary - float(boundary)
              >= CLOSING_BOUNDARY_INSET_MM - LAYER_EPS):
            reasons.append(
                f"boundary inset {open_boundary - float(boundary):.3f} mm")
        if (float(entry["interior"]) - open_interior
                >= CLOSING_BOUNDARY_INSET_MM - LAYER_EPS):
            reasons.append(
                "new cavity-interior extrusion "
                f"{float(entry['interior']) - open_interior:.3f} mm")
        # A missing sampled chord/slot is not itself evidence of obstruction:
        # seam placement can temporarily leave no segment crossing the exact
        # probe line even while the full D5 aperture remains unchanged.  Count
        # aperture loss only when the G-code supplies a finite, measured free
        # dimension that is actually below the nominal magnet envelope.
        aperture = entry["metrics"]["loading_aperture"]
        free_diameter = (
            aperture["free_transverse_diameter_mm"]
            if site["closure_kind"] == "transverse_gable_45deg"
            else aperture["free_radial_diameter_mm"])
        free_slot = (
            aperture["free_axial_slot_width_mm"]
            if site["closure_kind"] == "transverse_gable_45deg"
            else None)
        measured_aperture_blocked = (
            (free_diameter is not None
             and float(free_diameter)
             < site["magnet_diameter_mm"] - LAYER_EPS)
            or (free_slot is not None
                and float(free_slot)
                < site["magnet_depth_mm"] - LAYER_EPS))
        if measured_aperture_blocked:
            reasons.append("nominal D5 loading aperture is no longer open")
        return reasons

    closing_index: int | None = None
    onset_reasons: list[str] = []
    for index in range(first_open_index + 1, len(entries)):
        reasons = closing_reasons(entries[index])
        if reasons:
            closing_index = index
            onset_reasons = reasons
            break
    if closing_index is None:
        raise AuditError(
            f"{site['name']}: no roof-closing signature was found in G-code")
    if closing_index <= first_open_index:
        raise AuditError(
            f"{site['name']}: roof closure has no preceding fully open layer")

    prior_entries = entries[first_open_index:closing_index]
    prior_failures = []
    for entry in prior_entries:
        reasons = closing_reasons(entry)
        if reasons:
            prior_failures.append(
                f"Z={entry['layer'].z:.3f}: " + "; ".join(reasons))
    if prior_failures:
        raise AuditError(
            f"{site['name']}: earlier scheduled cavity layers already close: "
            + " | ".join(prior_failures))

    last_entry = entries[closing_index - 1]
    first_entry = entries[closing_index]
    if first_entry["boundary"] is None:
        raise AuditError(
            f"{site['name']}: first actual closing layer has no auditable "
            "roof boundary")
    cad_consistency_tolerance = max(
        float(site.get("minimum_retaining_path_mm", FALLBACK_LINE_WIDTH_MM)),
        LAYER_EPS)
    cad_bury_directly_bracketed = (
        last_entry["layer"].z <= bury + LAYER_EPS
        and first_entry["layer"].z > bury + LAYER_EPS)
    if (first_entry["layer"].z <= bury + LAYER_EPS
            or abs(last_entry["layer"].z - bury)
            > cad_consistency_tolerance + LAYER_EPS
            or abs(first_entry["layer"].z - bury)
            > cad_consistency_tolerance + LAYER_EPS):
        raise AuditError(
            f"{site['name']}: actual G-code closing onset "
            f"{first_entry['layer'].z:.3f} mm must begin above, and remain "
            f"within one nominal path width of, CAD bury plane {bury:.3f} "
            f"mm (tolerance {cad_consistency_tolerance:.3f} mm); last actual "
            f"open layer is {last_entry['layer'].z:.3f} mm")

    # A qualified 45-degree roof never reopens.  Inspect every remaining
    # boundary through the first fully sealed evidence layer, not merely the
    # first and last snapshots.
    previous_boundary = float(first_entry["boundary"])
    for entry in entries[closing_index + 1:]:
        boundary = entry["boundary"]
        if boundary is None:
            raise AuditError(
                f"{site['name']}: roof boundary vanished at "
                f"Z={entry['layer'].z:.3f} mm")
        boundary = float(boundary)
        if boundary > (
                previous_boundary
                + CLOSING_BOUNDARY_REOPEN_TOLERANCE_MM + LAYER_EPS):
            raise AuditError(
                f"{site['name']}: roof reopens in G-code at "
                f"Z={entry['layer'].z:.3f} mm "
                f"({previous_boundary:.3f} -> {boundary:.3f} mm)")
        previous_boundary = boundary

    representative_entry = prior_entries[len(prior_entries) // 2]
    selected = {
        "lowest_open": lowest,
        "representative_open": representative_entry["layer"],
        "last_fully_open": last_entry["layer"],
        "first_closing_pause": first_entry["layer"],
        "fully_sealed": sealed,
    }
    metrics_by_z = {
        round(float(entry["layer"].z), 6): entry["metrics"]
        for entry in entries
    }
    selected_metrics = {
        key: metrics_by_z.get(
            round(float(layer.z), 6),
            _toolpath_metrics(layer, site, placement_xy),
        )
        for key, layer in selected.items()
    }
    discovery = {
        "method": "earliest_actual_gcode_roof_closing_signature",
        "open_reference_boundary_half_width_mm": open_boundary,
        "open_reference_interior_path_length_mm": open_interior,
        "boundary_inset_threshold_mm": CLOSING_BOUNDARY_INSET_MM,
        "examined_layer_z_mm": [entry["layer"].z for entry in entries],
        "proven_fully_open_layer_z_mm": [
            entry["layer"].z for entry in prior_entries],
        "all_prior_scheduled_open_layers_pass": True,
        "first_closing_layer_z_mm": first_entry["layer"].z,
        "first_closing_signature": onset_reasons,
        "cad_bury_plane_bracketed": cad_bury_directly_bracketed,
        "cad_bury_plane_consistent_with_toolpath": True,
        "cad_bury_plane_consistency_tolerance_mm": (
            cad_consistency_tolerance),
    }
    return selected, selected_metrics, discovery


__all__ = tuple(
    name for name in globals()
    if name != "__all__" and not name.startswith("__"))
