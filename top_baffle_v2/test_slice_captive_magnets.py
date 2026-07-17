"""Pure-Python gates for the offline captive-magnet slicing pipeline."""

from __future__ import annotations

import json
import inspect
import math
from pathlib import Path
import tempfile

import slice_captive_magnets as audit
from front_down_contract import (
    RELEASE_ACOUSTIC_PRINT_CONTRACT,
    write_print_sidecar,
)


def _release_site_contract(axis=(1.0, 0.0, 0.0)) -> dict:
    return {
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "face_skin_mm": 0.45,
        "inner_skin_mm": 0.45,
        "captive_land_mm": 3.0,
        "interface_gap_mm": 0.05,
        "paired_magnet_face_separation_mm": 0.95,
        "roof_angle_deg": 45.0,
        "classic_retaining_path_mm": 0.42,
        "polarity_instruction": "marked/N pole follows installed axis",
        "installed_marked_pole_axis_xyz": list(axis),
        # All synthetic release fixtures use the same source front-down
        # convention as production: loading motion is source +Z, which X180
        # transforms to a vertically downward print-space -Z insertion.
        "insertion_direction_xyz": [0.0, 0.0, 1.0],
        "magnet_count": 1,
        "structural_load_credit_n": 0.0,
    }


def _catalog_document(artifacts: list[dict]) -> dict:
    """Minimal hash-shaped envelope for synthetic consumer fixtures."""
    for artifact in artifacts:
        artifact.setdefault("stl_sha256", "a" * 64)
        artifact.setdefault("print_sidecar", "print-authority.json")
        artifact.setdefault("print_sidecar_sha256", "b" * 64)
        artifact.setdefault("source_files", ["source.py"])
        artifact.setdefault("source_file_sha256", {
            value: "c" * 64 for value in artifact["source_files"]})
        if artifact.get("variant") in ("V1LF", "V1LF-split"):
            artifact.setdefault("stage_manifest", "stage_manifest.json")
            artifact.setdefault("stage_manifest_sha256", "e" * 64)
        if artifact.get("variant") in ("V1LF-Ac", "V1LF-Ae"):
            artifact.setdefault("transaction_manifest", "transaction.json")
            artifact.setdefault("transaction_manifest_sha256", "f" * 64)
            artifact.setdefault("facts", "facts.json")
            artifact.setdefault("facts_sha256", "1" * 64)
        matrix = artifact.get("source_to_stl_matrix")
        if isinstance(matrix, list) and len(matrix) == 4:
            for site in artifact.get("sites", ()):
                if not isinstance(site, dict) or "print_space" in site:
                    continue
                print_space = {}
                for key in (
                        "cavity_center_xyz_mm",
                        "seated_magnet_center_xyz_mm",
                        "actual_face_xyz_mm"):
                    value = site.get(key)
                    if isinstance(value, list) and len(value) == 3:
                        print_space[key] = [
                            sum(float(matrix[row][column]) * value[column]
                                for column in range(3))
                            + float(matrix[row][3])
                            for row in range(3)
                        ]
                for key in (
                        "marked_pole_axis_xyz", "material_inward_xyz",
                        "insertion_direction_xyz"):
                    value = site.get(key)
                    if isinstance(value, list) and len(value) == 3:
                        print_space[key] = [
                            sum(float(matrix[row][column]) * value[column]
                                for column in range(3))
                            for row in range(3)
                        ]
                if {
                        "cavity_center_xyz_mm",
                        "seated_magnet_center_xyz_mm",
                        "marked_pole_axis_xyz",
                } <= set(print_space):
                    site["print_space"] = print_space
    magnet_count = sum(
        int(site.get("magnet_count", 0))
        for artifact in artifacts for site in artifact.get("sites", ()))
    return {
        "schema_version": 1,
        "schema_sha256": audit.sha256_file(audit.CATALOG_SCHEMA),
        "catalog_kind": "released_pause_and_bury_captive_magnets",
        "generated_by": "synthetic-test",
        "source_revision": "d" * 64,
        "print_contract": dict(RELEASE_ACOUSTIC_PRINT_CONTRACT),
        "geometry": {},
        "inventory": {
            "artifact_count": len(artifacts),
            "magnet_count": magnet_count,
        },
        "exclusions": [{"path": "none", "reason": "synthetic fixture"}],
        "artifacts": artifacts,
    }


def _minimal_catalog_site() -> dict:
    return {
        **_release_site_contract((0.0, 0.0, -1.0)),
        "name": "synthetic_site",
        "closure_kind": "axis_opposed_conical_45deg",
        "cavity_bury_roof_start_print_z_mm": 5.8,
        "roof_apex_print_z_mm": 8.4,
        "cavity_center_xyz_mm": [0.0, 0.0, 3.0],
        "seated_magnet_center_xyz_mm": [0.0, 0.0, 3.0],
        "marked_pole_axis_xyz": [0.0, 0.0, -1.0],
    }


def _bind_existing_stl_and_sidecar(
        root: Path, raw: dict, *, subdirectory: str | None = None) -> Path:
    stl = root / raw["stl"]
    assert stl.is_file()
    matrix = raw["source_to_stl_matrix"]
    translation = [float(matrix[row][3]) for row in range(3)]
    sidecar = write_print_sidecar(
        stl,
        part=stl.stem,
        transform={
            "print_orientation": raw["print_orientation"],
            "rotation_deg": raw["rotation_deg"],
            "source_to_stl_matrix": matrix,
            "pre_translation_bbox_min_mm": [
                -value for value in translation],
            "stl_origin_translation_mm": translation,
        },
    )
    if subdirectory is not None:
        destination = root / subdirectory / sidecar.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        sidecar.replace(destination)
        sidecar = destination
    raw["stl_sha256"] = audit.sha256_file(stl)
    raw["print_sidecar"] = sidecar.relative_to(root).as_posix()
    raw["print_sidecar_sha256"] = audit.sha256_file(sidecar)
    source_hashes = {}
    for value in raw["source_files"]:
        source = (root / value).resolve()
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            source.write_text("# synthetic bound source\n", encoding="utf-8")
        source_hashes[value] = audit.sha256_file(source)
    raw["source_file_sha256"] = source_hashes
    for path_key, hash_key in (
            ("transaction_manifest", "transaction_manifest_sha256"),
            ("facts", "facts_sha256"),
            ("stage_manifest", "stage_manifest_sha256")):
        value = raw.get(path_key)
        if value is None:
            continue
        bound = (root / value).resolve()
        bound.parent.mkdir(parents=True, exist_ok=True)
        if not bound.is_file():
            bound.write_text(
                json.dumps({"kind": path_key}) + "\n", encoding="utf-8")
        raw[hash_key] = audit.sha256_file(bound)
    return sidecar


def _write_bound_stl_and_sidecar(
        root: Path, raw: dict, *, subdirectory: str | None = None) -> Path:
    stl = root / raw["stl"]
    stl.parent.mkdir(parents=True, exist_ok=True)
    stl.write_text("""solid fixture
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid fixture
""", encoding="ascii")
    return _bind_existing_stl_and_sidecar(
        root, raw, subdirectory=subdirectory)


def _synthetic_gcode(
        path: Path, *, first_closing_z: float | None) -> None:
    """Write cavity-local Classic paths with an explicit roof onset."""
    lines = [
        "M83", "M104 S245", "M140 S55", "G90", "G1 X10 Y10 Z0.2",
    ]
    schedule = [0.20]
    while schedule[-1] < 12.0:
        schedule.append(round(schedule[-1] + 0.16, 2))
    for z in schedule:
        if first_closing_z is None or z < first_closing_z - 1.0e-9:
            boundary = 2.82
        else:
            boundary = max(
                0.20,
                2.82 - 0.16 * (1.0 + round(
                    (z - first_closing_z) / 0.16)),
            )
        lines.extend((
            "; CHANGE_LAYER", f"; Z_HEIGHT: {z:.2f}",
            "; LAYER_HEIGHT: 0.16", "; FEATURE: Outer wall",
            "; LINE_WIDTH: 0.42", f"G1 Z{z:.2f}",
            # Continuous interface- and inner-skin retaining paths.
            "G1 X0.225 Y-2.80", "G1 X0.225 Y2.80 E0.20",
            "G1 X2.775 Y-2.80", "G1 X2.775 Y2.80 E0.20",
            # The two free roof/opening boundaries.  Their inward movement is
            # the actual sliced closing signature.
            f"G1 X0.70 Y{boundary:.3f}",
            f"G1 X2.30 Y{boundary:.3f} E0.08",
            f"G1 X0.70 Y{-boundary:.3f}",
            f"G1 X2.30 Y{-boundary:.3f} E0.08",
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_actual_bambu_layer_regression(tmp_path: Path):
    common = {
        "name": "coupon_equivalent",
        "closure_kind": "transverse_gable_45deg",
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
        "face_skin_mm": 0.45,
        "inner_skin_mm": 0.45,
        "print_actual_face_xyz_mm": (0.0, 0.0, 0.0),
        "print_material_inward_xyz": (1.0, 0.0, 0.0),
        "print_marked_pole_axis_xyz": (1.0, 0.0, 0.0),
    }
    um = {
        **common,
        "print_cavity_center_xyz_mm": (1.50, 0.0, 3.20),
        "print_seated_magnet_center_xyz_mm": (1.45, 0.0, 3.20),
        "cavity_bury_roof_start_print_z_mm": 5.80,
        "roof_apex_print_z_mm": 8.40,
    }
    lm = {
        **common,
        "print_cavity_center_xyz_mm": (1.50, 0.0, 5.75),
        "print_seated_magnet_center_xyz_mm": (1.45, 0.0, 5.75),
        "cavity_bury_roof_start_print_z_mm": 8.40,
        "roof_apex_print_z_mm": 11.00,
    }
    um_gcode = tmp_path / "um_synthetic.gcode"
    lm_gcode = tmp_path / "lm_synthetic.gcode"
    _synthetic_gcode(um_gcode, first_closing_z=5.96)
    _synthetic_gcode(lm_gcode, first_closing_z=8.52)
    um_layers, _um_metrics, um_discovery = (
        audit._discover_actual_closure_layers(
            audit.parse_gcode(um_gcode).layers, um, (0.0, 0.0)))
    lm_layers, _lm_metrics, lm_discovery = (
        audit._discover_actual_closure_layers(
            audit.parse_gcode(lm_gcode).layers, lm, (0.0, 0.0)))
    assert math.isclose(um_layers["last_fully_open"].z, 5.80)
    assert math.isclose(um_layers["first_closing_pause"].z, 5.96)
    assert math.isclose(lm_layers["last_fully_open"].z, 8.36)
    assert math.isclose(lm_layers["first_closing_pause"].z, 8.52)
    assert um_discovery["all_prior_scheduled_open_layers_pass"] is True
    assert lm_discovery["all_prior_scheduled_open_layers_pass"] is True
    assert um_discovery["method"] == (
        "earliest_actual_gcode_roof_closing_signature")

    # Horizontal coupon magnets have a 2.5-mm vertical radius.  A properly
    # seated D5 disc therefore remains below both completed last-open layers.
    assert math.isclose(
        um_layers["last_fully_open"].z
        - audit._seated_magnet_print_z_bounds(um)[1],
        0.10, abs_tol=1.0e-12)
    assert math.isclose(
        lm_layers["last_fully_open"].z
        - audit._seated_magnet_print_z_bounds(lm)[1],
        0.11, abs_tol=1.0e-12)


def test_actual_closure_discovery_rejects_early_or_missing_roof(
        tmp_path: Path):
    site = {
        "name": "unsafe",
        "closure_kind": "transverse_gable_45deg",
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
        "face_skin_mm": 0.45,
        "inner_skin_mm": 0.45,
        "print_actual_face_xyz_mm": (0.0, 0.0, 0.0),
        "print_material_inward_xyz": (1.0, 0.0, 0.0),
        "print_cavity_center_xyz_mm": (1.50, 0.0, 3.20),
        "print_seated_magnet_center_xyz_mm": (1.45, 0.0, 3.20),
        "print_marked_pole_axis_xyz": (1.0, 0.0, 0.0),
        "cavity_bury_roof_start_print_z_mm": 5.80,
        "roof_apex_print_z_mm": 8.40,
    }
    early = tmp_path / "early.gcode"
    _synthetic_gcode(early, first_closing_z=5.80)
    try:
        audit._discover_actual_closure_layers(
            audit.parse_gcode(early).layers, site, (0.0, 0.0))
    except audit.AuditError as exc:
        assert "actual G-code closing onset 5.800" in str(exc)
    else:
        raise AssertionError("an early actual roof onset was accepted")

    missing = tmp_path / "missing.gcode"
    _synthetic_gcode(missing, first_closing_z=None)
    try:
        audit._discover_actual_closure_layers(
            audit.parse_gcode(missing).layers, site, (0.0, 0.0))
    except audit.AuditError as exc:
        assert "no roof-closing signature" in str(exc)
    else:
        raise AssertionError("a CAD-only pause with no G-code roof was accepted")


def test_arc_fitted_full_circles_are_tessellated_and_auditable(
        tmp_path: Path):
    gcode = tmp_path / "arcs.gcode"
    gcode.write_text("\n".join((
        "M83", "M104 S220", "M140 S55", "G90",
        "; CHANGE_LAYER", "; Z_HEIGHT: 0.20", "; LAYER_HEIGHT: 0.20",
        "; FEATURE: Outer wall", "; LINE_WIDTH: 0.42",
        "G1 X12.81 Y10.0 Z0.20",
        "G2 X12.81 Y10.0 I-2.81 E1.0",
        "; CHANGE_LAYER", "; Z_HEIGHT: 0.36", "; LAYER_HEIGHT: 0.16",
        "G1 Z0.36",
        "G3 X12.81 Y10.0 I-2.81 E1.0",
    )) + "\n", encoding="utf-8")
    parsed = audit.parse_gcode(gcode)
    assert parsed.arc_commands == 2
    assert parsed.extrusion_moves == 2
    assert len(parsed.layers[0].segments) > 80
    assert len(parsed.layers[1].segments) > 80
    circumference = 2.0 * math.pi * 2.81
    for layer in parsed.layers:
        assert math.isclose(
            sum(segment.length for segment in layer.segments),
            circumference, rel_tol=0.002)
        assert math.isclose(
            sum(segment.e_delta for segment in layer.segments),
            1.0, abs_tol=1.0e-12)
    assert parsed.bounds_min[0] <= 7.19 + 1.0e-3
    assert parsed.bounds_min[1] <= 7.19 + 1.0e-3
    assert parsed.bounds_max[0] >= 12.81 - 1.0e-3
    assert parsed.bounds_max[1] >= 12.81 - 1.0e-3

    site = {
        "closure_kind": "axis_opposed_conical_45deg",
        "print_cavity_center_xyz_mm": (10.0, 10.0, 0.20),
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
    }
    metrics = audit._toolpath_metrics(
        parsed.layers[0], site, (0.0, 0.0))
    assert metrics["retaining_paths"]["pass"] is True
    assert audit._loading_aperture_pass(site, metrics)[0] is True

    # Production parsing retains only cavity-local extrusion, but global arc
    # bounds/counts remain authoritative even when this circle is outside ROI.
    bounded = audit.parse_gcode(
        gcode, retain_regions=((0.0, 0.0, 1.0, 1.0),))
    assert bounded.arc_commands == 2
    assert bounded.extrusion_moves == 2
    assert all(not layer.segments for layer in bounded.layers)
    assert bounded.bounds_min == parsed.bounds_min
    assert bounded.bounds_max == parsed.bounds_max


def test_feature_bridge_height_does_not_override_layer_schedule(
        tmp_path: Path):
    """Bambu bridge-flow metadata is not a variable layer-height command."""
    gcode = tmp_path / "bridge_metadata.gcode"
    gcode.write_text("\n".join((
        "M83", "M104 S220", "M140 S55", "G90",
        "; CHANGE_LAYER", "; Z_HEIGHT: 0.20", "; LAYER_HEIGHT: 0.20",
        "; FEATURE: Outer wall", "; LINE_WIDTH: 0.42",
        "G1 X0 Y0 Z0.20", "G1 X1 Y0 E0.1",
        "; FEATURE: Bridge", "; LINE_WIDTH: 0.40",
        "; LAYER_HEIGHT: 0.40", "G1 X1 Y1 E0.1",
        "; CHANGE_LAYER", "; Z_HEIGHT: 0.36", "; LAYER_HEIGHT: 0.16",
        "; FEATURE: Bridge", "; LINE_WIDTH: 0.40",
        "; LAYER_HEIGHT: 0.40", "G1 X0 Y1 Z0.36 E0.1",
    )) + "\n", encoding="utf-8")
    parsed = audit.parse_gcode(gcode)
    assert [layer.layer_height for layer in parsed.layers] == [0.20, 0.16]


def test_arc_parser_fails_closed_on_unsupported_encodings(tmp_path: Path):
    prefix = "\n".join((
        "M83", "M104 S220", "M140 S55", "G90",
        "; CHANGE_LAYER", "; Z_HEIGHT: 0.20", "; LAYER_HEIGHT: 0.20",
        "; FEATURE: Outer wall", "; LINE_WIDTH: 0.42",
        "G1 X12.6 Y10.0 Z0.20",
    ))
    for name, command, expected in (
        ("radius", "G2 X7.4 Y10.0 R2.6 E1.0", "radius-encoded"),
        ("missing_ij", "G2 X7.4 Y10.0 E1.0", "lacks relative I/J"),
        ("absolute_center", "G90.1", "absolute arc-centre"),
    ):
        path = tmp_path / f"{name}.gcode"
        path.write_text(prefix + "\n" + command + "\n", encoding="utf-8")
        try:
            audit.parse_gcode(path)
        except audit.AuditError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"unsupported arc encoding passed: {name}")


def test_profile_inheritance_and_include(tmp_path: Path):
    root = tmp_path / "BBL"
    root.mkdir()
    records = {
        "base.json": {"name": "base", "type": "process", "a": "1", "x": "base"},
        "template.json": {"name": "template", "b": "2", "x": "template"},
        "child.json": {"name": "child", "type": "process", "inherits": "base",
                       "include": ["template"], "x": "child", "c": "3"},
    }
    for name, data in records.items():
        (root / name).write_text(json.dumps(data), encoding="utf-8")
    resolver = audit.PresetResolver(root)
    result = resolver.resolve(root / "child.json")
    assert result["a"] == "1"
    assert result["b"] == "2"
    assert result["c"] == "3"
    assert result["x"] == "child"
    assert "inherits" not in result and "include" not in result
    assert len(resolver.dependencies) == 3


def test_catalog_rejects_non_front_down(tmp_path: Path):
    catalog = _catalog_document([{
            "id": "bad", "part": "bad", "variant": "test",
            "state": "test", "stl": "bad.stl",
            "print_orientation": "rear_face_down",
            "rotation_deg": {"x": 180.0, "z": 0.0},
            "source_to_stl_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "sites": [_minimal_catalog_site()],
        }])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    try:
        audit.normalize_catalog(path)
    except audit.AuditError as exc:
        assert "front_face_down" in str(exc)
    else:
        raise AssertionError("non-front-down release was accepted")


def test_catalog_rejects_stale_root_print_contract(tmp_path: Path):
    catalog = _catalog_document([{
        "id": "part", "part": "part", "variant": "test",
        "state": "test", "stl": "part.stl",
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": 0.0},
        "source_to_stl_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "sites": [_minimal_catalog_site()],
    }])
    catalog["print_contract"]["scope"] = "magnet_bearing_only"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    try:
        audit.normalize_catalog(path, enforce_release_inventory=False)
    except audit.AuditError as exc:
        assert "print_contract" in str(exc)
    else:
        raise AssertionError("stale root print contract was accepted")


def test_catalog_rejects_tilted_matrix_with_front_down_label(tmp_path: Path):
    catalog = _catalog_document([{
            "id": "tilted", "part": "tilted", "variant": "test",
            "state": "test", "stl": "tilted.stl",
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 0.0},
            "source_to_stl_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "sites": [_minimal_catalog_site()],
        }])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    try:
        audit.normalize_catalog(path)
    except audit.AuditError as exc:
        assert "X180 plus Z-only" in str(exc)
    else:
        raise AssertionError("tilted matrix with front-down label was accepted")


def test_source_contract_rejects_missing_safety_fields() -> None:
    site = {
        **_release_site_contract(),
        "name": "site",
        "closure_kind": "transverse_gable_45deg",
        "cavity_center_xyz_mm": [0.0, 0.0, 3.0],
        "seated_magnet_center_xyz_mm": [0.0, 0.0, 3.0],
        "marked_pole_axis_xyz": [1.0, 0.0, 0.0],
        "actual_face_xyz_mm": [0.0, 0.0, 3.0],
        "material_inward_xyz": [-1.0, 0.0, 0.0],
        "cavity_bury_roof_start_print_z_mm": 5.8,
        "roof_apex_print_z_mm": 8.4,
    }
    for key in (
            "magnet_diameter_mm", "cavity_diameter_mm", "face_skin_mm",
            "installed_marked_pole_axis_xyz", "polarity_instruction",
            "insertion_direction_xyz", "magnet_count",
            "structural_load_credit_n"):
        incomplete = dict(site)
        incomplete.pop(key)
        try:
            audit._source_site_contract(incomplete)
        except audit.AuditError as exc:
            assert "required" in str(exc) or key.replace("_", " ") in str(exc)
        else:
            raise AssertionError(f"missing safety field was defaulted: {key}")


def test_bambu_command_explicitly_disables_auto_orientation(tmp_path: Path):
    profile_bundle = {
        "paths": {
            "machine": tmp_path / "machine.json",
            "process": tmp_path / "process.json",
            "filament": tmp_path / "filament.json",
        },
    }
    command = audit._bambu_command(
        tmp_path / "BambuStudio", tmp_path / "part.stl",
        tmp_path / "out", profile_bundle)
    assert command[command.index("--orient") + 1] == "0"
    assert "--rotate-x" not in command
    assert "--rotate-y" not in command
    assert "--allow-rotations" not in command
    assert command[command.index("--export-3mf") + 1] == (
        audit.PLACED_3MF_FILENAME)


def test_cached_slice_reuse_is_hash_bound(tmp_path: Path) -> None:
    stl = tmp_path / "part.stl"
    gcode = tmp_path / "plate_1.gcode"
    result = tmp_path / "result.json"
    project = tmp_path / "audited_slice_project.3mf"
    stl.write_bytes(b"stl")
    gcode.write_bytes(b"gcode")
    result.write_bytes(b"result")
    project.write_bytes(b"3mf")
    prior = {
        "fingerprint": "input-contract",
        "stl_sha256": audit.sha256_file(stl),
        "gcode_sha256": audit.sha256_file(gcode),
        "result_sha256": audit.sha256_file(result),
        "project_3mf_sha256": audit.sha256_file(project),
    }
    assert audit._cached_slice_matches(
        prior, fingerprint="input-contract", stl=stl,
        gcode=gcode, result_path=result, project_3mf=project)
    gcode.write_bytes(b"tampered")
    assert not audit._cached_slice_matches(
        prior, fingerprint="input-contract", stl=stl,
        gcode=gcode, result_path=result, project_3mf=project)


def test_bambu_arrangement_transforms_every_site_roi_and_axis() -> None:
    matrix = (
        (0.0, -1.0, 0.0, 10.0),
        (1.0, 0.0, 0.0, 20.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    site = {
        "name": "station",
        "print_cavity_center_xyz_mm": [1.0, 2.0, 3.0],
        "print_seated_magnet_center_xyz_mm": [-1.0, 0.0, 4.0],
        "print_actual_face_xyz_mm": [2.0, -3.0, 5.0],
        "print_material_inward_xyz": [1.0, 0.0, 0.0],
        "print_marked_pole_axis_xyz": [0.0, 1.0, 0.0],
        "print_insertion_direction_xyz": [0.0, 0.0, -1.0],
    }
    transformed = audit._site_in_bambu_bed_space(site, matrix)
    assert transformed["print_cavity_center_xyz_mm"] == (8.0, 21.0, 3.0)
    assert transformed["print_seated_magnet_center_xyz_mm"] == (
        10.0, 19.0, 4.0)
    assert transformed["print_actual_face_xyz_mm"] == (13.0, 22.0, 5.0)
    assert transformed["print_material_inward_xyz"] == (0.0, 1.0, 0.0)
    assert transformed["print_marked_pole_axis_xyz"] == (-1.0, 0.0, 0.0)
    assert transformed["print_insertion_direction_xyz"] == (0.0, 0.0, -1.0)
    assert transformed["catalog_stl_print_space"] == {
        key: value for key, value in site.items() if key != "name"
    }


def test_failed_actual_slice_emits_no_pause_group() -> None:
    failed = {
        "audit_mode": "actual_p2s_slice",
        "status": "fail",
        "sites": [{
            "site": {"name": "unsafe"},
            "actual": {"bambu_studio_pause_marker_z_mm": 5.96},
        }],
    }
    assert audit._pause_groups(failed) == []


def test_pause_group_preserves_insertion_and_full_polarity_instruction() -> None:
    polarity = (
        "provisional unpaired V0 convention: marked/N pole points rearward; "
        "verify any future mate before burial")
    record = {
        "id": "state:V0:part",
        "audit_mode": "actual_p2s_slice",
        "status": "pass",
        "sites": [{
            "site": {
                "name": "v0_left",
                "print_insertion_direction_xyz": (0.0, 0.0, -1.0),
                "print_marked_pole_axis_xyz": (0.0, 0.0, 1.0),
                "installed_marked_pole_axis_xyz": (0.0, 0.0, -1.0),
                "polarity_instruction": polarity,
            },
            "actual": {
                "bambu_studio_pause_marker_z_mm": 15.32,
                "last_completely_open_layer_z_mm": 15.16,
                "cavity_bury_roof_start_plane_z_mm": 15.25,
            },
            "seated_magnet": {
                "below_last_open_layer_mm": 0.20,
                "below_first_closing_layer_mm": 0.36,
            },
        }],
    }
    groups = audit._pause_groups(record)
    assert len(groups) == 1
    assert groups[0]["print_insertion_direction_xyz"] == [0.0, 0.0, -1.0]
    assert groups[0]["insertion_instruction"] == (
        audit.PRINT_INSERTION_INSTRUCTION)
    assert groups[0]["polarity"][0]["instruction"] == polarity

    record["sites"][0]["site"]["print_insertion_direction_xyz"] = (
        0.0, 0.0, 1.0)
    try:
        audit._pause_groups(record)
    except audit.AuditError as exc:
        assert "unsafe insertion direction" in str(exc)
    else:
        raise AssertionError("unsafe pause-group insertion was accepted")


def test_only_and_dry_run_are_never_authoritative() -> None:
    assert audit._authoritative_run_requested([], False) is True
    assert audit._authoritative_run_requested(["*"], False) is False
    assert audit._authoritative_run_requested([], True) is False
    assert audit._authoritative_run_requested(["one-part"], True) is False


def test_svg_png_evidence_must_be_fresh_and_nonempty(
        tmp_path: Path) -> None:
    svg = tmp_path / "evidence.svg"
    png = tmp_path / "evidence.png"
    svg.write_text("<svg/>\n", encoding="utf-8")
    png.write_bytes(b"stale")
    real_which = audit.shutil.which
    real_run = audit.subprocess.run
    try:
        audit.shutil.which = lambda _name: None
        try:
            audit._svg_to_png(svg, png)
        except audit.AuditError as exc:
            assert "fresh PNG evidence" in str(exc)
        else:
            raise AssertionError("stale PNG evidence was accepted")
        assert not png.exists()

        audit.shutil.which = lambda name: (
            "/synthetic/rsvg-convert" if name == "rsvg-convert" else None)

        class Completed:
            returncode = 0
            stdout = ""

        def render(command, **_kwargs):
            Path(command[command.index("-o") + 1]).write_bytes(b"fresh-png")
            return Completed()

        audit.subprocess.run = render
        result = audit._svg_to_png(svg, png)
        assert png.read_bytes() == b"fresh-png"
        assert result["sha256"] == audit.sha256_file(png)
    finally:
        audit.shutil.which = real_which
        audit.subprocess.run = real_run


def test_release_inputs_are_sliced_from_immutable_staged_bytes(
        tmp_path: Path) -> None:
    raw = {
        "id": "state:test:part", "part": "part", "variant": "test",
        "state": "state", "stl": "part.stl",
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": 0.0},
        "source_to_stl_matrix": [
            [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "sites": [_minimal_catalog_site()],
    }
    payload = _catalog_document([raw])
    _write_bound_stl_and_sidecar(tmp_path, raw)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = audit.normalize_catalog(
        catalog_path, enforce_release_inventory=False)["artifacts"][0]
    stage_root = tmp_path / "stage"
    staged = audit._stage_release_inputs(
        catalog_path, [artifact], stage_root)
    staged_artifact = staged["artifacts"][0]
    assert staged_artifact["stl"] != artifact["stl"]
    assert staged_artifact["stl"].read_bytes() == artifact["stl"].read_bytes()
    assert staged_artifact["print_sidecar"].parent == staged_artifact[
        "stl"].parent
    original_staged_bytes = staged_artifact["stl"].read_bytes()
    artifact["stl"].write_bytes(b"concurrent release replacement")
    assert staged_artifact["stl"].read_bytes() == original_staged_bytes
    try:
        audit._verify_staged_release_inputs(staged, catalog_path)
    except audit.AuditError as exc:
        assert "STL hash differs" in str(exc)
    else:
        raise AssertionError("concurrent release replacement went undetected")


def _minimal_passing_release_record(
        tmp_path: Path, artifact: dict) -> dict:
    files = {}
    for name, payload in (
            ("evidence.svg", b"<svg/>"), ("evidence.png", b"png"),
            ("plate.gcode", b"G1 X1 E1\n"), ("result.json", b"{}\n"),
            ("audited.3mf", b"3mf")):
        path = tmp_path / name
        path.write_bytes(payload)
        files[name] = path
    site = artifact["sites"][0]
    return {
        "id": artifact["id"], "state": artifact["state"],
        "variant": artifact["variant"], "part": artifact["part"],
        "print_orientation": "front_face_down",
        "audit_mode": "actual_p2s_slice", "status": "pass", "errors": [],
        "input": {
            "stl": str(artifact["stl"]),
            "stl_sha256": artifact["stl_catalog_sha256"],
            "source_files": [{
                "path": str(path), "sha256": digest,
            } for path, digest in artifact["source_file_sha256"].items()],
        },
        "slicer": {
            "gcode": str(files["plate.gcode"]),
            "gcode_sha256": audit.sha256_file(files["plate.gcode"]),
            "result_json": str(files["result.json"]),
            "result_sha256": audit.sha256_file(files["result.json"]),
            "project_3mf": str(files["audited.3mf"]),
            "project_3mf_sha256": audit.sha256_file(files["audited.3mf"]),
        },
        "evidence": {
            "svg": str(files["evidence.svg"]),
            "svg_sha256": audit.sha256_file(files["evidence.svg"]),
            "png": {
                "path": str(files["evidence.png"]),
                "sha256": audit.sha256_file(files["evidence.png"]),
            },
        },
        "sites": [{
            "site": site,
            "actual": {
                "bambu_studio_pause_marker_z_mm": 5.96,
                "last_completely_open_layer_z_mm": 5.80,
                "cavity_bury_roof_start_plane_z_mm": 5.80,
            },
            "seated_magnet": {
                "below_last_open_layer_mm": 0.10,
                "below_first_closing_layer_mm": 0.26,
            },
        }],
    }


def test_complete_release_gate_rejects_partial_or_failed_coverage(
        tmp_path: Path) -> None:
    raw = {
        "id": "state:test:part", "part": "part", "variant": "test",
        "state": "state", "stl": "part.stl",
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": 0.0},
        "source_to_stl_matrix": [
            [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "sites": [_minimal_catalog_site()],
    }
    payload = _catalog_document([raw])
    _write_bound_stl_and_sidecar(tmp_path, raw)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    catalog = audit.normalize_catalog(
        catalog_path, enforce_release_inventory=False)
    artifact = catalog["artifacts"][0]
    record = _minimal_passing_release_record(tmp_path, artifact)
    audit._validate_complete_release(
        catalog, [record], enforce_expected_inventory=False)
    missing_3mf = {
        **record,
        "slicer": {
            key: value for key, value in record["slicer"].items()
            if not key.startswith("project_3mf")
        },
    }
    try:
        audit._validate_complete_release(
            catalog, [missing_3mf], enforce_expected_inventory=False)
    except audit.AuditError as exc:
        assert "audited Bambu 3MF path is missing" in str(exc)
    else:
        raise AssertionError("canonical release omitted its arranged Bambu 3MF")

    project = Path(record["slicer"]["project_3mf"])
    original_project = project.read_bytes()
    project.write_bytes(b"tampered")
    try:
        audit._validate_complete_release(
            catalog, [record], enforce_expected_inventory=False)
    except audit.AuditError as exc:
        assert "audited Bambu 3MF file differs" in str(exc)
    else:
        raise AssertionError("canonical release accepted a changed Bambu 3MF")
    project.write_bytes(original_project)
    for records, failures, expected in (
            ([], [], "not exact"),
            ([{**record, "status": "fail"}], [], "did not pass"),
            ([record], [{"id": record["id"], "error": "boom"}],
             "slice exception")):
        try:
            audit._validate_complete_release(
                catalog, records, failures,
                enforce_expected_inventory=False)
        except audit.AuditError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("incomplete/failed release became canonical")


def test_manifest_views_include_hash_bound_bambu_arrangement(
        tmp_path: Path) -> None:
    raw = {
        "id": "state:test:part", "part": "part", "variant": "test",
        "state": "state", "stl": "part.stl",
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": 0.0},
        "source_to_stl_matrix": [
            [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "sites": [_minimal_catalog_site()],
    }
    payload = _catalog_document([raw])
    _write_bound_stl_and_sidecar(tmp_path, raw)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    catalog = audit.normalize_catalog(
        catalog_path, enforce_release_inventory=False)
    catalog["_catalog_sha256"] = audit.sha256_file(catalog_path)
    record = _minimal_passing_release_record(tmp_path, catalog["artifacts"][0])
    record["slicer"]["bambu_3mf_audit"] = {
        "rigid_rz": {"rz_degrees": -37.25},
    }
    paths = audit._write_manifest_bundle(
        tmp_path / "manifest", catalog_path, catalog,
        {"identity": {
            "profile_set_sha256": "a" * 64,
            "binary_sha256": "b" * 64,
        }}, [record])
    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "audited_bambu_3mf" in csv_text
    assert "bambu_arrange_rz_degrees" in csv_text
    assert str(record["slicer"]["project_3mf"]) in csv_text
    assert "-37.25" in csv_text
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "## Audited Bambu arrangements" in markdown
    assert str(record["slicer"]["project_3mf"]) in markdown
    assert record["slicer"]["project_3mf_sha256"] in markdown
    assert "-37.250000 deg" in markdown


def test_manifest_transaction_rolls_back_all_canonical_files(
        tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    destination = tmp_path / "published"
    stage.mkdir()
    destination.mkdir()
    paths = {
        "json": stage / audit.CANONICAL_MANIFEST_FILENAMES[0],
        "csv": stage / audit.CANONICAL_MANIFEST_FILENAMES[1],
        "markdown": stage / audit.CANONICAL_MANIFEST_FILENAMES[2],
    }
    for key, path in paths.items():
        path.write_text(f"new-{key}\n", encoding="utf-8")
    old = {}
    for path in paths.values():
        target = destination / path.name
        target.write_text(f"old-{path.name}\n", encoding="utf-8")
        old[path.name] = target.read_bytes()
    real_replace = audit.os.replace
    calls = 0

    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == 5:  # three backups, first install, then fail
            raise OSError("synthetic publication failure")
        return real_replace(source, target)

    audit.os.replace = fail_once
    try:
        try:
            audit._transactional_publish_bundle(paths, destination)
        except audit.AuditError as exc:
            assert "transaction failed" in str(exc)
        else:
            raise AssertionError("synthetic transaction failure was ignored")
    finally:
        audit.os.replace = real_replace
    for name, payload in old.items():
        assert (destination / name).read_bytes() == payload


def test_gcode_skill_validation_must_explicitly_pass() -> None:
    source = inspect.getsource(audit._slice_one)
    assert 'skill_validation.get("ok") is not True' in source


def test_generator_style_source_matrix_is_consumed(tmp_path: Path):
    # This is the exact structural shape emitted by
    # generate_captive_magnet_catalog.py: source-space helper facts plus one
    # artifact-level source-to-STL matrix, not nested print-space facts.
    catalog = _catalog_document([{
            "id": "state:V1LF:part", "part": "part", "variant": "V1LF",
            "state": "state", "stl": "part.stl",
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 90.0},
            "source_to_stl_matrix": [
                [0.0, 1.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, -1.0, 18.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "sites": [{
                **_release_site_contract(),
                "name": "lm", "closure_kind": "transverse_gable_45deg",
                "cavity_bury_roof_start_print_z_mm": 8.4,
                "roof_apex_print_z_mm": 11.0,
                "cavity_center_xyz_mm": [1.0, 2.0, 12.55],
                "seated_magnet_center_xyz_mm": [1.1, 2.0, 12.55],
                "actual_face_xyz_mm": [2.0, 2.0, 12.55],
                "material_inward_xyz": [-1.0, 0.0, 0.0],
                "marked_pole_axis_xyz": [1.0, 0.0, 0.0],
            }],
        }])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    normalized = audit.normalize_catalog(
        path, enforce_release_inventory=False)
    site = normalized["artifacts"][0]["sites"][0]
    assert site["print_cavity_center_xyz_mm"] == (12.0, 21.0, 5.75)
    assert site["print_seated_magnet_center_xyz_mm"] == (12.0, 21.1, 5.75)
    assert site["print_actual_face_xyz_mm"] == (12.0, 22.0, 5.75)
    assert site["print_material_inward_xyz"] == (0.0, -1.0, 0.0)
    assert site["print_marked_pole_axis_xyz"] == (0.0, 1.0, 0.0)
    assert site["print_insertion_direction_xyz"] == (0.0, 0.0, -1.0)


def test_catalog_rejects_non_downward_print_insertion_direction(
        tmp_path: Path) -> None:
    raw = {
        "id": "state:test:unsafe-insertion",
        "part": "unsafe-insertion",
        "variant": "test",
        "state": "state",
        "stl": "unsafe-insertion.stl",
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": 0.0},
        "source_to_stl_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 18.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "sites": [_minimal_catalog_site()],
    }
    # This source-space direction transforms into print +X, not the only
    # approved physical loading motion (vertical print -Z from above).
    raw["sites"][0]["insertion_direction_xyz"] = [1.0, 0.0, 0.0]
    catalog = _catalog_document([raw])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    try:
        audit.normalize_catalog(path, enforce_release_inventory=False)
    except audit.AuditError as exc:
        assert "insertion direction must be print [0, 0, -1]" in str(exc)
    else:
        raise AssertionError("non-downward print insertion was accepted")


def test_explicit_print_space_must_match_source_matrix(tmp_path: Path):
    catalog = _catalog_document([{
            "id": "state:V0:part", "part": "part", "variant": "V0",
            "state": "state", "stl": "part.stl",
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 0.0},
            "source_to_stl_matrix": [
                [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 10.0],
                [0.0, 0.0, -1.0, 18.3], [0.0, 0.0, 0.0, 1.0]],
            "sites": [{
                **_release_site_contract((0.0, 0.0, -1.0)),
                "name": "v0", "closure_kind": "axis_opposed_conical_45deg",
                "cavity_bury_roof_start_print_z_mm": 15.25,
                "roof_apex_print_z_mm": 17.85,
                "cavity_center_xyz_mm": [2.0, 3.0, 4.1],
                "seated_magnet_center_xyz_mm": [2.0, 3.0, 4.1],
                "marked_pole_axis_xyz": [0.0, 0.0, -1.0],
                "print_space": {
                    "cavity_center_xyz_mm": [999.0, 7.0, 14.2],
                    "seated_magnet_center_xyz_mm": [2.0, 7.0, 14.2],
                    "marked_pole_axis_xyz": [0.0, 0.0, 1.0],
                    "insertion_direction_xyz": [0.0, 0.0, -1.0]
                }
            }]
        }])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    try:
        audit.normalize_catalog(path, enforce_release_inventory=False)
    except audit.AuditError as exc:
        assert "disagrees" in str(exc)
    else:
        raise AssertionError("inconsistent post-export print-space facts passed")


def test_seated_magnet_bounds_cover_transverse_and_axial_discs():
    transverse = {
        "print_seated_magnet_center_xyz_mm": (0.0, 0.0, 8.0),
        "print_marked_pole_axis_xyz": (1.0, 0.0, 0.0),
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
    }
    axial = {
        "print_seated_magnet_center_xyz_mm": (0.0, 0.0, 8.0),
        "print_marked_pole_axis_xyz": (0.0, 0.0, -1.0),
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
    }
    assert audit._seated_magnet_print_z_bounds(transverse) == (5.5, 10.5)
    assert audit._seated_magnet_print_z_bounds(axial) == (7.0, 9.0)


def test_transverse_retaining_gate_requires_continuous_paths():
    site = {
        "closure_kind": "transverse_gable_45deg",
        "print_cavity_center_xyz_mm": (0.0, 0.0, 3.0),
        "print_actual_face_xyz_mm": (0.0, 0.0, 3.0),
        "print_material_inward_xyz": (1.0, 0.0, 0.0),
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "face_skin_mm": 0.45,
        "inner_skin_mm": 0.45,
    }

    def segment(x: float, y0: float, y1: float) -> audit.Segment:
        return audit.Segment(
            x, y0, x, y1, 0.1, "Outer wall", 0.42, 1)

    interface_x = 0.45 / 2.0
    inner_x = 0.45 + 2.1 + 0.45 / 2.0
    fragments = []
    for x in (interface_x, inner_x):
        fragments.extend((
            segment(x, -2.55, -2.0),
            segment(x, 2.0, 2.55),
        ))
    disconnected = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, fragments, 1), site, (0.0, 0.0))
    assert disconnected["retaining_paths"][
        "interface_skin_transverse_span_mm"] > 4.0
    assert disconnected["retaining_paths"][
        "interface_skin_longest_contiguous_span_mm"] < 1.0
    assert disconnected["retaining_paths"]["pass"] is False

    continuous = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1),
            segment(inner_x, -2.1, 2.1),
        ], 1), site, (0.0, 0.0))
    assert continuous["retaining_paths"][
        "interface_skin_longest_contiguous_span_mm"] >= 3.0
    assert continuous["retaining_paths"]["pass"] is True

    # Projecting only onto V would incorrectly merge these two overlapping
    # halves.  Their U separation exceeds one 0.42-mm Classic bead, so they
    # are physically disconnected and neither component spans 3 mm.
    staggered = []
    for center_x in (interface_x, inner_x):
        staggered.extend((
            segment(center_x - 0.33, -2.1, 0.0),
            segment(center_x + 0.33, 0.0, 2.1),
        ))
    disconnected_in_2d = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, staggered, 1), site, (0.0, 0.0))
    assert disconnected_in_2d["retaining_paths"][
        "interface_skin_transverse_span_mm"] > 4.0
    assert disconnected_in_2d["retaining_paths"][
        "interface_skin_longest_contiguous_span_mm"] < 3.0
    assert disconnected_in_2d["retaining_paths"]["pass"] is False


def test_axial_retaining_gate_requires_complete_annular_coverage():
    radius = 2.6
    complete = [math.radians(value) for value in range(0, 360, 5)]
    fragmented = [
        math.radians(value)
        for value in (*range(0, 120, 5), *range(240, 360, 5))
    ]
    assert audit._largest_circular_sample_gap(complete, radius) < 0.25
    assert audit._largest_circular_sample_gap(
        fragmented, radius) > 5.0
    assert audit._largest_circular_sample_gap([], radius) == math.inf


def test_last_open_loading_aperture_checks_diameter_slot_and_obstruction():
    site = {
        "closure_kind": "transverse_gable_45deg",
        "print_cavity_center_xyz_mm": (0.0, 0.0, 3.0),
        "print_actual_face_xyz_mm": (0.0, 0.0, 3.0),
        "print_material_inward_xyz": (1.0, 0.0, 0.0),
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "face_skin_mm": 0.45,
        "inner_skin_mm": 0.45,
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
    }

    def line(x0, y0, x1, y1):
        return audit.Segment(
            x0, y0, x1, y1, 0.1, "Outer wall", 0.42, 1)

    def metrics(interface_x=0.225, inner_x=2.775, side_v=2.81,
                obstruction=False):
        segments = [
            line(interface_x, -2.81, interface_x, 2.81),
            line(inner_x, -2.81, inner_x, 2.81),
            line(0.45, -side_v, 2.55, -side_v),
            line(0.45, side_v, 2.55, side_v),
        ]
        if obstruction:
            segments.append(line(0.60, 0.0, 2.40, 0.0))
        return audit._toolpath_metrics(
            audit.Layer(3.0, 0.16, segments, 1), site, (0.0, 0.0))

    clear = metrics()
    assert clear["loading_aperture"]["free_transverse_diameter_mm"] >= 5.0
    assert clear["loading_aperture"]["free_axial_slot_width_mm"] >= 2.0
    assert audit._loading_aperture_pass(site, clear)[0] is True

    narrow_diameter = metrics(side_v=2.55)
    assert audit._loading_aperture_pass(site, narrow_diameter)[0] is False

    narrow_slot = metrics(interface_x=0.40, inner_x=2.60)
    assert narrow_slot["loading_aperture"]["free_axial_slot_width_mm"] < 2.0
    assert audit._loading_aperture_pass(site, narrow_slot)[0] is False

    obstructed = metrics(obstruction=True)
    assert obstructed["roof_interior_path_length_mm"] > 0.20
    assert audit._loading_aperture_pass(site, obstructed)[0] is False


def test_failed_metrics_remain_strict_finite_json(tmp_path: Path):
    site = {
        "closure_kind": "axis_opposed_conical_45deg",
        "print_cavity_center_xyz_mm": (0.0, 0.0, 3.0),
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
    }
    metrics = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [], 1), site, (0.0, 0.0))
    assert metrics["retaining_paths"]["largest_uncovered_arc_mm"] is None
    output = tmp_path / "metrics.json"
    audit._write_json(output, metrics)
    assert json.loads(output.read_text(encoding="utf-8"))[
        "retaining_paths"]["largest_uncovered_arc_mm"] is None
    try:
        audit._write_json(tmp_path / "bad.json", {"bad": math.inf})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite audit JSON was accepted")


def test_manifest_part_name_does_not_alias_artifact_id(tmp_path: Path):
    catalog = _catalog_document([{
            "id": "state:V0:descriptive-id",
            "part": "actual-print-part",
            "variant": "V0", "state": "state", "stl": "part.stl",
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 0.0},
            "source_to_stl_matrix": [
                [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 10.0],
                [0.0, 0.0, -1.0, 18.3], [0.0, 0.0, 0.0, 1.0]],
            "sites": [{
                **_release_site_contract((0.0, 0.0, -1.0)),
                "name": "v0", "closure_kind": "axis_opposed_conical_45deg",
                "cavity_bury_roof_start_print_z_mm": 15.25,
                "roof_apex_print_z_mm": 17.85,
                "cavity_center_xyz_mm": [2.0, 3.0, 4.1],
                "seated_magnet_center_xyz_mm": [2.0, 3.0, 4.1],
                "marked_pole_axis_xyz": [0.0, 0.0, -1.0],
            }],
        }])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    artifact = audit.normalize_catalog(
        path, enforce_release_inventory=False)["artifacts"][0]
    assert artifact["id"] == "state:V0:descriptive-id"
    assert artifact["part"] == "actual-print-part"


def _exact_split_proxy_catalog() -> dict:
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 20.0],
        [0.0, 0.0, -1.0, 18.3],
        [0.0, 0.0, 0.0, 1.0],
    ]

    def site(name: str, x: float) -> dict:
        return {
            "name": name,
            "closure_kind": "transverse_gable_45deg",
            "cavity_bury_roof_start_print_z_mm": 8.40,
            "roof_apex_print_z_mm": 11.0,
            "cavity_center_xyz_mm": [x, 10.0, 12.55],
            "seated_magnet_center_xyz_mm": [x + 0.05, 10.0, 12.55],
            "actual_face_xyz_mm": [x - 1.5, 10.0, 12.55],
            "material_inward_xyz": [1.0, 0.0, 0.0],
            "marked_pole_axis_xyz": [-1.0, 0.0, 0.0],
            "insertion_direction_xyz": [0.0, 0.0, 1.0],
            "installed_marked_pole_axis_xyz": [-1.0, 0.0, 0.0],
            "magnet_diameter_mm": 5.0,
            "magnet_depth_mm": 2.0,
            "cavity_diameter_mm": 5.2,
            "cavity_depth_mm": 2.1,
            "face_skin_mm": 0.45,
            "inner_skin_mm": 0.45,
            "roof_angle_deg": 45.0,
            "polarity_instruction": "marked/N pole follows installed axis",
            "captive_land_mm": 3.0,
            "interface_gap_mm": 0.05,
            "paired_magnet_face_separation_mm": 0.95,
            "classic_retaining_path_mm": 0.42,
            "magnet_count": 1,
            "structural_load_credit_n": 0.0,
        }

    def artifact(artifact_id: str, variant: str, part: str,
                 sites: list[dict]) -> dict:
        return {
            "id": artifact_id, "state": "floor_stand",
            "variant": variant, "part": part, "stl": f"{part}.stl",
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 0.0},
            "source_to_stl_matrix": matrix,
            "sites": json.loads(json.dumps(sites)),
        }

    lower = site("lm_lower_left", -32.0)
    upper = site("lm_upper_left", -75.0)
    monolith = artifact("floor:V1LF:mono", "V1LF", "mono", [lower, upper])
    monolith["p2s_printability"] = "not_printable_oversize"
    monolith["cavity_audit_proxies"] = [
        {"site": "lm_lower_left", "artifact_id": "floor:split:bottom",
         "proxy_site": "lm_lower_left"},
        {"site": "lm_upper_left", "artifact_id": "floor:split:top",
         "proxy_site": "lm_upper_left"},
    ]
    return _catalog_document([
        monolith,
        artifact("floor:split:bottom", "V1LF-split", "bottom", [lower]),
        artifact("floor:split:top", "V1LF-split", "top", [upper]),
    ])


def test_catalog_envelope_and_frozen_inventory_are_fail_closed(
        tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    missing = _exact_split_proxy_catalog()
    missing.pop("geometry")
    path.write_text(json.dumps(missing), encoding="utf-8")
    try:
        audit.normalize_catalog(path, enforce_release_inventory=False)
    except audit.AuditError as exc:
        assert "required root fields" in str(exc)
    else:
        raise AssertionError("catalog missing its geometry authority passed")

    truncated = _exact_split_proxy_catalog()
    path.write_text(json.dumps(truncated), encoding="utf-8")
    try:
        audit.normalize_catalog(path)
    except audit.AuditError as exc:
        assert "56 artifacts / 102 captive stations" in str(exc)
    else:
        raise AssertionError("truncated production inventory passed")


def test_catalog_artifact_hash_bindings_are_enforced(tmp_path: Path) -> None:
    payload = _exact_split_proxy_catalog()
    raw = payload["artifacts"][1]
    _write_bound_stl_and_sidecar(tmp_path, raw)
    stl = tmp_path / raw["stl"]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = audit.normalize_catalog(
        path, enforce_release_inventory=False)["artifacts"][1]
    audit._validate_artifact_bindings(artifact)
    stl.write_bytes(b"tampered")
    try:
        audit._validate_artifact_bindings(artifact)
    except audit.AuditError as exc:
        assert "STL hash differs" in str(exc)
    else:
        raise AssertionError("post-catalog STL tampering passed")


def test_catalog_schema_rejects_invalid_generated_by_and_exclusion(
        tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    invalid_cases = (
        ("generated_by", "minLength", lambda payload: payload.__setitem__(
            "generated_by", "")),
        ("exclusions", "minLength", lambda payload: payload["exclusions"][0].__setitem__(
            "reason", "")),
        ("source_revision", "expected type string", lambda payload: payload.__setitem__(
            "source_revision", None)),
        ("source_file_sha256", "missing required", lambda payload: payload["artifacts"][0].pop(
            "source_file_sha256")),
    )
    for expected, detail, mutate in invalid_cases:
        payload = _exact_split_proxy_catalog()
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            audit.normalize_catalog(path, enforce_release_inventory=False)
        except audit.AuditError as exc:
            assert "violates its JSON schema" in str(exc)
            assert expected in str(exc)
            assert detail in str(exc)
        else:
            raise AssertionError(
                f"schema-invalid {expected} metadata passed")


def test_wing_facts_and_transaction_manifest_are_hash_bound(
        tmp_path: Path) -> None:
    raw = {
        "id": "shared:V1LF-Ac:wing", "part": "wing",
        "variant": "V1LF-Ac", "state": "shared", "stl": "wing.stl",
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": 0.0},
        "source_to_stl_matrix": [
            [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "sites": [_minimal_catalog_site()],
    }
    payload = _catalog_document([raw])
    _write_bound_stl_and_sidecar(tmp_path, raw)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = audit.normalize_catalog(
        catalog_path, enforce_release_inventory=False)["artifacts"][0]
    audit._validate_artifact_bindings(artifact)
    artifact["facts"].write_text("tampered\n", encoding="utf-8")
    try:
        audit._validate_artifact_bindings(artifact)
    except audit.AuditError as exc:
        assert "facts hash differs" in str(exc)
    else:
        raise AssertionError("tampered Ac/Ae facts remained release-bound")


def test_catalog_rejects_nonadjacent_print_sidecar(tmp_path: Path) -> None:
    payload = _exact_split_proxy_catalog()
    raw = payload["artifacts"][1]
    _write_bound_stl_and_sidecar(
        tmp_path, raw, subdirectory="nonadjacent")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = audit.normalize_catalog(
        path, enforce_release_inventory=False)["artifacts"][1]
    try:
        audit._validate_artifact_bindings(artifact)
    except audit.AuditError as exc:
        assert "must be adjacent" in str(exc)
    else:
        raise AssertionError("nonadjacent print sidecar passed")


def test_catalog_rejects_wrong_sidecar_identity(tmp_path: Path) -> None:
    payload = _exact_split_proxy_catalog()
    raw = payload["artifacts"][1]
    authority = _write_bound_stl_and_sidecar(tmp_path, raw)
    sidecar = json.loads(authority.read_text(encoding="utf-8"))
    sidecar["part"] = "wrong_identity"
    authority.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    raw["print_sidecar_sha256"] = audit.sha256_file(authority)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = audit.normalize_catalog(
        path, enforce_release_inventory=False)["artifacts"][1]
    try:
        audit._validate_artifact_bindings(artifact)
    except audit.AuditError as exc:
        assert "does not match STL stem" in str(exc)
    else:
        raise AssertionError("wrong sidecar part identity passed")


def test_catalog_rejects_sidecar_transform_disagreement(
        tmp_path: Path) -> None:
    payload = _exact_split_proxy_catalog()
    raw = payload["artifacts"][1]
    _write_bound_stl_and_sidecar(tmp_path, raw)
    stl = tmp_path / raw["stl"]
    matrix = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 20.0],
        [0.0, 0.0, -1.0, 18.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    authority = write_print_sidecar(
        stl,
        part=stl.stem,
        transform={
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 90.0},
            "source_to_stl_matrix": matrix,
            "pre_translation_bbox_min_mm": [0.0, -20.0, -18.3],
            "stl_origin_translation_mm": [0.0, 20.0, 18.3],
        },
    )
    raw["print_sidecar_sha256"] = audit.sha256_file(authority)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = audit.normalize_catalog(
        path, enforce_release_inventory=False)["artifacts"][1]
    try:
        audit._validate_artifact_bindings(artifact)
    except audit.AuditError as exc:
        assert "sidecar rotation differs from catalog" in str(exc)
    else:
        raise AssertionError("sidecar/catalog transform disagreement passed")


def test_oversize_proxy_contract_is_exact_same_state_and_complete(
        tmp_path: Path):
    payload = _exact_split_proxy_catalog()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    catalog = audit.normalize_catalog(path, enforce_release_inventory=False)
    monolith = next(item for item in catalog["artifacts"]
                    if item["id"] == "floor:V1LF:mono")
    assert monolith["p2s_printability"] == "not_printable_oversize"
    assert len(monolith["cavity_audit_proxies"]) == 2
    assert {item["site"] for item in monolith["cavity_audit_proxies"]} == {
        "lm_lower_left", "lm_upper_left"}
    assert all(len(item["source_contract_sha256"]) == 64
               for item in monolith["cavity_audit_proxies"])

    mismatched = _exact_split_proxy_catalog()
    mismatched["artifacts"][2]["sites"][0][
        "polarity_instruction"] = "opposite and unsafe"
    path.write_text(json.dumps(mismatched), encoding="utf-8")
    try:
        audit.normalize_catalog(path, enforce_release_inventory=False)
    except audit.AuditError as exc:
        assert "source-space cavity contract differs" in str(exc)
    else:
        raise AssertionError("mismatched split polarity contract passed")

    incomplete = _exact_split_proxy_catalog()
    incomplete["artifacts"][0]["cavity_audit_proxies"].pop()
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    try:
        audit.normalize_catalog(path, enforce_release_inventory=False)
    except audit.AuditError as exc:
        assert "not every monolith site" in str(exc)
    else:
        raise AssertionError("incomplete split coverage passed")


def test_oversize_monolith_emits_no_fake_pause_group(tmp_path: Path):
    payload = _exact_split_proxy_catalog()
    raw_monolith = payload["artifacts"][0]
    # A minimal oversize STL is enough for this pure envelope/manifest test;
    # strict mesh validity remains owned by the released STL gate.
    stl_path = tmp_path / raw_monolith["stl"]
    stl_path.write_text("""solid oversize
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 300 0 0
vertex 0 10 1
endloop
endfacet
endsolid oversize
""", encoding="ascii")
    _bind_existing_stl_and_sidecar(tmp_path, raw_monolith)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    catalog = audit.normalize_catalog(
        catalog_path, enforce_release_inventory=False)
    by_id = {item["id"]: item for item in catalog["artifacts"]}
    monolith = by_id["floor:V1LF:mono"]
    def proxy_record(artifact_id: str) -> dict:
        artifact = by_id[artifact_id]
        site = artifact["sites"][0]
        return {
            "id": artifact_id,
            "audit_mode": "actual_p2s_slice",
            "status": "pass",
            "input": {"stl_sha256": "a" * 64},
            "slicer": {"gcode_sha256": "b" * 64},
            "evidence": {
                "svg_sha256": "c" * 64,
                "png": {"sha256": "d" * 64},
            },
            "sites": [{
                "site": site,
                "actual": {"bambu_studio_pause_marker_z_mm": 8.52},
                "retaining_paths_pass": True,
                "loading_aperture_pass": True,
                "seated_magnet": {"clearance_pass": True},
                "insertion_fit": {"pass": True},
                "regression_pass": True,
            }],
        }

    proxy_records = {
        artifact_id: proxy_record(artifact_id)
        for artifact_id in ("floor:split:bottom", "floor:split:top")
    }
    profile = {"identity": {"machine_bounds_mm": {
        "x": [0.0, 256.0], "y": [0.0, 256.0], "z": [0.0, 256.0],
    }}}
    record = audit._oversize_proxy_coverage_record(
        monolith, proxy_records, profile)
    assert record["status"] == audit.OVERSIZE_COVERED_STATUS
    assert record["p2s_printable"] is False
    assert record["p2s_bed_fit"]["pass"] is False
    assert record["cavity_audit_coverage"]["pass"] is True
    assert len(record["cavity_audit_coverage"]["sites"]) == 2
    assert "slicer" not in record and "sites" not in record
    assert audit._pause_groups(record) == []
    assert all(item["proxy_gcode_sha256"] == "b" * 64
               for item in record["cavity_audit_coverage"]["sites"])

    proxy_records["floor:split:top"]["status"] = "fail"
    failed = audit._oversize_proxy_coverage_record(
        monolith, proxy_records, profile)
    assert failed["status"] == "fail"
    assert failed["cavity_audit_coverage"]["pass"] is False
    assert audit._pause_groups(failed) == []


def main() -> None:
    """Run every pure-Python gate without requiring pytest on the worker."""
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        parameters = tuple(inspect.signature(test).parameters)
        if not parameters:
            test()
        elif parameters == ("tmp_path",):
            with tempfile.TemporaryDirectory(
                    prefix=f"{test.__name__}-") as directory:
                test(Path(directory))
        else:
            raise RuntimeError(
                f"unsupported standalone test signature for "
                f"{test.__name__}: {parameters}")
        print(f"PASS {test.__name__}")
    print(f"all {len(tests)} captive-magnet slicing tests passed")


if __name__ == "__main__":
    main()
