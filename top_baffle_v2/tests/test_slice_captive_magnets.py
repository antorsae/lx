"""Pure-Python gates for the offline captive-magnet slicing pipeline."""

from __future__ import annotations

import copy
import json
import inspect
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import tempfile
import zipfile

import slice_captive_magnets as audit
from lx521_baffle.print_contract import (
    RELEASE_ACOUSTIC_PRINT_CONTRACT,
    write_print_sidecar,
)


def _release_site_contract(axis=(1.0, 0.0, 0.0)) -> dict:
    return {
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "face_skin_mm": 0.52,
        "inner_skin_mm": 0.52,
        "captive_land_mm": 3.14,
        "interface_gap_mm": 0.05,
        "paired_magnet_face_separation_mm": 1.09,
        "roof_angle_deg": 45.0,
        "minimum_retaining_path_mm": 0.42,
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
        if artifact.get("variant") in ("Obi-Wan", "Obi-Wan-split"):
            artifact.setdefault("stage_manifest", "stage_manifest.json")
            artifact.setdefault("stage_manifest_sha256", "e" * 64)
        if artifact.get("variant") in ("Obi-Wan-Flat", "Obi-Wan-Graded"):
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
    """Write cavity-local Arachne paths with an explicit roof onset."""
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
        "face_skin_mm": 0.52,
        "inner_skin_mm": 0.52,
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
        "print_cavity_center_xyz_mm": (1.50, 0.0, 3.20),
        "print_seated_magnet_center_xyz_mm": (1.45, 0.0, 3.20),
        "cavity_bury_roof_start_print_z_mm": 5.80,
        "roof_apex_print_z_mm": 8.40,
    }
    um_gcode = tmp_path / "um_synthetic.gcode"
    lm_gcode = tmp_path / "lm_synthetic.gcode"
    _synthetic_gcode(um_gcode, first_closing_z=5.96)
    _synthetic_gcode(lm_gcode, first_closing_z=5.96)
    um_layers, _um_metrics, um_discovery = (
        audit._discover_actual_closure_layers(
            audit.parse_gcode(um_gcode).layers, um, (0.0, 0.0)))
    lm_layers, _lm_metrics, lm_discovery = (
        audit._discover_actual_closure_layers(
            audit.parse_gcode(lm_gcode).layers, lm, (0.0, 0.0)))
    assert math.isclose(um_layers["last_fully_open"].z, 5.80)
    assert math.isclose(um_layers["first_closing_pause"].z, 5.96)
    assert math.isclose(lm_layers["last_fully_open"].z, 5.80)
    assert math.isclose(lm_layers["first_closing_pause"].z, 5.96)
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
        0.10, abs_tol=1.0e-12)


def test_actual_closure_discovery_rejects_early_or_missing_roof(
        tmp_path: Path):
    site = {
        "name": "unsafe",
        "closure_kind": "transverse_gable_45deg",
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
        "face_skin_mm": 0.52,
        "inner_skin_mm": 0.52,
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


def test_support_toolpaths_are_filtered_and_fail_on_duct_collision(
        tmp_path: Path):
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    contract = {
        "schema_version": 1,
        "coordinate_space": "authoritative_source_mm",
        "split_half": "bottom",
        "split_seam_y_mm": 100.0,
        "modifier_clearance_mm": 0.25,
        "regions": [{
            "name": "test_duct",
            "kind": "polyline_tube",
            "radius_mm": 1.0,
            "points_xyz_mm": [[0.0, 0.0, 0.1], [10.0, 0.0, 0.1]],
        }],
    }

    def write(path: Path, support_y: float) -> None:
        path.write_text("\n".join((
            "M83", "M104 S220", "M140 S55", "G90",
            "; CHANGE_LAYER", "; Z_HEIGHT: 0.20", "; LAYER_HEIGHT: 0.20",
            "; FEATURE: Outer wall", "; LINE_WIDTH: 0.42",
            "G1 X0 Y4 Z0.20", "G1 X10 Y4 E0.1",
            "; FEATURE: Support", "; LINE_WIDTH: 0.42",
            f"G1 X0 Y{support_y:g}", f"G1 X10 Y{support_y:g} E0.1",
            "; CONFIG_BLOCK_START", "; layer_height = 0.2",
            "; CONFIG_BLOCK_END",
        )) + "\n", encoding="utf-8")

    clear = tmp_path / "support_clear.gcode"
    write(clear, 2.0)
    filtered = audit.parse_gcode(
        clear, retain_feature_prefixes=("support",))
    assert {
        segment.feature for layer in filtered.layers
        for segment in layer.segments
    } == {"Support"}
    result = audit.audit_support_toolpaths_vs_ducts(
        gcode=clear, contract=contract,
        source_to_stl_matrix=identity, stl_to_bed_matrix=identity)
    assert result["status"] == "pass"
    assert result["collision_count"] == 0
    assert result["support_extrusion_segments_checked"] == 1

    collision = tmp_path / "support_collision.gcode"
    write(collision, 0.0)
    try:
        audit.audit_support_toolpaths_vs_ducts(
            gcode=collision, contract=contract,
            source_to_stl_matrix=identity, stl_to_bed_matrix=identity)
    except audit.AuditError as exc:
        assert "support toolpath enters a cable duct" in str(exc)
    else:
        raise AssertionError("support extrusion inside a duct passed")


def test_segment_distance_clamps_degenerate_and_finite_segments() -> None:
    cases = (
        (
            (0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
            (12.0, 3.0, 0.0), (12.0, 3.0, 0.0),
            math.sqrt(13.0),
        ),
        (
            (12.0, 3.0, 0.0), (12.0, 3.0, 0.0),
            (0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
            math.sqrt(13.0),
        ),
        (
            (0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
            (5.0, -3.0, 4.0), (5.0, 3.0, 4.0),
            4.0,
        ),
        (
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0), (1.0, 2.0, 0.0),
            2.0,
        ),
    )
    for first_start, first_end, second_start, second_end, expected in cases:
        actual = audit._segment_distance_3d(
            first_start, first_end, second_start, second_end)
        assert math.isclose(actual, expected, abs_tol=1.0e-12)


def test_duct_collision_contract_requires_complete_state_inventory() -> None:
    regions = [{
        "name": name,
        "kind": "polyline_tube",
        "radius_mm": 1.0,
        "points_xyz_mm": [[0.0, 0.0, 0.0]],
    } for name in sorted(
        audit.EXPECTED_KEYED_LM_DUCT_REGION_NAMES["no_floor_stand"])]
    contract = {
        "schema_version": 1,
        "coordinate_space": "authoritative_source_mm",
        "split_half": "top",
        "split_seam_y_mm": audit.EXPECTED_LM_SPLIT_SEAM_Y_MM,
        "modifier_clearance_mm": 0.25,
        "regions": regions,
    }
    normalized = audit._normalize_duct_collision_contract(
        contract,
        artifact_id="no-floor:Obi-Wan-split:"
        "obiwan_optional_lm_keyed_2_of_2_top",
        state="no_floor_stand",
        variant="Obi-Wan-split",
        part="obiwan_optional_lm_keyed_2_of_2_top",
        modifier_clearance_mm=0.25,
    )
    assert {
        region["name"] for region in normalized["regions"]
    } == audit.EXPECTED_KEYED_LM_DUCT_REGION_NAMES["no_floor_stand"]

    for changed, expected in (
            ({**contract, "regions": regions[:-1]}, "inventory is incomplete"),
            ({
                **contract,
                "split_seam_y_mm":
                audit.EXPECTED_LM_SPLIT_SEAM_Y_MM + 0.001,
            }, "differs from R6F authority")):
        try:
            audit._normalize_duct_collision_contract(
                changed,
                artifact_id="no-floor:Obi-Wan-split:"
                "obiwan_optional_lm_keyed_2_of_2_top",
                state="no_floor_stand",
                variant="Obi-Wan-split",
                part="obiwan_optional_lm_keyed_2_of_2_top",
                modifier_clearance_mm=0.25,
            )
        except audit.AuditError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(
                "incomplete duct collision contract passed")


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


def test_profile_nil_vector_slots_inherit_parent_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "BBL"
    root.mkdir()
    (root / "base.json").write_text(json.dumps({
        "name": "base",
        "type": "filament",
        "temperature": ["255", "265"],
        "flow": ["0.95", "0.98"],
    }), encoding="utf-8")
    child = root / "child.json"
    child.write_text(json.dumps({
        "name": "child",
        "type": "filament",
        "inherits": "base",
        "temperature": ["260", "nil"],
        "flow": ["0.93", "nil"],
    }), encoding="utf-8")
    resolved = audit.PresetResolver(root).resolve(child)
    assert resolved["temperature"] == ["260", "265"]
    assert resolved["flow"] == ["0.93", "0.98"]


def test_petg_gf_profile_is_scoped_to_structural_core_only() -> None:
    # TINMORRY PETG-GF is exclusive to the 0.6-mm high-flow lane; the former
    # 0.4-mm PETG profile is retired and must not resurface.
    assert not (
        PROJECT_ROOT / "captive_magnet_slicing_profile_petg_gf.json"
    ).exists()
    config = audit._load_json(
        PROJECT_ROOT / "captive_magnet_slicing_profile_petg_gf_06hf.json")
    assert config["user_filament_preset"] == (
        "TINMORRY PETG-GF Profile @BBL P2S")
    assert config["requirements"]["nozzle_diameter_mm"] == 0.6
    assert config["repo_overrides"]["process"]["wall_loops"] == "6"
    assert config["user_filament_preset_sha256"] == (
        "2fe46f552422a202b221743c3a6a913243e375149fe080598ffd9b084a8b0346")

    scope = config["artifact_scope"]
    assert len(scope) == 6
    assert {
        (match["state"], match["variant"], match["part"])
        for match in scope
    } == {
        (state, variant, part)
        for state in ("floor_stand", "no_floor_stand")
        for variant, part in (
            (
                "Obi-Wan-split",
                "obiwan_optional_lm_keyed_1_of_2_bottom",
            ),
            (
                "Obi-Wan-split",
                "obiwan_optional_lm_keyed_2_of_2_top",
            ),
            (
                "Obi-Wan",
                "obiwan_core_2_of_2_um_carrier",
            ),
        )
    }
    audit._validate_profile_artifact_scope(
        [
            {
                "id": f"core-{index}",
                **match,
            }
            for index, match in enumerate(scope)
        ],
        config,
    )

    wing = {
        "id": "shared:Obi-Wan-Graded:obiwan_wing_graded_left_1_of_3_lm_lower",
        "state": "shared",
        "variant": "Obi-Wan-Graded",
        "part": "obiwan_wing_graded_left_1_of_3_lm_lower",
    }
    try:
        audit._validate_profile_artifact_scope([wing], config)
    except audit.AuditError as exc:
        assert "not authorized" in str(exc)
    else:
        raise AssertionError("PETG-GF structural profile accepted an graded wing")

    modifiers = config["parameter_modifiers"]
    assert len(modifiers) == 1
    assert modifiers[0]["match"] == {
        "state": "no_floor_stand",
        "variant": "Obi-Wan-split",
        "part": "obiwan_optional_lm_keyed_1_of_2_bottom",
    }


def _synthetic_profile_bundle(tmp_path: Path | None = None) -> dict:
    config = audit._load_json(audit.DEFAULT_PROFILE)
    flattened = {
        "machine": {
            "name": "machine",
            "printer_model": "Bambu Lab P2S",
            "nozzle_diameter": ["0.4"],
            "machine_pause_gcode": "wrong",
            "machine_max_speed_z": ["20"],
        },
        "process": {
            "name": "process",
            "layer_height": "0.16",
            "initial_layer_print_height": "0.2",
            "outer_wall_line_width": "0.42",
            "inner_wall_line_width": "0.45",
            "wall_generator": "classic",
            "enable_support": "1",
            "support_on_build_plate_only": "1",
            "support_critical_regions_only": "1",
            "support_remove_small_overhang": "1",
            "enable_arc_fitting": "0",
            "wall_loops": "2",
            "top_shell_layers": "2",
            "bottom_shell_layers": "2",
            "outer_wall_speed": ["200", "200"],
            "curr_bed_type": "Cool Plate",
            "sparse_infill_pattern": "grid",
            "sparse_infill_density": "15%",
            "precise_outer_wall": "0",
            "detect_thin_wall": "0",
            "ensure_vertical_shell_thickness": "disabled",
            "detect_narrow_internal_solid_infill": "0",
            "elefant_foot_compensation": "0",
            "xy_hole_compensation": "0",
        },
        "filament": {
            "name": "Bambu PLA Basic @BBL P2S",
            "nozzle_temperature": ["245", "245"],
            "nozzle_temperature_initial_layer": ["245", "245"],
            "fan_max_speed": ["100"],
            "overhang_fan_speed": ["50"],
            "filament_max_volumetric_speed": ["21", "21"],
            "textured_plate_temp": ["60"],
            "textured_plate_temp_initial_layer": ["60"],
            "eng_plate_temp": ["55"],
        },
    }
    resolved = audit._apply_profile_overrides(
        flattened, config["repo_overrides"], label="test repo overrides")
    effective = audit._effective_profile_contract(
        resolved, config, config["repo_overrides"])
    return {
        "config": config,
        "resolved": resolved,
        "enforced_overrides": config["repo_overrides"],
        "paths": {},
        "identity": {
            "effective": effective,
            "machine_bounds_mm": config["machine_bounds_mm"],
            "binary_sha256": "0" * 64,
            "profile_set_sha256": "1" * 64,
        },
        "audit_source_sha256": {},
    }


def test_repo_overrides_apply_after_flattening_and_are_exact() -> None:
    bundle = _synthetic_profile_bundle()
    process = bundle["resolved"]["process"]
    filament = bundle["resolved"]["filament"]
    assert process["wall_loops"] == "6"
    assert process["top_shell_layers"] == "6"
    assert process["bottom_shell_layers"] == "5"
    assert process["outer_wall_speed"] == ["60", "60"]
    assert process["curr_bed_type"] == "Textured PEI Plate"
    assert process["sparse_infill_pattern"] == "gyroid"
    assert process["sparse_infill_density"] == "30%"
    assert process["wall_generator"] == "arachne"
    assert process["min_bead_width"] == "100%"
    assert process["enable_support"] == "0"
    assert process["support_on_build_plate_only"] == "0"
    assert process["support_critical_regions_only"] == "0"
    assert process["support_remove_small_overhang"] == "0"
    assert process["precise_outer_wall"] == "1"
    assert process["detect_thin_wall"] == "1"
    assert process["ensure_vertical_shell_thickness"] == "enabled"
    assert process["detect_narrow_internal_solid_infill"] == "1"
    assert process["elefant_foot_compensation"] == "0.15"
    assert process["xy_hole_compensation"] == "0"
    assert filament["nozzle_temperature"] == ["245", "245"]
    assert filament["fan_max_speed"] == ["100"]
    assert filament["overhang_fan_speed"] == ["100"]
    assert filament["filament_max_volumetric_speed"] == ["21", "21"]
    assert filament["textured_plate_temp"] == ["60"]
    assert filament["textured_plate_temp_initial_layer"] == ["60"]
    assert bundle["resolved"]["machine"]["machine_pause_gcode"] == "M400 U1"
    assert bundle["identity"]["effective"]["detect_thin_wall"] is True
    assert bundle["identity"]["effective"]["support_enabled"] is False
    assert bundle["identity"]["effective"][
        "support_on_build_plate_only"] is False
    assert bundle["identity"]["effective"][
        "support_critical_regions_only"] is False
    assert bundle["identity"]["effective"][
        "support_remove_small_overhang"] is False


def test_profile_override_typo_fails_closed() -> None:
    bundle = _synthetic_profile_bundle()
    typo = {"process": {"detect_thin_wal": "1"}}
    try:
        audit._apply_profile_overrides(
            bundle["resolved"], typo, label="synthetic typo")
    except audit.AuditError as exc:
        assert "not a registered" in str(exc)
    else:
        raise AssertionError("unknown Bambu override key was accepted")


def test_artifact_density_overrides_are_exact(tmp_path: Path) -> None:
    bundle = _synthetic_profile_bundle()
    base = {
        "id": "synthetic",
        "variant": "Obi-Wan-split",
        "part": "obiwan_optional_lm_keyed_1_of_2_bottom",
    }
    floor = audit._artifact_profile_bundle(
        {**base, "state": "floor_stand"}, bundle, tmp_path / "floor")
    no_floor = audit._artifact_profile_bundle(
        {**base, "state": "no_floor_stand"}, bundle,
        tmp_path / "no_floor")
    fallback = audit._artifact_profile_bundle(
        {**base, "state": "another_state"}, bundle,
        tmp_path / "fallback")
    assert floor["resolved"]["process"]["sparse_infill_density"] == "100%"
    assert floor["resolved"]["process"][
        "sparse_infill_pattern"] == "zig-zag"
    assert no_floor["resolved"]["process"][
        "sparse_infill_density"] == "40%"
    assert no_floor["resolved"]["process"][
        "sparse_infill_pattern"] == "gyroid"
    assert fallback["resolved"]["process"][
        "sparse_infill_density"] == "30%"
    assert floor["identity"]["effective"][
        "sparse_infill_density_percent"] == 100.0
    assert floor["identity"]["effective"][
        "sparse_infill_pattern"] == "zig-zag"
    assert no_floor["identity"]["effective"][
        "sparse_infill_density_percent"] == 40.0
    for supported in (floor, no_floor):
        process = supported["resolved"]["process"]
        assert process["enable_support"] == "1"
        assert process["support_on_build_plate_only"] == "1"
        assert process["support_critical_regions_only"] == "1"
        assert process["support_remove_small_overhang"] == "1"
        effective = supported["identity"]["effective"]
        assert effective["support_enabled"] is True
        assert effective["support_on_build_plate_only"] is True
        assert effective["support_critical_regions_only"] is True
        assert effective["support_remove_small_overhang"] is True
    assert fallback["identity"]["effective"]["support_enabled"] is False
    for state in ("floor_stand", "no_floor_stand"):
        um = audit._artifact_profile_bundle({
            "id": f"{state}:um", "state": state,
            "variant": "Obi-Wan",
            "part": "obiwan_core_2_of_2_um_carrier",
        }, bundle, tmp_path / f"um_{state}")
        assert um["resolved"]["process"][
            "sparse_infill_density"] == "40%"
        assert um["identity"]["effective"]["support_enabled"] is True
        assert um["identity"]["effective"][
            "support_on_build_plate_only"] is True
        assert um["identity"]["effective"][
            "support_critical_regions_only"] is True
        assert um["identity"]["effective"][
            "support_remove_small_overhang"] is True


def test_every_artifact_override_must_match_once() -> None:
    config = audit._load_json(audit.DEFAULT_PROFILE)
    artifacts = [
        {"id": f"artifact-{index}", **rule["match"]}
        for index, rule in enumerate(config["artifact_overrides"])
    ]
    audit._validate_artifact_override_coverage(artifacts, config)
    try:
        audit._validate_artifact_override_coverage(artifacts[:-1], config)
    except audit.AuditError as exc:
        assert "expected exactly one" in str(exc)
    else:
        raise AssertionError("zero-match artifact override passed")
    duplicated = [*artifacts, {**artifacts[0], "id": "duplicate"}]
    try:
        audit._validate_artifact_override_coverage(duplicated, config)
    except audit.AuditError as exc:
        assert "expected exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous artifact override passed")


def test_support_override_targets_only_exact_duct_safe_jobs() -> None:
    config = audit._load_json(audit.DEFAULT_PROFILE)
    audit._validate_support_override_policy(config)
    base_process = config["repo_overrides"]["process"]
    assert base_process["enable_support"] == "0"
    assert base_process["support_on_build_plate_only"] == "0"
    assert base_process["support_critical_regions_only"] == "0"
    assert base_process["support_remove_small_overhang"] == "0"
    support_rules = [
        rule for rule in config["artifact_overrides"]
        if audit._boolish(rule.get("process", {}).get("enable_support"))
    ]
    assert len(support_rules) == 6
    assert {
        rule["match"]["state"] for rule in support_rules
    } == {"floor_stand", "no_floor_stand"}
    assert {
        rule["match"]["part"] for rule in support_rules
    } == {
        "obiwan_core_2_of_2_um_carrier",
        "obiwan_optional_lm_keyed_1_of_2_bottom",
        "obiwan_optional_lm_keyed_2_of_2_top",
    }
    assert {
        tuple(sorted(rule["match"].items())) for rule in support_rules
    } == {
        tuple(sorted(match.items()))
        for match in audit.SUPPORTED_ARTIFACT_MATCHES
    }
    for rule in support_rules:
        process = rule["process"]
        assert process["enable_support"] == "1"
        assert process["support_on_build_plate_only"] == "1"
        assert process["support_critical_regions_only"] == "1"
        assert process["support_remove_small_overhang"] == "1"

    extra = copy.deepcopy(config)
    extra["artifact_overrides"].append({
        "match": {"state": "floor_stand", "variant": "A",
                  "part": "unexpected"},
        "process": {
            "enable_support": "1",
            "support_on_build_plate_only": "1",
            "support_critical_regions_only": "1",
            "support_remove_small_overhang": "1",
        },
    })
    try:
        audit._validate_support_override_policy(extra)
    except audit.AuditError as exc:
        assert "support overrides must target exactly" in str(exc)
    else:
        raise AssertionError("unexpected support-enabled artifact passed")


def test_um_carrier_collision_contract_is_unsplit_and_exact() -> None:
    contract = {
        "schema_version": 1,
        "coordinate_space": "authoritative_source_mm",
        "owner": "um_carrier",
        "modifier_clearance_mm": 0.25,
        "regions": [{
            "name": "um_carrier_t_route_lumen",
            "kind": "polyline_tube",
            "radius_mm": 3.0,
            "points_xyz_mm": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }],
    }
    normalized = audit._normalize_duct_collision_contract(
        contract,
        artifact_id="no_floor_stand:Obi-Wan:"
        "obiwan_core_2_of_2_um_carrier",
        state="no_floor_stand",
        variant="Obi-Wan",
        part="obiwan_core_2_of_2_um_carrier",
        modifier_clearance_mm=0.25,
    )
    assert normalized["owner"] == "um_carrier"
    assert "split_half" not in normalized
    assert {
        region["name"] for region in normalized["regions"]
    } == audit.EXPECTED_UM_CARRIER_DUCT_REGION_NAMES


def test_floating_cantilever_warning_is_release_blocking(
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean.log"
    clean.write_text("[warning] no filament colors found\n", encoding="utf-8")
    audit._validate_bambu_slicer_log(
        clean, artifact_id="carrier", phase="audit slice")

    floating = tmp_path / "floating.log"
    floating.write_text(
        "found NON_CRITICAL slicing warnings: object has floating "
        "cantilever.\n",
        encoding="utf-8",
    )
    try:
        audit._validate_bambu_slicer_log(
            floating, artifact_id="carrier", phase="audit slice")
    except audit.AuditError as exc:
        assert "floating_cantilever" in str(exc)
    else:
        raise AssertionError("floating-cantilever warning passed")


def test_support_enabled_requires_all_three_safety_guards() -> None:
    bundle = _synthetic_profile_bundle()
    for missing in (
            "support_on_build_plate_only",
            "support_critical_regions_only",
            "support_remove_small_overhang"):
        resolved = copy.deepcopy(bundle["resolved"])
        enforced = copy.deepcopy(bundle["enforced_overrides"])
        resolved["process"]["enable_support"] = "1"
        enforced["process"]["enable_support"] = "1"
        resolved["process"][missing] = "0"
        enforced["process"][missing] = "0"
        for other in ({"support_on_build_plate_only",
                       "support_critical_regions_only",
                       "support_remove_small_overhang"} - {missing}):
            resolved["process"][other] = "1"
            enforced["process"][other] = "1"
        try:
            audit._effective_profile_contract(
                resolved, bundle["config"], enforced)
        except audit.AuditError as exc:
            assert "support-enabled artifact profiles" in str(exc)
        else:
            raise AssertionError(
                f"support profile without {missing} passed")


def test_bambu_version_banner_ignores_timestamped_trace() -> None:
    output = (
        "[2026-07-20 08:00:00] [trace] Initializing StaticPrintConfigs\n"
        "BambuStudio-02.07.01.62:\nUsage: bambu-studio\n")
    assert audit._parse_bambu_studio_version(output) == "02.07.01.62"
    for invalid in ("trace only\n", output + "BambuStudio-02.07.01.63:\n"):
        try:
            audit._parse_bambu_studio_version(invalid)
        except audit.AuditError as exc:
            assert "exactly one" in str(exc)
        else:
            raise AssertionError("ambiguous/missing Bambu version passed")


def test_actual_gcode_profile_checks_every_pinned_setting(
        tmp_path: Path) -> None:
    bundle = _synthetic_profile_bundle()
    effective = bundle["identity"]["effective"]
    config = {
        "layer_height": "0.16",
        "initial_layer_print_height": "0.2",
        "outer_wall_line_width": "0.42",
        "inner_wall_line_width": "0.45",
        "wall_loops": "6",
        "top_shell_layers": "6",
        "bottom_shell_layers": "5",
        # Bambu's actual CONFIG_BLOCK collapses identical preset vectors.
        "outer_wall_speed": "60",
        "curr_bed_type": "Textured PEI Plate",
        "elefant_foot_compensation": "0.15",
        "xy_hole_compensation": "0",
        "nozzle_temperature": "245",
        "nozzle_temperature_initial_layer": "245",
        "fan_max_speed": "100",
        "overhang_fan_speed": "100",
        "filament_max_volumetric_speed": "21",
        "textured_plate_temp": "60",
        "textured_plate_temp_initial_layer": "60",
        "wall_generator": "arachne",
        "enable_support": "0",
        "support_on_build_plate_only": "0",
        "support_critical_regions_only": "0",
        "support_remove_small_overhang": "0",
        "sparse_infill_pattern": "gyroid",
        "sparse_infill_density": (
            f"{effective['sparse_infill_density_percent']:g}%"),
        "precise_outer_wall": "1",
        "detect_thin_wall": "1",
        "ensure_vertical_shell_thickness": "enabled",
        "detect_narrow_internal_solid_infill": "1",
        "machine_pause_gcode": "M400 U1",
        "enable_arc_fitting": "0",
    }

    def parsed(values: dict) -> audit.ParsedGcode:
        return audit.ParsedGcode(
            [audit.Layer(0.20, 0.20, [], 1),
             audit.Layer(0.36, 0.16, [], 2)],
            1, 0, 1, 1, (0.0, 0.0, 0.0), (1.0, 1.0, 0.36), values)

    assert audit._validate_actual_gcode_profile(parsed(config), bundle) == []
    mutations = {
        "wall_generator": "classic",
        "enable_support": "1",
        "support_on_build_plate_only": "1",
        "support_critical_regions_only": "1",
        "support_remove_small_overhang": "1",
        "wall_loops": "5",
        "top_shell_layers": "5",
        "bottom_shell_layers": "4",
        "outer_wall_speed": "200",
        "curr_bed_type": "Cool Plate",
        "sparse_infill_pattern": "grid",
        "sparse_infill_density": "15%",
        "precise_outer_wall": "0",
        "detect_thin_wall": "0",
        "ensure_vertical_shell_thickness": "disabled",
        "detect_narrow_internal_solid_infill": "0",
        "elefant_foot_compensation": "0",
        "xy_hole_compensation": "0.05",
        "nozzle_temperature": "225",
        "fan_max_speed": "60",
        "overhang_fan_speed": "50",
        "filament_max_volumetric_speed": "16",
        "textured_plate_temp": "55",
        "textured_plate_temp_initial_layer": "55",
        "machine_pause_gcode": "M0",
    }
    for key, value in mutations.items():
        changed = dict(config)
        changed[key] = value
        errors = audit._validate_actual_gcode_profile(parsed(changed), bundle)
        assert errors, f"mutated pinned G-code setting passed: {key}"

    # Pattern validation follows the exact resolved artifact profile.  This is
    # required for the 100%-solid keyed LM bottom, where Bambu rejects gyroid.
    solid_bundle = {
        **bundle,
        "identity": {
            **bundle["identity"],
            "effective": {
                **effective,
                "sparse_infill_pattern": "zig-zag",
            },
        },
    }
    solid_config = {
        **config,
        "sparse_infill_pattern": "zig-zag",
    }
    assert audit._validate_actual_gcode_profile(
        parsed(solid_config), solid_bundle) == []

    changed = dict(config)
    changed["outer_wall_speed"] = "60,200"
    errors = audit._validate_actual_gcode_profile(parsed(changed), bundle)
    assert any("outer_wall_speed" in error for error in errors)

    support_bundle = audit._artifact_profile_bundle({
        "id": "floor:split:bottom",
        "state": "floor_stand",
        "variant": "Obi-Wan-split",
        "part": "obiwan_optional_lm_keyed_1_of_2_bottom",
    }, bundle, tmp_path / "support_profile")
    support_config = {
        **config,
        "enable_support": "1",
        "support_on_build_plate_only": "1",
        "support_critical_regions_only": "1",
        "support_remove_small_overhang": "1",
        "sparse_infill_pattern": "zig-zag",
        "sparse_infill_density": "100%",
    }
    assert audit._validate_actual_gcode_profile(
        parsed(support_config), support_bundle) == []
    for key in (
            "enable_support", "support_on_build_plate_only",
            "support_critical_regions_only",
            "support_remove_small_overhang"):
        changed = dict(support_config)
        changed[key] = "0"
        errors = audit._validate_actual_gcode_profile(
            parsed(changed), support_bundle)
        assert any(key in error for error in errors)


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
    assert "--allow-rotations=0" in command
    assert command[command.index("--export-3mf") + 1] == (
        audit.PLACED_3MF_FILENAME)

    custom = tmp_path / "custom_gcodes.json"
    ready = audit._bambu_command(
        tmp_path / "BambuStudio", tmp_path / "part.stl",
        tmp_path / "out", profile_bundle,
        project_filename=audit.READY_3MF_FILENAME,
        custom_gcodes=custom)
    assert ready[ready.index("--export-3mf") + 1] == (
        audit.READY_3MF_FILENAME)
    assert ready[ready.index("--load-custom-gcodes") + 1] == str(custom)
    assert ready[-1] == str(tmp_path / "part.stl")


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
        "provisional unpaired axial convention: marked/N pole points "
        "rearward; verify any future mate before burial")
    record = {
        "id": "state:AX:part",
        "audit_mode": "actual_p2s_slice",
        "status": "pass",
        "sites": [{
            "site": {
                "name": "axial_left",
                "print_insertion_direction_xyz": (0.0, 0.0, -1.0),
                "print_marked_pole_axis_xyz": (0.0, 0.0, 1.0),
                "installed_marked_pole_axis_xyz": (0.0, 0.0, -1.0),
                "polarity_instruction": polarity,
            },
            "actual": {
                "bambu_studio_pause_marker_z_mm": 15.32,
                "last_completely_open_layer_z_mm": 15.16,
                "cavity_bury_roof_start_plane_z_mm": 15.18,
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


def test_ready_custom_gcodes_and_archive_are_self_contained(
        tmp_path: Path) -> None:
    record = {
        "id": "state:Obi-Wan:part",
        "audit_mode": "actual_p2s_slice",
        "status": "pass",
        "sites": [{
            "site": {
                "name": "um_left",
                "print_insertion_direction_xyz": (0.0, 0.0, -1.0),
                "print_marked_pole_axis_xyz": (1.0, 0.0, 0.0),
                "installed_marked_pole_axis_xyz": (1.0, 0.0, 0.0),
                "polarity_instruction": "marked pole follows +X",
            },
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
    bundle = _synthetic_profile_bundle()
    pause_policy = bundle["identity"]["effective"]["magnet_insertion_pause"]
    custom, pause_z = audit._custom_gcodes_document(record, pause_policy)
    assert pause_z == [5.96]
    assert custom["mode"] == "SingleExtruder"
    assert custom["gcodes"][0]["type"] == "Custom"
    assert custom["gcodes"][0]["print_z"] == 5.96
    assert custom["gcodes"][0]["extruder"] == 1
    program = custom["gcodes"][0]["extra"]
    assert audit.MAGNET_INSERTION_PAUSE_COMMAND in program
    assert "G1 Z250 F1200" in program
    assert "G1 Z5.96 F1200" in program

    settings = {}
    for values in bundle["enforced_overrides"].values():
        settings.update(values)
    gcode = tmp_path / "plate_1.gcode"
    gcode.write_text("\n".join((
        "; CHANGE_LAYER", "; Z_HEIGHT: 5.96", "; CUSTOM_GCODE",
        program, "G1 X1 Y1 E0.1",
    )) + "\n", encoding="utf-8")
    xml_program = (program.replace("&", "&amp;")
                   .replace("\"", "&quot;")
                   .replace("\n", "&#10;"))
    custom_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<custom_gcodes_per_layer><plate><plate_info id="1"/>'
        '<layer top_z="5.96" type="4" extruder="1" color="" '
        f'extra="{xml_program}" gcode="{xml_program}"/>'
        '<mode value="SingleExtruder"/></plate>'
        '</custom_gcodes_per_layer>')
    model_settings = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<config><object id="2"><metadata key="name" '
        'value="synthetic.stl"/></object></config>')
    project = tmp_path / audit.READY_3MF_FILENAME
    with zipfile.ZipFile(project, "w") as archive:
        archive.writestr(
            "Metadata/project_settings.config", json.dumps(settings))
        archive.writestr("Metadata/model_settings.config", model_settings)
        archive.writestr(
            "Metadata/custom_gcode_per_layer.xml", custom_xml)
        archive.writestr("Metadata/plate_1.gcode", gcode.read_bytes())
    assert audit._inject_ready_project_object_support(
        project, enabled=False) == ["2"]
    result = audit._validate_ready_project_archive(
        project, gcode, expected_pause_z=pause_z,
        profile_bundle=bundle)
    assert result["pause_z_mm"] == [5.96]
    assert result["embedded_gcode_sha256"] == audit.sha256_file(gcode)
    assert result["gcode_pause_events"][0]["park_z_mm"] == 250.0
    assert result["gcode_pause_events"][0]["restore_z_mm"] == 5.96
    assert result["object_support_overrides"] == [{
        "object_id": "2",
        "enable_support": "0",
        "support_on_build_plate_only": "0",
        "support_critical_regions_only": "0",
        "support_remove_small_overhang": "0",
    }]

    support_bundle = audit._artifact_profile_bundle({
        "id": "floor:split:bottom",
        "state": "floor_stand",
        "variant": "Obi-Wan-split",
        "part": "obiwan_optional_lm_keyed_1_of_2_bottom",
    }, bundle, tmp_path / "support_ready_profile")
    support_settings = {}
    for values in support_bundle["enforced_overrides"].values():
        support_settings.update(values)
    support_project = tmp_path / "supported_ready.gcode.3mf"
    with zipfile.ZipFile(support_project, "w") as archive:
        archive.writestr(
            "Metadata/project_settings.config",
            json.dumps(support_settings))
        archive.writestr("Metadata/model_settings.config", model_settings)
        archive.writestr(
            "Metadata/custom_gcode_per_layer.xml", custom_xml)
        archive.writestr("Metadata/plate_1.gcode", gcode.read_bytes())
    assert audit._inject_ready_project_object_support(
        support_project, enabled=True) == ["2"]
    support_result = audit._validate_ready_project_archive(
        support_project, gcode, expected_pause_z=pause_z,
        profile_bundle=support_bundle)
    assert support_result["enforced_project_settings"][
        "enable_support"] == "1"
    assert support_result["enforced_project_settings"][
        "support_on_build_plate_only"] == "1"
    assert support_result["enforced_project_settings"][
        "support_critical_regions_only"] == "1"
    assert support_result["enforced_project_settings"][
        "support_remove_small_overhang"] == "1"
    assert support_result["object_support_overrides"] == [{
        "object_id": "2",
        "enable_support": "1",
        "support_on_build_plate_only": "1",
        "support_critical_regions_only": "1",
        "support_remove_small_overhang": "1",
    }]

    incomplete_object_project = (
        tmp_path / "incomplete_object_supported_ready.gcode.3mf")
    with zipfile.ZipFile(incomplete_object_project, "w") as archive:
        archive.writestr(
            "Metadata/project_settings.config",
            json.dumps(support_settings))
        archive.writestr(
            "Metadata/model_settings.config",
            model_settings.replace(
                'value="synthetic.stl"/>',
                'value="synthetic.stl"/><metadata key="enable_support" '
                'value="1"/>'))
        archive.writestr(
            "Metadata/custom_gcode_per_layer.xml", custom_xml)
        archive.writestr("Metadata/plate_1.gcode", gcode.read_bytes())
    try:
        audit._validate_ready_project_archive(
            incomplete_object_project, gcode, expected_pause_z=pause_z,
            profile_bundle=support_bundle)
    except audit.AuditError as exc:
        assert "support_on_build_plate_only" in str(exc)
    else:
        raise AssertionError(
            "ready project with incomplete per-object support passed")

    support_settings["support_critical_regions_only"] = "0"
    invalid_support_project = tmp_path / "invalid_supported_ready.gcode.3mf"
    with zipfile.ZipFile(invalid_support_project, "w") as archive:
        archive.writestr(
            "Metadata/project_settings.config",
            json.dumps(support_settings))
        archive.writestr(
            "Metadata/model_settings.config",
            model_settings.replace(
                'value="synthetic.stl"/>',
                'value="synthetic.stl"/><metadata key="enable_support" '
                'value="1"/>'))
        archive.writestr(
            "Metadata/custom_gcode_per_layer.xml", custom_xml)
        archive.writestr("Metadata/plate_1.gcode", gcode.read_bytes())
    try:
        audit._validate_ready_project_archive(
            invalid_support_project, gcode, expected_pause_z=pause_z,
            profile_bundle=support_bundle)
    except audit.AuditError as exc:
        assert "support_critical_regions_only" in str(exc)
    else:
        raise AssertionError(
            "ready project missing critical-region support passed")

    events = result["gcode_pause_events"]
    parsed = audit.ParsedGcode(
        [audit.Layer(5.96, 0.16, [], 1,
                     first_extrusion_line_number=16)],
        1, 0, 1, 1, (0.0, 0.0, 0.0), (1.0, 1.0, 5.96), {})
    ordering = audit._assert_pauses_precede_layer_extrusion(parsed, events)
    assert ordering[0]["pass"] is True
    parsed.layers[0].first_extrusion_line_number = 12
    try:
        audit._assert_pauses_precede_layer_extrusion(parsed, events)
    except audit.AuditError as exc:
        assert "not before first layer extrusion" in str(exc)
    else:
        raise AssertionError("pause after layer extrusion passed")


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


def test_gcode_skill_wrapper_uses_active_bed_and_native_profile_stack(
        tmp_path: Path) -> None:
    bundle = _synthetic_profile_bundle()
    bundle["paths"] = {
        name: tmp_path / f"resolved_{name}.json"
        for name in ("machine", "process", "filament")
    }
    wrapper = audit._gcode_validation_wrapper(bundle)
    assert wrapper["filament"] == {
        "type": "Bambu PLA Basic @BBL P2S",
        "nozzle_temp_c": 245.0,
        "bed_temp_c": 60.0,
    }
    assert wrapper["native_settings"] == [
        str(bundle["paths"]["machine"]),
        str(bundle["paths"]["process"]),
    ]
    assert wrapper["native_filaments"] == [
        str(bundle["paths"]["filament"]),
    ]


def test_generator_style_source_matrix_is_consumed(tmp_path: Path):
    # This is the exact structural shape emitted by
    # generate_captive_magnet_catalog.py: source-space helper facts plus one
    # artifact-level source-to-STL matrix, not nested print-space facts.
    catalog = _catalog_document([{
            "id": "state:Obi-Wan:part", "part": "part", "variant": "Obi-Wan",
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
                "name": "lm_upper_right",
                "closure_kind": "transverse_gable_45deg",
                "interface_kind": "ring",
                "carrier_cavity_face_inset_mm": 0.15,
                "carrier_cavity_datum_xy_mm": [2.0, 2.0],
                "outer_surface_face_xy_mm": [2.15, 2.0],
                "paired_magnet_face_separation_mm": 1.24,
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
            "id": "state:AX:part", "part": "part", "variant": "AX",
            "state": "state", "stl": "part.stl",
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 0.0},
            "source_to_stl_matrix": [
                [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 10.0],
                [0.0, 0.0, -1.0, 18.3], [0.0, 0.0, 0.0, 1.0]],
            "sites": [{
                **_release_site_contract((0.0, 0.0, -1.0)),
                "name": "axial", "closure_kind": "axis_opposed_conical_45deg",
                "cavity_bury_roof_start_print_z_mm": 15.18,
                "roof_apex_print_z_mm": 17.78,
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
        "face_skin_mm": 0.52,
        "inner_skin_mm": 0.52,
    }

    def segment(
        x: float, y0: float, y1: float, *,
        feature: str = "Outer wall", width: float = 0.42,
        path_id: int = 1,
    ) -> audit.Segment:
        return audit.Segment(
            x, y0, x, y1, 0.1, feature, width, 1,
            path_id=path_id)

    def side_wall(y: float) -> audit.Segment:
        return audit.Segment(
            0.52, y, 2.62, y, 0.1, "Outer wall", 0.42, 1)

    interface_x = 0.52 / 2.0
    inner_x = 0.52 + 2.1 + 0.52 / 2.0
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
    assert continuous["retaining_paths"][
        "interface_skin_single_path"]["estimated_path_count"] == 1

    # Bambu serializes ordinary 0.42-mm paths as 0.419996 mm and Arachne can
    # locally resolve the same nominal one-wall skin to 0.415656 mm. Admit
    # that bounded lower-side modulation, but reject a materially thin bead
    # below the explicit 0.415-mm floor.
    arachne_lower_edge = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1, width=0.415656),
            segment(inner_x, -2.1, 2.1, width=0.415656),
        ], 1), site, (0.0, 0.0))
    assert arachne_lower_edge["retaining_paths"]["pass"] is True
    assert arachne_lower_edge["retaining_paths"][
        "interface_skin_single_path"]["lower_width_tolerance_mm"] == 0.005

    materially_underwidth = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1, width=0.414),
            segment(inner_x, -2.1, 2.1, width=0.414),
        ], 1), site, (0.0, 0.0))
    assert materially_underwidth["retaining_paths"]["pass"] is False

    # Bambu's thin-wall medial bead may widen beyond the nominal 0.42 mm.  It
    # remains valid only as one traversal and while the path-width-aware D5x2
    # loading aperture remains clear.
    widened = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x - (0.586 - 0.45) / 2.0,
                    -2.1, 2.1, width=0.586),
            segment(inner_x + 0.085, -2.1, 2.1, width=0.586),
            side_wall(-2.81), side_wall(2.81),
        ], 1), site, (0.0, 0.0))
    assert widened["retaining_paths"]["pass"] is True
    assert widened["loading_aperture"]["free_axial_slot_width_mm"] >= 2.0
    assert audit._loading_aperture_pass({
        **site, "magnet_diameter_mm": 5.0, "magnet_depth_mm": 2.0,
    }, widened)[0] is True

    # Legacy V1's inner skin is also one adaptive-width traversal.  Its
    # measured 0.661027-mm bead remains valid because the opposing 0.484-mm
    # path and actual centre placement leave more than 2.0 mm of free slot.
    legacy_widened = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1, width=0.484),
            segment(inner_x + 0.09, -2.1, 2.1, width=0.661),
            side_wall(-2.81), side_wall(2.81),
        ], 1), site, (0.0, 0.0))
    assert legacy_widened["retaining_paths"]["pass"] is True
    assert legacy_widened["loading_aperture"][
        "free_axial_slot_width_mm"] >= 2.0
    assert audit._loading_aperture_pass({
        **site, "magnet_diameter_mm": 5.0, "magnet_depth_mm": 2.0,
    }, legacy_widened)[0] is True

    overwide = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x - (0.785 - 0.52) / 2.0,
                    -2.1, 2.1, width=0.785),
            segment(inner_x + (0.785 - 0.52) / 2.0,
                    -2.1, 2.1, width=0.785),
            side_wall(-2.81), side_wall(2.81),
        ], 1), site, (0.0, 0.0))
    assert overwide["retaining_paths"]["pass"] is False

    # A speed/feature split can divide one physical centreline into several
    # G-code moves; it remains exactly one path in every transverse scan bin.
    segmented_one = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 0.0),
            segment(interface_x, 0.0, 2.1),
            segment(inner_x, -2.1, -0.3),
            segment(inner_x, -0.3, 2.1),
        ], 1), site, (0.0, 0.0))
    assert segmented_one["retaining_paths"]["pass"] is True

    # The two real Bambu traces separated by only 0.03 mm are still two
    # extrusion passes.  Neither a full parallel pass nor a short second pass
    # may be hidden by loose spatial clustering.  Both 0.42-mm passes keep
    # their cavity-facing edges within the 0.06-mm boundary tolerance of the
    # 0.52-mm skin, so the strict exact-one gate sees them both.
    two_full = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x + 0.02, -2.1, 2.1),
            segment(interface_x + 0.05, -2.1, 2.1),
            segment(inner_x - 0.05, -2.1, 2.1),
            segment(inner_x - 0.02, -2.1, 2.1),
        ], 1), site, (0.0, 0.0))
    assert two_full["retaining_paths"][
        "interface_skin_single_path"]["estimated_path_count"] == 2
    assert two_full["retaining_paths"]["pass"] is False

    short_second = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1),
            segment(interface_x + 0.03, -0.2, 0.2),
            segment(inner_x, -2.1, 2.1),
        ], 1), site, (0.0, 0.0))
    assert short_second["retaining_paths"]["pass"] is False
    assert max(short_second["retaining_paths"][
        "interface_skin_single_path"][
            "path_count_by_scan_bin"].values()) == 2

    # A surrounding-body return can pass through the broad centreline band
    # without forming the cavity wall.  Candidate classification uses the
    # path-width-aware cavity-facing bead edge, so that body-side geometry is
    # excluded while exact-one checks remain strict at the actual boundary.
    body_side_return = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1, path_id=10),
            segment(inner_x, -2.1, 2.1, path_id=20),
            segment(inner_x + 0.24, -1.0, -0.7, path_id=20),
        ], 1), site, (0.0, 0.0))
    edge_summary = body_side_return["retaining_paths"][
        "inner_skin_single_path"]
    assert body_side_return["retaining_paths"]["pass"] is True
    assert edge_summary["estimated_path_count"] == 1
    assert edge_summary["candidate_selection"] == "cavity_facing_bead_edge"
    assert edge_summary["cavity_edge_tolerance_mm"] == 0.06

    independent_body_return = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1, path_id=10),
            segment(inner_x, -2.1, 2.1, path_id=20),
            segment(inner_x + 0.24, -1.0, -0.7, path_id=21),
        ], 1), site, (0.0, 0.0))
    assert independent_body_return["retaining_paths"]["pass"] is False

    long_same_path_return = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1, path_id=10),
            segment(inner_x, -2.1, 2.1, path_id=20),
            segment(inner_x + 0.24, -1.0, -0.6, path_id=20),
        ], 1), site, (0.0, 0.0))
    assert long_same_path_return["retaining_paths"]["pass"] is False

    # A full duplicate only 0.001 mm outside the cavity-edge classifier must
    # not become invisible.  Its bead still overlaps the selected wall across
    # all scan bins, so the independent nearby-duplicate guard rejects it even
    # though the D5x2 loading slot remains open.
    adaptive_interface_x = 0.45 - 0.484 / 2.0
    just_outside_cutoff = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(adaptive_interface_x, -2.1, 2.1,
                    width=0.484, path_id=30),
            segment(adaptive_interface_x + 0.061, -2.1, 2.1,
                    width=0.484, path_id=31),
            segment(inner_x, -2.1, 2.1, path_id=40),
            side_wall(-2.81), side_wall(2.81),
        ], 1), site, (0.0, 0.0))
    cutoff_summary = just_outside_cutoff["retaining_paths"][
        "interface_skin_single_path"]
    assert cutoff_summary["estimated_path_count"] == 1
    assert cutoff_summary[
        "nearby_overlapping_maximum_crossings_per_scan_bin"] == 2
    assert cutoff_summary["nearby_duplicate_guard_pass"] is False
    assert just_outside_cutoff["retaining_paths"]["pass"] is False
    assert audit._loading_aperture_pass({
        **site, "magnet_diameter_mm": 5.0, "magnet_depth_mm": 2.0,
    }, just_outside_cutoff)[0] is True

    gap_fill = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            segment(interface_x, -2.1, 2.1),
            segment(interface_x + 0.03, -0.2, 0.2,
                    feature="Gap infill"),
            segment(inner_x, -2.1, 2.1),
        ], 1), site, (0.0, 0.0))
    assert gap_fill["retaining_paths"][
        "interface_skin_single_path"]["outer_wall_only_pass"] is False
    assert gap_fill["retaining_paths"]["pass"] is False

    # Projecting only onto V would incorrectly merge these two overlapping
    # halves.  Their U separation exceeds one 0.42-mm Arachne path, so they
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

    site = {
        "closure_kind": "axis_opposed_conical_45deg",
        "print_cavity_center_xyz_mm": (0.0, 0.0, 3.0),
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "magnet_diameter_mm": 5.0,
        "magnet_depth_mm": 2.0,
        "face_skin_mm": 0.52,
    }

    def ring(
        ring_radius: float, *, path_id: int = 1, width: float = 0.42,
    ) -> list[audit.Segment]:
        points = [
            (ring_radius * math.cos(math.radians(value)),
             ring_radius * math.sin(math.radians(value)))
            for value in range(0, 361, 2)
        ]
        return [
            audit.Segment(
                x0, y0, x1, y1, 0.01, "Outer wall", width, index,
                path_id=path_id)
            for index, ((x0, y0), (x1, y1))
            in enumerate(zip(points, points[1:]), 1)
        ]

    one = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, ring(2.81), 1), site, (0.0, 0.0))
    assert one["retaining_paths"]["single_classic_path_pass"] is True
    serialized_nominal = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, ring(2.81, width=0.419996), 1),
        site, (0.0, 0.0))
    assert serialized_nominal["retaining_paths"][
        "single_classic_path_pass"] is True
    materially_underwidth = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, ring(2.81, width=0.41999), 1),
        site, (0.0, 0.0))
    assert materially_underwidth["retaining_paths"][
        "single_classic_path_pass"] is False
    two = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            *ring(2.81, path_id=1), *ring(2.985, path_id=2)], 1),
        site, (0.0, 0.0))
    assert two["retaining_paths"]["annular_single_path"][
        "estimated_path_count"] == 2
    assert two["retaining_paths"]["single_classic_path_pass"] is False

    # Even if malformed input labels two complete circumferences as one raw
    # path, the bounded seam exception cannot hide 72 doubled ray crossings.
    same_path_double = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            *ring(2.81, path_id=1), *ring(2.985, path_id=1)], 1),
        site, (0.0, 0.0))
    annular = same_path_double["retaining_paths"]["annular_single_path"]
    assert annular["estimated_path_count"] == 1
    assert len(annular["multiple_crossing_ray_bins"]) > 2
    assert same_path_double["retaining_paths"][
        "single_classic_path_pass"] is False

    def arc(
        arc_radius: float, start_deg: float, end_deg: float, path_id: int,
    ) -> list[audit.Segment]:
        values = [start_deg]
        while values[-1] + 1.0 < end_deg:
            values.append(values[-1] + 1.0)
        values.append(end_deg)
        points = [
            (arc_radius * math.cos(math.radians(value)),
             arc_radius * math.sin(math.radians(value)))
            for value in values
        ]
        return [
            audit.Segment(
                x0, y0, x1, y1, 0.01, "Outer wall", 0.42, index,
                path_id=path_id)
            for index, ((x0, y0), (x1, y1))
            in enumerate(zip(points, points[1:]), 1)
        ]

    # Two angularly complementary arcs can still be physically disconnected
    # when their radii differ.  This case otherwise clears the D5 aperture and
    # the 0.52-mm angular-gap gate; only endpoint-local Euclidean seam checking
    # proves that it is not one printable annular bead.
    disconnected_arcs = audit._toolpath_metrics(
        audit.Layer(3.0, 0.16, [
            *arc(2.711, 7.4, 172.6, 11),
            *arc(2.985, 182.4, 357.6, 12),
        ], 1), site, (0.0, 0.0))
    disconnected_summary = disconnected_arcs["retaining_paths"][
        "annular_single_path"]
    assert disconnected_arcs["retaining_paths"][
        "largest_uncovered_arc_mm"] < 0.52
    assert disconnected_arcs["loading_aperture"][
        "free_radial_diameter_mm"] >= 5.0
    assert disconnected_summary[
        "complementary_component_coverage_pass"] is True
    assert disconnected_summary["component_seam_continuity_pass"] is False
    assert disconnected_arcs["retaining_paths"]["pass"] is False


def test_axial_single_path_allows_only_bounded_local_seam_anomaly() -> None:
    expected_radius = 2.825
    bins = tuple(range(72))

    # Model the measured one-component seam: bin 12 is missed and adjacent bin
    # 13 crosses the same raw path twice.  Both anomalies remain in one bounded
    # endpoint neighborhood of one circumference.
    local_seam = [
        (expected_radius, index, "Outer wall", 0.42, 17)
        for index in bins if index != 12
    ]
    local_seam.append((
        expected_radius + 0.03, 13, "Outer wall", 0.631, 17))
    summary = audit._single_annular_classic_track_summary(
        track_samples=local_seam,
        required_bins=tuple(range(72)),
        expected_center_mm=expected_radius,
        allowed_width_range_mm=audit.AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM)
    assert summary["unique_annular_path_pass"] is True
    assert summary["occupied_ray_bin_count"] == 71
    assert summary["missing_scan_bins"] == [12]
    assert summary["multiple_crossing_ray_bins"] == [13]
    assert summary["anomaly_endpoint_local_pass"] is True
    assert summary["single_classic_path_pass"] is True

    # The measured right axial ring is one geometric bead emitted as two long,
    # complementary arcs: 46 exclusive rays plus 24, with only the two seam
    # bins missing and no cross-component overlap.
    split_components = [
        (expected_radius, index, "Outer wall", 0.42, 17)
        for index in range(16, 62)
    ]
    split_components.extend(
        (expected_radius + 0.02, index, "Outer wall", 0.631, 18)
        for index in (*range(63, 72), *range(0, 15)))
    split_bead = audit._single_annular_classic_track_summary(
        track_samples=split_components,
        required_bins=bins,
        expected_center_mm=expected_radius,
        allowed_width_range_mm=audit.AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM)
    assert split_bead["unique_annular_path_pass"] is False
    assert split_bead["bounded_component_path_count_pass"] is True
    assert split_bead["component_exclusive_ray_bin_count"] == {
        "17": 46, "18": 24,
    }
    assert split_bead["cross_component_overlap_ray_bins"] == []
    assert split_bead["complementary_component_coverage_pass"] is True
    assert split_bead["component_seam_continuity_pass"] is True
    assert split_bead["single_classic_path_pass"] is True

    # Angular complementarity alone is insufficient: two arcs at different
    # radii can cover 70/72 rays while leaving a physical void at both seams.
    # Their nearest ray-sampled centreline gaps exceed both the bead-footprint
    # limit and the general 0.52-mm connectivity cap.
    disconnected_split = [
        (2.711, index, "Outer wall", 0.42, 17)
        for index in range(1, 35)
    ]
    disconnected_split.extend(
        (2.984, index, "Outer wall", 0.42, 18)
        for index in range(36, 72))
    disconnected = audit._single_annular_classic_track_summary(
        track_samples=disconnected_split,
        required_bins=bins,
        expected_center_mm=expected_radius,
        allowed_width_range_mm=audit.AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM)
    assert disconnected["complementary_component_coverage_pass"] is True
    assert len(disconnected["component_seam_junctions"]) == 2
    assert disconnected["component_seam_continuity_pass"] is False
    assert disconnected["single_classic_path_pass"] is False

    # A complete ring plus one stray sample is not a complementary split.  It
    # overlaps the first component and contributes no meaningful exclusive arc.
    complete_plus_stray = [
        (expected_radius, index, "Outer wall", 0.42, 17)
        for index in bins
    ]
    complete_plus_stray.append((
        expected_radius + 0.02, 13, "Outer wall", 0.42, 18))
    stray = audit._single_annular_classic_track_summary(
        track_samples=complete_plus_stray,
        required_bins=bins,
        expected_center_mm=expected_radius,
        allowed_width_range_mm=audit.AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM)
    assert stray["bounded_component_path_count_pass"] is True
    assert stray["complementary_component_coverage_pass"] is False
    assert stray["single_classic_path_pass"] is False

    # A bounded count is not enough when the missing and doubled rays are far
    # apart rather than local to the same single-component seam.
    remote_double = [
        (expected_radius, index, "Outer wall", 0.42, 17)
        for index in bins if index != 12
    ]
    remote_double.append((
        expected_radius + 0.03, 40, "Outer wall", 0.631, 17))
    nonlocal_seam = audit._single_annular_classic_track_summary(
        track_samples=remote_double,
        required_bins=bins,
        expected_center_mm=expected_radius,
        allowed_width_range_mm=audit.AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM)
    assert nonlocal_seam["bounded_combined_anomaly_count_pass"] is True
    assert nonlocal_seam["anomaly_endpoint_local_pass"] is False
    assert nonlocal_seam["single_classic_path_pass"] is False

    # More than two independently emitted components is no longer a bounded
    # two-arc representation of one medial ring.
    third_path = [*split_components, (
        expected_radius + 0.01, 15, "Outer wall", 0.42, 19)]
    rejected = audit._single_annular_classic_track_summary(
        track_samples=third_path,
        required_bins=bins,
        expected_center_mm=expected_radius,
        allowed_width_range_mm=audit.AXIAL_RETAINING_BEAD_WIDTH_RANGE_MM)
    assert rejected["bounded_component_path_count_pass"] is False
    assert rejected["single_classic_path_pass"] is False


def test_last_open_loading_aperture_checks_diameter_slot_and_obstruction():
    site = {
        "closure_kind": "transverse_gable_45deg",
        "print_cavity_center_xyz_mm": (0.0, 0.0, 3.0),
        "print_actual_face_xyz_mm": (0.0, 0.0, 3.0),
        "print_material_inward_xyz": (1.0, 0.0, 0.0),
        "cavity_diameter_mm": 5.2,
        "cavity_depth_mm": 2.1,
        "face_skin_mm": 0.52,
        "inner_skin_mm": 0.52,
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
            "id": "state:AX:descriptive-id",
            "part": "actual-print-part",
            "variant": "AX", "state": "state", "stl": "part.stl",
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 0.0},
            "source_to_stl_matrix": [
                [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 10.0],
                [0.0, 0.0, -1.0, 18.3], [0.0, 0.0, 0.0, 1.0]],
            "sites": [{
                **_release_site_contract((0.0, 0.0, -1.0)),
                "name": "axial", "closure_kind": "axis_opposed_conical_45deg",
                "cavity_bury_roof_start_print_z_mm": 15.18,
                "roof_apex_print_z_mm": 17.78,
                "cavity_center_xyz_mm": [2.0, 3.0, 4.1],
                "seated_magnet_center_xyz_mm": [2.0, 3.0, 4.1],
                "marked_pole_axis_xyz": [0.0, 0.0, -1.0],
            }],
        }])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    artifact = audit.normalize_catalog(
        path, enforce_release_inventory=False)["artifacts"][0]
    assert artifact["id"] == "state:AX:descriptive-id"
    assert artifact["part"] == "actual-print-part"


def _exact_split_proxy_catalog() -> dict:
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 20.0],
        [0.0, 0.0, -1.0, 18.3],
        [0.0, 0.0, 0.0, 1.0],
    ]

    def site(name: str, x: float) -> dict:
        interface_kind = (
            "shoulder" if name.startswith("lm_lower_") else "ring")
        cavity_inset = 0.15
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
            "face_skin_mm": 0.52,
            "inner_skin_mm": 0.52,
            "roof_angle_deg": 45.0,
            "polarity_instruction": "marked/N pole follows installed axis",
            "captive_land_mm": 3.14,
            "interface_gap_mm": 0.05,
            "paired_magnet_face_separation_mm": round(
                1.09 + cavity_inset, 9),
            "interface_kind": interface_kind,
            "carrier_cavity_face_inset_mm": cavity_inset,
            "carrier_cavity_datum_xy_mm": [x - 1.5, 10.0],
            "outer_surface_face_xy_mm": [
                x - 1.5 - cavity_inset, 10.0],
            "minimum_retaining_path_mm": 0.42,
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
    monolith = artifact("floor:Obi-Wan:mono", "Obi-Wan", "mono", [lower, upper])
    monolith["p2s_printability"] = "not_printable_oversize"
    monolith["cavity_audit_proxies"] = [
        {"site": "lm_lower_left", "artifact_id": "floor:split:bottom",
         "proxy_site": "lm_lower_left"},
        {"site": "lm_upper_left", "artifact_id": "floor:split:top",
         "proxy_site": "lm_upper_left"},
    ]
    return _catalog_document([
        monolith,
        artifact("floor:split:bottom", "Obi-Wan-split", "bottom", [lower]),
        artifact("floor:split:top", "Obi-Wan-split", "top", [upper]),
    ])


def test_obiwan_ring_and_shoulder_pair_spacing_is_1p24_mm(
        tmp_path: Path) -> None:
    payload = _exact_split_proxy_catalog()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    normalized = audit.normalize_catalog(
        path, enforce_release_inventory=False)
    monolith_sites = {
        site["name"]: site
        for site in normalized["artifacts"][0]["sites"]
    }
    assert monolith_sites["lm_lower_left"][
        "paired_magnet_face_separation_mm"] == 1.24
    assert monolith_sites["lm_upper_left"][
        "paired_magnet_face_separation_mm"] == 1.24

    payload["artifacts"][0]["sites"][0][
        "paired_magnet_face_separation_mm"] = 0.95
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        audit.normalize_catalog(path, enforce_release_inventory=False)
    except audit.AuditError as exc:
        assert (
            "paired magnet-face separation must be 1.240" in str(exc)
            or "paired_magnet_face_separation_mm: value does not equal "
               "const 1.24" in str(exc)
            or "paired_magnet_face_separation_mm: value is not in enum"
            in str(exc))
    else:
        raise AssertionError("stale 0.95-mm Obi-Wan shoulder spacing passed")


def test_standard_curved_pair_spacing_uses_declared_interface_profile(
        tmp_path: Path) -> None:
    """Curved stock/slim sites retain the intentional 0.14-mm base inset."""
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 20.0],
        [0.0, 0.0, -1.0, 18.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    site = {
        **_release_site_contract((1.0, 0.0, 0.0)),
        "name": "stock_upper_left_receiver",
        "closure_kind": "transverse_gable_45deg",
        "cavity_bury_roof_start_print_z_mm": 5.80,
        "roof_apex_print_z_mm": 8.40,
        "cavity_center_xyz_mm": [-18.38, 420.37, 15.10],
        "seated_magnet_center_xyz_mm": [-18.33, 420.37, 15.10],
        "actual_face_xyz_mm": [-17.88, 420.37, 15.10],
        "material_inward_xyz": [1.0, 0.0, 0.0],
        "marked_pole_axis_xyz": [1.0, 0.0, 0.0],
        "interface_profile": "standard_curved",
        "carrier_cavity_face_inset_mm": 0.14,
        "paired_magnet_face_separation_mm": 1.23,
    }
    artifact = {
        "id": "floor_stand:A:curved",
        "state": "floor_stand",
        "variant": "A",
        "part": "shoulder_top_left",
        "stl": "shoulder_top_left.stl",
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": 0.0},
        "source_to_stl_matrix": matrix,
        "sites": [site],
    }
    payload = _catalog_document([artifact])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    normalized = audit.normalize_catalog(
        path, enforce_release_inventory=False)
    assert normalized["artifacts"][0]["sites"][0][
        "paired_magnet_face_separation_mm"] == 1.23

    payload["artifacts"][0]["sites"][0][
        "paired_magnet_face_separation_mm"] = 0.95
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        audit.normalize_catalog(path, enforce_release_inventory=False)
    except audit.AuditError as exc:
        assert (
            "paired magnet-face separation must be 1.230" in str(exc)
            or "paired_magnet_face_separation_mm: value is not in enum"
            in str(exc))
    else:
        raise AssertionError("stale 0.95-mm standard curved spacing passed")


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
        assert "58 artifacts / 94 captive stations" in str(exc)
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
    wing_site = _minimal_catalog_site()
    wing_site.update({
        "interface_kind": "ring",
        "carrier_cavity_face_inset_mm": 0.15,
        "carrier_cavity_datum_xy_mm": [0.0, 0.0],
        "outer_surface_face_xy_mm": [0.0, 0.15],
        "paired_magnet_face_separation_mm": 1.24,
    })
    raw = {
        "id": "shared:Obi-Wan-Flat:wing", "part": "wing",
        "variant": "Obi-Wan-Flat", "state": "shared", "stl": "wing.stl",
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": 0.0},
        "source_to_stl_matrix": [
            [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "sites": [wing_site],
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
        raise AssertionError("tampered flat/graded facts remained release-bound")


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
                    if item["id"] == "floor:Obi-Wan:mono")
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
    monolith = by_id["floor:Obi-Wan:mono"]
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
                "actual": {"bambu_studio_pause_marker_z_mm": 5.96},
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
