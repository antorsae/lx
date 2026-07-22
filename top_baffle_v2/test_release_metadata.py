#!/usr/bin/env python3
"""Pure static gates for released front-down metadata and Make wiring."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
from unittest import mock

import write_obiwan_release_manifest as release_manifest
import check_manifold as manifold_checker
from check_manifold import (
    EXPECTED_NONPOLAR_STATE_STL_COUNT,
    EXPECTED_WING_STL_COUNT,
    FLOOR_POLAR_SIDECAR_EXCLUSIONS,
    _print_sidecar_inventory_errors,
    expected_wing_stl_names,
    stl_diagnostics,
)
from front_down_contract import (
    RELEASE_ACOUSTIC_PRINT_CONTRACT,
    write_print_sidecar,
)


ROOT = Path(__file__).resolve().parent


def _stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_catalog_generator_without_cad():
    """Load only generator control flow with geometry authorities stubbed.

    The transaction tests must remain runnable on the Mac without importing
    build123d/OCC.  Geometry helpers are referenced only when ``generate`` is
    invoked, so inert stubs are sufficient for testing publication semantics.
    """
    module_name = "_catalog_generator_transaction_test"
    stubs = {
        "captive_magnets": _stub_module(
            "captive_magnets",
            DEFAULT_SPEC=object(),
            NOMINAL_PAIRED_FACE_SEPARATION_MM=0.95,
            axial_cavity_tools=lambda **_kwargs: None,
            wall_cavity_tools=lambda **_kwargs: None,
        ),
        "top_baffle_nd25fw4": _stub_module(
            "top_baffle_nd25fw4", THICKNESS_MM=18.3),
        "top_baffle_nd25fw4_b": _stub_module(
            "top_baffle_nd25fw4_b",
            BASE_CAVITY_FACE_INSET_MM=(0.0, 0.14),
            MAGNET_SITES=(),
        ),
        "top_baffle_nd25fw4_v0": _stub_module(
            "top_baffle_nd25fw4_v0", V0_MAGNET_SITES=()),
        "top_baffle_nd25fw4_v1": _stub_module(
            "top_baffle_nd25fw4_v1", V1_MAGNET_ZC=()),
        "top_baffle_nd25fw4_obiwan": _stub_module(
            "top_baffle_nd25fw4_obiwan",
            SIDE_INTERFACE_GAP=0.05,
            side_magnet_sites=lambda *_args, **_kwargs: (),
        ),
    }
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "generate_captive_magnet_catalog.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**stubs, module_name: module}):
        spec.loader.exec_module(module)
    return module


def _load_piece_mesh_sanitizer_without_cad():
    """Load the binary-STL repair functions without importing build123d.

    ``export_piece_stls`` imports OCC at module scope.  The apex sanitizer is
    intentionally pure binary processing, so extract only it and its binary
    validator from the source AST for a Mac-safe regression test.
    """
    path = ROOT / "export_piece_stls.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {"_validate_binary_stl", "_remove_collapsed_apex_facets"}
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    extracted = ast.fix_missing_locations(ast.Module(
        body=definitions, type_ignores=[]))
    namespace = {"Path": Path, "struct": struct}
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace["_remove_collapsed_apex_facets"]


def _load_coupon_mesh_gate_without_cad():
    """Extract the coupon STL transaction helpers without importing OCC."""
    path = ROOT / "export_coupon.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {
        "_validate_binary_stl",
        "_canonicalize_transform_zeros",
        "_strict_mesh_facts",
    }
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    extracted = ast.fix_missing_locations(ast.Module(
        body=definitions, type_ignores=[]))
    namespace = {
        "Path": Path,
        "struct": struct,
        "STL_TRANSFORM_ZERO_EPSILON_MM": 2.0e-7,
    }
    exec(compile(extracted, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in (
        "_canonicalize_transform_zeros", "_strict_mesh_facts"))


def _load_ts_nudge_without_cad():
    """Extract the shared captive TS detour without build123d/OCC."""
    path = ROOT / "top_baffle_nd25fw4_cables.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = {
        "TS_ROUTE_STANDARD",
        "TS_ROUTE_CAPTIVE",
        "TS_CAPTIVE_NUDGE_MAX_MM",
        "TS_CAPTIVE_NUDGE_KNOTS",
    }
    functions = {"_smoothstep01", "_ts_captive_nudge_mm"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & constants:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
    extracted = ast.fix_missing_locations(ast.Module(
        body=body, type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(extracted, str(path), "exec"), namespace)
    assert constants | functions <= namespace.keys()
    return namespace


def _logical_make_lines(text: str) -> tuple[str, ...]:
    lines = []
    current = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if current:
            current += " " + stripped
        else:
            current = stripped
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        lines.append(current)
        current = ""
    if current:
        lines.append(current)
    return tuple(lines)


def test_obiwan_release_manifest_binds_print_sidecars() -> None:
    expected_obiwan = {
        "stl/lx521_top_obiwan_core_1of2_lm_carrier.print.json",
        "stl/lx521_top_obiwan_core_2of2_um_carrier.print.json",
        "stl/lx521_top_obiwan_optional_lm_keyed_1of2_bottom.print.json",
        "stl/lx521_top_obiwan_optional_lm_keyed_2of2_top.print.json",
        "stl/lx521_top_obiwan_addon_tweeter_crescent.print.json",
        *(
            f"stl/lx521_coupon_{name}.print.json"
            for name in (
                "1_fit_plate",
                "2_fit_key",
                "3_fish_entry",
                "4_um_outlet_proud",
                "5_fish_ts_dive",
                "6_fish_foot",
                "7_recess_seat",
                "8_fish_ts_oval_proud",
                "9_um_faston_clocking",
                "12_obiwan_closed_bore_bump",
            )
        ),
    }
    for stand_foot in (False, True):
        names = set(release_manifest.expected_artifact_names(stand_foot))
        actual_sidecars = {name for name in names if name.endswith(".print.json")}
        assert actual_sidecars == expected_obiwan
        assert len(actual_sidecars) == 15
        assert len(names) == 42
        stls = {
            name for name in names
            if name.startswith("stl/") and name.endswith(".stl")
        }
        assert actual_sidecars == {
            name.removesuffix(".stl") + ".print.json" for name in stls
        }
        blocker_base = (
            "support_blockers/"
            "lx521_top_obiwan_optional_lm_keyed_1of2_bottom."
            "support_blocker")
        blocker_artifacts = {
            name for name in names if name.startswith("support_blockers/")
        }
        assert blocker_artifacts == (
            {f"{blocker_base}.stl", f"{blocker_base}.json"}
            if not stand_foot else set())
    assert release_manifest.FORMAT_VERSION == 10
    assert (ROOT / "front_down_contract.py") in (
        release_manifest.generation_source_paths())


def test_catalog_schema_requires_x180_plus_numeric_z() -> None:
    schema = json.loads((
        ROOT / "captive_magnet_release_catalog.schema.json"
    ).read_text(encoding="utf-8"))
    assert "schema_sha256" in schema["required"]
    assert "source_revision" in schema["required"]
    assert schema["properties"]["source_revision"] == {
        "type": "string", "pattern": "^[0-9a-f]{64}$"}
    print_space = schema["$defs"]["printSpace"]
    assert "seated_magnet_center_xyz_mm" in print_space["required"]
    assert "insertion_direction_xyz" in print_space["required"]
    site = schema["$defs"]["site"]
    assert "seated_magnet_center_xyz_mm" in site["properties"]
    assert {
        "magnet_diameter_mm", "magnet_depth_mm",
        "cavity_diameter_mm", "cavity_depth_mm",
        "face_skin_mm", "inner_skin_mm", "captive_land_mm",
        "interface_gap_mm", "paired_magnet_face_separation_mm",
        "roof_angle_deg", "minimum_retaining_path_mm",
        "polarity_instruction", "installed_marked_pole_axis_xyz",
        "insertion_direction_xyz", "magnet_count",
        "structural_load_credit_n", "print_space",
    } <= set(site["required"])
    assert site["properties"]["magnet_count"] == {"const": 1}
    assert site["properties"]["structural_load_credit_n"] == {"const": 0.0}
    artifact = schema["$defs"]["artifact"]
    assert {"stl_sha256", "print_sidecar", "print_sidecar_sha256",
            "source_files", "source_file_sha256"} <= set(
        artifact["required"])
    assert artifact["properties"]["source_file_sha256"] == {
        "type": "object",
        "additionalProperties": {
            "type": "string", "pattern": "^[0-9a-f]{64}$"},
    }
    assert "rotation_deg" in artifact["required"]
    rotation = artifact["properties"]["rotation_deg"]
    assert rotation["additionalProperties"] is False
    assert set(rotation["required"]) == {"x", "z"}
    assert rotation["properties"]["x"] == {"const": 180.0}
    assert rotation["properties"]["z"] == {"type": "number"}
    print_contract = schema["properties"]["print_contract"]
    assert print_contract["additionalProperties"] is False
    assert set(print_contract["required"]) == set(
        RELEASE_ACOUSTIC_PRINT_CONTRACT)
    assert {
        key: value["const"]
        for key, value in print_contract["properties"].items()
    } == RELEASE_ACOUSTIC_PRINT_CONTRACT
    assert set(schema["$defs"]["cavityAuditProxy"]["required"]) == {
        "site", "artifact_id", "proxy_site"}
    assert artifact["properties"]["p2s_printability"] == {
        "enum": ["not_printable_oversize"]}
    assert artifact["properties"]["cavity_audit_proxies"][
        "items"] == {"$ref": "#/$defs/cavityAuditProxy"}
    assert {
        "catalog_kind", "generated_by", "print_contract", "geometry",
        "inventory", "exclusions", "source_revision",
    } <= set(schema["required"])
    conditional_requirements = {
        tuple(item.get("then", {}).get("required", ()))
        for item in artifact["allOf"]
    }
    assert ("stage_manifest", "stage_manifest_sha256") in (
        conditional_requirements)
    assert (
        "transaction_manifest", "transaction_manifest_sha256",
        "facts", "facts_sha256",
    ) in conditional_requirements


def test_catalog_global_pair_spacing_is_not_ambiguous() -> None:
    payload = json.loads((
        ROOT / "review" / "captive_magnet_release_catalog.json"
    ).read_text(encoding="utf-8"))
    geometry = payload["geometry"]
    assert "paired_magnet_face_separation_mm" not in geometry
    assert "nominal_paired_magnet_face_separation_mm" not in geometry
    assert geometry[
        "paired_magnet_face_separation_by_interface_profile_mm"
    ] == {
        "standard_straight": 0.95,
        "standard_curved": 1.09,
        "obiwan_base_side": 0.95,
        "obiwan_ring": 1.10,
    }


def _release_catalog() -> dict[str, object]:
    path = ROOT / "review" / "captive_magnet_release_catalog.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert isinstance(payload.get("artifacts"), list)
    return payload


def _one_toleranced_value(
    values: list[tuple[float, str]], *, label: str, tolerance: float = 1.0e-6,
) -> float:
    """Require one numeric release datum while retaining useful failures."""
    assert values, f"{label}: no release values"
    reference = values[0][0]
    drift = [
        (value, owner) for value, owner in values
        if not math.isclose(
            value, reference, abs_tol=tolerance, rel_tol=0.0)
    ]
    assert not drift, (
        f"{label}: expected one value, reference={reference:.9f}, "
        f"drift={drift}")
    return reference


def test_transverse_magnet_plane_is_uniform_per_design_family() -> None:
    """Stock, slim and Obi-Wan each publish one insertion/closure plane.

    Grouping by the user-facing design family is deliberate: attachments,
    split alternatives, both stand states and Ac/Ae receivers must not
    silently retain a stale lower/upper or LM/UM depth.  V0 is axial and the
    fit coupon is not an assembled baffle, so neither belongs to this gate.
    """
    payload = _release_catalog()
    family_variants = {
        "stock": {"A", "B1", "B2", "C7"},
        "slim": {"V1", "V1-A", "V1-B1", "V1L"},
        "obiwan": {
            "Obi-Wan", "Obi-Wan-split", "Obi-Wan-Ac", "Obi-Wan-Ae",
        },
    }
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    for family, variants in family_variants.items():
        selected = [
            artifact for artifact in artifacts
            if isinstance(artifact, dict) and artifact.get("variant") in variants
        ]
        assert {artifact["variant"] for artifact in selected} == variants
        fields = {
            "source seated-magnet Z": [],
            "print-space seated-magnet Z": [],
            "raw roof-start print Z": [],
            "snapped roof-start print Z": [],
        }
        for artifact in selected:
            artifact_id = str(artifact["id"])
            sites = artifact.get("sites")
            assert isinstance(sites, list) and sites, artifact_id
            for site in sites:
                assert isinstance(site, dict), artifact_id
                if site.get("closure_kind") != "transverse_gable_45deg":
                    continue
                owner = f"{artifact_id}/{site.get('name')}"
                source_center = site["seated_magnet_center_xyz_mm"]
                print_center = site["print_space"][
                    "seated_magnet_center_xyz_mm"]
                fields["source seated-magnet Z"].append(
                    (float(source_center[2]), owner))
                fields["print-space seated-magnet Z"].append(
                    (float(print_center[2]), owner))
                fields["raw roof-start print Z"].append(
                    (float(site["raw_roof_start_print_z_mm"]), owner))
                fields["snapped roof-start print Z"].append((
                    float(site["cavity_bury_roof_start_print_z_mm"]), owner,
                ))
        for field, values in fields.items():
            _one_toleranced_value(values, label=f"{family} {field}")


def _mirrored_name(name: str) -> tuple[str, str] | None:
    """Return (side, side-neutral name) for one released site name."""
    tokens = name.split("_")
    side_tokens = [token for token in tokens if token in {"left", "right"}]
    if not side_tokens:
        return None
    assert len(side_tokens) == 1, f"ambiguous side in site name {name!r}"
    side = side_tokens[0]
    neutral = "_".join("SIDE" if token == side else token for token in tokens)
    return side, neutral


def _assert_source_mirror(
    left: list[float], right: list[float], *, label: str,
    tolerance: float = 1.0e-6,
) -> None:
    assert len(left) == len(right) in {2, 3}, label
    assert math.isclose(
        float(left[0]), -float(right[0]),
        abs_tol=tolerance, rel_tol=0.0,
    ), f"{label}: X is not mirrored: left={left}, right={right}"
    for axis in range(1, len(left)):
        assert math.isclose(
            float(left[axis]), float(right[axis]),
            abs_tol=tolerance, rel_tol=0.0,
        ), f"{label}: axis {axis} drift: left={left}, right={right}"


def test_every_released_magnet_site_has_exact_left_right_symmetry() -> None:
    """Mirror all source-space magnet geometry, not only nominal centres."""
    payload = _release_catalog()
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    paired: dict[tuple[str, str, str, str], dict[str, dict[str, object]]] = {}
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        for site in artifact["sites"]:
            assert isinstance(site, dict)
            # V0's two rear-axis sites deliberately avoid different nearby
            # cable corridors; this regression governs the transverse flank
            # interfaces whose released parts are true left/right mirrors.
            if site.get("closure_kind") != "transverse_gable_45deg":
                continue
            parsed = _mirrored_name(str(site["name"]))
            if parsed is None:
                continue
            side, neutral = parsed
            key = (
                str(artifact["state"]), str(artifact["variant"]),
                str(site.get("owner")), neutral,
            )
            bucket = paired.setdefault(key, {})
            # A given released family/state may intentionally repeat the same
            # site in a monolith and a split/wing inventory only when its
            # variant identity differs.  Duplicate one-side records here
            # therefore indicate catalog ambiguity.
            assert side not in bucket, (key, side)
            bucket[side] = site

    assert paired
    vector_fields = (
        "interface_datum_xyz_mm",
        "actual_face_xyz_mm",
        "cavity_center_xyz_mm",
        "seated_magnet_center_xyz_mm",
        "pair_axis_xyz",
        "material_inward_xyz",
        "marked_pole_axis_xyz",
        "installed_marked_pole_axis_xyz",
        "carrier_cavity_datum_xy_mm",
        "outer_surface_face_xy_mm",
    )
    for key, by_side in paired.items():
        assert set(by_side) == {"left", "right"}, (key, sorted(by_side))
        left, right = by_side["left"], by_side["right"]
        for field in vector_fields:
            assert (field in left) == (field in right), (key, field)
            if field in left:
                _assert_source_mirror(
                    left[field], right[field], label=f"{key}/{field}")


def test_catalog_source_freezes_56_stls_and_102_stations() -> None:
    """Audit release arithmetic without importing build123d/OCC locally."""
    path = ROOT / "generate_captive_magnet_catalog.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {
            "EXPECTED_ARTIFACT_COUNT", "EXPECTED_MAGNET_COUNT",
            "EXPECTED_STATE_ARTIFACT_COUNT", "EXPECTED_STATE_MAGNET_COUNT",
            "EXPECTED_SHARED_ARTIFACT_COUNT", "EXPECTED_SHARED_MAGNET_COUNT",
            "EXPECTED_FAMILY_COUNTS",
        }
    }
    assert assignments["EXPECTED_ARTIFACT_COUNT"] == 56
    assert assignments["EXPECTED_MAGNET_COUNT"] == 102
    assert assignments["EXPECTED_STATE_ARTIFACT_COUNT"] == 22
    assert assignments["EXPECTED_STATE_MAGNET_COUNT"] == 45
    assert assignments["EXPECTED_SHARED_ARTIFACT_COUNT"] == 12
    assert assignments["EXPECTED_SHARED_MAGNET_COUNT"] == 12
    families = assignments["EXPECTED_FAMILY_COUNTS"]
    assert sum(counts[0] for counts in families.values()) == 56
    assert sum(counts[1] for counts in families.values()) == 102
    assert set(families) == {
        "B2", "C7", "A", "B1", "V0", "V1", "V1-A", "V1-B1",
        "V1L", "Obi-Wan", "Obi-Wan-split", "Obi-Wan-Ac", "Obi-Wan-Ae",
        "coupon1",
    }


def test_wing_catalog_identity_preserves_frozen_release_case() -> None:
    """Filesystem slugs stay lowercase; released family IDs must not."""
    generator = _load_catalog_generator_without_cad()
    expected = {"ac": "Obi-Wan-Ac", "ae": "Obi-Wan-Ae"}
    assert generator.RELEASED_WING_VARIANTS == expected
    for slug, variant in expected.items():
        assert generator._released_wing_variant(slug) == variant
        assert variant in generator.EXPECTED_FAMILY_COUNTS
        assert variant != f"Obi-Wan-{slug}"
    try:
        generator._released_wing_variant("unknown")
    except RuntimeError as exc:
        assert "unsupported released wing slug" in str(exc)
    else:
        raise AssertionError("unknown wing slug entered the release catalog")

    source = (ROOT / "generate_captive_magnet_catalog.py").read_text(
        encoding="utf-8")
    assert '"id": f"shared:{wing_variant}:{entry[\'label\']}"' in source
    assert '"variant": wing_variant' in source
    assert 'f"Obi-Wan-{slug}"' not in source


def test_catalog_generator_uses_release_wide_acoustic_print_contract() -> None:
    source = (ROOT / "generate_captive_magnet_catalog.py").read_text(
        encoding="utf-8")
    assert "RELEASE_ACOUSTIC_PRINT_CONTRACT" in source
    assert '"print_contract": dict(RELEASE_ACOUSTIC_PRINT_CONTRACT)' in source


def test_coupon1_polarity_is_explicitly_unpaired_and_axis_specific() -> None:
    generator = _load_catalog_generator_without_cad()
    instruction = generator._polarity("base", "coupon1")
    assert "unpaired coupon1 regression station" in instruction
    assert "installed -Y" in instruction
    assert "print +Y" in instruction
    assert "no mating magnet" in instruction
    assert "no attraction claim" in instruction


def test_catalog_publication_validates_same_directory_candidate_then_replaces(
        ) -> None:
    generator = _load_catalog_generator_without_cad()
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        output = directory / "captive_magnet_release_catalog.json"
        old_bytes = b'{"release":"known-good"}\n'
        output.write_bytes(old_bytes)
        payload = {"release": "new", "finite": 1.0}
        state = {"validated": False, "replaced": False}
        real_replace = os.replace

        def validator(candidate: Path) -> None:
            assert candidate.parent == output.parent
            assert candidate != output
            assert candidate.name.endswith(".candidate")
            assert output.read_bytes() == old_bytes
            assert json.loads(candidate.read_text(encoding="utf-8")) == payload
            state["validated"] = True

        def checked_replace(source, destination) -> None:
            source = Path(source)
            destination = Path(destination)
            assert state["validated"] is True
            assert source.parent == destination.parent == output.parent
            state["replaced"] = True
            real_replace(source, destination)

        with mock.patch.object(
                generator.os, "replace", side_effect=checked_replace):
            generator._publish_validated_catalog(
                output, payload, validator=validator)

        assert state == {"validated": True, "replaced": True}
        assert json.loads(output.read_text(encoding="utf-8")) == payload
        assert not list(directory.glob(".*.candidate"))


def test_catalog_validation_failure_preserves_prior_authority() -> None:
    generator = _load_catalog_generator_without_cad()
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        output = directory / "captive_magnet_release_catalog.json"
        old_bytes = b'{"release":"known-good","sha":"fixed"}\n'
        output.write_bytes(old_bytes)
        observed_candidate: Path | None = None

        def reject(candidate: Path) -> None:
            nonlocal observed_candidate
            observed_candidate = candidate
            assert candidate.parent == output.parent
            assert output.read_bytes() == old_bytes
            raise RuntimeError("synthetic schema/binding failure")

        try:
            generator._publish_validated_catalog(
                output, {"release": "invalid"}, validator=reject)
        except RuntimeError as exc:
            assert "schema/binding" in str(exc)
        else:
            raise AssertionError("invalid catalog was published")

        assert output.read_bytes() == old_bytes
        assert observed_candidate is not None
        assert not observed_candidate.exists()
        assert not list(directory.glob(".*.candidate"))


def test_catalog_render_failure_preserves_prior_authority() -> None:
    generator = _load_catalog_generator_without_cad()
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        output = directory / "captive_magnet_release_catalog.json"
        old_bytes = b'{"release":"known-good"}\n'
        output.write_bytes(old_bytes)
        validator_called = False

        def should_not_validate(_candidate: Path) -> None:
            nonlocal validator_called
            validator_called = True

        try:
            generator._publish_validated_catalog(
                output, {"invalid": float("nan")},
                validator=should_not_validate)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite JSON was published")

        assert validator_called is False
        assert output.read_bytes() == old_bytes
        assert not list(directory.glob(".*.candidate"))


def test_catalog_candidate_mutation_during_validation_is_rejected() -> None:
    generator = _load_catalog_generator_without_cad()
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        output = directory / "captive_magnet_release_catalog.json"
        old_bytes = b'{"release":"known-good"}\n'
        output.write_bytes(old_bytes)

        def mutate(candidate: Path) -> None:
            candidate.write_text('{"release":"swapped"}\n', encoding="utf-8")

        try:
            generator._publish_validated_catalog(
                output, {"release": "new"}, validator=mutate)
        except RuntimeError as exc:
            assert "changed while" in str(exc)
        else:
            raise AssertionError("post-validation candidate mutation passed")

        assert output.read_bytes() == old_bytes
        assert not list(directory.glob(".*.candidate"))


def test_catalog_candidate_runs_normalize_then_every_binding_gate() -> None:
    generator = _load_catalog_generator_without_cad()
    with tempfile.TemporaryDirectory() as directory_name:
        candidate = Path(directory_name) / ".catalog.candidate"
        candidate.write_text("{}\n", encoding="utf-8")
        calls: list[tuple[str, object]] = []
        artifacts = [{"id": "a"}, {"id": "b"}]

        def normalize(path: Path) -> dict:
            calls.append(("normalize", path))
            return {"artifacts": artifacts}

        def bind(artifact: dict) -> None:
            calls.append(("binding", artifact["id"]))

        slicer = _stub_module(
            "slice_captive_magnets",
            normalize_catalog=normalize,
            _validate_artifact_bindings=bind,
        )
        with mock.patch.dict(sys.modules, {"slice_captive_magnets": slicer}):
            normalized = generator._validate_catalog_candidate(candidate)

        assert normalized == {"artifacts": artifacts}
        assert calls == [
            ("normalize", candidate),
            ("binding", "a"),
            ("binding", "b"),
        ]


def test_catalog_source_revision_is_mandatory_immutable_snapshot_hash() -> None:
    generator = _load_catalog_generator_without_cad()
    with mock.patch.dict(os.environ, {}, clear=True):
        try:
            generator._source_revision()
        except RuntimeError as exc:
            assert "LX_CAD_SOURCE_SHA256" in str(exc)
        else:
            raise AssertionError("catalog accepted missing source revision")
    revision = "a" * 64
    with mock.patch.dict(
            os.environ, {"LX_CAD_SOURCE_SHA256": revision}, clear=True):
        assert generator._source_revision() == revision
    with mock.patch.dict(
            os.environ, {"LX_CAD_SOURCE_SHA256": "not-a-hash"}, clear=True):
        try:
            generator._source_revision()
        except RuntimeError:
            pass
        else:
            raise AssertionError("catalog accepted malformed source revision")


def test_catalog_source_file_hash_map_exactly_covers_source_files() -> None:
    generator = _load_catalog_generator_without_cad()
    with tempfile.TemporaryDirectory() as directory_name:
        root = Path(directory_name)
        generator.HERE = root
        (root / "one.py").write_bytes(b"one\n")
        (root / "nested").mkdir()
        (root / "nested" / "two.json").write_bytes(b'{"two":2}\n')
        output = root / "review" / "catalog.json"
        source_files, hashes = generator._source_provenance(
            ("one.py", "nested/two.json", "one.py"), output)
        assert source_files == ["../one.py", "../nested/two.json"]
        assert set(hashes) == set(source_files)
        assert hashes["../one.py"] == generator._sha256(root / "one.py")
        assert hashes["../nested/two.json"] == generator._sha256(
            root / "nested" / "two.json")


def test_obiwan_catalog_binds_staged_exporter_and_state_manifest() -> None:
    source = (ROOT / "generate_captive_magnet_catalog.py").read_text(
        encoding="utf-8")
    assert '"export_obiwan_staged.py"' in source
    assert 'f"{state}/.obiwan_stage/manifest.json"' in source
    assert "stage_manifest=stage_manifest" in source
    assert '"stage_manifest_sha256": _sha256(manifest_path)' in source


def test_receiver_polarity_docs_match_pair_axis_convention() -> None:
    for relative in (
        "top_baffle_nd25fw4_b.py",
        "top_baffle_nd25fw4_v1_attachments.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8").lower()
        assert "marked pole in" not in text
        assert "marked poles point in" not in text


def test_obiwan_release_target_requires_captive_catalog() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        line.split(":", 1)[0]: line.split(":", 1)[1]
        for line in _logical_make_lines(makefile)
        if ":" in line and not line.startswith(("#", "\t"))
    }
    assert "$(CAPTIVE_MAGNET_CATALOG)" in targets["obiwan_release"]


def test_release_metadata_waits_for_current_captive_catalog() -> None:
    """The static consumer must not race the generated catalog in candidate."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        line.split(":", 1)[0]: line.split(":", 1)[1]
        for line in _logical_make_lines(makefile)
        if ":" in line and not line.startswith(("#", "\t"))
    }
    assert "$(CAPTIVE_MAGNET_CATALOG)" in targets["check_release_metadata"]


def test_check_waits_for_both_obiwan_native_stages() -> None:
    """Parallel R6F keyed-split checks must never race stage creation."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        line.split(":", 1)[0]: line.split(":", 1)[1]
        for line in _logical_make_lines(makefile)
        if ":" in line and not line.startswith(("#", "\t"))
    }
    assert "validate_obiwan_stages" in targets["check"]
    assert "validate_obiwan_stages" in targets["check_obiwan"]


def test_r6f_carriers_reuse_make_stage_across_assertion_edits() -> None:
    """R6F assertions must never own a second carrier build/cache key.

    The Make prerequisites already publish source/runtime/hash-validated
    carrier BREPs.  Keep assertion code outside carrier identity entirely:
    an assertion-only edit may invalidate test-only shell chunks, but cannot
    rebuild LM, UM, or tweeter geometry.
    """
    module_name = "_r6f_stage_reuse_static_test"
    path = ROOT / "test_obiwan_r6f.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    r6f = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {module_name: r6f}):
        spec.loader.exec_module(r6f)

    # Exercise the adapter itself without importing OCC or requiring a built
    # stage.  This freezes both state selection and the two authenticated
    # manifest APIs: R6F must not regress to a raw path read or a private
    # carrier exporter when an assertion changes.
    import export_obiwan_staged as staged_export

    for stand_foot, state_name in (
            (False, "no_floor_stand"), (True, "floor_stand")):
        manifest = ROOT / state_name / ".obiwan_stage/manifest.json"
        payload = {"authenticated": state_name}
        resolved = {"core_lm_carrier": Path(f"/{state_name}/lm.brep")}
        with mock.patch.object(r6f, "_state") as select_state, \
                mock.patch.object(
                    staged_export, "load_stage_manifest",
                    return_value=payload) as load_manifest, \
                mock.patch.object(
                    staged_export, "staged_part_paths",
                    return_value=resolved) as resolve_parts:
            assert r6f._validated_obiwan_stage_paths(stand_foot) is resolved
        select_state.assert_called_once_with(stand_foot)
        load_manifest.assert_called_once_with(
            manifest, stand_foot=stand_foot,
            require_active_environment=False)
        resolve_parts.assert_called_once_with(manifest, payload)

    stage = {
        "core_lm_carrier": Path("/validated/core_lm_carrier.brep"),
        "core_um_carrier": Path("/validated/core_um_carrier.brep"),
        "addon_tweeter_crescent": Path(
            "/validated/addon_tweeter_crescent.brep"),
    }
    expected = {
        "lm": stage["core_lm_carrier"],
        "um": stage["core_um_carrier"],
        "tweeter": stage["addon_tweeter_crescent"],
    }
    with mock.patch.object(
            r6f, "_validated_obiwan_stage_paths", return_value=stage), \
            mock.patch.object(
                r6f, "_stage_shell_contract_breps_unlocked",
                side_effect=AssertionError("carrier path entered shell CAD")):
        assert r6f._stage_shell_contract_breps(
            False, "LM", Path("unused"), shell_keys=()) == expected

    captured = {}

    def fake_shell_stage(
            stand_foot, route_name, directory, shell_keys, *, seed_targets):
        captured.update({
            "stand_foot": stand_foot,
            "route_name": route_name,
            "directory": directory,
            "shell_keys": shell_keys,
            "seed_targets": dict(seed_targets),
        })
        return {**seed_targets, "shell_nominal": (Path("shell.brep"),)}

    with tempfile.TemporaryDirectory() as directory_name, \
            mock.patch.object(
                r6f, "_validated_obiwan_stage_paths", return_value=stage), \
            mock.patch.object(
                r6f, "_stage_shell_contract_breps_unlocked",
                side_effect=fake_shell_stage), \
            mock.patch.object(
                r6f.tempfile, "gettempdir", return_value=directory_name):
        result = r6f._stage_shell_contract_breps(
            True, "UM", Path("unused"), shell_keys=("nominal",))
    assert captured["seed_targets"] == expected
    assert captured["shell_keys"] == ("nominal",)
    assert result["lm"] == stage["core_lm_carrier"]

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    unlocked = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_stage_shell_contract_breps_unlocked")
    unlocked_source = ast.get_source_segment(source, unlocked)
    assert unlocked_source is not None
    assert "LX_R6F_EXPORT_CARRIER" not in unlocked_source
    assert "LX_R6F_EXPORT_TWEETER" not in unlocked_source
    assert "carrier_cache" not in unlocked_source

    # The authenticated stage identity and its Make producers are geometry
    # authorities, not assertion authorities.  Otherwise changing only this
    # test would still force the very carrier rebuild eliminated above.
    staged_source = (ROOT / "export_obiwan_staged.py").read_text(
        encoding="utf-8")
    staged_tree = ast.parse(
        staged_source, filename=str(ROOT / "export_obiwan_staged.py"))
    fingerprint = next(
        node for node in staged_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_source_fingerprint")
    fingerprint_source = ast.get_source_segment(staged_source, fingerprint)
    assert fingerprint_source is not None
    assert 'glob("top_baffle_nd25fw4*.py")' in fingerprint_source
    assert "test_obiwan_r6f.py" not in fingerprint_source

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        line.split(":", 1)[0]: line.split(":", 1)[1]
        for line in _logical_make_lines(makefile)
        if ":" in line and not line.startswith(("#", "\t"))
    }
    for target in (
            "validate_floor_obiwan_stage",
            "validate_no_floor_obiwan_stage"):
        assert "$(OBIWAN_SRCS)" in targets[target]
        assert "test_obiwan_r6f.py" not in targets[target]


def test_coupon_mesh_inventory_excludes_print_sidecars() -> None:
    source = (ROOT / "check_manifold.py").read_text(encoding="utf-8")
    assert 'and name.endswith(".stl")' in source


def test_manifold_cli_splits_mesh_and_metadata_phases() -> None:
    facts = {
        "triangles": 4,
        "open": 0,
        "over_shared": 0,
        "winding": 0,
        "degenerate": 0,
        "duplicates": 0,
        "nonfinite": 0,
        "signed_volume": 1.0,
        "zero_volume": 0,
        "negative_volume": 0,
        "components": 1,
        "nested_void_components": 0,
        "component_error": 0,
    }
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        mesh = root / "part.stl"
        mesh.write_bytes(b"placeholder")
        with mock.patch.object(
                manifold_checker, "stl_diagnostics", return_value=facts), \
                mock.patch.object(
                    manifold_checker, "_obiwan_manifest_errors",
                    side_effect=AssertionError("STL-only read metadata")), \
                mock.patch.object(
                    sys, "argv",
                    ["check_manifold.py", "--stl-only", str(mesh)]):
            assert manifold_checker.main() == 0

        with mock.patch.object(
                manifold_checker, "stl_diagnostics",
                side_effect=AssertionError("metadata-only read a mesh")), \
                mock.patch.object(
                    manifold_checker, "_obiwan_manifest_errors",
                    return_value=[]), \
                mock.patch.object(
                    sys, "argv",
                    ["check_manifold.py", "--metadata-only", str(root)]):
            assert manifold_checker.main() == 0


def test_physical_reference_coupon_freezes_zero_interface_gap() -> None:
    """Do not let the production 0.05-mm gap mutate the tested coupon."""
    path = ROOT / "coupons/obiwan_ae_embed/obiwan_ae_embed_coupon.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "COUPON_INTERFACE_GAP_MM"
    }
    assert constants == {"COUPON_INTERFACE_GAP_MM": 0.0}
    assert "MAGNET_FACE_GAP_MM" not in source
    assert source.count("COUPON_INTERFACE_GAP_MM") >= 8
    readme = (path.parent / "README.md").read_text(encoding="utf-8")
    assert "coupon magnets by 0.90 mm" in readme
    assert "1.10 mm nominal magnet-to-magnet separation" in readme
    assert "LM-lower base-side pair" in readme
    assert "0.95 mm" in readme


def _rotation_keywords(path: Path, function_name: str) -> tuple[tuple[str, object], ...]:
    """Return every build123d ``Rot`` keyword used by one exporter helper.

    This is deliberately an AST-only gate: it does not import build123d/OCC,
    but it prevents a future packing optimization from silently restoring an
    X/Y tilt in any of the four released/coupon print pipelines.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    rotations = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "Rot":
            continue
        assert not node.args, f"{path.name}:{function_name}: positional Rot"
        assert len(node.keywords) == 1, (
            f"{path.name}:{function_name}: compound Rot is not allowed")
        keyword = node.keywords[0]
        assert keyword.arg in {"X", "Z"}, (
            f"{path.name}:{function_name}: only X180 and Z are allowed")
        value = (
            keyword.value.value
            if isinstance(keyword.value, ast.Constant)
            else ast.unparse(keyword.value)
        )
        rotations.append((keyword.arg, value))
    return tuple(rotations)


def test_every_export_pipeline_is_x180_plus_z_only() -> None:
    pipelines = (
        (ROOT / "export_piece_stls.py", "main", True),
        (ROOT / "export_obiwan_wings.py", "_best_print_orientation", True),
        (ROOT / "export_coupon.py", "_front_face_down", False),
        (ROOT / "coupons/obiwan_ae_embed/obiwan_ae_embed_coupon.py",
         "_front_down", False),
    )
    for path, function_name, allow_z in pipelines:
        rotations = _rotation_keywords(path, function_name)
        assert rotations.count(("X", 180.0)) == 1, (
            f"{path.name}:{function_name}: missing exact X180 front-down")
        assert all(axis == "X" or (allow_z and axis == "Z")
                   for axis, _value in rotations), (
            f"{path.name}:{function_name}: out-of-plane rotation returned")
        assert sum(axis == "X" for axis, _value in rotations) == 1


def test_docs_do_not_describe_generated_stls_front_up() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "print-ready pieces (flat, z = thickness, front face up)" not in readme


def test_ts_captive_detour_is_smooth_full_lumen_and_wired() -> None:
    """Mac-safe gate for the stock/slim lower-left land repair."""
    contract = _load_ts_nudge_without_cad()
    nudge = contract["_ts_captive_nudge_mm"]
    knots = contract["TS_CAPTIVE_NUDGE_KNOTS"]
    maximum = contract["TS_CAPTIVE_NUDGE_MAX_MM"]

    assert maximum == 0.60
    assert tuple(offset for _y, offset in knots) == (
        0.0, 0.3, 0.6, 0.6, 0.3, 0.0)
    for y, expected in knots:
        assert abs(nudge(y) - expected) < 1.0e-12
    assert nudge(knots[0][0] - 100.0) == 0.0
    assert nudge(knots[-1][0] + 100.0) == 0.0

    samples = [
        nudge(knots[0][0]
              + (knots[-1][0] - knots[0][0]) * index / 1000.0)
        for index in range(1001)
    ]
    assert min(samples) >= 0.0
    assert max(samples) <= maximum + 1.0e-12
    epsilon = 1.0e-4
    assert nudge(knots[0][0] + epsilon) / epsilon < 1.0e-4
    assert nudge(knots[-1][0] - epsilon) / epsilon < 1.0e-4

    cables = (ROOT / "top_baffle_nd25fw4_cables.py").read_text(
        encoding="utf-8")
    split = (ROOT / "top_baffle_nd25fw4_b2_split.py").read_text(
        encoding="utf-8")
    v1 = (ROOT / "top_baffle_nd25fw4_v1_split.py").read_text(
        encoding="utf-8")
    v1l = (ROOT / "top_baffle_nd25fw4_v1l_split.py").read_text(
        encoding="utf-8")
    assert "x + _ts_captive_nudge_mm(y)" in cables
    assert "ts_route_key=ts_route_key" in cables
    assert "ts_route_key=ts_route_key" in split
    assert "ts_route_key=TS_ROUTE_CAPTIVE" in v1
    assert "ts_route_key=TS_ROUTE_CAPTIVE" in v1l
    assert "ts_route_key: str = TS_ROUTE_CAPTIVE" in split
    # No cavity land is fused into the conduit: the repair is centerline-only
    # and `_ts_cutter` continues deriving every section from `ts_section`.
    assert "w2, h2, zc = ts_section(py)" in cables


def test_wing_review_uses_captive_cavity_schema() -> None:
    source = (ROOT / "export_obiwan_wings.py").read_text(
        encoding="utf-8")
    assert 'receiver["cavity_diameter_mm"]' in source
    assert 'receiver["cavity_depth_mm"]' in source
    assert 'receiver["pocket_diameter_mm"]' not in source
    assert 'receiver["pocket_depth_mm"]' not in source


def test_ae_unions_overlapping_relief_tools_before_final_cut() -> None:
    source = (ROOT / "obiwan_wings_cad.py").read_text(encoding="utf-8")
    assert "combined_cutter = relief_cutter.fuse(edge_cutter)" in source
    assert "carved = blank - combined_cutter" in source
    assert "carved = carved - edge_cutter" not in source


def test_docs_reject_fake_p2s_monolith_pauses() -> None:
    slicing = (ROOT / "CAPTIVE_MAGNET_SLICING.md").read_text(encoding="utf-8")
    printing = (ROOT / "PRINTING.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    combined = " ".join("\n".join((slicing, printing, readme)).lower().split())
    assert combined.count("not p2s-printable") >= 5
    assert "no monolith g-code and no fake pause row" in combined
    assert "no monolith pause is synthesized" in combined
    assert "exact same-state keyed halves" in combined
    assert "virtual bed" in combined
    assert "front-face-down" in combined


def _cube_triangles(origin, size=1.0, *, inward=False):
    ox, oy, oz = origin
    vertices = (
        (ox, oy, oz), (ox + size, oy, oz),
        (ox + size, oy + size, oz), (ox, oy + size, oz),
        (ox, oy, oz + size), (ox + size, oy, oz + size),
        (ox + size, oy + size, oz + size), (ox, oy + size, oz + size),
    )
    indices = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    )
    triangles = tuple(tuple(vertices[index] for index in triangle)
                      for triangle in indices)
    if inward:
        triangles = tuple((triangle[0], triangle[2], triangle[1])
                          for triangle in triangles)
    return triangles


def _write_binary_stl(path: Path, triangles) -> None:
    payload = bytearray(b"pure mesh topology fixture".ljust(80, b"\0"))
    payload.extend(struct.pack("<I", len(triangles)))
    for triangle in triangles:
        payload.extend(struct.pack("<3f", 0.0, 0.0, 0.0))
        payload.extend(struct.pack(
            "<9f", *(coordinate for vertex in triangle for coordinate in vertex)))
        payload.extend(struct.pack("<H", 0))
    path.write_bytes(payload)


def _write_valid_front_down_sidecar(stl: Path) -> Path:
    return write_print_sidecar(
        stl,
        part=stl.stem,
        transform={
            "print_orientation": "front_face_down",
            "rotation_deg": {"x": 180.0, "z": 0.0},
            "source_to_stl_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "pre_translation_bbox_min_mm": [0.0, 0.0, 0.0],
            "stl_origin_translation_mm": [0.0, 0.0, 0.0],
        },
    )


def test_release_sidecars_fail_closed() -> None:
    """Missing/orphaned/stale/tilted/contradictory records all fail."""
    triangles = _cube_triangles((0.0, 0.0, 0.0))
    expected = {"fixture.stl"}

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)

        def fresh_case(name: str) -> tuple[Path, Path, Path]:
            root = directory / name
            root.mkdir()
            stl = root / "fixture.stl"
            _write_binary_stl(stl, triangles)
            sidecar = _write_valid_front_down_sidecar(stl)
            return root, stl, sidecar

        root, _stl, _sidecar = fresh_case("valid")
        assert _print_sidecar_inventory_errors(root, expected) == []

        root, _stl, sidecar = fresh_case("missing")
        sidecar.unlink()
        errors = _print_sidecar_inventory_errors(root, expected)
        assert len(errors) == 1 and "missing adjacent print sidecars" in errors[0]

        root, _stl, _sidecar = fresh_case("extra")
        (root / "orphan.print.json").write_text("{}\n", encoding="utf-8")
        errors = _print_sidecar_inventory_errors(root, expected)
        assert len(errors) == 1 and "stale/extra print sidecars" in errors[0]

        root, stl, _sidecar = fresh_case("stale")
        content = bytearray(stl.read_bytes())
        content[0] ^= 1
        stl.write_bytes(content)
        errors = _print_sidecar_inventory_errors(root, expected)
        assert len(errors) == 1 and "stl_sha256 does not match" in errors[0]

        root, _stl, sidecar = fresh_case("wrong_transform")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["rotation_deg"]["x"] = 179.0
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        errors = _print_sidecar_inventory_errors(root, expected)
        assert len(errors) == 1 and "X rotation must be exactly 180" in errors[0]

        root, _stl, sidecar = fresh_case("wrong_translation")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["stl_origin_translation_mm"][0] = 1.0
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        errors = _print_sidecar_inventory_errors(root, expected)
        assert len(errors) == 1
        assert "matrix translation disagrees" in errors[0]


def test_sidecar_inventory_exact_counts_and_only_polar_exclusion() -> None:
    assert EXPECTED_NONPOLAR_STATE_STL_COUNT == 45
    assert EXPECTED_WING_STL_COUNT == 6
    assert FLOOR_POLAR_SIDECAR_EXCLUSIONS == {
        "lx521_polar_base_1of2_base.stl",
        "lx521_polar_base_2of2_rotor.stl",
    }
    for slug in ("ac", "ae"):
        names = expected_wing_stl_names(slug)
        assert len(names) == EXPECTED_WING_STL_COUNT
        assert len({Path(name).with_suffix(".print.json").name
                    for name in names}) == EXPECTED_WING_STL_COUNT

    with tempfile.TemporaryDirectory() as directory_name:
        root = Path(directory_name)
        ordinary = root / "fixture.stl"
        _write_binary_stl(ordinary, _cube_triangles((0.0, 0.0, 0.0)))
        _write_valid_front_down_sidecar(ordinary)
        for name in FLOOR_POLAR_SIDECAR_EXCLUSIONS:
            _write_binary_stl(
                root / name, _cube_triangles((0.0, 0.0, 0.0)))
        expected = {ordinary.name, *FLOOR_POLAR_SIDECAR_EXCLUSIONS}
        assert _print_sidecar_inventory_errors(
            root, expected,
            excluded_stl_names=FLOOR_POLAR_SIDECAR_EXCLUSIONS) == []
        # A polar sidecar is not silently accepted: excluded means no
        # acoustic/front-down authority record exists for that jig.
        polar = root / next(iter(FLOOR_POLAR_SIDECAR_EXCLUSIONS))
        _write_valid_front_down_sidecar(polar)
        errors = _print_sidecar_inventory_errors(
            root, expected,
            excluded_stl_names=FLOOR_POLAR_SIDECAR_EXCLUSIONS)
        assert len(errors) == 1 and "stale/extra print sidecars" in errors[0]


def test_mesh_gate_accepts_only_nested_inward_cavity_shells() -> None:
    """A buried cavity is a void boundary, never a second material body."""
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        single = directory / "single.stl"
        nested = directory / "nested.stl"
        separated = directory / "separated.stl"
        outside = directory / "outside.stl"
        outer = _cube_triangles((0.0, 0.0, 0.0), 4.0)
        inner = _cube_triangles((1.0, 1.0, 1.0), 1.0, inward=True)
        _write_binary_stl(single, outer)
        _write_binary_stl(nested, outer + inner)
        _write_binary_stl(
            separated, outer + _cube_triangles((6.0, 0.0, 0.0), 1.0))
        _write_binary_stl(
            outside,
            outer + _cube_triangles((6.0, 0.0, 0.0), 1.0, inward=True))

        single_facts = stl_diagnostics(single)
        assert single_facts["component_error"] == 0
        assert single_facts["components"] == 1
        nested_facts = stl_diagnostics(nested)
        assert nested_facts["component_error"] == 0
        assert nested_facts["components"] == 2
        assert nested_facts["outer_components"] == 1
        assert nested_facts["nested_void_components"] == 1
        assert nested_facts["nonnested_void_components"] == 0
        separated_facts = stl_diagnostics(separated)
        assert separated_facts["component_error"] == 1
        assert separated_facts["disconnected_material_components"] == 1
        outside_facts = stl_diagnostics(outside)
        assert outside_facts["component_error"] == 1
        assert outside_facts["nonnested_void_components"] == 1


def test_collapsed_apex_sanitizer_is_lossless_and_fail_closed() -> None:
    """Remove only OCC's zero-area cone-apex record, then gate strictly."""
    sanitize = _load_piece_mesh_sanitizer_without_cad()
    strict_keys = (
        "open", "over_shared", "winding", "degenerate", "duplicates",
        "nonfinite", "zero_volume", "negative_volume", "component_error",
    )
    cube = _cube_triangles((0.0, 0.0, 0.0), 4.0)
    # The bottom-face diagonal is already owned by exactly two real facets.
    # OCC's cone-apex artifact repeats one endpoint, so its two remaining
    # directed edges raise that real edge's exact count from two to four.
    a, b = cube[0][0], cube[0][1]
    collapsed = (a, b, b)

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        apex = directory / "collapsed-apex.stl"
        _write_binary_stl(apex, (*cube, collapsed))
        original = apex.read_bytes()
        before = stl_diagnostics(apex)
        assert before["degenerate"] == 1
        assert before["over_shared"] == 1

        assert sanitize(apex) == 1
        repaired = apex.read_bytes()
        assert repaired[:80] == original[:80]
        assert struct.unpack_from("<I", repaired, 80)[0] == len(cube)
        assert repaired[84:] == original[84:84 + 50 * len(cube)]
        after = stl_diagnostics(apex)
        assert not any(after[key] for key in strict_keys), after
        assert after["signed_volume"] == before["signed_volume"]
        assert sanitize(apex) == 0
        assert apex.read_bytes() == repaired

        # Three distinct collinear vertices are not the known OCC apex
        # encoding.  Preserve them so the downstream strict gate rejects the
        # malformed mesh rather than broadening this into generic healing.
        collinear = directory / "distinct-collinear.stl"
        line_facet = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
        _write_binary_stl(collinear, (*cube, line_facet))
        collinear_bytes = collinear.read_bytes()
        assert sanitize(collinear) == 0
        assert collinear.read_bytes() == collinear_bytes
        assert stl_diagnostics(collinear)["degenerate"] == 1

        malformed = directory / "malformed-length.stl"
        malformed.write_bytes(repaired + b"x")
        try:
            sanitize(malformed)
        except RuntimeError as exc:
            assert "transaction invalid" in str(exc)
        else:
            raise AssertionError("malformed STL length was silently repaired")

    # Export ordering is part of the safety argument: normalize only rigid
    # transform zeros first, remove exact collapsed records, then run the
    # unchanged strict topology gate before atomic publication.
    source = (ROOT / "export_piece_stls.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    wanted_calls = (
        "export_stl", "_validate_binary_stl",
        "_canonicalize_transform_zeros", "_remove_collapsed_apex_facets",
        "_strict_mesh_facts", "replace",
    )
    call_lines: dict[str, list[int]] = {name: [] for name in wanted_calls}
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute)
                else None)
        if (name == "replace"
                and not (isinstance(node.func, ast.Attribute)
                         and isinstance(node.func.value, ast.Name)
                         and node.func.value.id == "temporary")):
            continue
        if name in call_lines:
            call_lines[name].append(node.lineno)
    assert all(len(call_lines[name]) == 1 for name in wanted_calls), call_lines
    ordered = [call_lines[name][0] for name in wanted_calls]
    assert ordered == sorted(ordered), dict(zip(wanted_calls, ordered))


def test_coupon_export_canonicalizes_only_transform_zeros_then_gates() -> None:
    """Rx180 roundoff is normalized before every coupon is published."""
    canonicalize, strict = _load_coupon_mesh_gate_without_cad()
    cube = list(_cube_triangles((0.0, 0.0, 0.0), 4.0))

    def with_noisy_first_x(value: float):
        triangles = list(cube)
        triangle = list(triangles[0])
        vertex = list(triangle[0])
        assert vertex[0] == 0.0
        vertex[0] = value
        triangle[0] = tuple(vertex)
        triangles[0] = tuple(triangle)
        return tuple(triangles)

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        repairable = directory / "rx180-roundoff.stl"
        _write_binary_stl(repairable, with_noisy_first_x(1.0e-8))
        before = stl_diagnostics(repairable)
        assert before["open"] > 0
        original_size = repairable.stat().st_size
        assert canonicalize(repairable) == 1
        assert repairable.stat().st_size == original_size
        after = strict(repairable)
        assert after["open"] == 0
        assert after["over_shared"] == 0
        assert canonicalize(repairable) == 0

        # Coordinates outside the 0.2-nm transform neighbourhood are real
        # mesh data: preserve them and let the unchanged strict gate fail.
        too_large = directory / "real-offset.stl"
        _write_binary_stl(too_large, with_noisy_first_x(5.0e-7))
        original = too_large.read_bytes()
        assert canonicalize(too_large) == 0
        assert too_large.read_bytes() == original
        try:
            strict(too_large)
        except RuntimeError as exc:
            assert "strict manifold contract" in str(exc)
            assert "open" in str(exc)
        else:
            raise AssertionError("non-transform mesh offset was silently healed")

        try:
            canonicalize(repairable, epsilon_mm=0.0)
        except ValueError as exc:
            assert "must be positive" in str(exc)
        else:
            raise AssertionError("nonpositive transform epsilon was accepted")

    # The publication transaction must validate bytes, canonicalize only the
    # rigid-transform zero neighbourhood, run every strict topology check,
    # and only then replace the prior artifact.  No coupon-name exemption is
    # permitted in the directory-wide release checker.
    coupon_path = ROOT / "export_coupon.py"
    tree = ast.parse(coupon_path.read_text(encoding="utf-8"))
    exporter = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_export_group")
    wanted_calls = (
        "export_stl", "_validate_binary_stl",
        "_canonicalize_transform_zeros", "_strict_mesh_facts", "replace",
    )
    call_lines: dict[str, list[int]] = {name: [] for name in wanted_calls}
    for node in ast.walk(exporter):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute)
                else None)
        if (name == "replace"
                and not (isinstance(node.func, ast.Attribute)
                         and isinstance(node.func.value, ast.Name)
                         and node.func.value.id == "temporary")):
            continue
        if name in call_lines:
            call_lines[name].append(node.lineno)
    assert all(len(call_lines[name]) == 1 for name in wanted_calls), call_lines
    ordered = [call_lines[name][0] for name in wanted_calls]
    assert ordered == sorted(ordered), dict(zip(wanted_calls, ordered))
    checker = (ROOT / "check_manifold.py").read_text(encoding="utf-8")
    for coupon in ("2_fit_key", "4_um_outlet_proud", "7_recess_seat"):
        assert coupon not in checker
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    coupon_target = next(
        line for line in _logical_make_lines(makefile)
        if line.startswith("$(1)/stl/.stamp_coupon:"))
    assert "export_coupon.py" in coupon_target
    assert "check_manifold.py" in coupon_target
    assert "write_obiwan_release_manifest.py" in coupon_target


def test_routing_metadata_keeps_human_name_and_machine_slug_distinct() -> None:
    generator = (ROOT / "gen_cable_routing.py").read_text(encoding="utf-8")
    checker = (ROOT / "check_manifold.py").read_text(encoding="utf-8")
    assert '"Obi-Wan" if ROUTING_PROFILE == "obiwan"' in generator
    assert '("Obi-Wan", "obiwan", "R6F")' in checker
    assert 'f"LX_ROUTING_PROFILE={profile_slug}"' in checker


def main() -> None:
    tests = (
        test_obiwan_release_manifest_binds_print_sidecars,
        test_catalog_schema_requires_x180_plus_numeric_z,
        test_catalog_global_pair_spacing_is_not_ambiguous,
        test_transverse_magnet_plane_is_uniform_per_design_family,
        test_every_released_magnet_site_has_exact_left_right_symmetry,
        test_catalog_source_freezes_56_stls_and_102_stations,
        test_wing_catalog_identity_preserves_frozen_release_case,
        test_catalog_generator_uses_release_wide_acoustic_print_contract,
        test_coupon1_polarity_is_explicitly_unpaired_and_axis_specific,
        test_catalog_publication_validates_same_directory_candidate_then_replaces,
        test_catalog_validation_failure_preserves_prior_authority,
        test_catalog_render_failure_preserves_prior_authority,
        test_catalog_candidate_mutation_during_validation_is_rejected,
        test_catalog_candidate_runs_normalize_then_every_binding_gate,
        test_catalog_source_revision_is_mandatory_immutable_snapshot_hash,
        test_catalog_source_file_hash_map_exactly_covers_source_files,
        test_obiwan_catalog_binds_staged_exporter_and_state_manifest,
        test_receiver_polarity_docs_match_pair_axis_convention,
        test_obiwan_release_target_requires_captive_catalog,
        test_release_metadata_waits_for_current_captive_catalog,
        test_check_waits_for_both_obiwan_native_stages,
        test_r6f_carriers_reuse_make_stage_across_assertion_edits,
        test_coupon_mesh_inventory_excludes_print_sidecars,
        test_manifold_cli_splits_mesh_and_metadata_phases,
        test_physical_reference_coupon_freezes_zero_interface_gap,
        test_every_export_pipeline_is_x180_plus_z_only,
        test_docs_do_not_describe_generated_stls_front_up,
        test_ts_captive_detour_is_smooth_full_lumen_and_wired,
        test_wing_review_uses_captive_cavity_schema,
        test_ae_unions_overlapping_relief_tools_before_final_cut,
        test_docs_reject_fake_p2s_monolith_pauses,
        test_release_sidecars_fail_closed,
        test_sidecar_inventory_exact_counts_and_only_polar_exclusion,
        test_mesh_gate_accepts_only_nested_inward_cavity_shells,
        test_collapsed_apex_sanitizer_is_lossless_and_fail_closed,
        test_coupon_export_canonicalizes_only_transform_zeros_then_gates,
        test_routing_metadata_keeps_human_name_and_machine_slug_distinct,
    )
    for test in tests:
        test()
    print(f"release metadata: {len(tests)} pure gates pass")


if __name__ == "__main__":
    main()
