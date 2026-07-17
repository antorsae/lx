#!/usr/bin/env python3
"""Generate the authoritative released captive-magnet STL catalog.

The catalog is deliberately generated *after* every STEP-first artifact has
been meshed.  It binds source-space cavity facts to the exact front-face-down
STL transform recorded by each exporter.  ``slice_captive_magnets.py`` then
uses this file without importing OCC and derives pauses from real Bambu
G-code rather than from nominal CAD heights.

Run only through the remote-first Make target; this module imports build123d
geometry helpers but never constructs a complete baffle solid.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from captive_magnets import (
    DEFAULT_SPEC,
    NOMINAL_PAIRED_FACE_SEPARATION_MM,
    axial_cavity_tools,
    wall_cavity_tools,
)
from front_down_contract import (
    FrontDownContractError,
    RELEASE_ACOUSTIC_PRINT_CONTRACT,
    validate_front_down_transform,
    validate_print_sidecar,
)
from top_baffle_nd25fw4 import THICKNESS_MM
from top_baffle_nd25fw4_b import MAGNET_SITES
from top_baffle_nd25fw4_v0 import V0_MAGNET_SITES
from top_baffle_nd25fw4_v1 import V1_MAGNET_ZC
from top_baffle_nd25fw4_v1lf import (
    SIDE_INTERFACE_GAP,
    side_magnet_sites,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "review" / "captive_magnet_release_catalog.json"
SCHEMA_PATH = HERE / "captive_magnet_release_catalog.schema.json"
SCHEMA_VERSION = 1
EXPECTED_ARTIFACT_COUNT = 56
EXPECTED_MAGNET_COUNT = 102
EXPECTED_STATE_ARTIFACT_COUNT = 22
EXPECTED_STATE_MAGNET_COUNT = 45
EXPECTED_SHARED_ARTIFACT_COUNT = 12
EXPECTED_SHARED_MAGNET_COUNT = 12
EXPECTED_FAMILY_COUNTS = {
    # family: (released STL count, total captive-station count)
    "B2": (2, 8),
    "C7": (2, 8),
    "A": (8, 8),
    "B1": (4, 8),
    "V0": (2, 4),
    "V1": (2, 8),
    "V1-A": (8, 8),
    "V1-B1": (4, 8),
    "V1L": (2, 8),
    "V1LF": (4, 12),
    "V1LF-split": (4, 8),
    "V1LF-Ac": (6, 6),
    "V1LF-Ae": (6, 6),
    "coupon1": (2, 2),
}
RELEASED_WING_VARIANTS = {
    "ac": "V1LF-Ac",
    "ae": "V1LF-Ae",
}

SOURCE_REVISION_ENV = "LX_CAD_SOURCE_SHA256"


def _released_wing_variant(slug: str) -> str:
    """Return the frozen release-family spelling for one wing slug."""
    try:
        variant = RELEASED_WING_VARIANTS[slug]
    except KeyError as exc:
        raise RuntimeError(f"unsupported released wing slug: {slug!r}") from exc
    if variant not in EXPECTED_FAMILY_COUNTS:
        raise RuntimeError(
            f"wing slug {slug!r} maps outside frozen release inventory: "
            f"{variant!r}")
    return variant


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_revision() -> str:
    """Return the immutable remote snapshot identity or fail closed."""
    revision = os.environ.get(SOURCE_REVISION_ENV, "").strip()
    if re.fullmatch(r"[0-9a-f]{64}", revision) is None:
        raise RuntimeError(
            f"{SOURCE_REVISION_ENV} must contain the 64-hex immutable "
            "remote source snapshot identity")
    return revision


def _render_catalog_candidate(path: Path, payload: Any) -> Path:
    """Render one same-directory candidate without touching ``path``.

    The candidate deliberately lives beside the authoritative destination so
    its eventual ``os.replace`` cannot cross filesystems.  Strict JSON and a
    read-back comparison make serialization itself part of the release gate.
    The caller owns the returned path and must either publish or remove it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, candidate_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".candidate",
        text=True,
    )
    candidate = Path(candidate_name)
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1  # ``stream`` now owns and always closes this fd.
        with stream:
            json.dump(
                payload, stream, indent=2, sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if _read_json(candidate) != payload:
            raise RuntimeError(f"catalog round-trip failed: {path}")
        return candidate
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_catalog_candidate(candidate: Path) -> dict[str, Any]:
    """Run the complete fail-closed consumer contract on ``candidate``."""
    from slice_captive_magnets import (
        _validate_artifact_bindings,
        normalize_catalog,
    )

    # ``normalize_catalog`` validates the checked-in schema, schema digest,
    # release inventory, source/print-space facts, and cross-artifact rules.
    # The candidate is in the destination directory, so all catalog-relative
    # artifact paths resolve exactly as they will after publication.
    normalized = normalize_catalog(candidate)
    for artifact in normalized["artifacts"]:
        _validate_artifact_bindings(artifact)
    return normalized


def _publish_validated_catalog(
        output: Path,
        payload: Any,
        *,
        validator: Callable[[Path], Any] | None = None) -> None:
    """Validate a staged catalog, then atomically publish it.

    Any render, schema, inventory, or artifact-binding failure removes only
    the candidate.  A previously released catalog therefore remains byte-for-
    byte intact.  ``validator`` is injectable solely so the transaction can be
    exercised by pure tests without importing any CAD geometry authority.
    """
    candidate = _render_catalog_candidate(output, payload)
    try:
        candidate_sha256 = _sha256(candidate)
        (validator or _validate_catalog_candidate)(candidate)
        if _sha256(candidate) != candidate_sha256:
            raise RuntimeError(
                "catalog candidate changed while it was being validated")
        os.replace(candidate, output)
    finally:
        candidate.unlink(missing_ok=True)


def _relative(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.parent.resolve())


def _source_provenance(
        source_files: Sequence[str], output: Path,
) -> tuple[list[str], dict[str, str]]:
    """Bind every artifact source path to its exact generation-time bytes."""
    unique = tuple(dict.fromkeys(source_files))
    if not unique:
        raise RuntimeError("artifact source provenance must not be empty")
    paths = [HERE / name for name in unique]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing source provenance: {missing}")
    relatives = [_relative(path, output) for path in paths]
    if len(relatives) != len(set(relatives)):
        raise RuntimeError("duplicate resolved source provenance path")
    hashes = {
        relative: _sha256(path)
        for relative, path in zip(relatives, paths)
    }
    if set(hashes) != set(relatives):
        raise RuntimeError("source hash coverage differs from source_files")
    return relatives, hashes


def _resolve_release_relative(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{label}: path must be a nonempty string")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"{label}: unsafe relative path: {relative}")
    resolved_root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise RuntimeError(f"{label}: path escapes release root: {relative}")
    return root / relative_path


def _transform_point(matrix: Sequence[Sequence[float]],
                     point: Sequence[float]) -> list[float]:
    return [
        sum(float(matrix[row][column]) * float(point[column])
            for column in range(3)) + float(matrix[row][3])
        for row in range(3)
    ]


def _transform_vector(matrix: Sequence[Sequence[float]],
                      vector: Sequence[float]) -> list[float]:
    return [
        sum(float(matrix[row][column]) * float(vector[column])
            for column in range(3))
        for row in range(3)
    ]


def _add_print_space(site: Mapping[str, Any],
                     matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Add schema-required post-export datums without losing source facts."""
    result = dict(site)
    required = (
        "cavity_center_xyz_mm", "seated_magnet_center_xyz_mm",
        "marked_pole_axis_xyz", "insertion_direction_xyz",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise RuntimeError(
            f"site {result.get('name')} lacks source datums: {missing}")
    print_space = {
        "cavity_center_xyz_mm": _transform_point(
            matrix, result["cavity_center_xyz_mm"]),
        "seated_magnet_center_xyz_mm": _transform_point(
            matrix, result["seated_magnet_center_xyz_mm"]),
        "marked_pole_axis_xyz": _transform_vector(
            matrix, result["marked_pole_axis_xyz"]),
        "insertion_direction_xyz": _transform_vector(
            matrix, result["insertion_direction_xyz"]),
    }
    for key in ("actual_face_xyz_mm", "material_inward_xyz"):
        if key in result:
            print_space[key] = (
                _transform_vector(matrix, result[key])
                if key.endswith("axis_xyz") or key == "material_inward_xyz"
                else _transform_point(matrix, result[key]))
    result["print_space"] = print_space
    return result


def _polarity(owner: str, family: str) -> str:
    if family == "v0":
        return (
            "provisional unpaired V0 convention: marked/N pole points rearward "
            "along installed_marked_pole_axis_xyz; verify any future mate "
            "before burial"
        )
    if family == "coupon1":
        return (
            "unpaired coupon1 regression station: marked/N pole points "
            "along installed -Y (print +Y under its X180/Z0 transform); "
            "there is no mating magnet and no attraction claim"
        )
    if owner in {"base", "carrier"}:
        return (
            "marked/N pole points OUT from carrier toward its mating piece, "
            "along installed_marked_pole_axis_xyz"
        )
    return (
        "marked/N pole points along the same installed pair-axis vector as "
        "the carrier magnet; the face toward the carrier is therefore the "
        "opposite pole"
    )


def _site_record(tools, *, family: str,
                 expected_pause: float | None = None) -> dict[str, Any]:
    record = dict(tools.facts())
    record["installed_marked_pole_axis_xyz"] = list(tools.pair_axis_xyz)
    record["polarity_instruction"] = _polarity(tools.owner, family)
    record["magnet_count"] = 1
    record["structural_load_credit_n"] = 0.0
    if expected_pause is not None:
        record["expected_pause_marker_z_mm"] = expected_pause
    return record


def _standard_sites(*, owner: str, z_centres: Sequence[float],
                    family: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for index, (x, y, nx, ny, _pin, _released_z) in enumerate(MAGNET_SITES):
        vertical = "lower" if index == 0 else "upper"
        zc = float(z_centres[index])
        for side, sign in (("right", 1.0), ("left", -1.0)):
            tools = wall_cavity_tools(
                name=f"{family}_{vertical}_{side}_{owner}",
                face=(sign * x, y, zc),
                outward=(sign * nx, ny, 0.0),
                owner=owner,
                print_up=(0.0, 0.0, -1.0),
                bed_datum=(0.0, 0.0, THICKNESS_MM),
            )
            records[f"{vertical}_{side}"] = _site_record(
                tools, family=family)
    return records


def _v0_sites() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if set(V0_MAGNET_SITES) != {"right", "left"}:
        raise RuntimeError("V0 released site inventory changed")
    for side, (x, y) in V0_MAGNET_SITES.items():
        tools = axial_cavity_tools(
            name=f"v0_rear_axis_{side}",
            face=(x, y, 0.0),
            inward=(0.0, 0.0, 1.0),
            pair_axis=(0.0, 0.0, -1.0),
            print_up=(0.0, 0.0, -1.0),
            bed_datum=(0.0, 0.0, THICKNESS_MM),
        )
        records[side] = _site_record(tools, family="v0")
    return records


def _v1lf_sites(*, owner: str, driver: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for site in side_magnet_sites(driver):
        tools = wall_cavity_tools(
            name=str(site["name"]),
            face=site["face"],
            outward=(*site["normal"], 0.0),
            owner=owner,
            axis_z=float(site["z_mm"]),
            print_up=(0.0, 0.0, -1.0),
            front_z=THICKNESS_MM,
            interface_gap_mm=SIDE_INTERFACE_GAP,
        )
        expected = 8.52 if driver == "lm" else 5.96
        records[str(site["name"])] = _site_record(
            tools, family="v1lf", expected_pause=expected)
    return records


def _coupon1_site() -> dict[str, Any]:
    tools = wall_cavity_tools(
        name="coupon1_v1_upper_base",
        face=(28.0, 0.0, 14.4),
        outward=(0.0, -1.0, 0.0),
        owner="base",
        print_up=(0.0, 0.0, -1.0),
        bed_datum=(0.0, 0.0, THICKNESS_MM),
    )
    record = _site_record(tools, family="coupon1")
    if record["installed_marked_pole_axis_xyz"] != [0.0, -1.0, 0.0]:
        raise RuntimeError("coupon1 marked/N pole must remain installed -Y")
    return record


def _load_sidecar(stl: Path) -> dict[str, Any]:
    sidecar = stl.with_suffix(".print.json")
    try:
        payload = validate_print_sidecar(stl, sidecar)
    except (FrontDownContractError, OSError) as exc:
        raise RuntimeError(
            f"invalid front-down print sidecar: {sidecar}: {exc}") from exc
    return payload


def _artifact(*, output: Path, state: str, variant: str, part: str,
              stl: Path, sites: Iterable[Mapping[str, Any]],
              source_files: Sequence[str],
              print_exporter: str = "export_piece_stls.py",
              stage_manifest: str | None = None) -> dict[str, Any]:
    if print_exporter not in {"export_piece_stls.py", "export_coupon.py"}:
        raise RuntimeError(
            f"unsupported state print exporter: {print_exporter!r}")
    bound_sources = tuple(dict.fromkeys((
        *source_files, "front_down_contract.py", print_exporter,
        *((stage_manifest,) if stage_manifest is not None else ()),
    )))
    source_paths, source_hashes = _source_provenance(
        bound_sources, output)
    sidecar = _load_sidecar(stl)
    matrix = sidecar["source_to_stl_matrix"]
    site_list = [_add_print_space(site, matrix) for site in sites]
    if not site_list:
        raise RuntimeError(f"magnet artifact has no sites: {stl}")
    artifact = {
        "id": f"{state}:{variant}:{part}",
        "state": state,
        "variant": variant,
        "part": part,
        "stl": _relative(stl, output),
        "stl_sha256": _sha256(stl),
        "print_sidecar": _relative(stl.with_suffix(".print.json"), output),
        "print_sidecar_sha256": _sha256(stl.with_suffix(".print.json")),
        "print_orientation": "front_face_down",
        "rotation_deg": sidecar["rotation_deg"],
        "source_to_stl_matrix": matrix,
        "sites": site_list,
        "source_files": source_paths,
        "source_file_sha256": source_hashes,
    }
    if stage_manifest is not None:
        manifest_path = HERE / stage_manifest
        artifact.update({
            "stage_manifest": _relative(manifest_path, output),
            "stage_manifest_sha256": _sha256(manifest_path),
        })
    return artifact


def _state_artifacts(state: str, output: Path) -> list[dict[str, Any]]:
    root = HERE / state / "stl"
    standard_base = _standard_sites(
        owner="base", z_centres=[site[5] for site in MAGNET_SITES],
        family="standard")
    standard_receiver = _standard_sites(
        owner="receiver", z_centres=[site[5] for site in MAGNET_SITES],
        family="standard")
    v1_base = _standard_sites(
        owner="base", z_centres=V1_MAGNET_ZC, family="v1")
    v1_receiver = _standard_sites(
        owner="receiver", z_centres=V1_MAGNET_ZC, family="v1")
    v0 = _v0_sites()
    v1lf_lm = _v1lf_sites(owner="carrier", driver="lm")
    v1lf_um = _v1lf_sites(owner="carrier", driver="um")

    result: list[dict[str, Any]] = []

    def add(variant: str, stem: str, selected: Iterable[Mapping[str, Any]],
            sources: Sequence[str], *,
            print_exporter: str = "export_piece_stls.py",
            stage_manifest: str | None = None) -> dict[str, Any]:
        artifact = _artifact(
            output=output, state=state, variant=variant, part=stem,
            stl=root / f"{stem}.stl", sites=selected,
            source_files=sources, print_exporter=print_exporter,
            stage_manifest=stage_manifest)
        result.append(artifact)
        return artifact

    base_sources = ("captive_magnets.py", "top_baffle_nd25fw4_b.py")
    add("B2", "lx521_top_base_4of4_vase_b2",
        standard_base.values(),
        (*base_sources, "top_baffle_nd25fw4_b2.py",
         "top_baffle_nd25fw4_b2_split.py"))
    add("C7", "lx521_top_c7base_4of4_vase_b2",
        standard_base.values(),
        (*base_sources, "top_baffle_nd25fw4_b2.py",
         "top_baffle_nd25fw4_c7.py",
         "top_baffle_nd25fw4_c7_split.py"))

    a_map = (
        ("lx521_top_addonA_1of4_shoulder_top_left", "upper_left"),
        ("lx521_top_addonA_2of4_shoulder_top_right", "upper_right"),
        ("lx521_top_addonA_3of4_shoulder_bottom_left", "lower_left"),
        ("lx521_top_addonA_4of4_shoulder_bottom_right", "lower_right"),
    )
    for stem, key in a_map:
        add("A", stem, (standard_receiver[key],),
            (*base_sources, "top_baffle_nd25fw4_b2.py",
             "top_baffle_nd25fw4_a_comp.py",
             "top_baffle_nd25fw4_attachments.py"))
    for side in ("left", "right"):
        add("B1", f"lx521_top_addonB1_{1 if side == 'left' else 2}of2_wing_{side}",
            (standard_receiver[f"lower_{side}"],
             standard_receiver[f"upper_{side}"]),
            (*base_sources, "top_baffle_nd25fw4_b1.py",
             "top_baffle_nd25fw4_b2.py",
             "top_baffle_nd25fw4_attachments.py"))

    add("V0", "lx521_top_v0_4of4_vase", v0.values(),
        ("captive_magnets.py", "top_baffle_nd25fw4_b2.py",
         "top_baffle_nd25fw4_v0.py",
         "top_baffle_nd25fw4_v0_split.py"))
    add("V1", "lx521_top_v1_4of4_vase", v1_base.values(),
        (*base_sources, "top_baffle_nd25fw4_b2.py",
         "top_baffle_nd25fw4_v1.py",
         "top_baffle_nd25fw4_v1_split.py"))
    for stem, key in (
        ("lx521_top_v1addonA_shoulder_top_left", "upper_left"),
        ("lx521_top_v1addonA_shoulder_top_right", "upper_right"),
        ("lx521_top_v1addonA_shoulder_bottom_left", "lower_left"),
        ("lx521_top_v1addonA_shoulder_bottom_right", "lower_right"),
    ):
        add("V1-A", stem, (v1_receiver[key],),
            (*base_sources, "top_baffle_nd25fw4_a_comp.py",
             "top_baffle_nd25fw4_b2.py",
             "top_baffle_nd25fw4_v1.py",
             "top_baffle_nd25fw4_v1_attachments.py"))
    for side in ("left", "right"):
        add("V1-B1", f"lx521_top_v1addonB1_wing_{side}",
            (v1_receiver[f"lower_{side}"], v1_receiver[f"upper_{side}"]),
            (*base_sources, "top_baffle_nd25fw4_b1.py",
             "top_baffle_nd25fw4_b2.py",
             "top_baffle_nd25fw4_v1.py",
             "top_baffle_nd25fw4_v1_attachments.py"))
    add("V1L", "lx521_top_v1l_4of4_vase_b2", v1_base.values(),
        (*base_sources, "top_baffle_nd25fw4_b2.py",
         "top_baffle_nd25fw4_v1.py", "top_baffle_nd25fw4_v1l.py",
         "top_baffle_nd25fw4_v1l_split.py"))

    lm_lower = tuple(v1lf_lm[name] for name in (
        "lm_lower_left", "lm_lower_right"))
    lm_upper = tuple(v1lf_lm[name] for name in (
        "lm_upper_left", "lm_upper_right"))
    v1lf_stage_manifest = f"{state}/.v1lf_stage/manifest.json"
    v1lf_sources = (
        "captive_magnets.py", "top_baffle_nd25fw4_v1lf.py",
        "export_v1lf_staged.py",
    )
    lm_monolith = add("V1LF", "lx521_top_v1lf_core_1of2_lm_carrier",
        (*lm_lower, *lm_upper),
        v1lf_sources, stage_manifest=v1lf_stage_manifest)
    add("V1LF", "lx521_top_v1lf_core_2of2_um_carrier",
        v1lf_um.values(),
        v1lf_sources, stage_manifest=v1lf_stage_manifest)
    add("V1LF-split", "lx521_top_v1lf_optional_lm_keyed_1of2_bottom",
        lm_lower,
        (*v1lf_sources, "top_baffle_nd25fw4_v1lf_lm_split.py"),
        stage_manifest=v1lf_stage_manifest)
    add("V1LF-split", "lx521_top_v1lf_optional_lm_keyed_2of2_top",
        lm_upper,
        (*v1lf_sources, "top_baffle_nd25fw4_v1lf_lm_split.py"),
        stage_manifest=v1lf_stage_manifest)
    split_bottom_id = (
        f"{state}:V1LF-split:"
        "lx521_top_v1lf_optional_lm_keyed_1of2_bottom")
    split_top_id = (
        f"{state}:V1LF-split:"
        "lx521_top_v1lf_optional_lm_keyed_2of2_top")
    lm_monolith["p2s_printability"] = "not_printable_oversize"
    lm_monolith["cavity_audit_proxies"] = [
        {
            "site": name,
            "artifact_id": (
                split_bottom_id if name.startswith("lm_lower_")
                else split_top_id),
            "proxy_site": name,
        }
        for name in (
            "lm_lower_left", "lm_lower_right",
            "lm_upper_left", "lm_upper_right",
        )
    ]
    add("coupon1", "lx521_coupon_1_fit_plate", (_coupon1_site(),),
        ("captive_magnets.py", "top_baffle_nd25fw4_b2_split.py",
         "export_coupon.py"), print_exporter="export_coupon.py")

    if len(result) != 22:
        raise RuntimeError(f"{state}: expected 22 magnet STLs, got {len(result)}")
    return result


def _wing_artifacts(slug: str, output: Path) -> list[dict[str, Any]]:
    wing_variant = _released_wing_variant(slug)
    root = HERE / "wings" / slug
    facts_path = root / f"v1lf_wing_{slug}_facts.json"
    manifest_path = root / f"v1lf_wing_{slug}_print_manifest.json"
    facts = _read_json(facts_path)
    manifest = _read_json(manifest_path)
    if facts.get("schema_version") != 2:
        raise RuntimeError(f"{slug}: wing facts lack print-sidecar schema")
    if manifest.get("schema_version") != 2:
        raise RuntimeError(f"{slug}: wing manifest lacks print-sidecar schema")
    facts_sha = _sha256(facts_path)
    if manifest.get("facts_sha256") != facts_sha:
        raise RuntimeError(f"{slug}: stale transactional facts manifest")
    manifest_artifacts = {
        item["path"]: item for item in manifest.get("artifacts", ())
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    wing_source_files, wing_source_hashes = _source_provenance((
        "captive_magnets.py",
        "front_down_contract.py",
        "top_baffle_nd25fw4_v1lf.py",
        "gen_v1lf_basic_variants.py",
        "v1lf_basic_wings_cad.py",
        "export_v1lf_basic_wings.py",
    ), output)
    receivers = facts["geometry"]["interface_contract"]["receivers"]
    parts = facts["exports"]["print_parts"]
    if not isinstance(parts, list) or len(parts) != 6:
        raise RuntimeError(f"{slug}: expected six transactional wing parts")
    expected_stls = [item.get("path") for item in parts]
    expected_sidecars = [item.get("print_sidecar") for item in parts]
    if (any(not isinstance(path, str) or not path
            for path in (*expected_stls, *expected_sidecars))
            or len(set(expected_stls)) != 6
            or len(set(expected_sidecars)) != 6
            or manifest.get("print_parts") != expected_stls
            or manifest.get("print_sidecars") != expected_sidecars):
        raise RuntimeError(
            f"{slug}: facts/manifest do not bind six unique STL/sidecar pairs")
    by_key = {(item["side"], item["role"]): item for item in parts}
    result: list[dict[str, Any]] = []
    for side in ("left", "right"):
        receiver_by_name = {
            item["name"]: item for item in receivers[side]
        }
        for role in ("lm_lower", "lm_upper", "um"):
            entry = by_key[(side, role)]
            expected_name = (
                f"{role}_{side}" if role != "um" else f"um_{side}")
            site = dict(receiver_by_name[expected_name])
            site["installed_marked_pole_axis_xyz"] = site[
                "marked_pole_axis_xyz"]
            site["polarity_instruction"] = _polarity("wing", "v1lf")
            site["magnet_count"] = 1
            site["structural_load_credit_n"] = 0.0
            site["expected_pause_marker_z_mm"] = (
                8.52 if role.startswith("lm") else 5.96)
            stl = _resolve_release_relative(
                root, entry.get("path"), f"{slug}:{entry['label']} STL")
            transaction = manifest_artifacts.get(entry["path"])
            if (transaction is None
                    or transaction.get("kind") != "print_stl"
                    or transaction.get("sha256") != _sha256(stl)):
                raise RuntimeError(
                    f"{slug}: STL is not bound to its transaction: {stl}")
            transform = entry.get("print_transform")
            if not isinstance(transform, dict):
                raise RuntimeError(f"wing lacks front-down transform: {stl}")
            try:
                validate_front_down_transform(
                    transform, label=f"{slug}:{entry['label']}")
            except FrontDownContractError as exc:
                raise RuntimeError(
                    f"wing has invalid front-down transform: {stl}: "
                    f"{exc}") from exc
            sidecar_relative = entry.get("print_sidecar")
            sidecar = _resolve_release_relative(
                root, sidecar_relative,
                f"{slug}:{entry['label']} print sidecar")
            sidecar_transaction = manifest_artifacts.get(sidecar_relative)
            sidecar_sha = _sha256(sidecar) if sidecar.is_file() else None
            if (sidecar_transaction is None
                    or sidecar_transaction.get("kind") != "print_sidecar"
                    or sidecar_transaction.get("sha256") != sidecar_sha
                    or entry.get("print_sidecar_sha256") != sidecar_sha):
                raise RuntimeError(
                    f"{slug}: sidecar is not bound to its transaction: "
                    f"{sidecar}")
            try:
                sidecar_payload = validate_print_sidecar(stl, sidecar)
            except FrontDownContractError as exc:
                raise RuntimeError(
                    f"wing has invalid print sidecar: {sidecar}: {exc}") from exc
            if sidecar_payload.get("part") != stl.stem:
                raise RuntimeError(
                    f"{slug}: print sidecar part identity drifted: {sidecar}")
            if sidecar_payload.get("assembly_label") != entry["label"]:
                raise RuntimeError(
                    f"{slug}: print sidecar assembly identity drifted: "
                    f"{sidecar}")
            expected_release_metadata = {
                "artifact_family": "v1lf_basic_wings",
                "variant_slug": slug,
                "side": side,
                "order": entry.get("order"),
                "role": role,
                "mesh": {
                    "tolerance_mm": entry.get("mesh_tolerance_mm"),
                    "angular_tolerance": entry.get(
                        "mesh_angular_tolerance"),
                },
            }
            if {
                    key: sidecar_payload.get(key)
                    for key in expected_release_metadata
            } != expected_release_metadata:
                raise RuntimeError(
                    f"{slug}: print sidecar release metadata drifted: "
                    f"{sidecar}")
            sidecar_transform = {
                key: sidecar_payload.get(key) for key in transform
            }
            if sidecar_transform != transform:
                raise RuntimeError(
                    f"{slug}: print sidecar and facts transform differ: "
                    f"{sidecar}")
            matrix = sidecar_payload["source_to_stl_matrix"]
            site = _add_print_space(site, matrix)
            result.append({
                "id": f"shared:{wing_variant}:{entry['label']}",
                "state": "shared",
                "variant": wing_variant,
                "part": entry["label"],
                "stl": _relative(stl, output),
                "stl_sha256": _sha256(stl),
                "print_sidecar": _relative(sidecar, output),
                "print_sidecar_sha256": sidecar_sha,
                "transaction_manifest": _relative(manifest_path, output),
                "transaction_manifest_sha256": _sha256(manifest_path),
                "facts": _relative(facts_path, output),
                "facts_sha256": facts_sha,
                "print_orientation": "front_face_down",
                "rotation_deg": sidecar_payload["rotation_deg"],
                "source_to_stl_matrix": matrix,
                "sites": [site],
                "source_files": wing_source_files,
                "source_file_sha256": wing_source_hashes,
            })
    if len(result) != 6:
        raise RuntimeError(f"{slug}: expected six wing STLs, got {len(result)}")
    return result


def generate(output: Path) -> dict[str, Any]:
    source_revision = _source_revision()
    artifacts = [
        *_state_artifacts("floor_stand", output),
        *_state_artifacts("no_floor_stand", output),
        *_wing_artifacts("ac", output),
        *_wing_artifacts("ae", output),
    ]
    if len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise RuntimeError(
            f"release inventory must contain {EXPECTED_ARTIFACT_COUNT} "
            f"magnet-bearing STLs, got {len(artifacts)}")
    ids = [item["id"] for item in artifacts]
    paths = [item["stl"] for item in artifacts]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise RuntimeError("duplicate artifact ID or STL in release catalog")
    for artifact in artifacts:
        if artifact["print_orientation"] != "front_face_down":
            raise RuntimeError(f"orientation drift: {artifact['id']}")

    invalid_site_counts = [
        (artifact["id"], site.get("name"), site.get("magnet_count"))
        for artifact in artifacts
        for site in artifact["sites"]
        if site.get("magnet_count") != 1
    ]
    if invalid_site_counts:
        raise RuntimeError(
            "every released captive station must contain exactly one "
            f"magnet: {invalid_site_counts}")
    state_artifact_counts = Counter(item["state"] for item in artifacts)
    state_magnet_counts = Counter()
    family_artifact_counts = Counter(item["variant"] for item in artifacts)
    family_magnet_counts = Counter()
    for artifact in artifacts:
        count = sum(int(site["magnet_count"])
                    for site in artifact["sites"])
        state_magnet_counts[artifact["state"]] += count
        family_magnet_counts[artifact["variant"]] += count
    magnet_count = sum(state_magnet_counts.values())
    if magnet_count != EXPECTED_MAGNET_COUNT:
        raise RuntimeError(
            f"release inventory must contain {EXPECTED_MAGNET_COUNT} "
            f"per-STL captive magnet stations, got {magnet_count}")
    expected_state_artifacts = {
        "floor_stand": EXPECTED_STATE_ARTIFACT_COUNT,
        "no_floor_stand": EXPECTED_STATE_ARTIFACT_COUNT,
        "shared": EXPECTED_SHARED_ARTIFACT_COUNT,
    }
    expected_state_magnets = {
        "floor_stand": EXPECTED_STATE_MAGNET_COUNT,
        "no_floor_stand": EXPECTED_STATE_MAGNET_COUNT,
        "shared": EXPECTED_SHARED_MAGNET_COUNT,
    }
    if dict(state_artifact_counts) != expected_state_artifacts:
        raise RuntimeError(
            "released state/artifact inventory drifted: "
            f"{dict(state_artifact_counts)}")
    if dict(state_magnet_counts) != expected_state_magnets:
        raise RuntimeError(
            "released state/magnet inventory drifted: "
            f"{dict(state_magnet_counts)}")
    actual_family_counts = {
        family: (family_artifact_counts[family],
                 family_magnet_counts[family])
        for family in family_artifact_counts
    }
    if actual_family_counts != EXPECTED_FAMILY_COUNTS:
        raise RuntimeError(
            "released family inventory drifted: "
            f"{actual_family_counts}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": _sha256(SCHEMA_PATH),
        "catalog_kind": "released_pause_and_bury_captive_magnets",
        "generated_by": Path(__file__).name,
        "source_revision": source_revision,
        "print_contract": dict(RELEASE_ACOUSTIC_PRINT_CONTRACT),
        "geometry": {
            **DEFAULT_SPEC.facts(),
            "nominal_magnet": "D5.0 x 2.0 mm disc",
            "nominal_paired_magnet_face_separation_mm": (
                NOMINAL_PAIRED_FACE_SEPARATION_MM),
            "glue": False,
            "external_access_opening": False,
            "internal_support_material": False,
            "structural_load_credit_n": 0.0,
        },
        "inventory": {
            "artifact_count": len(artifacts),
            "magnet_count": magnet_count,
            "count_semantics": (
                "per-STL captive stations, including both stand-state "
                "copies and mutually exclusive monolithic/split LM print "
                "alternatives"),
            "state_artifact_count_each": EXPECTED_STATE_ARTIFACT_COUNT,
            "state_magnet_count_each": EXPECTED_STATE_MAGNET_COUNT,
            "shared_wing_artifact_count": EXPECTED_SHARED_ARTIFACT_COUNT,
            "shared_wing_magnet_count": EXPECTED_SHARED_MAGNET_COUNT,
            "family_counts": {
                family: {
                    "artifact_count": counts[0],
                    "magnet_count": counts[1],
                }
                for family, counts in sorted(EXPECTED_FAMILY_COUNTS.items())
            },
            "families": [
                "B2", "C7", "A", "B1", "V0", "V1", "V1-A",
                "V1-B1", "V1L", "V1LF", "V1LF-split",
                "V1LF-Ac", "V1LF-Ae", "coupon1",
            ],
        },
        "exclusions": [
            {
                "path": (
                    "oversized one-piece STEP masters, *_assembled.step, "
                    "*_attachments.step, and other STEP review packages"),
                "reason": (
                    "STEP-first geometry authorities or non-print review/"
                    "assembly containers; every corresponding bed-split "
                    "released STL is catalogued and sliced, while the STEP "
                    "master/container is not a separate print"),
            },
            {
                "path": "coupons/v1lf_ae_embed",
                "reason": (
                    "physically validated reference implementation, not a "
                    "production/released baffle STL; retained unchanged as "
                    "the geometry and Z-marker regression authority"),
            },
            {
                "path": "floor_stand/stl/lx521_coupon_7_recess_seat.stl",
                "reason": (
                    "driver-seat diagnostic crop; its current x/y crop "
                    "contains no released captive site and needs no pause"),
            },
            {
                "path": "no_floor_stand/stl/lx521_coupon_7_recess_seat.stl",
                "reason": (
                    "driver-seat diagnostic crop; its current x/y crop "
                    "contains no released captive site and needs no pause"),
            },
            {
                "path": "review/v1lf_wing_*_concept.png",
                "reason": "concept-only raster studies; no printable solid",
            },
            {
                "path": "gen_v1lf_wing_concepts.py",
                "reason": (
                    "concept-only raster generator; it emits no released "
                    "printable solid or magnet cavity"),
            },
            {
                "path": (
                    "gen_v1lf_basic_variants.py B1/B2 layout panels and "
                    "baffle_variants_drivers_v1lf.png"),
                "reason": (
                    "dimensioned feasibility/layout studies only; Ac and "
                    "Ae are the two implemented STEP-first V1LF wing "
                    "families and all twelve of their print segments are "
                    "catalogued"),
            },
            {
                "path": "gen_c_variants.py and baffle_C_variants_drivers.png",
                "reason": (
                    "C1-C6 concept-only magnet-boss drawings; no released "
                    "printable solid is generated"),
            },
            {
                "path": "gen_um_knife_draft.py",
                "reason": (
                    "obsolete pin-magnet slide concept drawing only; it "
                    "emits no released printable solid"),
            },
            {
                "path": "review/V1LF_NEXTGEN_SIMPLIFIED.md",
                "reason": (
                    "future/concept design note only; explicitly does not "
                    "define released printable geometry"),
            },
            {
                "path": "top_baffle_nd25fw4_v0.py future mating attachment",
                "reason": (
                    "no released V0 mating magnet print exists; V0's two "
                    "rear-axis magnets retain an explicit provisional "
                    "polarity convention"),
            },
            {
                "path": "legacy exposed-pocket generated artifacts",
                "reason": (
                    "obsolete outputs are replaced in place by the 56 "
                    "hash-bound regenerated STLs listed in this catalog"),
            },
        ],
        "artifacts": sorted(artifacts, key=lambda item: item["id"]),
    }
    # Never expose an unvalidated catalog at the authoritative path.  Render
    # beside it, run the same schema/binding contract as the slicer consumer,
    # and publish only after every gate passes.
    _publish_validated_catalog(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    payload = generate(output)
    print(json.dumps({
        "output": str(output),
        "artifact_count": len(payload["artifacts"]),
        "sha256": _sha256(output),
        "source_revision": payload["source_revision"],
        "print_orientation": payload["print_contract"]["orientation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
