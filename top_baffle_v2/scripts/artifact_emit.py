"""Evidence, Bambu invocation, pause projects and atomic bundles."""

from __future__ import annotations

from release_validation import *
from gcode_analysis import *


FATAL_BAMBU_SLICER_DIAGNOSTICS = {
    "floating_cantilever": "floating cantilever",
}


def _validate_bambu_slicer_log(
    path: Path, *, artifact_id: str, phase: str,
) -> None:
    """Reject geometric warnings that make an otherwise-successful slice unsafe."""
    if not path.is_file():
        raise AuditError(
            f"{artifact_id}: {phase} Bambu Studio log is missing: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    failures = [
        name for name, marker in FATAL_BAMBU_SLICER_DIAGNOSTICS.items()
        if marker in lowered
    ]
    if failures:
        raise AuditError(
            f"{artifact_id}: {phase} Bambu Studio reported release-blocking "
            f"geometry diagnostics: {', '.join(failures)}; see {path}")


def _color_for_feature(feature: str) -> str:
    value = feature.lower()
    if "outer wall" in value:
        return "#1769aa"
    if "inner wall" in value:
        return "#00a878"
    if "bridge" in value:
        return "#e4572e"
    if "infill" in value:
        return "#775da6"
    if "gap" in value:
        return "#f3a712"
    return "#66717e"


def _render_evidence_svg(
    path: Path,
    artifact: Mapping[str, Any],
    site_records: Sequence[Mapping[str, Any]],
    placement_xy: tuple[float, float],
) -> None:
    stages = (
        "lowest_open", "representative_open", "last_fully_open",
        "first_closing_pause", "fully_sealed")
    width = EVIDENCE_CELL_PX * len(stages)
    header = 74
    height = header + EVIDENCE_CELL_PX * len(site_records)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="12" y="25" font-family="sans-serif" font-size="17" font-weight="bold">{html.escape(artifact["id"])} — Bambu P2S 0.4 / 0.16 Arachne</text>',
        '<text x="12" y="49" font-family="sans-serif" font-size="12">front face down · local sliced G-code toolpaths · red box = nominal cavity plan</text>',
    ]
    for column, stage in enumerate(stages):
        x = column * EVIDENCE_CELL_PX + EVIDENCE_CELL_PX / 2
        elements.append(
            f'<text x="{x:.1f}" y="69" text-anchor="middle" font-family="sans-serif" font-size="11">{stage.replace("_", " ")}</text>')
    for row, record in enumerate(site_records):
        site = record["site"]
        cx, cy, _ = site["print_cavity_center_xyz_mm"]
        cx += placement_xy[0]
        cy += placement_xy[1]
        radius = site["cavity_diameter_mm"] / 2.0
        world_half = radius + EVIDENCE_MARGIN_MM
        scale = (EVIDENCE_CELL_PX - 28.0) / (2.0 * world_half)
        y0 = header + row * EVIDENCE_CELL_PX
        for column, stage in enumerate(stages):
            x0 = column * EVIDENCE_CELL_PX
            metrics = record["layer_metrics"][stage]
            elements.append(
                f'<rect x="{x0 + 1}" y="{y0 + 1}" width="{EVIDENCE_CELL_PX - 2}" height="{EVIDENCE_CELL_PX - 2}" fill="#fbfcfd" stroke="#d7dde4"/>')
            def map_xy(x: float, y: float) -> tuple[float, float]:
                return (
                    x0 + EVIDENCE_CELL_PX / 2 + (x - cx) * scale,
                    y0 + EVIDENCE_CELL_PX / 2 - (y - cy) * scale,
                )
            for segment in metrics.pop("segments"):
                sx0, sy0 = map_xy(segment.x0, segment.y0)
                sx1, sy1 = map_xy(segment.x1, segment.y1)
                stroke = _color_for_feature(segment.feature)
                width_px = max(0.75, (segment.line_width or 0.42) * scale * 0.55)
                elements.append(
                    f'<line x1="{sx0:.2f}" y1="{sy0:.2f}" x2="{sx1:.2f}" y2="{sy1:.2f}" stroke="{stroke}" stroke-width="{width_px:.2f}" stroke-linecap="round"/>')
            # Draw nominal open cavity projection in the print plane.
            if site["closure_kind"] == "transverse_gable_45deg":
                fx, fy, _ = site["print_actual_face_xyz_mm"]
                fx += placement_xy[0]
                fy += placement_xy[1]
                ux, uy = _unit_xy(site["print_material_inward_xyz"], "material inward")
                vx, vy = -uy, ux
                u0 = site["face_skin_mm"]
                u1 = u0 + site["cavity_depth_mm"]
                corners = []
                for u, v in ((u0, -radius), (u1, -radius),
                             (u1, radius), (u0, radius)):
                    corners.append(map_xy(fx + u * ux + v * vx,
                                          fy + u * uy + v * vy))
                points = " ".join(f"{x:.2f},{y:.2f}" for x, y in corners)
                elements.append(
                    f'<polygon points="{points}" fill="none" stroke="#d62828" stroke-width="1.2" stroke-dasharray="4 3"/>')
            else:
                pcx, pcy = map_xy(cx, cy)
                elements.append(
                    f'<circle cx="{pcx:.2f}" cy="{pcy:.2f}" r="{radius * scale:.2f}" fill="none" stroke="#d62828" stroke-width="1.2" stroke-dasharray="4 3"/>')
            elements.append(
                f'<text x="{x0 + 8}" y="{y0 + 17}" font-family="monospace" font-size="10">{html.escape(site["name"])} Z={metrics["z_mm"]:.2f}</text>')
            elements.append(
                f'<text x="{x0 + 8}" y="{y0 + EVIDENCE_CELL_PX - 7}" font-family="monospace" font-size="9">roof interior={metrics["roof_interior_path_length_mm"]:.2f} mm</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _svg_to_png(svg: Path, png: Path) -> dict[str, Any]:
    if not svg.is_file() or svg.stat().st_size == 0:
        raise AuditError(f"SVG evidence is missing or empty: {svg}")
    # Never accept a renderer's stale output from a prior audit.
    png.unlink(missing_ok=True)
    commands = []
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        commands.append([rsvg, "-o", str(png), str(svg)])
    magick = shutil.which("magick")
    if magick:
        commands.append([magick, str(svg), str(png)])
    convert = shutil.which("convert")
    if convert:
        commands.append([convert, str(svg), str(png)])
    errors = []
    for command in commands:
        png.unlink(missing_ok=True)
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        if run.returncode == 0 and png.is_file() and png.stat().st_size:
            return {"path": str(png), "sha256": sha256_file(png),
                    "renderer": command[0]}
        errors.append(f"{' '.join(command)}: {run.stdout[-1000:]}")
    png.unlink(missing_ok=True)
    detail = "; ".join(errors or ["no SVG-to-PNG renderer found"])
    raise AuditError(f"fresh PNG evidence could not be rendered: {detail}")


def _gcode_tool_path() -> Path | None:
    candidates = sorted((Path.home() / ".codex" / "plugins" / "cache"
                         / "text-to-cad" / "cad").glob(
        "*/skills/gcode/scripts/gcode_tool.py"), reverse=True)
    return candidates[0] if candidates else None


def _validate_with_gcode_skill(
    gcode: Path,
    out_dir: Path,
    profile_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    tool = _gcode_tool_path()
    if tool is None:
        return {"ok": None, "reason": "gcode skill validator not installed"}
    effective = profile_bundle["identity"]["effective"]
    bounds = profile_bundle["identity"]["machine_bounds_mm"]
    filament = profile_bundle["resolved"]["filament"]
    nozzle_temp = _scalar(filament, "nozzle_temperature", "filament")
    bed_temp = _scalar(filament, "eng_plate_temp", "filament")
    wrapper = {
        # The skill validator's schema currently enumerates Orca/Prusa/Cura.
        # This value is only a validation-schema compatibility field: actual
        # slicing provenance remains BambuStudio in every manifest record.
        "backend": "orcaslicer",
        "native_config": str(profile_bundle["paths"]["machine"]),
        "machine": {
            "name": effective["printer_model"],
            "bed_size_mm": [bounds["x"][1], bounds["y"][1]],
            "z_height_mm": bounds["z"][1],
            "motion_bounds_mm": bounds,
        },
        "filament": {
            "type": effective["filament"],
            "nozzle_temp_c": nozzle_temp,
            "bed_temp_c": bed_temp,
        },
    }
    wrapper_path = out_dir / "gcode_validation_profile.json"
    _write_json(wrapper_path, wrapper)
    run = subprocess.run(
        [sys.executable, str(tool), "validate", "--gcode", str(gcode),
         "--profile", str(wrapper_path), "--json"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False)
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError:
        result = {"ok": False, "returncode": run.returncode,
                  "raw_output": run.stdout[-4000:]}
    _write_json(out_dir / "gcode_skill_validation.json", result)
    return result


def _cached_slice_matches(
    prior: Mapping[str, Any], *, fingerprint: str, stl: Path,
    gcode: Path, result_path: Path, project_3mf: Path,
) -> bool:
    """Reuse only hash-bound slicer outputs, never merely their input key."""
    required = {
        "fingerprint": fingerprint,
        "stl_sha256": sha256_file(stl),
        "gcode_sha256": sha256_file(gcode),
        "result_sha256": sha256_file(result_path),
        "project_3mf_sha256": sha256_file(project_3mf),
    }
    return all(prior.get(key) == value for key, value in required.items())


def _source_hashes(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.is_file():
            records.append({"path": str(path), "sha256": sha256_file(path)})
        else:
            records.append({"path": str(path), "sha256": None,
                            "error": "missing source"})
    return records


def _artifact_fingerprint(
    artifact: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
    catalog_sha: str,
) -> str:
    stl = artifact["stl"]
    payload = {
        "catalog_sha256": catalog_sha,
        "catalog_source_revision": artifact["catalog_source_revision"],
        "catalog_record": artifact["catalog_record"],
        "stl_sha256": sha256_file(stl),
        "print_sidecar_sha256": sha256_file(artifact["print_sidecar"]),
        "source_file_sha256": sorted(
            artifact["catalog_record"]["source_file_sha256"].items()),
        "transaction_manifest_sha256": artifact.get(
            "transaction_manifest_sha256"),
        "facts_sha256": artifact.get("facts_sha256"),
        "stage_manifest_sha256": artifact.get("stage_manifest_sha256"),
        "support_blocker_sha256": artifact.get("support_blocker_sha256"),
        "support_blocker_binding_sha256": artifact.get(
            "support_blocker_binding_sha256"),
        "profile_set_sha256": profile_bundle["identity"]["profile_set_sha256"],
        "bambu_binary_sha256": profile_bundle["identity"]["binary_sha256"],
        "audit_source_sha256": sorted(
            profile_bundle["audit_source_sha256"].items()),
    }
    return _sha256_bytes(_canonical_json(payload))


def _write_bambu_assemble_list(
    path: Path,
    *,
    stl: Path,
    support_blockers: Sequence[Path],
) -> None:
    """Describe one printable object with co-located support blockers."""
    if not support_blockers:
        raise AuditError("an assemble list requires at least one modifier")

    def object_record(mesh: Path, subtype: str) -> dict[str, Any]:
        return {
            "path": str(mesh.resolve()),
            "subtype": subtype,
            "count": 1,
            "filaments": [1],
            "assemble_index": [1],
            "pos_x": [0],
            "pos_y": [0],
            "pos_z": [0],
        }

    _write_json(path, {
        "plates": [{
            "plate_name": stl.stem,
            "need_arrange": True,
            "objects": [
                object_record(stl, "normal_part"),
                *(object_record(blocker, "support_blocker")
                  for blocker in support_blockers),
            ],
            "assembled_params": [{
                "assemble_index": 1,
                "print_params": {
                    "enable_support": "1",
                    "support_on_build_plate_only": "1",
                    "support_critical_regions_only": "1",
                    "support_remove_small_overhang": "1",
                },
            }],
        }],
    })


def _bambu_command(
    bambu: Path,
    stl: Path,
    output: Path,
    profile_bundle: Mapping[str, Any],
    *,
    project_filename: str = PLACED_3MF_FILENAME,
    custom_gcodes: Path | None = None,
    assemble_list: Path | None = None,
) -> list[str]:
    settings = ";".join(str(profile_bundle["paths"][key])
                        for key in ("machine", "process"))
    command = [
        str(bambu), "--debug", "2", "--slice", "0", "--arrange", "1",
        "--orient", "0", "--allow-rotations=0",
        "--export-3mf", project_filename,
        "--load-settings", settings,
        "--load-filaments", str(profile_bundle["paths"]["filament"]),
        "--outputdir", str(output),
    ]
    if custom_gcodes is not None:
        command.extend(("--load-custom-gcodes", str(custom_gcodes)))
    if assemble_list is None:
        command.append(str(stl))
    else:
        command.extend(("--load-assemble-list", str(assemble_list)))
    return command


def _slice_one(
    artifact: Mapping[str, Any],
    *,
    output_root: Path,
    profile_bundle: Mapping[str, Any],
    bambu: Path,
    catalog_sha: str,
    reuse: bool,
    dry_run: bool,
    emit_ready_projects: bool = False,
) -> dict[str, Any]:
    stl: Path = artifact["stl"]
    support_blockers = tuple(
        [artifact["support_blocker"]]
        if "support_blocker" in artifact else [])
    release_stl: Path = artifact.get("release_stl", stl)
    _validate_artifact_bindings(artifact)
    mesh = inspect_stl(stl)
    if abs(mesh.bounds_min[2]) > 0.02:
        raise AuditError(
            f"{artifact['id']}: front-down STL must sit at Z=0; "
            f"min Z={mesh.bounds_min[2]:.4f}")
    slug = _slug(artifact["id"])
    out_dir = output_root / "slices" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    ready_dir = out_dir / "ready"
    # A later diagnostic discovery-only run must not leave a stale packaged
    # project beside fresh evidence where it could be mistaken for this run's
    # validated primary artifact.
    if not emit_ready_projects and not dry_run:
        shutil.rmtree(ready_dir, ignore_errors=True)
    artifact_profile_bundle = _artifact_profile_bundle(
        artifact, profile_bundle, out_dir)
    bounds = artifact_profile_bundle["identity"]["machine_bounds_mm"]
    if mesh.size[0] > bounds["x"][1] - bounds["x"][0] + 1e-4 \
            or mesh.size[1] > bounds["y"][1] - bounds["y"][0] + 1e-4 \
            or mesh.size[2] > bounds["z"][1] - bounds["z"][0] + 1e-4:
        raise AuditError(
            f"{artifact['id']}: {mesh.size} mm exceeds P2S 256-mm envelope")
    fingerprint = _artifact_fingerprint(
        artifact, artifact_profile_bundle, catalog_sha)
    fingerprint_path = out_dir / "slice_fingerprint.json"
    gcode = out_dir / "plate_1.gcode"
    result_path = out_dir / "result.json"
    project_3mf = out_dir / PLACED_3MF_FILENAME
    assemble_list = (
        out_dir / "bambu_assemble_list.json"
        if support_blockers else None)
    if assemble_list is not None:
        _write_bambu_assemble_list(
            assemble_list, stl=stl,
            support_blockers=support_blockers)
    reused = False
    if (reuse and fingerprint_path.is_file() and gcode.is_file()
            and result_path.is_file() and project_3mf.is_file()):
        prior = _load_json(fingerprint_path)
        if isinstance(prior, dict) and _cached_slice_matches(
                prior, fingerprint=fingerprint, stl=stl,
                gcode=gcode, result_path=result_path,
                project_3mf=project_3mf):
            reused = True
    command = _bambu_command(
        bambu, stl, out_dir, artifact_profile_bundle,
        assemble_list=assemble_list)
    if dry_run:
        return {"id": artifact["id"], "dry_run": True, "command": command,
                "fingerprint": fingerprint}
    if not reused:
        for stale in (gcode, result_path, project_3mf):
            stale.unlink(missing_ok=True)
        run = subprocess.run(
            command, cwd=out_dir, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(
                artifact_profile_bundle["config"]["slicing"][
                    "timeout_seconds"]),
            check=False,
            env={**os.environ, "LC_ALL": "C"})
        (out_dir / "bambu_studio.log").write_text(
            run.stdout, encoding="utf-8", errors="replace")
        if run.returncode != 0:
            raise AuditError(
                f"{artifact['id']}: Bambu Studio exited {run.returncode}; "
                f"see {out_dir / 'bambu_studio.log'}")
        if (not gcode.is_file() or not result_path.is_file()
                or not project_3mf.is_file()):
            raise AuditError(
                f"{artifact['id']}: Bambu Studio did not create "
                "plate_1.gcode/result.json/audited 3MF")
        _write_json(fingerprint_path, {
            "fingerprint": fingerprint,
            "command": command,
            "stl_sha256": sha256_file(stl),
            "gcode_sha256": sha256_file(gcode),
            "result_sha256": sha256_file(result_path),
            "project_3mf_sha256": sha256_file(project_3mf),
        })
    _validate_bambu_slicer_log(
        out_dir / "bambu_studio.log",
        artifact_id=artifact["id"],
        phase="audit slice")
    result_json = _load_json(result_path)
    if result_json.get("return_code") != 0:
        raise AuditError(f"{artifact['id']}: slicer result is not Success: {result_json}")
    plates = result_json.get("sliced_plates")
    if not isinstance(plates, list) or len(plates) != 1:
        raise AuditError(f"{artifact['id']}: expected exactly one sliced plate")
    objects = plates[0].get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise AuditError(f"{artifact['id']}: expected exactly one sliced object")
    if int(plates[0].get("triangle_count", -1)) != mesh.triangle_count:
        raise AuditError(
            f"{artifact['id']}: sliced plate triangle count differs from "
            f"staged STL ({plates[0].get('triangle_count')} != "
            f"{mesh.triangle_count})")
    if int(objects[0].get("triangle_count", -1)) != mesh.triangle_count:
        raise AuditError(
            f"{artifact['id']}: sliced object triangle count differs from "
            f"staged STL ({objects[0].get('triangle_count')} != "
            f"{mesh.triangle_count})")
    bbox = objects[0].get("bbox")
    if not isinstance(bbox, dict):
        raise AuditError(f"{artifact['id']}: missing Bambu object bbox")
    try:
        project_audit = audit_bambu_3mf(
            project_3mf, stl,
            support_blocker_stls=support_blockers)
        expected_bbox = validate_bambu_result_bbox(
            bbox, project_audit.source_bounds,
            project_audit.stl_to_bed_matrix)
        bed_clearances = validate_bambu_bed_fit(
            project_audit.transformed_actual_mesh_bounds, bounds)
    except Bambu3MFAuditError as exc:
        raise AuditError(
            f"{artifact['id']}: Bambu 3MF placement/mesh audit failed: "
            f"{exc}") from exc
    if project_audit.triangle_count != mesh.triangle_count:
        raise AuditError(
            f"{artifact['id']}: archived 3MF triangle count differs from "
            "staged STL")
    slicer_sites = [
        _site_in_bambu_bed_space(site, project_audit.stl_to_bed_matrix)
        for site in artifact["sites"]
    ]
    # All local geometry now carries Bambu's full audited Rz+XY transform.
    # The legacy additive placement parameter remains zero so the cavity
    # toolpath routines can stay independent of the 3MF parser.
    placement = (0.0, 0.0)
    parsed = parse_gcode(
        gcode,
        retain_regions=_cavity_retain_regions(slicer_sites, placement),
    )
    site_records = []
    errors = _validate_actual_gcode_profile(parsed, artifact_profile_bundle)
    for site in slicer_sites:
        selected, metrics, closure_discovery = (
            _discover_actual_closure_layers(parsed.layers, site, placement))
        roof_pass, roof_detail = _roof_progression_pass(metrics)
        retaining_stage_pass = {
            stage: _retaining_stage_pass(site, stage, metrics[stage])
            for stage in (
                "lowest_open", "representative_open", "last_fully_open")
        }
        retaining_pass = all(retaining_stage_pass.values())
        aperture_pass, aperture_detail = _loading_aperture_pass(
            site, metrics["last_fully_open"])
        magnet_bottom_z, magnet_top_z = _seated_magnet_print_z_bounds(site)
        seated_below_last_open = (
            selected["last_fully_open"].z - magnet_top_z)
        seated_below_first_closing = (
            selected["first_closing_pause"].z - magnet_top_z)
        seated_clearance_pass = (
            seated_below_last_open >= -LAYER_EPS
            and seated_below_first_closing >= 0.02 - LAYER_EPS
        )
        diametric_clearance = (
            site["cavity_diameter_mm"] - site["magnet_diameter_mm"])
        axial_clearance = (
            site["cavity_depth_mm"] - site["magnet_depth_mm"])
        insertion_fit_pass = (
            diametric_clearance >= 0.19
            and axial_clearance >= 0.09
        )
        expected = site.get("expected_pause_marker_z_mm")
        actual_pause = selected["first_closing_pause"].z
        regression_pass = expected is None or math.isclose(
            actual_pause, expected, abs_tol=0.001)
        if not roof_pass:
            errors.append(f"{site['name']}: no first-closing roof progression ({roof_detail})")
        if not retaining_pass:
            failed_stages = [
                stage for stage, passed in retaining_stage_pass.items()
                if not passed
            ]
            errors.append(
                f"{site['name']}: retaining paths missing at open stage(s) "
                + ", ".join(failed_stages))
        if not aperture_pass:
            errors.append(
                f"{site['name']}: last-open loading aperture rejects the "
                f"nominal magnet ({aperture_detail})")
        if not seated_clearance_pass:
            errors.append(
                f"{site['name']}: fully seated magnet top Z "
                f"{magnet_top_z:.3f} is not below the completed last-open "
                f"layer Z {selected['last_fully_open'].z:.3f} and clear of "
                f"first-closing Z {actual_pause:.3f}")
        if not insertion_fit_pass:
            errors.append(
                f"{site['name']}: nominal magnet cannot be dropped/seated; "
                f"diametric clearance={diametric_clearance:.3f} mm, "
                f"axial clearance={axial_clearance:.3f} mm")
        if not regression_pass:
            errors.append(
                f"{site['name']}: pause regression {actual_pause:.2f} != {expected:.2f}")
        stage_records = {}
        for key, layer in selected.items():
            clean_metrics = {k: v for k, v in metrics[key].items() if k != "segments"}
            stage_records[key] = clean_metrics
        site_records.append({
            "site": site,
            "actual": {
                "lowest_open_layer_z_mm": selected["lowest_open"].z,
                "representative_open_layer_z_mm": selected["representative_open"].z,
                "last_completely_open_layer_z_mm": selected["last_fully_open"].z,
                "cavity_bury_roof_start_plane_z_mm": site[
                    "cavity_bury_roof_start_print_z_mm"],
                "first_closing_layer_z_mm": actual_pause,
                "bambu_studio_pause_marker_z_mm": actual_pause,
                "fully_sealed_inspection_layer_z_mm": selected["fully_sealed"].z,
            },
            "layer_metrics": metrics,
            "layer_evidence": stage_records,
            "roof_progression_pass": roof_pass,
            "roof_progression_detail": roof_detail,
            "retaining_paths_pass": retaining_pass,
            "retaining_paths_stage_pass": retaining_stage_pass,
            "loading_aperture_pass": aperture_pass,
            "loading_aperture_detail": aperture_detail,
            "seated_magnet": {
                "print_center_xyz_mm": list(
                    site["print_seated_magnet_center_xyz_mm"]),
                "print_bottom_z_mm": magnet_bottom_z,
                "print_top_z_mm": magnet_top_z,
                "below_last_open_layer_mm": seated_below_last_open,
                "below_first_closing_layer_mm": seated_below_first_closing,
                "clearance_pass": seated_clearance_pass,
            },
            "insertion_fit": {
                "diametric_clearance_mm": diametric_clearance,
                "axial_clearance_mm": axial_clearance,
                "pass": insertion_fit_pass,
            },
            "regression_expected_z_mm": expected,
            "regression_pass": regression_pass,
            "closure_discovery": closure_discovery,
        })
    evidence_svg = out_dir / "captive_toolpath_evidence.svg"
    evidence_svg.unlink(missing_ok=True)
    evidence_artifact = dict(artifact)
    evidence_artifact["sites"] = slicer_sites
    _render_evidence_svg(
        evidence_svg, evidence_artifact, site_records, placement)
    if not evidence_svg.is_file() or evidence_svg.stat().st_size == 0:
        raise AuditError(
            f"{artifact['id']}: fresh SVG evidence was not created")
    evidence_png = out_dir / "captive_toolpath_evidence.png"
    png_record = _svg_to_png(evidence_svg, evidence_png)
    # Renderer consumes/removes segment objects from metrics.  Only serializable
    # stage evidence remains in the final record.
    for record in site_records:
        record.pop("layer_metrics", None)
    skill_validation = _validate_with_gcode_skill(
        gcode, out_dir, artifact_profile_bundle)
    if skill_validation.get("ok") is not True:
        errors.append(
            "plain G-code static validation did not return an explicit pass")
    sources = _source_hashes(artifact.get(
        "release_source_files", artifact["source_files"]))
    missing_sources = [item["path"] for item in sources if item["sha256"] is None]
    if missing_sources:
        errors.append("missing source files: " + ", ".join(missing_sources))
    project_audit_record = project_audit.as_record()
    project_audit_record.pop("staged_stl", None)
    project_audit_record["audited_release_stl"] = str(release_stl)
    project_audit_record["audited_stl_sha256"] = sha256_file(stl)
    record = {
        "id": artifact["id"],
        "state": artifact["state"],
        "variant": artifact["variant"],
        "part": artifact["part"],
        "print_orientation": artifact["print_orientation"],
        "audit_mode": "actual_p2s_slice",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "reused_slice": reused,
        "command": command,
        "fingerprint": fingerprint,
        "profile_set_sha256": artifact_profile_bundle[
            "identity"]["profile_set_sha256"],
        "profile_effective": artifact_profile_bundle[
            "identity"]["effective"],
        "artifact_profile_override": artifact_profile_bundle[
            "identity"].get("artifact_override"),
        "input": {
            "stl": str(release_stl),
            "stl_sha256": sha256_file(stl),
            "sliced_from_immutable_stage": (
                release_stl.resolve() != stl.resolve()),
            "triangle_count": mesh.triangle_count,
            "bounds_min_mm": mesh.bounds_min,
            "bounds_max_mm": mesh.bounds_max,
            "size_mm": mesh.size,
            "source_files": sources,
        },
        "slicer": {
            "result_json": str(result_path),
            "result_sha256": sha256_file(result_path),
            "project_3mf": str(project_3mf),
            "project_3mf_sha256": sha256_file(project_3mf),
            "gcode": str(gcode),
            "gcode_sha256": sha256_file(gcode),
            "bambu_3mf_audit": project_audit_record,
            "bambu_expected_result_bbox": expected_bbox,
            "actual_mesh_bed_clearance_mm": {
                axis: list(values) for axis, values in bed_clearances.items()
            },
            "sliced_bbox": bbox,
            "layer_count": len(parsed.layers),
            "movement_commands": parsed.movement_commands,
            "arc_commands": parsed.arc_commands,
            "extrusion_moves": parsed.extrusion_moves,
            "temperature_commands": parsed.temperature_commands,
            "gcode_bounds_min_mm": parsed.bounds_min,
            "gcode_bounds_max_mm": parsed.bounds_max,
            "effective_config": {
                key: parsed.config.get(key) for key in (
                    "layer_height", "initial_layer_print_height",
                    "wall_generator", "outer_wall_line_width",
                    "inner_wall_line_width", "wall_loops",
                    "top_shell_layers", "bottom_shell_layers",
                    "outer_wall_speed", "curr_bed_type",
                    "sparse_infill_pattern", "sparse_infill_density",
                    "precise_outer_wall", "detect_thin_wall",
                    "ensure_vertical_shell_thickness",
                    "detect_narrow_internal_solid_infill",
                    "elefant_foot_compensation", "xy_hole_compensation",
                    "enable_support", "support_on_build_plate_only",
                    "support_critical_regions_only",
                    "support_remove_small_overhang", "enable_arc_fitting",
                    "nozzle_temperature",
                    "nozzle_temperature_initial_layer", "fan_max_speed",
                    "overhang_fan_speed",
                    "filament_max_volumetric_speed",
                    "textured_plate_temp",
                    "textured_plate_temp_initial_layer",
                    "machine_pause_gcode")
            },
            "gcode_skill_validation": skill_validation,
        },
        "sites": site_records,
        "evidence": {
            "svg": str(evidence_svg),
            "svg_sha256": sha256_file(evidence_svg),
            "png": png_record,
        },
    }
    if emit_ready_projects:
        if record["status"] == "pass":
            record["slicer"]["ready_project"] = _emit_ready_project(
                record=record,
                artifact=artifact,
                stl=stl,
                mesh=mesh,
                ready_dir=ready_dir,
                profile_bundle=artifact_profile_bundle,
                bambu=bambu,
                discovery_fingerprint=fingerprint,
                reuse=reuse,
            )
        else:
            shutil.rmtree(ready_dir, ignore_errors=True)
    _write_json(out_dir / "captive_magnet_slice_audit.json", record)
    return record


def _oversize_proxy_coverage_record(
    artifact: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
    profile_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Cover an unprintable monolith only through exact passing split sites."""
    stl: Path = artifact["stl"]
    release_stl: Path = artifact.get("release_stl", stl)
    errors: list[str] = []
    _validate_artifact_bindings(artifact)
    mesh = inspect_stl(stl)
    bounds = profile_bundle["identity"]["machine_bounds_mm"]
    limits = tuple(
        bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z"))
    exceeds = tuple(
        size > limit + 1.0e-4 for size, limit in zip(mesh.size, limits))
    if not (exceeds[0] or exceeds[1]):
        errors.append(
            "catalog declares this artifact P2S-oversize, but its front-down "
            f"XY footprint {mesh.size[:2]} fits {limits[:2]}; contract is stale")
    if exceeds[2]:
        errors.append(
            f"artifact also exceeds P2S Z: {mesh.size[2]:.3f} > "
            f"{limits[2]:.3f} mm")
    coverage = []
    for proxy in artifact["cavity_audit_proxies"]:
        proxy_record = records_by_id.get(proxy["artifact_id"])
        if proxy_record is None:
            errors.append(
                f"{proxy['site']}: proxy artifact was not successfully sliced: "
                f"{proxy['artifact_id']}")
            continue
        if (proxy_record.get("audit_mode") != "actual_p2s_slice"
                or proxy_record.get("status") != "pass"):
            errors.append(
                f"{proxy['site']}: proxy did not pass a normal P2S slice: "
                f"{proxy['artifact_id']} status={proxy_record.get('status')}")
            continue
        matches = [
            record for record in proxy_record.get("sites", ())
            if record.get("site", {}).get("name") == proxy["proxy_site"]
        ]
        if len(matches) != 1:
            errors.append(
                f"{proxy['site']}: proxy audit site is missing or ambiguous: "
                f"{proxy['artifact_id']}/{proxy['proxy_site']}")
            continue
        site_record = matches[0]
        if site_record["site"].get("source_contract_sha256") != proxy[
                "source_contract_sha256"]:
            errors.append(
                f"{proxy['site']}: sliced proxy source contract hash drifted")
            continue
        if not all((
                site_record.get("retaining_paths_pass") is True,
                site_record.get("loading_aperture_pass") is True,
                site_record.get("seated_magnet", {}).get(
                    "clearance_pass") is True,
                site_record.get("insertion_fit", {}).get("pass") is True,
                site_record.get("regression_pass") is True,
        )):
            errors.append(
                f"{proxy['site']}: proxy site did not pass every cavity gate")
            continue
        evidence = proxy_record.get("evidence", {})
        png = evidence.get("png", {})
        coverage.append({
            "site": proxy["site"],
            "source_contract_sha256": proxy["source_contract_sha256"],
            "proxy_artifact_id": proxy["artifact_id"],
            "proxy_site": proxy["proxy_site"],
            "proxy_stl_sha256": proxy_record["input"]["stl_sha256"],
            "proxy_gcode_sha256": proxy_record["slicer"]["gcode_sha256"],
            "proxy_site_audit_sha256": _sha256_bytes(
                _canonical_json(site_record)),
            "proxy_evidence_svg_sha256": evidence.get("svg_sha256"),
            "proxy_evidence_png_sha256": png.get("sha256"),
            "pause_marker_z_mm": site_record["actual"][
                "bambu_studio_pause_marker_z_mm"],
        })
    expected_sites = {site["name"] for site in artifact["sites"]}
    covered_sites = {item["site"] for item in coverage}
    if covered_sites != expected_sites:
        errors.append(
            "exact split coverage is incomplete; missing="
            + ",".join(sorted(expected_sites - covered_sites)))
    sources = _source_hashes(artifact.get(
        "release_source_files", artifact["source_files"]))
    missing_sources = [item["path"] for item in sources
                       if item["sha256"] is None]
    if missing_sources:
        errors.append("missing source files: " + ", ".join(missing_sources))
    status = OVERSIZE_COVERED_STATUS if not errors else "fail"
    return {
        "id": artifact["id"],
        "state": artifact["state"],
        "variant": artifact["variant"],
        "part": artifact["part"],
        "print_orientation": artifact["print_orientation"],
        "audit_mode": "exact_split_proxy_coverage",
        "status": status,
        "errors": errors,
        "p2s_printable": False,
        "input": {
            "stl": str(release_stl),
            "stl_sha256": sha256_file(stl),
            "sliced_from_immutable_stage": (
                release_stl.resolve() != stl.resolve()),
            "triangle_count": mesh.triangle_count,
            "bounds_min_mm": mesh.bounds_min,
            "bounds_max_mm": mesh.bounds_max,
            "size_mm": mesh.size,
            "source_files": sources,
        },
        "p2s_bed_fit": {
            "pass": False,
            "machine_limits_mm": limits,
            "exceeds_axes": {
                axis: value for axis, value in zip(("x", "y", "z"), exceeds)
            },
            "policy": (
                "no virtual bed, scaling, tilting, clipping, G-code, or fake "
                "pause group for this canonical monolith"),
        },
        "source_site_contracts": [
            {
                "site": site["name"],
                "sha256": site["source_contract_sha256"],
            }
            for site in artifact["sites"]
        ],
        "cavity_audit_coverage": {
            "pass": not errors,
            "method": "exact_same_state_keyed_split_p2s_gcode",
            "sites": coverage,
        },
    }


def _pause_groups(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (record.get("audit_mode") != "actual_p2s_slice"
            or record.get("status") != "pass"):
        return []
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for site_record in record.get("sites", ()):
        z = float(site_record["actual"]["bambu_studio_pause_marker_z_mm"])
        grouped.setdefault(z, []).append(site_record)
    result = []
    for index, (z, sites) in enumerate(sorted(grouped.items()), 1):
        insertion_directions = [
            _vec3(_required(
                item["site"], "print_insertion_direction_xyz",
                f"{record.get('id', '<unknown>')}/"
                f"{item['site'].get('name', '<unnamed>')}: print insertion "
                "direction"), "print insertion direction")
            for item in sites
        ]
        if any(any(not math.isclose(
                actual, expected, abs_tol=1.0e-9, rel_tol=0.0)
                for actual, expected in zip(
                    direction, PRINT_INSERTION_DIRECTION_XYZ, strict=True))
                for direction in insertion_directions):
            raise AuditError(
                f"{record.get('id', '<unknown>')}: pause group {index} has "
                f"an unsafe insertion direction: {insertion_directions}")
        result.append({
            "group": index,
            "pause_marker_z_mm": z,
            "sites": [item["site"]["name"] for item in sites],
            "magnet_count": len(sites),
            "last_completely_open_layer_z_mm": max(
                item["actual"]["last_completely_open_layer_z_mm"] for item in sites),
            "cavity_bury_roof_start_plane_z_mm": sorted({
                item["actual"]["cavity_bury_roof_start_plane_z_mm"] for item in sites}),
            "first_closing_layer_z_mm": z,
            "print_insertion_direction_xyz": list(
                PRINT_INSERTION_DIRECTION_XYZ),
            "insertion_instruction": PRINT_INSERTION_INSTRUCTION,
            "minimum_seated_below_last_open_layer_mm": min(
                item["seated_magnet"]["below_last_open_layer_mm"]
                for item in sites),
            "minimum_seated_below_first_closing_layer_mm": min(
                item["seated_magnet"]["below_first_closing_layer_mm"]
                for item in sites),
            "polarity": [{
                "site": item["site"]["name"],
                "print_marked_pole_axis_xyz": item["site"]["print_marked_pole_axis_xyz"],
                "installed_marked_pole_axis_xyz": item["site"].get(
                    "installed_marked_pole_axis_xyz"),
                "instruction": item["site"]["polarity_instruction"],
            } for item in sites],
        })
    return result


def _magnet_pause_commands(
    pause_z: float,
    pause_policy: Mapping[str, Any],
) -> list[str]:
    """Return the exact physical motion/pause/resume command sequence."""
    park_z = _float(
        pause_policy.get("park_z_mm"), "magnet insertion park Z")
    feedrate = _float(
        pause_policy.get("z_travel_feedrate_mm_min"),
        "magnet insertion Z feedrate")
    pause_command = " ".join(str(
        pause_policy.get("pause_command", "")).split())
    if pause_command != MAGNET_INSERTION_PAUSE_COMMAND:
        raise AuditError(
            "magnet insertion pause command must be exactly "
            f"{MAGNET_INSERTION_PAUSE_COMMAND!r}")
    if park_z <= pause_z + 1.0e-6:
        raise AuditError(
            f"magnet insertion park Z {park_z:g} is not above its "
            f"pause layer {pause_z:g}")
    return [
        "G90",
        "M400",
        f"G1 Z{park_z:g} F{feedrate:g}",
        "M400",
        pause_command,
        f"G1 Z{pause_z:g} F{feedrate:g}",
        "M400",
    ]


def _magnet_pause_program(
    group: Mapping[str, Any],
    pause_policy: Mapping[str, Any],
) -> str:
    """Build the self-describing custom G-code event stored in the 3MF."""
    pause_z = _float(group.get("pause_marker_z_mm"), "magnet pause Z")
    sites = ", ".join(str(value) for value in group["sites"])
    park_z = _float(
        pause_policy.get("park_z_mm"), "magnet insertion park Z")
    return "\n".join((
        MAGNET_INSERTION_PARK_BEGIN,
        f"; Insert {group['magnet_count']} magnet(s): {sites}",
        "; P2S: no XY move; absolute Z raises the nozzle and lowers the bed.",
        f"; Park at Z={park_z:g} mm, then restore Z={pause_z:g} mm after Continue.",
        *_magnet_pause_commands(pause_z, pause_policy),
        MAGNET_INSERTION_PARK_END,
    ))


def _custom_gcodes_document(
    record: Mapping[str, Any],
    pause_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[float]]:
    groups = _pause_groups(record)
    if not groups:
        raise AuditError(
            f"{record.get('id', '<unknown>')}: no passing pause groups are "
            "available for a ready-to-print project")
    gcodes = []
    pause_z = []
    for group in groups:
        z = float(group["pause_marker_z_mm"])
        pause_z.append(z)
        gcodes.append({
            "type": MAGNET_INSERTION_CUSTOM_GCODE_TYPE,
            "print_z": z,
            "color": "",
            "extruder": 1,
            "extra": _magnet_pause_program(group, pause_policy),
        })
    return {"mode": "SingleExtruder", "gcodes": gcodes}, pause_z


def _gcode_pause_events(
    path: Path,
    pause_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Locate Bambu custom magnet pauses and prove park/pause/restore order."""
    current_z: float | None = None
    pending_change = False
    awaiting: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if line == "; CHANGE_LAYER":
                if awaiting is not None:
                    raise AuditError(
                        f"{path}:{awaiting['line_number']}: custom magnet "
                        "pause is unfinished before the next layer")
                pending_change = True
                continue
            if pending_change and line.startswith("; Z_HEIGHT:"):
                current_z = _float(
                    line.split(":", 1)[1], "pause G-code layer Z")
                pending_change = False
                continue
            if line == "; PAUSE_PRINTING":
                raise AuditError(
                    f"{path}:{line_number}: found obsolete bare PausePrint; "
                    "ready projects must use the magnet park/pause/restore "
                    "custom G-code")
            if line == "; CUSTOM_GCODE":
                if current_z is None:
                    raise AuditError(f"{path}:{line_number}: custom magnet "
                                     "pause precedes any layer Z")
                if awaiting is not None:
                    raise AuditError(f"{path}:{line_number}: nested custom "
                                     "magnet pauses")
                awaiting = {
                    "z_mm": current_z,
                    "line_number": line_number,
                    "inside_program": False,
                    "commands": [],
                }
                continue
            if awaiting is None:
                continue
            if line == MAGNET_INSERTION_PARK_BEGIN:
                if awaiting["inside_program"]:
                    raise AuditError(
                        f"{path}:{line_number}: nested magnet pause program")
                awaiting["inside_program"] = True
                continue
            if line == MAGNET_INSERTION_PARK_END:
                if not awaiting["inside_program"]:
                    raise AuditError(
                        f"{path}:{line_number}: magnet pause program ended "
                        "before it began")
                commands = awaiting["commands"]
                expected = _magnet_pause_commands(
                    float(awaiting["z_mm"]), pause_policy)
                actual = [command for command, _line in commands]
                # Bambu keeps the park feedrate but coalesces the identical
                # restore feedrate into the modal state, yielding ``G1 Z...``
                # rather than ``G1 Z... F...``.  The preceding park move
                # proves the inherited value is the pinned safe Z speed.
                modal_restore = list(expected)
                modal_restore[5] = f"G1 Z{float(awaiting['z_mm']):g}"
                if actual not in (expected, modal_restore):
                    raise AuditError(
                        f"{path}:{awaiting['line_number']}: magnet pause "
                        f"commands {actual!r} != required {expected!r} "
                        f"(or modal restore {modal_restore!r})")
                events.append({
                    "z_mm": awaiting["z_mm"],
                    "line_number": awaiting["line_number"],
                    "custom_gcode_type": MAGNET_INSERTION_CUSTOM_GCODE_TYPE,
                    "command": MAGNET_INSERTION_PAUSE_COMMAND,
                    "park_z_mm": _float(
                        pause_policy.get("park_z_mm"),
                        "magnet insertion park Z"),
                    "park_command_line_number": commands[2][1],
                    "command_line_number": commands[4][1],
                    "restore_z_mm": awaiting["z_mm"],
                    "restore_command_line_number": commands[5][1],
                })
                awaiting = None
                continue
            if not awaiting["inside_program"]:
                if line and not line.startswith(";"):
                    raise AuditError(
                        f"{path}:{line_number}: magnet custom G-code has a "
                        "command before its park marker")
                continue
            command = line.split(";", 1)[0].strip()
            if not command:
                continue
            awaiting["commands"].append((" ".join(command.split()), line_number))
    if awaiting is not None:
        raise AuditError(
            f"{path}:{awaiting['line_number']}: custom magnet pause has no "
            "completed park/pause/restore program")
    return events


def _validate_magnet_pause_program_text(
    program: str,
    pause_z: float,
    pause_policy: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Require the self-contained custom event to be exactly motion-safe."""
    lines = [line.strip() for line in program.splitlines()]
    try:
        begin = lines.index(MAGNET_INSERTION_PARK_BEGIN)
        end = lines.index(MAGNET_INSERTION_PARK_END)
    except ValueError as exc:
        raise AuditError(f"{label} lacks the magnet park program markers") from exc
    if begin >= end:
        raise AuditError(f"{label} has invalid magnet park program marker order")
    commands = []
    for line in lines[begin + 1:end]:
        command = line.split(";", 1)[0].strip()
        if command:
            commands.append(" ".join(command.split()))
    expected = _magnet_pause_commands(pause_z, pause_policy)
    if commands != expected:
        raise AuditError(
            f"{label} commands {commands!r} != required {expected!r}")


def _local_xml_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _read_single_3mf_member(project_3mf: Path, member: str) -> bytes:
    try:
        with zipfile.ZipFile(project_3mf) as archive:
            if archive.namelist().count(member) != 1:
                raise AuditError(
                    f"{project_3mf}: expected exactly one {member}")
            return archive.read(member)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise AuditError(
            f"cannot read {member} from ready project {project_3mf}: {exc}") from exc


def _replace_single_3mf_member(
    project_3mf: Path,
    member: str,
    replacement_bytes: bytes,
) -> None:
    """Atomically replace one 3MF member while preserving every other member."""
    try:
        with zipfile.ZipFile(project_3mf) as archive:
            infos = archive.infolist()
            if sum(info.filename == member for info in infos) != 1:
                raise AuditError(
                    f"{project_3mf}: expected exactly one {member}")
            entries = [(info, archive.read(info.filename)) for info in infos]
            comment = archive.comment
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise AuditError(
            f"cannot rewrite ready project {project_3mf}: {exc}") from exc
    temporary = project_3mf.with_name(
        f".{project_3mf.name}.{Path(member).name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as replacement:
            replacement.comment = comment
            for info, payload in entries:
                replacement.writestr(
                    info,
                    replacement_bytes if info.filename == member else payload,
                    compress_type=info.compress_type)
        temporary.replace(project_3mf)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AuditError(
            f"cannot rewrite ready project {project_3mf}: {exc}") from exc


def _encode_ready_project_custom_gcode_newlines(project_3mf: Path) -> bool:
    """Use XML character references so reopening retains multiline custom G-code."""
    member = "Metadata/custom_gcode_per_layer.xml"
    source = _read_single_3mf_member(project_3mf, member)
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(
            f"{project_3mf}: custom G-code XML is not UTF-8") from exc

    def encode_attribute(match: re.Match[str]) -> str:
        name = match.group("name")
        value = match.group("value")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        return f'{name}="{value.replace("\n", "&#10;")}"'

    encoded = re.sub(
        r'(?P<name>extra|gcode)="(?P<value>[^"]*)"',
        encode_attribute, text, flags=re.DOTALL)
    if encoded == text:
        return False
    _replace_single_3mf_member(project_3mf, member, encoded.encode("utf-8"))
    return True


def _inject_ready_project_object_support(
    project_3mf: Path,
    *,
    enabled: bool,
) -> list[str]:
    """Persist object-level support for Bambu Studio reopening/re-slicing.

    Every support field is repeated on every model object as an explicit one
    or zero.  A reopened project therefore cannot inherit a stale object-level
    support policy after its global process preset changes.
    """
    expected_value = "1" if enabled else "0"
    member = "Metadata/model_settings.config"
    model_settings = _read_single_3mf_member(project_3mf, member)
    try:
        root = ET.fromstring(model_settings)
    except ET.ParseError as exc:
        raise AuditError(
            f"{project_3mf}: model settings XML is invalid") from exc
    objects = [element for element in list(root)
               if _local_xml_tag(element) == "object"]
    if not objects:
        raise AuditError(f"{project_3mf}: model settings contain no objects")
    object_ids = []
    changed = False
    for object_element in objects:
        object_id = object_element.attrib.get("id")
        if not object_id:
            raise AuditError(f"{project_3mf}: model settings object lacks id")
        object_ids.append(object_id)
        for key in SUPPORT_PROCESS_KEYS:
            metadata = [
                element for element in list(object_element)
                if (_local_xml_tag(element) == "metadata"
                    and element.attrib.get("key") == key)
            ]
            if len(metadata) > 1:
                raise AuditError(
                    f"{project_3mf}: object {object_id} has duplicate "
                    f"{key} metadata")
            if metadata:
                if metadata[0].attrib.get("value") != expected_value:
                    metadata[0].set("value", expected_value)
                    changed = True
                continue
            support_metadata = ET.Element(
                "metadata", {"key": key, "value": expected_value})
            insertion_index = 0
            for index, child in enumerate(list(object_element)):
                if _local_xml_tag(child) == "metadata":
                    insertion_index = index + 1
            object_element.insert(insertion_index, support_metadata)
            changed = True
    if not changed:
        return object_ids
    replacement_bytes = ET.tostring(
        root, encoding="utf-8", xml_declaration=True)
    _replace_single_3mf_member(project_3mf, member, replacement_bytes)
    return object_ids


def _validate_ready_project_object_support(
    model_settings: bytes,
    *,
    project_3mf: Path,
    enabled: bool,
) -> list[dict[str, str]]:
    """Verify all object-level support safety fields in a packed project."""
    expected_value = "1" if enabled else "0"
    try:
        root = ET.fromstring(model_settings)
    except ET.ParseError as exc:
        raise AuditError(
            f"{project_3mf}: model settings XML is invalid") from exc
    objects = [element for element in list(root)
               if _local_xml_tag(element) == "object"]
    if not objects:
        raise AuditError(f"{project_3mf}: model settings contain no objects")
    result = []
    for object_element in objects:
        object_id = object_element.attrib.get("id")
        record = {"object_id": str(object_id)}
        for key in SUPPORT_PROCESS_KEYS:
            values = [
                element.attrib.get("value")
                for element in list(object_element)
                if (_local_xml_tag(element) == "metadata"
                    and element.attrib.get("key") == key)
            ]
            if values != [expected_value]:
                raise AuditError(
                    f"{project_3mf}: object {object_id!r} must explicitly "
                    f"embed {key}={expected_value} for its resolved support "
                    "policy")
            record[key] = expected_value
        result.append(record)
    return result


def _assert_exact_pause_z(
    actual: Sequence[float], expected: Sequence[float], *, label: str,
) -> None:
    if len(actual) != len(expected) or any(
            not math.isclose(a, e, abs_tol=0.001, rel_tol=0.0)
            for a, e in zip(actual, expected, strict=True)):
        raise AuditError(
            f"{label} pause layers {list(actual)} != expected "
            f"{list(expected)}")


def _validate_ready_project_archive(
    project_3mf: Path,
    plain_gcode: Path,
    *,
    expected_pause_z: Sequence[float],
    profile_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the final 3MF embeds settings, G-code, and park/pause metadata."""
    required = (
        "Metadata/project_settings.config",
        "Metadata/model_settings.config",
        "Metadata/custom_gcode_per_layer.xml",
        "Metadata/plate_1.gcode",
    )
    try:
        with zipfile.ZipFile(project_3mf) as archive:
            names = archive.namelist()
            for name in required:
                if names.count(name) != 1:
                    raise AuditError(
                        f"{project_3mf}: expected exactly one {name}")
            settings_bytes = archive.read(required[0])
            model_settings = archive.read(required[1])
            custom_xml = archive.read(required[2])
            embedded_gcode = archive.read(required[3])
            corrupt = archive.testzip()
            if corrupt is not None:
                raise AuditError(
                    f"{project_3mf}: corrupt ZIP member {corrupt}")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise AuditError(
            f"cannot inspect ready-to-print 3MF {project_3mf}: {exc}") from exc

    plain_bytes = plain_gcode.read_bytes()
    if embedded_gcode != plain_bytes:
        raise AuditError(
            f"{project_3mf}: embedded plate_1.gcode differs from the "
            "validated plain G-code")
    try:
        settings = json.loads(settings_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(
            f"{project_3mf}: project settings are not valid JSON") from exc
    if not isinstance(settings, Mapping):
        raise AuditError(f"{project_3mf}: project settings are not an object")
    pause_policy = profile_bundle["identity"]["effective"].get(
        "magnet_insertion_pause")
    if not isinstance(pause_policy, Mapping):
        raise AuditError(
            f"{project_3mf}: profile has no magnet insertion pause policy")
    embedded_settings = {}
    for section, values in profile_bundle["enforced_overrides"].items():
        for key, expected in values.items():
            actual = settings.get(key)
            if not _profile_value_equal(actual, expected):
                raise AuditError(
                    f"{project_3mf}: embedded setting {key}={actual!r}, "
                    f"expected {expected!r} from {section} profile")
            embedded_settings[key] = actual
    object_support_overrides = _validate_ready_project_object_support(
        model_settings, project_3mf=project_3mf,
        enabled=bool(profile_bundle["identity"]["effective"].get(
            "support_enabled")))

    try:
        root = ET.fromstring(custom_xml)
    except ET.ParseError as exc:
        raise AuditError(
            f"{project_3mf}: custom pause XML is invalid") from exc
    local_tag = lambda element: element.tag.rsplit("}", 1)[-1]
    if local_tag(root) != "custom_gcodes_per_layer":
        raise AuditError(
            f"{project_3mf}: custom pause XML has unexpected root "
            f"{local_tag(root)!r}")
    plates = [element for element in list(root)
              if local_tag(element) == "plate"]
    if len(plates) != 1 or len(list(root)) != 1:
        raise AuditError(
            f"{project_3mf}: custom pause XML must contain exactly one "
            "direct plate")
    plate = plates[0]
    plate_info = [element for element in list(plate)
                  if local_tag(element) == "plate_info"]
    if (len(plate_info) != 1
            or plate_info[0].attrib.get("id") != "1"):
        raise AuditError(
            f"{project_3mf}: custom pause XML must contain exactly one "
            "plate_info id=1")
    layers = [element for element in list(plate)
              if local_tag(element) == "layer"]
    modes = [element.attrib.get("value") for element in list(plate)
             if local_tag(element) == "mode"]
    if modes != ["SingleExtruder"]:
        raise AuditError(
            f"{project_3mf}: custom G-code mode is {modes!r}, expected "
            "SingleExtruder")
    xml_z: list[float] = []
    for element in layers:
        if element.attrib.get("type") != "4":
            raise AuditError(
                f"{project_3mf}: custom layer is not Custom type 4")
        if element.attrib.get("extruder") != "1":
            raise AuditError(
                f"{project_3mf}: magnet custom G-code extruder is not 1")
        pause_z = _float(
            element.attrib.get("top_z"), "custom pause XML top_z")
        gcode_program = element.attrib.get("gcode", "")
        extra_program = element.attrib.get("extra", "")
        if gcode_program != extra_program:
            raise AuditError(
                f"{project_3mf}: custom G-code XML gcode and extra differ")
        _validate_magnet_pause_program_text(
            gcode_program, pause_z, pause_policy,
            label=f"{project_3mf}: custom G-code XML")
        xml_z.append(pause_z)
    _assert_exact_pause_z(
        xml_z, expected_pause_z, label="embedded custom XML")

    events = _gcode_pause_events(plain_gcode, pause_policy)
    _assert_exact_pause_z(
        [event["z_mm"] for event in events], expected_pause_z,
        label="ready G-code")
    return {
        "project_settings_member": required[0],
        "project_settings_sha256": _sha256_bytes(settings_bytes),
        "model_settings_member": required[1],
        "model_settings_sha256": _sha256_bytes(model_settings),
        "object_support_overrides": object_support_overrides,
        "enforced_project_settings": embedded_settings,
        "custom_gcode_member": required[2],
        "custom_gcode_xml_sha256": _sha256_bytes(custom_xml),
        "embedded_gcode_member": required[3],
        "embedded_gcode_sha256": _sha256_bytes(embedded_gcode),
        "magnet_insertion_pause": dict(pause_policy),
        "pause_z_mm": list(expected_pause_z),
        "gcode_pause_events": events,
    }


def _cached_ready_project_matches(
    prior: Mapping[str, Any],
    *,
    fingerprint: str,
    gcode: Path,
    result_path: Path,
    project_3mf: Path,
) -> bool:
    required = {
        "fingerprint": fingerprint,
        "gcode_sha256": sha256_file(gcode),
        "result_sha256": sha256_file(result_path),
        "project_3mf_sha256": sha256_file(project_3mf),
    }
    return all(prior.get(key) == value for key, value in required.items())


def _support_toolpath_summary(path: Path) -> dict[str, int]:
    """Count emitted Bambu support feature blocks, not merely support flags."""
    support = 0
    interface = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if not line.startswith("; FEATURE:"):
                continue
            feature = line.split(":", 1)[1].strip().lower()
            if feature.startswith("support"):
                support += 1
                if "interface" in feature:
                    interface += 1
    return {
        "support_feature_blocks": support,
        "support_interface_feature_blocks": interface,
    }


def _validate_ready_cavity_toolpaths(
    *,
    artifact: Mapping[str, Any],
    discovery_record: Mapping[str, Any],
    gcode: Path,
    stl_to_bed_matrix: BambuMatrix4,
) -> tuple[ParsedGcode, list[dict[str, Any]]]:
    """Re-run every cavity gate against the pause-bearing final G-code."""
    sites = [
        _site_in_bambu_bed_space(site, stl_to_bed_matrix)
        for site in artifact["sites"]
    ]
    parsed = parse_gcode(
        gcode, retain_regions=_cavity_retain_regions(sites, (0.0, 0.0)))
    discovery_by_name = {
        item["site"]["name"]: item for item in discovery_record["sites"]
    }
    results = []
    errors = []
    for site in sites:
        name = site["name"]
        expected = discovery_by_name.get(name)
        if expected is None:
            errors.append(f"{name}: missing discovery-pass site")
            continue
        selected, metrics, _closure = _discover_actual_closure_layers(
            parsed.layers, site, (0.0, 0.0))
        roof_pass, roof_detail = _roof_progression_pass(metrics)
        retaining_stage_pass = {
            stage: _retaining_stage_pass(site, stage, metrics[stage])
            for stage in (
                "lowest_open", "representative_open", "last_fully_open")
        }
        aperture_pass, aperture_detail = _loading_aperture_pass(
            site, metrics["last_fully_open"])
        actual_pause = selected["first_closing_pause"].z
        expected_pause = float(
            expected["actual"]["bambu_studio_pause_marker_z_mm"])
        same_pause = math.isclose(
            actual_pause, expected_pause, abs_tol=0.001, rel_tol=0.0)
        if not roof_pass:
            errors.append(f"{name}: final roof progression failed: {roof_detail}")
        if not all(retaining_stage_pass.values()):
            errors.append(
                f"{name}: final retaining path gate failed at "
                + ", ".join(stage for stage, passed
                            in retaining_stage_pass.items() if not passed))
        if not aperture_pass:
            errors.append(
                f"{name}: final loading aperture failed: {aperture_detail}")
        if not same_pause:
            errors.append(
                f"{name}: final closing layer {actual_pause:.3f} differs "
                f"from discovery {expected_pause:.3f}")
        results.append({
            "site": name,
            "first_closing_layer_z_mm": actual_pause,
            "discovery_first_closing_layer_z_mm": expected_pause,
            "same_closing_layer_pass": same_pause,
            "roof_progression_pass": roof_pass,
            "retaining_paths_stage_pass": retaining_stage_pass,
            "loading_aperture_pass": aperture_pass,
            "single_classic_path_pass": metrics[
                "last_fully_open"]["retaining_paths"][
                    "single_classic_path_pass"],
        })
    if errors:
        raise AuditError(
            f"{artifact['id']}: ready G-code cavity audit failed: "
            + "; ".join(errors))
    return parsed, results


def _assert_pauses_precede_layer_extrusion(
    parsed: ParsedGcode, events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence = []
    for event in events:
        matches = [layer for layer in parsed.layers if math.isclose(
            layer.z, float(event["z_mm"]), abs_tol=0.001, rel_tol=0.0)]
        if len(matches) != 1:
            raise AuditError(
                f"pause at Z={event['z_mm']} does not map to exactly one layer")
        layer = matches[0]
        first_extrusion = layer.first_extrusion_line_number
        if first_extrusion is None:
            raise AuditError(
                f"pause layer Z={layer.z:.3f} contains no extrusion")
        if int(event["command_line_number"]) >= first_extrusion:
            raise AuditError(
                f"pause M400 U1 at line {event['command_line_number']} is not "
                f"before first layer extrusion at line {first_extrusion}")
        evidence.append({
            "z_mm": layer.z,
            "pause_command_line_number": event["command_line_number"],
            "first_extrusion_line_number": first_extrusion,
            "pass": True,
        })
    return evidence


def _emit_ready_project(
    *,
    record: Mapping[str, Any],
    artifact: Mapping[str, Any],
    stl: Path,
    mesh: MeshFacts,
    ready_dir: Path,
    profile_bundle: Mapping[str, Any],
    bambu: Path,
    discovery_fingerprint: str,
    reuse: bool,
) -> dict[str, Any]:
    """Second pass: reslice the STL with discovered pauses embedded."""
    ready_dir.mkdir(parents=True, exist_ok=True)
    support_blockers = tuple(
        [artifact["support_blocker"]]
        if "support_blocker" in artifact else [])
    pause_policy = profile_bundle["identity"]["effective"].get(
        "magnet_insertion_pause")
    if not isinstance(pause_policy, Mapping):
        raise AuditError(
            f"{artifact['id']}: profile has no magnet insertion pause policy")
    custom_document, pause_z = _custom_gcodes_document(record, pause_policy)
    custom_path = ready_dir / "custom_gcodes.json"
    _write_json(custom_path, custom_document)
    gcode = ready_dir / "plate_1.gcode"
    result_path = ready_dir / "result.json"
    project_3mf = ready_dir / READY_3MF_FILENAME
    assemble_list = (
        ready_dir / "bambu_assemble_list.json"
        if support_blockers else None)
    if assemble_list is not None:
        _write_bambu_assemble_list(
            assemble_list, stl=stl,
            support_blockers=support_blockers)
    command = _bambu_command(
        bambu, stl, ready_dir, profile_bundle,
        project_filename=READY_3MF_FILENAME,
        custom_gcodes=custom_path,
        assemble_list=assemble_list)
    fingerprint = _sha256_bytes(_canonical_json({
        "discovery_fingerprint": discovery_fingerprint,
        "custom_gcodes_sha256": sha256_file(custom_path),
        "profile_set_sha256": profile_bundle[
            "identity"]["profile_set_sha256"],
        "bambu_binary_sha256": profile_bundle["identity"]["binary_sha256"],
        "stl_sha256": sha256_file(stl),
        "command": command,
    }))
    fingerprint_path = ready_dir / "ready_project_fingerprint.json"
    reused = False
    support_enabled = bool(profile_bundle["identity"]["effective"].get(
        "support_enabled"))
    if (reuse and fingerprint_path.is_file() and gcode.is_file()
            and result_path.is_file() and project_3mf.is_file()):
        prior = _load_json(fingerprint_path)
        if isinstance(prior, Mapping) and _cached_ready_project_matches(
                prior, fingerprint=fingerprint, gcode=gcode,
                result_path=result_path, project_3mf=project_3mf):
            reused = True
    if not reused:
        for stale in (gcode, result_path, project_3mf, fingerprint_path):
            stale.unlink(missing_ok=True)
        run = subprocess.run(
            command, cwd=ready_dir, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=int(profile_bundle["config"]["slicing"][
                "timeout_seconds"]),
            check=False, env={**os.environ, "LC_ALL": "C"})
        (ready_dir / "bambu_studio.log").write_text(
            run.stdout, encoding="utf-8", errors="replace")
        if run.returncode != 0:
            raise AuditError(
                f"{artifact['id']}: ready-project Bambu Studio pass exited "
                f"{run.returncode}; see {ready_dir / 'bambu_studio.log'}")
        if (not gcode.is_file() or not result_path.is_file()
                or not project_3mf.is_file()):
            raise AuditError(
                f"{artifact['id']}: ready-project pass did not create "
                "plate_1.gcode/result.json/ready_to_print.gcode.3mf")
        _encode_ready_project_custom_gcode_newlines(project_3mf)
        _inject_ready_project_object_support(
            project_3mf, enabled=support_enabled)
        _write_json(fingerprint_path, {
            "fingerprint": fingerprint,
            "command": command,
            "custom_gcodes_sha256": sha256_file(custom_path),
            "gcode_sha256": sha256_file(gcode),
            "result_sha256": sha256_file(result_path),
            "project_3mf_sha256": sha256_file(project_3mf),
        })
    _validate_bambu_slicer_log(
        ready_dir / "bambu_studio.log",
        artifact_id=artifact["id"],
        phase="ready-project slice")

    result = _load_json(result_path)
    if result.get("return_code") != 0:
        raise AuditError(
            f"{artifact['id']}: ready-project slicer result is not Success")
    plates = result.get("sliced_plates")
    if not isinstance(plates, list) or len(plates) != 1:
        raise AuditError(
            f"{artifact['id']}: ready project must contain one sliced plate")
    objects = plates[0].get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise AuditError(
            f"{artifact['id']}: ready project must contain one sliced object")
    if (int(plates[0].get("triangle_count", -1)) != mesh.triangle_count
            or int(objects[0].get("triangle_count", -1))
            != mesh.triangle_count):
        raise AuditError(
            f"{artifact['id']}: ready project triangle count differs from STL")
    try:
        project_audit = audit_bambu_3mf(
            project_3mf, stl,
            support_blocker_stls=support_blockers)
        ready_bbox = objects[0].get("bbox")
        if not isinstance(ready_bbox, Mapping):
            raise Bambu3MFAuditError("ready result lacks an object bbox")
        validate_bambu_result_bbox(
            ready_bbox, project_audit.source_bounds,
            project_audit.stl_to_bed_matrix)
        validate_bambu_bed_fit(
            project_audit.transformed_actual_mesh_bounds,
            profile_bundle["identity"]["machine_bounds_mm"])
    except Bambu3MFAuditError as exc:
        raise AuditError(
            f"{artifact['id']}: ready 3MF placement/mesh audit failed: "
            f"{exc}") from exc
    discovery_matrix = record["slicer"]["bambu_3mf_audit"][
        "stl_to_bed_matrix"]
    matrix_delta = max(
        abs(float(actual) - float(expected))
        for actual_row, expected_row in zip(
            project_audit.stl_to_bed_matrix, discovery_matrix, strict=True)
        for actual, expected in zip(actual_row, expected_row, strict=True)
    )
    if matrix_delta > 1.0e-6:
        raise AuditError(
            f"{artifact['id']}: ready-pass STL placement differs from "
            f"discovery by {matrix_delta:.9f}")
    parsed, cavity_audit = _validate_ready_cavity_toolpaths(
        artifact=artifact, discovery_record=record, gcode=gcode,
        stl_to_bed_matrix=project_audit.stl_to_bed_matrix)
    profile_errors = _validate_actual_gcode_profile(parsed, profile_bundle)
    if profile_errors:
        raise AuditError(
            f"{artifact['id']}: ready G-code profile mismatch: "
            + "; ".join(profile_errors))
    support_toolpaths = _support_toolpath_summary(gcode)
    if support_enabled and support_toolpaths["support_feature_blocks"] <= 0:
        raise AuditError(
            f"{artifact['id']}: support is enabled but the ready G-code has "
            "no emitted Support feature blocks")
    if not support_enabled and any(support_toolpaths.values()):
        raise AuditError(
            f"{artifact['id']}: support is disabled but the ready G-code "
            "contains support feature blocks")
    if support_enabled:
        contract = artifact.get("duct_collision_contract")
        if not isinstance(contract, Mapping):
            raise AuditError(
                f"{artifact['id']}: support-enabled duct-bearing part lacks "
                "a hash-bound duct collision contract")
        try:
            duct_support_toolpath_audit = audit_support_toolpaths_vs_ducts(
                gcode=gcode,
                contract=contract,
                source_to_stl_matrix=artifact["source_to_stl_matrix"],
                stl_to_bed_matrix=project_audit.stl_to_bed_matrix,
            )
        except AuditError as exc:
            raise AuditError(
                f"{artifact['id']}: support-vs-duct collision gate failed: "
                f"{exc}") from exc
    else:
        duct_support_toolpath_audit = {
            "status": "pass",
            "gate": "support_disabled_no_support_feature_blocks",
            "support_extrusion_segments_checked": 0,
            "collision_count": 0,
        }
    archive_audit = _validate_ready_project_archive(
        project_3mf, gcode, expected_pause_z=pause_z,
        profile_bundle=profile_bundle)
    pause_before_extrusion = _assert_pauses_precede_layer_extrusion(
        parsed, archive_audit["gcode_pause_events"])
    skill_validation = _validate_with_gcode_skill(
        gcode, ready_dir, profile_bundle)
    if skill_validation.get("ok") is not True:
        raise AuditError(
            f"{artifact['id']}: ready G-code static validation did not pass")
    output_hashes = {
        "custom_gcodes_sha256": sha256_file(custom_path),
        "result_sha256": sha256_file(result_path),
        "gcode_sha256": sha256_file(gcode),
        "project_3mf_sha256": sha256_file(project_3mf),
    }
    output_fingerprint = _sha256_bytes(_canonical_json({
        "input_fingerprint": fingerprint,
        **output_hashes,
        "archive_audit": archive_audit,
        "cavity_toolpath_audit": cavity_audit,
        "duct_support_toolpath_audit": duct_support_toolpath_audit,
        "pause_before_first_layer_extrusion": pause_before_extrusion,
    }))
    placement_audit = project_audit.as_record()
    placement_audit.pop("staged_stl", None)
    placement_audit["max_matrix_delta_from_discovery"] = matrix_delta
    return {
        "status": "pass",
        "reused": reused,
        "direct_stl_reslice": True,
        "auto_rotation_disabled": "--allow-rotations=0" in command,
        "command": command,
        "input_fingerprint": fingerprint,
        "output_fingerprint": output_fingerprint,
        "custom_gcodes_json": str(custom_path),
        "result_json": str(result_path),
        "gcode": str(gcode),
        "project_3mf": str(project_3mf),
        **output_hashes,
        "pause_z_mm": pause_z,
        "archive_audit": archive_audit,
        "support_toolpaths": support_toolpaths,
        "duct_support_toolpath_audit": duct_support_toolpath_audit,
        "bambu_3mf_audit": placement_audit,
        "cavity_toolpath_audit": cavity_audit,
        "pause_before_first_layer_extrusion": pause_before_extrusion,
        "gcode_skill_validation": skill_validation,
    }


def _write_manifest_bundle(
    output: Path,
    catalog_path: Path,
    catalog: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]] = (),
) -> dict[str, Path]:
    all_groups = []
    for record in records:
        for group in _pause_groups(record):
            discovery = record["slicer"]
            ready = discovery.get("ready_project")
            ready_pass = (
                isinstance(ready, Mapping) and ready.get("status") == "pass")
            primary = ready if ready_pass else discovery
            placement_audit = (
                ready["bambu_3mf_audit"] if ready_pass
                else discovery["bambu_3mf_audit"])
            all_groups.append({
                "artifact_id": record["id"],
                "state": record["state"],
                "variant": record["variant"],
                "part": record["part"],
                "print_orientation": record["print_orientation"],
                "stl": record["input"]["stl"],
                "stl_sha256": record["input"]["stl_sha256"],
                "gcode": primary["gcode"],
                "gcode_sha256": primary["gcode_sha256"],
                "audited_bambu_3mf": primary["project_3mf"],
                "audited_bambu_3mf_sha256": primary[
                    "project_3mf_sha256"],
                "ready_project": ready_pass,
                "ready_project_output_fingerprint": (
                    ready.get("output_fingerprint") if ready_pass else None),
                "discovery_gcode": discovery["gcode"],
                "discovery_gcode_sha256": discovery["gcode_sha256"],
                "discovery_bambu_3mf": discovery["project_3mf"],
                "discovery_bambu_3mf_sha256": discovery[
                    "project_3mf_sha256"],
                "bambu_arrange_rz_degrees": placement_audit[
                    "rigid_rz"]["rz_degrees"],
                **group,
            })
    actual_slice_records = [
        record for record in records
        if record.get("audit_mode") == "actual_p2s_slice"
    ]
    oversize_records = [
        record for record in records
        if record.get("audit_mode") == "exact_split_proxy_coverage"
    ]
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "authoritative": True,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "safety_boundary": (
            "local slicing and static inspection only; no printer upload, "
            "MQTT, FTPS, or print start"),
        "catalog": {
            "path": str(catalog_path),
            "sha256": catalog["_catalog_sha256"],
            "source_revision": catalog.get("source_revision"),
        },
        "profile": profile_bundle["identity"],
        "summary": {
            "catalog_artifact_count": catalog.get(
                "inventory", {}).get("artifact_count"),
            "catalog_magnet_station_count": catalog.get(
                "inventory", {}).get("magnet_count"),
            "requested_artifact_count": len(records) + len(failures),
            "sliced_artifact_count": len(actual_slice_records),
            "p2s_oversize_artifact_count": len(oversize_records),
            "p2s_oversize_exact_split_covered": sum(
                record.get("status") == OVERSIZE_COVERED_STATUS
                for record in oversize_records),
            "pause_group_count": len(all_groups),
            "ready_project_count": sum(
                record.get("slicer", {}).get(
                    "ready_project", {}).get("status") == "pass"
                for record in actual_slice_records),
            "magnet_count": sum(group["magnet_count"] for group in all_groups),
            "p2s_pause_magnet_count": sum(
                group["magnet_count"] for group in all_groups),
            "oversize_proxy_covered_site_count": sum(
                len(record.get("cavity_audit_coverage", {}).get("sites", ()))
                for record in oversize_records),
            "passed_artifacts": sum(
                record.get("status") in ("pass", OVERSIZE_COVERED_STATUS)
                for record in records),
            "failed_artifacts": (
                sum(record.get("status") not in (
                    "pass", OVERSIZE_COVERED_STATUS)
                    for record in records)
                + len(failures)),
        },
        "slice_failures": list(failures),
        "pause_groups": all_groups,
        "artifacts": records,
    }
    json_path = output / "captive_magnet_pause_manifest.json"
    _write_json(json_path, manifest)
    csv_path = output / "captive_magnet_pause_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "artifact_id", "state", "variant", "part", "print_orientation",
            "group", "sites", "magnet_count",
            "last_completely_open_layer_z_mm",
            "cavity_bury_roof_start_plane_z_mm",
            "first_closing_layer_z_mm", "pause_marker_z_mm",
            "minimum_seated_below_last_open_layer_mm",
            "minimum_seated_below_first_closing_layer_mm",
            "print_insertion_direction_xyz", "insertion_instruction",
            "stl", "stl_sha256", "gcode", "gcode_sha256",
            "audited_bambu_3mf", "audited_bambu_3mf_sha256",
            "ready_project", "ready_project_output_fingerprint",
            "discovery_gcode", "discovery_gcode_sha256",
            "discovery_bambu_3mf", "discovery_bambu_3mf_sha256",
            "bambu_arrange_rz_degrees", "polarity")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for group in all_groups:
            writer.writerow({
                **{key: group.get(key) for key in fields},
                "sites": "; ".join(group["sites"]),
                "cavity_bury_roof_start_plane_z_mm": "; ".join(
                    f"{value:.2f}" for value in group[
                        "cavity_bury_roof_start_plane_z_mm"]),
                "print_insertion_direction_xyz": json.dumps(
                    group["print_insertion_direction_xyz"],
                    separators=(",", ":")),
                "polarity": json.dumps(group["polarity"], separators=(",", ":")),
            })
    md_path = output / "CAPTIVE_MAGNET_PAUSE_MANIFEST.md"
    lines = [
        "# Captive-magnet pause manifest",
        "",
        ("Authoritative for the exact STL and profile hashes below. This run "
         "used Bambu Lab P2S, 0.4 mm nozzle, 0.16 mm High Quality, Arachne "
         "walls, and Bambu PLA Tough+. All parts are front-face-down."),
        "",
        "This audit did not contact a printer and did not upload or start a print.",
        "",
        "## Insertion procedure",
        "",
        "1. Open the hash-listed ready-to-print 3MF; do not auto-orient it.",
        ("2. The Bambu Custom park/pause/restore events at the exact "
         "**first-closing** Z values below are already embedded and were "
         "verified in both project XML and G-code; do not add or move them "
         "manually. Each raises the nozzle to Z=250 mm (lowering the bed), "
         "pauses with `M400 U1`, then restores the exact layer Z on "
         "Continue."),
        ("3. At each pause, insert the listed number of D5 x 2 mm magnets "
         "vertically downward from above (+Z side) along print `-Z` "
         "(`print_insertion_direction_xyz = [0, 0, -1]`), with the marked "
         "pole oriented exactly as specified."),
        "4. Ensure every magnet is fully seated below the completed layer and cannot rise into the toolhead path.",
        "5. Resume printing. Polarity cannot be corrected after the roof buries the magnet.",
        "",
        "## Exact pauses",
        "",
        "| State | Variant / part | Pause Z | Last open | Seated margin | Magnets / sites | Insertion | Polarity |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for group in all_groups:
        polarity = "<br>".join(
            (f"`{item['site']}`: marked pole → "
             f"`{item['print_marked_pole_axis_xyz']}` in print coordinates; "
             f"{item['instruction']}")
            for item in group["polarity"])
        lines.append(
            f"| {group['state']} | {group['variant']} / `{group['part']}` | "
            f"**{group['pause_marker_z_mm']:.2f} mm** | "
            f"{group['last_completely_open_layer_z_mm']:.2f} mm | "
            f"{group['minimum_seated_below_last_open_layer_mm']:.2f} mm | "
            f"{group['magnet_count']} / {', '.join(group['sites'])} | "
            f"`{group['print_insertion_direction_xyz']}`: "
            f"{group['insertion_instruction']} | {polarity} |")
    placement_groups = {
        group["artifact_id"]: group for group in all_groups
    }
    lines.extend((
        "",
        "## Audited Bambu arrangements",
        "",
        ("Every listed 3MF was exported by the same Bambu slice invocation, "
         "hash-bound to the staged STL, and audited as an exact mesh with "
         "only a proper unit-scale rotation about print Z plus XY placement."),
        "",
        "| State | Variant / part | Arrange Rz | Ready-to-print 3MF | SHA-256 | Ready fingerprint |",
        "|---|---|---:|---|---|---|",
    ))
    for group in placement_groups.values():
        lines.append(
            f"| {group['state']} | {group['variant']} / "
            f"`{group['part']}` | "
            f"{group['bambu_arrange_rz_degrees']:.6f} deg | "
            f"`{group['audited_bambu_3mf']}` | "
            f"`{group['audited_bambu_3mf_sha256']}` | "
            f"`{group['ready_project_output_fingerprint']}` |")
    if oversize_records:
        lines.extend((
            "",
            "## Explicitly not P2S-printable",
            "",
            ("These canonical monoliths exceed the P2S bed in their mandatory "
             "front-face-down orientation. They have no generated monolith "
             "G-code and no pause group. Their cavity evidence comes only "
             "from the exact same-state keyed split prints listed below."),
            "",
            "| State | Canonical part | Front-down size | Coverage status | Exact split proxies |",
            "|---|---|---|---|---|",
        ))
        for record in oversize_records:
            proxies = record.get("cavity_audit_coverage", {}).get("sites", ())
            proxy_text = ", ".join(
                f"`{item['site']}` → `{item['proxy_artifact_id']}`"
                for item in proxies) or "none"
            size = " × ".join(
                f"{float(value):.2f}"
                for value in record["input"]["size_mm"])
            lines.append(
                f"| {record['state']} | `{record['part']}` | {size} mm | "
                f"`{record['status']}` | {proxy_text} |")
    lines.extend((
        "",
        "## Profile and evidence",
        "",
        f"- Catalog SHA-256: `{manifest['catalog']['sha256']}`",
        f"- Resolved profile-set SHA-256: `{manifest['profile']['profile_set_sha256']}`",
        f"- Bambu Studio binary SHA-256: `{manifest['profile']['binary_sha256']}`",
        f"- Artifacts: {manifest['summary']['passed_artifacts']} passed, "
        f"{manifest['summary']['failed_artifacts']} failed",
        ("- Each printable artifact directory under `slices/` contains the "
         "hash-bound arranged Bambu 3MF, plain G-code, Bambu `result.json`, "
         "static validator output, and five-layer SVG/PNG toolpath evidence "
         "for every cavity."),
        "",
        "The JSON file is the machine-readable authority; this Markdown and the CSV are derived views.",
        "",
    ))
    if failures:
        lines.extend(("## Slice failures", ""))
        for failure in failures:
            lines.append(f"- `{failure['id']}`: {failure['error']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _transactional_publish_bundle(
    paths: Mapping[str, Path], destination: Path,
) -> dict[str, Path]:
    """Replace the canonical three-file set, rolling back any process error."""
    destination.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(
        prefix=".captive-manifest-backup-", dir=destination))
    targets = {key: destination / path.name for key, path in paths.items()}
    backups: dict[str, Path] = {}
    installed: list[str] = []
    retain_backup = False
    try:
        for key, target in targets.items():
            if target.exists():
                backup = backup_dir / target.name
                os.replace(target, backup)
                backups[key] = backup
        for key, staged_path in paths.items():
            os.replace(staged_path, targets[key])
            installed.append(key)
    except Exception as exc:
        restore_errors = []
        # A backup can replace a newly installed file atomically.  Only a
        # target that did not exist before this transaction needs deletion.
        for key in installed:
            if key in backups:
                continue
            try:
                targets[key].unlink(missing_ok=True)
            except Exception as remove_exc:  # pragma: no cover - catastrophic FS
                restore_errors.append(str(remove_exc))
        for key, backup in backups.items():
            try:
                os.replace(backup, targets[key])
            except Exception as restore_exc:  # pragma: no cover - catastrophic FS
                restore_errors.append(str(restore_exc))
        detail = ("; rollback errors: " + "; ".join(restore_errors)
                  if restore_errors else "")
        retain_backup = bool(restore_errors)
        raise AuditError(
            f"canonical manifest transaction failed: {exc}{detail}"
            + (f"; retained backups at {backup_dir}"
               if retain_backup else "")) from exc
    finally:
        if not retain_backup:
            shutil.rmtree(backup_dir, ignore_errors=True)
    return targets


def write_manifests(
    output: Path,
    catalog_path: Path,
    catalog: Mapping[str, Any],
    profile_bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]] = (),
) -> dict[str, Path]:
    """Validate, stage, then transactionally publish canonical manifests."""
    _validate_complete_release(
        catalog, records, failures, require_ready_projects=True)
    if sha256_file(catalog_path) != catalog.get("_catalog_sha256"):
        raise AuditError("release catalog changed before manifest publication")
    if sha256_file(CATALOG_SCHEMA) != catalog.get(
            "_catalog_schema_sha256"):
        raise AuditError(
            "release catalog schema changed before manifest publication")
    for artifact in catalog["artifacts"]:
        _validate_artifact_bindings(artifact)
    _verify_profile_inputs(
        profile_bundle, Path(profile_bundle["identity"]["binary"]))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".captive-manifest-stage-", dir=output) as directory:
        staged_paths = _write_manifest_bundle(
            Path(directory), catalog_path, catalog, profile_bundle,
            records, failures)
        _validate_manifest_bundle(staged_paths)
        _validate_complete_release(
            catalog, records, failures, require_ready_projects=True)
        if (sha256_file(catalog_path) != catalog["_catalog_sha256"]
                or sha256_file(CATALOG_SCHEMA)
                != catalog["_catalog_schema_sha256"]):
            raise AuditError(
                "release catalog authority changed during manifest staging")
        for artifact in catalog["artifacts"]:
            _validate_artifact_bindings(artifact)
        _verify_profile_inputs(
            profile_bundle, Path(profile_bundle["identity"]["binary"]))
        return _transactional_publish_bundle(staged_paths, output)


__all__ = tuple(
    name for name in globals()
    if name != "__all__" and not name.startswith("__"))
