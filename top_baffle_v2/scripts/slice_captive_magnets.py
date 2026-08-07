#!/usr/bin/env python3
"""Offline captive-magnet slicer CLI and compatibility API facade."""

from __future__ import annotations

from release_validation import *
from gcode_analysis import *
from artifact_emit import *


def _filter_artifacts(
    artifacts: Sequence[Mapping[str, Any]], patterns: Sequence[str],
) -> list[Mapping[str, Any]]:
    if not patterns:
        return list(artifacts)
    selected = []
    for artifact in artifacts:
        haystacks = (artifact["id"], artifact["part"], artifact["variant"],
                     str(artifact["stl"]))
        if any(fnmatch.fnmatch(value, pattern)
               for pattern in patterns for value in haystacks):
            selected.append(artifact)
    if not selected:
        raise AuditError(f"--only patterns matched no artifacts: {patterns}")
    return selected


def _validate_artifact_override_coverage(
    artifacts: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> None:
    """Fail if a supposedly exact artifact override is stale or ambiguous."""
    _validate_support_override_policy(config)
    rules = config.get("artifact_overrides", [])
    if not isinstance(rules, list):
        raise AuditError("artifact_overrides must be an array")
    matched_by_artifact: dict[str, list[int]] = {}
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping) or not isinstance(
                rule.get("match"), Mapping):
            raise AuditError(f"artifact_overrides[{index}] is invalid")
        match = rule["match"]
        matches = [artifact for artifact in artifacts if all(
            artifact.get(key) == expected for key, expected in match.items())]
        if len(matches) != 1:
            raise AuditError(
                f"artifact_overrides[{index}] match {dict(match)!r} resolved "
                f"to {len(matches)} catalog artifacts; expected exactly one")
        matched_by_artifact.setdefault(matches[0]["id"], []).append(index)
    ambiguous = {
        artifact_id: indexes for artifact_id, indexes
        in matched_by_artifact.items() if len(indexes) > 1
    }
    if ambiguous:
        raise AuditError(
            f"multiple artifact overrides target the same artifact: {ambiguous}")


def _authoritative_run_requested(
    only_patterns: Sequence[str], dry_run: bool,
) -> bool:
    """Only an unfiltered, executed release audit may publish pauses."""
    return not only_patterns and not dry_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Bambu P2S captive-magnet slicing and pause audit")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bambu-studio")
    parser.add_argument("--bambu-system-root", type=Path)
    parser.add_argument("--only", action="append", default=[],
                        help="glob against artifact id/part/variant/STL; repeatable")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel local Bambu Studio processes (default 1)")
    parser.add_argument("--no-reuse", action="store_true",
                        help="ignore content-addressed completed slices")
    parser.add_argument("--prepare-profiles-only", action="store_true")
    parser.add_argument(
        "--auxiliary-catalog", action="store_true",
        help=("consume one isolated optional-artifact catalog without "
              "weakening the protected release inventory gate"))
    parser.add_argument(
        "--emit-ready-projects", action="store_true",
        help=("after discovery, reslice each direct STL with exact Bambu "
              "Custom magnet park/pause/restore events and export "
              "ready_to_print.gcode.3mf"))
    parser.add_argument("--dry-run", action="store_true",
                        help="write resolved profiles and print commands only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise AuditError("--jobs must be positive")
    if args.emit_ready_projects and args.dry_run:
        raise AuditError(
            "--emit-ready-projects requires executed discovery slices; "
            "it cannot be combined with --dry-run")
    if (not args.only and not args.dry_run and not args.prepare_profiles_only
            and not args.emit_ready_projects):
        raise AuditError(
            "an unfiltered authoritative run requires "
            "--emit-ready-projects; discovery-only/manual-pause output is "
            "allowed only for explicit --only diagnostic subsets")
    authoritative_request = _authoritative_run_requested(
        args.only, args.dry_run)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bambu = _find_bambu_binary(args.bambu_studio)
    profile_bundle = prepare_profiles(
        args.profile.expanduser().resolve(), output,
        system_root=(args.bambu_system_root.expanduser().resolve()
                     if args.bambu_system_root else None),
        bambu_binary=bambu)
    if args.prepare_profiles_only:
        print(output / "profiles" / "profile_provenance.json")
        return 0
    catalog_path = args.catalog.expanduser().resolve()
    profile_mode = profile_bundle["config"].get("catalog_mode", "release")
    expected_mode = "auxiliary" if args.auxiliary_catalog else "release"
    if profile_mode != expected_mode:
        raise AuditError(
            f"profile catalog_mode {profile_mode!r} does not match "
            f"requested {expected_mode!r} slicing")
    if args.auxiliary_catalog and output == DEFAULT_OUTPUT.resolve():
        raise AuditError(
            "an auxiliary catalog must use an isolated --output directory")
    catalog = normalize_catalog(
        catalog_path,
        enforce_release_inventory=not args.auxiliary_catalog,
    )
    _validate_artifact_override_coverage(
        catalog["artifacts"], profile_bundle["config"])
    _validate_parameter_modifier_coverage(
        catalog["artifacts"], profile_bundle)
    selected = _filter_artifacts(catalog["artifacts"], args.only)
    _validate_profile_artifact_scope(
        selected, profile_bundle["config"])
    catalog_by_id = {
        artifact["id"]: artifact for artifact in catalog["artifacts"]
    }
    # Selecting an oversized monolith implicitly selects every exact split
    # dependency needed to prove its cavities.  This never selects a virtual,
    # scaled, tilted, or clipped version of the monolith itself.
    selected_by_id = {artifact["id"]: artifact for artifact in selected}
    for artifact in tuple(selected):
        for proxy in artifact.get("cavity_audit_proxies", ()):
            selected_by_id[proxy["artifact_id"]] = catalog_by_id[
                proxy["artifact_id"]]
    requested_artifacts = sorted(
        selected_by_id.values(), key=lambda item: item["id"])
    with tempfile.TemporaryDirectory(
            prefix=".captive-input-stage-", dir=output) as stage_directory:
        staged = _stage_release_inputs(
            catalog_path, requested_artifacts, Path(stage_directory),
            expected_catalog_sha256=catalog["_catalog_sha256"],
            expected_catalog_schema_sha256=catalog[
                "_catalog_schema_sha256"])
        artifacts = staged["artifacts"]
        oversized = [
            artifact for artifact in artifacts
            if artifact.get("p2s_printability") == "not_printable_oversize"
        ]
        slice_targets = [
            artifact for artifact in artifacts if artifact not in oversized
        ]
        catalog_sha = staged["catalog_sha256"]
        records: list[Mapping[str, Any]] = []
        failures: list[dict[str, str]] = []

        def work(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
            return _slice_one(
                artifact, output_root=output, profile_bundle=profile_bundle,
                bambu=bambu, catalog_sha=catalog_sha,
                reuse=not args.no_reuse, dry_run=args.dry_run,
                emit_ready_projects=args.emit_ready_projects)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.jobs) as pool:
            future_map = {
                pool.submit(work, artifact): artifact
                for artifact in slice_targets
            }
            for future in concurrent.futures.as_completed(future_map):
                artifact = future_map[future]
                try:
                    record = future.result()
                    records.append(record)
                    print(
                        f"{record['id']}: "
                        f"{record.get('status', 'dry-run')}", flush=True)
                except Exception as exc:  # keep auditing independent parts
                    failures.append({
                        "id": artifact["id"], "error": str(exc)})
                    print(
                        f"{artifact['id']}: ERROR: {exc}",
                        file=sys.stderr, flush=True)

        _verify_staged_release_inputs(staged, catalog_path)
        _verify_profile_inputs(profile_bundle, bambu)
        if args.dry_run:
            _write_json(output / "dry_run_commands.json", {
                "authoritative": False,
                "canonical_pause_manifest_published": False,
                "catalog_sha256": catalog_sha,
                "profile": profile_bundle["identity"],
                "records": records,
                "oversize_not_sliced": [{
                    "id": artifact["id"],
                    "p2s_printable": False,
                    "proxy_artifact_ids": sorted({
                        proxy["artifact_id"]
                        for proxy in artifact["cavity_audit_proxies"]
                    }),
                    "policy": "no monolith G-code or pause group",
                } for artifact in oversized],
                "failures": failures,
            })
            return 1 if failures else 0

        records_by_id = {record["id"]: record for record in records}
        for artifact in oversized:
            records.append(_oversize_proxy_coverage_record(
                artifact, records_by_id, profile_bundle))
        records.sort(key=lambda item: item["id"])
        # Oversize coverage reads proxy evidence after the first immutable
        # verification, so verify every authority once more before any output
        # is eligible to become canonical.
        _verify_staged_release_inputs(staged, catalog_path)
        _verify_profile_inputs(profile_bundle, bambu)
        failed_records = [
            record for record in records
            if record.get("status") not in (
                "pass", OVERSIZE_COVERED_STATUS)
        ]
        if args.emit_ready_projects:
            missing_ready = [
                record["id"] for record in records
                if (record.get("audit_mode") == "actual_p2s_slice"
                    and record.get("status") == "pass"
                    and record.get("slicer", {}).get(
                        "ready_project", {}).get("status") != "pass")
            ]
            if missing_ready:
                failures.append({
                    "id": "ready-project-coverage",
                    "error": (
                        "passing actual slices lack validated ready projects: "
                        + ", ".join(missing_ready)),
                })

        if not authoritative_request:
            # Dry runs returned above, so the remaining non-authoritative
            # mode is necessarily an explicit --only subset.
            if not args.only:
                raise AuditError(
                    "internal authority classification is inconsistent")
            # A filtered audit is useful diagnostics but can never be mistaken
            # for the release-wide pause authority.
            subset_path = output / "subset_slice_results.json"
            _write_json(subset_path, {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "authoritative": False,
                "canonical_pause_manifest_published": False,
                "catalog_sha256": catalog_sha,
                "requested_patterns": list(args.only),
                "requested_artifact_ids": [
                    artifact["id"] for artifact in artifacts],
                "records": records,
                "failures": failures,
                "pause_groups": [],
                "note": (
                    "Subset audits never publish pause instructions; run the "
                    "complete unfiltered release audit for authority."),
            })
            print(
                f"subset (non-authoritative): {subset_path}\n"
                "canonical pause manifests were not modified")
            return 1 if failures or failed_records else 0

        if failures or failed_records:
            stamp = dt.datetime.now(dt.timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ")
            failure_path = (
                output / "failed_runs" / f"failed_slice_{stamp}.json")
            _write_json(failure_path, {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "authoritative": False,
                "canonical_pause_manifest_published": False,
                "catalog_sha256": catalog_sha,
                "records": records,
                "failures": failures,
                "pause_groups": [],
            })
            print(
                f"release audit failed: {failure_path}\n"
                "canonical pause manifests were not modified",
                file=sys.stderr)
            return 1

        _validate_complete_release(
            catalog, records, failures,
            enforce_expected_inventory=not args.auxiliary_catalog,
            require_ready_projects=True)
        paths = write_manifests(
            output, catalog_path, catalog, profile_bundle, records, failures,
            enforce_expected_inventory=not args.auxiliary_catalog)
        print("\n".join(
            f"{key}: {path}" for key, path in paths.items()))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
