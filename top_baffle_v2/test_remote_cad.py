"""Lightweight regression checks for the resumable remote CAD transport."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace

import remote_cad as remote


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_source_tree(
        work: Path, files: dict[str, bytes], *, mtime_ns: int) -> str:
    records = []
    for relative, data in sorted(files.items()):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.utime(path, ns=(mtime_ns, mtime_ns))
        records.append({
            "path": relative,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    source_hash = remote._source_identity(records)
    _write(work / ".lx-cad-source.json", json.dumps({
        "protocol_version": remote.PROTOCOL_VERSION,
        "source_sha256": source_hash,
        "files": records,
    }))
    os.utime(
        work / ".lx-cad-source.json", ns=(mtime_ns, mtime_ns))
    return source_hash


def _completed_cache_job(
        root: Path, job_id: str, *, source_files: dict[str, bytes],
        source_mtime_ns: int, output_data: bytes,
        output_mtime_ns: int, completed_ns: int,
        attestation_hash: str = "a" * 64) -> tuple[Path, dict]:
    job = root / "jobs" / job_id
    work = job / "work"
    source_hash = _write_source_tree(
        work, source_files, mtime_ns=source_mtime_ns)
    output = (
        work / "top_baffle_v2" / "floor_stand" / "stl" / "part.stl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(output_data)
    os.utime(output, ns=(output_mtime_ns, output_mtime_ns))
    environment_hash = "e" * 64
    metadata = {
        "protocol_version": remote.PROTOCOL_VERSION,
        "job_id": job_id,
        "remote_root": str(root),
        "source_sha256": source_hash,
        "environment_sha256": environment_hash,
        "environment_attestation_sha256": attestation_hash,
        "targets": ["floor_stand"],
        "include_candidate_outputs": False,
    }
    _write(job / "job.json", json.dumps(metadata))
    _write(job / "status.json", json.dumps({
        "protocol_version": remote.PROTOCOL_VERSION,
        "state": "succeeded", "exit_code": 0,
    }))
    _write(job / "exit_code", "0\n")
    archive = job / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(
            output,
            arcname=output.relative_to(work).as_posix(),
            recursive=False)
    artifact_record = {
        "path": output.relative_to(work).as_posix(),
        "size": len(output_data),
        "sha256": hashlib.sha256(output_data).hexdigest(),
    }
    _write(job / "artifacts.json", json.dumps({
        "protocol_version": remote.PROTOCOL_VERSION,
        "job_id": job_id,
        "source_sha256": source_hash,
        "environment_sha256": environment_hash,
        "environment_attestation_sha256": attestation_hash,
        "targets": ["floor_stand"],
        "build_completed_ns": completed_ns,
        "archive_sha256": remote._sha256_file(archive),
        "files": [artifact_record],
    }))
    return job, metadata


def _rewrite_job_artifacts(
        job: Path, metadata: dict, *, completed_ns: int) -> None:
    """Write the exact archive/manifest for a pure cache-publication fixture."""
    paths = remote._artifact_paths(job, metadata)
    archive = job / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in paths:
            bundle.add(
                path, arcname=path.relative_to(job / "work").as_posix(),
                recursive=False)
    records = [{
        "path": path.relative_to(job / "work").as_posix(),
        "size": path.stat().st_size,
        "sha256": remote._sha256_file(path),
    } for path in paths]
    _write(job / "artifacts.json", json.dumps({
        "protocol_version": remote.PROTOCOL_VERSION,
        "job_id": metadata["job_id"],
        "source_sha256": metadata["source_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "environment_attestation_sha256": metadata[
            "environment_attestation_sha256"],
        "targets": metadata["targets"],
        "build_completed_ns": completed_ns,
        "archive_sha256": remote._sha256_file(archive),
        "files": records,
    }))


def _completed_sparse_cache_job(
        root: Path, job_id: str, *, source_files: dict[str, bytes],
        source_mtime_ns: int, completed_ns: int,
        target: str = "check_route_contract",
        cache_files: dict[str, bytes] | None = None) -> tuple[Path, dict]:
    """Successful focused job containing only sources and private Make state."""
    job = root / "jobs" / job_id
    work = job / "work"
    source_hash = _write_source_tree(
        work, source_files, mtime_ns=source_mtime_ns)
    for relative, data in sorted((cache_files or {}).items()):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.utime(path, ns=(source_mtime_ns, source_mtime_ns))
    metadata = {
        "protocol_version": remote.PROTOCOL_VERSION,
        "job_id": job_id,
        "remote_root": str(root),
        "source_sha256": source_hash,
        "environment_sha256": "e" * 64,
        "environment_attestation_sha256": "a" * 64,
        "targets": [target],
        "include_candidate_outputs": False,
    }
    _write(job / "job.json", json.dumps(metadata))
    _write(job / "status.json", json.dumps({
        "protocol_version": remote.PROTOCOL_VERSION,
        "state": "succeeded", "exit_code": 0,
    }))
    _write(job / "exit_code", "0\n")
    _rewrite_job_artifacts(job, metadata, completed_ns=completed_ns)
    return job, metadata


def _cached_public_output_fixture(root: Path) -> tuple[Path, dict[str, Path]]:
    """One unchanged cache seed containing every non-root public output."""
    job = root / "job"
    work = job / "work"
    _write_source_tree(
        work, {"top_baffle_v2/Makefile": b"all:\n\t@:\n"},
        mtime_ns=time.time_ns() - 2_000_000_000)
    outputs: dict[str, Path] = {}

    def output(relative: str, text: str = "artifact\n") -> Path:
        path = work / relative
        _write(path, text)
        outputs[relative] = path
        return path

    floor_artifacts = {
        "stl/obiwan-print.stl": {"sha256": "fixture"},
        "baffle_cable_routing_obiwan.png": {"sha256": "fixture"},
    }
    output(
        "top_baffle_v2/floor_stand/obiwan_release_manifest.json",
        json.dumps({"artifacts": floor_artifacts}))
    for relative in floor_artifacts:
        output(f"top_baffle_v2/floor_stand/{relative}")
    output("top_baffle_v2/floor_stand/stl/unrelated-legacy.stl")
    output("top_baffle_v2/no_floor_stand/stl/no-floor.stl")
    output("top_baffle_v2/wings/ac/stl/wing.stl")
    output(remote.COMMON_ARTIFACT)
    output(remote.OBIWAN_WING_DESIGN_MAP_ARTIFACT)
    output(remote.CAPTIVE_MAGNET_CATALOG_ARTIFACT, "{}\n")

    _write(job / "cache-seed.json", json.dumps({
        "format_version": remote.BUILD_CACHE_VERSION,
        "files": remote._tree_records(work),
    }))
    return job, outputs


def _logical_make_lines(text: str) -> tuple[str, ...]:
    lines = []
    current = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        current = f"{current} {stripped}".strip() if current else stripped
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        lines.append(current)
        current = ""
    if current:
        lines.append(current)
    return tuple(lines)


def _make_variable_words(name: str) -> tuple[str, ...]:
    makefile = Path(__file__).with_name("Makefile").read_text(encoding="utf-8")
    prefixes = (f"{name} := ", f"{name} = ")
    line = next(
        line for line in _logical_make_lines(makefile)
        if any(line.startswith(prefix) for prefix in prefixes))
    prefix = next(prefix for prefix in prefixes if line.startswith(prefix))
    return tuple(line.removeprefix(prefix).split())


def _ast_function_registry(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in node.targets)
                and isinstance(node.value, ast.List)
                and all(isinstance(item, ast.Name) for item in node.value.elts)):
            matches.append(tuple(item.id for item in node.value.elts))
    assert len(matches) == 1, (path, name, matches)
    return matches[0]


def _transaction_fixture(root: Path):
    repo = root / "lx"
    baffle = repo / "top_baffle_v2"
    incoming = root / "incoming"
    for state in remote.STATE_OUTPUT_ROOTS:
        _write(baffle / state / "marker", f"old-{state}")
        _write(incoming / "top_baffle_v2" / state / "marker", f"new-{state}")
    _write(baffle / "wings" / "marker", "old-wings")
    _write(incoming / "top_baffle_v2" / "wings" / "marker", "new-wings")
    _write(baffle / "top_baffle_nd25fw4_attachments.step", "old-common")
    _write(
        incoming / "top_baffle_v2" / "top_baffle_nd25fw4_attachments.step",
        "new-common")
    _write(baffle / "review" / "keep.png", "keep-review")
    return repo, baffle, incoming


def _with_fake_project(root: Path):
    repo, baffle, incoming = _transaction_fixture(root)
    originals = (
        remote.REPO_ROOT, remote.BAFFLE_DIR, remote.LOCAL_STATE,
        remote._current_source_identity, remote._check_promoted_roots,
    )
    remote.REPO_ROOT = repo
    remote.BAFFLE_DIR = baffle
    remote.LOCAL_STATE = baffle / ".remote-cad"
    remote._current_source_identity = lambda _metadata: "source"
    remote._check_promoted_roots = lambda _roots: None
    return repo, baffle, incoming, originals


def _restore_project(originals) -> None:
    (remote.REPO_ROOT, remote.BAFFLE_DIR, remote.LOCAL_STATE,
     remote._current_source_identity,
     remote._check_promoted_roots) = originals


def test_target_contract() -> None:
    assert remote._validate_targets([]) == ["all"]
    assert "check_floor_integrated_mount" in remote.REMOTE_MAKE_TARGETS
    assert "floor_obiwan" in remote.REMOTE_MAKE_TARGETS
    assert "no_floor_obiwan" in remote.REMOTE_MAKE_TARGETS
    assert "obiwan_release" in remote.REMOTE_MAKE_TARGETS
    assert "obiwan_wings" in remote.REMOTE_MAKE_TARGETS
    assert "obiwan_wing_artifacts" in remote.REMOTE_MAKE_TARGETS
    assert "check_obiwan_wings" in remote.REMOTE_MAKE_TARGETS
    assert "check_captive_magnets" in remote.REMOTE_MAKE_TARGETS
    assert "check_obiwan_lm_profile" in remote.REMOTE_MAKE_TARGETS
    assert "check_obiwan_junction_closure_plans" in (
        remote.REMOTE_MAKE_TARGETS)
    assert "check_obiwan_junction_closures" in remote.REMOTE_MAKE_TARGETS
    assert "check_obiwan_service" in remote.REMOTE_MAKE_TARGETS
    assert "check_obiwan_closure_focus" in remote.REMOTE_MAKE_TARGETS
    assert "check_obiwan_lm_split_two_pin_static" in (
        remote.REMOTE_MAKE_TARGETS)
    assert "check_floor_support" not in remote.REMOTE_MAKE_TARGETS
    assert remote._full_output_roots(["obiwan_wings"]) == {"wings"}
    assert remote._full_output_roots(["check_obiwan_wings"]) == {
        "wings"}
    assert remote._full_output_roots(["obiwan_release"]) == {
        "floor_stand", "no_floor_stand", "wings"}
    assert remote._full_output_roots(["no_floor_obiwan"]) == set()
    assert remote._full_output_roots(["all"]) == {
        "floor_stand", "no_floor_stand", "wings"}
    for target in remote.REMOTE_MAKE_TARGETS:
        assert remote._validate_targets([target]) == [target]
    for targets in (["all", "clean"], ["clean", "clean"],
                    ["manifold", "manifold"]):
        try:
            remote._validate_targets(targets)
        except ValueError:
            pass
        else:
            raise AssertionError(f"multiple targets accepted: {targets!r}")
    for target in (
            "PYTHON=/tmp/python", "LX_CAD_EXECUTION=local", "-j8",
            "../all", "floor_stand/private"):
        try:
            remote._validate_targets([target])
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe target accepted: {target}")


def test_default_remote_parallelism() -> None:
    previous = os.environ.pop("LX_CAD_REMOTE_JOBS", None)
    try:
        assert remote.DEFAULT_REMOTE_JOBS == 16
        assert remote._remote_job_count(None) == 16
        assert ((remote.REMOTE_MEMORY_MAX_MIB
                 - remote.REMOTE_MEMORY_FLOOR_MIB) // 16) == 28 * 1024
    finally:
        if previous is not None:
            os.environ["LX_CAD_REMOTE_JOBS"] = previous


def test_remote_make_and_guard_share_parallelism_authority() -> None:
    source = Path(remote.__file__).read_text(encoding="utf-8")
    assert '"LX_CAD_GUARD_SLOTS": str(metadata["parallel_jobs"])' in source
    assert 'str(metadata["parallel_jobs"]),' in source
    command_block = source[source.index("command = [\n", source.index(
        '"LX_CAD_GUARD_SLOTS"')):]
    assert 'make, "--no-print-directory", "-j",' in command_block


def test_remote_worker_exports_source_snapshot_identity() -> None:
    """Generated release metadata must name the immutable input snapshot."""
    source = Path(remote.__file__).read_text(encoding="utf-8")
    assert '"LX_CAD_SOURCE_SHA256": metadata["source_sha256"]' in source


def test_bambu_status_json_is_not_a_source_input() -> None:
    """A slicer run must not invalidate an otherwise identical CAD snapshot."""
    with tempfile.TemporaryDirectory() as text:
        repo = Path(text)
        baffle = repo / "top_baffle_v2"
        _write(baffle / "result.json", '{"return_code": 0}\n')
        _write(baffle / "captive_magnet_slicing_profile.json", "{}\n")
        original = (remote.REPO_ROOT, remote.BAFFLE_DIR,
                    remote.REFERENCE_INPUTS)
        remote.REPO_ROOT = repo
        remote.BAFFLE_DIR = baffle
        remote.REFERENCE_INPUTS = ()
        try:
            relative = {
                path.relative_to(baffle).as_posix()
                for path in remote._source_paths()
            }
            assert "result.json" not in relative
            assert "captive_magnet_slicing_profile.json" in relative
        finally:
            (remote.REPO_ROOT, remote.BAFFLE_DIR,
             remote.REFERENCE_INPUTS) = original


def test_protocol_rejection() -> None:
    with tempfile.TemporaryDirectory() as text:
        state = Path(text)
        original = remote.LOCAL_STATE
        remote.LOCAL_STATE = state
        try:
            job = state / "jobs" / "old-job"
            _write(job / "job.json", json.dumps({
                "protocol_version": remote.PROTOCOL_VERSION - 1,
                "job_id": "old-job",
            }))
            try:
                remote._load_local_job("old-job")
            except RuntimeError:
                pass
            else:
                raise AssertionError("old job protocol was accepted")
        finally:
            remote.LOCAL_STATE = original


def test_dead_job_reconciliation_record() -> None:
    with tempfile.TemporaryDirectory() as text:
        job = Path(text)
        _write(job / "status.json", json.dumps({
            "protocol_version": remote.PROTOCOL_VERSION,
            "state": "running",
        }))
        remote._mark_failed(SimpleNamespace(
            job_dir=str(job), exit_code=99,
            reason="synthetic inactive cgroup",
        ))
        status = json.loads((job / "status.json").read_text())
        assert status["state"] == "failed"
        assert status["exit_code"] == 99
        assert status["failure_reason"] == "synthetic inactive cgroup"
        assert (job / "exit_code").read_text() == "99\n"


def test_launch_uses_snapshot_transition() -> None:
    digest = "a" * 64
    metadata = {
        "remote_root": "/srv/lx-cad",
        "source_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
        "job_id": "launch-test",
        "systemd_unit": "lx-cad-launch-test",
        "transport": {
            "source_path": remote.TRANSPORT_SOURCE_PATH,
            "sha256": digest,
            "size": 123,
        },
    }

    class FakeRemote:
        def __init__(self):
            self.commands = []

        def command(self, command, **_kwargs):
            self.commands.append(command)
            if "sha256sum" in command:
                return subprocess.CompletedProcess(
                    command, 0, f"{digest}  executor\n", "")
            if " _launch-job " in command:
                return subprocess.CompletedProcess(command, 0, "", "")
            raise AssertionError(f"unexpected fake remote command: {command}")

    fake = FakeRemote()
    remote._launch_remote_job(fake, metadata)
    launch = next(command for command in fake.commands if " _launch-job " in command)
    assert remote._job_executor_path(metadata) in launch
    assert "/srv/lx-cad/jobs/launch-test" in launch


def test_launch_cancel_transition_is_serial_and_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        job = root / "transition-test"
        job.mkdir()
        metadata = {
            "protocol_version": remote.PROTOCOL_VERSION,
            "remote_root": str(root),
            "source_sha256": "b" * 64,
            "environment_sha256": "c" * 64,
            "job_id": job.name,
            "systemd_unit": "lx-cad-transition-test",
            "transport": {
                "source_path": remote.TRANSPORT_SOURCE_PATH,
                "sha256": "a" * 64,
                "size": 123,
            },
        }
        _write(job / "job.json", json.dumps(metadata))
        _write(job / "status.json", json.dumps({
            "protocol_version": remote.PROTOCOL_VERSION,
            "state": "queued",
        }))
        originals = (
            remote._validated_remote_job,
            remote._local_systemd_unit_facts,
            remote.subprocess.run,
        )
        launches = []
        try:
            remote._validated_remote_job = lambda _job: (metadata, {})
            remote._local_systemd_unit_facts = lambda _unit: {
                "load_state": "not-found", "active_state": "inactive",
                "sub_state": "dead", "result": "success",
            }

            def fake_run(command, **_kwargs):
                launches.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            remote.subprocess.run = fake_run
            remote._launch_job_transition(SimpleNamespace(job_dir=str(job)))
            assert len(launches) == 1
            launch = launches.pop()
            assert launch[0] == "systemd-run"
            assert "--property=MemoryMax=512G" in launch
            assert "--property=MemorySwapMax=0" in launch

            # Reset to the pre-launch state, let cancellation acquire the
            # transition first, then prove a later launch is a no-op.
            (job / "launch.accepted").unlink(missing_ok=True)
            _write(job / "status.json", json.dumps({
                "protocol_version": remote.PROTOCOL_VERSION,
                "state": "queued",
            }))
            remote._cancel_job_transition(SimpleNamespace(job_dir=str(job)))
            assert json.loads((job / "status.json").read_text())["state"] == (
                "canceled")
            remote._launch_job_transition(SimpleNamespace(job_dir=str(job)))
            assert launches == []
        finally:
            (remote._validated_remote_job,
             remote._local_systemd_unit_facts,
             remote.subprocess.run) = originals


def test_environment_hash_is_binary_stable() -> None:
    with tempfile.TemporaryDirectory() as text:
        lock = Path(text) / "requirements.lock"
        payload = b"package==1.2.3\n\x00binary-contract\n"
        lock.write_bytes(payload)
        original = remote.LOCK_FILE
        remote.LOCK_FILE = lock
        try:
            header = (
                f"protocol={remote.PROTOCOL_VERSION}\n"
                f"attestation={remote.ENVIRONMENT_ATTESTATION_VERSION}\n"
                f"python={remote.REMOTE_PYTHON_VERSION}\n"
                "platform=linux-x86_64\n"
            ).encode("utf-8")
            assert remote._environment_hash() == hashlib.sha256(
                header + payload).hexdigest()
        finally:
            remote.LOCK_FILE = original


def test_transport_is_bound_to_source_snapshot() -> None:
    data = b"#!/usr/bin/env python3\nprint('sealed transport')\n"
    digest = hashlib.sha256(data).hexdigest()
    requirements = b"build123d==sealed\n"
    manifest = {
        "protocol_version": remote.PROTOCOL_VERSION,
        "source_sha256": "source-identity",
        "files": [
            {
                "path": remote.TRANSPORT_SOURCE_PATH,
                "size": len(data),
                "sha256": digest,
            },
            {
                "path": remote.REQUIREMENTS_SOURCE_PATH,
                "size": len(requirements),
                "sha256": hashlib.sha256(requirements).hexdigest(),
            },
        ],
    }
    metadata = {
        "source_sha256": "source-identity",
        "transport": {
            "source_path": remote.TRANSPORT_SOURCE_PATH,
            "size": len(data),
            "sha256": digest,
        },
    }
    with tempfile.TemporaryDirectory() as text:
        directory = Path(text)
        archive = directory / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            member = tarfile.TarInfo(remote.TRANSPORT_SOURCE_PATH)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
            member = tarfile.TarInfo(remote.REQUIREMENTS_SOURCE_PATH)
            member.size = len(requirements)
            bundle.addfile(member, io.BytesIO(requirements))
        tool, binding = remote._extract_snapshot_transport(
            archive, directory, manifest)
        assert tool.read_bytes() == data
        assert remote._extract_snapshot_requirements(
            archive, manifest) == requirements
        assert remote._environment_hash(requirements) != remote._environment_hash(
            requirements + b"# drift\n")
        assert binding == metadata["transport"]
        assert remote._validate_transport_provenance(
            metadata, manifest, executing_tool=tool) == binding

        drifted = dict(metadata)
        drifted["transport"] = {**binding, "sha256": "0" * 64}
        try:
            remote._validate_transport_provenance(
                drifted, manifest, executing_tool=tool)
        except RuntimeError:
            pass
        else:
            raise AssertionError("drifted transport provenance was accepted")


def test_atomic_promotion_and_rollback() -> None:
    metadata = {
        "job_id": "transaction-test", "targets": ["candidate"],
        "source_sha256": "source", "include_candidate_outputs": False,
    }
    with tempfile.TemporaryDirectory() as text:
        _repo, baffle, incoming, originals = _with_fake_project(Path(text))
        try:
            promoted = remote._promote_artifacts(incoming, metadata)
            assert promoted == 4
            for state in remote.STATE_OUTPUT_ROOTS:
                assert (baffle / state / "marker").read_text() == f"new-{state}"
            assert (baffle / "wings" / "marker").read_text() == "new-wings"
            assert (baffle / "top_baffle_nd25fw4_attachments.step").read_text() == (
                "new-common")
            assert (baffle / "review" / "keep.png").read_text() == "keep-review"
        finally:
            _restore_project(originals)

    with tempfile.TemporaryDirectory() as text:
        _repo, baffle, incoming, originals = _with_fake_project(Path(text))
        try:
            remote._check_promoted_roots = lambda _roots: (_ for _ in ()).throw(
                RuntimeError("synthetic QA failure"))
            try:
                remote._promote_artifacts(incoming, metadata)
            except RuntimeError as exc:
                assert "synthetic QA" in str(exc)
            else:
                raise AssertionError("synthetic QA failure did not roll back")
            for state in remote.STATE_OUTPUT_ROOTS:
                assert (baffle / state / "marker").read_text() == f"old-{state}"
            assert (baffle / "wings" / "marker").read_text() == "old-wings"
            assert (baffle / "top_baffle_nd25fw4_attachments.step").read_text() == (
                "old-common")
        finally:
            _restore_project(originals)


def test_persistent_promotion_recovery() -> None:
    metadata = {
        "job_id": "recovery-test", "targets": ["candidate"],
        "source_sha256": "source", "include_candidate_outputs": False,
    }
    with tempfile.TemporaryDirectory() as text:
        _repo, baffle, _incoming, originals = _with_fake_project(Path(text))
        try:
            destination = baffle / "floor_stand"
            relative = destination.relative_to(remote.REPO_ROOT)
            backup_root = (
                remote.LOCAL_STATE / "backups" / metadata["job_id"])
            remote._begin_promotion_transaction(
                backup_root, metadata, [(destination, relative, None)])
            _write(destination / "marker", "interrupted-new")

            remote._recover_promotion_transaction(backup_root, metadata)
            assert destination.joinpath("marker").read_text() == (
                "old-floor_stand")
            assert not backup_root.exists()
            assert (baffle / "review" / "keep.png").read_text() == (
                "keep-review")
        finally:
            _restore_project(originals)


def test_foreign_promotion_recovered_before_new_job() -> None:
    metadata = {
        "protocol_version": remote.PROTOCOL_VERSION,
        "job_id": "orphan-job", "targets": ["candidate"],
        "source_sha256": "source", "include_candidate_outputs": False,
    }
    with tempfile.TemporaryDirectory() as text:
        _repo, baffle, _incoming, originals = _with_fake_project(Path(text))
        try:
            job_dir = remote.LOCAL_STATE / "jobs" / metadata["job_id"]
            job_dir.mkdir(parents=True)
            _write(job_dir / "job.json", json.dumps(metadata))
            destination = baffle / "floor_stand"
            relative = destination.relative_to(remote.REPO_ROOT)
            backup_root = (
                remote.LOCAL_STATE / "backups" / metadata["job_id"])
            remote._begin_promotion_transaction(
                backup_root, metadata, [(destination, relative, None)])
            _write(destination / "marker", "partial-orphan-generation")

            remote._recover_outstanding_promotion_transaction()
            assert destination.joinpath("marker").read_text() == (
                "old-floor_stand")
            assert not backup_root.exists()
        finally:
            _restore_project(originals)


def test_fetch_requires_succeeded_status() -> None:
    with tempfile.TemporaryDirectory() as text:
        state = Path(text) / ".remote-cad"
        original = remote.LOCAL_STATE
        remote.LOCAL_STATE = state
        metadata = {"job_id": "fetch-canceled", "remote_root": "/remote"}

        class FakeRemote:
            def download(self, _source, destination, *, required):
                name = destination.name.removesuffix(".partial")
                if name == "status.json":
                    _write(destination, json.dumps({
                        "protocol_version": remote.PROTOCOL_VERSION,
                        "state": "canceled", "exit_code": 0,
                    }))
                elif name == "exit_code":
                    _write(destination, "0\n")
                elif name == "build.log":
                    _write(destination, "synthetic canceled fetch\n")
                elif required:
                    raise AssertionError(f"unexpected required file: {name}")
                return True

        try:
            result = remote._fetch(FakeRemote(), metadata)
            assert result == 99
            assert not (state / "jobs" / metadata["job_id"]
                        / "artifacts.json").exists()
        finally:
            remote.LOCAL_STATE = original


def test_unexplained_backup_is_preserved() -> None:
    metadata = {
        "job_id": "stale-test", "targets": ["candidate"],
        "source_sha256": "source", "include_candidate_outputs": False,
    }
    with tempfile.TemporaryDirectory() as text:
        _repo, _baffle, incoming, originals = _with_fake_project(Path(text))
        try:
            backup_root = (
                remote.LOCAL_STATE / "backups" / metadata["job_id"])
            _write(backup_root / "mystery", "do-not-delete")
            try:
                remote._promote_artifacts(incoming, metadata)
            except RuntimeError as exc:
                assert "unexplained stale promotion backup" in str(exc)
            else:
                raise AssertionError("unexplained stale backup was accepted")
            assert (backup_root / "mystery").read_text() == "do-not-delete"
        finally:
            _restore_project(originals)


def test_promotion_signals_roll_back() -> None:
    for signum in (signal.SIGINT, signal.SIGTERM):
        with tempfile.TemporaryDirectory() as text:
            _repo, baffle, incoming, originals = _with_fake_project(Path(text))
            previous_handler = signal.getsignal(signum)
            try:
                metadata = {
                    "protocol_version": remote.PROTOCOL_VERSION,
                    "job_id": f"signal-{signum}", "targets": ["candidate"],
                    "source_sha256": "source",
                    "include_candidate_outputs": False,
                }
                job_dir = remote.LOCAL_STATE / "jobs" / metadata["job_id"]
                job_dir.mkdir(parents=True)
                job_incoming = job_dir / "incoming"
                incoming.replace(job_incoming)
                _write(job_dir / "job.json", json.dumps(metadata))
                remote._check_promoted_roots = lambda _roots: os.kill(
                    os.getpid(), signum)
                args = SimpleNamespace(
                    job_dir=str(job_dir), incoming=str(job_incoming),
                    result=str(job_dir / "result.json"),
                )
                try:
                    remote._promote_local(args)
                except remote._PromotionInterrupted as exc:
                    assert signal.Signals(signum).name in str(exc)
                else:
                    raise AssertionError(f"signal {signum} did not interrupt")
                for state in remote.STATE_OUTPUT_ROOTS:
                    assert baffle.joinpath(state, "marker").read_text() == (
                        f"old-{state}")
                assert baffle.joinpath("wings", "marker").read_text() == (
                    "old-wings")
                assert baffle.joinpath(
                    "top_baffle_nd25fw4_attachments.step").read_text() == (
                        "old-common")
                assert (baffle / "review" / "keep.png").read_text() == (
                    "keep-review")
                assert signal.getsignal(signum) == previous_handler
            finally:
                _restore_project(originals)


def test_remote_clean_preserves_review() -> None:
    with tempfile.TemporaryDirectory() as text:
        _repo, baffle, incoming, originals = _with_fake_project(Path(text))
        try:
            metadata = {
                "job_id": "clean-test", "targets": ["clean"],
                "source_sha256": "source", "include_candidate_outputs": False,
            }
            remote._promote_artifacts(incoming, metadata)
            assert not (baffle / "floor_stand").exists()
            assert not (baffle / "no_floor_stand").exists()
            assert not (baffle / "wings").exists()
            assert not (baffle / "top_baffle_nd25fw4_attachments.step").exists()
            assert (baffle / "review" / "keep.png").read_text() == "keep-review"
        finally:
            _restore_project(originals)

    with tempfile.TemporaryDirectory() as text:
        _repo, baffle, incoming, originals = _with_fake_project(Path(text))
        try:
            metadata = {
                "job_id": "clean-drift-test", "targets": ["clean"],
                "source_sha256": "source", "include_candidate_outputs": False,
            }
            identities = iter(("source", "changed"))
            remote._current_source_identity = lambda _metadata: next(identities)
            try:
                remote._promote_artifacts(incoming, metadata)
            except RuntimeError as exc:
                assert "changed during promotion" in str(exc)
            else:
                raise AssertionError("clean promotion ignored source drift")
            for state in remote.STATE_OUTPUT_ROOTS:
                assert baffle.joinpath(state, "marker").read_text() == (
                    f"old-{state}")
            assert baffle.joinpath("wings", "marker").read_text() == (
                "old-wings")
            assert baffle.joinpath(
                "top_baffle_nd25fw4_attachments.step").read_text() == (
                    "old-common")
            assert (baffle / "review" / "keep.png").read_text() == (
                "keep-review")
        finally:
            _restore_project(originals)


def test_verified_make_cache_seed_and_checksum_overlay() -> None:
    """A fresh job reuses outputs while exact source bytes drive Make mtimes."""
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        old_mtime = time.time_ns() - 20_000_000_000
        output_mtime = old_mtime + 1_000_000_000
        makefile = (
            b"floor_stand/stl/part.stl: model.py\n"
            b"\tcp model.py $@\n")
        source_files = {
            "top_baffle_v2/Makefile": makefile,
            "top_baffle_v2/model.py": b"old model\n",
            "top_baffle_v2/removed.py": b"removed input\n",
        }
        old_job, old_metadata = _completed_cache_job(
            root, "cache-old", source_files=source_files,
            source_mtime_ns=old_mtime, output_data=b"old mesh\n",
            output_mtime_ns=output_mtime,
            completed_ns=old_mtime + 2_000_000_000,
        )
        assert remote._publish_build_cache(old_job) is True
        environment_hash = old_metadata["environment_sha256"]
        attestation_hash = old_metadata["environment_attestation_sha256"]
        entry = remote._build_cache_entry(
            root, environment_hash, attestation_hash)
        remote._verify_build_cache_entry(
            entry, environment_hash, attestation_hash)
        assert subprocess.run(
            ["make", "-q", "floor_stand/stl/part.stl"],
            cwd=entry / "work" / "top_baffle_v2").returncode == 0

        current = root / "jobs" / "cache-current"
        current_work = current / "work"
        current_files = {
            "top_baffle_v2/Makefile": makefile,
            "top_baffle_v2/model.py": b"new model\n",
            "top_baffle_v2/added.py": b"new input\n",
            "top_baffle_v2/removed.py": b"removed input\n",
        }
        current_hash = _write_source_tree(
            current_work, current_files,
            mtime_ns=old_mtime + 5_000_000_000)
        current_metadata = {
            **old_metadata,
            "job_id": "cache-current",
            "source_sha256": current_hash,
            "targets": ["check_obiwan_junction_closures"],
        }
        seed = remote._seed_build_cache(current, current_metadata)
        assert seed is not None
        assert seed["source_sha256"] == old_metadata["source_sha256"]
        assert set(seed["changed_source_paths"]) == {
            "top_baffle_v2/added.py", "top_baffle_v2/model.py"}
        remote._verify_source(current_work, current_hash, allow_extra=True)

        cached_make = current_work / "top_baffle_v2" / "Makefile"
        changed_model = current_work / "top_baffle_v2" / "model.py"
        output = (
            current_work / "top_baffle_v2" / "floor_stand" / "stl"
            / "part.stl")
        assert cached_make.stat().st_mtime_ns == old_mtime
        assert changed_model.stat().st_mtime_ns > output.stat().st_mtime_ns
        assert (current_work / "top_baffle_v2" / "removed.py").is_file()
        assert output.read_bytes() == b"old mesh\n"
        make_query = subprocess.run(
            ["make", "-q", "floor_stand/stl/part.stl"],
            cwd=current_work / "top_baffle_v2")
        assert make_query.returncode == 1

        # A focused check must not re-bundle an unchanged inherited artifact.
        assert remote._artifact_paths(current, current_metadata) == []
        output.write_bytes(b"rebuilt mesh\n")
        assert remote._artifact_paths(current, current_metadata) == [output]

        # A complete-root target always returns the whole declared root, even
        # when a generated byte happens to match the inherited seed.
        output.write_bytes(b"old mesh\n")
        os.utime(output, ns=(output_mtime, output_mtime))
        assert remote._artifact_paths(
            current, {**current_metadata, "targets": ["floor_stand"]}) == [
                output]

        # A focused delta cannot silently lose an inherited generated file:
        # individual-file promotion has no deletion tombstone.
        output.unlink()
        try:
            remote._artifact_paths(current, current_metadata)
        except RuntimeError as exc:
            assert "representable promotion deletion" in str(exc)
        else:
            raise AssertionError("focused artifact deletion was ignored")


def test_warm_cache_always_bundles_public_target_outputs() -> None:
    """A Make no-op must still deliver the result promised by its target."""
    with tempfile.TemporaryDirectory() as text:
        job, outputs = _cached_public_output_fixture(Path(text))

        def selected(target: str) -> set[str]:
            return {
                path.relative_to(job / "work").as_posix()
                for path in remote._artifact_paths(
                    job, {"targets": [target]})
            }

        floor_required = {
            "top_baffle_v2/floor_stand/obiwan_release_manifest.json",
            "top_baffle_v2/floor_stand/stl/obiwan-print.stl",
            "top_baffle_v2/floor_stand/baffle_cable_routing_obiwan.png",
        }
        assert selected("floor_obiwan") == floor_required
        assert "top_baffle_v2/floor_stand/stl/unrelated-legacy.stl" not in (
            selected("floor_obiwan"))
        assert selected("common") == {remote.COMMON_ARTIFACT}

        complete_roots = {
            relative for relative in outputs
            if relative.startswith((
                "top_baffle_v2/floor_stand/",
                "top_baffle_v2/no_floor_stand/",
                "top_baffle_v2/wings/",
            ))
        }
        expected_all = complete_roots | {
            remote.COMMON_ARTIFACT,
            remote.OBIWAN_WING_DESIGN_MAP_ARTIFACT,
            remote.CAPTIVE_MAGNET_CATALOG_ARTIFACT,
        }
        assert selected("all") == expected_all
        assert selected("candidate") == expected_all


def test_warm_cache_outputs_promote_into_fresh_local_trees() -> None:
    """Selected cached outputs survive extraction/promotion when local is empty."""
    for target in ("all", "floor_obiwan", "common"):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text)
            job, _outputs = _cached_public_output_fixture(root / "remote")
            selected = remote._artifact_paths(job, {"targets": [target]})
            incoming = root / "incoming"
            for source in selected:
                relative = source.relative_to(job / "work")
                destination = incoming / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())

            repo = root / "local" / "lx"
            baffle = repo / "top_baffle_v2"
            baffle.mkdir(parents=True)
            originals = (
                remote.REPO_ROOT, remote.BAFFLE_DIR, remote.LOCAL_STATE,
                remote._current_source_identity,
                remote._check_promoted_roots,
            )
            remote.REPO_ROOT = repo
            remote.BAFFLE_DIR = baffle
            remote.LOCAL_STATE = baffle / ".remote-cad"
            remote._current_source_identity = lambda _metadata: "source"
            remote._check_promoted_roots = lambda _roots: None
            metadata = {
                "protocol_version": remote.PROTOCOL_VERSION,
                "job_id": f"fresh-{target.replace('_', '-')}",
                "targets": [target],
                "source_sha256": "source",
                "include_candidate_outputs": False,
            }
            try:
                remote._promote_artifacts(incoming, metadata)
                if target == "all":
                    for state in remote.STATE_OUTPUT_ROOTS:
                        assert (baffle / state).is_dir()
                    assert (baffle / "wings").is_dir()
                    for relative in (
                            remote.COMMON_ARTIFACT,
                            remote.OBIWAN_WING_DESIGN_MAP_ARTIFACT,
                            remote.CAPTIVE_MAGNET_CATALOG_ARTIFACT):
                        assert (repo / relative).is_file()
                elif target == "floor_obiwan":
                    assert (baffle / "floor_stand"
                            / "obiwan_release_manifest.json").is_file()
                    assert (baffle / "floor_stand" / "stl"
                            / "obiwan-print.stl").is_file()
                    assert not (baffle / "floor_stand" / "stl"
                                / "unrelated-legacy.stl").exists()
                elif target == "common":
                    assert (repo / remote.COMMON_ARTIFACT).is_file()
            finally:
                _restore_project(originals)


def test_source_deletion_forces_true_cold_cache_fallback() -> None:
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        base = time.time_ns() - 20_000_000_000
        old_sources = {
            "top_baffle_v2/Makefile": b"graph\n",
            "top_baffle_v2/obsolete_generator.py": b"old generator\n",
        }
        old_job, metadata = _completed_cache_job(
            root, "cache-source-delete-old", source_files=old_sources,
            source_mtime_ns=base, output_data=b"obsolete output\n",
            output_mtime_ns=base + 1_000_000_000,
            completed_ns=base + 2_000_000_000,
        )
        assert remote._publish_build_cache(old_job) is True
        entry = remote._build_cache_entry(
            root, metadata["environment_sha256"],
            metadata["environment_attestation_sha256"])

        current_job = root / "jobs" / "cache-source-delete-current"
        current_sources = {"top_baffle_v2/Makefile": b"graph\n"}
        current_hash = _write_source_tree(
            current_job / "work", current_sources,
            mtime_ns=base + 3_000_000_000)
        current_metadata = {
            **metadata,
            "job_id": "cache-source-delete-current",
            "source_sha256": current_hash,
            "targets": ["check_obiwan_junction_closures"],
        }
        assert remote._seed_build_cache(
            current_job, current_metadata) is None
        assert entry.is_dir()  # valid for its original source snapshot
        assert not (current_job / "cache-seed.json").exists()
        assert not (
            current_job / "work" / "top_baffle_v2" / "floor_stand"
        ).exists()
        remote._verify_source(
            current_job / "work", current_hash, allow_extra=False)


def test_exact_source_focused_publication_unions_cache_coverage() -> None:
    """A later narrow job cannot evict a richer exact-source Make seed."""
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        base = time.time_ns() - 20_000_000_000
        sources = {"top_baffle_v2/Makefile": b"graph\n"}
        rich_job, metadata = _completed_cache_job(
            root, "cache-rich", source_files=sources,
            source_mtime_ns=base, output_data=b"rich mesh\n",
            output_mtime_ns=base + 1_000_000_000,
            completed_ns=base + 2_000_000_000,
        )
        shared_stamp = (
            rich_job / "work" / "top_baffle_v2" / ".check-stamps"
            / "shared.ok")
        _write(shared_stamp, "older check\n")
        os.chmod(shared_stamp, 0o640)
        os.utime(shared_stamp, ns=(base, base))
        assert remote._publish_build_cache(rich_job) is True

        focused_job, focused_metadata = _completed_sparse_cache_job(
            root, "cache-focused", source_files=sources,
            source_mtime_ns=base,
            completed_ns=base + 4_000_000_000,
            cache_files={
                "top_baffle_v2/.check-stamps/shared.ok": b"newer check\n",
                "top_baffle_v2/.check-stamps/route.ok": b"route passed\n",
            },
        )
        focused_shared = (
            focused_job / "work" / "top_baffle_v2" / ".check-stamps"
            / "shared.ok")
        os.chmod(focused_shared, 0o600)
        focused_shared_mtime = focused_shared.stat().st_mtime_ns
        assert remote._publish_build_cache(focused_job) is True

        entry = remote._build_cache_entry(
            root, metadata["environment_sha256"],
            metadata["environment_attestation_sha256"])
        marker, records = remote._verify_build_cache_entry(
            entry, metadata["environment_sha256"],
            metadata["environment_attestation_sha256"])
        cached_output = (
            entry / "work" / "top_baffle_v2" / "floor_stand" / "stl"
            / "part.stl")
        cached_shared = (
            entry / "work" / "top_baffle_v2" / ".check-stamps"
            / "shared.ok")
        cached_route = cached_shared.with_name("route.ok")
        assert cached_output.read_bytes() == b"rich mesh\n"
        assert cached_route.read_bytes() == b"route passed\n"
        assert cached_shared.read_bytes() == b"newer check\n"
        assert (cached_shared.stat().st_mode & 0o777) == 0o600
        assert cached_shared.stat().st_mtime_ns == focused_shared_mtime
        assert marker["published_from_job"] == focused_metadata["job_id"]
        assert marker["coverage_union"] == {
            "exact_source": True, "retained_file_count": 1}
        assert cached_output.relative_to(entry / "work").as_posix() in records

        consumer = root / "jobs" / "cache-union-consumer"
        consumer_hash = _write_source_tree(
            consumer / "work", sources, mtime_ns=base)
        consumer_metadata = {
            **focused_metadata,
            "job_id": "cache-union-consumer",
            "source_sha256": consumer_hash,
        }
        assert remote._seed_build_cache(
            consumer, consumer_metadata) is not None
        assert (consumer / "work" / cached_output.relative_to(
            entry / "work")).read_bytes() == b"rich mesh\n"
        assert (consumer / "work" / cached_route.relative_to(
            entry / "work")).read_bytes() == b"route passed\n"


def test_cache_union_respects_complete_roots_and_clean() -> None:
    """Union retains other coverage but never revives represented deletions."""
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        base = time.time_ns() - 20_000_000_000
        sources = {"top_baffle_v2/Makefile": b"graph\n"}
        old_job, metadata = _completed_cache_job(
            root, "cache-root-old", source_files=sources,
            source_mtime_ns=base, output_data=b"old part\n",
            output_mtime_ns=base + 1_000_000_000,
            completed_ns=base + 2_000_000_000,
        )
        _write(
            old_job / "work" / "top_baffle_v2" / "floor_stand" / "stl"
            / "obsolete.stl", "obsolete\n")
        _write(
            old_job / "work" / "top_baffle_v2" / "no_floor_stand" / "stl"
            / "keep.stl", "other state\n")
        _rewrite_job_artifacts(
            old_job, metadata, completed_ns=base + 2_000_000_000)
        assert remote._publish_build_cache(old_job) is True

        new_job, _new_metadata = _completed_cache_job(
            root, "cache-root-new", source_files=sources,
            source_mtime_ns=base, output_data=b"new part\n",
            output_mtime_ns=base + 3_000_000_000,
            completed_ns=base + 4_000_000_000,
        )
        assert remote._publish_build_cache(new_job) is True
        entry = remote._build_cache_entry(
            root, metadata["environment_sha256"],
            metadata["environment_attestation_sha256"])
        work = entry / "work" / "top_baffle_v2"
        assert (work / "floor_stand" / "stl" / "part.stl").read_bytes() == (
            b"new part\n")
        assert not (
            work / "floor_stand" / "stl" / "obsolete.stl").exists()
        assert (work / "no_floor_stand" / "stl" / "keep.stl").read_bytes() == (
            b"other state\n")

        clean_job, _clean_metadata = _completed_sparse_cache_job(
            root, "cache-root-clean", source_files=sources,
            source_mtime_ns=base,
            completed_ns=base + 6_000_000_000, target="clean")
        assert remote._publish_build_cache(clean_job) is True
        remote._verify_build_cache_entry(
            entry, metadata["environment_sha256"],
            metadata["environment_attestation_sha256"])
        assert not (entry / "work" / "top_baffle_v2" / "floor_stand").exists()
        assert not (
            entry / "work" / "top_baffle_v2" / "no_floor_stand").exists()
        assert remote._publish_build_cache(new_job) is False
        assert not (entry / "work" / "top_baffle_v2" / "floor_stand").exists()


def test_cache_publication_does_not_union_across_source_deletion() -> None:
    """A changed/deleted source snapshot replaces rather than unions coverage."""
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        base = time.time_ns() - 20_000_000_000
        old_sources = {
            "top_baffle_v2/Makefile": b"graph\n",
            "top_baffle_v2/obsolete.py": b"old input\n",
        }
        old_job, metadata = _completed_cache_job(
            root, "cache-delete-rich", source_files=old_sources,
            source_mtime_ns=base, output_data=b"old mesh\n",
            output_mtime_ns=base + 1_000_000_000,
            completed_ns=base + 2_000_000_000,
        )
        assert remote._publish_build_cache(old_job) is True

        new_sources = {"top_baffle_v2/Makefile": b"graph\n"}
        sparse_job, sparse_metadata = _completed_sparse_cache_job(
            root, "cache-delete-sparse", source_files=new_sources,
            source_mtime_ns=base + 3_000_000_000,
            completed_ns=base + 4_000_000_000,
            cache_files={
                "top_baffle_v2/.check-stamps/new-source.ok": b"passed\n"},
        )
        assert remote._publish_build_cache(sparse_job) is True
        entry = remote._build_cache_entry(
            root, metadata["environment_sha256"],
            metadata["environment_attestation_sha256"])
        marker, _records = remote._verify_build_cache_entry(
            entry, metadata["environment_sha256"],
            metadata["environment_attestation_sha256"])
        assert marker["source_sha256"] == sparse_metadata["source_sha256"]
        assert "coverage_union" not in marker
        assert not (
            entry / "work" / "top_baffle_v2" / "obsolete.py").exists()
        assert not (
            entry / "work" / "top_baffle_v2" / "floor_stand").exists()
        assert (entry / "work" / "top_baffle_v2" / ".check-stamps"
                / "new-source.ok").is_file()


def test_make_cache_publication_order_and_damage_fallback() -> None:
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        base = time.time_ns() - 20_000_000_000
        source_files = {"top_baffle_v2/Makefile": b"graph\n"}
        old_job, metadata = _completed_cache_job(
            root, "cache-order-old", source_files=source_files,
            source_mtime_ns=base, output_data=b"old\n",
            output_mtime_ns=base + 1_000_000_000,
            completed_ns=base + 2_000_000_000,
        )
        new_job, _new_metadata = _completed_cache_job(
            root, "cache-order-new", source_files=source_files,
            source_mtime_ns=base, output_data=b"new\n",
            output_mtime_ns=base + 3_000_000_000,
            completed_ns=base + 4_000_000_000,
        )
        assert remote._publish_build_cache(new_job) is True
        assert remote._publish_build_cache(old_job) is False

        # A rebuilt interpreter/environment with the same lock identity gets
        # a separate seed; actual runtime attestation is part of the key.
        alternate_job, alternate_metadata = _completed_cache_job(
            root, "cache-order-alternate", source_files=source_files,
            source_mtime_ns=base, output_data=b"alternate\n",
            output_mtime_ns=base + 5_000_000_000,
            completed_ns=base + 6_000_000_000,
            attestation_hash="b" * 64,
        )
        assert remote._publish_build_cache(alternate_job) is True
        alternate_entry = remote._build_cache_entry(
            root, alternate_metadata["environment_sha256"],
            alternate_metadata["environment_attestation_sha256"])
        assert alternate_entry.is_dir()

        incomplete_job, _incomplete_metadata = _completed_cache_job(
            root, "cache-incomplete-delta", source_files=source_files,
            source_mtime_ns=base, output_data=b"unreported\n",
            output_mtime_ns=base + 7_000_000_000,
            completed_ns=base + 8_000_000_000,
            attestation_hash="c" * 64,
        )
        archive = incomplete_job / "artifacts.tar.gz"
        with tarfile.open(archive, "w:gz"):
            pass
        artifact_manifest = json.loads(
            (incomplete_job / "artifacts.json").read_text())
        artifact_manifest["files"] = []
        artifact_manifest["archive_sha256"] = remote._sha256_file(archive)
        _write(
            incomplete_job / "artifacts.json",
            json.dumps(artifact_manifest))
        try:
            remote._publish_build_cache(incomplete_job)
        except RuntimeError as exc:
            assert "exact generated delta" in str(exc)
        else:
            raise AssertionError("incomplete artifact delta published cache")

        environment_hash = metadata["environment_sha256"]
        entry = remote._build_cache_entry(
            root, environment_hash,
            metadata["environment_attestation_sha256"])
        cached_output = (
            entry / "work" / "top_baffle_v2" / "floor_stand" / "stl"
            / "part.stl")
        assert cached_output.read_bytes() == b"new\n"
        cached_output.write_bytes(b"damaged\n")

        cold_job = root / "jobs" / "cache-damage-consumer"
        source_hash = _write_source_tree(
            cold_job / "work", source_files, mtime_ns=base + 5_000_000_000)
        cold_metadata = {
            **metadata,
            "job_id": "cache-damage-consumer",
            "source_sha256": source_hash,
        }
        assert remote._seed_build_cache(cold_job, cold_metadata) is None
        assert not entry.exists()
        assert not (
            cold_job / "work" / "top_baffle_v2" / "floor_stand").exists()
        remote._verify_source(
            cold_job / "work", source_hash, allow_extra=False)

        # A remote job that did not complete successfully is never eligible.
        _write(new_job / "status.json", json.dumps({
            "protocol_version": remote.PROTOCOL_VERSION,
            "state": "failed", "exit_code": 99,
        }))
        try:
            remote._publish_build_cache(new_job)
        except RuntimeError as exc:
            assert "succeeded" in str(exc)
        else:
            raise AssertionError("failed job published a Make cache")


def test_cache_publication_follows_verified_local_promotion() -> None:
    source = Path(remote.__file__).read_text(encoding="utf-8")
    fetch = source[source.index("def _fetch("):source.index(
        "def _wait_and_fetch(")]
    assert fetch.index("_verify_and_extract_artifacts") < fetch.index(
        "_run_guarded_local_promotion")
    assert fetch.index("_run_guarded_local_promotion") < fetch.index(
        '"_publish-cache"')


def test_parallel_stage_dag() -> None:
    result = subprocess.run(
        ["make", "-n", "-j4", "LX_CAD_EXECUTION=remote-worker",
         f"PYTHON={sys.executable}", "validate_obiwan_stages",
         "floor_stand/.obiwan_stage/manifest.json",
         "no_floor_stand/.obiwan_stage/manifest.json"],
        cwd=Path(__file__).resolve().parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        env={**os.environ, "LX_CAD_GUARD_SLOTS": "4"},
    )
    assert result.stdout.count("export_obiwan_staged.py stage") == 2


def test_obiwan_basic_wing_parallel_dag() -> None:
    result = subprocess.run(
        ["make", "-n", "-B", "-j4", "LX_CAD_EXECUTION=remote-worker",
         f"PYTHON={sys.executable}", "obiwan_wing_artifacts"],
        cwd=Path(__file__).resolve().parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        env={**os.environ, "LX_CAD_GUARD_SLOTS": "4"},
    )
    assert result.stdout.count(
        "export_obiwan_wings.py --slug ac --output-root wings") == 1
    assert result.stdout.count(
        "export_obiwan_wings.py --slug ae --output-root wings") == 1
    assert result.stdout.count(
        "test_obiwan_wings.py --artifact-root wings") == 1


def test_make_parallel_manifold_dag() -> None:
    with tempfile.TemporaryDirectory() as text:
        floor_root = Path(text) / "floor_stand" / "stl"
        no_floor_root = Path(text) / "no_floor_stand" / "stl"
        # Mirror the exact release topology inventory: 45 acoustic meshes in
        # each state, plus the two floor-only polar fixtures.  This proves the
        # Make expansion has one and only one check node for all 92 state STLs.
        for index in range(47):
            _write(floor_root / f"floor_fixture_{index}.stl", "fixture")
        for index in range(45):
            _write(no_floor_root / f"no_floor_fixture_{index}.stl", "fixture")
        stamp_root = Path(text) / "stamps"
        result = subprocess.run(
            ["make", "-n", "-B", "-j8",
             "LX_CAD_EXECUTION=remote-worker", f"PYTHON={sys.executable}",
             "_manifold_parallel",
             f"MANIFOLD_ROOTS={floor_root} {no_floor_root}",
             f"MANIFOLD_STAMP_DIR={stamp_root}"],
            cwd=Path(__file__).resolve().parent, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
            env={**os.environ, "LX_CAD_GUARD_SLOTS": "8"},
        )
        output = result.stdout
        assert output.count("check_manifold.py --stl-only ") == 92
        assert output.count(
            f"check_manifold.py --stl-only {floor_root}/") == 47
        assert output.count(
            f"check_manifold.py --stl-only {no_floor_root}/") == 45
        assert output.count("check_manifold.py --metadata-only") == 1
        assert str(stamp_root) in output


def test_make_obiwan_only_manifold_filters_warm_legacy_meshes() -> None:
    with tempfile.TemporaryDirectory() as text:
        root = Path(text)
        floor_root = root / "floor_stand" / "stl"
        no_floor_root = root / "no_floor_stand" / "stl"
        expected = []
        legacy = []
        for state_root in (floor_root, no_floor_root):
            for index in range(4):
                path = state_root / f"lx521_top_v1l_legacy_{index}.stl"
                _write(path, "cached legacy mesh")
                legacy.append(path)
            for name in (
                    "lx521_top_obiwan_core_1of2_lm_carrier.stl",
                    "lx521_top_obiwan_core_2of2_um_carrier.stl",
                    "lx521_coupon_12_obiwan_closed_bore_bump.stl"):
                path = state_root / name
                _write(path, "focused obiwan mesh")
                expected.append(path)

        result = subprocess.run(
            ["make", "-n", "-B", "-j8",
             "LX_CAD_EXECUTION=remote-worker", f"PYTHON={sys.executable}",
             "_manifold_parallel", "MANIFOLD_OBIWAN_ONLY=1",
             f"MANIFOLD_ROOTS={floor_root} {no_floor_root}",
             f"MANIFOLD_STAMP_DIR={root / 'stamps'}"],
            cwd=Path(__file__).resolve().parent, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
            env={**os.environ, "LX_CAD_GUARD_SLOTS": "8"},
        )
        output = result.stdout
        commands = [
            line for line in output.splitlines()
            if "check_manifold.py --stl-only " in line]
        assert len(commands) == len(expected)
        for path in expected:
            assert sum(str(path) in line for line in commands) == 1
        for path in legacy:
            assert str(path) not in output
        assert output.count("check_manifold.py --metadata-only") == 1
        assert output.count("--obiwan-only") == 1


def test_candidate_is_one_flat_make_dag() -> None:
    makefile = (Path(__file__).resolve().parent / "Makefile").read_text(
        encoding="utf-8")
    assert "candidate: check all\n" in makefile
    assert "candidate: check\n" not in makefile
    assert "ifeq ($(LX_CAD_EXECUTION),local)\n.NOTPARALLEL:\nendif" in makefile
    all_rule = next(
        line for line in makefile.splitlines()
        if line.startswith("all:"))
    assert "floor_stand_artifacts" in all_rule
    assert "no_floor_stand_artifacts" in all_rule
    assert " floor_stand " not in f" {all_rule} "
    assert " no_floor_stand " not in f" {all_rule} "

    # State-local mesh fan-outs are recipes of independent artifact nodes, so
    # they may overlap through the enclosing GNU Make jobserver.  The public
    # wrappers do not repeat them, and the join performs metadata only.
    floor_block = makefile.split(
        "floor_stand_artifacts: $(call VARIANT_TARGETS,floor_stand)", 1,
    )[1].split("\nfloor_stand:", 1)[0]
    no_floor_block = makefile.split(
        "no_floor_stand_artifacts: validate_no_floor_obiwan_stage", 1,
    )[1].split("\nno_floor_stand:", 1)[0]
    assert floor_block.count(
        "MANIFOLD_ROOTS='floor_stand/stl'") == 1
    assert no_floor_block.count(
        "MANIFOLD_ROOTS='no_floor_stand/stl'") == 1

    all_block = makefile.split("all:", 1)[1].split(
        "\n# This catalog", 1)[0]
    assert "_manifold_parallel" not in all_block
    assert "check_manifold.py --metadata-only" in all_block
    assert "floor_stand/stl no_floor_stand/stl" in all_block

    floor_wrapper = makefile.split(
        "floor_stand: floor_stand_artifacts", 1,
    )[1].split("\n# Focused", 1)[0]
    no_floor_wrapper = makefile.split(
        "no_floor_stand: no_floor_stand_artifacts", 1,
    )[1].split("\ncommon:", 1)[0]
    assert "_manifold_parallel" not in floor_wrapper
    assert "_manifold_parallel" not in no_floor_wrapper


def test_make_check_registries_match_python_and_do_not_fan_out() -> None:
    root = Path(__file__).resolve().parent
    clearance = _ast_function_registry(root / "test_clearances.py", "checks")
    r6f = _ast_function_registry(root / "test_obiwan_r6f.py", "CHECKS")
    assert _make_variable_words("CLEARANCE_CHECK_NAMES") == clearance
    assert _make_variable_words("R6F_CHECK_NAMES") == r6f

    result = subprocess.run(
        ["make", "-n", "-B", "-j8", "LX_CAD_EXECUTION=remote-worker",
         f"PYTHON={sys.executable}", "check"],
        cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
        env={**os.environ, "LX_CAD_GUARD_SLOTS": "8"},
    )
    commands = result.stdout
    assert commands.count("export_obiwan_staged.py stage") == 2
    assert commands.count("LX_CLEARANCE_SINGLE_CHECK=") == len(clearance)
    assert commands.count("LX_R6F_SINGLE_CHECK=") == len(r6f)
    invoked = [line.strip() for line in commands.splitlines()]
    clearance_commands = [
        line for line in invoked if line.endswith("test_clearances.py")]
    r6f_commands = [
        line for line in invoked if line.endswith("test_obiwan_r6f.py")]
    assert len(clearance_commands) == len(clearance)
    assert len(r6f_commands) == len(r6f)
    assert all("LX_CLEARANCE_SINGLE_CHECK=" in line
               for line in clearance_commands)
    assert all("LX_R6F_SINGLE_CHECK=" in line for line in r6f_commands)


def test_public_shell_targets_wait_for_their_validated_stage() -> None:
    """Focused shell joins must not race the Make-owned native stage."""
    root = Path(__file__).resolve().parent
    for target, state, selector in (
            ("check_floor_um_shell", "floor_stand", "test_floor_um_shell"),
            ("check_no_floor_t_shell", "no_floor_stand",
             "test_no_floor_t_shell")):
        result = subprocess.run(
            ["make", "-n", "-B", "-j1", "LX_CAD_EXECUTION=remote-worker",
             f"PYTHON={sys.executable}", target],
            cwd=root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True,
            env={**os.environ, "LX_CAD_GUARD_SLOTS": "1"},
        )
        commands = result.stdout
        stage = commands.index(
            f"--manifest {state}/.obiwan_stage/manifest.json")
        shell = commands.index(f"LX_R6F_SINGLE_CHECK={selector}")
        assert stage < shell


def test_make_check_stamps_skip_unchanged_selector_work() -> None:
    """Real .ok nodes make repeated checks incremental under GNU Make."""
    root = Path(__file__).resolve().parent
    clearance = _make_variable_words("CLEARANCE_CHECK_NAMES")
    r6f = _make_variable_words("R6F_CHECK_NAMES")
    with tempfile.TemporaryDirectory() as text:
        stamps = Path(text) / "checks"
        future = time.time() + 86_400
        for group, names in (("clearance", clearance), ("r6f", r6f)):
            for name in names:
                path = stamps / group / f"{name}.ok"
                _write(path, "verified\n")
                os.utime(path, (future, future))
        closure = stamps / "obiwan_junction_closures.ok"
        _write(closure, "verified\n")
        os.utime(closure, (future, future))

        result = subprocess.run(
            ["make", "-n", "-j8", "LX_CAD_EXECUTION=remote-worker",
             f"PYTHON={sys.executable}", f"CHECK_STAMP_DIR={stamps}",
             "check"],
            cwd=root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True,
            env={**os.environ, "LX_CAD_GUARD_SLOTS": "8"},
        )
        commands = result.stdout
        assert "LX_CLEARANCE_SINGLE_CHECK=" not in commands
        assert "LX_R6F_SINGLE_CHECK=" not in commands
        assert "LX_OBIWAN_JUNCTION_CLOSURE_FULL=1" not in commands

        # A persistent workstation environment is not attestation-keyed like
        # the remote cache, so local mode conservatively reruns selectors.
        local = subprocess.run(
            ["make", "-n", "LX_CAD_EXECUTION=local",
             f"PYTHON={sys.executable}", f"CHECK_STAMP_DIR={stamps}",
             "check_clearance_suite"],
            cwd=root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True,
        )
        assert local.stdout.count("LX_CLEARANCE_SINGLE_CHECK=") == len(
            clearance)


def test_make_uses_scoped_obiwan_prerequisite_group() -> None:
    root = Path(__file__).resolve().parent
    exclusive = _make_variable_words("OBIWAN_EXCLUSIVE_CAD_SRCS")
    assert "top_baffle_nd25fw4_obiwan.py" in exclusive
    assert "top_baffle_nd25fw4_obiwan_lm_split.py" in exclusive
    assert "top_baffle_nd25fw4_obiwan_route.py" not in exclusive

    result = subprocess.run(
        ["make", "-np", "LX_CAD_EXECUTION=remote-worker",
         f"PYTHON={sys.executable}", "floor_stand/stl/.stamp",
         "floor_stand/stl/.stamp_obiwan"],
        cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
        env={**os.environ, "LX_CAD_GUARD_SLOTS": "8"},
    )
    ordinary = next(
        line for line in result.stdout.splitlines()
        if line.startswith("floor_stand/stl/.stamp:"))
    obiwan = next(
        line for line in result.stdout.splitlines()
        if line.startswith("floor_stand/stl/.stamp_obiwan:"))
    for path in exclusive:
        assert path not in ordinary
        assert path in obiwan


def test_local_manifold_mode_is_private() -> None:
    result = subprocess.run(
        ["make", "-n", "LX_CAD_EXECUTION=local-manifold", "all"],
        cwd=Path(__file__).resolve().parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode != 0
    assert "accepts only the private _manifold_parallel target" in result.stdout


def test_promoted_roots_use_make_parallel_full_sweeps() -> None:
    calls = []
    original_run = remote.subprocess.run

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    remote.subprocess.run = capture
    try:
        remote._check_promoted_roots({
            "floor_stand", "no_floor_stand", "wings"})
    finally:
        remote.subprocess.run = original_run
    assert len(calls) == 2
    for command, kwargs in calls:
        assert command[0:2] == ["make", "--no-print-directory"]
        assert f"-j{remote.LOCAL_PROMOTION_JOBS}" in command
        assert "LX_CAD_EXECUTION=local-manifold" in command
        assert "_manifold_parallel" in command
        assert any(item.startswith("MANIFOLD_STAMP_DIR=") for item in command)
        assert kwargs["cwd"] == remote.BAFFLE_DIR
        assert kwargs["check"] is True
    assert "MANIFOLD_ROOTS=floor_stand/stl no_floor_stand/stl" in calls[0][0]
    assert "MANIFOLD_ROOTS=wings/ac/stl wings/ae/stl" in calls[1][0]


def test_obiwan_basic_wing_contract_dependency() -> None:
    result = subprocess.run(
        ["make", "-np", "LX_CAD_EXECUTION=remote-worker",
         f"PYTHON={sys.executable}", "wings/.stamp_ac"],
        cwd=Path(__file__).resolve().parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        env={**os.environ, "LX_CAD_GUARD_SLOTS": "4"},
    )
    rule = next(
        line for line in result.stdout.splitlines()
        if line.startswith("wings/.stamp_ac:"))
    assert "front_down_contract.py" in rule


def test_obiwan_release_parallel_dag() -> None:
    result = subprocess.run(
        ["make", "-n", "-B", "-j4", "LX_CAD_EXECUTION=remote-worker",
         f"PYTHON={sys.executable}", "obiwan_release"],
        cwd=Path(__file__).resolve().parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        env={**os.environ, "LX_CAD_GUARD_SLOTS": "4"},
    )
    output = result.stdout
    assert output.count("export_obiwan_staged.py stage") == 2
    assert output.count(
        "export_piece_stls.py --variant obiwan "
        "--obiwan-stage-manifest") == 2
    assert output.count(
        "cd floor_stand && LX_STAND_FOOT=1 LX_ROUTING_PROFILE=obiwan "
        "PYTHONPATH=..") == 1
    assert output.count(
        "cd no_floor_stand && LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan "
        "PYTHONPATH=..") == 1
    assert output.count(
        "export_obiwan_wings.py --slug ac --output-root wings") == 1
    assert output.count(
        "export_obiwan_wings.py --slug ae --output-root wings") == 1
    assert output.count(
        "write_obiwan_release_manifest.py --state-dir floor_stand") == 1
    assert output.count(
        "write_obiwan_release_manifest.py --state-dir no_floor_stand") == 1
    for check in (
            "test_route_contract", "test_floor_lm_keyed_split",
            "test_no_floor_lm_keyed_split", "test_floor_um_shell",
            "test_floor_t_shell", "test_no_floor_um_shell",
            "test_no_floor_t_shell",
            "test_floor_feed_and_flush_mouth_contract",
            "test_feed_and_flush_mouth_contract",
            "test_floor_lm_burial_web_contract",
            "test_lm_burial_web_contract",
            "test_floor_um_burial_web_contract",
            "test_um_burial_web_contract",
            "test_floor_bump_backfill_contract",
            "test_bump_backfill_contract",
            "test_floor_integrated_mount"):
        assert output.count(f"LX_R6F_SINGLE_CHECK={check}") == 1
    # The captive catalog is project-wide, so the public release target now
    # regenerates both states of every magnet-bearing family before writing
    # its 56-STL inventory.  V1L therefore appears once per stand state.
    assert output.count(
        "export_piece_stls.py --variant v1l --outdir") == 2
    assert output.count(
        "generate_captive_magnet_catalog.py --output ") == 1
    assert "check_manifold.py --obiwan-only" not in output
    # Fresh remote snapshots intentionally contain no generated roots, so a
    # dry-run cannot discover per-STL submake nodes.  The 92-node fixture test
    # above proves that expansion independently.  Metadata runs once per
    # completed state, once for Ac/Ae, and once at the cross-state join.
    assert output.count("check_manifold.py --metadata-only") == 4
    assert output.count(
        "MANIFOLD_ROOTS='floor_stand/stl'") == 1
    assert output.count(
        "MANIFOLD_ROOTS='no_floor_stand/stl'") == 1
    assert "MANIFOLD_ROOTS='floor_stand/stl no_floor_stand/stl'" not in output
    assert output.count(
        "MANIFOLD_ROOTS='wings/ac/stl wings/ae/stl'") == 1
    assert "check_manifold.py floor_stand/stl" not in output


def test_profile_specific_stl_dag() -> None:
    root = Path(__file__).resolve().parent
    targets = (
        "floor_stand/stl/.stamp_v1l",
        "floor_stand/stl/.stamp_obiwan",
    )
    remote_result = subprocess.run(
        ["make", "-n", "-B", "LX_CAD_EXECUTION=remote-worker",
         f"PYTHON={sys.executable}", *targets],
        cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
        env={**os.environ, "LX_CAD_GUARD_SLOTS": "8"},
    )
    assert remote_result.stdout.count(
        "export_piece_stls.py --variant v1l --outdir") == 1
    assert remote_result.stdout.count(
        "export_piece_stls.py --variant obiwan --obiwan-stage-manifest") == 1
    assert "--v1l-piece" not in remote_result.stdout
    assert "--obiwan-part" not in remote_result.stdout

    local_result = subprocess.run(
        ["make", "-n", "-B", "LX_CAD_EXECUTION=local",
         f"PYTHON={sys.executable}", *targets],
        cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
        env=os.environ.copy(),
    )
    assert local_result.stdout.count("--v1l-piece") == 5
    assert local_result.stdout.count("--obiwan-part") == 4
    assert "--obiwan-part lm_split" in local_result.stdout
    assert "--obiwan-part support" not in local_result.stdout


def test_local_checker_interpreter() -> None:
    python = remote._local_checker_python()
    subprocess.run(
        [str(python), "-c", "from PIL import Image"], check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_local_memory_profile_has_no_host_free_floor() -> None:
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["LX_CAD_MEMORY_PROFILE"] = "local-macos"
    env.pop("LX_CAD_MIN_FREE_MB", None)
    env.pop("LX_CAD_MEMORY_GUARDED", None)
    env.pop("LX_CAD_MEMORY_GUARD_PID", None)
    probe = (
        "import json, run_memory_guarded as guard; "
        "print(json.dumps({"
        "'max_rss_mb': guard.MAX_RSS_MB, "
        "'min_free_mb': guard.MIN_FREE_MB, "
        "'local_profile': guard.MEMORY_PROFILES['local-macos'], "
        "'remote_profile': guard.MEMORY_PROFILES['osado-512g']}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=root, env=env, check=True,
        text=True, stdout=subprocess.PIPE)
    policy = json.loads(result.stdout)
    assert policy["max_rss_mb"] == 8192
    assert policy["min_free_mb"] == 0
    assert policy["local_profile"] == {
        "max_rss_mb": 8192,
        "min_free_mb": 0,
        "max_guard_slots": 1,
    }
    assert policy["remote_profile"]["min_free_mb"] == 64 * 1024
    assert remote.REMOTE_MEMORY_FLOOR_MIB == 64 * 1024
    assert remote.REMOTE_MEMORY_MAX_MIB == 512 * 1024

    # Prove a disabled local floor never samples the host-free metric. RSS
    # monitoring remains live while the short child sleeps.
    no_floor_probe = (
        "import run_memory_guarded as guard, sys; "
        "guard._free_memory_mib=lambda: (_ for _ in ()).throw("
        "AssertionError('disabled free-memory sampler was called')); "
        "raise SystemExit(guard.main([sys.executable, '-c', "
        "'import time; time.sleep(0.2)']))"
    )
    subprocess.run(
        [sys.executable, "-c", no_floor_probe], cwd=root, env=env,
        check=True)

    rss_probe = (
        "import run_memory_guarded as guard, sys; "
        "guard._free_memory_mib=lambda: (_ for _ in ()).throw("
        "AssertionError('disabled free-memory sampler was called')); "
        "guard._process_tree_rss_kib=lambda _pid: "
        "(guard.MAX_RSS_MB + 1) * 1024; "
        "raise SystemExit(guard.main([sys.executable, '-c', "
        "'import time; time.sleep(5)']))"
    )
    result = subprocess.run(
        [sys.executable, "-c", rss_probe], cwd=root, env=env)
    assert result.returncode == 99

    # Local callers may opt into a stricter positive floor.
    env["LX_CAD_MIN_FREE_MB"] = "777"
    result = subprocess.run(
        [sys.executable, "-c",
         "import run_memory_guarded as guard; print(guard.MIN_FREE_MB)"],
        cwd=root, env=env, check=True, text=True, stdout=subprocess.PIPE)
    assert result.stdout.strip() == "777"
    floor_probe = (
        "import run_memory_guarded as guard, sys; "
        "guard._free_memory_mib=lambda: 776.0; "
        "raise SystemExit(guard.main([sys.executable, '-c', 'pass']))"
    )
    result = subprocess.run(
        [sys.executable, "-c", floor_probe], cwd=root, env=env)
    assert result.returncode == 99


def test_step_label_line_wrapping() -> None:
    with tempfile.TemporaryDirectory() as text:
        step = Path(text) / "wrapped.step"
        step.write_bytes(
            b"#1=PRODUCT('REFERENCE_UM_D7_LM_printed_cover_then_free_"
            b"behind_UM_R15_R20_Faston_\r\nhandoff','',());\n")
        python = remote._local_checker_python()
        subprocess.run(
            [str(python), "-c",
             "from pathlib import Path; import sys; "
             "from check_manifold import _contains_bytes; "
             "token=(b'REFERENCE_UM_D7_LM_printed_cover_then_free_behind_UM_'"
             "+b'R15_R20_Faston_handoff'); "
             "path=Path(sys.argv[1]); "
             "assert _contains_bytes(path,token); "
             "assert not _contains_bytes(path,token+b'_obsolete')",
             str(step)],
            cwd=Path(__file__).resolve().parent,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    test_target_contract()
    test_default_remote_parallelism()
    test_remote_make_and_guard_share_parallelism_authority()
    test_remote_worker_exports_source_snapshot_identity()
    test_protocol_rejection()
    test_dead_job_reconciliation_record()
    test_launch_uses_snapshot_transition()
    test_launch_cancel_transition_is_serial_and_fail_closed()
    test_environment_hash_is_binary_stable()
    test_transport_is_bound_to_source_snapshot()
    test_atomic_promotion_and_rollback()
    test_persistent_promotion_recovery()
    test_foreign_promotion_recovered_before_new_job()
    test_fetch_requires_succeeded_status()
    test_unexplained_backup_is_preserved()
    test_promotion_signals_roll_back()
    test_remote_clean_preserves_review()
    test_verified_make_cache_seed_and_checksum_overlay()
    test_warm_cache_always_bundles_public_target_outputs()
    test_warm_cache_outputs_promote_into_fresh_local_trees()
    test_source_deletion_forces_true_cold_cache_fallback()
    test_exact_source_focused_publication_unions_cache_coverage()
    test_cache_union_respects_complete_roots_and_clean()
    test_cache_publication_does_not_union_across_source_deletion()
    test_make_cache_publication_order_and_damage_fallback()
    test_cache_publication_follows_verified_local_promotion()
    test_parallel_stage_dag()
    test_obiwan_basic_wing_parallel_dag()
    test_make_parallel_manifold_dag()
    test_make_obiwan_only_manifold_filters_warm_legacy_meshes()
    test_candidate_is_one_flat_make_dag()
    test_make_check_registries_match_python_and_do_not_fan_out()
    test_public_shell_targets_wait_for_their_validated_stage()
    test_make_check_stamps_skip_unchanged_selector_work()
    test_make_uses_scoped_obiwan_prerequisite_group()
    test_local_manifold_mode_is_private()
    test_promoted_roots_use_make_parallel_full_sweeps()
    test_obiwan_basic_wing_contract_dependency()
    test_obiwan_release_parallel_dag()
    test_profile_specific_stl_dag()
    test_local_checker_interpreter()
    test_local_memory_profile_has_no_host_free_floor()
    test_step_label_line_wrapping()
    print("all remote CAD transport checks passed")


if __name__ == "__main__":
    main()
