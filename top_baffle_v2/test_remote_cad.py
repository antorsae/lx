"""Lightweight regression checks for the resumable remote CAD transport."""

from __future__ import annotations

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
from types import SimpleNamespace

import remote_cad as remote


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _transaction_fixture(root: Path):
    repo = root / "lx"
    baffle = repo / "top_baffle_v2"
    incoming = root / "incoming"
    for state in remote.STATE_OUTPUT_ROOTS:
        _write(baffle / state / "marker", f"old-{state}")
        _write(incoming / "top_baffle_v2" / state / "marker", f"new-{state}")
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
            assert promoted == 3
            for state in remote.STATE_OUTPUT_ROOTS:
                assert (baffle / state / "marker").read_text() == f"new-{state}"
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
            assert baffle.joinpath(
                "top_baffle_nd25fw4_attachments.step").read_text() == (
                    "old-common")
            assert (baffle / "review" / "keep.png").read_text() == (
                "keep-review")
        finally:
            _restore_project(originals)


def test_parallel_stage_dag() -> None:
    result = subprocess.run(
        ["make", "-n", "-j4", "LX_CAD_EXECUTION=remote-worker",
         f"PYTHON={sys.executable}", "validate_v1lf_stages",
         "floor_stand/.v1lf_stage/manifest.json",
         "no_floor_stand/.v1lf_stage/manifest.json"],
        cwd=Path(__file__).resolve().parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        env={**os.environ, "LX_CAD_GUARD_SLOTS": "4"},
    )
    assert result.stdout.count("export_v1lf_staged.py stage") == 2


def test_profile_specific_stl_dag() -> None:
    root = Path(__file__).resolve().parent
    targets = (
        "floor_stand/stl/.stamp_v1l",
        "floor_stand/stl/.stamp_v1lf",
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
        "export_piece_stls.py --variant v1lf --v1lf-stage-manifest") == 1
    assert "--v1l-piece" not in remote_result.stdout
    assert "--v1lf-part" not in remote_result.stdout

    local_result = subprocess.run(
        ["make", "-n", "-B", "LX_CAD_EXECUTION=local",
         f"PYTHON={sys.executable}", *targets],
        cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
        env=os.environ.copy(),
    )
    assert local_result.stdout.count("--v1l-piece") == 5
    assert local_result.stdout.count("--v1lf-part") == 5
    assert "--v1lf-part lm_split" in local_result.stdout


def test_local_checker_interpreter() -> None:
    python = remote._local_checker_python()
    subprocess.run(
        [str(python), "-c", "from PIL import Image"], check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    test_parallel_stage_dag()
    test_profile_specific_stl_dag()
    test_local_checker_interpreter()
    test_step_label_line_wrapping()
    print("all remote CAD transport checks passed")


if __name__ == "__main__":
    main()
