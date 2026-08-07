#!/usr/bin/env python3
"""Run the top-baffle Make workflow on a bounded, resumable remote host.

The local working tree is authoritative.  Each invocation packages the exact
source inputs into a content-addressed snapshot, creates an isolated remote
job, runs it in a 512 GiB systemd cgroup, and returns only hash-verified build
artifacts.  A lost SSH connection does not stop the remote systemd job.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import platform
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback


SCRIPT = Path(__file__).resolve()
BAFFLE_DIR = PROJECT_ROOT
REPO_ROOT = BAFFLE_DIR.parent
LOCAL_STATE = BAFFLE_DIR / ".remote-cad"
LOCK_FILE = BAFFLE_DIR / "cad-remote-requirements.lock"
TRANSPORT_SOURCE_PATH = "top_baffle_v2/scripts/remote_cad.py"
LEGACY_TRANSPORT_SOURCE_PATH = "top_baffle_v2/remote_cad.py"
TRANSPORT_SOURCE_PATHS = (
    TRANSPORT_SOURCE_PATH,
    LEGACY_TRANSPORT_SOURCE_PATH,
)
REQUIREMENTS_SOURCE_PATH = "top_baffle_v2/cad-remote-requirements.lock"

PROTOCOL_VERSION = 3
ENVIRONMENT_ATTESTATION_VERSION = 3
PROMOTION_TRANSACTION_VERSION = 1
BUILD_CACHE_VERSION = 1
PERFORMANCE_PROFILE_VERSION = 1
REMOTE_PYTHON_VERSION = "3.12.12"
REMOTE_MEMORY_PROFILE = "osado-512g"
REMOTE_MEMORY_MAX_MIB = 512 * 1024
REMOTE_MEMORY_FLOOR_MIB = 64 * 1024
REMOTE_MEMORY_MAX_SYSTEMD = "512G"
DEFAULT_REMOTE_JOBS = 16
MAX_REMOTE_JOBS = 16
LOCAL_PROMOTION_JOBS = 8

DEFAULT_HOST = "osado.lan"
DEFAULT_REMOTE_ROOT = "~/temp/lx-cad"
OUTPUT_ROOT_PREFIXES = {
    "floor_stand": Path("build/floor_stand"),
    "no_floor_stand": Path("build/no_floor_stand"),
    "wings": Path("build/wings"),
    "tebm35c10_4": Path("build/vase_TEBM35C10-4"),
}
STATE_OUTPUT_ROOTS = ("floor_stand", "no_floor_stand")
ARTIFACT_SCAN_PREFIXES = (
    *OUTPUT_ROOT_PREFIXES.values(),
    Path("build/common"),
    Path("review"),
)
GENERATED_SUFFIXES = {
    ".3mf", ".brep", ".glb", ".json", ".png", ".step", ".stl",
}
COMMON_ARTIFACT = (
    "top_baffle_v2/build/common/attachments.step")
OBIWAN_WING_DESIGN_MAP_ARTIFACT = (
    "top_baffle_v2/build/common/obiwan_wing_design_map.png")
CAPTIVE_MAGNET_CATALOG_ARTIFACT = (
    "top_baffle_v2/review/captive_magnet_release_catalog.json")
SOURCE_EXCLUDED_DIRS = {".remote-cad", "__pycache__", "review"}
SOURCE_EXCLUDED_SUFFIXES = {
    ".3mf", ".brep", ".glb", ".png", ".pyc", ".step", ".stl",
}
# BambuStudio writes this transient status payload beside the invoked model.
# It is neither an input nor a release artifact, and including it made an
# otherwise identical snapshot miss the verified Make cache after slicing.
SOURCE_EXCLUDED_NAMES = {"result.json"}
REFERENCE_INPUTS = (
    REPO_ROOT / "E0022_W22EX001.stp",
    REPO_ROOT / "linkwitz" / "H1658-04_MU10RB-SL_driver.stl",
    REPO_ROOT / "linkwitz" / "H1658-04_MU10RB-SL_driver_STL_notes.md",
    REPO_ROOT / "linkwitz" / "H1658-04_MU10RB-SL_Datasheet.pdf",
)
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
HOST_RE = re.compile(r"^(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9_.-]+$")
TARGET_RE = re.compile(r"^[A-Za-z0-9_./:+%-]+$")
REMOTE_MAKE_TARGETS = {
    "all", "candidate", "release", "floor_stand", "floor_obiwan",
    "no_floor_stand", "no_floor_obiwan", "no_floor_obiwan_01a",
    "obiwan_release", "obiwan_state_releases",
    "obiwan_wings", "obiwan_wing_exports",
    "obiwan_wing_artifacts", "check_obiwan_wings",
    "common", "check", "check_captive_magnets", "check_obiwan",
    "check_obiwan_shells",
    "check_obiwan_t_shells", "check_obiwan_service",
    "check_obiwan_closure_focus",
    "check_obiwan_um_pilot_spoke",
    "check_obiwan_mouths", "check_obiwan_burial",
    "check_obiwan_um_burial",
    "check_obiwan_backfills", "check_obiwan_route_boundaries",
    "check_floor_um_shell", "check_floor_t_shell",
    "check_no_floor_um_shell", "check_no_floor_t_shell",
    "check_floor_obiwan_mouths", "check_no_floor_obiwan_mouths",
    "check_no_floor_obiwan_mouths_focused",
    "check_floor_obiwan_burial", "check_no_floor_obiwan_burial",
    "check_floor_obiwan_um_burial", "check_no_floor_obiwan_um_burial",
    "check_floor_obiwan_backfills", "check_no_floor_obiwan_backfills",
    "check_route_contract", "check_bump_brep",
    "check_floor_integrated_mount",
    "check_no_floor_lm_mesh", "check_obiwan_lm_split",
    "check_obiwan_lm_profile", "check_obiwan_junction_closure_plans",
    "check_obiwan_junction_closure_base",
    "check_obiwan_junction_closures",
    "check_obiwan_lm_split_two_pin_static",
    "manifold", "clean",
    "refresh_captive_magnet_catalog_existing",
    "vase_tebm35c10_4_cad",
    "validate_obiwan_stages",
    "validate_floor_obiwan_stage", "validate_no_floor_obiwan_stage",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _git_fact(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _is_under_relative_prefix(relative: Path, prefix: Path) -> bool:
    """Return whether a project-relative path belongs to ``prefix``."""
    return relative == prefix or prefix in relative.parents


def _logical_output_root(relative: Path) -> str | None:
    """Map a project-relative artifact path to its atomic logical root."""
    matches = [
        logical for logical, prefix in OUTPUT_ROOT_PREFIXES.items()
        if _is_under_relative_prefix(relative, prefix)
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"overlapping remote output-root registry for {relative}")
    return matches[0] if matches else None


def _output_prefix(logical_root: str) -> Path:
    try:
        return OUTPUT_ROOT_PREFIXES[logical_root]
    except KeyError as exc:
        raise ValueError(
            f"unknown remote output root: {logical_root!r}") from exc


def _source_paths(*, include_candidate_outputs: bool = False) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for path in BAFFLE_DIR.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(BAFFLE_DIR)
        logical_root = _logical_output_root(relative)
        if logical_root is not None:
            if include_candidate_outputs and logical_root in STATE_OUTPUT_ROOTS:
                paths.add(path)
            continue
        if _is_under_relative_prefix(relative, Path("build/common")):
            continue
        if any(part in SOURCE_EXCLUDED_DIRS for part in relative.parts):
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        if relative.name in SOURCE_EXCLUDED_NAMES:
            continue
        if path.suffix.lower() in SOURCE_EXCLUDED_SUFFIXES:
            continue
        paths.add(path)
    paths.update(REFERENCE_INPUTS)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required remote CAD input is missing: "
            + ", ".join(str(path) for path in missing)
        )
    return tuple(sorted(paths, key=lambda item: item.relative_to(REPO_ROOT).as_posix()))


def _source_identity(records: list[dict]) -> str:
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "files": records,
    }
    return _sha256_bytes(_canonical_json(identity))


_SOURCE_DIGEST_CACHE: dict[str, tuple[int, int, str]] = {}


def _cached_sha256_file(path: Path) -> tuple[int, str]:
    """File digest behind an mtime/size-validated per-process cache.

    The dispatcher recomputes the complete source identity at several
    race-detection checkpoints per run.  Those checkpoints stay: any content
    change bumps st_mtime_ns and invalidates the cached digest, so each
    checkpoint still detects modification — unchanged files just stop being
    re-read and re-hashed at every checkpoint.
    """
    info = path.stat()
    key = path.as_posix()
    cached = _SOURCE_DIGEST_CACHE.get(key)
    if (cached is not None and cached[0] == info.st_size
            and cached[1] == info.st_mtime_ns):
        return info.st_size, cached[2]
    digest = _sha256_file(path)
    _SOURCE_DIGEST_CACHE[key] = (info.st_size, info.st_mtime_ns, digest)
    return info.st_size, digest


def _source_records(
        paths: tuple[Path, ...] | None = None, *,
        include_candidate_outputs: bool = False) -> list[dict]:
    """Hash the exact local inputs without constructing an archive."""
    paths = (_source_paths(include_candidate_outputs=include_candidate_outputs)
             if paths is None else paths)
    records = []
    for path in paths:
        size, digest = _cached_sha256_file(path)
        records.append({
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "size": size,
            "sha256": digest,
        })
    return records


def _create_source_archive(
        directory: Path, *,
        include_candidate_outputs: bool = False) -> tuple[Path, dict]:
    paths = _source_paths(
        include_candidate_outputs=include_candidate_outputs)
    records = _source_records(paths)
    source_hash = _source_identity(records)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source_sha256": source_hash,
        "created_utc": _utc_now(),
        "git_head": _git_fact("rev-parse", "HEAD"),
        "git_branch": _git_fact("rev-parse", "--abbrev-ref", "HEAD"),
        "files": records,
    }
    archive = directory / f"lx-cad-source-{source_hash}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as bundle:
        for path in paths:
            relative = path.relative_to(REPO_ROOT).as_posix()
            info = bundle.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
        encoded = json.dumps(
            manifest, indent=2, sort_keys=True,
        ).encode("utf-8") + b"\n"
        info = tarfile.TarInfo(".lx-cad-source.json")
        info.size = len(encoded)
        info.mode = 0o444
        info.mtime = 0
        bundle.addfile(info, fileobj=_BytesReader(encoded))
    return archive, manifest


def _transport_binding_from_manifest(manifest: dict) -> dict:
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported source protocol for CAD transport")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("source manifest has no transport record list")
    matches = [
        record for record in records
        if isinstance(record, dict)
        and record.get("path") in TRANSPORT_SOURCE_PATHS
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "source snapshot must contain exactly one remote CAD transport")
    record = matches[0]
    if (set(record) != {"path", "size", "sha256"}
            or type(record["size"]) is not int or record["size"] < 1
            or not isinstance(record["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
        raise RuntimeError("source transport record is malformed")
    return {
        "source_path": record["path"],
        "sha256": record["sha256"],
        "size": record["size"],
    }


def _extract_snapshot_transport(
        archive: Path, directory: Path, manifest: dict) -> tuple[Path, dict]:
    """Materialize the exact tool already sealed into the source archive."""
    binding = _transport_binding_from_manifest(manifest)
    with tarfile.open(archive, "r:gz") as bundle:
        members = [
            member for member in bundle.getmembers()
            if member.name == binding["source_path"]
        ]
        if len(members) != 1 or not members[0].isfile():
            raise RuntimeError("source archive transport member is malformed")
        stream = bundle.extractfile(members[0])
        if stream is None:
            raise RuntimeError("cannot read source archive transport member")
        data = stream.read()
    if (len(data) != binding["size"]
            or _sha256_bytes(data) != binding["sha256"]):
        raise RuntimeError("source archive transport differs from its manifest")
    local_tool = directory / f"remote_cad-{binding['sha256']}.py"
    local_tool.write_bytes(data)
    os.chmod(local_tool, 0o444)
    return local_tool, binding


def _extract_snapshot_requirements(
        archive: Path, manifest: dict) -> bytes:
    """Return the exact lockfile bytes sealed into the source archive."""
    records = [
        record for record in manifest.get("files", [])
        if isinstance(record, dict)
        and record.get("path") == REQUIREMENTS_SOURCE_PATH
    ]
    if len(records) != 1:
        raise RuntimeError(
            "source snapshot must contain exactly one remote requirements lock")
    record = records[0]
    if (set(record) != {"path", "size", "sha256"}
            or type(record.get("size")) is not int or record["size"] < 1
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
        raise RuntimeError("source requirements-lock record is malformed")
    with tarfile.open(archive, "r:gz") as bundle:
        members = [
            member for member in bundle.getmembers()
            if member.name == REQUIREMENTS_SOURCE_PATH
        ]
        if len(members) != 1 or not members[0].isfile():
            raise RuntimeError("source requirements-lock member is malformed")
        stream = bundle.extractfile(members[0])
        if stream is None:
            raise RuntimeError("cannot read source requirements-lock member")
        data = stream.read()
    if (len(data) != record["size"]
            or _sha256_bytes(data) != record["sha256"]):
        raise RuntimeError(
            "source requirements lock differs from its manifest")
    return data


def _validate_transport_provenance(
        metadata: dict, source_manifest: dict, *,
        executing_tool: Path = SCRIPT) -> dict:
    if source_manifest.get("source_sha256") != metadata.get("source_sha256"):
        raise RuntimeError("remote CAD transport source identity mismatch")
    binding = _transport_binding_from_manifest(source_manifest)
    if metadata.get("transport") != binding:
        raise RuntimeError("remote CAD transport provenance mismatch")
    if (executing_tool.stat().st_size != binding["size"]
            or _sha256_file(executing_tool) != binding["sha256"]):
        raise RuntimeError(
            "executing remote CAD transport differs from source snapshot")
    return binding


class _BytesReader:
    """Minimal file object accepted by TarFile.addfile without io imports."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.data) - self.offset
        result = self.data[self.offset:self.offset + size]
        self.offset += len(result)
        return result


def _safe_member_path(root: Path, name: str) -> Path:
    if not name or name.startswith("/"):
        raise RuntimeError(f"unsafe archive path: {name!r}")
    path = Path(name)
    if ".." in path.parts:
        raise RuntimeError(f"unsafe archive path: {name!r}")
    target = root.joinpath(*path.parts)
    if root.resolve() not in (target.resolve(), *target.resolve().parents):
        raise RuntimeError(f"archive path escapes destination: {name!r}")
    return target


def _extract_regular_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"remote CAD archives may contain only files/directories: "
                    f"{member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot extract {member.name}")
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(stream, output, 1024 * 1024)
                os.chmod(temporary, member.mode & 0o777)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def _verify_source(root: Path, expected_hash: str, allow_extra: bool) -> dict:
    manifest_path = root / ".lx-cad-source.json"
    manifest = _read_json(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("source manifest has no file list")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported remote CAD source protocol")
    actual_identity = _source_identity(records)
    if actual_identity != expected_hash or manifest.get("source_sha256") != expected_hash:
        raise RuntimeError("remote CAD source identity mismatch")
    expected_paths = set()
    for record in records:
        if (not isinstance(record, dict)
                or set(record) != {"path", "size", "sha256"}
                or not isinstance(record.get("path"), str)
                or type(record.get("size")) is not int
                or record["size"] < 0
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or record["path"] in expected_paths):
            raise RuntimeError("invalid source-manifest record")
        relative = record["path"]
        path = _safe_member_path(root, relative)
        expected_paths.add(relative)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source input missing or not regular: {relative}")
        if path.stat().st_size != record.get("size"):
            raise RuntimeError(f"source input size mismatch: {relative}")
        if _sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"source input hash mismatch: {relative}")
    if not allow_extra:
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ".lx-cad-source.json"
        }
        if actual_paths != expected_paths:
            raise RuntimeError("immutable source snapshot has unexpected files")
    return manifest


def _tree_records(root: Path) -> list[dict]:
    """Hash one cache tree, including the mtimes consumed by GNU Make.

    The cache is only a byte-for-byte seed for a fresh isolated job.  It has
    no dependency logic of its own: after the exact source overlay below,
    GNU Make remains the sole authority deciding which targets are stale.
    Binding mtimes as well as bytes prevents a damaged cache entry from
    silently making an old target look newer than its prerequisites.
    """
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"build cache may not contain symlinks: {path}")
        if not path.is_file():
            continue
        stat_result = path.stat()
        records.append({
            "path": path.relative_to(root).as_posix(),
            "size": stat_result.st_size,
            "sha256": _sha256_file(path),
            "mode": stat_result.st_mode & 0o777,
            "mtime_ns": stat_result.st_mtime_ns,
        })
    return records


def _validated_tree_records(records: object) -> dict[str, dict]:
    if not isinstance(records, list):
        raise RuntimeError("build cache has no file record list")
    validated = {}
    for record in records:
        if (not isinstance(record, dict)
                or set(record) != {
                    "path", "size", "sha256", "mode", "mtime_ns"}
                or not isinstance(record.get("path"), str)
                or type(record.get("size")) is not int
                or record["size"] < 0
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or type(record.get("mode")) is not int
                or not 0 <= record["mode"] <= 0o777
                or type(record.get("mtime_ns")) is not int
                or record["mtime_ns"] < 0
                or record["path"] in validated):
            raise RuntimeError("invalid/duplicate build-cache file record")
        # Apply the same traversal/absolute-path rejection as archive input.
        _safe_member_path(Path("/tmp/lx-cad-cache-record-root"), record["path"])
        validated[record["path"]] = record
    return validated


def _verify_tree_records(root: Path, records: object) -> dict[str, dict]:
    expected = _validated_tree_records(records)
    actual_paths = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"build cache may not contain symlinks: {path}")
        if path.is_file():
            actual_paths[path.relative_to(root).as_posix()] = path
    if set(actual_paths) != set(expected):
        raise RuntimeError("build-cache tree differs from its manifest")
    for relative, path in actual_paths.items():
        record = expected[relative]
        stat_result = path.stat()
        if (stat_result.st_size != record["size"]
                or (stat_result.st_mode & 0o777) != record["mode"]
                or stat_result.st_mtime_ns != record["mtime_ns"]
                or _sha256_file(path) != record["sha256"]):
            raise RuntimeError(f"build-cache file mismatch: {relative}")
    return expected


def _build_cache_entry(
        remote_root: Path, environment_hash: str,
        environment_attestation_hash: str) -> Path:
    if (not re.fullmatch(r"[0-9a-f]{64}", environment_hash)
            or not re.fullmatch(
                r"[0-9a-f]{64}", environment_attestation_hash)):
        raise RuntimeError("invalid build-cache environment identity")
    identity = f"{environment_hash}-{environment_attestation_hash}"
    return remote_root / "cache" / "make" / identity


@contextmanager
def _build_cache_lock(
        remote_root: Path, environment_hash: str,
        environment_attestation_hash: str):
    identity = f"{environment_hash}-{environment_attestation_hash}"
    lock_path = (
        remote_root / "locks" / f"make-cache-{identity}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _copy_tree(source: Path, destination: Path) -> None:
    """Copy a cache tree, using Linux reflinks when the host supports them."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    if sys.platform == "linux" and shutil.which("cp"):
        try:
            subprocess.run(
                ["cp", "-a", "--reflink=auto", f"{source}/.",
                 str(destination)],
                check=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            return
        except (OSError, subprocess.CalledProcessError):
            shutil.rmtree(destination, ignore_errors=True)
            destination.mkdir(parents=True)
    shutil.copytree(
        source, destination, dirs_exist_ok=True,
        copy_function=shutil.copy2)


def _copy_missing_tree_files(
        source: Path, destination: Path, *,
        excluded_prefixes: tuple[Path, ...] = ()) -> tuple[str, ...]:
    """Add only paths absent from an already copied exact-source worktree.

    Cache publication uses this to retain the union of successful Make target
    coverage for one immutable source snapshot.  Existing paths belong to the
    job with the newer completion timestamp and are never overwritten by an
    older/sparser publisher.
    """
    added = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"build cache may not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(relative == prefix or prefix in relative.parents
               for prefix in excluded_prefixes):
            continue
        target = destination / relative
        if target.exists() or target.is_symlink():
            if not target.is_file() or target.is_symlink():
                raise RuntimeError(
                    f"build-cache union path-type collision: {relative}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        added.append(relative.as_posix())
    return tuple(added)


def _verify_build_cache_entry(
        entry: Path, environment_hash: str,
        environment_attestation_hash: str) -> tuple[dict, dict[str, dict]]:
    marker = _read_json(entry / "cache.json")
    if (marker.get("format_version") != BUILD_CACHE_VERSION
            or marker.get("environment_sha256") != environment_hash
            or marker.get("environment_attestation_sha256")
            != environment_attestation_hash
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(marker.get("source_sha256", "")))
            or not JOB_ID_RE.fullmatch(
                str(marker.get("published_from_job", "")))
            or type(marker.get("build_completed_ns")) is not int
            or marker["build_completed_ns"] <= 0
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(marker.get("artifact_manifest_sha256", "")))):
        raise RuntimeError("build-cache marker is malformed")
    work = entry / "work"
    records = _verify_tree_records(work, marker.get("files"))
    source = _verify_source(
        work, marker["source_sha256"], allow_extra=True)
    if source.get("source_sha256") != marker["source_sha256"]:
        raise RuntimeError("build-cache source binding is inconsistent")
    return marker, records


def _overlay_source_snapshot(
        cached_work: Path, current_work: Path,
        current_source_hash: str) -> tuple[str, ...]:
    """Overlay exact sources while retaining mtimes for unchanged inputs.

    Checksums identify only which source bytes changed between immutable
    snapshots.  Changed/new source mtimes advance beyond every cached target;
    unchanged source mtimes and all generated target mtimes remain untouched.
    Make then performs the actual dependency traversal.
    """
    old_manifest = _verify_source(
        cached_work,
        _read_json(cached_work / ".lx-cad-source.json")["source_sha256"],
        allow_extra=True)
    current_manifest = _verify_source(
        current_work, current_source_hash, allow_extra=False)
    old_records = {
        record["path"]: record for record in old_manifest["files"]}
    current_records = {
        record["path"]: record for record in current_manifest["files"]}

    removed = sorted(set(old_records) - set(current_records))
    if removed:
        raise RuntimeError(
            "build-cache source overlay may not retain outputs across "
            f"deleted inputs: {', '.join(removed)}")

    maximum_cached_mtime = max(
        (path.stat().st_mtime_ns for path in cached_work.rglob("*")
         if path.is_file() and not path.is_symlink()),
        default=0)
    advanced_mtime = max(time.time_ns(), maximum_cached_mtime + 1)
    changed = []
    for relative, record in sorted(current_records.items()):
        old = old_records.get(relative)
        if (old is not None
                and old["size"] == record["size"]
                and old["sha256"] == record["sha256"]):
            continue
        source = _safe_member_path(current_work, relative)
        destination = _safe_member_path(cached_work, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.source-overlay")
        try:
            shutil.copyfile(source, temporary)
            os.chmod(temporary, source.stat().st_mode & 0o777)
            os.utime(temporary, ns=(advanced_mtime, advanced_mtime))
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        changed.append(relative)

    # The manifest is control metadata, not a Make prerequisite.  Replace it
    # after the content overlay and bind it to the current immutable source.
    shutil.copy2(
        current_work / ".lx-cad-source.json",
        cached_work / ".lx-cad-source.json")
    _verify_source(cached_work, current_source_hash, allow_extra=True)
    return tuple(changed)


def _seed_build_cache(job: Path, metadata: dict) -> dict | None:
    """Atomically replace a source-only job worktree with a verified seed."""
    if metadata.get("include_candidate_outputs"):
        return None
    remote_root = Path(metadata["remote_root"])
    environment_hash = metadata["environment_sha256"]
    attestation_hash = metadata["environment_attestation_sha256"]
    entry = _build_cache_entry(
        remote_root, environment_hash, attestation_hash)
    with _build_cache_lock(
            remote_root, environment_hash, attestation_hash):
        if not entry.is_dir():
            return None
        try:
            marker, cache_records = _verify_build_cache_entry(
                entry, environment_hash, attestation_hash)
        except (OSError, ValueError, RuntimeError):
            # A cache is never authoritative.  Remove a damaged entry under
            # its exclusive lock and continue with the exact cold snapshot.
            shutil.rmtree(entry, ignore_errors=True)
            return None

        source_work = job / "work"
        # Make can determine which declared targets are stale after a source
        # edit or addition, but it cannot generically infer which generated
        # files belonged exclusively to a now-deleted source.  Reject reuse
        # before cloning so the fresh source-only job remains a true cold
        # build with no stale generated artifacts to inherit.
        cached_source = _read_json(entry / "work" / ".lx-cad-source.json")
        current_source = _verify_source(
            source_work, metadata["source_sha256"], allow_extra=False)
        cached_paths = {
            record["path"] for record in cached_source["files"]}
        current_paths = {
            record["path"] for record in current_source["files"]}
        removed_sources = sorted(cached_paths - current_paths)
        if removed_sources:
            return None

        temporary = job / f".work-cache-seed-{os.getpid()}"
        backup = job / f".work-source-only-{os.getpid()}"
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        try:
            _copy_tree(entry / "work", temporary)
            changed = _overlay_source_snapshot(
                temporary, source_work, metadata["source_sha256"])
            # The baseline lets artifact packaging exclude inherited cache
            # files from focused jobs.  Complete output-root targets still
            # publish their entire declared roots.
            _atomic_json(job / "cache-seed.json", {
                "format_version": BUILD_CACHE_VERSION,
                "cache_source_sha256": marker["source_sha256"],
                "cache_build_completed_ns": marker["build_completed_ns"],
                "changed_source_paths": list(changed),
                "files": list(cache_records.values()),
            })
            source_work.replace(backup)
            try:
                temporary.replace(source_work)
            except BaseException:
                backup.replace(source_work)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
        return {
            "source_sha256": marker["source_sha256"],
            "build_completed_ns": marker["build_completed_ns"],
            "changed_source_paths": changed,
        }


class Remote:
    def __init__(self, host: str):
        if not HOST_RE.fullmatch(host):
            raise ValueError(f"invalid LX_CAD_REMOTE_HOST: {host!r}")
        self.host = host
        self.ssh = shutil.which("ssh")
        self.rsync = shutil.which("rsync")
        if not self.ssh or not self.rsync:
            raise RuntimeError("remote CAD builds require ssh and rsync")
        # One multiplexed master connection per host: the status poll loop
        # issues several ssh commands every few seconds, and without
        # ControlMaster each one pays a full TCP+auth handshake.  The socket
        # must live at a SHORT path — sun_path caps at ~104 bytes on macOS
        # and ssh appends a 17-byte temporary suffix while binding — so it
        # goes under ~/.ssh, not under the project state directory.  %C
        # hashes host/port/user, keeping it collision-free.
        control_dir = Path.home() / ".ssh"
        try:
            control_dir.mkdir(mode=0o700, exist_ok=True)
        except OSError:
            pass
        self.ssh_options = [
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control_dir}/lx-cad-%C",
            "-o", "ControlPersist=60",
        ]
        self.ssh_base = [self.ssh, *self.ssh_options, host]

    def command(
        self, command: str, *, check: bool = True, text: bool = True,
        quiet_stderr: bool = False,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*self.ssh_base, "bash -lc " + shlex.quote(command)],
            check=check, text=text,
            stdout=subprocess.PIPE,
            stderr=(subprocess.DEVNULL if quiet_stderr else None),
        )

    def _rsync_transport(self) -> list[str]:
        return ["-e", shlex.join([self.ssh, *self.ssh_options])]

    def upload(self, local: Path, remote_path: str) -> None:
        subprocess.run(
            [self.rsync, "-a", "--partial", *self._rsync_transport(),
             str(local), f"{self.host}:{remote_path}"],
            check=True,
        )

    def download(self, remote_path: str, local: Path, *, required: bool) -> bool:
        local.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [self.rsync, "-a", "--partial", "--append-verify",
             *self._rsync_transport(),
             f"{self.host}:{remote_path}", str(local)],
            stdout=subprocess.DEVNULL,
            stderr=(None if required else subprocess.DEVNULL),
        )
        if required and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        return result.returncode == 0


def _remote_home(remote: Remote) -> str:
    home = remote.command('printf "%s" "$HOME"').stdout
    if not home.startswith("/") or "\n" in home:
        raise RuntimeError(f"invalid remote HOME from {remote.host}: {home!r}")
    return home


def _resolve_remote_root(remote: Remote, configured: str) -> str:
    if configured == "~":
        return _remote_home(remote)
    if configured.startswith("~/"):
        return _remote_home(remote).rstrip("/") + "/" + configured[2:]
    if not configured.startswith("/") or "\n" in configured:
        raise ValueError("LX_CAD_REMOTE_ROOT must be an absolute path or start with ~/")
    return configured.rstrip("/")


def _remote_python(remote: Remote, tool: str, *args: str) -> subprocess.CompletedProcess:
    command = " ".join(shlex.quote(value) for value in ("python3", tool, *args))
    return remote.command(command)


def _bootstrap_tool(
        remote: Remote, remote_root: str, *, local_tool: Path = SCRIPT,
        expected_sha256: str | None = None) -> str:
    digest = _sha256_file(local_tool)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            "bootstrap transport does not match the source snapshot")
    tools_dir = f"{remote_root}/tools"
    tool = f"{tools_dir}/remote_cad-{digest}.py"
    remote.command(f"mkdir -p {shlex.quote(tools_dir)}")
    probe = remote.command(
        f"test -f {shlex.quote(tool)} && sha256sum {shlex.quote(tool)}",
        check=False,
    )
    if probe.returncode or not probe.stdout.startswith(digest):
        temporary = f"{tool}.partial-{os.getpid()}"
        remote.upload(local_tool, temporary)
        remote.command(
            f"test \"$(sha256sum {shlex.quote(temporary)} | cut -d' ' -f1)\" = "
            f"{shlex.quote(digest)} && chmod 0444 {shlex.quote(temporary)} && "
            f"mv {shlex.quote(temporary)} {shlex.quote(tool)}"
        )
    return tool


def _verify_remote_tool(
        remote: Remote, tool: str, expected_sha256: str) -> None:
    result = remote.command(
        f"test -f {shlex.quote(tool)} && sha256sum {shlex.quote(tool)}")
    fields = result.stdout.split()
    if len(fields) < 1 or fields[0] != expected_sha256:
        raise RuntimeError("remote CAD transport hash mismatch")


def _environment_hash(requirements_bytes: bytes | None = None) -> str:
    header = (
        f"protocol={PROTOCOL_VERSION}\n"
        f"attestation={ENVIRONMENT_ATTESTATION_VERSION}\n"
        f"python={REMOTE_PYTHON_VERSION}\n"
        f"platform=linux-x86_64\n"
    ).encode("utf-8")
    if requirements_bytes is None:
        requirements_bytes = LOCK_FILE.read_bytes()
    if not isinstance(requirements_bytes, bytes) or not requirements_bytes:
        raise RuntimeError("remote requirements lock must be non-empty bytes")
    payload = header + requirements_bytes
    return _sha256_bytes(payload)


def _current_cgroup_attestation() -> dict:
    try:
        record = next(
            line for line in Path("/proc/self/cgroup").read_text(
                encoding="utf-8").splitlines()
            if line.startswith("0::"))
        relative = record.split(":", 2)[2].lstrip("/")
        root = Path("/sys/fs/cgroup").resolve()
        directory = (root / relative).resolve()
        if directory != root and root not in directory.parents:
            raise RuntimeError("cgroup path escapes /sys/fs/cgroup")
        memory_max = (directory / "memory.max").read_text(
            encoding="ascii").strip()
        swap_max = (directory / "memory.swap.max").read_text(
            encoding="ascii").strip()
    except (OSError, StopIteration, ValueError) as exc:
        raise RuntimeError("cannot attest remote cgroup-v2 limits") from exc
    return {
        "path": "/" + relative,
        "memory_max_bytes": int(memory_max) if memory_max != "max" else None,
        "memory_swap_max_bytes": int(swap_max) if swap_max != "max" else None,
    }


def _current_cgroup_metrics() -> dict:
    """Read aggregate CPU, memory, process and I/O counters for this job."""
    try:
        record = next(
            line for line in Path("/proc/self/cgroup").read_text(
                encoding="utf-8").splitlines()
            if line.startswith("0::"))
        relative = record.split(":", 2)[2].lstrip("/")
        root = Path("/sys/fs/cgroup").resolve()
        directory = (root / relative).resolve()
        if directory != root and root not in directory.parents:
            raise RuntimeError("cgroup path escapes /sys/fs/cgroup")

        def keyed(name: str) -> dict[str, int]:
            values = {}
            for line in (directory / name).read_text(
                    encoding="ascii").splitlines():
                fields = line.split()
                if len(fields) == 2:
                    values[fields[0]] = int(fields[1])
            return values

        memory_peak = int((directory / "memory.peak").read_text(
            encoding="ascii").strip())
        pids_peak = int((directory / "pids.peak").read_text(
            encoding="ascii").strip())
        cpu = keyed("cpu.stat")
        memory_events = keyed("memory.events")
        io_path = directory / "io.stat"
        io_totals = None
        if io_path.is_file():
            io_totals = {
                key: 0 for key in (
                    "rbytes", "wbytes", "rios", "wios", "dbytes", "dios")
            }
            for line in io_path.read_text(encoding="ascii").splitlines():
                for field in line.split()[1:]:
                    key, separator, value = field.partition("=")
                    if separator and key in io_totals:
                        io_totals[key] += int(value)
    except (OSError, StopIteration, ValueError) as exc:
        raise RuntimeError("cannot measure remote cgroup-v2 work") from exc
    return {
        "memory_peak_bytes": memory_peak,
        "pids_peak": pids_peak,
        "cpu": cpu,
        "memory_events": memory_events,
        "io": io_totals,
    }


def _uv_identity(uv: str) -> dict:
    path = Path(uv).resolve()
    if not path.is_file():
        raise RuntimeError("uv provisioner is not a regular file")
    return {
        "version": subprocess.check_output(
            [uv, "--version"], text=True).strip(),
        "sha256": _sha256_file(path),
    }


def _measure_environment(environment: Path, expected: dict) -> dict:
    python = environment / "bin" / "python"
    script = r'''
import hashlib, importlib.metadata, json, platform, re, sys, sysconfig
from pathlib import Path
executable = Path(sys.executable).resolve()
digest = hashlib.sha256()
with executable.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
package_digest = hashlib.sha256()
package_files = 0
roots = sorted({
    Path(value).resolve() for value in (
        sysconfig.get_path("purelib"), sysconfig.get_path("platlib"))
    if value
})
for root_index, root in enumerate(roots):
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.suffix == ".pyc"
                or "__pycache__" in path.parts):
            continue
        relative = f"{root_index}/{path.relative_to(root).as_posix()}"
        package_digest.update(relative.encode("utf-8") + b"\0")
        package_digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                package_digest.update(chunk)
        package_files += 1
distributions = sorted(
    f"{re.sub(r'[-_.]+', '-', (dist.metadata.get('Name') or '').strip()).lower()}"
    f"=={dist.version}"
    for dist in importlib.metadata.distributions()
)
print(json.dumps({
    "system": platform.system(),
    "machine": platform.machine(),
    "python_version": sys.version,
    "implementation": sys.implementation.name,
    "soabi": sysconfig.get_config_var("SOABI"),
    "platform_tag": sysconfig.get_platform(),
    "python_executable_sha256": digest.hexdigest(),
    "installed_package_tree_sha256": package_digest.hexdigest(),
    "installed_package_file_count": package_files,
    "installed_distributions": distributions,
}, sort_keys=True))
'''
    runtime = json.loads(subprocess.check_output(
        [str(python), "-c", script], text=True))
    measured = {
        **expected,
        "runtime": runtime,
    }
    measured["attestation_sha256"] = _sha256_bytes(
        _canonical_json(measured))
    return measured


def _probe_environment(
        environment: Path, payload: dict, base_expected: dict) -> bool:
    """Cheap cached-environment validation for dispatch time.

    The full content measurement reads and hashes every site-packages file
    (gigabyte-scale for this environment).  The worker still performs that
    full measurement immediately before Make and again after it; at dispatch
    time the cached marker only needs enough scrutiny to catch real cache
    damage: structural drift, marker tampering, a swapped interpreter,
    package (re)installs, and file additions or removals.  Set
    LX_CAD_ENV_REVALIDATE=1 to force the full measurement here as well.
    """
    if any(payload.get(key) != value for key, value in base_expected.items()):
        return False
    consistency = {
        key: value for key, value in payload.items()
        if key not in {"attestation_sha256", "created_utc"}
    }
    if payload.get("attestation_sha256") != _sha256_bytes(
            _canonical_json(consistency)):
        return False
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        return False
    python = environment / "bin" / "python"
    script = r'''
import hashlib, importlib.metadata, json, platform, re, sys, sysconfig
from pathlib import Path
executable = Path(sys.executable).resolve()
digest = hashlib.sha256()
with executable.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
package_files = 0
roots = sorted({
    Path(value).resolve() for value in (
        sysconfig.get_path("purelib"), sysconfig.get_path("platlib"))
    if value
})
for root in roots:
    for path in root.rglob("*"):
        if (not path.is_file() or path.suffix == ".pyc"
                or "__pycache__" in path.parts):
            continue
        package_files += 1
distributions = sorted(
    f"{re.sub(r'[-_.]+', '-', (dist.metadata.get('Name') or '').strip()).lower()}"
    f"=={dist.version}"
    for dist in importlib.metadata.distributions()
)
print(json.dumps({
    "system": platform.system(),
    "machine": platform.machine(),
    "python_version": sys.version,
    "implementation": sys.implementation.name,
    "soabi": sysconfig.get_config_var("SOABI"),
    "platform_tag": sysconfig.get_platform(),
    "python_executable_sha256": digest.hexdigest(),
    "installed_package_file_count": package_files,
    "installed_distributions": distributions,
}, sort_keys=True))
'''
    try:
        probed = json.loads(subprocess.check_output(
            [str(python), "-c", script], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    return all(
        runtime.get(key) == value for key, value in probed.items())


def _expected_environment(
        environment_sha256: str, requirements: Path) -> dict:
    return {
        "attestation_version": ENVIRONMENT_ATTESTATION_VERSION,
        "environment_sha256": environment_sha256,
        "python_version": REMOTE_PYTHON_VERSION,
        "requirements_sha256": _sha256_file(requirements),
        "expected_system": "Linux",
        "expected_machine": "x86_64",
    }


def _validate_targets(targets: list[str]) -> list[str]:
    targets = targets or ["all"]
    if len(targets) != 1:
        raise ValueError(
            "remote CAD accepts exactly one public Make target per job")
    for target in targets:
        if (not TARGET_RE.fullmatch(target) or target.startswith("-")
                or ".." in Path(target).parts
                or target not in REMOTE_MAKE_TARGETS):
            raise ValueError(f"unsafe/unsupported Make target: {target!r}")
    return targets


def _remote_job_count(explicit: int | None) -> int:
    raw = explicit if explicit is not None else os.environ.get(
        "LX_CAD_REMOTE_JOBS", str(DEFAULT_REMOTE_JOBS))
    try:
        jobs = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("LX_CAD_REMOTE_JOBS must be an integer") from exc
    if not 1 <= jobs <= MAX_REMOTE_JOBS:
        raise ValueError(
            f"LX_CAD_REMOTE_JOBS must be between 1 and {MAX_REMOTE_JOBS}")
    return jobs


def _local_job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise ValueError(f"invalid remote CAD job id: {job_id!r}")
    return LOCAL_STATE / "jobs" / job_id


def _new_job_id(source_hash: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{source_hash[:12]}-{secrets.token_hex(3)}"


def _install_source(args: argparse.Namespace) -> int:
    root = Path(args.remote_root)
    archive = Path(args.archive)
    source = root / "sources" / args.expected
    lock_path = root / "locks" / f"source-{args.expected}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if source.exists():
            _verify_source(source, args.expected, allow_extra=False)
            archive.unlink(missing_ok=True)
            return 0
        temporary = root / "sources" / f".{args.expected}.{os.getpid()}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            _extract_regular_archive(archive, temporary)
            _verify_source(temporary, args.expected, allow_extra=False)
            for path in sorted(temporary.rglob("*"), reverse=True):
                os.chmod(path, 0o555 if path.is_dir() else 0o444)
            os.chmod(temporary, 0o555)
            temporary.replace(source)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
            archive.unlink(missing_ok=True)
    return 0


def _prepare_environment(args: argparse.Namespace) -> int:
    root = Path(args.remote_root)
    environment = root / "envs" / args.environment_hash
    lock_path = root / "locks" / f"env-{args.environment_hash}.lock"
    base_expected = _expected_environment(
        args.environment_hash, Path(args.requirements))
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required on the remote CAD host")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = environment / ".lx-cad-environment.json"
        if marker.is_file():
            try:
                payload = _read_json(marker)
                provisioner = payload.get("provisioner")
                if (not isinstance(provisioner, dict)
                        or set(provisioner) != {"version", "sha256"}
                        or not isinstance(provisioner["version"], str)
                        or not re.fullmatch(
                            r"[0-9a-f]{64}", str(provisioner["sha256"]))):
                    raise RuntimeError(
                        "cached environment provisioner record is malformed")
                if getattr(args, "revalidate", False):
                    measured = _measure_environment(
                        environment,
                        {**base_expected, "provisioner": provisioner})
                    cache_valid = all(
                        payload.get(key) == value
                        for key, value in measured.items())
                else:
                    cache_valid = _probe_environment(
                        environment, payload, base_expected)
                if (cache_valid
                        and payload["runtime"]["system"] == "Linux"
                        and payload["runtime"]["machine"] == "x86_64"):
                    print(json.dumps(payload, sort_keys=True))
                    return 0
            except (OSError, subprocess.CalledProcessError, ValueError,
                    KeyError, RuntimeError):
                # A damaged cache entry is not authoritative. The exclusive
                # env lock makes replacement safe with respect to live jobs.
                pass
        temporary = root / "envs" / f".{args.environment_hash}.{os.getpid()}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        if environment.exists():
            shutil.rmtree(environment)
        try:
            subprocess.run(
                [uv, "venv", "--python", REMOTE_PYTHON_VERSION,
                 str(temporary)], check=True,
            )
            python = temporary / "bin" / "python"
            subprocess.run(
                [uv, "pip", "sync", "--python", str(python),
                 args.requirements], check=True,
            )
            expected = {
                **base_expected,
                "provisioner": _uv_identity(uv),
            }
            payload = {
                **_measure_environment(temporary, expected),
                "created_utc": _utc_now(),
            }
            if (payload["runtime"]["system"] != "Linux"
                    or payload["runtime"]["machine"] != "x86_64"):
                raise RuntimeError(
                    "remote CAD environment must be Linux x86_64")
            _atomic_json(temporary / ".lx-cad-environment.json", payload)
            temporary.replace(environment)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _prepare_job(args: argparse.Namespace) -> int:
    root = Path(args.remote_root)
    source = root / "sources" / args.source_hash
    job = root / "jobs" / args.job_id
    incoming = Path(args.metadata)
    metadata = _read_json(incoming)
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported remote CAD job protocol")
    if metadata.get("job_id") != args.job_id:
        raise RuntimeError("remote CAD job metadata id mismatch")
    source_manifest = _verify_source(
        source, args.source_hash, allow_extra=False)
    _validate_transport_provenance(metadata, source_manifest)
    if job.exists():
        raise FileExistsError(f"remote CAD job already exists: {job}")
    temporary = root / "jobs" / f".{args.job_id}.{os.getpid()}.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        work = temporary / "work"
        work.mkdir(parents=True)
        subprocess.run(
            ["cp", "-a", "--reflink=auto", f"{source}/.", str(work)],
            check=True,
        )
        for path in temporary.rglob("*"):
            mode = path.stat().st_mode & 0o777
            os.chmod(path, mode | 0o200 | (0o100 if path.is_dir() else 0))
        shutil.copy2(incoming, temporary / "job.json")
        _atomic_json(temporary / "status.json", {
            "protocol_version": PROTOCOL_VERSION,
            "state": "queued", "updated_utc": _utc_now(),
        })
        temporary.replace(job)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        incoming.unlink(missing_ok=True)
    return 0


def _write_status(job: Path, state: str, **facts: object) -> None:
    _atomic_json(job / "status.json", {
        "protocol_version": PROTOCOL_VERSION,
        "state": state, "updated_utc": _utc_now(), **facts,
    })


def _obiwan_release_artifact_relatives(
        work: Path, state: str) -> set[str]:
    """Return the public files promised by one focused Obi-Wan release target.

    A cache seed may make every Make prerequisite a no-op, so output delivery
    cannot be inferred from files whose bytes changed during this particular
    job.  The hash-bound state manifest is already the release inventory
    authority; use its artifact keys and include the manifest itself.  Native
    BREP staging remains an internal cache and is intentionally not promoted
    as part of the focused printable release contract.
    """
    if state not in STATE_OUTPUT_ROOTS:
        raise ValueError(f"unknown Obi-Wan release state: {state!r}")
    state_root = (
        work / "top_baffle_v2" / _output_prefix(state))
    manifest = state_root / "obiwan_release_manifest.json"
    payload = _read_json(manifest)
    records = payload.get("artifacts")
    if not isinstance(records, dict) or not records:
        raise RuntimeError(
            f"{state} Obi-Wan release manifest has no artifact inventory")
    required = {manifest.relative_to(work).as_posix()}
    seen = set()
    for relative in records:
        if (not isinstance(relative, str) or not relative
                or relative in seen):
            raise RuntimeError(
                f"{state} Obi-Wan release manifest has an invalid artifact key")
        seen.add(relative)
        path = _safe_member_path(state_root, relative)
        required.add(path.relative_to(work).as_posix())
    return required


def _target_required_artifact_relatives(
        work: Path, targets: list[str]) -> set[str]:
    """Exact cached outputs that a public target must always return.

    Complete state/wing roots are handled separately by
    :func:`_full_output_roots`.  This registry covers outputs outside those
    roots and focused state targets whose public result is a manifest-defined
    subset.  Keeping these paths in every artifact bundle makes warm-cache
    execution equivalent to a cold build even when the local output tree is
    absent or stale.
    """
    targets = _validate_targets(list(targets))
    target = targets[0]
    required: set[str] = set()
    if target in {"all", "candidate", "release"}:
        required.update({
            COMMON_ARTIFACT,
            OBIWAN_WING_DESIGN_MAP_ARTIFACT,
            CAPTIVE_MAGNET_CATALOG_ARTIFACT,
        })
    elif target == "obiwan_release":
        required.update({
            OBIWAN_WING_DESIGN_MAP_ARTIFACT,
            CAPTIVE_MAGNET_CATALOG_ARTIFACT,
        })
    elif target == "refresh_captive_magnet_catalog_existing":
        required.add(CAPTIVE_MAGNET_CATALOG_ARTIFACT)

    if target in {
            "obiwan_wings", "obiwan_wing_exports",
            "obiwan_wing_artifacts",
            "check_obiwan_wings"}:
        required.add(OBIWAN_WING_DESIGN_MAP_ARTIFACT)
    elif target == "common":
        required.add(COMMON_ARTIFACT)
    elif target == "obiwan_state_releases":
        required.update(_obiwan_release_artifact_relatives(
            work, "floor_stand"))
        required.update(_obiwan_release_artifact_relatives(
            work, "no_floor_stand"))
    elif target == "floor_obiwan":
        required.update(_obiwan_release_artifact_relatives(
            work, "floor_stand"))
    elif target == "no_floor_obiwan":
        required.update(_obiwan_release_artifact_relatives(
            work, "no_floor_stand"))
    elif target == "no_floor_obiwan_01a":
        base = "top_baffle_v2/build/no_floor_stand"
        stem = "obiwan_optional_lm_keyed_1_of_2_bottom"
        required.update({
            f"{base}/stl/{stem}.stl",
            f"{base}/stl/{stem}.print.json",
            f"{base}/support_blockers/{stem}.support_blocker.stl",
            f"{base}/support_blockers/{stem}.support_blocker.json",
            f"{base}/obiwan_lm_split.step",
            f"{base}/baffle_cable_routing_obiwan.png",
        })
    elif target == "vase_tebm35c10_4_cad":
        base = "top_baffle_v2/build/vase_TEBM35C10-4"
        stem = "vase_TEBM35C10-4"
        for profile in ("stock", "slim"):
            child = f"{base}/{profile}"
            required.update({
                f"{child}/{stem}.brep",
                f"{child}/{stem}.step",
                f"{child}/{stem}.stl",
                f"{child}/{stem}.print.json",
                f"{child}/{stem}.facts.json",
                f"{child}/{stem}.catalog.json",
                f"{child}/{stem}.slicing_profile.json",
                f"{child}/cad_manifest.json",
            })
    return required


def _artifact_paths(job: Path, metadata: dict) -> list[Path]:
    work = job / "work"
    source = _read_json(work / ".lx-cad-source.json")
    source_paths = {record["path"] for record in source["files"]}
    paths: set[Path] = set()
    baffle = work / "top_baffle_v2"
    for prefix in ARTIFACT_SCAN_PREFIXES:
        root = baffle / prefix
        if root.is_dir():
            paths.update(
                path for path in root.rglob("*")
                if (path.is_file() and not path.is_symlink()
                    and path.relative_to(work).as_posix()
                    not in source_paths)
            )
    by_relative = {
        path.relative_to(work).as_posix(): path for path in paths}
    required_relatives = _target_required_artifact_relatives(
        work, metadata["targets"])
    missing_required = sorted(required_relatives - set(by_relative))
    if missing_required:
        raise RuntimeError(
            "remote Make target omitted required artifact(s): "
            + ", ".join(missing_required))
    required_paths = {by_relative[relative]
                      for relative in required_relatives}

    seed_path = job / "cache-seed.json"
    if not seed_path.is_file():
        return sorted(
            paths, key=lambda item: item.relative_to(work).as_posix())
    seed = _read_json(seed_path)
    if seed.get("format_version") != BUILD_CACHE_VERSION:
        raise RuntimeError("unsupported build-cache seed protocol")
    baseline = _validated_tree_records(seed.get("files"))
    full_roots = _full_output_roots(metadata["targets"])
    current_paths = {
        path.relative_to(work).as_posix() for path in paths}
    if metadata["targets"] != ["clean"]:
        missing = []
        for relative in baseline:
            if relative in source_paths or relative in current_paths:
                continue
            parts = Path(relative).parts
            if len(parts) < 2 or parts[0] != "top_baffle_v2":
                continue
            project_relative = Path(*parts[1:])
            under_scan_root = any(
                _is_under_relative_prefix(project_relative, prefix)
                for prefix in ARTIFACT_SCAN_PREFIXES)
            if not under_scan_root:
                continue
            if _logical_output_root(project_relative) in full_roots:
                # A complete-root promotion replaces the directory, so an
                # omitted old member is an intentional, represented deletion.
                continue
            missing.append(relative)
        if missing:
            raise RuntimeError(
                "focused cached build removed generated artifacts without "
                "a representable promotion deletion: " + ", ".join(missing))
    selected = set(required_paths)
    for path in paths:
        relative = path.relative_to(work).as_posix()
        parts = Path(relative).parts
        project_relative = (
            Path(*parts[1:])
            if len(parts) >= 2 and parts[0] == "top_baffle_v2"
            else None)
        if (project_relative is not None
                and _logical_output_root(project_relative) in full_roots):
            selected.add(path)
            continue
        previous = baseline.get(relative)
        if previous is None:
            selected.add(path)
            continue
        stat_result = path.stat()
        if (stat_result.st_size != previous["size"]
                or _sha256_file(path) != previous["sha256"]):
            selected.add(path)
    return sorted(
        selected, key=lambda item: item.relative_to(work).as_posix())


def _create_artifact_bundle(job: Path, metadata: dict) -> None:
    work = job / "work"
    _verify_source(work, metadata["source_sha256"], allow_extra=True)
    profile_path = job / "profile.json"
    if not profile_path.is_file():
        raise RuntimeError("remote job omitted its performance profile")
    paths = _artifact_paths(job, metadata)
    records = [{
        "path": path.relative_to(work).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    } for path in paths]
    temporary = job / f".artifacts.{os.getpid()}.tmp.tar.gz"
    with tarfile.open(temporary, "w:gz", compresslevel=3) as bundle:
        for path in paths:
            bundle.add(path, arcname=path.relative_to(work).as_posix(), recursive=False)
    archive = job / "artifacts.tar.gz"
    temporary.replace(archive)
    _atomic_json(job / "artifacts.json", {
        "protocol_version": PROTOCOL_VERSION,
        "job_id": metadata["job_id"],
        "source_sha256": metadata["source_sha256"],
        "environment_sha256": metadata["environment_sha256"],
        "environment_attestation_sha256": metadata[
            "environment_attestation_sha256"],
        "environment_attestation": metadata["environment_attestation"],
        "transport": metadata["transport"],
        "targets": metadata["targets"],
        "include_candidate_outputs": metadata["include_candidate_outputs"],
        "execution": {
            "memory_profile": metadata["memory_profile"],
            "memory_max_mib": metadata["memory_max_mib"],
            "memory_floor_mib": metadata["memory_floor_mib"],
            "parallel_jobs": metadata["parallel_jobs"],
            "guard_slots": metadata["parallel_jobs"],
            "worker_max_rss_mib": metadata["worker_max_rss_mib"],
            "systemd_unit": metadata["systemd_unit"],
            "cgroup_attestation": metadata["cgroup_attestation"],
        },
        "created_utc": _utc_now(),
        "build_completed_ns": time.time_ns(),
        "performance_profile": {
            "schema_version": PERFORMANCE_PROFILE_VERSION,
            "size": profile_path.stat().st_size,
            "sha256": _sha256_file(profile_path),
        },
        "archive_sha256": _sha256_file(archive),
        "files": records,
    })


def _validated_completed_job_for_cache(job: Path) -> tuple[dict, dict]:
    metadata = _read_json(job / "job.json")
    status = _read_json(job / "status.json")
    if (metadata.get("protocol_version") != PROTOCOL_VERSION
            or status.get("protocol_version") != PROTOCOL_VERSION
            or status.get("state") != "succeeded"
            or status.get("exit_code") != 0
            or (job / "exit_code").read_text(encoding="ascii").strip()
            != "0"):
        raise RuntimeError("only a succeeded remote job may publish a cache")
    if metadata.get("include_candidate_outputs"):
        raise RuntimeError("manifold-only source jobs do not publish a cache")
    work = job / "work"
    _verify_source(work, metadata["source_sha256"], allow_extra=True)
    artifacts_path = job / "artifacts.json"
    artifacts = _read_json(artifacts_path)
    for field in (
            "protocol_version", "job_id", "source_sha256",
            "environment_sha256", "environment_attestation_sha256",
            "targets"):
        expected = (PROTOCOL_VERSION if field == "protocol_version"
                    else metadata.get(field))
        if artifacts.get(field) != expected:
            raise RuntimeError(
                f"cache artifact provenance mismatch: {field}")
    completed_ns = artifacts.get("build_completed_ns")
    if type(completed_ns) is not int or completed_ns <= 0:
        raise RuntimeError("cache artifact completion order is missing")
    archive = job / "artifacts.tar.gz"
    if _sha256_file(archive) != artifacts.get("archive_sha256"):
        raise RuntimeError("cache artifact archive hash mismatch")
    profile_record = artifacts.get("performance_profile")
    profile_path = job / "profile.json"
    if (not isinstance(profile_record, dict)
            or set(profile_record) != {"schema_version", "size", "sha256"}
            or profile_record.get("schema_version")
            != PERFORMANCE_PROFILE_VERSION
            or not profile_path.is_file()
            or profile_path.stat().st_size != profile_record.get("size")
            or _sha256_file(profile_path) != profile_record.get("sha256")):
        raise RuntimeError("cache performance-profile binding is invalid")
    records = artifacts.get("files")
    if not isinstance(records, list):
        raise RuntimeError("cache artifact manifest has no file list")
    seen = set()
    for record in records:
        if (not isinstance(record, dict)
                or set(record) != {"path", "size", "sha256"}
                or not isinstance(record.get("path"), str)
                or type(record.get("size")) is not int
                or record["size"] < 0
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(record.get("sha256", "")))
                or record["path"] in seen):
            raise RuntimeError("invalid cache artifact record")
        seen.add(record["path"])
        path = _safe_member_path(work, record["path"])
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != record["size"]
                or _sha256_file(path) != record["sha256"]):
            raise RuntimeError(
                f"cache artifact/work mismatch: {record['path']}")
    expected_artifacts = {
        path.relative_to(work).as_posix()
        for path in _artifact_paths(job, metadata)}
    if seen != expected_artifacts:
        raise RuntimeError(
            "cache artifact manifest is not the exact generated delta: "
            f"manifest={sorted(seen)}, expected={sorted(expected_artifacts)}")
    return metadata, artifacts


def _publish_build_cache(job: Path) -> bool:
    """Publish this locally accepted job as the next Make cache seed.

    This transition is invoked by the local fetcher only after artifact
    archive verification, promoted-root QA and atomic promotion all succeed.
    The remote job is revalidated before publication.  Completion ordering
    prevents a delayed fetch of an older job from replacing a newer cache.
    For one exact source snapshot, a later focused job retains missing files
    from the previous verified seed so narrow target coverage cannot evict a
    richer cache.  The later worktree always wins overlapping paths.
    """
    metadata, artifacts = _validated_completed_job_for_cache(job)
    remote_root = Path(metadata["remote_root"])
    environment_hash = metadata["environment_sha256"]
    attestation_hash = metadata["environment_attestation_sha256"]
    entry = _build_cache_entry(
        remote_root, environment_hash, attestation_hash)
    with _build_cache_lock(
            remote_root, environment_hash, attestation_hash):
        supplement_work = None
        excluded_prefixes: tuple[Path, ...] = ()
        if entry.is_dir():
            try:
                current, _records = _verify_build_cache_entry(
                    entry, environment_hash, attestation_hash)
            except (OSError, ValueError, RuntimeError):
                shutil.rmtree(entry, ignore_errors=True)
            else:
                if current["build_completed_ns"] >= (
                        artifacts["build_completed_ns"]):
                    return False
                if (current["source_sha256"] == metadata["source_sha256"]
                        and metadata["targets"] != ["clean"]):
                    supplement_work = entry / "work"
                    # A complete-root target represents deletion by replacing
                    # that root during local promotion.  Do not resurrect an
                    # omitted old member while retaining unrelated coverage.
                    excluded_prefixes = tuple(
                        Path("top_baffle_v2") / _output_prefix(root)
                        for root in sorted(
                            _full_output_roots(metadata["targets"])))

        parent = entry.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".{entry.name}.{os.getpid()}.tmp"
        previous = parent / f".{entry.name}.{os.getpid()}.previous"
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(previous, ignore_errors=True)
        try:
            temporary.mkdir()
            _copy_tree(job / "work", temporary / "work")
            retained = ()
            if supplement_work is not None:
                retained = _copy_missing_tree_files(
                    supplement_work, temporary / "work",
                    excluded_prefixes=excluded_prefixes)
            records = _tree_records(temporary / "work")
            marker = {
                "format_version": BUILD_CACHE_VERSION,
                "environment_sha256": environment_hash,
                "environment_attestation_sha256": attestation_hash,
                "source_sha256": metadata["source_sha256"],
                "published_from_job": metadata["job_id"],
                "build_completed_ns": artifacts["build_completed_ns"],
                "artifact_manifest_sha256": _sha256_file(
                    job / "artifacts.json"),
                "created_utc": _utc_now(),
                "files": records,
            }
            if supplement_work is not None:
                marker["coverage_union"] = {
                    "exact_source": True,
                    "retained_file_count": len(retained),
                }
            _atomic_json(temporary / "cache.json", marker)
            _verify_build_cache_entry(
                temporary, environment_hash, attestation_hash)
            if entry.exists():
                entry.replace(previous)
            try:
                temporary.replace(entry)
            except BaseException:
                if previous.exists():
                    previous.replace(entry)
                raise
            shutil.rmtree(previous, ignore_errors=True)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(previous, ignore_errors=True)
    return True


def _publish_cache_command(args: argparse.Namespace) -> int:
    published = _publish_build_cache(Path(args.job_dir))
    print("published" if published else "retained-newer-cache")
    return 0


GC_DEFAULT_RETAIN_DAYS = 7.0
GC_FAILED_RETAIN_DAYS = 1.0


def _gc_retain_days() -> float:
    raw = os.environ.get("LX_CAD_GC_DAYS")
    if raw is None:
        return GC_DEFAULT_RETAIN_DAYS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid LX_CAD_GC_DAYS: {raw!r}") from exc
    if value < 1.0:
        raise ValueError("LX_CAD_GC_DAYS must be at least 1")
    return value


def _rmtree_force(path: Path) -> None:
    """Remove a tree even when members are read-only (0444/0555 sources)."""
    def _onerror(func, target, _exc_info):
        try:
            os.chmod(target, 0o700)
            parent = os.path.dirname(target)
            if parent:
                os.chmod(parent, 0o700)
            func(target)
        except OSError:
            pass
    shutil.rmtree(path, onerror=_onerror)


def _tree_bytes(path: Path) -> int:
    total = 0
    for member in path.rglob("*"):
        try:
            if member.is_file() and not member.is_symlink():
                total += member.stat().st_size
        except OSError:
            continue
    return total


def _gc_local_jobs(*, retain_days: float | None = None) -> dict:
    """Prune local job history under .remote-cad/jobs.

    Successful jobs keep their small metadata and logs indefinitely but drop
    the large artifacts archive once older than the retention window.
    Failed or interrupted jobs are removed entirely after one day.  Jobs
    with no recorded exit code are removed after the retention window — a
    genuinely live detached job refreshes its local status well within it.
    """
    retain = _gc_retain_days() if retain_days is None else retain_days
    now = time.time()
    summary = {"removed_job_dirs": 0, "removed_bytes": 0}
    jobs_root = LOCAL_STATE / "jobs"
    if not jobs_root.is_dir():
        return summary
    for job in sorted(jobs_root.iterdir()):
        if not job.is_dir() or job.is_symlink():
            continue
        try:
            age_days = (now - job.stat().st_mtime) / 86400.0
            exit_path = job / "exit_code"
            if exit_path.is_file():
                code = exit_path.read_text(
                    encoding="ascii", errors="replace").strip()
                if code == "0":
                    archive = job / "artifacts.tar.gz"
                    if age_days > retain and archive.is_file():
                        summary["removed_bytes"] += archive.stat().st_size
                        archive.unlink()
                elif age_days > GC_FAILED_RETAIN_DAYS:
                    summary["removed_bytes"] += _tree_bytes(job)
                    _rmtree_force(job)
                    summary["removed_job_dirs"] += 1
            elif age_days > retain:
                summary["removed_bytes"] += _tree_bytes(job)
                _rmtree_force(job)
                summary["removed_job_dirs"] += 1
        except OSError:
            continue
    return summary


def _gc_remote_state(args: argparse.Namespace) -> int:
    """Prune remote job, source, environment, and incoming state.

    Runs on the remote host under a non-blocking exclusive build lock so it
    can never race a live build; when a build holds the lock the collection
    is skipped and reported.  The Make cache entries themselves are kept —
    they are the warm-build value and there is one per environment identity —
    but their crash-leftover .tmp/.previous siblings are collected.
    """
    root = Path(args.remote_root)
    retain = float(args.retain_days)
    now = time.time()
    summary = {
        "removed_jobs": 0, "removed_sources": 0,
        "removed_environments": 0, "removed_incoming": 0,
        "removed_cache_debris": 0,
    }
    lock_path = root / "locks" / "build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(json.dumps({"skipped": "build lock busy"}, sort_keys=True))
            return 0
        try:
            keep_sources: set[str] = set()
            keep_environments: set[str] = set()
            jobs_root = root / "jobs"
            if jobs_root.is_dir():
                for job in sorted(jobs_root.iterdir()):
                    if not job.is_dir() or job.is_symlink():
                        continue
                    terminal = False
                    try:
                        status = _read_json(job / "status.json")
                        terminal = status.get("state") in {
                            "succeeded", "failed", "canceled"}
                    except (OSError, ValueError, RuntimeError):
                        terminal = (job / "exit_code").is_file()
                    try:
                        age_days = (now - job.stat().st_mtime) / 86400.0
                    except OSError:
                        continue
                    if terminal and age_days > retain:
                        _rmtree_force(job)
                        summary["removed_jobs"] += 1
                        continue
                    try:
                        metadata = _read_json(job / "job.json")
                    except (OSError, ValueError, RuntimeError):
                        continue
                    source = metadata.get("source_sha256")
                    if isinstance(source, str):
                        keep_sources.add(source)
                    env_hash = metadata.get("environment_sha256")
                    if isinstance(env_hash, str):
                        keep_environments.add(env_hash)
            cache_root = root / "cache"
            if cache_root.is_dir():
                for entry in sorted(cache_root.iterdir()):
                    name = entry.name
                    if name.startswith("."):
                        try:
                            age_days = (
                                now - entry.stat().st_mtime) / 86400.0
                        except OSError:
                            continue
                        if age_days > GC_FAILED_RETAIN_DAYS:
                            _rmtree_force(entry)
                            summary["removed_cache_debris"] += 1
                        continue
                    if "-" in name:
                        keep_environments.add(name.split("-", 1)[0])
                    marker = entry / "cache.json"
                    try:
                        payload = _read_json(marker)
                    except (OSError, ValueError, RuntimeError):
                        continue
                    source = payload.get("source_sha256")
                    if isinstance(source, str):
                        keep_sources.add(source)
                    env_hash = payload.get("environment_sha256")
                    if isinstance(env_hash, str):
                        keep_environments.add(env_hash)
            sources_root = root / "sources"
            if sources_root.is_dir():
                for source_dir in sorted(sources_root.iterdir()):
                    if not source_dir.is_dir() or source_dir.is_symlink():
                        continue
                    if source_dir.name in keep_sources:
                        continue
                    try:
                        age_days = (
                            now - source_dir.stat().st_mtime) / 86400.0
                    except OSError:
                        continue
                    if source_dir.name.startswith(".") or age_days > retain:
                        _rmtree_force(source_dir)
                        summary["removed_sources"] += 1
            envs_root = root / "envs"
            if envs_root.is_dir():
                for env_dir in sorted(envs_root.iterdir()):
                    if not env_dir.is_dir() or env_dir.is_symlink():
                        continue
                    if env_dir.name in keep_environments:
                        continue
                    try:
                        age_days = (now - env_dir.stat().st_mtime) / 86400.0
                    except OSError:
                        continue
                    if env_dir.name.startswith(".") or age_days > retain:
                        _rmtree_force(env_dir)
                        summary["removed_environments"] += 1
            incoming_root = root / "incoming"
            if incoming_root.is_dir():
                for item in sorted(incoming_root.iterdir()):
                    try:
                        age_days = (now - item.stat().st_mtime) / 86400.0
                    except OSError:
                        continue
                    if age_days > GC_FAILED_RETAIN_DAYS:
                        if item.is_dir() and not item.is_symlink():
                            _rmtree_force(item)
                        else:
                            item.unlink(missing_ok=True)
                        summary["removed_incoming"] += 1
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _maybe_collect_garbage(remote: Remote, metadata: dict) -> None:
    """Best-effort post-success retention pass; never fails the build."""
    if os.environ.get("LX_CAD_GC_DISABLE") == "1":
        return
    try:
        retain = _gc_retain_days()
        local_summary = _gc_local_jobs(retain_days=retain)
        result = _remote_python(
            remote, _job_executor_path(metadata), "_gc-remote",
            "--remote-root", metadata["remote_root"],
            "--retain-days", str(retain),
        )
        remote_summary = result.stdout.strip().splitlines()
        print(
            "Retention: local "
            f"{local_summary['removed_job_dirs']} job dirs / "
            f"{local_summary['removed_bytes'] / 1e6:.0f} MB removed; "
            f"remote {remote_summary[-1] if remote_summary else 'n/a'}")
    except (OSError, subprocess.CalledProcessError, RuntimeError,
            ValueError) as exc:
        print(f"Warning: retention pass skipped: {exc}", file=sys.stderr)


def _gc_command(args: argparse.Namespace) -> int:
    retain = (
        float(args.retain_days) if args.retain_days is not None
        else _gc_retain_days())
    local_summary = _gc_local_jobs(retain_days=retain)
    print(json.dumps({"local": local_summary}, sort_keys=True))
    if args.local_only:
        return 0
    host = os.environ.get("LX_CAD_REMOTE_HOST", DEFAULT_HOST)
    configured_root = os.environ.get(
        "LX_CAD_REMOTE_ROOT", DEFAULT_REMOTE_ROOT)
    remote = Remote(host)
    remote_root = _resolve_remote_root(remote, configured_root)
    jobs_root = LOCAL_STATE / "jobs"
    tool = None
    if jobs_root.is_dir():
        for job in sorted(jobs_root.iterdir(), reverse=True):
            metadata_path = job / "job.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = _read_json(metadata_path)
            except (OSError, ValueError, RuntimeError):
                continue
            if metadata.get("remote_root") == remote_root:
                tool = _job_executor_path(metadata)
                break
    if tool is None:
        print("No local job references this remote root; remote GC skipped.")
        return 0
    result = _remote_python(
        remote, tool, "_gc-remote",
        "--remote-root", remote_root, "--retain-days", str(retain),
    )
    print(result.stdout.strip())
    return 0


def _read_guard_profile_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid guard profile event at line {line_number}") from exc
        if (not isinstance(event, dict)
                or event.get("schema_version") != 1
                or not isinstance(event.get("command"), list)
                or not isinstance(event.get("wall_seconds"), (int, float))
                or not isinstance(
                    event.get("peak_process_tree_rss_mib"), (int, float))
                or type(event.get("exit_code")) is not int):
            raise RuntimeError(
                f"malformed guard profile event at line {line_number}")
        events.append(event)
    return events


def _guard_profile_label(event: dict) -> str:
    command = [str(value) for value in event.get("command", ())]
    script = next(
        (Path(value).name for value in command if value.endswith(".py")),
        Path(command[0]).name if command else "unknown",
    )
    context = event.get("context")
    if isinstance(context, dict):
        selector = (
            context.get("LX_R6F_CASE_ID")
            or context.get("LX_CLEARANCE_SINGLE_CHECK")
            or context.get("LX_OBIWAN_WING_SINGLE_CHECK"))
        if selector:
            slug = context.get("LX_OBIWAN_WING_LIVE_SLUG")
            if slug:
                return f"{script}:{selector}:slug={slug}"
            return f"{script}:{selector}"
        dense_state = context.get("LX_OBIWAN_CLOSURE_DENSE_STATE")
        dense_shard = context.get("LX_OBIWAN_CLOSURE_DENSE_SHARD")
        if dense_state is not None or dense_shard is not None:
            return (
                f"{script}:state={dense_state or '?'}:"
                f"shard={dense_shard or '?'}")
        state = context.get("LX_STAND_FOOT")
        profile = context.get("LX_ROUTING_PROFILE")
        if state is not None or profile is not None:
            return f"{script}:foot={state or '?'}:route={profile or '?'}"
    return script


def _make_trace_targets(log_path: Path) -> list[str]:
    if not log_path.is_file():
        return []
    pattern = re.compile(
        r"^[^\n:]+:\d+: (?:update target|target) '([^']+)'",
        re.MULTILINE,
    )
    return pattern.findall(log_path.read_text(
        encoding="utf-8", errors="replace"))


def _stage_phase_profile_events(log_path: Path) -> list[dict]:
    """Read structured child-phase timings emitted by staged CAD exporters."""
    if not log_path.is_file():
        return []
    prefix = "[obiwan-stage-profile] "
    events = []
    for line_number, line in enumerate(log_path.read_text(
            encoding="utf-8", errors="replace").splitlines(), start=1):
        marker = line.find(prefix)
        if marker < 0:
            continue
        try:
            event = json.loads(line[marker + len(prefix):])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid stage phase profile at log line {line_number}"
            ) from exc
        if (not isinstance(event, dict)
                or event.get("schema_version") != 1
                or not isinstance(event.get("label"), str)
                or not event["label"]
                or not isinstance(event.get("wall_seconds"), (int, float))
                or isinstance(event.get("wall_seconds"), bool)
                or float(event["wall_seconds"]) < 0.0
                or type(event.get("exit_code")) is not int
                or type(event.get("stand_foot")) is not bool):
            raise RuntimeError(
                f"malformed stage phase profile at log line {line_number}")
        events.append(event)
    return events


def _write_performance_profile(
    job: Path,
    metadata: dict,
    *,
    executor_started_ns: int,
    make_started_ns: int,
    make_completed_ns: int,
    make_exit_code: int,
    make_command: list[str],
    cache_profile: dict,
    cgroup_before_make: dict,
) -> None:
    """Aggregate per-recipe guard samples and whole-cgroup measurements."""
    events_path = job / "profile-events.jsonl"
    events = _read_guard_profile_events(events_path)
    stage_phase_events = _stage_phase_profile_events(job / "build.log")
    groups: dict[str, dict] = {}
    for event in events:
        label = _guard_profile_label(event)
        group = groups.setdefault(label, {
            "count": 0,
            "wall_seconds": 0.0,
            "child_cpu_seconds": 0.0,
            "peak_process_tree_rss_mib": 0.0,
            "nonzero_exit_count": 0,
        })
        group["count"] += 1
        group["wall_seconds"] += float(event["wall_seconds"])
        group["child_cpu_seconds"] += (
            float(event.get("child_user_cpu_seconds", 0.0))
            + float(event.get("child_system_cpu_seconds", 0.0)))
        group["peak_process_tree_rss_mib"] = max(
            group["peak_process_tree_rss_mib"],
            float(event["peak_process_tree_rss_mib"]))
        group["nonzero_exit_count"] += int(event["exit_code"] != 0)

    top_events = []
    for event in sorted(
            events, key=lambda item: float(item["wall_seconds"]),
            reverse=True)[:40]:
        top_events.append({
            "label": _guard_profile_label(event),
            "wall_seconds": float(event["wall_seconds"]),
            "child_cpu_seconds": (
                float(event.get("child_user_cpu_seconds", 0.0))
                + float(event.get("child_system_cpu_seconds", 0.0))),
            "peak_process_tree_rss_mib": float(
                event["peak_process_tree_rss_mib"]),
            "exit_code": event["exit_code"],
            "command": event["command"],
            "context": event.get("context", {}),
        })

    stage_phase_groups: dict[str, dict] = {}
    for event in stage_phase_events:
        state = "floor" if event["stand_foot"] else "no_floor"
        label = f"{state}:{event['label']}"
        group = stage_phase_groups.setdefault(label, {
            "count": 0,
            "wall_seconds": 0.0,
            "max_wall_seconds": 0.0,
            "nonzero_exit_count": 0,
        })
        wall_seconds = float(event["wall_seconds"])
        group["count"] += 1
        group["wall_seconds"] += wall_seconds
        group["max_wall_seconds"] = max(
            group["max_wall_seconds"], wall_seconds)
        group["nonzero_exit_count"] += int(event["exit_code"] != 0)

    slowest_stage_phases = [{
        "label": event["label"],
        "state": "floor" if event["stand_foot"] else "no_floor",
        "wall_seconds": float(event["wall_seconds"]),
        "exit_code": event["exit_code"],
    } for event in sorted(
        stage_phase_events,
        key=lambda item: float(item["wall_seconds"]),
        reverse=True,
    )[:80]]

    trace_targets = _make_trace_targets(job / "build.log")
    make_wall = max(0.0, (make_completed_ns - make_started_ns) / 1.0e9)
    cgroup_after_make = _current_cgroup_metrics()
    cpu_delta = {
        key: max(
            0,
            int(cgroup_after_make["cpu"].get(key, 0))
            - int(cgroup_before_make["cpu"].get(key, 0)),
        )
        for key in set(cgroup_before_make["cpu"]) | set(
            cgroup_after_make["cpu"])
    }
    memory_event_delta = {
        key: max(
            0,
            int(cgroup_after_make["memory_events"].get(key, 0))
            - int(cgroup_before_make["memory_events"].get(key, 0)),
        )
        for key in set(cgroup_before_make["memory_events"]) | set(
            cgroup_after_make["memory_events"])
    }
    io_delta = None
    if (cgroup_before_make.get("io") is not None
            and cgroup_after_make.get("io") is not None):
        io_delta = {
            key: max(
                0,
                int(cgroup_after_make["io"].get(key, 0))
                - int(cgroup_before_make["io"].get(key, 0)),
            )
            for key in set(cgroup_before_make["io"]) | set(
                cgroup_after_make["io"])
        }
    cpu_seconds = float(cpu_delta.get("usage_usec", 0)) / 1.0e6
    guard_wall = sum(float(event["wall_seconds"]) for event in events)
    profile = {
        "schema_version": PERFORMANCE_PROFILE_VERSION,
        "job_id": metadata["job_id"],
        "source_sha256": metadata["source_sha256"],
        "targets": metadata["targets"],
        "parallel_jobs": metadata["parallel_jobs"],
        "cache": cache_profile,
        "executor_started_epoch_ns": executor_started_ns,
        "make_started_epoch_ns": make_started_ns,
        "make_completed_epoch_ns": make_completed_ns,
        "make_wall_seconds": make_wall,
        "executor_to_make_completion_seconds": max(
            0.0, (make_completed_ns - executor_started_ns) / 1.0e9),
        "make_exit_code": make_exit_code,
        "make_command": make_command,
        "cgroup": {
            "before_make": cgroup_before_make,
            "after_make": cgroup_after_make,
            "make_delta": {
                "cpu": cpu_delta,
                "memory_events": memory_event_delta,
                "io": io_delta,
            },
        },
        "derived": {
            "aggregate_cpu_seconds": cpu_seconds,
            "average_cpu_cores_during_make": (
                cpu_seconds / make_wall if make_wall > 0.0 else 0.0),
            "guarded_recipe_count": len(events),
            "guarded_recipe_wall_seconds_sum": guard_wall,
            "guarded_recipe_parallelism_equivalent": (
                guard_wall / make_wall if make_wall > 0.0 else 0.0),
            "max_guarded_recipe_rss_mib": max(
                (float(event["peak_process_tree_rss_mib"])
                 for event in events), default=0.0),
            "make_trace_update_count": len(trace_targets),
            "make_trace_unique_target_count": len(set(trace_targets)),
            "stage_phase_count": len(stage_phase_events),
            "stage_phase_wall_seconds_sum": sum(
                float(event["wall_seconds"])
                for event in stage_phase_events),
        },
        "make_trace_targets": trace_targets,
        "guard_groups": dict(sorted(
            groups.items(),
            key=lambda item: item[1]["wall_seconds"],
            reverse=True)),
        "slowest_guarded_recipes": top_events,
        "stage_phase_groups": dict(sorted(
            stage_phase_groups.items(),
            key=lambda item: item[1]["wall_seconds"],
            reverse=True)),
        "slowest_stage_phases": slowest_stage_phases,
        "guard_event_log": {
            "path": events_path.name,
            "size": events_path.stat().st_size if events_path.is_file() else 0,
            "sha256": (
                _sha256_file(events_path) if events_path.is_file() else None),
        },
        "created_utc": _utc_now(),
    }
    _atomic_json(job / "profile.json", profile)


def _execute_job(args: argparse.Namespace) -> int:
    job = Path(args.job_dir)
    exit_code = 99
    log_path = job / "build.log"
    executor_started_ns = time.time_ns()
    try:
        metadata = _read_json(job / "job.json")
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("unsupported remote CAD job protocol")
        if (job / "cancel.request.json").is_file():
            with _remote_job_transition_lock(job):
                status = _read_json(job / "status.json")
                if status.get("state") in {
                        "succeeded", "failed", "canceled"}:
                    recorded = status.get("exit_code", 130)
                    return recorded if type(recorded) is int else 99
                else:
                    (job / "exit_code").write_text(
                        "130\n", encoding="ascii")
                    _write_status(job, "canceled", exit_code=130)
            return 130
        remote_root = Path(metadata["remote_root"])
        environment = remote_root / "envs" / metadata["environment_sha256"]
        python = environment / "bin" / "python"
        make = shutil.which("make")
        if not make:
            raise RuntimeError("remote Make environment is incomplete")
        cgroup = _current_cgroup_attestation()
        if (metadata["systemd_unit"] not in cgroup["path"]
                or cgroup["memory_max_bytes"] != (
                    REMOTE_MEMORY_MAX_MIB * 1024 * 1024)
                or cgroup["memory_swap_max_bytes"] != 0):
            raise RuntimeError("remote job lacks the required systemd cgroup")
        metadata["cgroup_attestation"] = cgroup
        _write_status(job, "waiting_for_global_lock", pid=os.getpid())
        build_lock_path = remote_root / "locks" / "build.lock"
        environment_lock_path = (
            remote_root / "locks"
            / f"env-{metadata['environment_sha256']}.lock"
        )
        build_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with build_lock_path.open("a+b") as build_lock, \
                environment_lock_path.open("a+b") as environment_lock, \
                log_path.open(
            "a", encoding="utf-8", buffering=1,
        ) as log:
            fcntl.flock(build_lock, fcntl.LOCK_EX)
            # Environment preparation/replacement takes this same lock
            # exclusively.  Keep a shared claim through Make and packaging.
            fcntl.flock(environment_lock, fcntl.LOCK_SH)
            if not python.is_file():
                raise RuntimeError("remote Python environment is incomplete")
            # Cache seeding is an optimization, never an input authority.  A
            # verified source-only job tree is still present until the seed
            # swap completes atomically, so any copy/filesystem failure can
            # safely fall back to the cold build.
            try:
                cache_seed = _seed_build_cache(job, metadata)
            except (OSError, ValueError, RuntimeError) as exc:
                cache_seed = None
                print(
                    f"make-cache=unavailable cold-fallback reason={exc}",
                    file=log,
                )
            if cache_seed is None:
                print("make-cache=cold", file=log)
                cache_profile = {
                    "state": "cold",
                    "seed_file_count": 0,
                    "changed_source_count": None,
                    "changed_source_paths": [],
                }
            else:
                seed_manifest = _read_json(job / "cache-seed.json")
                cache_profile = {
                    "state": "verified",
                    "source_sha256": cache_seed["source_sha256"],
                    "build_completed_ns": cache_seed[
                        "build_completed_ns"],
                    "seed_file_count": len(seed_manifest["files"]),
                    "changed_source_count": len(
                        cache_seed["changed_source_paths"]),
                    "changed_source_paths": list(
                        cache_seed["changed_source_paths"]),
                }
                print(
                    "make-cache=verified "
                    f"source={cache_seed['source_sha256']} "
                    f"changed_sources="
                    f"{len(cache_seed['changed_source_paths'])}",
                    file=log,
                )
            source_manifest = _verify_source(
                job / "work", metadata["source_sha256"], allow_extra=True)
            _validate_transport_provenance(metadata, source_manifest)
            marker = _read_json(
                environment / ".lx-cad-environment.json")
            expected = _expected_environment(
                metadata["environment_sha256"],
                job / "work" / "top_baffle_v2" / LOCK_FILE.name,
            )
            attestation = metadata.get("environment_attestation")
            if not isinstance(attestation, dict):
                raise RuntimeError(
                    "remote Python environment attestation is missing")
            provisioner = attestation.get("provisioner")
            if (not isinstance(provisioner, dict)
                    or set(provisioner) != {"version", "sha256"}):
                raise RuntimeError(
                    "remote Python environment provisioner is malformed")
            measured = _measure_environment(
                environment, {**expected, "provisioner": provisioner})
            attestation_measurement = {
                key: value for key, value in attestation.items()
                if key != "created_utc"
            }
            marker_drift = sorted(
                key for key in set(marker) | set(attestation)
                if marker.get(key) != attestation.get(key))
            measurement_drift = sorted(
                key for key in set(measured) | set(attestation_measurement)
                if measured.get(key) != attestation_measurement.get(key))
            hash_drift = (
                measured.get("attestation_sha256") != metadata.get(
                    "environment_attestation_sha256"))
            if marker_drift or measurement_drift or hash_drift:
                raise RuntimeError(
                    "remote Python environment attestation drifted: "
                    f"marker={marker_drift}, measured={measurement_drift}, "
                    f"hash={hash_drift}")
            _write_status(job, "running", pid=os.getpid())
            print(f"[{_utc_now()}] job={metadata['job_id']}", file=log)
            print(f"source={metadata['source_sha256']}", file=log)
            print(f"targets={' '.join(metadata['targets'])}", file=log)
            print(
                f"memory=systemd:{REMOTE_MEMORY_MAX_SYSTEMD} "
                f"workers:{metadata['parallel_jobs']} "
                f"guard-per-worker:{metadata['worker_max_rss_mib']}MiB "
                f"floor:{REMOTE_MEMORY_FLOOR_MIB}MiB",
                file=log,
            )
            env = os.environ.copy()
            for name in (
                "LX_CAD_MEMORY_GUARDED", "LX_CAD_MEMORY_GUARD_PID",
                "LX_CLEARANCE_SINGLE_CHECK", "LX_R6F_CASE_ID",
                "LX_OBIWAN_WING_SINGLE_CHECK", "LX_OBIWAN_WING_LIVE_SLUG",
            ):
                env.pop(name, None)
            cache = job / "cache"
            temporary = job / "tmp"
            cache.mkdir(exist_ok=True)
            temporary.mkdir(exist_ok=True)
            env.update({
                "LX_CAD_EXECUTION": "remote-worker",
                # Bind generated release metadata to the exact immutable
                # source snapshot that this worker is executing.  Catalog
                # producers must not infer provenance from a mutable checkout
                # after artifacts have been promoted back to the Mac.
                "LX_CAD_SOURCE_SHA256": metadata["source_sha256"],
                "LX_CAD_ALLOW_PARALLEL": "1",
                "LX_CAD_GUARD_SLOTS": str(metadata["parallel_jobs"]),
                "LX_CAD_MEMORY_PROFILE": REMOTE_MEMORY_PROFILE,
                "LX_CAD_MAX_RSS_MB": str(metadata["worker_max_rss_mib"]),
                "LX_CAD_MIN_FREE_MB": str(REMOTE_MEMORY_FLOOR_MIB),
                "LX_CAD_PROFILE_EVENTS": str(
                    (job / "profile-events.jsonl").resolve()),
                "MAKEFLAGS": "",
                "MPLCONFIGDIR": str(cache / "matplotlib"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": str(temporary),
                "XDG_CACHE_HOME": str(cache / "xdg"),
                "PATH": str(environment / "bin") + os.pathsep + env.get("PATH", ""),
            })
            command = [
                make, "--no-print-directory", "--trace", "-j",
                str(metadata["parallel_jobs"]),
                "LX_CAD_EXECUTION=remote-worker",
                f"PYTHON={python}", "--", *metadata["targets"],
            ]
            print("command=" + shlex.join(command), file=log)
            cgroup_before_make = _current_cgroup_metrics()
            make_started_ns = time.time_ns()
            result = subprocess.run(
                command, cwd=job / "work" / "top_baffle_v2", env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
            make_completed_ns = time.time_ns()
            exit_code = int(result.returncode)
            _write_performance_profile(
                job,
                metadata,
                executor_started_ns=executor_started_ns,
                make_started_ns=make_started_ns,
                make_completed_ns=make_completed_ns,
                make_exit_code=exit_code,
                make_command=command,
                cache_profile=cache_profile,
                cgroup_before_make=cgroup_before_make,
            )
            if exit_code == 0:
                final_measurement = _measure_environment(
                    environment, {**expected, "provisioner": provisioner})
                final_drift = sorted(
                    key for key in (
                        set(final_measurement) | set(attestation_measurement))
                    if final_measurement.get(key)
                    != attestation_measurement.get(key)
                )
                if final_drift:
                    raise RuntimeError(
                        "remote Python environment changed during Make: "
                        f"{final_drift}")
                _create_artifact_bundle(job, metadata)
            print(f"[{_utc_now()}] exit_code={exit_code}", file=log)
    except BaseException:
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        exit_code = 99
    with _remote_job_transition_lock(job):
        status = _read_json(job / "status.json")
        if status.get("state") in {"succeeded", "failed", "canceled"}:
            recorded = status.get("exit_code", exit_code)
            exit_code = recorded if type(recorded) is int else 99
        else:
            canceled = (job / "cancel.request.json").is_file()
            if canceled:
                exit_code = 130
            (job / "exit_code").write_text(
                f"{exit_code}\n", encoding="ascii")
            _write_status(
                job, ("canceled" if canceled
                      else "succeeded" if exit_code == 0 else "failed"),
                exit_code=exit_code,
            )
    return exit_code


def _load_local_job(job_id: str) -> tuple[Path, dict]:
    directory = _local_job_dir(job_id)
    metadata = _read_json(directory / "job.json")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported remote CAD job protocol")
    if metadata.get("job_id") != job_id:
        raise RuntimeError("local remote-CAD job metadata mismatch")
    _job_executor_path(metadata)
    return directory, metadata


def _remote_status(remote: Remote, metadata: dict) -> dict:
    path = f"{metadata['remote_root']}/jobs/{metadata['job_id']}/status.json"
    result = remote.command(
        f"cat {shlex.quote(path)}", quiet_stderr=True,
    )
    payload = json.loads(result.stdout)
    if (not isinstance(payload, dict)
            or payload.get("protocol_version") != PROTOCOL_VERSION):
        raise RuntimeError("invalid remote CAD status")
    return payload


def _remote_unit_facts(remote: Remote, metadata: dict) -> dict | None:
    unit = metadata.get("systemd_unit")
    if not unit:
        return None
    result = remote.command(
        "systemctl --user show " + shlex.quote(unit)
        + " --no-pager --property=LoadState --property=ActiveState"
        + " --property=SubState --property=Result",
        check=False, quiet_stderr=True,
    )
    if result.returncode and not result.stdout.strip():
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout)
    facts = {}
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            facts[name] = value
    if "LoadState" not in facts or "ActiveState" not in facts:
        raise RuntimeError("remote systemd unit returned malformed state")
    return {
        "load_state": facts["LoadState"],
        "active_state": facts["ActiveState"],
        "sub_state": facts.get("SubState", ""),
        "result": facts.get("Result", ""),
    }


def _job_executor_path(metadata: dict) -> str:
    transport = metadata.get("transport")
    if (not isinstance(transport, dict)
            or set(transport) != {"source_path", "sha256", "size"}
            or transport.get("source_path") not in TRANSPORT_SOURCE_PATHS
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(transport.get("sha256", "")))
            or type(transport.get("size")) is not int
            or transport["size"] < 1):
        raise RuntimeError("remote CAD job transport provenance is malformed")
    return (
        f"{metadata['remote_root']}/sources/{metadata['source_sha256']}"
        f"/{transport['source_path']}"
    )


def _remote_launch_was_accepted(remote: Remote, metadata: dict) -> bool:
    marker = (
        f"{metadata['remote_root']}/jobs/{metadata['job_id']}"
        "/launch.accepted"
    )
    result = remote.command(
        f"test -f {shlex.quote(marker)}", check=False, quiet_stderr=True)
    if result.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return result.returncode == 0


def _launch_remote_job(remote: Remote, metadata: dict) -> None:
    """Launch through the snapshot tool's serialized remote transition."""
    executor = _job_executor_path(metadata)
    _verify_remote_tool(remote, executor, metadata["transport"]["sha256"])
    job_dir = f"{metadata['remote_root']}/jobs/{metadata['job_id']}"
    _remote_python(
        remote, executor, "_launch-job", "--job-dir", job_dir)


def _reconcile_dead_remote_job(
        remote: Remote, metadata: dict, facts: dict) -> None:
    reason = (
        "remote systemd unit stopped before terminal status: "
        f"load={facts['load_state']}, active={facts['active_state']}, "
        f"sub={facts['sub_state']}, result={facts['result']}"
    )
    executor = _job_executor_path(metadata)
    _verify_remote_tool(remote, executor, metadata["transport"]["sha256"])
    _remote_python(
        remote, executor, "_mark-failed",
        "--job-dir",
        f"{metadata['remote_root']}/jobs/{metadata['job_id']}",
        "--exit-code", "99", "--reason", reason,
    )


def _stream_log(remote: Remote, metadata: dict, offset: int) -> int:
    path = f"{metadata['remote_root']}/jobs/{metadata['job_id']}/build.log"
    command = (
        f"if test -f {shlex.quote(path)}; then "
        f"tail -c +{offset + 1} {shlex.quote(path)}; fi"
    )
    result = remote.command(command, text=False, quiet_stderr=True)
    data = result.stdout
    if data:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    return offset + len(data)


def _download_job_file(
    remote: Remote, metadata: dict, local_dir: Path, name: str,
    *, required: bool,
) -> bool:
    source = f"{metadata['remote_root']}/jobs/{metadata['job_id']}/{name}"
    destination = local_dir / name
    temporary = destination.with_name(destination.name + ".partial")
    present = remote.download(source, temporary, required=required)
    if present:
        temporary.replace(destination)
    return present


def _verify_performance_profile(local_dir: Path, metadata: dict) -> dict:
    profile = _read_json(local_dir / "profile.json")
    if (profile.get("schema_version") != PERFORMANCE_PROFILE_VERSION
            or profile.get("job_id") != metadata["job_id"]
            or profile.get("source_sha256") != metadata["source_sha256"]
            or profile.get("targets") != metadata["targets"]
            or type(profile.get("make_exit_code")) is not int):
        raise RuntimeError("remote CAD performance profile is malformed")
    return profile


def _verify_and_extract_artifacts(local_dir: Path, metadata: dict) -> Path:
    manifest = _read_json(local_dir / "artifacts.json")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported remote CAD artifact protocol")
    for field in (
            "job_id", "source_sha256", "environment_sha256",
            "environment_attestation_sha256", "environment_attestation",
            "transport", "targets", "include_candidate_outputs"):
        if manifest.get(field) != metadata.get(field):
            raise RuntimeError(f"artifact provenance mismatch: {field}")
    expected_execution = {
        "memory_profile": metadata["memory_profile"],
        "memory_max_mib": metadata["memory_max_mib"],
        "memory_floor_mib": metadata["memory_floor_mib"],
        "parallel_jobs": metadata["parallel_jobs"],
        "guard_slots": metadata["parallel_jobs"],
        "worker_max_rss_mib": metadata["worker_max_rss_mib"],
        "systemd_unit": metadata["systemd_unit"],
    }
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("artifact execution-policy record is malformed")
    cgroup = execution.get("cgroup_attestation")
    without_cgroup = {
        key: value for key, value in execution.items()
        if key != "cgroup_attestation"
    }
    if without_cgroup != expected_execution:
        raise RuntimeError("artifact execution-policy provenance mismatch")
    if (not isinstance(cgroup, dict)
            or set(cgroup) != {
                "path", "memory_max_bytes", "memory_swap_max_bytes"}
            or metadata["systemd_unit"] not in str(cgroup.get("path"))
            or cgroup.get("memory_max_bytes") != (
                REMOTE_MEMORY_MAX_MIB * 1024 * 1024)
            or cgroup.get("memory_swap_max_bytes") != 0):
        raise RuntimeError("artifact cgroup attestation is invalid")
    profile_record = manifest.get("performance_profile")
    profile_path = local_dir / "profile.json"
    if (not isinstance(profile_record, dict)
            or set(profile_record) != {"schema_version", "size", "sha256"}
            or profile_record.get("schema_version")
            != PERFORMANCE_PROFILE_VERSION
            or not profile_path.is_file()
            or profile_path.stat().st_size != profile_record.get("size")
            or _sha256_file(profile_path) != profile_record.get("sha256")):
        raise RuntimeError("performance profile hash/provenance mismatch")
    archive = local_dir / "artifacts.tar.gz"
    if _sha256_file(archive) != manifest.get("archive_sha256"):
        raise RuntimeError("downloaded artifact archive hash mismatch")
    incoming = Path(tempfile.mkdtemp(prefix="incoming-", dir=local_dir))
    try:
        _extract_regular_archive(archive, incoming)
        records = manifest.get("files")
        if not isinstance(records, list):
            raise RuntimeError("artifact manifest has no file list")
        expected = {}
        for record in records:
            if (not isinstance(record, dict)
                    or set(record) != {"path", "size", "sha256"}
                    or not isinstance(record["path"], str)
                    or type(record["size"]) is not int or record["size"] < 0
                    or not isinstance(record["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                    or record["path"] in expected):
                raise RuntimeError("invalid/duplicate artifact-manifest record")
            expected[record["path"]] = record
        actual = {
            path.relative_to(incoming).as_posix(): path
            for path in incoming.rglob("*") if path.is_file()
        }
        if set(actual) != set(expected):
            raise RuntimeError("artifact archive content differs from manifest")
        for relative, path in actual.items():
            record = expected[relative]
            if (path.stat().st_size != record.get("size")
                    or _sha256_file(path) != record.get("sha256")):
                raise RuntimeError(f"artifact hash mismatch: {relative}")
        return incoming
    except BaseException:
        shutil.rmtree(incoming, ignore_errors=True)
        raise


def _full_output_roots(targets: list[str]) -> set[str]:
    full = set()
    if any(target in {
            "all", "candidate", "release", "obiwan_release",
    } for target in targets):
        full.update(("floor_stand", "no_floor_stand", "wings"))
    full.update(target for target in targets if target in {"floor_stand", "no_floor_stand"})
    if any(target in {
            "obiwan_wings", "obiwan_wing_exports",
            "obiwan_wing_artifacts",
            "check_obiwan_wings",
    } for target in targets):
        full.add("wings")
    if "vase_tebm35c10_4_cad" in targets:
        full.add("tebm35c10_4")
    return full


def _current_source_identity(metadata: dict) -> str:
    return _source_identity(_source_records(
        include_candidate_outputs=bool(
            metadata.get("include_candidate_outputs"))))


@contextmanager
def _local_fetch_and_promotion_lock():
    """Exclude concurrent waiters/resumes while the parent fetches."""
    LOCAL_STATE.mkdir(parents=True, exist_ok=True)
    fetch_path = LOCAL_STATE / "fetch.lock"
    handle = fetch_path.open("a+b")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "another fetch/promotion is already active") from exc
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


@contextmanager
def _local_promotion_lock():
    """Lock owned by the actual promoter, surviving a fetching-parent death."""
    LOCAL_STATE.mkdir(parents=True, exist_ok=True)
    path = LOCAL_STATE / "promotion.lock"
    handle = path.open("a+b")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _check_promoted_roots(full_roots: set[str]) -> None:
    if not full_roots:
        return

    def make_check(roots: list[str]) -> None:
        if not roots:
            return
        # Promotion itself is already one authenticated local-macos guard
        # process capped at 8 GiB aggregate RSS.  A fresh stamp root forces a
        # complete post-download sweep while GNU Make's inherited jobserver
        # fans only the pure STL diagnostics; local OCC remains serial-only.
        with tempfile.TemporaryDirectory(
                prefix="lx-promoted-manifold-") as stamp_text:
            subprocess.run(
                ["make", "--no-print-directory",
                 f"-j{LOCAL_PROMOTION_JOBS}",
                 "LX_CAD_EXECUTION=local-manifold",
                 f"PYTHON={sys.executable}",
                 "_manifold_parallel",
                 "MANIFOLD_ROOTS=" + " ".join(roots),
                 f"MANIFOLD_STAMP_DIR={stamp_text}"],
                cwd=BAFFLE_DIR, check=True)

    state_roots = sorted(full_roots & set(STATE_OUTPUT_ROOTS))
    make_check([
        (_output_prefix(state) / "stl").as_posix()
        for state in state_roots
    ])
    if "wings" in full_roots:
        make_check([
            (_output_prefix("wings") / slug / "stl").as_posix()
            for slug in ("ac", "ae")
        ])


class _PromotionInterrupted(RuntimeError):
    """Turn process-control signals into rollback-safe promotion failures."""


@contextmanager
def _promotion_signal_guard():
    previous = {}

    def interrupted(signum, _frame):
        name = signal.Signals(signum).name
        raise _PromotionInterrupted(
            f"local artifact promotion interrupted by {name}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _promotion_path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_promoted_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _promotion_journal_path(backup_root: Path) -> Path:
    return backup_root / "transaction.json"


def _load_promotion_transaction(
        backup_root: Path, metadata: dict) -> tuple[dict, list[dict]]:
    journal_path = _promotion_journal_path(backup_root)
    if not journal_path.is_file() or journal_path.is_symlink():
        raise RuntimeError(
            f"unexplained stale promotion backup preserved: {backup_root}")
    journal = _read_json(journal_path)
    if (set(journal) != {
            "format_version", "job_id", "source_sha256", "phase", "entries"}
            or journal.get("format_version") != PROMOTION_TRANSACTION_VERSION
            or journal.get("job_id") != metadata.get("job_id")
            or journal.get("source_sha256") != metadata.get("source_sha256")
            or journal.get("phase") not in {"active", "committed"}
            or not isinstance(journal.get("entries"), list)):
        raise RuntimeError(
            f"invalid stale promotion journal preserved: {journal_path}")

    entries = []
    destinations: set[str] = set()
    temporaries: set[str] = set()
    for record in journal["entries"]:
        if (not isinstance(record, dict)
                or set(record) != {"relative", "had_original", "temporary"}
                or not isinstance(record.get("relative"), str)
                or type(record.get("had_original")) is not bool
                or (record.get("temporary") is not None
                    and not isinstance(record.get("temporary"), str))):
            raise RuntimeError(
                f"invalid stale promotion journal preserved: {journal_path}")
        relative = Path(record["relative"])
        if (relative.as_posix() != record["relative"]
                or record["relative"] in destinations):
            raise RuntimeError(
                f"invalid stale promotion journal preserved: {journal_path}")
        destination = _safe_member_path(REPO_ROOT, record["relative"])
        destinations.add(record["relative"])
        temporary = None
        if record["temporary"] is not None:
            temporary_path = Path(record["temporary"])
            if (temporary_path.as_posix() != record["temporary"]
                    or record["temporary"] in temporaries
                    or record["temporary"] in destinations):
                raise RuntimeError(
                    f"invalid stale promotion journal preserved: {journal_path}")
            temporary = _safe_member_path(REPO_ROOT, record["temporary"])
            temporaries.add(record["temporary"])
        entries.append({
            **record,
            "destination": destination,
            "saved": _safe_member_path(backup_root, record["relative"]),
            "temporary_path": temporary,
        })
    if destinations & temporaries:
        raise RuntimeError(
            f"invalid stale promotion journal preserved: {journal_path}")

    explained_roots = [
        entry["saved"] for entry in entries if entry["had_original"]]
    for path in backup_root.rglob("*"):
        if path == journal_path:
            continue
        if not any(
                path == explained or path in explained.parents
                or explained in path.parents
                for explained in explained_roots):
            raise RuntimeError(
                f"unexplained stale promotion payload preserved: {path}")
    return journal, entries


def _discard_promotion_transaction(backup_root: Path, metadata: dict) -> None:
    journal, entries = _load_promotion_transaction(backup_root, metadata)
    if journal["phase"] != "committed":
        raise RuntimeError("refusing to discard an active promotion transaction")
    for entry in reversed(entries):
        _remove_promoted_path(entry["saved"])
    _finish_promotion_transaction(backup_root)


def _finish_promotion_transaction(backup_root: Path) -> None:
    journal_path = _promotion_journal_path(backup_root)
    remaining = [
        path for path in backup_root.rglob("*")
        if (path != journal_path and (path.is_file() or path.is_symlink()))
    ]
    if remaining:
        raise RuntimeError(
            f"promotion transaction left unexplained data: {remaining[0]}")
    resolved = backup_root.with_name(
        f".{backup_root.name}.{os.getpid()}.{secrets.token_hex(4)}.resolved")
    backup_root.replace(resolved)
    shutil.rmtree(resolved)


def _recover_promotion_transaction(backup_root: Path, metadata: dict) -> None:
    journal, entries = _load_promotion_transaction(backup_root, metadata)
    if journal["phase"] == "committed":
        _discard_promotion_transaction(backup_root, metadata)
        return

    for entry in reversed(entries):
        temporary = entry["temporary_path"]
        if temporary is not None:
            _remove_promoted_path(temporary)
        destination = entry["destination"]
        saved = entry["saved"]
        if entry["had_original"]:
            if _promotion_path_exists(saved):
                _remove_promoted_path(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                saved.replace(destination)
            elif not _promotion_path_exists(destination):
                raise RuntimeError(
                    "promotion recovery cannot find either the original or "
                    f"its backup: {entry['relative']}")
        else:
            if _promotion_path_exists(saved):
                raise RuntimeError(
                    "promotion backup contradicts its journal: "
                    f"{entry['relative']}")
            _remove_promoted_path(destination)

    _finish_promotion_transaction(backup_root)


def _begin_promotion_transaction(
        backup_root: Path, metadata: dict,
        destinations: list[tuple[Path, Path, Path | None]]) -> list[dict]:
    if backup_root.exists():
        _recover_promotion_transaction(backup_root, metadata)

    records = []
    seen: set[str] = set()
    for destination, relative, temporary in destinations:
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise RuntimeError(
                f"duplicate promotion destination: {relative_text}")
        seen.add(relative_text)
        if destination.is_symlink():
            raise RuntimeError(
                f"refusing to promote through a symlink: {destination}")
        temporary_text = None
        if temporary is not None:
            temporary_text = temporary.relative_to(REPO_ROOT).as_posix()
            if _promotion_path_exists(temporary):
                raise RuntimeError(
                    f"unexplained stale promotion temporary preserved: "
                    f"{temporary}")
        records.append({
            "relative": relative_text,
            "had_original": _promotion_path_exists(destination),
            "temporary": temporary_text,
        })

    backup_root.parent.mkdir(parents=True, exist_ok=True)
    preparing = backup_root.with_name(
        f".{backup_root.name}.{os.getpid()}.{secrets.token_hex(4)}.preparing")
    preparing.mkdir()
    transaction_visible = False
    try:
        _atomic_json(_promotion_journal_path(preparing), {
            "format_version": PROMOTION_TRANSACTION_VERSION,
            "job_id": metadata["job_id"],
            "source_sha256": metadata["source_sha256"],
            "phase": "active",
            "entries": records,
        })
        transaction_visible = True
        preparing.replace(backup_root)

        _journal, entries = _load_promotion_transaction(backup_root, metadata)
        for entry in entries:
            destination = entry["destination"]
            if entry["had_original"]:
                if not _promotion_path_exists(destination):
                    raise RuntimeError(
                        f"promotion destination disappeared: "
                        f"{entry['relative']}")
                saved = entry["saved"]
                saved.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(saved)
            elif _promotion_path_exists(destination):
                raise RuntimeError(
                    f"promotion destination appeared: {entry['relative']}")
        return entries
    except BaseException:
        if transaction_visible and backup_root.exists():
            _recover_promotion_transaction(backup_root, metadata)
        elif preparing.exists():
            shutil.rmtree(preparing)
        raise


def _commit_promotion_transaction(backup_root: Path, metadata: dict) -> None:
    journal, _entries = _load_promotion_transaction(backup_root, metadata)
    journal["phase"] = "committed"
    _atomic_json(_promotion_journal_path(backup_root), journal)
    _discard_promotion_transaction(backup_root, metadata)


def _recover_outstanding_promotion_transaction() -> None:
    """Recover the sole prior transaction before any new job mutates roots."""
    backups = LOCAL_STATE / "backups"
    if not backups.exists():
        return
    if not backups.is_dir() or backups.is_symlink():
        raise RuntimeError(
            f"invalid promotion-backup root preserved: {backups}")
    entries = sorted(backups.iterdir(), key=lambda path: path.name)
    if not entries:
        return
    if len(entries) != 1:
        raise RuntimeError(
            "multiple outstanding promotion transactions preserved; "
            "refusing to guess recovery order")
    backup_root = entries[0]
    if (backup_root.is_symlink() or not backup_root.is_dir()
            or not JOB_ID_RE.fullmatch(backup_root.name)):
        raise RuntimeError(
            f"unexplained promotion backup preserved: {backup_root}")
    job_metadata = LOCAL_STATE / "jobs" / backup_root.name / "job.json"
    if not job_metadata.is_file() or job_metadata.is_symlink():
        raise RuntimeError(
            f"promotion recovery metadata is missing: {job_metadata}")
    metadata = _read_json(job_metadata)
    if (metadata.get("protocol_version") != PROTOCOL_VERSION
            or metadata.get("job_id") != backup_root.name):
        raise RuntimeError(
            f"promotion recovery metadata is invalid: {job_metadata}")
    _recover_promotion_transaction(backup_root, metadata)


def _promote_artifacts(incoming: Path, metadata: dict) -> int:
    targets = metadata["targets"]
    backup_root = LOCAL_STATE / "backups" / metadata["job_id"]
    if backup_root.exists():
        _recover_promotion_transaction(backup_root, metadata)
    if _current_source_identity(metadata) != metadata["source_sha256"]:
        raise RuntimeError("local CAD sources changed before promotion")
    full_roots = _full_output_roots(targets)
    incoming_root = incoming / "top_baffle_v2"
    for root_name in sorted(full_roots):
        source = incoming_root / _output_prefix(root_name)
        if not source.is_dir():
            raise RuntimeError(f"complete remote target omitted {root_name}")

    extras = []
    for source in sorted(incoming.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(incoming)
        if len(relative.parts) >= 2 and relative.parts[0] == "top_baffle_v2":
            project_relative = Path(*relative.parts[1:])
        else:
            project_relative = None
        if (project_relative is not None
                and _logical_output_root(project_relative) in full_roots):
            continue
        extras.append((source, REPO_ROOT / relative, relative))

    promoted = 0

    if targets == ["clean"]:
        clean_paths = [
            *(BAFFLE_DIR / _output_prefix(name)
              for name in OUTPUT_ROOT_PREFIXES),
            BAFFLE_DIR / "__pycache__",
            BAFFLE_DIR / "build/common/attachments.step",
            BAFFLE_DIR / "build/common/obiwan_wing_design_map.png",
        ]
        destinations = [
            (destination, destination.relative_to(REPO_ROOT), None)
            for destination in clean_paths
        ]
    else:
        destinations = [
            (BAFFLE_DIR / _output_prefix(root_name),
             (BAFFLE_DIR / _output_prefix(root_name)).relative_to(REPO_ROOT),
             None)
            for root_name in sorted(full_roots)
        ]
        transaction_tag = hashlib.sha256(
            metadata["job_id"].encode("utf-8")).hexdigest()[:16]
        for _source, destination, relative in extras:
            temporary = destination.with_name(
                f".{destination.name}.{transaction_tag}.remote")
            destinations.append((destination, relative, temporary))

    entries = _begin_promotion_transaction(
        backup_root, metadata, destinations)
    entry_by_relative = {entry["relative"]: entry for entry in entries}

    try:
        if targets != ["clean"]:
            for root_name in sorted(full_roots):
                source = incoming_root / _output_prefix(root_name)
                destination = BAFFLE_DIR / _output_prefix(root_name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                promoted += sum(
                    1 for path in destination.rglob("*") if path.is_file())
            for source, destination, relative in extras:
                destination.parent.mkdir(parents=True, exist_ok=True)
                entry = entry_by_relative[relative.as_posix()]
                temporary = entry["temporary_path"]
                if temporary is None:
                    raise RuntimeError("promotion journal omitted a temporary")
                try:
                    shutil.copy2(source, temporary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                promoted += 1

            _check_promoted_roots(full_roots)
        if _current_source_identity(metadata) != metadata["source_sha256"]:
            raise RuntimeError(
                "local CAD sources changed during promotion; rolled back")
    except BaseException:
        if backup_root.exists():
            _recover_promotion_transaction(backup_root, metadata)
        raise
    else:
        _commit_promotion_transaction(backup_root, metadata)
        return promoted


def _promote_local(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).resolve()
    incoming = Path(args.incoming).resolve()
    result = Path(args.result).resolve()
    if job_dir not in incoming.parents or job_dir not in result.parents:
        raise RuntimeError("local promotion paths escape the job directory")
    metadata = _read_json(job_dir / "job.json")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported local promotion protocol")
    with _local_promotion_lock():
        with _promotion_signal_guard():
            _recover_outstanding_promotion_transaction()
            promoted = _promote_artifacts(incoming, metadata)
    _atomic_json(result, {"promoted_files": promoted})
    return 0


def _local_checker_python() -> Path:
    candidates = (REPO_ROOT / ".venv" / "bin" / "python", Path(sys.executable))
    seen = set()
    for candidate in candidates:
        candidate = candidate.absolute()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        result = subprocess.run(
            [str(candidate), "-c", "from PIL import Image"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return candidate
    raise RuntimeError(
        "no local Python with Pillow is available for promoted-artifact QA")


def _run_guarded_local_promotion(
        incoming: Path, local_dir: Path) -> int:
    result = local_dir / f"promotion-result-{os.getpid()}.json"
    result.unlink(missing_ok=True)
    env = os.environ.copy()
    env["LX_CAD_MEMORY_PROFILE"] = "local-macos"
    env["LX_CAD_GUARD_SLOTS"] = "1"
    for name in ("LX_CAD_MEMORY_GUARDED", "LX_CAD_MEMORY_GUARD_PID"):
        env.pop(name, None)
    checker_python = _local_checker_python()
    command = [
        str(checker_python),
        str(BAFFLE_DIR / "scripts/run_memory_guarded.py"), "--",
        str(checker_python), str(SCRIPT), "_promote-local",
        "--incoming", str(incoming), "--job-dir", str(local_dir),
        "--result", str(result),
    ]
    try:
        subprocess.run(command, cwd=BAFFLE_DIR, env=env, check=True)
        payload = _read_json(result)
        promoted = payload.get("promoted_files")
        if type(promoted) is not int or promoted < 0:
            raise RuntimeError("local promotion returned an invalid result")
        return promoted
    finally:
        result.unlink(missing_ok=True)


def _fetch(remote: Remote, metadata: dict) -> int:
    with _local_fetch_and_promotion_lock():
        local_dir = _local_job_dir(metadata["job_id"])
        _download_job_file(
            remote, metadata, local_dir, "status.json", required=True)
        _download_job_file(
            remote, metadata, local_dir, "exit_code", required=True)
        _download_job_file(
            remote, metadata, local_dir, "build.log", required=False)
        profile_present = _download_job_file(
            remote, metadata, local_dir, "profile.json", required=False)
        _download_job_file(
            remote, metadata, local_dir, "profile-events.jsonl",
            required=False)
        exit_code = int((local_dir / "exit_code").read_text(
            encoding="ascii").strip())
        status = _read_json(local_dir / "status.json")
        status_exit = status.get("exit_code")
        if (status.get("protocol_version") != PROTOCOL_VERSION
                or status.get("state") != "succeeded"
                or type(status_exit) is not int or status_exit != 0
                or exit_code != 0):
            effective_exit = (
                exit_code if exit_code else
                status_exit if type(status_exit) is int and status_exit else
                99
            )
            print(
                f"Remote CAD job {metadata['job_id']} failed "
                f"(status {status.get('state')!r}, exit {effective_exit}).")
            print(f"Log: {local_dir / 'build.log'}")
            if profile_present:
                print(f"Profile: {local_dir / 'profile.json'}")
            return effective_exit
        if _current_source_identity(metadata) != metadata["source_sha256"]:
            raise RuntimeError(
                "local CAD sources changed while the remote job ran; "
                "refusing to promote stale artifacts")
        _download_job_file(
            remote, metadata, local_dir, "artifacts.json", required=True)
        _download_job_file(
            remote, metadata, local_dir, "artifacts.tar.gz", required=True)
        if not profile_present:
            raise RuntimeError(
                "succeeded remote CAD job omitted its performance profile")
        profile = _verify_performance_profile(local_dir, metadata)
        if profile["make_exit_code"] != 0:
            raise RuntimeError(
                "succeeded remote CAD profile records a failed Make")
        incoming = _verify_and_extract_artifacts(local_dir, metadata)
        try:
            if _current_source_identity(metadata) != metadata["source_sha256"]:
                raise RuntimeError(
                    "local CAD sources changed during artifact transfer; "
                    "refusing promotion")
            promoted = _run_guarded_local_promotion(incoming, local_dir)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
        # A cache is performance-only and becomes eligible only after the
        # downloaded archive, promoted-root QA and atomic local promotion all
        # succeeded.  Failure to publish cannot invalidate already-promoted
        # artifacts; the next remote job simply starts cold or from the last
        # older verified entry.
        try:
            publish_result = _remote_python(
                remote, _job_executor_path(metadata), "_publish-cache",
                "--job-dir",
                f"{metadata['remote_root']}/jobs/{metadata['job_id']}",
            )
            cache_state = publish_result.stdout.strip().splitlines()
            if cache_state:
                print(f"Remote Make cache: {cache_state[-1]}.")
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            print(
                f"Warning: verified artifacts were promoted, but the remote "
                f"Make cache was not updated: {exc}",
                file=sys.stderr,
            )
    print(
        f"Remote CAD job {metadata['job_id']} succeeded; "
        f"verified/promoted {promoted} artifact files."
    )
    print(f"Performance profile: {local_dir / 'profile.json'}")
    _maybe_collect_garbage(remote, metadata)
    return 0


def _wait_and_fetch(remote: Remote, metadata: dict) -> int:
    last_state = None
    offset = 0
    failures = 0
    try:
        while True:
            try:
                status = _remote_status(remote, metadata)
                failures = 0
                offset = _stream_log(remote, metadata, offset)
            except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                failures += 1
                if failures >= 12:
                    raise RuntimeError(
                        "remote status unavailable; the systemd job may still be running"
                    ) from exc
                time.sleep(5)
                continue
            state = status.get("state")
            if state != last_state:
                print(f"[{_utc_now()}] remote CAD state: {state}", flush=True)
                last_state = state
            if state in {"succeeded", "failed", "canceled"}:
                offset = _stream_log(remote, metadata, offset)
                break
            facts = _remote_unit_facts(remote, metadata)
            if facts is not None and not (
                    facts["load_state"] != "not-found"
                    and facts["active_state"] in {
                        "active", "activating", "reloading", "deactivating",
                    }):
                if state == "queued" and facts["load_state"] == "not-found":
                    if _remote_launch_was_accepted(remote, metadata):
                        accepted_facts = {
                            **facts, "result": "accepted-before-worker-status",
                        }
                        _reconcile_dead_remote_job(
                            remote, metadata, accepted_facts)
                    else:
                        _launch_remote_job(remote, metadata)
                else:
                    _reconcile_dead_remote_job(remote, metadata, facts)
                continue
            time.sleep(float(os.environ.get("LX_CAD_REMOTE_POLL_SECONDS", "5")))
    except KeyboardInterrupt:
        print(
            f"\nRemote job continues. Resume with: "
            f"python3 scripts/remote_cad.py resume {metadata['job_id']}",
            file=sys.stderr,
        )
        return 130
    return _fetch(remote, metadata)


def _run_remote(args: argparse.Namespace) -> int:
    targets = _validate_targets(args.targets)
    include_candidate_outputs = targets == ["manifold"]
    parallel_jobs = _remote_job_count(args.jobs)
    worker_max_rss_mib = (
        REMOTE_MEMORY_MAX_MIB - REMOTE_MEMORY_FLOOR_MIB) // parallel_jobs
    host = args.host or os.environ.get("LX_CAD_REMOTE_HOST", DEFAULT_HOST)
    configured_root = args.remote_root or os.environ.get(
        "LX_CAD_REMOTE_ROOT", DEFAULT_REMOTE_ROOT,
    )
    remote = Remote(host)
    remote_root = _resolve_remote_root(remote, configured_root)
    with tempfile.TemporaryDirectory(prefix="lx-cad-source-") as temporary_text:
        archive, source_manifest = _create_source_archive(
            Path(temporary_text),
            include_candidate_outputs=include_candidate_outputs)
        local_tool, transport = _extract_snapshot_transport(
            archive, Path(temporary_text), source_manifest)
        requirements_bytes = _extract_snapshot_requirements(
            archive, source_manifest)
        source_hash = source_manifest["source_sha256"]
        environment_hash = _environment_hash(requirements_bytes)
        job_id = _new_job_id(source_hash)
        unit = "lx-cad-" + re.sub(r"[^A-Za-z0-9-]", "-", job_id.lower())
        local_dir = _local_job_dir(job_id)
        local_dir.mkdir(parents=True)
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job_id,
            "created_utc": _utc_now(),
            "host": host,
            "remote_root": remote_root,
            "source_sha256": source_hash,
            "environment_sha256": environment_hash,
            "python_version": REMOTE_PYTHON_VERSION,
            "memory_profile": REMOTE_MEMORY_PROFILE,
            "memory_max_mib": REMOTE_MEMORY_MAX_MIB,
            "memory_floor_mib": REMOTE_MEMORY_FLOOR_MIB,
            "parallel_jobs": parallel_jobs,
            "worker_max_rss_mib": worker_max_rss_mib,
            "systemd_unit": unit,
            "targets": targets,
            "include_candidate_outputs": include_candidate_outputs,
            "transport": transport,
        }
        _atomic_json(local_dir / "source.json", source_manifest)
        print(f"Remote CAD job: {job_id}", flush=True)
        print(f"Source snapshot: {source_hash}", flush=True)
        tool = _bootstrap_tool(
            remote, remote_root, local_tool=local_tool,
            expected_sha256=transport["sha256"])
        for directory in ("incoming", "sources", "jobs", "envs", "locks", "cache"):
            remote.command(f"mkdir -p {shlex.quote(remote_root + '/' + directory)}")
        remote_archive = f"{remote_root}/incoming/{archive.name}"
        # A content-addressed source tree that already exists remotely does
        # not need the archive again; _install-source re-verifies the cached
        # tree either way and fails the job on any mismatch.
        cached_source = remote.command(
            "test -d "
            + shlex.quote(f"{remote_root}/sources/{source_hash}")
            + " && echo cached || echo missing").stdout.strip()
        if cached_source != "cached":
            remote.upload(archive, remote_archive)
        _remote_python(
            remote, tool, "_install-source", "--remote-root", remote_root,
            "--archive", remote_archive, "--expected", source_hash,
        )
        source_root = f"{remote_root}/sources/{source_hash}"
        tool = f"{source_root}/{transport['source_path']}"
        _verify_remote_tool(remote, tool, transport["sha256"])
        requirements = f"{source_root}/top_baffle_v2/{LOCK_FILE.name}"
        environment_result = _remote_python(
            remote, tool, "_prepare-environment", "--remote-root", remote_root,
            "--environment-hash", environment_hash,
            "--requirements", requirements,
            *(["--revalidate"]
              if os.environ.get("LX_CAD_ENV_REVALIDATE") == "1" else []),
        )
        attestation_lines = [
            line for line in environment_result.stdout.splitlines()
            if line.startswith("{")
        ]
        if not attestation_lines:
            raise RuntimeError("remote environment returned no attestation")
        attestation = json.loads(attestation_lines[-1])
        if (attestation.get("environment_sha256") != environment_hash
                or attestation.get("expected_system") != "Linux"
                or attestation.get("expected_machine") != "x86_64"
                or attestation.get("runtime", {}).get("system") != "Linux"
                or attestation.get("runtime", {}).get("machine") != "x86_64"):
            raise RuntimeError("remote environment attestation is incompatible")
        attestation_hash = attestation.get("attestation_sha256")
        attestation_payload = {
            key: value for key, value in attestation.items()
            if key not in {"attestation_sha256", "created_utc"}
        }
        if attestation_hash != _sha256_bytes(
                _canonical_json(attestation_payload)):
            raise RuntimeError("remote environment attestation hash mismatch")
        metadata["environment_attestation"] = attestation
        metadata["environment_attestation_sha256"] = attestation_hash
        _atomic_json(local_dir / "job.json", metadata)
        incoming_metadata = f"{remote_root}/incoming/{job_id}.json"
        remote.upload(local_dir / "job.json", incoming_metadata)
        _remote_python(
            remote, tool, "_prepare-job", "--remote-root", remote_root,
            "--source-hash", source_hash, "--job-id", job_id,
            "--metadata", incoming_metadata,
        )
        _launch_remote_job(remote, metadata)
    if args.detach:
        print(
            f"Detached. Resume with: "
            f"python3 scripts/remote_cad.py resume {job_id}")
        return 0
    return _wait_and_fetch(remote, metadata)


def _resume(args: argparse.Namespace) -> int:
    _directory, metadata = _load_local_job(args.job_id)
    remote = Remote(metadata["host"])
    return _wait_and_fetch(remote, metadata)


def _status(args: argparse.Namespace) -> int:
    local_dir, metadata = _load_local_job(args.job_id)
    remote = Remote(metadata["host"])
    status = _remote_status(remote, metadata)
    print(json.dumps(status, indent=2, sort_keys=True))
    path = f"{metadata['remote_root']}/jobs/{args.job_id}/build.log"
    result = remote.command(
        f"if test -f {shlex.quote(path)}; then tail -n 25 {shlex.quote(path)}; fi",
        quiet_stderr=True,
    )
    if result.stdout:
        print("\nRecent log:\n" + result.stdout, end="")
    _atomic_json(local_dir / "status.json", status)
    return 0


def _mark_failed(args: argparse.Namespace) -> int:
    """Reconcile a dead service that could not write its terminal status."""
    job = Path(args.job_dir)
    with _remote_job_transition_lock(job):
        status = _read_json(job / "status.json")
        if status.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("unsupported remote CAD status protocol")
        if status.get("state") in {"succeeded", "failed", "canceled"}:
            return 0
        exit_code = int(args.exit_code)
        (job / "exit_code").write_text(f"{exit_code}\n", encoding="ascii")
        _write_status(
            job, "failed", exit_code=exit_code,
            failure_reason=args.reason,
        )
    return 0


@contextmanager
def _remote_job_transition_lock(job: Path):
    """Serialize launch/cancel decisions for one prepared remote job."""
    handle = (job / "transition.lock").open("a+b")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _local_systemd_unit_facts(unit: str) -> dict:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "--no-pager",
         "--property=LoadState", "--property=ActiveState",
         "--property=SubState", "--property=Result"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode and not result.stdout.strip():
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout)
    raw = {}
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            raw[name] = value
    if "LoadState" not in raw or "ActiveState" not in raw:
        raise RuntimeError("local systemd returned malformed unit state")
    return {
        "load_state": raw["LoadState"],
        "active_state": raw["ActiveState"],
        "sub_state": raw.get("SubState", ""),
        "result": raw.get("Result", ""),
    }


def _validated_remote_job(job: Path) -> tuple[dict, dict]:
    metadata = _read_json(job / "job.json")
    if (metadata.get("protocol_version") != PROTOCOL_VERSION
            or metadata.get("job_id") != job.name):
        raise RuntimeError("invalid prepared remote CAD job metadata")
    expected_job = (
        Path(metadata["remote_root"]) / "jobs" / metadata["job_id"]
    ).resolve()
    if job != expected_job:
        raise RuntimeError("remote CAD job directory escapes its remote root")
    source = (
        Path(metadata["remote_root"]) / "sources"
        / metadata["source_sha256"]
    )
    source_manifest = _verify_source(
        source, metadata["source_sha256"], allow_extra=False)
    _validate_transport_provenance(metadata, source_manifest)
    return metadata, source_manifest


def _launch_job_transition(args: argparse.Namespace) -> int:
    """Remote-only, CAS-like queued-to-systemd launch transition."""
    job = Path(args.job_dir).resolve()
    metadata, _source_manifest = _validated_remote_job(job)
    with _remote_job_transition_lock(job):
        status = _read_json(job / "status.json")
        if status.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("unsupported remote CAD status protocol")
        state = status.get("state")
        if state in {"succeeded", "failed", "canceled"}:
            return 0
        if state != "queued":
            raise RuntimeError(
                f"remote CAD job is {state!r}, not launchable 'queued'")
        if (job / "cancel.request.json").is_file():
            (job / "exit_code").write_text("130\n", encoding="ascii")
            _write_status(job, "canceled", exit_code=130)
            return 0

        unit = metadata["systemd_unit"]
        facts = _local_systemd_unit_facts(unit)
        active = (
            facts["load_state"] != "not-found"
            and facts["active_state"] in {
                "active", "activating", "reloading", "deactivating",
            })
        if active:
            return 0
        if (job / "launch.accepted").is_file():
            (job / "exit_code").write_text("99\n", encoding="ascii")
            _write_status(
                job, "failed", exit_code=99,
                failure_reason=(
                    "systemd accepted the unit but the worker wrote no "
                    "terminal status"),
            )
            return 0
        if facts["load_state"] != "not-found":
            raise RuntimeError(
                "refusing to relaunch an existing inactive remote CAD unit: "
                + json.dumps(facts, sort_keys=True))

        environment_python = (
            f"{metadata['remote_root']}/envs/"
            f"{metadata['environment_sha256']}/bin/python"
        )
        executor = _job_executor_path(metadata)
        marker_then_exec = (
            'set -eu; umask 077; : > "$1"; shift; exec "$@"'
        )
        launch = [
            "systemd-run", "--user", "--quiet", "--collect",
            f"--unit={unit}", "--property=Type=exec",
            f"--property=MemoryMax={REMOTE_MEMORY_MAX_SYSTEMD}",
            "--property=MemorySwapMax=0",
            "--property=MemoryAccounting=yes",
            "--property=KillMode=control-group",
            "--property=OOMPolicy=stop",
            "/bin/sh", "-c", marker_then_exec, "lx-cad-launch",
            str(job / "launch.accepted"), environment_python, executor,
            "_execute-job", "--job-dir", str(job),
        ]
        subprocess.run(launch, check=True)
    return 0


def _cancel_job_transition(args: argparse.Namespace) -> int:
    """Remote-only transition that makes cancellation win over launch."""
    job = Path(args.job_dir).resolve()
    metadata, _source_manifest = _validated_remote_job(job)
    with _remote_job_transition_lock(job):
        status = _read_json(job / "status.json")
        if status.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("unsupported remote CAD status protocol")
        if status.get("state") in {"succeeded", "failed", "canceled"}:
            return 0
        _atomic_json(job / "cancel.request.json", {
            "protocol_version": PROTOCOL_VERSION,
            "requested_utc": _utc_now(),
        })
        unit = metadata["systemd_unit"]
        facts = _local_systemd_unit_facts(unit)
        if (facts["load_state"] != "not-found"
                and facts["active_state"] in {
                    "active", "activating", "reloading", "deactivating",
                }):
            subprocess.run(
                ["systemctl", "--user", "stop", unit], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            facts = _local_systemd_unit_facts(unit)
        if (facts["load_state"] != "not-found"
                and facts["active_state"] in {
                    "active", "activating", "reloading", "deactivating",
                }):
            raise RuntimeError(
                "remote CAD unit remains active; cancellation is pending: "
                + json.dumps(facts, sort_keys=True))
        # This job was nonterminal when cancellation acquired the transition
        # lock. Once its unit is inactive, cancellation wins even if the
        # worker managed to write a terminal status while stop was in flight.
        (job / "exit_code").write_text("130\n", encoding="ascii")
        _write_status(job, "canceled", exit_code=130)
    return 0


def _cancel(args: argparse.Namespace) -> int:
    _local_dir, metadata = _load_local_job(args.job_id)
    remote = Remote(metadata["host"])
    status = _remote_status(remote, metadata)
    if status.get("state") in {"succeeded", "failed", "canceled"}:
        print(
            f"Remote CAD job {args.job_id} is already "
            f"{status.get('state')}.")
        return 0
    tool = _job_executor_path(metadata)
    _verify_remote_tool(remote, tool, metadata["transport"]["sha256"])
    job_dir = f"{metadata['remote_root']}/jobs/{args.job_id}"
    _remote_python(remote, tool, "_cancel-job", "--job-dir", job_dir)
    final_status = _remote_status(remote, metadata)
    final_state = final_status.get("state")
    if final_state == "canceled":
        print(f"Remote CAD job {args.job_id} is stopped/canceled.")
    elif final_state in {"succeeded", "failed"}:
        print(
            f"Remote CAD job {args.job_id} became {final_state} before "
            "cancellation acquired the transition.")
    else:
        raise RuntimeError(
            f"remote CAD cancellation left nonterminal state {final_state!r}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="snapshot, launch, wait, and fetch")
    run.add_argument("targets", nargs="*", help="Make targets (default: all)")
    run.add_argument("--detach", action="store_true", help="return after launch")
    run.add_argument("--host")
    run.add_argument("--remote-root")
    run.add_argument(
        "--jobs", type=int,
        help=f"parallel remote recipe slots (default: {DEFAULT_REMOTE_JOBS})",
    )
    run.set_defaults(function=_run_remote)
    resume = commands.add_parser("resume", help="reattach and fetch a known job")
    resume.add_argument("job_id")
    resume.set_defaults(function=_resume)
    status = commands.add_parser("status", help="show state and recent remote log")
    status.add_argument("job_id")
    status.set_defaults(function=_status)
    gc = commands.add_parser(
        "gc", help="prune old job/source/environment state (local + remote)")
    gc.add_argument("--retain-days", default=None)
    gc.add_argument("--local-only", action="store_true")
    gc.set_defaults(function=_gc_command)
    gc_remote = commands.add_parser("_gc-remote")
    gc_remote.add_argument("--remote-root", required=True)
    gc_remote.add_argument("--retain-days", required=True)
    gc_remote.set_defaults(function=_gc_remote_state)
    cancel = commands.add_parser("cancel", help="stop a remote job's cgroup")
    cancel.add_argument("job_id")
    cancel.set_defaults(function=_cancel)

    verify = commands.add_parser("_verify-source")
    verify.add_argument("--root", required=True)
    verify.add_argument("--expected", required=True)
    verify.add_argument("--allow-extra", action="store_true")
    verify.set_defaults(function=lambda args: (
        _verify_source(Path(args.root), args.expected, args.allow_extra) and 0
    ))
    install = commands.add_parser("_install-source")
    install.add_argument("--remote-root", required=True)
    install.add_argument("--archive", required=True)
    install.add_argument("--expected", required=True)
    install.set_defaults(function=_install_source)
    environment = commands.add_parser("_prepare-environment")
    environment.add_argument("--remote-root", required=True)
    environment.add_argument("--environment-hash", required=True)
    environment.add_argument("--requirements", required=True)
    environment.add_argument("--revalidate", action="store_true")
    environment.set_defaults(function=_prepare_environment)
    job = commands.add_parser("_prepare-job")
    job.add_argument("--remote-root", required=True)
    job.add_argument("--source-hash", required=True)
    job.add_argument("--job-id", required=True)
    job.add_argument("--metadata", required=True)
    job.set_defaults(function=_prepare_job)
    execute = commands.add_parser("_execute-job")
    execute.add_argument("--job-dir", required=True)
    execute.set_defaults(function=_execute_job)
    publish_cache = commands.add_parser("_publish-cache")
    publish_cache.add_argument("--job-dir", required=True)
    publish_cache.set_defaults(function=_publish_cache_command)
    launch_job = commands.add_parser("_launch-job")
    launch_job.add_argument("--job-dir", required=True)
    launch_job.set_defaults(function=_launch_job_transition)
    cancel_job = commands.add_parser("_cancel-job")
    cancel_job.add_argument("--job-dir", required=True)
    cancel_job.set_defaults(function=_cancel_job_transition)
    failed = commands.add_parser("_mark-failed")
    failed.add_argument("--job-dir", required=True)
    failed.add_argument("--exit-code", type=int, default=99)
    failed.add_argument("--reason", default="remote systemd unit stopped")
    failed.set_defaults(function=_mark_failed)
    promote = commands.add_parser("_promote-local")
    promote.add_argument("--incoming", required=True)
    promote.add_argument("--job-dir", required=True)
    promote.add_argument("--result", required=True)
    promote.set_defaults(function=_promote_local)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.function(args))
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"remote CAD error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
