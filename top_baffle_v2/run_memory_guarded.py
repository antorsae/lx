"""Run one CAD command without allowing it to starve the workstation.

The default ``local-macos`` profile terminates the complete child process
group above 8 GiB RSS or below 0.5 GiB immediately reclaimable memory.  The
``osado-512g`` profile is valid only on an appropriately large Linux host;
the remote dispatcher tightens its per-worker limit and puts all workers in
one 512 GiB cgroup.  Environment settings may make a selected profile
stricter, never relax it.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time


MEMORY_PROFILES = {
    "local-macos": {
        "max_rss_mb": 8192,
        "min_free_mb": 512,
        "max_guard_slots": 1,
    },
    "osado-512g": {
        "max_rss_mb": 512 * 1024,
        "min_free_mb": 64 * 1024,
        "max_guard_slots": 16,
    },
}
MEMORY_PROFILE = os.environ.get("LX_CAD_MEMORY_PROFILE", "local-macos")
if MEMORY_PROFILE not in MEMORY_PROFILES:
    raise RuntimeError(
        f"unknown LX_CAD_MEMORY_PROFILE {MEMORY_PROFILE!r}; expected one of "
        + ", ".join(sorted(MEMORY_PROFILES)))
PROFILE_MAX_RSS_MB = int(MEMORY_PROFILES[MEMORY_PROFILE]["max_rss_mb"])
PROFILE_MIN_FREE_MB = int(MEMORY_PROFILES[MEMORY_PROFILE]["min_free_mb"])
PROFILE_MAX_GUARD_SLOTS = int(
    MEMORY_PROFILES[MEMORY_PROFILE]["max_guard_slots"])


def _positive_environment_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


MAX_RSS_MB = min(
    _positive_environment_int("LX_CAD_MAX_RSS_MB", PROFILE_MAX_RSS_MB),
    PROFILE_MAX_RSS_MB,
)
MIN_FREE_MB = max(
    _positive_environment_int("LX_CAD_MIN_FREE_MB", PROFILE_MIN_FREE_MB),
    PROFILE_MIN_FREE_MB,
)
GUARD_SLOTS = min(
    _positive_environment_int("LX_CAD_GUARD_SLOTS", 1),
    PROFILE_MAX_GUARD_SLOTS,
)


def _linux_total_memory_mib() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^MemTotal:\s+(\d+) kB", text, re.M)
    return int(match.group(1)) / 1024.0 if match else None


def _linux_cgroup_v2_value(name: str) -> str | None:
    """Read one value from this process's unified cgroup."""
    try:
        lines = Path("/proc/self/cgroup").read_text(
            encoding="utf-8").splitlines()
        record = next(line for line in lines if line.startswith("0::"))
        relative = record.split(":", 2)[2].lstrip("/")
        root = Path("/sys/fs/cgroup").resolve()
        directory = (root / relative).resolve()
        if directory != root and root not in directory.parents:
            return None
        return (directory / name).read_text(encoding="ascii").strip()
    except (OSError, StopIteration, ValueError):
        return None


CGROUP_MEMORY_MAX_MIB: int | None = None
if MEMORY_PROFILE == "osado-512g":
    total_mib = _linux_total_memory_mib()
    required_mib = MAX_RSS_MB * GUARD_SLOTS + MIN_FREE_MB
    if sys.platform != "linux" or total_mib is None or total_mib < required_mib:
        raise RuntimeError(
            "osado-512g requires Linux with at least "
            f"{required_mib / 1024:.0f} GiB physical RAM")
    memory_max = _linux_cgroup_v2_value("memory.max")
    memory_swap_max = _linux_cgroup_v2_value("memory.swap.max")
    expected_bytes = PROFILE_MAX_RSS_MB * 1024 * 1024
    if memory_max != str(expected_bytes) or memory_swap_max != "0":
        raise RuntimeError(
            "osado-512g requires the remote executor's cgroup-v2 limits "
            f"memory.max={expected_bytes} and memory.swap.max=0")
    CGROUP_MEMORY_MAX_MIB = PROFILE_MAX_RSS_MB
_GUARD_MARKER = "LX_CAD_MEMORY_GUARDED"
_GUARD_PID = "LX_CAD_MEMORY_GUARD_PID"

_CONTEXT_NONE = "none"
_CONTEXT_VALID = "valid"
_CONTEXT_STALE = "stale"
_CONTEXT_ESCAPED = "escaped"
_CONTEXT_INDETERMINATE = "indeterminate"


def _process_snapshot() -> dict[int, tuple[int, str]] | None:
    """Current pid -> (ppid, command) table for guard authentication."""
    try:
        rows = subprocess.check_output(
            ("ps", "-axo", "pid=,ppid=,command="), text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    records = {}
    for row in rows.splitlines():
        fields = row.strip().split(None, 2)
        if len(fields) < 2:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        records[pid] = (ppid, fields[2] if len(fields) == 3 else "")
    return records


def _guard_context_status() -> str:
    """Classify the inherited guard marker without trusting it alone.

    The marker alone is insufficient because a stale exported shell value
    would silently bypass monitoring. A valid child names the live guard
    PID, that PID is still an ancestor, and its command is this wrapper.

    A live ancestor with a mismatched SID/PGID is materially different from
    a stale marker: starting another guard there would allow the escaped
    descendant to survive termination of the original process group. Callers
    must fail closed for ``escaped`` and ``indeterminate``.

    This prevents accidental stale-state bypass; it is not a security
    boundary against a deliberately hostile local process.
    """
    marker = os.environ.get(_GUARD_MARKER)
    if marker is None and os.environ.get(_GUARD_PID) is None:
        return _CONTEXT_NONE
    if marker != "1":
        return _CONTEXT_STALE
    try:
        guard_pid = int(os.environ[_GUARD_PID])
    except (KeyError, TypeError, ValueError):
        return _CONTEXT_STALE
    records = _process_snapshot()
    if records is None:
        return _CONTEXT_INDETERMINATE
    if guard_pid not in records:
        return _CONTEXT_STALE
    if Path(__file__).name not in records[guard_pid][1]:
        return _CONTEXT_STALE
    seen = set()
    pid = os.getpid()
    guarded_root_pid = None
    while pid > 1 and pid not in seen:
        if pid == guard_pid:
            break
        seen.add(pid)
        record = records.get(pid)
        if record is None:
            return _CONTEXT_INDETERMINATE
        if record[0] == guard_pid:
            guarded_root_pid = pid
        pid = record[0]
    if pid != guard_pid or guarded_root_pid is None:
        return _CONTEXT_STALE
    # The guard starts its root child as a new session. Descendants remain
    # killable/accounted only while they share that root PGID and SID. Reject
    # an escaped setsid/setpgid descendant even though its ancestry remains.
    try:
        if (os.getpgid(0) != guarded_root_pid
                or os.getsid(0) != guarded_root_pid):
            return _CONTEXT_ESCAPED
    except OSError:
        return _CONTEXT_INDETERMINATE
    return _CONTEXT_VALID


def is_guarded_process() -> bool:
    """Return true only inside the original live guarded SID and PGID."""
    return _guard_context_status() == _CONTEXT_VALID


def _workspace_root() -> Path:
    """Nearest repository root, falling back to this script's directory."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return here


def _workspace_lock_path(slot: int = 0) -> Path:
    """Per-user/workspace lock for one permitted outer-guard slot."""
    uid = os.getuid() if hasattr(os, "getuid") else 0
    workspace = str(_workspace_root())
    identity = hashlib.sha256(
        f"{uid}\0{workspace}".encode("utf-8")).hexdigest()[:20]
    suffix = f"-slot-{slot}" if GUARD_SLOTS > 1 else ""
    return (Path(tempfile.gettempdir())
            / f"lx-cad-memory-{uid}-{identity}{suffix}.lock")


def _acquire_workspace_lock() -> tuple[int, Path] | None:
    """Acquire one non-blocking outer-guard slot or return ``None``.

    Make's ``.NOTPARALLEL`` serializes one build, but it cannot protect the
    workstation from a second Make process or a direct CAD CLI. Holding this
    advisory lock for the complete guarded child lifetime enforces one local
    slot.  The large-host profile may expose multiple slots; its dispatcher
    also applies a per-worker cap and one aggregate systemd cgroup cap.
    """
    for slot in range(GUARD_SLOTS):
        path = _workspace_lock_path(slot)
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(
                    f"CAD guard lock is not a regular file: {path}")
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                raise RuntimeError(f"CAD guard lock has the wrong owner: {path}")
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    os.close(fd)
                    continue
                raise
            os.ftruncate(fd, 0)
            record = (
                f"pid={os.getpid()} workspace={_workspace_root()} "
                f"slot={slot + 1}/{GUARD_SLOTS}\n"
            ).encode("utf-8")
            os.write(fd, record)
            return fd, path
        except Exception:
            os.close(fd)
            raise
    return None


def _release_workspace_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _process_tree_rss_kib(root_pid: int) -> int | None:
    """Resident KiB for the root process and all current descendants."""
    try:
        rows = subprocess.check_output(
            ("ps", "-axo", "pid=,ppid=,pgid=,rss="), text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    records = []
    for row in rows.splitlines():
        fields = row.split()
        if len(fields) == 4:
            records.append(tuple(map(int, fields)))
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _pgid, _rss in records:
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    # PGID membership also catches a re-parented helper that remains in the
    # CAD command's session/process group.
    return sum(rss for pid, _ppid, pgid, rss in records
               if pid in descendants or pgid == root_pid)


def _free_memory_mib() -> float | None:
    """Conservative immediately reclaimable memory.

    On macOS, free + speculative + purgeable pages can be reclaimed
    without paging application memory. Inactive and compressed pages are
    deliberately excluded. Linux continues to use MemAvailable.
    """
    try:
        output = subprocess.check_output(("vm_stat",), text=True)
        page_match = re.search(r"page size of (\d+) bytes", output)
        if page_match:
            pages = 0
            found = False
            for label in ("free", "speculative", "purgeable"):
                match = re.search(
                    rf"Pages {label}:\s+(\d+)\.", output)
                if match:
                    pages += int(match.group(1))
                    found = True
            if found:
                return (int(page_match.group(1)) * pages
                        / 1024.0 / 1024.0)
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        match = re.search(r"^MemAvailable:\s+(\d+) kB", meminfo, re.M)
        if match:
            return int(match.group(1)) / 1024.0
    except OSError:
        pass
    return None


def _group_exists(pgid: int) -> bool:
    # killpg(..., 0) can report EPERM for a group containing only an
    # unreaped zombie on macOS. Such a group consumes no CAD memory and
    # cannot be signalled. Prefer an explicit live-member query.
    try:
        rows = subprocess.check_output(
            ("ps", "-axo", "pgid=,state="), text=True)
        return any(
            int(fields[0]) == pgid and "Z" not in fields[1]
            for row in rows.splitlines()
            if len(fields := row.split()) >= 2
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_group(process: subprocess.Popen) -> None:
    """Terminate every member even when the original leader has exited."""
    pgid = process.pid
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 2.0
    while _group_exists(pgid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + 2.0
        while _group_exists(pgid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.05)
    if process.poll() is None:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            # SIGKILL has already been sent to the complete group. Reap the
            # leader as soon as the kernel reports it.
            process.wait()


def _measured(callable_):
    """Retry a transient host-metric failure, then fail closed."""
    for _attempt in range(3):
        value = callable_()
        if value is not None:
            return value
        time.sleep(0.05)
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":
        args.pop(0)
    if not args:
        print("usage: run_memory_guarded.py -- COMMAND [ARG ...]",
              file=sys.stderr)
        return 2

    context_status = _guard_context_status()

    # Orchestrators intentionally invoke this wrapper once per fresh CAD
    # child.  When the orchestrator itself is already the leader monitored
    # by an outer guard, opening another session would put the grandchild
    # outside the outer process group. Replace this wrapper in-place instead:
    # the fresh child remains in the already monitored/killed group and the
    # outer guard still accounts for its complete process tree.
    if context_status == _CONTEXT_VALID:
        try:
            os.execvpe(args[0], args, os.environ.copy())
        except OSError as exc:
            print(f"CAD memory guard exec failed: {exc}", file=sys.stderr)
            return 127

    # Never re-guard a process that still belongs to a live guard's ancestry
    # but has escaped its killable SID/PGID. An unmeasurable live context is
    # equally unsafe: opening another session could strand an OCC descendant.
    if context_status in (_CONTEXT_ESCAPED, _CONTEXT_INDETERMINATE):
        print(
            "CAD memory guard refusing nested command: inherited live guard "
            f"context is {context_status}",
            file=sys.stderr,
            flush=True,
        )
        return 99

    # An unauthenticated marker is stale or manually inherited. Sanitize it
    # before establishing the real outer guard so descendants cannot mistake
    # it for an actively monitored process tree.
    if context_status == _CONTEXT_STALE:
        print(
            "CAD memory guard ignoring unauthenticated inherited marker",
            file=sys.stderr, flush=True)
        os.environ.pop(_GUARD_MARKER, None)
        os.environ.pop(_GUARD_PID, None)

    # Refuse before spawning OCC/Python when the workstation is already
    # below the requested safety floor.  The loop below still enforces the
    # same limit continuously after a successful launch.
    free_mib = _measured(_free_memory_mib)
    if free_mib is None:
        print(
            "CAD memory guard refusing to start command: cannot measure "
            "host free memory", file=sys.stderr, flush=True)
        return 99
    if free_mib < MIN_FREE_MB:
        print(
            "CAD memory guard refusing to start command: "
            f"free memory {free_mib:.0f} MiB < {MIN_FREE_MB} MiB",
            file=sys.stderr,
            flush=True,
        )
        return 99

    try:
        lock = _acquire_workspace_lock()
    except (OSError, RuntimeError) as exc:
        print(
            f"CAD memory guard refusing to start command: cannot acquire "
            f"workspace lock: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 99
    if lock is None:
        print(
            "CAD memory guard refusing to start command: all "
            f"{GUARD_SLOTS} guarded CAD slot(s) are occupied near "
            f"{_workspace_lock_path(0)}",
            file=sys.stderr,
            flush=True,
        )
        return 99
    lock_fd, _lock_path = lock

    process = None
    stop_signal = None

    # The CAD command owns a separate process group so a make/Codex
    # interruption would otherwise kill only this guard and leave OCC
    # consuming CPU and memory as an orphan.  Defer work out of the
    # signal handler, then terminate the complete child group below.
    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum

    watched = (
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGHUP,
        signal.SIGQUIT,
        signal.SIGTSTP,
    )
    previous = {}
    try:
        previous = {sig: signal.getsignal(sig) for sig in watched}
        for sig in watched:
            signal.signal(sig, request_stop)
        child_env = os.environ.copy()
        child_env[_GUARD_MARKER] = "1"
        child_env[_GUARD_PID] = str(os.getpid())
        process = subprocess.Popen(
            args, start_new_session=True, env=child_env)
        while process.poll() is None:
            reason = None
            if stop_signal is not None:
                reason = f"received {signal.Signals(stop_signal).name}"
            else:
                rss_kib = _measured(
                    lambda: _process_tree_rss_kib(process.pid))
                free_mib = _measured(_free_memory_mib)
                if rss_kib is None or free_mib is None:
                    reason = "cannot measure RSS/free memory"
                elif rss_kib / 1024.0 > MAX_RSS_MB:
                    rss_mib = rss_kib / 1024.0
                    reason = f"RSS {rss_mib:.0f} MiB > {MAX_RSS_MB} MiB"
                elif free_mib < MIN_FREE_MB:
                    reason = (f"free memory {free_mib:.0f} MiB < "
                              f"{MIN_FREE_MB} MiB")
            if reason:
                print(
                    f"CAD memory guard terminating pid {process.pid}: "
                    f"{reason}", file=sys.stderr, flush=True)
                _terminate_group(process)
                if stop_signal is not None:
                    return 128 + int(stop_signal)
                return 99
            time.sleep(0.5)
        return_code = int(process.returncode or 0)
        if return_code < 0:
            return_code = 128 + abs(return_code)
        # A command that backgrounds a child is not complete. Remove every
        # surviving group member before returning even after leader exit 0.
        if _group_exists(process.pid):
            print(
                f"CAD memory guard cleaning descendants after pid "
                f"{process.pid} exited", file=sys.stderr, flush=True)
            _terminate_group(process)
            # Never certify a command that left writers behind. In
            # particular, Make must not touch an output stamp after the
            # guard had to kill a background child whose files may be only
            # partly written.
            return 99
        return return_code
    finally:
        # Covers Python exceptions as well as the explicit limit paths.
        # Signal handlers are restored before returning to a caller that
        # imports and invokes main() directly.
        if process is not None and (
                process.poll() is None or _group_exists(process.pid)):
            _terminate_group(process)
        try:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        finally:
            _release_workspace_lock(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
