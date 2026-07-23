"""Generic fresh-process case dispatch for heavyweight direct-CLI suites."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence


GUARDED_CASE = "guarded"
SERVICE_ORCHESTRATOR_CASE = "service_orchestrator"


@dataclass(frozen=True)
class GuardedCase:
    """One stable selector mapped to an in-process assertion callable."""

    case_id: str
    function: Callable[..., None]
    args: tuple[Any, ...]
    stand_state: bool | None
    service_orchestrator_class: str
    make_stamp: str
    legacy_selector: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.make_stamp or not self.legacy_selector:
            raise ValueError("case identifiers must be non-empty")
        if self.service_orchestrator_class not in {
                GUARDED_CASE, SERVICE_ORCHESTRATOR_CASE}:
            raise ValueError(
                "unknown service-orchestrator class: "
                f"{self.service_orchestrator_class}")

    def run(self) -> None:
        self.function(*self.args)


def _case_map(cases: Sequence[GuardedCase]) -> dict[str, GuardedCase]:
    result: dict[str, GuardedCase] = {}
    stamps: set[str] = set()
    legacy: set[str] = set()
    for case in cases:
        if case.case_id in result:
            raise ValueError(f"duplicate case ID: {case.case_id}")
        if case.make_stamp in stamps:
            raise ValueError(f"duplicate Make stamp: {case.make_stamp}")
        if case.legacy_selector in legacy:
            raise ValueError(
                f"duplicate legacy selector: {case.legacy_selector}")
        result[case.case_id] = case
        stamps.add(case.make_stamp)
        legacy.add(case.legacy_selector)
    return result


def select_case(
    cases: Sequence[GuardedCase], case_id: str,
) -> GuardedCase:
    """Resolve exactly one stable selector or fail closed."""
    case = _case_map(cases).get(case_id)
    if case is None:
        raise SystemExit(f"unknown case ID: {case_id}")
    return case


def _is_local_service_parent(
    case: GuardedCase, *, large_host: bool,
) -> bool:
    return (
        case.service_orchestrator_class == SERVICE_ORCHESTRATOR_CASE
        and not large_host
    )


def run_selected_case(
    cases: Sequence[GuardedCase],
    case_id: str,
    *,
    script: Path,
    guard: Path,
    is_guarded_process: Callable[[], bool],
    large_host: bool,
    before_case: Callable[[GuardedCase], None] | None = None,
) -> None:
    """Run one selected case, self-wrapping in the guard when required."""
    case = select_case(cases, case_id)
    local_service_parent = _is_local_service_parent(
        case, large_host=large_host)
    if not is_guarded_process() and not local_service_parent:
        proc = subprocess.run(
            [sys.executable, str(guard), "--", sys.executable, str(script)],
            env=os.environ.copy())
        raise SystemExit(proc.returncode)
    if before_case is not None:
        before_case(case)
    print(f"{case.legacy_selector}:", flush=True)
    case.run()


def _run_case_process(
    case: GuardedCase,
    *,
    script: Path,
    guard: Path,
    selector_env: str,
    private_env_prefix: str,
    large_host: bool,
) -> tuple[str, int, str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith(private_env_prefix):
            env.pop(name)
    env[selector_env] = case.case_id
    command = (
        [sys.executable, str(script)]
        if _is_local_service_parent(case, large_host=large_host)
        else [sys.executable, str(guard), "--", sys.executable, str(script)]
    )
    proc = subprocess.run(
        command, env=env, text=True, capture_output=True)
    return case.case_id, proc.returncode, proc.stdout, proc.stderr


def run_suite(
    cases: Sequence[GuardedCase],
    *,
    script: Path,
    guard: Path,
    selector_env: str,
    private_env_prefix: str,
    requested_workers: int,
    large_host: bool,
    suite_label: str,
    success_message: str,
) -> None:
    """Run cases in registry order with bounded optional concurrency."""
    ordered = tuple(cases)
    _case_map(ordered)
    workers = (
        min(requested_workers, len(ordered)) if large_host else 1)
    results: list[tuple[str, int, str, str]] = []

    def run(case: GuardedCase) -> tuple[str, int, str, str]:
        return _run_case_process(
            case, script=script, guard=guard, selector_env=selector_env,
            private_env_prefix=private_env_prefix, large_host=large_host)

    if workers == 1:
        for case in ordered:
            result = run(case)
            results.append(result)
            _name, _returncode, stdout, stderr = result
            print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", file=sys.stderr, flush=True)
    else:
        print(
            f"{suite_label} remote runner: {workers} concurrent isolated "
            "cases; shared staging remains lock-serialized",
            flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run, case): case.case_id for case in ordered
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _name, _returncode, stdout, stderr = result
                print(stdout, end="", flush=True)
                if stderr:
                    print(stderr, end="", file=sys.stderr, flush=True)

    failed_ids = {
        case_id for case_id, returncode, _stdout, _stderr in results
        if returncode
    }
    failed = [case.case_id for case in ordered if case.case_id in failed_ids]
    if failed:
        raise SystemExit(f"{suite_label} FAILED: " + ", ".join(failed))
    print(f"\n{success_message}")
