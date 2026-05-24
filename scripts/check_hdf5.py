#!/usr/bin/env python3
"""
Decide whether REW loading is needed based on HDF5 vs .mdat mtimes.

Env:
  SET_NAME   Measurement set name (config.MEASUREMENT_SETS key).
  HDF5_PATH  Target HDF5 path to validate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import config
import h5py


def needs_reload(hdf5_path: Path, dirs: list[Path], expected_peak_policy: str | None = None) -> bool:
    if not hdf5_path.exists():
        return True
    if expected_peak_policy:
        try:
            with h5py.File(hdf5_path, "r") as h5:
                actual = str(h5.attrs.get("direct_ir_peak_policy", ""))
        except OSError:
            return True
        if actual != expected_peak_policy:
            return True
    try:
        hdf5_mtime = hdf5_path.stat().st_mtime
    except OSError:
        return True

    for d in dirs:
        if not d:
            continue
        d_path = Path(d)
        if not d_path.exists():
            continue
        for p in d_path.rglob("*.mdat"):
            try:
                if p.stat().st_mtime > hdf5_mtime:
                    return True
            except OSError:
                continue
    return False


def main() -> int:
    set_name = os.environ.get("SET_NAME")
    hdf5_env = os.environ.get("HDF5_PATH")
    if not set_name:
        print("Error: SET_NAME is not set")
        return 1
    if not hdf5_env:
        print("Error: HDF5_PATH is not set")
        return 1

    mset = config.MEASUREMENT_SETS.get(set_name)
    if not mset:
        print(f"Error: Unknown measurement set '{set_name}'")
        return 1

    sources = mset.get("sources")
    if sources:
        dirs = [src.get("path") for src in sources]
    else:
        dirs = [mset.get("path")]

    hdf5_path = Path(hdf5_env)
    expected_peak_policy = mset.get("direct_ir_peak_policy")
    if needs_reload(hdf5_path, dirs, expected_peak_policy=expected_peak_policy):
        os.execv(sys.executable, [sys.executable, "run_pipeline.py", "-m", set_name, "--skip-viz"])

    print(f"OK: {hdf5_path} is up to date (skipping REW load)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
