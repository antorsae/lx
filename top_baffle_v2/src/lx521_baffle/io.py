"""Dependency-free byte serialization and SHA-256 primitives.

Publication policy deliberately stays with each caller.  These helpers do
not create directories, choose temporary paths, replace files, or validate a
producer-specific transaction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for *data*."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pretty_json_bytes(data: Any, *, allow_nan: bool) -> bytes:
    """Render deterministic, indented UTF-8 JSON with one final newline.

    ``allow_nan`` is intentionally required so each producer preserves its
    established strict or permissive serialization contract explicitly.
    """
    return (
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            allow_nan=allow_nan,
        )
        + "\n"
    ).encode("utf-8")
