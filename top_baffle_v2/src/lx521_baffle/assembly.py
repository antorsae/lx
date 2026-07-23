"""Small ordered/labeled build123d assembly constructor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from build123d import Compound


def ordered_labeled_compound(
    parts: Mapping[str, Any],
    *,
    label: str,
) -> Any:
    """Return a compound whose children follow mapping iteration order."""
    children = []
    for child_label, solid in parts.items():
        solid.label = child_label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = label
    return assembly
