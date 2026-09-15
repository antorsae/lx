"""Shared, CAD-free delivery dispositions and directory ownership.

The tracked shelf catalog owns part choices. This module defines the file
format and directory for each delivery route; publishers and validators use
the same rules. A GUI project can never acquire a ready-G-code filename.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
from typing import Mapping


class DeliveryKind(str, Enum):
    SLICED = "sliced_project"
    GUI = "gui_project"


@dataclass(frozen=True)
class Lane:
    id: str
    directory: str
    suffix: str
    kind: DeliveryKind
    material: str
    nozzle_mm: float

    def project_path(self, family: str, name: str) -> Path:
        return Path(family) / self.directory / (name + self.suffix)


LANES = {
    "pla04": Lane("pla04", "3mf_04", ".gcode.3mf", DeliveryKind.SLICED, "PLA Basic", 0.4),
    "pla06hf": Lane("pla06hf", "3mf_06hf", "_06hf.gcode.3mf", DeliveryKind.SLICED, "PLA Basic", 0.6),
    "petg_gf_gui": Lane("petg_gf_gui", "3mf_06hf_petg-gf_pla", "_GUI.3mf", DeliveryKind.GUI, "TINMORRY PETG-GF; PLA support interface where enabled", 0.6),
    "petg_gf_wings": Lane("petg_gf_wings", "3mf_06hf_petg-gf", "_06hf_petg-gf.gcode.3mf", DeliveryKind.SLICED, "TINMORRY PETG-GF", 0.6),
}
GUI_DIRECTORY = LANES["petg_gf_gui"].directory
PETG_WING_DIRECTORY = LANES["petg_gf_wings"].directory
LEGACY_DIRECTORIES = {
    "3mf_06hf_petg-cf_pla": GUI_DIRECTORY,
    "3mf_06hf_petg-cf": PETG_WING_DIRECTORY,
}


def is_gui(entry: Mapping) -> bool:
    return entry.get("delivery_kind") == DeliveryKind.GUI.value or entry.get("lane") == "06hf"


def entry_lanes(entry: Mapping) -> tuple[Lane, ...]:
    if is_gui(entry):
        return (LANES["petg_gf_gui"],)
    result = [LANES["pla04"]]
    if not entry.get("auxiliary_delivery"):
        result.append(LANES["pla06hf"])
    if entry.get("selection") in {"flat_wings_b_2piece_plate", "graded_wings_b_2piece_plate"}:
        result.append(LANES["petg_gf_wings"])
    return tuple(result)


def primary_lane(entry: Mapping) -> Lane:
    return entry_lanes(entry)[0]


def stl_path(entry: Mapping) -> Path:
    return Path(entry["family"]) / "stl" / f"{entry['name']}.stl"


def authority_path(entry: Mapping) -> Path:
    return stl_path(entry).with_suffix('.plate.json' if entry.get('composite_plate') else '.print.json')


def deliver_authority(source: Path, destination: Path) -> None:
    """Preserve exact orientation/plate facts under the friendly STL filename."""
    data = json.loads(source.read_text())
    data['stl'] = destination.name.removesuffix('.print.json').removesuffix('.plate.json') + '.stl'
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def expected_projects(entries: list[Mapping]) -> dict[Path, tuple[Mapping, Lane]]:
    return {lane.project_path(entry["family"], entry["name"]): (entry, lane)
            for entry in entries for lane in entry_lanes(entry)}


def canonical_source(root: Path, entry: Mapping) -> Path:
    relative = Path(entry["source_stl"])
    if relative.parts[0] in {"floor_stand", "no_floor_stand", "wings"}:
        relative = Path("build") / relative
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid source path: {relative}")
    return root / relative
