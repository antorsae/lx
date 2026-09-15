"""Project-owned reference paths; importing this module never loads OCC."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
W22_REFERENCE_STEP = PROJECT_ROOT / "vendor/SEAS/W22EX001/E0022_W22EX001.stp"
MU10_REFERENCE_ROOT = PROJECT_ROOT / "vendor/SEAS/MU10RB-SL"
MU10_REFERENCE_STL = MU10_REFERENCE_ROOT / "H1658-04_MU10RB-SL_driver.stl"
MU10_REFERENCE_NOTES = MU10_REFERENCE_ROOT / "H1658-04_MU10RB-SL_driver_STL_notes.md"
MU10_REFERENCE_DATASHEET = MU10_REFERENCE_ROOT / "H1658-04_MU10RB-SL_Datasheet.pdf"


def geometry_reference_paths() -> tuple[Path, ...]:
    """Keep exporters, remote snapshots and fit checks on the same files."""
    return (MU10_REFERENCE_STL, MU10_REFERENCE_NOTES,
            MU10_REFERENCE_DATASHEET, W22_REFERENCE_STEP)
