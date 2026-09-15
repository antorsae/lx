"""Dayton ND25FN-4 release names and immutable retained-design input mapping."""
from pathlib import Path
import json

from ..io import sha256_file

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / 'candidates/nd25fn4_crescent'
BODY = 'h2c_dayton_nd25fn4_body.stl'
CAP = 'h2c_dayton_nd25fn4_cap.stl'
RETAINER = 'h2c_dayton_nd25fn4_retainer.stl'
RETAINED_SOURCES = {
    BODY: SOURCE_ROOT / 'print/geometry/01_UM_Crescent_V4_PRINT.stl',
    CAP: SOURCE_ROOT / 'STL/02_Closed_Cap_PRINT_TWO.stl',
    RETAINER: SOURCE_ROOT / 'STL/03_Tweeter_Retainer_PRINT_TWO.stl',
}


def mesh_name(source):
    source = Path(source).resolve()
    for name, original in RETAINED_SOURCES.items():
        if original.resolve() == source:
            return name
    raise ValueError(f'Unknown Dayton ND25FN-4 source: {source}')


def body_authority():
    source = RETAINED_SOURCES[BODY].with_suffix('.print.json')
    original = json.loads(source.read_text())
    return dict(original, part=Path(BODY).stem, stl=BODY,
                source_authority=str(source.relative_to(ROOT)), source_authority_sha256=sha256_file(source))


def original_accessory_preparation(preparation):
    """Use identical retained cap geometry with the existing ceiling checker.

    The checker identifies the retained parts by their original filenames.
    Only its source references change; plate translations and bytes are checked.
    """
    from copy import deepcopy
    result = deepcopy(preparation)
    for item in result['sources']:
        path = ROOT / item['path']
        original = RETAINED_SOURCES[path.name]
        assert sha256_file(path) == sha256_file(original) == item['sha256']
        item['path'] = str(original.relative_to(ROOT))
    return result
