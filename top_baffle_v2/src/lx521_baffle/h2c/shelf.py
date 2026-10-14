"""Where each H2C job lands on the ``to_print`` shelf.

The shelf is product first: ``to_print/<product>/<group>/``, with every file
of one part side by side (STL, editable project and audited slice). Internal
job names stay stable; only these paths are shown to the person printing. Superseded P2S files live under ``to_print/archive/p2s``.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHELF = ROOT / 'to_print'
CATALOG = SHELF / 'catalog.json'
P2S_ARCHIVE = SHELF / 'archive' / 'p2s'
PRODUCTS = ('stock', 'slim', 'obiwan')
GROUPS = ('lm', 'upper', 'dipole', 'wings', 'addons')
LANE_TAGS = {'petg_gf_pla': 'petg-gf'}

_OBIWAN = {
    'core_lm_carrier_floor_stand': ('lm', 'lm_carrier_floor_stand'),
    'core_lm_carrier_no_floor_stand': ('lm', 'lm_carrier_no_floor_stand'),
    'lm_keyed_bottom_floor_stand': ('lm', 'lm_keyed_bottom_floor_stand'),
    'lm_keyed_bottom_no_floor_stand': ('lm', 'lm_keyed_bottom_no_floor_stand'),
    'lm_keyed_top': ('lm', 'lm_keyed_top'),
    'core_2_of_2_um_carrier': ('upper', 'um_carrier'),
    'addon_tweeter_crescent': ('upper', 'crescent_nd25fw4'),
    'bmr_crescent_TEBM35C10-4': ('upper', 'crescent_bmr'),
    'bmr_crescent_opposed_TEBM35C10-4': ('upper', 'crescent_bmr_opposed'),
    'addon_nl8_service_lid': ('addons', 'nl8_service_lid'),
}


def location(name: str) -> tuple[str, str, str]:
    """(product, group, file stem) for an internal H2C job name."""
    rest = name.removeprefix('h2c_')
    m = re.fullmatch(r'dayton_nd25fn4_(stock|obiwan)_tweeters_(right|left)_(.+)', rest)
    if m:
        product, hand, part = m.groups()
        part = re.sub(r'^cap_(front|rear)_tweeter$', r'cap_\1', part)
        group = 'wings' if part.startswith('wing_') else 'upper'
        return product, group, f'{product}_nd25fn4_tweeters_{hand}_{part}'
    m = re.fullmatch(r'(grs_pt2522_4|mundorf_amt17d2_1)_obiwan_(t_above|t_below)_(upright|rotated)(?:_(lm_top|um_t))?', rest)
    if m:
        family, layout, orient, part = m.groups()
        short = {'grs_pt2522_4': 'pt2522', 'mundorf_amt17d2_1': 'amt17d2'}[family]
        return 'obiwan', 'dipole', f'obiwan_{short}_{layout}_{orient}' + (f'_{part}' if part else '')
    product, _, part = rest.partition('_')
    if product == 'obiwan':
        if part in _OBIWAN:
            group, stem = _OBIWAN[part]
        elif part.startswith('wing_'):
            group, stem = 'wings', part
        else:
            raise KeyError(name)
        return product, group, f'obiwan_{stem}'
    if product in ('stock', 'slim'):
        if part.startswith('lm_'):
            group = 'lm'
        elif part.startswith('upper'):
            group = 'upper'
        elif re.match(r'\d\d_(A_shoulder|B1_wing)_', part):
            group, part = 'wings', part[3:].lower()
        else:
            raise KeyError(name)
        return product, group, f'{product}_{part}'
    raise KeyError(name)


def folder(name: str) -> Path:
    product, group, _ = location(name)
    return SHELF / product / group


def stl(name: str) -> Path:
    return folder(name) / f'{location(name)[2]}.stl'


def project(name: str, lane: str) -> Path:
    return folder(name) / f'{location(name)[2]}.{LANE_TAGS[lane]}.3mf'


def sliced(project_path: Path) -> Path:
    return project_path.with_name(project_path.stem + '.gcode.3mf')
