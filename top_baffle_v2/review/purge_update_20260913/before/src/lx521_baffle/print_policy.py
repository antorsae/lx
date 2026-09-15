"""Shared print decisions; geometry and native slicer checks remain independent."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[2] / 'print_policy.json'


def policy():
    return json.loads(POLICY_PATH.read_text())


def policy_sha256():
    return hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()


def role_settings(role):
    return deepcopy(policy()['roles'][role])


def normalize_material_mapping(settings):
    """Normalize all mapping vectors together, without changing filament recipes.

    Bambu 2.7 P2S retains both profile variant columns. Partial vector repairs
    can silently lose PLA even after a zero-exit slice; actual tool validation
    is a separate mandatory gate. This does not masquerade as two nozzles.
    """
    ids = settings.get('filament_settings_id')
    if not isinstance(ids, list) or len(ids) not in (1, 2):
        raise ValueError('Expected one GF filament or GF plus PLA')
    count = len(ids)
    if 'PETG' not in ids[0] or (count == 2 and 'PLA' not in ids[1]):
        raise ValueError(f'Material order must be GF, then PLA: {ids}')
    settings['filament_map_mode'] = 'Manual'
    for key in ('filament_map', 'filament_map_2', 'filament_nozzle_map', 'filament_volume_map'):
        settings[key] = ['1'] * count
    settings['nozzle_volume_type'] = ['High Flow']
    settings['extruder_nozzle_stats'] = ['High Flow#1']
    colours = ['#456C86', '#EEEEEE'][:count]
    settings['filament_colour'] = colours
    settings['default_filament_colour'] = colours.copy()
    purge = policy()['materials']['purge_each_direction_mm3']
    settings['flush_volumes_matrix'] = [str(0 if a == b else purge) for a in range(count) for b in range(count)]
    settings['flush_volumes_vector'] = [str(purge // 2)] * (count * 2)
    return settings


def validate_material_mapping(settings):
    expected = normalize_material_mapping(deepcopy(settings))
    for key in ('filament_map_mode', 'filament_map', 'filament_map_2',
                'filament_nozzle_map', 'filament_volume_map', 'nozzle_volume_type',
                'extruder_nozzle_stats', 'flush_volumes_matrix', 'flush_volumes_vector'):
        if settings.get(key) != expected[key]:
            raise ValueError(f'Incomplete or incorrect material mapping: {key}')


def apply_supported_policy(settings):
    settings.update(policy()['support'])
    return normalize_material_mapping(settings)


def validate_object_infill(global_settings, object_settings, role):
    """Resolve slicer precedence; a correct global value cannot hide an override."""
    wanted = role_settings(role)
    for obj in object_settings:
        for key in ('sparse_infill_density', 'sparse_infill_pattern'):
            actual = obj.get(key, global_settings.get(key))
            if actual != wanted[key]:
                raise ValueError(f'{role}: effective {key}={actual!r}, expected {wanted[key]!r}')


def native_material_values(actual, model, interface, nozzle_variant='High Flow'):
    """Expand scalar settings into Bambu's per-material variant columns.

    Check both materials: accepting just the first value would hide a wrong
    PLA recipe. Do not infer an extra nozzle from the number of columns.
    """
    if not isinstance(actual, list) or len(actual) not in (2, 4):
        return model
    columns = len(actual)//2
    def expand(value):
        values = value if isinstance(value, list) else [value]
        if columns == 1 and len(values) == 2:
            return [selected_filament_value(values, nozzle_variant)]
        return values*columns if len(values)==1 else values
    return expand(model)+expand(interface)


def selected_filament_value(value, nozzle_variant):
    values = value if isinstance(value, list) else [value]
    if len(values) == 1:
        return values[0]
    if len(values) != 2:
        raise ValueError(f'Unexpected filament variant columns: {values}')
    return values[1 if 'High Flow' in str(nozzle_variant) else 0]
