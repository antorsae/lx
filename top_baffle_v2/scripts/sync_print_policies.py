"""Apply authoritative process decisions to existing unsliced GUI projects.

Meshes, placements, blockers and pauses are byte-preserved. This is not a
reslice of a ready project; projects containing G-code are rejected.
"""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lx521_baffle.print_policy import normalize_material_mapping, role_settings, policy_sha256
from build_petg_gui_projects import GUI_PROCESS_ID


def main():
    profile = json.loads((ROOT/'captive_magnet_slicing_profile_petg_gf_06hf.json').read_text())
    directory = ROOT/'to_print/obiwan/3mf_06hf_petg-gf_pla'
    report = []
    for path in sorted(directory.glob('*.3mf')):
        with zipfile.ZipFile(path) as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        assert not any(n.endswith('.gcode') for n in members), path
        settings = json.loads(members['Metadata/project_settings.config'])
        before = settings.copy()
        settings.update(profile['repo_overrides']['process'])
        name = path.stem
        if 'combo' in name:
            role = 'core_combo'
        elif 'LM_top' in name:
            role = 'lm_top'
        elif 'LM_bottom' in name:
            role = 'lm_bottom'
        elif '_UM_carrier' in name:
            role = 'um'
        else:
            role = 'regular_tweeter'
        settings.update(role_settings(role))
        if role in ('core_combo', 'lm_top', 'lm_bottom', 'um'):
            settings.update(enable_support='1', support_on_build_plate_only='1',
                            support_critical_regions_only='1', support_remove_small_overhang='1')
        # Preserve layout-specific tower coordinates and all filament recipes.
        for key in ('wipe_tower_x', 'wipe_tower_y'):
            if key in before: settings[key] = before[key]
        normalize_material_mapping(settings)
        settings['print_settings_id'] = GUI_PROCESS_ID
        members['Metadata/project_settings.config'] = (json.dumps(settings, indent=4)+'\n').encode()
        temp = path.with_suffix('.tmp')
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
            for key, value in members.items(): archive.writestr(key, value)
        temp.replace(path)
        report.append(dict(path=str(path.relative_to(ROOT)), role=role,
                           changed={k: dict(before=before.get(k), after=v) for k,v in settings.items() if v != before.get(k)}))
    destination = ROOT/'review/print_policy_update_20260912/gui_policy_sync.json'
    destination.write_text(json.dumps(dict(policy_sha256=policy_sha256(), projects=report), indent=2)+'\n')
    print(f'Updated {len(report)} unsliced projects; mesh members preserved')


if __name__ == '__main__': main()
