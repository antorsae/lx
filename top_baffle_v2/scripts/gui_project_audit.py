"""Reconstruct GUI project geometry against canonical meshes and placements.

Temporary staged meshes are regenerated from sources, never trusted from the
slicer workspace. No source, shelf file, cache or manifest is changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from delivery_contract import canonical_source
from bambu_3mf_audit import audit_bambu_3mf, validate_bed_fit
from composite_bambu_3mf_audit import audit_bambu_composite_3mf


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_gui_geometry(root: Path, entry: dict, project: Path) -> dict:
    import build_obiwan_combo_plate as combo
    source = canonical_source(root, entry)
    sources = {source}
    profile_path = root / 'captive_magnet_slicing_profile_petg_gf_06hf.json'
    profile = json.loads(profile_path.read_text())
    sources.add(profile_path)

    def modifiers(artifact_id):
        if not artifact_id:
            return []
        state, variant, part = artifact_id.split(':')
        identity = dict(state=state, variant=variant, part=part)
        result = []
        for spec in profile.get('parameter_modifiers', []):
            if all(identity.get(k) == v for k, v in spec['match'].items()):
                contract = root / spec['contract']
                data = json.loads(contract.read_text())
                mesh = root / data['modifier_stl']
                if sha256(mesh) != data['modifier_stl_sha256']:
                    raise ValueError(f'modifier hash mismatch: {mesh}')
                sources.update((contract, mesh))
                result.append((mesh, data['process']))
        return result

    with tempfile.TemporaryDirectory(prefix='lx521-gui-audit-') as temp:
        staging = Path(temp)
        if entry.get('composite_plate'):
            api = combo.get_variant(entry['state'])
            normals, blockers, mods = [], [], []
            sources.add(api.PLATE_MANIFEST)
            for part in api.PARTS:
                matrix = combo._placement_matrix(part)
                placed = staging / part.staged_name
                combo._materialize(part.source_stl, placed, matrix)
                normals.append((placed, (0, 0, 0)))
                sources.add(part.source_stl)
                if part.support_blocker:
                    placed_blocker = staging / part.support_blocker.name
                    combo._materialize(part.support_blocker, placed_blocker, matrix)
                    blockers.append((placed_blocker, (0, 0, 0)))
                    sources.add(part.support_blocker)
                for mesh, settings in modifiers(part.artifact_id):
                    placed_modifier = staging / mesh.name
                    combo._materialize(mesh, placed_modifier, matrix)
                    mods.append((placed_modifier, (0, 0, 0), settings))
            facts = audit_bambu_composite_3mf(
                project, source, normal_part_stls=normals,
                support_blocker_stls=blockers, parameter_modifier_stls=mods)
            matrix = facts.stl_to_bed_matrix
            if max(abs(matrix[r][c] - float(r == c)) for r in range(4) for c in range(4)) > 2e-6:
                raise ValueError('GUI combo moved from its locked plate placement')
        else:
            blocker = source.parent.parent / 'support_blockers' / (source.stem + '.support_blocker.stl')
            blockers = [blocker] if blocker.is_file() else []
            sources.update(blockers)
            facts = audit_bambu_3mf(project, source, support_blocker_stls=blockers,
                                   parameter_modifier_stls=modifiers(entry.get('catalog_artifact_id')))
        validate_bed_fit(facts.transformed_actual_mesh_bounds, {axis: (0.0, 256.0) for axis in ('x', 'y', 'z')}, tolerance_mm=0.01)
        result = facts.as_record()
        # Do not persist workstation or disposable staging paths.
        result.pop('project_3mf', None)
        result.pop('staged_stl', None)
    for mesh in list(sources):
        sidecar = mesh.with_suffix('.print.json')
        if sidecar.is_file():
            data = json.loads(sidecar.read_text())
            if data['stl_sha256'] != sha256(mesh):
                raise ValueError(f'print sidecar hash mismatch: {mesh}')
            sources.add(sidecar)
    result['sources'] = {str(p.relative_to(root)): sha256(p) for p in sorted(sources)}
    result['project_sha256'] = sha256(project)
    result['status'] = 'pass'
    result['scope'] = 'geometry, front-down placement, bed fit, blockers and modifier settings; not sliced toolpaths'
    return result


def audit_gui_settings(root: Path, entry: dict, project: Path) -> dict:
    """Check unsliced settings against the source recipe, not the file itself."""
    import build_petg_gui_projects as gui
    from lx521_baffle.print_policy import validate_material_mapping, policy
    with zipfile.ZipFile(project) as z:
        settings = json.loads(z.read('Metadata/project_settings.config'))
    validate_material_mapping(settings)
    if settings.get('enable_support') == '1':
        for key, wanted in policy()['support'].items():
            assert settings.get(key) == wanted, (entry['name'], key, settings.get(key), wanted)
    profile = json.loads((root / 'captive_magnet_slicing_profile_petg_gf_06hf.json').read_text())
    if entry.get('composite_plate'):
        return gui._validate(project, label=entry['name'], expected_infill='100%',
                             expected_pattern='zig-zag', expected_parts=5 if entry['state']=='floor_stand' else 4)
    artifact_id = entry.get('catalog_artifact_id')
    process = dict(profile['repo_overrides']['process'])
    if artifact_id:
        state, variant, part = artifact_id.split(':')
        identity = dict(state=state, variant=variant, part=part)
        for spec in profile['artifact_overrides']:
            if all(identity.get(k) == v for k, v in spec['match'].items()):
                process.update(spec['process'])
    supported = str(process.get('enable_support', '0')) == '1'
    magnets = 0
    if supported:
        catalog = json.loads((root / 'review/captive_magnet_release_catalog.json').read_text())
        artifact = next(a for a in catalog['artifacts'] if a['id'] == artifact_id)
        magnets = len(artifact['sites'])
    return gui._validate_single(project, label=entry['name'], expects={
        'support': supported, 'infill': process['sparse_infill_density'],
        'pattern': process['sparse_infill_pattern'], 'magnets': magnets,
        'blockers': int(supported),
        'modifiers': int('no_floor_stand' in entry['name'] and 'bottom' in entry['name']),
    })
