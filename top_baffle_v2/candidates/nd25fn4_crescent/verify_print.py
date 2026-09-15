"""Read-only verification of delivered projects, current sources and audit hashes."""
import hashlib
import json
import zipfile
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lx521_baffle.print_policy import policy_sha256


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    manifest=json.loads((HERE/'print/manifest.json').read_text())
    expected={'body','accessories','V4_flat_left_UPPER','V4_flat_right_UPPER',
              'V4_graded_left_UPPER','V4_graded_right_UPPER'}
    assert {r['key'] for r in manifest['projects']}==expected
    assert len(manifest['projects'])==len(expected)
    prep_path=ROOT/manifest['preparation']
    assert sha(prep_path)==manifest['preparation_sha256']
    prep=json.loads(prep_path.read_text());authority=prep['authority']
    assert sha(ROOT/authority['project'])==authority['sha256']
    for path,digest in {**authority['profile_files'],**manifest['source_scripts']}.items():
        assert sha(ROOT/path)==digest,path
    for row in manifest['projects']:
        project=ROOT/row['path'];audit_path=ROOT/row['audit'];static_path=ROOT/row['static_audit']
        assert sha(project)==row['sha256'],project
        assert sha(audit_path)==row['audit_sha256'],audit_path
        assert sha(static_path)==row['static_audit_sha256'],static_path
        audit=json.loads(audit_path.read_text());static=json.loads(static_path.read_text())
        assert audit['status']=='pass' and audit['project_sha256']==row['sha256']
        assert audit['policy_sha256']==policy_sha256()
        assert static['ok'] and not static['errors']
        for part in audit['geometry']:
            source=ROOT/part['source'];assert sha(source)==part['sha256'],source
            if source.parent==HERE/'print/geometry':
                sidecar=json.loads(source.with_suffix('.print.json').read_text())
                assert sha(ROOT/sidecar['approved_source'])==sidecar['approved_source_sha256']
                assert sidecar['stl_sha256']==part['sha256']
                assert sidecar['sliced_project_sha256']==row['sha256']
        with zipfile.ZipFile(project) as archive:
            gcode=archive.read('Metadata/plate_1.gcode')
        digest=hashlib.sha256(gcode).hexdigest()
        assert digest==sha(audit_path.parent/'plate_1.gcode')
        if row['key']=='body':
            inserts=audit['ducts']
            assert inserts['pass_'] and inserts['insert_site_count']==12
            assert inserts['insert_definition_sha256']==sha(ROOT/'review/nd25fn4_print/inputs/insert_bores.json')
            assert len(inserts['insert_bores'])==12  # all M3, flat four-mm insert floors
            for contact in inserts['insert_boundary_contacts']:
                assert contact['nominal_boundary_overlap_mm']<=contact['maximum_allowed_boundary_overlap_mm']+.0001
            for bore in inserts['insert_bores']:
                clearance=bore['minimum_support_clearance_mm']
                assert (clearance is None and bore['support_samples']==0) or clearance>=-.03,bore
            views=json.loads((HERE/'print/views_manifest.json').read_text())
            assert digest==views['gcode_sha256']
            for name,image_hash in views['images'].items():
                assert sha(HERE/'print'/name)==image_hash,name
            fit=json.loads((HERE/'print/LM_assembly_check.json').read_text())
            assert fit['source_script_sha256']==sha(HERE/'verify_lm_assembly.py')
            assert fit['body_project']['sha256']==row['sha256']
            assert sha(ROOT/fit['body']['path'])==fit['body']['sha256']
            assert sha((ROOT/fit['body']['path']).with_suffix('.print.json'))==fit['body']['sidecar_sha256']
            assert set(fit['configurations'])=={'no_floor_stand','floor_stand'}
            for state,result in fit['configurations'].items():
                assert sha(ROOT/result['LM']['path'])==result['LM']['sha256'],state
                assert result['mount_compatible'] and result['exterior_flush'],state
                assert result['overlap_mm3']<.002 and result['LM_front_occluded_area_mm2']<.002,state
                assert max(abs(s['front_surface_step_mm']) for s in result['seam_samples'])<.002,state
        if row['key']=='accessories':
            caps=audit['cap_ceilings']
            assert caps['status']=='pass' and caps['gcode_sha256']==digest
            assert caps['source_sha256']==sha(HERE/'cap_support_check.py')
            assert len(caps['caps'])==2 and len(caps['retainers'])==2
            view=json.loads((HERE/'print/cap_support_validation.json').read_text())
            assert view['gcode_sha256']==digest and view['project_sha256']==row['sha256']
            assert sha(HERE/'print'/view['image'])==view['image_sha256']
            for cap in caps['caps']:
                assert len(cap['interface_layers'])==3 and abs(cap['interface_to_ceiling_layer_gap_mm'])<.02
                assert all(l['inner_ceiling_coverage']>.98 for l in cap['interface_layers'])
        assert row['pauses_mm']==audit.get('magnet_pauses_mm',[])
        if row['key']!='accessories':
            walls=audit['D6_retaining_walls']
            assert walls['status']=='pass' and walls['gcode_sha256']==digest
            assert walls['source_sha256']==sha(ROOT/'scripts/captive_wall_audit.py')
            assert len(walls['sites'])==(4 if row['key']=='body' else 2)
            for site in walls['sites']:
                assert site['maximum_connected_components']==walls['maximum_retaining_components']
                assert site['minimum_interlayer_overlap_fraction']>=walls['minimum_interlayer_overlap_fraction']
        print(row['key']+': delivered file, sources, toolpaths and evidence match')
    print('PASS: all six native print jobs and current previews verified; no files changed')


if __name__=='__main__':main()
