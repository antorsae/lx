"""Verify and index the direct STL deliverables; no archive is produced."""
import json
from pathlib import Path
from rebuild import sha
from v4_model import BODY_FILE, LM_STATES

HERE=Path(__file__).resolve().parent


def main():
    checks=json.loads((HERE/'validation.json').read_text())
    wings=json.loads((HERE/'wing_validation.json').read_text())
    build=json.loads((HERE/'build_manifest.json').read_text())
    assert checks['status']=='passed' and set(checks['configurations'])==set(LM_STATES)
    assert checks['shared_body_file']==BODY_FILE
    for row in checks['configurations'].values():
        assert row['body_file']==BODY_FILE and row['body_sha256']==checks['STL'][BODY_FILE]['sha256']
    assert checks['validation_source_sha256']==sha(HERE/'validate.py')
    assert checks['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    lower_band=json.loads((HERE/'lower_band_validation.json').read_text())
    assert lower_band['status']=='passed'
    assert lower_band['body_sha256']==checks['STL'][BODY_FILE]['sha256']
    assert lower_band['model_sha256']==sha(HERE/'v4_model.py')
    assert lower_band['source_sha256']==sha(HERE/'validate_lower_band.py')
    assert lower_band['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    outline=json.loads((HERE/'outline_validation.json').read_text())
    assert outline['status']=='passed'
    assert outline['body_sha256']==checks['STL'][BODY_FILE]['sha256']
    assert outline['model_source_sha256']==sha(HERE/'v4_model.py')
    assert outline['source_sha256']==sha(HERE/'validate_outline.py')
    assert outline['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    waist=json.loads((HERE/'waist_validation.json').read_text())
    assert waist['status']=='passed' and waist['body_sha256']==checks['STL'][BODY_FILE]['sha256']
    assert waist['source_sha256']==sha(HERE/'validate_waist.py')
    assert waist['model_source_sha256']==sha(HERE/'v4_model.py')
    assert waist['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    waist_review=json.loads((HERE/'views/waist_review_manifest.json').read_text())
    assert waist_review['source_sha256']==sha(HERE/'waist_review.py')
    assert waist_review['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    for name,digest in waist_review['images'].items():
        assert sha(HERE/'views'/name)==digest,name
    reference=json.loads((HERE/'reference_validation.json').read_text())
    assert reference['status']=='passed'
    assert reference['body_sha256']==checks['STL'][BODY_FILE]['sha256']
    assert reference['model_source_sha256']==sha(HERE/'v4_model.py')
    assert reference['source_sha256']==sha(HERE/'validate_reference.py')
    assert reference['trace_source_sha256']==sha(HERE/'reference_outline.py')
    assert reference['reference_sha256']==sha(HERE.parents[1]/reference['reference_path'])
    assert reference['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    for name,checksum in reference['outputs'].items():
        assert sha(HERE/'views'/name)==checksum,name
    depth=json.loads((HERE/'depth_magnet_validation.json').read_text())
    assert depth['status']=='passed' and depth['body_sha256']==checks['STL'][BODY_FILE]['sha256']
    assert depth['source_sha256']==sha(HERE/'validate_depth_magnets.py')
    assert depth['model_source_sha256']==sha(HERE/'v4_model.py')
    assert depth['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    assert depth['source_wing_validation_sha256']==sha(HERE/'wing_validation.json')
    finish_checks={}
    for state,filename in [('no_floor_stand','um_finish_no_floor_validation.json'),
                           ('floor_stand','um_finish_floor_validation.json')]:
        report=json.loads((HERE/filename).read_text())
        assert report['status']=='passed' and report['LM_state']==state
        assert report['LM_sha256']==sha(HERE.parents[1]/'build'/state/'stl/obiwan_core_1_of_2_lm_carrier.stl')
        assert report['um_sha256']==checks['STL'][BODY_FILE]['sha256']
        assert report['source_sha256']==sha(HERE.parents[1]/'scripts/check_um_finish.py')
        assert report['gallery_reference_sha256']==sha(HERE/'assembly/wire_gallery_reference.json')
        finish_checks[filename]=sha(HERE/filename)
    for key,name in [('model_source_sha256','v4_model.py'),('builder_source_sha256','rebuild.py'),
                     ('mesh_ops_source_sha256','mesh_ops.py')]:
        assert build[key]==sha(HERE/name),name
    assert wings['source_sha256']==sha(HERE/'build_wings.py')
    assert wings['model_source_sha256']==sha(HERE/'v4_model.py')
    assert wings['housing_stl_sha256']==checks['STL'][BODY_FILE]['sha256']
    for name,row in wings['parts'].items():
        assert checks['matching_upper_wings'][name]['stl_sha256']==row['stl_sha256'],name
    snapshots=json.loads((HERE/'snapshot_validation.json').read_text())
    assert snapshots['status']=='reviewed'
    assert snapshots['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    assert snapshots['source_wing_validation_sha256']==sha(HERE/'wing_validation.json')
    assert snapshots['source_review_sha256']==sha(HERE/'review.py')
    for name,checksum in snapshots['reviewed_pngs'].items():
        assert sha(HERE/'views'/name)==checksum,name
    review=json.loads((HERE/'views/manifest.json').read_text())
    assert review['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    assert review['source_review_sha256']==sha(HERE/'review.py')
    assert review['glb_sha256']==sha(HERE/'views/nd25fn4_color_review.glb')
    for row in review['wing_scenes'].values():
        assert row['sha256']==sha(HERE/'views'/row['path'])
    links=json.loads((HERE/'views/viewer_link_validation.json').read_text())
    assert set(links)=={'nd25fn4_color_review.glb','nd25fn4_flat_wings.glb','nd25fn4_graded_wings.glb'}
    for name,row in links.items():
        assert row['asset_http_status']==200 and row['served_sha256']==sha(HERE/'views'/name)
        assert row['catalog_file']==str((HERE/'views'/name).resolve())
        assert row['bytes']==(HERE/'views'/name).stat().st_size
    comparison=json.loads((HERE/'views/lower_band_comparison_manifest.json').read_text())
    assert comparison['source_sha256']==sha(HERE/'lower_band_review.py')
    assert comparison['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    for name,checksum in comparison['inputs'].items():
        assert checksum==sha(HERE.parents[1]/name)
    assert comparison['output_sha256']==sha(HERE/'views/UM_lower_band_comparison.png')
    outline_comparison=json.loads((HERE/'views/outline_comparison_manifest.json').read_text())
    assert outline_comparison['source_sha256']==sha(HERE/'outline_review.py')
    assert outline_comparison['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    for name,checksum in outline_comparison['inputs'].items():
        assert checksum==sha(HERE.parents[1]/name)
    for name,checksum in outline_comparison['outputs'].items():
        assert checksum==sha(HERE/'views'/name)
    depth_review=json.loads((HERE/'views/depth_review_manifest.json').read_text())
    assert depth_review['source_sha256']==sha(HERE/'depth_review.py')
    assert depth_review['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
    for name,checksum in depth_review['inputs'].items():
        assert sha(HERE.parents[1]/name)==checksum,name
    for name,checksum in depth_review['outputs'].items():
        assert sha(HERE/'views'/name)==checksum,name
    orthographic=json.loads((HERE/'views/orthographic_manifest.json').read_text())
    for name,checksum in orthographic['source_sha256'].items():
        assert sha(HERE.parents[1]/name)==checksum,name
    for row in orthographic['outputs'].values():
        assert sha(HERE/'views'/row['path'])==row['sha256'],row['path']
    expected={**{f'STL/{k}':v['sha256'] for k,v in checks['STL'].items()},
              **{f'STL/wings/{k}':v['stl_sha256'] for k,v in wings['parts'].items()}}
    assert len(expected)==7
    assert {str(f.relative_to(HERE)) for f in (HERE/'STL').rglob('*.stl')}==set(expected)
    for name,checksum in expected.items():
        assert sha(HERE/name)==checksum,name
    fit_path=HERE/'print/LM_assembly_check.json'
    fit=json.loads(fit_path.read_text())
    assert fit['source_script_sha256']==sha(HERE/'verify_lm_assembly.py')
    print_body=HERE.parents[1]/fit['body']['path']
    authority=json.loads(print_body.with_suffix('.print.json').read_text())
    assert authority['approved_source_sha256']==checks['STL'][BODY_FILE]['sha256']
    assert sha(print_body)==fit['body']['sha256']
    assert set(fit['configurations'])==set(LM_STATES)
    for state,row in fit['configurations'].items():
        assert sha(HERE.parents[1]/row['LM']['path'])==row['LM']['sha256'],state
        assert row['exterior_flush'] and row['LM_front_occluded_area_mm2']<.002,state
    interface=json.loads((HERE/'views/LM_interface_render_manifest.json').read_text())
    assert interface['assembly_check_sha256']==sha(fit_path)
    assert interface['GLB']['sha256']==sha(HERE/'views/LM_interface_detail.glb')
    hardware=json.loads((HERE/'hardware_validation.json').read_text())
    assert hardware==checks['M3_hardware'] and hardware['status']=='pass'
    policy_path=HERE.parents[1]/'print_policy.json'
    policy=json.loads(policy_path.read_text())
    assert hardware['thread']==policy['hardware']['crescent_retainer']['thread']
    assert hardware['count']==policy['hardware']['crescent_retainer']['screw_count']
    print_manifest=json.loads((HERE/'print/manifest.json').read_text())
    assert len(print_manifest['projects'])==6
    for row in print_manifest['projects']:
        assert sha(HERE.parents[1]/row['path'])==row['sha256']
        audit_path=HERE.parents[1]/row['audit']
        assert sha(audit_path)==row['audit_sha256']
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='pass' and audit['policy_sha256']==sha(policy_path)
    coupon_path=HERE/'print/qualification/qualification.json'
    coupon=json.loads(coupon_path.read_text())
    assert coupon['status']=='pass' and coupon['policy_sha256']==sha(policy_path)
    assert sha(HERE.parents[1]/coupon['project'])==coupon['project_sha256']
    assert coupon['source_script_sha256']==sha(HERE/'build_magnet_coupon.py')
    stock_path=HERE.parents[1]/'review/print_policy_update_20260912/M3_stock_confirmation.json'
    stock=json.loads(stock_path.read_text())
    assert stock['status']=='pass' and stock['policy_sha256']==sha(policy_path)
    assert stock['stock']==policy['hardware_stock']['M3_insert']
    assert stock['body_sha256']==checks['STL'][BODY_FILE]['sha256']
    assert stock['retainer_sha256']==sha(HERE/'STL/03_Tweeter_Retainer_PRINT_TWO.stl')
    assert stock['source_sha256']==sha(HERE/'verify_m3_stock.py')
    for source,digest in stock['dimensional_sources_sha256'].items():
        assert sha(HERE.parents[1]/source)==digest,source
    manifest={'format':'direct_stl_files','status':'geometry_checked_unsliced','units':'mm',
        'STLs':expected,'validation_sha256':sha(HERE/'validation.json'),
        'print_policy_sha256':sha(policy_path),
        'hardware':{'validation':'hardware_validation.json','sha256':sha(HERE/'hardware_validation.json'),
            'validator_sha256':sha(HERE/'validate_hardware.py'),'thread':hardware['thread'],
            'insert_stock':stock['stock'],
            'stock_validation':{'path':str(stock_path.relative_to(HERE.parents[1])),'sha256':sha(stock_path)},
            'matched_revision':['STL/'+BODY_FILE,'STL/03_Tweeter_Retainer_PRINT_TWO.stl']},
        'native_print_jobs':{'manifest':'print/manifest.json','sha256':sha(HERE/'print/manifest.json'),
            'project_count':len(print_manifest['projects']),'digital_audit':'passed'},
        'D6_qualification':{'report':'print/qualification/qualification.json','sha256':sha(coupon_path),
            'digital_audit':coupon['status'],'physical_qualification':coupon['physical_qualification']},
        'UM_finish_validation':finish_checks,
        'lower_band_validation_sha256':sha(HERE/'lower_band_validation.json'),
        'outline_validation_sha256':sha(HERE/'outline_validation.json'),
        'waist_validation_sha256':sha(HERE/'waist_validation.json'),
        'waist_comparison':{'path':'views/waist_comparison.png','sha256':sha(HERE/'views/waist_comparison.png')},
        'reference_validation_sha256':sha(HERE/'reference_validation.json'),
        'reference_outline_rms_error_mm':reference['rms_outline_error_mm'],
        'reference_overlay':{'path':'views/UM_reference_overlay.png','sha256':sha(HERE/'views/UM_reference_overlay.png')},
        'depth_magnet_validation_sha256':sha(HERE/'depth_magnet_validation.json'),
        'depth_comparison':{'path':'views/UM_depth_comparison.png','sha256':sha(HERE/'views/UM_depth_comparison.png')},
        'outline_comparison':{'path':'views/UM_outline_comparison.png',
            'sha256':sha(HERE/'views/UM_outline_comparison.png')},
        'review_glb':{'path':'views/nd25fn4_color_review.glb','sha256':review['glb_sha256']},
        'viewer_links':{name:row['viewer_url'] for name,row in links.items()},
        'viewer_link_validation_sha256':sha(HERE/'views/viewer_link_validation.json'),
        'UM_magnets':{'size_mm':[6,3],'quantity_per_body_and_upper_wing_pair':8,
            'recommended_part':'Superimanes D-06-03 N45','physical_holding_force':'not measured'},
        'rear_comparison':{'path':'views/UM_lower_band_comparison.png','sha256':comparison['output_sha256']},
        'shared_body':f'STL/{BODY_FILE}','compatible_LM_configurations':list(LM_STATES),
        'LM_interface':{'front_z_mm':18.3,'exterior_flush':True,'LM_front_occluded_area_mm2':0,
            'print_mesh_check':'print/LM_assembly_check.json','check_sha256':sha(fit_path),
            'render':'views/LM_interface_orthographic_3D.png',
            'render_sha256':sha(HERE/'views/LM_interface_orthographic_3D.png')},
        'quantities':'ONE shared body for either LM configuration, TWO caps, TWO retainers; ONE matching upper-wing pair',
        'instructions':'README.md','archive_produced':False}
    path=HERE/'delivery_manifest.json'
    # Independent material lanes remain catalogued after a GF-only refresh.
    prior=json.loads(path.read_text()) if path.exists() else {}
    variants=prior.get('native_print_material_variants',{})
    for name,variant in variants.items():
        lane=HERE/variant['manifest'];data=json.loads(lane.read_text())
        assert all(sha(HERE.parents[1]/row['path'])==row['sha256'] for row in data['projects'])
        variant['sha256']=sha(lane)
        if 'surface_test' in variant:
            qualification=HERE/variant['surface_test']
            test=json.loads(qualification.read_text())
            assert test['status']=='pass' and sha(HERE.parents[1]/test['project'])==test['project_sha256']
            variant['surface_test_sha256']=sha(qualification)
    if variants:manifest['native_print_material_variants']=variants
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verified seven direct STL files, including one shared body:',HERE/'STL',flush=True)


if __name__=='__main__':
    main()
