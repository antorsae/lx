#!/usr/bin/env python3
"""Fail closed if the H2C shelf is incomplete, stale or contains orphan slices."""
from pathlib import Path
import json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from lx521_baffle.h2c.printing import policy,write_json
from lx521_baffle.io import sha256_file
from lx521_baffle.print_contract import validate_print_sidecar
from lx521_baffle.h2c.dayton import BODY, RETAINED_SOURCES, SOURCE_ROOT, body_authority
from lx521_baffle.tweeter_options import TWEETER_FAMILIES, ND25FN, compatible_tweeters
from build_h2c_release import inventory,prepared_is_current
from audit_h2c_print import audit_is_current


def validate_mesh_authority(path):
    """Validate native print datums or the three unchanged retained Dayton ND25FN-4 meshes.

    The retained cap and retainer were authored directly in print coordinates.
    They have hash-bound build records, not installed-baffle front-down datums.
    The retained body carries its original Dayton ND25FN-4 transform schema. Do not invent
    canonical front-down sidecars for those different source contracts.
    """
    if path.parent!=ROOT/'to_print/h2c/STL/dayton_nd25fn4':
        validate_print_sidecar(path)
        return None
    assert path.name in RETAINED_SOURCES,('Unknown retained Dayton ND25FN-4 mesh',path)
    source=RETAINED_SOURCES[path.name];digest=sha256_file(path)
    assert digest==sha256_file(source),('Retained Dayton ND25FN-4 geometry changed',path)
    evidence={str(source.relative_to(ROOT)):digest}
    if path.name==BODY:
        authority=source.with_suffix('.print.json')
        payload=json.loads(path.with_suffix('.print.json').read_text())
        assert payload==body_authority(),('Renamed body authority differs',path)
        assert payload['stl_sha256']==digest and payload['stl']==path.name
        approved=ROOT/payload['approved_source']
        assert sha256_file(approved)==payload['approved_source_sha256']
        evidence.update({str(authority.relative_to(ROOT)):sha256_file(authority),
                         str(approved.relative_to(ROOT)):sha256_file(approved)})
        kind='retained Dayton ND25FN-4 installed-to-print transform'
    else:
        manifest=SOURCE_ROOT/'build_manifest.json'
        record=json.loads(manifest.read_text())['files'][source.name]
        assert record['sha256']==digest and record['watertight'] and record['winding_consistent']
        assert record['components']==1 and record['volume_mm3']>0 and abs(record['bounds_mm'][0][2])<1e-6
        evidence[str(manifest.relative_to(ROOT))]=sha256_file(manifest)
        kind='retained accessory authored in print coordinates'
    return dict(mesh=str(path.relative_to(ROOT)),contract=kind,inputs=evidence)


def main():
    shelf=ROOT/'to_print/h2c';catalog=json.loads((shelf/'catalog.json').read_text());jobs=catalog['jobs']
    assert catalog['tweeter_families']==list(TWEETER_FAMILIES)
    family_products={row['id']:set(row['products']) for row in TWEETER_FAMILIES}
    expected={row['name']+'__'+lane for row in inventory() for lane in (policy()['lanes'] if row['family']=='dayton_nd25fn4' else ['petg_gf_pla'])}
    expected|={f'h2c_dayton_nd25fn4_{role}__{lane}' for role in ['body','accessories'] for lane in policy()['lanes']}
    ids=[j['id'] for j in jobs]
    assert len(ids)==len(set(ids)) and set(ids)==expected,('Catalog inventory differs',sorted(expected-set(ids)),sorted(set(ids)-expected))
    meshes=set();slices=set();projects=set()
    for job in jobs:
        assert job['tweeter_families'],('Missing tweeter compatibility',job['id'])
        assert job['tweeter_families']==compatible_tweeters(job['name'],job['family'],job['role']),('Stale tweeter compatibility',job['id'])
        for tweeter in job['tweeter_families']:
            assert job['product_family'] in family_products[tweeter],('Incompatible tweeter family',job['id'],tweeter)
        if job['family']==ND25FN:
            assert job['product_family']=='obiwan' and job['tweeter_families']==[ND25FN]
        # Recover the disposable plain-G-code cache from the qualified 3MF
        # when validating a fresh checkout. Never infer a new qualification.
        raw=ROOT/job['work']/'plate_1.gcode'
        if not raw.exists() and job.get('sliced_project'):
            audit=json.loads((ROOT/job['audit']).read_text());path=ROOT/job['sliced_project']
            assert sha256_file(path)==audit['sliced_project_sha256']
            with zipfile.ZipFile(path) as archive:data=archive.read('Metadata/plate_1.gcode')
            import hashlib
            assert hashlib.sha256(data).hexdigest()==audit['gcode_sha256']
            raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(data)
        assert audit_is_current(job),('Missing or stale print qualification',job['id'])
        assert prepared_is_current(job,job['prepare_inputs']),('Changed prepared geometry',job['id'])
        for path,digest in job['prepare_inputs']['files'].items():
            assert sha256_file(ROOT/path)==digest,('Changed preparation input',job['id'],path)
        projects.add((ROOT/job['project']).resolve());slices.add((ROOT/job['sliced_project']).resolve())
        meshes.update(ROOT/s['path'] for s in job['preparation']['sources'] if s['subtype']=='normal_part')
    retained=[record for path in sorted(meshes) if (record:=validate_mesh_authority(path)) is not None]
    actual={p.resolve() for p in shelf.rglob('*.gcode.3mf')}
    assert actual==slices,('Unlisted or missing slices',sorted(map(str,actual-slices)),sorted(map(str,slices-actual)))
    editable={p.resolve() for p in shelf.rglob('*.3mf') if not p.name.endswith('.gcode.3mf')}
    assert editable==projects,('Unlisted or missing editable projects',editable-projects,projects-editable)
    assert {p.resolve() for p in shelf.rglob('*.stl')}=={p.resolve() for p in meshes},'Unlisted or missing print meshes'
    for path in [ROOT/'build/h2c/geometry_validation.json',ROOT/'build/h2c/obiwan_interface_validation.json']:
        data=json.loads(path.read_text());assert data['status']=='pass'
    result=dict(status='pass',jobs=len(jobs),editable_projects=len(projects),sliced_projects=len(slices),
        distinct_print_meshes=len(meshes),checkpoint=catalog['checkpoint'],physical_qualification='pending H2C hardware trial',
        retained_mesh_authorities=retained,
        catalog_sha256=sha256_file(shelf/'catalog.json'),policy_sha256=sha256_file(ROOT/'print_policy_h2c.json'),
        source_sha256=sha256_file(Path(__file__)))
    write_json(ROOT/'build/h2c/release_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
