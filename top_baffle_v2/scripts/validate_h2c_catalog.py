#!/usr/bin/env python3
"""Fail closed if the H2C shelf is incomplete, stale or contains orphan slices."""
from pathlib import Path
import json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from lx521_baffle.h2c.printing import policy,write_json
from lx521_baffle.io import sha256_file
from lx521_baffle.print_contract import validate_print_sidecar
from lx521_baffle.tweeter_options import TWEETER_FAMILIES, ND25FN, compatible_tweeters
from build_h2c_release import inventory,prepared_is_current,row_lanes
from lx521_baffle.h2c import shelf as layout
from audit_h2c_print import audit_is_current


def main():
    shelf=layout.SHELF;catalog=json.loads(layout.CATALOG.read_text());jobs=catalog['jobs']
    # Combined per-product files: listed in plates.json, built from the current audited parts.
    import build_h2c_plates as plates_builder
    combined=plates_builder.manifest()['files']
    assert set(combined)==set(plates_builder.FILES),('Combined files missing or unlisted',sorted(set(plates_builder.FILES)^set(combined)))
    extra=set()
    for key,entry in combined.items():
        for k in ('project','sliced_project'):
            assert sha256_file(ROOT/entry[k])==entry[k+'_sha256'],('Changed combined file',key,k)
            extra.add((ROOT/entry[k]).resolve())
        assert entry['status']=='sliced_audited',('Combined file not audited',key)
        assert all(sha256_file(ROOT/f)==d for f,d in entry['inputs']['code'].items()),('Stale combined file (code/policy changed)',key)
    def shelf_files(pattern):
        return {p.resolve() for p in shelf.rglob(pattern) if not p.is_relative_to(layout.SHELF/'archive')}-extra
    assert catalog['tweeter_families']==list(TWEETER_FAMILIES)
    family_products={row['id']:set(row['products']) for row in TWEETER_FAMILIES}
    expected={row['name']+'__'+lane for row in inventory() for lane in row_lanes(row)}
    ids=[j['id'] for j in jobs]
    assert len(ids)==len(set(ids)) and set(ids)==expected,('Catalog inventory differs',sorted(expected-set(ids)),sorted(set(ids)-expected))
    meshes=set();slices=set();projects=set()
    for job in jobs:
        assert job['tweeter_families'],('Missing tweeter compatibility',job['id'])
        assert job['tweeter_families']==compatible_tweeters(job['name'],job['family'],job['role']),('Stale tweeter compatibility',job['id'])
        for tweeter in job['tweeter_families']:
            assert job['product_family'] in family_products[tweeter],('Incompatible tweeter family',job['id'],tweeter)
        if job['family']==ND25FN:
            assert job['product_family'] in {'stock','obiwan'} and job['tweeter_families']==[ND25FN]
            assert job['name'].startswith(f"h2c_dayton_nd25fn4_{job['product_family']}_"),('ND25FN-4 name must lead with its product',job['id'])
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
        assert ROOT/job['project']==layout.project(job['name'],job['lane']) and ROOT/job['stl']==layout.stl(job['name']),('Off-layout shelf file',job['id'])
        projects.add((ROOT/job['project']).resolve());slices.add((ROOT/job['sliced_project']).resolve())
        meshes.update(ROOT/s['path'] for s in job['preparation']['sources'] if s['subtype']=='normal_part')
    for path in sorted(meshes):validate_print_sidecar(path)
    actual=shelf_files('*.gcode.3mf')
    assert actual==slices,('Unlisted or missing slices',sorted(map(str,actual-slices)),sorted(map(str,slices-actual)))
    editable={p for p in shelf_files('*.3mf') if not p.name.endswith('.gcode.3mf')}
    assert editable==projects,('Unlisted or missing editable projects',editable-projects,projects-editable)
    assert shelf_files('*.stl')=={p.resolve() for p in meshes},'Unlisted or missing print meshes'
    for path in [ROOT/'build/h2c/geometry_validation.json',ROOT/'build/h2c/obiwan_interface_validation.json']:
        data=json.loads(path.read_text());assert data['status']=='pass'
    by_name={j['name']:j for j in jobs if j['lane']=='petg_gf_pla'}
    for key,entry in combined.items():
        stale=[n for n,d in entry['inputs']['parts'].items() if by_name[n]['project_sha256']!=d]
        assert not stale,('Combined file built from superseded parts; run make h2c_plates',key,stale)
    result=dict(status='pass',jobs=len(jobs),editable_projects=len(projects),sliced_projects=len(slices),
        combined_files=len(combined),combined_plates=sum(len(e['plates']) for e in combined.values()),
        distinct_print_meshes=len(meshes),checkpoint=catalog['checkpoint'],physical_qualification='pending H2C hardware trial',
        catalog_sha256=sha256_file(layout.CATALOG),policy_sha256=sha256_file(ROOT/'print_policy_h2c.json'),
        source_sha256=sha256_file(Path(__file__)))
    write_json(ROOT/'build/h2c/release_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
