"""Delivery regressions: stand identity, ownership and read-only validation."""
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import delivery_contract as contract
import manage_delivery as delivery
import build_to_print_shelf as shelf_builder
import build_petg_gui_projects as gui


def test_canonical_pruning_preserves_independent_06_lane(tmp_path):
    entries = [dict(name='wing', family='obiwan'),
               dict(name='core', family='obiwan', lane='06hf', delivery_kind='gui_project')]
    for e in entries:
        p = tmp_path/contract.stl_path(e);p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(b'stl')
    pla = tmp_path/'obiwan/3mf_04/wing.gcode.3mf';pla.parent.mkdir();pla.write_bytes(b'pla')
    hf = tmp_path/'obiwan/3mf_06hf/wing_06hf.gcode.3mf';hf.parent.mkdir();hf.write_bytes(b'hf')
    stale = pla.parent/'retired.gcode.3mf';stale.write_bytes(b'old')
    shelf_builder._prune_delivery_view(tmp_path, entries)
    assert hf.read_bytes() == b'hf'
    assert not stale.exists()


def test_state_resolution_cannot_select_other_stand(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, 'PROJECT_ROOT', tmp_path)
    for state in ('floor_stand', 'no_floor_stand'):
        p=tmp_path/'build'/state/'stl/shared.stl';p.parent.mkdir(parents=True);p.write_text(state)
    assemble=tmp_path/'assemble.json'
    assemble.write_text(json.dumps({'plates':[{'objects':[{'path':'/expired/shared.stl'}]}]}))
    gui._remap_assemble_paths(assemble, 'no_floor_stand')
    actual=Path(json.loads(assemble.read_text())['plates'][0]['objects'][0]['path'])
    assert actual.read_text() == 'no_floor_stand'
    actual.unlink()
    with pytest.raises(gui.GuiProjectError, match='no durable source'):
        gui._remap_assemble_paths(assemble, 'no_floor_stand')


def test_gui_can_never_have_a_ready_filename():
    e=dict(name='core', family='obiwan', lane='06hf')
    lanes=contract.entry_lanes(e)
    assert len(lanes)==1
    assert lanes[0].kind == contract.DeliveryKind.GUI
    assert str(lanes[0].project_path('obiwan','core')).endswith('_GUI.3mf')


@pytest.fixture
def tiny_shelf(tmp_path, monkeypatch):
    monkeypatch.setattr(delivery,'ROOT',tmp_path)
    shelf=tmp_path/'to_print';shelf.mkdir()
    e=dict(name='sample',family='stock',state='shared',description='Sample',source_stl='build/sample.stl')
    source=tmp_path/e['source_stl'];source.parent.mkdir();source.write_bytes(b'canonical STL')
    authority=source.with_suffix('.print.json');authority.write_text(json.dumps({'stl_sha256':delivery.sha256(source)}))
    stl=shelf/contract.stl_path(e);stl.parent.mkdir(parents=True);stl.write_bytes(source.read_bytes())
    catalog=shelf/'catalog.json';catalog.write_text(json.dumps({'entries':[e]}))
    records=[]
    for rel,(entry,lane) in contract.expected_projects([e]).items():
        p=shelf/rel;p.parent.mkdir(parents=True);p.write_bytes(b'audited project')
        records.append(dict(name=e['name'],family='stock',path=str(rel),kind=lane.kind.value,lane=lane.id,
                            sha256=delivery.sha256(p),source_stl=e['source_stl'],source_stl_sha256=delivery.sha256(source),
                            source_authority=dict(path=str(authority.relative_to(tmp_path)),sha256=delivery.sha256(authority)),
                            estimated_time='1m',estimated_filament_g='1'))
    alias=shelf/contract.authority_path(e)
    contract.deliver_authority(authority,alias)
    for record in records:
        record['delivered_authority']=dict(path=str(alias.relative_to(shelf)),sha256=delivery.sha256(alias))
    manifest=dict(choice_count=1,project_count=2,catalog_sha256=delivery.sha256(catalog),projects=records,
                  disposition_counts={'sliced_project':2})
    delivery.write(shelf/'delivery_manifest.json',manifest)
    delivery.write_guide(shelf,[e],manifest)
    return shelf,source


def snapshot(root):
    return {str(p.relative_to(root)):(p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
            for p in [root,*root.rglob('*')]}


def test_validate_is_read_only_including_timestamps(tiny_shelf):
    shelf,_=tiny_shelf
    before=snapshot(shelf.parent)
    assert delivery.validate(shelf)['project_count']==2
    assert snapshot(shelf.parent)==before


def test_unmanifested_gcode_is_rejected(tiny_shelf):
    shelf,_=tiny_shelf
    (shelf/'stock/3mf_06hf/obsolete.gcode.3mf').write_bytes(b'old')
    with pytest.raises(ValueError,match='unexpected='):
        delivery.validate(shelf)


def test_changed_source_is_rejected(tiny_shelf):
    shelf,source=tiny_shelf;source.write_bytes(b'new geometry')
    with pytest.raises(ValueError,match='source/STL mismatch'):
        delivery.validate(shelf)


def test_changed_delivery_is_rejected(tiny_shelf):
    shelf,_=tiny_shelf
    next(shelf.glob('*/*/*.3mf')).write_bytes(b'GUI saved over job')
    with pytest.raises(ValueError,match='delivery hash mismatch'):
        delivery.validate(shelf)


def test_validation_cli_cannot_call_publisher(tiny_shelf,monkeypatch):
    shelf,_=tiny_shelf
    def forbidden(**kwargs):
        raise AssertionError('validation entered publisher')
    monkeypatch.setattr(shelf_builder,'build_shelf',forbidden)
    before=snapshot(shelf.parent)
    assert shelf_builder.main(['--shelf',str(shelf),'--catalog',str(shelf/'catalog.json'),'--validate-only'])==0
    assert snapshot(shelf.parent)==before


def test_gui_archive_with_gcode_is_rejected(tmp_path):
    p=tmp_path/'core_GUI.3mf'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('Metadata/plate_1.gcode','G1 X0')
        z.writestr('Metadata/project_settings.config','{}')
    with pytest.raises(ValueError,match='disposition mismatch'):
        delivery.archive_facts(p,contract.LANES['petg_gf_gui'])


def test_actual_interface_tool_is_checked(tmp_path):
    from audit_gui_slice import material_extrusions
    p=tmp_path/'slice.gcode'
    text='M83\nT0\n; FEATURE: Outer wall\nG1 X1 Y1 E1\n; FEATURE: Support\nG1 X2 Y1 E1\nT1\n; FEATURE: Support interface\nG1 X3 Y1 E1\n'
    p.write_text(text)
    assert material_extrusions(p,True)==dict(model=1,support=1,interface=1)
    p.write_text(text.replace('T1','T0'))
    with pytest.raises(ValueError,match='material assignment failed'):
        material_extrusions(p,True)
