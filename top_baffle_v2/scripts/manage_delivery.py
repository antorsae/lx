#!/usr/bin/env python3
"""Bind, validate and package the local delivery without contacting a printer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from delivery_contract import (DeliveryKind, LANES, canonical_source,
                               expected_projects, stl_path, authority_path, deliver_authority)
from gui_project_audit import audit_gui_geometry, audit_gui_settings, sha256


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(shelf, entries):
    expected = set(expected_projects(entries))
    actual = {p.relative_to(shelf) for p in shelf.glob('*/*/*.3mf')}
    require(actual == expected,
            f'project inventory mismatch: missing={sorted(map(str, expected-actual))}; unexpected={sorted(map(str, actual-expected))}')
    wanted_stls = {stl_path(e) for e in entries}
    actual_stls = {p.relative_to(shelf) for p in shelf.glob('*/stl/*.stl')}
    require(actual_stls == wanted_stls, 'STL inventory differs from catalog')


def archive_facts(path, lane):
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, f'corrupt ZIP: {path}')
        members = archive.namelist()
        gcode_members = [n for n in members if n.endswith('.gcode')]
        require(bool(gcode_members) == (lane.kind == DeliveryKind.SLICED),
                f'GUI/sliced disposition mismatch: {path}')
        settings = json.loads(archive.read('Metadata/project_settings.config'))
        result = {'settings': {k: settings.get(k) for k in (
            'filament_settings_id', 'nozzle_diameter', 'nozzle_volume_type',
            'layer_height', 'wall_loops', 'sparse_infill_density', 'sparse_infill_pattern',
            'enable_support', 'support_type', 'support_style', 'support_interface_filament',
            'support_top_z_distance', 'flush_volumes_matrix', 'print_settings_id')},
            'gcode_members': {n: hashlib.sha256(archive.read(n)).hexdigest() for n in gcode_members}}
        if gcode_members:
            gcode = archive.read(gcode_members[0]).decode('utf-8', errors='replace')
            time = re.search(r'total estimated time: ([^\r\n]+)', gcode)
            grams = re.search(r'total filament weight \[g\]\s*:\s*([^\r\n]+)', gcode)
            result['estimated_time'] = time[1].strip() if time else None
            result['estimated_filament_g'] = grams[1].strip() if grams else None
        else:
            result.update(estimated_time=None, estimated_filament_g=None)
        return result


def prior_sliced_evidence(shelf):
    """Preserve and check existing slice provenance; do not invent a new CAD build."""
    evidence = {}
    for e in read(shelf / 'release_manifest.json')['entries']:
        rel = Path(e['p2s_project']).relative_to('to_print')
        evidence[rel] = (e['p2s_project_sha256'], 'release_manifest.json')
    for e in read(shelf / 'catalog_06hf.json')['entries']:
        evidence[Path(e['project'])] = (e['project_sha256'], 'catalog_06hf.json')
    report = shelf / 'obiwan' / LANES['petg_gf_wings'].directory / 'wing_plates.json'
    for e in read(report)['plates']:
        rel = Path(e['project']).relative_to('to_print')
        evidence[rel] = (e['project_sha256'], str(report.relative_to(shelf)))
    return evidence


def refresh(shelf, catalog):
    entries = read(catalog)['entries']
    inventory(shelf, entries)
    prior = prior_sliced_evidence(shelf)
    records = []
    for rel, (entry, lane) in expected_projects(entries).items():
        path = shelf / rel
        source = canonical_source(ROOT, entry)
        require(sha256(source) == sha256(shelf / stl_path(entry)), f'stale shelf STL: {entry["name"]}')
        facts = archive_facts(path, lane)
        record = dict(name=entry['name'], family=entry['family'], lane=lane.id,
                      kind=lane.kind.value, path=str(rel), sha256=sha256(path),
                      source_stl=str(source.relative_to(ROOT)), source_stl_sha256=sha256(source),
                      **facts)
        sidecar = source.with_suffix('.plate.json' if entry.get('composite_plate') else '.print.json')
        require(sidecar.is_file(), f'missing source authority: {sidecar}')
        require(read(sidecar)['stl_sha256'] == sha256(source), f'stale source authority: {sidecar}')
        record['source_authority'] = {'path': str(sidecar.relative_to(ROOT)), 'sha256': sha256(sidecar)}
        delivered_authority = shelf / authority_path(entry)
        deliver_authority(sidecar, delivered_authority)
        record['delivered_authority'] = {'path': str(authority_path(entry)), 'sha256': sha256(delivered_authority)}
        if lane.kind == DeliveryKind.GUI:
            print(f'Auditing GUI mesh: {entry["name"]}', flush=True)
            record['geometry_audit'] = audit_gui_geometry(ROOT, entry, path)
            record['settings_audit'] = audit_gui_settings(ROOT, entry, path)
            record['qualification'] = 'GUI slice and physical qualification pending'
        else:
            require(rel in prior, f'no prior slice audit for {rel}')
            digest, authority = prior[rel]
            require(sha256(path) == digest, f'project differs from prior slice audit: {rel}')
            record['slice_authority'] = authority
        records.append(record)
    manifest = dict(schema_version=1, kind='lx521_delivery_inventory', catalog_sha256=sha256(catalog),
                    choice_count=len(entries), family_counts=dict(Counter(e['family'] for e in entries)),
                    project_count=len(records), disposition_counts=dict(Counter(r['kind'] for r in records)),
                    provenance_note='Existing slice authority retained; GUI geometry re-audited against current STL sources. This is not a new CAD build or physical qualification.',
                    projects=records)
    write(shelf / 'delivery_manifest.json', manifest)
    write_guide(shelf, entries, manifest)
    # Update delivery metadata only. The original slice records, CAD revision
    # and slice-source fingerprints retain their historical identity.
    legacy_path = shelf / 'release_manifest.json'
    legacy = read(legacy_path)
    legacy.setdefault('prior_delivery_manifest_sha256', legacy['manifest_sha256'])
    legacy['shelf_catalog'].update(sha256=sha256(catalog), selection_rules=read(catalog)['selection_rules'])
    legacy['gui_delivered_entries'] = {
        r['name']: {'kind': r['kind'], 'project': 'to_print/' + r['path'],
                    'project_sha256': r['sha256'], 'geometry_audit': r['geometry_audit'],
                    'slicing_required': True}
        for r in records if r['kind'] == 'gui_project'
    }
    legacy['delivery_metadata_note'] = manifest['provenance_note']
    from lx521_baffle.io import sha256_bytes
    import build_to_print_shelf as builder
    legacy['manifest_sha256'] = sha256_bytes(builder._canonical_json({k: v for k, v in legacy.items() if k != 'manifest_sha256'}))
    write(legacy_path, legacy)
    gui_report = shelf / 'obiwan' / LANES['petg_gf_gui'].directory / 'gui_projects.json'
    gui = read(gui_report)
    gui['projects'] = [dict(**r['settings_audit'], name=r['name'], project='to_print/' + r['path'],
                            project_sha256=r['sha256'], geometry_audit=r['geometry_audit'],
                            kind='plate' if 'combo' in r['name'] else 'single')
                       for r in records if r['kind'] == 'gui_project']
    write(gui_report, gui)
    return manifest


def validate(shelf, catalog=None, *, geometry=True):
    catalog = catalog or shelf / 'catalog.json'
    manifest = read(shelf / 'delivery_manifest.json')
    entries = read(catalog)['entries']
    inventory(shelf, entries)
    require(sha256(catalog) == manifest['catalog_sha256'], 'catalog changed since delivery refresh')
    expected = expected_projects(entries)
    require({Path(r['path']) for r in manifest['projects']} == set(expected), 'manifest project inventory differs from catalog')
    require(len(manifest['projects']) == len(expected), 'duplicate manifest projects')
    for r in manifest['projects']:
        path = shelf / r['path']
        entry, lane = expected[Path(r['path'])]
        require(r['kind'] == lane.kind.value and r['lane'] == lane.id, f'wrong disposition: {path}')
        require(sha256(path) == r['sha256'], f'delivery hash mismatch: {path}')
        source = canonical_source(ROOT, entry)
        require(str(source.relative_to(ROOT)) == r['source_stl'], f'wrong source binding: {path}')
        require(sha256(source) == r['source_stl_sha256'] == sha256(shelf / stl_path(entry)), f'source/STL mismatch: {path}')
        authority = r['source_authority']
        require(sha256(ROOT / authority['path']) == authority['sha256'], f'source authority changed: {path}')
        alias = r['delivered_authority']
        require(alias['path'] == str(authority_path(entry)) and sha256(shelf / alias['path']) == alias['sha256'], f'delivered orientation/plate authority changed: {path}')
        if lane.kind == DeliveryKind.GUI:
            for p, digest in r['geometry_audit']['sources'].items():
                require(sha256(ROOT / p) == digest, f'GUI source changed: {p}')
            if geometry:
                audit_gui_settings(ROOT, entry, path)
                audit_gui_geometry(ROOT, entry, path)
    require(manifest['choice_count'] == len(entries) and manifest['project_count'] == len(expected), 'incorrect manifest counts')
    require((shelf / 'FILE_GUIDE.md').read_text() == guide_text(entries, manifest), 'file guide is stale')
    return manifest


def guide_text(entries, manifest):
    rows = ['# Files and slicer estimates', '',
            'Generated from `catalog.json` and `delivery_manifest.json`; run `make delivery_refresh` to update.', '',
            f"{len(entries)} choices; {manifest['disposition_counts'].get('sliced_project', 0)} sliced projects and {manifest['disposition_counts'].get('gui_project', 0)} GUI projects across all lanes. These are alternatives, not a per-speaker part count.", '',
            'Times and grams are slicer estimates per job, including its encoded setup/purge where reported. GUI estimates remain pending until sliced. Multiply the chosen jobs by two for stereo.', '',
            '| Part / plate | Lane | State | Delivery | Time | Filament (g) |',
            '|---|---|---|---|---|---|']
    by_name = {e['name']: e for e in entries}
    for r in manifest['projects']:
        e = by_name[r['name']]
        rows.append(f"| [{e['description']}]({r['path']}) | {r['lane']} | {e['state']} | {'Slice in GUI' if r['kind']=='gui_project' else 'Sliced'} | {r['estimated_time'] or 'Pending slice'} | {r['estimated_filament_g'] or 'Pending slice'} |")
    return '\n'.join(rows) + '\n'


def write_guide(shelf, entries, manifest):
    (shelf / 'FILE_GUIDE.md').write_text(guide_text(entries, manifest))


def package(shelf, output):
    manifest = validate(shelf)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {p.relative_to(ROOT): p for p in shelf.rglob('*') if p.is_file() and p.name != '.DS_Store'}
    for directory in ('docs', 'images/generated/iso', 'qualification'):
        for p in (ROOT / directory).rglob('*'):
            if p.is_file() and not any(part.startswith('.') or part == '__cadgen__' for part in p.relative_to(ROOT).parts):
                files[p.relative_to(ROOT)] = p
    files[Path('README.md')] = ROOT / 'to_print/PACK_README.md'
    files.pop(Path('to_print/PACK_README.md'), None)
    for r in manifest['projects']:
        authority = ROOT / r['source_authority']['path']
        files[Path('source_authorities') / authority.relative_to(ROOT)] = authority
    checksums = {str(rel): sha256(p) for rel, p in files.items()}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel, path in sorted(files.items()):
            # write() follows symlinks: every archive member has real bytes.
            z.write(path, str(rel))
        z.writestr('SHA256SUMS.json', json.dumps(checksums, indent=2, sort_keys=True) + '\n')
        z.writestr('START_HERE.txt', 'Open docs/BUILD_GUIDE.md, then to_print/FILE_GUIDE.md. GUI projects require slicing and qualification. This print pack contains actual files, not source-tree symlinks. Verify with verify_package.py.\n')
        z.write(ROOT / 'scripts/verify_delivery_package.py', 'verify_package.py')
    from verify_delivery_package import verify
    verify(output)
    output.with_suffix(output.suffix + '.sha256').write_text(sha256(output) + '  ' + output.name + '\n')
    print(f'Package verified: {output} ({output.stat().st_size:,} bytes)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('refresh', 'validate', 'package'))
    parser.add_argument('--shelf', type=Path, default=ROOT / 'to_print')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/lx521-print-pack.zip')
    args = parser.parse_args()
    if args.command == 'refresh':
        result = refresh(args.shelf, args.shelf / 'catalog.json')
        print(f"Bound {result['project_count']} projects")
    elif args.command == 'validate':
        result = validate(args.shelf)
        print(f"Valid: {result['choice_count']} choices, {result['project_count']} projects")
    else:
        package(args.shelf, args.output)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f'Delivery failed: {exc}', file=sys.stderr)
        raise SystemExit(2)
