#!/usr/bin/env python3
"""Rebuild the current H2C release without invoking a printer or a remote host."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry():
    tasks=[]
    for family in ('stock','slim'):
        tasks += [(family,'lm',s) for s in ('no_floor_stand','floor_stand')]
        tasks += [(family,o,'no_floor_stand') for o in ('upper','upper_bmr')]
        tasks += [(family,o,'floor_stand') for o in ('lm_lower','lm_upper')]
    tasks += [('obiwan','core_lm_carrier',s) for s in ('no_floor_stand','floor_stand')]
    tasks += [('wing',f'{slug}:{side}','no_floor_stand') for slug in ('flat','graded') for side in ('left','right')]
    for family,owner,state in tasks:
        name=f'h2c_obiwan_wing_{owner.replace(":","_")}' if family=='wing' else f'h2c_{family}_{owner}'
        if owner.startswith('lm') or owner=='core_lm_carrier':name+='_'+state
        facts=ROOT/'build/h2c/STEP'/(name+'.facts.json')
        data=json.loads(facts.read_text()) if facts.exists() else {}
        current=bool(data) and all((ROOT/p).exists() and sha(ROOT/p)==digest for p,digest in data.get('source_files',{}).items()
                                  if p!='src/lx521_baffle/h2c/printing.py')
        for suffix,key,directory in [('.step','step_sha256','STEP'),('.stl','stl_sha256','STL')]:
            p=ROOT/f'build/h2c/{directory}'/(name+suffix)
            current=current and p.exists() and sha(p)==data.get(key)
        command=[sys.executable,'scripts/export_h2c_geometry.py',family,owner,'--state',state]
        if not current:subprocess.run(command,cwd=ROOT,check=True)
        if family in ('stock','slim'):
            blocker=ROOT/'build/h2c/support_blockers'/(name+'.stl')
            if not current or not blocker.exists():subprocess.run(command+['--support-only'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/verify_h2c_geometry.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/verify_h2c_obiwan_interfaces.py'],cwd=ROOT,check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['all','geometry','prepare','validate','review','docs'])
    ap.add_argument('--workers',type=int,choices=(1,2),default=2);args=ap.parse_args()
    if args.command in ('all','geometry'):geometry()
    if args.command in ('all','prepare'):
        subprocess.run([sys.executable,'scripts/build_h2c_release.py','prepare'],cwd=ROOT,check=True)
    if args.command in ('all','validate'):
        subprocess.run([sys.executable,'scripts/audit_h2c_print.py','--workers',str(args.workers)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'scripts/validate_h2c_catalog.py'],cwd=ROOT,check=True)
    if args.command in ('all','review'):
        subprocess.run([sys.executable,'scripts/review_h2c_geometry.py'],cwd=ROOT,check=True)
        comparison=[sys.executable,'scripts/gen_product_iso_matrix.py']
        for cell in ('tweeter_nd25fw4_crescent','tweeter_tebm35c10_4_vase',
                     'tweeter_tebm35c10_4_crescent','tweeter_tebm35c10_4_crescent_opposed',
                     'tweeter_nd25fn4_waveguide'):
            comparison+=['--cell',cell]
        for row in ('tweeter_row','obiwan_upper_row','obiwan_wing_row'):
            comparison+=['--row',row]
        subprocess.run(comparison,cwd=ROOT,check=True)
    if args.command in ('all','validate','docs'):
        subprocess.run([sys.executable,'scripts/document_h2c_release.py'],cwd=ROOT,check=True)


if __name__=='__main__':main()
