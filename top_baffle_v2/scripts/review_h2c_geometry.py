#!/usr/bin/env python3
"""Refresh native STEP review packets and the actual-mesh V4 assembly views."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'build/h2c'
sys.path[:0]=[str(ROOT/'src')]
from lx521_baffle.io import sha256_file
from lx521_baffle.h2c.printing import write_json
from PIL import Image,ImageOps,ImageDraw,ImageFont
TOOLS=ROOT.parent/'.agents/skills/cad/scripts'


def main():
    manifest=WORK/'views/review_validation.json'
    inputs={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/'scripts/render_h2c_mesh_reviews.py',
        *sorted((WORK/'STEP').glob('*.step')),*sorted((WORK/'review_models').glob('*.py')),
        *sorted((ROOT/'to_print/h2c/STL').glob('*.stl')),
        *sorted((ROOT/'to_print/h2c/STL/v4').glob('*.stl'))]}
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get('inputs')==inputs and all((ROOT/p).exists() and sha256_file(ROOT/p)==h for p,h in old.get('outputs',{}).items()):
            print('H2C review packets match current geometry.');return
    targets=[f'review_models/{family}_{state}.py' for family in ('stock','slim') for state in ('no_floor_stand','floor_stand')]
    subprocess.run([sys.executable,str(TOOLS/'step'),'--force','--mesh-tolerance','0.05',
        '--mesh-angular-tolerance','0.08',*targets],cwd=WORK,check=True)
    jobs=[]
    for target in targets:
        name=Path(target).stem
        jobs.append(dict(input=target,mode='view',outputs=[dict(path=str(WORK/'views'/f'{name}_{key}.png'),camera=camera)
            for key,camera in [('iso','iso'),('opposite',dict(direction=[-1,1,-.8])),('front','top'),('edge','front')]],
            render=dict(viewLabels=True,padding=.12,sizeProfile='diagnostic')))
        result=subprocess.run([sys.executable,str(TOOLS/'inspect'),'refs',target,'--facts','--planes','--positioning'],
            cwd=WORK,text=True,stdout=subprocess.PIPE,check=True)
        data=json.loads(result.stdout);assert data.get('ok'),target
        write_json(WORK/'views'/(name+'_inspection.json'),data)
    for family in ('stock','slim'):
        jobs.append(dict(input=f'STEP/h2c_{family}_upper_bmr.step',mode='view',
            outputs=[dict(path=str(WORK/'views'/f'{family}_upper_bmr.png'),camera='iso')],
            render=dict(viewLabels=True,padding=.12,sizeProfile='diagnostic')))
    write_json(WORK/'snapshot_jobs.json',dict(jobs=jobs))
    subprocess.run([sys.executable,str(TOOLS/'snapshot'),'--job','snapshot_jobs.json'],cwd=WORK,check=True)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    sheet=Image.new('RGB',(1440,1600),'#f5f7f9');draw=ImageDraw.Draw(sheet)
    keys=[Path(t).stem for t in targets];outputs=[]
    for row,name in enumerate(keys):
        for col,view in enumerate(('iso','opposite','front','edge')):
            path=max((WORK/'views').glob(f'{name}_{view}_*.png'),key=lambda p:p.stat().st_mtime_ns)
            outputs.append(path);panel=ImageOps.contain(Image.open(path).convert('RGB'),(350,275))
            sheet.paste(panel,(360*col+(350-panel.width)//2,315*row+30))
            draw.text((360*col+10,315*row+8),f'{name.replace("_"," ")} / {view}',fill='#263c4a',font=font)
    for col,family in enumerate(('stock','slim')):
        path=max((WORK/'views').glob(f'{family}_upper_bmr_*.png'),key=lambda p:p.stat().st_mtime_ns)
        outputs.append(path);panel=ImageOps.contain(Image.open(path).convert('RGB'),(700,280));sheet.paste(panel,(720*col,1300))
        draw.text((720*col+10,1270),f'{family.title()} / BMR alternative',fill='#263c4a',font=font)
    contact=WORK/'views/primary_review_contact_sheet.png';sheet.save(contact);outputs.append(contact)
    subprocess.run([sys.executable,'scripts/render_h2c_mesh_reviews.py'],cwd=ROOT,check=True)
    outputs+=list((WORK/'views').glob('H2C_V4_*'))+list((WORK/'views').glob('*_inspection.json'))
    write_json(manifest,dict(status='generated',inputs=inputs,outputs={str(p.relative_to(ROOT)):sha256_file(p) for p in outputs},
        projection='orthographic',physical_geometry_changed=False))
    print('H2C native and actual-mesh review packets refreshed.',flush=True)


if __name__=='__main__':main()
