#!/usr/bin/env python3
"""Refresh native STEP review packets and the actual-mesh Dayton ND25FN-4 assembly views."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'build/h2c'
sys.path[:0]=[str(ROOT/'src')]
from lx521_baffle.io import sha256_file
from lx521_baffle.h2c.printing import write_json
from PIL import Image,ImageOps,ImageDraw,ImageFont
TOOLS=ROOT.parent/'.agents/skills/cad/scripts'
VIEWS=[('iso','Front three-quarter',[.8,.3,1]),
       ('opposite','Rear three-quarter',[-.8,.3,-1]),
       ('front','Front',[0,0,1]),('edge','Right side',[1,0,0])]


def upright_camera(label,direction):
    # Installed coordinates are X lateral, Y up, Z forward. The default
    # CAD cameras assume Z up, which made these speakers look laid flat.
    return dict(name=label,direction=direction,up=[0,1,0])


def main():
    manifest=WORK/'views/review_validation.json'
    inputs={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/'scripts/render_h2c_mesh_reviews.py',
        *sorted((WORK/'STEP').glob('*.step')),*sorted((WORK/'review_models').glob('*.py')),
        *sorted((ROOT/'to_print/h2c/STL').glob('*.stl')),
        *sorted((ROOT/'to_print/h2c/STL/dayton_nd25fn4').glob('*.stl'))]}
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get('inputs')==inputs and all((ROOT/p).exists() and sha256_file(ROOT/p)==h for p,h in old.get('outputs',{}).items()):
            print('H2C review packets match current geometry.');return
    targets=[f'review_models/{family}_{state}.py' for family in ('stock','slim') for state in ('no_floor_stand','floor_stand')]
    subprocess.run([sys.executable,str(TOOLS/'step'),'--mesh-tolerance','0.05',
        '--mesh-angular-tolerance','0.08',*targets],cwd=WORK,check=True)
    jobs=[]
    for target in targets:
        name=Path(target).stem
        jobs.append(dict(input=target,mode='view',outputs=[dict(path=str(WORK/'views'/f'{name}_{key}.png'),
            camera=upright_camera(label,direction),width=1100,height=1600)
            for key,label,direction in VIEWS],appearance=dict(floor=dict(enabled=False)),
            render=dict(viewLabels=False,padding=.06)))
        result=subprocess.run([sys.executable,str(TOOLS/'inspect'),'refs',target,'--facts','--planes','--positioning'],
            cwd=WORK,text=True,stdout=subprocess.PIPE,check=True)
        data=json.loads(result.stdout);assert data.get('ok'),target
        write_json(WORK/'views'/(name+'_inspection.json'),data)
    for family in ('stock','slim'):
        jobs.append(dict(input=f'STEP/h2c_{family}_upper_bmr.step',mode='view',
            outputs=[dict(path=str(WORK/'views'/f'{family}_upper_bmr.png'),
                camera=upright_camera('Front three-quarter',[.8,.3,1]),width=1000,height=1000)],
            appearance=dict(floor=dict(enabled=False)),render=dict(viewLabels=False,padding=.06)))
    write_json(WORK/'snapshot_jobs.json',dict(jobs=jobs))
    subprocess.run([sys.executable,str(TOOLS/'snapshot'),'--job','snapshot_jobs.json'],cwd=WORK,check=True)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',22)
    heading=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',28)
    sheet=Image.new('RGB',(1800,2630),'#f5f7f9');draw=ImageDraw.Draw(sheet)
    draw.text((24,16),'H2C — upright assemblies',fill='#263c4a',font=heading)
    keys=[Path(t).stem for t in targets];outputs=[]
    for row,name in enumerate(keys):
        y=65+520*row
        family,state=name.split('_',1)
        label=f'{family.title()} — '+('without floor stand' if state=='no_floor_stand' else 'with floor stand')
        draw.text((24,y),label,fill='#263c4a',font=heading)
        for col,(view,label,_) in enumerate(VIEWS):
            path=max((WORK/'views').glob(f'{name}_{view}_*.png'),key=lambda p:p.stat().st_mtime_ns)
            outputs.append(path);panel=ImageOps.contain(Image.open(path).convert('RGB'),(430,425))
            sheet.paste(panel,(450*col+(450-panel.width)//2,y+70))
            draw.text((450*col+24,y+39),label,fill='#263c4a',font=font)
    for col,family in enumerate(('stock','slim')):
        path=max((WORK/'views').glob(f'{family}_upper_bmr_*.png'),key=lambda p:p.stat().st_mtime_ns)
        outputs.append(path);panel=ImageOps.contain(Image.open(path).convert('RGB'),(860,400))
        sheet.paste(panel,(900*col+(900-panel.width)//2,2200))
        draw.text((900*col+24,2160),f'{family.title()} — BMR alternative',fill='#263c4a',font=heading)
    contact=WORK/'views/primary_review_contact_sheet.png';sheet.save(contact);outputs.append(contact)
    subprocess.run([sys.executable,'scripts/render_h2c_mesh_reviews.py'],cwd=ROOT,check=True)
    outputs+=list((WORK/'views').glob('H2C_Dayton_ND25FN4_*'))+list((WORK/'views').glob('*_inspection.json'))
    write_json(manifest,dict(status='generated',inputs=inputs,outputs={str(p.relative_to(ROOT)):sha256_file(p) for p in outputs},
        projection='orthographic',camera_up_axis='installed +Y',physical_geometry_changed=False))
    print('H2C native and actual-mesh review packets refreshed.',flush=True)


if __name__=='__main__':main()
