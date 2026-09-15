"""Before/after front views of actual source-derived UM and matching wings."""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont

import v4_model as model
from review import render_object,mu10_reference
from rebuild import sha
from validate import restored_print_parts


def main():
    import render_mesh as renderer
    original=renderer.actor
    def flat(*a,**kw):
        result=original(*a,**kw);result.GetProperty().SetInterpolationToFlat();return result
    renderer.actor=flat
    previous=json.loads((model.HERE/'superseded.json').read_text())['previous_front_wide_taper']
    archive=model.ROOT/previous
    source=archive/'STL'/model.BODY_FILE
    authority=source.with_suffix('.print.json')
    assert sha(source)==json.loads(authority.read_text())['stl_sha256']
    body=trimesh.load_mesh(source,process=True)
    body.apply_transform(np.linalg.inv(json.loads(authority.read_text())['source_to_stl_matrix']))
    lm,_=model.restored_core('no_floor_stand','lm')
    objects=[render_object(body,[66,141,189,255]),
             render_object(mu10_reference(),[213,137,49,255]),
             render_object(lm,[157,170,177,255])]
    inputs=[source,authority,archive/'wing_validation.json',
            model.HERE/'views/UM_outline_front.png',
            model.HERE/'STL'/model.BODY_FILE,
            model.HERE/'wing_validation.json']
    # Restore the archived service pieces in their original package datums.
    # The current M3 retainer differs from this historical M2 assembly.
    for name in ['02_Closed_Cap_PRINT_TWO.stl','03_Tweeter_Retainer_PRINT_TWO.stl']:
        old=archive/'STL'/name;new=model.HERE/'STL'/name
        part=trimesh.load_mesh(old,process=True);v=part.vertices.copy()
        if name.startswith('02_'):
            part.vertices=np.c_[-6.05-v[:,2],v[:,0],-31-v[:,1]]
            color=[52,108,153,255]
        else:
            part.vertices=np.c_[v[:,2]+.2,v[:,0],v[:,1]-31]
            color=[100,173,204,255]
        objects.append(render_object(model.installed(part),color))
        opposite=part.copy();opposite.apply_transform(np.diag([-1.,1.,-1.,1.]))
        objects.append(render_object(model.installed(opposite),color))
        inputs.extend([old,new])
    for i in (0,1):
        objects.append(render_object(model.installed(model.retained.driver_mesh(i)),[238,148,43,255]))
    wings=json.loads((archive/'wing_validation.json').read_text())
    for side in ['left','right']:
        name=f'V4_flat_{side}_UPPER.stl';path=archive/'STL/wings'/name
        assert sha(path)==wings['parts'][name]['stl_sha256']
        wing=trimesh.load_mesh(path,process=True)
        wing.apply_transform(np.linalg.inv(wings['parts'][name]['source_to_stl_matrix']))
        objects.append(render_object(wing,[145,168,179,255]));inputs.append(path)
    output=model.HERE/'views'
    before=output/'UM_outline_before.png'
    renderer.render(objects,before,camera=(250,0,-55.419),focus=(5,0,-55.419),
                    scale=73,size=(1450,1450))
    sheet=Image.new('RGB',(1860,1000),'#f5f7f9');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',30)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    for i,(path,label) in enumerate([(before,'Previous outline'),
                (output/'UM_outline_front.png','Reference-shaped outline')]):
        picture=Image.open(path).convert('RGB').resize((900,900),Image.Resampling.LANCZOS)
        sheet.paste(picture,(20+920*i,65));draw.text((40+920*i,20),label,fill='#203545',font=font)
    draw.text((40,970),'Blue: printed body   Orange: driver reference   Grey: LM and matching wings',
              fill='#425566',font=small)
    final=output/'UM_outline_comparison.png';sheet.save(final)
    report={'source_sha256':sha(__file__),
        'source_build_manifest_sha256':sha(model.HERE/'build_manifest.json'),
        'inputs':{str(p.relative_to(model.ROOT)):sha(p) for p in inputs},
        'outputs':{p.name:sha(p) for p in [before,final]},
        'camera':'Same orthographic front camera, scale and lighting for both actual STL assemblies.'}
    (output/'outline_comparison_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('UM outline comparison:',final,flush=True)


if __name__=='__main__':
    main()
