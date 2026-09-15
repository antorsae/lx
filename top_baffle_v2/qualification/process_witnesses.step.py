"""PETG-GF process witnesses, mm. No production mating geometry is changed.

One native LM route sector is cropped from the current promoted STEP.
A separate 30x24 support/skin witness has an accessible supported roof and
0.8/1.2/1.6 mm walls for destructive process comparison. It is not a strength
surrogate for the real carrier. STEP is the primary review assembly.
"""
from pathlib import Path
import hashlib, json
from build123d import Box, Compound, Pos, Rot, import_step, export_step, export_stl
ROOT=Path(__file__).resolve().parents[1]


def gen_step():
    source=import_step(str(ROOT/'build/no_floor_stand/obiwan_lm_split.step'))
    lm=next(c for c in source.children if 'bottom' in c.label)
    crop=lm & (Pos(92,149,5.65)*Box(60,62,25.3))
    if not crop.is_valid or len(crop.solids())!=1:
        raise RuntimeError('route crop must be one valid solid')
    crop=Rot(Z=26.0)*Rot(X=180)*crop
    bb=crop.bounding_box()
    crop=Pos(-bb.min.X,-bb.min.Y,-bb.min.Z)*crop
    crop.label='native_route_cover'
    base=Pos(15,12,1)*Box(30,24,2)
    # Two end walls leave an accessible tunnel for support removal.
    for x in (1,29):
        base=base.fuse(Pos(x,12,5)*Box(2,24,8))
    base=base.fuse(Pos(15,12,9.4)*Box(30,24,0.8))
    for y,thickness in ((4,.8),(12,1.2),(20,1.6)):
        base=base.fuse(Pos(15,y,13)*Box(26,thickness,6.4))
    base=Pos(100,0,0)*base
    base.label='supported_roof_and_wall_steps'
    if not base.is_valid or len(base.solids())!=1:
        raise RuntimeError('support witness must be one valid solid')
    return Compound(children=[crop,base],label='PETG-GF qualification process witnesses')


if __name__=='__main__':
    out=Path(__file__).parent
    assembly=gen_step()
    export_step(assembly,str(out/'process_witnesses.step'))
    records=[]
    for c in assembly.children:
        p=out/(c.label+'.stl')
        bb=c.bounding_box()
        printable=Pos(-bb.min.X,-bb.min.Y,-bb.min.Z)*c
        export_stl(printable,str(p),tolerance=.005,angular_tolerance=.08)
        records.append(dict(name=c.label,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            valid=c.is_valid,volume_mm3=c.volume,size_mm=list(c.bounding_box().size),
                            min_z_mm=c.bounding_box().min.Z))
    (out/'process_witnesses.json').write_text(json.dumps(dict(kind='qualification_only',parts=records,
        native_source_sha256=hashlib.sha256((ROOT/'build/no_floor_stand/obiwan_lm_split.step').read_bytes()).hexdigest()),indent=2)+'\n')
    print('Exported two valid process witnesses')
