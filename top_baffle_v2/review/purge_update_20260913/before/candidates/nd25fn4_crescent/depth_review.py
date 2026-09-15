"""Actual-mesh views and radial sections for the UM depth revision."""
import json
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import v4_model as model
from review import render_object,mu10_reference
from validate import restored_print_parts
from validate_depth_magnets import depth_profile,shoulder_depths,constant_draft
from rebuild import sha


def main():
    import render_mesh as renderer
    actor=renderer.actor
    def flat(*a,**kw):
        result=actor(*a,**kw);result.GetProperty().SetInterpolationToFlat();return result
    renderer.actor=flat
    _,parts=restored_print_parts();body=model.installed(parts['housing'])
    previous=json.loads((model.HERE/'superseded.json').read_text())['previous_front_wide_taper']
    archived=model.ROOT/previous/'STL'/model.BODY_FILE
    authority=json.loads(archived.with_suffix('.print.json').read_text())
    assert sha(archived)==authority['stl_sha256']
    old=trimesh.load_mesh(archived,process=True)
    old.apply_transform(np.linalg.inv(authority['source_to_stl_matrix']))
    driver=mu10_reference();out=model.HERE/'views'
    outputs=[]
    for label,mesh in [('before',old),('after',body)]:
        objects=[render_object(mesh,[66,141,189,255]),render_object(driver,[213,137,49,255])]
        for i in (0,1):objects.append(render_object(model.installed(model.retained.driver_mesh(i)),[238,148,43,255]))
        path=out/f'UM_depth_{label}.png'
        renderer.render(objects,path,camera=(180,-135,-20),focus=(10,0,-55.419),scale=73,size=(1450,1450))
        outputs.append(path)
    for name,camera in [('UM_depth_side.png',(10,250,-55.419)),
                        ('UM_depth_front.png',(250,0,-55.419))]:
        path=out/name
        renderer.render(objects,path,camera=camera,focus=(10,0,-55.419),scale=73,size=(1450,1450))
        outputs.append(path)
    sheet=Image.new('RGB',(1860,1030),'#f5f7f9');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',29)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    for i,(p,title) in enumerate(zip(outputs[:2],['Previous outline',
                                                'New: traced reference outline'])):
        picture=Image.open(p).convert('RGB').resize((900,900),Image.Resampling.LANCZOS)
        sheet.paste(picture,(20+920*i,70));draw.text((25+920*i,23),title,fill='#203545',font=font)
    draw.text((35,987),'Actual STL geometry in the same camera. Blue: printed body. Orange: driver references.',fill='#425566',font=small)
    path=out/'UM_depth_comparison.png';sheet.save(path);outputs.append(path)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharey=True,layout='constrained')
    rows=[]
    for ax,angle,title in zip(axes,[0.,45.,-45.],['Middle','Upper diagonal shoulder','Lower diagonal / neck']):
        row={}
        for label,mesh,color,style in [('Previous',old,'#8e989e','--'),('New',body,'#328abe','-')]:
            p=depth_profile(mesh,angle);row[label]=p
            ax.plot(p['radius_mm'],p['front_z_mm'],style,color=color,lw=2,label=label)
            ax.plot(p['radius_mm'],p['rear_z_mm'],style,color=color,lw=1.5)
        ax.axhline(18.3,color='#bf862f',lw=1,label='Fixed driver rim')
        ax.set(title=title,xlabel='Distance from driver axis (mm)',xlim=(49.4,66),ylim=(-3,24))
        ax.grid(alpha=.2);ax.legend(fontsize=8);rows.append(row)
    axes[0].set_ylabel('Depth Z (mm); front is upward')
    fig.suptitle('Measured front and rear surfaces from the exported STLs',fontsize=13)
    path=out/'UM_depth_sections.png';fig.savefig(path,dpi=150);plt.close(fig);outputs.append(path)
    shoulders=shoulder_depths(body)
    draft=constant_draft(body)
    old_shoulders=shoulder_depths(old,validate_shape=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    for ax,row,previous_row in zip(axes,shoulders,old_shoulders):
        ax.plot(previous_row['lateral_half_width_mm'],previous_row['depth_z_mm'],
                color='#8e989e',lw=2,ls='--',label='Previous')
        ax.plot(row['lateral_half_width_mm'],row['depth_z_mm'],color='#328abe',lw=2.5,
                label="New: front wider")
        ax.axhline(8,color='#87929a',ls='--',lw=1)
        ax.axhline(18,color='#bf862f',ls='--',lw=1)
        ax.set(title=row['section'].replace('_',' ').title()+f" — Y={row['vertical_y_mm']:.1f}",
               xlabel='Right outline X (mm)',ylabel='Depth Z (mm); front is upward',ylim=(3,23))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f"Front wider than rear — local side draft {draft['minimum_actual_draft_deg']:.1f}–{draft['maximum_actual_draft_deg']:.1f}° along the outline",fontsize=13)
    path=out/'UM_depth_taper_sections.png';fig.savefig(path,dpi=150);plt.close(fig);outputs.append(path)
    report={'source_sha256':sha(__file__),'source_build_manifest_sha256':sha(model.HERE/'build_manifest.json'),
        'inputs':{str(p.relative_to(model.ROOT)):sha(p) for p in [archived,archived.with_suffix('.print.json'),
            model.HERE/'STL'/model.BODY_FILE,model.HERE/'validate_depth_magnets.py']},
        'outputs':{p.name:sha(p) for p in outputs},'actual_STL_sections':rows,
        'actual_shoulder_sections':shoulders,'local_draft':draft,
        'previous_shoulder_sections':old_shoulders,
        'middle_front_minus_rear_full_width_mm':shoulders[1]['front_minus_rear_full_width_mm'],
        'note':'Identical cameras/lighting. No cosmetic smoothing or image generation; line sections measured from actual triangles.'}
    (out/'depth_review_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Depth comparison and measured sections:',out/'UM_depth_comparison.png',flush=True)


if __name__=='__main__':main()
