"""Render actual sliced infill, support, plate layout and magnet pause sites."""
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import trimesh
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];WORK=ROOT/'review/nd25fn4_print'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from gcode_analysis import parse_gcode
from prepare_print import sha


def main():
    prep=json.loads((WORK/'preparation.json').read_text())['body']
    gcode=WORK/'body/plate_1.gcode'
    parsed=parse_gcode(gcode,retain_feature_prefixes=('Outer wall','Inner wall','Sparse infill','Internal solid','Support','Brim','Prime tower'))
    matrix=np.asarray(prep['source_to_stl_matrix']);matrix[:3,3]+=prep['offset'];inv=np.linalg.inv(matrix)
    out=HERE/'print';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(9.4,9.4),layout='constrained')
    for ax,target in zip(axes,[6.12,12.52]):
        layer=min(parsed.layers,key=lambda l:abs(l.z-target))
        groups={}
        for s in layer.segments:
            points=trimesh.transform_points(np.array([[s.x0,s.y0,layer.z],[s.x1,s.y1,layer.z]]),inv)[:,:2]
            if points[:,1].min()<300 or points[:,1].max()>558 or np.max(abs(points[:,0]))>80:continue
            if s.feature.startswith('Support'):color='#c1c7cc'
            elif 'infill' in s.feature.lower():color='#d7891d' if points[:,1].mean()>=421 else '#167580'
            elif s.feature.startswith(('Prime','Brim')):continue
            else:color='#384452'
            groups.setdefault(color,[]).append(points)
        for color,lines in groups.items():ax.add_collection(LineCollection(lines,colors=color,linewidths=.34,alpha=.95))
        ax.axhline(421,color='#9c5130',ls='--',lw=.8)
        ax.set(xlim=(-78,78),ylim=(303,558),aspect='equal',title=f'Actual slice at Z = {layer.z:.2f} mm',xlabel='Installed lateral X (mm)')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
    axes[0].set_ylabel('Installed height Y (mm)')
    fig.suptitle('ND25FN V4 • Regional infill\nTeal: UM 100% zig-zag   ·   Amber: tweeters 15% gyroid\nGrey: removable supports   ·   Six walls throughout',fontsize=13)
    fig.savefig(out/'infill_preview.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    chosen=min(parsed.layers,key=lambda l:abs(l.z-9.64))
    for feature,color in [('Outer wall','#326280'),('Inner wall','#89a8ba'),('Support','#dadada'),('Prime tower','#777777')]:
        lines=[[(s.x0,s.y0),(s.x1,s.y1)] for s in chosen.segments if s.feature.startswith(feature)]
        if lines:ax.add_collection(LineCollection(lines,colors=color,linewidths=.45))
    rows=json.loads((WORK/'body/magnet_discovery.json').read_text())
    for index,r in enumerate(sorted(rows,key=lambda r:r['name'])):
        x,y,_=r['center_bed_mm'];ax.add_patch(Circle((x,y),5,fill=False,color='#cf6730',lw=1.5))
        ax.annotate(f'M{index+1}',(x,y),xytext=(8,8),textcoords='offset points',color='#994214',weight='bold')
    ax.set(xlim=(0,256),ylim=(0,256),aspect='equal',xlabel='Bed X (mm)',ylabel='Bed Y (mm)',
           title='Body plate • pause before Z = 9.80 mm\nInsert four Ø6 × 3 mm magnets; then resume')
    ax.grid(alpha=.15);fig.savefig(out/'magnet_pause_map.png',dpi=170);plt.close(fig)
    (out/'views_manifest.json').write_text(json.dumps(dict(gcode_sha256=sha(gcode),
        images={name:sha(out/name) for name in ['infill_preview.png','magnet_pause_map.png']}),indent=2)+'\n')
    print('Rendered actual infill and magnet pause map')


if __name__=='__main__':main()
