"""Orthographic 3D close-ups of the measured, unmodified UM/LM assembly.

Use the delivered print body and the existing LM top at their installed
transforms. The PNGs show original full-resolution surfaces; only the camera
crops the scene. A separate review GLB is cropped outside the interface.
"""
from pathlib import Path
import json
import sys
import argparse

import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image,ImageDraw,ImageFont

from verify_lm_assembly import installed,sha
from mesh_ops import solid,to_trimesh
import manifold3d as md

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'views'
sys.path.insert(0,str(ROOT/'design_inputs/MU10_ND25FN_V4_Retained/source'))
import render_mesh

BG=(245,247,249)
BLUE=(59,143,193)
GREY=(163,176,183)
INK=(35,52,63)


def font(size,bold=False):
    path=Path('/System/Library/Fonts/Supplemental')/('Arial Bold.ttf' if bold else 'Arial.ttf')
    return ImageFont.truetype(str(path),size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',action='store_true')
    args=parser.parse_args()
    check_path=HERE/'print'/('LM_design_assembly_check.json' if args.design else 'LM_assembly_check.json')
    check=json.loads(check_path.read_text())
    parts=[]
    for name,row,color in [('UM + crescent',check['body'],BLUE),
                            ('LM top',check['configurations']['no_floor_stand']['LM'],GREY)]:
        path=ROOT/row['path'];assert sha(path)==row['sha256']
        mesh,source=installed(path);parts.append((name,mesh,color,source))
    renderer=vtk.vtkRenderer();renderer.SetBackground(*(c/255 for c in BG))
    for _,mesh,color,_ in parts:
        actor=render_mesh.actor(mesh,tuple(c/255 for c in color))
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().SetAmbient(.25)
        actor.GetProperty().SetDiffuse(.75)
        renderer.AddActor(actor)
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1)
    window.SetMultiSamples(8);window.AddRenderer(renderer)
    camera=renderer.GetActiveCamera()
    specs={
        'oblique':dict(title='ORTHOGRAPHIC 3D — FRONT / BELOW',direction=[.52,-.42,1],focus=[0,315,10],scale=30.,size=(1500,940)),
        'front':dict(title='FRONT — ORTHOGRAPHIC',direction=[0,0,1],focus=[0,318,10],scale=28.,size=(1400,850)),
        'side':dict(title='RIGHT SIDE — ORTHOGRAPHIC',direction=[1,0,0],focus=[0,315,10],scale=21.,size=(1050,850)),
    }
    images={};outputs={};cameras={}
    try:
        for name,spec in specs.items():
            focus=np.array(spec['focus']);direction=np.array(spec['direction'],dtype=float)
            direction/=np.linalg.norm(direction)
            camera.SetPosition(*(focus+1000*direction));camera.SetFocalPoint(*focus)
            camera.SetViewUp(0,1,0);camera.OrthogonalizeViewUp()
            camera.ParallelProjectionOn();camera.SetParallelScale(spec['scale'])
            renderer.ResetCameraClippingRange();width,height=spec['size'];window.SetSize(width,height);window.Render()
            capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.SetInputBufferTypeToRGB()
            capture.ReadFrontBufferOff();capture.Update()
            pixels=vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
            photo=Image.fromarray(np.flipud(pixels.reshape(height,width,3)).copy())
            panel=Image.new('RGB',(width,height+145),BG);panel.paste(photo,(0,95))
            draw=ImageDraw.Draw(panel)
            draw.text((30,20),spec['title'],fill=INK,font=font(29,True))
            draw.text((30,61),'Blue: current UM + crescent   |   Grey: existing LM top',fill=INK,font=font(21))
            footer='Front is left. Flush at Z18.30 mm; LM face remains uncovered.' if name=='side' else 'Actual print geometry, assembled position. No perspective or exploded spacing.'
            draw.text((30,height+107),footer,fill=INK,font=font(20))
            path=OUT/f'LM_interface_ortho_{name}.png';panel.save(path)
            images[name]=panel
            outputs[name]=dict(path=path.name,sha256=sha(path),pixels=list(panel.size))
            cameras[name]=dict(projection='orthographic',position_mm=list(camera.GetPosition()),
                              focal_point_mm=list(camera.GetFocalPoint()),up=list(camera.GetViewUp()),
                              parallel_scale_mm=camera.GetParallelScale(),size_px=list(spec['size']))
            print('Rendered',path.name,flush=True)
    finally:window.Finalize()
    # An image sheet keeps the 3D view prominent and includes true axial views.
    sheet=Image.new('RGB',(1900,1975),BG)
    draw=ImageDraw.Draw(sheet)
    draw.text((45,25),'UM / LM INTERFACE',font=font(40,True),fill=INK)
    draw.text((45,78),'Current assembly — orthographic 3D renders',font=font(24),fill=INK)
    main=images['oblique'].resize((1425,1031))
    sheet.paste(main,(240,118))
    front=images['front'].resize((1045,743));side=images['side'].resize((784,743))
    sheet.paste(front,(25,1200));sheet.paste(side,(1090,1200))
    path=OUT/'LM_interface_orthographic_3D.png';sheet.save(path)
    outputs['sheet']=dict(path=path.name,sha256=sha(path),pixels=list(sheet.size))
    scene=trimesh.Scene()
    crop=md.Manifold.cube((180,80,100)).translate((-90,275,-50))
    for name,mesh,color,_ in parts:
        detail=to_trimesh(solid(mesh)^crop)
        # Match the flat-surface PNG review. Averaging normals across the
        # long planar boolean triangles creates false ridges in the GLB.
        # Splitting render vertices changes no triangle positions or shape.
        detail.unmerge_vertices()
        detail.vertex_normals=np.repeat(detail.face_normals,3,axis=0)
        detail.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=name,baseColorFactor=[*color,255],metallicFactor=.02,roughnessFactor=.64))
        detail.apply_translation([0,-315,0]);detail.apply_scale(.001)
        scene.add_geometry(detail,node_name=name,geom_name=name)
    glb=OUT/'LM_interface_detail.glb';scene.export(glb,include_normals=True)
    exported=trimesh.load_scene(glb);assert len(exported.geometry)==2
    manifest=dict(source_script_sha256=sha(Path(__file__)),
                  source_render_helper_sha256=sha(Path(render_mesh.__file__)),
                  assembly_check_path=str(check_path.relative_to(ROOT)),assembly_check_sha256=sha(check_path),
                  sources={name:source for name,_,_,source in parts},
                  cameras=cameras,outputs=outputs,geometry_changed=False,
                  GLB=dict(path=glb.name,sha256=sha(glb),units='metres',up='+Y',
                           installed_origin_mm=[0,315,0],crop_installed_y_mm=[275,355],
                           note='Review-only cropped assembly with per-face normals; PNGs use the original full-resolution meshes.'))
    (OUT/'LM_interface_render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved 3D sheet and interactive detail',flush=True)


if __name__=='__main__':main()
