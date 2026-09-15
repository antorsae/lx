#!/usr/bin/env python3
"""Orthographic review of the actual continuous H2C wing/LM print meshes."""
from pathlib import Path
import json
import sys
import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
from lx521_baffle.io import sha256_file
from lx521_baffle.h2c.printing import write_json


def restored(path):
    authority=json.loads(path.with_suffix('.print.json').read_text())
    assert sha256_file(path)==authority['stl_sha256']
    mesh=trimesh.load_mesh(path,process=True)
    mesh.apply_transform(np.linalg.inv(np.asarray(authority['source_to_stl_matrix'])))
    return mesh


def actor(mesh,color):
    data=vtk.vtkPolyData();points=vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices,dtype=np.float32),deep=True));data.SetPoints(points)
    cells=vtk.vtkCellArray();values=np.c_[np.full(len(mesh.faces),3),mesh.faces].astype(np.int64)
    cells.SetCells(len(values),numpy_to_vtkIdTypeArray(values.ravel(),deep=True));data.SetPolys(cells)
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(data)
    item=vtk.vtkActor();item.SetMapper(mapper);item.GetProperty().SetColor(*(np.asarray(color)/255))
    item.GetProperty().SetInterpolationToFlat()
    return item


def render(parts,path,direction,size=(950,1400)):
    renderer=vtk.vtkRenderer();renderer.SetBackground(.96,.97,.98)
    for mesh,color in parts:renderer.AddActor(actor(mesh,color))
    bounds=np.asarray([m.bounds for m,c in parts]);lo=bounds[:,0].min(axis=0);hi=bounds[:,1].max(axis=0);focus=(hi+lo)/2
    direction=np.asarray(direction,dtype=float);direction/=np.linalg.norm(direction)
    up=np.array([0.,1.,0.]);right=np.cross(up,direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
    corners=np.asarray(np.meshgrid(*zip(lo,hi))).T.reshape(-1,3);xy=(corners-focus)@np.stack([right,up],axis=1)
    scale=1.10*max(abs(xy[:,1]).max(),abs(xy[:,0]).max()*size[1]/size[0])
    camera=renderer.GetActiveCamera();camera.SetPosition(*(focus+direction*1500));camera.SetFocalPoint(*focus);camera.SetViewUp(*up)
    camera.ParallelProjectionOn();camera.SetParallelScale(scale);renderer.ResetCameraClippingRange()
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetMultiSamples(8);window.SetSize(*size);window.AddRenderer(renderer)
    try:
        window.Render();capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.ReadFrontBufferOff();capture.SetInputBufferTypeToRGB();capture.Update()
        pixels=vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
        Image.fromarray(np.flipud(pixels.reshape(size[1],size[0],3)).copy()).save(path)
    finally:window.Finalize()
    return dict(projection='orthographic',direction=direction.tolist(),focus_mm=focus.tolist(),scale_mm=scale,image_sha256=sha256_file(path))


def main():
    work=ROOT/'build/h2c';out=work/'views';out.mkdir(exist_ok=True)
    lm=ROOT/'to_print/h2c/STL/h2c_obiwan_core_lm_carrier_no_floor_stand.stl'
    body=ROOT/'to_print/h2c/STL/v4/01_UM_Crescent_V4_PRINT.stl'
    manifest={}
    for slug in ('flat','graded'):
        sources=[lm,body]+[ROOT/f'to_print/h2c/STL/h2c_v4_wing_{slug}_{s}.stl' for s in ('left','right')]
        colors=[(133,151,163),(66,141,189),(213,216,219),(213,216,219)]
        parts=[(restored(p),c) for p,c in zip(sources,colors)]
        scene=trimesh.Scene()
        for source,(mesh,color) in zip(sources,parts):
            # Use original facets in both the review GLB and PNGs.
            preview=mesh.copy()
            preview.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=[*color,255],roughnessFactor=.75))
            preview.apply_scale(.001);scene.add_geometry(preview,node_name=source.stem,geom_name=source.stem)
        glb=out/f'H2C_V4_{slug}_assembly.glb';scene.export(glb)
        views={}
        for name,direction in [('front',[0,0,1]),('rear_oblique',[-.7,.2,-1])]:
            path=out/f'H2C_V4_{slug}_{name}.png';views[name]=render(parts,path,direction)
        manifest[slug]=dict(sources={str(p.relative_to(ROOT)):sha256_file(p) for p in sources},views=views,
            glb_sha256=sha256_file(glb),glb_units='metres',glb_is_decimated_preview=False)
        print('Rendered V4 continuous wings:',slug,flush=True)
    write_json(out/'mesh_review_manifest.json',manifest)


if __name__=='__main__':main()
