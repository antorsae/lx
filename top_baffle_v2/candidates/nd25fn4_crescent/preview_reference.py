"""Coarse source-geometry check; never promoted as a printable deliverable."""
import json
import numpy as np
import trimesh
import manifold3d as md
from PIL import Image,ImageDraw
import v4_model as m
from reference_outline import trace,REFERENCE
from review import render_object,mu10_reference
from mesh_ops import solid


def main():
    import render_mesh as r
    actor=r.actor
    def flat(*a,**kw):
        o=actor(*a,**kw);o.GetProperty().SetInterpolationToFlat();return o
    r.actor=flat
    def field(x,y,z):
        Y=z+m.INSTALLED_Y_OFFSET;Z=x+m.INSTALLED_Z_OFFSET
        bore=np.hypot(y,Y-m.interface.UM_CUTOUT[1])-41.
        recess=np.maximum(np.hypot(y,Y-m.interface.UM_CUTOUT[1])-49.3,14.3-Z)
        lm=np.maximum(np.hypot(y,Y-m.interface.L22_CUTOUT[1])-(m.interface.LM_VISIBLE_RING_R+.2),Z-m.UM_APRON_REAR_Z)
        return np.maximum.reduce(np.broadcast_arrays(m.housing_field(x,y,z),-bore,-recess,-lm))
    mesh=m.installed(m.extract_surface(field,[[-19.5,23],[-72,72],[298-m.INSTALLED_Y_OFFSET,76]],.65,m.special_x()))
    mesh.export(m.HERE/'assembly/reference_preview_only.stl')
    driver=mu10_reference()
    objects=[render_object(mesh,[66,141,189,255]),render_object(driver,[213,137,49,255])]
    for i in (0,1):objects.append(render_object(m.installed(m.retained.driver_mesh(i)),[238,148,43,255]))
    for name,camera in [('front',(250,0,-55.419)),('oblique',(230,-95,-15)),('side',(10,250,-55.419))]:
        r.render(objects,m.HERE/f'views/reference_preview_{name}.png',camera=camera,focus=(10,0,-55.419),scale=73,size=(1200,1200))
    data,_,_=trace();cx,cy=data['driver_pixel_center'];scale=data['pixels_per_mm']
    image=Image.open(REFERENCE).convert('RGB');draw=ImageDraw.Draw(image)
    for poly in solid(mesh).project().to_polygons():
        points=[(cx+x*scale,cy-(y-m.interface.UM_CUTOUT[1])*scale) for x,y in poly]
        draw.line(points+[points[0]],fill='#ff6171',width=2)
    image.save(m.HERE/'views/reference_preview_overlay.png')
    print('Coarse geometry preview and reference overlay ready',flush=True)

if __name__=='__main__':main()
