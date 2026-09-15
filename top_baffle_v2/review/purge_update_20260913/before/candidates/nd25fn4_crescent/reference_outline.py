"""Trace the supplied UM concept, registered by the visible driver flange.

Raster proportions are a design target, not physical driver measurements.
The actual STL comparison uses this independent image-derived outline.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import UnivariateSpline

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REFERENCE=ROOT/'design_inputs/nd25fn4_UM_reference_front_latest.png'
DRIVER_RADIUS=49.3
DRIVER_Y=366.081


def trace():
    rgb=np.asarray(Image.open(REFERENCE).convert('RGB')).astype(float)
    red,green,blue=rgb.transpose(2,0,1)
    def largest(mask):
        labels,_=ndimage.label(mask)
        sizes=np.bincount(labels.ravel());sizes[0]=0
        return labels==sizes.argmax()
    gold=largest((red>1.2*green)&(green>1.2*blue)&(red>50))
    boundary=gold&~ndimage.binary_erosion(gold)
    yy,xx=np.where(boundary)
    fit=np.linalg.lstsq(np.c_[2*xx,2*yy,np.ones(len(xx))],xx*xx+yy*yy,rcond=None)[0]
    cx,cy=fit[:2];radius=np.sqrt(fit[2]+cx*cx+cy*cy)
    scale=radius/DRIVER_RADIUS
    mask=largest((blue>green*1.025)&(green>red*1.1)&((blue-red)/np.maximum(blue,1)>.43))
    rows=[]
    for py in range(5,476):
        pixels=np.flatnonzero(mask[py])
        rows.append([DRIVER_Y-(py-cy)/scale,(pixels[-1]-pixels[0])/(2*scale)])
    rows=np.asarray(rows)[::-1]
    curve=UnivariateSpline(rows[:,0],rows[:,1],s=len(rows)*.16**2,k=3)
    return {'driver_pixel_center':[float(cx),float(cy)],'driver_pixel_radius':float(radius),
            'pixels_per_mm':float(scale),'rows_y_half_width_mm':rows.tolist()},curve,mask


def main():
    import hashlib
    data,curve,mask=trace()
    data['reference_sha256']=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    data['station_y_half_width_slope']= [[float(y),float(curve(y)),float(curve(y,1))]
        for y in [312.5,315.,318.,323.,330.,338.,346.,356.,366.081,376.,386.,396.,406.,413.,420.,427.]]
    path=HERE/'reference_outline_trace.json';path.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data['station_y_half_width_slope'],indent=2),flush=True)


if __name__=='__main__':main()
