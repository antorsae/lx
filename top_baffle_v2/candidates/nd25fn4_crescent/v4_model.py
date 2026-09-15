"""Compact V4 with an organic UM exterior and verified Obi-Wan mounts (mm).

The lower flare and free UM shell are reshaped. The LM interface and UM
driver mounting features are retained from the verified source mesh.
Package axes: X forward, Y lateral, Z up. Project: X lateral, Y up, Z forward.
"""
from functools import lru_cache
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
from scipy.interpolate import BPoly, CubicSpline, RectBivariateSpline
from scipy.spatial import cKDTree
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "design_inputs/MU10_ND25FN_V4_Retained"
sys.path.insert(0, str(PACKAGE / "source"))
sys.path.insert(0, str(ROOT / "src"))
os.environ["LX_ROUTING_PROFILE"] = "obiwan"
# Pin the established upper gallery profile; the optional stand is an LM
# choice and must never silently alter this shared UM/tweeter part.
os.environ["LX_STAND_FOOT"] = "1"
import mechanical_v4 as retained
from lx521_baffle.obiwan import carriers as interface
from lx521_baffle.obiwan import route
from lx521_baffle.print_policy import policy
from mesh_ops import solid, to_trimesh, preserve_void_winding

# Adapt the immutable design input to the project's shared M3 convention.
# The larger bolt circle preserves 1.6 mm beside the driver flange cavity.
# A 27.95 mm retainer still passes through the unchanged 28.2 mm cap entry.
_hardware = policy()['hardware']
_m3, _retainer = _hardware['M3_insert'], _hardware['crescent_retainer']
retained.M = replace(retained.M,
    service_radius=_retainer['service_radius_mm'],
    retainer_outer_radius=_retainer['retainer_outer_radius_mm'],
    screw_circle_radius=_retainer['screw_circle_radius_mm'],
    screw_clearance_radius=_m3['clearance_diameter_mm']/2,
    screw_head_radius=_retainer['head_diameter_mm']/2,
    screw_head_height=_retainer['head_height_mm'],
    screw_length=_retainer['screw_length_mm'],
    insert_pilot_radius=_m3['pilot_diameter_mm']/2,
    insert_length=_m3['insert_length_mm'],
    insert_tip_x=retained.M.service_end_x+_m3['insert_length_mm'],
    insert_tip_relief_radius=_m3['clearance_diameter_mm']/2)

MESH_STEP = .30
FLARE_SLOPE_FACTOR = 1.5
FRONT_LIGAMENT = 2.4
THROAT_R = retained.P.throat_diameter / 2
LOWER_MOUTH_REACH = THROAT_R + (retained.P.mouth_height/2 - THROAT_R)/FLARE_SLOPE_FACTOR
LOWER_AXIS_Y = interface.UM_CUTOUT[1] + interface.UM_RECESS_R + FRONT_LIGAMENT + LOWER_MOUTH_REACH
INSTALLED_Y_OFFSET = LOWER_AXIS_Y + retained.P.center_spacing/2
PREVIOUS_Y_OFFSET = 501.781
LOWERING_MM = PREVIOUS_Y_OFFSET - INSTALLED_Y_OFFSET
INSTALLED_Z_OFFSET = interface.THICKNESS_MM - retained.P.overall_depth/2
PACKAGE_TO_INSTALLED = np.array([
    [0.,1.,0.,0.], [0.,0.,1.,INSTALLED_Y_OFFSET],
    [1.,0.,0.,INSTALLED_Z_OFFSET], [0.,0.,0.,1.]])
LM_STATES = ("no_floor_stand", "floor_stand")
UM_SOURCE_STATE = "no_floor_stand"
BODY_FILE = "01_UM_Crescent_V4.stl"
MAGNET_SKIN = .82
WING_GAP = .35
MAGNET_Z = 13.4
MAGNET_DIAMETER = 6.0
MAGNET_DEPTH = 3.0
MAGNET_CAVITY_DIAMETER = 6.2
MAGNET_CAVITY_DEPTH = 3.1
MAGNET_ANGLES = (12.,168.,-12.,192.)
UM_MAGNET_COUNT = len(MAGNET_ANGLES)
UM_OUTER_R = 63.8
UM_SCULPTED_HALF_WIDTH = 63.15
UM_REFERENCE_EDGE_ALLOWANCE = .65
UM_RIM_RISE = 2.4
UM_RIM_RUN_FRACTION = .95
UM_DRAFT_SLOPE = .50
UM_EDGE_ROUNDING = 2.4
UM_LM_FRONT_Z = 18.3
UM_LM_BLEND_START_Y = 322.0
UM_LM_BLEND_END_Y = 336.0
UM_REFERENCE_WIDTH_JETS = ((313.2, 42.05, 0.0, -0.2222222222222222), (315.0, 41.54804210575331, -0.16930673888599562, 0.0808009845607481), (318.0, 41.41575167341984, 0.0838214033804613, 0.08217280991349407), (323.0, 42.604041273978474, 0.3398989287905222, 0.020258200250530305), (330.0, 45.15789833562941, 0.40077031096252985, 0.01901038276661596), (338.0, 49.31566721799836, 0.6815811778968349, 0.05119233396696033), (346.0, 55.83596798911203, 0.796328635315128, -0.047487886767033693), (356.0, 61.10146788807533, 0.3025423965422797, -0.0365358607082822), (366.081, 62.50049776069863, -0.00458915214132184, -0.02439689371758845), (376.0, 61.1822747998293, -0.2983598541063769, -0.04306490101932409), (386.0, 55.57761119657102, -0.8382274331202919, -0.04018679108204351), (396.0, 46.59672002357339, -0.8980647209215373, -0.0040873745734557616), (406.0, 37.581188375247734, -0.8229494109323721, 0.10612324732251043), (413.0, 36.76876714380007, 0.09762236500414272, 0.09980007846098814), (420.0, 40.38520390744726, 1.0053564655127658, 0.15955252168433276), (427.0, 51.81971423362676, 2.3313576685848014, 0.21930496490767737), (436.0, 65.5, 0.5, -0.06), (445.0, 67.0, 0.0, 0.0))
CONNECT_START_Y = 326.0
CONNECT_END_Y = 439.0
CONNECT_OVERLAP_Y = 322.0
GALLERY_R = 45.6
GALLERY_Z = 6.2
UM_MECHANICAL_R = 49.4
WAIST_BLEND = 16.0
REAR_LOFT_START_Y = 380.0
REAR_LOFT_END_OFFSET = 28.65
REAR_LOFT_SHOULDER_CURVE = .01
REAR_LOFT_TABLE_STEP = .25
REAR_LOFT_Y_BOUNDS = (378.,464.)
REAR_LOFT_LATERAL_LIMIT = 66.
REAR_LOFT_CAP_R = 28.2
LOWER_REAR_START_Y = 314.0
LOWER_REAR_END_Y = 334.0
LOWER_REAR_SEAM_Z = 4.5
LOWER_REAR_EAR_FLAT_R = 4.95
LOWER_REAR_EAR_BLEND_R = 12.5
LOWER_REAR_SIDE_FADE = (36.0, 44.0)

def smooth01(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10+t*(-15+6*t))


def installed(mesh):
    mesh = mesh.copy(); mesh.apply_transform(PACKAGE_TO_INSTALLED)
    return mesh


def package_frame(mesh):
    mesh = mesh.copy(); mesh.apply_transform(np.linalg.inv(PACKAGE_TO_INSTALLED))
    return mesh


def restored_core(state, owner="um"):
    """Restore the exact released print mesh and verify its authority."""
    index = 2 if owner == "um" else 1
    path = ROOT/f"build/{state}/stl/obiwan_core_{index}_of_2_{owner}_carrier.stl"
    sidecar = path.with_suffix(".print.json")
    authority = json.loads(sidecar.read_text())
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert checksum == authority["stl_sha256"], (state,owner,"stale print authority")
    mesh = trimesh.load_mesh(path, process=True)
    mesh.apply_transform(np.linalg.inv(np.array(authority["source_to_stl_matrix"])))
    return mesh, {"source_stl": str(path.relative_to(ROOT)), "stl_sha256": checksum,
                  "print_authority_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()}


@lru_cache(None)
def connector_curve():
    """C2 buried route through the narrower reference-derived shoulders.

    Pass below the two right-hand pilot floors, rise within the middle
    annulus, then ease into the retained T trunk. The longer final turn
    preserves bend radius without changing the LM or tweeter interfaces.
    """
    p=route.ts_cable_points(.08);p=p[p[:,1]>310]
    original=CubicSpline(p[:,1],p[:,[0,2]])
    def arc(y,depth):
        dy=y-interface.UM_CUTOUT[1];x=np.sqrt(GALLERY_R**2-dy**2)
        return [np.array([x,depth]),np.array([-dy/x,0.]),
                np.array([-GALLERY_R**2/x**3,0.])]
    curve=BPoly.from_derivatives(
        [CONNECT_START_Y,344.,interface.UM_CUTOUT[1],390.,406.,421.,CONNECT_END_Y],
        [[original(CONNECT_START_Y,i) for i in range(3)],
         arc(344.,GALLERY_Z),arc(interface.UM_CUTOUT[1],9.5),arc(390.,9.5),
         [np.array([26.,5.9]),np.array([-.35,-.25]),np.array([.06,.02])],
         [np.array([27.7,4.5]),np.array([.5,-.25]),np.array([0.,.015])],
         [np.array([34.,INSTALLED_Z_OFFSET]),np.zeros(2),np.zeros(2)]])
    # A broad C2 adjustment maintains both the upper shoulder cover and
    # the driver-bore wall; it is zero in value/slope/curvature at joins.
    source=curve.c;coefficients=np.zeros((7,*source.shape[1:]))
    coefficients[0]=source[0];coefficients[-1]=source[-1]
    for k in range(1,6):
        coefficients[k]=(k/6)*source[k-1]+(1-k/6)*source[k]
    coefficients[3,3]+=3.2*np.array([-.35,.8])
    return original,BPoly(coefficients,curve.x)


def connector_radius(vertical):
    """Smooth UM passage with pilot clearance, matching the retained taper.

    The join is above the retained trunk's full-diameter section. Match
    its local radius and derivatives instead of forcing R3.4 at the join.
    """
    vertical=np.asarray(vertical)
    package_z=vertical-INSTALLED_Y_OFFSET
    t=np.clip((-package_z-39.)/10.,0.,1.)
    retained_radius=retained.M.branch_radius+(retained.M.trunk_radius-retained.M.branch_radius)*t*t*(3.-2.*t)
    weight=smooth01((vertical-425.)/(CONNECT_END_Y-425.))
    pilot_relief=.3*smooth01((vertical-394.)/4.)*(1-smooth01((vertical-407.)/14.))
    return 3.+(retained_radius-3.)*weight-pilot_relief


@lru_cache(None)
def wiring():
    original,curve = connector_curve()
    vertical = np.linspace(CONNECT_START_Y,CONNECT_END_Y,1601)
    xz = curve(vertical)
    join = np.c_[xz[:,0],vertical,xz[:,1]]
    tail_y = np.linspace(CONNECT_OVERLAP_Y,CONNECT_START_Y,81)
    tail = np.c_[original(tail_y)[:,0],tail_y,original(tail_y)[:,1]]
    conn = np.r_[tail[:-1],join]
    package = trimesh.transform_points(conn,np.linalg.inv(PACKAGE_TO_INSTALLED))
    radii = connector_radius(conn[:,1])
    old,old_r = retained.ROUTES["upper_and_trunk"]
    keep = old[:,2] > CONNECT_END_Y-INSTALLED_Y_OFFSET
    routes = {"upper_and_trunk":(np.r_[old[keep],package[::-1]],np.r_[old_r[keep],radii[::-1]]),
              "lower_branch":retained.ROUTES["lower_branch"]}
    return routes,{k:cKDTree(v[0]) for k,v in routes.items()},package,radii


def ducts_field(x,y,z):
    x,y,z = np.broadcast_arrays(x,y,z)
    out = np.full(x.shape,100.,dtype=float)
    mask = (abs(x)<16)&(y>10)&(y<59)&(z>CONNECT_OVERLAP_Y-INSTALLED_Y_OFFSET-4)&(z<29)
    if np.any(mask):
        q=np.c_[x[mask],y[mask],z[mask]]
        routes,trees,_,_=wiring()
        fields=[retained.tube_distance(q,c,r,trees[k]) for k,(c,r) in routes.items()]
        out[mask]=retained.smooth_union(*fields,retained.M.duct_union_blend)
    return out


def lower_horn_field(x,y,z):
    """Compress only the lower flare expansion, keeping the circular throat."""
    a,b=retained.v3.inner_radius(x,retained.P)
    b=THROAT_R+(b-THROAT_R)*(1-(1-1/FLARE_SLOPE_FACTOR)*smooth01(-z/THROAT_R))
    radial=(np.sqrt((y/a)**2+(z/b)**2)-1)*np.minimum(a,b)
    return np.maximum(radial,retained.P.overall_depth/2-retained.P.waveguide_depth-x)


def tweeter_radial(x,y,z):
    p=retained.P
    local=z+p.center_spacing/2
    a,b=retained.v3.outer_radius(x,p)
    shift=retained.v3.outer_center_shift(x,p)
    compression=(p.mouth_height/2-LOWER_MOUTH_REACH)*smooth01((x-9.1)/8.8)
    b=b-compression*smooth01(-(local-shift)/THROAT_R)
    lower=(np.sqrt((y/a)**2+((local-shift)/b)**2)-1)*np.minimum(a,b)
    upper=retained.v3.outer_radial_field(-x,y,-z+p.center_spacing/2,p)
    radial=retained.smooth_union(lower,upper,p.blend_radius)
    return radial


def tweeter_outer(x,y,z):
    return np.maximum(tweeter_radial(x,y,z),abs(x)-retained.P.overall_depth/2)


def um_rear(vertical, lateral=0.):
    """Continuous rear thickness; the only low flat lands are the LM ears.

    The old broad Z=12.4 bound exposed the Z=6.8 rectangular native web.
    Roll away from each actual receiver land with zero slope/curvature at
    both ends, into a shallow axial sweep shared by the entire lower band.
    Outside this local patch the established UM/upper rear surface is exact.
    """
    old=12.4-10.4*smooth01((vertical-320)/10)
    center=2.+(LOWER_REAR_SEAM_Z-2.)*(1-smooth01(
        (vertical-LOWER_REAR_START_Y)/(LOWER_REAR_END_Y-LOWER_REAR_START_Y)))
    distance=np.minimum(np.hypot(lateral-32.,vertical-interface.JOINT_EAR_Y),
                        np.hypot(lateral+32.,vertical-interface.JOINT_EAR_Y))
    ear_weight=1-smooth01((distance-LOWER_REAR_EAR_FLAT_R)/(
        LOWER_REAR_EAR_BLEND_R-LOWER_REAR_EAR_FLAT_R))
    curved=center+(12.4-center)*ear_weight
    side_weight=1-smooth01((abs(lateral)-LOWER_REAR_SIDE_FADE[0])/(
        LOWER_REAR_SIDE_FADE[1]-LOWER_REAR_SIDE_FADE[0]))
    return old+(curved-old)*side_weight-19.5*smooth01((vertical-392)/34)


@lru_cache(None)
def um_reference_curve():
    rows=np.array(UM_REFERENCE_WIDTH_JETS,dtype=float)
    return BPoly.from_derivatives(rows[:,0],
        [[w+UM_REFERENCE_EDGE_ALLOWANCE,d,dd] for _,w,d,dd in rows])


@lru_cache(None)
def um_reference_contour():
    """Image-scaled side splines, a bowed base and tangent round corners."""
    curve=um_reference_curve()
    # Keep the artificial closing edge farther away than any rear draft
    # offset. A cap at Y445 becomes an unintended sloping cutter at Y430.
    vertical=np.arange(313.2,520.0001,.2)
    right=np.c_[curve(np.minimum(vertical,445.)),vertical]
    outer=42.05+UM_REFERENCE_EDGE_ALLOWANCE
    corner_r=4.5;cx=outer-corner_r;bottom=313.2-corner_r
    x=np.linspace(-cx,cx,201)
    y=310.15-(310.15-bottom)*smooth01((abs(x)-7.)/(32.-7.))
    base=np.c_[x,y]
    angle=np.linspace(-np.pi/2,0,71)
    corner=np.c_[cx+corner_r*np.cos(angle),313.2+corner_r*np.sin(angle)]
    # Counter-clockwise continuous outline; the far top closes outside the
    # UM shaping zone and is never used to truncate the tweeter.
    return np.r_[base,corner[1:],right[1:],
        np.c_[-right[::-1,0],right[::-1,1]],
        np.c_[-corner[::-1,0],corner[::-1,1]][1:]]


@lru_cache(None)
def um_reference_distance_table():
    """True 2-D distance, interpolated smoothly for constant normal draft."""
    import shapely
    points=um_reference_contour()
    key=hashlib.sha256(points.tobytes()+b'reference-sdf-v1-.18').hexdigest()
    path=HERE/'assembly'/f'um_reference_distance_{key}.npz'
    if path.exists():
        data=np.load(path);xx=data['x'];yy=data['y'];distance=data['d']
    else:
        xx=np.arange(-75.,75.001,.18);yy=np.arange(298.,446.001,.18)
        x,y=np.meshgrid(xx,yy,indexing='ij')
        polygon=shapely.Polygon(points)
        assert polygon.is_valid
        q=shapely.points(x.ravel(),y.ravel())
        magnitude=shapely.distance(q,polygon.boundary).reshape(x.shape)
        distance=np.where(shapely.contains_xy(polygon,x,y),-magnitude,magnitude)
        temporary=path.with_name(path.stem+f'.{os.getpid()}.npz')
        np.savez_compressed(temporary,x=xx,y=yy,d=distance);os.replace(temporary,path)
    return RectBivariateSpline(xx,yy,distance,kx=3,ky=3,s=0)


def um_outline_distance(lateral,vertical):
    lateral,vertical=np.broadcast_arrays(lateral,vertical)
    return um_reference_distance_table().ev(lateral.ravel(),vertical.ravel()).reshape(lateral.shape)


def um_rim_height(vertical):
    # Meet the LM at its native front plane, then recover the approved bowl
    # with zero slope and curvature at both ends of this short transition.
    lower=smooth01((vertical-UM_LM_BLEND_START_Y)/(UM_LM_BLEND_END_Y-UM_LM_BLEND_START_Y))
    return UM_RIM_RISE*lower*(1-smooth01((vertical-396.)/22.))


def um_envelope(x,y,z,rear_override=None):
    """Wide front, narrow rear: one draft, gently rounded at its ends.

    A depth-linear offset replaces the rear-wide elliptical turnover.
    The normalized plan distance makes the draft comparable through the
    neck, middle and upper shoulder despite their different plan normals.
    Only the short lower transition fades into the exact LM attachments.
    """
    vertical=z+INSTALLED_Y_OFFSET;front=x+INSTALLED_Z_OFFSET
    rear=um_rear(vertical,y) if rear_override is None else rear_override
    # Retain a gentle lean even around the small lower mounting apron;
    # the free shoulders use the full, constant 26.6 degree draft.
    weight=.45+.55*smooth01((vertical-319.)/14.)
    side=um_outline_distance(y,vertical)+UM_DRAFT_SLOPE*weight*(18.3+um_rim_height(vertical)-front)
    legacy_cap=np.hypot(y,vertical-interface.UM_CUTOUT[1])-UM_OUTER_R
    activation=smooth01((vertical-400.)/15.)
    side=np.maximum(side,legacy_cap-(1-activation)*100.)
    face=front-18.3-um_front_rise(y,vertical)
    rounding=UM_EDGE_ROUNDING*weight
    front_join=-retained.smooth_union(-side,-face,np.maximum(rounding,1e-8))
    return -retained.smooth_union(-front_join,front-rear,np.maximum(rounding,1e-8))


def um_front_rise(lateral,vertical):
    """Broad bowl face, easing into the flush native LM mounting plane."""
    radius=np.hypot(lateral,vertical-interface.UM_CUTOUT[1])
    width=um_reference_curve()(np.clip(vertical,313.2,445.))
    outer_radius=np.hypot(width,vertical-interface.UM_CUTOUT[1])
    run=np.maximum(outer_radius-UM_MECHANICAL_R,1.)
    radial=smooth01((radius-UM_MECHANICAL_R)/(run*UM_RIM_RUN_FRACTION))
    apron=1-smooth01((vertical-320.)/14.)
    return (radial+(1-radial)*apron)*um_rim_height(vertical)


def rear_height(lateral,vertical,field=tweeter_outer):
    """First rear exterior height, in installed coordinates (no holes)."""
    lateral,vertical=np.broadcast_arrays(lateral,vertical)
    low=np.full(lateral.shape,INSTALLED_Z_OFFSET-retained.P.overall_depth/2)
    # Bracket at the widest tweeter section. Z4.5 lies outside the broad
    # shoulder: its old fallback created discontinuous endpoint derivatives
    # and rear-loft excursions of hundreds of millimetres near X64.
    high_z=INSTALLED_Z_OFFSET+retained.P.overall_depth/2-retained.P.rim_roll
    high=np.full(lateral.shape,high_z)
    valid=field(high-INSTALLED_Z_OFFSET,lateral,vertical-INSTALLED_Y_OFFSET)<0
    for _ in range(30):
        mid=(low+high)/2
        inside=field(mid-INSTALLED_Z_OFFSET,lateral,vertical-INSTALLED_Y_OFFSET)<0
        low=np.where(inside,low,mid);high=np.where(inside,mid,high)
    return np.where(valid,(low+high)/2,high_z)


def rear_height_derivatives(lateral,vertical,field):
    # This interval resolves the broad surface rather than amplifying
    # numerical interpolation noise in second derivatives.
    step=.75
    height=rear_height(lateral,vertical,field)
    low=rear_height(lateral,vertical-step,field)
    high=rear_height(lateral,vertical+step,field)
    return height,(high-low)/(2*step),(high-2*height+low)/step**2


@lru_cache(None)
def rear_loft_table():
    """One rear height surface, with matched value/slope/curvature.

    Evaluate the side draft at actual depth. Axially warping a clipped
    side wall creates folds as the reference waist narrows; a direct
    height surface meets that inclined wall through the ordinary rim roll.
    """
    lateral=np.arange(-REAR_LOFT_LATERAL_LIMIT,REAR_LOFT_LATERAL_LIMIT+.01,
                      REAR_LOFT_TABLE_STEP)
    vertical=np.arange(REAR_LOFT_Y_BOUNDS[0],REAR_LOFT_Y_BOUNDS[1]+.01,
                       REAR_LOFT_TABLE_STEP)
    lat,up=np.meshgrid(lateral,vertical,indexing='ij')
    end=LOWER_AXIS_Y-REAR_LOFT_END_OFFSET+REAR_LOFT_SHOULDER_CURVE*lateral**2
    # Keep endpoint and derivative samples inside the tweeter footprint,
    # including the outermost shoulders. Its centre is widest at every depth.
    end+=(LOWER_AXIS_Y-end)*smooth01((abs(lateral)-52.)/14.)
    span=end-REAR_LOFT_START_Y
    # The UM rear is exactly Z2 at the start, independent of the outline.
    # Derivatives outside a clipped silhouette are not valid height data.
    h0=np.full(lateral.shape,2.);d0=np.zeros_like(h0);c0=np.zeros_like(h0)
    h1,d1,c1=rear_height_derivatives(lateral,end,tweeter_outer)
    coefficients=np.array([h0,h0+d0*span/5,h0+2*d0*span/5+c0*span**2/20,
                            h1-2*d1*span/5+c1*span**2/20,h1-d1*span/5,h1])
    t=np.clip((up-REAR_LOFT_START_Y)/span[:,None],0,1)
    target=sum(c[:,None]*factor*t**i*(1-t)**(5-i)
               for i,(c,factor) in enumerate(zip(coefficients,[1,5,10,10,5,1])))
    pure_t=rear_height(lat,up,tweeter_outer)
    target=np.where(up>=end[:,None],pure_t,target)
    target=np.where(up<=REAR_LOFT_START_Y,um_rear(up,lat),target)
    return RectBivariateSpline(lateral,vertical,target,kx=3,ky=3,s=0)


def envelope_field(x,y,z):
    """Continuous rear loft joined to the actual inclined side surface."""
    x,y,z=np.broadcast_arrays(x,y,z)
    front=x+INSTALLED_Z_OFFSET;vertical=z+INSTALLED_Y_OFFSET
    rear=np.asarray(um_rear(vertical,y)).copy()
    mask=(vertical>REAR_LOFT_Y_BOUNDS[0])&(vertical<REAR_LOFT_Y_BOUNDS[1])&(abs(y)<REAR_LOFT_LATERAL_LIMIT)
    if np.any(mask):rear[mask]=rear_loft_table().ev(y[mask],vertical[mask])
    tweeter=tweeter_outer(x,y,z)
    combined=retained.smooth_union(tweeter,um_envelope(x,y,z,rear),WAIST_BLEND)
    reference_side=um_outline_distance(y,vertical)+UM_DRAFT_SLOPE*smooth01((vertical-319.)/14.)*(18.3+um_rim_height(vertical)-front)
    clip_weight=smooth01((vertical-390.)/10.)*(1-smooth01((vertical-430.)/12.))
    combined+=clip_weight*(np.maximum(combined,reference_side)-combined)
    radial=np.hypot(y,z+31)
    cap_weight=np.maximum(smooth01((radial-28.2)/9),smooth01((x+5.9)/8))
    combined=cap_weight*combined+(1-cap_weight)*tweeter
    mating_rear=um_rear(vertical,y)+19.5*smooth01((vertical-392)/34)-19.5*smooth01((vertical-380)/35)
    result=np.maximum.reduce(np.broadcast_arrays(combined,mating_rear-front,
        x-retained.P.overall_depth/2-um_front_rise(y,vertical),-x-retained.P.overall_depth/2))
    weight=smooth01((vertical-REAR_LOFT_START_Y)/3.)*(1-smooth01((vertical-438.)/8.))
    weight*=smooth01((radial-REAR_LOFT_CAP_R)/2.)
    weight*=1-smooth01((abs(y)-56.)/8.)
    weight*=mask
    rear_limit=-retained.smooth_union(front-rear,-reference_side,UM_EDGE_ROUNDING)
    original=result+weight*(np.maximum(result,rear_limit)-result)
    # One radial loft owns the outer UM/T waist. Blending capped solids
    # mixes their face planes into the outline and leaves the circular trim
    # visible as a notch. Blend only radial fields, then roll into the same
    # front plane and rear height surface. Keep all service interfaces in
    # the unchanged central region and the LM geometry far below this patch.
    blend=smooth01((vertical-420.)/28.)
    umside=um_outline_distance(y,vertical)+UM_DRAFT_SLOPE*(18.3+um_rim_height(vertical)-front)
    side=(1-blend)*umside+blend*tweeter_radial(x,y,z)
    rounding=np.maximum(UM_EDGE_ROUNDING*(1-blend),1e-8)
    face=front-18.3-um_front_rise(y,vertical)
    fair=-retained.smooth_union(-side,-face,rounding)
    fair=-retained.smooth_union(-fair,front-rear,rounding)
    fair=np.maximum(fair,-x-retained.P.overall_depth/2)
    patch=smooth01((vertical-402.)/10.)*(1-smooth01((vertical-446.)/8.))
    patch*=smooth01((abs(y)-28.5)/6.5)
    return original+patch*(fair-original)


def housing_field(x,y,z):
    """Sculpt the outside first, then cut original T retention and new flare."""
    a=envelope_field(x,y,z)
    mouths=np.minimum(lower_horn_field(x,y,z+31),
                       retained.v3.horn_field(-x,y,z-31,retained.P,False))
    a=np.maximum(a,-mouths)
    for i in (0,1):
        xx,yy,zz=retained.local_coords(x,y,z,i)
        m=retained.M
        # Retain the cap/O-ring entry, enlarge only the chamber in front of
        # the cap face. The retainer and M3 heads are removable through it.
        service=retained.cylinder(xx,yy,zz,m.service_radius,-5.9,m.service_end_x)
        a=np.maximum(a,-service)
        for radius,hi in [(27.,m.service_end_x),(m.flange_cavity_radius,m.seat_x),
                          (m.cap_radius+m.cap_clearance,-5.9)]:
            a=np.maximum(a,-retained.cylinder(xx,yy,zz,radius,-40,hi))
        for angle in retained.SCREW_ANGLES:
            cy=m.screw_circle_radius*np.cos(angle);cz=m.screw_circle_radius*np.sin(angle)
            pilot=retained.cylinder(xx,yy-cy,zz-cz,m.insert_pilot_radius,m.service_end_x-.5,m.service_end_x+m.insert_length)
            relief=retained.cylinder(xx,yy-cy,zz-cz,m.insert_tip_relief_radius,m.service_end_x-.5,m.insert_tip_x)
            a=np.maximum(a,-np.minimum(pilot,relief))
    b=-ducts_field(x,y,z)
    k=retained.M.entry_edge_blend
    h=np.maximum(1-abs(a-b)/k,0)
    return np.maximum(a,b)+k*h**3/6


def special_x():
    return np.unique(np.r_[retained.exact_special_x(),
        np.array([6.8,12.4,14.3,18.3])-INSTALLED_Z_OFFSET])


def extract_surface(field,bounds,step,special):
    """Retained root-refined meshing with conservative float32 cleanup.

    The original cleaner deletes finite, very narrow facets at blended
    duct openings. Preserve those boundary facets, as for the final STL.
    The source package on disk remains immutable.
    """
    original=retained.v3.stl_safe_mesh
    def clean(mesh):
        raw=mesh.copy()
        try:
            mesh=preserve_void_winding(mesh)
        except AssertionError:
            # Root refinement can leave sub-micron folds which collapse to
            # a shared float32 edge. Simplify only within export precision.
            np.savez_compressed(HERE/'assembly/precision_raw.npz', vertices=raw.vertices, faces=raw.faces)
            exact=solid(raw)
            simplified=exact.set_tolerance(.00003).simplify(.00003)
            assert abs(exact.volume()-simplified.volume()) < .01
            mesh=preserve_void_winding(to_trimesh(simplified))
        mesh.fix_normals(multibody=False)
        return mesh
    retained.v3.stl_safe_mesh=clean
    try:
        return retained.extract(field,bounds,step,special)
    finally:
        retained.v3.stl_safe_mesh=original


def positive_void(mesh):
    mesh=mesh.copy();mesh.faces=mesh.faces[:,[0,2,1]]
    return mesh


@lru_cache(None)
def magnet_sites(sign=None):
    sites=[]
    for angle in MAGNET_ANGLES:
        a=np.deg2rad(angle);radial=np.array([np.cos(a),np.sin(a),0.])
        if sign is not None and radial[0]*sign<0: continue
        center=np.array([0.,interface.UM_CUTOUT[1],MAGNET_Z])
        low,high=50.,74.
        for _ in range(36):
            middle=(low+high)/2;point=center+middle*radial
            value=envelope_field(point[2]-INSTALLED_Z_OFFSET,point[0],point[1]-INSTALLED_Y_OFFSET)
            if value<0:low=middle
            else:high=middle
        contact=center+(low+high)/2*radial
        # Pair axes with the actual inclined skin, including its depth
        # component. A local right-handed frame carries the cradle/roof.
        step=.01;gradient=[]
        for axis in (0,1,2):
            delta=np.zeros(3);delta[axis]=step
            p=contact+delta;q=contact-delta
            gradient.append((envelope_field(p[2]-INSTALLED_Z_OFFSET,p[0],p[1]-INSTALLED_Y_OFFSET)
                -envelope_field(q[2]-INSTALLED_Z_OFFSET,q[0],q[1]-INSTALLED_Y_OFFSET))/(2*step))
        n=np.array(gradient);n/=np.linalg.norm(n)
        tangent=np.array([-n[1],n[0],0.]);tangent/=np.linalg.norm(tangent)
        up=np.cross(n,tangent)
        sites.append({'angle_deg':angle,'normal':n,'tangent':tangent,'up':up,'contact':contact})
    return sites


def magnet_pockets(original,owner):
    """D6x3 captive pockets, placed beneath a completely independent skin.

    Preserve the qualified circular cradle/chimney/45-degree roof topology.
    Scaling both in-plane coordinates equally retains the roof angle; the
    axial allowance becomes 3.10 mm. New size/cover still needs a coupon.
    """
    result=[]
    for boundary in original.split(only_watertight=False):
        if boundary.volume>=0 or boundary.center_mass[1]<350: continue
        sign=1 if boundary.center_mass[0]>0 else -1
        angle=50.5 if sign>0 else 129.5
        for site in magnet_sites(sign):
            pocket=positive_void(boundary);normal=site['normal']
            basis=np.eye(4);basis[:3,:3]=np.column_stack([normal,site['tangent'],site['up']])
            transform=trimesh.transformations.translation_matrix(site['contact'])
            transform=transform @ basis
            transform=transform @ np.diag([MAGNET_CAVITY_DEPTH/2.1,
                MAGNET_CAVITY_DIAMETER/5.2,MAGNET_CAVITY_DIAMETER/5.2,1.])
            transform=transform @ trimesh.transformations.rotation_matrix(-np.deg2rad(angle),[0,0,1])
            transform=transform @ trimesh.transformations.translation_matrix([0,-interface.UM_CUTOUT[1],-15.1])
            pocket.apply_transform(transform)
            radial=(pocket.vertices-site['contact'])@normal
            present=np.max(radial) if owner=='body' else np.min(radial)
            desired=magnet_burial(site['angle_deg'],owner)
            shift=normal*(desired-present)
            pocket.apply_translation(shift);transform[:3,3]+=shift
            result.append((boundary,pocket,transform))
    return result


@lru_cache(None)
def magnet_burial(angle,owner):
    """Solve the pocket-face datum using the entire cradle plus cover.

    Offset a dense pocket hull by a sphere, then keep that hull wholly
    inside the body (or wholly outside the wing's clearance envelope).
    The face datum is internal and never cuts or flattens the exterior.
    Final STL checks independently measure the actual wall thickness.
    """
    site=next(s for s in magnet_sites() if s['angle_deg']==angle)
    n=site['normal'];t=site['tangent'];up=site['up'];r=MAGNET_CAVITY_DIAMETER/2
    theta=np.linspace(0,np.pi,65)
    cross=np.c_[r*np.cos(theta),r*np.sin(theta)]
    corners=np.array([[-r,0],[-r,-r],[0,-2*r],[r,-r],[r,0]])
    cross=np.r_[cross,np.concatenate([a+(b-a)*np.linspace(0,1,13)[:,None]
                     for a,b in zip(corners[:-1],corners[1:])])]
    sign=-1 if owner=='body' else 1
    hull=np.concatenate([cross[:,0,None]*t+cross[:,1,None]*up
                         +a*n for a in np.linspace(0,sign*MAGNET_CAVITY_DEPTH,3)])
    sphere=trimesh.creation.icosphere(subdivisions=2 if owner=='wing' else 1,radius=1).vertices
    cover=MAGNET_SKIN+(WING_GAP+.02 if owner=='wing' else 0)
    cloud=(hull[:,None,:]+sphere[None,:,:]*cover).reshape(-1,3)+site['contact']
    def fits_at(distance):
        q=cloud+sign*distance*n
        if owner=='body':
            value=envelope_field(q[:,2]-INSTALLED_Z_OFFSET,q[:,0],q[:,1]-INSTALLED_Y_OFFSET)
            return np.max(value)<0
        else:
            value=envelope_field(q[:,2]-INSTALLED_Z_OFFSET,q[:,0],q[:,1]-INSTALLED_Y_OFFSET)
            return np.min(value)>0
    # Inclined axes also move through depth. Excessive inward travel can
    # hit the front surface, so fit is not monotone over the entire 8 mm
    # search interval. Find the first feasible bracket before refining it.
    low=0.;high=None
    for trial in np.arange(.125,8.001,.125):
        if fits_at(trial):high=trial;break
        low=trial
    assert high is not None,('buried magnet cannot fit',angle,owner)
    for _ in range(22):
        middle=(low+high)/2
        if fits_at(middle):high=middle
        else:low=middle
    return sign*high


def protected_regions(padding=0.):
    """Exact driver seat/pilots, LM half-laps and central LM seam authority."""
    import manifold3d as md
    bore=md.Manifold.cylinder(100+2*padding,UM_MECHANICAL_R+padding,UM_MECHANICAL_R+padding,256).translate((0,interface.UM_CUTOUT[1],-50-padding))
    # Cylinder primitive is along Z (installed forward axis).
    low=md.Manifold.cube((54.2+2*padding,40+2*padding,100+2*padding)).translate((-27.1-padding,287-padding,-50-padding))
    # The inner underside of each ear is complementary to the LM's cusp
    # web, which extends above its circular ring. Keep that exact boundary.
    inner_height=interface.JOINT_EAR_Y-287
    low+=md.Manifold.cube((64+2*padding,inner_height+2*padding,100+2*padding)).translate((-32-padding,287-padding,-50-padding))
    ears=md.Manifold()
    for lateral in interface.JOINT_EAR_X:
        ears+=md.Manifold.cylinder(100+2*padding,4.85+padding,4.85+padding,128).translate((lateral,interface.JOINT_EAR_Y,-50-padding))
    # Mechanical datums end at the native front face. The driver cylinder
    # remains protected through all depths; cosmetic base material may
    # continue ahead of the closed LM receiver floors.
    # A real overlap inside the solid cover prevents coincident-plane
    # export cracks. The actual LM front contour clips the cosmetic
    # extension separately, so this overlap cannot grow into the LM.
    native_depth=md.Manifold.cube((240,240,68.29+padding)).translate((-120,260,-50))
    return bore+((low+ears)^native_depth)


def tube_mesh(centers,radii,step=.30):
    centers=np.array(centers);radii=np.broadcast_to(radii,len(centers))
    lo=np.min(centers-radii[:,None],axis=0)-.6
    hi=np.max(centers+radii[:,None],axis=0)+.6
    tree=cKDTree(centers)
    def field(x,y,z):
        xx,yy,zz=np.broadcast_arrays(x,y,z)
        q=np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
        return retained.tube_distance(q,centers,radii,tree).reshape(xx.shape)
    return retained.extract(field,np.c_[lo,hi].tolist(),step)


def fused_body(upper):
    import manifold3d as md
    um,provenance=restored_core(UM_SOURCE_STATE)
    # Use the implicit mesh's exact stored datums for coincident native
    # planes. Sub-micron double skins otherwise produce false edge facets
    # along the flat receiver lands after the boolean/print transform.
    for plane in (6.8,12.4,14.3,16.4,interface.THICKNESS_MM):
        stored=np.float32(plane-INSTALLED_Z_OFFSET).item()+INSTALLED_Z_OFFSET
        um.vertices[np.isclose(um.vertices[:,2],plane,atol=1e-6,rtol=0),2]=stored
    original=solid(um).set_tolerance(.00005)
    exact_original=original
    # Fill the old magnet cavities and abandoned upper cable turn before
    # retaining the exact inner mechanism. Their replacements are cut last.
    pockets=magnet_pockets(um,'body')
    for old,_,_ in pockets: original+=solid(positive_void(old))
    old_route=route.ts_cable_points(.08)
    old_route=old_route[old_route[:,1]>=CONNECT_START_Y-.3]
    original+=solid(tube_mesh(old_route,3.02))
    region=protected_regions().set_tolerance(.00005)
    full_blank=solid(installed(upper)).set_tolerance(.00005)
    blank=full_blank
    # Keep the new inner foot fairings outside the LM ring plus its seam.
    # Restore the original central seam/receivers afterwards, since the LM
    # has local reliefs there that a full circular keepout does not describe.
    lm_radius=interface.LM_VISIBLE_RING_R+.2
    lm_clearance=md.Manifold.cylinder(50,lm_radius,lm_radius,512).translate(
        (0,interface.L22_CUTOUT[1],-20))
    # Carry the LM keepout through the entire thickness: the UM must end at
    # the upper LM contour, never bridge over or conceal its front face.
    blank-=lm_clearance
    lm,_=restored_core(UM_SOURCE_STATE,'lm')
    # Use the LM's actual front contour, including its cusp web outside
    # the circular ring. Its rear half-laps are below this slice and do
    # not create false cutouts over the UM receiver caps.
    lm_front=solid(lm)^md.Manifold.cube((240,240,.1)).translate((-120,150,18.27))
    lm_front_shadow=lm_front.project().offset(.05).extrude(40.).translate((0,0,18.27))
    blank-=lm_front_shadow
    # Replace exposed old cable ribs with a continuous rear annulus. The
    # independent UM-driver outlet ends below Y=332 and stays untouched.
    organic_zone=md.Manifold.cube((180,110,80)).translate((-90,332,-40))
    original-=organic_zone-blank
    # Sculpt the free rear of the old lower closure web to the SAME surface
    # as the organic shell. Keeping its entire old rectangular extrusion
    # would expose the X=+/-27.1 and Y=327 partition edges after fusion.
    lower_zone=md.Manifold.cube((90,27,70)).translate((-45,309,-50))
    def lower_rear_solid(x,y,z):
        return np.maximum.reduce(np.broadcast_arrays(
            um_rear(y,x)-z,z-20.,abs(x)-45.5,308.5-y,y-336.5))
    rear_limit=solid(extract_surface(lower_rear_solid,
        [[-46,46],[308,337],[-1,21]],.22,[]))
    original-=lower_zone-rear_limit
    # Back the entire seat annulus onto the curved exterior, including its
    # bottom sector. The former Y=332 cutoff left a circular recessed band
    # and two sharp lower shelves even after the native seat was made solid.
    backfill=blank
    backfill^=md.Manifold.cylinder(60,UM_MECHANICAL_R+.03,UM_MECHANICAL_R+.03,256).translate(
        (0,interface.UM_CUTOUT[1],-50))
    backfill-=md.Manifold.cylinder(100,41.,41.,256).translate((0,interface.UM_CUTOUT[1],-50))
    # Stop below the retained pilot floors and seat.
    backfill^=md.Manifold.cube((180,140,59.9)).translate((-90,300,-50))
    # Extend the new curved surface across the native closure-web plan,
    # which lies partly inside the conservative circular LM keepout. Its
    # actual projected ownership is retained; no LM seam is moved. The
    # receiver lands remain Z=12.4 and therefore cannot grow into the LM.
    native_footprint=exact_original.project().extrude(100).translate((0,0,-50))
    lower_fill=(full_blank^native_footprint)^lower_zone
    lower_fill^=md.Manifold.cube((180,140,62.5)).translate((-90,300,-50))
    # Projection of the diagonal T collar alone overstates UM ownership at
    # its handoff. Exclude the actual LM silhouette with the normal seam
    # clearance before adding material behind the old closure-web plane.
    lm_shadow=solid(lm).project().offset(.05).extrude(100).translate((0,0,-50))
    lower_fill-=lm_shadow
    backfill+=lower_fill
    # Preserve the incoming source lumen and the rounded M2 access exactly.
    # These are functional voids, unlike the abandoned under-seat cavity.
    incoming=route.ts_cable_points(.08)
    incoming=incoming[(incoming[:,1]>307)&(incoming[:,1]<326.4)]
    functional_void_region=solid(tube_mesh(incoming,3.35))
    # The separate UM-driver lead crosses behind this lower sector. Keep
    # its service corridor open while filling the surrounding annular band.
    um_lead=route.route_cable_points(.08)
    um_lead=um_lead[um_lead[:,1]>309]
    functional_void_region+=solid(tube_mesh(um_lead,4.2))
    # Start above the stored native rear plane. A mask starting exactly at
    # nominal 6.8 can leave an open sub-micron air film after float32 datum
    # alignment, even though all of the real M2 passage is above Z=7.8.
    functional_void_region+=md.Manifold.cube((6.4,27,6.99)).translate((-20.2,308,6.81))
    # The ear rolls rise above the former 9.9-mm backing cutoff. Fill right
    # up to their flat lands, preserving the actual pilot/receiver voids,
    # rather than leaving little rectangular pockets below Z=12.4.
    for px,py in interface.UM_PILOT_XY:
        functional_void_region+=md.Manifold.cylinder(12,3.,3.,128).translate((px,py,10.2))
    for px in interface.JOINT_EAR_X:
        functional_void_region+=md.Manifold.cylinder(20,4.85,4.85,128).translate(
            (px,interface.JOINT_EAR_Y,6.))
    backfill-=functional_void_region-exact_original
    original+=backfill
    # A continuous cover crosses the old sparse inner carrier into the new
    # gallery. It is clipped to the organic exterior, rather than becoming
    # another exposed tube on the surface.
    _,_,connection,radii=wiring()
    connection=trimesh.transform_points(connection,PACKAGE_TO_INSTALLED)
    mask=connection[:,1]>=CONNECT_START_Y-.02
    cover=solid(tube_mesh(connection[mask],radii[mask]+1.8))^blank
    cover^=md.Manifold.cube((180,180,64.3)).translate((-90,300,-50))
    functional=md.Manifold.cylinder(10,49.25,49.25,256).translate((0,interface.UM_CUTOUT[1],13.3))
    # Keep the entire D82 insertion bore open below the seat. The gallery
    # needs only its measured wall cover, not extra material inside this
    # already-clear native opening.
    functional+=md.Manifold.cylinder(64.3,41.,41.,256).translate((0,interface.UM_CUTOUT[1],-50.))
    for px,py in interface.UM_PILOT_XY:
        functional+=md.Manifold.cylinder(14,3.,3.,128).translate((px,py,10.2))
    cover-=functional-exact_original
    original+=cover
    # A small internal overlap joins the preserved mechanism to the sculpted
    # shell. An exactly coincident partition creates float32 edge contacts.
    body=(blank-region)+(original^protected_regions(.03))
    # The original protected mechanical core is itself the inner opening.
    # Remove the new route from both the reshaped shell and that exact core.
    routes,_,_,_=wiring()
    for centers,radii in routes.values(): body-=solid(installed(tube_mesh(centers,radii)))
    for _,pocket,_ in pockets: body-=solid(pocket)
    # Finish M3 seats with exact cylinders after implicit meshing. Linear
    # interpolation at the blind floor otherwise nips the outermost 0.09 mm
    # of depth, despite a nominal four-mm field. Keep these functional bores
    # independent of cosmetic tessellation and the subsequent UM union.
    for which in (0,1):
        for angle in retained.SCREW_ANGLES:
            m=retained.M
            cutter=md.Manifold.cylinder(m.insert_length+.02,m.insert_pilot_radius,
                                       m.insert_pilot_radius,256).rotate((0,90,0))
            cutter=cutter.translate((m.service_end_x-.02,
                                    m.screw_circle_radius*np.cos(angle),
                                    m.screw_circle_radius*np.sin(angle)-31))
            if which: cutter=cutter.rotate((0,180,0))
            body-=solid(installed(to_trimesh(cutter)))
    # Backfilling the former surface duct can trap closed air crescents.
    # Fill these before STL coordinate welding, where a trapped void that
    # just touches a seam could otherwise turn into a non-manifold edge.
    body=body.simplify(.0001)
    magnet_volumes=[pocket.volume for _,pocket,_ in pockets]
    for cavity in body.decompose():
        if cavity.volume()<0 and min(abs(-cavity.volume()-v) for v in magnet_volumes)>.01:
            boundary=cavity.to_mesh64()
            fill=md.Manifold(md.Mesh64(np.array(boundary.vert_properties),
                np.array(boundary.tri_verts)[:,[0,2,1]]))
            body+=fill
    result=package_frame(to_trimesh(body))
    return result,provenance


def print_pose(mesh):
    """Front face down, diagonal XY placement to fit the fused outline."""
    # Installed (X,Y,Z) becomes (X,-Y,-Z), a proper rotation.
    pose = np.diag([1.,-1.,-1.,1.]) @ PACKAGE_TO_INSTALLED
    base = mesh.copy(); base.apply_transform(pose)
    # A fixed 45-degree rotation fits the tall narrow outline on a square bed.
    turn = trimesh.transformations.rotation_matrix(np.pi/4,[0,0,1])
    transform = turn @ pose
    result = mesh.copy(); result.apply_transform(transform)
    transform[:3,3] += -result.bounds[0]
    result = mesh.copy(); result.apply_transform(transform)
    # Boolean intersections create sub-micron slivers where the source STL
    # and implicit front planes differ by float32 rounding. Collapse only
    # those numerical edges before the final STL quantisation.
    result = to_trimesh(solid(result).set_tolerance(.0001))
    return preserve_void_winding(result),transform
