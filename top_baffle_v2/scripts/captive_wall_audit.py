"""Full-height captive-wall presence checks on actual model extrusion.

Unlike the D5 coupon's single-traversal rule, inclined D6 skins can contain
more than one variable-width bead. Test the union of actual bead footprints.
This is digital qualification; it cannot prove physical extrusion or bonding.
"""
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from gcode_analysis import parse_gcode
from lx521_baffle.print_policy import policy

_gate = policy()['magnets']['crescent_D6']['digital_wall_gate']
BOUNDARY_SAMPLE_MM = _gate['boundary_sample_mm']
BOUNDARY_TOLERANCE_MM = _gate['max_boundary_gap_mm']
MIN_RETAINING_BEAD_MM = _gate['min_bead_width_mm']
MAX_RETAINING_BEAD_MM = _gate['max_bead_width_mm']
MAX_RETAINING_COMPONENTS = _gate['maximum_retaining_components']
MIN_INTERLAYER_OVERLAP = _gate['minimum_interlayer_overlap_fraction']


def retaining_continuity(ring, boundary, previous):
    """Check connected wall footprints and contact with prior model material."""
    components = [ring] if ring.geom_type == 'Polygon' else list(ring.geoms)
    contacts = [p for p in components if p.distance(boundary) <= BOUNDARY_TOLERANCE_MM]
    assert len(contacts) == MAX_RETAINING_COMPONENTS, ('disconnected retaining wall', len(contacts))
    assert previous is not None and ring.area > 0, 'missing previous retaining substrate'
    overlap = ring.intersection(previous).area / ring.area
    assert overlap >= MIN_INTERLAYER_OVERLAP, ('insufficient retaining interlayer overlap', overlap)
    return dict(connected_components=len(contacts), interlayer_overlap_fraction=overlap)


def audit_captive_walls(stl, offset, specs, gcode, *, diameters=(6,)):
    mesh = trimesh.load_mesh(stl, process=True)
    mesh.apply_translation(offset)
    cavities = [s for s in mesh.split(only_watertight=False) if s.volume < 0]
    rois = [(s['center_bed_mm'][0]-10, s['center_bed_mm'][1]-10,
             s['center_bed_mm'][0]+10, s['center_bed_mm'][1]+10) for s in specs]
    parsed = parse_gcode(gcode, retain_regions=rois)
    reports = []
    for spec, roi in zip(specs, rois):
        if spec['diameter_mm'] not in diameters: continue
        cavity = min(cavities, key=lambda s: np.linalg.norm(s.center_mass-spec['center_bed_mm']))
        layers = []
        widths = []
        previous_coverage = None
        continuity = []
        for layer in parsed.layers:
            z = layer.z-(layer.layer_height or .16)/2
            if not cavity.bounds[0,2] < z < cavity.bounds[1,2]: continue
            edges = trimesh.intersections.mesh_plane(cavity, [0,0,1], [0,0,z])
            samples = []
            for a,b in edges[:,:,:2]:
                count = max(2, int(np.ceil(np.linalg.norm(b-a)/BOUNDARY_SAMPLE_MM))+1)
                samples.extend(a+(b-a)*np.linspace(0,1,count)[:,None])
            if not samples: continue
            beads, paths = [], []
            for segment in layer.segments:
                feature = segment.feature.lower()
                if feature in ('custom','undefined','prime tower') or feature.startswith('support'): continue
                x,y = (segment.x0+segment.x1)/2, (segment.y0+segment.y1)/2
                if not roi[0]-1 < x < roi[2]+1 or not roi[1]-1 < y < roi[3]+1: continue
                line = LineString([(segment.x0,segment.y0),(segment.x1,segment.y1)])
                width = segment.line_width or .62
                beads.append(line.buffer(width/2, quad_segs=4))
                if feature == 'outer wall': paths.append((line, width))
            coverage = unary_union(beads)
            points = [Point(p) for p in samples]
            distances = [coverage.distance(p) for p in points]
            worst = max(distances)
            assert worst <= BOUNDARY_TOLERANCE_MM, (spec['name'], layer.z, 'missing retaining boundary', worst)
            # Measure widths only in the seated disc's working region,
            # excluding the cradle tip and closing roof's tiny facets.
            near_widths = []
            contact = None
            if spec['seated_bottom_z_mm']+.5 < z < spec['seated_top_z_mm']-.5:
                boundary = unary_union([LineString(e[:,:2]) for e in edges])
                near_widths = [w for line,w in paths if line.distance(boundary) < w/2+.04]
                assert near_widths, (spec['name'],layer.z,'no cavity outer wall')
                assert min(near_widths) >= MIN_RETAINING_BEAD_MM-.005, (spec['name'],layer.z,min(near_widths))
                assert max(near_widths) <= MAX_RETAINING_BEAD_MM, (spec['name'],layer.z,max(near_widths))
                widths += near_widths
                ring = unary_union([line.buffer(w/2, quad_segs=4) for line,w in paths
                                    if line.distance(boundary) < w/2+.04])
                contact = retaining_continuity(ring, boundary, previous_coverage)
                continuity.append(contact)
            previous_coverage = coverage
            layers.append(dict(z_mm=layer.z, sample_count=len(points), max_boundary_gap_mm=worst,
                               retaining_width_range_mm=[min(near_widths),max(near_widths)] if near_widths else None,
                               retaining_continuity=contact))
        minimum_layers=30 if spec['diameter_mm']==6 else max(3,int(cavity.extents[2]/.16)-2)
        assert len(layers) >= minimum_layers and widths, (spec['name'], 'insufficient wall evidence')
        reports.append(dict(site=spec['name'], diameter_mm=spec['diameter_mm'], checked_layers=len(layers),
            width_range_mm=[min(widths),max(widths)], maximum_boundary_gap_mm=max(r['max_boundary_gap_mm'] for r in layers),
            checked_continuity_layers=len(continuity),
            minimum_interlayer_overlap_fraction=min(c['interlayer_overlap_fraction'] for c in continuity),
            maximum_connected_components=max(c['connected_components'] for c in continuity), layers=layers))
    return dict(status='pass', physical_qualification='pending same-geometry coupon print',
                gcode_sha256=hashlib.sha256(gcode.read_bytes()).hexdigest(),
                stl_sha256=hashlib.sha256(stl.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                sample_spacing_mm=BOUNDARY_SAMPLE_MM, boundary_tolerance_mm=BOUNDARY_TOLERANCE_MM,
                accepted_width_mm=[MIN_RETAINING_BEAD_MM,MAX_RETAINING_BEAD_MM],
                minimum_interlayer_overlap_fraction=MIN_INTERLAYER_OVERLAP,
                maximum_retaining_components=MAX_RETAINING_COMPONENTS, sites=reports)
