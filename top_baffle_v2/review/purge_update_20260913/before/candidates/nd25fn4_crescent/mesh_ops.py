"""Oriented mesh booleans: preserve enclosed magnet cavities as voids."""
import manifold3d
import numpy as np
import trimesh


def solid(mesh):
    result = manifold3d.Manifold(manifold3d.Mesh64(
        np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.uint64)))
    assert result.status() == manifold3d.Error.NoError, result.status()
    return result


def to_trimesh(manifold):
    m = manifold.to_mesh64()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:,:3], np.asarray(m.tri_verts), process=True)


def preserve_void_winding(mesh):
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32).astype(np.float64)
    mesh.merge_vertices()
    # Preserve finite sliver faces created by float32 quantisation; deleting
    # a nonzero-area boundary triangle would open an otherwise closed mesh.
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-12))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        stitch_precision_boundaries(mesh)
    return mesh


def stitch_precision_boundaries(mesh, tolerance=.00003):
    """Retriangulate only collapsed, collinear boundary slivers.

    Split the long adjacent triangle at an EXISTING intermediate boundary
    vertex. No vertex moves and no real opening is capped. The deviation
    from the old triangle plane is bounded by float32 export precision.
    """
    count_fixed=0
    maximum=0.
    for _ in range(64):
        if mesh.is_watertight:
            if count_fixed:
                print(f'Retriangulated {count_fixed} precision slivers; max edge deviation {maximum:.8f} mm',flush=True)
            return mesh
        edges,counts=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
        assert np.all(counts<=2),'Non-manifold edge with more than two incident faces'
        boundary=edges[counts==1]
        ids=np.unique(boundary)
        points=mesh.vertices[ids]
        lengths=np.linalg.norm(mesh.vertices[boundary[:,1]]-mesh.vertices[boundary[:,0]],axis=1)
        repaired=False
        for a,b in boundary[np.argsort(-lengths)]:
            start=mesh.vertices[a];direction=mesh.vertices[b]-start
            t=(points-start)@direction/(direction@direction)
            deviation=np.linalg.norm(points-(start+t[:,None]*direction),axis=1)
            eligible=(t>1e-7)&(t<1-1e-7)&(deviation<=tolerance)&(ids!=a)&(ids!=b)
            if not np.any(eligible): continue
            candidates=np.flatnonzero(eligible)
            chosen=candidates[np.argmax(np.minimum(t[candidates],1-t[candidates]))]
            mid=int(ids[chosen])
            faces=np.asarray(mesh.faces).copy()
            rows=np.flatnonzero(np.any(faces==a,axis=1)&np.any(faces==b,axis=1))
            assert len(rows)==1
            row=rows[0];face=faces[row]
            i=next(i for i in range(3) if {int(face[i]),int(face[(i+1)%3])}=={int(a),int(b)})
            a,b,c=face[i],face[(i+1)%3],face[(i+2)%3]
            faces[row]=[a,mid,c]
            mesh.faces=np.r_[faces,[[mid,b,c]]]
            maximum=max(maximum,float(deviation[chosen]));count_fixed+=1;repaired=True
            break
        assert repaired,('Non-collinear opening cannot be repaired as precision noise',boundary.tolist())
    raise AssertionError('Too many precision boundaries')
