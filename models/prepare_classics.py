"""Rebuild the four scene-ready OBJ models from originals (NumPy + fast-simplification)."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '.model-tools'))
import fast_simplification


def read_obj(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == 'f':
            face = [int(x.split('/')[0]) for x in parts[1:]]
            faces.append([x - 1 if x > 0 else len(vertices) + x for x in face])
    return np.array(vertices, dtype=np.float64), faces


def subdivide(vertices, faces):
    """One Catmull-Clark step; use boundary rules for Suzanne's open eye rims."""
    face_points = np.array([vertices[f].mean(axis=0) for f in faces])
    edges, adjacent_faces, adjacent_edges = {}, [[] for _ in vertices], [[] for _ in vertices]
    for face_id, face in enumerate(faces):
        for i, a in enumerate(face):
            b = face[(i + 1) % len(face)]
            edge = tuple(sorted((a, b)))
            edges.setdefault(edge, []).append(face_id)
            adjacent_faces[a].append(face_id)
    for edge in edges:
        for v in edge:
            adjacent_edges[v].append(edge)
    points = vertices.copy()
    for i, (fids, eids) in enumerate(zip(adjacent_faces, adjacent_edges)):
        boundary = [e for e in eids if len(edges[e]) == 1]
        if boundary:
            neighbors = [e[1] if e[0] == i else e[0] for e in boundary]
            assert len(neighbors) == 2
            points[i] = 0.75 * vertices[i] + 0.125 * vertices[neighbors].sum(axis=0)
        elif fids:
            n = len(fids)
            midpoints = np.array([vertices[list(e)].mean(axis=0) for e in eids])
            points[i] = (face_points[fids].mean(axis=0) + 2 * midpoints.mean(axis=0)
                         + (n - 3) * vertices[i]) / n
    edge_ids, extra = {}, []
    for edge, fids in edges.items():
        assert len(fids) in (1, 2)
        edge_ids[edge] = len(vertices) + len(extra)
        extra.append(vertices[list(edge)].mean(axis=0) if len(fids) == 1 else
                     (vertices[list(edge)].sum(axis=0) + face_points[fids].sum(axis=0)) / 4)
    face_offset = len(vertices) + len(extra)
    quads = []
    for fid, face in enumerate(faces):
        for i, a in enumerate(face):
            quads.append([a, edge_ids[tuple(sorted((a, face[(i + 1) % len(face)])))],
                          face_offset + fid, edge_ids[tuple(sorted((face[i - 1], a)))]])
    return np.vstack((points, extra, face_points)), quads


def prepare(name):
    source = ROOT / 'originals' / f'{name}.obj'
    vertices, polygons = read_obj(source)
    original_vertices = len(vertices)
    operation = 'triangulation + quadric edge-collapse decimation'
    if name == 'suzanne':
        vertices, polygons = subdivide(vertices, polygons)
        operation = 'one Catmull-Clark subdivision + triangulation'
    faces = np.array([[p[0], p[i], p[i + 1]] for p in polygons for i in range(1, len(p) - 1)], dtype=np.int32)
    # Remove zero-area input triangles before decimation.
    valid = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                    vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) > 1e-12
    faces = faces[valid]
    if len(faces) > 4000:
        vertices, faces = fast_simplification.simplify(vertices, faces, target_count=4000)
    used, faces = np.unique(faces, return_inverse=True)
    vertices, faces = vertices[used], faces.reshape(-1, 3)
    # The OBJ snapshots used here already share this renderer's Y-up frame.
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    # After the renderer's p*6+(0.2,-1,0): center XZ=0, bottom=-0.98,
    # maximum extent=1.15. Fit inside the room without changing C++.
    vertices -= np.array([(lower[0] + upper[0]) / 2, lower[1], (lower[2] + upper[2]) / 2])
    vertices *= (1.15 / 6) / max(upper - lower)
    vertices += np.array([-0.2 / 6, 0.02 / 6, 0])
    assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
    assert np.all(np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                          vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) > 1e-14)
    target = ROOT / f'{name}-low.obj'
    text = f'# {name}: {operation}; scene normalized. See classics-source.txt\n'
    text += ''.join('v ' + ' '.join(f'{c:.10g}' for c in v) + '\n' for v in vertices)
    text += ''.join('f ' + ' '.join(str(int(i) + 1) for i in f) + '\n' for f in faces)
    target.write_text(text, encoding='ascii')
    result = dict(name=name, file=target.name, vertices=len(vertices), triangles=len(faces),
                  original_vertices=original_vertices, processing=operation,
                  source=f'https://github.com/alecjacobson/common-3d-test-models/blob/master/data/{name}.obj',
                  original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    print(f'{target.name}: {len(vertices)} vertices, {len(faces)} triangles')
    return result


if __name__ == '__main__':
    results = [prepare(name) for name in ('teapot', 'suzanne', 'cow', 'fandisk')]
    (ROOT / 'classics-manifest.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
