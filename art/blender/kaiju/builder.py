"""Builder: turns SDF programs + rigid part pieces into rigged, glTF-ready Blender objects.

Frame modules (kaiju/frames/<id>.py) expose `build(K)` and work purely in GAME coordinates
(+Y up, +Z forward, +X right, ground at y=0, a kaiju is ~9-10 units tall), same as
js/render/kaiju-model.js. Conversion to Blender (Z up, -Y forward) happens here.

Object / naming contract of the exported GLB:
  rig            armature (bones named by the frame, always has `root`)
  body           skinned mesh, material slots body / belly / accent (+ horn, eye, mouth, plate ...)
  pow_t0..t3     EXCLUSIVE claw / horn sets (show exactly one)
  arm_t1..t3, enr_t1..t3, spd_t1..t3, fcs_t1..t3   ADDITIVE upgrade parts (show all <= tier)
  vis_<key>      optional ability cosmetics
  mount empties  head, jaw, mouth, chest, back, tailTip, handL, handR, shoulderL, shoulderR, crown
"""
import math
import os
import re

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from . import geom, materials
from . import sdf as S
from .noise import fbm

POW_SCALE = [1.0, 1.3, 1.62, 1.95]
HP_SIZE = [1, 1.1, 1.22, 1.36]


SCALE = 1.0   # model-units -> exported units (set per frame; tiny wild frames model at 10x and export small)


def g2b(P):
    P = np.asarray(P, float) * SCALE
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


def b2g(P):
    P = np.asarray(P, float) / SCALE
    return np.stack([P[..., 0], P[..., 2], -P[..., 1]], -1)


def new_mesh(name, V, Fc):
    """V in game coords; Fc (M,3|4) int array."""
    me = bpy.data.meshes.new(name)
    Vb = g2b(V)
    Fc = np.asarray(Fc, np.int32)
    n = Fc.shape[1]
    me.vertices.add(len(Vb))
    me.vertices.foreach_set('co', Vb.astype(np.float32).ravel())
    me.loops.add(Fc.size)
    me.loops.foreach_set('vertex_index', Fc.ravel())
    me.polygons.add(len(Fc))
    me.polygons.foreach_set('loop_start', np.arange(len(Fc), dtype=np.int32) * n)
    me.polygons.foreach_set('loop_total', np.full(len(Fc), n, np.int32))
    me.update(calc_edges=True)
    return me


def mesh_arrays(me):
    """-> V (game), T (tris)."""
    V = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', V)
    me.calc_loop_triangles()
    T = np.zeros(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get('vertices', T)
    return b2g(V.reshape(-1, 3)), T.reshape(-1, 3)


def decimated(V, Fc, tris, name='tmp'):
    """Collapse-decimate to ~tris triangles. Returns a new triangulated bpy mesh."""
    me = new_mesh(name, V, Fc)
    cur = len(Fc) * (2 if np.asarray(Fc).shape[1] == 4 else 1)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if cur > tris:
        m = ob.modifiers.new('dec', 'DECIMATE')
        m.ratio = max(0.002, tris / cur)
        m.use_collapse_triangulate = True
    ob.modifiers.new('tri', 'TRIANGULATE')
    dg = bpy.context.evaluated_depsgraph_get()
    out = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), depsgraph=dg)
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    out.name = name
    return out


def split_by_mask(me, fn, mat_index):
    """Cut the mesh along the zero contour of fn(P_game) and give faces with fn<0 `mat_index`.
    Produces a clean material boundary instead of a jagged triangle staircase."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    P = b2g(np.array([v.co[:] for v in bm.verts]))
    val = np.asarray(fn(P), float)
    lay = bm.verts.layers.float.new('mask_tmp')
    for v, x in zip(bm.verts, val):
        v[lay] = x if abs(x) > 1e-7 else 1e-7
    cross = []
    for e in bm.edges:
        a, b = e.verts
        va, vb = a[lay], b[lay]
        if va * vb < 0:
            t = va / (va - vb)
            if t < 0.15:
                a[lay] = 0.0
            elif t > 0.85:
                b[lay] = 0.0
            else:
                cross.append(e)
    for e in cross:
        a, b = e.verts
        va, vb = a[lay], b[lay]
        if va * vb < 0:
            _, nv = bmesh.utils.edge_split(e, a, va / (va - vb))
            nv[lay] = 0.0
    for f in list(bm.faces):
        on = [v for v in f.verts if v[lay] == 0.0]
        if len(on) == 2 and not any(on[1] in e.verts for e in on[0].link_edges):
            try:
                bmesh.utils.face_split(f, on[0], on[1])
            except Exception:
                pass
    for f in bm.faces:
        if sum(v[lay] for v in f.verts) < 0:
            f.material_index = mat_index
    bm.verts.layers.float.remove(lay)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bm.to_mesh(me)
    bm.free()
    me.update()


def cube_uv(me, scale):
    """Seam-tolerant box projection with consistent handedness (tileable detail normal maps)."""
    nl = len(me.loops)
    co = np.zeros(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    li = np.zeros(nl, np.int32); me.loops.foreach_get('vertex_index', li)
    pn = np.zeros(len(me.polygons) * 3, np.float32); me.polygons.foreach_get('normal', pn); pn = pn.reshape(-1, 3)
    ls = np.zeros(len(me.polygons), np.int32); me.polygons.foreach_get('loop_start', ls)
    lt = np.zeros(len(me.polygons), np.int32); me.polygons.foreach_get('loop_total', lt)
    lp = np.repeat(np.arange(len(me.polygons)), lt)
    n = pn[lp]; p = co[li]
    ax = np.abs(n).argmax(1); sg = np.sign(n[np.arange(nl), ax]); sg[sg == 0] = 1
    u = np.where(ax == 0, p[:, 1] * sg, np.where(ax == 1, -p[:, 0] * sg, p[:, 0] * sg))
    v = np.where(ax == 2, p[:, 1], p[:, 2])
    uv = np.stack([u, v], 1) * scale
    lay = me.uv_layers.new(name='UVMap') if not me.uv_layers else me.uv_layers[0]
    try:
        lay.uv.foreach_set('vector', uv.astype(np.float32).ravel())
    except Exception:
        lay.data.foreach_set('uv', uv.astype(np.float32).ravel())


def set_vcol(me, grey):
    grey = np.asarray(grey, np.float32)
    rgba = np.ones((len(grey), 4), np.float32)
    rgba[:, :3] = grey[:, None] if grey.ndim == 1 else grey
    ca = me.color_attributes.get('Col') or me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', rgba.ravel())
    try:
        me.color_attributes.active_color = ca
        me.color_attributes.render_color_index = list(me.color_attributes).index(ca)
    except Exception:
        pass


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


class Builder:
    def __init__(self, frame_id, colors, out_dir, detail='scales', metal=False, eye_accent=False, scale=1.0):
        global SCALE
        SCALE = scale
        self.scale = scale
        self.id = frame_id
        self.colors = colors
        self.out_dir = out_dir
        self.tex_dir = os.path.join(out_dir, 'tex')
        self.detail = detail
        self.uv_scale = 0.35
        self.pal = materials.palette(colors, metal=metal, eye_accent=eye_accent)
        self._mats = {}
        self.pieces = {}
        self.skins = []
        self.bones = []
        self.mounts = []
        self.body = None
        self.objects = []
        self.rig = None
        self.scale_ref = None       # optional: frame asks for a size-comparison shot
        self.t3_scale = HP_SIZE[3]  # preview scale for the fully-upgraded render
        self.log = []

    # ------------------------------------------------------------ materials
    def mat(self, name, **kw):
        """Override / define a material slot before it is first used. kw: color (hex or rgb), rough, metal,
        emit, emit_strength, vcol, detail, double."""
        spec = dict(self.pal.get(name, dict(color=np.array([0.5, 0.5, 0.5]), rough=0.6)))
        for k, v in kw.items():
            spec[k] = materials.hex_rgb(v) if isinstance(v, str) and v.startswith('#') else v
        self.pal[name] = spec
        return spec

    def _material(self, name):
        if name not in self._mats:
            if name not in self.pal:
                raise KeyError(f'unknown material {name}; define it with K.mat()')
            self._mats[name] = materials.build_material(name, self.pal[name], self.tex_dir, self.pal[name].get('detail_kind', self.detail))
        return self._mats[name]

    # ------------------------------------------------------------ sdf / skin
    def sdf(self):
        return S.SDF()

    def skin(self, sdf, name='body', voxel=0.05, tris=22000, belly=None, regions=(), tint=None, ao=0.6, main=True,
             custom_normals=True, base_mat='body', sharp=None):
        """Mesh an SDF into a skinned object. `belly` / regions=[(mat, mask)] cut clean material regions;
        a mask is an SDF (inside = region) or fn(P)->values (<0 = region). tint(P, N)->grey multiplier."""
        if main:
            self.body = sdf
        V, Q = sdf.mesh(voxel)
        me = decimated(V, Q, tris, name)
        mats = [base_mat]
        regs = ([('belly', belly)] if belly is not None else []) + list(regions)
        for mname, mask in regs:
            if mname not in mats:
                mats.append(mname)
            fn = mask.at if hasattr(mask, 'at') else mask
            split_by_mask(me, fn, mats.index(mname))
        for m in mats:
            me.materials.append(self._material(m))
        me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
        P, _ = mesh_arrays(me)
        N = sdf.grad(P, h=voxel * 0.8)
        if custom_normals:
            try:
                me.normals_split_custom_set_from_vertices([tuple(n) for n in g2b(N)])
            except Exception as e:
                self.log.append(f'custom normals failed: {e}')
        if sharp and not custom_normals:
            try:
                me.set_sharp_from_angle(angle=math.radians(sharp))
            except Exception:
                pass
        occ = sdf.ao(P, N)
        grey = (1 - ao) + ao * occ ** 1.3
        grey *= 1.0 - 0.10 * np.clip(N[:, 1], 0, 1)                      # counter-shading
        grey *= 1.0 + 0.10 * fbm(P[:, 0], P[:, 1], P[:, 2], 3, freq=1.3, seed=5)  # mottling
        if tint is not None:
            grey = grey * np.asarray(tint(P, N), np.float32)
        set_vcol(me, np.clip(grey, 0.05, 1.0))
        cube_uv(me, self.uv_scale / SCALE)
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        self.skins.append((ob, sdf))
        self.log.append(f'skin {name}: {len(Q) * 2} -> {len(me.polygons)} tris')
        return ob

    def snap(self, P, sdf=None):
        """Project point(s) onto the body surface -> (points, outward normals)."""
        P = np.asarray(P, float)
        p, n = (sdf or self.body).snap(P.reshape(-1, 3))
        return (p[0], n[0]) if P.ndim == 1 else (p, n)

    # ------------------------------------------------------------ rig
    def bone(self, name, head, tail, parent=None, r=1.0):
        """r ~ body radius around the bone; larger r = the bone claims vertices from farther away."""
        self.bones.append(dict(name=name, head=np.asarray(head, float), tail=np.asarray(tail, float), parent=parent, r=r))

    def chain(self, prefix, pts, parent=None, r=1.0):
        """Bone chain through points; r scalar or (start, end). Returns last bone name."""
        pts = np.asarray(pts, float)
        r0, r1 = (r, r) if np.ndim(r) == 0 else r
        for i in range(len(pts) - 1):
            nm = f'{prefix}{i + 1}' if len(pts) > 2 else prefix
            self.bone(nm, pts[i], pts[i + 1], parent, r0 + (r1 - r0) * i / max(1, len(pts) - 2))
            parent = nm
        return parent

    def mount(self, name, pos, bone=None):
        self.mounts.append((name, np.asarray(pos, float), bone))

    # ------------------------------------------------------------ rigid pieces
    def piece(self, part, V, T, mat='horn', smooth=True, col=None, bone=None):
        V = np.asarray(V, float)
        if col is None:
            col = np.ones(len(V))
        self.pieces.setdefault(part, []).append(dict(V=V, T=np.asarray(T, np.int64), mat=mat, smooth=smooth, col=np.broadcast_to(col, (len(V),)).astype(float), bone=bone))

    def horn(self, part, pts, radii, mat='horn', sides=6, per=2, flat=1.0, bone=None, shade=(0.55, 1.0), up=None):
        """Curved tapered spike / claw / tooth / horn through `pts` (end with radius 0 for a sharp tip)."""
        V, T, u = geom.sweep(pts, radii, sides, per, flat, up)
        self.piece(part, V, T, mat, col=shade[0] + (shade[1] - shade[0]) * u, bone=bone)

    def spike(self, part, base, n, length, r=None, mat='horn', bend=None, sides=6, bone=None, sink=0.15):
        """Straight (or gently bent) cone standing on the surface at `base` along `n`."""
        base, n = np.asarray(base, float), geom.unit(n)
        r = length * 0.22 if r is None else r
        b = np.zeros(3) if bend is None else np.asarray(bend, float) * length
        pts = [base - n * r * sink * 3, base + n * length * 0.5 + b * 0.25, base + n * length + b]
        self.horn(part, pts, [r * 1.1, r * 0.6, 0.0], mat, sides, bone=bone)

    def claw(self, part, base, direction, length, r, curl=(0, -1, 0), mat='horn', bone=None, flat=0.8):
        """Hooked claw: grows along `direction`, curls toward `curl`."""
        base, d, c = np.asarray(base, float), geom.unit(direction), np.asarray(curl, float)
        pts = [base - d * r, base + d * length * 0.45 + c * length * 0.04, base + d * length * 0.85 + c * length * 0.22, base + d * length + c * length * 0.5]
        self.horn(part, pts, [r, r * 0.85, r * 0.45, 0.0], mat, 6, flat=flat, bone=bone)

    def crystal(self, part, base, n, h, r=None, mat='accent', bone=None, cluster=True, seed=0):
        """Faceted glowing crystal (cluster of 3 by default) growing out of the surface."""
        base, n = np.asarray(base, float), geom.unit(n)
        r = h * 0.3 if r is None else r
        rng = np.random.default_rng(seed + int(abs(base).sum() * 1000) % 100000)
        x, _, z = geom.basis(n)
        todo = [(base, n, h, r)]
        if cluster:
            for k in range(2):
                a = rng.uniform(0, 6.28)
                off = (np.cos(a) * x + np.sin(a) * z)
                todo.append((base + off * r * 0.9, geom.unit(n + off * rng.uniform(0.35, 0.7)), h * rng.uniform(0.45, 0.7), r * 0.65))
        for b, nn, hh, rr in todo:
            V, T, u = geom.crystal(b, nn, hh, rr)
            self.piece(part, V, T, mat, smooth=False, bone=bone)

    def ball(self, part, c, r, mat='eye', rot=None, seg=12, rings=8, bone=None, col=None):
        V, T, _ = geom.sphere(c, r, seg, rings, rot)
        self.piece(part, V, T, mat, bone=bone, col=col)

    def eye(self, part, c, n, r, bone=None, pupil=True, lid=True, up=(0, 1, 0)):
        """Glowing eyeball + dark slit pupil + dark socket rim. `n` = gaze direction."""
        c, n = np.asarray(c, float), geom.unit(n)
        x, y, z = geom.basis(n, up)
        if lid:
            V, T, _ = geom.sphere((0, 0, 0), 1, 10, 6)
            V = c - n * r * 0.35 + (V[:, [0]] * x * 1.3 + V[:, [1]] * y * 0.75 + V[:, [2]] * z * 1.12) * r
            self.piece(part, V, T, 'dark', bone=bone)
        self.ball(part, c, r, 'eye', bone=bone)
        if pupil:
            V, T, _ = geom.sphere((0, 0, 0), 1, 8, 6)
            V = c + n * r * 0.72 + (V[:, [0]] * x * 0.2 + V[:, [1]] * y * 0.34 + V[:, [2]] * z * 0.62) * r
            self.piece(part, V, T, 'dark', bone=bone)

    def fin(self, part, base, n, t, h=1.2, w=1.0, mat='membrane', bone=None, sweep_back=0.45):
        V, T, u = geom.fin_profile(base, n, t, h, w, sweep_back=sweep_back)
        self.piece(part, V, T, mat, col=0.6 + 0.4 * u, bone=bone)

    def leaf(self, part, base, n, t, h, w, th=0.12, lobes=5, mat='plate', bone=None, seed=0, core=False):
        """Jagged bony dorsal plate (maple-leaf style): a bi-convex blade on a spiky outline.
        core=True builds the smaller, fatter inner blade that pokes through the faces (glow core)."""
        x, y, z = geom.basis(n, t)
        base = np.asarray(base, float)
        m = 6 * lobes + 1
        k = np.linspace(0, 1, m)
        tri = np.abs(((k * lobes) % 1.0) - 0.5) * 2          # 1 at lobe valleys
        jag = 1.0 - 0.3 * tri
        ang = np.pi * k
        sc = 0.6 if core else 1.0
        oz = np.cos(ang) * w * 0.5 * (0.75 + 0.25 * jag) * sc
        oy = (np.sin(ang) ** 0.75 * h * jag) * sc + (0.12 * h if core else -0.12 * h)
        outline = base[None] + y[None] * oy[:, None] + z[None] * oz[:, None]
        c0 = base + y * h * 0.36
        thick = th * (1.45 if core else 1.0)
        V = np.vstack([outline, c0 + x * thick, c0 - x * thick])
        T = []
        for i in range(m - 1):
            T += [(i, i + 1, m), (i + 1, i, m + 1)]
        T += [(m - 1, 0, m), (0, m - 1, m + 1)]
        T = geom._consistent(V, np.array(T))
        col = np.concatenate([0.5 + 0.5 * (oy / h).clip(0, 1), [0.75, 0.75]])
        self.piece(part, V, T, mat, smooth=False, col=col, bone=bone)

    def sdf_piece(self, part, sdf, voxel=0.05, tris=600, mat='plate', bone=None, col=None, lo=None, hi=None, ao=0.0):
        """Mesh any SDF as a rigid piece (armour, clubs, masks ...)."""
        V, Q = sdf.mesh(voxel, margin=0.15, lo=lo, hi=hi)
        if not len(Q):
            self.log.append(f'{part}: empty sdf piece')
            return
        me = decimated(V, Q, tris, 'piece')
        V, T = mesh_arrays(me)
        bpy.data.meshes.remove(me)
        if ao and col is None:
            N = sdf.grad(V)
            col = (1 - ao) + ao * sdf.ao(V, N, steps=(0.08, 0.2, 0.4))
        self.piece(part, V, T, mat, col=col, bone=bone)

    def plate(self, part, c, r, off=0.05, th=0.08, mat='plate', sdf=None, voxel=0.045, tris=360, rot=None, k=0.06, bone=None,
              snap=True, region=None, lames=0, lame_axis=(0, 1, 0), lame_step=0.11, bevel=True):
        """Armour plate that hugs the body: a thin offset shell of the body SDF clipped to an ellipsoid
        (or any `region` prim). lames=N splits it into N overlapping bands along lame_axis."""
        base = sdf or self.body
        c = np.asarray(c, float)
        if snap:
            c, _ = self.snap(c, base)
        reg = region or S.Ell(c, r, rot)
        band = None
        if lames:
            ax = geom.unit(lame_axis)
            ext = float(np.abs(np.asarray(reg.hi) - np.asarray(reg.lo)) @ np.abs(ax))
            band = (ax, ext / lames, lame_step, float(c @ ax))
        w = 0.3 * float(np.min(np.asarray(reg.hi) - np.asarray(reg.lo))) / 2
        cut = S.Inter(S.Shell(base, off, th, band, taper=(reg, max(w, 0.08), off + th * 0.75) if bevel else None), reg, k)
        P = S.SDF()
        P.ops.append(('add', cut, 0.0))
        P._lo, P._hi = cut.lo.copy(), cut.hi.copy()
        V, Q = P.mesh(voxel, margin=0.12)
        if len(Q) < 8:
            self.log.append(f'{part}: empty plate at {np.round(c, 2)}')
            return
        me = decimated(V, Q, tris, 'plate')
        V, T = mesh_arrays(me)
        bpy.data.meshes.remove(me)
        dr = reg.dist(S.Points(V))
        e = np.clip(1 + dr / 0.22, 0, 1)
        col = 0.66 + 0.34 * e - 0.5 * np.clip(1 + dr / 0.05, 0, 1) ** 2   # worn highlight, dark seated edge
        if band:
            col *= 0.8 + 0.2 * (((V @ band[0]) - band[3]) / band[1] % 1.0)
        self.piece(part, V, T, mat, col=np.clip(col, 0, 1), bone=bone)

    def frill(self, part, base, n, t, h=1.2, w=1.0, rays=5, spread=110, mat='membrane', ray_mat='horn', bone=None, scallop=0.18):
        """Spined fin / frill: `rays` bony spines fanning in the (n, t) plane with a scalloped membrane."""
        _, y, z = geom.basis(n, t)
        base = np.asarray(base, float)
        roots, tips = [], []
        for i in range(rays):
            f = i / (rays - 1)
            a = math.radians(spread) * (f - 0.5)
            ln = h * (0.62 + 0.38 * math.sin(math.pi * (0.15 + 0.7 * f)))
            root = base + z * w * 0.6 * (f - 0.5) - y * 0.05 * h
            roots.append(root)
            tips.append(root + (math.cos(a) * y + math.sin(a) * z) * ln)
            self.horn(part, [root, root * 0.5 + tips[-1] * 0.5, tips[-1]], [0.045 * h, 0.03 * h, 0.0], ray_mat, sides=4, bone=bone, shade=(0.7, 1.0))
        V, T, col = [], [], []
        for i in range(rays - 1):
            o = len(V)
            mid = (tips[i] + tips[i + 1]) * 0.5
            mid = mid + ((roots[i] + roots[i + 1]) * 0.5 - mid) * scallop
            V += [roots[i], roots[i + 1], tips[i + 1], mid, tips[i]]
            col += [0.5, 0.5, 1.0, 0.9, 1.0]
            T += [(o, o + 1, o + 3), (o + 1, o + 2, o + 3), (o, o + 3, o + 4)]
        self.piece(part, np.array(V), np.array(T), mat, col=np.array(col), bone=bone, smooth=False)

    # ------------------------------------------------------------ finalize
    def _weights(self, P, edges=None, smooth_iters=4):
        nb = len(self.bones)
        D = np.zeros((len(P), nb))
        for j, b in enumerate(self.bones):
            D[:, j] = seg_dist(P, b['head'], b['tail']) / max(b['r'], 1e-3)
        W = 1.0 / (D ** 4 + 1e-3)
        W /= W.sum(1, keepdims=True)
        if edges is not None and len(edges):
            deg = np.zeros(len(P)); np.add.at(deg, edges[:, 0], 1); np.add.at(deg, edges[:, 1], 1)
            for _ in range(smooth_iters):
                acc = np.zeros_like(W)
                np.add.at(acc, edges[:, 0], W[edges[:, 1]]); np.add.at(acc, edges[:, 1], W[edges[:, 0]])
                W = 0.5 * W + 0.5 * acc / np.maximum(deg, 1)[:, None]
        if nb > 4:
            cut = np.sort(W, 1)[:, -4][:, None]
            W = np.where(W >= cut, W, 0)
        W[W < 0.02] = 0
        return W / np.maximum(W.sum(1, keepdims=True), 1e-9)

    def _nearest_bone(self, c):
        d = [seg_dist(c[None], b['head'], b['tail'])[0] / max(b['r'], 1e-3) for b in self.bones]
        return self.bones[int(np.argmin(d))]['name']

    def _assign_groups(self, ob, W):
        for j, b in enumerate(self.bones):
            idx = np.nonzero(W[:, j] > 0)[0]
            if not len(idx):
                continue
            vg = ob.vertex_groups.new(name=b['name'])
            for i in idx:
                vg.add([int(i)], float(W[i, j]), 'REPLACE')

    def _piece_object(self, name, pieces):
        V = np.vstack([p['V'] for p in pieces])
        offs = np.cumsum([0] + [len(p['V']) for p in pieces[:-1]])
        T = np.vstack([p['T'] + o for p, o in zip(pieces, offs)])
        me = new_mesh(name, V, T)
        mats = []
        mi, sm = [], []
        for p in pieces:
            if p['mat'] not in mats:
                mats.append(p['mat'])
            mi += [mats.index(p['mat'])] * len(p['T'])
            sm += [p['smooth']] * len(p['T'])
        for m in mats:
            me.materials.append(self._material(m))
        me.polygons.foreach_set('material_index', np.array(mi, np.int32))
        me.polygons.foreach_set('use_smooth', np.array(sm, bool))
        try:
            me.set_sharp_from_angle(angle=math.radians(50))
        except Exception:
            pass
        set_vcol(me, np.concatenate([p['col'] for p in pieces]))
        if any(self.pal.get(m, {}).get('detail') for m in mats):   # detail-mapped slots need real UVs
            cube_uv(me, self.uv_scale / SCALE)
        me.update()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        if self.bones:
            names = [b['name'] for b in self.bones]
            W = np.zeros((len(V), len(names)))
            for p, o in zip(pieces, offs):
                bn = p['bone'] or self._nearest_bone(p['V'].mean(0))
                W[o:o + len(p['V']), names.index(bn)] = 1
            self._assign_groups(ob, W)
        return ob

    def finalize(self):
        scene = bpy.context.scene
        arm_ob = None
        if self.bones:
            if not any(b['name'] == 'root' for b in self.bones):
                self.bones.insert(0, dict(name='root', head=np.zeros(3), tail=np.array([0, 0.6, 0.0]), parent=None, r=0.05))
            arm = bpy.data.armatures.new('rig')
            arm_ob = bpy.data.objects.new('rig', arm)
            scene.collection.objects.link(arm_ob)
            bpy.context.view_layer.objects.active = arm_ob
            bpy.ops.object.mode_set(mode='EDIT')
            for b in self.bones:
                eb = arm.edit_bones.new(b['name'])
                eb.head, eb.tail = g2b(b['head']), g2b(b['tail'])
                par = b['parent'] or ('root' if b['name'] != 'root' else None)
                if par:
                    eb.parent = arm.edit_bones[par]
            bpy.ops.object.mode_set(mode='OBJECT')
        else:
            arm_ob = bpy.data.objects.new('rig', None)
            scene.collection.objects.link(arm_ob)
        self.rig = arm_ob
        objs = {}
        for ob, _ in self.skins:
            if self.bones:
                me = ob.data
                P, _t = mesh_arrays(me)
                E = np.zeros(len(me.edges) * 2, np.int32); me.edges.foreach_get('vertices', E)
                self._assign_groups(ob, self._weights(P, E.reshape(-1, 2)))
            objs[ob.name] = ob
        for name, pieces in self.pieces.items():
            ob = self._piece_object(name + ('__p' if name in objs else ''), pieces)
            if name in objs:  # rigid base pieces (teeth, eyes ...) join the skin object
                with bpy.context.temp_override(active_object=objs[name], selected_editable_objects=[objs[name], ob], selected_objects=[objs[name], ob]):
                    bpy.ops.object.join()
            else:
                objs[name] = ob
        for ob in objs.values():
            ob.parent = arm_ob
            if self.bones:
                md = ob.modifiers.new('Armature', 'ARMATURE')
                md.object = arm_ob
        if self.bones:
            names = {b['name']: b for b in self.bones}
        for name, pos, bone in self.mounts:
            e = bpy.data.objects.new(name, None)
            e.empty_display_size = 0.3 * SCALE
            scene.collection.objects.link(e)
            e.parent = arm_ob
            world = Matrix.Translation(tuple(g2b(pos)))
            if self.bones:
                bn = bone or self._nearest_bone(pos)
                pb = arm_ob.data.bones[bn]
                e.parent_type = 'BONE'
                e.parent_bone = bn
                e.matrix_parent_inverse = Matrix.Identity(4)
                e.matrix_basis = (pb.matrix_local @ Matrix.Translation((0, pb.length, 0))).inverted() @ world
            else:
                e.matrix_basis = world
            objs[name] = e
        self.objects = list(objs.values())
        bpy.context.view_layer.update()
        return self.objects

    # ------------------------------------------------------------ tiers (preview only)
    def set_tier(self, t):
        """t=0 base, t=3 fully upgraded. pow_* is exclusive, other families additive."""
        for ob in self.objects:
            m = re.match(r'^([a-z]+)_t(\d)$', ob.name)
            hide = False
            if m:
                fam, n = m.group(1), int(m.group(2))
                hide = (n != t) if fam == 'pow' else (n > t)
            elif ob.name.startswith('vis_'):
                hide = t < 3
            ob.hide_render = hide
            ob.hide_viewport = hide
        s = self.t3_scale if t >= 3 else 1.0
        self.rig.scale = (s, s, s)
        bpy.context.view_layer.update()
