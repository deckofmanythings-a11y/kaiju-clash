"""Studio, camera fitting, EEVEE render and the contact sheet."""
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def setup_render(scene, res=1024, samples=48):
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    ee = scene.eevee
    for k, v in (('taa_render_samples', samples), ('use_raytracing', True), ('use_shadows', True),
                 ('shadow_ray_count', 2), ('shadow_step_count', 6), ('use_gtao', True)):
        try:
            setattr(ee, k, v)
        except Exception:
            pass
    try:
        scene.view_settings.view_transform = 'AgX'
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass
    world = bpy.data.worlds.new('studio')
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
    bg.inputs[0].default_value = (0.42, 0.45, 0.5, 1)
    bg.inputs[1].default_value = 0.4
    scene.world = world
    _glare(scene)


def _glare(scene):
    """Soft bloom on emissive accents (compositor). Best effort across API versions."""
    try:
        tree = bpy.data.node_groups.new('kaiju_comp', 'CompositorNodeTree')
        tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        rl = tree.nodes.new('CompositorNodeRLayers')
        gl = tree.nodes.new('CompositorNodeGlare')
        out = tree.nodes.new('NodeGroupOutput')
        for key, val in (('Type', 'Bloom'), ('Quality', 'High'), ('Threshold', 1.6), ('Highlights Threshold', 1.6),
                         ('Strength', 0.55), ('Size', 0.45), ('Saturation', 1.0)):
            try:
                gl.inputs[key].default_value = val
            except Exception:
                pass
        try:
            gl.glare_type = 'BLOOM'
        except Exception:
            pass
        tree.links.new(rl.outputs[0], gl.inputs[0])
        tree.links.new(gl.outputs[0], out.inputs[0])
        scene.compositing_node_group = tree
        try:
            scene.render.use_compositing = True
        except Exception:
            pass
    except Exception as e:
        print('[render] glare unavailable:', e)


def world_verts(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for ob in objs:
        if ob.type != 'MESH' or ob.hide_render:
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.zeros(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get('co', co)
        M = np.array(ev.matrix_world)
        out.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        ev.to_mesh_clear()
    return np.vstack(out) if out else np.zeros((1, 3))


def make_stage(scene, P, color=(0.06, 0.066, 0.076)):
    """Curved backdrop + 3 sun lights, sized from the model's world-space vertices P."""
    lo, hi = P.min(0), P.max(0)
    R = float(np.linalg.norm(hi - lo)) * 0.5
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    W, front, back, H, rc = R * 14, R * 12, R * 3.2, R * 12, R * 2.5
    prof = [(-front, 0.0)] + [(back - rc + rc * math.sin(a), rc - rc * math.cos(a)) for a in np.linspace(0, math.pi / 2, 14)] + [(back, H)]
    V, Fc = [], []
    for sx in (-W, W):
        for y, z in prof:
            V.append((cx + sx, cy + y, z))
    n = len(prof)
    for i in range(n - 1):
        Fc.append((i, n + i, n + i + 1, i + 1))
    me = bpy.data.meshes.new('cyc')
    me.from_pydata(V, [], Fc)
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    mat = bpy.data.materials.new('cyc')
    mat.use_nodes = True
    b = next(nd for nd in mat.node_tree.nodes if nd.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (*color, 1)
    b.inputs['Roughness'].default_value = 0.85
    me.materials.append(mat)
    cyc = bpy.data.objects.new('cyc', me)
    scene.collection.objects.link(cyc)
    made = [cyc]
    for name, strength, col, rot, ang, shadow in (
            ('key', 4.2, (1.0, 0.95, 0.88), (52, 0, -38), 12, True),
            ('fill', 1.1, (0.75, 0.85, 1.0), (65, 0, 60), 40, False),
            ('rim', 3.4, (0.85, 0.92, 1.0), (-58, 0, 150), 8, False)):
        L = bpy.data.lights.new(name, 'SUN')
        L.energy, L.color, L.angle = strength, col, math.radians(ang)
        try:
            L.use_shadow = shadow
        except Exception:
            pass
        ob = bpy.data.objects.new(name, L)
        ob.rotation_euler = [math.radians(a) for a in rot]
        scene.collection.objects.link(ob)
        made.append(ob)
    return made


def fit_camera(scene, P, az=35, el=9, lens=85, margin=1.07):
    cam = scene.camera
    if cam is None:
        cd = bpy.data.cameras.new('cam')
        cam = bpy.data.objects.new('cam', cd)
        scene.collection.objects.link(cam)
        scene.camera = cam
    cam.data.lens = lens
    cam.data.sensor_width = 36
    a, e = math.radians(az), math.radians(el)
    f = -np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])  # view dir (cam at -Y side)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c = (P.min(0) + P.max(0)) / 2
    th = math.atan(18 / lens)
    for _ in range(3):
        rel = P - c
        x, y, z = rel @ r, rel @ u, rel @ f
        c = c + r * (x.min() + x.max()) / 2 + u * (y.min() + y.max()) / 2
        rel = P - c
        x, y, z = rel @ r, rel @ u, rel @ f
        d = float(np.max(np.maximum(np.abs(x), np.abs(y)) / math.tan(th) - z)) * margin
    pos = c - f * d
    M = Matrix(((r[0], u[0], -f[0], pos[0]), (r[1], u[1], -f[1], pos[1]), (r[2], u[2], -f[2], pos[2]), (0, 0, 0, 1)))
    cam.matrix_world = M
    cam.data.clip_start, cam.data.clip_end = max(0.01, d * 0.01), d * 40
    return cam


def render(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def silhouette_material():
    m = bpy.data.materials.new('silhouette')
    m.use_nodes = True
    b = next(nd for nd in m.node_tree.nodes if nd.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (0.02, 0.022, 0.026, 1)
    b.inputs['Roughness'].default_value = 0.9
    return m


def import_silhouette(glb_path, offset):
    """Load a built kaiju GLB as a dark size-reference statue."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    new = [o for o in bpy.data.objects if o not in before]
    mat = silhouette_material()
    import re
    for o in new:
        if o.type == 'MESH':
            if re.match(r'^(pow_t[123]|arm_t\d|enr_t\d|spd_t\d|fcs_t\d|vis_.*)', o.name):
                o.hide_render = True
            o.data.materials.clear()
            o.data.materials.append(mat)
    roots = [o for o in new if o.parent is None]
    for o in roots:
        o.location = Vector(offset)
    bpy.context.view_layer.update()
    return new


# ------------------------------------------------------------------ contact sheet
def contact_sheet(render_dir, frames, out_path, cell=384, cols=8):
    scene = reset_scene()
    scene.render.engine = 'BLENDER_EEVEE'
    try:
        scene.view_settings.view_transform = 'Standard'
    except Exception:
        pass
    world = bpy.data.worlds.new('w'); world.use_nodes = True
    next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND').inputs[0].default_value = (0.012, 0.013, 0.016, 1)
    scene.world = world
    try:
        scene.eevee.taa_render_samples = 8
    except Exception:
        pass
    rows = math.ceil(len(frames) / cols)
    label_h, gap = 0.22, 0.03
    band = 2 + label_h + gap * 2          # in cell units: label + t0 + t3
    Wt, Ht = cols * (1 + gap) + gap, rows * band + gap

    def emit_mat(name, img=None, col=(1, 1, 1)):
        m = bpy.data.materials.new(name); m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            if n.type != 'OUTPUT_MATERIAL':
                nt.nodes.remove(n)
        outn = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
        em = nt.nodes.new('ShaderNodeEmission')
        if img is not None:
            tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img
            nt.links.new(tex.outputs['Color'], em.inputs['Color'])
        else:
            em.inputs['Color'].default_value = (*col, 1)
        nt.links.new(em.outputs[0], outn.inputs['Surface'])
        return m

    def plane(name, x, y, w, h, mat):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(x, y, 0), (x + w, y, 0), (x + w, y + h, 0), (x, y + h, 0)], [], [(0, 1, 2, 3)])
        uv = me.uv_layers.new(name='UVMap')
        for i, c in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            uv.data[i].uv = c
        me.materials.append(mat)
        ob = bpy.data.objects.new(name, me); scene.collection.objects.link(ob)
        return ob

    def text(s, x, y, size, col=(0.92, 0.93, 0.95), align='CENTER'):
        cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.size = size; cu.align_x = align
        ob = bpy.data.objects.new('t', cu); ob.location = (x, y, 0.01)
        cu.materials.append(emit_mat('txt', col=col))
        scene.collection.objects.link(ob)

    for i, (fid, title) in enumerate(frames):
        cx, ry = i % cols, i // cols
        x = gap + cx * (1 + gap)
        top = Ht - gap - ry * band
        text(title, x + 0.5, top - label_h * 0.78, 0.115)
        for k, tier in enumerate(('t0', 't3')):
            p = os.path.join(render_dir, f'{fid}_{tier}.png')
            y = top - label_h - (k + 1) * 1.0 - k * gap
            if os.path.exists(p):
                plane(f'{fid}_{tier}', x, y, 1, 1, emit_mat(f'{fid}_{tier}', bpy.data.images.load(p)))
            text('BASE' if tier == 't0' else 'TIER 3', x + 0.03, y + 0.03, 0.055, (0.7, 0.75, 0.8), 'LEFT')
    cd = bpy.data.cameras.new('cam'); cd.type = 'ORTHO'
    cam = bpy.data.objects.new('cam', cd); cam.location = (Wt / 2, Ht / 2, 5)
    scene.collection.objects.link(cam); scene.camera = cam
    scene.render.resolution_x = int(Wt * cell); scene.render.resolution_y = int(Ht * cell)
    cd.ortho_scale = max(Wt, Ht * scene.render.resolution_x / scene.render.resolution_y)
    render(scene, out_path)
