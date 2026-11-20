"""Kaiju Clash model generator.

  blender.exe -b --factory-startup --python build_models.py -- [options]

Options (after the `--`):
  --frames a,b,c     only these frame ids (default: all 16)
  --no-render        build + export GLBs only
  --no-sheet         skip contact_sheet.png
  --sheet-only       only (re)build the contact sheet from existing renders
  --angles 35,160    extra debug views (azimuth degrees) written to out/debug/ at --res
  --res 1024         render resolution
  --tiers 0,3        which tiers to render
  --pose             also render out/debug/<frame>_pose.png with every limb bone rotated (skinning check)

Outputs: out/glb/<frame>.glb, out/renders/<frame>_t0.png / _t3.png (+ _scale.png for the tiny wild
frames), out/renders/contact_sheet.png, out/stats.json.
"""
import importlib
import json
import os
import re
import struct
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402

from kaiju import render as R  # noqa: E402
from kaiju.builder import Builder  # noqa: E402

OUT = os.path.join(HERE, 'out')
FRAMES_JS = os.path.normpath(os.path.join(HERE, '..', '..', 'js', 'data', 'frames.js'))
ORDER = ['saurian', 'ape', 'kraken', 'lepidra', 'carapace', 'mantid', 'serpent', 'mecha', 'golem', 'hydra', 'arachnid',
         'stormray', 'superhero', 'heroteam', 'army', 'magicalgirl']


def frame_table():
    """id -> {name, colors} parsed from js/data/frames.js so colours never drift from the game."""
    src = open(FRAMES_JS, encoding='utf8').read()
    out = {}
    for m in re.finditer(r"id:\s*'(\w+)',\s*name:\s*'([^']+)'.*?colors:\s*\{\s*body:\s*'(#\w+)',\s*belly:\s*'(#\w+)',\s*accent:\s*'(#\w+)'", src, re.S):
        out[m.group(1)] = dict(name=m.group(2), colors=dict(body=m.group(3), belly=m.group(4), accent=m.group(5)))
    return out


def glb_stats(path):
    with open(path, 'rb') as f:
        data = f.read()
    jlen = struct.unpack_from('<I', data, 12)[0]
    js = json.loads(data[20:20 + jlen])
    acc = js.get('accessors', [])
    per, total = {}, 0
    mesh_node = {n['mesh']: n.get('name') for n in js.get('nodes', []) if 'mesh' in n}
    for i, m in enumerate(js.get('meshes', [])):
        t = sum(acc[p['indices']]['count'] // 3 for p in m['primitives'] if 'indices' in p)
        per[mesh_node.get(i, m.get('name'))] = t
        total += t
    base = sum(t for k, t in per.items() if not re.match(r'^(pow_t[123]|arm_t\d|enr_t\d|spd_t\d|fcs_t\d|vis_)', k or ''))
    CT = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
    by_attr = {}
    for m in js.get('meshes', []):
        for p in m['primitives']:
            for k, ai in list(p['attributes'].items()) + [('indices', p.get('indices'))]:
                if ai is not None:
                    a = acc[ai]
                    by_attr[k] = by_attr.get(k, 0) + a['count'] * CT[a['componentType']] * NC[a['type']]
    by_attr['images'] = sum(js['bufferViews'][im['bufferView']]['byteLength'] for im in js.get('images', []) if 'bufferView' in im)
    return dict(bytes=len(data), bytes_by_attr=by_attr, tris_total=total, tris_base=base, tris_by_object=per,
                materials=[m['name'] for m in js.get('materials', [])],
                nodes=sorted(n.get('name', '') for n in js.get('nodes', []) if 'mesh' not in n and 'children' not in n and 'skin' not in n)[:40],
                joints=len(js['skins'][0]['joints']) if js.get('skins') else 0,
                images=len(js.get('images', [])),
                has_vertex_color=any('COLOR_0' in p['attributes'] for m in js.get('meshes', []) for p in m['primitives']))


def export_glb(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    want = dict(filepath=path, export_format='GLB', export_yup=True, export_apply=False, export_skins=True,
                export_animations=False, export_cameras=False, export_lights=False, export_extras=False,
                export_image_format='JPEG', export_image_quality=88, export_jpeg_quality=88, export_vertex_color='MATERIAL',
                export_all_vertex_colors=False, export_tangents=False, export_morph=False, use_selection=False,
                export_def_bones=False, export_try_sparse_sk=False)
    props = {p.identifier for p in bpy.ops.export_scene.gltf.get_rna_type().properties}
    bpy.ops.export_scene.gltf(**{k: v for k, v in want.items() if k in props})


def build_frame(fid, info, args):
    t0 = time.time()
    scene = R.reset_scene()
    mod = importlib.import_module(f'kaiju.frames.{fid}')
    K = Builder(fid, info['colors'], OUT, detail=getattr(mod, 'DETAIL', 'scales'), metal=getattr(mod, 'METAL', False),
                eye_accent=getattr(mod, 'EYE_ACCENT', False), scale=getattr(mod, 'SCALE', 1.0))
    K.t3_scale = getattr(mod, 'T3_SCALE', K.t3_scale)
    K.uv_scale = getattr(mod, 'UV_SCALE', K.uv_scale)
    mod.build(K)
    K.finalize()
    K.set_tier(3)
    glb = os.path.join(OUT, 'glb', f'{fid}.glb')
    K.rig.scale = (1, 1, 1)
    for ob in K.objects:
        ob.hide_viewport = False
    bpy.context.view_layer.update()
    export_glb(glb)
    st = glb_stats(glb)
    st['build_seconds'] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, 'stats'), exist_ok=True)
    json.dump(st, open(os.path.join(OUT, 'stats', fid + '.json'), 'w'), indent=1)
    print(f'[{fid}] glb {st["bytes"] / 1e6:.2f} MB, {st["tris_total"]} tris total ({st["tris_base"]} base) in {st["build_seconds"]}s')
    for line in K.log:
        print('   ', line)
    if not args['no_render']:
        R.setup_render(scene, args['res'])
        cam = getattr(mod, 'CAM', {})
        K.set_tier(3)
        P3 = R.world_verts(K.objects)
        R.make_stage(scene, P3)
        rd = os.path.join(OUT, 'renders')
        for t in args['tiers']:
            K.set_tier(t)
            # same view direction for both tiers; each is framed to fit (t3 is HP_SIZE[3]=1.36x bigger in game)
            R.fit_camera(scene, R.world_verts(K.objects), cam.get('az', 35), cam.get('el', 9))
            R.render(scene, os.path.join(rd, f'{fid}_t{t}.png'))
        if args['pose']:
            import math
            K.set_tier(3)
            for pb in K.rig.pose.bones if K.rig.type == 'ARMATURE' else []:
                if pb.name != 'root' and not pb.name.startswith(('pelvis', 'spine', 'chest')):
                    pb.rotation_mode = 'XYZ'
                    pb.rotation_euler = (math.radians(18), 0, math.radians(10))
            bpy.context.view_layer.update()
            R.fit_camera(scene, R.world_verts(K.objects), cam.get('az', 35), cam.get('el', 9))
            R.render(scene, os.path.join(OUT, 'debug', f'{fid}_pose.png'))
            for pb in K.rig.pose.bones if K.rig.type == 'ARMATURE' else []:
                pb.rotation_euler = (0, 0, 0)
            bpy.context.view_layer.update()
        for az in args['angles']:
            K.set_tier(3)
            R.fit_camera(scene, P3, az, cam.get('el', 9))
            R.render(scene, os.path.join(OUT, 'debug', f'{fid}_a{int(az)}.png'))
        if getattr(mod, 'SCALE_REF', False):
            ref = os.path.join(OUT, 'glb', 'saurian.glb')
            if os.path.exists(ref) and fid != 'saurian':
                K.set_tier(0)
                P0 = R.world_verts(K.objects)
                w = P0[:, 0].max() - P0[:, 0].min()
                new = R.import_silhouette(ref, (P0[:, 0].max() + 4.2, 2.0, 0))
                import numpy as np
                allP = np.vstack([P0, R.world_verts(new)])
                for o in list(scene.collection.objects):
                    if o.name in ('cyc', 'key', 'fill', 'rim'):
                        bpy.data.objects.remove(o)
                R.make_stage(scene, allP)
                R.fit_camera(scene, allP, 20, 6)
                R.render(scene, os.path.join(rd, f'{fid}_scale.png'))
    return st


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    table = frame_table()
    frames = [f for f in (opt('--frames', ','.join(ORDER)).split(',')) if f]
    args = dict(pose='--pose' in argv, no_render='--no-render' in argv, res=int(opt('--res', 1024)),
                angles=[float(a) for a in opt('--angles', '').split(',') if a],
                tiers=[int(t) for t in opt('--tiers', '0,3').split(',')])
    os.makedirs(OUT, exist_ok=True)
    stats_path = os.path.join(OUT, 'stats.json')
    stats = {}
    failed = []
    if '--sheet-only' not in argv:
        for fid in frames:
            try:
                importlib.import_module(f'kaiju.frames.{fid}')
            except ModuleNotFoundError:
                print(f'[{fid}] no frame module yet, skipped')
                continue
            try:
                stats[fid] = build_frame(fid, table[fid], args)
            except Exception:
                traceback.print_exc()
                failed.append(fid)
    if '--no-sheet' not in argv and not args['no_render'] and (len(frames) == len(ORDER) or '--sheet-only' in argv):
        R.contact_sheet(os.path.join(OUT, 'renders'), [(f, table[f]['name']) for f in ORDER], os.path.join(OUT, 'renders', 'contact_sheet.png'))
    sd = os.path.join(OUT, 'stats')
    if os.path.isdir(sd):
        allst = {f[:-5]: json.load(open(os.path.join(sd, f))) for f in sorted(os.listdir(sd)) if f.endswith('.json')}
        json.dump({k: allst[k] for k in ORDER if k in allst}, open(stats_path, 'w'), indent=1)
    print('DONE', 'failed: ' + ','.join(failed) if failed else 'ok')


main()
