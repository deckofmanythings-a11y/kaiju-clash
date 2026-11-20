"""glTF-safe Principled materials + procedurally generated tileable detail normal maps.

Slots `body`, `belly`, `accent` carry the per-player colours (plain base-colour factor, multiplied by a
grey vertex-colour AO/tint layer). Extra slots (horn, plate, eye, dark, mouth, membrane, skin, ...) are
derived the same way js/render/kaiju-model.js makeMats() derives them.
"""
import os
import bpy
import numpy as np

TEX = 512


def srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def lerp(a, b, t):
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t


# ------------------------------------------------------------------ tileable detail maps
def _pnoise(n, cu, cv, rng):
    lat = rng.random((cu, cv))
    u = np.arange(n) / n * cu
    v = np.arange(n) / n * cv
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - iu, v - iv
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a = lat[iu % cu][:, iv % cv]; b = lat[(iu + 1) % cu][:, iv % cv]
    c = lat[iu % cu][:, (iv + 1) % cv]; d = lat[(iu + 1) % cu][:, (iv + 1) % cv]
    FU, FV = fu[:, None], fv[None, :]
    return a * (1 - FU) * (1 - FV) + b * FU * (1 - FV) + c * (1 - FU) * FV + d * FU * FV


def _pfbm(n, c0, rng, octaves=4, ridged=False, aniso=1):
    out, amp, tot = 0, 1.0, 0
    for o in range(octaves):
        x = _pnoise(n, c0 * 2 ** o * aniso, c0 * 2 ** o, rng)
        out = out + amp * ((1 - np.abs(2 * x - 1)) if ridged else x); tot += amp; amp *= 0.5
    return out / tot


def _pvoronoi(n, c, rng, jitter=0.85):
    pts = (rng.random((c, c, 2)) - 0.5) * jitter + 0.5
    u, v = np.meshgrid(np.arange(n) / n * c, np.arange(n) / n * c, indexing='ij')
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    f1 = np.full((n, n), 9.0); f2 = f1.copy(); cid = np.zeros((n, n))
    rid = rng.random((c, c))
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            cu, cv = iu + du, iv + dv
            p = pts[cu % c, cv % c]
            d = np.hypot(cu + p[..., 0] - u, cv + p[..., 1] - v)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            cid = np.where(closer, rid[cu % c, cv % c], cid)
            f1 = np.where(closer, d, f1)
    return f1, f2, cid


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _height(kind, n=TEX):
    rng = np.random.default_rng(abs(hash(kind)) % 9999 if False else sum(map(ord, kind)))
    if kind == 'scales':
        f1, f2, cid = _pvoronoi(n, 14, rng)
        h = _ss(0.0, 0.22, f2 - f1) * (0.75 + 0.25 * (1 - f1)) + 0.25 * cid
        h += 0.25 * _pfbm(n, 24, rng)
        return h, 1.3
    if kind == 'rock':
        f1, f2, _ = _pvoronoi(n, 7, rng)
        h = _pfbm(n, 6, rng, 5, ridged=True) - 0.7 * (1 - _ss(0, 0.06, f2 - f1)) + 0.3 * _pfbm(n, 40, rng, 2)
        return h, 6.0
    if kind == 'skin':
        f1, f2, _ = _pvoronoi(n, 22, rng)
        h = 0.6 * _pfbm(n, 10, rng, 4, ridged=True) + 0.4 * _ss(0, 0.3, f2 - f1)
        return h, 3.5
    if kind == 'chitin':
        f1, _, _ = _pvoronoi(n, 26, rng)
        h = 0.5 * _pfbm(n, 8, rng, 4) - 0.5 * (1 - _ss(0.0, 0.28, f1)) + 0.2 * _pfbm(n, 6, rng, 3, aniso=6)
        return h, 3.0
    if kind == 'fur':
        h = _pfbm(n, 6, rng, 4, aniso=10) + 0.4 * _pfbm(n, 20, rng, 2, aniso=8)
        return h, 5.0
    if kind == 'metal':
        h = 0.25 * _pfbm(n, 6, rng, 3, aniso=14) + 0.2 * _pfbm(n, 48, rng, 2)
        f1, _, _ = _pvoronoi(n, 9, rng)
        h -= 0.5 * (1 - _ss(0.0, 0.1, f1))  # sparse rivet pits
        return h, 2.0
    if kind == 'cloth':
        u = np.arange(n)[:, None] / n; v = np.arange(n)[None, :] / n
        h = 0.5 * np.sin(u * 2 * np.pi * 64) * np.sin(v * 2 * np.pi * 64) + 0.3 * _pfbm(n, 12, rng, 3)
        return h, 1.5
    raise KeyError(kind)


def detail_image(kind, tex_dir):
    """512px tileable tangent-space normal map (OpenGL convention), cached on disk."""
    name = f'detail_{kind}_n'
    if name in bpy.data.images:
        return bpy.data.images[name]
    os.makedirs(tex_dir, exist_ok=True)
    path = os.path.join(tex_dir, name + '.png')
    if not os.path.exists(path):
        h, strength = _height(kind)
        dx = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * strength * TEX / 64
        dy = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * strength * TEX / 64
        nrm = np.stack([-dx, -dy, np.ones_like(h)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        rgba = np.concatenate([nrm * 0.5 + 0.5, np.ones((TEX, TEX, 1))], -1)
        img = bpy.data.images.new(name + '_gen', TEX, TEX, alpha=False, is_data=True)
        img.pixels.foreach_set(np.transpose(rgba, (1, 0, 2)).astype(np.float32).ravel())  # rows = v
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
        bpy.data.images.remove(img)
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = 'Non-Color'
    return img


# ------------------------------------------------------------------ materials
def palette(colors, metal=False, eye_accent=False):
    body, belly, accent = hex_rgb(colors['body']), hex_rgb(colors['belly']), hex_rgb(colors['accent'])
    eye = accent if eye_accent else hex_rgb('#ffc83a')
    P = {
        'body': dict(color=body, rough=0.36 if metal else 0.62, metal=0.6 if metal else 0.0, detail=True, vcol=True),
        'belly': dict(color=belly, rough=0.42 if metal else 0.72, metal=0.45 if metal else 0.0, detail=True, vcol=True, detail_strength=0.5),
        'accent': dict(color=accent * 0.6, emit=accent, emit_strength=3.0, rough=0.3),
        'plate': dict(color=lerp(body * 0.7, belly, 0.42), rough=0.35 if metal else 0.45, metal=0.7 if metal else 0.15, vcol=True),
        'horn': dict(color=lerp(hex_rgb('#e6dcc4'), belly, 0.25), rough=0.38, vcol=True),
        'dark': dict(color=hex_rgb('#16130f'), rough=0.5, metal=0.6 if metal else 0.0),
        'eye': dict(color=hex_rgb('#0c0c0c'), emit=eye, emit_strength=2.6, rough=0.2),
        'mouth': dict(color=hex_rgb('#6a2026'), rough=0.35, vcol=True),
        'membrane': dict(color=lerp(belly, accent, 0.3), rough=0.6, vcol=True, double=True),
    }
    return P


def build_material(name, spec, tex_dir, detail_kind=None, uv_scale=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    col = tuple(srgb2lin(np.clip(spec['color'], 0, 1))) + (1.0,)
    bsdf.inputs['Base Color'].default_value = col
    bsdf.inputs['Roughness'].default_value = spec.get('rough', 0.6)
    bsdf.inputs['Metallic'].default_value = spec.get('metal', 0.0)
    if spec.get('vcol'):
        ca = nt.nodes.new('ShaderNodeVertexColor'); ca.layer_name = 'Col'
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
        mix.inputs[0].default_value = 1.0
        mix.inputs[6].default_value = col
        nt.links.new(ca.outputs['Color'], mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    if spec.get('emit') is not None:
        bsdf.inputs['Emission Color'].default_value = tuple(srgb2lin(np.clip(spec['emit'], 0, 1))) + (1.0,)
        bsdf.inputs['Emission Strength'].default_value = spec.get('emit_strength', 2.0)
    if spec.get('detail') and detail_kind and detail_kind != 'smooth':
        tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = detail_image(detail_kind, tex_dir)
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value = spec.get('detail_strength', 1.0)
        nt.links.new(tex.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    m.use_backface_culling = not spec.get('double', False)
    if spec.get('alpha') is not None:
        bsdf.inputs['Alpha'].default_value = spec['alpha']
        try:
            m.surface_render_method = 'BLENDED'
        except Exception:
            pass
    return m
