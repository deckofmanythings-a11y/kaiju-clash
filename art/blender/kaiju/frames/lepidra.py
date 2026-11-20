"""Lepidra: colossal moth (moth plan), hovering with its four wings spread in a raised V. Game coords, faces +Z.

Fuzzy SDF thorax / collar / head and a segmented striped abdomen; the wings are dense fan meshes on real moth
outlines (falcate forewing, rounded hindwing with a long luna tail), with radial vein ribs, banded vertex
colour, wavy body-coloured cross bands and layered eyespots on both faces.
"""
import math

import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, voronoi, smoothstep

DETAIL = 'fur'
UV_SCALE = 0.7
EYE_ACCENT = True
CAM = dict(az=28, el=12)

Y = np.array([0.0, 1, 0])
THX = np.array([0.0, 5.9, 0.2])                     # thorax centre
HEAD = np.array([0.0, 6.6, 1.9])
AB0, AB1 = np.array([0.0, 5.25, -0.75]), np.array([0.0, 1.95, -3.45])
AB_DIR = (AB1 - AB0) / np.linalg.norm(AB1 - AB0)
AB_LEN = float(np.linalg.norm(AB1 - AB0))
SEG = AB_LEN / 6.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def rot_about(v, axis, deg):
    axis = unit(axis)
    a = math.radians(deg)
    v = np.asarray(v, float)
    return v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * (axis @ v) * (1 - math.cos(a))


# ------------------------------------------------------------------ wings
FORE = dict(root=(0.95, 6.5, 0.55), dihedral=34, sweep=-12, pitch=52, camber=0.014,
            ctrl=[(-24, 0.9), (-15, 3.6), (-8, 6.3), (0, 7.6), (9, 7.2), (19, 6.15), (31, 5.15), (40, 4.6), (55, 3.4), (72, 2.3),
                  (90, 1.35), (104, 0.6)],
            bands=[(0.0, 0.2, 'body', 0.75), (0.2, 0.3, 'belly', 0.7), (0.3, 0.335, 'body', 0.6), (0.335, 0.6, 'belly', 0.92),
                   (0.6, 0.67, 'body', 0.7), (0.67, 0.86, 'belly', 1.0), (0.86, 1.0, 'body', 0.85)],
            eyes=[(17, 0.47, 0.62)], veins=[-14, -4, 6, 16, 27, 39, 54, 72], nt=54)
HIND = dict(root=(0.9, 5.85, -0.25), dihedral=4, sweep=8, pitch=44, camber=0.01,
            ctrl=[(-12, 0.9), (2, 3.6), (14, 4.75), (28, 5.25), (42, 5.2), (54, 4.85), (63, 4.6), (68, 5.5), (71.5, 7.1), (74, 7.9),
                  (76.5, 7.0), (80, 5.3), (86, 3.6), (98, 2.0), (112, 0.7)],
            bands=[(0.0, 0.22, 'body', 0.75), (0.22, 0.58, 'belly', 0.95), (0.58, 0.64, 'body', 0.7), (0.64, 0.86, 'belly', 1.0),
                   (0.86, 1.0, 'body', 0.85)],
            eyes=[(38, 0.5, 0.78)], veins=[0, 14, 28, 42, 56, 70, 74, 90], nt=64)


class WingMap:
    """Fan parametrisation of one wing: (theta, r) -> outline-relative point; (a, b) -> 3D on the cambered sheet."""

    def __init__(self, spec, s):
        self.s = s
        self.root = np.array(spec['root'], float) * [s, 1, 1]
        es = rot_about([1.0, 0, 0], [0, 0, 1], spec['dihedral'])
        es = rot_about(es, [0, 1, 0], spec['sweep'])
        ec = rot_about([0.0, 0, -1], [1, 0, 0], -spec['pitch'])
        ec = unit(ec - es * (ec @ es))
        n = np.cross(es, ec)
        if n[1] < 0:
            n = -n
        m = np.array([s, 1, 1])
        self.es, self.ec, self.n = es * m, ec * m, n * m
        self.camber = spec['camber']
        C = S.catmull(np.array(spec['ctrl'], float), 6)
        o = np.argsort(C[:, 0])
        self.th_c, self.R_c = C[o, 0], C[o, 1]
        self.th0, self.th1 = spec['ctrl'][0][0], spec['ctrl'][-1][0]

    def R(self, th):
        return np.interp(th, self.th_c, self.R_c)

    def ab(self, th, r):
        t = np.radians(th)
        rr = r * self.R(th)
        return rr * np.cos(t), rr * np.sin(t)

    def P(self, a, b, off=0.0):
        a, b = np.asarray(a, float), np.asarray(b, float)
        h = self.camber * a ** 2
        nn = self.normal(a)
        return self.root + a[..., None] * self.es + b[..., None] * self.ec + h[..., None] * self.n + off * nn

    def normal(self, a):
        g = 2 * self.camber * np.asarray(a, float)
        return unit(self.n[None] - g[..., None] * self.es[None]).reshape(np.shape(a) + (3,))

    def tr(self, th, r, off=0.0):
        a, b = self.ab(th, r)
        return self.P(a, b, off)


def wave(th, r, amp=0.035, k=9.0):
    return amp * np.sin(np.radians(th) * k) * np.sin(np.pi * np.clip((r - 0.06) / 0.94, 0, 1))


def build_wing(K, spec, s, bone):
    W = WingMap(spec, s)
    nt = spec['nt']
    ths = np.linspace(W.th0, W.th1, nt + 1)
    lv = set(np.round(np.linspace(0.06, 1.0, 9), 4))
    for r0, r1, _, _ in spec['bands']:
        lv |= {max(r0, 0.06), r1}
    rs = np.array(sorted(lv))
    # vertex grid (theta x r), wavy rings so the cross bands undulate; outline stays exact (wave -> 0 at r=1)
    TH, RR = np.meshgrid(ths, rs, indexing='ij')
    RW = np.clip(RR + wave(TH, RR) * (RR < 0.999), 0.04, 1.0)
    A, Bc = W.ab(TH, RW)
    V = W.P(A, Bc)
    rng = np.random.default_rng(7 + int(s > 0))
    # vertex colour: darker at the base, scalloped chevrons in the margin, faint streaks along the veins
    chev = 0.5 + 0.5 * np.cos(np.radians(TH) * 26)
    col = (0.72 + 0.28 * np.clip(RR * 1.4, 0, 1)) * (1 - 0.25 * chev * (RR > 0.86)) * (1 - 0.08 * (np.cos(np.radians(TH) * 13) > 0.8))
    col *= 1 + 0.06 * fbm(V[..., 0], V[..., 1], V[..., 2], 2, freq=1.2, seed=3)
    nr = len(rs)
    idx = np.arange(V.shape[0] * nr).reshape(V.shape[0], nr)
    Vf, cf = V.reshape(-1, 3), col.reshape(-1)
    by_mat = {}
    for j in range(nr - 1):
        mid = 0.5 * (rs[j] + rs[j + 1])
        mat = next((m for r0, r1, m, _ in spec['bands'] if r0 <= mid < r1), 'belly')
        tone = next((t for r0, r1, _, t in spec['bands'] if r0 <= mid < r1), 1.0)
        for i in range(nt):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            by_mat.setdefault((mat, tone), []).append((a, b, c, d))
    for (mat, tone), quads in by_mat.items():
        Q = np.array(quads)
        used = np.unique(Q)
        remap = -np.ones(len(Vf), int)
        remap[used] = np.arange(len(used))
        T = np.vstack([remap[Q[:, [0, 1, 2]]], remap[Q[:, [0, 2, 3]]]])
        K.piece('body', Vf[used], T, mat, col=np.clip(cf[used] * tone, 0.05, 1), bone=bone)
    # vein ribs
    for th in spec['veins']:
        rv = np.linspace(0.07, 0.97, 9)
        pts = W.tr(np.full_like(rv, th), np.clip(rv + wave(th, rv), 0, 1))
        K.horn('body', pts, [0.06, 0.05, 0.04, 0.032, 0.026, 0.02, 0.016, 0.012, 0.008], mat='dark', sides=4, per=1, bone=bone, shade=(0.6, 0.9))
    # costa: thicker leading-edge rib
    rv = np.linspace(0.06, 0.95, 10)
    th = W.th0 + 5
    K.horn('body', W.tr(np.full_like(rv, th), rv), list(np.linspace(0.09, 0.02, 10)), mat='dark', sides=5, per=1, bone=bone, shade=(0.6, 0.9))
    # eyespots: layered discs on both faces
    for th, r, sz in spec['eyes']:
        a0, b0 = W.ab(th, r)
        for side in (1, -1):
            for k, (mat, rad, dx) in enumerate((('dark', 1.0, 0), ('accent', 0.78, 0), ('dark', 0.46, 0), ('horn', 0.14, 0.18))):
                ang = np.linspace(0, 2 * np.pi, 18, endpoint=False)
                aa = a0 + (np.cos(ang) * sz * rad + dx * sz)
                bb = b0 + (np.sin(ang) * sz * rad * 1.15 - dx * sz * 0.5)
                ring = W.P(aa, bb, side * (0.012 + 0.008 * k))
                ctr = W.P(np.array(a0 + dx * sz), np.array(b0 - dx * sz * 0.5), side * (0.012 + 0.008 * k))
                Vd = np.vstack([ctr[None], ring])
                Td = np.array([(0, 1 + q, 1 + (q + 1) % 18) for q in range(18)])
                fn = np.cross(Vd[Td[:, 1]] - Vd[Td[:, 0]], Vd[Td[:, 2]] - Vd[Td[:, 0]])
                if (fn @ (W.n * side)).sum() < 0:
                    Td = Td[:, ::-1]
                K.piece('body', Vd, Td, mat, col=np.ones(len(Vd)), bone=bone)
    return W


def build(K):
    K.mat('body', rough=0.85, double=True, detail_strength=0.6)
    K.mat('belly', rough=0.8, double=True, detail_strength=0.4)
    K.mat('accent', double=True)
    K.mat('dark', double=True, rough=0.6)
    K.mat('horn', double=True)
    K.mat('eye', rough=0.15, emit_strength=1.5)

    B = K.sdf()
    # ---- thorax, collar ruff, shoulder tufts, head
    B.ell(THX, (1.15, 1.2, 1.3), k=0.0)
    B.ell(THX + (0, 0.55, 0.2), (0.95, 0.75, 1.0), k=0.4)                  # scutum hump
    B.ell((0, 6.4, 1.15), (1.1, 1.0, 0.75), rot=(-25, 0, 0), k=0.35)       # collar ruff
    B.ell((0.85, 6.45, 0.45), (0.55, 0.55, 0.78), rot=(0, 0, -20), k=0.3, mirror=True)  # tegulae over the wing roots
    B.ell((0.6, 5.75, -0.35), (0.6, 0.65, 0.7), k=0.4, mirror=True)        # hindwing root tufts
    B.ell(HEAD, (0.78, 0.74, 0.62), k=0.25)
    B.ell(HEAD + (0, 0.05, 0.42), (0.5, 0.55, 0.3), k=0.2)                 # frons
    B.tube([(0.22, 6.2, 2.3), (0.3, 6.05, 2.75), (0.25, 6.2, 3.05)], [0.17, 0.15, 0.1], k=0.12, mirror=True)   # labial palps
    # ---- abdomen: tapered, segmented
    B.tube([AB0, AB0 + AB_DIR * AB_LEN * 0.3 + [0, -0.05, 0], AB0 + AB_DIR * AB_LEN * 0.68, AB1], [0.95, 1.08, 0.88, 0.32], k=0.45)
    B.ell(AB1 - AB_DIR * 0.1, (0.3, 0.3, 0.35), k=0.2)

    def along(x, y, z):
        return (y - AB0[1]) * AB_DIR[1] + (z - AB0[2]) * AB_DIR[2]

    def fur(x, y, z):
        f1, _, _ = voronoi(x, y, z, size=0.2, seed=21)
        tuft = -0.07 * (1 - smoothstep(0.0, 0.55, f1))
        fluff = -0.05 * fbm(x, y, z, 3, freq=2.4, seed=4)
        al = along(x, y, z)
        abd = smoothstep(0.2, 0.9, al)                        # abdomen: segment grooves, shorter fur
        fr = (al / SEG) % 1.0
        groove = 0.1 * (1 - smoothstep(0.0, 0.1, fr)) + 0.1 * smoothstep(0.92, 1.0, fr) - 0.03 * smoothstep(0.2, 0.6, fr)
        return (tuft + fluff) * (1 - 0.55 * abd) + groove * abd * (al < AB_LEN - 0.2)
    B.disp(fur, band=0.18)

    def belly(P):
        P = np.asarray(P, float)
        al = along(P[:, 0], P[:, 1], P[:, 2])
        fr = (al / SEG) % 1.0
        ring = np.where((al > 0.6) & (al < AB_LEN - 0.35), np.abs(fr - 0.72) - 0.14, 1.0)
        collar = S.Ell((0, 6.4, 1.25), (1.2, 1.05, 0.62), rot=(-25, 0, 0)).dist(S.Points(P))
        under = S.Ell(THX + (0, -0.75, 0.4), (0.85, 0.6, 0.9)).dist(S.Points(P))
        return np.minimum(np.minimum(ring, collar), under)

    def tint(P, N):
        return 1.0 + 0.12 * fbm(P[:, 0], P[:, 1], P[:, 2], 3, freq=1.6, seed=9)

    K.skin(B, voxel=0.045, tris=13000, belly=belly, tint=tint)

    # ---- skeleton
    K.bone('spine', AB0 + (0, 0.2, 0.2), THX + (0, 0.1, 1.1), None, 1.3)
    K.bone('head', (0, 6.45, 1.35), HEAD + (0, 0.1, 0.65), 'spine', 0.8)
    K.bone('jaw', (0, 6.1, 2.25), (0, 5.75, 2.6), 'head', 0.25)
    K.chain('tail', [AB0, AB0 + AB_DIR * AB_LEN * 0.35, AB0 + AB_DIR * AB_LEN * 0.7, AB1], 'spine', (1.0, 0.45))

    # ---- wings
    wings = {}
    for key, spec in (('F', FORE), ('H', HIND)):
        for s, sd in ((-1, 'L'), (1, 'R')):
            W = WingMap(spec, s)
            tip_th = spec['ctrl'][3][0] if key == 'F' else 40
            K.bone(f'wing{key}{sd}', W.root, W.tr(tip_th, 0.95), 'spine', 0.35)
            wings[key + sd] = build_wing(K, spec, s, f'wing{key}{sd}')

    # ---- head: compound eyes, antennae, proboscis
    for s in (-1, 1):
        c = HEAD + (s * 0.5, 0.1, 0.22)
        V, T, _ = geom.sphere(c, (0.46, 0.56, 0.5), 12, 9, rot=(0, s * 25, 0))
        sh = 0.55 + 0.45 * np.clip((V - c) @ unit([s * 0.6, 0.5, 0.6]) / 0.5, 0, 1)
        K.piece('body', V, T, 'eye', smooth=False, col=sh, bone='head')
    ant = {}
    for s, sd in ((-1, 'L'), (1, 'R')):
        pts = np.array([(s * 0.25, 7.15, 2.05), (s * 0.75, 8.15, 2.5), (s * 1.35, 9.1, 2.4), (s * 1.95, 9.7, 1.8)])
        C = S.catmull(pts, 5)
        rad = np.linspace(0.085, 0.03, len(C))
        K.horn('body', C[:11], list(rad[:11]), mat='dark', sides=5, per=1, bone='ant' + sd + '1')
        K.horn('body', C[10:], list(rad[10:]), mat='dark', sides=5, per=1, bone='ant' + sd + '2')
        ant[sd] = C
        K.chain('ant' + sd, pts[[0, 2, 3]], 'head', (0.15, 0.08))
        T_ = unit(np.gradient(C, axis=0))
        for i in range(2, len(C) - 1):
            f = i / (len(C) - 1)
            ln = 0.55 * math.sin(math.pi * min(1, f * 1.15)) + 0.08
            out = unit(np.cross(T_[i], [0, 0, 1]) * s)
            fwd = unit(np.cross(T_[i], out))
            for side in (-1, 1):
                d = unit(out * 0.25 + fwd * side * 1.0 + T_[i] * 0.55)
                K.horn('body', [C[i], C[i] + d * ln * 0.55 + T_[i] * 0.03, C[i] + d * ln], [0.022, 0.015, 0.0], mat='body', sides=3, per=1,
                       bone='ant' + sd + ('1' if i < 10 else '2'), shade=(0.5, 0.9))
    coil = [(0, 6.05, 2.35), (0, 5.7, 2.6), (0, 5.4, 2.5), (0, 5.33, 2.25), (0, 5.5, 2.1), (0, 5.65, 2.25), (0, 5.57, 2.37)]
    K.horn('body', coil, [0.07, 0.065, 0.06, 0.055, 0.05, 0.04, 0.0], mat='dark', sides=5, per=3, bone='jaw')

    # ---- six dangling legs (rigid segments + fuzzy tibia tuft)
    hips = [(0.45, 5.3, 1.0), (0.55, 5.0, 0.4), (0.5, 4.95, -0.25)]
    legs = {}
    for li, hip in enumerate(hips):
        for s, sd in ((-1, 'L'), (1, 'R')):
            m = np.array([s, 1, 1])
            fx = [1.05, 1.35, 1.25][li]
            p0 = np.array(hip) * m
            p1 = p0 + np.array([s * (fx - 0.45), -0.75, 0.55 - li * 0.15])
            p2 = p1 + np.array([s * 0.25, -1.05, 0.3 - li * 0.25])
            p3 = p2 + np.array([s * -0.05, -0.75, 0.25 - li * 0.15])
            nm = f'leg{li}{sd}'
            K.chain(nm, [p0, p1, p2, p3], 'spine', (0.18, 0.1))
            K.horn('body', [p0, (p0 + p1) / 2 + [0, 0.05, 0], p1], [0.17, 0.15, 0.12], mat='body', sides=6, per=2, bone=nm + '1', shade=(0.7, 0.85))
            K.horn('body', [p1, (p1 + p2) / 2, p2], [0.12, 0.1, 0.08], mat='body', sides=6, per=2, bone=nm + '2', shade=(0.75, 0.9))
            K.horn('body', [p2, (p2 + p3) / 2, p3], [0.075, 0.06, 0.045], mat='dark', sides=5, per=2, bone=nm + '3')
            legs[(li, sd)] = (p0, p1, p2, p3)

    # ---- mounts
    K.mount('head', HEAD + (0, 0.3, 0), 'head')
    K.mount('jaw', (0, 6.05, 2.35), 'jaw')
    K.mount('mouth', (0, 5.95, 2.7), 'jaw')
    K.mount('chest', (0, 5.4, 1.3), 'spine')
    K.mount('back', THX + (0, 1.25, -0.2), 'spine')
    K.mount('tailTip', AB1 + AB_DIR * 0.3, 'tail3')
    K.mount('crown', HEAD + (0, 0.75, 0.1), 'head')
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.mount('hand' + sd, legs[(0, sd)][3], f'leg0{sd}3')
        K.mount('shoulder' + sd, (s * 1.0, 6.6, 0.55), 'spine')

    # ---- Might: tarsal claws -> tibial spurs -> abdominal stinger + fangs
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        for (li, sd), (p0, p1, p2, p3) in legs.items():
            d = unit(p3 - p2)
            s = -1 if sd == 'L' else 1
            for sg in (-1, 1):
                K.claw(part, p3, d + [s * 0.2 * sg, 0, 0.35], 0.22 * sc, 0.04 * (0.8 + 0.2 * sc), curl=(0, 0, -1), bone=f'leg{li}{sd}3')
            if t >= 1:
                for f in (0.35, 0.75):
                    q = p1 + (p2 - p1) * f
                    K.spike(part, q, unit(np.cross(d, [1, 0, 0])) * -1 + [0, 0, 0.3], 0.22 * sc, r=0.04, bone=f'leg{li}{sd}2')
        if t >= 2:
            K.horn(part, [AB1 - AB_DIR * 0.1, AB1 + AB_DIR * 0.45 * sc, AB1 + AB_DIR * 0.85 * sc + [0, -0.1, 0]], [0.16, 0.1, 0.0], bone='tail3')
        if t >= 1:
            for s in (-1, 1):
                K.horn(part, [(s * 0.2, 6.1, 2.35), (s * 0.3, 5.75, 2.6 + 0.1 * sc), (s * 0.15, 5.5 - 0.1 * sc, 2.7 + 0.15 * sc)],
                       [0.07, 0.05, 0], bone='head')

    # ---- Armor: chitin scutum + head capsule, tergite plates, spikes
    K.plate('arm_t1', THX + (0, 1.15, 0.0), (0.9, 0.55, 1.15), bone='spine', lames=2, lame_axis=(0, 0, 1), tris=300)
    K.plate('arm_t1', HEAD + (0, 0.6, -0.15), (0.6, 0.35, 0.5), bone='head', tris=160)
    dors = np.array([0, AB_DIR[2], -AB_DIR[1]]) * -1      # dorsal normal of the abdomen (up/back)
    if dors[1] < 0:
        dors = -dors
    rx = -math.degrees(math.atan2(-AB_DIR[1], -AB_DIR[2]))
    pc, _ = K.snap(AB0 + AB_DIR * AB_LEN * 0.5 + dors * 1.2)
    K.plate('arm_t2', pc, None, region=S.Ell(pc, (1.0, 0.7, 1.75), rot=(rx, 0, 0)), snap=False, bone='tail2', lames=5,
            lame_axis=-AB_DIR, tris=420)
    for j in range(4):
        p, n = K.snap(AB0 + AB_DIR * (SEG * (j + 1.6)) + dors * 1.2)
        K.spike('arm_t3', p + n * 0.1, n - AB_DIR * 0.5, 0.55 - j * 0.07, bone='tail' + str(min(3, j // 2 + 1)))
    for s in (-1, 1):
        p, n = K.snap(THX + (s * 0.85, 1.0, 0.3))
        K.spike('arm_t3', p, n + [s * 0.3, 0.3, -0.2], 0.6, bone='spine')
        p, n = K.snap(THX + (s * 0.6, 1.2, -0.5))
        K.spike('arm_t3', p, n + [0, 0.3, -0.4], 0.5, bone='spine')
        p, n = K.snap(HEAD + (s * 0.35, 0.7, -0.2))
        K.spike('arm_t3', p, n + [s * 0.2, 0.5, -0.5], 0.45, bone='head')

    # ---- Energy: glowing wing-margin spots, lumen veins, abdominal lumen glands + toxic dust crystals
    for key, W in wings.items():
        spec = FORE if key[0] == 'F' else HIND
        bn = 'wing' + key
        ths = np.linspace(W.th0 + 8, W.th1 - 14, 7)
        for th in ths:
            K.ball('enr_t1', W.tr(th, 0.93), (0.1, 0.1, 0.1), mat='accent', seg=6, rings=4, bone=bn)
        for th in spec['veins'][1::2]:
            rv = np.linspace(0.25, 0.85, 6)
            K.horn('enr_t2', W.tr(np.full_like(rv, th), np.clip(rv + wave(th, rv), 0, 1)), [0.0, 0.045, 0.05, 0.05, 0.045, 0.0],
                   mat='accent', sides=4, per=1, bone=bn)
    for j in range(4):
        c = AB0 + AB_DIR * (SEG * (j + 1.72))
        for s in (-1, 1):
            p, n = K.snap(c + np.array([s * 1.2, -0.2, 0]))
            K.ball('enr_t2', p, (0.13, 0.13, 0.13), mat='accent', seg=8, rings=5, bone='tail' + str(min(3, j // 2 + 1)))
    p, n = K.snap(AB1 + AB_DIR * 0.4)
    K.ball('enr_t3', AB1 + AB_DIR * 0.12, (0.36, 0.36, 0.4), mat='accent', seg=10, rings=7, bone='tail3')
    for s in (-1, 1):
        p, n = K.snap(THX + (s * 0.5, 1.3, -0.2))
        K.crystal('enr_t3', p - n * 0.05, n + [s * 0.4, 0, -0.2], 0.6, seed=3, bone='spine')
        p, n = K.snap((s * 0.75, 7.0, 1.2))
        K.crystal('enr_t3', p - n * 0.05, n + [s * 0.5, 0.2, 0.3], 0.45, seed=6, bone='spine')

    # ---- Speed: tail streamers on the hindwings, scaled forewing fringe, collar ruff frills
    for s, sd in ((-1, 'L'), (1, 'R')):
        W = wings['H' + sd]
        tip = W.tr(74, 1.0)
        K.fin('spd_t1', tip - W.es * 0.1, W.normal(np.array(5.0)), unit(tip - W.tr(74, 0.8)), 1.1, 0.5, bone='wingH' + sd)
        p, n = K.snap(AB1 + AB_DIR * -0.35 + np.array([s * 0.35, 0, 0]))     # anal tuft fan
        K.frill('spd_t2', p, unit(AB_DIR + [s * 0.9, 0, 0]), (0, 1, 0), 0.9, 0.8, rays=4, spread=80, bone='tail3')
        p, n = K.snap(AB0 + AB_DIR * AB_LEN * 0.45 + np.array([s * 1.2, 0, 0]))
        K.fin('spd_t2', p - n * 0.05, unit(n + [0, 0, 0] + AB_DIR * 0.4), AB_DIR * -1, 0.7, 0.9, bone='tail2')
        p, n = K.snap((s * 0.95, 6.65, 1.3))
        K.frill('spd_t3', p, n + [s * 0.4, 0.4, 0.2], (0, 1, 0.2), 0.9, 1.0, rays=5, spread=120, bone='head')
        K.frill('spd_t3', legs[(2, sd)][2], (s * 0.6, -0.2, -1), (0, -1, 0), 0.6, 0.55, rays=3, spread=60, bone=f'leg2{sd}2')

    # ---- Focus: ocelli, crest, third eye, glowing antenna tips
    for s in (-1, 1):
        p, n = K.snap(HEAD + (s * 0.28, 0.72, 0.15))
        K.eye('fcs_t1', p - n * 0.02, n + [0, 0, 0.4], 0.09, bone='head', lid=False)
    p, n = K.snap(HEAD + (0, 0.8, -0.3))
    K.leaf('fcs_t2', p - n * 0.04, n * [1, 1, 0.3] + [0, 0, -0.3], (0, 0.2, 1), 0.65, 1.1, th=0.05, lobes=4, mat='accent', bone='head')
    p, n = K.snap(HEAD + (0, 0.25, 0.75))
    K.eye('fcs_t3', p - n * 0.04, n, 0.15, bone='head')
    for sd in ('L', 'R'):
        K.ball('fcs_t3', ant[sd][-1], (0.14, 0.14, 0.14), mat='accent', seg=8, rings=5, bone='ant' + sd + '2')
