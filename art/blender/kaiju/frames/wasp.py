"""Vespa Rex: colossal hornet (flyer plan), hovering with the body pitched nose-up ~34 deg and the striped gaster
curled down and forward under the thorax so the stinger points at the enemy. Game coords, faces +Z.

Humped SDF thorax on a pin-thin petiole, teardrop gaster with clean body/belly bands cut along its curved
centreline (arc-length parametrised) and sawtooth tergites, a triangular head with faceted kidney-shaped compound
eyes (SDF pieces with the inner notch cut out), heavy toothed mandibles, elbowed antennae, six jointed dangling legs
(forelegs raised) and two pairs of long narrow translucent wings with dark venation and a pterostigma.
"""
import math

import bpy
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE, decimated, mesh_arrays
from ..materials import hex_rgb
from ..noise import fbm, voronoi, smoothstep

INFO = dict(name='Vespa Rex', colors=dict(body='#1e1813', belly='#f5b50d', accent='#a8ff2b'))
DETAIL = 'chitin'
UV_SCALE = 0.7
EYE_ACCENT = True
CAM = dict(az=32, el=11)


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ body frame (thorax pitched nose-up)
PB = 34.0
RB = S.rot_xyz((-PB, 0, 0))                         # local -> world (local z = body axis, local y = dorsal)
BD = RB[:, 2]                                       # forward / up along the body
BU = RB[:, 1]                                       # dorsal normal
THC = np.array([0.0, 6.65, 0.15])                   # thorax centre


def T(a, u=0.0, x=0.0):
    """Thorax-frame point: a along the body axis, u dorsal, x lateral."""
    return THC + BD * a + BU * u + np.array([x, 0, 0])


# ------------------------------------------------------------------ head frame (face looks forward, a bit down)
HS = 1.25
HC = np.array([0.0, 8.3, 2.05])
HR = S.rot_xyz((15, 0, 0))


def H(p):
    return HC + (np.asarray(p, float) * HS) @ HR.T


def HD(d):
    return np.asarray(d, float) @ HR.T


# ------------------------------------------------------------------ gaster (petiole -> stinger socket)
PET = [T(-1.5, -0.3), (0, 5.15, -1.42), (0, 4.88, -1.8)]
ABD = np.array([(0, 4.85, -1.85), (0, 4.25, -2.6), (0, 3.2, -3.15), (0, 2.05, -3.0), (0, 1.25, -2.25), (0, 0.95, -1.3), (0, 0.97, -0.88)])
ABD_R = np.array([0.32, 0.98, 1.28, 1.14, 0.8, 0.38, 0.13])
GC = S.catmull(ABD, 10)
GR = np.clip(S.catmull(ABD_R, 10)[:, 0], 0.05, None)
GS = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(GC, axis=0), axis=1))])
GT = unit(np.gradient(GC, axis=0))
GLEN = float(GS[-1])
NSEG = 6
SEG = GLEN / NSEG
STB = np.array([0.0, 0.97, -0.95])                  # stinger base (inside the tip)
STD = unit([0, 0.1, 1])                             # stinger direction (toward the enemy)


def gparam(P):
    """Nearest gaster centreline sample -> (arc length s, distance, sample index)."""
    P = np.asarray(P, float).reshape(-1, 3)
    s = np.empty(len(P)); d = np.empty(len(P)); ix = np.empty(len(P), int)
    for a in range(0, len(P), 150000):
        Q = P[a:a + 150000]
        d2 = ((Q[:, None, :] - GC[None]) ** 2).sum(-1)
        i = d2.argmin(1)
        rel = Q - GC[i]
        s[a:a + len(Q)] = np.clip(GS[i] + (rel * GT[i]).sum(1), 0, GLEN)
        d[a:a + len(Q)] = np.sqrt(d2[np.arange(len(Q)), i])
        ix[a:a + len(Q)] = i
    return s, d, ix


def gat(s):
    """Centreline point, tangent, radius at arc length s."""
    i = int(np.clip(np.searchsorted(GS, s), 1, len(GS) - 1))
    f = (s - GS[i - 1]) / max(GS[i] - GS[i - 1], 1e-9)
    return GC[i - 1] + (GC[i] - GC[i - 1]) * f, unit(GT[i - 1] + (GT[i] - GT[i - 1]) * f), GR[i - 1] + (GR[i] - GR[i - 1]) * f


def gframe(s):
    c, t, r = gat(s)
    x = np.array([1.0, 0, 0])
    y = unit(np.cross(t, x))          # points to the outer (dorsal / rear) side of the curl
    if y @ (c - np.array([0, 2.6, -1.6])) < 0:
        y = -y
    return c, t, r, x, y


# ------------------------------------------------------------------ legs: (hip, knee, ankle, toe) for +X side
LEGS = [
    [(0.45, 6.5, 1.28), (1.3, 5.85, 2.2), (1.5, 6.65, 3.25), (1.22, 6.45, 4.35)],    # raised forelegs
    [(0.55, 5.93, 0.69), (1.7, 5.45, 1.75), (1.85, 4.0, 2.6), (1.6, 2.8, 3.25)],      # mid legs reach forward
    [(0.5, 5.55, 0.05), (2.55, 5.2, -0.5), (3.45, 3.65, -0.95), (3.55, 2.25, -0.7)],  # hind legs splay wide
]

# ------------------------------------------------------------------ wings
FORE = dict(root=(0.8, 7.6, 0.1), es=(0.8, 0.42, -0.42), ec=(0, -0.12, -1), L=7.2, cmax=1.6, camber=0.06,
            veins=[([(0.02, 0.2), (0.3, 0.16), (0.5, 0.12), (0.58, 0.08)], 0.045),
                   ([(0.03, 0.42), (0.3, 0.4), (0.55, 0.35), (0.8, 0.28)], 0.04),
                   ([(0.05, 0.7), (0.3, 0.68), (0.5, 0.72), (0.64, 0.9)], 0.035),
                   ([(0.06, 0.9), (0.22, 0.97)], 0.03),
                   ([(0.3, 0.16), (0.3, 0.4)], 0.03), ([(0.3, 0.4), (0.3, 0.68)], 0.03),
                   ([(0.5, 0.12), (0.55, 0.35)], 0.028), ([(0.45, 0.39), (0.5, 0.72)], 0.028),
                   ([(0.66, 0.06), (0.7, 0.31)], 0.025)],
            stigma=(0.56, 0.66, 0.05))
HIND = dict(root=(0.75, 7.1, -0.6), es=(0.82, 0.24, -0.56), ec=(0, -0.2, -1), L=5.2, cmax=1.2, camber=0.05,
            veins=[([(0.02, 0.25), (0.3, 0.2), (0.55, 0.16)], 0.035),
                   ([(0.04, 0.55), (0.3, 0.5), (0.55, 0.48), (0.7, 0.6)], 0.03),
                   ([(0.3, 0.2), (0.33, 0.5)], 0.025)],
            stigma=None)


class Wing:
    def __init__(self, spec, s):
        m = np.array([s, 1, 1])
        self.root = np.array(spec['root'], float) * m
        es = unit(np.array(spec['es'], float) * m)
        ec = np.array(spec['ec'], float)
        ec = unit(ec - es * (ec @ es))
        n = np.cross(es, ec)
        if n[1] < 0:
            n = -n
        self.es, self.ec, self.n = es, ec, unit(n)
        self.L, self.cmax, self.camber = spec['L'], spec['cmax'], spec['camber']

    def P(self, u, v, off=0.0):
        u, v = np.asarray(u, float), np.asarray(v, float)
        wfull = self.cmax * (0.26 + 0.74 * smoothstep(0.0, 0.42, u))
        tip = np.sqrt(np.clip(1 - (np.maximum(u - 0.5, 0) / 0.5) ** 2, 0, 1))
        width = wfull * tip
        le = 0.42 * (wfull - width)
        b = le + v * width
        h = self.camber * np.sin(np.pi * np.clip(v, 0, 1)) * width + 0.035 * self.L * u * u + off
        return self.root + u[..., None] * self.L * self.es + b[..., None] * self.ec + h[..., None] * self.n

    def tip(self):
        return self.P(np.array(1.0), np.array(0.5))


def build_wing(K, spec, s, bone, part='body'):
    W = Wing(spec, s)
    nu, nv = 20, 6
    us = np.linspace(0, 1, nu + 1) ** 1.0
    vs = np.linspace(0, 1, nv + 1)
    U_, V_ = np.meshgrid(us, vs, indexing='ij')
    Pm = W.P(U_, V_).reshape(-1, 3)
    col = (0.62 + 0.38 * np.clip(U_ * 1.6, 0, 1)) * (1 - 0.22 * (V_ < 0.2)) * (1 - 0.12 * smoothstep(0.85, 1.0, U_))
    Tq = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            Tq += [(a, a + nv + 1, a + nv + 2), (a, a + nv + 2, a + 1)]
    K.piece(part, Pm, np.array(Tq), 'membrane', col=col.reshape(-1), bone=bone)
    # costa: thick leading-edge rib, then the venation
    uc = np.linspace(0.0, 0.86, 8)
    K.horn(part, W.P(uc, np.zeros_like(uc) + 0.0), list(np.linspace(0.085, 0.03, 8)), mat='dark', sides=5, per=2, bone=bone, shade=(0.6, 0.9))
    for pts, r in spec['veins']:
        uv = np.array(pts, float)
        K.horn(part, W.P(uv[:, 0], uv[:, 1]), list(np.linspace(r, r * 0.6, len(uv))), mat='dark', sides=4, per=2, bone=bone, shade=(0.6, 0.9))
    if spec['stigma']:
        u0, u1, v0 = spec['stigma']
        uu = np.linspace(u0, u1, 5)
        for side in (1, -1):
            K.horn(part, W.P(uu, np.full(5, v0), side * 0.02), [0.02, 0.07, 0.075, 0.06, 0.0], mat='dark', sides=5, per=2, bone=bone, flat=0.45)
    return W


# ------------------------------------------------------------------ helpers
def mesh_sdf(sdf, voxel, tris):
    V, Q = sdf.mesh(voxel, margin=0.1)
    me = decimated(V, Q, tris, 'tmp_piece')
    V, Tt = mesh_arrays(me)
    bpy.data.meshes.remove(me)
    return V, Tt


def kidney_eye(K, part, s, bone, scale=1.0):
    """Faceted, kidney-shaped compound eye (notch on the inner front margin where the antenna sits)."""
    Rl = HR @ S.rot_xyz((0, s * 24, s * 6))
    c = H((s * 0.8, 0.02, 0.08))
    r = np.array([0.3, 0.64, 0.4]) * HS * scale
    E = S.SDF()
    E.add(S.Ell(c, r, Rl), 0.0)
    E.cut(S.Ell(H((s * 0.6, 0.2, 0.42)), np.array([0.2, 0.19, 0.32]) * HS, HR), 0.06)
    E.cut(S.Ell(H((s * 0.2, 0.0, 0.0)), np.array([0.5, 1.0, 0.6]) * HS, HR), 0.05)     # buried inner half

    def fac(x, y, z):
        f1, f2, _ = voronoi(x, y, z, size=0.075, seed=11 + s)
        return 0.013 * (1 - smoothstep(0.0, 0.12, f2 - f1))
    E.disp(fac, band=0.05)
    V, Tt = mesh_sdf(E, 0.02, 1100)
    _, f2, cid = voronoi(V[:, 0], V[:, 1], V[:, 2], size=0.075, seed=11 + s)
    f1, _, _ = voronoi(V[:, 0], V[:, 1], V[:, 2], size=0.075, seed=11 + s)
    edge = smoothstep(0.0, 0.1, f2 - f1)
    lit = 0.65 + 0.35 * np.clip(unit(V - c) @ unit(HD((s * 0.3, 0.6, 0.7))), 0, 1)
    col = (0.15 + 0.85 * edge) * (0.6 + 0.4 * cid) * lit
    K.piece(part, V, Tt, 'eye', col=col, bone=bone)


def loop_tube(K, part, pts, nrm, rr, mat, bone, sides=4):
    """Closed tube along a loop of points (nrm = outward surface normals)."""
    M = len(pts)
    V, Tt = [], []
    for i in range(M):
        t = unit(pts[(i + 1) % M] - pts[i - 1])
        n = unit(nrm[i] - t * (nrm[i] @ t))
        b = np.cross(t, n)
        for k in range(sides):
            a = 2 * np.pi * k / sides
            V.append(pts[i] + rr * (np.cos(a) * n + np.sin(a) * b))
    V = np.array(V)
    for i in range(M):
        i2 = (i + 1) % M
        for k in range(sides):
            k2 = (k + 1) % sides
            a, b, c, d = i * sides + k, i2 * sides + k, i2 * sides + k2, i * sides + k2
            for tri in ((a, b, c), (a, c, d)):
                fn = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
                cen = (pts[i] + pts[i2]) * 0.5
                Tt.append(tri if fn @ (V[list(tri)].mean(0) - cen) >= 0 else tri[::-1])
    K.piece(part, V, np.array(Tt), mat, bone=bone)


def gaster_ring(K, part, s, off=0.03, rr=0.055, mat='accent', bone='tail3', n=28):
    c, t, r, x, y = gframe(s)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = c[None] + (np.cos(ang)[:, None] * x + np.sin(ang)[:, None] * y) * r
    p, nn = K.snap(P)
    loop_tube(K, part, p + nn * off, nn, rr, mat, bone)


def tail_bone(s):
    return 'tail' + str(1 + int(np.clip(s / GLEN * 4.2, 0, 3)))


# ------------------------------------------------------------------ build
def build(K):
    body_c, belly_c, acc = (hex_rgb(K.colors[k]) for k in ('body', 'belly', 'accent'))
    K.mat('body', rough=0.28, detail_strength=0.5)
    K.mat('belly', rough=0.32, detail_strength=0.35)
    K.mat('plate', color=body_c * 0.6 + belly_c * 0.25, rough=0.25)
    K.mat('eye', color=acc * 0.42, emit=acc, emit_strength=0.06, rough=0.22, vcol=True)
    K.mat('dark', color=hex_rgb('#120e0b'), rough=0.3, double=True)
    K.mat('horn', color=belly_c * 0.35 + np.array([0.75, 0.62, 0.42]) * 0.5, rough=0.25)
    K.mat('membrane', color='#b48a4c', rough=0.12, vcol=True, double=True, alpha=0.42, detail=False)

    B = K.sdf()
    # ---------------------------------------------------------------- mesosoma: humped, pitched thorax
    B.ell(T(0, 0), (1.12, 1.02, 1.5), rot=RB, k=0.0)
    B.ell(T(1.15, -0.05), (0.98, 0.86, 0.5), rot=RB, k=0.22)                 # pronotum collar
    B.ell(T(0.25, 0.5), (0.98, 0.62, 0.98), rot=RB, k=0.3)                   # scutum hump
    B.ell(T(-0.72, 0.68), (0.56, 0.32, 0.4), rot=RB, k=0.14)                 # scutellum knob
    B.ell(T(-1.15, 0.0), (0.78, 0.78, 0.62), rot=RB, k=0.3)                  # propodeum
    for x, a, u, rr in ((0.86, 0.5, 0.78, (0.3, 0.2, 0.38)), (0.8, -0.35, 0.72, (0.24, 0.17, 0.3))):
        B.ell(T(a, u, x), rr, rot=RB, k=0.12, mirror=True)                   # tegulae / wing sockets
    for (hip, _, _, _), rr in zip(LEGS, (0.36, 0.4, 0.4)):
        B.ell(hip, (rr, rr * 1.1, rr), k=0.18, mirror=True)                  # coxae
    B.ell(T(0.0, 0.0, 1.0), (0.3, 0.55, 0.85), rot=RB, k=0.25, mirror=True)  # mesopleuron bulge
    # ---------------------------------------------------------------- neck + petiole (stay thin)
    B.tube([T(1.45, 0.1), (0, 7.85, 1.62)], [0.36, 0.32], k=0.1)
    B.tube(PET, [0.3, 0.19, 0.26], k=0.06)
    # ---------------------------------------------------------------- gaster: teardrop, curled under
    B.tube(ABD, ABD_R, k=0.08, s=(0.96, 1, 1), per=8)
    # ---------------------------------------------------------------- head (triangular: wide eyes, narrow jaws)
    B.ell(H((0, 0.1, -0.05)), np.array((0.95, 0.72, 0.5)) * HS, rot=HR, k=0.2)
    B.tube([H((0, 0.05, 0.12)), H((0, -0.45, 0.24)), H((0, -0.86, 0.3))], np.array([0.62, 0.44, 0.27]) * HS, k=0.2, s=(1.25, 1, 0.78))
    B.ell(H((0, -0.5, 0.48)), np.array((0.4, 0.38, 0.13)) * HS, rot=HR, k=0.08)          # clypeus shield
    B.ell(H((0, 0.15, 0.33)), np.array((0.28, 0.34, 0.14)) * HS, rot=HR, k=0.12)         # frons
    B.ell(H((0, 0.45, -0.22)), np.array((0.74, 0.34, 0.4)) * HS, rot=HR, k=0.2)          # vertex / occiput
    B.ell(H((0.6, -0.22, -0.12)), np.array((0.34, 0.45, 0.34)) * HS, rot=HR, k=0.18, mirror=True)  # genae
    B.ell(H((0.17, 0.12, 0.45)), np.array((0.1, 0.1, 0.08)) * HS, rot=HR, k=0.05, mirror=True)    # antennal sockets

    # ---------------------------------------------------------------- sculpt: tergite sawtooth, sutures, micro
    sut = [(0.82, 0.05), (-0.5, 0.045), (-0.98, 0.04)]

    def chit(x, y, z):
        P = np.stack([x, y, z], 1).astype(float)
        base = -0.014 * fbm(x, y, z, 3, freq=1.5, seed=4)
        s, d, ix = gparam(P)
        wg = (d < GR[ix] + 0.35) & (s > 0.15) & (s < GLEN - 0.15)
        f = (s / SEG) % 1.0
        terg = (-0.1 * f ** 1.6 + 0.075 * (1 - smoothstep(0.0, 0.07, f)) + 0.03) * wg
        a = (P - THC) @ BD
        th = np.linalg.norm(P - THC, axis=1) < 2.0
        g = 0
        for a0, w in sut:
            g = g + 0.035 * (1 - smoothstep(0.0, w, np.abs(a - a0)))
        return base + terg + g * th
    B.disp(chit, band=0.14)

    # ---------------------------------------------------------------- material regions
    clyp = S.Ell(H((0, -0.52, 0.55)), np.array((0.4, 0.38, 0.3)) * HS, rot=HR)
    scutel = S.Ell(T(-0.72, 0.7), (0.5, 0.4, 0.42), rot=RB)
    side = [S.Ell(T(0.15, 0.05, s * 1.05), (0.3, 0.25, 0.55), rot=RB) for s in (-1, 1)]
    side += [S.Ell(T(-1.3, 0.5, s * 0.36), (0.16, 0.2, 0.3), rot=RB) for s in (-1, 1)]     # propodeal spots (read from behind)
    gena = [S.Ell(H((s * 0.62, -0.3, -0.02)), np.array((0.3, 0.3, 0.3)) * HS, rot=HR) for s in (-1, 1)]
    gena += [S.Ell(H((s * 0.6, 0.2, 0.5)), np.array((0.09, 0.26, 0.3)) * HS, rot=HR @ S.rot_xyz((0, 0, -s * 15))) for s in (-1, 1)]   # eye-notch marks
    anchor = S.Ell(H((0, -0.36, 0.6)), np.array((0.07, 0.22, 0.3)) * HS, rot=HR)

    def belly(P):
        P = np.asarray(P, float)
        s, d, ix = gparam(P)
        f = (s / SEG) % 1.0
        k = np.floor(s / SEG)
        # chevron: the band front edge dips forward at the dorsal midline
        dors = np.exp(-(P[:, 0] / 0.5) ** 2)
        band = (np.abs(f - 0.71 + 0.07 * dors) - 0.215 - 0.035 * dors) * SEG
        band = np.where((d < GR[ix] + 0.5) & (k < NSEG - 1) & (s > 0.3), band, 1.0)
        a = (P - THC) @ BD
        collar = np.where(np.linalg.norm(P - T(1.1), axis=1) < 1.4, np.abs(a - 1.2) - 0.13, 1.0)
        out = np.minimum(band, collar)
        out = np.minimum(out, np.maximum(clyp.dist(S.Points(P)), -anchor.dist(S.Points(P))))
        for prim in [scutel] + side + gena:
            out = np.minimum(out, prim.dist(S.Points(P)))
        return out

    def tint(P, N):
        return (1 - 0.2 * smoothstep(0.2, -0.7, N[:, 1])) * (1 + 0.06 * fbm(P[:, 0], P[:, 1], P[:, 2], 2, freq=1.1, seed=8))

    K.skin(B, voxel=0.045, tris=12500, belly=belly, tint=tint)

    # ---------------------------------------------------------------- skeleton
    K.bone('spine', T(-1.3, 0), T(1.35, 0), None, 1.3)
    K.bone('head', T(1.45, 0.1), H((0, -0.2, 0.45)), 'spine', 0.95)
    K.bone('jaw', H((0, -0.72, 0.3)), H((0, -1.25, 0.62)), 'head', 0.25)
    tp = [np.array(PET[0]), np.array((0, 4.55, -2.25)), np.array((0, 3.2, -3.15)), np.array((0, 1.6, -2.75)), np.array((0, 0.97, -0.95))]
    for i, r in enumerate((0.42, 1.15, 1.1, 0.7)):
        K.bone(f'tail{i + 1}', tp[i], tp[i + 1], 'spine' if i == 0 else f'tail{i}', r)

    # ---------------------------------------------------------------- wings
    wings = {}
    for key, spec in (('F', FORE), ('H', HIND)):
        for s, sd in ((-1, 'L'), (1, 'R')):
            W = Wing(spec, s)
            K.bone(f'wing{key}{sd}', W.root, W.P(np.array(0.92), np.array(0.3)), 'spine', 0.3)
            wings[key + sd] = build_wing(K, spec, s, f'wing{key}{sd}')

    # ---------------------------------------------------------------- eyes, antennae
    for s in (-1, 1):
        kidney_eye(K, 'body', s, 'head')
    ant = {}
    for s, sd in ((-1, 'L'), (1, 'R')):
        b0 = H((s * 0.17, 0.12, 0.5))
        el = b0 + unit([s * 0.4, 0.82, 0.42]) * 1.15
        f1 = el + np.array([s * 0.55, 0.38, 0.85])
        f2 = el + np.array([s * 1.15, 0.28, 1.65])
        f3 = el + np.array([s * 1.65, -0.2, 2.15])
        K.chain('ant' + sd, [b0, el, f1, f3], 'head', (0.14, 0.08))
        K.horn('body', [b0, (b0 + el) / 2 + [0, 0.03, 0], el], [0.1, 0.09, 0.085], mat='belly', sides=6, per=2, bone='ant' + sd + '1', shade=(0.8, 1))
        K.ball('body', el, (0.11, 0.11, 0.11), mat='dark', seg=8, rings=5, bone='ant' + sd + '1')
        C = S.catmull(np.array([el, f1, f2, f3]), 6)
        nC = len(C)
        for i in range(nC - 1):
            bn = 'ant' + sd + ('2' if i < 6 else '3')
            r0 = 0.085 - 0.04 * i / nC
            K.horn('body', [C[i], C[i + 1]], [r0, r0 * 0.9], mat='dark', sides=5, per=1, bone=bn, shade=(0.7, 0.95))
        ant[sd] = C

    # ---------------------------------------------------------------- legs (rigid segmented pieces)
    legs = {}
    for li, pts in enumerate(LEGS):
        for s, sd in ((-1, 'L'), (1, 'R')):
            m = np.array([s, 1, 1])
            p0, p1, p2, p3 = (np.array(p) * m for p in pts)
            nm = f'leg{li}{sd}'
            K.chain(nm, [p0, p1, p2, p3], 'spine', (0.22, 0.12))
            bow = unit(np.cross(p1 - p0, [s, 0, 0])) * 0.08
            K.horn('body', [p0, (p0 + p1) / 2 + bow, p1], [0.24, 0.25, 0.16], mat='body', sides=7, per=2, bone=nm + '1', shade=(0.7, 0.95))
            K.ball('body', p1, (0.17, 0.17, 0.17), mat='dark', seg=8, rings=5, bone=nm + '2')
            K.horn('body', [p1, (p1 + p2) / 2, p2], [0.15, 0.15, 0.13], mat='belly', sides=7, per=2, bone=nm + '2', shade=(0.75, 1))
            K.ball('body', p2, (0.12, 0.12, 0.12), mat='dark', seg=8, rings=5, bone=nm + '3')
            for k in range(4):                                          # segmented tarsus
                a, b = p2 + (p3 - p2) * (k / 4), p2 + (p3 - p2) * ((k + 0.92) / 4)
                r = 0.09 - 0.012 * k
                K.horn('body', [a, (a + b) / 2, b], [r * 0.75, r, r * 0.7], mat='belly', sides=6, per=1, bone=nm + '3', shade=(0.7, 1))
            legs[(li, sd)] = (p0, p1, p2, p3)

    # ---------------------------------------------------------------- mounts
    K.mount('head', HC, 'head')
    K.mount('jaw', H((0, -0.95, 0.45)), 'jaw')
    K.mount('mouth', H((0, -1.0, 0.65)), 'jaw')
    K.mount('chest', T(0.9, -1.05), 'spine')
    K.mount('back', T(0.0, 1.25), 'spine')
    K.mount('tailTip', STB + STD * 1.25, 'tail4')
    K.mount('crown', H((0, 0.9, 0.0)), 'head')
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.mount('hand' + sd, legs[(0, sd)][3], f'leg0{sd}3')
        K.mount('shoulder' + sd, T(0.5, 0.85, s * 1.0), 'spine')

    # ---------------------------------------------------------------- Might: stinger + mandibles + claws/spurs
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        k = 0.78 + 0.22 * sc
        # stinger (sheath collar + blade)
        ln = 1.6 * sc
        r = 0.14 * (0.85 + 0.15 * sc)
        tip = STB + STD * ln + np.array([0, 0.05 * ln, 0])
        K.horn(part, [STB - STD * 0.15, STB + STD * ln * 0.45, tip], [r, r * 0.55, 0.0], mat='dark', sides=8, per=3, bone='tail4', shade=(0.55, 1))
        K.horn(part, [STB - STD * 0.1, STB + STD * 0.25], [r * 1.5, r * 1.15], mat='belly', sides=8, per=1, bone='tail4')
        if t >= 2:                                                   # barbs
            nb = 3 if t == 2 else 5
            for i in range(nb):
                f = 0.42 + 0.45 * i / nb
                q = STB + STD * ln * f
                rq = r * (1 - 0.6 * f)
                for sg in (-1, 1):
                    d = unit(-STD * 0.9 + np.array([sg * 0.6, 0.25, 0]))
                    K.horn(part, [q, q + d * (0.22 + 0.05 * t) * sc * 0.6], [rq * 0.7, 0.0], mat='dark', sides=4, per=1, bone='tail4')
        # mandibles
        for s in (-1, 1):
            m0, m1 = H((s * 0.42, -0.7, 0.1)), H((s * (0.42 + 0.02 * k), -0.7 - 0.3 * k, 0.1 + 0.22 * k))
            m2, m3 = H((s * (0.42 - 0.12 * k), -0.7 - 0.52 * k, 0.1 + 0.4 * k)), H((s * (0.42 - 0.3 * k), -0.7 - 0.6 * k, 0.1 + 0.46 * k))
            K.horn(part, [m0, m1, m2, m3], np.array([0.17, 0.15, 0.1, 0.0]) * HS * k, mat='horn', sides=7, per=3, bone='jaw', flat=0.45, shade=(0.6, 1),
                   up=HD((0, 0, 1)))
            nt = 3 + t
            for i in range(nt):
                f = (i + 0.5) / nt
                p = S.catmull(np.array([m1, m2, m3]), 6)[int(f * 12)]
                inn = unit(np.array([-s, -0.25, 0.1]))
                K.horn(part, [p - inn * 0.04, p + inn * (0.12 + 0.03 * t) * k], [0.05 * k, 0.0], mat='dark', sides=4, per=1, bone='jaw')
            K.horn(part, [m2 * 0.6 + m3 * 0.4, m3 + HD((0, -0.05, 0.05)) * HS * 0.2 * k], [0.08 * k, 0.0], mat='dark', sides=5, per=1, bone='jaw')
            if t >= 3:                                               # hooked outer prong
                K.horn(part, [m1, m1 + HD((s * 0.25, -0.15, 0.25)) * HS, m1 + HD((s * 0.15, -0.35, 0.45)) * HS], [0.07, 0.05, 0.0], mat='dark', sides=5, bone='jaw')
        # tarsal claws (paired) + tibial spurs
        for (li, sd), (p0, p1, p2, p3) in legs.items():
            s = -1 if sd == 'L' else 1
            d = unit(p3 - p2)
            for sg in (-1, 1):
                K.claw(part, p3, unit(d + np.array([s * 0.35 * sg, 0, 0.2])), 0.26 * sc, 0.05 * (0.8 + 0.2 * sc), curl=(0, 0, -1) if li else (0, -1, 0), mat='dark', bone=f'leg{li}{sd}3')
            if t >= 1:
                for sg in (-1, 1):
                    dd = unit(-unit(p2 - p1) * 0.3 + d * 0.8 + np.array([s * 0.3 * sg, 0, 0]))
                    K.horn(part, [p2, p2 + dd * 0.3 * sc], [0.05 * k, 0.0], mat='dark', sides=4, per=1, bone=f'leg{li}{sd}3')
            if t >= 2:
                for f in (0.3, 0.55, 0.8):
                    q = p1 + (p2 - p1) * f
                    out = unit(np.cross(p2 - p1, [0, 0, 1]) * s + np.array([s * 0.4, 0, 0]))
                    K.spike(part, q, unit(out - unit(p2 - p1) * 0.6), 0.16 * sc, r=0.04, mat='dark', bone=f'leg{li}{sd}2')
            if t >= 3 and li == 0:                                   # raptorial foreleg hooks
                for f in (0.35, 0.7):
                    q = p0 + (p1 - p0) * f
                    K.horn(part, [q, q + np.array([s * 0.15, -0.35, 0.3]), q + np.array([s * 0.1, -0.55, 0.6])], [0.07, 0.05, 0.0], mat='dark', sides=5, bone=f'leg0{sd}1')

    # ---------------------------------------------------------------- Armor
    K.plate('arm_t1', T(0.2, 1.0), (1.05, 0.6, 1.25), rot=RB, bone='spine', lames=3, lame_axis=-BD, tris=360)
    K.plate('arm_t1', H((0, 0.55, -0.05)), np.array((0.7, 0.4, 0.55)) * HS, rot=HR, bone='head', tris=200)
    # gaster armour: narrow saddle bands that sit ONLY over the black stripes (yellow bands stay bare)
    BLK = [SEG * (k + 0.21) for k in (1, 2, 3, 4)]
    for sj in BLK:
        c, tg, r, x, y = gframe(sj)
        Rg = np.column_stack([x, y, tg])
        reg = S.Ell(c + y * r * 0.35, (r * 1.08, r * 0.95, SEG * 0.2), Rg)
        p, _ = K.snap(c + y * r)
        K.plate('arm_t2', p, None, region=reg, snap=False, bone=tail_bone(sj), tris=150, th=0.07)
    K.plate('arm_t2', T(1.15, 0.2), (1.05, 0.55, 0.45), rot=RB, bone='spine', tris=220)
    for j, sj in enumerate(BLK):
        c, tg, r, x, y = gframe(sj)
        p, n = K.snap(c + y * r)
        K.spike('arm_t3', p + n * 0.1, unit(n - tg * 0.7), 0.5 - 0.05 * j, bone=tail_bone(sj))
    for s in (-1, 1):
        p, n = K.snap(T(0.3, 1.0, s * 0.55))
        K.spike('arm_t3', p, unit(n + BU * 0.3 - BD * 0.6 + [s * 0.3, 0, 0]), 0.7, bone='spine')
        p, n = K.snap(T(1.1, 0.6, s * 0.75))
        K.spike('arm_t3', p, unit(n + [s * 0.4, 0.2, 0]), 0.5, bone='spine')
        p, n = K.snap(H((s * 0.4, 0.7, -0.15)))
        K.horn('arm_t3', [p - n * 0.05, p + unit(n + [s * 0.3, 0.4, -0.6]) * 0.45, p + unit(n + [s * 0.4, 0.2, -1.2]) * 0.85],
               [0.11, 0.07, 0.0], mat='plate', bone='head')

    # ---------------------------------------------------------------- Energy: venom sacs, glowing seams, venom drip
    # venom sacs: plump teardrop bladders on the lower flanks, each with a dark duct collar
    for j, (sj, sz) in enumerate(((SEG * 2.0, 1.0), (SEG * 3.3, 0.8))):
        c, tg, r, x, y = gframe(sj)
        for s in (-1, 1):
            p, n = K.snap(c + x * s * r - y * r * 0.4)
            n = unit(n)
            q0 = p - tg * 0.5 * sz + n * 0.04
            pts = [q0, q0 + tg * 0.18 * sz + n * 0.12 * sz, p + n * 0.2 * sz, p + tg * 0.3 * sz + n * 0.16 * sz, p + tg * 0.55 * sz + n * 0.04]
            K.horn('enr_t1', pts, list(np.array([0.06, 0.2, 0.27, 0.2, 0.0]) * sz), mat='accent', sides=10, per=2, bone=tail_bone(sj), shade=(0.75, 1.0))
            K.horn('enr_t1', [q0 - tg * 0.04, q0 + tg * 0.1 * sz + n * 0.05], [0.09 * sz, 0.08 * sz], mat='dark', sides=8, per=1, bone=tail_bone(sj))
    for j in range(1, NSEG):
        gaster_ring(K, 'enr_t2', SEG * j - 0.02, rr=0.04, n=24, bone=tail_bone(SEG * j))
    K.ball('enr_t3', STB + STD * 0.05, (0.3, 0.3, 0.36), mat='accent', seg=10, rings=7, bone='tail4')
    for i, (f, rd) in enumerate(((0.2, 0.06), (0.45, 0.075), (0.75, 0.1))):
        K.ball('enr_t3', STB + STD * 0.25 + np.array([0, -0.12 - f * 0.55, 0]), (rd * 0.8, rd, rd * 0.8), mat='accent', seg=8, rings=5, bone='tail4')
    for s in (-1, 1):
        q = H((s * 0.15, -1.3, 0.62))
        for i, rd in enumerate((0.05, 0.065, 0.085)):
            K.ball('enr_t3', q + np.array([0, -0.12 - i * 0.16, 0.02]), (rd * 0.8, rd, rd * 0.8), mat='accent', seg=8, rings=5, bone='jaw')

    # ---------------------------------------------------------------- Speed: wing-tip streamers, tail vanes, winglets
    for s, sd in ((-1, 'L'), (1, 'R')):
        W = wings['F' + sd]
        u = np.linspace(0, 1, 9)
        base = W.P(np.full(9, 0.88), np.array(np.linspace(0.05, 0.75, 9)))
        V = [W.P(np.array(0.86), np.array(0.08))]
        tipp = W.tip() + W.es * 1.7 + W.ec * 0.35 + W.n * 0.25
        lead = [W.P(np.array(0.86), np.array(0.08)) * (1 - f) + tipp * f + W.ec * -0.12 * math.sin(math.pi * f) for f in u]
        trail = [W.P(np.array(0.8), np.array(0.85)) * (1 - f) + tipp * f + W.ec * 0.3 * math.sin(math.pi * f) for f in u]
        Vv, Tv, uu = geom.strip(lead, trail)
        K.piece('spd_t1', Vv, Tv, 'membrane', col=0.6 + 0.4 * uu, bone='wingF' + sd)
        K.horn('spd_t1', lead, list(np.linspace(0.05, 0.0, 9)), mat='dark', sides=4, per=1, bone='wingF' + sd)
        # tail vanes near the gaster tip
        sv = GLEN - SEG * 1.1
        c, tg, r, x, y = gframe(sv)
        p, n = K.snap(c + x * s * r)
        K.frill('spd_t2', p - n * 0.05, unit(x * s + y * 0.3), -tg, 1.45, 1.2, rays=5, spread=90, bone='tail4')
        # third pair of small winglets behind the hind wings
        Wh = wings['H' + sd]
        sp = dict(root=T(-0.95, 0.55, 0.6), es=(0.75, 0.15, -0.65), ec=(0, -0.3, -1), L=2.6, cmax=0.75, camber=0.04,
                  veins=[([(0.02, 0.3), (0.4, 0.25), (0.7, 0.3)], 0.03)], stigma=None)
        build_wing(K, sp, s, 'wingH' + sd, part='spd_t3')
        tipH = Wh.tip()
        K.horn('spd_t3', [Wh.P(np.array(0.8), np.array(0.1)), tipH + Wh.es * 0.4, tipH + Wh.es * 1.1 + Wh.ec * 0.25], [0.05, 0.035, 0.0], mat='dark', sides=4, per=2, bone='wingH' + sd)

    # ---------------------------------------------------------------- Focus: ocelli, feathered antennae, crest
    for p in (H((0, 0.58, 0.28)), H((-0.17, 0.66, 0.15)), H((0.17, 0.66, 0.15))):
        q, n = K.snap(p)
        K.eye('fcs_t1', q - n * 0.03, n + HD((0, 0, 0.4)), 0.1, bone='head', lid=False)
    for s, sd in ((-1, 'L'), (1, 'R')):
        C = ant[sd]
        Tg = unit(np.gradient(C, axis=0))
        for i in range(2, len(C) - 1):
            f = i / (len(C) - 1)
            ln = 0.42 * math.sin(math.pi * min(1, f * 1.1)) + 0.1
            out = unit(np.cross(Tg[i], [0, 1, 0]) * s)
            for sg in (-1, 1):
                d = unit(out * 0.3 + np.array([0, sg, 0]) * 0.9 + Tg[i] * 0.5)
                K.horn('fcs_t2', [C[i], C[i] + d * ln], [0.025, 0.0], mat='body', sides=3, per=1, bone='ant' + sd + ('2' if i < 6 else '3'))
        K.ball('fcs_t3', C[-1], (0.14, 0.14, 0.14), mat='accent', seg=8, rings=5, bone='ant' + sd + '3')
    p, n = K.snap(H((0, 0.75, -0.25)))
    K.leaf('fcs_t3', p - n * 0.05, unit(n + HD((0, 0.3, -0.5))), HD((0, 0, 1)), 0.75, 1.2, th=0.05, lobes=4, mat='accent', bone='head')
    p, n = K.snap(T(0.25, 1.1))
    K.leaf('fcs_t3', p - n * 0.05, unit(n), BD, 0.5, 1.0, th=0.05, lobes=3, mat='accent', bone='spine')
