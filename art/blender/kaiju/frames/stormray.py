"""Storm Ray: a manta the size of a thunderhead (ray plan), hovering ~3.5 units above the ground.
Game coords, faces +Z. The wings are one analytic SDF planform (falcate manta outline, thick rounded leading
edge thinning to the trailing edge, gull dihedral with down-curled tips); body, head, cephalic fins and the
whip tail are blended on top. Countershaded: dark back, white belly, glowing lightning markings on the back.
"""
import math

import numpy as np

from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep

DETAIL = 'skin'
UV_SCALE = 0.55
EYE_ACCENT = True
CAM = dict(az=32, el=24)

F = np.float32
Y0 = 4.35          # hover height of the wing mid-plane
SPAN = 7.6         # half span
TAIL = [(0, 4.3, -2.6), (0, 4.22, -4.2), (0, 4.0, -6.0), (0.2, 3.7, -7.8), (0.65, 3.35, -9.4), (1.2, 3.1, -10.7), (1.75, 3.0, -11.6)]
TAIL_R = [0.42, 0.3, 0.22, 0.17, 0.13, 0.11, 0.1]


# ------------------------------------------------------------------ wing planform (numpy, vectorised)
def z_le(u):
    return 2.55 - 2.6 * u ** 1.3 - 2.4 * u ** 4


def z_te(u):
    return -2.95 + 1.6 * u ** 0.7 - 1.05 * u ** 5


def mid_y(u, v):
    """Mid-surface height: gull dihedral, tips curl down, slight trailing-edge droop near the body."""
    tip = np.maximum(u - 0.68, 0)
    return Y0 + 1.75 * u ** 1.6 - 15.0 * tip ** 2 - 0.12 * v * (1 - u)


def half_thick(u, v):
    af = 2.75 * np.sqrt(v) * (1 - v) ** 1.25            # rounded leading edge, long taper to the trailing edge
    return (0.72 * (1 - u) ** 1.25 + 0.05) * af + 0.04


def plan_uv(x, z):
    u = np.clip(np.abs(x) / SPAN, 0, 1)
    le, te = z_le(u), z_te(u)
    v = np.clip((le - z) / np.maximum(le - te, 1e-3), 0, 1)
    return u, v, le, te


class Wing:
    """Both wings as one symmetric SDF primitive (distance is approximate; the zero set is exact)."""

    def __init__(self):
        self.lo = np.array([-SPAN, Y0 - 1.0, z_te(np.linspace(0, 1, 50)).min()])
        self.hi = np.array([SPAN, Y0 + 2.0, z_le(0.0)])

    def mirrored(self):
        return self

    def dist(self, reg, pad=0):
        x, y, z = reg.x, reg.y, reg.z
        ax = np.abs(x)
        u, v, le, te = plan_uv(x, z)
        dy = np.abs(y - mid_y(u, v)) - half_thick(u, v)
        dplan = np.maximum(np.maximum(z - le, te - z), ax - SPAN)
        out = np.where(dplan > 0, np.sqrt(dplan ** 2 + np.maximum(dy, 0) ** 2), np.maximum(dy, dplan))
        return np.broadcast_to(out, reg.shape).astype(F)


def surface_y(P):
    """Mid-plane height under each point: wing mid-surface, or the tail centreline behind the disc."""
    P = np.asarray(P, float)
    u, v, le, te = plan_uv(P[:, 0], P[:, 2])
    ym = mid_y(u, v)
    T = np.array(TAIL)
    tail_y = np.interp(-P[:, 2], -T[:, 2], T[:, 1])
    behind = (P[:, 2] < te + 0.05) & (np.abs(P[:, 0]) < 2.0)
    return np.where(behind, tail_y, ym)


# lightning bolts on the back (x, z) polylines, mirrored
BOLTS = [
    ([(0.55, 1.2), (1.75, 0.85), (2.0, 1.3), (3.3, 0.35), (3.65, 0.85), (5.1, -0.35), (5.4, -0.05), (6.6, -1.0)], 0.22, 0.09),
    ([(3.3, 0.35), (3.75, -0.55), (3.45, -0.45), (4.1, -1.4)], 0.15, 0.07),
    ([(0.65, -0.8), (1.8, -1.0), (2.05, -0.65), (3.0, -1.5)], 0.17, 0.08),
    ([(0.0, 2.1), (0.3, 1.35), (-0.22, 0.75), (0.25, -0.2), (-0.2, -1.1), (0.0, -2.0)], 0.18, 0.1),
]


def seg_dist2(px, pz, a, b):
    ab = np.subtract(b, a)
    t = np.clip(((px - a[0]) * ab[0] + (pz - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
    return np.hypot(px - (a[0] + t * ab[0]), pz - (a[1] + t * ab[1])), t


def bolt_mask(P):
    P = np.asarray(P, float)
    best = np.full(len(P), 9.0)
    for pts, w0, w1 in BOLTS:
        for sgn in ((1,) if pts[0][0] == 0.0 and pts[-1][0] == 0.0 else (1, -1)):
            Q = np.array(pts) * [sgn, 1]
            n = len(Q) - 1
            for i in range(n):
                d, t = seg_dist2(P[:, 0], P[:, 2], Q[i], Q[i + 1])
                w = w0 + (w1 - w0) * (i + t) / n
                best = np.minimum(best, d - w)
    top = P[:, 1] - surface_y(P)
    return np.where((top > 0.02) & (P[:, 2] < 2.6), best, 1.0)


def build(K):
    K.mat('body', rough=0.38, detail_strength=0.4)
    K.mat('belly', rough=0.45, detail_strength=0.25)
    K.mat('plate', color='#2f3c57', rough=0.32, metal=0.35)
    B = K.sdf()
    B.add(Wing(), k=0.0)
    # ---- body disc, back ridge, head
    B.ell((0, Y0 + 0.05, -0.1), (1.65, 0.92, 2.9), k=0.6)
    B.ell((0, Y0 + 0.5, 0.1), (0.8, 0.5, 2.2), k=0.5)                                     # raised spine
    B.ell((0, Y0 - 0.35, 0.4), (1.25, 0.55, 2.2), k=0.5)                                  # belly keel
    B.ell((1.15, Y0 + 0.2, 0.5), (0.8, 0.4, 1.5), rot=(0, 18, 0), k=0.5, mirror=True)     # wing-root muscle
    B.ell((0, Y0 - 0.05, 2.3), (1.45, 0.5, 0.95), k=0.45)                                  # head
    B.ell((1.25, Y0 + 0.05, 2.45), (0.38, 0.32, 0.42), k=0.3, mirror=True)                 # eye turrets
    # cephalic fins: rolled paddles curling forward/down from the head corners
    B.tube([(1.0, Y0 - 0.05, 2.6), (1.28, Y0 - 0.15, 3.45), (1.12, Y0 - 0.45, 4.15), (0.78, Y0 - 0.75, 4.25)], [0.32, 0.28, 0.2, 0.13],
           k=0.25, s=(0.55, 1, 1), mirror=True)
    # tail
    B.tube(TAIL, TAIL_R, k=0.35)
    # pelvic fins (claspers) at the tail root
    B.tube([(0.5, Y0 - 0.25, -2.4), (0.85, Y0 - 0.35, -3.2), (1.0, Y0 - 0.4, -3.6)], [0.28, 0.18, 0.08], k=0.25, s=(1, 0.5, 1), mirror=True)
    # mouth + eye sockets + gill slits (located on the un-cut body)
    B.ell((0, Y0 - 0.2, 3.2), (0.92, 0.17, 0.42), cut=True, k=0.06)
    B.ell((1.38, Y0 + 0.1, 2.55), (0.21, 0.21, 0.21), cut=True, k=0.04, mirror=True)
    gills = []
    for i in range(5):
        q = np.array([0.95 + 0.05 * i, Y0 - 1.0, 1.75 - i * 0.3])
        p, n = B.snap(q[None])
        gills.append((p[0], n[0]))
    for p, n in gills:
        B.box(p, (0.36, 0.06, 0.032), rot=(0, -12, 0), rad=0.03, k=0.025, cut=True, mirror=True)

    def skin(x, y, z):
        pores = -0.012 * fbm(x, y, z, 2, freq=3.3, seed=5)
        ridges = -0.02 * (fbm(x, y, z, 3, freq=0.8, seed=8, ridged=True) - 0.5)
        return pores + ridges
    B.disp(skin, band=0.12)

    CF = np.array([(1.1, Y0 - 0.1, 3.05), (1.28, Y0 - 0.15, 3.45), (1.12, Y0 - 0.45, 4.15), (0.78, Y0 - 0.75, 4.25)])
    cfz = np.array([2.6, 3.45, 4.15, 4.3])
    cfx = np.array([1.0, 1.28, 1.12, 0.78])

    def belly_fn(P):
        P = np.asarray(P, float)
        d = (P[:, 1] - surface_y(P)) + 0.03
        fin = np.full(len(P), False)
        for sg in (-1, 1):
            for i in range(len(CF) - 1):
                a, b = CF[i] * [sg, 1, 1], CF[i + 1] * [sg, 1, 1]
                ab = b - a
                t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
                fin |= np.linalg.norm(P - (a + t[:, None] * ab), axis=1) < 0.42
        inner = np.abs(P[:, 0]) - (np.interp(P[:, 2], cfz, cfx) - 0.04)       # cephalic fins: white only on the inner face
        return np.where(fin, inner, d)
    mouth = S.Ell((0, Y0 - 0.2, 3.2), (0.88, 0.22, 0.46))

    def tint(P, N):
        mott = 1.0 + 0.14 * fbm(P[:, 0], P[:, 1], P[:, 2], 3, freq=0.45, seed=4)
        u, v, _, _ = plan_uv(P[:, 0], P[:, 2])
        edge = 1.0 - 0.25 * smoothstep(0.75, 1.0, v) * (P[:, 1] > surface_y(P))      # darker trailing edge on top
        spots = 1.0 - 0.18 * (fbm(P[:, 0], 0 * P[:, 1], P[:, 2], 2, freq=1.4, seed=12) > 0.35)
        return mott * edge * np.where(P[:, 1] > surface_y(P), spots, 1.0)

    K.skin(B, voxel=0.04, tris=17500, belly=belly_fn, tint=tint,
           regions=[('accent', bolt_mask), ('mouth', lambda P: mouth.dist(S.Points(P)))])

    # ---- skeleton
    K.bone('spine', (0, Y0, -2.6), (0, Y0, 1.6), None, 1.6)
    K.bone('head', (0, Y0, 1.6), (0, Y0, 3.3), 'spine', 1.2)
    K.bone('jaw', (0, Y0 - 0.3, 2.6), (0, Y0 - 0.3, 3.5), 'head', 0.4)
    wing_pts = lambda s: [(s * 1.3, mid_y(0.17, 0.4), 0.2), (s * 3.2, mid_y(0.42, 0.4), -0.15), (s * 5.2, mid_y(0.68, 0.4), -0.9),
                          (s * 7.3, mid_y(0.96, 0.4), -2.1)]
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.chain('wing' + sd, wing_pts(s), 'spine', (1.5, 0.7))
        K.chain('horn' + sd, [(s * 1.0, Y0 - 0.05, 2.6), (s * 1.28, Y0 - 0.15, 3.45), (s * 0.85, Y0 - 0.7, 4.25)], 'head', (0.32, 0.2))
        K.mount('hand' + sd, (s * 0.8, Y0 - 0.75, 4.3), 'horn' + sd + '2')
        K.mount('shoulder' + sd, (s * 1.6, Y0 + 0.6, 0.4), 'spine')
    K.chain('tail', TAIL, 'spine', (0.45, 0.15))
    K.mount('head', (0, Y0 + 0.35, 2.3), 'head')
    K.mount('jaw', (0, Y0 - 0.4, 3.0), 'jaw')
    K.mount('mouth', (0, Y0 - 0.2, 3.55), 'head')
    K.mount('chest', (0, Y0 - 0.7, 1.4), 'spine')
    K.mount('back', (0, Y0 + 0.85, -0.6), 'spine')
    K.mount('tailTip', TAIL[-1], 'tail6')
    K.mount('crown', (0, Y0 + 0.6, 2.0), 'head')

    # ---- eyes + electric nodes (base)
    for s in (-1, 1):
        K.eye('body', (s * 1.38, Y0 + 0.1, 2.55), (s * 0.85, 0.15, 0.5), 0.19, bone='head')
    node_spots = [(3.2, 0.5), (1.85, 1.35), (2.9, -1.4)]
    for s in (-1, 1):
        for x, z in node_spots:
            p, n = K.snap((s * x, Y0 + 2.0, z))
            K.ball('body', p + n * 0.02, (0.13, 0.08, 0.13), mat='accent', seg=8, rings=5)
    p, n = K.snap((0, Y0 + 2, 2.15))
    K.ball('body', p, (0.15, 0.09, 0.15), mat='accent', seg=8, rings=5, bone='head')

    # ---- Might: the tail barb grows into a serrated stinger
    tp = np.array(TAIL[-1])
    td = (tp - np.array(TAIL[-2])) / np.linalg.norm(tp - np.array(TAIL[-2]))
    side = np.cross(td, [0, 1, 0])
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        ln = 0.95 * sc
        K.horn(part, [tp - td * 0.15, tp + td * ln * 0.45 + [0, 0.05, 0], tp + td * ln], [0.14 * (0.8 + 0.2 * sc), 0.11 * sc, 0.0],
               mat='horn', sides=6, flat=0.55, bone='tail6')
        nb = [0, 2, 3, 4][t]
        for j in range(nb):
            for sg in (-1, 1):
                b = tp + td * ln * (0.15 + 0.2 * j)
                K.claw(part, b, side * sg * 0.9 - td * 0.6, 0.28 * sc * (1 - 0.15 * j), 0.05 * sc, curl=-td, bone='tail6')
        if t >= 1:   # barbs along the tail
            for j in range(t * 2):
                q = S.catmull(TAIL, 4)[-3 - j * 2]
                K.spike(part, q + [0, 0.1, 0], (0, 1, -0.4), 0.3 + 0.08 * t)
        if t >= 2:   # horn tips on the cephalic fins
            for s, sd in ((-1, 'L'), (1, 'R')):
                K.horn(part, [(s * 0.8, Y0 - 0.72, 4.2), (s * 0.7, Y0 - 0.85, 4.6), (s * 0.55, Y0 - 1.15, 4.75)], [0.1, 0.07, 0], bone='horn' + sd + '2')
        if t == 3:
            for s, sd in ((-1, 'L'), (1, 'R')):
                K.horn(part, [(s * 1.25, Y0 - 0.1, 3.4), (s * 1.6, Y0 + 0.0, 3.9), (s * 1.75, Y0 + 0.1, 4.4)], [0.11, 0.07, 0], bone='horn' + sd + '1')

    # ---- Armor: dorsal scutes, wing-root shoulder plates, spikes
    def top_plate(part, x, z, rx, rz, bone, lames=0, axis=(1, 0, 0), tris=220):
        p, n = K.snap((x, Y0 + 3, z))
        K.plate(part, p, None, region=S.Ell(p + n * 0.14, (rx, 0.28, rz)), snap=False, bone=bone, lames=lames, lame_axis=axis, tris=tris)
        return p, n

    for j in range(4):   # overlapping dorsal scutes down the spine
        top_plate('arm_t1', 0, 1.3 - j * 0.8, 0.8 - j * 0.06, 0.55, 'spine', tris=180)
    for s, sd in ((-1, 'L'), (1, 'R')):
        top_plate('arm_t1', s * 0.95, 2.35, 0.45, 0.45, 'head', tris=140)
        top_plate('arm_t2', s * 2.35, 0.55, 1.1, 0.8, 'wing' + sd + '1', lames=4, axis=(s, 0, -0.35), tris=300)
        top_plate('arm_t2', s * 3.9, -0.05, 0.75, 0.6, 'wing' + sd + '2', lames=3, axis=(s, 0, -0.45), tris=220)
        for x in (1.7, 2.6, 3.5, 4.4, 5.3):   # leading-edge spikes
            u = x / SPAN
            p, n = K.snap((s * x, mid_y(u, 0.05) + 0.3, z_le(u) - 0.15))
            K.spike('arm_t3', p, n * 0.5 + [0, 0.35, 0.9], 0.55 - (x - 1.7) * 0.05, bone='wing' + sd + ('1' if x < 3.2 else '2'))
    for j in range(5):   # dorsal ridge spikes
        p, n = K.snap((0, Y0 + 2, 1.2 - j * 0.75))
        K.spike('arm_t3', p, n + [0, 0, -0.5], 0.7 - abs(j - 1.5) * 0.08, bone='spine')

    # ---- Energy: wing nodes, glowing leading edge, tesla crystals
    for s, sd in ((-1, 'L'), (1, 'R')):
        for x, z in ((4.9, -0.1), (6.5, -0.95), (3.95, -1.75), (4.05, -1.3)):
            p, n = K.snap((s * x, Y0 + 3, z))
            K.ball('enr_t1', p + n * 0.02, (0.13, 0.07, 0.13), mat='accent', seg=8, rings=5, bone='wing' + sd + ('2' if x < 5.5 else '3'))
        edge = []
        for u in np.linspace(0.12, 0.95, 16):
            p, n = K.snap((s * u * SPAN, mid_y(u, 0.0) + 0.2, z_le(u) + 0.2))
            edge.append(p + n * 0.025)
        rad = np.interp(np.linspace(0, 1, len(edge)), [0, 0.3, 0.7, 1], [0.06, 0.075, 0.05, 0.0])
        for j in range(0, len(edge) - 1, 3):
            K.horn('enr_t2', edge[j:j + 4], list(rad[j:j + 4]), mat='accent', sides=5, per=1)
        p, n = K.snap((s * 1.1, Y0 + 2, -0.4))
        K.crystal('enr_t3', p - n * 0.05, n + [s * 0.3, 0, -0.2], 0.85, seed=2, bone='spine')
        p, n = K.snap((s * 3.0, Y0 + 3, -0.5))
        K.crystal('enr_t3', p - n * 0.05, n + [s * 0.2, 0, 0], 0.6, seed=5, bone='wing' + sd + '1')
    for j in range(4):   # glowing nodes down the tail
        q = S.catmull(TAIL, 4)[3 + j * 4]
        K.ball('enr_t3', q + [0, TAIL_R[min(j + 1, 6)] * 0.8, 0], (0.1, 0.08, 0.12), mat='accent', seg=8, rings=5)

    # ---- Speed: dorsal fin, tail finlets, wingtip frills
    p, n = K.snap((0, Y0 + 1, -2.75))
    K.fin('spd_t1', p - n * 0.05, (0, 1, -0.15), (0, 0, 1), 0.9, 0.9, bone='spine')
    for j in range(3):
        q = S.catmull(TAIL, 4)[5 + j * 4]
        nxt = S.catmull(TAIL, 4)[6 + j * 4]
        K.fin('spd_t2', q + [0, 0.12, 0], (0, 1, 0), q - nxt, 0.45 - j * 0.08, 0.6 - j * 0.1, bone='tail' + str(2 + j))
    for s, sd in ((-1, 'L'), (1, 'R')):
        u = 0.86
        p, n = K.snap((s * u * SPAN, mid_y(u, 0.9), z_te(u) + 0.1))
        K.frill('spd_t3', p, (s * 0.4, -0.2, -1), (s, -0.3, 0), 1.1, 1.3, rays=4, spread=60, bone='wing' + sd + '3')
        p, n = K.snap((s * 0.85, Y0 - 0.1, -2.9))
        K.fin('spd_t3', p, (s * 0.5, -0.1, -1), (s * 0.5, 0, 0.3), 0.7, 0.6, bone='spine')

    # ---- Focus: extra eyes, crest, third eye
    for s in (-1, 1):
        p, n = K.snap((s * 0.75, Y0 + 1, 2.45))
        K.eye('fcs_t1', p - n * 0.03, n + [s * 0.2, 0, 0.5], 0.11, bone='head')
        p, n = K.snap((s * 1.35, Y0 - 0.4, 2.3))
        K.eye('fcs_t3', p - n * 0.03, n + [s * 0.4, 0, 0.4], 0.1, bone='head')
    p, n = K.snap((0, Y0 + 1, 1.55))
    K.leaf('fcs_t2', p - n * 0.05, n, (0, 0, 1), 0.7, 1.4, th=0.06, lobes=4, mat='accent', bone='head')
    p, n = K.snap((0, Y0 + 1, 2.7))
    K.eye('fcs_t3', p - n * 0.04, n + [0, 0, 0.5], 0.16, bone='head')
