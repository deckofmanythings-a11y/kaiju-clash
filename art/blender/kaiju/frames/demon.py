"""Archfiend: a towering horned demon (biped plan with tail + bat wings). Game coords, ~11.2 units to the
top of the skull (horns / wings higher), faces +Z.

Hunched, heavily muscled torso on digitigrade goat legs (shaggy fur below the knee, cloven hooves), long
clawed arms, a whip tail with a barbed spade, and half-spread bat wings built as rigid double-sided
membrane panels between finger bones (every panel bound to its own wing bone). Hellfire fissures (Voronoi
cell borders) glow through the chest and forearms; glowing eyes and throat.
"""
import math

import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep, voronoi

DETAIL = 'skin'
UV_SCALE = 0.7
EYE_ACCENT = True
CAM = dict(az=35, el=9)

INFO = dict(name='Archfiend', colors=dict(body='#8c1c1a', belly='#2b2220', accent='#ff8a1c'))

# ---- key joints (right side, +X); L mirrored
SH = np.array([2.55, 8.95, 0.55])     # shoulder
EL = np.array([3.35, 6.55, 0.05])     # elbow
WR = np.array([3.25, 4.35, 1.3])      # wrist
HIP = np.array([1.35, 5.35, -0.5])
KN = np.array([1.75, 3.75, 0.65])     # knee (forward)
HK = np.array([1.65, 1.95, -0.85])    # hock (raised heel, back)
FT = np.array([1.65, 0.62, -0.15])    # fetlock
HOOF = np.array([1.65, 0.3, 0.3])

TAIL = [(0, 5.3, -1.35), (0, 4.55, -2.75), (0.2, 3.4, -4.15), (0.7, 2.25, -5.35), (1.45, 1.4, -6.35), (2.45, 1.05, -7.0), (3.45, 1.25, -7.35)]
TAIL_R = [0.78, 0.56, 0.42, 0.32, 0.24, 0.18, 0.13]

# wing skeleton (right side)
W_ROOT = np.array([1.05, 9.45, -1.45])
W_ELB = np.array([2.6, 10.35, -2.55])
W_WR = np.array([4.0, 11.75, -2.7])
FINGERS = [   # wrist -> knuckle -> tip
    (np.array([6.0, 12.85, -3.05]), np.array([8.1, 13.35, -3.45])),
    (np.array([6.4, 11.55, -3.35]), np.array([8.7, 10.75, -3.9])),
    (np.array([5.75, 9.85, -3.3]), np.array([7.45, 7.95, -3.8])),
    (np.array([4.75, 9.0, -2.95]), np.array([5.2, 6.55, -3.1])),
]
W_BODY = np.array([1.3, 6.9, -1.35])   # membrane trailing attachment on the flank


def M(s):
    return np.array([s, 1, 1])


def poly_at(pts, u):
    """Point at arc-length fraction u along a polyline."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    i = int(np.clip(np.searchsorted(cum, u) - 1, 0, len(seg) - 1))
    t = (u - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
    return pts[i] + (pts[i + 1] - pts[i]) * t


def ray_ring(sdf, c, bx, bz, n=15, rmax=1.3, off=0.05):
    """Ring hugging the surface around axis point c: bisect along rays from c in the (bx, bz) plane."""
    A = np.linspace(0, 2 * math.pi, n)
    D = np.array([math.cos(a) * bx + math.sin(a) * bz for a in A])
    lo, hi = np.zeros(n), np.full(n, rmax)
    for _ in range(18):
        mid = (lo + hi) * 0.5
        inside = sdf.at(c + D * mid[:, None]) < 0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return c + D * (lo[:, None] + off), D


def seg_index(pts, u):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    return int(np.clip(np.searchsorted(cum, u, side='right') - 1, 0, len(seg) - 1))


def membrane(K, part, A, B, bone, scallop=0.22, billow=0.28, nu=10, nv=6, mat='membrane', bones_a=None, bones_b=None):
    """Thin double-sided panel between two polylines A and B (both run from the wing root outward);
    the free edge between A[-1] and B[-1] is scalloped inward, the sheet billows back (-Z).
    bones_a / bones_b (one bone per polyline segment) give the panel blended per-vertex weights: each edge
    follows exactly the bones of the finger it lies on (joints coincide, so edges stay welded to the
    rigid finger tubes and to neighbouring panels), the interior is a harmonic blend between the edges."""
    def f(u, v):
        a, b = poly_at(A, u), poly_at(B, u)
        p = a + (b - a) * v
        edge = smoothstep(0.55, 1.0, u)
        mid = math.sin(math.pi * v)
        root = (A[0] + B[0]) * 0.5
        p = p + (root - p) * scallop * edge * mid
        p = p + np.array([0, 0, -billow]) * mid * math.sin(math.pi * min(u, 1.0) * 0.5 + 0.3)
        return p
    V, T, _ = geom.grid(f, nu, nv)
    vv = np.tile(np.linspace(0, 1, nv + 1), nu + 1)
    uu = np.repeat(np.linspace(0, 1, nu + 1), nv + 1)
    col = 0.55 + 0.45 * np.sin(np.pi * vv) ** 0.6 * (0.7 + 0.3 * uu)
    K.piece(part, V, T, mat, col=col, bone=bone)
    if bones_a is None and bones_b is None:
        return
    bones_a = bones_a or [bone] * (len(A) - 1)
    bones_b = bones_b or [bone] * (len(B) - 1)
    names = sorted(set(bones_a) | set(bones_b))
    W = np.zeros((nu + 1, nv + 1, len(names)))
    for i in range(nu + 1):
        u = i / nu
        wa = np.zeros(len(names)); wa[names.index(bones_a[seg_index(A, u)])] = 1
        wb = np.zeros(len(names)); wb[names.index(bones_b[seg_index(B, u)])] = 1
        for j in range(nv + 1):
            v = j / nv
            W[i, j] = wa * (1 - v) + wb * v
    for _ in range(40):   # relax the interior (edges v=0 / v=1 fixed, u ends free)
        up = np.concatenate([W[1:2], W[:-1]], 0)
        dn = np.concatenate([W[1:], W[-2:-1]], 0)
        avg = (up + dn + np.roll(W, 1, 1) + np.roll(W, -1, 1)) * 0.25
        W[:, 1:-1] = avg[:, 1:-1]
    K.pieces[part][-1]['W'] = (names, W.reshape(-1, len(names)))


def install_blend_weights(K):
    """Pieces carrying p['W'] = (bone names, per-vertex weight matrix) get those weights instead of the
    builder's rigid single-bone binding (wrapped here so no shared builder code changes)."""
    orig = K._piece_object

    def patched(name, pieces):
        ob = orig(name, pieces)
        off = 0
        for p in pieces:
            n = len(p['V'])
            if 'W' in p:
                names, Wm = p['W']
                idx = list(range(off, off + n))
                for vg in ob.vertex_groups:
                    vg.remove(idx)
                for k, bn in enumerate(names):
                    vg = ob.vertex_groups.get(bn) or ob.vertex_groups.new(name=bn)
                    for i in range(n):
                        if Wm[i, k] > 0.01:
                            vg.add([off + i], float(Wm[i, k]), 'REPLACE')
            off += n
        return ob
    K._piece_object = patched


def install_outward_horns(K):
    """geom.sweep orients each triangle away from the whole mesh centroid, which flips the faces on the
    inner side of strongly curved tubes (the C-shaped horns, the bent wing arm) inward; with backface
    culling they vanish and the tube looks torn / hollow. Re-orient each triangle away from its own
    point on the centreline instead (axis direction for end caps)."""
    def horn(part, pts, radii, mat='horn', sides=6, per=2, flat=1.0, bone=None, shade=(0.55, 1.0), up=None):
        V, T, u = geom.sweep(pts, radii, sides, per, flat, up)
        C = geom.catmull(pts, per)
        n = len(C)
        tan = np.gradient(C, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        ri = u[T].mean(1) * (n - 1)
        i0 = np.clip(np.floor(ri).astype(int), 0, n - 2)
        f = (ri - i0)[:, None]
        ax = C[i0] * (1 - f) + C[i0 + 1] * f
        cen = V[T].mean(1)
        d = cen - ax
        capm = np.ptp(u[T], axis=1) < 1e-9        # end caps: all three vertices on one ring
        d[capm] = np.where((ri[capm] < (n - 1) * 0.5)[:, None], -tan[0], tan[-1])
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        flip = (fn * d).sum(1) < 0
        T = T.copy()
        T[flip] = T[flip][:, ::-1]
        K.piece(part, V, T, mat, col=shade[0] + (shade[1] - shade[0]) * u, bone=bone)
    K.horn = horn


def blade(K, part, base, d, n, length, width, th, mat, bone, barb=0.35):
    """Flat spade / barb: outline in the (d, side) plane, bi-convex thickness along n."""
    d, n = geom.unit(d), geom.unit(n)
    side = geom.unit(np.cross(n, d))
    base = np.asarray(base, float)
    half = [(0.0, 0.07), (-barb * 0.55, 0.5), (0.18, 0.46), (0.42, 0.42), (0.7, 0.24), (1.0, 0.0)]
    out = [(a, b) for a, b in half] + [(a, -b) for a, b in reversed(half[:-1])]
    out = out[:-1] if out[-1] == out[0] else out
    P = [base + d * a * length + side * b * width for a, b in out]
    c = base + d * 0.38 * length
    V = np.array(P + [c + n * th, c - n * th])
    m = len(P)
    T = []
    for i in range(m):
        j = (i + 1) % m
        T += [(i, j, m), (j, i, m + 1)]
    T = geom._consistent(V, np.array(T))
    K.piece(part, V, T, mat, smooth=False, col=np.concatenate([np.full(m, 0.7), [1.0, 1.0]]), bone=bone)


def build(K):
    K.mat('body', rough=0.5, detail_strength=0.35)
    K.mat('belly', rough=0.65, detail_strength=0.3)
    K.mat('horn', color='#2a2321', rough=0.42)
    K.mat('hair', color='#2c1a16', rough=0.92, detail=True, detail_kind='fur', vcol=True)
    K.mat('tooth', color='#e3d6b8', rough=0.35, vcol=True)
    K.mat('bone', color='#cbbd9c', rough=0.45, vcol=True)
    K.mat('plate', color='#6a5f57', rough=0.34, metal=0.65, vcol=True)
    K.mat('membrane', color='#4c1311', rough=0.7, vcol=True, double=True, detail=False)
    K.mat('accent', emit_strength=4.0)
    K.mat('eye', emit_strength=5.0)

    install_blend_weights(K)
    install_outward_horns(K)
    B = K.sdf()
    # ---- torso: narrow waist, huge hunched rib cage, trapezius hump
    B.tube([(0, 5.35, -0.7), (0, 6.5, -0.35), (0, 7.8, 0.15), (0, 9.0, 0.35)], [1.45, 1.3, 1.85, 1.95], k=0.0, s=(1.15, 1, 0.95))
    B.ell((0, 5.45, -0.85), (1.75, 1.15, 1.35), k=0.45)                                   # hips / glutes
    B.ell((0, 8.2, 0.6), (1.95, 1.55, 1.3), k=0.5)                                      # rib cage
    B.ell((0.92, 8.5, 1.72), (1.0, 0.78, 0.6), rot=(25, 0, 0), k=0.3, mirror=True)        # pectorals
    B.ell((0, 6.65, 0.85), (1.05, 1.15, 0.72), k=0.45)                                    # abdomen
    B.ell((0, 9.35, -0.2), (1.45, 0.72, 1.1), k=0.5)                                      # trapezius hump
    B.tube([(0, 9.4, 0.1), (1.4, 9.4, 0.3), (2.3, 9.1, 0.5)], [0.9, 0.82, 0.75], k=0.4, mirror=True)
    B.ell(SH, (1.08, 1.05, 1.08), k=0.3, mirror=True)                                     # deltoids
    B.ell((1.6, 7.7, -0.45), (0.9, 1.45, 1.1), rot=(-20, 0, 0), k=0.4, mirror=True)       # lats
    B.ell((1.15, 6.4, 0.45), (0.55, 0.9, 0.6), k=0.35, mirror=True)                       # obliques
    B.ell((1.05, 9.3, -1.25), (0.6, 0.75, 0.5), k=0.35, mirror=True)                      # wing-root muscles
    # ---- arms: long, ropey, clawed hands hanging to the knees
    B.tube([SH, (3.1, 7.7, 0.45), EL], [1.0, 0.82, 0.62], k=0.3, mirror=True)
    B.ell((3.0, 7.65, 0.9), (0.58, 0.95, 0.58), rot=(-8, 0, 12), k=0.28, mirror=True)     # biceps
    B.ell((3.45, 7.6, 0.05), (0.56, 1.0, 0.56), rot=(10, 0, 12), k=0.28, mirror=True)     # triceps
    B.ell(EL + (0.1, 0.0, -0.3), (0.45, 0.45, 0.42), k=0.2, mirror=True)                  # elbow point
    B.tube([EL, (3.45, 5.45, 0.7), WR], [0.66, 0.8, 0.46], k=0.28, mirror=True)
    B.ell((3.5, 5.6, 0.62), (0.75, 1.0, 0.7), rot=(25, 0, -6), k=0.28, mirror=True)       # forearm muscle
    B.ell(WR + (-0.05, -0.45, 0.3), (0.3, 0.62, 0.52), rot=(25, 0, 0), k=0.22, mirror=True)  # palm (faces -X)
    FING = []
    for i, dz in enumerate((-0.27, -0.09, 0.09, 0.27)):
        ln = (0.85, 1.0, 0.95, 0.8)[i]
        r0 = WR + (-0.05, -0.85, 0.3 + dz * 1.15)
        pts = [r0, r0 + (-0.05, -0.45 * ln, 0.18 * ln), r0 + (-0.22, -0.8 * ln, 0.3 * ln)]
        FING.append(pts)
        B.tube(pts, [0.15, 0.13, 0.1], k=0.08, mirror=True)
    THUMB = [WR + (-0.25, -0.4, 0.68), WR + (-0.38, -0.7, 0.95), WR + (-0.52, -0.95, 1.0)]
    B.tube(THUMB, [0.17, 0.14, 0.11], k=0.1, mirror=True)
    # ---- legs: digitigrade goat legs
    B.tube([HIP, (1.65, 4.6, 0.15), KN], [1.25, 1.02, 0.7], k=0.35, s=(0.9, 1, 1.05), mirror=True)
    B.ell((1.6, 4.55, 0.4), (0.8, 1.05, 0.72), rot=(-30, 0, 0), k=0.3, mirror=True)       # quadriceps
    B.ell((1.5, 4.7, -0.55), (0.75, 1.0, 0.7), rot=(30, 0, 0), k=0.3, mirror=True)        # hamstring
    B.ell(KN + (0, 0, 0.1), (0.5, 0.5, 0.5), k=0.2, mirror=True)                          # knee
    B.tube([KN, (1.72, 2.85, -0.1), HK], [0.66, 0.6, 0.42], k=0.22, mirror=True)
    B.ell((1.72, 3.0, -0.25), (0.62, 0.85, 0.62), rot=(40, 0, 0), k=0.25, mirror=True)    # calf (furred)
    B.ell(HK + (0, 0.05, -0.08), (0.36, 0.38, 0.4), k=0.15, mirror=True)                  # hock
    B.tube([HK, (1.65, 1.25, -0.5), FT], [0.38, 0.33, 0.33], k=0.15, mirror=True)        # cannon bone
    B.ell(FT + (0, -0.02, 0.05), (0.5, 0.42, 0.5), k=0.2, mirror=True)                    # fetlock feathering
    B.tube([FT, HOOF + (0, 0.05, -0.1)], [0.33, 0.36], k=0.15, mirror=True)               # pastern
    for dx in (-0.21, 0.21):                                                              # cloven hoof
        B.ell(HOOF + (dx, -0.05, 0.12), (0.24, 0.27, 0.42), k=0.1, mirror=True)
    B.box(HOOF + (0, -0.05, 0.38), (0.035, 0.32, 0.3), k=0.03, rad=0.02, cut=True, mirror=True)
    B.ell(FT + (0, -0.35, -0.42), (0.11, 0.12, 0.14), k=0.06, mirror=True)               # dewclaws
    # ---- tail
    B.tube(TAIL, TAIL_R, k=0.45)
    # ---- neck + head: low, thrust forward between the shoulders
    B.tube([(0, 9.15, 0.6), (0, 10.0, 1.5), (0, 10.4, 2.2)], [1.05, 0.88, 0.78], k=0.4)
    B.ell((0, 10.85, 2.55), (0.85, 0.8, 0.9), k=0.3)                                      # cranium
    B.tube([(-0.82, 11.08, 3.05), (-0.4, 11.0, 3.38), (0, 10.9, 3.48), (0.4, 11.0, 3.38), (0.82, 11.08, 3.05)],
           [0.24, 0.27, 0.22, 0.27, 0.24], k=0.12, s=(1, 0.75, 1))                         # heavy brow shelf
    B.ell((0.6, 10.7, 3.1), (0.32, 0.28, 0.35), k=0.2, mirror=True)                      # cheekbones
    B.ell((0, 10.4, 3.3), (0.68, 0.48, 0.62), k=0.28)                                     # brutish muzzle
    B.ell((0, 10.65, 3.75), (0.32, 0.2, 0.22), k=0.15)                                    # flat broad nose
    B.ell((0.7, 10.25, 2.65), (0.36, 0.55, 0.5), k=0.25, mirror=True)                      # masseters
    B.tube([(0, 9.95, 2.3), (0, 9.68, 3.05), (0, 9.55, 3.55)], [0.62, 0.55, 0.45], k=0.2, s=(1.35, 0.62, 1))  # open lower jaw
    B.ell((0, 9.85, 3.55), (0.3, 0.2, 0.2), k=0.12)                                       # chin
    B.ell((0, 10.08, 3.32), (0.58, 0.23, 0.6), rot=(12, 0, 0), cut=True, k=0.08)           # mouth
    B.ell((0.4, 10.87, 3.3), (0.22, 0.14, 0.19), rot=(0, 0, -12), cut=True, k=0.05, mirror=True)   # eye sockets
    B.ell((0.13, 10.6, 3.95), (0.08, 0.06, 0.08), cut=True, k=0.03, mirror=True)          # nostrils

    # ---- back: erector muscles flanking a ridge of vertebra knobs, shoulder blades under the wing roots
    def back_z(x, y):
        lo, hi = 0.0, 3.5
        for _ in range(20):
            mid = (lo + hi) * 0.5
            if B.at(np.array([[x, y, -mid]]))[0] < 0:
                lo = mid
            else:
                hi = mid
        return -lo
    ys = [9.3, 8.6, 7.9, 7.2, 6.5, 5.9]
    B.tube([(0.5, y, back_z(0.5, y) + 0.22) for y in ys], [0.38, 0.45, 0.45, 0.42, 0.38, 0.32], k=0.3, mirror=True)
    for y in np.arange(9.75, 5.6, -0.42):
        B.ell((0, y, back_z(0, y) + 0.02), (0.15, 0.13, 0.14), k=0.12)
    B.ell((0.95, 8.55, back_z(0.95, 8.55) + 0.12), (0.62, 0.78, 0.22), rot=(-10, 0, 18), k=0.25, mirror=True)

    # ---- material regions
    belly = K.sdf()
    belly.tube([(0, 9.35, 2.0), (0, 8.75, 1.95), (0, 7.6, 1.7), (0, 6.4, 1.3), (0, 5.4, 0.55)], [0.55, 1.05, 1.35, 1.0, 0.8], k=0)
    belly.tube([(0, 9.65, 2.75), (0, 9.48, 3.5)], [0.42, 0.32], k=0.1)        # under-jaw
    for s in (-1, 1):
        belly.ell(M(s) * (WR + (-0.3, -0.8, 0.45)), (0.28, 0.8, 0.65), k=0)     # palms / inner fingers

    legs = []
    for s in (-1, 1):
        legs.append((M(s) * KN, M(s) * (1.72, 2.85, -0.1), M(s) * HK, M(s) * FT))

    def hair_mask(P):
        P = np.asarray(P, float)
        best = np.full(len(P), 9.0)
        for kn, cf, hk, ft in legs:
            for a, b in ((kn + (0, -0.25, 0), cf), (cf, hk), (hk, ft)):
                ab = b - a
                t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
                best = np.minimum(best, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
        wav = 0.12 * np.sin(P[:, 0] * 9 + P[:, 2] * 7)
        top = P[:, 1] - (3.35 + wav)
        bot = (0.54 + wav * 0.3) - P[:, 1]
        return np.maximum(np.maximum(best - 1.15, top), bot)

    def hoof_mask(P):
        P = np.asarray(P, float)
        d = np.full(len(P), 9.0)
        for s in (-1, 1):
            h = M(s) * HOOF
            d = np.minimum(d, np.maximum(np.abs(P[:, 0] - h[0]) - 0.7, np.abs(P[:, 2] - h[2]) - 0.9))
        return np.maximum(d, P[:, 1] - (0.54 + 0.12 * np.sin(P[:, 0] * 9 + P[:, 2] * 7) * 0.3))

    # hellfire fissures: Voronoi cell borders on the chest/abdomen and forearms
    def zone(P):
        P = np.asarray(P, float)
        z1 = S.Ell((0, 7.55, 1.75), (1.9, 1.75, 1.0)).dist(S.Points(P))
        z2 = np.full(len(P), 9.0)
        for s in (-1, 1):
            z2 = np.minimum(z2, S.Tube([M(s) * (EL + (0.05, -0.35, 0.25)), M(s) * (3.45, 5.45, 0.8), M(s) * (WR + (0, 0.25, 0))], [0.95, 1.0, 0.7]).dist(S.Points(P)))
        return np.minimum(z1, z2)

    def crack(P):
        P = np.asarray(P, float)
        f1, f2, _ = voronoi(P[:, 0] * 1.25, P[:, 1] * 0.95, P[:, 2] * 1.25, 1.0, seed=31)
        w = 0.045 + 0.03 * fbm(P[:, 0], P[:, 1], P[:, 2], 2, freq=2.0, seed=4)
        return (f2 - f1) - w

    def fissure_mask(P):
        P = np.asarray(P, float)
        z = zone(P)
        fade = np.clip(z / 0.35, 0, 1)
        return np.where(z < 0.35, crack(P) + 0.12 * fade, 1.0)

    def skin(x, y, z):
        P = np.stack([x, y, z], 1)
        rough = -0.035 * (fbm(x, y, z, 3, freq=0.7, seed=3, ridged=True) - 0.5) - 0.008 * fbm(x, y, z, 2, freq=3.0, seed=9)
        # sinew striations
        stri = -0.02 * np.sin(x * 7 + y * 2.5 + 3 * fbm(x, y, z, 2, freq=0.8, seed=12))
        hm = hair_mask(P)
        wh = smoothstep(0.1, -0.15, hm)
        g1, g2, _ = voronoi(x * 3.2, y * 0.9, z * 3.2, 1.0, seed=7)
        clump = smoothstep(0.0, 0.35, g2 - g1)
        hair = -0.13 * clump - 0.03 * fbm(x, y * 0.4, z, 2, freq=5.0, seed=2) + 0.05
        zf = zone(P)
        cr = np.where(zf < 0.35, crack(P), 1.0)
        groove = 0.05 * smoothstep(0.06, -0.02, cr)
        return (rough + stri) * (1 - wh) + hair * wh + groove
    smooth = K.sdf()           # undisplaced body for armour plates
    smooth.ops = list(B.ops)
    smooth._lo, smooth._hi = B._lo.copy(), B._hi.copy()
    B.disp(skin, band=0.2)

    def tint(P, N):
        y = P[:, 1]
        ember = 1.0 - 0.28 * smoothstep(1.2, -0.4, zone(P))       # darker, charred around the cracks
        top = 1.0 + 0.08 * smoothstep(8.0, 10.5, y)
        return ember * top

    mouth = S.Ell((0, 10.08, 3.32), (0.56, 0.25, 0.58), rot=(12, 0, 0))
    K.skin(B, voxel=0.05, tris=12000, belly=belly, tint=tint,
           regions=[('hair', hair_mask), ('horn', hoof_mask), ('accent', fissure_mask), ('mouth', lambda P: mouth.dist(S.Points(P)))])

    # ---- skeleton
    K.bone('pelvis', (0, 5.3, -0.8), (0, 6.6, -0.45), None, 1.7)
    K.bone('spine', (0, 6.6, -0.45), (0, 8.0, -0.05), 'pelvis', 1.9)
    K.bone('chest', (0, 8.0, -0.05), (0, 9.4, 0.45), 'spine', 2.0)
    K.bone('neck', (0, 9.4, 0.7), (0, 10.4, 2.05), 'chest', 0.9)
    K.bone('head', (0, 10.4, 2.05), (0, 10.65, 3.7), 'neck', 0.85)
    K.bone('jaw', (0, 10.0, 2.3), (0, 9.55, 3.6), 'head', 0.45)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        K.bone('upperarm' + sd, m * SH, m * EL, 'chest', 0.95)
        K.bone('forearm' + sd, m * EL, m * WR, 'upperarm' + sd, 0.8)
        K.bone('hand' + sd, m * WR, m * (WR + (-0.15, -1.6, 0.6)), 'forearm' + sd, 0.6)
        K.bone('thigh' + sd, m * HIP, m * KN, 'pelvis', 1.1)
        K.bone('shin' + sd, m * KN, m * HK, 'thigh' + sd, 0.7)
        K.bone('foot' + sd, m * HK, m * (HOOF + (0, -0.1, 0.45)), 'shin' + sd, 0.5)
        K.chain('wing' + sd, [m * W_ROOT, m * W_WR, m * FINGERS[0][0], m * FINGERS[0][1]], 'chest', 0.25)
        K.mount('hand' + sd, m * (WR + (-0.1, -1.0, 0.55)), 'hand' + sd)
        K.mount('shoulder' + sd, m * (SH + (0.1, 1.0, 0.0)), 'chest')
    K.chain('tail', TAIL, 'pelvis', (0.75, 0.2))
    K.mount('head', (0, 10.85, 2.6), 'head')
    K.mount('jaw', (0, 9.8, 2.95), 'jaw')
    K.mount('mouth', (0, 10.1, 3.85), 'head')
    K.mount('chest', (0, 8.4, 2.25), 'chest')
    K.mount('back', (0, 8.4, -1.7), 'spine')
    K.mount('tailTip', TAIL[-1], 'tail6')
    K.mount('crown', (0, 12.15, 2.5), 'head')

    # ---- eyes, teeth, throat glow (base)
    for s in (-1, 1):
        K.eye('body', (s * 0.4, 10.87, 3.3), (s * 0.35, -0.05, 1), 0.14, bone='head', up=(0, 1, 0), pupil=False)
        for i in range(5):
            a = math.radians(14 + i * 14)
            x, z = s * math.sin(a) * 0.5, 3.32 + math.cos(a) * 0.5
            ln = 0.2 if i != 2 else 0.32
            K.horn('body', [(x, 10.26, z), (x * 1.02, 10.26 - ln * 0.6, z + 0.02), (x, 10.26 - ln, z)], [0.06, 0.04, 0], mat='tooth', sides=5, bone='head', shade=(0.7, 1))
        for i in range(4):
            a = math.radians(18 + i * 15)
            x, z = s * math.sin(a) * 0.44, 3.3 + math.cos(a) * 0.42
            K.horn('body', [(x, 9.82, z), (x * 1.02, 9.95, z), (x, 10.02, z - 0.02)], [0.05, 0.035, 0], mat='tooth', sides=5, bone='jaw', shade=(0.7, 1))
    K.ball('body', (0, 10.06, 2.92), (0.34, 0.16, 0.22), mat='accent', bone='head')   # throat furnace

    # ---- wings: arm + finger bones, membrane panels
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        root, elb, wr = m * W_ROOT, m * W_ELB, m * W_WR
        b1, b2, b3 = 'wing' + sd + '1', 'wing' + sd + '2', 'wing' + sd + '3'
        K.horn('body', [root + (0, -0.05, 0.25), elb, wr], [0.34, 0.26, 0.2], mat='body', sides=7, per=3, bone=b1, shade=(0.75, 1.0))
        K.ball('body', wr, (0.24, 0.22, 0.24), mat='body', bone=b1)
        fpts = []
        for k, (kn, tip) in enumerate(FINGERS):
            kn, tip = m * kn, m * tip
            fpts.append((wr, kn, tip))
            if k == 0:
                K.horn('body', [wr, kn], [0.17, 0.13], mat='body', sides=5, per=2, bone=b2, shade=(0.8, 1.0))
                K.horn('body', [kn, (kn + tip) * 0.5 + (0, 0.15, 0), tip], [0.13, 0.08, 0.0], mat='body', sides=5, per=3, bone=b3, shade=(0.8, 1.0))
            else:
                K.horn('body', [wr, kn, tip], [0.15, 0.1, 0.0], mat='body', sides=5, per=3, bone=b2, shade=(0.8, 1.0))
        # panels: inner (flank -> arm + last finger), between fingers, outermost bound to the tip bone
        body_line = [root + (0, -0.3, 0.1), (root + m * W_BODY) * 0.5 + (s * 0.15, 0, -0.15), m * W_BODY]
        membrane(K, 'body', body_line, [root, elb, wr, fpts[3][1], fpts[3][2]], b1, scallop=0.18, billow=0.35, nu=12, nv=7,
                 bones_a=['chest', 'spine'], bones_b=[b1, b1, b2, b2])
        membrane(K, 'body', list(fpts[3]), list(fpts[2]), b2, scallop=0.22, billow=0.3)
        membrane(K, 'body', list(fpts[2]), list(fpts[1]), b2, scallop=0.22, billow=0.3)
        membrane(K, 'body', list(fpts[1]), list(fpts[0]), b2, scallop=0.2, billow=0.25, bones_a=[b2, b2], bones_b=[b2, b3])

    # ---- tail spade (base)
    tp, tq = np.array(TAIL[-1]), np.array(TAIL[-2])
    td = geom.unit(tp - tq)
    blade(K, 'body', tp - td * 0.15, td, (0, 1, 0.2), 1.75, 1.15, 0.11, 'body', 'tail6')

    # ---- Might: horns + hand claws + tusks + wing-thumb hooks (exclusive, complete sets)
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        for s, sd in ((-1, 'L'), (1, 'R')):
            m = M(s)
            # sweeping horns: out + back, then up, curling forward more each tier
            h = [0.85, 1.02, 1.2, 1.38][t]
            curl = [0.0, 0.3, 0.6, 0.9][t]
            base = np.array([s * 0.58, 11.3, 2.55])
            pts = [base - (s * 0.15, 0.12, -0.05),
                   base + (s * 0.5 * h, 0.05 * h, -0.15 * h),
                   base + (s * 1.0 * h, 0.25 * h, -0.3 * h),
                   base + (s * 1.35 * h, 0.7 * h, -0.3 * h),
                   base + (s * 1.4 * h, 1.2 * h, (-0.1 + curl * 0.4) * h),
                   base + (s * (1.25 - curl * 0.3) * h, 1.55 * h, (0.15 + curl * 0.6) * h)]
            r0 = 0.36 + 0.06 * t
            K.horn(part, pts, [r0, r0 * 0.98, r0 * 0.9, r0 * 0.74, r0 * 0.48, 0.0], bone='head', per=3, sides=8, flat=0.9)
            if t >= 3:   # second, smaller pair rising off the brow
                b2_ = np.array([s * 0.35, 11.2, 3.15])
                K.horn(part, [b2_, b2_ + (s * 0.2, 0.45, -0.05), b2_ + (s * 0.3, 0.85, -0.3)], [0.11, 0.07, 0], bone='head', sides=6)
            # hand claws
            k = 0.75 + 0.3 * sc
            for f in FING + [THUMB]:
                tip = m * np.asarray(f[-1])
                d = geom.unit(m * (np.asarray(f[-1]) - np.asarray(f[-2])))
                K.claw(part, tip - d * 0.03, d, 0.38 * k, 0.085 * (0.8 + 0.2 * sc), curl=(-s * 0.7, 0, -0.5), bone='hand' + sd)
            # tusks from the lower jaw
            tl = [0.32, 0.5, 0.7, 0.92][t]
            tb = np.array([s * 0.44, 9.82, 3.38])
            K.horn(part, [tb, tb + (s * 0.08, tl * 0.6, 0.1 * tl), tb + (s * 0.2, tl * 1.0, 0.0), tb + (s * 0.3, tl * 1.15, -0.25 * tl)],
                   [0.085 + 0.015 * t, 0.07, 0.04, 0], mat='tooth', bone='jaw', per=3, sides=6)
            # wing-thumb hook at the wrist
            wr = m * W_WR
            K.claw(part, wr + (0, 0.1, 0.1), (s * 0.2, 0.6, 0.8), 0.45 * k, 0.09, curl=(0, -1, 0), bone='wing' + sd + '1')
        if t >= 2:   # barbs on the tail spade
            for s in (-1, 1):
                b = tp + td * 0.25 + np.cross((0, 1, 0), td) * s * 0.35
                K.claw(part, b, geom.unit(-td + np.cross((0, 1, 0), td) * s * 0.6), 0.35 * sc * 0.6, 0.06, curl=-td, bone='tail6')
        if t >= 1:   # spinal ridge spikes on the tail
            for j in range(t + 1):
                q = S.catmull(TAIL, 4)[3 + j * 3]
                K.spike(part, q + (0, TAIL_R[min(j, 6)] * 0.7, 0), (0, 1, -0.4), 0.25 + 0.07 * t, bone='tail' + str(1 + min(j, 5)))

    # ---- Armour: pauldrons + bracers, ribcage breastplate + greaves, spikes
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        # pauldron: shell of a clean convex cap (not the lumpy deltoid), cut flat by a box, 3 stepped lames
        pe = S.Ell(m * (SH + (0.12, 0.25, 0.0)), (1.22, 1.08, 1.22), rot=(0, 0, -24 * s))
        pc = m * (SH + (0.3, 0.85, 0.0))
        K.plate('arm_t1', pc, None, region=S.Box(pc, (1.4, 0.72, 1.4), rot=(0, 0, -24 * s), rad=0.03), snap=False,
                bone='upperarm' + sd, lames=3, lame_step=0.12, sdf=pe, off=0.02, th=0.07, k=0.0, bevel=False, tris=480)
        ax = geom.unit(WR - (3.45, 5.45, 0.7))
        bx, _, bz = geom.basis(ax)
        for k, f in enumerate((0.15, 0.45, 0.75)):
            c = (3.45, 5.45, 0.7) + (WR - (3.45, 5.45, 0.7)) * f
            rp, D = ray_ring(smooth, np.asarray(c, float), bx, bz, rmax=0.95)
            K.horn('arm_t1', list(m * rp), [0.09] * len(rp), mat='plate', sides=5, per=1, bone='forearm' + sd)
            if k == 1:   # bracer studs -> t3 spikes
                for j in (2, 6, 10):
                    K.spike('arm_t3', m * rp[j], m * D[j], 0.4, bone='forearm' + sd)
        ke = S.Ell(m * (KN + (0, 0.02, 0.12)), (0.6, 0.7, 0.58))
        kc = m * (KN + (0, 0.02, 0.5))
        K.plate('arm_t2', kc, None, region=S.Box(kc, (0.75, 0.62, 0.42), rot=(-12, 0, 0), rad=0.03), snap=False,
                bone='shin' + sd, lames=2, lame_step=0.1, sdf=ke, off=0.03, th=0.07, k=0.0, bevel=False, tris=320)
        # spikes (t3) on the pauldrons, elbows and knees
        for off, ln in (((0.25, 1.0, 0.0), 0.8), ((0.75, 0.55, 0.15), 0.6), ((0.45, 0.8, -0.55), 0.55)):
            p, n = K.snap(m * (SH + off), smooth)
            K.spike('arm_t3', p + n * 0.1, n + np.array([s * 0.2, 0.5, -0.1]), ln, bone='upperarm' + sd)
        p, n = K.snap(m * (EL + (0.15, 0.0, -0.45)), smooth)
        K.spike('arm_t3', p, n + [0, -0.3, -0.5], 0.65, bone='forearm' + sd)
        p, n = K.snap(m * (KN + (0, 0.1, 0.45)), smooth)
        K.spike('arm_t3', p, n + [0, 0.4, 0.3], 0.5, bone='shin' + sd)
        # ribcage breastplate (t2): bone ribs curving around the chest from a sternum
        for i in range(4):
            y = 8.85 - i * 0.48
            pts = []
            for a in np.linspace(0.08, 1.0, 6):
                q = np.array([s * a * 1.75, y - a * a * 0.35, 2.6 - a * 1.1])
                p, n = K.snap(q, smooth)
                pts.append(p + n * (0.07 + 0.02 * math.sin(a * 3)))
            K.horn('arm_t2', pts, [0.11, 0.1, 0.09, 0.08, 0.06, 0.035], mat='bone', sides=5, per=2, bone='chest' if i < 3 else 'spine', shade=(0.7, 1.0))
    st = []
    for y in (9.15, 8.65, 8.15, 7.65, 7.3):
        p, n = K.snap((0, y, 2.8), smooth)
        st.append(p + n * 0.08)
    K.horn('arm_t2', st, [0.15, 0.16, 0.15, 0.13, 0.05], mat='bone', sides=7, bone='chest')
    for j, (y, z) in enumerate(((9.85, -0.55), (9.2, -1.25), (8.4, -1.65), (7.6, -1.75))):   # back spines
        p, n = K.snap((0, y, z), smooth)
        K.spike('arm_t3', p, n + [0, 0.35, -0.3], 0.75 - 0.08 * j, bone='chest' if j < 2 else 'spine')

    # ---- Energy: glowing veins, flame crystals, chest brand
    def vein(part, pts, r, bone):
        P, N = K.snap(np.array(pts, float), smooth)
        K.horn(part, list(P + N * 0.02), list(np.linspace(r, r * 0.4, len(P))), mat='accent', sides=5, per=2, bone=bone)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        vein('enr_t1', [m * (1.4, 5.3, 0.55), m * (1.75, 4.85, 0.8), m * (1.6, 4.45, 0.95)], 0.06, 'thigh' + sd)
        vein('enr_t1', [m * (2.0, 5.15, -0.2), m * (2.35, 4.6, 0.0), m * (2.15, 4.1, 0.3)], 0.05, 'thigh' + sd)
        vein('enr_t1', [m * (2.7, 8.6, 1.3), m * (3.1, 8.0, 1.25), m * (3.05, 7.4, 1.2), m * (3.35, 6.9, 0.9)], 0.055, 'upperarm' + sd)
        vein('enr_t1', [m * (2.2, 9.5, 0.3), m * (1.6, 9.9, 0.2), m * (0.9, 10.05, 0.35)], 0.05, 'chest')
        for c, h, bn, tier in (((2.15, 9.75, 0.05), 0.7, 'chest', 2), ((2.85, 9.6, 0.4), 0.55, 'upperarm', 2)):
            p, n = K.snap(m * c, smooth)
            K.crystal(f'enr_t{tier}', p - n * 0.08, n + [0, 0.6, -0.2], h, bone=bn + sd if bn == 'upperarm' else bn, seed=3 + tier)
        K.crystal('enr_t3', m * W_WR + (0, 0.1, -0.1), (s * 0.3, 1, -0.3), 0.6, bone='wing' + sd + '1', seed=9)
    for j, (y, z) in enumerate(((9.65, -0.85), (9.0, -1.45), (8.2, -1.8), (7.35, -1.8), (6.5, -1.6))):
        p, n = K.snap((0, y, z), smooth)
        K.crystal('enr_t2', p - n * 0.1, n + [0, 0.4, -0.3], 0.75 - 0.07 * j, bone='chest' if j < 2 else 'spine', seed=j)
    # chest brand: a burning ring with a horned rune, seared onto the sternum
    c0, n0 = K.snap((0, 8.15, 2.8), smooth)
    x0, y0, z0 = geom.basis(n0, (0, 1, 0))
    ring = [c0 + n0 * 0.06 + (math.cos(a) * x0 + math.sin(a) * z0) * 0.62 for a in np.linspace(0, 2 * math.pi, 25)]
    ring_s, _ = K.snap(np.array(ring), smooth)
    _, nr = K.snap(np.array(ring), smooth)
    ring_s = ring_s + nr * 0.05
    for j in range(0, 24, 4):
        K.horn('enr_t3', list(ring_s[j:j + 5]), [0.06] * 5, mat='accent', sides=5, per=1, bone='chest')
    rune = [[(0, -0.45), (0, 0.35)], [(-0.38, 0.05), (0, 0.35), (0.38, 0.05)], [(-0.28, 0.55), (-0.18, 0.3)], [(0.28, 0.55), (0.18, 0.3)]]
    for stroke in rune:
        Q = np.array([c0 + x0 * a * 0.85 + z0 * b * 0.85 for a, b in stroke])
        Qs, Nq = K.snap(Q, smooth)
        K.horn('enr_t3', list(Qs + Nq * 0.05), [0.055] * len(Qs), mat='accent', sides=5, per=3, bone='chest')
    blade(K, 'enr_t3', tp + td * 0.05, td, (0, 1, 0.2), 1.05, 0.6, 0.15, 'accent', 'tail6', barb=0.2)

    # ---- Speed: calf vanes, tail fins, forearm fins + trailing wing spurs
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        p, n = K.snap(m * (1.72, 2.6, -0.55), smooth)
        K.frill('spd_t1', p, n * [0.4, 0.2, 1] + [0, 0.25, -0.6], (0, 1, 0.3), 0.85, 0.9, rays=4, spread=80, bone='shin' + sd)
        p, n = K.snap(m * (3.7, 5.6, 0.2), smooth)
        K.frill('spd_t3', p, n * [1, 0, 0.4] + [s * 0.2, 0.1, -0.7], (0, 1, 0.2), 0.9, 1.0, rays=4, spread=80, bone='forearm' + sd)
        for k in (1, 2, 3):
            kn, tip = m * FINGERS[k][0], m * FINGERS[k][1]
            d = geom.unit(tip - kn)
            K.horn('spd_t3', [tip - d * 0.15, tip + d * 0.5 + (0, 0, -0.1), tip + d * 0.95 + (0, -0.2, -0.2)], [0.06, 0.04, 0], bone='wing' + sd + '2')
    cat = S.catmull(TAIL, 4)
    for j in range(3):
        q, nxt = cat[6 + j * 5], cat[7 + j * 5]
        rr = TAIL_R[min(2 + j, 6)]
        for s in (-1, 1):
            K.fin('spd_t2', q + (s * rr * 0.6, rr * 0.3, 0), (s, 0.35, 0), q - nxt, 0.55 - j * 0.1, 0.7 - j * 0.12, bone='tail' + str(3 + j))

    # ---- Focus: extra eyes, third eye, floating crown of flame horns
    for s in (-1, 1):
        K.eye('fcs_t1', (s * 0.62, 11.02, 3.12), (s * 0.6, 0.15, 0.8), 0.075, bone='head', lid=False)
        K.eye('fcs_t3', (s * 0.75, 10.65, 2.95), (s * 1.0, -0.1, 0.5), 0.07, bone='head', lid=False)
    p, n = K.snap((0, 11.3, 3.3))
    K.eye('fcs_t2', p - n * 0.02, n + [0, 0.1, 0.5], 0.13, bone='head')
    cc = np.array([0, 12.25, 2.45])
    for j in range(9):
        a = 2 * math.pi * j / 9
        rp = cc + (math.cos(a) * 0.62, 0, math.sin(a) * 0.62)
        h = 0.55 if j % 2 == 0 else 0.38
        out = np.array([math.cos(a), 0, math.sin(a)])
        K.horn('fcs_t3', [rp, rp + out * 0.06 + (0, h * 0.55, 0), rp + out * 0.12 + (0, h, 0)], [0.07, 0.05, 0], mat='accent', sides=5, bone='head')
    ring = [cc + (math.cos(a) * 0.62, -0.02, math.sin(a) * 0.62) for a in np.linspace(0, 2 * math.pi, 19)]
    for j in range(0, 18, 3):
        K.horn('fcs_t3', ring[j:j + 4], [0.045] * 4, mat='accent', sides=5, per=1, bone='head')
