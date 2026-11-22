"""Elder Spawn: secret Wild-only easter egg. A squat, hunched juvenile elder horror (biped plan), ~7.8 units tall,
faces +Z.

Bulbous mottled cephalopod head pushed down and forward so it glares from under a heavy brow shelf; a cold,
asymmetric cluster of small deep-set eyes with horizontal slit pupils and heavy lids; eight writhing face tentacles
of varied length and thickness (two reaching forward, two curling up along the cheeks); a slick near-black green
hide with dark blotches, veins and warts; sinewy arms held out in a grasping pose with long hooked talons; a stubby
curled tail and small ragged bat wings with torn membranes.
"""
import math

import numpy as np

from .. import sdf as S
from .. import geom
from ..builder import POW_SCALE
from ..noise import fbm, voronoi, smoothstep

INFO = dict(name='Elder Spawn', colors=dict(body='#1f3a34', belly='#69705f', accent='#b4e03c'))
DETAIL = 'skin'
UV_SCALE = 0.6
EYE_ACCENT = True
SCALE = 1.0
CAM = dict(az=35, el=9)

Y = np.array([0.0, 1, 0])

# key joints (right side, +X); L mirrored
SH = np.array([1.18, 3.9, 0.2])      # shoulder (hunched forward)
EL = np.array([1.88, 3.18, 0.32])    # elbow out and down
WR = np.array([2.02, 3.22, 1.22])    # wrist forward
HC = np.array([2.05, 3.27, 1.52])    # palm centre (palm down, fingers forward)
HIP = np.array([0.8, 2.0, -0.12])
KN = np.array([1.0, 1.12, 0.22])
AN = np.array([1.0, 0.45, 0.12])

HD = np.array([0.0, -0.14, 0.26])    # head pushed down and forward (hunched, glaring from under the brow)


def H(p):
    return np.asarray(p, float) + HD


HEAD_C = H([0.0, 5.6, 0.7])   # inside the head, used to ray-march the eye sites

TAIL = [(0, 2.0, -1.05), (0, 1.42, -1.8), (0, 1.0, -2.35), (0, 0.82, -2.9), (0.22, 0.95, -3.3), (0.42, 1.3, -3.42)]
TAIL_R = [0.58, 0.42, 0.28, 0.17, 0.1, 0.04]

# eye cluster: (target on the face, radius). Deliberately mismatched and asymmetric.
EYES = [((0.58, 5.66, 2.0), 0.2),     # right main
        ((-0.62, 5.6, 1.98), 0.165),  # left main, lower and smaller
        ((-1.1, 5.95, 1.4), 0.11),    # left outer
        ((0.3, 6.5, 1.85), 0.085),    # above the right brow
        ((1.12, 5.33, 1.5), 0.075)]   # right cheek

# face tentacles: (angle deg around muzzle, length, base radius, main direction, wiggle amp, freq, phase, tip curl)
TENTS = [(-64, 1.4, 0.16, (-1.0, 0.55, 0.35), 0.22, 0.9, 0.0, (0.3, 1.0, 0.2)),     # curls up the left cheek
         (-42, 2.2, 0.25, (-0.32, -1.0, 0.22), 0.34, 0.8, 1.2, (-0.4, 0.7, 0.6)),
         (-20, 2.45, 0.28, (-0.25, -0.35, 1.0), 0.32, 0.7, 2.3, (0.2, 1.0, -0.3)),   # reaches forward
         (-6, 2.3, 0.2, (-0.12, -1.0, 0.28), 0.36, 1.0, 0.7, (0.8, 0.2, 0.7)),
         (8, 2.6, 0.3, (0.2, -0.3, 1.0), 0.3, 0.75, 3.6, (-0.3, 1.0, -0.2)),       # big forward reacher
         (22, 2.0, 0.19, (0.3, -1.0, 0.25), 0.36, 1.1, 4.4, (-0.8, 0.3, 0.6)),
         (42, 2.25, 0.25, (0.34, -1.0, 0.25), 0.34, 0.85, 5.3, (0.4, 0.6, 0.7)),
         (66, 1.25, 0.15, (1.0, 0.4, 0.4), 0.22, 1.0, 2.0, (-0.2, 1.0, -0.3))]     # curls up the right cheek


def M(s):
    return np.array([s, 1.0, 1.0])


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def taper(n, r0, r1, p=0.85):
    t = np.linspace(0, 1, n)
    return np.array(r1 + (r0 - r1) * (1 - t) ** p)


def torso_sdf(B):
    """Body masses shared by the main SDF and the tentacle clearance SDF."""
    B.ell((0, 2.65, 0.3), (1.42, 1.38, 1.38), k=0.0)                                      # pot belly
    B.ell((0, 2.0, -0.25), (1.28, 0.95, 1.12), k=0.5)                                     # hips / rump
    B.ell((0, 3.75, 0.05), (1.25, 0.95, 1.08), k=0.55)                                    # chest
    B.ell((0, 4.35, -0.62), (1.1, 0.82, 0.85), k=0.6)                                     # hunched upper back
    B.ell(SH + (0, 0.05, -0.08), (0.55, 0.5, 0.52), k=0.35, mirror=True)                  # shoulders
    B.tube([SH, (1.58, 3.55, 0.22), EL], [0.5, 0.43, 0.36], k=0.3, mirror=True)           # upper arm
    B.tube([EL, (2.0, 3.12, 0.78), WR], [0.36, 0.32, 0.26], k=0.18, mirror=True)          # sinewy forearm
    B.ell(HC, (0.3, 0.16, 0.3), k=0.15, mirror=True)                                      # palm (down)
    B.tube([HIP, (0.95, 1.58, 0.05), KN], [0.78, 0.7, 0.6], k=0.35, mirror=True)
    B.tube([KN, (1.02, 0.8, 0.18), AN], [0.6, 0.55, 0.5], k=0.25, mirror=True)


def head_sdf(B):
    B.tube([(0, 4.0, 0.05), H((0, 4.7, 0.6))], [1.0, 0.95], k=0.45)                         # short neck, leaning forward
    B.ell(H((0, 6.35, -0.2)), (1.55, 1.28, 1.7), rot=(-6, 0, 0), k=0.0)                     # cranium dome
    B.ell(H((0, 5.55, 1.0)), (1.42, 1.05, 1.12), k=0.55)                                     # face
    B.ell(H((0, 4.85, 1.42)), (1.05, 0.6, 0.75), k=0.4)                                      # muzzle
    B.ell(H((1.0, 5.25, 0.85)), (0.55, 0.55, 0.6), k=0.4, mirror=True)                       # cheeks


def tentacle(k, a_deg, L, r0, d, amp, freq, ph, curl, clear):
    """Control points for one writhing tentacle, pushed clear of the torso/face."""
    a = math.radians(a_deg)
    base = np.array([0.86 * math.sin(a), 4.72 + (0.1 if k % 2 else -0.04) - 0.1 * abs(math.sin(a)),
                     1.42 + 0.66 * math.cos(a)]) + HD
    o = np.array([math.sin(a), 0, math.cos(a)])
    d = unit(np.asarray(d, float))
    d0 = unit(o * 0.5 + np.array([0, -0.7, 0.45]))
    p1 = unit(np.cross(d, Y if abs(d[1]) < 0.9 else np.array([0, 0, 1.0])))
    p2 = np.cross(d, p1)
    n = 7
    ts = np.linspace(0, 1, n)
    pts = [base]
    for i in range(1, n):
        w = smoothstep(0.0, 0.4, ts[i - 1])
        pts.append(pts[-1] + unit(d0 * (1 - w) + d * w) * L / (n - 1))
    pts = np.array(pts)
    for i, t in enumerate(ts):
        pts[i] += amp * t * (math.sin(math.pi * 2 * freq * t + ph) * p1 + 0.6 * math.cos(math.pi * 2 * freq * 1.3 * t + ph) * p2)
        if t > 0.6:
            pts[i] += np.asarray(curl, float) * L * 0.26 * ((t - 0.6) / 0.4) ** 2
    R = taper(n, r0, 0.035, 0.9)
    # push the free part clear of the chest / face (keeps the rest pose readable)
    for _ in range(8):
        P = pts[2:]
        dist = clear.at(P)
        need = R[2:] + 0.14 - dist
        if not (need > 0).any():
            break
        g = clear.grad(P)
        pts[2:] += g * np.clip(need, 0, None)[:, None]
    return pts, R


def resample(pts, R, per=5):
    C = S.catmull(pts, per)
    Rr = np.clip(S.catmull(R, per)[:, 0], 0.03, None)
    T = unit(np.gradient(C, axis=0))
    A = np.gradient(T, axis=0)
    U = A - T * (A * T).sum(1, keepdims=True)
    fb = -Y - T * (-T[:, 1:2])
    weak = np.linalg.norm(U, axis=1) < 1e-3
    U[weak] = fb[weak]
    U = unit(U)
    for _ in range(3):   # smooth the concave-side frame so suckers do not flip around
        U[1:-1] = unit(U[:-2] + U[1:-1] * 2 + U[2:])
        U = unit(U - T * (U * T).sum(1, keepdims=True))
    return C, Rr, U, T


def sucker(K, c, u, t, rs, h, sides=6, bone=None):
    """Cup: outer wall, rolled rim, recessed centre (belly material, darker inside)."""
    u = unit(u)
    t = unit(t - u * (t @ u))
    sd = np.cross(t, u)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rings = [(c - u * h * 0.4, rs), (c + u * h, rs * 0.86), (c + u * h * 0.85, rs * 0.48)]
    V, col = [], []
    for (o, rr), cv in zip(rings, (0.75, 0.95, 0.6)):
        for a in ang:
            V.append(o + (np.cos(a) * sd + np.sin(a) * t) * rr)
            col.append(cv)
    V.append(c + u * h * 0.25)
    col.append(0.35)
    V = np.array(V)
    T, ctr = [], len(V) - 1
    for b in range(2):
        for j in range(sides):
            j2 = (j + 1) % sides
            a0, a1, b0, b1 = b * sides + j, b * sides + j2, (b + 1) * sides + j, (b + 1) * sides + j2
            T += [(a0, a1, b1), (a0, b1, b0)]
    for j in range(sides):
        T.append((2 * sides + j, 2 * sides + (j + 1) % sides, ctr))
    T = np.array(T)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    fc = V[T].mean(1) - c
    rad = fc - np.outer(fc @ u, u)
    exp = np.where(np.arange(len(T))[:, None] < 2 * sides, rad, u[None] + 0 * rad)
    exp[2 * sides:] = u
    flip = (fn * exp).sum(1) < 0
    T[flip] = T[flip][:, ::-1]
    K.piece('body', V, T, 'belly', col=np.array(col), bone=bone)


def elder_eye(K, part, c, g, r, bone='head', hood=False):
    """Small glowing eyeball, horizontal slit pupil, dark socket, heavy hooded upper lid (no catch-light)."""
    c, g = np.asarray(c, float), geom.unit(g)
    x, y, z = geom.basis(g, (0, 1, 0))     # y = gaze, z ~ up, x = side
    V, T, _ = geom.sphere((0, 0, 0), 1, 8, 5)
    Vs = c - g * r * 0.3 + (V[:, [0]] * x * 1.25 + V[:, [1]] * g * 0.8 + V[:, [2]] * z * 1.2) * r
    K.piece(part, Vs, T, 'dark', bone=bone)
    K.ball(part, c, r, 'eye', seg=10, rings=6, bone=bone)
    Vp = c + g * r * 0.74 + (V[:, [0]] * x * 0.8 + V[:, [1]] * g * 0.3 + V[:, [2]] * z * 0.4) * r
    K.piece(part, Vp, T, 'dark', bone=bone)
    if hood:
        Vh = c + z * r * 0.88 + g * r * 0.04 + (V[:, [0]] * x * 1.2 + V[:, [1]] * g * 1.06 + V[:, [2]] * z * 0.5) * r
        K.piece(part, Vh, T, 'body', bone=bone)


class Snaps:
    """Collect snap queries, project them all in ONE K.snap call."""
    def __init__(self):
        self.q, self.k = [], {}

    def add(self, name, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        self.k[name] = (sum(len(a) for a in self.q), len(pts))
        self.q.append(pts)

    def run(self, K):
        self.P, self.N = K.snap(np.concatenate(self.q))

    def __getitem__(self, name):
        i, n = self.k[name]
        return self.P[i:i + n], self.N[i:i + n]


def wing_geo(s):
    root = M(s) * (0.55, 4.45, -0.98)
    W = M(s) * (1.1, 5.05, -1.6)
    F = [M(s) * (1.12, 5.95, -1.92), M(s) * (1.85, 5.62, -2.02), M(s) * (2.18, 4.88, -1.85)]
    low = M(s) * (0.85, 3.88, -1.12)
    return root, W, F, low


# torn trailing edges: (u centre, depth toward the wrist, half width) per membrane panel
NOTCH = {1: [(0.35, 0.3, 0.09), (0.72, 0.18, 0.06)],
         2: [(0.28, 0.22, 0.07), (0.62, 0.38, 0.08)],
         3: [(0.5, 0.33, 0.1)]}


def build(K):
    K.mat('body', rough=0.3, detail_strength=0.24)
    K.mat('belly', rough=0.38, detail_strength=0.14)
    K.mat('plate', color='#8b8672', rough=0.4, metal=0.05)
    K.mat('horn', color='#b9ad8c', rough=0.3, vcol=True)
    K.mat('membrane', color='#262822', rough=0.45, double=True, vcol=True)
    K.mat('coral', color='#5a4a5c', rough=0.45, vcol=True)
    K.mat('eye', rough=0.1, emit_strength=2.4)
    B = K.sdf()
    belly = K.sdf()
    clear = K.sdf()

    # ---- body and head masses
    torso_sdf(B)
    torso_sdf(clear)
    head_sdf(B)
    head_sdf(clear)
    # ---- heavy brow shelf the eyes stare out from under, cheekbone ridges, wrinkled snout
    B.tube([H(q) for q in ((0.12, 6.02, 2.05), (0.55, 6.13, 2.03), (0.98, 5.98, 1.72), (1.3, 5.7, 1.28))], [0.21, 0.25, 0.2, 0.1],
           k=0.2, mirror=True)
    B.ell(H((0, 5.96, 2.04)), (0.28, 0.2, 0.2), k=0.15)
    B.tube([H(q) for q in ((0.35, 5.38, 2.0), (0.8, 5.33, 1.85), (1.15, 5.45, 1.45))], [0.08, 0.1, 0.06], k=0.15, mirror=True)
    # ---- eye sites: ray-march the face, then cut deep sockets
    eyes = []
    for tgt, r in EYES:
        dvec = unit(H(tgt) - HEAD_C)
        ts = np.arange(0.5, 3.5, 0.01)
        P = HEAD_C + ts[:, None] * dvec
        dd = B.at(P)
        i = int(np.argmax(dd > 0))
        sp = P[i]
        nrm = B.grad(sp)[0]
        eyes.append((sp, nrm, r))
    for sp, nrm, r in eyes:
        B.ell(sp - nrm * r * 0.2, (r * 1.32,) * 3, cut=True, k=0.08)
    B.ell(H((0, 4.72, 1.85)), (0.45, 0.25, 0.4), cut=True, k=0.12)                          # mouth hollow under the tentacles
    # ---- grasping hands: three hooked fingers + a thumb, curled down
    fingers = []
    for s in (-1, 1):
        fl = []
        for j in (-1, 0, 1):
            d = unit(M(s) * (0.35 * j + 0.1, -0.12, 1.0))
            b = M(s) * (HC + (0.14 * j, 0.0, 0.16))
            pts = [b, b + d * 0.32 + (0, 0.03, 0), b + d * 0.55 - Y * 0.14]
            fl.append(pts)
        d = unit(M(s) * (-0.6, -0.3, 0.6))
        b = M(s) * (HC + (-0.2, -0.04, 0.05))
        fl.append([b, b + d * 0.24, b + d * 0.4 - Y * 0.08])
        fingers.append(fl)
        if s == 1:
            for q, pts in enumerate(fl):
                B.tube(pts, [0.11, 0.09, 0.07] if q < 3 else [0.1, 0.085, 0.065], k=0.07, mirror=True)
    # ---- wide webbed feet
    B.ell((1.0, 0.25, 0.42), (0.58, 0.25, 0.72), k=0.3, mirror=True)
    toes = []
    for j in (-1, 0, 1):
        b = np.array([1.0 + 0.24 * j, 0.22, 0.78])
        tp = np.array([1.0 + 0.5 * j, 0.14, 1.42 - 0.08 * abs(j)])
        toes.append((b, tp))
        B.tube([b, tp], [0.17, 0.12], k=0.1, mirror=True)
    B.ell((1.0, 0.1, 1.05), (0.58, 0.07, 0.34), k=0.12, mirror=True)
    # ---- curled stubby tail
    B.tube(TAIL, TAIL_R, k=0.45)

    # ---- eight writhing face tentacles (low blend so they read as separate limbs)
    sampled = []
    for k, t in enumerate(TENTS):
        pts, R = tentacle(k, *t, clear)
        B.tube(pts, R, k=0.05)
        C, Rr, U, T = resample(pts, R)
        sampled.append((C, Rr, U, T, pts))

    # ---- dull belly, throat, soles
    belly.ell((0, 2.55, 1.05), (0.92, 1.12, 0.95), k=0.0)
    belly.ell((0, 3.55, 0.95), (0.7, 0.5, 0.6), k=0.0)
    for s in (-1, 1):
        belly.ell(M(s) * (1.0, 0.0, 0.7), (0.75, 0.12, 1.1), k=0.0)

    # ---- sculpt: slick skin, warts on cranium / shoulders / hump, raised veins, belly kept smooth
    def skin(x, y, z):
        P = np.stack([x, y, z], 1)
        w = smoothstep(0.1, -0.1, belly.at(P))
        f1, _, _ = voronoi(x, y, z, size=0.32, seed=5)
        wart = -0.06 * (1 - smoothstep(0.0, 0.17, f1))
        crown = smoothstep(5.9, 6.8, y) * (1 - smoothstep(1.2, 1.9, z))
        sh = np.exp(-((np.abs(x) - 1.15) ** 2 + (y - 4.15) ** 2 + (z + 0.1) ** 2) / 0.35)
        hump = np.exp(-(x ** 2 + (y - 4.3) ** 2 + (z + 1.1) ** 2) / 0.5)
        m = np.clip(crown + sh + hump, 0, 1)
        v = fbm(x, y, z, 3, freq=1.1, seed=11)
        vein = -0.03 * (1 - smoothstep(0.0, 0.05, np.abs(v)))
        vm = np.clip(smoothstep(5.2, 6.4, y) + sh, 0, 1)
        swell = -0.03 * fbm(x, y, z, 2, freq=0.9, seed=7)
        pleat = -0.022 * (0.5 + 0.5 * np.sin(y * 10.5 + 0.8 * fbm(x, y, z, 2, freq=0.7, seed=3))) * smoothstep(0.9, 0.0, np.abs(x))
        return (wart * m + vein * vm + swell) * (1 - w) + pleat * w
    B.disp(skin, band=0.14)

    def tint(P, N):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        blotch = 1.0 - 0.62 * smoothstep(-0.04, 0.12, fbm(x, y, z, 3, freq=1.15, seed=13))
        f2, _, _ = voronoi(x, y, z, size=0.32, seed=5)
        wart = 1.0 - 0.25 * (1 - smoothstep(0.0, 0.15, f2)) * smoothstep(5.9, 6.8, y)
        mott = 1.0 + 0.2 * fbm(x, y, z, 3, freq=0.8, seed=4)
        v = fbm(x, y, z, 3, freq=1.1, seed=11)
        vein = 1.0 - 0.35 * (1 - smoothstep(0.0, 0.05, np.abs(v))) * smoothstep(5.2, 6.4, y)
        dorsal = 1.0 - 0.4 * np.clip(-N[:, 2] * 0.7 + N[:, 1] * 0.5, 0, 1) * smoothstep(3.0, 7.0, y)
        return np.clip(blotch * wart * mott * vein * dorsal, 0.25, 1.0)

    K.skin(B, voxel=0.05, tris=17000, belly=belly, tint=tint)

    # ---- every surface query in ONE snap call
    SN = Snaps()
    cand = []
    for k, (C, R, U, T, _) in enumerate(sampled):
        Lc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        idx = np.linspace(0, len(C) - 1, 7).round().astype(int)
        pos, row = Lc[-1] * 0.32, 0
        while pos < Lc[-1] * 0.92:
            i = min(int(np.searchsorted(Lc, pos)), len(C) - 1)
            r = R[i]
            if r < 0.075:
                break
            side = np.cross(T[i], U[i])
            bn = f'tent{k + 1}_{int(np.clip(np.searchsorted(idx, i), 1, 6))}'
            cand.append((C[i] + unit(U[i] + side * 0.45 * (1 if row % 2 else -1)) * r, T[i], r, bn))
            row += 1
            pos += max(r * 1.15, 0.12)
    SN.add('suck', [c[0] for c in cand])
    dome = [(s * x, y, z) for s in (-1, 1) for (x, y, z) in ((1.45, 6.3, 0.2), (1.5, 6.6, -0.5), (1.35, 6.4, -1.2),
                                                            (1.15, 7.0, 0.2), (1.2, 7.1, -0.6), (0.95, 5.9, -1.4))]
    SN.add('dome', [H(q) for q in dome])
    SN.add('bspot', [(s * 1.25, y, z) for s in (-1, 1) for (y, z) in ((2.3, 0.65), (2.95, 0.7), (3.5, 0.45))])
    SN.add('sh', [M(s) * (SH + d) for s in (-1, 1) for d in ((0.1, 0.45, 0.05), (0.38, 0.3, 0.3), (0.35, 0.3, -0.3),
                                                               (0.55, 0.05, 0.05), (-0.15, 0.45, -0.35), (0.25, 0.42, -0.08))])
    SN.add('coral', [M(s) * p for s in (-1, 1) for p in ((1.25, 4.4, -0.15), (1.1, 6.85, -1.0), (0.55, 7.5, -1.05))])
    SN.add('fcs1', [H((-0.55, 6.75, 1.3)), H((0.8, 6.85, 1.0))])
    SN.add('fcs3', [H(q) for q in ((0.75, 7.25, 0.45), (-1.2, 6.85, 0.3), (0.05, 7.55, -0.05))])
    SN.add('bud', [M(s) * (SH + (0.0, 0.5, 0.25)) for s in (-1, 1)])
    SN.add('farm', [M(s) * ((EL + WR) * 0.5 + (0.3, 0.05, 0.0)) for s in (-1, 1)])
    SN.add('tailtop', [(0, 1.9, -1.55), (0, 1.3, -2.2)])
    SN.add('tailside', [(s * 0.2, 0.95, -2.55) for s in (-1, 1)])
    SN.add('ear', [H((s * 1.5, 6.0, 0.15)) for s in (-1, 1)])
    ring_b = [(0.55 * math.sin(a), 2.75 + 0.55 * math.cos(a), 1.1) for a in np.linspace(0, 2 * np.pi, 19)]
    rune_b = [(0, 3.15, 1.1), (0, 2.75, 1.1), (-0.3, 2.45, 1.1), (0, 2.75, 1.1), (0.3, 2.45, 1.1)]
    ring_h = [(0.3 * math.sin(a), 7.2 + 0.26 * math.cos(a), 0.55) for a in np.linspace(0, 2 * np.pi, 15)]
    SN.add('ringb', ring_b)
    SN.add('runeb', rune_b)
    SN.add('ringh', [H(q) for q in ring_h])
    SN.add('runeh', [H((0, 7.4, 0.55)), H((0, 7.0, 0.55))])
    SN.run(K)

    # ---- sucker cups on the concave side of each tentacle
    SP, SNn = SN['suck']
    suckers = []
    for (q, t, r, bn), p, n in zip(cand, SP, SNn):
        if p[1] > 0.15 and np.linalg.norm(p - q) < r * 1.2:
            sucker(K, p - n * r * 0.05, n, t, r * 0.34, r * 0.2, sides=6, bone=bn)
            suckers.append((p, n, r, bn))

    # ---- skeleton
    K.bone('pelvis', (0, 1.95, -0.25), (0, 2.85, -0.05), None, 1.5)
    K.bone('spine', (0, 2.85, -0.05), (0, 3.6, 0.0), 'pelvis', 1.55)
    K.bone('chest', (0, 3.6, 0.0), (0, 4.25, 0.1), 'spine', 1.3)
    K.bone('neck', (0, 4.25, 0.1), H((0, 4.85, 0.6)), 'chest', 1.0)
    K.bone('head', H((0, 4.85, 0.6)), H((0, 7.0, -0.1)), 'neck', 1.75)
    K.bone('jaw', H((0, 4.85, 1.35)), H((0, 4.55, 1.95)), 'head', 0.4)
    for k, (C, R, U, T, pts) in enumerate(sampled):
        K.chain(f'tent{k + 1}_', pts, 'head', (R[1] * 1.2, R[-2] * 1.4))
    K.chain('tail', TAIL, 'pelvis', (0.6, 0.12))
    wings = {}
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        K.bone('thigh' + sd, m * HIP, m * KN, 'pelvis', 0.78)
        K.bone('shin' + sd, m * KN, m * AN, 'thigh' + sd, 0.6)
        K.bone('foot' + sd, m * AN, m * (1.0, 0.15, 1.3), 'shin' + sd, 0.5)
        K.bone('upperarm' + sd, m * SH, m * EL, 'chest', 0.5)
        K.bone('forearm' + sd, m * EL, m * WR, 'upperarm' + sd, 0.36)
        K.bone('hand' + sd, m * WR, m * (HC + (0, -0.05, 0.6)), 'forearm' + sd, 0.34)
        K.mount('hand' + sd, m * (HC + (0, -0.1, 0.3)), 'hand' + sd)
        K.mount('shoulder' + sd, m * (SH + (0.1, 0.55, -0.05)), 'chest')
        root, W, F, low = wing_geo(s)
        K.bone('wingF' + sd, root, (W + F[1]) * 0.5, 'chest', 0.4)
        wings[sd] = (root, W, F, low)
    K.mount('head', H((0, 6.0, 0.6)), 'head')
    K.mount('jaw', H((0, 4.8, 1.6)), 'jaw')
    K.mount('mouth', H((0, 4.55, 2.15)), 'jaw')
    K.mount('chest', (0, 3.5, 1.35), 'chest')
    K.mount('back', (0, 4.2, -1.35), 'spine')
    K.mount('tailTip', TAIL[-1], 'tail5')
    K.mount('crown', H((0, 8.2, -0.15)), 'head')

    # ---- the eye cluster: cold, hooded, staring slightly down
    for sp, nrm, r in eyes:
        g = unit(nrm * 0.45 + np.array([0, -0.22, 1.0]))
        elder_eye(K, 'body', sp - nrm * r * 0.7, g, r)
    # faint sickly glow deep in the mouth between the tentacle roots
    K.ball('body', H((0, 4.62, 1.92)), (0.22, 0.14, 0.12), mat='accent', seg=8, rings=5, bone='jaw')

    # ---- small ragged bat wings: knuckled finger bones, torn notched membrane
    for sd, (root, W, F, low) in wings.items():
        bn = 'wingF' + sd
        nw = unit(np.cross(F[0] - W, F[2] - W))
        if nw[2] > 0:
            nw = -nw
        K.horn('body', [root, (root + W) * 0.5 + nw * 0.05, W], [0.12, 0.09, 0.075], mat='body', sides=6, bone=bn)
        K.ball('body', W, (0.11,) * 3, mat='body', seg=6, rings=4, bone=bn)
        for f in F:
            mid = W + (f - W) * 0.55 + nw * 0.04
            K.horn('body', [W, mid, f], [0.07, 0.045, 0.0], mat='body', sides=5, per=3, bone=bn)
            K.ball('body', mid, (0.06,) * 3, mat='body', seg=5, rings=3, bone=bn)
        K.claw('body', W + (W - root) * 0.05, unit(W - root) + Y * 0.4, 0.26, 0.05, curl=-nw, mat='horn', bone=bn)
        edges = [root, low, F[2], F[1], F[0]]
        for i in range(len(edges) - 1):
            A, Bp = edges[i], edges[i + 1]
            sc = 0.0 if i == 0 else 0.26
            cuts = NOTCH.get(i, [])

            def fn(u, v, A=A, Bp=Bp, sc=sc, cuts=cuts):
                E = A + (Bp - A) * u
                cut = sc * math.sin(math.pi * u) + sum(dp * max(0.0, 1 - abs(u - uc) / hw) for uc, dp, hw in cuts)
                E = E + (W - E) * min(cut, 0.8)
                return W + (E - W) * v + nw * 0.07 * math.sin(math.pi * u) * v
            V, T, uu = geom.grid(fn, 12 if cuts else 5, 3)
            K.piece('body', V, T, 'membrane', col=0.55 + 0.45 * uu, bone=bn, smooth=True)

    # ---- Might: long hooked hand talons + foot claws (all tiers), tentacle-tip hooks (t1+), beak (t2+), brow spikes (t3)
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        k = sc ** 0.8
        for si, (s, sd) in enumerate(((-1, 'L'), (1, 'R'))):
            for q, pts in enumerate(fingers[si]):
                a, b = np.asarray(pts[-2]), np.asarray(pts[-1])
                d = unit(b - a)
                ln = (0.46 if q < 3 else 0.3) * k
                K.claw(part, b + d * 0.02, unit(d - Y * 0.15), ln, 0.068 * (0.9 + 0.1 * sc), curl=(0, -1, 0.15), bone='hand' + sd)
            for b, tp in toes:
                tp = M(s) * tp
                d = unit(np.asarray(tp - M(s) * b) * [1, 0, 1] - Y * 0.15)
                K.claw(part, tp + d * 0.05 + [0, 0.02, 0], d, 0.32 * k, 0.085 * (0.85 + 0.15 * sc), curl=(0, -1, 0), bone='foot' + sd)
        if t >= 1:
            order = np.argsort([-tt[1] for tt in TENTS])
            for kk, (C, R, U, T, _) in enumerate(sampled):
                nh = 1 if t == 1 else (2 if kk in order[:4] or t == 3 else 1)
                for j in range(nh):
                    i = int(len(C) * (0.86 - j * 0.12))
                    side = np.cross(T[i], U[i]) * (1 if j % 2 else -1)
                    out = -U[i]   # outer (convex) side
                    K.claw(part, C[i] + out * R[i] * 0.8 + side * R[i] * 0.2, out + side * 0.3, R[i] * (1.0 + 0.4 * sc),
                           R[i] * 0.32, curl=T[i], bone=f'tent{kk + 1}_{6 if j == 0 else 5}')
        if t >= 2:
            b = 0.8 + 0.2 * sc
            K.horn(part, [H(q) for q in ((0, 4.5, 1.95), (0, 4.42, 2.25 + 0.1 * b), (0, 4.18, 2.4 + 0.15 * b), (0, 3.92, 2.32 + 0.18 * b))],
                   [0.2 * b, 0.17 * b, 0.09 * b, 0], mat='dark', sides=8, flat=0.85, bone='jaw')
            K.horn(part, [H(q) for q in ((0, 4.12, 1.98), (0, 4.04, 2.2 + 0.06 * b), (0, 4.12, 2.33 + 0.1 * b))], [0.14 * b, 0.11 * b, 0],
                   mat='dark', sides=8, flat=0.85, bone='jaw')
        if t >= 3:
            for s in (-1, 1):
                K.horn(part, [H(q) for q in ((s * 0.95, 6.02, 1.78), (s * 1.18, 6.32, 1.62), (s * 1.38, 6.55, 1.3))],
                       [0.1, 0.065, 0], bone='head', per=3, sides=6)
                K.horn(part, [H(q) for q in ((s * 0.45, 6.15, 2.0), (s * 0.55, 6.45, 1.95), (s * 0.62, 6.68, 1.75))],
                       [0.07, 0.045, 0], bone='head', per=2, sides=5)

    # ---- Armour: barnacle shoulder shells (t1), cranium ridge plates (t2), dark coral spikes (t3)
    P, N = SN['sh']
    for i, (p, n) in enumerate(zip(P, N)):
        sd = 'L' if i < 6 else 'R'
        hgt = [0.26, 0.2, 0.17, 0.16, 0.13, 0.12][i % 6]
        r0 = [0.19, 0.15, 0.14, 0.13, 0.11, 0.1][i % 6]
        base = p - n * 0.02
        K.horn('arm_t1', [base - n * 0.08, base + n * hgt * 0.55, base + n * hgt], [r0 * 1.1, r0 * 0.8, r0 * 0.5], mat='plate',
               sides=7, per=1, bone='upperarm' + sd)
        K.ball('arm_t1', base + n * (hgt + 0.005), (r0 * 0.36,) * 3, mat='dark', seg=7, rings=4, bone='upperarm' + sd)
    for c in ((0, 7.58, 0.55), (0, 7.68, -0.1), (0, 7.55, -0.75), (0, 7.15, -1.35)):
        K.plate('arm_t2', H(c), (0.34, 0.3, 0.3), bone='head', off=0.04, th=0.08, tris=120)
    for s in (-1, 1):
        for c in ((s * 0.85, 7.3, 0.2), (s * 0.9, 7.25, -0.7)):
            K.plate('arm_t2', H(c), (0.26, 0.24, 0.26), bone='head', off=0.04, th=0.07, tris=90)
    P, N = SN['coral']
    for i, (p, n) in enumerate(zip(P, N)):
        s = -1 if i < 3 else 1
        bn = 'upperarm' + ('L' if s < 0 else 'R') if i % 3 == 0 else 'head'
        up = unit(n + Y * 0.6 + [s * 0.2, 0, -0.2])
        h = 0.65 if i % 3 == 0 else 0.5
        side = unit(np.cross(up, [0, 0, 1]))
        tip = p + up * h
        K.horn('arm_t3', [p - n * 0.1, p + up * h * 0.5, tip], [0.1, 0.07, 0.025], mat='coral', sides=6, bone=bn)
        for sg in (-1, 1):
            q = p + up * h * (0.45 if sg < 0 else 0.62)
            K.horn('arm_t3', [q, q + unit(up + side * sg * 0.9) * h * 0.22, q + unit(up * 1.2 + side * sg * 0.7) * h * 0.42],
                   [0.055, 0.04, 0.015], mat='coral', sides=5, bone=bn)

    # ---- Energy: cranium photophores (t1), belly sigil + flank spots (t2), glowing suckers + brow sigil (t3)
    P, N = SN['dome']
    for i, (p, n) in enumerate(zip(P, N)):
        K.ball('enr_t1', p + n * 0.02, (0.1 - (i % 3) * 0.015,) * 3, mat='accent', seg=8, rings=5, bone='head')
    P, N = SN['ringb']
    K.horn('enr_t2', P + N * 0.03, [0.045] * len(P), mat='accent', sides=5, per=2, bone='spine')
    P, N = SN['runeb']
    K.horn('enr_t2', P[:3] + N[:3] * 0.03, [0.05, 0.05, 0.0], mat='accent', sides=5, per=2, bone='spine')
    K.horn('enr_t2', P[3:] + N[3:] * 0.03, [0.05, 0.0], mat='accent', sides=5, per=2, bone='spine')
    P, N = SN['bspot']
    for i, (p, n) in enumerate(zip(P, N)):
        K.ball('enr_t2', p + n * 0.02, (0.09,) * 3, mat='accent', seg=8, rings=5, bone='spine')
    for p, n, r, bn in suckers:
        K.ball('enr_t3', p + n * r * 0.12, (r * 0.14,) * 3, mat='accent', seg=5, rings=3, bone=bn)
    P, N = SN['ringh']
    K.horn('enr_t3', P + N * 0.03, [0.04] * len(P), mat='accent', sides=5, per=2, bone='head')
    P, N = SN['runeh']
    K.horn('enr_t3', P + N * 0.03, [0.045, 0.045], mat='accent', sides=5, per=2, bone='head')

    # ---- Speed: forearm fin frills (t1), tail frills (t2), long wing-finger spurs + ear fins (t3)
    P, N = SN['farm']
    for s, sd, p, n in zip((-1, 1), ('L', 'R'), P, N):
        fa = unit(M(s) * (WR - EL))
        K.frill('spd_t1', p, unit(n + np.array([s * 0.2, 0.35, 0])), -fa, 0.55, 0.6, rays=4, spread=70, bone='forearm' + sd)
    P, N = SN['tailtop']
    K.frill('spd_t2', P[0], N[0] + [0, 0.4, -0.3], (0, -0.6, -1), 0.6, 0.85, rays=5, spread=80, bone='tail1')
    K.frill('spd_t2', P[1], N[1] + [0, 0.3, -0.6], (0, -0.2, -1), 0.45, 0.6, rays=4, spread=70, bone='tail2')
    P, N = SN['tailside']
    for s, p, n in zip((-1, 1), P, N):
        K.fin('spd_t2', p, n + [s * 0.4, 0, -0.4], (0, 0, -1), 0.45, 0.45, bone='tail3')
    for sd, (root, W, F, low) in wings.items():
        bn = 'wingF' + sd
        nw = unit(np.cross(F[0] - W, F[2] - W))
        for f in F:
            d = unit(f - W)
            K.horn('spd_t3', [f - d * 0.15, f + d * 0.3 + nw * 0.03, f + d * 0.6 - Y * 0.05], [0.05, 0.035, 0.0], mat='horn', sides=5, bone=bn)
        for i in range(2):
            a0, a1 = F[i + 1], F[i]
            t0 = a0 + unit(a0 - W) * 0.55
            t1 = a1 + unit(a1 - W) * 0.55
            V = np.array([a0, a1, t1, (t0 + t1) * 0.5 + ((a0 + a1) * 0.5 - (t0 + t1) * 0.5) * 0.45, t0])
            T = np.array([(0, 1, 3), (1, 2, 3), (0, 3, 4)])
            K.piece('spd_t3', V, T, 'membrane', col=np.array([0.6, 0.6, 1, 0.8, 1]), bone=bn, smooth=False)
    P, N = SN['ear']
    for s, p, n in zip((-1, 1), P, N):
        K.frill('spd_t3', p, n * [1, 0.2, 0.2] + [s * 0.3, 0.25, -0.5], (0, 0.3, -1), 0.75, 0.8, rays=5, spread=90, bone='head')

    # ---- Focus: two more forehead eyes (t1), floating thorn halo (t2), cranium eyes + shoulder eye buds (t3)
    P, N = SN['fcs1']
    for i, (p, n) in enumerate(zip(P, N)):
        r = 0.12 if i == 0 else 0.095
        elder_eye(K, 'fcs_t1', p - n * r * 0.35, unit(n + np.array([0, -0.1, 0.6])), r)
    cc = H([0, 8.25, -0.15])
    ring = [cc + np.array([0.82 * math.sin(a), 0.12 * math.cos(a), 0.82 * math.cos(a)]) for a in np.linspace(0, 2 * np.pi, 25)]
    K.horn('fcs_t2', ring, [0.055] * len(ring), mat='accent', sides=6, per=1, bone='head')
    for j in range(7):
        a = 2 * math.pi * j / 7
        d = np.array([math.sin(a), 0.0, math.cos(a)])
        b = cc + d * 0.82 + Y * 0.12 * math.cos(a)
        K.crystal('fcs_t2', b, d * 0.6 + Y, 0.36 + 0.12 * (j % 2), r=0.06, cluster=False, bone='head')
    K.ball('fcs_t2', cc + Y * 0.05, (0.13, 0.13, 0.13), mat='eye', seg=10, rings=6, bone='head')
    P, N = SN['fcs3']
    for i, (p, n) in enumerate(zip(P, N)):
        r = (0.14, 0.1, 0.12)[i]
        elder_eye(K, 'fcs_t3', p - n * r * 0.35, unit(n + np.array([0, 0.2, 0.4])), r)
    P, N = SN['bud']
    for s, sd, p, n in zip((-1, 1), ('L', 'R'), P, N):
        elder_eye(K, 'fcs_t3', p - n * 0.03, unit(n + np.array([0, 0.3, 0.5])), 0.075, bone='upperarm' + sd, hood=False)
