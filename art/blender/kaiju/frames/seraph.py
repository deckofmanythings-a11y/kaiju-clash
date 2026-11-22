"""Seraph: secret Wild-only "biblically accurate" angel (ophanim / seraphim mash-up). Hovering, faces +Z.

No flesh body: a glowing accent orb caged in gold ribs with one great eye (the `body` skin), three nested
gimbal rings full of eyes (crisp procedural profile sweeps, each fully weighted to its own ringN bone so the
runtime can spin them independently) and three pairs of layered feather wings rooted on the back boss and
fanned out BEHIND the gyroscope (upper swept up, middle spread wide, lower swept down), clear of every ring's
swept volume. A sunburst of gold light-blades (Might) radiates from behind the core into the inner ring space.
"""
import math

import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE

DETAIL = 'metal'
UV_SCALE = 0.75
METAL = True
EYE_ACCENT = True
CAM = dict(az=28, el=8)

INFO = dict(name='Seraph', colors=dict(body='#d9a93c', belly='#f3ecdc', accent='#9fdcff'))

C = np.array([0.0, 5.5, 0.0])     # core centre
AX4 = (0.3, 0.22, 1.0)            # arm_t3 outer ring axis (tilted front ring, stays in front of the wings)
ORB = 1.3
F32 = np.float32


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def frame_R(axis, t=None):
    x, y, z = geom.basis(axis, t)
    return np.stack([x, y, z], 1)          # columns = local axes (local Y = axis)


class Band:
    """Torus with a rounded-rect cross-section (radial half `tr`, axial half `ty`); local Y = ring axis."""

    def __init__(self, c, R, axis, tr, ty, rad=0.03):
        self.c, self.R0, self.tr, self.ty, self.rad = np.asarray(c, float), R, tr, ty, rad
        self.Rm = frame_R(axis)
        e = R + max(tr, ty) + 0.05
        self.lo, self.hi = self.c - e, self.c + e

    def dist(self, reg, pad=0):
        lx, ly, lz = S._local(reg, self.c, self.Rm)
        q = np.sqrt(lx * lx + lz * lz) - F32(self.R0)
        r = F32(self.rad)
        dx, dy = np.abs(q) - F32(self.tr) + r, np.abs(ly) - F32(self.ty) + r
        return np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2) + np.minimum(np.maximum(dx, dy), 0) - r


# ------------------------------------------------------------------ rings
# cross-section (u = radial offset, v = axial): inner face, two raised rims, recessed channel on the outer face
WHEEL = [(-0.13, -0.32), (-0.13, 0.32), (0.12, 0.34), (0.18, 0.30), (0.18, 0.22),
         (0.13, 0.19), (0.13, -0.19), (0.18, -0.22), (0.18, -0.30), (0.12, -0.34)]
WHEEL_MAT = ['belly', 'body', 'body', 'body', 'body', 'engrave', 'body', 'body', 'body', 'body']
WASHER = [(v, u) for u, v in WHEEL]      # same section turned 90 deg: channel on the front (+axis) face
SLIM = [(v * 0.8, u) for u, v in WHEEL]  # narrower washer for the middle ring
CLASP = [(-0.2, -0.42), (-0.2, 0.42), (0.25, 0.42), (0.25, -0.42)]
PROFILES = dict(wheel=WHEEL, washer=WASHER, slim=SLIM)
WASHER_MAT = ['body', 'body', 'body', 'body', 'body', 'engrave', 'body', 'body', 'body', 'belly']   # gold back face
PROF_MAT = dict(wheel=WHEEL_MAT, washer=WASHER_MAT, slim=WASHER_MAT)
HALF_W = dict(wheel=0.2, washer=0.2, slim=0.16)   # half width of the channel for bars

# One side ring (axis ~X, its back arc passes through the vertical gap between the wing pairs) and two tilted
# front rings whose deepest point stays in front of the wing layer. Radii are nested with clearance, so the
# rings can spin about their own axes forever without touching each other, the core or the wings.
RINGS = [  # name, radius, axis, profile kind, eye count, segments
    ('ring1', 2.65, (1.0, 0.08, 0.15), 'wheel', 10, 56),
    ('ring2', 3.3, (-0.35, 0.3, 1.0), 'slim', 12, 64),
    ('ring3', 4.1, (0.3, -0.25, 1.0), 'washer', 14, 72),
]
R4 = 4.85                                # arm_t3 outer ring radius (axis AX4)


def ring_pt(Rm, R, th, u, v):
    th = np.asarray(th, float)
    loc = np.stack([(R + u) * np.cos(th), np.full_like(th, v), (R + u) * np.sin(th)], -1)
    return C + loc @ Rm.T


def ring_band(K, part, R, axis, prof, mats, nseg, bone, th0=0.0, th1=2 * math.pi, shade=1.0):
    Rm = frame_R(axis)
    closed = th1 - th0 >= 2 * math.pi - 1e-6
    ths = np.linspace(th0, th1, nseg, endpoint=not closed)
    m = len(ths)
    P = np.array(prof, float)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    for e in range(len(P)):
        (u0, v0), (u1, v1) = P[e], P[(e + 1) % len(P)]
        du, dv = u1 - u0, v1 - v0
        nu, nv = (dv, -du) if area > 0 else (-dv, du)
        A, Bv = ring_pt(Rm, R, ths, u0, v0), ring_pt(Rm, R, ths, u1, v1)
        V = np.vstack([A, Bv])
        T = []
        for i in range(m if closed else m - 1):
            j = (i + 1) % m
            T += [(i, j, m + j), (i, m + j, m + i)]
        T = np.array(T)
        mu, mv = (u0 + u1) / 2, (v0 + v1) / 2
        out = ring_pt(Rm, R, ths[:1], mu + nu * 0.01, mv + nv * 0.01)[0] - ring_pt(Rm, R, ths[:1], mu, mv)[0]
        fn = np.cross(V[T[0, 1]] - V[T[0, 0]], V[T[0, 2]] - V[T[0, 0]])
        if fn @ out < 0:
            T = T[:, ::-1].copy()
        face = 0.8 + 0.2 * max(0.0, nu / max(math.hypot(nu, nv), 1e-9))
        K.piece(part, V, T, mats[e] if isinstance(mats, list) else mats, col=np.full(len(V), face * shade), bone=bone)
    return Rm


def box(K, part, c, ax, h, mat, bone, col=1.0):
    """Oriented box: ax = 3 unit axes, h = half sizes."""
    c = np.asarray(c, float)
    V = np.array([c + sx * h[0] * ax[0] + sy * h[1] * ax[1] + sz * h[2] * ax[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    T = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    K.piece(part, V, geom.fix_winding(V, geom._consistent(V, T)), mat, smooth=False, col=col, bone=bone)


def oriented_ball(c, frame, radii, seg, rings):
    V, T, _ = geom.sphere((0, 0, 0), 1, seg, rings)
    x, y, z = frame
    V = np.asarray(c, float) + V[:, [0]] * x * radii[0] + V[:, [1]] * y * radii[1] + V[:, [2]] * z * radii[2]
    return V, T


def seye(K, part, c, n, t, r, bone, hi=False, almond=1.2):
    """Ivory eyeball + glowing iris + dark limbal ring + pupil. n = gaze, t = long axis of the almond."""
    n = unit(n)
    x = unit(np.asarray(t, float) - n * (np.asarray(t, float) @ n))
    z = np.cross(x, n)
    fr = (x, n, z)
    if not hi:   # cheap small eye: low ball + domed iris and pupil caps (they sit outside the sclera)
        V, T = oriented_ball(c, fr, (r * almond, r, r * 0.9), 7, 4)
        K.piece(part, V, T, 'sclera', col=0.95, bone=bone)
        cap(K, part, c, fr, r * 0.6, r * 0.8, r * 1.0, 7, 'eye', bone)
        cap(K, part, c, fr, r * 0.25, r * 0.95, r * 1.05, 5, 'dark', bone)
        return
    s1 = 16
    V, T = oriented_ball(c, fr, (r * almond, r, r * 0.9), s1, 10)
    K.piece(part, V, T, 'sclera', col=0.95, bone=bone)
    V, T = oriented_ball(c + n * r * 0.74, fr, (r * 0.66, r * 0.26, r * 0.66), s1, 6)
    K.piece(part, V, T, 'dark', bone=bone)
    V, T = oriented_ball(c + n * r * 0.8, fr, (r * 0.56, r * 0.25, r * 0.56), s1, 6)
    K.piece(part, V, T, 'eye', bone=bone)
    V, T = oriented_ball(c + n * r * 0.95, fr, (r * 0.24, r * 0.12, r * 0.24), 8, 4)
    K.piece(part, V, T, 'dark', bone=bone)


def cap(K, part, c, fr, rad, h0, h1, sides, mat, bone):
    """Low dome (fan) facing frame Y: rim of radius `rad` at height h0, apex at h1."""
    x, n, z = fr
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    V = np.vstack([c + n * h1] + [c + n * h0 + (x * math.cos(a) + z * math.sin(a)) * rad for a in ang])
    T = np.array([(0, 1 + i, 1 + (i + 1) % sides) for i in range(sides)])
    fn = np.cross(V[T[0, 1]] - V[0], V[T[0, 2]] - V[0])
    if fn @ n < 0:
        T = T[:, ::-1].copy()
    K.piece(part, V, T, mat, bone=bone)


def ring_eyes(K, part, R, axis, kind, n, bone, r=0.17, offset=0.0, seed=0, step=1):
    Rm = frame_R(axis)
    rng = np.random.default_rng(seed)
    out = []
    for k in range(0, n, step):
        th = 2 * math.pi * (k + offset) / n
        if kind == 'wheel':
            c = ring_pt(Rm, R, [th], 0.13 - r * 0.25, 0.0)[0]
            g = unit(ring_pt(Rm, R, [th], 1.0, 0.0)[0] - ring_pt(Rm, R, [th], 0.0, 0.0)[0])
        else:
            c = ring_pt(Rm, R, [th], 0.0, 0.13 - r * 0.25)[0]
            g = Rm[:, 1].copy()
        tng = unit(ring_pt(Rm, R, [th + 0.01], 0.0, 0.0)[0] - ring_pt(Rm, R, [th - 0.01], 0.0, 0.0)[0])
        g = unit(g + tng * rng.uniform(-0.25, 0.25) + np.cross(g, tng) * rng.uniform(-0.2, 0.2))
        seye(K, part, c, g, tng, r, bone)
        out.append((th, c, g, tng))
    return out


def ring_bars(K, part, R, axis, kind, n, bone, mat='body'):
    Rm = frame_R(axis)
    for k in range(n):
        th = 2 * math.pi * (k + 0.5) / n
        tng = unit(ring_pt(Rm, R, [th + 0.01], 0, 0)[0] - ring_pt(Rm, R, [th - 0.01], 0, 0)[0])
        rad = unit(ring_pt(Rm, R, [th], 1, 0)[0] - ring_pt(Rm, R, [th], 0, 0)[0])
        ax = Rm[:, 1]
        hw = HALF_W[kind]
        if kind == 'wheel':
            box(K, part, ring_pt(Rm, R, [th], 0.16, 0)[0], (tng, rad, ax), (0.035, 0.04, hw), mat, bone)
        else:
            box(K, part, ring_pt(Rm, R, [th], 0, 0.16)[0], (tng, rad, ax), (0.035, hw, 0.04), mat, bone)


# ------------------------------------------------------------------ feathers / wings
US = np.array([0.0, 0.24, 0.48, 0.7, 0.88, 1.0])
PROF = np.where(US < 0.55, 0.32 + 0.68 * np.sin(np.clip(US / 0.55, 0, 1) * np.pi / 2), np.sqrt(np.clip(1 - ((US - 0.55) / 0.45) ** 2, 0, 1)))


def feather(K, part, base, d, nrmv, L, W, bone, mat='belly', tone=1.0, asym=0.7, droop=0.07, tip_mat=None):
    d = unit(d)
    n = unit(nrmv - d * (nrmv @ d))
    side = np.cross(n, d)
    w = W * PROF
    V, col = [], []
    for i, u in enumerate(US):
        p = base + d * L * u - n * droop * L * u * u
        V += [p - side * w[i] * asym - n * 0.25 * w[i], p + n * 0.02 * L * (1 - u) ** 0.5, p + side * w[i] - n * 0.25 * w[i]]
        col += [tone * (0.74 + 0.22 * u), tone * (0.86 + 0.14 * u), tone * (0.8 + 0.2 * u)]
    T = []
    for i in range(len(US) - 1):
        a, b = 3 * i, 3 * (i + 1)
        T += [(a, b, b + 1), (a, b + 1, a + 1), (a + 1, b + 1, b + 2), (a + 1, b + 2, a + 2)]
    V, T = np.array(V), np.array(T)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    flip = (fn @ n) < 0
    T[flip] = T[flip][:, ::-1]
    K.piece(part, V, T, mat, col=np.clip(col, 0.05, 1), bone=bone)


class Wing:
    def __init__(self, arm, s, chord, hint=(0, 0, 1)):
        m = np.array([s, 1, 1])
        self.s = s
        self.A = S.catmull(np.array(arm, float) * m, 6)
        seg = np.linalg.norm(np.diff(self.A, axis=0), axis=1)
        self.u = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
        self.chord = unit(np.array(chord, float) * m)
        n = unit(np.cross(self.A[-1] - self.A[0], self.chord))
        if n @ np.asarray(hint, float) < 0:
            n = -n
        self.n = n

    def at(self, t):
        return np.array([np.interp(t, self.u, self.A[:, i]) for i in range(3)])

    def tan(self, t):
        return unit(self.at(min(t + 0.02, 1)) - self.at(max(t - 0.02, 0)))


def build_wing(K, part, W, L, wd, bone, rows=None, arm_r=(0.17, 0.05), lift=0.0):
    K.horn(part, W.A, list(np.linspace(arm_r[0], arm_r[1], len(W.A))), mat='body', sides=7, per=1, bone=bone, shade=(0.8, 1.0))
    K.ball(part, W.at(1.0), (arm_r[1] * 1.8,) * 3, mat='body', seg=8, rings=5, bone=bone)
    rows = rows or [
        # count, t0, t1, length, width, layer offset, fan toward the arm tip, tone
        (7, 0.54, 1.0, 1.0, 0.32, 0.0, 0.8, 1.0),     # primaries
        (7, 0.06, 0.6, 0.66, 0.27, 0.0, 0.05, 0.93),  # secondaries
        (8, 0.04, 0.96, 0.38, 0.26, 0.1, 0.25, 0.97),  # greater coverts
        (6, 0.03, 0.85, 0.2, 0.24, 0.17, 0.3, 1.0),   # lesser coverts
    ]
    feats = []
    for ri, (cnt, t0, t1, ln, wf, off, fan, tone) in enumerate(rows):
        for i in range(cnt):
            f = i / max(cnt - 1, 1)
            t = t0 + (t1 - t0) * f
            p = W.at(t)
            if ri == 0:
                d = unit(W.chord * (1 - fan * f) + W.tan(t) * fan * f * 1.1)
                lf = 0.78 + 0.38 * f - 0.22 * f ** 5
            elif ri == 1:
                d = unit(W.chord + W.tan(t) * fan)
                lf = 1.0 - 0.18 * (1 - f) ** 2
            else:
                d = unit(W.chord * (1 - fan * f * f) + W.tan(t) * (0.12 + fan * f * f))
                lf = 0.9 + 0.2 * f
            base = p + W.n * (off + 0.009 * i) - W.chord * 0.04
            feather(K, part, base, d, W.n, L * ln * lf, wd * wf / 0.25, bone, tone=tone)
            feats.append((ri, t, base, d, L * ln * lf))
    return feats


WINGS = {  # key: (bone prefix, arm pts R side, chord, length, width, front hint)
    # all roots on the back boss, arms swept outward AND back so every feather lies in a layer behind the
    # deepest point of the front rings; the side ring's back arc passes through the vertical gap between pairs
    'F': ('wingF', [(0.75, 6.6, -1.55), (1.45, 7.7, -2.15), (2.15, 8.8, -2.5), (2.6, 9.75, -2.65)], (1.0, -0.4, -0.35), 2.4, 0.25, (0, 0, 1)),
    'H': ('wingH', [(0.85, 5.75, -1.7), (2.1, 6.0, -2.45), (3.4, 6.15, -2.85), (4.6, 5.95, -3.0)], (0.1, -1.0, -0.3), 2.3, 0.26, (0, 0, 1)),
    'L': ('wingL', [(0.75, 4.55, -1.5), (1.4, 3.6, -1.9), (2.0, 2.65, -2.15), (2.3, 1.9, -2.25)], (0.85, -0.5, -0.12), 1.9, 0.23, (0, 0, 1)),
}


# ------------------------------------------------------------------ build
def build(K):
    K.mat('belly', metal=0.0, rough=0.6, double=True, detail_kind='fur', detail_strength=0.35)
    K.mat('body', double=True, metal=0.85, rough=0.3, detail_strength=0.3)
    K.mat('sclera', color='#f5f0e4', rough=0.28, metal=0.0, vcol=True)
    K.mat('engrave', color='#16245a', rough=0.3, metal=0.35, vcol=True)
    K.mat('plate', color='#c08a2c', metal=0.85, rough=0.32)
    K.mat('accent', double=True, emit_strength=3.4)
    K.mat('eye', emit_strength=3.0)

    # ---- core housing: glowing orb in a gold cage, eye bezel + ivory petals, finial, pendant, wing boss
    B = K.sdf()
    B.ell(C, ORB, k=0.0)
    for phi in (75, 125, 180):
        pts = []
        for th in np.linspace(14, 166, 9):
            a, b = math.radians(th), math.radians(phi)
            pts.append(C + 1.32 * np.array([math.sin(a) * math.sin(b), math.cos(a), math.sin(a) * math.cos(b)]))
        B.tube(pts, [0.07, 0.1, 0.11, 0.12, 0.12, 0.12, 0.11, 0.1, 0.07], k=0.06, mirror=phi != 180)
    B.tube([C + (0, 0.95, 0), C + (0, 1.35, 0), C + (0, 1.8, 0), C + (0, 2.25, 0)], [0.6, 0.3, 0.13, 0.015], k=0.25)  # finial spire
    B.ell(C + (0, 1.2, 0), (0.62, 0.15, 0.62), k=0.1)
    B.ell(C + (0, 1.62, 0), (0.22, 0.22, 0.22), k=0.08)
    B.tube([C - (0, 0.95, 0), C - (0, 1.35, 0), C - (0, 1.8, 0), C - (0, 2.25, 0)], [0.6, 0.36, 0.15, 0.015], k=0.25)  # pendant
    B.ell(C - (0, 1.2, 0), (0.64, 0.15, 0.64), k=0.1)
    B.ell(C - (0, 1.65, 0), (0.2, 0.2, 0.2), k=0.08)
    B.ell(C + (0, 0.05, -1.15), (0.78, 1.45, 0.6), k=0.3)                                   # wing boss
    for key, (bn, arm, *_r) in WINGS.items():
        B.ell(arm[0], (0.3, 0.3, 0.3), k=0.2, mirror=True)
    collar = Band(C + (0, 0, 1.02), 0.87, (0, 0, 1), 0.1, 0.09, rad=0.08)
    B.add(collar, k=0.05)
    petals = []
    for k in range(10):
        beta = 2 * math.pi * k / 10 + math.pi / 2
        alpha = math.radians(56)
        dirv = np.array([math.sin(alpha) * math.cos(beta), math.sin(alpha) * math.sin(beta), math.cos(alpha)])
        mer = unit(np.array([math.cos(alpha) * math.cos(beta), math.cos(alpha) * math.sin(beta), -math.sin(alpha)]))
        tng = np.cross(dirv, mer)
        Rm = np.stack([tng, dirv, mer], 1)
        p = S.Ell(C + dirv * 1.3, (0.19, 0.08, 0.4), Rm)
        petals.append(p)
        B.add(p, k=0.06)

    def groove(x, y, z):
        P = np.stack([x, y, z], 1) - C
        rr = np.linalg.norm(P, axis=1)
        out = np.zeros(len(x), np.float32)
        spire = (np.abs(P[:, 0]) < 0.8) & (np.abs(P[:, 2]) < 0.8) & (np.abs(P[:, 1]) > 1.0)
        out += spire * 0.02 * (np.sin(P[:, 1] * 40) > 0.6)
        return out * (rr > 0.9)
    B.disp(groove, band=0.06)

    def bellymask(P):
        Pp = S.Points(P)
        d = collar.dist(Pp)
        for p in petals:
            d = np.minimum(d, p.dist(Pp))
        return d - 0.015

    def accentmask(P):
        P = np.asarray(P, float)
        Q = P - (C + (0, 0.05, 0))
        rxy = np.hypot(Q[:, 0], Q[:, 1])
        rosette = np.abs(rxy - 0.56) - 0.045                                          # glowing ring on the boss back
        seam = np.maximum(np.maximum(np.abs(Q[:, 0]) - 0.04, np.abs(Q[:, 1]) - 1.3), 0.62 - rxy)   # light seam up/down the boss
        back = np.maximum(np.minimum(rosette, seam), Q[:, 2] + 1.3)
        return np.maximum(np.minimum(np.linalg.norm(P - C, axis=1) - (ORB + 0.035), back), -bellymask(P))

    K.skin(B, voxel=0.03, tris=5800, belly=bellymask, regions=[('accent', accentmask)])

    # ---- skeleton
    K.bone('spine', C - (0, 2.3, 0), C + (0, 2.3, 0), None, 1.5)
    K.bone('head', C + (0, 0, 0.3), C + (0, 0, 1.5), 'spine', 0.3)
    for name, R, axis, kind, ne, ns in RINGS:
        K.bone(name, C, C + unit(axis) * 1.2, 'spine', 0.01)
    K.bone('ring4', C, C + unit(AX4) * 1.2, 'spine', 0.01)
    wings = {}
    for key, (bn, arm, chord, L, wd, hint) in WINGS.items():
        for s, sd in ((-1, 'L'), (1, 'R')):
            W = Wing(arm, s, chord, hint)
            K.bone(bn + sd, W.at(0.03), W.at(1.0), 'spine', 0.3)
            wings[key + sd] = (W, L, wd, bn + sd)

    # ---- the great eye (head)
    seye(K, 'body', C + (0, 0, 0.6), (0, 0, 1), (1, 0, 0), 0.9, 'head', hi=True, almond=1.0)

    # ---- rings
    for name, R, axis, kind, ne, ns in RINGS:
        ring_band(K, 'body', R, axis, PROFILES[kind], PROF_MAT[kind], ns, name)
        ring_eyes(K, 'body', R, axis, kind, ne, name, r=0.17 if kind != 'slim' else 0.15, seed=len(name) + ne)
        ring_bars(K, 'body', R, axis, kind, ne, name)

    # ---- wings
    feats = {}
    for key, (W, L, wd, bn) in wings.items():
        feats[key] = build_wing(K, 'body', W, L, wd, bn)

    # ---- mounts
    K.mount('head', C + (0, 0, 0.7), 'head')
    K.mount('jaw', C + (0, -0.55, 1.2), 'head')
    K.mount('mouth', C + (0, 0, 1.6), 'head')
    K.mount('chest', C + (0, -0.9, 1.1), 'spine')
    K.mount('back', C + (0, 0, -1.85), 'spine')
    K.mount('tailTip', C - (0, 2.35, 0), 'spine')
    K.mount('crown', C + (0, 2.4, 0), 'head')
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.mount('hand' + sd, wings['H' + sd][0].at(1.0), 'wingH' + sd)
        K.mount('shoulder' + sd, wings['F' + sd][0].at(0.0) + (0, 0.25, 0), 'spine')

    # ---- Might: sunburst of gold light-blades radiating from behind the core into the inner ring space
    # (exclusive, complete sets: more, longer, wider blades per tier; glowing fullers from t2, sword guards at t3)
    a1 = unit(RINGS[0][2])
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        n = [8, 10, 12, 14][t]
        tip = [2.2, 2.42, 2.6, 2.78][t]
        for i in range(n):
            a = 2 * math.pi * (i + 0.5) / n + math.pi / 2
            d = unit((math.cos(a), math.sin(a), 0.0))
            long = i % 2 == 0
            r1 = tip if long else 1.0 + 0.62 * (tip - 1.0)
            if abs((d * r1) @ a1) < 0.75:             # near the side ring's plane: stay inside its inner face
                r1 = min(r1, 2.3)
            base = C + d * 0.8 + (0, 0, -0.08)
            L = r1 - 0.8
            w = (0.25 if long else 0.17) * (0.9 + 0.1 * sc)
            K.horn(part, [base, base + d * L * 0.3, base + d * L * 0.78, base + d * L], [w * 0.75, w, w * 0.7, 0.0], mat='plate',
                   sides=4, per=2, flat=0.22, up=(0, 0, 1), bone='spine', shade=(0.75, 1.0))
            if t >= 2:   # glowing fuller along the blade's front face
                K.horn(part, [base + d * 0.55 + (0, 0, 0.03), base + d * L * 0.5 + (0, 0, w * 0.2), base + d * L * 0.9 + (0, 0, 0.02)],
                       [0.0, w * 0.42, 0.0], mat='accent', sides=4, per=2, flat=0.3, up=(0, 0, 1), bone='spine')
            if t >= 3 and long:   # sword guard just outside the orb silhouette
                g = base + d * 0.72
                side = unit(np.cross(d, (0, 0, 1)))
                K.horn(part, [g - side * w * 1.9, g, g + side * w * 1.9], [0.045, 0.065, 0.045], mat='body', sides=5, per=1, bone='spine')

    # ---- Armor
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = np.array([s, 1, 1])
        K.plate('arm_t1', np.array(WINGS['F'][1][0]) * m + (0, 0.1, 0), (0.55, 0.5, 0.55), bone='spine', lames=2, tris=200)
        K.plate('arm_t1', np.array(WINGS['H'][1][0]) * m, (0.45, 0.42, 0.5), bone='spine', lames=0, tris=160)
        for key in ('F', 'H'):
            W, L, wd, bn = wings[key + sd]
            for j, t in enumerate((0.1, 0.24, 0.38)):
                a, b = W.at(t), W.at(t + 0.16)
                K.horn('arm_t1', [a - W.n * 0.02, (a + b) / 2 + W.n * 0.08, b], [0.2, 0.24 - j * 0.02, 0.12], mat='plate', sides=6, per=2,
                       flat=0.55, up=W.n, bone=bn)
    for s in (-1, 1):   # flank plates on the boss back, leaving the glowing rosette + seam clear
        K.plate('arm_t1', C + (s * 0.62, -0.25, -1.45), (0.32, 0.8, 0.45), bone='spine', lames=2, tris=140)
    for name, R, axis, kind, ne, ns in RINGS:
        prof = [(-0.16, -0.42), (-0.16, 0.42), (0.21, 0.42), (0.21, -0.42)] if kind == 'wheel' else [(v * (0.72 if kind == 'slim' else 0.86), u) for u, v in CLASP]
        for k in range(4):
            th = 2 * math.pi * (k * (ne // 4) + 0.5) / ne
            ring_band(K, 'arm_t2', R, axis, prof, 'plate', 6, name, th - 0.07, th + 0.07)
    ring_band(K, 'arm_t3', R4, AX4, [(-0.08, -0.16), (-0.08, 0.16), (0.1, 0.2), (0.14, 0.0), (0.1, -0.2)], 'body', 72, 'ring4')
    Rm4 = frame_R(AX4)
    for k in range(16):
        th = 2 * math.pi * k / 16
        ring_band(K, 'arm_t3', R4, AX4, [(-0.12, -0.25), (-0.12, 0.25), (0.2, 0.25), (0.2, -0.25)], 'plate', 4, 'ring4', th - 0.06, th + 0.06)
        if k % 2 == 0:
            p = ring_pt(Rm4, R4, [th], 0.2, 0)[0]
            K.spike('arm_t3', p, p - C, 0.55, r=0.1, mat='plate', bone='ring4', sides=4)

    # ---- Energy
    rng = np.random.default_rng(11)
    for k in range(8):                      # flame tongues licking up off the orb's front rim
        beta = 2 * math.pi * k / 8 + 0.2
        dirv = unit((math.cos(beta), math.sin(beta) * 0.85 + 0.15, 0.55))
        b = C + dirv * 1.22
        up = np.array([0, 1.0, 0])
        L = rng.uniform(0.5, 0.75)
        K.horn('enr_t1', [b, b + dirv * L * 0.45 + up * L * 0.15, b + dirv * L * 0.65 + up * L * 0.45, b + dirv * L * 0.55 + up * L * 0.85],
               [0.12, 0.1, 0.05, 0.0], mat='accent', sides=5, per=2, bone='spine')
    for dx in (-0.1, 0.0, 0.1):             # spire flame (stays inside the side ring's inner face)
        tip = C + (0, 2.15, 0)
        K.horn('enr_t1', [tip - (0, 0.12, 0), tip + (dx, 0.12, 0), tip + (dx * 1.6, 0.2 - abs(dx) * 0.8, 0)], [0.07, 0.05, 0.0], mat='accent', sides=5, bone='spine')
    hc, hR = C + (0, 0, -1.0), (1.72, 1.95)   # nimbus disc behind the core, in front of the wing roots
    ths = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    ring_in = np.stack([hc[0] + hR[0] * np.cos(ths), hc[1] + hR[0] * np.sin(ths), np.full_like(ths, hc[2])], 1)
    ring_out = np.stack([hc[0] + hR[1] * np.cos(ths), hc[1] + hR[1] * np.sin(ths), np.full_like(ths, hc[2])], 1)
    V = np.vstack([ring_in, ring_out])
    T = []
    for i in range(64):
        j = (i + 1) % 64
        T += [(i, j, 64 + j), (i, 64 + j, 64 + i)]
    K.piece('enr_t2', V, np.array(T), 'accent', bone='spine')
    for i in range(20):
        a = 2 * math.pi * (i + 0.5) / 20
        d = np.array([math.cos(a), math.sin(a), 0])
        if abs(math.cos(a)) < 0.4:          # leave the side ring's lane clear
            continue
        L = 0.3 if i % 2 else 0.5
        K.horn('enr_t2', [hc + d * (hR[1] - 0.05), hc + d * (hR[1] + L * 0.5), hc + d * (hR[1] + L)], [0.05, 0.035, 0.0], mat='accent', sides=4, per=1, bone='spine')
    for name, R, axis, kind, ne, ns in RINGS[1:]:   # crystals growing forward off the front rings' faces
        Rm = frame_R(axis)
        for k in range(3):
            th = 2 * math.pi * (k + 0.5) / 3
            p = ring_pt(Rm, R, [th], 0.0, 0.15)[0]
            K.crystal('enr_t3', p, Rm[:, 1], 0.5, seed=k + len(name), bone=name)
    for s, sd in ((-1, 'L'), (1, 'R')):     # glowing primary tips on the upper wings
        for ri, t, base, d, L in feats['F' + sd]:
            if ri == 0:
                K.horn('enr_t3', [base + d * L * 0.55, base + d * L * 0.8, base + d * L * 1.0], [0.0, 0.05, 0.0], mat='accent', sides=4, per=1, bone='wingF' + sd)

    # ---- Speed
    for s, sd in ((-1, 'L'), (1, 'R')):     # t1: a back layer of long swept vane-feathers under the upper and middle wings
        for key in ('F', 'H'):
            W, L, wd, bn = wings[key + sd]
            for i in range(5):
                f = i / 4
                t = 0.5 + 0.5 * f
                d = unit(W.chord * (1 - 0.85 * f) + W.tan(t) * (0.25 + 0.95 * f))
                feather(K, 'spd_t1', W.at(t) - W.n * 0.1, d, W.n, L * (1.05 + 0.3 * f), 0.3, bn, tone=0.9, droop=0.03)
    for s, sd in ((-1, 'L'), (1, 'R')):     # t2: long streamer plumes off the upper wing tips
        W, L, wd, bn = wings['F' + sd]
        for j in range(3):
            d = unit(W.tan(1.0) * (1.0 - j * 0.12) + W.chord * (0.12 + j * 0.22))
            feather(K, 'spd_t2', W.at(1.0) - W.n * (0.03 + 0.01 * j), d, W.n, 2.0 - j * 0.4, 0.22, bn, tone=1.0, droop=-0.04)
    for s, sd in ((-1, 'L'), (1, 'R')):     # t3: long trailing tip feathers on the middle and lower wings
        for key, nl in (('H', 3), ('L', 2)):
            W, L, wd, bn = wings[key + sd]
            for j in range(nl):
                t = 1.0 - j * 0.07
                d = unit(W.chord * (0.45 - j * 0.1) + W.tan(t) * 1.0)
                feather(K, 'spd_t3', W.at(t) + W.n * (-0.03 - j * 0.01), d, W.n, L * (1.55 - j * 0.25), 0.2, bn, tone=1.0)
                K.ball('spd_t3', W.at(t) + d * L * (1.55 - j * 0.25) * 0.93, (0.07, 0.07, 0.07), mat='accent', seg=6, rings=4, bone=bn)

    # ---- Focus: eyes open on the wings, more eyes on the rings, a crown of eyes
    for s, sd in ((-1, 'L'), (1, 'R')):
        for key, picks in (('H', (1, 4, 7)), ('F', (2, 5))):
            W, L, wd, bn = wings[key + sd]
            cov = [f for f in feats[key + sd] if f[0] == 2]
            for pi in picks:
                ri, t, base, d, ln = cov[pi]
                seye(K, 'fcs_t1', base + d * ln * 0.55 + W.n * 0.06, W.n, d, 0.15, bn)
        W, L, wd, bn = wings['L' + sd]
        cov = [f for f in feats['L' + sd] if f[0] == 2]
        for pi in (2, 5):
            ri, t, base, d, ln = cov[pi]
            seye(K, 'fcs_t2', base + d * ln * 0.55 + W.n * 0.06, W.n, d, 0.13, bn)
    for name, R, axis, kind, ne, ns in RINGS:
        ring_eyes(K, 'fcs_t2', R, axis, kind, ne, name, r=0.15 if kind != 'slim' else 0.13, offset=0.5, seed=ne * 3, step=2)
    crown_c = C + (0, 1.92, 0)
    hoop = [crown_c + 0.7 * np.array([math.cos(a), 0.0, math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 21)]
    K.horn('fcs_t3', hoop, [0.055] * 21, mat='body', sides=6, per=1, bone='head')
    for k in range(7):
        a = 2 * math.pi * k / 7 + math.pi / 2
        out = np.array([math.cos(a), 0.0, math.sin(a)])
        p = crown_c + out * 0.7 + (0, 0.15, 0)
        K.horn('fcs_t3', [crown_c + out * 0.7, p], [0.05, 0.05], mat='body', sides=5, per=1, bone='head')
        seye(K, 'fcs_t3', p + (0, 0.04, 0), unit(out + (0, 0.25, 0)), np.cross((0, 1, 0), out), 0.14, 'head')
    seye(K, 'fcs_t3', C + (0, 0.05, -1.82), (0, 0.1, -1), (1, 0, 0), 0.34, 'spine', almond=1.25)   # rear eye on the boss
