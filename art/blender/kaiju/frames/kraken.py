"""Kraken: tentacled horror from the deep (cephalopod plan). Game coords, ~10 units tall, faces +Z.

A bulging head with huge glowing eyes, a squat keeled mantle dome hooking back over it, eight thick arms that sprawl
and curl on the ground, the two front ones reared up beside the face to strike. Suckers + pale underside
follow each arm's ventral side.
"""
import math

import numpy as np

from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, voronoi, smoothstep

DETAIL = 'skin'
UV_SCALE = 0.55
EYE_ACCENT = True
CAM = dict(az=30, el=13)

Y = np.array([0.0, 1, 0])
HEAD_C = np.array([0.0, 4.15, 1.0])
# squat, domed mantle: rises from behind the brow, swells, then hooks back over the rear arms
MANTLE = [(0, 4.9, -0.35), (0, 5.8, -0.95), (0, 6.45, -1.75), (0, 6.7, -2.6), (0, 6.45, -3.35), (0, 5.95, -3.8)]
MANTLE_R = [2.05, 2.45, 2.3, 1.75, 1.0, 0.4]
MANTLE_S = (1.14, 1, 1)
EYE = np.array([1.42, 4.95, 2.2])
EYE_R = 0.56
EYE_N = np.array([0.62, 0.12, 0.78])


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def taper(n, r0, r1, p=0.85):
    t = np.linspace(0, 1, n)
    return list(r1 + (r0 - r1) * (1 - t) ** p)


# ------------------------------------------------------------------ arm paths
def ground_arm(a_deg, reach, sway, curl, cr=0.6):
    """Arm that drops from the web to the ground, snakes outward in an S and curls its tip.
    curl: 'up' (tip rolls up and back over itself, suckers outside) or +-1 (flat spiral to that side)."""
    a = math.radians(a_deg)
    o = np.array([math.sin(a), 0, math.cos(a)])
    l = np.array([math.cos(a), 0, -math.sin(a)])
    c0 = np.array([0, 0, 0.55])
    dl = [(1.05, 0.0), (1.9, 0.12 * sway), (2.75, 0.45 * sway), (3.55 * reach, 0.5 * sway), (4.25 * reach, 0.05 * sway),
          (4.85 * reach, -0.4 * sway), (5.35 * reach, -0.45 * sway)]
    n_c = 5
    R = taper(len(dl) + n_c, 0.72, 0.12, 0.9)
    pts, und = [], []
    tw = 1 if sway >= 0 else -1
    for i, (d, lat) in enumerate(dl):
        h = 2.75 if i == 0 else (1.45 if i == 1 else R[i] * 0.88)
        pts.append(c0 + o * d + l * lat + Y * h)
        und.append(unit(-Y + l * 0.75 * tw) if i >= 2 else unit(-Y * 0.9 + o * 0.4))
    E = pts[-1]
    f = unit((pts[-1] - pts[-2]) * [1, 0, 1])
    side = np.cross(Y, f)
    for j in range(n_c):
        th = math.radians(62 * (j + 1))
        rr = cr * (1 - 0.3 * (j + 1) / n_c)
        rad = R[len(dl) + j]
        if curl == 'up':
            cc = E + Y * cr
            p = cc + f * rr * math.sin(th) - Y * rr * math.cos(th)
            u = unit(p - cc)
        else:
            cc = E + side * curl * cr
            p = cc + f * rr * math.sin(th) - side * curl * rr * math.cos(th)
            p[1] = rad * 0.88
            u = unit(unit(p - cc) * 0.8 - Y * 0.5)
        pts.append(p)
        und.append(u)
    return dict(pts=np.array(pts), r=np.array(R), under=np.array(und), raised=False)


def raised_arm(s):
    """Front arm reared up beside the face like a cobra, the tip hooked forward at the enemy."""
    P = [(0.85, 2.7, 1.75), (1.9, 1.05, 2.6), (3.15, 1.2, 2.7), (3.95, 2.7, 2.15), (4.1, 4.5, 1.7), (3.8, 6.2, 1.65),
         (3.4, 7.5, 2.1), (3.1, 8.05, 3.0), (2.95, 7.8, 3.75), (2.9, 7.15, 3.95)]
    U = [(0, -1, 0.3), (0, -1, 0.3), (-0.3, -0.5, 1), (-0.45, -0.1, 1), (-0.5, 0, 1), (-0.5, -0.1, 1), (-0.3, -0.6, 0.75),
         (-0.2, -1, -0.05), (-0.2, -0.55, -0.85), (-0.2, 0.15, -1)]
    pts = np.array(P) * [s, 1, 1]
    und = unit(np.array(U) * [s, 1, 1])
    R = np.array(taper(len(P), 0.8, 0.14, 0.75))
    return dict(pts=pts, r=R, under=und, raised=True)


def sucker(K, c, u, t, rs, h, sides=7, bone=None):
    """Cup: outer wall, rolled rim, recessed centre (belly material, darker inside)."""
    u = unit(u)
    t = unit(t - u * (t @ u))
    sd = np.cross(t, u)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rings = [(c - u * h * 0.4, rs), (c + u * h, rs * 0.86), (c + u * h * 0.85, rs * 0.48)]
    V, col = [], []
    for (o, rr), cv in zip(rings, (0.8, 1.0, 0.72)):
        for a in ang:
            V.append(o + (np.cos(a) * sd + np.sin(a) * t) * rr)
            col.append(cv)
    V.append(c + u * h * 0.25)
    col.append(0.45)
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
    # orient: outer wall radially out, rim + cup facing along u
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    fc = V[T].mean(1) - c
    rad = fc - np.outer(fc @ u, u)
    exp = np.where(np.arange(len(T))[:, None] < 2 * sides, rad, u[None] + 0 * rad)
    exp[2 * sides:] = u
    flip = (fn * exp).sum(1) < 0
    T[flip] = T[flip][:, ::-1]
    K.piece('body', V, T, 'belly', col=np.array(col), bone=bone)


def resample(arm, per=5):
    C = S.catmull(arm['pts'], per)
    R = np.clip(S.catmull(arm['r'], per)[:, 0], 0.05, None)
    U = unit(S.catmull(arm['under'], per))
    T = unit(np.gradient(C, axis=0))
    U = unit(U - T * (U * T).sum(1, keepdims=True))
    return C, R, U, T


def mantle_frame(u):
    """Axis point, radius, tangent, dorsal normal and lateral axis at fraction u (0 = neck, 1 = tip)."""
    C = S.catmull(MANTLE, 12)
    R = S.catmull(MANTLE_R, 12)[:, 0]
    T = unit(np.gradient(C, axis=0))
    i = int(round(np.clip(u, 0, 1) * (len(C) - 1)))
    t = T[i]
    n = unit(np.array([0, -t[2], t[1]]))     # outside of the curve = dorsal
    return C[i], R[i], t, n, np.array([1.0, 0, 0])


def mpt(u, ang, out=1.15):
    """Rough point just outside the mantle at axis fraction u, angle ang (deg, 0 = dorsal, +90 = +X)."""
    c, r, t, n, x = mantle_frame(u)
    a = math.radians(ang)
    d = n * math.cos(a) + x * math.sin(a) * MANTLE_S[0]
    return c + d * r * out


class Snaps:
    """Collect snap queries, project them all in ONE K.snap call (separate snaps are slow here)."""
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


def build(K):
    K.mat('body', rough=0.36, detail_strength=0.35)
    K.mat('belly', rough=0.4, detail_strength=0.25)
    K.mat('plate', color='#3d2036', rough=0.3, metal=0.25)
    B = K.sdf()
    belly = K.sdf()

    # ---- head: broad skull mass, brow shelf, muzzle over the beak
    B.ell(HEAD_C, (2.1, 1.75, 1.85), k=0.0)
    B.ell((0, 3.55, 1.85), (1.4, 1.05, 1.0), k=0.5)                                     # muzzle mass over the beak
    B.ell((0, 5.35, 1.15), (1.75, 0.75, 1.25), k=0.55)                                  # heavy forehead/brow mass
    # ---- mantle: squat muscular dome, broad in X, hooking back at the tip
    B.tube(MANTLE, MANTLE_R, k=0.7, s=MANTLE_S)
    B.ell((1.05, 5.25, -0.35), (1.0, 1.15, 1.05), rot=(-30, 0, 0), k=0.65, mirror=True)  # mantle shoulders
    # dorsal keel + paired lateral keels (armoured, ridged structure)
    keel, lat = [], []
    for u in np.linspace(0.04, 1.0, 8):
        c, r, t, n, x = mantle_frame(u)
        keel.append(c + n * r * 0.96)
        lat.append(c + (n * 0.62 + x * 0.78 * MANTLE_S[0]) * r * 0.97)
    B.tube(keel, [0.2, 0.28, 0.3, 0.3, 0.3, 0.28, 0.24, 0.12], k=0.35, s=(0.55, 1, 1))
    B.tube(lat[1:6], [0.1, 0.2, 0.22, 0.18, 0.08], k=0.4, mirror=True)
    # mantle rim: a thick lip where the mantle opening wraps round behind the head
    rim = []
    for a in np.linspace(-1, 1, 9):
        th = math.radians(118 * a)
        rim.append((2.2 * math.sin(th), 4.55 + 0.45 * math.cos(th), 0.15 - 1.35 * math.cos(th)))
    B.tube(rim, [0.12, 0.26, 0.32, 0.32, 0.32, 0.32, 0.32, 0.26, 0.12], k=0.25)
    # squid fins: a broad, flat arrowhead pair at the mantle tip, swept back low behind the body (never ear-like)
    ct, rt, tt, nt, _ = mantle_frame(0.9)
    for sx in (-1, 1):
        for (off, ln, wd, sw) in ((0.0, 1.55, 0.95, 0.75), (0.55, 0.9, 0.55, 1.1)):
            o = unit(np.array([sx * 1.0, -0.12, 0]) + tt * sw)
            w = unit(np.cross(tt, o))
            if w[1] < 0:
                w = -w
            t2 = unit(np.cross(o, w))
            ctr = ct + np.array([sx * 0.35, -0.25, 0]) + o * (ln * 0.75) + tt * off
            B.add(S.Ell(ctr, (ln, 0.085, wd), np.stack([o, w, t2], 1)), 0.3)
    # eye bulges + heavy brows
    B.ell(EYE - EYE_N * 0.3, (0.88, 0.85, 0.85), k=0.4, mirror=True)
    B.tube([(0.35, 5.75, 2.3), (0.95, 6.0, 2.45), (1.7, 5.85, 2.25), (2.2, 5.3, 1.75)], [0.2, 0.36, 0.33, 0.15], k=0.2, mirror=True)
    B.ell(EYE, (EYE_R * 1.08,) * 3, cut=True, k=0.06, mirror=True)                       # eye sockets
    B.tube([(0.0, 5.55, 2.35), (0.0, 4.6, 2.7), (0, 3.9, 2.8)], [0.24, 0.26, 0.2], k=0.4)    # ridge between the eyes
    # funnel / siphon: thick muscular tube from under the mantle rim, flaring forward on the left side
    B.tube([(-1.45, 4.2, -0.2), (-2.05, 3.75, 0.35), (-2.4, 3.45, 0.95), (-2.5, 3.35, 1.25)], [0.55, 0.46, 0.4, 0.45], k=0.3)
    B.ell((-2.53, 3.33, 1.33), (0.26, 0.26, 0.24), cut=True, k=0.05)
    # web where the arms meet + beak cavity
    B.ell((0, 2.65, 0.6), (1.85, 0.75, 1.85), k=0.45)
    B.ell((0, 2.3, 1.5), (0.6, 0.42, 0.6), cut=True, k=0.15)

    # ---- arms: 2 reared, 6 sprawling (deliberately asymmetric)
    arms = [raised_arm(-1), raised_arm(1),
            ground_arm(-60, 1.0, 1.0, 'up', 0.62), ground_arm(64, 1.05, -1.0, 1, 0.6),
            ground_arm(-106, 1.1, -1.0, -1, 0.65), ground_arm(102, 0.95, 1.0, 'up', 0.55),
            ground_arm(-150, 0.95, 1.0, 1, 0.55), ground_arm(148, 1.05, -1.0, 'up', 0.6)]
    sampled = []
    for arm in arms:
        B.tube(arm['pts'], arm['r'], k=0.32)
        C, R, U, T = resample(arm)
        sampled.append((C, R, U, T))
        belly.tube(C + U * R[:, None] * 0.5, R * 0.72, k=0.0, per=1)
    belly.ell((0, 2.4, 0.8), (1.6, 0.55, 1.6), k=0.0)

    # ---- sculpt: wet skin, sparse warts, transverse muscle bands on the mantle, belly kept smooth
    def skin(x, y, z):
        f1, _, _ = voronoi(x, y, z, size=0.62, seed=11)
        wart = -0.065 * (1 - smoothstep(0.0, 0.26, f1))
        mantle = smoothstep(5.3, 6.3, y - 0.5 * z) * smoothstep(0.6, -0.4, z)
        ang = np.arctan2(y - 5.0, -z + 0.5)            # sweeps along the dome from neck to tip
        band = np.sin(ang * 22.0 + 0.8 * fbm(x, y, z, 2, freq=0.7, seed=2))
        ribs = -0.11 * smoothstep(-0.1, 0.9, band) * mantle
        swell = -0.035 * fbm(x, y, z, 2, freq=0.9, seed=7)
        return wart * (1 - 0.6 * mantle) + ribs + swell
    B.disp(skin, band=0.16)

    def tint(P, N):
        f1, _, _ = voronoi(P[:, 0], P[:, 1], P[:, 2], size=0.62, seed=11)
        spots = 1.0 + 0.16 * (1 - smoothstep(0.0, 0.22, f1))        # warts catch a paler tip
        mott = 1.0 + 0.14 * fbm(P[:, 0], P[:, 1], P[:, 2], 3, freq=0.5, seed=4)
        dorsal = 1.0 - 0.2 * np.clip(N[:, 1], 0, 1) * smoothstep(5.5, 8.0, P[:, 1])
        return spots * mott * dorsal

    K.skin(B, voxel=0.05, tris=14600, belly=belly, tint=tint)

    # ---- every surface query in ONE snap call
    SN = Snaps()
    cand = []
    for k, (arm, (C, R, U, T)) in enumerate(zip(arms, sampled)):
        idx = np.linspace(0, len(C) - 1, 7).round().astype(int)
        nm = ['tentL', 'tentR'][k] if k < 2 else f'tent{k - 1}_'
        Lc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        pos, row = Lc[-1] * (0.1 if arm['raised'] else 0.26), 0
        while pos < Lc[-1] * 0.97:
            i = min(int(np.searchsorted(Lc, pos)), len(C) - 1)
            r = R[i]
            if r < 0.15:
                break
            side = np.cross(T[i], U[i])
            bn = f'{nm}{int(np.clip(np.searchsorted(idx, i), 1, 6))}'
            cand.append((C[i] + unit(U[i] + side * 0.5 * (1 if row % 2 else -1)) * r, T[i], r, bn))
            row += 1
            pos += r * 0.54
    SN.add('suck', [c[0] for c in cand])
    SN.add('keel', [mpt(u, 0) for u in np.linspace(0.14, 0.9, 6)])
    SN.add('flsp', [mpt(u, s * 50) for s in (-1, 1) for u in (0.18, 0.4, 0.62)])
    SN.add('ph1', [mpt(u, s * 82) for s in (-1, 1) for u in np.linspace(0.05, 0.85, 6)])
    SN.add('ph2', [mpt(u, s * 30) for s in (-1, 1) for u in np.linspace(0.12, 0.88, 6)])
    SN.add('cry', [mpt(u, s * 22) for s in (-1, 1) for u in (0.45, 0.62, 0.79)])
    SN.add('tip', mpt(1.0, 0, 1.0) + np.array([0, 0.1, -0.4]))
    SN.add('brow', [(s * 1.9, 5.8, 1.9) for s in (-1, 1)])
    SN.add('browsp', [(s * x, 6.25, z) for s in (-1, 1) for x, z in ((0.55, 2.3), (1.25, 2.3))])
    SN.add('finrear', [mantle_frame(0.9)[0] + np.array([s * 2.6, -0.45, -1.3]) for s in (-1, 1)])
    SN.add('tipfin', [mpt(0.97, s * 70, 1.1) for s in (-1, 1)])
    SN.add('siph', [(s * 2.05, 3.9, -0.1) for s in (-1, 1)])
    SN.add('eye1', [(s * 0.75, 5.6, 2.55) for s in (-1, 1)])
    SN.add('eye3a', [(s * 2.15, 4.0, 1.5) for s in (-1, 1)])
    SN.add('eye3b', [(s * 0.85, 4.15, 2.75) for s in (-1, 1)])
    SN.add('crest', (0, 6.5, 0.9))
    SN.add('eye3c', (0, 5.35, 2.7))
    rings = []
    for s in (-1, 1):
        m = np.array([s, 1, 1])
        e, n0 = EYE * m, EYE_N * m
        ex = unit(np.cross(Y, n0))
        ey = unit(np.cross(n0, ex))
        for j in range(8):
            a = math.radians(-160 + j * 20)
            rings.append(e + (ex * math.cos(a) + ey * math.sin(a)) * EYE_R * 1.42 - n0 * 0.2)
    SN.add('ring', rings)
    SN.run(K)

    # ---- sucker cups (rigid, crisp at game scale): two staggered rows along each arm's ventral side
    SP, SNn = SN['suck']
    for (q, t, r, bn), p, n in zip(cand, SP, SNn):
        if p[1] > 0.12 and n[1] > -0.75:
            sucker(K, p - n * r * 0.06, n, t, r * 0.3, r * 0.2, sides=6, bone=bn)

    # ---- skeleton
    K.bone('pelvis', (0, 2.5, 0.6), (0, 3.7, 0.8), None, 1.8)
    K.bone('head', (0, 3.7, 0.8), (0, 4.8, 2.4), 'pelvis', 1.8)
    K.bone('jaw', (0, 2.7, 1.3), (0, 2.15, 2.1), 'head', 0.35)
    K.chain('spine', MANTLE, 'pelvis', (2.0, 1.0))
    tips = []
    for k, (arm, (C, R, U, T)) in enumerate(zip(arms, sampled)):
        idx = np.linspace(0, len(C) - 1, 7).round().astype(int)
        nm = ['tentL', 'tentR'][k] if k < 2 else f'tent{k - 1}_'
        last = K.chain(nm, C[idx], 'pelvis', (R[idx[1]] * 1.1, R[idx[-2]] * 1.1))
        tips.append((C[-1], last, nm))
    K.mount('handL', tips[0][0], tips[0][1])
    K.mount('handR', tips[1][0], tips[1][1])
    K.mount('head', (0, 4.9, 1.3), 'head')
    K.mount('jaw', (0, 2.4, 1.7), 'jaw')
    K.mount('mouth', (0, 2.25, 2.15), 'jaw')
    K.mount('chest', (0, 3.2, 2.75), 'head')
    K.mount('back', mpt(0.5, 0, 1.1), 'spine2')
    K.mount('tailTip', mpt(1.0, 0, 0.0) + mantle_frame(1.0)[2] * 0.9, 'spine4')
    K.mount('crown', (0, 6.3, 1.4), 'head')
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.mount('shoulder' + sd, (s * 2.15, 5.4, -0.2), 'spine1')

    # ---- eyes: huge, glowing, horizontal slit pupils
    for s in (-1, 1):
        m = np.array([s, 1, 1])
        K.eye('body', EYE * m - EYE_N * m * 0.12, EYE_N * m, EYE_R, bone='head', up=(-s * 0.78, 0, 0.62))

    # ---- beak (pow tiers: it grows, then hooked claws appear in the reared arms' suckers)
    def beak(part, sc):
        b = 0.85 + 0.15 * sc
        K.horn(part, [(0, 2.72, 1.3), (0, 2.6, 1.8 + 0.2 * b), (0, 2.25, 2.0 + 0.3 * b), (0, 1.95, 1.85 + 0.3 * b)],
               [0.36 * b, 0.3 * b, 0.16 * b, 0], mat='dark', sides=8, flat=0.85, bone='head')
        K.horn(part, [(0, 2.15, 1.2), (0, 2.1, 1.65 + 0.1 * b), (0, 2.25, 1.88 + 0.2 * b)], [0.26 * b, 0.2 * b, 0], mat='dark',
               sides=8, flat=0.85, bone='jaw')

    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        beak(part, sc)
        if t == 0:
            continue
        for k in (0, 1):
            C, R, U, T = sampled[k]
            n_h = [0, 4, 6, 8][t]
            for j in range(n_h):
                i = int(len(C) * (0.95 - j * 0.07))
                side = np.cross(T[i], U[i]) * (1 if j % 2 else -1)
                base = C[i] + U[i] * R[i] * 0.85 + side * R[i] * 0.25
                K.claw(part, base, U[i] * 0.8 + side * 0.3, R[i] * 1.0 * (0.65 + 0.35 * sc), R[i] * 0.22, curl=-T[i])
        if t >= 2:   # barbed hooks on the curled ground-arm tips
            for k in range(2, 8):
                C, R, U, T = sampled[k]
                for frac in (0.8, 0.88):
                    i = int(len(C) * frac)
                    K.claw(part, C[i] + U[i] * R[i] * 0.85, U[i], R[i] * 1.2 * (0.6 + 0.3 * sc), R[i] * 0.25, curl=-T[i])
        if t == 3:   # tusks flanking the beak
            for s in (-1, 1):
                K.horn(part, [(s * 0.6, 2.7, 2.1), (s * 0.78, 2.25, 2.55), (s * 0.66, 1.85, 2.8)], [0.15, 0.1, 0], bone='head')

    # ---- armour
    # t1: segmented carapace over the mantle dome
    for j, u in enumerate(np.linspace(0.08, 0.84, 6)):    # overlapping dorsal bands that follow the dome's curve
        c, r, t, n, x = mantle_frame(u)
        reg = S.Ell(c + n * r, (2.6, 1.5 + 0.25 * math.sin(math.pi * u), 0.78), np.stack([x, n, t], 1))
        K.plate('arm_t1', c + n * r, None, region=reg, snap=False, off=0.06 + 0.035 * (j % 2), bone=f'spine{min(5, 1 + int(u * 5))}', tris=130)
    # t2: brow casque + scutes down the reared arms
    K.plate('arm_t2', (0, 6.1, 1.0), (1.45, 0.65, 1.0), rot=(-20, 0, 0), bone='head', lames=3, lame_axis=(0, 0.3, -1), tris=300)
    for k in (0, 1):
        C, R, U, T = sampled[k]
        nm = tips[k][2]
        for j, frac in enumerate(np.linspace(0.3, 0.7, 6)):
            i = int(len(C) * frac)
            sd_ = np.cross(T[i], U[i])
            Rm = np.stack([sd_, U[i], T[i]], 1)
            reg = S.Ell(C[i] - U[i] * R[i] * 0.45, (R[i] * 1.15, R[i] * 0.85, R[i] * 0.62), Rm)
            bn = f'{nm}{min(6, int(frac * 6) + 1)}'
            K.plate('arm_t2', C[i] - U[i] * R[i], None, region=reg, snap=False, bone=bn, tris=110)
            if j % 2 == 1:
                K.spike('arm_t3', C[i] - U[i] * R[i] * 1.08, -U[i] + T[i] * 0.4, 0.4 + R[i] * 0.5, bone=bn)
    # t3: keel spikes, brow horns, armoured cheeks + spiked outer edge on the reared arms (reads from the front)
    P, N = SN['keel']
    for i, (p, n) in enumerate(zip(P, N)):
        K.spike('arm_t3', p, n - mantle_frame(0.14 + i * 0.152)[2] * 0.45, 1.0 - abs(i - 2) * 0.12, bone='spine2')
    P, N = SN['brow']
    for s, p, n in zip((-1, 1), P, N):
        K.spike('arm_t3', p, n + [s * 0.2, 0.8, -0.4], 0.9, bone='head')
    P, N = SN['browsp']
    for i, (p, n) in enumerate(zip(P, N)):
        K.spike('arm_t3', p, n + [0, 0.5, 0.6], 0.5 + 0.1 * (i % 2), bone='head')
    P, N = SN['flsp']
    for p, n in zip(P, N):
        K.spike('arm_t3', p, n + [0, 0.4, 0], 0.55, bone='spine2')
    for s in (-1, 1):   # armoured brow shields over each eye
        K.plate('arm_t3', (s * 1.15, 6.05, 2.35), (1.1, 0.55, 0.75), rot=(0, 0, s * -18), bone='head', tris=180, off=0.07, th=0.13)
    for k in (0, 1):
        C, R, U, T = sampled[k]
        nm = tips[k][2]
        so = -1 if k == 0 else 1
        for frac in np.linspace(0.18, 0.62, 5):
            i = int(len(C) * frac)
            outw = unit(np.cross(T[i], U[i]))
            if outw[0] * so < 0:
                outw = -outw
            bn = f'{nm}{min(6, int(frac * 6) + 1)}'
            K.spike('arm_t3', C[i] + outw * R[i] * 0.9, outw + U[i] * 0.35, 0.35 + R[i] * 0.45, bone=bn)

    # ---- energy: bioluminescent photophores / lines
    for nmx, rr in (('ph1', 0.15), ('ph2', 0.1)):
        P, N = SN[nmx]
        for i, (p, n) in enumerate(zip(P, N)):
            K.ball('enr_t1', p + n * 0.02, (rr - (i % 6) * 0.008,) * 3, mat='accent', seg=8, rings=5)
    for k, (C, R, U, T) in enumerate(sampled):   # t2: glowing stripe along each arm's back
        i0, i1 = int(len(C) * 0.25), int(len(C) * 0.86)
        idx = np.arange(i0, i1, 2)
        line = C[idx] - U[idx] * R[idx, None] * 0.97
        K.horn('enr_t2', line, list(np.clip(R[idx] * 0.13, 0.025, 0.07)), mat='accent', sides=5, per=1)
    P, N = SN['cry']   # t3: crystal photophores along the dome + glowing gland ring round each eye
    for i, (p, n) in enumerate(zip(P, N)):
        s = -1 if i < 3 else 1
        K.crystal('enr_t3', p - n * 0.05, n + [s * 0.3, 0, 0], 0.55 - (i % 3) * 0.08, seed=i, bone='spine3')
    P, N = SN['ring']
    for s in range(2):
        arc = P[s * 8:(s + 1) * 8] + N[s * 8:(s + 1) * 8] * 0.03
        K.horn('enr_t3', arc, [0.0, 0.06, 0.07, 0.07, 0.07, 0.06, 0.0], mat='accent', sides=5, per=2, bone='head')
    p, n = SN['tip']
    K.crystal('enr_t3', p[0] - n[0] * 0.05, n[0], 0.85, seed=9, bone='spine4')

    # ---- speed: bigger skirt fins and arm webbing
    tipdir = mantle_frame(0.9)[2]
    P, N = SN['finrear']
    for s, p, n in zip((-1, 1), P, N):
        K.frill('spd_t1', p, np.array([s, -0.35, -0.6]), tipdir, 1.1, 1.7, rays=6, spread=60, bone='spine4')
    P, N = SN['tipfin']
    for s, p, n in zip((-1, 1), P, N):
        K.fin('spd_t3', p, n + [s * 0.5, 0, 0] + tipdir * 0.5, tipdir, 1.0, 0.9, bone='spine4')
    P, N = SN['siph']
    for p, n in zip(P, N):
        K.frill('spd_t3', p, n + [0, -0.1, -0.5], (0, 1, -0.2), 0.9, 1.0, rays=4, spread=90, bone='head')
    for k in (0, 1):   # web frills on the outer side of the reared arms
        C, R, U, T = sampled[k]
        nm = tips[k][2]
        for frac in (0.48, 0.62):
            i = int(len(C) * frac)
            outw = unit(np.cross(T[i], U[i]) * (-1 if k == 0 else 1) - U[i] * 0.4)
            if outw[0] * (-1 if k == 0 else 1) < 0:
                outw = -outw
            K.frill('spd_t2', C[i] + outw * R[i] * 0.85, outw, T[i], 0.75 + R[i] * 0.4, 1.0, rays=4, spread=70,
                    bone=f'{nm}{int(frac * 6) + 1}')

    # ---- focus: extra eyes, crest, third eye
    for nmx, part, rr, d in (('eye1', 'fcs_t1', 0.15, 0.05), ('eye3a', 'fcs_t3', 0.17, 0.05), ('eye3b', 'fcs_t3', 0.12, 0.04)):
        P, N = SN[nmx]
        for s, p, n in zip((-1, 1), P, N):
            g = {'eye1': [s * 0.2, 0.2, 0.4], 'eye3a': [0, 0, 0.5], 'eye3b': [0, 0, 0.6]}[nmx]
            K.eye(part, p - n * d, n + g, rr, bone='head')
    p, n = SN['crest']
    K.leaf('fcs_t2', p[0] - n[0] * 0.05, n[0] * [1, 1, 0.3] + [0, 0, -0.2], (0, 0.2, 1), 0.85, 1.7, th=0.07, lobes=4, mat='accent', bone='head')
    p, n = SN['eye3c']
    K.eye('fcs_t3', p[0] - n[0] * 0.05, n[0] + [0, 0.1, 0.3], 0.22, bone='head')
