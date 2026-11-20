"""Hydra: three reptilian heads on long S-curved necks over one heavy quadruped body.
Game coords, faces +Z, ground y=0. Also hosts small local helpers (oriented SDF prims + a posable
reptile head) that serpent.py reuses."""
import numpy as np

from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep

DETAIL = 'scales'
UV_SCALE = 0.75

F = np.float32


# ====================================================================== helpers (shared with serpent)
def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def frame(fwd, up=(0, 1, 0)):
    """3x3 rotation whose columns are local x (right), y (up), z (forward) in world space."""
    z = unit(fwd)
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x = unit(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def rotx(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class Xf:
    """Any kaiju.sdf prim authored in a local frame and placed at world = o + sc * R @ local.
    Lets tubes keep their per-axis squash `s` on rotated parts (heads, fins ...)."""

    def __init__(self, prim, o, R, sc=1.0):
        self.prim, self.o, self.R, self.sc = prim, np.asarray(o, float), np.asarray(R, float), float(sc)
        lo, hi = np.asarray(prim.lo, float), np.asarray(prim.hi, float)
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        W = self.o + self.sc * C @ self.R.T
        self.lo, self.hi = W.min(0), W.max(0)

    def mirrored(self):
        return _MirXf(self)

    def dist(self, reg, pad=0.3):
        X, Y, Z = np.broadcast_arrays(reg.x, reg.y, reg.z)
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(float) - self.o
        Lc = (P @ self.R) / self.sc
        d = np.asarray(self.prim.dist(S.Points(Lc), pad / self.sc), float) * self.sc
        return d.reshape(X.shape).astype(F)

    def at(self, P):
        return self.dist(S.Points(np.asarray(P, float).reshape(-1, 3)))


class _MirXf(Xf):
    """X-mirrored copy of an Xf (mirror the query points instead of the prim)."""

    def __init__(self, xf):
        self.src = xf
        self.lo = np.array([-xf.hi[0], xf.lo[1], xf.lo[2]])
        self.hi = np.array([-xf.lo[0], xf.hi[1], xf.hi[2]])

    def dist(self, reg, pad=0.3):
        X, Y, Z = np.broadcast_arrays(reg.x, reg.y, reg.z)
        return self.src.dist(S.Points(np.stack([-X.ravel(), Y.ravel(), Z.ravel()], 1)), pad).reshape(X.shape)


class Head:
    """Posable reptile head. Local frame (unit scale): origin at the skull base, +Z along the snout
    (tip at z~2.3), +Y up. The jaw hinges at `hinge` and opens by `open_deg`."""
    HINGE = np.array([0.0, -0.3, 0.1])

    def __init__(self, o, fwd, sc=1.0, open_deg=20.0, up=(0, 1, 0), roll=0.0):
        self.o = np.asarray(o, float)
        R = frame(fwd, up)
        if roll:
            a = np.radians(roll)
            c, s = np.cos(a), np.sin(a)
            R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.R, self.sc, self.open = R, sc, open_deg
        self.Rj = R @ rotx(open_deg)
        self.Rm = R @ rotx(open_deg * 0.5)
        h = self.HINGE
        self.oj = self.o + sc * (R @ h) - sc * (self.Rj @ h)
        self.om = self.o + sc * (R @ h) - sc * (self.Rm @ h)

    # local -> world
    def W(self, p):
        return self.o + self.sc * (np.asarray(p, float) @ self.R.T)

    def WJ(self, p):
        return self.oj + self.sc * (np.asarray(p, float) @ self.Rj.T)

    def D(self, v):
        return np.asarray(v, float) @ self.R.T

    def DJ(self, v):
        return np.asarray(v, float) @ self.Rj.T

    def xf(self, prim, part='head'):
        if part == 'jaw':
            return Xf(prim, self.oj, self.Rj, self.sc)
        if part == 'mid':
            return Xf(prim, self.om, self.Rm, self.sc)
        return Xf(prim, self.o, self.R, self.sc)

    def mouth_prim(self, grow=0.0):
        g = grow
        return self.xf(S.Ell((0, -0.4, 1.35), (0.74 + g, 0.13 + 0.42 * np.sin(np.radians(self.open)) + g, 1.08 + g), rot=(4, 0, 0)), 'mid')

    def sculpt(self, B, snout=1.0, brow=1.0, cheek=1.0, crest=0.0, wide=1.0):
        """Adds the head + jaw masses and carves mouth / eye sockets / nostrils."""
        a = B.add
        # skull + snout
        a(self.xf(S.Ell((0, 0.15, 0.2), (0.86 * wide, 0.74, 1.0))), 0.3)
        a(self.xf(S.Tube([(0, 0.07, 0.7), (0, -0.04, 1.5 * snout), (0, -0.14, 2.2 * snout)], [0.66 * wide, 0.54 * wide, 0.4 * wide], s=(1.15, 0.7, 1))), 0.25)
        for s in (-1, 1):
            # maxillary lip ridge (tooth row), brow ridge, nostril bump, jaw muscle
            a(self.xf(S.Tube([(s * 0.46 * wide, -0.22, 0.6), (s * 0.44 * wide, -0.28, 1.4 * snout), (s * 0.27 * wide, -0.3, 2.1 * snout)], [0.2, 0.18, 0.13])), 0.15)
            a(self.xf(S.Tube([(s * 0.42, 0.6, 0.12), (s * 0.63 * wide, 0.6, 0.72), (s * 0.46 * wide, 0.4, 1.32)], [0.2 * brow, 0.25 * brow, 0.1])), 0.12)
            a(self.xf(S.Ell((s * 0.26, 0.14, 2.06 * snout), (0.18, 0.13, 0.2))), 0.1)
            a(self.xf(S.Ell((s * 0.7 * wide, -0.28, 0.32), (0.4 * cheek, 0.48, 0.55))), 0.25)
            # cheek horn knob
            a(self.xf(S.Ell((s * 0.86 * wide, -0.05, 0.0), (0.22, 0.2, 0.32))), 0.15)
        if crest:
            a(self.xf(S.Tube([(0, 0.62, 0.9), (0, 0.78, 0.2), (0, 0.7, -0.5)], [0.12, 0.2 * crest, 0.14])), 0.15)
        # lower jaw (closed-jaw coords, hinged) + chin + throat skin
        a(self.xf(S.Tube([(0, -0.55, 0.2), (0, -0.62, 1.1), (0, -0.57, 1.95 * snout)], [0.5 * wide, 0.42 * wide, 0.28 * wide], s=(1.2, 0.58, 1)), 'jaw'), 0.2)
        a(self.xf(S.Ell((0, -0.68, 1.85 * snout), (0.3, 0.17, 0.24)), 'jaw'), 0.12)
        for s in (-1, 1):
            a(self.xf(S.Ell((s * 0.45 * wide, -0.62, 0.55), (0.25, 0.25, 0.5)), 'jaw'), 0.2)
        a(self.xf(S.Tube([(0, -0.35, -0.55), (0, -0.62, 0.2)], [0.72, 0.5]), 'mid'), 0.35)
        # carve
        B.cut(self.mouth_prim(), 0.06)
        for s in (-1, 1):
            B.cut(self.xf(S.Ell((s * 0.72 * wide, 0.25, 0.86), (0.2, 0.17, 0.26))), 0.08)
            B.cut(self.xf(S.Ell((s * 0.28, 0.14, 2.32 * snout), (0.06, 0.05, 0.08))), 0.03)
        # tongue sits in the carved mouth
        B.add(self.xf(S.Tube([(0, -0.5, 0.35), (0, -0.47, 1.05), (0.05, -0.42, 1.5)], [0.2, 0.17, 0.08], s=(1.3, 0.45, 1)), 'jaw'), 0.06)
        self.snout, self.wide = snout, wide

    def dress(self, K, part='body', bone=None, jaw_bone=None, eye_r=0.14, teeth=7):
        sn, wd = self.snout, self.wide
        for s in (-1, 1):
            eye(K, part, self.W((s * 0.66 * wd, 0.27, 0.9)), self.D((s * 0.85, 0.1, 0.55)), eye_r * self.sc, bone=bone, up=self.D((0, 1, 0)))
            for i in range(teeth):
                z = (0.78 + i * 0.19) * sn
                x = s * (0.55 - i * 0.042) * wd
                ln = (0.28 - 0.01 * i + (0.12 if i in (teeth - 3, teeth - 2) else 0)) * (0.85 + 0.15 * sn)
                K.horn(part, self.W([(x, -0.26, z), (x * 1.03, -0.26 - ln * 0.6, z + 0.02), (x, -0.26 - ln, z + 0.06)]), [0.075 * self.sc, 0.05 * self.sc, 0],
                       sides=4, bone=bone, shade=(0.8, 1))
            for i in range(teeth - 1):
                z = (0.75 + i * 0.18) * sn
                x = s * (0.46 - i * 0.035) * wd
                ln = 0.22 + (0.1 if i == teeth - 3 else 0)
                K.horn(part, self.WJ([(x, -0.44, z), (x * 1.03, -0.44 + ln * 0.6, z), (x, -0.44 + ln, z - 0.03)]), [0.065 * self.sc, 0.045 * self.sc, 0],
                       sides=4, bone=jaw_bone, shade=(0.8, 1))


def eye(K, part, c, n, r, bone=None, lid=True, up=(0, 1, 0), seg=10, rings=7):
    """Low-poly version of K.eye (glowing ball + slit pupil + dark lid) with adjustable tessellation."""
    from .. import geom
    c, n = np.asarray(c, float), geom.unit(n)
    x, y, z = geom.basis(n, up)

    def blob(cc, sx, sy, sz, sg, rg, mat):
        V, T, _ = geom.sphere((0, 0, 0), 1, sg, rg)
        V = cc + (V[:, [0]] * x * sx + V[:, [1]] * y * sy + V[:, [2]] * z * sz) * r
        K.piece(part, V, T, mat, bone=bone)
    if lid:
        blob(c - n * r * 0.35, 1.3, 0.75, 1.12, 8, 6, 'dark')
    blob(c, 1, 1, 1, seg, rings, 'eye')
    blob(c + n * r * 0.72, 0.2, 0.34, 0.62, 6, 5, 'dark')


def neck_frames(pts):
    """Per-point tangent + 'throat' (front/under) direction for a neck/body spline."""
    P = np.asarray(pts, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    h = P[-1] - P[0]
    h[1] = 0
    lat = unit(np.cross([0, 1, 0], unit(h) if np.linalg.norm(h) > 1e-6 else [0, 0, 1]))
    Th = np.cross(lat[None], T)
    Th /= np.linalg.norm(Th, axis=1, keepdims=True)
    return T, Th, lat


# ====================================================================== hydra
TAIL = [(0, 4.9, -3.4), (0, 4.2, -5.0), (0.3, 3.1, -6.6), (1.0, 2.0, -8.0), (2.1, 1.2, -9.1), (3.4, 0.82, -9.7), (4.6, 0.78, -9.8)]
TAIL_R = [1.6, 1.32, 1.02, 0.76, 0.5, 0.3, 0.1]

# (name, spline pts, radii, head origin, head fwd, head scale, jaw open, roll)
NECKS = [
    ('A', [(0, 6.4, 1.3), (0, 7.7, 2.0), (0, 8.9, 1.75), (0, 10.0, 1.8), (0, 10.85, 2.45)], [0.95, 0.8, 0.7, 0.62, 0.58],
     (0, 11.05, 2.85), (0, 0.55, 1), 1.0, 40, 0),
    ('B', [(-1.2, 6.2, 1.5), (-2.15, 6.95, 2.55), (-2.65, 6.7, 3.75), (-2.85, 5.85, 4.8), (-2.9, 5.15, 5.7)], [0.9, 0.75, 0.66, 0.6, 0.55],
     (-2.92, 5.0, 6.02), (0.02, -0.12, 1), 0.9, 26, 0),
    ('C', [(1.2, 6.2, 1.5), (2.2, 7.45, 1.75), (2.95, 8.35, 1.5), (3.25, 9.2, 1.85), (3.3, 9.65, 2.6)], [0.9, 0.75, 0.66, 0.6, 0.55],
     (3.32, 9.72, 2.92), (0.35, -0.08, 1), 0.9, 14, -8),
]


def build(K):
    B = K.sdf()
    # ---- torso: heavy barrel, chest, hips, neck root mound
    B.tube([(0, 4.7, -3.0), (0, 5.0, -1.2), (0, 5.5, 0.6), (0, 5.85, 1.6)], [1.9, 2.2, 2.3, 2.0], k=0.0, s=(1.1, 0.95, 1))
    B.ell((0, 4.25, -0.6), (2.0, 1.6, 2.6), k=0.5)                     # belly sag
    B.ell((0, 5.6, 1.4), (2.0, 1.9, 1.7), k=0.5)                       # chest
    B.ell((0, 5.0, -2.6), (1.9, 1.75, 1.7), k=0.5)                     # hips
    B.ell((0, 6.55, 1.45), (1.85, 1.05, 1.35), k=0.55)                 # neck root mound
    B.ell((1.95, 5.55, 1.3), (1.0, 1.15, 1.1), k=0.4, mirror=True)     # shoulders
    B.ell((1.95, 4.5, -2.4), (1.05, 1.5, 1.35), k=0.4, mirror=True)    # thighs
    B.ell((1.2, 6.6, -1.2), (1.1, 0.8, 1.9), k=0.5, mirror=True)       # back muscle
    # ---- legs (sturdy, slightly sprawled)
    B.tube([(2.1, 5.2, 1.5), (2.6, 3.7, 1.8), (2.7, 2.6, 1.75)], [1.0, 0.85, 0.7], k=0.35, mirror=True)
    B.ell((2.75, 2.65, 1.7), (0.62, 0.62, 0.6), k=0.2, mirror=True)
    B.ell((2.55, 4.0, 1.25), (0.75, 1.15, 0.7), rot=(-12, 0, 0), k=0.3, mirror=True)     # triceps
    B.ell((2.85, 3.9, 2.05), (0.55, 0.9, 0.55), rot=(10, 0, 0), k=0.3, mirror=True)      # biceps
    B.ell((2.8, 1.9, 1.85), (0.62, 0.7, 0.62), k=0.25, mirror=True)                       # forearm mass
    B.tube([(2.7, 2.6, 1.75), (2.66, 1.45, 1.95), (2.62, 0.62, 2.15)], [0.68, 0.6, 0.6], k=0.2, mirror=True)
    B.tube([(1.95, 4.9, -2.4), (2.5, 3.4, -1.65), (2.62, 2.55, -1.5)], [1.25, 1.0, 0.75], k=0.35, mirror=True)
    B.ell((2.66, 2.6, -1.42), (0.66, 0.64, 0.64), k=0.2, mirror=True)
    B.ell((2.6, 1.95, -2.0), (0.62, 0.85, 0.66), k=0.25, mirror=True)                     # calf
    B.tube([(2.62, 2.55, -1.5), (2.56, 1.5, -1.75), (2.52, 0.62, -1.6)], [0.76, 0.66, 0.62], k=0.2, mirror=True)
    for fz in (2.15, -1.6):
        fx = 2.62 if fz > 0 else 2.52
        B.ell((fx, 0.42, fz + 0.3), (0.78, 0.42, 0.9), k=0.25, mirror=True)
        for dx in (-0.45, 0.0, 0.45):
            B.tube([(fx + dx * 0.6, 0.4, fz + 0.6), (fx + dx * 0.9, 0.32, fz + 1.0), (fx + dx * 1.05, 0.24, fz + 1.3)], [0.3, 0.26, 0.2], k=0.12, mirror=True)
        B.tube([(fx, 0.55, fz - 0.4), (fx, 0.3, fz - 0.8)], [0.32, 0.18], k=0.15, mirror=True)
    # ---- tail
    B.tube(TAIL, TAIL_R, k=0.5, s=(0.95, 1.05, 1))
    # ---- necks + heads
    heads = {}
    for nm, pts, rr, ho, hf, hs, op, roll in NECKS:
        B.tube(pts, rr, k=0.45)
        H = Head(ho, hf, sc=hs, open_deg=op, roll=roll)
        H.sculpt(B, snout=1.0 if nm != 'B' else 1.08, brow=1.15, crest=0.0)
        heads[nm] = H

    # ---- belly / throat mask
    belly = K.sdf()
    belly.tube([(0, 2.75, -2.9), (0, 2.6, -0.6), (0, 3.4, 1.7), (0, 4.6, 2.9), (0, 5.9, 3.1), (0, 6.9, 2.6)], [1.0, 1.45, 1.45, 1.2, 0.95, 0.8], k=0)
    for sx in (-1, 1):   # chest bib forks up into the side throats
        belly.tube([(0, 6.2, 3.0), (sx * 0.9, 6.7, 2.7), (sx * 1.6, 7.0, 2.6)], [0.8, 0.65, 0.55], k=0)
    tc = S.catmull(TAIL, 3)
    tr = S.catmull(TAIL_R, 3)[:, 0]
    belly.tube(tc[:-2] - np.stack([0 * tr[:-2], tr[:-2] * 1.0, 0 * tr[:-2]], 1), tr[:-2] * 0.62, k=0)
    glands = []
    for nm, pts, rr, ho, hf, hs, op, roll in NECKS:
        C = S.catmull(pts, 3)
        Cr = S.catmull(rr, 3)[:, 0]
        T, Th, lat = neck_frames(C)
        belly.tube(C + Th * Cr[:, None] * 0.95, Cr * 0.58, k=0)
        H = heads[nm]
        belly.add(H.xf(S.Tube([(0, -0.75, 0.0), (0, -0.85, 0.9), (0, -0.78, 1.6)], [0.38, 0.3, 0.16]), 'jaw'), 0)
        belly.add(H.xf(S.Ell((0, -0.75, -0.3), (0.45, 0.4, 0.6)), 'mid'), 0)
        for s in (-1, 1):   # row of glossy acid glands on the throat sides, just behind the jaw
            for j, f in enumerate((0.84, 0.74, 0.64)):
                i = int(f * (len(C) - 1))
                d = unit(lat * s * 0.8 + Th[i] * 0.6)
                rg = 0.25 - 0.04 * j
                glands.append(S.Ell(C[i] + d * (Cr[i] + rg * 0.05), (rg, rg * 1.15, rg)))

    def skin(x, y, z):
        rough = -0.1 * (fbm(x, y, z, 3, freq=0.9, seed=7, ridged=True) - 0.55) - 0.03 * fbm(x, y, z, 2, freq=3.1, seed=11)
        w = smoothstep(0.12, -0.12, belly.at(np.stack([x, y, z], 1)))
        band = np.abs(np.sin(np.pi * (y * 0.8 + z * 0.6) / 0.55))
        scute = 0.055 * (1 - smoothstep(0.0, 0.3, band)) - 0.02
        return rough * (1 - w) + scute * w
    B.disp(skin, band=0.16)
    for g in glands:     # glossy gland bulges (added after the scale sculpt so they stay smooth)
        B.add(g, 0.06)

    mouths = [heads[n].mouth_prim(0.04) for n in heads]

    def mouth_fn(P):
        return np.min([m.at(P) for m in mouths], 0)

    def gland_fn(P):
        pp = S.Points(P)
        return np.min([g.dist(pp) - 0.02 for g in glands], 0)

    K.skin(B, voxel=0.05, tris=15500, belly=belly, regions=[('mouth', mouth_fn), ('accent', gland_fn)])

    # ---- skeleton
    K.bone('pelvis', (0, 5.0, -2.8), (0, 5.2, -0.8), None, 2.0)
    K.bone('spine', (0, 5.2, -0.8), (0, 5.6, 0.8), 'pelvis', 2.1)
    K.bone('chest', (0, 5.6, 0.8), (0, 6.2, 1.5), 'spine', 1.9)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = np.array([s, 1, 1])
        K.bone('upperarm' + sd, m * (2.1, 5.2, 1.5), m * (2.7, 2.6, 1.75), 'chest', 0.95)
        K.bone('forearm' + sd, m * (2.7, 2.6, 1.75), m * (2.62, 0.6, 2.15), 'upperarm' + sd, 0.65)
        K.bone('hand' + sd, m * (2.62, 0.6, 2.15), m * (2.62, 0.3, 3.45), 'forearm' + sd, 0.45)
        K.bone('thigh' + sd, m * (1.95, 4.9, -2.4), m * (2.62, 2.55, -1.5), 'pelvis', 1.15)
        K.bone('shin' + sd, m * (2.62, 2.55, -1.5), m * (2.52, 0.6, -1.6), 'thigh' + sd, 0.7)
        K.bone('foot' + sd, m * (2.52, 0.6, -1.6), m * (2.52, 0.3, -0.3), 'shin' + sd, 0.45)
        K.mount('hand' + sd, m * (2.62, 0.4, 2.9), 'hand' + sd)
        K.mount('shoulder' + sd, m * (2.05, 6.6, 1.2), 'chest')
    K.chain('tail', TAIL, 'pelvis', (1.5, 0.3))
    for nm, pts, rr, ho, hf, hs, op, roll in NECKS:
        H = heads[nm]
        last = K.chain('neck' + nm, list(pts) + [list(H.W((0, 0, -0.1)))], 'chest', (rr[0] * 1.05, rr[-1] * 1.1))
        K.bone('head' + nm, H.W((0, 0, -0.1)), H.W((0, -0.05, 2.1)), last, 0.75 * hs)
        K.bone('jaw' + nm, H.WJ((0, -0.45, 0.25)), H.WJ((0, -0.6, 1.9)), 'head' + nm, 0.42 * hs)
    A = heads['A']
    K.mount('head', A.W((0, 0.1, 0.3)), 'headA')
    K.mount('jaw', A.WJ((0, -0.55, 0.5)), 'jawA')
    K.mount('mouth', A.W((0, -0.4, 2.3)), 'headA')
    K.mount('crown', A.W((0, 0.95, 0.35)), 'headA')
    K.mount('chest', (0, 5.4, 3.15), 'chest')
    K.mount('back', (0, 7.4, -1.0), 'spine')
    K.mount('tailTip', TAIL[-1], 'tail6')

    # ---- eyes, teeth, glowing throat (base)
    for nm in heads:
        H = heads[nm]
        H.dress(K, 'body', bone='head' + nm, jaw_bone='jaw' + nm)
        K.ball('body', H.W((0, -0.42, 0.55)), 0.24 * H.sc, mat='accent', bone='head' + nm)   # acid glow down the throat

    # ---- dorsal row: back + tail + along each neck
    row = np.vstack([[(0, 7.25, 0.2), (0, 7.4, -0.7), (0, 7.3, -1.6), (0, 7.0, -2.5), (0, 6.55, -3.3)],
                     tc[2:13] + np.stack([0 * tr[2:13], tr[2:13] * 1.1, 0 * tr[2:13]], 1)])
    hs_ = [0.75, 0.95, 1.05, 1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.6, 0.55, 0.48, 0.42, 0.36, 0.3, 0.24]
    P, N = K.snap(row)
    for i, (p, n, h) in enumerate(zip(P, N, hs_)):
        t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        n = n * [0.25, 1, 1]
        K.leaf('body', p - n * 0.05, n, -t, h, h * 0.9, th=0.08 + h * 0.05, lobes=3, seed=i)
        if i < 9:
            K.leaf('enr_t2', p - n * 0.05, n, -t, h, h * 0.9, th=0.08 + h * 0.05, lobes=3, core=True, mat='accent')
    for nm, pts, rr, ho, hf, hs, op, roll in NECKS:
        C = S.catmull(pts, 3)
        Cr = S.catmull(rr, 3)[:, 0]
        T, Th, lat = neck_frames(C)
        idx = list(range(2, len(C) - 1, 2))
        Q, QN = K.snap(C - Th * Cr[:, None] * 1.1)
        for j, i in enumerate(idx):
            h = 0.5 + 0.25 * np.sin(np.pi * (j + 0.5) / len(idx))
            K.leaf('body', Q[i] - QN[i] * 0.05, QN[i], -T[i], h, h * 0.85, th=0.07, lobes=3, seed=20 + j, bone=None)
            if j % 2 == 0:
                K.leaf('enr_t3', Q[i] - QN[i] * 0.05, QN[i], -T[i], h, h * 0.85, th=0.07, lobes=3, core=True, mat='accent')

    # ---- acid glands (Energy t1): glowing sacs bulging on each neck; t3 crystal clusters on back
    for nm, pts, rr, ho, hf, hs, op, roll in NECKS:
        C = S.catmull(pts, 3)
        Cr = S.catmull(rr, 3)[:, 0]
        T, Th, lat = neck_frames(C)
        for f in (0.3, 0.45):
            i = int(f * (len(C) - 1))
            for s in (-1, 1):
                p, n = K.snap(C[i] + lat * s * Cr[i] + Th[i] * Cr[i] * 0.3)
                K.ball('enr_t1', p - n * 0.06, (0.2, 0.2, 0.2), mat='accent', bone='neck' + nm + '2', seg=8, rings=6)
    for i, (p, n) in enumerate(zip(P[1:6], N[1:6])):
        for s in (-1, 1):
            q, qn = K.snap(p + np.array([s * 1.1, -0.35, 0]))
            K.crystal('enr_t3', q - qn * 0.08, qn + [s * 0.3, 0.4, 0], 0.55 + 0.1 * (i % 2), seed=i * 2 + (s > 0))

    # ---- claws per Might tier + horns on every head from tier 1
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        kk = 0.75 + 0.25 * sc
        for s, sd in ((-1, 'L'), (1, 'R')):
            for fz, bone in ((2.15, 'hand'), (-1.6, 'foot')):
                fx = 2.62 if fz > 0 else 2.52
                for dx in (-0.45, 0.0, 0.45):
                    K.claw(part, (s * (fx + dx * 1.05), 0.22, fz + 1.32), (s * dx * 0.3, 0.05, 1), 0.6 * kk, 0.17 * (0.8 + 0.2 * sc), bone=bone + sd)
        if t >= 1:
            hl = [0, 0.85, 1.2, 1.55][t]
            for nm in heads:
                H = heads[nm]
                for s in (-1, 1):
                    K.horn(part, H.W([(s * 0.6, 0.6, 0.15), (s * 0.82, 0.68 + hl * 0.25, -0.1 - hl * 0.3), (s * 0.9, 0.75 + hl * 0.5, -0.4 - hl * 0.7), (s * 0.8, 0.9 + hl * 0.7, -0.5 - hl * 1.1)]),
                           np.array([0.19, 0.16, 0.09, 0]) * H.sc, bone='head' + nm, per=2, sides=6)
                    K.horn(part, H.W([(s * 0.86, -0.05, -0.05), (s * (1.0 + hl * 0.2), -0.1, -0.35 - hl * 0.2), (s * (1.05 + hl * 0.3), -0.2, -0.6 - hl * 0.35)]),
                           np.array([0.12, 0.08, 0]) * H.sc, bone='head' + nm)
                if t >= 3:
                    K.horn(part, H.W([(0, 0.32, 1.95), (0, 0.6, 2.0), (0, 0.85, 1.85)]), np.array([0.13, 0.09, 0]) * H.sc, bone='head' + nm)

    # ---- armour tiers
    spots = {
        1: [((2.05, 6.4, 1.2), (1.0, 0.9, 1.0), 'chest', 3), ((2.55, 4.4, -2.0), (0.85, 1.15, 1.05), 'thigh', 3)],
        2: [((2.85, 3.5, 1.9), (0.55, 0.75, 0.55), 'upperarm', 2), ((2.7, 1.8, -1.45), (0.55, 0.75, 0.5), 'shin', 2),
            ((1.45, 7.0, -1.3), (0.85, 0.9, 1.3), 'spine', 3)],
    }
    for tier, lst in spots.items():
        for c, r, bone, lames in lst:
            for s, sd in ((-1, 'L'), (1, 'R')):
                bn = bone + sd if bone in ('thigh', 'upperarm', 'shin') else bone
                cc = np.array([s * c[0], c[1], c[2]])
                K.plate(f'arm_t{tier}', cc, r, bone=bn, lames=lames, tris=200)
                if tier == 1:
                    p, n = K.snap(cc)
                    K.spike('arm_t3', p + n * 0.1, n + np.array([0, 0.45, -0.25]), 0.8, bone=bn)
                    K.spike('arm_t3', p + n * 0.1 + [0, -0.45, 0.25], n + np.array([0, 0.1, 0.1]), 0.45, bone=bn)
    for nm in heads:   # skull caps on every head
        H = heads[nm]
        K.plate('arm_t2', H.W((0, 0.78, 0.65)), np.array([0.62, 0.42, 0.75]) * H.sc * 1.1, bone='head' + nm, lames=0, tris=200)
        p, n = K.snap(H.W((0, 0.8, 0.2)))
        K.spike('arm_t3', p + n * 0.05, n + H.D((0, 0.2, -0.6)), 0.55 * H.sc, bone='head' + nm)
    for nm, pts, rr, ho, hf, hs, op, roll in NECKS:   # neck collar plates (t2) + spikes (t3)
        C = S.catmull(pts, 3)
        T, Th, lat = neck_frames(C)
        i = int(0.2 * (len(C) - 1))
        K.plate('arm_t2', C[i] - Th[i] * 0.55, (0.95, 0.75, 0.95), bone='neck' + nm + '1', lames=2, lame_axis=T[i], tris=220)
        for s in (-1, 1):
            p, n = K.snap(C[i] + lat * s * 0.8 - Th[i] * 0.3)
            K.spike('arm_t3', p + n * 0.08, n - Th[i] * 0.3, 0.5, bone='neck' + nm + '1')
    for i in (10, 11, 12, 13):   # tail spikes
        for s in (-1, 1):
            p, n = K.snap(P[i] + np.array([s * 0.5, -0.3, 0]))
            K.spike('arm_t3', p, n * [1, 0.3, 1] + [s * 0.6, 0.25, 0], 0.7 - (i - 10) * 0.1)
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.spike('arm_t3', (s * 3.25, 2.75, 1.85), (s * 1, 0.2, 0.4), 0.55, bone='forearm' + sd)
        K.spike('arm_t3', (s * 3.2, 2.7, -1.35), (s * 1, 0.2, -0.2), 0.55, bone='shin' + sd)

    # ---- speed tiers: frills
    for s, sd in ((-1, 'L'), (1, 'R')):
        p, n = K.snap((s * 3.25, 1.6, 1.8))
        K.frill('spd_t1', p, n * [1, 0.2, 0.2] + [0, 0.2, -0.6], (0, 1, -0.3), 0.85, 0.8, rays=4, bone='forearm' + sd)
        p, n = K.snap((s * 3.1, 1.6, -1.95))
        K.frill('spd_t1', p, n * [1, 0.2, 0.2] + [0, 0.2, -0.6], (0, 1, 0.2), 0.8, 0.8, rays=4, bone='shin' + sd)
    for i in (6, 8, 10):   # tail fins (t2)
        for s in (-1, 1):
            p, n = K.snap(P[i] + np.array([s * 0.9, -0.2, 0]))
            t = P[i + 1] - P[i - 1]
            K.frill('spd_t2', p, n * [1, 0.2, 1] + [s * 0.4, 0.1, 0], -t, 0.95 - (i - 6) * 0.1, 1.0, rays=4)
    for nm in heads:   # head frills (t3)
        H = heads[nm]
        for s in (-1, 1):
            K.frill('spd_t3', H.W((s * 0.85, 0.1, -0.15)), H.D((s * 1, 0.3, -0.75)), H.D((0, 1, 0.1)), 1.1 * H.sc, 1.0 * H.sc, rays=5, spread=120, bone='head' + nm)

    # ---- focus tiers: extra eyes, crest, third eye (every head)
    for nm in heads:
        H = heads[nm]
        for s in (-1, 1):
            eye(K, 'fcs_t1', H.W((s * 0.58, 0.47, 1.36)), H.D((s * 0.7, 0.5, 0.5)), 0.08 * H.sc, bone='head' + nm, lid=False, seg=8, rings=6)
            eye(K, 'fcs_t3', H.W((s * 0.86, 0.02, 0.55)), H.D((s * 1, 0, 0.3)), 0.085 * H.sc, bone='head' + nm, lid=False, seg=8, rings=6)
        K.leaf('fcs_t2', H.W((0, 0.72, 0.4)), H.D((0, 1, -0.1)), H.D((0, 0, 1)), 0.65 * H.sc, 1.3 * H.sc, th=0.05, lobes=3, mat='accent', bone='head' + nm)
        eye(K, 'fcs_t3', H.W((0, 0.66, 1.05)), H.D((0, 0.6, 0.8)), 0.11 * H.sc, bone='head' + nm, lid=False, seg=8, rings=6)
