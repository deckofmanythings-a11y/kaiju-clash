"""The Litharch: a secret Wild-only easter-egg encounter. A floating crystalline geometric being, ~9.5 units tall,
hovering with its lowest point ~1.5 units above the ground, faces +Z.

Design: a hexagonal trapezohedron-style hull of dark amethyst-obsidian facets, SPLIT at the equator. The upper half
(crown hull, tall) and the lower half (keel hull, short) float apart and expose a molten amber inner core: a twisted
faceted trapezohedron heart pierced by a six-point star of crystal spindles. A stepped hexagonal bezel on the core
holds the camera-iris aperture (facing +Z) that fires the beam. Every hull facet is terraced: brass edge inlay band,
an engraved groove, a raised terrace and a glowing inner groove around a low central pyramid. Six satellite shards
of varying size orbit the equator and a string of three crystal beads dangles under the keel apex.

The SDF skin (`body`) is the core assembly (heart + star spindles + bezel), built from exact convex-polyhedron SDFs so
it stays hard-edged; the hull halves, shards and beads are flat-shaded rigid pieces merged into the body mesh.
"""
import itertools

import numpy as np

from .. import geom as G
from .. import sdf as S

INFO = dict(name='The Litharch', colors=dict(body='#3a2150', belly='#c8923c', accent='#ff8a1e'))
DETAIL = 'smooth'
UV_SCALE = 0.75
EYE_ACCENT = True
CAM = dict(az=32, el=10)

C = np.array([0.0, 6.0, 0.0])          # core centre
Y = np.array([0.0, 1.0, 0.0])
APZ = 2.3                              # bezel front plane (z)

# hull rings: (radius, y, angle offset deg)
UPPER = [(3.1, 7.0, 0.0), (2.5, 8.5, 30.0), (1.3, 9.95, 0.0)]
UPPER_APEX = 11.0
LOWER = [(2.9, 5.0, 30.0), (2.0, 3.9, 0.0)]
LOWER_APEX = 2.75

# terrace profiles: (scale about the incentre, depth along the face normal, material of the segment ending here)
P_FACE = [(1.0, 0.0, None), (0.84, 0.0, 'belly'), (0.835, -0.07, 'belly'), (0.77, -0.07, 'dark'), (0.765, 0.0, 'body'),
          (0.6, 0.0, 'body'), (0.595, 0.05, 'body'), (0.43, 0.05, 'body'), (0.425, -0.03, 'body'), (0.35, -0.03, 'accent'),
          (0.345, 0.07, 'body'), (0.0, 0.12, 'body')]
GROOVE1 = (0.833, 0.772, -0.066)       # (outer scale, inner scale, depth) of the dark groove (enr inlay lives here)
P_CAP = [(1.0, 0.0, None), (0.9, 0.0, 'belly'), (0.895, -0.16, 'belly'), (0.78, -0.16, 'body'), (0.775, -0.32, 'body'),
         (0.64, -0.32, 'belly'), (0.635, -0.46, 'belly'), (0.5, -0.46, 'accent'), (0.0, -0.46, 'dark')]
P_SHARD = [(1.0, 0.0, None), (0.8, 0.0, 'belly'), (0.795, -0.03, 'belly'), (0.55, -0.03, 'body'), (0.0, 0.04, 'body')]
P_PLATE = [(1.0, -0.12, None), (0.995, 0.0, 'plate'), (0.86, 0.0, 'belly'), (0.855, -0.04, 'belly'), (0.7, -0.04, 'plate'),
           (0.695, 0.03, 'plate'), (0.4, 0.03, 'plate'), (0.395, -0.02, 'plate'), (0.32, -0.02, 'accent'), (0.0, 0.06, 'plate')]


# ------------------------------------------------------------------ math helpers
def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def ang(a_deg, R, y):
    a = np.radians(a_deg)
    return np.array([R * np.cos(a), y, R * np.sin(a)])


def rot_y(v, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    v = np.asarray(v, float)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]) if v.ndim == 1 else \
        np.stack([c * v[:, 0] + s * v[:, 2], v[:, 1], -s * v[:, 0] + c * v[:, 2]], 1)


def incenter(P):
    P = np.asarray(P, float)
    if len(P) != 3:
        return P.mean(0)
    a, b, c = np.linalg.norm(P[1] - P[2]), np.linalg.norm(P[0] - P[2]), np.linalg.norm(P[0] - P[1])
    return (a * P[0] + b * P[1] + c * P[2]) / (a + b + c)


def hull_planes(V):
    """Brute-force convex hull planes (outward normal n, offset d) of a small point set."""
    V = np.asarray(V, float)
    planes = []
    for i, j, k in itertools.combinations(range(len(V)), 3):
        n = np.cross(V[j] - V[i], V[k] - V[i])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        d = n @ V[i]
        s = V @ n - d
        if np.all(s <= 1e-6):
            pass
        elif np.all(s >= -1e-6):
            n, d = -n, -d
        else:
            continue
        if not any(np.allclose(n, m, atol=1e-5) and abs(d - e) < 1e-5 for m, e in planes):
            planes.append((n, d))
    return planes


class Poly:
    """Exact-inside convex polyhedron SDF (max over face planes): stays razor sharp when meshed."""

    def __init__(self, V):
        self.V = np.asarray(V, float)
        self.planes = hull_planes(self.V)
        self.lo, self.hi = self.V.min(0), self.V.max(0)

    def mirrored(self):
        return Poly(self.V * [-1, 1, 1])

    def dist(self, reg, pad=0):
        out = None
        F = np.float32
        for n, d in self.planes:
            v = reg.x * F(n[0]) + reg.y * F(n[1]) + reg.z * F(n[2]) - F(d)
            out = v if out is None else np.maximum(out, v)
        return np.array(np.broadcast_to(out, reg.shape), F)


# ------------------------------------------------------------------ mesh accumulation
class Acc:
    """Collects triangles per material, then emits one K.piece per material."""

    def __init__(self):
        self.d = {}

    def tri(self, mat, a, b, c, n, col):
        V, T, Cc = self.d.setdefault(mat, ([], [], []))
        a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
        fn = np.cross(b - a, c - a)
        if np.linalg.norm(fn) < 1e-10:
            return
        if fn @ n < 0:
            b, c = c, b
        o = len(V)
        V += [a, b, c]
        T.append((o, o + 1, o + 2))
        Cc += list(np.broadcast_to(col, (3,)))

    def emit(self, K, part, bone):
        for mat, (V, T, Cc) in self.d.items():
            if T:
                K.piece(part, np.array(V), np.array(T), mat=mat, smooth=False, col=np.clip(np.array(Cc), 0.05, 1.0), bone=bone)
        self.d = {}


def terrace(acc, poly, n, prof, shade=1.0, center=None):
    """Concentric stepped terraces on one planar convex face (heightfield over the face, so winding = +n)."""
    poly = np.asarray(poly, float)
    n = unit(n)
    c = incenter(poly) if center is None else np.asarray(center, float)
    k = len(poly)
    prev = None
    for s, dep, mat in prof:
        ring = c + (poly - c) * s + n * dep
        if prev is not None:
            ps, pd, pr = prev
            g = shade * (0.62 if dep < -0.01 and pd < -0.01 else 1.0) * (1.0 if mat != 'accent' else 1.0)
            for i in range(k):
                j = (i + 1) % k
                if s <= 1e-6:
                    acc.tri(mat, pr[i], pr[j], ring[0], n, g)
                else:
                    acc.tri(mat, pr[i], pr[j], ring[j], n, g)
                    acc.tri(mat, pr[i], ring[j], ring[i], n, g)
        prev = (s, dep, ring)


def ngon(c, n, R, sides=6, rot=0.0, t=None, sx=1.0):
    x, _, z = G.basis(n, t)
    a = np.radians(rot) + np.arange(sides) * 2 * np.pi / sides
    return np.asarray(c, float) + R * (np.cos(a)[:, None] * x * sx + np.sin(a)[:, None] * z)


def ring_piece(acc, c, n, r_out, r_in, th, mat='belly', sides=6, rot=0.0, t=None, wall_mat=None, shade=1.0):
    """Flat polygonal frame ring (lens ring / bezel) facing n, with thickness th backwards."""
    n = unit(n)
    O = ngon(c, n, r_out, sides, rot, t)
    I = ngon(c, n, r_in, sides, rot, t)
    Ob, Ib = O - n * th, I - n * th
    wm = wall_mat or mat
    cc = np.asarray(c, float)
    for i in range(sides):
        j = (i + 1) % sides
        acc.tri(mat, O[i], O[j], I[j], n, shade)
        acc.tri(mat, O[i], I[j], I[i], n, shade)
        acc.tri(mat, Ob[i], Ib[j], Ob[j], -n, shade * 0.6)
        acc.tri(mat, Ob[i], Ib[i], Ib[j], -n, shade * 0.6)
        mo = unit((O[i] + O[j]) * 0.5 - cc - n * ((O[i] + O[j]) * 0.5 - cc) @ n)
        acc.tri(wm, O[i], Ob[i], Ob[j], mo, shade * 0.8)
        acc.tri(wm, O[i], Ob[j], O[j], mo, shade * 0.8)
        acc.tri(wm, I[i], Ib[j], Ib[i], -mo, shade * 0.5)
        acc.tri(wm, I[i], I[j], Ib[j], -mo, shade * 0.5)


def disc(acc, c, n, R, mat, sides=12, rot=0.0, shade=1.0):
    P = ngon(c, n, R, sides, rot)
    for i in range(sides):
        acc.tri(mat, c, P[i], P[(i + 1) % sides], n, shade)


def iris(acc, c, n, r_out, r_open, mat='dark', blades=6, rot=0.0):
    """Camera-iris aperture: glowing lens disc behind six overlapping dark blades leaving a hexagonal opening."""
    n = unit(n)
    c = np.asarray(c, float)
    disc(acc, c, n, r_out * 1.02, 'eye', 12)
    x, _, z = G.basis(n)
    for i in range(blades):
        a0 = np.radians(rot + i * 360 / blades)
        a1 = np.radians(rot + (i + 1) * 360 / blades)
        dz = n * (0.012 + 0.006 * i)
        p0 = c + dz + r_open * (np.cos(a0) * x + np.sin(a0) * z)
        p1 = c + dz + r_open * (np.cos(a1) * x + np.sin(a1) * z)
        q1 = c + dz + r_out * (np.cos(a1 + 0.55) * x + np.sin(a1 + 0.55) * z)
        q0 = c + dz + r_out * (np.cos(a0 + 0.55) * x + np.sin(a0 + 0.55) * z)
        g = 0.75 + 0.25 * (i % 2)
        acc.tri(mat, p0, p1, q1, n, g)
        acc.tri(mat, p0, q1, q0, n, g)


def lens_eye(K, part, c, n, r, bone):
    """Small secondary aperture: brass hex bezel + glowing iris lens, sitting proud of a facet."""
    acc = Acc()
    n = unit(n)
    ring_piece(acc, c, n, r * 1.35, r * 0.98, 0.12, 'belly', 6, 0, wall_mat='horn')
    disc(acc, c - n * 0.1, n, r * 1.0, 'dark', 6)
    iris(acc, c - n * 0.06, n, r * 0.98, r * 0.32, rot=15)
    acc.emit(K, part, bone)


# ------------------------------------------------------------------ hull
def hull_half(upper):
    """-> faces [(poly(3,3), outward normal, meta)], cap (poly(6,3), normal), vertex rings."""
    rings = UPPER if upper else LOWER
    apex = np.array([0, UPPER_APEX if upper else LOWER_APEX, 0.0])
    R = [np.array([ang(a0 + 60 * k, r, y) for k in range(6)]) for r, y, a0 in rings]
    ctr = np.array([0, 7.0 + 1.5 if upper else 5.0 - 1.0, 0.0])
    faces = []

    def add(p, band):
        p = np.asarray(p)
        n = unit(np.cross(p[1] - p[0], p[2] - p[0]))
        if n @ (p.mean(0) - ctr) < 0:
            n = -n
        faces.append((p, n, band))
    for b in range(len(R) - 1):
        A, Bn = R[b], R[b + 1]
        # ring b+1 is rotated by +30 deg relative to ring b: B_k sits between A_k and A_k+1
        for k in range(6):
            add([A[k], A[(k + 1) % 6], Bn[k]], b)
            add([Bn[k], Bn[(k + 1) % 6], A[(k + 1) % 6]], b)
    for k in range(6):
        add([R[-1][k], R[-1][(k + 1) % 6], apex], len(R) - 1)
    cap = (R[0], np.array([0, -1.0 if upper else 1.0, 0]))
    return faces, cap, R, apex


def edges_of(faces, cap):
    E = {}
    polys = [f[0] for f in faces] + [cap[0]]
    for p in polys:
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            key = tuple(sorted([tuple(np.round(a, 4)), tuple(np.round(b, 4))]))
            E[key] = (a, b)
    return list(E.values())


def build_hull(K, upper, rng):
    faces, cap, R, apex = hull_half(upper)
    bone = 'chest' if upper else 'spine'
    acc = Acc()
    for p, n, band in faces:
        terrace(acc, p, n, P_FACE, shade=rng.uniform(0.8, 1.0) * (0.92 if band == 0 else 1.0))
    terrace(acc, cap[0], cap[1], P_CAP, shade=0.85, center=cap[0].mean(0))
    acc.emit(K, 'body', bone)
    for a, b in edges_of(faces, cap):     # bevelled brass edge rods (the inlay strips meet on them)
        K.horn('body', [a, b], [0.045, 0.045], mat='belly', sides=4, per=1, bone=bone, shade=(0.85, 1.0))
    return faces, cap, R, apex


def groove_inlay(K, part, faces, bone):
    """Glowing accent inlay filling groove 1 of the given faces (Energy tiers)."""
    acc = Acc()
    s0, s1, dep = GROOVE1
    for p, n, _ in faces:
        terrace(acc, p, n, [(s0, dep, None), (s1, dep, 'accent')])
    acc.emit(K, part, bone)


def shield(K, part, poly, n, off, bone, shrink=0.78):
    """Floating armour facet: a terraced plate parallel to a hull face, hovering `off` out from it."""
    poly = np.asarray(poly, float)
    c = incenter(poly)
    p2 = c + (poly - c) * shrink + n * off
    acc = Acc()
    terrace(acc, p2, n, P_PLATE, shade=0.95)
    back = p2 - n * 0.12
    terrace(acc, back, -n, [(1.0, 0.0, None), (0.0, 0.0, 'plate')], shade=0.5, center=incenter(back))
    acc.emit(K, part, bone)
    return incenter(p2) + n * 0.06


def pick(faces, d, band=None, used=()):
    d = unit(d)
    best, bi = -9, None
    for i, (p, n, b) in enumerate(faces):
        if band is not None and b != band or i in used:
            continue
        if n @ d > best:
            best, bi = n @ d, i
    return bi


# ------------------------------------------------------------------ shards
def bipyramid(acc, c, axis, r, h_up, h_dn, collar=0.08, sides=6, t=None, twist=0.0, flat=1.0, prof=P_SHARD, shade=1.0,
              collar_mat='belly'):
    axis = unit(axis)
    c = np.asarray(c, float)
    top, bot = c + axis * h_up, c - axis * h_dn
    x, _, z = G.basis(axis, t)
    a = np.radians(twist) + np.arange(sides) * 2 * np.pi / sides
    ring = (np.cos(a)[:, None] * x * flat + np.sin(a)[:, None] * z) * r
    Ru, Rd = c + axis * collar + ring, c - axis * collar + ring
    for i in range(sides):
        j = (i + 1) % sides
        for poly, tip in (([Ru[i], Ru[j]], top), ([Rd[j], Rd[i]], bot)):
            p = np.array([poly[0], poly[1], tip])
            n = unit(np.cross(p[1] - p[0], p[2] - p[0]))
            if n @ (p.mean(0) - c) < 0:
                n = -n
            terrace(acc, p, n, prof, shade=shade * (0.85 + 0.15 * (i % 2)))
        mo = unit(ring[i] + ring[j])
        acc.tri(collar_mat, Ru[i], Rd[i], Rd[j], mo, shade)
        acc.tri(collar_mat, Ru[i], Rd[j], Ru[j], mo, shade)
    return top, bot


# satellite shards: (angle deg, orbit radius, y, size, tangential tilt, radial tilt, twist)
SHARDS = [(0, 4.55, 6.3, 1.2, 0.25, -0.1, 0), (118, 4.8, 8.6, 0.72, -0.35, 0.2, 20), (140, 4.55, 4.35, 0.78, 0.4, -0.15, 10),
          (180, 4.55, 5.85, 1.12, -0.3, -0.1, 30), (280, 4.4, 4.5, 0.9, 0.3, 0.15, 5), (320, 4.45, 8.05, 0.92, -0.35, 0.1, 25)]


def shard_geo(i):
    a, R, y, s, tt, tr, tw = SHARDS[i]
    c = ang(a, R, y)
    radial = unit([c[0], 0, c[2]])
    tang = np.cross(Y, radial)
    axis = unit(Y + tang * tt + radial * tr)
    return c, axis, s, tw, radial, tang


# ------------------------------------------------------------------ build
def build(K):
    K.mat('body', rough=0.24, metal=0.3, detail=False)
    K.mat('belly', rough=0.28, metal=0.92, detail=False)
    K.mat('accent', emit_strength=4.5)
    K.mat('horn', color='#150d1c', rough=0.16, metal=0.35)            # polished obsidian prongs
    K.mat('plate', color='#4a3550', rough=0.3, metal=0.55, vcol=True)  # lighter amethyst armour facets
    K.mat('dark', color='#0c0810', rough=0.45, metal=0.2)
    K.mat('eye', emit_strength=5.0)
    rng = np.random.default_rng(7)

    # ---------------- SDF skin: the core assembly
    B = K.sdf()
    heart = [C + (0, 1.2, 0), C - (0, 1.2, 0)] + [C + ang(60 * k, 1.15, 0.38) for k in range(6)] + \
            [C + ang(30 + 60 * k, 1.15, -0.38) for k in range(6)]
    HEART = Poly(heart)
    B.add(HEART, k=0.0)
    for k in range(3):                    # six-point star: three long hexagonal spindles in the equatorial plane
        a = 60 * k
        ax = ang(a, 1.0, 0)
        side = np.cross(Y, ax)
        pts = [C + ax * 2.55, C - ax * 2.55]
        for j in range(6):
            t = np.radians(30 + 60 * j)
            pts.append(C + Y * np.sin(t) * 0.62 + side * np.cos(t) * 0.42)
        B.add(Poly(pts), k=0.0)
    bz = lambda R, z0, z1, rot: [C + (R * np.cos(np.radians(rot + 60 * j)), R * np.sin(np.radians(rot + 60 * j)), z) for j in range(6) for z in (z0, z1)]
    B.add(Poly(bz(1.08, 0.7, 1.9, 0)), k=0.0)            # stepped hexagonal bezel housing
    B.add(Poly(bz(0.86, 0.7, APZ, 30)), k=0.0)
    B.cut(Poly(bz(0.58, 1.95, APZ + 0.6, 30)), k=0.0)      # aperture well

    def belly_mask(P):
        r = np.hypot(P[:, 0] - C[0], P[:, 1] - C[1])
        return np.maximum(APZ - 0.06 - P[:, 2], r - 0.95)

    def heart_mask(P):
        return HEART.dist(S.Points(P)) - 0.03

    def tint(P, N):     # facet-wise variation on the spindles
        return 0.85 + 0.15 * np.abs(np.sin(N[:, 0] * 5 + N[:, 1] * 7 + N[:, 2] * 3))
    K.skin(B, voxel=0.035, tris=13000, belly=belly_mask, regions=[('accent', heart_mask)], tint=tint, ao=0.35,
           custom_normals=False, sharp=28)

    # ---------------- skeleton
    K.bone('spine', (0, LOWER_APEX, 0), (0, 4.2, 0), None, 0.6)
    K.bone('chest', (0, 7.8, 0), (0, UPPER_APEX, 0), 'spine', 0.6)
    K.bone('head', C + (0, 0, -0.6), C + (0, 0, APZ), 'spine', 3.0)
    for i in range(len(SHARDS)):
        c, axis, s, *_ = shard_geo(i)
        K.bone(f'shard{i + 1}', c - axis * 0.5 * s, c + axis * 0.9 * s, None, 0.05)
    TAIL = [(0, LOWER_APEX - 0.02, 0), (0, 2.22, 0), (0, 1.84, 0), (0, 1.5, 0)]
    K.chain('tail', TAIL, 'spine', (0.12, 0.08))

    # ---------------- hull halves (rigid, merged into the body)
    UF, UC, UR, UA = build_hull(K, True, rng)
    LF, LC, LR, LA = build_hull(K, False, rng)

    # ---------------- aperture lens (mouth) on the core bezel
    acc = Acc()
    iris(acc, C + (0, 0, 1.96), (0, 0, 1), 0.58, 0.27, mat='horn', rot=0)
    acc.emit(K, 'body', 'head')

    # ---------------- satellite shards + dangling bead string
    for i in range(len(SHARDS)):
        c, axis, s, tw, radial, tang = shard_geo(i)
        acc = Acc()
        bipyramid(acc, c, axis, 0.36 * s, 1.35 * s, 0.95 * s, collar=0.07 * s, t=radial, twist=tw, shade=rng.uniform(0.85, 1.0))
        acc.emit(K, 'body', f'shard{i + 1}')
    for i, (y, s) in enumerate(((2.42, 1.0), (2.02, 0.8), (1.68, 0.62))):
        acc = Acc()
        bipyramid(acc, (0, y, 0), Y, 0.17 * s, 0.24 * s, 0.3 * s, collar=0.03, twist=30 * i,
                  prof=[(1.0, 0.0, None), (0.75, 0.0, 'belly'), (0.0, 0.0, 'body')])
        acc.emit(K, 'body', f'tail{i + 1}')

    # ---------------- mounts
    sR, sL = shard_geo(0)[0], shard_geo(3)[0]
    K.mount('head', C + (0, 0, 0.8), 'head')
    K.mount('jaw', C + (0, -0.45, 2.0), 'head')
    K.mount('mouth', C + (0, 0, APZ + 0.15), 'head')
    K.mount('chest', (0, 8.0, 2.75), 'chest')
    K.mount('back', ang(270, 2.5, 8.5) + (0, 0, -0.1), 'chest')
    K.mount('crown', (0, UPPER_APEX + 0.15, 0), 'chest')
    K.mount('tailTip', TAIL[-1], 'tail3')
    K.mount('handR', sR, 'shard1')
    K.mount('handL', sL, 'shard4')
    K.mount('shoulderR', UR[0][0] + (0.1, 0.05, 0), 'chest')
    K.mount('shoulderL', UR[0][3] + (-0.1, 0.05, 0), 'chest')

    # ---------------- Might: focusing prongs + lens rings around the aperture (exclusive sets)
    for t in range(4):
        part = f'pow_t{t}'
        cnt = [3, 4, 6, 6][t]
        L = [0.95, 1.35, 1.75, 2.2][t]
        rb = [0.12, 0.14, 0.15, 0.17][t]
        a0 = [90, 45, 0, 0][t]
        z0 = 1.95
        env = lambda f: (1.0 - 0.62 * f)                 # prong envelope radius factor along the length
        for j in range(cnt):
            a = np.radians(a0 + j * 360 / cnt)
            d = np.array([np.cos(a), np.sin(a) * 0.78, 0])
            root = C + d * 1.0 + (0, 0, z0 - 0.15)
            tip = C + d * 0.3 + (0, 0, z0 + L)
            mid = root * 0.5 + tip * 0.5 + d * 0.12
            K.horn(part, [root, mid, tip], [rb, rb * 0.7, 0.0], mat='belly', sides=5, per=3, bone='head', shade=(0.75, 1.0))
            K.horn(part, [root + (0, 0, 0.1), root * 0.62 + tip * 0.38], [rb * 1.12, rb * 0.95], mat='horn', sides=5, per=1, bone='head')
        acc = Acc()
        for f in [[], [], [0.6], [0.42, 0.78]][t]:
            zf = z0 + L * f
            # radius of the prong spline at that depth (catmull through root/mid/tip ~ piecewise)
            rr = 1.0 + (0.3 - 1.0) * f + 0.12 * np.sin(np.pi * f)
            ring_piece(acc, C + (0, 0, zf), (0, 0, 1), rr + 0.12, rr - 0.02, 0.06, 'belly', 6, 30, t=(0, 1, 0), wall_mat='horn')
        if t >= 1:      # glowing focus gem where the prongs converge
            bipyramid(acc, C + (0, 0, z0 + L + 0.12), (0, 0, 1), 0.12 + 0.03 * t, 0.3 + 0.08 * t, 0.15, collar=0.0, sides=6,
                      prof=[(1.0, 0.0, None), (0.0, 0.0, 'accent')])
        if t == 3:
            for s in (-1, 1):
                bipyramid(acc, C + (s * 1.45, 0, 2.35), (s * 0.25, 0, 1), 0.13, 0.75, 0.3, collar=0.04, sides=5,
                          prof=[(1.0, 0.0, None), (0.7, 0.0, 'belly'), (0.0, 0.0, 'accent')])
        acc.emit(K, part, 'head')

    # ---------------- Armour: floating shield facets
    used = []
    def sh(part, faces, d, band, bone, off=0.32, shrink=0.8):
        i = pick(faces, d, band, used if faces is UF else ())
        if faces is UF:
            used.append(i)
        p, n, _ = faces[i]
        return shield(K, part, p, n, off, bone, shrink)
    for s in (-1, 1):
        sh('arm_t1', UF, (s * 0.8, 0.0, 0.6), 0, 'chest')
        sh('arm_t2', LF, (s * 0.85, -0.3, 0.3), 0, 'spine', off=0.28)
        sh('arm_t2', UF, (s * 0.7, 0.4, 0.45), 1, 'chest', off=0.28, shrink=0.78)
        sh('arm_t3', UF, (s * 0.6, 0.0, -0.75), 0, 'chest')
        sh('arm_t3', UF, (s * 0.9, 0.25, -0.1), 1, 'chest', off=0.3)
    sh('arm_t3', LF, (0, -0.3, -1), 0, 'spine', off=0.28)

    # ---------------- Energy: glowing groove inlays, core flare, glowing satellites
    groove_inlay(K, 'enr_t1', UF, 'chest')
    groove_inlay(K, 'enr_t2', LF, 'spine')
    acc = Acc()
    for k in range(6):                     # core flare: thin molten spikes between the star spindles
        a = 30 + 60 * k
        if a == 90:
            continue
        ax = ang(a, 1.0, 0)
        bipyramid(acc, C + ax * 1.0, ax, 0.12, 1.85, 0.2, collar=0.0, sides=4, t=Y, prof=[(1.0, 0.0, None), (0.0, 0.0, 'accent')])
    acc.emit(K, 'enr_t2', 'head')
    for i in range(len(SHARDS)):
        c, axis, s, tw, radial, tang = shard_geo(i)
        acc = Acc()
        for sg in (1, -1):
            top = c + axis * sg * (1.35 if sg > 0 else 0.95) * s
            bipyramid(acc, top - axis * sg * 0.05 * s, axis * sg, 0.08 * s, 0.4 * s, 0.1 * s, collar=0.0, sides=4,
                      t=radial, prof=[(1.0, 0.0, None), (0.0, 0.0, 'accent')])
        ring_piece(acc, c, axis, 0.48 * s, 0.38 * s, 0.05, 'accent', 6, tw, t=radial)
        acc.emit(K, 'enr_t3', f'shard{i + 1}')

    # ---------------- Speed: trailing keel spikes, orbiting fin shards, swept vanes
    def keel_spike(part, a, ln):
        root = ang(a, 1.15, 3.45)
        d = unit(ang(a, 0.55, -1.0))
        K.horn(part, [root, root + d * ln * 0.55, root + d * ln + (0, 0.05, 0)], [0.17, 0.11, 0.0], mat='horn', sides=5,
               per=2, bone='spine', shade=(0.6, 1.0))
        K.horn(part, [root - d * 0.05, root + d * 0.25], [0.21, 0.19], mat='belly', sides=5, per=1, bone='spine')
    for a in (30, 150, 270):
        keel_spike('spd_t1', a, 1.75)
    for k in range(6):
        a = 30 + 60 * k
        c = ang(a, 3.55, 3.75 + 0.35 * (k % 2))
        radial = unit([c[0], 0, c[2]])
        tang = np.cross(Y, radial)
        acc = Acc()
        bipyramid(acc, c, tang + Y * 0.12, 0.26, 0.85, 0.45, collar=0.03, sides=4, t=Y, flat=0.3,
                  prof=[(1.0, 0.0, None), (0.75, 0.0, 'belly'), (0.0, 0.03, 'horn')])
        acc.emit(K, 'spd_t2', f'shard{k + 1}' if k < 6 else 'shard1')
    for a in (90, 210, 330):
        keel_spike('spd_t3', a, 1.3)
    for s in (-1, 1):                      # swept trailing vanes off the back of the keel hull
        acc = Acc()
        root = ang(270 + s * 38, 2.1, 4.3)
        dv = unit((s * 0.35, -0.55, -1))
        bipyramid(acc, root + dv * 0.5, dv, 0.36, 1.7, 0.5, collar=0.04, sides=4,
                  t=(s, 0, 0), flat=0.26, prof=[(1.0, 0.0, None), (0.8, 0.0, 'belly'), (0.5, -0.02, 'horn'), (0.0, 0.02, 'accent')])
        acc.emit(K, 'spd_t3', 'spine')

    # ---------------- Focus: extra apertures, floating crown ring, crown shard
    used_f = []
    for s in (-1, 1):
        i = pick(UF, (s * 0.55, 0.3, 0.8), 1)
        p, n, _ = UF[i]
        lens_eye(K, 'fcs_t1', incenter(p) + n * 0.2, n, 0.24, 'chest')
        used_f.append(i)
    acc = Acc()
    ring_piece(acc, (0, UPPER_APEX + 0.75, 0), Y, 1.05, 0.82, 0.1, 'belly', 6, 30, t=(0, 0, 1))
    for k in range(6):
        p = ang(30 + 60 * k, 0.94, UPPER_APEX + 0.75)
        bipyramid(acc, p + (0, 0.12, 0), Y, 0.08, 0.42 if k % 2 else 0.28, 0.08, collar=0.0, sides=4,
                  prof=[(1.0, 0.0, None), (0.0, 0.0, 'accent')])
    acc.emit(K, 'fcs_t2', 'chest')
    acc = Acc()
    bipyramid(acc, (0, UPPER_APEX + 1.45, 0), Y, 0.3, 0.75, 0.42, collar=0.05, twist=30,
              prof=[(1.0, 0.0, None), (0.75, 0.0, 'belly'), (0.5, -0.02, 'body'), (0.0, 0.03, 'accent')])
    acc.emit(K, 'fcs_t3', 'chest')
    for d in ((-0.7, -0.35, 0.6), (0.7, -0.35, 0.6)):
        i = pick(LF, d, 0)
        p, n, _ = LF[i]
        lens_eye(K, 'fcs_t3', incenter(p) + n * 0.2, n, 0.2, 'spine')
    i = pick(UF, (0, 0.2, -1), 1)
    p, n, _ = UF[i]
    lens_eye(K, 'fcs_t3', incenter(p) + n * 0.2, n, 0.24, 'chest')
