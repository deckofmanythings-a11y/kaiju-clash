"""Mecha Saurian: a hunched robotic theropod (biped plan). Game coords, ~10.8 tall, faces +Z.

Shape language: a dark mechanical inner frame (SDF skin: spine column, joint balls, hydraulic pistons)
under blocky faceted armour shells (rigid flat-shaded prisms, each bound to one bone so the robot
articulates like a machine). Wedge head with a single visor-slit optic band, hinged jaw with a glowing
reactor throat, a glowing spine conduit flanked by heat-sink fins, flank + thigh heat vents, splayed
three-toed steel feet and a tail of separate armoured rings ending in a vented hammer block.
"""
import numpy as np

from .. import geom as G
from ..builder import POW_SCALE

DETAIL = 'metal'
UV_SCALE = 0.75
METAL = True
EYE_ACCENT = True
SCALE = 1.0

INFO = dict(name='Mecha Saurian', colors=dict(body='#2c3e5c', belly='#c4893f', accent='#ff7a1f'))


# ------------------------------------------------------------------ small math
def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def Rx(a):
    a = np.radians(a)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def Ry(a):
    a = np.radians(a)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def Rz(a):
    a = np.radians(a)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def E(rx=0, ry=0, rz=0):
    return Rx(rx) @ Ry(ry) @ Rz(rz)


class Fr:
    """Local frame: origin O + 3x3 R (columns = local axes in world)."""

    def __init__(self, O, R=None):
        self.O = np.asarray(O, float)
        self.R = np.eye(3) if R is None else np.asarray(R, float)

    def p(self, c):
        return self.O + self.R @ np.asarray(c, float)

    def r(self, rot=None):
        if rot is None:
            return self.R
        return self.R @ (rot if isinstance(rot, np.ndarray) else E(*rot))

    def v(self, d):
        return self.R @ np.asarray(d, float)

    def sub(self, c, rot=None):
        return Fr(self.p(c), self.r(rot))


def along(o, ydir, xdir=(1, 0, 0)):
    """Frame at o with local +Y = ydir, +X ~ xdir, +Z = X x Y."""
    y = unit(ydir)
    x = np.asarray(xdir, float)
    x = unit(x - y * (x @ y))
    return Fr(o, np.column_stack([x, y, np.cross(x, y)]))


def seg(a, b, out, s=1):
    """Frame at joint a for the bone a->b: local -Y points at b, +X ~ out, +Z forward-ish."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    y = unit(a - b)
    x = np.asarray(out, float) - y * (np.asarray(out, float) @ y)
    x = unit(x)
    z = np.cross(x, y) * s
    return Fr(a, np.column_stack([x, y, z]))


# ------------------------------------------------------------------ hard-surface prisms
def _ccw(P):
    P = np.asarray(P, float)
    a = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if a > 0 else P[::-1].copy()


def _chamfer(P, c):
    n = len(P)
    cs = np.broadcast_to(np.asarray(c, float), (n,))
    out = []
    for i in range(n):
        p, a, b = P[i], P[i - 1], P[(i + 1) % n]
        if cs[i] <= 0:
            out.append(p)
            continue
        la, lb = np.linalg.norm(a - p), np.linalg.norm(b - p)
        out += [p + (a - p) / la * min(cs[i], 0.45 * la), p + (b - p) / lb * min(cs[i], 0.45 * lb)]
    return np.array(out)


def _inset(P, b):
    n = len(P)
    L = []
    for i in range(n):
        a, c = P[i], P[(i + 1) % n]
        e = unit(c - a)
        L.append((a - np.array([e[1], -e[0]]) * b, e))
    out = []
    for i in range(n):
        (p1, e1), (p2, e2) = L[i - 1], L[i]
        A = np.array([e1, -e2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            out.append(p2)
        else:
            t, _ = np.linalg.solve(A, p2 - p1)
            out.append(p1 + t * e1)
    return np.array(out)


AX = {'z': [0, 1, 2], 'x': [2, 1, 0], 'y': [0, 2, 1]}


def prism(prof, d, bev=0.05, taper=1.0, chamf=0.0, shear=(0, 0), ax='z', taper2=None):
    """Convex 2D profile (u, v) extruded +-d along w with chamfered cap edges.
    'z': (u,v,w)=(x,y,z)  'x': (u,v,w)=(z,y,x)  'y': (u,v,w)=(x,z,y)."""
    P = _ccw(prof)
    if np.any(np.asarray(chamf) > 0):
        P = _chamfer(P, chamf)
    ctr = P.mean(0)
    b = min(bev, 0.4 * d)
    ts = np.array([taper, taper if taper2 is None else taper2])

    def ring(Q, w):
        f = (w + d) / (2 * d)
        s = 1 + (ts - 1) * f
        Q2 = ctr + (Q - ctr) * s + np.asarray(shear, float) * f
        return np.column_stack([Q2, np.full(len(Q2), w)])
    if b > 1e-4:
        Pi = _inset(P, b)
        rings = [ring(Pi, -d), ring(P, -d + b), ring(P, d - b), ring(Pi, d)]
    else:
        rings = [ring(P, -d), ring(P, d)]
    n = len(P)
    V = np.vstack(rings + [rings[0].mean(0)[None], rings[-1].mean(0)[None]])
    T = []
    for k in range(len(rings) - 1):
        o0, o1 = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            T += [(o0 + i, o0 + j, o1 + j), (o0 + i, o1 + j, o1 + i)]
    c0, c1 = len(V) - 2, len(V) - 1
    l = (len(rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        T += [(c0, j, i), (c1, l + i, l + j)]
    return V[:, AX[ax]], np.array(T)


def rect(hx, hy):
    return [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]


def ngon(r, n, rot=0.0, sx=1.0):
    a = np.radians(rot) + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(a) * r * sx, np.sin(a) * r])


# ------------------------------------------------------------------ skeleton layout (world)
HIP_C = np.array([0, 5.3, -0.7])
S0 = np.array([0, 5.5, -0.5])        # spine base
S1 = np.array([0, 6.75, 0.6])        # spine -> chest
S2 = np.array([0, 7.85, 1.65])       # chest top / neck base
N1 = np.array([0, 9.6, 2.95])        # neck top / head pivot
TF = along(S0, S2 - S0)              # torso frame: +Y along spine, +Z belly (front-down), -Z dorsal
T_LEN = float(np.linalg.norm(S2 - S0))
NF = along(S2, N1 - S2)              # neck frame
HF = Fr(N1, Rx(4))                   # head frame: +Z snout, +Y up
JF = HF.sub((0, -0.28, -0.05), Rx(15))   # jaw hinge frame (open ~15 deg)

TAIL = np.array([(0, 5.35, -1.55), (0, 4.85, -3.05), (0, 4.25, -4.5), (0, 3.65, -5.92), (0, 3.12, -7.25),
                 (0, 2.75, -8.5), (0, 2.62, -9.55)])
RING_R = [1.18, 1.0, 0.84, 0.7, 0.58]


def leg_joints(s):
    hip = np.array([s * 1.45, 5.0, -0.6])
    kn = np.array([s * 1.9, 3.15, 0.8])
    an = np.array([s * 1.82, 1.1, -0.55])
    ball = np.array([s * 1.86, 0.42, 0.45])
    return hip, kn, an, ball


def arm_joints(s):
    sh = np.array([s * 1.72, 7.55, 1.75])
    el = np.array([s * 2.28, 6.45, 1.3])
    wr = np.array([s * 2.12, 5.98, 2.6])
    tip = np.array([s * 2.02, 5.55, 3.45])
    return sh, el, wr, tip


# ------------------------------------------------------------------ build
def build(K):
    K.mat('horn', color='#ccd2da', rough=0.24, metal=0.9)          # bare steel claws / teeth / blades
    K.mat('dark', color='#1b1e24', rough=0.45, metal=0.75)
    K.mat('plate', color='#56657a', rough=0.32, metal=0.75, detail=True, detail_kind='metal', detail_strength=0.05)
    K.mat('body', detail_strength=0.03)
    K.mat('belly', detail_strength=0.02)

    def put(part, fr, V, T, c=(0, 0, 0), rot=None, mat='body', bone=None, col=1.0):
        R = fr.r(rot)
        Vw = fr.p(c) + V @ R.T
        K.piece(part, Vw, G._consistent(Vw, T), mat=mat, smooth=False, col=col, bone=bone)

    def blk(part, fr, c, half, rot=None, mat='body', bone=None, bev=0.05, chamf=0.0, taper=1.0, taper2=None,
            shear=(0, 0), ax='z', col=1.0):
        hx, hy, hz = half
        if ax == 'z':
            V, T = prism(rect(hx, hy), hz, bev, taper, chamf, shear, 'z', taper2)
        elif ax == 'y':
            V, T = prism(rect(hx, hz), hy, bev, taper, chamf, shear, 'y', taper2)
        else:
            V, T = prism(rect(hz, hy), hx, bev, taper, chamf, shear, 'x', taper2)
        put(part, fr, V, T, c, rot, mat, bone, col)

    def poly(part, fr, c, prof, d, rot=None, mat='body', bone=None, bev=0.05, chamf=0.0, taper=1.0, shear=(0, 0),
             ax='z', col=1.0, taper2=None):
        V, T = prism(prof, d, bev, taper, chamf, shear, ax, taper2)
        put(part, fr, V, T, c, rot, mat, bone, col)

    def line(part, fr, a, b, n, w=0.026, bone=None, mat='dark'):
        """Panel-line strip from a to b (local), lying on a surface with local normal n."""
        a, b, n = np.asarray(a, float), np.asarray(b, float), unit(n)
        t = unit(b - a)
        x = unit(np.cross(t, n))
        R = np.column_stack([x, t, n])
        Ln = np.linalg.norm(b - a) / 2
        V, T = prism(rect(w, Ln), 0.02, 0.0)
        put(part, fr, V, T, (a + b) / 2, R, mat, bone)

    def rivet(part, fr, c, n, r=0.045, bone=None, mat='horn'):
        n = unit(n)
        x, y, z = G.basis(n)
        R = np.column_stack([x, z, y])   # prism 'z' axis -> normal
        V, T = prism(ngon(r, 6), 0.025, 0.0)
        put(part, fr, V, T, c, R, mat, bone)

    def glow(part, fr, c, r, rot=None, mat='accent', bone=None, seg_=12):
        rr = fr.r(rot)
        V, T, _ = G.sphere((0, 0, 0), 1.0, seg_, 6)
        V = V * np.asarray(r, float)
        Vw = fr.p(c) + V @ rr.T
        K.piece(part, Vw, T, mat=mat, bone=bone)

    def thruster(part, fr, c, d, ln, r, bone):
        base, dd = fr.p(c), unit(fr.v(d))
        K.horn(part, [base - dd * 0.1, base + dd * ln * 0.45, base + dd * ln], [r * 0.62, r * 0.8, r], mat='dark', sides=10, bone=bone)
        K.horn(part, [base + dd * (ln - 0.06), base + dd * (ln + 0.01)], [r * 1.07, r * 1.07], mat='horn', sides=10, bone=bone)
        x, _, z = G.basis(dd)
        cc = base + dd * (ln - 0.1)
        V = [cc] + [cc + (np.cos(a) * x + np.sin(a) * z) * r * 0.86 for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
        T = [(0, 1 + i, 1 + (i + 1) % 10) for i in range(10)] + [(0, 1 + (i + 1) % 10, 1 + i) for i in range(10)]
        K.piece(part, np.array(V), np.array(T), mat='accent', bone=bone)

    def vent(part, fr, c, hw, hh, n_slats, bone, rot=None, glow_on=True, slat_mat='body', d=0.05):
        """Recessed heat vent in local (x, y) plane facing local +Z: dark recess, glow strip, louvre slats."""
        f = fr.sub(c, rot)
        blk(part, f, (0, 0, 0), (hw, hh, d), mat='dark', bone=bone, bev=0.015, chamf=0.03)
        if glow_on:
            blk(part, f, (0, 0, d + 0.005), (hw * 0.86, hh * 0.8, 0.012), mat='accent', bone=bone, bev=0.0)
        for j in range(n_slats):
            yy = -hh + (j + 0.5) * 2 * hh / n_slats
            blk(part, f, (0, yy, d + 0.03), (hw * 0.95, hh / n_slats * 0.36, 0.03), rot=(-28, 0, 0), mat=slat_mat, bone=bone, bev=0.0)

    legs = {s: leg_joints(s) for s in (-1, 1)}
    arms = {s: arm_joints(s) for s in (-1, 1)}

    # ================================================================ inner frame (dark SDF skin)
    B = K.sdf()
    k0 = 0.08
    B.box(HIP_C + [0, 0.05, 0.05], (1.05, 0.55, 0.85), rad=0.25, k=k0)                      # pelvis block
    B.box(TF.p((0, 1.55, -0.05)), (0.95, 1.6, 0.8), rot=TF.R, rad=0.35, k=0.15)            # torso core
    B.tube([TF.p((0, -0.3, -0.75)), TF.p((0, T_LEN + 0.2, -0.7))], [0.42, 0.42], k=0.1)   # spine column
    for i in range(6):                                                                       # vertebra discs
        yy = 0.1 + i * 0.6
        B.tube([TF.p((0, yy - 0.06, -0.85)), TF.p((0, yy + 0.06, -0.85))], [0.5, 0.5], k=0.03)
    # neck: column + collar rings + hydraulic rams under the throat
    nl = float(np.linalg.norm(N1 - S2))
    B.tube([NF.p((0, -0.4, 0)), NF.p((0, nl + 0.15, 0))], [0.48, 0.4], k=0.1)
    for i in range(4):
        yy = 0.25 + i * 0.5
        B.tube([NF.p((0, yy - 0.05, 0)), NF.p((0, yy + 0.05, 0))], [0.56 - 0.03 * i, 0.56 - 0.03 * i], k=0.03)
    for s in (-1, 1):
        B.tube([NF.p((s * 0.32, -0.25, 0.55)), NF.p((s * 0.3, nl - 0.1, 0.5))], [0.13, 0.11], k=0.03)   # rams
        B.tube([NF.p((s * 0.32, -0.25, 0.55)), NF.p((s * 0.32, nl * 0.45, 0.55))], [0.17, 0.17], k=0.03)  # ram sleeves
    B.ell(HF.p((0, 0.05, 0.1)), (0.5, 0.45, 0.55), k=0.1)                                     # head core
    B.tube([JF.p((-0.6, 0, 0)), JF.p((0.6, 0, 0))], [0.2, 0.2], k=0.04)                         # jaw axle
    # tail spine + vertebra discs in the ring gaps
    B.tube(TAIL, [0.82, 0.7, 0.58, 0.47, 0.38, 0.3, 0.26], k=0.12)
    for i in range(1, len(TAIL) - 1):
        d = unit(TAIL[i + 1] - TAIL[i - 1])
        r = [0.82, 0.7, 0.58, 0.47, 0.38, 0.3][i] + 0.12
        B.tube([TAIL[i] - d * 0.07, TAIL[i] + d * 0.07], [r, r], k=0.03)
    for s in (-1, 1):
        hip, kn, an, ball = legs[s]
        B.ell(hip, (0.55, 0.55, 0.55), k=k0)
        B.tube([hip, kn], [0.42, 0.36], k=k0)
        B.tube([kn + [-0.35 * s, 0, 0], kn + [0.35 * s, 0, 0]], [0.3, 0.3], k=0.04)        # knee axle
        B.tube([kn, an], [0.32, 0.28], k=k0)
        B.ell(an, (0.32, 0.32, 0.32), k=0.05)
        B.tube([an, ball], [0.26, 0.24], k=0.05)
        th = seg(hip, kn, (s, 0, 0), s)
        sn = seg(kn, an, (s, 0, 0), s)
        B.tube([th.p((0, -0.5, -0.55)), sn.p((0, -0.9, -0.42))], [0.1, 0.1], k=0.02)          # knee ram
        B.tube([th.p((0, -0.5, -0.55)), th.p((0, -1.3, -0.58))], [0.15, 0.15], k=0.02)        # ram sleeve
        B.tube([sn.p((0, -0.5, -0.38)), sn.p((0, -2.2, -0.32))], [0.09, 0.09], k=0.02)        # calf ram
        sh, el, wr, tip = arms[s]
        B.ell(sh, (0.32, 0.32, 0.32), k=k0)
        B.tube([sh, el], [0.27, 0.25], k=0.05)
        B.tube([el + [-0.25 * s, 0, 0], el + [0.25 * s, 0, 0]], [0.25, 0.25], k=0.03)        # elbow axle
        B.tube([el, wr], [0.25, 0.22], k=0.05)
        ua = seg(sh, el, (s, 0, 0), s)
        B.tube([ua.p((-0.05, -0.2, 0.3)), wr + (el - wr) * 0.35 + [0, 0.12, 0]], [0.07, 0.07], k=0.02)  # elbow ram
    K.skin(B, voxel=0.045, tris=9500, base_mat='dark', custom_normals=False, sharp=40, ao=0.5)

    # ================================================================ skeleton + mounts
    K.bone('pelvis', HIP_C, S0 + (S1 - S0) * 0.1, None, 1.4)
    K.bone('spine', S0 + (S1 - S0) * 0.1, S1, 'pelvis', 1.3)
    K.bone('chest', S1, S2, 'spine', 1.4)
    K.bone('neck', S2, N1, 'chest', 0.6)
    K.bone('head', N1, HF.p((0, 0, 2.2)), 'neck', 0.6)
    K.bone('jaw', JF.O, JF.p((0, -0.1, 2.0)), 'head', 0.3)
    for s, sd in ((-1, 'L'), (1, 'R')):
        hip, kn, an, ball = legs[s]
        K.bone('thigh' + sd, hip, kn, 'pelvis', 0.7)
        K.bone('shin' + sd, kn, an, 'thigh' + sd, 0.5)
        K.bone('foot' + sd, an, np.array([s * 1.88, 0.2, 1.9]), 'shin' + sd, 0.4)
        sh, el, wr, tip = arms[s]
        K.bone('upperarm' + sd, sh, el, 'chest', 0.4)
        K.bone('forearm' + sd, el, wr, 'upperarm' + sd, 0.35)
        K.bone('hand' + sd, wr, tip, 'forearm' + sd, 0.3)
        K.mount('hand' + sd, wr + (tip - wr) * 0.55, 'hand' + sd)
        K.mount('shoulder' + sd, sh + np.array([s * 0.25, 0.75, -0.2]), 'chest')
    K.chain('tail', TAIL, 'pelvis', (0.95, 0.35))
    K.mount('head', HF.p((0, 0.25, 0.5)), 'head')
    K.mount('jaw', JF.p((0, -0.15, 0.9)), 'jaw')
    K.mount('mouth', HF.p((0, -0.42, 2.15)), 'head')
    K.mount('chest', TF.p((0, 2.7, 1.25)), 'chest')
    K.mount('back', TF.p((0, 1.8, -1.6)), 'spine')
    K.mount('tailTip', TAIL[-1] + np.array([0, 0, -0.5]), 'tail6')
    K.mount('crown', HF.p((0, 1.0, 0.4)), 'head')

    P = 'body'
    # ================================================================ torso armour
    # dorsal carapace: four overlapping faceted shell sections, widest over the shoulders
    shell = [(-1.3, -0.15), (1.3, -0.15), (1.36, -0.72), (0.78, -1.28), (-0.78, -1.28), (-1.36, -0.72)]
    secs = [(0.35, 'spine', 1.0), (1.2, 'spine', 1.06), (2.05, 'chest', 1.12), (2.85, 'chest', 1.02)]
    for i, (yy, bn, w) in enumerate(secs):
        prof = [(x * w, z) for x, z in shell]
        poly(P, TF, (0, yy, 0), prof, 0.47, rot=(-5, 0, 0), ax='y', bone=bn, bev=0.07, chamf=0.0)
        # conduit channel + glow segment on top of each section
        blk(P, TF, (0, yy, -1.3), (0.2, 0.47, 0.06), rot=(-5, 0, 0), mat='dark', bone=bn, bev=0.0)
        blk(P, TF, (0, yy, -1.34), (0.09, 0.44, 0.04), rot=(-5, 0, 0), mat='accent', bone=bn, bev=0.0)
        for s in (-1, 1):
            # panel lines + rivets on the shell's upper facets
            line(P, TF, (s * 0.95 * w, yy - 0.4, -1.03), (s * 0.95 * w, yy + 0.4, -1.03), (s * 0.7, 0, -0.7), bone=bn)
            for dy in (-0.3, 0.3):
                rivet(P, TF, (s * 0.55 * w, yy + dy, -1.29), (0, 0, -1), bone=bn)
            # heat-sink fins flanking the conduit, swept back toward the tail
            h = [0.55, 0.75, 0.95, 0.7][i]
            ff = Fr(TF.p((s * 0.36, yy, -1.26)), np.column_stack([TF.v((1, 0, 0)), unit(TF.v((s * 0.35, 0, -1))), np.cross(TF.v((1, 0, 0)), unit(TF.v((s * 0.35, 0, -1))))]))
            fin = [(-0.38, 0.0), (0.38, 0.0), (-0.05, h * 0.85), (-0.42, h)]
            poly(P, ff, (0, 0, 0), fin, 0.045, ax='x', mat='plate', bone=bn, bev=0.015)
    # ventral plastron: stacked bronze bands
    for i in range(5):
        yy = 0.22 + i * 0.6
        wv = 0.8 + 0.06 * min(i, 3)
        blk(P, TF, (0, yy, 0.84 + 0.02 * i), (wv, 0.33, 0.09), rot=(8, 0, 0), mat='belly', bone='spine' if i < 3 else 'chest', bev=0.03, chamf=0.05, taper=0.92)
        line(P, TF, (-wv * 0.55, yy + 0.05, 0.95 + 0.02 * i), (wv * 0.55, yy + 0.05, 0.95 + 0.02 * i), (0, 0, 1), bone='spine' if i < 3 else 'chest', w=0.018)
    # flank plates with the heat vents
    for s in (-1, 1):
        for yy, bn in ((0.75, 'spine'), (1.95, 'chest')):
            fl = Fr(TF.p((s * 1.2, yy, 0.28)), TF.r() @ np.diag([s, 1, 1]))
            blk(P, fl, (0.05, 0, 0), (0.12, 0.55, 0.62), mat='body', bone=bn, bev=0.04, chamf=0.1)
            vent(P, fl.sub((0.18, 0, 0.05), Ry(90)), (0, 0, 0), 0.4, 0.34, 4, bn)
            for dz in (-0.48, 0.48):
                rivet(P, fl, (0.18, 0.44, dz), (1, 0, 0), bone=bn)
                rivet(P, fl, (0.18, -0.44, dz), (1, 0, 0), bone=bn)
    # chest prow under the neck + collar
    blk(P, TF, (0, 3.2, 0.55), (0.82, 0.3, 0.45), rot=(-25, 0, 0), mat='body', bone='chest', bev=0.06, chamf=0.15)
    blk(P, TF, (0, 3.3, 0.98), (0.6, 0.22, 0.08), rot=(-25, 0, 0), mat='belly', bone='chest', bev=0.03, chamf=0.08)
    # pelvis armour: hip block + tail-root plate
    PF = Fr(HIP_C)
    blk(P, PF, (0, -0.15, 0.35), (0.85, 0.4, 0.55), mat='body', bone='pelvis', bev=0.06, chamf=0.18)
    blk(P, PF, (0, -0.42, 0.88), (0.55, 0.3, 0.1), rot=(-15, 0, 0), mat='belly', bone='pelvis', bev=0.03, chamf=0.08)
    for s in (-1, 1):
        sd = 'L' if s < 0 else 'R'
        hip = legs[s][0]
        poly(P, Fr(hip + [s * 0.5, 0, 0]), (0, 0, 0), ngon(0.46, 6, 0), 0.08, ax='x', mat='body', bone='thigh' + sd, bev=0.03)
        poly(P, Fr(hip + [s * 0.6, 0, 0]), (0, 0, 0), ngon(0.2, 6, 0), 0.04, ax='x', mat='horn', bone='thigh' + sd, bev=0.01)

    # ================================================================ neck armour: segmented collars + throat plates
    for i in range(3):
        yy = 0.5 + i * 0.55
        bn = 'chest' if i == 0 else 'neck'
        poly(P, NF, (0, yy, -0.08), ngon(0.62 - 0.04 * i, 8, 22.5, sx=1.05), 0.17, rot=(-8, 0, 0), ax='y', bone=bn, bev=0.04)
        poly(P, NF, (0, yy, -0.42), [(-0.32, 0.0), (0.32, 0.0), (0.22, -0.42), (-0.22, -0.42)], 0.2, ax='y', mat='body', bone=bn, bev=0.04)   # dorsal neck spine block
        blk(P, NF, (0, yy, 0.82), (0.24, 0.17, 0.05), mat='belly', bone=bn, bev=0.02)
    blk(P, NF, (0, 0.5, -0.88), (0.08, 0.2, 0.05), mat='accent', bone='chest', bev=0.0)

    # ================================================================ head: wedge helmet, visor band, hinged jaw
    hb = 'head'
    helm = [(-0.55, -0.12), (1.05, -0.12), (1.38, 0.24), (0.95, 0.72), (-0.25, 0.84), (-0.62, 0.42)]
    poly(P, HF, (0, 0, 0), helm, 0.62, ax='x', bone=hb, bev=0.08)
    poly(P, HF, (0, 0.86, 0.15), [(-0.55, -0.04), (0.75, -0.04), (0.55, 0.06), (-0.45, 0.06)], 0.2, ax='x', mat='belly', bone=hb, bev=0.02)   # brow stripe
    line(P, HF, (0.32, 0.8, -0.45), (0.32, 0.86, 0.6), (0, 1, 0), bone=hb)
    line(P, HF, (-0.32, 0.8, -0.45), (-0.32, 0.86, 0.6), (0, 1, 0), bone=hb)
    # visor: one continuous slit across the sloped front facet, wrapping onto the sides
    vf = HF.sub((0, 0.485, 1.165), (-41, 0, 0))
    blk(P, vf, (0, 0, 0.0), (0.66, 0.14, 0.04), mat='dark', bone=hb, bev=0.0)
    blk(P, vf, (0, -0.01, 0.035), (0.62, 0.042, 0.03), mat='eye', bone=hb, bev=0.0)
    for s in (-1, 1):
        blk(P, HF, (s * 0.625, 0.42, 0.95), (0.04, 0.12, 0.3), mat='dark', bone=hb, bev=0.0)
        blk(P, HF, (s * 0.655, 0.42, 0.98), (0.03, 0.04, 0.27), mat='eye', bone=hb, bev=0.0)
        # cheek armour plates + rivets, side intakes
        poly(P, HF, (s * 0.66, 0.05, 0.25), [(-0.5, 0.25), (0.45, 0.22), (0.35, -0.22), (-0.42, -0.25)], 0.05, ax='x', mat='body', bone=hb, bev=0.02)
        for zz in (-0.15, 0.55):
            rivet(P, HF, (s * 0.715, 0.15, zz), (s, 0, 0), bone=hb, r=0.04)
        blk(P, HF, (s * 0.64, 0.42, -0.25), (0.04, 0.14, 0.18), mat='dark', bone=hb, bev=0.02)
    # upper snout (wedge, tapering to a blunt chisel)
    blk(P, HF, (0, -0.05, 1.55), (0.5, 0.22, 0.8), mat='body', bone=hb, bev=0.06, chamf=0.12, taper=0.66, taper2=0.75)
    blk(P, HF, (0, 0.17, 1.6), (0.2, 0.03, 0.62), mat='belly', bone=hb, bev=0.01, taper=0.6)
    blk(P, HF, (0, -0.28, 1.45), (0.42, 0.04, 0.72), mat='dark', bone=hb, bev=0.0, taper=0.66)          # palate
    # lower jaw
    blk(P, JF, (0, -0.2, 1.05), (0.5, 0.17, 0.98), mat='body', bone='jaw', bev=0.05, chamf=0.1, taper=0.7)
    blk(P, JF, (0, -0.38, 1.0), (0.38, 0.04, 0.85), mat='belly', bone='jaw', bev=0.02, taper=0.7)
    blk(P, JF, (0, -0.02, 1.0), (0.36, 0.02, 0.8), mat='dark', bone='jaw', bev=0.0, taper=0.66)
    blk(P, JF, (0, 0.0, 0.95), (0.13, 0.02, 0.65), mat='accent', bone='jaw', bev=0.0, taper=0.7)         # glowing throat channel
    for s in (-1, 1):
        poly(P, JF, (s * 0.56, 0, 0), ngon(0.22, 6, 0), 0.05, ax='x', mat='body', bone='jaw', bev=0.02)
        poly(P, JF, (s * 0.62, 0, 0), ngon(0.1, 6, 0), 0.02, ax='x', mat='horn', bone='jaw', bev=0.0)
    # reactor throat: glowing core at the back of the mouth inside a dark bezel
    poly(P, HF, (0, -0.42, 0.42), ngon(0.3, 8, 22.5, sx=1.25), 0.06, mat='dark', bone=hb, bev=0.02)
    glow(P, HF, (0, -0.42, 0.5), (0.3, 0.21, 0.06), bone=hb)
    glow(P, HF, (0, -0.42, 0.54), (0.14, 0.1, 0.04), mat='eye', bone=hb)

    # ================================================================ arms
    for s, sd in ((-1, 'L'), (1, 'R')):
        sh, el, wr, tip = arms[s]
        ua = seg(sh, el, (s, 0, 0), s)
        fa = seg(el, wr, (s, 0, 0), s)
        ub, fb, hb_ = 'upperarm' + sd, 'forearm' + sd, 'hand' + sd
        blk(P, ua, (0.0, -0.55, 0.0), (0.33, 0.45, 0.36), bone=ub, bev=0.05, chamf=0.1, ax='y', taper=1.15)
        poly(P, Fr(sh + [s * 0.27, 0.02, 0]), (0, 0, 0), ngon(0.36, 6, 0), 0.08, ax='x', mat='body', bone=ub, bev=0.03)
        poly(P, Fr(sh + [s * 0.36, 0.02, 0]), (0, 0, 0), ngon(0.16, 6, 0), 0.03, ax='x', mat='horn', bone=ub, bev=0.0)
        blk(P, ua, (0.0, 0.12, 0.0), (0.3, 0.12, 0.36), mat='belly', bone=ub, bev=0.03, chamf=0.1, ax='y')
        blk(P, ua, (0.34, -0.55, 0.0), (0.04, 0.34, 0.24), mat='belly', bone=ub, bev=0.02, chamf=0.06)
        for sx in (-1, 1):
            poly(P, ua, (sx * 0.3, -1.29, 0), ngon(0.24, 6, 0), 0.05, mat='body', bone=fb, bev=0.02, ax='x')
        poly(P, ua, (0.36, -1.29, 0), ngon(0.11, 6, 0), 0.025, mat='horn', bone=fb, bev=0.0, ax='x')
        # forearm: boxy gauntlet
        blk(P, fa, (0.0, -0.72, 0.0), (0.36, 0.5, 0.38), bone=fb, bev=0.06, chamf=0.12, ax='y', taper=0.85)
        blk(P, fa, (0.0, -1.2, 0.0), (0.4, 0.08, 0.42), mat='belly', bone=fb, bev=0.03, chamf=0.12, ax='y')
        line(P, fa, (0.375, -0.35, -0.2), (0.375, -1.05, -0.22), (1, 0, 0), bone=fb)
        rivet(P, fa, (0.385, -0.4, 0.2), (1, 0, 0), bone=fb, r=0.035)
        rivet(P, fa, (0.385, -1.0, 0.2), (1, 0, 0), bone=fb, r=0.035)
        # hand: dark palm + three blocky fingers + thumb
        hf = seg(wr, tip, (s, 0, 0), s)
        blk(P, hf, (0, -0.22, 0), (0.28, 0.22, 0.24), mat='dark', bone=hb_, bev=0.04, chamf=0.05, ax='y')
        blk(P, hf, (0.2, -0.22, 0), (0.08, 0.2, 0.25), mat='body', bone=hb_, bev=0.03, chamf=0.06, ax='y')
        for zz in (-0.17, 0.0, 0.17):
            blk(P, hf, (0, -0.58, zz), (0.1, 0.16, 0.07), rot=(0, 0, 0), mat='dark', bone=hb_, bev=0.025, ax='y')
            blk(P, hf, (0.1, -0.5, zz), (0.04, 0.07, 0.06), mat='body', bone=hb_, bev=0.015, ax='y')
        blk(P, hf, (-0.2, -0.32, 0.25), (0.08, 0.16, 0.07), rot=(25, 0, 0), mat='dark', bone=hb_, bev=0.025, ax='y')

    # ================================================================ legs + feet
    for s, sd in ((-1, 'L'), (1, 'R')):
        hip, kn, an, ball = legs[s]
        th = seg(hip, kn, (s, 0, 0), s)
        sn = seg(kn, an, (s, 0, 0), s)
        tb, sb, fb = 'thigh' + sd, 'shin' + sd, 'foot' + sd
        tl = float(np.linalg.norm(kn - hip))
        # thigh: heavy faceted drumstick shell
        blk(P, th, (0.04, -0.98, 0.06), (0.6, 0.92, 0.74), bone=tb, bev=0.08, chamf=0.3, ax='y', taper=1.25)
        poly(P, th, (0.66, -0.95, 0.05), [(-0.55, 0.72), (0.55, 0.72), (0.42, -0.62), (-0.42, -0.7)], 0.06, ax='x', mat='body', bone=tb, bev=0.03)
        vent(P, th.sub((0.73, -0.9, 0.05), Ry(90)), (0, 0, 0), 0.36, 0.4, 4, tb)            # thigh heat vent
        blk(P, th, (0.0, -1.0, 0.78), (0.38, 0.6, 0.06), mat='belly', bone=tb, bev=0.03, chamf=0.1)  # front thigh plate
        for yy in (-0.5, -1.45):
            rivet(P, th, (0.72, yy, 0.5), (1, 0, 0), bone=tb)
            rivet(P, th, (0.72, yy, -0.42), (1, 0, 0), bone=tb)
        # knee: hex axle caps + forward knee cap
        for sx in (-1, 1):
            poly(P, th, (sx * 0.4, -tl, 0), ngon(0.38, 6, 0), 0.06, mat='body', bone=sb, bev=0.025, ax='x')
        poly(P, th, (0.5, -tl, 0), ngon(0.15, 6, 0), 0.03, mat='horn', bone=sb, bev=0.0, ax='x')
        poly(P, sn, (0.0, 0.05, 0.36), [(-0.38, 0.38), (0.38, 0.38), (0.36, -0.3), (0.0, -0.55), (-0.36, -0.3)], 0.17,
             rot=(-12, 0, 0), mat='belly', bone=sb, bev=0.06, taper=0.8)
        # shin: slab armour on the front + side plate, pistons behind
        blk(P, sn, (0.0, -1.2, 0.12), (0.4, 0.85, 0.36), bone=sb, bev=0.06, chamf=0.14, ax='y', taper=1.1)
        poly(P, sn, (0.0, -1.15, 0.5), [(-0.32, 0.7), (0.32, 0.7), (0.26, -0.7), (0.0, -0.85), (-0.26, -0.7)], 0.05,
             mat='belly', bone=sb, bev=0.025)
        line(P, sn, (0.0, -0.6, 0.555), (0.0, -1.7, 0.555), (0, 0, 1), bone=sb, w=0.02)
        poly(P, sn, (0.42, -1.2, 0.1), [(-0.35, 0.6), (0.35, 0.6), (0.3, -0.62), (-0.3, -0.62)], 0.04, ax='x', mat='body', bone=sb, bev=0.02)
        # foot: metatarsal block, three splayed steel toes, rear spur
        mt = seg(an, ball, (s, 0, 0), s)
        ml = float(np.linalg.norm(ball - an))
        blk(P, mt, (0, -ml * 0.5, 0.05), (0.34, ml * 0.5 + 0.1, 0.3), bone=fb, bev=0.05, chamf=0.1, ax='y')
        blk(P, mt, (0, -ml * 0.5, 0.36), (0.22, ml * 0.42, 0.04), mat='belly', bone=fb, bev=0.02)
        for k, ang in enumerate((-32, 0, 32)):
            d = np.array([np.sin(np.radians(ang)) * s, 0, np.cos(np.radians(ang))])
            ln = 1.05 if ang == 0 else 0.85
            tf = Fr(ball + [0, -0.12, 0], np.column_stack([unit(np.cross([0, 1, 0], d)), [0, 1, 0], d]))
            blk(P, tf, (0, 0.04, ln * 0.35), (0.19, 0.2, ln * 0.38), mat='body', bone=fb, bev=0.04, chamf=0.07, taper=0.85)
            blk(P, tf, (0, 0.0, ln * 0.82), (0.16, 0.16, ln * 0.18), mat='plate', bone=fb, bev=0.035, chamf=0.05, taper=0.8)
            blk(P, tf, (0, -0.22, ln * 0.5), (0.18, 0.04, ln * 0.52), mat='dark', bone=fb, bev=0.0)                    # sole pad
            rivet(P, tf, (0, 0.25, ln * 0.35), (0, 1, 0), bone=fb, r=0.04)
        sp = Fr(an, np.diag([s, 1, 1]))
        blk(P, sp, (0, -0.4, -0.25), (0.17, 0.42, 0.15), rot=(35, 0, 0), mat='body', bone=fb, bev=0.04, chamf=0.05)   # rear strut
        for sx in (-1, 1):
            blk(P, sp, (sx * 0.34, 0.0, 0.0), (0.05, 0.26, 0.28), mat='body', bone=fb, bev=0.02, chamf=0.08)            # ankle guards

    # ================================================================ tail: separate armoured rings + hammer block
    for i in range(5):
        a, b = TAIL[i], TAIL[i + 1]
        sl = float(np.linalg.norm(b - a))
        tfr = along((a + b) * 0.5, b - a)                    # +Y toward the tip, +Z dorsal
        bn = f'tail{i + 1}'
        r = RING_R[i]
        poly(P, tfr, (0, 0, 0), ngon(r, 8, 22.5, sx=1.08), sl * 0.39, ax='y', bone=bn, bev=0.07, taper=0.9)
        blk(P, tfr, (0, 0, -r * 0.9), (r * 0.5, sl * 0.36, 0.08), mat='belly', bone=bn, bev=0.03, chamf=0.08, taper=0.9)
        blk(P, tfr, (0, 0, r * 0.93), (0.17, sl * 0.36, 0.06), mat='dark', bone=bn, bev=0.0)
        blk(P, tfr, (0, 0, r * 0.97), (0.08, sl * 0.33, 0.04), mat='accent', bone=bn, bev=0.0)
        for s in (-1, 1):
            line(P, tfr, (s * r * 0.98, -sl * 0.3, 0.0), (s * r * 0.98, sl * 0.3, 0.0), (s, 0, 0), bone=bn)
            rivet(P, tfr, (s * r * 0.6, -sl * 0.28, r * 0.78), (s * 0.6, 0, 0.8), bone=bn, r=0.04)
            # small heat-sink fin pair continues the dorsal row down the tail
            if i < 4:
                h = 0.55 - 0.1 * i
                ff = Fr(tfr.p((s * 0.3, 0, r * 0.88)), np.column_stack([tfr.v((1, 0, 0)), unit(tfr.v((s * 0.35, 0, 1))), np.cross(tfr.v((1, 0, 0)), unit(tfr.v((s * 0.35, 0, 1))))]))
                fin = [(-0.3, 0.0), (0.3, 0.0), (0.05, h * 0.85), (0.32, h)]
                poly(P, ff, (0, 0, 0), fin, 0.04, ax='x', mat='plate', bone=bn, bev=0.012)
    # hammer: heavy chamfered block with glowing vent grilles and a rear exhaust
    a, b = TAIL[5], TAIL[6]
    hf_ = along(b, b - a)
    bn = 'tail6'
    blk(P, hf_, (0, -0.15, 0), (0.62, 0.62, 0.55), mat='body', bone=bn, bev=0.08, chamf=0.18, ax='y', taper=1.12)
    blk(P, hf_, (0, -0.15, 0), (0.68, 0.32, 0.6), mat='belly', bone=bn, bev=0.04, chamf=0.15, ax='y')
    for s in (-1, 1):
        vent(P, hf_.sub((s * 0.7, -0.15, 0), Ry(90 * s)), (0, 0, 0), 0.34, 0.36, 3, bn)
    thruster(P, hf_, (0, 0.35, 0), (0, 1, 0), 0.3, 0.3, bn)

    # ================================================================ tiers
    # ---- Might (exclusive): toe/finger claws + teeth grow; t2 wrist blades; t3 big glowing blades + snout drill
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        k = 0.75 + 0.25 * sc
        for s, sd in ((-1, 'L'), (1, 'R')):
            hip, kn, an, ball = legs[s]
            for ang in (-32, 0, 32):
                d = np.array([np.sin(np.radians(ang)) * s, 0, np.cos(np.radians(ang))])
                ln = 1.05 if ang == 0 else 0.85
                base = ball + [0, -0.1, 0] + d * ln * 0.95
                K.claw(part, base, d + [0, -0.35, 0], 0.5 * k, 0.12 * (0.85 + 0.15 * sc), curl=(0, -1, 0), bone='foot' + sd, flat=0.6)
            K.claw(part, an + [0, -0.74, -0.5], (0, -0.6, -1), 0.3 * k, 0.09, bone='foot' + sd)
            sh, el, wr, tip = arms[s]
            hf = seg(wr, tip, (s, 0, 0), s)
            for zz in (-0.17, 0.0, 0.17):
                K.claw(part, hf.p((0, -0.72, zz)), hf.v((0.0, -1, 0.35)), 0.36 * k, 0.075 * (0.85 + 0.15 * sc), curl=hf.v((-1, 0, 0)), bone='hand' + sd)
            K.claw(part, hf.p((-0.24, -0.46, 0.3)), hf.v((-0.3, -0.8, 0.5)), 0.22 * k, 0.06, curl=hf.v((-1, 0, 0)), bone='hand' + sd)
            # teeth: upper (head) and lower (jaw) rows of steel fangs
            tl_ = 0.11 + 0.05 * t
            for j in range(6):
                z = 0.95 + j * 0.24
                x = s * (0.42 - j * 0.035)
                ln = tl_ * (1.4 if j == 1 else 1.0)
                p0 = HF.p((x, -0.3, z))
                K.horn(part, [p0 + HF.v((0, 0.04, 0)), p0 + HF.v((0, -ln * 0.6, 0.01)), p0 + HF.v((0, -ln, 0.03))], [0.05, 0.035, 0], sides=4, bone='head', shade=(0.8, 1))
                p1 = JF.p((x * 0.95, -0.02, z - 0.05))
                ln1 = tl_ * (1.3 if j == 2 else 0.9)
                K.horn(part, [p1 + JF.v((0, -0.04, 0)), p1 + JF.v((0, ln1 * 0.6, 0)), p1 + JF.v((0, ln1, -0.02))], [0.045, 0.03, 0], sides=4, bone='jaw', shade=(0.8, 1))
            if t >= 2:   # wrist blades along the outer forearm, past the claws
                fa = seg(el, wr, (s, 0, 0), s)
                L_ = 0.9 + 0.45 * (t - 2)
                K.horn(part, [fa.p((0.4, -0.35, 0.0)), fa.p((0.46, -1.0, 0.12)), fa.p((0.44, -1.4 - L_ * 0.5, 0.3)), fa.p((0.36, -1.4 - L_, 0.45))],
                       [0.15, 0.22, 0.16, 0], mat='horn', sides=4, flat=0.25, bone='forearm' + sd, up=fa.v((1, 0, 0)))
                blk(part, fa, (0.42, -0.6, 0.0), (0.07, 0.3, 0.16), mat='dark', bone='forearm' + sd, bev=0.02)
                if t >= 3:
                    K.horn(part, [fa.p((0.5, -1.0, 0.14)), fa.p((0.5, -1.4 - L_ * 0.5, 0.33)), fa.p((0.42, -1.35 - L_, 0.48))],
                           [0.025, 0.03, 0], mat='accent', sides=4, flat=0.5, bone='forearm' + sd)
                    K.horn(part, [fa.p((0.0, -0.3, -0.38)), fa.p((0.05, 0.1, -0.6)), fa.p((0.05, 0.45, -0.95))],
                           [0.1, 0.06, 0], mat='horn', sides=4, flat=0.3, bone='forearm' + sd)            # elbow spur
        if t >= 3:   # snout drill
            base = HF.p((0, -0.05, 2.3))
            dd = unit(HF.v((0, 0.12, 1)))
            K.horn(part, [base - dd * 0.1, base + dd * 0.05], [0.26, 0.26], mat='dark', sides=8, bone='head')
            K.horn(part, [base, base + dd * 0.35, base + dd * 0.75], [0.22, 0.13, 0.0], mat='horn', sides=8, bone='head')
            for j in range(3):
                q = base + dd * (0.12 + 0.18 * j)
                K.horn(part, [q, q + dd * 0.04], [0.2 - 0.05 * j, 0.2 - 0.05 * j], mat='accent', sides=8, bone='head')
        elif t >= 1:   # nose ram ridge
            K.horn(part, [HF.p((0, 0.12, 1.75)), HF.p((0, 0.25 + 0.08 * t, 2.0)), HF.p((0, 0.3 + 0.14 * t, 2.1))], [0.1, 0.06, 0], mat='horn', sides=4, flat=0.35, bone='head')

    # ---- Armour (additive): t1 shoulder shields + thigh slabs, t2 tail guards + brow + shin guards, t3 spikes
    for s, sd in ((-1, 'L'), (1, 'R')):
        hip, kn, an, ball = legs[s]
        th = seg(hip, kn, (s, 0, 0), s)
        sn = seg(kn, an, (s, 0, 0), s)
        sh, el, wr, tip = arms[s]
        so = Fr(sh + np.array([s * 0.2, 0.25, -0.05]), np.diag([s, 1, 1]) @ Rz(-28) @ Rx(-10))
        poly('arm_t1', so, (0, 0, 0), [(-0.6, -0.3), (0.55, -0.4), (0.7, 0.15), (0.2, 0.5), (-0.5, 0.4)], 0.62, ax='z', mat='plate', bone='chest', bev=0.07, taper=0.9)
        blk('arm_t1', so, (0.05, 0.36, 0), (0.48, 0.05, 0.55), mat='belly', bone='chest', bev=0.02)
        for zz in (-0.4, 0.4):
            rivet('arm_t1', so, (0.68, 0.0, zz), (1, 0.3, 0), bone='chest', r=0.05)
        blk('arm_t1', th, (0.8, -0.95, 0.05), (0.06, 0.62, 0.55), mat='plate', bone='thigh' + sd, bev=0.03, chamf=0.15)
        for yy in (-0.5, -1.4):
            rivet('arm_t1', th, (0.87, yy, 0.0), (1, 0, 0), bone='thigh' + sd, r=0.05)
        poly('arm_t2', sn, (0.0, -1.0, 0.62), [(-0.38, 0.6), (0.38, 0.6), (0.34, -0.55), (0.0, -0.8), (-0.34, -0.55)], 0.06,
             mat='plate', bone='shin' + sd, bev=0.03)
        poly('arm_t2', sn, (0.5, -1.1, 0.1), [(-0.42, 0.55), (0.42, 0.55), (0.36, -0.6), (-0.36, -0.6)], 0.05, ax='x', mat='plate', bone='shin' + sd, bev=0.025)
        # t3 spikes: shoulder shield, knee, shin
        for c, n, ln in (((0.45, 0.35, 0.3), (0.7, 0.8, 0.2), 0.6), ((0.45, 0.35, -0.35), (0.7, 0.8, -0.3), 0.5), ((0.7, 0.05, 0.0), (1, 0.2, 0), 0.45)):
            K.spike('arm_t3', so.p(c), so.v(n), ln, r=0.12, sides=4, bone='chest')
        K.spike('arm_t3', sn.p((0.0, 0.1, 0.55)), sn.v((0, 0.5, 1)), 0.5, r=0.12, sides=4, bone='shin' + sd)
        K.spike('arm_t3', th.p((0.86, -0.95, 0.0)), th.v((1, 0.25, 0)), 0.45, r=0.11, sides=4, bone='thigh' + sd)
    blk('arm_t2', HF, (0, 0.88, 0.45), (0.5, 0.06, 0.5), mat='plate', bone='head', bev=0.03, chamf=0.15, taper=0.8)     # brow armour
    for i in range(4):
        a, b = TAIL[i], TAIL[i + 1]
        sl = float(np.linalg.norm(b - a))
        tfr = along((a + b) * 0.5, b - a)
        r = RING_R[i]
        for s in (-1, 1):
            g = tfr.sub((s * r * 0.72, 0, r * 0.62), Ry(s * 45))
            blk('arm_t2', g, (0, 0, 0.0), (r * 0.34, sl * 0.34, 0.06), mat='plate', bone=f'tail{i + 1}', bev=0.025, chamf=0.06)
            K.spike('arm_t3', tfr.p((s * r * 0.85, 0, r * 0.75)), tfr.v((s * 0.8, -0.35, 0.7)), 0.5 - 0.06 * i, r=0.1, sides=4, bone=f'tail{i + 1}')
    hf_ = along(TAIL[6], TAIL[6] - TAIL[5])
    for s in (-1, 1):
        for zz in (-0.3, 0.3):
            K.spike('arm_t3', hf_.p((s * 0.72, -0.15, zz)), hf_.v((s, 0.15, zz)), 0.42, r=0.11, sides=4, bone='tail6')

    # ---- Energy (additive): t1 forearm + calf + tail vents, t2 chest reactor, t3 glowing dorsal heat-sinks
    for s, sd in ((-1, 'L'), (1, 'R')):
        sh, el, wr, tip = arms[s]
        fa = seg(el, wr, (s, 0, 0), s)
        for j in range(3):
            blk('enr_t1', fa, (0.0, -0.45 - 0.22 * j, 0.39), (0.24, 0.045, 0.02), mat='accent', bone='forearm' + sd, bev=0.0)
        hip, kn, an, ball = legs[s]
        sn = seg(kn, an, (s, 0, 0), s)
        vent('enr_t1', sn.sub((0.46, -1.15, 0.1), Ry(90)), (0, 0, 0), 0.32, 0.45, 4, 'shin' + sd)
    for i in range(1, 4):
        a, b = TAIL[i], TAIL[i + 1]
        tfr = along((a + b) * 0.5, b - a)
        r = RING_R[i]
        for s in (-1, 1):
            blk('enr_t1', tfr, (s * r * 1.0, 0, -0.1), (0.02, 0.3, 0.07), mat='accent', bone=f'tail{i + 1}', bev=0.0)
    rc = TF.sub((0, 2.55, 1.05), Rx(10))
    poly('enr_t2', rc, (0, 0, 0), ngon(0.42, 6, 30), 0.08, mat='dark', bone='chest', bev=0.03)
    poly('enr_t2', rc, (0, 0, 0.08), ngon(0.34, 6, 30), 0.03, mat='horn', bone='chest', bev=0.01)
    glow('enr_t2', rc, (0, 0, 0.11), (0.27, 0.27, 0.06), bone='chest')
    glow('enr_t2', rc, (0, 0, 0.15), (0.12, 0.12, 0.04), mat='eye', bone='chest')
    for s in (-1, 1):
        blk('enr_t2', rc, (s * 0.55, 0, 0.05), (0.12, 0.05, 0.03), mat='accent', bone='chest', bev=0.0)
    for i, (yy, bn, w) in enumerate(secs):
        h = [0.55, 0.75, 0.95, 0.7][i]
        for s in (-1, 1):
            ff = Fr(TF.p((s * 0.36, yy, -1.26)), np.column_stack([TF.v((1, 0, 0)), unit(TF.v((s * 0.35, 0, -1))), np.cross(TF.v((1, 0, 0)), unit(TF.v((s * 0.35, 0, -1))))]))
            core = [(-0.24, 0.12), (0.26, 0.12), (-0.04, h * 0.78), (-0.3, h * 0.88)]
            poly('enr_t3', ff, (0, 0, 0), core, 0.065, ax='x', mat='accent', bone=bn, bev=0.0)
    hf_ = along(TAIL[6], TAIL[6] - TAIL[5])
    glow('enr_t3', hf_, (0, -0.15, 0.58), (0.4, 0.4, 0.04), rot=(0, 0, 0), bone='tail6')
    glow('enr_t3', hf_, (0, -0.15, -0.58), (0.4, 0.4, 0.04), rot=(0, 0, 0), bone='tail6')

    # ---- Speed (additive): t1 back thrusters, t2 calf thrusters + vanes, t3 tail stabiliser vanes
    for s, sd in ((-1, 'L'), (1, 'R')):
        thruster('spd_t1', TF, (s * 0.85, 2.2, -1.15), (s * 0.25, -0.75, -0.6), 0.5, 0.26, 'chest')
        thruster('spd_t1', TF, (s * 0.9, 1.2, -1.05), (s * 0.3, -0.8, -0.5), 0.42, 0.21, 'spine')
        hip, kn, an, ball = legs[s]
        sn = seg(kn, an, (s, 0, 0), s)
        thruster('spd_t2', sn, (0.0, -0.85, -0.3), (0, 0.3, -1), 0.32, 0.17, 'shin' + sd)
        vf_ = sn.sub((0.45, -0.7, -0.2))
        poly('spd_t2', vf_, (0, 0, 0), [(-0.1, 0.25), (0.15, 0.3), (-0.35, -0.75), (-0.75, -0.55)], 0.03, ax='x', mat='body', bone='shin' + sd, bev=0.01)
        poly('spd_t2', vf_, (0.03, 0, 0), [(-0.12, 0.12), (0.05, 0.16), (-0.35, -0.55), (-0.55, -0.45)], 0.015, ax='x', mat='accent', bone='shin' + sd, bev=0.0)
    tv = along(TAIL[5], TAIL[6] - TAIL[4])     # +Y toward the tip, +Z up
    for s in (-1, 1):   # horizontal stabilisers, swept toward the tip
        g = tv.sub((s * 0.5, -0.1, 0.0))
        poly('spd_t3', g, (0, 0, 0), [(0.0, -0.45), (0.0, 0.3), (s * 0.95, 0.65), (s * 1.05, 0.2)], 0.04, ax='z', mat='body', bone='tail5', bev=0.012)
        poly('spd_t3', g, (0, 0, 0), [(s * 0.15, -0.25), (s * 0.15, 0.2), (s * 0.8, 0.45), (s * 0.85, 0.15)], 0.055, ax='z', mat='accent', bone='tail5', bev=0.0)
    g = tv.sub((0, -0.1, 0.45))     # vertical stabiliser
    poly('spd_t3', g, (0, 0, 0), [(0.0, -0.45), (0.0, 0.3), (0.9, 0.6), (1.0, 0.15)], 0.04, ax='x', mat='body', bone='tail5', bev=0.012)
    poly('spd_t3', g, (0, 0, 0), [(0.12, -0.25), (0.12, 0.2), (0.75, 0.42), (0.8, 0.12)], 0.055, ax='x', mat='accent', bone='tail5', bev=0.0)

    # ---- Focus (additive): t1 cheek optics, t2 sensor mast array, t3 targeting crest + rangefinder
    for s in (-1, 1):
        K.ball('fcs_t1', HF.p((s * 0.72, 0.05, 0.55)), (0.03, 0.08, 0.08), mat='eye', bone='head')
        K.ball('fcs_t1', HF.p((s * 0.72, 0.05, 0.1)), (0.03, 0.06, 0.06), mat='eye', bone='head')
        a = HF.p((s * 0.4, 0.72, -0.4))
        K.horn('fcs_t2', [a, a + HF.v((s * 0.1, 0.45, -0.3)), a + HF.v((s * 0.18, 0.95, -0.55))], [0.045, 0.03, 0.012], mat='horn', sides=4, bone='head')
        K.ball('fcs_t2', a + HF.v((s * 0.07, 0.3, -0.2)), (0.07, 0.07, 0.07), mat='accent', bone='head')
        blk('fcs_t2', HF, (s * 0.4, 0.72, -0.4), (0.09, 0.06, 0.12), mat='dark', bone='head', bev=0.02)
    crest = [(-0.55, 0.0), (0.7, 0.0), (0.2, 0.45), (-0.65, 0.6)]
    poly('fcs_t3', HF, (0, 0.88, 0.05), crest, 0.04, ax='x', mat='body', bone='head', bev=0.012)
    poly('fcs_t3', HF, (0, 0.9, 0.05), [(-0.4, 0.08), (0.45, 0.08), (0.12, 0.36), (-0.48, 0.46)], 0.05, ax='x', mat='accent', bone='head', bev=0.0)
    blk('fcs_t3', HF, (0.68, 0.52, 0.75), (0.07, 0.11, 0.16), mat='dark', bone='head', bev=0.02)
    glow('fcs_t3', HF, (0.68, 0.52, 0.92), (0.08, 0.08, 0.03), mat='eye', bone='head', seg_=8)
