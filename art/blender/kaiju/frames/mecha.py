"""Mecha: humanity's giant robot (mech plan). Game coords, ~11.5 tall.

Built as a Jaeger / Gundam style hero: a dark mechanical inner frame (SDF skin: joints, pistons, core)
under crisp low-poly chamfered armour plates (rigid flat-shaded prisms joined into the body mesh).
The rest pose is a fighting stance: wide A-stance with the left foot forward, torso twisted, right
fist cocked in a guard in front of the chest, left arm hanging ready. Every limb part is authored in
its own segment frame (local -Y runs down the bone, +X points outward, +Z forward) so the pose is
just a choice of joint positions.
"""
import numpy as np

from .. import geom as G
from ..builder import POW_SCALE

DETAIL = 'metal'
UV_SCALE = 0.75
METAL = True
EYE_ACCENT = True


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
    """Local frame: origin O + 3x3 R (columns = local axes in world; det may be -1 for mirrored sides)."""

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


def seg(a, b, out, s=1):
    """Frame at joint a for the bone a->b: local -Y points at b, +X ~ `out` (outward), +Z forward-ish."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    y = unit(a - b)
    x = np.asarray(out, float) - y * (np.asarray(out, float) @ y)
    x = unit(x)
    z = np.cross(x, y) * s
    return Fr(a, np.column_stack([x, y, z]))


def ik(a, c, l1, l2, pole):
    d = c - a
    L = min(np.linalg.norm(d), l1 + l2 - 1e-3)
    u = unit(d)
    x = (l1 * l1 - l2 * l2 + L * L) / (2 * L)
    h = np.sqrt(max(l1 * l1 - x * x, 0.0))
    p = unit(np.asarray(pole, float) - u * (np.asarray(pole, float) @ u))
    return a + u * x + p * h


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
    """Convex 2D profile (u, v) extruded +-d along w with chamfered cap edges. taper scales the +w end
    (taper2 = separate v scale), shear shifts it. ax picks the local axis of the extrusion:
    'z': (u,v,w)=(x,y,z)  'x': (u,v,w)=(z,y,x)  'y': (u,v,w)=(x,z,y). Returns local (V, T)."""
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


# ------------------------------------------------------------------ pose (joints in world)
PELV = Fr((0, 5.65, 0), Ry(6))
CHEST = Fr((0, 6.5, 0), Ry(13) @ Rx(4))          # waist pivot; torso authored in this frame
NECK = CHEST.p((0, 3.42, 0.05))
HEADF = Fr(NECK + np.array([0, 0.7, 0.12]), 1.2 * Ry(3) @ Rx(6))
L_UA, L_FA = 1.75, 1.85
L_TH, L_SN = 2.45, 2.5


def arm_joints(s):
    sh = CHEST.p((s * 2.3, 2.72, -0.05))
    if s > 0:   # right: punch chambered at the hip, elbow drawn back
        el = sh + L_UA * unit((0.32, -0.82, -0.48))
        wr = el + L_FA * unit((-0.12, -0.12, 1.0))
        out = unit((1.0, 0.0, 0.1))
    else:       # left: leading guard, forearm raised in front
        el = sh + L_UA * unit((-0.28, -0.78, 0.56))
        wr = el + L_FA * unit((0.18, 0.62, 0.76))
        out = unit((-1.0, 0.0, 0.25))
    return sh, el, wr, out


def leg_joints(s):
    hip = PELV.p((s * 1.0, -0.2, 0))
    an = np.array([1.75, 1.0, -0.8]) if s > 0 else np.array([-1.62, 1.0, 0.85])
    kn = ik(hip, an, L_TH, L_SN, (s * 0.25, 0, 1))
    yaw = 15 if s > 0 else -9
    return hip, kn, an, yaw


# ------------------------------------------------------------------ build
def build(K):
    K.mat('horn', color='#d4d8de', rough=0.25, metal=0.85)        # bare steel spikes / blades
    K.mat('dark', color='#1d2026', rough=0.42, metal=0.75)
    K.mat('plate', rough=0.3, metal=0.75, detail=True, detail_kind='metal', detail_strength=0.05)
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

    def line(part, fr, a, b, n, w=0.028, bone=None, mat='dark'):
        """Panel-line strip from a to b (local), lying on a surface with local normal n."""
        a, b, n = np.asarray(a, float), np.asarray(b, float), unit(n)
        t = unit(b - a)
        x = unit(np.cross(t, n))
        R = np.column_stack([x, t, n])
        L = np.linalg.norm(b - a) / 2
        V, T = prism(rect(w, L), 0.02, 0.0)
        put(part, fr, V, T, (a + b) / 2, R, mat, bone)

    def glow(part, fr, c, r, rot=None, mat='accent', bone=None, seg=12):
        rr = fr.r(rot)
        V, T, _ = G.sphere((0, 0, 0), r, seg, 6)
        Vw = fr.p(c) + V @ rr.T
        K.piece(part, Vw, T, mat=mat, bone=bone)

    def thruster(part, fr, c, d, ln, r, bone, glow_mat='accent'):
        base, dd = fr.p(c), unit(fr.v(d))
        K.horn(part, [base - dd * 0.1, base + dd * ln * 0.45, base + dd * ln], [r * 0.62, r * 0.78, r], mat='dark', sides=10, bone=bone)
        K.horn(part, [base + dd * (ln - 0.06), base + dd * (ln + 0.01)], [r * 1.06, r * 1.06], mat='horn', sides=10, bone=bone)
        x, _, z = G.basis(dd)
        cc = base + dd * (ln - 0.12)
        V = [cc] + [cc + (np.cos(a) * x + np.sin(a) * z) * r * 0.86 for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
        T = [(0, 1 + i, 1 + (i + 1) % 10) for i in range(10)] + [(0, 1 + (i + 1) % 10, 1 + i) for i in range(10)]
        K.piece(part, np.array(V), np.array(T), mat=glow_mat, bone=bone)

    # ================================================================ inner frame (dark SDF skin)
    B = K.sdf()
    k0 = 0.06

    def sbox(fr, c, half, rot=None, rad=0.08, k=k0):
        B.box(fr.p(c), half, rot=fr.r(rot), rad=rad, k=k)

    def stube(fr, pts, r, k=k0):
        B.tube([fr.p(p) for p in pts], r, k=k)

    # pelvis + waist + torso core
    sbox(PELV, (0, 0.05, -0.05), (0.95, 0.45, 0.7), rad=0.2)
    stube(CHEST, [(0, -0.7, 0), (0, 0.9, 0.02)], [0.62, 0.55])
    for yy in (-0.2, 0.2, 0.6):
        stube(CHEST, [(0, yy - 0.06, 0), (0, yy + 0.06, 0)], [0.7, 0.7], k=0.03)       # waist rings
    sbox(CHEST, (0, 2.15, -0.2), (1.2, 1.0, 0.8), rad=0.25)
    stube(CHEST, [(0, 2.9, 0.0), (0, 3.6, 0.1)], [0.36, 0.33])                          # neck
    for yy in (3.15, 3.4):
        stube(CHEST, [(0, yy - 0.04, 0.05), (0, yy + 0.04, 0.06)], [0.42, 0.42], k=0.02)

    arms = {s: arm_joints(s) for s in (-1, 1)}
    legs = {s: leg_joints(s) for s in (-1, 1)}
    for s in (-1, 1):
        sh, el, wr, out = arms[s]
        ua = seg(sh, el, (s, 0, 0), s)
        fa = seg(el, wr, out, s)
        B.ell(sh, (0.45, 0.45, 0.45), k=k0)
        stube(ua, [(0, -0.2, 0), (0, -L_UA + 0.25, 0)], [0.36, 0.33])
        for yy in (-0.6, -0.95):
            stube(ua, [(0, yy - 0.05, 0), (0, yy + 0.05, 0)], [0.42, 0.42], k=0.02)
        stube(ua, [(-0.12, -L_UA, 0), (0.12, -L_UA, 0)], [0.31, 0.31])                 # elbow axle
        stube(fa, [(0, -0.1, 0), (0, -L_FA + 0.1, 0)], [0.4, 0.34])
        stube(fa, [(0, -L_FA + 0.05, 0), (0, -L_FA - 0.25, 0)], [0.3, 0.28])            # wrist
        # pistons on the inside of the elbow
        stube(ua, [(-0.25, -0.9, 0.3), (-0.2, -L_UA - 0.6, 0.35)], [0.09, 0.09], k=0.02)
        hip, kn, an, yaw = legs[s]
        th = seg(hip, kn, (s, 0, 0), s)
        sn = seg(kn, an, (s, 0, 0), s)
        B.ell(hip, (0.42, 0.42, 0.42), k=k0)
        stube(th, [(0, -0.1, 0), (0, -L_TH + 0.2, 0)], [0.36, 0.3])
        stube(th, [(-0.18, -L_TH, 0), (0.18, -L_TH, 0)], [0.34, 0.34])                 # knee axle
        stube(sn, [(0, -0.3, 0), (0, -L_SN + 0.2, 0)], [0.32, 0.34])
        stube(sn, [(0.0, -0.4, -0.45), (0.0, -1.7, -0.55)], [0.12, 0.12], k=0.02)       # calf piston
        stube(th, [(0.0, -0.9, -0.42), (0.0, -L_TH - 0.5, -0.5)], [0.1, 0.1], k=0.02)   # knee piston
        B.ell(an, (0.34, 0.34, 0.34), k=k0)
    K.skin(B, voxel=0.045, tris=9000, base_mat='dark', custom_normals=False, sharp=40, ao=0.5)

    # ================================================================ skeleton + mounts
    K.bone('pelvis', PELV.p((0, -0.15, 0)), CHEST.O, None, 1.1)
    K.bone('spine', CHEST.O, CHEST.p((0, 1.3, 0)), 'pelvis', 1.0)
    K.bone('chest', CHEST.p((0, 1.3, 0)), NECK, 'spine', 1.6)
    K.bone('neck', NECK, HEADF.p((0, -0.35, -0.05)), 'chest', 0.4)
    K.bone('head', HEADF.p((0, -0.35, -0.05)), HEADF.p((0, 0.45, 0.1)), 'neck', 0.5)
    K.bone('jaw', HEADF.p((0, -0.2, 0.05)), HEADF.p((0, -0.38, 0.5)), 'head', 0.2)
    for s, sd in ((-1, 'L'), (1, 'R')):
        sh, el, wr, out = arms[s]
        fa = seg(el, wr, out, s)
        tip = fa.p((0, -L_FA - 0.9, 0))
        K.bone('upperarm' + sd, sh, el, 'chest', 0.55)
        K.bone('forearm' + sd, el, wr, 'upperarm' + sd, 0.6)
        K.bone('hand' + sd, wr, tip, 'forearm' + sd, 0.5)
        hip, kn, an, yaw = legs[s]
        toe = an + Ry(yaw) @ np.array([0, -0.6, 1.6])
        K.bone('thigh' + sd, hip, kn, 'pelvis', 0.6)
        K.bone('shin' + sd, kn, an, 'thigh' + sd, 0.75)
        K.bone('foot' + sd, an, toe, 'shin' + sd, 0.55)
        K.mount('hand' + sd, fa.p((0, -L_FA - 0.5, 0)), 'hand' + sd)
        K.mount('shoulder' + sd, CHEST.p((s * 2.65, 3.75, 0.0)), 'chest')
    K.mount('head', HEADF.O, 'head')
    K.mount('jaw', HEADF.p((0, -0.32, 0.45)), 'jaw')
    K.mount('mouth', HEADF.p((0, -0.18, 0.62)), 'head')
    K.mount('chest', CHEST.p((0, 2.05, 1.35)), 'chest')
    K.mount('back', CHEST.p((0, 2.0, -2.0)), 'chest')
    K.mount('tailTip', PELV.p((0, -0.1, -1.1)), 'pelvis')
    K.mount('crown', HEADF.p((0, 0.75, 0)), 'head')

    # ================================================================ armour: pelvis + skirt
    P = 'body'
    blk(P, PELV, (0, 0.18, 0.0), (0.9, 0.32, 0.78), bone='pelvis', bev=0.08, chamf=0.12)          # waist band
    blk(P, PELV, (0, -0.25, 0.72), (0.38, 0.4, 0.1), rot=(-12, 0, 0), mat='belly', bone='pelvis', bev=0.04, taper=0.85)  # codpiece
    poly(P, PELV, (0, -0.1, 0.88), [(-0.32, 0.3), (0.32, 0.3), (0.22, -0.45), (0, -0.62), (-0.22, -0.45)], 0.06,
         rot=(-12, 0, 0), mat='body', bone='pelvis', bev=0.03)
    for s in (-1, 1):
        sd = 'L' if s < 0 else 'R'
        fr = Fr(PELV.O, PELV.R @ np.diag([s, 1, 1]))
        # front skirt plates hang off the waist (angled out over the thighs)
        poly(P, fr, (0.68, -0.3, 0.62), [(-0.3, 0.32), (0.32, 0.32), (0.36, -0.55), (-0.18, -0.62)], 0.07,
             rot=(-14, 18, 6), mat='belly', bone='pelvis', bev=0.035)
        # side skirts
        poly(P, fr, (1.12, -0.25, 0.0), [(-0.5, 0.35), (0.5, 0.35), (0.42, -0.6), (-0.4, -0.6)], 0.08,
             rot=(0, 90, 14), mat='body', bone='pelvis', bev=0.04)
        line(P, fr, (1.22, -0.05, -0.38), (1.22, -0.05, 0.38), (1, 0, 0), bone='pelvis')
        # rear skirt
        poly(P, fr, (0.5, -0.25, -0.72), [(-0.42, 0.32), (0.42, 0.32), (0.36, -0.5), (-0.36, -0.5)], 0.07,
             rot=(14, -14, 0), mat='body', bone='pelvis', bev=0.035)

    # ================================================================ armour: torso
    C = CHEST
    cb = 'chest'
    # abdomen: three stacked chamfered plates over the dark waist
    for i, yy in enumerate((-0.15, 0.25, 0.65)):
        blk(P, C, (0, yy, 0.5 + 0.02 * i), (0.5 + 0.06 * i, 0.17, 0.16), rot=(-6, 0, 0), bone='spine', bev=0.05, chamf=0.07)
    for s in (-1, 1):
        fr = Fr(C.O, C.R @ np.diag([s, 1, 1]))
        blk(P, fr, (0.62, 0.35, 0.12), (0.12, 0.55, 0.45), rot=(0, 0, -8), bone='spine', bev=0.04)   # oblique plates
    # main torso shell: inverted trapezoid
    shell = [(-0.72, 1.05), (0.72, 1.05), (1.5, 2.15), (1.92, 2.95), (1.2, 3.25), (-1.2, 3.25), (-1.92, 2.95), (-1.5, 2.15)]
    poly(P, C, (0, 0, -0.12), shell, 0.85, mat='body', bone=cb, bev=0.12, taper=0.86, taper2=0.95)
    # lower chest: ribcage wedge under the pecs (narrow -> the inverted triangle)
    poly(P, C, (0, 1.32, 0.8), [(-0.7, -0.3), (0.7, -0.3), (1.0, 0.25), (0, 0.4), (-1.0, 0.25)], 0.12,
         rot=(-14, 0, 0), mat='body', bone=cb, bev=0.05)
    for s in (-1, 1):
        fr = Fr(C.O, C.R @ np.diag([s, 1, 1]))
        # pectoral plate: hinged at the centre line and swept back -> a prow-shaped chest
        pf = fr.sub((0.06, 1.62, 0.98), (-7, 17, 0))
        pec = [(0.0, -0.3), (0.7, -0.22), (1.62, 0.72), (1.6, 1.38), (0.0, 1.4)]
        poly(P, pf, (0, 0, 0), pec, 0.16, mat='body', bone=cb, bev=0.07)
        poly(P, pf, (0.06, 0.82, 0.17), [(0.0, 0.0), (1.2, 0.0), (1.42, 0.42), (0.0, 0.48)], 0.05,
             mat='belly', bone=cb, bev=0.03)                                                # upper trim stripe
        # chest vents: dark recess + louvres in the lower pec
        blk(P, pf, (0.98, 0.32, 0.14), (0.4, 0.2, 0.06), mat='dark', bone=cb, bev=0.02, chamf=0.05)
        for j in range(3):
            blk(P, pf, (0.98, 0.2 + 0.12 * j, 0.2), (0.36, 0.025, 0.05), rot=(30, 0, 0), mat='belly', bone=cb, bev=0.0)
        # collar / trapezius armour rising toward the neck
        poly(P, fr, (0.75, 3.22, -0.1), [(-0.55, -0.18), (0.55, -0.18), (0.45, 0.22), (-0.35, 0.3)], 0.62,
             rot=(0, 0, -8), mat='body', bone=cb, bev=0.07, taper=0.9)
        # flank plates under the arms
        poly(P, fr, (1.2, 1.62, -0.05), [(-0.6, -0.5), (0.6, -0.5), (0.55, 0.5), (-0.55, 0.5)], 0.08,
             rot=Rz(-38) @ Ry(90), mat='body', bone=cb, bev=0.04)                             # V-flank plates
    # reactor: dark hex housing, steel bezel, glowing core
    poly(P, C, (0, 1.98, 1.12), ngon(0.44, 6, 30), 0.16, mat='dark', bone=cb, bev=0.04)
    poly(P, C, (0, 1.98, 1.25), ngon(0.36, 6, 30), 0.05, mat='horn', bone=cb, bev=0.02)
    glow(P, C, (0, 1.98, 1.3), (0.28, 0.28, 0.06), bone=cb)
    glow(P, C, (0, 1.98, 1.34), (0.14, 0.14, 0.04), mat='eye', bone=cb)
    # collar vent + neck guard
    blk(P, C, (0, 3.1, 0.62), (0.5, 0.16, 0.18), rot=(20, 0, 0), mat='belly', bone=cb, bev=0.04, chamf=0.06)
    for j in range(3):
        blk(P, C, (0, 2.98 + 0.09 * j, 0.78), (0.36, 0.022, 0.05), rot=(20, 0, 0), mat='dark', bone=cb, bev=0.0)

    # ---- backpack: armoured block, main thrusters, booster pods above the shoulders
    blk(P, C, (0, 1.95, -1.3), (1.05, 1.05, 0.45), bone=cb, bev=0.09, chamf=0.18, taper=0.88)
    blk(P, C, (0, 2.0, -1.78), (0.65, 0.75, 0.08), mat='belly', bone=cb, bev=0.03, chamf=0.1)
    line(P, C, (0, 1.1, -1.76), (0, 2.85, -1.76), (0, 0, -1), bone=cb)
    for s in (-1, 1):
        fr = Fr(C.O, C.R @ np.diag([s, 1, 1]))
        pod = fr.sub((0.7, 2.9, -1.55), E(-32, 0, -22))
        blk(P, pod, (0, 0, 0), (0.22, 0.6, 0.24), bone=cb, bev=0.05, chamf=0.08, taper=0.8)
        blk(P, pod, (0, 0.62, 0), (0.15, 0.08, 0.17), mat='belly', bone=cb, bev=0.03)
        thruster(P, fr, (0.6, 1.05, -1.5), (0.05, -0.85, -0.5), 0.62, 0.36, cb)
        thruster(P, pod, (0.0, -0.2, -0.22), (0.0, -0.3, -1.0), 0.3, 0.16, cb)

    # ================================================================ armour: arms
    for s, sd in ((-1, 'L'), (1, 'R')):
        sh, el, wr, out = arms[s]
        ua = seg(sh, el, (s, 0, 0), s)
        fa = seg(el, wr, out, s)
        fr = Fr(C.O, C.R @ np.diag([s, 1, 1]))
        shl = fr.sub((2.3, 2.72, -0.05))
        # pauldron: big layered shoulder armour on the chest bone (Jaeger style)
        paul = [(-0.55, -0.35), (0.85, -0.75), (1.12, -0.1), (1.0, 0.62), (0.3, 0.95), (-0.45, 0.72)]
        poly(P, shl, (0.12, 0.25, 0.0), paul, 0.92, rot=(0, 0, -6), bone=cb, bev=0.14, taper=0.9, taper2=0.92)
        poly(P, shl, (0.2, 0.92, 0.0), [(-0.62, -0.12), (0.62, -0.18), (0.75, 0.12), (-0.5, 0.2)], 0.82,
             rot=(0, 0, -22), mat='belly', bone=cb, bev=0.06, taper=0.92)                 # top stripe plate
        poly(P, shl, (0.72, -0.72, 0.0), [(-0.32, 0.35), (0.32, 0.4), (0.38, -0.45), (-0.25, -0.38)], 0.72,
             rot=(0, 0, 12), bone=cb, bev=0.06, taper=0.9)                                # hanging lame
        pq = shl.sub((1.16, -0.1, 0.0), (0, 0, -8))                                       # outer plaque: chevron
        poly(P, pq, (0, 0, 0), [(-0.62, 0.42), (0.5, 0.52), (0.66, 0.0), (0.3, -0.5), (-0.55, -0.42)], 0.06,
             mat='belly', bone=cb, bev=0.025, ax='x')
        poly(P, pq, (0.06, 0.02, 0.0), [(-0.38, 0.22), (0.3, 0.28), (0.4, 0.0), (0.18, -0.26), (-0.34, -0.2)], 0.03,
             mat='body', bone=cb, bev=0.015, ax='x')
        blk(P, pq, (0.095, 0.02, -0.05), (0.01, 0.03, 0.26), mat='accent', bone=cb, bev=0.0)
        # missile hatch on top-rear of the pauldron
        blk(P, shl, (0.1, 1.08, -0.42), (0.42, 0.14, 0.3), rot=(0, 0, -22), mat='dark', bone=cb, bev=0.03)
        for i in range(3):
            glow(P, shl, (-0.15 + 0.25 * i, 1.12 + 0.1 * i, -0.42), (0.06, 0.04, 0.06), rot=(0, 0, -22), bone=cb, seg=6)
        # upper arm sleeve
        ub = 'upperarm' + sd
        blk(P, ua, (0.02, -0.95, 0.0), (0.4, 0.42, 0.42), bone=ub, bev=0.06, chamf=0.12, ax='y', taper=1.08)
        blk(P, ua, (0.43, -0.95, 0.0), (0.05, 0.36, 0.3), mat='belly', bone=ub, bev=0.02, chamf=0.08)
        # elbow axle caps (hex bosses over the dark joint)
        for sx in (-1, 1):
            poly(P, ua, (sx * 0.42, -L_UA, 0), ngon(0.29, 6, 0), 0.06, mat='body', bone=ub, bev=0.025, ax='x')
        poly(P, ua, (0.49, -L_UA, 0), ngon(0.14, 6, 0), 0.03, mat='horn', bone=ub, bev=0.01, ax='x')
        # elbow cap
        poly(P, ua, (0.0, -L_UA, -0.35), [(-0.35, 0.3), (0.35, 0.3), (0.3, -0.25), (0.0, -0.45), (-0.3, -0.25)], 0.16,
             rot=(18, 0, 0), bone=ub, bev=0.05)
        # forearm: big gauntlet, flared toward the wrist
        fb = 'forearm' + sd
        blk(P, fa, (0.0, -1.0, 0.0), (0.58, 0.82, 0.6), bone=fb, bev=0.08, chamf=0.2, ax='y', taper=0.78)
        blk(P, fa, (0.0, -1.72, 0.0), (0.64, 0.14, 0.66), mat='belly', bone=fb, bev=0.05, chamf=0.22, ax='y')   # cuff
        poly(P, fa, (0.6, -0.95, 0.0), [(-0.45, 0.6), (0.3, 0.75), (0.55, -0.6), (-0.5, -0.75)], 0.07,
             mat='body', bone=fb, bev=0.035, ax='x')                                       # outer plate
        poly(P, fa, (0.67, -1.0, 0.05), [(-0.12, 0.5), (0.12, 0.55), (0.14, -0.55), (-0.1, -0.6)], 0.02,
             mat='belly', bone=fb, bev=0.0, ax='x')
        line(P, fa, (0.665, -0.4, -0.25), (0.665, -1.45, -0.32), (1, 0, 0), bone=fb)
        blk(P, fa, (0.0, -0.95, 0.6), (0.3, 0.55, 0.05), mat='dark', bone=fb, bev=0.02)    # vent recess
        for j in range(3):
            blk(P, fa, (0.0, -0.7 - 0.22 * j, 0.63), (0.26, 0.04, 0.05), rot=(-25, 0, 0), bone=fb, bev=0.0)
        # fist (hand frame = continuation of the forearm)
        hb = 'hand' + sd
        hf = Fr(fa.p((0, -L_FA, 0)), fa.R * 1.15)
        blk(P, hf, (0.0, -0.42, 0.0), (0.32, 0.36, 0.44), mat='dark', bone=hb, bev=0.04, chamf=0.06)   # palm
        blk(P, hf, (0.3, -0.42, 0.0), (0.1, 0.4, 0.46), bone=hb, bev=0.04, chamf=0.1)                   # back of hand
        for i, zz in enumerate((-0.33, -0.11, 0.11, 0.33)):
            blk(P, hf, (0.12, -0.86, zz), (0.24, 0.13, 0.1), mat='dark', bone=hb, bev=0.03)        # proximal
            blk(P, hf, (-0.16, -0.78, zz), (0.13, 0.2, 0.1), mat='dark', bone=hb, bev=0.03)        # curled tips
            blk(P, hf, (0.3, -0.9, zz), (0.07, 0.11, 0.09), mat='body', bone=hb, bev=0.02)         # knuckle caps
        blk(P, hf, (-0.18, -0.48, 0.5), (0.14, 0.26, 0.13), rot=(0, 0, 20), mat='dark', bone=hb, bev=0.03)  # thumb

    # ================================================================ armour: legs
    for s, sd in ((-1, 'L'), (1, 'R')):
        hip, kn, an, yaw = legs[s]
        th = seg(hip, kn, (s, 0, 0), s)
        sn = seg(kn, an, (s, 0, 0), s)
        tb, sb, fb = 'thigh' + sd, 'shin' + sd, 'foot' + sd
        # thigh
        blk(P, th, (0.0, -1.12, 0.04), (0.5, 0.98, 0.52), bone=tb, bev=0.07, chamf=0.16, ax='y', taper=1.1)
        poly(P, th, (0.55, -1.1, 0.0), [(-0.4, 0.75), (0.4, 0.68), (0.42, -0.85), (-0.35, -0.9)], 0.07,
             mat='belly', bone=tb, bev=0.035, ax='x')
        blk(P, th, (0.0, -1.2, 0.56), (0.32, 0.6, 0.06), rot=(4, 0, 0), mat='body', bone=tb, bev=0.03, chamf=0.08)
        line(P, th, (-0.25, -0.5, 0.58), (-0.25, -1.55, 0.6), (0, 0, 1), bone=tb)
        # knee axle caps
        for sx in (-1, 1):
            poly(P, th, (sx * 0.5, -L_TH, 0), ngon(0.34, 6, 0), 0.07, mat='body', bone=sb, bev=0.03, ax='x')
        poly(P, th, (0.58, -L_TH, 0), ngon(0.16, 6, 0), 0.03, mat='horn', bone=sb, bev=0.01, ax='x')
        # knee armour: pointed cap jutting forward
        poly(P, sn, (0.0, 0.1, 0.42), [(-0.46, 0.5), (0.46, 0.5), (0.44, -0.35), (0.0, -0.7), (-0.44, -0.35)], 0.24,
             rot=(-10, 0, 0), mat='belly', bone=sb, bev=0.08, taper=0.75)
        # shin: massive lower leg, flaring toward the ankle
        blk(P, sn, (0.0, -1.4, -0.05), (0.68, 1.05, 0.8), bone=sb, bev=0.09, chamf=0.22, ax='y', taper=0.8)
        poly(P, sn, (0.0, -1.3, 0.77), [(-0.48, 0.85), (0.48, 0.85), (0.62, -0.6), (0.0, -1.0), (-0.62, -0.6)], 0.1,
             rot=(-6, 0, 0), mat='belly', bone=sb, bev=0.05, taper=0.8)                   # shin guard
        line(P, sn, (0.0, -0.55, 0.88), (0.0, -1.95, 0.9), (0, 0, 1), bone=sb, w=0.022)
        poly(P, sn, (0.7, -1.35, -0.05), [(-0.55, 0.55), (0.55, 0.55), (0.5, -0.7), (-0.6, -0.7)], 0.08,
             mat='body', bone=sb, bev=0.04, ax='x', taper=0.85)                            # outer calf plate
        line(P, sn, (0.785, -0.95, -0.3), (0.785, -1.8, -0.35), (1, 0, 0), bone=sb)
        blk(P, sn, (0.0, -1.2, -0.85), (0.4, 0.55, 0.1), mat='dark', bone=sb, bev=0.03)   # calf vent
        for j in range(4):
            blk(P, sn, (0.0, -0.85 - 0.2 * j, -0.92), (0.36, 0.035, 0.06), rot=(20, 0, 0), bone=sb, bev=0.0)
        # foot: authored relative to the ankle, flat on the ground
        ff = Fr(an, Ry(yaw) @ np.diag([s, 1, 1]))
        poly(P, ff, (0, -1.0, 0), [(-0.85, 0.02), (1.25, 0.02), (1.38, 0.22), (0.85, 0.55), (-0.15, 0.78), (-0.8, 0.6)],
             0.6, mat='body', bone=fb, bev=0.08, ax='x', taper=0.94)
        poly(P, ff, (0, -1.0, 0), [(0.75, 0.02), (1.75, 0.02), (1.82, 0.18), (1.25, 0.46), (0.7, 0.42)],
             0.52, mat='belly', bone=fb, bev=0.06, ax='x', taper=0.85)                     # toe cap
        blk(P, ff, (0, -0.95, 0.35), (0.5, 0.06, 1.3), mat='dark', bone=fb, bev=0.02)       # sole
        poly(P, ff, (0, -1.0, 0), [(-1.12, 0.02), (-0.5, 0.02), (-0.5, 0.45), (-0.9, 0.38)], 0.4,
             mat='body', bone=fb, bev=0.05, ax='x')                                         # heel spur
        for sx in (-1, 1):           # ankle guards
            blk(P, ff, (sx * 0.5, -0.12, -0.05), (0.06, 0.3, 0.34), rot=(0, 0, -sx * 8), mat='body', bone=fb, bev=0.03, chamf=0.12)

    # ================================================================ head
    H = HEADF
    hb = 'head'
    # helmet: top-view arrowhead prism (faceted front), crest ridge on top
    poly(P, H, (0, 0.06, 0), [(-0.4, -0.5), (0.4, -0.5), (0.42, 0.2), (0.12, 0.42), (-0.12, 0.42), (-0.42, 0.2)], 0.3,
         ax='y', bone=hb, bev=0.07, taper=0.86)
    poly(P, H, (0, 0.36, -0.05), [(-0.2, -0.05), (0.2, -0.05), (0.14, 0.12), (-0.14, 0.12)], 0.48,
         mat='belly', bone=hb, bev=0.04, taper=0.7)                                             # crest ridge
    blk(P, H, (0, 0.04, 0.425), (0.12, 0.09, 0.02), mat='dark', bone=hb, bev=0.0)
    blk(P, H, (0, 0.035, 0.445), (0.12, 0.03, 0.02), mat='eye', bone=hb, bev=0.0)
    blk(P, H, (0, 0.17, 0.43), (0.12, 0.045, 0.05), mat='body', bone=hb, bev=0.02)                # brow
    blk(P, H, (0, -0.2, 0.4), (0.12, 0.14, 0.07), rot=(-12, 0, 0), mat='belly', bone='jaw', bev=0.03)  # muzzle
    for s in (-1, 1):
        fr = Fr(H.O, H.R @ np.diag([s, 1, 1]))
        ff = fr.sub((0.12, 0, 0.42), Ry(36.3))                                                   # front facet frame
        blk(P, ff, (0.18, 0.04, 0.005), (0.19, 0.09, 0.02), mat='dark', bone=hb, bev=0.0)         # visor recess
        blk(P, ff, (0.16, 0.05, 0.025), (0.17, 0.03, 0.02), rot=(0, 0, 7), mat='eye', bone=hb, bev=0.0)  # visor slit
        blk(P, ff, (0.17, 0.17, 0.03), (0.21, 0.045, 0.05), rot=(0, 0, 9), mat='body', bone=hb, bev=0.02)  # scowling brow
        poly(P, ff, (0.0, -0.2, -0.02), [(0.0, 0.12), (0.3, 0.1), (0.24, -0.06), (0.0, -0.16)], 0.08,
             mat='belly', bone='jaw', bev=0.03)                                                   # muzzle facet
        blk(P, ff, (0.18, -0.17, 0.065), (0.04, 0.06, 0.02), mat='dark', bone='jaw', bev=0.0)
        poly(P, fr, (0.43, -0.1, -0.08), [(-0.38, 0.26), (0.36, 0.2), (0.28, -0.3), (-0.3, -0.3)], 0.06,
             rot=(0, 90, 6), mat='body', bone=hb, bev=0.025)                                    # cheek guards
        blk(P, fr, (0.48, 0.02, -0.15), (0.04, 0.11, 0.13), mat='dark', bone=hb, bev=0.02)        # ear vents
        # V-fin antenna blades
        poly(P, fr, (0.0, 0.2, 0.36), [(0.03, 0.0), (0.82, 0.55), (0.85, 0.63), (0.04, 0.14)], 0.025,
             rot=(-8, 0, 0), mat='belly', bone=hb, bev=0.0)
    poly(P, H, (0, 0.27, 0.44), [(-0.09, -0.06), (0.09, -0.06), (0.0, 0.1)], 0.05, mat='accent', bone=hb, bev=0.0)  # crest jewel

    # ================================================================ tiers
    # ---- Might (exclusive): knuckle guard -> armoured gauntlet -> spiked breaker -> rocket blade fist
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        for s, sd in ((-1, 'L'), (1, 'R')):
            sh, el, wr, out = arms[s]
            fa = seg(el, wr, out, s)
            hf = Fr(fa.p((0, -L_FA, 0)), fa.R * 1.15)
            hb_, fb_ = 'hand' + sd, 'forearm' + sd
            if t == 0:
                blk(part, hf, (0.12, -0.98, 0.0), (0.2, 0.05, 0.42), mat='plate', bone=hb_, bev=0.02)
                continue
            blk(part, hf, (0.12, -0.99 - 0.02 * t, 0.0), (0.26, 0.07 + 0.02 * t, 0.48), mat='plate', bone=hb_, bev=0.03, chamf=0.05)
            blk(part, hf, (0.42, -0.5, 0.0), (0.06, 0.45, 0.48), mat='plate', bone=hb_, bev=0.03, chamf=0.12)
            if t >= 2:
                for zz in (-0.33, -0.11, 0.11, 0.33):
                    p = hf.p((0.15, -1.05 - 0.02 * t, zz))
                    K.spike(part, p, hf.v((0.15, -1, 0)), 0.18 + 0.25 * (sc - 1), r=0.07 + 0.012 * t, sides=4, bone=hb_)
                # elbow blade sweeping back
                K.horn(part, [fa.p((0.1, 0.1, -0.45)), fa.p((0.15, 0.45, -0.75)), fa.p((0.12, 0.75, -1.05 - 0.2 * t))],
                       [0.14, 0.08, 0], mat='horn', sides=4, flat=0.3, bone=fb_)
            if t >= 3:
                # long blade along the outer forearm, past the fist, + glowing knuckle line
                K.horn(part, [fa.p((0.72, -0.2, 0.0)), fa.p((0.82, -1.0, 0.0)), fa.p((0.82, -1.9, 0.0)), fa.p((0.7, -2.8, 0.0))],
                       [0.16, 0.2, 0.14, 0], mat='horn', sides=4, flat=0.22, bone=fb_, up=fa.v((0, 0, 1)))
                glow(part, hf, (0.36, -0.95, 0.0), (0.04, 0.05, 0.42), bone=hb_, seg=8)
                thruster(part, fa, (0.0, -0.25, -0.62), (0, 0.6, -1), 0.3, 0.16, fb_)
        if t >= 2:   # central blade horn over the V-fin
            K.horn(part, [HEADF.p((0, 0.35, 0.25)), HEADF.p((0, 0.6, 0.4)), HEADF.p((0, 0.9 + 0.15 * t, 0.3))],
                   [0.08, 0.05, 0], mat='horn', sides=4, flat=0.4, bone='head')

    # ---- Armour (additive): bolt-on plates; t3 spikes
    for s, sd in ((-1, 'L'), (1, 'R')):
        hip, kn, an, yaw = legs[s]
        th = seg(hip, kn, (s, 0, 0), s)
        sn = seg(kn, an, (s, 0, 0), s)
        fr = Fr(C.O, C.R @ np.diag([s, 1, 1]))
        shl = fr.sub((2.3, 2.72, -0.05))
        sh, el, wr, out = arms[s]
        ua = seg(sh, el, (s, 0, 0), s)
        # t1: extra pauldron layer + thigh plates
        poly('arm_t1', shl, (0.78, 0.15, 0.0), [(-0.25, -0.75), (0.25, -0.8), (0.32, 0.65), (-0.2, 0.75)], 1.0,
             rot=(0, 0, -6), mat='plate', bone=cb, bev=0.06, taper=0.88)
        blk('arm_t1', th, (0.0, -0.95, 0.64), (0.4, 0.6, 0.07), mat='plate', bone='thigh' + sd, bev=0.03, chamf=0.12)
        # t2: chest plating, shin side armour, upper-arm plates
        pf = fr.sub((0.06, 1.62, 0.98), (-7, 17, 0))
        poly('arm_t2', pf, (0.1, 0.78, 0.25), [(0.0, 0.0), (1.28, 0.0), (1.55, 0.55), (0.0, 0.66)], 0.07,
             mat='plate', bone=cb, bev=0.03)
        poly('arm_t2', sn, (0.82, -1.35, -0.05), [(-0.5, 0.5), (0.5, 0.5), (0.45, -0.6), (-0.55, -0.62)], 0.06,
             mat='plate', bone='shin' + sd, bev=0.03, ax='x', taper=0.85)
        blk('arm_t2', ua, (0.48, -0.9, 0.0), (0.07, 0.55, 0.42), mat='plate', bone='upperarm' + sd, bev=0.03, chamf=0.12)
        # t3: spikes
        for c, n, ln in (((1.15, 0.5, 0.0), (1, 0.55, 0), 0.8), ((0.9, 0.85, 0.6), (0.6, 0.8, 0.4), 0.55),
                         ((0.9, 0.85, -0.6), (0.6, 0.8, -0.4), 0.55)):
            K.spike('arm_t3', shl.p(c), shl.v(n), ln, r=0.14, sides=4, bone=cb)
        K.spike('arm_t3', sn.p((0.0, -0.05, 0.75)), sn.v((0, 0.45, 1)), 0.5, r=0.12, sides=4, bone='shin' + sd)
        K.spike('arm_t3', sn.p((0.86, -1.1, -0.05)), sn.v((1, 0.3, 0)), 0.42, r=0.1, sides=4, bone='shin' + sd)
        K.spike('arm_t3', sn.p((0.86, -1.6, -0.05)), sn.v((1, 0.3, 0)), 0.36, r=0.09, sides=4, bone='shin' + sd)
    poly('arm_t2', C, (0, 0.3, 0.72), [(-0.62, 0.45), (0.62, 0.45), (0.42, -0.45), (-0.42, -0.45)], 0.06, rot=(-6, 0, 0), mat='plate', bone='spine', bev=0.03)

    # ---- Energy (additive): capacitors -> shoulder cores -> forearm coils + chest strips
    for s, sd in ((-1, 'L'), (1, 'R')):
        fr = Fr(C.O, C.R @ np.diag([s, 1, 1]))
        shl = fr.sub((2.3, 2.72, -0.05))
        sh, el, wr, out = arms[s]
        fa = seg(el, wr, out, s)
        for i in range(2):
            a = fr.p((0.3 + i * 0.36, 2.95, -1.85))
            K.horn('enr_t1', [a + C.v((0, -0.4, 0)), a, a + C.v((0, 0.5, 0))], [0.12, 0.12, 0.11], mat='accent', sides=8, bone=cb)
            K.horn('enr_t1', [a + C.v((0, -0.5, 0)), a + C.v((0, -0.3, 0))], [0.16, 0.16], mat='dark', sides=8, bone=cb)
            K.horn('enr_t1', [a + C.v((0, 0.45, 0)), a + C.v((0, 0.6, 0))], [0.15, 0.15], mat='dark', sides=8, bone=cb)
        poly('enr_t2', shl, (0.6, -0.05, 0.9), ngon(0.26, 6, 30), 0.08, mat='dark', bone=cb, bev=0.02)
        glow('enr_t2', shl, (0.6, -0.05, 0.98), (0.19, 0.19, 0.06), bone=cb)
        for j in range(3):
            glow('enr_t3', fa, (0.0, -0.55 - 0.35 * j, 0.6), (0.3, 0.05, 0.05), bone='forearm' + sd, seg=8)
        pf = fr.sub((0.06, 1.62, 0.98), (-7, 17, 0))
        for j in range(2):   # glowing vent gaps between the louvres
            blk('enr_t3', pf, (0.98, 0.26 + 0.12 * j, 0.215), (0.34, 0.022, 0.012), mat='accent', bone=cb, bev=0.0)
        glow('enr_t3', pf, (0.62, 1.15, 0.23), (0.07, 0.07, 0.03), bone=cb, seg=8)
    K.crystal('enr_t2', C.p((0, 2.85, -1.75)), C.v((0, 1, -0.4)), 0.8, r=0.2, cluster=False, bone=cb)

    # ---- Speed (additive): extra back thrusters -> leg verniers -> swept wing-boosters
    for s, sd in ((-1, 'L'), (1, 'R')):
        fr = Fr(C.O, C.R @ np.diag([s, 1, 1]))
        hip, kn, an, yaw = legs[s]
        sn = seg(kn, an, (s, 0, 0), s)
        thruster('spd_t1', fr, (1.0, 2.6, -1.6), (0.3, -0.4, -1), 0.45, 0.24, cb)
        thruster('spd_t1', fr, (1.0, 1.4, -1.6), (0.3, -0.6, -1), 0.42, 0.22, cb)
        thruster('spd_t2', sn, (0.0, -1.0, -0.95), (0, -0.4, -1), 0.36, 0.2, 'shin' + sd)
        thruster('spd_t2', sn, (0.0, -1.6, -0.95), (0, -0.4, -1), 0.36, 0.18, 'shin' + sd)
        # wing-boosters: two stacked swept blades rising in a V behind the shoulders
        wf = Fr(fr.p((0.75, 3.0, -1.75)), fr.r(E(-14, 10, -36)) * 1.35)
        poly('spd_t3', wf, (0.0, 0.9, 0.0), [(-0.3, -0.9), (0.3, -0.9), (0.42, 0.9), (0.12, 1.45), (-0.22, 0.95)], 0.08,
             mat='body', bone=cb, bev=0.035)
        poly('spd_t3', wf, (0.0, 0.95, 0.09), [(-0.18, -0.7), (0.18, -0.7), (0.26, 0.75), (0.05, 1.15), (-0.12, 0.75)], 0.03,
             mat='belly', bone=cb, bev=0.0)
        wf2 = Fr(fr.p((0.95, 2.5, -1.85)), fr.r(E(-14, 14, -64)) * 1.3)
        poly('spd_t3', wf2, (0.0, 0.85, 0.0), [(-0.26, -0.8), (0.26, -0.8), (0.34, 0.8), (0.1, 1.25), (-0.2, 0.85)], 0.07,
             mat='body', bone=cb, bev=0.03)
        glow('spd_t3', wf, (0.0, 1.0, 0.11), (0.05, 0.75, 0.03), bone=cb, seg=6)
        glow('spd_t3', wf2, (0.0, 0.9, 0.09), (0.04, 0.65, 0.03), bone=cb, seg=6)
        thruster('spd_t3', wf, (0.0, 0.3, -0.05), (0, -0.2, -1), 0.4, 0.2, cb)
        thruster('spd_t3', wf2, (0.0, 0.3, -0.05), (0, -0.2, -1), 0.36, 0.17, cb)

    # ---- Focus (additive): sensors -> antennas -> shoulder sensor pods + targeting monocle
    for s in (-1, 1):
        fr = Fr(HEADF.O, HEADF.R @ np.diag([s, 1, 1]))
        K.ball('fcs_t1', fr.p((0.47, 0.12, 0.2)), (0.04, 0.06, 0.06), mat='eye', bone='head')
        a = fr.p((0.36, 0.3, -0.35))
        K.horn('fcs_t2', [a, a + fr.v((0.06, 0.25, -0.35)), a + fr.v((0.1, 0.45, -0.75))], [0.045, 0.032, 0.015], mat='horn', sides=4, bone='head')
        K.ball('fcs_t2', a + fr.v((0.03, 0.12, -0.18)), (0.07, 0.07, 0.07), mat='accent', bone='head')
        cf = Fr(C.O, C.R @ np.diag([s, 1, 1]))
        pod = cf.sub((2.0, 3.85, 0.45))
        blk('fcs_t3', pod, (0, 0, 0), (0.18, 0.14, 0.26), mat='dark', bone=cb, bev=0.03, chamf=0.05)
        glow('fcs_t3', pod, (0, 0, 0.26), (0.09, 0.09, 0.04), mat='eye', bone=cb, seg=8)
    K.ball('fcs_t3', HEADF.p((0.24, 0.07, 0.47)), (0.08, 0.08, 0.03), mat='eye', bone='head')
