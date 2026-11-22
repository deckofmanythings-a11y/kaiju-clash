"""Superhero: a caped champion hovering in a power pose (Wild-only, human scale).

Modelled ~9.4 units tall (kaiju-friendly voxel sizes) and exported at SCALE = 0.1.
Also hosts the shared humanoid sculpting helpers used by heroteam.py / magicalgirl.py.
"""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep

DETAIL = 'cloth'
UV_SCALE = 0.9
SCALE = 0.1
SCALE_REF = True
T3_SCALE = 1.12
CAM = dict(az=30, el=6)


# ====================================================================== shared helpers
def A(*v):
    return np.array(v[0] if len(v) == 1 else v, float)


def nrm(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def frame(d, up=(0, 1, 0)):
    """Rotation matrix whose columns are local x, y (= d), z (~ up)."""
    x, y, z = geom.basis(d, up)
    return np.stack([x, y, z], 1)


def rot(pitch=0.0, yaw=0.0, roll=0.0):
    """Head-style rotation: pitch about X (+ = look up), yaw about Y (+ = turn to +X), roll about Z."""
    p, y, r = np.radians([pitch, yaw, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])   # +pitch tips +Z up
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def seg_d(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def capsule(P, a, b, r):
    return seg_d(P, a, b) - r


def beyond(P, q, n):
    """Negative on the side of the plane (through q) that n points to."""
    return -((P - np.asarray(q, float)) @ nrm(n))


def band(P, q, n, w):
    return np.abs((P - np.asarray(q, float)) @ nrm(n)) - w


def pts3(x, y, z):
    return np.stack([x, y, z], 1)


def torus(c, axis, R, r, nu=28, nv=6, wob=0.0, seed=0):
    """Closed ring (game coords). axis = ring normal."""
    x, y, z = geom.basis(axis)
    c = np.asarray(c, float)
    V = []
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28)
    for i in range(nu):
        a = 2 * np.pi * i / nu
        rr = R * (1 + wob * np.sin(3 * a + ph))
        ctr = c + rr * (np.cos(a) * x + np.sin(a) * z)
        rad = nrm(ctr - c)
        for j in range(nv):
            b = 2 * np.pi * j / nv
            V.append(ctr + r * (np.cos(b) * rad + np.sin(b) * y))
    T = []
    for i in range(nu):
        for j in range(nv):
            a0, a1 = i * nv + j, i * nv + (j + 1) % nv
            b0, b1 = ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv
            T += [(a0, b0, b1), (a0, b1, a1)]
    V, T = np.array(V), np.array(T)
    # outward winding
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    cen = V[T].mean(1)
    ring = cen - c
    ring_c = c + R * np.stack([nrm(v - (v @ y) * y) for v in ring]) if False else None
    out = []
    for k in range(len(T)):
        rc = ring[k] - (ring[k] @ y) * y
        rc = c + R * nrm(rc)
        out.append(fn[k] @ (cen[k] - rc) < 0)
    out = np.array(out)
    T[out] = T[out][:, ::-1]
    return V, T


def _puff_wind(V, T, y, top):
    """Orient a puffed double-faced shape: faces using the top apex point along +y, the rest along -y."""
    T = np.asarray(T).copy()
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    want = np.where((T == top).any(1), 1.0, -1.0)
    flip = (fn @ y) * want < 0
    T[flip] = T[flip][:, ::-1]
    return T


def star_pts(c, n, t, r_out, r_in, k=5, depth=0.1):
    """Faceted 5-point star (double-faced puffed solid), facing n, pointing along t."""
    x, y, z = geom.basis(n, t)
    c = np.asarray(c, float)
    ring = []
    for i in range(2 * k):
        a = np.pi * i / k
        rr = r_out if i % 2 == 0 else r_in
        ring.append(c + rr * (np.sin(a) * x + np.cos(a) * z))
    V = np.vstack([ring, [c + y * depth, c - y * depth]])
    m = 2 * k
    T = []
    for i in range(m):
        T += [(i, (i + 1) % m, m), ((i + 1) % m, i, m + 1)]
    return V, _puff_wind(V, np.array(T), y, m)


def heart_pts(c, n, t, s, depth=0.12, seg=28):
    """Puffy heart (double-faced), facing n, point toward -t."""
    x, y, z = geom.basis(n, t)
    c = np.asarray(c, float)
    ring = []
    for i in range(seg):
        a = 2 * np.pi * i / seg
        hx = 16 * np.sin(a) ** 3
        hz = 13 * np.cos(a) - 5 * np.cos(2 * a) - 2 * np.cos(3 * a) - np.cos(4 * a)
        ring.append(c + s / 17 * (hx * x + (hz + 2) * z))
    V = np.vstack([ring, [c + y * depth, c - y * depth]])
    T = []
    for i in range(seg):
        T += [(i, (i + 1) % seg, seg), ((i + 1) % seg, i, seg + 1)]
    return V, _puff_wind(V, np.array(T), y, seg)


def oball(K, part, c, r, R=None, mat='eye', bone=None, seg=12, rings=8, col=None):
    """Ellipsoid piece with an arbitrary rotation matrix (columns = local axes)."""
    V, T, _ = geom.sphere((0, 0, 0), r, seg, rings)
    if R is not None:
        V = V @ np.asarray(R, float).T
    K.piece(part, V + np.asarray(c, float), T, mat, bone=bone, col=col)


def tilt(R, axis, deg):
    """Rotate frame R about one of its own local axes (0=x, 1=y, 2=z)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    M = np.eye(3)
    M[i, i], M[i, j], M[j, i], M[j, j] = c, -s, s, c
    return R @ M


class Fig:
    """A sculpted stylised human. J = joint dict (world coords), p = build params.
    Joints: pelvis, waist, chest, neck, head  +  shL/R elL/R wrL/R  hipL/R knL/R anL/R toeL/R.
    p: s (overall scale), chest, waist, hip, arm, leg, neck, headR (head scale), musc, fem,
       hands=dict(L='fist'|'open'|'flat'|'none', R=...), up=dict(L=back-of-hand up vector, R=...),
       pitch/yaw/roll (head), bust, jaw (0..1 squareness), handS, foot, spread (open-hand finger fan).

    The body (torso, limbs, feet) is sculpted into the shared SDF by sculpt(B); the head and hands are
    fine-voxel rigid pieces added by pieces(K) AFTER K.skin (they need far more resolution than the body)."""

    def __init__(self, J, p):
        self.J = {k: np.asarray(v, float) for k, v in J.items()}
        self.p = dict(s=1.0, chest=1.0, waist=1.0, hip=1.0, arm=1.0, leg=1.0, neck=1.0, headR=1.0, musc=1.0, fem=False,
                      hands=dict(L='fist', R='fist'), up=dict(L=None, R=None), pitch=0.0, yaw=0.0, roll=0.0, bust=0.0, jaw=1.0,
                      spread=1.0)
        self.p.update(p)
        self.q = self.p['s']
        self.Rh = rot(self.p['pitch'], self.p['yaw'], self.p['roll'])
        self.hs = self.q * self.p['headR']
        self.head_sdf = None
        self._knuckles = {}

    # head-local -> world
    def H(self, v):
        return self.J['head'] + self.Rh @ (np.asarray(v, float) * self.hs)

    def hand_dir(self, sd):
        J = self.J
        return nrm(J['wr' + sd] - J['el' + sd])

    def hand_center(self, sd):
        return self.J['wr' + sd] + self.hand_dir(sd) * 0.27 * self.q

    def hand_tip(self, sd):
        return self.J['wr' + sd] + self.hand_dir(sd) * 0.6 * self.q

    def hand_frame(self, sd):
        d = self.hand_dir(sd)
        upv = self.p['up'][sd]
        if upv is None:
            upv = A(0, 0, 1) if abs(d[1]) > 0.7 else A(0, 1, 0)
        return frame(d, upv)            # y = along hand, z = back of hand, palm faces -z

    def axes(self):
        J = self.J
        up = nrm(J['chest'] - J['pelvis'])
        side = nrm(np.cross(up, A(0, 0, 1)))
        side = side if side[0] > 0 else -side
        fwd = nrm(np.cross(side, up))
        return up, side, fwd

    # ------------------------------------------------------------ body
    def sculpt(self, B):
        J, p, q = self.J, self.p, self.q
        fem, mu = p['fem'], p['musc']
        cw, ww, hw = p['chest'], p['waist'], p['hip']
        up, side, fwd = self.axes()
        Rt = frame(up, fwd)                      # x = side, y = up, z = fwd
        ch, wa, pe, nk = J['chest'], J['waist'], J['pelvis'], J['neck']
        if fem:
            B.tube([pe, wa, ch, nk - up * 0.25 * q],
                   np.array([0.86 * hw, 0.7 * ww, 1.0 * cw, 0.78 * cw]) * q, k=0.0, s=(1.22, 1, 0.76))
            B.ell(pe - up * 0.05 * q, A(0.98 * hw, 0.62, 0.68) * q, rot=Rt, k=0.35)
        else:
            m = 0.85 + 0.15 * mu
            # rib cage (deep, wide at the armpits) + shoulder girdle -> the top of the V
            B.ell(ch + up * 0.12 * q, A(0.98 * cw, 1.0, 0.64) * q, rot=Rt, k=0.0)
            B.ell(ch + up * 0.56 * q - fwd * 0.04 * q, A(1.12 * cw, 0.42, 0.56) * q, rot=Rt, k=0.3)
            # solid athletic waist + pelvis (no wasp waist): abdomen is a thick column
            B.tube([pe - up * 0.1 * q, wa, ch - up * 0.2 * q], np.array([0.8 * hw, 0.74 * ww, 0.82 * cw]) * q, k=0.35, s=(1.0, 1, 0.74))
            B.ell(pe - up * 0.08 * q, A(0.84 * hw, 0.56, 0.58) * q, rot=Rt, k=0.3)
        for s in (-1, 1):
            sx = side * s
            sd = 'R' if s > 0 else 'L'
            if not fem:
                m = 0.85 + 0.15 * mu
                # pectorals: broad, flat slabs that tilt up toward the armpit
                Rp = tilt(Rt, 2, s * 14)
                B.ell(ch + sx * 0.44 * q * cw + up * 0.4 * q + fwd * 0.42 * q, A(0.52 * cw, 0.34, 0.22) * q * m, rot=Rp, k=0.16)
                # lats: wedge from the armpit down into the waist
                top = ch + sx * 0.9 * q * cw + up * 0.3 * q - fwd * 0.12 * q
                bot = wa + sx * 0.62 * q * ww + up * 0.1 * q - fwd * 0.08 * q
                B.ell((top + bot) / 2, A(0.3 * m, np.linalg.norm(top - bot) / 2 / q + 0.12, 0.42) * q, rot=frame(top - bot, fwd), k=0.3)
                # obliques: keep the waist line solid down into the hip
                B.ell(wa + sx * 0.6 * q * ww - up * 0.1 * q, A(0.24, 0.55, 0.38) * q, rot=Rt, k=0.3)
                # trapezius: skull base -> shoulder slope
                B.tube([nk + up * 0.45 * q - fwd * 0.18 * q + sx * 0.1 * q, nk + sx * 0.55 * q + up * 0.12 * q - fwd * 0.08 * q,
                        J['sh' + sd] + up * 0.16 * q - sx * 0.12 * q], np.array([0.17, 0.25, 0.2]) * q * m, k=0.25)
                # sternocleidomastoid
                B.tube([self.H((s * 0.28, -0.32, -0.08)), nk + fwd * 0.32 * q + sx * 0.07 * q], np.array([0.075, 0.065]) * q * p['neck'], k=0.1)
            else:
                if p['bust'] > 0:
                    B.ell(ch + sx * 0.36 * q * cw + up * 0.05 * q + fwd * 0.45 * q, A(0.3, 0.28, 0.26) * q * p['bust'], k=0.18)
                B.tube([nk - up * 0.05 * q, J['sh' + sd] + up * 0.12 * q], np.array([0.34, 0.3]) * q * 0.55 * (0.8 + 0.2 * mu), k=0.3)
            # glutes
            B.ell(pe + sx * 0.4 * q * hw - up * 0.2 * q - fwd * 0.38 * q, A(0.44, 0.48, 0.4) * q * hw, rot=Rt, k=0.3)
            # deltoid: caps the shoulder and follows the upper arm
            u = nrm(J['el' + sd] - J['sh' + sd])
            dr = A(0.43, 0.5, 0.43) * q * p['arm'] * (0.72 if fem else (0.85 + 0.15 * mu))
            B.ell(J['sh' + sd] + u * 0.14 * q + sx * 0.04 * q, dr, rot=frame(u, fwd), k=0.2)
        # neck
        B.tube([nk - up * 0.2 * q - fwd * 0.04 * q, self.H((0, -0.5, -0.12))], np.array([0.31, 0.25]) * q * p['neck'] * (1 if not fem else 0.95), k=0.18)
        self.arms(B)
        self.legs(B)

    def arms(self, B):
        J, p, q = self.J, self.p, self.q
        ar = p['arm'] * q
        fem, mu = p['fem'], p['musc']
        up, side, fwd = self.axes()
        for sd in 'LR':
            sh, el, wr = J['sh' + sd], J['el' + sd], J['wr' + sd]
            u = nrm(el - sh)
            f = nrm(wr - el)
            # biceps face the way the forearm folds; straight arm -> forward
            bd = f - u * (f @ u)
            bd = nrm(bd) if np.linalg.norm(bd) > 0.25 else nrm(fwd - u * (fwd @ u))
            fm = 0.72 if fem else 1.0
            B.tube([sh, el], np.array([0.34, 0.25]) * ar * fm, k=0.12)
            if not fem:
                m = 0.85 + 0.2 * mu
                B.ell(sh + (el - sh) * 0.55 + bd * 0.09 * ar, A(0.27, 0.48, 0.27) * ar * m, rot=frame(u, bd), k=0.14)   # biceps
                B.ell(sh + (el - sh) * 0.42 - bd * 0.1 * ar, A(0.27, 0.55, 0.25) * ar * m, rot=frame(u, bd), k=0.16)    # triceps
                B.ell(el - bd * 0.06 * ar, A(0.22, 0.22, 0.22) * ar, k=0.1)                                              # elbow
            # forearm: thick below the elbow, tapering to a flat wrist
            we = wr - f * 0.06 * q
            B.tube([el, el + (wr - el) * 0.3, we], np.array([0.25, 0.27 if not fem else 0.22, 0.165]) * ar * (0.75 if fem else 1), k=0.1)

    def legs(self, B):
        J, p, q = self.J, self.p, self.q
        lr = p['leg'] * q
        fem, mu = p['fem'], p['musc']
        for sd in 'LR':
            hp, kn, an, toe = J['hip' + sd], J['kn' + sd], J['an' + sd], J['toe' + sd]
            s = 1 if sd == 'R' else -1
            u = nrm(kn - hp)
            fm = 0.85 if fem else 1
            B.tube([hp, hp + (kn - hp) * 0.4, kn], np.array([0.6, 0.5, 0.33]) * lr * fm, k=0.2)
            fw = nrm(np.cross(np.cross(u, A(0, 0, 1)), u)) if abs(u[2]) < 0.95 else A(0, 1, 0)
            if fw[2] < 0:
                fw = -fw
            if not fem:
                m = 0.85 + 0.2 * mu
                B.ell(hp + (kn - hp) * 0.55 + fw * 0.13 * lr, A(0.36, 0.72, 0.33) * lr * m, rot=frame(u, fw), k=0.2)            # quads
                B.ell(hp + (kn - hp) * 0.82 + fw * 0.08 * lr + A(-s * 0.1, 0, 0) * lr, A(0.22, 0.3, 0.2) * lr, rot=frame(u, fw), k=0.12)  # vastus medialis
                B.ell(hp + (kn - hp) * 0.45 - fw * 0.14 * lr, A(0.34, 0.65, 0.3) * lr * m, rot=frame(u, fw), k=0.2)            # hamstrings
            B.ell(kn, A(0.29, 0.29, 0.29) * lr * fm, k=0.12)
            v = nrm(an - kn)
            B.tube([kn, kn + (an - kn) * 0.35, an], np.array([0.31, 0.3, 0.17]) * lr * fm, k=0.12)
            bk = nrm(np.cross(np.cross(v, A(0, 0, -1)), v)) if abs(v[2]) < 0.95 else A(0, 1, 0)
            if bk[2] > 0:
                bk = -bk
            B.ell(kn + (an - kn) * 0.3 + bk * 0.1 * lr, A(0.3, 0.55, 0.3) * lr * (0.8 if fem else (0.85 + 0.15 * mu)), rot=frame(v, bk), k=0.15)
            self.foot(B, sd)

    def foot_frame(self, sd):
        J = self.J
        kn, an, toe = J['kn' + sd], J['an' + sd], J['toe' + sd]
        f = nrm(toe - an)
        top = nrm(np.cross(np.cross(f, kn - an), f))      # instep side (toward the knee)
        return f, top, np.linalg.norm(toe - an)

    def foot(self, B, sd):
        """Boot-shaped foot: heel, arch, toe box, flat sole."""
        J, p, q = self.J, self.p, self.q
        an = J['an' + sd]
        f, top, fl = self.foot_frame(sd)
        Rf = frame(f, top)
        fs = p['leg'] * q * (0.85 if p['fem'] else 1) * p.get('foot', 1.0)
        B.ell(an, A(0.18, 0.2, 0.19) * fs, rot=Rf, k=0.1)                                                       # ankle
        B.ell(an - f * 0.1 * fs - top * 0.13 * fs, A(0.165, 0.2, 0.17) * fs, rot=Rf, k=0.1)                      # heel
        B.ell(an + f * fl * 0.42 - top * 0.06 * fs, A(0.19 * fs, fl * 0.45, 0.16 * fs), rot=Rf, k=0.12)          # arch / instep
        B.ell(an + f * fl * 0.8 - top * 0.1 * fs, A(0.2 * fs, fl * 0.26, 0.12 * fs), rot=Rf, k=0.1)           # toe box
        B.box(an + f * fl * 0.35 - top * (0.255 + 0.5) * fs, A(0.6 * fs, fl * 0.95, 0.5 * fs), rot=Rf, cut=True, k=0.03, rad=0.0)

    def m_soles(self, th=0.07):
        J, q = self.J, self.q
        def fn(P):
            out = np.full(len(P), 9.0)
            for sd in 'LR':
                an, toe = J['an' + sd], J['toe' + sd]
                f, top, fl = self.foot_frame(sd)
                fs = self.p['leg'] * q * (0.85 if self.p['fem'] else 1) * self.p.get('foot', 1.0)
                r = np.maximum(capsule(P, an - f * 0.3 * fs, toe + f * 0.2, 0.45 * fs), beyond(P, an - top * (0.255 - th) * fs, -top))
                out = np.minimum(out, r)
            return out
        return fn

    # ------------------------------------------------------------ head (fine-voxel piece)
    def build_head(self):
        p = self.p
        H, hs, Rh = self.H, self.hs, self.Rh
        fem = p['fem']
        jw = p['jaw']
        B = S.SDF()
        B.tube([H((0, -0.3, -0.12)), H((0, -0.85, -0.14))], np.array([0.22, 0.22]) * hs, k=0.0)         # neck plug (sinks into the body neck)
        if not fem:
            B.ell(H((0, 0.08, -0.06)), A(0.41, 0.46, 0.5) * hs, rot=Rh, k=0.1)                          # cranium
            B.ell(H((0, -0.15, 0.12)), A(0.31, 0.35, 0.35) * hs, rot=Rh, k=0.14)                        # face mass
            B.ell(H((0, -0.42, 0.14)), A(0.25 + 0.02 * jw, 0.19, 0.24) * hs, rot=Rh, k=0.14)             # lower face / mandible block
            for s in (-1, 1):
                # square jaw: ramus down from the ear, angle, body of the mandible to the chin
                B.tube([H((s * 0.32, -0.08, -0.08)), H((s * (0.27 + 0.03 * jw), -0.42, -0.02)), H((s * 0.2, -0.56, 0.2)), H((s * 0.09, -0.6, 0.33))],
                       np.array([0.075, 0.075 + 0.01 * jw, 0.07, 0.065]) * hs, k=0.12)
                B.ell(H((s * 0.25, -0.09, 0.33)), A(0.08, 0.05, 0.07) * hs, rot=Rh, k=0.08)                   # cheekbone
                B.ell(H((s * 0.15, -0.005, 0.47)), A(0.095, 0.06, 0.07) * hs, rot=Rh, k=0.05, cut=True)    # eye socket
                B.ell(H((s * 0.43, -0.07, -0.02)), A(0.05, 0.125, 0.085) * hs, rot=Rh @ S.rot_xyz((0, s * 20, 0)), k=0.04)   # ear
                B.ell(H((s * 0.46, -0.07, 0.0)), A(0.025, 0.07, 0.045) * hs, rot=Rh @ S.rot_xyz((0, s * 20, 0)), k=0.02, cut=True)
                B.ell(H((s * 0.055, -0.225, 0.525)), A(0.042, 0.035, 0.04) * hs, rot=Rh, k=0.03)          # nostril wing
            B.ell(H((0, -0.585, 0.35)), A(0.11 + 0.025 * jw, 0.08, 0.08) * hs, rot=Rh, k=0.07)               # chin
            B.ell(H((0, -0.6, 0.43)), A(0.012, 0.035, 0.03) * hs, rot=Rh, k=0.02, cut=True)               # chin cleft
            B.tube([H((-0.32, 0.1, 0.31)), H((-0.16, 0.13, 0.43)), H((0, 0.11, 0.455)), H((0.16, 0.13, 0.43)), H((0.32, 0.1, 0.31))],
                   np.array([0.035, 0.042, 0.035, 0.042, 0.035]) * hs, k=0.07)                               # brow ridge
            B.tube([H((0, 0.07, 0.475)), H((0, -0.08, 0.54)), H((0, -0.19, 0.575))], np.array([0.034, 0.042, 0.05]) * hs, k=0.035)   # nose bridge
            B.ell(H((0, -0.205, 0.565)), A(0.052, 0.045, 0.048) * hs, rot=Rh, k=0.03)                      # nose tip
            B.ell(H((0, -0.31, 0.45)), A(0.15, 0.08, 0.07) * hs, rot=Rh, k=0.07)                          # muzzle / upper lip
            B.ell(H((0, -0.395, 0.475)), A(0.085, 0.03, 0.04) * hs, rot=Rh, k=0.03)                       # lower lip
            B.ell(H((0, -0.355, 0.53)), A(0.1, 0.012, 0.04) * hs, rot=Rh, k=0.012, cut=True)              # mouth line
        else:
            # soft anime face: round cheeks, small chin, tiny nose
            B.ell(H((0, 0.05, -0.04)), A(0.43, 0.47, 0.5) * hs, rot=Rh, k=0.1)
            B.ell(H((0, -0.16, 0.1)), A(0.37, 0.38, 0.38) * hs, rot=Rh, k=0.18)
            B.ell(H((0, -0.42, 0.27)), A(0.16, 0.13, 0.13) * hs, rot=Rh, k=0.14)
            for s in (-1, 1):
                B.ell(H((s * 0.18, -0.03, 0.42)), A(0.11, 0.09, 0.05) * hs, rot=Rh, cut=True, k=0.04)
                B.ell(H((s * 0.44, -0.08, -0.02)), A(0.04, 0.1, 0.07) * hs, rot=Rh @ S.rot_xyz((0, s * 20, 0)), k=0.04)
            B.tube([H((0, -0.1, 0.47)), H((0, -0.2, 0.495))], np.array([0.022, 0.035]) * hs, k=0.04)
        self.head_sdf = B
        return B

    def snap_head(self, K, P):
        if self.head_sdf is None:
            self.build_head()
        return K.snap(P, sdf=self.head_sdf)

    # ------------------------------------------------------------ hands (fine-voxel pieces)
    def build_hand(self, sd):
        J, p, q = self.J, self.p, self.q
        kind = p['hands'][sd]
        fem = p['fem']
        hq = q * (0.82 if fem else 1.0) * p.get('handS', 1.0)
        fr = 0.85 if fem else 1.0                        # finger slimness
        R = self.hand_frame(sd)
        ts = (1 if sd == 'L' else -1) * getattr(self, 'THUMB_FLIP', 1)   # thumb side in local x (palm faces -z); subclasses can mirror it
        wr = J['wr' + sd]
        L = lambda v: wr + R @ (np.asarray(v, float) * hq)
        B = S.SDF()
        ar = p['arm'] * q * (0.75 if fem else 1)
        B.ell(L((0, -0.02, 0)), A(0.15 * ar / hq, 0.15, 0.11 * ar / hq) * hq, rot=R, k=0.0)               # wrist
        knk = []
        fingers = ((ts * 0.135, 0.395, 0.062, 0.42), (ts * 0.045, 0.41, 0.064, 0.46), (-ts * 0.045, 0.4, 0.06, 0.43), (-ts * 0.13, 0.365, 0.052, 0.34))
        if kind == 'fist':
            B.box(L((0, 0.21, -0.005)), A(0.185, 0.17, 0.088) * hq, rot=R, rad=0.06 * hq, k=0.06)          # palm
            B.ell(L((ts * 0.13, 0.15, -0.08)), A(0.1, 0.16, 0.1) * hq, rot=R, k=0.05)                        # thenar pad
            for x, ky, r, _ in fingers:
                r = r * fr
                kp, pip = L((x, ky, 0.02)), L((x * 1.02, ky + 0.06, -0.165))
                dip, tip = L((x * 1.02, ky - 0.055, -0.195)), L((x, ky - 0.085, -0.115))
                B.tube([kp, pip, dip, tip], np.array([r, r * 0.97, r * 0.9, r * 0.82]) * hq, k=0.012, per=3)
                B.ell(L((x, ky + 0.01, 0.06)), A(0.058, 0.055, 0.05) * hq * fr, rot=R, k=0.015)                     # knuckle
                knk.append(L((x, ky + 0.02, 0.08)))
            B.tube([L((ts * 0.15, 0.04, -0.02)), L((ts * 0.215, 0.19, -0.1)), L((ts * 0.17, 0.32, -0.245)), L((ts * 0.02, 0.37, -0.285))],
                   np.array([0.085, 0.075, 0.068, 0.06]) * hq * fr, k=0.03, per=4)                            # wrapped thumb
        elif kind in ('open', 'flat'):
            spr = (0.0 if kind == 'flat' else 0.55) * p['spread']
            curl = 0.05 if kind == 'flat' else 0.6
            B.box(L((0, 0.21, -0.0)), A(0.185, 0.17, 0.07) * hq, rot=R, rad=0.055 * hq, k=0.06)
            B.ell(L((ts * 0.11, 0.1, -0.045)), A(0.1, 0.13, 0.075) * hq, rot=R, k=0.05)
            for i, (x, ky, r, ln) in enumerate(fingers):
                r = r * fr * 0.92
                dx = (x / 0.135) * spr * 0.22
                d = nrm(A(dx, 1, 0))
                kp = A(x, ky, 0.0)
                pts = [kp, kp + d * ln * 0.45 + A(0, 0, -0.03 * curl), kp + d * ln * 0.78 + A(0, 0, -0.1 * curl), kp + d * ln + A(0, 0, -0.18 * curl)]
                B.tube([L(v) for v in pts], np.array([r, r * 0.92, r * 0.82, r * 0.72]) * hq, k=0.008 if kind == 'open' else 0.004, per=3)
                knk.append(L((x, ky, 0.06)))
            th = 0.55 if kind == 'open' else 0.2
            B.tube([L((ts * 0.15, 0.05, -0.03)), L((ts * (0.22 + 0.06 * th), 0.17, -0.08)), L((ts * (0.25 + 0.1 * th), 0.29, -0.1)), L((ts * (0.26 + 0.1 * th), 0.39, -0.1))],
                   np.array([0.085, 0.068, 0.058, 0.05]) * hq * fr, k=0.02, per=4)
        else:
            return None
        self._knuckles[sd] = (knk, R @ A(0, 0.35, 1.0))
        return B

    def knuckles(self, sd):
        """World positions of the fist's knuckle row + outward normal (after build_hand / pieces)."""
        if sd not in self._knuckles:
            self.build_hand(sd)
        return self._knuckles[sd]

    def pieces(self, K, skin_mat='skin', glove_mat='belly', part='body', head=True, hands=True, bones=None,
               head_voxel=0.017, head_tris=2400, hand_tris=900):
        """Mesh the fine head + hands as rigid pieces (joined into `part`)."""
        bones = dict(dict(head='head', handL='handL', handR='handR'), **(bones or {}))
        if head:
            hb = self.head_sdf or self.build_head()
            K.sdf_piece(part, hb, voxel=head_voxel * self.hs, tris=head_tris, mat=skin_mat, bone=bones['head'], ao=0.35)
        elif self.head_sdf is None:
            self.build_head()
        if hands:
            for sd in 'LR':
                g = self.build_hand(sd)
                if g is None:
                    continue
                hq = self.q * (0.82 if self.p['fem'] else 1.0) * self.p.get('handS', 1.0)
                K.sdf_piece(part, g, voxel=0.0155 * hq, tris=hand_tris, mat=glove_mat, bone=bones['hand' + sd], ao=0.3)

    def face_parts(self, K, part='body', iris_mat='iris', bone='head', big=1.0, seg=8, rings=6):
        """Eyes (white + iris + pupil glint), brows and mouth as small rigid pieces snapped to the head."""
        H, hs, Rh = self.H, self.hs, self.Rh
        fem = self.p['fem']
        snap = lambda P: self.snap_head(K, P)
        out = {}
        for s in (-1, 1):
            if fem:
                c, n = snap(H((s * 0.18, -0.05, 0.5)))
                n = nrm(Rh @ A(s * 0.25, 0, 1))
                ew, eh = 0.1 * big, 0.115 * big
                K.ball(part, c - n * 0.015 * hs, A(ew, eh, 0.035) * hs, mat='sclera', bone=bone, seg=seg, rings=rings)
                K.ball(part, c + n * 0.015 * hs - Rh @ A(0, 0.01, 0) * hs, A(ew * 0.72, eh * 0.82, 0.03) * hs, mat=iris_mat, bone=bone, seg=seg, rings=rings)
                K.ball(part, c + n * 0.03 * hs - Rh @ A(0, 0.015, 0) * hs, A(ew * 0.38, eh * 0.45, 0.025) * hs, mat='dark', bone=bone, seg=seg, rings=rings)
                K.ball(part, c + n * 0.045 * hs + Rh @ A(s * 0.02, 0.04, 0) * hs, A(0.025, 0.028, 0.012) * hs, mat='glint', bone=bone, seg=seg, rings=rings)
                K.horn(part, [H((s * 0.08, 0.075, 0.47)), H((s * 0.18, 0.105, 0.475)), H((s * 0.3, 0.065, 0.42)), H((s * 0.34, 0.03, 0.39))],
                       np.array([0.018, 0.026, 0.02, 0.0]) * hs, mat='dark', sides=4, bone=bone)
                K.horn(part, [H((s * 0.09, 0.2, 0.47)), H((s * 0.2, 0.225, 0.46)), H((s * 0.3, 0.2, 0.41))], np.array([0.012, 0.015, 0.0]) * hs, mat='brow', sides=4, bone=bone)
                out['eye' + ('R' if s > 0 else 'L')] = c
            else:
                c = H((s * 0.15, -0.005, 0.445))
                Re = Rh @ S.rot_xyz((0, s * 12, 0))
                n = Re @ A(0, 0, 1)
                eb = big
                oball(K, part, c - n * 0.006 * hs, A(0.066, 0.036, 0.03) * hs * eb, Re, mat='sclera', bone=bone, seg=max(seg, 10), rings=rings)
                oball(K, part, c + n * 0.016 * hs, A(0.029, 0.031, 0.016) * hs * eb, Re, mat=iris_mat, bone=bone, seg=max(seg, 10), rings=rings)
                oball(K, part, c + n * 0.026 * hs, A(0.013, 0.014, 0.008) * hs * eb, Re, mat='dark', bone=bone, seg=8, rings=5)
                # upper lid / lash line hooding the eye
                lid = [c + Re @ A(-s * 0.072, -0.004, -0.012) * hs * eb, c + Re @ A(0, 0.036, 0.012) * hs * eb, c + Re @ A(s * 0.072, 0.006, -0.012) * hs * eb]
                K.horn(part, lid, np.array([0.006, 0.013, 0.006]) * hs * eb, mat='brow', sides=4, flat=0.5, up=n, bone=bone)
                # brow: thick at the inner end, tapering outward, sitting on the ridge
                bp, bn = snap(np.array([H((s * 0.06, 0.15, 0.6)), H((s * 0.17, 0.175, 0.56)), H((s * 0.3, 0.135, 0.45))]))
                K.horn(part, list(bp + bn * 0.012 * hs), np.array([0.026, 0.024, 0.008]) * hs, mat='brow', sides=5, flat=0.45, up=n, bone=bone)
                out['eye' + ('R' if s > 0 else 'L')] = c
        if fem:
            m, n = snap(H((0, -0.33, 0.5)))
            K.horn(part, [m + Rh @ A(-0.06, 0.012, 0) * hs, m + Rh @ A(0, -0.012, 0.01) * hs, m + Rh @ A(0.06, 0.012, 0) * hs], np.array([0.012, 0.016, 0.012]) * hs, mat='mouth', sides=4, bone=bone)
        else:
            mp, mn = snap(np.array([H((-0.1, -0.35, 0.6)), H((0, -0.358, 0.62)), H((0.1, -0.35, 0.6))]))
            K.horn(part, list(mp + mn * 0.002 * hs), np.array([0.008, 0.011, 0.008]) * hs, mat='mouth', sides=4, flat=0.6, bone=bone)
        return out

    # ------------------------------------------------------------ region masks (fn(P) < 0 inside)
    def m_gloves(self, cuff=0.4, sides='LR'):
        J, q = self.J, self.q
        def fn(P):
            out = np.full(len(P), 9.0)
            for sd in sides:
                d = nrm(J['wr' + sd] - J['el' + sd])
                r = np.maximum(capsule(P, J['el' + sd], self.hand_tip(sd) + d * 0.2 * q, 0.62 * q),
                               beyond(P, J['wr' + sd] - d * cuff * q, d))
                out = np.minimum(out, r)
            return out
        return fn

    def m_boots(self, frac=0.45, sides='LR'):
        J, q = self.J, self.q
        def fn(P):
            out = np.full(len(P), 9.0)
            for sd in sides:
                kn, an = J['kn' + sd], J['an' + sd]
                r = np.maximum(np.minimum(capsule(P, kn, an, 0.75 * q), capsule(P, an, J['toe' + sd], 0.6 * q)),
                               beyond(P, kn + (an - kn) * frac, an - kn))
                out = np.minimum(out, r)
            return out
        return fn

    def m_skin(self, collar=0.12, r=0.46):
        J, q = self.J, self.q
        up = nrm(J['head'] - J['neck'])
        def fn(P):
            return np.maximum(beyond(P, J['neck'] + up * collar * q, up), capsule(P, J['neck'], J['head'], r * q * self.p['neck'] ** 0.5))
        return fn

    def m_belt(self, at=0.15, w=0.11):
        J, q = self.J, self.q
        up = nrm(J['chest'] - J['pelvis'])
        c = J['pelvis'] + (J['waist'] - J['pelvis']) * at + up * 0.0
        def fn(P):
            return np.maximum(band(P, c, up, w * q), capsule(P, J['pelvis'] - up * q, J['chest'], 1.25 * q))
        return fn, c, up

    def leg_planes(self, at=0.33, tilt=0.6):
        J, q = self.J, self.q
        out = []
        for sd in 'LR':
            hp, kn = J['hip' + sd], J['kn' + sd]
            ld = nrm(kn - hp)
            s = -1 if sd == 'L' else 1
            out.append((sd, hp + ld * at * q * 2.0, nrm(ld + A(s * tilt, 0, 0))))
        return out

    def m_trunks(self, at=0.15, cut=0.33, tilt=0.6):
        J, q = self.J, self.q
        up = nrm(J['chest'] - J['pelvis'])
        top = J['pelvis'] + (J['waist'] - J['pelvis']) * at
        planes = self.leg_planes(cut, tilt)
        def fn(P):
            r = np.maximum(beyond(P, top, -up), capsule(P, J['pelvis'] - up * 1.2 * q, J['waist'], 1.35 * q * self.p['hip']))
            for sd, c, n in planes:
                r = np.maximum(r, -beyond(P, c, n))
            return r
        return fn

    # ------------------------------------------------------------ skeleton
    def rig(self, K, pre='', parent=None, full=True):
        J, q = self.J, self.q
        pp = pre
        K.bone(pp + 'pelvis', J['pelvis'], J['waist'], parent, 1.0 * q)
        K.bone(pp + 'spine', J['waist'], J['chest'], pp + 'pelvis', 1.0 * q)
        K.bone(pp + 'chest', J['chest'], J['neck'], pp + 'spine', 1.05 * q)
        hb = self.H((0, -0.45, -0.05))
        K.bone(pp + 'neck', J['neck'], hb, pp + 'chest', 0.38 * q)
        K.bone(pp + 'head', hb, self.H((0, 0.55, 0.0)), pp + 'neck', 0.62 * self.hs)
        if not full:
            return
        for sd in 'LR':
            K.bone(pp + 'upperarm' + sd, J['sh' + sd], J['el' + sd], pp + 'chest', 0.38 * q)
            K.bone(pp + 'forearm' + sd, J['el' + sd], J['wr' + sd], pp + 'upperarm' + sd, 0.3 * q)
            K.bone(pp + 'hand' + sd, J['wr' + sd], self.hand_tip(sd), pp + 'forearm' + sd, 0.26 * q)
            K.bone(pp + 'thigh' + sd, J['hip' + sd], J['kn' + sd], pp + 'pelvis', 0.6 * q)
            K.bone(pp + 'shin' + sd, J['kn' + sd], J['an' + sd], pp + 'thigh' + sd, 0.4 * q)
            K.bone(pp + 'foot' + sd, J['an' + sd], J['toe' + sd], pp + 'shin' + sd, 0.25 * q)

    def mounts(self, K, pre='', back=None, tail=None, tail_bone=None):
        J, q = self.J, self.q
        up = nrm(J['chest'] - J['pelvis'])
        K.mount('head', self.H((0, 0.0, 0.0)), pre + 'head')
        K.mount('jaw', self.H((0, -0.5, 0.3)), pre + 'head')
        K.mount('mouth', self.H((0, -0.33, 0.52)), pre + 'head')
        K.mount('crown', self.H((0, 0.72, 0.0)), pre + 'head')
        cp, _ = K.snap(J['chest'] + A(0, 0.1, 1.5) * q)
        K.mount('chest', cp, pre + 'chest')
        K.mount('back', back if back is not None else J['chest'] + A(0, 0, -0.85) * q, pre + 'spine')
        K.mount('tailTip', tail if tail is not None else J['pelvis'] + A(0, 0, -1.0) * q, tail_bone or pre + 'pelvis')
        for sd in 'LR':
            K.mount('hand' + sd, self.hand_center(sd), pre + 'hand' + sd)
            K.mount('shoulder' + sd, J['sh' + sd] + up * 0.35 * q, pre + 'chest')


def hair_disp(scale=1.0, seed=0, freq=26.0, amp=0.018):
    """Strand grooves for hair SDFs (noise-warped so they flow)."""
    def fn(x, y, z):
        w = fbm(x * 0.9, y * 0.9, z * 0.9, 2, freq=1.2 / scale, seed=seed) * 0.6
        g = np.abs(np.sin((x * 0.8 + z * 0.35 + w) * freq / scale))
        return -amp * scale * (1 - g) ** 2 + 0.006 * scale
    return fn


def cape_grid(K, part, top, tip_drop, back, flare, side_drift, folds=4.5, nu=26, nv=20, mat='cape', bone='chest', seed=0, wrap=0.4, amp=0.22, hem=0.12):
    """Billowing cape surface. `top` = list of attachment points across the shoulders (L->R)."""
    top = np.asarray(top, float)
    tc = S.catmull(top, 6)
    s_len = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tc, axis=0), axis=1))]
    s_len /= s_len[-1]

    def top_at(u):
        return np.array([np.interp(u, s_len, tc[:, i]) for i in range(3)])

    ph = seed * 1.7

    def F(u, v):
        p0 = top_at(u)
        xc = (u - 0.5) * 2
        dx = xc * flare * v ** 1.1 + side_drift * v ** 2
        dy = -tip_drop * v + back * 0.18 * v ** 2
        dz = -back * v ** 1.35 - wrap * (1 - xc * xc) * 0.0
        f = np.sin(xc * np.pi * folds + ph + v * 1.3) * amp * (0.15 + v) * (0.6 + 0.4 * np.sin(v * 3 + u * 5))
        hm = hem * np.sin(xc * np.pi * folds + ph + 1.3 + 1.6) * smoothstep(0.8, 1.0, np.array([v]))[0]
        return p0 + A(dx, dy + hm, dz + f)

    V, T, _ = geom.grid(F, nu, nv)
    # fold shading: valleys darker
    P = np.array([[F(i / nu, j / nv) for j in range(nv + 1)] for i in range(nu + 1)])
    col = []
    for i in range(nu + 1):
        for j in range(nv + 1):
            u, v = i / nu, j / nv
            xc = (u - 0.5) * 2
            fs = np.sin(xc * np.pi * folds + ph + v * 1.3)
            col.append(0.62 + 0.38 * (0.5 + 0.5 * fs) * (0.6 + 0.4 * (1 - v)) + 0.15 * (1 - v))
    K.piece(part, V, T, mat, col=np.clip(col, 0.3, 1), bone=bone)
    return F




# ====================================================================== the superhero
LIFT = 0.45


def joints():
    J = dict(
        pelvis=(0, 4.75, 0.0), waist=(0, 5.62, 0.04), chest=(0, 6.6, 0.12), neck=(0, 7.6, 0.04), head=(0, 8.5, 0.16),
        shR=(1.38, 7.28, 0.02), elR=(1.95, 8.45, 0.8), wrR=(1.85, 9.5, 1.6),
        shL=(-1.38, 7.28, 0.02), elL=(-1.85, 5.82, -0.18), wrL=(-2.0, 4.5, 0.12),
        hipR=(0.52, 4.48, 0.0), knR=(0.62, 2.4, 0.22), anR=(0.6, 0.42, -0.15), toeR=(0.62, -0.13, 0.75),
        hipL=(-0.52, 4.48, 0.05), knL=(-0.72, 3.05, 1.35), anL=(-0.76, 1.5, 0.0), toeL=(-0.76, 0.9, 0.85),
    )
    return {k: np.asarray(v, float) + A(0, LIFT, 0) for k, v in J.items()}


def build(K):
    col = K.colors
    K.mat('skin', color='#d6a07c', rough=0.6, vcol=True, detail=False)
    K.mat('hair', color='#201813', rough=0.42, vcol=True, detail=False)
    K.mat('brow', color='#1a130f', rough=0.6)
    K.mat('sclera', color='#f2efe8', rough=0.35)
    K.mat('iris', color='#2a62c9', rough=0.25, emit='#2a62c9', emit_strength=0.4)
    K.mat('mouth', color='#7a3a34', rough=0.5)
    K.mat('cape', color=col['belly'], rough=0.62, vcol=True, double=True, detail=False)
    K.mat('body', rough=0.42, detail_strength=0.35)
    K.mat('belly', rough=0.5, detail_strength=0.35)
    K.mat('plate', color='#c7ccd8', rough=0.28, metal=0.75, vcol=True)
    K.mat('gaunt', color=col['belly'], rough=0.3, metal=0.55, vcol=True)
    K.mat('gold', color='#e8b630', rough=0.25, metal=0.85, vcol=True)
    K.mat('sole', color='#2b2427', rough=0.7, vcol=True, detail=False)

    J = joints()
    fig = Fig(J, dict(s=1.0, chest=1.02, waist=1.0, hip=1.0, headR=1.0, arm=1.04, leg=1.0, musc=1.0, pitch=8, yaw=8,
                      hands=dict(L='fist', R='fist'), up=dict(L=A(-1, 0, 0.3), R=A(0, 0, 1)), jaw=1.1))
    B = K.sdf()
    fig.sculpt(B)
    up = nrm(J['chest'] - J['pelvis'])
    # boot cuffs: a flared top lip on each boot
    for sd in 'LR':
        kn, an = J['kn' + sd], J['an' + sd]
        v = nrm(an - kn)
        c0 = kn + (an - kn) * 0.4
        B.tube([c0 - v * 0.05, c0 + v * 0.12], [0.39, 0.32], k=0.05)

    # ---- costume relief: belt, glove + boot cuffs, ab definition
    belt_fn, belt_c, belt_up = fig.m_belt(at=0.18, w=0.12)
    gloves = fig.m_gloves(cuff=0.42)
    boots = fig.m_boots(frac=0.4)
    leg_pl = fig.leg_planes(0.33, 0.6)
    fz = belt_c + A(0, 0, 0.62)
    buckle = S.Ell(fz, A(0.2, 0.15, 0.3))

    def relief(x, y, z):
        P = pts3(x, y, z)
        d = np.zeros(len(P), np.float32)
        b = band(P, belt_c, belt_up, 0.12)
        d -= 0.045 * smoothstep(0.02, -0.02, b) * smoothstep(0.0, -0.1, capsule(P, J['pelvis'] - up, J['chest'], 1.25))
        for sd in 'LR':
            dd = nrm(J['wr' + sd] - J['el' + sd])
            cq = J['wr' + sd] - dd * 0.42
            d -= 0.05 * smoothstep(0.06, 0.0, np.abs((P - cq) @ dd)) * smoothstep(0.0, -0.1, capsule(P, J['el' + sd], J['wr' + sd], 0.5))
        # collar ridge + trunk leg-opening piping
        hu = nrm(J['head'] - J['neck'])
        d -= 0.035 * smoothstep(0.05, 0.0, np.abs((P - (J['neck'] + hu * 0.04)) @ hu)) * smoothstep(0.0, -0.1, capsule(P, J['neck'] - hu, J['head'], 0.6))
        for sd, c, n in leg_pl:
            d -= 0.03 * smoothstep(0.045, 0.0, np.abs((P - c) @ n)) * smoothstep(0.0, -0.1, beyond(P, belt_c, -belt_up)) * smoothstep(0.0, -0.1, capsule(P, J['pelvis'], J['kn' + sd], 1.0))
        d -= 0.05 * smoothstep(0.0, -0.05, buckle.dist(S.Points(P)))
        # abs: 3 rows of shallow pads + centre groove on the front of the waist
        lx, ly = x - 0.0, y - (J['waist'][1] + 0.1)
        front = smoothstep(-0.1, 0.35, z - J['waist'][2]) * smoothstep(0.55, 0.32, np.abs(lx)) * smoothstep(0.62, 0.36, np.abs(ly))
        grid = (np.abs(np.sin(ly * np.pi / 0.34)) ** 0.5) * smoothstep(0.0, 0.05, np.abs(lx))
        d -= 0.028 * front * grid
        # sternum groove + under-pec crease
        ch = J['chest']
        cz = smoothstep(-0.1, 0.4, z - ch[2])
        d += 0.02 * cz * smoothstep(0.06, 0.0, np.abs(x)) * smoothstep(-0.3, 0.0, y - ch[1]) * smoothstep(0.75, 0.5, y - ch[1])
        return d
    B.disp(relief, band=0.12)

    skin_fn = fig.m_skin(collar=0.06)
    trunks = fig.m_trunks()
    soles = fig.m_soles()
    belly_fn = lambda P: np.minimum(np.minimum(gloves(P), boots(P)), trunks(P))
    buck_fn = lambda P: buckle.dist(S.Points(P)) + 0.02
    K.skin(B, voxel=0.04, tris=10500, regions=[('belly', belly_fn), ('sole', soles), ('skin', skin_fn), ('gold', belt_fn), ('accent', buck_fn)])

    # ---- fine head + hands, face
    fig.pieces(K, skin_mat='skin', glove_mat='belly', head_tris=2000, hand_tris=750)
    fig.face_parts(K)
    H = fig.H

    # ---- hair: a short cap + combed-back locks with a side part and a forelock curl
    hb = K.sdf()
    hb.ell(H((0, 0.13, -0.08)), A(0.452, 0.47, 0.545), rot=fig.Rh, k=0.1)
    for j in range(6):                                   # right of the part: swept up, back and right
        x0 = -0.1 + j * 0.105
        hb.tube([H((x0, 0.36, 0.36)), H((x0 + 0.07, 0.57, 0.06)), H((x0 * 1.05 + 0.1, 0.5, -0.36)), H((x0 * 1.1 + 0.1, 0.3, -0.55))],
                [0.07, 0.095, 0.085, 0.06], k=0.09)
    for x0 in (-0.2, -0.3):                              # left of the part: swept down and back
        hb.tube([H((x0, 0.38, 0.33)), H((x0 - 0.12, 0.44, 0.06)), H((-0.42, 0.24, -0.28))], [0.07, 0.09, 0.07], k=0.09)
    hb.tube([H((-0.05, 0.42, 0.4)), H((0.08, 0.57, 0.38)), H((0.24, 0.6, 0.18))], [0.08, 0.11, 0.09], k=0.08)   # forelock / quiff
    hb.box(H((0, -0.18, 0.55)), A(0.7, 0.55, 0.4), rot=fig.Rh @ S.rot_xyz((-28, 0, 0)), cut=True, k=0.06, rad=0.1)   # face opening
    hb.box(H((0, -0.68, 0.0)), A(0.7, 0.45, 0.7), rot=fig.Rh, cut=True, k=0.1, rad=0.1)                               # nape
    for s in (-1, 1):
        hb.ell(H((s * 0.47, -0.06, 0.0)), A(0.1, 0.17, 0.13), rot=fig.Rh, cut=True, k=0.05)                          # ears free
    hb.disp(hair_disp(1.0, seed=2, freq=30, amp=0.016), band=0.06)
    K.sdf_piece('body', hb, voxel=0.02, tris=2600, mat='hair', bone='head', ao=0.5)
    K.horn('body', [H((0.02, 0.48, 0.45)), H((0.07, 0.36, 0.49)), H((0.02, 0.28, 0.485)), H((-0.02, 0.32, 0.475))], [0.035, 0.032, 0.024, 0.0], mat='hair', sides=6, bone='head')

    # ---- emblem: gold diamond with a red inner field + star
    cp, cn = K.snap(J['chest'] + A(0, 0.28, 1.2))
    dia = S.Box(cp, A(0.4, 0.4, 0.3), rot=frame(cn, A(0, 1, 0)) @ S.rot_xyz((0, 45, 0)), rad=0.06)
    K.plate('body', cp, None, off=0.0, th=0.04, mat='accent', region=dia, voxel=0.03, tris=500)
    dia2 = S.Box(cp, A(0.29, 0.29, 0.3), rot=frame(cn, A(0, 1, 0)) @ S.rot_xyz((0, 45, 0)), rad=0.04)
    K.plate('body', cp, None, off=0.05, th=0.025, mat='belly', region=dia2, voxel=0.03, tris=400)
    V, T = star_pts(cp + cn * 0.1, cn, A(0, 1, 0), 0.21, 0.085, depth=0.05)
    K.piece('body', V, T, 'accent', smooth=False, bone='chest')

    # ---- cape (clasped at both shoulders, billowing back and to the left)
    shL, shR = J['shL'], J['shR']
    top = [shL + A(0.1, 0.36, 0.1), shL + A(0.5, 0.42, -0.42), A(0, J['neck'][1] + 0.0, -0.6), shR + A(-0.5, 0.42, -0.42), shR + A(-0.1, 0.36, 0.1)]
    cape_grid(K, 'body', top, tip_drop=6.3, back=3.0, flare=1.5, side_drift=-1.1, folds=4.5, seed=1)
    for c0 in (top[0], top[-1]):
        p, n = K.snap(c0)
        K.ball('body', p + n * 0.04, A(0.16, 0.16, 0.16), mat='gold', bone='chest')
    cape_tip = A(-1.1, J['neck'][1] - 6.3 + 0.54, -0.6 - 3.0)

    # ---- skeleton + mounts
    fig.rig(K)
    fig.mounts(K, back=J['chest'] + A(0, 0, -1.0), tail=cape_tip, tail_bone='chest')

    # ---- POW: gauntlets (exclusive sets): flared cuffs, then armoured knuckles, then spiked
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        for sd in 'LR':
            wr, el = J['wr' + sd], J['el' + sd]
            d = nrm(wr - el)
            r0 = 0.22 + 0.025 * t
            r1 = (0.235 if t == 0 else 0.29 * (0.85 + 0.25 * sc))
            mat = 'belly' if t == 0 else 'gaunt'
            K.horn(part, [wr - d * (0.42 + 0.12 * t), wr - d * (0.25 + 0.06 * t), wr - d * 0.06], [r0, r0 * 1.02 + 0.02, r1], mat=mat, sides=14, per=2, bone='forearm' + sd)
            if t >= 1:
                R = fig.hand_frame(sd)
                gs = 0.9 + 0.12 * sc
                kn, kd = fig.knuckles(sd)
                kd = nrm(kd)
                ktop = [k + kd * 0.03 for k in kn]
                K.horn(part, [ktop[0] - (ktop[-1] - ktop[0]) * 0.12] + ktop + [ktop[-1] + (ktop[-1] - ktop[0]) * 0.12],
                       [0.055 * gs, 0.075 * gs, 0.08 * gs, 0.08 * gs, 0.07 * gs, 0.05 * gs], mat='gaunt', sides=8, bone='hand' + sd)
                oball(K, part, J['wr' + sd] + R @ A(0, 0.2, 0.1), A(0.19, 0.17, 0.05) * gs, R, mat='gaunt', bone='hand' + sd, seg=12, rings=6)
                for kb in ktop:
                    if t >= 2:
                        K.spike(part, kb + kd * 0.03, kd, 0.12 + 0.06 * t, r=0.05, mat='gold' if t == 2 else 'accent', bone='hand' + sd)
                    else:
                        K.ball(part, kb + kd * 0.05, 0.05, mat='gold', bone='hand' + sd)
                if t >= 2:
                    o = nrm(np.cross(d, A(0, 0, 1)) if abs(d[2]) < 0.9 else A(1, 0, 0))
                    o = o if (o @ (wr - J['pelvis']) * A(1, 0, 0)).sum() > 0 else -o
                    K.horn(part, [wr - d * 0.55 + o * 0.25, wr - d * 0.35 + o * (0.42 + 0.1 * t), wr - d * 0.1 + o * 0.3], [0.04, 0.05, 0.0], mat='gold', sides=4, flat=0.3, bone='forearm' + sd)

    # ---- ARMOR
    for sd, s in (('L', -1), ('R', 1)):
        sh = J['sh' + sd]
        K.plate('arm_t1', sh + A(s * 0.2, 0.25, 0), (0.62, 0.5, 0.62), bone='upperarm' + sd, lames=3, off=0.05, th=0.07)
        el, wr = J['el' + sd], J['wr' + sd]
        K.plate('arm_t2', el + (wr - el) * 0.45, (0.42, 0.42, 0.42), bone='forearm' + sd, lames=2, off=0.04, th=0.06, lame_axis=wr - el, tris=240)
        hp = J['hip' + sd]
        K.plate('arm_t2', hp + A(s * 0.45, 0.05, 0.1), (0.45, 0.55, 0.6), bone='thigh' + sd, lames=3, off=0.05, th=0.06, tris=240)
        kn, an = J['kn' + sd], J['an' + sd]
        K.plate('arm_t3', kn + (an - kn) * 0.6 + A(0, 0, 0.25), (0.42, 0.6, 0.42), bone='shin' + sd, lames=3, off=0.06, th=0.07, lame_axis=an - kn, tris=260)
        K.plate('arm_t3', kn + A(0, 0, 0.3), (0.3, 0.32, 0.16), bone='shin' + sd, lames=0, off=0.05, th=0.06, tris=160)
    K.plate('arm_t2', J['neck'] + A(0, -0.12, 0.25), (0.75, 0.32, 0.6), bone='chest', lames=2, off=0.05, th=0.06)   # gorget

    # ---- ENERGY: emblem halo, fist rings, aura rings
    V, T = torus(cp + cn * 0.06, cn, 0.58, 0.045, nu=32, nv=5)
    K.piece('enr_t1', V, T, 'accent', bone='chest')
    for sd in 'LR':
        hc = fig.hand_center(sd)
        d = fig.hand_dir(sd)
        for i, (o, rr) in enumerate(((-0.25, 0.42), (0.05, 0.5))):
            V, T = torus(hc + d * o, d, rr, 0.035, nu=24, nv=5)
            K.piece('enr_t2', V, T, 'accent', bone='hand' + sd)
        K.crystal('enr_t2', hc + d * 0.4, d, 0.35, r=0.08, bone='hand' + sd, seed=3)
    for i, (y, R, tl) in enumerate(((0.35, 1.25, 6), (1.3, 1.75, -8), (2.5, 2.15, 5))):
        ax = S.rot_xyz((tl, 0, tl * 0.6)) @ A(0, 1, 0)
        V, T = torus(A(0, y + LIFT, 0.1), ax, R, 0.035, nu=44, nv=5, wob=0.03, seed=i)
        K.piece('enr_t3', V, T, 'accent', bone='pelvis')

    # ---- SPEED: boot wings, longer under-cape, forearm fins
    for sd, s in (('L', -1), ('R', 1)):
        an, kn = J['an' + sd], J['kn' + sd]
        p, n = K.snap(an + (kn - an) * 0.25 + A(s * 0.5, 0, 0))
        K.fin('spd_t1', p, n * A(1, 0.2, 0.2) + A(0, 0.5, -0.6), nrm(kn - an) * -0.3 + A(0, 0, -1), 0.75, 0.75, mat='gold', bone='shin' + sd, sweep_back=0.8)
        K.fin('spd_t1', p + A(0, 0.2, 0), n * A(1, 0.2, 0.2) + A(0, 0.9, -0.5), A(0, 0, -1), 0.5, 0.5, mat='gold', bone='shin' + sd, sweep_back=0.8)
        el, wr = J['el' + sd], J['wr' + sd]
        p, n = K.snap(el + (wr - el) * 0.5 + nrm(np.cross(wr - el, A(0, 0, 1))) * 0.4 * s)
        K.fin('spd_t3', p, n, el - wr, 0.55, 0.9, mat='gold', bone='forearm' + sd, sweep_back=0.9)
        p, n = K.snap(kn + (an - kn) * 0.4 + A(0, 0, -0.5))
        K.fin('spd_t3', p, n * A(0.3, 0, 1) + A(0, 0.3, -0.3), A(0, 1, 0), 0.6, 0.9, mat='gold', bone='shin' + sd, sweep_back=0.9)
    top2 = [np.asarray(v) + A(0, -0.08, -0.06) for v in top]
    cape_grid(K, 'spd_t2', top2, tip_drop=7.6, back=4.2, flare=2.0, side_drift=-1.8, folds=4.5, seed=1, amp=0.26, hem=0.12)

    # ---- FOCUS: glowing eyes, circlet + gem, winged visor + crest
    for s in (-1, 1):
        c, n = H((s * 0.15, -0.005, 0.445)), fig.Rh @ A(s * 0.2, 0, 1)
        oball(K, 'fcs_t1', c + n * 0.03 * fig.hs, A(0.075, 0.045, 0.03) * fig.hs, fig.Rh @ S.rot_xyz((0, s * 12, 0)), mat='accent', bone='head')
    ring = K.sdf()
    ring.ell(H((0, 0.22, 0.0)), A(0.5, 0.2, 0.57), rot=fig.Rh, k=0)
    ring.ell(H((0, 0.22, 0.0)), A(0.465, 0.3, 0.535), rot=fig.Rh, k=0.02, cut=True)
    ring.box(H((0, 0.44, 0.0)), A(0.7, 0.13, 0.7), rot=fig.Rh, k=0.02, cut=True, rad=0.0)
    ring.box(H((0, 0.0, 0.0)), A(0.7, 0.13, 0.7), rot=fig.Rh, k=0.02, cut=True, rad=0.0)
    K.sdf_piece('fcs_t2', ring, voxel=0.022, tris=400, mat='gold', bone='head')
    K.crystal('fcs_t2', H((0, 0.25, 0.58)), fig.Rh @ A(0, 0.6, 1), 0.2, r=0.08, bone='head', cluster=False)
    for s in (-1, 1):
        K.fin('fcs_t3', H((s * 0.46, 0.08, -0.05)), fig.Rh @ A(s * 0.5, 1, -0.3), fig.Rh @ A(0, 0.3, -1), 0.6, 0.55, mat='gold', bone='head', sweep_back=1.0)
    K.leaf('fcs_t3', H((0, 0.62, 0.0)), fig.Rh @ A(0, 1, 0.05), fig.Rh @ A(0, 0, 1), 0.35, 0.8, th=0.04, lobes=3, mat='gold', bone='head')
