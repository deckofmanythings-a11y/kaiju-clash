"""Colossal Woman: a 1950s B-movie giantess in a tattered dress, striding through town (biped plan).

Kaiju scale: modelled ~10.8 units tall at SCALE = 1.0, faces +Z. Uses the humanoid helpers from superhero.py
(Fig) with a realistic female head, bare feet and clawing hands overridden here.
"""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep
from .superhero import A, Fig, band, beyond, capsule, frame, hair_disp, nrm, oball, pts3, torus
from .magicalgirl import ribbon

DETAIL = 'cloth'
UV_SCALE = 0.8
CAM = dict(az=32, el=8)
INFO = dict(name='Colossal Woman', colors=dict(body='#c22a46', belly='#f0c35c', accent='#6dff8e'))

Q = 1.2   # figure scale (Fig joints are authored for ~9.4 units at s=1)
HAND_S = 1.5   # big Kaiju-scale hands


def tri(x):
    return np.abs((x - np.floor(x)) - 0.5) * 2.0


class TornSkirt:
    """Hollow flared skirt hanging from c along -Y, with radial folds and a ragged, torn hem.
    theta = atan2(x, z): 0 = front (+Z), +pi/2 = right (+X)."""

    def __init__(self, c, h, r0, r1, zs=0.8, pw=0.55, n=13, amp=0.07, scal=0.42, th=0.13, hollow=0.42, tears=()):
        self.c = np.asarray(c, float)
        self.h, self.r0, self.r1, self.zs, self.pw, self.n, self.amp, self.scal, self.th = h, r0, r1, zs, pw, n, amp, scal, th
        self.hollow = hollow
        self.tears = list(tears)      # (theta, half-width, height)
        e = r1 + amp + 0.25
        self.lo = self.c - np.array([e, h + 0.3, e * zs])
        self.hi = self.c + np.array([e, 0.3, e * zs])

    def mirrored(self):
        return self

    def _parts(self, reg):
        F = np.float32
        lx, ly, lz = np.broadcast_arrays(reg.x - F(self.c[0]), reg.y - F(self.c[1]), reg.z - F(self.c[2]))
        lz = lz / F(self.zs)
        h = -ly
        t = np.clip(h / F(self.h), 0, 1)
        th = np.arctan2(lx, lz)
        R = (F(self.r0) + F(self.r1 - self.r0) * t ** F(self.pw)
             + F(self.amp) * t * np.sin(F(self.n) * th + F(0.6)) + F(self.amp * 0.45) * t * np.sin(F(23) * th + F(1.7)))
        rad = np.sqrt(lx * lx + lz * lz)
        slope = (self.r1 - self.r0) / self.h
        d_rad = (rad - R) / F(np.sqrt(1 + slope * slope))
        jag = 0.45 * tri(th * 7 / np.pi + 0.13) + 0.33 * tri(th * 13 / np.pi + 0.41) + 0.22 * tri(th * 29 / np.pi + 0.77)
        lift = F(self.scal) * jag * (0.7 + 0.3 * np.sin(th * 3 + 1.1)) ** 2
        for tc, hw, ht in self.tears:
            dth = np.angle(np.exp(1j * (th - tc))).astype(F)
            lift = np.maximum(lift, F(ht) * np.clip(1 - np.abs(dth) / F(hw), 0, 1) ** 1.3)
        d_bot = h - (F(self.h) - lift)
        return d_rad, d_bot, ly, rad, R, h

    def outer(self, P):
        d_rad, d_bot, d_top, _, _, _ = self._parts(S.Points(P))
        return np.maximum(np.maximum(d_rad, d_bot), d_top)

    def dist(self, reg, pad=0):
        F = np.float32
        d_rad, d_bot, d_top, rad, R, h = self._parts(reg)
        outer = np.maximum(np.maximum(d_rad, d_bot), d_top)
        inner = np.maximum(rad - (R - F(self.th)), F(self.h * self.hollow) - h)
        return np.maximum(outer, -inner).astype(F)


class Giantess(Fig):
    """Fig with a realistic adult female head, bare feet and curled 'claw' hands."""
    # 'L' joints sit at -x, which is her anatomical RIGHT (she faces +z), so the default thumb side built a left hand on
    # the right arm and vice versa (user report, 1.12.0). Mirror the thumb for every hand kind (claw and fist).
    THUMB_FLIP = -1

    def foot(self, B, sd):
        J, q = self.J, self.q
        an, toe = J['an' + sd], J['toe' + sd]
        f = toe - an
        f[1] = 0
        fl = np.linalg.norm(f)
        f = nrm(f)
        side = nrm(np.cross(A(0, 1, 0), f)) * (1 if sd == 'R' else -1)    # points away from the body midline
        Rf = frame(f, A(0, 1, 0))
        fs = q * 1.12
        y0 = A(0, 1, 0)
        B.ell(an, A(0.16, 0.18, 0.17) * fs, rot=Rf, k=0.08)                                                # ankle
        B.ell(an - f * 0.1 * fs - y0 * 0.2 * fs, A(0.14, 0.17, 0.18) * fs, rot=Rf, k=0.12)                 # heel
        B.ell(an + f * fl * 0.38 - y0 * 0.22 * fs, A(0.18 * fs, fl * 0.42, 0.13 * fs), rot=Rf, k=0.14)     # arch / instep
        B.ell(an + f * fl * 0.74 - y0 * 0.29 * fs + side * 0.02, A(0.21 * fs, fl * 0.17, 0.085 * fs), rot=Rf, k=0.1)   # ball
        # toes: big toe on the inside, four smaller ones fanning outward
        for i, (o, r, ln) in enumerate(((-0.13, 0.062, 0.2), (-0.03, 0.047, 0.17), (0.055, 0.043, 0.15), (0.13, 0.04, 0.13), (0.195, 0.036, 0.1))):
            b0 = an + f * fl * 0.8 + side * o * fs - y0 * (0.31 * fs - r * 0.4 * fs)
            b1 = b0 + f * ln * fs - y0 * 0.02
            B.tube([b0, b1], np.array([r, r * 0.92]) * fs, k=0.03)
            B.ell(b1, A(r * 1.05, r * 1.0, r * 0.95) * fs, k=0.02)
        B.box(A(an[0], -0.5, an[2]) + f * fl * 0.3, A(1.4, 0.5, 1.8), cut=True, k=0.02, rad=0.0)          # flat sole at y = 0

    def arms(self, B):
        """Strong, athletic arms: capped deltoids, biceps/triceps, thick forearms."""
        J, q = self.J, self.q
        up, side, fwd = self.axes()
        for sd in 'LR':
            sh, el, wr = J['sh' + sd], J['el' + sd], J['wr' + sd]
            sx = side * (1 if sd == 'R' else -1)
            u = nrm(el - sh)
            f = nrm(wr - el)
            bd = f - u * (f @ u)
            bd = nrm(bd) if np.linalg.norm(bd) > 0.25 else nrm(fwd - u * (fwd @ u))
            B.tube([sh, el], np.array([0.35, 0.25]) * q, k=0.12)
            B.ell(sh + u * 0.16 * q + sx * 0.05 * q + up * 0.03 * q, A(0.37, 0.46, 0.37) * q, rot=frame(u, fwd), k=0.18)    # deltoid cap
            B.ell(sh + (el - sh) * 0.56 + bd * 0.08 * q, A(0.2, 0.4, 0.2) * q, rot=frame(u, bd), k=0.14)                # biceps
            B.ell(sh + (el - sh) * 0.42 - bd * 0.08 * q, A(0.21, 0.46, 0.2) * q, rot=frame(u, bd), k=0.16)              # triceps
            B.ell(el - bd * 0.05 * q, A(0.18, 0.18, 0.18) * q, k=0.1)                                                   # elbow
            we = wr - f * 0.06 * q
            B.tube([el, el + (wr - el) * 0.3, we], np.array([0.25, 0.27, 0.16]) * q, k=0.1)
            B.ell(el + (wr - el) * 0.28 + bd * 0.06 * q, A(0.16, 0.34, 0.15) * q, rot=frame(f, bd), k=0.12)             # forearm flexors

    def legs(self, B):
        """Powerful legs: full thighs (quads / hamstrings / outer sweep), defined knees and calves."""
        J, q = self.J, self.q
        for sd in 'LR':
            hp, kn, an = J['hip' + sd], J['kn' + sd], J['an' + sd]
            s = 1 if sd == 'R' else -1
            u = nrm(kn - hp)
            fw = nrm(np.cross(np.cross(u, A(0, 0, 1)), u)) if abs(u[2]) < 0.95 else A(0, 1, 0)
            if fw[2] < 0:
                fw = -fw
            sx = nrm(np.cross(u, fw))
            if sx[0] * s < 0:
                sx = -sx                                                                    # outward
            B.tube([hp, hp + (kn - hp) * 0.4, kn], np.array([0.56, 0.47, 0.29]) * q, k=0.2)
            B.ell(hp + (kn - hp) * 0.52 + fw * 0.12 * q, A(0.3, 0.64, 0.27) * q, rot=frame(u, fw), k=0.2)               # quads
            B.ell(hp + (kn - hp) * 0.8 + fw * 0.07 * q - sx * 0.09 * q, A(0.19, 0.26, 0.18) * q, rot=frame(u, fw), k=0.12)   # vastus medialis
            B.ell(hp + (kn - hp) * 0.5 + sx * 0.13 * q, A(0.2, 0.55, 0.22) * q, rot=frame(u, fw), k=0.18)               # outer sweep
            B.ell(hp + (kn - hp) * 0.45 - fw * 0.13 * q, A(0.29, 0.58, 0.26) * q, rot=frame(u, fw), k=0.2)              # hamstrings
            B.ell(kn, A(0.25, 0.25, 0.25) * q, k=0.12)
            v = nrm(an - kn)
            B.tube([kn, kn + (an - kn) * 0.35, an], np.array([0.33, 0.315, 0.175]) * q, k=0.12)
            bk = nrm(np.cross(np.cross(v, A(0, 0, -1)), v)) if abs(v[2]) < 0.95 else A(0, 1, 0)
            if bk[2] > 0:
                bk = -bk
            B.ell(kn + (an - kn) * 0.27 + bk * 0.12 * q + sx * 0.05 * q, A(0.27, 0.48, 0.29) * q, rot=frame(v, bk), k=0.14)   # calf (outer head)
            B.ell(kn + (an - kn) * 0.32 + bk * 0.12 * q - sx * 0.07 * q, A(0.26, 0.46, 0.28) * q, rot=frame(v, bk), k=0.14)   # calf (inner head)
            B.tube([kn + (an - kn) * 0.5 + bk * 0.12 * q, an + bk * 0.14 * q - v * 0.15 * q], np.array([0.1, 0.075]) * q, k=0.12)   # achilles
            self.foot(B, sd)

    def build_head(self, mouth=True):
        """Strong-featured adult face with a B-movie scowl: furrowed brow, sharp cheekbones, defined jaw and an
        open, shouting mouth (mouth=False gives the closed surface used for snapping face details)."""
        H, hs, Rh = self.H, self.hs, self.Rh
        B = S.SDF()
        B.tube([H((0, -0.3, -0.12)), H((0, -0.85, -0.14))], np.array([0.2, 0.2]) * hs, k=0.0)
        B.ell(H((0, 0.08, -0.06)), A(0.41, 0.46, 0.5) * hs, rot=Rh, k=0.1)                      # cranium
        B.ell(H((0, -0.14, 0.13)), A(0.3, 0.34, 0.34) * hs, rot=Rh, k=0.14)                     # face mass
        B.ell(H((0, -0.4, 0.17)), A(0.2, 0.17, 0.21) * hs, rot=Rh, k=0.14)                      # lower face
        for s in (-1, 1):
            B.tube([H((s * 0.32, -0.06, -0.08)), H((s * 0.28, -0.38, 0.0)), H((s * 0.18, -0.52, 0.22)), H((s * 0.07, -0.58, 0.33))],
                   np.array([0.075, 0.082, 0.068, 0.06]) * hs, k=0.1)                            # defined jawline
            B.ell(H((s * 0.25, -0.07, 0.33)), A(0.1, 0.06, 0.085) * hs, rot=Rh, k=0.07)            # high cheekbone
            B.ell(H((s * 0.235, -0.27, 0.36)), A(0.06, 0.07, 0.03) * hs, rot=Rh, k=0.08, cut=True)  # hollow under the cheekbone
            B.ell(H((s * 0.15, -0.005, 0.47)), A(0.09, 0.05, 0.055) * hs, rot=Rh, k=0.06, cut=True)  # eye socket
            B.ell(H((s * 0.42, -0.08, -0.02)), A(0.045, 0.11, 0.075) * hs, rot=Rh @ S.rot_xyz((0, s * 20, 0)), k=0.04)   # ear
            B.ell(H((s * 0.052, -0.2, 0.505)), A(0.04, 0.032, 0.036) * hs, rot=Rh, k=0.03)      # nostril wing
            B.tube([H((s * 0.07, -0.21, 0.5)), H((s * 0.13, -0.3, 0.485)), H((s * 0.15, -0.4, 0.45))],
                   np.array([0.018, 0.022, 0.016]) * hs, k=0.05)                                 # snarl fold (nasolabial)
            B.ell(H((s * 0.032, 0.1, 0.5)), A(0.007, 0.05, 0.03) * hs, rot=Rh @ S.rot_xyz((0, 0, s * -12)), k=0.02, cut=True)   # frown crease
        B.ell(H((0, -0.55, 0.36)), A(0.11, 0.08, 0.08) * hs, rot=Rh, k=0.07)                    # chin
        B.tube([H((-0.31, 0.1, 0.33)), H((-0.17, 0.1, 0.45)), H((-0.05, 0.055, 0.49)), H((0.05, 0.055, 0.49)), H((0.17, 0.1, 0.45)), H((0.31, 0.1, 0.33))],
               np.array([0.03, 0.04, 0.04, 0.04, 0.04, 0.03]) * hs, k=0.07)                      # heavy brow ridge, knotted down in the middle
        B.ell(H((0, 0.04, 0.5)), A(0.045, 0.035, 0.03) * hs, rot=Rh, k=0.04)                     # procerus knot
        B.tube([H((0, 0.0, 0.48)), H((0, -0.09, 0.52)), H((0, -0.17, 0.555))], np.array([0.026, 0.03, 0.038]) * hs, k=0.05)   # nose bridge
        B.ell(H((0, -0.185, 0.55)), A(0.045, 0.038, 0.04) * hs, rot=Rh, k=0.03)                  # nose tip
        B.ell(H((0, -0.3, 0.455)), A(0.14, 0.07, 0.065) * hs, rot=Rh, k=0.06)                   # muzzle
        if mouth:
            B.ell(H((0, -0.375, 0.52)), A(0.118, 0.072, 0.12) * hs, rot=Rh, k=0.02, cut=True)      # open, shouting mouth
        self.head_sdf = B
        return B

    def build_hand(self, sd):
        if self.p['hands'][sd] != 'claw':
            return super().build_hand(sd)
        J, p, q = self.J, self.p, self.q
        hq = q * 0.82 * p.get('handS', 1.0)
        fr = 0.85
        R = self.hand_frame(sd)
        ts = (1 if sd == 'L' else -1) * self.THUMB_FLIP
        wr = J['wr' + sd]
        L = lambda v: wr + R @ (np.asarray(v, float) * hq)
        B = S.SDF()
        ar = p['arm'] * q * 0.75
        B.ell(L((0, -0.02, 0)), A(0.15 * ar / hq, 0.15, 0.11 * ar / hq) * hq, rot=R, k=0.0)
        B.box(L((0, 0.21, 0.0)), A(0.185, 0.17, 0.07) * hq, rot=R, rad=0.055 * hq, k=0.06)
        B.ell(L((ts * 0.11, 0.1, -0.045)), A(0.1, 0.13, 0.075) * hq, rot=R, k=0.05)
        fingers = ((ts * 0.135, 0.395, 0.062, 0.44), (ts * 0.045, 0.41, 0.064, 0.48), (-ts * 0.045, 0.4, 0.06, 0.45), (-ts * 0.13, 0.365, 0.052, 0.36))
        curl = p.get('curl', 1.0)
        spr = 0.65 * p['spread']
        z = A(0, 0, 1)
        tips, knk = [], []
        for x, ky, r, ln in fingers:
            r = r * fr * 0.95
            d = nrm(A((x / 0.135) * spr * 0.22, 1, 0))
            pts, v, a = [A(x, ky, 0.0)], d, 0.0
            for a, l in zip(np.array([0.4, 0.95, 1.45]) * curl, ln * np.array([0.46, 0.3, 0.24])):
                v = np.cos(a) * d - np.sin(a) * z
                pts.append(pts[-1] + v * l)
            B.tube([L(t) for t in pts], np.array([r, r * 0.92, r * 0.84, r * 0.74]) * hq, k=0.01, per=3)
            back = np.sin(a) * d + np.cos(a) * z
            tips.append((L(pts[-1] - v * 0.03), R @ v, r * hq, R @ back))
            knk.append(L((x, ky, 0.06)))
        th = [A(ts * 0.15, 0.05, -0.03), A(ts * 0.25, 0.16, -0.1), A(ts * 0.3, 0.27, -0.17), A(ts * 0.28, 0.36, -0.24)]
        B.tube([L(t) for t in th], np.array([0.085, 0.07, 0.06, 0.052]) * hq * fr, k=0.02, per=4)
        v = nrm(th[3] - th[2])
        tips.append((L(th[3] - v * 0.02), R @ v, 0.052 * hq * fr, R @ nrm(A(ts * 0.6, 0, 0.8))))
        self._knuckles[sd] = (knk, R @ A(0, 0.35, 1.0))
        self.tips = getattr(self, 'tips', {})
        self.tips[sd] = tips
        return B


def joints():
    def along(a, d, ln):
        return np.asarray(a, float) + nrm(d) * ln
    J = dict(pelvis=(0, 6.05, 0.0), waist=(0, 6.85, 0.06), chest=(0, 7.7, 0.26), neck=(0, 8.58, 0.32), head=(0, 9.68, 0.56),
             shR=(1.1, 8.3, 0.16), shL=(-1.1, 8.28, 0.2),
             hipR=(0.56, 5.85, 0.0), knR=(1.16, 3.12, 0.3), anR=(1.6, 0.44, -0.15),
             hipL=(-0.56, 5.85, 0.05), knL=(-1.06, 3.16, 0.7), anL=(-1.44, 0.44, 0.4))
    J = {k: np.asarray(v, float) for k, v in J.items()}
    J['elR'] = along(J['shR'], (0.75, 0.55, 0.1), 1.7)
    J['wrR'] = along(J['elR'], (-0.1, 0.8, 0.6), 1.45)
    J['elL'] = along(J['shL'], (-0.55, -0.5, 0.7), 1.7)
    J['wrL'] = along(J['elL'], (-0.3, -0.5, 0.8), 1.45)
    J['toeR'] = J['anR'] + A(0.28, -0.2, 1.26)
    J['toeL'] = J['anL'] + A(-0.24, -0.2, 1.28)
    return J


def boxmesh(c, R, half):
    """Box (game coords): R columns = local axes, half = half extents."""
    c = np.asarray(c, float)
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * np.asarray(half, float)
    V = c + V @ np.asarray(R, float).T
    T = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return V, geom.fix_winding(V, T)


def axes_for(d, hint=(0, 1, 0)):
    """Rotation whose columns are (x, y=d, z): a local frame along d."""
    return frame(d, hint)


def helix(c0, c1, r, turns, th, n=60, r1=None):
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    ax = nrm(c1 - c0)
    x, _, z = geom.basis(ax)
    k = np.linspace(0, 1, n)
    a = k * turns * 2 * np.pi
    rr = (r + ((r if r1 is None else r1) - r) * k)[:, None]
    pts = c0[None] + (c1 - c0)[None] * k[:, None] + rr * (np.cos(a)[:, None] * x[None] + np.sin(a)[:, None] * z[None])
    return pts


def surf_strap(K, ctrl, hw, off=0.035, per=4):
    """Strap lying on the body surface through approximate control points -> (centreline, half-width vectors)."""
    C = S.catmull(np.asarray(ctrl, float), per)
    p, n = K.snap(C)
    C = p + n * off
    tan = np.gradient(C, axis=0)
    W = np.array([nrm(np.cross(nn, t)) * hw for nn, t in zip(n, tan)])
    return C, W, n


def clean_pieces(K):
    """Drop degenerate / duplicate triangles from rigid pieces (the decimated fist SDF produced a few duplicate
    faces, which made Blender flag pow_t1-3 as invalid meshes on export)."""
    orig = K.piece

    def piece(part, V, T, mat='horn', smooth=True, col=None, bone=None):
        Tn = np.asarray(T)
        if len(Tn):
            rep = (Tn[:, 0] == Tn[:, 1]) | (Tn[:, 1] == Tn[:, 2]) | (Tn[:, 0] == Tn[:, 2])
            Tn = Tn[~rep]
            _, first = np.unique(np.sort(Tn, axis=1), axis=0, return_index=True)
            T = Tn[np.sort(first)]
        return orig(part, V, T, mat=mat, smooth=smooth, col=col, bone=bone)
    K.piece = piece


def build(K):
    clean_pieces(K)
    col = K.colors
    K.mat('skin', color='#efc3a4', rough=0.55, vcol=True, detail=False)
    K.mat('body', rough=0.7, detail_strength=0.45)
    K.mat('belly', rough=0.45, vcol=True, detail=False)
    K.mat('brow', color='#3a2210', rough=0.6)
    K.mat('lips', color='#a3122c', rough=0.32, vcol=False)
    K.mat('mouth', color='#3a0a10', rough=0.5, vcol=False)
    K.mat('teeth', color='#f4efe4', rough=0.3, vcol=False)
    K.mat('nail', color='#b0122e', rough=0.22, vcol=True)
    K.mat('lash', color='#15100c', rough=0.5)
    K.mat('iron', color='#3c4146', rough=0.45, metal=0.8, vcol=True)
    K.mat('chrome', color='#d9dde3', rough=0.18, metal=0.95, vcol=True)
    K.mat('rust', color='#7b4a2c', rough=0.7, metal=0.3, vcol=True)
    K.mat('plate', color='#e9e1cf', rough=0.5, metal=0.1, vcol=True)
    K.mat('gold', color='#e8b630', rough=0.25, metal=0.85, vcol=True)
    K.mat('glass', color='#ffe7a0', emit='#ffd56a', emit_strength=2.5, rough=0.2)
    K.mat('beacon', color='#ff3020', emit='#ff3020', emit_strength=3.0, rough=0.2)
    K.mat('concrete', color='#9c968c', rough=0.9, vcol=True)
    K.mat('scarf', color=col['body'], rough=0.6, vcol=True, double=True, detail=False)
    K.mat('ribbon', color=col['belly'], rough=0.5, vcol=True, double=True, detail=False)

    J = joints()
    fig = Giantess(J, dict(s=Q, fem=True, chest=0.8, waist=0.82, hip=1.0, arm=1.3, leg=1.1, neck=0.92, headR=1.06, bust=0.42,
                           pitch=-7, yaw=10, roll=-3, hands=dict(L='claw', R='claw'), up=dict(L=A(0, 1, 0.2), R=A(0, 0, -1)),
                           handS=HAND_S, spread=1.0, curl=1.0))
    H, hs = fig.H, fig.hs
    B = K.sdf()
    fig.sculpt(B)
    up, side, fwd = fig.axes()
    # athletic upper body: trapezius slope, upper back, lats and a solid core
    nk, ch, wa = J['neck'], J['chest'], J['waist']
    for s in (-1, 1):
        sx = side * s
        sd = 'R' if s > 0 else 'L'
        B.tube([nk + up * 0.05 - fwd * 0.12 + sx * 0.12, nk + sx * 0.55 - up * 0.08 - fwd * 0.1, J['sh' + sd] + up * 0.1 - sx * 0.15],
               np.array([0.2, 0.24, 0.22]) * Q, k=0.25)                                                          # trapezius
        top = ch + sx * 0.8 + up * 0.25 - fwd * 0.2
        bot = wa + sx * 0.55 + up * 0.1 - fwd * 0.1
        B.ell((top + bot) / 2, A(0.24, np.linalg.norm(top - bot) / 2 / Q + 0.1, 0.36) * Q, rot=frame(top - bot, fwd), k=0.3)   # lats
        B.ell(ch + sx * 0.42 + up * 0.25 - fwd * 0.45, A(0.36, 0.42, 0.22) * Q, rot=frame(up, fwd), k=0.25)                  # upper back / scapula
    B.ell(wa + fwd * 0.22, A(0.5, 0.62, 0.36) * Q, rot=frame(up, fwd), k=0.3)                                   # solid core

    # ---- dress: torn skirt + bodice
    sk_top = J['waist'] + A(0, -0.12, -0.02)
    SK_H, SK_R1, SK_ZS = 3.4, 1.76, 0.82
    skirt = TornSkirt(sk_top, SK_H, 0.92, SK_R1, zs=SK_ZS, pw=0.42, n=13, amp=0.085, scal=0.66, th=0.14,
                      tears=[(-0.75, 0.36, 1.45), (2.2, 0.3, 1.05), (0.5, 0.16, 0.8), (1.45, 0.2, 0.95), (-2.0, 0.24, 0.85), (3.05, 0.16, 0.65), (-1.35, 0.12, 0.6)])
    B.add(skirt, k=0.04)
    B.ell(J['chest'] + up * 0.05 + A(0, 0, 0.05), A(0.92, 0.56, 0.64) * Q * 0.8, k=0.25)      # bodice fill so the dress reads solid

    neck_q = J['chest'] + up * 0.36 + A(0, 0, 0.0)
    neck_n = nrm(up + A(0, 0, 0.45))
    sash_c = J['waist'] + up * 0.02
    BELT_W = 0.21
    hip_lo = J['pelvis'][1] - 1.0

    def relief(x, y, z):
        P = pts3(x, y, z)
        d = np.zeros(len(P), np.float32)
        near_t = smoothstep(0.0, -0.1, capsule(P, J['pelvis'], J['chest'], 1.45))
        bd = band(P, sash_c, up, BELT_W)
        d -= 0.055 * smoothstep(0.02, -0.02, bd) * near_t                                           # wide belt, raised
        d -= 0.015 * smoothstep(0.03, 0.0, np.abs(bd + 0.03)) * near_t                              # stitched rolled edges
        # bodice: princess seams + neckline hem
        for s in (-1, 1):
            d += 0.012 * smoothstep(0.03, 0.0, np.abs(x - s * 0.4 - 0.05 * s * (y - 7.4))) * near_t * smoothstep(-0.2, 0.2, z)
        d -= 0.03 * smoothstep(0.05, 0.0, np.abs(beyond(P, neck_q, -neck_n))) * smoothstep(0.0, -0.1, capsule(P, J['chest'], J['neck'], 1.0))
        # skirt: soft vertical drape grooves (noise-warped)
        w = fbm(x * 0.6, y * 0.6, z * 0.6, 2, freq=1.0, seed=11)
        th = np.arctan2(x - sk_top[0], (z - sk_top[2]) / SK_ZS)
        dr = np.sin(th * 19 + w * 2.5) * smoothstep(sk_top[1], sk_top[1] - 1.2, y)
        d -= 0.018 * dr * smoothstep(0.0, -0.08, skirt.outer(P) - 0.05)
        return d
    B.disp(relief, band=0.12)

    def dress(P):
        torso = np.maximum(capsule(P, J['pelvis'] - up * 0.6, J['neck'], 1.45 * Q), beyond(P, neck_q, -neck_n))
        torso = np.maximum(torso, np.abs(P[:, 0]) - (1.02 + 0.08 * (P[:, 1] < 7.6) + 0.6 * (P[:, 1] < 7.1)))
        torso = np.maximum(torso, hip_lo - P[:, 1])
        sk = skirt.outer(P) - 0.045
        return np.minimum(torso, sk)

    def belt(P):
        return np.maximum(band(P, sash_c, up, BELT_W), capsule(P, J['pelvis'], J['chest'], 1.45))

    def tint(P, N):
        so = skirt.outer(P)
        hemy = sk_top[1] - SK_H
        dirt = smoothstep(hemy + 1.1, hemy, P[:, 1]) * smoothstep(0.12, 0.0, np.abs(so))
        return 1.0 - 0.38 * dirt
    ob = K.skin(B, voxel=0.045, tris=14000, base_mat='skin', regions=[('body', dress), ('belly', belt)], tint=tint)
    try:
        ob.data.validate(clean_customdata=False)      # the decimated skin had a few degenerate faces
    except Exception:
        pass

    # ---- fine head + left (base) hand, face
    fig.head_closed = fig.build_head(mouth=False)
    fig.build_head(mouth=True)
    fig.pieces(K, skin_mat='skin', glove_mat='skin', head_tris=2600, hands=False)
    gL = fig.build_hand('L')
    hq = Q * 0.82 * HAND_S
    K.sdf_piece('body', gL, voxel=0.0155 * hq, tris=900, mat='skin', bone='handL', ao=0.3)
    tipsL = list(fig.tips['L'])
    face(K, fig)

    # ---- hair: 1950s bouffant, big front wave, shoulder-length flipped curls, windswept locks
    hair(K, fig)

    # ---- belt buckle + shoulder straps (one intact, one torn and hanging)
    belt_buckle(K, sash_c, up, BELT_W)
    straps(K, J, up)

    # ---- skeleton + mounts
    fig.rig(K)
    K.bone('jaw', H((0, -0.3, 0.05)), H((0, -0.55, 0.35)), 'head', 0.12 * hs)
    K.mount('head', H((0, 0, 0)), 'head')
    K.mount('jaw', H((0, -0.5, 0.3)), 'jaw')
    K.mount('mouth', H((0, -0.34, 0.56)), 'head')
    K.mount('crown', H((0, 1.0, -0.05)), 'head')
    cp, cn = K.snap(J['chest'] + A(0, 0.1, 1.5))
    K.mount('chest', cp, 'chest')
    bk, bkn = K.snap(J['chest'] + A(0, -0.1, -1.5))
    K.mount('back', bk, 'spine')
    K.mount('tailTip', A(0, sk_top[1] - 3.3, sk_top[2] - (SK_R1 - 0.05) * SK_ZS), 'pelvis')
    for sd in 'LR':
        K.mount('hand' + sd, fig.hand_center(sd), 'hand' + sd)
        K.mount('shoulder' + sd, J['sh' + sd] + up * 0.42, 'chest')

    # ---- POW: nails -> talons on both hands; right hand = open claw (t0) or a fist gripping a weapon (t1-3)
    wrR, elR = J['wrR'], J['elR']
    dR = nrm(wrR - elR)
    g_dir = nrm(A(0.7, 0.4, -1.0))               # weapon points back over the shoulder, cocked to swing
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        if t == 0:
            fig.p['up']['R'] = A(0, 0, -1)
            fig.p['hands']['R'] = 'claw'
            g = fig.build_hand('R')
            K.sdf_piece(part, g, voxel=0.0155 * hq, tris=850, mat='skin', bone='handR', ao=0.3)
            tipsR = list(fig.tips['R'])
        else:
            fig.p['up']['R'] = np.cross(g_dir, dR)
            fig.p['hands']['R'] = 'fist'
            g = fig.build_hand('R')
            K.sdf_piece(part, g, voxel=0.0155 * hq, tris=850, mat='skin', bone='handR', ao=0.3)
            tipsR = []
        ln = [0.07, 0.15, 0.25, 0.36][t]
        for (tp, tv, tr, tb) in tipsL + tipsR:
            bone = 'handL' if np.linalg.norm(tp - J['wrL']) < np.linalg.norm(tp - J['wrR']) else 'handR'
            base = tp + tb * tr * 0.45 - tv * tr * 0.6
            if t == 0:
                V, T, _ = geom.sphere((0, 0, 0), 1, 8, 5)
                Rn = frame(tv, tb)
                V = base + tv * tr * 0.35 + (V * A(tr * 0.8, tr * 1.25, tr * 0.32)) @ Rn.T
                K.piece(part, V, T, 'nail', bone=bone)
            else:
                K.claw(part, base, tv + tb * 0.15, ln, tr * (0.7 + 0.08 * t), curl=-tb, mat='nail', bone=bone, flat=0.55)
        if t >= 1:
            weapon(K, part, t, fig, g_dir, hq)

    # ---- ARMOR: billboard pauldron, water-tower gauntlet, cable-wrapped shins
    armor(K, fig, J)
    # ---- ENERGY: glowing pearl necklace, bracelets, radiation shards
    energy(K, fig, J)
    # ---- SPEED: torn scarf, dress streamers, long sash tails
    speed(K, fig, J, sash_c, skirt, sk_top)
    # ---- FOCUS: earrings, tiara, third-eye gem
    focus(K, fig)


def face(K, fig):
    """B-movie scowl: slanted glowing eyes under heavy lids, thick V brows, open shouting mouth with teeth."""
    H, hs, Rh = fig.H, fig.hs, fig.Rh
    snap = lambda P: K.snap(P, sdf=fig.head_closed)
    for s in (-1, 1):
        c = H((s * 0.15, -0.012, 0.45))
        Re = Rh @ S.rot_xyz((0, s * 12, s * 15))          # outer corners up -> glare
        n = Re @ A(0, 0, 1)
        oball(K, 'body', c - n * 0.002 * hs, A(0.078, 0.034, 0.032) * hs, Re, mat='accent', bone='head', seg=12, rings=6)       # glowing eye
        oball(K, 'body', c + n * 0.022 * hs, A(0.032, 0.02, 0.012) * hs, Re, mat='glass', bone='head', seg=8, rings=5)         # hot core
        # heavy upper lid / lash line
        lid = [c + Re @ A(-s * 0.088, 0.004, -0.012) * hs, c + Re @ A(-s * 0.03, 0.036, 0.016) * hs, c + Re @ A(s * 0.04, 0.038, 0.01) * hs, c + Re @ A(s * 0.095, 0.02, -0.018) * hs]
        K.horn('body', lid, np.array([0.013, 0.019, 0.017, 0.006]) * hs, mat='lash', sides=4, flat=0.55, up=n, bone='head')
        # squinting lower lid (skin roll under the eye)
        low = [c + Re @ A(-s * 0.07, -0.026, 0.0) * hs, c + Re @ A(0, -0.04, 0.014) * hs, c + Re @ A(s * 0.08, -0.024, -0.006) * hs]
        K.horn('body', low, np.array([0.008, 0.016, 0.008]) * hs, mat='skin', sides=5, flat=0.6, up=n, bone='head')
        # thick angry brows: inner ends driven down to the nose, sharp arch outward
        bp, bn = snap(np.array([H((s * 0.035, 0.035, 0.6)), H((s * 0.12, 0.085, 0.58)), H((s * 0.23, 0.15, 0.52)), H((s * 0.33, 0.15, 0.42))]))
        K.horn('body', list(bp + bn * 0.014 * hs), np.array([0.034, 0.036, 0.026, 0.008]) * hs, mat='brow', sides=5, flat=0.5, up=bn[1], bone='head')
    # open shouting mouth: dark interior, two rows of teeth, full red lips with a snarl lift
    oball(K, 'body', H((0, -0.378, 0.45)), A(0.115, 0.08, 0.07) * hs, Rh, mat='mouth', bone='head', seg=12, rings=6)
    K.horn('body', [H((-0.08, -0.325, 0.515)), H((0, -0.318, 0.54)), H((0.08, -0.325, 0.515))], np.array([0.018, 0.021, 0.018]) * hs,
           mat='teeth', sides=6, flat=0.55, bone='head')
    K.horn('body', [H((-0.07, -0.425, 0.505)), H((0, -0.432, 0.525)), H((0.07, -0.425, 0.505))], np.array([0.015, 0.018, 0.015]) * hs,
           mat='teeth', sides=6, flat=0.55, bone='head')
    mp, mn = snap(np.array([H((-0.13, -0.39, 0.5)), H((-0.07, -0.305, 0.57)), H((0.0, -0.297, 0.6)), H((0.075, -0.298, 0.57)), H((0.13, -0.385, 0.5))]))
    K.horn('body', list(mp + mn * 0.006 * hs), np.array([0.012, 0.024, 0.022, 0.025, 0.012]) * hs, mat='lips', sides=6, flat=0.65, bone='head')
    lp, ln = snap(np.array([H((-0.125, -0.395, 0.5)), H((-0.065, -0.455, 0.55)), H((0, -0.465, 0.57)), H((0.065, -0.455, 0.55)), H((0.125, -0.395, 0.5))]))
    K.horn('body', list(lp + ln * 0.006 * hs), np.array([0.012, 0.027, 0.03, 0.027, 0.012]) * hs, mat='lips', sides=6, flat=0.65, bone='head')


def belt_buckle(K, c, up, w):
    """Square chrome buckle on the front of the wide belt, with a prong and a few punched holes."""
    bp, bn = K.snap(c + A(0, 0, 1.6))
    n = nrm(bn - up * (bn @ up))
    sx = nrm(np.cross(up, n))
    R = np.stack([sx, up, n], 1)
    o = bp + n * 0.07
    hw, hh, t = 0.27, w + 0.05, 0.05
    for cc, half in ((o + up * (hh - t), A(hw, t, 0.035)), (o - up * (hh - t), A(hw, t, 0.035)),
                     (o + sx * (hw - t), A(t, hh, 0.035)), (o - sx * (hw - t), A(t, hh, 0.035))):
        V, T = boxmesh(cc, R, half)
        K.piece('body', V, T, 'chrome', smooth=False, bone='pelvis')
    V, T = boxmesh(o - n * 0.02, R, A(hw - t, hh - t, 0.02))                                     # belt passing behind the frame
    K.piece('body', V, T, 'belly', smooth=False, bone='pelvis')
    K.horn('body', [o + sx * 0.02 + n * 0.02, o - sx * 0.2 + n * 0.035], [0.022, 0.018], mat='chrome', sides=5, bone='pelvis')   # prong
    for i in range(3):
        hp, hn = K.snap(c + A(-0.55 - 0.2 * i, 0, 1.6))
        K.ball('body', hp + hn * 0.06, A(0.03, 0.03, 0.03), mat='dark', seg=6, rings=4, bone='pelvis')


def straps(K, J, up):
    """Right shoulder strap intact (front bodice -> over the shoulder -> back bodice); left strap torn: one end
    still stitched to the front of the bodice and flopped down, a ragged stub left at the back."""
    C, W, _ = surf_strap(K, [(0.46, 7.45, 1.2), (0.52, 7.95, 1.0), (0.6, 8.45, 0.65), (0.66, 8.72, 0.15), (0.6, 8.6, -0.35), (0.5, 8.3, -0.7)], 0.1)
    ribbon(K, 'body', C, W, mat='scarf', bone='chest', th=0.025, cup=0.15)
    # torn left strap: attached part on the bodice front, then hanging free over the bust line
    C0, W0, n0 = surf_strap(K, [(-0.46, 7.3, 1.2), (-0.48, 7.48, 1.16), (-0.5, 7.66, 1.12)], 0.1, per=3)
    p1, n1, t1 = C0[-1], n0[-1], nrm(C0[-1] - C0[-2])
    k = np.linspace(0, 1, 10)[1:]
    # peeled over the bodice edge, then hanging down and swinging slightly outward
    out = n1[None] * (0.05 + 0.1 * np.sin(np.clip(k * 3, 0, 1) * np.pi * 0.5) + 0.06 * k)[:, None]
    drop = (t1[None] * (0.07 * np.sin(np.clip(k * 3, 0, 1) * np.pi))[:, None] - up[None] * (1.15 * k ** 1.2)[:, None]
            + A(-0.42, 0, 0.08)[None] * (k ** 1.6)[:, None])
    Cf = p1[None] + out + drop
    C = np.concatenate([C0, Cf])
    tan = np.gradient(C, axis=0)
    nn = np.concatenate([n0, np.repeat(n1[None], len(Cf), 0)])
    W = np.array([nrm(np.cross(a, b)) * 0.1 * (1 - 0.2 * max(0, i - len(C0)) / len(Cf)) for i, (a, b) in enumerate(zip(nn, tan))])
    ribbon(K, 'body', C, W, mat='scarf', bone='chest', th=0.025, cup=0.15, notch=1.4)
    C, W, _ = surf_strap(K, [(-0.5, 8.3, -0.7), (-0.56, 8.5, -0.5), (-0.6, 8.62, -0.3)], 0.08, per=3)
    ribbon(K, 'body', C, W, mat='scarf', bone='chest', th=0.025, cup=0.15, notch=1.5)


def hair(K, fig):
    H, hs, Rh = fig.H, fig.hs, fig.Rh
    hb = K.sdf()
    hb.ell(H((0, 0.16, -0.08)), A(0.5, 0.52, 0.6) * hs, rot=Rh, k=0.1)                      # cap
    hb.ell(H((0, 0.48, -0.12)), A(0.5, 0.38, 0.58) * hs, rot=Rh, k=0.2)                     # bouffant crown
    hb.tube([H((-0.5, 0.3, 0.12)), H((-0.25, 0.6, 0.36)), H((0.08, 0.66, 0.38)), H((0.38, 0.52, 0.26)), H((0.55, 0.25, 0.05))],
            np.array([0.15, 0.2, 0.22, 0.19, 0.14]) * hs, k=0.12)                               # big front wave
    for s in (-1, 1):
        hb.tube([H((s * 0.42, 0.25, 0.0)), H((s * 0.55, -0.05, -0.12)), H((s * 0.58, -0.38, -0.18)), H((s * 0.64, -0.62, -0.08))],
                np.array([0.2, 0.24, 0.22, 0.12]) * hs, k=0.14)                                # side volume swept behind the ear
        hb.tube([H((s * 0.66, -0.55, 0.02)), H((s * 0.82, -0.66, -0.12)), H((s * 0.8, -0.58, -0.32)), H((s * 0.68, -0.5, -0.36))],
                np.array([0.1, 0.11, 0.08, 0.05]) * hs, k=0.06)                                # flipped curl end
    hb.ell(H((0, -0.18, -0.42)), A(0.55, 0.52, 0.36) * hs, rot=Rh, k=0.18)                  # back mass
    hb.tube([H((-0.62, -0.6, -0.15)), H((-0.35, -0.72, -0.5)), H((0.0, -0.75, -0.6)), H((0.35, -0.72, -0.5)), H((0.62, -0.6, -0.15))],
            np.array([0.12, 0.15, 0.16, 0.15, 0.12]) * hs, k=0.12)                             # rolled back hem
    for i, (y0, ln, rr) in enumerate(((0.0, 0.32, 0.24),)):
        a0 = H((-0.12 - 0.1 * i, y0, -0.48))
        hb.tube([a0, a0 + Rh @ A(-0.3, 0.02, -0.35) * ln * hs, a0 + Rh @ A(-0.62, -0.06, -0.55) * ln * hs, a0 + Rh @ A(-0.85, 0.06 - 0.08 * i, -0.62) * ln * hs],
                np.array([rr, rr * 0.88, rr * 0.62, rr * 0.3]) * hs, k=0.18, s=(1, 0.8, 1))   # two broad windswept locks streaming back-left
    hb.box(H((0, -0.42, 0.66)), A(0.36, 0.46, 0.4) * hs, rot=Rh @ S.rot_xyz((-14, 0, 0)), cut=True, k=0.08, rad=0.12)   # face opening
    for s in (-1, 1):
        hb.ell(H((s * 0.44, -0.1, 0.02)), A(0.07, 0.15, 0.1) * hs, rot=Rh, cut=True, k=0.05)   # ears free
    def waves(x, y, z):
        P = pts3(x, y, z)
        L = (P - fig.J['head']) @ Rh / hs
        w = fbm(x * 0.8, y * 0.8, z * 0.8, 2, freq=1.4, seed=4) * 0.8
        # broad S-waves running diagonally around the head + fine strand grooves
        big = np.sin((L[:, 1] * 1.0 - L[:, 2] * 0.45 + 0.35 * np.sin(np.arctan2(L[:, 0], L[:, 2]) * 2) + w * 0.2) * 9.0)
        fine = np.abs(np.sin((L[:, 0] * 0.8 + L[:, 2] * 0.4 + w) * 34))
        return (-0.03 * big - 0.012 * (1 - fine) ** 2 + 0.005) * hs
    hb.disp(waves, band=0.1)
    K.sdf_piece('body', hb, voxel=0.024 * hs, tris=3000, mat='belly', bone='head', ao=0.45)


def weapon(K, part, t, fig, g, hq):
    J = fig.J
    R = fig.hand_frame('R')
    grip = J['wrR'] + R @ (A(0, 0.33, -0.08) * hq)
    up = nrm(np.cross(g, A(1, 0, 0)))
    if up[1] < 0:
        up = -up
    if t == 1:    # uprooted street lamp: iron pole, curved arm, glowing lantern; concrete clod + wires at the butt
        L0, L1 = grip - g * 0.45, grip + g * 3.6
        K.horn(part, [L0, grip, L1], [0.1, 0.09, 0.07], mat='iron', sides=8, bone='handR')
        for o in (0.5, 1.6, 2.6):
            V, T = torus(grip + g * o, g, 0.095, 0.03, nu=12, nv=4)
            K.piece(part, V, T, 'iron', bone='handR')
        arm = [L1, L1 + g * 0.3 + up * 0.25, L1 + g * 0.25 + up * 0.55 + nrm(np.cross(g, up)) * 0.0, L1 - g * 0.1 + up * 0.7]
        K.horn(part, arm, [0.06, 0.055, 0.05, 0.045], mat='iron', sides=6, bone='handR')
        lc = arm[-1] - up * 0.25
        K.horn(part, [lc + up * 0.18, lc - up * 0.12], [0.2, 0.12], mat='glass', sides=8, bone='handR')
        K.horn(part, [lc + up * 0.32, lc + up * 0.18], [0.0, 0.25], mat='iron', sides=8, bone='handR')
        clod = K.sdf()
        clod.ell(L0 - g * 0.05, A(0.2, 0.17, 0.19), k=0)
        clod.ell(L0 - g * 0.14 + up * 0.06, A(0.14, 0.13, 0.15), k=0.08)
        clod.disp(lambda x, y, z: -0.04 * fbm(x, y, z, 2, freq=6.0, seed=9), band=0.08)
        K.sdf_piece(part, clod, voxel=0.035, tris=260, mat='concrete', bone='handR', ao=0.3)
        for i in range(3):
            a = i * 2.1
            w0 = L0 - g * 0.15
            lat = np.cos(a) * nrm(np.cross(g, up)) * 0.12
            K.horn(part, [w0, w0 - g * 0.08 - up * 0.2 + lat, w0 - g * 0.04 - up * 0.42 + lat * 1.6], [0.025, 0.022, 0.015], mat='dark', sides=4, bone='handR')   # torn wires dangling
    elif t == 2:  # radio mast: triangular lattice with braces, red beacon, dish
        side = nrm(np.cross(g, up))
        L0, L1 = grip - g * 0.36, grip + g * 4.2
        rr = 0.24
        rails = [L0 + (np.cos(a) * up + np.sin(a) * side) * rr for a in (0.5, 2.6, 4.7)]
        for r0 in rails:
            K.horn(part, [r0, r0 + (L1 - L0)], [0.045, 0.04], mat='rust', sides=5, per=1, bone='handR')
        nseg = 11
        for i in range(nseg):
            for j in range(3):
                a0 = rails[j] + (L1 - L0) * i / nseg
                a1 = rails[(j + 1) % 3] + (L1 - L0) * (i + 1) / nseg
                K.horn(part, [a0, a1], [0.022, 0.022], mat='rust', sides=4, per=1, bone='handR')
        K.horn(part, [L1, L1 + g * 0.8], [0.04, 0.0], mat='iron', sides=5, bone='handR')
        K.ball(part, L1 + g * 0.12, 0.13, mat='beacon', bone='handR')
        dc = grip + g * 3.0 + up * 0.35
        V, T, _ = geom.sphere((0, 0, 0), 1, 14, 6)
        Rd = frame(nrm(up + side * 0.5), g)
        K.piece(part, dc + (V * A(0.5, 0.12, 0.5)) @ Rd.T, T, 'plate', bone='handR')
        K.horn(part, [dc, dc + nrm(up + side * 0.5) * 0.45], [0.03, 0.0], mat='iron', sides=4, bone='handR')
        clod = K.sdf()
        clod.box(L0 - g * 0.02, A(0.22, 0.2, 0.22), rot=axes_for(g), rad=0.07, k=0)
        clod.disp(lambda x, y, z: -0.06 * fbm(x, y, z, 2, freq=5.0, seed=2), band=0.1)
        K.sdf_piece(part, clod, voxel=0.04, tris=260, mat='concrete', bone='handR', ao=0.3)
    else:         # steel girder club: riveted I-beam with a bent, concrete-crusted head
        side = nrm(np.cross(g, up))
        L0, L1 = grip - g * 0.5, grip + g * 3.4
        mid, ln = (L0 + L1) * 0.5, np.linalg.norm(L1 - L0)
        Rb = np.stack([side, g, up], 1)
        for c, hf in ((mid + up * 0.21, A(0.23, ln / 2, 0.03)), (mid - up * 0.21, A(0.23, ln / 2, 0.03)), (mid, A(0.03, ln / 2, 0.2))):
            V, T = boxmesh(c, Rb, hf)
            K.piece(part, V, T, 'iron', smooth=False, bone='handR')
        for i in range(7):
            o = L0 + g * (0.35 + i * (ln - 0.7) / 6)
            for s in (-1, 1):
                for u in (-1, 1):
                    K.ball(part, o + up * u * 0.245 + side * s * 0.15, 0.03, mat='iron', bone='handR', seg=6, rings=4)
        head = K.sdf()
        head.ell(L1 - g * 0.22, A(0.5, 0.58, 0.46), k=0)
        head.ell(L1 + g * 0.08 + up * 0.2, A(0.34, 0.34, 0.38), k=0.18)
        head.disp(lambda x, y, z: -0.1 * fbm(x, y, z, 3, freq=3.5, seed=6, ridged=True), band=0.13)
        K.sdf_piece(part, head, voxel=0.045, tris=650, mat='concrete', bone='handR', ao=0.4)
        rng = np.random.default_rng(1)
        for i in range(6):
            d = nrm(rng.normal(size=3) + g * 1.5)
            b0 = L1 - g * 0.15 + d * 0.32
            K.horn(part, [b0, b0 + d * 0.4, b0 + d * 0.56 + rng.normal(size=3) * 0.08], [0.028, 0.028, 0.022], mat='rust', sides=4, bone='handR')
        V, T = torus(grip - g * 0.05, g, 0.3, 0.045, nu=14, nv=4)       # cable lashing at the grip
        K.piece(part, V, T, 'dark', bone='handR')


def armor(K, fig, J):
    # T1: billboard pauldron on the raised right shoulder
    sh, el = J['shR'], J['elR']
    u = nrm(el - sh)
    p, n = K.snap(sh + u * 0.3 + A(0.25, 0.1, 0.05))
    n = nrm(A(1, 0.55, 0.25))
    x, y, z = geom.basis(n, A(0, 1, -0.3))
    c = p + n * 0.07
    R = np.stack([x, n, z], 1)
    V, T = boxmesh(c, R, A(0.75, 0.04, 0.55))
    K.piece('arm_t1', V, T, 'plate', smooth=False, bone='upperarmR')
    for hz, mat in ((0.3, 'body'), (-0.05, 'belly')):
        V, T = boxmesh(c + z * hz + n * 0.045, R, A(0.68, 0.012, 0.12))
        K.piece('arm_t1', V, T, mat, smooth=False, bone='upperarmR')
    V, T = boxmesh(c + z * -0.36 + n * 0.045 + x * 0.3, R, A(0.25, 0.012, 0.07))
    K.piece('arm_t1', V, T, 'accent', smooth=False, bone='upperarmR')
    for sx in (-1, 1):                                    # frame rails
        V, T = boxmesh(c + x * sx * 0.77, R, A(0.04, 0.07, 0.6))
        K.piece('arm_t1', V, T, 'iron', smooth=False, bone='upperarmR')
    for sz in (-1, 1):
        V, T = boxmesh(c + z * sz * 0.57, R, A(0.8, 0.07, 0.04))
        K.piece('arm_t1', V, T, 'iron', smooth=False, bone='upperarmR')
    for sx in (-0.45, 0.45):                              # struts down to the shoulder + lamps
        K.horn('arm_t1', [c + x * sx - n * 0.03, p + x * sx * 0.6 - n * 0.05], [0.04, 0.04], mat='iron', sides=4, per=1, bone='upperarmR')
        K.horn('arm_t1', [c + x * sx + z * 0.57, c + x * sx + z * 0.75 + n * 0.12], [0.025, 0.02], mat='iron', sides=4, per=1, bone='upperarmR')
        K.ball('arm_t1', c + x * sx + z * 0.75 + n * 0.12, A(0.08, 0.05, 0.06), mat='glass', bone='upperarmR')
    # T2: water-tower gauntlet on the reaching left forearm
    el, wr = J['elL'], J['wrL']
    d = nrm(wr - el)
    a0, a1 = el + d * 0.35, wr - d * 0.12
    K.horn('arm_t2', [a0, a0 * 0.5 + a1 * 0.5, a1], [0.4, 0.42, 0.39], mat='plate', sides=16, per=1, bone='forearmL')
    for f in (0.08, 0.5, 0.92):
        V, T = torus(a0 + (a1 - a0) * f, d, 0.43, 0.035, nu=20, nv=4)
        K.piece('arm_t2', V, T, 'iron', bone='forearmL')
    K.horn('arm_t2', [a0 - d * 0.02, a0 - d * 0.32], [0.41, 0.13], mat='rust', sides=16, per=1, bone='forearmL')
    x, _, z = geom.basis(d)
    for i in range(4):                                    # stubby tower legs turned into knuckle spikes
        a = i * np.pi / 2 + 0.4
        o = np.cos(a) * x + np.sin(a) * z
        K.horn('arm_t2', [a1 - d * 0.1 + o * 0.34, a1 + d * 0.22 + o * 0.4], [0.04, 0.03], mat='iron', sides=4, per=1, bone='forearmL')
    lp = a0 + (a1 - a0) * 0.5 + nrm(A(0, 1, 0) - d * d[1]) * 0.425
    V, T = boxmesh(lp, frame(nrm(A(0, 1, 0) - d * d[1]), d), A(0.2, 0.01, 0.12))
    K.piece('arm_t2', V, T, 'body', smooth=False, bone='forearmL')
    # T3: steel cables wrapped around both shins + the right forearm
    for sd in 'LR':
        kn, an = J['kn' + sd], J['an' + sd]
        pts = helix(kn + (an - kn) * 0.18, kn + (an - kn) * 0.78, 0.42, 3.2, 0.04, n=72, r1=0.27)
        K.horn('arm_t3', list(pts), [0.045] * len(pts), mat='iron', sides=5, per=1, bone='shin' + sd)
        p, n = K.snap(kn + A(0, 0, 0.4))
        V, T, _ = geom.sphere((0, 0, 0), 1, 14, 6)                                           # hubcap knee guard
        K.piece('arm_t3', p + n * 0.06 + (V * A(0.26, 0.07, 0.26)) @ frame(n, A(0, 1, 0)).T, T, 'chrome', bone='shin' + sd)
    el, wr = J['elR'], J['wrR']
    pts = helix(el + (wr - el) * 0.2, el + (wr - el) * 0.85, 0.34, 2.5, 0.04, r1=0.24)
    K.horn('arm_t3', list(pts), [0.04] * len(pts), mat='iron', sides=5, per=1, bone='forearmR')


def energy(K, fig, J):
    up = nrm(J['chest'] - J['pelvis'])
    # T1: glowing pearl necklace
    cn = J['neck'] + up * -0.08 + A(0, 0, 0.05)
    for i in range(17):
        a = -np.pi * 0.85 + i * (1.7 * np.pi) / 16
        p = cn + A(np.sin(a) * 0.5, -0.12 * np.cos(a * 0.5) ** 2 + 0.0, np.cos(a) * 0.42)
        p, n = K.snap(p)
        K.ball('enr_t1', p + n * 0.05, 0.055 if i != 8 else 0.1, mat='accent', bone='chest', seg=8, rings=5)
    # T2: glowing bangles on both wrists + a pendant
    for sd in 'LR':
        el, wr = J['el' + sd], J['wr' + sd]
        d = nrm(wr - el)
        for o, rr in ((0.16, 0.24), (0.34, 0.27)):
            V, T = torus(wr - d * o, d, rr, 0.035, nu=20, nv=5)
            K.piece('enr_t2', V, T, 'accent', bone='forearm' + sd)
    p, n = K.snap(J['chest'] + A(0, 0.15, 1.2))
    K.crystal('enr_t2', p + n * 0.05, nrm(n - up * 0.6), 0.32, r=0.11, cluster=False, bone='chest')
    # T3: radioactive shards orbiting the hips + glowing veins in the reaching arm
    rng = np.random.default_rng(8)
    for i in range(7):
        a = 2 * np.pi * i / 7 + 0.3
        rr = 2.3 + rng.uniform(-0.2, 0.3)
        p = A(np.sin(a) * rr, 4.4 + 1.7 * np.sin(a * 2 + 1.0), np.cos(a) * rr * 0.85)
        K.crystal('enr_t3', p, nrm(A(rng.uniform(-0.4, 0.4), 1, rng.uniform(-0.4, 0.4))), 0.55 + 0.2 * (i % 3), r=0.13, bone='pelvis', seed=i)
    for sd in 'LR':
        kn, an = J['kn' + sd], J['an' + sd]
        p, n = K.snap(an + (kn - an) * 0.12 + A(0, 0, 0.3))
        V, T = torus(an + (kn - an) * 0.12, nrm(kn - an), 0.27, 0.035, nu=20, nv=5)    # anklets
        K.piece('enr_t3', V, T, 'accent', bone='shin' + sd)


def speed(K, fig, J, sash_c, skirt, sk_top):
    up = nrm(J['chest'] - J['pelvis'])
    # T1: torn scarf knotted at the throat, tail streaming back-left
    nk = J['neck'] + up * 0.1
    V, T = torus(nk, nrm(up + A(0, 0, 0.2)), 0.4, 0.1, nu=18, nv=6, wob=0.05)
    K.piece('spd_t1', V, T, 'scarf', bone='neck')
    k = np.linspace(0, 1, 12)
    st = nk + A(-0.2, -0.05, -0.32)
    C = st[None] + np.stack([-1.2 * k - 0.3 * k ** 2, 0.25 * np.sin(k * 6) * k - 0.4 * k, -2.4 * k], 1)
    W = np.stack([0 * k, 0.17 - 0.06 * k, 0.03 + 0 * k], 1)
    ribbon(K, 'spd_t1', C, W, mat='scarf', bone='neck', th=0.025, cup=0.3, notch=1.2)
    # T2: torn dress streamers flying off the hem
    for i, a in enumerate((-2.3, -1.6, 2.6)):
        hb = A(np.sin(a) * 1.76, sk_top[1] - 3.1, np.cos(a) * 1.44) + A(sk_top[0], 0, sk_top[2])
        d = nrm(A(np.sin(a) * 0.5 - 0.6, -0.15, -1))
        C = hb[None] + d[None] * (1.6 * k)[:, None] + nrm(np.cross(d, A(0, 1, 0)))[None] * (0.18 * np.sin(k * 5 + i) * k)[:, None] + A(0, 0.5, 0)[None] * (k ** 2)[:, None]
        sd = nrm(np.cross(nrm(np.cross(d, A(0, 1, 0))), d))
        W = sd[None] * (0.2 - 0.12 * k)[:, None]
        ribbon(K, 'spd_t2', C, W, mat='scarf', bone='thigh' + ('L' if np.sin(a) < 0 else 'R'), th=0.02, cup=0.25, notch=1.5)
    # T3: long sash tails streaming from the back of the waist
    bp, bn = K.snap(sash_c + A(0, 0, -1.2))
    for s in (-1, 1):
        C = (bp + bn * 0.05)[None] + np.stack([s * 0.3 * k - 1.0 * k ** 1.5, -1.4 * k + 0.2 * np.sin(k * 5 + s), -2.6 * k], 1)
        W = np.stack([0 * k + 0.02, 0.2 - 0.05 * k, 0 * k], 1)
        ribbon(K, 'spd_t3', C, W, mat='ribbon', bone='pelvis', th=0.025, cup=0.3, notch=1.3)
    V, T = torus(bp + bn * 0.05, bn, 0.12, 0.06, nu=10, nv=5)
    K.piece('spd_t3', V, T, 'ribbon', bone='pelvis')


def focus(K, fig):
    H, hs, Rh = fig.H, fig.hs, fig.Rh
    # T1: drop earrings with glowing gems
    for s in (-1, 1):
        e0 = H((s * 0.44, -0.2, 0.0))
        V, T = torus(e0 - Rh @ A(0, 0.05, 0) * hs, Rh @ A(0, 0, 1), 0.05 * hs, 0.012 * hs, nu=10, nv=4)
        K.piece('fcs_t1', V, T, 'gold', bone='head')
        K.horn('fcs_t1', [e0 - Rh @ A(0, 0.1, 0) * hs, e0 - Rh @ A(0, 0.3, 0) * hs], [0.012 * hs, 0.01 * hs], mat='gold', sides=4, bone='head')
        K.crystal('fcs_t1', e0 - Rh @ A(0, 0.3, 0) * hs, Rh @ A(0, -1, 0), 0.14 * hs, r=0.05 * hs, cluster=False, bone='head')
    # T2: tiara band over the front wave with three points
    ti = K.sdf()
    ti.ell(H((0, 0.5, 0.0)), A(0.6, 0.2, 0.62) * hs, rot=Rh, k=0)
    ti.ell(H((0, 0.5, 0.0)), A(0.56, 0.3, 0.58) * hs, rot=Rh, k=0.02, cut=True)
    ti.box(H((0, 0.72, 0.0)), A(0.9, 0.16, 0.9) * hs, rot=Rh, k=0.02, cut=True, rad=0)
    ti.box(H((0, 0.32, 0.0)), A(0.9, 0.14, 0.9) * hs, rot=Rh, k=0.02, cut=True, rad=0)
    ti.box(H((0, 0.5, -0.5)), A(0.9, 0.4, 0.45) * hs, rot=Rh, k=0.02, cut=True, rad=0)
    K.sdf_piece('fcs_t2', ti, voxel=0.03, tris=400, mat='gold', bone='head')
    for i, x in enumerate((-0.28, 0.0, 0.28)):
        b = H((x, 0.62, 0.58 - abs(x) * 0.35))
        hgt = (0.32 if x == 0 else 0.2) * hs
        K.horn('fcs_t2', [b, b + Rh @ A(x * 0.2, 1, 0.25) * hgt], [0.06 * hs, 0.0], mat='gold', sides=4, bone='head')
    # T3: glowing third-eye gem on the brow + crest gem on the tiara
    oball(K, 'fcs_t3', H((0, 0.2, 0.5)), A(0.05, 0.075, 0.03) * hs, Rh, mat='accent', bone='head', seg=10, rings=6)
    K.horn('fcs_t3', [H((-0.06, 0.2, 0.49)), H((0, 0.13, 0.51)), H((0.06, 0.2, 0.49))], [0.008 * hs, 0.01 * hs, 0.008 * hs], mat='gold', sides=4, bone='head')
    K.crystal('fcs_t3', H((0, 0.85, 0.5)), Rh @ A(0, 1, 0.35), 0.4 * hs, r=0.1 * hs, bone='head', seed=4)
