"""Magical Girl: sparkly anime heroine leaping mid-air with a star wand (Wild-only, human scale).

Modelled ~9.5 units tall and exported at SCALE = 0.1. Uses the humanoid helpers from superhero.py.
"""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep
from .superhero import A, Fig, band, beyond, capsule, frame, hair_disp, heart_pts, nrm, oball, pts3, star_pts, torus

DETAIL = 'cloth'
UV_SCALE = 0.9
SCALE = 0.1
SCALE_REF = True
T3_SCALE = 1.12
CAM = dict(az=28, el=6)

LIFT = 0.7


class Skirt:
    """Flared skirt layer: solid cone with radial folds and a scalloped hem (top at c, hanging along -Y)."""

    def __init__(self, c, h, r0, r1, n=12, amp=0.08, scal=0.1, pw=0.8, phase=0.0, rot=None, zs=1.0):
        self.c = np.asarray(c, float)
        self.h, self.r0, self.r1, self.n, self.amp, self.scal, self.pw, self.ph = h, r0, r1, n, amp, scal, pw, phase
        self.R = None if rot is None else S.rot_xyz(rot)
        self.zs = zs
        e = r1 + amp + 0.2
        pad = 0.0 if self.R is None else 0.25 * h
        self.lo = self.c - np.array([e + pad, h + scal + 0.2 + pad, e * zs + pad])
        self.hi = self.c + np.array([e + pad, 0.3 + pad, e * zs + pad])

    def mirrored(self):
        return self

    def dist(self, reg, pad=0):
        lx, ly, lz = np.broadcast_arrays(*S._local(reg, self.c, self.R))
        lz = lz / self.zs
        F = np.float32
        h = -ly
        t = np.clip(h / F(self.h), 0, 1)
        th = np.arctan2(lx, lz)
        R = F(self.r0) + F(self.r1 - self.r0) * t ** F(self.pw) + F(self.amp) * t * np.sin(F(self.n) * th + F(self.ph))
        rad = np.sqrt(lx * lx + lz * lz)
        slope = (self.r1 - self.r0) / self.h
        d_rad = (rad - R) / F(np.sqrt(1 + slope * slope))
        hem = F(self.h) - F(self.scal) * (0.5 - 0.5 * np.sin(F(self.n) * th + F(self.ph)))
        d_bot = h - hem
        d_top = ly
        return np.maximum(np.maximum(d_rad, d_bot), d_top).astype(F)


def ribbon(K, part, C, W, mat='ribbon', bone=None, shade=(0.7, 1.0), th=0.022, cup=0.35, notch=0.0, seg=8):
    """Ribbon with real thickness: a rounded-rectangle cross-section along centreline C with half-width
    vectors W, cupped across its width (so it curls instead of lying flat) and an optional V-notched end."""
    C, W = np.asarray(C, float), np.asarray(W, float)
    n = len(C)
    tan = np.gradient(C, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    sx = np.sign(ca) * np.abs(ca) ** 0.35
    sy = np.sign(sa) * np.abs(sa) ** 0.6
    V, col = [], []
    for i in range(n):
        w = W[i]
        wl = max(np.linalg.norm(w), 1e-6)
        nv = nrm(np.cross(tan[i], w))
        t = i / max(n - 1, 1)
        for j in range(seg):
            p = C[i] + w * sx[j] + nv * (th * sy[j] + cup * wl * (sx[j] ** 2 - 0.5))
            if notch and i == n - 1:
                p = p - tan[i] * notch * wl * (1 - abs(sx[j]))
            V.append(p)
            col.append(shade[0] + (shade[1] - shade[0]) * (0.55 + 0.45 * np.cos(t * np.pi * 2.0)) * (0.85 + 0.15 * abs(sx[j])))
    T = []
    for i in range(n - 1):
        for j in range(seg):
            a, b = i * seg + j, i * seg + (j + 1) % seg
            T += [(a, a + seg, b + seg), (a, b + seg, b)]
    c0, c1 = len(V), len(V) + 1
    V += [C[0], C[-1] - (tan[-1] * notch * np.linalg.norm(W[-1]) if notch else 0)]
    col += [shade[0], shade[1]]
    for j in range(seg):
        T += [(c0, j, (j + 1) % seg), (c1, (n - 1) * seg + (j + 1) % seg, (n - 1) * seg + j)]
    V, T = np.array(V), np.array(T)
    K.piece(part, V, geom.fix_winding(V, T), mat, col=np.clip(col, 0, 1), bone=bone)


def bow(K, part, c, side, back, up, L=1.0, D=0.6, w0=0.12, w1=0.4, tail=1.6, tail_dir=None, mat='ribbon', bone=None, knot=0.16, lift=0.45, n=18):
    """Ribbon bow: two loops (spread along +/-side, curling toward `back`) + two hanging tails."""
    c, side, back, up = [np.asarray(v, float) for v in (c, side, back, up)]
    for s in (-1, 1):
        sd = nrm(side * s + up * lift)
        a = np.linspace(0.12, 2 * np.pi - 0.12, n)
        C = c[None] + sd[None] * (L * 0.5 * (1 - np.cos(a)))[:, None] + back[None] * (D * 0.5 * np.sin(a))[:, None]
        wv = w0 + (w1 - w0) * np.sin(a * 0.5) ** 1.5
        wd = nrm(up - sd * (up @ sd))
        W = wd[None] * wv[:, None]
        ribbon(K, part, C, W, mat, bone, th=max(0.018, w0 * 0.22), cup=0.3)
        if tail:
            td = nrm(tail_dir if tail_dir is not None else -up)
            k = np.linspace(0, 1, 10)
            C = c[None] + (td[None] * tail * k[:, None] + side[None] * s * (0.25 * k + 0.12 * np.sin(k * 5))[:, None] * tail * 0.4 + back[None] * (0.1 * np.sin(k * 4 + s))[:, None] * tail * 0.3)
            wv = w0 * 0.9 + (w1 * 0.6 - w0 * 0.9) * k
            W = nrm(side - td * (side @ td))[None] * wv[:, None]
            ribbon(K, part, C, W, mat, bone, notch=1.2, cup=0.25)
    K.ball(part, c, A(knot * 1.15, knot * 1.05, knot * 0.85), mat=mat, bone=bone, seg=14, rings=8)


def feather_wing(K, part, root, spread, up, normal, n=6, length=1.6, width=0.22, mat='feather', bone=None, tip_mat=None, droop=0.35):
    """Fan of flattened feathers from `root`, spreading between `up` and `spread` directions."""
    root, spread, up, normal = [np.asarray(v, float) for v in (root, spread, up, normal)]
    for i in range(n):
        f = i / max(n - 1, 1)
        d = nrm(up * (1 - f) + spread * f * 1.1)
        ln = length * (0.55 + 0.45 * np.sin(np.pi * (0.25 + 0.6 * f)))
        tip = root + d * ln - up * droop * f * f * ln * 0.3
        mid = root + d * ln * 0.5 + nrm(np.cross(normal, d)) * 0.04 * ln
        K.horn(part, [root, mid, tip], [width * 0.45, width, 0.0], mat=mat, sides=8, per=3, flat=0.22, up=normal, bone=bone, shade=(0.75, 1.0))


def angel_wing(K, part, root, s, span=2.0, rise=1.0, n=7, flen=1.2, width=0.25, back=0.35, mat='feather', bone=None, rows=3):
    """Angel wing: a smooth leading-edge arm with layered feather rows hanging off it. Each row is one
    smooth cambered sheet whose trailing edge is scalloped into rounded feather tips (primaries longest at
    the back, coverts shortest in front), so it shades smoothly instead of as faceted spikes."""
    root = np.asarray(root, float)
    out, upv, bk = A(s, 0, 0), A(0, 1, 0), A(0, 0, -1)
    nrmv = nrm(A(0, 0.15, 1))

    def edge(t):
        return root + out * span * t + upv * rise * np.sin(np.pi * min(t * 0.85, 1.0)) * (1 - 0.25 * t) + bk * back * t
    E = [edge(t) for t in np.linspace(0, 1, 9)]
    K.horn(part, E, np.linspace(width * 0.55, width * 0.18, 9), mat=mat, sides=10, per=3, flat=0.6, up=nrmv, bone=bone, shade=(0.85, 1.0))
    oball(K, part, root + out * span * 0.04, A(width * 0.7, width * 0.8, width * 0.45), None, mat=mat, bone=bone, seg=10, rings=6)
    spec = ((1.0, 0.0, 0.0, 0.78), (0.64, 0.045, 0.15, 0.86), (0.34, 0.09, 0.3, 0.95))[:rows]
    for lm, off, sh, c0 in spec:
        nf = n + int(round(sh * 6))

        def F(u, v, lm=lm, off=off, nf=nf):
            t = 0.04 + 0.96 * u
            d = nrm(-upv * (1.0 - 0.75 * t) + out * (0.25 + 1.1 * t ** 1.4))
            tip = 0.8 + 0.2 * np.abs(np.sin(np.pi * nf * u)) ** 0.5
            ln = flen * lm * (0.6 + 0.6 * t ** 0.8) * (tip if v > 0.55 else 1.0 - (1.0 - tip) * (v / 0.55) ** 3)
            p = edge(t) + d * ln * v
            return p + nrmv * (off + 0.07 * np.sin(np.pi * v) * ln) + bk * 0.02 * v * ln
        nu, nv = nf * 3, 4
        V, T, _ = geom.grid(F, nu, nv)
        col = []
        for a in range(nu + 1):
            for b in range(nv + 1):
                u, v = a / nu, b / nv
                col.append(c0 * (0.82 + 0.18 * np.abs(np.sin(np.pi * nf * u)) ** 0.35) * (0.88 + 0.12 * v))
        K.piece(part, V, T, mat, col=np.clip(col, 0, 1), bone=bone)


def joints():
    J = dict(
        pelvis=(0, 4.95, -0.02), waist=(0, 5.6, 0.0), chest=(0, 6.3, 0.06), neck=(0, 6.95, 0.0), head=(0, 8.12, 0.15),
        shR=(0.74, 6.72, 0.0), elR=(1.3, 7.62, 0.28), wrR=(1.5, 8.62, 0.62),
        shL=(-0.74, 6.72, 0.0), elL=(-1.62, 6.42, 0.22), wrL=(-2.42, 6.86, 0.55),
        hipR=(0.4, 4.75, 0.0), knR=(0.47, 2.78, 0.3), anR=(0.45, 0.88, -0.02), toeR=(0.46, 0.3, 0.42),
        hipL=(-0.4, 4.75, 0.0), knL=(-0.52, 3.0, 0.8), anL=(-0.55, 2.38, -1.02), toeL=(-0.56, 2.12, -1.58),
    )
    return {k: np.asarray(v, float) + A(0, LIFT, 0) for k, v in J.items()}


def build(K):
    col = K.colors
    K.mat('skin', color='#f3cdb6', rough=0.6, vcol=True, detail=False)
    K.mat('hair', color='#c9b4ff', rough=0.45, vcol=True, detail=False)
    K.mat('brow', color='#7a5fb8', rough=0.6)
    K.mat('sclera', color='#ffffff', rough=0.3)
    K.mat('iris', color='#7b3fe0', rough=0.2, emit='#9a6bff', emit_strength=0.6)
    K.mat('glint', color='#ffffff', emit='#ffffff', emit_strength=2.0)
    K.mat('mouth', color='#c0506a', rough=0.5)
    K.mat('blush', color='#ff9fb0', rough=0.7)
    K.mat('body', rough=0.5, detail_strength=0.3)
    K.mat('belly', rough=0.55, detail_strength=0.3)
    K.mat('ribbon', color=col['belly'], rough=0.5, vcol=True, double=True, detail=False)
    K.mat('ribbon2', color=col['body'], rough=0.5, vcol=True, double=True, detail=False)
    K.mat('feather', color='#ffffff', rough=0.55, vcol=True, double=True)
    K.mat('gold', color='#f1c654', rough=0.25, metal=0.85, vcol=True)
    K.mat('wand', color='#fff2fa', rough=0.3, vcol=True)
    K.mat('plate', color='#ffd1ea', rough=0.3, metal=0.35, vcol=True)

    J = joints()
    fig = Fig(J, dict(s=1.0, fem=True, chest=0.66, waist=0.78, hip=0.8, arm=0.82, leg=0.88, neck=0.62, headR=1.32, bust=0.75,
                      pitch=6, yaw=-6, roll=4, hands=dict(L='open', R='fist'), up=dict(L=A(0, 0.3, -1), R=A(0, 0, 1)), foot=0.9))
    B = K.sdf()
    fig.sculpt(B)
    up = nrm(J['chest'] - J['pelvis'])

    # ---- dress: puff sleeves, frilled collar, three skirt layers
    puffs = []
    for sd, s in (('L', -1), ('R', 1)):
        sh, el = J['sh' + sd], J['el' + sd]
        c = sh + (el - sh) * 0.18 + A(s * 0.05, 0.08, 0)
        puffs.append(S.Ell(c, A(0.4, 0.36, 0.38)))
        B.ell(c, A(0.4, 0.36, 0.38), k=0.08)
    collar = Skirt(J['neck'] + A(0, 0.05, 0), 0.32, 0.3, 0.72, n=14, amp=0.04, scal=0.06, pw=0.9, zs=0.85)
    B.add(collar, k=0.04)
    sk_top = J['waist'] + A(0, -0.1, 0)
    petti = Skirt(sk_top, 1.95, 0.62, 1.68, n=26, amp=0.05, scal=0.08, pw=0.75, rot=(-6, 0, 0))
    mid = Skirt(sk_top + A(0, 0.02, 0), 1.72, 0.66, 1.85, n=11, amp=0.11, scal=0.14, pw=0.75, rot=(-6, 0, 0), phase=0.4)
    topl = Skirt(sk_top + A(0, 0.06, 0), 0.95, 0.7, 1.5, n=9, amp=0.08, scal=0.12, pw=0.8, rot=(-4, 0, 0), phase=1.1)
    for sk in (petti, mid, topl):
        B.add(sk, k=0.03)

    # ---- relief: glove / boot frills, sash, collar ribbon
    gloves = fig.m_gloves(cuff=0.62)
    boots = fig.m_boots(frac=0.2)
    sash_c = J['waist'] + A(0, 0.0, 0)

    def relief(x, y, z):
        P = pts3(x, y, z)
        d = np.zeros(len(P), np.float32)
        for sd in 'LR':
            dd = nrm(J['wr' + sd] - J['el' + sd])
            cq = J['wr' + sd] - dd * 0.62
            near = smoothstep(0.0, -0.1, capsule(P, J['el' + sd], J['wr' + sd], 0.45))
            th = np.arctan2(*(P - cq)[:, [0, 2]].T)
            d -= (0.05 + 0.02 * np.sin(th * 9)) * smoothstep(0.08, 0.0, np.abs((P - cq) @ dd)) * near
            kn, an = J['kn' + sd], J['an' + sd]
            cq = kn + (an - kn) * 0.2
            near = smoothstep(0.0, -0.1, capsule(P, kn, an, 0.55))
            d -= 0.06 * smoothstep(0.09, 0.0, np.abs((P - cq) @ nrm(an - kn))) * near
        # sash band at the waist
        d -= 0.05 * smoothstep(0.02, -0.02, band(P, sash_c, up, 0.14)) * smoothstep(0.0, -0.1, capsule(P, J['pelvis'], J['chest'], 1.0))
        # puff sleeve gathers
        for e in puffs:
            r = P - e.c
            th = np.arctan2(r[:, 0], r[:, 2]) + np.arctan2(r[:, 1], r[:, 2])
            d -= 0.025 * np.sin(th * 7) * smoothstep(0.0, -0.1, e.dist(S.Points(P)) - 0.05)
        return d
    B.disp(relief, band=0.12)

    neck_plane = J['chest'] + up * 0.32
    collar_top = J['neck'] + A(0, 0.06, 0)

    def dress(P):
        torso = np.maximum(capsule(P, J['pelvis'] - up * 0.3, J['neck'] + up * 0.2, 0.95), beyond(P, J['neck'] + up * 0.0, -up))
        sk = np.minimum(np.minimum(mid.dist(S.Points(P)), topl.dist(S.Points(P))), torso)
        pf = np.minimum(puffs[0].dist(S.Points(P)), puffs[1].dist(S.Points(P))) - 0.06
        return np.minimum(sk - 0.02, pf)

    def white(P):
        pt = np.maximum(petti.dist(S.Points(P)) - 0.02, np.maximum(0.025 - mid.dist(S.Points(P)), 0.025 - topl.dist(S.Points(P))))
        col_ = np.maximum(collar.dist(S.Points(P)) - 0.03, beyond(P, collar_top, -up))
        sash = np.maximum(band(P, sash_c, up, 0.14), capsule(P, J['pelvis'], J['chest'], 1.0))
        return np.minimum(np.minimum(np.minimum(gloves(P), boots(P)), np.minimum(pt, col_)), sash)

    K.skin(B, voxel=0.04, tris=9000, base_mat='skin', regions=[('body', dress), ('belly', white)])

    # ---- fine head + hands, face
    fig.pieces(K, skin_mat='skin', glove_mat='belly', head_tris=1400, hand_tris=500)
    fig.face_parts(K, big=1.15, seg=14, rings=8)
    H, hs = fig.H, fig.hs
    for s in (-1, 1):
        c, n = fig.snap_head(K, H((s * 0.25, -0.2, 0.5)))
        K.ball('body', c + n * 0.004, A(0.075, 0.035, 0.012) * hs, mat='blush', bone='head', rot=None)

    # ---- hair: cap, bangs, side locks, twin tails
    hb = K.sdf()
    hb.ell(H((0, 0.1, -0.07)), A(0.475, 0.52, 0.55) * hs, rot=fig.Rh, k=0.1)
    hb.ell(H((0, -0.15, -0.22)), A(0.46, 0.45, 0.4) * hs, rot=fig.Rh, k=0.15)
    hb.box(H((0, -0.42, 0.62)), A(0.37, 0.5, 0.45) * hs, rot=fig.Rh @ S.rot_xyz((-18, 0, 0)), cut=True, k=0.08, rad=0.12)
    hb.box(H((0, -0.9, 0.0)), A(0.8, 0.3, 0.8) * hs, rot=fig.Rh, cut=True, k=0.08, rad=0.1)
    # bangs: swept strands ending in points above the eyes
    for i, (x0, x1, ln) in enumerate(((-0.34, -0.42, 0.42), (-0.2, -0.26, 0.5), (-0.05, -0.08, 0.55), (0.1, 0.12, 0.52), (0.25, 0.3, 0.46), (0.36, 0.44, 0.38))):
        hb.tube([H((x0 * 0.6, 0.52, 0.28)), H((x0, 0.32, 0.5)), H((x1, 0.3 - ln * 0.35, 0.54)), H((x1 * 1.05, 0.32 - ln * 0.48, 0.52))],
                [0.13, 0.11, 0.07, 0.02], k=0.05)
    for s in (-1, 1):   # side locks framing the face
        hb.tube([H((s * 0.42, 0.3, 0.25)), H((s * 0.5, -0.05, 0.3)), H((s * 0.47, -0.45, 0.28)), H((s * 0.44, -0.72, 0.24))], [0.13, 0.12, 0.08, 0.02], k=0.06)
    hb.disp(hair_disp(1.0, seed=7, freq=20, amp=0.04), band=0.12)
    K.sdf_piece('body', hb, voxel=0.035, tris=1700, mat='hair', bone='head', ao=0.45)
    # twin tails: S-curved bundles of tapered, ribbon-flattened locks around a thick core
    tail = np.array([(0.66, 0.56, -0.29), (1.05, 0.63, -0.36), (1.42, 0.43, -0.42), (1.62, -0.1, -0.44), (1.55, -0.8, -0.4),
                     (1.36, -1.5, -0.35), (1.46, -2.15, -0.42), (1.8, -2.72, -0.56)])
    TC = S.catmull(tail, 4)
    strands = [(0.0, 0.0, 1.0, 0.24)] + [(2 * np.pi * j / 7 + 0.3, 1.0, (0.74, 0.97, 0.82, 0.9, 0.7, 0.93, 0.8)[j], (0.15, 0.17, 0.14, 0.16, 0.13, 0.16, 0.15)[j]) for j in range(7)]
    for s in (-1, 1):
        for j, (ang, spr, lf, rm) in enumerate(strands):
            k = max(6, int(len(TC) * lf))
            pts = []
            for i in range(k):
                t = i / (len(TC) - 1)
                sp = spr * (0.06 + 0.13 * t) * min(1.0, t * 6 + 0.3)
                wig = 0.07 * np.sin(t * 7.5 + j * 1.3) * min(1.0, t * 3) * spr
                o = A(np.cos(ang) * sp + wig, 0.0, np.sin(ang) * sp)
                pts.append(H((s * (TC[i][0] + o[0]), TC[i][1] + o[1], TC[i][2] + o[2])))
            pts = pts[::3] + ([pts[-1]] if (k - 1) % 3 else [])
            u = np.linspace(0, 1, len(pts))
            rad = rm * 1.3 * hs * np.interp(u, [0, 0.12, 0.45, 0.78, 1.0], [0.62, 1.0, 0.92, 0.55, 0.0])
            wd = fig.Rh @ (A(-np.sin(ang) * s, 0, np.cos(ang)) if spr else A(0, 0, 1))
            K.horn('body', pts, rad, mat='hair', sides=6, per=2, flat=0.55, up=wd, bone='head', shade=(0.68, 1.0))
    # hair ties: scrunchie ring + bow + gem around each tail root
    for s in (-1, 1):
        a0, a1 = H((s * 0.62, 0.55, -0.27)), H((s * 1.0, 0.58, -0.35))
        c = (a0 + a1) * 0.5
        V, T = torus(c, a1 - a0, 0.2 * hs, 0.075 * hs, nu=16, nv=6)
        K.piece('body', V, T, 'ribbon', bone='head')
        bow(K, 'body', c + fig.Rh @ A(0, 0.38, 0.05) * hs, fig.Rh @ A(0, 0, 1), fig.Rh @ A(0, 0.3, -1) * -1, fig.Rh @ A(s, 0.6, 0), L=0.55, D=0.28, w0=0.07, w1=0.2,
            tail=0.0, mat='ribbon2', bone='head', knot=0.09, lift=0.15)
        K.crystal('body', c + fig.Rh @ A(0, 0.4, 0.18) * hs, fig.Rh @ A(0, 0.4, 1), 0.13, r=0.07, cluster=False, bone='head')

    # ---- chest bow + choker gem, giant back bow
    cp, cn = K.snap(J['chest'] + A(0, 0.25, 1.0))
    bow(K, 'body', cp + cn * 0.06, A(1, 0, 0), cn, A(0, 1, 0), L=0.72, D=0.42, w0=0.07, w1=0.24, tail=0.8, tail_dir=A(0, -1, 0.15), mat='ribbon', bone='chest', knot=0.1, lift=0.25)
    K.crystal('body', cp + cn * 0.14, cn, 0.16, r=0.09, cluster=False, bone='chest')
    ch, chn = K.snap(J['neck'] + A(0, 0.25, 0.5))
    K.crystal('body', ch + chn * 0.03, chn, 0.1, r=0.06, cluster=False, bone='neck')
    bk, bkn = K.snap(J['waist'] + A(0, 0.05, -1.0))
    bow(K, 'body', bk + bkn * 0.08, A(1, 0, 0), bkn, A(0, 1, 0), L=1.65, D=0.7, w0=0.14, w1=0.48, tail=2.4, tail_dir=A(0, -1, -0.45), mat='ribbon', bone='spine', knot=0.2, lift=0.5, n=22)

    # ---- wand (shaft in base, star topper per Might tier)
    hc = fig.hand_center('R')
    wd = nrm(A(0.12, 1, 0.12))
    w0, w1 = hc - wd * 0.55, hc + wd * 1.9
    K.horn('body', [w0, hc, w1], [0.07, 0.065, 0.06], mat='wand', sides=8, bone='handR')
    K.ball('body', w0, 0.1, mat='gold', bone='handR')
    for o in (0.3, 0.45):
        V, T = torus(hc + wd * o, wd, 0.085, 0.025, nu=12, nv=4)
        K.piece('body', V, T, 'gold', bone='handR')
    star_c = {}
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        r = 0.42 * (0.75 + 0.35 * sc)
        c = w1 + wd * r * 0.85
        star_c[t] = c
        V, T = star_pts(c, A(0.15, 0.05, 1), wd, r, r * 0.45, depth=r * 0.32)
        K.piece(part, V, T, 'accent', smooth=False, bone='handR')
        V, T = torus(c, A(0.15, 0.05, 1), r * 1.08, 0.035, nu=24, nv=4)
        K.piece(part, V, T, 'gold', bone='handR')
        # little wings on the wand head
        for s in (-1, 1):
            feather_wing(K, part, c - wd * r * 0.4 + A(s * r * 0.8, 0, 0), A(s, -0.3, 0), A(s * 0.4, 1, 0), A(0, 0, 1), n=3 + min(t, 2), length=0.35 + 0.2 * t, width=0.07 + 0.015 * t, mat='feather', bone='handR')
        if t >= 1:
            V, T = heart_pts(w1 - wd * 0.15, A(0, 0, 1), wd, 0.18 + 0.04 * t, depth=0.07)
            K.piece(part, V, T, 'ribbon2', smooth=False, bone='handR')
        if t >= 2:   # streaming ribbons off the wand
            for s in (-1, 1):
                k = np.linspace(0, 1, 12)
                C = (w1 - wd * 0.25)[None] + np.stack([s * (0.3 + 1.2 * k), -0.4 * k - 0.6 * k ** 2 + 0.15 * np.sin(k * 7), -0.2 * k + 0.12 * np.cos(k * 6 + s)], 1) * (0.9 + 0.3 * (t - 2))
                W = np.stack([0 * k, 0.06 + 0.05 * k, 0.02 + 0 * k], 1)
                ribbon(K, part, C, W, mat='ribbon2', bone='handR')
        if t >= 3:
            for i in range(5):
                a = 2 * np.pi * i / 5
                p = c + (np.cos(a) * A(1, 0, 0) + np.sin(a) * wd) * r * 1.6
                K.crystal(part, p, nrm(p - c), 0.18, r=0.06, cluster=False, bone='handR')

    # ---- skeleton + mounts
    fig.rig(K)
    K.bone('wand', hc, w1, 'handR', 0.1)
    fig.mounts(K, back=bk + bkn * 0.3, tail=J['pelvis'] + A(0, -1.6, -1.5), tail_bone='pelvis')

    # ---- ARMOR: heart plates, heart barrier, bracers / shin guards
    for sd, s in (('L', -1), ('R', 1)):
        sh = J['sh' + sd]
        p, n = K.snap(sh + A(s * 0.5, 0.25, 0))
        V, T = heart_pts(p + n * 0.12, n, A(0, 1, 0), 0.32, depth=0.08)
        K.piece('arm_t1', V, T, 'plate', smooth=False, bone='upperarm' + sd)
        el, wr = J['el' + sd], J['wr' + sd]
        K.plate('arm_t2', el + (wr - el) * 0.45, (0.3, 0.3, 0.3), bone='forearm' + sd, lames=2, off=0.04, th=0.05, lame_axis=wr - el, tris=200)
        kn, an = J['kn' + sd], J['an' + sd]
        p, n = K.snap(kn + A(0, 0, 0.4))
        V, T = heart_pts(p + n * 0.1, n, A(0, 1, 0), 0.22, depth=0.06)
        K.piece('arm_t3', V, T, 'plate', smooth=False, bone='shin' + sd)
    hl = fig.hand_center('L')
    V, T = heart_pts(hl + A(-0.3, 0.05, 0.35), nrm(A(-0.55, 0.1, 1)), A(0, 1, 0), 0.9, depth=0.12)
    K.piece('arm_t2', V, T, 'accent', smooth=False, bone='handL')
    V, T = torus(hl + A(-0.3, 0.05, 0.35) + A(0, 0.08, 0), nrm(A(-0.55, 0.1, 1)), 1.05, 0.03, nu=28, nv=4)
    K.piece('arm_t2', V, T, 'gold', bone='handL')
    for i in range(3):   # orbiting heart shields
        a = 2 * np.pi * i / 3 + 0.5
        p = A(np.cos(a) * 2.3, 5.2 + LIFT + 0.6 * np.sin(a * 2), np.sin(a) * 2.3)
        V, T = heart_pts(p, nrm(A(p[0], 0, p[2])), A(0, 1, 0), 0.45, depth=0.1)
        K.piece('arm_t3', V, T, 'plate', smooth=False, bone='pelvis')

    # ---- ENERGY: glowing gems, floating stars, halo
    for sd, s in (('L', -1), ('R', 1)):
        p, n = K.snap(J['kn' + sd] + (J['an' + sd] - J['kn' + sd]) * 0.2 + A(0, 0, 0.45))
        K.crystal('enr_t1', p + n * 0.03, n, 0.14, r=0.07, cluster=False, bone='shin' + sd)
        p, n = K.snap(J['wr' + sd] - nrm(J['wr' + sd] - J['el' + sd]) * 0.55 + A(0, 0.1, 0.3))
        K.crystal('enr_t1', p + n * 0.03, n, 0.12, r=0.06, cluster=False, bone='forearm' + sd)
        K.ball('enr_t1', H((s * 0.47, -0.35, 0.0)), 0.06, mat='accent', bone='head')
    rng = np.random.default_rng(5)
    for i in range(7):
        a = 2 * np.pi * i / 7 + rng.uniform(-0.3, 0.3)
        rr = rng.uniform(2.2, 2.9)
        p = A(np.cos(a) * rr, rng.uniform(2.5, 8.5) + LIFT, np.sin(a) * rr * 0.8)
        V, T = star_pts(p, nrm(A(rng.uniform(-1, 1), 0.3, 1)), A(0, 1, 0), 0.2 + 0.08 * (i % 3), 0.09, depth=0.07)
        K.piece('enr_t2', V, T, 'accent', smooth=False, bone='pelvis')
    halo_c = H((0, 1.02, -0.2))
    V, T = torus(halo_c, fig.Rh @ A(0, 1, 0.6), 0.64, 0.08, nu=36, nv=6)
    K.piece('enr_t3', V, T, 'accent', bone='head')
    for i in range(8):
        a = 2 * np.pi * i / 8
        ax = fig.Rh @ A(0, 1, 0.6)
        x, y, z = geom.basis(ax)
        p = halo_c + (np.cos(a) * x + np.sin(a) * z) * 0.62
        V, T = star_pts(p, ax, nrm(p - halo_c), 0.11, 0.05, depth=0.04)
        K.piece('enr_t3', V, T, 'gold', smooth=False, bone='head')

    # ---- SPEED: boot wings, back wings (small -> big)
    for sd, s in (('L', -1), ('R', 1)):
        an, kn = J['an' + sd], J['kn' + sd]
        p, n = K.snap(an + (kn - an) * 0.12 + A(s * 0.35, 0, 0))
        sh_d = nrm(an - kn)
        feather_wing(K, 'spd_t1', p, nrm(A(s * 0.3, 0, 0) - sh_d * 0.2 + A(0, 0, -1)), nrm(A(s * 0.6, 0, 0) - sh_d), A(1, 0, 0), n=4, length=0.65, width=0.12, bone='shin' + sd, droop=0.0)
    wr0, wn = K.snap(J['chest'] + A(0, 0.0, -1.0))
    for sd, s in (('L', -1), ('R', 1)):
        root = wr0 + A(s * 0.28, 0.05, -0.05)
        angel_wing(K, 'spd_t2', root, s, span=1.6, rise=0.9, n=6, flen=0.9, width=0.2, bone='chest', rows=2)
        angel_wing(K, 'spd_t3', root + A(s * 0.05, 0.12, -0.1), s, span=2.9, rise=1.6, n=9, flen=1.7, width=0.27, bone='chest')

    # ---- FOCUS: tiara, star hair pins, crest gem
    ti = K.sdf()
    ti.ell(H((0, 0.4, 0.05)), A(0.56, 0.2, 0.6), rot=fig.Rh, k=0)
    ti.ell(H((0, 0.4, 0.05)), A(0.52, 0.3, 0.56), rot=fig.Rh, k=0.02, cut=True)
    ti.box(H((0, 0.62, 0.05)), A(0.8, 0.15, 0.8), rot=fig.Rh, k=0.02, cut=True, rad=0)
    ti.box(H((0, 0.2, 0.05)), A(0.8, 0.13, 0.8), rot=fig.Rh, k=0.02, cut=True, rad=0)
    ti.box(H((0, 0.4, -0.45)), A(0.8, 0.4, 0.4), rot=fig.Rh, k=0.02, cut=True, rad=0)
    K.sdf_piece('fcs_t1', ti, voxel=0.025, tris=360, mat='gold', bone='head')
    tc = H((0, 0.52, 0.6))
    V, T = star_pts(tc, fig.Rh @ A(0, 0.25, 1), fig.Rh @ A(0, 1, 0), 0.16, 0.07, depth=0.06)
    K.piece('fcs_t1', V, T, 'accent', smooth=False, bone='head')
    for s in (-1, 1):
        p = H((s * 0.38, 0.62, 0.3))
        V, T = star_pts(p, fig.Rh @ A(s * 0.5, 0.4, 1), fig.Rh @ A(0, 1, 0), 0.14, 0.06, depth=0.05)
        K.piece('fcs_t2', V, T, 'gold', smooth=False, bone='head')
        K.crystal('fcs_t2', H((s * 0.62, 0.48, -0.1)), fig.Rh @ A(s, 0.6, 0.3), 0.18, r=0.06, cluster=False, bone='head')
    K.crystal('fcs_t3', H((0, 0.66, 0.55)), fig.Rh @ A(0, 1, 0.5), 0.42, r=0.11, bone='head', seed=2)
    for s in (-1, 1):   # little golden head wings at the tiara's sides
        feather_wing(K, 'fcs_t3', H((s * 0.55, 0.42, 0.1)), fig.Rh @ A(s, 0.1, -0.6), fig.Rh @ A(s * 0.5, 1, -0.3), fig.Rh @ A(0.0, 0.0, 1) * 0 + fig.Rh @ A(-s * 0.2, 0, 1), n=4, length=0.55, width=0.09, mat='gold', bone='head', droop=0.0)
