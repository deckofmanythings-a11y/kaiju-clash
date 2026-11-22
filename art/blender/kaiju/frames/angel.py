"""Angel: a statuesque winged divine woman in a flowing Grecian gown, hovering with a halo (secret Wild easter egg).

Human scale (like superhero.py): modelled ~10 units tall (wings rise to ~12.5) and exported at SCALE = 0.1, toes
pointed just above the ground (the runtime adds hover; GLB_FRAMES marks it `tiny`). The Seraph is the kaiju-sized angel. Uses the humanoid helpers from superhero.py with an adult head / face of its own.
"""
import numpy as np

from .. import geom
from .. import sdf as S
from ..noise import fbm, smoothstep
from .magicalgirl import Skirt, ribbon
from .superhero import A, Fig, band, beyond, capsule, hair_disp, nrm, oball, pts3, torus

INFO = dict(name='Angel', colors=dict(body='#f4efe4', belly='#d8a743', accent='#ffd894'))

DETAIL = 'cloth'
UV_SCALE = 0.8
CAM = dict(az=30, el=6)
SCALE = 0.1
SCALE_REF = True
T3_SCALE = 1.12


# ====================================================================== small geometry helpers
def closed_tube(C, r, nv=6):
    """Tube along a closed polyline C (N,3)."""
    C = np.asarray(C, float)
    n = len(C)
    c = C.mean(0)
    pn = np.zeros(3)
    for i in range(n):
        pn += np.cross(C[i] - c, C[(i + 1) % n] - c)
    pn = nrm(pn)
    V = []
    for i in range(n):
        rad = C[i] - c
        rad = nrm(rad - pn * (rad @ pn))
        for j in range(nv):
            b = 2 * np.pi * j / nv
            V.append(C[i] + r * (np.cos(b) * rad + np.sin(b) * pn))
    T = []
    for i in range(n):
        for j in range(nv):
            a0, a1 = i * nv + j, i * nv + (j + 1) % nv
            b0, b1 = ((i + 1) % n) * nv + j, ((i + 1) % n) * nv + (j + 1) % nv
            T += [(a0, b0, b1), (a0, b1, a1)]
    V, T = np.array(V), np.array(T)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    ring = np.repeat(C, nv, 0)[T[:, 0]]
    flip = (fn * (V[T].mean(1) - ring)).sum(1) < 0
    T[flip] = T[flip][:, ::-1]
    return V, T


def feather(K, part, b, d, n, L, w, bone, mat='feather', asym=0.2, cup=0.12, bend=0.04, nu=7, nv=4, shade=(0.72, 1.0), tip=5.0, rr=None):
    """One vaned feather: a cupped, round-tipped blade from base b along d, broad face toward n."""
    b, d = np.asarray(b, float), nrm(d)
    n = nrm(np.asarray(n, float) - d * (np.asarray(n, float) @ d))
    wv = nrm(np.cross(n, d))

    def F(u, v):
        x = v * 2 - 1
        u = 1 - (1 - u) ** 1.8
        rr_ = rr if rr is not None else min(0.45, w * tip * 0.25 / max(L, 1e-6))   # elliptic tip length (fraction of L)
        e = max(0.0, (u - (1 - rr_)) / rr_)
        hw = w * min(1.0, u / 0.14) ** 0.6 * np.sqrt(max(0.0, 1 - e * e)) * (1 - 0.08 * u)
        sd = (1 - asym) if x < 0 else (1 + asym * 0.5)
        c = b + d * (L * u) + wv * (bend * L * u * u) + n * (0.035 * L * np.sin(np.pi * u))
        return c + wv * (hw * x * sd) + n * (cup * hw * (1 - x * x))
    V, T, _ = geom.grid(F, nu, nv)
    col = []
    for i in range(nu + 1):
        for j in range(nv + 1):
            u, x = 1 - (1 - i / nu) ** 1.8, j / nv * 2 - 1
            col.append((shade[0] + (shade[1] - shade[0]) * min(1.0, u * 1.6) ** 0.7) * (0.9 + 0.1 * (1 - abs(x))) * (1 - 0.12 * u ** 6))
    K.piece(part, V, T, mat, col=np.clip(col, 0, 1), bone=bone)


def wing(K, part, P, bones, scale=1.0, mat='feather', n_sec=13, n_pri=10, coverts=3, back=True, g=(0, -1, -0.25), arm_r=0.2, lift=0.25, cov_n=1.0, fnu=0):
    """Layered feathered wing along the leading-edge arm P[0..3] (root, elbow, wrist, hand tip).
    Primaries fan from the hand, secondaries/tertials hang off the forearm/arm, 3 covert rows on the front,
    one on the back. Every feather is bound to the wing bone of the arm segment it grows from."""
    P = [np.asarray(p, float) for p in P]
    for i in range(3):
        a, b = P[i], P[i + 1]
        r0, r1 = arm_r * scale * (1 - 0.22 * i), arm_r * scale * (1 - 0.22 * (i + 1)) * (0.3 if i == 2 else 1)
        K.horn(part, [a, a + (b - a) * 0.5, b], [r0, (r0 + r1) / 2, r1], mat=mat, sides=6, per=2, bone=bones[i], shade=(0.85, 1.0))
        if i < 2:
            K.ball(part, b, r1 * 1.08, mat=mat, bone=bones[i + 1], seg=6, rings=5)
    C = S.catmull(np.array(P), 8)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    tot = seg[-1]
    knots = [np.argmin(np.abs(np.linalg.norm(C - p, axis=1))) for p in P]
    uk = [seg[k] / tot for k in knots]

    def pos(u):
        s = u * tot
        return np.array([np.interp(s, seg, C[:, i]) for i in range(3)])

    def tan(u):
        return nrm(pos(min(1, u + 0.02)) - pos(max(0, u - 0.02)))

    def bone_at(u):
        return bones[0] if u < uk[1] else (bones[1] if u < uk[2] else bones[2])

    g = nrm(g)
    nW = nrm(np.cross(P[2] - P[0], g))
    if nW[2] < 0:
        nW = -nW
    down = nrm(g - nW * (g @ nW))
    upv = A(0, 1, 0)
    uw = uk[2]
    d_pri0 = nrm(down + tan(uw) * 0.55)
    d_pri1 = nrm(tan(0.995) + upv * lift)

    def flight(u):
        """direction, length, width of the flight feather under arm position u."""
        if u <= uw:
            f = np.clip((u - 0.06) / max(uw - 0.06, 1e-3), 0, 1)
            return nrm(down + tan(u) * (-0.05 + 0.45 * f)), (2.1 + 1.3 * f ** 0.8) * scale, 0.33 * scale
        f = np.clip((u - uw) / max(1 - uw, 1e-3), 0, 1)
        return nrm(d_pri0 * (1 - f) + d_pri1 * f), (3.4 + 0.9 * np.sin(np.pi * (0.15 + 0.7 * f))) * scale, (0.34 - 0.07 * f) * scale
    # secondaries + tertials
    for i in range(n_sec):
        f = i / (n_sec - 1)
        u = 0.07 + (uw - 0.07) * f
        d, L, w = flight(u)
        feather(K, part, pos(u) - nW * (0.03 + 0.006 * (i % 2)) * scale, d, nW, L, w, bone_at(u), mat=mat, asym=0.1, bend=0.03, nu=fnu or 6, rr=0.24)
    # primaries
    for i in range(n_pri):
        f = i / (n_pri - 1)
        u = uw + (0.985 - uw) * f
        d, L, w = flight(u + 1e-4)
        feather(K, part, pos(u) - nW * (0.05 + 0.012 * i) * scale, d, nW, L, w, bone_at(u), mat=mat, asym=0.15 + 0.3 * f, bend=0.05, nu=fnu or 8, rr=0.3)
    # coverts: greater, median, marginal (front), greater (back)
    rows = [(0.46, 0.05, 16, 0.05, 1.15, (0.7, 0.98)), (0.27, 0.1, 14, 0.04, 1.0, (0.68, 0.97)), (0.14, 0.145, 11, 0.03, 0.8, (0.66, 0.95))][:coverts]
    if back:
        rows.append((0.42, -0.11, 13, 0.06, 1.15, (0.62, 0.92)))
    for lf, off, cnt, u0, ws, sh in rows:
        cnt = max(4, int(round(cnt * cov_n)))
        for i in range(cnt):
            f = i / (cnt - 1)
            u = u0 + (0.97 - u0) * f
            d, L, w = flight(u)
            nn = nW if off > 0 else -nW
            feather(K, part, pos(u) + nW * (off + 0.008 * (i % 2)) * scale, d, nn, L * lf + 0.12 * scale, w * ws * 1.05, bone_at(u), mat=mat,
                    asym=0.05, cup=0.18, bend=0.0, nu=4, nv=3, shade=sh, tip=3.0)
    return nW


def sword_blade(K, part, base, ax, wv, fn, L, prof, mat='accent', bone=None, shade=(0.78, 1.0)):
    """Crisp flat-diamond blade from `base` along `ax`: edges along +-wv, flat faces toward +-fn.
    prof: [(u along L, half width, half thickness), ...] ending at (1, 0, 0) for the point."""
    V, col = [], []
    rings = prof[:-1]
    for u, hw, t in rings:
        c = base + ax * (L * u)
        V += [c + wv * hw, c + fn * t, c - wv * hw, c - fn * t]
        col += [shade[0] + (shade[1] - shade[0]) * u] * 4
    V.append(base + ax * (L * prof[-1][0]))
    col.append(shade[1])
    n = len(rings)
    T = []
    for i in range(n - 1):
        for j in range(4):
            a0, a1 = i * 4 + j, i * 4 + (j + 1) % 4
            T += [(a0, a0 + 4, a1 + 4), (a0, a1 + 4, a1)]
    last, tip = (n - 1) * 4, n * 4
    for j in range(4):
        T.append((last + j, tip, last + (j + 1) % 4))
    T += [(0, 2, 1), (0, 3, 2)]
    V, T = np.array(V), np.array(T)
    K.piece(part, V, geom.fix_winding(V, T), mat, smooth=False, col=np.array(col), bone=bone)

# ====================================================================== the figure
class AngelFig(Fig):
    """Fig with an adult, softly sculpted female head (instead of the anime face)."""

    def build_head(self):
        H, hs, Rh = self.H, self.hs, self.Rh
        B = S.SDF()
        B.tube([H((0, -0.3, -0.1)), H((0, -0.85, -0.12))], np.array([0.2, 0.2]) * hs, k=0.0)       # neck plug
        B.ell(H((0, 0.08, -0.05)), A(0.395, 0.45, 0.475) * hs, rot=Rh, k=0.1)                       # cranium
        B.ell(H((0, -0.12, 0.11)), A(0.305, 0.36, 0.335) * hs, rot=Rh, k=0.14)                      # face mass
        B.ell(H((0, -0.34, 0.17)), A(0.2, 0.16, 0.21) * hs, rot=Rh, k=0.14)                         # lower face
        for s in (-1, 1):
            B.tube([H((s * 0.31, -0.06, -0.06)), H((s * 0.25, -0.32, 0.05)), H((s * 0.15, -0.475, 0.23)), H((s * 0.05, -0.535, 0.315))],
                   np.array([0.068, 0.066, 0.058, 0.052]) * hs, k=0.12)                              # soft jawline
            B.ell(H((s * 0.235, -0.095, 0.33)), A(0.085, 0.055, 0.065) * hs, rot=Rh, k=0.09)          # cheekbone
            B.ell(H((s * 0.19, -0.21, 0.3)), A(0.12, 0.1, 0.1) * hs, rot=Rh, k=0.12)                 # soft cheek
            B.ell(H((s * 0.155, -0.005, 0.47)), A(0.088, 0.052, 0.058) * hs, rot=Rh, k=0.045, cut=True)   # eye socket
            B.ell(H((s * 0.41, -0.08, -0.02)), A(0.045, 0.11, 0.075) * hs, rot=Rh @ S.rot_xyz((0, s * 20, 0)), k=0.04)  # ear
            B.ell(H((s * 0.04, -0.2, 0.505)), A(0.032, 0.028, 0.032) * hs, rot=Rh, k=0.025)         # nostril wing
        B.ell(H((0, -0.515, 0.315)), A(0.07, 0.058, 0.064) * hs, rot=Rh, k=0.07)                    # chin
        B.tube([H((-0.3, 0.1, 0.33)), H((-0.15, 0.115, 0.44)), H((0, 0.1, 0.462)), H((0.15, 0.115, 0.44)), H((0.3, 0.1, 0.33))],
               np.array([0.028, 0.033, 0.028, 0.033, 0.028]) * hs, k=0.08)                          # soft brow
        B.tube([H((0, 0.06, 0.465)), H((0, -0.08, 0.515)), H((0, -0.17, 0.54))], np.array([0.022, 0.028, 0.035]) * hs, k=0.035)  # nose
        B.ell(H((0, -0.18, 0.536)), A(0.04, 0.035, 0.038) * hs, rot=Rh, k=0.03)                     # nose tip
        B.ell(H((0, -0.27, 0.445)), A(0.115, 0.05, 0.058) * hs, rot=Rh, k=0.06)                        # philtrum / upper lip mass
        B.ell(H((0, -0.295, 0.482)), A(0.08, 0.025, 0.034) * hs, rot=Rh, k=0.025)                  # upper lip
        B.ell(H((0, -0.345, 0.472)), A(0.066, 0.03, 0.036) * hs, rot=Rh, k=0.03)                    # lower lip
        B.ell(H((0, -0.319, 0.508)), A(0.082, 0.007, 0.04) * hs, rot=Rh, k=0.008, cut=True)         # mouth line
        self.head_sdf = B
        return B

    def face(self, K, part='body', bone='head'):
        H, hs, Rh = self.H, self.hs, self.Rh
        snap = lambda P: self.snap_head(K, P)
        for s in (-1, 1):
            c = H((s * 0.155, -0.012, 0.442))
            Re = Rh @ S.rot_xyz((0, s * 14, 0))
            n = Re @ A(0, 0, 1)
            oball(K, part, c - n * 0.004 * hs, A(0.078, 0.04, 0.03) * hs, Re, mat='sclera', bone=bone, seg=12, rings=6)
            oball(K, part, c + n * 0.016 * hs, A(0.037, 0.037, 0.016) * hs, Re, mat='accent', bone=bone, seg=10, rings=6)
            oball(K, part, c + n * 0.026 * hs, A(0.015, 0.016, 0.008) * hs, Re, mat='pupil', bone=bone, seg=8, rings=5)
            oball(K, part, c + n * 0.03 * hs + Re @ A(s * 0.012, 0.012, 0) * hs, A(0.007, 0.007, 0.004) * hs, Re, mat='glint', bone=bone, seg=6, rings=4)
            # upper lash line with a little outer wing, lower lid hint
            lid = [c + Re @ A(-s * 0.082, -0.004, -0.01) * hs, c + Re @ A(0, 0.042, 0.013) * hs, c + Re @ A(s * 0.078, 0.016, -0.008) * hs,
                   c + Re @ A(s * 0.11, 0.034, -0.03) * hs]
            K.horn(part, lid, np.array([0.006, 0.013, 0.01, 0.0]) * hs, mat='lash', sides=4, flat=0.5, up=n, bone=bone)
            low = [c + Re @ A(-s * 0.06, -0.022, -0.004) * hs, c + Re @ A(0, -0.034, 0.008) * hs, c + Re @ A(s * 0.06, -0.016, -0.006) * hs]
            K.horn(part, low, np.array([0.003, 0.005, 0.003]) * hs, mat='lash', sides=4, flat=0.5, up=n, bone=bone)
            bp, bn = snap(np.array([H((s * 0.06, 0.13, 0.56)), H((s * 0.17, 0.165, 0.53)), H((s * 0.29, 0.125, 0.44))]))
            K.horn(part, list(bp + bn * 0.008 * hs), np.array([0.016, 0.014, 0.003]) * hs, mat='brow', sides=4, flat=0.45, up=n, bone=bone)
        # rose lips
        for y, z, w_, r in ((-0.298, 0.512, 0.078, 0.014), (-0.344, 0.505, 0.064, 0.016)):
            mp, mn = snap(np.array([H((-w_, y + 0.008, z - 0.03)), H((0, y, z)), H((w_, y + 0.008, z - 0.03))]))
            K.horn(part, list(mp - mn * 0.006 * hs), np.array([0.006, r, 0.006]) * hs, mat='lips', sides=6, flat=0.4, up=mn[1], bone=bone)


def joints():
    return {k: np.asarray(v, float) for k, v in dict(
        pelvis=(0, 5.72, 0.0), waist=(0, 6.52, 0.04), chest=(0, 7.36, 0.08), neck=(0, 8.36, 0.02), head=(0, 9.3, 0.13),
        shR=(0.84, 7.98, 0.0), elR=(1.5, 6.98, -0.1), wrR=(1.98, 6.66, 0.8),
        shL=(-0.84, 7.98, 0.0), elL=(-1.42, 7.05, 0.72), wrL=(-1.78, 6.88, 1.8),
        hipR=(0.42, 5.5, 0.0), knR=(0.47, 3.38, 0.08), anR=(0.43, 1.28, -0.16), toeR=(0.44, 0.7, -0.02),
        hipL=(-0.42, 5.42, 0.04), knL=(-0.4, 3.62, 0.98), anL=(-0.46, 1.62, 0.4), toeL=(-0.48, 1.0, 0.3),
    ).items()}


def build(K):
    col = K.colors
    K.mat('skin', color='#f4cdb2', rough=0.55, vcol=True, detail=False)
    K.mat('sclera', color='#ece6dc', rough=0.35)
    K.mat('pupil', color='#6b4a1a', emit='#a8742a', emit_strength=0.6, rough=0.2)
    K.mat('glint', color='#ffffff', emit='#ffffff', emit_strength=3.0)
    K.mat('lash', color='#4a3020', rough=0.6)
    K.mat('brow', color='#a77a3a', rough=0.6)
    K.mat('lips', color='#c46d72', rough=0.4, vcol=True)
    K.mat('body', rough=0.55, detail_strength=0.25)
    K.mat('belly', rough=0.42, metal=0.12, detail=False, double=True)
    K.mat('feather', color='#fbf8f1', rough=0.62, vcol=True, double=True, detail=False)
    K.mat('gold', color='#e6b94e', rough=0.24, metal=0.88, vcol=True, detail=False)
    K.mat('gilt', color='#f0c862', rough=0.38, metal=0.45, vcol=True, double=True, detail=False)
    K.mat('plate', color='#e9c56a', rough=0.26, metal=0.82, vcol=True, detail=False)
    K.mat('blade', color='#eef2f8', rough=0.16, metal=0.9, vcol=True, detail=False)
    K.mat('shaft', color='#f5ecd8', rough=0.4, vcol=True, detail=False)

    J = joints()
    yR = nrm(J['wrR'] - J['elR'])
    fig = AngelFig(J, dict(s=1.0, fem=True, chest=0.74, waist=0.72, hip=0.86, arm=0.86, leg=0.95, neck=0.66, headR=1.1, bust=0.55,
                           pitch=-4, yaw=6, roll=-5, hands=dict(L='open', R='fist'), up=dict(L=A(0, -1, 0.1), R=np.cross(A(0, 1, 0), yR)),
                           foot=0.92, spread=0.8))
    H, hs = fig.H, fig.hs
    up, side, fwd = fig.axes()
    B = K.sdf()
    fig.sculpt(B)

    # ---- gown: wind-trailed floor-length skirt + drape over the forward (left) knee
    sk = Skirt(J['waist'] + A(0, 0.05, 0.06), 5.0, 0.6, 1.5, n=14, amp=0.085, scal=0.14, pw=0.85, phase=0.6, rot=(3, 0, -3), zs=0.9)
    B.add(sk, k=0.05)
    knL, hipL = J['knL'], J['hipL']
    B.tube([hipL + A(0.05, 0.3, 0.45), knL + A(0, 0.1, 0.1), knL + A(-0.02, -0.45, 0.02)], [0.4, 0.34, 0.16], k=0.5)
    # cap sleeves (Grecian chiton gathered at the shoulder)
    for sd, s in (('L', -1), ('R', 1)):
        sh, el = J['sh' + sd], J['el' + sd]
        B.ell(sh + (el - sh) * 0.14 + A(s * 0.04, 0.06, 0), A(0.34, 0.3, 0.34), k=0.1)

    def hem_local(P):
        lx, ly, lz = np.broadcast_arrays(*S._local(S.Points(P), sk.c, sk.R))
        th = np.arctan2(lx, lz / sk.zs)
        hem = sk.h - sk.scal * (0.5 - 0.5 * np.sin(sk.n * th + sk.ph))
        return -ly, hem, th

    sash_c = J['waist'] + up * 0.1

    def neck_y(P):
        ax = np.abs(P[:, 0])
        front = P[:, 2] > J['chest'][2]
        return J['chest'][1] + np.where(front, 0.36, 0.56) + 0.85 * np.clip(ax / 0.6, 0, 1.5) ** 2

    def armband(P, sd):
        sh, el = J['sh' + sd], J['el' + sd]
        u = nrm(el - sh)
        return np.maximum(band(P, sh + (el - sh) * 0.62, u, 0.045), capsule(P, sh, el, 0.4))

    def sandals(P):
        out = np.full(len(P), 9.0)
        for sd in 'LR':
            kn, an, toe = J['kn' + sd], J['an' + sd], J['toe' + sd]
            v = nrm(an - kn)
            f = nrm(toe - an)
            near = capsule(P, an - v * 0.3, toe, 0.36)
            out = np.minimum(out, np.maximum(band(P, an - v * 0.06, v, 0.035), near))
            out = np.minimum(out, np.maximum(band(P, an + f * 0.38, f, 0.03), near))
        return out

    # ---- relief: bodice gathers radiating from the sash, skirt folds, trims, bands, sandal straps
    def relief(x, y, z):
        P = pts3(x, y, z)
        d = np.zeros(len(P), np.float32)
        torso = smoothstep(0.0, -0.15, capsule(P, J['pelvis'], J['neck'], 1.05))
        sash = smoothstep(0.02, -0.02, band(P, sash_c, up, 0.13)) * torso
        d -= 0.05 * sash
        # bodice: gathered folds fanning up from the sash centre toward the shoulders
        rel = P - (sash_c - up * 0.25)
        th = np.arctan2(rel[:, 0], rel[:, 1])
        bod = torso * smoothstep(0.0, 0.2, y - sash_c[1] - 0.13) * smoothstep(0.0, 0.12, neck_y(P) - y)
        d -= 0.022 * np.sin(th * 26) * bod
        # skirt: finer secondary folds growing toward the hem + trim band
        hl, hem, th2 = hem_local(P)
        t = np.clip(hl / sk.h, 0, 1)
        skz = smoothstep(0.0, -0.15, sk.dist(S.Points(P)) - 0.25)
        d -= (0.025 * np.sin(th2 * 31 + 1.3 * t) * t ** 1.5 + 0.012 * fbm(x, y, z, 2, freq=1.5, seed=3)) * skz * smoothstep(0.0, 0.3, t)
        d -= 0.02 * smoothstep(-0.02, 0.02, hl - (hem - 0.2)) * skz
        # neckline trim
        d -= 0.02 * smoothstep(0.04, 0.0, np.abs(y - neck_y(P) + 0.06)) * torso
        for sd in 'LR':
            d -= 0.035 * smoothstep(0.01, -0.01, armband(P, sd))
        d -= 0.022 * smoothstep(0.01, -0.01, sandals(P))
        return d
    B.disp(relief, band=0.12)

    def feet_fn(P):
        out = np.full(len(P), 9.0)
        for sd in 'LR':
            kn, an, toe = J['kn' + sd], J['an' + sd], J['toe' + sd]
            out = np.minimum(out, capsule(P, an + (kn - an) * 0.18, toe, 0.42))
        return np.maximum(out, 0.04 - sk.dist(S.Points(P)))

    def skin_fn(P):
        out = feet_fn(P)
        for sd in 'LR':
            sh, el = J['sh' + sd], J['el' + sd]
            u = nrm(el - sh)
            arm = np.maximum(capsule(P, sh, fig.hand_tip(sd), 0.5), beyond(P, sh + u * 0.42, u))
            out = np.minimum(out, arm)
        top = np.maximum(capsule(P, J['chest'], J['head'], 1.2), neck_y(P) - P[:, 1])
        return np.minimum(out, top)

    def gold_robe(P):
        torso = capsule(P, J['pelvis'] - up * 0.3, J['neck'], 1.1)
        sash = np.maximum(band(P, sash_c, up, 0.13), torso)
        hl, hem, _ = hem_local(P)
        trim = np.maximum(hem - 0.2 - hl, sk.dist(S.Points(P)) - 0.5)
        neck = np.maximum(np.abs(P[:, 1] - neck_y(P) + 0.06) - 0.05, capsule(P, J['chest'], J['neck'], 1.2))
        return np.minimum(np.minimum(sash, trim), neck)

    def gold_skin(P):
        return np.minimum(np.minimum(armband(P, 'L'), armband(P, 'R')), sandals(P))

    K.skin(B, voxel=0.045, tris=8300, base_mat='body', regions=[('belly', gold_robe), ('skin', skin_fn), ('belly', gold_skin)])

    # ---- fine head + hands, face
    fig.pieces(K, skin_mat='skin', glove_mat='skin', head_tris=2000, hand_tris=550)
    fig.face(K)

    # ---- hair: centre-parted cap + face-framing curtains (head), long flowing back mass + locks (chest)
    Rh = fig.Rh
    hb = K.sdf()
    hb.ell(H((0, 0.12, -0.07)), A(0.445, 0.5, 0.53) * hs, rot=Rh, k=0.1)
    hb.ell(H((0, -0.12, -0.24)), A(0.45, 0.52, 0.42) * hs, rot=Rh, k=0.15)
    for s in (-1, 1):
        hb.tube([H((s * 0.04, 0.5, 0.33)), H((s * 0.3, 0.4, 0.4)), H((s * 0.45, 0.08, 0.3)), H((s * 0.47, -0.3, 0.2)), H((s * 0.43, -0.68, 0.1))],
                np.array([0.07, 0.11, 0.12, 0.1, 0.05]) * hs, k=0.08)
    hb.tube([H((0.02, 0.5, 0.4)), H((0.18, 0.37, 0.47)), H((0.35, 0.22, 0.43)), H((0.46, 0.0, 0.31))], np.array([0.07, 0.09, 0.09, 0.07]) * hs, k=0.06)
    hb.tube([H((-0.02, 0.5, 0.4)), H((-0.2, 0.42, 0.45)), H((-0.38, 0.28, 0.4))], np.array([0.07, 0.08, 0.06]) * hs, k=0.06)
    hb.box(H((0, -0.42, 0.62)), A(0.355, 0.56, 0.42) * hs, rot=Rh @ S.rot_xyz((-14, 0, 0)), cut=True, k=0.07, rad=0.12)
    hb.box(H((0, -1.05, 0.35)), A(0.8, 0.35, 0.5) * hs, rot=Rh, cut=True, k=0.08, rad=0.1)
    hb.disp(lambda x, y, z: hair_disp(1.0, seed=7, freq=24, amp=0.03)(x, y, z) + 0.02 * smoothstep(0.03, 0.0, np.abs(x - J['head'][0] - 0.0)) * smoothstep(9.5, 9.9, y),
            band=0.1)
    K.sdf_piece('body', hb, voxel=0.024, tris=1500, mat='belly', bone='head', ao=0.45)

    # long back hair mass, offset from the snapped back surface, drifting left (-X) in the wind
    lb = K.sdf()
    ys = [8.85, 8.2, 7.4, 6.6, 5.8, 5.1]
    drift = [0.0, -0.03, -0.1, -0.22, -0.38, -0.58]
    rad = [0.44, 0.6, 0.66, 0.62, 0.5, 0.24]
    pts = []
    for y, dx in zip(ys, drift):
        p, n = K.snap(A(dx, y, -1.5))
        pts.append(p + A(0, 0, -0.18 - 0.03 * (8.85 - y)))
    pts[0] = H((0, -0.3, -0.4))
    lb.tube(pts, rad, k=0.0, s=(1.0, 1.0, 0.6))
    lb.disp(hair_disp(1.0, seed=3, freq=16, amp=0.032), band=0.12)
    K.sdf_piece('body', lb, voxel=0.035, tris=1000, mat='belly', bone='chest', ao=0.4)
    # flowing locks over the mass edges
    rng = np.random.default_rng(4)
    for j in range(9):
        a = (j / 8 - 0.5) * 2
        x0 = a * 0.45
        lk = []
        for i, (y, dx) in enumerate(zip(ys, drift)):
            wob = 0.05 * np.sin(i * 1.4 + j * 0.9)
            lk.append(pts[i] + A(x0 * (1 + 0.15 * i) + wob + dx * 0.3 * a, 0, -0.06 - 0.03 * abs(a)))
        lk[0] = H((x0 * 0.9, -0.1, -0.42))
        lk.append(lk[-1] + A(-0.1 - 0.18 * a + rng.uniform(-0.05, 0.05), -0.45 - 0.3 * (1 - abs(a)), -0.04))
        u = np.linspace(0, 1, len(lk))
        K.horn('body', lk, 0.16 * np.interp(u, [0, 0.2, 0.7, 1], [0.6, 1.0, 0.8, 0.0]), mat='belly', sides=5, per=2, flat=0.45, up=A(0, 0, 1), bone='chest', shade=(0.65, 1.0))
    # a lock falling over the right shoulder in front
    fl = []
    for P0 in (A(0.42, 8.8, -0.05), A(0.62, 8.35, 0.2), A(0.66, 7.8, 0.45), A(0.6, 7.2, 0.55), A(0.56, 6.75, 0.5)):
        p, n = K.snap(P0)
        fl.append(p + n * 0.1)
    fl[0] = H((0.4, -0.25, -0.05))
    for k, o in enumerate((-0.08, 0.0, 0.08)):
        C = [q + A(o * (1 + 0.6 * i) + 0.04 * np.sin(i * 1.7 + k * 2), 0.12 * k * (i == 4), 0) for i, q in enumerate(fl)]
        K.horn('body', C, np.array([0.07, 0.085, 0.075, 0.05, 0.0]), mat='belly', sides=5, per=3, flat=0.55, up=A(0, 0, 1), bone='chest', shade=(0.7, 1.0))

    # ---- halo (base) + shoulder brooches + sash knot and tails
    halo_c = H((0, 0.98, -0.2))
    halo_ax = Rh @ nrm(A(0, 1, -0.35))
    V, T = torus(halo_c, halo_ax, 0.66 * hs, 0.045, nu=40, nv=6)
    K.piece('body', V, T, 'accent', bone='head')
    for sd, s in (('L', -1), ('R', 1)):
        p, n = K.snap(J['sh' + sd] + A(-s * 0.18, 0.5, 0.15))
        V, T = torus(p + n * 0.03, n, 0.1, 0.035, nu=14, nv=5)
        K.piece('body', V, T, 'gold', bone='chest')
        K.ball('body', p + n * 0.05, A(0.07, 0.07, 0.07), mat='accent', bone='chest')
    kp, kn_ = K.snap(sash_c + A(-0.35, 0, 1.2))
    K.ball('body', kp + kn_ * 0.06, A(0.16, 0.12, 0.1), mat='belly', bone='pelvis', seg=12, rings=8)
    for k, (dx, ln) in enumerate(((-0.12, 2.4), (0.1, 2.0))):
        C = []
        for t_ in np.linspace(0, 1, 9):
            q = kp + A(dx * t_ * 2 - 0.05 * np.sin(t_ * 5 + k), -ln * t_, 0.3 * t_)
            p, n = K.snap(q)
            C.append(p + n * (0.05 + 0.04 * t_))
        C = np.array(C)
        W = np.tile(A(0.09, 0, 0), (len(C), 1)) * (1 + 0.4 * np.linspace(0, 1, len(C)))[:, None]
        ribbon(K, 'body', C, W, mat='belly', bone='pelvis', th=0.02, cup=0.2, notch=1.0)

    # ---- great wings (base)
    WP = {}
    for sd, s in (('L', -1), ('R', 1)):
        P = [A(s * 0.36, 7.9, -0.62), A(s * 1.75, 8.95, -1.25), A(s * 3.05, 10.15, -1.65), A(s * 3.95, 10.95, -1.95)]
        WP[sd] = P
        wing(K, 'body', P, [f'wing{sd}1', f'wing{sd}2', f'wing{sd}3'])

    # ---- skeleton + mounts
    fig.rig(K)
    K.bone('jaw', H((0, -0.3, 0.05)), H((0, -0.55, 0.3)), 'head', 0.08)
    K.bone('gown', J['pelvis'] + A(0, -0.1, 0), sk.c + sk.R @ A(0, -3.7, 0), 'pelvis', 1.6)   # keeps the long gown riding the pelvis
    for sd in 'LR':
        K.chain(f'wing{sd}', WP[sd], 'chest', r=(0.16, 0.1))
    hc = fig.hand_center('R')
    tail = sk.c + sk.R @ A(0, -sk.h + 0.1, -sk.r1 * sk.zs)
    K.mount('head', H((0, 0, 0)), 'head')
    K.mount('jaw', H((0, -0.5, 0.3)), 'head')
    K.mount('mouth', H((0, -0.36, 0.52)), 'head')
    K.mount('crown', halo_c, 'head')
    cp, _ = K.snap(J['chest'] + A(0, 0.1, 1.5))
    K.mount('chest', cp, 'chest')
    K.mount('back', A(0, 8.0, -0.85), 'chest')
    K.mount('tailTip', tail, 'pelvis')
    for sd in 'LR':
        K.mount('hand' + sd, fig.hand_center(sd), 'hand' + sd)
        K.mount('shoulder' + sd, J['sh' + sd] + up * 0.35, 'chest')

    # ---- POW: her signature radiant longsword in the right fist, growing in grandeur every tier
    # (slim longsword -> winged-guard longsword -> greatsword with fuller + light flares -> great flaming sword)
    Rr = fig.hand_frame('R')
    ga = Rr[:, 0] if Rr[1, 0] > 0 else -Rr[:, 0]          # fist grip axis (finger-curl axis), pointing up
    ga = nrm(ga + A(0.2, 0, 0.08) + fwd * 0.12 + side * 0.32)   # raised up, a touch forward and out, clear of the head
    gc = J['wrR'] + Rr @ A(0, 0.24, -0.07)                 # centre of the closed fist grip hole
    cv = nrm(A(0.5, 0.1, 0.87))                           # default 3/4 camera direction: blade faces it
    fnb = nrm(cv - ga * (cv @ ga))                          # blade face normal
    wvb = nrm(np.cross(ga, fnb))                            # blade width (edge-to-edge) direction
    hb_ = 'handR'

    def hilt(p, grip_lo, guard_o, pommel_r, guard_w, guard_r, gem_r):
        """gold grip + crossguard + pommel; returns the guard centre."""
        K.horn(p, [gc - ga * grip_lo, gc, gc + ga * guard_o], [0.062, 0.066, 0.062], mat='gold', sides=8, bone=hb_, shade=(0.7, 0.95))
        for o in np.linspace(-grip_lo + 0.12, guard_o - 0.1, 3):
            V, T = torus(gc + ga * o, ga, 0.072, 0.018, nu=10, nv=4)
            K.piece(p, V, T, 'gold', bone=hb_)
        pm = gc - ga * (grip_lo + pommel_r * 0.8)
        K.ball(p, pm, A(pommel_r, pommel_r * 1.15, pommel_r), mat='gold', bone=hb_, seg=10, rings=7)
        K.ball(p, pm + fnb * pommel_r * 0.82, pommel_r * 0.42, mat='accent', bone=hb_, seg=8, rings=5)
        g = gc + ga * (guard_o + guard_r * 0.6)
        cg = [g - wvb * guard_w + ga * guard_w * 0.22, g - wvb * guard_w * 0.55 + ga * 0.02, g, g + wvb * guard_w * 0.55 + ga * 0.02, g + wvb * guard_w + ga * guard_w * 0.22]
        K.horn(p, cg, np.array([0.35, 0.75, 1.0, 0.75, 0.35]) * guard_r, mat='gold', sides=6, flat=0.7, up=fnb, bone=hb_)
        for s in (-1, 1):
            K.ball(p, cg[2 + 2 * s] + (cg[2 + 2 * s] - cg[2 + s]) * 0.12, guard_r * 0.62, mat='gold', bone=hb_, seg=8, rings=5)
        oball(K, p, g, A(guard_r * 1.7, guard_r * 1.5, guard_r * 1.1), np.stack([wvb, ga, fnb], 1), mat='gold', bone=hb_, seg=10, rings=6)
        for s in (-1, 1):
            K.ball(p, g + fnb * s * guard_r * 0.95, gem_r, mat='accent', bone=hb_, seg=10, rings=6)
        return g

    def wings_guard(p, g, n, L, w, spread=0.32, rise=0.3):
        """small gilded feather wings sweeping up-and-out from the crossguard, broad faces to the camera."""
        for s in (-1, 1):
            for i in range(n):
                f = i / max(n - 1, 1)
                feather(K, p, g + wvb * s * (0.12 + 0.05 * i) - ga * 0.03, nrm(wvb * s * (1.0 - spread * f) + ga * (rise + 0.75 * f)), fnb,
                        L * (1 - 0.22 * f), w, hb_, mat='gilt', asym=0.25, cup=0.1, bend=0.05, nu=5, nv=3, shade=(0.85, 1.0))

    def star(p, c, s, rot=0.0):
        """4-point light flare in the blade plane (two crossed flat diamonds + a small diagonal pair)."""
        a, b = np.cos(rot) * ga + np.sin(rot) * wvb, -np.sin(rot) * ga + np.cos(rot) * wvb
        for d, k in ((a, 1.0), (b, 0.8), (nrm(a + b), 0.42), (nrm(a - b), 0.42)):
            K.horn(p, [c - d * s * k, c, c + d * s * k], [0.0, s * 0.11, 0.0], mat='accent', sides=4, per=1, flat=0.25, up=fnb, bone=hb_)

    def ridge(p, b0, L, t, r, u1=0.8):
        """gilded centre ridge / fuller line down both faces of the blade."""
        for s in (-1, 1):
            K.horn(p, [b0 + ga * 0.05 + fnb * s * t, b0 + ga * L * 0.45 + fnb * s * t * 0.95, b0 + ga * L * (u1 - 0.08) + fnb * s * t * 0.85,
                       b0 + ga * L * u1 + fnb * s * t * 0.6], [r, r * 0.92, r * 0.8, 0.0], mat='gold', sides=4, per=2, flat=0.3, up=fnb, bone=hb_)

    # t0: a slim, elegant longsword
    p = 'pow_t0'
    g = hilt(p, 0.36, 0.3, 0.1, 0.5, 0.075, 0.045)
    b0 = g + ga * 0.05
    L0 = 3.7
    sword_blade(K, p, b0, ga, wvb, fnb, L0, [(0.0, 0.125, 0.032), (0.08, 0.145, 0.033), (0.55, 0.135, 0.031), (0.82, 0.112, 0.026), (0.94, 0.06, 0.017), (1.0, 0.0, 0.0)], bone=hb_)
    ridge(p, b0, L0, 0.031, 0.016)

    # t1: longer blade with a winged crossguard
    p = 'pow_t1'
    g = hilt(p, 0.38, 0.3, 0.11, 0.42, 0.085, 0.055)
    wings_guard(p, g, 3, 0.82, 0.12)
    b0 = g + ga * 0.06
    L1 = 4.3
    sword_blade(K, p, b0, ga, wvb, fnb, L1, [(0.0, 0.13, 0.034), (0.07, 0.155, 0.035), (0.55, 0.14, 0.032), (0.83, 0.115, 0.027), (0.94, 0.06, 0.017), (1.0, 0.0, 0.0)], bone=hb_)
    ridge(p, b0, L1, 0.036, 0.02)

    # t2: a greatsword (hand-and-a-half grip) with a gilded fuller and light flares
    p = 'pow_t2'
    g = hilt(p, 0.62, 0.3, 0.13, 0.62, 0.1, 0.07)
    wings_guard(p, g, 4, 0.85, 0.12)
    b0 = g + ga * 0.07
    L2 = 4.9
    sword_blade(K, p, b0, ga, wvb, fnb, L2, [(0.0, 0.2, 0.05), (0.06, 0.24, 0.052), (0.5, 0.225, 0.048), (0.8, 0.185, 0.04), (0.93, 0.1, 0.026), (1.0, 0.0, 0.0)], bone=hb_)
    ridge(p, b0, L2, 0.05, 0.045, u1=0.78)
    star(p, g + fnb * 0.12, 0.55)
    star(p, b0 + ga * L2 * 0.97 + fnb * 0.03, 0.32, rot=0.4)
    for s, u in ((1, 0.62), (-1, 0.36)):
        star(p, b0 + ga * L2 * u + wvb * s * 0.24 + fnb * 0.02, 0.22, rot=0.3)

    # t3: the great radiant flaming sword: broad blade, light licking along both edges, sun-ring guard
    p = 'pow_t3'
    g = hilt(p, 0.66, 0.3, 0.15, 0.72, 0.11, 0.08)
    wings_guard(p, g, 5, 1.05, 0.13, spread=0.4, rise=0.25)
    V, T = torus(g + fnb * 0.02, fnb, 0.38, 0.03, nu=24, nv=4)
    K.piece(p, V, T, 'accent', bone=hb_)
    b0 = g + ga * 0.08
    L3 = 5.4
    prof = [(0.0, 0.22, 0.055), (0.06, 0.27, 0.058), (0.5, 0.255, 0.054), (0.8, 0.21, 0.046), (0.93, 0.11, 0.028), (1.0, 0.0, 0.0)]
    sword_blade(K, p, b0, ga, wvb, fnb, L3, prof, bone=hb_)
    ridge(p, b0, L3, 0.056, 0.05, u1=0.78)
    # flame tongues: flat, in the blade plane, curling up off each edge, longer toward the tip
    for s in (-1, 1):
        for i in range(6):
            u = 0.12 + 0.13 * i + (0.06 if s < 0 else 0.0)
            if u > 0.86:
                continue
            hw = float(np.interp(u, [q[0] for q in prof], [q[1] for q in prof]))
            e = b0 + ga * L3 * u + wvb * s * (hw - 0.03)
            Lf = 0.32 + 0.3 * u + 0.08 * np.sin(i * 2.3 + s)
            K.horn(p, [e, e + wvb * s * 0.13 + ga * Lf * 0.4, e + wvb * s * (0.06 + 0.04 * np.sin(i + s)) + ga * Lf * 0.78, e + wvb * s * 0.01 + ga * Lf],
                   [0.07, 0.06, 0.03, 0.0], mat='accent', sides=4, per=2, flat=0.3, up=fnb, bone=hb_)
    tp = b0 + ga * L3
    K.horn(p, [tp - ga * 0.35, tp + ga * 0.2 + wvb * 0.05, tp + ga * 0.55 - wvb * 0.03, tp + ga * 0.8], [0.09, 0.07, 0.035, 0.0], mat='accent', sides=4, per=2, flat=0.3, up=fnb, bone=hb_)
    star(p, g + fnb * 0.14, 0.75)
    star(p, tp + fnb * 0.03, 0.42, rot=0.4)
    for s, u in ((1, 0.55), (-1, 0.3)):
        star(p, b0 + ga * L3 * u + wvb * s * 0.27 + fnb * 0.03, 0.24, rot=0.3)

    # ---- ARMOR (additive): pauldrons + bracers, breastplate + tassets, winged helm + greaves
    for sd, s in (('L', -1), ('R', 1)):
        sh, el, wr = J['sh' + sd], J['el' + sd], J['wr' + sd]
        # pauldrons: three overlapping gilded feather plates fanning down over each shoulder
        p_, n_ = K.snap(sh + A(s * 0.12, 0.45, 0))
        for i, (dz, ln) in enumerate(((-0.2, 0.7), (0.2, 0.7), (0.0, 0.85))):
            feather(K, 'arm_t1', p_ + n_ * (0.05 + 0.04 * i) + A(-s * 0.08, 0.04, dz), nrm(A(s * 0.75, -1, dz * 0.8)), n_, ln, 0.27, 'upperarm' + sd,
                    mat='plate', asym=0.0, cup=0.45, bend=0.0, nu=5, nv=4, shade=(0.75, 1.0), tip=3.0)
        K.crystal('arm_t1', p_ + n_ * 0.2 + A(-s * 0.05, 0.02, 0), n_, 0.14, r=0.06, cluster=False, bone='upperarm' + sd)
        K.plate('arm_t1', el + (wr - el) * 0.55, (0.3, 0.3, 0.3), bone='forearm' + sd, lames=2, off=0.04, th=0.05, lame_axis=wr - el, tris=220)
        an, kn = J['an' + sd], J['kn' + sd]
        K.plate('arm_t3', an + (kn - an) * 0.1 + A(0, 0, 0.12), (0.26, 0.26, 0.24), bone='shin' + sd, lames=2, off=0.04, th=0.05, lame_axis=kn - an, tris=200)
    # breastplate: a smooth domed cuirass front (independent of the bust shape) with gold filigree + sun gem
    bc = J['chest'] + A(0, 0.02, 0.0)
    bpl = K.sdf()
    bpl.ell(bc, A(0.86, 0.78, 0.76), k=0)
    bpl.ell(bc, A(0.81, 0.73, 0.71), k=0.02, cut=True)
    bpl.box(bc + A(0, 0, -0.55), A(1.2, 1.2, 0.85), k=0.04, cut=True, rad=0)
    bpl.box(bc + A(0, 1.12, 0), A(1.2, 0.6, 1.2), k=0.04, cut=True, rad=0)
    bpl.box(bc + A(0, -1.0, 0), A(1.2, 0.6, 1.2), k=0.04, cut=True, rad=0)
    K.sdf_piece('arm_t2', bpl, voxel=0.03, tris=600, mat='plate', bone='chest', ao=0.2)
    bp_ = bc + A(0, 0.05, 0.765)
    bn_ = A(0, 0.05, 1)
    K.crystal('arm_t2', bp_, bn_, 0.18, r=0.1, cluster=False, bone='chest')
    for s in (-1, 1):
        for i in range(3):
            C = [bp_ + A(s * (0.1 + 0.02 * i), 0.0, 0.01), bp_ + A(s * (0.26 + 0.05 * i), 0.1 - 0.14 * i, -0.02 - 0.01 * i),
                 bp_ + A(s * (0.42 + 0.05 * i), 0.14 - 0.24 * i, -0.08 - 0.04 * i)]
            K.horn('arm_t2', [c + A(0, 0, 0.02) for c in C], [0.024, 0.018, 0.0], mat='gold', sides=4, bone='chest')
    # gilded girdle over the sash + hanging pteruges strips over the gown front
    gring, gn = [], []
    for a in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        p_, n_ = K.snap(sash_c + A(np.cos(a) * 0.7, 0, np.sin(a) * 0.6))
        gring.append(p_ + n_ * 0.07)
    V, T = closed_tube(gring, 0.06, nv=5)
    K.piece('arm_t2', V, T, 'plate', bone='pelvis')
    for k in range(7):
        a = np.radians(-54 + 18 * k)
        C = []
        for t_ in np.linspace(0, 1, 5):
            p_, n_ = K.snap(sash_c + A(np.sin(a) * 1.5, -0.12 - 1.25 * t_, np.cos(a) * 1.5))
            C.append(p_ + n_ * 0.06)
        K.horn('arm_t2', C, [0.12, 0.12, 0.12, 0.11, 0.07], mat='plate', sides=4, per=2, flat=0.22, up=A(0, 0, 1) * 0 + nrm(C[2] - sash_c) * A(1, 0, 1), bone='pelvis', shade=(0.8, 1.0))
    hm = K.sdf()
    hm.ell(H((0, 0.2, -0.08)), A(0.52, 0.54, 0.6) * hs, rot=Rh, k=0)
    hm.ell(H((0, 0.2, -0.08)), A(0.48, 0.5, 0.56) * hs, rot=Rh, k=0.02, cut=True)
    hm.box(H((0, -0.35, 0.3)), A(0.8, 0.45, 0.8) * hs, rot=Rh, k=0.03, cut=True, rad=0)
    hm.box(H((0, -0.1, 0.65)), A(0.8, 0.5, 0.35) * hs, rot=Rh @ S.rot_xyz((-25, 0, 0)), k=0.03, cut=True, rad=0)
    hm.tube([H((0, 0.55, 0.38)), H((0, 0.74, 0.0)), H((0, 0.6, -0.45))], np.array([0.04, 0.07, 0.04]) * hs, k=0.04)
    K.sdf_piece('arm_t3', hm, voxel=0.025, tris=700, mat='plate', bone='head', ao=0.3)
    for s in (-1, 1):
        for i in range(4):
            feather(K, 'arm_t3', H((s * 0.5, 0.25, -0.05)), Rh @ nrm(A(s * (0.8 - 0.12 * i), 0.4 + 0.35 * i, -0.7)), Rh @ A(s, 0, 0.3), 0.75 - 0.1 * i, 0.1, 'head', mat='plate', nu=5, nv=3)

    # ---- ENERGY (additive): glowing robe trim -> light feathers & motes -> radiant sun-disc halo
    th_ = np.linspace(0, 2 * np.pi, 56, endpoint=False)
    hem_pts = []
    for th in th_:
        hem = sk.h - sk.scal * (0.5 - 0.5 * np.sin(sk.n * th + sk.ph)) - 0.06
        R_ = sk.r0 + (sk.r1 - sk.r0) * (hem / sk.h) ** sk.pw + sk.amp * (hem / sk.h) * np.sin(sk.n * th + sk.ph) + 0.03
        hem_pts.append(sk.c + sk.R @ A(R_ * np.sin(th), -hem, R_ * np.cos(th) * sk.zs))
    V, T = closed_tube(hem_pts, 0.04, nv=5)
    K.piece('enr_t1', V, T, 'accent', bone='pelvis')
    ring = []
    for a in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        p_, n_ = K.snap(sash_c + A(np.cos(a) * 0.7, 0, np.sin(a) * 0.6))
        ring.append(p_ + n_ * 0.07)
    V, T = closed_tube(ring, 0.035, nv=5)
    K.piece('enr_t1', V, T, 'accent', bone='spine')
    for s in (-1, 1):
        sh, el = J['sh' + (('L', 'R')[s > 0])], J['el' + (('L', 'R')[s > 0])]
        c0 = sh + (el - sh) * 0.62
        V, T = torus(c0, el - sh, 0.24, 0.025, nu=16, nv=4)
        K.piece('enr_t1', V, T, 'accent', bone='upperarm' + ('L', 'R')[s > 0])
    rng = np.random.default_rng(11)
    for i in range(7):
        a = -0.3 + np.pi * 1.6 * i / 8 + rng.uniform(-0.15, 0.15)
        rr = rng.uniform(2.3, 3.0)
        c0 = A(np.sin(a) * rr, rng.uniform(1.8, 8.8), np.cos(a) * rr * 0.75 + 0.3)
        d = nrm(A(rng.uniform(-1, 1), rng.uniform(-0.4, 0.8), rng.uniform(-0.5, 0.5)))
        feather(K, 'enr_t2', c0, d, nrm(A(rng.uniform(-0.3, 0.3), 0.2, 1)), rng.uniform(0.55, 0.85), 0.12, 'pelvis', mat='accent', nu=5, nv=3)
    for i in range(14):
        a = rng.uniform(-1.6, 1.6)
        rr = rng.uniform(1.8, 3.2)
        K.ball('enr_t2', A(np.sin(a) * rr, rng.uniform(1.2, 9.5), np.cos(a) * rr * 0.8 + 0.2), rng.uniform(0.05, 0.1), mat='accent', bone='pelvis', seg=8, rings=5)
    sd_c = H((0, 0.3, -0.78))
    sd_ax = Rh @ nrm(A(0, 0.12, 1))
    V, T = torus(sd_c, sd_ax, 1.3, 0.05, nu=40, nv=4)
    K.piece('enr_t3', V, T, 'accent', bone='head')
    V, T = torus(sd_c, sd_ax, 1.42, 0.03, nu=40, nv=3)
    K.piece('enr_t3', V, T, 'gold', bone='head')
    bx, by, bz = geom.basis(sd_ax)
    for i in range(24):
        a = 2 * np.pi * i / 24
        rd = np.cos(a) * bx + np.sin(a) * bz
        L = 0.85 if i % 2 == 0 else 0.45
        if abs(np.sin(a)) > 0.92 and (rd @ A(0, -1, 0)) > 0.5:
            continue
        b0 = sd_c + rd * 1.45
        K.horn('enr_t3', [b0, b0 + rd * L * 0.5, b0 + rd * L], [0.07 if i % 2 == 0 else 0.05, 0.04, 0.0], mat='accent', sides=3, flat=0.4, up=sd_ax, bone='head')

    # ---- SPEED (additive): ribbon streamers -> ankle wings -> second (hip) wing pair
    for k, s in enumerate((-1, 1)):
        p_, n_ = K.snap(sash_c + A(s * 0.25, 0, -1.2))
        k_ = np.linspace(0, 1, 14)
        C = (p_ + n_ * 0.06)[None] + np.stack([s * 0.4 * k_ - 0.55 * k_ ** 1.3 + 0.12 * np.sin(k_ * 6 + k), -2.6 * k_ + 0.6 * k_ ** 2, -1.9 * k_ - 0.2 * np.sin(k_ * 5 + s)], 1)
        W = np.stack([0.15 + 0.08 * k_, 0.0 * k_, 0.0 * k_ - 0.02], 1)
        ribbon(K, 'spd_t1', C, W, mat='belly', bone='pelvis', th=0.02, cup=0.08, notch=1.1)
    for sd, s in (('L', -1), ('R', 1)):
        sh, el = J['sh' + sd], J['el' + sd]
        c0 = sh + (el - sh) * 0.62 + A(s * 0.2, 0, -0.1)
        k_ = np.linspace(0, 1, 12)
        C = c0[None] + np.stack([s * (0.2 + 0.5 * k_) - 0.4 * k_ ** 2, -1.8 * k_ + 0.12 * np.sin(k_ * 7), -1.4 * k_ - 0.25 * k_ ** 2], 1)
        W = np.stack([0.0 * k_, 0.0 * k_, 0.07 + 0.04 * k_], 1)
        ribbon(K, 'spd_t1', C, W, mat='belly', bone='upperarm' + sd, th=0.016, cup=0.2, notch=1.1)
    for sd, s in (('L', -1), ('R', 1)):
        an, kn, toe = J['an' + sd], J['kn' + sd], J['toe' + sd]
        v = nrm(an - kn)
        base = an - v * 0.12 + A(s * 0.17, 0, -0.05)
        for i in range(4):
            feather(K, 'spd_t2', base, nrm(A(s * (0.5 + 0.1 * i), 0.2 + 0.35 * i, -1.0)), A(s, 0, 0), 0.75 - 0.1 * i, 0.12, 'foot' + sd, mat='feather', nu=5, nv=3)
    for sd, s in (('L', -1), ('R', 1)):
        P = [A(s * 0.35, 6.2, -0.62), A(s * 1.1, 5.9, -1.15), A(s * 1.8, 5.75, -1.55), A(s * 2.25, 5.7, -1.75)]
        wing(K, 'spd_t3', P, ['pelvis', 'pelvis', 'pelvis'], scale=0.4, n_sec=6, n_pri=5, coverts=1, back=False, g=(0, -1, -0.45), arm_r=0.3, lift=-0.2, cov_n=0.6, fnu=4)

    # ---- FOCUS (additive): tiara + forehead gem -> extra halo ring -> third ring + crown of light
    ti = K.sdf()
    ti.ell(H((0, 0.36, 0.02)), A(0.5, 0.2, 0.57) * hs, rot=Rh, k=0)
    ti.ell(H((0, 0.36, 0.02)), A(0.465, 0.3, 0.535) * hs, rot=Rh, k=0.02, cut=True)
    ti.box(H((0, 0.56, 0.0)), A(0.8, 0.13, 0.8) * hs, rot=Rh, k=0.02, cut=True, rad=0)
    ti.box(H((0, 0.2, 0.0)), A(0.8, 0.11, 0.8) * hs, rot=Rh, k=0.02, cut=True, rad=0)
    ti.box(H((0, 0.36, -0.42)), A(0.8, 0.4, 0.4) * hs, rot=Rh, k=0.02, cut=True, rad=0)
    K.sdf_piece('fcs_t1', ti, voxel=0.022, tris=420, mat='gold', bone='head')
    gem_c = H((0, 0.36, 0.565))
    K.crystal('fcs_t1', gem_c, Rh @ A(0, 0.15, 1), 0.12, r=0.08, cluster=False, bone='head')
    K.horn('fcs_t1', [H((0, 0.4, 0.57)), H((0, 0.55, 0.56)), H((0, 0.68, 0.53))], np.array([0.05, 0.035, 0.0]) * hs, mat='gold', sides=4, flat=0.4, bone='head')
    ax2 = Rh @ nrm(A(0.45, 1, -0.2))
    V, T = torus(halo_c + Rh @ A(0, 0.05, 0), ax2, 0.82 * hs, 0.035, nu=40, nv=5)
    K.piece('fcs_t2', V, T, 'accent', bone='head')
    hx, hy, hz = geom.basis(halo_ax)
    for i in range(8):
        a = 2 * np.pi * i / 8
        p_ = halo_c + (np.cos(a) * hx + np.sin(a) * hz) * 0.66 * hs
        K.crystal('fcs_t3', p_, nrm(halo_ax * 1.0 + (p_ - halo_c) * 0.3), 0.26 if i % 2 == 0 else 0.16, r=0.05, cluster=False, bone='head')
