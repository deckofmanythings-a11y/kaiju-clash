"""Carapace: a walking fortress turtle. Domed shell with raised scutes and a serrated rim, four stout
elephantine legs, hooked beak, wrinkly neck, short spiked tail, magma vents. Game coords, faces +Z."""
import numpy as np

from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep, voronoi
from .hydra import Xf, eye, frame, rotx, unit

DETAIL = 'scales'
UV_SCALE = 0.7
CAM = dict(az=35, el=15)

DC = np.array([0.0, 2.95, -0.4])      # dome centre
DR = np.array([4.1, 3.7, 4.8])        # dome radii
RIM_Y = 2.85
HS = 1.15                             # head scale


def dome_pt(theta, phi, out=0.0):
    """Point on the dome ellipsoid. theta: azimuth from +Z (deg, +X positive), phi: elevation (deg)."""
    t, p = np.radians(theta), np.radians(phi)
    d = np.array([np.cos(p) * np.sin(t), np.sin(p), np.cos(p) * np.cos(t)])
    n = unit(d / DR)
    return DC + DR * d + n * out, n


def scute_centres():
    C = []
    for th, ph in ((0, 52), (0, 70), (0, 90), (180, 70), (180, 52)):              # vertebrals
        C.append(dome_pt(th, ph)[0])
    for sd in (-1, 1):
        for th in (38, 72, 108, 142):                                             # costals
            C.append(dome_pt(sd * th, 36)[0])
    for th in np.arange(0, 360, 15):                                              # marginals
        C.append(dome_pt(th + 7.5, 9)[0])
    return np.array(C)


SC = scute_centres()
NV, NC = 5, 8


class Cell:
    """SDF-like region of scute i (Voronoi cell of the scute centres), shrunk by `margin`."""

    def __init__(self, i, margin=0.08, ext=1.7):
        self.i, self.margin = i, margin
        self.lo, self.hi = SC[i] - ext, SC[i] + ext

    def dist(self, reg, pad=0.3):
        X, Y, Z = np.broadcast_arrays(reg.x, reg.y, reg.z)
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
        out = np.empty(len(P), np.float32)
        C = SC.astype(np.float32)
        for a in range(0, len(P), 60000):
            d = np.linalg.norm(P[a:a + 60000, None, :] - C[None], axis=2)
            di = d[:, self.i].copy()
            d[:, self.i] = 1e9
            out[a:a + 60000] = (di - d.min(1)) * 0.5 + self.margin
        return out.reshape(X.shape)


def _cone(base, n, h, r_top, r_bot):
    """Volcanic vent chimney: truncated cone with a crater."""
    P = S.SDF()
    n = unit(n)
    P.tube([base, base + n * h], [r_bot, r_top], k=0)
    P.ell(base + n * (h + 0.05), (r_top * 0.7,) * 3, cut=True, k=0.05)
    P.disp(lambda x, y, z: -0.06 * fbm(x, y, z, 2, freq=2.5, seed=31, ridged=True), band=0.12)
    return P


def build(K):
    K.mat('hide', color=np.array([0.45, 0.42, 0.36]), rough=0.75, vcol=True, detail=True, detail_kind='skin')
    K.mat('body', detail_kind='chitin', detail_strength=0.7, rough=0.5)
    K.mat('beak', color='#4a4036', rough=0.35, vcol=True, detail=False)

    # ------------------------------------------------------------ shell (carapace + rim + plastron)
    dome = K.sdf()
    domeprim = S.Inter(S.Ell(DC, DR), S.Box((0, RIM_Y + 6, -0.4), (6, 6, 7)), 0.25)
    dome.add(domeprim, 0.0)
    ang = np.radians(np.arange(0, 360, 7.5))
    rim = np.stack([np.sin(ang) * (DR[0] + 0.3), np.full_like(ang, RIM_Y + 0.05), DC[2] + np.cos(ang) * (DR[2] + 0.3)], 1)
    back = np.clip(-np.cos(ang), 0, 1)
    rim_r = 0.42 + 0.14 * (np.arange(len(ang)) % 2) + 0.1 * back      # scalloped, heavier at the back
    rim[:, 1] -= 0.25 * back ** 2
    rimtube = S.Tube(np.vstack([rim, rim[:2]]), np.concatenate([rim_r, rim_r[:2]]), s=(1, 0.5, 1))
    dome.add(rimtube, 0.3)

    B = K.sdf()
    B.add(domeprim, 0.0)
    B.add(rimtube, 0.3)
    # serrated rim: a pointed tooth on every marginal scute, long hooks over the tail
    teeth = K.sdf()
    for th in np.arange(0, 360, 15) + 7.5:
        p, n = dome_pt(th, 0, 0.3)
        d = unit([p[0], 0, p[2] - DC[2]])
        bk = max(0.0, -np.cos(np.radians(th)))
        ln = 0.5 + 0.45 * bk
        B.tube([p + [0, -0.05 - 0.2 * bk, 0], p + d * ln + [0, -0.15 - 0.3 * bk, 0]], [0.32 + 0.06 * bk, 0.05], k=0.12)
        teeth.tube([p + [0, -0.05 - 0.2 * bk, 0], p + d * ln + [0, -0.15 - 0.3 * bk, 0]], [0.32 + 0.06 * bk, 0.05], k=0)
    # soft body + plastron + bridges
    B.ell((0, 2.75, -0.3), (3.4, 1.2, 4.1), k=0.3)
    B.ell((0, 2.3, -0.3), (3.1, 0.72, 4.0), k=0.3)
    B.ell((3.3, 2.6, -0.3), (0.75, 0.55, 2.3), k=0.35, mirror=True)
    # magma vents (base): two volcanic cones on the flanks of the shell
    vents = []
    for sd in (-1, 1):
        p, n = dome_pt(sd * 100, 34)
        n = unit(n + [sd * 0.35, 0, 0.25])
        B.tube([p - n * 0.2, p + n * 0.5], [0.85, 0.5], k=0.35)
        vents.append((p + n * 1.02, n))

    # ------------------------------------------------------------ legs (elephantine columns)
    LEGS = {}
    for sd, nm in ((-1, 'L'), (1, 'R')):
        m = np.array([sd, 1, 1])
        front = [m * (2.3, 2.2, 2.1), m * (3.3, 1.75, 2.65), m * (3.45, 0.95, 2.85), m * (3.45, 0.42, 3.0)]
        rear = [m * (2.2, 2.15, -2.8), m * (3.25, 1.7, -3.25), m * (3.4, 0.95, -3.15), m * (3.4, 0.42, -3.05)]
        LEGS[nm] = dict(front=front, rear=rear)
        for P_, r0 in ((front, 1.28), (rear, 1.36)):
            B.tube(P_[:2], [r0 * 0.9, r0 * 0.9], k=0.4)
            B.ell(P_[1], (r0 * 0.88, r0 * 0.85, r0 * 0.9), k=0.25)                  # elbow / knee mass
            B.tube(P_[1:3], [r0 * 0.82, r0 * 0.78], k=0.25)
            B.ell(P_[3] + [0, -0.02, 0.08], (r0 * 0.95, 0.42, r0 * 1.0), k=0.3)   # foot pad
            for a in np.radians([-50, -25, 0, 25, 50]):                          # toe bulges
                d = np.array([np.sin(a), 0, np.cos(a)])
                B.ell(P_[3] + d * r0 * 0.82 + [0, -0.08, 0], (0.26, 0.24, 0.28), k=0.12)

    # ------------------------------------------------------------ neck + head
    NECK = [(0, 3.0, 3.4), (0, 3.55, 4.55), (0, 4.1, 5.45)]
    B.tube(NECK, [1.15, 1.0, 0.88], k=0.35)
    for sd in (-1, 1):   # neck skin folds / cords
        B.tube([(sd * 0.6, 2.9, 3.7), (sd * 0.55, 3.5, 4.7), (sd * 0.45, 3.95, 5.5)], [0.45, 0.42, 0.33], k=0.25)
    HO, HF = np.array([0.0, 4.32, 5.7]), (0, -0.1, 1)
    R = frame(HF)
    OPEN = 14
    hinge = np.array([0, -0.25, 0.2])
    Rj = R @ rotx(OPEN)
    oj = HO + HS * (R @ hinge - Rj @ hinge)
    Rm = R @ rotx(OPEN / 2)
    om = HO + HS * (R @ hinge - Rm @ hinge)

    def W(p):
        return HO + HS * (np.asarray(p, float) @ R.T)

    def WJ(p):
        return oj + HS * (np.asarray(p, float) @ Rj.T)

    def Dd(v):
        return np.asarray(v, float) @ R.T

    def h(prim):
        return Xf(prim, HO, R, HS)

    def j(prim):
        return Xf(prim, oj, Rj, HS)

    def mid(prim):
        return Xf(prim, om, Rm, HS)

    B.add(h(S.Ell((0, 0.2, 0.6), (0.9, 0.74, 1.08))), 0.3)
    B.add(h(S.Tube([(0, 0.12, 1.15), (0, 0.02, 1.75), (0, -0.2, 2.15)], [0.64, 0.46, 0.22], s=(1.05, 0.9, 1))), 0.25)
    B.add(h(S.Tube([(0, -0.05, 2.0), (0, -0.3, 2.17), (0, -0.55, 2.12)], [0.2, 0.15, 0.04], s=(1.1, 1, 0.9))), 0.08)   # beak hook
    for sd in (-1, 1):
        B.add(h(S.Ell((sd * 0.62, -0.08, 0.45), (0.44, 0.52, 0.62))), 0.25)               # cheeks
        B.add(h(S.Tube([(sd * 0.36, 0.7, 0.5), (sd * 0.6, 0.62, 1.05), (sd * 0.44, 0.42, 1.5)], [0.17, 0.22, 0.09])), 0.12)
        B.add(h(S.Ell((sd * 0.82, 0.25, 0.05), (0.2, 0.2, 0.25))), 0.15)                  # tympanum bump
    B.add(j(S.Tube([(0, -0.5, 0.25), (0, -0.62, 1.15), (0, -0.56, 1.85)], [0.62, 0.46, 0.2], s=(1.15, 0.6, 1))), 0.2)
    B.add(j(S.Ell((0, -0.55, 1.9), (0.2, 0.16, 0.16))), 0.1)
    B.add(mid(S.Tube([(0, -0.45, -0.4), (0, -0.62, 0.3)], [0.75, 0.55])), 0.35)        # throat
    mouth = mid(S.Ell((0, -0.4, 1.3), (0.7, 0.13 + 0.45 * np.sin(np.radians(OPEN)), 0.95), rot=(4, 0, 0)))
    B.cut(mouth, 0.05)
    for sd in (-1, 1):
        B.cut(h(S.Ell((sd * 0.64, 0.32, 1.0), (0.19, 0.17, 0.24))), 0.07)
        B.cut(h(S.Ell((sd * 0.12, 0.12, 2.08), (0.05, 0.05, 0.07))), 0.03)
    B.add(j(S.Tube([(0, -0.48, 0.4), (0, -0.46, 1.1)], [0.22, 0.14], s=(1.3, 0.45, 1))), 0.06)   # tongue

    # ------------------------------------------------------------ tail
    TAIL = [(0, 2.6, -4.4), (0, 2.05, -5.5), (0.25, 1.4, -6.4), (0.7, 0.85, -7.05)]
    B.tube(TAIL, [0.85, 0.62, 0.38, 0.1], k=0.35)

    # ------------------------------------------------------------ masks
    plastron = K.sdf()
    plastron.ell((0, 1.85, -0.3), (3.0, 0.75, 3.95), k=0)
    soft = K.sdf()     # everything that is not shell: legs, neck, head, tail, body under the rim
    soft.ell((0, 2.6, -0.3), (3.7, 0.4, 4.4), k=0)
    for nm in 'LR':
        for P_ in LEGS[nm].values():
            soft.tube(P_ + [np.array([0, -0.6, 0])], [1.85, 1.6, 1.5, 1.6, 1.6], k=0)
    soft.tube(NECK + [tuple(W((0, 0, 1.0)))], [1.5, 1.4, 1.4, 1.5], k=0)
    soft.add(h(S.Ell((0, 0, 1.0), (1.6, 1.5, 2.0))), 0)
    soft.tube(TAIL, [1.0, 0.8, 0.6, 0.4], k=0)
    under = K.sdf()    # pale skin: throat + chin
    under.tube([(0, 2.3, 3.5), (0, 2.8, 4.7), W((0, -0.75, -0.2)), WJ((0, -0.8, 1.0))], [0.95, 0.85, 0.65, 0.48], k=0)
    beak = K.sdf()
    beak.add(h(S.Ell((0, -0.2, 2.05), (0.6, 0.45, 0.55), rot=(-20, 0, 0))), 0)
    beak.add(j(S.Ell((0, -0.6, 1.85), (0.5, 0.3, 0.42))), 0)

    def scutes(P):
        d = np.linalg.norm(P[:, None, :] - SC[None, :, :], axis=2)
        srt = np.sort(d, 1)
        return srt[:, 0], srt[:, 1] - srt[:, 0]

    def skin(x, y, z):
        P = np.stack([x, y, z], 1)
        dd = np.abs(dome.at(P))
        ws = (1 - smoothstep(0.1, 0.35, dd)) * smoothstep(2.45, 2.75, y)
        out = np.zeros(len(P), np.float32)
        i = ws > 0
        if i.any():
            f1, e = scutes(P[i])
            groove = 1 - smoothstep(0.0, 0.16, e)
            raise_ = smoothstep(0.12, 0.9, e)
            rings = 0.014 * np.sin(f1 * 22.0) * smoothstep(0.15, 0.4, e)
            pit = -0.025 * fbm(x[i], y[i], z[i], 2, freq=2.6, seed=21)
            out[i] = (0.12 * groove - 0.07 * raise_ + rings + pit) * ws[i]
        # soft hide: deep wrinkles + big leg scales
        wrink = 0.035 * np.sin(y * 9.0 + 1.6 * fbm(x, y, z, 2, freq=0.8, seed=2)) + 0.03 * fbm(x, y, z, 3, freq=2.2, seed=13, ridged=True)
        f1, f2, _ = voronoi(x, y, z, size=0.42, seed=5)
        legsc = 0.05 * smoothstep(0.0, 0.25, (f2 - f1))
        legw = smoothstep(2.3, 1.4, y)
        out = out + (1 - ws) * (wrink * (1 - legw) - legsc * legw + 0.02 * legw)
        return out
    B.disp(skin, band=0.18)

    vent_cut = [S.Ell(p, (0.42, 0.42, 0.42)) for p, n in vents]
    for v in vent_cut:
        B.cut(v, 0.06)

    def vent_fn(P):
        pp = S.Points(P)
        return np.min([v.dist(pp) - 0.1 for v in vent_cut], 0)

    def soft_fn(P):
        dd = dome.at(P)
        y = P[:, 1]
        shell = ((np.abs(dd) < 0.3) & (y > RIM_Y - 0.45)) | (teeth.at(P) < 0.12)
        return np.where(shell, 1.0, soft.at(P))

    K.skin(B, voxel=0.05, tris=17000, belly=plastron,
           regions=[('hide', soft_fn), ('belly', under), ('beak', beak), ('mouth', lambda P: mouth.at(P) - 0.03), ('accent', vent_fn)])

    # ------------------------------------------------------------ skeleton + mounts
    K.bone('pelvis', (0, 3.0, -3.6), (0, 3.1, -0.6), None, 3.6)
    K.bone('chest', (0, 3.1, -0.6), (0, 3.2, 2.6), 'pelvis', 3.6)
    for sd, nm in ((-1, 'L'), (1, 'R')):
        f, r = LEGS[nm]['front'], LEGS[nm]['rear']
        K.bone('upperarm' + nm, f[0] + [0, -0.3, 0], f[1], 'chest', 1.0)
        K.bone('forearm' + nm, f[1], f[3], 'upperarm' + nm, 1.0)
        K.bone('hand' + nm, f[3], f[3] + [0, -0.1, 1.1], 'forearm' + nm, 0.6)
        K.bone('thigh' + nm, r[0] + [0, -0.3, 0], r[1], 'pelvis', 1.05)
        K.bone('shin' + nm, r[1], r[3], 'thigh' + nm, 1.05)
        K.bone('foot' + nm, r[3], r[3] + [0, -0.1, 1.1], 'shin' + nm, 0.6)
        K.mount('hand' + nm, f[3] + [0, 0.1, 0.8], 'hand' + nm)
        K.mount('shoulder' + nm, dome_pt(sd * 48, 30)[0], 'chest')
    K.chain('neck', NECK + [tuple(W((0, 0, 0.1)))], 'chest', (1.05, 0.85))
    K.bone('head', W((0, 0, 0.1)), W((0, -0.1, 2.1)), 'neck3', 0.85 * HS)
    K.bone('jaw', WJ((0, -0.45, 0.3)), WJ((0, -0.55, 1.9)), 'head', 0.45 * HS)
    K.chain('tail', TAIL, 'pelvis', (0.75, 0.2))
    K.mount('head', W((0, 0.15, 0.5)), 'head')
    K.mount('jaw', WJ((0, -0.55, 0.6)), 'jaw')
    K.mount('mouth', W((0, -0.35, 2.2)), 'head')
    K.mount('crown', W((0, 0.95, 0.5)), 'head')
    K.mount('chest', (0, 2.45, 3.9), 'chest')
    K.mount('back', dome_pt(180, 62, 0.1)[0], 'pelvis')
    K.mount('tailTip', TAIL[-1], 'tail3')

    # ------------------------------------------------------------ face + base tail knobs
    for sd in (-1, 1):
        eye(K, 'body', W((sd * 0.6, 0.33, 1.02)), Dd((sd * 0.9, 0.1, 0.45)), 0.14 * HS, bone='head')
    TT = S.catmull(TAIL, 4)

    def tb(i):
        return 'tail%d' % min(3, 1 + i // 4)
    for i in range(1, len(TT) - 2, 2):
        p, n = K.snap(TT[i] + [0, 0.6, 0])
        K.spike('body', p, n + [0, 0, -0.4], 0.34 - i * 0.015, mat='plate', bone=tb(i))

    # ------------------------------------------------------------ Might: toenails + horns
    for t, scl in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        kk = 0.7 + 0.3 * scl
        for sd, nm in ((-1, 'L'), (1, 'R')):
            for key, bn, r0 in (('front', 'hand', 1.28), ('rear', 'foot', 1.36)):
                fp = LEGS[nm][key][3]
                for a in np.radians([-50, -25, 0, 25, 50]):
                    d = np.array([np.sin(a), 0, np.cos(a)])
                    base = fp + d * r0 * 0.98 + [0, -0.2, 0]
                    K.claw(part, base, d + [0, -0.15, 0], 0.34 * kk, 0.16 * (0.8 + 0.2 * scl), curl=(0, -1, 0), bone=bn + nm, flat=0.65)
        if t >= 1:
            hl = [0, 0.75, 1.1, 1.45][t]
            for sd in (-1, 1):
                K.horn(part, [W((sd * 0.5, 0.62, 0.6)), W((sd * (0.7 + hl * 0.15), 0.75 + hl * 0.2, 0.25 - hl * 0.25)),
                              W((sd * (0.85 + hl * 0.25), 0.85 + hl * 0.35, -0.15 - hl * 0.55)), W((sd * (0.9 + hl * 0.3), 1.0 + hl * 0.4, -0.35 - hl * 0.9))],
                       np.array([0.17, 0.14, 0.08, 0]) * HS, bone='head', per=2, sides=6)
                K.horn(part, [W((sd * 0.84, 0.2, 0.1)), W((sd * (1.0 + hl * 0.2), 0.15, -0.2)), W((sd * (1.1 + hl * 0.3), 0.05, -0.45 - hl * 0.2))],
                       np.array([0.11, 0.07, 0]) * HS, bone='head')
        if t >= 3:
            K.horn(part, [W((0, 0.42, 1.55)), W((0, 0.72, 1.6)), W((0, 0.98, 1.45))], np.array([0.13, 0.09, 0]) * HS, bone='head')

    # ------------------------------------------------------------ Armor: vertebral slabs + keels (t1), costal slabs (t2), spikes (t3)
    for i in range(NV + NC):
        c = SC[i]
        p, n = K.snap(c)
        bn = 'chest' if c[2] > -0.4 else 'pelvis'
        K.plate('arm_t1' if i < NV else 'arm_t2', p, (1, 1, 1), off=0.07, th=0.1, sdf=dome, region=Cell(i, 0.1), voxel=0.06, tris=220, k=0.04, bone=bn)
        q = p + n * 0.17
        if i < NV:
            t = np.array([0, 0, 1.0]) - n * n[2]
            K.leaf('arm_t1', q - n * 0.05, n, t, 0.38, 0.95, th=0.12, lobes=3, mat='plate', bone=bn)
            K.spike('arm_t3', q + n * 0.2, n + [0, 0.15, -0.35], 0.95, bone=bn)
        else:
            K.spike('arm_t3', q, n + [0, 0.35, 0], 0.75, bone=bn)
    for k in range(0, 24, 2):     # rim spikes
        p, n = dome_pt(k * 15 + 7.5, 0, 0.0)
        d = unit([p[0], 0, p[2] - DC[2]])
        q, qn = K.snap(p + d * 0.3)
        K.spike('arm_t3', q - d * 0.05, d + [0, 0.25, 0], 0.55, bone='chest' if p[2] > -0.4 else 'pelvis')
    for sd, nm in ((-1, 'L'), (1, 'R')):   # leg plates (t2)
        f, r = LEGS[nm]['front'], LEGS[nm]['rear']
        K.plate('arm_t2', f[1] + np.array([sd * 1.0, -0.2, 0.25]), (0.7, 1.0, 0.85), lames=3, tris=180, bone='forearm' + nm)
        K.plate('arm_t2', r[1] + np.array([sd * 1.05, -0.2, -0.2]), (0.7, 1.0, 0.85), lames=3, tris=180, bone='shin' + nm)
    K.plate('arm_t2', W((0, 0.85, 0.75)), np.array([0.62, 0.4, 0.75]) * HS, tris=180, bone='head')
    for i in range(1, len(TT) - 2, 2):
        for sd in (-1, 1):
            p, n = K.snap(TT[i] + [sd * 0.5, 0.25, 0])
            K.spike('arm_t3', p, n + [sd * 0.3, 0.3, -0.3], 0.42, bone=tb(i))

    # ------------------------------------------------------------ Energy: glowing seams (t1 crown, t2 flanks), vents (t3)
    def seam_layer(part, region):
        sh = S.Inter(S.Shell(dome, -0.075, 0.02), region, 0.05)
        P = S.SDF()
        P.ops.append(('add', sh, 0.0))
        P._lo, P._hi = sh.lo.copy(), sh.hi.copy()
        K.sdf_piece(part, P, voxel=0.06, tris=700, mat='accent', bone='chest')
    seam_layer('enr_t1', S.Ell(DC + [0, DR[1], 0.2], (2.6, 1.6, 3.4)))
    seam_layer('enr_t2', S.Inter(S.Ell(DC + [0, 1.4, 0], (DR[0] + 0.3, 1.25, DR[2] + 0.3)), S.Ell(DC + [0, 2.3, 0], (DR[0] + 1, 2.4, DR[2] + 1))))
    for p, n in vents:   # glowing vent cores
        K.crystal('enr_t1', p - n * 0.45, n, 0.75, seed=3)
    for th, ph in ((0, 52), (180, 52), (90, 40), (-90, 40)):
        p, n = dome_pt(th, ph)
        q, qn = K.snap(p)
        K.sdf_piece('enr_t3', _cone(q - qn * 0.3, qn, 0.85, 0.32, 0.62), voxel=0.05, tris=220, mat='plate', bone='chest' if q[2] > -0.4 else 'pelvis', ao=0.5)
        K.crystal('enr_t3', q + qn * 0.35, qn, 0.65, seed=int(th) % 7)
    K.ball('enr_t2', W((0, -0.4, 0.75)), np.array([0.3, 0.2, 0.4]) * HS, mat='accent', bone='head', seg=8, rings=6)

    # ------------------------------------------------------------ Speed: flipper fins
    for sd, nm in ((-1, 'L'), (1, 'R')):
        f, r = LEGS[nm]['front'], LEGS[nm]['rear']
        p, n = K.snap(f[1] + np.array([sd * 0.6, -0.3, -0.6]))
        K.frill('spd_t1', p, n * [1, 0.2, 0.3] + [0, 0, -0.8], (0, 1, -0.2), 1.0, 1.0, rays=4, bone='forearm' + nm)
        p, n = K.snap(r[1] + np.array([sd * 0.6, -0.2, -0.6]))
        K.frill('spd_t2', p, n * [1, 0.2, 0.3] + [0, 0, -0.8], (0, 1, 0.2), 0.95, 1.0, rays=4, bone='shin' + nm)
        K.frill('spd_t3', W((sd * 0.82, 0.3, 0.0)), Dd((sd * 1, 0.3, -0.7)), Dd((0, 1, 0.1)), 1.0 * HS, 0.9 * HS, rays=5, spread=110, bone='head')
    p, n = K.snap(TT[-3] + [0, 0.3, 0])
    K.frill('spd_t3', p, (0, 1, -0.2), (0, 0, -1), 0.9, 1.0, rays=4, bone='tail3')

    # ------------------------------------------------------------ Focus: extra eyes, crest, third eye
    for sd in (-1, 1):
        eye(K, 'fcs_t1', W((sd * 0.5, 0.52, 1.45)), Dd((sd * 0.7, 0.5, 0.5)), 0.08 * HS, bone='head', lid=False, seg=8, rings=6)
        eye(K, 'fcs_t3', W((sd * 0.82, 0.05, 0.6)), Dd((sd * 1, 0, 0.3)), 0.085 * HS, bone='head', lid=False, seg=8, rings=6)
    K.leaf('fcs_t2', W((0, 0.85, 0.55)), Dd((0, 1, -0.15)), Dd((0, 0, 1)), 0.65 * HS, 1.3 * HS, th=0.05, lobes=3, mat='accent', bone='head')
    eye(K, 'fcs_t3', W((0, 0.72, 1.12)), Dd((0, 0.6, 0.8)), 0.11 * HS, bone='head', lid=False, seg=8, rings=6)
