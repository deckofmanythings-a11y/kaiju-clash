"""Sea Serpent: a coiling sea dragon rearing up out of its own coils. Game coords, faces +Z, ground y=0."""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep
from .hydra import Head, eye, unit

DETAIL = 'scales'
UV_SCALE = 0.8

# body spline tail tip -> neck end: ground coil, then rising S out of the coil centre
PTS = [(3.6, 0.25, 3.6), (4.2, 0.38, 1.6), (4.0, 0.52, -0.9), (2.9, 0.65, -3.0), (0.8, 0.78, -4.1), (-1.7, 0.88, -3.7),
       (-3.4, 0.96, -1.9), (-3.7, 1.02, 0.4), (-2.6, 1.06, 2.3), (-0.4, 1.1, 3.0), (1.5, 1.15, 2.1), (1.95, 1.45, 0.1),
       (1.0, 2.3, -1.3), (-0.2, 3.5, -1.3), (-0.5, 4.8, -0.4), (-0.3, 6.1, 0.3), (0.0, 7.35, 0.1), (0.0, 8.5, 0.5),
       (0.0, 9.35, 1.4), (0.0, 9.75, 2.45)]
RAD = [0.1, 0.28, 0.45, 0.6, 0.74, 0.86, 0.95, 1.01, 1.06, 1.1, 1.12, 1.13, 1.12, 1.1, 1.05, 0.98, 0.9, 0.8, 0.72, 0.66]
# reference 'belly' direction per control point (ground coil: down; rising part: forward)
REF = [(0, -1, 0)] * 11 + [(0, -1, 0.2), (0.3, -0.6, 0.8), (0, -0.3, 1), (0, -0.2, 1), (0, -0.2, 1), (0, -0.3, 1), (0, -0.4, 1), (0, -0.8, 0.6), (0, -1, 0.2)]
HEAD_O = (0.0, 9.8, 2.72)
HEAD_F = (0, -0.28, 1)
HEAD_SC = 0.98


def spline():
    C = S.catmull(PTS, 6)
    R = S.catmull(RAD, 6)[:, 0]
    Ref = S.catmull(REF, 6)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Th = Ref - T * (Ref * T).sum(1, keepdims=True)
    Th /= np.linalg.norm(Th, axis=1, keepdims=True)
    Lat = np.cross(T, Th)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    return C, R, T, Th, Lat, s


def nearest(P, C, chunk=40000):
    """Index of the nearest spline sample for each point (chunked brute force)."""
    out = np.empty(len(P), np.int64)
    for i in range(0, len(P), chunk):
        Q = P[i:i + chunk]
        d = ((Q[:, None, :] - C[None, :, :]) ** 2).sum(2)
        out[i:i + chunk] = d.argmin(1)
    return out


def build(K):
    C, R, T, Th, Lat, s = spline()
    n = len(C)
    B = K.sdf()
    # ---- body: one long tube (body slightly flattened by sculpt below), belly keel
    B.tube(PTS, RAD, k=0.0, per=6, s=(1.06, 0.88, 1.06))
    head = Head(HEAD_O, HEAD_F, sc=HEAD_SC, open_deg=24)
    # throat / jowl into the head
    B.tube([C[-14], C[-6], head.W((0, -0.45, 0.0))], [R[-14] * 1.02, R[-6] * 1.05, 0.62], k=0.3)
    # back muscle along the rising part
    for i in range(int(n * 0.6), n - 8, 6):
        B.ell(C[i] - Th[i] * R[i] * 0.35, (R[i] * 0.85,) * 3, k=0.35)
    head.sculpt(B, snout=1.18, brow=1.25, cheek=0.95, crest=1.0, wide=0.92)
    for sd in (-1, 1):   # fin-ear roots + whisker pads
        B.add(head.xf(S.Ell((sd * 0.82, 0.22, -0.15), (0.2, 0.28, 0.35))), 0.15)
        B.add(head.xf(S.Ell((sd * 0.42, -0.12, 1.95), (0.17, 0.15, 0.22))), 0.1)

    # ---- belly region (ventral scutes) + photophores (accent)
    def ventral(P):
        idx = nearest(P, C)
        d = P - C[idx]
        cosv = (d * Th[idx]).sum(1) / np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        return idx, cosv
    bel = K.sdf()
    bel.tube(list(C + Th * R[:, None] * 0.95), list(R * 0.66), k=0, per=1)
    bel.add(head.xf(S.Tube([(0, -0.75, 0.0), (0, -0.85, 0.9), (0, -0.78, 1.8)], [0.4, 0.3, 0.16]), 'jaw'), 0)
    bel.add(head.xf(S.Ell((0, -0.75, -0.3), (0.45, 0.4, 0.6)), 'mid'), 0)

    dots = []
    for i in range(12, n - 12, 4):
        for sd in (-1, 1):
            a = 0.55 + 0.1 * ((i // 4) % 2)
            dots.append(C[i] + (Lat[i] * sd * np.sqrt(1 - a * a) + Th[i] * a) * R[i])
    dots = np.array(dots)

    def skin(x, y, z):
        P = np.stack([x, y, z], 1)
        idx = nearest(P, C)
        ss = s[idx]
        w = smoothstep(0.12, -0.12, bel.at(P))
        band = np.abs(np.sin(np.pi * ss / 0.42))
        scute = 0.06 * (1 - smoothstep(0.0, 0.32, band)) - 0.02
        # overlapping diamond scales on the back/sides
        u = ss / 0.5
        d = P - C[idx]
        ang = np.arctan2((d * Lat[idx]).sum(1), -(d * Th[idx]).sum(1)) * R[idx] / 0.5
        cell = np.abs(np.sin(np.pi * (u + ang))) * np.abs(np.sin(np.pi * (u - ang)))
        sc = -0.05 * smoothstep(0.0, 0.5, cell) + 0.02
        rough = -0.04 * fbm(x, y, z, 2, freq=2.2, seed=4)
        # flatten the coil slightly: push the very bottom in
        return (sc + rough) * (1 - w) + scute * w
    B.disp(skin, band=0.14)

    mouth = head.mouth_prim(0.04)

    def dot_fn(P):
        d = np.full(len(P), 9.0)
        for c in dots:
            d = np.minimum(d, np.linalg.norm(P - c, axis=1) - 0.13)
        return d
    K.skin(B, voxel=0.05, tris=17000, belly=bel, regions=[('mouth', lambda P: mouth.at(P)), ('accent', dot_fn)])

    # ---- skeleton: tail chain round the coil, long spine chain up to the head
    i0 = 8 * 6   # control point 8 (front-left of the coil)
    tail_idx = np.linspace(i0, 0, 7).astype(int)
    spine_idx = np.linspace(i0, n - 1, 13).astype(int)
    K.chain('spine', C[spine_idx], None, (1.15, 0.75))
    K.chain('tail', C[tail_idx], 'spine1', (0.95, 0.25))
    K.bone('head', head.W((0, 0, -0.1)), head.W((0, -0.05, 2.4)), 'spine12', 0.8)
    K.bone('jaw', head.WJ((0, -0.45, 0.25)), head.WJ((0, -0.6, 2.1)), 'head', 0.42)

    def sb(i):   # spine bone owning spline sample i
        j = int(np.searchsorted(spine_idx, i, 'right'))
        return f'spine{min(max(j, 1), 12)}'

    K.mount('head', head.W((0, 0.1, 0.3)), 'head')
    K.mount('jaw', head.WJ((0, -0.55, 0.5)), 'jaw')
    K.mount('mouth', head.W((0, -0.4, 2.5)), 'head')
    K.mount('crown', head.W((0, 1.0, 0.35)), 'head')
    ic = int(n * 0.8)
    K.mount('chest', C[ic] + Th[ic] * R[ic], sb(ic))
    ib = int(n * 0.55)
    K.mount('back', C[ib] - Th[ib] * R[ib], sb(ib))
    K.mount('tailTip', C[0], 'tail6')
    ish = int(n * 0.74)
    for sd, nm in ((-1, 'L'), (1, 'R')):
        K.mount('hand' + nm, head.W((sd * 0.45, -0.3, 1.9)), 'head')
        K.mount('shoulder' + nm, C[ish] + Lat[ish] * sd * R[ish] * 1.0 - Th[ish] * 0.3, sb(ish))

    # ---- face: eyes + teeth, whiskers, fin ears
    head.dress(K, 'body', bone='head', jaw_bone='jaw', eye_r=0.15, teeth=8)
    K.ball('body', head.W((0, -0.42, 0.55)), 0.22 * HEAD_SC, mat='accent', bone='head', seg=8, rings=6)
    for sd in (-1, 1):
        r0 = head.W((sd * 0.45, -0.14, 2.02))
        K.horn('body', [r0, r0 + head.D((sd * 0.55, -0.15, 0.25)), r0 + head.D((sd * 1.0, -0.65, 0.05)), r0 + head.D((sd * 1.15, -1.3, -0.4)), r0 + head.D((sd * 1.05, -1.85, -0.75))],
               [0.075, 0.06, 0.045, 0.03, 0.0], mat='belly', sides=5, per=3, bone='head')
        r1 = head.W((sd * 0.35, -0.95, 1.4))
        K.horn('body', [head.WJ((sd * 0.3, -0.75, 1.5)), head.WJ((sd * 0.38, -1.15, 1.45)), head.WJ((sd * 0.35, -1.55, 1.25))], [0.05, 0.035, 0.0], mat='belly', sides=5, bone='jaw')
        K.frill('body', head.W((sd * 0.86, 0.25, -0.1)), head.D((sd * 1, 0.55, -0.6)), head.D((0, 1, 0.15)), 1.15, 0.95, rays=5, spread=115, bone='head')

    # ---- dorsal sail: continuous membrane with bony rays from the head down onto the coil
    dorsal = np.arange(58, n - 6, 2)
    base, nrm = K.snap(C[dorsal] - Th[dorsal] * R[dorsal, None] * 1.05)
    u = (dorsal - dorsal[0]) / (dorsal[-1] - dorsal[0])          # 0 tail end .. 1 head end
    env = 0.15 + 0.5 * smoothstep(0.0, 0.1, u) + 1.0 * smoothstep(0.1, 0.45, u) * (1 - 0.55 * smoothstep(0.8, 1.0, u))
    rays = np.arange(len(dorsal)) % 3 == 0
    hgt = env * np.where(rays, 1.0, 0.72)
    back = -T[dorsal] * 0.35
    tips = base + (nrm * [0.15, 1, 1] + back) / np.linalg.norm(nrm * [0.15, 1, 1] + back, axis=1, keepdims=True) * hgt[:, None]
    base = base - nrm * 0.06
    V, Tr, uu = geom.strip(base, tips)
    bones = [sb(i) for i in dorsal]
    # split the sail into bone-sized panels so it bends with the spine
    for j in range(len(dorsal) - 1):
        q = np.array([base[j], base[j + 1], tips[j + 1], tips[j]])
        K.piece('body', q, np.array([(0, 1, 2), (0, 2, 3)]), 'membrane', col=np.array([0.55, 0.55, 1, 1]), bone=bones[j], smooth=False)
    for j in np.nonzero(rays)[0]:
        K.horn('body', [base[j], base[j] * 0.45 + tips[j] * 0.55, tips[j] + (tips[j] - base[j]) * 0.08], [0.07, 0.05, 0.0], sides=4, bone=bones[j], shade=(0.7, 1))
        if hgt[j] > 0.7:
            K.crystal('enr_t2', tips[j] - (tips[j] - base[j]) * 0.05, tips[j] - base[j], 0.3, cluster=False, bone=bones[j])

    # ---- low keel of bony scutes along the coil, leading the sail down into the tail fin
    for k, i in enumerate(range(8, 58, 3)):
        p, nn = K.snap(C[i] - Th[i] * R[i] * 1.05)
        hk = 0.18 + 0.32 * (i / 58)
        K.leaf('body', p - nn * 0.04, nn, -T[i], hk * 1.2, hk * 1.3, th=0.06, lobes=3, seed=40 + k, bone=sb(i) if i >= i0 else None)

    # ---- tail fin
    tt, tn = C[0], unit(C[0] - C[3])
    K.frill('body', C[2] + [0, 0.15, 0], (0, 1, 0), tn, 1.3, 1.4, rays=5, spread=100, bone='tail6')
    K.frill('body', C[2] + [0, 0.1, 0], -tn * 0.3 + [0, 0.05, 0] + np.cross([0, 1, 0], tn) * 1.0, tn, 1.0, 1.2, rays=4, spread=90, bone='tail6')
    K.frill('body', C[2] + [0, 0.1, 0], -tn * 0.3 + [0, 0.05, 0] - np.cross([0, 1, 0], tn) * 1.0, tn, 1.0, 1.2, rays=4, spread=90, bone='tail6')

    # ---- Might: fangs + tail blade grow, horns from tier 1
    for t, scl in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        kk = 0.7 + 0.3 * scl
        for sd in (-1, 1):
            x = sd * 0.36
            ln = 0.42 * kk
            K.horn(part, head.W([(x, -0.22, 1.6), (x * 1.05, -0.22 - ln * 0.6, 1.64), (x * 0.95, -0.22 - ln, 1.6)]), [0.1, 0.07, 0], sides=6, bone='head')
            K.horn(part, head.WJ([(sd * 0.3, -0.45, 1.85), (sd * 0.31, -0.45 + ln * 0.5, 1.83), (sd * 0.29, -0.45 + ln * 0.8, 1.78)]), [0.08, 0.055, 0], sides=6, bone='jaw')
            if t >= 1:
                hl = [0, 0.9, 1.25, 1.6][t]
                K.horn(part, head.W([(sd * 0.5, 0.62, 0.3), (sd * 0.62, 0.7 + hl * 0.2, -0.2 - hl * 0.4), (sd * 0.68, 0.8 + hl * 0.35, -0.7 - hl * 0.8), (sd * 0.6, 0.95 + hl * 0.4, -1.0 - hl * 1.2)]),
                       [0.17, 0.14, 0.08, 0], bone='head', per=2, sides=6)
            if t >= 3:
                K.horn(part, head.W([(sd * 0.28, 0.35, 1.9), (sd * 0.33, 0.62, 1.85), (sd * 0.35, 0.85, 1.65)]), [0.09, 0.06, 0], bone='head')
        p, nn = C[1], unit(C[0] - C[2])
        K.horn(part, [p + [0, 0.25, 0], p + [0, 0.45 + 0.2 * scl, 0] + nn * 0.4 * scl, p + [0, 0.55 + 0.3 * scl, 0] + nn * 0.9 * scl], [0.16, 0.1, 0], bone='tail5', sides=5)

    # ---- Armor: plates along the rising back (t1), coil + head (t2), spikes (t3)
    for k, i in enumerate(range(int(n * 0.62), n - 8, 7)):
        c = C[i] - Th[i] * R[i] * 0.7
        K.plate('arm_t1', c, (R[i] * 1.0, R[i] * 0.85, R[i] * 1.0), bone=sb(i), lames=2, lame_axis=T[i], tris=180)
        for sd in (-1, 1):
            p, nn = K.snap(C[i] + Lat[i] * sd * R[i] * 0.95 - Th[i] * R[i] * 0.3)
            K.spike('arm_t3', p + nn * 0.08, nn - T[i] * 0.4, 0.55, bone=sb(i))
    for i in range(20, i0 + 12, 8):
        for sd in (-1, 1):
            c = C[i] + Lat[i] * sd * R[i] * 0.62 - Th[i] * R[i] * 0.55
            rot = np.stack([Lat[i], -Th[i], T[i]], 1)
            K.plate('arm_t2', c, (R[i] * 0.7, R[i] * 0.7, R[i] * 1.5), rot=rot, bone=sb(i) if i >= i0 else None, lames=3, lame_axis=T[i], tris=150)
            p, nn = K.snap(c)
            if (i // 9) % 2 == 0:
                K.spike('arm_t3', p + nn * 0.08, nn + [0, 0.3, 0], 0.5, bone=sb(i) if i >= i0 else None)
    K.plate('arm_t2', head.W((0, 0.72, 0.7)), np.array([0.6, 0.42, 0.8]) * HEAD_SC * 1.1, bone='head', tris=200)
    p, nn = K.snap(head.W((0, 0.75, 0.15)))
    K.spike('arm_t3', p, nn + head.D((0, 0.1, -0.8)), 0.6, bone='head')

    # ---- Energy: photophore pearls (t1), sail ray tips (t2, above), back crystals + throat core (t3)
    for k, i in enumerate(range(i0 + 4, n - 10, 4)):
        for sd in (-1, 1):
            a = 0.2
            p, nn = K.snap(C[i] + (Lat[i] * sd * np.sqrt(1 - a * a) + Th[i] * a) * R[i])
            K.ball('enr_t1', p - nn * 0.07, (0.15, 0.15, 0.15), mat='accent', seg=8, rings=6, bone=sb(i))
    for j in range(len(dorsal) // 3, len(dorsal) - 2, 3):
        for sd in (-1, 1):
            i = dorsal[j]
            p, nn = K.snap(C[i] + Lat[i] * sd * R[i] * 0.55 - Th[i] * R[i] * 0.8)
            K.crystal('enr_t3', p - nn * 0.08, nn + Lat[i] * sd * 0.5, 0.5, seed=j + (sd > 0), bone=sb(i))

    # ---- Speed: pectoral fins (t1), coil side fins (t2), big fin-ear fans (t3)
    ip = int(n * 0.7)
    for sd in (-1, 1):
        p, nn = K.snap(C[ip] + Lat[ip] * sd * R[ip] + Th[ip] * R[ip] * 0.3)
        K.frill('spd_t1', p, nn * [1, 0.3, 0.3] + Th[ip] * 0.3 - T[ip] * 0.5, -T[ip], 1.3, 1.1, rays=5, bone=sb(ip))
        for i in (24, 42):
            p, nn = K.snap(C[i] + Lat[i] * sd * R[i])
            K.frill('spd_t2', p, nn * [1, 0.6, 1] + [0, 0.5, 0], -T[i], 0.9, 1.1, rays=4, bone=sb(i) if i >= i0 else None)
        K.frill('spd_t3', head.W((sd * 0.8, -0.35, 0.25)), head.D((sd * 1, -0.35, -0.6)), head.D((0, -1, 0.1)), 1.0, 0.9, rays=5, spread=110, bone='head')
        K.frill('spd_t3', head.W((sd * 0.6, 0.6, 0.0)), head.D((sd * 0.5, 1, -0.7)), head.D((0, 1, 0.1)), 1.3, 1.0, rays=5, spread=110, bone='head')

    # ---- Focus: extra eyes, crest, third eye
    for sd in (-1, 1):
        eye(K, 'fcs_t1', head.W((sd * 0.55, 0.45, 1.5)), head.D((sd * 0.7, 0.5, 0.5)), 0.08 * HEAD_SC, bone='head', lid=False, seg=8, rings=6)
        eye(K, 'fcs_t3', head.W((sd * 0.82, 0.02, 0.6)), head.D((sd * 1, 0, 0.3)), 0.085 * HEAD_SC, bone='head', lid=False, seg=8, rings=6)
    K.leaf('fcs_t2', head.W((0, 0.8, 0.5)), head.D((0, 1, -0.2)), head.D((0, 0, 1)), 0.75, 1.5, th=0.05, lobes=3, mat='accent', bone='head')
    eye(K, 'fcs_t3', head.W((0, 0.68, 1.15)), head.D((0, 0.6, 0.8)), 0.115 * HEAD_SC, bone='head', lid=False, seg=8, rings=6)
