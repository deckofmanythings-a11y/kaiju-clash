"""Saurian: the classic upright city-flattener (biped plan). Game coords, ~11 units tall, faces +Z."""
import numpy as np

from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep

DETAIL = 'scales'
UV_SCALE = 0.75

L = 0.7   # upper-body lift over the legs
TAIL = [(0, 4.9, -1.6), (0, 4.0, -3.6), (0.25, 2.8, -5.6), (0.9, 1.75, -7.4), (1.9, 1.05, -8.9), (3.1, 0.72, -9.8), (4.3, 0.75, -10.0)]
TAIL_R = [1.8, 1.45, 1.1, 0.8, 0.52, 0.3, 0.1]


def U(p):
    """Lift upper-body coordinates (point or list of points)."""
    a = np.asarray(p, float)
    return a + np.array([0, L, 0])


def build(K):
    B = K.sdf()
    # ---- torso, hips, neck
    B.tube(U([(0, 4.3, -1.0), (0, 5.6, -0.45), (0, 7.0, 0.35), (0, 8.1, 1.1)]), [1.9, 1.95, 2.2, 1.45], k=0.0, s=(1.08, 1, 1))
    B.ell(U((0, 4.4, -0.9)), (2.0, 1.6, 1.9), k=0.5)
    B.ell(U((0, 7.7, -0.35)), (1.5, 1.2, 1.15), k=0.5)                                   # trapezius hump
    B.tube(U([(0, 7.9, 0.9), (0, 8.7, 1.5), (0, 9.25, 2.2)]), [1.3, 1.0, 0.85], k=0.4)
    B.ell(U((0.8, 7.0, 1.85)), (0.9, 0.8, 0.55), rot=(20, 0, 0), k=0.25, mirror=True)       # pectorals
    B.ell(U((1.7, 7.6, 0.75)), (0.9, 0.9, 0.95), k=0.35, mirror=True)                     # shoulders
    # ---- tail
    B.tube(TAIL, TAIL_R, k=0.5, s=(0.95, 1.05, 1))
    # ---- legs (digitigrade)
    B.tube([(1.5, 5.2, -0.8), (2.0, 3.9, 0.4), (2.0, 3.0, 1.0)], [1.55, 1.3, 0.9], k=0.35, s=(0.85, 1, 1.1), mirror=True)
    B.ell((2.0, 3.0, 1.1), (0.6, 0.62, 0.6), k=0.2, mirror=True)                          # knee
    B.tube([(2.0, 3.0, 1.0), (1.9, 1.9, 0.2), (1.8, 1.0, -0.5)], [0.85, 0.68, 0.5], k=0.2, mirror=True)
    B.ell((1.92, 2.35, 0.0), (0.6, 0.85, 0.72), k=0.25, mirror=True)                      # calf
    B.tube([(1.8, 1.0, -0.5), (1.85, 0.55, 0.4), (1.88, 0.38, 1.1)], [0.5, 0.5, 0.46], k=0.2, s=(1.15, 0.8, 1), mirror=True)
    for dx in (-0.6, 0.0, 0.6):
        B.tube([(1.88 + dx * 0.3, 0.34, 1.0), (1.88 + dx * 0.7, 0.3, 1.6), (1.88 + dx, 0.22, 2.05)], [0.3, 0.27, 0.2], k=0.12, mirror=True)
    B.tube([(1.8, 0.7, -0.55), (1.75, 0.28, -1.0)], [0.35, 0.18], k=0.15, mirror=True)
    # ---- arms
    B.tube(U([(1.8, 7.45, 0.95), (2.35, 6.4, 1.25), (2.3, 5.85, 1.6)]), [0.66, 0.58, 0.46], k=0.3, mirror=True)
    B.tube(U([(2.3, 5.85, 1.6), (2.1, 5.55, 2.3), (1.9, 5.45, 2.9)]), [0.46, 0.45, 0.33], k=0.15, mirror=True)
    B.ell(U((1.85, 5.4, 3.1)), (0.4, 0.27, 0.42), k=0.15, mirror=True)
    for dx in (-0.33, -0.11, 0.11, 0.33):
        B.tube(U([(1.85 + dx, 5.4, 3.3), (1.85 + dx * 1.3, 5.3, 3.72), (1.85 + dx * 1.4, 5.1, 3.95)]), [0.14, 0.13, 0.1], k=0.06, mirror=True)
    # ---- head
    B.ell(U((0, 9.45, 2.55)), (0.92, 0.8, 1.05), k=0.3)
    B.tube(U([(0, 9.4, 3.0), (0, 9.3, 3.8), (0, 9.15, 4.55)]), [0.68, 0.56, 0.42], k=0.25, s=(1.15, 0.72, 1))
    B.tube(U([(0.45, 9.98, 2.5), (0.66, 9.95, 3.1), (0.5, 9.72, 3.7)]), [0.2, 0.25, 0.12], k=0.12, mirror=True)   # brow ridges
    B.ell(U((0.28, 9.5, 4.5)), (0.2, 0.14, 0.22), k=0.1, mirror=True)                                         # nostril bumps
    B.ell(U((0.74, 9.0, 2.75)), (0.42, 0.5, 0.55), k=0.25, mirror=True)                                       # jaw muscle
    B.tube(U([(0, 8.75, 2.6), (0, 8.2, 3.5), (0, 7.95, 4.3)]), [0.55, 0.45, 0.3], k=0.2, s=(1.2, 0.6, 1))      # lower jaw, open
    B.tube(U([(0, 8.6, 1.9), (0, 8.3, 2.6)]), [0.8, 0.6], k=0.4)                                              # throat
    B.ell(U((0, 8.72, 3.9)), (0.85, 0.2, 1.15), rot=(10, 0, 0), cut=True, k=0.08)                             # mouth
    B.ell(U((0.76, 9.55, 3.2)), (0.2, 0.18, 0.26), cut=True, k=0.08, mirror=True)                             # eye sockets
    B.ell(U((0.3, 9.5, 4.78)), (0.06, 0.05, 0.08), cut=True, k=0.03, mirror=True)                             # nostrils

    # ---- belly region (throat -> chest -> under-tail)
    belly = K.sdf()
    belly.tube(list(U([(0, 8.2, 2.3), (0, 7.1, 2.35), (0, 5.6, 1.8), (0, 4.0, 0.9)])) + [(0, 3.2, -1.8), (0, 2.45, -3.6), (0.25, 1.6, -5.6), (0.9, 0.9, -7.4), (1.9, 0.5, -8.9)],
               [0.55, 1.15, 1.4, 1.2, 1.0, 0.9, 0.7, 0.5, 0.3], k=0)
    belly.tube(U([(0, 8.1, 2.5), (0, 7.75, 4.2)]), [0.4, 0.3], k=0.1)

    def skin(x, y, z):
        rough = -0.11 * (fbm(x, y, z, 3, freq=0.9, seed=3, ridged=True) - 0.55) - 0.03 * fbm(x, y, z, 2, freq=3.1, seed=9)
        w = smoothstep(0.12, -0.12, belly.at(np.stack([x, y, z], 1)))
        band = np.abs(np.sin(np.pi * (y - z * 0.6) / 0.6))
        scute = 0.06 * (1 - smoothstep(0.0, 0.3, band)) - 0.02
        return rough * (1 - w) + scute * w
    B.disp(skin, band=0.16)

    mouth = S.Ell(U((0, 8.72, 3.9)), (0.82, 0.26, 1.12), rot=(10, 0, 0))
    K.skin(B, voxel=0.05, tris=20000, belly=belly, regions=[('mouth', lambda P: mouth.dist(S.Points(P)))])

    # ---- skeleton
    K.bone('pelvis', U((0, 4.4, -0.9)), U((0, 5.5, -0.5)), None, 1.9)
    K.bone('spine', U((0, 5.5, -0.5)), U((0, 6.9, 0.3)), 'pelvis', 2.0)
    K.bone('chest', U((0, 6.9, 0.3)), U((0, 8.0, 1.0)), 'spine', 1.8)
    K.bone('neck', U((0, 8.0, 1.0)), U((0, 9.2, 2.2)), 'chest', 1.0)
    K.bone('head', U((0, 9.2, 2.2)), U((0, 9.3, 4.4)), 'neck', 0.8)
    K.bone('jaw', U((0, 8.7, 2.6)), U((0, 8.0, 4.3)), 'head', 0.42)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = np.array([s, 1, 1])
        K.bone('thigh' + sd, m * (1.5, 5.2, -0.8), m * (2.0, 3.0, 1.0), 'pelvis', 1.2)
        K.bone('shin' + sd, m * (2.0, 3.0, 1.0), m * (1.8, 1.0, -0.5), 'thigh' + sd, 0.7)
        K.bone('foot' + sd, m * (1.8, 1.0, -0.5), m * (1.88, 0.25, 1.8), 'shin' + sd, 0.5)
        K.bone('upperarm' + sd, U(m * (1.8, 7.45, 0.95)), U(m * (2.3, 5.85, 1.6)), 'chest', 0.6)
        K.bone('forearm' + sd, U(m * (2.3, 5.85, 1.6)), U(m * (1.9, 5.45, 2.9)), 'upperarm' + sd, 0.45)
        K.bone('hand' + sd, U(m * (1.9, 5.45, 2.9)), U(m * (1.9, 5.15, 3.95)), 'forearm' + sd, 0.32)
        K.mount('hand' + sd, U(m * (1.88, 5.35, 3.4)), 'hand' + sd)
        K.mount('shoulder' + sd, U(m * (1.95, 8.45, 0.7)), 'chest')
    K.chain('tail', TAIL, 'pelvis', (1.6, 0.3))
    K.mount('head', U((0, 9.3, 2.3)), 'head')
    K.mount('jaw', U((0, 8.6, 2.7)), 'jaw')
    K.mount('mouth', U((0, 8.7, 4.6)), 'head')
    K.mount('chest', U((0, 7.0, 2.5)), 'chest')
    K.mount('back', U((0, 7.0, -1.7)), 'spine')
    K.mount('tailTip', TAIL[-1], 'tail6')
    K.mount('crown', U((0, 10.25, 2.5)), 'head')

    # ---- eyes + teeth (base, joined into the body mesh)
    for s in (-1, 1):
        K.eye('body', U((s * 0.67, 9.56, 3.24)), (s * 0.85, 0.1, 0.55), 0.14, bone='head')
        for i in range(8):
            z = 3.1 + i * 0.2
            x = s * (0.6 - i * 0.04)
            ln = 0.3 - 0.012 * i + (0.1 if i in (5, 6) else 0)
            K.horn('body', U([(x, 8.98, z), (x * 1.04, 8.98 - ln * 0.6, z + 0.02), (x, 8.98 - ln, z + 0.05)]), [0.075, 0.05, 0], sides=4, bone='head', shade=(0.8, 1))
        for i in range(7):
            z = 3.05 + i * 0.19
            x = s * (0.5 - i * 0.035)
            y = 8.48 - (z - 2.6) * 0.3
            ln = 0.24 + (0.1 if i == 5 else 0)
            K.horn('body', U([(x, y, z), (x * 1.04, y + ln * 0.6, z), (x, y + ln, z - 0.03)]), [0.065, 0.045, 0], sides=4, bone='jaw', shade=(0.8, 1))

    # ---- dorsal plates (base silhouette) + glow cores per Energy tier
    tc, tr = S.catmull(TAIL, 2), S.catmull(TAIL_R, 2)[:, 0]
    row = np.vstack([U([(0, 9.7, 1.5), (0, 9.2, 0.6), (0, 8.6, -0.5), (0, 7.8, -1.4), (0, 6.7, -2.0), (0, 5.8, -2.5), (0, 5.0, -3.0)]),
                     tc[2:11] + np.stack([0 * tr[2:11], tr[2:11] * 1.1, 0 * tr[2:11]], 1)])
    hs = [0.5, 0.85, 1.3, 1.7, 1.95, 1.75, 1.45, 1.25, 1.1, 0.95, 0.8, 0.66, 0.52, 0.4, 0.3, 0.22]
    P, N = K.snap(row)
    for i, (p, n, h) in enumerate(zip(P, N, hs)):
        t = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)])
        n = n * [0.25, 1, 1]
        K.leaf('body', p - n * 0.05, n, -t, h, h * 0.95, th=0.09 + h * 0.05, seed=i)
        K.leaf(f'enr_t{1 if i % 2 == 0 else 2}', p - n * 0.05, n, -t, h, h * 0.95, th=0.09 + h * 0.05, core=True, mat='accent')
        if 2 <= i <= 8:  # flanking rows
            for s in (-1, 1):
                q, qn = K.snap(p + np.array([s * (0.55 + h * 0.22), -0.1, 0.1]))
                qn = qn * [0.6, 1, 1] + [s * 0.25, 0, 0]
                K.leaf('body', q - qn * 0.04, qn, -t, h * 0.55, h * 0.6, th=0.07 + h * 0.03, lobes=3)
                K.leaf('enr_t3', q - qn * 0.04, qn, -t, h * 0.55, h * 0.6, th=0.07 + h * 0.03, lobes=3, core=True, mat='accent')

    # ---- claws per Might tier (exclusive sets) + horns from tier 1
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        k = 0.75 + 0.25 * sc
        for s, sd in ((-1, 'L'), (1, 'R')):
            for dx in (-0.6, 0.0, 0.6):
                K.claw(part, (s * (1.88 + dx), 0.24, 2.08), (s * dx * 0.25, 0.05, 1), 0.62 * k, 0.17 * (0.8 + 0.2 * sc), bone='foot' + sd)
            K.claw(part, (s * 1.75, 0.26, -1.05), (0, -0.1, -1), 0.34 * k, 0.12, bone='foot' + sd)
            for dx in (-0.33, -0.11, 0.11, 0.33):
                K.claw(part, U((s * (1.85 + dx * 1.4), 5.08, 3.97)), (s * dx * 0.3, -0.75, 0.65), 0.42 * k, 0.09 * (0.8 + 0.2 * sc), curl=(0, -0.6, -0.8), bone='hand' + sd)
            if t >= 1:
                hl = [0, 0.9, 1.25, 1.6][t]
                K.horn(part, U([(s * 0.62, 9.9, 2.5), (s * 0.85, 10.0 + hl * 0.3, 2.3 - hl * 0.3), (s * 0.95, 10.1 + hl * 0.65, 2.0 - hl * 0.75), (s * 0.85, 10.3 + hl * 0.9, 1.9 - hl * 1.2)]),
                       [0.2, 0.17, 0.1, 0], bone='head', per=3, sides=7)
            if t >= 3:
                K.horn(part, U([(s * 0.45, 9.55, 4.0), (s * 0.5, 9.85, 4.15), (s * 0.5, 10.1, 4.05)]), [0.13, 0.09, 0], bone='head')

    # ---- armour (Armor tiers): lamellar plates hug the body, tier 3 adds spikes
    spots = {
        1: [(U((1.95, 8.3, 0.7)), (1.0, 0.85, 1.0), 'chest', 3), ((2.5, 4.3, 0.2), (0.8, 1.15, 1.05), 'thigh', 3)],
        2: [(U((2.65, 6.3, 1.2)), (0.5, 0.75, 0.55), 'upperarm', 2), ((2.1, 2.1, 0.85), (0.55, 0.8, 0.5), 'shin', 2),
            (U((1.4, 6.2, -1.9)), (0.8, 1.25, 0.8), 'spine', 3), (U((0.0, 10.05, 2.75)), (0.7, 0.4, 0.8), 'head', 0)],
    }
    for tier, lst in spots.items():
        for c, r, bone, lames in lst:
            for s, sd in ((-1, 'L'), (1, 'R')):
                if c[0] == 0 and s == 1:
                    continue
                bn = bone + sd if bone in ('thigh', 'upperarm', 'shin') else bone
                cc = np.array([s * c[0], c[1], c[2]])
                K.plate(f'arm_t{tier}', cc, r, bone=bn, lames=lames)
                p, n = K.snap(cc)
                if tier == 1:
                    K.spike('arm_t3', p + n * 0.1, n + np.array([0, 0.45, -0.25]), 0.8, bone=bn)
                    K.spike('arm_t3', p + n * 0.1 + [0, -0.45, 0.25], n + np.array([0, 0.1, 0.1]), 0.45, bone=bn)
    for i in (11, 12, 13, 14):   # tail thagomizer spikes
        for s in (-1, 1):
            p, n = K.snap(P[i] + np.array([s * 0.5, -0.3, 0]))
            K.spike('arm_t3', p, n * [1, 0.3, 1] + [s * 0.6, 0.25, 0], 0.75 - (i - 11) * 0.1)
    for s, sd in ((-1, 'L'), (1, 'R')):   # knee + elbow spikes
        K.spike('arm_t3', (s * 2.05, 3.05, 1.6), (s * 0.2, 0.3, 1), 0.6, bone='shin' + sd)
        K.spike('arm_t3', U((s * 2.55, 5.8, 1.2)), (s * 0.5, -0.2, -1), 0.6, bone='forearm' + sd)

    # ---- Speed tiers: spined fins
    for s, sd in ((-1, 'L'), (1, 'R')):
        p, n = K.snap(U((s * 2.5, 5.7, 1.9)))
        K.frill('spd_t1', p, n * [1, 0.2, 0.2] + [0, 0.25, -0.5], (0, -0.4, 1), 0.9, 0.9, rays=4, bone='forearm' + sd)
        p, n = K.snap((s * 2.0, 2.2, -0.75))
        K.frill('spd_t2', p, n * [0.5, 0, 1] + [0, 0.1, -0.5], (0, 1, 0.2), 1.0, 1.0, rays=4, bone='shin' + sd)
        p, n = K.snap(U((s * 0.95, 9.1, 2.2)))
        K.frill('spd_t3', p, n * [1, 0.3, 0] + [s * 0.3, 0.1, -0.75], (0, 1, 0.1), 1.35, 1.2, rays=6, spread=130, bone='head')

    # ---- Focus tiers: extra eyes, crest, third eye
    for s in (-1, 1):
        K.eye('fcs_t1', U((s * 0.6, 9.74, 3.72)), (s * 0.7, 0.5, 0.5), 0.085, bone='head', lid=False)
        K.eye('fcs_t3', U((s * 0.88, 9.25, 2.9)), (s * 1, 0, 0.3), 0.09, bone='head', lid=False)
        K.eye('fcs_t3', U((s * 0.48, 9.62, 4.05)), (s * 0.7, 0.5, 0.5), 0.065, bone='head', lid=False)
    K.leaf('fcs_t2', U((0, 10.1, 2.95)), (0, 1, 0.15), (0, 0, 1), 0.7, 1.4, th=0.06, lobes=3, mat='accent', bone='head')
    K.eye('fcs_t3', U((0, 9.98, 3.4)), (0, 0.6, 0.8), 0.12, bone='head', lid=False)
