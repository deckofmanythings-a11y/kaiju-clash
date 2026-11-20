"""Titan Ape: knuckle-walking mountain of muscle (ape plan). Game coords, ~10.5 units tall, faces +Z."""
import numpy as np

from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep, voronoi

DETAIL = 'fur'
UV_SCALE = 0.75

# key joints (right side, +X); L is mirrored
SH = np.array([2.75, 8.1, 0.9])      # shoulder
EL = np.array([3.75, 5.35, 1.15])    # elbow (flared out + back)
WR = np.array([3.5, 2.3, 2.4])       # wrist
FI = np.array([3.4, 1.05, 2.7])      # fist centre
HIP = np.array([1.45, 4.7, -2.0])
KN = np.array([2.0, 2.6, -0.65])
AN = np.array([1.9, 0.8, -1.55])
KNUCKLE_X = (-0.6, -0.2, 0.2, 0.6)


def M(s):
    return np.array([s, 1, 1])


def build(K):
    K.mat('belly', detail_kind='skin', rough=0.75)
    K.mat('plate', color='#7f705e', rough=0.5, metal=0.0)
    K.mat('mane', color='#7a7068', rough=0.9, metal=0.0, vcol=True, detail=True)
    B = K.sdf()
    # ---- torso: sloping gorilla back, barrel chest thrust forward, giant trapezius + deltoids
    B.tube([(0, 4.75, -2.2), (0, 5.9, -1.6), (0, 7.2, -0.5), (0, 8.3, 0.55)], [1.6, 1.85, 2.15, 2.05], k=0.0, s=(1.12, 1, 1))
    B.ell((0, 4.95, -2.0), (1.6, 1.3, 1.3), k=0.5)                               # rump
    B.ell((0, 7.35, 0.75), (2.1, 1.75, 1.6), k=0.5)                                  # rib cage
    B.ell((0.98, 7.55, 1.95), (1.02, 0.85, 0.62), rot=(28, 0, 0), k=0.3, mirror=True)  # pectorals
    B.ell((0, 5.8, 0.6), (1.45, 1.25, 1.2), k=0.55)                                 # gut
    B.ell((0, 9.0, 0.2), (1.6, 0.95, 1.3), k=0.5)                                    # trapezius hump
    B.tube([(0, 9.05, 0.3), (1.5, 9.0, 0.6), (2.45, 8.55, 0.8)], [1.0, 0.92, 0.8], k=0.4, mirror=True)
    B.ell(SH, (1.22, 1.2, 1.22), k=0.3, mirror=True)                                 # deltoids
    B.ell((1.7, 6.8, -0.9), (0.95, 1.55, 1.3), rot=(-25, 0, 0), k=0.4, mirror=True)  # lats
    B.ell((0, 8.05, -1.0), (1.45, 1.25, 1.0), rot=(-30, 0, 0), k=0.5)                # upper back
    # ---- arms: long, massive forearms, knuckle-walking fists
    B.tube([SH, (3.35, 6.7, 1.05), EL], [1.05, 0.9, 0.68], k=0.32, mirror=True)
    B.ell((3.15, 6.85, 1.55), (0.68, 1.0, 0.64), rot=(-10, 0, 14), k=0.28, mirror=True)    # biceps
    B.ell((3.7, 7.0, 0.65), (0.64, 1.05, 0.62), rot=(15, 0, 14), k=0.28, mirror=True)     # triceps
    B.ell((EL + (0.15, 0.05, -0.35)), (0.55, 0.55, 0.5), k=0.2, mirror=True)               # elbow point
    B.tube([EL, (3.8, 4.1, 1.75), WR], [0.8, 1.12, 0.6], k=0.3, mirror=True)
    B.ell((3.95, 4.45, 1.55), (0.95, 1.2, 0.88), rot=(18, 0, -8), k=0.28, mirror=True)       # forearm muscle
    B.ell(FI + (0, 0.1, -0.05), (1.0, 0.85, 0.95), k=0.22, mirror=True)
    for dx in KNUCKLE_X:
        B.tube([FI + (dx * 0.9, 0.05, 0.45), FI + (dx, -0.5, 0.88)], [0.25, 0.25], k=0.03, mirror=True)  # curled fingers
        B.ell(FI + (dx, -0.72, 0.86), (0.23, 0.3, 0.3), k=0.03, mirror=True)                          # knuckle pads
    for dx in (-0.4, 0.0, 0.4):   # grooves between the fingers
        B.tube([FI + (dx * 0.9, -0.1, 0.75), FI + (dx, -0.55, 1.12), FI + (dx, -0.95, 1.0)], [0.08, 0.1, 0.08], k=0.04, cut=True, mirror=True)
    B.tube([FI + (-0.82, 0.15, 0.15), FI + (-1.0, -0.3, 0.5), FI + (-0.88, -0.6, 0.78)], [0.33, 0.29, 0.25], k=0.14, mirror=True)  # thumb
    # ---- legs: short, bowed, flat-footed with grasping big toe
    B.tube([HIP, (2.0, 3.6, -1.1), KN], [1.25, 1.08, 0.82], k=0.35, mirror=True)
    B.tube([KN, (1.98, 1.6, -1.2), AN], [0.8, 0.75, 0.58], k=0.25, mirror=True)
    B.ell((2.02, 1.85, -1.55), (0.62, 0.85, 0.6), k=0.25, mirror=True)                    # calf
    B.tube([(1.9, 0.42, -2.0), (1.95, 0.4, -1.05), (2.0, 0.34, -0.35)], [0.52, 0.54, 0.45], k=0.2, s=(1.3, 0.75, 1), mirror=True)
    for dx in (-0.25, 0.08, 0.38):
        B.tube([(2.0 + dx, 0.3, -0.35), (2.0 + dx * 1.2, 0.22, 0.05)], [0.21, 0.17], k=0.08, mirror=True)
    B.tube([(1.55, 0.32, -0.65), (1.28, 0.25, -0.2)], [0.25, 0.19], k=0.1, mirror=True)    # splayed big toe
    # ---- head: thrust forward, low between the shoulders
    B.ell((0, 9.3, 2.4), (1.02, 0.92, 1.02), k=0.3)                                  # cranium
    B.tube([(0, 10.05, 2.75), (0, 10.5, 2.05), (0, 10.25, 1.3), (0, 9.65, 0.75)], [0.24, 0.44, 0.38, 0.2], k=0.25, s=(0.5, 1, 1))  # sagittal crest
    B.tube([(-0.95, 9.55, 2.95), (-0.45, 9.45, 3.32), (0, 9.33, 3.45), (0.45, 9.45, 3.32), (0.95, 9.55, 2.95)], [0.27, 0.3, 0.26, 0.3, 0.27], k=0.12, s=(1, 0.72, 1))     # heavy brow shelf
    B.ell((0, 8.82, 3.5), (0.32, 0.26, 0.3), k=0.15)                               # nose bridge / flat nose
    B.ell((0, 8.45, 3.22), (0.82, 0.5, 0.64), k=0.28)                                # prognathic muzzle
    B.ell((0.8, 8.65, 2.5), (0.42, 0.58, 0.55), k=0.25, mirror=True)                 # cheek / jaw muscle
    B.tube([(0, 7.98, 2.3), (0, 7.7, 3.0), (0, 7.62, 3.42)], [0.62, 0.55, 0.45], k=0.2, s=(1.3, 0.7, 1))  # open lower jaw
    B.ell((0, 8.1, 3.38), (0.62, 0.25, 0.62), rot=(12, 0, 0), cut=True, k=0.08)      # mouth
    B.ell((0.42, 9.13, 3.22), (0.24, 0.17, 0.24), cut=True, k=0.05, mirror=True)       # eye sockets
    B.ell((0.16, 8.8, 3.78), (0.11, 0.08, 0.1), rot=(0, 0, 25), cut=True, k=0.03, mirror=True)       # nostrils

    # ---- bare leathery skin: chest + belly, face, knuckle pads/palms, soles
    belly = K.sdf()
    belly.tube([(0, 8.4, 2.25), (0, 7.45, 2.5), (0, 6.2, 2.0), (0, 5.1, 1.3)], [0.85, 1.55, 1.45, 0.95], k=0)
    belly.ell((0, 8.85, 3.45), (0.98, 0.62, 0.72), k=0)      # face mask (eyes, nose, muzzle)
    belly.ell((0, 8.15, 3.3), (0.8, 0.5, 0.7), k=0)
    for s in (-1, 1):
        belly.ell(M(s) * (FI + (0, -0.45, 0.85)), (1.05, 0.55, 0.42), k=0)
        belly.ell(M(s) * (1.95, 0.12, -1.0), (0.8, 0.26, 1.3), k=0)

    def fur(x, y, z):
        P = np.stack([x, y, z], 1)
        w = smoothstep(0.1, -0.1, belly.at(P))
        f1, f2, _ = voronoi(x * 2.4, y * 1.0, z * 2.4, 1.0, seed=7)      # vertically stretched clumps
        clump = smoothstep(0.0, 0.32, f2 - f1)
        g1, g2, _ = voronoi(x * 5.0, y * 2.2, z * 5.0, 1.0, seed=11)
        strand = smoothstep(0.0, 0.3, g2 - g1)
        hair = -0.07 * clump - 0.025 * strand + 0.035
        leather = -0.03 * fbm(x, y, z, 3, freq=2.2, seed=4, ridged=True) - 0.012 * fbm(x, y, z, 2, freq=6.0, seed=2)
        return hair * (1 - w) + leather * w
    # chest definition: sternum groove + pectoral underline carved along the surface
    st, _ = B.snap(np.array([(0, 8.2, 3.2), (0, 7.6, 3.2), (0, 7.0, 3.2), (0, 6.5, 3.0)]))
    B.tube(st, [0.1, 0.2, 0.18, 0.06], k=0.2, cut=True)
    for sx in (-1, 1):
        pu, _ = B.snap(np.array([(sx * 0.15, 6.9, 3.0), (sx * 0.8, 6.85, 3.0), (sx * 1.45, 7.05, 2.8), (sx * 1.85, 7.45, 2.2)]))
        B.tube(pu, [0.08, 0.17, 0.15, 0.05], k=0.2, cut=True)
    smooth = K.sdf()           # undisplaced body: armour plates hug this, not the fur clumps
    smooth.ops = list(B.ops)
    smooth._lo, smooth._hi = B._lo.copy(), B._hi.copy()
    B.disp(fur, band=0.18)

    mouth = S.Ell((0, 8.1, 3.38), (0.6, 0.27, 0.6), rot=(12, 0, 0))
    K.skin(B, voxel=0.05, tris=20000, belly=belly, regions=[('mouth', lambda P: mouth.dist(S.Points(P)))])

    # ---- skeleton
    K.bone('pelvis', (0, 4.8, -2.1), (0, 6.0, -1.4), None, 1.8)
    K.bone('spine', (0, 6.0, -1.4), (0, 7.3, -0.3), 'pelvis', 2.0)
    K.bone('chest', (0, 7.3, -0.3), (0, 8.8, 0.5), 'spine', 2.0)
    K.bone('neck', (0, 8.8, 0.7), (0, 9.1, 1.9), 'chest', 0.9)
    K.bone('head', (0, 9.1, 1.9), (0, 9.1, 3.6), 'neck', 0.85)
    K.bone('jaw', (0, 8.0, 2.3), (0, 7.62, 3.5), 'head', 0.45)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        K.bone('upperarm' + sd, m * SH, m * EL, 'chest', 1.0)
        K.bone('forearm' + sd, m * EL, m * WR, 'upperarm' + sd, 0.95)
        K.bone('hand' + sd, m * WR, m * (FI + (0, -0.6, 0.7)), 'forearm' + sd, 0.9)
        K.bone('thigh' + sd, m * HIP, m * KN, 'pelvis', 1.15)
        K.bone('shin' + sd, m * KN, m * AN, 'thigh' + sd, 0.75)
        K.bone('foot' + sd, m * AN, m * (1.95, 0.3, 0.0), 'shin' + sd, 0.5)
        K.mount('hand' + sd, m * FI, 'hand' + sd)
        K.mount('shoulder' + sd, m * (SH + (0, 1.1, -0.1)), 'chest')
    K.mount('head', (0, 9.3, 2.4), 'head')
    K.mount('jaw', (0, 7.9, 2.7), 'jaw')
    K.mount('mouth', (0, 8.1, 3.85), 'head')
    K.mount('chest', (0, 7.5, 2.6), 'chest')
    K.mount('back', (0, 8.1, -1.9), 'spine')
    K.mount('tailTip', (0, 4.6, -3.5), 'pelvis')
    K.mount('crown', (0, 10.6, 2.0), 'head')

    # ---- eyes + teeth along the mouth arc
    for s in (-1, 1):
        K.eye('body', (s * 0.42, 9.1, 3.2), (s * 0.3, 0.0, 1), 0.125, bone='head')
    for a in np.radians([-62, -42, -16, 16, 42, 62]):
        x, z = np.sin(a) * 0.52, 3.38 + np.cos(a) * 0.5
        canine = abs(abs(np.degrees(a)) - 42) < 1
        ln = 0.42 if canine else 0.16
        r = 0.09 if canine else 0.06
        K.horn('body', [(x, 8.4, z), (x * 1.02, 8.4 - ln * 0.6, z + 0.02), (x, 8.4 - ln, z)], [r, r * 0.7, 0], sides=6, bone='head', shade=(0.75, 1))
        ln = 0.34 if canine else 0.12
        K.horn('body', [(x * 0.92, 7.86, z - 0.04), (x * 0.94, 7.86 + ln * 0.6, z - 0.03), (x * 0.92, 7.86 + ln, z - 0.06)], [r * 0.95, r * 0.65, 0], sides=6, bone='jaw', shade=(0.75, 1))

    # ---- Might: knuckle studs -> bony gauntlet fists, tusks, horns (exclusive sets)
    for t, sc in enumerate(POW_SCALE):
        if t == 0:
            continue
        part = f'pow_t{t}'
        for s, sd in ((-1, 'L'), (1, 'R')):
            m = M(s)
            hb = 'hand' + sd
            for dx in KNUCKLE_X:
                p, n = K.snap(m * (FI + (dx, -0.45, 0.85)))
                d = n * 0.6 + np.array([0, 0.25, 0.75])
                K.spike(part, p + n * (0.2 if t >= 2 else 0.0), d, 0.2 + 0.28 * (sc - 1), r=0.1 + 0.03 * sc, bone=hb)
            if t >= 2:
                K.plate(part, m * (FI + (0, -0.15, 0.85)), (1.05, 0.5, 0.55), off=0.09, th=0.11, bone=hb, lames=4, lame_axis=(1, 0, 0), lame_step=0.06, sdf=smooth)
                K.plate(part, m * (FI + (0, 0.7, 0.2)), (0.95, 0.55, 0.9), off=0.09, th=0.09, bone=hb, lames=2, lame_axis=(0, 0.4, 1), sdf=smooth)
            if t >= 3:
                p, n = K.snap(m * (FI + (0.85, 0.3, 0.0)))
                K.spike(part, p, n + [0, 0.3, 0], 0.55, bone=hb)
                p, n = K.snap(m * (WR + (0.6, 0.6, -0.3)))
                K.horn(part, [p - n * 0.1, p + n * 0.4 + [0, 0.3, -0.3], p + n * 0.6 + [0, 0.9, -0.8]], [0.16, 0.1, 0], bone='forearm' + sd, per=3)
            # tusks from the lower jaw
            tl = [0, 0.35, 0.6, 0.85][t]
            base = np.array([s * 0.52, 7.72, 3.2])
            K.horn(part, [base, base + [s * 0.12, tl * 0.6, tl * 0.25], base + [s * 0.25, tl * 1.05, tl * 0.15], base + [s * 0.32, tl * 1.25, -tl * 0.15]],
                   [0.1 + 0.02 * t, 0.085, 0.05, 0], bone='jaw', per=3, sides=7)
            if t >= 3:
                p, n = K.snap((s * 0.75, 9.55, 2.6))
                K.horn(part, [p - n * 0.1, p + [s * 0.4, 0.25, 0.1], p + [s * 0.85, 0.35, 0.35], p + [s * 1.05, 0.75, 0.7]], [0.2, 0.15, 0.08, 0], bone='head', per=3, sides=7)

    # ---- Armour: pauldrons, bracers, back + thigh plates; tier 3 spikes
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        K.plate('arm_t1', m * (SH + (0.35, 0.55, 0)), (1.25, 1.0, 1.3), bone='upperarm' + sd, lames=3, sdf=smooth, off=0.09)
        K.plate('arm_t1', m * (3.9, 4.0, 1.7), (0.85, 1.4, 0.85), bone='forearm' + sd, lames=3, sdf=smooth, off=0.09)
        K.plate('arm_t2', m * (2.05, 2.75, -0.2), (0.8, 0.85, 0.7), bone='shin' + sd, lames=2, sdf=smooth, off=0.09)
        K.plate('arm_t2', m * (1.3, 8.7, -0.6), (1.1, 0.9, 1.2), bone='chest', lames=3, lame_axis=(0, 0.4, -1), sdf=smooth, off=0.09)
        for off, ln in (((0.2, 1.05, 0.0), 0.75), ((0.75, 0.55, 0.2), 0.55), ((0.4, 0.75, -0.6), 0.5)):
            p, n = K.snap(m * (SH + off))
            K.spike('arm_t3', p + n * 0.08, n + np.array([s * 0.2, 0.5, -0.1]), ln, bone='upperarm' + sd)
        p, n = K.snap(m * (4.25, 4.3, 1.45))
        K.spike('arm_t3', p + n * 0.08, n + [0, 0.2, -0.3], 0.5, bone='forearm' + sd)
        p, n = K.snap(m * (2.05, 2.75, 0.2))
        K.spike('arm_t3', p + n * 0.12, n + [0, 0.35, 0], 0.5, bone='shin' + sd)
        p, n = K.snap(m * (EL + (0.2, 0.1, -0.7)))
        K.spike('arm_t3', p, n + [0, -0.3, -0.6], 0.6, bone='forearm' + sd)
    K.plate('arm_t2', (0, 10.15, 2.0), (0.6, 0.6, 1.3), bone='head', lames=3, lame_axis=(0, 0, 1), sdf=smooth, off=0.09)
    for i, (y, z) in enumerate(((9.6, -0.3), (8.9, -1.1), (7.9, -1.8))):
        p, n = K.snap((0, y, z))
        K.spike('arm_t3', p + n * 0.05, n + [0, 0.3, -0.4], 0.7 - 0.1 * i, bone='chest')

    # ---- Energy: glowing amber crystals breaking through the fur
    for (c, h, bone, tier) in [((0.55, 9.75, -0.1), 0.85, 'chest', 1), ((-0.55, 9.75, -0.1), 0.8, 'chest', 1),
                               ((0.0, 9.3, -0.75), 0.95, 'chest', 1),
                               ((1.55, 9.2, 0.15), 0.55, 'chest', 2), ((-1.55, 9.2, 0.15), 0.55, 'chest', 2),
                               ((0.65, 8.0, -1.6), 0.7, 'spine', 2), ((-0.65, 8.0, -1.6), 0.7, 'spine', 2),
                               ((4.25, 4.7, 1.1), 0.55, 'forearmR', 3), ((-4.25, 4.7, 1.1), 0.55, 'forearmL', 3),
                               ((0.0, 7.0, -2.15), 0.75, 'spine', 3), ((0.0, 10.35, 1.6), 0.5, 'head', 3)]:
        p, n = K.snap(c)
        K.crystal(f'enr_t{tier}', p - n * 0.08, n + np.array([0, 0.5, -0.2]), h, bone=bone)
    # ---- Speed: wind-swept fur tufts (forearms, calves) and a shoulder mane
    def tuft(part, c, back, n_spikes, ln, bone, seed=0, spread=0.35):
        rng = np.random.default_rng(seed)
        p0, n0 = K.snap(c)
        for i in range(n_spikes):
            off = rng.normal(0, spread, 3) * [1, 0.6, 0.6]
            p, n = K.snap(p0 + off)
            d = np.asarray(back, float) + n * 0.45
            L = ln * rng.uniform(0.75, 1.1)
            K.horn(part, [p - n * 0.15, p + d * L * 0.45 + n * 0.1, p + d * L + [0, -0.25 * L, 0]], [0.2, 0.15, 0], mat='mane', sides=6, per=3, bone=bone, shade=(0.35, 1.0))
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        tuft('spd_t1', m * (4.2, 3.7, 1.35), (s * 0.5, -0.2, -1), 5, 0.75, 'forearm' + sd, seed=1 + s)
        tuft('spd_t2', m * (2.15, 1.8, -2.1), (s * 0.3, -0.3, -1), 4, 0.6, 'shin' + sd, seed=5 + s)
        tuft('spd_t2', m * (EL + (0.3, -0.1, -0.7)), (s * 0.4, -0.5, -1), 4, 0.7, 'forearm' + sd, seed=8 + s)
        tuft('spd_t3', m * (SH + (-0.4, 1.0, -0.6)), (s * 0.5, -0.1, -1), 7, 1.0, 'chest', seed=12 + s, spread=0.45)
    tuft('spd_t3', (0, 10.0, 0.9), (0, -0.2, -1), 6, 0.95, 'head', seed=20, spread=0.3)

    # ---- Focus: extra eyes on the brow, crest sensor-blade, third eye
    for s in (-1, 1):
        K.eye('fcs_t1', (s * 0.72, 9.45, 2.98), (s * 0.6, 0.3, 0.8), 0.075, bone='head', lid=False)
        K.eye('fcs_t3', (s * 0.9, 9.0, 2.6), (s * 1, 0, 0.4), 0.08, bone='head', lid=False)
    K.leaf('fcs_t2', (0, 10.55, 1.95), (0, 1, -0.1), (0, 0, 1), 0.6, 1.3, th=0.06, lobes=3, mat='accent', bone='head')
    K.eye('fcs_t3', (0, 9.66, 3.12), (0, 0.4, 1), 0.1, bone='head', lid=False)
