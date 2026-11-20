"""Magma Golem: hulking living volcano of cracked basalt (golem plan). Game coords, ~10.5 units tall, faces +Z."""
import numpy as np

from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep, voronoi

DETAIL = 'rock'
UV_SCALE = 0.75
EYE_ACCENT = True

SH = np.array([3.15, 8.3, 1.21])     # shoulder pivot (R)
EL = np.array([3.75, 5.6, 1.02])     # elbow
WR = np.array([3.95, 3.4, 1.6])      # wrist
FI = np.array([4.0, 2.4, 1.85])    # fist centre
HIP = np.array([1.55, 4.5, 0.0])
KN = np.array([1.85, 2.55, 0.2])
AN = np.array([1.95, 0.9, 0.3])
HEAD = np.array([0, 9.35, 3.17])
MOUTH = HEAD + (0, -0.38, 0.62)
CRACK = 1.25                        # basalt slab size


def M(s):
    return np.array([s, 1, 1])


def build(K):
    K.mat('horn', color='#151213', rough=0.22, metal=0.1)                      # obsidian
    K.mat('plate', color='#3d3632', detail=True, detail_kind='rock', rough=0.85, metal=0.0)
    K.mat('accent', emit_strength=4.0)
    B = K.sdf()
    # ---- torso: a leaning slab-sided mountain
    B.ell((0, 7.1, 0.55), (2.1, 2.2, 1.75), rot=(22, 0, 0), k=0.0)
    B.box((0, 7.7, 1.39), (1.7, 1.15, 0.85), rot=(16, 0, 0), rad=0.35, k=0.35)          # chest slab
    B.box((0, 8.7, 0.0), (1.9, 0.95, 1.35), rot=(-8, 0, 0), rad=0.4, k=0.45)           # hunched upper back
    B.box((0, 4.95, 0.0), (1.4, 0.9, 1.15), rot=(0, 0, 0), rad=0.4, k=0.45)             # pelvis block
    B.box((0, 6.0, 1.10), (1.3, 0.9, 0.8), rot=(-8, 0, 0), rad=0.35, k=0.35)              # gut slab
    # shoulder boulders: huge, rising above the head
    B.box((2.8, 9.45, 1.29), (1.35, 1.15, 1.35), rot=(12, 18, 24), rad=0.38, k=0.45, mirror=True)
    B.ell((2.55, 8.85, 1.12), (1.35, 1.25, 1.35), k=0.45, mirror=True)
    B.box((1.95, 6.7, -0.21), (0.75, 0.95, 0.85), rot=(20, 30, 10), rad=0.3, k=0.35, mirror=True)   # flank boulders
    B.ell((0, 9.0, -1.15), (1.7, 1.2, 1.3), rot=(-15, 0, 0), k=0.45)                                 # volcanic hump
    B.box((1.55, 9.6, 0.57), (0.85, 0.65, 0.95), rot=(5, 25, 14), rad=0.3, k=0.4, mirror=True)   # trapezius slabs
    # ---- arms
    B.tube([SH, (3.6, 6.95, 1.14), EL], [1.0, 0.95, 0.72], k=0.3, mirror=True)
    B.box((3.75, 7.0, 1.10), (0.85, 1.0, 0.85), rot=(8, 15, 12), rad=0.3, k=0.3, mirror=True)       # upper-arm slab
    B.tube([EL, (3.87, 4.45, 1.25), WR], [0.72, 1.0, 0.95], k=0.25, mirror=True)
    B.box((3.87, 4.45, 1.25), (0.95, 0.95, 0.9), rot=(18, 10, -6), rad=0.3, k=0.3, mirror=True)      # forearm slab
    B.box(FI, (1.05, 0.95, 1.1), rot=(12, 18, 6), rad=0.4, k=0.35, mirror=True)                     # boulder fist
    B.ell(FI + (0, -0.05, 0.1), (1.15, 1.0, 1.15), k=0.3, mirror=True)
    for i, dx in enumerate((-0.55, -0.18, 0.18, 0.55)):
        B.box(FI + (dx, -0.35 - 0.04 * i, 1.0), (0.2, 0.3, 0.26), rot=(20, 0, 0), rad=0.1, k=0.12, mirror=True)  # knuckle stones
    # ---- legs: short pillars on slab feet
    B.tube([HIP, (1.72, 3.5, 0.1), KN], [1.15, 1.05, 0.8], k=0.3, mirror=True)
    B.box((1.8, 3.55, 0.12), (0.95, 0.9, 0.95), rot=(0, 12, 8), rad=0.3, k=0.3, mirror=True)       # thigh slab
    B.box((1.92, 1.55, 0.3), (0.95, 1.0, 1.0), rot=(0, -10, -4), rad=0.32, k=0.25, mirror=True)       # shin boulder
    B.box((1.95, 0.38, 0.55), (1.05, 0.38, 1.35), rot=(0, -5, 0), rad=0.16, k=0.25, mirror=True)     # foot slab
    for dx in (-0.55, 0.0, 0.55):
        B.box((1.95 + dx, 0.3, 1.8), (0.24, 0.26, 0.28), rot=(0, dx * 15, 0), rad=0.1, k=0.1, mirror=True)  # toe stones
    # ---- head: small, sunk between the shoulders, heavy brow slab
    B.box(HEAD, (0.72, 0.6, 0.65), rot=(10, 0, 0), rad=0.22, k=0.3)
    B.box(HEAD + (0, 0.45, 0.45), (0.8, 0.17, 0.4), rot=(-20, 0, 0), rad=0.08, k=0.1)       # brow slab
    B.box(HEAD + (0.62, 0.05, 0.1), (0.2, 0.42, 0.45), rot=(0, 0, -10), rad=0.08, k=0.1, mirror=True)  # cheek stones
    B.box(HEAD + (0, -0.6, 0.22), (0.66, 0.32, 0.58), rot=(8, 0, 0), rad=0.12, k=0.15)     # jaw block
    B.ell(MOUTH, (0.5, 0.19, 0.4), cut=True, k=0.05)                                         # furnace mouth
    B.box(HEAD + (0.32, 0.2, 0.6), (0.22, 0.07, 0.2), rot=(0, 0, 16), rad=0.03, cut=True, k=0.03, mirror=True)  # eye slits
    # ---- furnace in the chest
    B.ell((0, 7.55, 2.36), (0.75, 0.62, 0.55), cut=True, k=0.12)

    # ---- glowing regions: crack network + joint seams + furnace + mouth
    joints = []
    for s in (-1, 1):
        joints += [S.Ell(M(s) * EL, (0.95, 0.42, 0.95)), S.Ell(M(s) * KN, (1.0, 0.36, 1.0)),
                   S.Ell(M(s) * (WR + (0, 0.05, 0)), (1.0, 0.25, 1.0)), S.Ell(M(s) * (SH + (-0.4, -0.75, 0)), (0.5, 1.0, 1.2))]
    joints += [S.Ell((0, 5.75, 0.35), (2.3, 0.22, 1.9)), S.Ell(HEAD + (0, -0.25, -0.75), (0.9, 0.35, 0.5))]
    furn = S.Ell((0, 7.55, 2.36), (0.8, 0.67, 0.6))
    mouth = S.Ell(MOUTH, (0.54, 0.22, 0.44))
    headz = S.Ell(HEAD + (0, 0, 0.2), (1.0, 1.0, 1.0))

    def crack_field(x, y, z):
        f1, f2, cid = voronoi(x, y * 0.85, z, CRACK, seed=21)
        return f2 - f1, cid

    def joint_d(P):
        pts = S.Points(P)
        return np.min([j.dist(pts) for j in joints], 0) + 0.1 * fbm(P[:, 0], P[:, 1], P[:, 2], 2, freq=1.3, seed=31)

    def rock(x, y, z):
        P = np.stack([x, y, z], 1)
        e, cid = crack_field(x, y, z)
        jd = joint_d(P)
        crack = (1 - smoothstep(0.0, 0.11, e)) * smoothstep(-0.1, 0.25, headz.dist(S.Points(P)))
        seam = smoothstep(0.12, -0.12, jd)
        slab = -0.08 * cid - 0.025 * fbm(x, y, z, 3, freq=1.6, seed=5, ridged=True)
        return slab + 0.1 * crack + 0.16 * seam
    B.disp(rock, band=0.2)

    def glow(P):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        e, _ = crack_field(x, y, z)
        pts = S.Points(P)
        thr = 0.04 * smoothstep(0.0, 0.35, fbm(x, y, z, 2, freq=0.32, seed=8)) * smoothstep(0.0, 0.3, headz.dist(pts))
        c = np.where(thr > 0.004, e - thr, 1.0)                           # only patches of cracks glow
        return np.minimum.reduce([c, joint_d(P) + 0.02, furn.dist(pts), mouth.dist(pts)])

    belly = K.sdf()       # lighter player-coloured slabs: chest plate, gut, shins
    belly.box((0, 7.7, 2.09), (1.55, 1.05, 0.5), rot=(16, 0, 0), rad=0.3, k=0)
    belly.box((0, 6.0, 1.80), (1.2, 0.75, 0.5), rot=(-8, 0, 0), rad=0.3, k=0)
    for s in (-1, 1):
        belly.box(M(s) * (1.92, 1.55, 1.15), (0.9, 0.95, 0.45), rad=0.2, k=0)
        belly.ell(M(s) * (FI + (0, 0.75, -0.1)), (0.95, 0.5, 0.9), k=0)

    K.skin(B, voxel=0.05, tris=12000, belly=belly, regions=[('accent', glow)], custom_normals=False, sharp=32)

    # ---- skeleton
    K.bone('pelvis', (0, 4.6, 0.0), (0, 5.9, 0.33), None, 1.8)
    K.bone('spine', (0, 5.9, 0.33), (0, 7.3, 0.76), 'pelvis', 2.1)
    K.bone('chest', (0, 7.3, 0.76), (0, 9.0, 0.80), 'spine', 2.2)
    K.bone('neck', (0, 8.60, 1.82), (0, 9.10, 2.72), 'chest', 0.7)
    K.bone('head', (0, 9.10, 2.72), (0, 9.50, 4.00), 'neck', 0.7)
    K.bone('jaw', (0, 8.80, 2.66), (0, 8.70, 3.89), 'head', 0.45)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        K.bone('upperarm' + sd, m * SH, m * EL, 'chest', 1.0)
        K.bone('forearm' + sd, m * EL, m * WR, 'upperarm' + sd, 0.95)
        K.bone('hand' + sd, m * WR, m * (FI + (0, -0.9, 0.3)), 'forearm' + sd, 1.1)
        K.bone('thigh' + sd, m * HIP, m * KN, 'pelvis', 1.1)
        K.bone('shin' + sd, m * KN, m * AN, 'thigh' + sd, 0.95)
        K.bone('foot' + sd, m * AN, m * (1.95, 0.35, 1.9), 'shin' + sd, 0.7)
        K.mount('hand' + sd, m * FI, 'hand' + sd)
        K.mount('shoulder' + sd, m * (2.8, 10.6, 1.22), 'chest')
    K.mount('head', tuple(HEAD), 'head')
    K.mount('jaw', (0, 8.75, 3.35), 'jaw')
    K.mount('mouth', (0, 8.97, 4.09), 'head')
    K.mount('chest', (0, 7.55, 2.71), 'chest')
    K.mount('back', (0, 8.4, -1.02), 'chest')
    K.mount('tailTip', (0, 4.6, -1.5), 'pelvis')
    K.mount('crown', (0, 10.10, 3.32), 'head')

    # ---- eyes (glowing slits), furnace grate + core
    for s in (-1, 1):
        K.ball('body', (s * 0.32, 9.55, 3.76), (0.2, 0.065, 0.1), mat='eye', rot=(0, 0, s * 16), bone='head')
    K.ball('body', (0, 7.55, 2.2), (0.62, 0.52, 0.4), mat='accent', bone='chest')
    for i, y in enumerate((7.22, 7.55, 7.88)):
        w = 0.7 - abs(i - 1) * 0.16
        pl, _ = K.snap((-w - 0.15, y, 3.8))
        pr, _ = K.snap((w + 0.15, y, 3.8))
        z = 0.5 * (pl[2] + pr[2]) - 0.06
        K.horn('body', [(-w - 0.1, y, z - 0.05), (0, y, z + 0.06), (w + 0.1, y, z - 0.05)], [0.075, 0.085, 0.075], mat='horn', sides=5, bone='chest', shade=(0.8, 1))
    for s in (-1, 1):   # stone fangs at the mouth corners
        K.horn('body', [(s * 0.36, 9.15, 3.93), (s * 0.34, 8.98, 3.94), (s * 0.32, 8.84, 3.89)], [0.08, 0.06, 0], mat='horn', sides=4, bone='head')

    # ---- basalt columns on the back + shoulders (base silhouette)
    for c, h, nv in [((0.0, 9.1, -0.58), 1.3, (0, 0.8, -0.6)), ((0.9, 8.4, -0.87), 1.0, (0.3, 0.6, -0.8)), ((-0.9, 8.4, -0.87), 1.0, (-0.3, 0.6, -0.8)),
                     ((2.6, 10.4, 0.78), 1.0, (0.3, 1, -0.3)), ((-2.6, 10.4, 0.78), 1.0, (-0.3, 1, -0.3)), ((0.0, 7.4, -1.37), 0.8, (0, 0.4, -1))]:
        p, n = K.snap(c)
        K.crystal('body', p - n * 0.15, np.asarray(nv, float) + n * 0.3, h, r=h * 0.32, mat='body')

    # ---- Might: obsidian knuckle spikes growing into a spiked boulder + horns (exclusive sets)
    for t, sc in enumerate(POW_SCALE):
        if t == 0:
            continue
        part = f'pow_t{t}'
        for s, sd in ((-1, 'L'), (1, 'R')):
            m = M(s)
            hb = 'hand' + sd
            for i, dx in enumerate((-0.55, -0.18, 0.18, 0.55)):
                p, n = K.snap(m * (FI + (dx, -0.25, 1.4)))
                K.spike(part, p, n * 0.5 + [0, 0.1, 1], 0.25 + 0.3 * (sc - 1), r=0.12 + 0.03 * sc, sides=5, bone=hb)
            if t >= 2:
                for off in ((0.9, 0.2, 0.2), (0.4, 0.85, 0.3), (0.95, -0.4, -0.3)):
                    p, n = K.snap(m * (FI + off))
                    K.spike(part, p, n + [0, 0.2, 0], 0.35 * sc, r=0.16, sides=5, bone=hb)
            if t >= 3:
                p, n = K.snap(m * (FI + (0.2, 0.9, -0.6)))
                K.spike(part, p, n + [0, 0.5, -0.4], 0.8, r=0.2, sides=5, bone=hb)
                p, n = K.snap((s * 0.6, 9.85, 3.07))
                K.horn(part, [p - n * 0.1, p + [s * 0.35, 0.35, 0.05], p + [s * 0.55, 0.8, -0.1], p + [s * 0.5, 1.25, -0.4]], [0.18, 0.14, 0.07, 0], sides=5, per=3, bone='head')
            hl = [0, 0.3, 0.45, 0.6][t]
            p, n = K.snap((s * 0.7, 9.80, 3.71))
            K.horn(part, [p - n * 0.05, p + [s * hl * 0.6, hl * 0.3, hl * 0.4], p + [s * hl * 0.9, hl * 0.9, hl * 0.5]], [0.11, 0.07, 0], sides=5, bone='head')

    # ---- Armour: hewn obsidian slabs (box-clipped shells); tier 3 obsidian spikes
    def slab(part, c, half, rot, bone, lames=2, axis=(0, 1, 0)):
        p, _ = K.snap(c)
        K.plate(part, p, half, off=0.06, th=0.13, mat='horn', bone=bone, lames=lames, lame_axis=axis, snap=False,
                region=S.Box(p, half, rot, rad=0.12))
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        slab('arm_t1', m * (2.9, 10.3, 1.46), (1.25, 0.85, 1.25), (0, s * 20, s * 22), 'chest', 3)
        slab('arm_t1', m * (4.45, 4.5, 1.25), (0.8, 1.1, 0.9), (18, 0, 0), 'forearm' + sd, 3)
        slab('arm_t2', m * (2.7, 3.6, 0.2), (0.8, 0.9, 0.9), (0, 0, s * 8), 'thigh' + sd, 2)
        slab('arm_t2', m * (0.75, 4.95, 1.3), (0.62, 0.75, 0.6), (0, 0, s * -6), 'pelvis', 2)
        for off, ln in (((0.1, 0.75, 0.3), 0.9), ((0.75, 0.35, 0.3), 0.75), ((0.3, 0.55, -0.7), 0.7)):
            p, n = K.snap(m * ((2.9, 10.3, 1.46) + np.array(off)))
            K.spike('arm_t3', p + n * 0.12, n + [s * 0.2, 0.5, 0], ln, r=ln * 0.24, sides=5, bone='chest')
        p, n = K.snap(m * (4.65, 4.6, 1.05))
        K.spike('arm_t3', p + n * 0.12, n + [0, 0.2, -0.3], 0.6, sides=5, bone='forearm' + sd)
        p, n = K.snap(m * (1.92, 2.5, 1.2))
        K.spike('arm_t3', p + n * 0.05, n + [0, 0.3, 0.2], 0.5, sides=5, bone='shin' + sd)
    slab('arm_t2', (0, 9.95, 3.14), (0.85, 0.38, 0.7), (10, 0, 0), 'head', 2, (0, 0, 1))

    # ---- Energy: magma crystals erupting through the basalt
    for c, h, bone, tier in [((1.35, 10.1, 0.32), 0.85, 'chest', 1), ((-1.35, 10.1, 0.32), 0.8, 'chest', 1),
                             ((0.55, 8.0, -1.05), 0.7, 'chest', 2), ((-0.55, 8.0, -1.05), 0.7, 'chest', 2),
                             ((3.6, 9.5, 1.80), 0.65, 'chest', 2), ((-3.6, 9.5, 1.80), 0.65, 'chest', 2),
                             ((4.6, 2.9, 1.1), 0.6, 'handR', 3), ((-4.6, 2.9, 1.1), 0.6, 'handL', 3),
                             ((1.2, 6.2, 1.54), 0.55, 'spine', 3), ((-1.2, 6.2, 1.54), 0.55, 'spine', 3), ((0, 6.0, -1.40), 0.8, 'spine', 3)]:
        p, n = K.snap(c)
        K.crystal(f'enr_t{tier}', p - n * 0.1, n + np.array([0, 0.4, 0]), h, bone=bone)

    # ---- Speed: volcanic vents on the back that vent magma
    def vent(part, c, h, r, bone, tilt=(0, 0, 0)):
        p, n = K.snap(c)
        d = n + np.asarray(tilt, float)
        d = d / np.linalg.norm(d)
        top = p + d * h
        K.horn(part, [p - d * 0.25, p + d * h * 0.55, top], [r, r * 0.8, r * 0.62], mat='plate', sides=7, per=2, bone=bone)
        x = np.cross(d, [0.0, 0, 1]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0, 0])
        x /= np.linalg.norm(x)
        V, T = [top + d * 0.02], []
        for i in range(10):
            a = 2 * np.pi * i / 10
            V.append(top + d * 0.02 + (np.cos(a) * x + np.sin(a) * np.cross(d, x)) * r * 0.55)
        for i in range(10):
            T.append((0, 1 + i, 1 + (i + 1) % 10))
        K.piece(part, np.array(V), np.array(T), mat='accent', bone=bone)
        T2 = [(0, 1 + (i + 1) % 10, 1 + i) for i in range(10)]
        K.piece(part, np.array(V), np.array(T2), mat='accent', bone=bone)
    for s in (-1, 1):
        vent('spd_t1', (s * 0.8, 9.3, -0.49), 0.6, 0.32, 'chest', (s * 0.2, 0.6, -0.5))
        vent('spd_t2', (s * 2.4, 9.6, 0.02), 0.7, 0.34, 'chest', (s * 0.3, 0.8, -0.5))
        vent('spd_t3', (s * 1.1, 6.6, -1.28), 0.8, 0.4, 'spine', (s * 0.3, 0.4, -0.6))
        vent('spd_t3', (s * 1.6, 4.4, -1.1), 0.6, 0.3, 'pelvis', (s * 0.4, 0.2, -0.6))

    # ---- Focus: more glowing eye slits, crystal crest, third eye
    for s in (-1, 1):
        K.ball('fcs_t1', (s * 0.55, 9.72, 3.69), (0.11, 0.04, 0.07), mat='eye', rot=(0, 0, s * 25), bone='head')
        K.ball('fcs_t3', (s * 0.75, 9.30, 3.51), (0.1, 0.04, 0.07), mat='eye', rot=(0, 0, -s * 10), bone='head')
    for i, (x, h) in enumerate(((0, 0.75), (0.35, 0.5), (-0.35, 0.5))):
        p, n = K.snap((x, 10.05, 2.96))
        K.crystal('fcs_t2', p - n * 0.08, n + [x * 0.6, 0.6, -0.2], h, r=h * 0.25, cluster=False, bone='head')
    K.ball('fcs_t3', (0, 9.82, 3.98), (0.1, 0.1, 0.05), mat='eye', bone='head')
