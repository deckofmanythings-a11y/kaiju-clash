"""Arachnid: giant spider. Low, wide, eight arched legs, glowing hourglass. Game coords, faces +Z."""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..materials import hex_rgb
from ..noise import fbm, smoothstep
from .mantid import fan_wing

DETAIL = 'chitin'
UV_SCALE = 0.7
EYE_ACCENT = True
CAM = dict(az=30, el=22)
T3_SCALE = 1.3

CEPH = np.array([0, 3.45, 1.05])                     # cephalothorax centre
ABD = np.array([(0, 3.85, -1.45), (0, 4.45, -2.6), (0, 4.95, -3.9), (0, 5.05, -5.0), (0, 4.6, -5.95)])
ABD_R = np.array([0.7, 1.95, 2.35, 2.0, 0.55])
ABD_S = (1.0, 0.9, 1.0)
# legs: yaw angle from +Z, length scale, knee height
LEGS = [(0.62, 1.1, 2.8), (1.15, 1.0, 2.75), (1.95, 0.93, 2.55), (2.5, 1.06, 2.7)]


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def leg_pts(k, s):
    a, L, kh = LEGS[k]
    d = np.array([s * np.sin(a), 0, np.cos(a)])
    hip = CEPH + np.array([s * np.sin(a) * 1.15, -0.2, np.cos(a) * 1.3])
    cox = hip + d * 0.62 + [0, 0.32, 0]
    knee = hip + d * 2.75 * L + [0, kh, 0]
    ank = knee + d * 2.55 * L + [0, -kh - 0.85, 0]
    ank[1] = max(ank[1], 1.75)
    foot = ank + d * 1.55 * L
    foot[1] = 0.08
    fem_mid = cox * 0.45 + knee * 0.55 + [0, 0.55, 0] + d * -0.1
    tib_mid = knee * 0.5 + ank * 0.5 + d * 0.25 + [0, 0.25, 0]
    met_mid = ank * 0.5 + foot * 0.5 + d * 0.12 + [0, 0.12, 0]
    return dict(hip=hip, cox=cox, knee=knee, ank=ank, foot=foot, fm=fem_mid, tm=tib_mid, mm=met_mid, d=d)


def build(K):
    body_c, belly_c, acc = (hex_rgb(K.colors[k]) for k in ('body', 'belly', 'accent'))
    K.mat('body', rough=0.22, detail_strength=0.3)
    K.mat('belly', rough=0.4, detail_strength=0.35)
    K.mat('plate', color=np.array([0.2, 0.19, 0.2]) + belly_c * 0.15, rough=0.2, metal=0.55)
    K.mat('horn', color=np.array([0.1, 0.08, 0.075]) + belly_c * 0.25, rough=0.18)
    K.mat('eye', color=acc * 0.5, emit=acc, emit_strength=1.6, rough=0.05, vcol=True)
    K.mat('membrane', color=body_c * 0.5 + acc * 0.28, rough=0.5)

    B = K.sdf()
    # ---------------------------------------------------------------- cephalothorax (carapace + raised head)
    B.ell(CEPH, (1.62, 0.95, 1.9), k=0.1)
    B.ell(CEPH + [0, 0.42, 0.95], (1.05, 0.6, 1.05), k=0.35)                          # head region
    B.ell(CEPH + [0, -0.45, 0.1], (1.35, 0.55, 1.55), k=0.25)                         # sternum mass
    B.ell(CEPH + [0, 1.0, -0.3], (0.12, 0.08, 0.2), cut=True, k=0.12)                # fovea pit
    # ---------------------------------------------------------------- pedicel + abdomen
    B.tube([CEPH + [0, 0.05, -1.6], (0, 3.7, -1.5)], [0.42, 0.5], k=0.12)
    B.tube(ABD, ABD_R, k=0.15, s=ABD_S, per=8)
    for x, z in ((0.22, -6.05), (-0.22, -6.05), (0, -6.15)):                           # spinneret mounds
        B.ell((x, 4.3, z), (0.2, 0.22, 0.28), k=0.08)
    # ---------------------------------------------------------------- chelicerae + pedipalps
    for s in (-1, 1):
        B.tube([CEPH + [s * 0.4, 0.55, 1.85], CEPH + [s * 0.46, 0.0, 2.25], CEPH + [s * 0.4, -0.7, 2.35]], [0.42, 0.44, 0.3], k=0.08, s=(0.95, 1, 0.95))
    PALP = []
    for s in (-1, 1):
        p0 = CEPH + [s * 0.85, -0.35, 1.75]
        pts = [p0, p0 + [s * 0.45, 0.75, 0.75], p0 + [s * 0.7, 0.55, 1.5], p0 + [s * 0.75, -0.35, 2.05], p0 + [s * 0.7, -0.95, 2.35]]
        PALP.append(pts)
        B.tube(pts[:3], [0.2, 0.2, 0.17], k=0.05)
        B.ell(pts[2], (0.17, 0.17, 0.17), k=0.03)
        B.tube(pts[2:], [0.16, 0.17, 0.13], k=0.04)
    # ---------------------------------------------------------------- eight legs
    LP = {}
    for k in range(4):
        for s, sd in ((-1, 'L'), (1, 'R')):
            P = leg_pts(k, s)
            LP[(k, sd)] = P
            sc = LEGS[k][1]
            B.ell(P['hip'], (0.36, 0.34, 0.36), k=0.1)                                    # coxa
            B.tube([P['hip'], P['cox']], [0.36, 0.33], k=0.06)
            B.ell(P['cox'], (0.3, 0.3, 0.3), k=0.03)                                      # trochanter
            B.tube([P['cox'], P['fm'], P['knee']], [0.27 * sc, 0.3 * sc, 0.23 * sc], k=0.04)
            B.ell(P['knee'], (0.29 * sc,) * 3, k=0.03)                                   # patella knob
            B.tube([P['knee'], P['tm'], P['ank']], [0.22 * sc, 0.22 * sc, 0.16 * sc], k=0.035)
            B.ell(P['ank'], (0.19 * sc,) * 3, k=0.03)
            B.tube([P['ank'], P['mm'], P['foot']], [0.15 * sc, 0.13 * sc, 0.085], k=0.03)

    # ---------------------------------------------------------------- sculpt detail: radial carapace grooves, hair stipple, abdomen folds
    def detail(x, y, z):
        th = np.abs(np.arctan2(x, z - CEPH[2]))
        r = np.hypot(x, z - CEPH[2])
        top = smoothstep(CEPH[1] + 0.1, CEPH[1] + 0.6, y) * smoothstep(-1.5, -0.9, z)
        g = 0
        for gc in (0.72, 1.5, 2.25, 2.95):
            g = g + np.exp(-((th - gc) / 0.08) ** 2)
        groove = 0.05 * g * smoothstep(0.25, 0.6, r) * (1 - smoothstep(1.4, 1.9, r)) * top
        hair = -0.025 * fbm(x, y, z, 2, freq=4.5, seed=11, ridged=True) - 0.015 * fbm(x, y, z, 3, freq=1.2, seed=2)
        ab = smoothstep(-1.6, -2.2, z)
        folds = 0.025 * ab * (1 - smoothstep(0.0, 0.12, np.abs(np.sin((x * 0.9) * np.pi)))) * smoothstep(4.5, 3.2, y)
        return groove + hair + folds
    B.disp(detail, band=0.12)

    # ---------------------------------------------------------------- material regions: dark-red underside, glowing hourglass + chevrons
    belly = K.sdf()
    belly.add(S.Inter(S.Tube(ABD, ABD_R * 1.05, s=ABD_S), S.Box((0, 2.6, -3.7), (3, 1.2, 3.5), rot=(-8, 0, 0)), 0), 0)
    belly.ell(CEPH + [0, -0.7, 0.1], (1.1, 0.4, 1.35), k=0)

    def marks(P):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        v = z + 3.25
        hw = 0.1 + 0.75 * np.clip(np.abs(v) / 1.5, 0, 1) ** 1.2
        hg = np.maximum.reduce([np.abs(x) - hw, np.abs(v) - 1.5, 6.1 - y])
        out = hg
        for vc in (-2.1, -2.65, -3.15):
            ch = np.maximum.reduce([np.abs(v - vc + 0.35 * np.abs(x)) - 0.1, np.abs(x) - (0.95 + (vc + 1.9) * 0.35), 5.0 - y])
            out = np.minimum(out, ch)
        return out
    K.skin(B, voxel=0.045, tris=19500, belly=belly, regions=[('accent', marks)])

    # ---------------------------------------------------------------- skeleton
    K.bone('pelvis', CEPH + [0, 0, -1.2], CEPH + [0, 0.2, 1.6], None, 1.5)
    K.bone('spine', CEPH + [0, 0.1, -1.5], ABD[1], 'pelvis', 0.6)
    K.bone('abdomen', ABD[1], ABD[4], 'spine', 2.0)
    K.bone('head', CEPH + [0, 0.5, 0.9], CEPH + [0, 0.9, 2.1], 'pelvis', 0.8)
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.bone('chel' + sd, CEPH + [s * 0.43, 0.5, 1.9], CEPH + [s * 0.42, -0.75, 2.38], 'head', 0.4)
        p = PALP[0 if s < 0 else 1]
        K.chain('palp' + sd, [p[0], p[2], p[4]], 'pelvis', (0.2, 0.16))
        for k in range(4):
            P = LP[(k, sd)]
            K.chain(f'leg{k + 1}{sd}', [P['hip'], P['cox'], P['knee'], P['ank'], P['foot']], 'pelvis', (0.36, 0.17))
    K.mount('head', CEPH + [0, 0.85, 1.6], 'head')
    K.mount('jaw', CEPH + [0, -0.3, 2.45], 'head')
    K.mount('mouth', CEPH + [0, -0.9, 2.6], 'head')
    K.mount('chest', CEPH + [0, -0.2, 2.2], 'pelvis')
    K.mount('back', (0, 7.1, -3.6), 'abdomen')
    K.mount('tailTip', (0, 4.25, -6.45), 'abdomen')
    K.mount('crown', CEPH + [0, 1.35, 1.1], 'head')
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.mount('hand' + sd, PALP[0 if s < 0 else 1][4], 'palp' + sd + '2')
        K.mount('shoulder' + sd, LP[(0, sd)]['cox'] + [0, 0.35, 0], f'leg1{sd}1')

    # ---------------------------------------------------------------- eyes (8, glowing, in a cluster on the head)
    EYES = [(0.28, 0.62, 1.95, 0.25), (0.68, 0.48, 1.7, 0.17), (0.17, 0.95, 1.75, 0.14), (0.52, 0.88, 1.45, 0.13)]
    for s in (-1, 1):
        for x, y, z, r in EYES:
            c = CEPH + [s * x, y, z]
            q, n = K.snap(c)
            c = q + n * r * 0.35
            Ve, Te, _ = geom.sphere(c, r, 14, 9)
            dn = (Ve - c) / r
            K.piece('body', Ve, Te, 'eye', col=0.35 + 0.65 * np.clip(dn @ n, 0, 1) ** 0.7, bone='head')
            x, _, z = geom.basis(n)
            V2, T2, _ = geom.sphere((0, 0, 0), 1, 12, 6)
            V2 = c - n * r * 0.55 + (V2[:, [0]] * x * 1.22 + V2[:, [1]] * n * 0.5 + V2[:, [2]] * z * 1.22) * r
            K.piece('body', V2, T2, 'dark', bone='head')
    # ---------------------------------------------------------------- spinnerets, bristles, tarsal claws
    for x, z, dy in ((0.22, -6.15, 0.0), (-0.22, -6.15, 0.0), (0, -6.3, 0.12)):
        K.horn('body', [(x, 4.25 + dy, z), (x * 1.3, 4.15 + dy, z - 0.3), (x * 1.5, 4.0 + dy, z - 0.5)], [0.11, 0.08, 0.03], mat='body', sides=6, bone='abdomen')
    rng = np.random.default_rng(3)
    for (k, sd), P in LP.items():
        s = -1 if sd == 'L' else 1
        nm = f'leg{k + 1}{sd}'
        for seg, (a, b, r0, bone) in enumerate(((P['cox'], P['knee'], 0.3, nm + '2'), (P['knee'], P['ank'], 0.24, nm + '3'), (P['ank'], P['foot'], 0.15, nm + '4'))):
            n_br = (7, 6, 4)[seg]
            for i in range(n_br):
                f = (i + 0.5) / n_br
                p = a + (b - a) * f
                ax = unit(b - a)
                ang = rng.uniform(0, 2 * np.pi)
                side = unit(np.cross(ax, [0, 1, 0]))
                up = unit(np.cross(side, ax))
                rd = unit(np.cos(ang) * side + np.sin(ang) * up * 0.8 + up * 0.5)
                base = p + rd * r0 * 0.85
                ln = (0.35, 0.3, 0.22)[seg] * rng.uniform(0.75, 1.2)
                tip = base + unit(rd + ax * 1.1) * ln
                K.horn('body', [base - rd * 0.03, base * 0.5 + tip * 0.5 + rd * 0.03, tip], [0.028, 0.016, 0], mat='dark', sides=3, per=1, bone=bone)
        d = P['d']
        for sp in (-0.3, 0.3):
            K.claw('body', P['foot'], d + np.cross(d, [0, 1, 0]) * sp + [0, -0.4, 0], 0.32, 0.055, curl=(0, -1, 0), mat='horn', bone=nm + '4')
    for i in range(26):   # abdomen bristles
        ang = rng.uniform(0, 2 * np.pi)
        f = rng.uniform(0.15, 0.85)
        c = S.catmull(ABD, 4)[int(f * 16)]
        q, n = K.snap(c + np.array([np.cos(ang), np.sin(ang), 0]) * 3)
        if n[1] < -0.4:
            continue
        K.horn('body', [q - n * 0.03, q + n * 0.15 + [0, 0, -0.08], q + n * 0.3 + [0, 0, -0.22]], [0.03, 0.018, 0], mat='dark', sides=3, per=1, bone='abdomen')

    # ---------------------------------------------------------------- fangs per Might tier (exclusive sets)
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        for s, sd in ((-1, 'L'), (1, 'R')):
            base = CEPH + [s * 0.4, -0.72, 2.4]
            ln = 1.0 * sc
            K.horn(part, [base + [0, 0.08, -0.05], base + [-s * 0.05, -ln * 0.5, ln * 0.12], base + [-s * 0.2, -ln * 0.88, -0.02], base + [-s * 0.32, -ln * 1.0, -ln * 0.28]],
                   [0.14 * (0.8 + 0.2 * sc), 0.11, 0.06, 0.0], mat='horn', sides=8, per=3, bone='chel' + sd)
            if t >= 1:   # palp hooks
                p = PALP[0 if s < 0 else 1][4]
                K.claw(part, p, (s * 0.1, -0.6, 1), 0.3 + 0.15 * sc, 0.07, curl=(0, -1, 0), bone='palp' + sd + '2')
            if t >= 2:   # chelicera teeth
                for i, f in enumerate((0.3, 0.6)):
                    q, n = K.snap(CEPH + [s * 0.46, 0.4 - f * 1.0, 2.6])
                    K.spike(part, q, n + [0, 0, 0.5], 0.2 + 0.12 * sc, r=0.06, mat='horn', bone='chel' + sd)
            if t >= 3:   # front-leg talons
                P = LP[(0, sd)]
                K.claw(part, P['knee'] + [0, 0.2, 0], P['d'] * 0.3 + [0, 1, 0.4], 0.75, 0.12, curl=(0, 0, 1), bone=f'leg1{sd}2')

    # ---------------------------------------------------------------- Armor: carapace shield, abdomen plates, then spikes
    K.plate('arm_t1', CEPH + [0, 1.0, 0.1], (1.4, 0.7, 1.6), bone='pelvis', lames=3, lame_axis=(0, 0, 1))
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.plate('arm_t2', CEPH + [s * 1.15, 0.35, 0.3], (0.75, 0.6, 1.2), bone='pelvis', lames=2, lame_axis=(0, 0, 1))
    K.plate('arm_t2', (0, 5.5, -1.75), (1.9, 1.2, 0.9), bone='abdomen', lames=3, lame_axis=(0, 0.8, -0.6))
    K.plate('arm_t2', (0, 5.4, -5.6), (1.6, 1.1, 0.8), bone='abdomen', lames=3, lame_axis=(0, -0.6, -0.8))
    for s, sd in ((-1, 'L'), (1, 'R')):
        for z, y in ((-2.3, 6.4), (-3.3, 7.0), (-4.3, 7.0), (-5.15, 6.4)):
            p, n = K.snap((s * 1.3, y, z))
            K.spike('arm_t3', p, n + [0, 0.6, -0.3], 0.75, bone='abdomen')
        for k in range(4):
            P = LP[(k, sd)]
            K.spike('arm_t3', P['knee'] + [0, 0.25, 0], P['d'] * 0.3 + [0, 1, 0], 0.55, r=0.1, bone=f'leg{k + 1}{sd}2')
        p, n = K.snap(CEPH + [s * 1.3, 0.6, 0.0])
        K.spike('arm_t3', p, n + [0, 0.6, 0], 0.5, bone='pelvis')
    for z, y in ((-3.0, 7.3), (-4.0, 7.35)):
        p, n = K.snap((0, y, z))
        K.spike('arm_t3', p, n + [0, 0, -0.4], 0.9, bone='abdomen')

    # ---------------------------------------------------------------- Energy: venom glands -> glowing leg bands -> crystal growths
    for s, sd in ((-1, 'L'), (1, 'R')):
        for z, y in ((-2.6, 4.9), (-3.6, 5.1), (-4.6, 4.95)):
            p, n = K.snap((s * 2.4, y, z))
            K.ball('enr_t1', p - n * 0.04, (0.2, 0.2, 0.2), 'accent', seg=10, rings=6, bone='abdomen')
        p, n = K.snap(CEPH + [s * 0.5, -0.1, 2.5])
        K.ball('enr_t1', p - n * 0.02, 0.12, 'accent', seg=8, rings=6, bone='chel' + sd)
        for k in range(4):
            P = LP[(k, sd)]
            for a, b, r, bn, f in ((P['knee'], P['ank'], 0.27, '3', 0.12), (P['ank'], P['foot'], 0.2, '4', 0.1)):
                ax = unit(b - a)
                c = a + (b - a) * f
                side = unit(np.cross(ax, [0, 1, 0]))
                up = unit(np.cross(side, ax))
                ring = [c + (np.cos(t) * side + np.sin(t) * up) * r * LEGS[k][1] for t in np.linspace(0, 2 * np.pi, 13)]
                K.horn('enr_t2', ring, [0.045] * 13, mat='accent', sides=4, per=1, bone=f'leg{k + 1}{sd}{bn}')
    for i, (x, y, z) in enumerate(((0.0, 7.2, -3.3), (0.7, 6.9, -4.3), (-0.7, 6.9, -4.3), (0, 6.6, -5.1), (0.0, 7.1, -2.4))):
        p, n = K.snap((x, y, z))
        K.crystal('enr_t3', p - n * 0.08, n + [0, 0.2, -0.2], 0.85 - 0.08 * i, bone='abdomen', seed=i)
    for x in (0.22, -0.22, 0.0):
        K.ball('enr_t3', (x * 1.5, 4.0 + (0.12 if x == 0 else 0), -6.65), 0.1, 'accent', seg=8, rings=6, bone='abdomen')

    # ---------------------------------------------------------------- Speed: leg bristle tufts -> side frills -> peacock fan
    for s, sd in ((-1, 'L'), (1, 'R')):
        for k in range(4):
            P = LP[(k, sd)]
            c = P['knee'] * 0.55 + P['ank'] * 0.45
            K.frill('spd_t1', c + [0, 0.15, 0], P['d'] * 0.5 + [0, 1, 0], -P['d'] + [0, 0.5, 0], 0.55, 0.45, rays=4, spread=70, bone=f'leg{k + 1}{sd}3', ray_mat='dark')
        p, n = K.snap((s * 2.2, 5.6, -3.4))
        K.frill('spd_t2', p, n + [0, 0.3, 0], (0, 0.2, -1), 1.1, 2.2, rays=6, spread=60, bone='abdomen', ray_mat='dark')
    p, n = K.snap((0, 6.9, -4.7))
    fan_wing(K, 'spd_t3', p + [0, -0.15, 0.1], (-1, 0.45, -0.35), (1, 0.45, -0.35), 3.0, veins=11, cup=-0.12, bone='abdomen')
    for x, y in ((-1.25, 1.55), (0, 2.25), (1.25, 1.55)):
        K.ball('spd_t3', p + [x, y + 0.1, -0.95 - 0.1 * (x == 0)], (0.26, 0.26, 0.07), 'accent', rot=(-20, 0, 0), bone='abdomen')

    # ---------------------------------------------------------------- Focus: extra eyes, sensory hairs, head crest
    for s in (-1, 1):
        for x, y, z, r, part in ((0.92, 0.32, 1.35, 0.1, 'fcs_t1'), (0.36, 1.05, 1.25, 0.09, 'fcs_t1'), (0.78, 0.72, 1.0, 0.1, 'fcs_t3'), (0.12, 0.3, 2.05, 0.08, 'fcs_t3')):
            q, n = K.snap(CEPH + [s * x, y, z])
            K.ball(part, q + n * r * 0.3, r, 'eye', seg=10, rings=6, bone='head')
        for k in (0, 1):
            P = LP[(k, sd if False else ('L' if s < 0 else 'R'))]
            c = P['knee'] * 0.5 + P['ank'] * 0.5
            K.horn('fcs_t2', [c, c + [0, 0.8, 0] + P['d'] * 0.3, c + [0, 1.6, 0] + P['d'] * 0.8], [0.035, 0.02, 0.0], mat='accent', sides=4, bone=f'leg{k + 1}{"L" if s < 0 else "R"}3')
        K.horn('fcs_t3', [CEPH + [s * 0.32, 1.0, 1.15], CEPH + [s * 0.45, 1.55, 0.85], CEPH + [s * 0.5, 1.95, 0.4]], [0.1, 0.07, 0.0], mat='accent', bone='head')
    K.leaf('fcs_t2', CEPH + [0, 1.08, 0.75], (0, 1, -0.2), (0, 0, -1), 0.7, 1.0, th=0.05, lobes=3, mat='accent', bone='head')
