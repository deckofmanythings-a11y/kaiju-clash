"""Scorpion: giant armoured scorpion. Low banded body on eight splayed legs, two raised pincers and a
five-bead tail arching up and forward so the stinger hangs over the head. Game coords, faces +Z."""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..materials import hex_rgb
from ..noise import fbm, smoothstep

INFO = dict(name='Sandreaper', colors=dict(body='#33200f', belly='#c99a4e', accent='#8cff2e'))

DETAIL = 'chitin'
UV_SCALE = 0.7
EYE_ACCENT = True
SCALE = 1.0
CAM = dict(az=30, el=20)

Y0 = 2.5                                              # body centre height
CEPH = np.array([0, Y0 + 0.05, 2.2])                  # carapace centre
# mesosoma tergites: z centre, half-width
TERG = [(0.75 - 0.68 * i, w) for i, w in enumerate((1.42, 1.56, 1.66, 1.7, 1.64, 1.48, 1.1))]
# tail (metasoma): 5 bead segments J0..J5, then the telson vesicle + stinger
TJ = np.array([(0, 2.75, -3.85), (0, 3.95, -5.05), (0, 5.6, -5.6), (0, 7.3, -5.15), (0, 8.7, -4.05), (0, 9.55, -2.55)])
TR = [0.66, 0.62, 0.58, 0.56, 0.58]
VES = np.array([0, 9.6, -1.35])                       # telson vesicle (venom bulb) centre
VES_R = (0.72, 0.66, 0.9)
STING = VES + [0, -0.05, 0.75]                        # stinger root on the vesicle front
# legs: yaw from +Z, length scale, knee lift, hip z
LEGS = [(1.05, 0.92, 1.55, 2.9), (1.45, 1.0, 1.7, 2.25), (1.9, 1.05, 1.7, 1.55), (2.35, 1.12, 1.6, 0.85)]


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def leg_pts(k, s):
    a, L, kh, hz = LEGS[k]
    d = np.array([s * np.sin(a), 0, np.cos(a)])
    hip = np.array([s * 0.95, Y0 - 0.6, hz])
    cox = hip + d * 0.55 + [0, 0.18, 0]
    knee = hip + d * 1.75 * L + [0, kh, 0]
    ank = knee + d * 1.35 * L
    ank[1] = 1.25
    foot = ank + d * 1.15 * L
    foot[1] = 0.1
    fm = cox * 0.45 + knee * 0.55 + [0, 0.35, 0] - d * 0.1
    tm = knee * 0.5 + ank * 0.5 + d * 0.18 + [0, 0.1, 0]
    return dict(hip=hip, cox=cox, knee=knee, ank=ank, foot=foot, fm=fm, tm=tm, d=d)


def palp_pts(s):
    sh = np.array([s * 0.85, Y0 - 0.05, 3.3])
    el = np.array([s * 2.25, Y0 + 1.45, 4.05])
    wr = np.array([s * 2.05, Y0 + 1.6, 5.25])
    pc = np.array([s * 1.95, Y0 + 1.55, 6.1])          # palm centre
    fb = pc + [0, 0.02, 0.85]                           # finger base
    fix = np.array([fb + [s * 0.3, -0.1, -0.15], fb + [s * 0.38, -0.12, 0.65], fb + [s * 0.12, -0.1, 1.4]])
    mov = np.array([fb + [-s * 0.22, 0.06, -0.2], fb + [-s * 0.58, 0.1, 0.55], fb + [-s * 0.44, 0.06, 1.3]])
    return dict(sh=sh, el=el, wr=wr, pc=pc, fb=fb, fix=fix, mov=mov)


def tail_out(a, b):
    """Unit normal in the tail's YZ plane pointing toward the arc centre: the tail curls dorsally, so this is
    the dorsal (keeled) side."""
    u = unit(np.asarray(b) - np.asarray(a))
    n = unit(np.cross(u, [1, 0, 0]))
    c = (np.asarray(a) + np.asarray(b)) * 0.5
    return -n if n @ (c - np.array([0, 6.0, -2.6])) > 0 else n


def ring(K, part, c, ax, r, rad, bone, mat='accent'):
    ax = unit(ax)
    side = unit(np.cross(ax, [1, 0, 0])) if abs(ax[0]) < 0.9 else unit(np.cross(ax, [0, 1, 0]))
    up = unit(np.cross(side, ax))
    pts = [c + (np.cos(t) * side + np.sin(t) * up) * r for t in np.linspace(0, 2 * np.pi, 15)]
    K.horn(part, pts, [rad] * 15, mat=mat, sides=4, per=1, bone=bone)


def build(K):
    body_c, belly_c, acc = (hex_rgb(K.colors[k]) for k in ('body', 'belly', 'accent'))
    K.mat('body', rough=0.4, detail_strength=0.3)
    K.mat('belly', rough=0.42, detail_strength=0.35)
    K.mat('plate', color=body_c * 0.85 + belly_c * 0.12, rough=0.3, metal=0.3)
    K.mat('horn', color=np.array([0.06, 0.045, 0.035]) + belly_c * 0.22, rough=0.18)
    K.mat('eye', color=acc * 0.5, emit=acc, emit_strength=1.6, rough=0.05, vcol=True)
    K.mat('membrane', color=body_c * 0.6 + acc * 0.22, rough=0.45)

    B = K.sdf()
    # ---------------------------------------------------------------- prosoma: carapace, front lobe, eye tubercle
    B.ell(CEPH, (1.42, 0.62, 1.48), k=0.1)
    B.ell(CEPH + [0, -0.08, 1.1], (0.98, 0.5, 0.62), k=0.3)
    B.ell(CEPH + [0, 0.58, 0.45], (0.34, 0.2, 0.32), k=0.15)                   # median eye tubercle
    B.ell(CEPH + [0, -0.42, -0.1], (1.15, 0.45, 1.3), k=0.25)                  # sternum / coxal mass
    # chelicerae (small pincer jaws at the front)
    for s in (-1, 1):
        B.tube([CEPH + [s * 0.3, 0.05, 1.45], CEPH + [s * 0.32, 0.0, 1.85], CEPH + [s * 0.28, -0.12, 2.2]], [0.24, 0.22, 0.15], k=0.08)
    # ---------------------------------------------------------------- mesosoma: shingled tergites over a ventral tube
    B.tube([(0, Y0 - 0.25, 1.3), (0, Y0 - 0.25, -1.3), (0, Y0 - 0.1, -3.6)], [1.25, 1.4, 0.8], k=0.2, s=(1, 0.52, 1))
    for i, (z, w) in enumerate(TERG):
        B.ell((0, Y0 + 0.05 + 0.02 * i, z), (w, 0.66, 0.5), rot=(-6, 0, 0), k=0.07)
    B.tube([(0, Y0 + 0.05, -3.35), TJ[0]], [0.92, 0.62], k=0.12)
    # ---------------------------------------------------------------- tail: five separate beads
    for i in range(5):
        a, b = TJ[i], TJ[i + 1]
        u = b - a
        r = TR[i]
        B.tube([a + u * 0.04, a + u * 0.3, a + u * 0.68, b - u * 0.03], [0.6 * r, r, r, 0.62 * r], k=0.05, s=(1.1, 1, 1))
        n = tail_out(a, b)
        for kx in (-0.32, 0.32):   # paired dorsal keels + ventral keel
            off = n * r * 0.86 + np.array([kx * r, 0, 0])
            B.tube([a + u * 0.12 + off * 0.8, a + u * 0.5 + off, b - u * 0.14 + off * 0.82], [0.09, 0.13, 0.08], k=0.06)
        for kx in (-1, 1):         # lateral keels
            off = np.array([kx * r * 1.02, 0, 0])
            B.tube([a + u * 0.15 + off * 0.85, a + u * 0.5 + off, b - u * 0.15 + off * 0.85], [0.07, 0.1, 0.06], k=0.06)
        B.tube([a + u * 0.15 - n * r * 0.8, a + u * 0.5 - n * r * 0.95, b - u * 0.15 - n * r * 0.8], [0.06, 0.09, 0.05], k=0.07)
    B.tube([TJ[5] - (TJ[5] - TJ[4]) * 0.05, VES + [0, -0.05, -0.75]], [0.3, 0.4], k=0.08)
    B.ell(VES, VES_R, k=0.12)
    # ---------------------------------------------------------------- pedipalps: femur, patella, chunky chela
    PP = {}
    for s, sd in ((-1, 'L'), (1, 'R')):
        P = palp_pts(s)
        PP[sd] = P
        B.tube([P['sh'], P['sh'] * 0.5 + P['el'] * 0.5 + [0, 0.1, 0], P['el']], [0.36, 0.38, 0.36], k=0.08)
        B.ell(P['el'], (0.42, 0.42, 0.42), k=0.04)
        B.tube([P['el'], P['el'] * 0.5 + P['wr'] * 0.5 + [s * 0.15, 0.08, 0], P['wr']], [0.38, 0.46, 0.4], k=0.05)
        B.ell(P['wr'], (0.4, 0.4, 0.4), k=0.04)
        B.ell(P['pc'], (0.7, 0.62, 1.02), rot=(0, s * -6, 0), k=0.12)                  # palm (manus)
        B.tube([P['pc'] + [s * 0.35, 0.5, -0.75], P['pc'] + [s * 0.42, 0.56, 0.0], P['pc'] + [s * 0.32, 0.4, 0.75]], [0.08, 0.12, 0.08], k=0.1)   # palm keel
        B.tube([P['pc'] + [-s * 0.2, 0.55, -0.7], P['pc'] + [-s * 0.22, 0.6, 0.0], P['pc'] + [-s * 0.18, 0.45, 0.7]], [0.07, 0.1, 0.07], k=0.1)
        B.tube(P['fix'], [0.36, 0.26, 0.08], k=0.1)
        B.tube(P['mov'], [0.32, 0.24, 0.07], k=0.08)
    # ---------------------------------------------------------------- eight walking legs
    LP = {}
    for k in range(4):
        for s, sd in ((-1, 'L'), (1, 'R')):
            P = leg_pts(k, s)
            LP[(k, sd)] = P
            sc = LEGS[k][1]
            B.ell(P['hip'], (0.34, 0.3, 0.32), k=0.12)
            B.tube([P['hip'], P['cox']], [0.32, 0.29], k=0.06)
            B.ell(P['cox'], (0.27, 0.27, 0.27), k=0.03)
            B.tube([P['cox'], P['fm'], P['knee']], [0.27 * sc, 0.3 * sc, 0.24 * sc], k=0.04)
            B.ell(P['knee'], (0.28 * sc,) * 3, k=0.03)
            B.tube([P['knee'], P['tm'], P['ank']], [0.23 * sc, 0.23 * sc, 0.17 * sc], k=0.035)
            B.ell(P['ank'], (0.17 * sc,) * 3, k=0.03)
            B.tube([P['ank'], P['ank'] * 0.5 + P['foot'] * 0.5 + [0, 0.1, 0], P['foot']], [0.14 * sc, 0.12 * sc, 0.08], k=0.03)

    # ---------------------------------------------------------------- sculpt detail: carapace grooves, tergite rims, stipple
    def detail(x, y, z):
        top = smoothstep(Y0 + 0.2, Y0 + 0.55, y)
        car = smoothstep(0.6, 1.0, z) * smoothstep(3.6, 3.0, z) * top
        groove = 0.04 * car * np.exp(-((np.abs(x) - 0.45 - 0.12 * (z - 2.2)) / 0.07) ** 2)
        mid = 0.03 * top * smoothstep(1.2, 0.6, z) * smoothstep(-3.8, -3.2, z) * np.exp(-(x / 0.07) ** 2)
        st = -0.018 * fbm(x, y, z, 2, freq=4.2, seed=11, ridged=True) - 0.012 * fbm(x, y, z, 3, freq=1.3, seed=2)
        return groove + mid + st
    B.disp(detail, band=0.1)

    # ---------------------------------------------------------------- material regions
    belly = K.sdf()
    belly.ell((0, Y0 - 0.85, -0.6), (1.6, 0.55, 4.0), k=0)
    belly.ell(CEPH + [0, -0.75, -0.1], (1.2, 0.35, 1.4), k=0)
    joints = K.sdf()
    for (k, sd), P in LP.items():
        sc = LEGS[k][1]
        joints.ell(P['knee'], (0.33 * sc,) * 3, k=0)
        joints.ell(P['ank'], (0.23 * sc,) * 3, k=0)
        joints.ell(P['cox'], (0.33,) * 3, k=0)
    for sd, P in PP.items():
        joints.ell(P['el'], (0.48,) * 3, k=0)
        joints.ell(P['wr'], (0.46,) * 3, k=0)
    for i in range(1, 5):
        joints.ell(TJ[i], (0.4,) * 3, k=0)

    def venom(P):
        z, d = P[:, 2], np.linalg.norm((P - VES) / np.array(VES_R), axis=1)
        bands = np.minimum(np.abs(z - (VES[2] - 0.35)), np.abs(z - (VES[2] + 0.15))) - 0.07
        return np.maximum(bands, d - 1.15)
    def tint(P, N):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        on = smoothstep(1.1, 0.8, z) * smoothstep(-3.9, -3.5, z) * smoothstep(Y0 - 0.3, Y0 + 0.2, y)
        ph = np.zeros_like(z)
        for zc, _ in TERG:      # darker leading edge of every tergite (tucked under the plate in front)
            ph = np.maximum(ph, np.exp(-((z - (zc + 0.32)) / 0.12) ** 2))
        return 1.0 - 0.38 * on * ph
    K.skin(B, voxel=0.045, tris=18500, belly=belly, regions=[('belly', joints), ('accent', venom)], tint=tint)

    # ---------------------------------------------------------------- skeleton
    K.bone('pelvis', CEPH + [0, 0, -1.2], CEPH + [0, 0.1, 1.3], None, 1.4)
    K.bone('spine', (0, Y0 + 0.05, 0.7), (0, Y0 + 0.15, -3.5), 'pelvis', 1.5)
    K.bone('head', CEPH + [0, 0.35, 0.6], CEPH + [0, 0.3, 1.6], 'pelvis', 0.7)
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.bone('chel' + sd, CEPH + [s * 0.3, 0.05, 1.45], CEPH + [s * 0.28, -0.12, 2.3], 'head', 0.28)
    K.chain('tail', list(TJ) + [STING], 'spine', (0.7, 0.62))
    for s, sd in ((-1, 'L'), (1, 'R')):
        P = PP[sd]
        tip = (P['fix'][2] + P['mov'][2]) * 0.5
        K.bone('upperarm' + sd, P['sh'], P['el'], 'pelvis', 0.42)
        K.bone('forearm' + sd, P['el'], P['wr'], 'upperarm' + sd, 0.46)
        K.bone('hand' + sd, P['wr'], tip, 'forearm' + sd, 0.75)
        for k in range(4):
            L = LP[(k, sd)]
            K.chain(f'leg{k + 1}{sd}', [L['hip'], L['cox'], L['knee'], L['ank'], L['foot']], 'pelvis', (0.3, 0.14))
        K.mount('hand' + sd, P['fb'] + [0, 0, 0.55], 'hand' + sd)
        K.mount('shoulder' + sd, P['sh'] + [s * 0.35, 0.55, -0.2], 'upperarm' + sd)
    K.mount('head', CEPH + [0, 0.6, 1.0], 'head')
    K.mount('jaw', CEPH + [0, -0.05, 2.1], 'head')
    K.mount('mouth', CEPH + [0, -0.15, 2.45], 'head')
    K.mount('chest', CEPH + [0, -0.4, 1.3], 'pelvis')
    K.mount('back', (0, Y0 + 0.75, -1.3), 'spine')
    K.mount('crown', CEPH + [0, 1.0, 0.5], 'head')

    # ---------------------------------------------------------------- eyes: median pair + lateral clusters
    for s in (-1, 1):
        q, n = K.snap(CEPH + [s * 0.2, 0.75, 0.5])
        K.eye('body', q + n * 0.06, n + [s * 0.3, 0, 0.5], 0.13, bone='head')
        for x, y, z, r in ((0.82, 0.18, 1.3, 0.085), (0.92, 0.12, 1.05, 0.075), (0.72, 0.28, 1.45, 0.07)):
            q, n = K.snap(CEPH + [s * x, y, z])
            K.ball('body', q + n * 0.03, r, 'eye', seg=10, rings=6, bone='head')
    # chelicera tips, tarsal claws
    for s, sd in ((-1, 'L'), (1, 'R')):
        base = CEPH + [s * 0.28, -0.12, 2.2]
        K.claw('body', base, (-s * 0.3, -0.3, 1), 0.3, 0.07, curl=(-s, -0.3, 0), bone='chel' + sd)
        K.claw('body', base + [s * 0.1, 0, 0], (s * 0.15, -0.4, 1), 0.28, 0.06, curl=(-s, -0.3, 0), bone='chel' + sd)
    for (k, sd), P in LP.items():
        d = P['d']
        for sp in (-0.3, 0.3):
            K.claw('body', P['foot'], d + np.cross(d, [0, 1, 0]) * sp + [0, -0.4, 0], 0.3, 0.05, curl=(0, -1, 0), bone=f'leg{k + 1}{sd}4')

    def curve(pts, f):
        C = S.catmull(pts, 8)
        return C[int(round(f * (len(C) - 1)))]

    # ---------------------------------------------------------------- Might: stinger + pincer teeth (exclusive sets)
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        L = 1.05 * sc
        hook = 0.15 + 0.2 * t
        K.horn(part, [STING - [0, 0, 0.25], STING + [0, 0.04, 0.42 * L], STING + [0, -0.28 * L, 0.8 * L], STING + [0, -0.72 * L, (0.88 - hook) * L]],
               [0.3 * (0.85 + 0.15 * sc), 0.22 * (0.85 + 0.15 * sc), 0.1, 0.0], mat='horn', sides=9, per=4, bone='tail6')
        if t >= 2:   # barb under the stinger
            K.claw(part, STING + [0, -0.3, 0.2], (0, -1, 0.5), 0.35 + 0.1 * t, 0.1, curl=(0, 0, 1), bone='tail6')
        for s, sd in ((-1, 'L'), (1, 'R')):
            P = PP[sd]
            nt = 4 + t
            for fing, inward, rad in ((P['fix'], -s, (0.26, 0.08)), (P['mov'], s, (0.23, 0.07))):
                for i in range(nt):
                    f = 0.18 + 0.66 * i / max(1, nt - 1)
                    c = curve(fing, f)
                    r = rad[0] + (rad[1] - rad[0]) * f
                    dirn = np.array([inward, 0, 0.35])
                    K.spike(part, c + np.array([inward * r * 0.75, 0, 0]), dirn, (0.16 + 0.05 * t) * (1.1 - 0.4 * f), r=0.05 + 0.012 * t, bone='hand' + sd)
            if t >= 1:   # hooked finger tips
                for fing, inward in ((P['fix'], -s), (P['mov'], s)):
                    K.claw(part, fing[2] - (fing[2] - fing[1]) * 0.1, unit(fing[2] - fing[1]), 0.25 + 0.1 * t, 0.07, curl=(inward, 0, 0), bone='hand' + sd)
            if t >= 2:   # cutting blade along the movable finger top
                m = P['mov']
                K.horn(part, [m[0] + [0, 0.18, 0.1], m[1] + [0, 0.2, 0], m[2] + [0, 0.12, -0.05], m[2] + [0, 0.05, 0.25]], [0.09, 0.08, 0.05, 0.0],
                       flat=0.3, sides=6, bone='hand' + sd, up=(1, 0, 0))
            if t >= 3:   # serrated palm and forearm spines
                for i, f in enumerate((0.25, 0.5, 0.75)):
                    c = P['el'] * (1 - f) + P['wr'] * f
                    K.spike(part, c + [s * 0.3, 0.25, 0], (s * 0.8, 1, 0.2), 0.5, r=0.11, bone='forearm' + sd)
                for f in (-0.4, 0.1, 0.55):
                    q, n = K.snap(P['pc'] + [s * 0.7, 0.3, f])
                    K.spike(part, q, n + [0, 0.4, 0.3], 0.45, r=0.1, bone='hand' + sd)

    # ---------------------------------------------------------------- Armor: thicker carapace + tergites, then limb plates, then spikes
    K.plate('arm_t1', CEPH + [0, 0.6, -0.1], (1.3, 0.55, 1.35), bone='pelvis', lames=2, lame_axis=(0, 0, 1))
    K.plate('arm_t1', (0, Y0 + 0.75, -1.3), (1.9, 0.6, 2.55), bone='spine', lames=7, lame_axis=(0, 0, 1), lame_step=0.09)
    for s, sd in ((-1, 'L'), (1, 'R')):
        P = PP[sd]
        K.plate('arm_t2', P['pc'] + [s * 0.2, 0.45, -0.1], (0.7, 0.5, 0.85), bone='hand' + sd, lames=2, lame_axis=(0, 0, 1))
        K.plate('arm_t2', P['el'] * 0.5 + P['wr'] * 0.5 + [s * 0.15, 0.3, 0], (0.45, 0.4, 0.65), bone='forearm' + sd)
    for i in range(5):
        a, b = TJ[i], TJ[i + 1]
        c = a * 0.5 + b * 0.5
        u = unit(b - a)
        n = tail_out(a, b)
        K.plate('arm_t2', c + n * 0.6, (0.82, 0.82, 0.82), bone=f'tail{i + 1}', lames=2, lame_axis=u)
        K.spike('arm_t3', c + n * TR[i] * 0.9, n + u * -0.3, 0.45, r=0.1, bone=f'tail{i + 1}')
    for s, sd in ((-1, 'L'), (1, 'R')):
        for i, (z, w) in enumerate(TERG):
            if i % 2 == 0:
                q, n = K.snap((s * (w - 0.1), Y0 + 0.25, z))
                K.spike('arm_t3', q, n + [0, 0.5, -0.4], 0.55 - 0.03 * i, r=0.12, bone='spine')
        q, n = K.snap(CEPH + [s * 1.2, 0.25, 0.2])
        K.spike('arm_t3', q, n + [0, 0.5, 0.2], 0.45, r=0.1, bone='pelvis')
        P = PP[sd]
        K.spike('arm_t3', P['el'] + [s * 0.3, 0.2, -0.1], (s * 0.6, 0.6, -0.6), 0.6, r=0.13, bone='upperarm' + sd)

    # ---------------------------------------------------------------- Energy: venom blisters -> glowing tail joints -> tergite seams
    for s in (-1, 1):            # glowing venom veins over the bulb
        for dy in (0.3, -0.2):
            pts = [VES + [s * 0.3, dy * 0.6, -0.85], VES + [s * 0.75, dy, -0.3], VES + [s * 0.78, dy, 0.25], VES + [s * 0.45, dy * 0.7, 0.75]]
            q, n = K.snap(np.array(pts))
            K.horn('enr_t1', q + n * 0.02, [0.04, 0.07, 0.07, 0.03], mat='accent', sides=5, per=3, bone='tail6')
    q, n = K.snap(VES + [0, 0.7, 0])
    K.ball('enr_t1', q - n * 0.05, (0.22, 0.14, 0.5), 'accent', seg=10, rings=6, bone='tail6')
    for i in range(1, 5):
        ax = TJ[i + 1] - TJ[i - 1]
        ring(K, 'enr_t2', TJ[i], ax, TR[i] * 0.62, 0.075, f'tail{i + 1}')
    ring(K, 'enr_t2', TJ[5] + (VES - TJ[5]) * 0.3, VES - TJ[5], 0.36, 0.07, 'tail6')
    for i in range(len(TERG) - 1):
        z = (TERG[i][0] + TERG[i + 1][0]) * 0.5
        w = min(TERG[i][1], TERG[i + 1][1]) * 0.85
        xs = np.linspace(-w, w, 9)
        q, n = K.snap(np.stack([xs, np.full(9, Y0 + 1.0), np.full(9, z)], 1))
        K.horn('enr_t3', q + n * 0.15, [0.0] + [0.06] * 7 + [0.0], mat='accent', sides=4, per=2, bone='spine')
    for s in (-1, 1):            # glowing carapace grooves
        pts = [CEPH + [s * 0.32, 0.6, 1.1], CEPH + [s * 0.55, 0.6, 0.3], CEPH + [s * 0.75, 0.55, -0.5], CEPH + [s * 0.6, 0.5, -1.2]]
        q, n = K.snap(np.array(pts))
        K.horn('enr_t3', q + n * 0.16, [0.03, 0.06, 0.06, 0.03], mat='accent', sides=4, per=3, bone='pelvis')

    # ---------------------------------------------------------------- Speed: leg fringes -> tail fins -> lateral vanes
    for s, sd in ((-1, 'L'), (1, 'R')):
        for k in range(4):
            P = LP[(k, sd)]
            c = P['knee'] * 0.5 + P['ank'] * 0.5
            K.frill('spd_t1', c + [0, 0.1, 0], P['d'] * 0.6 + [0, 1, 0], -P['d'] + [0, 0.4, 0], 0.5, 0.45, rays=4, spread=70, bone=f'leg{k + 1}{sd}3', ray_mat='dark')
        for i in (1, 2, 3):
            a, b = TJ[i], TJ[i + 1]
            c = a * 0.55 + b * 0.45
            K.frill('spd_t2', c + [s * TR[i] * 0.8, 0, 0], (s, 0, 0), unit(a - b), 0.8, 0.9, rays=4, spread=60, bone=f'tail{i + 1}', ray_mat='dark')
        q, n = K.snap((s * 1.6, Y0 + 0.35, -1.4))
        K.frill('spd_t3', q, n + [0, 0.7, 0], (0, 0.1, -1), 1.0, 2.6, rays=6, spread=55, bone='spine', ray_mat='dark')
    K.frill('spd_t3', VES + [0, VES_R[1] * 0.85, -0.2], (0, 1, -0.15), (0, 0, -1), 0.8, 0.9, rays=4, spread=70, bone='tail6', ray_mat='dark')

    # ---------------------------------------------------------------- Focus: extra eye rows, pincer sensory hairs + pectines, crest + third eye
    for s in (-1, 1):
        for x, y, z in ((1.05, 0.08, 0.75), (1.12, 0.02, 0.45), (0.98, 0.15, 0.95)):
            q, n = K.snap(CEPH + [s * x, y, z])
            K.ball('fcs_t1', q + n * 0.03, 0.065, 'eye', seg=8, rings=6, bone='head')
        q, n = K.snap(CEPH + [s * 0.45, 0.62, 0.95])
        K.ball('fcs_t1', q + n * 0.03, 0.08, 'eye', seg=8, rings=6, bone='head')
    for s, sd in ((-1, 'L'), (1, 'R')):
        P = PP[sd]
        for f, ang in ((-0.5, 0.2), (0.0, 0.6), (0.45, 1.0)):
            q, n = K.snap(P['pc'] + [s * 0.55, 0.35, f])
            tip = q + unit(n + [s * 0.2, 0.8, 0.3 * ang]) * 1.0
            K.horn('fcs_t2', [q, q * 0.45 + tip * 0.55 + [0, 0.1, 0], tip], [0.03, 0.018, 0.0], mat='accent', sides=4, bone='hand' + sd)
        for i in range(7):   # pectines: comb teeth under the body
            z = 0.55 - 0.13 * i
            base = np.array([s * 0.3, Y0 - 0.85, 0.55])
            tip = np.array([s * (0.55 + 0.11 * i), Y0 - 0.95, z - 0.2])
            K.horn('fcs_t2', [base, base * 0.5 + tip * 0.5, tip], [0.05, 0.04, 0.0], mat='accent', sides=4, bone='spine')
    K.leaf('fcs_t3', CEPH + [0, 0.62, 0.05], (0, 1, -0.1), (0, 0, -1), 0.6, 1.0, th=0.05, lobes=3, mat='accent', bone='head')
    q, n = K.snap(CEPH + [0, 0.8, 0.85])
    K.eye('fcs_t3', q + n * 0.04, n + [0, 0, 0.6], 0.14, bone='head', lid=False)

    K.mount('tailTip', STING + [0, -0.75, 0.75], 'tail6')
