"""Giant Baby: a colossal wailing toddler in a cloth diaper (biped plan). Game coords, ~9.8 units tall, faces +Z.

Body = skin (player colour tints the baby), belly = diaper cloth, accent = glowing eyes / pin heads / toy glow.
Right fist shakes a rattle that grows into a spiked mace-rattle (Might); left hand reaches forward to grab.
"""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep
from .magicalgirl import ribbon
from .superhero import A, cape_grid, capsule, frame, nrm, oball, star_pts, torus

INFO = dict(name='Giant Baby', colors=dict(body='#f2b494', belly='#a9d9f2', accent='#ff3d5a'))
DETAIL = 'skin'
UV_SCALE = 0.75
EYE_ACCENT = True
CAM = dict(az=32, el=8)

# ---------------------------------------------------------------- joints (R side, +X; L mirrored unless listed)
PELVIS = A(0, 2.75, -0.1)
WAIST = A(0, 4.05, 0.05)
CHEST = A(0, 5.45, 0.0)
NECK = A(0, 6.35, 0.1)
HEAD = A(0, 8.2, 0.2)          # cranium centre
HEAD_R = A(1.6, 1.6, 1.6)

HIP = A(0.95, 2.6, 0.0)
KNEE = A(1.6, 1.35, 0.32)
ANKLE = A(1.82, 0.52, 0.08)
TOE = A(2.05, 0.3, 1.05)

SH = {'L': A(-1.7, 6.0, 0.1), 'R': A(1.7, 6.0, 0.1)}
EL = {'L': A(-2.3, 5.75, 1.5), 'R': A(2.55, 6.55, 1.0)}
WR = {'L': A(-2.2, 6.25, 2.8), 'R': A(2.78, 7.8, 1.5)}

MOUTH_C = A(0, 7.12, 1.62)
MOUTH_R = A(0.5, 0.44, 0.6)
MOUTH_ROT = (-12, 0, 0)


def M(s):
    return np.array([s, 1, 1])


def leg(sd):
    m = M(-1 if sd == 'L' else 1)
    return m * HIP, m * KNEE, m * ANKLE, m * TOE


# ---------------------------------------------------------------- hands
def hand_frame(sd):
    d = nrm(WR[sd] - EL[sd])
    if sd == 'L':
        b = A(0, 1, -0.25)          # back of the reaching hand faces up/back, palm forward-down
    else:
        b = A(1, 0, 0)               # raised right fist: back of hand faces outward
    return frame(d, b)               # x, y = along hand, z = back of hand


def open_hand(B, sd):
    """Chubby reaching hand: fat palm, five short splayed fingers curled toward the palm. Returns fingertips."""
    R = hand_frame(sd)
    w = WR[sd]
    L = lambda v: w + R @ np.asarray(v, float)
    ts = -1 if sd == 'L' else 1      # thumb side (local x); 'L' (at -x) is the anatomical right hand
    B.ell(L((0, 0.38, 0.0)), A(0.46, 0.44, 0.24), rot=R, k=0.18)                       # palm
    B.ell(L((ts * 0.32, 0.22, -0.08)), A(0.2, 0.26, 0.18), rot=R, k=0.1)                 # thumb pad
    tips = []
    fingers = ((ts * 0.27, 0.62, 0.15, 0.5, 0.35), (ts * 0.09, 0.7, 0.155, 0.58, 0.12), (-ts * 0.1, 0.68, 0.15, 0.55, -0.1), (-ts * 0.27, 0.6, 0.13, 0.42, -0.32))
    for x, ky, r, ln, spr in fingers:
        dd = nrm(A(spr * 0.6, 1, 0))
        k0 = A(x, ky, 0.0)
        pts = [k0, k0 + dd * ln * 0.5 + A(0, 0, -0.06), k0 + dd * ln * 0.85 + A(0, 0, -0.2), k0 + dd * ln * 1.0 + A(0, 0, -0.34)]
        B.tube([L(p) for p in pts], [r, r * 0.97, r * 0.92, r * 0.85], k=0.035, per=3)
        tips.append((L(pts[-1]), R @ nrm(pts[-1] - pts[-2])))
    tp = [A(ts * 0.36, 0.25, -0.05), A(ts * 0.6, 0.42, -0.12), A(ts * 0.72, 0.6, -0.2)]
    B.tube([L(p) for p in tp], [0.16, 0.15, 0.135], k=0.05, per=3)
    tips.append((L(tp[-1]), R @ nrm(tp[-1] - tp[-2])))
    return tips


def fist(B, sd):
    """Pudgy fist (gripping the rattle handle along local x). Returns (grip centre, handle axis)."""
    R = hand_frame(sd)
    w = WR[sd]
    L = lambda v: w + R @ np.asarray(v, float)
    B.box(L((0, 0.38, 0.0)), A(0.36, 0.3, 0.27), rot=R, rad=0.22, k=0.16)
    for i, x in enumerate((-0.24, -0.08, 0.08, 0.24)):
        B.ell(L((x, 0.66, -0.06)), A(0.11, 0.13, 0.2), rot=R, k=0.05)                    # curled finger rolls
        B.ell(L((x, 0.58, 0.16)), A(0.1, 0.1, 0.1), rot=R, k=0.05)                       # knuckle dimples
    B.tube([L((0.3, 0.2, -0.1)), L((0.38, 0.42, -0.25)), L((0.2, 0.58, -0.3))], [0.16, 0.15, 0.13], k=0.05)  # thumb over fingers
    grip = L((0.0, 0.5, -0.12))
    axis = nrm(A(0.42, 0.75, 0.5))       # handle leaves the fist up/out/forward, away from the head
    return grip, axis


# ---------------------------------------------------------------- creases (fat folds on wrists, ankles, limbs)
def make_creases():
    C = []   # (centre, axis, limb a, limb b, limb radius, depth)
    for sd in 'LR':
        s = -1 if sd == 'L' else 1
        d = nrm(WR[sd] - EL[sd])
        C.append((WR[sd] - d * 0.05, d, EL[sd], WR[sd] + d * 0.4, 0.62, 0.09))
        u = nrm(EL[sd] - SH[sd])
        C.append((EL[sd] - u * 0.08, u, SH[sd], EL[sd] + d * 0.4, 0.75, 0.06))
        hp, kn, an, toe = leg(sd)
        v = nrm(an - kn)
        C.append((an - v * 0.12, v, kn, an, 0.72, 0.1))
        t = nrm(kn - hp)
        C.append((hp + (kn - hp) * 0.62, t, hp, kn, 1.0, 0.08))
        C.append((kn + v * 0.08, v, kn - t * 0.4, an, 0.85, 0.07))
    return C


CREASES = make_creases()


def crease_disp(x, y, z):
    P = np.stack([x, y, z], 1)
    d = np.zeros(len(P), np.float32)
    for c, ax, a, b, r, dep in CREASES:
        near = smoothstep(0.25, -0.05, capsule(P, a, b, r))
        along = (P - c) @ ax
        d += dep * np.exp(-(along / 0.07) ** 2) * near
    return d


# ---------------------------------------------------------------- build
def build(K):
    col = K.colors
    K.mat('body', rough=0.5, detail_strength=0.07)
    K.mat('belly', rough=0.78, detail_kind='cloth', detail_strength=0.7)
    K.mat('hair', color='#6e4a2c', rough=0.5, vcol=True, detail=False)
    K.mat('mouth', color='#7a2633', rough=0.45, vcol=True)
    K.mat('tongue', color='#d9626e', rough=0.4, vcol=True)
    K.mat('horn', color='#fbf6ea', rough=0.3, vcol=True)
    K.mat('dark', color='#1a1214', rough=0.45)
    K.mat('pin', color='#d9dde4', rough=0.22, metal=0.9, vcol=True)
    K.mat('toy', color='#ffcf3a', rough=0.32, vcol=True)
    K.mat('toy2', color='#4aa8ff', rough=0.3, vcol=True)
    K.mat('plate', color='#ffffff', rough=0.62, vcol=True, detail=True, detail_kind='cloth')
    K.mat('ribbon', color='#ff8fc0', rough=0.5, vcol=True, double=True, detail=False)
    K.mat('blanket', color='#ffe7a8', rough=0.75, vcol=True, double=True, detail=True, detail_kind='cloth')

    B = K.sdf()
    # ---- torso: round belly, narrow baby shoulders, no real neck
    B.ell((0, 4.35, 0.3), (1.85, 1.65, 1.72), k=0.0)                       # belly ball
    B.ell(CHEST, (1.62, 1.15, 1.3), k=0.6)                                  # chest
    B.ell((0, 3.3, -0.25), (1.75, 1.2, 1.55), k=0.5)                        # hips under the diaper
    for s in (-1, 1):
        B.ell(M(s) * A(1.45, 5.85, 0.1), (0.72, 0.62, 0.7), k=0.45)          # shoulder pads of fat
    B.tube([NECK - A(0, 0.4, 0), HEAD - A(0, 1.2, -0.1)], [1.0, 0.95], k=0.45)
    B.ell((0, 6.45, 0.75), (1.05, 0.38, 0.72), k=0.35)                      # double chin roll
    B.ell((0, 4.15, 2.02), (0.09, 0.12, 0.1), k=0.05, cut=True)              # belly button

    # ---- diaper: puffy cloth wrap with a waistband and leg cuffs
    DIA_C, DIA_R = A(0, 2.85, -0.12), A(2.02, 1.18, 1.82)
    B.ell(DIA_C, DIA_R, k=0.25)
    B.ell((0, 2.25, -0.2), (1.25, 0.85, 1.45), k=0.3)                       # crotch bulk
    B.ell((0, 2.95, -1.25), (1.6, 0.95, 0.8), k=0.35)                        # padded seat
    band = S.Tube(S.catmull(np.array([(np.cos(a) * 2.04, 3.82 + 0.12 * np.cos(a) ** 2, np.sin(a) * 1.86 - 0.08) for a in np.linspace(0, 2 * np.pi, 25)]), 1), [0.14], per=1)
    B.add(band, k=0.08)

    # ---- legs: short, fat, bowed out in a wide toddler stance
    global CUFFS
    CUFFS = []
    for sd in 'LR':
        s = -1 if sd == 'L' else 1
        hp, kn, an, toe = leg(sd)
        ax = nrm(nrm(kn - hp) + A(s * 0.35, 0.25, 0))
        c = hp + (kn - hp) * 0.3 + A(s * 0.05, 0.05, 0)
        CUFFS.append((c, ax, hp, an))
        x, y, z = geom.basis(ax)
        ring = [c + (np.cos(a) * x + np.sin(a) * z) * 1.02 for a in np.linspace(0, 2 * np.pi, 19)]
        B.add(S.Tube(S.catmull(np.array(ring), 2), [0.12], per=1), k=0.06)
    for sd in 'LR':
        s = -1 if sd == 'L' else 1
        hp, kn, an, toe = leg(sd)
        B.tube([hp, hp + (kn - hp) * 0.45, kn], [1.0, 0.92, 0.72], k=0.3)
        B.ell(hp + (kn - hp) * 0.35 + A(s * 0.1, 0, 0.1), (0.95, 0.75, 0.92), k=0.3)   # thigh fat
        B.ell(kn + A(0, 0, 0.12), (0.7, 0.62, 0.68), k=0.2)                            # knee
        B.tube([kn, kn + (an - kn) * 0.4, an], [0.7, 0.66, 0.46], k=0.25)
        B.ell(kn + (an - kn) * 0.35 + A(0, 0, -0.18), (0.62, 0.6, 0.6), k=0.2)          # calf fat
        # pudgy foot, toes splay outward a little
        fd = nrm(toe - an)
        B.ell(an + A(0, -0.12, -0.05), (0.5, 0.42, 0.5), k=0.2)                        # heel/ankle
        fc = an + (toe - an) * 0.55 + A(0, -0.18, 0)
        B.ell(fc, (0.56, 0.32, 0.72), rot=(0, s * 12, 0), k=0.22)                       # foot pad
        side = nrm(np.cross(A(0, 1, 0), nrm(A(fd[0], 0, fd[2]))))
        for i, (o, r) in enumerate(((-0.34, 0.15), (-0.15, 0.13), (0.03, 0.12), (0.19, 0.11), (0.33, 0.1))):
            o2 = o * s
            tp = an + (toe - an) * 0.98 + side * o2 * 1.1 + A(0, -0.12 - (0.02 * i), -0.06 * abs(o) * 3)
            B.ell(tp, (r * 1.05, r * 0.95, r * 1.15), k=0.06)
    B.box((0, -1.0, 0), (6, 1.0, 6), rad=0.0, k=0.04, cut=True)                         # flat ground plane

    # ---- arms: short and chubby, left reaches forward, right shakes a rattle overhead
    for sd in 'LR':
        sh, el, wr = SH[sd], EL[sd], WR[sd]
        B.tube([sh, sh + (el - sh) * 0.5, el], [0.7, 0.66, 0.52], k=0.3)
        B.ell(sh + (el - sh) * 0.45, (0.66, 0.62, 0.66), k=0.25)                        # upper-arm fat
        B.tube([el, el + (wr - el) * 0.45, wr], [0.52, 0.55, 0.38], k=0.22)
        d = nrm(wr - el)
        B.ell(el + (wr - el) * 0.42, A(0.5, 0.62, 0.5), rot=frame(d), k=0.2)            # forearm fat
    tips = open_hand(B, 'L')
    grip, gax = fist(B, 'R')

    # ---- head: huge round cranium, fat cheeks, small chin, wailing mouth
    B.ell(HEAD, HEAD_R, k=0.0)
    B.ell((0, 7.35, 0.7), (1.32, 1.0, 1.15), k=0.5)                                    # lower face
    for s in (-1, 1):
        B.ell((s * 0.86, 7.42, 1.18), (0.64, 0.6, 0.58), k=0.35)                         # chubby cheeks
        B.ell((s * 1.56, 7.95, 0.2), (0.2, 0.44, 0.34), rot=(0, s * 15, 0), k=0.1)       # ears
        B.ell((s * 1.68, 7.95, 0.24), (0.07, 0.24, 0.16), rot=(0, s * 15, 0), k=0.04, cut=True)
    B.ell((0, 6.72, 1.2), (0.58, 0.36, 0.5), k=0.3)                                     # chin
    B.ell((0, 7.85, 1.92), (0.22, 0.16, 0.17), k=0.1)                                   # button nose
    for s in (-1, 1):
        B.ell((s * 0.09, 7.8, 2.06), (0.05, 0.04, 0.05), k=0.03, cut=True)               # nostrils
    # furrowed brows (inner ends pulled down) + crease between them
    for s in (-1, 1):
        B.tube([(s * 1.0, 8.78, 1.3), (s * 0.6, 8.68, 1.62), (s * 0.2, 8.46, 1.82)], [0.12, 0.16, 0.13], k=0.16)
        B.ell((s * 0.6, 8.2, 1.68), (0.32, 0.25, 0.22), k=0.08, cut=True)                 # eye sockets
    B.ell((0, 8.68, 1.88), (0.04, 0.22, 0.08), k=0.04, cut=True)                           # furrow crease
    # wailing mouth: lips ring + deep cavity
    B.ell(MOUTH_C + A(0, 0, 0.12), MOUTH_R * A(1.18, 1.18, 0.6), rot=MOUTH_ROT, k=0.2)
    B.ell(MOUTH_C, MOUTH_R, rot=MOUTH_ROT, k=0.08, cut=True)

    B.disp(crease_disp, band=0.2)

    def skin_d(x, y, z):
        return -0.012 * fbm(x, y, z, 2, freq=2.0, seed=4)

    dia = S.Ell(DIA_C, DIA_R + 0.06)
    crotch = S.Ell((0, 2.25, -0.2), (1.32, 0.92, 1.52))
    seat = S.Ell((0, 2.95, -1.25), (1.66, 1.0, 0.86))

    def dia_mask(P):
        p = S.Points(P)
        top = P[:, 1] - (3.86 + 0.1 * np.clip(-P[:, 2], 0, 2))
        legs = np.full(len(P), 9.0)
        for c, ax, hp, an in CUFFS:
            legs = np.minimum(legs, np.maximum(capsule(P, hp, an, 1.5), -((P - c) @ ax)))
        m = np.maximum(top, -legs)
        m = np.maximum(m, capsule(P, A(0, 1.5, 0), A(0, 4.5, 0), 2.8))
        return np.minimum(m, band.dist(p) - 0.06)

    def cloth(x, y, z):
        P = np.stack([x, y, z], 1)
        w = smoothstep(0.1, -0.1, dia_mask(P))
        th = np.arctan2(x, z)
        pleat = 0.035 * np.sin(th * 14 + 0.6 * np.sin(y * 3)) * smoothstep(3.1, 3.7, y)          # gathers under the band
        wrinkle = 0.03 * (fbm(x * 1.2, y * 2.4, z * 1.2, 3, freq=1.4, seed=8, ridged=True) - 0.5)
        return (pleat + wrinkle) * w + skin_d(x, y, z) * (1 - w)
    B.disp(cloth, band=0.14)

    mouth = S.Ell(MOUTH_C, MOUTH_R * 1.02, MOUTH_ROT)
    K.skin(B, voxel=0.05, tris=19500, belly=dia_mask, regions=[('mouth', lambda P: mouth.dist(S.Points(P)) - 0.02)])

    # ---- eyes: glowing (accent) with round pupils, angry skin lids slanting to the furrow
    for s in (-1, 1):
        c = A(s * 0.6, 8.2, 1.55)
        n = nrm(A(s * 0.22, 0.02, 1))
        K.ball('body', c, 0.25, mat='eye', bone='head', seg=14, rings=10)
        K.ball('body', c + n * 0.2, A(0.1, 0.1, 0.06), mat='dark', bone='head', seg=10, rings=6)
        R = frame(A(s * 1, -0.42, 0), A(0, 0.3, 1))       # lid tilts down toward the nose
        oball(K, 'body', c + A(-s * 0.02, 0.15, 0.05), A(0.2, 0.36, 0.2) * A(1.0, 1.0, 1.0), R, mat='body', bone='head', seg=14, rings=8)
    # tongue in the open mouth
    K.ball('body', MOUTH_C + A(0, -0.3, 0.05), A(0.32, 0.12, 0.34), mat='tongue', bone='jaw', seg=12, rings=6)

    # ---- hair: a single big curl + wisps
    curl = [A(0, 9.62, 0.55), A(0.05, 10.0, 0.8), A(0.0, 10.25, 1.15), A(-0.1, 10.05, 1.45), A(-0.05, 9.8, 1.35), A(0.0, 9.82, 1.12), A(0.04, 9.98, 1.1)]
    K.horn('body', curl, [0.2, 0.19, 0.16, 0.13, 0.1, 0.07, 0.0], mat='hair', sides=8, per=4, flat=0.6, up=A(1, 0, 0), bone='head', shade=(0.7, 1.0))
    for i, (x, z, h) in enumerate(((-0.45, 0.25, 0.4), (0.4, 0.3, 0.45), (-0.2, -0.3, 0.35), (0.25, -0.35, 0.35), (0.0, 0.0, 0.5))):
        p, n = K.snap(A(x, 9.9, z))
        K.horn('body', [p - n * 0.05, p + n * h * 0.5 + A(x * 0.2, 0, 0.1), p + n * h + A(x * 0.4, -0.05, 0.25)], [0.08, 0.06, 0.0], mat='hair', sides=5, bone='head')

    # ---- safety pins on the diaper (accent clasp heads)
    pins = []
    for s in (-1, 1):
        p, n = K.snap(A(s * 1.3, 3.3, 1.45))
        pins.append((p, n, s))
        d = nrm(np.cross(A(0, 1, 0), n) * -s + A(0, -0.15, 0))
        v = nrm(np.cross(n, d))
        a0, a1 = p - d * 0.5 + n * 0.07, p + d * 0.5 + n * 0.07
        K.horn('body', [a0, (a0 + a1) / 2 + n * 0.04, a1], [0.05, 0.05, 0.05], mat='pin', sides=6, bone='pelvis')
        K.horn('body', [a0 + v * 0.18, (a0 + a1) / 2 + v * 0.17 + n * 0.06, a1 + v * 0.14], [0.045, 0.045, 0.045], mat='pin', sides=6, bone='pelvis')
        V, T = torus(a0 + v * 0.09, nrm(np.cross(v, n)), 0.12, 0.035, nu=12, nv=4)
        K.piece('body', V, T, 'pin', bone='pelvis')
        K.ball('body', a1 + v * 0.07 + n * 0.02, A(0.16, 0.16, 0.12), mat='accent', bone='pelvis')

    # ---- skeleton
    K.bone('pelvis', PELVIS, WAIST, None, 1.7)
    K.bone('spine', WAIST, CHEST, 'pelvis', 1.8)
    K.bone('chest', CHEST, NECK, 'spine', 1.6)
    K.bone('neck', NECK, A(0, 7.0, 0.25), 'chest', 1.0)
    K.bone('head', A(0, 7.0, 0.25), A(0, 9.7, 0.3), 'neck', 1.55)
    K.bone('jaw', A(0, 7.05, 0.75), A(0, 6.62, 1.5), 'head', 0.36)
    for sd in 'LR':
        s = -1 if sd == 'L' else 1
        hp, kn, an, toe = leg(sd)
        K.bone('thigh' + sd, hp, kn, 'pelvis', 0.95)
        K.bone('shin' + sd, kn, an, 'thigh' + sd, 0.68)
        K.bone('foot' + sd, an, toe, 'shin' + sd, 0.5)
        K.bone('upperarm' + sd, SH[sd], EL[sd], 'chest', 0.66)
        K.bone('forearm' + sd, EL[sd], WR[sd], 'upperarm' + sd, 0.52)
        d = nrm(WR[sd] - EL[sd])
        K.bone('hand' + sd, WR[sd], WR[sd] + d * 0.95, 'forearm' + sd, 0.45)
        K.mount('hand' + sd, WR[sd] + d * 0.45, 'hand' + sd)
        K.mount('shoulder' + sd, SH[sd] + A(0, 0.75, -0.1), 'chest')
    K.mount('head', HEAD, 'head')
    K.mount('jaw', A(0, 6.8, 1.4), 'jaw')
    K.mount('mouth', MOUTH_C + A(0, 0, 0.55), 'head')
    K.mount('chest', A(0, 5.4, 1.35), 'chest')
    K.mount('back', A(0, 5.0, -1.4), 'spine')
    K.mount('tailTip', A(0, 2.9, -2.1), 'pelvis')
    K.mount('crown', A(0, 10.0, 0.3), 'head')

    # ---- MIGHT: rattle -> spiked mace-rattle, bigger teeth, fingernail claws (exclusive sets)
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        hl = 0.95 + 0.25 * t
        h0, h1 = grip - gax * 0.55, grip + gax * hl
        K.horn(part, [h0, grip, h1], [0.15, 0.13 + 0.02 * t, 0.13 + 0.02 * t], mat='toy', sides=10, bone='handR')
        K.ball(part, h0, 0.2 + 0.03 * t, mat='toy', bone='handR', seg=12, rings=8)
        R = 0.46 * (0.8 + 0.2 * sc) + 0.06 * t
        c = h1 + gax * R * 0.85
        K.ball(part, c, R, mat='toy2', bone='handR', seg=16, rings=10)
        V, T = torus(c, gax, R * 1.0, 0.07 + 0.015 * t, nu=28, nv=6)
        K.piece(part, V, T, 'toy' if t < 2 else 'accent', bone='handR')
        V, T = torus(h1, gax, 0.2 + 0.03 * t, 0.06, nu=12, nv=4)
        K.piece(part, V, T, 'toy', bone='handR')
        x, y, z = geom.basis(gax)
        if t == 1:   # round nubs
            for i in range(8):
                a, e = 2 * np.pi * i / 8, (0.5 if i % 2 else -0.5)
                dv = nrm(np.cos(a) * x + np.sin(a) * z + y * e)
                K.ball(part, c + dv * R * 0.95, 0.12, mat='toy', bone='handR', seg=8, rings=6)
        if t >= 2:   # studded spikes
            n_ring = 7 if t == 2 else 8
            for e, rot0 in ((-0.55, 0.0), (0.0, 0.4), (0.55, 0.0)) + (((0.92, 0.2),) if t == 3 else ()):
                cnt = n_ring if abs(e) < 0.9 else 4
                for i in range(cnt):
                    a = 2 * np.pi * i / cnt + rot0
                    dv = nrm(np.cos(a) * x * np.sqrt(1 - e * e) + np.sin(a) * z * np.sqrt(1 - e * e) + y * e)
                    K.spike(part, c + dv * R * 0.9, dv, R * (0.55 if t == 2 else 0.75), r=R * 0.13, mat='pin', bone='handR', sides=5)
            K.spike(part, c + gax * R * 0.9, gax, R * 0.9, r=R * 0.15, mat='pin', bone='handR')
        if t == 3:
            for i in range(6):   # glowing cracks / gems between the spikes
                a = 2 * np.pi * i / 6 + 0.2
                dv = nrm(np.cos(a) * x * 0.95 + np.sin(a) * z * 0.95 + y * 0.28 * (1 if i % 2 else -1))
                K.crystal(part, c + dv * R * 0.92, dv, 0.25, r=0.1, cluster=False, bone='handR')
        # teeth: two lower + two upper; grow into little fangs
        tl = 0.16 + 0.07 * t
        for s in (-1, 1):
            for j, (yy, sgn, bn) in enumerate(((-0.74, 1, 'jaw'), (0.86, -1, 'head'))):
                xo = s * 0.13
                base = MOUTH_C + A(xo * MOUTH_R[0] / 0.5, yy * MOUTH_R[1], MOUTH_R[2] * (0.62 if sgn < 0 else 0.42)) + A(0, sgn * -0.02, 0)
                tip = base + A(0, sgn * tl, -0.02)
                K.horn(part, [base - A(0, sgn * 0.06, 0), base + (tip - base) * 0.5, tip], [0.07 + 0.012 * t, 0.06 + 0.01 * t, 0.01 if t < 2 else 0.0], mat='horn', sides=6, bone=bn, flat=0.7, up=A(0, 0, 1))
        if t >= 1:   # extra fangs at the corners from t2
            pass
        if t >= 2:
            for s in (-1, 1):
                base = MOUTH_C + A(s * MOUTH_R[0] * 0.55, 0.82 * MOUTH_R[1] * 0.85, MOUTH_R[2] * 0.45)
                K.horn(part, [base + A(0, 0.05, 0), base - A(0, 0.12 + 0.05 * t, 0)], [0.06, 0.0], mat='horn', sides=5, bone='head')
        # fingernail claws on the grabbing hand, toe claws from t2
        if t >= 1:
            ln = 0.12 + 0.12 * t
            for p, dv in tips:
                K.claw(part, p - dv * 0.08, dv, ln, 0.07 + 0.01 * t, curl=-hand_frame('L')[:, 2], bone='handL')
        if t >= 2:
            for sd in 'LR':
                s = -1 if sd == 'L' else 1
                hp, kn, an, toe = leg(sd)
                fd = nrm(toe - an)
                side = nrm(np.cross(A(0, 1, 0), nrm(A(fd[0], 0, fd[2]))))
                for o in (-0.34, -0.15, 0.03, 0.19):
                    p = an + (toe - an) * 0.98 + side * o * s * 1.1 + A(0, -0.12, 0.06)
                    K.claw(part, p, A(fd[0], -0.1, fd[2]), 0.12 + 0.08 * (t - 1), 0.06, curl=(0, -1, 0), bone='foot' + sd)

    # ---- ARMOR: padded bib, knee + elbow pads, inflatable arm floaties
    K.plate('arm_t1', A(0, 5.3, 1.4), (1.3, 1.05, 1.0), off=0.04, th=0.09, mat='plate', bone='chest', lames=0, tris=600)
    p, n = K.snap(A(0, 5.0, 1.9))
    K.ball('arm_t1', p + n * 0.12, A(0.32, 0.32, 0.08), mat='toy2', bone='chest', rot=None)
    star = star_pts(p + n * 0.2, n, A(0, 1, 0), 0.24, 0.11, depth=0.05)
    K.piece('arm_t1', star[0], star[1], 'toy', smooth=False, bone='chest')
    V, T = torus(A(0, 6.25, 0.2), A(0, 1, 0.12), 1.12, 0.12, nu=36, nv=6)   # bib collar
    K.piece('arm_t1', V, T, 'plate', bone='chest')
    for sd in 'LR':
        s = -1 if sd == 'L' else 1
        hp, kn, an, toe = leg(sd)
        K.plate('arm_t2', kn + A(0, 0, 0.6), (0.62, 0.6, 0.5), off=0.04, th=0.1, mat='toy2', bone='shin' + sd, lames=2, lame_axis=an - kn, tris=320)
        u = nrm(EL[sd] - SH[sd])
        K.plate('arm_t2', EL[sd] - nrm(np.cross(u, A(0, 1, 0))) * 0.0 + A(s * 0.3, -0.2, -0.3), (0.5, 0.5, 0.5), off=0.04, th=0.1, mat='toy2', bone='forearm' + sd, lames=2, lame_axis=u, tris=280)
        # floaties: fat inflatable ring around each upper arm
        c = SH[sd] + (EL[sd] - SH[sd]) * 0.5
        V, T = torus(c, u, 0.8, 0.3, nu=22, nv=9)
        K.piece('arm_t3', V, T, 'toy', bone='upperarm' + sd)
        V, T = torus(c + u * 0.0, u, 1.04, 0.06, nu=22, nv=4)
        K.piece('arm_t3', V, T, 'ribbon', bone='upperarm' + sd)
        x, y, z = geom.basis(u)
        K.horn('arm_t3', [c + z * 1.02 - u * 0.05, c + z * 1.18], [0.07, 0.05], mat='toy', sides=6, bone='upperarm' + sd)   # valve
    # thick cloth knee pad straps visible from the side handled by the lames; tier 3 also adds a padded crash helmet-band

    # ---- ENERGY: glowing pacifier on a cord, orbiting toy blocks, star mobile
    pc = MOUTH_C + A(0, -1.55, 0.95)
    pcp, pcn = K.snap(pc)
    pc = pcp + pcn * 0.35
    cord = [A(-0.95, 6.35, 0.95), A(-0.6, 6.1, 1.55), pc + A(-0.15, 0.25, -0.1), pc]
    K.horn('enr_t1', cord, [0.04, 0.04, 0.04, 0.04], mat='ribbon', sides=5, bone='chest')
    cord2 = [A(0.95, 6.35, 0.95), A(0.6, 6.1, 1.55), pc + A(0.15, 0.25, -0.1), pc]
    K.horn('enr_t1', cord2, [0.04, 0.04, 0.04, 0.04], mat='ribbon', sides=5, bone='chest')
    fwd = nrm(pcn + A(0, -0.2, 0))
    oball(K, 'enr_t1', pc, A(0.42, 0.07, 0.3), frame(fwd, A(0, 1, 0)), mat='accent', bone='chest')
    K.ball('enr_t1', pc + fwd * 0.12, 0.14, mat='accent', bone='chest')
    V, T = torus(pc + fwd * 0.28, A(0, 1, 0), 0.16, 0.045, nu=16, nv=5)
    K.piece('enr_t1', V, T, 'accent', bone='chest')
    # toy blocks hovering by the shoulders
    for i, (p, rot) in enumerate(((A(-3.0, 7.6, -0.6), (20, 30, 10)), (A(3.35, 6.4, -0.8), (-15, 40, 25)), (A(-3.3, 5.6, -1.1), (35, -20, 15)))):
        Bk = K.sdf()
        Bk.box(p, (0.36, 0.36, 0.36), rot=rot, rad=0.07, k=0)
        K.sdf_piece('enr_t2', Bk, voxel=0.04, tris=200, mat='toy2' if i % 2 else 'toy', bone='chest')
        R = S.rot_xyz(rot)
        for f in range(6):
            ax = A(*np.eye(3)[f // 2]) * (1 if f % 2 else -1)
            nn = R @ ax
            st = star_pts(p + nn * 0.37, nn, R @ A(*np.eye(3)[(f // 2 + 1) % 3]), 0.2, 0.09, depth=0.025)
            K.piece('enr_t2', st[0], st[1], 'accent', smooth=False, bone='chest')
    # dream ring: glowing stars + crescent moon drifting in a tilted ring around the head and shoulders
    for i in range(5):
        a = 2 * np.pi * i / 5 + 0.9
        p = A(np.cos(a) * 3.0, 8.4 + 0.7 * np.sin(a + 0.6), np.sin(a) * 2.4 - 0.5)
        st = star_pts(p, nrm(A(np.cos(a), 0.2, np.sin(a)) * 0.3 + A(0, 0, 1)), A(0, 1, 0), 0.34 - 0.03 * (i % 2), 0.15, depth=0.1)
        K.piece('enr_t3', st[0], st[1], 'accent', smooth=False, bone='chest')
    mp = A(-2.2, 10.0, -0.9)
    moon = K.sdf()
    moon.ell(mp, (0.45, 0.45, 0.14), k=0)
    moon.ell(mp + A(0.22, 0.1, 0), (0.38, 0.38, 0.25), k=0.02, cut=True)
    K.sdf_piece('enr_t3', moon, voxel=0.03, tris=300, mat='accent', bone='chest')

    # ---- SPEED: ribbon streamers (ankles, wrists) -> security blanket cape
    for sd in 'LR':
        s = -1 if sd == 'L' else 1
        hp, kn, an, toe = leg(sd)
        k = np.linspace(0, 1, 12)
        C = an[None] + A(s * 0.5, 0.45, -0.4)[None] + np.stack([s * (0.3 * k + 0.12 * np.sin(k * 6)), 0.55 * k + 0.12 * np.sin(k * 5), -1.9 * k], 1)
        W = np.stack([0 * k, 0.13 + 0.05 * k, 0 * k], 1)
        ribbon(K, 'spd_t1', C, W, mat='ribbon', bone='shin' + sd, notch=1.0)
        d = nrm(WR[sd] - EL[sd])
        C = (WR[sd] - d * 0.1)[None] + np.stack([s * (0.4 * k + 0.15 * np.sin(k * 5)), -0.6 * k - 0.4 * k ** 2, -1.8 * k + 0.2 * np.sin(k * 4)], 1)
        W = np.stack([0 * k + 0.02, 0.11 + 0.04 * k, 0 * k], 1)
        ribbon(K, 'spd_t2', C, W, mat='ribbon', bone='forearm' + sd, notch=1.0)
    top = [A(-1.2, 6.55, -0.45), A(-0.6, 6.75, -0.9), A(0, 6.8, -1.0), A(0.6, 6.75, -0.9), A(1.2, 6.55, -0.45)]
    cape_grid(K, 'spd_t3', top, tip_drop=4.4, back=2.6, flare=1.3, side_drift=0.5, folds=3.5, mat='blanket', bone='chest', seed=3, amp=0.18, hem=0.1)
    kn_ = A(0, 6.6, 0.95)
    K.ball('spd_t3', kn_ + A(0.0, 0, -0.05), A(0.3, 0.22, 0.2), mat='blanket', bone='chest')
    for s in (-1, 1):
        K.horn('spd_t3', [A(s * 1.2, 6.55, -0.45), A(s * 1.05, 6.6, 0.5), kn_], [0.11, 0.1, 0.09], mat='blanket', sides=7, bone='chest')

    # ---- FOCUS: glowing tear streaks, propeller beanie, third eye
    for s in (-1, 1):
        p0 = A(s * 0.66, 7.95, 1.68)
        pts, nn = K.snap(np.array([p0, p0 + A(s * 0.08, -0.3, -0.04), p0 + A(s * 0.14, -0.62, -0.12), p0 + A(s * 0.16, -0.9, -0.24)]))
        K.horn('fcs_t1', list(pts + nn * 0.03), [0.06, 0.07, 0.06, 0.03], mat='accent', sides=6, flat=0.4, up=nn[1], bone='head')
        K.ball('fcs_t1', pts[-1] + nn[-1] * 0.04 + A(0, -0.1, 0), A(0.09, 0.12, 0.09), mat='accent', bone='head')
    cap = K.sdf()
    cap.ell(HEAD + A(0, 0.25, -0.05), HEAD_R * A(1.05, 1.0, 1.05) + 0.06, k=0)
    cap.ell(HEAD + A(0, 0.25, -0.05), HEAD_R * A(1.05, 1.0, 1.05) - 0.06, k=0.0, cut=True)
    cap.box(HEAD + A(0, -0.25, 0.0), (2.2, 0.85, 2.2), rot=(-12, 0, 0), rad=0, k=0.03, cut=True)
    K.sdf_piece('fcs_t2', cap, voxel=0.045, tris=600, mat='toy2', bone='head', ao=0.3)
    for i in range(4):   # gores in alternating colour
        a = np.pi / 2 * i + np.pi / 4
        g = [HEAD + A(np.cos(a) * 1.25, 1.2, np.sin(a) * 1.25 - 0.05) + A(0, -0.0, 0), HEAD + A(np.cos(a) * 0.75, 1.65, np.sin(a) * 0.75 - 0.03), HEAD + A(0, 1.86, -0.05)]
        gp, gn = K.snap(np.array(g), sdf=cap)
        K.horn('fcs_t2', list(gp + gn * 0.01), [0.05, 0.04, 0.03], mat='toy', sides=5, bone='head')
    ptop = HEAD + A(0, 1.88, -0.05)
    K.horn('fcs_t2', [ptop, ptop + A(0, 0.45, 0)], [0.07, 0.06], mat='pin', sides=8, bone='head')
    K.ball('fcs_t2', ptop + A(0, 0.48, 0), 0.11, mat='accent', bone='head')
    for s in (-1, 1):
        bl = []
        Rb = frame(A(s, 0, 0.35), A(0, 1, 0))
        oball(K, 'fcs_t2', ptop + A(0, 0.48, 0) + Rb @ A(0, 0.55, 0), A(0.14, 0.5, 0.03), Rb @ S.rot_xyz((0, 25 * s, 0)), mat='toy', bone='head', seg=12, rings=8)
    tp, tn = K.snap(A(0, 9.02, 1.7))
    K.ball('fcs_t3', tp - tn * 0.04, A(0.2, 0.26, 0.12), mat='dark', bone='head', rot=None)
    K.ball('fcs_t3', tp + tn * 0.03, A(0.15, 0.2, 0.09), mat='accent', bone='head')
    K.ball('fcs_t3', tp + tn * 0.11, A(0.05, 0.12, 0.03), mat='dark', bone='head')
    for s in (-1, 1):   # glowing rune marks on the cheeks
        p, n = K.snap(A(s * 1.15, 7.35, 1.25))
        st = star_pts(p + n * 0.03, n, A(0, 1, 0), 0.17, 0.08, depth=0.025)
        K.piece('fcs_t3', st[0], st[1], 'accent', smooth=False, bone='head')
