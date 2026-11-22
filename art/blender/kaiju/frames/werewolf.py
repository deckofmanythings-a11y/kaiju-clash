"""Dire Lycan: hunched, hulking wolf-man (biped plan with tail). Game coords, ~10.8 units tall, faces +Z."""
import numpy as np

from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import fbm, smoothstep, voronoi

DETAIL = 'fur'
UV_SCALE = 0.75
EYE_ACCENT = True

INFO = dict(name='Dire Lycan', colors=dict(body='#7a7064', belly='#ebe2d2', accent='#5fc8ff'))

# key joints (right side, +X); L is mirrored. Arms held away from the torso so the waist reads.
SH = np.array([2.35, 8.35, 0.55])     # shoulder
EL = np.array([3.4, 6.1, -0.1])       # elbow
WR = np.array([3.6, 4.0, 0.8])        # wrist
PALM = np.array([3.62, 3.45, 1.0])
UAM = (SH + EL) * 0.5                 # upper-arm middle
FAM = (EL + WR) * 0.5                 # forearm middle
HIP = np.array([1.25, 5.3, -0.9])
KN = np.array([1.5, 3.6, 0.55])       # knee (forward)
HK = np.array([1.45, 1.55, -1.1])     # hock (back)
BALL = np.array([1.45, 0.38, 0.35])   # ball of the paw
FINGER_DZ = (-0.33, -0.11, 0.11, 0.33)
TOE_DX = (-0.33, -0.11, 0.11, 0.33)
TAIL = [(0, 5.35, -1.75), (0, 4.85, -2.7), (0.15, 4.25, -3.6), (0.45, 3.8, -4.5), (0.85, 3.6, -5.3), (1.2, 3.65, -5.95)]
TAIL_R = [0.5, 0.68, 0.82, 0.8, 0.6, 0.22]


HP = np.array([0, 9.2, 2.0])      # head pivot (authored head coords are scaled about this)
HSC = 1.2
HOFF = np.array([0, 0.15, 0.35])
EAR = [(0.47, 9.85, 2.42), (0.6, 10.4, 2.36), (0.76, 10.95, 2.22)]   # right ear spine (head space)


def H(p):
    """Authored head-space point(s) -> final position (bigger head, pushed forward)."""
    return HP + HOFF + (np.asarray(p, float) - HP) * HSC


class Xf:
    """Proxy that adds head-space primitives to an SDF program."""
    def __init__(self, B):
        self.B = B

    def tube(self, pts, radii, **kw):
        self.B.tube([H(q) for q in pts], [r * HSC for r in radii], **kw)

    def ell(self, c, r, **kw):
        self.B.ell(H(c), tuple(np.asarray(r, float) * HSC), **kw)


def M(s):
    return np.array([s, 1, 1])


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def finger_pts(s, dz):
    m = M(s)
    return [m * (PALM + (0.0, -0.45, dz)), m * (PALM + (0.02, -0.95, dz * 1.1 + 0.15)), m * (PALM + (-0.08, -1.25, dz + 0.3))]


def toe_pts(s, dx):
    return [M(s) * (BALL[0] + dx, 0.3, 0.7), M(s) * (BALL[0] + dx * 1.3, 0.2, 1.15)]


def torus(c, n, R, r, seg=20, sides=6):
    """Ring mesh around axis n."""
    c, n = np.asarray(c, float), unit(n)
    a = unit(np.cross(n, [0, 1, 0]) if abs(n[1]) < 0.9 else np.cross(n, [1, 0, 0]))
    b = np.cross(n, a)
    V = []
    for i in range(seg):
        u = 2 * np.pi * i / seg
        d = np.cos(u) * a + np.sin(u) * b
        for j in range(sides):
            v = 2 * np.pi * j / sides
            Ri = R[i] if np.ndim(R) else R
            V.append(c + d * (Ri + r * np.cos(v)) + n * r * np.sin(v))
    T = []
    for i in range(seg):
        for j in range(sides):
            p, q = i * sides + j, ((i + 1) % seg) * sides + j
            p2, q2 = i * sides + (j + 1) % sides, ((i + 1) % seg) * sides + (j + 1) % sides
            T += [(p, q, q2), (p, q2, p2)]
    return np.array(V), np.array(T)


def build(K):
    K.mat('belly', detail_kind='fur', rough=0.8)
    K.mat('plate', color='#6b5643', rough=0.7, metal=0.0)          # scarred leather / bone
    K.mat('iron', color='#8d8c92', rough=0.38, metal=0.85, vcol=True, detail=False)
    K.mat('horn', color='#e8e0cc', rough=0.35, vcol=True)

    B = K.sdf()
    # ---- torso: hunched, barrel chest thrust forward, wasp waist, huge shoulder hump
    B.tube([(0, 5.6, -0.95), (0, 6.5, -0.5), (0, 7.5, 0.2), (0, 8.4, 0.75)], [1.1, 0.8, 1.6, 1.65], k=0.0, s=(1.0, 1, 0.95))
    B.ell((0, 5.55, -0.95), (1.12, 0.95, 1.05), k=0.4)                                # hips
    B.ell((0, 7.85, 0.95), (1.65, 1.3, 1.3), k=0.45)                                 # rib cage (high, so the waist pinches)
    B.ell((0.85, 7.95, 1.95), (0.95, 0.75, 0.55), rot=(30, 0, 0), k=0.3, mirror=True)  # pectorals
    B.ell((0, 6.4, 0.25), (0.76, 0.8, 0.76), k=0.4)                                  # abdomen
    B.ell((0, 8.95, 0.15), (1.45, 0.9, 1.2), k=0.5)                                  # trapezius hump
    B.tube([(0, 9.0, 0.3), (1.3, 8.95, 0.5), (2.15, 8.6, 0.55)], [0.95, 0.88, 0.75], k=0.4, mirror=True)
    B.ell(SH, (1.0, 1.05, 1.05), k=0.3, mirror=True)                                  # deltoids
    B.ell((0, 8.2, -0.65), (1.45, 1.2, 1.05), rot=(-35, 0, 0), k=0.5)                # upper back
    B.ell((1.25, 7.8, -0.3), (0.7, 0.9, 0.9), rot=(-20, 0, 8), k=0.35, mirror=True)   # lats
    # ---- neck, thrust forward
    B.tube([(0, 8.5, 0.9), (0, 9.0, 1.75), H((0, 9.15, 2.2))], [1.0, 0.88, 0.8], k=0.4)
    # ---- arms: long, ropy, hanging past the knees, held out from the body
    B.tube([SH, UAM + (0.05, 0, 0.05), EL], [0.88, 0.7, 0.5], k=0.3, mirror=True)
    B.ell(UAM + (0.1, 0.05, 0.38), (0.5, 0.85, 0.5), rot=(-10, 0, 22), k=0.26, mirror=True)   # biceps
    B.ell(UAM + (0.3, -0.05, -0.42), (0.54, 0.95, 0.56), rot=(12, 0, 22), k=0.26, mirror=True)  # triceps
    B.ell(EL + (0.05, 0.0, -0.3), (0.42, 0.42, 0.4), k=0.18, mirror=True)               # elbow point
    B.tube([EL + (0, -0.15, 0), FAM + (0.12, 0, -0.05), WR], [0.46, 0.74, 0.38], k=0.24, mirror=True)
    B.ell(FAM + (0.17, 0.3, -0.1), (0.62, 0.92, 0.58), rot=(-18, 0, 5), k=0.24, mirror=True)   # forearm muscle
    B.ell(PALM, (0.38, 0.6, 0.48), k=0.25, mirror=True)                                 # palm (faces inward)
    for dz in FINGER_DZ:
        B.tube(finger_pts(1, dz), [0.16, 0.14, 0.11], k=0.06, mirror=True)
    B.tube([PALM + (-0.25, 0.1, 0.35), PALM + (-0.4, -0.35, 0.55), PALM + (-0.45, -0.7, 0.6)], [0.17, 0.14, 0.11], k=0.12, mirror=True)  # thumb
    # ---- legs: digitigrade, big thigh, knee forward, high hock, long paw
    B.tube([HIP, (1.5, 4.4, -0.1), KN], [1.15, 1.0, 0.6], k=0.35, mirror=True)
    B.ell((1.45, 4.55, -0.35), (0.76, 1.08, 0.88), rot=(-25, 0, 0), k=0.35, mirror=True)  # thigh mass
    B.ell(KN + (0, 0, 0.08), (0.45, 0.45, 0.42), k=0.2, mirror=True)                    # knee cap
    B.tube([KN, (1.48, 2.6, -0.3), HK], [0.55, 0.42, 0.3], k=0.25, mirror=True)
    B.ell((1.5, 2.95, -0.2), (0.48, 0.75, 0.55), rot=(-38, 0, 0), k=0.25, mirror=True)  # calf
    B.ell(HK + (0, 0.05, -0.12), (0.3, 0.34, 0.34), k=0.15, mirror=True)                # hock point
    B.tube([HK, (1.45, 0.95, -0.35), BALL], [0.3, 0.3, 0.34], k=0.2, s=(1.1, 1, 1), mirror=True)
    B.ell(BALL + (0, -0.04, 0.2), (0.55, 0.32, 0.55), k=0.2, mirror=True)               # paw pad
    for dx in TOE_DX:
        B.tube(toe_pts(1, dx), [0.17, 0.14], k=0.07, mirror=True)
    # ---- tail
    B.tube(TAIL, TAIL_R, k=0.4)
    # ---- head: wolf skull, long muzzle, open snarling jaws (authored small, scaled by H)
    HB = Xf(B)
    HB.ell((0, 9.35, 2.45), (0.8, 0.76, 0.92), k=0.3)                                 # cranium
    HB.tube([(0, 9.45, 2.85), (0, 9.25, 3.65), (0, 9.05, 4.4)], [0.55, 0.42, 0.3], k=0.25, s=(1.0, 0.8, 1))   # muzzle
    HB.tube([(-0.72, 9.72, 2.8), (-0.32, 9.7, 3.12), (0, 9.62, 3.18), (0.32, 9.7, 3.12), (0.72, 9.72, 2.8)], [0.17, 0.2, 0.17, 0.2, 0.17], k=0.14, s=(1, 0.7, 1))  # brow
    HB.ell((0.58, 9.0, 2.7), (0.42, 0.48, 0.55), k=0.25, mirror=True)                 # cheek / masseter
    HB.tube([(0, 8.55, 2.45), (0, 8.2, 3.35), (0, 7.98, 4.05)], [0.46, 0.32, 0.22], k=0.2, s=(1.1, 0.6, 1))   # lower jaw, gaping
    HB.ell((0, 8.55, 3.7), (0.4, 0.28, 0.85), rot=(14, 0, 0), cut=True, k=0.08)        # mouth
    HB.ell((0.43, 9.5, 3.12), (0.2, 0.15, 0.22), cut=True, k=0.06, mirror=True)       # eye sockets
    # snarl wrinkles across the top of the muzzle
    for i, z in enumerate((3.55, 3.8, 4.05)):
        HB.tube([(-0.33, 9.32 - i * 0.07, z), (0, 9.48 - i * 0.07, z + 0.04), (0.33, 9.32 - i * 0.07, z)], [0.04, 0.06, 0.04], k=0.05, cut=True)
    # ears: tall, pointed, pricked up and slightly splayed so they break the head silhouette
    HB.tube(EAR, [0.38, 0.27, 0.03], k=0.16, s=(1, 1, 0.42), mirror=True)

    smooth = K.sdf()           # unshagged, undisplaced body for armour plates
    smooth.ops = list(B.ops)
    smooth._lo, smooth._hi = B._lo.copy(), B._hi.copy()

    # ---- mane / ruff / tufts: fewer, bigger tapered SDF spikes (neck, shoulders, spine, tail only)
    rng = np.random.default_rng(3)

    def shag(anchors, direction, length, r, k=0.14, jitter=0.2, sink=0.15):
        P, N = smooth.snap(np.asarray(anchors, float))
        for p, n in zip(P, N):
            d = unit(np.asarray(direction, float) + n * 0.55 + rng.normal(0, jitter, 3))
            L = length * rng.uniform(0.8, 1.12)
            B.tube([p - n * sink, p + d * L * 0.5 + n * 0.05, p + d * L], [r, r * 0.55, 0.035], k=k, per=2)

    spine = S.catmull([(0, 10.0, 1.2), (0, 9.85, 0.5), (0, 9.4, -0.4), (0, 8.5, -1.2), (0, 7.4, -1.5), (0, 6.6, -1.6)], 1)
    for i, p in enumerate(spine):
        f = i / (len(spine) - 1)
        ln = 1.2 * (1 - 0.55 * f)
        shag([p], (0, 0.45 - 0.5 * f, -1), ln, 0.3 * (1 - 0.35 * f), jitter=0.1)
        if f < 0.6:
            for s in (-1, 1):
                shag([p + (s * 0.65, -0.15, 0.05)], (s * 0.55, 0.15 - 0.4 * f, -1), ln * 0.85, 0.26, jitter=0.12)
    # shoulder / neck ruff ring (kept below the collar line so the collar can sit over it)
    for a in np.radians(np.linspace(-70, 70, 5)):
        for s in (-1, 1):
            c = (s * 1.25 * np.cos(a * 0.5) + s * 0.1, 8.95 + 0.3 * np.cos(a), 1.0 + 0.9 * np.sin(a))
            shag([c], (s * 1.0, -0.4, -0.5), 0.85, 0.27, jitter=0.15)
    for s in (-1, 1):
        shag([(s * 1.9, 9.1, 0.3), (s * 2.6, 9.0, 0.2)], (s * 0.6, -0.4, -0.8), 0.8, 0.24)
        shag([H((s * 0.85, 9.0, 2.45)), H((s * 0.8, 8.8, 2.2))], (s * 1.0, -0.3, -0.6), 0.6, 0.17)   # cheek ruff
        shag([M(s) * (EL + (0.1, 0.0, -0.4))], (s * 0.3, -0.7, -0.8), 0.55, 0.17)                    # elbow tuft
    # throat / chest bib
    shag([(0, 8.4, 2.1), (-0.42, 8.15, 2.1), (0.42, 8.15, 2.1)], (0, -1, 0.35), 0.6, 0.2)
    # bushy tail: a few big tufts on the top / sides
    tc, tr = S.catmull(TAIL, 2), S.catmull(TAIL_R, 2)[:, 0]
    for i in range(3, len(tc) - 1, 2):
        for ang in (40, 140):
            a = np.radians(ang + i * 25)
            t = unit(tc[i + 1] - tc[i])
            side = unit(np.cross(t, [0, 1, 0]))
            up = np.cross(side, t)
            off = (np.cos(a) * side + np.sin(a) * up) * tr[i]
            shag([tc[i] + off], t * 1.4 + off * 0.8, 0.6 + tr[i] * 0.5, 0.22, jitter=0.08)
    shag([tc[-1]], unit(tc[-1] - tc[-2]), 0.65, 0.24)

    # ---- pale fur: throat, chest, belly, muzzle + cheeks, inner arms / thighs, tail underside, inner ears
    belly = K.sdf()
    belly.tube([(0, 8.7, 2.1), (0, 7.85, 2.4), (0, 6.8, 1.45), (0, 6.0, 0.75), (0, 5.15, 0.05)], [1.0, 1.4, 1.1, 0.9, 0.8], k=0)
    BH = Xf(belly)
    BH.tube([(0, 8.45, 2.6), (0, 8.1, 3.4), (0, 7.9, 4.1)], [0.5, 0.38, 0.3], k=0)            # lower jaw
    BH.tube([(0, 8.95, 3.2), (0, 8.9, 3.75), (0, 8.85, 4.3)], [0.5, 0.42, 0.32], k=0, s=(1.15, 0.5, 1))  # muzzle sides / upper lip
    BH.ell((0, 8.82, 2.65), (0.85, 0.32, 0.55), k=0)                                         # cheeks / throat
    for s in (-1, 1):
        m = M(s)
        BH.ell(m * ((EAR[0][0] + EAR[1][0]) / 2 - 0.02, 10.25, 2.5), (0.17, 0.42, 0.1), k=0)       # inner ears (front face)
        belly.tube([m * (UAM + (-0.6, 0, 0.2)), m * (EL + (-0.5, 0, 0.2)), m * (WR + (-0.36, 0, 0.15))], [0.45, 0.38, 0.32], k=0)  # inner arm
        belly.ell(m * (PALM + (-0.2, -0.4, 0.15)), (0.3, 0.7, 0.55), k=0)                    # palm pads
        belly.tube([m * (HIP + (-0.95, -0.3, 0.55)), m * (KN + (-0.62, 0.25, 0.2))], [0.5, 0.26], k=0)  # inner thigh
        belly.ell(m * (BALL + (0, -0.25, 0.3)), (0.7, 0.18, 0.8), k=0)                       # paw pads
    belly.tube([np.array(q) + (0, -0.5, 0.05) for q in TAIL], [0.35, 0.45, 0.55, 0.55, 0.45, 0.25], k=0)  # tail underside

    # mane weight: full shag on neck / shoulders / spine, calm fur elsewhere so anatomy reads
    def mane_w(x, y, z):
        top = smoothstep(7.7, 8.7, y) * (1 - smoothstep(2.4, 3.2, np.abs(x))) * (1 - smoothstep(1.5, 2.2, z))   # not the head
        dors = smoothstep(-0.5, -1.2, z) * smoothstep(6.3, 7.3, y) * (1 - smoothstep(0.9, 1.6, np.abs(x)))
        tail = 0.55 * smoothstep(-1.9, -2.6, z) * smoothstep(5.0, 4.4, y)
        return np.clip(np.maximum(np.maximum(top, dors), tail), 0, 1)

    def fur(x, y, z):
        P = np.stack([x, y, z], 1)
        w = smoothstep(0.1, -0.1, belly.at(P))
        f1, f2, _ = voronoi(x * 2.6, y * 1.1, z * 2.0, 1.0, seed=17)
        clump = smoothstep(0.0, 0.3, f2 - f1)
        g1, g2, _ = voronoi(x * 5.5, y * 2.4, z * 5.5, 1.0, seed=23)
        strand = smoothstep(0.0, 0.3, g2 - g1)
        hair = -0.075 * clump - 0.028 * strand + 0.038
        calm = -0.03 * clump - 0.014 * strand + 0.016
        soft = -0.025 * clump - 0.008 * fbm(x, y, z, 2, freq=4.0, seed=2) + 0.012
        mw = mane_w(x, y, z)
        return (hair * mw + calm * (1 - mw)) * (1 - w) + soft * w

    B.disp(fur, band=0.18)

    mouth = S.Ell(H((0, 8.55, 3.7)), (0.38 * HSC, 0.3 * HSC, 0.82 * HSC), rot=(14, 0, 0))

    def tint(P, N):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        # dark saddle / mane down the back and over the shoulders, darker tail top
        back = smoothstep(-0.1, 0.9, -z * 0.55 + (y - 7.3) * 0.6 - np.abs(x) * 0.25)   # upper back / shoulders, not the haunches
        back = np.maximum(back, mane_w(x, y, z) * smoothstep(-0.2, 0.6, N[:, 1] - N[:, 2] * 0.5))
        g = 1.0 - 0.4 * back
        # lighter fur tips on the clump ridges, darker roots in the gaps
        f1, f2, _ = voronoi(x * 2.6, y * 1.1, z * 2.0, 1.0, seed=17)
        clump = smoothstep(0.0, 0.3, f2 - f1)
        g = g * (0.8 + 0.3 * clump)
        # soft rim: silhouette-facing (sideways) normals a touch lighter
        g = g * (1.0 + 0.1 * (1 - np.abs(N[:, 1])))
        # darker lower legs / paws ("socks") (tier-0 contrast at the feet)
        g = g * (1.0 - 0.18 * smoothstep(2.2, 1.0, y))
        return g
    K.skin(B, voxel=0.05, tris=15000, belly=belly, regions=[('mouth', lambda P: mouth.dist(S.Points(P)))], tint=tint)

    # ---- skeleton
    K.bone('pelvis', (0, 5.3, -1.0), (0, 6.4, -0.5), None, 1.5)
    K.bone('spine', (0, 6.4, -0.5), (0, 7.4, 0.1), 'pelvis', 1.6)
    K.bone('chest', (0, 7.4, 0.1), (0, 8.6, 0.8), 'spine', 1.9)
    K.bone('neck', (0, 8.6, 0.8), H((0, 9.2, 2.0)), 'chest', 0.95)
    K.bone('head', H((0, 9.2, 2.0)), H((0, 9.1, 4.4)), 'neck', 0.95)
    K.bone('jaw', H((0, 8.6, 2.5)), H((0, 7.98, 4.1)), 'head', 0.5)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        K.bone('upperarm' + sd, m * SH, m * EL, 'chest', 0.9)
        K.bone('forearm' + sd, m * EL, m * WR, 'upperarm' + sd, 0.75)
        K.bone('hand' + sd, m * WR, m * (PALM + (-0.05, -1.2, 0.3)), 'forearm' + sd, 0.6)
        K.bone('thigh' + sd, m * HIP, m * KN, 'pelvis', 1.1)
        K.bone('shin' + sd, m * KN, m * HK, 'thigh' + sd, 0.55)
        K.bone('foot' + sd, m * HK, m * (BALL[0], 0.2, 1.2), 'shin' + sd, 0.45)
        K.mount('hand' + sd, m * (PALM + (0, -0.5, 0.15)), 'hand' + sd)
        K.mount('shoulder' + sd, m * (SH + (0, 1.0, -0.1)), 'chest')
    K.chain('tail', TAIL, 'pelvis', (0.7, 0.3))
    K.mount('head', H((0, 9.4, 2.5)), 'head')
    K.mount('jaw', H((0, 8.25, 3.2)), 'jaw')
    K.mount('mouth', H((0, 8.5, 4.5)), 'head')
    K.mount('chest', (0, 7.6, 2.4), 'chest')
    K.mount('back', (0, 8.4, -1.5), 'spine')
    K.mount('tailTip', TAIL[-1], 'tail5')
    K.mount('crown', H((0, 10.25, 2.35)), 'head')

    def hhorn(part, pts, radii, **kw):   # horn authored in head space
        K.horn(part, [H(q) for q in pts], [r * HSC for r in radii], **kw)

    def heye(part, c, n, r, **kw):
        K.eye(part, H(c), n, r * HSC, **kw)

    # ---- eyes, nose, small teeth (base, joined into the body)
    for s in (-1, 1):
        heye('body', (s * 0.43, 9.5, 3.16), (s * 0.5, 0.08, 1), 0.145, bone='head')
    K.ball('body', H((0, 9.1, 4.58)), (0.3, 0.21, 0.22), mat='dark', seg=10, rings=6, bone='head')
    for a in np.radians([-50, -28, -9, 9, 28, 50]):
        x, z = np.sin(a) * 0.3, 4.25 + np.cos(a) * 0.12
        hhorn('body', [(x, 8.86, z), (x, 8.77, z + 0.01), (x, 8.7, z)], [0.04, 0.03, 0], sides=5, bone='head', shade=(0.8, 1))
        hhorn('body', [(x * 0.85, 8.02, z - 0.25), (x * 0.85, 8.1, z - 0.24), (x * 0.85, 8.16, z - 0.26)], [0.035, 0.025, 0], sides=5, bone='jaw', shade=(0.8, 1))
    for s in (-1, 1):
        for i in range(2):   # premolar row
            z = 3.4 + i * 0.22
            hhorn('body', [(s * 0.33, 8.84, z), (s * 0.33, 8.72, z), (s * 0.32, 8.66, z)], [0.045, 0.03, 0], sides=4, bone='head', shade=(0.8, 1))
            hhorn('body', [(s * 0.27, 8.24 - i * 0.04, z - 0.15), (s * 0.27, 8.34 - i * 0.04, z - 0.15), (s * 0.26, 8.4 - i * 0.04, z - 0.15)], [0.04, 0.03, 0], sides=4, bone='jaw', shade=(0.8, 1))

    # ---- Might: hand + foot claws and fangs (exclusive, complete sets)
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        k = 0.7 + 0.3 * sc
        for s, sd in ((-1, 'L'), (1, 'R')):
            m = M(s)
            for dz in FINGER_DZ:
                tip = finger_pts(s, dz)[-1]
                K.claw(part, tip + (0, 0.05, 0), (-s * 0.15, -1, 0.35), 0.5 * k, 0.1 * (0.85 + 0.15 * sc), curl=(-s * 0.4, 0, 0.8), bone='hand' + sd)
            K.claw(part, m * (PALM + (-0.45, -0.7, 0.6)), (-s * 0.3, -0.7, 0.6), 0.36 * k, 0.09, curl=(0, -0.8, 0.2), bone='hand' + sd)
            for dx in TOE_DX:
                b = toe_pts(s, dx)[-1]
                K.claw(part, b + (0, 0.02, 0.05), (s * dx * 0.3, -0.25, 1), 0.38 * k, 0.1 * (0.85 + 0.15 * sc), curl=(0, -1, 0.1), bone='foot' + sd)
            K.claw(part, m * (BALL + (-0.42, 0.45, -0.45)), (-s * 0.4, -0.3, 0.2), 0.22 * k, 0.07, bone='foot' + sd)   # dewclaw
            # fangs: upper canines (sabres at t3), lower canines
            ul = [0.36, 0.48, 0.62, 1.0][t]
            ub = np.array([s * 0.3, 8.86, 4.08])
            hhorn(part, [ub, ub + [s * 0.02, -ul * 0.55, 0.02], ub + [s * 0.0, -ul, -0.04 - 0.08 * (t == 3)]], [0.075 + 0.012 * t, 0.055, 0], sides=6, per=3, bone='head', shade=(0.75, 1))
            ll = [0.28, 0.36, 0.46, 0.5][t]
            lb = np.array([s * 0.25, 8.12, 3.82])
            hhorn(part, [lb, lb + [s * 0.03, ll * 0.6, 0.02], lb + [s * 0.04, ll, -0.04]], [0.065 + 0.008 * t, 0.045, 0], sides=6, per=3, bone='jaw', shade=(0.75, 1))
            if t >= 2:   # bony spurs on the knuckles of the wrist
                p, n = K.snap(m * (WR + (0.35, 0.1, -0.2)))
                K.horn(part, [p - n * 0.1, p + n * 0.3 + [0, 0.1, -0.2], p + n * 0.45 + [0, 0.35 * sc, -0.55]], [0.1, 0.07, 0], bone='forearm' + sd, per=3)

    def limb_radius(c, axis):   # outer radius of the (unshagged) limb around an axis point
        axis = unit(axis)
        a = unit(np.cross(axis, [0, 1, 0]) if abs(axis[1]) < 0.9 else np.cross(axis, [1, 0, 0]))
        b = np.cross(axis, a)
        ring = np.array([c + (np.cos(u) * a + np.sin(u) * b) * 0.25 for u in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
        P, _ = K.snap(ring, smooth)
        d = np.linalg.norm(np.cross(P - c, axis), axis=1)
        return float(np.clip(np.percentile(d, 85), 0.3, 1.0))

    # ---- Armour: leather/bone pauldrons + bracers, broken shackles, spiked collar
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        K.plate('arm_t1', m * (SH + (0.3, 0.45, 0.0)), (1.15, 0.95, 1.2), bone='upperarm' + sd, lames=3, sdf=smooth, off=0.12)
        a0, a1 = m * (FAM + (0.11, 0.05, 0.02)), m * (WR + (0.03, 0.22, -0.06))
        rr = [limb_radius(a0 * (1 - f) + a1 * f, a1 - a0) + 0.07 for f in (0.0, 0.5, 1.0)]
        K.horn('arm_t1', [a0, a0 * 0.5 + a1 * 0.5, a1], rr, mat='plate', sides=10, per=2, bone='forearm' + sd, shade=(0.7, 1.0))
        for q, r0 in ((a0, rr[0]), (a1, rr[2])):
            V, T = torus(q, a1 - a0, r0 + 0.02, 0.06, seg=20, sides=5)
            K.piece('arm_t1', V, T, 'iron', bone='forearm' + sd)
        # broken shackle ring on the wrist + dangling chain links
        V, T = torus(m * (WR + (0, -0.1, 0.05)), (-s * 0.05, 1, -0.3), limb_radius(m * (WR + (0, -0.1, 0.05)), (-s * 0.05, 1, -0.3)) + 0.1, 0.1, seg=20, sides=6)
        K.piece('arm_t2', V, T, 'iron', bone='forearm' + sd)
        c0 = m * (WR + (0.3, -0.25, -0.4))
        for i in range(3):
            ax = (0, 0, 1) if i % 2 == 0 else (1, 0, 0)
            V, T = torus(c0 + np.array([0, -0.28 * i, -0.05 * i]), ax, 0.15, 0.045, seg=12, sides=5)
            K.piece('arm_t2', V, T, 'iron', bone='forearm' + sd)
        for off, ln in (((0.25, 0.95, 0.0), 0.7), ((0.75, 0.45, 0.25), 0.5), ((0.45, 0.7, -0.55), 0.5)):
            p, n = K.snap(m * (SH + off), smooth)
            K.spike('arm_t3', p + n * 0.1, n + np.array([s * 0.2, 0.5, -0.1]), ln, mat='iron', bone='upperarm' + sd)
        p, n = K.snap(m * (FAM + (0.45, -0.35, 0.0)), smooth)
        K.spike('arm_t3', p + n * 0.1, n + [0, 0.1, -0.4], 0.45, mat='iron', bone='forearm' + sd)
    for s in (-1, 1):   # crossed leather harness (front + back)
        for zs in (1, -1):
            pts = np.array([(s * 1.6, 9.0, 0.6 * zs + 0.4), (s * 0.9, 8.1, 1.7 if zs > 0 else -1.0), (0.0, 7.2, 2.0 if zs > 0 else -1.25), (-s * 0.85, 6.3, 1.2 if zs > 0 else -1.2)])
            pts = S.catmull(pts, 3)
            P, N = K.snap(pts, smooth)
            P = P + N * 0.1
            K.horn('arm_t2', list(P), [0.14] * len(P), mat='plate', sides=6, per=1, flat=0.35, bone='chest' if zs > 0 else 'spine', shade=(0.75, 1.0))
            for j in (2, 5, 8):
                K.ball('arm_t2', P[j] + N[j] * 0.06, 0.07, mat='iron', seg=6, rings=4, bone='chest' if zs > 0 else 'spine')
    # tier 3: big spiked iron collar fitted OVER the ruff (radius measured per angle on the furred body)
    cc, cn = np.array([0, 9.0, 1.5]), unit((0, 0.75, 1))
    a = unit(np.cross(cn, [0, 1, 0]))   # same basis as torus()
    b = np.cross(cn, a)
    angs = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    rs = np.linspace(0.5, 1.75, 26)
    D = np.array([np.cos(u) * a + np.sin(u) * b for u in angs])
    Q = cc + D[:, None, :] * rs[None, :, None]
    inside = B.at(Q.reshape(-1, 3)).reshape(len(angs), len(rs)) < 0
    rad = np.array([rs[np.nonzero(row)[0].max()] if row.any() else 0.9 for row in inside])
    inner = smooth.at(Q.reshape(-1, 3)).reshape(len(angs), len(rs)) < 0   # don't climb over the spine mane
    rad = np.minimum(rad, np.array([rs[np.nonzero(row)[0].max()] if row.any() else 0.9 for row in inner]) + 0.3)
    rad = np.convolve(np.r_[rad[-2:], rad, rad[:2]], np.ones(5) / 5, 'valid') + 0.1
    V, T = torus(cc, cn, rad, 0.2, seg=28, sides=7)
    K.piece('arm_t3', V, T, 'iron', bone='neck')
    for i in range(2, 28, 4):
        K.spike('arm_t3', cc + D[i] * (rad[i] + 0.12), D[i] + cn * 0.15, 0.6, r=0.12, mat='iron', bone='neck')

    # ---- Energy: glowing claw-mark scars (chest, arms), glowing spine tufts
    def slash(part, a, b, bone, r=0.055, push=0.02):
        pts = np.linspace(a, b, 6)
        P, N = K.snap(pts, smooth)
        P = P + N * push
        K.horn(part, list(P), [0.0, r * 0.8, r, r, r * 0.7, 0.0], mat='accent', sides=5, per=3, bone=bone, shade=(1, 1))
    for i, dx in enumerate((-0.3, 0.0, 0.3)):
        slash('enr_t1', (0.95 + dx, 8.5, 2.2), (-0.15 + dx, 7.05, 2.4), 'chest')
    for s, sd in ((-1, 'L'), (1, 'R')):
        for dz in (-0.18, 0.12):
            slash('enr_t2', M(s) * (UAM + (0.55, 0.45, 0.05 + dz)), M(s) * (UAM + (0.5, -0.55, 0.5 + dz)), 'upperarm' + sd, r=0.045)
            slash('enr_t2', M(s) * (FAM + (0.5, 0.55, -0.2 + dz)), M(s) * (FAM + (0.45, -0.45, 0.25 + dz)), 'forearm' + sd, r=0.045)
    for i, p in enumerate(S.catmull([(0, 9.95, 1.4), (0, 9.7, 0.3), (0, 9.2, -0.6), (0, 8.3, -1.3), (0, 7.3, -1.6)], 1)):
        q, n = K.snap(p)
        d = unit(n * 0.6 + np.array([0, 0.35, -1]))
        L = 0.9 - 0.08 * i
        K.horn('enr_t3', [q - n * 0.1, q + d * L * 0.5 + n * 0.08, q + d * L], [0.11, 0.07, 0], mat='accent', sides=5, per=3, bone='chest' if i < 3 else 'spine', shade=(1, 1))
    slash('enr_t3', H((-0.45, 9.85, 2.75)), H((0.45, 9.85, 2.75)), 'head', r=0.045)

    # ---- Speed: wind-swept fur streamers (forearms, calves, tail), ear tufts
    srng = np.random.default_rng(11)

    def streamer(part, c, back, n_spikes, ln, bone, spread=0.3):
        p0, _ = K.snap(c)
        for i in range(n_spikes):
            off = srng.normal(0, spread, 3) * [0.6, 1, 0.6]
            p, n = K.snap(p0 + off)
            d = unit(np.asarray(back, float) + n * 0.4)
            L = ln * srng.uniform(0.8, 1.1)
            K.horn(part, [p - n * 0.12, p + d * L * 0.45 + n * 0.08, p + d * L + [0, -0.2 * L, 0]], [0.13, 0.09, 0], mat='body', sides=5, per=3, bone=bone, shade=(0.45, 1.0))
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = M(s)
        streamer('spd_t1', m * (FAM + (0.4, 0.0, -0.45)), (s * 0.3, -0.25, -1), 4, 0.85, 'forearm' + sd)
        streamer('spd_t2', m * (1.5, 2.8, -0.55), (s * 0.2, -0.2, -1), 4, 0.75, 'shin' + sd)
        streamer('spd_t3', H((s * 0.66, 10.55, 2.2)), (s * 0.4, 0.35, -1), 3, 0.55, 'head', spread=0.06)   # ear tufts (back of the ears)
    for i in (5, 7, 9):
        streamer('spd_t2', tc[i], unit(tc[i] - tc[i - 1]) + [0, 0.15, 0], 3, 0.9, f'tail{min(5, i // 2)}', spread=0.15)
    streamer('spd_t3', tc[-1], unit(tc[-1] - tc[-2]), 4, 1.1, 'tail5', spread=0.12)

    # ---- Focus: extra eyes, crescent-moon brand, raised hackle crest + third eye
    for s in (-1, 1):
        heye('fcs_t1', (s * 0.62, 9.68, 2.92), (s * 0.7, 0.3, 0.7), 0.075, bone='head', lid=False)
        heye('fcs_t3', (s * 0.3, 9.25, 3.72), (s * 0.6, 0.3, 0.8), 0.06, bone='head', lid=False)
    # crescent moon on the forehead
    ang = np.radians(np.linspace(-120, 120, 9))
    cpts = np.stack([np.sin(ang) * 0.24, 9.9 + np.cos(ang) * 0.18, np.full(9, 2.7)], 1)
    P, N = K.snap(H(cpts))
    K.horn('fcs_t2', list(P + N * 0.03), [0.0, 0.03, 0.05, 0.065, 0.07, 0.065, 0.05, 0.03, 0.0], mat='accent', sides=5, per=2, bone='head', shade=(1, 1))
    # raised hackles: tall dark spikes with glowing tips along the neck
    for i, p in enumerate([(0, 10.05, 1.15), (0, 9.95, 0.6), (0, 9.7, 0.05)]):
        q, n = K.snap(p)
        d = unit(n + np.array([0, 0.5, -0.4]))
        L = 1.05 - 0.15 * i
        K.horn('fcs_t3', [q - n * 0.1, q + d * L * 0.55, q + d * L * 0.8], [0.13, 0.08, 0.04], mat='dark', sides=5, bone='neck', shade=(0.6, 1))
        K.horn('fcs_t3', [q + d * L * 0.78, q + d * L * 0.9, q + d * L * 1.05], [0.045, 0.035, 0], mat='accent', sides=5, bone='neck', shade=(1, 1))
    heye('fcs_t3', (0, 9.78, 3.0), (0, 0.5, 0.85), 0.09, bone='head', lid=False)
