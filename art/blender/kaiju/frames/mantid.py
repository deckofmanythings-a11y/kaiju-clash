"""Mantid: giant praying mantis kaiju, reared up in a raised-scythe threat pose. Game coords, ~10 units tall, faces +Z."""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..materials import hex_rgb
from ..noise import fbm, smoothstep

DETAIL = 'chitin'
UV_SCALE = 0.7
EYE_ACCENT = True
CAM = dict(az=38, el=10)

# abdomen centreline (thorax -> tip, curling up at the end) and radii: fat, segmented
ABD = np.array([(0, 3.8, -0.85), (0, 3.62, -2.05), (0, 3.45, -3.45), (0, 3.5, -4.85), (0, 3.85, -6.05), (0, 4.4, -6.85)])
ABD_R = np.array([1.1, 1.62, 1.82, 1.66, 1.15, 0.38])
SY = 0.78   # abdomen vertical squash

# walking legs: (hip, knee, ankle, toe) for the +X side -- high knees, wide planted stance
MID = [(0.95, 3.85, 0.55), (3.4, 5.5, 1.35), (4.65, 0.95, 2.2), (5.4, 0.07, 2.95)]
HIND = [(0.95, 3.7, -0.5), (3.5, 5.35, -2.1), (4.6, 0.95, -3.85), (5.15, 0.07, -5.05)]

# reared prothorax (tilts forward ~24 deg)
PRO = [(0, 4.3, 0.75), (0, 5.3, 1.2), (0, 6.3, 1.65), (0, 7.1, 2.0), (0, 7.6, 2.25)]
PRO_T = 24.0
PRO_UP = np.array([0, np.cos(np.radians(PRO_T)), np.sin(np.radians(PRO_T))])
PRO_BACK = np.array([0, np.sin(np.radians(PRO_T)), -np.cos(np.radians(PRO_T))])

# raised raptorial forelegs (+X side): shoulder, coxa end, knee (held high beside the head)
SHO = np.array((0.8, 7.1, 2.2))
COX = np.array((1.7, 6.3, 2.95))
KNEE = np.array((3.35, 9.0, 3.75))

# head: lowered and pitched forward; authored in a head-local frame (x right, y up, z = face)
HS = 1.42
PITCH = 14.0
HC = np.array([0, 8.3, 3.3])
HR = S.rot_xyz((PITCH, 0, 0))


def H(p):
    """Head-local -> game coords."""
    return HC + (np.asarray(p, float) * HS) @ HR.T


def HD(d):
    """Head-local direction -> game."""
    return np.asarray(d, float) @ HR.T



def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def blade_mesh(pts, widths, thick, plane_n, edge_sign=1.0, back=0.5, per=4):
    """Flat curved blade along a spline. Cross-section is a diamond: sharp edge on +E, thicker spine on -E.
    plane_n = normal of the blade plane. Last width 0 -> single tip vertex. Returns V, T, centreline, tangents."""
    C = S.catmull(pts, per)
    n = len(C)
    t = np.linspace(0, 1, n)
    w = np.interp(t, np.linspace(0, 1, len(widths)), widths)
    th = np.interp(t, np.linspace(0, 1, len(thick)), thick)
    tan = np.gradient(C, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    pn = np.asarray(plane_n, float)
    V, ref, rings = [], [], []
    for i in range(n):
        X = unit(pn - tan[i] * (pn @ tan[i]))
        E = unit(np.cross(tan[i], X)) * edge_sign
        if i == n - 1 and w[i] < 1e-3:
            rings.append([len(V)])
            V.append(C[i]); ref.append(C[i] - tan[i] * 0.01)
            continue
        rings.append(list(range(len(V), len(V) + 4)))
        V += [C[i] + E * w[i], C[i] + X * th[i], C[i] - E * w[i] * back, C[i] - X * th[i]]
        ref += [C[i]] * 4
    T = []
    for i in range(n - 1):
        a, b = rings[i], rings[i + 1]
        for j in range(4):
            j2 = (j + 1) % 4
            if len(b) == 1:
                T.append((a[j], a[j2], b[0]))
            else:
                T += [(a[j], a[j2], b[j2]), (a[j], b[j2], b[j])]
    c0 = len(V); V.append(C[0]); ref.append(C[0] + tan[0] * 0.01)
    for j in range(4):
        T.append((c0, rings[0][(j + 1) % 4], rings[0][j]))
    V, T, ref = np.array(V), np.array(T), np.array(ref)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    fc = V[T].mean(1) - ref[T].mean(1)
    flip = (fn * fc).sum(1) < 0
    T[flip] = T[flip][:, ::-1]
    return V, T, C, tan


def fan_wing(K, part, base, a, b, R, veins=7, cup=0.18, bone=None, mat='membrane', spot=None):
    """Fan-folded membrane wing: spans directions a -> b from `base`, scalloped edge, dark veins in vertex colour.
    spot=(u, v, r) adds a glowing eyespot."""
    base, a, b = np.asarray(base, float), unit(a), unit(b)
    nrm = unit(np.cross(a, b))
    nu, nv = 7, veins * 4
    V, col = [], []
    for i in range(nu + 1):
        u = i / nu
        for j in range(nv + 1):
            v = j / nv
            ang = np.arccos(np.clip(a @ b, -1, 1))
            d = (np.sin((1 - v) * ang) * a + np.sin(v * ang) * b) / np.sin(ang)
            f = (v * veins) % 1.0
            dd = min(f, 1 - f)
            r = R * u * (0.86 + 0.14 * np.cos(np.pi * 2 * f) * 0.5 + 0.07) * (0.8 + 0.2 * np.sin(np.pi * v))
            p = base + d * r + nrm * cup * R * np.sin(np.pi * v) * u * u + nrm * 0.08 * R * u * (dd < 0.13)
            V.append(p)
            col.append((0.45 + 0.55 * smoothstep(0.0, 0.14, dd)) * (0.55 + 0.45 * u))
    T = []
    for i in range(nu):
        for j in range(nv):
            q = i * (nv + 1) + j
            T += [(q, q + nv + 1, q + nv + 2), (q, q + nv + 2, q + 1)]
    K.piece(part, np.array(V), np.array(T), mat, col=np.array(col), bone=bone)
    if spot:
        u, v, r = spot
        ang = np.arccos(np.clip(a @ b, -1, 1))
        d = (np.sin((1 - v) * ang) * a + np.sin(v * ang) * b) / np.sin(ang)
        c = base + d * R * u + nrm * cup * R * np.sin(np.pi * v) * u * u
        x = unit(np.cross(nrm, d))
        Rm = np.stack([x, nrm, d], 1)   # local -> world
        from .. import geom
        Vs, Ts, _ = geom.sphere((0, 0, 0), (r, 0.05, r * 0.8), 12, 6)
        K.piece(part, c + Vs @ Rm.T, Ts, 'accent', bone=bone)
        Vs, Ts, _ = geom.sphere((0, 0, 0), (r * 0.45, 0.07, r * 0.38), 10, 6)
        K.piece(part, c + Vs @ Rm.T, Ts, 'dark', bone=bone)



def compound_eye(K, part, c, Rl, r, gaze, out, bone=None, nu=28, nv=16, pupil=0.0, point=0.3, seed=0):
    """Structured compound eye: a dark dome tiled with flat-shaded glowing facets (dark gaps between them),
    a dark pseudo-pupil cluster facing `gaze`, and a dark socket rim. Rl = 3x3 (columns = eye-local x, y, z
    in game coords), r = local radii, out = outward direction from the head (buried facets are skipped)."""
    c, r, gaze, out = np.asarray(c, float), np.asarray(r, float), unit(gaze), unit(out)
    rng = np.random.default_rng(seed)

    def P(ph, th):
        p = np.array([np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)])
        q = p * r
        if p[1] > 0:                      # pointed, swept-up top
            q[1] *= 1 + point * p[1] ** 3
        return q

    V, T, col, mats = [], [], [], []
    facets = {'eye': ([], [], []), 'dark': ([], [], [])}
    for i in range(nv):
        for j in range(nu):
            ph0, ph1 = np.pi * i / nv, np.pi * (i + 1) / nv
            off = 0.5 * (i % 2)            # staggered rows read as a honeycomb
            th0, th1 = 2 * np.pi * (j + off) / nu, 2 * np.pi * (j + 1 + off) / nu
            Q = np.array([P(ph0, th0), P(ph0, th1), P(ph1, th1), P(ph1, th0)])
            cen = Q.mean(0)
            nl = unit(cen / (r * r))
            nw = Rl @ nl
            if nw @ out < -0.15:
                continue
            Q = cen + (Q - cen) * 0.9 + nl * 0.018 * r.mean()
            W = c + Q @ Rl.T
            mat = 'dark' if np.degrees(np.arccos(np.clip(nw @ gaze, -1, 1))) < pupil else 'eye'
            Vv, Tt, Cc = facets[mat]
            o = len(Vv)
            Vv.extend(W)
            if i == 0 or i == nv - 1:      # pole: degenerate quad -> one triangle
                Tt.append((o, o + 2, o + 3) if i == 0 else (o, o + 1, o + 2))
            else:
                Tt.extend([(o, o + 1, o + 2), (o, o + 2, o + 3)])
            Cc.extend([rng.uniform(0.6, 1.0)] * 4)
    for mat, (Vv, Tt, Cc) in facets.items():
        if Tt:
            Vv, Tt = np.array(Vv), np.array(Tt)
            # outward winding per triangle
            fn = np.cross(Vv[Tt[:, 1]] - Vv[Tt[:, 0]], Vv[Tt[:, 2]] - Vv[Tt[:, 0]])
            fl = (fn * (Vv[Tt].mean(1) - c)).sum(1) < 0
            Tt[fl] = Tt[fl][:, ::-1]
            K.piece(part, Vv, Tt, mat, smooth=False, col=np.array(Cc), bone=bone)
    # dark dome under the facets + socket rim
    Vs, Ts, _ = geom.sphere((0, 0, 0), 1, 12, 8)
    q = Vs * r * 0.97
    q[:, 1] *= 1 + point * np.clip(Vs[:, 1], 0, 1) ** 3
    K.piece(part, c + q @ Rl.T, Ts, 'dark', bone=bone)
    # pseudo-pupil: a round dark spot on the surface that seems to follow the viewer
    gl = Rl.T @ gaze
    sp = gl / np.sqrt(((gl / r) ** 2).sum())
    x, y, z = geom.basis(Rl @ unit(sp / (r * r)), Rl[:, 1])
    Vd, Td, _ = geom.sphere((0, 0, 0), 1, 12, 6)
    rp = r.mean() * 0.24
    K.piece(part, c + Rl @ sp + (Vd[:, [0]] * x * rp * 0.42 + Vd[:, [1]] * y * 0.03 + Vd[:, [2]] * z * rp * 1.9), Td, 'dark', bone=bone)
    K.piece(part, c - gaze * r.mean() * 0.42 + (Vs * r * [1.06, 1.04, 1.06]) @ Rl.T, Ts, 'dark', bone=bone)


def build(K):
    body, belly_c, acc = (hex_rgb(K.colors[k]) for k in ('body', 'belly', 'accent'))
    K.mat('body', rough=0.26, detail_strength=0.5)
    K.mat('belly', rough=0.34, detail_strength=0.3)
    K.mat('plate', color=body * 0.5 + belly_c * 0.3, rough=0.24)
    K.mat('membrane', color=body * 0.45 + acc * 0.4, rough=0.45)
    K.mat('eye', color=acc * 0.7, emit=acc, emit_strength=1.6, rough=0.15, vcol=True)
    K.mat('horn', color=belly_c * 0.45 + np.array([0.88, 0.86, 0.72]) * 0.55, rough=0.22)

    B = K.sdf()
    # ---------------------------------------------------------------- meso/metathorax: chunky barrel
    B.tube([(0, 3.72, -0.95), (0, 3.98, -0.1), (0, 4.28, 0.75)], [1.08, 1.28, 1.02], k=0.1, s=(1.1, 0.92, 1))
    B.ell((0, 4.78, -0.2), (0.86, 0.42, 0.9), k=0.18)                                    # mesonotum hump
    # ---------------------------------------------------------------- reared, armoured prothorax
    B.tube(PRO, [0.92, 0.76, 0.76, 0.86, 0.62], k=0.14, s=(1.1, 1, 1))
    pm = np.array(PRO[2]) + PRO_BACK * 0.2
    B.ell(pm - PRO_UP * 0.4 + PRO_BACK * 0.12, (1.18, 1.85, 0.42), rot=(PRO_T, 0, 0), k=0.16)                # pronotum shield (lateral flanges)
    B.ell(pm + PRO_BACK * 0.42 - PRO_UP * 0.3, (0.18, 1.55, 0.2), rot=(PRO_T, 0, 0), k=0.1)  # dorsal keel
    B.ell((0, 7.12, 2.02), (1.08, 0.64, 0.62), rot=(PRO_T, 0, 0), k=0.14)                  # pronotal flare (foreleg base)
    B.ell(np.array(PRO[1]) - PRO_BACK * 0.35, (0.62, 0.9, 0.5), rot=(PRO_T, 0, 0), k=0.2)   # deep chest / prosternum
    B.tube([(0, 7.5, 2.28), (0, 7.95, 2.85)], [0.44, 0.4], k=0.08)                        # neck
    # ---------------------------------------------------------------- triangular head (head-local, pitched down)
    HRm = HR
    B.ell(H((0, 0, 0)), np.array((0.98, 0.5, 0.48)) * HS, rot=HRm, k=0.08)
    B.tube([H((0, -0.15, 0.14)), H((0, -0.6, 0.3)), H((0, -1.02, 0.4))], np.array([0.6, 0.42, 0.25]) * HS, k=0.14, s=(1.25, 1, 0.9))
    B.ell(H((0.82, 0.06, 0.02)), np.array((0.36, 0.42, 0.38)) * HS, rot=HRm, k=0.1, mirror=True)       # eye socket bulge
    B.ell(H((0, -0.42, 0.42)), np.array((0.36, 0.3, 0.12)) * HS, rot=HRm, k=0.06)                     # clypeus plate
    B.ell(H((0, -0.84, 0.42)), np.array((0.26, 0.16, 0.12)) * HS, rot=HRm, k=0.06)                    # labrum
    B.ell(H((0, 0.42, 0.06)), np.array((0.4, 0.15, 0.3)) * HS, rot=HRm, k=0.08)                       # vertex ridge
    for s in (-1, 1):
        B.tube([H((s * 0.12, 0.3, 0.42)), H((s * 0.5, 0.42, 0.38)), H((s * 0.8, 0.62, 0.22))], np.array([0.1, 0.12, 0.07]) * HS, k=0.08)  # angry brow ridges
        B.ell(H((s * 1.0, 0.1, 0.05)), np.array((0.4, 0.52, 0.44)) * HS, rot=HR @ S.rot_xyz((0, 0, -s * 22)), cut=True, k=0.04)          # eye seats
    # ---------------------------------------------------------------- raptorial forelegs: coxa + massive femur (tibia blade is pow)
    B.tube([SHO, (1.2, 6.7, 2.65), COX], [0.42, 0.4, 0.34], k=0.08, mirror=True)
    B.ell(COX, (0.36, 0.36, 0.36), k=0.06, mirror=True)                                    # trochanter
    fd = unit(KNEE - COX)
    B.tube([COX, COX + (KNEE - COX) * 0.45 + [0.05, 0, 0.15], KNEE], [0.36, 0.62, 0.3], k=0.08, s=(0.62, 1, 1), mirror=True)
    B.ell(KNEE, (0.26, 0.3, 0.3), k=0.05, mirror=True)
    # ---------------------------------------------------------------- walking legs: thick femurs
    for hip, kn, an, toe in (MID, HIND):
        hip, kn, an = np.array(hip), np.array(kn), np.array(an)
        B.ell(hip, (0.52, 0.46, 0.52), k=0.08, mirror=True)                                # coxa
        mid = hip * 0.45 + kn * 0.55 + [0, 0.35, 0]
        B.tube([hip, mid, kn], [0.5, 0.56, 0.36], k=0.06, mirror=True)
        B.ell(kn, (0.37, 0.37, 0.37), k=0.04, mirror=True)
        B.tube([kn, kn * 0.6 + an * 0.4, an], [0.4, 0.34, 0.25], k=0.04, mirror=True)
        B.ell(an, (0.27, 0.27, 0.27), k=0.03, mirror=True)
    # ---------------------------------------------------------------- abdomen: continuous fat tube, tergites by displacement
    B.tube(ABD, ABD_R, k=0.14, s=(1, SY, 1), per=8)
    Cd = S.catmull(ABD, 12)
    sarr = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Cd, axis=0), axis=1))])
    seglen = sarr[-1] / 7.0
    abd = K.sdf()
    abd.tube(ABD, ABD_R * 1.04, k=0, s=(1, SY, 1))
    # folded wing cases (tegmina) over the front of the abdomen; the segmented rear stays bare
    WB = dict(c=(0, 5.05, -2.5), h=(1.55, 1.05, 1.85), rot=(-6, 0, 0))
    wreg = S.Box(WB['c'], WB['h'], rot=WB['rot'], rad=0.55)
    B.add(S.Inter(S.Shell(abd, 0.12, 0.06), wreg, 0.04), 0.0)
    B.box((0, 5.5, -2.5), (0.016, 0.8, 2.1), rad=0.01, k=0.0, cut=True)                 # seam between the cases
    wingm = S.Inter(S.Shell(abd, 0.12, 0.16), S.Box(WB['c'], np.array(WB['h']) + 0.05, rot=WB['rot'], rad=0.55))
    shield = S.Ell(pm - PRO_UP * 0.4 + PRO_BACK * 0.22, (1.24, 1.92, 0.56), rot=(PRO_T, 0, 0))

    # ---------------------------------------------------------------- sculpt detail
    def chit(x, y, z):
        P = np.stack([x, y, z], 1)
        base = -0.016 * fbm(x, y, z, 3, freq=1.4, seed=4)
        ww = smoothstep(0.05, -0.05, wingm.dist(S.Points(P)))
        ven = 1 - smoothstep(0.0, 0.08, np.abs(np.sin((np.abs(x) * 2.6 + (z + 3) * 0.1) * np.pi)))
        cross = 1 - smoothstep(0.0, 0.05, np.abs(np.sin((z * 1.3 + np.abs(x) * 0.9) * np.pi)))
        vein = -0.02 * np.maximum(ven, cross * 0.5)
        # abdominal tergites/sternites: sawtooth (each plate flares toward its rear edge) + a deep groove at each joint
        s = np.interp(-z, -Cd[:, 2], sarr)
        f = (s / seglen + 0.3) % 1.0
        wa = smoothstep(-0.7, -1.2, z) * (1 - ww)
        terg = (-0.11 * f ** 1.6 + 0.07 * (1 - smoothstep(0.0, 0.08, f)) + 0.03) * wa
        # pronotum: lames across the shield (sawtooth along the prothorax axis)
        sh = smoothstep(0.06, -0.12, shield.dist(S.Points(P)))
        a = (P @ PRO_UP) / 0.55
        g = a % 1.0
        lam = (-0.05 * g ** 1.5 + 0.035 * (1 - smoothstep(0.0, 0.1, g)) + 0.012) * sh * (z < 1.9)
        return base * (1 - ww) + vein * ww + terg + lam
    B.disp(chit, band=0.15)

    # ---------------------------------------------------------------- material regions
    belly_m = K.sdf()
    belly_m.add(S.Inter(S.Tube(ABD[1:], ABD_R[1:] * 1.1, s=(0.95, SY, 1)), S.Box((0, 2.0, -3.9), (1.9, 1.05, 3.6), rot=(-4, 0, 0)), 0.0), 0)
    belly_m.tube([np.array(p) - PRO_BACK * 0.75 for p in PRO[:4]], [0.5, 0.42, 0.4, 0.38], k=0)   # prosternum (front of the reared chest)
    belly_m.ell((0, 3.3, 0.0), (0.7, 0.42, 0.95), k=0)                                     # thorax underside
    belly_m.ell(H((0, -0.75, 0.5)), (0.3, 0.3, 0.15), k=0)                                  # mouth plate

    def tint(P, N):
        feet = 1 - 0.72 * smoothstep(3.2, 0.4, P[:, 1]) * (np.abs(P[:, 0]) > 1.6)        # legs darken toward the feet
        return (1 - 0.18 * smoothstep(0.2, -0.6, N[:, 1])) * feet
    K.skin(B, voxel=0.04, tris=16000, belly=belly_m, regions=[('plate', lambda P: wingm.dist(S.Points(P)) + 0.04)], tint=tint)

    # ---------------------------------------------------------------- skeleton
    K.bone('spine', (0, 3.85, -0.8), (0, 4.28, 0.8), None, 1.0)
    K.chain('abd', [ABD[0], ABD[1], ABD[2], ABD[3], ABD[5]], 'spine', (1.3, 0.7))
    K.bone('chest', (0, 4.28, 0.8), (0, 7.6, 2.25), 'spine', 0.85)
    K.bone('head', (0, 7.7, 2.5), H((0, 0.6, 0.1)), 'chest', 0.9)
    K.bone('jaw', H((0, -0.85, 0.35)), H((0, -1.3, 0.55)), 'head', 0.2)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = np.array([s, 1, 1])
        K.bone('coxa' + sd, m * SHO, m * COX, 'chest', 0.42)
        K.bone('femur' + sd, m * COX, m * KNEE, 'coxa' + sd, 0.55)
        K.bone('tibia' + sd, m * KNEE, m * (KNEE + blade_dir(s) * 2.0), 'femur' + sd, 0.3)
        for nm, (hip, kn, an, toe) in (('m', MID), ('h', HIND)):
            K.bone(nm + 'femur' + sd, m * hip, m * kn, 'spine', 0.4)
            K.bone(nm + 'tibia' + sd, m * kn, m * an, nm + 'femur' + sd, 0.26)
            K.bone(nm + 'tarsus' + sd, m * an, m * toe, nm + 'tibia' + sd, 0.15)
        K.mount('shoulder' + sd, m * (SHO + [0.15, 0.3, 0]), 'chest')
    K.mount('head', HC, 'head')
    K.mount('jaw', H((0, -1.05, 0.5)), 'jaw')
    K.mount('mouth', H((0, -1.0, 0.6)), 'head')
    K.mount('chest', np.array(PRO[1]) - PRO_BACK * 0.85, 'chest')
    K.mount('back', (0, 5.25, -1.2), 'spine')
    K.mount('tailTip', ABD[-1] + [0, 0.1, -0.4], 'abd4')
    K.mount('crown', H((0, 0.75, 0.0)), 'head')

    # ---------------------------------------------------------------- eyes, mouthparts, antennae, tarsi (base)
    for s, sd in ((-1, 'L'), (1, 'R')):
        c = H((s * 1.0, 0.1, 0.05))
        Rl = HR @ S.rot_xyz((0, 0, -s * 22))
        compound_eye(K, 'body', c, Rl, np.array((0.42, 0.54, 0.46)) * HS, HD((s * 0.55, -0.15, 0.85)), HD((s, 0.25, 0.3)), bone='head', seed=3 + s)
        # open mandibles: splayed, toothed, dark
        m0, m1, m2, m3 = H((s * 0.24, -0.92, 0.45)), H((s * 0.72, -1.18, 0.7)), H((s * 0.62, -1.55, 0.98)), H((s * 0.2, -1.7, 1.02))
        K.horn('body', [m0, m1, m2, m3], [0.15 * HS, 0.13 * HS, 0.08 * HS, 0], mat='dark', sides=6, bone='jaw', flat=0.6)
        for f, ln in ((0.25, 0.2), (0.55, 0.17), (0.8, 0.13)):
            p = m1 * (1 - f) + m2 * f
            inn = unit(m3 - m0) * 0.3 + np.array([-s, 0, 0])
            K.horn('body', [p - inn * 0.03, p + unit(inn) * ln], [0.04, 0], mat='horn', sides=4, bone='jaw')
        K.horn('body', [H((s * 0.4, -0.8, 0.38)), H((s * 0.62, -1.15, 0.45)), H((s * 0.58, -1.45, 0.4))], [0.06, 0.05, 0.03], mat='body', sides=5, bone='jaw')   # palps
        # hooded brow: a dark chitin lid hugging the upper-front rim of each eye, dipping low at the inner end
        er = np.array((0.42, 0.54, 0.46)) * HS
        bp = []
        for d in ((-0.75, 0.05, 0.66), (-0.45, 0.5, 0.74), (0.0, 0.78, 0.62), (0.45, 0.88, 0.2), (0.7, 0.7, -0.15)):
            d = unit((d[0] * s, d[1], d[2]))
            q = d / np.sqrt(((d / er) ** 2).sum())
            q[1] *= 1 + 0.3 * max(d[1], 0) ** 3
            bp.append(c + Rl @ (q * 1.05))
        K.horn('body', bp, np.array([0.03, 0.08, 0.1, 0.07, 0.0]) * HS, mat='dark', sides=6, flat=0.45, bone='head')
        # antennae: whip back over the shield
        K.horn('body', [H((s * 0.18, 0.38, 0.4)), H((s * 0.4, 0.95, 0.45)), H((s * 0.85, 1.45, 0.0)), H((s * 1.3, 1.65, -0.7)), H((s * 1.65, 1.55, -1.4))],
               [0.06, 0.05, 0.04, 0.028, 0.012], mat='dark', sides=5, per=4, bone='head')
        for nm, (hip, kn, an, toe) in (('m', MID), ('h', HIND)):
            an, toe, kk = np.array(an) * [s, 1, 1], np.array(toe) * [s, 1, 1], np.array(kn) * [s, 1, 1]
            bn = nm + 'tarsus' + sd
            for f0, f1 in ((0.0, 0.42), (0.42, 0.7), (0.7, 1.0)):   # beaded tarsal segments
                p0, p1 = an + (toe - an) * f0, an + (toe - an) * f1
                K.horn('body', [p0, p0 * 0.5 + p1 * 0.5, p1], [0.15 - f0 * 0.05, 0.13 - f0 * 0.05, 0.1 - f0 * 0.04], mat='body', sides=6, per=1, bone=bn)
            d = unit(toe - an) * [1, 0, 1]
            for sp in (-0.3, 0.3):
                K.claw('body', toe, d + [sp * s, -0.35, 0], 0.34, 0.06, curl=(0, -1, 0), bone=bn, mat='dark')
            for f in (0.3, 0.55, 0.8):   # tibial spurs
                p = kk + (an - kk) * f
                K.spike('body', p + [0, 0, -0.12 if nm == 'm' else 0.12], unit(an - kk) * 0.6 + [s * 0.3, 0, -0.8 if nm == 'm' else 0.8], 0.3, r=0.05, mat='dark', bone=nm + 'tibia' + sd)
            hp = np.array(hip) * [s, 1, 1]
            for f in (0.4, 0.65):        # femur spines (upper ridge)
                p, n = K.snap(hp + (kk - hp) * f + [0, 0.6, 0])
                K.spike('body', p, n + unit(kk - hp) * 0.7, 0.26, r=0.055, mat='dark', bone=nm + 'femur' + sd)

    # ---------------------------------------------------------------- scythes per Might tier (exclusive sets)
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        scythe(K, part, t, sc)

    # ---------------------------------------------------------------- Armor: pronotum shield, flank + head plates, spikes
    K.plate('arm_t1', pm + PRO_BACK * 0.3 - PRO_UP * 0.3, (1.15, 1.7, 0.6), bone='chest', lames=4, lame_axis=PRO_UP)
    K.plate('arm_t1', (0, 4.9, -0.25), (0.95, 0.55, 0.85), bone='spine', lames=2, lame_axis=(0, 0, 1))
    for s, sd in ((-1, 'L'), (1, 'R')):
        K.plate('arm_t2', (s * 1.6, 3.35, -3.3), (0.55, 0.6, 2.0), bone='abd2', lames=4, lame_axis=(0, 0, 1), tris=280)
        K.plate('arm_t2', H((s * 0.55, 0.45, 0.05)), np.array((0.42, 0.22, 0.4)) * HS, bone='head')
        K.plate('arm_t2', (s * 1.2, 6.75, 2.75), (0.42, 0.55, 0.5), bone='coxa' + sd)
        for f in (0.2, 0.45, 0.7, 0.92):   # spikes along the shield flange
            p, n = K.snap(pm + PRO_UP * (f - 0.55) * 3.0 + [s * 1.1, 0, 0] + PRO_BACK * 0.1)
            K.spike('arm_t3', p, n + [s * 0.5, 0.3, -0.5], 0.5 + 0.15 * (f > 0.6), r=0.11, bone='chest')
        p, n = K.snap(KNEE * [s, 1, 1] + [s * 0.2, 0.3, -0.1])
        K.spike('arm_t3', p, (s * 0.3, 1, -0.2), 0.75, r=0.13, bone='femur' + sd)
        for nm, (hip, kn, an, toe) in (('m', MID), ('h', HIND)):
            k3 = np.array(kn) * [s, 1, 1]
            K.spike('arm_t3', k3 + [0, 0.18, 0], (s * 0.35, 1, 0.1), 0.6, r=0.11, bone=nm + 'femur' + sd)
        for z in (-2.3, -3.5, -4.7, -5.7):
            p, n = K.snap((s * 1.8, 3.45, z))
            K.spike('arm_t3', p, n + [0, 0.25, -0.5], 0.5, r=0.1, bone='abd2' if z > -3 else ('abd3' if z > -5 else 'abd4'))
    p, n = K.snap(H((0, 0.55, 0.0)))
    K.spike('arm_t3', p, HD((0, 1, -0.4)), 0.65, r=0.12, bone='head')

    # ---------------------------------------------------------------- Energy: spiracles -> pronotum glands + knee crystals -> vein lines
    Cn, Rn = S.catmull(ABD, 6), S.catmull(ABD_R, 6)[:, 0]
    for s, sd in ((-1, 'L'), (1, 'R')):
        for j in range(4, len(Cn) - 6, 4):
            p, n = K.snap(Cn[j] + [s * Rn[j] * 1.2, -0.1, 0])
            K.ball('enr_t1', p - n * 0.02, (0.13, 0.1, 0.13), 'accent', seg=8, rings=6, bone='abd2' if Cn[j][2] > -3 else 'abd3')
        for f in np.linspace(0.1, 0.85, 5):
            p, n = K.snap(np.array(PRO[0]) * (1 - f) + np.array(PRO[3]) * f - PRO_BACK * 0.5 + [s * 0.45, 0, 0])
            K.ball('enr_t2', p - n * 0.03, 0.09, 'accent', seg=8, rings=6, bone='chest')
        K.crystal('enr_t2', KNEE * [s, 1, 1] + [s * 0.1, 0.18, -0.08], (s * 0.3, 1, -0.3), 0.5, cluster=False, bone='femur' + sd)
        for off in (-0.18, 0.18):
            P, Nn = K.snap(np.array([(COX + (KNEE - COX) * f) * [s, 1, 1] + [s * 0.7, 0, off] for f in np.linspace(0.12, 0.88, 8)]))
            K.horn('enr_t3', P + Nn * 0.01, [0.05] * 7 + [0.0], mat='accent', sides=5, bone='femur' + sd)
        P, Nn = K.snap(np.array([Cn[j] + [s * Rn[j] * 1.2, -0.1, 0] for j in range(3, len(Cn) - 5)]))
        K.horn('enr_t3', P, np.r_[[0.0], [0.05] * (len(P) - 2), [0.0]], mat='accent', sides=5, bone='abd2')
    for f in (0.2, 0.45, 0.7):   # glowing tubercles up the keel
        p, n = K.snap(pm + PRO_UP * (f - 0.5) * 2.6 + PRO_BACK * 0.8)
        K.crystal('enr_t3', p - n * 0.04, n + PRO_UP * 0.3, 0.42, cluster=False, bone='chest', seed=int(f * 10))

    # ---------------------------------------------------------------- Speed: hind wings fan out (deimatic display)
    for s, sd in ((-1, 'L'), (1, 'R')):
        base = np.array([s * 0.5, 5.15, -0.7])
        fan_wing(K, 'spd_t1', base, (s * 0.35, 0.2, -1), (s * 0.85, 0.4, -0.45), 2.6, veins=6, bone='spine')
        fan_wing(K, 'spd_t2', base + [0, 0.08, 0.05], (s * 0.5, 0.6, -0.9), (s * 0.95, 0.8, -0.1), 3.5, veins=7, bone='spine')
        fan_wing(K, 'spd_t3', base + [0, 0.16, 0.1], (s * 0.25, 1, -0.55), (s * 1, 0.6, 0.0), 4.3, veins=8, bone='spine', spot=(0.68, 0.45, 0.5))

    # ---------------------------------------------------------------- Focus: ocelli, head crest, extra eyes
    for p in (H((0, 0.5, 0.3)), H((-0.2, 0.42, 0.38)), H((0.2, 0.42, 0.38))):
        K.ball('fcs_t1', p, 0.085, 'eye', seg=8, rings=6, bone='head')
    K.leaf('fcs_t2', H((0, 0.5, -0.1)), HD((0, 1, -0.35)), HD((0, 0.25, -1)), 1.3, 0.95, th=0.08, lobes=3, mat='accent', bone='head')
    for s in (-1, 1):
        K.ball('fcs_t3', H((s * 0.62, -0.42, 0.4)), 0.11, 'eye', seg=8, rings=6, bone='head')
        K.ball('fcs_t3', H((s * 0.42, -0.15, 0.46)), 0.085, 'eye', seg=8, rings=6, bone='head')


def blade_dir(s):
    return unit((0.32 * s, -0.6, 0.74))


def scythe(K, part, t, sc):
    """Raptorial strike set for one Might tier: serrated sickle tibia + femur spine rows (+ spurs at t2/t3)."""
    L = 2.9 * sc ** 0.8
    wd = 0.5 * (0.85 + 0.15 * sc)
    for s, sd in ((-1, 'L'), (1, 'R')):
        m = np.array([s, 1, 1])
        kn, cx = KNEE * m, COX * m
        a = blade_dir(s)
        fdb = unit(cx - kn)
        b = unit(fdb - a * (fdb @ a))                 # toward the femur (inner, cutting side)
        nrm = unit(np.cross(a, b))
        us = np.array([0.0, 0.25, 0.5, 0.75, 0.92, 1.0])
        hk = np.array([0.0, -0.02, -0.03, 0.04, 0.2, 0.42])
        pts = [kn + a * L * u * (1 - 0.12 * u ** 2) + b * L * h for u, h in zip(us, hk)]
        widths = [wd * 0.7, wd, wd * 0.95, wd * 0.75, wd * 0.4, 0.0]
        thick = [0.17, 0.15, 0.12, 0.09, 0.06, 0.03]
        V, T, Cb, tan = blade_mesh(pts, widths, thick, nrm, 1.0, back=0.75, per=5)
        n = len(Cb)
        E1 = unit(np.cross(tan[n // 2], unit(nrm - tan[n // 2] * (nrm @ tan[n // 2]))))
        sg = 1.0 if E1 @ b > 0 else -1.0
        if sg < 0:
            V, T, Cb, tan = blade_mesh(pts, widths, thick, nrm, -1.0, back=0.75, per=5)
        u = np.linspace(0, 1, len(V))
        K.piece(part, V, T, 'body', bone='tibia' + sd, col=np.clip(1.0 - 0.72 * u ** 1.4, 0, 1))
        E = np.array([unit(np.cross(tan[i], unit(nrm - tan[i] * (nrm @ tan[i])))) * sg for i in range(n)])
        wv = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(widths)), widths)
        edge = Cb + E * wv[:, None]
        K.horn(part, list(edge[1:-1:2]) + [Cb[-1]], np.linspace(0.06, 0.0, len(edge[1:-1:2]) + 1), mat='horn', sides=4, per=1, bone='tibia' + sd)
        # serrated cutting edge: alternating long/short teeth raking toward the tip
        for i in range(2, n - 3):
            e = edge[i]
            big = i % 3 == 0
            if not big and i % 3 == 2:
                continue
            ln = (0.32 if big else 0.17) * (0.85 + 0.15 * sc)
            K.horn(part, [e - E[i] * 0.05, e + E[i] * ln * 0.6 + tan[i] * ln * 0.25, e + E[i] * ln + tan[i] * ln * 0.65],
                   [0.06 if big else 0.04, 0.035, 0], sides=4, per=1, bone='tibia' + sd)
        # terminal hook claw
        K.claw(part, Cb[-3], unit(tan[-3] + E[-3] * 0.6), 0.5 * (0.8 + 0.2 * sc), 0.08, curl=E[-3], bone='tibia' + sd, mat='horn')
        # femur: two rows of big spines on the face toward the blade
        fd = unit(kn - cx)
        inner = unit(a - fd * (a @ fd))
        fn = unit(np.cross(fd, inner))
        for row, off in enumerate((-0.17, 0.17)):
            for i, f in enumerate(np.linspace(0.18, 0.92, 7)):
                p = cx + (kn - cx) * f
                q, nq = K.snap(p + inner * 0.8 + fn * off)
                big = (i + row) % 2 == 0
                ln = (0.48 if big else 0.28) * (0.75 + 0.25 * sc) * (0.75 + 0.35 * np.sin(np.pi * f))
                d = unit(inner * 1.0 + fd * 0.35 + fn * off)
                K.horn(part, [q - d * 0.05, q + d * ln * 0.55 + fd * ln * 0.08, q + d * ln + fd * ln * 0.25],
                       [0.08 if big else 0.055, 0.045, 0], sides=4, per=1, bone='femur' + sd, mat='dark' if big else 'horn')
        if t >= 2:   # tibial spur at the knee, pointing forward along the blade
            K.claw(part, kn + inner * 0.2, unit(a + b * 0.5), 0.45 + 0.25 * sc, 0.1, curl=b, bone='tibia' + sd)
        if t >= 3:   # crest of spines up the femur back
            for f in (0.3, 0.5, 0.7, 0.88):
                p, n2 = K.snap(cx + (kn - cx) * f - inner * 0.8)
                K.spike(part, p, unit(n2 - fd * 0.2), 0.65, r=0.11, bone='femur' + sd)
        if t == 0:
            K.mount('hand' + sd, Cb[-1], 'tibia' + sd)
