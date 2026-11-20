"""Superhero Team: five distinct heroes in a V formation (Wild-only, human scale).

One skinned SDF mesh for all five, modelled ~9 units tall per hero and exported at SCALE = 0.1.
Leader (shield) front-centre, bruiser + flyer flanking, tech hero + speedster at the back corners.
"""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE
from ..noise import smoothstep
from .superhero import A, Fig, band, beyond, capsule, frame, hair_disp, nrm, oball, pts3, star_pts, torus

DETAIL = 'cloth'
UV_SCALE = 0.9
SCALE = 0.1
SCALE_REF = True
T3_SCALE = 1.12
CAM = dict(az=-18, el=13)


def stand():
    J = dict(pelvis=(0, 4.75, 0), waist=(0, 5.6, 0.03), chest=(0, 6.6, 0.1), neck=(0, 7.55, 0.03), head=(0, 8.45, 0.12),
             shR=(1.36, 7.3, 0), elR=(1.8, 5.95, -0.1), wrR=(2.0, 4.65, 0.15),
             hipR=(0.55, 4.5, 0), knR=(0.75, 2.5, 0.15), anR=(0.85, 0.45, -0.05), toeR=(0.9, 0.12, 0.75))
    for k in list(J):
        if k.endswith('R'):
            v = J[k]
            J[k[:-1] + 'L'] = (-v[0], v[1], v[2])
    return J


def place(J, pos, yaw=0.0, s=1.0, lift=0.0):
    a = np.radians(yaw)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return {k: np.asarray(pos, float) + R @ (np.asarray(v, float) * s) + A(0, lift, 0) for k, v in J.items()}, R


def rot_y(v, R):
    return R @ np.asarray(v, float)


def build(K):
    col = K.colors
    K.mat('skin', color='#d6a07c', rough=0.6, vcol=True, detail=False)
    K.mat('skin2', color='#8a5a3c', rough=0.6, vcol=True, detail=False)
    K.mat('hair', color='#2a1d16', rough=0.45, vcol=True, detail=False)
    K.mat('hair2', color='#e8c66a', rough=0.45, vcol=True, detail=False)
    K.mat('brow', color='#1a130f', rough=0.6)
    K.mat('sclera', color='#f2efe8', rough=0.35)
    K.mat('iris', color='#3b2a1e', rough=0.3)
    K.mat('mouth', color='#7a3a34', rough=0.5)
    K.mat('glint', color='#ffffff', emit='#ffffff', emit_strength=1.5)
    K.mat('body', rough=0.5, detail_strength=0.35)
    K.mat('belly', rough=0.5, detail_strength=0.3)
    K.mat('plate', color='#b9c0cc', rough=0.3, metal=0.7, vcol=True)
    K.mat('gun', color='#3a3f4c', rough=0.35, metal=0.7, vcol=True)

    B = K.sdf()
    white, skin, skin2, red = [], [], [], []
    heroes = {}

    # ================================================================ 1. LEADER (shield, front centre)
    J = stand()
    J.update(elR=(1.95, 6.35, -0.45), wrR=(1.85, 5.6, 0.7), elL=(-1.85, 6.35, 0.65), wrL=(-1.15, 6.85, 1.55),
             hipR=(0.58, 4.5, 0), knR=(1.05, 2.5, 0.35), anR=(1.3, 0.45, 0.1), toeR=(1.45, 0.12, 0.85),
             hipL=(-0.58, 4.5, 0), knL=(-0.95, 2.45, -0.15), anL=(-1.2, 0.45, -0.55), toeL=(-1.4, 0.12, 0.15))
    J1, R1 = place(J, (0, 0, 3.4), 0, 1.0)
    f1 = Fig(J1, dict(s=1.0, chest=1.05, waist=1.0, arm=1.05, musc=1.0, headR=1.08, pitch=4, yaw=-10, jaw=1.15,
                      hands=dict(L='fist', R='fist'), up=dict(L=A(0, 1, 0), R=A(0, 1, 0))))
    f1.sculpt(B)
    heroes[1] = f1
    up1 = nrm(J1['chest'] - J1['pelvis'])
    # white chevron chest panel + gloves + boots
    cv_c = J1['chest'] + A(0, 0.05, 0)

    def chev(P, f=f1, c=cv_c, w=0.55):
        x = P[:, 0] - c[0]
        y = P[:, 1] - c[1]
        return np.maximum(np.maximum(-(P[:, 2] - c[2] - 0.15), np.abs(y - 0.55 * np.abs(x) + 0.1) - w * 0.5), capsule(P, f.J['waist'], f.J['neck'], 1.3))
    white += [chev, f1.m_gloves(0.5), f1.m_boots(0.35)]
    skin.append(f1.m_skin(0.05))
    red.append(f1.m_belt(0.2, 0.1)[0])

    # ================================================================ 2. BRUISER (left wing)
    J = dict(pelvis=(0, 4.2, 0), waist=(0, 5.05, 0.18), chest=(0, 6.2, 0.42), neck=(0, 7.12, 0.6), head=(0, 7.82, 0.92),
             shR=(1.95, 6.85, 0.32), elR=(2.75, 5.35, 0.5), wrR=(2.6, 3.95, 1.15),
             hipR=(0.75, 4.0, 0), knR=(1.2, 2.25, 0.55), anR=(1.35, 0.5, 0.05), toeR=(1.45, 0.15, 0.85))
    for k in list(J):
        if k.endswith('R'):
            v = J[k]
            J[k[:-1] + 'L'] = (-v[0], v[1], v[2])
    J['wrL'] = (-2.6, 4.2, 1.5)
    J2, R2 = place(J, (-4.3, 0, 0.9), 14, 1.1)
    f2 = Fig(J2, dict(s=1.1, chest=1.38, waist=1.12, hip=1.12, arm=1.4, leg=1.3, neck=1.5, headR=0.98, musc=1.4, jaw=1.4, pitch=-4, yaw=-8,
                      handS=1.75, hands=dict(L='fist', R='fist'), up=dict(L=A(0, 0, 1), R=A(0, 0, 1))))
    f2.sculpt(B)
    heroes[2] = f2
    B.ell(J2['chest'] + rot_y(A(0, 0.55, -0.45), R2) * 1.1, A(1.25, 0.65, 0.75) * 1.1, k=0.45)   # trapezius hump
    white += [f2.m_gloves(0.9)]
    skin2.append(f2.m_skin(0.05))
    red.append(f2.m_belt(0.22, 0.13)[0])

    # ================================================================ 3. FLYER (right wing, airborne)
    J = dict(pelvis=(0, 4.75, -0.1), waist=(0, 5.6, 0.05), chest=(0, 6.48, 0.25), neck=(0, 7.25, 0.38), head=(0, 8.15, 0.55),
             shR=(0.85, 6.9, 0.3), elR=(1.05, 7.95, 0.95), wrR=(1.2, 8.95, 1.65),
             shL=(-0.85, 6.9, 0.3), elL=(-1.15, 6.0, -0.55), wrL=(-1.3, 5.0, -1.25),
             hipR=(0.38, 4.55, -0.1), knR=(0.42, 2.75, -0.7), anR=(0.4, 1.0, -1.4), toeR=(0.4, 0.55, -1.9),
             hipL=(-0.38, 4.55, -0.1), knL=(-0.48, 3.05, 0.1), anL=(-0.5, 1.75, -1.0), toeL=(-0.5, 1.4, -1.55))
    J3, R3 = place(J, (3.7, 0, 0.9), -24, 0.95, lift=2.4)
    f3 = Fig(J3, dict(s=0.95, fem=True, chest=0.72, waist=0.78, hip=0.85, arm=0.85, leg=0.9, neck=0.7, headR=1.14, bust=0.7, pitch=12, yaw=5,
                      hands=dict(L='flat', R='flat'), up=dict(L=A(0, 0, -1), R=A(0, 0, 1))))
    f3.sculpt(B)
    heroes[3] = f3
    white += [f3.m_boots(0.3), f3.m_gloves(0.45)]
    skin.append(f3.m_skin(0.05))
    # white side stripes down the torso
    def stripes(P, f=f3, R=R3):
        sx = R @ A(1, 0, 0)
        lx = (P - f.J['waist']) @ sx
        return np.maximum(np.abs(np.abs(lx) - 0.55) - 0.09, capsule(P, f.J['pelvis'], f.J['neck'], 0.9))
    white.append(stripes)
    red.append(f3.m_belt(0.25, 0.08)[0])

    # ================================================================ 4. TECH hero (left back, arm cannon)
    J = stand()
    J.update(elR=(1.8, 6.3, 0.5), wrR=(2.65, 6.55, 1.5), elL=(-1.85, 5.95, -0.05), wrL=(-2.05, 4.75, 0.35),
             hipR=(0.6, 4.5, 0), knR=(0.95, 2.5, 0.3), anR=(1.1, 0.45, 0.0), toeR=(1.2, 0.12, 0.8),
             hipL=(-0.6, 4.5, 0), knL=(-0.85, 2.5, 0.05), anL=(-1.0, 0.45, -0.25), toeL=(-1.1, 0.12, 0.5))
    J4, R4 = place(J, (8.4, 0, -1.7), -18, 1.0)
    f4 = Fig(J4, dict(s=1.0, chest=1.15, waist=1.1, hip=1.05, arm=1.15, leg=1.08, musc=0.6, headR=1.02, pitch=0, yaw=-12,
                      hands=dict(L='fist', R='none'), up=dict(L=None, R=A(0, 1, 0))))
    f4.sculpt(B)
    heroes[4] = f4
    H4 = f4.H
    B.ell(H4((0, 0.02, 0.0)), A(0.56, 0.62, 0.62) * f4.hs, rot=f4.Rh, k=0.08)              # helmet dome
    B.tube([H4((-0.5, -0.25, 0.05)), H4((0, -0.62, 0.42)), H4((0.5, -0.25, 0.05))], [0.16 * f4.hs], k=0.1)   # jaw guard
    B.ell(J4['chest'] + rot_y(A(0, 0.1, -0.95), R4), A(0.9, 1.0, 0.45), k=0.25)          # backpack
    helm = S.Ell(H4((0, 0.0, 0.0)), A(0.66, 0.75, 0.72) * f4.hs)
    cuir = S.Ell(J4['chest'] + rot_y(A(0, 0.2, 0.45), R4), A(1.15, 0.8, 0.7))
    pads = [S.Ell(J4['sh' + sd] + A(0, 0.15, 0), A(0.6, 0.55, 0.6)) for sd in 'LR']
    white += [lambda P: helm.dist(S.Points(P)), lambda P: cuir.dist(S.Points(P)), lambda P: np.minimum(pads[0].dist(S.Points(P)), pads[1].dist(S.Points(P))),
              f4.m_boots(0.45), f4.m_gloves(0.4, sides='L')]

    # ================================================================ 5. SPEEDSTER (right back, sprint)
    J = dict(pelvis=(0, 4.3, 0), waist=(0, 4.95, 0.45), chest=(0, 5.62, 1.1), neck=(0, 6.22, 1.68), head=(0, 6.85, 2.25),
             shR=(1.02, 6.08, 1.45), elR=(1.3, 5.3, 2.25), wrR=(1.25, 5.95, 3.0),
             shL=(-1.02, 6.08, 1.45), elL=(-1.35, 5.4, 0.6), wrL=(-1.45, 5.05, -0.3),
             hipR=(0.48, 4.1, 0.12), knR=(0.56, 3.4, 1.9), anR=(0.6, 1.75, 1.35), toeR=(0.6, 1.35, 2.05),
             hipL=(-0.48, 4.1, -0.05), knL=(-0.58, 2.35, -0.85), anL=(-0.62, 0.8, -1.8), toeL=(-0.62, 0.12, -1.4))
    J5, R5 = place(J, (-9.6, 0, -1.1), 50, 0.9)
    f5 = Fig(J5, dict(s=0.9, chest=0.92, waist=0.9, hip=0.9, arm=0.85, leg=0.95, musc=0.7, headR=1.1, pitch=30, yaw=4,
                      hands=dict(L='fist', R='fist'), up=dict(L=None, R=A(0, 0, 1))))
    f5.sculpt(B)
    heroes[5] = f5
    H5 = f5.H
    white += [f5.m_boots(0.3), f5.m_gloves(0.4)]
    skin.append(f5.m_skin(0.05))

    # ---------------------------------------------------------------- mesh
    def U(lst):
        def fn(P):
            out = np.full(len(P), 9.0)
            for f in lst:
                out = np.minimum(out, f(P))
            return out
        return fn
    K.skin(B, voxel=0.045, tris=8800, regions=[('belly', U(white)), ('skin', U(skin)), ('skin2', U(skin2)), ('accent', U(red))])

    # ---------------------------------------------------------------- faces + hair
    f1.pieces(K, head_tris=1050, hand_tris=260)
    f2.pieces(K, skin_mat='skin2', bones=dict(head='hero2_head', handL='hero2', handR='hero2'), head_tris=850, hand_tris=260)
    f3.pieces(K, bones=dict(head='hero3_head', handL='hero3', handR='hero3'), head_tris=850, hand_tris=260)
    f4.pieces(K, head=False, bones=dict(head='hero4_head', handL='hero4', handR='hero4'), hand_tris=260)
    f5.pieces(K, bones=dict(head='hero5_head', handL='hero5', handR='hero5'), head_tris=850, hand_tris=260)
    f1.face_parts(K)
    f2.face_parts(K, bone='hero2_head')
    f3.face_parts(K, big=0.9, bone='hero3_head')
    f5.face_parts(K, bone='hero5_head')
    # leader: swept hair + red domino mask
    H1, h1 = f1.H, f1.hs
    hb = K.sdf()
    hb.ell(H1((0, 0.16, -0.08)), A(0.5, 0.47, 0.56), rot=f1.Rh, k=0.1)
    hb.tube([H1((-0.32, 0.45, 0.25)), H1((0, 0.6, 0.35)), H1((0.3, 0.52, 0.3))], [0.15, 0.2, 0.14], k=0.12)
    hb.box(H1((0, -0.35, 0.55)), A(0.7, 0.55, 0.4), rot=f1.Rh @ S.rot_xyz((-28, 0, 0)), cut=True, k=0.08, rad=0.1)
    hb.box(H1((0, -0.65, 0.0)), A(0.7, 0.45, 0.7), rot=f1.Rh, cut=True, k=0.1, rad=0.1)
    hb.disp(hair_disp(1.0, seed=3, freq=28, amp=0.022), band=0.08)
    K.sdf_piece('body', hb, voxel=0.04, tris=500, mat='hair', bone='head', ao=0.5)
    mk = K.sdf()
    mk.ell(H1((0, 0.0, 0.05)), A(0.5, 0.13, 0.55), rot=f1.Rh, k=0)
    mk.ell(H1((0, 0.0, 0.02)), A(0.47, 0.25, 0.53), rot=f1.Rh, k=0.02, cut=True)
    mk.box(H1((0, 0.0, -0.45)), A(0.7, 0.3, 0.4), rot=f1.Rh, k=0.02, cut=True, rad=0)
    K.sdf_piece('body', mk, voxel=0.03, tris=240, mat='body', bone='head')
    # leader chevron emblem + shield
    cp, cn = K.snap(J1['chest'] + A(0, 0.35, 1.2))
    for i, o in enumerate((0.0, -0.22)):
        pts = [cp + A(-0.42, 0.18 + o, 0), cp + A(0, -0.12 + o, 0.08), cp + A(0.42, 0.18 + o, 0)]
        P2, N2 = K.snap(np.array(pts))
        K.horn('body', list(P2 + N2 * 0.05), [0.07, 0.08, 0.07], mat='accent', sides=4, flat=0.4, up=cn, bone='chest')
    sc_c = J1['elL'] + (J1['wrL'] - J1['elL']) * 0.55 + A(-0.15, 0.0, 0.42)
    sn = nrm(A(-0.4, 0.08, 1))
    shield_c, shield_n = sc_c, sn

    # bruiser: flat-top
    H2 = f2.H
    hb = K.sdf()
    hb.box(H2((0, 0.36, -0.05)), A(0.42, 0.22, 0.48) * f2.hs, rot=f2.Rh, k=0.05, rad=0.1)
    hb.disp(hair_disp(1.0, seed=5, freq=40, amp=0.015), band=0.06)
    K.sdf_piece('body', hb, voxel=0.04, tris=250, mat='hair', bone='hero2_head')
    # flyer: blonde ponytail streaming back
    H3, h3 = f3.H, f3.hs
    hb = K.sdf()
    hb.ell(H3((0, 0.14, -0.08)), A(0.5, 0.49, 0.56), rot=f3.Rh, k=0.1)
    hb.box(H3((0, -0.4, 0.6)), A(0.42, 0.5, 0.45), rot=f3.Rh @ S.rot_xyz((-18, 0, 0)), cut=True, k=0.08, rad=0.12)
    hb.box(H3((0, -0.85, 0.0)), A(0.8, 0.3, 0.8), rot=f3.Rh, cut=True, k=0.08, rad=0.1)
    for x0, x1, ln in ((-0.3, -0.38, 0.45), (-0.1, -0.12, 0.5), (0.12, 0.15, 0.48), (0.32, 0.4, 0.4)):
        hb.tube([H3((x0 * 0.6, 0.5, 0.28)), H3((x0, 0.3, 0.5)), H3((x1, 0.28 - ln * 0.5, 0.52))], [0.12, 0.1, 0.03], k=0.05)
    hb.tube([H3((0, 0.3, -0.5)), H3((0.05, 0.1, -1.1)), H3((-0.15, -0.3, -1.8)), H3((0.1, -0.5, -2.6))], np.array([0.22, 0.25, 0.2, 0.04]), k=0.12)
    hb.disp(hair_disp(1.0, seed=8, freq=22, amp=0.03), band=0.1)
    K.sdf_piece('body', hb, voxel=0.04, tris=700, mat='hair2', bone='hero3_head', ao=0.4)
    # flyer: red visor glasses
    for s in (-1, 1):
        c, n = K.snap(H3((s * 0.18, -0.03, 0.55)))
    vg = K.sdf()
    vg.ell(H3((0, -0.02, 0.05)), A(0.47, 0.1, 0.56), rot=f3.Rh, k=0)
    vg.ell(H3((0, -0.02, 0.02)), A(0.43, 0.2, 0.52), rot=f3.Rh, k=0.02, cut=True)
    vg.box(H3((0, 0.0, -0.4)), A(0.7, 0.3, 0.4), rot=f3.Rh, k=0.02, cut=True, rad=0)
    K.sdf_piece('body', vg, voxel=0.03, tris=250, mat='accent', bone='hero3_head')
    # tech: glowing visor slit + cannon + backpack vents
    vs = K.sdf()
    vs.ell(H4((0, 0.02, 0.15)), A(0.6, 0.11, 0.62) * f4.hs, rot=f4.Rh, k=0)
    vs.ell(H4((0, 0.02, 0.1)), A(0.5, 0.2, 0.55) * f4.hs, rot=f4.Rh, k=0.02, cut=True)
    vs.box(H4((0, 0.0, -0.35)), A(0.8, 0.3, 0.45), rot=f4.Rh, k=0.02, cut=True, rad=0)
    K.sdf_piece('body', vs, voxel=0.03, tris=250, mat='accent', bone='hero4_head')
    d4 = nrm(J4['wrR'] - J4['elR'])
    can0, can1 = J4['elR'] + d4 * 0.1, J4['wrR'] + d4 * 0.75
    # bruiser / flyer / tech chest emblems
    c2, n2 = K.snap(J2['chest'] + rot_y(A(0, 0.3, 2.0), R2))
    K.plate('body', c2, None, off=0.0, th=0.035, mat='belly', region=S.Ell(c2, A(0.55, 0.55, 0.55)), voxel=0.035, tris=200)
    V, T = star_pts(c2 + n2 * 0.08, n2, A(0, 1, 0), 0.42, 0.17, depth=0.05)
    K.piece('body', V, T, 'accent', smooth=False, bone='hero2')
    c3, n3 = K.snap(J3['chest'] + rot_y(A(0, 0.15, 1.5), R3))
    V, T = star_pts(c3 + n3 * 0.06, n3, A(0, 1, 0), 0.28, 0.11, depth=0.04)
    K.piece('body', V, T, 'accent', smooth=False, bone='hero3')
    c4, n4 = K.snap(J4['chest'] + rot_y(A(0, 0.2, 2.0), R4))
    V, T = torus(c4 + n4 * 0.04, n4, 0.26, 0.06, nu=16, nv=4)
    K.piece('body', V, T, 'gun', bone='hero4')
    K.ball('body', c4 + n4 * 0.02, A(0.2, 0.2, 0.2), mat='accent', bone='hero4', seg=10, rings=6)
    # speedster: lightning emblem + helmet fins
    cw = K.sdf()   # speedster cowl: covers the skull, open from the brows down
    cw.ell(H5((0, 0.1, -0.06)), A(0.47, 0.5, 0.56) * f5.hs, rot=f5.Rh, k=0)
    cw.tube([H5((0, -0.3, -0.18)), H5((0, -0.8, -0.2))], np.array([0.3, 0.32]) * f5.hs, k=0.12)
    cw.box(H5((0, -0.38, 0.56)), A(0.355, 0.46, 0.36) * f5.hs, rot=f5.Rh, cut=True, k=0.05, rad=0.06)
    cw.box(H5((0, -1.15, 0.0)), A(0.8, 0.3, 0.8) * f5.hs, rot=f5.Rh, cut=True, k=0.05, rad=0.0)
    K.sdf_piece('body', cw, voxel=0.022, tris=700, mat='body', bone='hero5_head', ao=0.3)
    cp5, cn5 = K.snap(J5['chest'] + rot_y(A(0, 0.2, 1.2), R5))
    zz = [cp5 + A(0.12, 0.3, 0), cp5 + A(-0.08, 0.02, 0), cp5 + A(0.1, 0.0, 0), cp5 + A(-0.12, -0.3, 0)]
    P5, N5 = K.snap(np.array(zz))
    K.horn('body', list(P5 + N5 * 0.06), [0.07, 0.07, 0.07, 0.0], mat='accent', sides=4, flat=0.5, up=cn5, bone='hero5', per=1)
    for s in (-1, 1):
        K.fin('body', H5((s * 0.52, 0.15, 0.05)), f5.Rh @ A(s, 0.35, -0.2), f5.Rh @ A(0, 0.15, -1), 0.45, 0.55, mat='belly', bone='hero5_head', sweep_back=1.0)

    # ---------------------------------------------------------------- skeleton
    K.bone('hero1', J1['pelvis'] - A(0, 0.4, 0), J1['pelvis'], None, 0.6)
    f1.rig(K, parent='hero1')
    for i, f in ((2, f2), (3, f3), (4, f4), (5, f5)):
        J = f.J
        K.bone(f'hero{i}', J['pelvis'], J['neck'], None, 2.4 * f.q)
        K.bone(f'hero{i}_head', f.H((0, -0.45, 0)), f.H((0, 0.55, 0)), f'hero{i}', 0.7 * f.hs)
    f1.mounts(K, back=J1['chest'] + A(0, 0, -1.0), tail=J1['pelvis'] + A(0, -0.5, -1.2))

    # ---------------------------------------------------------------- base kit: shield, cannon, backpack thrusters
    def shield(part, scale, rim_mat='accent', boss=True):
        g = K.sdf()
        Rs = frame(shield_n, A(0, 1, 0))
        g.ell(shield_c, A(1.2 * scale, 0.14, 1.2 * scale), rot=Rs, k=0)
        g.ell(shield_c - shield_n * 0.25, A(1.12 * scale, 0.2, 1.12 * scale), rot=Rs, k=0.05, cut=True)
        K.sdf_piece(part, g, voxel=0.04, tris=300, mat='belly', bone='forearmL', ao=0.3)
        for rr in (1.12, 0.62):
            V, T = torus(shield_c + shield_n * (0.14 if rr < 1 else 0.06), shield_n, rr * scale, 0.07 * scale, nu=24, nv=4)
            K.piece(part, V, T, rim_mat, bone='forearmL')
        V, T = star_pts(shield_c + shield_n * 0.2, shield_n, A(0, 1, 0), 0.45 * scale, 0.18 * scale, depth=0.09)
        K.piece(part, V, T, 'accent', smooth=False, bone='forearmL')
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        shield(part, 0.85 + 0.15 * sc, rim_mat='accent' if t < 2 else 'plate')
        # bruiser gauntlets grow
        for sd in 'LR':
            hc, d = f2.hand_center(sd), f2.hand_dir(sd)
            gs = 0.8 + 0.25 * sc
            K.horn(part, [J2['wr' + sd] - d * 1.0, J2['wr' + sd] - d * 0.5, J2['wr' + sd] - d * 0.05], np.array([0.42, 0.46, 0.55]) * gs, mat='belly' if t == 0 else 'plate', sides=10, bone='hero2')
            if t >= 2:
                for xx in (-0.25, 0.0, 0.25):
                    K.spike(part, hc + d * 0.45 * gs + rot_y(A(xx, 0.25, 0), R2), d + A(0, 0.4, 0), 0.25 + 0.08 * t, r=0.08, mat='accent', bone='hero2')
        # tech cannon barrel grows
        cr = 0.38 * (0.85 + 0.2 * sc)
        K.horn(part, [can0, can0 + d4 * 0.6, can1], [cr * 1.15, cr, cr * 0.9], mat='gun', sides=12, bone='hero4')
        V, T = torus(can1, d4, cr * 0.75, 0.08, nu=16, nv=5)
        K.piece(part, V, T, 'accent', bone='hero4')
        if t >= 2:
            for a in (0, 2.1, 4.2):
                o = np.cos(a) * nrm(np.cross(d4, A(0, 1, 0))) + np.sin(a) * A(0, 1, 0)
                K.horn(part, [can0 + d4 * 0.4 + o * cr, can1 - d4 * 0.2 + o * cr * 1.15], [0.07, 0.06], mat='gun', sides=6, bone='hero4')
    for s in (-1, 1):
        bp = J4['chest'] + rot_y(A(s * 0.45, -0.3, -1.35), R4)
        K.horn('body', [bp + A(0, 0.4, 0), bp, bp - A(0, 0.45, 0)], [0.2, 0.22, 0.26], mat='gun', sides=10, bone='hero4')

    # ---------------------------------------------------------------- ARMOR: shoulder pads for all, then chest + shin plates
    for i, f in heroes.items():
        J = f.J
        bn = (lambda nm, sd: nm + sd) if i == 1 else (lambda nm, sd, i=i: f'hero{i}')
        for sd, s in (('L', -1), ('R', 1)):
            sh = J['sh' + sd]
            out = nrm(sh - J['neck']) * A(1, 0, 1) + A(0, 0.4, 0)
            K.plate('arm_t1', sh + out * 0.35 * f.q, A(0.55, 0.48, 0.55) * f.q * f.p['arm'] ** 0.6, bone=bn('upperarm', sd), lames=3, off=0.05, th=0.06, tris=160)
            kn, an = J['kn' + sd], J['an' + sd]
            K.plate('arm_t3', kn + (an - kn) * 0.5 + rot_y(A(0, 0, 0.3), np.eye(3)) * f.q, A(0.38, 0.6, 0.38) * f.q * f.p['leg'] ** 0.6, bone=bn('shin', sd), lames=2, off=0.05, th=0.06, lame_axis=an - kn, tris=140)
        for sd in 'LR':
            el, wr = J['el' + sd], J['wr' + sd]
            K.plate('arm_t2', el + (wr - el) * 0.4, A(0.36, 0.36, 0.36) * f.q * f.p['arm'] ** 0.7, bone=bn('forearm', sd), lames=2, off=0.04, th=0.05, lame_axis=wr - el, tris=120)

    # ---------------------------------------------------------------- ENERGY: glowing emblems, fist / cannon energy, team aura
    for i, f in heroes.items():
        J = f.J
        for sd in 'LR':
            if i == 4 and sd == 'R':
                continue
            d = nrm(J['wr' + sd] - J['el' + sd])
            bn = ('forearm' + sd) if i == 1 else f'hero{i}'
            rr = 0.3 * f.q * f.p['arm'] * (0.8 if f.p['fem'] else 1)
            V, T = torus(J['wr' + sd] - d * 0.3 * f.q, d, rr, 0.045, nu=16, nv=4)
            K.piece('enr_t1', V, T, 'accent', bone=bn)
    for sd in 'LR':
        hc = f2.hand_center(sd)
        K.crystal('enr_t2', hc + f2.hand_dir(sd) * 0.5, f2.hand_dir(sd) + A(0, 0.5, 0), 0.45, r=0.12, bone='hero2', seed=1)
    K.crystal('enr_t2', can1 + d4 * 0.25, d4, 0.6, r=0.18, bone='hero4', seed=4)
    V, T = torus(f3.hand_center('R'), f3.hand_dir('R'), 0.5, 0.05, nu=20, nv=4)
    K.piece('enr_t2', V, T, 'accent', bone='hero3')
    for k in range(3):   # speed lightning trail behind the speedster
        o = A(0, 1.0 + k * 1.2, 0)
        pts = [J5['pelvis'] + rot_y(A(0.3 * (-1) ** k, 0, -1.5), R5) + o - A(0, 2.5, 0) * 0, J5['pelvis'] + rot_y(A(-0.3, 0.3, -2.6), R5) + o, J5['pelvis'] + rot_y(A(0.3, -0.2, -3.6), R5) + o]
        K.horn('enr_t2', pts, [0.08, 0.06, 0.0], mat='accent', sides=4, bone='hero5', per=1)
    # team aura: glowing ground ring around the formation + leader halo ring
    V, T = torus(A(-0.6, 0.15, 0.6), A(0, 1, 0), 10.4, 0.09, nu=64, nv=4)
    K.piece('enr_t3', V, T, 'accent', bone='root')
    for i, f in heroes.items():
        V, T = torus(A(f.J['pelvis'][0], 0.12 + (2.4 if i == 3 else 0), f.J['pelvis'][2]), A(0, 1, 0), 1.6 * f.q, 0.05, nu=32, nv=4)
        K.piece('enr_t3', V, T, 'accent', bone='hero1' if i == 1 else f'hero{i}')

    # ---------------------------------------------------------------- SPEED: boot fins, flyer jet trail, speedster helmet wings
    for i, f in heroes.items():
        J = f.J
        for sd, s in (('L', -1), ('R', 1)):
            an, kn = J['an' + sd], J['kn' + sd]
            sx = nrm(np.cross(an - kn, A(0, 0, 1))) * (-1 if np.cross(an - kn, A(0, 0, 1))[0] * s < 0 else 1)
            p, n = K.snap(an + (kn - an) * 0.2 + A(s, 0, 0) * 0.4 * f.q)
            K.fin('spd_t1', p, n * A(1, 0.2, 0.2) + A(0, 0.5, -0.6), A(0, 0, -1), 0.55 * f.q, 0.55 * f.q, mat='accent', bone=('shin' + sd) if i == 1 else f'hero{i}', sweep_back=0.8)
    for s in (-1, 1):   # flyer: jet streams from the boots
        a = f3.J['toe' + ('R' if s > 0 else 'L')]
        K.horn('spd_t2', [a, a + rot_y(A(0, -0.3, -1.2), R3), a + rot_y(A(0, -0.4, -2.4), R3)], [0.16, 0.12, 0.0], mat='accent', sides=6, bone='hero3')
    V, T = torus(f3.J['toeR'] * 0.5 + f3.J['toeL'] * 0.5 + rot_y(A(0, 0, -0.6), R3), rot_y(A(0, 0.3, 1), R3), 0.6, 0.04, nu=20, nv=4)
    K.piece('spd_t2', V, T, 'accent', bone='hero3')
    for s in (-1, 1):
        K.fin('spd_t3', H5((s * 0.55, 0.3, -0.1)), f5.Rh @ A(s, 0.6, -0.3), f5.Rh @ A(0, 0.3, -1), 0.7, 0.8, mat='accent', bone='hero5_head', sweep_back=1.2)
        K.fin('spd_t3', H1((s * 0.5, 0.1, -0.05)), f1.Rh @ A(s, 0.5, -0.3), f1.Rh @ A(0, 0.3, -1), 0.45, 0.45, mat='plate', bone='head', sweep_back=1.0)

    # ---------------------------------------------------------------- FOCUS: visors / goggles / comms
    for s in (-1, 1):
        c = f2.H((s * 0.15, -0.005, 0.47))
        oball(K, 'fcs_t1', c, A(0.08, 0.05, 0.03) * f2.hs, f2.Rh, mat='accent', bone='hero2_head')
        K.ball('fcs_t1', H1((s * 0.48, -0.05, 0.0)), A(0.06, 0.12, 0.1), mat='accent', bone='head')
    gg = K.sdf()   # speedster goggles
    gg.ell(H5((0, 0.0, 0.07)), A(0.52, 0.12, 0.55) * f5.hs, rot=f5.Rh, k=0)
    gg.ell(H5((0, 0.0, 0.04)), A(0.48, 0.22, 0.51) * f5.hs, rot=f5.Rh, k=0.02, cut=True)
    gg.box(H5((0, 0.0, -0.4)), A(0.7, 0.3, 0.4), rot=f5.Rh, k=0.02, cut=True, rad=0)
    K.sdf_piece('fcs_t1', gg, voxel=0.03, tris=200, mat='accent', bone='hero5_head')
    K.horn('fcs_t2', [H4((0.5, 0.2, -0.1)), H4((0.55, 0.7, -0.2)), H4((0.5, 1.15, -0.3))], [0.05, 0.035, 0], mat='gun', sides=5, bone='hero4_head')
    K.ball('fcs_t2', H4((0.5, 1.18, -0.3)), 0.08, mat='accent', bone='hero4_head')
    K.leaf('fcs_t2', H1((0, 0.62, 0.05)), f1.Rh @ A(0, 1, 0.1), f1.Rh @ A(0, 0, 1), 0.35, 0.7, th=0.04, lobes=3, mat='accent', bone='head')
    for i, f in heroes.items():
        hb_ = 'head' if i == 1 else f'hero{i}_head'
        c, n = K.snap(f.H((0, 0.32, 0.6))) if i == 4 else f.snap_head(K, f.H((0, 0.32, 0.6)))
        K.crystal('fcs_t3', c + n * 0.02, n + f.Rh @ A(0, 0.5, 0), 0.2 * f.hs, r=0.07 * f.hs, bone=hb_, cluster=False)
