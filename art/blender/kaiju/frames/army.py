"""Army: a military task force arrayed against the monster (rare Wild-only enemy).

Four MBTs up front, a radar command truck (the rig "head") behind them, two self-propelled howitzers and a
rocket truck at the rear, an attack helicopter overhead and squads of infantry behind sandbags.
Game coords, faces +Z.

Everything is built as clean low-poly hard-surface geometry (lofted chamfered sections, faceted wedges,
octagonal barrels / wheels) emitted straight into K.piece(..., smooth=False): no SDF meshing, no
decimation, so every edge and bevel is deliberate and the faces stay flat-shaded and crisp.
Body / belly are not detail-mapped, so no mesh needs UVs (keeps the GLB small).
"""
import numpy as np

from .. import geom
from .. import sdf as S
from ..builder import POW_SCALE

DETAIL = 'smooth'
UV_SCALE = 1.0
SCALE = 1.0
CAM = dict(az=30, el=24)
T3_SCALE = 1.1

TANKS = [((-5.75, 0, 2.55), -9), ((-1.95, 0, 3.4), -2), ((1.95, 0, 3.4), 2), ((5.75, 0, 2.55), 9)]
SPH = [((-4.35, 0, -3.7), -6), ((4.35, 0, -3.7), 6)]
MLRS = ((0, 0, -6.7), 0)
TRUCK = ((0, 0, -1.25), 0)
HELI = ((-5.2, 7.4, 1.0), 24)
G = 1.1   # tank scale
TL = 1.95  # tank hull half length (local units)


# ===================================================================== low-poly hard-surface kit
def R_y(deg):
    return S.rot_xyz((0, deg, 0))


class Frame:
    """A local frame (origin on the ground, +Z forward). Callable: local points -> world."""

    def __init__(self, p, yaw=0.0, g=1.0, R=None):
        self.p, self.g = np.asarray(p, float), g
        self.R = R_y(yaw) if R is None else R

    def P(self, q):
        q = np.asarray(q, float) * self.g
        return q @ self.R.T + self.p

    __call__ = P

    def D(self, d):
        return np.asarray(d, float) @ self.R.T


def loft(rings, cap0=True, cap1=True, hub0=None, hub1=None):
    """Closed solid through equal-length rings. cap: True = flat fan, 'hide' = only used to orient
    the winding then dropped. hub = colour multiplier of an extra centre vertex (dished hub / bore).
    Returns V, T, M (per-vertex colour multiplier)."""
    R = [np.asarray(r, float) for r in rings]
    n, m = len(R[0]), len(R)
    V = list(np.vstack(R))
    M = [1.0] * len(V)
    T = []
    for i in range(m - 1):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            j2 = (j + 1) % n
            T += [(a + j, a + j2, b + j2), (a + j, b + j2, b + j)]
    hidden = []
    for which, on, hub in ((0, cap0, hub0), (m - 1, cap1, hub1)):
        if not on:
            continue
        idx = list(range(which * n, which * n + n))
        if which == 0:
            idx = idx[::-1]
        if hub is not None:
            c = len(V)
            V.append(R[which].mean(0))
            M.append(hub)
            tri = [(c, idx[k], idx[(k + 1) % n]) for k in range(n)]
        else:
            tri = [(idx[0], idx[k], idx[k + 1]) for k in range(1, n - 1)]
        if on == 'hide':
            hidden += list(range(len(T), len(T) + len(tri)))
        T += tri
    V, T = np.array(V), np.array(T)
    T = geom.fix_winding(V, T)
    keep = np.ones(len(T), bool)
    keep[hidden] = False
    a = np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    keep &= a > 1e-9
    T = T[keep]
    used = np.unique(T)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return V[used], remap[T], np.array(M)[used]


def sec_rect(w, y0, y1, ct=0.0, cb=0.0):
    """Rectangle cross-section (x, y) with optional top / bottom chamfers."""
    p = [(-w, y0 + cb), (-w + cb, y0), (w - cb, y0), (w, y0 + cb)] if cb else [(-w, y0), (w, y0)]
    p += [(w, y1 - ct), (w - ct, y1), (-w + ct, y1), (-w, y1 - ct)] if ct else [(w, y1), (-w, y1)]
    return p


def sec_hex(wb, wt, y0, ym, y1):
    """Sloped-armour section: vertical lower sides, faces raked in to a narrower roof."""
    return [(-wb, y0), (wb, y0), (wb, ym), (wt, y1), (-wt, y1), (-wb, ym)]


def loft_z(stations, **kw):
    """stations = [(z, [(x, y), ...]), ...] -> loft along +Z."""
    return loft([[(x, y, z) for x, y in sec] for z, sec in stations], **kw)


def loft_x(x0, x1, prof, **kw):
    """Side profile [(z, y), ...] extruded along X."""
    return loft([[(x0, y, z) for z, y in prof], [(x1, y, z) for z, y in prof]], **kw)


def ngon(c, u, r, sides, up=(0, 1, 0)):
    x, _, z = geom.basis(u, up)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.pi / sides
    return np.asarray(c, float) + r * (np.cos(a)[:, None] * x + np.sin(a)[:, None] * z)


def cyl(a, b, prof, sides=8, cap0=True, cap1=True, hub0=None, hub1=None, up=(0, 1, 0)):
    """Lathe along a->b. prof = [(distance along axis, radius), ...]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = geom.unit(b - a)
    if abs(u @ geom.unit(up)) > 0.95:
        up = (0, 0, 1)
    return loft([ngon(a + u * d, u, r, sides, up) for d, r in prof], cap0, cap1, hub0, hub1)


def box(c, h, rot=None, ch=0.0, che=0.0):
    """Box, optional chamfer on the 4 long (local Z) edges (ch) and on the end edges (che)."""
    hx, hy, hz = h
    full = sec_rect(hx, -hy, hy, ch, ch)
    if che:
        ins = sec_rect(hx - che, -hy + che, hy - che, ch * 0.5, ch * 0.5) if ch else sec_rect(hx - che, -hy + che, hy - che)
        st = [(-hz, ins), (-hz + che, full), (hz - che, full), (hz, ins)]
    else:
        st = [(-hz, full), (hz, full)]
    V, T, M = loft_z(st)
    if rot is not None:
        V = V @ S.rot_xyz(rot).T
    return V + np.asarray(c, float), T, M


def unweld(V, T, M, cos_tol=0.999):
    """Give every planar face group its own vertices. The builder auto-smooths all pieces at 50 deg,
    which would round off chamfers / raked armour; disconnected groups shade perfectly flat while
    coplanar triangles still share vertices (keeps the glTF vertex count low)."""
    N = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    par = list(range(len(T)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    edge = {}
    for t, tri in enumerate(T):
        for k in range(3):
            e = (min(tri[k], tri[(k + 1) % 3]), max(tri[k], tri[(k + 1) % 3]))
            if e in edge:
                o = edge[e]
                if N[o] @ N[t] > cos_tol:
                    par[find(o)] = find(t)
            else:
                edge[e] = t
    key, V2, M2, T2 = {}, [], [], []
    for t, tri in enumerate(T):
        g = find(t)
        row = []
        for v in tri:
            kk = (int(v), g)
            if kk not in key:
                key[kk] = len(V2)
                V2.append(V[v])
                M2.append(M[v])
            row.append(key[kk])
        T2.append(row)
    return np.array(V2), np.array(T2), np.array(M2)


def shade(W):
    """Cheap ground occlusion baked into vertex colour (darker toward y=0)."""
    return 0.66 + 0.34 * np.clip(W[:, 1] / 1.6, 0, 1)


class Army:
    def __init__(self, K):
        self.K = K

    def add(self, part, X, VTM, mat='body', bone=None, k=1.0, flat=True):
        V, T, M = VTM
        if flat:
            V, T, M = unweld(V, T, M)
        W = X(V)
        self.K.piece(part, W, T, mat, smooth=False, col=np.clip(shade(W) * M * k, 0.03, 1.0), bone=bone)

    def box(self, part, X, c, h, mat='body', bone=None, rot=None, ch=0.0, che=0.0, k=1.0):
        self.add(part, X, box(c, h, rot, ch, che), mat, bone, k)

    def cyl(self, part, X, a, b, prof, mat='gun', bone=None, sides=8, k=1.0, flat=False, **kw):
        """Lathed parts keep the builder's 50-deg auto smooth: round sides, sharp steps / caps."""
        self.add(part, X, cyl(a, b, prof, sides, **kw), mat, bone, k, flat)

    def rod(self, part, X, a, b, r, mat='gun', bone=None, sides=6, **kw):
        L = np.linalg.norm(np.asarray(b, float) - np.asarray(a, float))
        self.cyl(part, X, a, b, [(0, r), (L, r)], mat, bone, sides, **kw)

    def wheel(self, part, X, c, r, w, side, mat='body', bone=None, hub=0.38, sides=8):
        """Road wheel / tyre along local X; the outer face gets a dark dished centre."""
        c = np.asarray(c, float)
        self.cyl(part, X, c - [side * w, 0, 0], c + [side * w, 0, 0], [(0, r), (2 * w, r)], mat, bone, sides,
                 cap0='hide', hub1=hub)


# ===================================================================== vehicles
def chassis(A, f, bn, L, nw, deck=1.38):
    """Tracked hull: lower hull, wide upper hull with shallow glacis, tracks, road wheels, segmented skirts."""
    A.add('body', f, loft_z([(-L + 0.05, sec_rect(0.9, 0.55, 1.08)), (-L + 0.25, sec_rect(0.9, 0.4, 1.08)),
                             (L - 0.4, sec_rect(0.9, 0.4, 1.08)), (L + 0.02, sec_rect(0.9, 0.84, 1.08))]), 'body', bn)
    A.add('body', f, loft_z([(-L - 0.06, sec_rect(1.5, 1.04, deck - 0.08, 0.05)), (-L + 0.12, sec_rect(1.58, 1.04, deck - 0.02, 0.07)),
                             (L - 0.75, sec_rect(1.58, 1.04, deck, 0.07)), (L + 0.14, sec_rect(1.52, 1.04, 1.14, 0.03))]), 'body', bn)
    for x in (-0.45, 0.45):        # engine deck grilles
        A.box('body', f, (x, deck - 0.005, -L + 0.75), (0.34, 0.012, 0.42), 'dark', bn)
    A.box('body', f, (0, deck - 0.03, L - 1.0), (0.85, 0.03, 0.14), 'belly', bn, ch=0.02)   # glacis tool box / tow cable
    prof = [(-L - 0.12, 0.72), (-L + 0.12, 1.02), (L - 0.1, 1.02), (L + 0.22, 0.8), (L - 0.02, 0.26), (L - 0.3, 0.05),
            (-L + 0.3, 0.05), (-L - 0.05, 0.28)]
    for s in (-1, 1):
        A.add('body', f, loft_x(s * 0.9, s * 1.5, prof, cap0='hide'), 'dark', bn)
        for z in np.linspace(-L + 0.45, L - 0.45, nw):
            A.wheel('body', f, (s * 1.36, 0.33, z), 0.27, 0.17, s, 'body', bn)
        A.wheel('body', f, (s * 1.33, 0.75, L - 0.02), 0.2, 0.15, s, 'gun', bn)          # idler
        A.wheel('body', f, (s * 1.33, 0.72, -L + 0.0), 0.24, 0.15, s, 'gun', bn)         # sprocket
        # segmented side skirts
        zs = np.linspace(-L + 0.05, L - 0.15, 4)
        for z0, z1 in zip(zs[:-1], zs[1:]):
            A.box('body', f, (s * 1.6, 0.86, (z0 + z1) / 2), (0.035, 0.25, (z1 - z0) / 2 - 0.025), 'body', bn, k=0.95)
        A.box('body', f, (s * 1.6, 0.93, L + 0.02), (0.035, 0.18, 0.15), 'body', bn, rot=(-12, 0, 0), k=0.95)
        A.box('body', f, (s * 1.17, 1.09, L + 0.135), (0.1, 0.035, 0.02), 'accent', bn)       # headlights
        A.box('body', f, (s * 1.3, 1.22, -L - 0.075), (0.08, 0.04, 0.02), 'accent', bn)       # tail lights


def tank(A, f, i):
    """Modern MBT: arrow-wedge turret with sloped faces, bustle rack, cupola, segmented skirts."""
    bn, tb = f'tank{i}', f'tank{i}_tur'
    chassis(A, f, bn, TL, 6)
    A.box('body', f, (-0.42, 1.4, 1.3), (0.16, 0.03, 0.12), 'belly', bn)                     # driver hatch
    A.box('body', f, (-0.42, 1.43, 1.44), (0.1, 0.03, 0.025), 'dark', bn)                    # periscope
    # turret shell: raked sides, arrow-head front wedge, long bustle
    A.add('body', f, loft_z([(-1.85, sec_hex(0.72, 0.62, 1.42, 1.74, 1.86)), (-1.25, sec_hex(0.96, 0.8, 1.36, 1.76, 1.95)),
                             (0.35, sec_hex(1.0, 0.84, 1.36, 1.76, 1.95)), (1.2, sec_hex(0.4, 0.32, 1.44, 1.68, 1.8))]), 'body', tb)
    A.add('body', f, cyl((0, 1.3, -0.4), (0, 1.42, -0.4), [(0, 0.9), (0.12, 0.9)], 10, cap0='hide', cap1=False), 'dark', tb)   # turret ring
    A.box('body', f, (0, 1.64, 1.12), (0.26, 0.17, 0.24), 'body', tb, ch=0.05, che=0.03)    # mantlet
    # roof
    A.cyl('body', f, (0.45, 1.9, -0.45), (0.45, 2.06, -0.45), [(0, 0.22), (0.16, 0.2)], 'body', tb, 8)  # cupola
    A.box('body', f, (0.45, 2.08, -0.5), (0.17, 0.025, 0.17), 'belly', tb)                  # cupola hatch
    A.box('body', f, (-0.42, 1.965, -0.55), (0.2, 0.025, 0.22), 'belly', tb, ch=0.01)        # loader hatch
    A.box('body', f, (0.6, 2.03, 0.3), (0.12, 0.09, 0.14), 'body', tb, ch=0.02)              # gunner sight
    A.box('body', f, (0.6, 2.03, 0.445), (0.08, 0.05, 0.01), 'glass', tb)
    A.box('body', f, (0.45, 2.17, -0.25), (0.025, 0.025, 0.25), 'gun', tb)                    # cupola MG
    A.rod('body', f, (-0.55, 1.95, -1.55), (-0.6, 3.1, -1.6), 0.012, 'dark', tb, 4)           # whip antenna
    # stowage: bustle basket + side bins, smoke dischargers, ID panel
    A.box('body', f, (0, 1.62, -2.0), (0.66, 0.17, 0.16), 'belly', tb, ch=0.03)
    for s in (-1, 1):
        A.box('body', f, (s * 0.99, 1.6, -1.2), (0.07, 0.14, 0.42), 'belly', tb, ch=0.02)
        A.box('body', f, (s * 0.88, 1.9, 0.12), (0.1, 0.07, 0.13), 'gun', tb, rot=(0, s * 20, 0), ch=0.02)
        A.box('body', f, (s * 1.005, 1.6, -0.35), (0.008, 0.07, 0.28), 'accent', tb)
    return tb


def gun_barrel(A, part, f, bone, base, d, L, r, brake=False, evac=0.45, sides=8):
    """Smoothbore: thick root, bore evacuator bulge, slim muzzle (+ optional box-ish muzzle brake)."""
    base, d = np.asarray(base, float), geom.unit(d)
    e0, e1 = evac * L, evac * L + 0.18 * L
    prof = [(0, r * 1.3), (0.3, r * 1.15), (0.36, r), (e0 - 0.08, r), (e0, r * 1.55), (e1, r * 1.55), (e1 + 0.08, r * 0.93), (L, r * 0.86)]
    if brake:
        prof += [(L, r * 1.5), (L + 0.32, r * 1.5)]
    A.cyl(part, f, base, base + d * prof[-1][0], prof, 'gun', bone, sides, cap0='hide', hub1=0.12)


def sph(A, f, i):
    """Self-propelled howitzer: long tracked hull + big faceted rear turret."""
    bn, tb = f'art{i}', f'art{i}_tur'
    chassis(A, f, bn, 2.15, 7, deck=1.36)
    A.add('body', f, loft_z([(-2.1, sec_hex(0.98, 0.84, 1.34, 2.2, 2.42)), (0.45, sec_hex(1.06, 0.9, 1.34, 2.26, 2.46)),
                             (0.95, sec_hex(0.88, 0.74, 1.34, 2.0, 2.18))]), 'body', tb)
    A.box('body', f, (0, 1.92, 1.02), (0.34, 0.26, 0.14), 'body', tb, ch=0.06)               # mantlet
    A.box('body', f, (0.5, 2.5, -0.3), (0.2, 0.04, 0.24), 'belly', tb, ch=0.015)             # hatches
    A.box('body', f, (-0.5, 2.5, -0.8), (0.2, 0.04, 0.24), 'belly', tb, ch=0.015)
    A.box('body', f, (0.5, 2.6, 0.05), (0.1, 0.08, 0.1), 'gun', tb, ch=0.02)                 # sight
    A.box('body', f, (0, 1.95, -2.13), (0.42, 0.32, 0.03), 'dark', tb)                       # ammo door
    A.box('body', f, (0, 2.0, -2.25), (0.75, 0.14, 0.1), 'belly', tb, ch=0.02)               # stowage
    for s in (-1, 1):
        A.box('body', f, (s * 1.08, 1.85, -0.9), (0.02, 0.3, 0.32), 'belly', tb)             # side doors
        A.box('body', f, (s * 1.075, 2.32, -0.6), (0.01, 0.05, 0.9), 'accent', tb)           # ID stripe
    return tb


def truck_base(A, f, bn, cab_z=1.55, wheels=(1.4, -0.65, -2.0), length=2.35):
    """6x6 cab-over military truck chassis."""
    A.add('body', f, loft_z([(cab_z - 0.55, sec_rect(0.95, 0.9, 2.05, 0.1)), (cab_z + 0.25, sec_rect(0.95, 0.9, 2.05, 0.1)),
                             (cab_z + 0.55, sec_rect(0.95, 0.9, 1.52, 0.05))]), 'body', bn)
    A.box('body', f, (0, 0.74, -0.35 + (cab_z - 1.55) * 0.3), (0.62, 0.12, length), 'dark', bn)         # frame
    n = geom.unit((0, 0.3, 0.53))
    A.box('body', f, np.array((0, 1.79, cab_z + 0.4)) + n * 0.012, (0.82, 0.25, 0.012), 'glass', bn, rot=(-29.5, 0, 0))
    A.box('body', f, (0, 1.24, cab_z + 0.555), (0.45, 0.16, 0.012), 'dark', bn)                       # grille
    A.box('body', f, (0, 0.98, cab_z + 0.62), (0.93, 0.09, 0.07), 'dark', bn)                         # bumper
    for s in (-1, 1):
        A.box('body', f, (s * 0.952, 1.72, cab_z - 0.05), (0.01, 0.2, 0.22), 'glass', bn)              # side windows
        A.box('body', f, (s * 0.7, 1.12, cab_z + 0.56), (0.13, 0.06, 0.012), 'accent', bn)            # headlights
        A.box('body', f, (s * 1.02, 1.6, cab_z + 0.3), (0.06, 0.12, 0.03), 'dark', bn)                # mirrors
        for z in wheels:
            A.wheel('body', f, (s * 0.9, 0.46, z), 0.46, 0.17, s, 'dark', bn, hub=0.9)
            A.wheel('body', f, (s * 1.075, 0.46, z), 0.24, 0.02, s, 'body', bn, hub=0.45)
        A.box('body', f, (s * 0.92, 1.0, wheels[0]), (0.22, 0.03, 0.52), 'body', bn)                  # mudguard
        A.box('body', f, (s * 0.92, 1.0, (wheels[1] + wheels[2]) / 2), (0.22, 0.03, 0.62 + abs(wheels[1] - wheels[2]) / 2), 'body', bn)


def heli(A, part, f, bone, rotor_bone, fcs3=None):
    """Attack helicopter (Apache-like), faceted low-poly fuselage."""
    st = [(2.5, 0.12, -0.32, -0.12, 0.04), (2.15, 0.3, -0.52, 0.05, 0.1), (1.5, 0.36, -0.6, 0.22, 0.12),
          (0.3, 0.44, -0.62, 0.38, 0.14), (-0.9, 0.44, -0.5, 0.45, 0.14), (-1.6, 0.26, -0.1, 0.42, 0.08), (-4.15, 0.1, 0.28, 0.5, 0.03)]
    A.add(part, f, loft_z([(z, sec_rect(w, y0, y1, c, c)) for z, w, y0, y1, c in st]), 'body', bone)
    can = [(2.05, 0.2, 0.1, -0.02, 0.04, 0.12), (1.55, 0.3, 0.16, 0.1, 0.28, 0.52), (0.98, 0.3, 0.16, 0.15, 0.33, 0.58),
           (0.88, 0.31, 0.17, 0.15, 0.42, 0.7), (0.15, 0.33, 0.17, 0.25, 0.46, 0.76), (-0.15, 0.26, 0.12, 0.3, 0.42, 0.6)]
    A.add(part, f, loft_z([(z, sec_hex(wb, wt, y0, ym, y1)) for z, wb, wt, y0, ym, y1 in can]), 'glass', bone)
    A.box(part, f, (0, 0.72, 0.92), (0.17, 0.03, 0.06), 'body', bone)                       # canopy frame bar
    for s in (-1, 1):
        A.cyl(part, f, (s * 0.62, 0.32, 0.28), (s * 0.62, 0.32, -1.4), [(0, 0.15), (0.14, 0.24), (1.3, 0.24), (1.68, 0.17)], 'body', bone, 6, hub1=0.25)
        A.box(part, f, (s * 0.98, -0.12, -0.15), (0.52, 0.04, 0.27), 'body', bone, rot=(0, 0, s * -6), ch=0.015)  # stub wing
        A.box(part, f, (s * 1.32, -0.3, -0.12), (0.06, 0.07, 0.45), 'gun', bone)               # hellfire rail
        for k in (-1, 1):
            A.cyl(part, f, (s * 1.32 + k * 0.1, -0.44, 0.25), (s * 1.32 + k * 0.1, -0.44, -0.45), [(0, 0.05), (0.08, 0.065), (0.7, 0.065)], 'belly', bone, 6)
        A.box(part, f, (s * 0.4, -0.68, 0.75), (0.03, 0.17, 0.05), 'gun', bone)                # gear leg
        A.wheel(part, f, (s * 0.44, -0.86, 0.75), 0.12, 0.05, s, 'dark', bone, hub=0.9, sides=6)
        A.box(part, f, (s * 0.28, -0.02, 2.05), (0.03, 0.025, 0.02), 'accent', bone)
    A.box(part, f, (0, -0.42, 2.2), (0.16, 0.12, 0.16), 'gun', bone, ch=0.04)                 # TADS sensor turret
    A.box(part, f, (0, -0.42, 2.37), (0.1, 0.07, 0.01), 'glass', bone)
    A.box(part, f, (0, -0.62, 1.65), (0.1, 0.06, 0.1), 'gun', bone)                           # chin gun mount
    A.rod(part, f, (0, -0.64, 1.7), (0, -0.7, 2.5), 0.04, 'gun', bone, 6)                     # chin gun
    A.box(part, f, (0, 0.95, -4.05), (0.045, 0.55, 0.3), 'body', bone, rot=(-18, 0, 0), ch=0.02)   # fin
    A.box(part, f, (0, 0.38, -3.85), (0.72, 0.03, 0.18), 'body', bone, ch=0.01)              # stabilator
    A.box(part, f, (0, 0.62, -4.3), (0.04, 0.04, 0.04), 'accent', bone)
    # main rotor: mast (body bone) + hub and 4 blades on the rotor bone
    A.rod(part, f, (0, 0.45, -0.25), (0, 0.98, -0.25), 0.1, 'gun', bone, 6)
    hub = f.P((0, 1.02, -0.25))
    A.add(part, f, cyl((0, 0.95, -0.25), (0, 1.08, -0.25), [(0, 0.2), (0.13, 0.16)], 6), 'gun', rotor_bone)
    for k in range(4):
        a = np.radians(20 + k * 90)
        d = f.D((np.sin(a), 0, np.cos(a)))
        Rb = np.stack([np.cross([0, 1, 0], d), [0, 1, 0], d], 1) @ S.rot_xyz((0, 0, 6))
        V, T, M = box((0, 0, 0), (0.13, 0.022, 2.1))
        A.add(part, lambda V, c=hub + d * 2.2 + [0, -0.04, 0], R=Rb: V @ R.T + c, (V, T, M), 'dark', rotor_bone)
    tr = f.P((0.12, 0.95, -4.35))
    for k in range(4):
        a = np.radians(k * 90 + 30)
        d = f.D((0, np.sin(a), np.cos(a)))
        Rb = np.stack([f.D((1, 0, 0)), np.cross(d, f.D((1, 0, 0))), d], 1)
        A.add(part, lambda V, c=tr + d * 0.38, R=Rb: V @ R.T + c, box((0, 0, 0), (0.015, 0.06, 0.38)), 'dark', bone)
    if fcs3:   # Longbow mast radome
        A.add(fcs3, f, cyl((0, 1.1, -0.25), (0, 1.5, -0.25), [(0, 0.12), (0.06, 0.34), (0.3, 0.34), (0.4, 0.15)], 8), 'belly', rotor_bone)


def soldier(A, part, p, yaw, kneel=False, bone=None, scale=1.45):
    """Readable chunky infantryman: helmet, face, torso, arms, rifle, legs."""
    f = Frame(p, yaw, scale)
    dk = 0.17 if kneel else 0.0
    if kneel:
        A.box(part, f, (-0.085, 0.24, 0.06), (0.065, 0.065, 0.17), 'body', bone, k=0.75)          # thigh fwd
        A.box(part, f, (-0.085, 0.13, 0.2), (0.06, 0.13, 0.06), 'body', bone, k=0.75)            # shin
        A.box(part, f, (0.085, 0.16, -0.06), (0.06, 0.16, 0.065), 'body', bone, k=0.75)          # kneeling leg
        A.box(part, f, (0.085, 0.035, -0.2), (0.06, 0.035, 0.12), 'body', bone, k=0.75)
    else:
        for s in (-1, 1):
            A.box(part, f, (s * 0.075, 0.24, 0), (0.06, 0.24, 0.07), 'body', bone, k=0.75)
    y = 0.48 - dk
    A.add(part, f, loft_z([(-0.07, sec_rect(0.13, y, y + 0.34, 0.03)), (0.08, sec_rect(0.15, y, y + 0.34, 0.03))]), 'body', bone)  # torso
    A.box(part, f, (0, y + 0.18, -0.1), (0.1, 0.12, 0.04), 'belly', bone)                   # pack
    A.box(part, f, (0, y + 0.42, 0.01), (0.06, 0.07, 0.06), 'belly', bone)                  # face
    A.add(part, f, cyl((0, y + 0.43, 0), (0, y + 0.62, 0), [(0, 0.115), (0.06, 0.11), (0.14, 0.075)], 6, hub1=1.0), 'body', bone)  # helmet
    for s in (-1, 1):   # arms reaching to the rifle
        A.add(part, f, loft([ngon(np.array((s * 0.15, y + 0.3, 0.0)), (0, -0.4, 1), 0.04, 4), ngon(np.array((s * 0.05, y + 0.16, 0.18)), (0, -0.4, 1), 0.035, 4)]), 'body', bone)
    A.box(part, f, (0.02, y + 0.2, 0.22), (0.022, 0.035, 0.26), 'dark', bone, rot=(-12, -8, 0))   # rifle


def sandbags(A, part, f, bone, n=4):
    for row in range(2):
        for j in range(n - row):
            x = (j - (n - row - 1) / 2) * 0.46
            A.box(part, f, (x, 0.11 + row * 0.2, 0.0 + 0.03 * (-1) ** j), (0.22, 0.1, 0.15), 'belly', bone, ch=0.05, k=0.9 + 0.1 * ((j + row) % 2))


# ===================================================================== build
def build(K):
    K.mat('body', rough=0.6, metal=0.15, detail=False)
    K.mat('belly', rough=0.75, metal=0.0, detail=False, double=True)
    K.mat('dark', color='#1d1e18', rough=0.8, metal=0.2)
    K.mat('gun', color='#3a3d34', rough=0.4, metal=0.7, detail=False)
    K.mat('glass', color='#16222c', rough=0.06, metal=0.7, detail=False, vcol=False)
    A = Army(K)

    # ---------------------------------------------------------------- ground vehicles
    tankF = [Frame(p, y, G) for p, y in TANKS]
    turret_bones = [tank(A, f, i + 1) for i, f in enumerate(tankF)]
    sphF = [Frame(p, y) for p, y in SPH]
    sph_bones = [sph(A, sphF[0], 1), sph(A, sphF[1], 3)]
    # command truck
    tf = Frame(*TRUCK)
    truck_base(A, tf, 'truck')
    A.box('body', tf, (0, 1.95, -1.0), (1.0, 0.9, 1.85), 'belly', 'truck', ch=0.07, che=0.05)       # shelter
    A.box('body', tf, (0, 2.6, -1.0), (1.012, 0.045, 1.862), 'accent', 'truck')                    # glowing band
    A.box('body', tf, (0.45, 1.75, 0.86), (0.25, 0.5, 0.012), 'dark', 'truck')                     # door
    A.box('body', tf, (-0.5, 2.0, 0.86), (0.2, 0.14, 0.012), 'glass', 'truck')
    A.box('body', tf, (-0.5, 2.92, 0.2), (0.32, 0.1, 0.35), 'body', 'truck', ch=0.03)              # AC unit
    A.box('body', tf, (0, 2.92, -1.85), (0.35, 0.07, 0.35), 'body', 'truck', ch=0.03)              # mast base
    for s in (-1, 1):
        A.box('body', tf, (s * 1.012, 1.6, -1.6), (0.01, 0.3, 0.5), 'dark', 'truck')               # side hatches
    # rocket truck (MLRS)
    mf = Frame(*MLRS)
    truck_base(A, mf, 'art2', cab_z=1.85, wheels=(1.7, -0.6, -1.85), length=2.6)
    A.box('body', mf, (0, 1.08, -0.9), (0.84, 0.2, 1.4), 'body', 'art2', ch=0.05)                 # launcher bed
    A.add('body', mf, cyl((0, 1.28, -0.85), (0, 1.48, -0.85), [(0, 0.62), (0.2, 0.55)], 10), 'gun', 'art2_tur')   # turntable
    for s in (-1, 1):
        A.box('body', mf, (s * 0.86, 1.08, -1.6), (0.01, 0.12, 0.4), 'belly', 'art2')

    # sandbag walls + infantry squads on the flanks
    WALLS = [((-8.55, 0, 3.95), 18), ((8.55, 0, 3.95), -18), ((-7.7, 0, -1.35), 30), ((7.7, 0, -1.35), -30)]

    # ---------------------------------------------------------------- skeleton: one bone per vehicle (+ turrets)
    def hull_bone(name, f, ln=1.3, r=1.3, y=0.75):
        K.bone(name, f.P((0, y, -ln) / np.array([f.g] * 3)), f.P((0, y, ln) / np.array([f.g] * 3)), None, r)
    for i, f in enumerate(tankF):
        hull_bone(f'tank{i + 1}', f)
        K.bone(f'tank{i + 1}_tur', f.P((0, 1.66, -0.6)), f.P((0, 1.66, 0.6)), f'tank{i + 1}', 0.3)
    for f, nm in zip(sphF, ('art1', 'art3')):
        hull_bone(nm, f)
        K.bone(nm + '_tur', f.P((0, 1.85, -1.1)), f.P((0, 1.85, 0.3)), nm, 0.3)
    hull_bone('art2', mf, ln=1.8, r=1.1)
    K.bone('art2_tur', mf.P((0, 1.6, -1.6)), mf.P((0, 2.2, 0.2)), 'art2', 0.3)
    hull_bone('truck', tf, ln=1.8, r=1.1, y=1.2)
    K.bone('radar', tf.P((0, 2.9, -1.85)), tf.P((0, 4.3, -1.85)), 'truck', 0.25)
    hf = Frame(*HELI)
    K.bone('heli', hf.P((0, 0, -1.5)), hf.P((0, 0, 1.6)), None, 0.7)
    K.bone('rotor', hf.P((0, 0.8, -0.25)), hf.P((0, 1.3, -0.25)), 'heli', 0.2)
    K.bone('infantry', (0, 0.05, -0.5), (0, 0.6, -0.5), None, 0.05)
    for j, (p, yaw) in enumerate(WALLS):
        f = Frame(p, yaw)
        K.bone(f'wall{j + 1}', f.P((-1.1, 0.3, 0)), f.P((1.1, 0.3, 0)), None, 0.5)
        sandbags(A, 'body', f, f'wall{j + 1}')

    # ---------------------------------------------------------------- radar dish (the head), masts
    dish_c = tf.P((0, 4.45, -1.75))
    A.rod('body', tf, (0, 2.95, -1.85), (0, 4.2, -1.85), 0.09, 'gun', 'radar', 6)
    dn = geom.unit((0, 0.55, 1))
    x, y, z = geom.basis(dn, (0, 1, 0))
    V, T = [], []
    nr, ns = 3, 12
    for i in range(nr + 1):
        rr = 1.05 * i / nr
        for j in range(ns):
            a = 2 * np.pi * (j + 0.5 * (i % 2)) / ns
            V.append(dish_c + (np.cos(a) * x + np.sin(a) * z) * max(rr, 1e-3) - dn * (rr * rr * 0.38))
    for i in range(nr):
        for j in range(ns):
            a0, a1 = i * ns + j, i * ns + (j + 1) % ns
            T += [(a0, a1 + ns, a1), (a0, a0 + ns, a1 + ns)]
    K.piece('body', np.array(V), np.array(T), 'belly', smooth=False, col=np.repeat(np.linspace(0.7, 1.0, nr + 1), ns), bone='radar')
    A.add('body', lambda V: V, cyl(dish_c - dn * 0.3, dish_c + dn * 0.85, [(0, 0.045), (1.15, 0.03)], 4), 'gun', 'radar')
    A.box('body', lambda V: V, dish_c + dn * 0.9, (0.08, 0.08, 0.08), 'accent', 'radar')
    A.rod('body', tf, (0.75, 2.85, -2.6), (0.75, 4.6, -2.6), 0.015, 'dark', 'truck', 4)
    A.rod('body', tf, (-0.75, 2.85, -0.3), (-0.75, 3.9, -0.3), 0.015, 'dark', 'truck', 4)

    # ---------------------------------------------------------------- MLRS launcher pod
    Rp = mf.R @ S.rot_xyz((-24, 0, 0))

    def pod(part, offx=0.0, glow=None):
        c = mf.P((offx, 2.05, -0.95))
        X = lambda V: V @ Rp.T + c   # noqa: E731
        w = 0.55 if not offx else 0.42
        A.add(part, X, box((0, 0, 0), (w, 0.52, 1.55), ch=0.05, che=0.04), 'body', 'art2_tur')
        A.add(part, X, box((0, 0.535, -0.2), (w - 0.12, 0.02, 1.1)), 'belly', 'art2_tur')
        nx = 3 if not offx else 2
        for ix in range(nx):
            for iy in range(2):
                hc = np.array([(ix - (nx - 1) / 2) * 0.34, (iy - 0.5) * 0.46, 1.555])
                A.add(part, X, cyl(hc - [0, 0, 0.02], hc + [0, 0, 0.01], [(0, 0.13), (0.03, 0.13)], 6, cap0='hide', hub1=0.25, up=(0, 1, 0)), 'dark', 'art2_tur')
                if glow:
                    A.add(glow, X, cyl(hc - [0, 0, 0.0], hc + [0, 0, 0.05], [(0, 0.09), (0.05, 0.07)], 6, cap0='hide'), 'accent', 'art2_tur')

    # ---------------------------------------------------------------- guns per Might tier (exclusive sets)
    for t, sc in enumerate(POW_SCALE):
        part = f'pow_t{t}'
        kL = 1 + 0.13 * t
        rr = 1 + 0.12 * t
        for i, f in enumerate(tankF):
            tb = turret_bones[i]
            Lb = 2.75 * kL
            xs = (-0.15, 0.15) if t == 3 else (0.0,)
            for xo in xs:
                gun_barrel(A, part, f, tb, (xo, 1.64, 1.3), (0, 0, 1), Lb, 0.098 * rr, brake=t >= 2)
            if t >= 1:
                A.rod(part, f, (-0.34, 1.7, 1.2), (-0.34, 1.7, 1.6 + 0.2 * t), 0.035, 'gun', tb, 6)       # coax / 2nd MG
            if t == 0:
                mz = f.P((0, 1.64, 1.3 + Lb))
                if i == 1:
                    K.mount('mouth', mz, tb)
                if i == 0:
                    K.mount('handL', mz, tb)
                if i == 3:
                    K.mount('handR', mz, tb)
        for f, tb in zip(sphF, sph_bones):
            el = np.radians(28)
            d = np.array([0, np.sin(el), np.cos(el)])
            b0 = np.array([0, 1.92, 1.1])
            gun_barrel(A, part, f, tb, b0, d, 3.4 * kL, 0.1 * rr, brake=True, evac=0.3)
            A.rod(part, f, b0 + [0, 0.22, -0.15], b0 + [0, 0.22, 0] + d * 1.2, 0.055, 'gun', tb, 6)   # recoil cylinder
        pod(part)
        if t >= 2:
            pod(part, offx=-1.0)
            if t >= 3:
                pod(part, offx=1.0)
        if t >= 2:   # helicopter chin cannon gets longer
            A.rod(part, hf, (0, -0.64, 2.4), (0, -0.72, 2.85 + 0.2 * t), 0.055, 'gun', 'heli', 6)

    # ---------------------------------------------------------------- helicopter
    heli(A, 'body', hf, 'heli', 'rotor', fcs3='fcs_t3')

    # ---------------------------------------------------------------- infantry (squads behind the sandbags)
    for (p, yaw) in WALLS:
        f = Frame(p, yaw)
        for j in range(4):
            x = (j - 1.5) * 0.58
            soldier(A, 'body', f.P((x, 0, -0.55 - 0.12 * (j % 2))), yaw + (j - 1.5) * 4, kneel=(j % 2 == 1), bone='infantry')

    # ---------------------------------------------------------------- mounts
    K.mount('head', dish_c, 'radar')
    K.mount('jaw', tf.P((0, 1.3, 2.15)), 'truck')
    K.mount('chest', (0, 2.0, 0.8), 'truck')
    K.mount('back', mf.P((0, 2.6, -0.9)), 'art2')
    K.mount('tailTip', mf.P((0, 1.3, -3.2)), 'art2')
    K.mount('crown', dish_c + [0, 1.3, 0], 'radar')
    K.mount('shoulderL', tankF[0].P((0, 2.05, -0.3)), turret_bones[0])
    K.mount('shoulderR', tankF[3].P((0, 2.05, -0.3)), turret_bones[3])

    # ---------------------------------------------------------------- Armor: ERA blocks -> slat cages + heavy skirts -> mine plows
    for i, f in enumerate(tankF):
        tb, bn = turret_bones[i], f'tank{i + 1}'
        for s in (-1, 1):
            for c in range(3):    # glacis ERA bricks
                A.box('arm_t1', f, (s * (0.28 + c * 0.36), 1.32 - c * 0.0, TL - 0.45), (0.16, 0.05, 0.22), 'body', bn, rot=(17, 0, 0), ch=0.02, k=0.92)
            for u in (0.2, 0.5, 0.8):    # turret cheek ERA, flush on the raked wedge face
                q = np.array((1.0, 0.35)) + u * np.array((-0.6, 0.85)) + np.array((0.817, 0.577)) * 0.035
                A.box('arm_t1', f, (s * q[0], 1.6 - 0.04 * u, q[1]), (0.035, 0.12 - 0.03 * u, 0.13), 'belly', tb, rot=(0, -s * 35.2, 0), ch=0.015)
            for c in range(5):    # heavy skirt plates
                A.box('arm_t2', f, (s * 1.68, 0.84, -1.6 + c * 0.78), (0.05, 0.27, 0.36), 'belly', bn, ch=0.02)
            A.box('arm_t2', f, (s * 1.05, 1.72, -1.2), (0.015, 0.2, 0.62), 'gun', tb)             # slat cage
            for zz in np.linspace(-1.75, -0.6, 6):
                A.box('arm_t2', f, (s * 1.05, 1.72, zz), (0.02, 0.27, 0.015), 'gun', tb)
        A.box('arm_t2', f, (0, 1.72, -2.22), (1.05, 0.015, 0.015), 'gun', tb)
        A.box('arm_t3', f, (0, 0.55, TL + 0.75), (1.1, 0.22, 0.04), 'gun', bn, rot=(-25, 0, 0), ch=0.01)   # mine plow
        for s in (-1, 1):
            A.box('arm_t3', f, (s * 0.7, 0.82, TL + 0.4), (0.05, 0.05, 0.38), 'gun', bn, rot=(25, 0, 0))
        for j in range(7):
            base = f.P((-0.9 + j * 0.3, 0.32, TL + 0.85))
            K.spike('arm_t3', base, f.D((0, -0.55, 1)), 0.42, r=0.07, bone=bn, mat='gun', sides=4)
        for s in (-1, 1):
            K.spike('arm_t3', f.P((s * 0.8, 1.95, -0.2)), f.D((s * 0.5, 1, 0)), 0.35, r=0.05, bone=tb, mat='gun', sides=4)
    for f, tb, bn in zip(sphF, sph_bones, ('art1', 'art3')):
        for s in (-1, 1):
            for c in range(4):
                A.box('arm_t1', f, (s * 1.1, 1.85, -1.55 + c * 0.42), (0.04, 0.3, 0.17), 'body', tb, ch=0.015, k=0.92)
            A.box('arm_t2', f, (s * 1.68, 0.84, 0), (0.05, 0.27, 1.95), 'belly', bn, ch=0.02)
        for j in range(6):
            K.spike('arm_t3', f.P((-0.75 + j * 0.3, 0.6, 2.35)), f.D((0, -0.3, 1)), 0.38, r=0.07, bone=bn, mat='gun', sides=4)
    for s in (-1, 1):
        A.box('arm_t1', tf, (s * 1.04, 1.95, -1.0), (0.03, 0.7, 1.6), 'body', 'truck', ch=0.01)
        A.box('arm_t2', tf, (s * 0.99, 1.5, 1.55), (0.03, 0.45, 0.45), 'gun', 'truck')
        for zz in np.linspace(-2.6, 0.6, 9):
            A.box('arm_t3', tf, (s * 1.13, 1.95, zz), (0.02, 0.85, 0.02), 'gun', 'truck')

    # ---------------------------------------------------------------- Energy: searchlights -> missile pods -> energy emitters
    for i, f in enumerate(tankF):
        tb = turret_bones[i]
        A.box('enr_t1', f, (-0.6, 2.05, 0.3), (0.12, 0.1, 0.1), 'gun', tb, ch=0.02)
        A.box('enr_t1', f, (-0.6, 2.05, 0.405), (0.09, 0.07, 0.01), 'accent', tb)
        A.box('enr_t2', f, (0.75, 2.12, -0.95), (0.17, 0.15, 0.38), 'body', tb, ch=0.03)          # turret missile box
        for xx in (0.67, 0.83):
            for yy in (2.05, 2.19):
                A.box('enr_t2', f, (xx, yy, -0.565), (0.05, 0.05, 0.01), 'accent', tb)
        A.cyl('enr_t3', f, (0, 1.64, 1.4), (0, 1.64, 1.75), [(0, 0.15), (0.12, 0.15), (0.12, 0.12), (0.22, 0.12), (0.22, 0.15), (0.35, 0.15)], 'accent', tb, 8, cap0='hide', cap1=False)
    for f, tb in zip(sphF, sph_bones):
        A.box('enr_t1', f, (0.6, 2.52, 0.1), (0.1, 0.06, 0.05), 'accent', tb)
        d = np.array([0, np.sin(np.radians(28)), np.cos(np.radians(28))])
        b0 = np.array([0, 1.92, 1.1])
        A.cyl('enr_t3', f, b0 + d * 0.15, b0 + d * 0.65, [(0, 0.19), (0.5, 0.19)], 'accent', tb, 8, cap0='hide', cap1=False)
    for s in (-1, 1):   # heli rocket pods
        A.cyl('enr_t2', hf, (s * 0.82, -0.36, 0.45), (s * 0.82, -0.36, -0.6), [(0, 0.15), (0.06, 0.17), (1.05, 0.17)], 'body', 'heli', 8, hub0=0.3)
        A.box('enr_t2', hf, (s * 0.82, -0.36, 0.455), (0.07, 0.07, 0.01), 'accent', 'heli')
    pod('enr_t3', glow='enr_t2')
    for k2, yy in enumerate((3.2, 3.55, 3.9)):   # emitter rings on the radar mast
        A.cyl('enr_t3', tf, (0, yy - 0.05, -1.85), (0, yy + 0.05, -1.85), [(0, 0.28 - k2 * 0.04), (0.1, 0.28 - k2 * 0.04)], 'accent', 'radar', 8)
    for s in (-1, 1):
        A.cyl('enr_t1', tf, (s * 0.6, 2.85, 0.6), (s * 0.6, 3.1, 0.6), [(0, 0.1), (0.25, 0.1)], 'accent', 'truck', 6)

    # ---------------------------------------------------------------- Speed: recon drone -> fighter jet -> second gunship
    K.bone('drone', (6.6, 5.4, 4.0), (6.6, 5.4, 5.0), None, 0.3)
    df = Frame((6.6, 5.4, 4.5), -20)
    A.add('spd_t1', df, loft_z([(-0.95, sec_rect(0.07, -0.07, 0.07, 0.03, 0.03)), (-0.3, sec_rect(0.13, -0.12, 0.13, 0.05, 0.05)),
                                (0.55, sec_rect(0.12, -0.11, 0.14, 0.05, 0.05)), (0.95, sec_rect(0.05, -0.05, 0.05, 0.02, 0.02))]), 'belly', 'drone')
    A.box('spd_t1', df, (0, 0.1, 0.05), (1.6, 0.02, 0.15), 'belly', 'drone', ch=0.01)
    A.box('spd_t1', df, (0, 0.17, -0.85), (0.5, 0.015, 0.09), 'belly', 'drone')
    A.box('spd_t1', df, (0, -0.1, 0.62), (0.06, 0.04, 0.06), 'accent', 'drone')
    A.cyl('spd_t1', df, (0, 0, -0.95), (0, 0, -1.02), [(0, 0.1), (0.07, 0.1)], 'accent', 'drone', 6)
    K.bone('jet', (2.6, 10.2, -4.2), (2.6, 10.2, -2.2), None, 0.5)
    jf = Frame((2.6, 10.2, -3.2), R=R_y(15) @ S.rot_xyz((0, 0, -25)))
    A.add('spd_t2', jf, loft_z([(-2.0, sec_rect(0.3, -0.25, 0.25, 0.1, 0.1)), (-0.5, sec_rect(0.38, -0.3, 0.3, 0.12, 0.12)),
                                (1.2, sec_rect(0.3, -0.25, 0.24, 0.1, 0.1)), (2.65, sec_rect(0.04, -0.04, 0.03, 0.01, 0.01))]), 'belly', 'jet')
    A.add('spd_t2', jf, loft_z([(0.5, sec_hex(0.12, 0.06, 0.15, 0.25, 0.3)), (1.25, sec_hex(0.2, 0.12, 0.15, 0.38, 0.48)),
                                (1.85, sec_hex(0.1, 0.05, 0.1, 0.22, 0.27))]), 'glass', 'jet')
    for s in (-1, 1):
        A.add('spd_t2', jf, loft([[(s * 0.3, -0.04, 0.7), (s * 0.3, -0.04, -1.1), (s * 0.3, 0.04, -1.1), (s * 0.3, 0.04, 0.7)],
                                  [(s * 2.3, -0.02, -0.9), (s * 2.3, -0.02, -1.3), (s * 2.3, 0.02, -1.3), (s * 2.3, 0.02, -0.9)]]), 'belly', 'jet')
        A.add('spd_t2', jf, loft([[(s * 0.25, -0.03, -1.4), (s * 0.25, -0.03, -2.1), (s * 0.25, 0.03, -2.1), (s * 0.25, 0.03, -1.4)],
                                  [(s * 1.1, -0.015, -1.95), (s * 1.1, -0.015, -2.2), (s * 1.1, 0.015, -2.2), (s * 1.1, 0.015, -1.95)]]), 'belly', 'jet')
        A.add('spd_t2', jf, loft([[(s * 0.22, 0.2, -1.2), (s * 0.22, 0.2, -2.0), (s * 0.24, 0.24, -2.0), (s * 0.24, 0.24, -1.2)],
                                  [(s * 0.55, 1.0, -1.75), (s * 0.55, 1.0, -2.15), (s * 0.57, 1.0, -2.15), (s * 0.57, 1.0, -1.75)]]), 'belly', 'jet')
        A.box('spd_t2', jf, (s * 2.3, 0, -1.1), (0.03, 0.03, 0.2), 'accent', 'jet')
    A.cyl('spd_t2', jf, (0, 0, -2.0), (0, 0, -3.3), [(0, 0.28), (1.3, 0.04)], 'accent', 'jet', 8)
    hf2 = Frame((4.2, 6.6, -5.0), -30)
    K.bone('heli2', hf2.P((0, 0, -1.5)), hf2.P((0, 0, 1.6)), None, 0.7)
    K.bone('rotor2', hf2.P((0, 0.8, -0.25)), hf2.P((0, 1.3, -0.25)), 'heli2', 0.2)
    heli(A, 'spd_t3', hf2, 'heli2', 'rotor2')

    # ---------------------------------------------------------------- Focus: sensor masts -> lattice comms tower -> phased arrays + Longbow radome
    for i, f in enumerate(tankF):
        tb = turret_bones[i]
        A.rod('fcs_t1', f, (0.0, 1.95, -1.1), (0.0, 2.55, -1.1), 0.03, 'gun', tb, 6)
        A.box('fcs_t1', f, (0.0, 2.62, -1.1), (0.13, 0.08, 0.1), 'body', tb, ch=0.02)
        A.box('fcs_t1', f, (0.0, 2.62, -0.995), (0.08, 0.05, 0.01), 'accent', tb)
    K.bone('mast', (-7.8, 0.0, -6.9), (-7.8, 1.0, -6.9), None, 0.3)
    mH = 8.2
    mx = Frame((-7.8, 0, -6.9))
    for x, z in ((-0.45, -0.45), (0.45, -0.45), (0.45, 0.45), (-0.45, 0.45)):
        A.rod('fcs_t2', mx, (x, 0, z), (x * 0.25, mH, z * 0.25), 0.045, 'gun', 'mast', 4)
    for k2 in range(1, 8):
        yy = k2 * mH / 8
        w = 0.45 * (1 - 0.75 * yy / mH)
        A.box('fcs_t2', mx, (0, yy, 0), (w, 0.025, w), 'gun', 'mast')
    A.box('fcs_t2', mx, (0, mH + 0.1, 0), (0.12, 0.12, 0.12), 'accent', 'mast', rot=(45, 45, 0))
    for s in (-1, 1):
        A.box('fcs_t2', mx, (s * 0.35, mH - 0.4, 0), (0.02, 0.4, 0.3), 'belly', 'mast', rot=(0, s * 30, 0))
    for s in (-1, 1):   # phased-array panels on the command shelter
        A.box('fcs_t3', tf, (s * 0.6, 3.1, -0.3), (0.32, 0.35, 0.04), 'belly', 'truck', rot=(-20, s * 35, 0), ch=0.02)
        A.box('fcs_t3', tf, (s * 0.6, 3.1, -0.3) + S.rot_xyz((-20, s * 35, 0)) @ np.array([0, 0, 0.045]), (0.26, 0.29, 0.008), 'accent', 'truck', rot=(-20, s * 35, 0))
