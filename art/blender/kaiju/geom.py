"""Small numpy mesh generators for rigid parts (claws, teeth, horns, crystals, eyes, fins).
Everything is in game coords and returns (V (N,3), T (M,3) int, u (N,) 0..1 along-length param)."""
import numpy as np
from .sdf import catmull, rot_xyz


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def basis(n, t=None):
    """Orthonormal (x, y, z) with y = n and z ~ t."""
    y = unit(n)
    z = np.array([0.0, 0, 1]) if t is None else np.asarray(t, float)
    z = z - y * (z @ y)
    if np.linalg.norm(z) < 1e-4:
        z = np.array([1.0, 0, 0]) if abs(y[0]) < 0.9 else np.array([0.0, 0, 1])
        z = z - y * (z @ y)
    z = unit(z)
    return np.cross(y, z), y, z


def fix_winding(V, T):
    v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    c = V.mean(0)
    vol = (np.cross(v0 - c, v1 - c) * (v2 - c)).sum()
    return T[:, ::-1].copy() if vol < 0 else T


def sweep(pts, radii, sides=8, per=3, flat=1.0, up=None):
    """Swept tube along a spline; a (near) zero end radius collapses to a point. `flat` squashes the
    cross-section along the frame binormal (blade-like claws)."""
    C = catmull(pts, per)
    r = np.clip(catmull(np.asarray(radii, float), per)[:, 0], 0, None) if len(radii) == len(pts) else np.interp(
        np.linspace(0, 1, len(C)), np.linspace(0, 1, len(radii)), radii)
    n = len(C)
    tan = np.gradient(C, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    nx, _, _ = basis(tan[0], up)
    N = [nx]
    for i in range(1, n):
        v = N[-1] - tan[i] * (N[-1] @ tan[i])
        N.append(unit(v))
    N = np.array(N)
    B = np.cross(tan, N)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V, U, ring_start = [], [], []
    for i in range(n):
        if r[i] < 1e-3 and i in (0, n - 1):
            ring_start.append((len(V), 1))
            V.append(C[i]); U.append(i / (n - 1))
        else:
            ring_start.append((len(V), sides))
            for a in ang:
                V.append(C[i] + r[i] * (np.cos(a) * N[i] + np.sin(a) * B[i] * flat)); U.append(i / (n - 1))
    T = []
    for i in range(n - 1):
        (a0, ca), (b0, cb) = ring_start[i], ring_start[i + 1]
        for j in range(sides):
            j2 = (j + 1) % sides
            if ca == 1 and cb == 1:
                continue
            if ca == 1:
                T.append((a0, b0 + j, b0 + j2))
            elif cb == 1:
                T.append((a0 + j, b0, a0 + j2))
            else:
                T.append((a0 + j, b0 + j, b0 + j2)); T.append((a0 + j, b0 + j2, a0 + j2))
    for idx, (s0, cnt) in ((0, ring_start[0]), (n - 1, ring_start[-1])):
        if cnt > 1:  # cap
            c = len(V); V.append(C[idx]); U.append(idx / (n - 1))
            for j in range(sides):
                T.append((c, s0 + j, s0 + (j + 1) % sides))
    V, T = np.array(V), np.array(T)
    return V, _consistent(V, T), np.array(U)


def _consistent(V, T):
    """Per-triangle outward flip relative to the mesh centroid-ish axis (robust for convex-ish parts)."""
    c = V.mean(0)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    fc = V[T].mean(1) - c
    flip = (fn * fc).sum(1) < 0
    T = T.copy()
    T[flip] = T[flip][:, ::-1]
    return T


def crystal(base, n, h, r, sides=6, t=None, lean=0.0):
    x, y, z = basis(n, t)
    y = unit(y + x * lean)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False) + 0.3
    V = [base + r * 0.75 * (np.cos(a) * x + np.sin(a) * z) - y * r * 0.3 for a in ang]
    V += [base + r * (np.cos(a) * x + np.sin(a) * z) + y * h * 0.62 for a in ang]
    V.append(base + y * h)
    T = []
    for j in range(sides):
        j2 = (j + 1) % sides
        T += [(j, sides + j, sides + j2), (j, sides + j2, j2), (sides + j, 2 * sides, sides + j2)]
    V, T = np.array(V), np.array(T)
    u = np.array([0.0] * sides + [0.6] * sides + [1.0])
    return V, _consistent(V, T), u


def sphere(c, r, seg=12, rings=8, rot=None):
    r = np.broadcast_to(np.asarray(r, float), (3,))
    V = [(0, 1, 0)]
    for i in range(1, rings):
        ph = np.pi * i / rings
        for j in range(seg):
            th = 2 * np.pi * j / seg
            V.append((np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)))
    V.append((0, -1, 0))
    V = np.array(V) * r
    if rot is not None:
        V = V @ rot_xyz(rot).T
    V = V + np.asarray(c, float)
    T = []
    for j in range(seg):
        T.append((0, 1 + j, 1 + (j + 1) % seg))
    for i in range(rings - 2):
        a, b = 1 + i * seg, 1 + (i + 1) * seg
        for j in range(seg):
            j2 = (j + 1) % seg
            T += [(a + j, b + j, b + j2), (a + j, b + j2, a + j2)]
    last = len(V) - 1
    a = 1 + (rings - 2) * seg
    for j in range(seg):
        T.append((a + j, last, a + (j + 1) % seg))
    T = np.array(T)
    return V, _consistent(V, T), np.linspace(0, 1, len(V))


def strip(base, tips, close=False):
    """Membrane between two equal-length polylines (double-sided material expected)."""
    A, B = np.asarray(base, float), np.asarray(tips, float)
    n = len(A)
    V = np.vstack([A, B])
    T = []
    for i in range(n - 1):
        T += [(i, n + i, n + i + 1), (i, n + i + 1, i + 1)]
    u = np.concatenate([np.zeros(n), np.ones(n)])
    return V, np.array(T), u


def fin_profile(base, n, t, h=1.2, w=1.0, seg=9, sweep_back=0.45):
    """Curved shark-fin style sail: base at `base`, rising along n, chord along t."""
    _, y, z = basis(n, t)
    base = np.asarray(base, float)
    lead, trail = [], []
    for i in range(seg + 1):
        k = i / seg
        # leading edge arcs up and back; trailing edge is concave
        lead.append(base + z * w * (0.55 - (0.55 + sweep_back) * k ** 1.6) + y * h * (1 - (1 - k) ** 1.8))
        trail.append(base + z * w * (-0.85 + (0.85 - sweep_back) * k ** 0.7) + y * h * k ** 1.5 * 0.98)
    return strip(trail, lead)


def grid(fn, nu=16, nv=10):
    """Parametric surface fn(u, v) -> (3,), u,v in [0,1]."""
    V = np.array([[fn(i / nu, j / nv) for j in range(nv + 1)] for i in range(nu + 1)]).reshape(-1, 3)
    T = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = a + nv + 1
            T += [(a, b, b + 1), (a, b + 1, a + 1)]
    u = np.tile(np.linspace(0, 1, nv + 1), nu + 1)
    return V, np.array(T), u
