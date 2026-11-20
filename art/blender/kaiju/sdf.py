"""Numpy signed-distance-field modelling toolkit (game coords: +Y up, +Z forward, +X right).

An SDF is a small CSG "program": smooth unions / cuts of tubes, ellipsoids and boxes plus
narrow-band displacement. It can be evaluated on a voxel grid (with per-primitive bbox culling)
and meshed with OpenVDB, or sampled at arbitrary points (normals, AO, snapping, masks).
"""
import numpy as np

BIG = 1e3
F = np.float32


def rot_xyz(deg):
    rx, ry, rz = np.radians(np.asarray(deg, float))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def catmull(pts, per=5):
    """Catmull-Rom resample of an (N,D) polyline; returns (N-1)*per+1 points."""
    P = np.asarray(pts, float)
    if P.ndim == 1:
        P = P[:, None]
    if per <= 1:
        return P
    if len(P) < 3:
        out = [P[i] + (P[i + 1] - P[i]) * t for i in range(len(P) - 1) for t in np.linspace(0, 1, per, endpoint=False)]
        return np.array(out + [P[-1]])
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return np.array(out)


def smin(a, b, k):
    if k <= 1e-6:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


class Region:
    """Axis-aligned voxel sub-grid."""
    grid = True

    def __init__(self, xs, ys, zs, sl=None):
        self.xs, self.ys, self.zs = xs, ys, zs
        self.x, self.y, self.z = xs[:, None, None], ys[None, :, None], zs[None, None, :]
        self.shape = (len(xs), len(ys), len(zs))
        self.sl = sl

    def sub(self, lo, hi):
        i0, i1 = np.searchsorted(self.xs, lo[0]), np.searchsorted(self.xs, hi[0], 'right')
        j0, j1 = np.searchsorted(self.ys, lo[1]), np.searchsorted(self.ys, hi[1], 'right')
        k0, k1 = np.searchsorted(self.zs, lo[2]), np.searchsorted(self.zs, hi[2], 'right')
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            return None
        return Region(self.xs[i0:i1], self.ys[j0:j1], self.zs[k0:k1], (slice(i0, i1), slice(j0, j1), slice(k0, k1)))

    def coords(self, mask):
        s = self.shape
        return (np.broadcast_to(self.x, s)[mask], np.broadcast_to(self.y, s)[mask], np.broadcast_to(self.z, s)[mask])


class Points:
    grid = False

    def __init__(self, P):
        P = np.asarray(P, F)
        self.x, self.y, self.z = P[:, 0], P[:, 1], P[:, 2]
        self.shape = (len(P),)
        self.sl = slice(None)

    def sub(self, lo, hi):
        return self

    def coords(self, mask):
        return self.x[mask], self.y[mask], self.z[mask]


def _local(reg, c, R):
    px, py, pz = reg.x - F(c[0]), reg.y - F(c[1]), reg.z - F(c[2])
    if R is None:
        return px, py, pz
    R = R.astype(F)
    return (R[0, 0] * px + R[1, 0] * py + R[2, 0] * pz,
            R[0, 1] * px + R[1, 1] * py + R[2, 1] * pz,
            R[0, 2] * px + R[1, 2] * py + R[2, 2] * pz)


_MX = np.diag([-1.0, 1, 1])


def _R(rot):
    if rot is None or isinstance(rot, np.ndarray):
        return rot
    return rot_xyz(rot)


class Ell:
    def __init__(self, c, r, rot=None):
        self.c = np.asarray(c, float)
        self.r = np.broadcast_to(np.asarray(r, float), (3,)).copy()
        self.R = _R(rot)
        e = self.r if self.R is None else np.sqrt(((self.R * self.r[None, :]) ** 2).sum(1))
        self.lo, self.hi = self.c - e, self.c + e

    def mirrored(self):
        return Ell(self.c * [-1, 1, 1], self.r, None if self.R is None else _MX @ self.R @ _MX)

    def dist(self, reg, pad=0):
        lx, ly, lz = _local(reg, self.c, self.R)
        rx, ry, rz = [F(v) for v in self.r]
        k0 = np.sqrt((lx / rx) ** 2 + (ly / ry) ** 2 + (lz / rz) ** 2)
        k1 = np.sqrt((lx / (rx * rx)) ** 2 + (ly / (ry * ry)) ** 2 + (lz / (rz * rz)) ** 2) + F(1e-9)
        return np.where(k0 < 0.5, (k0 - 1) * F(self.r.min()), k0 * (k0 - 1) / k1)


class Box:
    def __init__(self, c, h, rot=None, rad=0.0):
        self.c, self.h, self.rad = np.asarray(c, float), np.asarray(h, float), rad
        self.R = _R(rot)
        e = self.h if self.R is None else (np.abs(self.R) * self.h[None, :]).sum(1)
        self.lo, self.hi = self.c - e, self.c + e

    def mirrored(self):
        return Box(self.c * [-1, 1, 1], self.h, None if self.R is None else _MX @ self.R @ _MX, self.rad)

    def dist(self, reg, pad=0):
        lx, ly, lz = _local(reg, self.c, self.R)
        r = F(self.rad)
        qx, qy, qz = np.abs(lx) - F(self.h[0]) + r, np.abs(ly) - F(self.h[1]) + r, np.abs(lz) - F(self.h[2]) + r
        out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
        return out + np.minimum(np.maximum(np.maximum(qx, qy), qz), 0) - r


class Tube:
    """Round-cone chain along a Catmull-Rom spline. `s` scales the cross-section per world axis.
    `radii` may have fewer keys than `pts`; it is then spread evenly along the spline."""

    def __init__(self, pts, radii, s=(1, 1, 1), per=5, raw=False):
        P = np.asarray(pts, float)
        r = np.atleast_1d(np.asarray(radii, float))
        if len(r) != len(P):
            r = np.interp(np.linspace(0, 1, len(P)), np.linspace(0, 1, len(r)), r) if len(r) > 1 else np.full(len(P), r[0])
        if raw:
            self.P, self.r = P, r
        else:
            self.P = catmull(P, per)
            self.r = np.clip(catmull(r, per)[:, 0], 0.0, None)
        self.s = np.asarray(s, float)
        e = self.r[:, None] * self.s[None, :]
        self.lo, self.hi = (self.P - e).min(0), (self.P + e).max(0)

    def mirrored(self):
        return Tube(self.P * [-1, 1, 1], self.r, self.s, raw=True)

    def dist(self, reg, pad=0.3):
        out = np.full(reg.shape, BIG, F)
        s = self.s.astype(F)
        smin_ = F(s.min())
        for i in range(len(self.P) - 1):
            a, b, ra, rb = self.P[i], self.P[i + 1], self.r[i], self.r[i + 1]
            e = max(ra, rb) * self.s + pad
            sub = reg.sub(np.minimum(a, b) - e, np.maximum(a, b) + e)
            if sub is None:
                continue
            ba = ((b - a) / self.s).astype(F)
            pax, pay, paz = (sub.x - F(a[0])) / s[0], (sub.y - F(a[1])) / s[1], (sub.z - F(a[2])) / s[2]
            bb = float(ba @ ba) + 1e-12
            h = np.clip((pax * ba[0] + pay * ba[1] + paz * ba[2]) / F(bb), 0, 1)
            d = np.sqrt((pax - ba[0] * h) ** 2 + (pay - ba[1] * h) ** 2 + (paz - ba[2] * h) ** 2) - (F(ra) + F(rb - ra) * h)
            d = d * smin_
            v = out[sub.sl]
            np.minimum(v, d, out=v)
        return out


class Shell:
    """Thin shell hugging another SDF: |base - off| - th. `band=(axis, spacing, step, origin)` makes the
    offset a sawtooth along `axis`, which reads as overlapping armour lames."""

    def __init__(self, base, off, th, band=None, taper=None):
        # taper=(region_prim, width, drop): the offset sinks by `drop` toward the region rim (bevelled, seated edge)
        self.base, self.off, self.th, self.band, self.taper = base, off, th, band, taper
        e = off + th + (band[2] if band else 0)
        self.lo, self.hi = base.lo - e, base.hi + e

    def dist(self, reg, pad=0.3):
        off = F(self.off)
        extra = 0.0
        if self.band:
            ax, sp, step, org = self.band
            u = ((reg.x * F(ax[0]) + reg.y * F(ax[1]) + reg.z * F(ax[2])) - F(org)) / F(sp)
            off = off + F(step) * (u - np.floor(u))
            extra = step
        if self.taper:
            rg, w, drop = self.taper
            t = np.clip(1 + rg.dist(reg, pad) / F(w), 0, 1)
            off = off - F(drop) * t * t
        return np.abs(self.base.dist(reg, pad + self.off + self.th + extra) - off) - F(self.th)


class Inter:
    def __init__(self, a, b, k=0.0):
        self.a, self.b, self.k = a, b, k
        self.lo, self.hi = np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi)

    def dist(self, reg, pad=0.3):
        return smax(self.a.dist(reg, pad), self.b.dist(reg, pad), self.k)


class SDF:
    def __init__(self):
        self.ops = []
        self._lo = np.full(3, 1e9)
        self._hi = np.full(3, -1e9)

    # ---- building
    def _push(self, kind, prim, k, mirror):
        self.ops.append((kind, prim, k))
        if mirror:
            self.ops.append((kind, prim.mirrored(), k))
        if kind == 'add':
            for p in ([prim, prim.mirrored()] if mirror else [prim]):
                self._lo = np.minimum(self._lo, p.lo)
                self._hi = np.maximum(self._hi, p.hi)
        return self

    def add(self, prim, k=0.2, mirror=False):
        return self._push('add', prim, k, mirror)

    def cut(self, prim, k=0.1, mirror=False):
        return self._push('cut', prim, k, mirror)

    def tube(self, pts, radii, k=0.25, s=(1, 1, 1), mirror=False, cut=False, per=5):
        return self._push('cut' if cut else 'add', Tube(pts, radii, s, per), k, mirror)

    def ell(self, c, r, rot=None, k=0.25, mirror=False, cut=False):
        return self._push('cut' if cut else 'add', Ell(c, r, rot), k, mirror)

    def box(self, c, h, rot=None, rad=0.1, k=0.1, mirror=False, cut=False):
        return self._push('cut' if cut else 'add', Box(c, h, rot, rad), k, mirror)

    def disp(self, fn, band=0.35):
        """fn(x, y, z) -> distance offset (negative = outward bump), applied near the surface only."""
        self.ops.append(('disp', fn, band))
        return self

    @property
    def lo(self):
        return self._lo

    @property
    def hi(self):
        return self._hi

    # ---- evaluation
    def dist(self, reg, pad=0.4):
        D = np.full(reg.shape, BIG, F)
        for kind, prim, k in self.ops:
            if kind == 'disp':
                mask = np.abs(D) < k
                if mask.any():
                    D[mask] += np.asarray(prim(*reg.coords(mask)), F)
                continue
            p = max(k, pad) + 0.05
            sub = reg.sub(prim.lo - p, prim.hi + p)
            if sub is None:
                continue
            d = prim.dist(sub, p)
            v = D[sub.sl]
            v[...] = smin(v, d, k) if kind == 'add' else smax(v, -d, k)
        return D

    def at(self, P):
        return self.dist(Points(np.asarray(P, F).reshape(-1, 3)))

    def grad(self, P, h=0.02):
        P = np.asarray(P, F).reshape(-1, 3)
        g = np.zeros_like(P)
        for i in range(3):
            e = np.zeros(3, F)
            e[i] = h
            g[:, i] = self.at(P + e) - self.at(P - e)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.maximum(n, 1e-9)

    def snap(self, P, iters=6):
        """Project points onto the surface. Returns (points, outward normals)."""
        P = np.array(P, F).reshape(-1, 3)
        for _ in range(iters):
            P = P - self.grad(P) * self.at(P)[:, None]
        return P.astype(float), self.grad(P).astype(float)

    def ao(self, P, N, steps=(0.12, 0.3, 0.6, 1.0)):
        P = np.asarray(P, F)
        N = np.asarray(N, F)
        acc = np.zeros(len(P), F)
        for h in steps:
            acc += np.clip(self.at(P + N * F(h)) / h, 0, 1)
        return acc / len(steps)

    def mesh(self, voxel=0.05, margin=0.3, lo=None, hi=None):
        """Returns (verts (N,3) float64, quads (M,4) int64) with outward winding."""
        import openvdb as vdb
        lo = (self._lo if lo is None else np.asarray(lo, float)) - margin
        hi = (self._hi if hi is None else np.asarray(hi, float)) + margin
        xs, ys, zs = [np.arange(lo[i], hi[i] + voxel, voxel, dtype=F) for i in range(3)]
        D = self.dist(Region(xs, ys, zs), pad=max(0.3, 4 * voxel))
        bg = 3 * voxel
        g = vdb.FloatGrid(bg)
        g.copyFromArray(np.ascontiguousarray(np.clip(D, -bg, bg)))
        pts, quads = g.convertToQuads(0.0)
        V = np.asarray(pts, np.float64) * voxel + lo
        Q = np.asarray(quads, np.int64)
        if len(Q):
            q = Q[:: max(1, len(Q) // 400)]
            fn = np.cross(V[q[:, 1]] - V[q[:, 0]], V[q[:, 2]] - V[q[:, 0]])
            if (fn * self.grad(V[q[:, 0]])).sum() < 0:
                Q = Q[:, ::-1].copy()
        return V, Q
