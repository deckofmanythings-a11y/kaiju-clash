"""Vectorised value noise / fbm / cellular noise for SDF displacement and vertex tinting."""
import numpy as np


def _hash(ix, iy, iz, seed=0):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + iz.astype(np.int64) * 1274126177 + seed * 974711) & 0xffffffff
    h = ((h ^ (h >> 13)) * 1103515245) & 0xffffffff
    h = h ^ (h >> 16)
    return (h & 0xffffff).astype(np.float32) / np.float32(0x1000000)


def vnoise(x, y, z, seed=0):
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(z, np.float32))
    ix, iy, iz = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    fx, fy, fz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                out = out + w * _hash(ix + dx, iy + dy, iz + dz, seed)
    return out


def fbm(x, y, z, octaves=4, freq=1.0, gain=0.5, seed=0, ridged=False):
    """Returns roughly [-1, 1] (or [0, 1] when ridged)."""
    x, y, z = np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(z, np.float32)
    out, amp, tot = 0, 1.0, 0.0
    for o in range(octaves):
        n = vnoise(x * freq, y * freq, z * freq, seed + o * 17)
        n = (1 - np.abs(2 * n - 1)) if ridged else (2 * n - 1)
        out = out + amp * n
        tot += amp
        amp *= gain
        freq *= 2.03
    return out / tot


def voronoi(x, y, z, size=1.0, seed=0, jitter=0.9):
    """Cellular noise. Returns (F1, F2, cell_random) with distances in units of `size`."""
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float32) / size, np.asarray(y, np.float32) / size, np.asarray(z, np.float32) / size)
    ix, iy, iz = np.floor(x), np.floor(y), np.floor(z)
    f1 = np.full(x.shape, 9.0, np.float32)
    f2 = f1.copy()
    cid = np.zeros(x.shape, np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx, cy, cz = ix + dx, iy + dy, iz + dz
                px = cx + 0.5 + jitter * (_hash(cx, cy, cz, seed) - 0.5)
                py = cy + 0.5 + jitter * (_hash(cx, cy, cz, seed + 1) - 0.5)
                pz = cz + 0.5 + jitter * (_hash(cx, cy, cz, seed + 2) - 0.5)
                d = np.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
                closer = d < f1
                f2 = np.where(closer, f1, np.minimum(f2, d))
                cid = np.where(closer, _hash(cx, cy, cz, seed + 3), cid)
                f1 = np.where(closer, d, f1)
    return f1, f2, cid


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float32) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)
