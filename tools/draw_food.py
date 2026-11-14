#!/usr/bin/env python3
"""Render one picture of food: a bowl of tonkotsu ramen under a single warm lamp.

Deterministic, offline, no external assets: a fixed seed and fixed geometry.
The scene is shaded as a 2.5D height field (numpy), with a photographic-ish
light rig: warm key from the upper left, cool fill from the right, a hot rim
light from behind, contact shadow, shallow depth of field, bloom and grain.

Writes art/entry.png (1024x1024, well under 1 MiB).
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageFilter

S = 2048                      # supersampled render size
OUT = 1024                    # final size
SEED = 20261114
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "art")

# ---------------------------------------------------------------- geometry ---
CX, CY = 0.500, 0.585         # bowl centre
RX = 0.455                    # bowl radius (u units)
KY = 0.60                     # vertical squash: the camera looks down at ~37 deg
Z_RIM = 1.00
Z_BROTH = 0.27
RIM_IN = 0.945                # inner edge of the rim band
FR = 0.795                    # broth radius

TABLE, CERAMIC, WALL, BROTH = 0, 1, 2, 3
EGG_W, YOLK, CHASHU, NOODLE = 4, 5, 6, 7
SCALLION, CHILI, WOOD, NORI = 8, 9, 10, 11
FAT = 12


def smoothstep(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Separable box blur along one axis, edge-padded (no scipy on this host)."""
    if r < 1:
        return a
    pad = [(r, r) if i == axis else (0, 0) for i in range(a.ndim)]
    p = np.pad(a.astype(np.float32), pad, mode="edge")
    cs = np.cumsum(p, axis=axis, dtype=np.float64)
    zeros_shape = list(cs.shape)
    zeros_shape[axis] = 1
    cs = np.concatenate([np.zeros(zeros_shape, dtype=np.float64), cs], axis=axis)
    n = a.shape[axis]
    k = 2 * r + 1
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(k, k + n)
    lo[axis] = slice(0, n)
    return ((cs[tuple(hi)] - cs[tuple(lo)]) / k).astype(np.float32)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Three box passes: a close, fast approximation of a gaussian of `sigma` px."""
    if sigma <= 0:
        return a.astype(np.float32)
    r = max(1, int(round(float(sigma) * 0.9)))
    out = a.astype(np.float32)
    for _ in range(3):
        out = _box(out, r, 0)
        out = _box(out, r, 1)
    return out


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def ellipse_dome(dx, dy, cx, cy, ra, rb, rot=0.0):
    """Normalised elliptical radius and a smooth dome profile for one topping."""
    ca, sa = math.cos(rot), math.sin(rot)
    px, py = dx - cx, dy - cy
    ex = (px * ca + py * sa) / ra
    ey = (-px * sa + py * ca) / rb
    q = np.sqrt(ex * ex + ey * ey)
    return q, np.sqrt(np.clip(1.0 - q * q, 0.0, 1.0))


def seg_dist(dx, dy, x0, y0, x1, y1):
    """Distance from every pixel to one segment, in (dx, dy) scene units."""
    vx, vy = x1 - x0, y1 - y0
    wx, wy = dx - x0, dy - y0
    t = np.clip((wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-9), 0.0, 1.0)
    return np.hypot(wx - t * vx, wy - t * vy)


def value_noise(rng, size, cells):
    """Cheap smooth noise: a small random field scaled up with bicubic filtering."""
    g = rng.random((cells, cells)).astype(np.float32)
    im = Image.fromarray(g, mode="F").resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)

    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    u = (x + 0.5) / S
    v = (y + 0.5) / S
    dx = (u - CX) / RX
    dy = (v - CY) / (RX * KY)
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)

    # ---------------------------------------------------------- base surface
    z = np.zeros_like(r)
    mat = np.full(r.shape, TABLE, dtype=np.int8)

    z = np.where(r < RIM_IN, Z_RIM - 0.010 * smoothstep(0.0, RIM_IN, r), z)
    z = np.where((r >= RIM_IN) & (r < 1.0), Z_RIM, z)
    mat = np.where(r < 1.0, CERAMIC, mat)

    wall_t = (RIM_IN - r) / (RIM_IN - FR)
    z = np.where((r < RIM_IN) & (r >= FR), Z_RIM - (Z_RIM - Z_BROTH) * smoothstep(0.0, 1.0, wall_t), z)
    mat = np.where((r < RIM_IN) & (r >= FR), WALL, mat)

    z = np.where(r < FR, Z_BROTH, z)
    mat = np.where(r < FR, BROTH, mat)

    # broth surface: gentle swell so the light has something to catch
    swell = 0.004 * np.sin(2.1 * dx + 0.7) * np.cos(1.7 * dy - 0.4) + 0.003 * np.sin(4.3 * dx - 2.0 * dy)
    z = np.where(r < FR, Z_BROTH + swell, z)

    # ------------------------------------------------------------- toppings
    det = np.ones_like(z)

    # two halved soft-boiled eggs, cut face up, jammy yolks glowing
    for (ex, ey, ra, rb, rot) in ((-0.30, 0.10, 0.255, 0.215, 0.35),
                                  (0.22, -0.24, 0.235, 0.200, -0.55)):
        q, dome = ellipse_dome(dx, dy, ex, ey, ra, rb, rot)
        inside = q < 1.0
        z = np.where(inside, Z_BROTH + 0.010 + 0.052 * dome, z)
        mat = np.where(inside, EGG_W, mat)
        det = np.where(inside, 0.84 + 0.20 * (1.0 - q) + 0.06 * dome, det)
        yq, ydome = ellipse_dome(dx, dy, ex, ey, ra * 0.52, rb * 0.55, rot)
        yin = yq < 1.0
        z = np.where(yin, Z_BROTH + 0.030 + 0.075 * ydome, z)
        mat = np.where(yin, YOLK, mat)
        det = np.where(yin, 0.72 + 0.50 * ydome, det)

    # three chashu slices: rolled fat edge, braised grain, a little char
    for (px, py, ra, rb, rot) in ((0.33, 0.17, 0.255, 0.150, 0.55),
                                  (0.44, -0.04, 0.235, 0.140, 0.30),
                                  (-0.05, 0.42, 0.245, 0.145, -0.18)):
        q, dome = ellipse_dome(dx, dy, px, py, ra, rb, rot)
        inside = q < 1.0
        z = np.where(inside, Z_BROTH + 0.005 + 0.042 * dome, z)
        mat = np.where(inside, CHASHU, mat)
        ca, sa = math.cos(rot), math.sin(rot)
        along = ((dx - px) * ca + (dy - py) * sa) / ra
        grain_c = 0.82 + 0.22 * np.sin(74.0 * along + 2.4 * np.sin(9.0 * along))
        det = np.where(inside, grain_c, det)
        fat = inside & (q > 0.87)
        mat = np.where(fat, FAT, mat)
        z = np.where(fat, Z_BROTH + 0.003 + 0.030 * dome, z)
        det = np.where(fat, 0.95, det)

    # noodles: wavy tubes across the front of the broth, each a little different
    for k, (y0, amp, freq, phase, w) in enumerate((
            (-0.30, 0.050, 3.2, 0.0, 0.040),
            (-0.16, 0.058, 2.6, 1.2, 0.046),
            (-0.02, 0.046, 3.6, 2.4, 0.037),
            (0.12, 0.055, 2.8, 0.6, 0.044),
            (0.26, 0.042, 4.0, 1.8, 0.033))):
        centre = y0 + amp * np.sin(freq * dx + phase)
        d = np.abs(dy - centre)
        band = (r < FR * 0.995) & (d < w)
        prof = np.sqrt(np.clip(1.0 - (d / w) ** 2, 0.0, 1.0))
        z = np.where(band, np.maximum(z, Z_BROTH + 0.018 + 0.052 * prof), z)
        mat = np.where(band, NOODLE, mat)
        det = np.where(band, 0.86 + 0.22 * prof + 0.05 * np.sin(150.0 * dx + k), det)

    # the lifted coil: arcs rising out of the broth toward the chopsticks
    for (x0, y0, x1, y1, w) in ((-0.34, -0.44, 0.02, -0.26, 0.030),
                                (-0.28, -0.38, 0.06, -0.22, 0.026),
                                (-0.10, -0.52, 0.20, -0.36, 0.024)):
        d = seg_dist(dx, dy, x0, y0, x1, y1)
        band = d < w
        prof = np.sqrt(np.clip(1.0 - (d / w) ** 2, 0.0, 1.0))
        z = np.where(band, np.maximum(z, Z_RIM * 1.02 + 0.10 * prof), z)
        mat = np.where(band, NOODLE, mat)
        det = np.where(band, 0.90 + 0.18 * prof, det)

    # scallion rings
    for (sx, sy, sr) in ((-0.16, -0.16, 0.045), (0.06, 0.30, 0.040), (0.30, 0.34, 0.036),
                         (-0.46, 0.34, 0.033), (0.14, 0.14, 0.038), (0.44, -0.26, 0.034),
                         (-0.36, -0.06, 0.030)):
        dd = np.hypot(dx - sx, dy - sy)
        ring = (np.abs(dd - sr) < sr * 0.42) & (r < FR * 0.99)
        z = np.where(ring, np.maximum(z, Z_BROTH + 0.032), z)
        mat = np.where(ring, SCALLION, mat)
        det = np.where(ring, 1.08, det)

    # chilli oil droplets, glossy
    for (px, py, pr) in ((-0.58, 0.02, 0.030), (0.55, 0.24, 0.026), (0.30, -0.44, 0.022),
                         (-0.36, 0.46, 0.024), (0.10, 0.46, 0.020), (-0.10, 0.18, 0.018),
                         (0.52, 0.02, 0.017), (-0.52, -0.24, 0.016)):
        dq = np.hypot(dx - px, dy - py)
        dot = (dq < pr) & (r < FR * 0.99)
        z = np.where(dot, np.maximum(z, Z_BROTH + 0.016), z)
        mat = np.where(dot, CHILI, mat)
        det = np.where(dot, 1.10, det)

    # a sheet of nori standing at the back of the bowl, leaning away
    npx, npy = dx + 0.42, dy + 0.50
    nca, nsa = math.cos(-0.20), math.sin(-0.20)
    nxx = (npx * nca + npy * nsa) / 0.28
    nyy = (-npx * nsa + npy * nca) / 0.19
    qn = np.maximum(np.abs(nxx), np.abs(nyy))
    sheet = qn < 1.0
    edge = smoothstep(1.0, 0.80, qn)
    z = np.where(sheet, Z_BROTH + 0.30 + 0.16 * edge + 0.14 * (0.5 - 0.5 * nyy), z)
    mat = np.where(sheet, NORI, mat)
    det = np.where(sheet, 0.72 + 0.40 * edge - 0.08 * np.clip(nyy, -1.0, 1.0), det)

    # wooden chopsticks dipping in from the upper right
    for (x0, y0, x1, y1, w) in ((0.62, -1.10, 0.16, -0.30, 0.036),
                                (0.72, -1.12, 0.24, -0.34, 0.034)):
        d = seg_dist(dx, dy, x0, y0, x1, y1)
        band = d < w
        prof = np.sqrt(np.clip(1.0 - (d / w) ** 2, 0.0, 1.0))
        z = np.where(band, np.maximum(z, Z_RIM * 1.10 + 0.12 * prof), z)
        mat = np.where(band, WOOD, mat)

    # ------------------------------------------------------------- shading
    scale = 300.0
    gy, gx = np.gradient(z)
    nrm = np.dstack([-gx * scale, -gy * scale, np.ones_like(z)])
    nrm /= np.sqrt((nrm * nrm).sum(axis=2, keepdims=True)) + 1e-9

    # wood grain on the table, stretched along the light direction
    grain = 0.55 * value_noise(rng, S, 5) + 0.30 * value_noise(rng, S, 21) + 0.15 * value_noise(rng, S, 90)
    grain = blur(grain, 1.2)

    albedo = np.zeros((S, S, 3), dtype=np.float32)
    albedo[mat == TABLE] = (0.115, 0.078, 0.058)
    albedo[mat == CERAMIC] = (0.132, 0.110, 0.100)
    albedo[mat == WALL] = (0.098, 0.080, 0.072)
    albedo[mat == BROTH] = (0.500, 0.235, 0.078)
    albedo[mat == EGG_W] = (0.945, 0.885, 0.795)
    albedo[mat == YOLK] = (0.975, 0.615, 0.135)
    albedo[mat == CHASHU] = (0.660, 0.330, 0.255)
    albedo[mat == FAT] = (0.880, 0.700, 0.610)
    albedo[mat == NOODLE] = (0.920, 0.790, 0.505)
    albedo[mat == SCALLION] = (0.330, 0.620, 0.230)
    albedo[mat == CHILI] = (0.880, 0.210, 0.060)
    albedo[mat == WOOD] = (0.310, 0.175, 0.108)
    albedo[mat == NORI] = (0.048, 0.070, 0.058)

    tbl = mat == TABLE
    albedo[tbl, 0] *= 0.72 + 0.85 * grain[tbl]
    albedo[tbl, 1] *= 0.66 + 0.80 * grain[tbl]
    albedo[tbl, 2] *= 0.62 + 0.75 * grain[tbl]

    br = mat == BROTH
    mottle = blur(value_noise(rng, S, 26), 6.0)
    albedo[br, 0] *= 0.80 + 0.50 * mottle[br]
    albedo[br, 1] *= 0.78 + 0.46 * mottle[br]
    albedo[br, 2] *= 0.80 + 0.40 * mottle[br]

    ch = (mat == CHASHU) | (mat == FAT)
    albedo[ch, 0] *= 0.80 + 0.55 * blur(value_noise(rng, S, 40), 2.0)[ch]

    # every topping carries its own detail field (grain, dome falloff, fat edge)
    albedo *= det[..., None]

    # rendered fat and oil: tiny glossy beads floating on the broth
    beads = (mottle > 0.62) & br
    albedo[beads] *= 1.35

    # light rig ------------------------------------------------------------
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    warm = np.array([1.00, 0.845, 0.660], dtype=np.float32)
    cool = np.array([0.360, 0.470, 0.640], dtype=np.float32)
    hot = np.array([1.00, 0.660, 0.320], dtype=np.float32)

    L1 = unit((-0.44, -0.62, 0.72))     # warm key, upper left
    L2 = unit((0.78, 0.10, 0.50))       # cool fill, right
    L3 = unit((0.12, -0.92, 0.34))      # hot rim, behind

    def ndl(L):
        return np.clip(nrm[..., 0] * L[0] + nrm[..., 1] * L[1] + nrm[..., 2] * L[2], 0.0, 1.0)

    # the key is a lamp above the table, not a flat sky: it falls off with distance
    lamp = np.exp(-(((u - 0.30) ** 2) / 0.34 + ((v - 0.20) ** 2) / 0.46))

    key_d = ndl(L1)[..., None] * warm * (0.55 + 1.35 * lamp[..., None])
    fill_d = ndl(L2)[..., None] * cool * 0.30
    rim_d = ndl(L3)[..., None] * hot * (0.55 + 0.85 * lamp[..., None])

    # ambient occlusion from local height relative to its neighbourhood
    relief = z - blur(z, 26.0)
    ao = np.clip(0.62 + 9.0 * relief, 0.34, 1.06)

    # contact shadow under the bowl, on the table only
    contact = 1.0 - 0.75 * (1.0 - smoothstep(0.90, 1.28, r)) * tbl
    ao = ao * contact

    amb = np.array([0.150, 0.128, 0.118], dtype=np.float32)
    light = amb * ao[..., None] + (key_d + fill_d + rim_d) * ao[..., None]

    spec_power = np.zeros_like(z)
    spec_strength = np.zeros_like(z)
    spec_power[mat == BROTH] = 180.0
    spec_strength[mat == BROTH] = 1.30
    spec_power[mat == YOLK] = 90.0
    spec_strength[mat == YOLK] = 0.85
    spec_power[mat == EGG_W] = 46.0
    spec_strength[mat == EGG_W] = 0.45
    spec_power[mat == CERAMIC] = 110.0
    spec_strength[mat == CERAMIC] = 0.80
    spec_power[mat == NOODLE] = 34.0
    spec_strength[mat == NOODLE] = 0.32
    spec_power[mat == SCALLION] = 40.0
    spec_strength[mat == SCALLION] = 0.35
    spec_power[mat == CHILI] = 120.0
    spec_strength[mat == CHILI] = 0.90
    spec_power[mat == FAT] = 70.0
    spec_strength[mat == FAT] = 0.55
    spec_power[mat == NORI] = 64.0
    spec_strength[mat == NORI] = 0.30
    spec_power[mat == WOOD] = 26.0
    spec_strength[mat == WOOD] = 0.18

    pw = np.maximum(spec_power, 6.0)
    H1 = unit((L1 + V).tolist())
    h1 = np.clip(nrm[..., 0] * H1[0] + nrm[..., 1] * H1[1] + nrm[..., 2] * H1[2], 0.0, 1.0)
    H3 = unit((L3 + V).tolist())
    h3 = np.clip(nrm[..., 0] * H3[0] + nrm[..., 1] * H3[1] + nrm[..., 2] * H3[2], 0.0, 1.0)
    spec = ((h1 ** pw) * 1.0 + (h3 ** pw) * 0.60) * spec_strength

    # a hard highlight running along the near lip of the bowl
    lip = np.exp(-((r - 0.982) / 0.016) ** 2) * (0.30 + 0.70 * np.clip(np.cos(ang + 2.35), 0.0, 1.0) ** 6)
    spec = spec + lip * 1.5 * (mat == CERAMIC)

    col = albedo * light + spec[..., None] * (warm * 1.0 + hot * 0.55)

    # a broad, soft reflection of the lamp lying on the broth, plus the wet meniscus
    sheen = np.exp(-(((u - 0.40) ** 2) / 0.055 + ((v - 0.44) ** 2) / 0.030))
    col += (sheen * 0.50)[..., None] * np.array([1.00, 0.86, 0.66], dtype=np.float32) * (mat == BROTH)[..., None]

    # directioned cast shadow of the bowl on the table, thrown away from the lamp
    sh = np.exp(-(((dx - 0.34) / 1.05) ** 2 + ((dy - 0.30) / 0.72) ** 2) * 1.6)
    col *= (1.0 - 0.62 * sh * tbl)[..., None]

    # steam wisps ----------------------------------------------------------
    steam = np.zeros_like(r)
    for (sx, amp, freq, phase, wid, z0) in ((-0.30, 0.10, 2.2, 0.3, 0.085, 0.0),
                                           (0.02, 0.13, 1.7, 1.9, 0.110, 0.6),
                                           (0.30, 0.09, 2.6, 3.1, 0.070, 1.4),
                                           (-0.10, 0.16, 1.3, 4.2, 0.135, 2.2),
                                           (0.16, 0.06, 3.4, 5.1, 0.055, 2.9)):
        rise = np.clip((CY - v) / 0.52, 0.0, 1.6)
        cxv = sx + amp * np.sin(freq * rise * 3.0 + phase)
        w = wid * (0.50 + 1.7 * rise)
        band = np.exp(-((dx - cxv) ** 2) / (w * w))
        vert = smoothstep(0.0, 0.07, rise) * (1.0 - smoothstep(0.45, 1.30, rise))
        steam += band * vert * (0.60 + 0.55 * np.sin(2.7 * rise * 3.0 + z0))
    steam = blur(np.clip(steam, 0.0, None), 16.0)
    steam *= smoothstep(1.70, 1.05, np.hypot((dx - 0.02) / 1.1, dy * 0.55))
    steam = np.clip(steam * 0.95, 0.0, 0.78)
    steam_col = np.array([1.00, 0.95, 0.88], dtype=np.float32)
    col = col * (1.0 - steam[..., None]) + steam_col * steam[..., None] * 1.08

    # background falloff: warm pool of lamp light, near-black at the edges
    rr = np.hypot((u - CX) / 1.05, (v - CY) / 1.15)
    pool = np.exp(-(rr * 1.55) ** 2)
    col += (pool[..., None] * np.array([0.115, 0.078, 0.048], dtype=np.float32)) * (tbl[..., None] * 1.0)

    # tone map
    col = np.clip(col, 0.0, None)
    col = (col * 1.55) / (1.0 + col * 1.05)
    col = np.clip(col, 0.0, 1.0) ** (1.0 / 1.06)

    # bloom
    luma = (col * np.array([0.30, 0.62, 0.08], dtype=np.float32)).sum(axis=2)
    bright = np.clip((luma - 0.60) / 0.40, 0.0, 1.0)[..., None] * col
    col = np.clip(col + blur(bright, 26.0) * 0.34, 0.0, 1.1)

    # vignette
    vig = 1.0 - 0.55 * smoothstep(0.62, 1.32, np.hypot(u - 0.5, v - 0.52) * 1.55)
    col *= vig[..., None]

    # film grain, tiny
    col = np.clip(col + rng.normal(0.0, 0.0045, size=(S, S, 1)).astype(np.float32), 0.0, 1.0)

    # shallow depth of field: the far edge and the corners go soft
    dof_w = smoothstep(0.78, 1.35, np.hypot(dx, dy * 0.92))
    soft = np.dstack([blur(col[..., c], 22.0) for c in range(3)])
    col = col * (1.0 - dof_w[..., None] * 0.62) + soft * (dof_w[..., None] * 0.62)

    img = Image.fromarray((np.clip(col, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    img = img.resize((OUT, OUT), Image.LANCZOS)
    os.makedirs(ART, exist_ok=True)
    out = os.path.join(ART, "entry.png")
    img.save(out, format="PNG", optimize=True, compress_level=9)
    print(f"wrote {out} ({os.path.getsize(out)} bytes, {img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
