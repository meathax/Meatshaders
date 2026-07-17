"""Numerical model of MiSTer's fixed video pipeline for fitting and validation.

Pipeline modeled (in order):
    8-bit input -> per-channel gamma LUT -> horizontal 4-tap polyphase FIR
    (8-bit clamp) -> vertical 4-tap polyphase FIR, optionally adaptive
    (8-bit clamp) -> v2 shadow mask (1/16-step multipliers) -> 8-bit output.

ARITHMETIC — VERIFIED AGAINST RTL 2026-07-17 (ascal.vhd / shadowmask.sv /
video.cpp at the pinned commits; see rtl-verify agent report):
    F1. FIR: file coefficients c (1/256 units) are scaled x128 to 3.15 fixed
        point; each 2-tap pair-product sum is truncated >>8, the two pair
        results are added, and the sum is truncated >>7 (both are arithmetic
        floors). Negative results hard-clamp to 0, >=256 clamps to 255.
        The 8-bit clamp happens after EACH axis (H results are stored 8-bit
        in the line buffers) — there is no headroom between H and V.
    F2. Adaptive blend (V stage): control lum = max(R,G,B) of the H-FILTERED,
        clamped, NEAREST-line pixel at the current column (nearest switches
        at vfrac = 0.5). Blend is linear with one 1-bit truncation:
        c128 = ((A128*(256-lum) + B128*lum) >> 9) << 1. Set A applies at
        lum=0; set B is never fully reached (max weight 255/256).
        H-stage control is the nearest SOURCE pixel (post-gamma, unfiltered).
        Only ONE adaptive slot exists: H's adaptive set beats V's.
    F3. Shadow mask: per-channel multiplier m/16 computed as a sum of
        INDEPENDENTLY truncated shifts over the set bits of m
        (x*m/16 = sum over set bits k of floor(x >> (4-k))), 9-bit result,
        >=256 saturates to 255. Multipliers with many set bits (e.g. 15/16)
        truncate hardest (up to ~3.75 codes low on dark values).
    F4. Phase selection: 256 phases = top 8 bits of the 12-bit fraction;
        taps at [-1, 0, +1, +2] around the base line; bottom edge replicates.
    F5. Invalid filter files silently fall back to nearest-neighbour tables;
        v6 (non-adaptive) cores receive only the FIRST set of adaptive files.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ stages

def apply_gamma(img: np.ndarray, lut: np.ndarray | None) -> np.ndarray:
    """img: (..., 3) uint8-range ints. lut: (256, 3) or None."""
    if lut is None:
        return img.astype(np.int64)
    img = img.astype(np.int64)
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = lut[:, c][img[..., c]]
    return out


def _fir_accumulate(taps: np.ndarray, c128: np.ndarray) -> np.ndarray:
    """Exact ascal accumulator (F1). taps: (M, 4, ...), c128: (M, 4[, ...])
    coefficients already in x128 (3.15) units. Returns clamped 8-bit."""
    c128 = c128.reshape(c128.shape + (1,) * (taps.ndim - c128.ndim))
    prod = taps.astype(np.int64) * c128.astype(np.int64)      # v * 128c
    p01 = (prod[:, 0] + prod[:, 1]) >> 8                      # floor, also negatives
    p23 = (prod[:, 2] + prod[:, 3]) >> 8
    return np.clip((p01 + p23) >> 7, 0, 255)


def _positions_to_taps(positions: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.floor(positions).astype(np.int64)
    ph = np.round((positions - base) * 256).astype(np.int64)
    base = base + (ph == 256).astype(np.int64)                # carry into next window
    phase = ph & 0xFF
    taps_idx = np.clip(base[:, None] + np.arange(-1, 3)[None, :], 0, n - 1)
    return taps_idx, phase


def fir_1d(line_values: np.ndarray, coeffs: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Polyphase-filter a 1-D sequence at fractional source positions (F1/F4).

    line_values : (N, ...) integer samples along the filtered axis.
    coeffs      : (256, 4) integer coefficient table (1/256 units).
    positions   : (M,) float source positions for each output sample.
    Returns (M, ...) clamped 8-bit results.
    """
    taps_idx, phase = _positions_to_taps(positions, line_values.shape[0])
    return _fir_accumulate(line_values[taps_idx], coeffs[phase].astype(np.int64) * 128)


def adaptive_c128(dark: np.ndarray, bright: np.ndarray, phase: np.ndarray,
                  ctrl: np.ndarray) -> np.ndarray:
    """Blended coefficients in x128 units per F2. ctrl in 0..255 (lum)."""
    a128 = dark[phase].astype(np.int64) * 128                 # (M, 4)
    b128 = bright[phase].astype(np.int64) * 128
    lum = ctrl[:, None].astype(np.int64)
    return (((a128 * (256 - lum) + b128 * lum) >> 9) << 1)    # 1-bit truncation


def fir_1d_adaptive(line_values: np.ndarray, dark: np.ndarray, bright: np.ndarray,
                    positions: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """Adaptive vertical FIR (F1 + F2).

    ctrl: (M,) control values 0..255 = max RGB of the H-filtered nearest-line
    pixel (callers must supply the nearest-line value, switching at frac 0.5).
    """
    taps_idx, phase = _positions_to_taps(positions, line_values.shape[0])
    return _fir_accumulate(line_values[taps_idx], adaptive_c128(dark, bright, phase, ctrl))


def mask_multiply(v: np.ndarray, m16: np.ndarray) -> np.ndarray:
    """Exact shadowmask.sv multiplier (F3): sum of independently truncated
    shifts over the set bits of m16; saturates at 255."""
    v = v.astype(np.int64)
    m16 = m16.astype(np.int64)
    acc = np.zeros_like(v)
    for bit, shift in ((1, 4), (2, 3), (4, 2), (8, 1), (16, 0)):
        acc = acc + np.where(m16 & bit, v >> shift, 0)
    return np.minimum(acc, 255)


def apply_mask(img: np.ndarray, mult16: np.ndarray) -> np.ndarray:
    """img: (H, W, 3) codes. mult16: (h, w, 3) integer multipliers in 16ths.

    The mask tile repeats across the output (anchored to the active area).
    """
    H, W, _ = img.shape
    h, w, _ = mult16.shape
    tile = np.tile(mult16, (H // h + 1, W // w + 1, 1))[:H, :W]
    return mask_multiply(img, tile)


# ------------------------------------------------------------- flat fields

def flat_field_profile(x_code: int, vscale: float, dark: np.ndarray,
                       bright: np.ndarray | None = None,
                       lut: np.ndarray | None = None,
                       n_out: int = 256) -> np.ndarray:
    """Vertical intensity profile (pre-mask) of a uniform field of code x_code.

    Returns (n_out,) 8-bit codes. For a flat field the H stage is identity up
    to its own DC gain; callers whose H filter is not unity-DC should fold the
    H gain into x_code first.
    """
    g = lut[:, 0][x_code] if lut is not None else x_code
    lines = np.full(64, g, dtype=np.int64)
    pos = (np.arange(n_out) + 0.5) / vscale - 0.5 + 8.0   # away from edges
    if bright is None:
        return fir_1d(lines, dark, pos)
    ctrl = np.full(n_out, g, dtype=np.int64)              # flat field: ctrl = level
    return fir_1d_adaptive(lines, dark, bright, pos, ctrl)


def moire_metrics(profile: np.ndarray, vscale: float) -> dict:
    """Per-scanline-period peak/trough statistics of a vertical profile."""
    period = vscale
    n_per = int(len(profile) / period) - 2
    peaks, troughs = [], []
    for k in range(1, n_per):
        seg = profile[int(k * period):int((k + 1) * period) + 1]
        if len(seg):
            peaks.append(seg.max())
            troughs.append(seg.min())
    peaks, troughs = np.array(peaks, dtype=np.float64), np.array(troughs, dtype=np.float64)
    return {
        "mean": float(profile.mean()),
        "peak_mean": float(peaks.mean()),
        "trough_mean": float(troughs.mean()),
        "peak_std": float(peaks.std()),
        "trough_std": float(troughs.std()),
        "modulation": float((peaks.mean() - troughs.mean())
                            / max(peaks.mean() + troughs.mean(), 1e-9)),
    }
