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
        at vfrac = 0.5).  The stored 2.8 coefficients are multiplied by the
        2.8 blend weights, summed in 4.16, then sliced with t(18 downto 1):
        c128 = (A*(256-lum) + B*lum) >> 1.  This is one arithmetic truncation
        and can produce odd 3.15 values.  Set A applies at lum=0; set B is
        never fully reached (max weight 255/256).
        H-stage control is the nearest SOURCE pixel (post-gamma, unfiltered).
        Only ONE adaptive-secondary slot exists.  Main uploads H's secondary
        set when both axes request adaptation, but the RTL blend arbitration
        gives V priority; selecting adaptive filters on both axes therefore
        mismatches the sets and is unsupported.
    F3. Shadow mask: per-channel multiplier m/16 computed as a sum of
        INDEPENDENTLY truncated shifts over the set bits of m
        (x*m/16 = sum over set bits k of floor(x >> (4-k))), 9-bit result,
        >=256 saturates to 255. Multipliers with many set bits (e.g. 15/16)
        truncate hardest (up to ~3.75 codes low on dark values).
    F4. Phase selection: 256 phases = top 8 bits of the 12-bit fraction;
        taps at [-1, 0, +1, +2] around the base line; both edges replicate.
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
    """Resolve source positions exactly like ascal's 12-bit fraction path.

    ``o_[hv]frac`` is a 12-bit unsigned fraction and a 256-phase core indexes
    the coefficient RAM with bits 11..4.  Selecting the phase is therefore a
    truncation (floor), not a nearest-phase rounding.  Keeping the explicit
    12-bit step here also makes boundary behaviour testable and prevents a
    value just below the next source pixel from carrying into that pixel.
    """
    base = np.floor(positions).astype(np.int64)
    frac12 = np.floor((positions - base) * 4096.0).astype(np.int64)
    # Floating-point callers can only approach the [0, 4095] hardware range;
    # clamp defensively rather than inventing a carry that the RTL cannot emit.
    frac12 = np.clip(frac12, 0, 4095)
    phase = frac12 >> 4                                      # bits 11..4
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
    a = dark[phase].astype(np.int64)                          # 2.8 (M, 4)
    b = bright[phase].astype(np.int64)
    lum = ctrl[:, None].astype(np.int64)
    # ascal poly_lerp: (2.8 * 2.8 + 2.8 * 2.8) is 4.16, then
    # t(18 downto 1) converts it to 3.15.  This is one arithmetic >> 1;
    # the result may legitimately be odd.  The old >>9/<<1 formulation
    # discarded an additional bit and incorrectly forced every result even.
    return (a * (256 - lum) + b * lum) >> 1


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
                       n_out: int = 256, channel: int = 0) -> np.ndarray:
    """Vertical intensity profile (pre-mask) of a uniform field of code x_code.

    Returns (n_out,) 8-bit codes for ``channel``.  Adaptive control is the
    maximum RGB LUT output, independently of the signal channel, matching the
    scaler RTL.  For a flat field the H stage is identity up to its own DC
    gain; callers whose H filter is not unity-DC should fold the H gain into
    x_code first.
    """
    if channel not in (0, 1, 2):
        raise ValueError(f"channel must be 0, 1, or 2, got {channel}")
    if lut is None:
        signal = ctrl_level = x_code
    else:
        signal = int(lut[x_code, channel])
        ctrl_level = int(lut[x_code].max())
    lines = np.full(64, signal, dtype=np.int64)
    pos = (np.arange(n_out) + 0.5) / vscale - 0.5 + 8.0   # away from edges
    if bright is None:
        return fir_1d(lines, dark, pos)
    ctrl = np.full(n_out, ctrl_level, dtype=np.int64)
    return fir_1d_adaptive(lines, dark, bright, pos, ctrl)


def moire_metrics(profile: np.ndarray, vscale: float) -> dict:
    """Exact sampled beam-centre/trough statistics for one output frame.

    Output pixel ``j`` samples source position ``(j+.5)/scale-.5``.  The old
    implementation split the profile into overlapping integer slices and
    included both endpoints in every slice.  At fractional scale that lets two
    neighbouring scanlines reuse whichever shared trough is darker, hiding the
    real alternating line thickness (most visibly at 240p -> 1080p = 4.5x).

    For every complete source-line cell in the supplied frame, sample the two
    output pixels bracketing its exact beam centre and retain the brighter one;
    do the analogous operation at every inter-line midpoint and retain the
    darker one.  These are the discrete extrema actually present on screen.
    """
    profile = np.asarray(profile)
    if profile.ndim != 1:
        raise ValueError(f"profile must be one-dimensional, got {profile.shape}")
    if len(profile) < 2:
        raise ValueError("profile must contain at least two output samples")
    if not np.isfinite(vscale) or vscale <= 0:
        raise ValueError(f"vscale must be finite and positive, got {vscale}")

    # Source coordinates covered by the first/last output-pixel centres.
    pos_first = 0.5 / vscale - 0.5
    pos_last = (len(profile) - 0.5) / vscale - 0.5
    # Retain only source-line cells whose two half-line boundaries lie inside
    # the sampled frame; this removes incomplete edge cells deterministically.
    first_center = int(np.ceil(pos_first + 0.5))
    last_center = int(np.floor(pos_last - 0.5))
    centers = np.arange(first_center, last_center + 1, dtype=np.float64)
    if len(centers) < 2:
        raise ValueError("profile does not cover two complete source-line cells")

    def bracketing_indices(source_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Invert source_pos=(j+.5)/scale-.5, then inspect both neighbouring
        # integer output pixels.  floor==ceil at an exactly sampled extremum.
        jf = (source_positions + 0.5) * vscale - 0.5
        lo = np.clip(np.floor(jf).astype(np.int64), 0, len(profile) - 1)
        hi = np.clip(np.ceil(jf).astype(np.int64), 0, len(profile) - 1)
        return lo, hi

    plo, phi = bracketing_indices(centers)
    tlo, thi = bracketing_indices(centers[:-1] + 0.5)
    peaks = np.maximum(profile[plo], profile[phi]).astype(np.float64)
    troughs = np.minimum(profile[tlo], profile[thi]).astype(np.float64)
    return {
        "mean": float(profile.mean()),
        "peak_mean": float(peaks.mean()),
        "trough_mean": float(troughs.mean()),
        "peak_std": float(peaks.std()),
        "trough_std": float(troughs.std()),
        "modulation": float((peaks.mean() - troughs.mean())
                            / max(peaks.mean() + troughs.mean(), 1e-9)),
    }
