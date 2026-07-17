"""Quantize ideal float polyphase kernels into MiSTer integer coefficient tables.

Conventions (matching the v4 pack and MiSTer's ascal):
  - 256 phases; phase p places the sample point at fraction p/256 between the
    two center taps of a 4-tap window [t-1, t, t+1, t+2].
  - Stored tables keep exact conjugate symmetry: row[256-p] == reverse(row[p])
    for p in 1..127, and row[128] is palindromic (self-symmetric).
  - Coefficients are signed 10-bit; unity DC gain corresponds to row sum 256.
  - A palindromic phase-128 row always has an even sum, so an odd target sum
    there is met to within one unit (documented v4 behavior).
"""

from __future__ import annotations

import numpy as np

PHASES = 256
COEF_MIN, COEF_MAX = -512, 511


def round_row_to_sum(row: np.ndarray, target_sum: int) -> np.ndarray:
    """Round a float 4-tap row to integers with an exact target sum.

    Largest-remainder method: floor everything, then hand out the remaining
    units to the taps with the largest fractional parts (ties: center taps
    first, for stability).
    """
    row = np.asarray(row, dtype=np.float64)
    n = row.shape[0]
    base = np.floor(row).astype(np.int64)
    need = int(target_sum - base.sum())
    if need < 0 or need > n:
        # Row scale is far from target; rescale first, then retry once.
        scale = target_sum / row.sum() if row.sum() != 0 else 0.0
        row = row * scale
        base = np.floor(row).astype(np.int64)
        need = int(target_sum - base.sum())
    frac = row - base
    # Preference order: larger fractional part wins; center taps break ties.
    center = (n - 1) / 2.0
    order = sorted(range(n), key=lambda i: (-frac[i], abs(i - center)))
    out = base.copy()
    for k in range(need):
        out[order[k % n]] += 1
    return out.astype(np.int64)


def quantize_symmetric(ideal: np.ndarray, target_sums: np.ndarray) -> np.ndarray:
    """Quantize (256, 4) float kernel rows into a symmetric integer table.

    ideal        : (256, 4) float rows (need not be pre-normalized).
    target_sums  : (256,) integer row sums (the fitted DC gain per phase * 256).

    Phases 0..128 are quantized directly; 129..255 are emitted as exact tap
    reversals of their conjugates, guaranteeing the stored-symmetry invariant.
    For symmetry to also be *correct*, callers must supply target_sums with
    target_sums[256-p] == target_sums[p] (true for any physical scan profile,
    which is a function of |phase distance|); this is asserted.
    """
    ideal = np.asarray(ideal, dtype=np.float64)
    sums = np.asarray(target_sums, dtype=np.int64)
    if ideal.shape != (PHASES, 4) or sums.shape != (PHASES,):
        raise ValueError("expected (256,4) kernel and (256,) sums")
    for p in range(1, 128):
        if sums[PHASES - p] != sums[p]:
            raise ValueError(f"target sums not conjugate-symmetric at phase {p}")

    out = np.zeros((PHASES, 4), dtype=np.int64)
    for p in range(0, 129):
        if p == 128:
            # Palindromic row (a, b, b, a): fit under that constraint.
            half = 0.5 * (ideal[p] + ideal[p][::-1])
            s = int(sums[p])
            if s % 2:                       # odd sum unreachable; use nearest even
                s += 1 if half.sum() * 256 >= sums[p] else -1
            pair = round_row_to_sum(half[:2], s // 2)
            out[p] = np.array([pair[0], pair[1], pair[1], pair[0]])
        else:
            out[p] = round_row_to_sum(ideal[p], int(sums[p]))
    for p in range(129, PHASES):
        out[p] = out[PHASES - p][::-1]

    if out.min() < COEF_MIN or out.max() > COEF_MAX:
        raise ValueError(f"coefficient out of 10-bit range: min {out.min()}, max {out.max()}")
    return out
