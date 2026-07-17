"""Regression tests for exact full-period shadow-mask scoring.

The hardware and shader masks often have different periods.  These tests
compare the FFT/correlation implementation against a deliberately slow brute
force evaluator over the LCM supercell, exercise MiSTer's 2x token expansion,
and prove that an upper-left crop can report a false near-perfect match.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "targets"))

import fileio  # noqa: E402
import fitting  # noqa: E402
import mister_model as mm  # noqa: E402
import guest_advanced_ref as ref  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FB = "CRT Guest Advanced (Port)"
CODES = np.array([0, 17, 93, 255], dtype=np.int64)
FS = np.arange(0, 33) / 64.0


def brute(dark, bright, h, lut, tokens, target, mask_scale):
    """Independent exhaustive roll scorer over the complete LCM supercell."""
    mask = fileio.MaskFile([], len(tokens[0]), len(tokens), tokens)
    m16 = np.rint(mask.multipliers() * 16).astype(np.int64)
    if mask_scale == 2:
        m16 = np.repeat(np.repeat(m16, 2, axis=0), 2, axis=1)
    mh, mw, _ = m16.shape
    th, tw, _ = target.shape
    sh, sw = int(np.lcm(mh, th)), int(np.lcm(mw, tw))
    hardware = np.tile(m16, (sh // mh, sw // mw, 1))
    base_target = np.tile(target, (sh // th, sw // tw, 1))

    scores = np.zeros((th, tw), dtype=np.float64)
    maxima = np.zeros((th, tw), dtype=np.float64)
    for code in CODES:
        sim = fitting.simulate_flat_rgb(
            dark, bright, h, lut, int(code), FS)
        port = mm.mask_multiply(sim[None, None, :, :],
                                hardware[:, :, None, :])
        beam = np.array([[
            fitting._ref_vertical_unclipped(
                ref, f, int(code) / 255.0, channel)
            for channel in "rgb"] for f in FS])
        for dy in range(th):
            for dx in range(tw):
                rolled = np.roll(base_target, (dy, dx), axis=(0, 1))
                expected = 255.0 * np.minimum(
                    rolled[:, :, None, :] * beam[None, None, :, :], 1.0)
                error = port - expected
                scores[dy, dx] += float((error * error).sum())
                maxima[dy, dx] = max(maxima[dy, dx],
                                     float(np.abs(error).max()))
    best = np.unravel_index(int(np.argmin(scores)), scores.shape)
    count = len(CODES) * sh * sw * len(FS) * 3
    return (float(np.sqrt(scores[best] / count)),
            float(maxima[best]), tuple(int(v) for v in best))


def check_scale(dark, bright, h, lut, tokens, target, scale):
    exact = fitting._rmse_exact_masked_periodic(
        ref, dark, bright, h, lut, tokens, target,
        codes=CODES, mask_scale=scale)
    slow = brute(dark, bright, h, lut, tokens, target, scale)
    if not np.isclose(exact[0], slow[0], atol=1e-9):
        raise AssertionError(f"scale {scale}: RMSE {exact[0]} != brute {slow[0]}")
    if not np.isclose(exact[1], slow[1], atol=1e-9):
        raise AssertionError(f"scale {scale}: max {exact[1]} != brute {slow[1]}")
    if exact[2] != slow[2]:
        raise AssertionError(f"scale {scale}: roll {exact[2]} != brute {slow[2]}")


def main():
    h = fileio.parse_filter(os.path.join(ROOT, "Filters", f"{FB}_H.txt")).sets[0]
    v = fileio.parse_filter(os.path.join(
        ROOT, "Filters", f"{FB}_V Adaptive.txt"))
    lut = fileio.parse_gamma(os.path.join(ROOT, "Gamma", f"{FB}.txt"))
    tokens = [["50e", "20e"], ["10e", "40e"]]

    # A deliberately asymmetric 4x4 period exercises nontrivial LCM tiling and
    # gives the best-roll correlation a unique answer.
    yy, xx = np.indices((4, 4))
    target = np.empty((4, 4, 3), dtype=np.float64)
    target[..., 0] = 0.25 + 0.21 * ((2 * xx + yy) % 5)
    target[..., 1] = 0.18 + 0.17 * ((xx + 3 * yy) % 6)
    target[..., 2] = 0.12 + 0.13 * ((3 * xx + 2 * yy) % 7)
    check_scale(v.dark, v.bright, h, lut, tokens, target, 1)
    check_scale(v.dark, v.bright, h, lut, tokens, target, 2)

    # Crop-regression witness: the first shader cell matches a unity hardware
    # token, but the three cells outside the old 1x1 crop do not.  Full-period
    # scoring must expose those cells rather than returning the crop's result.
    crop_target = np.array([
        [[1.0, 1.0, 1.0], [0.15, 0.40, 0.75]],
        [[1.65, 0.25, 0.35], [0.30, 1.45, 0.20]],
    ])
    unity_tokens = [["700"]]
    full, _, _ = fitting._rmse_exact_masked_periodic(
        ref, v.dark, v.bright, h, lut, unity_tokens, crop_target,
        codes=CODES, align=False)

    # Independently score only the upper-left cell, reproducing the historical
    # bug's domain.  It should look materially better than the honest 2x2 cell.
    cropped, _, _ = fitting._rmse_exact_masked_periodic(
        ref, v.dark, v.bright, h, lut, unity_tokens,
        crop_target[:1, :1], codes=CODES, align=False)
    if not full > cropped + 5.0:
        raise AssertionError(
            f"crop regression witness too weak: full {full:.3f}, crop {cropped:.3f}")

    print("mask period scale 1: FFT == brute")
    print("mask period scale 2: FFT == brute")
    print(f"crop witness: cropped {cropped:.3f}, full-period {full:.3f}")
    print("ALL OK")


if __name__ == "__main__":
    main()
