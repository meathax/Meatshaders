"""Guard the retired Easymode anti-ring profiles against fidelity regression.

MiSTer's mandatory 0..255 clamp already reproduces Easymode's local clamp for
binary neighbourhoods: equal inner samples bound the result to 0 or 255, while
different binary inner samples span the complete hardware range.  The former
binary least-squares table omitted that hardware clamp and was substantially
worse than the canonical kernel under exact arithmetic.

The old preset names remain as compatibility aliases.  This test makes that
choice explicit and scores both normal and late-boost paths against the true
source-clamped response.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "targets"))

import easymode_ref
import fileio


GAIN = 1.0677
BINARY = np.array(list(itertools.product((0, 255), repeat=4)), dtype=np.int64)
MULTILEVEL = np.array(
    list(itertools.product((0, 64, 128, 192, 255), repeat=4)), dtype=np.int64)


def _load(name: str) -> fileio.FilterFile:
    return fileio.parse_filter(os.path.join(ROOT, "Filters", name))


def score(table: np.ndarray, patterns: np.ndarray,
          gain: float = 1.0) -> tuple[float, float]:
    """Exact MiSTer pairwise-truncation RMSE/max over unique phases 0..128."""
    errors = []
    for phase in range(129):
        weights = np.array(
            [weight for _, weight in easymode_ref.h_kernel(phase / 256.0)])
        lower = np.minimum(patterns[:, 1], patterns[:, 2])
        upper = np.maximum(patterns[:, 1], patterns[:, 2])
        target = gain * np.clip(patterns @ weights, lower, upper)
        target = np.clip(target, 0.0, 255.0)

        row = table[phase].astype(np.int64)
        pair01 = (patterns[:, 0] * row[0] + patterns[:, 1] * row[1]) >> 1
        pair23 = (patterns[:, 2] * row[2] + patterns[:, 3] * row[3]) >> 1
        output = np.clip((pair01 + pair23) >> 7, 0, 255)
        errors.append(output - target)
    error = np.concatenate(errors)
    return float(np.sqrt(np.mean(np.square(error)))), float(np.abs(error).max())


def test_pixel_art_names_are_safe_canonical_aliases() -> None:
    canonical = _load("CRT Easymode (Port)_H.txt")
    pixel = _load("CRT Easymode Pixel-Art Anti-Ring (Port)_H.txt")
    canonical_boost = _load("CRT Easymode (Port)_H Matched Boost.txt")
    pixel_boost = _load("CRT Easymode Pixel-Art Anti-Ring Matched Boost (Port)_H.txt")

    np.testing.assert_array_equal(pixel.sets[0], canonical.sets[0])
    np.testing.assert_array_equal(pixel_boost.sets[0], canonical_boost.sets[0])
    assert any("compatibility alias" in line.lower() for line in pixel.header)
    assert any("compatibility alias" in line.lower() for line in pixel_boost.header)


def test_aliases_keep_hardware_exact_fidelity_bounds() -> None:
    normal = _load("CRT Easymode Pixel-Art Anti-Ring (Port)_H.txt").sets[0]
    boosted = _load(
        "CRT Easymode Pixel-Art Anti-Ring Matched Boost (Port)_H.txt").sets[0]

    binary = score(normal, BINARY)
    multilevel = score(normal, MULTILEVEL)
    boosted_binary = score(boosted, BINARY, GAIN)
    boosted_multilevel = score(boosted, MULTILEVEL, GAIN)
    assert binary[0] < 0.55 and binary[1] < 1.70
    assert multilevel[0] < 2.90 and multilevel[1] <= 24.0
    assert boosted_binary[0] < 0.56 and boosted_binary[1] < 1.95
    assert boosted_multilevel[0] < 3.10 and boosted_multilevel[1] < 26.34

    assert np.all(normal.sum(axis=1) == 256)
    boost_sums = boosted.sum(axis=1)
    assert np.all(boost_sums[np.arange(256) != 128] == 273)
    assert int(boost_sums[128]) == 274
    np.testing.assert_array_equal(normal[129:], normal[127:0:-1, ::-1])
    np.testing.assert_array_equal(boosted[129:], boosted[127:0:-1, ::-1])


if __name__ == "__main__":
    test_pixel_art_names_are_safe_canonical_aliases()
    test_aliases_keep_hardware_exact_fidelity_bounds()
    normal = _load("CRT Easymode Pixel-Art Anti-Ring (Port)_H.txt").sets[0]
    boosted = _load(
        "CRT Easymode Pixel-Art Anti-Ring Matched Boost (Port)_H.txt").sets[0]
    print("normal binary", score(normal, BINARY))
    print("normal multilevel", score(normal, MULTILEVEL))
    print("boosted binary", score(boosted, BINARY, GAIN))
    print("boosted multilevel", score(boosted, MULTILEVEL, GAIN))
    print("ALL OK")
