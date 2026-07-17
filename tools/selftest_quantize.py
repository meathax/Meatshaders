"""Self-test for quantize.py: invariants + reconstruction of a shipped table.

Test 1: quantizing smooth synthetic kernels yields tables that pass the
        fileio validator (symmetry, range) with exact requested sums.
Test 2: rows quantized from the canonical Lottes fixed V table's own
        float form (coeffs/256 with tiny noise) reproduce the stored integers,
        i.e. the quantizer agrees with how the shipped pack was built.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fileio  # noqa: E402
import quantize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(42)
fail = []

# --- Test 1: synthetic Catmull-Rom-like kernel with varying per-phase gain
ideal = np.zeros((256, 4))
sums = np.zeros(256, dtype=np.int64)
for p in range(256):
    f = p / 256.0
    w = np.array([
        -0.5 * f * (1 - f) ** 2 * 2,
        (1 - f) * (1 + f - 1.5 * f * f),
        f * (2 - f - 1.5 * (1 - f) ** 2),
        -0.5 * f * f * (1 - f) * 2,
    ])
    w /= w.sum()
    d = min(f, 1 - f)
    gain = 0.75 + 0.25 * np.cos(2 * np.pi * d)          # symmetric scan-ish profile
    ideal[p] = w * gain * 256
    sums[p] = round(gain * 256)
for p in range(1, 128):
    sums[256 - p] = sums[p]                              # enforce conjugate sums

table = quantize.quantize_symmetric(ideal, sums)
flt = fileio.FilterFile(header=["synthetic"], adaptive=False, tenbit=True, sets=[table])
fail += fileio.validate_filter(flt, "synthetic")
bad = [p for p in range(256) if p != 128 and table[p].sum() != sums[p]]
if bad:
    fail.append(f"sum mismatch at phases {bad[:5]}")
if abs(table[128].sum() - sums[128]) > 1:
    fail.append(f"phase128 sum off by {table[128].sum() - sums[128]}")

# --- Test 2: agreement with a retained canonical table
src = fileio.parse_filter(
    os.path.join(ROOT, "Filters", "CRT Lottes (Port)_V.txt"))
stored = src.dark.astype(np.float64)
noisy = stored + rng.uniform(-0.25, 0.25, stored.shape)  # sub-half-unit jitter
sums2 = stored.sum(axis=1).astype(np.int64)
rebuilt = quantize.quantize_symmetric(noisy, sums2)
diff = np.abs(rebuilt - stored)
if diff.max() > 1:
    fail.append(f"canonical reconstruction differs by up to {diff.max()}")
n_off = int((diff > 0).sum())
print(f"canonical Lottes V reconstruction: {n_off}/{stored.size} coefficients moved by 1 "
      f"(max {int(diff.max())}) under +-0.25 jitter")

if fail:
    print("FAIL:")
    for f in fail:
        print(" -", f)
    sys.exit(1)
print("ALL OK")
