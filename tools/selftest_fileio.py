"""Self-test: parse, validate and round-trip every existing v4 file in the pack.

The v4 files passed the original audit, so every one of them must parse clean,
validate clean, and survive write -> re-parse with identical data.
"""

import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fileio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []
counts = {"filter": 0, "gamma": 0, "mask": 0, "preset": 0}

with tempfile.TemporaryDirectory() as tmp:
    for path in sorted(glob.glob(os.path.join(ROOT, "Filters", "*.txt"))):
        name = os.path.basename(path)
        try:
            flt = fileio.parse_filter(path)
            failures += [f"{name}: {p}" for p in fileio.validate_filter(flt, name)]
            out = os.path.join(tmp, "f.txt")
            fileio.write_filter(out, flt)
            again = fileio.parse_filter(out)
            if len(again.sets) != len(flt.sets) or any(
                    not np.array_equal(a, b) for a, b in zip(flt.sets, again.sets)):
                failures.append(f"{name}: round-trip coefficient mismatch")
            if (again.adaptive, again.tenbit) != (flt.adaptive, flt.tenbit):
                failures.append(f"{name}: round-trip keyword mismatch")
            counts["filter"] += 1
        except Exception as e:
            failures.append(f"{name}: EXCEPTION {e}")

    for path in sorted(glob.glob(os.path.join(ROOT, "Gamma", "*.txt"))):
        name = os.path.basename(path)
        try:
            g = fileio.parse_gamma(path)
            failures += [f"{name}: {p}" for p in fileio.validate_gamma(g, name)]
            out = os.path.join(tmp, "g.txt")
            fileio.write_gamma(out, g)
            if not np.array_equal(fileio.parse_gamma(out), g):
                failures.append(f"{name}: round-trip mismatch")
            counts["gamma"] += 1
        except Exception as e:
            failures.append(f"{name}: EXCEPTION {e}")

    for path in sorted(glob.glob(os.path.join(ROOT, "Shadow_Masks", "*.txt"))):
        name = os.path.basename(path)
        try:
            m = fileio.parse_mask(path)
            failures += [f"{name}: {p}" for p in fileio.validate_mask(m, name)]
            out = os.path.join(tmp, "m.txt")
            fileio.write_mask(out, m)
            again = fileio.parse_mask(out)
            if again.tokens != m.tokens or (again.width, again.height) != (m.width, m.height):
                failures.append(f"{name}: round-trip mismatch")
            counts["mask"] += 1
        except Exception as e:
            failures.append(f"{name}: EXCEPTION {e}")

    for path in sorted(glob.glob(os.path.join(ROOT, "Presets", "*.ini"))):
        name = os.path.basename(path)
        try:
            p = fileio.parse_preset(path)
            failures += fileio.validate_preset(p, ROOT, name)
            out = os.path.join(tmp, "p.ini")
            fileio.write_preset(out, p)
            if fileio.parse_preset(out) != p:
                failures.append(f"{name}: round-trip mismatch")
            counts["preset"] += 1
        except Exception as e:
            failures.append(f"{name}: EXCEPTION {e}")

print(f"parsed: {counts}")
if failures:
    print(f"FAIL ({len(failures)}):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("ALL OK")
