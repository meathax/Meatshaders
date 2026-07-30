"""Self-test: parse, validate and round-trip every data file in the pack.

Every shipped file must parse cleanly, validate cleanly, and survive
write -> re-parse with identical data. Negative fixtures cover format hazards
that MiSTer otherwise accepts silently or interprets differently by field.
"""

import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fileio  # noqa: E402
import preset_contracts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []
counts = {"filter": 0, "gamma": 0, "mask": 0, "preset": 0}
EXPECTED_COUNTS = {"filter": 11, "gamma": 4, "mask": 4, "preset": 8}

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

    # Regression checks for malformed files and Main's field-specific preset
    # sentinels. These deliberately exercise failures, not shipped data.
    sample_path = sorted(glob.glob(os.path.join(ROOT, "Presets", "*.ini")))[0]
    sample = fileio.parse_preset(sample_path)

    missing = dict(sample)
    del missing["mask"]
    if not any("missing required key 'mask'" in p
               for p in fileio.validate_preset(missing, ROOT, "missing.ini")):
        failures.append("negative preset test: missing key was accepted")

    bad_mode = dict(sample, maskmode="3x")
    if not any("unsupported maskmode" in p
               for p in fileio.validate_preset(bad_mode, ROOT, "mode.ini")):
        failures.append("negative preset test: invalid maskmode was accepted")

    for key, value in (("gamma", "same"), ("mask", "same"), ("ifilter", "none")):
        bad_sentinel = dict(sample, **{key: value})
        if not any("not a valid sentinel" in p
                   for p in fileio.validate_preset(bad_sentinel, ROOT, "sentinel.ini")):
            failures.append(f"negative preset test: {key}={value} was accepted")

    oversized = fileio.MaskFile([], 17, 1, [["000"] * 17])
    if not any("outside hardware range" in p for p in fileio.validate_mask(oversized)):
        failures.append("negative mask test: 17-column mask was accepted")

    duplicate_path = os.path.join(tmp, "duplicate.ini")
    with open(duplicate_path, "w", encoding="utf-8") as f:
        f.write("gamma=off\ngamma=none\n")
    try:
        fileio.parse_preset(duplicate_path)
        failures.append("negative preset test: duplicate key was accepted")
    except ValueError:
        pass

    late_marker_path = os.path.join(tmp, "late-marker.txt")
    with open(late_marker_path, "w", encoding="utf-8") as f:
        f.write("0,256,0,0\n10bit\n")
    try:
        fileio.parse_filter(late_marker_path)
        failures.append("negative filter test: late 10bit marker was accepted")
    except ValueError:
        pass

if counts != EXPECTED_COUNTS:
    failures.append(f"inventory mismatch: expected {EXPECTED_COUNTS}, got {counts}")
for folder, expected_names in preset_contracts.PACK_MANIFEST.items():
    folder_path = os.path.join(ROOT, folder)
    actual_names = {
        entry.name for entry in os.scandir(folder_path) if entry.is_file()
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        failures.append(
            f"{folder}/ manifest mismatch: missing={missing}, extra={extra}")

print(f"parsed: {counts}")
if failures:
    print(f"FAIL ({len(failures)}):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("ALL OK")
