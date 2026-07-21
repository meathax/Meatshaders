"""Acceptance validation for the canonical 1080p MiSTer CRT preset pack.

The public runtime surface is intentionally tiny: one landscape preset and one
TATE preset for each of five shaders.  This validator hard-gates their exact
MiSTer arithmetic, 1080p mask pipeline, full-period error, clipping, brightness,
and fractional-scale stability for common 224p and 240p sources.

Usage: py -3 tools/validate_port.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fileio
import fitting
import mister_model as mm
import preset_contracts


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = os.environ.get(
    "MISTER_SHADER_REFERENCES", os.path.join(os.path.dirname(__file__), "targets"))
if not os.path.isdir(TARGETS):
    sys.exit("Reference models not found. Set MISTER_SHADER_REFERENCES to the "
             "directory containing *_ref.py modules.")
sys.path.insert(0, TARGETS)

OUTPUT_LINES = 1080
SOURCE_HEIGHTS = (224, 240)
MOIRE_LIMIT = 0.03 * 255

FAMILIES = {
    "easymode": {
        "module": "easymode_ref", "base": "CRT Easymode (Port)",
        "vertical": "CRT Easymode (Port)_V Adaptive.txt", "adaptive": True,
        "gamma": "CRT Easymode (Port).txt", "no_scan": True,
        "rmse": 3.80, "max": 20.0, "brightness": 12.5, "selector": 42,
    },
    "guest": {
        "module": "guest_advanced_ref", "base": "CRT Guest Advanced (Port)",
        "vertical": "CRT Guest Advanced (Port)_V Adaptive.txt", "adaptive": True,
        "gamma": "CRT Guest Advanced (Port).txt", "no_scan": True,
        "rmse": 1.30, "max": 8.0, "brightness": 5.0, "selector": 25,
    },
    "guest_fast": {
        "module": "guest_fast_ref", "base": "CRT Guest Advanced Fast (Port)",
        "vertical": "CRT Guest Advanced Fast (Port)_V Adaptive.txt",
        "adaptive": True,
        "gamma": "CRT Guest Advanced Fast (Port).txt", "no_scan": True,
        "rmse": 1.30, "max": 8.0, "brightness": 5.0, "selector": 25,
    },
    "guest_fastest": {
        "module": "guest_fastest_ref",
        "base": "CRT Guest Advanced Fastest (Port)",
        "vertical": "CRT Guest Advanced Fastest (Port)_V Adaptive.txt",
        "adaptive": True,
        "gamma": "CRT Guest Advanced Fastest (Port).txt", "no_scan": True,
        "rmse": 1.30, "max": 8.0, "brightness": 5.0, "selector": 25,
    },
    "lottes": {
        "module": "lottes_ref", "base": "CRT Lottes (Port)",
        "vertical": "CRT Lottes (Port)_V.txt", "adaptive": False,
        "gamma": None, "no_scan": False,
        "rmse": 1.30, "max": 6.0, "brightness": 1.5, "selector": 0,
    },
    "royale": {
        "module": "royale_ref", "base": "CRT Royale (Port)",
        "vertical": "CRT Royale (Port)_V Adaptive.txt", "adaptive": True,
        "gamma": "CRT Royale (Port).txt", "no_scan": True,
        # The 1080p-integrated beam deliberately trades a small amount of
        # source-shape fidelity for a hard full-frame moire pass at 240p.
        "rmse": 20.10, "max": 90.0, "brightness": 4.0, "selector": 115,
    },
    "kurozumi": {
        "module": "kurozumi_ref", "base": "CRT Royale Kurozumi (Port)",
        "vertical": "CRT Royale Kurozumi (Port)_V Adaptive.txt", "adaptive": True,
        "gamma": "CRT Royale Kurozumi (Port).txt", "no_scan": True,
        "rmse": 20.50, "max": 125.0, "brightness": 12.0, "selector": 2,
    },
}


def identity_lut() -> np.ndarray:
    return np.repeat(np.arange(256, dtype=np.int64)[:, None], 3, axis=1)


def brightness_parity(ref, dark, bright, h, lut, mask, mask_encoded):
    """Mean exact masked output versus the shader, including final clipping."""
    rows = []
    mask_mult16 = np.round(mask.multipliers() * 16).astype(np.int64)
    fs = np.arange(0, 33) / 64.0
    for x in (0.25, 0.5, 0.75, 1.0):
        code = int(round(x * 255))
        sim_profile = fitting.simulate_flat_rgb(dark, bright, h, lut, code, fs)
        masked = mm.mask_multiply(
            sim_profile[None, None, :, :], mask_mult16[:, :, None, :])
        sim = float(masked.mean())
        beam = np.array([[
            fitting._ref_vertical_unclipped(ref, f, code / 255.0, ch)
            for ch in "rgb"] for f in fs])
        target = 255.0 * np.minimum(
            mask_encoded[:, :, None, :] * beam[None, None, :, :], 1.0)
        ref_masked = float(target.mean())
        rows.append((x, sim, ref_masked, sim - ref_masked))
    return rows


def moire_1080(dark, bright, lut) -> dict[int, float]:
    """Full-frame peak/trough instability at exact 1080p scale ratios."""
    results = {}
    for source_lines in SOURCE_HEIGHTS:
        scale = OUTPUT_LINES / source_lines
        worst = 0.0
        for x in (0.25, 0.5, 0.75, 1.0):
            code = int(round(x * 255))
            for channel in range(3):
                profile = mm.flat_field_profile(
                    code, scale, dark, bright, lut, n_out=OUTPUT_LINES,
                    channel=channel)
                metrics = mm.moire_metrics(profile, scale)
                worst = max(worst, metrics["peak_std"], metrics["trough_std"])
        results[source_lines] = worst
    return results


def gamma_stats(lut: np.ndarray) -> dict[str, int]:
    steps = np.diff(lut, axis=0)
    unique = min(len(np.unique(lut[:, channel])) for channel in range(3))
    longest = run = 1
    for same in np.all(steps == 0, axis=1):
        run = run + 1 if same else 1
        longest = max(longest, run)
    return {
        "unique": int(unique),
        "longest_plateau": int(longest),
        "max_step": int(steps.max(initial=0)),
    }


def selector_control_jump(h, dark, bright, lut) -> int:
    """Worst discontinuity caused solely by the nearest-line control switch."""
    cases = (
        ((64, 64, 64), (255, 255, 255)),
        ((0, 0, 0), (255, 255, 255)),
        ((255, 0, 0), (0, 0, 255)),
        ((255, 64, 64), (64, 64, 255)),
    )

    def h_rgb(rgb):
        values = []
        for channel, code in enumerate(rgb):
            gamma = int(lut[code, channel])
            values.append(int(mm.fir_1d(
                np.full(16, gamma, dtype=np.int64), h, np.array([8.0]))[0]))
        return np.asarray(values, dtype=np.int64)

    worst = 0
    position = np.array([8.5], dtype=np.float64)
    for upper_rgb, lower_rgb in cases:
        upper, lower = h_rgb(upper_rgb), h_rgb(lower_rgb)
        lines = np.empty((16, 3), dtype=np.int64)
        lines[:9], lines[9:] = upper, lower
        before = np.empty(3, dtype=np.int64)
        after = np.empty(3, dtype=np.int64)
        for channel in range(3):
            before[channel] = mm.fir_1d_adaptive(
                lines[:, channel], dark, bright, position,
                np.array([upper.max()], dtype=np.int64))[0]
            after[channel] = mm.fir_1d_adaptive(
                lines[:, channel], dark, bright, position,
                np.array([lower.max()], dtype=np.int64))[0]
        worst = max(worst, int(np.abs(after - before).max()))
    return worst


hard_fail: list[str] = []
preset_problems = preset_contracts.validate_collection(ROOT)
print("=== Canonical preset collection ===")
print(f"contracts: {'OK' if not preset_problems else preset_problems}")
hard_fail += preset_problems

for key, spec in FAMILIES.items():
    ref = __import__(spec["module"])
    base = spec["base"]
    print(f"\n=== {base} ===")

    h_path = os.path.join(ROOT, "Filters", f"{base}_H.txt")
    v_path = os.path.join(ROOT, "Filters", spec["vertical"])
    mask_path = os.path.join(ROOT, "Shadow_Masks", f"{base}.txt")
    h_file = fileio.parse_filter(h_path)
    v_file = fileio.parse_filter(v_path)
    mask = fileio.parse_mask(mask_path)
    formats = []
    formats += fileio.validate_filter(h_file, os.path.basename(h_path))
    formats += fileio.validate_filter(v_file, os.path.basename(v_path))
    formats += fileio.validate_mask(mask, os.path.basename(mask_path))
    if h_file.adaptive or len(h_file.sets) != 1:
        formats.append(f"{base}: H must contain one fixed coefficient set")
    expected_sets = 2 if spec["adaptive"] else 1
    if v_file.adaptive != spec["adaptive"] or len(v_file.sets) != expected_sets:
        mode = "two adaptive" if spec["adaptive"] else "one fixed"
        formats.append(f"{base}: V must contain {mode} coefficient set(s)")
    if spec["no_scan"]:
        no_scan_path = os.path.join(
            ROOT, "Filters", f"{base}_V No Scanlines.txt")
        no_scan = fileio.parse_filter(no_scan_path)
        formats += fileio.validate_filter(no_scan, os.path.basename(no_scan_path))
        if no_scan.adaptive or len(no_scan.sets) != 1:
            formats.append(f"{base}: interlace fallback must be fixed")

    if spec["gamma"] is None:
        lut = identity_lut()
    else:
        gamma_path = os.path.join(ROOT, "Gamma", spec["gamma"])
        lut = fileio.parse_gamma(gamma_path)
        formats += fileio.validate_gamma(lut, os.path.basename(gamma_path))
        stats = gamma_stats(lut)
        print("gamma quantization: "
              f"{stats['unique']} minimum levels, plateau "
              f"{stats['longest_plateau']}, max step {stats['max_step']}")
        if stats["unique"] < 128 or stats["longest_plateau"] > 16 \
                or stats["max_step"] > 8:
            formats.append(f"{base}: gamma quantization regression {stats}")

    print(f"formats: {'OK' if not formats else formats}")
    hard_fail += formats
    h = h_file.sets[0]
    dark, bright = v_file.dark, v_file.bright
    maximum_coefficient = max(int(np.abs(table).max())
                              for table in (h, dark, bright))
    if maximum_coefficient > 511:
        hard_fail.append(f"{base}: signed coefficient overflow {maximum_coefficient}")

    flat = fitting.worst_flat_field_output(lut, dark, bright)
    print(f"clipping: worst flat field {flat:.1f}/255; "
          f"max |coefficient| {maximum_coefficient}/511")
    if flat > 255.5:
        hard_fail.append(f"{base}: flat field clips ({flat:.1f})")

    target_mask = fitting.mask_encoded_tile(ref)
    rmse, max_error = fitting.rmse_exact_masked(
        ref, dark, bright, h, lut, mask.tokens, target_mask)
    strict_rmse, strict_max = fitting.rmse_exact_masked(
        ref, dark, bright, h, lut, mask.tokens, target_mask, align=False)
    metric = fitting.rmse_exact_rgb if key == "kurozumi" else fitting.rmse_exact
    mask_off = metric(ref, dark, bright, h, lut)
    print(f"masked endpoint-inclusive RMSE: {rmse:.3f} codes "
          f"(max {max_error:.1f}); strict origin {strict_rmse:.3f} "
          f"(max {strict_max:.1f}); mask off {mask_off:.3f}")
    if rmse > spec["rmse"]:
        hard_fail.append(
            f"{base}: masked RMSE {rmse:.3f} over {spec['rmse']:.3f}")
    if max_error > spec["max"]:
        hard_fail.append(
            f"{base}: max error {max_error:.1f} over {spec['max']:.1f}")
    if strict_rmse > rmse + 0.05:
        hard_fail.append(
            f"{base}: mask origin mismatch ({strict_rmse:.3f} vs {rmse:.3f})")

    selector = selector_control_jump(h, dark, bright, lut)
    print(f"nearest-line selector-only jump: {selector} codes")
    if selector > spec["selector"]:
        hard_fail.append(
            f"{base}: selector jump {selector} over {spec['selector']}")

    parity = brightness_parity(ref, dark, bright, h, lut, mask, target_mask)
    print("brightness parity (sim/reference/delta codes):")
    for x, simulated, reference, delta in parity:
        print(f"  {x:4}: {simulated:6.1f} / {reference:6.1f} / {delta:+5.1f}")
    worst_parity = max(abs(row[3]) for row in parity)
    if worst_parity > spec["brightness"]:
        hard_fail.append(
            f"{base}: brightness delta {worst_parity:.1f} over "
            f"{spec['brightness']:.1f}")

    moire = moire_1080(dark, bright, lut)
    print("1080p moire: " + ", ".join(
        f"{source}p->{OUTPUT_LINES}p {value:.2f}"
        for source, value in moire.items()))
    if max(moire.values()) > MOIRE_LIMIT:
        hard_fail.append(
            f"{base}: 1080p moire {max(moire.values()):.2f} over {MOIRE_LIMIT:.2f}")

    if key == "kurozumi":
        black_rmse, black_max = fitting.rmse_exact_masked(
            ref, dark, bright, h, lut, mask.tokens, target_mask,
            codes=np.array([0]))
        print(f"code-0 colour gate: RMSE {black_rmse:.3f}, max {black_max:.1f}, "
              f"LUT {lut[0].tolist()}")
        if black_rmse > 1.5 or black_max > 5.0 or \
                not int(lut[0, 0]) > int(lut[0, 1]):
            hard_fail.append(f"{base}: near-black colour regression")

print("\n" + ("HARD FAILURES:\n" + "\n".join(hard_fail)
               if hard_fail else "ALL REQUIRED 1080P GATES PASS"))
sys.exit(1 if hard_fail else 0)
