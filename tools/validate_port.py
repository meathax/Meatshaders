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
        "rgb_rmse": 6.50, "rgb_max": 36.0,
        "step_rmse": 43.0, "step_max": 78.0,
    },
    "guest": {
        "module": "guest_advanced_ref", "base": "CRT Guest Advanced (Port)",
        "vertical": "CRT Guest Advanced (Port)_V Adaptive.txt", "adaptive": True,
        "gamma": "CRT Guest Advanced (Port).txt", "no_scan": True,
        "rmse": 2.50, "max": 10.0, "brightness": 5.0, "selector": 16,
        "rgb_rmse": 6.25, "rgb_max": 24.0,
    },
    "lottes": {
        "module": "lottes_ref", "base": "CRT Lottes (Port)",
        "vertical": "CRT Lottes (Port)_V.txt", "adaptive": False,
        "gamma": None, "no_scan": False,
        # True piecewise-sRGB mask target, hard-gated over all 256 input codes.
        "rmse": 1.75, "max": 8.10, "brightness": 1.5, "selector": 0,
    },
    "royale": {
        "module": "royale_ref", "base": "CRT Royale (Port)",
        "vertical": "CRT Royale (Port)_V Adaptive.txt", "adaptive": True,
        "gamma": "CRT Royale (Port).txt", "no_scan": True,
        # The 1080p-integrated beam deliberately trades a small amount of
        # source-shape fidelity for a hard full-frame moire pass at 240p.
        "rmse": 19.05, "max": 78.0, "brightness": 4.0, "selector": 95,
    },
    "kurozumi": {
        "module": "kurozumi_ref", "base": "CRT Royale Kurozumi (Port)",
        "vertical": "CRT Royale Kurozumi (Port)_V Adaptive.txt", "adaptive": True,
        "gamma": "CRT Royale Kurozumi (Port).txt", "no_scan": True,
        # Historical matched Kuro Grade/preset plus the later upstream-fixed
        # Royale bloom path.  RGB is scored on the physical 1x2 grille cadence;
        # the neutral metric above still covers all 16x16 resized source cells.
        "rmse": 28.00, "max": 180.0, "brightness": 24.5, "selector": 0,
        "rgb_rmse": 34.0, "rgb_max": 178.0,
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
        target = fitting.masked_reference_tile(ref, fs, code, mask_encoded)
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


def easymode_rgb_oracle(ref, dark, bright, h, lut, mask) -> tuple[float, float]:
    """Exact RGB-field gate against Easymode's complete source pipeline."""
    levels = (0, 64, 128, 192, 255)
    fs = np.arange(0, 33) / 64.0
    m16 = np.round(mask.multipliers() * 16.0).astype(np.int64)
    errors = []
    for red in levels:
        for green in levels:
            for blue in levels:
                rgb = (red, green, blue)
                h_rgb = np.array([int(mm.fir_1d(
                    np.full(16, int(lut[code, channel]), dtype=np.int64),
                    h, np.array([8.0]))[0])
                    for channel, code in enumerate(rgb)], dtype=np.int64)
                ctrl = np.full(len(fs), int(h_rgb.max()), dtype=np.int64)
                simulated = np.stack([mm.fir_1d_adaptive(
                    np.full(16, int(h_rgb[channel]), dtype=np.int64),
                    dark, bright, 8.0 + fs, ctrl) for channel in range(3)], axis=1)
                port = mm.mask_multiply(simulated[None, None, :, :],
                                        m16[:, :, None, :])
                normalized = tuple(code / 255.0 for code in rgb)
                # Build in the same (mask_y, mask_x, phase, RGB) order as port.
                target = 255.0 * np.array([[[
                    ref.ref_uniform_rgb(float(f), normalized, px, py)
                    for f in fs] for px in range(mask.width)]
                    for py in range(mask.height)], dtype=np.float64)
                errors.append((port - target).ravel())
    error = np.concatenate(errors)
    return float(np.sqrt(np.mean(error * error))), float(np.abs(error).max())


def easymode_step_oracle(ref, dark, bright, h, lut, mask) -> tuple[float, float]:
    """Exact black/white horizontal-step gate, including source anti-ringing."""
    size, split = 16, 8
    source_codes = np.zeros((size, size, 3), dtype=np.int64)
    source_codes[:, split:] = 255
    source_ref = (source_codes.astype(np.float64) / 255.0).tolist()
    positions = split - 0.5 + np.arange(-8, 9, dtype=np.float64) / 8.0
    fs = np.arange(0, 33) / 64.0
    m16 = np.round(mask.multipliers() * 16.0).astype(np.int64)
    gamma = mm.apply_gamma(source_codes, lut)
    errors = []
    for x in positions:
        h_rgb = np.array([mm.fir_1d(gamma[8, :, channel], h,
                                    np.array([x]))[0]
                          for channel in range(3)], dtype=np.int64)
        ctrl = np.full(len(fs), int(h_rgb.max()), dtype=np.int64)
        simulated = np.stack([mm.fir_1d_adaptive(
            np.full(16, int(h_rgb[channel]), dtype=np.int64),
            dark, bright, 8.0 + fs, ctrl) for channel in range(3)], axis=1)
        port = mm.mask_multiply(simulated[None, None, :, :],
                                m16[:, :, None, :])
        target = 255.0 * np.array([[[
            ref.ref_source_rgb(source_ref, float(x), 8.0 + float(f), px, py)
            for f in fs] for px in range(mask.width)]
            for py in range(mask.height)], dtype=np.float64)
        errors.append((port - target).ravel())
    error = np.concatenate(errors)
    return float(np.sqrt(np.mean(error * error))), float(np.abs(error).max())


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
    masked_codes = (np.arange(256, dtype=np.int64)
                    if key == "lottes" else None)
    rmse, max_error = fitting.rmse_exact_masked(
        ref, dark, bright, h, lut, mask.tokens, target_mask,
        codes=masked_codes)
    strict_rmse, strict_max = fitting.rmse_exact_masked(
        ref, dark, bright, h, lut, mask.tokens, target_mask,
        codes=masked_codes, align=False)
    metric = fitting.rmse_exact_rgb if key == "kurozumi" else fitting.rmse_exact
    mask_off = metric(ref, dark, bright, h, lut)
    masked_label = "all-256-code" if key == "lottes" else "endpoint-inclusive"
    print(f"masked {masked_label} RMSE: {rmse:.3f} codes "
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

    if key in ("guest", "kurozumi"):
        rgb_kwargs = {}
        rgb_label = "9-level RGB cube"
        if key == "kurozumi":
            rgb_kwargs = {
                "levels": np.array([0, 64, 128, 192, 255]),
                "phases": np.arange(0, 9) / 16.0,
                "reference_period": (1, 2),
            }
            rgb_label = "historical 5-level RGB cube / projected 1x2 grille"
        rgb = fitting.rgb_masked_metrics(
            ref, dark, bright, h, lut, mask.tokens, **rgb_kwargs)
        print(f"exact masked {rgb_label}: RMSE {rgb['rms']:.3f}, "
              f"max {rgb['max']:.1f} codes ({rgb['colors']} colours, "
              f"{rgb['samples']} samples)")
        if rgb["rms"] > spec["rgb_rmse"] or rgb["max"] > spec["rgb_max"]:
            hard_fail.append(
                f"{base}: exact RGB regression {rgb['rms']:.3f}/"
                f"{rgb['max']:.1f} over {spec['rgb_rmse']:.2f}/"
                f"{spec['rgb_max']:.1f}")

    parity = brightness_parity(ref, dark, bright, h, lut, mask, target_mask)
    print("brightness parity (sim/reference/delta codes):")
    for x, simulated, reference, delta in parity:
        print(f"  {x:4}: {simulated:6.1f} / {reference:6.1f} / {delta:+5.1f}")
    worst_parity = max(abs(row[3]) for row in parity)
    if worst_parity > spec["brightness"]:
        hard_fail.append(
            f"{base}: brightness delta {worst_parity:.1f} over "
            f"{spec['brightness']:.1f}")

    if key == "easymode":
        rgb_rmse, rgb_max = easymode_rgb_oracle(
            ref, dark, bright, h, lut, mask)
        step_rmse, step_max = easymode_step_oracle(
            ref, dark, bright, h, lut, mask)
        print(f"exact RGB lattice: RMSE {rgb_rmse:.3f}, max {rgb_max:.1f} codes")
        print(f"exact B/W source step: RMSE {step_rmse:.3f}, "
              f"max {step_max:.1f} codes")
        if rgb_rmse > spec["rgb_rmse"] or rgb_max > spec["rgb_max"]:
            hard_fail.append(
                f"{base}: exact RGB regression {rgb_rmse:.3f}/{rgb_max:.1f}")
        if step_rmse > spec["step_rmse"] or step_max > spec["step_max"]:
            hard_fail.append(
                f"{base}: exact source-step regression "
                f"{step_rmse:.3f}/{step_max:.1f}")

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
        if black_rmse > 0.05 or black_max > 0.5 or np.any(lut[0] != 0):
            hard_fail.append(f"{base}: historical zero-black regression")

print("\n" + ("HARD FAILURES:\n" + "\n".join(hard_fail)
               if hard_fail else "ALL REQUIRED 1080P GATES PASS"))
sys.exit(1 if hard_fail else 0)
