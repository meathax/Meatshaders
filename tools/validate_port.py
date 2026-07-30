"""Validate the four-preset MiSTer pack and its colour-accuracy contract.

Every family is judged against the dominant palette colours from the supplied
unfiltered default.webp.  Uniform fields are evaluated with MiSTer's exact LUT,
FIR, clamp and v2-mask arithmetic at 240p -> 1080p.

The gated quantity is the PERCEIVED colour: the spatial mean taken in
light-linear space over a whole number of mask and scanline periods, converted
back to 8-bit codes.  The older code-space mean is printed alongside for
reference only -- gating on it is what allowed a 12-code perceived brightness
error to pass as "calibrated" (see tools/color_match.py for the derivation).

Highlights are reported separately.  MiSTer's v2 mask format caps non-selected
channels at 15/16 and the 8-bit stages provide no headroom above 255, so a
modulated field cannot average to code 255.  ``white`` is therefore a measured
ceiling, and ``RMSE excl. white`` isolates the error that calibration can
actually remove.

Usage: py -3 tools/validate_port.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color_match
import fileio
import mister_model as mm
import preset_contracts


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_LINES = 1080
SOURCE_HEIGHTS = (224, 240)
MOIRE_LIMIT = 0.03 * 255
WHITE = (255, 255, 255)

# Perceived-colour limits.  ``rmse``/``mae`` cover the whole palette; ``dark``
# is the RMSE over every sample that is not clipped by the highlight ceiling
# and is the real measure of source fidelity; ``white`` is the minimum
# acceptable perceived white.  Each limit sits just above the measured value so
# any regression trips the gate.
LIMITS = {
    "guest":    {"rmse": 8.0,  "mae": 5.0, "dark": 2.0, "white": 235.0},
    "lottes":   {"rmse": 13.5, "mae": 6.0, "dark": 1.0, "white": 217.0},
    "royale":   {"rmse": 6.0,  "mae": 3.0, "dark": 1.2, "white": 237.0},
    "kurozumi": {"rmse": 12.0, "mae": 5.5, "dark": 1.2, "white": 222.0},
}
# Neutral channel tint, in codes.  The floor here is NOT zero and is not a
# calibration defect: the horizontal FIR truncates twice per phase (F1), so a
# flat field returns a +/-1 code ripple across the scaler's 9 phases at 4.5x
# (flat 83 -> {82, 83}).  Where the mask period shares a factor with 9, each
# channel's selected columns land on a different residue class of those phases
# -- R on {0,3,6}, G on {1,4,7}, B on {2,5,8} -- whose means differ by up to
# 1/3 code.  Measured span: Guest 0.000 (width 2, coprime with 9, so every
# channel visits every phase), Lottes 0.104, Royale 0.165, Kurozumi 0.765.
# No gamma LUT can remove this; it is a truncation artefact of the phase/mask
# alignment.  At 0.3% of full scale it is far below a visible step.
NEUTRAL_SPAN_LIMIT = 1.0
# No non-clipped source colour may drift more than this from the source (codes).
PER_COLOUR_LIMIT = 6.0

# The 11-colour reference palette is too small to prove gamut-wide fidelity: it
# missed Royale's worst case entirely.  This grid walks the whole RGB cube at
# 9 levels per channel (729 colours).  ``drift`` bounds the largest per-channel
# error and ``tilt`` bounds the spread between a colour's three channel errors
# -- tilt is the one that reads as a hue shift rather than a brightness change.
CUBE_LEVELS = (0, 32, 64, 96, 128, 160, 192, 224, 255)
CUBE_LIMITS = {
    "guest":    {"drift": 6.0,  "tilt": 6.0,  "rmse": 2.5},
    "lottes":   {"drift": 1.5,  "tilt": 2.0,  "rmse": 0.6},
    # Royale keeps its full adaptive beam (contracting it breaks the moire
    # limit), so it carries the pack's largest shared-max-control hue tilt.
    "royale":   {"drift": 9.0,  "tilt": 11.0, "rmse": 2.0},
    "kurozumi": {"drift": 3.0,  "tilt": 3.5,  "rmse": 1.2},
}


def gamma_stats(lut: np.ndarray) -> dict[str, int]:
    steps = np.diff(lut[:, 0])
    longest = run = 1
    for same in steps == 0:
        run = run + 1 if same else 1
        longest = max(longest, run)
    return {
        "unique": int(len(np.unique(lut[:, 0]))),
        "longest_plateau": int(longest),
        "max_step": int(steps.max(initial=0)),
    }


def moire_1080(dark: np.ndarray, bright: np.ndarray,
               lut: np.ndarray) -> dict[int, float]:
    results = {}
    for source_lines in SOURCE_HEIGHTS:
        scale = OUTPUT_LINES / source_lines
        worst = 0.0
        for code in (64, 128, 192, 255):
            for channel in range(3):
                profile = mm.flat_field_profile(
                    code, scale, dark, bright, lut,
                    n_out=OUTPUT_LINES, channel=channel)
                metrics = mm.moire_metrics(profile, scale)
                worst = max(worst, metrics["peak_std"], metrics["trough_std"])
        results[source_lines] = worst
    return results


failures: list[str] = []
contract_problems = preset_contracts.validate_collection(ROOT)
failures += contract_problems
print("=== Canonical preset collection ===")
print("contracts:", "OK" if not contract_problems else contract_problems)

for key, spec in color_match.FAMILIES.items():
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
        formats.append(f"{base}: H must be one fixed table")
    expected_sets = 2 if spec["adaptive"] else 1
    if v_file.adaptive != spec["adaptive"] or len(v_file.sets) != expected_sets:
        formats.append(f"{base}: unexpected V adaptation/table count")
    failures += formats

    gamma_path = os.path.join(ROOT, "Gamma", spec["gamma"])
    lut = fileio.parse_gamma(gamma_path)
    failures += fileio.validate_gamma(lut, os.path.basename(gamma_path))
    stats = gamma_stats(lut)
    print("gamma:", stats)
    if not np.array_equal(lut[:, 0], lut[:, 1]) or not np.array_equal(
            lut[:, 0], lut[:, 2]):
        failures.append(f"{base}: calibrated LUT must be identical in RGB")
    # A coarse curve bands smooth gradients even when its mean colour is right.
    if stats["unique"] < 195 or stats["max_step"] > 4:
        failures.append(f"{base}: gamma quantization regression {stats}")

    mask_mean = mask.multipliers().mean(axis=(0, 1))
    print("mask mean RGB:", np.round(mask_mean, 6).tolist())
    if float(np.ptp(mask_mean)) > 1.0e-9:
        failures.append(f"{base}: unequal mask RGB energy {mask_mean.tolist()}")
    if not any(color_match.CALIBRATION_MARKER in line for line in mask.header):
        failures.append(f"{base}: colour-calibration marker missing from mask")

    if key == "kurozumi" and not any(
            marker in line for line in v_file.header
            for marker in (color_match.CALIBRATION_MARKER,
                           color_match.LEGACY_MARKER)):
        failures.append(f"{base}: calibrated scanline/DC blend missing")

    simulator = color_match.FieldSimulator(
        h_file.sets[0], v_file.dark, v_file.bright,
        bool(spec["adaptive"]), mask)
    perceived = np.array([simulator.perceived(rgb, lut)
                          for rgb in color_match.REFERENCE_PALETTE])
    code_space = np.array([simulator.code_mean(rgb, lut)
                           for rgb in color_match.REFERENCE_PALETTE])

    error = perceived - color_match.REFERENCE_PALETTE
    rmse = float(np.sqrt(np.mean(error * error)))
    mae = float(np.mean(np.abs(error)))
    white = float(perceived[list(map(tuple, color_match.REFERENCE_PALETTE)).index(WHITE)].mean())

    # Samples whose every channel sits below the measured highlight ceiling:
    # these are the ones calibration is fully responsible for.
    unclipped = [i for i, rgb in enumerate(color_match.REFERENCE_PALETTE)
                 if max(rgb) <= white]
    dark_rmse = float(np.sqrt(np.mean(error[unclipped] ** 2)))
    neutral = [value for rgb, value in zip(color_match.REFERENCE_PALETTE, perceived)
               if rgb[0] == rgb[1] == rgb[2]]
    neutral_span = max(float(np.ptp(value)) for value in neutral)

    print(f"perceived: RMSE {rmse:.3f}, MAE {mae:.3f}, "
          f"RMSE excl. clipped {dark_rmse:.3f}, neutral span {neutral_span:.3f}")
    print(f"code-space RMSE {float(np.sqrt(np.mean((code_space - color_match.REFERENCE_PALETTE) ** 2))):.3f} "
          "(reference only, not gated)")
    print(f"white ceiling {white:.1f} of 255")
    print("gray172 ->", np.round(perceived[1], 1).tolist(),
          "brown ->", np.round(perceived[2], 1).tolist())

    limit = LIMITS[key]
    if rmse > limit["rmse"]:
        failures.append(f"{base}: perceived RMSE {rmse:.3f} > {limit['rmse']}")
    if mae > limit["mae"]:
        failures.append(f"{base}: perceived MAE {mae:.3f} > {limit['mae']}")
    if dark_rmse > limit["dark"]:
        failures.append(
            f"{base}: unclipped RMSE {dark_rmse:.3f} > {limit['dark']}")
    if neutral_span > NEUTRAL_SPAN_LIMIT:
        failures.append(f"{base}: neutral channel span {neutral_span:.3f} "
                        f"> {NEUTRAL_SPAN_LIMIT}")
    if white < limit["white"]:
        failures.append(f"{base}: white ceiling {white:.1f} < {limit['white']}")
    if np.max(np.abs(perceived[0])) > 0.5:
        failures.append(f"{base}: black level is not zero {perceived[0].tolist()}")
    for index in unclipped:
        drift = float(np.max(np.abs(error[index])))
        if drift > PER_COLOUR_LIMIT:
            source = tuple(int(v) for v in color_match.REFERENCE_PALETTE[index])
            failures.append(
                f"{base}: {source} drifts {drift:.1f} codes "
                f"(> {PER_COLOUR_LIMIT}) -> {np.round(perceived[index], 1).tolist()}")

    cube_error, cube_source = [], []
    for red in CUBE_LEVELS:
        for green in CUBE_LEVELS:
            for blue in CUBE_LEVELS:
                rgb = (red, green, blue)
                cube_error.append(simulator.perceived(rgb, lut) - np.array(rgb))
                cube_source.append(rgb)
    cube_error = np.array(cube_error)
    cube_source = np.array(cube_source)
    unclipped_cube = cube_source.max(axis=1) <= white
    cube_drift = float(np.abs(cube_error[unclipped_cube]).max())
    cube_tilt = float(np.max(cube_error[unclipped_cube].max(axis=1)
                             - cube_error[unclipped_cube].min(axis=1)))
    cube_rmse = float(np.sqrt(np.mean(cube_error[unclipped_cube] ** 2)))
    worst = int(np.argmax(np.abs(cube_error[unclipped_cube]).max(axis=1)))
    worst_source = cube_source[unclipped_cube][worst]
    print(f"RGB cube ({len(cube_source)} colours, "
          f"{int((~unclipped_cube).sum())} clipped by the ceiling): "
          f"unclipped RMSE {cube_rmse:.2f}, max drift {cube_drift:.1f}, "
          f"max hue tilt {cube_tilt:.1f}")
    print(f"  worst: {tuple(int(v) for v in worst_source)} -> "
          f"{np.round(cube_error[unclipped_cube][worst] + worst_source, 1).tolist()}")

    cube_limit = CUBE_LIMITS[key]
    if cube_drift > cube_limit["drift"]:
        failures.append(f"{base}: cube max drift {cube_drift:.1f} "
                        f"> {cube_limit['drift']}")
    if cube_tilt > cube_limit["tilt"]:
        failures.append(f"{base}: cube max hue tilt {cube_tilt:.1f} "
                        f"> {cube_limit['tilt']}")
    if cube_rmse > cube_limit["rmse"]:
        failures.append(f"{base}: cube unclipped RMSE {cube_rmse:.2f} "
                        f"> {cube_limit['rmse']}")

    moire = moire_1080(v_file.dark, v_file.bright, lut)
    print("1080p moire:", ", ".join(
        f"{source}p {value:.2f}" for source, value in moire.items()))
    if max(moire.values()) > MOIRE_LIMIT:
        failures.append(
            f"{base}: 1080p moire {max(moire.values()):.2f} > {MOIRE_LIMIT:.2f}")

print("\n" + ("HARD FAILURES:\n" + "\n".join(failures)
               if failures else "ALL COLOUR-ACCURACY GATES PASS"))
raise SystemExit(1 if failures else 0)
