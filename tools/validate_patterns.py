"""Deterministic nonuniform-pattern diagnostics for the MiSTer shader ports.

This validator complements ``validate_port.py``'s flat-field/reference gates.
It deliberately probes effects that a uniform gray ramp cannot expose:

* the adaptive nearest-line control switch between phases 127 and 128;
* shared-max-RGB control versus the per-channel control implied by the scalar
  table fit; and
* exact pre-mask impulse, step, and checkerboard responses.

Quality ceilings are REPORT thresholds by default: they print warnings but do
not fail the suite while the pattern-aware tables are being stabilized.  File
format, coefficient-range, arithmetic-range, and determinism failures remain
hard failures.  Pass ``--strict`` to promote report warnings to failures.

The public simulation functions are intentionally independent of ``fitting``
and can be imported by future model-in-loop fitting experiments.

Usage:
    py -3 tools/validate_patterns.py
    py -3 tools/validate_patterns.py --json
    py -3 tools/validate_patterns.py --strict
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fileio
import mister_model as mm


LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
PHASE_127 = 127.0 / 256.0
PHASE_128 = 128.0 / 256.0


@dataclass(frozen=True)
class FamilySpec:
    key: str
    file_base: str
    gamma_name: str | None
    vertical_name: str
    adaptive: bool = True


@dataclass(frozen=True)
class LoadedFamily:
    spec: FamilySpec
    h: np.ndarray
    dark: np.ndarray
    bright: np.ndarray
    lut: np.ndarray


FAMILIES = (
    FamilySpec("guest", "CRT Guest Advanced (Port)",
               "CRT Guest Advanced (Port).txt",
               "CRT Guest Advanced (Port)_V Adaptive.txt"),
    FamilySpec("guest_fast", "CRT Guest Advanced Fast (Port)",
               "CRT Guest Advanced Fast (Port).txt",
               "CRT Guest Advanced Fast (Port)_V Adaptive.txt"),
    FamilySpec("guest_fastest", "CRT Guest Advanced Fastest (Port)",
               "CRT Guest Advanced Fastest (Port).txt",
               "CRT Guest Advanced Fastest (Port)_V Adaptive.txt"),
    FamilySpec("royale", "CRT Royale (Port)",
               "CRT Royale (Port).txt",
               "CRT Royale (Port)_V Adaptive.txt"),
    FamilySpec("kurozumi", "CRT Royale Kurozumi (Port)",
               "CRT Royale Kurozumi (Port).txt",
               "CRT Royale Kurozumi (Port)_V Adaptive.txt"),
    FamilySpec("easymode", "CRT Easymode (Port)",
               "CRT Easymode (Port).txt",
               "CRT Easymode (Port)_V Adaptive.txt"),
    # Lottes has no LUT in the pack.  Its presets use the input codes directly.
    FamilySpec("lottes", "CRT Lottes (Port)", None,
               "CRT Lottes (Port)_V.txt", adaptive=False),
)


# These are intentionally conservative first-pass report ceilings, not claims
# that values just below them are perceptually perfect.  --strict exists for
# the point at which regenerated tables have stable pattern baselines.
REPORT_LIMITS = {
    "selector_control_jump": 16.0,
    "shared_max_delta": 12.0,
    "impulse_centroid_error": 0.10,
    "step_backtrack": 1.0,
    "checker_chroma_span": 2.0,
}


HARD_LIMITS = {
    "coefficient_abs": 511,
    "output_code_min": 0,
    "output_code_max": 255,
}


SELECTOR_CASES = (
    ("gray64_to_white", (64, 64, 64), (255, 255, 255)),
    ("black_to_white", (0, 0, 0), (255, 255, 255)),
    ("red_to_blue", (255, 0, 0), (0, 0, 255)),
    ("warm_to_cool", (255, 64, 64), (64, 64, 255)),
)


SATURATED_COLORS = (
    (255, 64, 64),
    (255, 128, 0),
    (64, 64, 255),
    (255, 255, 0),
    (192, 64, 32),
)


SATURATED_PHASES = np.array([0.125, 0.25, 0.375, 0.5], dtype=np.float64)


def identity_lut() -> np.ndarray:
    """Return a three-channel 8-bit identity LUT."""
    return np.repeat(np.arange(256, dtype=np.int64)[:, None], 3, axis=1)


def load_family(root: str, spec: FamilySpec) -> tuple[LoadedFamily, list[str]]:
    """Load one family and return it with any hard format/range problems."""
    h_path = os.path.join(root, "Filters", f"{spec.file_base}_H.txt")
    v_path = os.path.join(root, "Filters", spec.vertical_name)
    h_file = fileio.parse_filter(h_path)
    v_file = fileio.parse_filter(v_path)
    problems = fileio.validate_filter(h_file, os.path.relpath(h_path, root))
    problems += fileio.validate_filter(v_file, os.path.relpath(v_path, root))
    expected_sets = 2 if spec.adaptive else 1
    if v_file.adaptive != spec.adaptive or len(v_file.sets) != expected_sets:
        mode = "two-set adaptive" if spec.adaptive else "single-set fixed"
        raise ValueError(f"{spec.key}: expected a {mode} V filter")

    if spec.gamma_name is None:
        lut = identity_lut()
    else:
        gamma_path = os.path.join(root, "Gamma", spec.gamma_name)
        lut = fileio.parse_gamma(gamma_path)
        problems += fileio.validate_gamma(lut, os.path.relpath(gamma_path, root))

    h = h_file.sets[0].astype(np.int64)
    dark = v_file.sets[0].astype(np.int64)
    # Keep one loaded shape for the diagnostics, but do not run a fixed table
    # through the adaptive coefficient interpolator: its numeric scaling is a
    # different hardware format.
    bright = (v_file.sets[1] if spec.adaptive else v_file.sets[0]).astype(np.int64)
    max_coef = max(int(np.abs(x).max()) for x in (h, dark, bright))
    if max_coef > HARD_LIMITS["coefficient_abs"]:
        problems.append(f"{spec.key}: signed 10-bit coefficient overflow {max_coef}")
    return LoadedFamily(spec, h, dark, bright, lut.astype(np.int64)), problems


def horizontal_uniform_rgb(family: LoadedFamily,
                           rgb: Iterable[int]) -> np.ndarray:
    """Exact H-stage result for one horizontally uniform encoded RGB row."""
    rgb_arr = np.asarray(tuple(rgb), dtype=np.int64)
    if rgb_arr.shape != (3,) or np.any((rgb_arr < 0) | (rgb_arr > 255)):
        raise ValueError("rgb must contain exactly three codes in 0..255")
    gamma = np.array([family.lut[rgb_arr[c], c] for c in range(3)],
                     dtype=np.int64)
    out = np.empty(3, dtype=np.int64)
    for c in range(3):
        line = np.full(16, int(gamma[c]), dtype=np.int64)
        out[c] = mm.fir_1d(line, family.h, np.array([8.0]))[0]
    return out


def simulate_pipeline(source_rgb: np.ndarray, family: LoadedFamily,
                      x_positions: np.ndarray, y_positions: np.ndarray,
                      control_mode: str = "shared_max") -> np.ndarray:
    """Run an RGB source image through exact gamma, H, and adaptive V stages.

    The result is pre-mask and has shape ``(len(y_positions),
    len(x_positions), 3)``.  ``control_mode='shared_max'`` is current MiSTer
    behavior.  ``'per_channel'`` is a diagnostic counterfactual that feeds
    each channel its own nearest-line value; it is useful because Royale and
    related source shaders shape their beams independently per channel.
    """
    source = np.asarray(source_rgb, dtype=np.int64)
    xpos = np.asarray(x_positions, dtype=np.float64)
    ypos = np.asarray(y_positions, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source_rgb must have shape (height, width, 3)")
    if np.any((source < 0) | (source > 255)):
        raise ValueError("source codes must be in 0..255")
    if control_mode not in ("shared_max", "per_channel"):
        raise ValueError("control_mode must be shared_max or per_channel")

    gamma = mm.apply_gamma(source, family.lut)
    src_h = source.shape[0]
    hstage = np.empty((src_h, len(xpos), 3), dtype=np.int64)
    for yy in range(src_h):
        for cc in range(3):
            hstage[yy, :, cc] = mm.fir_1d(
                gamma[yy, :, cc], family.h, xpos)

    base = np.floor(ypos).astype(np.int64)
    frac = ypos - base
    nearest = np.clip(base + (frac >= 0.5).astype(np.int64), 0, src_h - 1)
    out = np.empty((len(ypos), len(xpos), 3), dtype=np.int64)
    for xx in range(len(xpos)):
        nearest_rgb = hstage[nearest, xx, :]
        shared_ctrl = nearest_rgb.max(axis=1)
        for cc in range(3):
            if family.spec.adaptive:
                ctrl = (shared_ctrl if control_mode == "shared_max"
                        else nearest_rgb[:, cc])
                out[:, xx, cc] = mm.fir_1d_adaptive(
                    hstage[:, xx, cc], family.dark, family.bright, ypos, ctrl)
            else:
                out[:, xx, cc] = mm.fir_1d(
                    hstage[:, xx, cc], family.dark, ypos)
    return out


def selector_boundary_response(family: LoadedFamily,
                               upper_rgb: Iterable[int],
                               lower_rgb: Iterable[int],
                               base: int = 8) -> dict:
    """Measure phase motion and the nearest-line control jump separately."""
    upper = horizontal_uniform_rgb(family, upper_rgb)
    lower = horizontal_uniform_rgb(family, lower_rgb)
    lines = np.empty((16, 3), dtype=np.int64)
    lines[:base + 1] = upper
    lines[base + 1:] = lower
    ctrl_upper = int(upper.max())
    ctrl_lower = int(lower.max())
    p127 = np.array([base + PHASE_127], dtype=np.float64)
    p128 = np.array([base + PHASE_128], dtype=np.float64)

    def run(pos: np.ndarray, ctrl: int) -> np.ndarray:
        result = np.empty(3, dtype=np.int64)
        for cc in range(3):
            if family.spec.adaptive:
                result[cc] = mm.fir_1d_adaptive(
                    lines[:, cc], family.dark, family.bright, pos,
                    np.array([ctrl], dtype=np.int64))[0]
            else:
                result[cc] = mm.fir_1d(
                    lines[:, cc], family.dark, pos)[0]
        return result

    before = run(p127, ctrl_upper)
    phase_only_after = run(p128, ctrl_upper)
    actual_after = run(p128, ctrl_lower)
    phase_delta = phase_only_after - before
    selector_delta = actual_after - phase_only_after
    actual_delta = actual_after - before
    return {
        "upper_h": upper.tolist(),
        "lower_h": lower.tolist(),
        "controls": [ctrl_upper, ctrl_lower],
        "phase127": before.tolist(),
        "phase128": actual_after.tolist(),
        "phase_only_delta": phase_delta.tolist(),
        "selector_delta": selector_delta.tolist(),
        "actual_delta": actual_delta.tolist(),
        "max_abs_phase_only": int(np.abs(phase_delta).max()),
        "max_abs_selector": int(np.abs(selector_delta).max()),
        "max_abs_actual": int(np.abs(actual_delta).max()),
    }


def saturated_shared_max_response(family: LoadedFamily,
                                  rgb: Iterable[int],
                                  phases: np.ndarray = SATURATED_PHASES) -> dict:
    """Compare shared max-RGB control with per-channel adaptive control."""
    color = np.asarray(tuple(rgb), dtype=np.int64)
    source = np.tile(color, (16, 16, 1))
    positions = 8.0 + np.asarray(phases, dtype=np.float64)
    shared = simulate_pipeline(source, family, np.array([8.0]), positions,
                               "shared_max")[:, 0, :]
    per_channel = simulate_pipeline(source, family, np.array([8.0]), positions,
                                    "per_channel")[:, 0, :]
    delta = shared - per_channel
    return {
        "h_rgb": horizontal_uniform_rgb(family, color).tolist(),
        "phases": np.asarray(phases, dtype=float).tolist(),
        "shared": shared.tolist(),
        "per_channel": per_channel.tolist(),
        "delta": delta.tolist(),
        "max_abs_delta": int(np.abs(delta).max()),
    }


def _luma(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float64) @ LUMA


def _impulse_response(family: LoadedFamily) -> dict:
    size = 11
    center = size // 2
    source = np.zeros((size, size, 3), dtype=np.int64)
    source[center, center] = 255
    positions = center + np.arange(-8, 9, dtype=np.float64) / 4.0
    result = simulate_pipeline(source, family, positions, positions)
    baseline = simulate_pipeline(np.zeros_like(source), family, positions, positions)
    signal = np.maximum(_luma(result) - _luma(baseline), 0.0)
    total = float(signal.sum())
    xx, yy = np.meshgrid(positions, positions)
    if total > 0.0:
        cx = float((signal * xx).sum() / total)
        cy = float((signal * yy).sum() / total)
    else:
        cx = cy = float("nan")
    mid = len(positions) // 2
    return {
        "positions": positions.tolist(),
        "center_row_rgb": result[mid].tolist(),
        "center_column_rgb": result[:, mid].tolist(),
        "peak_luma": float(signal.max()),
        "signal_sum": total,
        "centroid": [cx, cy],
        "centroid_error": float(np.hypot(cx - center, cy - center)),
    }


def _step_response(family: LoadedFamily) -> dict:
    size = 12
    split = size // 2
    source = np.zeros((size, size, 3), dtype=np.int64)
    source[:, split:] = 255
    edge = split - 0.5
    positions = edge + np.arange(-8, 9, dtype=np.float64) / 8.0
    result = simulate_pipeline(source, family, positions,
                               np.array([float(split)]))[0]
    profile = _luma(result)
    backward = np.maximum(profile[:-1] - profile[1:], 0.0)
    return {
        "positions": positions.tolist(),
        "profile_rgb": result.tolist(),
        "profile_luma": profile.tolist(),
        "max_backtrack": float(backward.max(initial=0.0)),
        "backtrack_count_over_one_code": int(np.count_nonzero(backward > 1.0)),
        "endpoint_delta": float(profile[-1] - profile[0]),
    }


def _checkerboard_response(family: LoadedFamily) -> dict:
    size = 12
    source = np.zeros((size, size, 3), dtype=np.int64)
    yy, xx = np.indices((size, size))
    source[(xx + yy) % 2 == 0] = 255
    positions = 4.0 + np.arange(13, dtype=np.float64) / 4.0
    result = simulate_pipeline(source, family, positions, positions)
    luma = _luma(result)

    center_samples = result[::4, ::4]
    center_luma = _luma(center_samples)
    source_x = np.rint(positions[::4]).astype(np.int64)
    source_y = np.rint(positions[::4]).astype(np.int64)
    sample_yy, sample_xx = np.meshgrid(source_y, source_x, indexing="ij")
    white = (sample_xx + sample_yy) % 2 == 0
    white_mean = float(center_luma[white].mean())
    black_mean = float(center_luma[~white].mean())
    chroma_span = np.ptp(result.astype(np.float64), axis=2)
    return {
        "positions": positions.tolist(),
        "diagonal_rgb": result[np.arange(len(positions)),
                               np.arange(len(positions))].tolist(),
        "min_luma": float(luma.min()),
        "max_luma": float(luma.max()),
        "mean_luma": float(luma.mean()),
        "std_luma": float(luma.std()),
        "white_center_mean": white_mean,
        "black_center_mean": black_mean,
        "center_contrast": white_mean - black_mean,
        "max_grayscale_chroma_span": float(chroma_span.max()),
    }


def pattern_responses(family: LoadedFamily) -> dict:
    """Return representative exact pre-mask pattern responses."""
    return {
        "impulse": _impulse_response(family),
        "step": _step_response(family),
        "checkerboard": _checkerboard_response(family),
    }


def audit_family(family: LoadedFamily) -> tuple[dict, list[str], list[str]]:
    """Run every diagnostic and return results, hard failures, and reports."""
    hard: list[str] = []
    reports: list[str] = []

    selectors = {
        name: selector_boundary_response(family, upper, lower)
        for name, upper, lower in SELECTOR_CASES
    }
    selector_worst = max(v["max_abs_selector"] for v in selectors.values())
    selector_phase_worst = max(v["max_abs_phase_only"] for v in selectors.values())
    selector_actual_worst = max(v["max_abs_actual"] for v in selectors.values())

    saturated = {
        "_".join(map(str, rgb)): saturated_shared_max_response(family, rgb)
        for rgb in SATURATED_COLORS
    }
    shared_worst = max(v["max_abs_delta"] for v in saturated.values())
    patterns = pattern_responses(family)

    metrics = {
        "selector_control_jump": float(selector_worst),
        "selector_phase_only": float(selector_phase_worst),
        "selector_actual_jump": float(selector_actual_worst),
        "shared_max_delta": float(shared_worst),
        "impulse_centroid_error": patterns["impulse"]["centroid_error"],
        "step_backtrack": patterns["step"]["max_backtrack"],
        "checker_chroma_span": patterns["checkerboard"]["max_grayscale_chroma_span"],
    }
    for name, value in metrics.items():
        if not np.isfinite(value):
            hard.append(f"{family.spec.key}: non-finite {name}")
    for name, limit in REPORT_LIMITS.items():
        value = metrics[name]
        if np.isfinite(value) and value > limit:
            reports.append(
                f"{family.spec.key}: {name} {value:.3f} > report ceiling {limit:.3f}")

    # Model outputs are clamped by construction, but keeping these checks here
    # catches accidental bypasses or dtype regressions in this validator.
    arrays = []
    for response in selectors.values():
        arrays.extend((response["phase127"], response["phase128"]))
    for response in saturated.values():
        arrays.extend((response["shared"], response["per_channel"]))
    arrays.extend((patterns["impulse"]["center_row_rgb"],
                   patterns["impulse"]["center_column_rgb"],
                   patterns["step"]["profile_rgb"],
                   patterns["checkerboard"]["diagonal_rgb"]))
    for values in arrays:
        arr = np.asarray(values)
        if np.any((arr < HARD_LIMITS["output_code_min"])
                  | (arr > HARD_LIMITS["output_code_max"])):
            hard.append(f"{family.spec.key}: pattern response outside 0..255")
            break

    # Small deterministic replay: it exercises gamma, both FIR axes, adaptive
    # control, and phase selection without doubling every diagnostic's cost.
    replay_source = np.zeros((5, 5, 3), dtype=np.int64)
    replay_source[2, 2] = (255, 64, 192)
    replay_pos = np.array([1.75, 2.0, 2.5], dtype=np.float64)
    first = simulate_pipeline(replay_source, family, replay_pos, replay_pos)
    second = simulate_pipeline(replay_source, family, replay_pos, replay_pos)
    if not np.array_equal(first, second):
        hard.append(f"{family.spec.key}: nondeterministic pattern response")

    result = {
        "metrics": metrics,
        "selector_boundary": selectors,
        "saturated_shared_max": saturated,
        "patterns": patterns,
    }
    return result, hard, reports


def _print_family(spec: FamilySpec, result: dict, reports: list[str]) -> None:
    metrics = result["metrics"]
    print(f"\n=== {spec.file_base} ===")
    print("selector boundary: "
          f"max phase-only {metrics['selector_phase_only']:.0f}, "
          f"control-only {metrics['selector_control_jump']:.0f}, "
          f"actual {metrics['selector_actual_jump']:.0f} codes")
    for name, response in result["selector_boundary"].items():
        print(f"  {name:18s} phase {response['phase_only_delta']}  "
              f"selector {response['selector_delta']}  actual {response['actual_delta']}")
    print("saturated RGB: "
          f"max shared-minus-per-channel {metrics['shared_max_delta']:.0f} codes")
    impulse = result["patterns"]["impulse"]
    step = result["patterns"]["step"]
    checker = result["patterns"]["checkerboard"]
    print(f"impulse: centroid ({impulse['centroid'][0]:.4f}, "
          f"{impulse['centroid'][1]:.4f}), error "
          f"{impulse['centroid_error']:.4f}, peak luma {impulse['peak_luma']:.1f}")
    print(f"step: max backtrack {step['max_backtrack']:.1f}, "
          f"endpoint delta {step['endpoint_delta']:.1f}")
    print(f"checkerboard: center contrast {checker['center_contrast']:.1f}, "
          f"range {checker['min_luma']:.1f}..{checker['max_luma']:.1f}, "
          f"gray chroma span {checker['max_grayscale_chroma_span']:.1f}")
    for warning in reports:
        print(f"REPORT: {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), help="shader-pack root")
    parser.add_argument("--family", action="append",
                        choices=[spec.key for spec in FAMILIES],
                        help="limit to one or more families")
    parser.add_argument("--json", action="store_true",
                        help="emit complete machine-readable responses")
    parser.add_argument("--strict", action="store_true",
                        help="promote report ceilings to failures")
    args = parser.parse_args(argv)
    root = os.path.abspath(args.root)
    selected = [spec for spec in FAMILIES
                if args.family is None or spec.key in args.family]

    all_results: dict[str, dict] = {}
    hard_failures: list[str] = []
    all_reports: list[str] = []
    for spec in selected:
        try:
            family, load_problems = load_family(root, spec)
        except (OSError, ValueError, IndexError) as exc:
            hard_failures.append(f"{spec.key}: unable to load family: {exc}")
            continue
        hard_failures.extend(load_problems)
        result, hard, reports = audit_family(family)
        hard_failures.extend(hard)
        all_reports.extend(reports)
        all_results[spec.key] = result
        if not args.json:
            _print_family(spec, result, reports)

    if args.json:
        print(json.dumps({
            "hard_limits": HARD_LIMITS,
            "report_limits": REPORT_LIMITS,
            "families": all_results,
            "hard_failures": hard_failures,
            "reports": all_reports,
        }, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("\nReport ceilings are informational; use --strict after "
              "pattern-aware tables are stabilized.")
        print("Hard limits: signed coefficient |c| <= "
              f"{HARD_LIMITS['coefficient_abs']}, output codes "
              f"{HARD_LIMITS['output_code_min']}..{HARD_LIMITS['output_code_max']}, "
              "finite deterministic responses.")
        if hard_failures:
            print("HARD FAILURES:")
            for failure in hard_failures:
                print(f"  {failure}")
        else:
            print("HARD CHECKS PASS")
        print(f"REPORT WARNINGS: {len(all_reports)}")

    if hard_failures or (args.strict and all_reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
