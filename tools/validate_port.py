"""Acceptance validation for the generated shader ports (DESIGN.md gates).

Usage: py -3 validate_port.py

Reference modules are loaded from tools/targets/ when present, or from the
directory named by MISTER_SHADER_REFERENCES. A legacy local audit path is kept
as a final fallback for reproducibility on the original build machine.

Gates per shader:
  1. Format validators pass on every generated file; the complete preset
     collection resolves and satisfies axis/pairing contracts.
  2. End-to-end RMSE (exact RTL model, through the mask) vs the reference
     over an f x gray-ramp grid.
  3. Flat-field brightness parity at x = 0.25/0.5/0.75/1.0 (mask on, both sides).
  4. Behavioural clipping audit plus signed 10-bit coefficient range.
  5. Moire table (peak & trough stddev) at 4.5x and 4.821x.
Exit code 1 if any hard gate fails.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fileio
import fitting
import mister_model as mm  # noqa: F401  (imported for its RTL-arithmetic model)
import preset_contracts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_TARGETS = (r"C:\Users\meath\AppData\Local\Temp\claude\D--Arcade-AI-shaders"
                  r"\dd37dd4b-cacf-4be5-845c-14d761e0e4b1\scratchpad\targets")
TARGETS = os.environ.get(
    "MISTER_SHADER_REFERENCES", os.path.join(os.path.dirname(__file__), "targets"))
if not os.path.isdir(TARGETS) and os.path.isdir(LEGACY_TARGETS):
    TARGETS = LEGACY_TARGETS
if not os.path.isdir(TARGETS):
    sys.exit("Reference models not found. Set MISTER_SHADER_REFERENCES to the "
             "directory containing *_ref.py modules.")
sys.path.insert(0, TARGETS)

PORTS = {
    "guest": ("guest_advanced_ref", "CRT Guest Advanced (Port)", "CRT Guest Advanced"),
    "royale": ("royale_ref", "CRT Royale (Port)", "CRT Royale"),
    "kurozumi": ("kurozumi_ref", "CRT Royale Kurozumi (Port)", "CRT Royale Kurozumi"),
    "easymode": ("easymode_ref", "CRT Easymode v5 (Port)", "CRT Easymode v5"),
}
MOIRE_LIMIT = 7.65
REGRESSION_LIMITS = {
    # Locked to the regenerated full-LCM baselines with a small allowance for
    # deterministic fitter/NumPy differences.  L-infinity ceilings ensure an
    # apparently good RMS cannot hide a catastrophic single phosphor cell.
    "guest": {"masked": 1.30, "masked_max": 8.0,
              "fixed": 5.00, "fixed_max": 29.0,
              "no_gamma": 2.20, "no_gamma_max": 12.0, "brightness": 5.0},
    "royale": {"masked": 17.20, "masked_max": 82.0,
               "fixed": 24.20, "fixed_max": 104.0,
               "no_gamma": 18.00, "no_gamma_max": 82.0, "brightness": 4.0},
    "kurozumi": {"masked": 20.50, "masked_max": 125.0,
                 "fixed": 26.20, "fixed_max": 165.0,
                 "no_gamma": 20.20, "no_gamma_max": 85.0, "brightness": 12.0},
    "easymode": {"masked": 3.80, "masked_max": 20.0,
                 "fixed": 9.80, "fixed_max": 34.0,
                 "no_gamma": 7.00, "no_gamma_max": 24.0, "brightness": 12.5},
}

hard_fail = []

preset_problems = preset_contracts.validate_collection(ROOT)
print(f"=== Complete preset collection ===\n"
      f"contracts: {'OK' if not preset_problems else preset_problems}")
hard_fail += preset_problems


def simulate_grid(ref, dark, bright, h, lut):
    """Exact-model output codes vs reference, mask off, over f x x grid."""
    xs = fitting.EVAL_CODES
    fs = np.arange(0, 33) / 64.0                      # f = 0 .. 0.5
    errs = []
    for x in xs:
        g = int(lut[x, 1])
        # H stage on a flat field (captures its truncation) per channel-neutral
        hline = mm.fir_1d(np.full(16, g, dtype=np.int64), h, np.array([8.0]))[0]
        ctrl = np.full(len(fs), int(hline), dtype=np.int64)
        lines = np.full(16, int(hline), dtype=np.int64)
        pos = 8.0 + fs                                 # fractional line positions
        out = mm.fir_1d_adaptive(lines, dark, bright, pos, ctrl).astype(np.float64)
        ref_codes = np.array([255.0 * ref.ref_vertical(f, x / 255.0) for f in fs])
        errs.append(out - ref_codes)
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e * e))), float(np.abs(e).max())


def brightness_parity(ref, dark, bright, h, lut, mask, mask_encoded):
    """Mean exact masked output vs the shader, including post-mask clipping."""
    rows = []
    mask_mult16 = np.round(mask.multipliers() * 16).astype(np.int64)
    fs = np.arange(0, 33) / 64.0
    for x in (0.25, 0.5, 0.75, 1.0):
        code = int(round(x * 255))
        sim_profile = fitting.simulate_flat_rgb(dark, bright, h, lut, code, fs)
        masked = mm.mask_multiply(sim_profile[None, None, :, :],
                                  mask_mult16[:, :, None, :])
        sim = float(masked.mean())
        beam = np.array([[
            fitting._ref_vertical_unclipped(ref, f, code / 255.0, ch)
            for ch in "rgb"] for f in fs])
        target = 255.0 * np.minimum(
            mask_encoded[:, :, None, :] * beam[None, None, :, :], 1.0)
        ref_masked = float(target.mean())
        rows.append((x, sim, ref_masked, sim - ref_masked))
    return rows


def worst_moire(dark, bright, lut):
    """Worst peak/trough instability over the fractional-scale guard grid."""
    worst = 0.0
    for x in (0.25, 0.5, 0.75, 1.0):
        code = int(round(x * 255))
        for scale, n in ((4.5, 450), (4.821, 482)):
            prof = mm.flat_field_profile(code, scale, dark, bright, lut, n_out=n)
            met = mm.moire_metrics(prof, scale)
            worst = max(worst, met["peak_std"], met["trough_std"])
    return worst


def gamma_stats(lut: np.ndarray) -> dict[str, int]:
    """Quantization diagnostics that catch crushed ranges and large LUT jumps."""
    steps = np.diff(lut, axis=0)
    unique = min(len(np.unique(lut[:, c])) for c in range(3))
    longest = 1
    run = 1
    for same in np.all(steps == 0, axis=1):
        run = run + 1 if same else 1
        longest = max(longest, run)
    return {"unique": int(unique), "longest_plateau": int(longest),
            "max_step": int(steps.max(initial=0))}


def selector_control_jump(h: np.ndarray, dark: np.ndarray, bright: np.ndarray,
                          lut: np.ndarray) -> int:
    """Worst output jump caused only by the nearest-line control switch."""
    cases = (((64, 64, 64), (255, 255, 255)),
             ((0, 0, 0), (255, 255, 255)),
             ((255, 0, 0), (0, 0, 255)),
             ((255, 64, 64), (64, 64, 255)))

    def h_rgb(rgb):
        out = []
        for ch, code in enumerate(rgb):
            g = int(lut[code, ch])
            out.append(int(mm.fir_1d(
                np.full(16, g, dtype=np.int64), h, np.array([8.0]))[0]))
        return np.asarray(out, dtype=np.int64)

    worst = 0
    pos = np.array([8.5], dtype=np.float64)
    for upper_rgb, lower_rgb in cases:
        upper, lower = h_rgb(upper_rgb), h_rgb(lower_rgb)
        lines = np.empty((16, 3), dtype=np.int64)
        lines[:9] = upper
        lines[9:] = lower
        before, after = np.empty(3, dtype=np.int64), np.empty(3, dtype=np.int64)
        for ch in range(3):
            before[ch] = mm.fir_1d_adaptive(
                lines[:, ch], dark, bright, pos,
                np.array([upper.max()], dtype=np.int64))[0]
            after[ch] = mm.fir_1d_adaptive(
                lines[:, ch], dark, bright, pos,
                np.array([lower.max()], dtype=np.int64))[0]
        worst = max(worst, int(np.abs(after - before).max()))
    return worst


for key, (modname, fb, pb) in PORTS.items():
    ref = __import__(modname)
    print(f"\n=== {fb} ===")

    # --- 1. formats -----------------------------------------------------------
    problems = []
    files = {
        "H": os.path.join("Filters", f"{fb}_H.txt"),
        "VA": os.path.join("Filters", f"{fb}_V Adaptive.txt"),
        "VF": os.path.join("Filters", f"{fb}_V Fixed.txt"),
        "VNG": os.path.join("Filters", f"{fb}_V Adaptive No Gamma.txt"),
        "VN": os.path.join("Filters", f"{fb}_V No Scanlines.txt"),
    }
    if key in {"guest", "royale", "easymode"}:
        files["VAE"] = os.path.join(
            "Filters", f"{fb}_V Adaptive Edge Stable.txt")
    if key == "kurozumi":
        files["VAM"] = os.path.join(
            "Filters", f"{fb}_V Adaptive Anti-Moire.txt")
    parsed = {}
    for tag, rel in files.items():
        flt = fileio.parse_filter(os.path.join(ROOT, rel))
        problems += fileio.validate_filter(flt, rel)
        parsed[tag] = flt
    lut = fileio.parse_gamma(os.path.join(ROOT, "Gamma", f"{fb}.txt"))
    problems += fileio.validate_gamma(lut, f"{fb} gamma")
    fixed_lut = fileio.parse_gamma(
        os.path.join(ROOT, "Gamma", f"{fb} Fixed.txt"))
    problems += fileio.validate_gamma(fixed_lut, f"{fb} Fixed gamma")
    mask = fileio.parse_mask(os.path.join(ROOT, "Shadow_Masks", f"{fb}.txt"))
    problems += fileio.validate_mask(mask, f"{fb} mask")
    fixed_mask = fileio.parse_mask(
        os.path.join(ROOT, "Shadow_Masks", f"{fb} Fixed.txt"))
    problems += fileio.validate_mask(fixed_mask, f"{fb} Fixed mask")
    no_gamma_mask = fileio.parse_mask(
        os.path.join(ROOT, "Shadow_Masks", f"{fb} No Gamma.txt"))
    problems += fileio.validate_mask(no_gamma_mask, f"{fb} No Gamma mask")
    perceptual_mask = None
    if key == "kurozumi":
        perceptual_mask = fileio.parse_mask(os.path.join(
            ROOT, "Shadow_Masks", f"{fb} Perceptual Dither.txt"))
        problems += fileio.validate_mask(
            perceptual_mask, f"{fb} Perceptual Dither mask")
    import glob as _g
    for prp in _g.glob(os.path.join(ROOT, "Presets", f"{pb} - *.ini")):
        problems += fileio.validate_preset(fileio.parse_preset(prp), ROOT,
                                           os.path.basename(prp))
    print(f"formats: {'OK' if not problems else problems}")
    if problems:
        hard_fail += problems
    stats = gamma_stats(lut)
    print("gamma quantization: "
          f"{stats['unique']} minimum per-channel levels, longest RGB plateau "
          f"{stats['longest_plateau']} inputs, max step {stats['max_step']} codes")
    if stats["unique"] < 96 or stats["longest_plateau"] > 20 or stats["max_step"] > 20:
        hard_fail.append(f"{fb}: gamma quantization regression {stats}")

    # --- 4. clipping (behavioural) ------------------------------------------------
    # "max V row sum <= 256" is the WRONG invariant: the hardware clips on the
    # BLENDED row times the level, and the dark set's weight vanishes exactly
    # where the level is high, so an over-unity dark row is harmless while a
    # legal-looking pair can still clip. Test the property itself instead, and
    # gate the coefficient range separately (which the old rule covered by luck).
    va = parsed["VA"]
    worst_flat = fitting.worst_flat_field_output(lut, va.dark, va.bright)
    vertical_tags = [tag for tag in parsed if tag.startswith("V")]
    coef = max(int(np.abs(s).max()) for tag in vertical_tags
               for s in parsed[tag].sets)
    rows = max(int(s.sum(axis=1).max()) for tag in vertical_tags
               for s in parsed[tag].sets)
    ok_clip, ok_fmt = worst_flat <= 255.5, coef <= 511
    print(f"clipping: worst flat-field output {worst_flat:.1f}/255 "
          f"{'OK' if ok_clip else 'FAIL'}   (max row sum {rows}, "
          f"max |coeff| {coef}/511 {'OK' if ok_fmt else 'FAIL'})")
    if not ok_clip:
        hard_fail.append(f"{fb}: flat field clips ({worst_flat:.1f})")
    if not ok_fmt:
        hard_fail.append(f"{fb}: coefficient {coef} outside signed 10-bit")
    fixed_flat = fitting.worst_flat_field_output(
        fixed_lut, parsed["VF"].sets[0], parsed["VF"].sets[0])
    identity_lut = np.repeat(np.arange(256, dtype=np.int32)[:, None], 3, axis=1)
    no_gamma_flat = fitting.worst_flat_field_output(
        identity_lut, parsed["VNG"].dark, parsed["VNG"].bright)
    print(f"paired clipping: fixed {fixed_flat:.1f}/255, "
          f"no-gamma {no_gamma_flat:.1f}/255")
    if fixed_flat > 255.5:
        hard_fail.append(f"{fb} Fixed: flat field clips ({fixed_flat:.1f})")
    if no_gamma_flat > 255.5:
        hard_fail.append(f"{fb} No Gamma: flat field clips ({no_gamma_flat:.1f})")
    if "VAE" in parsed:
        edge = parsed["VAE"]
        edge_flat = fitting.worst_flat_field_output(lut, edge.dark, edge.bright)
        print(f"edge-stable clipping: {edge_flat:.1f}/255 "
              f"{'OK' if edge_flat <= 255.5 else 'FAIL'}")
        if edge_flat > 255.5:
            hard_fail.append(f"{fb} Edge Stable: flat field clips ({edge_flat:.1f})")
    if key == "kurozumi":
        anti = parsed["VAM"]
        anti_flat = fitting.worst_flat_field_output(lut, anti.dark, anti.bright)
        print(f"anti-moire clipping: worst flat-field output {anti_flat:.1f}/255 "
              f"{'OK' if anti_flat <= 255.5 else 'FAIL'}")
        if anti_flat > 255.5:
            hard_fail.append(f"{fb} Anti-Moire: flat field clips ({anti_flat:.1f})")

    # --- 2. RMSE ------------------------------------------------------------------
    # The masked metric is the acceptance number: it compares port pixels
    # against shader pixels THROUGH the mask, so it sees clamp-order error and
    # is meaningful for gain-split builds (where mask-off is G times dark by
    # construction). Mask-off is reported alongside for continuity only.
    m_target = fitting.mask_encoded_tile(ref)
    rmse, emax = fitting.rmse_exact_masked(ref, va.dark, va.bright,
                                           parsed["H"].sets[0], lut, mask.tokens,
                                           m_target)
    strict_rmse, _ = fitting.rmse_exact_masked(
        ref, va.dark, va.bright, parsed["H"].sets[0], lut, mask.tokens,
        m_target, align=False)
    rmse_off, _ = simulate_grid(ref, va.dark, va.bright, parsed["H"].sets[0], lut)
    print(f"end-to-end RMSE through the mask: {rmse:.3f} codes (max |err| {emax:.1f})"
          f"   [strict origin {strict_rmse:.3f}; mask off {rmse_off:.3f}]")
    selector_jump = selector_control_jump(
        parsed["H"].sets[0], va.dark, va.bright, lut)
    print(f"nearest-line selector-only jump: {selector_jump} codes")
    if rmse > REGRESSION_LIMITS[key]["masked"]:
        hard_fail.append(f"{fb}: masked RMSE {rmse:.3f} over regression ceiling "
                         f"{REGRESSION_LIMITS[key]['masked']:.3f}")
    if emax > REGRESSION_LIMITS[key]["masked_max"]:
        hard_fail.append(f"{fb}: max masked error {emax:.1f} over regression ceiling "
                         f"{REGRESSION_LIMITS[key]['masked_max']:.1f}")
    # Non-adaptive (v6) cores silently receive only the FIRST coefficient set.
    rdark, _ = fitting.rmse_exact_masked(ref, va.dark, va.dark, parsed["H"].sets[0],
                                         lut, mask.tokens, m_target)
    print(f"degradation on a non-adaptive core (dark set only): {rdark:.3f} codes")

    # fixed-table RMSE against its own paired LUT (what the preset actually loads)
    rmse_f, emax_f = fitting.rmse_exact_masked(
        ref, parsed["VF"].sets[0], parsed["VF"].sets[0], parsed["H"].sets[0],
        fixed_lut, fixed_mask.tokens, m_target)
    print(f"fixed-fallback masked RMSE: {rmse_f:.3f} codes (max |err| {emax_f:.1f})")
    if rmse_f > REGRESSION_LIMITS[key]["fixed"]:
        hard_fail.append(f"{fb}: fixed masked RMSE {rmse_f:.3f} over regression ceiling "
                         f"{REGRESSION_LIMITS[key]['fixed']:.3f}")
    if emax_f > REGRESSION_LIMITS[key]["fixed_max"]:
        hard_fail.append(f"{fb}: fixed max error {emax_f:.1f} over regression ceiling "
                         f"{REGRESSION_LIMITS[key]['fixed_max']:.1f}")

    # gamma=off is a distinct supported preset, not merely a syntax variation.
    rmse_ng, emax_ng = fitting.rmse_exact_masked(
        ref, parsed["VNG"].dark, parsed["VNG"].bright,
        parsed["H"].sets[0], identity_lut, no_gamma_mask.tokens, m_target)
    print(f"no-gamma masked RMSE: {rmse_ng:.3f} codes (max |err| {emax_ng:.1f})")
    if rmse_ng > REGRESSION_LIMITS[key]["no_gamma"]:
        hard_fail.append(f"{fb}: no-gamma masked RMSE {rmse_ng:.3f} over regression ceiling "
                         f"{REGRESSION_LIMITS[key]['no_gamma']:.3f}")
    if emax_ng > REGRESSION_LIMITS[key]["no_gamma_max"]:
        hard_fail.append(f"{fb}: no-gamma max error {emax_ng:.1f} over regression ceiling "
                         f"{REGRESSION_LIMITS[key]['no_gamma_max']:.1f}")

    if perceptual_mask is not None:
        rp, ep = fitting.rmse_exact_masked(
            ref, va.dark, va.bright, parsed["H"].sets[0], lut,
            perceptual_mask.tokens, m_target)
        print(f"perceptual-dither raw-pixel RMSE (informational): {rp:.3f} "
              f"codes (max |err| {ep:.1f}); this preset targets pair integration")

    if key == "kurozumi":
        black_rmse, black_max = fitting.rmse_exact_masked(
            ref, va.dark, va.bright, parsed["H"].sets[0], lut, mask.tokens,
            m_target, codes=np.array([0]))
        print(f"near-black colour gate (code 0): RMSE {black_rmse:.3f}, "
              f"max |err| {black_max:.1f}, LUT {lut[0].tolist()}")
        if black_rmse > 1.5:
            hard_fail.append(f"{fb}: code-0 colour RMSE {black_rmse:.3f} over 1.5")
        if black_max > 5.0:
            hard_fail.append(f"{fb}: code-0 max colour error {black_max:.1f} over 5.0")
        if not int(lut[0, 0]) > int(lut[0, 1]):
            hard_fail.append(f"{fb}: code-0 warm red flare was erased ({lut[0].tolist()})")

    if "VAE" in parsed:
        edge = parsed["VAE"]
        edge_rmse, _ = fitting.rmse_exact_masked(
            ref, edge.dark, edge.bright, parsed["H"].sets[0], lut,
            mask.tokens, m_target)
        edge_moire = worst_moire(edge.dark, edge.bright, lut)
        edge_jump = selector_control_jump(
            parsed["H"].sets[0], edge.dark, edge.bright, lut)
        print(f"edge-stable preset: masked RMSE {edge_rmse:.3f}, "
              f"moire {edge_moire:.2f}, selector jump {edge_jump} codes")
        if edge_moire > MOIRE_LIMIT:
            hard_fail.append(f"{fb} Edge Stable: moire {edge_moire:.2f} over {MOIRE_LIMIT}")
        if edge_jump > 16:
            hard_fail.append(f"{fb} Edge Stable: selector jump {edge_jump} over 16")

    # --- 3. brightness parity -------------------------------------------------------
    print("brightness parity (exact masked means, sim vs ref, codes):")
    parity = brightness_parity(ref, va.dark, va.bright,
                               parsed["H"].sets[0], lut, mask, m_target)
    for x, sim, refv, d in parity:
        print(f"  x={x:4}: sim {sim:6.1f}  ref {refv:6.1f}  delta {d:+5.1f}")
    worst_parity = max(abs(row[3]) for row in parity)
    if worst_parity > REGRESSION_LIMITS[key]["brightness"]:
        hard_fail.append(f"{fb}: brightness delta {worst_parity:.1f} over regression ceiling "
                         f"{REGRESSION_LIMITS[key]['brightness']:.1f}")

    # --- 5. moire ---------------------------------------------------------------------
    worst_m = worst_moire(va.dark, va.bright, lut)
    print(f"moire: worst peak/trough stddev {worst_m:.2f} codes "
          f"{'OK' if worst_m <= MOIRE_LIMIT else 'OVER GUARD'}")
    if key == "kurozumi":
        anti = parsed["VAM"]
        anti_m = worst_moire(anti.dark, anti.bright, lut)
        print(f"anti-moire preset: worst peak/trough stddev {anti_m:.2f} codes "
              f"{'OK' if anti_m <= MOIRE_LIMIT else 'FAIL'}")
        if anti_m > MOIRE_LIMIT:
            hard_fail.append(f"{fb} Anti-Moire: moire {anti_m:.2f} over {MOIRE_LIMIT}")
    elif worst_m > MOIRE_LIMIT:
        hard_fail.append(f"{fb}: moire {worst_m:.2f} over {MOIRE_LIMIT}")


# Lottes predates the generated v5 family and has different filenames plus
# gamma=off by design, so validate its matched adaptive/fixed pipelines here.
print("\n=== CRT Lottes (Port) ===")
lottes_ref = __import__("lottes_ref")
lottes_h_file = fileio.parse_filter(
    os.path.join(ROOT, "Filters", "CRT Lottes (Port)_H.txt"))
lottes_va = fileio.parse_filter(
    os.path.join(ROOT, "Filters", "CRT Lottes (Port)_V Adaptive.txt"))
lottes_vf = fileio.parse_filter(
    os.path.join(ROOT, "Filters", "CRT Lottes (Port)_V.txt"))
lottes_mask = fileio.parse_mask(os.path.join(
    ROOT, "Shadow_Masks", "CRT Lottes Matched Mask3 StretchedVGA (Port).txt"))
lottes_problems = []
lottes_problems += fileio.validate_filter(lottes_h_file, "CRT Lottes H")
lottes_problems += fileio.validate_filter(lottes_va, "CRT Lottes V Adaptive")
lottes_problems += fileio.validate_filter(lottes_vf, "CRT Lottes V Fixed")
lottes_problems += fileio.validate_mask(lottes_mask, "CRT Lottes matched mask")
print(f"formats: {'OK' if not lottes_problems else lottes_problems}")
hard_fail += lottes_problems

identity_lut = np.repeat(np.arange(256, dtype=np.int32)[:, None], 3, axis=1)
lottes_h = lottes_h_file.sets[0]
lottes_target = fitting.mask_encoded_tile(lottes_ref)
lottes_rmse, lottes_emax = fitting.rmse_exact_masked(
    lottes_ref, lottes_va.dark, lottes_va.bright, lottes_h, identity_lut,
    lottes_mask.tokens, lottes_target)
lottes_fixed_rmse, lottes_fixed_emax = fitting.rmse_exact_masked(
    lottes_ref, lottes_vf.sets[0], lottes_vf.sets[0], lottes_h, identity_lut,
    lottes_mask.tokens, lottes_target)
print(f"adaptive masked RMSE: {lottes_rmse:.3f} codes "
      f"(max |err| {lottes_emax:.1f})")
print(f"fixed masked RMSE: {lottes_fixed_rmse:.3f} codes "
      f"(max |err| {lottes_fixed_emax:.1f})")
if lottes_rmse > 1.4:
    hard_fail.append(f"CRT Lottes: adaptive masked RMSE {lottes_rmse:.3f} over 1.400")
if lottes_emax > 6.0:
    hard_fail.append(f"CRT Lottes: adaptive max error {lottes_emax:.1f} over 6.0")
if lottes_fixed_rmse > 1.3:
    hard_fail.append(f"CRT Lottes: fixed masked RMSE {lottes_fixed_rmse:.3f} over 1.300")
if lottes_fixed_emax > 6.0:
    hard_fail.append(f"CRT Lottes: fixed max error {lottes_fixed_emax:.1f} over 6.0")

lottes_flat = fitting.worst_flat_field_output(
    identity_lut, lottes_va.dark, lottes_va.bright)
print(f"clipping: worst flat-field output {lottes_flat:.1f}/255 "
      f"{'OK' if lottes_flat <= 255.5 else 'FAIL'}")
if lottes_flat > 255.5:
    hard_fail.append(f"CRT Lottes: flat field clips ({lottes_flat:.1f})")

lottes_parity = brightness_parity(
    lottes_ref, lottes_va.dark, lottes_va.bright, lottes_h, identity_lut,
    lottes_mask, lottes_target)
print("brightness parity (exact masked means, sim vs ref, codes):")
for x, sim, refv, delta in lottes_parity:
    print(f"  x={x:4}: sim {sim:6.1f}  ref {refv:6.1f}  delta {delta:+5.1f}")
lottes_worst_parity = max(abs(row[3]) for row in lottes_parity)
if lottes_worst_parity > 1.5:
    hard_fail.append(f"CRT Lottes: brightness delta {lottes_worst_parity:.1f} over 1.5")

lottes_moire = worst_moire(lottes_va.dark, lottes_va.bright, identity_lut)
print(f"moire: worst peak/trough stddev {lottes_moire:.2f} codes "
      f"{'OK' if lottes_moire <= MOIRE_LIMIT else 'FAIL'}")
if lottes_moire > MOIRE_LIMIT:
    hard_fail.append(f"CRT Lottes: moire {lottes_moire:.2f} over {MOIRE_LIMIT}")

success = "ALL REQUIRED GATES PASS"
print("\n" + ("HARD FAILURES:\n" + "\n".join(hard_fail) if hard_fail else success))
sys.exit(1 if hard_fail else 0)
