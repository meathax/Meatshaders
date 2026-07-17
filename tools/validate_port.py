"""Acceptance validation for the generated shader ports (DESIGN.md gates).

Usage: py -3 validate_port.py

Gates per shader:
  1. Format validators pass on every generated file; presets resolve.
  2. End-to-end RMSE (exact RTL model, mask off) vs reference ref_vertical
     over f x gray-ramp grid.
  3. Flat-field brightness parity at x = 0.25/0.5/0.75/1.0 (mask on, both sides).
  4. Clipping audit: no V row sum > 256.
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
import mister_model as mm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = (r"C:\Users\meath\AppData\Local\Temp\claude\D--Arcade-AI-shaders"
           r"\dd37dd4b-cacf-4be5-845c-14d761e0e4b1\scratchpad\targets")
sys.path.insert(0, TARGETS)

PORTS = {
    "guest": ("guest_advanced_ref", "CRT Guest Advanced (Port)", "CRT Guest Advanced"),
    "royale": ("royale_ref", "CRT Royale (Port)", "CRT Royale"),
    "kurozumi": ("kurozumi_ref", "CRT Royale Kurozumi (Port)", "CRT Royale Kurozumi"),
}

hard_fail = []


def simulate_grid(ref, dark, bright, h, lut):
    """Exact-model output codes vs reference, mask off, over f x x grid."""
    xs = np.arange(8, 256, 4)
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


def brightness_parity(ref, dark, bright, h, lut, mask_mult16):
    """Simulated masked f/x-averaged output vs reference x mask average."""
    rows = []
    mask_avg_enc = mask_mult16.astype(np.float64).mean(axis=(0, 1)) / 16.0
    for x in (0.25, 0.5, 0.75, 1.0):
        code = int(round(x * 255))
        prof = mm.flat_field_profile(code, 4.5, dark, bright, lut, n_out=450)
        # apply mask to the profile broadcast across a tile-wide strip
        strip = np.repeat(prof[:, None], mask_mult16.shape[1], axis=1)
        img = np.stack([strip] * 3, axis=2)
        masked = mm.apply_mask(img, mask_mult16)
        sim = float(masked.mean())
        ref_avg = np.mean([255.0 * ref.ref_vertical(f, x) for f in np.arange(33) / 64.0])
        ref_masked = ref_avg * float(mask_avg_enc.mean())
        rows.append((x, sim, ref_masked, sim - ref_masked))
    return rows


for key, (modname, fb, pb) in PORTS.items():
    ref = __import__(modname)
    print(f"\n=== {fb} ===")

    # --- 1. formats -----------------------------------------------------------
    problems = []
    files = {
        "H": f"Filters\\{fb}_H.txt",
        "VA": f"Filters\\{fb}_V Adaptive.txt",
        "VF": f"Filters\\{fb}_V Fixed.txt",
        "VN": f"Filters\\{fb}_V No Scanlines.txt",
    }
    parsed = {}
    for tag, rel in files.items():
        flt = fileio.parse_filter(os.path.join(ROOT, rel))
        problems += fileio.validate_filter(flt, rel)
        parsed[tag] = flt
    lut = fileio.parse_gamma(os.path.join(ROOT, "Gamma", f"{fb}.txt"))
    problems += fileio.validate_gamma(lut, f"{fb} gamma")
    mask = fileio.parse_mask(os.path.join(ROOT, "Shadow_Masks", f"{fb}.txt"))
    problems += fileio.validate_mask(mask, f"{fb} mask")
    import glob as _g
    for prp in _g.glob(os.path.join(ROOT, "Presets", f"{pb} - *.ini")):
        problems += fileio.validate_preset(fileio.parse_preset(prp), ROOT,
                                           os.path.basename(prp))
    print(f"formats: {'OK' if not problems else problems}")
    if problems:
        hard_fail += problems

    # --- 4. clipping ------------------------------------------------------------
    worst = max(int(s.sum(axis=1).max()) for tag in ("VA", "VF", "VN")
                for s in parsed[tag].sets)
    print(f"clipping: max V row sum {worst} / 256 {'OK' if worst <= 256 else 'FAIL'}")
    if worst > 256:
        hard_fail.append(f"{fb}: over-unity V row {worst}")

    # --- 2. RMSE ------------------------------------------------------------------
    va = parsed["VA"]
    rmse, emax = simulate_grid(ref, va.dark, va.bright, parsed["H"].sets[0], lut)
    print(f"end-to-end RMSE vs reference (mask off): {rmse:.3f} codes "
          f"(max |err| {emax:.1f})   [v4 Lottes benchmark: 1.8, Easymode: 4.0]")

    # fixed-table RMSE for the compatibility preset
    rmse_f, emax_f = simulate_grid(ref, parsed["VF"].sets[0], parsed["VF"].sets[0],
                                   parsed["H"].sets[0], lut)
    print(f"fixed-fallback RMSE: {rmse_f:.3f} codes (max |err| {emax_f:.1f})")

    # --- 3. brightness parity -------------------------------------------------------
    m16 = np.round(mask.multipliers() * 16).astype(np.int64)
    print("brightness parity (sim vs ref*maskavg, codes):")
    for x, sim, refv, d in brightness_parity(ref, va.dark, va.bright,
                                             parsed["H"].sets[0], lut, m16):
        print(f"  x={x:4}: sim {sim:6.1f}  ref {refv:6.1f}  delta {d:+5.1f}")

    # --- 5. moire ---------------------------------------------------------------------
    worst_m = 0.0
    for x in (0.25, 0.5, 0.75, 1.0):
        code = int(round(x * 255))
        for scale, n in ((4.5, 450), (4.821, 482)):
            prof = mm.flat_field_profile(code, scale, va.dark, va.bright, lut, n_out=n)
            met = mm.moire_metrics(prof, scale)
            worst_m = max(worst_m, met["peak_std"], met["trough_std"])
    print(f"moire: worst peak/trough stddev {worst_m:.2f} codes "
          f"{'OK' if worst_m <= 7.65 else 'OVER GUARD'}")

print("\n" + ("HARD FAILURES:\n" + "\n".join(hard_fail) if hard_fail else "ALL GATES PASS"))
sys.exit(1 if hard_fail else 0)
