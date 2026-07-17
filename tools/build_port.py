"""Build the complete MiSTer file set for one ported shader.

Usage: py -3 build_port.py <guest|royale|kurozumi>

Emits into the pack folders (Filters/, Gamma/, Shadow_Masks/, Presets/):
  H filter, V Adaptive, V Fixed, V No Scanlines, gamma LUT, mask(s), presets
per DESIGN.md. Prints fitting/moire diagnostics; validate_port.py does the
full acceptance run.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fileio
import fitting
import mister_model as mm
from fitting import rmse_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = (r"C:\Users\meath\AppData\Local\Temp\claude\D--Arcade-AI-shaders"
           r"\dd37dd4b-cacf-4be5-845c-14d761e0e4b1\scratchpad\targets")
sys.path.insert(0, TARGETS)

COMMIT = "3b0d6aa1d134a168478cd9c904a866d969f8882b"

SHADERS = {
    "guest": {
        "module": "guest_advanced_ref",
        "file_base": "CRT Guest Advanced (Port)",
        "preset_base": "CRT Guest Advanced",
        "shader_name": "crt-guest-advanced (guest.r)",
        "lut_channels": False,
        # transfer() never clips -> no headroom to split; 50e/20e mask is the
        # exact optimum under every weighting tested.
        "gain": None,
        "mask_strategy": None,
    },
    "royale": {
        "module": "royale_ref",
        "file_base": "CRT Royale (Port)",
        "preset_base": "CRT Royale",
        "shader_name": "crt-royale (TroggleMonkey)",
        "lut_channels": False,
        "gain": None,
        # The 24x24 slot tile is antisymmetric under y -> y+12, so averaging the
        # repeats cancels the slot modulation outright and leaves an aperture
        # grille. Take a verbatim slice instead and keep the slot.
        "mask_strategy": {"period": (12, 6), "kind": "verbatim"},
    },
    "kurozumi": {
        "module": "kurozumi_ref",
        "file_base": "CRT Royale Kurozumi (Port)",
        "preset_base": "CRT Royale Kurozumi",
        "shader_name": "crt-royale-kurozumi (P22/PVM preset over crt-royale)",
        "lut_channels": True,
        "gain": None,
        # (1.30, 1.01, 0.57) is unrepresentable in one v2 token (green near
        # unity while blue is halved, and the two share a nibble). Dither the
        # pair horizontally: exact-arithmetic error 11.6 -> 3.0 codes with the
        # R/B ripple restored to full amplitude and no vertical structure added.
        "mask_strategy": {"period": (1, 2), "dither": 2},
        "moire_soften_candidates": [0.05, 0.08, 0.12, 0.16, 0.22],
    },
    "easymode": {
        "module": "easymode_ref",
        "file_base": "CRT Easymode v5 (Port)",
        "preset_base": "CRT Easymode v5",
        "shader_name": "crt-easymode (Easymode, GPL)",
        "lut_channels": False,
        # BRIGHT_BOOST=1.2 clips the beam centre from x=0.849, which would pin
        # the LUT (and with it the adaptive control) over the top 15% of the
        # range. Carry B(x)/1.105 in the LUT and the gain in the mask instead:
        # saturation moves to the mask stage, where the shader also clips.
        "gain": 1.105,
        "mask_strategy": None,
    },
}

MOIRE_LIMIT = 0.03 * 255          # trough-stddev acceptance, output codes


class SoftenedRef:
    """Proxy that widens the vertical profile by Gaussian-averaging over f."""

    def __init__(self, ref, sigma: float):
        self._ref = ref
        self._sigma = sigma
        self._offs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * sigma
        w = np.exp(-0.5 * np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) ** 2)
        self._w = w / w.sum()

    def ref_vertical(self, f, x):
        vals = []
        for o in self._offs:
            fo = abs(f + o) % 1.0
            fo = min(fo, 1.0 - fo)
            vals.append(self._ref.ref_vertical(fo, x))
        return float((np.array(vals) * self._w).sum())

    def __getattr__(self, name):
        return getattr(self._ref, name)


def v_moire(dark, bright, lut, xs=(0.25, 0.5, 0.75, 1.0)) -> float:
    """Worst trough-stddev (codes) across gray levels and 4.5x / 4.821x."""
    worst = 0.0
    for x in xs:
        code = int(round(x * 255))
        for scale, n in ((4.5, 450), (4.821, 482)):
            prof = mm.flat_field_profile(code, scale, dark, bright, lut, n_out=n)
            m = mm.moire_metrics(prof, scale)
            worst = max(worst, m["trough_std"], m["peak_std"])
    return worst


def build(key: str) -> None:
    cfg = SHADERS[key]
    ref = __import__(cfg["module"])
    fb, pb = cfg["file_base"], cfg["preset_base"]
    provenance = f"{cfg['shader_name']}; libretro/slang-shaders @ {COMMIT[:12]}"
    made = []

    # ---- H filter (needed early: the exact refinement simulates through it) --
    h = fitting.fit_h(ref)
    gain = cfg.get("gain")

    # ---- mask ---------------------------------------------------------------
    m_target = fitting.mask_encoded_tile(ref)
    tokens, minfo = fitting.fit_mask_tile(ref, gain=gain or 1.0,
                                          strategy=cfg.get("mask_strategy"))
    print(f"[{key}] mask: source {m_target.shape[0]}x{m_target.shape[1]} -> "
          f"{len(tokens)}x{len(tokens[0])} ({minfo['kind']}, "
          f"structure {minfo['structure_kept'] * 100:.0f}%, "
          f"dither={minfo['dithered']}), exact err "
          f"{minfo.get('err_dither', minfo['err_plain']):.2f} codes")

    # ---- gamma + V, alternately refined against the exact hardware model -----
    if gain:
        lut = fitting.build_lut_gain(ref, gain, channels=cfg["lut_channels"])
        dark, bright = fitting.fit_v_adaptive_gain(ref, lut, gain)
        for it in range(3):
            lut = fitting.refine_lut_masked(ref, lut, h, dark, bright, tokens, m_target)
            dark, bright = fitting.fit_v_adaptive_gain(ref, lut, gain)
            r, _ = fitting.rmse_exact_masked(ref, dark, bright, h, lut, tokens, m_target)
            print(f"[{key}] refine pass {it + 1}: masked RMSE {r:.3f} codes")
    else:
        # Joint (warp, endpoint row-sum) fit. It supersedes optimize_lut +
        # fit_v_adaptive: those fit in row-sum space (an implicit 1/u^2 output
        # weighting) under a rows<=256 cap that is a far too strong proxy for
        # "no scaler clipping". Clip-safety is now measured, not assumed.
        lut, dark, bright, jinfo = fitting.fit_v_joint_safe(
            ref, channels=cfg["lut_channels"])
        print(f"[{key}] joint fit: RMSE {rmse_exact(ref, dark, bright, h, lut):.3f} "
              f"codes (mask off), bright cap {jinfo['cap_bright']:.0f}, "
              f"worst flat-field {jinfo['worst_flat']:.1f}/255")
        for it in range(2):
            lut2 = fitting.refine_lut_exact(ref, lut, h, dark, bright)
            if fitting.worst_flat_field_output(lut2, dark, bright) > 255.5:
                break                      # refinement must not break clip-safety
            lut = lut2
            print(f"[{key}] refine pass {it + 1}: RMSE "
                  f"{rmse_exact(ref, dark, bright, h, lut):.3f} codes (mask off)")

    gain_note = (f"Gain-split: LUT carries the pre-clip transfer / {gain:.3f}; "
                 f"the mask carries {gain:.3f}" if gain else
                 "LUT carries the beam-centre transfer")
    gpath = os.path.join(ROOT, "Gamma", f"{fb}.txt")
    fileio.write_gamma(gpath, lut, header=[
        f"Name: {fb}", provenance, gain_note,
        "Co-optimized with the adaptive V fit and refined against MiSTer's",
        "exact truncating arithmetic; pair with the matching (Port) files"])
    made.append(gpath)
    lut_ctrl = lut.max(axis=1)                    # adaptive control = max RGB

    mname = f"{fb}.txt"
    mpath = os.path.join(ROOT, "Shadow_Masks", mname)
    fileio.write_mask(mpath, fileio.MaskFile(
        [f"Name: {fb}", provenance,
         f"Shader mask as rendered at 1080p, fitted in encoded space with"
         f" MiSTer's exact mask arithmetic",
         (f"Carries the {gain:.3f} gain-split factor - use ONLY with the"
          f" matching (Port) gamma" if gain else
          "Pair with the matching (Port) gamma and V filters"),
         (f"Horizontally dithered: each column's target needs two tokens"
          if minfo["dithered"] else
          f"Tile kind: {minfo['kind']}")],
        len(tokens[0]), len(tokens), tokens))
    made.append(mpath)

    # ---- V filters ----------------------------------------------------------
    moire = v_moire(dark, bright, lut)
    print(f"[{key}] adaptive V moire (worst trough-std): {moire:.2f} codes "
          f"(limit {MOIRE_LIMIT:.2f})")

    header_common = [f"Original shader: {provenance}",
                     "Fitted to MiSTer RTL arithmetic (see tools/DESIGN.md)"]
    vpath = os.path.join(ROOT, "Filters", f"{fb}_V Adaptive.txt")
    fileio.write_filter(vpath, fileio.FilterFile(
        [f"Name: {fb}_V Adaptive"] + header_common +
        ["Pair with the matching (Port) gamma and mask; rows never exceed unity"],
        True, True, [dark, bright]),
        set_comments=["Primary coefficients (maximum RGB = 0)",
                      "Secondary coefficients (maximum RGB = 255)"])
    made.append(vpath)

    # Fixed fallback gets its own jointly-fitted LUT: without an adaptive
    # control the model is rank-1 and its optimal warp differs from the
    # adaptive one (sharing the adaptive LUT costs several codes of RMSE).
    fixed, fixed_lut = fitting.fit_v_fixed_paired(ref, channels=cfg["lut_channels"])
    for _ in range(2):
        fixed_lut = fitting.refine_lut_exact(ref, fixed_lut, h, fixed, fixed)
    print(f"[{key}] fixed fallback: RMSE {rmse_exact(ref, fixed, fixed, h, fixed_lut):.3f} "
          f"codes (own LUT)")
    fpath = os.path.join(ROOT, "Filters", f"{fb}_V Fixed.txt")
    fileio.write_filter(fpath, fileio.FilterFile(
        [f"Name: {fb}_V Fixed"] + header_common +
        ["Best single-profile compromise for non-adaptive (v6) cores",
         f"Pair with Gamma/{fb} Fixed.txt (NOT the adaptive gamma)"],
        False, True, [fixed]))
    made.append(fpath)
    fgpath = os.path.join(ROOT, "Gamma", f"{fb} Fixed.txt")
    fileio.write_gamma(fgpath, fixed_lut, header=[
        f"Name: {fb} Fixed", provenance,
        "Rank-1 transfer fitted jointly with the fixed (non-adaptive) V table;",
        "use only with the matching _V Fixed filter"])
    made.append(fgpath)

    nos = fitting.no_scanline_table(ref)
    npath = os.path.join(ROOT, "Filters", f"{fb}_V No Scanlines.txt")
    fileio.write_filter(npath, fileio.FilterFile(
        [f"Name: {fb}_V No Scanlines"] + header_common +
        ["Scan-free vertical interpolation; interlace fallback"],
        False, True, [nos]))
    made.append(npath)

    # ---- anti-moire variant (only if the exact fit fails the guard) ---------
    anti_name = None
    if moire > MOIRE_LIMIT:
        for sigma in cfg.get("moire_soften_candidates", [0.08, 0.16]):
            sref = SoftenedRef(ref, sigma)
            d2, b2 = fitting.fit_v_adaptive(sref, lut_ctrl)
            m2 = v_moire(d2, b2, lut)
            print(f"[{key}]   soften sigma={sigma}: moire {m2:.2f}")
            if m2 <= MOIRE_LIMIT:
                anti_name = f"{fb}_V Adaptive Anti-Moire.txt"
                apath = os.path.join(ROOT, "Filters", anti_name)
                fileio.write_filter(apath, fileio.FilterFile(
                    [f"Name: {fb}_V Adaptive Anti-Moire"] + header_common +
                    [f"Display-tuned: beam widened (sigma {sigma} lines) to pass",
                     "the fractional-scale moire guard; not pixel-exact"],
                    True, True, [d2, b2]),
                    set_comments=["Primary coefficients (maximum RGB = 0)",
                                  "Secondary coefficients (maximum RGB = 255)"])
                made.append(apath)
                break

    # ---- H filter ------------------------------------------------------------
    hpath = os.path.join(ROOT, "Filters", f"{fb}_H.txt")
    fileio.write_filter(hpath, fileio.FilterFile(
        [f"Name: {fb}_H"] + header_common, False, True, [h]))
    made.append(hpath)

    # ---- presets ------------------------------------------------------------------
    def preset(name: str, entries: dict[str, str]) -> None:
        p = os.path.join(ROOT, "Presets", f"{pb} - {name}.ini")
        fileio.write_preset(p, entries)
        made.append(p)

    base = {
        "hfilter": f"{fb}_H.txt",
        "vfilter": f"{fb}_V Adaptive.txt",
        "sfilter": "same",
        "ifilter": f"{fb}_V No Scanlines.txt",
        "gamma": f"{fb}.txt",
        "mask": mname,
        "maskmode": "1x",
    }
    preset("Default", base)
    preset("Fixed Compatibility", {**base, "vfilter": f"{fb}_V Fixed.txt",
                                   "gamma": f"{fb} Fixed.txt"})
    preset("No Gamma", {**base, "gamma": "off"})
    preset("TATE", {**base,
                    "hfilter": f"{fb}_V Adaptive.txt",
                    "vfilter": f"{fb}_H.txt",
                    "maskmode": "1x rotated"})
    if anti_name:
        preset("Anti-Moire", {**base, "vfilter": anti_name})

    print(f"[{key}] wrote {len(made)} files")
    for m in made:
        print("   ", os.path.relpath(m, ROOT))


if __name__ == "__main__":
    build(sys.argv[1])
