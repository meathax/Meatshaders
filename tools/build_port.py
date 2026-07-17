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
        "mask_style": "encoded_tile",
    },
    "royale": {
        "module": "royale_ref",
        "file_base": "CRT Royale (Port)",
        "preset_base": "CRT Royale",
        "shader_name": "crt-royale (TroggleMonkey)",
        "lut_channels": False,
        "mask_style": "rgb_amplify_tile",
    },
    "kurozumi": {
        "module": "kurozumi_ref",
        "file_base": "CRT Royale Kurozumi (Port)",
        "preset_base": "CRT Royale Kurozumi",
        "shader_name": "crt-royale-kurozumi (P22/PVM preset over crt-royale)",
        "lut_channels": True,
        "mask_style": "rgb_amplify_tile",
        "moire_soften_candidates": [0.05, 0.08, 0.12, 0.16, 0.22],
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


def mask_encoded_tile(ref, style: str) -> np.ndarray:
    spec = ref.mask_spec()
    if style == "encoded_tile":
        enc = spec.get("encoded_equivalent_multipliers")
        if enc is None:
            lin = np.array(spec["linear_multipliers"], dtype=np.float64)
            enc = lin ** (1.0 / 2.2)
        tile = np.array(enc, dtype=np.float64)
        if tile.ndim == 2:                       # (w, 3) row -> (1, w, 3)
            tile = tile[None, :, :]
        return tile
    rgb = np.array(spec["tile_rgb_0_255"], dtype=np.float64) / 255.0
    lin = rgb * float(spec["mask_amplify"])
    return np.clip(lin, 0.0, None) ** (1.0 / 2.2)


def build(key: str) -> None:
    cfg = SHADERS[key]
    ref = __import__(cfg["module"])
    fb, pb = cfg["file_base"], cfg["preset_base"]
    provenance = f"{cfg['shader_name']}; libretro/slang-shaders @ {COMMIT[:12]}"
    made = []

    # ---- gamma -------------------------------------------------------------
    lut = fitting.optimize_lut(ref, channels=cfg["lut_channels"])
    gpath = os.path.join(ROOT, "Gamma", f"{fb}.txt")
    fileio.write_gamma(gpath, lut, header=[
        f"Name: {fb}", provenance,
        "Warped beam-center transfer, co-optimized with the adaptive V fit;",
        "pair with the matching (Port) filters"])
    made.append(gpath)
    lut_ctrl = lut.max(axis=1)                    # adaptive control = max RGB

    # ---- V filters ----------------------------------------------------------
    dark, bright = fitting.fit_v_adaptive(ref, lut_ctrl)
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

    fixed = fitting.fit_v_fixed(ref, lut_ctrl)
    fpath = os.path.join(ROOT, "Filters", f"{fb}_V Fixed.txt")
    fileio.write_filter(fpath, fileio.FilterFile(
        [f"Name: {fb}_V Fixed"] + header_common +
        ["Best single-profile compromise for non-adaptive (v6) cores"],
        False, True, [fixed]))
    made.append(fpath)

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
    h = fitting.fit_h(ref)
    hpath = os.path.join(ROOT, "Filters", f"{fb}_H.txt")
    fileio.write_filter(hpath, fileio.FilterFile(
        [f"Name: {fb}_H"] + header_common, False, True, [h]))
    made.append(hpath)

    # ---- mask ------------------------------------------------------------------
    enc_tile = mask_encoded_tile(ref, cfg["mask_style"])
    tile, period_err = fitting.minimal_period_tile(enc_tile)
    tokens, tok_err = fitting.tile_from_encoded(tile)
    print(f"[{key}] mask: {enc_tile.shape[0]}x{enc_tile.shape[1]} -> "
          f"{len(tokens)}x{len(tokens[0])} tile, period rmse {period_err:.4f}, "
          f"token rmse {tok_err:.4f} (multiplier units)")
    mname = f"{fb}.txt"
    mpath = os.path.join(ROOT, "Shadow_Masks", mname)
    fileio.write_mask(mpath, fileio.MaskFile(
        [f"Name: {fb}", provenance,
         f"Encoded-space fit of the shader mask as rendered at 1080p "
         f"(token rmse {tok_err:.3f})",
         "Pair with the matching (Port) gamma and V filters"],
        len(tokens[0]), len(tokens), tokens))
    made.append(mpath)

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
    preset("Fixed Compatibility", {**base, "vfilter": f"{fb}_V Fixed.txt"})
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
