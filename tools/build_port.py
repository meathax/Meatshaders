"""Build one canonical 1080p MiSTer shader preset pair.

Usage: py -3 build_port.py <guest|royale|kurozumi|easymode>

Emits only the runtime assets used by ``<shader>.ini`` and
``<shader> - TATE.ini``: H, adaptive V, interlace-safe V, one gamma LUT, one
1080p mask, and the two presets.  Prints fitting/moire diagnostics;
validate_port.py performs the full acceptance run.
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

COMMIT = "3b0d6aa1d134a168478cd9c904a866d969f8882b"
# The three Guest families are built from guest.r's upstream release drop,
# which supersedes the libretro pin for them (verified identical at defaults
# for the advanced chain; see tools/targets/guest_*_ref.py provenance).
GUEST_SOURCE = "crt-guest-advanced-2026-07-12-release1"

SHADERS = {
    "guest": {
        "module": "guest_advanced_ref",
        "file_base": "CRT Guest Advanced (Port)",
        "preset_base": "CRT Guest Advanced",
        "shader_name": "crt-guest-advanced (guest.r)",
        "source": GUEST_SOURCE,
        "lut_channels": False,
        # transfer() never clips -> no headroom to split; 50e/20e mask is the
        # exact optimum under every weighting tested.
        "gain": None,
        "mask_strategy": None,
    },
    "guest_fast": {
        "module": "guest_fast_ref",
        "file_base": "CRT Guest Advanced Fast (Port)",
        "preset_base": "CRT Guest Advanced Fast",
        "shader_name": "crt-guest-advanced-fast (guest.r)",
        "source": GUEST_SOURCE,
        "lut_channels": False,
        # Upstream's performance rewrite of the identical look: its projection
        # onto MiSTer's fixed pipeline equals the advanced chain exactly, so
        # the fitter reproduces the advanced tables under this family's name.
        "gain": None,
        "mask_strategy": None,
    },
    "guest_fastest": {
        "module": "guest_fastest_ref",
        "file_base": "CRT Guest Advanced Fastest (Port)",
        "preset_base": "CRT Guest Advanced Fastest",
        "shader_name": "crt-guest-advanced-fastest (guest.r)",
        "source": GUEST_SOURCE,
        "lut_channels": False,
        # Single final pass: no glow lift, no pr_scan, transfer(1.0) == 1.0.
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
        # At 1080p, Royale's source-faithful beam is just narrow enough to
        # alternate between adjacent integer output-line placements at common
        # 224p/240p scale factors.  Blend in the smallest safely passing
        # sub-line integration fit (measured over the complete 1080-line
        # frame), retaining substantially more reference fidelity than using
        # the softened fit by itself.
        "stability_sigma": 0.16,
        "stability_blend": 0.55,
    },
    "kurozumi": {
        "module": "kurozumi_ref",
        "file_base": "CRT Royale Kurozumi (Port)",
        "preset_base": "CRT Royale Kurozumi",
        "shader_name": "crt-royale-kurozumi (P22/PVM preset over crt-royale)",
        "lut_channels": True,
        "gain": None,
        # Pixel-local is the default: every output pixel is scored separately.
        # The old horizontally-dithered pair was accurate only after averaging
        # partner pixels, while an individual pixel could be over 120 codes off.
        "mask_strategy": {"period": (1, 2), "kind": "mean"},
    },
    "easymode": {
        "module": "easymode_ref",
        "file_base": "CRT Easymode (Port)",
        "preset_base": "CRT Easymode",
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
OUTPUT_LINES = 1080
SOURCE_HEIGHTS = (224, 240)


class _SublineIntegratedRef:
    """Reference wrapper modelling the panel's finite vertical sample area."""

    def __init__(self, ref, sigma: float):
        self.ref = ref
        offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        weights = np.exp(-0.5 * offsets * offsets)
        self.offsets = offsets * sigma
        self.weights = weights / weights.sum()

    def ref_vertical(self, f, x):
        values = []
        for offset in self.offsets:
            wrapped = abs(f + offset) % 1.0
            wrapped = min(wrapped, 1.0 - wrapped)
            values.append(self.ref.ref_vertical(wrapped, x))
        return float(np.dot(values, self.weights))

    def __getattr__(self, name):
        return getattr(self.ref, name)


def v_moire(dark, bright, lut, xs=(0.25, 0.5, 0.75, 1.0)) -> float:
    """Worst full-frame trough/peak variation for common 1080p sources."""
    worst = 0.0
    for x in xs:
        code = int(round(x * 255))
        for source_lines in SOURCE_HEIGHTS:
            scale = OUTPUT_LINES / source_lines
            for channel in range(3):
                prof = mm.flat_field_profile(
                    code, scale, dark, bright, lut, n_out=OUTPUT_LINES,
                    channel=channel)
                m = mm.moire_metrics(prof, scale)
                worst = max(worst, m["trough_std"], m["peak_std"])
    return worst


def build(key: str) -> None:
    cfg = SHADERS[key]
    ref = __import__(cfg["module"])
    fb, pb = cfg["file_base"], cfg["preset_base"]
    source = cfg.get("source", f"libretro/slang-shaders @ {COMMIT[:12]}")
    provenance = f"{cfg['shader_name']}; {source}"
    made = []

    # ---- H filter (needed early: the exact refinement simulates through it) --
    h = fitting.fit_h(ref)
    gain = cfg.get("gain")

    def maskoff_metric(d, b, curve):
        fn = fitting.rmse_exact_rgb if cfg["lut_channels"] else rmse_exact
        return fn(ref, d, b, h, curve)

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
            old_score, _ = fitting.rmse_exact_masked(
                ref, dark, bright, h, lut, tokens, m_target)
            lut2 = fitting.refine_lut_masked(
                ref, lut, h, dark, bright, tokens, m_target,
                channel_aware=cfg["lut_channels"])
            dark2, bright2 = fitting.fit_v_adaptive_gain(ref, lut2, gain)
            new_score, _ = fitting.rmse_exact_masked(
                ref, dark2, bright2, h, lut2, tokens, m_target)
            if new_score >= old_score - 0.005:
                print(f"[{key}] refine pass {it + 1}: {new_score:.3f} vs "
                      f"{old_score:.3f} -- rejected by exact self-gate")
                break
            lut, dark, bright = lut2, dark2, bright2
            print(f"[{key}] refine pass {it + 1}: masked RMSE "
                  f"{new_score:.3f} codes")
    else:
        # Joint (warp, endpoint row-sum) fit. It supersedes optimize_lut +
        # fit_v_adaptive: those fit in row-sum space (an implicit 1/u^2 output
        # weighting) under a rows<=256 cap that is a far too strong proxy for
        # "no scaler clipping". Clip-safety is now measured, not assumed.
        lut, dark, bright, jinfo = fitting.fit_v_joint_safe(
            ref, channels=cfg["lut_channels"])
        print(f"[{key}] joint fit: RMSE {maskoff_metric(dark, bright, lut):.3f} "
              f"codes (mask off), bright cap {jinfo['cap_bright']:.0f}, "
              f"worst flat-field {jinfo['worst_flat']:.1f}/255")
        for it in range(2):
            lut2 = fitting.refine_lut_exact(
                ref, lut, h, dark, bright,
                channel_aware=cfg["lut_channels"])
            if fitting.worst_flat_field_output(lut2, dark, bright) > 255.5:
                break                      # refinement must not break clip-safety
            old_score = maskoff_metric(dark, bright, lut)
            new_score = maskoff_metric(dark, bright, lut2)
            if new_score >= old_score - 0.005:
                print(f"[{key}] refine pass {it + 1}: {new_score:.3f} vs "
                      f"{old_score:.3f} -- rejected by exact self-gate")
                break
            lut = lut2
            print(f"[{key}] refine pass {it + 1}: RMSE "
                  f"{new_score:.3f} codes (mask off)")

    # ---- mask, refit against the finished filters ---------------------------
    # The isolated fit matched the shader's multipliers; now that the V/LUT are
    # known, refit so the mask minimizes the error of the PRODUCT (absorbing
    # clamp-order and filter residual). Self-gating: kept only if it measurably
    # wins, since where the isolated fit was already optimal this is a no-op.
    r_before, _ = fitting.rmse_exact_masked(ref, dark, bright, h, lut, tokens, m_target)
    dgroup = len(tokens[0]) // 2 if minfo["dithered"] else None
    tok_j, r_after = fitting.fit_mask_joint(ref, lut, dark, bright, h, tokens,
                                            m_target, dither_group=dgroup)
    if r_after < r_before - 0.05:
        print(f"[{key}] mask refit (joint): masked RMSE {r_before:.3f} -> "
              f"{r_after:.3f} codes -- adopted")
        tokens = tok_j
        minfo["joint_refit"] = True
    else:
        print(f"[{key}] mask refit (joint): {r_after:.3f} vs {r_before:.3f} "
              f"-- isolated fit already optimal, kept")
        minfo["joint_refit"] = False

    # One final model-in-loop LUT pass uses the complete LCM mask supercell,
    # then refits the discrete tokens.  Keep it only when the WHOLE pipeline
    # improves and the behavioural clipping invariant remains true.
    lut_m = fitting.refine_lut_masked(
        ref, lut, h, dark, bright, tokens, m_target, radius=4,
        channel_aware=cfg["lut_channels"])
    if fitting.worst_flat_field_output(lut_m, dark, bright) <= 255.5:
        tok_m, r_m = fitting.fit_mask_joint(
            ref, lut_m, dark, bright, h, tokens, m_target)
        r_now, _ = fitting.rmse_exact_masked(
            ref, dark, bright, h, lut, tokens, m_target)
        if r_m < r_now - 0.05:
            print(f"[{key}] mask-aware LUT pass: masked RMSE {r_now:.3f} -> "
                  f"{r_m:.3f} codes -- adopted")
            lut, tokens = lut_m, tok_m
            minfo["mask_aware_lut"] = True
        else:
            print(f"[{key}] mask-aware LUT pass: {r_m:.3f} vs {r_now:.3f} "
                  "-- no material win, kept")
            minfo["mask_aware_lut"] = False
    else:
        print(f"[{key}] mask-aware LUT pass rejected: flat-field clipping")
        minfo["mask_aware_lut"] = False

    # Royale alone needs a narrowly scoped 1080p stability constraint.  Fit a
    # finite-subline reference using the FINISHED LUT, blend it into the exact
    # shader fit, then give the mask one last chance to follow the constrained
    # beam.  The full-frame guard below remains the authority.
    if "stability_blend" in cfg:
        alpha = cfg["stability_blend"]
        integrated = _SublineIntegratedRef(ref, cfg["stability_sigma"])
        soft_dark, soft_bright = fitting.fit_v_adaptive(
            integrated, lut.max(axis=1))
        dark = np.rint((1.0 - alpha) * dark + alpha * soft_dark).astype(np.int64)
        bright = np.rint(
            (1.0 - alpha) * bright + alpha * soft_bright).astype(np.int64)
        constrained_before, _ = fitting.rmse_exact_masked(
            ref, dark, bright, h, lut, tokens, m_target)
        constrained_tokens, constrained_after = fitting.fit_mask_joint(
            ref, lut, dark, bright, h, tokens, m_target)
        if constrained_after < constrained_before:
            tokens = constrained_tokens
        print(f"[{key}] 1080p stability blend: {alpha:.0%} integrated beam, "
              f"masked RMSE {min(constrained_before, constrained_after):.3f} codes")

    gain_note = (f"Gain-split: LUT carries the pre-clip transfer / {gain:.3f}; "
                 f"the mask carries {gain:.3f}" if gain else
                 "Monotone adaptive-control warp co-optimized with the V fit")
    gpath = os.path.join(ROOT, "Gamma", f"{fb}.txt")
    fileio.write_gamma(gpath, lut, header=[
        f"Name: {fb}", provenance, gain_note,
        "Co-optimized with the adaptive V fit and refined against MiSTer's",
        "exact truncating arithmetic; pair with the matching (Port) files"])
    made.append(gpath)
    mname = f"{fb}.txt"
    mpath = os.path.join(ROOT, "Shadow_Masks", mname)
    fileio.write_mask(mpath, fileio.MaskFile(
        [f"Name: {fb}", provenance,
         "Fitted in encoded space with MiSTer's exact saturating mask arithmetic",
         (f"Carries the {gain:.3f} gain-split factor; pair only with the "
          "matching (Port) gamma" if gain else
          "Jointly fitted with the matching adaptive gamma and V filter"),
         "Pixel-local objective over the complete hardware/reference LCM supercell",
         f"Tile kind: {minfo['kind']}"],
        len(tokens[0]), len(tokens), tokens))
    made.append(mpath)

    # ---- V filters ----------------------------------------------------------
    moire = v_moire(dark, bright, lut)
    print(f"[{key}] adaptive V moire (worst peak/trough stddev): {moire:.2f} codes "
          f"(limit {MOIRE_LIMIT:.2f})")
    if moire > MOIRE_LIMIT:
        raise RuntimeError(
            f"{key}: canonical adaptive profile exceeds the 1080p moire guard "
            f"({moire:.2f} > {MOIRE_LIMIT:.2f} codes)")

    header_common = [f"Original shader: {provenance}",
                     "Fitted to MiSTer RTL arithmetic (see tools/DESIGN.md)"]
    vpath = os.path.join(ROOT, "Filters", f"{fb}_V Adaptive.txt")
    fileio.write_filter(vpath, fileio.FilterFile(
        [f"Name: {fb}_V Adaptive"] + header_common +
        ["Pair with the matching (Port) gamma and mask; behaviourally clip-safe",
         "Endpoint rows may exceed unity where the adaptive blend keeps output in range"],
        True, True, [dark, bright]),
        set_comments=["Primary coefficients (maximum RGB = 0)",
                      "Secondary coefficients (maximum RGB = 255)"])
    made.append(vpath)

    nos = fitting.no_scanline_table(ref)
    npath = os.path.join(ROOT, "Filters", f"{fb}_V No Scanlines.txt")
    fileio.write_filter(npath, fileio.FilterFile(
        [f"Name: {fb}_V No Scanlines"] + header_common +
        ["Scan-free vertical interpolation; interlace fallback"],
        False, True, [nos]))
    made.append(npath)

    # ---- H filter ------------------------------------------------------------
    hpath = os.path.join(ROOT, "Filters", f"{fb}_H.txt")
    fileio.write_filter(hpath, fileio.FilterFile(
        [f"Name: {fb}_H"] + header_common, False, True, [h]))
    made.append(hpath)

    # ---- presets ------------------------------------------------------------------
    def preset(filename: str, entries: dict[str, str]) -> None:
        p = os.path.join(ROOT, "Presets", filename)
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
    preset(f"{pb}.ini", base)
    preset(f"{pb} - TATE.ini", {
        **base,
        "hfilter": f"{fb}_V Adaptive.txt",
        "vfilter": f"{fb}_H.txt",
        "ifilter": f"{fb}_H.txt",
        "maskmode": "1x rotated",
    })

    print(f"[{key}] wrote {len(made)} files")
    for m in made:
        print("   ", os.path.relpath(m, ROOT))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SHADERS:
        choices = "|".join(SHADERS)
        sys.exit(f"Usage: py -3 build_port.py <{choices}>")
    build(sys.argv[1])
