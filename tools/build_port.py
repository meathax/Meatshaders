"""Build the complete MiSTer file set for one ported shader.

Usage: py -3 build_port.py <guest|royale|kurozumi|easymode>

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
        # A two-phase selector-only regularization removes the nearest-line
        # control discontinuity with a small flat-field fidelity tradeoff.
        "edge_stable": {"kind": "phase", "radius": 2, "strength": 1.0},
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
        # Royale's narrow bright beam needs a global reduction in adaptation;
        # local selector smoothing alone aliases badly at fractional scales.
        # Move dark toward bright only so legal high-gain dark rows are never
        # imported into the highlight endpoint (which would clip).
        "edge_stable": {"kind": "dark_to_bright", "strength": 0.854},
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
        # Keep that lower-frequency perceptual compromise as an explicit option
        # for viewers who prefer pair-averaged PVM colour over local accuracy.
        "perceptual_mask_strategy": {"period": (1, 2), "dither": 2},
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
        "edge_stable": {"kind": "global", "strength": 0.70},
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


def anti_moire_sigmas(cfg: dict, default_moire: float) -> tuple[float, ...]:
    """Return the deterministic softening search for an optional safety preset.

    A configured candidate list means the Anti-Moire preset is part of that
    family's generated asset matrix, even when a newly fitted Default happens
    to pass the guard.  Otherwise a failing Default gets the generic fallback
    search.  This prevents incremental rebuilds from silently retaining a
    filter fitted against an older gamma/control LUT.
    """
    if "moire_soften_candidates" in cfg:
        return tuple(float(s) for s in cfg["moire_soften_candidates"])
    if default_moire > MOIRE_LIMIT:
        return (0.08, 0.16)
    return ()


def build(key: str) -> None:
    cfg = SHADERS[key]
    ref = __import__(cfg["module"])
    fb, pb = cfg["file_base"], cfg["preset_base"]
    provenance = f"{cfg['shader_name']}; libretro/slang-shaders @ {COMMIT[:12]}"
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

    gain_note = (f"Gain-split: LUT carries the pre-clip transfer / {gain:.3f}; "
                 f"the mask carries {gain:.3f}" if gain else
                 "Monotone adaptive-control warp co-optimized with the V fit")
    gpath = os.path.join(ROOT, "Gamma", f"{fb}.txt")
    fileio.write_gamma(gpath, lut, header=[
        f"Name: {fb}", provenance, gain_note,
        "Co-optimized with the adaptive V fit and refined against MiSTer's",
        "exact truncating arithmetic; pair with the matching (Port) files"])
    made.append(gpath)
    lut_ctrl = lut.max(axis=1)                    # adaptive control = max RGB

    def write_mask_variant(filename: str, variant_tokens: list[list[str]],
                           notes: list[str]) -> None:
        mpath = os.path.join(ROOT, "Shadow_Masks", filename)
        fileio.write_mask(mpath, fileio.MaskFile(
            [f"Name: {os.path.splitext(filename)[0]}", provenance,
             "Fitted in encoded space with MiSTer's exact saturating mask arithmetic",
             *notes], len(variant_tokens[0]), len(variant_tokens), variant_tokens))
        made.append(mpath)

    mname = f"{fb}.txt"
    write_mask_variant(mname, tokens, [
        (f"Carries the {gain:.3f} gain-split factor; pair only with the "
         "matching (Port) gamma" if gain else
         "Jointly fitted with the matching adaptive gamma and V filter"),
        "Pixel-local objective over the complete hardware/reference LCM supercell",
        f"Tile kind: {minfo['kind']}",
    ])

    perceptual_name = None
    if cfg.get("perceptual_mask_strategy"):
        perceptual_tokens, pinfo = fitting.fit_mask_tile(
            ref, gain=gain or 1.0, strategy=cfg["perceptual_mask_strategy"])
        perceptual_name = f"{fb} Perceptual Dither.txt"
        write_mask_variant(perceptual_name, perceptual_tokens, [
            "Optional pair-averaged perceptual dither; not pixel-local",
            "Partner columns integrate toward the PVM phosphor target while each",
            "individual pixel carries a deliberate high-frequency colour residual",
            f"Pair-averaged isolated-mask error: {pinfo['err_dither']:.2f} codes",
        ])
        rp, ep = fitting.rmse_exact_masked(
            ref, dark, bright, h, lut, perceptual_tokens, m_target)
        print(f"[{key}] perceptual-dither mask: pair-average "
              f"{pinfo['err_dither']:.2f}, pixel-local pipeline RMSE {rp:.3f} "
              f"(max {ep:.1f})")

    # ---- V filters ----------------------------------------------------------
    moire = v_moire(dark, bright, lut)
    print(f"[{key}] adaptive V moire (worst trough-std): {moire:.2f} codes "
          f"(limit {MOIRE_LIMIT:.2f})")

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

    # Fixed fallback gets its own jointly-fitted LUT: without an adaptive
    # control the model is rank-1 and its optimal warp differs from the
    # adaptive one (sharing the adaptive LUT costs several codes of RMSE).
    # Its mask and final LUT are also optimized as a PAIR instead of silently
    # reusing assets calibrated for the adaptive pipeline.
    fixed, fixed_lut = fitting.fit_v_fixed_paired(ref, channels=cfg["lut_channels"])
    for _ in range(2):
        fixed_lut2 = fitting.refine_lut_exact(
            ref, fixed_lut, h, fixed, fixed,
            channel_aware=cfg["lut_channels"])
        if maskoff_metric(fixed, fixed, fixed_lut2) >= \
                maskoff_metric(fixed, fixed, fixed_lut) - 0.005:
            break
        fixed_lut = fixed_lut2
    fixed_base, _ = fitting.rmse_exact_masked(
        ref, fixed, fixed, h, fixed_lut, tokens, m_target)
    fixed_tokens, fixed_mask_err = fitting.fit_mask_joint(
        ref, fixed_lut, fixed, fixed, h, tokens, m_target)
    fixed_lut_m = fitting.refine_lut_masked(
        ref, fixed_lut, h, fixed, fixed, fixed_tokens, m_target, radius=4,
        channel_aware=cfg["lut_channels"])
    fixed_tokens_m, fixed_pair_err = fitting.fit_mask_joint(
        ref, fixed_lut_m, fixed, fixed, h, fixed_tokens, m_target)
    fixed_candidates = [
        (fixed_base, fixed_lut, tokens, "adaptive mask retained"),
        (fixed_mask_err, fixed_lut, fixed_tokens, "fixed mask"),
        (fixed_pair_err, fixed_lut_m, fixed_tokens_m, "mask-aware fixed pair"),
    ]
    fixed_err, fixed_lut, fixed_tokens, fixed_choice = min(
        fixed_candidates, key=lambda item: item[0])
    print(f"[{key}] fixed fallback: masked RMSE {fixed_base:.3f} -> "
          f"{fixed_err:.3f} codes ({fixed_choice}); mask-off "
          f"{maskoff_metric(fixed, fixed, fixed_lut):.3f}")
    fixed_mask_name = f"{fb} Fixed.txt"
    write_mask_variant(fixed_mask_name, fixed_tokens, [
        "Dedicated pixel-local mask for the non-adaptive rank-1 pipeline",
        f"Selected by whole-pipeline self-gate: {fixed_choice}",
    ])
    fpath = os.path.join(ROOT, "Filters", f"{fb}_V Fixed.txt")
    fileio.write_filter(fpath, fileio.FilterFile(
        [f"Name: {fb}_V Fixed"] + header_common +
        ["Best single-profile compromise for non-adaptive (v6) cores",
         f"Pair with Gamma/{fb} Fixed.txt and Shadow_Masks/{fixed_mask_name}"],
        False, True, [fixed]))
    made.append(fpath)
    fgpath = os.path.join(ROOT, "Gamma", f"{fb} Fixed.txt")
    fileio.write_gamma(fgpath, fixed_lut, header=[
        f"Name: {fb} Fixed", provenance,
        "Rank-1 transfer fitted jointly with the fixed V table and refined",
        "through its dedicated full-period mask; use only with that pair"])
    made.append(fgpath)

    # gamma=off is a real supported pipeline.  Fit V endpoints and a mask for
    # the identity LUT, compare that pair with a mask-only recalibration of the
    # canonical tables, and keep the exact whole-pipeline winner.
    identity_lut = np.repeat(np.arange(256, dtype=np.int64)[:, None], 3, axis=1)
    ng_base, _ = fitting.rmse_exact_masked(
        ref, dark, bright, h, identity_lut, tokens, m_target)
    ng_base_tokens, ng_base_mask_err = fitting.fit_mask_joint(
        ref, identity_lut, dark, bright, h, tokens, m_target)
    ng_dark_fit, ng_bright_fit, ng_info = fitting.fit_v_for_lut_safe(
        ref, identity_lut)
    ng_fit_tokens, ng_fit_err = fitting.fit_mask_joint(
        ref, identity_lut, ng_dark_fit, ng_bright_fit, h, tokens, m_target)
    ng_candidates = [
        (ng_base, dark, bright, tokens, "adaptive pair retained"),
        (ng_base_mask_err, dark, bright, ng_base_tokens, "identity-mask recalibration"),
        (ng_fit_err, ng_dark_fit, ng_bright_fit, ng_fit_tokens,
         "identity-LUT V/mask co-fit"),
    ]
    ng_err, ng_dark, ng_bright, ng_tokens, ng_choice = min(
        ng_candidates, key=lambda item: item[0])
    print(f"[{key}] no-gamma pipeline: masked RMSE {ng_base:.3f} -> "
          f"{ng_err:.3f} codes ({ng_choice}); worst flat "
          f"{fitting.worst_flat_field_output(identity_lut, ng_dark, ng_bright):.1f}")
    ng_detail = (f"Safe-fit bright cap {ng_info['cap_bright']:.0f}"
                 if ng_choice == "identity-LUT V/mask co-fit" else
                 "Canonical endpoints retained; the mask-only calibration won")
    ng_vname = f"{fb}_V Adaptive No Gamma.txt"
    ng_vpath = os.path.join(ROOT, "Filters", ng_vname)
    fileio.write_filter(ng_vpath, fileio.FilterFile(
        [f"Name: {fb}_V Adaptive No Gamma"] + header_common + [
            "Dedicated identity-LUT (gamma=off) endpoint pair",
            f"Whole-pipeline self-gate selected: {ng_choice}",
            f"{ng_detail}; pair with its No Gamma mask",
        ], True, True, [ng_dark, ng_bright]),
        set_comments=["Primary coefficients (maximum RGB = 0)",
                      "Secondary coefficients (maximum RGB = 255)"])
    made.append(ng_vpath)
    ng_mask_name = f"{fb} No Gamma.txt"
    write_mask_variant(ng_mask_name, ng_tokens, [
        "Dedicated pixel-local mask for gamma=off / identity LUT",
        f"Selected by whole-pipeline self-gate: {ng_choice}",
    ])

    nos = fitting.no_scanline_table(ref)
    npath = os.path.join(ROOT, "Filters", f"{fb}_V No Scanlines.txt")
    fileio.write_filter(npath, fileio.FilterFile(
        [f"Name: {fb}_V No Scanlines"] + header_common +
        ["Scan-free vertical interpolation; interlace fallback"],
        False, True, [nos]))
    made.append(npath)

    # Optional pattern-stable profile.  Default remains the closest flat-field
    # shader match; this variant bounds the nearest-line selector discontinuity
    # for hard sprite/text edges and must also pass the fractional-scale guard.
    edge_name = None
    edge_cfg = cfg.get("edge_stable")
    if edge_cfg:
        if edge_cfg["kind"] == "phase":
            ed, eb = fitting.stabilize_adaptive_selector(
                dark, bright, radius=edge_cfg["radius"],
                strength=edge_cfg["strength"])
            edge_method = (f"phase-local selector regularization, radius "
                           f"{edge_cfg['radius']}/256 line")
        elif edge_cfg["kind"] == "global":
            ed, eb = fitting.blend_adaptive_endpoints(
                dark, bright, edge_cfg["strength"])
            edge_method = (f"global endpoint blend, strength "
                           f"{edge_cfg['strength']:.2f}")
        elif edge_cfg["kind"] == "dark_to_bright":
            ed, eb = fitting.blend_dark_toward_bright(
                dark, bright, edge_cfg["strength"])
            edge_method = (f"clip-safe dark-to-bright endpoint blend, strength "
                           f"{edge_cfg['strength']:.3f}")
        else:
            raise ValueError(f"unknown edge-stable strategy {edge_cfg['kind']!r}")
        edge_moire = v_moire(ed, eb, lut)
        edge_rmse, _ = fitting.rmse_exact_masked(
            ref, ed, eb, h, lut, tokens, m_target)
        if (fitting.worst_flat_field_output(lut, ed, eb) <= 255.5
                and edge_moire <= MOIRE_LIMIT):
            edge_name = f"{fb}_V Adaptive Edge Stable.txt"
            edge_path = os.path.join(ROOT, "Filters", edge_name)
            fileio.write_filter(edge_path, fileio.FilterFile(
                [f"Name: {fb}_V Adaptive Edge Stable"] + header_common + [
                    f"Pattern-tuned: {edge_method}",
                    "Reduces nearest-line control switching on hard nonuniform edges",
                    f"Masked flat-field RMSE {edge_rmse:.2f}; moire {edge_moire:.2f} codes",
                ], True, True, [ed, eb]),
                set_comments=["Primary coefficients (maximum RGB = 0)",
                              "Secondary coefficients (maximum RGB = 255)"])
            made.append(edge_path)
            print(f"[{key}] edge-stable variant: RMSE {edge_rmse:.3f}, "
                  f"moire {edge_moire:.2f} -- emitted")
        else:
            print(f"[{key}] edge-stable candidate rejected (moire "
                  f"{edge_moire:.2f} or clipping)")

    # ---- anti-moire safety variant -----------------------------------------
    # Explicitly configured families always regenerate this optional profile.
    # Its coefficients depend on the current LUT/control warp, so leaving an
    # older file in place merely because Default now passes would create a
    # stale cross-generation pair.
    anti_name = None
    anti_sigmas = anti_moire_sigmas(cfg, moire)
    if anti_sigmas:
        for sigma in anti_sigmas:
            sref = SoftenedRef(ref, sigma)
            d2, b2 = fitting.fit_v_adaptive(sref, lut_ctrl)
            m2 = v_moire(d2, b2, lut)
            print(f"[{key}]   soften sigma={sigma}: moire {m2:.2f}")
            if m2 <= MOIRE_LIMIT:
                anti_name = f"{fb}_V Adaptive Anti-Moire.txt"
                apath = os.path.join(ROOT, "Filters", anti_name)
                fileio.write_filter(apath, fileio.FilterFile(
                    [f"Name: {fb}_V Adaptive Anti-Moire"] + header_common +
                    [f"Display-tuned: beam widened (sigma {sigma} lines) for extra",
                     "fractional-scale stability; passes the moire guard; not pixel-exact"],
                    True, True, [d2, b2]),
                    set_comments=["Primary coefficients (maximum RGB = 0)",
                                  "Secondary coefficients (maximum RGB = 255)"])
                made.append(apath)
                break
        if anti_name is None:
            raise RuntimeError(
                f"{key}: configured Anti-Moire search did not pass the "
                f"{MOIRE_LIMIT:.2f}-code guard")

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
                                   "gamma": f"{fb} Fixed.txt",
                                   "mask": fixed_mask_name})
    preset("No Gamma", {**base, "vfilter": ng_vname,
                         "gamma": "off", "mask": ng_mask_name})
    preset("TATE", {**base,
                    "hfilter": f"{fb}_V Adaptive.txt",
                    "vfilter": f"{fb}_H.txt",
                    "ifilter": f"{fb}_H.txt",
                    "maskmode": "1x rotated"})
    if edge_name:
        preset("Edge Stable", {**base, "vfilter": edge_name})
    if anti_name:
        preset("Anti-Moire", {**base, "vfilter": anti_name})
    if perceptual_name:
        preset("Perceptual Dither", {**base, "mask": perceptual_name})

    print(f"[{key}] wrote {len(made)} files")
    for m in made:
        print("   ", os.path.relpath(m, ROOT))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SHADERS:
        choices = "|".join(SHADERS)
        sys.exit(f"Usage: py -3 build_port.py <{choices}>")
    build(sys.argv[1])
