"""Build one canonical 1080p MiSTer shader preset pair.

Usage: py -3 build_port.py <guest|royale|kurozumi|lottes>

Emits only the runtime assets used by ``<shader>.ini`` and
``<shader> - TATE.ini``: H, adaptive V, interlace-safe V, one gamma LUT, one
1080p mask, and the two presets.  Lottes and both Royale families receive the
default-reference colour calibration after their source geometry is built.
Prints fitting/moire diagnostics;
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
import quantize
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
# Guest Advanced is built from guest.r's upstream release drop, which
# supersedes the libretro pin for this family; see its reference provenance for
# the one temporal afterglow delta outside MiSTer's static spatial pipeline.
GUEST_SOURCE = "crt-guest-advanced-2026-07-23-release1"

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
        "rgb_pareto": True,
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
        "source": ("historical Kuro preset+Grade @ 7f34fc7469ec "
                   "(Grade d5d58f8c8836) + upstream-fixed Royale bloom @ 3b0d6aa1d134"),
        "lut_channels": True,
        "gain": None,
        # Pixel-local is the default: every output pixel is scored separately.
        # The old horizontally-dithered pair was accurate only after averaging
        # partner pixels, while an individual pixel could be over 120 codes off.
        "mask_strategy": {"period": (1, 2), "kind": "mean"},
    },
    "lottes": {
        "module": "lottes_ref",
        "file_base": "CRT Lottes (Port)",
        "preset_base": "CRT Lottes",
        "shader_name": "crt-lottes (Timothy Lottes)",
        "source": "libretro/glsl-shaders @ 2b2c5ee3fd8e",
        "lut_channels": False,
        "gain": None,
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


def _lottes_mask_tokens(ref, y: int = 7, z: int = 14) -> list[list[str]]:
    """Exact all-code piecewise-sRGB calibration on the source 6x2 phase."""
    rows = []
    for py, source_row in enumerate(ref.MASK_TILE_LINEAR):
        row = []
        for px, pixel in enumerate(source_row):
            lit = int(np.argmax(pixel))
            row.append(fileio.encode_mask_token(4 >> lit, 16 + y, z))
        rows.append(row)
    return rows


def _fit_lottes_fixed_v_exact(ref, h: np.ndarray,
                              tokens: list[list[str]],
                              mask_target: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Source-shaped fixed V refit against every 8-bit masked sRGB code.

    Only each phase's DC sum is searched.  Tap ratios remain the exact
    Tri+Bloom per-line decomposition, preventing a flat-field objective from
    inventing a non-source beam merely to exploit fixed-point truncation.
    """
    codes = np.arange(256, dtype=np.int64)
    h_levels = np.array([int(mm.fir_1d(
        np.full(16, int(code), dtype=np.int64), h, np.array([8.0]))[0])
        for code in codes], dtype=np.int64)
    taps = np.repeat(h_levels[:, None], 4, axis=1)
    mask = fileio.MaskFile([], len(tokens[0]), len(tokens), tokens)
    mask16 = np.round(mask.multipliers() * 16.0).astype(np.int64)
    table = np.zeros((256, 4), dtype=np.int64)
    baseline = np.zeros((256, 4), dtype=np.int64)

    for phase in range(129):
        f = phase / 256.0
        distances = np.array([f + 1.0, f, 1.0 - f, 2.0 - f])
        shape = np.array([max(ref.beam_weight(float(d), 0.5), 0.0)
                          for d in distances], dtype=np.float64)
        shape /= shape.sum()
        # The retained fixed-table calibration is the pure-power starting point;
        # the exact search below corrects its piecewise-sRGB/fixed-point bias.
        base_sum = int(round(256.0 * 0.84395
                             * ref._vertical_gain(f) ** (1.0 / 2.2)))
        target = np.stack([fitting.masked_reference_tile(
            ref, np.array([f]), int(code), mask_target)[:, :, 0, :]
            for code in codes])

        def row_for_sum(total: int) -> np.ndarray:
            if phase != 128:
                return quantize.round_row_to_sum(shape * total, total)
            total += total % 2
            symmetric = 0.5 * (shape + shape[::-1])
            pair = quantize.round_row_to_sum(
                symmetric[:2] * total, total // 2)
            return np.array([pair[0], pair[1], pair[1], pair[0]],
                            dtype=np.int64)

        baseline[phase] = row_for_sum(base_sum)
        best = None
        for total in range(max(0, base_sum - 20), base_sum + 21):
            row = row_for_sum(total)
            c128 = np.repeat((row * 128)[None, :], len(codes), axis=0)
            vertical = mm._fir_accumulate(taps, c128)
            rgb = np.repeat(vertical[:, None], 3, axis=1)
            port = mm.mask_multiply(rgb[:, None, None, :], mask16[None, :, :, :])
            error = float(np.mean((port - target) ** 2))
            score = (error, abs(total - base_sum), total)
            if best is None or score < best[0]:
                best = (score, row)
        table[phase] = best[1]

    for phase in range(129, 256):
        table[phase] = table[256 - phase][::-1]
        baseline[phase] = baseline[256 - phase][::-1]
    return table, baseline


def _build_lottes(cfg: dict) -> None:
    ref = __import__(cfg["module"])
    fb, pb = cfg["file_base"], cfg["preset_base"]
    provenance = f"{cfg['shader_name']}; {cfg['source']}"
    h = fitting.fit_h(ref)
    mask_target = fitting.mask_encoded_tile(ref)
    tokens = _lottes_mask_tokens(ref)
    v, baseline_v = _fit_lottes_fixed_v_exact(ref, h, tokens, mask_target)
    identity = np.repeat(np.arange(256, dtype=np.int64)[:, None], 3, axis=1)
    all_codes = np.arange(256, dtype=np.int64)
    before, _ = fitting.rmse_exact_masked(
        ref, baseline_v, baseline_v, h, identity, tokens, mask_target,
        codes=all_codes, align=False)
    after, max_error = fitting.rmse_exact_masked(
        ref, v, v, h, identity, tokens, mask_target,
        codes=all_codes, align=False)
    if after >= before - 1e-9:
        v = baseline_v
        after = before
    print(f"[lottes] exact all-code V refit: {before:.3f} -> {after:.3f} "
          f"codes (max {max_error:.2f})")

    common = [f"Original shader: {provenance}",
              "Default Tri + 0.15*Bloom, true piecewise sRGB reference"]
    hpath = os.path.join(ROOT, "Filters", f"{fb}_H.txt")
    fileio.write_filter(hpath, fileio.FilterFile(
        [f"Name: {fb}_H"] + common +
        ["Source-derived separable LS blend: H3 .06003828, H5 .71920149, "
         "H7 .22076023"], False, True, [h]))
    vpath = os.path.join(ROOT, "Filters", f"{fb}_V.txt")
    fileio.write_filter(vpath, fileio.FilterFile(
        [f"Name: {fb}_V"] + common +
        ["Fixed source-shaped V; phase DC refit over all 256 input codes",
         "Pair with identity gamma (off) and the matching exact-sRGB mask"],
        False, True, [v]))
    mpath = os.path.join(ROOT, "Shadow_Masks", f"{fb}.txt")
    fileio.write_mask(mpath, fileio.MaskFile(
        [f"Name: {fb}", provenance,
         "Source-anchored 6x2 stretched-VGA phase (pixel 0,0 is green-lit)",
         "47e/27e/17e calibrated over all 256 codes against linear mask + piecewise sRGB",
         f"Pair with {fb}_H and {fb}_V; gamma must be off"],
        6, 2, tokens))

    base = {"hfilter": f"{fb}_H.txt", "vfilter": f"{fb}_V.txt",
            "sfilter": "same", "ifilter": f"{fb}_V.txt", "gamma": "off",
            "mask": f"{fb}.txt", "maskmode": "1x"}
    fileio.write_preset(os.path.join(ROOT, "Presets", f"{pb}.ini"), base)
    fileio.write_preset(os.path.join(ROOT, "Presets", f"{pb} - TATE.ini"), {
        **base, "hfilter": f"{fb}_V.txt", "vfilter": f"{fb}_H.txt",
        "ifilter": f"{fb}_H.txt", "maskmode": "1x rotated"})
    print("[lottes] wrote H, fixed V, exact mask, and two presets")


def _build_kurozumi_bounded(cfg: dict) -> None:
    """Recalibrate the canonical stable Kuro tables without the generic BCD.

    The historical Grade oracle makes the generic full-period mask alternation
    needlessly expensive (millions of source-ordered bloom evaluations).  A
    bounded audit of direct Grade, scalarized, RGB-aware-V, and exact-LUT
    candidates found the existing nonzero three-channel warp plus stable V/mask
    to be the Pareto-safe seed.  Its only source regression was the modern
    Grade-era lifted first entry.  Keep the measured stable tables, force true
    historical code-zero, regenerate source-derived H/no-scanline tables, and
    hard-gate the projected physical 1x2 grille before writing anything.
    """
    ref = __import__(cfg["module"])
    fb, pb = cfg["file_base"], cfg["preset_base"]
    source = cfg["source"]
    provenance = f"{cfg['shader_name']}; {source}"

    h = fitting.fit_h(ref)
    v_cached = fileio.parse_filter(os.path.join(
        ROOT, "Filters", f"{fb}_V Adaptive.txt"))
    if not v_cached.adaptive or len(v_cached.sets) != 2:
        raise RuntimeError("kurozumi: cached stable V seed is missing or invalid")
    dark, bright = v_cached.dark, v_cached.bright
    lut = fileio.parse_gamma(os.path.join(ROOT, "Gamma", f"{fb}.txt"))
    mask_cached = fileio.parse_mask(os.path.join(
        ROOT, "Shadow_Masks", f"{fb}.txt"))
    tokens = mask_cached.tokens
    if lut.shape != (256, 3) or tokens != [["63a", "33a"]]:
        raise RuntimeError("kurozumi: canonical bounded LUT/mask seed changed")

    # The old 2023-Grade build lifted black to (5,2,2).  Preserve it only as a
    # deterministic before-control; the historical matched Grade maps black to
    # the intended/practical finite result (0,0,0).
    before_lut = lut.copy()
    before_lut[0] = (5, 2, 2)
    lut = lut.copy()
    lut[0] = 0

    phases = np.arange(0, 9, dtype=np.float64) / 16.0
    neutral_rgbs = np.repeat(fitting.EVAL_CODES[:, None], 3, axis=1)
    metric_kw = {"rgbs": neutral_rgbs, "phases": phases,
                 "reference_period": (1, 2)}
    neutral_before = fitting.rgb_masked_metrics(
        ref, dark, bright, h, before_lut, tokens, **metric_kw)
    neutral_after = fitting.rgb_masked_metrics(
        ref, dark, bright, h, lut, tokens, **metric_kw)
    rgb = fitting.rgb_masked_metrics(
        ref, dark, bright, h, lut, tokens,
        levels=np.array([0, 64, 128, 192, 255], dtype=np.int64),
        phases=phases, reference_period=(1, 2))
    moire = fitting.adaptive_moire_1080(dark, bright, lut)
    selector = fitting.selector_control_jump(h, dark, bright, lut)
    worst_flat = fitting.worst_flat_field_output(lut, dark, bright)
    m_target = fitting.mask_encoded_tile(ref)
    black_rms, black_max = fitting.rmse_exact_masked(
        ref, dark, bright, h, lut, tokens, m_target,
        codes=np.array([0], dtype=np.int64))

    if np.any(lut[0] != 0) or black_rms > 0.05 or black_max > 0.5:
        raise RuntimeError("kurozumi: historical zero-black gate failed")
    if not fitting.gamma_quality_ok(lut):
        raise RuntimeError(
            f"kurozumi: gamma quantization gate failed {fitting.gamma_stats(lut)}")
    if neutral_after["rms"] >= neutral_before["rms"] - 1.0e-6:
        raise RuntimeError(
            "kurozumi: bounded candidate did not improve historical neutral error")
    if (neutral_after["rms"] > 28.0 or neutral_after["max"] > 178.0
            or rgb["rms"] > 34.0 or rgb["max"] > 178.0
            or moire > MOIRE_LIMIT or selector > 2 or worst_flat > 255.5):
        raise RuntimeError(
            "kurozumi: bounded fidelity/stability gate failed: "
            f"neutral {neutral_after['rms']:.3f}/{neutral_after['max']:.1f}, "
            f"RGB {rgb['rms']:.3f}/{rgb['max']:.1f}, moire {moire:.2f}, "
            f"selector {selector}, clip {worst_flat:.2f}")

    print(f"[kurozumi] bounded historical candidate: neutral "
          f"{neutral_before['rms']:.3f}->{neutral_after['rms']:.3f}, "
          f"RGB {rgb['rms']:.3f}/{rgb['max']:.1f}, black "
          f"{black_rms:.3f}/{black_max:.1f}, moire {moire:.2f}, "
          f"selector {selector}, clip {worst_flat:.2f}/255")

    made = []
    gpath = os.path.join(ROOT, "Gamma", f"{fb}.txt")
    fileio.write_gamma(gpath, lut, header=[
        f"Name: {fb}", provenance,
        "Bounded historical Grade recalibration of the stable three-channel warp",
        "True code-zero black; projected neutral/RGB Pareto and 1080p moire gated",
        "Pair with the matching (Port) adaptive V and mask"])
    made.append(gpath)

    mpath = os.path.join(ROOT, "Shadow_Masks", f"{fb}.txt")
    fileio.write_mask(mpath, fileio.MaskFile([
        f"Name: {fb}", provenance,
        "Stable end-to-end 1x2 aperture-grille fit in MiSTer v2 arithmetic",
        "Selected against the historical Grade/Royale neutral and RGB oracle",
        "Pair only with the matching (Port) gamma and adaptive V"],
        len(tokens[0]), len(tokens), tokens))
    made.append(mpath)

    header_common = [f"Original shader: {provenance}",
                     "Fitted to MiSTer RTL arithmetic (see tools/DESIGN.md)"]
    vpath = os.path.join(ROOT, "Filters", f"{fb}_V Adaptive.txt")
    fileio.write_filter(vpath, fileio.FilterFile(
        [f"Name: {fb}_V Adaptive"] + header_common + [
            "Stable adaptive seed retained by bounded historical neutral/RGB gate",
            "Pair with the matching zero-black three-channel gamma and mask"],
        True, True, [dark, bright]),
        set_comments=["Primary coefficients (maximum RGB = 0)",
                      "Secondary coefficients (maximum RGB = 255)"])
    made.append(vpath)

    nos = fitting.no_scanline_table(ref)
    npath = os.path.join(ROOT, "Filters", f"{fb}_V No Scanlines.txt")
    fileio.write_filter(npath, fileio.FilterFile(
        [f"Name: {fb}_V No Scanlines"] + header_common + [
            "Scan-free vertical interpolation; interlace fallback"],
        False, True, [nos]))
    made.append(npath)

    hpath = os.path.join(ROOT, "Filters", f"{fb}_H.txt")
    fileio.write_filter(hpath, fileio.FilterFile(
        [f"Name: {fb}_H"] + header_common,
        False, True, [h]))
    made.append(hpath)

    def preset(filename: str, entries: dict[str, str]) -> None:
        path = os.path.join(ROOT, "Presets", filename)
        fileio.write_preset(path, entries)
        made.append(path)

    base = {
        "hfilter": f"{fb}_H.txt", "vfilter": f"{fb}_V Adaptive.txt",
        "sfilter": "same", "ifilter": f"{fb}_V No Scanlines.txt",
        "gamma": f"{fb}.txt", "mask": f"{fb}.txt", "maskmode": "1x",
    }
    preset(f"{pb}.ini", base)
    preset(f"{pb} - TATE.ini", {
        **base, "hfilter": f"{fb}_V Adaptive.txt", "vfilter": f"{fb}_H.txt",
        "ifilter": f"{fb}_H.txt", "maskmode": "1x rotated",
    })

    print(f"[kurozumi] wrote {len(made)} files")
    for path in made:
        print("   ", os.path.relpath(path, ROOT))


def build(key: str) -> None:
    # Kurozumi's historical branch uses its checked-in fitted tables as the
    # bounded seed.  Once those tables carry the colour calibration, rebuilding
    # the source seed would compound the blend; retain the calibrated result.
    if key == "kurozumi":
        import color_match
        if color_match.is_calibrated(key):
            print("[kurozumi] calibrated checked-in seed already current")
            return
    cfg = SHADERS[key]
    if key == "lottes":
        _build_lottes(cfg)
        return
    if key == "kurozumi":
        _build_kurozumi_bounded(cfg)
        return
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

        if cfg.get("rgb_pareto"):
            lut, dark, bright, rgb_info = fitting.fit_v_rgb_pareto(
                ref, h, tokens, baseline=(lut, dark, bright))
            print(f"[{key}] RGB Pareto fit: {rgb_info['label']}; "
                  f"RGB {rgb_info['rgb_rms']:.3f}/{rgb_info['rgb_max']:.1f}, "
                  f"neutral {rgb_info['neutral_rms']:.3f}/"
                  f"{rgb_info['neutral_max']:.1f}, "
                  f"moire {rgb_info['moire']:.2f}, "
                  f"clip {rgb_info['worst_flat']:.2f}/255")

    # ---- mask, refit against the finished filters ---------------------------
    # The isolated fit matched the shader's multipliers; now that the V/LUT are
    # known, refit so the mask minimizes the error of the PRODUCT (absorbing
    # clamp-order and filter residual). Self-gating: kept only if it measurably
    # wins, since where the isolated fit was already optimal this is a no-op.
    r_before, _ = fitting.rmse_exact_masked(ref, dark, bright, h, lut, tokens, m_target)
    dgroup = len(tokens[0]) // 2 if minfo["dithered"] else None
    tok_j, r_after = fitting.fit_mask_joint(ref, lut, dark, bright, h, tokens,
                                            m_target, dither_group=dgroup)
    if not cfg.get("rgb_pareto") and r_after < r_before - 0.05:
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
        if not cfg.get("rgb_pareto") and r_m < r_now - 0.05:
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

    # ---- final polish: wide-radius mask-aware LUT alternation -----------------
    # Runs AFTER any stability blend so the LUT is refined against the exact
    # tables that ship.  radius=8 searches a wider LUT neighbourhood than the
    # in-loop pass; the exact self-gate keeps only measured whole-pipeline wins
    # (2026-07-21 audit: adopted for Royale 19.968->18.989; Kurozumi now uses
    # the separate bounded historical branch above.)
    for it in range(8):
        r_now, _ = fitting.rmse_exact_masked(
            ref, dark, bright, h, lut, tokens, m_target)
        lut_p = fitting.refine_lut_masked(
            ref, lut, h, dark, bright, tokens, m_target, radius=8,
            channel_aware=cfg["lut_channels"])
        if fitting.worst_flat_field_output(lut_p, dark, bright) > 255.5:
            print(f"[{key}] polish pass {it + 1} rejected: flat-field clipping")
            break
        if not fitting.gamma_quality_ok(lut_p):
            print(f"[{key}] polish pass {it + 1} rejected: gamma quantization "
                  f"{fitting.gamma_stats(lut_p)}")
            break
        tok_p, r_p = fitting.fit_mask_joint(
            ref, lut_p, dark, bright, h, tokens, m_target)
        if not cfg.get("rgb_pareto") and r_p < r_now - 0.05:
            print(f"[{key}] polish pass {it + 1}: masked RMSE {r_now:.3f} -> "
                  f"{r_p:.3f} codes -- adopted")
            lut, tokens = lut_p, tok_p
        else:
            print(f"[{key}] polish pass {it + 1}: {r_p:.3f} vs {r_now:.3f} "
                  "-- kept")
            break

    if cfg.get("rgb_pareto"):
        selector = fitting.selector_control_jump(h, dark, bright, lut)
        if selector > 16:
            dark, bright, selector_info = fitting.constrain_selector_phase128(
                dark, bright, h, lut, limit=16)
            selector = selector_info["selector"]
            print(f"[{key}] phase-128 selector fit: "
                  f"deltas {selector_info['delta_dark']:+d}/"
                  f"{selector_info['delta_bright']:+d}, jump {selector} codes")
        rgb_final = fitting.rgb_masked_metrics(
            ref, dark, bright, h, lut, tokens)
        neutral_final, neutral_max = fitting.rmse_exact_masked(
            ref, dark, bright, h, lut, tokens, m_target)
        if (rgb_final["rms"] > 6.25 or rgb_final["max"] > 24.0
                or neutral_final > 2.5 or neutral_max > 10.0
                or selector > 16):
            raise RuntimeError(
                f"{key}: final RGB/neutral/selector self-gate failed: "
                f"RGB {rgb_final['rms']:.3f}/{rgb_final['max']:.1f}, "
                f"neutral {neutral_final:.3f}/{neutral_max:.1f}, "
                f"selector {selector}")
        print(f"[{key}] final RGB cube: RMSE {rgb_final['rms']:.3f}, "
              f"max {rgb_final['max']:.1f}, neutral {neutral_final:.3f}/"
              f"{neutral_max:.1f}, selector {selector}")

    gain_note = ("RGB-cube Pareto gamma/V fit with shared max-RGB control"
                 if cfg.get("rgb_pareto") else
                 (f"Gain-split: LUT carries the pre-clip transfer / {gain:.3f}; "
                  f"the mask carries {gain:.3f}" if gain else
                  "Monotone adaptive-control warp co-optimized with the V fit"))
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
    key = sys.argv[1]
    build(key)
    if key in ("lottes", "royale", "kurozumi"):
        import color_match
        color_match.apply(key)
