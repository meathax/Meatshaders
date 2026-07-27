"""Fit MiSTer coefficient tables / gamma LUTs / masks to a shader reference module.

A reference module (see DESIGN.md) provides:
    transfer(x[, channel])   encoded->encoded beam-center transfer
    beam_weight(d, L)        linear beam contribution at distance d source lines,
                             local encoded brightness L
    ref_vertical(f, x)       encoded output at vertical fraction f (0..0.5+),
                             uniform encoded input x, mask off
    h_kernel(frac)           [(tap_offset, weight), ...] at fractional offset
    mask_spec()              dict describing the rendered mask tile at 1080p

Fitting strategy (DESIGN.md): LUT = transfer -> V-row targets are
ref_vertical/transfer <= 1 (never clips); per-phase per-tap least squares of the
two adaptive endpoints against the hardware's linear blend over the gray ramp.
"""

from __future__ import annotations

import numpy as np

import mister_model as mm
import quantize

PHASES = 256
LUMA_W = np.array([0.2126, 0.7152, 0.0722])
# Preserve the dense near-black region: historical Kurozumi has an explicit
# RGBA8 Grade quantization boundary and true code-zero, so starting every
# objective at code 8 can hide a visible toe/black regression.  The remainder
# keeps the 4-code sampling cost while codes 253..255 cover peak white.
EVAL_CODES = np.concatenate((np.arange(0, 8), np.arange(8, 256, 4),
                             np.arange(253, 256)))
RGB_EVAL_LEVELS = np.array([0, 32, 64, 96, 128, 160, 192, 224, 255],
                           dtype=np.int64)


def gamma_stats(lut: np.ndarray) -> dict[str, int]:
    """Tone-smoothness statistics of a (256, 3) LUT.

    Mirrors the release gate in validate_port.py: a LUT that collapses unique
    levels or grows long plateaus / large steps scores better on flat-field
    RMSE while banding on real gradients, so refinement passes must respect
    the same bound the release does (unique >= 128, plateau <= 16, step <= 8).
    """
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


def gamma_quality_ok(lut: np.ndarray) -> bool:
    stats = gamma_stats(lut)
    return (stats["unique"] >= 128 and stats["longest_plateau"] <= 16
            and stats["max_step"] <= 8)


def _transfer(ref, x: float, channel: str | None = None) -> float:
    if channel is not None:
        try:
            return ref.transfer(x, channel)
        except TypeError:
            pass
    return ref.transfer(x)


def _ref_vertical(ref, f: float, x: float,
                  channel: str | None = None) -> float:
    # Profiles are symmetric about f=0.5; some modules only cover 0..0.5.
    f = min(f, 1.0 - f) if f > 0.5 else f
    if channel is not None:
        try:
            return ref.ref_vertical(f, x, channel)
        except TypeError:
            pass
    return ref.ref_vertical(f, x)


def _ref_vertical_unclipped(ref, f: float, x: float,
                            channel: str | None = None) -> float:
    """Pre-clip beam value B(f, x); falls back to the clipped value."""
    fn = getattr(ref, "ref_vertical_unclipped", None)
    if fn is None:
        return _ref_vertical(ref, f, x, channel)
    f = min(f, 1.0 - f) if f > 0.5 else f
    if channel is not None:
        try:
            return fn(f, x, channel)
        except TypeError:
            pass
    return fn(f, x)


def _transfer_unclipped(ref, x: float) -> float:
    fn = getattr(ref, "transfer_unclipped", None)
    return fn(x) if fn is not None else _transfer(ref, x)


def has_headroom(ref) -> bool:
    """True when the shader's beam-centre transfer clips before x=1.

    Such shaders (Lottes brightBoost/bloom gain, Easymode BRIGHT_BOOST,
    Kurozumi levels_contrast) cannot use the plain LUT-carries-transfer
    factorization without saturating the adaptive control — see fit_gain_split.
    """
    return _transfer_unclipped(ref, 1.0) > 1.0 + 1e-9


# ------------------------------------------------------------------ gamma

def build_lut(ref, channels: bool = False) -> np.ndarray:
    """(256, 3) LUT = round(255 * transfer(x)), forced monotone non-decreasing."""
    out = np.zeros((256, 3), dtype=np.int64)
    for c, name in enumerate("rgb"):
        vals = np.array([255.0 * _transfer(ref, i / 255.0, name if channels else None)
                         for i in range(256)])
        vals = np.maximum.accumulate(vals)            # monotone (smooths e.g. royale's
        out[:, c] = np.clip(np.round(vals), 0, 255)   # 1.2% brightpass dip)
    return out


def optimize_lut(ref, channels: bool = False, iters: int = 8) -> np.ndarray:
    """Co-optimize the gamma LUT warp with the adaptive endpoint fit.

    The no-clipping constraint only needs LUT >= transfer; anything above it
    rescales the V-row targets (flat-field peaks stay exact because
    output = LUT * row/256). The warp also *is* the adaptive control value,
    so we choose it to make the target family as linear-in-control as
    possible: alternate (a) per-phase 2-endpoint LS given the warp with
    (b) per-x grid search of the warp given the endpoints, then project onto
    monotone >= transfer.

    Returns a (256, 3) integer LUT (per-channel scaled for 3-channel refs).
    """
    xs_i = EVAL_CODES
    xs = xs_i / 255.0
    fs = np.arange(0, 33) / 64.0
    refv = np.array([[_ref_vertical(ref, f, x) for x in xs] for f in fs])   # (33, NX)
    tr = np.array([max(_transfer(ref, x), 1e-9) for x in xs])
    u = tr.copy()
    cand = np.linspace(0.0, 1.0, 41)                    # per-x search grid (rel.)
    for _ in range(iters):
        u = np.minimum(np.maximum.accumulate(np.maximum(u, tr)), 1.0)
        lum = np.round(u * 255.0)
        design = np.stack([(256.0 - lum) / 256.0, lum / 256.0], axis=1)
        ab = np.zeros((len(fs), 2))
        for i in range(len(fs)):
            t_rows = np.minimum(256.0 * refv[i] / u, 256.0)
            sol, *_ = np.linalg.lstsq(design, t_rows, rcond=None)
            ab[i] = np.clip(sol, 0.0, 256.0)
        for k in range(len(xs)):
            uk = tr[k] + cand * (1.0 - tr[k])           # candidates in [transfer, 1]
            lumk = np.round(uk * 255.0)
            blend = (ab[:, 0:1] * (256.0 - lumk) + ab[:, 1:2] * lumk) / 256.0
            out = uk * np.minimum(blend, 256.0) / 256.0        # (33, NC)
            err = ((out - refv[:, k:k + 1]) ** 2).sum(axis=0)
            u[k] = uk[np.argmin(err)]
    u = np.minimum(np.maximum.accumulate(np.maximum(u, tr)), 1.0)

    # Expand to a full 256-entry LUT; per-channel via the channel/neutral ratio.
    # Anchor x=0 on the real transfer: np.interp flat-extrapolates below xs[0],
    # which would otherwise lift black to the xs[0] level (a visible black lift).
    xfull = np.arange(256) / 255.0
    ufull = np.interp(xfull, np.concatenate([[0.0], xs]),
                      np.concatenate([[_transfer(ref, 0.0)], u]))
    ufull = np.minimum(np.maximum.accumulate(
        np.maximum(ufull, [_transfer(ref, x) for x in xfull])), 1.0)
    out = np.zeros((256, 3), dtype=np.int64)
    for c, name in enumerate("rgb"):
        if channels:
            ratio = np.array([_transfer(ref, x, name) / max(_transfer(ref, x), 1e-9)
                              for x in xfull])
            vals = 255.0 * np.clip(ufull * ratio, 0.0, 1.0)
        else:
            vals = 255.0 * ufull
        out[:, c] = np.clip(np.round(np.maximum.accumulate(vals)), 0, 255)
    return out


def simulate_flat(dark: np.ndarray, bright: np.ndarray, h: np.ndarray,
                  lut: np.ndarray, code: int, fs: np.ndarray) -> np.ndarray:
    """Exact-model output codes for a uniform field of `code` over phases fs."""
    lines = np.empty(16, dtype=np.int64)
    g = int(lut[code, 1])
    lines.fill(g)
    hout = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
    lines.fill(int(lut[code].max()))
    hctrl = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
    lines.fill(hout)
    return mm.fir_1d_adaptive(lines, dark, bright, 8.0 + fs,
                              np.full(len(fs), hctrl, dtype=np.int64))


def rmse_exact(ref, dark: np.ndarray, bright: np.ndarray, h: np.ndarray,
               lut: np.ndarray, codes: np.ndarray | None = None) -> float:
    """End-to-end RMSE (output codes) vs the reference, mask off, exact model."""
    if codes is None:
        codes = EVAL_CODES
    fs = np.arange(0, 33) / 64.0
    errs = []
    for code in codes:
        sim = simulate_flat(dark, bright, h, lut, int(code), fs)
        tgt = np.array([255.0 * _ref_vertical(ref, f, code / 255.0) for f in fs])
        errs.append(sim - tgt)
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e * e)))


def rmse_exact_rgb(ref, dark: np.ndarray, bright: np.ndarray, h: np.ndarray,
                   lut: np.ndarray, codes: np.ndarray | None = None) -> float:
    """Channel-aware mask-off RMSE for references with RGB-specific grading."""
    if codes is None:
        codes = EVAL_CODES
    fs = np.arange(0, 33) / 64.0
    errs = []
    for code in codes:
        sim = simulate_flat_rgb(dark, bright, h, lut, int(code), fs)
        x = int(code) / 255.0
        target = np.array([[255.0 * _ref_vertical(ref, f, x, channel)
                            for channel in "rgb"] for f in fs])
        errs.append(sim - target)
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e * e)))


def rgb_cube(levels: np.ndarray | None = None,
             exclude_black: bool = True) -> np.ndarray:
    """Return a deterministic encoded-RGB cube as an ``(N, 3)`` code array."""
    if levels is None:
        levels = RGB_EVAL_LEVELS
    levels = np.asarray(levels, dtype=np.int64)
    if levels.ndim != 1 or not len(levels) or np.any((levels < 0) | (levels > 255)):
        raise ValueError("RGB levels must be a non-empty 1-D array in 0..255")
    colors = np.array(np.meshgrid(levels, levels, levels, indexing="ij"),
                      dtype=np.int64).reshape(3, -1).T
    if exclude_black:
        colors = colors[np.any(colors != 0, axis=1)]
    return colors


def rgb_masked_metrics(ref, dark: np.ndarray, bright: np.ndarray,
                       h: np.ndarray, lut: np.ndarray,
                       tokens: list[list[str]],
                       levels: np.ndarray | None = None,
                       phases: np.ndarray | None = None,
                       rgbs: np.ndarray | None = None,
                       align: bool = True,
                       mask_scale: int = 1,
                       reference_period: tuple[int, int] | None = None) -> dict:
    """Exact masked uniform-RGB fidelity against a source-side RGB evaluator.

    This is the chromatic companion to :func:`rmse_exact_masked`.  It exercises
    a full RGB cube instead of a neutral ramp, drives MiSTer's adaptive V filter
    from the shared post-H maximum RGB, and applies both masks in their native
    arithmetic/order.  The reference must publish ``ref_vertical_rgb(f, rgb,
    mask_mult)``; in particular, this lets Guest apply its linear-light mask
    before brightboost/glow instead of approximating it as an encoded multiply.

    Returns a dictionary with ``rms``, ``max``, ``roll``, ``colors``, and
    ``samples``.  The default validation domain is the endpoint-inclusive
    9-level cube (black excluded), 33 half-phases, both mask cells, and RGB.
    """
    import fileio

    rgb_fn = getattr(ref, "ref_vertical_rgb", None)
    if rgb_fn is None:
        raise AttributeError("reference does not publish ref_vertical_rgb")
    colors = rgb_cube(levels) if rgbs is None else np.asarray(rgbs, dtype=np.int64)
    if colors.ndim != 2 or colors.shape[1] != 3 \
            or np.any((colors < 0) | (colors > 255)):
        raise ValueError("rgbs must have shape (N, 3) with codes in 0..255")
    if phases is None:
        phases = np.arange(0, 33, dtype=np.float64) / 64.0
    phases = np.asarray(phases, dtype=np.float64)
    if phases.ndim != 1 or not len(phases) or np.any(~np.isfinite(phases)):
        raise ValueError("phases must be a non-empty finite 1-D array")
    if mask_scale not in (1, 2):
        raise ValueError(f"mask_scale must be 1 or 2, got {mask_scale}")

    linear_mask = mask_linear_tile(ref)
    if reference_period is not None:
        ph, pw = map(int, reference_period)
        if ph <= 0 or pw <= 0 or ph > linear_mask.shape[0] \
                or pw > linear_mask.shape[1]:
            raise ValueError("reference_period must fit inside the source mask")
        # Project tiny source-resize noise onto its physically visible cadence.
        # Kurozumi's 16x16 Lanczos tile is vertically flat and horizontally a
        # two-pixel R/B alternation at 1080p; averaging modulo (1,2) preserves
        # that exact macrostructure without scoring 256 near-duplicate cells.
        projected = np.zeros((ph, pw, 3), dtype=np.float64)
        counts = np.zeros((ph, pw, 1), dtype=np.float64)
        for yy in range(linear_mask.shape[0]):
            for xx in range(linear_mask.shape[1]):
                projected[yy % ph, xx % pw] += linear_mask[yy, xx]
                counts[yy % ph, xx % pw, 0] += 1.0
        linear_mask = projected / counts
    rh, rw, _ = linear_mask.shape
    mask = fileio.MaskFile([], len(tokens[0]), len(tokens), tokens)
    m16 = np.round(mask.multipliers() * 16.0).astype(np.int64)
    if mask_scale == 2:
        m16 = np.repeat(np.repeat(m16, 2, axis=0), 2, axis=1)
    mh, mw, _ = m16.shape
    super_h = int(np.lcm(mh, rh))
    super_w = int(np.lcm(mw, rw))
    sse = np.zeros((super_h, super_w), dtype=np.float64)

    def pair(rgb):
        simulated = simulate_uniform_rgb(dark, bright, h, lut, rgb, phases)
        port = mm.mask_multiply(simulated[None, None, :, :],
                                m16[:, :, None, :])
        port = np.tile(port, (super_h // mh, super_w // mw, 1, 1))
        encoded = np.asarray(rgb, dtype=np.float64) / 255.0
        target = np.empty((rh, rw, len(phases), 3), dtype=np.float64)
        for yy in range(rh):
            for xx in range(rw):
                target[yy, xx] = 255.0 * np.array([
                    rgb_fn(float(f), encoded, linear_mask[yy, xx])
                    for f in phases], dtype=np.float64)
        target = np.tile(target, (super_h // rh, super_w // rw, 1, 1))
        return port, target

    for rgb in colors:
        port, target = pair(rgb)
        fp = np.fft.fft2(port, axes=(0, 1))
        ft = np.fft.fft2(target, axes=(0, 1))
        corr = np.fft.ifft2(fp * np.conj(ft), axes=(0, 1)).real
        sse += ((port * port).sum() + (target * target).sum()
                - 2.0 * corr.sum(axis=(2, 3)))
    if align:
        # The source tile has only rh*rw distinct rigid rolls even when the
        # source/hardware LCM supercell is larger.
        unique = sse[:rh, :rw]
        dy, dx = np.unravel_index(int(np.argmin(unique)), unique.shape)
    else:
        dy, dx = 0, 0
    count = len(colors) * super_h * super_w * len(phases) * 3
    rms = float(np.sqrt(max(sse[dy, dx], 0.0) / count))

    max_error = 0.0
    for rgb in colors:
        port, target = pair(rgb)
        target = np.roll(target, (dy, dx), axis=(0, 1))
        max_error = max(max_error, float(np.abs(port - target).max()))
    return {"rms": rms, "max": max_error, "roll": (int(dy), int(dx)),
            "colors": int(len(colors)), "samples": int(count)}


def rmse_exact_masked(ref, dark: np.ndarray, bright: np.ndarray, h: np.ndarray,
                      lut: np.ndarray, tokens: list[list[str]],
                      mask_encoded: np.ndarray,
                      codes: np.ndarray | None = None,
                      align: bool = True,
                      mask_scale: int = 1) -> tuple[float, float]:
    """End-to-end RMSE THROUGH the mask: the only honest whole-pipeline metric.

    Mask-off RMSE silently assumes the port's mask equals the shader's mask, so
    it cannot see two real effects:
      * clamp ORDER — the shader clips clamp(B*m) (after the mask), MiSTer
        clamps the V stage and then saturates in the mask. Off-channels at
        white differ by tens of codes.
      * gain-split — when gain lives in the mask (G), a mask-off measurement is
        G times dark by construction and rises with G, which is backwards.
    Target: the reference module's exact ``ref_masked`` result when available.
    This matters for piecewise transfer functions such as sRGB, where applying a
    linear-light mask is signal-dependent in encoded space.  References without
    that hook retain the power-law-equivalent ``min(B*m_encoded, 1)`` path.

    align=True scores every rigid roll of the reference tile and keeps the best.
    A mask tile shifted by a few output pixels is the SAME mask to the eye (the
    pattern has no anchor the viewer can see), so a fixed-phase comparison would
    reject a perfect port purely for starting on a different phosphor stripe.

    `mask_scale` models MiSTer's 1x/2x mask mode.  In 2x mode each token is a
    2x2 block of output pixels (shadowmask.sv indexes hcount[4:1]/vcount[4:1]).

    Returns (rmse, max_abs_err) in output codes.
    """
    rmse, max_err, _ = _rmse_exact_masked_periodic(
        ref, dark, bright, h, lut, tokens, mask_encoded, codes=codes,
        align=align, mask_scale=mask_scale)
    return rmse, max_err


def masked_reference_tile(ref, fs: np.ndarray, code: int,
                          mask_encoded: np.ndarray) -> np.ndarray:
    """Return exact source output ``(mask_y, mask_x, phase, RGB)`` in codes.

    A source module may expose ``ref_masked(f, x, px, py, channel)`` to model
    the complete mask application and final transfer.  The fallback is exact
    for a pure-power output encode and preserves existing family behavior.
    ``mask_encoded`` still supplies the reference period and the fallback tile.
    """
    fs = np.asarray(fs, dtype=np.float64)
    tile = np.asarray(mask_encoded, dtype=np.float64)
    if tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError("mask_encoded must have shape (height, width, 3)")
    h_tile, w_tile, _ = tile.shape
    x = int(code) / 255.0
    hook = getattr(ref, "ref_masked", None)
    if hook is not None:
        return 255.0 * np.array([
            [[
                [hook(float(f), x, px, py, channel) for channel in "rgb"]
                for f in fs]
             for px in range(w_tile)]
            for py in range(h_tile)
        ], dtype=np.float64)
    beam = np.array([[_ref_vertical_unclipped(ref, f, x, channel)
                      for channel in "rgb"] for f in fs])
    return 255.0 * np.minimum(tile[:, :, None, :] * beam[None, None, :, :], 1.0)


def _rmse_exact_masked_periodic(
        ref, dark: np.ndarray, bright: np.ndarray, h: np.ndarray,
        lut: np.ndarray, tokens: list[list[str]],
        mask_encoded: np.ndarray, codes: np.ndarray | None = None,
        align: bool = True, mask_scale: int = 1,
) -> tuple[float, float, tuple[int, int]]:
    """Exact masked metric over the complete periodic supercell.

    The hardware and reference masks can have different periods (Royale is
    12x6 in hardware versus 24x24 in the shader).  Comparing only the hardware
    tile silently ignores reference cells outside its upper-left corner.  The
    honest domain is lcm(period_hw, period_ref) on each axis.  Circular
    correlation finds the best rigid reference-tile alignment without an
    O(number_of_rolls * number_of_pixels) loop.

    Returns (rmse, max_abs_err, best_reference_roll).
    """
    import fileio
    if codes is None:
        codes = EVAL_CODES
    codes = np.asarray(codes, dtype=np.int64)
    if mask_scale not in (1, 2):
        raise ValueError(f"mask_scale must be 1 or 2, got {mask_scale}")
    fs = np.arange(0, 33) / 64.0
    h_tile, w_tile, _ = mask_encoded.shape
    mask = fileio.MaskFile([], len(tokens[0]), len(tokens), tokens)
    m16 = np.round(mask.multipliers() * 16.0).astype(np.int64)   # (h, w, 3)
    if mask_scale == 2:
        m16 = np.repeat(np.repeat(m16, 2, axis=0), 2, axis=1)
    mh, mw, _ = m16.shape
    super_h = int(np.lcm(mh, h_tile))
    super_w = int(np.lcm(mw, w_tile))
    lines = np.empty(16, dtype=np.int64)

    # Accumulate the exact spatial SSE for every rigid roll of the reference
    # tile.  For P and T, ||P-roll(T)||^2 differs only in the cross term, which
    # a 2-D circular correlation obtains for every roll at once.
    sse = np.zeros((super_h, super_w), dtype=np.float64)
    for code in codes:
        x = int(code) / 255.0
        ctrl_in = int(lut[int(code)].max())
        lines.fill(ctrl_in)
        hctrl = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
        ctrl = np.full(len(fs), hctrl, dtype=np.int64)
        per_ch = []
        for c in range(3):
            g = int(lut[int(code), c])
            lines.fill(g)
            hout = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
            lines.fill(hout)
            per_ch.append(mm.fir_1d_adaptive(lines, dark, bright, 8.0 + fs, ctrl))
        v = np.stack(per_ch, axis=1)                             # (F, 3)
        port = mm.mask_multiply(v[None, None, :, :], m16[:, :, None, :])
        port = np.tile(port, (super_h // mh, super_w // mw, 1, 1))

        target = masked_reference_tile(ref, fs, int(code), mask_encoded)
        target = np.tile(target,
                         (super_h // h_tile, super_w // w_tile, 1, 1))

        fp = np.fft.fft2(port, axes=(0, 1))
        ft = np.fft.fft2(target, axes=(0, 1))
        corr = np.fft.ifft2(fp * np.conj(ft), axes=(0, 1)).real
        sse += ((port * port).sum() + (target * target).sum()
                - 2.0 * corr.sum(axis=(2, 3)))

    # The reference has only h_tile*w_tile unique rigid rolls even when the
    # LCM supercell is larger than the reference period.
    if align:
        unique = sse[:h_tile, :w_tile]
        dy, dx = np.unravel_index(int(np.argmin(unique)), unique.shape)
    else:
        dy, dx = 0, 0
    count = len(codes) * super_h * super_w * len(fs) * 3
    rmse = float(np.sqrt(max(sse[dy, dx], 0.0) / count))

    # Re-simulate only the winning alignment to obtain the true L-infinity
    # error; it cannot be recovered from the correlation sums.
    max_err = 0.0
    for code in codes:
        x = int(code) / 255.0
        ctrl_in = int(lut[int(code)].max())
        lines.fill(ctrl_in)
        hctrl = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
        ctrl = np.full(len(fs), hctrl, dtype=np.int64)
        per_ch = []
        for c in range(3):
            lines.fill(int(lut[int(code), c]))
            hout = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
            lines.fill(hout)
            per_ch.append(mm.fir_1d_adaptive(lines, dark, bright,
                                              8.0 + fs, ctrl))
        v = np.stack(per_ch, axis=1)
        port = mm.mask_multiply(v[None, None, :, :], m16[:, :, None, :])
        port = np.tile(port, (super_h // mh, super_w // mw, 1, 1))
        target = masked_reference_tile(ref, fs, int(code), mask_encoded)
        target = np.tile(target,
                         (super_h // h_tile, super_w // w_tile, 1, 1))
        target = np.roll(target, (dy, dx), axis=(0, 1))
        max_err = max(max_err, float(np.abs(port - target).max()))
    return rmse, max_err, (int(dy), int(dx))


def refine_lut_masked(ref, lut: np.ndarray, h: np.ndarray, dark: np.ndarray,
                      bright: np.ndarray, tokens: list[list[str]],
                      mask_encoded: np.ndarray, radius: int = 6,
                      channel_aware: bool = False) -> np.ndarray:
    """Model-in-the-loop LUT refinement for the GAIN-SPLIT path.

    refine_lut_exact minimizes the mask-OFF error, which for a gain-split build
    is the wrong objective — it drags the LUT back toward carrying the clipped
    transfer and undoes the split. This variant scores each candidate LUT entry
    through the mask against 255*min(B*m, 1), the same target rmse_exact_masked
    uses, so refinement and evaluation agree.
    """
    import fileio
    fs = np.arange(0, 33) / 64.0
    h_tile, w_tile, _ = mask_encoded.shape
    mask = fileio.MaskFile([], len(tokens[0]), len(tokens), tokens)
    m16 = np.round(mask.multipliers() * 16.0).astype(np.int64)
    mh, mw, _ = m16.shape

    # Score the same complete periodic domain as rmse_exact_masked.  The old
    # implementation paired only the upper-left hardware cell with the
    # upper-left reference cell; that silently optimized the wrong target when
    # the periods differ (notably Royale's 12x6 hardware tile versus its 24x24
    # shader mask).  Freeze the current best rigid alignment for this LUT pass,
    # then vectorize every LCM-supercell pairing.
    _, _, (dy, dx) = _rmse_exact_masked_periodic(
        ref, dark, bright, h, lut, tokens, mask_encoded)
    super_h = int(np.lcm(mh, h_tile))
    super_w = int(np.lcm(mw, w_tile))
    mults = np.empty((super_h * super_w, 3), dtype=np.int64)
    reference_cells = np.empty((super_h * super_w, 2), dtype=np.int64)
    k = 0
    for yy in range(super_h):
        for xx in range(super_w):
            mults[k] = m16[yy % mh, xx % mw]
            reference_cells[k] = ((yy - dy) % h_tile, (xx - dx) % w_tile)
            k += 1
    lines = np.empty(16, dtype=np.int64)
    out = lut.copy()
    for code in range(256):
        x = code / 255.0
        reference = masked_reference_tile(ref, fs, code, mask_encoded)
        targets = reference[reference_cells[:, 0], reference_cells[:, 1]]
        base = lut[code].astype(np.int64)
        best = base.copy()

        def candidate_error(cand):
            lines.fill(int(cand.max()))
            hctrl = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
            ctrl = np.full(len(fs), hctrl, dtype=np.int64)
            per_ch = []
            for c in range(3):
                lines.fill(int(cand[c]))
                hout = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
                lines.fill(hout)
                per_ch.append(mm.fir_1d_adaptive(lines, dark, bright, 8.0 + fs, ctrl))
            v = np.stack(per_ch, axis=1)
            sim = mm.mask_multiply(v[None, :, :], mults[:, None, :])
            return float(((sim - targets) ** 2).sum())

        if channel_aware:
            # Coordinate descent over a true RGB LUT vector.  Kurozumi's P22
            # color transform cannot be represented by the old scalar
            # base+delta search, which forced every entry toward neutral gray.
            best_err = candidate_error(best)
            for _ in range(2):
                changed = False
                for ch in range(3):
                    start = int(best[ch])
                    local, local_err = best.copy(), best_err
                    for value in range(max(0, start - radius),
                                       min(255, start + radius) + 1):
                        cand = best.copy()
                        cand[ch] = value
                        err = candidate_error(cand)
                        if err < local_err:
                            local, local_err = cand, err
                    if local_err < best_err:
                        best, best_err, changed = local, local_err, True
                if not changed:
                    break
        else:
            best_err = np.inf
            for delta in range(-radius, radius + 1):
                cand = np.clip(base + delta, 0, 255)
                err = candidate_error(cand)
                if err < best_err:
                    best, best_err = cand, err
        out[code] = best
    for ch in range(3):
        out[:, ch] = np.maximum.accumulate(out[:, ch])
    return out


def refine_lut_exact(ref, lut: np.ndarray, h: np.ndarray, dark: np.ndarray,
                     bright: np.ndarray, radius: int = 6,
                     channel_aware: bool = False) -> np.ndarray:
    """Model-in-the-loop LUT refinement against the EXACT hardware arithmetic.

    optimize_lut works in ideal arithmetic, but MiSTer's FIR truncates: a
    unity-DC row returns g-1 (or worse where a fitted row sum lands below
    256), so every simulated level sits 1-4 codes under its ideal value.
    Rather than model that bias analytically, search it out: each input code's
    LUT entry affects only that code's outputs, so the problem is separable —
    for each code, simulate the whole phase grid through mister_model at
    candidate LUT values and keep the best. Monotonicity is restored after.

    Costs a few seconds per shader and removes the bias at its source (the
    control value shifts too, which the exact simulation accounts for).
    """
    fs = np.arange(0, 33) / 64.0
    pos = 8.0 + fs
    lines = np.empty(16, dtype=np.int64)
    out = lut.copy()
    for c in range(256):
        x = c / 255.0
        base = lut[c].astype(np.int64)
        best = base.copy()

        if channel_aware:
            target = np.array([[255.0 * _ref_vertical(ref, f, x, ch)
                                for ch in "rgb"] for f in fs])
        else:
            target = np.array([255.0 * _ref_vertical(ref, f, x) for f in fs])

        def candidate_error(cand):
            ctrl_in = int(cand.max())              # adaptive control = max RGB
            lines.fill(ctrl_in)
            hctrl = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
            ctrl = np.full(len(fs), hctrl, dtype=np.int64)
            if channel_aware:
                per_ch = []
                for ch in range(3):
                    lines.fill(int(cand[ch]))
                    hout = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
                    lines.fill(hout)
                    per_ch.append(mm.fir_1d_adaptive(
                        lines, dark, bright, pos, ctrl))
                sim = np.stack(per_ch, axis=1)
            else:
                lines.fill(int(cand[1]))
                hout = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
                lines.fill(hout)
                sim = mm.fir_1d_adaptive(lines, dark, bright, pos, ctrl)
            return float(((sim - target) ** 2).sum())

        if channel_aware:
            best_err = candidate_error(best)
            for _ in range(2):
                changed = False
                for ch in range(3):
                    start = int(best[ch])
                    local, local_err = best.copy(), best_err
                    for value in range(max(0, start - radius),
                                       min(255, start + radius) + 1):
                        cand = best.copy()
                        cand[ch] = value
                        err = candidate_error(cand)
                        if err < local_err:
                            local, local_err = cand, err
                    if local_err < best_err:
                        best, best_err, changed = local, local_err, True
                if not changed:
                    break
        else:
            best_err = np.inf
            for delta in range(-radius, radius + 1):
                cand = np.clip(base + delta, 0, 255)
                err = candidate_error(cand)
                if err < best_err:
                    best, best_err = cand, err
        out[c] = best
    for ch in range(3):
        out[:, ch] = np.maximum.accumulate(out[:, ch])
    return out


def mask_encoded_tile(ref) -> np.ndarray:
    """(h, w, 3) ENCODED per-channel mask multipliers as rendered at 1080p.

    Reference modules describe their mask differently (some publish encoded
    multipliers directly, some linear ones, some an amplified 0..255 tile), so
    normalize here rather than in every caller. Linear multipliers are
    converted with the shader's own output exponent where it publishes one —
    a mask applied in linear light before an encode of gamma g scales the
    encoded value by m**(1/g).
    """
    spec = ref.mask_spec()
    for key in ("encoded_equivalent_multipliers", "tile_encoded_multiplier"):
        if key in spec:
            tile = np.array(spec[key], dtype=np.float64)
            return tile[None, :, :] if tile.ndim == 2 else tile
    gamma = _output_gamma(ref)
    if "tile_rgb_0_255" in spec:                      # royale family
        rgb = np.array(spec["tile_rgb_0_255"], dtype=np.float64) / 255.0
        lin = np.clip(rgb * float(spec.get("mask_amplify", 1.0)), 0.0, None)
        return lin ** (1.0 / gamma)
    for key in ("linear_multipliers", "tile_linear_multiplier"):
        if key in spec:
            lin = np.array(spec[key], dtype=np.float64)
            if lin.ndim == 2:
                lin = lin[None, :, :]
            return np.clip(lin, 0.0, None) ** (1.0 / gamma)
    raise KeyError(f"mask_spec() has no recognized multiplier key: {sorted(spec)}")


def mask_linear_tile(ref) -> np.ndarray:
    """Return a reference mask as a native linear-light ``(h, w, 3)`` tile.

    RGB-aware references need this representation because mask ordering can be
    nonlinear: Guest applies the mask before shared brightboost and glow.  If a
    reference publishes only an encoded-equivalent tile, invert its declared
    output gamma as a conservative fallback.
    """
    spec = ref.mask_spec()
    for key in ("linear_multipliers", "tile_linear_multiplier"):
        if key in spec:
            tile = np.asarray(spec[key], dtype=np.float64)
            if tile.ndim == 2:
                tile = tile[None, :, :]
            if tile.ndim != 3 or tile.shape[2] != 3:
                raise ValueError(f"{key} must describe an (h, w, 3) RGB tile")
            return np.clip(tile, 0.0, None)
    if "tile_rgb_0_255" in spec:
        tile = (np.asarray(spec["tile_rgb_0_255"], dtype=np.float64) / 255.0
                * float(spec.get("mask_amplify", 1.0)))
        if tile.ndim == 2:
            tile = tile[None, :, :]
        return np.clip(tile, 0.0, None)
    for key in ("encoded_equivalent_multipliers", "tile_encoded_multiplier"):
        if key in spec:
            tile = np.asarray(spec[key], dtype=np.float64)
            if tile.ndim == 2:
                tile = tile[None, :, :]
            return np.clip(tile, 0.0, None) ** _output_gamma(ref)
    raise KeyError(f"mask_spec() has no recognized multiplier key: {sorted(spec)}")


def _output_gamma(ref) -> float:
    """The exponent the shader encodes its output with.

    A linear-light mask multiplier m scales the ENCODED value by m**(1/gamma),
    so this must be the shader's own output gamma — hardcoding 2.2 silently
    mis-scales any shader that encodes differently (Kurozumi's lcd_gamma is
    2.4: assuming 2.2 puts its red +2.4% and blue -4.9% off target).
    """
    try:
        d = ref.defaults()
    except Exception:
        d = {}
    for key in ("lcd_gamma", "gamma_out", "GAMMA_OUTPUT", "output_gamma"):
        if key in d and d[key]:
            return float(d[key])
    g = getattr(ref, "GAMMA_OUTPUT", None) or getattr(ref, "LCD_GAMMA", None)
    return float(g) if g else 2.2


def fit_gain_split(ref, mask_encoded: np.ndarray,
                   g_min: float = 1.0, g_max: float = 2.0) -> list[dict]:
    """Enumerate (mask token grid, gain G) candidates for the gain-split path.

    WHY: the gamma LUT is both the tone curve and the adaptive control. When a
    shader's beam-centre transfer clips before x=1, LUT pins at 255 over that
    band, the control pins with it, and every input there gets identical V rows
    while the true trough still varies — information destroyed before the V
    stage. Escape: scale the mask by G and have the LUT carry B(x)/G, which
    keeps rising (injective) as long as B(x)/G <= 1. Saturation then happens at
    the mask stage, which is also where the shader clips (clamp(B*m)), so the
    clamp ORDER matches too.

    Feasibility: a v2 token gives its selected channels (16+Y)/16 (<= 1.9375)
    and the others Z/16 (<= 0.9375), so G is bounded by the *dark* multipliers
    far more than the lit ones. Where G cannot reach max B(x), the damaged band
    shrinks rather than vanishing — still a large win.

    Returns candidates sorted by mask fidelity: dicts with keys
    tokens, gain, mask_rel_rmse, clip_x (input level where the LUT saturates,
    1.0 = never).
    """
    import fileio

    need = _transfer_unclipped(ref, 1.0)          # G >= this => never saturates
    out = []
    seen = set()
    for g in np.linspace(g_min, min(g_max, 2.0), 401):
        scaled = mask_encoded * g
        # Reject outright if any pixel is unrepresentable at this gain: the two
        # smallest channels in a pixel must fit under the shared 15/16 ceiling.
        feasible = True
        for row in scaled.reshape(-1, 3):
            if np.sort(row)[1] > 0.9375 + 0.03 or row.max() > 1.9375 + 0.06:
                feasible = False
                break
        if not feasible:
            continue
        tokens, _ = tile_from_encoded(scaled)
        key = tuple(tuple(r) for r in tokens)
        if key in seen:
            continue
        seen.add(key)
        h, w, _ = mask_encoded.shape
        dec = fileio.MaskFile([], w, h, tokens).multipliers()
        rel = float(np.sqrt(np.mean((dec / g - mask_encoded) ** 2)))
        # Where does LUT = 255*B(x)/g saturate?
        clip_x = 1.0
        if need > g:
            xs = np.linspace(0.0, 1.0, 512)
            over = [x for x in xs if _transfer_unclipped(ref, x) > g]
            clip_x = float(min(over)) if over else 1.0
        out.append({"tokens": tokens, "gain": float(g), "mask_rel_rmse": rel,
                    "clip_x": clip_x})
    out.sort(key=lambda c: (c["mask_rel_rmse"], -c["clip_x"]))
    return out


def build_lut_gain(ref, gain: float, channels: bool = False) -> np.ndarray:
    """LUT carrying the PRE-CLIP transfer divided by `gain`.

    Pairs with fit_v_adaptive_gain. Stays strictly increasing (so the adaptive
    control keeps resolving levels) until B(x)/gain exceeds 1.
    """
    xfull = np.arange(256) / 255.0
    u = np.array([min(_transfer_unclipped(ref, x) / gain, 1.0) for x in xfull])
    u = np.maximum.accumulate(u)
    lut = np.zeros((256, 3), dtype=np.int64)
    for c, name in enumerate("rgb"):
        if channels:
            ratio = np.array([_transfer(ref, x, name) / max(_transfer(ref, x), 1e-9)
                              for x in xfull])
            vals = 255.0 * np.clip(u * ratio, 0.0, 1.0)
        else:
            vals = 255.0 * u
        lut[:, c] = np.clip(np.round(np.maximum.accumulate(vals)), 0, 255)
    return lut


def fit_v_adaptive_gain(ref, lut: np.ndarray, gain: float,
                        xs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive V endpoints for the gain-split path.

    Target row sum = 256 * B(f,x) / B(0,x) capped at 256 (the peak is carried
    entirely by the LUT, so rows never need over-unity gain and never clip).
    """
    if xs is None:
        xs = EVAL_CODES / 255.0
    lums = np.array([lut[int(round(x * 255))].max() for x in xs], dtype=np.int64)
    rows = np.zeros((129, len(xs), 4))
    for p in range(129):
        f = p / 256.0
        d = np.array([f + 1.0, f, 1.0 - f, 2.0 - f])
        for k, x in enumerate(xs):
            peak = max(_transfer_unclipped(ref, x), 1e-9)
            # The LUT saturates once B/gain > 1; above that the peak it can
            # deliver is 255, so ask the rows for the ratio against what the
            # LUT actually carries, not against B(0,x).
            carried = min(peak / gain, 1.0) * gain
            s = 256.0 * min(_ref_vertical_unclipped(ref, f, x) / carried, 1.0)
            w = np.array([max(ref.beam_weight(di, x), 0.0) for di in d])
            if w.sum() < 1e-12:
                w = np.zeros(4)
                w[np.argmin(d)] = 1.0
            rows[p, k] = s * w / w.sum()
    ta = (256.0 - lums) / 256.0
    tb = lums / 256.0
    design = np.stack([ta, tb], axis=1)
    gram_inv = np.linalg.pinv(design.T @ design)
    dark_f = np.zeros((PHASES, 4))
    bright_f = np.zeros((PHASES, 4))
    for p in range(129):
        sol = gram_inv @ design.T @ rows[p]
        dark_f[p] = np.clip(sol[0], 0, 512)
        bright_f[p] = np.clip(sol[1], 0, 512)
    for p in range(129, PHASES):
        dark_f[p] = dark_f[PHASES - p][::-1]
        bright_f[p] = bright_f[PHASES - p][::-1]
    tables = []
    for fl in (dark_f, bright_f):
        fl = np.clip(fl, 0, None)
        sums = fl.sum(axis=1)
        over = sums > 256.0
        fl[over] *= (256.0 / sums[over])[:, None]
        target_sums = np.round(fl.sum(axis=1)).astype(np.int64)
        for p in range(1, 128):
            target_sums[PHASES - p] = target_sums[p]
        tables.append(quantize.quantize_symmetric(fl, target_sums))
    return tables[0], tables[1]


# ------------------------------------------------------------------ V filters

def _tap_targets(ref, lut_ctrl: np.ndarray, xs: np.ndarray):
    """Ideal per-phase, per-x 4-tap rows (float, units of 1/256 gain).

    Returns rows (129, NX, 4) for phases 0..128 and ctrl lum values (NX,).
    Row sum = 256 * ref_vertical(f, x) / (LUT(x)/255) — normalizing by the
    actual LUT (which may sit above transfer(x); see optimize_lut) keeps
    flat-field peaks exact and rows <= 256 by construction; shape follows
    beam_weight tap ratios.
    """
    nx = len(xs)
    rows = np.zeros((129, nx, 4))
    lums = np.array([lut_ctrl[int(round(x * 255))] for x in xs], dtype=np.int64)
    tr = np.maximum(lums / 255.0, 1e-9)
    for p in range(129):
        f = p / 256.0
        d = np.array([f + 1.0, f, 1.0 - f, 2.0 - f])
        for k, x in enumerate(xs):
            s_target = 256.0 * min(_ref_vertical(ref, f, x) / tr[k], 1.0)
            w = np.array([max(ref.beam_weight(di, x), 0.0) for di in d])
            if w.sum() < 1e-12:
                w = np.zeros(4)
                w[np.argmin(d)] = 1.0
            rows[p, k] = s_target * w / w.sum()
    return rows, lums


def fit_v_adaptive(ref, lut_ctrl: np.ndarray,
                   xs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit the two adaptive endpoint tables. Returns (dark, bright) int tables.

    For each phase and tap, solve min over (A, B) of
        sum_x [ ((256-lum)A + lum*B)/256 - target_tap(x) ]^2
    then cap row sums at 256 and quantize with exact symmetry.
    """
    if xs is None:
        xs = EVAL_CODES / 255.0
    rows, lums = _tap_targets(ref, lut_ctrl, xs)
    ta = (256.0 - lums) / 256.0
    tb = lums / 256.0
    design = np.stack([ta, tb], axis=1)                       # (NX, 2)
    gram_inv = np.linalg.pinv(design.T @ design)
    dark_f = np.zeros((PHASES, 4))
    bright_f = np.zeros((PHASES, 4))
    for p in range(129):
        sol = gram_inv @ design.T @ rows[p]                   # (2, 4)
        dark_f[p] = np.clip(sol[0], -64, 512)
        bright_f[p] = np.clip(sol[1], -64, 512)
    for p in range(129, PHASES):
        dark_f[p] = dark_f[PHASES - p][::-1]
        bright_f[p] = bright_f[PHASES - p][::-1]

    tables = []
    for fl in (dark_f, bright_f):
        fl = np.clip(fl, 0, None)                             # beams are non-negative
        sums = fl.sum(axis=1)
        over = sums > 256.0
        fl[over] *= (256.0 / sums[over])[:, None]             # cap: no scaler clipping
        target_sums = np.round(fl.sum(axis=1)).astype(np.int64)
        for p in range(1, 128):
            target_sums[PHASES - p] = target_sums[p]
        tables.append(quantize.quantize_symmetric(fl, target_sums))
    return tables[0], tables[1]


def _endpoint_shapes(ref, l_dark: float = 0.15, l_bright: float = 1.0) -> np.ndarray:
    """(2, 129, 4) normalized 4-tap beam shapes per phase for both endpoints."""
    sh = np.zeros((2, 129, 4))
    for j, L in enumerate((l_dark, l_bright)):
        for p in range(129):
            f = p / 256.0
            d = np.array([f + 1.0, f, 1.0 - f, 2.0 - f])
            w = np.array([max(ref.beam_weight(di, L), 0.0) for di in d])
            if w.sum() < 1e-12:
                w = np.zeros(4)
                w[np.argmin(d)] = 1.0
            sh[j, p] = w / w.sum()
    return sh


def _joint_features(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(g, dtype=np.float64) / 256.0
    return v - v * v, v * v


def _solve_ab(O: np.ndarray, g: np.ndarray, cap_a: np.ndarray, cap_b: np.ndarray,
              n: int = 49, refine: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Per-phase box-constrained 2-var fit. The clip makes it non-convex, so
    use a vectorized 2-D zoom grid rather than a normal-equation solve."""
    p, q = _joint_features(g)
    nf = O.shape[0]
    lo_a, hi_a = np.zeros(nf), cap_a.copy()
    lo_b, hi_b = np.zeros(nf), cap_b.copy()
    A = np.zeros(nf)
    B = np.zeros(nf)
    for _ in range(refine + 1):
        t = np.linspace(0, 1, n)
        ga = lo_a[:, None] + (hi_a - lo_a)[:, None] * t[None, :]
        gb = lo_b[:, None] + (hi_b - lo_b)[:, None] * t[None, :]
        pred = np.minimum(ga[:, :, None, None] * p[None, None, None, :]
                          + gb[:, None, :, None] * q[None, None, None, :], 255.0)
        d = pred - O[:, None, None, :]
        c = np.einsum("finx,finx->fin", d, d)
        ia, ib = np.unravel_index(c.reshape(nf, -1).argmin(axis=1), (n, n))
        A = ga[np.arange(nf), ia]
        B = gb[np.arange(nf), ib]
        da = (hi_a - lo_a) / (n - 1) * 2
        db = (hi_b - lo_b) / (n - 1) * 2
        lo_a, hi_a = np.maximum(0, A - da), np.minimum(cap_a, A + da)
        lo_b, hi_b = np.maximum(0, B - db), np.minimum(cap_b, B + db)
    return A, B


def _solve_ab_linear_box(O: np.ndarray, p: np.ndarray, q: np.ndarray,
                         weights: np.ndarray, cap_a: np.ndarray,
                         cap_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted two-feature least squares with exact per-phase box bounds.

    The RGB fit has independent signal and shared-control levels, so its
    features are not reducible to :func:`_joint_features`.  With clipping
    excluded by the caller's behavioural gate, the objective is convex.  Its
    optimum is either the unconstrained solution or lies on one of four box
    edges; enumerating those candidates is exact and much cheaper than a grid.
    """
    O = np.asarray(O, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if O.ndim != 2 or p.shape != (O.shape[1],) or q.shape != p.shape \
            or weights.shape != p.shape:
        raise ValueError("O/features/weights have incompatible shapes")
    wp, wq = weights * p, weights * q
    pp, pq, qq = float(wp @ p), float(wp @ q), float(wq @ q)
    determinant = pp * qq - pq * pq
    A = np.empty(O.shape[0], dtype=np.float64)
    B = np.empty(O.shape[0], dtype=np.float64)
    for phase in range(O.shape[0]):
        target = O[phase]
        po, qo = float(wp @ target), float(wq @ target)
        candidates = []
        if determinant > 1.0e-15:
            au = (po * qq - qo * pq) / determinant
            bu = (qo * pp - po * pq) / determinant
            if 0.0 <= au <= cap_a[phase] and 0.0 <= bu <= cap_b[phase]:
                candidates.append((au, bu))
        for av in (0.0, float(cap_a[phase])):
            bv = np.clip((qo - av * pq) / max(qq, 1.0e-15),
                         0.0, cap_b[phase])
            candidates.append((av, float(bv)))
        for bv in (0.0, float(cap_b[phase])):
            av = np.clip((po - bv * pq) / max(pp, 1.0e-15),
                         0.0, cap_a[phase])
            candidates.append((float(av), bv))
        costs = [float(weights @ ((av * p + bv * q - target) ** 2))
                 for av, bv in candidates]
        A[phase], B[phase] = candidates[int(np.argmin(costs))]
    return A, B


def _dp_g(O: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """EXACT optimal monotone integer warp given (A, B).

    The cost is separable across x; the only coupling is the monotone ordering,
    so a prefix-min dynamic program over (x, g) is globally optimal — no search
    heuristics needed for this block.
    """
    nx = O.shape[1]
    p, q = _joint_features(np.arange(256))
    P = np.minimum(np.outer(A, p) + np.outer(B, q), 255.0)
    cost = ((P[:, None, :] - O[:, :, None]) ** 2).sum(axis=0)          # (NX, 256)
    dp = np.empty((nx, 256))
    arg = np.zeros((nx, 256), dtype=np.int32)
    dp[0] = cost[0]
    for k in range(1, nx):
        run = np.minimum.accumulate(dp[k - 1])
        # arg[k][j] = argmin of dp[k-1] over the prefix 0..j. Positions where
        # dp equals the running min are the candidates; accumulate the LATEST
        # such index (taking the earliest instead would pin arg to 0, since
        # position 0 always ties its own prefix min, and collapse the warp).
        arg[k] = np.maximum.accumulate(
            np.where(dp[k - 1] <= run, np.arange(256), 0)).astype(np.int32)
        dp[k] = cost[k] + run
    g = np.zeros(nx, dtype=np.int64)
    j = int(np.argmin(dp[nx - 1]))
    g[nx - 1] = j
    for k in range(nx - 1, 0, -1):
        j = int(arg[k][j])
        g[k - 1] = j
    return g


def fit_v_joint_safe(ref, channels: bool = False, limit: float = 255.5,
                     caps=(256.0, 252.0, 248.0, 244.0, 240.0, 232.0)):
    """fit_v_joint with the no-flat-field-clipping property ENFORCED.

    The dark cap can be raised to the 10-bit ceiling safely, but the bright cap
    genuinely can push the blended row over 255 at high levels. Rather than
    assume a cap is safe, fit and then MEASURE the worst flat-field output,
    lowering the bright cap until the behavioural gate passes. Returns
    (lut, dark, bright, info).
    """
    for cap in caps:
        lut, dark, bright = fit_v_joint(ref, cap_bright=cap, channels=channels)
        worst = worst_flat_field_output(lut, dark, bright)
        if worst <= limit:
            return lut, dark, bright, {"cap_bright": cap, "worst_flat": worst}
    return lut, dark, bright, {"cap_bright": caps[-1], "worst_flat": worst}


def fit_v_for_lut_safe(ref, lut: np.ndarray, limit: float = 255.5,
                       caps=(256.0, 252.0, 248.0, 244.0, 240.0, 232.0)
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit adaptive V endpoints for a fixed, caller-supplied gamma LUT.

    This is the correct path for a real ``gamma=off`` preset.  Reusing tables
    co-optimized with a nonlinear LUT changes both the carried pixel level and
    the max-RGB adaptive control, so it is only a rough fallback.  With the LUT
    fixed, the exact flat-field model remains linear in the endpoint row sums::

        out(f,x) = A_f * (v-v^2) + B_f * v^2,  v=LUT(x)/256

    We solve those sums directly in output-code space, apply the shader's beam
    shapes, quantize with exact conjugate symmetry, and measure (rather than
    assume) the no-flat-field-clipping invariant.
    """
    xs_i = EVAL_CODES
    xs = xs_i / 255.0
    fs = np.arange(0, 33) / 64.0
    O = np.array([[255.0 * _ref_vertical(ref, f, x) for x in xs] for f in fs])
    g = lut[xs_i].max(axis=1).astype(np.int64)
    shapes = _endpoint_shapes(ref)
    cap_a129 = np.minimum(511.0 / np.maximum(shapes[0].max(axis=1), 1e-9),
                         2044.0)
    cap_a = np.interp(fs, np.arange(129) / 256.0, cap_a129)

    last = None
    for cap_bright in caps:
        A, B = _solve_ab(O, g, cap_a, np.full(len(fs), cap_bright),
                         n=97, refine=7)
        tables = []
        for sums33, shape in ((A, shapes[0]), (B, shapes[1])):
            sums129 = np.interp(np.arange(129) / 256.0, fs, sums33)
            fl = np.zeros((PHASES, 4))
            fl[:129] = sums129[:, None] * shape
            for p in range(129, PHASES):
                fl[p] = fl[PHASES - p][::-1]
            fl = np.clip(fl, 0, None)
            target_sums = np.round(fl.sum(axis=1)).astype(np.int64)
            for p in range(1, 128):
                target_sums[PHASES - p] = target_sums[p]
            tables.append(quantize.quantize_symmetric(fl, target_sums))
        dark, bright = tables
        worst = worst_flat_field_output(lut, dark, bright)
        last = (dark, bright, {"cap_bright": cap_bright,
                               "worst_flat": worst})
        if worst <= limit:
            return last
    return last


def fit_v_for_lut_rgb_safe(ref, lut: np.ndarray,
                           neutral_weight: float = 4.0,
                           rgb_weight: float = 1.0,
                           rgb_levels: np.ndarray | None = None,
                           limit: float = 255.5,
                           caps=(256.0, 252.0, 248.0, 244.0, 240.0, 232.0)
                           ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit shared adaptive V endpoints to a Pareto mix of gray and RGB fields.

    For a signal channel ``s`` driven by MiSTer's shared maximum-RGB control
    ``g``, the unclipped flat-field model is linear in endpoint row sums::

        out = A * s*(256-g)/65536 + B * s*g/65536

    A dense neutral ramp and a chromatic cube are assigned independent total
    weights, so increasing ``neutral_weight`` traces a useful fidelity Pareto
    frontier instead of letting the much larger RGB cube swamp gray quality.
    The source target comes from ``ref_vertical_rgb`` with its mask disabled.
    Quantization and the measured no-clipping gate match the gray-only fitter.
    """
    rgb_fn = getattr(ref, "ref_vertical_rgb", None)
    if rgb_fn is None:
        raise AttributeError("reference does not publish ref_vertical_rgb")
    if neutral_weight <= 0.0 or rgb_weight <= 0.0:
        raise ValueError("neutral_weight and rgb_weight must be positive")
    if rgb_levels is None:
        rgb_levels = np.array([0, 64, 128, 192, 255], dtype=np.int64)
    colors = rgb_cube(rgb_levels)
    colors = colors[~np.all(colors == colors[:, :1], axis=1)]
    fs = np.arange(0, 33, dtype=np.float64) / 64.0

    targets, p_features, q_features, groups = [], [], [], []
    for code in EVAL_CODES:
        signal = int(lut[int(code), 1])
        control = int(lut[int(code)].max())
        if signal == 0:
            continue
        targets.append(np.array([
            255.0 * rgb_fn(float(f), (code / 255.0,) * 3)[1]
            for f in fs], dtype=np.float64))
        p_features.append(signal * (256 - control) / 65536.0)
        q_features.append(signal * control / 65536.0)
        groups.append(0)
    for rgb in colors:
        gamma = np.array([lut[int(rgb[c]), c] for c in range(3)], dtype=np.int64)
        control = int(gamma.max())
        source = np.asarray(rgb, dtype=np.float64) / 255.0
        reference = np.array([rgb_fn(float(f), source) for f in fs],
                             dtype=np.float64)
        for channel, signal in enumerate(gamma):
            if signal == 0:
                continue
            targets.append(255.0 * reference[:, channel])
            p_features.append(int(signal) * (256 - control) / 65536.0)
            q_features.append(int(signal) * control / 65536.0)
            groups.append(1)
    O = np.stack(targets, axis=1)
    p = np.asarray(p_features, dtype=np.float64)
    q = np.asarray(q_features, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    neutral_count = max(int(np.count_nonzero(groups == 0)), 1)
    rgb_count = max(int(np.count_nonzero(groups == 1)), 1)
    weights = np.where(groups == 0, neutral_weight / neutral_count,
                       rgb_weight / rgb_count)

    shapes = _endpoint_shapes(ref)
    cap_a129 = np.minimum(511.0 / np.maximum(shapes[0].max(axis=1), 1e-9),
                         2044.0)
    cap_a = np.interp(fs, np.arange(129) / 256.0, cap_a129)
    last = None
    for cap_bright in caps:
        A, B = _solve_ab_linear_box(
            O, p, q, weights, cap_a, np.full(len(fs), cap_bright))
        tables = []
        for sums33, shape in ((A, shapes[0]), (B, shapes[1])):
            sums129 = np.interp(np.arange(129) / 256.0, fs, sums33)
            fl = np.zeros((PHASES, 4), dtype=np.float64)
            fl[:129] = sums129[:, None] * shape
            for phase in range(129, PHASES):
                fl[phase] = fl[PHASES - phase][::-1]
            fl = np.clip(fl, 0.0, None)
            target_sums = np.rint(fl.sum(axis=1)).astype(np.int64)
            for phase in range(1, 128):
                target_sums[PHASES - phase] = target_sums[phase]
            tables.append(quantize.quantize_symmetric(fl, target_sums))
        dark, bright = tables
        worst = worst_flat_field_output(lut, dark, bright)
        last = (dark, bright, {
            "cap_bright": cap_bright, "worst_flat": worst,
            "neutral_weight": float(neutral_weight),
            "rgb_weight": float(rgb_weight),
            "neutral_observations": neutral_count,
            "rgb_observations": rgb_count,
        })
        if worst <= limit:
            return last
    return last


def fit_v_joint(ref, xs: np.ndarray | None = None, cap_bright: float = 256.0,
                iters: int = 25, channels: bool = False,
                seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint fit of the gamma warp AND the adaptive endpoint row sums.

    Supersedes optimize_lut + fit_v_adaptive for shaders whose depth-vs-
    brightness family the separate fits handle badly (Royale: 15.49 -> 6.55).
    Three defects in the separate path, all fixed here:

    1. WRONG OBJECTIVE. Both fitted in ROW-SUM space with uniform weight over x.
       Output error is u*(row-sum error)/256, so uniform row-sum weighting is an
       implicit 1/u^2 weighting in OUTPUT space, over-weighting dark levels up to
       16x and dragging the bright endpoint down (shipped B(f=0.5)=144 where the
       target wants ~252 — precisely the 105-code error at white).
    2. u >= transfer. Costs ~2.4 codes alone and blocks all benefit from (3).
    3. ROW SUM <= 256 — the big one (~6.1 codes). It is a far too strong proxy
       for "no scaler clipping": the hardware clips on the BLENDED row sum times
       the level, and the dark set's weight (256-lum)/256 goes to zero exactly
       where the level is large, so a dark row sum of 511 never clips a flat
       field. The real ceiling is the signed 10-bit coefficient limit — the dark
       beam at f=0 is nearly a delta, so its peak tap IS the row sum.

    Exact flat-field algebra (verified against the RTL model to RMSE 0.59):
        v = LUT(x)/256;  out(f,x) = min(A_f*(v - v^2) + B_f*v^2, 255)
    linear in (A, B), so both blocks solve exactly and BCD is a true descent.

    Returns (lut, dark_table, bright_table).
    """
    if xs is None:
        xs = EVAL_CODES / 255.0
    xs_i = np.round(xs * 255).astype(int)
    fs = np.arange(0, 33) / 64.0
    O = np.array([[255.0 * _ref_vertical(ref, f, x) for x in xs] for f in fs])

    shapes = _endpoint_shapes(ref)
    cap_a129 = np.minimum(511.0 / np.maximum(shapes[0].max(axis=1), 1e-9), 2044.0)
    cap_a = np.interp(fs, np.arange(129) / 256.0, cap_a129)
    cap_b = np.full(len(fs), cap_bright)

    rng = np.random.default_rng(seed)
    starts = [np.maximum.accumulate(xs_i.astype(np.int64)),
              np.round(255 * (xs_i / 255.0) ** 0.5).astype(np.int64),
              np.array([255 * _transfer(ref, x) for x in xs]).astype(np.int64),
              np.full(len(xs), 255, dtype=np.int64)]
    starts += [np.round(255 * np.sort(rng.random(len(xs)))).astype(np.int64)
               for _ in range(6)]

    best = (np.inf, None, None, None)
    for g0 in starts:
        g = np.maximum.accumulate(np.asarray(g0, dtype=np.int64).copy())
        for _ in range(iters):
            A, B = _solve_ab(O, g, cap_a, cap_b)
            gn = _dp_g(O, A, B)
            if (gn == g).all():
                break
            g = gn
        A, B = _solve_ab(O, g, cap_a, cap_b, n=97, refine=7)
        pred = np.minimum(np.outer(A, _joint_features(g)[0])
                          + np.outer(B, _joint_features(g)[1]), 255.0)
        r = float(np.sqrt(((pred - O) ** 2).mean()))
        if r < best[0]:
            best = (r, g.copy(), A.copy(), B.copy())
    _, g, A, B = best

    # Expand (A, B) onto 129 phases, apply the beam shape, mirror, quantize.
    tables = []
    p129 = np.arange(129)
    for S33, shape in ((A, shapes[0]), (B, shapes[1])):
        S129 = np.interp(p129 / 256.0, fs, S33)
        fl = np.zeros((PHASES, 4))
        fl[:129] = S129[:, None] * shape
        for p in range(129, PHASES):
            fl[p] = fl[PHASES - p][::-1]
        fl = np.clip(fl, 0, None)
        tgt = np.round(fl.sum(axis=1)).astype(np.int64)
        for p in range(1, 128):
            tgt[PHASES - p] = tgt[p]
        tables.append(quantize.quantize_symmetric(fl, tgt))

    gf = np.interp(np.arange(256), xs_i, g)
    gf = np.maximum.accumulate(np.clip(np.round(gf), 0, 255))
    # Preserve a deliberate lift only when the reference explicitly has one;
    # historical Kurozumi resolves its intended/practical black to true zero.
    if max(_transfer(ref, 0.0, c) for c in "rgb") <= 1e-9:
        gf[0] = 0
    lut = np.zeros((256, 3), dtype=np.int64)
    for c, name in enumerate("rgb"):
        if channels:
            ratio = np.array([_transfer(ref, x / 255.0, name)
                              / max(_transfer(ref, x / 255.0), 1e-9)
                              for x in range(256)])
            lut[:, c] = np.clip(np.round(np.maximum.accumulate(gf * ratio)), 0, 255)
        else:
            lut[:, c] = gf
    return lut, tables[0], tables[1]


def worst_flat_field_output(lut: np.ndarray, dark: np.ndarray,
                            bright: np.ndarray) -> float:
    """Highest flat-field output any input level produces (gate: <= 255.5).

    This is the behavioural form of "no scaler clipping". Checking each
    endpoint's row sum against 256 is the wrong invariant: the hardware clips
    on the BLENDED row, and the dark set's weight vanishes where the level is
    high, so an over-unity dark row is harmless while a legal-looking pair can
    still clip.  Each RGB signal channel is tested against the shared max-RGB
    adaptive control, which matters for Kurozumi's channel-specific LUT.
    """
    worst = 0.0
    phases = np.arange(256)
    for xi in range(256):
        ctrl = int(lut[xi].max())
        c128 = mm.adaptive_c128(dark, bright, phases,
                                np.full(256, ctrl, dtype=np.int64))
        row_sums = c128.sum(axis=1)
        for signal in lut[xi]:
            worst = max(
                worst, float((int(signal) * row_sums / 32768.0).max()))
    return worst


def adaptive_moire_1080(dark: np.ndarray, bright: np.ndarray,
                        lut: np.ndarray,
                        source_heights=(224, 240), output_lines: int = 1080,
                        codes=(64, 128, 191, 255)) -> float:
    """Worst exact peak/trough instability at common 1080p scale ratios."""
    worst = 0.0
    for code in codes:
        for source_lines in source_heights:
            scale = output_lines / source_lines
            for channel in range(3):
                profile = mm.flat_field_profile(
                    int(code), scale, dark, bright, lut, n_out=output_lines,
                    channel=channel)
                metrics = mm.moire_metrics(profile, scale)
                worst = max(worst, metrics["peak_std"], metrics["trough_std"])
    return float(worst)


def selector_control_jump(h: np.ndarray, dark: np.ndarray,
                          bright: np.ndarray, lut: np.ndarray) -> int:
    """Worst discontinuity caused solely by the nearest-line control switch."""
    cases = (
        ((64, 64, 64), (255, 255, 255)),
        ((0, 0, 0), (255, 255, 255)),
        ((255, 0, 0), (0, 0, 255)),
        ((255, 64, 64), (64, 64, 255)),
    )

    def h_rgb(rgb):
        return np.array([int(mm.fir_1d(
            np.full(16, int(lut[int(rgb[c]), c]), dtype=np.int64),
            h, np.array([8.0]))[0]) for c in range(3)], dtype=np.int64)

    worst = 0
    position = np.array([8.5], dtype=np.float64)
    for upper_rgb, lower_rgb in cases:
        upper, lower = h_rgb(upper_rgb), h_rgb(lower_rgb)
        lines = np.empty((16, 3), dtype=np.int64)
        lines[:9], lines[9:] = upper, lower
        for channel in range(3):
            before = mm.fir_1d_adaptive(
                lines[:, channel], dark, bright, position,
                np.array([upper.max()], dtype=np.int64))[0]
            after = mm.fir_1d_adaptive(
                lines[:, channel], dark, bright, position,
                np.array([lower.max()], dtype=np.int64))[0]
            worst = max(worst, abs(int(after) - int(before)))
    return int(worst)


def fit_v_rgb_pareto(ref, h: np.ndarray, tokens: list[list[str]],
                     baseline: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                     neutral_weights=(2.0, 4.0, 8.0),
                     neutral_objective: float = 2.0,
                     neutral_rms_limit: float = 2.6,
                     neutral_max_limit: float = 12.0,
                     rgb_rms_limit: float = 8.0,
                     rgb_max_limit: float = 32.0,
                     moire_limit: float = 7.65
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Select an RGB-aware gamma/V point on a measured fidelity frontier.

    Guest's source beam narrows minority channels using chroma, while MiSTer's
    single adaptive V control is shared max-RGB.  A gray-only gamma warp hides
    shared brightboost in each LUT channel and is therefore badly wrong for
    mixed colors.  This routine searches smooth hardware-legal gamma families,
    fits each V pair with several explicit gray/RGB objective weights, then
    chooses by exact masked RGB + neutral error under clipping, quantization,
    selector-independent 1080p moire, and max-error constraints.
    """
    if baseline is None:
        base_lut, base_dark, base_bright, _ = fit_v_joint_safe(ref)
    else:
        base_lut, base_dark, base_bright = baseline
    base_lut = np.asarray(base_lut, dtype=np.int64)
    identity = np.arange(256, dtype=np.float64)
    base_curve = base_lut[:, 0].astype(np.float64)
    curves = [("joint-baseline", base_lut.copy())]
    for alpha in (0.35, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75):
        curve = np.rint((1.0 - alpha) * base_curve + alpha * identity)
        curves.append((f"joint/identity {alpha:.2f}",
                       np.repeat(curve[:, None], 3, axis=1).astype(np.int64)))
    x = np.arange(256, dtype=np.float64) / 255.0
    for power in (1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15):
        curve = np.rint(255.0 * x ** power)
        curves.append((f"power {power:.3f}",
                       np.repeat(curve[:, None], 3, axis=1).astype(np.int64)))

    unique_curves, seen = [], set()
    for label, curve in curves:
        curve = np.maximum.accumulate(np.clip(curve, 0, 255), axis=0)
        key = curve.tobytes()
        if key not in seen:
            seen.add(key)
            unique_curves.append((label, curve))
    mask_encoded = mask_encoded_tile(ref)
    score_levels = np.array([0, 64, 128, 192, 255], dtype=np.int64)
    score_phases = np.arange(0, 9, dtype=np.float64) / 16.0
    neutral_codes = np.unique(np.concatenate((np.arange(0, 256, 8),
                                              np.array([253, 254, 255]))))
    candidates = []
    for label, curve in unique_curves:
        variants = []
        if label == "joint-baseline":
            variants.append(("joint", base_dark.copy(), base_bright.copy(),
                             {"worst_flat": worst_flat_field_output(
                                 curve, base_dark, base_bright)}))
        gray_dark, gray_bright, gray_info = fit_v_for_lut_safe(ref, curve)
        variants.append(("gray-fixed", gray_dark, gray_bright, gray_info))
        for weight in neutral_weights:
            rgb_dark, rgb_bright, rgb_info = fit_v_for_lut_rgb_safe(
                ref, curve, neutral_weight=float(weight), rgb_weight=1.0)
            variants.append((f"rgb-n{weight:g}", rgb_dark, rgb_bright, rgb_info))
        for variant, dark, bright, fit_info in variants:
            if not gamma_quality_ok(curve) or fit_info["worst_flat"] > 255.5:
                continue
            neutral_rms, neutral_max = rmse_exact_masked(
                ref, dark, bright, h, curve, tokens, mask_encoded,
                codes=neutral_codes)
            rgb = rgb_masked_metrics(
                ref, dark, bright, h, curve, tokens, levels=score_levels,
                phases=score_phases)
            moire = adaptive_moire_1080(dark, bright, curve)
            objective = rgb["rms"] + neutral_objective * neutral_rms
            feasible = (neutral_rms <= neutral_rms_limit
                        and neutral_max <= neutral_max_limit
                        and rgb["rms"] <= rgb_rms_limit
                        and rgb["max"] <= rgb_max_limit
                        and moire <= moire_limit)
            candidates.append({
                "label": f"{label}; {variant}", "lut": curve,
                "dark": dark, "bright": bright, "objective": objective,
                "neutral_rms": neutral_rms, "neutral_max": neutral_max,
                "rgb_rms": rgb["rms"], "rgb_max": rgb["max"],
                "moire": moire, "worst_flat": fit_info["worst_flat"],
                "feasible": feasible,
            })
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if not feasible:
        summary = sorted(candidates, key=lambda candidate: candidate["objective"])[:3]
        raise RuntimeError(f"no feasible RGB Pareto fit; best candidates: {summary}")
    # An RGB-weighted endpoint fit is required for this path; gray-only variants
    # remain in the report as a control showing whether the extra objective is
    # buying anything, but cannot silently win by a few thousandths.
    rgb_feasible = [candidate for candidate in feasible
                    if "; rgb-" in candidate["label"]]
    pool = rgb_feasible or feasible
    best = min(pool, key=lambda candidate: (candidate["objective"],
                                             candidate["rgb_rms"],
                                             candidate["neutral_rms"]))
    public = {key: value for key, value in best.items()
              if key not in ("lut", "dark", "bright")}
    public["evaluated"] = len(candidates)
    public["feasible_count"] = len(feasible)
    public["frontier"] = [{key: value for key, value in candidate.items()
                           if key not in ("lut", "dark", "bright")}
                          for candidate in sorted(
                              feasible, key=lambda candidate: candidate["objective"])[:8]]
    return best["lut"], best["dark"], best["bright"], public


def constrain_selector_phase128(dark: np.ndarray, bright: np.ndarray,
                                h: np.ndarray, lut: np.ndarray,
                                limit: int = 16, radius: int = 12,
                                score_fn=None,
                                clip_limit: float = 255.5
                                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Constrain the phase-128 selector artifact with a bounded symmetric fit.

    Only the two centre coefficients of the self-symmetric half-phase rows are
    searched.  This is the smallest possible intervention: all other phases,
    endpoint beam shapes, and exact conjugacy remain untouched.  ``score_fn``
    may provide a source-aware scalar score; otherwise coefficient movement is
    minimized.  The returned candidate is behaviourally clip-safe.
    """
    d0 = np.asarray(dark, dtype=np.int64)
    b0 = np.asarray(bright, dtype=np.int64)
    if d0.shape != (PHASES, 4) or b0.shape != (PHASES, 4):
        raise ValueError("adaptive tables must both have shape (256, 4)")
    candidates = []
    for delta_dark in range(-radius, radius + 1):
        for delta_bright in range(-radius, radius + 1):
            d = d0.copy()
            b = b0.copy()
            d[128, 1:3] += delta_dark
            b[128, 1:3] += delta_bright
            if np.any(d[128] < -512) or np.any(d[128] > 511) \
                    or np.any(b[128] < -512) or np.any(b[128] > 511):
                continue
            selector = selector_control_jump(h, d, b, lut)
            if selector > limit:
                continue
            score = (float(score_fn(d, b)) if score_fn is not None else
                     float(delta_dark * delta_dark + delta_bright * delta_bright))
            candidates.append((score, abs(delta_dark) + abs(delta_bright),
                               selector, delta_dark, delta_bright, d, b))
    for candidate in sorted(candidates, key=lambda item: item[:5]):
        score, _, selector, delta_dark, delta_bright, d, b = candidate
        worst = worst_flat_field_output(lut, d, b)
        if worst <= clip_limit:
            return d, b, {"selector": int(selector), "score": float(score),
                          "delta_dark": int(delta_dark),
                          "delta_bright": int(delta_bright),
                          "worst_flat": float(worst),
                          "candidates": len(candidates)}
    raise RuntimeError(f"no clip-safe phase-128 selector fit reaches {limit} codes")


def stabilize_adaptive_selector(dark: np.ndarray, bright: np.ndarray,
                                radius: int = 8,
                                strength: float = 1.0
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Regularize endpoint sets around the nearest-line selector boundary.

    MiSTer's adaptive control samples the nearest source line, so at phase 128
    it can jump from the upper line's luminance to the lower line's luminance.
    Very different endpoint rows turn an ordinary dark/light edge into a large
    one-pixel discontinuity (over 100 codes in the old Royale table), a defect
    no flat-field objective can observe.

    Around the half-phase boundary, smoothly pull both sets toward their common
    midpoint.  This preserves their average response, exact phase conjugacy,
    signed-10-bit range, and full adaptation away from the selector switch.
    ``radius`` is measured in the 256-phase table; zero returns exact copies.
    """
    if radius < 0 or radius > 128:
        raise ValueError("radius must be in 0..128")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in 0..1")
    d = np.asarray(dark, dtype=np.float64).copy()
    b = np.asarray(bright, dtype=np.float64).copy()
    if d.shape != (PHASES, 4) or b.shape != (PHASES, 4):
        raise ValueError("adaptive tables must both have shape (256, 4)")
    if radius == 0 or strength == 0.0:
        return d.astype(np.int64), b.astype(np.int64)

    for p in range(129):
        dist = abs(128 - p)
        if dist > radius:
            continue
        # Raised cosine: full regularization at p=128, zero value and slope at
        # the edge of the transition band.
        w = strength * 0.5 * (1.0 + np.cos(np.pi * dist / radius))
        common = 0.5 * (d[p] + b[p])
        d[p] = (1.0 - w) * d[p] + w * common
        b[p] = (1.0 - w) * b[p] + w * common
    for p in range(129, PHASES):
        d[p] = d[PHASES - p][::-1]
        b[p] = b[PHASES - p][::-1]
    return np.rint(d).astype(np.int64), np.rint(b).astype(np.int64)


def blend_adaptive_endpoints(dark: np.ndarray, bright: np.ndarray,
                             strength: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Globally reduce adaptation while preserving its mean beam response.

    This is the display-safe companion to ``stabilize_adaptive_selector``.
    Some very narrow beams become unstable at fractional vertical scales when
    only phases around the nearest-line selector are regularized.  Pulling the
    two endpoint tables toward their common midpoint at every phase trades
    brightness-dependent beam variation for a bounded selector transition and
    materially lower moire.  ``strength=0`` is an exact copy; ``1`` collapses
    both sets to the same fixed profile.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in 0..1")
    d = np.asarray(dark, dtype=np.float64)
    b = np.asarray(bright, dtype=np.float64)
    if d.shape != (PHASES, 4) or b.shape != (PHASES, 4):
        raise ValueError("adaptive tables must both have shape (256, 4)")
    common = 0.5 * (d + b)
    ds = np.rint((1.0 - strength) * d + strength * common).astype(np.int64)
    bs = np.rint((1.0 - strength) * b + strength * common).astype(np.int64)
    # Rounding each endpoint independently can differ by one between conjugate
    # phases.  Reassert the exact stored-table invariants after quantization.
    for p in range(1, 128):
        ds[PHASES - p] = ds[p][::-1]
        bs[PHASES - p] = bs[p][::-1]
    ds[128] = np.rint(0.5 * (ds[128] + ds[128][::-1])).astype(np.int64)
    bs[128] = np.rint(0.5 * (bs[128] + bs[128][::-1])).astype(np.int64)
    return ds, bs


def blend_dark_toward_bright(dark: np.ndarray, bright: np.ndarray,
                             strength: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Reduce adaptation without importing high-gain dark rows into highlights.

    This one-sided form is useful for Royale: its dark endpoint legally exceeds
    unity because the dark control weight vanishes in highlights.  A symmetric
    midpoint blend moves those large rows into the bright endpoint and clips;
    moving only dark toward the already-safe bright table bounds selector jumps
    while retaining the original highlight response and clip safety.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in 0..1")
    d = np.asarray(dark, dtype=np.float64)
    b = np.asarray(bright, dtype=np.int64).copy()
    if d.shape != (PHASES, 4) or b.shape != (PHASES, 4):
        raise ValueError("adaptive tables must both have shape (256, 4)")
    ds = np.rint((1.0 - strength) * d + strength * b).astype(np.int64)
    for p in range(1, 128):
        ds[PHASES - p] = ds[p][::-1]
    ds[128] = np.rint(0.5 * (ds[128] + ds[128][::-1])).astype(np.int64)
    return ds, b


def fit_v_fixed(ref, lut_ctrl: np.ndarray, xs: np.ndarray | None = None,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Single-set best-compromise V table against a GIVEN LUT.

    Kept for callers that must reuse the adaptive LUT; fit_v_fixed_paired
    produces a better table by choosing its own LUT.
    """
    if xs is None:
        xs = EVAL_CODES / 255.0
    rows, _ = _tap_targets(ref, lut_ctrl, xs)
    w = np.ones(len(xs)) if weights is None else weights
    w = w / w.sum()
    fixed_f = np.zeros((PHASES, 4))
    for p in range(129):
        fixed_f[p] = (rows[p] * w[:, None]).sum(axis=0)
    for p in range(129, PHASES):
        fixed_f[p] = fixed_f[PHASES - p][::-1]
    fixed_f = np.clip(fixed_f, 0, None)
    sums = fixed_f.sum(axis=1)
    over = sums > 256.0
    fixed_f[over] *= (256.0 / sums[over])[:, None]
    target_sums = np.round(fixed_f.sum(axis=1)).astype(np.int64)
    for p in range(1, 128):
        target_sums[PHASES - p] = target_sums[p]
    return quantize.quantize_symmetric(fixed_f, target_sums)


def fit_v_fixed_paired(ref, channels: bool = False,
                       iters: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Jointly fit a fixed (non-adaptive) V table AND its own gamma LUT.

    Without an adaptive control the flat-field model collapses to
        out(f, x) = u(x) * S(f) / 256
    which is exactly a RANK-1 factorization of the target matrix — so the
    optimum is an alternating least squares on (u, S), not a weighted average
    of per-x ideal rows (what fit_v_fixed does), and it wants its own LUT
    rather than one warped for the adaptive path.

    Returns (fixed table, (256, 3) LUT).
    """
    xs = EVAL_CODES / 255.0
    fs = np.arange(0, 129) / 256.0
    O = np.array([[_ref_vertical(ref, f, x) for x in xs] for f in fs])   # (129, NX)

    u = np.array([max(_transfer(ref, x), 1e-6) for x in xs])
    S = np.full(len(fs), 1.0)
    for _ in range(iters):
        denom = float((u * u).sum())
        S = (O @ u) / max(denom, 1e-12)
        S = np.clip(S, 0.0, 1.0)                       # row sums <= 256: no clipping
        denom = float((S * S).sum())
        u = (O.T @ S) / max(denom, 1e-12)
        u = np.clip(u, 1e-6, 1.0)
        u = np.maximum.accumulate(u)                   # LUT must be monotone

    # Tap shapes still come from the beam; only the row sums come from S.
    fixed_f = np.zeros((PHASES, 4))
    for p in range(129):
        f = p / 256.0
        d = np.array([f + 1.0, f, 1.0 - f, 2.0 - f])
        w = np.array([max(ref.beam_weight(di, 0.5), 0.0) for di in d])
        if w.sum() < 1e-12:
            w = np.zeros(4)
            w[np.argmin(d)] = 1.0
        fixed_f[p] = 256.0 * S[p] * w / w.sum()
    for p in range(129, PHASES):
        fixed_f[p] = fixed_f[PHASES - p][::-1]
    fixed_f = np.clip(fixed_f, 0, None)
    sums = fixed_f.sum(axis=1)
    over = sums > 256.0
    fixed_f[over] *= (256.0 / sums[over])[:, None]
    target_sums = np.round(fixed_f.sum(axis=1)).astype(np.int64)
    for p in range(1, 128):
        target_sums[PHASES - p] = target_sums[p]
    table = quantize.quantize_symmetric(fixed_f, target_sums)

    xfull = np.arange(256) / 255.0
    ufull = np.maximum.accumulate(np.clip(
        np.interp(xfull, np.concatenate([[0.0], xs]),
                  np.concatenate([[_transfer(ref, 0.0)], u])), 0.0, 1.0))
    lut = np.zeros((256, 3), dtype=np.int64)
    for c, name in enumerate("rgb"):
        if channels:
            ratio = np.array([_transfer(ref, x, name) / max(_transfer(ref, x), 1e-9)
                              for x in xfull])
            vals = 255.0 * np.clip(ufull * ratio, 0.0, 1.0)
        else:
            vals = 255.0 * ufull
        lut[:, c] = np.clip(np.round(np.maximum.accumulate(vals)), 0, 255)
    return table, lut


def no_scanline_table(ref, at_x: float = 1.0) -> np.ndarray:
    """Scan-free vertical interpolation: bright beam shape normalized to DC 256."""
    ideal = np.zeros((PHASES, 4))
    for p in range(129):
        f = p / 256.0
        d = np.array([f + 1.0, f, 1.0 - f, 2.0 - f])
        w = np.array([max(ref.beam_weight(di, at_x), 0.0) for di in d])
        if w.sum() < 1e-12:
            w = np.zeros(4)
            w[np.argmin(d)] = 1.0
        ideal[p] = 256.0 * w / w.sum()
    for p in range(129, PHASES):
        ideal[p] = ideal[PHASES - p][::-1]
    sums = np.full(PHASES, 256, dtype=np.int64)
    return quantize.quantize_symmetric(ideal, sums)


# ------------------------------------------------------------------ H filter

def fit_h(ref) -> np.ndarray:
    """Fold h_kernel(frac) into the 4-tap window, renormalized to DC 256."""
    ideal = np.zeros((PHASES, 4))
    for p in range(129):
        f = p / 256.0
        taps = np.zeros(4)
        total = 0.0
        for off, w in ref.h_kernel(f):
            total += w
            idx = int(round(off)) + 1                 # offset -1 -> column 0
            if 0 <= idx <= 3:
                taps[idx] += w
        if taps.sum() > 1e-9:
            taps *= (total / taps.sum()) if total > 1e-9 else 1.0
        ideal[p] = 256.0 * taps / max(taps.sum(), 1e-9)
    for p in range(129, PHASES):
        ideal[p] = ideal[PHASES - p][::-1]
    sums = np.full(PHASES, 256, dtype=np.int64)
    return quantize.quantize_symmetric(ideal, sums)


# ------------------------------------------------------------------ masks

_RAMP = np.arange(0, 256, 5, dtype=np.int64)


def _token_response(m16: np.ndarray) -> np.ndarray:
    """(len(RAMP), 3) exact output codes for per-channel multipliers m16."""
    return mm.mask_multiply(np.repeat(_RAMP[:, None], 3, axis=1),
                            np.asarray(m16, dtype=np.int64)[None, :])


def fit_mask_token(target_mult: np.ndarray, weights: np.ndarray = LUMA_W,
                   y_max: int = 15, exact: bool = True) -> tuple[str, float]:
    """Best v2 token for one output pixel's encoded per-channel multipliers.

    exact=True scores through mister_model.mask_multiply over a gray ramp
    rather than comparing ideal m/16 values. The hardware multiplies by summing
    INDEPENDENTLY truncated shifts, so nominal error and real error disagree:
    m=15/16 runs ~1.5 codes low on average (4 truncations) while m=16/16 is
    exact, which can flip which token wins.
    """
    import fileio
    tgt = _RAMP[:, None] * np.asarray(target_mult, dtype=np.float64)[None, :]
    tgt = np.clip(np.round(tgt), 0, 255)
    best = (None, np.inf)
    for bm in range(1, 8):
        sel = np.array([bool(bm & 4), bool(bm & 2), bool(bm & 1)])
        for y in range(0, y_max + 1):
            for z in range(0, 16):
                m16 = np.where(sel, 16 + y, z)
                if exact:
                    resp = _token_response(m16)
                    err = float((weights * ((resp - tgt) ** 2)).sum() / len(_RAMP))
                else:
                    m = m16 / 16.0
                    err = float((weights * (m - target_mult) ** 2).sum())
                if err < best[1]:
                    best = (fileio.encode_mask_token(bm, 16 + y, z), err)
    return best[0], float(np.sqrt(best[1]))


def fit_mask_column_dither(target_mult: np.ndarray, repeats: int,
                           weights: np.ndarray = LUMA_W,
                           shortlist: int = 48) -> tuple[list[str], float]:
    """Dither one pixel's target across `repeats` horizontal copies.

    A v2 token gives its selected channels one value and BOTH others a single
    shared value, so a triple like Kurozumi's (1.30, 1.01, 0.57) — green near
    unity while blue is halved — is unrepresentable in any single token, and a
    one-token fit must sacrifice a channel (it spends blue, weight 0.07, to buy
    luma, which is what flattens the R/B ripple).

    Dithering across HORIZONTAL repeats fixes this: the eye integrates the pair
    and the pattern already repeats horizontally, so the tile gains no vertical
    structure at all — its vertical spectrum is unchanged, which matters because
    a 2-row tile would inject a Nyquist luma component that beats against the
    4.5x scanline (measured up to ~3.6 codes).

    Returns (tokens for each repeat, exact weighted rmse of their mean).

    The pair search is EXHAUSTIVE over all 7*16*16 tokens via a Gram matrix.
    Shortlisting by solo error is wrong here: the best dither pair is usually
    two individually-poor tokens that straddle the target (one over, one under)
    and average onto it, which any solo-error shortlist discards first.
    """
    import fileio
    if repeats == 1:
        tok, err = fit_mask_token(target_mult, weights)
        return [tok], err

    tgt = np.clip(np.round(_RAMP[:, None]
                           * np.asarray(target_mult, dtype=np.float64)[None, :]), 0, 255)
    names, resp = [], []
    for bm in range(1, 8):
        sel = np.array([bool(bm & 4), bool(bm & 2), bool(bm & 1)])
        for y in range(0, 16):
            for z in range(0, 16):
                m16 = np.where(sel, 16 + y, z)
                names.append(fileio.encode_mask_token(bm, 16 + y, z))
                resp.append(_token_response(m16).astype(np.float64))
    R = np.array(resp)                                   # (N, ramp, 3)
    w = np.sqrt(weights)[None, None, :]
    Rw = (R * w).reshape(len(names), -1)                 # weighted, flattened
    Tw = (tgt * w[0]).ravel()
    # ||(Ri+Rj)/2 - T||^2 = a_i + a_j + <Ri,Rj>/2 + ||T||^2
    a = 0.25 * (Rw * Rw).sum(axis=1) - Rw @ Tw
    G = Rw @ Rw.T
    err = a[:, None] + a[None, :] + 0.5 * G + float(Tw @ Tw)
    i, j = np.unravel_index(int(np.argmin(err)), err.shape)
    pair = [names[i], names[j]]
    toks = (pair * (repeats // 2 + 1))[:repeats]
    return toks, float(np.sqrt(max(err[i, j], 0.0) / len(_RAMP)))


def fit_mask_tile(ref, gain: float = 1.0, max_dim: int = 16,
                  strategy: dict | None = None) -> tuple[list[list[str]], dict]:
    """Full mask pipeline: encoded target -> hardware tile -> token grid.

    `strategy` pins the tile choice per shader (a look judgement that wants
    measurement, not a generic heuristic — see DESIGN.md):
        period   (ph, pw) hardware tile size to use
        kind     'mean' (average the repeats) | 'verbatim' (slice)
        dither   N: spread each column across N horizontal slots
    Omitted keys fall back to choose_mask_tile / no dither.
    `gain` scales the target for the gain-split factorization.
    """
    strategy = strategy or {}
    target = mask_encoded_tile(ref) * gain
    H, W, _ = target.shape

    if "period" in strategy:
        ph, pw = strategy["period"]
        kind = strategy.get("kind", "mean")
        if kind == "verbatim":
            tile = target[:ph, :pw].copy()
        else:
            acc = np.zeros((ph, pw, 3))
            cnt = np.zeros((ph, pw, 1))
            for yy in range(H):
                for xx in range(W):
                    acc[yy % ph, xx % pw] += target[yy, xx]
                    cnt[yy % ph, xx % pw, 0] += 1
            tile = acc / cnt
        rebuilt = np.tile(tile, (H // ph + 1, W // pw + 1, 1))[:H, :W]
        info = {"period": (ph, pw), "kind": kind,
                "spatial_rmse": float(np.sqrt(np.mean((rebuilt - target) ** 2))),
                "structure_kept": _structure_overlap(rebuilt, target)}
    else:
        tile, info = choose_mask_tile(target, max_dim=max_dim)
    h, w, _ = tile.shape

    tokens, _ = tile_from_encoded(tile)
    info = dict(info, err_plain=_tile_exact_err(tokens, tile), dithered=False)

    n = int(strategy.get("dither", 1))
    if n > 1 and w * n <= max_dim:
        # Partners sit one full period apart (col xx at positions xx, xx+w, ...),
        # NOT adjacent: the base pattern's own alternation must survive, so the
        # dither residual lands at a higher frequency (1/(n*w) cyc/px) instead of
        # replacing the ripple the mask exists to produce.
        rows = []
        for yy in range(h):
            row = [None] * (w * n)
            for xx in range(w):
                toks, _ = fit_mask_column_dither(tile[yy, xx], n)
                for k in range(n):
                    row[k * w + xx] = toks[k]
            rows.append(row)
        info.update(dithered=True,
                    err_dither=_tile_exact_err(rows, tile, group=w))
        return rows, info
    return tokens, info


def fit_mask_joint(ref, lut: np.ndarray, dark: np.ndarray, bright: np.ndarray,
                   h: np.ndarray, tokens: list[list[str]],
                   mask_encoded: np.ndarray, dither_group: int | None = None,
                   codes: np.ndarray | None = None) -> tuple[list[list[str]], float]:
    """Refit the mask tokens to minimize END-TO-END masked error.

    fit_mask_tile matches the shader's mask multipliers in isolation, which is
    the wrong objective: it leaves the mask unable to compensate for anything
    upstream. Two things it therefore cannot fix, and this can:

      * CLAMP ORDER. The shader clips clamp(B*m) after the mask; MiSTer clamps
        the V stage and then saturates inside the mask. Matching m exactly bakes
        that mismatch in.
      * V/LUT RESIDUAL. The fitted filters are not exact, and the mask
        multiplies whatever they produce — so the best token is the one that
        minimizes the error of the PRODUCT, not of the multiplier.

    Consequence in practice: the shipped Royale tile peaks at 1.625 encoded
    where the hardware ceiling is 1.9375, leaving flat white ~24% dark. The
    mask has headroom the isolated fit never asks for.

    Per output cell this is a search over all 7*16*16 tokens scored through the
    exact mask arithmetic against the real simulated pre-mask output — cheap,
    and exact rather than a heuristic.  Hardware and shader mask periods need
    not match, so every candidate is scored over their complete LCM supercell;
    fitting only the upper-left hardware-sized crop can choose a token that is
    badly wrong everywhere the shorter hardware tile repeats.

    `dither_group`: cells xx, xx+group, ... are perceptual dither partners, so
    their MEAN is fitted to the mean of every reference cell they cover.
    Returns (tokens, masked rmse).
    """
    import fileio
    if codes is None:
        codes = EVAL_CODES
    fs = np.arange(0, 33) / 64.0
    h_tile, w_tile, _ = mask_encoded.shape
    rows, cols = len(tokens), len(tokens[0])
    group = dither_group or cols
    super_h = int(np.lcm(rows, h_tile))
    super_w = int(np.lcm(cols, w_tile))

    # Pre-mask simulated output and exact post-mask source output, per level.
    sims, references = [], []
    for code in codes:
        sims.append(simulate_flat_rgb(dark, bright, h, lut, int(code), fs))
        references.append(masked_reference_tile(
            ref, fs, int(code), mask_encoded))
    V = np.stack(sims)                                   # (NC, F, 3)
    T = np.stack(references)                              # (NC, H, W, F, 3)

    names, resp = [], []
    for bm in range(1, 8):
        sel = np.array([bool(bm & 4), bool(bm & 2), bool(bm & 1)])
        for y in range(16):
            for z in range(16):
                m16 = np.where(sel, 16 + y, z)
                names.append(fileio.encode_mask_token(bm, 16 + y, z))
                resp.append(mm.mask_multiply(V.reshape(-1, 3),
                                             m16[None, :]).reshape(V.shape))
    R = np.stack(resp).astype(np.float64)                # (NT, NC, F, 3)

    flat = R.reshape(len(names), -1)
    r2 = (flat * flat).sum(axis=1)
    gram = flat @ flat.T
    out = [list(r) for r in tokens]

    # Token choices can change the best rigid mask alignment.  Two alternating
    # alignment/refit passes are deterministic and sufficient for these small
    # periodic tiles; a third is retained as a convergence check.
    for _ in range(3):
        _, _, (dy, dx) = _rmse_exact_masked_periodic(
            ref, dark, bright, h, lut, out, mask_encoded, codes=codes)
        previous = [list(r) for r in out]
        for yy in range(rows):
            for xx in range(group):
                members = list(range(xx, cols, group))
                reference_cells = []
                for sy in range(yy, super_h, rows):
                    for sx in range(super_w):
                        if sx % cols in members:
                            # Evaluator convention: np.roll(reference, (dy,dx)),
                            # hence output (sy,sx) sees unrolled cell (sy-dy,sx-dx).
                            reference_cells.append(((sy - dy) % h_tile,
                                                    (sx - dx) % w_tile))
                targets = np.stack([
                    T[:, ry, rx, :, :] for ry, rx in reference_cells], axis=0)

                if len(members) == 1:
                    # Sum_k ||R-T_k||^2 from sufficient statistics, avoiding a
                    # large (tokens x congruent-cells x levels x phases x RGB)
                    # temporary for every hardware cell.
                    tsum = targets.sum(axis=0).ravel()
                    err = (len(reference_cells) * r2 - 2.0 * (flat @ tsum)
                           + float((targets * targets).sum()))
                    out[yy][members[0]] = names[int(np.argmin(err))]
                else:
                    # Dither partners are deliberately integrated by the eye;
                    # fit their mean response to the mean target over every
                    # reference cell those hardware columns cover.
                    t = targets.mean(axis=0).ravel()
                    a = 0.25 * r2 - flat @ t
                    e = a[:, None] + a[None, :] + 0.5 * gram + float(t @ t)
                    i, j = np.unravel_index(int(np.argmin(e)), e.shape)
                    for k, mi in enumerate(members):
                        out[yy][mi] = names[i if k % 2 == 0 else j]
        if out == previous:
            break
    r, _ = rmse_exact_masked(
        ref, dark, bright, h, lut, out, mask_encoded, codes=codes)
    return out, r


def simulate_uniform_rgb(dark: np.ndarray, bright: np.ndarray, h: np.ndarray,
                         lut: np.ndarray, rgb: np.ndarray,
                         fs: np.ndarray) -> np.ndarray:
    """Exact pre-mask output for a horizontally/vertically uniform RGB field.

    The adaptive control is the shared maximum of the three exact post-H
    channels, matching the scaler RTL.  This distinction is invisible to a
    neutral ramp and is the central constraint for chromatic Guest fitting.
    """
    rgb = np.asarray(rgb, dtype=np.int64)
    fs = np.asarray(fs, dtype=np.float64)
    if rgb.shape != (3,) or np.any((rgb < 0) | (rgb > 255)):
        raise ValueError("rgb must contain exactly three codes in 0..255")
    lines = np.empty(16, dtype=np.int64)
    hvalues = np.empty(3, dtype=np.int64)
    for c in range(3):
        lines.fill(int(lut[int(rgb[c]), c]))
        hvalues[c] = int(mm.fir_1d(lines, h, np.array([8.0]))[0])
    ctrl = np.full(len(fs), int(hvalues.max()), dtype=np.int64)
    per_ch = []
    for c in range(3):
        lines.fill(int(hvalues[c]))
        per_ch.append(mm.fir_1d_adaptive(lines, dark, bright, 8.0 + fs, ctrl))
    return np.stack(per_ch, axis=1)


def simulate_flat_rgb(dark: np.ndarray, bright: np.ndarray, h: np.ndarray,
                      lut: np.ndarray, code: int, fs: np.ndarray) -> np.ndarray:
    """(F, 3) exact pre-mask output for a uniform neutral field of ``code``."""
    return simulate_uniform_rgb(dark, bright, h, lut,
                                np.full(3, int(code), dtype=np.int64), fs)


def _tile_exact_err(tokens: list[list[str]], target: np.ndarray,
                    group: int | None = None) -> float:
    """Exact-arithmetic luma-weighted RMSE (output codes) of a token grid.

    `group`: when the grid dithers, columns xx, xx+group, xx+2*group... are the
    repeats of ONE target column and it is their MEAN that must match it —
    scoring each token against the target alone would condemn every dither.
    """
    import fileio
    mask = fileio.MaskFile([], len(tokens[0]), len(tokens), tokens)
    m16 = np.round(mask.multipliers() * 16.0).astype(np.int64)
    h, w, _ = target.shape
    cols = m16.shape[1]
    group = group or cols
    errs = []
    for yy in range(m16.shape[0]):
        for xx in range(group):
            reps = [_token_response(m16[yy, k]).astype(np.float64)
                    for k in range(xx, cols, group)]
            resp = np.mean(reps, axis=0)
            tgt = np.clip(np.round(_RAMP[:, None]
                                   * target[yy % h, xx % w][None, :]), 0, 255)
            errs.append(LUMA_W * (resp - tgt) ** 2)
    return float(np.sqrt(np.mean(np.sum(np.array(errs), axis=2))))


def tile_from_encoded(mult_encoded: np.ndarray) -> tuple[list[list[str]], float]:
    """Fit a token grid to an (h, w, 3) encoded multiplier tile. Returns rmse."""
    h, w, _ = mult_encoded.shape
    tokens, errs = [], []
    for yy in range(h):
        row = []
        for xx in range(w):
            tok, e = fit_mask_token(mult_encoded[yy, xx])
            row.append(tok)
            errs.append(e)
        tokens.append(row)
    return tokens, float(np.sqrt(np.mean(np.square(errs))))


def _structure_spectrum(tile: np.ndarray) -> np.ndarray:
    """Per-channel 2-D magnitude spectrum with DC removed (structure only)."""
    s = np.abs(np.fft.fft2(tile, axes=(0, 1)))
    s[0, 0, :] = 0.0
    return s


def _structure_overlap(cand_tiled: np.ndarray, target: np.ndarray) -> float:
    """Fraction of the target's structure the candidate keeps AT THE RIGHT
    frequencies: sum_f min(|C(f)|, |T(f)|) / sum_f |T(f)|.

    Comparing total spectral energy is not enough — a tile can hold the same
    energy at the WRONG frequencies and score 100%. Per-frequency overlap
    cannot be gamed that way, and it punishes exactly the failure that matters:
    a band present in the target and zeroed in the candidate contributes 0.
    """
    c = _structure_spectrum(cand_tiled)
    t = _structure_spectrum(target)
    denom = float(t.sum())
    return float(np.minimum(c, t).sum() / denom) if denom > 1e-12 else 1.0


def choose_mask_tile(tile: np.ndarray, max_dim: int = 16,
                     structure_weight: float = 3.0) -> tuple[np.ndarray, dict]:
    """Pick the hardware mask tile (<= max_dim) that best represents `tile`.

    Mean-averaging the repeats is the L2-optimal period tile, which is exactly
    why picking by per-pixel RMSE is a trap: when a component is ANTISYMMETRIC
    under the chosen period, the mean cancels it to zero and the RMSE *improves*
    for having destroyed it. Royale's 24x24 slot tile is antisymmetric under
    y -> y+12, so a 12-row mean silently degrades a slot mask into a plain
    aperture grille (its whole point) while scoring better.

    So score candidates on spatial error PLUS how well they preserve the
    source's non-DC spectrum, and offer verbatim slices alongside means.
    Returns (tile, info).
    """
    H, W, _ = tile.shape
    tgt_spec = _structure_spectrum(tile)
    tgt_energy = float(np.sqrt((tgt_spec ** 2).sum()))
    best = (None, np.inf, None)
    for ph in [d for d in range(1, min(H, max_dim) + 1) if H % d == 0]:
        for pw in [d for d in range(1, min(W, max_dim) + 1) if W % d == 0]:
            acc = np.zeros((ph, pw, 3))
            cnt = np.zeros((ph, pw, 1))
            for yy in range(H):
                for xx in range(W):
                    acc[yy % ph, xx % pw] += tile[yy, xx]
                    cnt[yy % ph, xx % pw, 0] += 1
            variants = {"mean": acc / cnt, "verbatim": tile[:ph, :pw].copy()}
            for kind, cand in variants.items():
                rebuilt = np.tile(cand, (H // ph + 1, W // pw + 1, 1))[:H, :W]
                spatial = float(np.sqrt(np.mean((rebuilt - tile) ** 2)))
                kept = _structure_overlap(rebuilt, tile)
                score = spatial + structure_weight * (1.0 - kept) + 1e-4 * (ph * pw)
                if score < best[1]:
                    best = (cand, score, {"period": (ph, pw), "kind": kind,
                                          "spatial_rmse": spatial,
                                          "structure_kept": kept})
    return best[0], best[2]


def minimal_period_tile(tile: np.ndarray, max_dim: int = 16) -> tuple[np.ndarray, float]:
    """Best (ph<=max, pw<=max) periodic approximation of an (H, W, 3) tile.

    Candidate periods are restricted to divisors of the source dimensions:
    a non-divisor period can score well inside the sample window yet beat
    against the true pattern when tiled across the whole screen.

    Returns (period tile = mean over repeats, tiling rmse in multiplier units).
    """
    H, W, _ = tile.shape
    best = (None, np.inf)
    for ph in [d for d in range(1, min(H, max_dim) + 1) if H % d == 0]:
        for pw in [d for d in range(1, min(W, max_dim) + 1) if W % d == 0]:
            acc = np.zeros((ph, pw, 3))
            cnt = np.zeros((ph, pw, 1))
            for yy in range(H):
                for xx in range(W):
                    acc[yy % ph, xx % pw] += tile[yy, xx]
                    cnt[yy % ph, xx % pw, 0] += 1
            mean = acc / cnt
            rebuilt = np.tile(mean, (H // ph + 1, W // pw + 1, 1))[:H, :W]
            err = float(np.sqrt(np.mean((rebuilt - tile) ** 2)))
            # prefer smaller tiles on near-ties
            score = err + 1e-4 * (ph * pw)
            if score < best[1]:
                best = ((mean, err), score)
    return best[0]
