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

import quantize

PHASES = 256
LUMA_W = np.array([0.2126, 0.7152, 0.0722])


def _transfer(ref, x: float, channel: str | None = None) -> float:
    if channel is not None:
        try:
            return ref.transfer(x, channel)
        except TypeError:
            pass
    return ref.transfer(x)


def _ref_vertical(ref, f: float, x: float) -> float:
    # Profiles are symmetric about f=0.5; some modules only cover 0..0.5.
    return ref.ref_vertical(min(f, 1.0 - f) if f > 0.5 else f, x)


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
    xs_i = np.arange(8, 256, 4)
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
    xfull = np.arange(256) / 255.0
    ufull = np.interp(xfull, xs, u)
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
        xs = np.arange(8, 256, 4) / 255.0
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


def fit_v_fixed(ref, lut_ctrl: np.ndarray, xs: np.ndarray | None = None,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Single-set best-compromise V table (weighted mean of ideal rows over x)."""
    if xs is None:
        xs = np.arange(8, 256, 4) / 255.0
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

def fit_mask_token(target_mult: np.ndarray, weights: np.ndarray = LUMA_W,
                   y_max: int = 15) -> tuple[str, float]:
    """Best v2 token for one output pixel's encoded per-channel multipliers.

    Enumerates channel bitmask 1..7 and nibbles; returns (token, weighted_rmse).
    """
    import fileio
    best = (None, np.inf)
    for bm in range(1, 8):
        sel = np.array([bool(bm & 4), bool(bm & 2), bool(bm & 1)])
        for y in range(0, y_max + 1):
            for z in range(0, 16):
                m = np.where(sel, (16 + y) / 16.0, z / 16.0)
                err = float((weights * (m - target_mult) ** 2).sum())
                if err < best[1]:
                    best = (fileio.encode_mask_token(bm, 16 + y, z), err)
    return best[0], float(np.sqrt(best[1]))


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
