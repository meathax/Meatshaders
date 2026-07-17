"""
kurozumi_ref.py -- Reference implementation of CRT-ROYALE-KUROZUMI (the
P22/PVM-look preset over crt-royale, by Kurozumi/hunterk) from
libretro/slang-shaders pinned to commit
3b0d6aa1d134a168478cd9c904a866d969f8882b (presets/crt-royale-kurozumi.slangp,
13 passes: grade.slang prepended to the 12 crt-royale passes).

Models the KUROZUMI-parameter math for a 240p source displayed at 1920x1080
(4.5x vertical scale). Pure Python (math only; numpy not required).
Same interface as royale_ref.py: transfer, beam_weight, ref_vertical,
h_kernel, mask_spec, notes, defaults.

Effective parameter set (all from the .slangp `parameters` overrides; every
override is applied to the #pragma parameter uniforms at runtime):

    crt_gamma = 2.4           lcd_gamma = 2.4         levels_contrast = 0.67
    halation_weight = 0.0037  diffusion_weight = 0.0011
    bloom_underestimate_levels = 0.68   bloom_excess = 0.0
    beam_min_sigma = 0.02     beam_max_sigma = 0.16   beam_spot_power = 0.38
    beam_min_shape = 2.0      beam_max_shape = 4.0    beam_shape_power = 0.25
    beam_horiz_filter = 1.0 (GAUSSIAN)  beam_horiz_sigma = 0.32
    beam_horiz_linear_rgb_weight = 1.0 (all-linear H mixing)
    convergence_offset_x = (-0.05, 0, 0) texels (r,g,b)
    convergence_offset_y = (0.05, -0.05, 0.05) scanlines (r,g,b)
    mask_type = 0.0 (APERTURE GRILLE)   mask_sample_mode = 0.0 (Lanczos resize)
    mask_triad_size_desired = 1.0 px -> CLAMPED to 2.0 px by
        mask_min_allowed_triad_size = 2.0 (user-settings.h) => 16x16 px tile
    mask_specify_num_triads = 0.0 (mask_num_triads_desired = 900 is IGNORED)
    geom_mode = 0.0 (flat)    border_darkness = 0.0 (border off)
    interlace detect on (240p => progressive)

STRUCTURAL preset differences vs crt-royale.slangp (not just parameters):
  * Pass 0 is misc/shaders/grade.slang (Dogway).  At this commit grade is the
    2023 version; MANY of kurozumi's grade overrides target the older 2020
    grade parameter set (g_gamma_in, g_gamma_type, g_I_SHIFT/g_Q_SHIFT/
    g_I_MUL/g_Q_MUL, g_satr...) and NO LONGER EXIST -> silently dropped by
    RetroArch.  The surviving grade overrides: g_signal_type=0 (RGB),
    g_crtgamut=1 (P22-90s phosphor), g_space_out=-1 (Rec.709 display,
    pure-power TRC branch), wp_temperature=6504 (D65), g_vignette=0 (OFF),
    g_lum/g_cntrst/g_sat/g_lift/tints = neutral.  Grade defaults NOT
    overridden that still act: g_CRT_l=2.50 (EOTF-1886a black level 0.0382),
    g_CRT_b=50, g_CRT_c=50 (neutral), g_Dark_to_Dim=1 (encode gamma
    2.4/0.9811), g_CRT_rf=5.0 + g_CRT_sl=34 => additive Flare offset
    (+0.0167, +0.0149, +0.0047) encoded per channel (warm black lift),
    g_GCompress=1 (identity on the gray axis).
    On the GRAY AXIS grade reduces exactly to the per-channel curve
    grade_transfer() below (gamut/CAT steps are white-preserving; residual
    < 2e-4 from the white-point polynomial fit is ignored).
  * Halation blur passes are blur5fast (sigma = 0.9845703125), not blur9fast.

Pipeline gamma: grade outputs encoded (see grade_transfer); royale pass 1
linearizes with pow(x, crt_gamma=2.4); every intermediate FBO is linear light
(sRGB8 storage, clamps at 1.0); the last pass encodes with
pow(c, 1/lcd_gamma=2.4).

Interface (contract for the automated fitting stage):
    transfer(x)         encoded->encoded flat field at beam center, mask off
    beam_weight(d, L)   linear-light beam profile weight vs distance/brightness
    ref_vertical(f, x)  encoded output across the scanline period, mask off
    h_kernel(frac)      effective horizontal kernel (Gaussian + diffusion)
    mask_spec()         kurozumi phosphor mask as rendered at 1080p
    notes()             features not representable on MiSTer
    defaults()          dict of all effective parameter values
Optional `channel` kwarg ('r'/'g'/'b') on transfer/ref_vertical applies that
channel's grade flare AND vertical convergence offset; default None models a
neutral centered channel with green flare (recommended fitting target).
"""

import math

# ----------------------------------------------------------------------------
# Effective parameters (kurozumi overrides applied)
# ----------------------------------------------------------------------------

CRT_GAMMA = 2.4                  # was 2.5
LCD_GAMMA = 2.4                  # was 2.2
LEVELS_CONTRAST = 0.67           # was 1.0  -> linear ceiling 0.67
LEVELS_AUTODIM_TEMP = 0.5
HALATION_WEIGHT = 0.0037         # was 0.0
DIFFUSION_WEIGHT = 0.0011        # was 0.075
BLOOM_UNDERESTIMATE_LEVELS = 0.68  # was 0.8
BLOOM_EXCESS = 0.0
BEAM_MIN_SIGMA = 0.02
BEAM_MAX_SIGMA = 0.16            # was 0.3 -> much thinner bright beam
BEAM_SPOT_POWER = 0.38           # was 0.33
BEAM_MIN_SHAPE = 2.0
BEAM_MAX_SHAPE = 4.0
BEAM_SHAPE_POWER = 0.25
BEAM_HORIZ_FILTER = 1.0          # was 0.0 (Quilez) -> GAUSSIAN 4-tap
BEAM_HORIZ_SIGMA = 0.32          # ACTIVE now (Gaussian path), source px
BEAM_NUM_SCANLINES = 3
CONVERGENCE_X = (-0.05, 0.0, 0.0)    # r,g,b in source texels
CONVERGENCE_Y = (0.05, -0.05, 0.05)  # r,g,b in scanlines
MASK_TYPE = 0.0                  # was 1.0 (slot) -> aperture grille
MASK_SAMPLE_MODE = 0.0
MASK_TRIAD_SIZE_DESIRED = 1.0    # requested; clamped to 2.0 px at render
MASK_TRIAD_SIZE_EFFECTIVE = 2.0  # mask_min_allowed_triad_size = 2.0
MASK_TRIADS_PER_TILE = 8.0
MASK_GRILLE_AVG_COLOR = 53.0 / 255.0   # TileableLinearApertureGrille15Wide8And5d5Spacing
MASK_AMPLIFY = 1.0 / MASK_GRILLE_AVG_COLOR      # 4.811320754716981
GEOM_MODE = 0.0

# Geometry of the modeled configuration: 240p source -> 1080p output
SOURCE_LINES = 240.0
OUTPUT_LINES = 1080.0
PIXEL_HEIGHT = SOURCE_LINES / OUTPUT_LINES      # 0.2222 scanlines/output px
SCAN_PER_OUT_PX = SOURCE_LINES / OUTPUT_LINES

# Source width assumption for h_kernel diffusion scaling (256-wide console):
SRC_WIDTH = 256.0
BLOOM_APPROX_W = 320.0
BLOOM_APPROX_H = 240.0

SQRT2 = math.sqrt(2.0)

# ----------------------------------------------------------------------------
# grade.slang (2023) gray-axis transfer at kurozumi settings
# ----------------------------------------------------------------------------

G_CRT_L = 2.50                   # grade default "CRT Gamma" (not overridden)
# CRT_l macro: black level derived from the CRT gamma knob:
GRADE_BL = -(100000.0 * math.log(
    (72981.0 - 500000.0 / (3.0 * max(2.3, G_CRT_L))) / 9058.0)) / 945461.0
# = 0.03816808... (bl on a 0..100 white-level scale)

# Flare (surround reflectance): g_CRT_rf = 5.0, g_CRT_sl = 34.0 defaults:
_FLARE_BASE = 0.01 * (5.0 / 5.0) * (0.049433 * 34.0 - 0.188367)
GRADE_FLARE = (_FLARE_BASE * (0.459993 / 0.410702),   # +0.016715 R
               _FLARE_BASE * 1.0,                     # +0.014924 G
               _FLARE_BASE * (0.129305 / 0.410702))   # +0.004699 B

GRADE_ENCODE_GAMMA = 2.4 / 0.9811   # SPC=-1 pure power TRC x Dark-to-Dim OOTF

_CH_IDX = {'r': 0, 'g': 1, 'b': 2}


def _clamp01(v):
    return max(0.0, min(1.0, v))


def _grade_eotf_1886a(c):
    """grade.slang EOTF_1886a with bl=GRADE_BL, brightness=50, contrast=50."""
    bl = GRADE_BL
    b = bl ** (1.0 / 2.4)
    a = 100.0 ** (1.0 / 2.4) - b
    b = (50.0 - 50.0) / 250.0 + b / a      # brightness 50 -> b/a only
    a = 1.0                                # contrast == 50 -> 1.0
    Vc = 0.35
    Lw = 100.0 / 100.0 * a                 # 1.0
    Lb = min(b * a, Vc)
    a1, a2 = 2.6, 3.0
    k = Lw / (1.0 + Lb) ** a1
    sl = k * (Vc + Lb) ** (a1 - a2)
    c = k * (c + Lb) ** a1 if c >= Vc else sl * (c + Lb) ** a2
    # Black lift compensation:
    bc = 0.00446395 * bl ** 1.23486
    c = min(max(c - bc, 0.0) * (1.0 / (1.0 - bc)), 1.0)
    c = c ** (1.0 - 0.00843283 * bl ** 1.22744)
    return c


def _grade_rolled_gain(c, gain=0.0):
    """grade.slang rolled_gain at g_lum = 0.0 (negative/zero branch)."""
    gx = abs(gain) + 0.001
    if gain > 0.0:
        anch = 0.5 / (gx / 2.0)
        return c * ((c - anch) / (1.0 - anch))
    anch = 0.5 / gx                        # 500.0 at gain 0
    return c * ((1.0 - anch) / (c - anch)) * (1.0 - gain)


def grade_transfer(x, channel=None):
    """Gray-axis encoded->encoded transfer of the grade pass at kurozumi
    settings (signal RGB, gamut/CAT identity on grays, vignette off,
    contrast/sat/lift neutral).  channel selects the Flare offset
    (None -> green).  Steps: EOTF-1886a decode -> rolled_gain(0) ->
    clamp01 -> encode pow(0.9811/2.4) -> + Flare, min 1.0."""
    x = _clamp01(x)
    lin = _grade_eotf_1886a(x)
    lin = _clamp01(_grade_rolled_gain(lin, 0.0))
    enc = _clamp01(lin ** (1.0 / GRADE_ENCODE_GAMMA))
    fl = GRADE_FLARE[_CH_IDX.get(channel, 1)]
    return min(enc + fl, 1.0)


def lin_input(L, channel=None):
    """Linear value entering the scanline pass for encoded source level L:
    grade gray transfer followed by royale's pow(x, crt_gamma=2.4)."""
    return grade_transfer(L, channel) ** CRT_GAMMA


# ----------------------------------------------------------------------------
# Bloom constants (triad size 2.0 px at 1080p)
# ----------------------------------------------------------------------------

BLOOM_DIFF_THRESH = 1.0 / 256.0


def _min_sigma_to_blur_triad(triad_size, thresh=BLOOM_DIFF_THRESH):
    return (-0.05168 + 0.6113 * triad_size
            - 1.122 * triad_size * math.sqrt(0.000416 + thresh))


BLOOM_SIGMA = _min_sigma_to_blur_triad(MASK_TRIAD_SIZE_EFFECTIVE)  # ~1.02339


def _fast_gaussian_weight_sum_inv(sigma):
    """blur-functions.h get_fast_gaussian_weight_sum_inv (curve fit); with
    RUNTIME_PHOSPHOR_BLOOM_SIGMA get_center_weight() returns this 1D value."""
    return min(math.exp(math.exp(0.348348412457428 / (sigma - 0.0860587260734721))),
               0.399334576340352 / sigma)


CENTER_WEIGHT = _fast_gaussian_weight_sum_inv(BLOOM_SIGMA)         # ~0.390209

# 17-tap bloom blur (static PHOSPHOR_BLOOM_TRIADS_LARGER_THAN_3_PIXELS keeps
# tex2Dblur17fast even though sigma shrank), sigma = BLOOM_SIGMA:


def _blur17_weights(sigma):
    dinv = 0.5 / (sigma * sigma)
    w = [math.exp(-(j * j) * dinv) for j in range(9)]
    ninv = _fast_gaussian_weight_sum_inv(sigma)
    return [wj * ninv for wj in w]


_BLOOM_W = _blur17_weights(BLOOM_SIGMA)

# HALATION_BLUR flat gain: blur5fast (sigma = 0.9845703125) V then H:
BLUR5_STD_DEV = 0.9845703125


def _blur5_gain():
    sigma = BLUR5_STD_DEV
    dinv = 0.5 / (sigma * sigma)
    s = 1.0 + 2.0 * sum(math.exp(-(j * j) * dinv) for j in range(1, 3))
    return s * _fast_gaussian_weight_sum_inv(sigma)


HALATION_FLAT_GAIN = _blur5_gain() ** 2         # ~0.98600 (flat field)

# BLOOM_APPROX 2D resize Gaussian sigma: mask_triad_size_desired = 1.0 ->
# mask_num_triads_runtime = max(256, 1418.18/1.0) = 1418.18; NOTE the shader
# combines with beam_max_sigma_STATIC (user-settings.h) = 0.3, NOT the
# runtime 0.16:
_MAX_VP_X = 1080.0 * 1024.0 * (4.0 / 3.0)
_EST_VP_X = 1080.0 * (432.0 / 329.0)
_N_TRIADS = max(256.0, _EST_VP_X / MASK_TRIAD_SIZE_DESIRED)        # 1418.18
_BA_SIGMA_RESIZE = (_min_sigma_to_blur_triad(_MAX_VP_X / _N_TRIADS)
                    * BLOOM_APPROX_W / _MAX_VP_X)                  # ~0.12129
BEAM_MAX_SIGMA_STATIC = 0.3
BLOOM_APPROX_SIGMA = math.hypot(_BA_SIGMA_RESIZE, BEAM_MAX_SIGMA_STATIC)
# ~0.32360

# Total diffusion/halation blur sigma in BLOOM_APPROX pixels:
_DIFF_SIGMA_BA = math.hypot(BLOOM_APPROX_SIGMA, BLUR5_STD_DEV)     # ~1.0364
DIFFUSION_SIGMA_SRC_X = _DIFF_SIGMA_BA * SRC_WIDTH / BLOOM_APPROX_W
DIFFUSION_SIGMA_SRC_Y = _DIFF_SIGMA_BA


# ----------------------------------------------------------------------------
# Beam model (generalized Gaussian, 3x sampled AA) -- kurozumi sigmas
# ----------------------------------------------------------------------------

def _gaussian_sigma(c):
    return BEAM_MIN_SIGMA + (BEAM_MAX_SIGMA - BEAM_MIN_SIGMA) * (c ** BEAM_SPOT_POWER)


def _generalized_beta(c):
    return BEAM_MIN_SHAPE + (BEAM_MAX_SHAPE - BEAM_MIN_SHAPE) * (c ** BEAM_SHAPE_POWER)


def _beam_profile(d, c, pixel_height=PIXEL_HEIGHT):
    """Spatial weight w(d, c): a line of linear value c contributes c*w(d,c).
    d in source scanlines; 3x antialias averaging at ph = 240/1080."""
    c = max(0.0, min(1.0, c))
    alpha = SQRT2 * _gaussian_sigma(c)
    beta = _generalized_beta(c)
    scale = beta * 0.5 / (alpha * math.gamma(1.0 / beta))
    off = pixel_height / 3.0
    tot = 0.0
    for dd in (d, d + off, abs(d - off)):
        tot += math.exp(-(abs(dd) / alpha) ** beta)
    return scale * tot / 3.0


def beam_weight(d, L):
    """LINEAR-light contribution weight of one scanline at vertical distance d
    (source lines) when the line's encoded brightness (at the GRADE INPUT) is
    L in [0,1].  The line's linear value is E = lin_input(L) =
    grade_transfer(L)^2.4 and its light contribution is E * beam_weight(d, L).
    Per-channel: sigma/beta come from that channel's own linear value; this is
    the scalar (gray, green-flare) form, convergence offsets NOT applied."""
    E = lin_input(_clamp01(L))
    return _beam_profile(abs(d), E)


# ----------------------------------------------------------------------------
# Vertical scanline pass for a uniform field (with optional conv offset)
# ----------------------------------------------------------------------------

def _scanline_sum(u, E, conv_y=0.0):
    """Sum of the 3 modeled scanline contributions at fractional distance u
    in [0,1) below the previous scanline, uniform linear field E, with this
    channel's vertical convergence offset conv_y (dist2 = dist - offset; the
    outer-line selector round() uses the UNSHIFTED dist, as in the shader).
    Returns LINEAR intensity (pre auto-dim)."""
    if E <= 0.0:
        return 0.0
    d2 = u - conv_y
    r = 1.0 if u >= 0.5 else 0.0
    d_outer = (1.0 - r) * (1.0 + d2) + r * (2.0 - d2)
    w = (_beam_profile(abs(d2), E) + _beam_profile(abs(1.0 - d2), E)
         + _beam_profile(abs(d_outer), E))
    return E * w


# ----------------------------------------------------------------------------
# Halation / brightpass / bloom (mask disabled: mask == 1, amplify == 1)
# ----------------------------------------------------------------------------

MASK_AMPLIFY_DISABLED = 1.0      # mask disabled => amplify replaced by 1.0


def _masked_scanline_texel(u, E, conv_y=0.0):
    """MASKED_SCANLINES texel (mask=1): halation lerp of the (clamped)
    auto-dimmed scanline value toward the desaturated halation intensity.
    Flat field: halation_color = E * HALATION_FLAT_GAIN (blur5fast x2 of
    BLOOM_APPROX); halation_intensity_dim = auto_dim * mean(halation)."""
    scan_dim = _clamp01(LEVELS_AUTODIM_TEMP * _scanline_sum(u, E, conv_y))
    hal_dim = LEVELS_AUTODIM_TEMP * E * HALATION_FLAT_GAIN
    v1 = scan_dim + (hal_dim - scan_dim) * HALATION_WEIGHT
    return _clamp01(v1)


def _brightpass(v1, E, mask_amplify=MASK_AMPLIFY_DISABLED):
    """crt-royale-brightpass.slang for a horizontally-flat field."""
    intensity = v1 * (1.0 / LEVELS_AUTODIM_TEMP) * mask_amplify * LEVELS_CONTRAST
    blur_approx = LEVELS_CONTRAST * E
    cw = CENTER_WEIGHT
    max_area = max(0.0, blur_approx - cw * intensity)
    acu = BLOOM_UNDERESTIMATE_LEVELS * max_area
    iu = BLOOM_UNDERESTIMATE_LEVELS * intensity
    if iu <= 0.0:
        blur_ratio = 0.0
    else:
        blur_ratio = ((1.0 - acu) / iu - 1.0) / (cw - 1.0)
    blur_ratio = _clamp01(blur_ratio)
    blur_ratio = blur_ratio + (1.0 - blur_ratio) * BLOOM_EXCESS
    return _clamp01(v1 * blur_ratio)


def _bp_at_phase(u, E, conv_y=0.0):
    v1 = _masked_scanline_texel(u, E, conv_y)
    return _brightpass(v1, E)


def _blurred_brightpass(u, E, conv_y=0.0):
    """17-tap vertical Gaussian blur (sigma = BLOOM_SIGMA ~1.023 output px) of
    the brightpass scanline pattern, sampled at output-pixel pitch, plus the
    horizontal pass's flat fast-normalization gain."""
    tot = _BLOOM_W[0] * _bp_at_phase(u, E, conv_y)
    for k in range(1, 9):
        for s in (-1.0, 1.0):
            uk = (u + s * k * SCAN_PER_OUT_PX) % 1.0
            tot += _BLOOM_W[k] * _bp_at_phase(uk, E, conv_y)
    h_gain = (1.0 + 2.0 * sum(math.exp(-(j * j) * 0.5 / (BLOOM_SIGMA ** 2))
                              for j in range(1, 9))) \
        * _fast_gaussian_weight_sum_inv(BLOOM_SIGMA)
    return tot * h_gain


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------

def ref_vertical(f, x, channel=None):
    """Encoded (0..1) output at vertical fraction f within one scanline period
    (f=0 = scanline grid position, f=0.5 = midway between lines) for a uniform
    encoded input x, MASK DISABLED, kurozumi parameters, 240p at 1080p.

    Chain: grade (gray-axis curve + flare) -> linearize (2.4) -> 3-scanline
    generalized-Gaussian beam sum (sigma 0.02..0.16) -> autodim 0.5 + FBO
    clamp -> halation lerp (0.37% toward flat E*0.98600) -> [mask=1] ->
    brightpass (underest 0.68, cw 0.3902) / 17-tap bloom blur (sigma 1.0234)
    -> reconstitute (dimpass + blurred brightpass) * undim * CONTRAST 0.67
    -> diffusion lerp (0.11% toward 0.67*E*0.98600) -> FBO clamp
    -> encode pow(1/2.4).

    channel='r'/'g'/'b' applies that channel's grade flare and vertical
    convergence offset (r +0.05, g -0.05, b +0.05 scanlines); None models a
    neutral centered channel with green flare."""
    x = _clamp01(x)
    f = abs(f) % 1.0
    conv_y = CONVERGENCE_Y[_CH_IDX[channel]] if channel in _CH_IDX else 0.0
    E = lin_input(x, channel)
    v1 = _masked_scanline_texel(f, E, conv_y)
    bp = _brightpass(v1, E)
    bbp = _blurred_brightpass(f, E, conv_y)
    phosphor_bloom = (v1 - bp + bbp) * MASK_AMPLIFY_DISABLED \
        * (1.0 / LEVELS_AUTODIM_TEMP) * LEVELS_CONTRAST
    diffusion_color = LEVELS_CONTRAST * E * HALATION_FLAT_GAIN
    final = (phosphor_bloom * (1.0 - DIFFUSION_WEIGHT)
             + diffusion_color * DIFFUSION_WEIGHT)
    final = _clamp01(final)      # reconstitute 8-bit FBO clamp
    return final ** (1.0 / LCD_GAMMA)


def transfer(x, channel=None):
    """End-to-end encoded->encoded transfer for a uniform field sampled at
    f=0 (scanline grid position), mask disabled.  Includes the grade pass
    (EOTF-1886a + flare), CRT gamma 2.4, beam peak gain, halation, contrast
    0.67, bloom redistribution, diffusion, FBO clipping, and LCD gamma 1/2.4.
    NOTE: this is the SCANLINE-PEAK transfer; the thin beam concentrates the
    line energy, so the peak saturates at 1.0 from x ~0.738 (reconstitute
    clamp; the scanline-FBO peak clamp starts at x ~0.878).  The 0.67
    contrast ceiling applies to the flat-field SPATIAL AVERAGE (0.67 linear
    = 0.846 encoded), not the peak.  Encoded black is ~0.026 (grade flare
    black lift)."""
    return ref_vertical(0.0, x, channel)


def h_kernel(frac):
    """Effective horizontal kernel at fractional source-pixel offset frac
    (0..1) between texel centers, as (tap_offset, weight) pairs; tap_offset
    in SOURCE pixels relative to the output sample position.

    Composition (flat-field-consistent blend):
      * 99.89% scanline H resample: GAUSSIAN 4-tap (beam_horiz_filter = 1.0),
        sigma = 0.32 source px, taps at distances (1+frac, frac, 1-frac,
        2-frac), weights exp(-d^2/(2 sigma^2)) normalized to sum 1, applied
        in LINEAR light (beam_horiz_linear_rgb_weight = 1.0).
      * 0.11% diffusion: Gaussian sigma ~0.829 source px (BLOOM_APPROX 4x4
        resize sigma 0.3236 (+) blur5 sigma 0.9846, at 320->256 scale),
        discretized on the source grid, normalized.

    The Gaussian 4-tap IS a 4-tap FIR: it maps to MiSTer's horizontal
    polyphase filter almost exactly (up to encoded-vs-linear domain).
    Convergence: red is additionally sampled 0.05 source px offset
    (convergence_offset_x_r = -0.05) -- not representable in a shared FIR;
    not folded in here.  Total kernel sums to 1.0 by construction."""
    frac = frac % 1.0
    s2 = 2.0 * BEAM_HORIZ_SIGMA * BEAM_HORIZ_SIGMA
    dists = (1.0 + frac, frac, 1.0 - frac, 2.0 - frac)   # taps -1, 0, +1, +2
    w = [math.exp(-(d * d) / s2) for d in dists]
    wsum = sum(w)
    taps = {}
    for n, wi in zip((-1, 0, 1, 2), w):
        taps[n] = (wi / wsum) * (1.0 - DIFFUSION_WEIGHT)
    # Diffusion component (0.11%), Gaussian on source grid:
    sig = DIFFUSION_SIGMA_SRC_X
    rng = range(-6, 8)
    g = {n: math.exp(-((n - frac) ** 2) / (2.0 * sig * sig)) for n in rng}
    gs = sum(g.values())
    for n in rng:
        taps[n] = taps.get(n, 0.0) + DIFFUSION_WEIGHT * g[n] / gs
    return sorted((float(n - frac), w) for n, w in taps.items())


def mask_spec():
    """KUROZUMI phosphor mask as rendered at 1920x1080 output.

    mask_type = 0 (aperture grille), mask_sample_mode = 0 (Lanczos3 sinc
    manual resize of the 64x64 'TileableLinearApertureGrille15Wide8And5d5
    SpacingResizeTo64.png' LUT), mask_triad_size_desired = 1.0 -> desired
    tile 8 px, CLAMPED by mask_min_allowed_tile_size = 16 (min triad 2.0 px)
    -> 16x16-px tile, 2.0 px/triad, 120 x 67.5 tiles on screen (grille is
    vertically ~uniform so the fractional vertical tiling is invisible;
    horizontal sampling is exactly 1:1 texel:pixel).

    CRITICAL 1080p FINDING: 3 phosphor stripes cannot be resolved in a
    2-px triad; the shader's own Lanczos resize low-passes the grille into a
    2-px-period pattern: pixel A ~ (100, 54, 14)/255, pixel B ~
    (15, 54, 100)/255 -- i.e. RED and BLUE alternate in antiphase while
    GREEN is spatially FLAT (~54/255 everywhere).  The '15-wide' grille look
    kurozumi was tuned for only exists at >= 1440p.

    Applied in LINEAR light; net multiplier = (tile/255) * mask_amplify
    (255/53 = 4.8113).  Net linear multipliers: bright phase ~1.90, dim
    phase ~0.264, green ~1.019 flat; average 1.064.  Encoded (^(1/2.2)):
    bright 1.341, dim 0.546, green 1.009."""
    avg = sum(sum(sum(px) for px in row) for row in MASK_TILE) / (16 * 16 * 3 * 255.0)
    peak = max(max(max(px) for px in row) for row in MASK_TILE) / 255.0
    return {
        "tile_width_px": 16,
        "tile_height_px": 16,
        "triad_size_px": 2.0,
        "triad_size_requested_px": 1.0,
        "tiles_on_screen": [120, 67.5],
        "mask_type": "aperture grille (collapses to 2-px RB-alternating "
                     "pattern with flat green at 1080p)",
        "sample_mode": "manual Lanczos3 resize (mask_sample_mode=0)",
        "applied_in": "linear light, multiplies scanline intensity before "
                      "brightpass/bloom; net multiplier = value * mask_amplify",
        "mask_amplify": MASK_AMPLIFY,
        "avg_transmission": avg,
        "avg_net_multiplier": avg * MASK_AMPLIFY,
        "peak_net_multiplier": peak * MASK_AMPLIFY,
        "tile_rgb_0_255": MASK_TILE,
        "tile_linear_multiplier_note":
            "per-pixel LINEAR multiplier = tile_rgb_0_255/255 * mask_amplify "
            "(4.8113); for MiSTer's encoded-domain mask stage use "
            "(linear_mult)^(1/2.2).  Representative encoded triple per pixel: "
            "bright phase (1.341, 1.009, 0.546), dim phase mirrored in R/B.",
    }


def defaults():
    return {
        "commit": "3b0d6aa1d134a168478cd9c904a866d969f8882b",
        "preset": "presets/crt-royale-kurozumi.slangp",
        "crt_gamma": CRT_GAMMA, "lcd_gamma": LCD_GAMMA,
        "levels_contrast": LEVELS_CONTRAST,
        "levels_autodim_temp": LEVELS_AUTODIM_TEMP,
        "halation_weight": HALATION_WEIGHT,
        "diffusion_weight": DIFFUSION_WEIGHT,
        "bloom_underestimate_levels": BLOOM_UNDERESTIMATE_LEVELS,
        "bloom_excess": BLOOM_EXCESS,
        "beam_min_sigma": BEAM_MIN_SIGMA, "beam_max_sigma": BEAM_MAX_SIGMA,
        "beam_spot_power": BEAM_SPOT_POWER,
        "beam_min_shape": BEAM_MIN_SHAPE, "beam_max_shape": BEAM_MAX_SHAPE,
        "beam_shape_power": BEAM_SHAPE_POWER,
        "beam_generalized_gaussian": True,
        "beam_antialias_level": 1.0,
        "beam_num_scanlines": BEAM_NUM_SCANLINES,
        "beam_horiz_filter": "Gaussian (1.0)",
        "beam_horiz_sigma": BEAM_HORIZ_SIGMA,
        "beam_horiz_linear_rgb_weight": 1.0,
        "convergence_offsets_x_rgb": CONVERGENCE_X,
        "convergence_offsets_y_rgb": CONVERGENCE_Y,
        "mask_type": "aperture grille (0.0)",
        "mask_sample_mode": 0.0,
        "mask_triad_size_desired": MASK_TRIAD_SIZE_DESIRED,
        "mask_triad_size_effective": MASK_TRIAD_SIZE_EFFECTIVE,
        "mask_amplify": MASK_AMPLIFY,
        "geom_mode": "off (0.0)", "border_darkness": 0.0,
        "interlacing": "auto-detect; 240p is progressive",
        "grade": {
            "version_note": "2023 grade; kurozumi's 2020-era grade overrides "
                            "(g_gamma_in, g_gamma_type, IQ knobs) are dropped",
            "g_signal_type": 0.0, "g_crtgamut": "P22-90s (1.0)",
            "g_space_out": "Rec.709 / pure power (-1.0)",
            "wp_temperature": 6504.0, "g_vignette": 0.0,
            "g_CRT_l": G_CRT_L, "grade_black_level": GRADE_BL,
            "grade_encode_gamma": GRADE_ENCODE_GAMMA,
            "grade_flare_rgb": GRADE_FLARE,
        },
        "bloom_sigma_at_triad2": BLOOM_SIGMA,
        "bloom_center_weight": CENTER_WEIGHT,
        "bloom_approx_sigma": BLOOM_APPROX_SIGMA,
        "halation_flat_gain": HALATION_FLAT_GAIN,
        "halation_blur": "blur5fast x2 (sigma 0.9845703125), NOT blur9fast",
        "pixel_height_scanlines": PIXEL_HEIGHT,
        "source": "240p", "output": "1080p",
    }


def notes():
    return [
        "GRADE PASS: prepended misc/shaders/grade.slang.  On the gray axis it "
        "is a fixed per-channel monotone curve (EOTF-1886a bl=0.0382 decode, "
        "re-encode gamma 2.4/0.9811) plus an ADDITIVE per-channel flare "
        "(+0.0167 R, +0.0149 G, +0.0047 B encoded -> warm black lift).  This "
        "folds EXACTLY into MiSTer's 3-channel 256-entry gamma LUT (compose "
        "grade_transfer with the fitted royale-side curve per channel).  Its "
        "COLOR part (P22-90s->709 gamut matrix + CAT16) is a 3x3 matrix in "
        "linear light -- NOT representable in MiSTer's scaler (pointwise LUT "
        "only); off-gray colors shift slightly (identity on the gray axis).",
        "CONTRAST CEILING: levels_contrast = 0.67 caps flat-field linear "
        "output at 0.67 (encoded 0.67^(1/2.4) = 0.846).  On MiSTer bake the "
        "ceiling into the gamma LUT (max entry ~216/255), NOT into the "
        "V-filter DC gain, so beam-peak clipping behavior stays correct.",
        "SCANLINES DO NOT COLLAPSE AT WHITE: beam_max_sigma = 0.16 keeps "
        "beam_weight(0.5, 1.0) ~ 3e-6 (AA samples); the white-field trough "
        "is fed only by bloom redistribution + 0.11% diffusion + 0.37% "
        "halation (encoded trough 0.119 at white).  Encoded depth stays "
        "0.38..0.93 at ALL brightness levels (vs royale collapsing to 0.016 "
        "at white).  Moire risk at 4.5x is MUCH higher than default royale "
        "-- weigh trough-stddev heavily in the fit.",
        "BEAM-PEAK CLIPPING: with sigma_max 0.16 the beam peak w(0) is 2.42 "
        "at white (4.44 at L=0.25!); scanline peaks saturate the encoded "
        "output at 1.0 from x ~0.738 (reconstitute clamp, x0.67 undim x2), "
        "and the scanline FBO itself clamps from x ~0.878 (0.5*E*w0 >= 1). "
        "Flat-field brightpass only activates at x >= 0.976.  transfer() is "
        "therefore saturated over the top ~26% of the input range -- an "
        "8-bit-clamp behavior MiSTer reproduces naturally IF the V-filter "
        "peak rows are allowed to exceed unity DC... which MiSTer clamps "
        "per-tap/row-sum: instead reproduce via the gamma LUT reaching 255 "
        "early + a bright V set whose center tap carries most of the row "
        "sum (accept softer peak saturation).",
        "HORIZONTAL FILTER IS A REAL 4-TAP FIR: beam_horiz_filter = 1.0 is a "
        "normalized 4-tap Gaussian (sigma 0.32 source px) -- an almost exact "
        "match for MiSTer's 4-tap/256-phase H filter (fit per phase; only "
        "the linear-vs-encoded domain difference remains).  This is a "
        "BETTER H match than default royale's Quilez.",
        "MASK AT 1080p IS DEGENERATE: requested 1.0-px triads clamp to 2.0 "
        "px; the shader's own Lanczos resize collapses the grille to a "
        "2-px-period R/B-alternating pattern with spatially FLAT green "
        "(see mask_spec).  Required encoded per-pixel triple ~ (1.341, "
        "1.009, 0.546) / mirrored.  MiSTer v2 tokens give one boosted "
        "channel (16+Y)/16 and ONE shared Z/16 for the other two -> the "
        "triple is NOT exactly representable (G needs ~1.01 while B needs "
        "~0.55).  Closest options: [a] preserve luma: sel R Y=5 (1.3125), "
        "Z=15 (0.9375) -> B 72% too bright (linear 3.1x), mask reads as "
        "faint warm/cool ripple; [b] preserve chroma: Z=9 (0.5625) -> green "
        "drops 44%, image darkens ~30% -- NOT recommended; [c] omit the "
        "mask (avg net gain 1.064, fold 1.06 into the LUT) -- honest, since "
        "at 1080p+4K-2x the real preset's mask is already nearly invisible. "
        "Recommend [a] or [c].",
        "MASK PUNCH vs MiSTer CEILING: kurozumi's peak net linear multiplier "
        "is only 1.906 (encoded 1.341 < MiSTer's 1.9375 ceiling) because the "
        "2-px resize dilutes the grille; the '170%+' punch of the full-res "
        "grille (peak ~4.8 linear, 2.04 encoded) would exceed the ceiling, "
        "but that geometry never renders at 1080p.",
        "BLOOM/BRIGHTPASS: still present (underestimate 0.68, 17-tap blur "
        "sigma 1.0234 output px) but with the mask nearly flat it activates "
        "only near white (flat-field onset x ~0.92); magnitude is small.  On "
        "MiSTer, over-unity mask multipliers + 8-bit clamp substitute as "
        "before; with option [c] (no mask) there is nothing to substitute.",
        "HALATION (0.37%) & DIFFUSION (0.11%): both are 2D blurs (sigma "
        "~0.83 source px H, ~1.04 source lines V) of the pre-scanline image "
        "mixed post-mask.  At these weights they only add a ~0.5% flat veil "
        "-- fold the DC into the gamma LUT / V-filter HI set; the spatial "
        "spread is below MiSTer's representability threshold and below "
        "visibility.  (Halation also desaturates -- mean(RGB) -- ignored at "
        "0.37%.)",
        "CONVERGENCE: x = (-0.05, 0, 0) texels, y = (+0.05, -0.05, +0.05) "
        "scanlines.  MiSTer's H FIR and V FIR are shared across channels -> "
        "per-channel offsets are NOT representable.  Magnitude: 0.05 src px "
        "= 0.375 output px H (at 7.5x for 256-wide), 0.05 lines = 0.225 "
        "output px V -- subpixel; visible only as a faint red fringe. Omit.",
        "GAMMA: crt 2.4 / lcd 2.4 cancel on the DC transfer EXCEPT for the "
        "clamps and the grade curve in between; the end-to-end transfer() is "
        "close to grade_transfer scaled by 0.67^(1/2.4) with beam-clip "
        "saturation above x~0.92.  Fit the MiSTer gamma LUT to transfer() "
        "per channel (r/g/b flare variants), and the V filter to "
        "ref_vertical/transfer ratios.",
        "INTERLACING: 240p progressive; 288<lines<=576 would bob (not "
        "expressible on MiSTer).  interlace_1080i off.",
        "PER-CHANNEL BEAM WIDTH: as in royale, sigma/beta are per-channel "
        "from each channel's linear value; MiSTer's single max-RGB adaptive "
        "control loses this on saturated colors.",
        "NOT MODELED: 8-bit sRGB FBO quantization between passes (and the "
        "RGBA8 UNORM grade FBO); exact math.gamma instead of the shader's "
        "gamma approximation (rel err ~5e-4); BLOOM_APPROX texel-grid "
        "alignment idealized (affects only the 0.48% veil); wp_adjust "
        "white-point polynomial residual (<2e-4) treated as exact identity "
        "on grays.",
    ]


# ----------------------------------------------------------------------------
# MASK_TILE: 16x16x3, 0..255, computed with the shader's exact Lanczos3
# resize (24-sample static loop, magnification 16/64, 8-bit FBO quantization
# after each pass) from TileableLinearApertureGrille15Wide8And5d5Spacing
# ResizeTo64.png.  Vertically near-constant (max row deviation 3/255).
# ----------------------------------------------------------------------------
MASK_TILE = [
[[100,54,14],[15,54,100],[100,54,15],[15,55,100],[100,53,14],[14,54,101],[100,54,15],[15,54,100],[101,55,14],[15,55,100],[101,54,14],[14,54,101],[100,54,15],[14,54,100],[101,53,15],[15,53,100]],
[[100,55,14],[15,55,100],[101,54,14],[14,55,100],[100,54,14],[14,54,100],[100,54,14],[15,55,99],[101,54,15],[14,55,100],[100,55,14],[15,55,100],[100,54,15],[14,54,100],[101,54,15],[15,53,100]],
[[100,54,15],[14,54,100],[100,56,15],[15,55,100],[100,54,14],[14,55,100],[100,54,15],[15,54,100],[100,54,15],[14,54,100],[100,54,15],[15,55,100],[100,54,14],[14,54,100],[100,55,14],[14,55,100]],
[[100,55,15],[14,55,100],[100,55,15],[15,55,100],[100,54,14],[15,54,100],[100,55,14],[15,55,100],[100,55,15],[15,55,100],[101,54,15],[15,54,99],[100,55,14],[15,55,100],[100,55,15],[15,55,100]],
[[101,54,14],[15,54,100],[101,54,15],[14,54,100],[100,55,15],[15,55,100],[100,54,15],[14,55,100],[100,55,15],[14,55,100],[100,54,14],[15,54,100],[100,54,15],[15,54,100],[99,55,15],[15,55,100]],
[[100,54,14],[14,54,100],[100,55,15],[14,55,100],[100,54,15],[14,55,100],[100,54,15],[15,55,100],[100,54,14],[14,54,100],[100,55,15],[15,55,100],[100,54,15],[14,54,100],[99,54,14],[15,55,100]],
[[100,54,15],[15,54,100],[100,55,15],[15,55,100],[100,54,15],[14,54,100],[100,55,14],[14,55,100],[100,54,14],[15,54,101],[101,55,15],[15,55,100],[100,54,14],[14,54,100],[100,55,15],[14,55,100]],
[[100,55,15],[15,55,100],[101,54,15],[14,54,100],[100,54,14],[15,54,100],[99,53,14],[15,53,99],[100,55,15],[15,55,100],[100,54,15],[15,54,100],[100,55,14],[15,55,100],[101,55,15],[15,56,101]],
[[101,54,15],[15,54,100],[100,55,15],[15,54,100],[100,55,14],[15,54,101],[100,55,15],[15,54,101],[101,55,15],[15,55,100],[101,54,15],[15,55,100],[100,55,15],[15,55,100],[100,54,14],[15,54,101]],
[[100,54,15],[15,55,100],[100,55,15],[14,55,100],[100,54,14],[15,54,101],[100,55,15],[14,55,100],[100,54,15],[15,54,100],[100,55,15],[15,55,100],[100,54,15],[14,54,100],[100,55,14],[14,55,100]],
[[100,54,14],[15,54,101],[100,55,14],[14,55,100],[100,54,14],[15,54,101],[101,56,15],[15,56,100],[100,54,15],[15,54,100],[101,54,15],[15,54,101],[101,54,15],[15,54,100],[100,56,15],[15,56,100]],
[[100,54,14],[15,54,100],[100,55,14],[15,54,101],[101,55,15],[15,56,100],[100,54,14],[14,54,99],[100,54,14],[14,54,100],[100,54,15],[15,54,100],[100,54,15],[15,54,100],[100,54,14],[14,54,100]],
[[100,54,15],[15,55,100],[100,54,14],[15,54,100],[100,55,14],[15,55,100],[100,54,15],[15,54,100],[100,54,14],[14,54,100],[100,55,15],[15,55,100],[101,55,14],[14,55,100],[100,54,14],[15,54,100]],
[[100,54,15],[15,55,100],[100,54,15],[15,54,100],[101,54,14],[14,53,101],[100,55,15],[15,55,100],[100,54,14],[14,54,100],[100,55,15],[14,55,100],[100,55,14],[14,54,100],[100,55,14],[14,55,100]],
[[100,54,15],[15,54,100],[100,55,15],[14,55,100],[100,54,15],[15,54,100],[100,54,15],[15,55,101],[100,53,14],[15,54,100],[100,54,15],[14,54,100],[101,55,14],[15,54,100],[100,55,14],[15,55,100]],
[[100,55,15],[14,54,100],[100,55,15],[14,56,100],[100,54,15],[15,55,100],[100,54,15],[14,55,100],[100,55,14],[15,55,100],[100,54,15],[15,54,100],[100,54,14],[14,55,100],[100,54,14],[15,54,100]]
]

if __name__ == "__main__":
    print("defaults:", defaults())
    print("grade_transfer(0.5) =", grade_transfer(0.5))
    print("transfer(0.5) =", transfer(0.5))
    print("ref_vertical(0.5, 0.5) =", ref_vertical(0.5, 0.5))
    print("beam_weight(0, 0.5) =", beam_weight(0.0, 0.5))
    print("h_kernel(0.5) =", h_kernel(0.5))
    ms = mask_spec()
    print("mask avg transmission =", ms["avg_transmission"])
