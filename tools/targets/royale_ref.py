"""
royale_ref.py -- Reference implementation of CRT-ROYALE (TroggleMonkey) at
DEFAULT runtime parameters, from libretro/slang-shaders pinned to commit
3b0d6aa1d134a168478cd9c904a866d969f8882b (crt/crt-royale.slangp, 12 passes).

Models the DEFAULT-parameter math for a 240p source displayed at 1920x1080
(4.5x vertical scale). Pure Python (math only; numpy not required).

Key defaults (all from #pragma parameter lines in src/bind-shader-params.h;
the .slangp contains NO parameter overrides, and RUNTIME_SHADER_PARAMS_ENABLE
is #defined so runtime defaults -- not the static user-settings.h values --
are in force. In particular all convergence offsets are 0.0 at runtime even
though the static defaults are non-zero):

    crt_gamma = 2.5           lcd_gamma = 2.2         levels_contrast = 1.0
    halation_weight = 0.0     diffusion_weight = 0.075
    bloom_underestimate_levels = 0.8   bloom_excess = 0.0
    beam_min_sigma = 0.02     beam_max_sigma = 0.3    beam_spot_power = 0.33
    beam_min_shape = 2.0      beam_max_shape = 4.0    beam_shape_power = 0.25
    beam_horiz_filter = 0.0 (Quilez)   beam_horiz_sigma = 0.35 (unused w/ Quilez)
    beam_horiz_linear_rgb_weight = 1.0 (all-linear H mixing)
    mask_type = 1.0 (slot)    mask_sample_mode = 0.0 (Lanczos manual resize)
    mask_triad_size_desired = 3.0 px   (=> 24x24 px tile, 8 triads/tile)
    geom_mode = 0.0 (curvature OFF)    interlace_detect on (240p => progressive)
    levels_autodim_temp = 0.5 (static; dim in scanline pass, undone later)

Static codepath choices (user-settings.h): beam_generalized_gaussian = true,
beam_antialias_level = 1.0 (3x sampled AA, offsets +/- pixel_height/3),
beam_num_scanlines = 3.0, bloom_approx_filter = 2.0 (true 4x4 Gaussian resize),
PHOSPHOR_BLOOM_TRIADS_LARGER_THAN_3_PIXELS (17-tap bloom blur).

Pipeline gamma: pass0 linearizes with pow(x, crt_gamma=2.5); every
intermediate FBO is linear light (sRGB8 storage); the last pass encodes with
pow(c, 1/lcd_gamma=2.2).

Interface (contract for the automated fitting stage):
    transfer(x)         encoded->encoded flat field at beam center, mask off
    beam_weight(d, L)   linear-light beam profile weight vs distance/brightness
    ref_vertical(f, x)  encoded output across the scanline period, mask off
    h_kernel(frac)      effective horizontal kernel (Quilez + diffusion blend)
    mask_spec()         default phosphor mask as rendered at 1080p
    notes()             features not representable on MiSTer
    defaults()          dict of all default parameter values
"""

import math

# ----------------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------------

CRT_GAMMA = 2.5
LCD_GAMMA = 2.2
LEVELS_CONTRAST = 1.0
LEVELS_AUTODIM_TEMP = 0.5
HALATION_WEIGHT = 0.0
DIFFUSION_WEIGHT = 0.075
BLOOM_UNDERESTIMATE_LEVELS = 0.8
BLOOM_EXCESS = 0.0
BEAM_MIN_SIGMA = 0.02
BEAM_MAX_SIGMA = 0.3
BEAM_SPOT_POWER = 0.33          # runtime default (static was 1/3)
BEAM_MIN_SHAPE = 2.0
BEAM_MAX_SHAPE = 4.0
BEAM_SHAPE_POWER = 0.25
BEAM_HORIZ_FILTER = 0.0         # Quilez
BEAM_HORIZ_SIGMA = 0.35         # unused by Quilez path
BEAM_NUM_SCANLINES = 3
MASK_TYPE = 1.0                 # slot
MASK_SAMPLE_MODE = 0.0
MASK_TRIAD_SIZE_DESIRED = 3.0
MASK_TRIADS_PER_TILE = 8.0
MASK_SLOT_AVG_COLOR = 46.0 / 255.0
MASK_AMPLIFY = 1.0 / MASK_SLOT_AVG_COLOR        # 5.543478...
GEOM_MODE = 0.0

# Geometry of the modeled configuration: 240p source -> 1080p output
SOURCE_LINES = 240.0
OUTPUT_LINES = 1080.0
PIXEL_HEIGHT = SOURCE_LINES / OUTPUT_LINES      # 0.2222 scanlines/output px
SCAN_PER_OUT_PX = SOURCE_LINES / OUTPUT_LINES   # same thing (progressive)

# Source width assumption for h_kernel diffusion scaling (256-wide console):
SRC_WIDTH = 256.0
BLOOM_APPROX_W = 320.0
BLOOM_APPROX_H = 240.0

SQRT2 = math.sqrt(2.0)

# Bloom sigma at triad size 3.0 (bloom-functions.h get_min_sigma_to_blur_triad
# with bloom_diff_thresh = 1/256):
BLOOM_DIFF_THRESH = 1.0 / 256.0
def _min_sigma_to_blur_triad(triad_size, thresh=BLOOM_DIFF_THRESH):
    return (-0.05168 + 0.6113 * triad_size
            - 1.122 * triad_size * math.sqrt(0.000416 + thresh))

BLOOM_SIGMA = _min_sigma_to_blur_triad(3.0)     # ~1.5602 output px

def _fast_gaussian_weight_sum_inv(sigma):
    """blur-functions.h get_fast_gaussian_weight_sum_inv (curve fit).
    NOTE: with RUNTIME_PHOSPHOR_BLOOM_SIGMA (default), get_center_weight()
    returns this 1D value directly (it is NOT squared as in the static path)."""
    return min(math.exp(math.exp(0.348348412457428 / (sigma - 0.0860587260734721))),
               0.399334576340352 / sigma)

CENTER_WEIGHT = _fast_gaussian_weight_sum_inv(BLOOM_SIGMA)

# 17-tap bloom blur kernel (tex2Dblur17fast), sigma = BLOOM_SIGMA,
# normalized by the *fast approximation* (kernel sum is ~1.0009, replicated):
def _blur17_weights(sigma):
    dinv = 0.5 / (sigma * sigma)
    w = [math.exp(-(j * j) * dinv) for j in range(9)]
    ninv = _fast_gaussian_weight_sum_inv(sigma)
    return [wj * ninv for wj in w]              # w[0] center, w[1..8] each side

_BLOOM_W = _blur17_weights(BLOOM_SIGMA)

# HALATION_BLUR flat-field gain: blur9fast (sigma=1.7533203125, default
# blur9_std_dev) applied twice (V then H), each normalized by the fast approx:
def _blur9_gain():
    sigma = 1.7533203125
    dinv = 0.5 / (sigma * sigma)
    s = 1.0 + 2.0 * sum(math.exp(-(j * j) * dinv) for j in range(1, 5))
    return s * _fast_gaussian_weight_sum_inv(sigma)

HALATION_FLAT_GAIN = _blur9_gain() ** 2         # ~0.98336 (flat field)

# BLOOM_APPROX 2D resize Gaussian sigma (for h_kernel diffusion component):
# get_bloom_approx_sigma, 4x4 Gaussian path, estimated_viewport_size_x =
# 1080 * (432/329); mask_num_triads_runtime = max(256, 1418.18/3.0) = 472.73;
# asymptotic over max_viewport_size_x = 1080*1024*4/3:
_MAX_VP_X = 1080.0 * 1024.0 * (4.0 / 3.0)
_EST_VP_X = 1080.0 * (432.0 / 329.0)
_N_TRIADS = max(256.0, _EST_VP_X / MASK_TRIAD_SIZE_DESIRED)
_BA_SIGMA_RESIZE = (_min_sigma_to_blur_triad(_MAX_VP_X / _N_TRIADS)
                    * BLOOM_APPROX_W / _MAX_VP_X)
BLOOM_APPROX_SIGMA = math.hypot(_BA_SIGMA_RESIZE, BEAM_MAX_SIGMA)  # ~0.4716

# Total diffusion blur sigma in BLOOM_APPROX pixels (resize gauss + blur9 H):
_DIFF_SIGMA_BA = math.hypot(BLOOM_APPROX_SIGMA, 1.7533203125)      # ~1.8156
# In source pixels (horizontal, 256-wide source resampled to 320):
DIFFUSION_SIGMA_SRC_X = _DIFF_SIGMA_BA * SRC_WIDTH / BLOOM_APPROX_W
# In source lines (vertical, 240 -> 240 is 1:1):
DIFFUSION_SIGMA_SRC_Y = _DIFF_SIGMA_BA


# ----------------------------------------------------------------------------
# Beam model (scanline-functions.h, generalized Gaussian, sampled w/ 3x AA)
# ----------------------------------------------------------------------------

def _gaussian_sigma(c):
    """get_gaussian_sigma: power spot-shape function (beam_spot_shape_function=0)."""
    return BEAM_MIN_SIGMA + (BEAM_MAX_SIGMA - BEAM_MIN_SIGMA) * (c ** BEAM_SPOT_POWER)

def _generalized_beta(c):
    """get_generalized_gaussian_beta."""
    return BEAM_MIN_SHAPE + (BEAM_MAX_SHAPE - BEAM_MIN_SHAPE) * (c ** BEAM_SHAPE_POWER)

def _beam_profile(d, c, pixel_height=PIXEL_HEIGHT):
    """scanline_generalized_gaussian_sampled_contrib / color, i.e. the spatial
    weight w(d, c) so that a line of linear value c contributes c*w(d,c).
    d in source scanlines; c is the line's LINEAR value (per channel).
    beam_antialias_level=1 -> average 3 samples at d, d+ph/3, |d-ph/3|."""
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
    (source lines, may be fractional; valid for |d| <= 2+) when the line's
    encoded brightness is L in [0,1].  The line's linear value is E = L^2.5 and
    its light contribution to the sample is E * beam_weight(d, L).
    Per-channel color: the shader evaluates sigma/beta *independently per RGB
    channel* from that channel's linear value; for gray inputs all channels
    are identical (this function is the per-channel scalar form).
    Includes the shader's 3x antialias averaging at pixel_height = 240/1080."""
    E = max(0.0, min(1.0, L)) ** CRT_GAMMA
    return _beam_profile(abs(d), E)


# ----------------------------------------------------------------------------
# Vertical scanline pass (pass1) for a uniform field
# ----------------------------------------------------------------------------

def _scanline_sum(u, E):
    """Sum of the 3 modeled scanline contributions (beam_num_scanlines = 3)
    at fractional distance u in [0,1) below the previous scanline, uniform
    linear field E. Lines: #2 at dist u, #3 at 1-u, and #1-or-#4 at
    lerp(1+u, 2-u, round(u)). Returns LINEAR intensity (pre auto-dim)."""
    if E <= 0.0:
        return 0.0
    r = 1.0 if u >= 0.5 else 0.0
    d_outer = (1.0 - r) * (1.0 + u) + r * (2.0 - u)
    w = (_beam_profile(u, E) + _beam_profile(1.0 - u, E)
         + _beam_profile(d_outer, E))
    return E * w

def _clamp01(v):
    return max(0.0, min(1.0, v))


# ----------------------------------------------------------------------------
# Brightpass / bloom (passes 8-10), mask disabled (mask == 1, amplify == 1)
# ----------------------------------------------------------------------------

def _brightpass(v1, E, mask_amplify=1.0):
    """crt-royale-brightpass.slang for a horizontally-flat field.
    v1 = auto-dimmed masked scanline value (MASKED_SCANLINES texel).
    E  = flat-field linear value (== BLOOM_APPROX on a flat field).
    Returns the BRIGHTPASS texel (auto-dimmed, unamplified)."""
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

def _bp_at_phase(u, E):
    v1 = _clamp01(LEVELS_AUTODIM_TEMP * _scanline_sum(u, E))
    return _brightpass(v1, E)

def _blurred_brightpass(u, E):
    """17-tap vertical Gaussian blur (tex2Dblur17fast, sigma=BLOOM_SIGMA) of
    the brightpass scanline pattern, sampled at output-pixel pitch. Output
    rows step the scanline phase by 240/1080 per pixel. (The horizontal bloom
    blur is a no-op on a horizontally flat field except for the ~1.0009
    fast-normalization gain, which IS replicated per pass.)"""
    tot = _BLOOM_W[0] * _bp_at_phase(u, E)
    for k in range(1, 9):
        for s in (-1.0, 1.0):
            uk = (u + s * k * SCAN_PER_OUT_PX) % 1.0
            tot += _BLOOM_W[k] * _bp_at_phase(uk, E)
    # horizontal pass gain on a flat field (fast-normalization excess):
    h_gain = (1.0 + 2.0 * sum(math.exp(-(j * j) * 0.5 / (BLOOM_SIGMA ** 2))
                              for j in range(1, 9))) \
        * _fast_gaussian_weight_sum_inv(BLOOM_SIGMA)
    return tot * h_gain


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------

def ref_vertical(f, x):
    """Encoded (0..1) output at vertical fraction f within one scanline period
    (f=0 = scanline center, f=0.5 = midway between lines) for a uniform
    encoded input x, MASK DISABLED (mask == 1, mask_amplify == 1), all other
    defaults. 240p source at 1080p output.

    Chain: linearize (2.5) -> 3-scanline generalized-Gaussian beam sum
    -> autodim 0.5 (FBO clamp) -> [mask=1] -> brightpass/17x bloom blur
    -> reconstitute (dimpass + blurred brightpass) * undim
    -> diffusion lerp (7.5% toward flat HALATION_BLUR = E * 0.98336)
    -> FBO clamp -> encode pow(1/2.2)."""
    x = _clamp01(x)
    f = abs(f) % 1.0
    E = x ** CRT_GAMMA
    v1 = _clamp01(LEVELS_AUTODIM_TEMP * _scanline_sum(f, E))
    bp = _brightpass(v1, E)
    bbp = _blurred_brightpass(f, E)
    phosphor_bloom = (v1 - bp + bbp) * MASK_AMPLIFY_DISABLED \
        * (1.0 / LEVELS_AUTODIM_TEMP) * LEVELS_CONTRAST
    diffusion_color = LEVELS_CONTRAST * E * HALATION_FLAT_GAIN
    final = (phosphor_bloom * (1.0 - DIFFUSION_WEIGHT)
             + diffusion_color * DIFFUSION_WEIGHT)
    final = _clamp01(final)      # pass-10 8-bit FBO clamp
    return final ** (1.0 / LCD_GAMMA)

MASK_AMPLIFY_DISABLED = 1.0      # mask disabled => amplify replaced by 1.0

def transfer(x):
    """End-to-end encoded->encoded transfer for a uniform field sampled at the
    scanline beam CENTER (f=0), mask disabled. Includes CRT gamma 2.5, beam
    peak gain, autodim/undim, bloom redistribution at high brightness,
    diffusion (7.5% flat mix), FBO clipping, and LCD gamma 1/2.2."""
    return ref_vertical(0.0, x)

def h_kernel(frac):
    """Effective horizontal kernel at fractional source-pixel offset frac
    (0..1) between texel centers, as (tap_offset, weight) pairs; tap_offset is
    in SOURCE pixels relative to the output sample position (offset = texel
    position - sample position).

    Composition (flat-field-consistent blend):
      * 92.5% scanline H resample: Quilez smootherstep between the 2
        neighboring texels (beam_horiz_filter = 0.0), done in LINEAR light
        (beam_horiz_linear_rgb_weight = 1.0). Weights: w2 = s^3(s(6s-15)+10)
        at s=frac; taps (1-w2) at -frac and w2 at 1-frac.
      * 7.5% diffusion: HALATION_BLUR = Gaussian (sigma ~1.4525 source px for
        a 256-wide source: BLOOM_APPROX 4x4 resize sigma 0.4716 (+) blur9
        sigma 1.7533, at 320->256 scale), discretized on the source grid and
        normalized to sum 1.

    APPROXIMATION ERROR: the diffusion term is a 2D blur of the pre-scanline
    image applied *after* the mask/scanlines (additive veil), not a true part
    of the scanline H kernel; folding it in here is exact only for
    horizontally flat neighborhoods and ignores its vertical extent
    (sigma ~1.82 source lines) and the ~0.9834 flat gain. It also ignores
    bloom-blur redistribution (only active where the mask/brightness clip).
    Total blended kernel sums to 1.0 by construction."""
    frac = frac % 1.0
    s = frac
    w2 = s * s * s * (s * (s * 6.0 - 15.0) + 10.0)
    taps = {}
    # Quilez component (92.5%)
    taps[0] = (1.0 - w2) * (1.0 - DIFFUSION_WEIGHT)
    taps[1] = w2 * (1.0 - DIFFUSION_WEIGHT)
    # Diffusion component (7.5%), Gaussian on source grid
    sig = DIFFUSION_SIGMA_SRC_X
    rng = range(-6, 8)
    g = {n: math.exp(-((n - frac) ** 2) / (2.0 * sig * sig)) for n in rng}
    gs = sum(g.values())
    for n in rng:
        taps[n] = taps.get(n, 0.0) + DIFFUSION_WEIGHT * g[n] / gs
    return sorted((float(n - frac), w) for n, w in taps.items())
    # tap_offset = n - frac: position of source texel n relative to the sample

def mask_spec():
    """DEFAULT phosphor mask as rendered at 1920x1080 output.

    mask_type = 1 (slot mask), mask_sample_mode = 0 (Lanczos3 sinc manual
    resize of the 64x64 'TileableLinearSlotMaskTall15Wide9And4d5Horizontal
    9d14VerticalSpacingResizeTo64.png' LUT), mask_triad_size_desired = 3.0.
    get_resized_mask_tile_size clamps/floors to an integer 24x24-px tile
    (8 triads/tile -> 3.0 px/triad, 80x45 tiles on screen, nearest-sampled at
    exactly 1:1 texel:pixel).  MASK_TILE below is the tile computed with the
    shader's exact resize algorithm (24-sample static-loop Lanczos3, 8-bit
    FBO quantization after each of the two passes), values 0..255.

    Applied in LINEAR light: MASKED_SCANLINES = scanlines * (tile/255); the
    reconstitute pass then multiplies by mask_amplify = 1/(46/255) = 5.5435.
    Net per-pixel linear multiplier = (tile/255) * 5.5435 (mean ~1.048, peak
    ~3.543); energy clipped at bright phosphor dots is redistributed by the
    17x17 bloom blur (sigma 1.5602 px)."""
    avg = sum(sum(sum(px) for px in row) for row in MASK_TILE) / (24 * 24 * 3 * 255.0)
    return {
        "tile_width_px": 24,
        "tile_height_px": 24,
        "triad_size_px": 3.0,
        "tiles_on_screen": [80, 45],
        "mask_type": "slot",
        "sample_mode": "manual Lanczos3 resize (mask_sample_mode=0)",
        "applied_in": "linear light, multiplies scanline intensity before "
                      "brightpass/bloom; net multiplier = value * mask_amplify",
        "mask_amplify": MASK_AMPLIFY,
        "avg_transmission": avg,
        "avg_net_multiplier": avg * MASK_AMPLIFY,
        "tile_rgb_0_255": MASK_TILE,
        "tile_linear_multiplier_note":
            "per-pixel LINEAR multiplier = tile_rgb_0_255/255 * mask_amplify "
            "(5.5435); for MiSTer's encoded-domain mask stage use "
            "(linear_mult)^(1/2.2)",
    }

def defaults():
    return {
        "commit": "3b0d6aa1d134a168478cd9c904a866d969f8882b",
        "preset": "crt/crt-royale.slangp",
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
        "beam_horiz_filter": "Quilez (0.0)",
        "beam_horiz_sigma": BEAM_HORIZ_SIGMA,
        "beam_horiz_linear_rgb_weight": 1.0,
        "convergence_offsets": 0.0,
        "mask_type": "slot (1.0)",
        "mask_sample_mode": 0.0,
        "mask_triad_size_desired": MASK_TRIAD_SIZE_DESIRED,
        "mask_amplify": MASK_AMPLIFY,
        "geom_mode": "off (0.0)",
        "interlacing": "auto-detect; 240p is progressive",
        "bloom_sigma_at_triad3": BLOOM_SIGMA,
        "bloom_center_weight": CENTER_WEIGHT,
        "bloom_approx_sigma": BLOOM_APPROX_SIGMA,
        "halation_flat_gain": HALATION_FLAT_GAIN,
        "pixel_height_scanlines": PIXEL_HEIGHT,
        "source": "240p", "output": "1080p",
    }

def notes():
    return [
        "MASK GEOMETRY: default slot mask tile is 24x24 px with 8 slot triads "
        "(3.0 px/triad) at 1080p, RGB sub-stripes ~1 px each with vertical "
        "slot phase alternating every 3 px column-pair. MiSTer's v2 mask tile "
        "(per-pixel per-channel 1/16-step multipliers) CAN represent a 24x24 "
        "tile but only in 1/16 steps and in ENCODED space; convert linear "
        "multipliers m to encoded m^(1/2.2) and quantize. Peak net linear "
        "multiplier is 3.54 (encoded 1.78) -- above MiSTer's (16+15)/16=1.94 "
        "encoded ceiling? No: 1.78 < 1.94, representable, but 1/16 "
        "quantization (~6%) will flatten the dim slot gaps (min 0).",
        "BLOOM/BRIGHTPASS: crt-royale's brightpass + 17x17 Gaussian blur "
        "(sigma 1.560 output px) redistributes energy that would clip at "
        "bright phosphor dots. MiSTer has no post-mask blur; the standard "
        "substitute is mask multipliers >1.0 with hardware clamp at 255 "
        "(reads as phosphor bloom). Magnitude: at full white ~74% of the "
        "masked signal routes through the blur at defaults.",
        "DIFFUSION (7.5%): a 2D Gaussian veil (sigma ~1.45 source px H, "
        "~1.82 source lines V, of the PRE-scanline image) is mixed at 7.5% "
        "after masking. Vertically it fills scanline troughs by a flat "
        "+0.075*E term (modeled in ref_vertical); horizontally it is a wide "
        "low-pass a 4-tap FIR cannot fully express (folded into h_kernel as "
        "documented; residual error is the >4-tap tails, ~2.5% of total "
        "energy beyond +/-2 source px).",
        "HALATION: default halation_weight = 0.0 -- OFF, nothing to port.",
        "CONVERGENCE: runtime defaults are 0.0 for all RGB x/y offsets "
        "(static user-settings values 0.1..0.6 are overridden by runtime "
        "params) -- no misconvergence to port.",
        "CURVATURE/BORDERS/AA: geom_mode = 0 (flat), overscan (1,1); the "
        "border darkening (border_size 0.015, darkness 2.0) affects only the "
        "outer ~1.5% frame edge; the 12x-AA path is bypassed when flat. "
        "None of this exists on MiSTer scaler filters; omit.",
        "INTERLACING: 240p is detected progressive (no field bobbing). "
        "Sources with 288<lines<=576 would bob fields per frame -- cannot be "
        "expressed on MiSTer (frame-rate-dependent vertical offsets).",
        "PER-CHANNEL BEAM: sigma/shape vary per RGB channel with that "
        "channel's own linear value. MiSTer's adaptive V-filter interpolates "
        "between two coefficient sets from a shared max-RGB brightness "
        "control, so per-channel width variation on colored edges is lost "
        "(error largest on saturated colors adjacent to white).",
        "BEAM SHAPE vs 4-TAP FIR: the beam covers 3 scanlines with "
        "brightness-dependent sigma (0.02..0.3) AND shape (beta 2..4). At "
        "4.5x scale the dim-level profile (sigma 0.02) is far narrower than "
        "a 4-tap/256-phase FIR can render without aliasing; fitting should "
        "target the L-dependent family via the adaptive (dual-set) V filter "
        "driven by brightness, accepting clipped dim-level sharpness "
        "(the same full-frame trough-stddev metric applies).",
        "AUTODIM: levels_autodim_temp = 0.5 is dim-then-undim around the "
        "8-bit FBOs; net effect is only quantization headroom, no visible "
        "term to port.",
        "FBO CLIPPING: the reconstitute output clips at 1.0 in an 8-bit sRGB "
        "FBO; at defaults (mask on) bright mask dots exceed 1.0 well below "
        "full white -- on MiSTer this maps naturally to the 8-bit clamp "
        "after the mask stage.",
        "8-BIT sRGB FBO QUANTIZATION between passes is not modeled "
        "(sub-1/255 effects).",
        "GAMMA_IMPL: the shader's gamma()/ligamma approximations (rel err "
        "~5e-4) are replaced by exact math.gamma -- negligible.",
        "BLOOM-APPROX RESAMPLING: BLOOM_APPROX is a 320x240 4x4 Gaussian "
        "resize; on non-flat content its texel grid misaligns with 256/320-px "
        "sources; modeled as a continuous Gaussian (error only affects the "
        "diffusion veil, 7.5% weight).",
    ]

# ----------------------------------------------------------------------------
# MASK_TILE: 24x24x3, 0..255, computed by make_mask_tile.py (exact shader
# Lanczos3 resize of the 64x64 slot LUT, incl. 8-bit FBO quantization).
# ----------------------------------------------------------------------------
MASK_TILE = [
[[105,4,3],[15,111,13],[3,4,106],[152,6,1],[22,154,21],[0,7,152],[106,4,3],[16,108,13],[2,5,106],[151,6,0],[22,157,22],[0,7,151],[106,4,3],[16,110,13],[3,5,105],[152,6,0],[22,155,21],[0,7,152],[106,4,4],[16,109,13],[2,5,106],[151,6,0],[22,156,21],[0,7,151]],
[[153,7,0],[22,159,22],[0,7,154],[86,3,4],[13,89,11],[4,4,89],[153,7,0],[21,160,22],[0,8,153],[87,2,4],[13,88,11],[4,5,89],[153,7,0],[22,162,22],[0,7,152],[89,3,4],[13,87,11],[4,4,89],[153,7,0],[22,161,22],[0,7,153],[89,3,4],[13,88,11],[4,4,89]],
[[141,5,1],[21,143,20],[0,6,143],[130,5,2],[18,134,18],[1,6,131],[140,5,1],[20,144,18],[0,6,141],[131,4,3],[18,134,17],[2,6,131],[142,6,1],[20,145,20],[0,6,142],[130,5,2],[19,132,17],[1,5,131],[140,5,1],[20,141,19],[0,6,141],[132,5,2],[19,133,17],[1,6,132]],
[[80,2,4],[12,85,9],[4,4,82],[153,7,0],[23,162,23],[0,8,154],[80,2,5],[12,85,9],[4,4,81],[153,8,0],[23,163,22],[0,7,154],[81,2,4],[13,83,9],[4,3,82],[153,7,0],[22,160,22],[0,7,153],[82,3,5],[12,83,9],[4,4,81],[153,7,0],[23,160,22],[0,7,153]],
[[150,6,1],[20,150,21],[0,7,151],[119,4,3],[17,124,16],[2,6,123],[150,5,0],[22,151,21],[0,7,150],[121,4,2],[18,118,15],[1,5,120],[149,6,1],[22,151,20],[0,7,150],[121,5,2],[16,120,16],[2,5,121],[151,6,0],[21,149,21],[0,7,149],[121,4,2],[17,120,15],[3,6,120]],
[[154,7,0],[23,156,22],[0,7,152],[95,3,4],[14,99,12],[3,5,97],[151,7,0],[22,158,23],[0,7,153],[95,3,3],[14,98,12],[4,5,97],[153,7,0],[22,160,22],[0,7,152],[95,3,3],[13,98,12],[4,4,96],[152,7,0],[21,157,21],[0,7,153],[95,3,4],[14,98,12],[3,5,96]],
[[94,3,4],[14,97,12],[3,4,94],[153,7,0],[22,158,22],[0,7,153],[94,3,4],[14,96,12],[3,4,94],[153,6,0],[23,158,22],[0,7,153],[94,3,4],[14,96,12],[3,4,94],[153,7,0],[22,158,22],[0,6,154],[96,3,4],[14,97,12],[4,4,95],[152,7,0],[22,159,22],[0,7,152]],
[[120,5,3],[18,121,15],[2,5,121],[150,6,1],[21,154,21],[0,7,150],[120,5,3],[17,123,16],[2,5,121],[151,6,0],[21,151,21],[0,7,151],[121,4,2],[17,121,16],[2,6,122],[150,6,1],[21,151,20],[0,7,150],[122,5,3],[17,123,15],[2,5,120],[150,6,0],[21,150,20],[0,6,149]],
[[154,7,0],[23,161,23],[0,7,153],[81,3,4],[12,84,9],[5,3,82],[153,8,0],[21,160,23],[0,7,153],[82,2,4],[12,85,9],[4,4,82],[154,7,0],[23,160,23],[0,7,154],[81,3,4],[12,85,8],[4,4,81],[153,7,0],[23,160,22],[0,7,153],[81,3,5],[12,83,9],[4,3,83]],
[[132,4,2],[19,132,17],[1,6,131],[141,5,1],[20,142,19],[0,6,142],[131,5,2],[19,133,17],[1,6,131],[141,5,1],[20,142,20],[1,6,143],[131,5,2],[18,131,18],[1,5,133],[141,6,2],[20,144,20],[0,6,141],[131,5,2],[19,132,17],[1,6,131],[141,5,2],[21,141,19],[0,6,143]],
[[89,3,4],[13,89,11],[4,5,89],[153,7,0],[22,161,23],[0,8,154],[87,3,3],[12,89,11],[4,4,88],[153,7,0],[22,160,23],[0,7,153],[90,3,4],[13,90,10],[4,5,87],[154,7,0],[23,162,23],[0,7,153],[88,3,4],[13,90,10],[4,4,89],[153,7,0],[22,161,22],[0,8,153]],
[[152,6,0],[22,158,21],[0,7,152],[106,4,3],[15,109,13],[2,5,106],[151,6,0],[22,156,21],[0,6,152],[106,3,3],[15,108,14],[3,5,106],[151,6,0],[22,157,22],[0,7,151],[105,4,3],[16,109,13],[3,5,105],[152,7,0],[22,158,21],[0,7,151],[106,3,3],[16,107,13],[3,5,106]],
[[151,6,0],[22,155,21],[0,7,151],[107,4,3],[16,108,13],[3,5,107],[152,6,1],[21,153,21],[0,7,151],[107,4,3],[16,112,13],[3,5,107],[152,7,1],[21,154,21],[0,7,152],[107,4,3],[16,107,13],[2,4,108],[152,6,1],[22,155,21],[0,7,152],[106,3,4],[16,109,13],[2,5,107]],
[[87,3,4],[13,89,11],[3,4,88],[153,7,0],[23,160,23],[0,8,153],[86,2,4],[12,89,10],[4,5,87],[153,7,0],[22,162,23],[0,8,154],[90,3,4],[13,89,10],[4,5,88],[154,7,0],[22,159,22],[0,7,154],[88,3,4],[13,88,10],[4,4,88],[154,7,0],[22,161,22],[0,7,154]],
[[132,5,2],[19,135,17],[1,6,131],[140,5,1],[20,145,19],[0,6,140],[132,5,2],[19,132,18],[1,5,130],[139,5,1],[20,143,19],[0,6,142],[131,5,2],[18,134,18],[1,6,132],[140,5,1],[20,142,19],[0,6,142],[132,5,3],[19,133,17],[1,6,131],[139,5,1],[20,143,19],[0,6,141]],
[[153,7,0],[23,161,22],[0,7,153],[81,3,5],[12,82,9],[4,4,82],[154,7,0],[22,159,23],[0,7,154],[80,3,4],[12,84,9],[4,4,82],[153,7,0],[23,158,22],[0,7,152],[81,3,4],[12,84,9],[4,3,81],[155,7,0],[23,161,22],[0,7,153],[80,2,5],[12,85,9],[4,4,82]],
[[121,5,3],[17,121,15],[2,5,120],[149,6,0],[21,149,21],[0,7,150],[120,5,3],[17,120,15],[2,5,121],[150,6,1],[22,153,20],[0,6,150],[121,4,3],[18,120,16],[2,5,121],[150,6,0],[21,153,21],[0,7,151],[120,4,2],[17,123,15],[2,5,121],[150,5,1],[20,150,21],[0,7,150]],
[[94,3,3],[13,99,12],[4,5,95],[153,7,0],[22,159,23],[0,8,154],[96,3,4],[14,99,11],[4,4,95],[153,7,0],[22,161,22],[0,7,153],[96,3,4],[14,96,12],[3,4,97],[153,7,0],[22,158,23],[0,7,152],[96,3,3],[14,100,12],[4,4,96],[153,6,0],[22,158,22],[0,7,152]],
[[153,7,0],[22,159,22],[0,7,153],[96,3,4],[14,97,12],[3,4,95],[153,6,0],[23,158,22],[0,7,153],[95,3,4],[14,98,12],[4,4,94],[153,7,0],[22,158,22],[0,6,153],[95,3,4],[14,97,12],[3,4,94],[152,7,0],[23,159,22],[0,7,152],[95,3,3],[14,95,12],[4,4,96]],
[[150,6,1],[21,150,20],[0,7,150],[122,5,3],[18,124,15],[1,6,120],[149,6,1],[21,151,21],[0,6,151],[122,5,2],[17,123,16],[2,5,123],[149,6,0],[21,152,21],[0,6,149],[122,5,2],[18,125,17],[1,5,123],[150,6,0],[22,150,21],[0,6,150],[121,5,2],[17,122,16],[2,5,124]],
[[81,3,5],[12,84,9],[4,4,82],[153,7,0],[22,160,22],[0,8,153],[81,3,5],[12,84,9],[4,4,82],[154,7,0],[23,161,22],[0,7,152],[80,3,4],[12,84,9],[4,4,81],[154,7,0],[22,161,22],[0,7,152],[81,3,5],[12,84,8],[4,4,82],[153,8,0],[23,161,22],[0,7,154]],
[[141,6,1],[20,144,19],[1,6,142],[132,5,2],[18,133,18],[2,6,131],[141,5,1],[19,147,20],[1,6,142],[130,5,2],[18,133,18],[1,6,132],[141,6,1],[20,144,19],[1,6,142],[132,5,2],[19,132,18],[2,6,130],[141,6,1],[19,147,19],[0,6,140],[131,5,2],[19,136,17],[1,6,130]],
[[155,7,0],[22,162,22],[0,7,152],[88,3,3],[13,89,11],[4,4,90],[153,7,0],[21,161,22],[0,7,154],[88,3,4],[12,91,11],[4,5,89],[154,7,0],[22,160,23],[0,7,153],[89,3,4],[13,89,11],[4,4,89],[153,7,0],[22,163,23],[0,7,153],[89,3,4],[13,91,11],[5,5,91]],
[[105,4,3],[16,107,13],[2,4,104],[152,6,0],[22,155,21],[0,7,152],[104,4,4],[15,108,13],[3,4,104],[152,7,0],[22,158,21],[0,7,152],[105,4,4],[15,110,13],[2,4,106],[151,7,0],[22,157,21],[0,6,151],[106,3,3],[15,109,13],[3,5,107],[151,6,1],[22,156,21],[0,7,151]]
]

if __name__ == "__main__":
    print("defaults:", defaults())
    print("transfer(0.5) =", transfer(0.5))
    print("ref_vertical(0.5, 0.5) =", ref_vertical(0.5, 0.5))
    print("beam_weight(0, 0.5) =", beam_weight(0.0, 0.5))
    print("h_kernel(0.5) =", h_kernel(0.5))
    ms = mask_spec()
    print("mask avg transmission =", ms["avg_transmission"])
