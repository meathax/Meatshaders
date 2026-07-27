"""
kurozumi_ref.py -- Reference implementation of CRT-ROYALE-KUROZUMI (the
P22/PVM-look preset over crt-royale, by Kurozumi/hunterk) using the last
historically matched preset/Grade pair from libretro/slang-shaders:

    preset commit 7f34fc7469ecc7d90e03df45d1a10975136eb712
    Grade blob    d5d58f8c88369b062e683d3e52c733535d400a85

The same Grade blob and effective parameter set are present at the explicit
Kurozumi-fix commit 340b533a7c1889c8e18e59bc89dab1867a4116bb.
The preset is 13 passes: the historical misc/shaders/grade.slang prepended to
the 12 crt-royale passes.  One deliberate source-corrected hybrid is retained:
7f34fc's bloom-approx pass calculates its Gaussian result but accidentally
discards it; this model uses the later upstream-fixed Royale bloom behavior at
the active source pin 3b0d6aa1d134a168478cd9c904a866d969f8882b.  It therefore
matches the historical Kuro Grade/preset intent plus the upstream bug fix, not
the byte-exact visual output of every 7f pass.

Models the KUROZUMI-parameter math for a 240p source displayed at 1920x1080
(4.5x vertical scale). Pure Python (math only; numpy not required).
Same interface as royale_ref.py: transfer, beam_weight, ref_vertical,
h_kernel, mask_spec, notes, defaults.

Effective parameter set (from the .slangp `parameters` overrides that resolve
to #pragma parameter uniforms at runtime):

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
  * Pass 0 is Dogway's historical Grade shader.  The intended overrides are
    active here: g_gamma_in=2.4, signal=RGB, P22 measured-average phosphors,
    Rec.709/pure-power output, D65, vignette off, and neutral tone/saturation/
    tint controls.  g_gamma_out is not overridden and therefore remains the
    Grade default 2.5.  The preset's g_gamma_type=1 entry is a dead legacy key:
    blob d5d58f8 does not declare that pragma/uniform, so RetroArch drops it.
    LUT_Size1/LUT1/LUT_Size2/LUT2 are value-less preset entries and retain the
    shader defaults 16/0/64/0, leaving both Grade LUTs disabled.
  * Grade decodes each input component with the 2.4 monitor curve, transforms
    P22 RGB -> XYZ -> Rec.709 RGB with the source CAT02 white-point adjustment,
    applies the reverse 2.304 electron-gun curve, and runs the exactly cancelling
    Rec.709 surround OETF/EOTF pair around neutral controls.  Its RGBA8 output is
    quantized before Royale linearizes it with crt_gamma=2.4.  Black is zero.
  * Halation blur passes are blur5fast (sigma = 0.9845703125), not blur9fast.

Pipeline gamma: Grade outputs encoded RGBA8 (see grade_rgb); royale pass 1
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
channel's vertical convergence offset; default None models a neutral centered
channel (recommended shared-filter fitting target).  grade_rgb() and
ref_vertical_rgb() expose the nonseparable historical P22 color transform.
"""

import functools
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
# Historical grade.slang color transform at Kurozumi's matched settings
# ----------------------------------------------------------------------------

HISTORICAL_COMMIT = "7f34fc7469ecc7d90e03df45d1a10975136eb712"
EXPLICIT_FIX_COMMIT = "340b533a7c1889c8e18e59bc89dab1867a4116bb"
GRADE_BLOB = "d5d58f8c88369b062e683d3e52c733535d400a85"
ROYALE_BLOOM_FIX_COMMIT = "3b0d6aa1d134a168478cd9c904a866d969f8882b"
GRADE_GAMMA_IN = 2.4
GRADE_GAMMA_OUT = 2.5             # pragma default; Kurozumi does not override it
GRADE_GUN_GAMMA = GRADE_GAMMA_IN ** 2.0 / GRADE_GAMMA_OUT  # 2.304
GRADE_MONCURVE_OFFSET = 0.099
GRADE_TEMPERATURE = 6504.0
GRADE_SURROUND = 1.019264

_CH_IDX = {'r': 0, 'g': 1, 'b': 2}

# GLSL matrices written in mathematical row-major form.  The source constructors
# are column-major; these rows are the actual result of ``matrix * vector``.
_P22_TO_XYZ = (
    (0.4665636420249939, 0.3039233088493347, 0.1799621731042862),
    (0.25661000609397890, 0.66820019483566280, 0.07518967241048813),
    (0.005832045804709196, 0.105618737637996670, 0.977465748786926300),
)
_XYZ_TO_709 = (
    (3.24081254005432130, -1.53730857372283940, -0.49858659505844116),
    (-0.969243049621582000, 1.875966310501098600, 0.041555050760507584),
    (0.055638398975133896, -0.204007431864738460, 1.057129383087158200),
)
_RGB709_TO_XYZ = (
    (0.41241079568862915, 0.35758456587791443, 0.18045382201671600),
    (0.21264933049678802, 0.71516913175582890, 0.07218152284622192),
    (0.019331756979227066, 0.119194857776165010, 0.950390160083770800),
)
# Actual row-major GLSL matrix for ``vec3 * CAT02`` in wp_adjust().
_CAT02 = (
    (0.7328, -0.70360, 0.003),
    (0.4296, 1.6975, -0.0136),
    (-0.1624, 0.0061, 0.9834),
)


def _clamp01(v):
    return max(0.0, min(1.0, v))


def _mat_vec(matrix, vector):
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3))
                 for row in range(3))


def _vec_mat(vector, matrix):
    return tuple(sum(vector[row] * matrix[row][col] for row in range(3))
                 for col in range(3))


def _moncurve_f(color, gamma, offs):
    """Historical Grade forward monitor curve, verbatim algebra."""
    color = _clamp01(color)
    fs = ((gamma - 1.0) / offs) * (
        offs * gamma / ((gamma - 1.0) * (1.0 + offs))) ** gamma
    xb = offs / (gamma - 1.0)
    return ((color + offs) / (1.0 + offs)) ** gamma if color > xb else color * fs


def _moncurve_r(color, gamma, offs):
    """Historical Grade reverse monitor curve, verbatim algebra."""
    color = _clamp01(color)
    yb = (offs * gamma / ((gamma - 1.0) * (1.0 + offs))) ** gamma
    rs = ((gamma - 1.0) / offs) ** (gamma - 1.0) * (
        (1.0 + offs) / gamma) ** gamma
    return ((1.0 + offs) * color ** (1.0 / gamma) - offs
            if color > yb else color * rs)


def _grade_rolled_gain(color, gain=0.0):
    """Historical rolled_gain; even neutral gain=0 is microscopically nonidentity."""
    gx = abs(gain) + 0.001
    if gain > 0.0:
        anch = 0.5 / (gx / 2.0)
        return color * ((color - anch) / (1.0 - anch))
    anch = 0.5 / gx
    return color * ((1.0 - anch) / (color - anch)) * (1.0 - gain)


def _wp_adjust_d65(color):
    """Historical CAT02 white-point calculation at the preset's 6504 K."""
    temperature = GRADE_TEMPERATURE
    temp3 = 1000.0 / temperature
    temp6 = 1000000.0 / (temperature ** 2.0)
    temp9 = 1000000000.0 / (temperature ** 3.0)
    x = (0.244063 + 0.09911 * temp3 + 2.9678 * temp6 - 4.6070 * temp9
         if temperature <= 7000.0 else
         0.237040 + 0.24748 * temp3 + 1.9018 * temp6 - 2.0064 * temp9)
    y = -3.0 * x * x + 2.870 * x - 0.275
    z = 1.0 - x - y
    target = _vec_mat((x / y, 1.0, z / y), _CAT02)
    reference = _vec_mat((0.95045, 1.0, 1.088917), _CAT02)
    scale = tuple(target[i] / reference[i] for i in range(3))
    return tuple(color[i] * scale[i] for i in range(3))


def _unorm8(color):
    """Grade pass 0 writes a non-sRGB, non-float RGBA8 framebuffer."""
    return tuple(math.floor(_clamp01(value) * 255.0 + 0.5) / 255.0
                 for value in color)


def grade_rgb(rgb, quantize=True):
    """Exact matched-Grade encoded RGB output for one encoded RGB input.

    This transliterates the active source path: monitor-curve decode, measured
    P22 RGB->XYZ, CAT02 D65 adjustment, XYZ->Rec.709, reverse 2.304 gun curve,
    neutral surround OETF/control/EOTF path, and pass-0 RGBA8 quantization.
    The P22 transform is nonseparable; this function is the RGB source oracle.
    """
    if len(rgb) != 3:
        raise ValueError("rgb must have exactly three components")
    col = tuple(_moncurve_f(_clamp01(float(value)), GRADE_GAMMA_IN,
                            GRADE_MONCURVE_OFFSET) for value in rgb)
    xyz = _mat_vec(_P22_TO_XYZ, col)
    xyz = _wp_adjust_d65(xyz)
    adj = tuple(_clamp01(value) for value in _mat_vec(_XYZ_TO_709, xyz))
    adj = tuple(_moncurve_r(value, GRADE_GUN_GAMMA,
                            GRADE_MONCURVE_OFFSET) for value in adj)

    # SPC=-1 OETF, neutral contrast/controls, then matching EOTF.  These power
    # branches algebraically cancel, but retaining them preserves clamp/order.
    oetf = tuple(_clamp01((value ** (1.0 / GRADE_SURROUND)) ** 2.4)
                 for value in adj)
    xyz2 = _mat_vec(_RGB709_TO_XYZ, oetf)
    screen = tuple(max(value, 0.0) for value in _mat_vec(_XYZ_TO_709, xyz2))
    screen = tuple(_clamp01(_grade_rolled_gain(value, 0.0)) for value in screen)
    trc = tuple(_clamp01((value ** GRADE_SURROUND) ** (1.0 / 2.4))
                for value in screen)
    return _unorm8(trc) if quantize else trc


def grade_transfer(x, channel=None):
    """Ideal historical Grade neutral-axis transfer; practical black is 0."""
    rgb = grade_rgb((_clamp01(x),) * 3)
    return rgb[_CH_IDX.get(channel, 1)]


def lin_input(L, channel=None):
    """Linear value entering Royale after the quantized historical Grade pass."""
    return grade_transfer(L, channel) ** CRT_GAMMA


@functools.lru_cache(maxsize=4096)
def _lin_input_rgb_cached(rgb):
    return tuple(value ** CRT_GAMMA for value in grade_rgb(rgb))


def lin_input_rgb(rgb):
    """Per-channel linear input to Royale for an arbitrary encoded RGB source."""
    return _lin_input_rgb_cached(tuple(float(value) for value in rgb))


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
# active 4x4 branch combines that resize sigma with the runtime Kurozumi
# beam_max_sigma override (0.16).  The static 0.3 path is not selected:
_MAX_VP_X = 1080.0 * 1024.0 * (4.0 / 3.0)
_EST_VP_X = 1080.0 * (432.0 / 329.0)
_N_TRIADS = max(256.0, _EST_VP_X / MASK_TRIAD_SIZE_DESIRED)        # 1418.18
_BA_SIGMA_RESIZE = (_min_sigma_to_blur_triad(_MAX_VP_X / _N_TRIADS)
                    * BLOOM_APPROX_W / _MAX_VP_X)                  # ~0.12129
BLOOM_APPROX_SIGMA = math.hypot(_BA_SIGMA_RESIZE, BEAM_MAX_SIGMA)
# ~0.20075

# Total diffusion/halation blur sigma in BLOOM_APPROX pixels:
_DIFF_SIGMA_BA = math.hypot(BLOOM_APPROX_SIGMA, BLUR5_STD_DEV)     # ~1.0048
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
    the scalar neutral green-component form, convergence offsets NOT applied."""
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

    Chain: historical Grade neutral curve -> linearize (2.4) -> 3-scanline
    generalized-Gaussian beam sum (sigma 0.02..0.16) -> autodim 0.5 + FBO
    clamp -> halation lerp (0.37% toward flat E*0.98600) -> [mask=1] ->
    brightpass (underest 0.68, cw 0.3902) / 17-tap bloom blur (sigma 1.0234)
    -> reconstitute (dimpass + blurred brightpass) * undim * CONTRAST 0.67
    -> diffusion lerp (0.11% toward 0.67*E*0.98600) -> FBO clamp
    -> encode pow(1/2.4).

    channel='r'/'g'/'b' applies that channel's Grade transform and vertical
    convergence offset (r +0.05, g -0.05, b +0.05 scanlines); None models a
    neutral centered channel using the green component."""
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


def _electron_intensity_rgb(f, energies, convergence=True):
    """Vector form of the scanline + source-accurate desaturating halation."""
    conv = CONVERGENCE_Y if convergence else (0.0, 0.0, 0.0)
    scan_dim = tuple(_clamp01(
        LEVELS_AUTODIM_TEMP * _scanline_sum(f, energies[i], conv[i]))
        for i in range(3))
    halation_scalar = (LEVELS_AUTODIM_TEMP * HALATION_FLAT_GAIN
                       * sum(energies) / 3.0)
    return tuple(_clamp01(scan_dim[i] +
                          (halation_scalar - scan_dim[i]) * HALATION_WEIGHT)
                 for i in range(3))


def _masked_brightpass_rgb(f, energies, raw_mask, mask_amplify,
                           convergence=True):
    electron = _electron_intensity_rgb(f, energies, convergence)
    intensity_dim = tuple(_clamp01(electron[i] * raw_mask[i]) for i in range(3))
    bright = tuple(_brightpass(intensity_dim[i], energies[i], mask_amplify)
                   for i in range(3))
    return intensity_dim, bright


def _blurred_brightpass_rgb(f, energies, raw_mask, mask_amplify,
                            convergence=True):
    """Local-cell 17-tap bloom projection for a uniform source field.

    Vertical neighbors are exact for the supplied mask cell.  The real shader's
    horizontal bloom also integrates adjacent two-pixel grille cells; the API's
    point mask multiplier cannot express that nonlocal dependency, so this is
    the closest deterministic per-cell projection available to the MiSTer fit.
    """
    center = _masked_brightpass_rgb(
        f, energies, raw_mask, mask_amplify, convergence)[1]
    total = [_BLOOM_W[0] * center[i] for i in range(3)]
    for k in range(1, 9):
        for sign in (-1.0, 1.0):
            phase = (f + sign * k * SCAN_PER_OUT_PX) % 1.0
            sample = _masked_brightpass_rgb(
                phase, energies, raw_mask, mask_amplify, convergence)[1]
            for i in range(3):
                total[i] += _BLOOM_W[k] * sample[i]
    h_gain = ((1.0 + 2.0 * sum(
        math.exp(-(j * j) * 0.5 / (BLOOM_SIGMA ** 2)) for j in range(1, 9)))
        * _fast_gaussian_weight_sum_inv(BLOOM_SIGMA))
    return tuple(value * h_gain for value in total)


@functools.lru_cache(maxsize=262144)
def _ref_vertical_rgb_cached(f, rgb, mask_mult):
    """Cached tuple-only implementation for :func:`ref_vertical_rgb`."""
    energies = lin_input_rgb(rgb)
    if mask_mult is None:
        raw_mask = (1.0, 1.0, 1.0)
        mask_amplify = MASK_AMPLIFY_DISABLED
    else:
        raw_mask = tuple(max(0.0, value) / MASK_AMPLIFY
                         for value in mask_mult)
        mask_amplify = MASK_AMPLIFY
    intensity_dim, brightpass = _masked_brightpass_rgb(
        f, energies, raw_mask, mask_amplify, convergence=True)
    blurred = _blurred_brightpass_rgb(
        f, energies, raw_mask, mask_amplify, convergence=True)
    result = []
    for i in range(3):
        phosphor = ((intensity_dim[i] - brightpass[i] + blurred[i])
                    * mask_amplify * (1.0 / LEVELS_AUTODIM_TEMP)
                    * LEVELS_CONTRAST)
        diffusion = LEVELS_CONTRAST * energies[i] * HALATION_FLAT_GAIN
        final = _clamp01(phosphor * (1.0 - DIFFUSION_WEIGHT)
                         + diffusion * DIFFUSION_WEIGHT)
        result.append(final ** (1.0 / LCD_GAMMA))
    return tuple(result)


def ref_vertical_rgb(f, rgb, mask_mult=None):
    """Uniform-field RGB oracle through historical Grade and CRT-Royale.

    ``rgb`` is encoded source RGB.  With ``mask_mult=None`` the phosphor mask is
    disabled and mask amplification is one, matching the filter fit.  A supplied
    multiplier is the net linear tile returned by mask_spec(); it is split back
    into Royale's raw mask sample and later reconstitution amplification so the
    source clamp/order is retained.
    """
    if len(rgb) != 3:
        raise ValueError("rgb must have exactly three components")
    f = abs(float(f)) % 1.0
    rgb = tuple(_clamp01(float(value)) for value in rgb)
    if mask_mult is None:
        mask_mult = None
    else:
        if len(mask_mult) != 3:
            raise ValueError("mask_mult must have exactly three components")
        mask_mult = tuple(float(value) for value in mask_mult)
    return _ref_vertical_rgb_cached(f, rgb, mask_mult)


@functools.lru_cache(maxsize=262144)
def _ref_masked_cached(f, x, red, green, blue, index):
    pixel = (red, green, blue)
    net = tuple(value / 255.0 * MASK_AMPLIFY for value in pixel)
    return ref_vertical_rgb(f, (_clamp01(x),) * 3, net)[index]


def ref_masked(f, x, px, py, channel):
    """Exact local-order neutral mask hook used by the periodic fitter."""
    index = _CH_IDX[channel]
    pixel = MASK_TILE[int(py) % len(MASK_TILE)][int(px) % len(MASK_TILE[0])]
    return _ref_masked_cached(float(f), float(x), *pixel, index)


def transfer(x, channel=None):
    """End-to-end encoded->encoded transfer for a uniform field sampled at
    f=0 (scanline grid position), mask disabled.  Includes the grade pass
    (historical P22 Grade), CRT gamma 2.4, beam peak gain, halation, contrast
    0.67, bloom redistribution, diffusion, FBO clipping, and LCD gamma 1/2.4.
    NOTE: this is the SCANLINE-PEAK transfer; the thin beam concentrates the
    line energy, so the peak saturates at 1.0 from x ~0.738 (reconstitute
    clamp; the scanline-FBO peak clamp starts at x ~0.878).  The 0.67
    contrast ceiling applies to the flat-field SPATIAL AVERAGE (0.67 linear
    = 0.846 encoded), not the peak.  Encoded black is exactly zero."""
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
      * 0.11% diffusion: Gaussian sigma ~0.8039 source px (BLOOM_APPROX 4x4
        resize sigma ~0.20075 (+) blur5 sigma 0.9846, at 320->256 scale),
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
    phase ~0.264, green ~1.019 flat; average 1.064.  Encoded (^(1/2.4)):
    bright ~1.309, dim ~0.575, green ~1.008."""
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
            "(linear_mult)^(1/2.4).  Representative encoded triple per pixel: "
            "bright phase (~1.309, ~1.008, ~0.575), dim phase mirrored in R/B.",
    }


def defaults():
    return {
        "commit": HISTORICAL_COMMIT,
        "explicit_fix_commit": EXPLICIT_FIX_COMMIT,
        "grade_blob": GRADE_BLOB,
        "royale_bloom_fix_commit": ROYALE_BLOOM_FIX_COMMIT,
        "source_hybrid": "historical Kuro preset+Grade with later upstream-fixed "
                         "Royale bloom-approx output",
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
            "version_note": "historically matched Grade blob; active original "
                            "Kurozumi parameter family",
            "g_gamma_in": GRADE_GAMMA_IN,
            "g_gamma_out_default": GRADE_GAMMA_OUT,
            "gun_curve_gamma": GRADE_GUN_GAMMA,
            "g_signal_type": 0.0,
            "g_crtgamut": "P22 measured average (1.0)",
            "g_space_out": "Rec.709 / pure power (-1.0)",
            "wp_temperature": GRADE_TEMPERATURE,
            "g_vignette": 0.0,
            "pass0_storage": "RGBA8 UNORM",
            "black": 0.0,
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
        "GRADE PASS: prepended historical misc/shaders/grade.slang blob "
        "d5d58f8c8836.  It decodes the embedded 2.4 monitor curve, applies the "
        "measured-average P22 RGB->XYZ->Rec.709 matrix with CAT02 D65 mapping, "
        "then the reverse 2.304 gun curve and neutral surround path.  Pass 0 "
        "is RGBA8 UNORM, modeled by grade_rgb().  The source's all-zero path "
        "contains undefined 0/0 and atan(0,0) intermediates; zero is the "
        "intended and practical finite GPU result used by this oracle.  The "
        "P22 3x3 transform is NOT representable by MiSTer's independent 1-D "
        "RGB LUTs; ref_vertical_rgb() exposes the true vector target so the "
        "best legal diagonal compromise can be measured rather than hidden.",
        "DELIBERATE UPSTREAM-FIX HYBRID: 7f34fc's bloom-approx pass computes "
        "the Gaussian color but discards it.  This reference retains the later "
        "upstream-corrected Royale bloom output from 3b0d6aa, reproducing the "
        "intended effect rather than that historical implementation bug.",
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
        "(see mask_spec).  Required encoded per-pixel triple ~ (1.309, "
        "1.008, 0.575) / mirrored.  MiSTer v2 tokens give one boosted "
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
        "is only 1.906 (encoded ~1.309 < MiSTer's 1.9375 ceiling) because the "
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
        "visibility.  ref_vertical_rgb() retains halation's source-accurate "
        "mean(RGB) desaturation.",
        "CONVERGENCE: x = (-0.05, 0, 0) texels, y = (+0.05, -0.05, +0.05) "
        "scanlines.  MiSTer's H FIR and V FIR are shared across channels -> "
        "per-channel offsets are NOT representable.  Magnitude: 0.05 src px "
        "= 0.375 output px H (at 7.5x for 256-wide), 0.05 lines = 0.225 "
        "output px V -- subpixel; visible only as a faint red fringe. Omit.",
        "GAMMA: crt 2.4 / lcd 2.4 cancel on the DC transfer EXCEPT for clamps "
        "and the historical Grade curve.  Fit the neutral curve without "
        "inventing a black lift, then test primaries/secondaries through the "
        "shared max-RGB adaptive control and retain only a measured RGB/neutral "
        "Pareto improvement.",
        "INTERLACING: 240p progressive; 288<lines<=576 would bob (not "
        "expressible on MiSTer).  interlace_1080i off.",
        "PER-CHANNEL BEAM WIDTH: as in royale, sigma/beta are per-channel "
        "from each channel's linear value; MiSTer's single max-RGB adaptive "
        "control loses this on saturated colors.",
        "NOT MODELED: 8-bit sRGB FBO quantization between Royale passes; the "
        "RGBA8 UNORM Grade FBO IS modeled.  The oracle uses Python double "
        "precision: it agrees on the complete neutral code axis and fitting "
        "cube, while rare float32 boundary cases can differ by one UNORM8 "
        "code.  Uses math.gamma instead of the shader's "
        "gamma approximation (rel err ~5e-4); BLOOM_APPROX texel-grid "
        "alignment idealized (affects only the 0.48% veil).  Horizontal bloom "
        "between neighboring 2-pixel mask cells is approximated by the local "
        "cell in ref_vertical_rgb(); neutral mask-period validation still "
        "covers every real 16x16 source cell.",
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
