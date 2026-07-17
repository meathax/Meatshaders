"""
easymode_ref.py -- Reference implementation of CRT-EASYMODE at DEFAULT
parameters, from libretro/glsl-shaders pinned to commit
2b2c5ee3fd8e1a3884e20ed424fd9bfbc51cbb3d (crt/shaders/crt-easymode.glsl).

Single-pass shader. Models the DEFAULT-parameter math for a 240p source
displayed at 1920x1080 (4.5x vertical scale). Pure Python (math only).

Defaults (verbatim from the #else block of the PARAMETER_UNIFORM ifdef,
lines 152-172 of the pinned source):

    SHARPNESS_H 0.5          SHARPNESS_V 1.0
    MASK_STRENGTH 0.3        MASK_DOT_WIDTH 1.0     MASK_DOT_HEIGHT 1.0
    MASK_STAGGER 0.0         MASK_SIZE 1.0
    SCANLINE_STRENGTH 1.0
    SCANLINE_BEAM_WIDTH_MIN 1.5   SCANLINE_BEAM_WIDTH_MAX 1.5
    SCANLINE_BRIGHT_MIN 0.35      SCANLINE_BRIGHT_MAX 0.65
    SCANLINE_CUTOFF 400.0
    GAMMA_INPUT 2.0          GAMMA_OUTPUT 1.8
    BRIGHT_BOOST 1.2         DILATION 1.0
    ENABLE_LANCZOS 1         (compile-time #define, line 172)

Exact pipeline (main(), lines 212-268):

  1. pix_co = vTexCoord * SourceSize.xy - 0.5 ; tex_co = (floor(pix_co)+0.5)/size
     dist = fract(pix_co).  pix_co.y == n  <=>  centre of source row n.
  2. TEX2D(c) = dilate(texture(c));  dilate(col) = col * mix(1, col, DILATION).
     DILATION=1 => x = col => returns col*col.  Every fetch is SQUARED at
     fetch time: the filtering below runs in gamma-2.0 LINEAR light.
  3. curve_x = curve_distance(dist.x, SHARPNESS_H*SHARPNESS_H = 0.25)
     (the parameter is SQUARED at line 223 -> effective H sharpness 0.25).
     coeffs = Lanczos2 at t = (1+cx, cx, 1-cx, 2-cx) i.e. distances to texels
     n-1, n, n+1, n+2; c = PI*t; kernel = 2 sin(c) sin(c/2) / c^2, then
     normalized by dot(coeffs, 1).  filter_lanczos() additionally CLAMPS the
     result to [min,max] of the two INNER taps (color_matrix[1], [2]) -- an
     anti-ringing clamp (identity on monotone/flat neighbourhoods).
     Two rows are filtered: col (row n) and col2 (row n+1).
  4. col = mix(col, col2, curve_distance(dist.y, SHARPNESS_V=1.0))
     -> a 2-tap vertical resample warped by the half-circle s-curve.
  5. col = pow(col, GAMMA_INPUT/(DILATION+1)) = pow(col, 2.0/2.0) = IDENTITY.
  6. luma = Rec709.col ; bright = (max(col.rgb) + luma) * 0.5
     scan_bright = clamp(bright, 0.35, 0.65)
     scan_beam   = clamp(bright*1.5, 1.5, 1.5) == 1.5  (CONSTANT: min==max, so
                   the beam WIDTH does not adapt at defaults at all)
     scan_weight = 1 - pow(cos(vTexCoord.y*2*PI*SourceSize.y)*0.5+0.5, 1.5)*1.0
     At a texel centre vTexCoord.y*SourceSize.y = n+0.5, so cos(2*PI*(n+0.5))
     = -1 and scan_weight = 1: THE BEAM CENTRE IS THE TEXEL CENTRE, and the
     trough (scan_weight=0) sits exactly between rows.
  7. if (InputSize.y >= SCANLINE_CUTOFF=400) scan_weight = 1.0  (240p: no).
  8. col2 = col; col *= scan_weight; col = mix(col, col2, scan_bright)
     => col *= scan_bright + (1-scan_bright)*scan_weight.  scan_bright is a
     TROUGH FLOOR in gamma-2 linear space; the peak is always exactly 1.0.
  9. col *= mask_weight            (LINEAR light, 1.0/0.7 aperture grille)
 10. col = pow(col, 1/GAMMA_OUTPUT=1/1.8)
 11. FragColor = col * BRIGHT_BOOST(1.2)   -- AFTER output gamma, so it CLIPS
     against the 1.0 framebuffer ceiling for encoded col > 1/1.2 = 0.8333.

Interface (contract for the automated fitting stage):
    transfer(x)         encoded->encoded flat field at beam centre, mask off
    beam_weight(d, L)   linear-light weight of a source line at distance d
    ref_vertical(f, x)  encoded output at vertical fraction f, mask off
    h_kernel(frac)      effective horizontal kernel
    mask_spec()         default mask as rendered at 1080p
    notes()             features not representable on MiSTer
    defaults()          dict of all default parameter values
"""

import math

# ----------------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------------

SHARPNESS_H = 0.5
SHARPNESS_H_EFF = SHARPNESS_H * SHARPNESS_H      # 0.25 -- squared at line 223
SHARPNESS_V = 1.0
MASK_STRENGTH = 0.3
MASK_DOT_WIDTH = 1.0
MASK_DOT_HEIGHT = 1.0
MASK_STAGGER = 0.0
MASK_SIZE = 1.0
SCANLINE_STRENGTH = 1.0
SCANLINE_BEAM_WIDTH_MIN = 1.5
SCANLINE_BEAM_WIDTH_MAX = 1.5
SCANLINE_BRIGHT_MIN = 0.35
SCANLINE_BRIGHT_MAX = 0.65
SCANLINE_CUTOFF = 400.0
GAMMA_INPUT = 2.0
GAMMA_OUTPUT = 1.8
BRIGHT_BOOST = 1.2
DILATION = 1.0
ENABLE_LANCZOS = 1

# Derived gamma exponents
FETCH_GAMMA = DILATION + 1.0                     # 2.0 -- dilate() squares fetches
POST_FILTER_POW = GAMMA_INPUT / (DILATION + 1.0)  # 1.0 -- identity
OUT_POW = 1.0 / GAMMA_OUTPUT                      # 0.5555...

# Geometry of the modeled configuration: 240p source -> 1080p output
SOURCE_LINES = 240.0
OUTPUT_LINES = 1080.0
VSCALE = OUTPUT_LINES / SOURCE_LINES              # 4.5
INPUT_SIZE_Y = SOURCE_LINES                       # < SCANLINE_CUTOFF -> scanlines ON
SCANLINES_ENABLED = INPUT_SIZE_Y < SCANLINE_CUTOFF

LUMA_W = (0.2126, 0.7152, 0.0722)                 # sums to exactly 1.0

PI = math.pi

# BRIGHT_BOOST clips the encoded output at 1/1.2; in input-x terms:
CLIP_ENCODED = 1.0 / BRIGHT_BOOST                              # 0.833333
CLIP_X = CLIP_ENCODED ** (GAMMA_OUTPUT / GAMMA_INPUT)          # 0.848656...


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _clamp01(v):
    return max(0.0, min(1.0, v))


# ----------------------------------------------------------------------------
# curve_distance (lines 181-194) -- exact
# ----------------------------------------------------------------------------

def curve_distance(x, sharp):
    """float curve_distance(float x, float sharp):
           x_step = step(0.5, x)
           curve  = 0.5 - sqrt(0.25 - (x-x_step)^2) * sign(0.5 - x)
           return mix(x, curve, sharp)
    Half-circle s-curve; sharp=1 -> pure curve, sharp=0 -> linear."""
    x_step = 1.0 if x >= 0.5 else 0.0
    s = 0.5 - x
    sign = 0.0 if s == 0.0 else (1.0 if s > 0.0 else -1.0)
    inner = 0.25 - (x - x_step) * (x - x_step)
    curve = 0.5 - math.sqrt(max(inner, 0.0)) * sign
    return x * (1.0 - sharp) + curve * sharp


# ----------------------------------------------------------------------------
# Vertical resampling kernel  (line 240: mix(col, col2, curve_distance(dist.y, 1)))
# ----------------------------------------------------------------------------
#
# At vertical fraction dist = f in [0,1), the output mixes source row n
# (distance f) with row n+1 (distance 1-f), with mix factor t = cd(f, 1.0):
#       w(row n)   = 1 - t(f)
#       w(row n+1) =     t(f)
# The s-curve satisfies t(1-x) == 1 - t(x) exactly (both branches reduce to
# 0.5 +/- sqrt(0.25 - x^2)), so BOTH taps' weights are the same function of
# their OWN distance:
#       W(d) = 1 - t(d)  = 0.5 + sqrt(0.25 - d^2)          for 0 <= d <= 0.5
#                        = 0.5 - sqrt(0.25 - (1-d)^2)      for 0.5 <  d <= 1
#                        = 0                               for d > 1
# W(0)=1, W(0.5)=0.5, W(1)=0, and W(d) + W(1-d) = 1 for all d: the kernel is
# always exactly partition-of-unity, i.e. it never changes flat-field level.
# This is the pure GEOMETRIC resample; the brightness-dependent scanline
# modulation is a separate multiplicative term carried by ref_vertical().


def beam_weight(d, L):
    """LINEAR-light contribution weight of one source line at vertical distance
    d source-lines (|d| <= 2 covered) whose encoded brightness is L in [0,1].

    Easymode's vertical stage is a 2-TAP resample, not a beam integral: the
    only lines that contribute are the two straddling the sample, so this is
    identically 0 for |d| >= 1.  The weight is the half-circle-warped mix
    factor W(|d|) described above.

    L IS IGNORED -- and that is exact, not an approximation: at defaults
    scan_beam = clamp(bright*1.5, 1.5, 1.5) is CONSTANT 1.5, so no part of
    Easymode's vertical *shape* depends on brightness. The only brightness
    adaptation at defaults is scan_bright, a trough FLOOR that scales the
    whole 2-tap result and is therefore carried by ref_vertical()'s row sums
    (see fitting._tap_targets: shape from beam_weight, sum from ref_vertical).
    """
    d = abs(d)
    if d >= 1.0:
        return 0.0
    if d <= 0.5:
        return 0.5 + math.sqrt(max(0.25 - d * d, 0.0))
    e = 1.0 - d
    return 0.5 - math.sqrt(max(0.25 - e * e, 0.0))


# ----------------------------------------------------------------------------
# Scanline modulation (lines 243-263)
# ----------------------------------------------------------------------------

def scan_weight_at(f):
    """scan_weight = 1 - pow(cos(vTexCoord.y*2*PI*SourceSize.y)*0.5+0.5, 1.5).

    With f = vertical fraction from the beam CENTRE (= texel centre), we have
    vTexCoord.y*SourceSize.y = n + 0.5 + f, so
        cos(2*PI*(n+0.5+f)) = -cos(2*PI*f)
    and the cosine term becomes 0.5*(1 - cos(2*PI*f)):  0 at f=0 (peak,
    scan_weight=1), 1 at f=0.5 (trough, scan_weight=0)."""
    if not SCANLINES_ENABLED:
        return 1.0
    c = 0.5 * (1.0 - math.cos(2.0 * PI * f))
    return 1.0 - (c ** SCANLINE_BEAM_WIDTH_MIN) * SCANLINE_STRENGTH


def effective_scan(f, lin):
    """Net multiplier applied to the linear-light colour at fraction f, for a
    pixel whose pre-scanline linear value is `lin` (gray).

        col2 = col; col *= scan_weight; col = mix(col, col2, scan_bright)
      => col * (scan_bright + (1 - scan_bright)*scan_weight)

    scan_bright = clamp(bright, 0.35, 0.65) is a FLOOR: at f=0 the multiplier
    is exactly 1.0 for every brightness (no peak adaptation)."""
    bright = _bright_of_gray(lin)
    sb = _clamp(bright, SCANLINE_BRIGHT_MIN, SCANLINE_BRIGHT_MAX)
    return sb + (1.0 - sb) * scan_weight_at(f)


def _bright_of_gray(lin):
    """bright = (max(col.r,col.g,col.b) + dot(Rec709, col)) * 0.5.
    For a gray pixel max == luma == lin (Rec709 weights sum to 1.0)."""
    return lin


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------

def ref_vertical(f, x):
    """Encoded (0..1) output at vertical fraction f within one scanline period
    (f=0 = beam/texel centre, f=0.5 = midway between source rows) for a
    uniform encoded input x, MASK DISABLED, all other defaults.
    240p source at 1080p output.

    Uniform field => dilate gives x^2 at every texel; the H Lanczos (unity DC,
    anti-ring clamp inert) and the V mix (partition of unity) both preserve
    it, and pow(.,1.0) is identity. So the pre-scanline linear value is
    exactly x^2 for every f, and only the scan modulation varies with f."""
    return min(ref_vertical_unclipped(f, x), 1.0)   # framebuffer clip


def ref_vertical_unclipped(f, x):
    """ref_vertical WITHOUT the final framebuffer clip.

    The shader clips *after* the mask multiply (clamp(B*m)), so a port whose
    gain lives in the mask stage — which on MiSTer also saturates after the
    V stage — must fit against the pre-clip beam B(f,x), not the clipped
    value. Exposing B lets the gamma LUT carry B/G (strictly increasing, so
    the adaptive control stays injective) while the mask carries G.
    """
    x = _clamp01(x)
    f = abs(f) % 1.0
    lin = x ** FETCH_GAMMA                       # dilate(): col*col
    lin *= effective_scan(f, lin)                # scanline (linear light)
    enc = lin ** OUT_POW                         # pow(col, 1/1.8)
    return enc * BRIGHT_BOOST                    # *1.2, pre-clip


def transfer_unclipped(x):
    """Beam-centre transfer before the framebuffer clip: 1.2 * x^(10/9).
    Max value 1.2 at x=1 (BRIGHT_BOOST), vs transfer()'s clipped 1.0."""
    return ref_vertical_unclipped(0.0, x)


def transfer(x):
    """End-to-end encoded->encoded transfer for a uniform field sampled at the
    beam CENTRE (f=0), mask off.  At f=0 scan_weight=1 so effective_scan=1
    identically, giving the closed form

        transfer(x) = min(1.2 * x^(2/1.8), 1.0) = min(1.2 * x^(10/9), 1.0)

    which CLIPS for x >= (1/1.2)^(1.8/2) = 0.848656 (BRIGHT_BOOST is applied
    after the output gamma, line 267)."""
    return ref_vertical(0.0, x)


def h_kernel(frac):
    """Effective normalized horizontal kernel at fractional source-pixel offset
    frac (0..1) between texel centres, as (tap_offset, weight) pairs.

    tap_offset is the INTEGER source-texel index relative to the floor texel:
    -1, 0, +1, +2  (matching the color_matrix column order
    TEX2D(co-dx), TEX2D(co), TEX2D(co+dx), TEX2D(co+2*dx), and MiSTer's
    fixed [-1,0,+1,+2] tap window, F4).  This is exactly what
    fitting.fit_h consumes ("idx = int(round(off)) + 1; offset -1 -> column
    0") -- returning integers makes that mapping exact at every phase.

    Math (ENABLE_LANCZOS=1, lines 223-231):
        cx     = curve_distance(frac, SHARPNESS_H^2 = 0.25)
        t      = (1+cx, cx, 1-cx, 2-cx)        # |distance| to each texel
        c      = FIX(PI*t)                     # FIX(c)=max(|c|,1e-5)
        coeffs = 2*sin(c)*sin(c/2) / c^2       # Lanczos2 = sinc(t)*sinc(t/2)
        coeffs /= dot(coeffs, 1)               # normalized -> sums to 1
    Weights are applied to SQUARED (linear) samples.

    NOT captured here (both are nonlinear, so they cannot live in a linear
    kernel): filter_lanczos()'s anti-ringing clamp to the min/max of the two
    inner taps, and the fact that the kernel operates on x^2 rather than x."""
    frac = frac % 1.0
    cx = curve_distance(frac, SHARPNESS_H_EFF)
    ts = (1.0 + cx, cx, 1.0 - cx, 2.0 - cx)
    offs = (-1, 0, 1, 2)
    co = []
    for t in ts:
        c = max(abs(PI * t), 1e-5)
        co.append(2.0 * math.sin(c) * math.sin(c * 0.5) / (c * c))
    s = sum(co)
    return [(float(o), w / s) for o, w in zip(offs, co)]


def mask_spec():
    """DEFAULT mask as rendered at 1920x1080 output (lines 249-256, 264).

        mask     = 1.0 - MASK_STRENGTH = 0.7
        mod_fac  = floor(vTexCoord * outsize.xy * SourceSize.xy
                         / (InputSize.xy * vec2(MASK_SIZE, MASK_DOT_HEIGHT*MASK_SIZE)))
        dot_no   = int(mod((mod_fac.x + mod(mod_fac.y,2)*MASK_STAGGER)
                           / MASK_DOT_WIDTH, 3.0))

    With SourceSize == InputSize, vTexCoord*SourceSize/InputSize is the
    normalized image coordinate, so mod_fac == the integer OUTPUT pixel
    coordinate (MASK_SIZE = MASK_DOT_HEIGHT = 1). MASK_STAGGER=0 kills the
    row term and MASK_DOT_WIDTH=1 leaves

        dot_no = output_px_x mod 3      (no vertical variation at all)

    Channel assignment (lines 254-256), verified verbatim:
        dot_no == 0 -> vec3(1.0, 0.7, 0.7)     red column
        dot_no == 1 -> vec3(0.7, 1.0, 0.7)     green column
        else        -> vec3(0.7, 0.7, 1.0)     blue column

    => a 3x1 output-pixel APERTURE GRILLE tile (1 px per phosphor stripe at
    1080p, 640 triads across), applied in LINEAR (gamma-2) light BEFORE
    pow(1/1.8) at line 265."""
    m = 1.0 - MASK_STRENGTH
    tile = [[[1.0, m, m], [m, 1.0, m], [m, m, 1.0]]]     # 1 row x 3 cols x RGB
    avg = sum(sum(px) for row in tile for px in row) / (3 * 3)
    return {
        "tile_width_px": 3,
        "tile_height_px": 1,
        "triad_size_px": 3.0,
        "tiles_on_screen": [640, 1080],
        "mask_type": "aperture grille (1-px RGB stripes)",
        "sample_mode": "procedural, exact 1:1 output pixels (no resampling)",
        "applied_in": "linear light (gamma-2.0), after the scanline term and "
                      "before pow(1/GAMMA_OUTPUT); col *= mask_weight",
        "mask_strength": MASK_STRENGTH,
        "off_channel_multiplier": m,
        "mask_stagger": MASK_STAGGER,
        "vertical_period_px": 1,
        "dot_no_rule": "dot_no = output_px_x mod 3; 0->R bright, 1->G, 2->B",
        "avg_transmission": avg,                       # 0.8 linear, per channel
        "avg_net_multiplier": avg,                     # no amplify stage exists
        "tile_linear_multiplier": tile,
        "tile_encoded_multiplier": [[[v ** OUT_POW for v in px] for px in row]
                                    for row in tile],
        "tile_linear_multiplier_note":
            "per-pixel LINEAR multipliers, applied in gamma-2.0 light. MiSTer's "
            "v2 mask is applied in the ENCODED (post-LUT/FIR) domain, so use "
            "m^(1/GAMMA_OUTPUT) = 0.7^(1/1.8) = 0.82024 for the off channels; "
            "in 1/16 steps that is 13/16 = 0.8125 (token bitmask/16/13).",
    }


def defaults():
    return {
        "commit": "2b2c5ee3fd8e1a3884e20ed424fd9bfbc51cbb3d",
        "repo": "libretro/glsl-shaders",
        "shader": "crt/shaders/crt-easymode.glsl",
        "SHARPNESS_H": SHARPNESS_H,
        "SHARPNESS_H_effective_squared": SHARPNESS_H_EFF,
        "SHARPNESS_V": SHARPNESS_V,
        "MASK_STRENGTH": MASK_STRENGTH,
        "MASK_DOT_WIDTH": MASK_DOT_WIDTH,
        "MASK_DOT_HEIGHT": MASK_DOT_HEIGHT,
        "MASK_STAGGER": MASK_STAGGER,
        "MASK_SIZE": MASK_SIZE,
        "SCANLINE_STRENGTH": SCANLINE_STRENGTH,
        "SCANLINE_BEAM_WIDTH_MIN": SCANLINE_BEAM_WIDTH_MIN,
        "SCANLINE_BEAM_WIDTH_MAX": SCANLINE_BEAM_WIDTH_MAX,
        "SCANLINE_BRIGHT_MIN": SCANLINE_BRIGHT_MIN,
        "SCANLINE_BRIGHT_MAX": SCANLINE_BRIGHT_MAX,
        "SCANLINE_CUTOFF": SCANLINE_CUTOFF,
        "GAMMA_INPUT": GAMMA_INPUT,
        "GAMMA_OUTPUT": GAMMA_OUTPUT,
        "BRIGHT_BOOST": BRIGHT_BOOST,
        "DILATION": DILATION,
        "ENABLE_LANCZOS": ENABLE_LANCZOS,
        "fetch_gamma_from_dilation": FETCH_GAMMA,
        "post_filter_pow": POST_FILTER_POW,
        "scan_beam_constant": SCANLINE_BEAM_WIDTH_MIN,
        "scanlines_enabled_at_240p": SCANLINES_ENABLED,
        "bright_boost_clip_encoded": CLIP_ENCODED,
        "bright_boost_clip_x": CLIP_X,
        "source": "240p", "output": "1080p", "vscale": VSCALE,
    }


def notes():
    return [
        "BRIGHT_BOOST CLIPPING (largest single fidelity item): BRIGHT_BOOST=1.2 "
        "is applied AFTER pow(1/1.8) (line 267), so the top of the ramp is hard-"
        "clipped: every x >= 0.8487 maps to 255. That is 15.1% of the input "
        "range crushed to a single code, and transfer() is flat there. On "
        "MiSTer this is representable exactly (the gamma LUT is what carries "
        "transfer()), and it is REQUIRED to reproduce Easymode's look -- but it "
        "means the LUT saturates at index 216/255 and no highlight detail "
        "survives. Do not 'fix' it: the clip is why Easymode looks punchy.",

        "GAMMA MISMATCH IS INTENTIONAL: input decode is gamma 2.0 (dilate() "
        "squares fetches) and output encode is gamma 1.8, so even without the "
        "boost the transfer is x^1.111 -- a slight DARKENING (e.g. 0.5 -> "
        "0.463 pre-boost) that BRIGHT_BOOST then over-corrects to 0.556. The "
        "net is +11% at midgray vs identity. Any port that applies the boost "
        "in linear light instead will be ~6% off at midgray and will not clip "
        "at the same place. This mis-ordering is the #1 way to get the known "
        "'dim Easymode' failure.",

        "MASK IS 1-PX APERTURE GRILLE AT 1080p: 3x1 tile, linear multipliers "
        "1.0/0.7. MiSTer's v2 mask multiplies in the ENCODED domain, so the "
        "port must use 0.7^(1/1.8) = 0.82024, NOT 0.7 (using 0.7 directly "
        "would be -14.9% too dark in linear light -- a classic way to get the "
        "'dim Easymode' look). The 1/16 grid brackets it badly: 13/16 = 0.8125 "
        "is -1.69% in linear light (VERIFIED, the best choice); 14/16 = 0.875 "
        "is +12.3%; 12/16 = 0.75 is -14.9%. So use token bitmask/16/13. "
        "Additionally F3's truncating multiply for m=13 (bits 8+4+1 -> shifts "
        "1,2,4) loses up to 2.19 codes (VERIFIED worst case at x=15) -- a "
        "systematic extra darkening of the off channels. Net mask-stage error "
        "~2-3 codes, always in the DARK direction; consider compensating in "
        "the gamma LUT rather than the mask.",

        "MASK HAS NO VERTICAL STRUCTURE: MASK_STAGGER=0 and MASK_DOT_HEIGHT=1 "
        "mean dot_no depends only on output_px_x, so the grille is uniform "
        "down every column. Fully representable as a 3x1 (or 3x16) MiSTer "
        "tile; nothing lost.",

        "NO MASK COMPENSATION: unlike Royale there is no amplify/bloom stage. "
        "The mask is a pure 0.8-average dimming (per channel: 1.0 on 1 column "
        "of 3, 0.7 on the other 2). Mean luminance drops 20% in linear light "
        "with the mask on. If MiSTer's mask is used with the same LUT, expect "
        "the same 20% -- that is CORRECT, not a bug.",

        "H FILTER RUNS IN SQUARED (LINEAR) LIGHT -- BIGGEST UN-PORTABLE ITEM: "
        "the Lanczos taps weight x^2 samples, because dilate() squares at "
        "fetch. MiSTer's H FIR runs on post-gamma-LUT codes, i.e. on "
        "transfer(x) ~ 1.2*x^1.111 -- nearly ENCODED space, not x^2. Filtering "
        "in a convex vs a concave space is not a gain error, it is a different "
        "function, and no LUT+FIR ordering on MiSTer can fix it (it would need "
        "a pre-LUT of x^2 and a post-inverse around the FIR; the hardware has "
        "exactly one LUT, before the FIR). VERIFIED magnitude on a 0->255 hard "
        "edge at frac=0.5: shader linear avg 0.5 -> encoded 0.8165 = 208 "
        "codes; MiSTer FIR of LUT codes (0,0,255,255) = 128 codes; delta +81 "
        "codes. Worst over all phases/splits: +82 codes. Easymode's H edges "
        "are therefore MUCH brighter/softer-shouldered than any MiSTer port of "
        "them can be. This affects only horizontal transitions -- flat fields, "
        "and hence the fitted LUT and V tables, are unaffected.",

        "ANTI-RINGING CLAMP NOT REPRESENTABLE (and its error is content- "
        "dependent in a surprising way): filter_lanczos() clamps the 4-tap "
        "result to the min/max of the two INNER taps (line 207) -- nonlinear; "
        "MiSTer's H FIR is strictly linear polyphase. Inert on flat fields and "
        "monotone ramps (VERIFIED to 1e-12). Its strongest form is when the "
        "two inner taps are EQUAL: the clamp then pins the output to exactly "
        "that value and deletes the outer taps entirely, which a linear FIR "
        "cannot imitate at all. VERIFIED magnitudes (clamped shader vs the "
        "linear kernel this module returns, at 240p/1080p): random full-range "
        "content RMS 5.99 codes, max 48.8 codes; a 0.3->0.7 midtone edge peaks "
        "at 17.8 codes. NOTE the counter-intuitive part: on PURE 0/255 content "
        "the error is EXACTLY 0.00 codes, because the Lanczos overshoot above "
        "white is absorbed by the BRIGHT_BOOST framebuffer clip and the "
        "undershoot below black is absorbed by F1's negative hard-clamp to 0 "
        "-- the two clamps coincide. So this term is invisible on high- "
        "contrast pixel art and only bites on midtone detail.",

        "SHARPNESS_H IS SQUARED: line 223 passes SHARPNESS_H*SHARPNESS_H, so "
        "the documented default 0.5 acts as 0.25 -- a mostly-LINEAR distance "
        "warp (75% linear + 25% half-circle). This makes the effective H "
        "kernel wider/softer than the parameter name suggests. h_kernel() "
        "models the squared value; a port that forgets the squaring gets a "
        "visibly sharper, more pixelated image.",

        "VERTICAL IS A 2-TAP RESAMPLE, NOT A BEAM: with SHARPNESS_V=1.0 the "
        "vertical stage is a pure half-circle-warped 2-tap mix (beam_weight is "
        "0 beyond |d|=1). MiSTer's 4-tap V FIR can represent this EXACTLY: the "
        "outer two taps simply fit to 0. This is the easiest V fit of any "
        "shader in the pack -- expect near-zero shape error, with residual "
        "coming only from 1/256 coefficient quantization.",

        "SCAN BEAM WIDTH DOES NOT ADAPT: SCANLINE_BEAM_WIDTH_MIN == MAX == 1.5 "
        "makes scan_beam a compile-time constant; the cos^1.5 profile is "
        "IDENTICAL at every brightness. The ONLY brightness adaptation at "
        "defaults is scan_bright = clamp(bright, 0.35, 0.65), a trough floor. "
        "Consequence for the port: MiSTer's ADAPTIVE V filter has real work to "
        "do (trough depth must go from 35% floor at bright<=0.35 up to a 65% "
        "floor at bright>=0.65 -- i.e. troughs get RELATIVELY brighter as the "
        "image brightens) but the tap ratios never change. The dark endpoint "
        "and bright endpoint tables differ only by row SUM at the trough.",

        "*** CLIP-vs-TROUGH CONFLICT (irreducible; this is THE Easymode v5 "
        "design decision) ***: BRIGHT_BOOST clips the PEAK at x >= 0.8487, but "
        "the TROUGH is 0.65*x^2 encoded = 0.952 at white, which never clips. "
        "So across x in [216,255] the reference peak is pinned at 255 while "
        "the reference trough still rises 200 -> 241 codes. On MiSTer the "
        "gamma LUT must satisfy LUT >= 255*transfer (or the peak cannot be "
        "reached without row sums > 256), and transfer = 1.0 there, so LUT is "
        "FORCED to 255 for every x >= 216. The adaptive control lum is that "
        "same LUT value, so ctrl = 255 too: identical LUT value AND identical "
        "ctrl mean the hardware MUST emit an identical trough for every x in "
        "that band. VERIFIED cost: RMSE in the clipped band is 16.7 codes "
        "(max 42.0) vs 1.8 codes for x < 120; it alone lifts total RMSE from "
        "4.48 to 7.87. No V/LUT fit can fix this -- the information is gone "
        "before the V stage. The only levers are (a) accept flat highlight "
        "troughs (keeps Easymode's punch, costs ~17 codes of highlight "
        "scanline depth), or (b) scale the LUT below the clip so it stays "
        "invertible -- which reintroduces exactly the known 'dim Easymode' "
        "look. Recommend (a): the clip IS the Easymode aesthetic.",

        "SCAN_BRIGHT SATURATES EARLY, AND ITS KNEES COST MORE THAN EXPECTED: "
        "bright = x^2 in linear light, so scan_bright hits its 0.35 floor for "
        "all x <= 0.5916 and its 0.65 ceiling for all x >= 0.8062 -- the "
        "adaptive range is only x in [0.59, 0.81], 22% of the ramp, with a "
        "HARD KNEE at each end. MiSTer's adaptive blend is LINEAR in the "
        "control lum (= LUT ~ 1.2*x^1.111, not x^2), so a 2-endpoint fit must "
        "draw a straight line through a clamp-flat / ramp / clamp-flat curve. "
        "VERIFIED cost: RMSE 6.31 codes (max 18.7) over x in [120,216], vs "
        "1.77 codes (max 7.8) below 120 where scan_bright is pinned at its "
        "floor and the fit is nearly exact. So the mid-tones, not the darks, "
        "are where the adaptive V filter earns or loses its keep.",

        "PER-CHANNEL bright: bright = (max(R,G,B) + Rec709 luma)/2 is a SHARED "
        "scalar applied to all three channels, and MiSTer's adaptive control "
        "is max(R,G,B) only. On saturated colours the two differ: for pure "
        "blue, shader bright = (1 + 0.0722)/2 = 0.536 (scan_bright 0.536) "
        "while MiSTer's control = 1.0 (scan_bright 0.65). Pure-blue troughs "
        "will be ~21% too bright on the port; pure green (shader 0.858 -> "
        "0.65) matches. Unavoidable: the hardware exposes only max(R,G,B).",

        "SCANLINE_CUTOFF=400: at InputSize.y >= 400 (e.g. 480p sources) "
        "scan_weight is forced to 1.0 -- scanlines vanish while mask, filters, "
        "gamma and BRIGHT_BOOST stay. MiSTer has no such conditional; the port "
        "needs a separate 'V No Scanlines' table (fitting.no_scanline_table) "
        "selected by preset, which the pack already does.",

        "SCAN CENTRE PHASE: the shader's beam peak sits at the SOURCE TEXEL "
        "CENTRE (cos(2*PI*(n+0.5)) = -1 => scan_weight = 1), and its trough "
        "sits exactly between rows. The MiSTer V FIR's phase 0 is the texel "
        "centre too (F4), so f maps 1:1 to vfrac with no offset. Getting this "
        "half-phase wrong inverts the scanlines.",

        "TOOLCHAIN ARTIFACT (not a shader property, but it will bite): "
        "transfer(0) = 0 exactly, but fitting.optimize_lut samples xs starting "
        "at x=8/255 and np.interp flat-extrapolates below that, so the fitted "
        "LUT comes out with LUT[0..8] = 7 instead of 0 -- a +7-code BLACK LIFT. "
        "Easymode's whole look depends on a true black floor next to the "
        "clipped highlights, so the generator should pin LUT[0] = 0 (or extend "
        "the xs grid down to 0) before writing the Gamma file. This affects "
        "every ref in the pack equally; it is called out here because "
        "Easymode's contrast makes it most visible.",

        "FLOAT vs FIXED: the shader is float throughout with a single clamp at "
        "the framebuffer. MiSTer clamps to 8 bits after H, after V, and after "
        "the mask (F1/F3), and truncates in three places. Since Easymode's "
        "kernels are all partition-of-unity (H Lanczos normalized to 1, V mix "
        "sums to 1) nothing overshoots, so the intermediate clamps are inert "
        "on flat fields; the truncations cost <1 code each.",
    ]


if __name__ == "__main__":
    print("defaults:", defaults())
    for x in (0.0, 0.25, 0.5, 0.75, 0.848656, 0.9, 1.0):
        print(f"  transfer({x:.6f}) = {transfer(x):.6f}")
    for f in (0.0, 0.25, 0.5):
        print(f"  ref_vertical({f}, 0.5) = {ref_vertical(f, 0.5):.6f}")
    for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        print(f"  beam_weight({d}) = {beam_weight(d, 0.5):.6f}")
    print("  h_kernel(0.0) =", h_kernel(0.0))
    print("  h_kernel(0.5) =", h_kernel(0.5))
    ms = mask_spec()
    print("  mask avg transmission =", ms["avg_transmission"])
