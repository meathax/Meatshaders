"""
lottes_ref.py -- Reference implementation of CRT-LOTTES (Timothy Lottes) at
DEFAULT parameters, from libretro/glsl-shaders pinned to commit
2b2c5ee3fd8e1a3884e20ed424fd9bfbc51cbb3d (crt/shaders/crt-lottes.glsl).

Models the DEFAULT-parameter math for a 240p source displayed at 1920x1080
(4.5x vertical scale), flat / no-warp case. Pure Python (math only).

Defaults (crt-lottes.glsl lines 125-137, the #else branch of
#ifdef PARAMETER_UNIFORM -- identical to the #pragma parameter defaults on
lines 15-27, so the file's compiled-in constants and RetroArch's runtime
defaults agree):

    hardScan = -8.0        hardPix = -3.0         warpX = 0.031
    warpY = 0.041          maskDark = 0.5         maskLight = 1.5
    scaleInLinearGamma = 1.0                      shadowMask = 3.0
    brightBoost = 1.0      hardBloomPix = -1.5    hardBloomScan = -2.0
    bloomAmount = 0.15     shape = 2.0

DO_BLOOM is an unconditional `#define DO_BLOOM` (line 143) -- bloom is ON.
SIMPLE_LINEAR_GAMMA is commented out (line 142), so Fetch() uses the true
piecewise sRGB->linear and the final ToSrgb() is the piecewise sRGB encode.

Pipeline (main(), lines 403-424):
    pos      = Warp(...)                       # curvature; NOT modelled (flat)
    outColor = Tri(pos)                        # 3 lines, un-normalized Scan()
    outColor += Bloom(pos) * bloomAmount       # 5 lines, un-normalized BloomScan()
    outColor *= Mask(gl_FragCoord.xy*1.000001) # LINEAR-light mask
    FragColor = ToSrgb(outColor)               # clamped by the 8-bit target

Everything up to ToSrgb() is TRUE sRGB-LINEAR light.

Interface (contract for the automated fitting stage):
    transfer(x)         encoded->encoded flat field at beam center, mask off
    beam_weight(d, L)   linear-light beam profile weight vs distance/brightness
    ref_vertical(f, x)  encoded output across the scanline period, mask off
    h_kernel(frac)      effective horizontal kernel (Horz3/Horz5/Horz7 blend)
    mask_spec()         shadowMask=3 stretched-VGA mask as rendered at 1080p
    notes()             features not representable on MiSTer
    defaults()          dict of all default parameter values
"""

import math

# ----------------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------------

HARD_SCAN = -8.0
HARD_PIX = -3.0
WARP_X = 0.031
WARP_Y = 0.041
MASK_DARK = 0.5
MASK_LIGHT = 1.5
SCALE_IN_LINEAR_GAMMA = 1.0
SHADOW_MASK = 3.0
BRIGHT_BOOST = 1.0
HARD_BLOOM_PIX = -1.5
HARD_BLOOM_SCAN = -2.0
BLOOM_AMOUNT = 0.15
SHAPE = 2.0
DO_BLOOM = True

# Geometry of the modeled configuration: 240p source -> 1080p output
SOURCE_LINES = 240.0
OUTPUT_LINES = 1080.0
PIXEL_HEIGHT = SOURCE_LINES / OUTPUT_LINES      # 0.2222 source lines / output px

# Tri() taps: nearest source line +/- 1  -> distances {f, 1-f, 1+f} for f in [0,.5]
TRI_SUPPORT = 1.5
# Bloom() taps: nearest source line +/- 2 -> distances {f, 1-f, 1+f, 2-f, 2+f}
BLOOM_SUPPORT = 2.5
_EPS = 1e-9


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# ----------------------------------------------------------------------------
# Gamma (crt-lottes.glsl lines 163-195), exactly as written
# ----------------------------------------------------------------------------

def to_linear1(c):
    """ToLinear1: piecewise sRGB -> linear (scaleInLinearGamma = 1)."""
    if SCALE_IN_LINEAR_GAMMA == 0.0:
        return c
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def to_srgb1(c):
    """ToSrgb1: piecewise linear -> sRGB.  NOTE the shader writes the literal
    0.41666, NOT 1/2.4 = 0.4166667 -- reproduced verbatim (worst-case
    difference over [0,1] is well under 0.01 output codes)."""
    if SCALE_IN_LINEAR_GAMMA == 0.0:
        return c
    return c * 12.92 if c < 0.0031308 else 1.055 * (c ** 0.41666) - 0.055


def fetch_linear(x):
    """Fetch() (line 205): ToLinear(brightBoost * texel).  brightBoost
    multiplies in ENCODED space *before* linearization; at the default
    brightBoost = 1.0 this is a no-op."""
    return to_linear1(_clamp01(BRIGHT_BOOST * x))


# ----------------------------------------------------------------------------
# Gaussian kernel (line 218-221)
# ----------------------------------------------------------------------------

def gaus(pos, scale):
    """Gaus(pos, scale) = exp2(scale * pow(abs(pos), shape)); shape = 2.0."""
    return 2.0 ** (scale * abs(pos) ** SHAPE)


# ----------------------------------------------------------------------------
# Vertical: Tri() (lines 305-316) and Bloom() (lines 319-334)
#
# Dist(pos).y (line 210-215) = -((frac) - 0.5) = 0.5 - frac, i.e. the SIGNED
# offset from the sample to the NEAREST source line centre; the line at texel
# offset `off` therefore sits at distance dst + off, which is exactly what
# Scan()/BloomScan() evaluate.  Parameterizing by f = |dst| in [0, 0.5]:
#   Tri()   lines lie at distances {f, 1-f, 1+f}
#   Bloom() lines lie at distances {f, 1-f, 1+f, 2-f, 2+f}
# Scan()/BloomScan() weights are NOT normalized -- their un-normalized sum IS
# the scanline profile.  (The Horz3/5/7 H weights, by contrast, ARE normalized;
# on a flat field each Horz* returns exactly the flat value, so they drop out.)
# ----------------------------------------------------------------------------

def _tri_weight_sum(f):
    """Sum of Scan() weights over Tri()'s 3 lines at |dst| = f."""
    return gaus(f, HARD_SCAN) + gaus(1.0 - f, HARD_SCAN) + gaus(1.0 + f, HARD_SCAN)


def _bloom_weight_sum(f):
    """Sum of BloomScan() weights over Bloom()'s 5 lines at |dst| = f."""
    return (gaus(f, HARD_BLOOM_SCAN)
            + gaus(1.0 - f, HARD_BLOOM_SCAN) + gaus(1.0 + f, HARD_BLOOM_SCAN)
            + gaus(2.0 - f, HARD_BLOOM_SCAN) + gaus(2.0 + f, HARD_BLOOM_SCAN))


def _vertical_gain(f):
    """Total LINEAR-light gain applied to a uniform field at phase f, mask off.
    Verified: f=0 -> 1.007812 + 0.15*1.507813 = 1.233984 (peak)
              f=0.5 -> 0.500004 + 0.15*1.502775 = 0.725420 (trough)
              trough/peak = 58.79% in linear light."""
    g = _tri_weight_sum(f)
    if DO_BLOOM:
        g += BLOOM_AMOUNT * _bloom_weight_sum(f)
    return g


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------

def ref_vertical(f, x):
    """Encoded (0..1) output at vertical fraction f within one scanline period
    (f=0 = scanline centre, f=0.5 = midway between lines) for a uniform encoded
    input x, MASK DISABLED, all other defaults.  240p source at 1080p output.

    Chain: sRGB->linear (piecewise) -> un-normalized 3-line Scan() sum
    -> + 0.15 * un-normalized 5-line BloomScan() sum -> [mask = 1]
    -> linear->sRGB (piecewise) -> 8-bit render-target clamp.

    Scanline depth is CONSTANT vs brightness (Scan() has no L dependence): the
    linear profile is a pure scalar multiple of the input's linear value.  All
    brightness dependence in the ENCODED output comes from the sRGB curve plus
    the clamp (which engages at the beam centre for x > ~0.9115)."""
    return _clamp01(ref_vertical_unclipped(f, x))


def ref_vertical_unclipped(f, x):
    """ref_vertical WITHOUT the render-target clamp.

    The shader clamps only after the LINEAR-light mask multiply
    (ToSrgb(beam*mask) -> 8-bit target), so a port whose gain lives in the
    mask stage must fit against this pre-clip beam. Exposing it lets the gamma
    LUT carry B(x)/G, which keeps rising past x=0.9115 instead of pinning at
    255 and taking the adaptive control down with it.
    """
    x = _clamp01(x)
    f = abs(f) % 1.0
    if f > 0.5:
        f = 1.0 - f                     # profile is symmetric about f = 0.5
    lin = fetch_linear(x) * _vertical_gain(f)
    # ToSrgb() is monotone, so the render target's clamp is equivalent whether
    # applied before or after the encode; apply it on the encoded value.
    return to_srgb1(lin)


def transfer_unclipped(x):
    """Beam-centre transfer before the render-target clamp; peak linear gain
    1.233984 encodes to ~1.0938, so that is the headroom a gain-split needs."""
    return ref_vertical_unclipped(0.0, x)


def transfer(x):
    """End-to-end encoded->encoded transfer for a uniform field sampled at the
    scanline beam CENTRE (f = 0), mask disabled.  Peak linear gain is 1.233984,
    so the 8-bit target clamps for x > ~0.9115."""
    return ref_vertical(0.0, x)


def beam_weight(d, L):
    """LINEAR-light contribution weight of one scanline at vertical distance d
    (source lines; covers |d| <= 2) when the line's encoded brightness is L.

    BRIGHTNESS-INDEPENDENT by construction: Scan()/BloomScan() depend only on
    the geometric distance, so `L` is accepted for interface compatibility and
    ignored.  The pack's adaptive fitter will therefore produce two
    near-identical endpoint sets -- which is exactly why Lottes degrades
    gracefully on non-adaptive cores (and matches the shipped v4 tables, whose
    dark/bright phase-0 rows are both [7, 224, 7, 0]).

    Tri/Bloom DECOMPOSITION.  The reference's vertical response is
        out_linear(f) = E * [ sum_Tri Scan(d) + 0.15 * sum_Bloom BloomScan(d) ]
    Both terms are sums over source LINES at the same distances, so they fold
    exactly into ONE per-line weight:
        W(d) = Scan-part(d) + 0.15 * BloomScan-part(d)
    with each part gated by its own tap count (Tri covers the nearest line +/-1,
    i.e. |d| <= 1.5; Bloom covers +/-2, i.e. |d| <= 2.5).  Within the shader's
    own tap windows this decomposition is EXACT -- there is no approximation in
    splitting Tri from Bloom, because they are both linear per-line sums.

    RESIDUAL vs the fitter's 4-tap window.  fitting._tap_targets evaluates taps
    at d = {f+1, f, 1-f, 2-f}, which omits Bloom's far line at d = 2+f.  That
    line carries 0.15*BloomScan(2+f): 5.86e-4 of 1.233984 (0.047% relative) at
    f=0, falling to 2.6e-5 (0.0036%) at f=0.5.  Because _tap_targets uses
    beam_weight only for tap RATIOS and takes the row SUM from ref_vertical, the
    omitted energy is redistributed into the 4 kept taps rather than lost, so
    flat-field response stays exact; the residual is a <=0.05% mis-shaping of
    the beam tails (< 0.03 output codes).

    DEGENERATE PHASE f = 0.5.  There the tap distances collide (1+f == 2-f ==
    1.5) and a pure function of d cannot know that Tri() supplies only ONE line
    at 1.5, not two.  W(1.5) includes the Tri part for both, over-counting
    Gaus(1.5, -8) = 2^-18 = 3.8e-6, i.e. 5.3e-6 relative (sum 0.725424 vs the
    true 0.725420).  Left uncorrected on purpose: hacking it would break the
    "weight is a function of d alone" contract, and row sums are pinned by
    ref_vertical() regardless, so the effect on the fitted tables is nil."""
    d = abs(d)
    w = 0.0
    if d <= TRI_SUPPORT + _EPS:
        w += gaus(d, HARD_SCAN)
    if DO_BLOOM and d <= BLOOM_SUPPORT + _EPS:
        w += BLOOM_AMOUNT * gaus(d, HARD_BLOOM_SCAN)
    return w


# -- horizontal ---------------------------------------------------------------
#
# Tri() uses Horz3 (hardPix) for the outer 2 lines and Horz5 (hardPix) for the
# nearest line.  Bloom() uses Horz5 (hardPix) for lines +/-2 and Horz7
# (hardBloomPix) for lines -1/0/+1.  So the TRUE horizontal response is
# LINE-DEPENDENT: the operator is not separable and no single H FIR reproduces
# it.  The best single kernel is the LS-optimal rank-one (separable) factor
# GIVEN that the vertical profile is pinned to the true one (which is what
# fitting.fit_v_* does, taking row sums from ref_vertical).
#
# For each phase f, effective kernel = sum_g vw_g(f) * Hg_norm(.), and the
# separable model is vtot(f) * h(.).  Minimizing sum_f ||.||^2 over h gives the
# closed form   alpha_g = <vtot, vw_g> / <vtot, vtot>   (which sums to 1):
#
#     alpha_Horz3 = 0.060038   alpha_Horz5 = 0.719201   alpha_Horz7 = 0.220760
#
# (Horz3 and Horz5 share hardPix and differ only in their +/-2 taps, worth
# ~2.4e-4 relative, so their split barely matters; the Horz7 share is what sets
# horizontal softness.)

_ALPHA_H3 = 0.06003828
_ALPHA_H5 = 0.71920149
_ALPHA_H7 = 0.22076023


def _horz_kernel(frac, ntaps, scale):
    """Normalized Horz3/Horz5/Horz7 weights, keyed by source texel index k.

    Fetch(pos, vec2(off, .)) reads texel n0+off where n0 is the NEAREST texel;
    its weight is Gaus(dst + off) and its offset from the sample is dst + off.
    So a texel sitting at offset o from the sample always has weight Gaus(o),
    and the only phase dependence is WHICH texels are in the window (the window
    recentres on the nearest texel as frac crosses 0.5).

    Contract convention (same as royale_ref/guest_advanced_ref): texel k sits at
    offset k - frac, so texel 0 is the one at or left of the sample."""
    half = ntaps // 2
    n0 = 0 if frac <= 0.5 else 1                      # index of the nearest texel
    ks = range(n0 - half, n0 + half + 1)
    w = {k: gaus(k - frac, scale) for k in ks}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def h_kernel(frac):
    """[(tap_offset, weight), ...] effective normalized horizontal kernel at
    fractional source-pixel offset frac (0..1).  tap_offset is in SOURCE pixels
    relative to the output sample position (offset = texel position - sample
    position); weights sum to 1.

    Blend of the three kernels the shader actually uses, at their LS-optimal
    separable shares (see above):
        0.060038 * Horz3(hardPix = -3.0)      3 taps, exp2(-3 d^2)
        0.719201 * Horz5(hardPix = -3.0)      5 taps, exp2(-3 d^2)
        0.220760 * Horz7(hardBloomPix = -1.5) 7 taps, exp2(-1.5 d^2)
    Each Hg is normalized by its own weight sum exactly as the shader does
    (lines 238, 259, 285), so the blend is normalized too.

    APPROXIMATION.  This is a rank-one factorization of a genuinely
    non-separable operator; see notes() for the magnitude.  It also mixes in
    LINEAR light, whereas MiSTer's H FIR runs on gamma-LUT'd (encoded) codes --
    also quantified in notes()."""
    frac = frac % 1.0
    taps = {}
    for alpha, ntaps, scale in ((_ALPHA_H3, 3, HARD_PIX),
                                (_ALPHA_H5, 5, HARD_PIX),
                                (_ALPHA_H7, 7, HARD_BLOOM_PIX)):
        for k, w in _horz_kernel(frac, ntaps, scale).items():
            taps[k] = taps.get(k, 0.0) + alpha * w
    return sorted((float(k) - frac, w) for k, w in taps.items())


# -- mask ---------------------------------------------------------------------

def _mask_channel(px, py):
    """Mask() with shadowMask == 3 (lines 378-386) for output pixel (px, py),
    evaluated exactly as main() line 414 does: Mask(gl_FragCoord.xy*1.000001).

        pos.x += pos.y*3.0
        pos.x  = fract(pos.x*0.166666666)
        pos.x < 0.333 -> R,  < 0.666 -> G,  else B     (that channel = maskLight)

    gl_FragCoord for pixel (px, py) is (px+0.5, py+0.5), so
        pos.x + 3*pos.y = (px + 3*py + 2) * 1.000001
    Writing n = px + 3*py + 2 (an integer), the fract() argument is
        n * (1.000001 * 0.166666666) = n/6 + n * 1.66e-7
    Over the whole 1920x1080 frame n <= 5158, so the perturbation is at most
    8.6e-4 -- smaller than every threshold margin (fract(n/6) lands on
    0, .16667, .33333, .5, .66667, .83333 and the tests are at .333/.666, whose
    nearest margins are 3.3e-4 and 6.7e-4 UPWARD, and the perturbation only
    ever pushes upward).  So the selection is exactly n mod 6:
        n mod 6 in {0,1} -> R,  {2,3} -> G,  {4,5} -> B
    => period 6 in x, period 2 in y (from 3*py mod 6): a 6x2 tile, exact across
    the entire frame.  The 1.000001 nudge is inert here (it exists to break
    floor() ties in the shadowMask 1/4 paths)."""
    n = (px + 3 * py + 2) % 6
    return 0 if n < 2 else (1 if n < 4 else 2)


def _mask_tile():
    tile = []
    for py in range(2):
        row = []
        for px in range(6):
            lit = _mask_channel(px, py)
            row.append([MASK_LIGHT if c == lit else MASK_DARK for c in range(3)])
        tile.append(row)
    return tile


MASK_TILE_LINEAR = _mask_tile()


def _canonical_tokens():
    """MiSTer v2 mask tokens for the TRUE tile, derived (not hardcoded).

    Token XYZ hex: X = channel bitmask (4=R, 2=G, 1=B); selected channels get
    (16+Y)/16, others Z/16.  Encoded multipliers are m^(1/2.2), quantized to
    the 1/16 grid: 1.5 -> 1.202379 -> Y = round(1.202379*16) - 16 = 3;
    0.5 -> 0.729740 -> Z = round(0.729740*16) = 12 = 'c'."""
    y = round(MASK_LIGHT ** (1.0 / 2.2) * 16) - 16
    z = round(MASK_DARK ** (1.0 / 2.2) * 16)
    return [["%x%x%x" % (4 >> _mask_channel(px, py), y, z) for px in range(6)]
            for py in range(2)]


def mask_spec():
    """DEFAULT mask (shadowMask = 3.0, "stretched VGA") as rendered at 1080p.

    6x2 output-pixel tile, exact for the whole frame (derivation in
    _mask_channel).  Each output pixel lights exactly ONE channel at
    maskLight = 1.5; the other two sit at maskDark = 0.5.  Each channel is lit
    for 2 of every 6 pixels, so the per-channel LINEAR average transmission is
        (2*1.5 + 4*0.5) / 6 = 5/6 = 0.833333
    Applied in LINEAR light (main() line 414), BEFORE the sRGB encode.

    ENCODED-SPACE PORT.  MiSTer's v2 mask multiplies in the encoded 8-bit
    domain, after gamma+scaler.  sRGB is close to a pure 2.2 power over the
    mask's operating range, so multiplier m maps to m^(1/2.2):
        1.5 -> 1.202379   (x16 = 19.24 -> 19/16 = 1.1875, i.e. token nibble Y=3)
        0.5 -> 0.729740   (x16 = 11.68 -> 12/16 = 0.75,   i.e. token nibble Z=c)
    which reproduces the shipped canonical mask tokens 43c/23c/13c EXACTLY.
    Mean encoded multiplier = (2*1.202379 + 4*0.729740)/6 = 0.887286."""
    lin_avg = (2.0 * MASK_LIGHT + 4.0 * MASK_DARK) / 6.0
    enc = [[[m ** (1.0 / 2.2) for m in px] for px in row] for row in MASK_TILE_LINEAR]
    enc_avg = (2.0 * MASK_LIGHT ** (1 / 2.2) + 4.0 * MASK_DARK ** (1 / 2.2)) / 6.0
    return {
        "mask_type": "stretched VGA (shadowMask = 3.0)",
        "tile_width_px": 6,
        "tile_height_px": 2,
        "tiles_on_screen": [320, 540],
        "exact_at_1080p": True,
        "rule": "lit channel index = ((px + 3*py + 2) mod 6) // 2  "
                "(0=R, 1=G, 2=B); lit = maskLight 1.5, others = maskDark 0.5",
        "linear_multipliers": MASK_TILE_LINEAR,   # [py][px][channel]
        "applied_in": "LINEAR light, multiplies (Tri + 0.15*Bloom) before ToSrgb",
        "mask_light": MASK_LIGHT,
        "mask_dark": MASK_DARK,
        "avg_transmission": lin_avg,                       # 0.833333, linear
        "avg_transmission_per_channel": [lin_avg] * 3,
        "encoded_equivalent_multipliers": enc,             # m^(1/2.2)
        "avg_transmission_encoded": enc_avg,               # 0.887286
        "mister_v2_tokens_canonical": _canonical_tokens(),
        "mister_v2_tokens_shipped_v4": [["43c", "43c", "23c", "23c", "13c", "13c"],
                                        ["23c", "13c", "13c", "43c", "43c", "23c"]],
        "phase_note":
            "The shipped v4 tiles start at R for output pixel (0,0); the shader "
            "gives G there (n = 2). The v4 grid is the true tile rolled LEFT by "
            "4 px (equivalently right by 2) -- verified exactly; the tokens and "
            "the +3px/line stretch are otherwise identical. A whole-tile phase "
            "shift of a screen-wide repeating mask is invisible, but it IS a "
            "deviation from the source.",
    }


def defaults():
    return {
        "commit": "2b2c5ee3fd8e1a3884e20ed424fd9bfbc51cbb3d",
        "repo": "libretro/glsl-shaders",
        "shader": "crt/shaders/crt-lottes.glsl",
        "hardScan": HARD_SCAN, "hardPix": HARD_PIX,
        "warpX": WARP_X, "warpY": WARP_Y,
        "maskDark": MASK_DARK, "maskLight": MASK_LIGHT,
        "scaleInLinearGamma": SCALE_IN_LINEAR_GAMMA,
        "shadowMask": SHADOW_MASK,
        "brightBoost": BRIGHT_BOOST,
        "hardBloomPix": HARD_BLOOM_PIX, "hardBloomScan": HARD_BLOOM_SCAN,
        "bloomAmount": BLOOM_AMOUNT, "shape": SHAPE,
        "DO_BLOOM": DO_BLOOM,
        "SIMPLE_LINEAR_GAMMA": False,
        "gamma_space": "true piecewise sRGB both directions "
                       "(ToSrgb uses the literal exponent 0.41666)",
        "peak_linear_gain_f0": 1.233984375,
        "trough_linear_gain_f05": 0.7254199962242028,
        "trough_over_peak_linear": 0.5878690,
        "clamp_onset_x": 0.911511,       # bisected; = 232.44 input codes
        "h_alpha_horz3": _ALPHA_H3,
        "h_alpha_horz5": _ALPHA_H5,
        "h_alpha_horz7": _ALPHA_H7,
        "scanline_depth_brightness_adaptive": False,
        "pixel_height_scanlines": PIXEL_HEIGHT,
        "source": "240p", "output": "1080p",
    }


def notes():
    return [
        "CURVATURE (warpX = 0.031, warpY = 0.041) is ON by default and is NOT "
        "representable: MiSTer's scaler is a separable polyphase FIR with no "
        "geometric distortion. Magnitude at 1080p: Warp() maps pos -> "
        "pos*(1 + posy^2*warpX, 1 + posx^2*warpY) in [-1,1] NDC, so the corners "
        "pull in by warpX/(1+warpX) = 3.01% horizontally (~29 px of 960) and "
        "warpY/(1+warpY) = 3.94% vertically (~21 px of 540); the screen edges "
        "bow by the same amount and content outside is dropped. Beyond the "
        "geometry, warp also makes the LOCAL scanline phase drift across the "
        "frame (up to ~1 source line at the corners vs centre), which slightly "
        "changes moire beating. Omitting warp is the standard MiSTer-port "
        "choice and is what the v4 pack already does; the flat model is exact "
        "at screen centre and diverges monotonically toward the edges.",

        "LINE-DEPENDENT H KERNELS (Horz3/Horz5/Horz7) are NOT representable. "
        "Tri() filters the nearest line with Horz5 and the outer two with "
        "Horz3 (both hardPix = -3.0, exp2(-3 d^2)); Bloom() filters lines "
        "-1/0/+1 with the much softer Horz7 (hardBloomPix = -1.5, "
        "exp2(-1.5 d^2)) and lines +/-2 with Horz5. The 2D operator is "
        "therefore NOT separable, while MiSTer is strictly H-FIR-then-V-FIR. "
        "h_kernel() returns the LS-optimal rank-one factor "
        "(alpha = 0.0600/0.7192/0.2208 for H3/H5/H7). Magnitude of the error: "
        "the Horz7 share of the response swings from 18.23% at the scanline "
        "centre to 30.16% in the trough, so a fixed blend renders the bright "
        "core ~4% too soft and the troughs ~9% too sharp (relative to their "
        "correct Horz7 content). This is a horizontal-detail effect only: on "
        "flat fields every Horz* returns the flat value and the error is "
        "identically zero, so it does NOT enter the flat-field RMSE gate.",

        "H3/H5 SPLIT is nearly moot: Horz3 and Horz5 share hardPix and differ "
        "only by their +/-2 taps, worth exp2(-3*4) = 2.4e-4 relative. The "
        "meaningful horizontal degree of freedom is the Horz7 (bloom) share.",

        "H-BLEND CHOICE (LOOK RISK). The retained H header states "
        "'Horz5/Horz7 alpha = 0.163618283034'. "
        "This module's LS-optimal separable share is 0.2208. Neither "
        "beam-centre weighting (0.1823), plain period-mean (0.2279), nor "
        "vtot^2/vtot^3 weighting (0.2145/0.2092) reproduces 0.1636, so the "
        "retained table used a different rank-one objective. A newly fitted "
        "from this h_kernel will be measurably SOFTER horizontally than the "
        "gold-standard retained H table. Since the brief is 'beat the RMSE "
        "without changing the look', keep that H table verbatim (it is not "
        "what the flat-field RMSE measures anyway) or override alpha_H7 to "
        "0.1636 before fitting.",

        "MIXING SPACE MISMATCH. All of Lottes' filtering (Horz*, Tri, Bloom) "
        "sums in TRUE sRGB-LINEAR light; MiSTer's H and V FIRs sum in the "
        "gamma-LUT'd (encoded) domain. The pack's strategy hides this on flat "
        "fields (LUT = transfer, V row sums = 256*ref_vertical/transfer, so "
        "uniform response is exact by construction) but NOT on edges. Worst "
        "case, a black/white source edge at 50/50 mix: linear mean 0.5 encodes "
        "to 0.7354 (188 codes) while an encoded-domain mean gives 0.5 "
        "(128 codes) -- a 60-code gap on the sharpest possible transition. "
        "Realistic content is far milder because hardPix = -3 puts ~80% of the "
        "H kernel on one texel, but this is the single largest structural error "
        "in the port and it is invisible to every flat-field metric.",

        "SCANLINE DEPTH IS BRIGHTNESS-INDEPENDENT. Scan() = Gaus(dst+off, "
        "hardScan) has no dependence on the line's value, so the linear "
        "trough/peak ratio is a constant 58.79% at every brightness. The "
        "adaptive V fitter will therefore return two near-identical endpoint "
        "sets (the shipped v4 tables already do: dark and bright phase-0 rows "
        "are both [7, 224, 7, 0]). This is EXPECTED, and it is precisely why "
        "Lottes degrades gracefully on cores without adaptive filtering. All "
        "remaining brightness dependence in the encoded output comes from the "
        "sRGB curve and the 8-bit clamp, both of which the gamma LUT captures.",

        "PEAK GAIN > 1 AND CLIPPING. The un-normalized vertical sum peaks at "
        "1.233984 in linear light, so the shader's own render target clips at "
        "the beam centre for x > 0.9115 (encoded). ref_vertical()/transfer() "
        "model this clamp. Consequence for the port: with the gamma LUT "
        "carrying transfer(), the LUT saturates at 255 from x = 0.9115 upward "
        "and highlight detail above that is genuinely lost -- but it is lost in "
        "the source too, so this is faithful, not a defect. NOTE the shipped v4 "
        "files instead run with gamma = OFF and recover the >1 gain in the "
        "MASK stage (see the mask note); that is a different, non-canonical "
        "factorization of the same product.",

        "BLOOM'S FAR LINE (d = 2+f) falls outside the fitter's 4-tap window "
        "{f+1, f, 1-f, 2-f}. It carries 0.15*BloomScan(2+f): 0.047% of the "
        "total at f=0, 0.0036% at f=0.5. Since row sums come from "
        "ref_vertical() and beam_weight() only sets tap ratios, the energy is "
        "redistributed, not lost; flat-field response stays exact and the "
        "residual is a <0.03-code mis-shaping of the beam tails.",

        "BLOOM IS NOT A SEPARATE PASS, so nothing is lost vertically: Bloom() "
        "is an un-normalized 5-line sum evaluated at the same per-pixel phase "
        "as Tri(), so Tri + 0.15*Bloom folds EXACTLY into one per-line weight "
        "W(d) = [|d|<=1.5] Gaus(d,-8) + 0.15 [|d|<=2.5] Gaus(d,-2). The bloom "
        "fills troughs hard: it contributes 18.2% of the peak but 31.1% of the "
        "trough, lifting trough/peak from 49.6% (Tri alone) to 58.79%. Any "
        "port that drops bloom is ~9 points too contrasty -- the v4 audit "
        "measured the no-bloom 'Crisp' profile at 12.9 codes RMSE vs 1.8 for "
        "the bloom-matched pair.",

        "MASK IS APPLIED IN LINEAR LIGHT, MiSTer's in encoded. Converting via "
        "m^(1/2.2) gives 1.5 -> 1.2024 and 0.5 -> 0.7297, which quantize to "
        "MiSTer's 1/16 grid as 19/16 = 1.1875 and 12/16 = 0.75 -- exactly the "
        "shipped canonical tokens 43c/23c/13c. Quantization error is -1.2% on "
        "the lit channel and +2.8% on the dark channels. The 6x2 tile is well "
        "inside MiSTer's 16x16 limit and the lit multiplier 1.1875 is well "
        "under the 31/16 = 1.9375 ceiling, so the canonical mask is FULLY "
        "representable -- unusually, no compromise is needed here.",

        "THE SHIPPED 'MATCHED' MASK IS NOT CANONICAL. Shadow_Masks/CRT Lottes "
        "Matched Mask3 StretchedVGA (Port).txt uses 47e/27e/17e = 1.4375 lit / "
        "0.875 dark, i.e. the canonical tile scaled by ~1.1975 (mean encoded "
        "multiplier 1.0625 vs the true 0.8873). That +19.75% is a deliberate "
        "brightness compensation that lets the v4 port run gamma = OFF: the V "
        "table's peak row sum is 238/256 = 0.9297 and 0.9297 * 1.4375 = 1.3359 "
        "supplies the lit-pixel gain that the canonical chain gets from "
        "transfer(). The product is matched; the factors are not. This is why "
        "v4 CANNOT be scored mask-off against this reference (see report).",

        "MASK TILE PHASE. The true tile lights R at output pixel (0,0) only if "
        "n = px+3py+2 == 0 or 1 mod 6; at (0,0), n = 2, so the true tile starts "
        "on GREEN: [23c,23c,13c,13c,43c,43c / 13c,43c,43c,23c,23c,13c]. Both "
        "shipped v4 tiles start on RED and are exactly that tile rolled LEFT by "
        "4 px (verified). Invisible (a screen-wide repeating mask has arbitrary "
        "origin phase) but it is a deviation worth preserving deliberately "
        "rather than by accident.",

        "GL_ES BORDER CLAMP (lines 416-422) is compiled out on desktop and "
        "would only blank pixels outside the warped [0,1] source rect; with "
        "warp omitted there is nothing to model.",

        "8-BIT SOURCE QUANTIZATION: Fetch() is NEAREST (line 201, "
        "floor(pos*SourceSize + off) + 0.5), so there is no source-texel "
        "interpolation to model -- the shader's own H/V kernels do all the "
        "resampling. This matches MiSTer's polyphase FIR structure exactly and "
        "is one reason Lottes ports so cleanly.",

        "brightBoost = 1.0 is a no-op, but note it multiplies in ENCODED space "
        "BEFORE ToLinear() (line 205), not in linear light -- relevant only if "
        "a non-default preset is ever ported.",

        "ToSrgb1 USES THE LITERAL 0.41666, not 1/2.4 = 0.4166667 (line 186). "
        "Reproduced verbatim; the resulting deviation from an exact sRGB "
        "encode is < 0.01 output codes over [0,1] and is not a portability "
        "concern -- it is noted only so the reference is bit-honest.",
    ]


if __name__ == "__main__":
    print("defaults:", defaults())
    print("transfer(0.5) =", transfer(0.5))
    print("ref_vertical(0.0, 1.0) =", ref_vertical(0.0, 1.0))
    print("ref_vertical(0.5, 0.5) =", ref_vertical(0.5, 0.5))
    print("beam_weight(0, 0.5) =", beam_weight(0.0, 0.5))
    print("h_kernel(0.5) =", h_kernel(0.5))
    ms = mask_spec()
    print("mask avg transmission (linear) =", ms["avg_transmission"])
    print("mask avg transmission (encoded) =", ms["avg_transmission_encoded"])
