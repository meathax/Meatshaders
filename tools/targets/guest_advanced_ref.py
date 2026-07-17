"""
Reference implementation of CRT-GUEST-ADVANCED (guest.r) at DEFAULT parameters.

Source: libretro/slang-shaders @ 3b0d6aa1d134a168478cd9c904a866d969f8882b (master, 2026-07-15)
Preset: crt/crt-guest-advanced.slangp  (sets NO parameter overrides -> pure #pragma defaults)

Pass chain (12 passes):
  0 stock, 1 stock(StockPass), 2 afterglow0(AfterglowPass), 3 pre-shaders-afterglow(PrePass),
  4 avg-lum(AvgLumPass), 5 linearize(LinearizePass, float FB), 6 gaussian_horizontal(800xH),
  7 gaussian_vertical(800x600, GlowPass), 8 bloom_horizontal(800x600),
  9 bloom_vertical(source-size, BloomPass), 10 crt-guest-advanced(viewport), 11 deconvergence(viewport)

DEFAULTS that matter (see guest_advanced_report.md for the full table):
  GAMMA_INPUT=2.4, gamma_out=2.4, scangamma=2.4, mask_gamma=2.4  -> all spatial math in LINEAR light
  gsl=0 (scanline type 0 -> sw0), scanline1=6.0, scanline2=8.0
  beam_min=1.30, beam_max=1.00, beam_size=0.60, scans=0.50 (code uses 1.5*scans=0.75)
  scan_falloff=1.0, spike=1.0, tds=0, rolling_scan=0, no_scanlines=0
  brightboost(dark)=1.40, brightboost1(bright)=1.10, gamma_c=1.0, gamma_c2=1.0
  h_sharp=5.20, s_sharp=0.50, ring=0.0, smart_ei=0
  glow=0.08 (ordinary glow, m_glow=0), bloom=0.0, halation=0.0, mask_bloom=0.0
  shadowMask=0 (CGWG magenta/green), maskstr=0.3, masksize=1, mask_zoom=0, maskboost=1.0
  slotmask=slotmask1=0, bmask=0, mclip=0, edgemask=0, smoothmask=0
  pr_scan=0.10 ("Preserve Scanline Properties" -> post multiply, INCLUDED here)
  deconvergence all 0, warpX=warpY=0, post_br=1.0, interm=1 (inter trigger 375)

For a 240p/224p source at default settings the linearize pass emits intera=1.0,
so interb=false -> the full scanline path below is active.

All functions are pure; only `math` is required.
"""

import math

# ----------------------------------------------------------------------------- defaults

DEFAULTS = {
    # gamma chain
    "GAMMA_INPUT": 2.4, "gamma_out": 2.4, "scangamma": 2.4, "mask_gamma": 2.4,
    "gamma_c": 1.0, "gamma_c2": 1.0,
    # brightness
    "brightboost": 1.40, "brightboost1": 1.10, "glow": 0.08, "bloom": 0.0,
    "mask_bloom": 0.0, "bloom_dist": 0.0, "halation": 0.0, "bmask1": 0.0, "hmask1": 0.35,
    "clips": 0.0, "post_br": 1.0,
    # scanline
    "gsl": 0.0, "scanline1": 6.0, "scanline2": 8.0, "beam_min": 1.30, "beam_max": 1.00,
    "beam_size": 0.60, "scans_pragma": 0.50, "scans_effective": 0.75,  # code: 1.5*scans
    "scan_falloff": 1.0, "spike": 1.0, "tds": 0.0, "no_scanlines": 0.0, "rolling_scan": 0.0,
    # horizontal filter
    "h_sharp": 5.20, "s_sharp": 0.50, "ring": 0.0, "smart_ei": 0.0,
    # glow/bloom blur passes
    "SIZEH": 6.0, "SIGMA_H": 1.20, "SIZEV": 6.0, "SIGMA_V": 1.20, "FINE_GLOW": 1.0,
    "SIZEHB": 3.0, "SIGMA_HB": 0.75, "SIZEVB": 3.0, "SIGMA_VB": 0.60, "FINE_BLOOM": 1.0,
    "m_glow": 0.0,
    # mask
    "shadowMask": 0.0, "maskstr": 0.3, "mcut": 1.10, "maskboost": 1.0, "masksize": 1.0,
    "mask_zoom": 0.0, "mshift": 0.0, "mask_layout": 0.0, "maskDark": 0.5, "maskLight": 1.5,
    "slotmask": 0.0, "slotmask1": 0.0, "bmask": 0.0, "mclip": 0.0, "edgemask": 0.0,
    "smoothmask": 0.0, "smask_mit": 0.0,
    # post
    "pr_scan": 0.10,
    # interlacing / geometry
    "interm": 1.0, "inter": 375.0, "iscan": 0.20, "iscans": 0.25, "intres": 0.0,
    "warpX": 0.0, "warpY": 0.0, "IOS": 0.0, "OS": 1.0, "BLOOM_raster": 0.0,
    # afterglow / prepass
    "AS": 0.20, "agsat": 0.50, "PR": 0.32, "PG": 0.32, "PB": 0.32, "bth": 4.0,
    "TNTC": 0.0, "CP": 0.0, "CS": 0.0, "WP": 0.0, "wp_saturation": 1.0,
    "pre_bb": 1.0, "contr": 0.0, "pre_gc": 1.0, "BP": 0.0, "vigstr": 0.0,
    # deconvergence
    "deconrr": 0.0, "deconrg": 0.0, "deconrb": 0.0,
    "deconrry": 0.0, "deconrgy": 0.0, "deconrby": 0.0, "decons": 1.0,
    "addnoised": 0.0,
}

PROVENANCE = {
    "repo": "libretro/slang-shaders",
    "commit": "3b0d6aa1d134a168478cd9c904a866d969f8882b",
    "preset": "crt/crt-guest-advanced.slangp",
    "preset_overrides": "none",
}

GAMMA_IN = 2.4      # linearize GAMMA_INPUT; also gamma_in read back by later passes
GAMMA_OUT = 2.4

# ----------------------------------------------------------------------------- helpers

def _lin(x):
    """Linearize pass: c = pow(c, GAMMA_INPUT). Pre-pass is identity at defaults."""
    return max(x, 0.0) ** GAMMA_IN


def _beam_shape(d):
    """shape = mix(scanline1, scanline2, d):  6 + 2*d  for distance d in [0,1]."""
    return 6.0 + 2.0 * d


def _sw0_gray(d, E):
    """sw0() for a gray (fully unsaturated) pixel, gsl=0 path.

    vec3 sw0(float x, float color, float scanline, vec3 c) {
        float tmp = mix(beam_min, beam_max, color);   // 1.3 - 0.3*color
        vec3 sat = mix(1.0.xxx + scans, 1.0.xxx, c);  // c==1 for gray -> sat = 1
        float ex = x*tmp;
        ex = (gsl > -0.5) ? ex*ex : ...;              // gsl=0 -> ex = (x*tmp)^2
        return exp2(-scanline*ex*sat); }
    color argument = creff = (linear brightness)^scan_falloff = E at defaults.
    """
    tmp = 1.30 + (1.00 - 1.30) * E          # mix(beam_min, beam_max, E)
    ex = (d * tmp) ** 2
    return 2.0 ** (-_beam_shape(d) * ex)    # sat = 1 for gray


# ----------------------------------------------------------------------------- API

def beam_weight(d, L):
    """LINEAR-light contribution weight of one scanline at vertical distance d
    (in source lines, fractional ok) from the output sample, given the line's
    local ENCODED brightness L (0..1).

    The shader is a strict 2-tap vertical evaluator: only the two nearest lines
    (distances f and 1-f) ever contribute, so support is |d| <= 1; weight is 0
    beyond that.  Weight at d=0 is exactly 1.  The per-pixel joint
    normalization (sum of the two weights clamped to <= 1) is NOT applied here;
    it is applied inside ref_vertical().
    """
    d = abs(d)
    if d > 1.0:
        return 0.0
    E = _lin(L)
    return _sw0_gray(d, E)


def transfer(x):
    """End-to-end encoded->encoded transfer for a uniform field, sampled at the
    scanline beam CENTER (f=0), mask multiplier disabled, all defaults.
    Includes: gamma in (2.4), scanline normalization, brightboost + CGWG
    dark-compensate gain, glow flat-field contribution (glow=0.08, ordinary),
    pr_scan=0.10 post multiply, gamma out (1/2.4)."""
    return ref_vertical(0.0, x)


def ref_vertical(f, x, mask_mult=1.0):
    """Encoded (0..1) output at vertical fraction f within one scanline period
    (f=0 beam center, f=0.5 midway between lines) for a uniform encoded input
    field x, mask disabled (mask_mult=1), defaults otherwise.

    Chain implemented (uniform field => horizontal filter is unity):
      linearize:      E = x^2.4
      main pass:      w1 = sw0(f),  w2 = sw0(1-f)   [creff1=creff2=E for uniform]
                      W  = w1+w2, clamped: if max(w1+w2)>1 both scaled so sum==1
                      scangamma/gamma_in pows are identity (2.4/2.4)
                      output rgb = E*W, alpha = E (colmx of ctmp)
      deconvergence:  cm = maxrgb(color)^gamma_c = E*W ; colmx = max(alpha, cm) = E
                      w3 = min((max((cm-0.0005)*1.0005,0)+0.0001)/(colmx+0.0005),1)
                      mx = mxg^(1.40/gamma_in) with mxg = E  (uniform field)
                      mask apply in linear (mask_gamma==gamma_in): color *= cmask
                      dark_compensate dc = mix(max(clamp(mix(mcut,maskstr,mx),0,1)-1+fract(mwidth),0)+1, 1, mx)
                                       = 1 + (1-mx)*max(0.1-0.8*mx, 0)   [mwidth=2.0 -> fract=0]
                      bb = mix(1.40, 1.10, mx) * dc ; color *= bb ; clamp
                      glow (m_glow<0.5): Glow = mix(E, 0.25*color, colmx)
                                         color += 0.5*Glow*glow = 0.04*Glow ; clamp
                      pr_scan: color *= mix(1, mix(0.5*(1+w3), w3, mx), 0.10)
                      out = color^(1/2.4)
    """
    f = abs(f)
    E = _lin(x)

    # --- main CRT pass scanline weights (uniform field: creff1 = creff2 = E) ---
    w1 = _sw0_gray(f, E)          # line at distance f,   shape1 = mix(6,8,f)
    w2 = _sw0_gray(1.0 - f, E)    # line at distance 1-f, shape2 = mix(6,8,1-f)
    W = w1 + w2
    if W > 1.0:                   # "if (wf1 > 1.0) {wf1 = 1.0/wf1; w1*=wf1, w2*=wf1;}"
        W = 1.0

    color = E * W                 # gc() identity (gamma_c = 1)
    color = min(color, 1.0)
    alpha = E                     # FragColor.a = colmx = maxrgb(ctmp) = E

    # --- deconvergence pass ---
    cm = color                    # igc(maxrgb) with gamma_c=1
    colmx = max(alpha, cm)
    w3 = min((max((cm - 0.0005) * 1.0005, 0.0) + 0.0001) / (colmx + 0.0005), 1.0)

    mxg = max(alpha, cm)          # neighborhood alphas all == E on uniform field
    mx = mxg ** (1.40 / GAMMA_IN)

    # mask application (mask_gamma == gamma_in -> plain linear multiply)
    color = min(color * mask_mult, 1.0)

    # dark compensate (shadowMask=0 -> mwidths[0]=2.0 -> mask_compensate = 0.0)
    dc = (1.0 - mx) * max(0.1 - 0.8 * mx, 0.0) + 1.0
    bb = (1.40 + (1.10 - 1.40) * mx) * dc
    color = min(color * bb, 1.0)

    # bloom=0, halation=0, gamma_c2=1 (gc2 identity), smoothmask=0 -> skipped

    # ordinary glow, glow = 0.08 (GlowPass value on a uniform field == E)
    glow_v = E * (1.0 - colmx) + 0.25 * color * colmx   # mix(Glow, 0.25*color, colmx)
    color = min(color + 0.5 * 0.08 * glow_v, 1.0)

    # pr_scan = 0.10 scanline preservation
    pr = 1.0 + 0.10 * ((0.5 * (1.0 + w3)) * (1.0 - mx) + w3 * mx - 1.0)
    color *= pr

    # mclip=0 / bmask=0 / noise / humbar / corner / vignette / post_br -> identity
    color = min(max(color, 0.0), 1.0)
    return color ** (1.0 / GAMMA_OUT)


def h_kernel(frac):
    """Effective horizontal kernel at fractional offset frac (0..1), defaults.

    Main-pass filter (h_sharp=5.2, s_sharp=0.5, ring=0.0, fdivider=1 for 240p):
      zero  = exp2(-h_sharp); sharp1 = s_sharp*zero
      w(t)  = exp2(-h_sharp * t^2) with tap distances 2+f, 1+f, f, 1-f, 2-f, 3-f
      twl3 = max(wl3-sharp1, 0)
      twl2 = max(wl2-sharp1, -0.12*(1-f)^2)
      twl1 = max(wl1-sharp1, -0.12); twr1 = max(wr1-sharp1, -0.12)
      twr2 = max(wr2-sharp1, -0.12*f^2)
      twr3 = max(wr3-sharp1, 0)
      color = sum(tap*tw)/sum(tw)
    Returns [(offset, weight)] for offsets [-2,-1,0,1,2,3] relative to the
    floor() source pixel, normalized by the (possibly signed) weight sum,
    exactly as the shader does.  Kernel operates on LINEAR-light data.

    NOTE (approximation): with ring=0 the shader additionally clamps the result
    to [min,max] of the 4 inner taps (anti-ringing).  That nonlinearity cannot
    be expressed as a kernel; on a flat field it is inactive.  The glow/bloom
    gaussian blurs do NOT mix into the primary image at defaults except via the
    +0.04*Glow flat-field term already included in transfer()/ref_vertical(),
    so this kernel alone is the flat-field-consistent horizontal response.
    """
    fpx = frac
    h_sharp = 5.20
    sharp1 = 0.5 * 2.0 ** (-h_sharp)
    dists = [2.0 + fpx, 1.0 + fpx, fpx, 1.0 - fpx, 2.0 - fpx, 3.0 - fpx]
    w = [2.0 ** (-h_sharp * t * t) for t in dists]
    fp1 = 1.0 - fpx
    tw = [
        max(w[0] - sharp1, 0.0),
        max(w[1] - sharp1, -0.12 * fp1 * fp1),
        max(w[2] - sharp1, -0.12),
        max(w[3] - sharp1, -0.12),
        max(w[4] - sharp1, -0.12 * fpx * fpx),
        max(w[5] - sharp1, 0.0),
    ]
    s = sum(tw)
    offsets = [-2, -1, 0, 1, 2, 3]
    return [(o, t / s) for o, t in zip(offsets, tw)]


def mask_spec():
    """DEFAULT phosphor mask as rendered at 1080p output.

    shadowMask=0 (CGWG), maskstr=0.3, masksize=1, mask_zoom=0, mshift=0,
    slotmask off, maskboost=1, bmask=0.  Mask coordinate = integer OUTPUT pixel
    x (floor(gl_FragCoord.x)), period 2, no vertical variation:
        even x: (r,g,b) = (1.0, 0.7, 1.0)   [magenta]
        odd  x: (r,g,b) = (0.7, 1.0, 0.7)   [green]
    Applied in LINEAR light (mask_gamma 2.4 == gamma_in 2.4, so the
    pow(color, mask_gamma/gamma_in) sandwich is identity and the multiply
    happens directly on the linear beam color, before brightboost).
    """
    mc = 1.0 - 0.3   # 1.0 - max(maskstr, 0.0)
    tile = [[1.0, mc, 1.0], [mc, 1.0, mc]]
    return {
        "type": "CGWG magenta/green (shadowMask=0)",
        "tile_w_px": 2, "tile_h_px": 1,
        "linear_multipliers": tile,          # tile[x][channel], x = output px mod 2
        "applied_in": "linear light, before brightboost, before glow add",
        "avg_transmission_per_channel": [(1.0 + mc) / 2.0] * 3,
        "avg_transmission": (1.0 + mc) / 2.0,
        "encoded_equivalent_multipliers":    # value^(1/2.4) for encoded-space port
            [[m ** (1.0 / 2.4) for m in row] for row in tile],
        "notes": "masksize=1 -> 2 output px period at ANY output res incl. 1080p; "
                 "no brightness (mx) dependence for mask 0; maskboost=1 is identity.",
    }


def notes():
    return [
        "VERTICAL BEAM IS 2-TAP: only the two nearest source lines contribute (support |d|<=1). "
        "A MiSTer 4-tap V FIR can superset this; outer taps should fit ~0.",
        "Beam width is brightness-adaptive: exponent tmp = mix(1.30, 1.00, E_linear); dark lines "
        "narrower (trough weight 0.129 raw) than bright (0.297). Maps directly onto MiSTer "
        "ADAPTIVE vertical filter (two coefficient sets, brightness-interpolated).",
        "Per-pixel weight-sum clamp (w1+w2 scaled to <=1 when >1) is nonlinear; near f=0 it "
        "keeps the beam center at exactly 1.0 gain. A unity-DC FIR approximates this well "
        "since sum is within [0.59, 1.07] over f.",
        "'scans'=0.75 chroma term: for SATURATED colors the minority channels get sat up to 1.75 "
        "-> narrower beams per channel. MiSTer V filter is channel-uniform (driven by max RGB); "
        "NOT representable. Max effect: minority-channel trough weight 0.297->0.121 at full "
        "saturation white-level (encoded ~0.86x trough on minority channels).",
        "All filtering/scanline math runs in LINEAR light (gamma 2.4 in/out); MiSTer filters in "
        "encoded space. The V-FIR fit must target ref_vertical (already encoded->encoded). "
        "For the H kernel, fitting the encoded response of the linear-light kernel introduces "
        "edge-transition error (~ a few % on hard edges), unavoidable.",
        "Anti-ringing clamp on the horizontal filter (s_sharp=0.5 subtractive lobes are clamped "
        "to local min/max, ring=0): NOT representable; MiSTer FIR will ring by the negative-lobe "
        "amount (~1.8% of DC at frac=0.5) where the shader would clamp. Keep negative lobes.",
        "Glow (ordinary, strength 0.08): +0.04*mix(E,0.25*color,E) flat-field lift included in "
        "transfer(); its SPATIAL spread (gaussian sigma=1.2 src px H, sigma=1.2 lines V on a "
        "800x600 grid, radius 6) cannot be reproduced (no post-FIR blur on MiSTer). Magnitude: "
        "<=4% of full scale bleeding ~2-3 src px around bright areas.",
        "brightboost dark/bright gain bb = mix(1.40,1.10,E^0.583)*dark_compensate uses a "
        "NEIGHBORHOOD max (mx over +-1 output px and +-0.25 src px alphas); on flat fields it "
        "folds into transfer(); on edges it slightly brightens dark pixels next to bright ones. "
        "Not representable; small (<=e.g. gain diff 1.4 vs 1.1 confined to 1px transitions).",
        "pr_scan=0.10 'preserve scanline properties' post-multiply is included in ref_vertical; "
        "it deepens troughs by up to ~5% and is representable inside the V-FIR fit.",
        "Afterglow (AS=0.20): temporal phosphor decay on pixels darker than ~3% encoded; "
        "static-field magnitude <=0.001 linear. NOT representable (temporal). Negligible.",
        "avg-lum raster bloom: BLOOM=0 at defaults -> geometry static, only feeds OS overscan "
        "factor = 1.0. No effect.",
        "Interlace handling (interm=1, trigger 375 lines): 480i+ content gets field-blended "
        "no-scanline path with iscan=0.20 darkening. MiSTer handles interlace in the framework; "
        "not portable, out of scope for 240p/224p targets.",
        "Deconvergence/noise/curvature/slotmask/humbar/vignette/corner: all OFF at defaults.",
        "Mask is CGWG magenta/green strength 0.3, 2-output-px period, applied in linear light "
        "BEFORE brightboost and glow: on MiSTer the mask stage is post-scaler on ENCODED data, "
        "so use encoded multipliers 0.7^(1/2.4)=0.8619 (~14/16=0.875 in v2 1/16 steps) and "
        "accept that glow (4%) and the bb gain ordering differ slightly (<1.5% local error).",
        "Main pass renders at viewport; mask at output-pixel granularity -> at 1080p the "
        "magenta/green stripe pair is 2px; user's 4K TV 2x scale keeps it alignment-safe.",
    ]


# ----------------------------------------------------------------------------- validation

def _validate(verbose=True):
    """Hand-computed cross-checks against the quoted shader code."""
    checks = []

    # 1) transfer(1.0): E=1, bb=1.1 -> clamp 1; glow 0.25*1*1*0.04=0.01 -> clamp 1.
    # w3 epsilon formula gives 0.99960 at white -> pr = 0.999960 -> out = 0.999983
    # (the real shader is 0.0017% below 1.0 at white; this is faithful, not a bug)
    checks.append(("transfer(1.0)", transfer(1.0), 1.0, 2e-5))

    # 2) transfer(0.5) hand calc:
    # E=0.5^2.4=0.189465; mx=E^(1.4/2.4)=0.378929; dc=1 (0.1-0.8mx<0); bb=1.4-0.3*mx=1.286321
    # c1=E*bb=0.243714; glow=E*(1-E)+0.25*c1*E=0.153574+0.011543=0.165117 -> c2=c1+0.0066047=0.250319
    # w3~1 -> pr=1; out=0.250319^(1/2.4)=0.561523
    E = 0.5 ** 2.4
    mx = E ** (1.4 / 2.4)
    c1 = E * (1.4 - 0.3 * mx)
    g = E * (1 - E) + 0.25 * c1 * E
    c2 = c1 + 0.04 * g
    # w3 at f=0: cm=E (W=1), w3=min(((E-0.0005)*1.0005+0.0001)/(E+0.0005),1)
    w3 = min((max((E - 0.0005) * 1.0005, 0) + 0.0001) / (E + 0.0005), 1.0)
    pr = 1 + 0.1 * ((0.5 * (1 + w3)) * (1 - mx) + w3 * mx - 1)
    expect = (c2 * pr) ** (1 / 2.4)
    checks.append(("transfer(0.5)", transfer(0.5), expect, 1e-12))
    checks.append(("transfer(0.5)~hand", transfer(0.5), 0.5613, 5e-4))

    # 3) beam_weight(0.5, 1.0): tmp=1.0, shape=7, 2^(-7*0.25)=2^-1.75=0.297302
    checks.append(("beam_weight(0.5,1.0)", beam_weight(0.5, 1.0), 2 ** -1.75, 1e-12))

    # 4) beam_weight(0.5, 0.0): tmp=1.3, 2^(-7*(0.65)^2)=2^-2.9575=0.128762
    checks.append(("beam_weight(0.5,0.0)", beam_weight(0.5, 0.0),
                   2 ** (-7 * 0.65 ** 2), 1e-12))

    # 5) ref_vertical(0.5, 1.0) hand: W=2*2^-1.75=0.594604; cm=W; w3=(W-.0005)*1.0005+.0001)/1.0005
    W = 2 * 2 ** -1.75
    w3 = min((max((W - 0.0005) * 1.0005, 0) + 0.0001) / (1.0 + 0.0005), 1.0)
    c = min(W * 1.1, 1.0)
    gl = 1.0 * 0.0 + 0.25 * c * 1.0
    c = min(c + 0.04 * gl, 1.0)
    pr = 1 + 0.1 * (w3 - 1)      # mx=1
    expect = (c * pr) ** (1 / 2.4)
    checks.append(("ref_vertical(0.5,1.0)", ref_vertical(0.5, 1.0), expect, 1e-12))
    checks.append(("ref_vertical(0.5,1.0)~hand", ref_vertical(0.5, 1.0), 0.8271, 5e-4))

    # 6) h_kernel(0.0): center = (1-sharp1)/S, sides=(2^-5.2-sharp1)/S, outer=0
    z = 2 ** -5.2
    s1 = 0.5 * z
    S = (1 - s1) + 2 * (z - s1)
    k = dict(h_kernel(0.0))
    checks.append(("h_kernel(0)[0]", k[0], (1 - s1) / S, 1e-12))
    checks.append(("h_kernel(0)[-1]", k[-1], (z - s1) / S, 1e-12))
    checks.append(("h_kernel(0)[2]", k[2], 0.0, 1e-12))

    # 7) h_kernel(0.5): inner=2^-1.3-s1, outer2 = max(2^-11.7-s1, -0.03)
    wi = 2 ** (-5.2 * 0.25) - s1
    wo = max(2 ** (-5.2 * 2.25) - s1, -0.12 * 0.25)
    S = 2 * wi + 2 * wo
    k = dict(h_kernel(0.5))
    checks.append(("h_kernel(0.5)[0]", k[0], wi / S, 1e-12))
    checks.append(("h_kernel(0.5)[-1]", k[-1], wo / S, 1e-12))

    # 8) mask spec transmission
    ms = mask_spec()
    checks.append(("mask avg", ms["avg_transmission"], 0.85, 1e-12))

    ok = True
    for name, got, exp, tol in checks:
        good = abs(got - exp) <= tol
        ok &= good
        if verbose:
            print(f"{'PASS' if good else 'FAIL'}  {name}: got={got!r} expected={exp!r}")
    return ok


# ----------------------------------------------------------------------------- grid dump

def dump_targets(path):
    import json
    grid = {
        "provenance": PROVENANCE,
        "defaults": DEFAULTS,
        "ref_vertical": {
            "f_axis": [i / 64.0 for i in range(33)],          # 0 .. 0.5
            "x_axis": [i / 32.0 for i in range(33)],          # 0 .. 1
            "values": [[ref_vertical(i / 64.0, j / 32.0) for j in range(33)]
                       for i in range(33)],                   # values[f][x]
        },
        "beam_weight": {
            "d_axis": [i / 32.0 for i in range(65)],          # 0 .. 2
            "L_axis": [i / 8.0 for i in range(9)],            # 0 .. 1
            "values": [[beam_weight(i / 32.0, j / 8.0) for j in range(9)]
                       for i in range(65)],                   # values[d][L]
        },
        "h_kernel": {
            "frac_axis": [i / 32.0 for i in range(33)],       # 0 .. 1
            "tap_offsets": [-2, -1, 0, 1, 2, 3],
            "values": [[w for _, w in h_kernel(i / 32.0)] for i in range(33)],
        },
        "transfer": {
            "x_axis": [i / 255.0 for i in range(256)],
            "values": [transfer(i / 255.0) for i in range(256)],
        },
        "mask_spec": mask_spec(),
        "notes": notes(),
    }
    with open(path, "w") as fh:
        json.dump(grid, fh, indent=1)
    return path


if __name__ == "__main__":
    import os, sys
    ok = _validate()
    print("validation:", "OK" if ok else "FAILED")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "guest_advanced_targets.json")
    dump_targets(out)
    print("wrote", out)
    sys.exit(0 if ok else 1)
