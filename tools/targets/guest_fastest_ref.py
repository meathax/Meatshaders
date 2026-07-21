"""
Reference implementation of CRT-GUEST-ADVANCED-FASTEST (guest.r) at DEFAULT parameters.

Source: crt-guest-advanced-2026-07-12-release1 (guest.r upstream release drop,
        D:/Downloads/crt-guest-advanced-2026-07-12-release1)
Preset: crt-guest-advanced-fastest.slangp  (sets NO parameter overrides)

Pass chain (5 passes):
  0 perf-pass(PerfPass), 1 pre-shadersf(PrePassDontChange),
  2 linearizef(LinearizePass, float FB), 3 crt-guest-advanced-pass1f
  (viewport-x/source-y H filter), 4 crt-guest-advanced-pass2f (viewport:
  scanlines + mask + brightboost + gamma out, single final pass)

DEFAULTS that matter:
  GAMMA_INPUT=2.4, gamma_out=2.4 -> all spatial math in LINEAR light
  gsl=0 (sw0), scanline1=6.0, scanline2=8.0, beam_min=1.30, beam_max=1.00,
  beam_size=0.60, scans=0.50 (code uses 1.5*scans=0.75), scan_falloff=1.0
  brightboost(dark)=1.40, brightboost1(bright)=1.10, gamma_c=1.0
  h_sharp=5.20, s_sharp=0.50, ring=0.0, spike=1.0 (pass1f, 4-tap window)
  shadowMask=0 (CGWG magenta/green), maskstr=0.3, masksize=1, mask_zoom=0
  vertmask=0, DER/DEG/DEB=0 (DES=0.7 has no effect at zero offsets)
  bloom=0, halation=0, mask_bloom=0, clips=0, post_br=1.0, warpX=warpY=0
  interm=1, inter=400 (linearizef); intres=0

DIFFERENCES from the advanced/fast chain (all verified in source):
  - NO glow at all: pass2f has no glow parameter and no GlowPass texture;
    the advanced +0.5*glow*mix(E, 0.25*color, colmx) flat-field lift is absent.
  - NO pr_scan ("preserve scanline properties") post multiply.
  - NO w3-epsilon dimming: transfer(1.0) is exactly 1.0.
  - NO dark_compensate expression: bb = mix(brightboost*mask_compensate,
    brightboost1, mx) with mask_compensate = 1.0 for shadowMask <= 4.5,
    i.e. bb = mix(1.40, 1.10, mx) -- numerically the same bb as the advanced
    chain's (whose dark_compensate is identically 1.0 at defaults).
  - mx derives from sctmp = max(scanline-window alpha average, ctmp) instead
    of a 3-texel neighborhood max (equal on uniform fields).
  - mask multiply is a plain linear-light product (no mask_gamma sandwich;
    equivalent because advanced's sandwich is identity at 2.4/2.4).
  - pass1f uses the literal 4-tap horizontal window; identical numbers to the
    advanced 6-tap code whose outer taps clamp to 0 at default h_sharp.
  - corner()/post_br are exactly 1.0 in the interior at defaults.

For a 240p/224p source linearizef emits intera=1.0 (inter trigger 400), so
interb=false -> the full scanline path below is active.

All functions are pure; only `math` is required.
"""

import math

DEFAULTS = {
    "GAMMA_INPUT": 2.4, "gamma_out": 2.4, "gamma_c": 1.0,
    "brightboost": 1.40, "brightboost1": 1.10,
    "gsl": 0.0, "scanline1": 6.0, "scanline2": 8.0, "beam_min": 1.30,
    "beam_max": 1.00, "beam_size": 0.60, "scans_pragma": 0.50,
    "scans_effective": 0.75, "scan_falloff": 1.0, "no_scanlines": 0.0,
    "h_sharp": 5.20, "s_sharp": 0.50, "ring": 0.0, "spike": 1.0,
    "shadowMask": 0.0, "maskstr": 0.3, "mcut": 1.10, "maskboost": 1.0,
    "masksize": 1.0, "mask_zoom": 0.0, "mshift": 0.0, "mask_layout": 0.0,
    "slotmask": 0.0, "slotmask1": 0.0, "smask_mit": 0.0,
    "vertmask": 0.0, "DER": 0.0, "DEG": 0.0, "DEB": 0.0, "DES": 0.7,
    "bloom": 0.0, "bloom_size": 0.75, "bloom_dist": 0.0, "halation": 0.0,
    "hmask1": 0.5, "mask_bloom": 0.0, "clips": 0.0, "post_br": 1.0,
    "interm": 1.0, "inter": 400.0, "iscan": 0.20, "inters": 0.0, "intres": 0.0,
    "warpX": 0.0, "warpY": 0.0, "IOS": 0.0, "csize": 0.0, "bsize": 400.0,
    # glow / pr_scan / scangamma / mask_gamma do not exist in this chain
}

PROVENANCE = {
    "source": "crt-guest-advanced-2026-07-12-release1 (guest.r release drop)",
    "preset": "crt-guest-advanced-fastest.slangp",
    "preset_overrides": "none",
}

GAMMA_IN = 2.4
GAMMA_OUT = 2.4

# ----------------------------------------------------------------------------- helpers

def _lin(x):
    """linearizef: c = pow(c, GAMMA_INPUT). pre-shadersf is identity at defaults."""
    return max(x, 0.0) ** GAMMA_IN


def _beam_shape(d):
    """shape = mix(scanline1, scanline2, d): 6 + 2*d for distance d in [0,1]."""
    return 6.0 + 2.0 * d


def _sw0_gray(d, E):
    """sw0() for a gray pixel, gsl=0: exp2(-shape * (d*mix(1.3,1.0,E))^2)."""
    tmp = 1.30 + (1.00 - 1.30) * E
    ex = (d * tmp) ** 2
    return 2.0 ** (-_beam_shape(d) * ex)


# ----------------------------------------------------------------------------- API

def beam_weight(d, L):
    """LINEAR-light weight of one scanline at distance d for encoded level L.

    Identical sw0 evaluator to the advanced chain: strict 2-tap support
    (|d| <= 1), weight 1 at d=0, joint sum-clamp applied in ref_vertical().
    """
    d = abs(d)
    if d > 1.0:
        return 0.0
    E = _lin(L)
    return _sw0_gray(d, E)


def transfer(x):
    """Encoded->encoded uniform-field transfer at beam centre (f=0), mask off.

    Includes: gamma in 2.4, scanline weight-sum clamp, bb = mix(1.40,1.10,mx)
    (mask_compensate=1 for CGWG), final clamp, gamma out 1/2.4.  There is NO
    glow lift, NO pr_scan multiply and NO w3-epsilon dimming in this chain:
    transfer(1.0) == 1.0 exactly.
    """
    return ref_vertical(0.0, x)


def ref_vertical(f, x, mask_mult=1.0):
    """Encoded (0..1) output at vertical fraction f for uniform encoded x.

    Chain (uniform field => pass1f horizontal filter is unity):
      linearizef:  E = x^2.4
      pass2f:      w1 = sw0(f), w2 = sw0(1-f)   [creff1=creff2=E uniform]
                   if max(w1+w2) > 1: both scaled so the sum == 1
                   color = E*(w1+w2), min 1     [cd1=cd2=1: vertmask=0]
                   gc() identity (gamma_c=1)
                   mx = sctmp^(1.4/2.4) with sctmp = E on a uniform field
                   mask: color *= cmask (linear light), min 1
                   bb = mix(1.40*1.0, 1.10, mx); color *= bb
                   bloom/halation off; clamp; out = color^(1/2.4)
                   corner0 = post_br = 1 in the interior
    """
    f = abs(f)
    E = _lin(x)

    w1 = _sw0_gray(f, E)
    w2 = _sw0_gray(1.0 - f, E)
    W = w1 + w2
    if W > 1.0:                   # "if (wf1 > 1.0) {wf1 = 1.0/wf1; w1*=wf1, w2*=wf1;}"
        W = 1.0

    color = E * W
    color = min(color, 1.0)

    # mx: mcolor = sctmp = max(scolor0/(wt1+wt2), ctmp) = E on a uniform field
    mx = E ** (1.40 / GAMMA_IN)

    # mask multiply in linear light, before brightboost
    color = min(color * mask_mult, 1.0)

    # bb = mix(brightboost*mask_compensate, brightboost1, mx); CGWG -> 1.0
    bb = 1.40 + (1.10 - 1.40) * mx
    color = min(max(color * bb, 0.0), 1.0)

    return color ** (1.0 / GAMMA_OUT)


def h_kernel(frac):
    """Horizontal kernel of pass1f at fractional offset frac (0..1), defaults.

    Literal 4-tap window (h_sharp=5.2, s_sharp=0.5, ring=0):
      zero  = exp2(-h_sharp); sharp1 = s_sharp*zero
      w(t)  = exp2(-h_sharp*t^2) at tap distances 1+f, f, 1-f, 2-f
      twl2 = max(wl2-sharp1, -0.12*(1-f)^2)
      twl1 = max(wl1-sharp1, -0.12); twr1 = max(twr1-sharp1, -0.12)
      twr2 = max(wr2-sharp1, -0.12*f^2)
      color = sum(tap*tw)/sum(tw)
    Returns [(offset, weight)] for offsets [-1, 0, 1, 2] relative to the
    floor() source pixel.  Numerically identical to the advanced 6-tap
    kernel, whose outer taps always clamp to 0 at default h_sharp.  Kernel
    operates on LINEAR-light data; ring=0 additionally clamps the result to
    the [min,max] of the 4 taps (anti-ringing, not representable as a FIR).
    """
    fpx = frac
    h_sharp = 5.20
    sharp1 = 0.5 * 2.0 ** (-h_sharp)
    dists = [1.0 + fpx, fpx, 1.0 - fpx, 2.0 - fpx]
    w = [2.0 ** (-h_sharp * t * t) for t in dists]
    fp1 = 1.0 - fpx
    tw = [
        max(w[0] - sharp1, -0.12 * fp1 * fp1),
        max(w[1] - sharp1, -0.12),
        max(w[2] - sharp1, -0.12),
        max(w[3] - sharp1, -0.12 * fpx * fpx),
    ]
    s = sum(tw)
    offsets = [-1, 0, 1, 2]
    return [(o, t / s) for o, t in zip(offsets, tw)]


def mask_spec():
    """DEFAULT phosphor mask as rendered at 1080p output.

    shadowMask=0 (CGWG), maskstr=0.3, masksize=1, mask_zoom=0, mshift=0,
    slotmask off.  Mask coordinate = integer OUTPUT pixel x
    (floor(gl_FragCoord.xy * 1.00001)), period 2, no vertical variation:
        even x: (r,g,b) = (1.0, 0.7, 1.0)   [magenta]
        odd  x: (r,g,b) = (0.7, 1.0, 0.7)   [green]
    Applied as a plain LINEAR-light multiply before brightboost (pass2f has
    no mask_gamma sandwich), encoded afterwards with gamma_out = 2.4.
    """
    mc = 1.0 - 0.3   # 1.0 - max(maskstr, 0.0)
    tile = [[1.0, mc, 1.0], [mc, 1.0, mc]]
    return {
        "type": "CGWG magenta/green (shadowMask=0)",
        "tile_w_px": 2, "tile_h_px": 1,
        "linear_multipliers": tile,
        "applied_in": "linear light, before brightboost; no glow stage exists",
        "avg_transmission_per_channel": [(1.0 + mc) / 2.0] * 3,
        "avg_transmission": (1.0 + mc) / 2.0,
        "encoded_equivalent_multipliers":
            [[m ** (1.0 / 2.4) for m in row] for row in tile],
        "notes": "masksize=1 -> 2 output px period at any output res incl. "
                 "1080p; maskboost=1 identity; same tile as the advanced chain.",
    }


def notes():
    return [
        "VERTICAL BEAM IS 2-TAP with the identical sw0 evaluator as the advanced "
        "chain; the V fit differs only through the missing glow/pr_scan/w3 terms.",
        "NO GLOW: the advanced +0.04 flat-field lift is absent; peak white maps to "
        "exactly 255 and the whole transfer sits slightly below the advanced one.",
        "NO pr_scan: troughs are not deepened by the 0.10 preserve-scanlines "
        "multiply; the beam profile through V is marginally shallower.",
        "bb = mix(1.40, 1.10, mx) directly; the advanced dark_compensate expression "
        "does not exist here (it evaluates to 1.0 there anyway at defaults).",
        "Mask, mask strength, mask placement and gamma chain (2.4 in / 2.4 out) are "
        "identical to the advanced chain; encoded multipliers 0.7^(1/2.4)=0.8619.",
        "'scans'=0.75 chroma term present, as in advanced: per-channel beams for "
        "saturated colors are NOT representable with MiSTer's shared control.",
        "Anti-ringing clamp on the H filter (ring=0) is NOT representable; keep "
        "negative lobes and accept ~1.8% ring on hard edges (same as advanced).",
        "vertmask/deconvergence/bloom/halation/noise/curvature all OFF at defaults.",
    ]


# ----------------------------------------------------------------------------- validation

def _validate(verbose=True):
    """Hand-computed cross-checks against the quoted shader code."""
    checks = []

    # 1) transfer(1.0): E=1 -> W clamps to 1; mx=1 -> bb=1.1 -> clamp 1;
    # no glow, no pr_scan, no w3 dimming -> EXACTLY 1.0
    checks.append(("transfer(1.0)", transfer(1.0), 1.0, 0.0))

    # 2) transfer(0.5): E=0.5^2.4; mx=E^(1.4/2.4)=0.5^1.4; bb=1.4-0.3*mx;
    # out=(E*bb)^(1/2.4)
    E = 0.5 ** 2.4
    mx = 0.5 ** 1.4
    expect = (E * (1.4 - 0.3 * mx)) ** (1 / 2.4)
    checks.append(("transfer(0.5)", transfer(0.5), expect, 1e-12))
    checks.append(("transfer(0.5)~hand", transfer(0.5), 0.5553, 5e-4))

    # 3) beam_weight identical to advanced sw0
    checks.append(("beam_weight(0.5,1.0)", beam_weight(0.5, 1.0), 2 ** -1.75, 1e-12))
    checks.append(("beam_weight(0.5,0.0)", beam_weight(0.5, 0.0),
                   2 ** (-7 * 0.65 ** 2), 1e-12))

    # 4) ref_vertical(0.5, 1.0): W=2*2^-1.75=0.594604 (<1, no clamp);
    # c=min(W*1.1,1)=0.654065; out=c^(1/2.4)
    W = 2 * 2 ** -1.75
    expect = min(W * 1.1, 1.0) ** (1 / 2.4)
    checks.append(("ref_vertical(0.5,1.0)", ref_vertical(0.5, 1.0), expect, 1e-12))

    # 5) h_kernel(0.0): center=(1-s1)/S, sides=(z-s1)/S and max(2^-20.8-s1, 0)=0
    z = 2 ** -5.2
    s1 = 0.5 * z
    S = (1 - s1) + 2 * (z - s1)  # taps at dist 1 both sides; twr2(dist 2) = 0
    k = dict(h_kernel(0.0))
    checks.append(("h_kernel(0)[0]", k[0], (1 - s1) / S, 1e-12))
    checks.append(("h_kernel(0)[-1]", k[-1], (z - s1) / S, 1e-12))
    checks.append(("h_kernel(0)[2]", k[2], 0.0, 1e-12))

    # 6) h_kernel(0.5): inner pair 2^-1.3-s1; outer pair max(2^-11.7-s1, -0.03)
    wi = 2 ** (-5.2 * 0.25) - s1
    wo = max(2 ** (-5.2 * 2.25) - s1, -0.12 * 0.25)
    S = 2 * wi + 2 * wo
    k = dict(h_kernel(0.5))
    checks.append(("h_kernel(0.5)[0]", k[0], wi / S, 1e-12))
    checks.append(("h_kernel(0.5)[-1]", k[-1], wo / S, 1e-12))
    checks.append(("h_kernel(0.5)[2]", k[2], wo / S, 1e-12))

    # 7) mask spec transmission
    ms = mask_spec()
    checks.append(("mask avg", ms["avg_transmission"], 0.85, 1e-12))

    # 8) parity with the advanced module where the chains coincide:
    # beam weights identical for all (d, L)
    try:
        import guest_advanced_ref as adv
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            for L in (0.0, 0.5, 1.0):
                checks.append((f"beam=={d},{L}", beam_weight(d, L),
                               adv.beam_weight(d, L), 0.0))
        for fr in (0.0, 0.25, 0.5):
            ka, kb = dict(h_kernel(fr)), dict(adv.h_kernel(fr))
            for off in ka:
                checks.append((f"hk=={fr}[{off}]", ka[off], kb[off], 1e-12))
    except ImportError:
        pass

    ok = True
    for name, got, exp, tol in checks:
        good = abs(got - exp) <= tol
        ok &= good
        if verbose:
            print(f"{'PASS' if good else 'FAIL'}  {name}: got={got!r} expected={exp!r}")
    return ok


if __name__ == "__main__":
    import sys
    ok = _validate()
    print("validation:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
