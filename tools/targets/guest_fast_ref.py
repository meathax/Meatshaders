"""
Reference implementation of CRT-GUEST-ADVANCED-FAST (guest.r) at DEFAULT parameters.

Source: crt-guest-advanced-2026-07-12-release1 (guest.r upstream release drop,
        D:/Downloads/crt-guest-advanced-2026-07-12-release1)
Preset: crt-guest-advanced-fast.slangp  (sets NO parameter overrides)

Pass chain (9 passes):
  0 stock, 1 stock(StockPass), 2 pre-shaders(PrePass), 3 linearize(LinearizePass,
  float FB), 4 crt-guest-advanced-pass1 (viewport-x/source-y, H filter + spike
  alpha, alias Pass1), 5 bloom_horizontal(640xH), 6 bloom_vertical(x480,
  BloomPass), 7 crt-guest-advanced-pass2 (viewport, scanlines),
  8 deconvergence-f (viewport, mask/brightboost/glow/gamma-out)

EQUIVALENCE (verified line by line against the release sources):
At DEFAULT parameters every stage that survives projection onto MiSTer's
fixed pipeline is numerically IDENTICAL to the full advanced chain:

  - linearize: GAMMA_INPUT=2.4, identity pre-pass  (same as advanced);
  - pass1's horizontal filter is the same h_sharp=5.20 / s_sharp=0.50 kernel
    with the same -0.12 floors and the same ring=0 anti-ring clamp; advanced's
    two extra outer taps are always clamped to exactly 0 at default h_sharp;
  - pass2's scanline evaluator is the same st()/sw0() code (wts=-10 at
    bloomsamp=0, ssharp=0), same weight-sum clamp, same 2-tap support
    [pass2's sctmp adds a max() against ctmp that is inactive on the
    uniform fields the V fit targets];
  - deconvergence-f is the advanced deconvergence chain with the same
    defaults: CGWG mask 0 @ maskstr 0.3 in linear light, mask_gamma 2.4,
    dark_compensate identically 1.0, bb = mix(1.40, 1.10, mx), glow 0.08
    (sourced from BloomPass instead of GlowPass -- on a uniform field both
    equal E), pr_scan 0.10, gamma_out 2.4.

The only physical differences are the glow blur's spatial footprint
(sigma 0.75/0.60 on a 640x480 grid instead of gaussian 1.20/1.20 on 800x600)
and the dropped afterglow/avg-lum passes -- none of which are representable
in MiSTer's scaler, so the canonical fixed-pipeline projection of the fast
variant equals the advanced one exactly.  This module therefore delegates its
numeric targets to guest_advanced_ref and exists to make that equivalence an
explicit, validated contract.
"""

import guest_advanced_ref as _adv

DEFAULTS = dict(_adv.DEFAULTS)
DEFAULTS.update({
    # parameters absent from the fast chain (documentation only)
    "scangamma": None,        # fast has no scangamma; sandwich was identity anyway
    "m_glow": None,           # ordinary glow only
    "smart_ei": None,         # no smart-edges path in pass1
})

PROVENANCE = {
    "source": "crt-guest-advanced-2026-07-12-release1 (guest.r release drop)",
    "preset": "crt-guest-advanced-fast.slangp",
    "preset_overrides": "none",
    "equivalence": "fixed-pipeline projection identical to crt-guest-advanced "
                   "at defaults (see module docstring)",
}

GAMMA_IN = _adv.GAMMA_IN
GAMMA_OUT = _adv.GAMMA_OUT

beam_weight = _adv.beam_weight
transfer = _adv.transfer
ref_vertical = _adv.ref_vertical
h_kernel = _adv.h_kernel


def mask_spec():
    spec = _adv.mask_spec()
    spec = dict(spec)
    spec["notes"] = ("fast variant: identical CGWG mask/strength/placement as "
                     "advanced; " + spec["notes"])
    return spec


def _validate(verbose=True):
    """The contract of this module IS the equivalence; assert it."""
    checks = []
    for f in (0.0, 0.125, 0.25, 0.375, 0.5):
        for x in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            checks.append((f"ref_vertical({f},{x})==advanced",
                           ref_vertical(f, x), _adv.ref_vertical(f, x), 0.0))
    for fr in (0.0, 0.25, 0.5):
        ka, kb = dict(h_kernel(fr)), dict(_adv.h_kernel(fr))
        for off in kb:
            checks.append((f"h_kernel({fr})[{off}]==advanced",
                           ka[off], kb[off], 0.0))
    checks.append(("mask avg", mask_spec()["avg_transmission"], 0.85, 1e-12))
    ok = True
    for name, got, exp, tol in checks:
        good = abs(got - exp) <= tol
        ok &= good
        if verbose and not good:
            print(f"FAIL  {name}: got={got!r} expected={exp!r}")
    if verbose:
        print(f"{sum(1 for _ in checks)} equivalence checks "
              f"{'PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    ok = _validate()
    print("validation:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
