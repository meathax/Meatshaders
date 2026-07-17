# MiSTer CRT port design — exact-hardware edition

Date: 2026-07-17. This pack covers CRT Easymode, CRT Easymode v5,
CRT Guest Advanced, CRT Lottes, CRT Royale, and CRT Royale Kurozumi.
The five executable reference models are vendored in `tools/targets/` and pin
their source math to libretro/slang-shaders commit
`3b0d6aa1d134a168478cd9c904a866d969f8882b`.

These are MiSTer fixed-pipeline ports, not programmable shaders. The goal is
the closest deterministic output MiSTer's gamma LUT, separable four-tap scaler,
single adaptive-control path, and v2 shadow-mask tokens can represent.

## Hardware model

`mister_model.py` follows current Main RTL rather than an ideal FIR surrogate:

- scaler phase uses the upper eight fractional bits, `frac12 >> 4`; there is no
  half-up rounding or carry into the integer source coordinate;
- adaptive coefficients are interpolated as
  `(A*(256-lum) + B*lum) >> 1`; odd signed 3.15 results are valid;
- FIR pair sums truncate before the final sum and each scaler axis clamps to
  eight bits;
- adaptive control is max RGB from the H-filtered nearest source line and
  switches that line at half phase;
- mask multiplication is the saturating sum of independently truncated shifts;
- mask mode 2x repeats every token into a 2x2 output-pixel block.

Every stored table has 256 phases, signed 10-bit coefficients, exact conjugate
phase symmetry, and a self-symmetric phase 128.

## Fitting architecture

1. **Horizontal table.** `fit_h` folds the source kernel into MiSTer's four-tap
   window and preserves DC. The legacy Pixel-Art Anti-Ring names now alias the
   canonical Easymode tables: exact saturation/clamp testing proved the former
   surrogate fit regressed binary RMS by about 8.6x.

2. **Gamma/control warp plus adaptive V.** Guest, Royale, and Kurozumi use a
   monotone LUT warp co-optimized with the two V endpoint row sums in output
   space. The LUT is not claimed to be the shader's beam-centre transfer.
   Endpoint rows may exceed unity where their blend weight makes that safe;
   clipping is measured on the actual blended pipeline, never inferred from a
   row-sum proxy. Kurozumi uses a true RGB LUT and channel-aware refinement so
   its warm lifted black is retained.

3. **Gain split.** Easymode v5 carries the pre-clip transfer divided by 1.105 in
   the LUT and the remaining gain in its mask. This keeps the adaptive control
   resolving highlight levels and moves saturation to the same late stage as
   the source shader.

4. **Exact refinement.** LUT entries are refined through the real H/V arithmetic.
   Mask-aware refinement scores the full periodic mask domain, not an upper-left
   crop. Every refinement is self-gated and is adopted only when end-to-end
   error falls without violating clipping or monotonicity.

5. **Fixed Compatibility.** Each generated family has a dedicated rank-1 V
   table, jointly fitted LUT, and pixel-local mask. This is a complete calibrated
   pipeline for non-adaptive cores, not merely the adaptive dark endpoint.

6. **No Gamma.** Each generated family also has a dedicated adaptive V table
   and mask fitted with an identity LUT. The preset selects those assets and
   `gamma=off`; it does not silently reuse a nonlinear-LUT calibration.

Lottes is retained: its adaptive and fixed paths measure 1.260 and 1.166 codes
RMS with maximum error below 4.6 codes. Rebuilding it did not produce a material
gain, so changing its known-good coefficients would add risk without fidelity.

## Shadow masks

Mask targets use each shader's real output encoding and are evaluated with the
exact shift/add token arithmetic. If hardware and reference periods differ,
both fitting and acceptance use the complete
`lcm(hardware period, reference period)` supercell. The best rigid reference
roll is reported for visual equivalence; strict origin is reported separately.

- Guest's CGWG target maps closely to its 2x1 hardware tile.
- Royale retains slot structure with a 12-row x 6-column verbatim slice of the
  24x24 rendered source tile. The earlier cropped score was invalid because it
  ignored reference cells outside one hardware tile.
- Kurozumi Default uses a 1-row x 2-column **pixel-local** jointly fitted mask.
  The source's desired green/blue values cannot coexist in one v2 token, so
  exact identity is impossible. `Perceptual Dither` instead uses a 1x4 token
  pattern whose partner-pixel mean is close to the PVM target (2.99-code
  isolated pair-average error), while its raw individual-pixel pipeline error
  is explicitly reported as 30.901 RMS with a 118.3-code maximum.
- Easymode v5's mask carries the 1.105 gain split.

## Pattern and display variants

The scaler exposes one source-dependent discontinuity no flat-field fit can
remove: adaptive control samples the nearest line and switches at half phase.
Default remains the closest source-flat response. Optional `Edge Stable`
profiles bound that selector-only jump for hard text/sprite edges while also
passing the fractional-scale moire guard:

| Family | Default jump | Edge Stable jump | Edge RMS | Moire |
|---|---:|---:|---:|---:|
| Guest Advanced | 23 | 0 | 1.763 | 1.78 |
| Royale | 112 | 16 | 28.029 | 5.31 |
| Easymode v5 | 40 | 12 | 8.818 | 2.96 |

Royale needs a clip-safe one-sided dark-to-bright blend; a symmetric blend
imports legal high-gain dark rows into highlights and clips. Kurozumi already
has a zero-code selector jump, and Lottes has three, so neither needs the variant.

Moiré is measured at 4.5x and 4.821x vertical scale across the gray ramp, with
a 7.65-code peak/trough standard-deviation ceiling. Current Default results are
1.71 / 5.31 / 6.82 / 2.24 / 1.38 for Guest, Royale, Kurozumi, Easymode v5 and
Lottes, so every Default passes. Kurozumi's `Anti-Moire` remains an optional
wider-beam profile at 2.07 for extra fractional-scale stability; integer
vertical scaling also avoids fractional beat.

## Preset axis contracts

Landscape presets select `_H` on H, an appropriate `_V` table on V, and the
family's explicit No-Scanlines interlace fallback. TATE transposes the source
axes: the adaptive source-V table occupies MiSTer's H slot, the source-H table
occupies V, and `ifilter` equals that rotated last-axis H table. Only one scaler
axis may be adaptive. Directional masks use `1x rotated` in TATE.

`preset_contracts.py` validates the complete 40-preset matrix: all seven fields,
case-sensitive file resolution, axis transposition, explicit interlace fallback,
one-adaptive-axis limit, dedicated Fixed/No-Gamma pairing, and mask modes.

## Current acceptance baselines (output codes)

All figures are exact endpoint-inclusive end-to-end RMS through the mask over
the full periodic supercell. L-infinity ceilings are also hard-gated in
`validate_port.py`.

| Family | Default | Fixed | No Gamma |
|---|---:|---:|---:|
| Guest Advanced | 1.230 | 4.785 | 2.029 |
| Royale | 16.720 | 23.497 | 17.370 |
| Royale Kurozumi | 19.854 | 25.311 | 19.517 |
| Easymode v5 | 3.591 | 9.286 | 6.207 |
| Lottes | 1.260 | 1.166 | off by design |

Kurozumi code 0 is independently gated (1.005 RMS, 3.7 maximum,
LUT[0] = `5,2,2`) so a future scalar refinement cannot erase its coloured
black flare. Gamma unique-level, plateau, and maximum-step diagnostics guard
against crushed ranges.

## Validation

- `selftest_mister_model.py`: phase boundaries, signed/odd adaptive arithmetic,
  FIR truncation, clamps, and masks.
- `selftest_mask_period.py`: optimized full-LCM scoring versus independent
  brute force, including mask mode 2x and a regression witness for crop scoring.
- `selftest_quantize.py`: exact symmetry and coefficient reconstruction.
- `selftest_fileio.py` / `selftest_presets.py`: every generated asset and all
  cross-preset contracts.
- `selftest_easymode_antiring.py`: source clamp, hardware saturation, canonical
  aliases, and matched-boost gain placement.
- `validate_patterns.py`: selector boundaries, shared-max RGB control, impulse,
  step, and checkerboard diagnostics. Hardware-unrepresentable differences are
  reported rather than hidden.
- `validate_port.py`: formats, gamma quantization, behavioral clipping,
  full-period RMS and maximum error, strict origin, brightness parity, moire,
  dedicated variant pipelines, and Kurozumi near-black color.

Portable tests and full validation use only the repository's vendored targets;
`MISTER_SHADER_REFERENCES` remains available for deliberate reference overrides.
