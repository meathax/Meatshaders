# MiSTer CRT source-derived 1080p design (v7)

Date: 2026-07-27.

This pack contains one landscape and one TATE preset for five shader families:
Easymode, Guest Advanced, Lottes, Royale, and Royale Kurozumi. They are
deterministic projections of real shader defaults into MiSTer's fixed HDMI
scaler pipeline, specifically for a 1920x1080 output grid with
`vscale_mode=0` and mask mode 1x.

MiSTer cannot execute GLSL or Slang. Source behavior is evaluated by the
executable references in `tools/targets/`, fitted into legal gamma/filter/mask
assets, then evaluated again through the integer hardware model. An effect is
adopted only when that complete path measures closer and still passes tone,
clipping, pattern, and 1080-line stability gates.

## Locked source contract

- Easymode and Lottes: `libretro/glsl-shaders` commit
  `2b2c5ee3fd8e1a3884e20ed424fd9bfbc51cbb3d`.
- Guest Advanced: guest.r release
  `crt-guest-advanced-2026-07-23-release1`, archive SHA-256
  `a006a3785f787a3e0bff6c4ed3f505a396b07274bf7fe341b6877d02a4fa7560`;
  spatial defaults are cross-checked against `libretro/slang-shaders`
  `3b0d6aa1d134a168478cd9c904a866d969f8882b`.
- Royale: that same current `slang-shaders` commit.
- Royale Kurozumi: the matched historical preset and public-domain Grade blob
  from commit `7f34fc7469ecc7d90e03df45d1a10975136eb712`, combined with current pinned
  Royale. This is byte-identical for the sampled core passes except one
  deliberate later upstream correction in `crt-royale-bloom-approx.h`: the
  historical pass calculated its Gaussian color and accidentally output the
  unblurred input; the corrected pass outputs the calculated color. The modern
  Grade replacement is not used because it silently drops Kurozumi's old
  gamma/color overrides and creates an unintended raised black.

See `SOURCES.txt` for URLs, licenses, hardware revisions, and the exact
one-line Kurozumi hybrid provenance.

## Runtime contract

The public install surface remains exactly 14 filters, four gamma LUTs, five
masks, and ten presets. There are no quality tiers or compatibility aliases.

For adaptive families, landscape assigns the source horizontal response to H,
the brightness-dependent beam to V, and a no-scanline table to interlace. TATE
transposes H/V, retains the safe interlace response, and rotates the mask.
Lottes uses fixed H and V tables with gamma off because its default beam is
brightness-independent.

MiSTer presets can name H/V/scan/interlace filters, gamma, mask, and mask
orientation. They cannot set output resolution, scaling mode, HDR, color
range, or TV controls; those are documented in `README.txt`.

## Hardware model

`mister_model.py` follows the audited current Main/Template implementation:

- 256 phases are selected by truncating the 12-bit source fraction to its
  upper eight bits (`frac12 >> 4`);
- every filter row has four signed 10-bit coefficients;
- scaler accumulation, intermediate truncation, 8-bit clamps, and saturation
  occur in hardware order;
- adaptive coefficients use one signed shift after the A/B weighted sum;
- adaptive control is the nearest source line's maximum post-gamma/post-H RGB;
- gamma runs before scaling;
- a v2 mask runs after scaling and uses saturating shift/add token arithmetic.

Every generated table has exact conjugate symmetry and a self-symmetric phase
128. Accuracy is measured through these operations, not inferred from floating
coefficient sums.

## Shared fitting and acceptance

The fitter works in output-code space. It searches legal LUT curves and
four-tap endpoint tables, quantizes them, simulates hardware, and accepts only
Pareto-safe improvements. The source and hardware mask periods are scored over
their full least-common-multiple supercell at the declared output origin.

The v7 suite adds source-side RGB oracles and nonuniform-image checks where a
gray ramp concealed important errors:

- Guest uses a 9-level RGB cube, exact linear-mask ordering, shared max-RGB
  control, and a bounded selector discontinuity objective.
- Easymode uses a 5-level RGB lattice and an exact horizontal black/white step
  with the source's per-channel Lanczos anti-ring clamp.
- Lottes uses all 256 codes through its true linear-light mask followed by the
  literal piecewise sRGB encode.
- Kurozumi exposes the full nonseparable P22 RGB transform, historical Grade
  quantization, true black, and mask-dependent brightpass/bloom ordering.

All adaptive families are simulated over complete 1080-line frames at 224p
and 240p, four signal levels, and all channels. The larger peak/trough standard
deviation must not exceed `0.03*255 = 7.65` codes. LUTs must remain monotone,
retain adequate tone levels, use bounded plateaus/steps, and keep behavioral
flat output at or below the 255.5 rounding boundary.

## Family mappings

### CRT Easymode

The source's squared-sharpness Lanczos response, luma-assisted scan weight,
pre-clip transfer, and 1.105 late gain split are represented by H, adaptive V,
LUT, and mask. A partial-linear-light LUT family was tested and rejected: it
worsened neutral/RGB tone, reached coefficient limits, or created excessive
moire. The runtime tables therefore remain the measured optimum, while new RGB
and source-step gates prevent a gray-only regression.

Easymode disables scanlines automatically for source heights at or above 400.
MiSTer has no equivalent progressive-height predicate inside one preset;
`sfilter` is core-flag-selected, not a general height branch. The ten-preset
contract therefore targets low-resolution retro sources and reports this limit
instead of shipping misleading automatic or extra presets.

### CRT Guest Advanced

Guest's mask, brightboost, glow, beam saturation, and `pr_scan` ordering are
evaluated in the July 23 source path. The old gray-only warp was excellent on
neutral fields but badly wrong for saturated mixed colors because Guest narrows
minority channels while MiSTer has one shared maximum-RGB control. The v7
Pareto fit trades a small neutral error increase for a large RGB and worst-pixel
reduction, a smoother 239-level LUT, and a smaller phase-128 selector jump.

The release's temporal afterglow is intentionally excluded: it needs a
previous-frame buffer, which the scaler does not have. Default spatial glow is
included; wide spatial spread beyond four separable taps remains approximate.

### CRT Lottes

Lottes is evaluated in its literal order: input decode, source Tri/Bloom
kernels, 1.5/0.5 mask in linear light, then the shader's piecewise sRGB encode.
The source separable least-squares H blend uses H3 `0.06003828`, H5
`0.71920149`, and H7 `0.22076023`; replacing the older retained H7 value cuts
source-kernel weight error by 79.4%. V keeps the source beam proportions while
its phase DC is calibrated over all 256 codes. Gamma remains off.

The 6x2 stretched-VGA mask is anchored to the actual source origin: pixel
`(0,0)` is green-lit. Curvature and true off-screen sampling cannot be encoded.

### CRT Royale

Royale keeps the current maintained Slang defaults. The source 24x24 slot mask
projects best to a legal 12-row x 6-column token tile. Wider 12- or 15-column
tiles score worse; 15 columns also introduce a five-triad cadence reset. Mask
mode 2x is rejected because it converts every phosphor into a 2x2 block and
destroys one-pixel RGB detail.

At 240p-to-1080p the very narrow source beam can alternate between adjacent
integer line placements. The canonical V table blends 55% of a finite-subline
integrated fit into the point fit, retaining the closest source shape that
stays under the 7.65-code full-frame stability ceiling. Wide nonseparable bloom,
curvature, and per-channel convergence remain only partially projectable.

### CRT Royale Kurozumi

Kurozumi uses the historically active P22 RGB-to-XYZ-to-Rec.709 transform,
CAT02 D65 adjustment, reverse 2.304 electron-gun curve, pass-0 RGBA8
quantization, true zero black, and the Kurozumi Royale parameters. The later
upstream one-line bloom correction is retained as an explicit authenticity
enhancement because it makes the pass output the Gaussian it computes.

The 16x16 source aperture-grille reconstruction was independently regenerated
from the pinned texture and matched all 768 channel codes. MiSTer's token has
only one lit/off pair per pixel, so the shipped two-pixel projection is the
best pixel-local compromise; horizontal token dithering improves averages but
creates individual errors above 120 codes. The P22 3x3 transform and subtle
misconvergence are nonseparable and cannot be exactly represented by three
independent 1-D LUTs plus one shared beam control.

## Validation entry points

- `selftest_fileio.py`: emitted syntax and parser behavior.
- `selftest_quantize.py`: phase symmetry and signed-table quantization.
- `selftest_mister_model.py`: exact phase, FIR, adaptive, clamp, and mask RTL.
- `selftest_mask_period.py`: FFT/full-LCM scoring against brute force,
  including 2x masks and unequal source/hardware periods.
- `selftest_presets.py`: exact ten-preset inventory and axis relationships.
- `validate_patterns.py`: RGB ramps, saturated fields, impulses, steps,
  checkerboards, and selector-boundary diagnostics.
- `validate_port.py`: formats, inventory, tone, clipping, source accuracy,
  strict-origin masks, brightness, full-frame moire, RGB gates, and black.

`MISTER_SHADER_REFERENCES` may override `tools/targets/` for deliberate source
comparisons. Ordinary builds and validation are self-contained.
