# MiSTer CRT canonical 1080p design

Date: 2026-07-17; Guest expansion 2026-07-21. The pack contains one landscape
preset and one transposed TATE preset for each of seven shader families:
Easymode, Guest Advanced, Guest Advanced Fast, Guest Advanced Fastest,
Lottes, Royale, and Royale Kurozumi. Reference math is vendored in
`tools/targets/`. The non-Guest families are pinned to libretro/slang-shaders
commit `3b0d6aa1d134a168478cd9c904a866d969f8882b`; the three Guest families
are built from guest.r's upstream release drop
`crt-guest-advanced-2026-07-12-release1` (verified numerically identical to
the libretro pin at defaults for the advanced chain, which also serves as the
cross-check).

These are deterministic fixed-pipeline approximations, not executable GPU
shaders. The objective is the closest result MiSTer's gamma LUT, separable
four-tap scaler, one adaptive-control path, and v2 mask tokens can represent at
an exact 1920x1080 output grid.

## Canonical runtime contract

The runtime manifest is exact: 20 filters, six gamma LUTs, seven masks, and
fourteen presets. There are no selectable compatibility or alternate-look
presets.

Every family except Lottes uses a fixed H table, an adaptive V table, an
interlace-safe no-scanline V table, one fitted LUT, and one mask. Landscape
puts adaptation on V. TATE transposes the source axes, puts the adaptive
table on H, uses the fixed H response on V/interlace, and selects
`maskmode=1x rotated`.

Guest Advanced Fast is upstream's performance rewrite of the identical look:
its fixed-pipeline projection equals the advanced chain exactly at defaults
(validated contract in `guest_fast_ref.py`), so its family files carry the
same coefficients under their own names. Guest Advanced Fastest is a real
variant: its single final pass has no glow lift, no pr_scan multiply and no
w3-epsilon dimming (`transfer(1.0) == 1.0`), and gets its own LUT/V/mask fit.
The Guest re-audit also corrected the reference's dark_compensate term, which
the shader clamps to identically 1.0 at defaults; the previous model boosted
near-black targets by up to 10% below input code ~58.

Lottes uses one fixed H/V pair, gamma off, and one mask. Its beam shape is
brightness-independent, so the fixed vertical fit is measurably closer than an
adaptive pair.

## Hardware model

`mister_model.py` follows the audited Main RTL arithmetic:

- scaler phase uses the upper eight fractional bits (`frac12 >> 4`);
- 256 signed 10-bit coefficient rows use the hardware's intermediate
  truncation and 8-bit saturation order;
- adaptive coefficients evaluate `(A*(256-lum) + B*lum) >> 1`, preserving
  valid odd signed results;
- adaptive control is maximum RGB after the H/gamma path and samples the
  nearest source line at the half-phase boundary;
- v2 mask multiplication is a saturating sum of independently truncated
  shifts.

Every stored table has 256 phases, exact conjugate symmetry, and a
self-symmetric phase 128. Acceptance measures behavior through the integer
pipeline rather than relying on coefficient-sum proxies.

## Fitting

The horizontal fit folds each source response into MiSTer's four-tap window.
For adaptive families, the monotone LUT/control warp and dark/bright vertical
endpoints are co-optimized in output-code space. Model-in-loop refinement is
adopted only when exact end-to-end error improves without breaking LUT
monotonicity or flat-field clipping safety.

Easymode carries its pre-clip transfer divided by 1.105 in the LUT and the
remaining gain in the mask. This preserves useful highlight resolution in the
adaptive control and moves final saturation to the late mask stage.
Kurozumi retains a true RGB LUT so its warm lifted black is not collapsed to a
neutral scalar curve.

Royale's narrow beam alternates between adjacent integer output-line
placements at 240p-to-1080p. Its canonical table blends 55% of a finite-subline
integrated fit into the unconstrained shader fit. This is the smallest tested
blend with useful safety margin under the 7.65-code full-frame moire gate; the
final measured score is 7.00 codes.

## Masks

Mask fitting uses exact token shift/add arithmetic and the complete
least-common-multiple supercell of hardware and source periods. Strict output
origin is a release requirement; a visually equivalent cyclic roll does not
count as pixel-identical.

- Guest maps closely to a compact CGWG tile.
- Royale preserves slot structure with a hardware-legal 12x6 slice of its
  larger source tile.
- Kurozumi uses the best pixel-local 1x2 compromise; one v2 token cannot encode
  three independent phosphor multipliers at one output pixel.
- Easymode's mask carries the 1.105 gain split.
- Lottes uses the source-anchored 6x2 phase beginning with green at pixel
  `(0,0)`: `27e,27e,17e,17e,47e,47e` on row zero and
  `17e,47e,47e,27e,27e,17e` on row one.

## Exact 1080p acceptance

Stability is evaluated over complete 1080-line frames for 224p
(`1080/224`) and 240p (`4.5x`) sources, at 25%, 50%, 75%, and 100% input, for
all RGB channels. The larger peak/trough standard deviation must remain at or
below `0.03*255 = 7.65` output codes.

End-to-end reference RMS and worst stability scores are:

| Family | Masked RMS | Worst 1080p moire |
|---|---:|---:|
| Easymode | 3.591 | 5.50 |
| Guest Advanced | 1.237 | 4.50 |
| Guest Advanced Fast | 1.237 | 4.50 |
| Guest Advanced Fastest | 1.160 | 5.50 |
| Lottes | 1.166 | 3.50 |
| Royale | 19.968 | 7.00 |
| Royale Kurozumi | 19.854 | 6.65 |

The larger Royale/Kurozumi errors expose shader operations the hardware cannot
encode: wide/nonseparable bloom, large slot structures, local nonlinear clamps,
independent per-channel beam decisions, curvature, and misconvergence.

## Validation entry points

- `selftest_mister_model.py`: phase boundaries, signed adaptive math,
  truncation, saturation, flat-field control, and exact peak/trough windows.
- `selftest_mask_period.py`: full-period scoring against brute force.
- `selftest_quantize.py`: table symmetry and retained Lottes reconstruction.
- `selftest_fileio.py` and `selftest_presets.py`: exact manifest and ten-preset
  relationship contract.
- `validate_patterns.py`: RGB ramps, impulses, steps, checkerboards, and
  selector-boundary diagnostics.
- `validate_port.py`: formats, gamma quality, clipping, strict-origin masked
  error, brightness parity, full-frame 1080p stability, and Kurozumi black.

`MISTER_SHADER_REFERENCES` may override the vendored reference directory for a
deliberate comparison; ordinary validation is self-contained.
