====================================================================
 CRT EASYMODE & CRT LOTTES - MiSTer maximum-fidelity fixed-pipeline ports
 Audited edition v4 (2026-07-16)
====================================================================

These files reproduce as much of the canonical CRT Easymode and CRT Lottes
GLSL shaders as MiSTer's fixed video pipeline can express. They are scaler
filters, gamma curves, HDMI shadow masks and presets, not executable shaders.

Literal pixel-for-pixel identity is impossible on MiSTer without adding new
FPGA logic to every core. The original shaders use nonlinear operations after
filtering, neighbourhood clamps, wider/nonseparable kernels and coordinate
warping. MiSTer provides a pre-scaler gamma LUT, separable four-tap FIR filters
and a quantised post-scaler mask. This edition uses hardware-aware fitting,
late gain placement, full 256-phase/10-bit tables and exact phase symmetry to
produce the closest practical portable files.


QUICK INSTALL
-------------
Copy each folder's contents to the matching folder on the MiSTer SD card:

  Filters/*       -> /media/fat/Filters/
  Gamma/*         -> /media/fat/Gamma/
  Shadow_Masks/*  -> /media/fat/Shadow_Masks/
  Presets/*       -> /media/fat/Presets/

Then open OSD -> Video Processing -> Load preset.

Best starting presets:

  CRT Easymode - Adaptive
      Closest general Easymode profile on an adaptive-capable core.

  CRT Easymode - Pixel-Art Anti-Ring
      Often cleaner for hard-edged console/arcade pixel art. It approximates
      Easymode's nonlinear clamp; use Adaptive for the canonical FIR kernel.

  CRT Lottes - Default
      Closest default Lottes profile, including fitted bloom and matched mask.

  CRT Lottes - Fixed Compatibility
      Nearly identical Lottes result without adaptive filtering.


REQUIRED MISTER AND DISPLAY SETUP
---------------------------------
These settings materially affect the result and cannot be stored in a preset:

  Core Scandoubler Fx / Scanlines:  None / Off
  direct_video:                      0
  forced_scandoubler:                0
  Output resolution:                panel's native HDMI resolution
  TV/monitor scaling or overscan:    Off; use 1:1 / Just Scan / Screen Fit
  HDR:                              Off for source-faithful SDR output
  Aspect ratio:                     4:3 or the core's source-correct aspect
  Recommended output height:        1080p minimum

The core's own Scandoubler Fx is applied before MiSTer's scaler. Enabling its
scanlines causes double scanlines and gives adaptive filters the wrong
brightness signal.

Easymode's source recommendation is fractional vertical fit, corresponding to
vscale_mode=0. vscale_mode=1 is an optional integer-vertical aesthetic choice
with uniform scan spacing, but it can add borders and is not the shader default.

Use mask mode 1x for source-default shader-pixel geometry. The supplied "4K
Visual Pitch" presets use 2x only to keep a phosphor pitch closer to the 1080p
appearance on a 4K panel; they are display-tuned rather than pixel-identical.


EASYMODE MAXIMUM-FIDELITY DESIGN
--------------------------------
Horizontal filters:

  CRT Easymode (Port)_H
    Canonical four-tap Lanczos-family kernel. The shader default
    SHARPNESS_H=0.5 is squared internally, giving effective sharpness 0.25.

  CRT Easymode Pixel-Art Anti-Ring (Port)_H
    Fixed-FIR least-squares approximation of the shader's local min/max clamp
    over binary pixel neighbourhoods. It substantially reduces halos on hard
    pixel-art edges, but the canonical H filter is more faithful for gradients.

Vertical filters:

  ..._V Adaptive
    Recommended landscape profile. Each phase's two endpoint gains is fitted
    directly over RGB colours to Easymode's brightness-aware scan profile and
    MiSTer's real max-RGB adaptive control, 8-bit clamps and shift/add mask.
    It is not restricted to two fictional scan_bright parameter states.

  ..._V Balanced/Dark/Bright Matched
    Fixed matched alternatives with late gain in the final scaler stage.

  ..._V Balanced/Dark/Bright
    Unboosted versions retained for the Conventional gamma/mask factorisation.

  ..._V No Scanlines Matched
    Correct InputSize.y >= 400 path: scan attenuation is disabled while the
    canonical half-circle vertical interpolation remains active.

  ..._V Highlights
    Hardware-fitted saturated-white profile. Use only when peak-white scan
    depth is more important than general brightness adaptation.

Gamma and mask:

  CRT Easymode Matched (Port)
    Base curve round(255 * input^(2/1.8)). The previous scalar boost has been
    moved into the last scaler axis so H filtering retains highlight detail.

  CRT Easymode Matched Boost (Port)
    Coupled post-scaler mask: 112.5% selected phosphor, 93.75% others. Use only
    with Matched gamma and a Matched/adaptive filter or matching preset.

  CRT Easymode (Port) gamma + mask
    Conventional compatibility pair with the complete 1.2 boost in the LUT
    and the original 100%/81.25% mask approximation.

Source-height cutoff:

Easymode automatically disables scan attenuation when InputSize.y >= 400.
MiSTer cannot switch filters by source height, so load:

  CRT Easymode - 400p+ No Scanlines

for 400/480-line sources. Landscape presets explicitly use the no-scanline
table as their interlace fallback. The supplied TATE profiles place gain in the
hardware vertical (last) scaler stage and use explicit interlace filters.


LOTTES MAXIMUM-FIDELITY DESIGN
------------------------------
The canonical source compiles with DO_BLOOM enabled. Earlier ports omitted it.
The v4 default approximates Tri + 0.15*Bloom with the best practical separable
four-tap fit:

  - H blends 16.3618283% of the source H7 bloom kernel into H5.
  - V is least-squares fitted to the nonseparable 7x5 source kernel while
    preserving canonical flat-field gain.
  - At least 98.835% of bloom support lies inside the available four taps.
  - The constrained 4x4 spatial fit has about 1.8% relative kernel error.

Matched default:

  CRT Lottes (Port)_H
  CRT Lottes (Port)_V Adaptive     maximum-fidelity adaptive profile
  CRT Lottes (Port)_V              fixed fallback, almost as accurate

These must use a "CRT Lottes Matched ..." mask. Its 47e/27e/17e tokens use
143.75% for the selected channel and 87.5% for the others; the V gain is fitted
jointly to reproduce Lottes' linear-light 1.5/0.5 mask after encoded output.

The Lottes TATE preset rotates the fitted spatial pair and adaptive axis. The
single shared adaptive path prevents an exact rotated ordering; use Fixed TATE
if a core lacks adaptation or if the fixed response is preferred.

Crisp compatibility:

  CRT Lottes Crisp (Port)_H/V

This preserves the earlier bloom-free look and uses the original 43c masks.
Load "CRT Lottes - Crisp No Bloom". Do not mix a Matched V filter with a 43c
mask, or a Crisp V filter with a 47e Matched mask.

Use Gamma: Off for every Lottes preset. A pre-scaler LUT cannot reproduce the
shader's exact linearise -> filter/mask -> piecewise-sRGB ordering.


CORE / FILTER-INTERFACE COMPATIBILITY
-------------------------------------
The files use current MiSTer 256-phase, 10-bit format. Current Main downsamples
them safely for older core filter interfaces:

  v1  16-phase, 9-bit legacy
  v2  64-phase, 9-bit
  v3  64-phase, 9-bit adaptive
  v6  256-phase, 10-bit, no adaptive
  v7  256-phase, 10-bit adaptive (current full capability)

Use fixed presets on a non-adaptive core:

  CRT Easymode - Balanced
  CRT Easymode - Balanced TATE
  CRT Lottes - Fixed Compatibility
  CRT Lottes - Fixed TATE

A genuinely old Main binary should be updated. Only one scaler axis can use an
adaptive table at a time. TATE presets account for this limitation explicitly.


SAVING / DEFAULTS
-----------------
Video Processing selections are saved per core. To request a global starting
preset, add one of these to MiSTer.ini:

  preset_default=CRT Easymode - Adaptive
  preset_default=CRT Lottes - Default

Use only one preset_default line. Saved per-core settings load afterward and
override it. There is no gamma_default MiSTer.ini key.


UNAVOIDABLE DIFFERENCES FROM THE GLSL SOURCES
----------------------------------------------
Easymode cannot exactly reproduce:

  - the nonlinear per-pixel local min/max anti-ringing clamp;
  - output transfer, mask and final boost after filtering;
  - (max RGB + Rec.709 luma)/2 adaptive control;
  - automatic source-height switching.

Lottes cannot exactly reproduce:

  - linear-light filtering followed by exact piecewise sRGB;
  - the nonseparable 7x5 Tri/Bloom kernel and line-dependent H3/H5/H7 paths;
  - curvature, border treatment and off-screen sampling;
  - linear-light mask multiplication before sRGB conversion.

Both are also limited by MiSTer's four taps, 8-bit intermediate clamps and
1/16-step shadow-mask arithmetic. Direct Video has no HDMI mask, gamma support
is core-dependent, and internally pixel-doubled sources cannot reproduce a
four-logical-pixel GLSL kernel with four physical taps.

See AUDIT_REPORT.txt and SOURCES.txt for measured results and provenance.


====================================================================
 V5 ADDITIONS (2026-07-17): CRT GUEST ADVANCED, CRT ROYALE,
 CRT ROYALE KUROZUMI
====================================================================

Three further shader ports, fitted for 1080p output with a toolchain that
models MiSTer's scaler and mask arithmetic exactly (verified against
ascal.vhd / shadowmask.sv / video.cpp; see tools/DESIGN.md). All V filter
rows are at or below unity gain — these ports never clip in the scaler; the
transfer curve lives in the gamma LUT and the phosphor punch in the mask.

Install as before; new presets:

  CRT Guest Advanced - Default        crt-guest-advanced, closest port
  CRT Royale - Default                crt-royale defaults (slot mask)
  CRT Royale Kurozumi - Default       PVM-look Royale preset
  <each> - Fixed Compatibility        for cores without adaptive filters
  <each> - No Gamma                   for cores without gamma support
  <each> - TATE                       vertical/rotated games

PER-SHADER NOTES
----------------
CRT Guest Advanced: the closest of the three (end-to-end RMSE 3.2 codes vs
the v4 Lottes benchmark of 1.8). Brightness-adaptive beam maps naturally
onto the adaptive V filter; the CGWG magenta/green mask lands almost
exactly on hardware mask steps.

CRT Royale: scanline shape, brightness response and the slot mask track the
shader closely through dark and mid tones. One documented deviation: the
shader fills its scanlines to near-flat at peak white (bloom); MiSTer's
two-endpoint linear adaptive blend cannot follow that collapse, so bright
whites keep mild scanlines here (flat white fields read about 24% darker
than the shader would render them; nothing clips). The 24x24-pixel slot
tile the shader renders at 1080p exceeds the 16x16 hardware mask limit and
is approximated by its best 12x3 periodic fit.

CRT Royale Kurozumi: the grade pass's tone curve — including its warm black
lift — is reproduced exactly in a 3-channel gamma LUT, and the Gaussian
horizontal filter is a native 4-tap fit. Scanlines are deep at every
brightness, as intended. At 1080p the shader itself collapses its aperture
grille to a 2-pixel R/B alternation; the hardware mask token format cannot
give green its own value per pixel, so the port preserves the red-channel
ripple and overall luma while flattening the blue ripple. Misconvergence
(+-0.05 px) is below what 4-tap filters can express and is omitted.
Deep scanlines at fractional scales: 224-line cores at 1080p show a faint
line-thickness ripple (within the pack's moire guard); vscale_mode=1
removes it entirely and is recommended for this preset.

COMPATIBILITY (applies to all three)
------------------------------------
On cores built without adaptive filter support, adaptive files silently run
on their dark coefficient set only — use the Fixed Compatibility preset
there. On cores without gamma support the Default preset's LUT silently
does nothing — use No Gamma (a documented, slightly darker approximation).
If Default ever looks drastically darker/harsher than Fixed Compatibility
on some core, that core is non-adaptive: prefer the Fixed preset.
(The same applies to the v4 Easymode Adaptive preset, whose two sets differ
far more — if Easymode looked wrong on your core, this is the first thing
to check.)
