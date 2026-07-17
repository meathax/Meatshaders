====================================================================
 CRT SHADER PORTS FOR MiSTer - CANONICAL 1080p EDITION
 Pixel-accuracy build, 2026-07-17
====================================================================

This pack contains one canonical 1920x1080 setup for each of five CRT shader
families:

  CRT Easymode
  CRT Guest Advanced
  CRT Lottes
  CRT Royale
  CRT Royale Kurozumi

Each family has exactly two user-facing presets. The bare family name is the
landscape/default preset; "- TATE" is its correctly transposed portrait form:

  CRT Easymode.ini
  CRT Easymode - TATE.ini
  CRT Guest Advanced.ini
  CRT Guest Advanced - TATE.ini
  CRT Lottes.ini
  CRT Lottes - TATE.ini
  CRT Royale.ini
  CRT Royale - TATE.ini
  CRT Royale Kurozumi.ini
  CRT Royale Kurozumi - TATE.ini

There are no alternate looks or compatibility presets. The pack contains the
single closest measured mapping for its stated target.


WHAT THESE FILES ARE
--------------------
The original GPU shaders cannot run in MiSTer's scaler. These files are fixed
pipeline derivatives: 256-phase signed 10-bit FIR tables, gamma LUTs, HDMI
shadow masks and presets fitted to MiSTer's real integer arithmetic.

Literal identity is impossible where a source shader uses operations absent
from the FPGA path, including wide or nonseparable kernels, nonlinear local
clamps, curvature, misconvergence and per-channel adaptive decisions. The
toolchain models those limits explicitly and optimizes the representable
result instead of claiming that the shader program itself was ported.

Runtime inventory:

  Filters:       14
  Gamma LUTs:     4
  Shadow masks:   5
  Presets:       10

The filter count is larger than the preset count because presets compose
internal stages. Easymode, Guest Advanced, Royale and Royale Kurozumi each use
an H table, an adaptive V table and an interlace-safe no-scanline V table.
TATE transposes the H/adaptive-V pair and reuses those same files. Lottes uses
one fixed H/V pair. These extra files are implementation stages, not additional
looks to choose from.


INSTALLATION
------------
Copy each directory's contents to the matching directory on the MiSTer SD card:

  Filters/*       -> /media/fat/Filters/
  Gamma/*         -> /media/fat/Gamma/
  Shadow_Masks/*  -> /media/fat/Shadow_Masks/
  Presets/*       -> /media/fat/Presets/

Open OSD -> Video Processing -> Load preset, then select the bare family name
or its "- TATE" form.


REQUIRED 1080p SETUP
--------------------
The accuracy claims in this pack assume all of the following:

  MiSTer HDMI output:                 1920x1080
  TV/monitor scaling and overscan:    Off
  TV/monitor mode:                    1:1 / Just Scan / Screen Fit
  MiSTer mask mode:                   1x (stored in each preset)
  Core Scandoubler Fx / Scanlines:    None / Off
  direct_video:                       0
  forced_scandoubler:                 0
  HDR or display enhancement:         Off
  Aspect ratio:                       source-correct, normally 4:3

Do not ask a 4K panel to rescale these masks without accepting a different
phosphor pitch. The target is a real 1920x1080 output grid with no additional
display scaling.

Easymode, Guest Advanced, Royale and Royale Kurozumi require MiSTer's v7
adaptive-filter interface for their canonical result. Their presets use the
gamma LUT and one adaptive scaler axis. Lottes is intentionally fixed: its
source scan profile is brightness-independent, so a fixed V table is the
canonical mapping and gamma remains off.

Only one scaler axis can be adaptive. Landscape places adaptation on V; TATE
places it on H and uses the fixed H response on V. The preset files encode this
axis rotation explicitly.


1080p VALIDATION TARGET
-----------------------
The acceptance run evaluates complete 1080-line output frames for both common
source heights:

  240p -> 1080p: 4.5x vertical scale
  224p -> 1080p: 1080/224 = 4.821428571...x vertical scale

It tests all 256 scaler phases, endpoint-inclusive input codes through 255,
full hardware/reference mask supercells, strict mask origin, clipping and
peak/trough line-thickness variation. The moire guard is 7.65 output codes.
See AUDIT_REPORT.txt for the measured acceptance record and tools/DESIGN.md for
the arithmetic model.


FAMILY NOTES
------------
CRT Easymode
  Uses the current gain-split fit: its gamma LUT remains informative through
  bright values while final saturation occurs in the mask stage, matching the
  shader's clip order as closely as MiSTer permits. The nonlinear local
  anti-ringing clamp and Rec.709-assisted adaptive control are not available in
  hardware.

CRT Guest Advanced
  Maps especially closely to MiSTer's adaptive beam and compact mask model.
  Chroma-dependent beam behavior remains an unavoidable shared-control limit.

CRT Royale
  Retains source slot structure with a hardware-legal mask slice and fits its
  brightness-dependent beam through the adaptive V path. The original wide,
  nonseparable bloom and full 24x24 mask cannot be represented literally.

CRT Royale Kurozumi
  Uses a channel-aware gamma LUT for the PVM grade and a pixel-local mask fit.
  MiSTer's mask token format cannot assign three independent multipliers at one
  output pixel, so the result is the closest strict pixel-local compromise.

CRT Lottes
  Includes the source-default Tri + bloom response in a fixed separable H/V
  fit. Gamma is off because MiSTer's LUT cannot reproduce Lottes' exact
  linearize -> filter/mask -> piecewise-sRGB ordering. Its 6x2 mask is anchored
  to the shader's output-pixel origin; the first pixel is the source-selected
  green phase rather than a visually equivalent rolled tile.


IMPORTANT LIMITS
----------------
MiSTer's portable scaler path provides separable four-tap FIR filtering, one
max-RGB adaptive control value, 8-bit intermediate clamps, a pre-scaler LUT and
post-scaler mask tokens quantized in 1/16 steps. Consequently:

  - nonlinear neighbourhood clamps cannot be expressed by FIR coefficients;
  - nonseparable and wider kernels must be approximated;
  - shader curvature and misconvergence are omitted;
  - source-height conditionals can only be represented by the preset's
    interlace fallback, not by arbitrary runtime shader branching;
  - mask phases and per-pixel color multipliers are limited by the v2 token
    format;
  - Direct Video bypasses the HDMI mask path.

For provenance, licenses and pinned source revisions, see SOURCES.txt.
