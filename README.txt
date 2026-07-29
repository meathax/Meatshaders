###########################
CRT shaders for MiSTer FPGA
###########################

Use 1080p - video mode 8 in mister ini
-------
CRT Lottes
  Timothy Lottes' shader with soft bloom and a stretched-VGA shadow mask.
  Warm and glowy; gamma runs through the shader's own curve (LUT off).

CRT Guest Advanced
  guest.r's advanced CRT. Sharp adaptive scanlines with a compact CGWG
  phosphor mask; the all-round best-looking default.

CRT Royale
  TroggleMonkey's Royale with its slot-mask phosphor structure and
  brightness-dependent bloom. Rich, deep, mask-heavy look.

CRT Royale Kurozumi
  Kurozumi's P22/PVM grade over Royale: warm professional-monitor color,
  thin bright scanlines, aperture-grille mask.

CRT Easymode
  Clean, bright, low-cost scanline look. Good for a subtle CRT feel that
  stays sharp and readable.


Each shader preset has a TATE variant

INSTALL
-------
Copy each directory's contents to the matching directory on the MiSTer SD
card:

  Filters/*       -> /media/fat/Filters/
  Gamma/*         -> /media/fat/Gamma/
  Shadow_Masks/*  -> /media/fat/Shadow_Masks/
  Presets/*       -> /media/fat/Presets/

Then in the MiSTer OSD: Video Processing -> Load preset, and pick the shader.

For best results these require mister set to output 1920x1080, with the TV set to 1:1 / Just Scan (no
overscan, no display scaling) and MiSTer mask mode 1x

NOTES
-----
These are fixed-pipeline approximations, not the original GPU shaders:
MiSTer's scaler has no programmable shading, so wide/nonseparable blurs,
curvature and per-channel effects are approximated or omitted. Each file is
the closest measured mapping for a real 1920x1080 output grid.

For measured accuracy figures see AUDIT_REPORT.txt; for sources and licenses
see SOURCES.txt.
