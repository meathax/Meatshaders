====================================================================
 CRT SHADER PORTS FOR MiSTer - CANONICAL 1080p EDITION
====================================================================

Fixed-pipeline ports of five CRT shaders for MiSTer's HDMI scaler. Each
family has two presets: the bare family name (landscape) and its "- TATE"
portrait form.


INSTALL
-------
Copy each directory's contents to the matching directory on the MiSTer SD
card:

  Filters/*       -> /media/fat/Filters/
  Gamma/*         -> /media/fat/Gamma/
  Shadow_Masks/*  -> /media/fat/Shadow_Masks/
  Presets/*       -> /media/fat/Presets/

Then in the MiSTer OSD: Video Processing -> Load preset, and pick the shader.

Requires 1920x1080 HDMI output with the TV set to 1:1 / Just Scan (no
overscan, no display-side scaling) and MiSTer mask mode 1x. All families
except Lottes need the v7 adaptive-filter cores.


SHADERS
-------
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

CRT Lottes
  Timothy Lottes' shader with soft bloom and a stretched-VGA shadow mask.
  Warm and glowy; gamma runs through the shader's own curve (LUT off).


NOTES
-----
These are fixed-pipeline approximations, not the original GPU shaders:
MiSTer's scaler has no programmable shading, so wide/nonseparable blurs,
curvature and per-channel effects are approximated or omitted. Each file is
the closest measured mapping for a real 1920x1080 output grid.

For measured accuracy figures see AUDIT_REPORT.txt; for sources and licenses
see SOURCES.txt.
