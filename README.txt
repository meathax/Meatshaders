###########################
CRT shaders for MiSTer FPGA
###########################

Use 1080p - video mode 8 in mister ini

Four CRT presets for MiSTer at 1080p, calibrated so the shader does not
change your game's colours.

CRT Lottes
  A softer CRT look with glow, bloom and a stretched-VGA mask. Its fixed beam
  makes it the most colour-accurate preset in the pack - every unclipped colour
  is within 0.9 codes of the source - at the cost of the lowest white ceiling.

CRT Guest Advanced
  A balanced all-purpose CRT look with sharp scanlines, natural glow and a
  compact phosphor mask. Highest-resolution gamma curve of the four.

CRT Royale + Kurozumi
  A detailed RGB-triad CRT with the highest white ceiling and the most accurate
  highlights. It keeps its full adaptive beam, so one class of saturated colour
  against a bright channel can shift by up to 9 codes.

Each shader preset has a TATE variant for vertically rotated games.


INSTALL
-------
Copy each directory's contents to the matching directory on the MiSTer SD
card:

  Filters/*       -> /media/fat/Filters/
  Gamma/*         -> /media/fat/Gamma/
  Shadow_Masks/*  -> /media/fat/Shadow_Masks/
  Presets/*       -> /media/fat/Presets/

Then in the MiSTer OSD: Video Processing -> Load preset, and pick the shader.

For best results these require mister set to output 1920x1080, with the TV
set to 1:1 / Just Scan (no overscan, no display scaling) and MiSTer mask
mode 1x.


COLOUR ACCURACY
---------------
All four presets are calibrated in perceived light, not just raw 8-bit values:
the colour each block of mask/scanline output pixels averages to is matched to
the source. Below each preset's highlight ceiling every colour lands within a
few codes of the source, and a 172 grey arrives at 171.5-172.0.

The one thing a CRT mask cannot preserve is peak white. A modulated field
cannot average to code 255, so highlights are compressed toward a ceiling that
depends on how deep the preset's scanlines are: 240 for Royale, 239 for Guest
Advanced, 224 for Kurozumi, 220 for Lottes. Colour and hue are preserved;
maximum brightness is not. See AUDIT_REPORT.txt for the measurements and for
how to trade scanline depth against that ceiling.


NOTES
-----
These are fixed-pipeline approximations, not the original GPU shaders:
MiSTer's scaler has no programmable shading, so wide/nonseparable blurs,
curvature and per-channel effects are approximated or omitted. Each file is
the closest measured mapping for a real 1920x1080 output grid.

For measured accuracy figures see AUDIT_REPORT.txt; for sources and licenses
see SOURCES.txt.


VALIDATION
----------

  python tools/selftest_fileio.py
  python tools/selftest_presets.py
  python tools/selftest_mister_model.py
  python tools/selftest_quantize.py
  python tools/selftest_mask_period.py
  python tools/validate_patterns.py
  python tools/validate_port.py

The last command sweeps 729 colours across the whole RGB cube through the exact
MiSTer LUT, FIR, clamp and v2-mask arithmetic at 240p -> 1080p, and gates on the
perceived (light-linear) colour error. To regenerate the calibration after
changing a mask or filter, run:

  python tools/color_match.py all
