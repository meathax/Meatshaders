CRT SHADER PRESETS FOR MiSTer
=============================

Four CRT presets optimized for MiSTer at 1080p. Every preset also includes a
"- TATE" version for vertically rotated games.

All four presets are calibrated so the shader does not change your game's
colours. The masks and scanlines stay visible, but the colour each block of
output pixels averages to is matched to the source in perceived light, not just
in raw 8-bit values. Below each preset's highlight ceiling every colour lands
within a few codes of the source, and a 172 grey arrives at 171.5-172.0.

The one thing a CRT mask cannot preserve is peak white. A modulated field
cannot average to code 255, so highlights are compressed toward a ceiling that
depends on how deep the preset's scanlines are: 240 for Royale, 239 for Guest
Advanced, 224 for Kurozumi, 220 for Lottes. Colour and hue are preserved;
maximum brightness is not. See AUDIT_REPORT.txt for the measurements and for
how to trade scanline depth against that ceiling.


PRESETS
-------

CRT Guest Advanced
  A balanced all-purpose CRT look with sharp scanlines, natural glow and a
  compact phosphor mask. Highest-resolution gamma curve of the four.

CRT Lottes
  A softer CRT look with glow, bloom and a stretched-VGA mask. Its fixed beam
  makes it the most colour-accurate preset in the pack - every unclipped colour
  is within 0.9 codes of the source - at the cost of the lowest white ceiling.

CRT Royale
  A detailed RGB-triad CRT with the highest white ceiling and the most accurate
  highlights. It keeps its full adaptive beam, so one class of saturated colour
  against a bright channel can shift by up to 9 codes.

CRT Royale Kurozumi
  A PVM-style Royale variant with deep blacks and an aperture-grille mask.
  Second most accurate after Lottes, with a higher white ceiling.


INSTALLATION
------------

Copy the contents of each folder to the matching folder on the MiSTer SD card:

  Filters/*       -> /media/fat/Filters/
  Gamma/*         -> /media/fat/Gamma/
  Shadow_Masks/*  -> /media/fat/Shadow_Masks/
  Presets/*       -> /media/fat/Presets/

On MiSTer, open Video Processing, select Load Preset, and choose the preset
you want to use.


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
