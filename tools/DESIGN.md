# MiSTer CRT port design

## Objective

The pack contains Guest Advanced, Lottes, Royale and Royale Kurozumi, each as
one landscape preset and one exact TATE transpose. The upstream shader models
still define the CRT geometry, beam and mask cadence. For Lottes and both
Royale families, the final colour objective is the supplied unfiltered
`default.webp`, not the upstream shader's deliberate display tint.

The calibrated families preserve black, use equal spatial-mean R/G/B mask
energy, and keep one common transfer curve for all channels. This prevents a
neutral source colour from becoming magenta, green or cyan.

## Hardware model

MiSTer's usable pipeline is fixed:

1. per-channel 256-entry 8-bit gamma LUT;
2. horizontal four-tap, 256-phase FIR and 8-bit clamp;
3. vertical four-tap FIR, optionally max-RGB adaptive, and 8-bit clamp;
4. repeating v2 phosphor mask using 1/16-step shift/add multipliers.

`tools/mister_model.py` reproduces the truncation, saturation, phase selection
and mask arithmetic. Calibration and validation use that integer model, not a
floating approximation.

## Screenshot measurement

The four supplied WebP captures share the same 1423x801 coordinate system.
Measurement selects exact dominant colours from `default.webp`, erodes the
selection to exclude edges, and averages the corresponding shader pixels over
complete scanline and mask phases. This separates colour reproduction from
the intentionally high-frequency CRT texture.

The stable reference colours include neutral 172 and 101, brown (154,64,23),
purple (57,8,124), blue (71,69,218), skin (247,210,195), and white. The fitting
curve itself uses all 256 neutral codes, so it is not overfit to this palette.

## Family calibration

### Lottes

Lottes was already chromatically neutral, but its mask transmitted too little
light. The original 6x2 stretched-VGA selector layout is retained. Each token
uses the strongest legal selected/other pair (31/16 and 15/16), giving equal
mean R/G/B transmission of 1.270833. Gamma remains off, preserving all source
codes and the source-shaped fixed V filter.

### Royale

The source-fitted slot mask averaged R/G/B transmission as approximately
0.898/0.764/0.898. That green deficit is the visible magenta cast in neutral
stone and white UI elements. The RGB selector and three-level vertical slot
pattern remain, but every row now applies the same strength to R, G and B.
Mean transmission is exactly 1.203125 in each channel.

A single monotone LUT is fitted against the complete neutral response at
240p -> 1080p. The three LUT columns are identical, so the tone correction
cannot introduce a hue shift.

### Royale Kurozumi

The previous two-pixel RG/GB mask selected green on both phases and could not
have equal channel energy. It is replaced by a compact RGB aperture grille
with equal mean transmission of 1.270833 per channel. The former P22-oriented
per-channel LUT is replaced by the same kind of shared neutral LUT used for
Royale.

Kurozumi's very narrow beam also left too little spatial-average luminance for
colour matching. Its adaptive endpoints are blended 65% toward the existing
scan-free table. The remaining 35% retains visible PVM-style scanline
modulation while lifting midtones and highlights into the usable range.

## Limits

An unfiltered white field is 255 at every output pixel. A CRT field containing
dark scanline and mask samples cannot also average 255 because every hardware
stage clamps at 255; brighter phosphor samples cannot carry compensating
headroom. The calibrated white means therefore remain about 217 (Lottes), 229
(Royale), and 221 (Kurozumi). Midtones and chromatic palette entries are the
priority and generally land within a few RGB codes of the default.

## Reproduction and validation

`tools/build_port.py` builds the upstream-derived geometry, then calls
`tools/color_match.py` for the three calibrated families. Kurozumi's bounded
builder retains its checked-in calibrated seed to avoid repeatedly blending
the adaptive table.

`tools/validate_port.py` hard-gates:

- exact preset and runtime-file inventory;
- file format, table count and coefficient ranges;
- equal mean mask energy in R, G and B;
- identical RGB columns in calibrated gamma tables;
- dominant-palette RMSE/MAE, neutral-channel span, black and white level;
- full-frame 224p/240p to 1080p scanline stability.
