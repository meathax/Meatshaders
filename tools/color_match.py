"""Apply the source-preserving colour-accuracy calibration.

The source ports intentionally reproduce each upstream shader's tone and mask
transmission.  That is not the same objective as preserving the source game's
RGB values.  This module keeps the fitted CRT geometry, but maximises mask
throughput and derives one neutral RGB LUT per family against MiSTer's exact
integer pipeline at 240p -> 1080p (4.5x).

WHY THE FIT IS DONE IN LIGHT-LINEAR SPACE
-----------------------------------------
A CRT filter replaces every source pixel with a modulated block of output
pixels (mask phases x scanline phases).  The colour the viewer perceives is the
block's average *light*, not the average of its 8-bit codes.  Because the
display EOTF is convex, bright samples in the block emit disproportionately
more light than their codes suggest:

    mean(EOTF(code)) > EOTF(mean(code))          (Jensen)

Matching the code-space mean therefore leaves the image visibly BRIGHTER and
less saturated than the source.  Measured on the previous code-space fit, a
172 grey reached 172 in code space but 184 in perceived light -- a 12-code
error that the old validator could not see, plus a hue shift on saturated
colours because each channel's block sits at a different point on the curve.

Fitting against the light-linear mean removes both.  The result is robust to
the exact display gamma: a curve fitted for sRGB scores 5.36 / 5.37 / 5.49
palette RMSE when judged under sRGB / pure 2.2 / pure 2.4 respectively.

Usage: py -3 tools/color_match.py <guest|lottes|royale|kurozumi|all>
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fileio
import mister_model as mm


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_MARKER = "Source-preserving colour calibration (light-linear)"
LEGACY_MARKER = "Default-reference colour calibration"
# Each V-filter conditioning step carries its own marker so the two are
# independently one-shot.  A single shared flag would let whichever step ran
# first suppress the other for good.
BLEND_MARKER = "scan-free DC blend"
CONTRACT_MARKER = "Adaptive endpoints contracted"
SCALE = 4.5
# 72 = 8 x 9: an exact multiple of the 4.5x vertical phase period (9 output
# lines per 2 source lines) and of every mask dimension in the pack (1, 2, 3,
# 6, 12).  The spatial mean over this extent is therefore a whole number of
# mask and scanline periods, not a windowed approximation.
SAMPLE_EXTENT = 72
KUROZUMI_SCANLINE_BLEND = 0.65
# MiSTer's adaptive V stage picks its beam profile from max(R,G,B) of the whole
# pixel (mister_model F2), so all three channels share coefficients chosen by
# the BRIGHTEST channel.  A saturated colour therefore filters its dark
# channels with a bright-beam profile, and no shared LUT can undo that -- the
# error depends on the colour, not on the code.  Guest's two endpoints differ
# by up to 123 coefficient units, which cost (71,69,218) 8.4 codes in R and G.
# Contracting both endpoints toward their midpoint scales that coupling down
# directly.  Measured: 0% -> 8.4 codes drift, 50% -> 3.9, 100% (fixed beam)
# -> 1.1, while the mid-grey scanline depth stays at 20.8% throughout.  Half
# keeps Guest's adaptive beam recognisable and clears the accuracy gate.
GUEST_ADAPTIVE_CONTRACTION = 0.5

# Dominant, repeatable palette colours measured from the supplied default.webp.
# The validator uses these exact source values; the calibration curve itself is
# fitted over all 256 neutral codes rather than overfitting this small palette.
REFERENCE_PALETTE = np.array([
    (0, 0, 0),
    (172, 172, 172),
    (154, 64, 23),
    (101, 101, 101),
    (57, 8, 124),
    (255, 255, 255),
    (191, 134, 254),
    (71, 69, 218),
    (247, 210, 195),
    (103, 103, 0),
    (70, 35, 0),
], dtype=np.int64)

FAMILIES = {
    "guest": {
        "base": "CRT Guest Advanced (Port)",
        "vertical": "CRT Guest Advanced (Port)_V Adaptive.txt",
        "adaptive": True,
        "gamma": "CRT Guest Advanced (Port).txt",
        # The V beam, not the mask, sets Guest's highlight ceiling: every
        # selected multiplier at or above 20/16 reaches the same 238-code
        # white.  20/16 is therefore the strongest useful setting, and it
        # keeps the finest gamma quantisation (217 unique levels, max step 4)
        # and the smallest departure from the source mask.
        "mask_selected": 20,
        "mask_other": 15,
        "flatten_mask": False,
        "contract_adaptive": GUEST_ADAPTIVE_CONTRACTION,
    },
    "lottes": {
        "base": "CRT Lottes (Port)",
        "vertical": "CRT Lottes (Port)_V.txt",
        "adaptive": False,
        "gamma": "CRT Lottes (Port).txt",
        "mask_selected": 31,
        "mask_other": 15,
        "flatten_mask": False,
    },
    "royale": {
        "base": "CRT Royale (Port)",
        "vertical": "CRT Royale (Port)_V Adaptive.txt",
        "adaptive": True,
        "gamma": "CRT Royale (Port).txt",
        "mask_selected": 31,
        "mask_other": 15,
        # Royale is the one family that cannot take an endpoint contraction.
        # Its beam is the most strongly adaptive in the pack (endpoints differ
        # by up to 153 units), so contracting it drives 240p -> 1080p moire
        # straight through the 7.65-code limit: 0% -> 7.00, 30% -> 8.00,
        # 50% -> 10.00.  Uneven scanline thickness is a worse artefact than the
        # residual it would buy back (worst hue tilt 10.5 -> 8.7 codes), so
        # Royale keeps its full adaptive beam and accepts the tilt.
        # Royale's rows previously varied only between selected 29/30/31 --
        # a +/-2/16 vertical strength ripple on identical horizontal triads,
        # not a true offset slot.  Flattening it to full throughput costs no
        # visible structure and buys 10 codes of white plus a halved palette
        # error (RMSE 9.13 -> 5.36).  Scanlines still supply vertical
        # modulation; the RGB triads still supply the phosphor structure.
        "flatten_mask": True,
    },
    "kurozumi": {
        "base": "CRT Royale Kurozumi (Port)",
        "vertical": "CRT Royale Kurozumi (Port)_V Adaptive.txt",
        "adaptive": True,
        "gamma": "CRT Royale Kurozumi (Port).txt",
        "mask_selected": 31,
        "mask_other": 15,
        "flatten_mask": False,
        # Unlike Royale, Kurozumi's gentler endpoints (max gap 90) contract
        # without a moire penalty -- it actually falls, 2.50 -> 2.31 -- and
        # worst hue tilt drops 7.0 -> 2.9 codes for 1 code of white.
        "contract_adaptive": 0.3,
    },
}


# --------------------------------------------------------------- colour space

def srgb_to_linear(code: np.ndarray) -> np.ndarray:
    """sRGB EOTF: 8-bit code -> relative light (0..1)."""
    c = np.asarray(code, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(light: np.ndarray) -> np.ndarray:
    """Inverse sRGB EOTF: relative light -> 8-bit code (unrounded)."""
    l = np.clip(np.asarray(light, dtype=np.float64), 0.0, 1.0)
    return np.where(l <= 0.0031308, l * 12.92,
                    1.055 * l ** (1.0 / 2.4) - 0.055) * 255.0


def identity_lut() -> np.ndarray:
    return np.repeat(np.arange(256, dtype=np.int64)[:, None], 3, axis=1)


def is_calibrated(key: str) -> bool:
    spec = FAMILIES[key]
    mask_path = os.path.join(ROOT, "Shadow_Masks", f"{spec['base']}.txt")
    if not os.path.isfile(mask_path):
        return False
    return any(CALIBRATION_MARKER in line
               for line in fileio.parse_mask(mask_path).header)


# ------------------------------------------------------------------ mask

def balanced_mask(key: str, source: fileio.MaskFile) -> fileio.MaskFile:
    """Raise the mask to its target throughput, preserving cell geometry.

    Only the per-token strengths change; every token keeps its channel
    bitmask, so each family's phosphor layout, period and orientation survive
    untouched.  The v2 format caps non-selected channels at 15/16, so some
    attenuation is unavoidable and the highlight ceiling stays below 255.
    """
    spec = FAMILIES[key]
    selected, other = spec["mask_selected"], spec["mask_other"]

    tokens = []
    for row in source.tokens:
        new_row = []
        for token in row:
            bits, old_selected, old_other = fileio.decode_mask_token(token)
            if spec["flatten_mask"]:
                use_selected, use_other = selected, other
            else:
                # Preserve any intentional per-token ripple by carrying the
                # source's relative offset below the target strength.
                ripple = max(source_max_selected(source) - old_selected, 0)
                use_selected = max(selected - ripple, 16)
                use_other = other
            new_row.append(
                fileio.encode_mask_token(bits, use_selected, use_other))
        tokens.append(new_row)

    mask = fileio.MaskFile(
        [f"Name: {spec['base']}", CALIBRATION_MARKER,
         f"Equal mean R/G/B energy at {selected}/16 selected, {other}/16 other",
         "Fitted against the light-linear spatial mean at 240p -> 1080p"],
        len(tokens[0]), len(tokens), tokens)

    energy = mask.multipliers().mean(axis=(0, 1))
    if float(np.ptp(energy)) > 1.0e-12:
        raise RuntimeError(
            f"{key}: mask channel energy is unequal {energy.tolist()}; a "
            "neutral shared LUT cannot correct a per-channel tint")
    return mask


# parse_filter collects every '#' line into one header list, including the
# per-set labels that write_filter emits between the coefficient blocks.  Left
# alone they migrate into the header on each rewrite and get duplicated by the
# next set_comments.  Strip them so a written file is canonical regardless of
# how many times the calibration has been re-applied.
SET_COMMENTS = ("Primary coefficients (maximum RGB = 0)",
                "Secondary coefficients (maximum RGB = 255)")
_STALE_SET_COMMENTS = SET_COMMENTS + ("Dark endpoint", "Bright endpoint")


def clean_header(header: list[str]) -> list[str]:
    return [line for line in header if line not in _STALE_SET_COMMENTS]


def write_conditioned_filter(path: str, source: fileio.FilterFile,
                             sets: list[np.ndarray], notes: list[str]) -> fileio.FilterFile:
    """Write a re-conditioned V filter with a canonical, non-accumulating header."""
    conditioned = fileio.FilterFile(
        clean_header(source.header) + notes,
        source.adaptive, source.tenbit, sets)
    fileio.write_filter(path, conditioned, set_comments=list(SET_COMMENTS))
    return conditioned


def source_max_selected(source: fileio.MaskFile) -> int:
    return max(fileio.decode_mask_token(token)[1]
               for row in source.tokens for token in row)


# ------------------------------------------------------------- simulation

class FieldSimulator:
    """Exact MiSTer pipeline for a uniform source field at 240p -> 1080p.

    Caches the vertical stage by (horizontal value, adaptive control), which
    for a flat field collapses 72 columns to a handful of distinct cases.
    """

    def __init__(self, h: np.ndarray, dark: np.ndarray, bright: np.ndarray,
                 adaptive: bool, mask: fileio.MaskFile,
                 extent: int = SAMPLE_EXTENT, scale: float = SCALE):
        if extent % mask.width or extent % mask.height:
            raise ValueError(
                f"extent {extent} is not a whole number of {mask.width}x"
                f"{mask.height} mask periods; the spatial mean would be biased")
        self.h, self.dark, self.bright = h, dark, bright
        self.adaptive = adaptive
        self.extent = extent
        self.positions = (np.arange(extent) + 0.5) / scale - 0.5 + 8.0
        mult16 = np.rint(mask.multipliers() * 16.0).astype(np.int64)
        self.tile = np.tile(mult16, (extent // mask.height,
                                     extent // mask.width, 1))
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def _vertical(self, value: int, control: int) -> np.ndarray:
        key = (value, control)
        cached = self._cache.get(key)
        if cached is None:
            lines = np.full(64, value, dtype=np.int64)
            if self.adaptive:
                cached = mm.fir_1d_adaptive(
                    lines, self.dark, self.bright, self.positions,
                    np.full(self.extent, control, dtype=np.int64))
            else:
                cached = mm.fir_1d(lines, self.dark, self.positions)
            self._cache[key] = cached
        return cached

    def field(self, rgb, lut: np.ndarray) -> np.ndarray:
        """Full (extent, extent, 3) output block for one uniform source colour."""
        rgb = np.asarray(rgb, dtype=np.int64)
        horizontal = np.stack([
            mm.fir_1d(np.full(64, int(lut[int(code), channel]), dtype=np.int64),
                      self.h, self.positions)
            for channel, code in enumerate(rgb)], axis=1)

        out = np.empty((self.extent, self.extent, 3), dtype=np.int64)
        for x in range(self.extent):
            control = int(horizontal[x].max())
            for channel in range(3):
                out[:, x, channel] = self._vertical(
                    int(horizontal[x, channel]), control)
        return mm.mask_multiply(out, self.tile)

    def perceived(self, rgb, lut: np.ndarray) -> np.ndarray:
        """Perceived RGB: spatial mean in light, expressed back as 8-bit codes."""
        return linear_to_srgb(srgb_to_linear(self.field(rgb, lut)).mean(axis=(0, 1)))

    def code_mean(self, rgb, lut: np.ndarray) -> np.ndarray:
        """Spatial mean in 8-bit code space (reported for comparison only)."""
        return self.field(rgb, lut).mean(axis=(0, 1))


def simulate_mean(rgb, h: np.ndarray, dark: np.ndarray, bright: np.ndarray,
                  adaptive: bool, mask: fileio.MaskFile, lut: np.ndarray,
                  scale: float = SCALE,
                  extent: int = SAMPLE_EXTENT) -> np.ndarray:
    """Perceived (light-linear) spatial mean of a uniform source field."""
    sim = FieldSimulator(h, dark, bright, adaptive, mask, extent, scale)
    return sim.perceived(rgb, lut)


# ------------------------------------------------------------------ fitting

def fit_neutral_lut(h: np.ndarray, dark: np.ndarray, bright: np.ndarray,
                    adaptive: bool, mask: fileio.MaskFile) -> np.ndarray:
    """Invert the measured neutral light response with one shared monotone LUT.

    A single shared RGB curve is sufficient -- and is the only tint-free
    option -- because every mask in the pack presents the same multiset of
    multipliers to R, G and B over one tile.  Each channel therefore has an
    identical transfer function, so correcting the neutral axis corrects all
    three channels of a coloured pixel too.
    """
    sim = FieldSimulator(h, dark, bright, adaptive, mask)
    raw = identity_lut()
    response = np.array([
        srgb_to_linear(sim.field((code, code, code), raw)).mean()
        for code in range(256)])
    target = srgb_to_linear(np.arange(256))
    inverse = np.array([
        int(np.argmin(np.abs(response - value))) for value in target],
        dtype=np.int64)
    inverse = np.maximum.accumulate(inverse)
    inverse[0] = 0
    return np.repeat(inverse[:, None], 3, axis=1)


# ------------------------------------------------------------------ apply

def apply(key: str) -> None:
    if key not in FAMILIES:
        raise KeyError(key)
    spec = FAMILIES[key]
    base = spec["base"]
    h_path = os.path.join(ROOT, "Filters", f"{base}_H.txt")
    v_path = os.path.join(ROOT, "Filters", spec["vertical"])
    mask_path = os.path.join(ROOT, "Shadow_Masks", f"{base}.txt")

    h = fileio.parse_filter(h_path).sets[0]
    vertical = fileio.parse_filter(v_path)
    mask = balanced_mask(key, fileio.parse_mask(mask_path))

    # Each V conditioning step is one-shot: its own marker in the written
    # header is the guard, so re-running never compounds a blend or a
    # contraction.  The legacy marker also counts for the blend, because the
    # first calibrated release recorded it under that name.
    def conditioned(marker: str) -> bool:
        return any(marker in line or LEGACY_MARKER in line
                   for line in vertical.header) if marker == BLEND_MARKER else any(
            marker in line for line in vertical.header)

    if key == "kurozumi" and not conditioned(BLEND_MARKER):
        no_scan_path = os.path.join(
            ROOT, "Filters", f"{base}_V No Scanlines.txt")
        no_scan = fileio.parse_filter(no_scan_path).sets[0]
        blend = KUROZUMI_SCANLINE_BLEND
        sets = [np.rint((1.0 - blend) * table + blend * no_scan).astype(np.int64)
                for table in vertical.sets]
        vertical = write_conditioned_filter(
            v_path, vertical, sets,
            [CALIBRATION_MARKER,
             f"{blend:.0%} {BLEND_MARKER} for colour/luminance parity"])

    contraction = spec.get("contract_adaptive")
    if contraction and not conditioned(CONTRACT_MARKER):
        if not vertical.adaptive or len(vertical.sets) != 2:
            raise RuntimeError(
                f"{key}: contract_adaptive needs a two-endpoint adaptive V")
        midpoint = 0.5 * (vertical.sets[0].astype(np.float64)
                          + vertical.sets[1].astype(np.float64))
        sets = [np.rint((1.0 - contraction) * table + contraction * midpoint
                        ).astype(np.int64) for table in vertical.sets]
        vertical = write_conditioned_filter(
            v_path, vertical, sets,
            [f"{CONTRACT_MARKER} {contraction:.0%} toward their midpoint: "
             "the shared max(R,G,B) beam control otherwise filters a "
             "saturated colour's dark channels with a bright beam"])

    fileio.write_mask(mask_path, mask)

    lut = fit_neutral_lut(
        h, vertical.dark, vertical.bright, bool(spec["adaptive"]), mask)
    gamma_path = os.path.join(ROOT, "Gamma", spec["gamma"])
    fileio.write_gamma(gamma_path, lut, header=[
        f"Name: {base}", CALIBRATION_MARKER,
        "One shared monotone RGB curve; no channel tint",
        "Inverts the light-linear 240p-to-1080p spatial mean of this",
        "family's exact mask + scanline block (see tools/color_match.py)",
    ])

    print(f"[{key}] applied light-linear colour calibration")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in (*FAMILIES, "all"):
        choices = "|".join((*FAMILIES, "all"))
        sys.exit(f"Usage: py -3 color_match.py <{choices}>")
    selected = FAMILIES if sys.argv[1] == "all" else (sys.argv[1],)
    for family in selected:
        apply(family)
