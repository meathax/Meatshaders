"""Repository-level contracts for the complete MiSTer preset collection.

The generic syntax checks live in fileio.py.  This module checks relationships
that Main cannot infer from one file in isolation: portrait axis rotation,
interlace fallback selection, one-adaptive-axis limits, gamma/filter pairings,
mask modes, and the required variant matrix.
"""

from __future__ import annotations

import glob
import os

import fileio


EXPECTED_VARIANTS = {
    "CRT Easymode": {
        "400p+ No Scanlines",
        "400p+ No Scanlines TATE",
        "Adaptive",
        "Adaptive 4K Visual Pitch",
        "Balanced",
        "Balanced TATE",
        "Conventional",
        "Highlights",
        "Pixel-Art Anti-Ring",
        "Pixel-Art Anti-Ring TATE",
        "TATE",
    },
    "CRT Easymode v5": {
        "Default", "Edge Stable", "Fixed Compatibility", "No Gamma", "TATE"},
    "CRT Guest Advanced": {
        "Default", "Edge Stable", "Fixed Compatibility", "No Gamma", "TATE"},
    "CRT Lottes": {
        "Aperture",
        "Crisp No Bloom",
        "Default",
        "Default 4K Visual Pitch",
        "Fixed Compatibility",
        "Fixed TATE",
        "TATE",
        "VGA",
    },
    "CRT Royale": {
        "Default", "Edge Stable", "Fixed Compatibility", "No Gamma", "TATE"},
    "CRT Royale Kurozumi": {
        "Anti-Moire",
        "Default",
        "Fixed Compatibility",
        "No Gamma",
        "Perceptual Dither",
        "TATE",
    },
}

ADAPTIVE_VARIANTS = {
    "CRT Easymode": {
        "Adaptive",
        "Adaptive 4K Visual Pitch",
        "Pixel-Art Anti-Ring",
        "Pixel-Art Anti-Ring TATE",
        "TATE",
    },
    "CRT Easymode v5": {"Default", "Edge Stable", "No Gamma", "TATE"},
    "CRT Guest Advanced": {"Default", "Edge Stable", "No Gamma", "TATE"},
    "CRT Lottes": {"Aperture", "Default", "Default 4K Visual Pitch", "TATE", "VGA"},
    "CRT Royale": {"Default", "Edge Stable", "No Gamma", "TATE"},
    "CRT Royale Kurozumi": {
        "Anti-Moire", "Default", "No Gamma", "Perceptual Dither", "TATE"},
}

V5_FAMILIES = {
    "CRT Easymode v5",
    "CRT Guest Advanced",
    "CRT Royale",
    "CRT Royale Kurozumi",
}


def _split_name(filename: str) -> tuple[str, str] | None:
    stem = os.path.splitext(filename)[0]
    for family in sorted(EXPECTED_VARIANTS, key=len, reverse=True):
        prefix = family + " - "
        if stem.startswith(prefix):
            return family, stem[len(prefix):]
    return None


def _expected_mode(variant: str) -> str:
    if variant.endswith("TATE"):
        return "1x rotated"
    if "4K Visual Pitch" in variant:
        return "2x"
    return "1x"


def _belongs_to_family(family: str, filename: str) -> bool:
    if family == "CRT Easymode":
        prefixes = ("CRT Easymode (Port)", "CRT Easymode Pixel-Art")
    elif family == "CRT Lottes":
        prefixes = ("CRT Lottes (Port)", "CRT Lottes Crisp (Port)")
    else:
        prefixes = (f"{family} (Port)",)
    return filename.startswith(prefixes)


def _filter_contract(family: str, variant: str) -> dict[str, str]:
    """Return the exact axis and interlace tables selected by a preset."""
    if family in V5_FAMILIES:
        base = f"{family} (Port)"
        h = f"{base}_H.txt"
        adaptive = f"{base}_V Adaptive.txt"
        no_scan = f"{base}_V No Scanlines.txt"
        if variant == "Fixed Compatibility":
            return {"hfilter": h, "vfilter": f"{base}_V Fixed.txt", "ifilter": no_scan}
        if variant == "No Gamma":
            return {"hfilter": h,
                    "vfilter": f"{base}_V Adaptive No Gamma.txt",
                    "ifilter": no_scan}
        if variant == "Edge Stable":
            return {"hfilter": h,
                    "vfilter": f"{base}_V Adaptive Edge Stable.txt",
                    "ifilter": no_scan}
        if variant == "TATE":
            return {"hfilter": adaptive, "vfilter": h, "ifilter": h}
        if family == "CRT Royale Kurozumi" and variant == "Anti-Moire":
            adaptive = f"{base}_V Adaptive Anti-Moire.txt"
        return {"hfilter": h, "vfilter": adaptive, "ifilter": no_scan}

    if family == "CRT Lottes":
        h = "CRT Lottes (Port)_H.txt"
        adaptive = "CRT Lottes (Port)_V Adaptive.txt"
        fixed = "CRT Lottes (Port)_V.txt"
        if variant == "Crisp No Bloom":
            crisp_h = "CRT Lottes Crisp (Port)_H.txt"
            crisp_v = "CRT Lottes Crisp (Port)_V.txt"
            return {"hfilter": crisp_h, "vfilter": crisp_v, "ifilter": crisp_v}
        if variant == "Fixed Compatibility":
            return {"hfilter": h, "vfilter": fixed, "ifilter": fixed}
        if variant == "Fixed TATE":
            return {"hfilter": fixed, "vfilter": h, "ifilter": h}
        if variant == "TATE":
            return {"hfilter": adaptive, "vfilter": h, "ifilter": h}
        return {"hfilter": h, "vfilter": adaptive, "ifilter": adaptive}

    h = "CRT Easymode (Port)_H.txt"
    no_scan = "CRT Easymode (Port)_V No Scanlines Matched.txt"
    selections = {
        "400p+ No Scanlines": (h, no_scan, no_scan),
        "400p+ No Scanlines TATE": (
            "CRT Easymode (Port)_V No Scanlines.txt",
            "CRT Easymode (Port)_H Matched Boost.txt",
            "CRT Easymode (Port)_H Matched Boost.txt",
        ),
        "Adaptive": (h, "CRT Easymode (Port)_V Adaptive.txt", no_scan),
        "Adaptive 4K Visual Pitch": (
            h, "CRT Easymode (Port)_V Adaptive.txt", no_scan),
        "Balanced": (h, "CRT Easymode (Port)_V Balanced Matched.txt", no_scan),
        "Balanced TATE": (
            "CRT Easymode (Port)_V Balanced.txt",
            "CRT Easymode (Port)_H Matched Boost.txt",
            "CRT Easymode (Port)_H Matched Boost.txt",
        ),
        "Conventional": (
            h,
            "CRT Easymode (Port)_V Balanced.txt",
            "CRT Easymode (Port)_V No Scanlines.txt",
        ),
        "Highlights": (h, "CRT Easymode (Port)_V Highlights.txt", no_scan),
        "Pixel-Art Anti-Ring": (
            "CRT Easymode Pixel-Art Anti-Ring (Port)_H.txt",
            "CRT Easymode (Port)_V Adaptive.txt",
            no_scan,
        ),
        "Pixel-Art Anti-Ring TATE": (
            "CRT Easymode (Port)_V Adaptive TATE.txt",
            "CRT Easymode Pixel-Art Anti-Ring Matched Boost (Port)_H.txt",
            "CRT Easymode Pixel-Art Anti-Ring Matched Boost (Port)_H.txt",
        ),
        "TATE": (
            "CRT Easymode (Port)_V Adaptive TATE.txt",
            "CRT Easymode (Port)_H Matched Boost.txt",
            "CRT Easymode (Port)_H Matched Boost.txt",
        ),
    }
    expected = selections[variant]
    return dict(zip(("hfilter", "vfilter", "ifilter"), expected))


def _pairing_contract(family: str, variant: str) -> tuple[str, str]:
    """Return the exact expected (gamma, mask) pairing."""
    if family in V5_FAMILIES:
        if variant == "Fixed Compatibility":
            gamma = f"{family} (Port) Fixed.txt"
            mask = f"{family} (Port) Fixed.txt"
        elif variant == "No Gamma":
            gamma = "off"
            mask = f"{family} (Port) No Gamma.txt"
        elif variant == "Perceptual Dither":
            gamma = f"{family} (Port).txt"
            mask = f"{family} (Port) Perceptual Dither.txt"
        else:
            gamma = f"{family} (Port).txt"
            mask = f"{family} (Port).txt"
        return gamma, mask

    if family == "CRT Lottes":
        masks = {
            "Aperture": "CRT Lottes Matched Mask2 Aperture (Port).txt",
            "Crisp No Bloom": "CRT Lottes Mask3 StretchedVGA (Port).txt",
            "VGA": "CRT Lottes Matched Mask4 VGA (Port).txt",
        }
        return "off", masks.get(variant, "CRT Lottes Matched Mask3 StretchedVGA (Port).txt")

    if variant == "Conventional":
        return "CRT Easymode (Port).txt", "CRT Easymode (Port).txt"
    return "CRT Easymode Matched (Port).txt", "CRT Easymode Matched Boost (Port).txt"


def validate_entries(entries: dict[str, str], root: str, filename: str,
                     family: str, variant: str) -> list[str]:
    """Validate one parsed preset against the pack's semantic contracts."""
    problems = fileio.validate_preset(entries, root, filename)
    if any(key not in entries for key in fileio.PRESET_KEYS):
        return problems

    def problem(message: str) -> None:
        problems.append(f"{filename}: {message}")

    hfilter = entries["hfilter"]
    vfilter = entries["vfilter"]
    ifilter = entries["ifilter"]
    is_tate = variant.endswith("TATE")

    if entries["sfilter"].lower() != "same":
        problem("sfilter must be explicitly disabled with 'same'")
    if ifilter.lower() in {"same", "off", "none"}:
        problem("ifilter must name an explicit interlace-safe filter")

    expected_mode = _expected_mode(variant)
    if entries["maskmode"].lower() != expected_mode:
        problem(f"maskmode must be {expected_mode!r}, got {entries['maskmode']!r}")

    for key in ("hfilter", "vfilter", "ifilter"):
        value = entries[key]
        if value.lower() not in {"same", "off", "none"} and not _belongs_to_family(
                family, value):
            problem(f"{key} crosses shader families: {value!r}")

    expected_filters = _filter_contract(family, variant)
    for key, expected in expected_filters.items():
        if entries[key] != expected:
            problem(f"{key} pairing must be {expected!r}, got {entries[key]!r}")

    if is_tate:
        if "_V" not in hfilter:
            problem(f"TATE hfilter must carry the source vertical/scan axis, got {hfilter!r}")
        if "_H" not in vfilter:
            problem(f"TATE vfilter must carry the source horizontal/last axis, got {vfilter!r}")
        if ifilter != vfilter:
            problem("TATE ifilter must equal vfilter so interlace keeps the rotated last axis")
    else:
        if "_H" not in hfilter:
            problem(f"landscape hfilter must carry the source horizontal axis, got {hfilter!r}")
        if "_V" not in vfilter:
            problem(f"landscape vfilter must carry the source vertical/scan axis, got {vfilter!r}")
        if family == "CRT Lottes":
            if ifilter != vfilter:
                problem("Lottes landscape ifilter must equal its fitted vfilter")
        elif "No Scanlines" not in ifilter:
            problem("landscape ifilter must use the family's No Scanlines fallback")

    adaptive = {}
    for key, value in (("hfilter", hfilter), ("vfilter", vfilter)):
        path = os.path.join(root, "Filters", value)
        if not os.path.isfile(path):
            continue
        try:
            adaptive[key] = fileio.parse_filter(path).adaptive
        except Exception as exc:
            problem(f"cannot inspect {key} adaptation: {exc}")

    if len(adaptive) == 2:
        expect_adaptive = variant in ADAPTIVE_VARIANTS[family]
        h_adaptive, v_adaptive = adaptive["hfilter"], adaptive["vfilter"]
        if h_adaptive and v_adaptive:
            problem("both scaler axes are adaptive; Main can upload only one adaptive table")
        if expect_adaptive:
            expected_axis_ok = (h_adaptive and not v_adaptive) if is_tate else (
                v_adaptive and not h_adaptive)
            if not expected_axis_ok:
                axis = "hfilter" if is_tate else "vfilter"
                problem(f"maximum-fidelity variant requires adaptation in {axis} only")
        elif h_adaptive or v_adaptive:
            problem("fixed/compatibility variant unexpectedly references an adaptive table")

    expected_gamma, expected_mask = _pairing_contract(family, variant)
    if entries["gamma"] != expected_gamma:
        problem(f"gamma pairing must be {expected_gamma!r}, got {entries['gamma']!r}")
    if entries["mask"] != expected_mask:
        problem(f"mask pairing must be {expected_mask!r}, got {entries['mask']!r}")
    return problems


def _same_fields(problems: list[str], filename: str, left: dict[str, str],
                 right: dict[str, str], keys: tuple[str, ...], relationship: str) -> None:
    for key in keys:
        if left[key] != right[key]:
            problems.append(f"{filename}: {relationship} must preserve {key}")


def _validate_relations(presets: dict[tuple[str, str], dict[str, str]]) -> list[str]:
    """Check paired variants so independently valid files cannot drift apart."""
    problems = []
    for family in V5_FAMILIES:
        default = presets[(family, "Default")]
        no_gamma = presets[(family, "No Gamma")]
        fixed = presets[(family, "Fixed Compatibility")]
        tate = presets[(family, "TATE")]
        _same_fields(problems, f"{family} - No Gamma.ini", default, no_gamma,
                     ("hfilter", "sfilter", "ifilter", "maskmode"),
                     "No Gamma vs Default")
        _same_fields(problems, f"{family} - Fixed Compatibility.ini", default, fixed,
                     ("hfilter", "sfilter", "ifilter", "maskmode"),
                     "Fixed Compatibility vs Default")
        if fixed["vfilter"] == default["vfilter"]:
            problems.append(f"{family} - Fixed Compatibility.ini: "
                            "fixed vfilter must differ from Default")
        if tate["hfilter"] != default["vfilter"] or tate["vfilter"] != default["hfilter"]:
            problems.append(f"{family} - TATE.ini: TATE must transpose Default hfilter/vfilter")
        _same_fields(problems, f"{family} - TATE.ini", default, tate,
                     ("sfilter", "gamma", "mask"), "TATE vs Default")

        if "Edge Stable" in EXPECTED_VARIANTS[family]:
            edge = presets[(family, "Edge Stable")]
            _same_fields(problems, f"{family} - Edge Stable.ini", default, edge,
                         ("hfilter", "sfilter", "ifilter", "gamma", "mask", "maskmode"),
                         "Edge Stable vs Default")
            if edge["vfilter"] == default["vfilter"]:
                problems.append(f"{family} - Edge Stable.ini: stabilized vfilter is not selected")

    kurozumi_default = presets[("CRT Royale Kurozumi", "Default")]
    anti_moire = presets[("CRT Royale Kurozumi", "Anti-Moire")]
    _same_fields(problems, "CRT Royale Kurozumi - Anti-Moire.ini",
                 kurozumi_default, anti_moire,
                 ("hfilter", "sfilter", "ifilter", "gamma", "mask", "maskmode"),
                 "Anti-Moire vs Default")
    if anti_moire["vfilter"] == kurozumi_default["vfilter"]:
        problems.append("CRT Royale Kurozumi - Anti-Moire.ini: softened vfilter is not selected")
    perceptual = presets[("CRT Royale Kurozumi", "Perceptual Dither")]
    _same_fields(problems, "CRT Royale Kurozumi - Perceptual Dither.ini",
                 kurozumi_default, perceptual,
                 ("hfilter", "vfilter", "sfilter", "ifilter", "gamma", "maskmode"),
                 "Perceptual Dither vs Default")
    if perceptual["mask"] == kurozumi_default["mask"]:
        problems.append("CRT Royale Kurozumi - Perceptual Dither.ini: dither mask is not selected")

    lottes_default = presets[("CRT Lottes", "Default")]
    lottes_fixed = presets[("CRT Lottes", "Fixed Compatibility")]
    lottes_tate = presets[("CRT Lottes", "TATE")]
    lottes_fixed_tate = presets[("CRT Lottes", "Fixed TATE")]
    _same_fields(problems, "CRT Lottes - Fixed Compatibility.ini",
                 lottes_default, lottes_fixed,
                 ("hfilter", "sfilter", "gamma", "mask", "maskmode"),
                 "Fixed Compatibility vs Default")
    if lottes_tate["hfilter"] != lottes_default["vfilter"] or \
            lottes_tate["vfilter"] != lottes_default["hfilter"]:
        problems.append("CRT Lottes - TATE.ini: TATE must transpose Default hfilter/vfilter")
    if lottes_fixed_tate["hfilter"] != lottes_fixed["vfilter"] or \
            lottes_fixed_tate["vfilter"] != lottes_fixed["hfilter"]:
        problems.append("CRT Lottes - Fixed TATE.ini: Fixed TATE must "
                        "transpose Fixed Compatibility")

    easymode = presets[("CRT Easymode", "Adaptive")]
    easymode_4k = presets[("CRT Easymode", "Adaptive 4K Visual Pitch")]
    _same_fields(problems, "CRT Easymode - Adaptive 4K Visual Pitch.ini",
                 easymode, easymode_4k,
                 ("hfilter", "vfilter", "sfilter", "ifilter", "gamma", "mask"),
                 "4K Visual Pitch vs Adaptive")
    easymode_pixel = presets[("CRT Easymode", "Pixel-Art Anti-Ring")]
    _same_fields(problems, "CRT Easymode - Pixel-Art Anti-Ring.ini",
                 easymode, easymode_pixel,
                 ("vfilter", "sfilter", "ifilter", "gamma", "mask", "maskmode"),
                 "Pixel-Art Anti-Ring vs Adaptive")
    return problems


def validate_collection(root: str) -> list[str]:
    """Validate every Presets/*.ini file and the collection as a whole."""
    preset_dir = os.path.join(root, "Presets")
    paths = sorted(glob.glob(os.path.join(preset_dir, "*.ini")))
    expected_names = {
        f"{family} - {variant}.ini"
        for family, variants in EXPECTED_VARIANTS.items()
        for variant in variants
    }
    actual_names = {os.path.basename(path) for path in paths}
    problems = [f"Presets/: missing required preset {name!r}"
                for name in sorted(expected_names - actual_names)]
    problems += [f"Presets/: preset has no semantic contract {name!r}"
                 for name in sorted(actual_names - expected_names)]

    parsed: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        filename = os.path.basename(path)
        split = _split_name(filename)
        if split is None or filename not in expected_names:
            continue
        family, variant = split
        try:
            entries = fileio.parse_preset(path)
        except Exception as exc:
            problems.append(f"{filename}: EXCEPTION {exc}")
            continue
        parsed[(family, variant)] = entries
        problems += validate_entries(entries, root, filename, family, variant)

    if len(parsed) == len(expected_names):
        problems += _validate_relations(parsed)
    return problems
