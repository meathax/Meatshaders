"""Semantic and inventory contracts for the canonical MiSTer CRT pack.

Each of the five source shaders ships exactly two user-facing presets: a bare
landscape name and a ``- TATE`` transpose.  The preset files share one
maximum-fidelity pipeline per family; the extra files referenced by a preset
are hardware stages, not selectable shader variants.
"""

from __future__ import annotations

import glob
import os

import fileio


FAMILY_SPECS: dict[str, dict[str, str]] = {
    "CRT Easymode": {
        "file_base": "CRT Easymode (Port)",
        "vfilter": "CRT Easymode (Port)_V Adaptive.txt",
        "gamma": "CRT Easymode (Port).txt",
        "mask": "CRT Easymode (Port).txt",
        "ifilter": "CRT Easymode (Port)_V No Scanlines.txt",
    },
    "CRT Guest Advanced": {
        "file_base": "CRT Guest Advanced (Port)",
        "vfilter": "CRT Guest Advanced (Port)_V Adaptive.txt",
        "gamma": "CRT Guest Advanced (Port).txt",
        "mask": "CRT Guest Advanced (Port).txt",
        "ifilter": "CRT Guest Advanced (Port)_V No Scanlines.txt",
    },
    # The fast variant is upstream's performance rewrite of the identical look;
    # its fixed-pipeline projection equals the advanced one at defaults, so its
    # tables are numerically identical.  It still ships as a self-contained
    # family so each preset's stage files can be copied independently.
    "CRT Guest Advanced Fast": {
        "file_base": "CRT Guest Advanced Fast (Port)",
        "vfilter": "CRT Guest Advanced Fast (Port)_V Adaptive.txt",
        "gamma": "CRT Guest Advanced Fast (Port).txt",
        "mask": "CRT Guest Advanced Fast (Port).txt",
        "ifilter": "CRT Guest Advanced Fast (Port)_V No Scanlines.txt",
    },
    "CRT Guest Advanced Fastest": {
        "file_base": "CRT Guest Advanced Fastest (Port)",
        "vfilter": "CRT Guest Advanced Fastest (Port)_V Adaptive.txt",
        "gamma": "CRT Guest Advanced Fastest (Port).txt",
        "mask": "CRT Guest Advanced Fastest (Port).txt",
        "ifilter": "CRT Guest Advanced Fastest (Port)_V No Scanlines.txt",
    },
    "CRT Lottes": {
        "file_base": "CRT Lottes (Port)",
        "vfilter": "CRT Lottes (Port)_V.txt",
        "gamma": "off",
        "mask": "CRT Lottes (Port).txt",
        # Lottes' brightness-independent beam fits one fixed table more closely
        # than the older adaptive pair.  That canonical V is also interlace-safe.
        "ifilter": "CRT Lottes (Port)_V.txt",
    },
    "CRT Royale": {
        "file_base": "CRT Royale (Port)",
        "vfilter": "CRT Royale (Port)_V Adaptive.txt",
        "gamma": "CRT Royale (Port).txt",
        "mask": "CRT Royale (Port).txt",
        "ifilter": "CRT Royale (Port)_V No Scanlines.txt",
    },
    "CRT Royale Kurozumi": {
        "file_base": "CRT Royale Kurozumi (Port)",
        "vfilter": "CRT Royale Kurozumi (Port)_V Adaptive.txt",
        "gamma": "CRT Royale Kurozumi (Port).txt",
        "mask": "CRT Royale Kurozumi (Port).txt",
        "ifilter": "CRT Royale Kurozumi (Port)_V No Scanlines.txt",
    },
}

# Kept as a small public summary for validators and self-tests.
EXPECTED_VARIANTS = {family: {"Default", "TATE"} for family in FAMILY_SPECS}


def _preset_filename(family: str, variant: str) -> str:
    if variant == "Default":
        return f"{family}.ini"
    if variant == "TATE":
        return f"{family} - TATE.ini"
    raise ValueError(f"unknown canonical variant {variant!r}")


EXPECTED_PRESETS = frozenset(
    _preset_filename(family, variant)
    for family, variants in EXPECTED_VARIANTS.items()
    for variant in variants
)


def _family_filter_names(spec: dict[str, str]) -> set[str]:
    base = spec["file_base"]
    names = {f"{base}_H.txt", spec["vfilter"]}
    if spec["ifilter"].endswith("_V No Scanlines.txt"):
        names.add(spec["ifilter"])
    return names


# This is deliberately an exact allow-list.  A count-only check could let a
# deleted compatibility variant replace a missing canonical stage unnoticed.
PACK_MANIFEST: dict[str, frozenset[str]] = {
    "Filters": frozenset(
        name for spec in FAMILY_SPECS.values()
        for name in _family_filter_names(spec)
    ),
    "Gamma": frozenset(
        spec["gamma"] for spec in FAMILY_SPECS.values()
        if spec["gamma"] != "off"
    ),
    "Shadow_Masks": frozenset(spec["mask"] for spec in FAMILY_SPECS.values()),
    "Presets": EXPECTED_PRESETS,
}


def _split_name(filename: str) -> tuple[str, str] | None:
    """Map one canonical preset filename to ``(family, variant)``."""
    if os.path.splitext(filename)[1].lower() != ".ini":
        return None
    stem = os.path.splitext(filename)[0]
    for family in FAMILY_SPECS:
        if stem == family:
            return family, "Default"
        if stem == f"{family} - TATE":
            return family, "TATE"
    return None


def _filter_contract(family: str, variant: str) -> dict[str, str]:
    spec = FAMILY_SPECS[family]
    base = spec["file_base"]
    horizontal = f"{base}_H.txt"
    canonical_vertical = spec["vfilter"]
    if variant == "TATE":
        return {
            "hfilter": canonical_vertical,
            "vfilter": horizontal,
            "ifilter": horizontal,
        }
    return {
        "hfilter": horizontal,
        "vfilter": canonical_vertical,
        "ifilter": spec["ifilter"],
    }


def validate_entries(entries: dict[str, str], root: str, filename: str,
                     family: str, variant: str) -> list[str]:
    """Validate one parsed preset against its exact canonical pipeline."""
    problems = fileio.validate_preset(entries, root, filename)
    if family not in FAMILY_SPECS:
        problems.append(f"{filename}: unknown shader family {family!r}")
        return problems
    if variant not in EXPECTED_VARIANTS[family]:
        problems.append(f"{filename}: unsupported preset variant {variant!r}")
        return problems
    if any(key not in entries for key in fileio.PRESET_KEYS):
        return problems

    def problem(message: str) -> None:
        problems.append(f"{filename}: {message}")

    spec = FAMILY_SPECS[family]
    is_tate = variant == "TATE"

    if entries["sfilter"].lower() != "same":
        problem("sfilter must be explicitly disabled with 'same'")
    if entries["ifilter"].lower() in {"same", "off", "none"}:
        problem("ifilter must name an explicit interlace-safe filter")

    expected_mode = "1x rotated" if is_tate else "1x"
    if entries["maskmode"].lower() != expected_mode:
        problem(f"maskmode must be {expected_mode!r}, got {entries['maskmode']!r}")

    base = spec["file_base"]
    for key in ("hfilter", "vfilter", "ifilter"):
        value = entries[key]
        if not value.startswith(base):
            problem(f"{key} crosses shader families: {value!r}")

    expected_filters = _filter_contract(family, variant)
    for key, expected in expected_filters.items():
        if entries[key] != expected:
            problem(f"{key} pairing must be {expected!r}, got {entries[key]!r}")

    hfilter, vfilter, ifilter = (
        entries["hfilter"], entries["vfilter"], entries["ifilter"])
    if is_tate:
        if "_V" not in hfilter:
            problem(f"TATE hfilter must carry the source vertical axis, got {hfilter!r}")
        if "_H" not in vfilter:
            problem(f"TATE vfilter must carry the source horizontal axis, got {vfilter!r}")
        if ifilter != vfilter:
            problem("TATE ifilter must equal vfilter so interlace keeps the rotated last axis")
    else:
        if "_H" not in hfilter:
            problem(f"landscape hfilter must carry the source horizontal axis, got {hfilter!r}")
        if "_V" not in vfilter:
            problem(f"landscape vfilter must carry the source vertical axis, got {vfilter!r}")

    adaptive: dict[str, bool] = {}
    for key, value in (("hfilter", hfilter), ("vfilter", vfilter)):
        path = os.path.join(root, "Filters", value)
        if not os.path.isfile(path):
            continue
        try:
            adaptive[key] = fileio.parse_filter(path).adaptive
        except Exception as exc:
            problem(f"cannot inspect {key} adaptation: {exc}")

    if len(adaptive) == 2:
        h_adaptive, v_adaptive = adaptive["hfilter"], adaptive["vfilter"]
        if h_adaptive and v_adaptive:
            problem("both scaler axes are adaptive; Main can upload only one adaptive table")
        if family == "CRT Lottes":
            if h_adaptive or v_adaptive:
                problem("canonical Lottes uses fixed H/V tables on both axes")
        else:
            expected_axis_ok = (h_adaptive and not v_adaptive) if is_tate else (
                v_adaptive and not h_adaptive)
            if not expected_axis_ok:
                axis = "hfilter" if is_tate else "vfilter"
                problem(f"maximum-fidelity preset requires adaptation in {axis} only")

    if entries["gamma"] != spec["gamma"]:
        problem(f"gamma pairing must be {spec['gamma']!r}, got {entries['gamma']!r}")
    if entries["mask"] != spec["mask"]:
        problem(f"mask pairing must be {spec['mask']!r}, got {entries['mask']!r}")
    return problems


def _validate_relations(
        presets: dict[tuple[str, str], dict[str, str]]) -> list[str]:
    """Ensure every TATE file is exactly the rotated landscape pipeline."""
    problems: list[str] = []
    for family in FAMILY_SPECS:
        default = presets[(family, "Default")]
        tate = presets[(family, "TATE")]
        filename = _preset_filename(family, "TATE")
        if tate["hfilter"] != default["vfilter"] or \
                tate["vfilter"] != default["hfilter"]:
            problems.append(f"{filename}: TATE must transpose Default hfilter/vfilter")
        if tate["ifilter"] != tate["vfilter"]:
            problems.append(f"{filename}: TATE ifilter must equal its rotated vfilter")
        for key in ("sfilter", "gamma", "mask"):
            if tate[key] != default[key]:
                problems.append(f"{filename}: TATE must preserve {key}")
    return problems


def validate_collection(root: str) -> list[str]:
    """Validate the exact ten-file preset collection and its relationships."""
    preset_dir = os.path.join(root, "Presets")
    paths = sorted(glob.glob(os.path.join(preset_dir, "*.ini")))
    actual_names = {os.path.basename(path) for path in paths}
    problems = [f"Presets/: missing required preset {name!r}"
                for name in sorted(EXPECTED_PRESETS - actual_names)]
    problems += [f"Presets/: preset has no semantic contract {name!r}"
                 for name in sorted(actual_names - EXPECTED_PRESETS)]

    parsed: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        filename = os.path.basename(path)
        split = _split_name(filename)
        if split is None or filename not in EXPECTED_PRESETS:
            continue
        family, variant = split
        try:
            entries = fileio.parse_preset(path)
        except Exception as exc:
            problems.append(f"{filename}: EXCEPTION {exc}")
            continue
        parsed[(family, variant)] = entries
        problems += validate_entries(entries, root, filename, family, variant)

    if len(parsed) == len(EXPECTED_PRESETS):
        problems += _validate_relations(parsed)
    return problems
