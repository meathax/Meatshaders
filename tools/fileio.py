"""Readers, writers and validators for MiSTer video-pipeline files.

Formats (current MiSTer Main, 256-phase 10-bit era):
  Filter      : '#' comments, optional 'adaptive' / '10bit' keyword lines,
                then rows of 4 comma-separated signed ints. 256 rows per set;
                adaptive files carry two sets (dark endpoint first).
  Gamma       : 256 lines, each '<v>' or '<r>,<g>,<b>', values 0..255.
  Shadow mask : '#' comments, 'v2', '<W>,<H>', then H rows of W hex tokens.
                Token XYZ (hex): X = RGB channel bitmask (4=R, 2=G, 1=B),
                selected channels multiply by (16+Y)/16, others by Z/16.
  Preset .ini : keys hfilter, vfilter, sfilter, ifilter, gamma, mask, maskmode.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

TEN_BIT_MIN, TEN_BIT_MAX = -512, 511
PHASES = 256


# ---------------------------------------------------------------- filters

@dataclass
class FilterFile:
    header: list[str]              # comment lines without leading '#'
    adaptive: bool
    tenbit: bool
    sets: list[np.ndarray]         # each (256, 4) int arrays; adaptive: [dark, bright]

    @property
    def dark(self) -> np.ndarray:
        return self.sets[0]

    @property
    def bright(self) -> np.ndarray:
        return self.sets[-1]


def parse_filter(path: str) -> FilterFile:
    header, rows = [], []
    adaptive = tenbit = False
    saw_rows = False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                header.append(line.lstrip("#").strip())
                continue
            low = line.lower()
            if low == "adaptive":
                if saw_rows:
                    raise ValueError(f"{path}: adaptive marker after coefficient rows")
                if adaptive:
                    raise ValueError(f"{path}: duplicate adaptive marker")
                adaptive = True
                continue
            if low == "10bit":
                if saw_rows:
                    raise ValueError(f"{path}: 10bit marker after coefficient rows")
                if tenbit:
                    raise ValueError(f"{path}: duplicate 10bit marker")
                tenbit = True
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"{path}: bad row {line!r}")
            rows.append([int(p) for p in parts])
            saw_rows = True
    n = len(rows)
    if n == PHASES:
        sets = [np.array(rows, dtype=np.int32)]
    elif n == 2 * PHASES:
        sets = [np.array(rows[:PHASES], dtype=np.int32),
                np.array(rows[PHASES:], dtype=np.int32)]
    else:
        raise ValueError(f"{path}: {n} coefficient rows (expected 256 or 512)")
    if adaptive and len(sets) != 2:
        raise ValueError(f"{path}: adaptive keyword but {len(sets)} set(s)")
    if not adaptive and len(sets) != 1:
        raise ValueError(f"{path}: two sets but no adaptive keyword")
    return FilterFile(header, adaptive, tenbit, sets)


def write_filter(path: str, flt: FilterFile, set_comments: list[str] | None = None) -> None:
    lines = [f"# {h}" if h else "#" for h in flt.header]
    lines.append("")
    if flt.adaptive:
        lines.append("adaptive")
    if flt.tenbit:
        lines.append("10bit")
    lines.append("")
    for i, coeffs in enumerate(flt.sets):
        if set_comments and i < len(set_comments) and set_comments[i]:
            lines.append(f"# {set_comments[i]}")
        for row in coeffs:
            lines.append(",".join(f"{int(v):4d}" for v in row))
        lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")


def validate_filter(flt: FilterFile, path: str = "<filter>") -> list[str]:
    """Return a list of problems (empty list = valid)."""
    problems = []
    lo, hi = (TEN_BIT_MIN, TEN_BIT_MAX) if flt.tenbit else (-256, 255)
    for si, coeffs in enumerate(flt.sets):
        tag = f"{path} set{si}"
        if coeffs.shape != (PHASES, 4):
            problems.append(f"{tag}: shape {coeffs.shape}")
            continue
        if coeffs.min() < lo or coeffs.max() > hi:
            problems.append(f"{tag}: coefficient out of range [{lo},{hi}] "
                            f"(min {coeffs.min()}, max {coeffs.max()})")
        # Exact stored-phase symmetry: row[256-i] is the tap reversal of row[i].
        for i in range(1, 128):
            if not np.array_equal(coeffs[PHASES - i], coeffs[i][::-1]):
                problems.append(f"{tag}: phase {i}/{PHASES - i} not conjugate")
                break
        if not np.array_equal(coeffs[128], coeffs[128][::-1]):
            problems.append(f"{tag}: phase 128 not self-symmetric")
    return problems


# ---------------------------------------------------------------- gamma

def parse_gamma(path: str) -> np.ndarray:
    """Return (256, 3) int array (single-column files broadcast to 3 channels)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [int(p.strip()) for p in line.split(",")]
            if len(parts) == 1:
                parts = parts * 3
            if len(parts) != 3:
                raise ValueError(f"{path}: bad gamma row {line!r}")
            rows.append(parts)
    if len(rows) != 256:
        raise ValueError(f"{path}: {len(rows)} gamma rows (expected 256)")
    return np.array(rows, dtype=np.int32)


def write_gamma(path: str, curve: np.ndarray, header: list[str] | None = None) -> None:
    curve = np.asarray(curve)
    if curve.ndim == 1:
        curve = np.stack([curve] * 3, axis=1)
    lines = [f"# {h}" for h in (header or [])]
    mono = curve[:, 0]
    if np.array_equal(curve[:, 0], curve[:, 1]) and np.array_equal(curve[:, 0], curve[:, 2]):
        lines += [str(int(v)) for v in mono]
    else:
        lines += [f"{int(r)},{int(g)},{int(b)}" for r, g, b in curve]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def validate_gamma(curve: np.ndarray, path: str = "<gamma>") -> list[str]:
    problems = []
    if curve.shape != (256, 3):
        return [f"{path}: shape {curve.shape}"]
    if curve.min() < 0 or curve.max() > 255:
        problems.append(f"{path}: value out of range 0..255")
    for c, name in enumerate("RGB"):
        if np.any(np.diff(curve[:, c]) < 0):
            problems.append(f"{path}: channel {name} not monotonic")
    return problems


# ---------------------------------------------------------------- shadow masks

@dataclass
class MaskFile:
    header: list[str]
    width: int
    height: int
    tokens: list[list[str]]        # hex token strings, height rows x width cols

    def multipliers(self) -> np.ndarray:
        """(height, width, 3) float array of per-channel multipliers."""
        out = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for y, row in enumerate(self.tokens):
            for x, tok in enumerate(row):
                sel, mult_sel, mult_oth = decode_mask_token(tok)
                for c, bit in enumerate((4, 2, 1)):
                    out[y, x, c] = mult_sel / 16.0 if sel & bit else mult_oth / 16.0
        return out


def decode_mask_token(tok: str) -> tuple[int, int, int]:
    """Return (channel_bitmask, selected_multiplier_16ths, other_multiplier_16ths)."""
    v = int(tok, 16)
    x, y, z = (v >> 8) & 0xF, (v >> 4) & 0xF, v & 0xF
    return x, 16 + y, z


def encode_mask_token(channel_bitmask: int, selected_16ths: int, other_16ths: int) -> str:
    if not 16 <= selected_16ths <= 31:
        raise ValueError(f"selected multiplier {selected_16ths}/16 outside 16..31")
    if not 0 <= other_16ths <= 15:
        raise ValueError(f"other multiplier {other_16ths}/16 outside 0..15")
    return f"{channel_bitmask:x}{selected_16ths - 16:x}{other_16ths:x}"


def parse_mask(path: str) -> MaskFile:
    header, tokens = [], []
    dims = None
    saw_v2 = False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                header.append(line.lstrip("#").strip())
                continue
            if line.lower() == "v2":
                saw_v2 = True
                continue
            if dims is None:
                w, h = (int(p) for p in line.split(","))
                dims = (w, h)
                continue
            tokens.append([t.strip() for t in line.split(",")])
    if not saw_v2:
        raise ValueError(f"{path}: missing v2 marker")
    if dims is None:
        raise ValueError(f"{path}: missing dimensions")
    w, h = dims
    if len(tokens) != h or any(len(r) != w for r in tokens):
        raise ValueError(f"{path}: token grid does not match {w}x{h}")
    return MaskFile(header, w, h, tokens)


def write_mask(path: str, mask: MaskFile) -> None:
    lines = ["####"] + [f"# {h}" if h else "#" for h in mask.header] + ["####", "v2",
             f"{mask.width},{mask.height}"]
    lines += [",".join(row) for row in mask.tokens]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def validate_mask(mask: MaskFile, path: str = "<mask>") -> list[str]:
    problems = []
    if not 1 <= mask.width <= 16 or not 1 <= mask.height <= 16:
        problems.append(f"{path}: dimensions {mask.width}x{mask.height} "
                        "outside hardware range 1..16")
    if len(mask.tokens) != mask.height or any(len(row) != mask.width for row in mask.tokens):
        problems.append(f"{path}: token grid does not match {mask.width}x{mask.height}")
    for row in mask.tokens:
        for tok in row:
            if not re.fullmatch(r"[0-9a-fA-F]{1,3}", tok):
                problems.append(f"{path}: bad token {tok!r}")
                continue
            sel, _, _ = decode_mask_token(tok)
            if sel > 7:
                problems.append(f"{path}: token {tok!r} channel bits > 7")
    return problems


# ---------------------------------------------------------------- presets

PRESET_KEYS = ("hfilter", "vfilter", "sfilter", "ifilter", "gamma", "mask", "maskmode")
_PRESET_DIRS = {"hfilter": "Filters", "vfilter": "Filters", "sfilter": "Filters",
                "ifilter": "Filters", "gamma": "Gamma", "mask": "Shadow_Masks"}
_PRESET_DISABLED = {
    "hfilter": {"same", "off"},
    "vfilter": {"same", "off"},
    "sfilter": {"same", "off"},
    "ifilter": {"same", "off"},
    "gamma": {"off", "none"},
    "mask": {"off", "none"},
}
MASK_MODES = ("off", "none", "1x", "2x", "1x rotated", "2x rotated")


def parse_preset(path: str) -> dict[str, str]:
    entries = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(("#", ";", "[")):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                raise ValueError(f"{path}: malformed preset line {line!r}")
            key = key.strip().lower()
            if key in entries:
                raise ValueError(f"{path}: duplicate preset key {key!r}")
            entries[key] = value.strip()
    return entries


def write_preset(path: str, entries: dict[str, str]) -> None:
    lines = [f"{k}={entries[k]}" for k in PRESET_KEYS if k in entries]
    extra = [k for k in entries if k not in PRESET_KEYS]
    if extra:
        raise ValueError(f"unsupported preset keys: {extra}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def validate_preset(entries: dict[str, str], root: str, path: str = "<preset>") -> list[str]:
    """Validate a complete Main-compatible preset and resolve files exactly."""
    problems = [f"{path}: unknown key {k!r}" for k in entries if k not in PRESET_KEYS]
    problems += [f"{path}: missing required key {k!r}" for k in PRESET_KEYS if k not in entries]
    for key, sub in _PRESET_DIRS.items():
        value = entries.get(key)
        if value is None:
            continue
        if not value:
            problems.append(f"{path}: {key} is empty")
            continue
        low = value.lower()
        if low in _PRESET_DISABLED[key]:
            continue
        # These words are only special for some fields. For example, Main
        # treats gamma=same and ifilter=none as literal filenames, not no-ops.
        if low in {"same", "off", "none"}:
            allowed = ", ".join(sorted(_PRESET_DISABLED[key]))
            problems.append(f"{path}: {key}={value!r} is not a valid sentinel "
                            f"(allowed: {allowed})")
            continue
        folder = os.path.join(root, sub)
        listing = os.listdir(folder) if os.path.isdir(folder) else []
        if value not in listing:
            problems.append(f"{path}: {key}={value!r} not found (case-sensitive) in {sub}/")
    mode = entries.get("maskmode")
    if mode is not None:
        if not mode:
            problems.append(f"{path}: maskmode is empty")
        elif mode.lower() not in MASK_MODES:
            problems.append(f"{path}: unsupported maskmode {mode!r}")
    return problems
