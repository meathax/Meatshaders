"""Self-test the ten canonical presets and landscape/TATE contracts."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import fileio  # noqa: E402
import preset_contracts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
problems = preset_contracts.validate_collection(ROOT)
expected = sum(len(variants) for variants in preset_contracts.EXPECTED_VARIANTS.values())

# Reproduce the interlace-axis defect this audit fixed. A TATE preset remains
# syntactically valid if ifilter points at the old source-V table, so only the
# cross-field contract can catch it.
tate_name = "CRT Guest Advanced - TATE.ini"
tate_path = os.path.join(ROOT, "Presets", tate_name)
tate = fileio.parse_preset(tate_path)
tate["ifilter"] = "CRT Guest Advanced (Port)_V No Scanlines.txt"
mutation_problems = preset_contracts.validate_entries(
    tate, ROOT, tate_name, "CRT Guest Advanced", "TATE")
if not any("TATE ifilter must equal vfilter" in p for p in mutation_problems):
    problems.append("negative contract test: wrong TATE interlace axis was accepted")

print(f"preset contracts checked: {expected}")
if problems:
    print(f"FAIL ({len(problems)}):")
    for problem in problems:
        print(" -", problem)
    sys.exit(1)
print("ALL OK")
