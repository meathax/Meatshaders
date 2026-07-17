"""Focused bit-exact regression tests for the MiSTer scaler arithmetic model."""

from __future__ import annotations

import numpy as np

import fitting
import mister_model as mm


def test_phase_selection_uses_top_8_bits_of_12_bit_fraction() -> None:
    # Exercise both sides of phase boundaries and the last representable
    # fraction.  A nearest-phase implementation fails several of these cases.
    frac12 = np.array([0, 1, 15, 16, 17, 2047, 2048, 4094, 4095])
    positions = 8.0 + frac12 / 4096.0
    taps, phases = mm._positions_to_taps(positions, 32)
    np.testing.assert_array_equal(phases, frac12 >> 4)
    np.testing.assert_array_equal(taps[:, 0], np.full(len(frac12), 7))

    # The fraction immediately below the next integer remains phase 255 and
    # keeps the same tap window; it must never round/carry to phase zero.
    taps, phases = mm._positions_to_taps(np.array([8.0 + 4095.0 / 4096.0]), 32)
    assert int(phases[0]) == 255
    np.testing.assert_array_equal(taps[0], [7, 8, 9, 10])

    # Crossing the integer boundary advances the base and returns to phase 0.
    taps, phases = mm._positions_to_taps(np.array([9.0]), 32)
    assert int(phases[0]) == 0
    np.testing.assert_array_equal(taps[0], [8, 9, 10, 11])


def _rtl_poly_lerp_scalar(a: int, b: int, lum: int) -> int:
    """Integer equivalent of ascal.vhd poly_lerp's t(18 downto 1)."""
    return (a * (256 - lum) + b * lum) >> 1


def test_adaptive_blend_preserves_odd_3_15_values() -> None:
    dark = np.zeros((256, 4), dtype=np.int64)
    bright = np.zeros((256, 4), dtype=np.int64)
    dark[3] = [1, -3, 511, -512]
    bright[3] = [2, 4, -5, 7]
    phase = np.array([3, 3, 3, 3], dtype=np.int64)
    ctrl = np.array([0, 1, 127, 255], dtype=np.int64)

    got = mm.adaptive_c128(dark, bright, phase, ctrl)
    expected = np.array([
        [_rtl_poly_lerp_scalar(int(a), int(b), int(lum))
         for a, b in zip(dark[3], bright[3])]
        for lum in ctrl
    ], dtype=np.int64)
    np.testing.assert_array_equal(got, expected)

    # These cases specifically distinguish one-bit RTL truncation from the
    # previous implementation, which forced the complete matrix even.
    assert np.any(got & 1)
    assert got[2, 0] == 191
    assert got[1, 1] == -381


def test_fitting_grid_includes_black_and_peak_white() -> None:
    assert np.array_equal(fitting.EVAL_CODES[:8], np.arange(8))
    assert np.array_equal(fitting.EVAL_CODES[-4:], np.arange(252, 256))


def test_clipping_gate_uses_max_rgb_control_and_each_signal_channel() -> None:
    dark = np.zeros((256, 4), dtype=np.int64)
    bright = np.zeros((256, 4), dtype=np.int64)
    dark[:, 1] = 80
    bright[:, 1] = 300
    lut = np.zeros((256, 3), dtype=np.int64)
    lut[:, 0] = 255                 # red is signal and max-RGB control
    lut[:, 1:] = 1

    got = fitting.worst_flat_field_output(lut, dark, bright)
    phases = np.arange(256)
    blend = mm.adaptive_c128(
        dark, bright, phases, np.full(256, 255, dtype=np.int64))
    expected = float((255 * blend.sum(axis=1) / 32768.0).max())
    assert np.isclose(got, expected)
    assert got > 290.0              # the former green-only check reported < 1


if __name__ == "__main__":
    test_phase_selection_uses_top_8_bits_of_12_bit_fraction()
    test_adaptive_blend_preserves_odd_3_15_values()
    test_fitting_grid_includes_black_and_peak_white()
    test_clipping_gate_uses_max_rgb_control_and_each_signal_channel()
    print("ALL OK")
