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


def test_flat_profile_separates_signal_channel_from_max_rgb_control() -> None:
    dark = np.zeros((256, 4), dtype=np.int64)
    bright = np.zeros((256, 4), dtype=np.int64)
    dark[:, 1] = 64
    bright[:, 1] = 256
    lut = np.zeros((256, 3), dtype=np.int64)
    lut[10] = [200, 40, 80]

    pos = (np.arange(9) + 0.5) / 4.5 - 0.5 + 8.0
    ctrl = np.full(9, 200, dtype=np.int64)
    for channel, signal in enumerate((200, 40, 80)):
        expected = mm.fir_1d_adaptive(
            np.full(64, signal, dtype=np.int64), dark, bright, pos, ctrl)
        got = mm.flat_field_profile(
            10, 4.5, dark, bright, lut, n_out=9, channel=channel)
        np.testing.assert_array_equal(got, expected)

    # Red is the shared control even while green is the displayed signal.
    wrong_green_control = mm.fir_1d_adaptive(
        np.full(64, 40, dtype=np.int64), dark, bright, pos,
        np.full(9, 40, dtype=np.int64))
    assert not np.array_equal(
        mm.flat_field_profile(10, 4.5, dark, bright, lut,
                              n_out=9, channel=1),
        wrong_green_control)


def test_moire_metrics_measure_exact_fractional_frame_extrema() -> None:
    # A 4.5x frame alternates between troughs sampled exactly at the midpoint
    # and troughs sampled 1/9 line away.  Overlapping per-period slices used to
    # reuse the darker shared trough and incorrectly report zero variation.
    scale = 4.5
    n_out = 1080
    pos = (np.arange(n_out) + 0.5) / scale - 0.5
    frac = pos - np.floor(pos)
    distance = np.minimum(frac, 1.0 - frac)
    profile = np.round(255.0 * (1.0 - distance)).astype(np.int64)
    metrics = mm.moire_metrics(profile, scale)

    pos_first = 0.5 / scale - 0.5
    pos_last = (n_out - 0.5) / scale - 0.5
    centers = np.arange(int(np.ceil(pos_first + 0.5)),
                        int(np.floor(pos_last - 0.5)) + 1,
                        dtype=np.float64)

    def sampled(values: np.ndarray, keep_max: bool) -> np.ndarray:
        jf = (values + 0.5) * scale - 0.5
        lo = np.floor(jf).astype(np.int64)
        hi = np.ceil(jf).astype(np.int64)
        fn = np.maximum if keep_max else np.minimum
        return fn(profile[lo], profile[hi]).astype(np.float64)

    expected_peaks = sampled(centers, True)
    expected_troughs = sampled(centers[:-1] + 0.5, False)
    assert np.isclose(metrics["peak_std"], expected_peaks.std())
    assert np.isclose(metrics["trough_std"], expected_troughs.std())
    assert metrics["trough_std"] > 10.0


def test_moire_metrics_reject_invalid_inputs() -> None:
    for profile, scale in ((np.zeros((2, 2)), 4.5),
                           (np.zeros(10), 0.0)):
        try:
            mm.moire_metrics(profile, scale)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid moire input was accepted")


if __name__ == "__main__":
    test_phase_selection_uses_top_8_bits_of_12_bit_fraction()
    test_adaptive_blend_preserves_odd_3_15_values()
    test_fitting_grid_includes_black_and_peak_white()
    test_clipping_gate_uses_max_rgb_control_and_each_signal_channel()
    test_flat_profile_separates_signal_channel_from_max_rgb_control()
    test_moire_metrics_measure_exact_fractional_frame_extrema()
    test_moire_metrics_reject_invalid_inputs()
    print("ALL OK")
