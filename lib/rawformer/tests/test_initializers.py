"""Tests for weight initialization functions."""

import numpy as np

from rawformer.layers.initializers import he_init, xavier_init, zeros_init

_RNG = np.random.default_rng(0)


class TestXavierInit:
    def test_output_shape(self) -> None:
        w = xavier_init(10, 5, rng=_RNG)
        assert w.shape == (10, 5)

    def test_values_within_expected_range(self) -> None:
        rng = np.random.default_rng(0)
        w = xavier_init(100, 100, rng=rng)
        limit = np.sqrt(6.0 / 200)
        assert np.all(w >= -limit)
        assert np.all(w <= limit)

    def test_mean_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        w = xavier_init(1000, 1000, rng=rng)
        assert abs(float(np.mean(w))) < 0.01


class TestHeInit:
    def test_output_shape(self) -> None:
        w = he_init(10, 5, rng=_RNG)
        assert w.shape == (10, 5)

    def test_std_approximately_correct(self) -> None:
        rng = np.random.default_rng(0)
        w = he_init(1000, 1000, rng=rng)
        expected_std = np.sqrt(2.0 / 1000)
        assert abs(float(np.std(w)) - expected_std) < 0.01


class TestZerosInit:
    def test_output_shape(self) -> None:
        b = zeros_init(5)
        assert b.shape == (5,)

    def test_all_zeros(self) -> None:
        b = zeros_init(5)
        np.testing.assert_array_equal(b, np.zeros(5))
