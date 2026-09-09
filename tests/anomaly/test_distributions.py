"""Tests for stable Phase 5 distribution comparisons."""

import unittest

from backend.anomaly.distributions import compare_distributions, jensen_shannon_divergence, normalize_distribution


class TestDistributions(unittest.TestCase):
    def test_identical_distributions_have_zero_divergence(self) -> None:
        self.assertAlmostEqual(jensen_shannon_divergence({"A": 2.0, "B": 1.0}, {"A": 4.0, "B": 2.0}), 0.0)

    def test_changed_and_missing_keys_are_symmetric(self) -> None:
        first = {("J01", "J02"): 1.0}
        second = {("J01", "J03"): 1.0}
        self.assertAlmostEqual(jensen_shannon_divergence(first, second), 1.0)
        self.assertAlmostEqual(jensen_shannon_divergence(first, second), jensen_shannon_divergence(second, first))

    def test_empty_and_numerically_stable_inputs(self) -> None:
        self.assertEqual(normalize_distribution({}), {})
        self.assertEqual(jensen_shannon_divergence({}, {}), 0.0)
        self.assertEqual(jensen_shannon_divergence({}, {"A": 1.0}), 0.5)
        result = compare_distributions({"A": 1e-300}, {"A": 2e-300}, 0.01)
        self.assertFalse(result.threshold_exceeded)

    def test_invalid_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_distribution({"A": -1.0})
        with self.assertRaises(ValueError):
            compare_distributions({}, {}, -0.1)


if __name__ == "__main__":
    unittest.main()
