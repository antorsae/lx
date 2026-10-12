import unittest

import numpy as np

import config
from generate_visualizations import log_bin_frequency


class LogBinFrequencyTest(unittest.TestCase):
    def test_rew_grid_becomes_fractional_octave_bins(self):
        freq = np.arange(300.0, 20000.0, 0.366)
        octaves = np.log2(freq[-1] / freq[0])
        binned_freq, level, log_freq = log_bin_frequency(freq, np.full_like(freq, 80.0), np.log2(freq))

        self.assertLessEqual(len(binned_freq), np.ceil(octaves * config.INTERACTIVE_BINS_PER_OCTAVE) + 1)
        self.assertGreater(len(binned_freq), 0.9 * octaves * config.INTERACTIVE_BINS_PER_OCTAVE)
        np.testing.assert_allclose(level, 80.0)
        # Each bin's frequency is the geometric mean of its points
        np.testing.assert_allclose(log_freq, np.log2(binned_freq))

    def test_sparse_points_keep_their_own_bins(self):
        freq = np.array([20.0, 21.0, 22.0])
        binned_freq, values = log_bin_frequency(freq, np.array([1.0, 2.0, 3.0]))

        np.testing.assert_allclose(binned_freq, freq)
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])

    def test_matrices_bin_along_frequency(self):
        freq = np.arange(1000.0, 1100.0, 0.366)
        matrix = np.stack([np.zeros_like(freq), np.ones_like(freq)], axis=1)
        binned_freq, binned = log_bin_frequency(freq, matrix)

        self.assertEqual(binned.shape, (len(binned_freq), 2))
        np.testing.assert_allclose(binned[:, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
