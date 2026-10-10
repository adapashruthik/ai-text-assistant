
import unittest

from similarity import (
    dot_product,
    magnitude,
    cosine_similarity,
)


class TestSimilarity(unittest.TestCase):

    def test_dot_product(self):
        self.assertAlmostEqual(
            dot_product([1, 2], [3, 4]),
            11
        )

    def test_magnitude(self):
        self.assertAlmostEqual(
            magnitude([3, 4]),
            5
        )

    def test_cosine_similarity(self):
        self.assertAlmostEqual(
            cosine_similarity([1, 0], [0.8, 0.6]),
            0.8
        )

    def test_zero_vector(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 2])

    def test_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
