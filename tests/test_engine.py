import unittest

import numpy as np

from paa_rag.engine import exact_search


class ExactSearchTests(unittest.TestCase):
    def test_returns_nearest_vectors_in_descending_similarity(self):
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)

        indices = exact_search(vectors, query, 2)

        self.assertEqual(indices.tolist(), [0, 2])

    def test_rejects_invalid_k(self):
        vectors = np.eye(2, dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            exact_search(vectors, query, 0)


if __name__ == "__main__":
    unittest.main()
