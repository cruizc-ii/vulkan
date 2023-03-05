from unittest.case import TestCase
from app.utils import chunk_arrays
import numpy as np


class BuildingSpecificationTest(TestCase):
    """
    it should load and save spec
    """

    def test_even_array(self):
        a = [0.33, 0.2451, 0.21, 0.344]

        b = chunk_arrays(a, chunk_size=1)
        self.assertTrue(np.allclose(b, a))

        b = chunk_arrays(a, chunk_size=2)
        self.assertTrue(np.allclose(b, [0.33, 0.33, 0.21, 0.21]))

        b = chunk_arrays(a, chunk_size=3)
        self.assertTrue(np.allclose(b, [0.33, 0.33, 0.33, 0.344]))

        b = chunk_arrays(a, chunk_size=4)
        self.assertTrue(np.allclose(b, [0.33, 0.33, 0.33, 0.33]))

        b = chunk_arrays(a, chunk_size=5)
        self.assertTrue(np.allclose(b, [0.33, 0.33, 0.33, 0.33]))

        b = chunk_arrays(a, chunk_size=6)
        self.assertTrue(np.allclose(b, [0.33, 0.33, 0.33, 0.33]))

    def test_odd_array(self):
        a = [1.234, 0.3, 0.3, 1.345, 2.5, 0.222, 0.29]

        b = chunk_arrays(a, chunk_size=1)
        self.assertTrue(np.allclose(b, a))

        b = chunk_arrays(a, chunk_size=2)
        self.assertTrue(np.allclose(b, [1.234, 1.234, 0.3, 0.3, 2.5, 2.5, 0.29]))

        b = chunk_arrays(a, chunk_size=3)
        self.assertTrue(
            np.allclose(b, [1.234, 1.234, 1.234, 1.345, 1.345, 1.345, 0.29])
        )

        b = chunk_arrays(a, chunk_size=4)
        self.assertTrue(np.allclose(b, 4 * [1.234] + 3 * [2.5]))

        b = chunk_arrays(a, chunk_size=5)
        self.assertTrue(np.allclose(b, 5 * [1.234] + 2 * [0.222]))

        b = chunk_arrays(a, chunk_size=6)
        self.assertTrue(np.allclose(b, 6 * [1.234] + [0.29]))

        b = chunk_arrays(a, chunk_size=7)
        self.assertTrue(np.allclose(b, 7 * [1.234]))

        b = chunk_arrays(a, chunk_size=8)
        self.assertTrue(np.allclose(b, 7 * [1.234]))

        b = chunk_arrays(a, chunk_size=9)
        self.assertTrue(np.allclose(b, 7 * [1.234]))
