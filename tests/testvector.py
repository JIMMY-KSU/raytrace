import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from raytrace import *
import unittest


class TestV3(unittest.TestCase):
    
    #TODO: use in all places
    def assertAllEqual(self, a, b):
        if isinstance(a, V3):
            self.assertTrue(a.allEqual(b))
        else:
            self.assertTrue((a == b).all())
    
    def test_init(self):
        self.assertRaises(ValueError, V3, [1], [1], [1, 2])
        self.assertRaises(ValueError, V3,
            np.array([1]),
            np.array([1]),
            np.array([[1]])
        )
    
    def test_unvectorized(self):
        a = V3(1, 2, 3)
        b = V3(-1, 1, -1)
        c = V3(1, 4, 8)
        self.assertTrue(a == a)
        self.assertFalse(a == b)
        self.assertEqual(a + b, V3(0, 3, 2))
        self.assertEqual(a - b, V3(2, 1, 4))
        self.assertEqual(a * b, V3(-1, 2, -3))
        self.assertEqual(a.scale(4), V3(4, 8, 12))
        self.assertEqual(a.dot(b), -2)
        self.assertEqual(c.normsq(), 81)
        self.assertEqual(c.norm(), 9)
        self.assertEqual(c.unit(), V3(1/9, 4/9, 8/9))
        self.assertEqual(a.cross(b), V3(-5, -2, 3))
        self.assertTrue(
            a.mapToXYZ(lambda axis: np.repeat(axis, 3)).allEqual(
                V3(
                    np.array([1, 1, 1]),
                    np.array([2, 2, 2]),
                    np.array([3, 3, 3])
                )
            )
        )
        self.assertAllEqual(
            a.repeat(2),
            V3(
                np.array([1, 1]),
                np.array([2, 2]),
                np.array([3, 3])
            )
        )
    
    def test_vectorized(self):
        a = V3(
            np.array([0, 1]),
            np.array([1, 2]),
            np.array([2, 3])
        )
        b = V3(
            np.array([1, -1]),
            np.array([-2, 1]),
            np.array([1, -1])
        )
        c = V3(
            np.array([1, 2]),
            np.array([4, 3]),
            np.array([8, 6])
        )
        self.assertTrue(a.allEqual(a))
        self.assertFalse(a.allEqual(b))
        self.assertAllEqual(
            a + b,
            V3(
                np.array([1, 0]),
                np.array([-1, 3]),
                np.array([3, 2])
            )
        )
        self.assertAllEqual(
            a - b,
            V3(
                np.array([-1, 2]),
                np.array([3, 1]),
                np.array([1, 4])
            )
        )
        self.assertAllEqual(
            a * b,
            V3(
                np.array([0, -1]),
                np.array([-2, 2]),
                np.array([2, -3])
            )
        )
        self.assertAllEqual(
            a.scale(2),
            V3(
                np.array([0, 2]),
                np.array([2, 4]),
                np.array([4, 6])
            )
        )
        self.assertAllEqual(
            a.dot(b),
            np.array([0, -2])
        )
        self.assertAllEqual(
            c.normsq(),
            np.array([81, 49])
        )
        self.assertAllEqual(
            c.norm(),
            np.array([9, 7])
        )
        self.assertAllEqual(
            c.unit(),
            V3(
                np.array([1/9, 2/7]),
                np.array([4/9, 3/7]),
                np.array([8/9, 6/7])
            )
        )
        self.assertAllEqual(
            a.cross(b),
            V3(
                np.array([5, -5]),
                np.array([2, -2]),
                np.array([-1, 3])
            )
        )
        self.assertAllEqual(
            a.mapToXYZ(lambda axis: np.repeat(axis, 3)),
            V3(
                np.array([0, 0, 0, 1, 1, 1]),
                np.array([1, 1, 1, 2, 2, 2]),
                np.array([2, 2, 2, 3, 3, 3])
            )
        )
        self.assertAllEqual(
            a.repeat(2),
            V3(
                np.array([0, 0, 1, 1]),
                np.array([1, 1, 2, 2]),
                np.array([2, 2, 3, 3])
            )
        )

if __name__ == '__main__':
    unittest.main()