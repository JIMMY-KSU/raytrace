import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from raytrace import *
import unittest


class TestGeometry(unittest.TestCase):
    
    def setUp(self):
        self.geoms = [
            Sphere(V3(0, 0, 0), 1),
            Difference(
                Sphere(V3(0, 0, 0), 1),
                Sphere(V3(1, 0, 0), 1)
            ),
            Intersection(
                Sphere(V3(0, 0, 0), 1),
                Sphere(V3(1, 0, 0), 1)
            )
        ]
    
    def test_intersection(self):
        for geom in self.geoms:
            collisions = geom.intersections(Ray(V3(0, 0, 0), V3(1, 0, 0)))
            self.assertEqual(type(collisions.dist), np.ndarray)
            self.assertEqual(type(collisions.norm), V3)
            
            collisions = geom.intersections(Ray(V3(0, 0, 0), V3(1, 0, 0)), invert = True)
            self.assertEqual(type(collisions.dist), np.ndarray)
            self.assertEqual(type(collisions.norm), V3)
    
    def test_interior(self):
        for geom in self.geoms:
            result = geom.interior(V3(0, 0, 0))
            self.assertEqual(type(result), np.ndarray)
    
    def test_ground(self):
        ground = Ground(V3(10, 0, 0), V3(1, 1, 0))
        self.assertEqual(
            np.array([np.inf]),
            ground.intersections(Ray(
                V3(-1000, -100, -10),
                V3(1, 0, 0)
            )).dist
        )
        self.assertNotEqual(
            np.array([np.inf]),
            ground.intersections(Ray(
                V3(1000, 100, 10),
                V3(-1, 0, 0)
            )).dist
        )
        self.assertFalse(ground.interior(V3(11, 0, 0)).any())
        self.assertTrue(ground.interior(V3(0, -100, -10)).all())
    
    def test_sphere(self):
        sphere = Sphere(V3(100, 100, -100), 10)
        self.assertEqual(
            np.array([np.inf]),
            sphere.intersections(Ray(
                V3(0, 0, 0),
                V3(1, -1, -1).unit()
            )).dist
        )
        self.assertNotEqual(
            np.array([np.inf]),
            sphere.intersections(Ray(
                V3(0, 0, 0),
                V3(1, 1, -1).unit()
            )).dist
        )
        self.assertFalse(sphere.interior(V3(0, 0, 0)).any())
        self.assertFalse(sphere.interior(V3(110, 110, -100)).any())
        self.assertTrue(sphere.interior(V3(103, 103, -103)).all())

if __name__ == '__main__':
    unittest.main()