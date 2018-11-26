import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from raytrace import *
import unittest


class TestRender(unittest.TestCase):
    
    def assertAllEqual(self, a, b):
        if isinstance(a, V3):
            self.assertTrue(a.allEqual(b))
        else:
            self.assertTrue((a == b).all())
    
    def test_scene(self):
        resolution = (3, 3)
        origin = V3(0, 0, 0)
        direction = V3(1, 0, 0)
        dimensions = (5, 5)
        
        camera = CameraOrthogonal(origin, direction, dimensions, resolution)
        
        scene = {
            'objects': [
                Sphere(V3(4, 0, 0), 1, material = 'mat')
            ],
            'materials': {
                'mat': UniformMaterial(V3(1.0, 0.5, 0.25))
            },
            'lighting': [
                AmbientLight(0.5),
                DirectionalLight(V3(1, 0, 0), 0.5)
            ]
        }
        
        raster = render(camera, scene)
        
        self.assertAllEqual(
            raster,
            V3(
                1.00 * np.array([0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0]),
                0.50 * np.array([0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0]),
                0.25 * np.array([0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0])
            )
        )
    
    def test_reflection(self):
        resolution = (3, 3)
        origin = V3(0, 0, 0)
        direction = V3(1, 0, 0)
        dimensions = (5, 5)
        
        camera = CameraOrthogonal(origin, direction, dimensions, resolution)
        
        scene = {
            'objects': [
                Sphere(V3(4, 0, 0), 1, material = 'mirror'),
                Sphere(V3(-4, 0, 0), 1, material = 'mat')
            ],
            'materials': {
                'mat': UniformMaterial(V3(1.0, 0.5, 0.25)),
                'mirror': UniformMaterial(V3(0, 0, 0), reflectivity = 1.0)
            },
            'lighting': [
                AmbientLight(0.5),
                DirectionalLight(V3(1, 0, 0), 0.5)
            ]
        }
        
        raster = render(camera, scene)
        
        self.assertAllEqual(
            raster,
            V3(
                0.5 * 1.00 * np.array([0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0]),
                0.5 * 0.50 * np.array([0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0]),
                0.5 * 0.25 * np.array([0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0])
            )
        )


if __name__ == '__main__':
    unittest.main()
    