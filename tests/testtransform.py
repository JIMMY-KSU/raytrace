import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from raytrace import *
import unittest


class TestTranslationHelper(unittest.TestCase):
    
    def setUp(self):
        self.transformation = TranslationHelper(V3(-1, 1, 2.5))
        self.inverse = self.transformation.inverse()
    
    def test_apply(self):
        cases = [
            (V3(0, 0, 0), V3(-1, 1, 2.5)),
            (V3(10, 11, 12), V3(9, 12, 14.5))
        ]
        for before, after in cases:
            self.assertTrue(self.transformation.apply(before) == after)
            self.assertTrue(self.inverse.apply(after) == before)
    
    def test_applyToDifference(self):
        cases = [
            (V3(0, 0, 0), V3(0, 0, 0)),
            (V3(10, 11, 12), V3(10, 11, 12))
        ]
        for before, after in cases:
            self.assertTrue(self.transformation.applyToDifference(before) == after)
            self.assertTrue(self.inverse.applyToDifference(after) == before)
    
    def test_applyToNormal(self):
        cases = [
            (V3(1, 0, 0), V3(1, 0, 0)),
            (V3(3, 4, 5).unit(), V3(3, 4, 5).unit())
        ]
        for before, after in cases:
            self.assertTrue(self.transformation.applyToNormal(before) == after)
            self.assertTrue(self.inverse.applyToNormal(after) == before)
