import numpy as np

from .vector import *


class TranslationHelper:
    
    def __init__(self, delta):
        self.delta = delta
    
    def inverse(self):
        return TranslationHelper(self.delta * -1.0)
    
    def apply(self, v):
        return v + self.delta
    
    def applyToDifference(self, v):
        return v
    
    def applyToNormal(self, v):
        return v


class ScalingHelper:
    
    def __init__(self, factor):
        if isinstance(factor, V3):
            self.factor = factor
        else:
            self.factor = V3(factor, factor, factor)
    
    def inverse(self):
        return ScalingHelper(
            V3(
                1.0 / self.factor.x,
                1.0 / self.factor.y,
                1.0 / self.factor.z
            )
        )
    
    def apply(self, v):
        return self.factor * v
    
    def applyToDifference(self, v):
        return self.apply(v)
    
    def applyToNormal(self, v):
        return self.apply(v).unit()