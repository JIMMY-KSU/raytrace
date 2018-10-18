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


class RotationHelper:
    
    def __init__(self, axisIndex, angle):
        self.axisIndex = axisIndex
        self.angle = angle
    
    def inverse(self):
        return RotationHelper(self.axisIndex, self.angle * -1.0)
    
    def apply(self, v):
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        if self.axisIndex == 0:
            return V3(
                v.x,
                c * v.y - s * v.z,
                s * v.y + c * v.z
            )
        elif self.axisIndex == 1:
            return V3(
                c * v.x + s * v.z,
                v.y,
                c * v.z - s * v.x
            )
        elif self.axisIndex == 2:
            return V3(
                c * v.x - s * v.y,
                s * v.x + c * v.y,
                v.z
            )
        else:
            raise ValueError('Error: rotation axis index must be 0, 1, or 2')
    
    def applyToDifference(self, v):
        return self.apply(v)
    
    def applyToNormal(self, v):
        return self.apply(v)
