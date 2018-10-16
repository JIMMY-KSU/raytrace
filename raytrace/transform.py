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
