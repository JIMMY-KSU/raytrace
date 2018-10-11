import numpy as np


class UniformMaterial:
    
    def __init__(self, color, reflectivity = 0.0):
        self.color = color
        self.reflectivity = reflectivity
    
    def getColor(self, u = 0, v = 0):
        if isinstance(u, np.ndarray):
            return self.color.repeat(len(u))
        else:
            return self.color
    
    def getReflectivity(self, u = 0, v = 0):
        if isinstance(u, np.ndarray):
            return np.repeat([self.reflectivity], len(u))
        else:
            return self.reflectivity


class CheckeredMaterial:
    
    def __init__(self, mat1, mat2, scale = 1.0):
        self.mat1 = mat1
        self.mat2 = mat2
        self.scale = scale
    
    def getColor(self, u = 0, v = 0):
        us, vs = (u, v) / np.array(self.scale)
        uflag = np.floor(us) % 2 == 0
        vflag = np.floor(vs) % 2 == 0
        use_first = np.logical_xor(uflag, vflag)
        return self.mat1.getColor(u, v).where(use_first, self.mat2.getColor(u, v))
    
    def getReflectivity(self, u = 0, v = 0):
        us, vs = (u, v) / np.array(self.scale)
        uflag = np.floor(us) % 2 == 0
        vflag = np.floor(vs) % 2 == 0
        use_first = np.logical_xor(uflag, vflag)
        return np.where(use_first, self.mat1.getReflectivity(u, v), self.mat2.getReflectivity(u, v))