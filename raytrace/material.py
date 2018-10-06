class UniformMaterial:
    
    def __init__(self, color, reflectivity = 0.0):
        self.color = color
        self.reflectivity = reflectivity
    
    def getColor(self, u = 0, v = 0):
        return self.color
    
    def getReflectivity(self, u = 0, v = 0):
        return self.reflectivity


class CheckeredMaterial:
    
    def __init__(self, mat1, mat2):
        self.mat1 = mat1
        self.mat2 = mat2
    
    def getColor(self, u = 0, v = 0):
        use_first = np.floor(u + v) % 2 == 0
        return self.mat1.getColor(u, v).where(use_first, self.mat2.getColor(u, v))
    
    def getReflectivity(self, u = 0, v = 0):
        use_first = np.floor(u + v) % 2 == 0
        return self.mat1.getReflectivity(u, v).where(use_first, self.mat2.getReflectivity(u, v))